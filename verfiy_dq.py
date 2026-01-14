import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from rm_ros_interfaces.msg import Jointpos


# ================= 配置区域 =================
MAX_SAMPLES = 400  # 采集多少帧 (50Hz * 8s = 400)
# ===========================================

class DqAllPlotter(Node):
    def __init__(self):
        super().__init__('dq_all_plotter_node')

        # 1. 订阅位置 (用于计算差分真值)
        self.sub_pos = self.create_subscription(
            Jointpos,
            '/right_arm/rm_driver/movej_canfd_cmd',
            self.pos_callback,
            10
        )

        # 2. 订阅算法算出的 dq
        self.sub_dq = self.create_subscription(
            Jointpos,
            '/right_arm/dq_target', 
            self.dq_callback,
            10
        )

        # 缓存与状态
        self.latest_dq_msg = None
        self.last_pos = None
        self.last_pos_time = None
        self.start_time = None

        # 数据存储
        self.times = []
        self.data_v_num = []  # 差分真值
        self.data_v_alg = []  # 算法计算值
        
        self.collecting = True
        self.get_logger().info(f"开始采集 6轴 数据... 请大幅度移动机械臂 (目标: {MAX_SAMPLES}帧)")

    def dq_callback(self, msg):
        if not self.collecting: return
        self.latest_dq_msg = np.array(msg.joint)

    def pos_callback(self, msg):
        if not self.collecting: return

        current_pos = np.array(msg.joint)
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.last_pos is None:
            self.last_pos = current_pos
            self.last_pos_time = current_time
            self.start_time = current_time
            return
        
        # 1. 计算差分速度 (真值)
        dt = current_time - self.last_pos_time
        if dt < 0.002: return 
        
        v_numerical = (current_pos - self.last_pos) / dt

        # 2. 获取算法速度
        if self.latest_dq_msg is None: return
        
        # 3. 存入缓存
        rel_time = current_time - self.start_time
        self.times.append(rel_time)
        self.data_v_num.append(v_numerical)
        self.data_v_alg.append(self.latest_dq_msg)

        # 更新状态
        self.last_pos = current_pos
        self.last_pos_time = current_time
        
        # 打印进度
        current_len = len(self.times)
        if current_len % 50 == 0:
            print(f"采集进度: {current_len}/{MAX_SAMPLES}")

        # 4. 采集完成
        if current_len >= MAX_SAMPLES:
            self.collecting = False
            self.get_logger().info("采集完成！正在生成 6轴 对比图...")
            self.plot_data()
            raise SystemExit

    def plot_data(self):
        np_v_num = np.array(self.data_v_num)
        np_v_alg = np.array(self.data_v_alg)
        
        # 创建 3行 2列 的画布
        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        # 将二维数组 axes 展平为一维，方便循环 (ax1, ax2, ... ax6)
        axes_flat = axes.flatten()
        
        for i in range(6):
            ax = axes_flat[i]
            
            # 绘制差分速度 (灰色实线，作为背景参考)
            ax.plot(self.times, np_v_num[:, i], label='Diff (Real)', color='gray', alpha=0.5, linewidth=1)
            
            # 绘制算法速度 (红色虚线，作为验证对象)
            ax.plot(self.times, np_v_alg[:, i], label='Algo (Target)', color='red', linewidth=1.5, linestyle='--')
            
            ax.set_title(f"Joint {i + 1}", fontsize=10, pad=5)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # 只在第一个图显示图例，避免遮挡
            if i == 0:
                ax.legend(loc='upper right', fontsize='small')

        # 设置整体标题和标签
        fig.suptitle(f"Velocity Verification (6 DOF) - {MAX_SAMPLES} Frames", fontsize=14)
        # 给底部的图加上 X轴标签
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 1].set_xlabel("Time (s)")
        
        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = DqAllPlotter()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass 
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()