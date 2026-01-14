import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

class VelocityVerifier(Node):
    def __init__(self):
        super().__init__('velocity_verifier_node')

        # === 配置 ===
        self.record_duration = 10.0  # 记录时长 (秒)
        self.topic_pose = '/right_arm/ik_target_pose'
        self.topic_vel = '/right_arm/ik_target_vel'

        # === 数据存储 ===
        # Pose data: [time_sec, x, y, z, qx, qy, qz, qw]
        self.pose_data = []
        # Twist data: [time_sec, vx, vy, vz, wx, wy, wz]
        self.twist_data = []

        self.start_time = None
        self.is_recording = True

        # === 订阅者 ===
        self.sub_pose = self.create_subscription(
            PoseStamped, self.topic_pose, self.pose_callback, 10)
        self.sub_vel = self.create_subscription(
            Twist, self.topic_vel, self.vel_callback, 10)

        self.get_logger().info(f"开始记录数据 {self.record_duration} 秒...")
        self.get_logger().info(f"监听话题: {self.topic_pose} 和 {self.topic_vel}")

    def pose_callback(self, msg):
        if not self.is_recording:
            return

        # 获取时间戳 (转为秒)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 记录起始时间以便归零
        if self.start_time is None:
            self.start_time = t

        self.pose_data.append([
            t,
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y, 
            msg.pose.orientation.z, msg.pose.orientation.w
        ])

    def vel_callback(self, msg):
        if not self.is_recording:
            return

        # Twist 通常没有 header，使用当前系统时间
        # 注意：如果 Pose 使用的是仿真时间，这里可能会有偏差，但在实机或同步良好的仿真中通常没问题
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9

        # 如果还没有收到 Pose 数据，先不记录 Vel，以免时间轴对不齐
        if self.start_time is None:
            return

        self.twist_data.append([
            t,
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ])

    def process_and_plot(self):
        # 停止记录
        self.is_recording = False
        self.get_logger().info("记录结束，开始处理数据并绘图...")

        if len(self.pose_data) < 2:
            self.get_logger().warn("接收到的 Pose 数据不足，无法绘图")
            return

        # === 1. 数据转换为 Numpy 数组 ===
        pose_arr = np.array(self.pose_data)
        twist_arr = np.array(self.twist_data)

        # 时间归一化 (从0开始)
        t_pose = pose_arr[:, 0]
        # 如果 twist 数据存在，对其时间进行对齐尝试（仅供参考）
        if len(twist_arr) > 0:
            # Twist 时间轴通常会有偏移，这里不做硬性对齐，直接画在同一张图上观察趋势
            t_twist = twist_arr[:, 0] 

        # === 2. 计算微分速度 (Diff Velocity) ===
        # 这里的 diff_vel 将比原始数据少一个点
        diff_linear = []
        diff_angular = []
        diff_time = []

        for i in range(1, len(pose_arr)):
            dt = t_pose[i] - t_pose[i-1]
            if dt <= 0: continue # 防止除以0

            # --- 线速度微分 ---
            p_curr = pose_arr[i, 1:4]
            p_prev = pose_arr[i-1, 1:4]
            v_lin = (p_curr - p_prev) / dt

            # --- 角速度微分 (基于四元数) ---
            q_curr = pose_arr[i, 4:8] # x, y, z, w
            q_prev = pose_arr[i-1, 4:8]

            # 使用 Scipy 计算相对旋转
            # R_diff = R_curr * inv(R_prev)
            r_curr = R.from_quat(q_curr)
            r_prev = R.from_quat(q_prev)

            # 计算从 prev 到 curr 的相对旋转向量
            # 角速度向量方向即旋转轴，大小为角度/时间
            # 这种方法计算的是 "Body Frame" 还是 "World Frame" 取决于乘法顺序
            # 假设 ik_target_vel 是在世界坐标系下的，我们用 R_diff = R_curr * R_prev.inv()
            r_diff = r_curr * r_prev.inv()
            rot_vec = r_diff.as_rotvec() # 返回 (rx, ry, rz) 模长为旋转弧度

            v_ang = rot_vec / dt

            diff_linear.append(v_lin)
            diff_angular.append(v_ang)
            diff_time.append(t_pose[i]) # 使用当前点的时间

        diff_linear = np.array(diff_linear)
        diff_angular = np.array(diff_angular)
        diff_time = np.array(diff_time)

        # === 3. 绘图 ===
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        fig.suptitle(f'Velocity Verification: Calculated (Diff) vs Topic ({self.topic_vel})')

        labels = ['X', 'Y', 'Z']

        # 绘制线速度
        for i in range(3):
            ax = axes[0, i]
            # 绘制计算出的速度
            ax.plot(diff_time, diff_linear[:, i], label='Calculated (Diff)', color='orange', alpha=0.8)
            # 绘制话题接收的速度
            if len(twist_arr) > 0:
                ax.plot(t_twist, twist_arr[:, i+1], label='Topic Data', color='blue', linestyle='--')

            ax.set_title(f'Linear Velocity {labels[i]}')
            ax.grid(True)
            if i == 0: ax.legend()

        # 绘制角速度
        for i in range(3):
            ax = axes[1, i]
            # 绘制计算出的速度
            ax.plot(diff_time, diff_angular[:, i], label='Calculated (Diff)', color='orange', alpha=0.8)
            # 绘制话题接收的速度
            if len(twist_arr) > 0:
                ax.plot(t_twist, twist_arr[:, i+4], label='Topic Data', color='blue', linestyle='--')

            ax.set_title(f'Angular Velocity {labels[i]}')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = VelocityVerifier()

    # 在单独的线程中运行 ROS spin，以便主线程可以计时并绘图
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        # 等待记录时间
        import time
        start = time.time()
        while time.time() - start < node.record_duration:
            time.sleep(0.1)

        # 处理并绘图 (这会阻塞直到关闭窗口)
        node.process_and_plot()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()