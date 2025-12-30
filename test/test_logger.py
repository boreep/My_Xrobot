import rclpy
from rclpy.node import Node
import time
import math
import random

# === 导入消息类型 (必须与接收端一致) ===
from sensor_msgs.msg import JointState
from rm_ros_interfaces.msg import Jointpos
# 如果需要测试 gripper，取消下面注释
# from my_interfaces.msg import HeaderFloat32 

class DualArmMockSender(Node):
    def __init__(self):
        super().__init__('dual_arm_mock_sender')
        
        # 发送频率 (Hz) - 设为 50Hz，比接收端的 20Hz 快，测试接收端的降采样能力
        self.rate = 50.0 
        self.timer = self.create_timer(1.0/self.rate, self.timer_callback)
        
        # === 1. 定义发布者 ===
        # 必须与接收端的 topic_config 中的拼接结果完全一致
        
        # 左臂话题
        self.pub_left_cmd = self.create_publisher(Jointpos, 'left_arm/rm_driver/movej_canfd_cmd', 10)
        self.pub_left_state = self.create_publisher(JointState, 'left_arm/joint_states', 10)
        
        # 右臂话题
        self.pub_right_cmd = self.create_publisher(Jointpos, 'right_arm/rm_driver/movej_canfd_cmd', 10)
        self.pub_right_state = self.create_publisher(JointState, 'right_arm/joint_states', 10)

        self.start_time = time.time()
        self.get_logger().info(">>> Mock Sender Started! Sending Sine Wave Data... <<<")

    def timer_callback(self):
        # 1. 生成统一的时间戳 (这对 Synchronizer 至关重要)
        now = self.get_clock().now()
        t = time.time() - self.start_time
        
        # 2. 生成模拟数据 (正弦波，让 7 个关节动起来)
        # 让每个关节有不同的相位，方便看数据
        joints_base = [math.sin(t + i*0.5) for i in range(7)] 
        
        # === 构建消息 ===
        
        # --- 左臂数据 ---
        msg_left_cmd = Jointpos()
        # 假设 Jointpos 只有 joint 字段 (list[float])
        # 如果 Jointpos 有 header，最好也赋值，但如果没有，Synchronizer 会尝试用到达时间匹配
        msg_left_cmd.header.stamp = now.to_msg()
        msg_left_cmd.joint = [j * 1.0 for j in joints_base] 

        msg_left_state = JointState()
        msg_left_state.header.stamp = now.to_msg()
        msg_left_state.name = [f"joint_{i}" for i in range(7)]
        # 状态稍微滞后一点点 cmd，模拟真实物理延迟 (可选，这里设为相同方便测试)
        msg_left_state.position = [j * 1.0 + 0.01 for j in joints_base]

        # --- 右臂数据 (用 Cosine 区分) ---
        joints_right = [math.cos(t + i*0.5) for i in range(7)]
        
        msg_right_cmd = Jointpos()
        msg_right_cmd.header.stamp = now.to_msg()
        msg_right_cmd.joint = [j * 0.8 for j in joints_right] # 幅度小一点

        msg_right_state = JointState()
        msg_right_state.header.stamp = now.to_msg()
        msg_right_state.name = [f"joint_{i}" for i in range(7)]
        msg_right_state.position = [j * 0.8 + 0.01 for j in joints_right]

        # === 3. 几乎同时发布所有消息 ===
        # 这是为了满足接收端 ApproximateTimeSynchronizer 的要求
        self.pub_left_cmd.publish(msg_left_cmd)
        self.pub_left_state.publish(msg_left_state)
        
        self.pub_right_cmd.publish(msg_right_cmd)
        self.pub_right_state.publish(msg_right_state)

        # 打印调试信息 (每1秒一次)
        if int(t * self.rate) % 50 == 0:
            print(f"Sending batch at t={t:.2f}s | Left J1: {joints_base[0]:.2f}")

def main(args=None):
    rclpy.init(args=args)
    sender = DualArmMockSender()
    try:
        rclpy.spin(sender)
    except KeyboardInterrupt:
        pass
    finally:
        sender.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()