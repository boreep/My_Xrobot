import rclpy
from rclpy.node import Node
import math
import time
from rm_ros_interfaces.msg import Jointpos

class MultiJointTester(Node):
    def __init__(self):
        super().__init__('multi_joint_tester')
        
        # 话题名称
        self.pub = self.create_publisher(Jointpos, 'right_arm/rm_driver/movej_canfd_cmd', 10)
        
        # 设置频率：50Hz (20ms)
        # 透传模式下，建议频率不要低于 50Hz
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        self.t = 0.0
        self.get_logger().info('开始多关节联动测试 (1, 2, 3轴)...')
        self.get_logger().warn('【注意】请确保机械臂周围无障碍物，且当前处于零位附近！')

    def timer_callback(self):
        msg = Jointpos()
        
        # --- 运动参数设置 ---
        # 基础姿态：假设机械臂目前在零位附近 [0, 0, 0, 0, 0, 0]
        # 如果你的机械臂有特定的初始姿态，请修改这里，例如 [0, -0.5, 1.0, 0, 0, 0]
        base_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 振幅 (弧度)：控制运动幅度，0.3 rad 约等于 17度
        amp = 0.3 
        
        # --- 生成多关节轨迹 ---
        # 关节1 (Base): 正弦波
        j1_val = base_pos[0] + amp * math.sin(self.t)
        
        # 关节2 (Shoulder): 余弦波 (与关节1相差90度相位)，会有画圈的感觉
        j2_val = base_pos[1] + (amp * 0.8) * math.cos(self.t)
        
        # 关节3 (Elbow): 频率快一倍的正弦波，幅度小一点
        j3_val = base_pos[2] + (amp * 0.5) * math.sin(self.t * 2.0)

        # 填充 joint 数组 (注意：rm_driver 可能是 6 自由度或 7 自由度，根据图片是 [6])
        # 这里只动前三个关节，后三个保持 0
        msg.joint = [j1_val, j2_val, j3_val, 0.0, 0.0, 0.0]
        
        # --- 关键控制位 ---
        msg.follow = False   # 必须为 True 才能实现高动态跟随
        msg.expand = 0.0
        msg.dof = 6         # 设置自由度

        self.pub.publish(msg)
        
        # 步进时间，控制整体速度
        self.t += 0.05

def main(args=None):
    rclpy.init(args=args)
    node = MultiJointTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()