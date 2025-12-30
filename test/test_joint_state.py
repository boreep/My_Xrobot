#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('test_joint_publisher')
        # 请修改这里的 topic 名称，例如 '/left_arm/joint_states' 或 '/right_arm/joint_states'
        self.topic_name = '/right_arm/joint_states'
        self.topic_name2 = '/left_arm/joint_states'
        self.publisher_ = self.create_publisher(JointState, self.topic_name, 10)
        self.publisher2_= self.create_publisher(JointState, self.topic_name2, 10)
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50Hz 发送频率
        self.i = 0

    def timer_callback(self):
        msg = JointState()
    
        # 1. 填充 Header (满足 self.timestamp 计算)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        # 2. 填充关节名称 (可选，但为了规范建议加上)
        msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        # 3. 填充位置数据 (关键！必须至少6个)
        # 这里模拟一个简单的正弦波运动，方便观察数据变化
        val = math.sin(self.i * 0.02)
        msg.position = [val, val, val, val, val, val]
        
        # velocity 和 effort 可以为空
        msg.velocity = []
        msg.effort = []

        self.publisher_.publish(msg)
        self.publisher2_.publish(msg)
        # self.get_logger().info(f'Publishing to {self.topic_name}: {msg.position}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()