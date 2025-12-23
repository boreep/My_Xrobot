import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float32, Header
from my_interfaces.msg import HeaderFloat32
from rm_ros_interfaces.msg import Handangle, Handforce, Handspeed, Handstatus

import numpy as np
import time
import math

# ================= 配置区域 =================

# 修正：Angle 初始化最后一位为 9000
HAND_INIT_ANGLE = [226, 10022, 9781, 10138, 9884, 9000]

# 修正：Angle 限制最后一位为 9000
HAND_ANGLE_LIMITS = np.array([
    [226, 3676],
    [10022, 17837],
    [9781, 17606],
    [10138, 17654],
    [9884, 17486],
    [0, 9000]  # 这里保持 9000
], dtype=np.float32)


class Gripper2HandController(Node):
    def __init__(
        self,
        arm_side: str = "right_arm",
        gripper_sub_topic: str = "gripper_cmd",
        rate_hz: float = 20.0,
    ):
        super().__init__(f'{arm_side}_hand_controller')
        
        self.arm_side = arm_side
        
        # QoS Profile
        qos = QoSProfile(depth=1)

        # Publishers
        self.pub_pos = self.create_publisher(Handangle, f"{arm_side}/rm_driver/set_hand_follow_pos_cmd", qos)
        self.pub_angle = self.create_publisher(Handangle, f"{arm_side}/rm_driver/set_hand_follow_angle_cmd", qos)
        self.pub_speed = self.create_publisher(Handspeed, f"{arm_side}/rm_driver/set_hand_speed_cmd", qos)
        self.pub_force = self.create_publisher(Handforce, f"{arm_side}/rm_driver/set_hand_force_cmd", qos)
        
        # Subscribers
        self.sub = self.create_subscription(
            Handstatus, 
            f"{arm_side}/rm_driver/udp_hand_status", 
            self.hand_status_callback, 
            qos
        )
        self.sub_gripper = self.create_subscription(
            HeaderFloat32, 
            f"{arm_side}/{gripper_sub_topic}", 
            self.gripper_cmd_callback, 
            qos
        )
        
        self.timestamp = None
        # 状态变量
        self.q_des_gripper = None
        
        # 使用 uint16 存储，这样既可以存 9000，也可以兼容未来可能的 65535
        self.hand_angle_buffer = np.array(HAND_INIT_ANGLE, dtype=np.uint16)

        # --- 初始化硬件设置 ---
        self.init_hand_controller()

        # Create timer
        self.timer = self.create_timer(1.0 / rate_hz , self.control_loop)

    def init_hand_controller(self):
        """发送初始化力矩、速度和位置"""
        self.get_logger().info("正在初始化灵巧手参数...")
        
        # 1. 设置力矩
        handforce_msg = Handforce()
        handforce_msg.hand_force = 200
        self.pub_force.publish(handforce_msg)
        
        # 2. 设置速度
        handspeed_msg = Handspeed()
        handspeed_msg.hand_speed = 500
        self.pub_speed.publish(handspeed_msg)
        
        time.sleep(1.0) 

        # 3. 设置初始角度
        self.handangle_msg = Handangle()
        # 这里的 view(np.int16) 对于 9000 来说值不变 (还是 9000)
        # 但如果未来你要传 65535，它会自动变成 -1，非常安全
        self.handangle_msg.hand_angle = self.hand_angle_buffer.view(np.int16).tolist()
        self.handangle_msg.block = False
        self.pub_angle.publish(self.handangle_msg)
        
        self.get_logger().info(f"灵巧手初始化完成。初始角度指令已发送: {self.handangle_msg.hand_angle}")
        time.sleep(1.0)

    def hand_status_callback(self, msg: Handstatus):
        self.hand_pos = list(msg.hand_pos)
        self.hand_angle = list(msg.hand_angle)
        self.hand_state = list(msg.hand_state)
        self.hand_force = list(msg.hand_force)
        self.hand_err = msg.hand_err
        
    def gripper_cmd_callback(self, msg: HeaderFloat32):
            # [Safety] 1. NaN 检查
            if math.isnan(msg.data):
                self.get_logger().warn("收到 NaN 数据，忽略指令")
                return

            # [Safety] 2. 更新数据和时间戳
            self.q_des_gripper = msg.data
            self.timestamp= msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
    def gripper2handangle(self, gripper_trigger):
        """
        将 Gripper (0~1) 映射到 Hand Angle (0~5指)
        """
        # 1. 限制输入范围
        trigger = np.clip(gripper_trigger, 0.0, 1.0)
        
        # 2. 提取前5个手指的限制范围 (不包含第6个)
        limits_active = HAND_ANGLE_LIMITS[:5]
        p_min = limits_active[:, 0]
        p_max = limits_active[:, 1]
        
        # 3. 向量化计算前5个手指的角度
        calculated_angles = p_min + (p_max - p_min) * trigger
        
        # 4. 更新 buffer 的前5位
        self.hand_angle_buffer[:5] = calculated_angles.astype(np.uint16)
        
        # 5. 第6位保持初始化时的值 (即 9000)，这里不需要动它
        
        # 6. 返回 int16 列表
        # 9000 -> 9000
        # (假设这里有 65535) -> -1
        return self.hand_angle_buffer.view(np.int16).tolist()

    def control_loop(self):
        if self.q_des_gripper is None:
            return

        target_angles = self.gripper2handangle(self.q_des_gripper)
        self.handangle_msg.hand_angle = target_angles
        self.pub_angle.publish(self.handangle_msg)
        
    def stop(self):
        """
        Unregisters the ROS publishers and subscribers.
        """
        self.destroy_publisher(self.pub_angle)
        self.destroy_publisher(self.pub_speed)
        self.destroy_publisher(self.pub_force)
        self.destroy_subscription(self.sub)
        self.destroy_subscription(self.sub_gripper)

def main(args=None):
    rclpy.init(args=args)
    hand_controller = None
    try:
        hand_controller = Gripper2HandController()
        executor = MultiThreadedExecutor()
        executor.add_node(hand_controller)
        
        print("------------------------------------------")
        print("ROHand 控制节点 (Angle模式) 已启动")
        print("第6自由度固定值: 9000")
        print("------------------------------------------")
        
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if hand_controller:
            hand_controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()