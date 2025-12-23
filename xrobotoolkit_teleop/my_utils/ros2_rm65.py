import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
import os
import sys
# 自动定位到 My_Xrobot 这一层
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Header
from rm_ros_interfaces.msg import Jointpos, Movej
from my_interfaces.msg import HeaderFloat32
import time

from xrobotoolkit_teleop.my_utils.gripper2hand import Gripper2HandController

# LEFT_INITIAL_JOINT_DEG = np.deg2rad(np.array([-90, -45, -45, -90, 23, 0.0]))
RIGHT_INITIAL_JOINT_DEG = np.deg2rad(np.array([90, 45, 45, 90, 23, 0.0]))
LEFT_INITIAL_JOINT_DEG = -RIGHT_INITIAL_JOINT_DEG.copy()
# RIGHT_INITIAL_JOINT_DEG = np.deg2rad(np.array([0, 0, 0, 0, 0, 0.0]))

# 通用关节速度限制（不带左右前缀，仅关节编号）
ARM_VELOCITY_LIMITS = {
    "joint_1": 0.8,
    "joint_2": 0.8,
    "joint_3": 0.8,
    "joint_4": 1.2,
    "joint_5": 1.2,
    "joint_6": 1.6,
}

class RM65Controller(Node):
    def __init__(
        self,
        arm_side: str = "right_arm",
        gripper_control_topic: str = "gripper_cmd",
        rate_hz: float = 100.0,
        follow_mode: bool = False,
    ):
        super().__init__(f'{arm_side}_controller')
        
    # 手部controller
        self.handcontroller = Gripper2HandController(
            arm_side=arm_side,
            gripper_sub_topic=gripper_control_topic,
            rate_hz=20.0,
        )
        
        # QoS Profile meant to mimic queue_size=1
        qos = QoSProfile(depth=1)

        self.pub = self.create_publisher(Jointpos, f"{arm_side}/rm_driver/movej_canfd_cmd", qos)
        self.pub_movej = self.create_publisher(Movej, f"{arm_side}/rm_driver/movej_cmd", qos)
        
        self.gripper_pub = self.create_publisher(HeaderFloat32, f"{arm_side}/{gripper_control_topic}", qos)
          
        self.sub = self.create_subscription(
            JointState, 
            f"{arm_side}/joint_states",
            self.arm_state_callback, 
            qos
        )
        
        self.arm_side = arm_side
# 当前状态量q
        self.qpos = [0.0] * 6
        # self.qvel = [0.0] * 6
        # self.qpos_gripper = 0.0
        self.timestamp = 0.0

#目标q
        self.q_des = None

        self.arm_ctrl_msg = Jointpos()
        self.arm_ctrl_msg.follow= follow_mode
        self.arm_ctrl_msg.dof=6

        self.q_des_gripper = [0.0] #trigger 0~1

        self.gripper_ctrl_msg = HeaderFloat32()
        
        self.init_arm_controller
        # Create a timer to run the control loop
        # self.timer = self.create_timer(1.0 / rate_hz, self.control_loop)

    def init_arm_controller(self):
        """发送初始化位置"""
        self.get_logger().info("正在初始化RM65...")
        
        self.movej_msg = Movej()
        self.movej_msg.speed=20
        self.movej_msg.dof=6
        self.movej_msg.joint=[0.0]*6
        self.movej_msg.trajectory_connect=1
        self.movej_msg.block=True
        
        self.pub_movej.publish(self.movej_msg)
        
        if self.arm_side == "right_arm":
            self.movej_msg.joint=RIGHT_INITIAL_JOINT_DEG.tolist()
        elif self.arm_side == "left_arm":
            self.movej_msg.joint=LEFT_INITIAL_JOINT_DEG.tolist()
        self.movej_msg.block=True
        self.movej_msg.trajectory_connect=0
        self.pub_movej.publish(self.movej_msg)
        
        # 5. 等待确认
        self.get_logger().info("等待RM65初始化完成...5秒")
        time.sleep(5.0)  # 等待 3 秒确认
        self.get_logger().info("RM65初始化完成。")
        
    def arm_state_callback(self, msg: JointState):
        """
        Callback function to handle joint state updates.
        """
        self.qpos = list(msg.position[:6])
        # self.qvel = list(msg.velocity[:6])
        # 注意: 如果索引越界需要加 try-except 或检查 len
        # if len(msg.position) > 6:
        #     self.qpos_gripper = [msg.position[6]]
            # self.qvel_gripper = [msg.velocity[6]]
        if self.q_des is None:
            self.q_des = self.qpos
            
        # ROS2 Time conversion
        self.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def publish_arm_control(self):
        """
        Publishes motor control messages.
        """
        if self.q_des is None:
            return

        self.arm_ctrl_msg.header = Header()
        self.arm_ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.arm_ctrl_msg.header.frame_id = "rm_joint"
        self.arm_ctrl_msg.joint=self.q_des
        
        self.pub.publish(self.arm_ctrl_msg)

    def publish_gripper_control(self):

        if self.q_des_gripper is None:
            return
        self.gripper_ctrl_msg.header = Header()
        self.gripper_ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.gripper_ctrl_msg.header.frame_id = "gripper_link"
        self.gripper_ctrl_msg.data = self.q_des_gripper

        self.gripper_pub.publish(self.gripper_ctrl_msg)

    def control_loop(self):
        """
        Replaces the main loop. Called by timer.
        """
        self.publish_arm_control()
        self.publish_gripper_control()
        
    def stop(self):
        """
        Unregisters the ROS publishers and subscribers.
        """
        self.destroy_publisher(self.pub)
        self.destroy_publisher(self.gripper_pub)
        self.destroy_subscription(self.sub)
        
        self.handcontroller.stop()

        

import rclpy
from rclpy.executors import MultiThreadedExecutor
# ... (保留你上面的 imports 和 class RM65Controller 定义) ...

def main(args=None):
    # 1. 初始化 ROS 2
    rclpy.init(args=args)

    arm_node = None
    hand_node = None
    executor = None

    try:
        # 2. 创建主臂控制节点
        # 你可以通过这里修改 arm_side ("left_arm" / "right_arm")
        target_arm = "right_arm" 
        
        arm_node = RM65Controller(
            arm_side=target_arm,
            gripper_control_topic="gripper_cmd",
            rate_hz=100.0,    # 机械臂控制频率
            follow_mode=False # 是否开启跟随模式
        )

        # 3. 【关键步骤】获取内部创建的手部控制节点
        # 因为 handcontroller 是在该类内部实例化的另一个 Node 对象，
        # 它必须也被加入到 executor 才能工作。
        hand_node = arm_node.handcontroller

        # 4. 创建多线程执行器
        # 这允许机械臂(100Hz)和灵巧手(20Hz)的定时器并行运行，互不阻塞
        executor = MultiThreadedExecutor()
        
        # 5. 将两个节点都加入执行器
        executor.add_node(arm_node)
        executor.add_node(hand_node)

        print("--------------------------------------------------")
        print(f"RM65 组合控制器已启动 [{target_arm}]")
        print(f"  - Arm Control Rate : 100.0 Hz")
        print(f"  - Hand Control Rate: 20.0 Hz (Internal Node)")
        print("--------------------------------------------------")

        # 6. 开始循环
        executor.spin()

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在停止节点...")
    
    finally:
        # 7. 优雅退出与资源清理
        if arm_node:
            arm_node.destroy_node()
        # hand_node 在 arm_node 内部，虽然 arm_node 销毁时 Python 会回收，
        # 但显式销毁是 ROS 2 的好习惯，防止底层 DDS 句柄泄漏
        if hand_node:
            hand_node.destroy_node()
            
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()