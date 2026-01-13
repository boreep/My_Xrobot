import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Header
from rm_ros_interfaces.msg import Jointpos, Movej
from my_interfaces.msg import HeaderFloat32,JointPosAndVel
from geometry_msgs.msg import PoseStamped


# LEFT_INITIAL_JOINT_DEG = np.deg2rad(np.array([-90, -45, -45, -90, 23, 0.0]))
RIGHT_INITIAL_JOINT_DEG = np.deg2rad(np.array([90, 45, 45, 90, 23, 0.0]))
LEFT_INITIAL_JOINT_DEG = -RIGHT_INITIAL_JOINT_DEG.copy()
# RIGHT_INITIAL_JOINT_DEG = np.deg2rad(np.array([0, 0, 0, 0, 0, 0.0]))

# 通用关节速度限制（不带左右前缀，仅关节编号）
# ARM_VELOCITY_LIMITS = {
#     "joint_1": 3.1415926,
#     "joint_2": 3.1415926,
#     "joint_3": 3.927,
#     "joint_4": 3.927,
#     "joint_5": 3.927,
#     "joint_6": 3.927,
# }

ARM_VELOCITY_LIMITS = {
    "joint_1": 1.5,
    "joint_2": 1.5,
    "joint_3": 1.5,
    "joint_4": 2.5,
    "joint_5": 2.5,
    "joint_6": 2.5,
}


class RM65Controller(Node):
    def __init__(
        self,
        arm_side: str = "right_arm",
        # rate_hz: float = 100.0,
        follow_mode: bool = False,
    ):
        super().__init__(f'{arm_side}_controller')
        
    # # 手部controller，已舍弃到C++
    #     self.handcontroller = Gripper2HandController(
    #         arm_side=arm_side,
    #         gripper_sub_topic=gripper_control_topic,
    #         rate_hz=20.0,
    #     )
        
        # QoS Profile meant to mimic queue_size=1
        qos = QoSProfile(depth=1)

        self.pub = self.create_publisher(Jointpos, f"{arm_side}/rm_driver/movej_canfd_cmd", qos)
        self.pub_movej = self.create_publisher(Movej, f"{arm_side}/rm_driver/movej_cmd", qos)
        self.pub_ik_target = self.create_publisher(PoseStamped, f"{arm_side}/ik_target_pose", qos)
        self.pub_dq = self.create_publisher(JointPosAndVel, f"{arm_side}/dq_target", qos)
        
        self.gripper_pub = self.create_publisher(HeaderFloat32, f"{arm_side}/gripper_cmd", qos)
          
        self.sub = self.create_subscription(
            JointState, 
            f"{arm_side}/joint_states",
            self.arm_state_callback, 
            qos
        )
        
        self.arm_side = arm_side
# 当前状态量q
        self.qpos = None
        # self.qvel = [0.0] * 6
        # self.qpos_gripper = 0.0
        self.timestamp = 0.0

#目标q
        self.q_des = None
        self.dq_des = None
        

        self.ik_target = {"pos":None, "quat":None}
        self.ik_target_msg = PoseStamped()
        self.ik_target_msg.header=Header()
        self.ik_target_msg.header.frame_id = "ee_link"
        
        self.arm_ctrl_msg = Jointpos()
        self.arm_ctrl_msg.header = Header()
        self.arm_ctrl_msg.follow= follow_mode
        self.arm_ctrl_msg.dof=6

        self.q_des_gripper = [0.0] #trigger 0~1
    
        self.gripper_ctrl_msg = HeaderFloat32()
        self.gripper_ctrl_msg.header=Header()
        
        self.joint_vel_msg=JointPosAndVel()
        self.joint_vel_msg.header=Header()

        
        if self.arm_side == "right_arm":
            self.init_pos=RIGHT_INITIAL_JOINT_DEG.tolist()
        elif self.arm_side == "left_arm":
            self.init_pos=LEFT_INITIAL_JOINT_DEG.tolist() 
        
        
        # Create a timer to run the control loop
        # self.timer = self.create_timer(1.0 / 20, self.control_loop)

    def init_arm_cmd(self):

        # 1. 等待获取状态 (静默等待，不输出过程日志)
        wait_count = 0
        while self.qpos is None and wait_count < 20:
            rclpy.spin_once(self, timeout_sec=0.1)
            wait_count += 1

        # 2. 准备指令
        self.movej_msg = Movej()
        self.movej_msg.speed = 20
        self.movej_msg.dof = 6
        self.movej_msg.block = True 

        # 3. 距离判断
        go_zero_first = True 
        
        if self.qpos is not None:
            curr_arr = np.array(self.qpos)
            # 直接计算距离
            dist_to_zero = np.linalg.norm(curr_arr - np.zeros(6))
            dist_to_init = np.linalg.norm(curr_arr - np.array(self.init_pos))
            
            # 仅输出最终决策
            if dist_to_init < dist_to_zero:
                # self.get_logger().info(f"[{self.arm_side}] 距离检测：跳过回零，直达初始位")
                go_zero_first = False

        else:
            # 异常情况保留 Warning
            self.get_logger().warn(f"[{self.arm_side}] 状态获取超时")

        # 4. 发送指令
        if go_zero_first:
            self.movej_msg.joint = [0.0] * 6
            self.movej_msg.trajectory_connect = 1
            self.pub_movej.publish(self.movej_msg)
        
        self.movej_msg.joint = self.init_pos
        self.movej_msg.trajectory_connect = 0
        self.pub_movej.publish(self.movej_msg)
        
        # 5. 完成
        self.get_logger().info(f"{self.arm_side} 初始化运动指令发送完毕")
    
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
        Publishes arm control messages.
        """
        if self.q_des is None or self.ik_target["pos"] is None or self.ik_target["quat"] is None:
            return

        self.ik_target_msg.header.stamp=self.arm_ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.arm_ctrl_msg.header.frame_id = "rm_joint"
        self.arm_ctrl_msg.joint=self.q_des
        
        self.ik_target_msg.pose.position.x=self.ik_target["pos"][0]
        self.ik_target_msg.pose.position.y=self.ik_target["pos"][1]
        self.ik_target_msg.pose.position.z=self.ik_target["pos"][2]
        self.ik_target_msg.pose.orientation.x=self.ik_target["quat"][0]
        self.ik_target_msg.pose.orientation.y=self.ik_target["quat"][1]
        self.ik_target_msg.pose.orientation.z=self.ik_target["quat"][2]
        self.ik_target_msg.pose.orientation.w=self.ik_target["quat"][3]
        
        self.pub.publish(self.arm_ctrl_msg)
        self.pub_ik_target.publish(self.ik_target_msg)
        
    def dq_publish_arm_control(self):
        """
        Publishes arm control messages.
        """
        if self.q_des is None or self.ik_target["pos"] is None or self.ik_target["quat"] is None:
            return

        self.ik_target_msg.header.stamp=self.joint_vel_msg.header.stamp= self.get_clock().now().to_msg()
        self.joint_vel_msg.joint=self.q_des
        if self.dq_des is not None:
            self.joint_vel_msg.joint_vel=self.dq_des

        self.ik_target_msg.pose.position.x=self.ik_target["pos"][0]
        self.ik_target_msg.pose.position.y=self.ik_target["pos"][1]
        self.ik_target_msg.pose.position.z=self.ik_target["pos"][2]
        self.ik_target_msg.pose.orientation.x=self.ik_target["quat"][0]
        self.ik_target_msg.pose.orientation.y=self.ik_target["quat"][1]
        self.ik_target_msg.pose.orientation.z=self.ik_target["quat"][2]
        self.ik_target_msg.pose.orientation.w=self.ik_target["quat"][3]
        

        self.pub_dq.publish(self.joint_vel_msg)
        self.pub_ik_target.publish(self.ik_target_msg)

    def publish_gripper_control(self):
        """
        Publishes gripper control messages.
        """
        if not self.q_des_gripper:
            return
        
        self.gripper_ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.gripper_ctrl_msg.header.frame_id = "gripper_link"
        self.gripper_ctrl_msg.data = float(self.q_des_gripper[0])

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
        


import rclpy
from rclpy.executors import MultiThreadedExecutor
# ...如果要单独运行测试，需要取消上面关于control_loop和timer的注释 ...

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
            # rate_hz=100.0,    # 机械臂控制频率
            follow_mode=False # 是否开启跟随模式
        )


        # 4. 创建多线程执行器
        # 这允许机械臂(100Hz)和灵巧手(20Hz)的定时器并行运行，互不阻塞
        executor = MultiThreadedExecutor()
        
        # 5. 将两个节点都加入执行器
        executor.add_node(arm_node)
        # executor.add_node(hand_node)

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