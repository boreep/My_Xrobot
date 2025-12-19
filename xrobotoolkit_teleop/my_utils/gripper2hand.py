import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import Float32, Header

from my_interfaces.msg import HeaderFloat32, GripperFloat32
from rm_ros_interfaces.msg import Handangle,Handforce,Handspeed


class Gripper2HandController(Node):
    def __init__(
        self,
        arm_side: str = "right_arm",
        hand_control_topic: str = "rm_driver/movej_canfd_cmd",
        gripper_sub_topic: str = "rm_driver/gripper_cmd",

        rate_hz: float = 20.0,
        
    ):
        super().__init__(f'{arm_side}_hand_controller')
        
        # QoS Profile meant to mimic queue_size=1
        qos = QoSProfile(depth=1)

        self.pub = self.create_publisher(Handangle, f"{arm_side}/{hand_control_topic}", qos)
        
        self.sub = self.create_subscription(
            HeaderFloat32, 
            gripper_sub_topic, 
            self.arm_state_callback, 
            qos
        )
        
# 当前状态量q
        self.qpos = [0.0] * 6
        # self.qvel = [0.0] * 6
        self.qpos_gripper = 0.0
        self.timestamp = 0.0

#目标q
        self.q_des = None

        self.arm_ctrl_msg = Jointpos()
        self.arm_ctrl_msg.follow= follow_mode
        self.arm_ctrl_msg.dof=6

        self.q_des_gripper = [0.0,0.0]

        self.gripper_ctrl_msg = GripperFloat32()

        # Create a timer to run the control loop
        self.timer = self.create_timer(1.0 / rate_hz, self.control_loop)
        self._hand_setup()


    def _hand_setup(self):
        """
        Setup for the hand controller.
        """
        # Initialize any necessary parameters or variables here
        pass

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

        self.gripper_ctrl_msg.header = Header()
        self.gripper_ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.gripper_ctrl_msg.header.frame_id = "gripper_link"
        self.gripper_ctrl_msg.right_gripper = self.q_des_gripper[0]
        self.gripper_ctrl_msg.left_gripper = self.q_des_gripper[1]

        self.gripper_pub.publish(self.gripper_ctrl_msg)

    def control_loop(self):
        """
        Replaces the main loop. Called by timer.
        """
        self.publish_arm_control()
        self.publish_gripper_control()
        
    

