import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile


from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Header
from rm_ros_interfaces.msg import Jointpos
from my_interfaces.msg import GripperFloat32 ,HeaderFloat32

from xrobotoolkit_teleop.my_utils.gripper2hand import Gripper2HandController


class RM65Controller(Node):
    def __init__(
        self,
        arm_side: str = "right_arm",
        arm_control_topic: str = "rm_driver/movej_canfd_cmd",
        gripper_control_topic: str = "rm_driver/gripper_cmd",
        arm_state_topic: str = "joint_states",
        rate_hz: float = 100.0,
        follow_mode: bool = False,
    ):
        super().__init__(f'{arm_side}_controller')
        
        self.handcontroller = Gripper2HandController(
            arm_side=arm_side,
            gripper_sub_topic=gripper_control_topic
        )
        
        # QoS Profile meant to mimic queue_size=1
        qos = QoSProfile(depth=1)

        self.pub = self.create_publisher(Jointpos, f"{arm_side}/{arm_control_topic}", qos)
        
        self.gripper_pub = self.create_publisher(HeaderFloat32, f"{arm_side}/{gripper_control_topic}", qos)
          
        self.sub = self.create_subscription(
            JointState, 
            arm_state_topic, 
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

        self.q_des_gripper = None #trigger 0~1

        self.gripper_ctrl_msg = HeaderFloat32()

        # Create a timer to run the control loop
        self.timer = self.create_timer(1.0 / rate_hz, self.control_loop)

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
        
    

