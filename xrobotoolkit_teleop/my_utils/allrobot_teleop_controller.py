import os
from typing import Dict
import rclpy
import numpy as np
import time
import threading
from rclpy.executors import MultiThreadedExecutor

from xrobotoolkit_teleop.my_utils.base_robot_teleop_controller import RobotTeleopController
from xrobotoolkit_teleop.my_utils.ros2_rm65 import (
    LEFT_INITIAL_JOINT_DEG,
    RIGHT_INITIAL_JOINT_DEG,
    ARM_VELOCITY_LIMITS,
    RM65Controller
)

from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

# Default paths and configurations for R1 Lite dual arm
DEFAULT_ALLROBOT_URDF_PATH = os.path.join(ASSET_PATH, "all_robot/urdfmodel.urdf")
DEFAULT_SCALE_FACTOR = 1.13
CONTROLLER_DEADZONE = 0.1


def add_prefix_to_velocity_limits(prefix: str):
    return {f"{prefix}_{k}": v for k, v in ARM_VELOCITY_LIMITS.items()}

# R1 Lite always has both arms - no single arm configuration needed
DEFAULT_MANIPULATOR_CONFIG = {
    "right_arm": {
        "link_name": "right_ee_link",         # 末端执行器链接名称
        "pose_source": "right_controller",   # 使用右手控制器控制
        "control_trigger": "right_grip",     # 右手握持键触发控制
        # "control_mode": "position",       # 可选位置控制或全位姿控制
        "velocity_limits": add_prefix_to_velocity_limits("right"), 
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": [
                "right_gripper",
            ],
            "open_pos": [
                0.0,
            ],
            "close_pos": [
                1.0,
            ],
        },
    },
    "left_arm": {
        "link_name": "left_ee_link",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "velocity_limits": add_prefix_to_velocity_limits("left"),
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "left_trigger",
            "joint_names": [
                "left_gripper",
            ],
            "open_pos": [
                0.0,
            ],
            "close_pos": [
                1.0,
            ],
        },
    },
}


class AllRobotTeleopController(RobotTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_ALLROBOT_URDF_PATH,
        manipulator_config: dict = DEFAULT_MANIPULATOR_CONFIG,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        q_init: np.ndarray = np.concatenate([LEFT_INITIAL_JOINT_DEG,RIGHT_INITIAL_JOINT_DEG]),
        visualize_placo: bool = False,
        control_rate_hz: int = 100,
        self_collision_avoidance_enabled: bool = False,
        enable_log_data: bool = False,
        log_dir: str = "logs/allrobot",
        log_freq: float = 20,
        # 删除 enable_camera 和 camera_fps 参数
    ):

        self.control_rate_hz=control_rate_hz
        # 我们需要一个 Executor 来管理所有 RM65Controller 节点的通信
        self.executor = None
        self._ros_spin_thread = None
        
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            floating_base=False,
            scale_factor=scale_factor,
            q_init=q_init,
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            self_collision_avoidance_enabled=self_collision_avoidance_enabled,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,

        )

    def _placo_setup(self):
        super()._placo_setup()
        # AllRobot has both left and right arms
        self.placo_arm_joint_slice = {}
        for arm_name in ["left_arm", "right_arm"]:
            config = self.manipulator_config[arm_name]
            ee_link_name = config["link_name"]
            arm_prefix = ee_link_name.replace("ee_link", "")
            arm_joint_names = [f"{arm_prefix}joint_{i}" for i in range(1, 7)]
            self.placo_arm_joint_slice[arm_name] = slice(
                self.placo_robot.get_joint_offset(arm_joint_names[0]),
                self.placo_robot.get_joint_offset(arm_joint_names[-1]) + 1,
            )

    def _robot_setup(self):
        
        if self.executor is not None:
            raise ValueError("Executor already initialized!")
        
        self.executor = MultiThreadedExecutor()
        self.arm_controllers: Dict[str, RM65Controller] = {}
        for arm_name in ["left_arm", "right_arm"]:
            arm_prefix = arm_name.replace("_arm", "")
            controller = RM65Controller(
                arm_side=arm_name,
                gripper_control_topic="gripper_cmd",
            )
            self.arm_controllers[arm_name] = controller
            self.executor.add_node(controller)
        
        # 2. 显式启动 ROS 线程 (Start)
        self._ros_spin_thread = threading.Thread(target=self._ros_spin_loop, daemon=True)
        self._ros_spin_thread.start()
        print("ROS 通信线程已启动")


    def _ros_spin_loop(self):
            """后台线程：持续运行 ROS 事件循环"""
            try:
                self.executor.spin() # pyright: ignore[reportOptionalMemberAccess]
            except Exception as e:
                print(f"ROS2 Executor 线程错误: {e}")
            print("ROS2 Executor 线程已停止。")
            
    def wait_for_hardware(self, timeout_sec=10.0):
            """供外部调用：阻塞等待直到收到硬件数据"""
            print("正在等待机械臂心跳数据...")
            start_wait = time.time()
            
            while not all(c.timestamp > 0 for c in self.arm_controllers.values()):
                if time.time() - start_wait > timeout_sec:
                    # 这里可以选择抛出异常，或者返回 False 让外部决定怎么处理
                    print(f"警告：{timeout_sec}秒内未收到机械臂数据！")
                    return False
                time.sleep(0.1) # 这里的 sleep 安全，因为 ROS spin 是在独立线程跑的
                
            print("所有机械臂已连接！")
            return True

    def run(self):

        super().run()
            

    def _update_robot_state(self):
        """Reads current joint states from both arm controllers and updates Placo."""
        for arm_name, controller in self.arm_controllers.items():
            self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]] = controller.qpos.copy()

        

    def _send_command(self):
        """Sends the solved joint targets to both arm controllers."""
        for arm_name, controller in self.arm_controllers.items():
            if self.active.get(arm_name, False):
                controller.q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy().tolist()

            controller.q_des_gripper = [
                self.gripper_pos_target[arm_name][gripper_joint]
                for gripper_joint in self.gripper_pos_target[arm_name].keys()
            ]

            controller.publish_arm_control()
            controller.publish_gripper_control()


    def _get_robot_state_for_logging(self) -> Dict:
        """Returns a dictionary of robot-specific data for logging."""
        return {
            "qpos": {arm: controller.qpos for arm, controller in self.arm_controllers.items()},
            "qpos_des": {arm: controller.q_des for arm, controller in self.arm_controllers.items()},
            "gripper_qpos_des": {
                arm: controller.q_des_gripper for arm, controller in self.arm_controllers.items()
            },
            
        }

    def _should_keep_running(self) -> bool:
        """Returns True if the main loop should continue running."""
        return super()._should_keep_running() and rclpy.ok()

    def _shutdown_robot(self):
        """Performs graceful shutdown of the robot hardware."""
        for arm_controller in self.arm_controllers.values():
            arm_controller.stop()
        print("Arm controllers stopped.")

    # def _pre_ik_update(self):
    #     """Updates the chassis and torso velocity commands based on joystick input."""
    #     self._update_joystick_velocity_command()
    #     self._update_torso_velocity_command()

    # def _update_joystick_velocity_command(self):
    #     """Updates the chassis velocity commands based on joystick input."""
    #     left_axis = self.xr_client.get_joystick_state("left")
    #     right_axis = self.xr_client.get_joystick_state("right")

    #     vx = left_axis[1] * self.chassis_velocity_scale[0]
    #     vy = -left_axis[0] * self.chassis_velocity_scale[1]
    #     omega = -right_axis[0] * self.chassis_velocity_scale[2]

    #     self.chassis_controller.set_velocity_command(vx, vy, omega)

    # def _update_torso_velocity_command(self):
    #     buttonY = self.xr_client.get_button_state_by_name("Y")
    #     buttonX = self.xr_client.get_button_state_by_name("X")

    #     vz = 2.5 if buttonY else -2.5 if buttonX else 0.0
    #     self.torso_controller.set_velocity_command(vz)
