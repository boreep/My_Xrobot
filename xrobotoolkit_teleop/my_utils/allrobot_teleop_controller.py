#!/usr/bin/env python3
import os
from typing import Dict
import rclpy
import numpy as np
import time
import threading
import yaml
from rclpy.executors import MultiThreadedExecutor

from xrobotoolkit_teleop.my_utils.logger.universal_logger import UniversalDataLogger,RecorderState

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
from xrobotoolkit_teleop.utils.terminalcolor import TerminalColor

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
        visualize_placo: bool = True,
        control_rate_hz: int = 50,
        self_collision_avoidance_enabled: bool = False,
        enable_log_data: bool = False,
        logger_config_path: str = "config/default_dataset_config.yaml",
        # 删除 enable_camera 和 camera_fps 参数
    ):

        self.control_rate_hz=control_rate_hz
        # 我们需要一个 Executor 来管理所有 RM65Controller 节点的通信
        self.executor = None
        self._ros_spin_thread = None
        self.is_connected = False
        
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

        )
        
                # 如果启用了记录数据集，初始化logger ROS 节点
        if self.enable_log_data:
            self._init_logging_node(logger_config_path)
   

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
            print(f"[placo_setup] {arm_name} 的关节切片索引范围: {self.placo_arm_joint_slice[arm_name]}")
            
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            self.ik_targets[arm_name] = {
                "pos": np.array(ee_xyz),
                "quat": np.array(ee_quat),
            }
        
        
        
    def _robot_setup(self):
        
        if self.executor is not None:
            raise ValueError("[robot_setup] Executor already initialized!")
        
        self.executor = MultiThreadedExecutor()
        self.arm_controllers: Dict[str, RM65Controller] = {}
        for arm_name in ["left_arm", "right_arm"]:
            arm_prefix = arm_name.replace("_arm", "")
            controller = RM65Controller(
                arm_side=arm_name,

            )
            self.arm_controllers[arm_name] = controller
            self.executor.add_node(controller)
        
        # 2. 显式启动 ROS 线程 (Start)
        self._ros_spin_thread = threading.Thread(target=self._ros_spin_loop, daemon=True)
        self._ros_spin_thread.start()
        print(f"[robot_setup] {TerminalColor.OKGREEN}成功：ROS 通信线程已启动{TerminalColor.ENDC}")


    def _ros_spin_loop(self):
            """后台线程：持续运行 ROS 事件循环"""
            try:
                self.executor.spin() # pyright: ignore[reportOptionalMemberAccess]
            except Exception as e:
                print(f"{TerminalColor.FAIL}错误：ROS2 Executor 线程错误: {e}{TerminalColor.ENDC}")
            print(f"ROS2 Executor 线程已停止。")
            
    def wait_for_hardware(self, timeout_sec=10.0):
            """供外部调用：阻塞等待直到收到硬件数据"""
            print(f"[Hardware_Connect] 正在等待机械臂心跳数据...")
            start_wait = time.time()
            
            while not all(c.timestamp > 0 for c in self.arm_controllers.values()):
                if time.time() - start_wait > timeout_sec:
                    # 这里可以选择抛出异常，或者返回 False 让外部决定怎么处理
                    print(f"[Hardware_Connect] {TerminalColor.WARNING}警告：{timeout_sec}秒内未收到机械臂数据！{TerminalColor.ENDC}")
                    return False
                time.sleep(0.1) # 这里的 sleep 安全，因为 ROS spin 是在独立线程跑的
                
            print(f"[Hardware_Connect] {TerminalColor.OKGREEN}成功：所有机械臂已连接！{TerminalColor.ENDC}")
            return True
    
    def init_arm(self):
        """发送初始化位置"""
        self.is_connected = self.wait_for_hardware()
        if self.is_connected:
            print(f"[Init_Arm] {TerminalColor.OKGREEN}发送机械臂初始化角度Movej消息{TerminalColor.ENDC}")
            for arm_name, controller in self.arm_controllers.items():
                controller.init_arm_cmd()
        else:
            print(f"[Init_Arm] {TerminalColor.WARNING}警告：机械臂连接失败！ 不进行robot_state_update{TerminalColor.ENDC}")
            print(f"[Init_Arm] {TerminalColor.HEADER}进入placo仿真模式{TerminalColor.ENDC}")
            

    def _update_robot_state(self):
        """Reads current joint states from both arm controllers and updates Placo."""
        if self.is_connected:
            for arm_name, controller in self.arm_controllers.items():
                if controller.qpos is None:
                    print(f"[Pre_IK] {TerminalColor.WARNING}警告：{arm_name} 机械臂未读取到state数据!{TerminalColor.ENDC}")
                    time.sleep(0.5)
                    continue
                self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]] = controller.qpos.copy()
        


    def _ik_thread(self, stop_event: threading.Event):
        """Dedicated thread for running the IK solver."""
        while not stop_event.is_set():
            start_time = time.time()
            self._update_gripper_target()
            self._pre_ik_update()
            # if self.visualize_placo:
            #     self._update_placo_viz()
            self._update_ik() #IK执行完成后立即赋值，而非在_send_command中赋值
            for arm_name, controller in self.arm_controllers.items():
                if self.active[arm_name]:
                    controller.q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy().tolist()
                       
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print(f"[IK_Thread] {TerminalColor.FAIL}IK loop has stopped.{TerminalColor.FAIL}")

        
    def _send_command(self):
        """Sends the solved joint targets to both arm controllers."""
        for arm_name, controller in self.arm_controllers.items():
            # if self.active.get(arm_name, False):
            #     controller.q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy().tolist()
            controller.ik_target =self.ik_targets[arm_name]
            
            controller.q_des_gripper = [
                self.gripper_pos_target[arm_name][gripper_joint]
                for gripper_joint in self.gripper_pos_target[arm_name].keys()
            ]
            controller.publish_gripper_control()
            if self.active.get(arm_name, False):
                controller.publish_arm_control()


    def _init_logging_node(self, config_path: str):
        """
        初始化 ROS2 Logger 节点
        """
        # 1. 确保 rclpy 已初始化
        if not rclpy.ok():
            try:
                rclpy.init()
            except Exception:
                pass # 可能在外部已经初始化了

        # 2. 加载 Logger 配置
        # 这里假设 config_path 是相对于项目根目录的，或者绝对路径
        if not os.path.exists(config_path):
            # 尝试拼接路径 (根据您的项目结构调整)
            from xrobotoolkit_teleop.utils.path_utils import DATASET_PATH
            config_path = os.path.join(DATASET_PATH, config_path)
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[Error] Logger config not found at: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 3. 实例化对应的 Logger
        self.data_logger = UniversalDataLogger(config)

            
# === 修改后的日志逻辑处理 ===
    def _handle_logging_logic(self):

        # 1. 获取信号 (VR 手柄的 B 键)
        # 注意：这里假设 xr_client 已经初始化并可用
        try:
            b_button_state = self.xr_client.get_button_state_by_name("B")
        except Exception:
            # 如果 VR 还没准备好，默认 False
            b_button_state = False
        
        # 2. 计算 Active 状态 (死区开关或手柄检测)
        # 这里的 active 是从 BaseController 继承来的 dict，存储了各个手柄的激活状态
        current_is_active = bool(self.active) and any(self.active.values())
        
        # 3. 实时同步 Active 状态给 Logger
        # Logger 会利用这个状态决定是否在 RECORDING 模式下暂停写入数据 (Pause)
        self.data_logger.update_active_status(current_is_active)

        # 4. B 按钮生命周期控制 (Start / Stop Episode)
        # 上升沿：开始新的 Episode
        if b_button_state and not self._prev_b_button_state:
            # 只有在空闲时才能开始
            if self.data_logger.current_state == RecorderState.IDLE:
                self.data_logger.start_episode()

        # 下降沿：结束并保存 Episode
        elif not b_button_state and self._prev_b_button_state:
            # 只有在非空闲时才能停止
            if self.data_logger.current_state != RecorderState.IDLE:
                self.data_logger.stop_episode()
        
        self._prev_b_button_state = b_button_state
        
    def _data_logging_thread(self, stop_event: threading.Event):
        """专用线程：监控用户输入并指挥 Logger"""
        print(f"{TerminalColor.OKGREEN}Data logging logic thread started...{TerminalColor.ENDC}")
        while not stop_event.is_set():
            start_time = time.time()
            
            self._handle_logging_logic()

            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ==========================
    #      核心线程 4: ROS Spin
    # ==========================
    def _spin_logger_node(self, stop_event: threading.Event):
        """专用线程：驱动 ROS 节点接收数据和触发 Timer"""
        print(f"[Logger] {TerminalColor.OKGREEN}Logger ROS spin thread started.{TerminalColor.ENDC}")
        executor = MultiThreadedExecutor()
        executor.add_node(self.data_logger)

        try:
            # 2. 直接使用阻塞式 spin，它会自动利用多线程处理回调
            # 这种方式比 spin_once 循环效率高得多，且对 Timer 的响应最准
            executor.spin()
        except Exception as e:
            print(f"[Logger] {TerminalColor.WARNING}Logger spin error: {e}{TerminalColor.ENDC}")
        finally:
            executor.remove_node(self.data_logger)
            print(f"[Logger] {TerminalColor.OKGREEN}Logger spin thread stopped.{TerminalColor.ENDC}")


    def _should_keep_running(self) -> bool:
        """Returns True if the main loop should continue running."""
        return super()._should_keep_running() and rclpy.ok()

    def _shutdown_robot(self):
        """Performs graceful shutdown of the robot hardware."""
        for arm_controller in self.arm_controllers.values():
            arm_controller.stop()
        print("Arm controllers stopped.")
        
        
    def run(self):
        """启动所有线程"""
        self._start_time = time.time()
        self._stop_event = threading.Event()
        threads = []

        # 1. 启动核心控制与IK线程
        core_threads = {
            "_ik_thread": self._ik_thread,
            "_control_thread": self._control_thread,
        }
        for name, target in core_threads.items():
            thread = threading.Thread(name=name, target=target, args=(self._stop_event,))
            threads.append(thread)

        # 2. 启动日志相关线程 (如果启用)
        if self.enable_log_data and self.data_logger is not None:
            if self.is_connected:
                print(f"[Logger] {TerminalColor.OKGREEN}Logger线程已启动.{TerminalColor.ENDC}")
                # (A) 逻辑控制线程 (单独线程，处理按键，避免阻塞控制)
                log_thread = threading.Thread(
                    name="_data_logging_thread",
                    target=self._data_logging_thread,
                    args=(self._stop_event,),
                )
                threads.append(log_thread)

                # (B) ROS Spin 线程 (必须保留，用于数据接收)
                log_spin_thread = threading.Thread(
                    name="_logger_spin_thread",
                    target=self._spin_logger_node,
                    args=(self._stop_event,),
                )
                threads.append(log_spin_thread)
            else:
                print(f"[Logger] {TerminalColor.WARNING}Logger线程未启动，因为机器人未连接.{TerminalColor.ENDC}")

        # 3. 设置守护并运行
        for t in threads:
            t.daemon = True
            t.start()

        print(f"[Run] {TerminalColor.OKBLUE}Teleoperation running. Press Ctrl+C to exit.{TerminalColor.ENDC}")
        try:
            while self._should_keep_running():
                all_threads_alive = all(t.is_alive() for t in threads)
                if not all_threads_alive:
                    print(f"{TerminalColor.FAIL}A thread has died. Shutting down.{TerminalColor.ENDC}")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            print("Shutting down...")
            self._stop_event.set()
            
            for t in threads:
                t.join(timeout=2.0)
            
            # 清理 ROS 资源
            if self.data_logger:
                try:
                    self.data_logger.destroy_node()
                except:
                    pass
            if rclpy.ok():
                rclpy.shutdown()
                    
            print(f"[Run] {TerminalColor.OKBLUE}All threads have been shut down.{TerminalColor.ENDC}")


    def _get_robot_state_for_logging(self) -> Dict:
        """Returns a dictionary of robot-specific data for logging."""
        return {
            "qpos": {arm: controller.qpos for arm, controller in self.arm_controllers.items()},
            "qpos_des": {arm: controller.q_des for arm, controller in self.arm_controllers.items()},
            "gripper_qpos_des": {
                arm: controller.q_des_gripper for arm, controller in self.arm_controllers.items()
            },
            
        }
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
