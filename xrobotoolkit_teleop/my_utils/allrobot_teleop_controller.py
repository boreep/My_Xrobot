#!/usr/bin/env python3
import os
from typing import Dict
import rclpy
import numpy as np
import time
import threading
import copy
import yaml
from rclpy.executors import MultiThreadedExecutor

from xrobotoolkit_teleop.my_utils.logger.universal_logger import UniversalDataLogger,RecorderState
from xrobotoolkit_teleop.my_utils.filter.filters import OneEuroFilter

from xrobotoolkit_teleop.my_utils.base_robot_teleop_controller import RobotTeleopController
from xrobotoolkit_teleop.my_utils.ros2_rm65 import (
    LEFT_INITIAL_JOINT_DEG,
    RIGHT_INITIAL_JOINT_DEG,
    ARM_VELOCITY_LIMITS,
    DEFAULT_ALLROBOT_URDF_PATH,
    DEFAULT_SCALE_FACTOR,
    CONTROLLER_DEADZONE,
    DEFAULT_MANIPULATOR_CONFIG,
    RM65Controller
)

from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)

from xrobotoolkit_teleop.utils.terminalcolor import TerminalColor


class AllRobotTeleopController(RobotTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_ALLROBOT_URDF_PATH,
        manipulator_config: dict = DEFAULT_MANIPULATOR_CONFIG,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        q_init: np.ndarray = np.zeros(12),
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
        # === 新增：线程同步机制 ===
        self._cmd_lock = threading.Lock()
        # 共享缓冲区：存储最新的 IK 结果
        # 结构：{ arm_name: {'q_des': [], 'dq_des': [], 'gripper': ...} }
        self._latest_command_buffer = {}
        
        # 初始化缓冲区，防止发送线程启动时报错
        for arm_name in ["left_arm", "right_arm"]:
            self._latest_command_buffer[arm_name] = {
                'q_des': None,   # 初始为空，发送时做检查
                'dq_des': [0.0]*6,
                'gripper_pos': {},
                'ik_target': None
            }
        
                # 如果启用了记录数据集，初始化logger ROS 节点
        if self.enable_log_data:
            self._init_logging_node(logger_config_path)
        
        self._axis_click_state = {}
        
        self.joint_vel_filters = {
            "left_arm": OneEuroFilter(min_cutoff=1.0, beta=1.2, d_cutoff=2.0),
            "right_arm": OneEuroFilter(min_cutoff=1.0, beta=1.2, d_cutoff=2.0)
        }
        
   
    def _placo_setup(self):
        super()._placo_setup()

        # 1) 只做一次：缓存 placo 的 joint_names 顺序
        self.all_joint_names_cache = list(self.placo_robot.joint_names())

        # 2) 单臂 6 关节限速（你原来就有）
        limit_list = [ARM_VELOCITY_LIMITS[f"joint_{i}"] for i in range(1, 7)]
        self.vel_limits_array = np.array(limit_list, dtype=float)
        print(f"[placo_setup] Velocity Limits (scaled): {self.vel_limits_array}")

        # 3) 缓存：slice（用于写 state.q），以及严格 6 关节索引（用于 J 列选择）
        self.placo_arm_joint_slice = {}
        self.arm_joint_indices = {}
        self.arm_joint_names = {}   # 新增：保存严格 6 关节名，方便 debug

        for arm_name in ["left_arm", "right_arm"]:
            config = self.manipulator_config[arm_name]

            ee_link_name = config["link_name"]
            arm_prefix = ee_link_name.replace("ee_link", "")
            arm_joint_names_list = [f"{arm_prefix}joint_{i}" for i in range(1, 7)]
            self.arm_joint_names[arm_name] = arm_joint_names_list

            # slice：你原来逻辑保持
            self.placo_arm_joint_slice[arm_name] = slice(
                self.placo_robot.get_joint_offset(arm_joint_names_list[0]),
                self.placo_robot.get_joint_offset(arm_joint_names_list[-1]) + 1,
            )
            # print(f"[placo_setup] {arm_name} Slice Range: {self.placo_arm_joint_slice[arm_name]}")

            # ===== 关键修复：严格 6 关节名 -> 索引，不允许 prefix in name =====
            indices = []
            for jn in arm_joint_names_list:
                if jn not in self.all_joint_names_cache:
                    raise RuntimeError(
                        f"[placo_setup][{arm_name}] joint '{jn}' not found in placo_robot.joint_names(). "
                        f"Check URDF naming / link_name prefix."
                    )
                indices.append(self.all_joint_names_cache.index(jn))

            if len(indices) != 6:
                raise RuntimeError(f"[placo_setup][{arm_name}] IK indices len != 6: {indices}")

            self.arm_joint_indices[arm_name] = indices
            # print(f"[placo_setup] {arm_name} IK Indices (STRICT 6): {indices} -> {arm_joint_names_list}")

            # IK Target init：你原来逻辑保持
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
            start_wait = time.perf_counter()

            while not all(c.timestamp > 0 for c in self.arm_controllers.values()):
                if time.perf_counter() - start_wait > timeout_sec:
                    # 这里可以选择抛出异常，或者返回 False 让外部决定怎么处理
                    print(f"[Hardware_Connect] {TerminalColor.WARNING}警告：{timeout_sec}秒内未收到机械臂数据！{TerminalColor.ENDC}")
                    return False
                time.sleep(0.2) # 这里的 sleep 安全，因为 ROS spin 是在独立线程跑的
                
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

    def _updtae_ik_target_msg(self):

        for arm_name, config in self.manipulator_config.items():
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            self.ik_targets[arm_name] = {
                "pos": np.array(ee_xyz),
                "quat": np.array(ee_quat),#w,x,y,z
            }

    # ============================================================
    # 线程 1 (生产者): IK 计算线程
    # ============================================================
    def _ik_thread(self, stop_event: threading.Event):
        """
        IK 线程只负责计算，计算完毕后更新共享缓冲区。
        它不需要严格 sleep，算完就更新，保证数据最新鲜。
        """
        print(f"[IK_Thread] {TerminalColor.OKGREEN}IK 计算线程启动{TerminalColor.ENDC}")
        period = 1.0 / self.control_rate_hz  # 0.02s
        next_wake_time = time.perf_counter() # 使用高精度计时器
        
        while not stop_event.is_set():
            # 1. 计算下一次唤醒时间 (累计误差消除法)
            next_wake_time += period
        
            # 2. 执行计算逻辑
            self._update_gripper_target() # 更新夹爪目标
            self._pre_ik_update()         # 更新 state
            self._update_ik()             # 核心 IK 求解 (耗时波动点)
            # self._updtae_ik_target_msg()      # 更新IK目标

            # 3. 准备要提交的数据包
            new_commands = {}

            for arm_name in ["left_arm", "right_arm"]:
                if self.active[arm_name]:
                    # 获取目标关节角度
                    q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy().tolist()

                    dq_des = self.get_all_joint_velocities(arm_name).tolist() # pyright: ignore[reportOptionalMemberAccess]

                    # 计算前馈速度
                    # 注意：controller_velicity 拼写需在 Base 修正，这里假设是 controller_velocity
                    # 如果没有 velocity，默认为 0
                    # if hasattr(self, 'controller_velocity') and arm_name in self.controller_velocity:
                    #     #!!!DEBUG!!!
                    #     # dq_des = self.calculate_feedforward_velocity(arm_name, self.controller_velocity[arm_name]).tolist()
                    #     # 2. 获取速度切片
                    #     idx = self.arm_joint_indices[arm_name]  # 严格 6 关节索引（对应 joint_names_cache）
                    #     dq_des = np.asarray(self.placo_robot.state.qd, dtype=float)[idx].copy().tolist()

                    #     #!!!DEBUG!!!
                    # else:
                    #     dq_des = [0.0] * 6
                    # 获取 IK Target
                    ik_target = copy.deepcopy(self.ik_targets.get(arm_name))
                else:
                    # 如果未激活，保持 None 或由发送线程处理维持现状
                    q_des = None
                    dq_des = [0.0] * 6
                    ik_target = None

                # 获取夹爪数据
                gripper_data = copy.deepcopy(self.gripper_pos_target.get(arm_name, {}))

                new_commands[arm_name] = {
                    'q_des': q_des,
                    'dq_des': dq_des,
                    'gripper_pos': gripper_data,
                    'ik_target': ik_target
                }

            # 4. 【关键】加锁更新缓冲区
            with self._cmd_lock:
                # 这是一个极快的内存拷贝操作，几乎不阻塞
                for arm_name, cmd in new_commands.items():
                    self._latest_command_buffer[arm_name] = cmd

            # 5. 可选：微小的休眠防止 CPU 100% 占用，但不要 sleep 太多
            now=time.perf_counter()
            sleep_duration = next_wake_time - now

            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                # 如果计算超时（Lagging），不要死补之前的 sleep，直接重置时间基准
                # 这样可以防止机器人为了“赶进度”而快速执行多次，导致动作瞬变
                if sleep_duration < -period: 
                     # print(f"[IK_Thread] Loop running slow! Lag: {-sleep_duration:.4f}s")
                    next_wake_time =  now   # 重置基准，丢弃过去的时间


    # ============================================================
    # 线程 2 (消费者): 发送控制线程 - 绝对定时 50Hz
    # ============================================================
    def _control_thread(self, stop_event: threading.Event):
        """
        控制线程拥有最高优先级，严格按照 50Hz (20ms) 周期运行。
        无论 IK 是否算完，它都会到点发送。
        """
        print(f"[Control_Thread] {TerminalColor.OKGREEN}50Hz 控制发送线程启动{TerminalColor.ENDC}")
        
        period = 1.0 / self.control_rate_hz  # 0.02s
        next_wake_time = time.perf_counter() # 使用高精度计时器
        
        while not stop_event.is_set():
            # 1. 计算下一次唤醒时间 (累计误差消除法)
            next_wake_time += period
            
            # 2. 【关键】从缓冲区取出最新指令
            cmd_snapshot = None
            with self._cmd_lock:
                # 深拷贝一份数据，避免持有锁的时间过长
                # deepcopy 可能略慢，如果性能敏感，可以只做引用拷贝，但要小心数据篡改
                # 这里数据量小，copy 耗时微秒级，安全第一
                cmd_snapshot = copy.deepcopy(self._latest_command_buffer)
            
            # 3. 执行发送逻辑 (这是原来 _send_command 的内容)
            if cmd_snapshot:
                self._send_command(cmd_snapshot)
            
            # 4. 绝对睡眠
            # 计算还需要睡多久
            sleep_duration = next_wake_time - time.perf_counter()
            
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                # 如果 sleep_duration < 0，说明逻辑执行超时了！
                # 在 50Hz 下通常不会，除非系统极度卡顿
                # 可以选择打印警告，但不要 break，继续追赶
                print(f"[Control_Thread] {TerminalColor.WARNING}警告：控制线程执行超时 {sleep_duration:.6f}s{TerminalColor.ENDC}")
                pass

    def _send_command(self, cmd_data):
        
        if not hasattr(self, '_axis_click_state'):
            self._axis_click_state = {}
            
        current_time = time.perf_counter()
        
        """实际将数据分发给各个 Arm Controller"""
        for arm_name, controller in self.arm_controllers.items():
            
            # =========================================================
            # Part 1: 摇杆长按检测 (合并在这里)
            # =========================================================
            
            # A. 确定要检测的按钮
            target_btn = "left_axis_click" if "left" in arm_name else "right_axis_click"
            
            # B. 获取按钮状态 (保留异常处理，防止 XR 掉线导致整个控制循环崩溃)
            try:
                is_pressed = self.xr_client.get_button_state_by_name(target_btn)
            except Exception:
                is_pressed = False
            
            # C. 初始化该臂的状态记录
            if arm_name not in self._axis_click_state:
                self._axis_click_state[arm_name] = {'start_time': None, 'triggered': False}
            
            state = self._axis_click_state[arm_name]
            
            # D. 状态机判断
            if is_pressed:
                if state['start_time'] is None:
                    # 刚按下：记录时间
                    state['start_time'] = current_time
                    state['triggered'] = False
                else:
                    # 按住中：计算时长
                    duration = current_time - state['start_time']
                    if duration > 1.0 and not state['triggered']:
                        # ---> 触发初始化 <---
                        print(f"[Control] {TerminalColor.HEADER}检测到 {arm_name} 摇杆长按 > 1s，执行初始化Movej...{TerminalColor.ENDC}")
                        
                        # 【重要】必须使用独立线程，否则会卡死整个控制循环
                        threading.Thread(
                            target=controller.init_arm_cmd,
                            daemon=True
                        ).start()
                        
                        state['triggered'] = True # 锁定，防止重复触发
            else:
                # 松开：重置
                state['start_time'] = None
                state['triggered'] = False
                
            # =========================================================
            # Part 2: 正常的运动控制逻辑
            # =========================================================                
            arm_data = cmd_data.get(arm_name)
            
            if not arm_data:
                continue

            # 1. 处理机械臂运动
            # 如果 IK 线程算出的是 None (未激活)，则不发送新的 q_des
            # Controller 会保持上一次的位置 (或者您可以显式发送当前位置)
            controller.q_des = arm_data['q_des']
            if arm_data['dq_des'] is not None:
                controller.dq_des = arm_data['dq_des']
                     
                # 只有激活时才发布手臂控制
            if self.active.get(arm_name, False):
                controller.publish_arm_control() # 发布 ROS 话题
                controller.publish_dq_target()
                controller.publish_ik_vel_target()
        
            # 2. 处理夹爪
            controller.q_des_gripper = [
                arm_data['gripper_pos'][j_name] 
                for j_name in arm_data['gripper_pos']
            ]
            controller.publish_gripper_control()
            
            # 3. 更新 Debug 信息 (可选)
            if arm_data['ik_target']:
                controller.ik_target = arm_data['ik_target']


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
            # print(f"[Logging_Thread] B Button State: {b_button_state}")
        except Exception:
            # 如果 VR 还没准备好，默认 False
            b_button_state = False

        current_is_active = bool(self.active) and any(self.active.values())

        self.data_logger.update_active_status(current_is_active)

        if b_button_state and not self._prev_b_button_state:
            
            # 根据 Logger 当前的状态，决定是开始还是停止
            if self.data_logger.current_state == RecorderState.IDLE:
                print("[Logic] B键按下 -> 检测到空闲 -> 开始录制")
                self.data_logger.start_episode()
            else:
                # 只要不是 IDLE (即 RECORDING 或 PAUSED)，按一下就停止并保存
                print("[Logic] B键按下 -> 检测到录制中 -> 停止保存")
                self.data_logger.stop_episode()
        
        self._prev_b_button_state = b_button_state
        
    def _data_logging_thread(self, stop_event: threading.Event):
        """专用线程：监控用户输入并指挥 Logger"""
        print(f"{TerminalColor.OKGREEN}Data logging logic thread started...{TerminalColor.ENDC}")
        while not stop_event.is_set():
            start_time = time.perf_counter()
            
            self._handle_logging_logic()

            elapsed_time = time.perf_counter() - start_time
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
                time.sleep(0.2)
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
            
    def get_all_joint_velocities(self, arm_name):
        """
        获取指定手臂的所有关节速度，先进行【加速度限幅】，再经过 OneEuroFilter 滤波去噪。
        """
        # ================= 配置区域 =================
        # 最大允许加速度 (rad/s^2)。
        # 建议值：5.0 ~ 20.0。如果不确定，先设大一点(如 15.0)以免拖慢正常运动。
        MAX_ACCEL_LIMIT = 12.0  
        # ===========================================

        # 1. 获取位置切片
        q_slice = self.placo_arm_joint_slice[arm_name]
        
        # 2. 转换为速度切片 (索引 -1)
        v_slice = slice(q_slice.start - 1, q_slice.stop - 1)
        
        # 3. 读取原始速度 (含噪声)
        # 务必转为 numpy array 以支持后续的向量运算
        raw_velocities = np.array(self.placo_robot.state.qd[v_slice], dtype=float)
        
        # 获取当前时间
        current_time = time.perf_counter()

        # 4. === [新增] 加速度限幅 (Slew Rate Limiter) ===
        # 初始化存储结构
        if not hasattr(self, '_acc_limit_state'):
            self._acc_limit_state = {}

        if arm_name not in self._acc_limit_state:
            # 第一次运行：无法计算差分，直接通过，并初始化状态
            clamped_velocities = raw_velocities
            self._acc_limit_state[arm_name] = {
                'last_vel': raw_velocities,
                'last_time': current_time
            }
        else:
            # 获取上一帧状态
            state = self._acc_limit_state[arm_name]
            last_vel = state['last_vel']
            last_time = state['last_time']
            
            # 计算 dt (时间差)
            dt = current_time - last_time
            
            # 异常处理：防止 dt 过小导致除零或逻辑错误
            if dt < 1e-5: 
                dt = 1e-5 
            
            # A. 计算本帧允许的最大速度变化量 (Delta V = a * t)
            max_delta_v = MAX_ACCEL_LIMIT * dt
            
            # B. 计算原始数据的变化量
            delta_v = raw_velocities - last_vel
            
            # C. 核心限幅：将变化量钳制在 [-max, +max] 之间
            # np.clip 会对数组中每个关节单独处理，保留方向
            clamped_delta_v = np.clip(delta_v, -max_delta_v, max_delta_v)
            
            # D. 得到限幅后的速度 (这是物理上可行的速度)
            clamped_velocities = last_vel + clamped_delta_v
            
            # 更新状态供下一帧使用
            self._acc_limit_state[arm_name]['last_vel'] = clamped_velocities
            self._acc_limit_state[arm_name]['last_time'] = current_time

        # 5. === 应用 OneEuroFilter 滤波 ===
        # 防御性编程：确保滤波器存在
        if not hasattr(self, 'joint_vel_filters') or arm_name not in self.joint_vel_filters:
            if not hasattr(self, 'joint_vel_filters'): self.joint_vel_filters = {}
            # 注意：既然前面已经有了加速度限幅，这里的 min_cutoff 可以适当调低(如 0.5 或 0.1)
            # 因为大的尖刺已经被切掉了，滤波器只需负责平滑即可
            self.joint_vel_filters[arm_name] = OneEuroFilter(min_cutoff=0.2, beta=0.8)
        
        # 【关键】把限幅后的速度 (clamped_velocities) 传给滤波器，而不是 raw_velocities
        filtered_velocities = self.joint_vel_filters[arm_name].process(clamped_velocities, current_time)
        
        # [调试打印] 看看限幅器有没有生效 (观察 diff 是否被截断)
        # if arm_name == "right_arm":
        #    print(f"Raw: {raw_velocities[3]:.3f} | Clamped: {clamped_velocities[3]:.3f} | Filtered: {filtered_velocities[3]:.3f}")

        return filtered_velocities   
    # def get_all_joint_velocities(self, arm_name):
    #     """
    #     获取指定手臂的所有关节速度，并经过 OneEuroFilter 滤波去噪。
    #     """
        
    #     # ================= 配置区域 =================
    #     # 最大允许加速度 (rad/s^2)。
    #     # 建议值：5.0 ~ 20.0。如果不确定，先设大一点(如 15.0)以免拖慢正常运动。
    #     MAX_ACCEL_LIMIT = 12.0  
    #     # ===========================================
        
    #     # 1. 获取位置切片
    #     q_slice = self.placo_arm_joint_slice[arm_name]
        
    #     # 2. 转换为速度切片 (索引 -1)
    #     v_slice = slice(q_slice.start - 1, q_slice.stop - 1)
        
    #     # 3. 读取原始速度 (含噪声)
    #     raw_velocities = self.placo_robot.state.qd[v_slice]
        
    #     # 4. === 应用 OneEuroFilter 滤波 ===
    #     # 获取高精度时间戳
    #     current_time = time.perf_counter()
        
    #     # 防御性编程：确保滤波器存在
    #     if not hasattr(self, 'joint_vel_filters') or arm_name not in self.joint_vel_filters:
    #         # 如果没初始化，就现场造一个，防止报错
    #         if not hasattr(self, 'joint_vel_filters'): self.joint_vel_filters = {}
    #         self.joint_vel_filters[arm_name] = OneEuroFilter(min_cutoff=0.2, beta=1.0)
        
    #     # 执行滤波 (输入/输出都是 6维 numpy 数组)
    #     # process 内部会自动处理 dt，所以即便调用频率有波动也能保持稳定
    #     filtered_velocities = self.joint_vel_filters[arm_name].process(raw_velocities, current_time)
        
    #     # [可选调试] 打印对比
    #     # print(f"[{arm_name}] Raw: {raw_velocities[3]:.3f} -> Filtered: {filtered_velocities[3]:.3f}")

    #     return filtered_velocities
    
    
    # def calculate_feedforward_velocity(self, arm_name, controller_vel):
    #     """
    #     Final verified version:
    #     - twist(world) -> dq (model joint direction, NO manual sign flip)
    #     - adaptive DLS for singularity
    #     - task-space (twist) limiting before IK
    #     - joint velocity limiting using ARM_VELOCITY_LIMITS
    #     - mild temporal smoothing
    #     """

    #     # =========================
    #     # 0. 缓存初始化
    #     # =========================
    #     if not hasattr(self, "_dq_last"):
    #         self._dq_last = {}
    #     if arm_name not in self._dq_last:
    #         self._dq_last[arm_name] = np.zeros(6, dtype=float)

    #     if not hasattr(self, "_ff_last_print"):
    #         self._ff_last_print = {"left_arm": 0.0, "right_arm": 0.0}

    #     now = time.perf_counter()

    #     try:
    #         # =========================
    #         # 1. 输入 twist 整理
    #         # =========================
    #         v_cart = np.asarray(controller_vel, dtype=float).reshape(6,)

    #         # 极小速度直接归零（防抖）
    #         if np.linalg.norm(v_cart) < 1e-6:
    #             self._dq_last[arm_name].fill(0.0)
    #             return self._dq_last[arm_name]

    #         # -------- task-space 限幅（非常关键）--------
    #         # 经验值：与你 XR 手感 + 机械臂能力匹配
    #         max_lin = 0.40   # m/s
    #         max_ang = 1.20   # rad/s

    #         v_lin = v_cart[:3].copy()
    #         v_ang = v_cart[3:].copy()

    #         ln = np.linalg.norm(v_lin)
    #         an = np.linalg.norm(v_ang)

    #         if ln > max_lin:
    #             v_lin *= (max_lin / ln)
    #         if an > max_ang:
    #             v_ang *= (max_ang / an)

    #         v_cart = np.hstack([v_lin, v_ang])

    #         # =========================
    #         # 2. Jacobian 提取（你已验证正确）
    #         # =========================
    #         link_name = self.manipulator_config[arm_name]["link_name"]
    #         J_full = self.placo_robot.frame_jacobian(link_name, "world")

    #         n_joints = len(self.all_joint_names_cache)
    #         base_dofs = int(J_full.shape[1] - n_joints)

    #         arm_idx = self.arm_joint_indices[arm_name]          # len=6
    #         arm_cols = [base_dofs + i for i in arm_idx]

    #         J_arm = J_full[:, arm_cols]                          # (6,6)
    #         if J_arm.shape != (6, 6):
    #             raise RuntimeError(f"[{arm_name}] J_arm shape invalid: {J_arm.shape}")

    #         # =========================
    #         # 3. DLS 求解 dq
    #         # =========================
    #         U, S, Vt = np.linalg.svd(J_arm, full_matrices=False)

    #         sigma_min = float(S[-1])
    #         cond = float(S[0] / max(sigma_min, 1e-9))

    #         # -------- 自适应阻尼（已验证非常适合你）--------
    #         lam_min = 0.01
    #         lam_max = 0.10
    #         k = 0.02
    #         lam = lam_min + (lam_max - lam_min) * np.exp(-sigma_min / k)

    #         S_damped = S / (S**2 + lam**2)
    #         dq = Vt.T @ (S_damped * (U.T @ v_cart))

    #         if not np.all(np.isfinite(dq)):
    #             raise FloatingPointError("NaN/Inf in dq")

    #         # =========================
    #         # 4. 关节速度限幅（硬件约束）
    #         # =========================
    #         vel_limits = np.asarray(self.vel_limits_array, dtype=float).reshape(6,)

    #         ratios = np.abs(dq) / np.maximum(vel_limits, 1e-9)
    #         max_ratio = float(np.max(ratios))

    #         if max_ratio > 1.0:
    #             dq *= (1.0 / max_ratio)

    #         # =========================
    #         # 5. 平滑（避免 dq 抖动）
    #         # =========================
    #         alpha = 0.30
    #         dq = alpha * dq + (1.0 - alpha) * self._dq_last[arm_name]

    #         # =========================
    #         # 6. 自检打印（可长期保留）
    #         # =========================
    #         if now - self._ff_last_print[arm_name] > 0.5:
    #             self._ff_last_print[arm_name] = now

    #             v_pred = J_arm @ dq
    #             err = float(np.linalg.norm(v_pred - v_cart))
    #             denom = max(np.linalg.norm(v_pred) * np.linalg.norm(v_cart), 1e-9)
    #             cos = float(np.dot(v_pred, v_cart) / denom)

    #             print(
    #                 f"[FF_DQ][{arm_name}] cond={cond:.2f}, lam={lam:.3f}, "
    #                 f"err={err:.4f}, cos={cos:.3f}\n"
    #                 f"  v_cart={np.array2string(v_cart, precision=3, suppress_small=True)}\n"
    #                 f"  v_pred={np.array2string(v_pred, precision=3, suppress_small=True)}\n"
    #                 f"  dq    ={np.array2string(dq, precision=3, suppress_small=True)}"
    #             )

    #         # =========================
    #         # 7. 返回并缓存
    #         # =========================
    #         self._dq_last[arm_name] = dq
    #         return dq

    #     except Exception as e:
    #         print(f"[FF_DQ][{arm_name}] ERROR: {e}")
    #         # 出错时平滑退化，避免机械臂抽动
    #         self._dq_last[arm_name] *= 0.8
    #         return self._dq_last[arm_name]

