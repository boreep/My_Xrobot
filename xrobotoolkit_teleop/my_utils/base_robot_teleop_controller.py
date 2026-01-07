import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional

import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.my_utils.base_controller import BaseController
from xrobotoolkit_teleop.utils.terminalcolor import TerminalColor


class RobotTeleopController(BaseController, ABC):
    """
    An abstract base class for hardware teleoperation controllers that consolidates
    common logic for threading, logging, and visualization.
    """

    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: dict,
        R_headset_world: np.ndarray,
        floating_base: bool,
        scale_factor: float,
        q_init: np.ndarray,
        visualize_placo: bool,
        control_rate_hz: int,
        self_collision_avoidance_enabled: bool,
        enable_log_data: bool,
    ):
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            floating_base=floating_base,
            R_headset_world=R_headset_world,
            scale_factor=scale_factor,
            q_init=q_init,
            dt=1.0 / control_rate_hz,
            self_collision_avoidance_enabled=self_collision_avoidance_enabled,
            enable_log_data=enable_log_data,
        )

        self._start_time = 0
        self.control_rate_hz = control_rate_hz
        self.visualize_placo = visualize_placo

        if self.visualize_placo:
            self._init_placo_viz()

        # === 数据集记录相关初始化 ===
        self._prev_b_button_state = False
        self.data_logger = None
        self.logger_spin_thread = None
        
    @abstractmethod
    def _init_logging_node(self):
        """Initializes the ROS node for logging robot states."""
        pass


    @abstractmethod
    def _robot_setup(self):
        """Initializes hardware-specific interfaces (e.g., CAN, ROS)."""
        pass

    @abstractmethod
    def _update_robot_state(self):
        """Reads the current robot state from hardware and updates Placo."""
        pass

    @abstractmethod
    def _send_command(self):
        """Sends motor commands to the hardware."""
        pass

    @abstractmethod
    def _get_robot_state_for_logging(self) -> Dict:
        """Returns a dictionary of robot-specific data for logging."""
        pass

    @abstractmethod
    def _shutdown_robot(self):
        """Performs graceful shutdown of the robot hardware."""
        pass

    def _get_link_pose(self, link_name: str):
        """Gets the current world pose for a given link name from Placo."""
        T_world_link = self.placo_robot.get_T_world_frame(link_name)
        pos = T_world_link[:3, 3]
        quat = tf.quaternion_from_matrix(T_world_link)
        return pos, quat

    def _pre_ik_update(self):
        """Hook for subclasses to run logic before the main IK update."""
        """placo状态更新"""
        self._update_robot_state()
        self.placo_robot.update_kinematics()
        if self.visualize_placo:
            self._update_placo_viz()



    #已被重写覆盖
    def _ik_thread(self, stop_event: threading.Event):
        """Dedicated thread for running the IK solver."""
        while not stop_event.is_set():
            start_time = time.time()
            self._update_gripper_target()
            self._pre_ik_update()
            # if self.visualize_placo:
            #     self._update_placo_viz()
            self._update_ik()
            
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("IK loop has stopped.")

    def _control_thread(self, stop_event: threading.Event):
        """Dedicated thread for sending commands to hardware."""
        while not stop_event.is_set():
            start_time = time.time()
            self._send_command()
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._shutdown_robot()
        print("Control loop has stopped.")


# 假设这是在你的 VR 控制 Loop 中
    @abstractmethod
    def _handle_logging_logic(self):
        # 1. 获取信号
        pass

    def _data_logging_thread(self, stop_event: threading.Event):
        """Dedicated thread for data logging."""
        print(f"{TerminalColor.WARNING}Data logging thread started (Waiting for B button to start episode)...{TerminalColor.ENDC}")
        while not stop_event.is_set():
            start_time = time.time()
            
            # 执行核心逻辑
            self._handle_logging_logic()

            # 这里复用 control_rate_hz
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)


    def _should_keep_running(self) -> bool:
        """Returns True if the main loop should continue running."""
        return not self._stop_event.is_set()

    def run(self):
        """Main entry point that starts all threads."""

        self._start_time = time.time()
        self._stop_event = threading.Event()
        threads = []

        core_threads = {
            "_ik_thread": self._ik_thread,
            "_control_thread": self._control_thread,
        }
        for name, target in core_threads.items():
            thread = threading.Thread(name=name, target=target, args=(self._stop_event,))
            threads.append(thread)

        if self.enable_log_data:
            log_thread = threading.Thread(
                name="_data_logging_thread",
                target=self._data_logging_thread,
                args=(self._stop_event,),
            )
            threads.append(log_thread)

        for t in threads:
            t.daemon = True
            t.start()

        print("Teleoperation running. Press Ctrl+C to exit.")
        try:
            while self._should_keep_running():
                all_threads_alive = all(t.is_alive() for t in threads)
                if not all_threads_alive:
                    print("A thread has died. Shutting down.")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            print("Shutting down...")
            self._stop_event.set()
            for t in threads:
                t.join(timeout=2.0)
            print("All threads have been shut down.")