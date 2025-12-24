import threading
import time
from abc import ABC, abstractmethod
from typing import Dict

import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController


class RobotTeleopController(BaseTeleopController, ABC):
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
        enable_log_data: bool,
        log_dir: str,
        log_freq: float,
    ):
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            floating_base=floating_base,
            R_headset_world=R_headset_world,
            scale_factor=scale_factor,
            q_init=q_init,
            dt=1.0 / control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
        )

        self._start_time = 0
        self.control_rate_hz = control_rate_hz
        self.log_freq = log_freq
        self.visualize_placo = visualize_placo

        if self.visualize_placo:
            self._init_placo_viz()

        self._prev_b_button_state = False
        self._is_logging = False

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

    def _log_data(self):
        """Logs the current state of the robot."""
        if not self.enable_log_data:
            return

        timestamp = time.time() - self._start_time
        data_entry = {"timestamp": timestamp}
        data_entry.update(self._get_robot_state_for_logging())
        self.data_logger.add_entry(data_entry)

    def _pre_ik_update(self):
        """Hook for subclasses to run logic before the main IK update."""
        pass

    def _ik_thread(self, stop_event: threading.Event):
        """Dedicated thread for running the IK solver."""
        while not stop_event.is_set():
            start_time = time.time()
            self._update_robot_state()
            self._update_gripper_target()
            self._pre_ik_update()
            self._update_ik()
            if self.visualize_placo:
                self._update_placo_viz()
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

    def _data_logging_thread(self, stop_event: threading.Event):
        """Dedicated thread for data logging."""
        while not stop_event.is_set():
            start_time = time.time()
            self._check_logging_button()
            if self._is_logging:
                self._log_data()
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.log_freq) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("Data logging thread has stopped.")

    def _check_logging_button(self):
        """Checks for the 'B' button press to toggle data logging."""
        b_button_state = self.xr_client.get_button_state_by_name("B")
        right_axis_click = self.xr_client.get_button_state_by_name("right_axis_click")

        if b_button_state and not self._prev_b_button_state:
            self._is_logging = not self._is_logging
            if self._is_logging:
                print("--- Started data logging ---")
            else:
                print("--- Stopped data logging. Saving data... ---")
                self.data_logger.save()
                self.data_logger.reset()

        if right_axis_click and self._is_logging:
            print("--- Stopped data logging. Discarding data... ---")
            self.data_logger.reset()
            self._is_logging = False

        self._prev_b_button_state = b_button_state

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