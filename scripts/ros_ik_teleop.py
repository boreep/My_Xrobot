import os
import sys

import numpy as np
import rclpy
import tyro

from xrobotoolkit_teleop.my_utils.ros2_rm65 import LEFT_INITIAL_JOINT_DEG, RIGHT_INITIAL_JOINT_DEG
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset.test.ros_ik_controller import RosIkController


def main(
    robot_urdf_path: str = os.path.join(ASSET_PATH, "all_robot/urdfmodel.urdf"),
    scale_factor: float = 1.0,
    right_ik_topic: str = "/right_arm/ik_target_pose",
    left_ik_topic: str = "/left_arm/ik_target_pose",
    control_rate_hz: int = 15,
    target_timeout_sec: float = 2.0,
):
    rclpy.init()
    q_init = np.concatenate([LEFT_INITIAL_JOINT_DEG, RIGHT_INITIAL_JOINT_DEG])

    controller = RosIkController(
        right_ik_topic=right_ik_topic,
        left_ik_topic=left_ik_topic,
        target_timeout_sec=target_timeout_sec,
        robot_urdf_path=robot_urdf_path,
        scale_factor=scale_factor,
        q_init=q_init,
        visualize_placo=False,
        control_rate_hz=control_rate_hz,
        self_collision_avoidance_enabled=True, 
    )

    controller.solver.add_kinetic_energy_regularization_task(1e-6)
    controller.init_arm()
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
