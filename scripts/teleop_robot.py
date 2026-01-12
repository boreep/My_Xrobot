import os
import rclpy
import tyro
import numpy as np
from xrobotoolkit_teleop.my_utils.allrobot_teleop_controller import (
    AllRobotTeleopController,
    DEFAULT_MANIPULATOR_CONFIG
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

from xrobotoolkit_teleop.my_utils.ros2_rm65 import (
    LEFT_INITIAL_JOINT_DEG,
    RIGHT_INITIAL_JOINT_DEG,
)


def main(
    # robot_urdf_path: 机器人URDF模型文件路径，默认为RM65左臂模型
    robot_urdf_path: str = os.path.join(ASSET_PATH, "all_robot/urdfmodel.urdf"),
    # robot_urdf_path: str = os.path.join(ASSET_PATH, "right_rm65f/right_rm65.urdf"),
    # scale_factor: 控制缩放因子，增大操作幅度，默认值为1.3
    scale_factor: float = 1.15,
):
    rclpy.init()

    q_init =np.concatenate([LEFT_INITIAL_JOINT_DEG,RIGHT_INITIAL_JOINT_DEG])

    # 创建并初始化遥操作控制器
    controller = AllRobotTeleopController(
        robot_urdf_path=robot_urdf_path,      # 机器人URDF路径
        manipulator_config=DEFAULT_MANIPULATOR_CONFIG,            # 机械臂配置
        scale_factor=scale_factor,            # 控制缩放因子
        q_init=q_init,                        # 添加这一行来设置初始关节角度
        visualize_placo=True,
        control_rate_hz=50,
        self_collision_avoidance_enabled=True,
        enable_log_data=True,
        logger_config_path = "config/default_dataset_config.yaml",
        
    )

    kinetic_energy_task = controller.solver.add_kinetic_energy_regularization_task(1e-6)
    

    # 启动控制器运行
    controller.init_arm()
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)