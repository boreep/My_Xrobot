#!/usr/bin/env python3
import sys
import os
import rclpy
# 自动定位到 My_Xrobot 这一层
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    # scale_factor: 控制缩放因子，增大操作幅度，默认值为1.13
    scale_factor: float = 1.2,
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
    )
    # 可选的关节约束任务（当前被注释掉）
    # joints_task = controller.solver.add_joints_task()
    # joints_task.set_joints({joint: 0.0 for joint in controller.placo_robot.joint_names()})
    # joints_task.configure("joints_regularization", "soft", 1e-4)

    # if not controller.wait_for_hardware(timeout_sec=10.0):
    #     raise TimeoutError("尝试连接失败，超时")

    # 启动控制器运行
    controller.init_arm()
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)