import os
import sys
import rclpy
import tyro
import numpy as np
import time
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

from xrobotoolkit_teleop.my_utils.ros2_rm65 import (
    LEFT_INITIAL_JOINT_DEG,
    RIGHT_INITIAL_JOINT_DEG,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 修改：使用 insert(0, ...) 替代 append(...)
# 这样可以确保优先搜索项目根目录，覆盖 Python 内置的 test 模块
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在这个导入应该能正常工作了，前提是你本地的 test 文件夹包含 __init__.py
from dataset.test.dataset_controller import DatasetController


def main(
    # robot_urdf_path: 机器人URDF模型文件路径，默认为RM65左臂模型
    robot_urdf_path: str = os.path.join(ASSET_PATH, "all_robot/urdfmodel.urdf"),
    # robot_urdf_path: str = os.path.join(ASSET_PATH, "right_rm65f/right_rm65.urdf"),
    # scale_factor: 控制缩放因子，增大操作幅度，默认值为1.3
    scale_factor: float = 2.5,
    position_feedback_enabled: bool = True,
    position_feedback_gain: float = 1,
    position_feedback_max_step: float = 0.1,
    position_feedback_max_total: float = 0.5,
    csv_log_enabled: bool = True,
    csv_log_path: str = "data/with_feedback_high.csv",
):
    rclpy.init()

    q_init =np.concatenate([LEFT_INITIAL_JOINT_DEG,RIGHT_INITIAL_JOINT_DEG])

    # 创建并初始化遥操作控制器
    controller = DatasetController(
        dataset_path="dataset/test_dataset_20260326/run_174811/episode_0_174856.h5",
        robot_urdf_path=robot_urdf_path,      # 机器人URDF路径
        scale_factor=scale_factor,            # 控制缩放因子
        q_init=q_init,                        # 添加这一行来设置初始关节角度
        visualize_placo=True,
        control_rate_hz=40,
        position_feedback_enabled=position_feedback_enabled,
        position_feedback_gain=position_feedback_gain,
        position_feedback_max_step=position_feedback_max_step,
        position_feedback_max_total=position_feedback_max_total,
        csv_log_enabled=csv_log_enabled,
        csv_log_path=(csv_log_path or None),
        # self_collision_avoidance_enabled=True,
        
    )

    # controller.solver.add_kinetic_energy_regularization_task(1e-6)
    

    # 启动控制器运行
    # controller.init_arm()
    
    
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
