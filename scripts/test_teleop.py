import sys
import os
# 自动定位到 My_Xrobot 这一层
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tyro
import numpy as np
from xrobotoolkit_teleop.simulation.placo_teleop_controller import (
    PlacoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


from xrobotoolkit_teleop.hardware.interface.rm65f import (
    LEFT_INITIAL_JOINT_DEG,
    RIGHT_INITIAL_JOINT_DEG,
    ARM_VELOCITY_LIMITS
)



def main(
    # robot_urdf_path: 机器人URDF模型文件路径，默认为RM65左臂模型
    robot_urdf_path: str = os.path.join(ASSET_PATH, "all_robot/urdfmodel.urdf"),
    # robot_urdf_path: str = os.path.join(ASSET_PATH, "right_rm65f/right_rm65.urdf"),
    # scale_factor: 控制缩放因子，增大操作幅度，默认值为1.5
    scale_factor: float = 1.13,
):
    
    # 工具函数：给 ARM_VELOCITY_LIMITS 添加前缀
    def add_prefix_to_velocity_limits(prefix: str):
        return {f"{prefix}_{k}": v for k, v in ARM_VELOCITY_LIMITS.items()}
    """
    主函数：运行RM65双臂的Placo遥操作控制
    """
    # 配置机械臂参数
    config = {
        "right_hand": {
            "link_name": "right_ee_link",         # 末端执行器链接名称
            "pose_source": "right_controller",   # 使用右手控制器控制
            "control_trigger": "right_grip",     # 右手握持键触发控制
            # "control_mode": "position",       # 可选位置控制或全位姿控制
            "velocity_limits": add_prefix_to_velocity_limits("right"), 
        },
        "left_hand": {
            "link_name": "left_ee_link",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "velocity_limits": add_prefix_to_velocity_limits("left"),
        },
    }

    q_init =np.concatenate([LEFT_INITIAL_JOINT_DEG,RIGHT_INITIAL_JOINT_DEG])

    # 创建并初始化遥操作控制器
    controller = PlacoTeleopController(
        robot_urdf_path=robot_urdf_path,      # 机器人URDF路径
        manipulator_config=config,            # 机械臂配置
        scale_factor=scale_factor,            # 控制缩放因子
        q_init=q_init,                        # 添加这一行来设置初始关节角度
        self_collision_avoidance_enabled=True,  # 启用自碰撞避免
        dt=0.02,
    )
    # 可选的关节约束任务（当前被注释掉）
    # joints_task = controller.solver.add_joints_task()
    # joints_task.set_joints({joint: 0.0 for joint in controller.placo_robot.joint_names()})
    # joints_task.configure("joints_regularization", "soft", 1e-4)

    # 启动控制器运行
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)