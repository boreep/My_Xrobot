import numpy as np

# LEFT_INITIAL_JOINT_DEG = np.deg2rad(np.array([-90, -45, -45, -90, 23, 0.0]))
RIGHT_INITIAL_JOINT_DEG = np.deg2rad(np.array([90, 45, 45, 90, 23, 0.0]))
LEFT_INITIAL_JOINT_DEG = -RIGHT_INITIAL_JOINT_DEG.copy()
# RIGHT_INITIAL_JOINT_DEG = np.deg2rad(np.array([0, 0, 0, 0, 0, 0.0]))

# 通用关节速度限制（不带左右前缀，仅关节编号）
ARM_VELOCITY_LIMITS = {
    "joint_1": 0.8,
    "joint_2": 0.8,
    "joint_3": 0.8,
    "joint_4": 1.2,
    "joint_5": 1.2,
    "joint_6": 1.6,
}