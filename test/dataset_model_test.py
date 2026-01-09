import sys
import os
import time
import numpy as np
import h5py
import placo
from placo_utils.visualization import robot_viz

# ================= 环境与路径配置 =================
# 自动定位到 My_Xrobot 这一层 (根据你提供的代码逻辑)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

# 如果需要导入自定义的初始角度配置，保留此行，如果用不到可以注释
# from xrobotoolkit_teleop.my_utils.ros2_rm65 import (
#     LEFT_INITIAL_JOINT_DEG,
#     RIGHT_INITIAL_JOINT_DEG,
# )

# ================= 1. 初始化机器人与可视化 =================
robot_urdf_path: str = os.path.join(ASSET_PATH, "all_robot/urdfmodel.urdf")

# robot_state: 用于显示从 h5 读取的实际状态 (joint_state)
robot_state = placo.RobotWrapper(robot_urdf_path)

# robot_cmd: 用于显示从 h5 读取的指令状态 (joint_cmd) - 设置为半透明
robot_cmd = placo.RobotWrapper(robot_urdf_path)

for geom in robot_cmd.visual_model.geometryObjects:
    geom.meshColor[3] = 0.3
    geom.meshMaterial.transparent = True  # 必须加

# 初始化可视化器
viz_state = robot_viz(robot_state, "robot_real")
viz_cmd = robot_viz(robot_cmd, "robot_cmd")


# ================= 2. 建立关节索引映射 =================
# 根据你的代码，关节命名为 left_joint_1 ~ 6
left_joint_names = [f"left_joint_{i}" for i in range(1, 7)]
right_joint_names = [f"right_joint_{i}" for i in range(1, 7)]

# 获取 Placo q 向量中对应的索引位置
# robot.get_joint_offset(name) 返回该关节在全局 q 向量中的起始下标
left_indices = [robot_state.get_joint_offset(name) for name in left_joint_names]
right_indices = [robot_state.get_joint_offset(name) for name in right_joint_names]

# ================= 3. H5 回放逻辑 =================
def playback_h5(file_path):
    print(f"Opening H5 file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File does not exist -> {file_path}")
        return

    with h5py.File(file_path, 'r') as f:
        # --- 读取左臂数据 ---
        # 假设结构: group 'left_arm' -> dataset 'timestamp', 'joint_state', 'joint_cmd'
        if 'left_arm' not in f:
            print("Error: 'left_arm' group not found in H5 file.")
            return

        timestamps = f['timestamp'][:]
        l_states = f['left_arm']['joint_state'][:]  # Shape通常是 (N, 7) 或 (N, 6)
        l_cmds = f['left_arm']['joint_cmd'][:]
        
        # --- 读取右臂数据 (如果有) ---
        has_right = 'right_arm' in f
        if has_right:
            r_states = f['right_arm']['joint_state'][:]
            r_cmds = f['right_arm']['joint_cmd'][:]
            print("Right arm data found.")
        else:
            print("No right arm data found, skipping.")

        num_frames = len(timestamps)
        print(f"Total frames to play: {num_frames}")
        print("Starting playback in 3 seconds...")
        time.sleep(3)

        # --- 循环播放 ---
        start_real_time = time.time()
        start_log_time = timestamps[0] # 记录开始时的日志时间戳 (秒)

        for i in range(num_frames):
            # 1. 时间同步
            current_log_time = timestamps[i]
            
            # 计算这一帧在日志里相对于开始过了多久
            time_elapsed_log = current_log_time - start_log_time
            
            # 计算现实世界里过了多久
            time_elapsed_real = time.time() - start_real_time
            
            # 如果现实跑得太快，就等待；如果卡顿了，就全速追赶
            time_to_wait = time_elapsed_log - time_elapsed_real
            if time_to_wait > 0:
                time.sleep(time_to_wait)

            # 2. 更新 Robot State (实物)
            q_state = robot_state.state.q.copy()
            
            # 映射左臂: zip会自动将 joint_state 的前6个数据 填入 left_indices 对应的位置
            # 即使 h5 数据有 7 列 (含夹爪)，只要 indices 只有 6 个，就只取前 6 个
            for idx_placo, val_h5 in zip(left_indices, l_states[i]):
                q_state[idx_placo] = val_h5
            
            # 映射右臂
            if has_right:
                for idx_placo, val_h5 in zip(right_indices, r_states[i]):
                    q_state[idx_placo] = val_h5
            
            # robot_state.state.q = q_state
            
            # 3. 更新 Robot Cmd (指令/半透明)
            q_cmd = robot_cmd.state.q.copy()
            
            for idx_placo, val_h5 in zip(left_indices, l_cmds[i]):
                q_cmd[idx_placo] = val_h5
                
            if has_right:
                for idx_placo, val_h5 in zip(right_indices, r_cmds[i]):
                    q_cmd[idx_placo] = val_h5
            
            robot_cmd.state.q = q_cmd

            # 4. 渲染
            # 注意: 不需要调用 update_kinematics，除非你需要计算末端位置
            # display 函数只负责搬运 joint values 给 meshcat/viewer
            viz_state.display(robot_state.state.q)
            viz_cmd.display(robot_cmd.state.q)
            
            # 5. 打印状态 (可选)
            if i % 10 == 0:
                print(f"Frame {i}/{num_frames} | LogTime: {current_log_time:.4f}", end='\r')

    print("\nPlayback finished.")

# ================= 运行 =================
if __name__ == "__main__":
    # 请根据实际路径修改
    H5_PATH = "dataset/default_task_20260107/run_150106/episode_0_150205.h5" 
    playback_h5(H5_PATH)