import time
import numpy as np
import matplotlib.pyplot as plt
from xrobotoolkit_teleop.common.filter.filtered_xr_client import FilteredXrClient
from xrobotoolkit_teleop.utils.terminalcolor import TerminalColor

def main():
    # =========================
    # 1. 初始化带参数的 Client
    # =========================
    # 你可以在这里微调参数以观察不同效果
    client = FilteredXrClient(
        # 位置：稍微平滑一点，但要跟手
        pos_min_cutoff=0.5, pos_beta=0.05, 
        # 姿态：通常不需要太强滤波
        rot_min_cutoff=1.0, rot_beta=0.01,
        # 速度：噪声通常很大，需要强力滤波 (min_cutoff 设小)
        vel_min_cutoff=0.5, vel_beta=0.2
    )

    # 数据容器
    timestamps = []
    
    # 位置数据 (X, Y, Z)
    pos_raw = []
    pos_filt = []
    
    # 速度数据 (Linear X, Y, Z)
    vel_raw = []
    vel_filt = []

    print(f"{TerminalColor.OKBLUE}开始采集对比数据 (50Hz)...")
    print(f"{TerminalColor.WARNING}请移动手柄（快慢结合），按 Ctrl+C 停止并生成对比图...{TerminalColor.ENDC}\n")

    freq = 50.0
    dt = 1.0 / freq
    start_time = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()

            # -------------------------------------------------
            # 2. 获取数据
            # 调用 get_... 会自动触发滤波，并更新 last_raw_...
            # -------------------------------------------------
            # 获取滤波后的数据
            f_pose = client.get_pose_by_name("right_controller") # [x,y,z,qx,qy,qz,qw]
            f_vel = client.get_velocity_by_name("right_controller") # [vx,vy,vz, ...]

            # 获取刚才那帧的原始数据 (从 Client 的缓存中拿)
            r_pose = client.last_raw_pose.get("right_controller")
            r_vel = client.last_raw_vel.get("right_controller")

            # -------------------------------------------------
            # 3. 数据存储 (确保数据有效)
            # -------------------------------------------------
            if r_pose is not None and r_vel is not None:
                # 记录时间
                timestamps.append(loop_start)
                
                # 记录位置 (前3维)
                pos_raw.append(r_pose[:3])
                pos_filt.append(f_pose[:3])
                
                # 记录速度 (前3维)
                vel_raw.append(r_vel[:3])
                vel_filt.append(f_vel[:3])

                # 实时打印状态
                if len(timestamps) % 10 == 0:
                    print(f"\r采集点数: {len(timestamps)} | Filtered X-Pos: {f_pose[0]:.4f}", end="")

            # -------------------------------------------------
            # 4. 频率控制
            # -------------------------------------------------
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print(f"\n\n{TerminalColor.OKGREEN}采集结束，正在处理图表...{TerminalColor.ENDC}")
    finally:
        client.close()

    if len(timestamps) < 20:
        print("数据过少，取消绘图。")
        return

    # =========================
    # 5. 数据处理与绘图
    # =========================
    ts = np.array(timestamps) - timestamps[0]
    
    # 转换为 Numpy 数组方便切片
    p_raw = np.array(pos_raw)
    p_filt = np.array(pos_filt)
    v_raw = np.array(vel_raw)
    v_filt = np.array(vel_filt)

    # 创建 3行 x 2列 的图表
    # 左列：位置对比 | 右列：速度对比
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    
    labels = ['X Axis', 'Y Axis', 'Z Axis']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4'] # 橙、绿、蓝

    plt.suptitle("One Euro Filter Comparison: Raw (Gray) vs Filtered (Color)", fontsize=16)

    for i in range(3):
        # -----------------------
        # 左列：位置 (Position)
        # -----------------------
        ax_pos = axs[i, 0]
        # 原始数据：灰色虚线，半透明
        ax_pos.plot(ts, p_raw[:, i], color='gray', alpha=0.5, linestyle='--', linewidth=1, label='Raw Input')
        # 滤波数据：彩色实线
        ax_pos.plot(ts, p_filt[:, i], color=colors[i], linewidth=2, label='Filtered Output')
        
        ax_pos.set_ylabel(f'Pos {labels[i]} (m)', fontweight='bold')
        ax_pos.grid(True, which='both', linestyle='--', alpha=0.5)
        if i == 0: 
            ax_pos.set_title("Position Tracking", fontsize=14)
            ax_pos.legend(loc='upper right')

        # -----------------------
        # 右列：速度 (Velocity)
        # -----------------------
        ax_vel = axs[i, 1]
        # 原始数据
        ax_vel.plot(ts, v_raw[:, i], color='gray', alpha=0.4, linestyle='--', linewidth=1, label='Raw Input')
        # 滤波数据
        ax_vel.plot(ts, v_filt[:, i], color=colors[i], linewidth=1.5, label='Filtered Output')
        
        ax_vel.set_ylabel(f'Vel {labels[i]} (m/s)', fontweight='bold')
        ax_vel.grid(True, which='both', linestyle='--', alpha=0.5)
        if i == 0: 
            ax_vel.set_title("Velocity Tracking", fontsize=14)
            ax_vel.legend(loc='upper right')

    # 设置底部 X 轴标签
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 1].set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print(f"{TerminalColor.OKBLUE}图表已生成。请观察彩色线是否平滑且紧跟灰色线。{TerminalColor.ENDC}")
    plt.show()

if __name__ == "__main__":
    main()