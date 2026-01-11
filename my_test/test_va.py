import time
import numpy as np
import matplotlib.pyplot as plt
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.terminalcolor import TerminalColor

def main():
    try:
        client = XrClient()
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 数据存储列表
    timestamps = []
    raw_positions = []
    raw_velocities = []

    # --- 配置 ---
    FREQ = 50.0
    DT = 1.0 / FREQ
    WARMUP_COUNT = 20  
    
    print(f"{TerminalColor.OKBLUE}开始采集 (50Hz)...")
    print(f"前 {WARMUP_COUNT} 帧作为热身数据丢弃。")
    print(f"{TerminalColor.WARNING}按 Ctrl+C 停止并生成 3x2 对比图...{TerminalColor.ENDC}\n")

    frame_count = 0
    valid_data_started = False

    try:
        while True:
            start_loop = time.perf_counter()

            # 1. 获取数据
            pose = client.get_pose_by_name("right_controller")
            velocity = client.get_velocity_by_name("right_controller")

            # 2. 检查数据有效性
            if np.allclose(pose[:3], [0, 0, 0], atol=1e-6):
                time.sleep(0.01)
                continue

            frame_count += 1

            # 3. 热身阶段判断
            if frame_count <= WARMUP_COUNT:
                if frame_count % 5 == 0:
                    print(f"\r{TerminalColor.WARNING}[Warmup]{TerminalColor.ENDC} 等待数据稳定... ({frame_count}/{WARMUP_COUNT})", end="")
            else:
                if not valid_data_started:
                    print(f"\n{TerminalColor.OKBLUE}数据稳定，开始记录！{TerminalColor.ENDC}")
                    valid_data_started = True

                # 4. 记录数据
                timestamps.append(start_loop)
                raw_positions.append(pose[:3])
                raw_velocities.append(velocity[:3])

                # 实时显示
                if frame_count % 5 == 0:
                    vel_str = f"Vel: [{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}]"
                    print(f"\r{TerminalColor.OKGREEN}[Rec]{TerminalColor.ENDC} {vel_str} | Points: {len(timestamps)}", end="")

            # 5. 频率控制
            elapsed = time.perf_counter() - start_loop
            if elapsed < DT:
                time.sleep(DT - elapsed)

    except KeyboardInterrupt:
        print(f"\n\n{TerminalColor.WARNING}停止采集。正在处理...{TerminalColor.ENDC}")
    finally:
        client.close()

    # --- 绘图部分 ---
    if len(timestamps) < 10:
        print("有效数据太少，无法绘图。")
        return

    ts = np.array(timestamps) - timestamps[0]
    pos = np.array(raw_positions)
    vel_raw = np.array(raw_velocities)

    # 微分计算
    diff_pos = np.diff(pos, axis=0)
    diff_time = np.diff(ts).reshape(-1, 1)
    diff_time[diff_time < 1e-6] = 1e-6 
    vel_diff = diff_pos / diff_time

    # 6. 绘图 (3行2列)
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # sharex=True: 所有图共享时间轴
    # sharey='row': 每一行的左右两图共享Y轴刻度 (关键！保证对比真实)
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey='row')
    
    axis_labels = ['X-Axis', 'Y-Axis', 'Z-Axis']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4'] # 对应 X, Y, Z 的颜色

    for i in range(3):
        # --- 左列：SDK 直接获取的速度 ---
        axs[i, 0].plot(ts, vel_raw[:, i], color='gray', alpha=0.8, linewidth=1.5, label='SDK Direct')
        axs[i, 0].set_ylabel(f'{axis_labels[i]} Speed (m/s)', fontweight='bold')
        axs[i, 0].legend(loc='upper right')
        axs[i, 0].grid(True, which='both', linestyle='--', alpha=0.5)
        
        # --- 右列：位置微分得到的速度 ---
        # 注意：微分少一个点，时间轴用 ts[1:]
        axs[i, 1].plot(ts[1:], vel_diff[:, i], color=colors[i], linewidth=1.5, label='Calculated (Diff)')
        axs[i, 1].legend(loc='upper right')
        axs[i, 1].grid(True, which='both', linestyle='--', alpha=0.5)

    # 设置列标题
    axs[0, 0].set_title("SDK Direct Velocity", fontsize=14, fontweight='bold', pad=15)
    axs[0, 1].set_title("Calculated Velocity (from Position)", fontsize=14, fontweight='bold', pad=15)

    # 设置底部X轴标签
    axs[2, 0].set_xlabel('Time (s)', fontsize=12)
    axs[2, 1].set_xlabel('Time (s)', fontsize=12)

    plt.suptitle(f'Velocity Source Comparison (Frequency: {FREQ}Hz)', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 留出顶部标题空间
    plt.show()

if __name__ == "__main__":
    main()