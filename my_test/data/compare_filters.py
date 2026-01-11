import time
import os  # 新增: 用于创建目录
import numpy as np
import matplotlib.pyplot as plt
from xrobotoolkit_teleop.common.filter.filtered_xr_client import FilteredXrClient
from xrobotoolkit_teleop.utils.terminalcolor import TerminalColor

def main():
    # =========================
    # 1. 初始化带参数的 Client
    # =========================
    client = FilteredXrClient(
        # 位置
        pos_min_cutoff=1.0, pos_beta=0.2, 
        # 姿态
        rot_min_cutoff=1.0, rot_beta=0.1,
        # 速度
        vel_min_cutoff=1.0, vel_beta=0.3
    )

    # =========================
    # 数据容器 (用于绘图)
    # =========================
    timestamps = []
    pos_raw = []
    pos_filt = []
    vel_raw = []
    vel_filt = []

    # =========================
    # 数据容器 (用于保存文件 - 全量数据)
    # =========================
    full_data_log = []

    print(f"{TerminalColor.OKBLUE}开始采集对比数据 (50Hz)...")
    print(f"{TerminalColor.WARNING}请移动手柄，按 Ctrl+C 停止。停止后将自动生成图表并保存 CSV 文件...{TerminalColor.ENDC}\n")

    freq = 50.0
    dt = 1.0 / freq
    start_time = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            current_time = loop_start - start_time

            # -------------------------------------------------
            # 2. 获取数据
            # -------------------------------------------------
            # 获取滤波后的数据 (7维Pose, 3维Vel)
            f_pose = client.get_pose_by_name("right_controller") 
            f_vel = client.get_velocity_by_name("right_controller")

            # 获取原始数据
            r_pose = client.last_raw_pose.get("right_controller")
            r_vel = client.last_raw_vel.get("right_controller")

            # -------------------------------------------------
            # 3. 数据存储 (确保数据有效)
            # -------------------------------------------------
            if r_pose is not None and r_vel is not None:
                # --- A. 用于绘图的数据 ---
                timestamps.append(current_time)
                pos_raw.append(r_pose[:3])
                pos_filt.append(f_pose[:3])
                vel_raw.append(r_vel[:3])
                vel_filt.append(f_vel[:3])

                # --- B. 用于保存的数据 ---
                data_row = np.concatenate([
                    [current_time],
                    r_pose, r_vel[:3],
                    f_pose, f_vel[:3]
                ])
                full_data_log.append(data_row)

                # -------------------------------------------------
                # [修改] 实时打印状态：加入 Pose X 显示
                # -------------------------------------------------
                if len(timestamps) % 10 == 0:
                    # 这里 f_pose[0] 就是 X 轴位置
                    print(f"\r采集点数: {len(timestamps)} | Time: {current_time:.2f}s | Pose X: {f_pose[0]:.4f}", end="")

            # -------------------------------------------------
            # 4. 频率控制
            # -------------------------------------------------
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print(f"\n\n{TerminalColor.OKGREEN}采集结束，正在处理数据...{TerminalColor.ENDC}")
    finally:
        client.close()

    if len(timestamps) < 20:
        print("数据过少，取消保存和绘图。")
        return

    # =========================
    # 5. 保存数据到 CSV
    # =========================
    filename = "my_test/data/captured_motion_data.csv"
    print(f"\n正在保存数据到 {filename} ...")

    # [新增] 自动创建目录，防止因为文件夹不存在而报错
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 定义 CSV 表头
    header = (
        "Time,"
        "Raw_Px,Raw_Py,Raw_Pz,Raw_Qx,Raw_Qy,Raw_Qz,Raw_Qw,Raw_Vx,Raw_Vy,Raw_Vz,"
        "Filt_Px,Filt_Py,Filt_Pz,Filt_Qx,Filt_Qy,Filt_Qz,Filt_Qw,Filt_Vx,Filt_Vy,Filt_Vz"
    )
    
    try:
        np.savetxt(filename, np.array(full_data_log), delimiter=",", header=header, comments='')
        print(f"{TerminalColor.OKBLUE}数据保存成功！共 {len(full_data_log)} 帧。{TerminalColor.ENDC}")
    except Exception as e:
        print(f"{TerminalColor.FAIL}保存文件失败: {e}{TerminalColor.ENDC}")

    # =========================
    # 6. 数据处理与绘图
    # =========================
    print("正在生成对比图表...")
    ts = np.array(timestamps)
    p_raw = np.array(pos_raw)
    p_filt = np.array(pos_filt)
    v_raw = np.array(vel_raw)
    v_filt = np.array(vel_filt)

    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    labels = ['X Axis', 'Y Axis', 'Z Axis']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    plt.suptitle("One Euro Filter Comparison (Data Saved to CSV)", fontsize=16)

    for i in range(3):
        # 左列：位置
        ax_pos = axs[i, 0]
        ax_pos.plot(ts, p_raw[:, i], color='gray', alpha=0.5, linestyle='--', linewidth=1, label='Raw Input')
        ax_pos.plot(ts, p_filt[:, i], color=colors[i], linewidth=2, label='Filtered Output')
        ax_pos.set_ylabel(f'Pos {labels[i]} (m)', fontweight='bold')
        ax_pos.grid(True, which='both', linestyle='--', alpha=0.5)
        if i == 0: 
            ax_pos.set_title("Position Tracking", fontsize=14)
            ax_pos.legend(loc='upper right')

        # 右列：速度
        ax_vel = axs[i, 1]
        ax_vel.plot(ts, v_raw[:, i], color='gray', alpha=0.4, linestyle='--', linewidth=1, label='Raw Input')
        ax_vel.plot(ts, v_filt[:, i], color=colors[i], linewidth=1.5, label='Filtered Output')
        ax_vel.set_ylabel(f'Vel {labels[i]} (m/s)', fontweight='bold')
        ax_vel.grid(True, which='both', linestyle='--', alpha=0.5)
        if i == 0: 
            ax_vel.set_title("Velocity Tracking", fontsize=14)
            ax_vel.legend(loc='upper right')

    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 1].set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()