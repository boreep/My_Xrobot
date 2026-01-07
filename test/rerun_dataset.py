import h5py
import numpy as np
import rerun as rr
import os

# ================= 配置 =================
H5_PATH = "dataset/default_task_20260107/run_150106/episode_0_150205.h5" 
# =======================================

def main():
    # 0. 检查文件是否存在
    if not os.path.exists(H5_PATH):
        print(f"❌ 文件不存在: {H5_PATH}")
        return

    # 1. 初始化 Rerun (spawn=True 会自动弹出浏览器/窗口)
    rr.init("Robot_Replay_Stable", spawn=True)

    print(f"正在读取: {H5_PATH} ...")
    
    with h5py.File(H5_PATH, 'r') as f:
        # === A. 读取数据 ===

        # 1. 相机数据
        images = f['camera']['image'][:]# type: ignore
        points = f['camera']['point'][:]# type: ignore
        cam_timestamps = f['camera']['timestamp'][:]# type: ignore
        
        # 2. 机械臂数据
        joint_cmds = f['left_arm']['joint_cmd'][:]    # shape: (N, 7)# type: ignore
        joint_states = f['left_arm']['joint_state'][:] # shape: (N, 7)# type: ignore
        
        total_frames = len(cam_timestamps)# type: ignore
        print(f"✅ 加载完成，共 {total_frames} 帧。")
# ==========================================
        # === A+. [新增] 数据集频率分析逻辑 ===
        # ==========================================
        if total_frames > 1:
            # 计算相邻时间戳的差值 (dt)
            intervals = np.diff(cam_timestamps) # type: ignore
            
            # 统计指标
            avg_dt = np.mean(intervals)       # 平均间隔 (秒)
            std_dt = np.std(intervals)        # 标准差 (秒，反映抖动程度)
            max_dt = np.max(intervals)        # 最大间隔
            min_dt = np.min(intervals)        # 最小间隔
            
            # 计算频率 (Hz)
            freq = 1.0 / avg_dt if avg_dt > 0 else 0
            
            print("-" * 40)
            print(f"📊 数据集时间轴分析报告:")
            print(f"  - 总录制时长 : {cam_timestamps[-1] - cam_timestamps[0]:.2f} 秒") # type: ignore
            print(f"  - 平均频率   : {freq:.2f} Hz")
            print(f"  - 平均间隔   : {avg_dt*1000:.2f} ms ({avg_dt:.6f} s)")
            print(f"  - 间隔抖动(std): {std_dt*1000:.2f} ms")
            print(f"  - 最大间隔   : {max_dt:.6f} s")
            print(f"  - 最小间隔   : {min_dt:.6f} s")
            print("-" * 40)
        else:
            print("⚠️ 数据帧不足，无法计算频率。")
        # ==========================================


        # === B. 逐帧发送数据 ===
        for i in range(total_frames):
            # 设定时间轴
            # 1. 设置整数序列 (对应原来的 set_time_sequence)
            rr.set_time("frame_idx", sequence=i)

            # 2. 设置时间戳 (对应原来的 set_time_seconds)
            # 注意：timestamp 参数接受秒数 (float)
            rr.set_time("log_time", timestamp=cam_timestamps[i])# type: ignore
            # ------------------------------------------------
            # 1. 图像 (Image)
            # ------------------------------------------------
            rr.log("camera/image", rr.Image(images[i]))# type: ignore

            # ------------------------------------------------
            # 2. 点云 (Point Cloud)
            # ------------------------------------------------
            frame_p = points[i]# type: ignore
            xyz = frame_p[:, :3]# type: ignore
            
            # 颜色解析逻辑
            if frame_p.shape[1] == 6:# type: ignore
                # 假设格式为 [x, y, z, r, g, b]
                rgb = frame_p[:, 3:6].astype(np.uint8)# type: ignore
            else:
                # 假设格式为 [x, y, z, packed_rgb]
                packed = frame_p[:, 3].copy()# type: ignore
                rgb = packed.view(np.uint8).reshape(-1, 4)[:, [2, 1, 0]]

            rr.log(
                "camera/point_cloud", 
                rr.Points3D(xyz, colors=rgb, radii=0.01)# type: ignore
            )

            # ------------------------------------------------
            # 3. 关节数据 (关键修改：分离 CMD 和 STATE 的根路径)
            # ------------------------------------------------
            # 机械臂关节数
            num_joints = joint_cmds.shape[1] # type: ignore
            
            for j in range(num_joints):
                # 组1：命令数据 (CMD)
                # 将其放在 "plot_cmd" 目录下，Rerun 会为此创建一个单独的图表
                rr.log(
                    f"plot_cmd/joint_{j}", 
                    rr.Scalars(joint_cmds[i, j])# type: ignore
                )
                
                # 组2：状态数据 (STATE)
                # 将其放在 "plot_state" 目录下，Rerun 会为此创建另一个图表
                rr.log(
                    f"plot_state/joint_{j}", 
                    rr.Scalars(joint_states[i, j]) # type: ignore
                )

            # ------------------------------------------------
            # 进度打印
            if i % 50 == 0:
                print(f"已处理: {i}/{total_frames}")

    print("🎉 完成！Rerun 窗口已弹出。")
    print("💡 提示：如果图表依然混在一起，请点击 Rerun 界面顶部的 'Reset Layout' 或手动拖拽 'plot_cmd' 和 'plot_state' 标题栏进行分屏。")

if __name__ == "__main__":
    main()