import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def main():
    filename = "my_test/data/captured_motion_data_clean.csv"
    
    # 1. 检查文件
    if not os.path.exists(filename):
        print(f"错误: 找不到文件 '{filename}'")
        return

    print(f"正在读取 {filename} ...")
    
    try:
        # 2. 读取数据 (skiprows=1 跳过表头)
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
        
        # [修改] 不进行任何裁剪，直接使用全部数据
        print(f"成功读取 {len(data)} 帧数据。")

    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 3. 提取坐标数据
    # Col 1-3: Raw Px, Py, Pz
    # Col 11-13: Filt Px, Py, Pz
    
    raw_x = data[:, 1]
    raw_y = data[:, 2]
    raw_z = data[:, 3]
    
    filt_x = data[:, 11]
    filt_y = data[:, 12]
    filt_z = data[:, 13]

    # 4. 创建 3D 绘图
    print("正在生成 3D 轨迹图 (全量数据)...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- A. 绘制原始轨迹 (Raw) ---
    # [修改] 调整了线型为虚线，增加了线宽和不透明度，并加深了颜色，使其更明显
    ax.plot(raw_x, raw_y, raw_z, 
            label='Raw Trajectory', 
            color='#555555', linestyle='--', linewidth=2, alpha=0.8)

    # --- B. 绘制滤波轨迹 (Filtered) ---
    ax.plot(filt_x, filt_y, filt_z, 
            label='Filtered Trajectory', 
            color='#ff7f0e', linewidth=2, alpha=0.9)

    # --- C. 标记起点和终点 ---
    # 起点 (Start)
    ax.scatter(filt_x[0], filt_y[0], filt_z[0], 
               color='green', s=100, marker='o', label='Start')
    
    # 终点 (End)
    ax.scatter(filt_x[-1], filt_y[-1], filt_z[-1], 
               color='red', s=100, marker='^', label='End')

    # 5. 设置标签
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_zlabel('Z Position (m)', fontweight='bold')
    ax.set_title(f"3D Motion Trajectory (All Frames)", fontsize=14)
    
    ax.legend()
    
    # 尝试设置等比例 (根据 Matplotlib 版本可能效果不同)
    try:
        ax.set_box_aspect([1,1,1])
    except:
        pass

    print("图表已生成。")
    plt.show()

if __name__ == "__main__":
    main()