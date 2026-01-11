import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    filename = "my_test/data/captured_motion_data.csv"
    
    # 1. 检查文件是否存在
    if not os.path.exists(filename):
        print(f"错误: 找不到文件 '{filename}'")
        print("请确保你已经运行了采集脚本并成功保存了数据。")
        return

    print(f"正在读取 {filename} ...")
    
    try:
        # 2. 读取 CSV 数据
        # delimiter=",": 指定逗号分隔
        # comments="#": 跳过表头 (如果表头带#) 或者 skiprows=1
        # 这里使用 skiprows=1 跳过第一行表头
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 3. 数据切片与提取
    # 根据之前的保存逻辑，列索引如下：
    # [0]: Time
    # [1-3]: Raw Pos (X,Y,Z)
    # [4-7]: Raw Rot (QX,QY,QZ,QW)
    # [8-10]: Raw Vel (X,Y,Z)
    # [11-13]: Filtered Pos (X,Y,Z)
    # [14-17]: Filtered Rot (QX,QY,QZ,QW)
    # [18-20]: Filtered Vel (X,Y,Z)

    ts = data[:, 0]
    
    # 提取位置 (X, Y, Z)
    pos_raw = data[:, 1:4]
    pos_filt = data[:, 11:14]
    
    # 提取速度 (X, Y, Z)
    vel_raw = data[:, 8:11]
    vel_filt = data[:, 18:21]

    # 4. 绘图配置
    print("正在生成图表...")
    plt.style.use('seaborn-v0_8-darkgrid') # 如果报错可以改成 'ggplot' 或删掉这行
    
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    
    labels = ['X Axis', 'Y Axis', 'Z Axis']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4'] # 橙, 绿, 蓝

    plt.suptitle(f"Motion Data Replay: {filename}", fontsize=16)

    for i in range(3):
        # --- 左列：位置对比 ---
        ax_pos = axs[i, 0]
        # 原始数据 (灰色虚线)
        ax_pos.plot(ts, pos_raw[:, i], color='gray', alpha=0.5, linestyle='--', linewidth=1, label='Raw Input')
        # 滤波数据 (彩色实线)
        ax_pos.plot(ts, pos_filt[:, i], color=colors[i], linewidth=2, label='Filtered Output')
        
        ax_pos.set_ylabel(f'Pos {labels[i]} (m)', fontweight='bold')
        ax_pos.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # 只在第一行显示标题和图例
        if i == 0: 
            ax_pos.set_title("Position Tracking", fontsize=14)
            ax_pos.legend(loc='upper right')

        # --- 右列：速度对比 ---
        ax_vel = axs[i, 1]
        # 原始数据
        ax_vel.plot(ts, vel_raw[:, i], color='gray', alpha=0.4, linestyle='--', linewidth=1, label='Raw Input')
        # 滤波数据
        ax_vel.plot(ts, vel_filt[:, i], color=colors[i], linewidth=1.5, label='Filtered Output')
        
        ax_vel.set_ylabel(f'Vel {labels[i]} (m/s)', fontweight='bold')
        ax_vel.grid(True, which='both', linestyle='--', alpha=0.5)
        
        if i == 0: 
            ax_vel.set_title("Velocity Tracking", fontsize=14)
            ax_vel.legend(loc='upper right')

    # 设置底部 X 轴标签
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 1].set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print("图表已显示。")
    plt.show()

if __name__ == "__main__":
    main()