import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('my_test/data/captured_motion_data_clean1.csv')

# 计算微分
calc_vx = np.gradient(df['Filt_Px'], df['Time'])
calc_vy = np.gradient(df['Filt_Py'], df['Time'])
calc_vz = np.gradient(df['Filt_Pz'], df['Time'])

# 绘图
fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)

# 定义一个辅助函数来统一 Y 轴范围
def set_common_ylim(ax1, ax2, data1, data2):
    ymin = min(np.min(data1), np.min(data2))
    ymax = max(np.max(data1), np.max(data2))
    margin = (ymax - ymin) * 0.05  # 增加 5% 的边距
    ax1.set_ylim(ymin - margin, ymax + margin)
    ax2.set_ylim(ymin - margin, ymax + margin)

# Row 1: X-axis
axes[0, 0].plot(df['Time'], calc_vx, label='Calculated Vx', color='blue')
axes[0, 0].set_title('Calculated Vx (d(Filt_Px)/dt)')
axes[0, 0].set_ylabel('Velocity X (m/s)')
axes[0, 0].grid(True)
axes[0, 0].legend(loc='upper right')

axes[0, 1].plot(df['Time'], df['Filt_Vx'], label='Filt_Vx (CSV)', color='green')
axes[0, 1].set_title('Filt_Vx (from CSV)')
axes[0, 1].set_ylabel('Velocity X (m/s)')
axes[0, 1].grid(True)
axes[0, 1].legend(loc='upper right')

# 统一 X 方向的 Y 轴
set_common_ylim(axes[0, 0], axes[0, 1], calc_vx, df['Filt_Vx'])

# Row 2: Y-axis
axes[1, 0].plot(df['Time'], calc_vy, label='Calculated Vy', color='blue')
axes[1, 0].set_title('Calculated Vy (d(Filt_Py)/dt)')
axes[1, 0].set_ylabel('Velocity Y (m/s)')
axes[1, 0].grid(True)
axes[1, 0].legend(loc='upper right')

axes[1, 1].plot(df['Time'], df['Filt_Vy'], label='Filt_Vy (CSV)', color='green')
axes[1, 1].set_title('Filt_Vy (from CSV)')
axes[1, 1].set_ylabel('Velocity Y (m/s)')
axes[1, 1].grid(True)
axes[1, 1].legend(loc='upper right')

# 统一 Y 方向的 Y 轴
set_common_ylim(axes[1, 0], axes[1, 1], calc_vy, df['Filt_Vy'])

# Row 3: Z-axis
axes[2, 0].plot(df['Time'], calc_vz, label='Calculated Vz', color='blue')
axes[2, 0].set_title('Calculated Vz (d(Filt_Pz)/dt)')
axes[2, 0].set_ylabel('Velocity Z (m/s)')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].grid(True)
axes[2, 0].legend(loc='upper right')

axes[2, 1].plot(df['Time'], df['Filt_Vz'], label='Filt_Vz (CSV)', color='green')
axes[2, 1].set_title('Filt_Vz (from CSV)')
axes[2, 1].set_ylabel('Velocity Z (m/s)')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].grid(True)
axes[2, 1].legend(loc='upper right')

# 统一 Z 方向的 Y 轴
set_common_ylim(axes[2, 0], axes[2, 1], calc_vz, df['Filt_Vz'])

plt.tight_layout()
plt.savefig('velocity_comparison_grid_shared_y.png')