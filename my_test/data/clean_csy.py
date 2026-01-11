import pandas as pd
import os

input_file = "my_test/data/captured_motion_data.csv"
output_file = "my_test/data/captured_motion_data_clean.csv"
cut_frames = 50

if os.path.exists(input_file):
    # 1. 读取数据
    df = pd.read_csv(input_file)
    
    # 2. 检查长度
    if len(df) > cut_frames:
        # 3. 删除前 300 行 (iloc 是按行索引切片)
        df_new = df.iloc[cut_frames:].copy()
        
        # 4. [推荐] 重置时间戳：让第一行的时间变成 0.0
        # 假设第一列是 'Time'
        start_time = df_new.iloc[0, 0]
        df_new.iloc[:, 0] = df_new.iloc[:, 0] - start_time
        
        # 5. 保存 (index=False 表示不保存行号)
        df_new.to_csv(output_file, index=False)
        print(f"成功！已删除前 {cut_frames} 帧。")
        print(f"新文件已保存为: {output_file}")
    else:
        print("数据太短，没法切。")
else:
    print(f"找不到文件: {input_file}")