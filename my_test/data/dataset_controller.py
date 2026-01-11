import numpy as np
import os
import meshcat.transformations as tf
from xrobotoolkit_teleop.my_utils.allrobot_teleop_controller import AllRobotTeleopController
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose

class DatasetController(AllRobotTeleopController):
    def __init__(
        self,
        csv_path: str = "my_test/data/captured_motion_data_clean.csv",
        **kwargs
    ):
        """
        初始化数据集回放控制器
        :param csv_path: 动作捕捉数据CSV文件路径
        :param kwargs: 传递给父类的其他参数
        """
        # 调用父类构造函数 (传递所有通用参数)
        super().__init__(**kwargs)
        
        self.csv_path = csv_path
        
        # --- 1. 加载 CSV 数据 ---
        # 检查文件是否存在
        if not os.path.exists(self.csv_path):
             # 尝试在当前工作目录查找
             if os.path.exists(os.path.join(os.getcwd(), self.csv_path)):
                 self.csv_path = os.path.join(os.getcwd(), self.csv_path)
             else:
                 raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        print(f"[DatasetController] Loading motion data from {self.csv_path}...")
        try:
            # 读取数据，跳过表头
            data = np.loadtxt(self.csv_path, delimiter=",", skiprows=1)
            
            # 根据提供的CSV格式提取 Filtered Pose 数据
            # Col 11-13: Filt_Px, Filt_Py, Filt_Pz
            # Col 14-17: Filt_Qx, Filt_Qy, Filt_Qz, Filt_Qw
            self.csv_positions = data[:, 11:14]      # [N, 3]
            self.csv_quaternions = data[:, 14:18]    # [N, 4] -> [x, y, z, w]
            
            self.total_frames = len(data)
            print(f"[DatasetController] Loaded {self.total_frames} frames.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV data: {e}")

        # --- 2. 播放控制变量 ---
        self.current_frame_idx = 0
        self.playback_direction = 1  # 1: 正放, -1: 倒放
        
    def _update_ik(self):
        """
        重写IK更新逻辑：
        从CSV顺序读取位姿数据替代XR手柄输入，并执行逆运动学求解。
        """
        # --- 1. 更新播放帧索引 (乒乓循环逻辑) ---
        self.current_frame_idx += self.playback_direction
        
        # 边界检查与反转
        if self.current_frame_idx >= self.total_frames:
            self.current_frame_idx = self.total_frames - 2
            self.playback_direction = -1
            print("[DatasetController] Reached end, reversing playback.")
        elif self.current_frame_idx < 0:
            self.current_frame_idx = 1
            self.playback_direction = 1
            print("[DatasetController] Reached start, forward playback.")
            
        # --- 2. 获取当前帧的Pose ---
        # 获取位置 [x, y, z] 和 四元数 [x, y, z, w]
        current_pos = self.csv_positions[self.current_frame_idx]
        current_quat = self.csv_quaternions[self.current_frame_idx]
        
        # 构建 _process_xr_pose 所需的 xr_pose 格式
        # 格式为: [x, y, z, qx, qy, qz, qw]
        # (注意: BaseController._process_xr_pose 内部解析时，index 6是w，index 3,4,5是x,y,z)
        xr_pose = np.concatenate([current_pos, current_quat])

        # --- 3. 处理每个操作器的控制逻辑 ---
        for src_name, config in self.manipulator_config.items():
            # 强制设置为激活状态，无需按下手柄扳机
            self.active[src_name] = True
            
            if self.active[src_name]:
                # 如果是刚激活（或第一帧），初始化参考位姿
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} is activated (Dataset Start).")
                    self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name] = self._get_link_pose(config["link_name"])

                # --- 核心修改：使用CSV构建的 xr_pose 替代 self.xr_client.get_pose_by_name ---
                # 计算相对于初始参考系的 delta
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)
                
                # --- 后续逻辑保持不变 ---
                # 根据控制模式更新目标任务
                if self.effector_control_mode[src_name] == "position":
                    target_xyz = self.ref_ee_xyz[src_name] + delta_xyz
                    self.effector_task[src_name].target_world = target_xyz
                    
                    target_quat = self.ref_ee_quat[src_name]
                else:
                    target_xyz, target_quat = apply_delta_pose(
                        self.ref_ee_xyz[src_name],
                        self.ref_ee_quat[src_name],
                        delta_xyz,
                        delta_rot,
                    )
                    target_pose = tf.quaternion_matrix(target_quat)
                    target_pose[:3, 3] = target_xyz
                    self.effector_task[src_name].T_world_frame = target_pose
                    
                self.ik_targets[src_name] = {
                    "pos": target_xyz.copy(),
                    "quat": np.array(target_quat, copy=True),
                }
            else: 
                # 理论上这里的 else 分支在强制 active=True 后不会执行，保留以防万一
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} is deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_controller_xyz[src_name] = None

        # 处理运动追踪器数据 (如果需要，也可以在这里用CSV数据Mock)
        # self._update_motion_tracker_tasks()

        # --- 4. 求解逆运动学 ---
        try:
            self.solver.solve(True)
        except RuntimeError as e:
            print(f"[update_ik] IK solver failed: {e}")