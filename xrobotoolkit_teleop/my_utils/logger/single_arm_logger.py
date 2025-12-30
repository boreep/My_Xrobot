import rclpy
from rclpy.node import Node
import message_filters
import threading
import numpy as np
import h5py
import os
import time
import yaml
import shutil
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
import sys

# 自动定位项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..','..'))
from xrobotoolkit_teleop.utils.path_utils import DATASET_PATH

# === 消息类型导入 ===
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from my_interfaces.msg import HeaderFloat32 
from rm_ros_interfaces.msg import Jointpos

class RecorderState(Enum):
    IDLE = 0        
    RECORDING = 1   

# ==========================================
#    ArmRecorder 类 (保持不变)
# ==========================================
class ArmRecorder:
    def __init__(self, node: Node, arm_side: str, topic_mapping: Dict[str, str], handlers: Dict[str, Dict]):
        self.node = node
        self.arm_side = arm_side
        self.buffer = [] 
        self.subs = []   
        self.keys = []   
        self.handlers = handlers
        self._lock = threading.Lock()
        self._latest_frame = None  

        for key, topic_suffix in topic_mapping.items():
            if key not in handlers: continue
            # 支持 {arm_side} 占位符
            if "{arm_side}" in topic_suffix:
                full_topic = topic_suffix.format(arm_side=arm_side)
            else:
                full_topic = f"{arm_side}/{topic_suffix}".replace("//", "/")
            
            msg_type = handlers[key]['type']
            sub = message_filters.Subscriber(node, msg_type, full_topic)
            self.subs.append(sub)
            self.keys.append(key)
            node.get_logger().info(f"[{arm_side}] Listening: {full_topic}")

        self.sync = message_filters.ApproximateTimeSynchronizer(self.subs, queue_size=50, slop=0.05)
        self.sync.registerCallback(self.sync_callback)

    def sync_callback(self, *msgs):
        frame_data = {}
        now = self.node.get_clock().now().nanoseconds / 1e9
        try:
            for i, msg in enumerate(msgs):
                key = self.keys[i]
                parser = self.handlers[key]['parser']
                frame_data[key] = parser(msg)
                if i == 0:
                    if hasattr(msg, 'header'):
                        frame_data["timestamp"] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    else:
                        frame_data["timestamp"] = now
            with self._lock:
                self._latest_frame = frame_data
        except Exception as e:
            self.node.get_logger().error(f"[{self.arm_side}] Parse Error: {e}")

    def sample_current_frame(self) -> Optional[Dict]:
        with self._lock:
            if self._latest_frame is None: return None
            return self._latest_frame.copy()

    def record_frame(self, data):
        self.buffer.append(data)
        if len(self.buffer) % 50 == 0:
            self.node.get_logger().info(f"[{self.arm_side}] Recorded frame {len(self.buffer)}")

    def clear(self):
        self.buffer = []

    def has_data(self):
        return len(self.buffer) > 0


# ==========================================
#        类：双臂主节点 (DualArmDataLogger)
# ==========================================
class DualArmDataLogger(Node):
    def __init__(self, config: Dict[str, Any]):
        
        self.config_cache = config # 缓存原始配置
        super().__init__('dual_arm_logger')
        
        base_dir = config.get('base_dir', 'robot_dataset')
        task_name = config.get('task_name', 'default_task')
        target_rate = config.get('sampling_rate', 20)
        topic_logic_map = config.get('topics', {})

        self.current_state = RecorderState.IDLE
        self.is_active = False
        self.episode_count = 0 
        
        # === 1. 构建三级保存路径 ===
        # base_dir / task_date / run_HHMMSS
        date_str = datetime.now().strftime("%Y%m%d")
        session_root = os.path.join(base_dir, f"{task_name}_{date_str}")
        
        run_time_str = datetime.now().strftime("%H%M%S")
        self.run_dir = os.path.join(session_root, f"run_{run_time_str}")
        
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir, exist_ok=True)
            
        self.get_logger().info(f"[Dual] Data Dir: {self.run_dir}")

        # === 2. 独立备份 Config 文件 ===
        config_backup_path = os.path.join(self.run_dir, f"{task_name}_config.yaml")
        with open(config_backup_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        self.get_logger().info(f"Config backed up to: {config_backup_path}")
        
        # === 3. 元数据文件路径初始化 ===
        self.metadata_path = os.path.join(self.run_dir, "dataset_metadata.yaml")
        self.recorded_episodes_info = [] 
        self.target_rate = target_rate
        
        # 初始化 metadata 文件
        self._init_metadata_file()

        # 4. 初始化处理器
        self.handlers = {}
        self._register_handlers()
        
        # 5. 启动录制器 (双臂实例)
        # 注意：这里分别实例化 left 和 right
        self.left_arm = ArmRecorder(self, "left_arm", topic_logic_map, self.handlers)
        self.right_arm = ArmRecorder(self, "right_arm", topic_logic_map, self.handlers)
        
        # 6. 启动定时器
        self.timer = self.create_timer(1.0 / self.target_rate, self.timer_callback)
        
        self.get_logger().info(f"[Dual] Ready. Rate: {target_rate}Hz")

    def _register_handlers(self):
        self.handlers = {
            "joint_cmd":   { "type": Jointpos, "parser": lambda msg: np.array(msg.joint, dtype=np.float32) },
            "ik_target":   { "type": PoseStamped, "parser": self._parse_pose },
            "gripper":     { "type": HeaderFloat32, "parser": lambda msg: np.array([msg.data], dtype=np.float32) },
            "joint_state": { "type": JointState, "parser": lambda msg: np.array(msg.position, dtype=np.float32) }
        }

    def _parse_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

    def timer_callback(self):
        # 只有在 RECORDING 且 Active 状态下才采样
        if self.current_state != RecorderState.RECORDING or not self.is_active:
            return

        # 采样数据
        left_data = self.left_arm.sample_current_frame()
        right_data = self.right_arm.sample_current_frame()

        # === 严格同步逻辑 ===
        # 必须左右臂都有数据才记录，保证对齐
        if left_data is not None and right_data is not None:
            self.left_arm.record_frame(left_data)
            self.right_arm.record_frame(right_data)
        else:
            # 只要有一个没数据，就等待，不录制
            self.get_logger().warn(f"[Dual] Waiting for BOTH arms data...", throttle_duration_sec=1)

    def start_episode(self):
        if self.current_state == RecorderState.IDLE:
            # 开始前清空两个 Buffer
            self.left_arm.clear()
            self.right_arm.clear()
            self.current_state = RecorderState.RECORDING
            self.get_logger().info(f"[Dual] >>> START RECORDING (Ep {self.episode_count}) <<<")

    def stop_episode(self):
        if self.current_state == RecorderState.RECORDING:
            self.current_state = RecorderState.IDLE
            self.get_logger().info(f"[Dual] >>> STOP & SAVE <<<")
            # 停止后立即保存数据并更新 Metadata
            self._save_to_hdf5()
            
            # 保存完释放内存
            self.left_arm.clear()
            self.right_arm.clear()
            
            self.episode_count += 1

    def update_active_status(self, is_active: bool):
        self.is_active = is_active
        if is_active:
            self.get_logger().info(f"[System] ACTIVATED")
        else:
            self.get_logger().info(f"[System] DEACTIVATED")

    # ==========================================
    #      核心：保存数据 H5 + 更新 YAML
    # ==========================================
    def _save_to_hdf5(self):
        # 检查两个臂是否都有数据
        if not self.left_arm.has_data() or not self.right_arm.has_data():
            self.get_logger().warn("Buffer empty in one or both arms, skipping save.")
            return

        # 1. 准备文件名
        filename_only = f"dual_ep{self.episode_count}_{datetime.now().strftime('%H%M%S')}.h5"
        full_filepath = os.path.join(self.run_dir, filename_only)
        
        try:
            # 2. 执行实际写入 (分组写入)
            with h5py.File(full_filepath, 'w') as f:
                # 左臂 Group
                g_left = f.create_group("left")
                self._write_group(g_left, self.left_arm.buffer, self.left_arm.keys, "left_arm")
                
                # 右臂 Group
                g_right = f.create_group("right")
                self._write_group(g_right, self.right_arm.buffer, self.right_arm.keys, "right_arm")
            
            self.get_logger().info(f"Successfully saved data to: {filename_only}")
            
            # 3. 计算统计信息 (因为是对齐的，取左臂长度即可)
            current_duration = self.left_arm.buffer[-1]['timestamp'] - self.left_arm.buffer[0]['timestamp']
            current_frames = len(self.left_arm.buffer)
            
            # 4. 内存中记录 Episode 信息
            ep_info = {
                "id": self.episode_count,
                "filename": filename_only,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frame_count": current_frames,
                "duration_sec": float(current_duration)
            }
            self.recorded_episodes_info.append(ep_info)

            # 5. 计算全局累计
            total_frames = sum(e['frame_count'] for e in self.recorded_episodes_info)
            total_duration = sum(e['duration_sec'] for e in self.recorded_episodes_info)

            # 6. 分析数据结构 (双臂分别分析，虽然理论上结构一样)
            # 使用 total_frames 覆盖第一维度，表示累计量
            structure_left = self._analyze_structure(self.left_arm.buffer, self.left_arm.keys, total_frames)
            structure_right = self._analyze_structure(self.right_arm.buffer, self.right_arm.keys, total_frames)
            
            combined_structure = {
                "left": structure_left,
                "right": structure_right
            }

            # 7. 更新 YAML 文件
            self._update_metadata_file(combined_structure, total_frames, total_duration)

        except Exception as e:
            self.get_logger().error(f"Save Failed: {e}")
            import traceback
            traceback.print_exc()

    def _write_group(self, group, buffer, keys, arm_name):
        """
        实际写入 HDF5 Group 的辅助函数
        """
        for key in keys:
            data = np.stack([frame[key] for frame in buffer])
            group.create_dataset(key, data=data, compression="gzip", compression_opts=4)
        
        timestamps = np.array([frame["timestamp"] for frame in buffer])
        group.create_dataset("timestamp", data=timestamps, compression="gzip")
        
        group.attrs['num_frames'] = len(buffer)
        group.attrs['arm_side'] = arm_name

    def _analyze_structure(self, buffer, keys, total_frames_override=None) -> Dict:
        """
        分析数据结构
        """
        structure = {}
        # 如果提供了总帧数，就用总帧数，否则用当前 buffer 长度
        N = total_frames_override if total_frames_override is not None else len(buffer)
        
        structure['timestamp'] = {
            "shape": [N],
            "dtype": "float64"
        }
        
        if len(buffer) == 0: return structure

        first_frame = buffer[0]
        for key in keys:
            val = first_frame[key]
            if isinstance(val, np.ndarray):
                full_shape = [N] + list(val.shape)
                dtype_str = str(val.dtype)
            else:
                full_shape = [N, 1]
                dtype_str = str(type(val))
            
            structure[key] = {
                "shape": full_shape,
                "dtype": dtype_str
            }
        return structure

    def _init_metadata_file(self):
        """
        初次创建 metadata 文件
        """
        init_data = {
            "meta_info": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task_name": self.config_cache.get('task_name'),
                "node_name": self.get_name(),
                "mode": "DualArm", # 标记为双臂模式
                "config_copy": self.config_cache
            },
            "dataset_structure": {},
            "episodes": []
        }
        with open(self.metadata_path, 'w') as f:
            yaml.dump(init_data, f, default_flow_style=False, sort_keys=False)

    def _update_metadata_file(self, structure_info, total_frames, total_duration):
        """
        更新 metadata 文件
        """
        data = {
            "meta_info": {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_episodes": len(self.recorded_episodes_info),
                "total_frames": total_frames,
                "total_duration_sec": total_duration,
                
                # === 补全静态信息 ===
                "task_name": self.config_cache.get('task_name'),
                "mode": "DualArm",
                "sampling_rate": self.target_rate,          
                "topics": self.config_cache.get('topics')   
            },
            "dataset_structure": structure_info, # 这里包含 left 和 right 两个键
            "episodes": self.recorded_episodes_info
        }
        
        with open(self.metadata_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
        self.get_logger().info(f"Metadata updated (Total: {total_frames} frames)")

# ==========================================
#           Config 加载与主函数
# ==========================================
def load_config(config_path="config/recorder_config.yaml"):
    full_path = os.path.join(DATASET_PATH, config_path)
    
    if not os.path.exists(full_path):
        print(f"[Error] Config file not found: {full_path}")
        sys.exit(1)
        
    with open(full_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            print(f"[Config] Loaded from {full_path}")
            return config
        except yaml.YAMLError as e:
            print(f"[Error] YAML parse error: {e}")
            sys.exit(1)

def main(args=None):
    rclpy.init(args=args)

    # 1. 加载配置
    config = load_config("config/recorder_config.yaml")

    # 2. 启动节点 (DualArmDataLogger)
    logger_node = DualArmDataLogger(config)

    # 3. 线程与交互
    def ros_thread_entry():
        try:
            rclpy.spin(logger_node)
        except Exception:
            pass
    t = threading.Thread(target=ros_thread_entry, daemon=True)
    t.start()

    print(f"\n=== 配置驱动 [双臂] 录制控制台 ===")
    print(f"当前任务: {config.get('task_name')}")
    print(" [B] 开始/停止录制")
    print(" [A] 切换 Active 状态")
    print(" [Q] 退出")

    sim_b_pressed = False 
    sim_active = False 

    try:
        while rclpy.ok():
            cmd = input(f"[DualArm] Active:{sim_active} > ").strip().lower()
            if cmd == 'b':
                sim_b_pressed = not sim_b_pressed
                if sim_b_pressed: logger_node.start_episode()
                else: logger_node.stop_episode()
            elif cmd == 'a':
                sim_active = not sim_active
                logger_node.update_active_status(sim_active)
            elif cmd == 'q':
                if logger_node.current_state != RecorderState.IDLE:
                    logger_node.stop_episode()
                break
    except KeyboardInterrupt:
        pass
    finally:
        logger_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()