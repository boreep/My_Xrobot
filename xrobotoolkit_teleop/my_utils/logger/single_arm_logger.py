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
#           积木块：单臂录制器 (核心逻辑)
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
        # === [新增 1] 初始化版本计数器 ===
        self.frame_version = 0

        for key, topic_suffix in topic_mapping.items():
            if key not in handlers: continue
            
            # 自动替换占位符
            if "{arm_side}" in topic_suffix:
                full_topic = topic_suffix.format(arm_side=arm_side)
            else:
                full_topic = f"{arm_side}/{topic_suffix}".replace("//", "/")
            
            msg_type = handlers[key]['type']
            sub = message_filters.Subscriber(node, msg_type, full_topic)
            self.subs.append(sub)
            self.keys.append(key)
            node.get_logger().info(f"[{arm_side}] Listening: {full_topic}")

        # 使用 ApproximateTimeSynchronizer 确保同一只手臂的数据对齐
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
                # 使用第一个消息的时间戳作为该帧时间
                if i == 0:
                    if hasattr(msg, 'header'):
                        frame_data["timestamp"] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    else:
                        frame_data["timestamp"] = now
            
            with self._lock:
                self._latest_frame = frame_data
                # === [新增 2] 每次成功同步一帧新数据，版本号 +1 ===
                self.frame_version += 1
        except Exception as e:
            self.node.get_logger().error(f"[{self.arm_side}] Parse Error: {e}")

    # === [修改] 返回数据时，同时也返回版本号 ===
    def sample_current_frame(self):
        with self._lock:
            if self._latest_frame is None: 
                return None, -1  # 未初始化时，版本号为 -1
            return self._latest_frame.copy(), self.frame_version

    def record_frame(self, data):
        self.buffer.append(data)
        if len(self.buffer) % 50 == 0:
            self.node.get_logger().info(f"[{self.arm_side}] Buffer: {len(self.buffer)}")

    def clear(self):
        self.buffer = []

    def has_data(self):
        return len(self.buffer) > 0


# ==========================================
#           主节点：单臂数据记录器
# ==========================================
class SingleArmDataLogger(Node):
    def __init__(self, config: Dict[str, Any]):
        super().__init__('single_arm_logger')
        
        self.config_cache = config
        
        # === [新增 3] 记录上一次保存的版本号 ===
        self.last_ver = -1
        
        # 配置参数读取
        base_dir = config.get('base_dir', 'robot_dataset')
        task_name = config.get('task_name', 'default_task')
        target_rate = config.get('sampling_rate', 20)
        topic_logic_map = config.get('topics', {})
        # 默认使用左臂，也可在 config 中指定 arm_side
        self.target_arm = config.get('arm_side', 'right_arm') 
        self.target_rate = target_rate

        self.current_state = RecorderState.IDLE
        self.is_active = False
        self.episode_count = 0 
        
        # === 1. 构建目录结构 ===
        date_str = datetime.now().strftime("%Y%m%d")
        session_root = os.path.join(base_dir, f"{task_name}_{date_str}")
        
        run_time_str = datetime.now().strftime("%H%M%S")
        self.run_dir = os.path.join(session_root, f"run_{run_time_str}")
        
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir, exist_ok=True)
        self.get_logger().info(f"[System] Data Dir: {self.run_dir}")

        # === 2. 备份 Config ===
        config_backup_path = os.path.join(self.run_dir, f"{task_name}_config.yaml")
        with open(config_backup_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # === 3. 初始化 Metadata ===
        self.metadata_path = os.path.join(self.run_dir, "dataset_metadata.yaml")
        self.recorded_episodes_info = [] 
        self._init_metadata_file()

        # === 4. 初始化 单臂 Recorder ===
        self.handlers = {}
        self._register_handlers()
        
        # 实例化单臂
        self.arm_recorder = ArmRecorder(self, self.target_arm, topic_logic_map, self.handlers)

        # === 5. 启动核心 Timer ===
        self.timer = self.create_timer(1.0 / self.target_rate, self.timer_callback)
        self.get_logger().info(f"[System] Ready. Single Arm ({self.target_arm}) Sampling Rate: {target_rate}Hz")

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
        # 1. 基础状态检查
        if self.current_state != RecorderState.RECORDING or not self.is_active:
            return

        # 2. 获取数据和版本号
        data, current_ver = self.arm_recorder.sample_current_frame()

        # === [逻辑 A] 初始启动门槛 ===
        if data is None:
            self.get_logger().warn("Waiting for initial data...", throttle_duration_sec=1)
            return

        # === [逻辑 B] 智能去重/自动暂停逻辑 ===
        # 只有当 current_ver (当前版本) 大于 last_ver (上次保存的版本) 时，才说明有新数据
        if current_ver > self.last_ver:
            self.arm_recorder.record_frame(data)
            self.last_ver = current_ver # 更新已保存的版本号
        else:
            # 数据未更新，跳过写入 (Pause)
            # 可以在这里打印 debug 信息，但建议保持静默以免刷屏
            pass 

    def start_episode(self):
        if self.current_state == RecorderState.IDLE:
            self.arm_recorder.clear()
            # 重置版本记忆，确保下一轮录制从头开始
            self.last_ver = -1
            self.current_state = RecorderState.RECORDING
            self.get_logger().info(f">>> START RECORDING (Ep {self.episode_count}) <<<")

    def stop_episode(self):
        if self.current_state == RecorderState.RECORDING:
            self.current_state = RecorderState.IDLE
            self.get_logger().info(">>> STOP & SAVE <<<")
            self._save_to_hdf5()
            self.episode_count += 1

    def update_active_status(self, is_active: bool):
        self.is_active = is_active
        status_str = "ACTIVATED" if is_active else "DEACTIVATED"
        self.get_logger().info(f"[System] {status_str}")

    # ==========================================
    #      核心：单臂 HDF5 保存与元数据更新
    # ==========================================
    def _save_to_hdf5(self):
        if not self.arm_recorder.has_data():
            self.get_logger().warn("Buffer empty, skipping save.")
            return

        filename_only = f"ep{self.episode_count}_{datetime.now().strftime('%H%M%S')}.h5"
        full_filepath = os.path.join(self.run_dir, filename_only)
        
        try:
            # 1. 保存 HDF5 (扁平结构，保存到 Root)
            with h5py.File(full_filepath, 'w') as f:
                self._write_to_root(f, self.arm_recorder.buffer, self.arm_recorder.keys, self.target_arm)
            
            self.get_logger().info(f"Saved: {filename_only}")
            
            # 2. 统计信息
            buffer = self.arm_recorder.buffer
            frame_count = len(buffer)
            start_t = buffer[0]['timestamp']
            end_t = buffer[-1]['timestamp']
            duration = end_t - start_t

            ep_info = {
                "id": self.episode_count,
                "filename": filename_only,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frame_count": frame_count,
                "duration_sec": float(duration)
            }
            self.recorded_episodes_info.append(ep_info)

            # 3. 计算全局累计
            total_frames = sum(e['frame_count'] for e in self.recorded_episodes_info)
            total_duration = sum(e['duration_sec'] for e in self.recorded_episodes_info)

            # 4. 分析结构
            structure = self._analyze_structure(buffer, self.arm_recorder.keys, total_frames)
            
            # 5. 更新 YAML
            self._update_metadata_file(structure, total_frames, total_duration)

        except Exception as e:
            self.get_logger().error(f"Save Failed: {e}")
            import traceback
            traceback.print_exc()

    def _write_to_root(self, root, buffer, keys, arm_name):
        n_frames = len(buffer)
        for key in keys:
            data = np.stack([frame[key] for frame in buffer])
            # 单臂模式下，直接存在根目录，例如 /joint_cmd
            root.create_dataset(key, data=data, compression="gzip", compression_opts=4)
        
        timestamps = np.array([frame["timestamp"] for frame in buffer])
        root.create_dataset("timestamp", data=timestamps, compression="gzip")
        
        root.attrs['num_frames'] = n_frames
        root.attrs['arm_side'] = arm_name

    def _analyze_structure(self, buffer, keys, total_frames_override=None) -> Dict:
        """分析 Buffer 结构"""
        structure = {}
        N = total_frames_override if total_frames_override is not None else len(buffer)
        
        structure['timestamp'] = {"shape": [N], "dtype": "float64"}
        
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

    # === Metadata 文件操作 ===
    def _init_metadata_file(self):
        init_data = {
            "meta_info": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task_name": self.config_cache.get('task_name'),
                "node_name": self.get_name(),
                "mode": "SingleArm", # 标记为单臂模式
                "sampling_rate": self.target_rate,
                "topics": self.config_cache.get('topics')
            },
            "dataset_structure": {},
            "episodes": []
        }
        with open(self.metadata_path, 'w') as f:
            yaml.dump(init_data, f, default_flow_style=False, sort_keys=False)

    def _update_metadata_file(self, structure_info, total_frames, total_duration):
        data = {
            "meta_info": {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_episodes": len(self.recorded_episodes_info),
                "total_frames": total_frames,
                "total_duration_sec": total_duration,
                "task_name": self.config_cache.get('task_name'),
                "mode": "SingleArm",
                "sampling_rate": self.target_rate,
                "topics": self.config_cache.get('topics')
            },
            "dataset_structure": structure_info, # 单臂模式下，直接是扁平字典
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
        return yaml.safe_load(f)

def main(args=None):
    rclpy.init(args=args)

    # 1. 加载配置
    # 确保您的 config yaml 文件里 topics 用了 {arm_side} 占位符
    config = load_config("config/default_dataset_config.yaml")

    # 2. 启动单臂节点
    logger_node = SingleArmDataLogger(config)

    # 3. 线程与交互
    def ros_thread_entry():
        try:
            rclpy.spin(logger_node)
        except Exception:
            pass
    t = threading.Thread(target=ros_thread_entry, daemon=True)
    t.start()

    print(f"\n=== 单臂 ({logger_node.target_arm}) 录制控制台 ===")
    print(f"任务: {config.get('task_name')}")
    print(" [B] 开始/停止录制")
    print(" [A] 切换 Active 状态")
    print(" [Q] 退出")

    sim_b_pressed = False 
    sim_active = False 

    try:
        while rclpy.ok():
            cmd = input(f"[SingleArm] Active:{sim_active} > ").strip().lower()
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