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

# 自动定位项目根目录 (保留你的原始逻辑)
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
#           积木块：单臂录制器 (保持不变)
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
            
            # 自动替换占位符，支持 "left_arm/topic" 或 "topic_left" 格式
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
                # === [新增 2] 每次收到新数据，版本号 +1 ===
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
#           主节点：双臂数据记录器
# ==========================================
class DualArmDataLogger(Node):
    def __init__(self, config: Dict[str, Any]):
        super().__init__('dual_arm_logger')
        
        self.config_cache = config
        # === [新增 3] 记录上一次保存的版本号 ===
        self.last_ver_left = -1
        self.last_ver_right = -1
        
        # 配置参数读取
        base_dir = config.get('base_dir', 'robot_dataset')
        task_name = config.get('task_name', 'default_task')
        target_rate = config.get('sampling_rate', 20)
        topic_logic_map = config.get('topics', {})
        self.target_rate = target_rate

        self.current_state = RecorderState.IDLE
        self.is_active = False
        self.episode_count = 0 
        
        # === 1. 构建三级目录 (Run级隔离) ===
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

        # === 4. 初始化双臂 Recorder ===
        self.handlers = {}
        self._register_handlers()
        
        # 实例化左右臂 (topic_logic_map 里的 {arm_side} 会被自动替换)
        self.left_arm = ArmRecorder(self, "left_arm", topic_logic_map, self.handlers)
        self.right_arm = ArmRecorder(self, "right_arm", topic_logic_map, self.handlers)

        # === 5. 启动核心 Timer ===
        self.timer = self.create_timer(1.0 / self.target_rate, self.timer_callback)
        self.get_logger().info(f"[System] Ready. Dual Arm Sampling Rate: {target_rate}Hz")

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
        left_data, left_ver = self.left_arm.sample_current_frame()
        right_data, right_ver = self.right_arm.sample_current_frame()

        # === [逻辑 A] 初始启动门槛：原本的都为 None 则不开始 ===
        if left_data is None or right_data is None:
            self.get_logger().warn("Waiting for initial data...", throttle_duration_sec=1)
            return

        # === [逻辑 B] 检查是否有更新 ===
        # 只要当前的 ID 比上一次保存的 ID 大，就说明有新数据
        has_new_left = left_ver > self.last_ver_left
        has_new_right = right_ver > self.last_ver_right

        # === [逻辑 C] 核心判断 ===
        # 只有当两边都没有更新时，才暂停记录
        if not has_new_left and not has_new_right:
            return  # 直接跳过，不写入 Buffer

        # === [执行记录] ===
        # 只要走到这里，说明至少有一只手更新了。
        # 没更新的那只手，会复用 sample_current_frame 返回的旧数据 (Zero-Order Hold)
        self.left_arm.record_frame(left_data)
        self.right_arm.record_frame(right_data)

        # 更新记录下来的版本号
        self.last_ver_left = left_ver
        self.last_ver_right = right_ver

    def start_episode(self):
        if self.current_state == RecorderState.IDLE:
            self.left_arm.clear()
            self.right_arm.clear()
            # 重置版本记忆，确保下一轮录制从头开始
            self.last_ver_left = -1
            self.last_ver_right = -1
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
    #      核心：双臂 HDF5 保存与元数据更新
    # ==========================================
    def _save_to_hdf5(self):
        # 简单检查：任意一个臂没数据都跳过
        if not self.left_arm.has_data() or not self.right_arm.has_data():
            self.get_logger().warn("Buffer empty in one or both arms, skipping save.")
            return

        filename_only = f"dual_ep{self.episode_count}_{datetime.now().strftime('%H%M%S')}.h5"
        full_filepath = os.path.join(self.run_dir, filename_only)
        
        try:
            # 1. 保存 HDF5 (分组保存)
            with h5py.File(full_filepath, 'w') as f:
                # 创建 /left 组
                g_left = f.create_group("left")
                self._write_group(g_left, self.left_arm.buffer, self.left_arm.keys, "left_arm")
                
                # 创建 /right 组
                g_right = f.create_group("right")
                self._write_group(g_right, self.right_arm.buffer, self.right_arm.keys, "right_arm")
            
            self.get_logger().info(f"Saved: {filename_only}")
            
            # 2. 统计信息 (以左臂的时间为基准，通常两者帧数由 Timer 保证一致)
            # 但为了安全，取最小长度
            frame_count = min(len(self.left_arm.buffer), len(self.right_arm.buffer))
            start_t = self.left_arm.buffer[0]['timestamp']
            end_t = self.left_arm.buffer[-1]['timestamp']
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

            # 4. 分析双臂结构 (分别分析)
            # 这里我们传入 total_frames，让 Metadata 显示的总 Shape 是 [Total, ...]
            structure_left = self._analyze_structure(self.left_arm.buffer, self.left_arm.keys, total_frames)
            structure_right = self._analyze_structure(self.right_arm.buffer, self.right_arm.keys, total_frames)
            
            combined_structure = {
                "left": structure_left,
                "right": structure_right
            }

            # 5. 更新 YAML
            self._update_metadata_file(combined_structure, total_frames, total_duration)

        except Exception as e:
            self.get_logger().error(f"Save Failed: {e}")
            import traceback
            traceback.print_exc()

    def _write_group(self, group, buffer, keys, arm_name):
        # 确保数据对齐，虽然理论上一样长，切片保证安全
        n_frames = len(buffer)
        for key in keys:
            data = np.stack([frame[key] for frame in buffer])
            group.create_dataset(key, data=data, compression="gzip", compression_opts=4)
        
        timestamps = np.array([frame["timestamp"] for frame in buffer])
        group.create_dataset("timestamp", data=timestamps, compression="gzip")
        
        group.attrs['num_frames'] = n_frames
        group.attrs['arm_side'] = arm_name

    def _analyze_structure(self, buffer, keys, total_frames_override=None) -> Dict:
        """分析单个 Buffer 的结构"""
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
                "sampling_rate": self.target_rate,
                "topics": self.config_cache.get('topics')
            },
            "dataset_structure": structure_info, # 这里包含了 left 和 right 的字典
            "episodes": self.recorded_episodes_info
        }
        with open(self.metadata_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        self.get_logger().info(f"Metadata updated (Total: {total_frames} frames)")


# ==========================================
#           Config 加载与主函数
# ==========================================
def load_config(config_path="config/dual_arm_config.yaml.yaml"):
    full_path = os.path.join(DATASET_PATH, config_path)
    if not os.path.exists(full_path):
        print(f"[Error] Config file not found: {full_path}")
        sys.exit(1)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

def main(args=None):
    rclpy.init(args=args)

    # 1. 加载配置 (确保你的 yaml 文件里 topics 用了 {arm_side} 占位符)
    # 例如: "joint_cmd": "{arm_side}/rm_driver/movej_canfd_cmd"
    config = load_config("config/dual_arm_config.yaml")

    # 2. 启动双臂节点
    logger_node = DualArmDataLogger(config)

    # 3. 线程与交互
    def ros_thread_entry():
        try:
            rclpy.spin(logger_node)
        except Exception:
            pass
    t = threading.Thread(target=ros_thread_entry, daemon=True)
    t.start()

    print(f"\n=== 双臂 VR 录制控制台 ===")
    print(f"任务: {config.get('task_name')}")
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