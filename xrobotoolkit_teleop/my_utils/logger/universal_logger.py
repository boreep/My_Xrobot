import rclpy
from rclpy.node import Node
import message_filters
import threading
import numpy as np
import h5py
import os
import yaml
from datetime import datetime
from enum import Enum
from typing import Dict, Any
import sys
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from xrobotoolkit_teleop.utils.path_utils import DATASET_PATH  # 假设环境里有这个，如果报错请注释掉


# === 新增：视觉相关依赖 ===
import cv2
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2  # ROS2 里的标准点云处理库

# === 消息类型导入 ===
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from my_interfaces.msg import HeaderFloat32
from rm_ros_interfaces.msg import Jointpos, Handstatus


class RecorderState(Enum):
    IDLE = 0
    RECORDING = 1


# === 新增的工具函数 ===
def ros2_img_to_cv2(msg):
    dtype = np.uint8
    n_channels = 3
    if '16UC1' in msg.encoding or 'mono16' in msg.encoding:
        dtype = np.uint16
        n_channels = 1
    elif '8UC1' in msg.encoding or 'mono8' in msg.encoding:
        n_channels = 1
    elif 'bgra8' in msg.encoding or 'rgba8' in msg.encoding:
        n_channels = 4
    img = np.frombuffer(msg.data, dtype=dtype)
    try:
        img = img.reshape((msg.height, msg.width, n_channels))
    except ValueError:
        return np.zeros((msg.height, msg.width, 3), dtype=np.uint8)
    return img


# ==========================================
#           通用积木块：组数据接收器
# ==========================================
class GroupRecorder:
    def __init__(self, node: Node, group_name: str, topic_config: Dict[str, str], handlers: Dict[str, Dict]):
        """
        :param group_name: 组名 (例如 "left_arm", "vision", "mobile_base")
        :param topic_config: 该组下的配置字典 (key: topic_suffix)
        :param handlers: 全局的消息处理器注册表
        """
        self.node = node
        self.group_name = group_name
        self.buffer = []
        self.subs = []
        self.keys = []
        self.handlers = handlers
        self._lock = threading.Lock()
        self._latest_frame = None
        self.last_update_time = 0.0
        self.has_headerless_topic = False
        self.topic_by_key = {}  # 新增：记录 key -> full_topic，便于报缺失项
        cb_group = getattr(node, 'cb_group', None)

        # 遍历配置中的所有 key (例如 joint_cmd, joint_state)
        for key, topic_suffix in topic_config.items():
            # 1. 检查是否有对应的 Handler
            if key not in handlers:
                node.get_logger().warn(f"[{group_name}] No handler for key '{key}', skipping.")
                continue

            # 2. 拼接完整话题: /group_name/topic_suffix
            raw_topic = f"/{group_name}/{topic_suffix}"
            full_topic = raw_topic.replace("//", "/")

            msg_type = handlers[key]['type']
            if handlers[key].get("headerless", False):
                self.has_headerless_topic = True

            # 3. 创建订阅
            sub = message_filters.Subscriber(
                node,
                msg_type,
                full_topic,
                callback_group=cb_group
            )
            self.subs.append(sub)
            self.keys.append(key)
            self.topic_by_key[key] = full_topic
            node.get_logger().info(f"[{group_name}] Listening: {full_topic} (as {key})")

        # 4. 注册同步器
        if self.subs:
            self.sync = message_filters.ApproximateTimeSynchronizer(
                self.subs,
                queue_size=30,
                slop=0.075,
                allow_headerless=self.has_headerless_topic,
            )
            self.sync.registerCallback(self.sync_callback)
        else:
            node.get_logger().error(f"[{group_name}] No valid topics found to record!")

    def sync_callback(self, *msgs):
        """后台接收回调，将该组内多话题对齐后的数据存入 _latest_frame"""
        now = self.node.get_clock().now().nanoseconds / 1e9

        # 前端限流
        if self._latest_frame is not None:
            if (now - self.last_update_time) < 0.025:
                return

        frame_data = {}
        try:
            for i, msg in enumerate(msgs):
                key = self.keys[i]
                parser = self.handlers[key]['parser']
                frame_data[key] = parser(msg)

            with self._lock:
                self._latest_frame = frame_data
                self.last_update_time = now

        except Exception as e:
            self.node.get_logger().warn(f"[{self.group_name}] Parse Error: {e}")

    def sample_current_frame(self):
        """取出当前最新的帧 (Zero-Order Hold 用)"""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_missing_keys(self):
        """
        返回当前 group 里还未就绪的 key，带 topic 名，便于定位到底缺哪个消息。
        """
        with self._lock:
            if self._latest_frame is None:
                return [f"{k}<{self.topic_by_key.get(k, '?')}>" for k in self.keys]

            missing = []
            for key in self.keys:
                if key not in self._latest_frame or self._latest_frame[key] is None:
                    missing.append(f"{key}<{self.topic_by_key.get(key, '?')}>")
            return missing

    def record_frame(self, data):
        self.buffer.append(data)
        if len(self.buffer) % 50 == 0:
            self.node.get_logger().info(f"[{self.group_name}] Recording... Buffer: {len(self.buffer)}")

    def clear(self):
        self.buffer = []

    def has_data(self):
        return len(self.buffer) > 0


# ==========================================
#           主节点：通用数据记录器
# ==========================================
class UniversalDataLogger(Node):
    def __init__(self, config: Dict[str, Any]):
        super().__init__('universal_logger')

        # 1. 创建一个允许重入的回调组
        self.cb_group = ReentrantCallbackGroup()

        self.config_cache = config
        self.global_timestamp_buffer = []

        # 系统预留 key
        self.SYSTEM_KEYS = ['base_dir', 'task_name', 'sampling_rate', 'ros_distro']

        base_dir = config.get('base_dir', 'robot_dataset')
        task_name = config.get('task_name', 'default_task')
        target_rate = config.get('sampling_rate', 15)
        self.target_rate = target_rate

        self.current_state = RecorderState.IDLE
        self.is_active = False
        self.episode_count = 0

        # 2. 目录构建
        date_str = datetime.now().strftime("%Y%m%d")
        session_root = os.path.join(base_dir, f"{task_name}_{date_str}")
        run_time_str = datetime.now().strftime("%H%M%S")
        self.run_dir = os.path.join(session_root, f"run_{run_time_str}")

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir, exist_ok=True)
        self.get_logger().info(f"[System] Data Dir: {self.run_dir}")

        # 备份 Config
        config_backup_path = os.path.join(self.run_dir, f"{task_name}_config.yaml")
        with open(config_backup_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # 初始化 Metadata
        self.metadata_path = os.path.join(self.run_dir, "metadata.yaml")
        self.recorded_episodes_info = []
        self._init_metadata_file()

        # 3. 注册解析器
        self.handlers = {
            "joint_cmd":   {"type": Jointpos,     "parser": lambda msg: np.array(msg.joint, dtype=np.float32)},
            "ik_target":   {"type": Pose,         "parser": self._parse_pose_msg, "headerless": True},
            "ik_state":   {"type": Pose,         "parser": self._parse_pose_msg, "headerless": True},
            "gripper_cmd": {"type": HeaderFloat32,"parser": lambda msg: np.array([msg.data], dtype=np.float32)},
            "joint_state": {"type": JointState,   "parser": lambda msg: np.array(msg.position, dtype=np.float32)},
            "dq_target":   {"type": Jointpos,     "parser": lambda msg: np.array(msg.joint, dtype=np.float32)},
            "hand_state":  {"type": Handstatus,   "parser": lambda msg: np.array(msg.hand_pos, dtype=np.float32), "headerless": True},
            "image":       {"type": Image,        "parser": self._parse_image},
            "point":       {"type": PointCloud2,  "parser": self._parse_pointcloud},
        }

        # 4. 动态创建 GroupRecorder
        self.recorders: Dict[str, GroupRecorder] = {}

        for key, value in config.items():
            if key in self.SYSTEM_KEYS:
                continue

            if isinstance(value, dict):
                self.get_logger().info(f"[Init] Found Group Config: {key}")
                recorder = GroupRecorder(self, key, value, self.handlers)
                if recorder.subs:
                    self.recorders[key] = recorder
            else:
                self.get_logger().warn(f"[Init] Skipping unknown config key: {key} (value is not a dict)")

        if not self.recorders:
            self.get_logger().error("No valid recorder groups initialized! Check config.")

        # 5. 启动核心 Timer
        self.timer = self.create_timer(
            1.0 / self.target_rate,
            self.timer_callback,
            callback_group=self.cb_group
        )
        self.get_logger().info(
            f"[System] Ready. Sampling Rate: {target_rate}Hz. Active Groups: {list(self.recorders.keys())}"
        )

    def _pose_to_array(self, pose):
        p = pose.position
        q = pose.orientation
        # Keep quaternion order as wxyz to match ik_target.
        return np.array([p.x, p.y, p.z, q.w, q.x, q.y, q.z], dtype=np.float32)

    def _parse_pose_msg(self, msg: Pose):
        return self._pose_to_array(msg)

    def _parse_image(self, msg):
        try:
            image = ros2_img_to_cv2(msg)
            return cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            self.get_logger().warn(f"Image Parse Error: {e}")
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def _parse_pointcloud(self, msg):
        """
        极速透传模式：兼容 32字节对齐 (PCL默认) 和 16字节紧凑格式
        """
        n_points = msg.width * msg.height
        if n_points == 0:
            return np.zeros((0, 4), dtype=np.float32)

        stride = msg.point_step // 4
        raw_array = np.frombuffer(msg.data, dtype=np.float32).reshape(n_points, stride)

        if stride >= 5:
            return raw_array[:, [0, 1, 2, 4]]
        elif stride == 4:
            return raw_array
        else:
            self.get_logger().error(f"Unsupported point step: {msg.point_step} bytes")
            return np.zeros((n_points, 4), dtype=np.float32)

    def timer_callback(self):
        # 1. 门控
        if self.current_state != RecorderState.RECORDING or not self.is_active:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9

        # 2. 采集所有组的数据，并记录缺失组/话题
        current_frames = {}
        missing_info = {}

        for group_name, recorder in self.recorders.items():
            data = recorder.sample_current_frame()
            if data is None:
                missing_info[group_name] = recorder.get_missing_keys()
                continue
            current_frames[group_name] = data

        # 3. 初始启动保护：所有组都必须至少收到过一次数据
        if missing_info:
            msg = ", ".join(
                f"{group}: {keys}" for group, keys in missing_info.items()
            )
            self.get_logger().warn(
                f"Waiting for initial data. Missing groups/topics -> {msg}",
                throttle_duration_sec=1
            )
            return

        # 4. 记录当前时间戳
        self.global_timestamp_buffer.append(current_time)

        # 5. 写入 Buffer
        for group_name, recorder in self.recorders.items():
            recorder.record_frame(current_frames[group_name])

    def start_episode(self):
        if self.current_state == RecorderState.IDLE:
            for recorder in self.recorders.values():
                recorder.clear()

            self.global_timestamp_buffer = []

            # 新增：开始录制前打印一次所有 group 的订阅项
            for group_name, recorder in self.recorders.items():
                self.get_logger().info(
                    f"[Check] Group={group_name}, keys={recorder.keys}, topics={recorder.topic_by_key}"
                )

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
        self.get_logger().debug(f"[System] {status_str}")

    # ==========================================
    #      核心：动态 HDF5 保存
    # ==========================================
    def _save_to_hdf5(self):
        if not self.recorders:
            self.get_logger().warn("No recorder groups available, skipping save.")
            return

        first_group = next(iter(self.recorders.values()))
        if not first_group.has_data():
            self.get_logger().warn("Buffer empty, skipping save.")
            return

        total_frames = len(first_group.buffer)

        filename_only = f"episode_{self.episode_count}_{datetime.now().strftime('%H%M%S')}.h5"
        full_filepath = os.path.join(self.run_dir, filename_only)

        try:
            total_frames = 0
            structures = {}

            with h5py.File(full_filepath, 'w') as f:
                timestamps = np.array(self.global_timestamp_buffer, dtype=np.float64)
                f.create_dataset("timestamp", data=timestamps, compression="gzip")

                for group_name, recorder in self.recorders.items():
                    h5_group = f.create_group(group_name)
                    self._write_group(h5_group, recorder.buffer, recorder.keys, group_name)

                    total_frames = len(recorder.buffer)
                    structures[group_name] = self._analyze_structure(
                        recorder.buffer, recorder.keys, total_frames
                    )

            self.get_logger().info(f"Saved: {filename_only} (Frames: {total_frames})")

            duration = 0.0
            if len(self.global_timestamp_buffer) >= 2:
                duration = self.global_timestamp_buffer[-1] - self.global_timestamp_buffer[0]

            ep_info = {
                "id": self.episode_count,
                "filename": filename_only,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frame_count": total_frames,
                "duration_sec": float(duration),
                "groups_included": list(self.recorders.keys())
            }
            self.recorded_episodes_info.append(ep_info)

            total_global_frames = sum(e['frame_count'] for e in self.recorded_episodes_info)
            total_global_duration = sum(e['duration_sec'] for e in self.recorded_episodes_info)
            self._update_metadata_file(structures, total_global_frames, total_global_duration)

        except Exception as e:
            self.get_logger().error(f"Save Failed: {e}")
            import traceback
            traceback.print_exc()

    def _write_group(self, group, buffer, keys, group_name):
        n_frames = len(buffer)
        for key in keys:
            if not buffer:
                break

            first_val = buffer[0][key]

            if isinstance(first_val, tuple):
                sub_group = group.create_group(key)

                xyz_data = np.stack([frame[key][0] for frame in buffer])
                rgb_data = np.stack([frame[key][1] for frame in buffer])

                sub_group.create_dataset("xyz", data=xyz_data, compression="gzip", compression_opts=4)
                sub_group.create_dataset("rgb", data=rgb_data, compression="gzip", compression_opts=6)

            else:
                data = np.stack([frame[key] for frame in buffer])
                group.create_dataset(key, data=data, compression="gzip", compression_opts=6)

        group.attrs['num_frames'] = n_frames
        group.attrs['group_name'] = group_name

    def _analyze_structure(self, buffer, keys, total_frames) -> Dict:
        structure = {}

        if not buffer:
            return structure
        first_frame = buffer[0]

        for key in keys:
            val = first_frame[key]

            if isinstance(val, tuple):
                structure[key] = {
                    "type": "separated_pointcloud",
                    "xyz": {
                        "shape": [total_frames] + list(val[0].shape),
                        "dtype": str(val[0].dtype)
                    },
                    "rgb": {
                        "shape": [total_frames] + list(val[1].shape),
                        "dtype": str(val[1].dtype)
                    }
                }
            elif isinstance(val, np.ndarray):
                full_shape = [total_frames] + list(val.shape)
                dtype_str = str(val.dtype)
                structure[key] = {"shape": full_shape, "dtype": dtype_str}
            else:
                full_shape = [total_frames, 1]
                dtype_str = str(type(val))
                structure[key] = {"shape": full_shape, "dtype": dtype_str}

        return structure

    # === Metadata 文件操作 ===
    def _init_metadata_file(self):
        init_data = {
            "meta_info": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task_name": self.config_cache.get('task_name'),
                "sampling_rate": self.target_rate,
                "config_snapshot": self.config_cache
            },
            "dataset_structure": {},
            "episodes": []
        }
        with open(self.metadata_path, 'w') as f:
            yaml.dump(init_data, f, default_flow_style=False, sort_keys=False)

    def _update_metadata_file(self, structure_info, total_frames, total_duration):
        with open(self.metadata_path, 'r') as f:
            current_data = yaml.safe_load(f)

        current_data["meta_info"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_data["meta_info"]["total_frames"] = total_frames
        current_data["meta_info"]["total_duration_sec"] = total_duration
        current_data["dataset_structure"] = structure_info
        current_data["episodes"] = self.recorded_episodes_info

        with open(self.metadata_path, 'w') as f:
            yaml.dump(current_data, f, default_flow_style=False, sort_keys=False)


# ==========================================
#           Config 加载与主函数
# ==========================================
def load_config(config_path="config/dual_arm_config.yaml"):
    full_path = os.path.join(DATASET_PATH, config_path)
    if not os.path.exists(full_path):
        if os.path.exists(config_path):
            full_path = config_path
        else:
            print(f"[Error] Config file not found: {full_path}")
            sys.exit(1)

    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def main(args=None):
    rclpy.init(args=args)

    config = load_config(config_path="config/default_dataset_config.yaml")

    logger_node = UniversalDataLogger(config)

    executor = MultiThreadedExecutor()
    executor.add_node(logger_node)

    def ros_thread_entry():
        try:
            rclpy.spin(logger_node, executor=executor)
        except Exception:
            pass

    t = threading.Thread(target=ros_thread_entry, daemon=True)
    t.start()

    print(f"\n=== 通用数据录制控制台 ===")
    print(f"Task: {config.get('task_name')}")
    print(f"Groups: {list(logger_node.recorders.keys())}")
    print(" [B] 开始/停止录制")
    print(" [A] 切换 Active 状态")
    print(" [Q] 退出")

    sim_b_pressed = False
    sim_active = False

    try:
        while rclpy.ok():
            cmd = input(f"[Logger] Active:{sim_active} > ").strip().lower()
            if cmd == 'b':
                sim_b_pressed = not sim_b_pressed
                if sim_b_pressed:
                    logger_node.start_episode()
                else:
                    logger_node.stop_episode()
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