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

from xrobotoolkit_teleop.utils.path_utils import DATASET_PATH # 假设环境里有这个，如果报错请注释掉


# === 新增：视觉相关依赖 ===
import cv2
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2 # ROS2 里的标准点云处理库

# === 消息类型导入 ===
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from my_interfaces.msg import HeaderFloat32 
from rm_ros_interfaces.msg import Jointpos

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
        self.last_update_time = 0.0  # 初始化上次更新时间为 0
        cb_group = getattr(node, 'cb_group', None)
        
        # 遍历配置中的所有 key (例如 joint_cmd, joint_state)
        for key, topic_suffix in topic_config.items():
            # 1. 检查是否有对应的 Handler (如果没有注册Handler，则跳过，方便扩展Vision组)
            if key not in handlers:
                node.get_logger().warn(f"[{group_name}] No handler for key '{key}', skipping.")
                continue
            
            # 2. 拼接完整话题: /group_name/topic_suffix
            # 处理双斜杠问题，确保路径整洁
            raw_topic = f"/{group_name}/{topic_suffix}"
            full_topic = raw_topic.replace("//", "/")
            
            msg_type = handlers[key]['type']
            
            # 3. 创建订阅
            # 注意：message_filters.Subscriber 不需要 callback，它被传入 Synchronizer
        # 修改这里：显式将 callback_group 传给 Subscriber
            sub = message_filters.Subscriber(
                node, 
                msg_type, 
                full_topic,
                callback_group=cb_group # <-- 确保订阅者也加入这个可重入组
            )
            self.subs.append(sub)
            self.keys.append(key)
            node.get_logger().info(f"[{group_name}] Listening: {full_topic} (as {key})")

        # 4. 只有当该组内有有效话题时，才注册同步器
        if self.subs:
            # allow_headerless=True 允许处理没有 header 的消息（如果有的话），但最好都有 header
            self.sync = message_filters.ApproximateTimeSynchronizer(self.subs, queue_size=30, slop=0.075)
            self.sync.registerCallback(self.sync_callback)
        else:
            node.get_logger().error(f"[{group_name}] No valid topics found to record!")

    def sync_callback(self, *msgs):
        """后台接收回调，将该组内多话题对齐后的数据存入 _latest_frame"""
        # === [优化] 前端限流 ===
        # 获取当前时间
        now = self.node.get_clock().now().nanoseconds / 1e9
        
        # 只有当距离上次有效帧超过一定时间（比如 0.045秒）才进行解析
        # 这样可以防止 60Hz 的相机把 CPU 跑满
        # 0.045s 略小于 0.05s (20Hz)，保证 Timer 来取的时候肯定有较新的数据
        if self._latest_frame is not None:
            # 这里的 last_update_time 需要你在 __init__ 里初始化为 0
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
                self.last_update_time = now # 更新时间标记

        except Exception as e:
            self.node.get_logger().warn(f"[{self.group_name}] Parse Error: {e}")

    def sample_current_frame(self):
        """取出当前最新的帧 (Zero-Order Hold 用)"""
        with self._lock:
            if self._latest_frame is None: 
                return None
            return self._latest_frame.copy()

    def record_frame(self, data):
        self.buffer.append(data)
        # 简单的 Logging，避免刷屏
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
        
        # === 1. 解析系统级参数 ===
        # 这些 key 是系统预留的，不会被当做 "采集组"
        self.SYSTEM_KEYS = ['base_dir', 'task_name', 'sampling_rate', 'ros_distro']
        
        base_dir = config.get('base_dir', 'robot_dataset')
        task_name = config.get('task_name', 'default_task')
        target_rate = config.get('sampling_rate', 25)
        self.target_rate = target_rate

        self.current_state = RecorderState.IDLE
        self.is_active = False # 控制是否写入数据
        self.episode_count = 0 
        
        # === 2. 目录构建 ===
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

        # === 3. 注册解析器 (Handlers) ===
        # 这里定义了 config 中 key 对应的 消息类型 和 解析函数
        self.handlers = {
            "joint_cmd":   { "type": Jointpos, "parser": lambda msg: np.array(msg.joint, dtype=np.float32) },
            "ik_target":   { "type": PoseStamped, "parser": self._parse_pose },
            "gripper_cmd":     { "type": HeaderFloat32, "parser": lambda msg: np.array([msg.data], dtype=np.float32) },
            "joint_state": { "type": JointState, "parser": lambda msg: np.array(msg.position, dtype=np.float32) },
            # 视觉相关 (对应 config 里的 keys)
            "image":       { "type": Image, "parser": self._parse_image },
            "point":       { "type": PointCloud2, "parser": self._parse_pointcloud }
        }

        # === 4. 动态创建 GroupRecorder ===
        # 核心逻辑：遍历 Config，如果 key 不是系统参数，就认为是一个 Group
        self.recorders: Dict[str, GroupRecorder] = {}
        
        for key, value in config.items():
            if key in self.SYSTEM_KEYS:
                continue
            
            if isinstance(value, dict):
                self.get_logger().info(f"[Init] Found Group Config: {key}")
                recorder = GroupRecorder(self, key, value, self.handlers)
                # 只有当 Recorder 成功订阅了话题才加入列表
                if recorder.subs:
                    self.recorders[key] = recorder
            else:
                self.get_logger().warn(f"[Init] Skipping unknown config key: {key} (value is not a dict)")

        if not self.recorders:
            self.get_logger().error("No valid recorder groups initialized! Check config.")
            # 可以在这里抛出异常或退出，视需求而定

        # === 5. 启动核心 Timer ===
        self.timer = self.create_timer(
            1.0 / self.target_rate, 
            self.timer_callback, 
            callback_group=self.cb_group
        )
        self.get_logger().info(f"[System] Ready. Sampling Rate: {target_rate}Hz. Active Groups: {list(self.recorders.keys())}")

    def _parse_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)
    
    def _parse_image(self, msg):
            try:
                # 使用新的手动转换函数
                image = ros2_img_to_cv2(msg)
                
                # 缩放 (可选，保留你原有的逻辑)
                # 这里的 interpolation 参数可以保留
                return cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
                
            except Exception as e:
                self.get_logger().warn(f"Image Parse Error: {e}")
                # 返回一个空的黑图，防止整个程序崩溃
                return np.zeros((224, 224, 3), dtype=np.uint8)
            
    def _parse_pointcloud(self, msg):
        """ 
        极速透传模式：兼容 32字节对齐 (PCL默认) 和 16字节紧凑格式
        """
        n_points = msg.width * msg.height
        if n_points == 0:
            return np.zeros((0, 4), dtype=np.float32)
            
        # 1. 计算每个点包含多少个 float (步长)
        # msg.point_step 是字节数 (通常32)，除以4得到 float 数量 (8)
        stride = msg.point_step // 4 
        
        # 2. 读取原始数据并 Reshape 为 (N, stride)
        # 此时 raw_array shape 是 (4096, 8)
        raw_array = np.frombuffer(msg.data, dtype=np.float32).reshape(n_points, stride)

        # 3. 根据步长提取数据
        if stride >= 5:
            # PCL 标准对齐 (32字节): 
            # 内存布局: [x, y, z, padding, rgb, padding, padding, padding]
            # 索引对应: [0, 1, 2,    3,     4,     5,       6,       7   ]
            return raw_array[:, [0, 1, 2, 4]]
            
        elif stride == 4:
            # 紧凑格式 (16字节): [x, y, z, rgb]
            return raw_array
            
        else:
            self.get_logger().error(f"Unsupported point step: {msg.point_step} bytes")
            return np.zeros((n_points, 4), dtype=np.float32)
        
    # def _parse_pointcloud(self, msg):
    #     """ 
    #     解析 PointCloud2 -> Numpy (N, 6) float32
    #     输出格式: [x, y, z, r, g, b] (注意: r,g,b 是 0-255 的 float)
    #     兼容: 
    #       - PCL 标准对齐 (point_step=32, stride=8)
    #       - 紧凑格式 (point_step=16, stride=4)
    #     """
    #     n_points = msg.width * msg.height
    #     if n_points == 0:
    #         return np.zeros((0, 6), dtype=np.float32)

    #     # 1. 计算步长 (stride)
    #     # float32 占 4 字节，point_step // 4 得到每行有多少个 float
    #     stride = msg.point_step // 4
        
    #     # 2. 读取原始数据 (float32 视图)
    #     # 这一步是 Zero-Copy 的
    #     raw_array = np.frombuffer(msg.data, dtype=np.float32).reshape(n_points, stride)

    #     # 3. 提取 XYZ 和 Packed RGB
    #     # XYZ 永远在前 3 列
    #     xyz = raw_array[:, 0:3]
        
    #     # 确定 RGB 所在的列索引
    #     if stride >= 5: 
    #         # PCL 标准 (32字节对齐): x,y,z,_,rgb,... -> RGB 在 index 4
    #         rgb_packed = raw_array[:, 4]
    #     else: 
    #         # 紧凑格式 (16字节): x,y,z,rgb -> RGB 在 index 3
    #         rgb_packed = raw_array[:, 3]

    #     # 4. 解码 RGB (Packed Float -> R, G, B floats)
    #     # 必须 copy() 才能 view 为 uint8，因为切片内存可能不连续
    #     # 结果 shape: (N, 4) -> [B, G, R, A] (Little Endian)
    #     bgra_u8 = rgb_packed.copy().view(np.uint8).reshape(-1, 4)
        
    #     # 提取通道并转为 float32 (范围 0.0 - 255.0)
    #     # 使用 astype(np.float32) 会发生数据拷贝
    #     r = bgra_u8[:, 2].astype(np.float32)
    #     g = bgra_u8[:, 1].astype(np.float32)
    #     b = bgra_u8[:, 0].astype(np.float32)

    #     # 5. 堆叠合并
    #     # xyz 是 (N,3), r,g,b 是 (N,), 堆叠后变成 (N, 6)
    #     # np.column_stack 比 vstack/hstack 更智能处理这种情况
    #     return np.column_stack((xyz, r, g, b))
    def timer_callback(self):
        # 1. 门控
        if self.current_state != RecorderState.RECORDING or not self.is_active:
            return
        
        current_time = self.get_clock().now().nanoseconds / 1e9

        # 2. 采集所有组的数据 (Zero-Order Hold 准备)
        current_frames = {}
        all_ready = True
        
        for group_name, recorder in self.recorders.items():
            data = recorder.sample_current_frame()
            if data is None:
                all_ready = False
                break # 只要有一个组没准备好，就认为这帧废了（或者你可以改为 warn 并 return）
            current_frames[group_name] = data

        # 3. 初始启动保护：所有组都必须至少收到过一次数据
        if not all_ready:
            self.get_logger().warn("Waiting for initial data from ALL groups...", throttle_duration_sec=1)
            return
        
        # [修改点] 记录当前时间戳
        self.global_timestamp_buffer.append(current_time)
        
        # 4. 写入 Buffer
        for group_name, recorder in self.recorders.items():
            recorder.record_frame(current_frames[group_name])

    def start_episode(self):
        if self.current_state == RecorderState.IDLE:
            # 清空所有 recorder 的 buffer
            for recorder in self.recorders.values():
                recorder.clear()
            
            # [修改点] 清空全局时间戳
            self.global_timestamp_buffer = []
            
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
        # 检查是否有数据 (简单检查第一个组即可，因为是同步写入的)
        first_group = next(iter(self.recorders.values()))
        if not first_group.has_data():
            self.get_logger().warn("Buffer empty, skipping save.")
            return
        
        # 预先获取帧数，用于日志显示
        total_frames = len(first_group.buffer)
        
        filename_only = f"episode_{self.episode_count}_{datetime.now().strftime('%H%M%S')}.h5"
        full_filepath = os.path.join(self.run_dir, filename_only)
        
        try:
            total_frames = 0
            structures = {}
            
            with h5py.File(full_filepath, 'w') as f:
                # [修改点] 1. 在根目录写入全局时间戳
                timestamps = np.array(self.global_timestamp_buffer, dtype=np.float64)
                f.create_dataset("timestamp", data=timestamps, compression="gzip")
                # 遍历所有 Recorder，分别为它们创建 Group
                for group_name, recorder in self.recorders.items():
                    h5_group = f.create_group(group_name)
                    self._write_group(h5_group, recorder.buffer, recorder.keys, group_name)
                    
                    # 统计结构信息用于 Metadata
                    # 注意：假设所有组帧数一致 (因为是 timer 驱动的)
                    total_frames = len(recorder.buffer)
                    structures[group_name] = self._analyze_structure(recorder.buffer, recorder.keys, total_frames)

            # === [恢复] 保存成功提示，包含帧数信息 ===
            self.get_logger().info(f"Saved: {filename_only} (Frames: {total_frames})")
            
            # 计算时长
            duration = self.global_timestamp_buffer[-1] - self.global_timestamp_buffer[0]

            # 更新 Episode 记录
            ep_info = {
                "id": self.episode_count,
                "filename": filename_only,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frame_count": total_frames,
                "duration_sec": float(duration),
                "groups_included": list(self.recorders.keys())
            }
            self.recorded_episodes_info.append(ep_info)

            # 更新全局 YAML
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
            if not buffer: break
            
            # 取第一帧判断数据类型
            first_val = buffer[0][key]
            
            # === [修改点] 检测是否为元组 (即 separated pointcloud) ===
            if isinstance(first_val, tuple):
                # 创建子组，例如 "point"
                sub_group = group.create_group(key)
                
                # 假设元组只有两个元素 (xyz, rgb)，分别堆叠
                # buffer 中每一帧的 format 是 (xyz_frame, rgb_frame)
                xyz_data = np.stack([frame[key][0] for frame in buffer])
                rgb_data = np.stack([frame[key][1] for frame in buffer])
                
                # 分别保存
                sub_group.create_dataset("xyz", data=xyz_data, compression="gzip", compression_opts=4)
                sub_group.create_dataset("rgb", data=rgb_data, compression="gzip", compression_opts=6)
                
            else:
                # === 原有逻辑 (Image, Joint, Pose 等) ===
                data = np.stack([frame[key] for frame in buffer])
                group.create_dataset(key, data=data, compression="gzip", compression_opts=6)
        
        group.attrs['num_frames'] = n_frames
        group.attrs['group_name'] = group_name

    def _analyze_structure(self, buffer, keys, total_frames) -> Dict:
        structure = {}

        if not buffer: return structure
        first_frame = buffer[0]
        
        for key in keys:
            val = first_frame[key]
            
            # === [修改点] 适配元组结构 ===
            if isinstance(val, tuple):
                # 针对 xyz 和 rgb 分别记录
                # val[0] is xyz, val[1] is rgb
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
                "config_snapshot": self.config_cache # 直接把整个 config 存进去方便回溯
            },
            "dataset_structure": {},
            "episodes": []
        }
        with open(self.metadata_path, 'w') as f:
            yaml.dump(init_data, f, default_flow_style=False, sort_keys=False)

    def _update_metadata_file(self, structure_info, total_frames, total_duration):
        # 读取旧数据以保持 meta_info 不变 (或者你可以选择覆写)
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
    # 为了演示简单，这里不改你的path_utils，直接读取
    full_path = os.path.join(DATASET_PATH, config_path) 
    # 如果你是直接运行脚本，可以用绝对路径测试
    if not os.path.exists(full_path):
        # Fallback to local
        if os.path.exists(config_path):
            full_path = config_path
        else:
            print(f"[Error] Config file not found: {full_path}")
            sys.exit(1)
            
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

def main(args=None):
    rclpy.init(args=args)

    # 假设有个 config.yaml 在当前目录
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