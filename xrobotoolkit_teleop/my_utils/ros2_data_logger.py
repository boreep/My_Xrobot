# 
import rclpy
from rclpy.node import Node
import message_filters
import threading
import queue
import numpy as np
import h5py
import os
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Callable, Optional

# === 消息类型导入 ===
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from my_interfaces.msg import HeaderFloat32 

class RecorderState(Enum):
    IDLE = 0
    RECORDING = 1
    PAUSED = 2

class ROSDataLogger(Node):
    def __init__(self, 
                 log_root_dir: str = "robot_dataset",
                 # 核心改动：不再传入单个topic名，而是传入一个配置字典
                 # 格式：{"逻辑名称": "ROS话题名"}
                 topic_config: Optional[Dict[str, str]] = None,
                 main_topic_key: str = "joint_state", # 指定哪个是主话题（必须存在）
                 recording_rate: int = 20):
        
        super().__init__('robot_data_logger')
        
        # --- 1. 基础配置 ---
        self.log_root_dir = log_root_dir
        if not os.path.exists(self.log_root_dir):
            os.makedirs(self.log_root_dir)
            
        self.recording_rate = recording_rate
        self.min_interval = 1.0 / self.recording_rate
        self.last_record_time = 0.0
        
        # --- 2. 注册话题处理逻辑 (积木库) ---
        # 这里的 key 对应 topic_config 中的 key
        self.handlers = {} 
        self._register_handlers() 

        # --- 3. 动态构建订阅器 ---
        self.subscribers = []   # 存放 message_filter 对象
        self.topic_keys = []    # 存放顺序对应的 key (如 'ik_target', 'joint_state')
        
# --- 安全校验 (熔断机制) ---
        if not topic_config:
            # 如果配置为空或 None，直接抛出异常，阻止程序启动
            raise ValueError("❌ 错误: topic_config 不能为空！请传入至少一个话题配置。")

        if main_topic_key not in topic_config:
            # 如果缺少主话题，也直接抛出异常
            raise ValueError(f"❌ 错误: 配置中缺少主话题 key: '{main_topic_key}'。请确保字典中包含该 key。")

        # 遍历配置，像搭积木一样动态创建订阅
        for key, topic_name in topic_config.items():
            if key in self.handlers:
                msg_type = self.handlers[key]['type']
                
                # 创建订阅
                sub = message_filters.Subscriber(self, msg_type, topic_name)
                self.subscribers.append(sub)
                self.topic_keys.append(key) # 记录顺序，以便在回调中对应
                
                self.get_logger().info(f"Registered topic: [{key}] -> {topic_name} ({msg_type.__name__})")
            else:
                self.get_logger().warn(f"Config key '{key}' has no handler defined. Skipping.")

        # --- 4. 同步器 ---
        # 注意：ApproximateTimeSynchronizer 会等待列表里所有话题都收到消息
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, 
            queue_size=50, 
            slop=0.05
        )
        self.synchronizer.registerCallback(self.sync_callback)

        # --- 5. 线程与状态 ---
        self.current_state = RecorderState.IDLE
        self.episode_buffer = [] 
        self.processing_active = True
        self.data_queue = queue.Queue()
        
        self.write_thread = threading.Thread(target=self._process_queue_worker)
        self.write_thread.daemon = True
        self.write_thread.start()

        self.get_logger().info("Dynamic Logger Ready.")

    def _register_handlers(self):
        """
        [可扩展区域] 在这里定义每种数据类型的处理方式
        格式:
        'key': {
            'type': ROS消息类型,
            'parser': 解析函数(msg) -> 返回 numpy 数组或字典
        }
        """
        self.handlers = {
            # 1. 机械臂末端 Pose
            "ik_target": {
                "type": PoseStamped,
                "parser": self._parse_pose
            },
            # 2. 夹爪控制
            "gripper": {
                "type": HeaderFloat32,
                "parser": self._parse_gripper
            },
            # 3. 关节状态 (主话题)
            "joint_state": {
                "type": JointState,
                "parser": self._parse_joint_state
            },
            # --- [未来在这里添加更多积木] ---
            # "camera_rgb": { "type": Image, "parser": self._parse_image }
        }

    # ================= 数据解析函数 (Parsers) =================
    
    def _parse_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        # 返回 7维向量
        return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

    def _parse_gripper(self, msg):
        # 返回 1维向量
        return np.array([msg.data], dtype=np.float32)

    def _parse_joint_state(self, msg):
        # 返回关节角度数组 (根据需要，这里可能需要按关节名排序，这里仅作简单示例)
        return np.array(msg.position, dtype=np.float32)

    # ========================================================

    def sync_callback(self, *msgs):
        """
        通用回调：接收不定数量的消息 (*msgs)
        """
        if self.current_state != RecorderState.RECORDING:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9

        if (current_time - self.last_record_time) >= self.min_interval:
            # 将所有消息打包放入队列
            # msgs 是一个元组，顺序对应 self.topic_keys
            self.data_queue.put(msgs)
            self.last_record_time = current_time

    def _process_queue_worker(self):
        while rclpy.ok() and self.processing_active:
            try:
                # 获取消息元组
                msgs_tuple = self.data_queue.get(timeout=0.1)
                
                frame_data = {}
                
                # 遍历收到的消息，根据 key 找到对应的 parser 进行解析
                for i, msg in enumerate(msgs_tuple):
                    key = self.topic_keys[i]       # 获取逻辑名称 (e.g., 'ik_target')
                    parser = self.handlers[key]['parser'] # 获取解析函数
                    
                    # 执行解析并存入字典
                    frame_data[key] = parser(msg)
                    
                    # 统一使用第一个消息的时间戳 (通常是主话题)
                    if i == 0:
                        # 假设所有消息都有 header，如果没有需特殊处理
                        if hasattr(msg, 'header'):
                            frame_data["timestamp"] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                        else:
                            frame_data["timestamp"] = time.time()

                self.episode_buffer.append(frame_data)
                
                if len(self.episode_buffer) % self.recording_rate == 0: 
                    self.get_logger().info(f"Recording... {len(self.episode_buffer)} frames")

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Worker Error: {e}")

    def start_episode(self):
        if self.current_state == RecorderState.IDLE:
            self.episode_buffer = [] 
            with self.data_queue.mutex:
                self.data_queue.queue.clear()
            self.last_record_time = 0.0 
            self.current_state = RecorderState.PAUSED 
            self.get_logger().info(f">>> EPISODE STARTED <<<")

    def stop_episode(self):
        if self.current_state != RecorderState.IDLE:
            self.current_state = RecorderState.IDLE 
            while not self.data_queue.empty():
                time.sleep(0.01)
            self._save_to_hdf5()

    def update_active_status(self, is_active: bool):
        if self.current_state == RecorderState.IDLE: return 
        if is_active:
            if self.current_state == RecorderState.PAUSED:
                self.current_state = RecorderState.RECORDING
        else:
            if self.current_state == RecorderState.RECORDING:
                self.current_state = RecorderState.PAUSED

    def _save_to_hdf5(self):
        if not self.episode_buffer:
            self.get_logger().warn("Buffer empty.")
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_root_dir, f"episode_{timestamp_str}.h5")
        
        try:
            with h5py.File(filename, 'w') as f:
                # 获取所有被记录的 key (除了 timestamp)
                keys_to_save = [k for k in self.topic_keys]
                
                # 动态保存所有数据
                for key in keys_to_save:
                    # 提取该 key 对应的所有帧的数据并堆叠
                    data_stack = np.stack([frame[key] for frame in self.episode_buffer])
                    f.create_dataset(key, data=data_stack)

                # 保存时间戳
                timestamps = np.array([x["timestamp"] for x in self.episode_buffer])
                f.create_dataset("timestamp", data=timestamps)
                
                f.attrs['frames'] = len(self.episode_buffer)
                f.attrs['recording_rate'] = self.recording_rate
                # 记录一下包含了哪些 key
                f.attrs['keys'] = keys_to_save 

            self.get_logger().info(f"Saved {len(self.episode_buffer)} frames to {filename}")
        except Exception as e:
            self.get_logger().error(f"Save failed: {e}")
        finally:
            self.episode_buffer = []

# ==========================================
#               调用示例
# ==========================================

def main(args=None):
    rclpy.init(args=args)
    
    # 1. 定义你想录制的话题配置 (积木菜单)
    #    Key 必须在 _register_handlers 中定义过
    #    Value 是实际的 ROS Topic 名字
    my_topic_config = {
        "joint_state": "/arm_left/joint_states",      # 主话题
        "ik_target":   "/arm_left/ik_target",         # 现有话题
        "gripper":     "/arm_left/gripper_control"    # 现有话题
        # "camera_rgb": "/camera/color/image_raw"     # 以后想加直接在这里写一行
    }

    node = ROSDataLogger(
        log_root_dir="test_dataset_v2",
        topic_config=my_topic_config,    # <--- 传入字典
        main_topic_key="joint_state",    # <--- 指定核心话题
        recording_rate=30
    )


    # 共享状态
    sim_context = {"is_active": False, "running": True}

    # 2. ROS 线程
    def ros_spin_thread():
        rclpy.spin(node)
    
    t_ros = threading.Thread(target=ros_spin_thread, daemon=True)
    t_ros.start()

    # 3. 模拟逻辑循环线程 (高频)
    def game_loop_simulation():
        print(">>> Simulation Loop Started <<<")
        while rclpy.ok() and sim_context["running"]:
            node.update_active_status(sim_context["is_active"])
            time.sleep(0.01) # 循环频率 (比如 100Hz)，不影响录制频率

    t_sim = threading.Thread(target=game_loop_simulation, daemon=True)
    t_sim.start()

    # 4. 键盘控制
    print("\nControl: [s]Start [e]Stop [a]ToggleActive [q]Quit")

    try:
        while rclpy.ok():
            cmd = input("Cmd > ").strip().lower()
            
            if cmd == 's': 
                node.start_episode()
            elif cmd == 'e': 
                node.stop_episode()
            elif cmd == 'a': 
                sim_context["is_active"] = not sim_context["is_active"]
                state = "Active" if sim_context["is_active"] else "Inactive"
                print(f"-> Signal: {state}")
            elif cmd == 'q':
                if node.current_state != RecorderState.IDLE: 
                    node.stop_episode()
                sim_context["running"] = False
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        sim_context["running"] = False
        node.destroy_node()
        rclpy.shutdown()
        # 这里的 join 可能需要配合 daemon 使用，直接退出也可

if __name__ == '__main__':
    main()