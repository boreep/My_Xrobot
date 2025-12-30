import rclpy
from rclpy.node import Node
# 移除 message_filters，因为我们要手动实现“最新值”逻辑
# import message_filters 
import threading
import queue
import numpy as np
import h5py
import os
import time
from datetime import datetime
from enum import Enum

from geometry_msgs.msg import PoseStamped
from my_interfaces.msg import HeaderFloat32 

class RecorderState(Enum):
    IDLE = 0
    RECORDING = 1
    PAUSED = 2

class RateDataLogger(Node):
    def __init__(self, 
                 log_root_dir: str = "robot_dataset",
                 ik_topic: str = "arm_left/ik_target",
                 gripper_topic: str = "arm_left/gripper_control",
                 recording_rate: int = 20):
        
        super().__init__('robot_data_logger')
        
        # --- 配置 ---
        self.log_root_dir = log_root_dir
        if not os.path.exists(self.log_root_dir):
            os.makedirs(self.log_root_dir)
            
        self.recording_rate = recording_rate
        self.min_interval = 1.0 / self.recording_rate
        self.last_record_time = 0.0 
        
        # --- 关键修改：最新数据缓存 ---
        # 用来存储收到的最新的夹爪数据
        self.latest_gripper_msg = None 
        # 线程锁，防止读写冲突（虽然 Python GIL 某种程度上保证了安全，但加锁是好习惯）
        self.data_lock = threading.Lock()

        # --- 状态 ---
        self.current_state = RecorderState.IDLE
        self.episode_buffer = [] 
        self.processing_active = True

        self.data_queue = queue.Queue()
        self.write_thread = threading.Thread(target=self._process_queue_worker)
        self.write_thread.daemon = True
        self.write_thread.start()

        self.get_logger().info(f"Logger Mode: Master-Slave Trigger (Master: {ik_topic})")

        # --- 订阅 (不再使用 message_filters) ---
        
        # 1. 夹爪 (Slave)：只负责更新缓存
        self.create_subscription(
            HeaderFloat32, 
            gripper_topic, 
            self.gripper_callback, 
            10
        )
        
        # 2. Pose (Master)：负责触发录制
        self.create_subscription(
            PoseStamped, 
            ik_topic, 
            self.pose_callback, 
            10
        )

    # ==========================================
    #               回调逻辑
    # ==========================================

    def gripper_callback(self, msg):
        """
        从话题回调：不管录没录制，永远缓存最新的夹爪状态
        这实现了“最近邻” (Nearest Neighbor) 或“零阶保持” (Zero-Order Hold)
        """
        with self.data_lock:
            self.latest_gripper_msg = msg

    def pose_callback(self, pose_msg):
        """
        主话题回调：这是 '时钟'，决定什么时候保存数据
        """
        # 1. 基础状态检查
        if self.current_state != RecorderState.RECORDING:
            return

        # 2. 频率控制
        current_time = self.get_clock().now().nanoseconds / 1e9
        if (current_time - self.last_record_time) < self.min_interval:
            return # 还没到时间，丢弃这个 Pose

        # 3. 核心：组装数据
        with self.data_lock:
            current_gripper = self.latest_gripper_msg

        # 4. 安全检查：如果从未收到过夹爪数据，可能无法录制
        if current_gripper is None:
            # 你可以选择 return 等待，或者存一个默认值 (比如 0.0)
            # 这里选择打印警告并跳过，直到收到第一个夹爪数据
            # self.get_logger().warn("Waiting for first gripper msg...", throttle_duration_sec=1.0)
            return

        # 5. 发送去后台存储
        self.data_queue.put((pose_msg, current_gripper))
        self.last_record_time = current_time

    # ==========================================
    #               后台处理 (保持不变)
    # ==========================================
    def _process_queue_worker(self):
        while rclpy.ok() and self.processing_active:
            try:
                data = self.data_queue.get(timeout=0.1)
                pose_msg, gripper_msg = data
                
                p = pose_msg.pose.position
                q = pose_msg.pose.orientation
                
                frame_data = {
                    "timestamp": pose_msg.header.stamp.sec + pose_msg.header.stamp.nanosec * 1e-9,
                    "pose": np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32),
                    "gripper": np.array([gripper_msg.data], dtype=np.float32)
                }
                
                self.episode_buffer.append(frame_data)
                
                if len(self.episode_buffer) % self.recording_rate == 0: 
                    self.get_logger().info(f"Recorded {len(self.episode_buffer)} frames")

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Worker Error: {e}")

    # ==========================================
    #               控制接口 (保持不变)
    # ==========================================
    def start_episode(self):
        if self.current_state == RecorderState.IDLE:
            self.episode_buffer = [] 
            with self.data_queue.mutex:
                self.data_queue.queue.clear()
            self.last_record_time = 0.0 
            self.current_state = RecorderState.PAUSED 
            self.get_logger().info(">>> EPISODE STARTED <<<")

    def stop_episode(self):
        if self.current_state != RecorderState.IDLE:
            self.current_state = RecorderState.IDLE 
            while not self.data_queue.empty():
                time.sleep(0.01)
            self._save_to_hdf5()

    def update_active_status(self, is_active: bool):
        if self.current_state == RecorderState.IDLE: return 
        if is_active:
            if self.current_state == RecorderState.PAUSED: self.current_state = RecorderState.RECORDING
        else:
            if self.current_state == RecorderState.RECORDING: self.current_state = RecorderState.PAUSED

    def _save_to_hdf5(self):
        if not self.episode_buffer: return
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_root_dir, f"episode_{timestamp_str}.h5")
        try:
            with h5py.File(filename, 'w') as f:
                poses = np.stack([x["pose"] for x in self.episode_buffer])
                grippers = np.stack([x["gripper"] for x in self.episode_buffer])
                timestamps = np.array([x["timestamp"] for x in self.episode_buffer])
                f.create_dataset("pose", data=poses)
                f.create_dataset("gripper", data=grippers)
                f.create_dataset("timestamp", data=timestamps)
                f.attrs['frames'] = len(self.episode_buffer)
                f.attrs['recording_rate'] = self.recording_rate
            self.get_logger().info(f"Saved {len(self.episode_buffer)} frames to {filename}")
        except Exception as e:
            self.get_logger().error(f"Save failed: {e}")
        finally:
            self.episode_buffer = []

def main(args=None):
    rclpy.init(args=args)
    
    node = RateDataLogger(
        recording_rate=20  # 这里设置录制频率
    )

    sim_context = {"is_active": False, "running": True}

    def ros_spin_thread():
        rclpy.spin(node)
    t_ros = threading.Thread(target=ros_spin_thread, daemon=True)
    t_ros.start()

    def game_loop_simulation():
        while rclpy.ok() and sim_context["running"]:
            node.update_active_status(sim_context["is_active"])
            time.sleep(0.01)
    t_sim = threading.Thread(target=game_loop_simulation, daemon=True)
    t_sim.start()

    print("\nControl: [s]Start [e]Stop [a]ToggleActive [q]Quit")
    try:
        while rclpy.ok():
            cmd = input().strip().lower()
            if cmd == 's': node.start_episode()
            elif cmd == 'e': node.stop_episode()
            elif cmd == 'a': sim_context["is_active"] = not sim_context["is_active"]; print(sim_context["is_active"])
            elif cmd == 'q': 
                if node.current_state != RecorderState.IDLE: node.stop_episode()
                sim_context["running"] = False; break
    except KeyboardInterrupt: pass
    finally:
        sim_context["running"] = False
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()