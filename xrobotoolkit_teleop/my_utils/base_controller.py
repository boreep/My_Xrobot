import abc
import threading
import webbrowser
from typing import Any, Dict

import meshcat.transformations as tf
import numpy as np
import placo
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import (
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import (
    calc_parallel_gripper_position,
)

SELF_COLLISION_MARGIN = 0.075  # 自碰撞避免的默认安全距离 [m]
SELF_COLLISION_TRIGGER = 0.2  # 自碰撞避免的触发距离   [m]

class BaseController(abc.ABC):
    """遥操作控制器基类，定义了遥操作的基本框架和通用功能"""


    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base: bool,
        R_headset_world: np.ndarray,
        scale_factor: float,
        q_init: np.ndarray ,
        dt: float,
        self_collision_avoidance_enabled: bool = False,
        enable_log_data: bool = False,

    ):
        """初始化遥操作控制器基类"""
        self.robot_urdf_path = robot_urdf_path
        self.manipulator_config = manipulator_config
        self.floating_base = floating_base
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.q_init = q_init
        self.dt = dt
        self.xr_client = XrClient()
        
        self.enable_self_collision_avoidance = self_collision_avoidance_enabled

        self.enable_log_data = enable_log_data


        # 初始化位姿参考变量
        self.ref_ee_xyz: Dict[str, Any] = {name: None for name in manipulator_config.keys()}
        self.ref_ee_quat: Dict[str, Any] = {name: None for name in manipulator_config.keys()}
        self.ref_controller_xyz: Dict[str, Any] = {name: None for name in manipulator_config.keys()}
        self.ref_controller_quat: Dict[str, Any] = {name: None for name in manipulator_config.keys()}
        self.effector_task = {}
        self.effector_control_mode = {}  # 存储每个末端执行器的控制模式
        self.active = {}
        self.gripper_pos_target = {}    
        
        self.ik_targets = {}

        # 运动追踪器支持
        self.motion_tracker_task = {}
        self.ref_tracker_xyz = {}  # 存储初始追踪器位置
        self.ref_robot_xyz = {}  # 存储初始机器人末端执行器位置
        
        for name, config in self.manipulator_config.items():
            if "gripper_config" in config:
                gripper_config = config["gripper_config"]
                self.gripper_pos_target[name] = {
                    joint_name: joint_pos
                    for joint_name, joint_pos in zip(gripper_config["joint_names"], gripper_config["open_pos"])
                }

        self._stop_event = threading.Event()

        self._robot_setup()
        self._placo_setup()

    def _process_xr_pose(self, xr_pose, src_name):
        """处理XR控制器的当前位姿数据"""
        # 提取控制器位置和四元数姿态
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = [
            xr_pose[6],  # w
            xr_pose[3],  # x
            xr_pose[4],  # y
            xr_pose[5],  # z
        ]

        # 应用坐标变换
        controller_xyz = self.R_headset_world @ controller_xyz
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )

        # 计算相对位姿变化
        if self.ref_controller_xyz[src_name] is None:
            self.ref_controller_xyz[src_name] = controller_xyz
            self.ref_controller_quat[src_name] = controller_quat
            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            delta_xyz = (controller_xyz - self.ref_controller_xyz[src_name]) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat[src_name], controller_quat)

        return delta_xyz, delta_rot
    
    def _process_xr_velocity(self, xr_linear_vel, xr_angular_vel):
        """
        处理 XR 速度数据
        :param xr_linear_vel: SDK 返回的线速度 [vx, vy, vz]
        :param xr_angular_vel: SDK 返回的角速度 [wx, wy, wz]
        """
        # 1. 线速度处理 (同样需要旋转)
        # 即使你觉得 xyz 不需要调整，但如果 R_headset_world 存在旋转，线速度方向也必须跟着转，
        robot_linear_vel = self.R_headset_world @ np.array(xr_linear_vel) * self.scale_factor

        # 2. 角速度处理 (必须旋转)
        # 将 VR 坐标系下的旋转轴映射到机器人世界坐标系
        robot_angular_vel = self.R_headset_world @ np.array(xr_angular_vel)

        return robot_linear_vel, robot_angular_vel

    def _placo_setup(self):
        """设置Placo逆运动学求解器"""
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        print("[placo_setup] Joint names in the Placo model:")
        for joint_name in self.placo_robot.model.names:
            print(f"  {joint_name}")

        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self.dt
        
        # 设置初始配置
        if self.q_init is not None:
            if self.floating_base:
                self.placo_robot.state.q = self.q_init.copy()
            else:
                self.solver.mask_fbase(True)
                self.placo_robot.state.q[7:] = self.q_init.copy()
        else:
            if not self.floating_base:
                self.solver.mask_fbase(True)
            self.placo_robot.state.q[:7] = np.array([0, 0, 0, 0, 0, 0, 1])

        self.placo_robot.update_kinematics()
        
        if self.enable_self_collision_avoidance:
            avoid_self_collisions = self.solver.add_avoid_self_collisions_constraint()
            avoid_self_collisions.configure("avoid_self_collisions", "hard")
            avoid_self_collisions.self_collisions_margin = SELF_COLLISION_MARGIN  # [m]
            avoid_self_collisions.self_collisions_trigger = SELF_COLLISION_TRIGGER  # [m]
            print("[placo_setup] Self-collision avoidance enabled in Placo solver.")
        else:
            print("[placo_setup] Self-collision avoidance is NOT enabled.")
            
        # 为每个操作器设置末端执行器任务
        for name, config in self.manipulator_config.items():
            # 获取控制模式（默认为"pose"）
            control_mode = config.get("control_mode", "pose")
            self.effector_control_mode[name] = control_mode
            
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            
            if control_mode == "position":
                # 位置控制模式
                self.effector_task[name] = self.solver.add_position_task(config["link_name"], ee_xyz)
                print(f"[placo_setup] Created position task for {name} -> {config['link_name']}")
                self.effector_task[name].configure(name, "soft", 1.0)
            else:
                # 全位姿控制模式（默认）
                ee_target = tf.quaternion_matrix(ee_quat)
                ee_target[:3, 3] = ee_xyz
                self.effector_task[name] = self.solver.add_frame_task(config["link_name"], ee_target)
                print(f"[placo_setup] Created pose task for {name} -> {config['link_name']}")
                self.effector_task[name].configure(name, "soft", 1.0, 0.1)
            

            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0) 
            manipulability.configure("manipulability", "soft", 1e-2)    #奇异性约束

            if "velocity_limits" in config:
                # 1. 遍历并设置每一个关节的限速
                for joint_name, limit_val in config["velocity_limits"].items():
                    if joint_name in self.placo_robot.model.names:
                        self.placo_robot.set_velocity_limit(joint_name, limit_val)
                    else:
                        print(f"[placo_setup] 警告: 关节 {joint_name} 未在 placo 模型中找到，无法设置速度限制")
                
                # 2. 显式开启速度限制功能
                self.solver.enable_velocity_limits(True)
                print(f"[placo_setup] Velocity limits enabled for joints in {name}.")
                
    
            # 设置运动追踪器任务（如果配置了）
            if "motion_tracker" in config:
                tracker_config = config["motion_tracker"]
                link_target = tracker_config["link_target"]
                target_xyz, _ = self._get_link_pose(link_target)
                tracker_task_name = f"{name}_tracker"
                self.motion_tracker_task[name] = self.solver.add_position_task(link_target, target_xyz)
                self.motion_tracker_task[name].configure(tracker_task_name, "soft", 1.0)
                print(f"[placo_setup] Motion tracker position task created for {name} -> {link_target}")

        self.placo_robot.update_kinematics()

    def _update_ik(self):
        # 状态更新在pre_ik中处理
        # 处理每个操作器的控制逻辑

        for src_name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.active[src_name] = xr_grip_val > 0.8
            if self.active[src_name]:
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} is activated.")
                    self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name] = self._get_link_pose(config["link_name"])

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)
                
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
            else: #松开触发按钮后，下一次再按下时候，会以当前状态为新起点，避免跳变
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} is deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_controller_xyz[src_name] = None


        # 处理运动追踪器数据
        # self._update_motion_tracker_tasks()

        # 求解逆运动学
        try:
            self.solver.solve(True)
        except RuntimeError as e:
            print(f"[update_ik] IK solver failed: {e}")

    def _update_motion_tracker_tasks(self):
        """处理运动追踪器数据并更新相应的Placo任务"""
        motion_tracker_data = self.xr_client.get_motion_tracker_data()

        for src_name, config in self.manipulator_config.items():
            # 跳过未配置运动追踪器的操作器
            if "motion_tracker" not in config:
                continue

            # 跳过未激活的主控制器
            if not self.active.get(src_name, False):
                if src_name in self.ref_tracker_xyz:
                    del self.ref_tracker_xyz[src_name]
                    del self.ref_robot_xyz[src_name]
                continue

            tracker_config = config["motion_tracker"]
            serial = tracker_config["serial"]

            # 跳过不可用的追踪器
            if serial not in motion_tracker_data:
                continue

            # 获取运动追踪器位姿
            tracker_pose = motion_tracker_data[serial]["pose"]
            tracker_xyz = self.R_headset_world @ np.array(tracker_pose[:3])

            # 首次检测时初始化参考位置
            if src_name not in self.ref_tracker_xyz:
                self.ref_tracker_xyz[src_name] = tracker_xyz.copy()
                robot_xyz, _ = self._get_link_pose(config["motion_tracker"]["link_target"])
                self.ref_robot_xyz[src_name] = robot_xyz.copy()
                continue

            # 计算并应用追踪器移动
            tracker_delta = tracker_xyz - self.ref_tracker_xyz[src_name]
            final_target_xyz = self.ref_robot_xyz[src_name] + tracker_delta * self.scale_factor

            # 更新运动追踪器任务目标位置
            if src_name in self.motion_tracker_task:
                self.motion_tracker_task[src_name].target_world = final_target_xyz

    def _init_placo_viz(self):
        """初始化Placo可视化"""
        self.placo_vis = robot_viz(self.placo_robot)
        webbrowser.open(self.placo_vis.viewer.url())
        self.placo_vis.display(self.placo_robot.state.q)
        
        for name, config in self.manipulator_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            
            # 根据控制模式显示适当的可视化
            if self.effector_control_mode[name] == "position":
                target_frame = np.eye(4)
                target_frame[:3, 3] = self.effector_task[name].target_world
                frame_viz(f"vis_target_{name}", target_frame)

            else:
                frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

            # 可视化运动追踪器目标（如果配置了）
            if "motion_tracker" in config and name in self.motion_tracker_task:
                link_target = config["motion_tracker"]["link_target"]
                robot_frame_viz(self.placo_robot, link_target)
                tracker_frame = np.eye(4)
                tracker_frame[:3, 3] = self.motion_tracker_task[name].target_world
                frame_viz(f"vis_tracker_{name}", tracker_frame)
                            


    def _update_placo_viz(self):
        """更新Placo可视化"""
        self.placo_vis.display(self.placo_robot.state.q)
        
        for name, config in self.manipulator_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            
            # 根据控制模式更新可视化
            if self.effector_control_mode[name] == "position":
                target_frame = np.eye(4)
                target_frame[:3, 3] = self.effector_task[name].target_world
                frame_viz(f"vis_target_{name}", target_frame)
            else:
                frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

            # 更新运动追踪器可视化（如果配置了）
            if "motion_tracker" in config and name in self.motion_tracker_task:
                link_target = config["motion_tracker"]["link_target"]
                robot_frame_viz(self.placo_robot, link_target)
                tracker_frame = np.eye(4)
                tracker_frame[:3, 3] = self.motion_tracker_task[name].target_world
                frame_viz(f"vis_tracker_{name}", tracker_frame)

    def sync_end_effector_poses_to_placo_tasks(self):
        """将当前末端执行器位姿同步到placo任务"""
        for name, config in self.manipulator_config.items():
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            
            # 更新对应的placo任务
            if self.effector_control_mode[name] == "position":
                self.effector_task[name].target_world = ee_xyz
            else:
                ee_target = tf.quaternion_matrix(ee_quat)
                ee_target[:3, 3] = ee_xyz
                self.effector_task[name].T_world_frame = ee_target
            
            print(f"Synced {name} end effector pose to placo task: {config['link_name']}")

    def _update_gripper_target(self):
        """更新夹爪目标位置"""
        for gripper_name in self.manipulator_config.keys():
            if "gripper_config" not in self.manipulator_config[gripper_name]:
                continue

            gripper_config = self.manipulator_config[gripper_name]["gripper_config"]
            gripper_type = gripper_config["type"]
            
            if gripper_type == "parallel":
                trigger_value = self.xr_client.get_key_value_by_name(gripper_config["gripper_trigger"])
                for joint_name, open_pos, close_pos in zip(
                    gripper_config["joint_names"],
                    gripper_config["open_pos"],
                    gripper_config["close_pos"],
                ):
                    # 计算夹爪目标位置
                    gripper_pos = calc_parallel_gripper_position(open_pos, close_pos, trigger_value)
                    self.gripper_pos_target[gripper_name][joint_name] = gripper_pos
            else:
                raise ValueError(f"Unsupported gripper type: {gripper_type}")


    # ---------------------------------------------------------
    # --- 抽象方法 (必须由子类实现) ---
    # ---------------------------------------------------------

    @abc.abstractmethod
    def _robot_setup(self):
        """初始化特定后端（连接机器人、启动仿真等）"""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_robot_state(self):
        """从机器人/仿真中读取当前关节状态"""
        raise NotImplementedError

    @abc.abstractmethod
    def _send_command(self):
        """将目标关节位置发送到机器人/仿真"""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_link_pose(self, link_name):
        """获取给定链接的世界位姿"""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        """主执行入口点"""
        raise NotImplementedError