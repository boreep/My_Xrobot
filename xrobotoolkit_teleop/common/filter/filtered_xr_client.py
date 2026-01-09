import time
import math
import numpy as np
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.terminalcolor import TerminalColor


# =========================
# 1. 基础低通滤波器
# =========================
class LowPassFilter:
    def __init__(self, alpha=1.0):
        self.y_old = None
        self.alpha = alpha

    def filter(self, value, alpha=None):
        if alpha is not None:
            self.alpha = alpha

        value = np.asarray(value, dtype=float)

        if self.y_old is None:
            self.y_old = value.copy()
            return value.copy()

        filtered = self.alpha * value + (1.0 - self.alpha) * self.y_old
        self.y_old = filtered.copy()
        return filtered.copy()


# =========================
# 2. One Euro Filter
# =========================
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

        self.last_time = None
        self.x_prev = None

    def _smoothing_factor(self, dt, cutoff):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def process(self, x, t=None):
        x = np.asarray(x, dtype=float)

        if t is None:
            t = time.time()

        # ---- 首帧初始化 ----
        if self.last_time is None:
            self.last_time = t
            self.x_prev = x.copy()
            return self.x_filter.filter(x)

        dt = t - self.last_time
        self.last_time = t

        # ---- 时间异常保护 ----
        if dt <= 1e-6:
            return self.x_filter.filter(x)

        # ---- 导数估计 ----
        if self.x_prev is None:
            self.x_prev = x.copy()
            return self.x_filter.filter(x)

        dx = (x - self.x_prev) / dt
        self.x_prev = x.copy()

        # ---- 导数滤波 ----
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
        edx = self.dx_filter.filter(dx, alpha=alpha_d)

        # ---- 自适应 cutoff ----
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._smoothing_factor(dt, cutoff)

        # ---- 主滤波 ----
        return self.x_filter.filter(x, alpha=alpha)


# =========================
# 3. FilteredXrClient
# =========================
class FilteredXrClient(XrClient):
    """
    - left_controller / right_controller:
        pose:  [x,y,z,qx,qy,qz,qw]
        vel:   6D
    """

    def __init__(
        self,
        # 位置滤波参数
        pos_min_cutoff=0.1,
        pos_beta=0.05,

        # 姿态滤波参数
        rot_min_cutoff=0.3,
        rot_beta=0.02,

        # 速度滤波参数（6D）
        vel_min_cutoff=0.2,
        vel_beta=0.1,
    ):
        super().__init__()

        print(
            f"{TerminalColor.OKBLUE}"
            f"[FilteredXrClient] 滤波开启\n"
            f"  pos(min_cutoff={pos_min_cutoff}, beta={pos_beta})\n"
            f"  rot(min_cutoff={rot_min_cutoff}, beta={rot_beta})\n"
            f"  vel(min_cutoff={vel_min_cutoff}, beta={vel_beta})"
            f"{TerminalColor.ENDC}"
        )

        # ---- 各类滤波参数 ----
        self.pos_filter_cfg = dict(min_cutoff=pos_min_cutoff, beta=pos_beta, d_cutoff=1.0)
        self.rot_filter_cfg = dict(min_cutoff=rot_min_cutoff, beta=rot_beta, d_cutoff=1.0)
        self.vel_filter_cfg = dict(min_cutoff=vel_min_cutoff, beta=vel_beta, d_cutoff=1.0)

        # ---- 滤波器实例池 ----
        self.filters = {}

        # ---- 调试用：保存原始数据 ----
        self.last_raw_pose = {}
        self.last_raw_vel = {}

    # ------------------------------------------------
    # 工具函数：获取或创建滤波器
    # ------------------------------------------------
    def _get_filter(self, key, cfg):
        if key not in self.filters:
            self.filters[key] = OneEuroFilter(**cfg)
        return self.filters[key]

    # ------------------------------------------------
    # Pose 接口
    # ------------------------------------------------
    def get_pose_by_name(self, name: str) -> np.ndarray:
        raw_pose = np.asarray(super().get_pose_by_name(name), dtype=float)
        self.last_raw_pose[name] = raw_pose.copy()

        # 非控制器不滤波
        if name not in ["left_controller", "right_controller"]:
            return raw_pose

        t = time.time()

        pos = raw_pose[:3]
        rot = raw_pose[3:7]

        # ---- 位置滤波 ----
        pos_filter = self._get_filter(f"{name}_pos", self.pos_filter_cfg)
        f_pos = pos_filter.process(pos, t)

        # ---- 姿态滤波 ----
        rot_filter = self._get_filter(f"{name}_rot", self.rot_filter_cfg)
        f_rot = rot_filter.process(rot, t)

        # ---- 四元数归一化保护 ----
        norm = np.linalg.norm(f_rot)
        if norm > 1e-8:
            f_rot = f_rot / norm
        else:
            # 退化保护：使用原始姿态
            f_rot = rot.copy()

        return np.concatenate((f_pos, f_rot))

    # ------------------------------------------------
    # Velocity 接口（6D，按位置方式滤波）
    # ------------------------------------------------
    def get_velocity_by_name(self, name: str) -> np.ndarray:
        raw_vel = np.asarray(super().get_velocity_by_name(name), dtype=float)
        self.last_raw_vel[name] = raw_vel.copy()

        if name not in ["left_controller", "right_controller"]:
            return raw_vel

        t = time.time()

        vel_filter = self._get_filter(f"{name}_vel", self.vel_filter_cfg)
        f_vel = vel_filter.process(raw_vel, t)

        return f_vel
