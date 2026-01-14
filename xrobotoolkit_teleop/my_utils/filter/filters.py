import math
import numpy as np
import time

# ====================================================================
#  极速版 OneEuroFilter (支持向量化运算，耗时 < 0.01ms)
# ====================================================================
class LowPassFilter:
    def __init__(self, alpha=1.0):
        self.y_old = None
        self.alpha = alpha

    def filter(self, value, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        
        # 强制转为 numpy array 以支持 6关节同时运算
        value = np.asarray(value, dtype=float)

        if self.y_old is None:
            self.y_old = value.copy()
            return value
        
        # 核心公式: y = α * x + (1 - α) * y_old
        filtered = self.alpha * value + (1.0 - self.alpha) * self.y_old
        self.y_old = filtered
        return filtered

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None

    def _smoothing_factor(self, dt, cutoff):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def process(self, x, t=None):
        x = np.asarray(x, dtype=float)
        if t is None: t = time.perf_counter() # 使用高精度计时器

        if self.last_time is None:
            self.last_time = t
            return self.x_filter.filter(x)

        dt = t - self.last_time
        self.last_time = t

        # 极端情况保护 (避免除以0)
        if dt <= 1e-5: return self.x_filter.y_old

        # 1. 计算导数 (速度的变化率)
        if self.x_filter.y_old is None:
            dx = 0.0
        else:
            dx = (x - self.x_filter.y_old) / dt
            
        # 2. 对导数进行低通滤波
        edx = self.dx_filter.filter(dx, alpha=self._smoothing_factor(dt, self.d_cutoff))

        # 3. 【核心】动态计算截止频率
        #    速度越快 (abs(edx)越大)，cutoff 越高 -> 延迟越低
        #    速度越慢，cutoff 接近 min_cutoff -> 滤波越强
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        
        # 4. 主滤波
        alpha = self._smoothing_factor(dt, cutoff)
        return self.x_filter.filter(x, alpha=alpha)