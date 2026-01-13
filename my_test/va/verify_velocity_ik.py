import numpy as np

# ==========================================
# 1. 模拟环境构建 (Mock Classes)
# ==========================================
class MockPlacoRobot:
    """模拟 Placo 机器人对象，允许手动注入 Jacobian 矩阵"""
    def __init__(self):
        # 默认就是一个单位矩阵，方便口算验证 (J=I, 也就意味着 dq=v)
        self.mock_jacobian = np.eye(6)

    def frame_jacobian(self, frame_name):
        return self.mock_jacobian

    def set_jacobian(self, matrix):
        self.mock_jacobian = np.array(matrix)

class RobotController:
    """包含你待测函数的宿主类"""
    def __init__(self, mock_robot):
        self.placo_robot = mock_robot
        self.manipulator_config = {"left_arm": {"link_name": "end_effector"}}
        self.placo_arm_joint_slice = {"left_arm": slice(0, 6)} # 假设是6轴
        self._dq_last = {}

    def calculate_feedforward_velocity(self, arm_name, controller_vel):
        """
        你的函数 (经过 Bug 修复)
        """
        # --------- 0. 多臂状态隔离初始化 ---------
        if not hasattr(self, "_dq_last"):
            self._dq_last = {}
        if arm_name not in self._dq_last:
            self._dq_last[arm_name] = None
        
        # -------------------------------------------------------------
        # Step 0: 笛卡尔空间死区处理 (Deadzone)
        # -------------------------------------------------------------
        v_in = np.asarray(controller_vel, dtype=float).reshape(-1)
        
        v_linear = v_in[:3]
        v_angular = v_in[3:]
        
        LINEAR_DEADZONE = 0.01 
        ANGULAR_DEADZONE = 0.02  

        if np.linalg.norm(v_linear) < LINEAR_DEADZONE:
            v_linear = np.zeros(3)
        if np.linalg.norm(v_angular) < ANGULAR_DEADZONE:
            v_angular = np.zeros(3)
            
        # [处理后的输入]
        v_cartesian = np.concatenate((v_linear, v_angular))

        # -------------------------------------------------------------
        if np.allclose(v_cartesian, 0.0):
            print("  [Log] 进入死区，直接返回0")
            self._dq_last[arm_name] = np.zeros(6)
            return self._dq_last[arm_name]
        # -------------------------------------------------------------

        try:
            link_name = self.manipulator_config[arm_name]["link_name"]
            J = self.placo_robot.frame_jacobian(link_name)

            # !!! 严重修正 !!! 
            # 原代码在这里写了 v_cartesian = np.asarray(controller_vel)...
            # 这会导致上面的死区处理白做了！必须使用处理过的 v_cartesian
            # -------------------------------------------------
            
            # ---------- 2. SVD 分解 ----------
            U, S, Vt = np.linalg.svd(J, full_matrices=False)

            # ---------- 3. 动态阻尼 DLS ----------
            sigma_min = S[-1]
            lambda_min = 0.005 
            lambda_max = 0.05
            lambda_dls = lambda_min + (lambda_max - lambda_min) * np.exp(-sigma_min / 0.01)

            # ---------- 4. 计算阻尼伪逆 ----------
            S_damped = S / (S**2 + lambda_dls**2)
            J_pinv = Vt.T @ np.diag(S_damped) @ U.T

            dq_full = J_pinv @ v_cartesian # 使用死区处理后的 v

            # ---------- 5. 维度切片 ----------
            if hasattr(self, "placo_arm_joint_slice"):
                dq_arm = dq_full[self.placo_arm_joint_slice[arm_name]]
            else:
                dq_arm = dq_full

            # ---------- 6. 数值安全检查 ----------
            if not np.all(np.isfinite(dq_arm)):
                raise FloatingPointError("NaN/Inf detected")
            
            # 二次死区
            if np.max(np.abs(dq_arm)) < 0.01: 
                dq_arm = np.zeros_like(dq_arm)

            # ---------- 7. 更新缓存 ----------
            self._dq_last[arm_name] = dq_arm
            return dq_arm

        except Exception as e:
            print(f"  [Error] {e}")
            if self._dq_last.get(arm_name) is not None:
                return self._dq_last[arm_name] * 0.8
            else:
                return np.zeros(6)

# ==========================================
# 2. 验证工具函数
# ==========================================
def print_vec(name, vec):
    print(f"  {name}: {np.array2string(vec, precision=4, suppress_small=True)}")

def run_test_case(name, controller, vel_in, desc):
    print(f"\n=== 测试用例: {name} ===")
    print(f"说明: {desc}")
    print_vec("输入速度 VR", np.array(vel_in))
    
    start_t = time.perf_counter()
    dq_out = controller.calculate_feedforward_velocity("left_arm", vel_in)
    end_t = time.perf_counter()
    
    print_vec("输出关节 dq", dq_out)
    print(f"  计算耗时: {(end_t - start_t)*1000:.3f} ms")
    return dq_out

# ==========================================
# 3. 主程序
# ==========================================
import time

if __name__ == "__main__":
    # 初始化
    mock_robot = MockPlacoRobot()
    controller = RobotController(mock_robot)

    # -----------------------------------------------------
    # Case 1: 死区测试 (Deadzone)
    # -----------------------------------------------------
    # 输入一个极小的速度 (0.005 < 0.01), 期望被滤除
    vel_noise = [0.005, 0.0, 0.0,  0.0, 0.0, 0.0]
    run_test_case("噪声过滤", controller, vel_noise, 
                  "输入线速度 0.005 (小于阈值 0.01)，期望输出全 0")

    # -----------------------------------------------------
    # Case 2: 正常 Z 轴向上运动 (Normal Move)
    # -----------------------------------------------------
    # 此时 Jacobian 是单位矩阵，输入 [0,0,1,0,0,0]，期望关节 3 的速度接近 1
    # 实际上由于 DLS 的存在，会略小于 1 (例如 0.999...)
    vel_up = [0.0, 0.0, 1.0,  0.0, 0.0, 0.0]
    run_test_case("Z轴提升", controller, vel_up, 
                  "输入 Z=1.0，Jacobian=Identity，期望 dq[2]≈1.0")

    # -----------------------------------------------------
    # Case 3: 奇异点保护 (Singularity / DLS)
    # -----------------------------------------------------
    # 构造一个奇异 Jacobian：第3行全为0 (代表机器人在 Z 轴方向无法运动)
    # 此时如果我们在 Z 轴给速度，DLS 应该介入，防止 dq 变为无穷大
    singular_J = np.eye(6)
    singular_J[2, 2] = 0.0 # 破坏 Z 轴能力
    mock_robot.set_jacobian(singular_J)
    
    dq_singular = run_test_case("奇异点保护", controller, vel_up, 
                  "Jacobian 缺失 Z 轴能力，强行输入 Z 速度，期望 dq[2] 被 DLS 抑制（接近0），而不是无穷大")

    # -----------------------------------------------------
    # Case 4: 旋转运动 (Rotation)
    # -----------------------------------------------------
    # 恢复 Jacobian
    mock_robot.set_jacobian(np.eye(6))
    vel_rot = [0.0, 0.0, 0.0,  0.0, 0.0, 1.0] # 绕 Z 轴旋转
    run_test_case("偏航旋转", controller, vel_rot, 
                  "输入 Wz=1.0，期望 dq[5]≈1.0")