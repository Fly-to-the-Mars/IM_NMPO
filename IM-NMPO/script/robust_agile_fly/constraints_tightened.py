import numpy as np
import casadi as ca
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

class ConstraintContractor:
    """
    约束收缩器 - 将原始系统约束收缩为标称系统约束
    理论：𝕏̂ = 𝕏 ⊖ 𝕍₀, ℤ̂ = ℤ ⊖ Δ
    """
    
    def __init__(self, J, epsilon=0.1):
        """
        初始化约束收缩器
        
        参数:
        - J: 转动惯量矩阵 (3x3)
        - epsilon: 收缩因子 (0 < epsilon < 1)
        """
        self.J = J
        self.epsilon = epsilon
        
        # 原始系统约束
        self.omega_xy_max_original = 6.0    # rad/s
        self.omega_z_max_original = 2.0     # rad/s
        self.thrust_min_original = 0.0      # N
        self.thrust_max_original = 6.9      # N
        
        # 初始误差边界（需要根据实际情况调整）
        self.init_error_bound = 0.1
        
        # 收缩后的约束
        self.omega_xy_max_contracted = None
        self.omega_z_max_contracted = None
        self.thrust_min_contracted = None
        self.thrust_max_contracted = None
        
        # 计算收缩约束
        self._compute_contracted_constraints()
        
        print(f"ConstraintContractor initialized:")
        print(f"  Original constraints: ω_xy_max={self.omega_xy_max_original}, ω_z_max={self.omega_z_max_original}")
        print(f"  Original thrust: [{self.thrust_min_original}, {self.thrust_max_original}]")
        print(f"  Contracted constraints: ω_xy_max={self.omega_xy_max_contracted}, ω_z_max={self.omega_z_max_contracted}")
        print(f"  Contracted thrust: [{self.thrust_min_contracted}, {self.thrust_max_contracted}]")
        
    def _compute_contracted_constraints(self):
        """计算收缩后的约束"""
        # 误差集边界（简化为线性收缩）
        # 在实际应用中，应该使用更精确的误差分析
        error_bound_omega = self.init_error_bound * self.epsilon
        
        # 角速度约束收缩
        self.omega_xy_max_contracted = self.omega_xy_max_original * (1 - error_bound_omega)
        self.omega_z_max_contracted = self.omega_z_max_original * (1 - error_bound_omega)
        
        # 推力约束收缩
        thrust_range = self.thrust_max_original - self.thrust_min_original
        contraction_amount = thrust_range * self.epsilon * self.init_error_bound
        self.thrust_min_contracted = self.thrust_min_original + contraction_amount
        self.thrust_max_contracted = self.thrust_max_original - contraction_amount
        
    def update_error_bounds(self, e3_norm, e4_norm, delta_tau_norm):
        """
        根据当前误差更新误差边界
        
        参数:
        - e3_norm: 姿态误差范数
        - e4_norm: 角速度误差范数
        - delta_tau_norm: 自适应补偿扭矩范数
        """
        # 根据当前误差动态调整收缩因子
        # 这里使用一个简化的自适应机制
        max_error = max(e3_norm, e4_norm, delta_tau_norm)
        
        # 动态调整收缩因子：误差越大，收缩越大
        adaptive_epsilon = self.epsilon * (1 + 0.5 * max_error)
        
        # 更新收缩约束
        error_bound_omega = self.init_error_bound * adaptive_epsilon
        
        self.omega_xy_max_contracted = self.omega_xy_max_original * (1 - error_bound_omega)
        self.omega_z_max_contracted = self.omega_z_max_original * (1 - error_bound_omega)
        
        thrust_range = self.thrust_max_original - self.thrust_min_original
        contraction_amount = thrust_range * adaptive_epsilon * self.init_error_bound
        self.thrust_min_contracted = self.thrust_min_original + contraction_amount
        self.thrust_max_contracted = self.thrust_max_original - contraction_amount
        
        return adaptive_epsilon
    
    
    def get_omega_constraints(self):
        """获取角速度约束"""
        omega_max = np.array([
            self.omega_xy_max_contracted,
            self.omega_xy_max_contracted,
            self.omega_z_max_contracted
        ])
        return -omega_max, omega_max
    
    def get_thrust_constraints(self):
        """获取推力约束"""
        return self.thrust_min_contracted, self.thrust_max_contracted
    
    def get_contraction_factor(self, state_error):
        """
        计算收缩因子，基于状态误差
        """
        # 简化的收缩因子计算
        error_norm = np.linalg.norm(state_error)
        return min(0.9, max(0.1, self.epsilon * (1 + error_norm)))
    
    def compute_minkowski_difference(self, A, B):
        """
        计算闵可夫斯基差 A ⊖ B = {x | x + B ⊆ A}
        
        参数:
        - A: 原始约束集（多面体表示）
        - B: 误差集（多面体表示）
        
        返回:
        - C: 收缩后的约束集
        """
        # 简化实现：假设A和B都是超立方体
        A_min, A_max = A  # A = [min, max]
        B_min, B_max = B  # B = [min, max]
        
        # 对于超立方体，A ⊖ B = [A_min - B_min, A_max - B_max]
        C_min = A_min - B_min
        C_max = A_max - B_max
        
        # 确保C是有效的
        if np.any(C_min > C_max):
            raise ValueError("无效的闵可夫斯基差：收缩后约束为空")
        
        return C_min, C_max
    
    def get_error_set_approximation(self, V_e_0, m, dt=0.01):
        """
        估计误差集 𝕍₀ = {x_e ∈ ℝⁿ: V_e(t) ≤ V_e(0) + m}
        
        参数:
        - V_e_0: 初始李雅普诺夫函数值
        - m: 常数
        - dt: 时间步长
        
        返回:
        - error_set: 误差集的近似边界
        """
        # 使用简化的椭球近似
        # 假设 V_e = 0.5 * x_e^T P x_e，其中P是对角矩阵
        max_error_bound = np.sqrt(2 * (V_e_0 + m))
        
        # 对于不同的状态分量，给出不同的误差边界
        # 这里根据经验分配
        error_bounds = {
            'position': 0.1 * max_error_bound,
            'velocity': 0.1 * max_error_bound,
            'attitude': 0.3 * max_error_bound,
            'omega': 0.5 * max_error_bound
        }
        
        return error_bounds
    
    def is_constraint_violated(self, actual_state, nominal_state):
        """
        检查原始系统约束是否被违反
        
        参数:
        - actual_state: 实际系统状态
        - nominal_state: 标称系统状态
        
        返回:
        - violation: 是否违反约束
        - details: 违反详情
        """
        violation = False
        details = {}
        
        # 检查角速度约束
        omega_actual = actual_state[10:13]
        omega_nominal = nominal_state[10:13]
        
        # 检查x,y角速度
        for i in range(2):
            if abs(omega_actual[i]) > self.omega_xy_max_original:
                violation = True
                details[f'omega_{i}_violation'] = abs(omega_actual[i]) - self.omega_xy_max_original
        
        # 检查z角速度
        if abs(omega_actual[2]) > self.omega_z_max_original:
            violation = True
            details['omega_z_violation'] = abs(omega_actual[2]) - self.omega_z_max_original
        
        return violation, details