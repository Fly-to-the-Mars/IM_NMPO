import numpy as np
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist

class InnerModelCompensator:
    def __init__(self, wsin, J):
        """
        内模补偿器初始化
        
        参数:
        - wsin: 正弦扰动频率 [wx_freq, wy_freq, wz_freq]
        - J: 无人机转动惯量矩阵 (3x3)
        """
        self.wsin = wsin
        self._J = J  # 转动惯量矩阵
        
        # 构建所有矩阵
        self._build_matrices()
        
        # 内模状态 (6维向量)
        self.v_im = np.zeros((6, 1))
        
        # 上次控制时间
        self.last_time = None
        
    def _build_matrices(self):
        """构建内模控制器所需矩阵"""
        # 构建phi矩阵 (每个轴2x2)
        phi = [None] * 3
        for i in range(3):
            phi[i] = np.array([[0, 1], 
                               [-self.wsin[i] ** 2, 0]])
        
        # psi矩阵 (1x2)
        psi = np.array([1, 0])
        
        # m矩阵 (2x2)
        m = np.array([[0, 1], 
                      [-3, -2]])
        
        # n矩阵 (2x1)
        n = np.array([[0], [1]])
        
        # tinv矩阵 (每个轴2x2)
        tinv = [None] * 3
        for i in range(3):
            tinv[i] = np.array([[3 - self.wsin[i] ** 2, 2], 
                                [-2 * self.wsin[i] ** 2, 3 - self.wsin[i] ** 2]])
        
        # 构建分块对角矩阵
        # PHI (6x6) - 扰动态矩阵
        self.PHI = np.block([
            [phi[0], np.zeros((2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), phi[1], np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2)), phi[2]]
        ])
        
        # PSI (3x6) - 输出矩阵
        self.PSI = np.block([
            [psi, np.zeros((1, 2)), np.zeros((1, 2))],
            [np.zeros((1, 2)), psi, np.zeros((1, 2))],
            [np.zeros((1, 2)), np.zeros((1, 2)), psi]
        ])
        
        # M (6x6) - 内模系统矩阵
        self.M = np.block([
            [m, np.zeros((2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), m, np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2)), m]
        ])
        
        # N (6x3) - 内模输入矩阵
        self.N = np.block([
            [n, np.zeros((2, 1)), np.zeros((2, 1))],
            [np.zeros((2, 1)), n, np.zeros((2, 1))],
            [np.zeros((2, 1)), np.zeros((2, 1)),n]  # 注意：这里应该是n而不是zeros
        ])
        
        # TINV (6x6) - 逆变换矩阵
        self.TINV = np.block([
            [tinv[0], np.zeros((2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), tinv[1], np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2)), tinv[2]]
        ])
        
        # 计算(M@N)矩阵乘积，用于内模更新
        self.MN = self.M @ self.N

        
    def reset(self):
        """重置内模状态"""
        self.v_im = np.zeros((6, 1))
        self.last_time = None
        
    def update(self, tau_nominal, w_actual, dt=None):
        """
        更新内模状态
        
        参数:
        - tau_nominal: NMPC计算出的标称扭矩 [3x1] 或 [3,]
        - w_actual: 实际角速度 [wx, wy, wz] [3x1] 或 [3,]
        - dt: 时间步长，如果为None则自动计算
        
        返回:
        - v_im: 更新后的内模状态
        """
        tau_nominal = tau_nominal.reshape(-1, 1) if tau_nominal.ndim == 1 else tau_nominal
        w_actual = w_actual.reshape(-1, 1) if w_actual.ndim == 1 else w_actual
        
        current_time = rospy.Time.now().to_sec()
        if dt is None:
            if self.last_time is None:
                dt = 0.01  # 默认时间步长
            else:
                dt = current_time - self.last_time
                # 限制最大和最小时间步长
                dt = max(0.001, min(0.05, dt))
        self.last_time = current_time
        
        # 内模动力学: v_dot = M@v_im + N@tau - ((M@N)@J)@w
        # 注意: 这里的公式可能是 v_dot = M@v_im + N@tau - (M@N@J)@w
        # 根据你的具体公式调整
        
        # 计算 ((M@N)@J)@w
        disturbance_term = (self.MN @ self._J) @ w_actual
        
        # 计算状态导数
        v_dot = self.M @ self.v_im + self.N @ tau_nominal - disturbance_term
        
        # 欧拉积分更新
        self.v_im += v_dot * dt
        
        return self.v_im
    
    def get_compensation(self,v_im):
        """获取内模补偿量 (角速度补偿)"""
        # 补偿量 = (PSI @ TINV) @ v_im
        compensation = (self.PSI @ self.TINV) @ v_im
        return compensation  # 3x1向量
    
    
class AdaptiveFeedbackCompensator:
    """
    自适应反馈补偿器
    理论公式：δ_τ = -k3 * e3v - K4 * ẽ4 - (- ω× J e4 - e4× J ω + e4× J e4)
    """
    
    def __init__(self, J):
        self.J = J
        self.J_inv = np.linalg.inv(J)
        
        # 增益参数
        self.k3 = 0.5
        
        self.K4 = np.diag([1.0, 1.0, 1.0])
        self.E2 = np.diag([0.5, 0.5, 0.5])

    def compute_delta_tau(self, q_actual, w_actual, q_nominal, w_nominal):
        """
        计算自适应反馈补偿量 δ_τ
        
        参数:
        - q_actual: 实际四元数 [qw, qx, qy, qz]
        - w_actual: 实际角速度 [wx, wy, wz]
        - q_nominal: 标称四元数 [qw, qx, qy, qz]
        - w_nominal: 标称角速度 [wx, wy, wz]
        
        返回:
        - delta_tau: 自适应反馈补偿扭矩 [3x1]
        """
        # 确保输入是numpy数组
        q_actual = np.array(q_actual, dtype=float).flatten()
        w_actual = np.array(w_actual, dtype=float).flatten()
        q_nominal = np.array(q_nominal, dtype=float).flatten()
        w_nominal = np.array(w_nominal, dtype=float).flatten()
        
        # ===== 步骤1: 计算误差状态 e3 和 e4 =====
        # e3 = q̂^{-1} ⊙ q
        e3 = self.quaternion_error(q_nominal, q_actual)
        
        # e4 = ω - ω̂
        e4 = w_actual - w_nominal
        
        # ===== 步骤2: 坐标变换 =====
        # ẽ3 = e3 - q1 (其中q1 = [1, 0, 0, 0])
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        e3_tilde = e3 - q1
        
        # e3v: e3的向量部分
        e3v = e3[1:]
        
        # ẽ4 = e4 + E2 * e3v
        e4_tilde = e4 + self.E2 @ e3v
        
        # ===== 步骤3: 计算非线性项 =====
        # 非线性项: - ω× J e4 - e4× J ω + e4× J e4
        nonlinear_term = self._compute_nonlinear_term(w_actual, e4)
        
        # ===== 步骤4: 计算 δ_τ =====
        # δ_τ = -k3 * e3v - K4 * ẽ4 - nonlinear_term
        delta_tau = -self.k3 * e3v - self.K4 @ e4_tilde - nonlinear_term
        
        return delta_tau
    
    def _compute_nonlinear_term(self, w, e4):
        """
        计算非线性项: - ω× J e4 - e4× J ω + e4× J e4
        
        参数:
        - w: 角速度向量 [3x1]
        - e4: 角速度误差向量 [3x1]
        
        返回:
        - nonlinear_term: 非线性扭矩项 [3x1]
        """
        # 转换为列向量
        w = w.reshape(-1, 1)
        e4 = e4.reshape(-1, 1)
        
        # 计算叉乘矩阵
        w_skew = self._skew_symmetric(w.flatten())
        e4_skew = self._skew_symmetric(e4.flatten())
        
        # 第一项: - ω× J e4
        term1 = -w_skew @ self.J @ e4
        
        # 第二项: - e4× J ω
        term2 = -e4_skew @ self.J @ w
        
        # 第三项: + e4× J e4
        term3 = e4_skew @ self.J @ e4
        
        # 非线性项总和
        nonlinear_term = term1 + term2 + term3
        
        return nonlinear_term.flatten()
    
    def quaternion_error(self, q1, q2):
        """
        计算四元数误差: e = q1^{-1} ⊙ q2
        
        参数:
        - q1: 四元数 [qw, qx, qy, qz]
        - q2: 四元数 [qw, qx, qy, qz]
        
        返回:
        - e: 四元数误差 [qw, qx, qy, qz]
        """
        # 归一化四元数
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # q1的共轭
        q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
        
        # 四元数乘法: q1^{-1} ⊙ q2
        e = self._quaternion_multiply(q1_conj, q2)
        
        return e
    
    
    def _quaternion_multiply(self, q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def _skew_symmetric(self, v):
        """
        计算向量v的叉乘矩阵
        
        参数:
        - v: 3维向量 [vx, vy, vz]
        
        返回:
        - S: 叉乘矩阵 [3x3]
        """
        vx, vy, vz = v
        S = np.array([
            [0, -vz, vy],
            [vz, 0, -vx],
            [-vy, vx, 0]
        ])
        return S
    
    def compute_error_dynamics(self, q_actual, w_actual, q_nominal, w_nominal, dt=0.01):
        """
        计算误差动力学（用于分析和监控）
        
        返回:
        - errors: 包含各种误差的字典
        """
        q_actual = np.array(q_actual, dtype=float).flatten()
        w_actual = np.array(w_actual, dtype=float).flatten()
        q_nominal = np.array(q_nominal, dtype=float).flatten()
        w_nominal = np.array(w_nominal, dtype=float).flatten()
        
        # 计算误差状态
        e3 = self.quaternion_error(q_nominal, q_actual)
        e4 = w_actual - w_nominal
        
        # 坐标变换
        e3v = e3[1:]
        e4_tilde = e4 + self.E2 @ e3v
        
        # 误差大小
        e3_norm = np.linalg.norm(e3v)
        e4_norm = np.linalg.norm(e4)
        e4_tilde_norm = np.linalg.norm(e4_tilde)
        
        errors = {
            'e3': e3,
            'e4': e4,
            'e3v': e3v,
            'e4_tilde': e4_tilde,
            'e3_norm': e3_norm,
            'e4_norm': e4_norm,
            'e4_tilde_norm': e4_tilde_norm
        }
        
        return errors