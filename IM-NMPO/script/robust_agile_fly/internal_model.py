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
        self.last_w = None
        self.last_tau = None  # Last total torque command applied to the plant.
        self.disturbance_coeff = np.zeros((3, 2))
        self.disturbance_estimate = np.zeros(3)
        self.filtered_residual = np.zeros(3)
        self.adaptation_gain = 25.0
        self.residual_filter_alpha = 0.35
        self.low_frequency_injection_gain = 0.9
        self.low_frequency_threshold = 1.0
        self.max_compensation = np.array([1.5, 2.5, 1.5])

        self.rate_last_time = None
        self.rate_disturbance_coeff = np.zeros((3, 2))
        self.rate_disturbance_estimate = np.zeros(3)
        self.rate_filtered_sample = np.zeros(3)
        self.rate_last_compensation = np.zeros(3)
        self.rate_adaptation_gain = 18.0
        self.rate_sample_filter_alpha = 0.15
        self.rate_low_frequency_injection_gain = 0.85
        self.rate_max_compensation = np.array([2.0, 3.0, 1.0])
        
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
        self.last_w = None
        self.last_tau = None
        self.disturbance_coeff = np.zeros((3, 2))
        self.disturbance_estimate = np.zeros(3)
        self.filtered_residual = np.zeros(3)
        self.rate_last_time = None
        self.rate_disturbance_coeff = np.zeros((3, 2))
        self.rate_disturbance_estimate = np.zeros(3)
        self.rate_filtered_sample = np.zeros(3)
        self.rate_last_compensation = np.zeros(3)

    def set_frequencies(self, wsin, keep_rate_memory=True):
        """Update IM frequencies used by the oscillator basis."""
        wsin = np.array(wsin, dtype=float).flatten()
        if wsin.shape[0] != 3:
            return
        if np.allclose(self.wsin, wsin):
            return
        self.wsin = wsin
        self._build_matrices()
        if keep_rate_memory:
            self.rate_disturbance_coeff *= 0.7
        else:
            self.rate_disturbance_coeff = np.zeros((3, 2))
            self.rate_disturbance_estimate = np.zeros(3)
        
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
        """获取内模补偿扭矩."""
        # In the paper d_i = -Psi_i T_i^{-1} theta_i and the control input
        # adds Psi_i T_i^{-1} vartheta_i. When vartheta_i -> theta_i, this
        # term converges to -d_i.
        compensation = (self.PSI @ self.TINV) @ v_im
        return compensation  # 3x1向量

    def update_rate_internal_model(
        self,
        rate_nominal,
        rate_actual,
        rate_correction_without_imc=None,
        current_time=None,
        compensation_gain=1.0
    ):
        """
        Update an internal model in the angular-rate command channel.

        The paper's rotational compensation is defined in the same channel as
        the plant input torque. In the legacy simulator interface, however, the
        matched disturbance is added to the angular-rate command before the
        low-level PID. This method keeps the same IMP exosystem, but estimates
        the disturbance in that rate-command channel so the final controller can
        still use tau_nominal + adaptive_compensation + imc_compensation.
        """
        rate_nominal = np.array(rate_nominal, dtype=float).flatten()
        rate_actual = np.array(rate_actual, dtype=float).flatten()
        if rate_correction_without_imc is None:
            rate_correction_without_imc = np.zeros(3)
        else:
            rate_correction_without_imc = np.array(
                rate_correction_without_imc,
                dtype=float
            ).flatten()

        if current_time is None:
            current_time = rospy.Time.now().to_sec()

        if self.rate_last_time is None:
            dt = 0.02
        else:
            dt = max(0.002, min(0.05, current_time - self.rate_last_time))

        applied_imc = compensation_gain * self.rate_last_compensation
        disturbance_sample = (
            rate_actual
            - rate_nominal
            - rate_correction_without_imc
            - applied_imc
        )
        self.rate_filtered_sample = (
            self.rate_sample_filter_alpha * self.rate_filtered_sample
            + (1.0 - self.rate_sample_filter_alpha) * disturbance_sample
        )

        for i in range(3):
            if abs(self.wsin[i]) < 1e-6:
                basis = np.array([1.0, 0.0])
            else:
                basis = np.array([
                    np.sin(self.wsin[i] * current_time),
                    np.cos(self.wsin[i] * current_time)
                ])

            disturbance_hat = self.rate_disturbance_coeff[i] @ basis
            error = self.rate_filtered_sample[i] - disturbance_hat
            normalizer = 1.0 + basis @ basis
            self.rate_disturbance_coeff[i] += (
                self.rate_adaptation_gain * dt * basis * error / normalizer
            )
            disturbance_hat = self.rate_disturbance_coeff[i] @ basis

            if abs(self.wsin[i]) < self.low_frequency_threshold:
                disturbance_hat += self.rate_low_frequency_injection_gain * (
                    self.rate_filtered_sample[i] - disturbance_hat
                )

            self.rate_disturbance_estimate[i] = disturbance_hat

        compensation = -np.clip(
            self.rate_disturbance_estimate,
            -self.rate_max_compensation,
            self.rate_max_compensation
        )

        self.rate_last_time = current_time
        self.rate_last_compensation = compensation.copy()

        return compensation.reshape(3, 1)

    def _oscillator_state_from_coeff(self, coeff, current_time):
        """Return six IM states from per-axis sin/cos coefficients."""
        eta = np.zeros(6)
        for i in range(3):
            c_sin, c_cos = coeff[i]
            if abs(self.wsin[i]) < 1e-6:
                eta[2 * i] = c_sin
                eta[2 * i + 1] = c_cos
            else:
                phase = self.wsin[i] * current_time
                eta[2 * i] = c_sin * np.sin(phase) + c_cos * np.cos(phase)
                eta[2 * i + 1] = c_sin * np.cos(phase) - c_cos * np.sin(phase)
        return eta

    def get_rate_internal_model_state(self, current_time=None):
        """
        Return the six internal-model states used by the rate-channel IMC.

        Components 1, 3 and 5 are the disturbance output estimates for x, y
        and z. Components 2, 4 and 6 are the quadrature oscillator states.
        """
        if current_time is None:
            current_time = rospy.Time.now().to_sec()
        return self._oscillator_state_from_coeff(
            self.rate_disturbance_coeff,
            current_time
        )

    def get_rotational_internal_model_state(self, current_time=None):
        """Return the six internal-model states used by the torque-channel IMC."""
        if current_time is None:
            current_time = rospy.Time.now().to_sec()
        return self._oscillator_state_from_coeff(
            self.disturbance_coeff,
            current_time
        )

    def update_rotational_internal_model(self, tau_without_imc, w_actual, current_time=None, compensation_gain=1.0):
        """
        Update the rotational internal model and return torque compensation.

        The residual follows the rotational dynamics:
            d_tau = J*w_dot + w x Jw - tau_applied
        and is projected onto the known sinusoidal internal model basis. For
        very low frequencies, a residual-error injection is used in the
        observer output so the finite-time simulation does not have to wait
        for a full low-frequency period before compensation appears.
        """
        tau_without_imc = np.array(tau_without_imc, dtype=float).flatten()
        w_actual = np.array(w_actual, dtype=float).flatten()

        if current_time is None:
            current_time = rospy.Time.now().to_sec()

        if self.last_time is None or self.last_w is None or self.last_tau is None:
            compensation = -np.clip(
                self.disturbance_estimate,
                -self.max_compensation,
                self.max_compensation
            )
            self.last_time = current_time
            self.last_w = w_actual.copy()
            self.last_tau = tau_without_imc + compensation_gain * compensation
            return compensation.reshape(3, 1)

        dt = max(0.002, min(0.05, current_time - self.last_time))
        w_dot = (w_actual - self.last_w) / dt
        w_mid = 0.5 * (self.last_w + w_actual)
        sample_time = 0.5 * (self.last_time + current_time)
        residual = self._J @ w_dot + np.cross(w_mid, self._J @ w_mid) - self.last_tau
        self.filtered_residual = (
            self.residual_filter_alpha * self.filtered_residual
            + (1.0 - self.residual_filter_alpha) * residual
        )

        for i in range(3):
            basis = np.array([
                np.sin(self.wsin[i] * sample_time),
                np.cos(self.wsin[i] * sample_time)
            ])
            disturbance_hat = self.disturbance_coeff[i] @ basis
            error = self.filtered_residual[i] - disturbance_hat
            normalizer = 1.0 + basis @ basis
            self.disturbance_coeff[i] += self.adaptation_gain * dt * basis * error / normalizer
            disturbance_hat = self.disturbance_coeff[i] @ basis
            if abs(self.wsin[i]) < self.low_frequency_threshold:
                disturbance_hat += self.low_frequency_injection_gain * (
                    self.filtered_residual[i] - disturbance_hat
                )
            self.disturbance_estimate[i] = disturbance_hat

        compensation = -np.clip(
            self.disturbance_estimate,
            -self.max_compensation,
            self.max_compensation
        )

        self.last_time = current_time
        self.last_w = w_actual.copy()
        self.last_tau = tau_without_imc + compensation_gain * compensation

        return compensation.reshape(3, 1)

    def get_observer_state(self):
        return np.array([
            self.disturbance_coeff[0, 0], self.disturbance_coeff[0, 1],
            self.disturbance_coeff[1, 0], self.disturbance_coeff[1, 1],
            self.disturbance_coeff[2, 0], self.disturbance_coeff[2, 1],
        ])
    
    
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
