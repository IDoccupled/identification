import numpy as np
from identification.fourier_trajectory import FourierTrajectory
from identification.target_limb_regressor import TargetLimbRegressor

def format_array(value, per_line=5):
    """Format numeric arrays so that every `per_line` elements break to a new line.

    Produces a compact multi-line string with 3-decimal precision.
    """
    arr = np.asarray(value).ravel()
    # Handle empty
    if arr.size == 0:
        return '[]'

    formatted = [f"{float(x):.3f}" for x in arr]
    groups = [', '.join(formatted[i:i+per_line]) for i in range(0, len(formatted), per_line)]
    if len(groups) == 1:
        return '[' + groups[0] + ']'
    body = ',\n  '.join(groups)
    return '[\n  ' + body + '\n]'

class BayesianJFA:
    def __init__(self, max_iter=100000, tol=1e-4):
        """
        self.d: 输入 X 的维度（即动力学方程中回归向量的特征数）
        """
        self.max_iter = max_iter
        self.tol = tol              

    def fit(self, X, Y, 
            w_x_init=None, 
            w_z_init=None,
            psi_x_init=None,
            psi_z_init=None,
            psi_y_init=None):
        
        N, self.d = X.shape
        o = np.column_stack((X, Y)) # 联合观测矩阵 [X, Y], 形状为 (N, self.d+1)

        self.w_x = np.ones(self.d) if w_x_init is None else w_x_init
        self.w_z = np.ones(self.d) if w_z_init is None else w_z_init
        
        # 初始化噪声方差 (Variances of noise)
        self.psi_x = np.ones(self.d) * 0.4 if psi_x_init is None else psi_x_init
        self.psi_z = np.ones(self.d) * 0.4 if psi_z_init is None else psi_z_init
        self.psi_y = 0.2 if psi_y_init is None else psi_y_init
        
        # 初始化贝叶斯精度超参数 alpha (用于维度自动筛选)
        self.alpha = np.ones(self.d) * 1.0
        
        for it in range(self.max_iter):
            # =========================================================
            # 一、 E步 (Expectation Step): 利用当前参数推导隐变量的后验分布
            # =========================================================
            # 论文中隐变量组合向量 h = [t^T, z^T]^T, 维度为 2d
            # 观测变量组合向量 o_i = [x_i^T, y_i]^T, 维度为 self.d+1
            
            # 1. 构造观测变量的联合共现协方差矩阵 Sigma_oo (维度: (self.d+1) x (self.d+1))
            Sigma_oo = np.zeros((self.d + 1, self.d + 1))
            Sigma_oo[0:self.d, 0:self.d] = np.diag(self.w_x**2 + self.psi_x)
            cross_term = self.w_x * self.w_z
            Sigma_oo[0:self.d, self.d] = cross_term
            Sigma_oo[self.d, 0:self.d] = cross_term
            Sigma_oo[self.d, self.d] = np.sum(self.w_z**2 + self.psi_z) + self.psi_y
            
            # 2. 构造隐变量与观测变量的互协方差矩阵 Sigma_ho (维度: 2d x (self.d+1))
            Sigma_ho = np.zeros((2 * self.d, self.d + 1))
            Sigma_ho[0:self.d, 0:self.d] = np.diag(self.w_x)
            Sigma_ho[0:self.d, self.d] = self.w_z
            Sigma_ho[self.d:2*self.d, 0:self.d] = np.diag(cross_term)
            Sigma_ho[self.d:2*self.d, self.d] = self.w_z**2 + self.psi_z
            
            # 3. 构造隐变量自身的先验协方差矩阵 Sigma_hh (维度: 2d x 2d)
            Sigma_hh = np.zeros((2 * self.d, 2 * self.d))
            Sigma_hh[0:self.d, 0:self.d] = np.eye(self.d)
            Sigma_hh[self.d:2*self.d, self.d:2*self.d] = np.diag(self.w_z**2 + self.psi_z)
            Sigma_hh[0:self.d, self.d:2*self.d] = np.diag(self.w_z)
            Sigma_hh[self.d:2*self.d, 0:self.d] = np.diag(self.w_z)
            
            # 4. 根据多元高斯条件分布公式求取后验
            Sigma_oo_inv = np.linalg.inv(Sigma_oo)
            
            # 隐变量后验协方差 Cov(h|o)
            Cov_h = Sigma_hh - Sigma_ho @ Sigma_oo_inv @ Sigma_ho.T
            Cov_tt = Cov_h[0:self.d, 0:self.d]
            Cov_zz = Cov_h[self.d:2*self.d, self.d:2*self.d]
            Cov_tz = Cov_h[0:self.d, self.d:2*self.d]
            
            # 批量计算所有样本的隐变量后验期望值 E[h|o] (维度: N x 2d)
            E_h = (Sigma_ho @ Sigma_oo_inv @ o.T).T
            E_t = E_h[:, 0:self.d]       # 理想输入 t 的期望
            E_z = E_h[:, self.d:2*self.d]     # 中间解耦变量 z 的期望
            
            # 计算M步所需的二阶矩累加项
            E_tt_sum = Cov_tt * N + E_t.T @ E_t
            E_zz_sum = Cov_zz * N + E_z.T @ E_z
            E_tz_sum = Cov_tz * N + E_t.T @ E_z
            
            # =========================================================
            # 二、 M步 (Maximization Step): 极大化期望对数似然，更新模型参数
            # =========================================================
            old_w_z = self.w_z.copy()
            
            for m in range(self.d):
                # 结合超参数 alpha 更新映射权重矩阵 w_x 和 w_z
                self.w_x[m] = np.sum(X[:, m] * E_t[:, m]) / (E_tt_sum[m, m] + self.alpha[m] * self.psi_x[m])
                self.w_z[m] = E_tz_sum[m, m] / (E_tt_sum[m, m] + self.alpha[m] * self.psi_z[m])
                
                # 更新各个维度的输入和内部输出噪声方差
                self.psi_x[m] = (np.sum(X[:, m]**2) - 2 * self.w_x[m] * np.sum(X[:, m] * E_t[:, m]) + self.w_x[m]**2 * E_tt_sum[m, m]) / N
                self.psi_z[m] = (E_zz_sum[m, m] - 2 * self.w_z[m] * E_tz_sum[m, m] + self.w_z[m]**2 * E_tt_sum[m, m]) / N
                
                # 更新贝叶斯超参数 alpha 
                # 如果某个维度 m 的回归项无贡献，w_x 和 w_z 趋于 0，则 alpha 趋于无穷大（收缩压制冗余维度）
                self.alpha[m] = 2.0 / (self.w_x[m]**2 + self.w_z[m]**2 + 1e-8)
            
            # 更新输出力矩总体噪声方差 psi_y
            psi_y_sum = np.sum(Y**2) - 2 * np.sum(Y * np.sum(E_z, axis=1)) + N * np.sum(Cov_zz) + np.sum(np.sum(E_z, axis=1)**2)
            self.psi_y = psi_y_sum / N
            
            # 检查收敛
            if np.max(np.abs(self.w_z - old_w_z)) < self.tol:
                print(f"EM 算法在第 {it} 次迭代收敛。")
                break

    def get_b_hat(self):
        """
        三、 参数变换: 对应论文中的公式 (4)
        利用学到的隐空间映射 w_x, w_z 变换回针对物理无噪输入的真实回归系数 theta
        """
        W_z = np.diag(self.w_z)
        W_x_inv = np.diag(1.0 / (self.w_x + 1e-8))
        Psi_z_inv = np.diag(1.0 / (self.psi_z + 1e-8))
        
        ones = np.ones((self.d, 1))
        C = (ones @ ones.T) / self.psi_y + Psi_z_inv
        C_inv = np.linalg.inv(C)
        
        numerator = self.psi_y * (ones.T @ C_inv)
        denominator = self.psi_y - (ones.T @ C_inv @ ones)
        factor = numerator / denominator
        
        # 严格执行论文的公式(4): \hat{b}_true = 系数 * \Psi_z^-1 * W_z^T * W_x^-1
        b_true = factor @ Psi_z_inv @ W_z.T @ W_x_inv
        return b_true.flatten()
    
def test():
    np.random.seed(42)
    N = 10000  # 采集样本数
    d = 15     # 回归矩阵未知参数维度
    
    # 1. 模拟物理本质：生成无噪声的理想状态 T
    T_clean = np.random.normal(0, 1, (N, d))
    
    # 2. 设定真实的机械臂物理动力学参数 (即我们要辨识的目标)
    theta_true = np.array([2.5, -1.2, 4.0, 3.0, -2.0, 1.5, -0.5, 2.0, 1.0, -1.0, 0.5, -0.8, 1.2, 0.9, -1.1])  # 真实的动力学参数
    
    # 3. 构造满足因果映射的真实数据
    w_x_true = np.array([2.0, 1.5, 5.0, 3.0, 2.5, 1.0, 0.5, 4.0, 3.5, 2.0, 1.0, 0.8, 1.2, 0.9, 0.3]) 
    w_z_true = theta_true * w_x_true      # 理论比值等于 theta_true
    
    X_clean = T_clean * w_x_true
    Y_clean = np.sum(T_clean * w_z_true, axis=1)
    
    # 4. 模拟现实：给输入和输出同时加上传感器/差分噪声 (Errors-in-Variables)
    X_noisy = X_clean + np.random.normal(0, 0.4, (N, d)) # 关节角度加速度等含有 0.2 噪声
    Y_noisy = Y_clean + np.random.normal(0, 0.2, N)     # 力矩传感器含有 0.1 噪声
    
    # 5. 传统方法：普通最小二乘法 OLS 求解
    theta_ols = np.linalg.inv(X_noisy.T @ X_noisy) @ X_noisy.T @ Y_noisy
    
    # 6. 本文方法：贝叶斯因子分析去噪回归求解
    model = BayesianJFA(max_iter=100000, tol=1e-4)
    model.fit(X_noisy, Y_noisy)
    theta_bayes = model.get_b_hat()
    
    # 7. 打印对比结果
    print("================ 参数辨识结果 ==================")
    print(f"真实的动力学参数 (True Theta):\n{format_array(theta_true)}")
    print(f"传统普通最小二乘法 (OLS) 辨识结果:\n{format_array(theta_ols)}")
    print(f"与真实参数的误差 (OLS Error):\n{format_array(theta_ols - theta_true)}")
    print(f"论文贝叶斯去噪回归方法 辨识结果:\n{format_array(theta_bayes)}")
    print(f"与真实参数的误差 (Bayesian Error):\n{format_array(theta_bayes - theta_true)}")
    print("================================================")
    print(f"贝叶斯超参数 (Alpha):\n{format_array(model.alpha)}")
    print(f"输入噪声方差 (Psi_x):\n{format_array(model.psi_x)}")
    print(f"内部解耦变量噪声方差 (Psi_z):\n{format_array(model.psi_z)}")
    print(f"输出噪声方差 (Psi_y):\n{format_array(model.psi_y)}")
    print(f"映射权重矩阵 (W_z):\n{format_array(model.w_z)}")
    print(f"映射权重矩阵 (W_x):\n{format_array(model.w_x)}")
    print(f"W_z/W_x:\n{format_array(model.w_z / (model.w_x + 1e-8))}")
    print("================================================")

def main():
    np.random.seed(42)
    regressor = TargetLimbRegressor()
    N_HARMONICS = 5
    q, v, a = FourierTrajectory(regressor=regressor).generate_trajectory(
        coeffs=np.random.uniform(-0.5, 0.5, size=(regressor.dof * (N_HARMONICS * 2 + 2)))
    )
    X_true = np.empty((0, regressor.dof * 12))
    Y_true = np.empty((0,))
    for sample in range(q.shape[1]):
        (Y_aug, tau_aug, 
        pi_aug, pi_inertia, pi_friction,
        q_excess, v_excess, tau_excess, 
        q_excess_normalized, v_excess_normalized, tau_excess_normalized,
        collided
        ) = regressor.compute_regressor(
            q=q[:, sample],
            v=v[:, sample],
            a=a[:, sample]
            )
        X_true = np.vstack((X_true, Y_aug))
        Y_true = np.hstack((Y_true, tau_aug))
        theta_true = pi_aug
    print("X_true shape:", X_true.shape)
    print("Y_true shape:", Y_true.shape)
    print("theta_true Shape:", theta_true.shape)

    N, d = X_true.shape
    X_noisy = X_true + np.random.normal(0, 0.1, (N, d))
    Y_noisy = Y_true + np.random.normal(0, 0.2, N)

    theta_ols = np.linalg.inv(X_noisy.T @ X_noisy) @ X_noisy.T @ Y_noisy
    model = BayesianJFA(max_iter=100000, tol=1e-4)
    model.fit(X_noisy, Y_noisy,
              w_x_init=np.ones(d),
              w_z_init=theta_true.copy(),
              psi_x_init=np.ones(d) * 0.1,
              psi_z_init=np.ones(d) * 0.1,
              psi_y_init=0.2)
    theta_bayes = model.get_b_hat()

    print("================ 参数辨识结果 ==================")
    print(f"真实的动力学参数 (True Theta):\n{format_array(theta_true)}")
    print(f"传统普通最小二乘法 (OLS) 辨识结果:\n{format_array(theta_ols)}")
    print(f"与真实参数的误差 (OLS Error):\n{format_array(theta_ols - theta_true)}")
    print(f"论文贝叶斯去噪回归方法 辨识结果:\n{format_array(theta_bayes)}")
    print(f"与真实参数的误差 (Bayesian Error):\n{format_array(theta_bayes - theta_true)}")
    print("================================================")
    print(f"贝叶斯超参数 (Alpha):\n{format_array(model.alpha)}")
    print(f"输入噪声方差 (Psi_x):\n{format_array(model.psi_x)}")
    print(f"内部解耦变量噪声方差 (Psi_z):\n{format_array(model.psi_z)}")
    print(f"输出噪声方差 (Psi_y):\n{format_array(model.psi_y)}")
    print(f"映射权重矩阵 (W_z):\n{format_array(model.w_z)}")
    print(f"映射权重矩阵 (W_x):\n{format_array(model.w_x)}")
    print(f"W_z/W_x:\n{format_array(model.w_z / (model.w_x + 1e-8))}")
    print("================================================")


if __name__ == "__main__":
    main()