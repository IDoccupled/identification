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

def safe_percent_error(estimate, truth, min_abs_truth=1e-2):
    """Relative error with denominator floor to avoid exploding percentages near zero."""
    denom = np.maximum(np.abs(truth), float(min_abs_truth))
    return (estimate - truth) / denom * 100.0

def identifiable_projection_matrix(X, tol_ratio=1e-10):
    """Projector onto row-space(X), i.e. the identifiable parameter subspace."""
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        return np.zeros((X.shape[1], X.shape[1])), 0
    tol = s[0] * tol_ratio
    rank = int(np.sum(s > tol))
    v_r = vt[:rank, :].T
    return v_r @ v_r.T, rank

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
        o = np.column_stack((X, Y))

        # 使用 OLS 作为默认热启动，可避免无信息初始化导致的零解坍塌
        theta_ols = np.linalg.lstsq(X, Y, rcond=None)[0]

        self.w_x = np.ones(self.d) if w_x_init is None else np.asarray(w_x_init, dtype=float).copy()
        self.w_z = theta_ols.copy() if w_z_init is None else np.asarray(w_z_init, dtype=float).copy()

        # 初始化噪声方差
        self.psi_x = np.ones(self.d) * 0.4 if psi_x_init is None else psi_x_init
        self.psi_z = np.ones(self.d) * 0.4 if psi_z_init is None else psi_z_init
        self.psi_y = 0.2 if psi_y_init is None else psi_y_init

        # 初始化贝叶斯精度超参数 alpha
        self.alpha = np.ones(self.d) * 1.0

        eps = 1e-10

        for it in range(self.max_iter):
            # =========================================================
            # 一、 E步: 利用当前参数推导隐变量后验
            # =========================================================
            Sigma_oo = np.zeros((self.d + 1, self.d + 1))
            Sigma_oo[0:self.d, 0:self.d] = np.diag(self.w_x**2 + self.psi_x)
            cross_term = self.w_x * self.w_z
            Sigma_oo[0:self.d, self.d] = cross_term
            Sigma_oo[self.d, 0:self.d] = cross_term
            Sigma_oo[self.d, self.d] = np.sum(self.w_z**2 + self.psi_z) + self.psi_y

            Sigma_ho = np.zeros((2 * self.d, self.d + 1))
            Sigma_ho[0:self.d, 0:self.d] = np.diag(self.w_x)
            Sigma_ho[0:self.d, self.d] = self.w_z
            Sigma_ho[self.d:2*self.d, 0:self.d] = np.diag(cross_term)
            Sigma_ho[self.d:2*self.d, self.d] = self.w_z**2 + self.psi_z

            Sigma_hh = np.zeros((2 * self.d, 2 * self.d))
            Sigma_hh[0:self.d, 0:self.d] = np.eye(self.d)
            Sigma_hh[self.d:2*self.d, self.d:2*self.d] = np.diag(self.w_z**2 + self.psi_z)
            Sigma_hh[0:self.d, self.d:2*self.d] = np.diag(self.w_z)
            Sigma_hh[self.d:2*self.d, 0:self.d] = np.diag(self.w_z)

            Sigma_oo_inv = np.linalg.inv(Sigma_oo)
            Cov_h = Sigma_hh - Sigma_ho @ Sigma_oo_inv @ Sigma_ho.T
            Cov_tt = Cov_h[0:self.d, 0:self.d]
            Cov_zz = Cov_h[self.d:2*self.d, self.d:2*self.d]
            Cov_tz = Cov_h[0:self.d, self.d:2*self.d]

            E_h = (Sigma_ho @ Sigma_oo_inv @ o.T).T
            E_t = E_h[:, 0:self.d]
            E_z = E_h[:, self.d:2*self.d]

            E_tt_sum = Cov_tt * N + E_t.T @ E_t
            E_zz_sum = Cov_zz * N + E_z.T @ E_z
            E_tz_sum = Cov_tz * N + E_t.T @ E_z

            # =========================================================
            # 二、 M步: 更新 w_x, w_z, psi_x, psi_z, psi_y, alpha
            # =========================================================
            old_w_z = self.w_z.copy()

            for m in range(self.d):
                ett = E_tt_sum[m, m]
                etz = E_tz_sum[m, m]
                xet = np.sum(X[:, m] * E_t[:, m])

                self.w_x[m] = xet / (ett + self.alpha[m] * self.psi_x[m] + eps)
                self.w_z[m] = etz / (ett + self.alpha[m] * self.psi_z[m] + eps)

                self.psi_x[m] = (
                    np.sum(X[:, m]**2)
                    - 2.0 * self.w_x[m] * xet
                    + self.w_x[m]**2 * ett
                ) / N

                self.psi_z[m] = (
                    E_zz_sum[m, m]
                    - 2.0 * self.w_z[m] * etz
                    + self.w_z[m]**2 * ett
                ) / N

                self.alpha[m] = 2.0 / (self.w_x[m]**2 + self.w_z[m]**2 + eps)

            psi_y_sum = (
                np.sum(Y**2)
                - 2.0 * np.sum(Y * np.sum(E_z, axis=1))
                + N * np.sum(Cov_zz)
                + np.sum(np.sum(E_z, axis=1)**2)
            )
            self.psi_y = psi_y_sum / N

            # 数值稳定性保护
            self.psi_x = np.maximum(self.psi_x, eps)
            self.psi_z = np.maximum(self.psi_z, eps)
            self.psi_y = float(max(self.psi_y, eps))

            if np.max(np.abs(self.w_z - old_w_z)) < self.tol:
                print(f"EM 算法在第 {it} 次迭代收敛")
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
    X_noisy = X_clean + np.random.normal(0, 0.2, (N, d)) # 关节角度加速度等含有 0.2 噪声
    Y_noisy = Y_clean + np.random.normal(0, 0.1, N)     # 力矩传感器含有 0.1 噪声
    
    # 5. 传统方法：普通最小二乘法 OLS 求解
    theta_ols = np.linalg.lstsq(X_noisy, Y_noisy, rcond=None)[0]
    
    # 6. 本文方法：贝叶斯因子分析去噪回归求解
    model = BayesianJFA(max_iter=100000, tol=1e-4)
    model.fit(X_noisy, Y_noisy)
    theta_bayes = model.get_b_hat()
    
    # 7. 打印对比结果
    print("================ 参数辨识结果 ==================")
    print(f"真实的动力学参数 (True Theta):\n{format_array(theta_true)}")
    print(f"传统普通最小二乘法 (OLS) 辨识结果:\n{format_array(theta_ols)}")
    # print(f"与真实参数的误差 (OLS Error):\n{format_array(theta_ols - theta_true)}")
    print(f"论文贝叶斯去噪回归方法 辨识结果:\n{format_array(theta_bayes)}")
    # print(f"与真实参数的误差 (Bayesian Error):\n{format_array(theta_bayes - theta_true)}")
    print("================================================")
    # print(f"贝叶斯超参数 (Alpha):\n{format_array(model.alpha)}")
    # print(f"输入噪声方差 (Psi_x):\n{format_array(model.psi_x)}")
    # print(f"内部解耦变量噪声方差 (Psi_z):\n{format_array(model.psi_z)}")
    # print(f"输出噪声方差 (Psi_y):\n{format_array(model.psi_y)}")
    # print(f"映射权重矩阵 (W_z):\n{format_array(model.w_z)}")
    # print(f"映射权重矩阵 (W_x):\n{format_array(model.w_x)}")
    # print(f"W_z/W_x:\n{format_array(model.w_z / (model.w_x + 1e-8))}")
    # print("================================================")

def main():
    np.random.seed(42)
    regressor = TargetLimbRegressor()
    N_HARMONICS = 5
    fourier_traj = FourierTrajectory(regressor=regressor)


    import time
    start_time = time.time()


    q, v, a = fourier_traj.generate_trajectory(
        coeffs=np.random.uniform(-1.0, 1.0, size=(regressor.dof * (N_HARMONICS * 2 + 2)))
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

    # 60 维参数中并非都可辨识：后续比较应在 row-space(X) 中进行
    proj_ident, rank_x = identifiable_projection_matrix(X_true)
    theta_true_ident = proj_ident @ theta_true

    print("X_true shape:", X_true.shape)
    print("Y_true shape:", Y_true.shape)
    print("theta_true Shape:", theta_true.shape)
    print(f"X_true rank: {rank_x}/{X_true.shape[1]}")

    print(f"生成矩阵耗时: {time.time() - start_time:.2f} 秒")

    N, d = X_true.shape
    X_noisy = X_true + np.random.normal(0, 0.1, (N, d))
    Y_noisy = Y_true + np.random.normal(0, 0.2, N)

    theta_ols = np.linalg.lstsq(X_noisy, Y_noisy, rcond=None)[0]
    model = BayesianJFA(max_iter=100000, tol=1e-4)
    model.fit(X_noisy, Y_noisy,
              w_x_init=np.ones(d),
              # OLS 热启动比直接用 pi_aug 更稳定，能显著降低收缩到零解的概率
              w_z_init=theta_ols.copy(),
              psi_x_init=np.ones(d) * 0.1,
              psi_z_init=np.ones(d) * 0.1,
              psi_y_init=0.2)
    theta_bayes = model.get_b_hat()

    theta_ols_ident = proj_ident @ theta_ols
    theta_bayes_ident = proj_ident @ theta_bayes
    rmse_ols = np.sqrt(np.mean((Y_noisy - X_noisy @ theta_ols)**2))
    rmse_bayes = np.sqrt(np.mean((Y_noisy - X_noisy @ theta_bayes)**2))

    print("================ 参数辨识结果 ==================")
    print(f"真实参数的可辨识分量 P@theta_true:\n{format_array(theta_true_ident)}")
    print(f"OLS 在可辨识子空间的结果 P@theta_ols:\n{format_array(theta_ols_ident)}")
    print(f"Bayesian 在可辨识子空间的结果 P@theta_bayes:\n{format_array(theta_bayes_ident)}")
    print(f"OLS 安全相对误差(对 P@theta_true):\n{format_array(safe_percent_error(theta_ols_ident, theta_true_ident))}%")
    print(f"Bayesian 安全相对误差(对 P@theta_true):\n{format_array(safe_percent_error(theta_bayes_ident, theta_true_ident))}%")
    print(f"OLS 力矩拟合 RMSE: {rmse_ols:.6f}")
    print(f"Bayesian 力矩拟合 RMSE: {rmse_bayes:.6f}")
    print("================================================")
    print(f"贝叶斯超参数 (Alpha):\n{format_array(model.alpha)}")
    print(f"输入噪声方差 (Psi_x):\n{format_array(model.psi_x)}")
    print(f"内部解耦变量噪声方差 (Psi_z):\n{format_array(model.psi_z)}")
    print(f"输出噪声方差 (Psi_y):\n{format_array(model.psi_y)}")
    print(f"映射权重矩阵 (W_z):\n{format_array(model.w_z)}")
    print(f"映射权重矩阵 (W_x):\n{format_array(model.w_x)}")
    print(f"W_z/W_x:\n{format_array(model.w_z / (model.w_x + 1e-8))}")
    print("================================================")

    print("耗时: {:.2f} 秒".format(time.time() - start_time))

    # =========================================================
    


if __name__ == "__main__":
    main()
    # test()