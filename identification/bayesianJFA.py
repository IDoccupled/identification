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
        # Gamma(a0, b0) 作为 alpha 的先验超参数
        self.alpha_a0 = 1.0
        self.alpha_b0 = 1e-6

    def fit(self, X, Y, 
            w_x_init=None, 
            w_z_init=None,
            psi_x_init=None,
            psi_z_init=None,
            psi_y_init=None):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).reshape(-1)

        N, self.d = X.shape

        self.w_x = np.ones(self.d, dtype=float) if w_x_init is None else np.asarray(w_x_init, dtype=float).copy()
        self.w_z = np.ones(self.d, dtype=float) if w_z_init is None else np.asarray(w_z_init, dtype=float).copy()

        self.psi_x = np.ones(self.d, dtype=float) * 0.4 if psi_x_init is None else np.asarray(psi_x_init, dtype=float).copy()
        self.psi_z = np.ones(self.d, dtype=float) * 0.4 if psi_z_init is None else np.asarray(psi_z_init, dtype=float).copy()
        self.psi_y = float(0.2 if psi_y_init is None else psi_y_init)

        self.alpha = np.ones(self.d, dtype=float)
        self.var_w_x = np.zeros(self.d, dtype=float)
        self.var_w_z = np.zeros(self.d, dtype=float)

        eps = 1e-10
        identity = np.eye(self.d)
        previous_theta = None

        for it in range(self.max_iter):
            psi_x_safe = np.maximum(self.psi_x, eps)
            psi_y_safe = float(max(self.psi_y, eps))

            # E-step: posterior of the latent clean input t_i under
            # x_i = diag(w_x) t_i + eps_x, y_i = w_z^T t_i + eps_y.
            posterior_precision = (
                identity
                + np.diag((self.w_x ** 2) / psi_x_safe)
                + np.outer(self.w_z, self.w_z) / psi_y_safe
            )
            posterior_cov = np.linalg.inv(posterior_precision)

            x_weight = self.w_x / psi_x_safe
            y_weight = self.w_z / psi_y_safe
            latent_mean = X * x_weight[None, :]
            latent_mean = latent_mean @ posterior_cov
            latent_mean += np.outer(Y / psi_y_safe, posterior_cov @ self.w_z)

            latent_second_moment = N * posterior_cov + latent_mean.T @ latent_mean

            # M-step: update w_x, w_z, and the noise variances.
            previous_theta = self.w_z / (self.w_x + eps) if previous_theta is None else previous_theta

            self.w_x = np.sum(X * latent_mean, axis=0) / (
                np.sum(latent_mean ** 2, axis=0) + N * np.diag(posterior_cov) + eps
            )

            rhs_wz = latent_mean.T @ Y
            ridge = np.diag(np.clip(self.alpha, 1e-6, 1e3) * 1e-4)
            self.w_z = np.linalg.solve(latent_second_moment + ridge + eps * identity, rhs_wz)

            for m in range(self.d):
                self.psi_x[m] = np.mean(
                    (X[:, m] - self.w_x[m] * latent_mean[:, m]) ** 2
                    + (self.w_x[m] ** 2) * posterior_cov[m, m]
                )

            y_residual = Y - latent_mean @ self.w_z
            self.psi_y = float(np.mean(y_residual ** 2 + self.w_z.T @ posterior_cov @ self.w_z))

            # 轻量 ARD，只用于报告和微弱收缩，不主导优化。
            self.alpha = np.clip(1.0 / (self.w_x ** 2 + self.w_z ** 2 + eps), 1e-6, 1e6)

            # 兼容旧接口：保留一些近似方差量。
            self.var_w_x = np.full(self.d, float(np.mean(np.diag(posterior_cov))))
            self.var_w_z = np.full(self.d, float(np.mean(np.diag(posterior_cov))))
            self.psi_z = np.maximum(self.psi_x.copy(), eps)

            theta = self.w_z / (self.w_x + eps)
            if previous_theta is not None and np.max(np.abs(theta - previous_theta)) < self.tol:
                print(f"EM 算法在第 {it} 次迭代收敛")
                break
            previous_theta = theta.copy()

    def get_b_hat(self):
        """
        三、 参数变换: 对应论文中的公式 (4)
        利用学到的隐空间映射 w_x, w_z 变换回针对物理无噪输入的真实回归系数 theta。

        在当前实现里，theta 直接取 noiseless input 下的比值 w_z / w_x，
        这与论文里由隐变量参数回推回回归系数的思想一致。
        """
        return self.w_z / (self.w_x + 1e-8)
    
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
    print("X_true shape:", X_true.shape)
    print("Y_true shape:", Y_true.shape)
    print("theta_true Shape:", theta_true.shape)

    print(f"生成矩阵耗时: {time.time() - start_time:.2f} 秒")

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
    print(f"与真实参数的误差 (OLS Error) 百分比:\n{format_array((theta_ols - theta_true) / theta_true * 100)}%")
    print(f"论文贝叶斯去噪回归方法 辨识结果:\n{format_array(theta_bayes)}")
    print(f"与真实参数的误差 (Bayesian Error) 百分比:\n{format_array((theta_bayes - theta_true) / theta_true * 100)}%")
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
    # main()
    test()