import numpy as np
from scipy.linalg import inv

class BayesianJFA:
    """
    Bayesian Joint Factor Analysis for noisy input/output regression.
    Implements the EM algorithm described in Ting et al. (2006).
    Assumes a single output variable y.
    """

    def __init__(self, d, max_iter=100, tol=1e-4, a_alpha0=1e-6, b_alpha0=1e-6):
        """
        d: number of input dimensions (features)
        max_iter: maximum EM iterations
        tol: tolerance for log-likelihood change
        a_alpha0, b_alpha0: Gamma prior hyperparameters for alpha_m
        """
        self.d = d
        self.max_iter = max_iter
        self.tol = tol
        self.a_alpha0 = a_alpha0
        self.b_alpha0 = b_alpha0

    def fit(self, X, y):
        """
        X: (N, d) noisy input matrix
        y: (N, 1) noisy output vector
        """
        N, d = X.shape
        assert d == self.d

        # ----- 1. Initialization -----
        psi_y = np.var(y) * 0.5                 # output noise variance
        psi_z = np.ones(d) * 0.1                # noise var for z
        psi_x = np.ones(d) * 0.1                # noise var for x
        Wz = np.random.randn(d) * 0.1           # wz (d,)
        Wx = np.random.randn(d) * 0.1           # wx (d,)
        alpha = np.ones(d) * 10.0               # precision

        # Hyperparameters for Gamma prior
        a_alpha = np.ones(d) * self.a_alpha0
        b_alpha = np.ones(d) * self.b_alpha0

        # Storage for lower bound (optional)
        self.elbo_ = []

        for iteration in range(self.max_iter):
            # ----- 2. E-step: compute posterior statistics of Z and T -----
            # Build diagonal matrices
            Psi_z = np.diag(psi_z)
            Psi_x = np.diag(psi_x)
            Wx_diag = np.diag(Wx)
            Wz_diag = np.diag(Wz)

            # K matrix: I + Wx^T Psi_x^{-1} Wx + Wz^T Psi_z^{-1} Wz
            K = np.eye(d) + Wx_diag @ inv(Psi_x) @ Wx_diag + Wz_diag @ inv(Psi_z) @ Wz_diag

            # M matrix: Psi_z + Wz (I + ... )^{-1} Wz^T
            # Actually according to paper: M = Psi_z + Wz * K^{-1} * Wz^T
            # But careful: Wz is diagonal, so M is diagonal
            M_inv = inv(Psi_z + Wz_diag @ inv(K) @ Wz_diag)
            # But easier: compute Sigma_zz directly from paper Eq.(1)
            # Sigma_zz = M - (M 1 1^T M) / (psi_y + 1^T M 1)
            M = Psi_z + Wz_diag @ inv(K) @ Wz_diag
            one = np.ones(d)
            denom = psi_y + one @ M @ one
            Sigma_zz = M - np.outer(M @ one, M @ one) / denom

            # Sigma_zt = - Sigma_zz * Wz * inv(Psi_z) * inv(K)
            Sigma_zt = - Sigma_zz @ Wz_diag @ inv(Psi_z) @ inv(K)

            # Sigma_tt = inv(K) + inv(K) Wz^T inv(Psi_z) Sigma_zz inv(Psi_z) Wz inv(K)
            tmp = inv(K) @ Wz_diag @ inv(Psi_z)
            Sigma_tt = inv(K) + tmp @ Sigma_zz @ tmp.T

            # E-step expectations for each sample
            z_mean = np.zeros((N, d))
            t_mean = np.zeros((N, d))
            zz = np.zeros((N, d))   # diag of <z_i z_i^T>
            tt = np.zeros((N, d))
            zt = np.zeros((N, d))

            for i in range(N):
                xi = X[i]
                yi = y[i]

                # <z_i> = (yi/psi_y) * 1^T Sigma_zz + xi^T Wx^T inv(Psi_x) Sigma_tz
                # where Sigma_tz = Sigma_zt^T
                term1 = (yi / psi_y) * (one @ Sigma_zz)
                term2 = xi @ Wx_diag @ inv(Psi_x) @ Sigma_zt.T
                z_mean[i] = term1 + term2

                # <t_i> = (yi/psi_y) * 1^T Sigma_zz Wz^T inv(Psi_z) inv(K) + xi^T Wx^T inv(Psi_x) Sigma_tt
                term1_t = (yi / psi_y) * (one @ Sigma_zz @ Wz_diag @ inv(Psi_z) @ inv(K))
                term2_t = xi @ Wx_diag @ inv(Psi_x) @ Sigma_tt
                t_mean[i] = term1_t + term2_t

                # variances (diagonal only, for simplicity)
                zz[i] = np.diag(Sigma_zz) + z_mean[i]**2
                tt[i] = np.diag(Sigma_tt) + t_mean[i]**2
                zt[i] = np.diag(Sigma_zt) + z_mean[i] * t_mean[i]

            # ----- 3. M-step: update parameters -----
            # Update psi_y
            resid_y = y**2 - 2*y*(z_mean @ one) + (zz @ one)
            psi_y = np.mean(resid_y)

            # Update psi_z
            wz2 = Wz**2
            psi_z = np.mean(zz - 2*Wz*zt + wz2*tt, axis=0)

            # Update psi_x
            wx2 = Wx**2
            psi_x = np.mean(X**2 - 2*X*t_mean + wx2*tt, axis=0)

            # Update Wz (posterior mean)
            for m in range(d):
                denom = psi_z[m] * np.sum(tt[:, m]) + alpha[m]
                sigma_wz = 1.0 / denom
                Wz[m] = sigma_wz * psi_z[m] * np.sum(zt[:, m])

            # Update Wx
            for m in range(d):
                denom = psi_x[m] * np.sum(tt[:, m]) + alpha[m]
                sigma_wx = 1.0 / denom
                Wx[m] = sigma_wx * psi_x[m] * np.sum(X[:, m] * t_mean[:, m])

            # Update alpha (Gamma prior)
            a_alpha = self.a_alpha0 + 1
            b_alpha = self.b_alpha0 + 0.5*(Wz**2 + Wx**2)
            alpha = a_alpha / b_alpha   # mean of Gamma

            # Optional: compute log-likelihood lower bound (ELBO) for convergence
            # (omitted for brevity, but you can implement it)

            # Check convergence
            if iteration > 0 and abs(old_elbo - current_elbo) < self.tol:
                break
            old_elbo = current_elbo

        self.Wz_ = Wz
        self.Wx_ = Wx
        self.psi_y_ = psi_y
        self.psi_z_ = psi_z
        self.psi_x_ = psi_x
        self.alpha_ = alpha
        return self

    def predict_noiseless(self, X_test):
        """
        Predict output from noiseless input (t) using Eq.(4) in paper.
        Here we use the estimated Wz directly because we want E[y|t] = sum(Wz * t).
        But if you have only noisy X_test, you should first estimate t.
        For simplicity, return X_test @ self.Wz_ (this is for noiseless input).
        """
        return X_test @ self.Wz_
    
def main():
    # 生成随机数据 (N=1000, d=50, 其中只有前10维相关)
    np.random.seed(42)
    N, d_true, d_irrelevant = 1000, 10, 40
    d = d_true + d_irrelevant

    # 真实关系向量 (前10个非零)
    Wz_true = np.zeros(d)
    Wz_true[:d_true] = np.arange(1, d_true+1)

    # 生成无噪声输入 t (N, d)
    t = np.random.randn(N, d)
    # 输出 (无噪声)
    y_clean = t @ Wz_true

    # 添加噪声: 输入噪声 SNR=5, 输出噪声 SNR=5
    noise_input = np.random.randn(N, d) * (np.std(t, axis=0) / 5)
    noise_output = np.random.randn(N) * (np.std(y_clean) / 5)

    X = t + noise_input
    y = y_clean + noise_output

    # 使用贝叶斯JFA
    model = BayesianJFA(d=d, max_iter=50)
    model.fit(X, y)

    # 估计的关系向量
    print("Estimated Wz (first 15):", model.Wz_[:15])
    print("True Wz (first 15):     ", Wz_true[:15])

if __name__ == "__main__":    
    main()