import argparse
import json
import time
import warnings

import numpy as np

try:
    from identification.fourier_trajectory import FourierTrajectory
    from identification.target_limb_regressor import TargetLimbRegressor
except Exception:
    FourierTrajectory = None
    TargetLimbRegressor = None

SEED = 42
APPLY_PHYSICAL = False
MAX_ITER = 100000
TOL = 1e-5
VERBOSE = False
INPUT_SNR = 2.0
OUTPUT_SNR = 5.0
N_TRIALS = 3
N_TRAIN = 1000
N_TEST = 1000

YAML_FILE = "0724_2.yaml"


def format_array(value, per_line=5):
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return "[]"

    formatted = [f"{float(x):.3f}" for x in arr]
    groups = [
        ", ".join(formatted[i : i + per_line])
        for i in range(0, len(formatted), per_line)
    ]
    if len(groups) == 1:
        return "[" + groups[0] + "]"
    body = ",\n  ".join(groups)
    return "[\n  " + body + "\n]"


def safe_percent_error(estimate, truth, min_abs_truth=1e-2):
    denom = np.maximum(np.abs(truth), float(min_abs_truth))
    return (estimate - truth) / denom * 100.0


def identifiable_projection_matrix(X, tol_ratio=1e-10):
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        return np.zeros((X.shape[1], X.shape[1])), 0
    tol = s[0] * tol_ratio
    rank = int(np.sum(s > tol))
    v_r = vt[:rank, :].T
    return v_r @ v_r.T, rank


class VariationalBayesianJFA:
    """
    Paper-aligned variational Bayesian JFA for noisy linear regression.

    The updates follow Eq. (8)-(25) in:
    Ting et al., Neural Networks 24 (2011) 99-108.
    """

    def __init__(
        self,
        alpha_a0=1e-6,
        alpha_b0=1e-6,
        eps=1e-10,
        verbose=True,
    ):
        self.alpha_a0 = float(alpha_a0)
        self.alpha_b0 = float(alpha_b0)
        self.eps = float(eps)
        self.verbose = bool(verbose)

        self.fitted_ = False
        self.n_iter_ = 0

    def _as_vector(self, value, d):
        assert value is not None, "No Input Value Provided"
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.ones(d, dtype=float) * float(arr)
        if arr.shape != (d,):
            raise ValueError(f"Expected shape ({d},), got {arr.shape}")
        return arr.copy()

    def fit(
        self,
        X,
        Y,
        w_x_init=None,
        w_z_init=None,
        psi_x_init=None,
        psi_z_init=None,
        psi_y_init=None,
        max_iter=100000,
        tol=1e-5,
        cal_beta=False,
    ):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if Y.ndim != 1:
            raise ValueError("Y must be 1D")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y size mismatch")

        N, d = X.shape
        self.d = d

        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.w_x_mean = (
            self._as_vector(w_x_init, d)
            if w_x_init is not None
            else np.ones(d, dtype=float)
        )
        self.w_z_mean = (
            np.linalg.lstsq(X, Y, rcond=None)[0]
            if w_z_init is None
            else self._as_vector(w_z_init, d)
        )

        default_psi_x = np.maximum(np.var(X, axis=0) * 0.1, 1e-5)
        default_psi_z = np.maximum(np.var(X, axis=0) * 0.1, 1e-5)
        default_psi_y = float(max(np.var(Y) * 0.1, 1e-4))

        self.psi_x = (
            default_psi_x if psi_x_init is None else self._as_vector(psi_x_init, d)
        )
        self.psi_z = (
            default_psi_z if psi_z_init is None else self._as_vector(psi_z_init, d)
        )
        self.psi_y = default_psi_y if psi_y_init is None else float(psi_y_init)

        self.w_x_var = np.ones(d, dtype=float) * 1e-2
        self.w_z_var = np.ones(d, dtype=float) * 1e-2

        # Start from E[alpha] = 1.
        self.alpha_a = np.ones(d, dtype=float) * 1.0
        self.alpha_b = np.ones(d, dtype=float) * 1.0

        one = np.ones(d, dtype=float)
        prev_beta = None

        for it in range(self.max_iter):
            psi_x_safe = np.maximum(self.psi_x, self.eps)
            psi_z_safe = np.maximum(self.psi_z, self.eps)
            psi_y_safe = float(max(self.psi_y, self.eps))
            alpha_mean = self.alpha_a / np.maximum(self.alpha_b, self.eps)

            e_wx2 = self.w_x_mean**2 + self.w_x_var
            e_wz2 = self.w_z_mean**2 + self.w_z_var

            # Eq. (21)
            K_diag = 1.0 + e_wx2 / psi_x_safe + e_wz2 / psi_z_safe
            K_inv = np.diag(1.0 / np.maximum(K_diag, self.eps))

            # Eq. (22)
            inner_diag = 1.0 + e_wx2 / psi_x_safe + self.w_z_var / psi_z_safe
            M_diag = psi_z_safe + (self.w_z_mean**2) / np.maximum(inner_diag, self.eps)
            M = np.diag(M_diag)

            # Eq. (17)
            M1 = M @ one
            denom = psi_y_safe + float(one @ M1)
            Sigma_zz = M - np.outer(M1, M1) / max(denom, self.eps)

            Wz = np.diag(self.w_z_mean)
            Wx = np.diag(self.w_x_mean)
            Psi_z_inv = np.diag(1.0 / psi_z_safe)
            Psi_x_inv = np.diag(1.0 / psi_x_safe)

            # Eq. (19), Eq. (20), Eq. (18)
            Sigma_zt = Sigma_zz @ Wz @ Psi_z_inv @ K_inv
            Sigma_tz = Sigma_zt.T
            Sigma_tt = (
                K_inv + K_inv @ Wz @ Psi_z_inv @ Sigma_zz @ Psi_z_inv @ Wz @ K_inv
            )

            # Eq. (23), Eq. (24)
            const_z_row = one @ Sigma_zz
            const_t_row = one @ Sigma_zz @ Wz @ Psi_z_inv @ K_inv
            E_z = (Y[:, None] / psi_y_safe) * const_z_row[
                None, :
            ] + X @ Wx @ Psi_x_inv @ Sigma_tz
            E_t = (Y[:, None] / psi_y_safe) * const_t_row[
                None, :
            ] + X @ Wx @ Psi_x_inv @ Sigma_tt

            diag_Sigma_tt = np.diag(Sigma_tt)
            diag_Sigma_zz = np.diag(Sigma_zz)
            diag_Sigma_zt = np.diag(Sigma_zt)

            sum_t2 = N * diag_Sigma_tt + np.sum(E_t**2, axis=0)
            sum_z2 = N * diag_Sigma_zz + np.sum(E_z**2, axis=0)
            sum_zt = N * diag_Sigma_zt + np.sum(E_z * E_t, axis=0)
            sum_xt = np.sum(X * E_t, axis=0)
            sum_x2 = np.sum(X**2, axis=0)

            # Eq. (8)-(11)
            self.w_z_var = 1.0 / np.maximum(sum_t2 / psi_z_safe + alpha_mean, self.eps)
            self.w_z_mean = self.w_z_var * (sum_zt / psi_z_safe)

            self.w_x_var = 1.0 / np.maximum(sum_t2 / psi_x_safe + alpha_mean, self.eps)
            self.w_x_mean = self.w_x_var * (sum_xt / psi_x_safe)

            # Eq. (12), Eq. (13)
            e_wz2 = self.w_z_mean**2 + self.w_z_var
            e_wx2 = self.w_x_mean**2 + self.w_x_var
            self.alpha_a = np.ones(d, dtype=float) * (self.alpha_a0 + 1.0)
            self.alpha_b = self.alpha_b0 + 0.5 * (e_wz2 + e_wx2)

            # Eq. (15), Eq. (16)
            self.psi_z = (sum_z2 - 2.0 * self.w_z_mean * sum_zt + e_wz2 * sum_t2) / N
            self.psi_x = (sum_x2 - 2.0 * self.w_x_mean * sum_xt + e_wx2 * sum_t2) / N

            # Eq. (14)
            sum_ez = np.sum(E_z, axis=1)
            szz_scalar = float(one @ Sigma_zz @ one)
            self.psi_y = float(
                np.mean(Y**2 - 2.0 * Y * sum_ez + szz_scalar + sum_ez**2)
            )

            self.psi_x = np.maximum(self.psi_x, self.eps)
            self.psi_z = np.maximum(self.psi_z, self.eps)
            self.psi_y = float(max(self.psi_y, self.eps))

            if cal_beta:
                beta_now = self.get_beta_true()
                if prev_beta is not None:
                    if np.max(np.abs(beta_now - prev_beta)) < self.tol:
                        self.n_iter_ = it + 1
                        if self.verbose:
                            print(f"VB-EM converged at iteration {it}")
                        break
                prev_beta = beta_now
            else:
                # Lightweight convergence: track alpha_b changes (ARD weights)
                # instead of expensive get_beta_true().
                if prev_beta is not None:
                    if np.max(np.abs(self.alpha_b - prev_beta)) < self.tol:
                        self.n_iter_ = it + 1
                        if self.verbose:
                            print(f"VB-EM converged (alpha) at iteration {it}")
                        break
                prev_beta = self.alpha_b.copy()
        else:
            self.n_iter_ = self.max_iter
            if self.verbose:
                print("VB-EM reached max_iter without full convergence")

        self.fitted_ = True
        return self

    def get_beta_true(self):
        """Eq. (29): regression coefficients for noiseless input queries."""
        if self.d is None:
            raise RuntimeError("Model is not initialized")

        d = self.d
        eps = self.eps
        psi_y_safe = float(max(self.psi_y, eps))

        W_z = np.diag(self.w_z_mean)
        W_x_inv = np.diag(1.0 / (self.w_x_mean + eps))
        Psi_z_inv = np.diag(1.0 / (self.psi_z + eps))

        ones = np.ones((d, 1), dtype=float)
        C = (ones @ ones.T) / psi_y_safe + Psi_z_inv
        C_inv = np.linalg.inv(C)

        numerator = psi_y_safe * (ones.T @ C_inv)
        denominator = psi_y_safe - float((ones.T @ C_inv @ ones).item())
        if abs(denominator) < eps:
            denominator = eps if denominator >= 0.0 else -eps
        factor = numerator / denominator

        beta_true = factor @ Psi_z_inv @ W_z.T @ W_x_inv
        return beta_true.flatten()

    def get_alpha_mean(self):
        """Return E[alpha_j] = alpha_a_j / alpha_b_j for each input dimension."""
        if self.d is None:
            raise RuntimeError("Model is not initialized")
        return self.alpha_a / np.maximum(self.alpha_b, self.eps)

    def get_active_mask(self, threshold=100.0):
        """
        Return a boolean mask: True for dimensions whose ARD alpha is below
        *threshold*, meaning the parameter is "activated" (not pruned).

        Typical regime:
          - E[alpha_j] < 1e2  → strongly activated (identifiable)
          - E[alpha_j] > 1e3 → pruned (not identifiable)
          - 1e2~1e3          → intermediate (may be weakly excited)
        """
        if self.d is None:
            raise RuntimeError("Model is not initialized")
        alpha_mean = self.get_alpha_mean()
        return alpha_mean < threshold

    def count_small_alphas(self, threshold=100.0):
        """Return the integer count of activated (non-pruned) dimensions."""
        return int(np.sum(self.get_active_mask(threshold)))

    def get_theta_ratio(self):
        return self.w_z_mean / (self.w_x_mean + self.eps)


class RBDPhysicalConsistencyProjector:
    """
    Physical consistency projection from Section 4 of the paper.

    It enforces the 11-parameter rigid-body constraints via virtual variables
    and minimizes Eq. (35): 0.5 * dtheta^T * (X^T X) * dtheta.
    """

    def __init__(
        self,
        max_iter=80,
        damping=1e-6,
        tol=1e-8,
        fd_eps=1e-6,
        regularization=1e-10,
        verbose=False,
    ):
        self.max_iter = int(max_iter)
        self.damping = float(damping)
        self.tol = float(tol)
        self.fd_eps = float(fd_eps)
        self.regularization = float(regularization)
        self.verbose = bool(verbose)

    def _infer_layout(self, d):
        if d % 11 == 0:
            return np.arange(d, dtype=int)

        if d % 12 == 0:
            dof = d // 12
            idx = []
            for j in range(dof):
                base = 12 * j
                idx.extend(range(base, base + 11))
            return np.asarray(idx, dtype=int)

        return None

    def _nearest_spd(self, A, eps=1e-12):
        B = 0.5 * (A + A.T)
        eigvals, eigvecs = np.linalg.eigh(B)
        eigvals = np.maximum(eigvals, eps)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _theta11_from_virtual(self, h):
        theta = np.zeros(11, dtype=float)

        h1, h2, h3, h4 = h[0], h[1], h[2], h[3]
        h5, h6, h7, h8, h9, h10 = h[4], h[5], h[6], h[7], h[8], h[9]
        h11 = h[10]

        theta[0] = h1**2
        theta[1] = h2 * h1**2
        theta[2] = h3 * h1**2
        theta[3] = h4 * h1**2
        theta[4] = h5**2 + (h4**2 + h3**2) * h1**2
        theta[5] = h5 * h6 - h2 * h3 * h1**2
        theta[6] = h5 * h7 - h2 * h4 * h1**2
        theta[7] = h6**2 + h8**2 + (h2**2 + h4**2) * h1**2
        theta[8] = h6 * h7 + h8 * h9 - h3 * h4 * h1**2
        theta[9] = h7**2 + h9**2 + h10**2 + (h2**2 + h3**2) * h1**2
        theta[10] = h11**2
        return theta

    def _virtual11_from_theta(self, theta11, eps=1e-10):
        theta11 = np.asarray(theta11, dtype=float)
        h = np.zeros(11, dtype=float)

        mass = max(float(theta11[0]), eps)
        h[0] = np.sqrt(mass)
        h[1] = float(theta11[1]) / mass
        h[2] = float(theta11[2]) / mass
        h[3] = float(theta11[3]) / mass

        Ixx = float(theta11[4]) - (h[3] ** 2 + h[2] ** 2) * mass
        Ixy = float(theta11[5]) + h[1] * h[2] * mass
        Ixz = float(theta11[6]) + h[1] * h[3] * mass
        Iyy = float(theta11[7]) - (h[1] ** 2 + h[3] ** 2) * mass
        Iyz = float(theta11[8]) + h[2] * h[3] * mass
        Izz = float(theta11[9]) - (h[1] ** 2 + h[2] ** 2) * mass

        inertia = np.array(
            [
                [Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz],
            ],
            dtype=float,
        )
        inertia = self._nearest_spd(inertia, eps=eps)
        chol = np.linalg.cholesky(inertia)

        # Lower triangular terms map directly to h5..h10.
        h[4] = chol[0, 0]
        h[5] = chol[1, 0]
        h[6] = chol[2, 0]
        h[7] = chol[1, 1]
        h[8] = chol[2, 1]
        h[9] = chol[2, 2]
        h[10] = np.sqrt(max(float(theta11[10]), eps))
        return h

    def _map_virtual_subset(self, virtual):
        blocks = virtual.size // 11
        out = np.zeros(blocks * 11, dtype=float)
        for b in range(blocks):
            h = virtual[11 * b : 11 * (b + 1)]
            out[11 * b : 11 * (b + 1)] = self._theta11_from_virtual(h)
        return out

    def _init_virtual_subset(self, theta_subset):
        blocks = theta_subset.size // 11
        virtual = np.zeros(blocks * 11, dtype=float)
        for b in range(blocks):
            theta11 = theta_subset[11 * b : 11 * (b + 1)]
            virtual[11 * b : 11 * (b + 1)] = self._virtual11_from_theta(theta11)
        return virtual

    def _numeric_jacobian(self, f, x):
        fx = f(x)
        J = np.zeros((fx.size, x.size), dtype=float)
        for k in range(x.size):
            step = self.fd_eps * (1.0 + abs(float(x[k])))
            xp = x.copy()
            xm = x.copy()
            xp[k] += step
            xm[k] -= step
            fp = f(xp)
            fm = f(xm)
            J[:, k] = (fp - fm) / (2.0 * step)
        return J

    def _project_subset(self, theta_uc_subset, W_subset):
        n = theta_uc_subset.size
        if n == 0:
            return theta_uc_subset.copy(), {"iterations": 0, "objective": 0.0}

        W = 0.5 * (W_subset + W_subset.T)
        W = W + self.regularization * np.eye(W.shape[0])

        f = self._map_virtual_subset
        virtual = self._init_virtual_subset(theta_uc_subset)

        def objective_from_virtual(v):
            delta = f(v) - theta_uc_subset
            return 0.5 * float(delta.T @ W @ delta)

        objective = objective_from_virtual(virtual)

        for it in range(self.max_iter):
            theta_curr = f(virtual)
            delta = theta_curr - theta_uc_subset
            J = self._numeric_jacobian(f, virtual)

            g = J.T @ (W @ delta)
            H = J.T @ W @ J + self.damping * np.eye(n)

            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, -g, rcond=None)[0]

            step_norm = float(np.linalg.norm(step))
            if step_norm < self.tol * (1.0 + float(np.linalg.norm(virtual))):
                return theta_curr, {"iterations": it, "objective": objective}

            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.1, 0.05):
                cand_virtual = virtual + scale * step
                cand_obj = objective_from_virtual(cand_virtual)
                if cand_obj < objective:
                    virtual = cand_virtual
                    objective = cand_obj
                    accepted = True
                    break

            if not accepted:
                return theta_curr, {"iterations": it, "objective": objective}

        theta_final = f(virtual)
        return theta_final, {"iterations": self.max_iter, "objective": objective}

    def project(self, theta_uc, X):
        theta_uc = np.asarray(theta_uc, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != theta_uc.size:
            raise ValueError("theta dimension mismatch with X")

        phys_idx = self._infer_layout(theta_uc.size)
        if phys_idx is None:
            return theta_uc.copy(), {
                "applied": False,
                "reason": "dimension is not compatible with 11- or 12-parameter-per-DOF layout",
            }

        if phys_idx.size % 11 != 0:
            return theta_uc.copy(), {
                "applied": False,
                "reason": "physical index count is not divisible by 11",
            }

        W_full = (X.T @ X) / max(1, X.shape[0])
        W_phys = W_full[np.ix_(phys_idx, phys_idx)]
        theta_uc_phys = theta_uc[phys_idx]

        theta_phys, info = self._project_subset(theta_uc_phys, W_phys)

        theta_out = theta_uc.copy()
        theta_out[phys_idx] = theta_phys

        out_info = {
            "applied": True,
            "phys_dims": int(phys_idx.size),
            "iterations": int(info["iterations"]),
            "objective": float(info["objective"]),
        }
        return theta_out, out_info


def run_synthetic_demo(
    seed=SEED,
    input_snr=INPUT_SNR,
    output_snr=OUTPUT_SNR,
    n_trials=N_TRIALS,
    n_train=N_TRAIN,
    n_test=N_TEST,
    max_iter=MAX_ITER,
    tol=TOL,
    verbose=VERBOSE,
):
    """
    Section 5.1 synthetic evaluation (paper-aligned).

    Setup:
      - 100 total input dimensions, 10 relevant.
      - True regression vector β_true = [1, 2, ..., 10] (relevant dims only).
      - Output noise SNR = 5 (default), input noise SNR = 2 (default, strongly noisy).
      - Four (redundant r, irrelevant u) scenarios:
          (r=90, u=0), (r=0, u=90), (r=30, u=60), (r=60, u=30).
      - Redundant dims: random convex combinations of the 10 noisy relevant dims.
      - Irrelevant dims: Normal(0, 1).
      - Averaged over n_trials independent trials.
      - Test data is noiseless.
    Metric: nMSE = mean((y_pred - y_true)^2) / mean(y_true^2) on clean test data.
    Comparison: OLS vs Bayesian VB-JFA (no physical projection needed for synthetic).
    """
    print("\nRunning Section 5.1 synthetic evaluation...")
    n_relevant = 10
    n_total = 100
    scenarios = [(90, 0), (0, 90), (30, 60), (60, 30)]
    beta_true_rel = np.arange(1, n_relevant + 1, dtype=float)  # [1, 2, ..., 10]

    results = {}

    for r, u in scenarios:
        assert n_relevant + r + u == n_total, (
            f"r={r} u={u} does not sum to {n_total - n_relevant}"
        )
        nmse_ols_trials = []
        nmse_bayes_trials = []

        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial)

            # --- random covariance for the 10 relevant dimensions ---
            A = rng.standard_normal((n_relevant, n_relevant))
            cov = A @ A.T / n_relevant + np.eye(n_relevant) * 0.1

            # --- clean relevant inputs ---
            T_rel_train = rng.multivariate_normal(
                np.zeros(n_relevant), cov, size=n_train
            )
            T_rel_test = rng.multivariate_normal(np.zeros(n_relevant), cov, size=n_test)

            # --- clean outputs ---
            Y_clean_train = T_rel_train @ beta_true_rel
            Y_clean_test = T_rel_test @ beta_true_rel

            # --- add input noise to the 10 relevant training dimensions ---
            per_dim_var = np.var(T_rel_train, axis=0)
            input_noise_std = np.sqrt(per_dim_var / float(input_snr))
            X_rel_train = (
                T_rel_train
                + rng.standard_normal((n_train, n_relevant)) * input_noise_std
            )
            X_rel_test = T_rel_test.copy()  # noiseless test

            # --- add output noise ---
            output_noise_std = np.sqrt(float(np.var(Y_clean_train)) / float(output_snr))
            Y_noisy_train = (
                Y_clean_train + rng.standard_normal(n_train) * output_noise_std
            )
            Y_test = Y_clean_test.copy()  # noiseless test

            # --- redundant dimensions: convex combos of the 10 noisy relevant dims ---
            if r > 0:
                weights = rng.uniform(0.0, 1.0, size=(r, n_relevant))
                weights /= weights.sum(axis=1, keepdims=True)
                X_red_train = X_rel_train @ weights.T  # (n_train, r)
                X_red_test = X_rel_test @ weights.T
            else:
                X_red_train = np.empty((n_train, 0))
                X_red_test = np.empty((n_test, 0))

            # --- irrelevant dimensions ---
            if u > 0:
                X_irr_train = rng.standard_normal((n_train, u))
                X_irr_test = rng.standard_normal((n_test, u))
            else:
                X_irr_train = np.empty((n_train, 0))
                X_irr_test = np.empty((n_test, 0))

            # --- assemble full 100-dim matrices ---
            X_train = np.hstack([X_rel_train, X_red_train, X_irr_train])
            X_test = np.hstack([X_rel_test, X_red_test, X_irr_test])

            # --- OLS ---
            beta_ols = np.linalg.lstsq(X_train, Y_noisy_train, rcond=None)[0]

            # --- Bayesian VB-JFA ---
            model = VariationalBayesianJFA(verbose=verbose)
            model.fit(X_train, Y_noisy_train, cal_beta=True, max_iter=max_iter, tol=tol)
            beta_bayes = model.get_beta_true()

            # --- nMSE on noiseless test data ---
            denom = float(np.mean(Y_test**2))
            denom = max(denom, 1e-12)
            nmse_ols = float(np.mean((X_test @ beta_ols - Y_test) ** 2)) / denom
            nmse_bayes = float(np.mean((X_test @ beta_bayes - Y_test) ** 2)) / denom

            nmse_ols_trials.append(nmse_ols)
            nmse_bayes_trials.append(nmse_bayes)

            if verbose:
                print(
                    f"  [{r=:2d},{u=:2d}] trial {trial + 1:2d}/{n_trials}  "
                    f"nMSE_OLS={nmse_ols:.4f}  nMSE_BAYES={nmse_bayes:.4f}"
                )

        results[(r, u)] = {
            "ols": (float(np.mean(nmse_ols_trials)), float(np.std(nmse_ols_trials))),
            "bayes": (
                float(np.mean(nmse_bayes_trials)),
                float(np.std(nmse_bayes_trials)),
            ),
        }

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    sep = "=" * 72
    print("\n" + sep)
    print("  Section 5.1 — Synthetic Evaluation")
    print(
        f"  input SNR={input_snr:.0f}, output SNR={output_snr:.0f}, "
        f"n_train={n_train}, n_test={n_test}, n_trials={n_trials}"
    )
    print(sep)
    print(
        f"  {'Scenario':>14}  {'OLS nMSE':>20}  {'BAYES nMSE':>20}  {'Improvement':>12}"
    )
    print("-" * 72)
    for r, u in scenarios:
        v = results[(r, u)]
        ols_m, ols_s = v["ols"]
        bay_m, bay_s = v["bayes"]
        improvement = (ols_m - bay_m) / max(bay_m, 1e-12) * 100.0
        label = f"r={r:2d}, u={u:2d}"
        print(
            f"  {label:>14}  {ols_m:8.4f} ± {ols_s:.4f}  "
            f"{bay_m:8.4f} ± {bay_s:.4f}  {improvement:+10.1f}%"
        )
    print(sep + "\n")
    return results


def run_robot_demo(
    seed=42,
    apply_physical=True,
    max_iter=100000,
    tol=1e-5,
    verbose=True,
    parameters=None,
):

    np.random.seed(seed)
    regressor = TargetLimbRegressor()
    fourier_traj = FourierTrajectory(
        dim=regressor.dof,
        sample_rate=500,
    )

    start = time.time()

    q, v, a = fourier_traj.generate_trajectory_from_yaml(YAML_FILE)

    dof = q.shape[0]
    samples = q.shape[1]

    X_true = [np.empty((0, dof * 12), dtype=float)] * dof
    Y_true = [np.empty((0,), dtype=float)] * dof
    theta_true = None

    for sample in range(samples):
        (
            Y_aug,
            tau_aug,
            pi_aug,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = regressor.compute_regressor(q=q[:, sample], v=v[:, sample], a=a[:, sample])
        for d in range(dof):
            X_true[d] = np.vstack((X_true[d], Y_aug[d]))
            Y_true[d] = np.hstack((Y_true[d], tau_aug[d]))
        theta_true = pi_aug

    if theta_true is None:
        raise RuntimeError("Failed to build robot dataset")

    N, d_ = X_true[0].shape

    # Build per-joint subtree mask: mask[d, j] = True if joint j is in joint d's
    # kinematic subtree (i.e. joint d's torque depends on joint j's parameters).
    subtree_mask = regressor.get_subtree_mask()  # shape (dof, dof)

    for d in range(dof):
        X_noisy = X_true[d] + np.random.normal(0.0, 0.1, (N, d_))
        Y_noisy = Y_true[d] + np.random.normal(0.0, 0.2, N)

        proj_ident, rank_x = identifiable_projection_matrix(X_true[d])

        theta_ols = np.linalg.lstsq(X_noisy, Y_noisy, rcond=None)[0]

        # --- per-joint w_z_init: zero out parameters that are structurally
        #     irrelevant for THIS joint's torque equation ---
        w_z_init_joint = np.zeros(d_, dtype=float)
        for j in range(dof):
            if subtree_mask[d, j]:
                col_start = j * 12
                col_end = col_start + 12
                w_z_init_joint[col_start:col_end] = theta_true[col_start:col_end]

        # --- w_x_init: small value for structurally-irrelevant (zero) columns,
        #     ones for relevant columns ---
        zero_col_mask = ~np.any(np.abs(X_noisy) > 1e-12, axis=0)
        w_x_init_joint = np.where(zero_col_mask, 1e-6, 1.0)
        psi_x_init_joint = np.where(zero_col_mask, 1e-10, 0.1)

        model = VariationalBayesianJFA(verbose=verbose)
        model.fit(
            X=X_noisy,
            Y=Y_noisy,
            w_x_init=w_x_init_joint,
            w_z_init=w_z_init_joint,
            psi_x_init=psi_x_init_joint,
            psi_z_init=np.ones(d_) * 0.1,
            psi_y_init=0.2,
            tol=1e-5,
            cal_beta=True,
        )
        theta_bayes = model.get_beta_true()

        theta_bayes_phys = theta_bayes.copy()
        phys_info = {"applied": False, "reason": "disabled"}
        if apply_physical:
            projector = RBDPhysicalConsistencyProjector(verbose=True)
            theta_bayes_phys, phys_info = projector.project(theta_bayes, X_noisy)

        theta_true_ident = proj_ident @ theta_true
        theta_ols_ident = proj_ident @ theta_ols
        theta_bayes_ident = proj_ident @ theta_bayes
        theta_bayes_phys_ident = proj_ident @ theta_bayes_phys

        rmse_ols = float(np.sqrt(np.mean((Y_noisy - X_noisy @ theta_ols) ** 2)))
        rmse_bayes = float(np.sqrt(np.mean((Y_noisy - X_noisy @ theta_bayes) ** 2)))
        rmse_bayes_phys = float(
            np.sqrt(np.mean((Y_noisy - X_noisy @ theta_bayes_phys) ** 2))
        )

        print("================ Robot Identification =================")
        print(f"Joint {d + 1}/{dof}")
        # Show which joints are in this joint's subtree
        subtree_joints = [j for j in range(dof) if subtree_mask[d, j]]
        print(
            f"Subtree joints: {subtree_joints}  (active params: {len(subtree_joints) * 12}/{d_})"
        )
        print(f"VB-JFA converged in {model.n_iter_} iterations")
        print(
            f"VB-JFA E[alpha] (per 12-param block): "
            f"{format_array([float(np.mean(model.get_alpha_mean()[j * 12 : (j + 1) * 12])) for j in range(dof)])}"
        )
        print(f"VB-JFA active dims (alpha < 1e2): {model.count_small_alphas(100)}/{d_}")
        print(f"X_true shape: {X_true[d].shape}")
        print(f"Y_true shape: {Y_true[d].shape}")
        print(f"theta_true shape: {theta_true.shape}")
        print(f"rank(X_true): {rank_x}/{d_}")
        print(f"matrix generation time: {time.time() - start:.2f} s")

        print(f"P@theta_true:\n{format_array(theta_true_ident)}")
        print(f"P@theta_ols:\n{format_array(theta_ols_ident)}")
        print(f"P@theta_bayes:\n{format_array(theta_bayes_ident)}")
        print(f"P@theta_bayes_phys:\n{format_array(theta_bayes_phys_ident)}")

        print(
            f"OLS safe relative error (%):\n{format_array(safe_percent_error(theta_ols_ident, theta_true_ident))}"
        )
        print(
            f"VB-JFA safe relative error (%):\n{format_array(safe_percent_error(theta_bayes_ident, theta_true_ident))}"
        )
        print(
            "VB-JFA + physical safe relative error (%):\n"
            f"{format_array(safe_percent_error(theta_bayes_phys_ident, theta_true_ident))}"
        )

        print(f"physical projection applied: {phys_info.get('applied', False)}")
        if not phys_info.get("applied", False):
            print(f"physical projection reason: {phys_info.get('reason', 'n/a')}")
        else:
            print(
                "physical projection details: "
                f"phys_dims={phys_info['phys_dims']}, "
                f"iter={phys_info['iterations']}, "
                f"objective={phys_info['objective']:.6e}"
            )

        print(f"OLS torque RMSE: {rmse_ols:.6f}")
        print(f"VB-JFA torque RMSE: {rmse_bayes:.6f}")
        print(f"VB-JFA + physical torque RMSE: {rmse_bayes_phys:.6f}")
        print("======================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Paper-aligned Bayesian JFA with optional physical consistency projection"
    )
    parser.add_argument(
        "--demo",
        choices=["synthetic", "robot"],
        default="synthetic",
        help=(
            "synthetic: Section 5.1 setup (OLS vs BAYES, 4 scenarios, 10 trials)"
            "robot: full robot regression demo with input fourier trajectory parameters"
        ),
    )

    # General options
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for reproducibility"
    )
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--quiet", action="store_true", help="Reduce solver logs")

    # Section 5.1 specific options
    parser.add_argument(
        "--input-snr",
        type=float,
        default=INPUT_SNR,
        help=f"Input SNR for paper_synthetic demo (default {INPUT_SNR})",
    )
    parser.add_argument(
        "--output-snr",
        type=float,
        default=OUTPUT_SNR,
        help=f"Output SNR for paper_synthetic demo (default {OUTPUT_SNR})",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of trials to average for paper_synthetic demo (default {N_TRIALS})",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=N_TRAIN,
        help=f"Training samples per trial for paper_synthetic demo (default {N_TRAIN})",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=N_TEST,
        help=f"Test samples per trial for paper_synthetic demo (default {N_TEST})",
    )

    # Robot demo specific options
    parser.add_argument(
        "--no-physical",
        action="store_true",
        default=True,
        help="Disable physical consistency projection",
    )
    parser.add_argument(
        "--fourier-parameters",
        type=json.loads,
        help='Parameters for Fourier trajectory generation as a JSON array. Size [dof * (n_harmonics * 2 + 2)], n_harmonics=5 in this project. Usage: --fourier-parameters "[1.0, 2.0, 3.0]"',
    )
    args = parser.parse_args()

    if args.demo == "synthetic":
        run_synthetic_demo(
            seed=args.seed,
            input_snr=args.input_snr,
            output_snr=args.output_snr,
            n_trials=args.n_trials,
            n_train=args.n_train,
            n_test=args.n_test,
            max_iter=args.max_iter,
            tol=args.tol,
            verbose=not args.quiet,
        )
        return

    try:
        run_robot_demo(
            seed=args.seed,
            apply_physical=not args.no_physical,
            max_iter=args.max_iter,
            tol=args.tol,
            verbose=not args.quiet,
            parameters=args.fourier_parameters,
        )
    except Exception as exc:
        warnings.warn(f"Robot demo failed: {exc}")
        raise


if __name__ == "__main__":
    main()
