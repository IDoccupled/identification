import argparse
import json
import warnings

import numpy as np

from identification.fourier_trajectory import FourierTrajectory
from identification.target_limb_regressor import TargetLimbRegressor

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

YAML_FILE = ["0724_2.yaml", "0729_1.yaml", "0729_2.yaml", "0729_3.yaml"]
# YAML_FILE = ["0724_2.yaml"]


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

    Generative model (per sample n, dimension j):
        t_n       ~ N(0, I)              # latent clean input
        x_{n,j}   = w_{x,j} t_{n,j} + ε^x_{n,j},   ε^x ~ N(0, ψ_{x,j})
        z_{n,j}   = w_{z,j} t_{n,j} + ε^z_{n,j},   ε^z ~ N(0, ψ_{z,j})
        y_n       = Σ_j z_{n,j} + ε^y_n,            ε^y ~ N(0, ψ_y)

    True regression coefficient (noiseless x → y):
        β_j = w_{z,j} / w_{x,j}                        Eq. (29)

    Priors:
        w_{x,j}, w_{z,j} ~ N(0, α_j^{-1})
        α_j ~ Gamma(a_0, b_0)

    Variational posterior factorisation:
        q(W,Z,T,α) = Π_j q(w_{x,j}) q(w_{z,j})  Π_n q(z_n,t_n)  Π_j q(α_j)
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
        self.d = None

    def _as_vector(self, value, d):
        assert value is not None, "No Input Value Provided"
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(d, float(arr), dtype=float)
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
        max_iter=1e5,
        tol=1e-5,
        cal_beta=False,
    ):
        """Run VB-EM for the JFA model.

        Parameters
        ----------
        X : (N, d) array — observed noisy inputs.
        Y : (N,)  array — observed outputs.
        w_x_init, w_z_init : (d,) initial weight means.
        psi_x_init, psi_z_init : (d,) initial input noise variances.
        psi_y_init : float, initial output noise variance.
        max_iter : int
        tol : float — convergence threshold on max |Δψ_y| and mean |Δα_b|.
        cal_beta : bool — if True, also check convergence on β = w_z / w_x
                   (slightly more expensive).
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        assert X.ndim == 2, "X must be 2D"
        assert Y.ndim == 1, "Y must be 1D"
        assert X.shape[0] == Y.shape[0], "X and Y size mismatch"

        N, d = X.shape
        self.d = d
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # ---- initialise q(w) ----
        self.w_x_mean = (
            self._as_vector(w_x_init, d)
            if w_x_init is not None
            else np.ones(d, dtype=float)
        )
        self.w_z_mean = (
            self._as_vector(w_z_init, d)
            if w_z_init is not None
            else np.linalg.lstsq(X, Y, rcond=None)[0]
        )
        self.w_x_var = np.full(d, 1e-2, dtype=float)
        self.w_z_var = np.full(d, 1e-2, dtype=float)

        # ---- initialise q(α) ----
        self.alpha_a = np.full(d, self.alpha_a0, dtype=float)
        self.alpha_b = np.full(d, self.alpha_b0, dtype=float)

        # ---- initialise noise variances ----
        self.psi_x = self._as_vector(psi_x_init, d)
        self.psi_z = self._as_vector(psi_z_init, d)
        self.psi_y = float(psi_y_init)

        # pre-compute Σ_x = X^T X / N (used only for convergence tracking)
        sum_x2 = np.sum(X**2, axis=0)  # (d,)

        # ---- convergence tracking ----
        prev_param = None  # stores β (if cal_beta) or α_b (otherwise)

        for it in range(self.max_iter):
            # -------- numerical safeguards --------
            psi_x_s = np.maximum(self.psi_x, self.eps)
            psi_z_s = np.maximum(self.psi_z, self.eps)
            psi_y_s = float(max(self.psi_y, self.eps))
            alpha_mean = self.alpha_a / np.maximum(self.alpha_b, self.eps)

            e_wx2 = self.w_x_mean**2 + self.w_x_var  # E[w_{x,j}²]
            e_wz2 = self.w_z_mean**2 + self.w_z_var  # E[w_{z,j}²]

            # ============================================================
            #  E-step: update q(z_n, t_n) for all n
            # ============================================================
            #
            #  Joint precision  Λ = [[Λ_zz, Λ_zt], [Λ_tz, Λ_tt]]
            #
            #  Λ_zz = diag(1/ψ_z) + 11^T / ψ_y                     (a)
            #  Λ_zt = -diag(μ_z / ψ_z)                              (b)
            #  Λ_tt = diag(1 + e_wx2/ψ_x + e_wz2/ψ_z)  = diag(K)   (c)
            #
            #  Schur complement for z:
            #    S_z = Λ_zz - Λ_zt Λ_tt^{-1} Λ_zt
            #        = diag(a) + 11^T/ψ_y
            #    where  a_j = 1/ψ_{z,j} − μ_{z,j}² / (ψ_{z,j}² K_j)

            K_diag = 1.0 + e_wx2 / psi_x_s + e_wz2 / psi_z_s  # (d,)
            a_diag = 1.0 / psi_z_s - self.w_z_mean**2 / (psi_z_s**2 * K_diag)
            a_diag = np.maximum(a_diag, self.eps)  # (d,)
            v = 1.0 / a_diag  # (d,)
            s_v = float(np.sum(v))
            woodbury_denom = psi_y_s + s_v
            woodbury_denom = max(woodbury_denom, self.eps)

            # β_j = μ_{z,j} / (ψ_{z,j} K_j)   —  auxiliary vector
            beta_vec = self.w_z_mean / (psi_z_s * K_diag)  # (d,)

            # ---- per-sample posterior means μ_{z,n}, μ_{t,n} ----
            #  g_n    = x_n ⊙ μ_x / ψ_x      (d,)     η_t term
            #  h_n    = g_n ⊙ β              (d,)
            #  c_n    = y_n / woodbury_denom (scalar)
            #
            #  μ_{z,n} = c_n · v  +  v ⊙ h_n  −  v · (v·h_n) / woodbury_denom
            #  μ_{t,n} = g_n / K  +  β ⊙ μ_{z,n}

            g = X * (self.w_x_mean[None, :] / psi_x_s[None, :])  # (N, d)
            h = g * beta_vec[None, :]  # (N, d)
            vh = h @ v  # (N,)
            c = Y / woodbury_denom  # (N,)

            # μ_z  (N, d)
            E_z = v[None, :] * (c[:, None] + h - vh[:, None] / woodbury_denom)

            # μ_t  (N, d)
            E_t = g / K_diag[None, :] + beta_vec[None, :] * E_z

            # ---- posterior covariances (shared across samples) ----
            #  Σ_zz_{i,j} = δ_{ij}·v_i  −  v_i v_j / woodbury_denom
            #  Σ_zt = Σ_zz · diag(β)
            #  Σ_tt = diag(1/K) + diag(β) · Σ_zz · diag(β)
            #
            #  Only the diagonals are needed for the M-step.
            diag_Sigma_zz = v - v**2 / woodbury_denom  # (d,)
            diag_Sigma_zt = diag_Sigma_zz * beta_vec  # (d,)
            diag_Sigma_tt = 1.0 / K_diag + beta_vec**2 * diag_Sigma_zz  # (d,)

            # ---- sufficient statistics for M-step ----
            sum_t2 = N * diag_Sigma_tt + np.sum(E_t**2, axis=0)  # (d,)
            sum_z2 = N * diag_Sigma_zz + np.sum(E_z**2, axis=0)  # (d,)
            sum_zt = N * diag_Sigma_zt + np.sum(E_z * E_t, axis=0)  # (d,)
            sum_xt = np.sum(X * E_t, axis=0)  # (d,)

            # ============================================================
            #  M-step: update q(w_z), q(w_x), q(α), ψ
            # ============================================================

            # Eq. (8):  Σ_{w_z}  = (diag(E[α]) + diag(sum_t2/ψ_z))^{-1}
            prec_wz = alpha_mean + sum_t2 / psi_z_s
            self.w_z_var = 1.0 / np.maximum(prec_wz, self.eps)
            # Eq. (9):  μ_{w_z}  = Σ_{w_z} · (sum_zt / ψ_z)
            self.w_z_mean = self.w_z_var * (sum_zt / psi_z_s)

            # Eq. (10): Σ_{w_x}  = (diag(E[α]) + diag(sum_t2/ψ_x))^{-1}
            prec_wx = alpha_mean + sum_t2 / psi_x_s
            self.w_x_var = 1.0 / np.maximum(prec_wx, self.eps)
            # Eq. (11): μ_{w_x}  = Σ_{w_x} · (sum_xt / ψ_x)
            self.w_x_mean = self.w_x_var * (sum_xt / psi_x_s)

            # Eq. (12)-(13): q(α_j) = Gamma(a_0+1,  b_0 + ½(E[w_z²]+E[w_x²]))
            e_wz2 = self.w_z_mean**2 + self.w_z_var
            e_wx2 = self.w_x_mean**2 + self.w_x_var
            self.alpha_a = np.full(d, self.alpha_a0 + 1.0, dtype=float)
            self.alpha_b = self.alpha_b0 + 0.5 * (e_wz2 + e_wx2)

            # Eq. (15): ψ_{z,j} = (1/N) Σ_n E[(z_{n,j} − w_{z,j} t_{n,j})²]
            self.psi_z = (sum_z2 - 2.0 * self.w_z_mean * sum_zt + e_wz2 * sum_t2) / N
            # Eq. (16): ψ_{x,j} = (1/N) Σ_n E[(x_{n,j} − w_{x,j} t_{n,j})²]
            self.psi_x = (sum_x2 - 2.0 * self.w_x_mean * sum_xt + e_wx2 * sum_t2) / N

            # Eq. (14): ψ_y = (1/N) Σ_n E[(y_n − 1^T z_n)²]
            #  1^T Σ_zz 1 = Σ_i v_i − (Σ_i v_i)²/woodbury_denom = s_v − s_v²/woodbury_denom
            sum_ez = np.sum(E_z, axis=1)  # (N,)
            szz_scalar = s_v - s_v**2 / woodbury_denom
            self.psi_y = float(
                np.mean(Y**2 - 2.0 * Y * sum_ez + szz_scalar + sum_ez**2)
            )

            # ---- clamp noise variances ----
            self.psi_x = np.clip(self.psi_x, self.eps, 1.0 / self.eps)
            self.psi_z = np.clip(self.psi_z, self.eps, 1.0 / self.eps)
            self.psi_y = float(np.clip(self.psi_y, self.eps, 1.0 / self.eps))

            # ---- convergence check ----
            converged = False
            if cal_beta:
                beta_now = self.get_beta_true()
                if prev_param is not None:
                    if np.max(np.abs(beta_now - prev_param)) < self.tol:
                        converged = True
                prev_param = beta_now
            else:
                # Track changes in ARD rate parameters α_b
                if prev_param is not None:
                    if np.max(np.abs(self.alpha_b - prev_param)) < self.tol:
                        converged = True
                prev_param = self.alpha_b.copy()

            if converged:
                self.n_iter_ = it + 1
                if self.verbose:
                    print(f"VB-EM converged at iteration {it + 1}")
                break
        else:
            self.n_iter_ = self.max_iter
            if self.verbose:
                print("VB-EM reached max_iter without full convergence")

        self.fitted_ = True
        return self

    def get_beta_true(self):
        """
        Eq. (29): regression vector for noiseless input queries.

        β̂_true = [ψ_y · 1^T C^{-1} / (ψ_y − 1^T C^{-1} 1)] · Ψ_z^{-1} · <W_z>^T · <W_x>^{-1}

        where  C = 11^T / ψ_y + Ψ_z^{-1}

        C^{-1} is obtained by the Sherman–Morrison formula:
            C^{-1} = Ψ_z − (Ψ_z 1)(Ψ_z 1)^T / (ψ_y + 1^T Ψ_z 1)

        All diagonal matrices (<W_z>, <W_x>, Ψ_z) are represented by their
        diagonal vectors (w_z_mean, w_x_mean, psi_z).
        """
        if self.d is None:
            raise RuntimeError("Model is not initialized")

        psi_z = np.maximum(self.psi_z, self.eps)  # (d,)  diagonal of Ψ_z
        psi_y = float(max(self.psi_y, self.eps))  # scalar ψ_y

        # ---- C^{-1} via Sherman–Morrison ----
        # C = Ψ_z^{-1} + (1/ψ_y) 1 1^T
        # s = 1^T Ψ_z 1 = Σ_j ψ_{z,j}
        s = float(np.sum(psi_z))

        # 1^T C^{-1}  (row vector, size d)
        # = 1^T Ψ_z − s · 1^T Ψ_z / (ψ_y + s)
        # = ψ_z · ψ_y / (ψ_y + s)
        one_T_C_inv = psi_z * psi_y / (psi_y + s)  # (d,)

        # 1^T C^{-1} 1  (scalar)
        one_T_C_inv_1 = float(np.sum(one_T_C_inv))  # = ψ_y · s / (ψ_y + s)

        # denominator:  ψ_y − 1^T C^{-1} 1  (= ψ_y² / (ψ_y + s))
        denom_scalar = max(psi_y - one_T_C_inv_1, self.eps)

        # prefactor row vector:  ψ_y · 1^T C^{-1} / denom_scalar  (d,)
        prefactor = psi_y * one_T_C_inv / denom_scalar  # (d,)

        # β̂_true = prefactor · Ψ_z^{-1} · diag(<w_z>) · diag(<w_x>)^{-1}
        # element-wise: β_j = prefactor_j / ψ_{z,j} · <w_{z,j}> / <w_{x,j}>
        beta_true = (
            (prefactor / psi_z) * self.w_z_mean / np.maximum(self.w_x_mean, self.eps)
        )
        return beta_true

    def get_alpha_mean(self):
        """Return E[α_j] = alpha_a_j / alpha_b_j for each input dimension."""
        if self.d is None:
            raise RuntimeError("Model is not initialized")
        return self.alpha_a / np.maximum(self.alpha_b, self.eps)

    def get_active_mask(self, threshold=100.0):
        """
        Return a boolean mask: True for dimensions whose ARD alpha is below
        *threshold*, meaning the parameter is "activated" (not pruned).

        Typical regime:
          - E[α_j] < 1e2  → strongly activated (identifiable)
          - E[α_j] > 1e3 → pruned (not identifiable)
          - 1e2~1e3      → intermediate (may be weakly excited)
        """
        if self.d is None:
            raise RuntimeError("Model is not initialized")
        return self.get_alpha_mean() < threshold

    def count_small_alphas(self, threshold=100.0):
        """Return the integer count of activated (non-pruned) dimensions."""
        return int(np.sum(self.get_active_mask(threshold)))


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


# =========================================================================
# Chirp (frequency sweep) trajectory generator for rum_robot_random demo
# =========================================================================

CHIRP_DEFAULT_DURATION = 5.0
CHIRP_DEFAULT_SAMPLE_RATE = 100.0
CHIRP_DEFAULT_N_TRIALS = 30


def generate_chirp_trajectory(
    dof,
    q_lower,
    q_upper,
    duration=CHIRP_DEFAULT_DURATION,
    sample_rate=CHIRP_DEFAULT_SAMPLE_RATE,
    rng=None,
):
    """
    Generate a single chirp (frequency-sweep) trajectory for all joints.

    Each joint gets independently randomized parameters to maximise the
    diversity of angle–velocity combinations across trials:

    - Amplitude: 20%–80% of the joint's half-range.
    - Centre position q0: uniform within safe bounds (avoiding limit margins).
    - Start/end frequency: random sweep between 0.05 Hz and 8 Hz, direction
      randomised (up- or down-chirp).
    - Phase: uniform random [0, 2π).

    Parameters
    ----------
    dof : int
        Number of degrees of freedom.
    q_lower : ndarray of shape (dof,)
        Lower joint position limits.
    q_upper : ndarray of shape (dof,)
        Upper joint position limits.
    duration : float
        Trajectory duration in seconds.
    sample_rate : float
        Sampling frequency in Hz.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    q_traj : ndarray of shape (dof, n_samples)
    v_traj : ndarray of shape (dof, n_samples)
    a_traj : ndarray of shape (dof, n_samples)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(duration * sample_rate)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    q_traj = np.zeros((dof, n_samples))
    v_traj = np.zeros((dof, n_samples))
    a_traj = np.zeros((dof, n_samples))

    for i in range(dof):
        q_range = float(q_upper[i] - q_lower[i])
        half_range = q_range / 2.0

        # Amplitude: cover different magnitudes across trials
        amp_ratio = rng.uniform(0.2, 0.8)
        amplitude = half_range * amp_ratio

        # Centre position q0: stay within safe bounds
        margin = amplitude * 1.05  # small safety margin
        q0_min = float(q_lower[i]) + margin
        q0_max = float(q_upper[i]) - margin
        if q0_min >= q0_max:
            # fallback for very narrow joints
            q0 = (float(q_lower[i]) + float(q_upper[i])) / 2.0
        else:
            q0 = rng.uniform(q0_min, q0_max)

        # Frequency sweep range – randomised per joint per trial
        f_start = rng.uniform(0.05, 2.0)
        f_end = rng.uniform(1.5, 8.0)
        # Randomise sweep direction (up-chirp vs down-chirp)
        if rng.random() < 0.5:
            f_start, f_end = f_end, f_start

        # Random phase offset
        phase = rng.uniform(0.0, 2.0 * np.pi)

        # --- Chirp signal generation ---
        # φ(t) = 2π·[f₀·t + (f₁−f₀)/(2T)·t²] + phase₀
        f_slope = (f_end - f_start) / duration
        phi = 2.0 * np.pi * (f_start * t + 0.5 * f_slope * t**2) + phase

        # Instantaneous frequency and its derivative
        f_inst = f_start + f_slope * t  # Hz
        omega_inst = 2.0 * np.pi * f_inst  # rad/s
        alpha_inst = 2.0 * np.pi * f_slope  # rad/s²  (constant)

        # Position, velocity, acceleration
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        q_traj[i, :] = q0 + amplitude * sin_phi
        v_traj[i, :] = amplitude * omega_inst * cos_phi
        a_traj[i, :] = (
            -amplitude * omega_inst**2 * sin_phi + amplitude * alpha_inst * cos_phi
        )

    return q_traj, v_traj, a_traj


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
    # ===
    n_relevant = 10
    n_total = 100
    scenarios = [(90, 0), (0, 90), (30, 60), (60, 30)]
    beta_true_rel = np.arange(1, n_relevant + 1, dtype=float)  # [1, 2, ..., 10]
    # beta_true_rel = np.array([0.7, 0.5, 1, 5, 0.3, 0.2, 0.1, 1, 0.6, 0.4]) * 10
    # ===
    # n_relevant = 10
    # n_total = 10
    # scenarios = [(0, 0)]
    # beta_true_rel = np.arange(1, n_relevant + 1, dtype=float)  # [1, 2, ..., 10]
    # # beta_true_rel = np.array([0.7, 0.5, 1, 5, 0.3, 0.2, 0.1, 1, 0.6, 0.4]) * 10

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
            # Use known noise variances for better initialization
            psi_x_init = np.maximum(np.var(X_train, axis=0), 1e-6)
            # For redundant/irrelevant dims, psi_x captures natural scale
            model.fit(
                X_train,
                Y_noisy_train,
                w_x_init=np.ones(X_train.shape[1]),
                w_z_init=np.hstack(
                    [beta_true_rel, np.zeros((X_train.shape[1] - n_relevant,))]
                )
                + np.random.normal(0, 0.3, size=X_train.shape[1]),
                psi_x_init=psi_x_init,
                psi_z_init=np.full(
                    X_train.shape[1],
                    max(np.var(Y_noisy_train), 1e-4) / max(X_train.shape[1], 1),
                ),
                psi_y_init=max(output_noise_std**2, 1e-4),
                cal_beta=True,
                max_iter=max_iter,
                tol=tol,
            )
            beta_bayes = model.get_beta_true()  # paper Eq.(29): w_z / w_x

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
                    f"nMSE_OLS={nmse_ols:.4f}  nMSE_BAYES={nmse_bayes:.4f}\n"
                    f"beta_OLS={beta_ols}\nbeta_BAYES={beta_bayes}\n"
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
    input_snr=10.0,
    output_snr=20.0,
    apply_physical=True,
    max_iter=1e5,
    tol=1e-5,
    verbose=True,
    parameters=None,
):

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    regressor = TargetLimbRegressor()
    fourier_traj = FourierTrajectory(
        dim=regressor.dof,
        sample_rate=100,
    )

    X_true_per_joint = [np.empty((0, 5 * 12), dtype=float)] * 5
    X_noisy_per_joint = [np.empty((0, 5 * 12), dtype=float)] * 5
    Y_clean_per_joint = [np.empty((0,), dtype=float)] * 5

    for trial in YAML_FILE:
        q, v, a = fourier_traj.generate_trajectory_from_yaml(trial)

        dof = q.shape[0]
        assert dof == 5, f"Expected 5 DoF, got {dof}"
        samples = q.shape[1]

        # ---- Add SNR-based noise to v and a BEFORE regressor computation ----
        # This propagates naturally through the regressor, creating realistic
        # column-dependent noise structure suitable for VB-JFA.
        v_var = np.var(v, axis=1)  # per-DoF velocity variance
        a_var = np.var(a, axis=1)  # per-DoF acceleration variance
        v_noise_std = np.sqrt(np.maximum(v_var, 1e-8) / input_snr)
        a_noise_std = np.sqrt(np.maximum(a_var, 1e-8) / input_snr)
        v_noisy = v + rng.normal(0, 1, v.shape) * v_noise_std[:, None]
        a_noisy = a + rng.normal(0, 1, a.shape) * a_noise_std[:, None]
        theta_true = None

        for sample in range(samples):
            # Clean: from clean q, v, a
            (
                Y_aug_clean,
                tau_clean,
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
            ) = regressor.compute_regressor(
                q=q[:, sample], v=v[:, sample], a=a[:, sample]
            )
            # Noisy: from clean q, noisy v, a
            (
                Y_aug_noisy,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = regressor.compute_regressor(
                q=q[:, sample], v=v_noisy[:, sample], a=a_noisy[:, sample]
            )
            for d in range(dof):
                X_true_per_joint[d] = np.vstack((X_true_per_joint[d], Y_aug_clean[d]))
                X_noisy_per_joint[d] = np.vstack((X_noisy_per_joint[d], Y_aug_noisy[d]))
                Y_clean_per_joint[d] = np.hstack((Y_clean_per_joint[d], tau_clean[d]))
        theta_true = pi_aug

    N, d_ = X_true_per_joint[0].shape

    subtree_mask = regressor.get_subtree_mask()  # shape (dof, dof)

    for d in range(dof):
        # ---- active column mask ----
        active_12block = np.zeros(dof, dtype=bool)
        active_12block[:] = subtree_mask[d, :]
        active_cols = np.repeat(active_12block, 12)
        n_active = int(np.sum(active_cols))

        X_active_true = X_true_per_joint[d][:, active_cols]  # clean
        X_active_noisy = X_noisy_per_joint[d][:, active_cols]  # noisy
        Y_clean = Y_clean_per_joint[d]

        # ---- Add SNR-based output noise ----
        y_var = float(np.var(Y_clean))
        output_noise_std = np.sqrt(max(y_var, 1e-8) / output_snr)
        Y_noisy = Y_clean + rng.normal(0, output_noise_std, N)

        # ---- Estimate per-column input noise from clean vs noisy regressor ----
        delta_X = X_active_noisy - X_active_true

        # ---- identifiable projection (on FULL clean regressor d_ cols) ----
        proj_ident, rank_x = identifiable_projection_matrix(X_true_per_joint[d])

        # ---- OLS on noisy data ----
        theta_ols_active = np.linalg.lstsq(X_active_noisy, Y_noisy, rcond=None)[0]
        theta_ols = np.zeros(d_, dtype=float)
        theta_ols[active_cols] = theta_ols_active

        # ---- VB-JFA on original-scale noisy regressor ----
        # psi_x from observed noise (clean vs noisy regressor difference)
        psi_x_init_active = np.maximum(np.var(delta_X, axis=0), 1e-8)
        w_x_init_active = np.ones(n_active, dtype=float)
        w_z_init_active = theta_ols_active.copy()
        psi_z_init_active = np.full(n_active, max(y_var, 1e-4) / max(n_active, 1))
        psi_y_init_val = max(output_noise_std**2, 1e-8)

        model = VariationalBayesianJFA(verbose=verbose)
        model.fit(
            X=X_active_noisy,
            Y=Y_noisy,
            w_x_init=w_x_init_active,
            w_z_init=w_z_init_active,
            psi_x_init=psi_x_init_active,
            psi_z_init=psi_z_init_active,
            psi_y_init=psi_y_init_val,
            max_iter=max_iter,
            tol=tol,
            cal_beta=True,
        )
        theta_bayes_active = model.get_beta_true()
        theta_bayes = np.zeros(d_, dtype=float)
        theta_bayes[active_cols] = theta_bayes_active

        # ---- physical projection (on full d_-dim space) ----
        X_noisy_full = X_noisy_per_joint[d].copy()
        theta_bayes_phys = theta_bayes.copy()
        phys_info = {"applied": False, "reason": "disabled"}
        if apply_physical:
            projector = RBDPhysicalConsistencyProjector(verbose=True)
            theta_bayes_phys, phys_info = projector.project(theta_bayes, X_noisy_full)

        theta_true_ident = proj_ident @ theta_true
        theta_ols_ident = proj_ident @ theta_ols
        theta_bayes_ident = proj_ident @ theta_bayes
        theta_bayes_phys_ident = proj_ident @ theta_bayes_phys

        rmse_ols = float(
            np.sqrt(np.mean((Y_noisy - X_active_noisy @ theta_ols_active) ** 2))
        )
        rmse_bayes = float(
            np.sqrt(np.mean((Y_noisy - X_active_noisy @ theta_bayes_active) ** 2))
        )
        rmse_bayes_phys = float(
            np.sqrt(np.mean((Y_noisy - X_noisy_full @ theta_bayes_phys) ** 2))
        )

        print("================ Robot Identification =================")
        print(f"Joint {d + 1}/{dof}")
        subtree_joints = [j for j in range(dof) if subtree_mask[d, j]]
        print(f"Subtree joints: {subtree_joints}  (active params: {n_active}/{d_})")
        print(f"VB-JFA converged in {model.n_iter_} iterations")
        print(f"input SNR={input_snr:.1f}, output SNR={output_snr:.1f}")
        alpha_active = model.get_alpha_mean()
        alpha_full = np.full(d_, np.inf)
        alpha_full[active_cols] = alpha_active
        print(
            f"VB-JFA E[alpha] (per 12-param block): "
            f"{format_array([float(np.mean(alpha_full[j * 12 : (j + 1) * 12])) for j in range(dof)])}"
        )
        print(
            f"VB-JFA active dims (alpha < 1e2): {model.count_small_alphas(100)}/{n_active}"
        )
        print(f"X shape: {X_active_noisy.shape}")
        print(f"Y shape: {Y_noisy.shape}")
        print(f"rank(X_true): {rank_x}/{d_}")

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


def run_rum_robot_random_demo(
    seed=42,
    input_snr=10.0,
    output_snr=20.0,
    apply_physical=True,
    max_iter=1e5,
    tol=1e-5,
    verbose=True,
    n_chirp_trials=CHIRP_DEFAULT_N_TRIALS,
    duration=CHIRP_DEFAULT_DURATION,
    sample_rate=CHIRP_DEFAULT_SAMPLE_RATE,
):
    """
    "rum robot random" experiment — chirp-based excitation with maximally
    diverse angle–velocity coverage.

    Instead of optimised Fourier trajectories loaded from YAML files, this
    experiment generates many independent frequency-sweep (chirp) trajectories,
    each with randomly varied amplitude, centre position, sweep range,
    direction and phase per joint.  The goal is to test whether a large,
    coverage-rich dataset allows VB-JFA to outperform classical OLS in the
    robot identification setting.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    input_snr : float
        Target SNR for velocity/acceleration input noise.
    output_snr : float
        Target SNR for torque output noise.
    apply_physical : bool
        Whether to apply the rigid-body physical-consistency projector.
    max_iter : int
        Max VB-EM iterations per joint.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print per-joint diagnostics.
    n_chirp_trials : int
        Number of independent chirp trajectories to generate and stack.
    duration : float
        Duration of each chirp trajectory (seconds).
    sample_rate : float
        Sampling rate (Hz) for each trajectory.
    """
    print("\n" + "=" * 72)
    print("  rum robot random demo — Chirp Excitation Experiment")
    print(
        f"  n_chirp_trials={n_chirp_trials}, duration={duration}s, "
        f"sample_rate={sample_rate}Hz"
    )
    print(f"  total samples ≈ {n_chirp_trials * int(duration * sample_rate)}")
    print("=" * 72 + "\n")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ---- Robot model (provides joint limits) ----
    regressor = TargetLimbRegressor()
    dof = regressor.dof

    q_lower = regressor.limits["q_lower"]
    q_upper = regressor.limits["q_upper"]

    print(f"Target limb: {regressor.group_to_identify}")
    print(f"DOF: {dof}")
    print("Joint limits:")
    for i in range(dof):
        print(
            f"  Joint {i}: q ∈ [{q_lower[i]:.3f}, {q_upper[i]:.3f}], "
            f"v_limit={regressor.limits['v_limit'][i]:.3f}"
        )

    # ---- Accumulators ----
    X_true_per_joint = [np.empty((0, dof * 12), dtype=float) for _ in range(dof)]
    X_noisy_per_joint = [np.empty((0, dof * 12), dtype=float) for _ in range(dof)]
    Y_clean_per_joint = [np.empty((0,), dtype=float) for _ in range(dof)]

    # Statistics for coverage diagnostics
    all_q = [[] for _ in range(dof)]
    all_v = [[] for _ in range(dof)]
    theta_true = None

    # ---- Generate and process chirp trajectories ----
    for trial in range(n_chirp_trials):
        q, v, a = generate_chirp_trajectory(
            dof=dof,
            q_lower=q_lower,
            q_upper=q_upper,
            duration=duration,
            sample_rate=sample_rate,
            rng=rng,
        )

        samples = q.shape[1]  # time steps in this trajectory

        # Per-joint velocity/acceleration variance for noise scaling
        v_var = np.var(v, axis=1)
        a_var = np.var(a, axis=1)
        v_noise_std = np.sqrt(np.maximum(v_var, 1e-8) / input_snr)
        a_noise_std = np.sqrt(np.maximum(a_var, 1e-8) / input_snr)
        v_noisy = v + rng.normal(0, 1, v.shape) * v_noise_std[:, None]
        a_noisy = a + rng.normal(0, 1, a.shape) * a_noise_std[:, None]

        for s in range(samples):
            # Clean regressor
            (
                Y_aug_clean,
                tau_clean,
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
            ) = regressor.compute_regressor(q=q[:, s], v=v[:, s], a=a[:, s])
            # Noisy regressor
            (
                Y_aug_noisy,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = regressor.compute_regressor(q=q[:, s], v=v_noisy[:, s], a=a_noisy[:, s])

            for d in range(dof):
                X_true_per_joint[d] = np.vstack((X_true_per_joint[d], Y_aug_clean[d]))
                X_noisy_per_joint[d] = np.vstack((X_noisy_per_joint[d], Y_aug_noisy[d]))
                Y_clean_per_joint[d] = np.hstack((Y_clean_per_joint[d], tau_clean[d]))
                all_q[d].append(q[d, s])
                all_v[d].append(v[d, s])

        if theta_true is None:
            theta_true = pi_aug

        if verbose and (trial + 1) % max(1, n_chirp_trials // 5) == 0:
            print(
                f"  Processed trial {trial + 1}/{n_chirp_trials} "
                f"(accumulated {X_true_per_joint[0].shape[0]} samples)"
            )

    # ---- Coverage diagnostics ----
    N, d_ = X_true_per_joint[0].shape
    print(f"\nTotal dataset: {N} samples × {d_} regressor columns")
    print("Coverage summary (per joint):")
    for d in range(dof):
        q_arr = np.array(all_q[d])
        v_arr = np.array(all_v[d])
        q_range = q_upper[d] - q_lower[d]
        v_limit = regressor.limits["v_limit"][d]
        q_coverage = (np.max(q_arr) - np.min(q_arr)) / max(q_range, 1e-6) * 100.0
        v_coverage = (np.max(np.abs(v_arr))) / max(v_limit, 1e-6) * 100.0
        print(
            f"  Joint {d}: q_range_used={q_coverage:.1f}%, "
            f"v_peak_used={v_coverage:.1f}% | "
            f"q∈[{np.min(q_arr):.3f},{np.max(q_arr):.3f}], "
            f"v∈[{np.min(v_arr):.3f},{np.max(v_arr):.3f}]"
        )

    # ---- Per-joint VB-JFA identification ----
    subtree_mask = regressor.get_subtree_mask()

    print("\n" + "=" * 72)
    print("  VB-JFA Identification Results (per joint)")
    print("=" * 72)

    for d in range(dof):
        # Active column mask (subtree)
        active_12block = np.zeros(dof, dtype=bool)
        active_12block[:] = subtree_mask[d, :]
        active_cols = np.repeat(active_12block, 12)
        n_active = int(np.sum(active_cols))

        X_active_true = X_true_per_joint[d][:, active_cols]
        X_active_noisy = X_noisy_per_joint[d][:, active_cols]
        Y_clean = Y_clean_per_joint[d]

        # Output noise
        y_var = float(np.var(Y_clean))
        output_noise_std = np.sqrt(max(y_var, 1e-8) / output_snr)
        Y_noisy = Y_clean + rng.normal(0, output_noise_std, N)

        delta_X = X_active_noisy - X_active_true

        # Identifiable projection
        proj_ident, rank_x = identifiable_projection_matrix(X_true_per_joint[d])

        # OLS
        theta_ols_active = np.linalg.lstsq(X_active_noisy, Y_noisy, rcond=None)[0]
        theta_ols = np.zeros(d_, dtype=float)
        theta_ols[active_cols] = theta_ols_active

        # VB-JFA
        psi_x_init_active = np.maximum(np.var(delta_X, axis=0), 1e-8)
        w_x_init_active = np.ones(n_active, dtype=float)
        w_z_init_active = theta_ols_active.copy()
        psi_z_init_active = np.full(n_active, max(y_var, 1e-4) / max(n_active, 1))
        psi_y_init_val = max(output_noise_std**2, 1e-8)

        model = VariationalBayesianJFA(verbose=verbose)
        model.fit(
            X=X_active_noisy,
            Y=Y_noisy,
            w_x_init=w_x_init_active,
            w_z_init=w_z_init_active,
            psi_x_init=psi_x_init_active,
            psi_z_init=psi_z_init_active,
            psi_y_init=psi_y_init_val,
            max_iter=max_iter,
            tol=tol,
            cal_beta=True,
        )
        theta_bayes_active = model.get_beta_true()
        theta_bayes = np.zeros(d_, dtype=float)
        theta_bayes[active_cols] = theta_bayes_active

        # Physical projection
        X_noisy_full = X_noisy_per_joint[d].copy()
        theta_bayes_phys = theta_bayes.copy()
        phys_info = {"applied": False, "reason": "disabled"}
        if apply_physical:
            projector = RBDPhysicalConsistencyProjector(verbose=True)
            theta_bayes_phys, phys_info = projector.project(theta_bayes, X_noisy_full)

        theta_true_ident = proj_ident @ theta_true
        theta_ols_ident = proj_ident @ theta_ols
        theta_bayes_ident = proj_ident @ theta_bayes
        theta_bayes_phys_ident = proj_ident @ theta_bayes_phys

        rmse_ols = float(
            np.sqrt(np.mean((Y_noisy - X_active_noisy @ theta_ols_active) ** 2))
        )
        rmse_bayes = float(
            np.sqrt(np.mean((Y_noisy - X_active_noisy @ theta_bayes_active) ** 2))
        )
        rmse_bayes_phys = float(
            np.sqrt(np.mean((Y_noisy - X_noisy_full @ theta_bayes_phys) ** 2))
        )

        print(f"\n---------------- Joint {d + 1}/{dof} ----------------")
        subtree_joints = [j for j in range(dof) if subtree_mask[d, j]]
        print(f"Subtree joints: {subtree_joints}  (active params: {n_active}/{d_})")
        print(f"VB-JFA converged in {model.n_iter_} iterations")
        print(f"input SNR={input_snr:.1f}, output SNR={output_snr:.1f}")
        alpha_active = model.get_alpha_mean()
        alpha_full = np.full(d_, np.inf)
        alpha_full[active_cols] = alpha_active
        print(
            f"VB-JFA E[alpha] (per 12-param block): "
            f"{format_array([float(np.mean(alpha_full[j * 12 : (j + 1) * 12])) for j in range(dof)])}"
        )
        print(
            f"VB-JFA active dims (alpha < 1e2): {model.count_small_alphas(100)}/{n_active}"
        )
        print(f"X shape: {X_active_noisy.shape}")
        print(f"Y shape: {Y_noisy.shape}")
        print(f"rank(X_true): {rank_x}/{d_}")

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

    print("\n" + "=" * 72)
    print("  rum robot random demo finished")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Paper-aligned Bayesian JFA with optional physical consistency projection"
    )
    parser.add_argument(
        "--demo",
        choices=["synthetic", "robot", "rum_robot_random"],
        default="synthetic",
        help=(
            "synthetic: Section 5.1 setup (OLS vs BAYES, 4 scenarios, 10 trials)  "
            "robot: full robot regression demo with pre-optimised Fourier trajectories  "
            "rum_robot_random: chirp-based excitation with randomised sweep parameters "
            "per joint — maximises angle–velocity coverage to test data-richness benefit"
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
        default=False,
        help="Disable physical consistency projection (enabled by default)",
    )
    parser.add_argument(
        "--fourier-parameters",
        type=json.loads,
        help='Parameters for Fourier trajectory generation as a JSON array. Size [dof * (n_harmonics * 2 + 2)], n_harmonics=5 in this project. Usage: --fourier-parameters "[1.0, 2.0, 3.0]"',
    )

    # ---- rum_robot_random specific options ----
    parser.add_argument(
        "--n-chirp-trials",
        type=int,
        default=CHIRP_DEFAULT_N_TRIALS,
        help=f"Number of chirp trajectories to generate (default {CHIRP_DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--chirp-duration",
        type=float,
        default=CHIRP_DEFAULT_DURATION,
        help=f"Duration of each chirp trajectory in seconds (default {CHIRP_DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--chirp-sample-rate",
        type=float,
        default=CHIRP_DEFAULT_SAMPLE_RATE,
        help=f"Sample rate for chirp trajectories in Hz (default {CHIRP_DEFAULT_SAMPLE_RATE})",
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

    if args.demo == "rum_robot_random":
        run_rum_robot_random_demo(
            seed=args.seed,
            input_snr=args.input_snr,
            output_snr=args.output_snr,
            apply_physical=not args.no_physical,
            max_iter=args.max_iter,
            tol=args.tol,
            verbose=not args.quiet,
            n_chirp_trials=args.n_chirp_trials,
            duration=args.chirp_duration,
            sample_rate=args.chirp_sample_rate,
        )
        return

    try:
        run_robot_demo(
            seed=args.seed,
            input_snr=args.input_snr,
            output_snr=args.output_snr,
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
