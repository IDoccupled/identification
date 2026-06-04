#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from ament_index_python.packages import get_package_share_directory
from identification.target_limb_regressor import TargetLimbRegressor

LEFT_LEG_Q_INDICES  = [0, 1, 2, 3, 4, 5]
RIGHT_LEG_Q_INDICES = [6, 7, 8, 9, 10, 11]
WAIST_Q_INDICES     = [12]
LEFT_ARM_Q_INDICES  = [13, 14, 15, 16, 17]
RIGHT_ARM_Q_INDICES = [18, 19, 20, 21, 22]
NECK_Q_INDICES      = [23]

VALID_LIMB_GROUPS = {
    'left_leg': LEFT_LEG_Q_INDICES,
    'right_leg': RIGHT_LEG_Q_INDICES,
    'left_arm': LEFT_ARM_Q_INDICES,
    'right_arm': RIGHT_ARM_Q_INDICES,
    'waist': WAIST_Q_INDICES,
    'neck': NECK_Q_INDICES
}

GROUP_TO_IDENTIFY = 'left_arm' 

URDF_PATH = (
    Path(get_package_share_directory('identification'))
    / "resource"
    / "robot"
    / "urdf"
    / "serial_pm_v2_identify.urdf"
).resolve()

# ==============================
# Tunable trajectory configuration
# ==============================

# Series expansion parameters
N_HARMONICS = 5
TRAJ_PERIOD = 10.0
SAMPLE_RATE   = 50.0

# Soft constraint parameters
REG_EPS = 1e-6
RANK_REL_TOL = 1e-4
RANK_ABS_TOL = 1e-10

# Normalized penalty weights
PENALTY_W_Q = 2.0
PENALTY_W_V = 1.0
PENALTY_W_TAU = 2.0
PENALTY_W_MAX = 5.0

# Progressive penalty schedule: lambda(k) = lambda0 * (1 + alpha * progress)
PENALTY_LAMBDA0 = 400.0
PENALTY_LAMBDA_ALPHA = 2.0

# Limit buffers
Q_LIMIT_BUFFER = 0.1
V_LIMIT_BUFFER = 0.1
TAU_LIMIT_BUFFER = 0.1

# ==============================
# PSO parameters
# ==============================
POP_SIZE = 10
MAX_ITER = 600
PSO_W    = 0.7
PSO_C1   = 1.5
PSO_C2   = 1.5

RNG_SEED = 114


class FourierTrajectory:
    def __init__(
            self,
            regressor: TargetLimbRegressor = TargetLimbRegressor(
                urdf_path=URDF_PATH,
                group_to_identify=GROUP_TO_IDENTIFY,
                print_info=False
            ),
            plot_trajectory: bool = False
    ):
        self.regressor = regressor

        self.dim = len(self.regressor.group_to_identify)
        self.q_upper    = np.array(self.regressor.q_upper_limit) - Q_LIMIT_BUFFER
        self.q_lower    = np.array(self.regressor.q_lower_limit) + Q_LIMIT_BUFFER
        self.v_limit     = np.array(self.regressor.v_limit) + V_LIMIT_BUFFER
        self.tau_limit   = np.array(self.regressor.tau_limit) + TAU_LIMIT_BUFFER

        self.omega_f      = 2.0 * np.pi / TRAJ_PERIOD
        self.t_array     = np.linspace(0, TRAJ_PERIOD, int(TRAJ_PERIOD * SAMPLE_RATE), endpoint=False)
        self.n_harmonics = N_HARMONICS

        self.lb, self.ub = self._build_coeff_bounds()
        self._reset_iter_log_state()
        self.plot_trajectory = plot_trajectory
        np.random.seed(RNG_SEED)

    def _build_coeff_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.ones(self.dim * (self.n_harmonics * 2 + 2))
        ub = np.ones_like(lb)
        for i in range(self.dim):
            lb[i * (self.n_harmonics * 2 + 2) : (i + 1) * (self.n_harmonics * 2 + 2) - 1] *= self.q_lower[i] + Q_LIMIT_BUFFER
            ub[i * (self.n_harmonics * 2 + 2) : (i + 1) * (self.n_harmonics * 2 + 2) - 1] *= self.q_upper[i] - Q_LIMIT_BUFFER
            lb[(i + 1) * (self.n_harmonics * 2 + 2) - 1] = -TRAJ_PERIOD
            ub[(i + 1) * (self.n_harmonics * 2 + 2) - 1] = TRAJ_PERIOD
        return lb, ub

    def _reset_iter_log_state(self):
        self._iter_eval_counter = 0
        self._iter_best_objective = np.inf
        self._iter_best_info = np.nan
        self._iter_best_info_eff = np.nan
        self._iter_best_penalty = np.nan
        self._iter_best_weighted_penalty = np.nan
        self._iter_best_max_violation = np.nan
        self._iter_best_rank = -1
        self._iter_best_sval_count = 0
        self._iter_best_sigma_r = np.nan
        self._iter_best_kappa_eff = np.nan
        self._iter_best_q_over_max = np.nan
        self._iter_best_v_over_max = np.nan
        self._iter_best_tau_over_max = np.nan

    def _effective_identifiability_metrics(self, svals):
        sigma_max = float(svals[0])
        rank_tol = max(RANK_ABS_TOL, RANK_REL_TOL * sigma_max)
        rank = int(np.sum(svals > rank_tol))
        if rank > 0:
            sigma_r = float(svals[rank - 1])
            kappa_eff = np.inf if sigma_r <= 1e-12 else sigma_max / sigma_r
            info_eff = float(np.sum(np.log(svals[:rank] ** 2 + REG_EPS)))
        else:
            sigma_r = 0.0
            kappa_eff = np.inf
            info_eff = float("-inf")
        return rank, sigma_r, kappa_eff, info_eff
    
    def _compute_cost(self, Y_aug, q_excess_normalized, v_excess_normalized, tau_excess_normalized) -> float:
        pass

    def generate_trajectory(self, coeffs: np.ndarray):
        """
        Generate joint trajectories (q, v, a) from Fourier coefficients.

        :param coeffs: Flattened array of shape (dim * n_harmonics * 2,) containing [A1_1, B1_1, A2_1, B2_1, ..., A_N_1, B_N_1, q_1, t_1,
                                                                                 A1_2, B1_2, A2_2, B2_2, ..., A_N_2, B_N_2, q_2, t_2,
                                                                                 ...] for each joint.
        :return: Tuple of (q_traj, v_traj, a_traj), each of shape (len(t_array), dim)
        """
        params = coeffs.reshape(self.dim, self.n_harmonics * 2 + 2)
        q_traj = np.zeros((self.dim, len(self.t_array)))
        v_traj = np.zeros_like(q_traj)
        a_traj = np.zeros_like(q_traj)

        for i in range(self.dim):
            q0 = params[i, -2]
            t0 = params[i, -1]
            a = params[i, 0 : self.n_harmonics * 2 : 2]
            b = params[i, 1 : self.n_harmonics * 2 : 2]
            harmonics = np.arange(1, self.n_harmonics + 1, dtype=float)
            w = self.omega_f * harmonics

            sin_wt = np.sin(np.outer(self.t_array + t0, w))
            cos_wt = np.cos(np.outer(self.t_array + t0, w))

            q_traj[i, :] = (
                q0 
                + sin_wt @ (a / w) 
                - cos_wt @ (b / w)
            )
            v_traj[i, :] = (
                cos_wt @ a 
                + sin_wt @ b
            )
            a_traj[i, :] = (
                - sin_wt @ (a * w) 
                + cos_wt @ (b * w)
            )
        return q_traj, v_traj, a_traj
    
    def fitness_function(self, coeffs: np.ndarray) -> float:
        q_traj, v_traj, a_traj = self.generate_trajectory(coeffs)
        total_cost = 0.0
        for t in range(len(self.t_array)):
            (Y_aug, tau_aug, 
             q_excess, v_excess, tau_excess, 
             q_excess_normalized, v_excess_normalized, tau_excess_normalized) = self.regressor.compute_regressor(
                q=q_traj[:, t],
                v=v_traj[:, t],
                a=a_traj[:, t],
                print_info=False
            )
            cost = self._compute_cost(Y_aug, q_excess_normalized, v_excess_normalized, tau_excess_normalized)
            total_cost += cost
        return total_cost / len(self.t_array)
    
def main():
    regressor = TargetLimbRegressor(
        urdf_path=URDF_PATH,
        group_to_identify=GROUP_TO_IDENTIFY,
    )
    traj, _, _ = FourierTrajectory(regressor=regressor).generate_trajectory(
        coeffs=np.random.uniform(-0.5, 0.5, size=(regressor.dof * (N_HARMONICS * 2 + 2)))
    )
    print("生成的轨迹形状:", traj.shape)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    for i in range(traj.shape[0]):
        plt.subplot(traj.shape[0], 1, i + 1)
        plt.plot(traj[i, :])
        plt.title(f"Joint {i}")
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()