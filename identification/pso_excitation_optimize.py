import numpy as np

from identification.fourier_trajectory import FourierTrajectory
from identification.target_limb_regressor import TargetLimbRegressor
from identification.bayesianJFA import VariationalBayesianJFA

from sko.PSO import PSO
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

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

# ==============================
# PSO parameters
# ==============================
POP_SIZE = 10
MAX_ITER = 600
PSO_W    = 0.7
PSO_C1   = 1.5
PSO_C2   = 1.5

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

RNG_SEED = 114


class PSOFourierTrajectory(FourierTrajectory):
    def __init__(self, regressor=None, plot_trajectory=False):
        super().__init__(regressor=regressor, plot_trajectory=plot_trajectory)
        self._reset_iter_log_state()

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