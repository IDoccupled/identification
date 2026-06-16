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
PENALTY_W_Q = 20
PENALTY_W_V = 10
PENALTY_W_TAU = 20
PENALTY_W_MAX = 50

# Progressive penalty schedule: lambda(k) = lambda0 * (1 + alpha * progress)
PENALTY_LAMBDA0 = 400.0
PENALTY_LAMBDA_ALPHA = 2.0

# Limit buffers
Q_LIMIT_BUFFER = 0.1
V_LIMIT_BUFFER = 0.1
TAU_LIMIT_BUFFER = 0.1

RNG_SEED = 114


class PSOoptimizer:
    def __init__(
            self, 
            fourier_traj: FourierTrajectory, 
            regressor: TargetLimbRegressor,
            jfa: VariationalBayesianJFA
            ):
        self.fourier_traj = fourier_traj
        self.regressor = regressor
        self.jfa = jfa

        self.d = self.fourier_traj.dim * 12 # input dimension (12 * dof)
        self.N = len(self.fourier_traj.t_array) # sample number
        self.nq = self.regressor.dof

        print(f'd={self.d}, N={self.N}, nq={self.nq}')

        self.lb, self.ub = self._build_bounds()
        # self._reset_iter_log_state()

        np.random.seed(RNG_SEED)

    # def _reset_iter_log_state(self):
    #     self._iter_eval_counter = 0
    #     self._iter_best_objective = np.inf
    #     self._iter_best_info = np.nan
    #     self._iter_best_info_eff = np.nan
    #     self._iter_best_penalty = np.nan
    #     self._iter_best_weighted_penalty = np.nan
    #     self._iter_best_max_violation = np.nan
    #     self._iter_best_rank = -1
    #     self._iter_best_sval_count = 0
    #     self._iter_best_sigma_r = np.nan
    #     self._iter_best_kappa_eff = np.nan
    #     self._iter_best_q_over_max = np.nan
    #     self._iter_best_v_over_max = np.nan
    #     self._iter_best_tau_over_max = np.nan

    def _build_bounds(self):
        dim, harmonics = self.fourier_traj.dim, self.fourier_traj.n_harmonics
        n_coeffs_per_joint = harmonics * 2 + 2
        total_coeffs = dim * n_coeffs_per_joint
        lb = np.zeros(total_coeffs)
        ub = np.zeros(total_coeffs)
        for i in range(dim):
            # Amplitude bounds
            lb[i * n_coeffs_per_joint : i * n_coeffs_per_joint + harmonics * 2] = self.regressor.q_lower_limit[i] + Q_LIMIT_BUFFER
            ub[i * n_coeffs_per_joint : i * n_coeffs_per_joint + harmonics * 2] = self.regressor.q_upper_limit[i] - Q_LIMIT_BUFFER
            # q0 bounds
            lb[i * n_coeffs_per_joint + harmonics * 2] = self.regressor.q_lower_limit[i]
            ub[i * n_coeffs_per_joint + harmonics * 2] = self.regressor.q_upper_limit[i]
            # t0 bounds
            lb[i * n_coeffs_per_joint + harmonics * 2 + 1] = 0.0
            ub[i * n_coeffs_per_joint + harmonics * 2 + 1] = 0.0
        return lb, ub

    def _compute_cost(self, X, Y) -> float:
        """Evaluate excitation quality: higher = more inertial params activated."""
        # Y_list is a list of (N, d) regressor matrices, one per sample
        n_active = 0
        n_weakly_activated = 0

        for x, y in zip(X, Y):
            self.jfa.fit(x, y, cal_beta=False, tol=1e-5)
            n_active += self.jfa.count_small_alphas(threshold=100.0)
            n_weakly_activated += self.jfa.count_small_alphas(threshold=1e4)

        print(f"Active params: {n_active}, Weakly activated params: {n_weakly_activated}")
            
        return -float(n_active)

    def fitness_function(self, coeffs: np.ndarray) -> float:
        q_traj, v_traj, a_traj = self.fourier_traj.generate_trajectory(coeffs)
        xim_list = [np.empty((0, self.d)) for _ in range(self.nq)]
        yi_list = [np.empty((0,)) for _ in range(self.nq)]
        total_cost = 0.0
        excess_cost = 0.0
        for t in range(self.N):
            (Y_aug, tau_aug, 
             pi_aug, pi_inertia, pi_friction,
             q_excess, v_excess, tau_excess, 
             q_excess_normalized, v_excess_normalized, tau_excess_normalized,
             collided) = self.regressor.compute_regressor(
                q=q_traj[:, t],
                v=v_traj[:, t],
                a=a_traj[:, t],
            )
            if collided:
                print("Collision detected")
                return 1e6
            if q_excess_normalized or v_excess_normalized or tau_excess_normalized:
                cost = sum([
                    PENALTY_W_Q * q_excess_normalized,
                    PENALTY_W_V * v_excess_normalized,
                    PENALTY_W_TAU * tau_excess_normalized
                ])
                excess_cost += cost
                total_cost += cost
            for i, yi in enumerate(Y_aug):
                xim_list[i] = np.vstack((xim_list[i], yi))
                yi_list[i] = np.hstack((yi_list[i], tau_aug[i]))
        cost = self._compute_cost(xim_list, yi_list)
        total_cost += cost
        print(total_cost)
        return total_cost
    
def main():
    regressor = TargetLimbRegressor(
        urdf_path=URDF_PATH,
        group_to_identify='left_arm',
        print_info=True
    )
    fourier_traj = FourierTrajectory(dim=regressor.dof)
    jfa = VariationalBayesianJFA(verbose=False)
    optimizer = PSOoptimizer(fourier_traj, regressor, jfa)
    pso = PSO(
        func=optimizer.fitness_function,
        dim=fourier_traj.dim * (fourier_traj.n_harmonics * 2 + 2),
        pop=POP_SIZE,
        max_iter=MAX_ITER,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
        lb=optimizer.lb,
        ub=optimizer.ub,
        verbose=True
    )
    pso.run()

if __name__ == "__main__":    
    main()