import numpy as np
import yaml

from identification.fourier_trajectory import FourierTrajectory
from identification.target_limb_regressor import TargetLimbRegressor
from cache.bayesian.bayesianJFA import VariationalBayesianJFA

from sko.PSO import PSO
from sko.tools import set_run_mode
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

LEFT_LEG_Q_INDICES = [0, 1, 2, 3, 4, 5]
RIGHT_LEG_Q_INDICES = [6, 7, 8, 9, 10, 11]
WAIST_Q_INDICES = [12]
LEFT_ARM_Q_INDICES = [13, 14, 15, 16, 17]
RIGHT_ARM_Q_INDICES = [18, 19, 20, 21, 22]
NECK_Q_INDICES = [23]

VALID_LIMB_GROUPS = {
    "left_leg": LEFT_LEG_Q_INDICES,
    "right_leg": RIGHT_LEG_Q_INDICES,
    "left_arm": LEFT_ARM_Q_INDICES,
    "right_arm": RIGHT_ARM_Q_INDICES,
    "waist": WAIST_Q_INDICES,
    "neck": NECK_Q_INDICES,
}

GROUP_TO_IDENTIFY = "left_arm"

URDF_PATH = (
    Path(get_package_share_directory("identification"))
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
SAMPLE_RATE = 50.0

# Soft constraint parameters
REG_EPS = 1e-6
RANK_REL_TOL = 1e-4
RANK_ABS_TOL = 1e-10

# ==============================
# PSO parameters
# ==============================
POP_SIZE = 100
MAX_ITER = 500
PSO_W = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5

# Normalized penalty weights
PENALTY_W_Q = 20
PENALTY_W_V = 10
PENALTY_W_TAU = 20
PENALTY_W_MAX = 50
PENALTY_W_COLLISION = 1000

REWARD_ACTIVE = 30.0
REWARD_WEAKLY_ACTIVE = 20.0
REWARD_AMPLITUDE = 30.0

# Progressive penalty schedule: lambda(k) = lambda0 * (1 + alpha * progress)
PENALTY_LAMBDA0 = 400.0
PENALTY_LAMBDA_ALPHA = 2.0

# Limit buffers
Q_LIMIT_BUFFER = 0.1
V_LIMIT_BUFFER = 0.1
TAU_LIMIT_BUFFER = 0.1

RNG_SEED = 70


class PSOoptimizer:
    def __init__(
        self,
        fourier_traj: FourierTrajectory,
        regressor: TargetLimbRegressor,
        jfa: VariationalBayesianJFA,
    ):
        self.fourier_traj = fourier_traj
        self.regressor = regressor
        self.jfa = jfa

        self.d = self.fourier_traj.dim * 12  # input dimension (12 * dof)
        self.N = len(self.fourier_traj.t_array)  # sample number
        self.nq = self.regressor.dof

        print(f"d={self.d}, N={self.N}, nq={self.nq}")

        self.lb, self.ub = self._build_bounds()

        np.random.seed(RNG_SEED)

    def _build_bounds(self):
        dim, harmonics = self.fourier_traj.dim, self.fourier_traj.n_harmonics
        omega_f = self.fourier_traj.omega_f
        n_coeffs_per_joint = harmonics * 2 + 1
        total_coeffs = dim * n_coeffs_per_joint
        lb = np.zeros(total_coeffs)
        ub = np.zeros(total_coeffs)
        for i in range(dim):
            q_lo = self.regressor.q_lower_limit[i] + Q_LIMIT_BUFFER
            q_hi = self.regressor.q_upper_limit[i] - Q_LIMIT_BUFFER
            q_range = (q_hi - q_lo) / 2.0  # half-range for amplitude safety
            # Amplitude bounds: scale by w_k because trajectory uses a_k/w_k, b_k/w_k
            # Conservative: each harmonic gets 1/(2*N) of the half-range
            for k in range(harmonics):
                wk = omega_f * (k + 1)
                amp_bound = wk * q_range / (2.0 * harmonics)
                idx_a = i * n_coeffs_per_joint + k * 2
                idx_b = i * n_coeffs_per_joint + k * 2 + 1
                lb[idx_a] = -amp_bound
                ub[idx_a] = amp_bound
                lb[idx_b] = -amp_bound
                ub[idx_b] = amp_bound
            # q0 bounds: center of joint range
            q_center = (
                self.regressor.q_lower_limit[i] + self.regressor.q_upper_limit[i]
            ) / 2.0
            lb[i * n_coeffs_per_joint + harmonics * 2] = q_center - q_range * 0.3
            ub[i * n_coeffs_per_joint + harmonics * 2] = q_center + q_range * 0.3
        return lb, ub

    def _compute_cost(self, X, Y, w_z_init) -> float:
        """Evaluate excitation quality with diversity bonus.

        Builds a per-parameter hit-count table across joints, then applies
        diminishing returns (sqrt) so that activating *different* parameters
        yields higher reward than activating the *same* parameter repeatedly
        across different joints.
        """
        d_per_joint = X[0].shape[1]  # 12 * dof (same for all joints)
        # n_joints = len(X)

        # hit_count[k] = how many joints activated parameter index k
        hit_active = np.zeros(d_per_joint, dtype=int)
        hit_weakly = np.zeros(d_per_joint, dtype=int)

        for x, y in zip(X, Y):
            self.jfa.fit(
                X=x,
                Y=y,
                psi_x_init=1e-4,
                psi_z_init=1e-4,
                psi_y_init=1e-4,
                w_z_init=w_z_init,
                tol=1e-4,
            )
            hit_active += self.jfa.get_active_mask(threshold=100.0).astype(int)
            hit_weakly += self.jfa.get_active_mask(threshold=1e4).astype(int)

        active_score = float(np.sum(np.sqrt(hit_active)))
        weakly_score = float(np.sum(np.sqrt(hit_weakly)))

        return -(REWARD_ACTIVE * active_score + REWARD_WEAKLY_ACTIVE * weakly_score)

    def fitness_function(self, coeffs: np.ndarray) -> float:
        q_traj, v_traj, a_traj = self.fourier_traj.generate_trajectory(coeffs)
        xim_list = [np.empty((0, self.d)) for _ in range(self.nq)]
        yi_list = [np.empty((0,)) for _ in range(self.nq)]
        total_cost = 0.0
        excitation_cost = 0.0
        collision_count = 0
        collision_penalty = 0.0
        for t in range(self.N):
            (
                Y_aug,
                tau_aug,
                pi_aug,
                pi_inertia,
                pi_friction,
                q_excess,
                v_excess,
                tau_excess,
                q_excess_normalized,
                v_excess_normalized,
                tau_excess_normalized,
                collided,
            ) = self.regressor.compute_regressor(
                q=q_traj[:, t],
                v=v_traj[:, t],
                a=a_traj[:, t],
            )
            if collided:
                # Continuous collision penalty: accumulate based on normalized excess,
                # so PSO can differentiate "mild" from "severe" collisions
                collision_count += 1
                collision_penalty += sum(
                    [
                        PENALTY_W_MAX * q_excess_normalized,
                        PENALTY_W_MAX * v_excess_normalized,
                        PENALTY_W_MAX * tau_excess_normalized,
                    ]
                )
                # Still skip regressor data for collided timesteps
                continue
            if q_excess_normalized or v_excess_normalized or tau_excess_normalized:
                cost = sum(
                    [
                        PENALTY_W_Q * q_excess_normalized,
                        PENALTY_W_V * v_excess_normalized,
                        PENALTY_W_TAU * tau_excess_normalized,
                    ]
                )
                excitation_cost += cost
                total_cost += cost
            for i, yi in enumerate(Y_aug):
                xim_list[i] = np.vstack((xim_list[i], yi))
                yi_list[i] = np.hstack((yi_list[i], tau_aug[i]))
        # Add collision penalty scaled by count (makes all-collision trajectories worse)
        total_cost += collision_penalty + collision_count * PENALTY_W_COLLISION
        if collision_count > 5:
            print(
                f"Quit early due to excessive collisions {collision_count}/{self.N}. Cost:",
                total_cost,
            )
            print(f"coeffs: {coeffs}")
            print("-" * 50)
            return total_cost
        compute_cost = self._compute_cost(xim_list, yi_list, pi_aug)
        amplitude_cost = -np.sqrt(np.sum(coeffs**2)) * REWARD_AMPLITUDE
        total_cost += compute_cost + amplitude_cost
        print(
            f"Converged at iteration {self.jfa.n_iter_}. Total reward: {-total_cost} \n",
            f"  Collision penalty: {collision_penalty}, Collision count: {collision_count}, Excitation cost: {excitation_cost}\n",
            f"  Compute reward: {-compute_cost}, Amplitude reward: {-amplitude_cost}",
        )
        print("coeffs:\n", coeffs)
        print("-" * 50)
        return total_cost


class PSOWithYamlSave(PSO):
    """PSO subclass that saves structured coefficients to YAML at each iteration.

    Uses the same joint_0/joint_1/.../{a, b, q0} format as flat_to_yaml.
    Each iteration is appended as a new top-level key ``iter_N``, preserving
    the full optimization history in a single YAML file.
    """

    def __init__(
        self, save_path="pso_best.yaml", dim_structured=None, n_harmonics=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.save_path = (
            Path(__file__).resolve().parent
            / ".."
            / "trajectory_coefficients"
            / save_path
        )
        self._dim_structured = dim_structured
        self._n_harmonics = n_harmonics

    @staticmethod
    def _flat_to_yaml(flat_array: np.ndarray, dim: int, n_harmonics: int) -> dict:
        """Convert flat coeff array to {joint_i: {a, b, q0}} dict (same as flat_to_yaml)."""
        params = flat_array.reshape(dim, n_harmonics * 2 + 1)
        data = {}
        for i in range(dim):
            a = params[i, 0 : n_harmonics * 2 : 2]
            b = params[i, 1 : n_harmonics * 2 : 2]
            q0 = params[i, -1]
            data[f"joint_{i}"] = {
                "a": [round(float(v), 8) for v in a.tolist()],
                "b": [round(float(v), 8) for v in b.tolist()],
                "q0": round(float(q0), 8),
            }
        return data

    def run(self, max_iter=None, precision=None, N=20):
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0

            # Build structured coefficients (same format as flat_to_yaml)
            coeffs = self.gbest_x.flatten()
            if self._dim_structured is not None and self._n_harmonics is not None:
                coeffs_data = self._flat_to_yaml(
                    coeffs, dim=self._dim_structured, n_harmonics=self._n_harmonics
                )
            else:
                coeffs_data = coeffs.tolist()

            iter_data = {
                "gbest_y": float(self.gbest_y),
                "description": f"Iter: {iter_num}, Best fit: [{self.gbest_y}]",
                "rnd_seed": RNG_SEED,
                "coefficients": coeffs_data,
            }

            # Read -> append -> write back to preserve history
            history = {}
            if self.save_path.exists():
                try:
                    with open(self.save_path, "r") as f:
                        history = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    history = {}
            history[f"iter_{iter_num}"] = iter_data

            with open(self.save_path, "w") as f:
                yaml.dump(history, f, default_flow_style=False, sort_keys=False)

            if self.verbose:
                print(
                    "Iter: {}, Best fit: {} at {}".format(
                        iter_num, self.gbest_y, self.gbest_x
                    )
                )


def main():
    regressor = TargetLimbRegressor(
        urdf_path=URDF_PATH, group_to_identify="left_arm", print_info=True
    )
    fourier_traj = FourierTrajectory(dim=regressor.dof, sample_rate=50)
    jfa = VariationalBayesianJFA(verbose=False)
    optimizer = PSOoptimizer(fourier_traj, regressor, jfa)

    def fitness_wrapper(x):
        return optimizer.fitness_function(x)

    set_run_mode(fitness_wrapper, "multithreading")

    pso = PSOWithYamlSave(
        func=fitness_wrapper,
        dim=fourier_traj.dim * (fourier_traj.n_harmonics * 2 + 1),
        pop=POP_SIZE,
        max_iter=1,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
        lb=optimizer.lb,
        ub=optimizer.ub,
        verbose=True,
        save_path="0729_3.yaml",
        dim_structured=fourier_traj.dim,
        n_harmonics=fourier_traj.n_harmonics,
    )
    pso.run()


if __name__ == "__main__":
    main()
