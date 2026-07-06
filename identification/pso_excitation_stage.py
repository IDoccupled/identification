#!/usr/bin/env python3
"""
PSO-based excitation trajectory optimization for per-stage identification.

Usage:
  python -m identification.pso_excitation_stage balance   # Stage 1
  python -m identification.pso_excitation_stage armature  # Stage 2
  python -m identification.pso_excitation_stage friction  # Stage 3
  python -m identification.pso_excitation_stage all       # Run all three

Results saved to trajectory_coefficients/pso_{stage}.yaml
"""

import sys
import yaml
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from identification.fourier_trajectory import FourierTrajectory
from identification.target_limb_regressor import TargetLimbRegressor

from sko.PSO import PSO
from sko.tools import set_run_mode
from ament_index_python.packages import get_package_share_directory

# ---------------------------------------------------------------------------
URDF_PATH = (
    Path(get_package_share_directory("identification"))
    / "resource"
    / "robot"
    / "urdf"
    / "serial_pm_v2_identify.urdf"
).resolve()

YAML_DIR = Path(__file__).resolve().parent.parent / "trajectory_coefficients"

N_HARMONICS = 5
TRAJ_PERIOD = 5.0
SAMPLE_RATE = 100.0

# Per-stage PSO parameters (overnight run)
STAGE_CONFIG = {
    "balance": {"pop": 150, "iter": 800, "amp_scale": 1.0},
    "armature": {"pop": 120, "iter": 1000, "amp_scale": 3.0},
    "friction": {"pop": 120, "iter": 700, "amp_scale": 1.0},
}
PSO_W = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5

W_Q_LIMIT = 50.0
W_V_LIMIT = 30.0
W_TAU_LIMIT = 50.0
W_COLLISION = 500.0

Q_MARGIN = 0.15
V_MARGIN = 0.1
TAU_MARGIN = 0.1


# ---------------------------------------------------------------------------
def compute_fitness(
    coeffs: np.ndarray,
    ft: FourierTrajectory,
    reg: TargetLimbRegressor,
    stage: str,
    verbose: bool = False,
) -> float:
    """Fitness = -(reward) + penalties.  PSO minimizes, so we negate reward."""

    q_traj, v_traj, a_traj = ft.generate_trajectory(coeffs)
    N = len(ft.t_array)

    penalty = 0.0
    collision_count = 0

    # Storage for reward computation
    Y_list = []  # inertial regressor rows
    ddq_list = []  # accelerations
    dq_list = []  # velocities
    sign_dq_list = []  # sign of velocity

    for t in range(N):
        result = reg.compute_regressor(q=q_traj[:, t], v=v_traj[:, t], a=a_traj[:, t])
        (
            Y_aug,
            tau_aug,
            pi_aug,
            pi_inertia,
            pi_friction,
            q_excess,
            v_excess,
            tau_excess,
            q_excess_norm,
            v_excess_norm,
            tau_excess_norm,
            collided,
        ) = result

        if collided:
            collision_count += 1
            penalty += W_COLLISION
            continue

        # Limit violations
        if q_excess_norm > 0:
            penalty += W_Q_LIMIT * q_excess_norm
        if v_excess_norm > 0:
            penalty += W_V_LIMIT * v_excess_norm
        if tau_excess_norm > 0:
            penalty += W_TAU_LIMIT * tau_excess_norm

        if q_excess_norm > 50 or v_excess_norm > 50 or tau_excess_norm > 50:
            continue  # skip severely violating steps

        # Collect regressor data
        # Y_aug is the full augmented regressor (inertial + friction)
        # For balance: use Y_inertial = Y_aug[:, :-2] (exclude damping, friction cols)
        # For friction: use dq and sign(dq)
        if stage == "balance":
            Y_inertial = Y_aug[:, :-2]  # drop damping and frictionloss columns
            Y_list.append(Y_inertial)
        elif stage == "armature":
            ddq_list.append(a_traj[:, t])
        elif stage == "friction":
            dq_list.append(v_traj[:, t])
            sign_dq_list.append(np.sign(v_traj[:, t]))

    # ---- Compute reward ----
    reward = 0.0

    if stage == "balance":
        if len(Y_list) < 10:
            return 1e9
        Y_full = np.vstack(Y_list)

        # Remove structurally zero columns (constant zero across all rows)
        col_max = np.abs(Y_full).max(axis=0)
        nonzero_cols = col_max > 1e-12
        if nonzero_cols.sum() == 0:
            return 1e9
        Y_nz = Y_full[:, nonzero_cols]

        # SVD to find identifiable subspace
        # Y is structurally rank-deficient → only top r singular values matter
        try:
            U, S, Vt = np.linalg.svd(Y_nz, full_matrices=False)
            # Numerical rank: singular values > eps * S_max
            eps_rank = 1e-6
            r = int(np.sum(S > eps_rank * S[0]))
            if r < 3:
                reward = -1e3  # too few identifiable params → poor excitation
            else:
                S_id = S[:r]
                # D-optimal: maximize Σ log(σ_i) over identifiable subspace
                # = proportional to log(det) of Fisher info in that subspace
                reward = float(np.sum(np.log(S_id + 1e-12)))
                # Bonus: penalize high condition number within identifiable subspace
                cond_identifiable = S_id[0] / S_id[-1]
                reward -= max(0.0, cond_identifiable / 1000.0)  # soft penalty
        except np.linalg.LinAlgError:
            reward = -1e3

    elif stage == "armature":
        if len(ddq_list) < 10:
            return 1e9
        ddq_all = np.array(ddq_list)
        # Reward: max |ddq| per joint (push peak acceleration, not average)
        # This directly maximizes armature identifiability since sensitivity ∝ ddq
        peak_ddq = np.max(np.abs(ddq_all), axis=0)
        reward = float(np.mean(peak_ddq))  # average peak across joints

    elif stage == "friction":
        if len(dq_list) < 10:
            return 1e9
        dq_all = np.array(dq_list)
        # Zero-crossings
        zc = np.sum(np.diff(np.sign(dq_all), axis=0) != 0, axis=0)
        zc_reward = float(np.mean(zc))
        # Condition number of [dq, sign(dq)]
        X = np.column_stack([dq_all, np.array(sign_dq_list)])
        if X.shape[1] >= 2 and X.shape[0] > X.shape[1]:
            _, S, _ = np.linalg.svd(X, full_matrices=False)
            cond = S[0] / S[-1] if S[-1] > 1e-10 else 1e6
            cond_reward = 1.0 / max(cond, 1.0)  # small cond → high reward
        else:
            cond_reward = 0.0
        # Torque from damping+friction: reward large |dq|
        dq_rms = float(np.sqrt(np.mean(dq_all**2)))
        reward = zc_reward * 10.0 + cond_reward * 5.0 + dq_rms * 0.5

    total = -(reward) + penalty
    if verbose and np.random.random() < 0.05:  # print ~5% of evaluations
        print(
            f"  reward={reward:.3f}, penalty={penalty:.1f}, "
            f"collisions={collision_count}, total={total:.3f}"
        )
    return total


# ---------------------------------------------------------------------------
def build_bounds(
    ft: FourierTrajectory, reg: TargetLimbRegressor, amp_scale: float = 1.0
):
    """Build PSO bounds for Fourier coefficients respecting joint limits.

    amp_scale > 1.0 allows larger accelerations (useful for armature stage).
    """
    dim, harmonics = ft.dim, ft.n_harmonics
    omega_f = ft.omega_f
    n_coeffs = harmonics * 2 + 1
    total = dim * n_coeffs
    lb = np.zeros(total)
    ub = np.zeros(total)

    for i in range(dim):
        q_lo = reg.q_lower_limit[i] + Q_MARGIN
        q_hi = reg.q_upper_limit[i] - Q_MARGIN
        q_range = (q_hi - q_lo) / 2.0

        for k in range(harmonics):
            wk = omega_f * (k + 1)
            amp_bound = wk * q_range / (2.0 * harmonics) * amp_scale
            idx_a = i * n_coeffs + k * 2
            idx_b = i * n_coeffs + k * 2 + 1
            lb[idx_a] = -amp_bound
            ub[idx_a] = amp_bound
            lb[idx_b] = -amp_bound
            ub[idx_b] = amp_bound
            ub[idx_a] = amp_bound
            lb[idx_b] = -amp_bound
            ub[idx_b] = amp_bound

        q_center = (reg.q_lower_limit[i] + reg.q_upper_limit[i]) / 2.0
        lb[i * n_coeffs + harmonics * 2] = q_center - q_range * 0.5
        ub[i * n_coeffs + harmonics * 2] = q_center + q_range * 0.5

    return lb, ub


# ---------------------------------------------------------------------------
def _coeffs_to_yaml_dict(coeffs: np.ndarray, dim: int, n_harmonics: int) -> dict:
    """Convert flat PSO coeffs to {joint_0: {a,b,q0}, ...} for YAML."""
    params = coeffs.reshape(dim, n_harmonics * 2 + 1)
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


def run_stage(stage: str):
    cfg = STAGE_CONFIG[stage]
    pop = cfg["pop"]
    max_iter = cfg["iter"]
    amp_scale = cfg["amp_scale"]

    print(f"\n{'=' * 60}\n  PSO Stage: {stage}\n{'=' * 60}")
    print(f"  pop={pop}, iter={max_iter}, amp_scale={amp_scale}")

    reg = TargetLimbRegressor(
        urdf_path=URDF_PATH, group_to_identify="left_arm", print_info=False
    )
    ft = FourierTrajectory(dim=reg.dof, sample_rate=SAMPLE_RATE)
    ft.omega_f = 2.0 * np.pi / TRAJ_PERIOD
    ft.t_array = np.linspace(
        0, TRAJ_PERIOD, int(TRAJ_PERIOD * SAMPLE_RATE), endpoint=False
    )

    lb, ub = build_bounds(ft, reg, amp_scale=amp_scale)
    dim_total = ft.dim * (ft.n_harmonics * 2 + 1)
    print(f"PSO dim={dim_total}")

    t_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = time.time()

    def fitness(x):
        return compute_fitness(x, ft, reg, stage)

    set_run_mode(fitness, "multithreading")

    pso = PSO(
        func=fitness,
        dim=dim_total,
        pop=pop,
        max_iter=max_iter,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
        lb=lb,
        ub=ub,
        verbose=True,
    )
    pso.run()

    elapsed = time.time() - ts
    t_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    coeffs_best = pso.gbest_x.flatten()
    print(f"Best fitness: {pso.gbest_y}")

    # ---- Save to YAML with timestamp and unique suffix ----
    yaml_dict = _coeffs_to_yaml_dict(coeffs_best, ft.dim, ft.n_harmonics)
    yaml_dict["_meta"] = {
        "stage": stage,
        "started": t_start,
        "finished": t_end,
        "elapsed_s": round(elapsed, 1),
        "pop": pop,
        "iter": max_iter,
        "amp_scale": amp_scale,
        "best_fitness": float(pso.gbest_y[0])
        if hasattr(pso.gbest_y, "__iter__")
        else float(pso.gbest_y),
    }
    # Unique filename: pso_{stage}_{YYMMDD_HHMMSS}.yaml
    uid = datetime.now().strftime("%y%m%d_%H%M%S")
    yaml_path = YAML_DIR / f"pso_{stage}_{uid}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Saved to: {yaml_path}")

    # ---- Diagnostic report ----
    q, v, a = ft.generate_trajectory(coeffs_best)
    collisions = 0
    Y_all = []
    dq_all = []
    for t_idx in range(len(ft.t_array)):
        res = reg.compute_regressor(q=q[:, t_idx], v=v[:, t_idx], a=a[:, t_idx])
        if res[-1]:
            collisions += 1
        else:
            if stage == "balance":
                Y_all.append(res[0][:, :-2])
            dq_all.append(v[:, t_idx])

    print(f"Collisions: {collisions}/{len(ft.t_array)}")
    print(f"|dq|max={np.abs(v).max():.1f}, |ddq|max={np.abs(a).max():.1f}")
    zc = np.sum(np.diff(np.sign(v), axis=1) != 0, axis=1)
    print(f"Zero-crossings: {list(zc)}")

    if stage == "balance" and len(Y_all) > 10:
        Y_full = np.vstack(Y_all)
        Y_nz = Y_full[:, np.abs(Y_full).max(axis=0) > 1e-12]
        _, S, _ = np.linalg.svd(Y_nz, full_matrices=False)
        r = int(np.sum(S > 1e-6 * S[0]))
        print(
            f"Identifiable rank: {r}/{Y_nz.shape[1]}, "
            f"D-opt: {np.sum(np.log(S[:r] + 1e-12)):.2f}, "
            f"Cond: {S[0] / S[r - 1]:.1f}"
        )

    if stage == "armature":
        print(f"Peak |ddq|: {np.abs(a).max():.1f}")

    if stage == "friction":
        dq_arr = np.array(dq_all)
        X = np.column_stack([dq_arr, np.sign(dq_arr)])
        _, Sf, _ = np.linalg.svd(X, full_matrices=False)
        print(
            f"Friction cond: {Sf[0] / Sf[-1]:.1f}, RMS dq: {np.sqrt(np.mean(dq_arr**2)):.2f}"
        )


def main():
    stages = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if stages == ["all"]:
        stages = ["balance", "armature", "friction"]

    for s in stages:
        if s not in ("balance", "armature", "friction"):
            print(f"Unknown stage: {s}. Use: balance, armature, friction, all")
            continue
        run_stage(s)


if __name__ == "__main__":
    main()
