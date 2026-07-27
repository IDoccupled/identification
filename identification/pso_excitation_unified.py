"""
PSO-based unified excitation trajectory optimization.

Single-stage: uses the FULL augmented regressor Y_aug (inertial + armature + friction)
and maximizes D-optimality over its identifiable subspace.  No per-stage separation.

Usage:
  python -m identification.pso_excitation_unified

Results saved to trajectory_coefficients/pso_unified_{timestamp}.yaml

Y_aug structure per time step (left_arm, 5 DoF):
  [ 50 inertial columns | 5 armature cols (a_i) | 10 friction cols (v_i, sign(v_i)) ]
  = 65 columns per row, 5 rows per time step.
"""

# ruff: noqa: F841

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

import traceback

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
SAMPLE_RATE = 50.0

# PSO parameters
POP = 800
MAX_ITER = 300
AMP_SCALE = 2.0  # moderate amp to excite both inertial and friction
PSO_W = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5
RANDOM_SEED = 67

# Penalty weights
W_Q_LIMIT = 2000.0
W_V_LIMIT = 500.0
W_TAU_LIMIT = 1000.0
W_COLLISION = 100000.0

Q_MARGIN = 0.2
V_MARGIN = 0.2
TAU_MARGIN = 0.2


# ---------------------------------------------------------------------------
def compute_fitness(
    coeffs: np.ndarray,
    ft: FourierTrajectory,
    reg: TargetLimbRegressor,
    verbose: bool = False,
) -> float:
    """Fitness = -(reward) + penalties.  PSO minimizes, so we negate reward.

    Reward: D-optimal (Σ log σ_i) on the identifiable subspace of Y_aug,
    with a soft condition-number penalty.
    """
    q_traj, v_traj, a_traj = ft.generate_trajectory(coeffs)
    N = len(ft.t_array)

    penalty = 0.0
    collision_count = 0
    Y_aug_list = []  # stacked regressor rows

    for t in range(N):
        result = reg.compute_regressor(q=q_traj[:, t], v=v_traj[:, t], a=a_traj[:, t])
        # New API: 19 return values
        (
            Y_aug,  # 0  (dof, dof*13) augmented regressor
            _Y_target_inertial,  # 1
            _Y_target_armature,  # 2
            _Y_target_friction,  # 3
            _tau_aug,  # 4
            _tau_inertia,  # 5
            _tau_armature,  # 6
            _tau_friction,  # 7
            _pi_aug,  # 8
            _pi_inertia,  # 9
            _pi_armature,  # 10
            _pi_friction,  # 11
            _q_excess,  # 12
            _v_excess,  # 13
            _tau_excess,  # 14
            q_excess_norm,  # 15
            v_excess_norm,  # 16
            tau_excess_norm,  # 17
            collided,  # 18
        ) = result

        if collided:
            collision_count += 1
            penalty += W_COLLISION
            continue
        if q_excess_norm:
            penalty += W_Q_LIMIT * q_excess_norm
        if tau_excess_norm:
            penalty += W_TAU_LIMIT * tau_excess_norm
        if v_excess_norm:
            penalty += W_V_LIMIT * v_excess_norm

        Y_aug_list.append(Y_aug)

    # ---- Compute D-optimal reward on Y_aug ----
    if len(Y_aug_list) < 10:
        return 1e9

    Y_full = np.vstack(Y_aug_list)  # (N * dof, 65)

    # Remove structurally zero columns
    col_max = np.abs(Y_full).max(axis=0)
    nonzero_cols = col_max > 1e-12
    if nonzero_cols.sum() == 0:
        return 1e9
    Y_nz = Y_full[:, nonzero_cols]

    try:
        U, S, Vt = np.linalg.svd(Y_nz, full_matrices=False)
        # --- Soft-threshold D-optimal reward ---
        # All singular values participate; small σ are "floored" instead of
        # discarded.  This gives the optimizer continuous gradient to push
        # borderline σ_i upward — no discrete rank jumps.
        sigma_floor = 1e-6 * S[0]
        # Normalised so that σ=sigma_floor → contribution ≈ 0
        reward = float(np.sum(np.log(S + sigma_floor)))
        reward -= len(S) * np.log(sigma_floor)

        # --- Soft condition-number penalty ---
        # Compute effective rank by the same σ_floor, then penalise κ > 1000
        # in that subspace.  Still a soft penalty (no hard rejection).
        r_eff = int(np.sum(S > sigma_floor))
        if r_eff >= 2:
            S_eff = S[:r_eff]
            cond = S_eff[0] / S_eff[-1]
            reward -= max(0.0, cond / 1000.0)
    except np.linalg.LinAlgError:
        reward = -1e3

    total = -(reward) + penalty
    if verbose and np.random.random() < 0.05:
        print(
            f"  reward={reward:.3f}, penalty={penalty:.1f}, "
            f"collisions={collision_count}, total={total:.3f}"
        )
    return total


# ---------------------------------------------------------------------------
def build_bounds(
    ft: FourierTrajectory, reg: TargetLimbRegressor, amp_scale: float = 1.0
):
    """Build PSO bounds for Fourier coefficients respecting joint limits."""
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


def _save_yaml(
    coeffs,
    ft,
    t_start,
    elapsed,
    pop,
    max_iter,
    amp_scale,
    best_fitness,
    extra_meta=None,
):
    """Save trajectory coefficients to YAML. Returns path."""
    yaml_dict = _coeffs_to_yaml_dict(coeffs, ft.dim, ft.n_harmonics)
    meta = {
        "stage": "unified",
        "started": t_start,
        "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
        "pop": pop,
        "iter": max_iter,
        "amp_scale": amp_scale,
        "seed": RANDOM_SEED,
        "best_fitness": float(best_fitness[0])
        if hasattr(best_fitness, "__iter__")
        else float(best_fitness),
    }
    if extra_meta:
        meta.update(extra_meta)
    yaml_dict["_meta"] = meta
    uid = datetime.now().strftime("%y%m%d_%H%M%S")
    yaml_path = YAML_DIR / f"pso_unified_{uid}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Saved to: {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
def main():
    stage = "unified"
    pop = POP
    max_iter = MAX_ITER
    amp_scale = AMP_SCALE

    print(f"\n{'=' * 60}\n  PSO Unified Excitation Optimization\n{'=' * 60}")
    print(f"  pop={pop}, iter={max_iter}, amp_scale={amp_scale}, seed={RANDOM_SEED}")

    np.random.seed(RANDOM_SEED)

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
        return compute_fitness(x, ft, reg)

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

    try:
        pso.run()
        status = "completed"
        error_info = None
    except (KeyboardInterrupt, MemoryError, Exception) as e:
        status = "interrupted"
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"\n  !!! PSO interrupted: {type(e).__name__}: {e}")
        print("  Saving current best before exit ...")

    elapsed = time.time() - ts

    coeffs_best = pso.gbest_x.flatten()
    print(f"Best fitness: {pso.gbest_y}")

    extra = {"status": status}
    if error_info:
        extra["error"] = error_info
    _save_yaml(
        coeffs_best,
        ft,
        t_start,
        elapsed,
        pop,
        max_iter,
        amp_scale,
        pso.gbest_y,
        extra_meta=extra,
    )

    if status == "interrupted":
        raise

    # ---- Diagnostic report ----
    q, v, a = ft.generate_trajectory(coeffs_best)
    collisions = 0
    Y_aug_all = []
    for t_idx in range(len(ft.t_array)):
        res = reg.compute_regressor(q=q[:, t_idx], v=v[:, t_idx], a=a[:, t_idx])
        if res[18]:  # collided
            collisions += 1
        else:
            Y_aug_all.append(res[0])  # Y_aug

    print(f"Collisions: {collisions}/{len(ft.t_array)}")
    print(f"|v|max={np.abs(v).max():.1f}, |a|max={np.abs(a).max():.1f}")

    # Per-joint zero-crossings
    zc = np.sum(np.diff(np.sign(v), axis=1) != 0, axis=1)
    print(f"Zero-crossings: {list(zc)}")

    if len(Y_aug_all) > 10:
        Y_full = np.vstack(Y_aug_all)
        Y_nz = Y_full[:, np.abs(Y_full).max(axis=0) > 1e-12]
        _, S, _ = np.linalg.svd(Y_nz, full_matrices=False)
        sigma_floor = 1e-6 * S[0]
        r_eff = int(np.sum(S > sigma_floor))
        dopt_soft = float(np.sum(np.log(S + sigma_floor))) - len(S) * np.log(
            sigma_floor
        )
        cond = S[0] / S[r_eff - 1] if r_eff >= 2 else float("inf")
        print(
            f"Y_aug total cols: {Y_nz.shape[1]}, "
            f"eff. rank (σ>{sigma_floor:.1e}): {r_eff}, "
            f"D-opt (soft): {dopt_soft:.2f}, "
            f"Cond: {cond:.1f}"
        )

    print(f"\n{'=' * 60}\n  Done.\n{'=' * 60}")


if __name__ == "__main__":
    main()
