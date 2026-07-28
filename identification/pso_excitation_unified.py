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
POP = 500
MAX_ITER = 50
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

# Per-parameter variance reward weight.
# Balances D-optimal (overall info) vs per-parameter precision.
# Increase to push PSO harder toward uniform per-parameter identifiability.
W_PARAM_PER = 8.0

Q_MARGIN = 0.2
V_MARGIN = 0.2
TAU_MARGIN = 0.2


# ---------------------------------------------------------------------------
def compute_fitness(
    coeffs: np.ndarray,
    ft: FourierTrajectory,
    reg: TargetLimbRegressor,
    theta_nominal: np.ndarray,
    verbose: bool = True,
) -> float:
    """Fitness = -(reward) + penalties.  PSO minimizes, so we negate reward.

    Reward composition:
      - D-optimal (Σ log σ_i) on the identifiable subspace of Y_aug
      - Soft condition-number penalty (κ > 1000)
      - Per-parameter variance reward (piecewise scoring by relative std)
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

    # Track reward components for diagnostic output
    r_dopt = 0.0  # D-optimal
    r_cond = 0.0  # condition-number penalty (≤ 0)
    r_param = 0.0  # per-parameter variance reward
    scores = np.array([])  # per-param scores for diagnostics

    try:
        U, S, Vt = np.linalg.svd(Y_nz, full_matrices=False)
        # --- Soft-threshold D-optimal reward ---
        # All singular values participate; small σ are "floored" instead of
        # discarded.  This gives the optimizer continuous gradient to push
        # borderline σ_i upward — no discrete rank jumps.
        sigma_floor = 1e-6 * S[0]
        # Normalised so that σ=sigma_floor → contribution ≈ 0
        r_dopt = float(np.sum(np.log(S + sigma_floor)))
        r_dopt -= len(S) * np.log(sigma_floor)

        # --- Soft condition-number penalty ---
        # Compute effective rank by the same σ_floor, then penalise κ > 1000
        # in that subspace.  Still a soft penalty (no hard rejection).
        r_eff = int(np.sum(S > sigma_floor))
        if r_eff >= 2:
            S_eff = S[:r_eff]
            cond = S_eff[0] / S_eff[-1]
            r_cond = -max(0.0, cond / 1000.0)

        # --- Per-parameter variance reward ---
        # Compute each parameter's relative std via SVD: std_i ∝ sqrt([V Σ⁻² Vᵀ]_ii)
        # Then apply piecewise scoring: reward precise params, penalise noisy ones.
        if r_eff >= 2:
            # Vt rows correspond to Y_nz columns → map back via nonzero_cols
            V_r = Vt[:r_eff, :].T  # (n_nz, r_eff)
            weighted = V_r / S_eff[np.newaxis, :]  # V_{ij} / σ_j
            var_per_param_nz = np.sum(weighted**2, axis=1)  # (n_nz,)
            std_raw_nz = np.sqrt(var_per_param_nz + 1e-12)

            # Map nominal parameter values to the non-zero column space
            theta_nom_nz = theta_nominal[nonzero_cols]  # (n_nz,)

            # Normalised relative standard deviation (dimensionless)
            std_norm = std_raw_nz / (np.abs(theta_nom_nz) + 1e-8)

            # Piecewise scoring per parameter:
            #   rel_std < 0.1  →  +1.0  (saturated reward)
            #   0.1 .. 1.0     →  linear transition +1.0 → -0.5
            #   rel_std > 1.0  →  -0.5 - log(rel_std)  (growing penalty)
            scores = np.where(
                std_norm < 0.1,
                1.0,
                np.where(
                    std_norm < 1.0,
                    1.0 - 1.5 * (std_norm - 0.1) / 0.9,
                    -0.5 - np.log(std_norm),
                ),
            )
            r_param = W_PARAM_PER * float(np.sum(scores))
    except np.linalg.LinAlgError:
        r_dopt = -1e3

    reward = r_dopt + r_cond + r_param
    total = -(reward) + penalty
    if verbose and np.random.random() < 0.05:
        n_good = int(np.sum(scores > 0.5)) if len(scores) else 0
        n_ok = int(np.sum((scores >= -0.5) & (scores <= 0.5))) if len(scores) else 0
        n_bad = int(np.sum(scores < -0.5)) if len(scores) else 0
        print(
            f"  d_opt={r_dopt:.1f}, cond={r_cond:.1f}, param={r_param:.1f} "
            f"[good:{n_good} ok:{n_ok} bad:{n_bad}], "
            f"reward={reward:.1f}, penalty={penalty:.1f}, "
            f"coll={collision_count}, total={total:.1f}"
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


# ---------------------------------------------------------------------------
# Inertial parameter names per link (Pinocchio toDynamicParameters order)
_INERTIAL_PARAM_NAMES = [
    "mass",
    "mcx",
    "mcy",
    "mcz",
    "Ixx",
    "Ixy",
    "Iyy",
    "Ixz",
    "Iyz",
    "Izz",
]


def _build_param_names(joint_names: list[str]) -> list[str]:
    """Build human-readable parameter names matching Y_aug column order.

    Y_aug = [50 inertial | 5 armature | 10 friction] for 5-DoF limb.
    """
    names = []
    for jname in joint_names:
        for pname in _INERTIAL_PARAM_NAMES:
            names.append(f"{jname}/{pname}")
    for jname in joint_names:
        names.append(f"{jname}/armature")
    for jname in joint_names:
        names.append(f"{jname}/damping")
        names.append(f"{jname}/frictionloss")
    return names


def compute_regressor_diagnostics(
    Y_full: np.ndarray,
    theta_nominal: np.ndarray,
    param_names: list[str],
) -> dict:
    """Compute comprehensive diagnostics from stacked regressor Y_full.

    Returns a dict suitable for YAML serialisation.
    """
    # Remove structurally zero columns
    col_max = np.abs(Y_full).max(axis=0)
    nonzero_cols = col_max > 1e-12
    n_total = Y_full.shape[1]
    n_nz = int(nonzero_cols.sum())

    if n_nz == 0:
        return {"error": "All columns are structurally zero"}

    Y_nz = Y_full[:, nonzero_cols]

    try:
        U, S, Vt = np.linalg.svd(Y_nz, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"error": "SVD failed"}

    sigma_floor = 1e-6 * S[0]
    r_eff = int(np.sum(S > sigma_floor))
    cond = float(S[0] / S[r_eff - 1]) if r_eff >= 2 else float("inf")

    # D-optimal (soft)
    r_dopt = float(np.sum(np.log(S + sigma_floor))) - len(S) * np.log(sigma_floor)

    # Condition-number penalty
    r_cond = -max(0.0, cond / 1000.0) if r_eff >= 2 else 0.0

    # Per-parameter variance
    param_entries = []
    r_param = 0.0
    n_good = n_ok = n_bad = 0

    if r_eff >= 2:
        S_eff = S[:r_eff]
        V_r = Vt[:r_eff, :].T
        weighted = V_r / S_eff[np.newaxis, :]
        var_nz = np.sum(weighted**2, axis=1)
        std_raw_nz = np.sqrt(var_nz + 1e-12)
        theta_nom_nz = theta_nominal[nonzero_cols]
        std_norm = std_raw_nz / (np.abs(theta_nom_nz) + 1e-8)

        # Expand to full parameter space (fill NaN for zero columns)
        rel_std_full = np.full(n_total, np.nan)
        score_full = np.full(n_total, np.nan)
        nz_indices = np.where(nonzero_cols)[0]

        for j, full_idx in enumerate(nz_indices):
            rel_std_full[full_idx] = float(std_norm[j])
            sn = float(std_norm[j])
            if sn < 0.1:
                sc = 1.0
            elif sn < 1.0:
                sc = 1.0 - 1.5 * (sn - 0.1) / 0.9
            else:
                sc = -0.5 - np.log(sn)
            score_full[full_idx] = sc

            quality = "good" if sc > 0.5 else ("ok" if sc >= -0.5 else "bad")
            if quality == "good":
                n_good += 1
            elif quality == "ok":
                n_ok += 1
            else:
                n_bad += 1

            param_entries.append(
                {
                    "idx": int(full_idx),
                    "name": param_names[full_idx]
                    if full_idx < len(param_names)
                    else f"param_{full_idx}",
                    "nominal": float(theta_nominal[full_idx])
                    if full_idx < len(theta_nominal)
                    else 0.0,
                    "rel_std": float(sn),
                    "score": float(sc),
                    "quality": quality,
                }
            )

        r_param = W_PARAM_PER * float(np.sum(score_full[nz_indices]))

    return {
        "reward_breakdown": {
            "d_opt": round(r_dopt, 2),
            "cond_penalty": round(r_cond, 2),
            "param_reward": round(r_param, 2),
            "total_reward": round(r_dopt + r_cond + r_param, 2),
        },
        "regression": {
            "total_cols": n_total,
            "nonzero_cols": n_nz,
            "eff_rank": r_eff,
            "cond": round(cond, 1),
            "sigma_floor": float(sigma_floor),
            "singular_values": [
                round(float(s), 3) for s in S[: min(r_eff + 5, len(S))]
            ],
        },
        "param_quality": {
            "n_good": n_good,
            "n_ok": n_ok,
            "n_bad": n_bad,
        },
        "per_param": param_entries,
    }


# ---------------------------------------------------------------------------
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
    diagnostics=None,
):
    """Save trajectory coefficients and diagnostics to YAML. Returns path."""
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
        "w_param_per": W_PARAM_PER,
        "best_fitness": float(best_fitness[0])
        if hasattr(best_fitness, "__iter__")
        else float(best_fitness),
    }
    if extra_meta:
        meta.update(extra_meta)
    yaml_dict["_meta"] = meta
    if diagnostics:
        yaml_dict["_diagnostics"] = diagnostics
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

    # Extract nominal (CAD) parameter vector for per-param variance normalisation.
    # pi_aug = [50 inertial | 5 armature | 10 friction], matches Y_aug columns.
    theta_nominal = np.hstack(
        [
            np.hstack(
                [
                    reg.model.inertias[joint_id + 1].toDynamicParameters()
                    for joint_id in reg.group_to_identify
                ]
            ),
            np.hstack(
                [reg.target_joint_infos[idx]["armature"] for idx in range(reg.dof)]
            ),
            np.hstack(
                [
                    [
                        reg.target_joint_infos[idx]["damping"],
                        reg.target_joint_infos[idx]["friction"],
                    ]
                    for idx in range(reg.dof)
                ]
            ),
        ]
    )
    print(f"theta_nominal shape: {theta_nominal.shape}")

    # Build human-readable parameter names matching Y_aug column order
    joint_names = [reg.target_joint_infos[i]["name"] for i in range(reg.dof)]
    param_names = _build_param_names(joint_names)
    print(
        f"param_names: {len(param_names)} entries, e.g. {param_names[0]}, ..., {param_names[-1]}"
    )

    lb, ub = build_bounds(ft, reg, amp_scale=amp_scale)
    dim_total = ft.dim * (ft.n_harmonics * 2 + 1)
    print(f"PSO dim={dim_total}")

    t_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = time.time()

    def fitness(x):
        return compute_fitness(x, ft, reg, theta_nominal)

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

    # ---- Compute diagnostics on best trajectory ----
    q, v, a = ft.generate_trajectory(coeffs_best)
    collisions = 0
    Y_aug_all = []
    for t_idx in range(len(ft.t_array)):
        res = reg.compute_regressor(q=q[:, t_idx], v=v[:, t_idx], a=a[:, t_idx])
        if res[18]:  # collided
            collisions += 1
        else:
            Y_aug_all.append(res[0])

    zc = np.sum(np.diff(np.sign(v), axis=1) != 0, axis=1)
    traj_stats = {
        "collisions": f"{collisions}/{len(ft.t_array)}",
        "v_max": round(float(np.abs(v).max()), 2),
        "a_max": round(float(np.abs(a).max()), 2),
        "zero_crossings": [int(x) for x in zc],
    }

    diagnostics = {}
    if len(Y_aug_all) > 10:
        Y_full = np.vstack(Y_aug_all)
        diagnostics = compute_regressor_diagnostics(Y_full, theta_nominal, param_names)
        diagnostics["trajectory_stats"] = traj_stats

        # Console summary
        rd = diagnostics.get("reward_breakdown", {})
        rg = diagnostics.get("regression", {})
        pq = diagnostics.get("param_quality", {})
        print(
            f"  d_opt={rd.get('d_opt', '?')}, cond_pen={rd.get('cond_penalty', '?')}, "
            f"param={rd.get('param_reward', '?')}, "
            f"rank={rg.get('eff_rank', '?')}/{rg.get('nonzero_cols', '?')}, "
            f"κ={rg.get('cond', '?')}, "
            f"good/ok/bad={pq.get('n_good', '?')}/{pq.get('n_ok', '?')}/{pq.get('n_bad', '?')}"
        )
        print(
            f"  Collisions: {traj_stats['collisions']}, "
            f"|v|max={traj_stats['v_max']}, |a|max={traj_stats['a_max']}, "
            f"zc={traj_stats['zero_crossings']}"
        )
    else:
        print("  Not enough data for diagnostics.")
        diagnostics = {"error": "Too few valid samples"}

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
        diagnostics=diagnostics,
    )

    if status == "interrupted":
        raise

    print(f"\n{'=' * 60}\n  Done.\n{'=' * 60}")


if __name__ == "__main__":
    main()
