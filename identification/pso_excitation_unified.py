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

from dataclasses import dataclass
import argparse
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

# ============================================================================
#  Identification task setup
# ============================================================================
URDF_PATH = (
    Path(get_package_share_directory("identification"))
    / "resource"
    / "robot"
    / "urdf"
    / "serial_pm_v2_identify.urdf"
).resolve()

YAML_DIR = Path(__file__).resolve().parent.parent / "trajectory_coefficients"
DEFAULT_TARGET_GROUP = "left_leg"  # overridable with --group

FIXED_HOME_POSE: dict[int, float] = {13: -2.5}

# ============================================================================
#  Trajectory parameterisation
# ============================================================================
N_HARMONICS = 5  # number of Fourier harmonics per joint
TRAJ_PERIOD = 5.0  # [s] trajectory duration
SAMPLE_RATE = 50.0  # [Hz] trajectory sample rate (coarse for PSO speed)

# ============================================================================
#  PSO hyper-parameters
# ============================================================================
POP = 200
MAX_ITER = 100
AMP_SCALE = 2.0  # Fourier coefficient amplitude scale
PSO_W = 0.7  # inertia weight
PSO_C1 = 1.5  # cognitive acceleration
PSO_C2 = 1.5  # social acceleration
RANDOM_SEED = 67

# ============================================================================
#  Constraint margins
# ============================================================================
Q_MARGIN = 0.2  # [rad]  position margin from joint limits
V_MARGIN = 0.2  # [rad/s] velocity margin
TAU_MARGIN = 0.2  # [Nm]   torque margin


# ============================================================================
#  Reward / penalty weights  —  all independently tunable
# ============================================================================
@dataclass
class RewardConfig:
    """All tunable reward & penalty parameters in one place.

    For the diagnostic function the same parameters are used.
    """

    # --- Constraint-penalty weights (applied per violation, summed over time) ---
    w_q_limit: float = 2000.0  # position-limit violation multiplier
    w_v_limit: float = 500.0  # velocity-limit violation multiplier
    w_tau_limit: float = 1000.0  # torque-limit violation multiplier
    w_collision: float = 100000.0  # per-collision penalty

    # --- D-optimal: soft SVD threshold ---
    sigma_floor_rel: float = 1e-6  # σ_floor = sigma_floor_rel · σ_max

    # --- Condition-number soft penalty ---
    # r_cond = -w_cond · max(0, κ / cond_threshold)^cond_penalty_power
    cond_threshold: float = 100.0  # κ > cond_threshold → penalty active
    w_cond: float = 1.0  # overall weight multiplier (heavier than linear default)
    cond_penalty_power: float = 2.0  # exponent: 1=linear, 2=quadratic

    # --- Per-parameter variance reward (tanh score) ---
    # score = tanh(tanh_k · (1 − rel_std / tanh_rel_zero)) ∈ (−1, 1)
    #   rel_std → 0             : score → +1  (reward, bounded above)
    #   rel_std = tanh_rel_zero : score = 0  (reward ↔ penalty boundary)
    #   rel_std → ∞             : score → −1  (bad penalty, bounded below)
    # rank_deficient & small params are EXCLUDED from the rating (score = 0).
    w_param_per: float = 50.0  # overall weight for per-param term
    tanh_rel_zero: float = (
        1.0  # rel_std where score = 0 (below = reward, above = penalty)
    )
    tanh_k: float = 1.0  # tanh steepness (larger → sharper transition, bigger gradient)
    abs_std_good: float = (
        0.01  # abs_std below this → floor score at ≥ 0 (never penalised)
    )
    nominal_small: float = (
        1e-4  # |nominal| < this → quality "small" (excluded from rating)
    )
    nullspace_threshold: float = (
        0.3  # nullspace weight > this → quality "rank_deficient" (excluded from rating)
    )


# Default config instance
RWD = RewardConfig()


# ---------------------------------------------------------------------------
def _cond_penalty(cond: float, cfg: RewardConfig) -> float:
    """Soft condition-number penalty.

    Active only when κ > cfg.cond_threshold; grows as
        -cfg.w_cond · max(0, κ / cfg.cond_threshold)^cfg.cond_penalty_power
    (power 1 = linear, power 2 = quadratic).
    """
    if cond <= 0.0:
        return 0.0
    ratio = cond / cfg.cond_threshold
    return -cfg.w_cond * max(0.0, ratio) ** cfg.cond_penalty_power


# ---------------------------------------------------------------------------
def compute_fitness(
    coeffs: np.ndarray,
    ft: FourierTrajectory,
    reg: TargetLimbRegressor,
    theta_nominal: np.ndarray,
    cfg: RewardConfig,
    verbose: bool = True,
) -> float:
    """Fitness = -(reward) + penalties.  PSO minimizes, so we negate reward.

    Reward composition:
      - D-optimal (Σ log σ_i) on the identifiable subspace of Y_aug (soft threshold)
      - Soft condition-number penalty  (κ > cfg.cond_threshold)
      - Per-parameter variance reward  (piecewise scoring by relative std)
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
            Y_aug,
            _Y_target_inertial,
            _Y_target_armature,
            _Y_target_friction,
            _tau_aug,
            _tau_inertia,
            _tau_armature,
            _tau_friction,
            _pi_aug,
            _pi_inertia,
            _pi_armature,
            _pi_friction,
            _q_excess,
            _v_excess,
            _tau_excess,
            q_excess_norm,
            v_excess_norm,
            tau_excess_norm,
            collided,
        ) = result

        if collided:
            collision_count += 1
            penalty += cfg.w_collision
            continue
        if q_excess_norm:
            penalty += cfg.w_q_limit * q_excess_norm
        if tau_excess_norm:
            penalty += cfg.w_tau_limit * tau_excess_norm
        if v_excess_norm:
            penalty += cfg.w_v_limit * v_excess_norm

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
    r_dopt = 0.0
    r_cond = 0.0
    r_param = 0.0
    scores = np.array([])
    rated = np.zeros(0, dtype=bool)  # rated = nonzero & not rank_deficient/small

    try:
        U, S, Vt = np.linalg.svd(Y_nz, full_matrices=False)
        # --- Soft-threshold D-optimal reward ---
        sigma_floor = cfg.sigma_floor_rel * S[0]
        r_dopt = float(np.sum(np.log(S + sigma_floor)))
        r_dopt -= len(S) * np.log(sigma_floor)

        # --- Soft condition-number penalty ---
        r_eff = int(np.sum(S > sigma_floor))
        if r_eff >= 2:
            S_eff = S[:r_eff]
            cond = S_eff[0] / S_eff[-1]
            r_cond = _cond_penalty(cond, cfg)

        # --- Per-parameter variance reward (tanh, rated params only) ---
        if r_eff >= 2:
            V_r = Vt[:r_eff, :].T  # (n_nz, r_eff)
            weighted = V_r / S_eff[np.newaxis, :]
            var_per_param_nz = np.sum(weighted**2, axis=1)
            std_raw_nz = np.sqrt(var_per_param_nz + 1e-12)

            theta_nom_nz = theta_nominal[nonzero_cols]
            std_norm = std_raw_nz / (np.abs(theta_nom_nz) + 1e-8)

            # Rated = nonzero & NOT rank_deficient & NOT small (excluded from rating)
            nullspace_weight = np.zeros_like(var_per_param_nz)
            if r_eff < Y_nz.shape[1]:
                null_V = Vt[r_eff:, :].T  # (n_nz, n_nz - r_eff)
                nullspace_weight = np.sum(null_V**2, axis=1)  # ∈ [0, 1]
            is_rank_def = nullspace_weight > cfg.nullspace_threshold
            is_small = np.abs(theta_nom_nz) < cfg.nominal_small
            rated = ~is_rank_def & ~is_small

            scores = np.zeros_like(std_norm)
            if rated.any():
                scores[rated] = _score_param_std(
                    std_norm[rated], std_raw_nz[rated], cfg
                )
            r_param = cfg.w_param_per * float(np.sum(scores))
    except np.linalg.LinAlgError:
        r_dopt = -1e3

    reward = r_dopt + r_cond + r_param
    total = -(reward) + penalty
    if verbose and np.random.random() < 0.05:
        if len(scores) and len(rated) == len(scores):
            n_good = int(np.sum((scores > 0.5) & rated))
            n_ok = int(np.sum(((scores >= -0.5) & (scores <= 0.5)) & rated))
            n_bad = int(np.sum((scores < -0.5) & rated))
        else:
            n_good = n_ok = n_bad = 0
        print(
            f"  d_opt={r_dopt:.1f}, cond={r_cond:.1f}, param={r_param:.1f} "
            f"[good:{n_good} ok:{n_ok} bad:{n_bad}], "
            f"reward={reward:.1f}, penalty={penalty:.1f}, "
            f"coll={collision_count}, total={total:.1f}"
        )
    return total


def _score_param_std(
    std_norm: np.ndarray,
    abs_std: np.ndarray,
    cfg: RewardConfig,
) -> np.ndarray:
    """tanh-based per-parameter score ∈ (−1, 1), bounded both ways.

    score = tanh(tanh_k · (1 − rel_std / tanh_rel_zero))

      - rel_std → 0             : score → +1 (reward saturated; upper bound)
      - rel_std = tanh_rel_zero : score =  0 (reward ↔ penalty boundary)
      - rel_std → ∞             : score → −1 (bad penalty saturated; lower bound)

    tanh yields a smooth reward gradient (steepest around the boundary) and is
    naturally bounded, so a single extreme rel_std can no longer blow up the
    reward (this replaces the old piecewise + clamp scheme).  Dual threshold:
    if abs_std < cfg.abs_std_good the score is floored at ≥ 0 (tiny absolute
    uncertainty is never penalised).
    """
    score = np.tanh(cfg.tanh_k * (1.0 - std_norm / cfg.tanh_rel_zero))
    abs_good_mask = abs_std < cfg.abs_std_good
    return np.where(abs_good_mask, np.maximum(score, 0.0), score)


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
    cfg: RewardConfig,
    traj_stats: dict | None = None,
) -> dict:
    """Compute comprehensive diagnostics from stacked regressor Y_full.

    All 65 parameters are recorded, with quality categories:
      - "null"            — structurally zero column (constant zero across all rows)
      - "rank_deficient"  — significant nullspace component (linear dependency)
      - "small"           — |nominal| < cfg.nominal_small (negligible dynamics impact)
      - "good" / "ok" / "bad"  — tanh scoring from _score_param_std
                                 (rank_deficient/small are NOT scored, score = 0)

    Priority: null > rank_deficient > small > good/ok/bad
    """
    n_total = Y_full.shape[1]
    col_max = np.abs(Y_full).max(axis=0)
    nonzero_cols = col_max > 1e-12
    n_nz = int(nonzero_cols.sum())
    nz_indices = np.where(nonzero_cols)[0]

    if n_nz == 0:
        return {"error": "All columns are structurally zero"}

    Y_nz = Y_full[:, nonzero_cols]

    try:
        U, S, Vt = np.linalg.svd(Y_nz, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"error": "SVD failed"}

    sigma_floor = cfg.sigma_floor_rel * S[0]
    r_eff = int(np.sum(S > sigma_floor))
    cond = float(S[0] / S[r_eff - 1]) if r_eff >= 2 else float("inf")

    # D-optimal (soft)
    r_dopt = float(np.sum(np.log(S + sigma_floor)) - len(S) * np.log(sigma_floor))

    # Condition-number penalty
    r_cond = _cond_penalty(cond, cfg) if r_eff >= 2 else 0.0
    cond_detail = {
        "penalty": float(round(r_cond, 2)),
        "cond": float(round(cond, 1)) if r_eff >= 2 else None,
        "cond_threshold": cfg.cond_threshold,
        "w_cond": cfg.w_cond,
        "cond_penalty_power": cfg.cond_penalty_power,
        "cond_ratio": (
            float(round(cond / cfg.cond_threshold, 4)) if r_eff >= 2 else None
        ),
        "sigma_max": float(round(float(S[0]), 3)) if len(S) else None,
        "sigma_min_eff": (float(round(float(S[r_eff - 1]), 3)) if r_eff >= 2 else None),
    }

    # ---- Per-parameter variance & nullspace analysis ----
    param_entries = []
    quality_summary: dict[str, list] = {
        "null": [],
        "rank_deficient": [],
        "small": [],
        "good": [],
        "ok": [],
        "bad": [],
    }
    r_param = 0.0

    # Compute per-param std for nonzero columns
    std_raw_nz = np.full(n_nz, np.nan)
    std_norm_nz = np.full(n_nz, np.nan)
    scores_nz = np.zeros(n_nz)
    nullspace_weight = np.zeros(n_nz)
    rated_nz = np.zeros(n_nz, dtype=bool)

    if r_eff >= 2:
        S_eff = S[:r_eff]
        V_r = Vt[:r_eff, :].T
        weighted = V_r / S_eff[np.newaxis, :]
        var_nz = np.sum(weighted**2, axis=1)
        std_raw_nz = np.sqrt(var_nz + 1e-12)
        theta_nom_nz = theta_nominal[nonzero_cols]
        std_norm_nz = std_raw_nz / (np.abs(theta_nom_nz) + 1e-8)

        # Nullspace analysis: strong nullspace projection → rank-deficient
        if r_eff < n_nz:
            null_V = Vt[r_eff:, :].T  # (n_nz, n_nz - r_eff)
            nullspace_weight = np.sum(null_V**2, axis=1)  # ∈ [0, 1]

        # Rated = nonzero & NOT rank_deficient & NOT small (excluded from rating)
        is_rank_def = nullspace_weight > cfg.nullspace_threshold
        is_small = np.abs(theta_nom_nz) < cfg.nominal_small
        rated_nz = ~is_rank_def & ~is_small

        # tanh score only for rated params; non-rated → 0 (neutral)
        if rated_nz.any():
            scores_nz[rated_nz] = _score_param_std(
                std_norm_nz[rated_nz], std_raw_nz[rated_nz], cfg
            )
        r_param = cfg.w_param_per * float(np.sum(scores_nz))

    # ---- Build per-param entries for ALL 65 columns ----
    nz_idx_map = {int(idx): j for j, idx in enumerate(nz_indices)}

    for full_idx in range(n_total):
        name = (
            param_names[full_idx]
            if full_idx < len(param_names)
            else f"param_{full_idx}"
        )
        nominal = (
            float(theta_nominal[full_idx]) if full_idx < len(theta_nominal) else 0.0
        )

        # --- Structurally zero column ---
        if full_idx not in nz_idx_map:
            quality_summary["null"].append(int(full_idx))
            param_entries.append(
                {
                    "idx": int(full_idx),
                    "name": name,
                    "quality": "null",
                }
            )
            continue

        j = nz_idx_map[full_idx]
        sn = float(std_norm_nz[j])
        sc = float(scores_nz[j])
        as_ = float(std_raw_nz[j])
        nw = float(nullspace_weight[j])

        # --- Determine quality (priority order) ---
        if nw > cfg.nullspace_threshold:
            quality = "rank_deficient"
        elif abs(nominal) < cfg.nominal_small:
            quality = "small"
        elif sc > 0.5:
            quality = "good"
        elif sc >= -0.5:
            quality = "ok"
        else:
            quality = "bad"

        quality_summary[quality].append(int(full_idx))

        param_entries.append(
            {
                "idx": int(full_idx),
                "name": name,
                "nominal": nominal,
                "abs_std": as_,
                "rel_std": sn,
                "nullspace_weight": nw,
                "score": sc,
                "quality": quality,
            }
        )

    # Build quality_summary with names for readability
    quality_summary_named = {}
    for q, idx_list in quality_summary.items():
        quality_summary_named[q] = {
            "count": len(idx_list),
            "indices": idx_list,
            "names": [
                param_names[i] if i < len(param_names) else f"param_{i}"
                for i in idx_list
            ],
        }

    # ---- Param-reward breakdown (like cond_detail) ----
    # value = w_param_per · sum_scores (rated params only);
    # sum_scores = Σ good + Σ ok + Σ bad  (rank_deficient/small excluded)
    n_rated = int(rated_nz.sum())
    if n_rated:
        good_m = rated_nz & (scores_nz > 0.5)
        ok_m = rated_nz & (scores_nz >= -0.5) & (scores_nz <= 0.5)
        bad_m = rated_nz & (scores_nz < -0.5)
        rated_scores = scores_nz[rated_nz]
        param_detail = {
            "value": float(round(r_param, 2)),
            "w_param_per": cfg.w_param_per,
            "sum_scores": float(round(float(np.sum(rated_scores)), 4)),
            "n_nonzero": int(n_nz),
            "n_rated": n_rated,
            "score_min": float(round(float(rated_scores.min()), 4)),
            "score_max": float(round(float(rated_scores.max()), 4)),
            "score_mean": float(round(float(rated_scores.mean()), 4)),
            "by_quality": {
                "good": {
                    "count": int(good_m.sum()),
                    "score_sum": float(round(float(scores_nz[good_m].sum()), 4)),
                },
                "ok": {
                    "count": int(ok_m.sum()),
                    "score_sum": float(round(float(scores_nz[ok_m].sum()), 4)),
                },
                "bad": {
                    "count": int(bad_m.sum()),
                    "score_sum": float(round(float(scores_nz[bad_m].sum()), 4)),
                },
            },
        }
    else:
        param_detail = {
            "value": float(round(r_param, 2)),
            "w_param_per": cfg.w_param_per,
            "sum_scores": 0.0,
            "n_nonzero": int(n_nz),
            "n_rated": 0,
            "score_min": None,
            "score_max": None,
            "score_mean": None,
            "by_quality": None,
        }

    # Assemble reward_breakdown (excess is moved here from trajectory_stats)
    reward_breakdown = {
        "d_opt": float(round(r_dopt, 2)),
        "cond_penalty": cond_detail,
        "param_reward": param_detail,
        "total_reward": float(round(r_dopt + r_cond + r_param, 2)),
    }
    traj_stats_out = traj_stats
    if traj_stats is not None and "excess" in traj_stats:
        reward_breakdown["excess"] = traj_stats["excess"]
        traj_stats_out = {k: v for k, v in traj_stats.items() if k != "excess"}

    result = {
        "reward_breakdown": reward_breakdown,
        "regression": {
            "total_cols": int(n_total),
            "nonzero_cols": int(n_nz),
            "eff_rank": int(r_eff),
            "nullspace_dim": int(n_nz - r_eff) if r_eff < n_nz else 0,
            "cond": float(round(cond, 1)),
            "sigma_floor": float(sigma_floor),
            "singular_values": [
                float(round(float(s), 3)) for s in S[: min(r_eff + 5, len(S))]
            ],
        },
        "quality_summary": quality_summary_named,
        "per_param": param_entries,
    }
    if traj_stats_out is not None:
        # trajectory_stats sits right before regression
        result = {
            "reward_breakdown": result["reward_breakdown"],
            "trajectory_stats": traj_stats_out,
            "regression": result["regression"],
            "quality_summary": result["quality_summary"],
            "per_param": result["per_param"],
        }
    return result


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
    cfg: RewardConfig,
    group: str,
    extra_meta=None,
    diagnostics=None,
):
    """Save trajectory coefficients and diagnostics to YAML. Returns path."""
    yaml_dict = _coeffs_to_yaml_dict(coeffs, ft.dim, ft.n_harmonics)
    meta = {
        "stage": "unified",
        "group": group,
        "started": t_start,
        "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
        "pop": pop,
        "iter": max_iter,
        "amp_scale": amp_scale,
        "seed": RANDOM_SEED,
        "reward_config": {
            "w_q_limit": cfg.w_q_limit,
            "w_v_limit": cfg.w_v_limit,
            "w_tau_limit": cfg.w_tau_limit,
            "w_collision": cfg.w_collision,
            "sigma_floor_rel": cfg.sigma_floor_rel,
            "cond_threshold": cfg.cond_threshold,
            "w_cond": cfg.w_cond,
            "cond_penalty_power": cfg.cond_penalty_power,
            "w_param_per": cfg.w_param_per,
            "tanh_rel_zero": cfg.tanh_rel_zero,
            "tanh_k": cfg.tanh_k,
            "abs_std_good": cfg.abs_std_good,
            "nominal_small": cfg.nominal_small,
            "nullspace_threshold": cfg.nullspace_threshold,
        },
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
    yaml_path = YAML_DIR / f"pso_unified_{uid}_{group}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Saved to: {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PSO unified excitation trajectory optimization"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=DEFAULT_TARGET_GROUP,
        help="limb group to identify, e.g. left_arm, left_leg (default: %(default)s)",
    )
    args = parser.parse_args()
    target_group = args.group

    cfg = RWD  # use the default RewardConfig; edit RWD above to tune

    print(f"\n{'=' * 60}\n  PSO Unified Excitation Optimization\n{'=' * 60}")
    print(
        f"  target_group={target_group}, pop={POP}, iter={MAX_ITER}, amp_scale={AMP_SCALE}, seed={RANDOM_SEED}"
    )
    print(
        f"  reward cfg: w_q={cfg.w_q_limit}, w_v={cfg.w_v_limit}, "
        f"w_tau={cfg.w_tau_limit}, w_coll={cfg.w_collision}, "
        f"σ_floor={cfg.sigma_floor_rel}, cond_thr={cfg.cond_threshold}, "
        f"cond_w={cfg.w_cond}, cond_p={cfg.cond_penalty_power}, "
        f"w_param={cfg.w_param_per}, "
        f"tanh_z0={cfg.tanh_rel_zero}, tanh_k={cfg.tanh_k}, "
        f"abs_good={cfg.abs_std_good}, "
        f"nom_small={cfg.nominal_small}"
    )

    np.random.seed(RANDOM_SEED)

    reg = TargetLimbRegressor(
        urdf_path=URDF_PATH,
        group_to_identify=target_group,
        print_info=False,
        fixed_pose=FIXED_HOME_POSE if target_group == "left_leg" else None,
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

    lb, ub = build_bounds(ft, reg, amp_scale=AMP_SCALE)
    dim_total = ft.dim * (ft.n_harmonics * 2 + 1)
    print(f"PSO dim={dim_total}")

    t_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = time.time()

    def fitness(x):
        return compute_fitness(x, ft, reg, theta_nominal, cfg)

    set_run_mode(fitness, "multithreading")

    pso = PSO(
        func=fitness,
        dim=dim_total,
        pop=POP,
        max_iter=MAX_ITER,
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
    # Per-joint exceedance over limits (physical units), max over trajectory.
    # NOTE: acceleration has NO limit in the model (Pinocchio limits are q/v/tau only).
    q_lo = np.asarray(reg.q_lower_limit, dtype=float)
    q_hi = np.asarray(reg.q_upper_limit, dtype=float)
    v_lim = np.asarray(reg.v_limit, dtype=float)
    tau_lim = np.asarray(reg.tau_limit, dtype=float)
    q_over = np.zeros(reg.dof)
    v_over = np.zeros(reg.dof)
    tau_over = np.zeros(reg.dof)
    q_viol_steps = 0
    v_viol_steps = 0
    tau_viol_steps = 0
    for t_idx in range(len(ft.t_array)):
        res = reg.compute_regressor(q=q[:, t_idx], v=v[:, t_idx], a=a[:, t_idx])
        if res[18]:  # collided
            collisions += 1
        else:
            Y_aug_all.append(res[0])
        tau_aug = np.asarray(res[4])
        # Per-joint exceedance at this sample (rad, rad/s, Nm)
        dq = np.maximum(np.maximum(q_lo - q[:, t_idx], q[:, t_idx] - q_hi), 0.0)
        dv = np.maximum(np.abs(v[:, t_idx]) - v_lim, 0.0)
        dt = np.maximum(np.abs(tau_aug) - tau_lim, 0.0)
        q_over = np.maximum(q_over, dq)
        v_over = np.maximum(v_over, dv)
        tau_over = np.maximum(tau_over, dt)
        q_viol_steps += int(np.any(dq > 0.0))
        v_viol_steps += int(np.any(dv > 0.0))
        tau_viol_steps += int(np.any(dt > 0.0))

    def _excess_entry(max_over, viol_steps, unit):
        return {
            "max_over": [round(float(x), 4) for x in max_over],
            "violating_steps": int(viol_steps),
            "unit": unit,
        }

    zc = np.sum(np.diff(np.sign(v), axis=1) != 0, axis=1)
    traj_stats = {
        "collisions": f"{collisions}/{len(ft.t_array)}",
        "v_max": round(float(np.abs(v).max()), 2),
        "a_max": round(float(np.abs(a).max()), 2),
        "zero_crossings": [int(x) for x in zc],
        "excess": {
            "q": {
                **_excess_entry(q_over, q_viol_steps, "rad"),
                "limit_lo": [round(float(x), 4) for x in q_lo],
                "limit_hi": [round(float(x), 4) for x in q_hi],
            },
            "v": {
                **_excess_entry(v_over, v_viol_steps, "rad/s"),
                "limit": [round(float(x), 4) for x in v_lim],
            },
            "tau": {
                **_excess_entry(tau_over, tau_viol_steps, "Nm"),
                "limit": [round(float(x), 4) for x in tau_lim],
            },
        },
    }

    diagnostics = {}
    if len(Y_aug_all) > 10:
        Y_full = np.vstack(Y_aug_all)
        diagnostics = compute_regressor_diagnostics(
            Y_full, theta_nominal, param_names, cfg, traj_stats=traj_stats
        )

        # Console summary
        rd = diagnostics.get("reward_breakdown", {})
        rg = diagnostics.get("regression", {})
        qs = diagnostics.get("quality_summary", {})
        counts = {k: v["count"] for k, v in qs.items()}
        cond_pen = rd.get("cond_penalty", {})
        cond_pen_val = (
            cond_pen.get("penalty", "?") if isinstance(cond_pen, dict) else cond_pen
        )
        param_reward = rd.get("param_reward", {})
        param_val = (
            param_reward.get("value", "?")
            if isinstance(param_reward, dict)
            else param_reward
        )
        print(
            f"  d_opt={rd.get('d_opt', '?')}, cond_pen={cond_pen_val}, "
            f"param={param_val}, "
            f"rank={rg.get('eff_rank', '?')}/{rg.get('nonzero_cols', '?')} "
            f"(nullspace={rg.get('nullspace_dim', '?')}), "
            f"κ={rg.get('cond', '?')}, "
            f"null={counts.get('null', 0)} "
            f"rank_def={counts.get('rank_deficient', 0)} "
            f"small={counts.get('small', 0)} "
            f"good={counts.get('good', 0)} "
            f"ok={counts.get('ok', 0)} "
            f"bad={counts.get('bad', 0)}"
        )
        print(
            f"  Collisions: {traj_stats['collisions']}, "
            f"|v|max={traj_stats['v_max']}, |a|max={traj_stats['a_max']}, "
            f"zc={traj_stats['zero_crossings']}"
        )
        for key in ("q", "v", "tau"):
            e = rd.get("excess", {}).get(key, {})
            print(
                f"  excess {key}: max_over={e.get('max_over', '?')} "
                f"{e.get('unit', '')} [{e.get('violating_steps', 0)} steps]"
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
        POP,
        MAX_ITER,
        AMP_SCALE,
        pso.gbest_y,
        cfg=cfg,
        group=target_group,
        extra_meta=extra,
        diagnostics=diagnostics,
    )

    if status == "interrupted":
        raise

    print(f"\n{'=' * 60}\n  Done.\n{'=' * 60}")


if __name__ == "__main__":
    main()
