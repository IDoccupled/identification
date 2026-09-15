# \!/usr/bin/env python3
"""
Global SDP-based Parameter Identification for Serial Robot Limbs
(no per-joint sequential identification — all joints are identified together
in a single solve).

Regularization: quality-weighted RIDGE (L2 toward the URDF prior) instead of
the per-quality box constraints.  Good-quality parameters get a small penalty
for deviating from the prior; poor-quality ones get a large penalty.  Joint
armature/damping/friction are additionally **force-frozen** at their URDF
values (see ``RIDGE_FREEZE_LOCAL_INDICES``), so only link inertials are
identified unless that constant is changed.

============================================================================
Architecture:  solver (pure)  ←  data (prepared externally)
============================================================================

  ┌─────────────────────────────────────────────────────┐
  │  Data preparation (prepare_data_from_urdf / main)   │
  │  · FourierTrajectory  →  q(t), v(t), a(t)           │
  │  · TargetLimbRegressor  →  Y_aug, pi_prior          │
  │  · tau_measured = Y_stack @ pi_true (or real data)  │
  └──────────────────────┬──────────────────────────────┘
                         │  Y_stack, tau, pi_prior,
                         │  subtree_mask, joint_order
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  SDPSolver (pure, stateless)                        │
  │  · Single all-joints SOCP/SDP (all joints at once)  │
  │  · LMI physical-consistency constraint              │
  │  · Quality-weighted ridge toward the URDF prior     │
  └─────────────────────────────────────────────────────┘

This design means you can later swap in real sensor torque data
by only changing the data-preparation step — the solver stays the same.

After solving, ``main`` writes the identified parameters back out as a URDF
(``write_identified_urdf``): link inertials + joint armature/damping/friction
replace their prior values in a copy of the prior URDF, saved next to it with
a ``<name>_<YYMMDD_HHMMSS>.urdf`` timestamp suffix.

Dependencies: cvxpy + MOSEK (or SCS), numpy
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cvxpy as cp
import numpy as np

# ============================================================================
# Constants — per-joint parameter layout (13 scalars)
# ============================================================================
# Pinocchio Inertia::toDynamicParameters() ordering:
#   [m,  mc_x,  mc_y,  mc_z,   Ixx, Ixy, Iyy, Ixz, Iyz, Izz,  arm, damp, fric]
#    0     1      2      3      4    5    6    7    8    9     10    11    12
N_PER_JOINT = 13

_PARAM_LABELS = [
    "mass",
    "mc_x",
    "mc_y",
    "mc_z",
    "Ixx",
    "Ixy",
    "Iyy",
    "Ixz",
    "Iyz",
    "Izz",
    "armature",
    "damping",
    "friction",
]

# --- Weighted ridge (quality → L2 penalty toward the URDF prior) ---
RIDGE_QUALITY_WEIGHTS = {
    "good": 1.0,
    "ok": 3.0,
    "bad": 10.0,
    "rank_deficient": 20.0,
    "small": 1e-3,
}
# RIDGE_FREEZE_QUALITIES = frozenset({"null", "small"})
RIDGE_FREEZE_QUALITIES = frozenset({"null"})
# RIDGE_FREEZE_LOCAL_INDICES = frozenset({10, 11, 12})
RIDGE_FREEZE_LOCAL_INDICES = frozenset({})
DEFAULT_QUALITY = "ok"
RIDGE_LAMBDA = 3.0
RIDGE_SCALE_FLOOR = 1e-4

DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "resource"
    / "robot"
    / "urdf"
    / "serial_pm_v2_identify.urdf"
).resolve()

TRUE_URDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "resource"
    / "robot"
    / "urdf"
    / "serial_pm_v2_identify_true.urdf"
).resolve()

TRAJ_YAML_PATH = (
    Path(__file__).resolve().parent.parent
    / "trajectory_coefficients"
    / "recovered_exc_arm_1.yaml"
).resolve()

VAL_YAML_PATH = (
    Path(__file__).resolve().parent.parent
    / "trajectory_coefficients"
    / "verify_left_arm.yaml"
).resolve()

QUALITY_YAML_PATH = (
    Path(__file__).resolve().parent.parent
    / "trajectory_coefficients"
    / "excite_left_arm.yaml"
).resolve()

SIM_GRAVITY = np.array([-9.76935626, 0.39412975, -0.98615969])
SIM_WAIST_YAW_OFFSET = 0.02079

# bag CSV 的 topic 名：IMU 读数用于求重力，关节状态用于求腰关节角度。
IMU_CSV_TOPIC = "hardware_imu_info"

# [meas] 实测力矩相对状态的时延补偿（秒，--tau-delay；默认 0 = 不补偿）。
# 见 data_from_measurement：>0 表示“力矩滞后于状态”，回归器在 t − τ 上求值。
TAU_DELAY = 0.0

# 4×4 pseudo-inertia 严格正定余量（J ≽ eps·I₄，min_eig ≥ eps > 0）。
# 这只是**数值下限**（保证严格 PD + 给内点法留点余量），不是形状约束：连杆形状完全
# 交给下面的软铰链（LMI_SHAPE_FRAC / LMI_SHAPE_WEIGHT）。
# 历史（2026-09-15 前）这里还有一层 per-joint 相对硬余量
#   eps_j = max(LMI_EPS, rel · min_eig(J_prior_j))   （rel=0.1）
# 实测结论：它是**墙**不是**拉力** —— 目标函数在近似零空间方向上几乎是平的（原始
# 回归器 Y 的奇异值从 1799 掉到 ~1e-15），于是内点解**总是**精确停在墙上
# （min_eig/ε = 1.0000），只把扁平度从 1e-7 抬到 1.6e-5~3.4e-5，每个连杆的
# pseudo-inertia 依旧被压成薄片（三角不等式取等号，导出的 URDF 是近退化的惯量）。
# 而且它还会把真解排除在外（真值 J16/J17 的 slack 只有先验的 28%/17%，rel≥0.3 就
# 把真值排除在可行域之外）。⇒ 整层已被软铰链取代（见下），常量随之删除。
LMI_EPS = 1e-7

# --- 软铰链形状惩罚（2026-09-15；取代原来的 per-joint 相对硬余量）---
# 把硬余量的“墙”换成“拉力” —— 对每个关节引入标量 t_d 和仿射 LMI
#       J_d − t_d·blkdiag(I₃,0) ≽ 0    ⟺    t_d ≤ λ_min(Σ_C,d)
# （Σ_C = Σ_O − h·hᵀ/m 是绕质心二阶矩矩阵，λ_min(Σ_C) 就是三个三角不等式余量里最小
# 的那个），目标函数再加一项**无量纲、有上界**的铰链
#       W · Σ_d max(0, 1 − t_d / (LMI_SHAPE_FRAC · λ_min(Σ_C,d^prior)))
# 即“丢掉关节 d 先验 slack 的 100% 最多付 W”。解于是会主动往锥内部挪，而不是被动
# 停在边界上；先验 slack 偏保守（J16/J17 的真值只有先验的 28%/17%），所以用
# LMI_SHAPE_FRAC < 1 把目标压到真值量级，避免过冲。
# 实测（sim：exc_arm_1 训练 / verify_left_arm 验证；J15/J16/J17 的相对扁平度
# slack/mean(eig(Σ_C))，真值 = 0.456 / 0.030 / 0.030）：
#   frac  W     train%  val%   static  参数误差比 | J15/J16/J17 相对扁平度
#   —     0     93.09   93.18  0.0485    0.659   | 0.042 0.010 0.009  ← 无形状约束（压扁）
#   0.5   1     93.09   93.19  0.0481    0.664   | 0.208 0.049 0.042
#   0.3   1     93.10   93.19  0.0483    0.662   | 0.126 0.030 0.020  ← 默认
#   1.0  ≥0.3   92.92   93.22  0.0476    0.672   | 0.401 0.099 0.088  ← 过冲
# 结论：目标设在先验 slack 的 30~50% 几乎不要钱（train 完全不变、held-out 与静态
# 重力反而略好），却能把三个最扁的关节拉回接近真值的形状；frac=1.0 会把 J16/J17
# 撑得比真值还“胖”。W≥0.3 就饱和（铰链有上界），所以真正要调的是 frac。
# 另外：这套软铰链与原硬余量 0.1 在默认设置下**结果逐位一致**（铰链目标
# 4.7e-5~1.0e-4 高于原硬余量 1.6e-5~3.4e-5，硬约束本来就非激活）⇒ 删掉硬余量没有
# 任何代价。W=0 则退回“只有 LMI_EPS 下限”的旧行为（解贴在锥边界、连杆被压扁）。
LMI_SHAPE_FRAC = 0.3
LMI_SHAPE_WEIGHT = 1.0  # 0 = 关闭（退回只有 LMI_EPS 下限的压扁行为）

# --- 静态姿势重力测试（--static-test，仅 sim） ---
# 随机采样 N 个静态姿势（v = a = 0 ⇒ 只剩重力项），比较真值 URDF 的关节力矩
# 与辨识参数算出的关节力矩之差。
STATIC_TEST_POSES = 20  # 姿势数量（--static-poses）
STATIC_TEST_SEED = 0  # 采样种子（--static-seed）


# ============================================================================
# Parameter helpers
# ============================================================================
def split_joint_params(pi_full: np.ndarray) -> list[np.ndarray]:
    """(dof*13,) → list of dof × (13,)."""
    dof = len(pi_full) // N_PER_JOINT
    return [pi_full[i * N_PER_JOINT : (i + 1) * N_PER_JOINT] for i in range(dof)]


def join_joint_params(pi_list: list[np.ndarray]) -> np.ndarray:
    """list of dof × (13,) → (dof*13,)."""
    return np.concatenate(pi_list)


# ============================================================================
# Physical-consistency LMI  —  4×4 pseudo-inertia, second-moment form
#
#   J = [[Σ_O, h], [hᵀ, m]] ≽ 0 ,   Σ_O = ½·tr(I_O)·I₃ − I_O
#
# Schur complement: Σ_O − h·hᵀ/m = Σ_C = ½·tr(I_C)·I₃ − I_C ≽ 0, which is
# equivalent to {m > 0, I_C ≻ 0, triangle inequality on the principal moments},
# i.e. realizability by a non-negative mass density (Wensing et al. 2017).
# Putting I_O in the (1,1) block instead would only enforce I_C ≻ 0 and would
# silently admit inertias that violate the triangle inequality (e.g.
# I_C = diag(10,1,1)), which MuJoCo rejects at model load.
# Both forms are affine in π and therefore valid LMIs; the second-moment form is
# the complete one, and it is 4×4 instead of 6×6.
# ============================================================================
def build_pseudo_inertia_LMI(
    m: cp.Variable,
    mc: cp.Variable,
    I_vec: cp.Variable,
) -> cp.Expression:
    """4×4 pseudo-inertia [[Σ_O, h]; [hᵀ, m]] ≽ 0.  Pinocchio ordering."""
    I_mat = cp.bmat(
        [
            [I_vec[0], I_vec[1], I_vec[3]],  # Ixx Ixy Ixz
            [I_vec[1], I_vec[2], I_vec[4]],  # Ixy Iyy Iyz
            [I_vec[3], I_vec[4], I_vec[5]],  # Ixz Iyz Izz
        ]
    )
    tr_I = I_vec[0] + I_vec[2] + I_vec[5]  # tr(I_O) = Ixx + Iyy + Izz
    Sigma = 0.5 * tr_I * np.eye(3) - I_mat
    return cp.bmat(
        [
            [Sigma, cp.reshape(mc, (3, 1), order="F")],
            [cp.reshape(mc, (1, 3), order="F"), cp.reshape(m, (1, 1), order="F")],
        ]
    )


def check_lmi_feasibility(pi: np.ndarray) -> tuple[bool, float, np.ndarray]:
    """Numeric counterpart of ``build_pseudo_inertia_LMI`` (same ordering)."""
    m_val, mc_val = pi[0], pi[1:4]
    I_vals = pi[4:10]
    I_mat = np.array(
        [
            [I_vals[0], I_vals[1], I_vals[3]],
            [I_vals[1], I_vals[2], I_vals[4]],
            [I_vals[3], I_vals[4], I_vals[5]],
        ]
    )
    tr_I = I_vals[0] + I_vals[2] + I_vals[5]
    Sigma = 0.5 * tr_I * np.eye(3) - I_mat
    J = np.block(
        [
            [Sigma, mc_val.reshape(3, 1)],
            [mc_val.reshape(1, 3), np.array([[m_val]])],
        ]
    )
    eig_min = np.linalg.eigvalsh(J).min()
    return eig_min > 0, eig_min, J


# ============================================================================
# Shape helpers — triangle-inequality slack and the soft shape hinge
#
#   Σ_C = Σ_O − h·hᵀ/m      (second moment about the CoM; Schur complement of J)
#   eig(Σ_C) = {(b+c−a)/2, (a+c−b)/2, (a+b−c)/2}  ← the three triangle slacks
#
# J ≽ 0 ⟺ m>0 ∧ Σ_C ≽ 0 ⟺ {I_C ≻ 0 ∧ triangle inequality}.  So λ_min(Σ_C) ≥ 0
# is the complete physical-consistency condition, and λ_min(Σ_C)/mean(eig(Σ_C)) ∈
# [0,1] is a scale-free "how flat is this link" measure (0 = squashed flat, 1 =
# spherical).  λ_min(Σ_C) is *concave* in π, so "≥ t" is an LMI and hinging on it
# keeps the problem convex; the relative form λ_min/λ_max is NOT convex and is
# therefore not used.
# ============================================================================
LMI_SHAPE_B = np.diag([1.0, 1.0, 1.0, 0.0])  # blkdiag(I₃, 0): t·B only shifts Σ_O


def second_moment_about_com(pi_joint: np.ndarray) -> np.ndarray:
    """Σ_C = Σ_O − h·hᵀ/m (3×3) for one 13-param joint block (Pinocchio order)."""
    I_vals = np.asarray(pi_joint, dtype=float)[4:10]
    m = float(pi_joint[0])
    mc = np.asarray(pi_joint, dtype=float)[1:4]
    I_mat = np.array(
        [
            [I_vals[0], I_vals[1], I_vals[3]],
            [I_vals[1], I_vals[2], I_vals[4]],
            [I_vals[3], I_vals[4], I_vals[5]],
        ]
    )
    tr_I = I_vals[0] + I_vals[2] + I_vals[5]
    Sigma_O = 0.5 * tr_I * np.eye(3) - I_mat
    return Sigma_O - np.outer(mc, mc) / m


def triangle_slack(pi_joint: np.ndarray) -> float:
    """λ_min(Σ_C): the tightest of the three triangle-inequality slacks (0 = flat)."""
    return float(np.linalg.eigvalsh(second_moment_about_com(pi_joint)).min())


def relative_flatness(pi_joint: np.ndarray) -> float:
    """λ_min(Σ_C)/mean(eig(Σ_C)) ∈ [0, 1]: 0 = squashed flat, 1 = spherical."""
    e = np.linalg.eigvalsh(second_moment_about_com(pi_joint))
    mean_e = float(e.mean())
    return float(e.min() / mean_e) if mean_e > 0 else 0.0


def build_shape_reference(
    pi_prior: np.ndarray,
    dof: int,
    frac: float = LMI_SHAPE_FRAC,
) -> np.ndarray:
    """(dof,) hinge target: ``frac × λ_min(Σ_C)`` of each joint's URDF prior.

    Used as the reference of the soft shape hinge in ``_identify_all``; see
    ``LMI_SHAPE_FRAC`` for the measured effect of ``frac``.
    """
    prior_list = split_joint_params(pi_prior)
    return np.array(
        [float(frac) * triangle_slack(prior_list[d]) for d in range(int(dof))],
        dtype=float,
    )


def print_lmi_feasibility(
    pi_full: np.ndarray,
    joint_names: list[str] | None = None,
    joint_order: list[int] | None = None,
    label: str = "identified",
    inertia_eps: float = LMI_EPS,
    shape_ref: np.ndarray | None = None,
    verbose: bool = True,
) -> None:
    """Run ``check_lmi_feasibility`` per joint and print the results.

    The solver enforces ``J_d ≽ eps·I₄`` with the single scalar floor ``eps``
    (``LMI_EPS`` — strict PD, numerical only), so a joint is marked ``YES``
    only if it clears that floor: a solution parked on a 1e-7 boundary would
    pass a bare ``min_eig > 0`` test while being physically degenerate
    (saturated triangle inequality).  Non-``YES`` joints also print their
    identified ``(m, mc, I)`` and the eigenvalues of ``J``.

    The *shape* is not constrained by that floor but by the soft hinge, so
    ``shape_ref`` (optional, ``(dof,)``) adds three shape columns: the triangle
    slack ``λ_min(Σ_C)``, its ratio to the soft-hinge reference (``1.000`` =
    exactly on target) and the scale-free ``relative_flatness``.
    """
    pi_list = split_joint_params(pi_full)
    if joint_order is None:
        joint_order = list(range(len(pi_list)))
    eps = float(inertia_eps)
    shape_arr = (
        None if shape_ref is None else np.asarray(shape_ref, dtype=float).reshape(-1)
    )
    print(f"\nLMI physical-consistency check ({label} params):")
    shape_cols = (
        ""
        if shape_arr is None
        else (f" {'slack':>11s} {'slack/ref':>10s} {'rel.flat':>9s}")
    )
    print(
        f"{'Joint':<24s} {'feasible':>9s} {'min_eig':>12s} {'floor':>10s} "
        f"{'min_eig/eps':>12s}" + shape_cols
    )
    print("-" * (70 + (33 if shape_arr is not None else 0)))
    all_ok = True
    for d in joint_order:
        name = joint_names[d] if joint_names else f"joint_{d}"
        ok, eig_min, J = check_lmi_feasibility(pi_list[d])
        # Absolute slack absorbs the solver's own feasibility tolerance (~1e-10
        # here): an *active* constraint lands ~1e-10 below its bound, while a
        # genuinely flattened link is orders of magnitude below.
        clears = bool(ok and eig_min >= eps - (1e-9 + 1e-6 * eps))
        all_ok = all_ok and clears
        line = (
            f"{name:<24s} {('YES' if clears else 'NO'):>9s} {eig_min:>12.6g} "
            f"{eps:>10.3g} {eig_min / eps:>12.4g}"
        )
        if shape_arr is not None:
            slack = triangle_slack(pi_list[d])
            ref_d = float(shape_arr[d]) if d < shape_arr.size else 0.0
            ratio = f"{slack / ref_d:>10.3f}" if abs(ref_d) > 0 else f"{'-':>10s}"
            line += f" {slack:>11.4g} {ratio} {relative_flatness(pi_list[d]):>9.3f}"
        print(line)
        if verbose and not clears:
            p = pi_list[d]
            print(
                f"      m={p[0]:.6g}  mc=({p[1]:.6g},{p[2]:.6g},{p[3]:.6g})  "
                f"I=({p[4]:.6g},{p[5]:.6g},{p[6]:.6g},{p[7]:.6g},{p[8]:.6g},{p[9]:.6g})"
            )
            print(f"      J eigenvalues = {np.linalg.eigvalsh(J)}")
    print("-" * 70)
    print(
        f"All joints above the LMI floor: {all_ok}  "
        f"(uniform eps = {eps:.1g}, strict PD only)"
    )
    if shape_arr is not None:
        print(
            "  slack = λ_min(Σ_C) (triangle-inequality slack); "
            "slack/ref = identified ÷ soft-hinge target (1.000 = just on "
            "target, ~0.1 = still squashed); "
            "rel.flat = slack/mean(eig(Σ_C)) (0 = flat, 1 = sphere)."
        )
    if eps <= 1e-6:
        print(
            "  ⚠ eps <= 1e-6 is a numerical floor, not a shape constraint: a "
            "solution sitting on it has saturated the triangle inequality "
            "(flattened link). The shape is controlled by the soft hinge — "
            "keep --lmi-shape-weight > 0."
        )


# ============================================================================
# Result container
# ============================================================================
@dataclass
class IdentificationResult:
    pi_identified: np.ndarray  # (dof*13,)
    pi_prior: np.ndarray  # (dof*13,)  — from URDF
    pi_reference: np.ndarray | None  # (dof*13,)  — ground-truth (None if unknown)
    joint_solve_times: list[float]
    joint_objectives: list[float]
    joint_order: list[int]


# ============================================================================
# Torque RMSE helpers
# ============================================================================
def _joint_rmse(
    pi: np.ndarray,
    joint_order: list[int],
    Y_stack: np.ndarray,
    tau_measured: np.ndarray,
) -> np.ndarray:
    """Per-joint RMSE of ``Y_stack @ pi`` vs ``tau_measured``.

    Samples are laid out row-major by joint (sample k, joint d → row k·dof + d),
    matching ``_plot_torque_comparison_panels``.
    """
    dof = len(joint_order)
    n_total = len(tau_measured)
    tau_pred = Y_stack @ pi
    rmse = np.empty(dof)
    for idx, d in enumerate(joint_order):
        row = np.arange(d, n_total, dof)
        rmse[idx] = np.sqrt(np.mean((tau_pred[row] - tau_measured[row]) ** 2))
    return rmse


def _rmse_comparison(
    result: IdentificationResult,
    joint_order: list[int],
    Y_stack: np.ndarray,
    tau_measured: np.ndarray,
) -> dict:
    """Per-joint and overall torque RMSE (prior vs identified); no printing.

    Returns ``{"prior": (dof,), "ident": (dof,), "prior_all": float,
    "ident_all": float}``; the two arrays are indexed by *position in
    ``joint_order``* (same convention as ``_joint_rmse``).
    """
    return {
        "prior": _joint_rmse(result.pi_prior, joint_order, Y_stack, tau_measured),
        "ident": _joint_rmse(result.pi_identified, joint_order, Y_stack, tau_measured),
        "prior_all": float(
            np.sqrt(np.mean((Y_stack @ result.pi_prior - tau_measured) ** 2))
        ),
        "ident_all": float(
            np.sqrt(np.mean((Y_stack @ result.pi_identified - tau_measured) ** 2))
        ),
    }


def _improve_pct(a: float, b: float) -> float:
    """1 − b/a in percent (nan when a ≈ 0)."""
    return (1 - b / a) * 100 if a > 1e-12 else float("nan")


def print_rmse_comparison(
    result: IdentificationResult,
    joint_names: list[str] | None,
    Y_stack: np.ndarray,
    tau_measured: np.ndarray,
) -> dict:
    """Print torque RMSE before (prior) vs after (identified) identification.

    Returns the same numbers as ``_rmse_comparison`` (the multi-yaml
    cross-validation summary aggregates them).
    """
    stats = _rmse_comparison(result, result.joint_order, Y_stack, tau_measured)
    rmse_prior = stats["prior"]
    rmse_ident = stats["ident"]

    def _improve(a: float, b: float) -> float:
        return (1 - b / a) * 100 if a > 1e-12 else float("nan")

    print("\nTorque RMSE comparison (prior vs identified):")
    print(
        f"{'Joint':<20s} {'Prior [Nm]':>12s} {'Identified [Nm]':>15s} {'Improve %':>10s}"
    )
    print("-" * 53)
    for idx, d in enumerate(result.joint_order):
        name = joint_names[d] if joint_names else f"joint_{d}"
        print(
            f"{name:<20s} {rmse_prior[idx]:>12.6g} {rmse_ident[idx]:>15.6g} "
            f"{_improve(rmse_prior[idx], rmse_ident[idx]):>9.2f}%"
        )
    print("-" * 53)
    rp_all = stats["prior_all"]
    ri_all = stats["ident_all"]
    print(
        f"{'ALL':<20s} {rp_all:>12.6g} {ri_all:>15.6g} "
        f"{_improve(rp_all, ri_all):>9.2f}%"
    )
    return stats


def print_cv_summary(
    rows: list[dict],
    joint_names: list[str] | None = None,
    joint_order: list[int] | None = None,
) -> None:
    """One-line-per-yaml summary of a multi-yaml cross-validation run.

    ``rows`` entries: ``{"yaml": str, "note": str, "stats": dict|None,
    "error": str (only when the yaml failed)}``.  Per-joint columns are the
    improvement %% of that joint on that held-out trajectory, followed by the
    mean over all yamls — i.e. the "does the identified URDF generalise"
    verdict, aggregated instead of one table per trajectory.
    """
    if not rows:
        return
    order = sorted(joint_order) if joint_order is not None else []
    shorts = [
        (joint_names[d] if joint_names else f"joint_{d}").split("_")[0] for d in order
    ]
    w_yaml = max(20, max(len(f"{r['yaml']}") for r in rows) + 2)
    w_note = max(10, max(len(f"{r.get('note', '-')}") for r in rows) + 2)
    wj = max(9, max((len(s) for s in shorts), default=0) + 4)
    hdr = (
        f"{'yaml':<{w_yaml}}{'note':<{w_note}}{'prior ALL':>12}{'ident ALL':>12}"
        f"{'ALL imp%':>9}" + "".join(f"{s:>{wj}}" for s in shorts)
    )
    print("\n" + "=" * len(hdr))
    print(f"CROSS-VALIDATION SUMMARY ({len(rows)} held-out yaml)".center(len(hdr)))
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    per_joint_imp: dict[int, list[float]] = {d: [] for d in order}
    all_imp: list[float] = []
    for r in rows:
        st = r.get("stats")
        row_note = f"{r.get('note', '-')}"
        if not st:
            row_err = f"{r.get('error', '')}"[:44]
            print(
                f"{r['yaml']:<{w_yaml}}{row_note:<{w_note}}{'FAILED':>33}   {row_err}"
            )
            continue
        by_d = {d: i for i, d in enumerate(joint_order)}
        imps = [_improve_pct(st["prior"][by_d[d]], st["ident"][by_d[d]]) for d in order]
        imp_all = _improve_pct(st["prior_all"], st["ident_all"])
        all_imp.append(imp_all)
        for d, v in zip(order, imps):
            per_joint_imp[d].append(v)
        print(
            f"{r['yaml']:<{w_yaml}}{row_note:<{w_note}}"
            f"{st['prior_all']:>12.5g}{st['ident_all']:>12.5g}{imp_all:>8.2f}%"
            + "".join(f"{v:>{wj}.1f}" for v in imps)
        )
    if all_imp:
        print("-" * len(hdr))

        def _mean(xs: list[float]) -> float:
            return float(np.mean(xs)) if xs else float("nan")

        print(
            f"{'MEAN over yamls':<{w_yaml}}{'':<{w_note}}"
            f"{'':>12}{'':>12}{_mean(all_imp):>8.2f}%"
            + "".join(f"{_mean(per_joint_imp[d]):>{wj}.1f}" for d in order)
        )


# ============================================================================
# Pure SDP solver
# ============================================================================
class SDPSolver:
    """
    Pure global SDP-based parameter identification (weighted ridge variant).

    All joints are identified **simultaneously** in a single SDP/SOCP — there
    is no distal→proximal sequencing and no fixed-joint torque compensation:
    one SOC per joint over its own torque rows, plus one 4×4 pseudo-inertia
    LMI per joint.  Instead of per-quality box constraints, a **quality
    weighted ridge (L2-toward-prior)** term is added to the objective —
    good-quality parameters may deviate freely from the URDF prior, poor ones
    are pulled back strongly.

    Takes pre-computed data — no URDF, no Pinocchio, no regressor computation.
    """

    def __init__(
        self,
        solver_name: str = "MOSEK",
        verbose: bool = True,
    ):
        self.solver_name = solver_name
        self.verbose = verbose

    # ------------------------------------------------------------------
    def solve(
        self,
        Y_stack: np.ndarray,  # (N·dof, 13·dof)  stacked augmented regressor
        tau_measured: np.ndarray,  # (N·dof,)          measured joint torques
        pi_prior: np.ndarray,  # (dof*13,)         prior params (ridge attractor)
        joint_order: list[int],  # distal → proximal indices
        ridge_weights: np.ndarray,  # (dof*13,) per-param ridge coefficients
        freeze_mask: np.ndarray | None = None,  # (dof*13,) bool: hard-freeze
        inertia_eps: float = LMI_EPS,
        shape_ref: np.ndarray | None = None,
        shape_weight: float = LMI_SHAPE_WEIGHT,
        joint_names: list[str] | None = None,
    ) -> IdentificationResult:
        """
        Run **global (all-at-once) weighted-ridge** identification.

        All ``dof`` joints are identified simultaneously in a single SDP/SOCP:
        one SOC per joint over its own torque rows, one 4×4 pseudo-inertia LMI
        per joint, and a **quality-weighted ridge** penalty pulling every
        parameter toward its URDF prior (good quality → small penalty, poor
        quality → large penalty).  There is no per-quality box constraint and
        no distal→proximal torque compensation — the cross-coupling columns
        between a joint and its (distal) subtree members are handled directly
        because every joint's parameters are free variables of the same
        problem.

        Parameters
        ----------
        ridge_weights : (dof*13,) ndarray
            Per-parameter quadratic coefficients c_i (≥ 0) of the L2
            toward-prior penalty ``c_i · (pi_i − prior_i)²``; built from the
            quality labels by ``build_ridge_weights``.
        freeze_mask : (dof*13,) bool ndarray or None
            Where ``True``, hard-freeze the parameter at its URDF prior via an
            equality constraint ``pi == prior``.  Used for the ``null``/``small``
            quality labels (too small for a large ridge weight to pin exactly,
            so they are frozen literally) **and** for every joint's
            armature/damping/friction (``RIDGE_FREEZE_LOCAL_INDICES``).  ``None``
            = no freeze.  Frozen parameters must also carry ``c = 0`` — see
            ``build_freeze_mask`` / ``build_ridge_weights``.
        inertia_eps : float
            Strict positive-definiteness floor of the LMI (``J_d ≽ eps·I₄``),
            ``LMI_EPS`` by default — numerical only (strict PD + solver
            slack).  The link *shape* is **not** governed by this floor but by
            the soft hinge below (``shape_ref`` / ``shape_weight``).
        shape_ref : (dof,) ndarray or None
            Soft shape-hinge reference per joint: the triangle-inequality slack
            ``λ_min(Σ_C)`` the solution is *paid* to keep — build it with
            ``build_shape_reference``.  Used only when ``shape_weight > 0``.
        shape_weight : float
            Price ``W`` of the dimensionless hinge ``W·Σ_d max(0, 1 −
            t_d/ref_d)``; ``0`` = off.  See ``LMI_SHAPE_FRAC``.
        """
        dof = len(joint_order)
        pi_identified = np.zeros(dof * N_PER_JOINT)
        pi_prior_list = split_joint_params(pi_prior)

        # --- Pre-compute per-joint row masks once: sample k, joint d is at
        #     row k·dof + d (joint-major regressor layout). ---
        row_masks = []
        for d in range(dof):
            m = np.zeros(Y_stack.shape[0], dtype=bool)
            m[d::dof] = True
            row_masks.append(m)

        if self.verbose:
            print(
                f"Global SDP (weighted ridge) — all {dof} joints identified "
                f"together (distal→proximal order {joint_order}):"
            )

        # --- Single all-at-once solve ---
        pi_opt, dt, lam_vals = self._identify_all(
            Y_stack,
            tau_measured,
            pi_prior_list,
            joint_order,
            row_masks,
            ridge_weights=ridge_weights,
            freeze_mask=freeze_mask,
            inertia_eps=inertia_eps,
            shape_ref=shape_ref,
            shape_weight=shape_weight,
            joint_names=joint_names,
        )
        pi_identified = np.asarray(pi_opt).flatten()

        # One global solve → one solve time; per-joint objectives = λ_d.
        solve_times: list[float] = [dt] if dt is not None else []
        objectives: list[float] = list(lam_vals) if lam_vals is not None else []

        return IdentificationResult(
            pi_identified=pi_identified,
            pi_prior=pi_prior,
            pi_reference=None,
            joint_solve_times=solve_times,
            joint_objectives=objectives,
            joint_order=joint_order,
        )

    # ------------------------------------------------------------------
    def _identify_all(
        self,
        Y_stack: np.ndarray,  # (N·dof, 13·dof)  stacked augmented regressor
        tau_measured: np.ndarray,  # (N·dof,)          measured joint torques
        pi_prior_list: list[np.ndarray],  # dof × (13,) prior parameter blocks
        joint_order: list[int],
        row_masks: list[np.ndarray],
        ridge_weights: np.ndarray,  # (dof*13,) quadratic ridge coefficients
        freeze_mask: np.ndarray | None = None,  # (dof*13,) bool: hard-freeze
        inertia_eps: float = LMI_EPS,
        shape_ref: np.ndarray | None = None,
        shape_weight: float = 0.0,
        joint_names: list[str] | None = None,
    ) -> tuple[np.ndarray, float, list[float]]:
        """Single all-joints weighted-ridge SDP/SOCP.

        Returns ``(pi_opt, solve_time, λ_list)``.

        Variables: one 13-param block per joint ``pi[13d:13d+13]``, one
        non-negative scalar ``λ_d`` per joint, the ridge epigraph scalar ``u``,
        and — when the shape hinge is on — one scalar ``t_d`` per joint.

        Objective ``min Σ_d λ_d + u`` (``+ W·Σ_d max(0, 1 − t_d/ref_d)`` with
        the shape hinge enabled).

        Constraints per joint ``d``:
          · SOC  ‖Y_d_rows @ pi − τ_d‖₂ ≤ λ_d   (rows of joint d only; the
            columns are full because the cross-coupling of a proximal joint
            with its distal subtree members is resolved inside this one
            problem)
          · LMI  4×4 pseudo-inertia J_d ≽ eps·I₄  (strict PD; ``eps`` is a
            numerical floor, ``LMI_EPS`` by default)
          · LMI  J_d − t_d·blkdiag(I₃,0) ≽ 0  ⟺  t_d ≤ λ_min(Σ_C,d)
            (shape hinge only)
          · hard physical non-negativity of armature/damping/friction
            (mass ≥ eps is implied by the LMI)
        plus the weighted-ridge epigraph SOC ``‖√c ⊙ (pi − prior)‖₂ ≤ u`` with
        ``c = ridge_weights`` (the L2-toward-prior penalty replaces the old
        per-quality box constraints).

        If ``freeze_mask`` is given, parameters flagged ``True`` (``null``/
        ``small`` quality, plus every joint's armature/damping/friction) are
        additionally hard-frozen by equality ``pi == prior`` and excluded from
        the ridge norm.

        ``shape_ref`` / ``shape_weight`` are the link's *shape* control: with
        ``W > 0`` the solution is *paid* to keep ``t_d`` (the triangle slack)
        near ``shape_ref[d]``.  The bare LMI floor above cannot do that job — a
        flat objective parks the solution exactly on it — and a *hard* relative
        margin (removed 2026-09-15) only moved the wall.  See ``LMI_SHAPE_FRAC``.
        """
        dof = len(joint_order)
        pi_prior_full = np.concatenate(pi_prior_list)
        ridge_weights = np.asarray(ridge_weights, dtype=float).reshape(-1)
        assert ridge_weights.size == dof * N_PER_JOINT, (
            f"ridge_weights size {ridge_weights.size} != dof*13 = {dof * N_PER_JOINT}"
        )
        # --- Strict-PD floor of the pseudo-inertia LMI.  Numerical only: the
        #     link shape is controlled by the soft hinge below. ---
        eps = float(inertia_eps)
        # --- Soft shape hinge: see LMI_SHAPE_FRAC.  Dimensionless and bounded,
        #     so W reads as "objective units I am willing to pay for one joint's
        #     full reference slack" and ≥ ~0.3 saturates. ---
        shape_weight = float(shape_weight or 0.0)
        shape_on = shape_weight > 0.0
        shape_ref_vec: np.ndarray | None = None
        if shape_on:
            if shape_ref is None:
                raise ValueError(
                    "shape_weight > 0 requires shape_ref (build_shape_reference)"
                )
            shape_ref_vec = np.asarray(shape_ref, dtype=float).reshape(-1)
            assert shape_ref_vec.size == dof, (
                f"shape_ref size {shape_ref_vec.size} != dof = {dof}"
            )
            assert np.all(shape_ref_vec > 0.0), (
                "shape_ref must be strictly positive (it is a dividend: "
                f"got {shape_ref_vec.min():.3g})"
            )
        c_sqrt = np.sqrt(np.maximum(ridge_weights, 0.0))  # √c per parameter

        # --- Hard freeze (null/small): equality pi == prior.  These parameters
        #     already carry c = 0 (build_ridge_weights excludes them from the
        #     ridge), so also dropping them from the group norm keeps
        #     ‖√c ⊙ δ‖₂ well conditioned. ---
        freeze_arr = np.zeros(dof * N_PER_JOINT, dtype=bool)
        if freeze_mask is not None:
            freeze_arr = np.asarray(freeze_mask, dtype=bool).reshape(-1)
            assert freeze_arr.size == dof * N_PER_JOINT, (
                f"freeze_mask size {freeze_arr.size} != dof*13 = {dof * N_PER_JOINT}"
            )
        n_frozen = int(freeze_arr.sum())
        c_sqrt = c_sqrt.copy()
        c_sqrt[freeze_arr] = 0.0

        # --- Guardrail: c = 0 is reserved for the hard-frozen (null/small)
        #     parameters.  A zero-weight parameter that is NOT frozen would be
        #     a completely unregularised free variable (excluded from the ridge
        #     norm *and* free to move) — fail loudly instead.  The all-zero case
        #     is exempt: it is the documented `--ridge-lambda 0` pure-fit mode. ---
        unpenalised = (ridge_weights <= 0.0) & (~freeze_arr)
        if unpenalised.any() and (ridge_weights > 0.0).any():
            bad = np.nonzero(unpenalised)[0]
            raise ValueError(
                f"{int(unpenalised.sum())} parameter(s) have zero ridge weight "
                f"but are not hard-frozen (global idx: {bad[:10].tolist()}"
                f"{' ...' if bad.size > 10 else ''}).  Zero ridge weight is "
                "reserved for RIDGE_FREEZE_QUALITIES (null/small), which MUST "
                "be frozen — pass a matching freeze_mask (build_freeze_mask) "
                "or drop the null/small labels from the quality map."
            )
        if self.verbose and n_frozen:
            print(
                f"  hard-freeze (null/small): {n_frozen} params at prior "
                f"(pi == prior, excluded from the ridge)"
            )

        # --- Pre-solve diagnostics (verbose): per-joint prior residual, LMI
        #     feasibility and per-joint ridge-weight summary ---
        if self.verbose:
            for d in joint_order:
                name = joint_names[d] if joint_names else f"joint_{d}"
                prior = pi_prior_list[d]
                lmi_ok, eig_min, _ = check_lmi_feasibility(prior)
                lmi_txt = "OK" if lmi_ok else f"FAIL (min eig={eig_min:.4e})"
                Y_d = Y_stack[row_masks[d]]
                tau_d = tau_measured[row_masks[d]]
                resid_prior = np.linalg.norm(Y_d @ pi_prior_full - tau_d)
                # Back out the dimensionless per-quality weight w_q ≈ c·scale²
                # (frozen params have c = 0 → show 0)
                c2 = np.square(c_sqrt)
                w_rel = np.asarray(
                    [
                        c2[d * N_PER_JOINT + i]
                        * max(abs(prior[i]), RIDGE_SCALE_FLOOR) ** 2
                        for i in range(N_PER_JOINT)
                    ]
                )
                print(
                    f"  Joint {d} ({name}): rows=({Y_d.shape[0]},{Y_d.shape[1]}), "
                    f"‖Y·prior−τ‖={resid_prior:.4e}, LMI={lmi_txt}, "
                    f"ridge_w∈[{w_rel.min():.3g},{w_rel.max():.3g}]"
                )

        # --- Variables ---
        pi = cp.Variable(dof * N_PER_JOINT)  # all joints' parameters at once
        lam = cp.Variable(dof, nonneg=True)  # per-joint SOC objective terms
        t_shape = cp.Variable(dof) if shape_on else None  # per-joint slack bound

        # --- Constraints ---
        cstr: list = []
        for idx, d in enumerate(joint_order):
            cs = d * N_PER_JOINT
            ce = cs + N_PER_JOINT
            pi_d = pi[cs:ce]  # (13,) view of this joint's parameters
            Y_d = Y_stack[row_masks[d]]  # (N, 13*dof) rows of joint d
            tau_d = tau_measured[row_masks[d]]  # (N,)

            # 1) torque-fit SOC over joint d's own rows (all columns, since
            #    every joint's parameters are optimised together)
            cstr.append(cp.SOC(lam[idx], Y_d @ pi - tau_d))

            # 2) physical consistency: 4×4 pseudo-inertia LMI strictly
            #    positive definite: J ≽ eps·I₄  (min_eig ≥ eps > 0)
            J = build_pseudo_inertia_LMI(pi_d[0], pi_d[1:4], pi_d[4:10])
            cstr.append(J - eps * np.eye(4) >> 0)

            # 2b) shape hinge: t_d ≤ λ_min(Σ_C,d), written as the affine
            #     block-shift LMI J_d − t_d·blkdiag(I₃,0) ≽ 0 (Σ_C is the
            #     Schur complement of J w.r.t. m, so subtracting t from the
            #     Σ_O block is exactly Σ_C ≽ t·I₃).  Soft, so the solution is
            #     pulled inside the cone instead of resting on the eps wall.
            if shape_on:
                cstr.append(J - t_shape[d] * LMI_SHAPE_B >> 0)

            # 3) hard physical non-negativity of armature/damping/friction
            for i in (10, 11, 12):
                cstr.append(pi_d[i] >= 0.0)

        # --- Hard-freeze equality constraints (null/small): pi == prior ---
        if n_frozen:
            for g in np.nonzero(freeze_arr)[0]:
                cstr.append(pi[g] == pi_prior_full[g])

        # --- Weighted ridge epigraph: obj += u,  u ≥ ‖√c ⊙ (pi − prior)‖₂ ---
        u = cp.Variable(nonneg=True)
        cstr.append(cp.SOC(u, cp.multiply(c_sqrt, pi - pi_prior_full)))

        # --- Solve ---
        objective = cp.sum(lam) + u
        if shape_on:
            # Bounded, dimensionless deficit: 1.0 = the joint has lost all of
            # its reference slack.  cp.pos keeps it convex (λ_min is concave).
            deficit = cp.pos(1.0 - cp.multiply(1.0 / shape_ref_vec, t_shape))
            objective = objective + shape_weight * cp.sum(deficit)
            if self.verbose:
                print(
                    f"  shape hinge ON: W={shape_weight:g}, ref = "
                    f"[{shape_ref_vec.min():.4g}, {shape_ref_vec.max():.4g}] "
                    f"(prior triangle slack × {LMI_SHAPE_FRAC:g})"
                )
        problem = cp.Problem(cp.Minimize(objective), cstr)
        try:
            problem.solve(solver=self.solver_name, verbose=False)
        except cp.error.SolverError:
            if self.verbose:
                print(f"    [{self.solver_name}] failed → SCS ...")
            problem.solve(solver="SCS", verbose=False, max_iters=5000)

        t = problem.solver_stats.solve_time if problem.solver_stats else 0.0

        if pi.value is None:
            msg = (
                f"SDP infeasible for the all-joints problem (status={problem.status})."
            )
            if self.verbose:
                msg += (
                    "\n    See the per-joint prior LMI feasibility printed above; "
                    "\n    Try: a smaller --ridge-lambda."
                )
            raise RuntimeError(msg)

        pi_opt = np.array(pi.value).flatten()
        lam_vals = (
            list(np.asarray(lam.value).flatten()) if lam.value is not None else []
        )

        if self.verbose:
            print(
                f"  → all-joints SDP solved [{problem.status}] "
                f"in {t:.3f}s  (Σλ = {float(np.sum(lam_vals)):.6g}, "
                f"ridge u* = {float(u.value):.6g})"
            )
            for idx, d in enumerate(joint_order):
                name = joint_names[d] if joint_names else f"joint_{d}"
                print(f"      joint {d} ({name}): λ* = {lam_vals[idx]:.6g}")

        return pi_opt, t, lam_vals

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def print_results(
        result: IdentificationResult,
        joint_names: list[str] | None = None,
        pi_reference: np.ndarray | None = None,
        quality_map: dict[int, str] | None = None,
        Y_stack: np.ndarray | None = None,
        tau_measured: np.ndarray | None = None,
        is_sim: bool = False,
    ):
        """Pretty-print identification results joint by joint.

        If ``Y_stack`` and ``tau_measured`` are provided, prints a per-joint
        torque RMSE comparison (prior vs identified) instead of Σ‖τ_residual‖₂.

        ``is_sim=True`` (真值已知): the ``Err%`` column is replaced by a
        before→after comparison ``prior→identified`` 相对真值的误差，例如
        ``+5.00% -> +3.00%``；``small`` / ``null`` 档参数直接显示 ``N/A``。
        """
        if pi_reference is None:
            pi_reference = result.pi_reference
        has_ref = pi_reference is not None

        print("\n" + "=" * 100)
        print("IDENTIFICATION RESULTS".center(100))
        print("=" * 100)

        pi_id = split_joint_params(result.pi_identified)
        pi_prior = split_joint_params(result.pi_prior)
        pi_ref = split_joint_params(pi_reference) if has_ref else None

        for d in result.joint_order:
            name = joint_names[d] if joint_names else f"joint_{d}"
            print(f"\n--- Joint {d}: {name} ---")
            if has_ref and is_sim:
                hdr = (
                    f"{'Param':<10s} {'Prior':>12s} {'Identified':>12s} "
                    f"{'True':>12s} {'Err% (prior→id)':>20s} {'Δ%':>8s} {'Quality':>10s}"
                )
            elif has_ref:
                hdr = (
                    f"{'Param':<10s} {'Prior':>12s} {'Identified':>12s} "
                    f"{'True':>12s} {'Err%':>8s} {'Δ%':>8s} {'Quality':>10s}"
                )
            else:
                hdr = (
                    f"{'Param':<10s} {'Prior':>12s} {'Identified':>12s} "
                    f"{'Δ%':>8s} {'Quality':>10s}"
                )
            print(hdr)
            print("-" * len(hdr))

            for i in range(N_PER_JOINT):
                pr = pi_prior[d][i]
                ident = pi_id[d][i]
                g = d * N_PER_JOINT + i
                q = quality_map.get(g, "?") if quality_map else "?"
                dp = (ident - pr) / max(abs(pr), 1e-12) * 100  # Δ% from prior
                if has_ref and is_sim:
                    ref = pi_ref[d][i]
                    if q in RIDGE_FREEZE_QUALITIES:
                        err_str = "N/A"
                    else:
                        denom = abs(ref) if abs(ref) > 1e-12 else 1.0
                        err_prior = (pr - ref) / denom * 100
                        err_ident = (ident - ref) / denom * 100
                        err_str = f"{err_prior:+.2f}% -> {err_ident:+.2f}%"
                    print(
                        f"{_PARAM_LABELS[i]:<10s} {pr:>12.6g} {ident:>12.6g} "
                        f"{ref:>12.6g} {err_str:>20s} {dp:>7.2f}% {q:>10s}"
                    )
                elif has_ref:
                    ref = pi_ref[d][i]
                    denom = abs(ref) if abs(ref) > 1e-12 else 1.0
                    err = (ident - ref) / denom * 100
                    print(
                        f"{_PARAM_LABELS[i]:<10s} {pr:>12.6g} {ident:>12.6g} "
                        f"{ref:>12.6g} {err:>7.2f}% {dp:>7.2f}% {q:>10s}"
                    )
                else:
                    print(
                        f"{_PARAM_LABELS[i]:<10s} {pr:>12.6g} {ident:>12.6g} "
                        f"{dp:>7.2f}% {q:>10s}"
                    )

        print("\n" + "-" * 90)
        print(f"Total solve time: {sum(result.joint_solve_times):.3f}s")

        if Y_stack is not None and tau_measured is not None:
            print_rmse_comparison(result, joint_names, Y_stack, tau_measured)
        else:
            # Backward-compatible fallback if torque data isn't available
            print(f"Σ ‖τ_residual‖₂: {sum(result.joint_objectives):.6g}")


# ============================================================================
# Data preparation
# ============================================================================
def _reorder_y_aug(Y: np.ndarray, dof: int) -> np.ndarray:
    """Convert a type-major ``Y_aug`` → joint-major (13 cols/joint) layout.

    ``Y_aug`` columns (from ``TargetLimbRegressor.compute_regressor``) are
    type-major:
        [inertial_j0(10) ... inertial_j{D-1}(10),
         arm_j0(1) ... arm_j{D-1}(1),
         fric_j0(2) ... fric_j{D-1}(2)]
    while ``pi`` is joint-major: [j0(10+1+2), j1(10+1+2), ...].
    Reordering makes ``Y @ pi`` correct.
    """
    Yr = np.zeros((dof, dof * N_PER_JOINT))
    for j in range(dof):
        # inertial (10 cols)
        Yr[:, j * N_PER_JOINT : j * N_PER_JOINT + 10] = Y[:, j * 10 : (j + 1) * 10]
        # armature (1 col)
        Yr[:, j * N_PER_JOINT + 10] = Y[:, 10 * dof + j]
        # friction (2 cols)
        Yr[:, j * N_PER_JOINT + 11 : j * N_PER_JOINT + 13] = Y[
            :, 10 * dof + dof + 2 * j : 10 * dof + dof + 2 * j + 2
        ]
    return Yr


def _stack_regressor(reg, q_arm, v_arm, a_arm) -> np.ndarray:
    """Stack per-sample joint-major regressors → Y_stack (N·dof, 13·dof).

    Rows are sample-major, joint-minor: sample k, joint d → row k·dof + d
    (matches the solver layout).
    """
    dof = reg.dof
    Y_list = []
    for k in range(len(q_arm)):
        Y_aug, *_ = reg.compute_regressor(
            q_arm[k], v_arm[k], a_arm[k], print_info=False
        )
        Y_list.append(_reorder_y_aug(Y_aug, dof))
    return np.vstack(Y_list)


def prepare_data_from_urdf(
    urdf_path: str | Path,
    yaml_filename: str,
    limb_group: str | None = None,
    sample_rate: float = 200.0,
    time_coeffs: float = 1.0,
    twin: str | None = None,
    urdf_true_path: str | Path | None = None,
    verbose: bool = True,
    gravity: np.ndarray | None = None,
    waist_yaw_offset: float | None = None,
) -> dict:
    """
    Prepare all data needed by ``SDPSolver.solve()``.

    Parameters
    ----------
    urdf_path : str or Path
        Initial/prior URDF: used for regressor, pi_prior, subtree mask.
    urdf_true_path : str, Path, or None
        "True" robot URDF.  If None, same as urdf_path (debug mode).
        When different: only used to generate tau_measured; its params
        are treated as unknown by the solver.
    time_coeffs : float
        Fourier-trajectory playback speed (>1 speeds up, <1 slows down;
        physical period = TRAJ_PERIOD / time_coeffs).  Velocity/acceleration
        are scaled by time_coeffs / time_coeffs**2 accordingly.
    twin : str or None
        Optional ``'START:END'`` physical-time window (seconds) to use for
        identification, e.g. ``'10:16.667'`` (either end may be omitted;
        ``None`` = whole trajectory).  The trajectory is periodic, so a single
        good-quality period carries all the information; cropping to it also
        reduces the problem size.
    """
    from identification.fourier_trajectory import FourierTrajectory
    from identification.target_limb_regressor import (
        TargetLimbRegressor,
        VALID_LIMB_GROUPS,
    )

    if urdf_true_path is None:
        urdf_true_path = urdf_path

    # Limb group defaults to the YAML _meta.group (fallback 'left_arm' for
    # legacy recovered_*.yaml that predate the group field).
    if limb_group is None:
        limb_group = FourierTrajectory.load_group(yaml_filename, default="left_arm")

    # --- Trajectory ---
    dof_limb = len(VALID_LIMB_GROUPS[limb_group])
    ft = FourierTrajectory(
        dim=dof_limb, sample_rate=sample_rate, time_coeffs=time_coeffs
    )
    yaml_name = Path(yaml_filename).name
    q_traj, v_traj, a_traj = ft.generate_trajectory_from_yaml(yaml_name)
    N = q_traj.shape[1]
    if verbose:
        print(
            f"Trajectory: {N} steps from {yaml_name} "
            f"(time_coeffs={time_coeffs}, period={ft.duration:.3f}s)"
        )

    # --- Optional identification time window (sim): keep only the samples in
    #     the chosen physical-time segment 'START:END' (same as meas --twin-id).
    #     The trajectory is periodic, so one good period is enough and also
    #     reduces the problem size. ---
    if twin is not None:
        tw0, tw1 = _parse_window(twin, float(ft.t_array[-1]))
        wmask = (ft.t_array >= tw0) & (ft.t_array <= tw1)
        if not np.any(wmask):
            raise ValueError(
                f"Identification time window [{tw0:.3g}, {tw1:.3g}] "
                f"contains no samples (trajectory t∈[0,{ft.duration:.3f}]s)"
            )
        q_traj = q_traj[:, wmask]
        v_traj = v_traj[:, wmask]
        a_traj = a_traj[:, wmask]
        N = q_traj.shape[1]
        if verbose:
            print(
                f"  [sim] ID time window {twin}: N={N} "
                f"(of {ft.t_array.size}, t∈[{ft.t_array[wmask][0]:.4g},"
                f"{ft.t_array[wmask][-1]:.4g}]s)"
            )

    # --- Initial (prior) model: regressor, pi_prior, subtree, joint order ---
    reg = TargetLimbRegressor(
        urdf_path=Path(urdf_path),
        group_to_identify=limb_group,
        print_info=False,
        gravity=gravity,
        waist_yaw_offset=waist_yaw_offset,
    )
    dof = reg.dof

    pi_prior_list = []
    for idx, joint_id in enumerate(reg.group_to_identify):
        pin_jid = joint_id + 1
        pi_inertial = reg.model.inertias[pin_jid].toDynamicParameters()
        info = reg.target_joint_infos[idx]
        pi_prior_list.extend(pi_inertial)
        pi_prior_list.append(info["armature"])
        pi_prior_list.append(info["damping"])
        pi_prior_list.append(info["friction"])
        if verbose:
            print(
                f"  [prior] [{idx}] {info['name']}: "
                f"m={pi_inertial[0]:.6g} "
                f"Ixx={pi_inertial[4]:.6g} Iyy={pi_inertial[6]:.6g} Izz={pi_inertial[9]:.6g}"
            )
    pi_prior = np.array(pi_prior_list)

    reg.compute_regressor(print_info=False)
    subtree_mask = reg.subtree_mask.copy()
    subtree_size = subtree_mask.sum(axis=1)
    joint_order = sorted(range(dof), key=lambda d: subtree_size[d])
    joint_names = [reg.target_joint_infos[d]["name"] for d in range(dof)]

    # --- True model: only extract pi_true (unknown to solver) ---
    if Path(urdf_true_path).resolve() != Path(urdf_path).resolve():
        if verbose:
            print(f"  [true]  loading separate URDF: {urdf_true_path}")
        reg_true = TargetLimbRegressor(
            urdf_path=Path(urdf_true_path),
            group_to_identify=limb_group,
            print_info=False,
            gravity=gravity,
            waist_yaw_offset=waist_yaw_offset,
        )
        pi_true_list = []
        for idx, joint_id in enumerate(reg_true.group_to_identify):
            pin_jid = joint_id + 1
            pi_i = reg_true.model.inertias[pin_jid].toDynamicParameters()
            info_t = reg_true.target_joint_infos[idx]
            pi_true_list.extend(pi_i)
            pi_true_list.append(info_t["armature"])
            pi_true_list.append(info_t["damping"])
            pi_true_list.append(info_t["friction"])
            if verbose:
                print(
                    f"  [true]  [{idx}] {info_t['name']}: "
                    f"m={pi_i[0]:.6g} "
                    f"Ixx={pi_i[4]:.6g} Iyy={pi_i[6]:.6g} Izz={pi_i[9]:.6g}"
                )
        pi_true = np.array(pi_true_list)
    else:
        pi_true = pi_prior.copy()
        if verbose:
            print("  [true]  same as prior URDF (debug mode)")
    subtree_size = subtree_mask.sum(axis=1)
    joint_order = sorted(range(dof), key=lambda d: subtree_size[d])
    joint_names = [reg.target_joint_infos[d]["name"] for d in range(dof)]

    # --- Stack regressor ---
    # Y_aug from compute_regressor is TYPE-MAJOR:
    #   columns = [inertial_j0(10)...inertial_j{D-1}(10), arm_j0(1)...arm_j{D-1}(1), fric_j0(2)...fric_j{D-1}(2)]
    # pi_prior is JOINT-MAJOR:
    #   [j0(10+1+2), j1(10+1+2), ...]
    # Reorder Y_aug to joint-major so Y @ pi works correctly.
    # (``_reorder_y_aug`` is defined at module level and shared with
    #  ``data_from_measurement``.)
    Y_list, tau_list = [], []
    for k in range(N):
        Y_aug, *_ = reg.compute_regressor(
            q_traj[:, k],
            v_traj[:, k],
            a_traj[:, k],
            print_info=False,
        )
        Y_aug = _reorder_y_aug(Y_aug, dof)
        Y_list.append(Y_aug)
        tau_list.append(Y_aug @ pi_true)

    Y_stack = np.vstack(Y_list)  # (N*dof, 13*dof)
    tau_measured = np.hstack(tau_list)  # (N*dof,)

    if verbose:
        print(f"Y_stack: {Y_stack.shape}, tau: {tau_measured.shape}")

    return {
        "Y_stack": Y_stack,
        "tau_measured": tau_measured,
        "pi_prior": pi_prior,
        "pi_true": pi_true,
        "subtree_mask": subtree_mask,
        "joint_order": joint_order,
        "joint_names": joint_names,
        "dof": dof,
        "reg": reg,  # kept for the --static-test (needs sample_state/collision)
    }


# ---------------------------------------------------------------------------
# Measurement-data helpers
# ---------------------------------------------------------------------------
def _resolve_bag_dir(bag_name: str) -> Path:
    """Locate ``bag_data/<bag_name>`` (accepts a short fragment)."""
    bag_root = Path(__file__).resolve().parent.parent / "bag_data"
    bag_dir = bag_root / bag_name
    if (bag_dir / "csv").is_dir():
        return bag_dir
    matches = sorted(bag_root.glob(f"*{bag_name}*"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Short name '{bag_name}' matches multiple bags: {[m.name for m in matches]}"
        )
    raise FileNotFoundError(f"No bag found under {bag_root} matching '{bag_name}'")


def _load_measurement_csv(
    bag_name: str,
    csv_topic: str = "hardware_joint_state",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read measured joint data from ``bag_data/<bag>/csv/<csv_topic>.csv``.

    Returns ``(t, q, v, tau)``, each of shape ``(N, 24)`` (full robot joint
    vector), with ``t`` the ``t_s`` column re-zeroed to start at 0.

    ``csv_topic='hardware_joint_state'`` is the recorded joint **state**
    (feedback): its torque is the real measured torque.  Do **not** use
    ``hardware_joint_command_feedback`` here — that torque is the controller
    PD/command torque, not the actual joint torque.
    """
    import pandas as pd

    bag_dir = _resolve_bag_dir(bag_name)
    csv_path = bag_dir / "csv" / f"{csv_topic}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    n_joints = sum(c.startswith("position_") for c in df.columns)
    t = df["t_s"].to_numpy(dtype=float)
    q = df[[f"position_{i}" for i in range(n_joints)]].to_numpy(dtype=float)
    v = df[[f"velocity_{i}" for i in range(n_joints)]].to_numpy(dtype=float)
    tau = df[[f"torque_{i}" for i in range(n_joints)]].to_numpy(dtype=float)
    t = t - t[0]
    if verbose:
        print(
            f"  [measurement] {csv_path}\n"
            f"  N={len(t)}  dur={t[-1]:.2f}s  dt~{np.median(np.diff(t)) * 1e3:.2f}ms  "
            f"joints={n_joints}"
        )
    return t, q, v, tau


def read_setup_from_bag(bag_name: str) -> tuple[np.ndarray, float]:
    """从 bag 读数求平均，得到机体系重力向量与腰关节固定角度。

    * ``gravity = -mean(IMU linear_acceleration.x/y/z)``：加速度计测的是“比力”，
      机器人静止时读数 ≈ -g（指向支撑力方向），所以重力方向与 IMU 读数**相反**。
      测量时整体姿态固定不动，因此对全部样本取平均即可。
    * ``waist_yaw_offset = mean(joint_state position_12)``：测量时 J12_WAIST_YAW
      固定不动。注意这是**全程均值**——若该 bag 录制期间腰关节真的动过，均值只是
      近似，请改用 ``--waist-offset`` 手动指定。

    Returns ``(gravity (3,), waist_yaw_offset)``.
    """
    import pandas as pd

    from identification.target_limb_regressor import WAIST_Q_INDICES

    bag_dir = _resolve_bag_dir(bag_name)

    # --- 重力：IMU 线加速度取平均后取负 ---
    imu_path = bag_dir / "csv" / f"{IMU_CSV_TOPIC}.csv"
    if not imu_path.is_file():
        raise FileNotFoundError(
            f"IMU CSV not found: {imu_path} → 用 --gravity 手动指定"
        )
    acc = pd.read_csv(
        imu_path,
        usecols=[
            "linear_acceleration.x",
            "linear_acceleration.y",
            "linear_acceleration.z",
        ],
    ).to_numpy(dtype=float)
    acc_mean = acc.mean(axis=0)
    gravity = -acc_mean  # 重力方向与 IMU 读数相反

    # --- 腰关节：joint_state 位置列求平均 ---
    waist_idx = WAIST_Q_INDICES[0]
    _, q, _, _ = _load_measurement_csv(bag_name, verbose=False)
    waist_col = q[:, waist_idx]
    waist_yaw = float(waist_col.mean())

    print(f"  [setup] bag={bag_dir.name}")
    print(
        f"    IMU {imu_path.name}: N={len(acc)}  "
        f"mean(linear_acc)=[{acc_mean[0]:.6f}, {acc_mean[1]:.6f}, {acc_mean[2]:.6f}]"
        f"  -> gravity=[{gravity[0]:.6f}, {gravity[1]:.6f}, {gravity[2]:.6f}]"
        f"  |g|={np.linalg.norm(gravity):.6f} m/s²"
    )
    print(
        f"    joint_state position_{waist_idx}: N={len(waist_col)}  "
        f"mean={waist_yaw:.9f} rad "
        f"(min={waist_col.min():.6f}, max={waist_col.max():.6f})"
    )
    return gravity, waist_yaw


def _latest_recovered_yaml(exclude: str | None = None) -> str:
    """Return the latest ``recovered_*.yaml`` in ``trajectory_coefficients/``.

    ``exclude`` (optional): a filename to skip, e.g. the identification YAML
    when picking a different validation trajectory by default.
    """
    from identification.fourier_trajectory import FourierTrajectory

    matches = sorted(FourierTrajectory._coeffs_dir.glob("recovered_*.yaml"))
    if exclude:
        matches = [m for m in matches if m.name != exclude]
    if not matches:
        raise FileNotFoundError(
            "No recovered_*.yaml in trajectory_coefficients/ — run fourier_fit.py first"
        )
    return matches[-1].name


def _yaml_source_bag(trajectory_yaml: str) -> str:
    """Return the ``_meta.source_bag`` (short fragment) recorded in a YAML.

    e.g. ``recovered_260817_104829.yaml`` → ``'55_31'`` (bag ``13_55_31``).
    """
    import yaml

    from identification.fourier_trajectory import FourierTrajectory

    yaml_path = FourierTrajectory._coeffs_dir / trajectory_yaml
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Trajectory coeffs YAML not found: {yaml_path}")
    with open(yaml_path) as f:
        meta = yaml.safe_load(f).get("_meta", {})
    sb = meta.get("source_bag")
    if not sb:
        raise ValueError(
            f"{trajectory_yaml} _meta has no source_bag; pass bag_name explicitly"
        )
    return str(sb)


def _latest_pso_yaml_for_group(group: str, exclude: str | None = None) -> str | None:
    """Latest ``pso_unified_*.yaml`` whose ``_meta.group`` matches ``group``.

    Used to auto-link the parameter-quality YAML (from the PSO excitation
    design) to a recovered (measured) trajectory YAML, which carries no
    ``_diagnostics``.  Returns ``None`` if no match is found.
    """
    from identification.fourier_trajectory import FourierTrajectory

    matches = sorted(FourierTrajectory._coeffs_dir.glob("pso_unified_*.yaml"))
    for m in reversed(matches):
        if exclude and m.name == exclude:
            continue
        if FourierTrajectory.load_meta(m.name).get("group") == group:
            return m.name
    return None


def _yaml_has_quality(yaml_name: str) -> bool:
    """True if a coeffs YAML carries ``_diagnostics.per_param`` quality labels."""
    from identification.fourier_trajectory import FourierTrajectory

    path = FourierTrajectory._coeffs_dir / yaml_name
    if not path.is_file():
        return False
    return bool(load_yaml_param_quality(path))


def _recovered_at_times(
    t: np.ndarray,
    dof: int,
    trajectory_yaml: str,
    grid_sample_rate: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a recovered Fourier trajectory at arbitrary (CSV) times.

    Uses the new ``generate_trajectory(t=...)`` interface: the Fourier phase
    for a physical time ``t_phys`` is ``time_coeffs * t_phys``, so q/v/a are
    computed **directly** at the CSV times (no interpolation needed).

    Returns ``(q, v, a)``, each ``(N, dof)``.
    """
    import yaml

    from identification.fourier_trajectory import TRAJ_PERIOD, FourierTrajectory

    yaml_path = FourierTrajectory._coeffs_dir / trajectory_yaml
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Trajectory coeffs YAML not found: {yaml_path}")
    with open(yaml_path) as f:
        meta = yaml.safe_load(f).get("_meta", {})
    tc = meta.get("time_coeffs")
    if tc is None:
        f0 = meta.get("f0_hz")
        if f0 is None:
            raise ValueError(
                f"{trajectory_yaml} _meta has no time_coeffs/f0_hz; "
                "re-fit with fourier_fit.py or pass --time-coeffs"
            )
        tc = float(f0) * TRAJ_PERIOD
    tc = float(tc)
    ft = FourierTrajectory(dim=dof, sample_rate=grid_sample_rate, time_coeffs=tc)
    q_th, v_th, a_th = ft.generate_trajectory_from_yaml(
        trajectory_yaml, t=tc * np.asarray(t, dtype=float)
    )
    return q_th.T, v_th.T, a_th.T  # each (N, dof)


def data_from_measurement(
    urdf_path: str | Path,
    bag_name: str | None = None,
    limb_group: str | None = None,
    csv_topic: str = "hardware_joint_state",
    sample_rate: float = 100.0,
    verbose: bool = True,
    gravity: np.ndarray | None = None,
    waist_yaw_offset: float | None = None,
    trajectory_yaml: str | None = None,
    twin: str | None = None,
    tau_delay: float = 0.0,
) -> dict:
    """Prepare SDP data from **real** bag measurement (CSV).

    Prior model, subtree mask and joint order are identical to
    ``prepare_data_from_urdf`` (the quality-weighted ridge is configured in
    ``main``).  Only the two data streams change:

    1. ``tau_measured`` is read **directly** from the bag CSV
       (``torque_<j>`` columns for the limb joints) instead of ``Y @ pi_true``.
    2. The URDF regressor ``Y_stack`` is evaluated at the **CSV time samples**
       (optionally decimated to ``sample_rate``) using the recovered Fourier
       trajectory's ``(q, v, a)`` at those times, so ``Y_stack @ pi`` and
       ``tau_measured`` are aligned row by row.

    Parameters
    ----------
    urdf_path : str or Path
        Initial/prior URDF → regressor, ``pi_prior``, subtree mask, joint order.
    bag_name : str or None
        Bag under ``bag_data/`` (full dir name or short fragment).  ``None``
        (default) → read from ``trajectory_yaml``'s ``_meta.source_bag``.
    limb_group : str or None
        Limb group, e.g. ``'left_arm'`` → CSV joints 13..17.  ``None``
        (default) → read from ``trajectory_yaml``'s ``_meta.group``
        (fallback ``'left_arm'``).
    csv_topic : str
        CSV topic under ``<bag>/csv/``.  Default ``'hardware_joint_state'``
        (real measured torque).  Avoid ``hardware_joint_command_feedback``.
    sample_rate : float
        Decimation rate (Hz) for the regression samples.  The CSV (~500 Hz) is
        decimated to roughly this rate; the time axis stays the actual CSV
        ``t_s`` values.
    trajectory_yaml : str or None
        A ``recovered_*.yaml`` in ``trajectory_coefficients/`` to evaluate the
        state ``(q, v, a)`` at the CSV times.  ``None`` (default) → latest
        ``recovered_*.yaml``.
    twin : str or None
        Optional ``'START:END'`` time window (seconds) applied to the CSV
        samples before building the regressor.  The trajectory is periodic, so
        a single good-quality period already carries all the information;
        cropping to it reduces data size / noise.  Either end may be omitted;
        ``None`` = full span.
    tau_delay : float
        Time-delay compensation of the **torque channel** (seconds).  The
        measured torque at time ``t`` is assumed to describe the state at
        ``t − tau_delay``, so the regressor is evaluated there.
        ``> 0`` = the torque lags the state (the usual transport / filter
        delay); negative values are allowed (torque leading the state).  A pure
        time shift rotates the regressor columns — in particular the armature
        column becomes ``cos(ωδ)·q̈(t) + ω·sin(ωδ)·q̇(t)`` — so it moves weight
        between **armature and damping**, which is precisely the direction in
        which the free fit drifts away from the URDF prior.  ``0`` = no
        compensation.

    Notes
    -----
    The CSV provides position/velocity but **no acceleration**, and the
    measured velocity is quantized, so numeric differentiation is too noisy.
    The regressor state ``(q, v, a)`` is therefore taken **only** from the
    recovered Fourier trajectory, evaluated **directly** at the CSV time
    samples (``generate_trajectory(t=time_coeffs * t_phys)`` — see
    ``_recovered_at_times``).

    In this measurement mode there is **no ground truth**: ``pi_true`` is
    always ``None`` (unknown) and the identification uses the measured torque
    only.
    """
    from identification.fourier_trajectory import FourierTrajectory
    from identification.target_limb_regressor import (
        TargetLimbRegressor,
        VALID_LIMB_GROUPS,
    )

    # --- Resolve trajectory yaml + limb group + bag name.  The bag is read
    #     from the yaml's _meta.source_bag (e.g. '55_31' → bag 13_55_31) and
    #     the limb group from the yaml's _meta.group (fallback 'left_arm')
    #     when not given explicitly. ---
    if trajectory_yaml is None:
        trajectory_yaml = _latest_recovered_yaml()
    if limb_group is None:
        limb_group = FourierTrajectory.load_group(trajectory_yaml, default="left_arm")
    if bag_name is None:
        bag_name = _yaml_source_bag(trajectory_yaml)
        if verbose:
            print(f"  [measurement] bag from yaml _meta.source_bag: {bag_name}")
    if verbose:
        print(f"  [measurement] limb group from yaml _meta.group: {limb_group}")

    joint_indices = list(VALID_LIMB_GROUPS[limb_group])

    # --- Prior model (identical to prepare_data_from_urdf) ---
    reg = TargetLimbRegressor(
        urdf_path=Path(urdf_path),
        group_to_identify=limb_group,
        print_info=False,
        gravity=gravity,
        waist_yaw_offset=waist_yaw_offset,
    )
    dof = reg.dof

    pi_prior_list = []
    for idx, joint_id in enumerate(reg.group_to_identify):
        pin_jid = joint_id + 1
        pi_inertial = reg.model.inertias[pin_jid].toDynamicParameters()
        info = reg.target_joint_infos[idx]
        pi_prior_list.extend(pi_inertial)
        pi_prior_list.append(info["armature"])
        pi_prior_list.append(info["damping"])
        pi_prior_list.append(info["friction"])
        if verbose:
            print(
                f"  [prior] [{idx}] {info['name']}: "
                f"m={pi_inertial[0]:.6g} "
                f"Ixx={pi_inertial[4]:.6g} Iyy={pi_inertial[6]:.6g} Izz={pi_inertial[9]:.6g}"
            )
    pi_prior = np.array(pi_prior_list)

    reg.compute_regressor(print_info=False)
    subtree_mask = reg.subtree_mask.copy()
    subtree_size = subtree_mask.sum(axis=1)
    joint_order = sorted(range(dof), key=lambda d: subtree_size[d])
    joint_names = [reg.target_joint_infos[d]["name"] for d in range(dof)]

    # 测量模式没有地面真值：pi_true 恒为 None（未知），只靠实测 tau 辨识。
    pi_true = None

    # --- Measurement: read CSV — only the time axis and measured tau ---
    t_meas, q_meas, v_meas, tau_meas = _load_measurement_csv(
        bag_name, csv_topic, verbose=verbose
    )
    tau_arm = tau_meas[:, joint_indices]  # (N, dof)

    # --- Keep the CSV time samples (decimated to ~sample_rate) ---
    dt_meas = float(np.median(np.diff(t_meas)))
    step = max(1, round((1.0 / sample_rate) / dt_meas))
    idx = np.arange(0, len(t_meas), step)
    t_sel = t_meas[idx]
    tau_arm = tau_arm[idx]
    if verbose:
        print(
            f"  [measurement] decimate ~500Hz -> ~{sample_rate}Hz "
            f"(step={step}): N={len(t_sel)}"
        )

    # --- Optional identification time window (like plot TWIN): the trajectory
    #     is periodic, so a single good period carries all the information;
    #     cropping to it reduces noise / data size. ---
    if twin is not None:
        tw0, tw1 = _parse_window(twin, t_sel[-1])
        wmask = (t_sel >= tw0) & (t_sel <= tw1)
        if not np.any(wmask):
            raise ValueError(
                f"Identification time window [{tw0:.3g}, {tw1:.3g}] contains no samples"
            )
        t_sel = t_sel[wmask]
        tau_arm = tau_arm[wmask]
        if verbose:
            print(f"  [measurement] ID time window {twin}: N={len(t_sel)}")

    # --- State q/v/a: ONLY from the recovered Fourier trajectory, evaluated
    #     at the CSV time samples (phase-aligned, see _recovered_at_times).
    #     The CSV has position/velocity but no acceleration; numeric
    #     differentiation of the quantized velocity is too noisy, so it is
    #     NOT used — acceleration comes from the analytic Fourier trajectory
    #     together with q and v. ---
    # --- Optional torque-channel delay compensation: the measured torque at
    #     time t is assumed to describe the state at t − tau_delay, so the
    #     regressor is evaluated there.  A pure shift turns the armature column
    #     q̈(t−δ) into cos(ωδ)·q̈(t) + ω·sin(ωδ)·q̇(t), i.e. it trades weight
    #     between armature and damping — exactly where the free fit drifted. ---
    t_state = t_sel - float(tau_delay)
    q_arm, v_arm, a_arm = _recovered_at_times(
        t_state, dof, trajectory_yaml, grid_sample_rate=500.0
    )
    if verbose:
        print(f"  [measurement] q/v/a from Fourier trajectory {trajectory_yaml}")
        if tau_delay:
            print(
                f"  [measurement] tau delay {tau_delay * 1e3:+.2f} ms → regressor "
                f"evaluated at t − {tau_delay:g}s (torque lags state if > 0)"
            )

    # --- Stack regressor at the CSV time samples ---
    Y_stack = _stack_regressor(reg, q_arm, v_arm, a_arm)  # (N*dof, 13*dof)
    # Row-major: sample k, joint d → row k·dof + d (matches the solver layout)
    tau_measured = tau_arm.reshape(-1)  # (N*dof,)

    if verbose:
        print(f"Y_stack: {Y_stack.shape}, tau (measured): {tau_measured.shape}")

    return {
        "Y_stack": Y_stack,
        "tau_measured": tau_measured,
        "pi_prior": pi_prior,
        "pi_true": pi_true,
        "subtree_mask": subtree_mask,
        "joint_order": joint_order,
        "joint_names": joint_names,
        "dof": dof,
        "t": t_sel,
    }


# ============================================================================
# YAML parameter-quality helpers
# ============================================================================
# YAML parameter-order constants (MUST match pso_excitation_unified output)
#
# ⚠️ FIXED 2026-09-14 (user "Iyy 和 Ixz 的质量映射反了"): this used to be
#    {4:4, 5:5, 6:7, 7:6, 8:8, 9:9}, whose comment claimed the YAML inertia order
#    differs from ours ([Ixx,Ixy,Iyy,Ixz,Iyz,Izz] vs [Ixx,Ixy,Ixz,Iyy,Iyz,Izz]).
#    That claim is wrong — the two orders are THE SAME, so the map is the
#    IDENTITY.  Evidence:
#      · pin.Inertia(5,[0,0,0],[[1,.1,.2],[.1,2,.3],[.2,.3,3]]).toDynamicParameters()
#        [4:10] == [1, .1, 2, .2, .3, 3] = [Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
#        (verified numerically; Iyy is idx 6, Ixz is idx 7);
#      · pso_traj_excitation._INERTIAL_PARAM_NAMES (which WRITES the YAML) is
#        [mass, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz];
#      · our own `_PARAM_LABELS` above is idx 6 = Iyy, idx 7 = Ixz;
#      · every YAML in trajectory_coefficients/ carries `per_param[].name`
#        ("<JOINT>/<param>") and idx 6/7 are named Iyy/Ixz there too.
#    Effect of the bug: ridge weight + hard-freeze mask were applied to the wrong
#    physical parameter for every joint (which one is allowed to move).
#    `load_yaml_param_quality` now cross-checks the `name` field against this
#    mapping, so this class of drift fails loudly instead of silently.
_YAML_INERTIA_YAML2OUR = {4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
# YAML inertia: [Ixx, Ixy, Iyy, Ixz, Iyz, Izz] → our Pinocchio: [Ixx, Ixy, Iyy, Ixz, Iyz, Izz]

# YAML `per_param[].name` suffix → our local index.  Used ONLY to validate the
# index mapping above (the index mapping stays the source of truth); an unknown
# suffix is skipped rather than guessed at.
_YAML_NAME_SUFFIX_TO_LOCAL = {
    "mass": 0,
    "mcx": 1,
    "mcy": 2,
    "mcz": 3,
    "Ixx": 4,
    "Ixy": 5,
    "Iyy": 6,
    "Ixz": 7,
    "Iyz": 8,
    "Izz": 9,
    "armature": 10,
    "damping": 11,
    "friction": 12,
}


def _yaml_global_to_local(global_idx: int, dof: int) -> tuple[int, int]:
    """
    Convert YAML global index → (joint, param) in our 13-per-joint layout.

    YAML layout (total = 10*dof + dof + 2*dof = 13*dof):
      [j0_inertial(10), j1_inertial(10), ..., j{D-1}_inertial(10),
       j0_arm(1), ..., j{D-1}_arm(1),
       j0_damp(1), j0_fric(1), j1_damp(1), j1_fric(1), ...]
    """
    n_inertial = 10 * dof
    if global_idx < n_inertial:
        joint = global_idx // 10
        yaml_i = global_idx % 10
        our_i = yaml_i if yaml_i < 4 else _YAML_INERTIA_YAML2OUR[yaml_i]
        return joint, our_i
    elif global_idx < n_inertial + dof:
        joint = global_idx - n_inertial
        return joint, 10
    else:
        idx = global_idx - n_inertial - dof
        joint = idx // 2
        our_i = 11 + (idx % 2)
        return joint, our_i


def load_yaml_param_quality(yaml_path: str | Path) -> dict[int, str]:
    """
    Parse YAML diagnostics, return quality labels keyed by OUR global index
    (0..dof*13-1, joint-major by 13).

    Every entry's ``name`` field (``"<JOINT>/<param>"``) is cross-checked against
    the index mapping; if the name and the index disagree, a ``ValueError`` is
    raised (that is how the 2026-09-14 Iyy/Ixz swap would have been caught).
    """
    import yaml

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    per_param = data.get("_diagnostics", {}).get("per_param", [])
    dof = len(per_param) // N_PER_JOINT

    result = {}
    mismatch: list[str] = []
    for entry in per_param:
        yaml_g = entry["idx"]
        j, i = _yaml_global_to_local(yaml_g, dof)
        name = entry.get("name")
        if name:
            suffix = str(name).rsplit("/", 1)[-1]
            expected = _YAML_NAME_SUFFIX_TO_LOCAL.get(suffix)
            if expected is not None and expected != i:
                mismatch.append(
                    f"idx {yaml_g}: name '{name}' says '{suffix}' (local "
                    f"{expected}) but the index mapping gives local {i}"
                )
        our_g = j * N_PER_JOINT + i
        result[our_g] = entry["quality"]
    if mismatch:
        raise ValueError(
            f"{yaml_path}: quality YAML parameter order does not match "
            "_YAML_INERTIA_YAML2OUR / _yaml_global_to_local — "
            f"{len(mismatch)} of {len(per_param)} entries disagree:\n  "
            + "\n  ".join(mismatch[:8])
            + (f"\n  ... and {len(mismatch) - 8} more" if len(mismatch) > 8 else "")
        )
    return result


def _ridge_weight_for_quality(quality: str | None) -> float:
    """Map a YAML quality label → its relative-deviation ridge weight (w_q).

    ``RIDGE_FREEZE_QUALITIES`` labels (``null``/``small``) are **excluded from
    the ridge** — they are hard-frozen instead — so asking for their weight is
    a programming error, not a fallback case.
    """
    if quality in RIDGE_FREEZE_QUALITIES:
        raise ValueError(
            f"quality '{quality}' does not participate in the ridge "
            f"(it is hard-frozen via RIDGE_FREEZE_QUALITIES); no ridge weight "
            f"exists for it"
        )
    if quality is None or quality not in RIDGE_QUALITY_WEIGHTS:
        return RIDGE_QUALITY_WEIGHTS[DEFAULT_QUALITY]
    return RIDGE_QUALITY_WEIGHTS[quality]


def build_ridge_weights(
    quality_map: dict[int, str] | None,
    pi_prior: np.ndarray,
    *,
    ridge_lambda: float = RIDGE_LAMBDA,
    scale_floor: float = RIDGE_SCALE_FLOOR,
    freeze_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build per-parameter weighted-ridge quadratic coefficients c_i.

    Replaces the per-quality box (freeze / widen) with a ridge penalty.  For
    every global parameter i with URDF prior ``prior_i``, deviation from the
    prior is penalised in *relative* terms

        c_i · (pi_i − prior_i)² ,   c_i = ridge_lambda · w_q / scale_i²

    with ``scale_i = max(|prior_i|, scale_floor)`` and ``w_q`` from
    ``RIDGE_QUALITY_WEIGHTS``.  Better quality → smaller ``w_q`` → the
    parameter may drift further from the prior; poor quality → larger ``w_q``
    → it is pulled back hard.

    Parameters whose quality label is in ``RIDGE_FREEZE_QUALITIES``
    (``null``/``small``) get ``c_i = 0`` — they are **excluded from the ridge
    norm** and are instead hard-frozen at the prior by ``build_freeze_mask``.
    The same applies to every entry flagged in ``freeze_mask``, so the mask
    built by ``build_freeze_mask`` is the single source of truth for "frozen ⇒
    no ridge".

    Parameters
    ----------
    quality_map : dict[int, str] or None
        ``{global_idx: quality}`` as returned by ``load_yaml_param_quality``.
        ``None``/empty → every parameter gets the default quality weight.
    pi_prior : (dof*13,) ndarray
        URDF prior parameters (the ridge attractor).
    ridge_lambda : float
        Global ridge strength multiplying every ``w_q``.
    scale_floor : float
        Lower bound on the relative-deviation scale, so parameters whose
        prior is ~0 do not get an astronomically large coefficient.
    freeze_mask : (dof*13,) bool ndarray or None
        Where ``True``, the parameter is forced to ``c_i = 0`` (excluded from
        the ridge norm).  Pass the mask from ``build_freeze_mask`` to keep the
        "frozen" and "zero ridge weight" sets identical.

    Returns
    -------
    c : (dof*13,) ndarray
        Quadratic-coefficient vector (exactly 0 wherever the parameter is
        excluded from the ridge).
    """
    dof = len(pi_prior) // N_PER_JOINT
    pi_list = split_joint_params(pi_prior)
    c = np.zeros(dof * N_PER_JOINT)
    for g in range(dof * N_PER_JOINT):
        q = (quality_map or {}).get(g)
        if q in RIDGE_FREEZE_QUALITIES:
            continue  # 冻结档：不参与岭回归（c 保持 0）
        w_q = _ridge_weight_for_quality(q)
        j, i = g // N_PER_JOINT, g % N_PER_JOINT
        prior_val = pi_list[j][i]
        scale = max(abs(prior_val), scale_floor)
        c[g] = ridge_lambda * w_q / (scale * scale)
    if freeze_mask is not None:
        # 冻结 ⇒ 退出岭范数（与 build_freeze_mask 同源，保证二者一致）
        c[np.asarray(freeze_mask, dtype=bool).reshape(-1)] = 0.0
    return c


def build_freeze_mask(
    quality_map: dict[int, str] | None,
    dof: int,
    freeze_qualities: frozenset[str] = RIDGE_FREEZE_QUALITIES,
    freeze_local_indices: frozenset[int] = RIDGE_FREEZE_LOCAL_INDICES,
) -> np.ndarray:
    """
    (dof*13,) boolean mask of parameters to **hard-freeze** (pi == prior).

    Two independent sources, unioned:

    1. Quality labels in ``freeze_qualities`` (default ``null``/``small``) —
       too tiny for a large ridge weight to pin exactly.
    2. The local indices in ``freeze_local_indices`` (default
       ``RIDGE_FREEZE_LOCAL_INDICES`` = armature/damping/friction) for **every**
       joint, regardless of quality label.

    Frozen parameters are frozen at their URDF prior by an equality constraint
    in the solver, so they cannot move at all.  They are **simultaneously
    excluded from the ridge norm** (``c = 0``, see ``build_ridge_weights``);
    both halves are required, since a parameter with ``c = 0`` that is *not*
    frozen would be completely unregularised.

    Raises
    ------
    ValueError
        If a quality entry points outside ``[0, dof*13)`` — i.e. the quality
        YAML does not belong to this limb group, which would otherwise
        silently freeze the wrong parameters.
    """
    n_par = dof * N_PER_JOINT
    mask = np.zeros(n_par, dtype=bool)
    if quality_map:
        for g, q in quality_map.items():
            if not 0 <= g < n_par:
                raise ValueError(
                    f"quality YAML index {g} is outside [0, {n_par}) for "
                    f"dof={dof} — the quality YAML does not match this limb group"
                )
            if q in freeze_qualities:
                mask[g] = True
    for j in range(dof):
        for i in freeze_local_indices:
            mask[j * N_PER_JOINT + i] = True
    return mask


# ============================================================================
# Torque comparison plot
# ============================================================================
def _parse_window(s: str | None, t_end: float) -> tuple[float, float]:
    """Parse a ``'START:END'`` time window (either end optional); None = full."""
    if s is None:
        return 0.0, t_end
    parts = str(s).split(":")
    if len(parts) > 2:
        raise ValueError(f"Time window should be 'start:end', got: '{s}'")
    t0 = float(parts[0]) if parts[0].strip() else 0.0
    t1 = float(parts[1].strip()) if len(parts) == 2 and parts[1].strip() else t_end
    return t0, t1


def _plot_torque_comparison_panels(
    tau_true: np.ndarray,
    tau_prior: np.ndarray,
    tau_ident: np.ndarray,
    joint_names: list[str],
    joint_order: list[int],
    dof: int,
    sample_rate: float = 100.0,
    offset: float | None = None,
    show_residual: bool = True,
    true_label: str = "true",
    title: str = "Joint Torque Comparison",
    twin: str | None = None,
) -> list:
    """Per-joint torque comparison plots — **one figure per joint**.

    Each figure has the main torque panel and (optionally) a residual panel
    below it.  ``twin`` is an optional ``'START:END'`` time window (seconds)
    applied to the time axis (either end may be omitted; ``None`` = full),
    matching ``compare_torque.py -w/--twin``.

    The three curves often nearly overlap when identification is good,
    so this version improves readability via:

    * **Residual panels** (default on): a sub-panel below each torque plot
      shows ``τ_true − τ_prior`` and ``τ_true − τ_identified`` with per-curve
      RMSE in the legend — even sub-0.1 Nm gaps become clearly visible.
    * **offset** (optional, Nm): shift the curves vertically
      (prior +offset, identified −offset) to fully separate them.
      Residuals are always computed from the un-shifted data.

    Returns the list of created ``matplotlib.figure.Figure`` objects.
    """
    import matplotlib.pyplot as plt

    N_total = len(tau_true)
    N = N_total // dof
    t = np.arange(N) / sample_rate

    # --- Optional time window (like compare_torque.py -w/--twin) ---
    t0, t1 = _parse_window(twin, t[-1])
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        raise ValueError(f"Time window [{t0:.3g}, {t1:.3g}] contains no samples")
    t = t[mask]

    figs: list = []
    for idx, d in enumerate(joint_order):
        row = np.arange(d, N_total, dof)[mask]
        y_true = tau_true[row]
        y_prior = tau_prior[row]
        y_ident = tau_ident[row]

        if offset:
            y_prior = y_prior + offset
            y_ident = y_ident - offset

        if show_residual:
            fig, (ax, ax_r) = plt.subplots(
                2,
                1,
                figsize=(12, 6.4),
                sharex=True,
                gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.1},
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 3.4))
            ax_r = None

        # --- Torque comparison panel ---
        ax.plot(t, y_true, "r-", linewidth=2.2, label=true_label)
        ax.plot(t, y_prior, "b--", linewidth=1.6, alpha=0.9, label="prior")
        ax.plot(t, y_ident, "g-", linewidth=2.0, label="identified")
        ax.set_ylabel(f"{joint_names[d]}\n[Nm]")
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t[0], t[-1])

        # --- Residual panel: differences vs true ---
        if ax_r is not None:
            rmse_prior = np.sqrt(np.mean((tau_true[row] - tau_prior[row]) ** 2))
            rmse_ident = np.sqrt(np.mean((tau_true[row] - tau_ident[row]) ** 2))
            ax_r.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
            ax_r.plot(
                t,
                tau_true[row] - tau_prior[row],
                "b--",
                linewidth=1.4,
                alpha=0.9,
                label=f"{true_label}−prior   RMSE {rmse_prior:.3g}",
            )
            ax_r.plot(
                t,
                tau_true[row] - tau_ident[row],
                "g-",
                linewidth=1.6,
                label=f"{true_label}−identified   RMSE {rmse_ident:.3g}",
            )
            ax_r.set_ylabel("Δτ [Nm]")
            ax_r.legend(loc="upper right", fontsize=7)
            ax_r.grid(True, alpha=0.3)
            ax_r.set_xlim(t[0], t[-1])
            ax_r.set_xlabel("Time [s]")
        else:
            ax.set_xlabel("Time [s]")

        figs.append(fig)

    for fig in figs:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()
    return figs


def plot_torque_comparison_simulated(
    result: IdentificationResult,
    joint_names: list[str],
    Y_stack: np.ndarray,
    pi_reference: np.ndarray | None = None,
    sample_rate: float = 100.0,
    offset: float | None = None,
    show_residual: bool = True,
    twin: str | None = None,
):
    """Plot τ_true vs τ_prior vs τ_identified — simulation case (pi_true known).

    ``tau_true = Y_stack @ pi_reference`` (falls back to the prior when no
    reference is given).  This is the original ``plot_torque_comparison``
    behaviour for URDF-synthesized data.
    """
    dof = len(result.joint_order)
    pi_true = pi_reference if pi_reference is not None else result.pi_prior
    tau_true = Y_stack @ pi_true
    tau_prior = Y_stack @ result.pi_prior
    tau_ident = Y_stack @ result.pi_identified

    return _plot_torque_comparison_panels(
        tau_true,
        tau_prior,
        tau_ident,
        joint_names,
        result.joint_order,
        dof,
        sample_rate=sample_rate,
        offset=offset,
        show_residual=show_residual,
        true_label="true",
        title="Joint Torque Comparison (simulated): true vs prior vs identified",
        twin=twin,
    )


def plot_torque_comparison_measured(
    result: IdentificationResult,
    urdf_path: str | Path,
    val_yaml: str,
    val_bag_name: str | None = None,
    joint_names: list[str] | None = None,
    limb_group: str | None = None,
    csv_topic: str = "hardware_joint_state",
    sample_rate: float = 100.0,
    gravity: np.ndarray | None = None,
    waist_yaw_offset: float | None = None,
    grid_sample_rate: float = 500.0,
    tau_delay: float = 0.0,
    offset: float | None = None,
    show_residual: bool = True,
    verbose: bool = True,
    twin: str | None = None,
):
    """Cross-validation plot on a held-out measurement (pi_true unknown).

    Identification used one bag + one recovered trajectory.  To validate, we
    run a **different** (manually specified) trajectory YAML at a **different**
    bag's CSV times with the prior and the identified parameters, and compare
    the resulting joint torques against that bag's measured torque:

        1. Print the identified pi parameters (per joint).
        2. Build the URDF regressor at the validation trajectory's q/v/a
           sampled at the validation bag's CSV times.
        3. ``tau_prior = Y_val @ pi_prior``,
           ``tau_ident = Y_val @ pi_identified``.
        4. ``tau_measured`` read from the validation bag CSV.
        5. Per-joint plot (measured vs prior vs identified + residuals) and a
           per-joint RMSE table.

    ``val_bag_name`` defaults to ``val_yaml``'s ``_meta.source_bag`` when not
    given explicitly.
    """
    from identification.fourier_trajectory import FourierTrajectory
    from identification.target_limb_regressor import TargetLimbRegressor

    # Limb group defaults to the validation yaml's _meta.group (fallback
    # 'left_arm' for legacy recovered_*.yaml that predate the group field).
    if limb_group is None:
        limb_group = FourierTrajectory.load_group(val_yaml, default="left_arm")

    dof = len(result.joint_order)
    if joint_names is None:
        joint_names = [f"joint_{d}" for d in range(dof)]

    # --- 1) Print identified pi parameters ---
    # pi_prior_list = split_joint_params(result.pi_prior)
    # pi_ident_list = split_joint_params(result.pi_identified)
    # print("\n" + "=" * 100)
    # print("IDENTIFIED PARAMETERS (prior → identified)".center(100))
    # print("=" * 100)
    # for d in result.joint_order:
    #     print(f"\n--- Joint {d}: {joint_names[d]} ---")
    #     print(f"{'Param':<10s} {'Prior':>12s} {'Identified':>12s} {'Δ%':>9s}")
    #     print("-" * 46)
    #     for i in range(N_PER_JOINT):
    #         pr = pi_prior_list[d][i]
    #         idn = pi_ident_list[d][i]
    #         dp = (idn - pr) / max(abs(pr), 1e-12) * 100
    #         print(f"{_PARAM_LABELS[i]:<10s} {pr:>12.6g} {idn:>12.6g} {dp:>8.2f}%")

    # --- 2) Regressor (same prior model as identification) ---
    reg = TargetLimbRegressor(
        urdf_path=Path(urdf_path),
        group_to_identify=limb_group,
        print_info=False,
        gravity=gravity,
        waist_yaw_offset=waist_yaw_offset,
    )
    joint_indices = list(reg.group_to_identify)
    assert reg.dof == dof, (
        f"limb_group '{limb_group}' dof={reg.dof} != identification dof={dof}"
    )

    # --- 3) Load the validation bag (different from the identification bag).
    #        If not given, the bag name is read from the yaml's
    #        _meta.source_bag (e.g. '57_28' → bag 13_57_28). ---
    if val_bag_name is None:
        val_bag_name = _yaml_source_bag(val_yaml)
        if verbose:
            print(f"  [validation] val_bag from yaml _meta.source_bag: {val_bag_name}")
    t_val, _, _, tau_val = _load_measurement_csv(
        val_bag_name, csv_topic, verbose=verbose
    )
    tau_arm = tau_val[:, joint_indices]

    # --- Keep the validation bag's CSV times (decimated to ~sample_rate) ---
    dt_val = float(np.median(np.diff(t_val)))
    step = max(1, round((1.0 / sample_rate) / dt_val))
    idx = np.arange(0, len(t_val), step)
    t_sel = t_val[idx]
    tau_arm = tau_arm[idx]
    if verbose:
        print(
            f"  [validation] bag={val_bag_name}  yaml={val_yaml}\n"
            f"  decimate ~500Hz -> ~{sample_rate}Hz (step={step}): N={len(t_sel)}"
        )

    # --- 4) State q/v/a from the validation trajectory at the bag times ---
    q_arm, v_arm, a_arm = _recovered_at_times(
        t_sel - float(tau_delay), dof, val_yaml, grid_sample_rate=grid_sample_rate
    )
    if verbose:
        print(f"  [validation] q/v/a from Fourier trajectory {val_yaml}")

    # --- 5) Regressor + predicted torques (prior & identified) ---
    Y_val = _stack_regressor(reg, q_arm, v_arm, a_arm)  # (N*dof, 13*dof)
    tau_measured = tau_arm.reshape(-1)  # (N*dof,)
    tau_prior = Y_val @ result.pi_prior
    tau_ident = Y_val @ result.pi_identified
    if verbose:
        print(
            f"  [validation] Y_val: {Y_val.shape}, tau (measured): {tau_measured.shape}"
        )

    # --- 6) RMSE table on the held-out data ---
    stats = print_rmse_comparison(result, joint_names, Y_val, tau_measured)

    # --- 7) Plot ---
    figures = _plot_torque_comparison_panels(
        tau_measured,
        tau_prior,
        tau_ident,
        joint_names,
        result.joint_order,
        dof,
        sample_rate=sample_rate,
        offset=offset,
        show_residual=show_residual,
        true_label="measured",
        title=(
            f"Joint Torque Comparison (validation): measured vs prior vs identified\n"
            f"bag={val_bag_name}  yaml={val_yaml}"
        ),
        twin=twin,
    )
    return {"stats": stats, "figures": figures}


def plot_torque_comparison_simulated_validation(
    result: IdentificationResult,
    urdf_path: str | Path,
    val_yaml: str,
    pi_true: np.ndarray,
    joint_names: list[str] | None = None,
    limb_group: str | None = None,
    sample_rate: float = 100.0,
    time_coeffs: float = 1.0,
    gravity: np.ndarray | None = None,
    waist_yaw_offset: float | None = None,
    offset: float | None = None,
    show_residual: bool = True,
    verbose: bool = True,
    twin: str | None = None,
    plot: bool = True,
):
    """Held-out cross-validation of sim identification on a different YAML.

    The sim identification was fit on one (training) excitation YAML whose
    synthetic torque came from the true URDF.  This stage is the sim analogue
    of ``plot_torque_comparison_measured`` (which validates on a different
    measured bag): it re-runs **both** the URDF prior and the identified
    parameters on a *different* (manually specified, e.g. ``verify_*.yaml``)
    trajectory, and compares their predicted torques against the true torque
    ``tau_true = Y_val @ pi_true`` synthesised from the **same true model** on
    that new trajectory:

        1. Build the URDF regressor (prior model) at the validation
           trajectory's q/v/a (same sample_rate / time_coeffs playback as
           identification).
        2. ``tau_true  = Y_val @ pi_true``
           ``tau_prior = Y_val @ pi_prior``
           ``tau_ident = Y_val @ pi_identified``.
        3. Per-joint RMSE table (prior vs identified) on the held-out data and
           per-joint plots — so you can see whether identification actually
           *generalises* (does it still beat the URDF prior on a trajectory it
           never saw?), as opposed to merely memorising the training data.

    ``plot=False`` prints only the RMSE table (no figures).
    """
    from identification.fourier_trajectory import FourierTrajectory
    from identification.target_limb_regressor import TargetLimbRegressor

    # The validation regressor must belong to the identified limb (same dof).
    # If not given, it is read from the validation yaml's _meta.group.
    if limb_group is None:
        limb_group = FourierTrajectory.load_group(val_yaml, default="left_arm")

    dof = len(result.joint_order)
    if joint_names is None:
        joint_names = [f"joint_{d}" for d in range(dof)]

    # --- 1) Regressor: same prior (URDF) model as identification ---
    reg = TargetLimbRegressor(
        urdf_path=Path(urdf_path),
        group_to_identify=limb_group,
        print_info=False,
        gravity=gravity,
        waist_yaw_offset=waist_yaw_offset,
    )
    assert reg.dof == dof, (
        f"limb_group '{limb_group}' dof={reg.dof} != identification dof={dof} "
        f"— the validation yaml must belong to the same limb group"
    )

    # --- 2) Generate the held-out trajectory (same playback convention as the
    #        identification data, i.e. same sample_rate / time_coeffs). ---
    ft = FourierTrajectory(dim=dof, sample_rate=sample_rate, time_coeffs=time_coeffs)
    q_val, v_val, a_val = ft.generate_trajectory_from_yaml(Path(val_yaml).name)
    N = q_val.shape[1]
    if verbose:
        print(
            f"  [sim validation] held-out yaml={Path(val_yaml).name}  "
            f"N={N} steps  (time_coeffs={time_coeffs}, "
            f"period={ft.duration:.3f}s)"
        )

    # --- 3) Stack regressor + reference true torque on the new trajectory.
    #        Y depends only on kinematics/geometry, so the true torque is
    #        Y_val @ pi_true (pi_true from the true URDF) — same as the
    #        training-data preparation. ---
    Y_val = _stack_regressor(reg, q_val.T, v_val.T, a_val.T)  # (N*dof, 13*dof)
    tau_true = Y_val @ np.asarray(pi_true)
    if verbose:
        print(f"  [sim validation] Y_val: {Y_val.shape}, tau_true: {tau_true.shape}")

    # --- 4) RMSE table on the held-out data (prior vs identified) ---
    stats = print_rmse_comparison(result, joint_names, Y_val, tau_true)

    # --- 5) Optional plot ---
    figures = []
    if plot:
        figures = _plot_torque_comparison_panels(
            tau_true,
            Y_val @ result.pi_prior,
            Y_val @ result.pi_identified,
            joint_names,
            result.joint_order,
            dof,
            sample_rate=sample_rate,
            offset=offset,
            show_residual=show_residual,
            true_label="true",
            title=(
                "Joint Torque Comparison (sim held-out validation, different yaml): "
                "true vs prior vs identified\n"
                f"held-out yaml={Path(val_yaml).name}   "
                "(identified on a different training yaml)"
            ),
            twin=twin,
        )
    return {"stats": stats, "figures": figures}


# ============================================================================
# Static-pose gravity test (--static-test, sim only)
# ============================================================================
def sample_static_poses(
    reg,
    pi_true: np.ndarray,
    n_poses: int = STATIC_TEST_POSES,
    seed: int = STATIC_TEST_SEED,
    max_tries: int = 50,
    verbose: bool = True,
) -> dict:
    """Sample static poses that are collision-free and within the joint limits.

    A *static pose* is a joint configuration with ``v = a = 0``, so the only
    joint torque left is the gravity term.  Poses are drawn uniformly inside the
    **effective** position limits (``q_margin`` already applied) through
    ``TargetLimbRegressor.sample_state`` — the position limits therefore hold by
    construction — and then kept only if, using the regressor's own machinery:

    * ``collided == False`` (self-collision pairs + table pairs for this limb),
    * ``q_excess == 0`` (belt and braces against the effective limits),
    * ``|tau_gravity,i| <= effortLimit_i`` for every identified joint: a pose the
      joint could not hold statically is not a usable test pose.

    The global numpy RNG is reseeded with ``seed`` (mirroring
    ``target_limb_regressor``, which also samples through the global RNG), so a
    given seed always reproduces the same poses.

    Returns
    -------
    dict with ``q`` (n, dof), ``Y`` (n*dof, 13*dof) — the stacked joint-major
    regressors, so any parameter vector can be evaluated — ``tau_true``
    (n, dof), ``dof`` and the rejection counts.
    """
    dof = reg.dof
    np.random.seed(int(seed))
    q_poses: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    tau_list: list[np.ndarray] = []
    n_try = n_rej_coll = n_rej_tau = 0
    pi_true = np.asarray(pi_true, dtype=float).reshape(-1)

    while len(q_poses) < n_poses and n_try < n_poses * max_tries:
        n_try += 1
        q_full, _, _ = reg.sample_state([])  # no v/a requested => static pose
        q_g = np.asarray([q_full[i] for i in reg.group_to_identify], dtype=float)
        res = reg.compute_regressor(q_g, np.zeros(dof), np.zeros(dof), print_info=False)
        if bool(res[18]) or float(res[12]) > 0.0:  # collision / outside limits
            n_rej_coll += 1
            continue
        Y_g = _reorder_y_aug(res[0], dof)  # (dof, 13*dof)
        tau_g = Y_g @ pi_true
        if any(abs(tau_g[i]) > reg.tau_limit[i] for i in range(dof)):
            n_rej_tau += 1
            continue
        q_poses.append(q_g)
        Y_list.append(Y_g)
        tau_list.append(tau_g)

    n_found = len(q_poses)
    if n_found == 0:
        raise RuntimeError(
            f"no valid static pose found in {n_try} tries "
            f"(rejected: {n_rej_coll} collision/limits, {n_rej_tau} torque limit)"
        )
    if verbose:
        note = (
            "" if n_found >= n_poses else f"  WARNING: fewer than requested ({n_poses})"
        )
        print(
            f"  [static] {n_found} static poses, seed={seed}, {n_try} tries "
            f"(rejected: {n_rej_coll} collision/limits, {n_rej_tau} torque limit)."
            f"{note}"
        )

    return {
        "q": np.asarray(q_poses),
        "Y": np.vstack(Y_list),  # (n*dof, 13*dof), row = pose k, joint d
        "tau_true": np.asarray(tau_list),
        "dof": dof,
        "n_tries": n_try,
        "n_rejected_collision": n_rej_coll,
        "n_rejected_torque": n_rej_tau,
    }


def plot_static_pose_errors(
    err: np.ndarray,
    err_prior: np.ndarray | None,
    tau_true: np.ndarray | None,
    joint_names: list[str],
    joint_order: list[int],
    title: str = "Static-pose gravity test",
):
    """Diverging scatter of the static-pose torque error.

    x = one column per static pose (left → right), y = ``tau − tau_true`` with
    **0 exactly in the vertical middle** (the y limits are forced symmetric about
    zero, covering *both* series), one marker per joint —
    **filled = identified, hollow = prior (URDF)** — plus a thin vertical whisker
    joining the two at each pose.  The joints are **fanned out horizontally**
    inside each pose column (identified and prior for the same joint share the
    same x offset, so the whisker stays vertical) instead of being stacked on one
    line, and the pose columns are separated by **alternating white / light-grey
    background bands**.  A marker on the centre line means that model reproduces
    the true gravity torque for that joint at that pose; above / below means
    over- / under-estimation.  The whisker length is the change the
    identification made, and "shorter and closer to the centre line" = better.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_poses = err.shape[0]
    x = np.arange(1, n_poses + 1).astype(float)
    # Symmetric limits around 0 (covering BOTH series) → centre is exactly y = 0.
    spread = [float(np.max(np.abs(err)))]
    if err_prior is not None:
        spread.append(float(np.max(np.abs(err_prior))))
    ymax = 1.15 * max(max(spread), 1e-9)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, 10))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    # 同一姿势的多个关节左右叉开，避免 N 个点叠在一条竖线上；同一关节的
    # identified / prior 共用同一偏移，所以对比竖线仍然是竖直的。
    n_j = len(joint_order)
    half = 0.34 if n_j > 1 else 0.0
    off_map = {
        d: float(off)
        for d, off in zip(sorted(joint_order), np.linspace(-half, half, n_j))
    }

    fig, ax = plt.subplots(figsize=(max(9.5, 0.55 * n_poses + 4.5), 6.2))
    for j, d in enumerate(joint_order):
        name = joint_names[d] if joint_names else f"joint_{d}"
        color = colors[j % 10]
        marker = markers[j % len(markers)]
        xj = x + off_map[d]  # <- horizontal fan inside the pose column
        rms = float(np.sqrt(np.mean(err[:, j] ** 2)))
        t_rms = (
            float(np.sqrt(np.mean(tau_true[:, j] ** 2)))
            if tau_true is not None
            else float("nan")
        )
        if err_prior is not None:
            ax.vlines(
                xj,
                err_prior[:, j],
                err[:, j],
                color=color,
                linewidth=0.9,
                alpha=0.35,
                zorder=1,
            )
            ax.scatter(
                xj,
                err_prior[:, j],
                s=46,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=1.3,
                alpha=0.9,
                zorder=2,
            )
            rms_p = float(np.sqrt(np.mean(err_prior[:, j] ** 2)))
            label = (
                f"{name}: ident RMS {rms:.3g} / prior {rms_p:.3g} Nm (true {t_rms:.3g})"
            )
        else:
            label = f"{name}: err RMS {rms:.3g} Nm  (true RMS {t_rms:.3g})"
        ax.scatter(
            xj,
            err[:, j],
            s=68,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=1.4, zorder=2)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(0.35, n_poses + 0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("Static pose #   (each pose's joints are fanned out horizontally)")
    ax.set_ylabel(
        "identified / prior − true   [Nm]"
        if err_prior is not None
        else "identified − true   [Nm]"
    )
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # 交替白 / 浅灰背景带，把每个姿势的列分隔开（带边界正好落在姿势之间）。
    for i in range(1, n_poses + 1):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="0.93", zorder=0, linewidth=0)

    handles, labels = ax.get_legend_handles_labels()
    if err_prior is not None:
        gray = "0.35"
        handles += [
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=9,
                color=gray,
                label="filled = identified",
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor=gray,
                markeredgewidth=1.3,
                label="hollow = prior (URDF)",
            ),
        ]
    ax.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig


def run_static_pose_test(
    reg,
    pi_true: np.ndarray,
    pi_identified: np.ndarray,
    joint_names: list[str],
    joint_order: list[int],
    pi_prior: np.ndarray | None = None,
    n_poses: int = STATIC_TEST_POSES,
    seed: int = STATIC_TEST_SEED,
    plot: bool = True,
    verbose: bool = True,
):
    """Static-pose gravity test: identified / prior vs true joint torques.

    All torques use the **same** regressor (the prior URDF kinematics — exactly
    how the training torque was synthesised), so the differences isolate the
    *parameter* error:

        tau_true  = Y_static @ pi_true
        tau_ident = Y_static @ pi_identified
        tau_prior = Y_static @ pi_prior          (if pi_prior is given)
        err       = tau − tau_true   [Nm]

    Every pose has ``v = a = 0``, so only the gravity term is exercised: this
    checks the mass / CoM / inertia part of the model, which armature / damping
    / friction cannot mask.  With ``pi_prior`` the plot/table also show how much
    the identification improved the static gravity prediction over the raw URDF.
    Returns ``(err, err_prior_or_None, fig_or_None)``.
    """
    dof = reg.dof
    sampled = sample_static_poses(
        reg, pi_true, n_poses=n_poses, seed=seed, verbose=verbose
    )
    Y_static = sampled["Y"]  # (n*dof, 13*dof)
    tau_true = sampled["tau_true"]  # (n, dof)
    n = int(tau_true.shape[0])
    tau_ident = (Y_static @ np.asarray(pi_identified, dtype=float).reshape(-1)).reshape(
        n, dof
    )
    err = tau_ident - tau_true
    err_prior = None
    if pi_prior is not None:
        tau_prior = (Y_static @ np.asarray(pi_prior, dtype=float).reshape(-1)).reshape(
            n, dof
        )
        err_prior = tau_prior - tau_true

    print(
        f"\nStatic-pose gravity test — {n} poses, seed={seed}  "
        f"(err = model − tau_true;  improve = 1 − ident RMS / prior RMS):"
    )
    if err_prior is None:
        hdr = (
            f"{'Joint':<22s} {'errRMS':>9s} {'errMax':>9s} {'errMean':>9s} "
            f"{'trueRMS':>9s} {'err/true':>8s}"
        )
    else:
        hdr = (
            f"{'Joint':<22s} {'identRMS':>9s} {'identMax':>9s} {'priorRMS':>9s} "
            f"{'priorMax':>9s} {'impr':>8s} {'trueRMS':>8s}"
        )
    print(hdr)
    print("-" * len(hdr))
    for j, d in enumerate(joint_order):
        name = joint_names[d] if joint_names else f"joint_{d}"
        rms = float(np.sqrt(np.mean(err[:, j] ** 2)))
        rms_max = float(np.abs(err[:, j]).max())
        t_rms = float(np.sqrt(np.mean(tau_true[:, j] ** 2)))
        if err_prior is None:
            rel = rms / t_rms * 100 if t_rms > 1e-12 else float("nan")
            print(
                f"{name:<22s} {rms:>9.4f} {rms_max:>9.4f} "
                f"{float(err[:, j].mean()):>9.4f} {t_rms:>9.4f} {rel:>7.1f}%"
            )
        else:
            rms_p = float(np.sqrt(np.mean(err_prior[:, j] ** 2)))
            imp = (1 - rms / rms_p) * 100 if rms_p > 1e-12 else float("nan")
            print(
                f"{name:<22s} {rms:>9.4f} {rms_max:>9.4f} {rms_p:>9.4f} "
                f"{float(np.abs(err_prior[:, j]).max()):>9.4f} {imp:>7.1f}% "
                f"{t_rms:>8.4f}"
            )
    print("-" * len(hdr))
    rms_all = float(np.sqrt(np.mean(err**2)))
    rms_all_p = float(np.sqrt(np.mean(err_prior**2))) if err_prior is not None else None
    if rms_all_p is None:
        print(f"{'ALL':<22s} {rms_all:>9.4f}")
    else:
        imp = (1 - rms_all / rms_all_p) * 100 if rms_all_p > 1e-12 else float("nan")
        print(
            f"{'ALL':<22s} {rms_all:>9.4f} {'':>9s} {rms_all_p:>9.4f} "
            f"{'':>9s} {imp:>7.1f}%"
        )

    if not plot:
        return err, err_prior, None
    fig = plot_static_pose_errors(
        err,
        err_prior,
        tau_true,
        joint_names,
        joint_order,
        title=(
            "Static-pose gravity test — model − true joint torque\n"
            f"{n} random collision-free static poses (seed={seed})  |  "
            f"RMS: identified {rms_all:.4f} Nm"
            + (f" vs prior {rms_all_p:.4f} Nm" if rms_all_p is not None else "")
        ),
    )
    return err, err_prior, fig


# ============================================================================
# URDF export — write the identified parameters back out as a URDF
# ============================================================================
# Per-joint parameter layout (Pinocchio ``toDynamicParameters()`` order):
#   [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, arm, damp, fric]
# The 6 inertia entries are the inertia **about the joint/link frame origin**
# (I_O).  A URDF ``<inertial>`` stores mass + CoM + inertia **about the CoM**
# (I_C), so the parallel-axis theorem is applied in reverse:
#
#   I_C = I_O − m·(‖c‖²·I₃ − c·cᵀ) ,   c = mc / m
#
# (verified numerically against ``pin.Inertia.FromDynamicParameters`` and
# against the URDF values read by ``pin.buildModelFromUrdf`` — round-trip
# exact to ~1e-19).
#
# ``pin.Inertia`` for joint ``j`` == the URDF inertial of joint j's *child*
# link (Pinocchio's URDF convention: the joint frame IS the child link frame),
# and the URDF inertia component order is
#   [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
# i.e. the same physical tensor, just laid out differently from π.
URDF_NUM_DECIMALS = 8


def _fmt_urdf(v: float) -> str:
    """URDF-style decimal formatting (fixed decimals, no exponent)."""
    return f"{float(v):.{URDF_NUM_DECIMALS}f}"


def dynamic_params_to_inertial(
    pi_joint: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One joint's 13 params → ``(mass, com, I_about_com)``.

    ``pi_joint`` = [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, arm,
    damp, fric] (Pinocchio ordering; the 6 inertia entries are about the
    frame origin).  Returns the mass, the CoM ``c = mc/m`` and the inertia
    tensor about the CoM — exactly what a URDF ``<inertial>`` needs.
    """
    p = np.asarray(pi_joint, dtype=float).reshape(-1)
    if p.size != N_PER_JOINT:
        raise ValueError(f"expected {N_PER_JOINT} params, got {p.size}")
    m = float(p[0])
    if m <= 0.0:
        raise ValueError(f"identified mass must be > 0 (got {m:.6g})")
    mc = p[1:4]
    # Pinocchio inertia layout: [Ixx, Ixy, Iyy, Ixz, Iyz, Izz] about origin
    I_O = np.array(
        [
            [p[4], p[5], p[7]],
            [p[5], p[6], p[8]],
            [p[7], p[8], p[9]],
        ]
    )
    com = mc / m
    I_C = I_O - m * (float(com @ com) * np.eye(3) - np.outer(com, com))
    I_C = 0.5 * (I_C + I_C.T)  # guard tiny numeric asymmetry
    return m, com, I_C


def _urdf_child_link_map(root: ET.Element) -> dict[str, str]:
    """joint name → child link name (URDF has one child link per joint)."""
    out: dict[str, str] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        child = joint.find("child")
        if name and child is not None:
            out[name] = child.attrib.get("link", "")
    return out


def _inertial_block_lines(indent: str, upd: dict) -> list[str]:
    """Render a fresh ``<inertial>`` block (used for links that had none)."""
    ine = upd["inertia"]
    return [
        f"{indent}<inertial>",
        f'{indent}    <origin xyz="{upd["xyz"]}" rpy="0 0 0"/>',
        f'{indent}    <mass value="{upd["mass"]}"/>',
        (
            f'{indent}    <inertia ixx="{ine["ixx"]}" ixy="{ine["ixy"]}" '
            f'ixz="{ine["ixz"]}" iyy="{ine["iyy"]}" iyz="{ine["iyz"]}" '
            f'izz="{ine["izz"]}"/>'
        ),
        f"{indent}</inertial>",
    ]


def _rewrite_urdf_text(
    text: str,
    link_updates: dict[str, dict],
    joint_updates: dict[str, dict],
) -> str:
    """Rewrite only the targeted ``<inertial>`` / ``<dynamics>`` values.

    Everything else (header comments, mesh paths, environment links, tag
    layout, ``/>`` style) is preserved verbatim — this file's layout keeps one
    element per line.  Target links that have no ``<inertial>`` at all get one
    inserted at the end of the ``<link>`` block.
    """
    lines = text.splitlines()
    out: list[str] = []
    cur_link: str | None = None
    cur_joint: str | None = None
    in_inertial = False
    link_had_inertial: set[str] = set()

    for line in lines:
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        lm = re.match(r'<link\s+name="([^"]+)"', stripped)
        if lm:
            cur_link, cur_joint = lm.group(1), None
        jm = re.match(r'<joint\s+name="([^"]+)"', stripped)
        if jm:
            cur_joint, cur_link = jm.group(1), None

        if stripped.startswith("<inertial"):
            in_inertial = True
            link_had_inertial.add(cur_link or "")

        upd = link_updates.get(cur_link) if cur_link else None
        if in_inertial and upd is not None:
            if stripped.startswith("<origin"):
                line = re.sub(r'xyz="[^"]*"', f'xyz="{upd["xyz"]}"', line, count=1)
            elif stripped.startswith("<mass"):
                line = re.sub(r'value="[^"]*"', f'value="{upd["mass"]}"', line, count=1)
            elif stripped.startswith("<inertia"):
                for key, val in upd["inertia"].items():
                    line = re.sub(rf'{key}="[^"]*"', f'{key}="{val}"', line, count=1)

        if stripped.startswith("</inertial>"):
            in_inertial = False

        jupd = joint_updates.get(cur_joint) if cur_joint else None
        if jupd is not None and stripped.startswith("<dynamics"):
            line = (
                f'{indent}<dynamics armature="{jupd["armature"]}" '
                f'damping="{jupd["damping"]}" friction="{jupd["friction"]}"/>'
            )

        if stripped == "</link>":
            # Link without <inertial> (e.g. a bare frame link) → add one.
            lupd = link_updates.get(cur_link) if cur_link else None
            if lupd is not None and cur_link not in link_had_inertial:
                out.extend(_inertial_block_lines(indent + "    ", lupd))
            cur_link = None
        elif stripped == "</joint>":
            cur_joint = None

        out.append(line)
    return "\n".join(out) + "\n"


def _provenance_comment(prov: dict) -> list[str]:
    """XML comment block recording where the identified URDF came from."""
    body = [f"  · {k}: {v}" for k, v in prov.items() if v not in (None, "")]
    if not body:
        return []
    header = "    identified model — generated by sdp_solver_alljoints_ridge.py"
    return ["    <!--", header, *body, "    -->"]


def _unique_path(path: Path) -> Path:
    """Append ``_1``, ``_2``, … until the path does not exist (uniqueness)."""
    if not path.exists():
        return path
    for k in range(1, 1000):
        cand = path.with_name(f"{path.stem}_{k}{path.suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"cannot find a free filename for {path}")


def write_identified_urdf(
    pi_full: np.ndarray,
    joint_names: list[str],
    prior_urdf: str | Path,
    out_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    provenance: dict | None = None,
    timestamp: str | None = None,
    verbose: bool = True,
) -> Path:
    """Write the identified parameters into a copy of the prior URDF.

    The identified link inertials (mass / CoM / inertia-about-CoM) and joint
    dynamics (armature / damping / friction) replace their prior values; every
    other element of the prior URDF is copied verbatim (mesh paths, limits,
    the simulation environment, comments).

    Parameters
    ----------
    pi_full : (dof*13,) ndarray
        Identified parameters, joint-major (same layout as ``data['pi_prior']``
        / ``IdentificationResult.pi_identified``).
    joint_names : list[str]
        URDF joint names, index-aligned with the joints in ``pi_full``
        (``data['joint_names']``).
    prior_urdf : str or Path
        The prior/initial URDF the identification started from.  Its
        directory is the default output directory ("保存回先验 URDF 路径").
    out_path : str or Path or None
        Explicit output file.  ``None`` (default) → ``<prior_stem>_<timestamp>
        .urdf`` inside ``out_dir``.
    out_dir : str or Path or None
        Output directory; ``None`` → the prior URDF's directory.
    provenance : dict or None
        Optional key/value pairs written as a comment block right after the
        ``<robot>`` tag (mode, trajectory yaml, ridge lambda, …).
    timestamp : str or None
        Timestamp suffix (``YYMMDD_HHMMSS``); ``None`` → now.

    Returns
    -------
    Path
        The written file.
    """
    prior = Path(prior_urdf).resolve()
    if not prior.is_file():
        raise FileNotFoundError(f"prior URDF not found: {prior}")
    text = prior.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    child_of = _urdf_child_link_map(root)

    pi_list = split_joint_params(np.asarray(pi_full, dtype=float))
    if len(joint_names) != len(pi_list):
        raise ValueError(
            f"joint_names has {len(joint_names)} entries but pi_full encodes "
            f"{len(pi_list)} joints"
        )

    link_updates: dict[str, dict] = {}
    joint_updates: dict[str, dict] = {}
    skipped: list[str] = []
    infeasible: list[str] = []
    for jname, pi_j in zip(joint_names, pi_list):
        ok, eig_min, _ = check_lmi_feasibility(pi_j)
        if not ok:
            infeasible.append(f"{jname} (min_eig={eig_min:.3g})")
        link = child_of.get(jname)
        if not link:
            skipped.append(jname)
            continue
        m, com, I_C = dynamic_params_to_inertial(pi_j)
        link_updates[link] = {
            "mass": _fmt_urdf(m),
            "xyz": " ".join(_fmt_urdf(x) for x in com),
            "inertia": {
                key: _fmt_urdf(val)
                for key, val in zip(
                    ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"),
                    (I_C[0, 0], I_C[0, 1], I_C[0, 2], I_C[1, 1], I_C[1, 2], I_C[2, 2]),
                )
            },
        }
        joint_updates[jname] = {
            "armature": _fmt_urdf(pi_j[10]),
            "damping": _fmt_urdf(pi_j[11]),
            "friction": _fmt_urdf(pi_j[12]),
        }

    new_text = _rewrite_urdf_text(text, link_updates, joint_updates)

    if provenance:
        lines = new_text.splitlines()
        insert_at = next(
            (k + 1 for k, ln in enumerate(lines) if re.match(r"<robot\b", ln.strip())),
            0,
        )
        lines[insert_at:insert_at] = _provenance_comment(provenance)
        new_text = "\n".join(lines) + "\n"

    # ---- output path: prior URDF directory + timestamped name ----
    if out_path is not None:
        dest = Path(out_path)
    else:
        ts = timestamp or datetime.now().strftime("%y%m%d_%H%M%S")
        directory = Path(out_dir) if out_dir is not None else prior.parent
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / f"{prior.stem}_{ts}{prior.suffix or '.urdf'}"
        dest = _unique_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(new_text, encoding="utf-8")

    # ---- self-check: re-parse and compare against the identified values ----
    max_err = 0.0
    if not skipped:
        chk = ET.parse(str(dest)).getroot()
        chk_links = {elem.attrib["name"]: elem for elem in chk.findall("link")}
        for link, upd in link_updates.items():
            iel = chk_links[link].find("inertial")
            max_err = max(
                max_err,
                abs(float(iel.find("mass").attrib["value"]) - float(upd["mass"])),
            )
            max_err = max(
                max_err,
                max(
                    abs(float(x) - float(y))
                    for x, y in zip(
                        iel.find("origin").attrib["xyz"].split(), upd["xyz"].split()
                    )
                ),
            )
            ine = iel.find("inertia")
            max_err = max(
                max_err,
                max(
                    abs(float(ine.attrib[k]) - float(v))
                    for k, v in upd["inertia"].items()
                ),
            )
    if verbose:
        print(f"  [urdf] wrote {dest}")
        print(
            f"  [urdf] updated {len(link_updates)} link inertials, "
            f"{len(joint_updates)} joint dynamics "
            f"(max format round-trip error {max_err:.2e})"
        )
        if skipped:
            print(f"  [urdf] WARNING: joints not found in URDF, skipped: {skipped}")
        if infeasible:
            print(
                "  [urdf] WARNING: non physically-consistent inertia (the LMI in "
                "the solver should have prevented this; MuJoCo may reject the "
                "file): " + ", ".join(infeasible)
            )
    return dest


# ============================================================================
# main / demo — argparse CLI
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Global SDP 参数辨识（所有关节一次联合辨识，不再分关节顺序辨识）："
            "默认用 bag 实测力矩（meas）；加 --sim 改用 URDF 合成力矩做仿真验证。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 模式 / 数据来源 ----
    ap.add_argument(
        "--sim",
        action="store_true",
        help="仿真模式：tau 由 URDF 合成（prepare_data_from_urdf），pi_true 已知；"
        "不加 --sim 即实测模式（data_from_measurement，默认）",
    )
    ap.add_argument(
        "--urdf",
        default=str(DEFAULT_URDF_PATH),
        help="先验/初始 URDF（regressor + pi_prior + subtree mask）",
    )
    ap.add_argument(
        "--urdf-true",
        default=str(TRUE_URDF_PATH),
        help="[sim] '真值' URDF，用于合成 tau_measured",
    )
    ap.add_argument(
        "--yaml",
        "-y",
        default=TRAJ_YAML_PATH,
        metavar="NAME",
        help="trajectory_coefficients/ 下的轨迹系数 YAML（bag/group 从该 YAML 读）；"
        "默认：meas=最新 recovered_*.yaml，sim=最新 pso_unified_*.yaml",
    )
    ap.add_argument(
        "--quality-yaml",
        "-q-y",
        default=QUALITY_YAML_PATH,
        metavar="NAME",
        help="带 _diagnostics.per_param 质量标签的 pso_unified YAML：每个参数的"
        "质量分档决定其岭回归（L2 拉回 prior）权重 —— 质量越好权重越小、允许偏离"
        "prior 越多；质量越差权重越大。实测的 recovered YAML 不带质量，须指向训练"
        "它的 pso_unified。默认：--yaml 本身带质量则用它；否则自动取与 --yaml 同"
        " _meta.group 的最新 pso_unified_*.yaml；找不到则全部用默认分档权重",
    )
    ap.add_argument(
        "--ridge-lambda",
        type=float,
        default=RIDGE_LAMBDA,
        help="岭回归全局强度：乘在每个质量分档权重上（0 = 纯数据拟合、不惩罚偏离"
        "prior；越大越把参数拉回 URDF prior）",
    )
    ap.add_argument(
        "--lmi-shape-weight",
        type=float,
        default=LMI_SHAPE_WEIGHT,
        metavar="W",
        help="软铰链形状惩罚权重：目标函数加 "
        "W·Σ_d max(0, 1 − λ_min(Σ_C,d)/(FRAC·λ_min(Σ_C,d^prior)))。"
        "物理一致性只剩数值下限 LMI_EPS（严格 PD），连杆形状完全靠这一项拉出来："
        "解会主动往锥内部挪，而不是贴在边界上被压扁。"
        "0 = 关闭（退回“贴在锥边界”的压扁行为）；≥0.3 就饱和，默认 1.0",
    )
    ap.add_argument(
        "--lmi-shape-frac",
        type=float,
        default=LMI_SHAPE_FRAC,
        metavar="FRAC",
        help="软铰链目标 = FRAC × 先验自身的三角不等式余量 λ_min(Σ_C^prior)。"
        "实测：0.3 几乎零代价（train 不变、val/静态略好）就能把最扁的 J15/J16/J17 "
        "拉回接近真值；0.5 也免费；1.0 会过冲（把 J16/J17 撑得比真值还胖）",
    )
    ap.add_argument(
        "--csv-topic",
        default="hardware_joint_state",
        help="[meas] <bag>/csv/ 下的 CSV 主题（实测力矩列）",
    )
    ap.add_argument(
        "--sample-rate",
        type=float,
        default=100.0,
        help="[meas] CSV(~500Hz) 抽取到 ~sample_rate Hz 作为回归采样率",
    )
    ap.add_argument(
        "--time-coeffs",
        "-t",
        type=float,
        default=1.0,
        help="[sim] 傅立叶轨迹回放时间倍率（>1 加速，<1 减速；默认 1.0 = "
        "正常速度，物理周期 = TRAJ_PERIOD/time_coeffs）",
    )
    ap.add_argument(
        "--tau-delay",
        type=float,
        default=TAU_DELAY,
        metavar="SECONDS",
        help="[meas] 实测力矩通道的时延补偿：认为 torque(t) 对应 t−τ 时刻的状态，"
        "把回归器在 (t−τ) 上求值。>0 = 力矩滞后于状态（传输/滤波延迟的典型情形），"
        "负值 = 力矩超前。时移会把 armature 列变成 "
        "cos(ωτ)·q̈ + ω·sin(ωτ)·q̇，即在 armature 与 damping 之间转移权重。"
        "默认 0 = 不补偿",
    )

    # ---- 模型 / 环境 ----
    ap.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="机体系重力向量（[meas] 默认取 bag IMU 读数的负均值；[sim] 默认按"
        " MuJoCo 躺姿模型取 (-9.81, 0, 0)）",
    )
    ap.add_argument(
        "--waist-offset",
        type=float,
        default=None,
        help="腰关节 J12_WAIST_YAW 固定角度 (rad)（[meas] 默认取 bag "
        "joint_state position_12 的均值；[sim] 默认按躺姿模型取 0.0）",
    )

    # ---- 验证 / 绘图 ----
    ap.add_argument(
        "--val-yaml",
        "-v-y",
        default=None,
        nargs="+",
        metavar="NAME",
        help="交叉验证用轨迹 YAML（可以给**多个**，空格分隔；每一个都必须是不同于 "
        "--yaml 的轨迹）：\n"
        "  [sim]  辨识后逐条当作 held-out 轨迹：在每条上用同一 pi_true 重算真值"
        "tau，打印 prior vs identified 的 RMSE 改善，判断辨识是否泛化；多条时最后"
        "再汇总一张跨 yaml 总表。不给 → 默认不做交叉对比\n"
        "  [meas] 在每个 yaml 的 _meta.source_bag 对应 bag 的实测数据上分别验证"
        "（各自用**自己那个 bag** 的 IMU 重力/腰角）；不给则默认用文件顶部"
        " VAL_YAML_PATH 常量",
    )
    ap.add_argument(
        "-w",
        "--twin",
        default=None,
        metavar="START:END",
        help="[plot] 对比图时间窗缩放，如 '0:13.4'（任一端可省略；None=全程）",
    )
    ap.add_argument(
        "--twin-id",
        default=None,
        metavar="START:END",
        help="辨识用时间窗：手动选一段采样质量好的时间段做辨识（sim 与 meas 通用；"
        "轨迹是周期的，选一个质量好的周期即可，如 '10:16.667'；None=全程）",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="跳过力矩对比图",
    )
    ap.add_argument(
        "--static-test",
        "-static",
        action="store_true",
        help="[仅 --sim] 额外的静态姿势重力测试：用给定种子随机采样 N 个不碰撞、"
        "不超限的静态姿势（v=a=0 ⇒ 只剩重力项），比较真值 URDF 与辨识参数算出的"
        "关节力矩之差，并画成以 0 为纵轴中心的散点图（每列一个姿势、每个关节一个点）",
    )
    ap.add_argument(
        "--static-poses",
        type=int,
        default=STATIC_TEST_POSES,
        metavar="N",
        help="[--static-test] 随机静态姿势数量",
    )
    ap.add_argument(
        "--static-seed",
        type=int,
        default=STATIC_TEST_SEED,
        metavar="SEED",
        help="[--static-test] 姿势采样随机种子（固定则每次姿态相同）",
    )

    # ---- 结果导出 ----
    ap.add_argument(
        "--no-save-urdf",
        "-no",
        action="store_true",
        help="不把辨识结果写成 URDF（默认会写）",
    )
    ap.add_argument(
        "--urdf-out-dir",
        default=None,
        metavar="DIR",
        help="辨识结果 URDF 的输出目录；默认为 --urdf 所在目录（即先验 URDF 路径），"
        "文件名 = <先验文件名去后缀>_<YYMMDD_HHMMSS>.urdf（重名自动加 _1/_2…）",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from identification.fourier_trajectory import FourierTrajectory

    is_sim = args.sim
    if args.static_test and not is_sim:
        raise SystemExit(
            "--static-test 只在 --sim 模式下可用（meas 模式没有 pi_true，"
            "无法算“实际 URDF 的关节力矩”）"
        )

    # 结果 URDF 的时间戳（整次运行共用一个，保证同步；重名时再追加 _1/_2…）。
    urdf_ts = datetime.now().strftime("%y%m%d_%H%M%S")

    # ---- 解析默认 YAML（bag / group 跟着它走） ----
    if is_sim:
        traj_yaml = args.yaml or FourierTrajectory.find_latest_yaml()  # pso_unified
    else:
        traj_yaml = args.yaml or _latest_recovered_yaml()

    # ---- 质量边界 YAML：recovered（实测）不带 _diagnostics，须指向训练它/同一
    #     肢体组的 pso_unified YAML（可用 --quality-yaml 显式指定） ----
    if args.quality_yaml:
        quality_yaml = args.quality_yaml
    elif _yaml_has_quality(traj_yaml):
        quality_yaml = traj_yaml  # 轨迹 YAML 本身带质量标签（如 pso_unified）
    else:
        group = FourierTrajectory.load_meta(traj_yaml).get("group") or "left_arm"
        quality_yaml = _latest_pso_yaml_for_group(group, exclude=traj_yaml)
        if quality_yaml:
            print(
                f"  [quality] 自动关联 pso_unified 质量 YAML: "
                f"{quality_yaml} (group={group})"
            )
        else:
            print(
                f"  [quality] 未找到 group='{group}' 的 pso_unified_*.yaml"
                " → 全部参数用默认分档权重"
            )

    # ---- 重力 / 腰关节偏置 ----
    #   meas：辨识 bag 由 traj_yaml 的 _meta.source_bag 决定，直接读它的读数求平均
    #         （gravity = -mean(IMU 线加速度)，waist = mean(position_12)）。
    #   sim ：pso_unified 轨迹是仿真激励、没有 source_bag，读不到 bag → 用躺姿
    #         默认 SIM_GRAVITY（MuJoCo 里 LINK_BASE 被 -90° 绕 Y 旋转躺下，世界
    #         重力 (0,0,-9.81) → 机体系 ≈ (-9.81, 0, 0)）与 SIM_WAIST_YAW_OFFSET
    #         （躺姿模型 J12 停在零位 → 0.0）。
    #   --gravity / --waist-offset 显式给定时覆盖（两者可单独覆盖）。
    gravity: np.ndarray | None = None
    waist_yaw: float | None = None
    setup_src = (
        f"sim 躺姿默认 (gravity={SIM_GRAVITY.tolist()}, "
        f"waist={SIM_WAIST_YAW_OFFSET:.4f} rad)"
    )
    if is_sim:
        gravity = SIM_GRAVITY.copy()
        waist_yaw = SIM_WAIST_YAW_OFFSET
    else:
        gravity, waist_yaw = read_setup_from_bag(_yaml_source_bag(traj_yaml))
        setup_src = "bag average"
    overrides = []
    if args.gravity is not None:
        gravity = np.asarray(args.gravity, dtype=float)
        overrides.append("--gravity")
        print(f"  [setup] --gravity 覆盖: {np.round(gravity, 6).tolist()}")
    if args.waist_offset is not None:
        waist_yaw = float(args.waist_offset)
        overrides.append("--waist-offset")
        print(f"  [setup] --waist-offset 覆盖: {waist_yaw:.9f} rad")
    if overrides:
        setup_src += " + " + " ".join(overrides)

    # ---- 配置回显 ----
    print("\n" + "=" * 100)
    print("SDP IDENTIFICATION (weighted ridge)".center(100))
    print("=" * 100)
    print(f"mode            : {'sim' if is_sim else 'meas'}")
    print(f"urdf (prior)    : {args.urdf}")
    if is_sim:
        print(f"urdf (true)     : {args.urdf_true}")
        print(f"time_coeffs     : {args.time_coeffs}")
    print(f"trajectory yaml : {traj_yaml}")
    if args.quality_yaml:
        print(f"quality yaml    : {quality_yaml}  (--quality-yaml)")
    else:
        print(f"quality yaml    : {quality_yaml or '(none → default quality weights)'}")
    print(f"bag             : (auto: 从 {traj_yaml} _meta.source_bag)")
    print(f"group           : (auto: 从 {traj_yaml} _meta.group)")
    print(f"sample_rate     : {args.sample_rate}")
    print(f"ridge_lambda    : {args.ridge_lambda}")
    print(f"tau_delay       : {args.tau_delay * 1e3:+.2f} ms")
    print(f"twin-id         : {args.twin_id or '(Global)'}")
    if args.static_test:
        print(
            f"static test     : {args.static_poses} poses, seed={args.static_seed} "
            f"(sim only)"
        )
    if args.no_save_urdf:
        save_urdf_desc = "(disabled: --no-save-urdf)"
    else:
        out_dir = (
            Path(args.urdf_out_dir)
            if args.urdf_out_dir
            else Path(args.urdf).resolve().parent
        )
        save_urdf_desc = f"{out_dir}/{Path(args.urdf).stem}_{urdf_ts}.urdf"
    print(f"save urdf       : {save_urdf_desc}")
    print(
        "gravity         : "
        + (
            f"{np.round(gravity, 6).tolist()}  |g|={np.linalg.norm(gravity):.6f} m/s²"
            if gravity is not None
            else "(未指定 → TargetLimbRegressor 默认 (0, 0, -9.81) 正立)"
        )
        + f"   [{setup_src}]"
    )
    print(
        "waist_yaw_offset: "
        + (
            f"{waist_yaw:.9f} rad"
            if waist_yaw is not None
            else "(未指定 → 用默认 0.0 rad)"
        )
    )

    # ---- 交叉验证轨迹 YAML（--val-yaml，可多条）----
    #   meas：在另一条轨迹/bag 上做 held-out 验证（不给时默认 VAL_YAML_PATH 常量）。
    #   sim：只有显式给了 --val-yaml 才在辨识后做 held-out 交叉对比；不给 = 不做。
    if is_sim:
        val_yamls: list[str] = list(args.val_yaml or [])  # 空 → 默认不交叉对比
    else:
        val_yamls = list(args.val_yaml or [str(VAL_YAML_PATH)])

    # 1. Prepare data
    # ------------------------------------------------------------------
    # sim：tau = Y @ pi_true（由 URDF 合成），pi_true 已知 → 可做误差/收敛对比。
    # meas：tau 直接取自 bag 的 CSV（torque_<limb> 列），回归状态 q/v/a 全部来自
    #       恢复的 Fourier 轨迹（在 CSV 时间上直接求值）；加速度不用实测速度差分
    #       （CSV 无加速度列，量化噪声大）。测量模式无地面真值 → pi_true 为 None。
    if is_sim:
        data = prepare_data_from_urdf(
            urdf_path=args.urdf,  # 先验模型 → regressor / pi_prior / bounds
            yaml_filename=traj_yaml,
            limb_group=None,  # 从 YAML _meta.group 读取
            sample_rate=args.sample_rate,
            time_coeffs=args.time_coeffs,  # 傅立叶轨迹回放倍率
            urdf_true_path=args.urdf_true,
            gravity=gravity,
            waist_yaw_offset=waist_yaw,
            twin=args.twin_id,  # 辨识用时间窗（sim/meas 通用，None=全程）
        )
    else:
        data = data_from_measurement(
            urdf_path=args.urdf,  # 先验模型 → regressor / pi_prior / bounds
            bag_name=None,  # 从 YAML _meta.source_bag 读取
            limb_group=None,  # 从 YAML _meta.group 读取
            csv_topic=args.csv_topic,  # 实测力矩
            sample_rate=args.sample_rate,  # CSV(~500Hz) 抽取到 ~sample_rate Hz
            gravity=gravity,
            waist_yaw_offset=waist_yaw,
            trajectory_yaml=traj_yaml,
            twin=args.twin_id,  # 辨识用时间窗（选一个质量好的周期，None=全程）
            tau_delay=args.tau_delay,  # 力矩通道时延补偿（秒，--tau-delay）
        )

    # 2. Configure weighted-ridge regularization.
    #    每个参数按质量分档获得 L2 惩罚系数（质量好 → 系数小 → 允许偏离 URDF
    #    prior 更多；质量差 → 系数大 → 更强地拉回 prior），由 build_ridge_weights
    #    从质量标签生成。连杆形状另由 2c 的软铰链约束（见 LMI_SHAPE_FRAC），
    #    LMI 自身只留数值下限 LMI_EPS。
    quality_map = None
    if quality_yaml is not None:
        quality_path = FourierTrajectory._coeffs_dir / quality_yaml
        if quality_path.is_file():
            quality_map = load_yaml_param_quality(quality_path)
    freeze_mask = build_freeze_mask(quality_map, dof=int(data["dof"]))
    # freeze_mask 是**唯一来源**：被冻结的参数同时退出岭范数（c = 0）。
    # 反之 c = 0 而未冻结 = 参数既无惩罚又可自由移动，求解器会直接报错拦下。
    ridge_weights = build_ridge_weights(
        quality_map,
        data["pi_prior"],
        ridge_lambda=args.ridge_lambda,
        freeze_mask=freeze_mask,
    )
    from collections import Counter

    dof_ = int(data["dof"])
    n_freeze = int(freeze_mask.sum())
    n_dyn = dof_ * len(RIDGE_FREEZE_LOCAL_INDICES)
    n_qual = int(
        build_freeze_mask(quality_map, dof=dof_, freeze_local_indices=frozenset()).sum()
    )
    print(
        f"  hard-freeze: {n_freeze}/{dof_ * N_PER_JOINT} params → pi == prior, "
        f"EXCLUDED from the ridge (c = 0)  [quality null/small: {n_qual}; "
        f"armature/damping/friction: {n_dyn}; overlap "
        f"{n_qual + n_dyn - n_freeze}]"
    )
    if quality_map:
        qc = Counter(quality_map.values())
        print(f"  YAML quality distribution: {dict(qc)}")
        print(
            "  ridge per-quality weights (participating bands only): "
            f"{dict(RIDGE_QUALITY_WEIGHTS)}"
        )
    else:
        print(
            "  无 _diagnostics.per_param 质量标签 → 剩余参数全部按默认分档 "
            f"'{DEFAULT_QUALITY}' 加岭惩罚"
        )
    c_eff = ridge_weights.copy()
    c_eff[freeze_mask] = np.nan
    print(
        f"  ridge_lambda={args.ridge_lambda}: active c range "
        f"[{np.nanmin(c_eff):.4g}, {np.nanmax(c_eff):.4g}] "
        f"(hard-frozen: {n_freeze})"
    )

    # 2b. 物理一致性 LMI 只剩数值下限 LMI_EPS（严格 PD）：形状不再用硬余量约束，
    #     而是全部交给 2c 的软铰链（见 LMI_SHAPE_FRAC 的实测表）。
    print(
        f"  LMI floor: uniform eps = {LMI_EPS:.1g} (strict PD only; "
        f"link shape comes from the soft hinge below)"
    )

    # 2c. 软铰链形状惩罚：把硬余量的“墙”换成“拉力”。参考值 = FRAC × 先验自身的
    #     三角不等式余量 λ_min(Σ_C^prior)；W=0 时完全不进入问题（退回“只有 LMI_EPS
    #     下限”的旧行为：解贴在锥边界、连杆被压扁）。
    if args.lmi_shape_frac is not None and args.lmi_shape_frac <= 0:
        raise SystemExit("--lmi-shape-frac must be > 0 (it is a dividend)")
    if args.lmi_shape_weight is None or args.lmi_shape_weight < 0:
        raise SystemExit("--lmi-shape-weight must be >= 0")
    shape_weight = float(args.lmi_shape_weight or 0.0)
    shape_ref = build_shape_reference(data["pi_prior"], dof_, frac=args.lmi_shape_frac)
    if shape_weight > 0:
        pri_slack = shape_ref / float(args.lmi_shape_frac)
        print(
            f"  shape hinge ON: W={shape_weight:g}, target = {args.lmi_shape_frac:g}"
            f"×prior slack → [{shape_ref.min():.4g}, {shape_ref.max():.4g}] "
            f"(prior slack [{pri_slack.min():.4g}, {pri_slack.max():.4g}])"
        )
    else:
        print(
            f"  shape hinge OFF (W=0) → 解会贴在 LMI 锥边界、连杆被压扁；"
            f"target would be {args.lmi_shape_frac:g}×prior slack = "
            f"[{shape_ref.min():.4g}, {shape_ref.max():.4g}] (--lmi-shape-weight 1)"
        )

    # 3. Solve (weighted ridge + null/small hard-freeze)
    solver = SDPSolver(solver_name="MOSEK", verbose=True)
    result = solver.solve(
        Y_stack=data["Y_stack"],
        tau_measured=data["tau_measured"],
        pi_prior=data["pi_prior"],
        joint_order=data["joint_order"],
        ridge_weights=ridge_weights,
        freeze_mask=freeze_mask,
        shape_ref=shape_ref,
        shape_weight=shape_weight,
        joint_names=data["joint_names"],
    )

    # 4. Report
    solver.print_results(
        result,
        joint_names=data["joint_names"],
        pi_reference=data["pi_true"],
        quality_map=quality_map,
        Y_stack=data["Y_stack"],
        tau_measured=data["tau_measured"],
        is_sim=is_sim,
    )

    # 4b. 对辨识后的惯性参数做物理一致性（pseudo-inertia LMI）检查并打印。
    print("\n" + "=" * 100)
    print("CROSS-VALIDATION RESULTS".center(100))
    print("=" * 100)
    print_lmi_feasibility(
        result.pi_identified,
        data["joint_names"],
        result.joint_order,
        "identified",
        shape_ref=shape_ref,
    )

    # 4c. 导出辨识结果 URDF —— 把辨识出的 link 惯性（质量/质心/绕质心惯量）与关节
    #     dynamics（armature/damping/friction）写回先验 URDF 的一份拷贝，其余部分
    #     （mesh、限位、环境 link、注释）原样保留。文件名加时间戳保证唯一：
    #       <先验文件名去后缀>_<YYMMDD_HHMMSS>.urdf
    #     默认落在先验 URDF 所在目录（可用 --urdf-out-dir 覆盖，--no-save-urdf 跳过）。
    if not args.no_save_urdf:
        print("\n" + "=" * 100)
        print("EXPORT IDENTIFIED URDF".center(100))
        print("=" * 100)
        write_identified_urdf(
            result.pi_identified,
            joint_names=data["joint_names"],
            prior_urdf=args.urdf,
            out_dir=args.urdf_out_dir,
            timestamp=urdf_ts,
            provenance={
                "source": f"{Path(args.urdf).name} (prior, from SDP identification)",
                "mode": "sim" if is_sim else "meas",
                "trajectory_yaml": traj_yaml,
                "quality_yaml": quality_yaml or "(none)",
                "ridge_lambda": args.ridge_lambda,
                "lmi_floor": f"uniform LMI_EPS={LMI_EPS:.1g} (strict PD only)",
                "lmi_shape_hinge": (
                    f"W={shape_weight:g}, target = {args.lmi_shape_frac:g}×prior "
                    f"lambda_min(Sigma_C) = [{shape_ref.min():.4g}, "
                    f"{shape_ref.max():.4g}]"
                    if shape_weight > 0
                    else "off (W=0)"
                ),
                "dof": f"{data['dof']} ({', '.join(data['joint_names'])})",
                "gravity": np.round(gravity, 6).tolist()
                if gravity is not None
                else "(default)",
                "waist_yaw_offset": waist_yaw,
                "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    # 4d. 静态姿势重力测试（--static-test，仅 sim）—— 随机采样不碰撞、不超限的静态
    #     姿势（v = a = 0 ⇒ 只剩重力项），比较“真值 URDF 的关节力矩”与“辨识参数算
    #     出的关节力矩”之差，画成以 0 为中心纵轴的散点图（每列一个姿势，每个关节
    #     在纵向上一个点；误差为 0 则停在最中间）。两者用**同一个回归器**（prior URDF
    #     的几何），与训练数据的合成方式一致 ⇒ 差异纯粹来自参数误差。
    if args.static_test:
        print("\n" + "=" * 100)
        print("STATIC-POSE GRAVITY TEST (--static-test)".center(100))
        print("=" * 100)
        run_static_pose_test(
            reg=data["reg"],
            pi_true=data["pi_true"],
            pi_identified=result.pi_identified,
            pi_prior=data["pi_prior"],  # 同期叠加 URDF 先验的误差作对比
            joint_names=data["joint_names"],
            joint_order=result.joint_order,
            n_poses=args.static_poses,
            seed=args.static_seed,
            plot=not args.no_plot,
        )

    # 5. Cross-validation / comparison plot
    #    sim：真值 vs prior vs identified 对比（训练轨迹上）。
    #    meas：用另一条轨迹/bag 验证辨识结果 —— pi_prior 与 pi_identified 分别在
    #          验证轨迹上算关节 tau，与验证 bag 的实测 tau 对比；验证 bag 也
    #          从验证 YAML 的 _meta.source_bag 读取。
    if not args.no_plot:
        if is_sim:
            plot_torque_comparison_simulated(
                result,
                joint_names=data["joint_names"],
                Y_stack=data["Y_stack"],
                pi_reference=data["pi_true"],
                sample_rate=args.sample_rate,
                twin=args.twin,
            )
        else:
            # 逐条验证轨迹：每条 yaml 的验证 bag 用它**自己** bag 读到的机体系重力/
            # 腰关节角度（不同录制场次的姿态不同，不能沿用辨识 bag 的值）；
            # --gravity / --waist-offset 显式覆盖时沿用用户给的值。
            cv_rows = []
            for i_cv, val_yaml in enumerate(val_yamls, 1):
                if len(val_yamls) > 1:
                    print("\n" + "-" * 100)
                    print(
                        f"[{i_cv}/{len(val_yamls)}] validation yaml: {val_yaml}".center(
                            100
                        )
                    )
                    print("-" * 100)
                note = "?"
                try:
                    note = _yaml_source_bag(val_yaml)
                    val_gravity, val_waist = gravity, waist_yaw
                    if args.gravity is None or args.waist_offset is None:
                        print(
                            "  [setup] validation bag (from val_yaml _meta.source_bag):"
                        )
                        g_val, w_val = read_setup_from_bag(note)
                        if args.gravity is None:
                            val_gravity = g_val
                        if args.waist_offset is None:
                            val_waist = w_val
                    out = plot_torque_comparison_measured(
                        result,
                        urdf_path=args.urdf,  # 先验模型（regressor）
                        val_yaml=val_yaml,
                        val_bag_name=None,  # 从 val YAML _meta.source_bag 读取
                        joint_names=data["joint_names"],
                        limb_group=None,
                        csv_topic=args.csv_topic,
                        sample_rate=args.sample_rate,
                        gravity=val_gravity,
                        waist_yaw_offset=val_waist,
                        tau_delay=args.tau_delay,  # 与辨识一致的时延补偿
                        twin=args.twin,  # 时间窗缩放，如 "0:13.4"
                    )
                    cv_rows.append(
                        {
                            "yaml": Path(val_yaml).name,
                            "note": note,
                            "stats": out["stats"],
                        }
                    )
                except Exception as exc:
                    # 一条坏 yaml（没有 source_bag / 不同 limb group / 缺 bag）不应把
                    # 整次多轨迹验证干掉 —— 报告并继续下一条。
                    print(f"  [CV] FAILED for {val_yaml}: {type(exc).__name__}: {exc}")
                    cv_rows.append(
                        {
                            "yaml": Path(val_yaml).name,
                            "note": note,
                            "stats": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            print_cv_summary(cv_rows, data["joint_names"], result.joint_order)

    # 5b. sim held-out cross-validation（可选）—— 只有显式给了 --val-yaml（一条或
    #     多条 *不同的* 轨迹）才做：用同一 pi_true 在每条上重算真值 tau，看
    #     identified 相对 prior 在新轨迹上是否仍有改善（Improve %）——即辨识是
    #     泛化还是只背下了训练数据。不给 --val-yaml = 默认不做。即使 --no-plot
    #     也打印 RMSE 表，最后汇总成一张跨 yaml 总表。
    if is_sim and val_yamls:
        print("\n" + "=" * 100)
        print("SIM HELD-OUT CROSS-VALIDATION (different trajectory yaml)".center(100))
        print("=" * 100)
        cv_rows = []
        for i_cv, val_yaml in enumerate(val_yamls, 1):
            if len(val_yamls) > 1:
                print("\n" + "-" * 100)
                print(
                    f"[{i_cv}/{len(val_yamls)}] held-out yaml: {val_yaml}".center(100)
                )
                print("-" * 100)
            note = "?"
            try:
                note = FourierTrajectory.load_group(val_yaml, default="?")
                out = plot_torque_comparison_simulated_validation(
                    result,
                    urdf_path=args.urdf,
                    val_yaml=val_yaml,
                    pi_true=data["pi_true"],
                    joint_names=data["joint_names"],
                    limb_group=None,  # 从 val YAML 的 _meta.group 读取（须与辨识同 dof）
                    sample_rate=args.sample_rate,
                    time_coeffs=args.time_coeffs,
                    gravity=gravity,
                    waist_yaw_offset=waist_yaw,
                    twin=args.twin,  # 时间窗缩放，如 "0:13.4"
                    plot=not args.no_plot,
                )
                cv_rows.append(
                    {"yaml": Path(val_yaml).name, "note": note, "stats": out["stats"]}
                )
            except Exception as exc:
                # 一条坏 yaml（不同 limb group / dof 不匹配）不应把整批验证干掉。
                print(f"  [CV] FAILED for {val_yaml}: {type(exc).__name__}: {exc}")
                cv_rows.append(
                    {
                        "yaml": Path(val_yaml).name,
                        "note": note,
                        "stats": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        print_cv_summary(cv_rows, data["joint_names"], result.joint_order)

    print("\n" + "=" * 100)
    print("SDP identification finished.".center(100))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
