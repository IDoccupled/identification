# \!/usr/bin/env python3
"""
Global SDP-based Parameter Identification for Serial Robot Limbs
(no per-joint sequential identification — all joints are identified together
in a single solve).

Regularization: quality-weighted RIDGE (L2 toward the URDF prior) instead of
the per-quality box constraints.  Good-quality parameters get a small penalty
for deviating from the prior; poor-quality ones get a large penalty.

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

Dependencies: cvxpy + MOSEK (or SCS), numpy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
# 相对偏离 r_i = (pi_i − prior_i)/scale_i 上的二次惩罚权重（按质量分档）：
#   质量越好（good/ok）→ 权重越小 → 允许参数偏离 URDF prior 更多；
#   质量越差（bad / rank_deficient / small / null）→ 权重越大 → 越强地把参数
#   拉回 prior。全局强度 RIDGE_LAMBDA 乘在每个分档权重上。
#
#   null / small 的“字面冻结”：这类参数量级太小（绝对偏离 ~1e-6..1e-5），仅靠
#   增大权重并不能做到“纹丝不动”。所以 RIDGE_FREEZE_QUALITIES 里的质量标签
#   （null/small）会由求解器以等式约束 pi == prior 直接冻结 —— 它们的方向本来
#   就没有被激励，理应留在 prior。冻结后其权重不再起作用（对应岭系数被清零），
#   表里的数值仅在“未启用冻结”时作为兜底。
RIDGE_QUALITY_WEIGHTS = {
    "good": 1.0,
    "ok": 3.0,
    "bad": 12.0,
    "rank_deficient": 100.0,
    "small": 1e5,
    "null": 1e5,
}
# 需要“字面冻结”（pi == prior，完全不动）的质量标签。
RIDGE_FREEZE_QUALITIES = frozenset({"null", "small"})
DEFAULT_QUALITY = "ok"  # 无质量标签参数的兜底分档
RIDGE_LAMBDA = 1.0  # 全局岭强度（乘在每个分档权重上，--ridge-lambda）
RIDGE_SCALE_FLOOR = 1e-4  # 相对尺度下限：scale_i = max(|prior_i|, floor)

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

VAL_YAML_PATH = (
    Path(__file__).resolve().parent.parent
    / "trajectory_coefficients"
    / "recovered_260817_142958.yaml"
).resolve()

QUALITY_YAML_PATH = (
    Path(__file__).resolve().parent.parent
    / "trajectory_coefficients"
    / "pso_unified_260803_180859.yaml"
).resolve()

# 机体系重力向量（IMU 读得）与腰关节固定偏置 J12_WAIST_YAW (rad)。
SETUP = {
    "gravity": np.array([-9.712746, 0.390467, -1.393897]),
    "waist_yaw_offset": 0.076485634,
}

# 6×6 pseudo-inertia 严格正定余量（J ≽ eps·I₆，min_eig ≥ eps > 0）。
LMI_EPS = 1e-7


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
# Physical-consistency LMI  — 6×6 pseudo-inertia (Jung et al. eq.22)
# ============================================================================
def _skew3(v) -> cp.Expression:
    if isinstance(v, np.ndarray):
        v = cp.Constant(v)
    return cp.bmat([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def build_pseudo_inertia_LMI(
    m: cp.Variable,
    mc: cp.Variable,
    I_vec: cp.Variable,
) -> cp.Expression:
    """6×6  [I, S(mc); S(mc)ᵀ, m·I₃] ≽ 0.  Pinocchio ordering."""
    I_mat = cp.bmat(
        [
            [I_vec[0], I_vec[1], I_vec[3]],  # Ixx Ixy Ixz
            [I_vec[1], I_vec[2], I_vec[4]],  # Ixy Iyy Iyz
            [I_vec[3], I_vec[4], I_vec[5]],  # Ixz Iyz Izz
        ]
    )
    S = _skew3(mc)
    return cp.bmat([[I_mat, S], [S.T, m * np.eye(3)]])


def check_lmi_feasibility(pi: np.ndarray) -> tuple[bool, float, np.ndarray]:
    m_val, mc_val = pi[0], pi[1:4]
    I_vals = pi[4:10]
    I_mat = np.array(
        [
            [I_vals[0], I_vals[1], I_vals[3]],
            [I_vals[1], I_vals[2], I_vals[4]],
            [I_vals[3], I_vals[4], I_vals[5]],
        ]
    )
    S = np.array(
        [
            [0, -mc_val[2], mc_val[1]],
            [mc_val[2], 0, -mc_val[0]],
            [-mc_val[1], mc_val[0], 0],
        ]
    )
    J = np.block([[I_mat, S], [S.T, m_val * np.eye(3)]])
    eig_min = np.linalg.eigvalsh(J).min()
    return eig_min > 0, eig_min, J


def print_lmi_feasibility(
    pi_full: np.ndarray,
    joint_names: list[str] | None = None,
    joint_order: list[int] | None = None,
    label: str = "identified",
    inertia_eps: float = 1e-6,
    verbose: bool = True,
) -> None:
    """Run ``check_lmi_feasibility`` per joint and print the results.

    The solver enforces the *strictly* positive-definite LMI
    ``J ≽ inertia_eps·I₆`` (``min_eig ≥ inertia_eps > 0``, the same margin the
    solver uses), so an accepted solution must satisfy ``min_eig > 0``.
    A joint is marked ``YES`` if ``min_eig > 0``, else ``NO``.  Non-``YES``
    joints also print their identified ``(m, mc, I)`` and the eigenvalues of
    ``J``, so you can see which direction is off.
    """
    pi_list = split_joint_params(pi_full)
    if joint_order is None:
        joint_order = list(range(len(pi_list)))
    print(f"\nLMI physical-consistency check ({label} params):")
    print(f"{'Joint':<24s} {'feasible':>9s} {'min_eig':>12s}")
    print("-" * 47)
    all_ok = True
    for d in joint_order:
        name = joint_names[d] if joint_names else f"joint_{d}"
        ok, eig_min, J = check_lmi_feasibility(pi_list[d])
        all_ok = all_ok and ok
        print(f"{name:<24s} {('YES' if ok else 'NO'):>9s} {eig_min:>12.6g}")
        if verbose and not ok:
            p = pi_list[d]
            print(
                f"      m={p[0]:.6g}  mc=({p[1]:.6g},{p[2]:.6g},{p[3]:.6g})  "
                f"I=({p[4]:.6g},{p[5]:.6g},{p[6]:.6g},{p[7]:.6g},{p[8]:.6g},{p[9]:.6g})"
            )
            print(f"      J eigenvalues = {np.linalg.eigvalsh(J)}")
    print("-" * 47)
    print(
        f"All joints strictly feasible: {all_ok}  "
        f"(solver LMI strict margin eps={inertia_eps:.1g})"
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


def print_rmse_comparison(
    result: IdentificationResult,
    joint_names: list[str] | None,
    Y_stack: np.ndarray,
    tau_measured: np.ndarray,
) -> None:
    """Print torque RMSE before (prior) vs after (identified) identification."""
    rmse_prior = _joint_rmse(result.pi_prior, result.joint_order, Y_stack, tau_measured)
    rmse_ident = _joint_rmse(
        result.pi_identified, result.joint_order, Y_stack, tau_measured
    )

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
    rp_all = float(np.sqrt(np.mean((Y_stack @ result.pi_prior - tau_measured) ** 2)))
    ri_all = float(
        np.sqrt(np.mean((Y_stack @ result.pi_identified - tau_measured) ** 2))
    )
    print(
        f"{'ALL':<20s} {rp_all:>12.6g} {ri_all:>15.6g} "
        f"{_improve(rp_all, ri_all):>9.2f}%"
    )


# ============================================================================
# Pure SDP solver
# ============================================================================
class SDPSolver:
    """
    Pure global SDP-based parameter identification (weighted ridge variant).

    All joints are identified **simultaneously** in a single SDP/SOCP — there
    is no distal→proximal sequencing and no fixed-joint torque compensation:
    one SOC per joint over its own torque rows, plus one 6×6 pseudo-inertia
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
        joint_names: list[str] | None = None,
    ) -> IdentificationResult:
        """
        Run **global (all-at-once) weighted-ridge** identification.

        All ``dof`` joints are identified simultaneously in a single SDP/SOCP:
        one SOC per joint over its own torque rows, one 6×6 pseudo-inertia LMI
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
            equality constraint ``pi == prior`` (used for the ``null``/``small``
            quality labels — these are too small for a large ridge weight to
            pin exactly, so they are frozen literally).  ``None`` = no freeze.
        inertia_eps : float
            Strict positive-definiteness margin of the LMI (J ≽ eps·I₆).
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
        joint_names: list[str] | None = None,
    ) -> tuple[np.ndarray, float, list[float]]:
        """Single all-joints weighted-ridge SDP/SOCP.

        Returns ``(pi_opt, solve_time, λ_list)``.

        Variables: one 13-param block per joint ``pi[13d:13d+13]``, one
        non-negative scalar ``λ_d`` per joint, and the ridge epigraph scalar
        ``u``.  Objective ``min Σ_d λ_d + u``.

        Constraints per joint ``d``:
          · SOC  ‖Y_d_rows @ pi − τ_d‖₂ ≤ λ_d   (rows of joint d only; the
            columns are full because the cross-coupling of a proximal joint
            with its distal subtree members is resolved inside this one
            problem)
          · LMI  6×6 pseudo-inertia J_d ≽ inertia_eps·I₆  (strict, PD)
          · hard physical non-negativity of armature/damping/friction
            (mass ≥ eps is implied by the LMI)
        plus the weighted-ridge epigraph SOC ``‖√c ⊙ (pi − prior)‖₂ ≤ u`` with
        ``c = ridge_weights`` (the L2-toward-prior penalty replaces the old
        per-quality box constraints).

        If ``freeze_mask`` is given, parameters flagged ``True`` (quality
        ``null``/``small``) are additionally hard-frozen by equality
        ``pi == prior`` and excluded from the ridge norm.
        """
        dof = len(joint_order)
        pi_prior_full = np.concatenate(pi_prior_list)
        ridge_weights = np.asarray(ridge_weights, dtype=float).reshape(-1)
        assert ridge_weights.size == dof * N_PER_JOINT, (
            f"ridge_weights size {ridge_weights.size} != dof*13 = {dof * N_PER_JOINT}"
        )
        c_sqrt = np.sqrt(np.maximum(ridge_weights, 0.0))  # √c per parameter

        # --- Hard freeze (null/small): equality pi == prior, and zero out the
        #     (now irrelevant) ridge coefficient for frozen parameters so the
        #     epigraph SOC stays well-conditioned. ---
        n_frozen = 0
        if freeze_mask is not None:
            freeze_mask = np.asarray(freeze_mask, dtype=bool).reshape(-1)
            assert freeze_mask.size == dof * N_PER_JOINT, (
                f"freeze_mask size {freeze_mask.size} != dof*13 = {dof * N_PER_JOINT}"
            )
            n_frozen = int(freeze_mask.sum())
            c_sqrt = c_sqrt.copy()
            c_sqrt[freeze_mask] = 0.0
        if self.verbose and n_frozen:
            print(
                f"  hard-freeze (null/small): {n_frozen} params at prior (pi == prior)"
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

            # 2) physical consistency: 6×6 pseudo-inertia LMI strictly
            #    positive definite: J ≽ eps·I₆  (min_eig ≥ inertia_eps > 0)
            J = build_pseudo_inertia_LMI(pi_d[0], pi_d[1:4], pi_d[4:10])
            cstr.append(J - inertia_eps * np.eye(6) >> 0)

            # 3) hard physical non-negativity of armature/damping/friction
            for i in (10, 11, 12):
                cstr.append(pi_d[i] >= 0.0)

        # --- Hard-freeze equality constraints (null/small): pi == prior ---
        if freeze_mask is not None and n_frozen:
            for g in np.nonzero(freeze_mask)[0]:
                cstr.append(pi[g] == pi_prior_full[g])

        # --- Weighted ridge epigraph: obj += u,  u ≥ ‖√c ⊙ (pi − prior)‖₂ ---
        u = cp.Variable(nonneg=True)
        cstr.append(cp.SOC(u, cp.multiply(c_sqrt, pi - pi_prior_full)))

        # --- Solve ---
        problem = cp.Problem(cp.Minimize(cp.sum(lam) + u), cstr)
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
    ):
        """Pretty-print identification results joint by joint.

        If ``Y_stack`` and ``tau_measured`` are provided, prints a per-joint
        torque RMSE comparison (prior vs identified) instead of Σ‖τ_residual‖₂.
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
            hdr = (
                f"{'Param':<10s} {'Prior':>12s} {'Identified':>12s} "
                f"{'True':>12s} {'Err%':>8s} {'Δ%':>8s} {'Quality':>10s}"
                if has_ref
                else f"{'Param':<10s} {'Prior':>12s} {'Identified':>12s} {'Δ%':>8s} {'Quality':>10s}"
            )
            print(hdr)
            print("-" * len(hdr))

            for i in range(N_PER_JOINT):
                pr = pi_prior[d][i]
                ident = pi_id[d][i]
                g = d * N_PER_JOINT + i
                q = quality_map.get(g, "?") if quality_map else "?"
                dp = (ident - pr) / max(abs(pr), 1e-12) * 100  # Δ% from prior
                if has_ref:
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
    waist_yaw_offset: float = 0.0,
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
    waist_yaw_offset: float = 0.0,
    trajectory_yaml: str | None = None,
    twin: str | None = None,
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
    q_arm, v_arm, a_arm = _recovered_at_times(
        t_sel, dof, trajectory_yaml, grid_sample_rate=500.0
    )
    if verbose:
        print(f"  [measurement] q/v/a from Fourier trajectory {trajectory_yaml}")

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
_YAML_INERTIA_YAML2OUR = {4: 4, 5: 5, 6: 7, 7: 6, 8: 8, 9: 9}
# YAML inertia: [Ixx, Ixy, Iyy, Ixz, Iyz, Izz] → our Pinocchio: [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]


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
    """
    import yaml

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    per_param = data.get("_diagnostics", {}).get("per_param", [])
    dof = len(per_param) // N_PER_JOINT

    result = {}
    for entry in per_param:
        yaml_g = entry["idx"]
        j, i = _yaml_global_to_local(yaml_g, dof)
        our_g = j * N_PER_JOINT + i
        result[our_g] = entry["quality"]
    return result


def _ridge_weight_for_quality(quality: str | None) -> float:
    """Map a YAML quality label → its relative-deviation ridge weight (w_q)."""
    if quality is None or quality not in RIDGE_QUALITY_WEIGHTS:
        return RIDGE_QUALITY_WEIGHTS[DEFAULT_QUALITY]
    return RIDGE_QUALITY_WEIGHTS[quality]


def build_ridge_weights(
    quality_map: dict[int, str] | None,
    pi_prior: np.ndarray,
    *,
    ridge_lambda: float = RIDGE_LAMBDA,
    scale_floor: float = RIDGE_SCALE_FLOOR,
) -> tuple[np.ndarray, list[str]]:
    """
    Build per-parameter weighted-ridge quadratic coefficients c_i.

    Replaces the per-quality box (freeze / widen) with a ridge penalty.  For
    every global parameter i with URDF prior ``prior_i``, deviation from the
    prior is penalised in *relative* terms

        c_i · (pi_i − prior_i)² ,   c_i = ridge_lambda · w_q / scale_i²

    with ``scale_i = max(|prior_i|, scale_floor)`` and ``w_q`` from
    ``RIDGE_QUALITY_WEIGHTS``.  Better quality → smaller ``w_q`` → the
    parameter may drift further from the prior; poor quality → larger ``w_q``
    → it is pulled back hard (``null`` ≈ frozen).

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

    Returns
    -------
    (c, labels) : the (dof*13,) quadratic-coefficient vector and the
    per-parameter quality label used (for reporting).
    """
    dof = len(pi_prior) // N_PER_JOINT
    pi_list = split_joint_params(pi_prior)
    c = np.zeros(dof * N_PER_JOINT)
    labels: list[str] = []
    for g in range(dof * N_PER_JOINT):
        q = (quality_map or {}).get(g)
        w_q = _ridge_weight_for_quality(q)
        labels.append(q if q in RIDGE_QUALITY_WEIGHTS else DEFAULT_QUALITY)
        j, i = g // N_PER_JOINT, g % N_PER_JOINT
        prior_val = pi_list[j][i]
        scale = max(abs(prior_val), scale_floor)
        c[g] = ridge_lambda * w_q / (scale * scale)
    return c, labels


def build_freeze_mask(
    quality_map: dict[int, str] | None,
    dof: int,
    freeze_qualities: frozenset[str] = RIDGE_FREEZE_QUALITIES,
) -> np.ndarray:
    """
    (dof*13,) boolean mask of parameters to **hard-freeze** (pi == prior).

    Parameters whose YAML quality label is in ``freeze_qualities`` (default
    ``RIDGE_FREEZE_QUALITIES`` = null/small) are frozen at their URDF prior by
    an equality constraint in the solver, so they cannot move at all — a plain
    large ridge weight cannot fully pin such tiny-magnitude parameters.
    """
    mask = np.zeros(dof * N_PER_JOINT, dtype=bool)
    if quality_map:
        for g, q in quality_map.items():
            if q in freeze_qualities:
                mask[g] = True
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
    waist_yaw_offset: float = 0.0,
    grid_sample_rate: float = 500.0,
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
        t_sel, dof, val_yaml, grid_sample_rate=grid_sample_rate
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
    print_rmse_comparison(result, joint_names, Y_val, tau_measured)

    # --- 7) Plot ---
    return _plot_torque_comparison_panels(
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
        default=None,
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

    # ---- 模型 / 环境 ----
    ap.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        default=SETUP["gravity"].tolist(),
        metavar=("X", "Y", "Z"),
        help="机体系重力向量（IMU 读得）",
    )
    ap.add_argument(
        "--waist-offset",
        type=float,
        default=SETUP["waist_yaw_offset"],
        help="腰关节 J12_WAIST_YAW 固定角度 (rad)",
    )

    # ---- 验证 / 绘图 ----
    ap.add_argument(
        "--val-yaml",
        default=str(VAL_YAML_PATH),
        metavar="NAME",
        help="[meas] 验证用轨迹 YAML（默认取文件顶部 VAL_YAML_PATH 常量）；"
        "其 bag 也从该 YAML 的 _meta.source_bag 读取",
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
    return ap


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from identification.fourier_trajectory import FourierTrajectory

    is_sim = args.sim

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

    gravity = np.asarray(args.gravity, dtype=float)

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
    print(f"twin-id         : {args.twin_id or '(全程)'}")

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
            waist_yaw_offset=args.waist_offset,
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
            waist_yaw_offset=args.waist_offset,
            trajectory_yaml=traj_yaml,
            twin=args.twin_id,  # 辨识用时间窗（选一个质量好的周期，None=全程）
        )

    # 2. Configure weighted-ridge regularization.
    #    每个参数按质量分档获得 L2 惩罚系数（质量好 → 系数小 → 允许偏离 URDF
    #    prior 更多；质量差 → 系数大 → 更强地拉回 prior），由 build_ridge_weights
    #    从质量标签生成；LMI 严格正定余量直接用模块常量 LMI_EPS。
    quality_map = None
    if quality_yaml is not None:
        quality_path = FourierTrajectory._coeffs_dir / quality_yaml
        if quality_path.is_file():
            quality_map = load_yaml_param_quality(quality_path)
    ridge_weights, ridge_labels = build_ridge_weights(
        quality_map, data["pi_prior"], ridge_lambda=args.ridge_lambda
    )
    freeze_mask = build_freeze_mask(quality_map, dof=int(data["dof"]))
    from collections import Counter

    n_freeze = int(freeze_mask.sum())
    if quality_map:
        qc = Counter(quality_map.values())
        print(f"  YAML quality distribution: {dict(qc)}")
        print(f"  ridge per-quality weights: {dict(RIDGE_QUALITY_WEIGHTS)}")
        print(f"  hard-freeze (null/small): {n_freeze} params → pi == prior")
    else:
        print(
            f"  无 _diagnostics.per_param 质量标签 → 全部按默认分档 "
            f"'{DEFAULT_QUALITY}' 加岭惩罚，无冻结"
        )
    c_eff = ridge_weights.copy()
    c_eff[freeze_mask] = np.nan
    print(
        f"  ridge_lambda={args.ridge_lambda}: active c range "
        f"[{np.nanmin(c_eff):.4g}, {np.nanmax(c_eff):.4g}] "
        f"(hard-frozen: {n_freeze})"
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
        inertia_eps=LMI_EPS,
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
    )

    # 4b. 对辨识后的惯性参数做物理一致性（pseudo-inertia LMI）检查并打印。
    print_lmi_feasibility(
        result.pi_identified, data["joint_names"], result.joint_order, "identified"
    )

    # 5. Cross-validation / comparison plot
    #    sim：真值 vs prior vs identified 对比。
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
            plot_torque_comparison_measured(
                result,
                urdf_path=args.urdf,  # 先验模型（regressor）
                val_yaml=args.val_yaml,
                val_bag_name=None,  # 从 val YAML _meta.source_bag 读取
                joint_names=data["joint_names"],
                limb_group=None,
                csv_topic=args.csv_topic,
                sample_rate=args.sample_rate,
                gravity=gravity,
                waist_yaw_offset=args.waist_offset,
                twin=args.twin,  # 时间窗缩放，如 "0:13.4"
            )

    print("\n" + "=" * 100)
    print("SDP identification finished.".center(100))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
