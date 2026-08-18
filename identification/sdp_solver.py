# \!/usr/bin/env python3
"""
Sequential SDP-based Parameter Identification for Serial Robot Limbs.

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
  │  · Sequential distal→proximal SOCP                  │
  │  · LMI physical-consistency constraint              │
  │  · Per-joint, per-parameter bounds via ParamBounds  │
  │  · Auto-freeze low-sensitivity parameters           │
  └─────────────────────────────────────────────────────┘

This design means you can later swap in real sensor torque data
by only changing the data-preparation step — the solver stays the same.

Dependencies: cvxpy + MOSEK (or SCS), numpy
"""

from __future__ import annotations

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

# BOUNDRY_PARAMS = {
#     "null": {"freeze": True, "widen": 0.0},
#     "rank_deficient": {"freeze": False, "widen": 0.1},
#     "small": {"freeze": True, "widen": 0.0},
#     "bad": {"freeze": False, "widen": 0.1},
#     "ok": {"freeze": False, "widen": 0.3},
#     "good": {"freeze": False, "widen": 0.5},
# }
BOUNDRY_PARAMS = {
    "null": {"freeze": True, "widen": 0.0},
    "rank_deficient": {"freeze": False, "widen": 0.1},
    "small": {"freeze": False, "widen": 0.1},
    "bad": {"freeze": False, "widen": 0.2},
    "ok": {"freeze": False, "widen": 0.5},
    "good": {"freeze": False, "widen": 0.7},
}


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
    ``J ≽ inertia_eps·I₆`` (``min_eig ≥ inertia_eps > 0``, see
    ``ParamBounds``), so an accepted solution must satisfy ``min_eig > 0``.
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
# Bound configuration — per-joint, per-parameter
# ============================================================================
class ParamBounds:
    """
    Per-joint, per-parameter bounds for the 13-parameter vector.

    The bounds are built automatically in ``__init__`` as symmetric relative
    intervals around the prior ``pi_prior`` (the baseline), e.g. for a
    parameter ``p`` with relative width ``rel``:

        lb = p·(1 − rel)      ub = p·(1 + rel)

    (off-diagonal inertia uses ±|p|·rel since it can be negative).

    ``lb_matrix`` / ``ub_matrix`` have shape ``(dof, 13)``.
    Use ``np.inf`` / ``-np.inf`` for unconstrained dimensions.
    To freeze a parameter: set ``lb == ub == prior``, or call ``set_frozen()``.

    After construction, refine the baseline per-parameter with
    ``configure_joint()``, ``apply_from_global_indices()`` or
    ``apply_yaml_quality()``.
    """

    def __init__(
        self,
        pi_prior: np.ndarray,  # (dof*13,)
        rel_mass: float = 0.5,
        rel_mc: float = 0.5,
        rel_inertia_diag: float = 0.5,
        rel_inertia_offdiag: float = 0.5,
        rel_armature: float = 0.5,
        rel_damping: float = 0.5,
        rel_friction: float = 0.5,
        *,
        # Hard physical constraints (merged on top of user bounds)
        enforce_positive_mass: bool = True,
        enforce_nonneg_friction: bool = True,
        # Strict PD margin: enforce J ≽ eps·I₆ (min_eig(J) ≥ eps > 0)
        inertia_eps: float = 1e-6,
    ):
        """
        Build bounds from symmetric relative intervals around prior values.

        For each parameter p:   lb = p·(1 − rel),  ub = p·(1 + rel).
        Off-diagonal inertia uses ±|p|·rel (they can be negative).
        """
        dof = len(pi_prior) // N_PER_JOINT
        lb = np.full((dof, N_PER_JOINT), -np.inf)
        ub = np.full((dof, N_PER_JOINT), +np.inf)
        pi_list = split_joint_params(pi_prior)

        for j in range(dof):
            p = pi_list[j]
            # mass (0) — always positive
            lb[j, 0] = p[0] * (1 - rel_mass)
            ub[j, 0] = p[0] * (1 + rel_mass)
            # mc (1-3) and all inertia (4-9) — can be negative
            for i in range(1, 10):
                w = abs(p[i]) * (
                    rel_mc
                    if i <= 3
                    else rel_inertia_diag
                    if i in (4, 6, 9)  # Ixx, Iyy, Izz in Pinocchio order
                    else rel_inertia_offdiag
                )
                lb[j, i] = p[i] - w
                ub[j, i] = p[i] + w
            # armature (10), damping (11), friction (12) — always non-negative
            lb[j, 10] = p[10] * (1 - rel_armature)
            ub[j, 10] = p[10] * (1 + rel_armature)
            lb[j, 11] = p[11] * (1 - rel_damping)
            ub[j, 11] = p[11] * (1 + rel_damping)
            lb[j, 12] = p[12] * (1 - rel_friction)
            ub[j, 12] = p[12] * (1 + rel_friction)

        self.lb_matrix = lb
        self.ub_matrix = ub
        self.enforce_positive_mass = enforce_positive_mass
        self.enforce_nonneg_friction = enforce_nonneg_friction
        self.inertia_eps = inertia_eps

    # ------------------------------------------------------------------
    def get_bounds_for_joint(self, joint_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) for joint ``joint_idx``, with hard constraints merged."""
        lb = self.lb_matrix[joint_idx].copy()
        ub = self.ub_matrix[joint_idx].copy()

        if self.enforce_positive_mass:
            lb[0] = max(lb[0], 1e-6)
        if self.enforce_nonneg_friction:
            for i in (10, 11, 12):
                lb[i] = max(lb[i], 0.0)
        return lb, ub

    # ------------------------------------------------------------------
    def configure_joint(
        self,
        joint_idx: int,
        prior: np.ndarray,
        *,
        freeze: list[int] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        widen: dict[str, float] | None = None,
    ):
        """
        Per-joint, per-parameter manual configuration.

        Parameters
        ----------
        joint_idx : int
            Joint index in group_to_identify order (0=most proximal).
        prior : (13,) ndarray
            Prior values for this joint.
        freeze : list[int] or None
            Indices to freeze at prior, e.g. ``[0,1,2,3]`` freezes mass+mc.
        bounds : dict[str, (lo, hi)] or None
            Absolute bounds by name, e.g. ``{"mass": (0.5, 2.0)}``.
            Names match ``_PARAM_LABELS``: mass, mc_x/y/z, Ixx/Ixy/.../Izz,
            armature, damping, friction.
        widen : dict[str, float] or None
            Widen relative bounds: ``{"mass": 0.8}`` gives ±80%.
        """
        name_to_idx = {n.strip(): i for i, n in enumerate(_PARAM_LABELS)}
        if freeze:
            self.set_frozen(joint_idx, freeze, prior)
        if bounds:
            for name, (lo, hi) in bounds.items():
                i = name_to_idx[name]
                self.lb_matrix[joint_idx, i] = lo
                self.ub_matrix[joint_idx, i] = hi
        if widen:
            for name, rel in widen.items():
                i = name_to_idx[name]
                p = prior[i]
                w = abs(p) * rel
                self.lb_matrix[joint_idx, i] = p - w
                self.ub_matrix[joint_idx, i] = p + w

    # ------------------------------------------------------------------
    def set_frozen(self, joint_idx: int, param_indices: list[int], prior: np.ndarray):
        """Freeze specific parameters at their prior values."""
        for i in param_indices:
            self.lb_matrix[joint_idx, i] = prior[i]
            self.ub_matrix[joint_idx, i] = prior[i]

    def is_frozen(self, joint_idx: int, i: int) -> bool:
        return np.isclose(self.lb_matrix[joint_idx, i], self.ub_matrix[joint_idx, i])

    # ------------------------------------------------------------------
    def apply_from_global_indices(
        self,
        pi_prior: np.ndarray,  # (dof*13,)
        freeze_global: list[int] | None = None,
        widen_global: dict[int, float] | None = None,
        bounds_global: dict[int, tuple[float, float]] | None = None,
    ):
        """
        Apply freeze/widen/bounds using **global** YAML parameter indices.

        Parameters
        ----------
        pi_prior : (dof*13,) ndarray
        freeze_global : list[int] or None
            Global indices to freeze at prior, e.g. ``[0, 4, 5, 7]``.
        widen_global : dict[int, float] or None
            ``{global_idx: rel_width}``, e.g. ``{3: 1.0}`` gives ±100%.
        bounds_global : dict[int, (lo, hi)] or None
            ``{global_idx: (lo, hi)}`` for absolute bounds.
        """
        pi_list = split_joint_params(pi_prior)
        dof = len(pi_prior) // N_PER_JOINT
        if freeze_global:
            for g in freeze_global:
                j, i = _yaml_global_to_local(g, dof)
                self.set_frozen(j, [i], pi_list[j])
        if widen_global:
            for g, rel in widen_global.items():
                j, i = _yaml_global_to_local(g, dof)
                p = pi_list[j][i]
                w = abs(p) * rel
                self.lb_matrix[j, i] = p - w
                self.ub_matrix[j, i] = p + w
        if bounds_global:
            for g, (lo, hi) in bounds_global.items():
                j, i = _yaml_global_to_local(g, dof)
                self.lb_matrix[j, i] = lo
                self.ub_matrix[j, i] = hi

    # ------------------------------------------------------------------
    def apply_yaml_quality(
        self,
        yaml_path: str | Path,
        pi_prior: np.ndarray,
        *,
        overrides: dict[str, dict] | None = None,
    ):
        """
        Apply freeze/widen based on YAML ``_diagnostics.per_param`` quality labels.

        Default strategy (matches ``BOUNDRY_PARAMS``, customisable via
        ``overrides``):

        ===============  ======  =====
        quality           freeze  widen
        ===============  ======  =====
        null              yes     0
        rank_deficient    no      0.1
        small             yes     0
        bad               no      0.1
        ok                no      0.3
        good              no      0.5
        ===============  ======  =====

        Parameters
        ----------
        yaml_path : str or Path
        pi_prior : (dof*13,) ndarray
        overrides : dict or None
            e.g. ``{"rank_deficient": {"freeze": False, "widen": 0.5}}``.
        """
        quality_map = load_yaml_param_quality(yaml_path)
        if not quality_map:
            print(f"  ⚠ apply_yaml_quality: NO quality data found in {yaml_path}")
            return
        pi_list = split_joint_params(pi_prior)
        frozen_count = 0
        widen_count = 0

        for g, quality in quality_map.items():
            j, i = g // N_PER_JOINT, g % N_PER_JOINT
            strategy = dict(_quality_defaults(quality))
            if overrides and quality in overrides:
                strategy.update(overrides[quality])
            prior_val = pi_list[j][i]

            if strategy["freeze"]:
                self.set_frozen(j, [i], pi_list[j])
                frozen_count += 1
            elif strategy["widen"] > 0:
                w = abs(prior_val) * strategy["widen"]
                self.lb_matrix[j, i] = prior_val - w
                self.ub_matrix[j, i] = prior_val + w
                widen_count += 1

        print(
            f"  apply_yaml_quality: {len(quality_map)} params loaded, "
            f"{frozen_count} frozen, {widen_count} widened"
        )


# ============================================================================
# Identifiability analysis
# ============================================================================
def analyse_identifiability(
    Y_blk: np.ndarray,  # (M, 13)
    prior: np.ndarray,  # (13,)
    freeze_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-parameter sensitivity:  ‖∂τ/∂π_i‖₂ · |prior_i|.

    Returns (weighted_norms, freeze_suggestion).
    Parameters below ``freeze_threshold × max`` are suggested for freezing.
    """
    col_norms = np.linalg.norm(Y_blk, axis=0)
    weighted = col_norms * np.abs(prior)
    max_w = weighted.max()
    if max_w < 1e-15:
        return weighted, np.ones(N_PER_JOINT, dtype=bool)
    return weighted, weighted < freeze_threshold * max_w


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
    Pure sequential SDP-based parameter identification.

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
        pi_prior: np.ndarray,  # (dof*13,)         prior params (for bounds)
        subtree_mask: np.ndarray,  # (dof, dof) bool   kinematic dependency
        joint_order: list[int],  # distal → proximal indices
        bounds: ParamBounds,
        auto_freeze_threshold: float = 0.0,
        joint_names: list[str] | None = None,
    ) -> IdentificationResult:
        """
        Run sequential identification.

        Parameters
        ----------
        auto_freeze_threshold : float
            If > 0, auto-freeze params below this fraction of max sensitivity.
        """
        dof = len(joint_order)
        N = Y_stack.shape[0] // dof
        pi_identified = np.zeros(dof * N_PER_JOINT)
        pi_prior_list = split_joint_params(pi_prior)

        solve_times: list[float] = []
        objectives: list[float] = []

        if self.verbose:
            print(f"Sequential SDP ({dof} joints, order: {joint_order}):")

        for d in joint_order:
            name = joint_names[d] if joint_names else f"joint_{d}"
            prior = pi_prior_list[d]

            # --- Extract rows for joint d ---
            row_mask = np.zeros(Y_stack.shape[0], dtype=bool)
            for k in range(N):
                row_mask[k * dof + d] = True
            Y_d_rows = Y_stack[row_mask, :]  # (N, 13*dof)
            tau_d = tau_measured[row_mask]  # (N,)

            # --- Compensate: subtract frozen distal-joint contributions ---
            tau_comp = tau_d.copy()
            for j in range(dof):
                if j == d or not subtree_mask[d, j]:
                    continue
                if joint_order.index(j) < joint_order.index(d):
                    cs = j * N_PER_JOINT
                    ce = cs + N_PER_JOINT
                    tau_comp -= Y_d_rows[:, cs:ce] @ pi_identified[cs:ce]

            # --- Regressor block for joint d ---
            cs = d * N_PER_JOINT
            ce = cs + N_PER_JOINT
            Y_blk = Y_d_rows[:, cs:ce]  # (N, 13)

            # --- Auto-freeze ---
            if auto_freeze_threshold > 0:
                _, freeze_sug = analyse_identifiability(
                    Y_blk,
                    prior,
                    freeze_threshold=auto_freeze_threshold,
                )
                if np.any(freeze_sug):
                    bounds.set_frozen(d, list(np.where(freeze_sug)[0]), prior)
                    if self.verbose:
                        frozen = [
                            _PARAM_LABELS[i]
                            for i in range(N_PER_JOINT)
                            if freeze_sug[i]
                        ]
                        print(f"  Joint {d} ({name}): auto-freeze {frozen}")

            if self.verbose:
                # --- Data consistency + freeze status ---
                fit_prior = Y_blk @ prior
                resid_prior = np.linalg.norm(fit_prior - tau_comp)
                frozen_names = [
                    _PARAM_LABELS[i]
                    for i in range(N_PER_JOINT)
                    if bounds.is_frozen(d, i)
                ]
                print(
                    f"  Joint {d} ({name}): Y_blk=({Y_blk.shape}), "
                    f"‖τ_res‖={np.linalg.norm(tau_comp):.4f}, "
                    f"‖Y·prior−τ‖={resid_prior:.4e}"
                )
                if frozen_names:
                    print(f"    frozen: {frozen_names}")
                else:
                    print(f"    frozen: (none)")

            pi_opt, dt, obj = self._identify_one(d, Y_blk, tau_comp, prior, bounds)
            pi_identified[cs:ce] = pi_opt
            solve_times.append(dt)
            objectives.append(obj)

            if self.verbose:
                print(f"    λ*={obj:.6g}, {dt:.3f}s")

        return IdentificationResult(
            pi_identified=pi_identified,
            pi_prior=pi_prior,
            pi_reference=None,
            joint_solve_times=solve_times,
            joint_objectives=objectives,
            joint_order=joint_order,
        )

    # ------------------------------------------------------------------
    def _identify_one(
        self,
        joint_idx: int,
        Y_blk: np.ndarray,  # (M, 13)
        tau_res: np.ndarray,  # (M,)
        prior: np.ndarray,  # (13,)
        bounds: ParamBounds,
    ) -> tuple[np.ndarray, float, float]:
        """Single-joint SOCP.  Returns (pi_opt, solve_time, objective)."""
        lb, ub = bounds.get_bounds_for_joint(joint_idx)

        # Pre-solve diagnostics
        lmi_ok, eig_min, _ = check_lmi_feasibility(prior)
        in_bounds = np.all((prior >= lb) & (prior <= ub))

        if self.verbose:
            if not lmi_ok:
                print(f"    ⚠ prior LMI FAIL (min eig={eig_min:.4e})")
            if not in_bounds:
                viol = np.where((prior < lb) | (prior > ub))[0]
                print(f"    ⚠ prior OUT OF BOUNDS at indices {list(viol)}")
                for i in viol:
                    print(
                        f"       {_PARAM_LABELS[i]}: prior={prior[i]:.6g}  bounds=[{lb[i]:.6g}, {ub[i]:.6g}]"
                    )
            if lmi_ok and in_bounds:
                print(f"    ✓ prior feasible (LMI OK, in bounds)")

        # Variables
        pi = cp.Variable(N_PER_JOINT)
        lam = cp.Variable(nonneg=True)

        # Constraints
        cstr: list = []
        cstr.append(cp.SOC(lam, Y_blk @ pi - tau_res))

        # Physical consistency: 6×6 pseudo-inertia LMI strictly positive
        # definite: J ≽ eps·I₆  (min_eig ≥ inertia_eps > 0)
        J = build_pseudo_inertia_LMI(pi[0], pi[1:4], pi[4:10])
        cstr.append(J - bounds.inertia_eps * np.eye(6) >> 0)

        for i in range(N_PER_JOINT):
            if bounds.is_frozen(joint_idx, i):
                cstr.append(pi[i] == prior[i])
            else:
                if np.isfinite(lb[i]):
                    cstr.append(pi[i] >= lb[i])
                if np.isfinite(ub[i]):
                    cstr.append(pi[i] <= ub[i])

        # Solve
        problem = cp.Problem(cp.Minimize(lam), cstr)
        try:
            problem.solve(solver=self.solver_name, verbose=False)
        except cp.error.SolverError:
            if self.verbose:
                print(f"    [{self.solver_name}] failed → SCS ...")
            problem.solve(solver="SCS", verbose=False, max_iters=5000)

        t = problem.solver_stats.solve_time if problem.solver_stats else 0.0

        if pi.value is None:
            msg = f"SDP infeasible for joint {joint_idx} (status={problem.status})."
            if self.verbose:
                msg += (
                    f"\n    inertia prior: {'OK' if lmi_ok else f'FAIL (min eig={eig_min:.4e})'}"
                    f"\n    Try: wider bounds, or freeze more params."
                )
            raise RuntimeError(msg)

        return np.array(pi.value).flatten(), t, float(lam.value)

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
    limb_group: str = "left_arm",
    sample_rate: float = 200.0,
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
    """
    from identification.fourier_trajectory import FourierTrajectory
    from identification.target_limb_regressor import (
        TargetLimbRegressor,
        VALID_LIMB_GROUPS,
    )

    if urdf_true_path is None:
        urdf_true_path = urdf_path

    # --- Trajectory ---
    dof_limb = len(VALID_LIMB_GROUPS[limb_group])
    ft = FourierTrajectory(dim=dof_limb, sample_rate=sample_rate)
    yaml_name = Path(yaml_filename).name
    q_traj, v_traj, a_traj = ft.generate_trajectory_from_yaml(yaml_name)
    N = q_traj.shape[1]
    if verbose:
        print(f"Trajectory: {N} steps from {yaml_name}")

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


def _latest_recovered_yaml() -> str:
    """Return the latest ``recovered_*.yaml`` in ``trajectory_coefficients/``."""
    from identification.fourier_trajectory import FourierTrajectory

    matches = sorted(FourierTrajectory._coeffs_dir.glob("recovered_*.yaml"))
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
    limb_group: str = "left_arm",
    csv_topic: str = "hardware_joint_state",
    sample_rate: float = 100.0,
    verbose: bool = True,
    gravity: np.ndarray | None = None,
    waist_yaw_offset: float = 0.0,
    trajectory_yaml: str | None = None,
    twin: str | None = None,
) -> dict:
    """Prepare SDP data from **real** bag measurement (CSV).

    Prior model, subtree mask, joint order and (in ``main``) the bounds
    configuration are identical to ``prepare_data_from_urdf``.  Only the two
    data streams change:

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
    limb_group : str
        Limb group, e.g. ``'left_arm'`` → CSV joints 13..17.
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
    from identification.target_limb_regressor import (
        TargetLimbRegressor,
        VALID_LIMB_GROUPS,
    )

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

    # --- Resolve trajectory yaml + bag name.  The bag is read from the
    #     yaml's _meta.source_bag (e.g. '55_31' → bag 13_55_31) when not
    #     given explicitly. ---
    if trajectory_yaml is None:
        trajectory_yaml = _latest_recovered_yaml()
    if bag_name is None:
        bag_name = _yaml_source_bag(trajectory_yaml)
        if verbose:
            print(f"  [measurement] bag from yaml _meta.source_bag: {bag_name}")

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


def _quality_defaults(quality: str) -> dict:
    """Default freeze/widen strategy per YAML quality label."""
    assert quality in ("null", "rank_deficient", "small", "bad", "ok", "good")
    return BOUNDRY_PARAMS.get(quality)


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
    limb_group: str = "left_arm",
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
    from identification.target_limb_regressor import TargetLimbRegressor

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
# main / demo
# ============================================================================
def main():
    from identification.fourier_trajectory import FourierTrajectory

    URDF_PATH = (
        Path(__file__).resolve().parent.parent
        / "resource"
        / "robot"
        / "urdf"
        / "serial_pm_v2_identify.urdf"
    ).resolve()
    # 注意：只有 prepare_data_from_urdf 的合成 demo 才有“真值”URDF
    # （urdf_true_path，例如 .../serial_pm_v2_identify_nominal.urdf）；
    # 真实测量辨识没有地面真值，因此这里不再定义 URDF_TRUE_PATH。
    SETUP = {
        "gravity": np.array([-9.712746, 0.390467, -1.393897]),
        "waist_yaw_offset": 0.076485634,
    }
    # 时间窗缩放（参考 compare_torque.py 的 -w/--twin）：'START:END'（秒），
    # 任一端可省略；None = 全程。
    TWIN = None  # 例如 "0:13.4"（验证绘图的时间窗）

    # 辨识用时间窗：轨迹是周期的，重复的周期不增加信息量；手动选一个质量
    # 较好的周期（如 "10:16.667"）只用一个周期辨识，通常能提升效果。None=全程。
    TWIN_ID = None  # 例如 "10:16.667"

    latest_yaml = FourierTrajectory.find_latest_yaml("unified")

    # 1. Prepare data
    # ------------------------------------------------------------------
    # Option ① (original, URDF-synthesized tau): prepare_data_from_urdf 保留，
    # 需要时取消注释即可（demo 的“真值”URDF 需自行指定 urdf_true_path，
    # 例如 resource/robot/urdf/serial_pm_v2_identify_nominal.urdf）。
    # data = prepare_data_from_urdf(
    #     urdf_path=URDF_PATH,  # initial/prior model → regressor, bounds, freeze target
    #     yaml_filename=latest_yaml,
    #     limb_group="left_arm",
    #     sample_rate=100.0,
    #     urdf_true_path=None,  # demo 可改为 NOMINAL_URDF 路径以生成参考 tau
    #     gravity=SETUP["gravity"],
    #     waist_yaw_offset=SETUP["waist_yaw_offset"],
    # )

    # Option ② (measurement): tau 直接取自 bag 的 CSV（torque_13..17），
    # 回归状态 q/v/a 全部来自恢复的 Fourier 轨迹（在 CSV 时间上直接求值）；
    # 加速度不用实测速度差分（CSV 无加速度列，量化噪声大）。
    # 测量模式无地面真值 → pi_true 为 None，只靠实测 tau 辨识。
    data = data_from_measurement(
        urdf_path=URDF_PATH,  # 先验模型 → regressor / pi_prior / bounds
        limb_group="left_arm",
        csv_topic="hardware_joint_state",  # 实测力矩（勿用 command_feedback）
        sample_rate=100.0,  # CSV(~500Hz) 抽取到 ~100Hz
        gravity=SETUP["gravity"],
        waist_yaw_offset=SETUP["waist_yaw_offset"],
        # trajectory_yaml="recovered_260817_104829.yaml",
        trajectory_yaml="recovered_260817_142958.yaml",
        twin=TWIN_ID,  # 辨识用时间窗（选一个质量好的周期，None=全程）
    )

    # 2. Configure bounds — baseline from relative intervals around prior,
    #    then refine (freeze/widen) per-parameter below.
    #    LMI_EPS：6×6 pseudo-inertia 严格正定余量（J ≽ eps·I₆，min_eig ≥ eps > 0）。
    LMI_EPS = 1e-7
    bounds = ParamBounds(pi_prior=data["pi_prior"], inertia_eps=LMI_EPS)

    # --- Option A: manually freeze/widen by global YAML indices ---
    # null params (idx 0,4,5,7) → freeze; bad params (idx 3,12) → widen
    # bounds.apply_from_global_indices(
    #     pi_prior=data["pi_prior"],
    #     freeze_global=[0, 4, 5, 7],
    #     widen_global={3: 0.05, 12: 0.05},
    # )

    # --- Option B (alternative): auto-apply from YAML quality labels ---
    yaml_path = FourierTrajectory._coeffs_dir / latest_yaml
    bounds.apply_yaml_quality(yaml_path, data["pi_prior"])

    tau_measured = data["tau_measured"]

    # 3. Solve
    solver = SDPSolver(solver_name="MOSEK", verbose=True)
    result = solver.solve(
        Y_stack=data["Y_stack"],
        tau_measured=tau_measured,
        pi_prior=data["pi_prior"],
        subtree_mask=data["subtree_mask"],
        joint_order=data["joint_order"],
        bounds=bounds,
        auto_freeze_threshold=0.0,
        joint_names=data["joint_names"],
    )

    # 4. Report
    quality_map = load_yaml_param_quality(yaml_path)
    # Print quality distribution
    from collections import Counter

    qc = Counter(quality_map.values())
    print(f"  YAML quality distribution: {dict(qc)}")
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

    # 5. Cross-validation plot（测量模式）：用另一条轨迹/bag 验证辨识结果。
    #    pi_prior 与 pi_identified 分别在验证轨迹上算关节 tau，与验证 bag 的
    #    实测 tau 对比；验证 yaml/bag 与辨识用的不同。
    plot_torque_comparison_measured(
        result,
        urdf_path=URDF_PATH,  # 先验模型（regressor）
        # val_yaml="recovered_260817_142958.yaml",
        val_yaml="recovered_260817_104829.yaml",
        joint_names=data["joint_names"],
        gravity=SETUP["gravity"],
        waist_yaw_offset=SETUP["waist_yaw_offset"],
        sample_rate=100.0,
        twin=TWIN,  # 时间窗缩放，如 "0:13.4"
    )

    print("\n" + "=" * 100)
    print("SDP identification finished.".center(100))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
