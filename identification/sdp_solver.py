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

BOUNDRY_PARAMS = {
    "null": {"freeze": True, "widen": 0.0},
    "rank_deficient": {"freeze": False, "widen": 0.1},
    "small": {"freeze": True, "widen": 0.0},
    "bad": {"freeze": False, "widen": 0.1},
    "ok": {"freeze": False, "widen": 0.3},
    "good": {"freeze": False, "widen": 0.5},
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


# ============================================================================
# Bound configuration — per-joint, per-parameter
# ============================================================================
@dataclass
class ParamBounds:
    """
    Per-joint, per-parameter bounds for the 13-parameter vector.

    ``lb_matrix`` / ``ub_matrix`` have shape ``(dof, 13)``.
    Use ``np.inf`` / ``-np.inf`` for unconstrained dimensions.
    To freeze a parameter: set ``lb == ub == prior``, or call ``set_frozen()``.

    For convenience, use ``ParamBounds.from_relative(pi_prior, ...)``.
    """

    lb_matrix: np.ndarray  # (dof, 13)
    ub_matrix: np.ndarray  # (dof, 13)

    # Hard physical constraints (merged on top of user bounds)
    enforce_positive_mass: bool = True
    enforce_nonneg_friction: bool = True

    # Pseudo-inertia LMI relaxation:  J + eps·I₆ ≽ 0  (numerical tolerance)
    inertia_eps: float = 1e-2

    # ------------------------------------------------------------------
    @classmethod
    def from_relative(
        cls,
        pi_prior: np.ndarray,  # (dof*13,)
        rel_mass: float = 0.5,
        rel_mc: float = 0.5,
        rel_inertia_diag: float = 0.5,
        rel_inertia_offdiag: float = 0.5,
        rel_armature: float = 0.5,
        rel_damping: float = 0.5,
        rel_friction: float = 0.5,
        **kwargs,
    ) -> ParamBounds:
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

        return cls(lb_matrix=lb, ub_matrix=ub, **kwargs)

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

        Default strategy (customisable via ``overrides``):

        ===============  ======  =====
        quality           freeze  widen
        ===============  ======  =====
        null              yes     0
        rank_deficient    yes     0
        small             yes     0
        bad               no      0.05
        ok                no      0.1
        good              no      0.3
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
        fit_prior = Y_blk @ prior
        resid_prior = np.linalg.norm(fit_prior - tau_res)
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

        # Physical consistency: 6×6 pseudo-inertia LMI ≽ 0
        J = build_pseudo_inertia_LMI(pi[0], pi[1:4], pi[4:10])
        cstr.append(J + bounds.inertia_eps * np.eye(6) >> 0)

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
    ):
        """Pretty-print identification results joint by joint."""
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
        print(f"Σ ‖τ_residual‖₂: {sum(result.joint_objectives):.6g}")


# ============================================================================
# Data preparation
# ============================================================================
def prepare_data_from_urdf(
    urdf_path: str | Path,
    yaml_filename: str,
    limb_group: str = "left_arm",
    sample_rate: float = 200.0,
    urdf_true_path: str | Path | None = None,
    verbose: bool = True,
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
    def _reorder_y_aug(Y: np.ndarray, dof: int) -> np.ndarray:
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
def plot_torque_comparison(
    result: IdentificationResult,
    joint_names: list[str],
    Y_stack: np.ndarray,
    pi_reference: np.ndarray | None = None,
    sample_rate: float = 100.0,
):
    """Plot τ_true vs τ_prior vs τ_identified for each joint."""
    import matplotlib.pyplot as plt

    dof = len(result.joint_order)
    pi_true = pi_reference if pi_reference is not None else result.pi_prior
    pi_id = result.pi_identified
    pi_pr = result.pi_prior

    tau_true = Y_stack @ pi_true
    tau_prior = Y_stack @ pi_pr
    tau_ident = Y_stack @ pi_id

    N_total = len(tau_true)
    N = N_total // dof
    t = np.arange(N) / sample_rate

    fig, axes = plt.subplots(dof, 1, figsize=(12, 3 * dof), sharex=True)
    if dof == 1:
        axes = [axes]

    for idx, d in enumerate(result.joint_order):
        ax = axes[idx]
        row = np.arange(d, N_total, dof)
        ax.plot(t, tau_true[row], "k-", linewidth=1.0, alpha=0.7, label="true")
        ax.plot(t, tau_prior[row], "b--", linewidth=1.0, alpha=0.7, label="prior")
        ax.plot(t, tau_ident[row], "r-", linewidth=1.5, label="identified")
        ax.set_ylabel(f"{joint_names[d]}\n[Nm]")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Joint Torque Comparison: true vs prior vs identified", fontsize=13)
    plt.tight_layout()
    plt.show()


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
    URDF_TRUE_PATH = (
        Path(__file__).resolve().parent.parent
        / "resource"
        / "robot"
        / "urdf"
        # / "serial_pm_v2_identify_20pct.urdf"
        / "serial_pm_v2_identify_nominal.urdf"
    ).resolve()

    latest_yaml = FourierTrajectory.find_latest_yaml("unified")

    # 1. Prepare data — two URDFs: initial (prior) vs true (unknown to solver)
    # For debug with same URDF: omit urdf_true_path
    # For real test: urdf_true_path=URDF_PATH, urdf_path=NOMINAL_URDF
    data = prepare_data_from_urdf(
        urdf_path=URDF_PATH,  # initial/prior model → regressor, bounds, freeze target
        yaml_filename=latest_yaml,
        limb_group="left_arm",
        sample_rate=100.0,
        urdf_true_path=URDF_TRUE_PATH,  # ← change to NOMINAL_URDF for real test
    )

    # 2. Configure bounds
    bounds = ParamBounds.from_relative(
        pi_prior=data["pi_prior"],
        rel_mass=0.3,
        rel_mc=0.5,
        rel_inertia_diag=0.5,
        rel_inertia_offdiag=0.8,
        rel_armature=0.3,
        rel_damping=0.3,
        rel_friction=0.3,
    )

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

    # 3. Solve
    solver = SDPSolver(solver_name="MOSEK", verbose=True)
    result = solver.solve(
        Y_stack=data["Y_stack"],
        tau_measured=data["tau_measured"],
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
    )

    # 5. Plot torque comparison
    plot_torque_comparison(
        result,
        data["joint_names"],
        data["Y_stack"],
        pi_reference=data["pi_true"],
        sample_rate=100.0,
    )


if __name__ == "__main__":
    main()
