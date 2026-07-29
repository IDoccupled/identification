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
#   [m,  mc_x,  mc_y,  mc_z,   Ixx, Ixy, Ixz, Iyy, Iyz, Izz,  arm, damp, fric]
#    0     1      2      3      4    5    6    7    8    9     10    11    12
N_PER_JOINT = 13

_PARAM_LABELS = [
    "mass",
    "mc_x",
    "mc_y",
    "mc_z",
    "Ixx",
    "Ixy",
    "Ixz",
    "Iyy",
    "Iyz",
    "Izz",
    "armature",
    "damping",
    "friction",
]


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
# Physical-consistency LMI (Section 4, eq.22 of Jung et al. 2018)
# ============================================================================
def _skew3(v) -> cp.Expression:
    """3×3 skew-symmetric matrix."""
    if isinstance(v, np.ndarray):
        v = cp.Constant(v)
    return cp.bmat(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def build_pseudo_inertia_LMI(
    m: cp.Variable,
    mc: cp.Variable,
    I_vec: cp.Variable,
) -> cp.Expression:
    """
    6×6 pseudo-inertia matrix  [I, S(mc); S(mc)ᵀ, m·I₃] ≽ 0.

    Pinocchio stores rotational inertia about the JOINT FRAME (I_frame),
    not about COM.  The 6×6 LMI with I_frame is the standard physical
    consistency condition — its Schur complement gives I_com ≻ 0.

    Pinocchio ordering: I_vec = [Ixx, Ixy, Ixz, Iyy, Iyz, Izz].
    """
    I_mat = cp.bmat(
        [
            [I_vec[0], I_vec[1], I_vec[2]],  # Ixx Ixy Ixz
            [I_vec[1], I_vec[3], I_vec[4]],  # Ixy Iyy Iyz
            [I_vec[2], I_vec[4], I_vec[5]],  # Ixz Iyz Izz
        ]
    )
    S = _skew3(mc)
    return cp.bmat(
        [
            [I_mat, S],
            [S.T, m * np.eye(3)],
        ]
    )


def check_lmi_feasibility(pi: np.ndarray) -> tuple[bool, float, np.ndarray]:
    """Check 6×6 pseudo-inertia LMI.  Returns (ok, min_eig, J)."""
    m_val, mc_val = pi[0], pi[1:4]
    I_vals = pi[4:10]
    I_mat = np.array(
        [
            [I_vals[0], I_vals[1], I_vals[2]],
            [I_vals[1], I_vals[3], I_vals[4]],
            [I_vals[2], I_vals[4], I_vals[5]],
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
    lmi_eps: float = 1e-2

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
                    if i in (4, 7, 9)
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
    def set_frozen(self, joint_idx: int, param_indices: list[int], prior: np.ndarray):
        """Freeze specific parameters at their prior values."""
        for i in param_indices:
            self.lb_matrix[joint_idx, i] = prior[i]
            self.ub_matrix[joint_idx, i] = prior[i]

    def is_frozen(self, joint_idx: int, i: int) -> bool:
        return np.isclose(self.lb_matrix[joint_idx, i], self.ub_matrix[joint_idx, i])


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
                print(
                    f"  Joint {d} ({name}): Y_blk=({Y_blk.shape}), "
                    f"‖τ_res‖={np.linalg.norm(tau_comp):.4f}"
                )

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

        # Diagnostic
        lmi_ok, eig_min, J_prior = check_lmi_feasibility(prior)
        if not lmi_ok and self.verbose:
            print(f"    ⚠ prior LMI NOT satisfied (min eig={eig_min:.4e})")
            print(f"       m={prior[0]:.6g}  mc={prior[1:4]}")
            print("       I_frame =")
            for row in J_prior[:3, :3]:
                print(f"         [{row[0]:>12.6g}, {row[1]:>12.6g}, {row[2]:>12.6g}]")
            eigs = np.linalg.eigvalsh(J_prior)
            print(f"       6×6 eigenvalues = {eigs}")

        # Variables
        pi = cp.Variable(N_PER_JOINT)
        lam = cp.Variable(nonneg=True)

        # Constraints
        cstr: list = []
        cstr.append(cp.SOC(lam, Y_blk @ pi - tau_res))

        # Physical consistency: 6×6 pseudo-inertia LMI (Jung et al. eq.22)
        J = build_pseudo_inertia_LMI(pi[0], pi[1:4], pi[4:10])
        cstr.append(J + bounds.lmi_eps * np.eye(6) >> 0)

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
                    f"\n    COM inertia prior: "
                    f"{'OK' if lmi_ok else f'FAIL (min eig={eig_min:.4e})'}"
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
    ):
        """Pretty-print identification results joint by joint."""
        if pi_reference is None:
            pi_reference = result.pi_reference
        has_ref = pi_reference is not None

        print("\n" + "=" * 90)
        print("IDENTIFICATION RESULTS".center(90))
        print("=" * 90)

        pi_id = split_joint_params(result.pi_identified)
        pi_prior = split_joint_params(result.pi_prior)
        pi_ref = split_joint_params(pi_reference) if has_ref else None

        for d in result.joint_order:
            name = joint_names[d] if joint_names else f"joint_{d}"
            print(f"\n--- Joint {d}: {name} ---")
            if has_ref:
                hdr = (
                    f"{'Param':<10s} {'Prior':>12s} {'Identified':>12s} "
                    f"{'Ref':>12s} {'Err%':>9s}"
                )
            else:
                hdr = f"{'Param':<10s} {'Prior':>12s} {'Identified':>12s}"
            print(hdr)
            print("-" * len(hdr))

            for i in range(N_PER_JOINT):
                pr = pi_prior[d][i]
                ident = pi_id[d][i]
                if has_ref:
                    ref = pi_ref[d][i]
                    denom = abs(ref) if abs(ref) > 1e-12 else 1.0
                    err = (ident - ref) / denom * 100
                    print(
                        f"{_PARAM_LABELS[i]:<10s} {pr:>12.6g} {ident:>12.6g} "
                        f"{ref:>12.6g} {err:>8.2f}%"
                    )
                else:
                    print(f"{_PARAM_LABELS[i]:<10s} {pr:>12.6g} {ident:>12.6g}")

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
    sample_rate: float = 50.0,
    verbose: bool = True,
) -> dict:
    """
    Prepare all data needed by ``SDPSolver.solve()``.

    Uses FourierTrajectory + TargetLimbRegressor to compute:
        Y_stack, tau_measured, pi_prior, subtree_mask, joint_order,
        joint_names, dof, pi_true

    ``tau_measured`` is computed as Y_stack @ pi_true where pi_true is
    read from the URDF.  For real-robot data, replace ``tau_measured``
    with sensor readings.
    """
    from identification.fourier_trajectory import FourierTrajectory
    from identification.target_limb_regressor import (
        TargetLimbRegressor,
        VALID_LIMB_GROUPS,
    )

    # --- Trajectory ---
    dof_limb = len(VALID_LIMB_GROUPS[limb_group])
    ft = FourierTrajectory(dim=dof_limb, sample_rate=sample_rate)
    yaml_name = Path(yaml_filename).name
    q_traj, v_traj, a_traj = ft.generate_trajectory_from_yaml(yaml_name)
    N = q_traj.shape[1]
    if verbose:
        print(f"Trajectory: {N} steps from {yaml_name}")

    # --- Regressor & params ---
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
                f"  [{idx}] j{joint_id} {info['name']}: "
                f"pinocchio_jid={pin_jid} "
                f"name_in_model={reg.model.names[pin_jid]} "
                f"m={pi_inertial[0]:.6g} "
                f"Ixx={pi_inertial[4]:.6g} Iyy={pi_inertial[7]:.6g} Izz={pi_inertial[9]:.6g}"
            )
    pi_prior = np.array(pi_prior_list)  # (dof*13,)
    pi_true = pi_prior.copy()

    # Subtree mask + joint order
    reg.compute_regressor(print_info=False)
    subtree_mask = reg.subtree_mask.copy()
    subtree_size = subtree_mask.sum(axis=1)
    joint_order = sorted(range(dof), key=lambda d: subtree_size[d])
    joint_names = [reg.target_joint_infos[d]["name"] for d in range(dof)]

    # --- Stack regressor ---
    Y_list, tau_list = [], []
    for k in range(N):
        Y_aug, *_ = reg.compute_regressor(
            q_traj[:, k],
            v_traj[:, k],
            a_traj[:, k],
            print_info=False,
        )
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
# main / demo
# ============================================================================
def main():
    from identification.fourier_trajectory import FourierTrajectory
    from identification.target_limb_regressor import URDF_PATH

    latest_yaml = FourierTrajectory.find_latest_yaml("unified")

    # 1. Prepare data
    data = prepare_data_from_urdf(
        urdf_path=URDF_PATH,
        yaml_filename=latest_yaml,
        limb_group="left_arm",
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
    solver.print_results(
        result, joint_names=data["joint_names"], pi_reference=data["pi_true"]
    )


if __name__ == "__main__":
    main()
