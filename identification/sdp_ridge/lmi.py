"""Physical-consistency LMI (4×4 pseudo-inertia, second-moment form) + shape hinge.

The LMI is the *hard* part of the problem::

    J = [[Σ_O, h], [hᵀ, m]] ≽ 0 ,   Σ_O = ½·tr(I_O)·I₃ − I_O

with Schur complement ``Σ_C = Σ_O − h·hᵀ/m``.  ``J ≽ 0`` ⟺ ``{m > 0, I_C ≻ 0,
triangle inequality}``, i.e. realizability by a non-negative mass density
(Wensing et al. 2017) — see the long comment block below.

The solver enforces ``J_d ≽ LMI_EPS·I₄`` (numerical floor only) and lets the
**soft shape hinge** (``LMI_SHAPE_FRAC`` / ``LMI_SHAPE_WEIGHT``) pull the
solution off that boundary: ``t_d ≤ λ_min(Σ_C,d)`` is affine in π and
``λ_min(Σ_C)`` is concave, so hinging on it stays convex.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from .params import split_joint_params


# Strict positive-definiteness margin of the 4×4 pseudo-inertia (J ≽ eps·I₄,
# min_eig ≥ eps > 0).
# This is a **numerical floor only** (strict PD plus some slack for the
# interior-point solver), not a shape constraint: the link shape is left
# entirely to the soft shape hinge below (LMI_SHAPE_FRAC / LMI_SHAPE_WEIGHT).
# History: before 2026-09-15 there was an additional per-joint relative hard
# margin here,
#   eps_j = max(LMI_EPS, rel · min_eig(J_prior_j))   (rel = 0.1)
# Measured conclusion: it is a **wall**, not a **pull** — the objective is
# almost flat along the near-null-space directions (the singular values of
# the raw regressor Y drop from 1799 to ~1e-15), so the interior-point
# solution **always** stops exactly on that wall (min_eig/ε = 1.0000) and
# only lifts the flatness from 1e-7 to 1.6e-5…3.4e-5; every link's
# pseudo-inertia is still squashed into a thin sheet (triangle inequality
# saturated, so the exported URDF carries near-degenerate inertias).
# It also pushed the true solution out of the feasible set (the true J16/J17
# only keep 28%/17% of the prior slack, which rel ≥ 0.3 would exclude).
# ⇒ the whole layer was replaced by the soft shape hinge (see below) and the
# constants were removed.
LMI_EPS = 1e-7


# --- Soft shape hinge (2026-09-15; replaces the former per-joint relative
#     hard margin) ---
# Turn the hard margin's "wall" into a "pull": introduce one scalar t_d per
# joint plus the affine LMI
#       J_d − t_d·blkdiag(I₃,0) ≽ 0    ⟺    t_d ≤ λ_min(Σ_C,d)
# (Σ_C = Σ_O − h·hᵀ/m is the second moment matrix about the CoM, so
# λ_min(Σ_C) is the tightest of the three triangle-inequality slacks) and add
# a **dimensionless, bounded** hinge term to the objective:
#       W · Σ_d max(0, 1 − t_d / (LMI_SHAPE_FRAC · λ_min(Σ_C,d^prior)))
# i.e. "losing 100% of joint d's prior slack costs at most W".  The solution
# then moves into the interior of the cone by itself instead of passively
# resting on the boundary; the prior slack is conservative (the true J16/J17
# only have 28%/17% of it), so LMI_SHAPE_FRAC < 1 scales the target down to
# the order of the true value and prevents overshoot.
# Measured (sim: exc_arm_1 training / verify_left_arm validation; relative
# flatness slack/mean(eig(Σ_C)) of J15/J16/J17, true values
# 0.456 / 0.030 / 0.030):
#   frac  W     train%  val%   static  param err | J15/J16/J17 rel. flatness
#   —     0     93.09   93.18  0.0485    0.659   | 0.042 0.010 0.009  ← squashed
#   0.5   1     93.09   93.19  0.0481    0.664   | 0.208 0.049 0.042
#   0.3   1     93.10   93.19  0.0483    0.662   | 0.126 0.030 0.020  ← default
#   1.0  ≥0.3   92.92   93.22  0.0476    0.672   | 0.401 0.099 0.088  ← overshoot
# Conclusion: asking for 30–50% of the prior slack is essentially free (the
# training fit is completely unchanged, held-out and static gravity are even
# slightly better) yet pulls the three flattest joints back close to their
# true shape; frac = 1.0 inflates J16/J17 beyond the true value.  W ≥ 0.3
# saturates (the hinge is bounded), so frac is the knob that really matters.
# Note also that at the default setting this soft hinge is **bit-identical**
# to the former hard margin 0.1 (the hinge target 4.7e-5…1.0e-4 lies above
# the old margin 1.6e-5…3.4e-5, so the hard constraint was never active)
# ⇒ removing it cost nothing.  W = 0 falls back to the old behaviour with
# LMI_EPS as the only floor (solution on the cone boundary, links squashed).
LMI_SHAPE_FRAC = 0.3
LMI_SHAPE_WEIGHT = 1.0  # 0 = off (back to the squashed LMI_EPS-floor behaviour)


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
