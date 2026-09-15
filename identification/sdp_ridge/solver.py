"""The pure all-joints SDP/SOCP solver (no URDF, no Pinocchio, no regressor).

``SDPSolver.solve()`` takes pre-computed data (``Y_stack``, ``tau_measured``,
``pi_prior``, ridge weights, freeze mask) and returns an
``IdentificationResult``.  Regularisation and the physical-consistency / shape
constraints are described in :mod:`identification.sdp_ridge.lmi` and
:mod:`identification.sdp_ridge.ridge`.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from .lmi import (
    LMI_EPS,
    LMI_SHAPE_B,
    LMI_SHAPE_FRAC,
    LMI_SHAPE_WEIGHT,
    build_pseudo_inertia_LMI,
    check_lmi_feasibility,
)
from .metrics import print_rmse_comparison
from .params import N_PER_JOINT, PARAM_LABELS, IdentificationResult, split_joint_params
from .ridge import RIDGE_FREEZE_QUALITIES, RIDGE_SCALE_FLOOR


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

        ``is_sim=True`` (ground truth known): the ``Err%`` column is
        replaced by a before→after comparison ``prior→identified`` of the
        error relative to the ground truth, e.g. ``+5.00% -> +3.00%``;
        parameters of the ``small`` / ``null`` tier are shown as ``N/A``.
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
                        f"{PARAM_LABELS[i]:<10s} {pr:>12.6g} {ident:>12.6g} "
                        f"{ref:>12.6g} {err_str:>20s} {dp:>7.2f}% {q:>10s}"
                    )
                elif has_ref:
                    ref = pi_ref[d][i]
                    denom = abs(ref) if abs(ref) > 1e-12 else 1.0
                    err = (ident - ref) / denom * 100
                    print(
                        f"{PARAM_LABELS[i]:<10s} {pr:>12.6g} {ident:>12.6g} "
                        f"{ref:>12.6g} {err:>7.2f}% {dp:>7.2f}% {q:>10s}"
                    )
                else:
                    print(
                        f"{PARAM_LABELS[i]:<10s} {pr:>12.6g} {ident:>12.6g} "
                        f"{dp:>7.2f}% {q:>10s}"
                    )

        print("\n" + "-" * 90)
        print(f"Total solve time: {sum(result.joint_solve_times):.3f}s")

        if Y_stack is not None and tau_measured is not None:
            print_rmse_comparison(result, joint_names, Y_stack, tau_measured)
        else:
            # Backward-compatible fallback if torque data isn't available
            print(f"Σ ‖τ_residual‖₂: {sum(result.joint_objectives):.6g}")
