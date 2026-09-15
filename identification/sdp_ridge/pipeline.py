"""The identification pipeline steps (what ``cli.main`` orchestrates).

Each function is one former block of the monolithic ``main()``; the split is
purely structural — the order of the calls, the console output and the numbers
are unchanged.  Data flows through the dict returned by
``load_identification_data`` (``Y_stack`` / ``tau_measured`` / ``pi_prior`` /
``joint_order`` / ``joint_names`` / ``dof``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .config import SIM_GRAVITY, SIM_WAIST_YAW_OFFSET, VAL_YAML_PATH
from .data import (
    data_from_measurement,
    latest_pso_yaml_for_group,
    latest_recovered_yaml,
    prepare_data_from_urdf,
    read_setup_from_bag,
    yaml_has_quality,
    yaml_source_bag,
)
from .lmi import LMI_EPS, build_shape_reference, print_lmi_feasibility
from .metrics import print_cv_summary
from .params import N_PER_JOINT
from .plotting import (
    plot_torque_comparison_measured,
    plot_torque_comparison_simulated,
    plot_torque_comparison_simulated_validation,
)
from .ridge import (
    DEFAULT_QUALITY,
    RIDGE_FREEZE_LOCAL_INDICES,
    RIDGE_QUALITY_WEIGHTS,
    build_freeze_mask,
    build_ridge_weights,
    load_yaml_param_quality,
)
from .solver import SDPSolver
from .static_test import run_static_pose_test
from .urdf_export import write_identified_urdf


def resolve_yamls(args, is_sim):
    """Resolve the trajectory YAML (--yaml or mode default) + quality YAML."""
    from identification.fourier_trajectory import FourierTrajectory

    # ---- resolve the default YAML (bag / limb group follow it) ----
    if is_sim:
        traj_yaml = args.yaml or FourierTrajectory.find_latest_yaml()  # pso_unified
    else:
        traj_yaml = args.yaml or latest_recovered_yaml()

    # ---- quality-label YAML: a recovered (measured) YAML carries no
    #      _diagnostics, so it must point at the pso_unified YAML it was
    #      trained with / that belongs to the same limb group
    #      (--quality-yaml overrides this) ----
    if args.quality_yaml:
        quality_yaml = args.quality_yaml
    elif yaml_has_quality(traj_yaml):
        quality_yaml = traj_yaml  # the trajectory YAML carries labels itself
    else:
        group = FourierTrajectory.load_meta(traj_yaml).get("group") or "left_arm"
        quality_yaml = latest_pso_yaml_for_group(group, exclude=traj_yaml)
        if quality_yaml:
            print(
                f"  [quality] auto-linked pso_unified quality YAML: "
                f"{quality_yaml} (group={group})"
            )
        else:
            print(
                f"  [quality] no pso_unified_*.yaml found for group='{group}'"
                " → every parameter keeps the default tier weight"
            )

    return traj_yaml, quality_yaml


def resolve_setup(args, is_sim, traj_yaml):
    """Gravity + waist offset: bag averages, sim defaults, CLI overrides."""
    # ---- gravity / waist-joint offset ----
    #   meas: the identification bag is given by traj_yaml's
    #         _meta.source_bag and its readings are averaged directly
    #         (gravity = -mean(IMU linear acceleration), waist = mean(position_12)).
    #   sim : a pso_unified trajectory is a simulation excitation without a
    #         source_bag, so no bag can be read → the lying-pose defaults
    #         SIM_GRAVITY (in MuJoCo LINK_BASE is rotated -90° about Y, so the
    #         world gravity (0,0,-9.81) becomes ≈ (-9.81, 0, 0) in the base
    #         frame) and SIM_WAIST_YAW_OFFSET (in the lying pose J12 rests at
    #         zero → 0.0).
    #   --gravity / --waist-offset override this when given (independently).
    gravity: np.ndarray | None = None
    waist_yaw: float | None = None
    setup_src = (
        f"sim lying-pose default (gravity={SIM_GRAVITY.tolist()}, "
        f"waist={SIM_WAIST_YAW_OFFSET:.4f} rad)"
    )
    if is_sim:
        gravity = SIM_GRAVITY.copy()
        waist_yaw = SIM_WAIST_YAW_OFFSET
    else:
        gravity, waist_yaw = read_setup_from_bag(yaml_source_bag(traj_yaml))
        setup_src = "bag average"
    overrides = []
    if args.gravity is not None:
        gravity = np.asarray(args.gravity, dtype=float)
        overrides.append("--gravity")
        print(f"  [setup] --gravity override: {np.round(gravity, 6).tolist()}")
    if args.waist_offset is not None:
        waist_yaw = float(args.waist_offset)
        overrides.append("--waist-offset")
        print(f"  [setup] --waist-offset override: {waist_yaw:.9f} rad")
    if overrides:
        setup_src += " + " + " ".join(overrides)

    return gravity, waist_yaw, setup_src


def print_run_config(
    args,
    is_sim,
    traj_yaml,
    quality_yaml,
    gravity,
    waist_yaw,
    setup_src,
    urdf_ts,
):
    """Echo the effective configuration (mode, YAMLs, ridge/LMI, setup)."""
    # ---- configuration echo ----
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
    print(f"bag             : (auto: from {traj_yaml} _meta.source_bag)")
    print(f"group           : (auto: from {traj_yaml} _meta.group)")
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
            else "(not given → TargetLimbRegressor default (0, 0, -9.81), upright)"
        )
        + f"   [{setup_src}]"
    )
    print(
        "waist_yaw_offset: "
        + (
            f"{waist_yaw:.9f} rad"
            if waist_yaw is not None
            else "(not given → default 0.0 rad)"
        )
    )


def resolve_val_yamls(args, is_sim):
    """Held-out YAMLs: sim = only --val-yaml; meas = default VAL_YAML."""
    # ---- cross-validation trajectory YAMLs (--val-yaml, several allowed) ----
    #   meas: held-out validation on another trajectory/bag (defaults to the
    #         VAL_YAML_PATH constant when not given).
    #   sim : the held-out cross-comparison is only run after identification
    #         when --val-yaml is given explicitly; not given = not run.
    if is_sim:
        val_yamls: list[str] = list(args.val_yaml or [])  # empty → no comparison
    else:
        val_yamls = list(args.val_yaml or [str(VAL_YAML_PATH)])

    return val_yamls


def load_identification_data(args, is_sim, traj_yaml, gravity, waist_yaw):
    """Build Y_stack / tau / pi_prior (synthetic in sim, bag in meas)."""
    # 1. Prepare data
    # ------------------------------------------------------------------
    # sim: tau = Y @ pi_true (synthesised from the URDF) with pi_true known →
    #      error/convergence comparisons are possible.
    # meas: tau is read directly from the bag CSV (torque_<limb> columns) and
    #       the regression state q/v/a comes entirely from the recovered
    #       Fourier trajectory (evaluated directly at the CSV times);
    #       acceleration is not obtained by differentiating the measured
    #       velocity (the CSV has no acceleration column and the quantisation
    #       noise is large).  The measured mode has no ground truth →
    #       pi_true is None.
    if is_sim:
        data = prepare_data_from_urdf(
            urdf_path=args.urdf,  # prior model → regressor / pi_prior / subtree
            yaml_filename=traj_yaml,
            limb_group=None,  # read from the YAML _meta.group
            sample_rate=args.sample_rate,
            time_coeffs=args.time_coeffs,  # Fourier-trajectory playback factor
            urdf_true_path=args.urdf_true,
            gravity=gravity,
            waist_yaw_offset=waist_yaw,
            twin=args.twin_id,  # ID time window (sim and meas; None = full)
        )
    else:
        data = data_from_measurement(
            urdf_path=args.urdf,  # prior model → regressor / pi_prior / subtree
            bag_name=None,  # read from the YAML _meta.source_bag
            limb_group=None,  # read from the YAML _meta.group
            csv_topic=args.csv_topic,  # measured torque
            sample_rate=args.sample_rate,  # decimate the CSV to ~sample_rate Hz
            gravity=gravity,
            waist_yaw_offset=waist_yaw,
            trajectory_yaml=traj_yaml,
            twin=args.twin_id,  # ID time window (one good period; None = full)
            tau_delay=args.tau_delay,  # torque-channel delay (s), see --tau-delay
        )

    return data


def configure_ridge(args, data, quality_yaml):
    """Per-parameter ridge weights c_i + hard-freeze mask (null/small)."""
    from identification.fourier_trajectory import FourierTrajectory

    # 2. Configure weighted-ridge regularization.
    #    Every parameter gets an L2 penalty coefficient from its quality tier
    #    (good quality → small coefficient → may deviate further from the URDF
    #    prior; poor quality → large coefficient → pulled back harder), built
    #    from the quality labels by build_ridge_weights.  The link shape is
    #    handled separately by the soft hinge of step 2c (see LMI_SHAPE_FRAC);
    #    the LMI itself keeps only the numerical floor LMI_EPS.
    quality_map = None
    if quality_yaml is not None:
        quality_path = FourierTrajectory._coeffs_dir / quality_yaml
        if quality_path.is_file():
            quality_map = load_yaml_param_quality(quality_path)
    freeze_mask = build_freeze_mask(quality_map, dof=int(data["dof"]))
    # freeze_mask is the **single source of truth**: a frozen parameter also
    # drops out of the ridge norm (c = 0).  Conversely c = 0 without freezing
    # would mean "no penalty and free to move", which the solver rejects.
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
            "  no _diagnostics.per_param quality labels → every remaining "
            f"parameter gets the default tier '{DEFAULT_QUALITY}' ridge penalty"
        )
    c_eff = ridge_weights.copy()
    c_eff[freeze_mask] = np.nan
    print(
        f"  ridge_lambda={args.ridge_lambda}: active c range "
        f"[{np.nanmin(c_eff):.4g}, {np.nanmax(c_eff):.4g}] "
        f"(hard-frozen: {n_freeze})"
    )

    return quality_map, freeze_mask, ridge_weights, dof_


def configure_shape(args, data, dof_):
    """Soft shape-hinge reference (FRAC x prior triangle slack) + weight."""
    # 2b. The physical-consistency LMI now keeps only the numerical floor
    #     LMI_EPS (strict PD): the shape is no longer held by a hard margin but
    #     entirely by the soft hinge of step 2c (see the table at
    #     LMI_SHAPE_FRAC).
    print(
        f"  LMI floor: uniform eps = {LMI_EPS:.1g} (strict PD only; "
        f"link shape comes from the soft hinge below)"
    )

    # 2c. Soft shape hinge: turn the hard margin's "wall" into a "pull".  The
    #     reference is FRAC × the prior's own triangle-inequality slack
    #     λ_min(Σ_C^prior); with W = 0 the term does not enter the problem at
    #     all (back to the old LMI_EPS-floor behaviour: the solution rests on
    #     the cone boundary and the links are squashed).
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
            f"  shape hinge OFF (W=0) → the solution rests on the LMI cone "
            f"boundary and the links are squashed; "
            f"target would be {args.lmi_shape_frac:g}×prior slack = "
            f"[{shape_ref.min():.4g}, {shape_ref.max():.4g}] (--lmi-shape-weight 1)"
        )

    return shape_weight, shape_ref


def run_solver(data, ridge_weights, freeze_mask, shape_ref, shape_weight):
    """Run the single all-joints SDP and return (solver, result)."""
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

    return solver, result


def report_results(solver, result, data, quality_map, is_sim, shape_ref):
    """Print the per-parameter, torque-RMSE and LMI feasibility tables."""
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

    # 4b. Check the identified inertial parameters for physical consistency
    #     (pseudo-inertia LMI) and print the table.
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


def export_identified_urdf(
    args,
    result,
    data,
    traj_yaml,
    quality_yaml,
    gravity,
    waist_yaw,
    shape_weight,
    shape_ref,
    is_sim,
    urdf_ts,
):
    """Write the identified URDF next to the prior one (unless disabled)."""
    # 4c. Export the identified URDF — write the identified link inertias
    #     (mass / CoM / inertia about the CoM) and joint dynamics
    #     (armature/damping/friction) back into a copy of the prior URDF; all
    #     remaining parts (meshes, limits, environment links, comments) are
    #     preserved verbatim.  The file name carries a timestamp for
    #     uniqueness:
    #       <prior stem>_<YYMMDD_HHMMSS>.urdf
    #     It lands in the prior URDF's directory (override with
    #     --urdf-out-dir, skip with --no-save-urdf).
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


def run_static_test(args, result, data):
    """Static-pose gravity test (--static-test, sim only)."""
    # 4d. Static-pose gravity test (--static-test, sim only) — sample random
    #     collision-free, within-limit static poses (v = a = 0 ⇒ only the
    #     gravity term survives) and compare the joint torques of the true URDF
    #     with those computed from the identified parameters, drawn as a scatter
    #     whose y axis is centred on 0 (one column per pose, one point per joint
    #     vertically; zero error sits exactly in the middle).  Both use the
    #     **same regressor** (the prior URDF geometry), matching how the
    #     training torque is synthesised ⇒ the difference comes purely from the
    #     parameter error.
    if args.static_test:
        print("\n" + "=" * 100)
        print("STATIC-POSE GRAVITY TEST (--static-test)".center(100))
        print("=" * 100)
        run_static_pose_test(
            reg=data["reg"],
            pi_true=data["pi_true"],
            pi_identified=result.pi_identified,
            pi_prior=data["pi_prior"],  # overlay the URDF prior error too
            joint_names=data["joint_names"],
            joint_order=result.joint_order,
            n_poses=args.static_poses,
            seed=args.static_seed,
            plot=not args.no_plot,
        )


def run_validation_plots(
    args,
    result,
    data,
    is_sim,
    val_yamls,
    gravity,
    waist_yaw,
):
    """Training plot (sim) / bag-wise measured cross-validation + summary."""
    # 5. Cross-validation / comparison plot
    #    sim: true vs prior vs identified on the training trajectory.
    #    meas: validate the identification on another trajectory/bag — pi_prior
    #          and pi_identified each produce joint torques on the validation
    #          trajectory, compared against the measured tau of the validation
    #          bag; the validation bag is taken from the validation YAML's
    #          _meta.source_bag.
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
            # Per validation trajectory: each yaml's bag contributes its **own**
            # base-frame gravity / waist angle (different recording sessions
            # have different poses, so the identification bag's values are not
            # reused); explicit --gravity / --waist-offset overrides are kept.
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
                    note = yaml_source_bag(val_yaml)
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
                        urdf_path=args.urdf,  # prior model (regressor)
                        val_yaml=val_yaml,
                        val_bag_name=None,  # read from the val YAML source_bag
                        joint_names=data["joint_names"],
                        limb_group=None,
                        csv_topic=args.csv_topic,
                        sample_rate=args.sample_rate,
                        gravity=val_gravity,
                        waist_yaw_offset=val_waist,
                        tau_delay=args.tau_delay,  # same delay as in the ID
                        twin=args.twin,  # time-window zoom, e.g. "0:13.4"
                    )
                    cv_rows.append(
                        {
                            "yaml": Path(val_yaml).name,
                            "note": note,
                            "stats": out["stats"],
                        }
                    )
                except Exception as exc:
                    # One bad yaml (no source_bag / different limb group /
                    # missing bag) must not kill the whole multi-trajectory run
                    # — report it and continue with the next one.
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


def cross_validate_sim_heldout(
    args,
    result,
    data,
    is_sim,
    val_yamls,
    gravity,
    waist_yaw,
):
    """Sim held-out cross-validation on extra --val-yaml yamls."""
    from identification.fourier_trajectory import FourierTrajectory

    # 5b. sim held-out cross-validation (optional) — only run when --val-yaml
    #     was given explicitly (one or several *different* trajectories):
    #     recompute the true torque on each of them with the same pi_true and
    #     check whether identified still beats prior on an unseen trajectory
    #     (Improve %) — i.e. does the identification generalise or has it merely
    #     memorised the training data?  Without --val-yaml it is skipped.  The
    #     RMSE table is printed even with --no-plot, and several yamls are
    #     summarised in one cross-yaml table.
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
                    limb_group=None,  # val YAML _meta.group (same dof as the ID)
                    sample_rate=args.sample_rate,
                    time_coeffs=args.time_coeffs,
                    gravity=gravity,
                    waist_yaw_offset=waist_yaw,
                    twin=args.twin,  # time-window zoom, e.g. "0:13.4"
                    plot=not args.no_plot,
                )
                cv_rows.append(
                    {"yaml": Path(val_yaml).name, "note": note, "stats": out["stats"]}
                )
            except Exception as exc:
                # One bad yaml (different limb group / dof mismatch) must not
                # kill the whole batch.
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


def print_finished():
    """Final banner (keeps the console layout of the old main)."""
    print("\n" + "=" * 100)
    print("SDP identification finished.".center(100))
    print("=" * 100 + "\n")
