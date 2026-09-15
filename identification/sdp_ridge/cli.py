"""Command-line interface: ``build_parser()`` + the thin ``main()``.

The actual work lives in :mod:`identification.sdp_ridge.pipeline` — ``main``
only resolves the mode, then calls the steps in order::

    resolve_yamls → resolve_setup → print_run_config → resolve_val_yamls
    → load_identification_data → configure_ridge → configure_shape
    → run_solver → report_results → export_identified_urdf
    → run_static_test → run_validation_plots → cross_validate_sim_heldout

Run:  ``python -m identification.sdp_ridge --sim --val-yaml <name>.yaml``
(needs ``source install/setup.bash`` from the workspace root and an interpreter
with cvxpy + MOSEK, e.g. ``/home/xiaoran/venv/venv_identify/bin/python``).
"""

from __future__ import annotations

import argparse
from datetime import datetime

from .config import (
    DEFAULT_URDF_PATH,
    QUALITY_YAML_PATH,
    TAU_DELAY,
    TRAJ_YAML_PATH,
    TRUE_URDF_PATH,
)
from .lmi import LMI_SHAPE_FRAC, LMI_SHAPE_WEIGHT
from .pipeline import (
    configure_ridge,
    configure_shape,
    cross_validate_sim_heldout,
    export_identified_urdf,
    load_identification_data,
    print_finished,
    print_run_config,
    report_results,
    resolve_setup,
    resolve_val_yamls,
    resolve_yamls,
    run_solver,
    run_static_test,
    run_validation_plots,
)
from .ridge import RIDGE_LAMBDA
from .static_test import STATIC_TEST_POSES, STATIC_TEST_SEED


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m identification.sdp_ridge",
        description=(
            "Global SDP parameter identification (all joints identified in a "
            "single solve, no sequential per-joint loop): by default the "
            "measured bag torque is used (meas); add --sim to identify on "
            "URDF-synthesised torque instead."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- mode / data source ----
    ap.add_argument(
        "--sim",
        action="store_true",
        help="Simulation mode: tau is synthesised from the URDF "
        "(prepare_data_from_urdf) and pi_true is known; without --sim the "
        "measured mode is used (data_from_measurement, default)",
    )
    ap.add_argument(
        "--urdf",
        default=str(DEFAULT_URDF_PATH),
        help="Prior/initial URDF (regressor + pi_prior + subtree mask)",
    )
    ap.add_argument(
        "--urdf-true",
        default=str(TRUE_URDF_PATH),
        help="[sim] 'true' URDF used to synthesise tau_measured",
    )
    ap.add_argument(
        "--yaml",
        "-y",
        default=TRAJ_YAML_PATH,
        metavar="NAME",
        help="Trajectory-coefficient YAML under trajectory_coefficients/ (the "
        "bag and limb group are read from this YAML); default: meas = latest "
        "recovered_*.yaml, sim = latest pso_unified_*.yaml",
    )
    ap.add_argument(
        "--quality-yaml",
        "-q-y",
        default=QUALITY_YAML_PATH,
        metavar="NAME",
        help="pso_unified YAML carrying the _diagnostics.per_param quality "
        "labels: each parameter's quality tier sets its ridge weight (L2 pull "
        "toward the prior) — the better the quality, the smaller the weight "
        "and the further the parameter may deviate from the prior; the worse "
        "the quality, the stronger the pull-back.  The measured "
        "recovered_*.yaml carries no labels, so this must point at the "
        "pso_unified YAML it was trained with.  Default: use --yaml itself if "
        "it carries labels, otherwise the latest pso_unified_*.yaml with the "
        "same _meta.group; if none is found, every parameter keeps the "
        "default tier weight",
    )
    ap.add_argument(
        "--ridge-lambda",
        type=float,
        default=RIDGE_LAMBDA,
        help="Global ridge strength, multiplying every quality-tier weight "
        "(0 = pure data fit, no penalty for deviating from the prior; larger "
        "values pull the parameters back toward the URDF prior)",
    )
    ap.add_argument(
        "--lmi-shape-weight",
        type=float,
        default=LMI_SHAPE_WEIGHT,
        metavar="W",
        help="Weight of the soft shape hinge, which adds "
        "W·Σ_d max(0, 1 − λ_min(Σ_C,d)/(FRAC·λ_min(Σ_C,d^prior))) to the "
        "objective.  Physical consistency then keeps only the numerical floor "
        "LMI_EPS (strict PD), and the link shape is produced entirely by this "
        "term: the solution moves into the interior of the cone on its own "
        "instead of resting on the boundary and being squashed.  "
        "0 = off (back to the squashed on-the-cone-boundary behaviour); it "
        "saturates at ≥0.3, default 1.0",
    )
    ap.add_argument(
        "--lmi-shape-frac",
        type=float,
        default=LMI_SHAPE_FRAC,
        metavar="FRAC",
        help="Soft-hinge target = FRAC × the prior's own triangle-inequality "
        "slack λ_min(Σ_C^prior).  Measured: 0.3 is essentially free (training "
        "fit unchanged, validation/static slightly better) and already pulls "
        "the flattest joints J15/J16/J17 close to the truth; 0.5 is free too; "
        "1.0 overshoots (inflates J16/J17 beyond the true value)",
    )
    ap.add_argument(
        "--csv-topic",
        default="hardware_joint_state",
        help="[meas] CSV topic under <bag>/csv/ (holds the measured torque "
        "columns)",
    )
    ap.add_argument(
        "--sample-rate",
        type=float,
        default=100.0,
        help="[meas] decimate the CSV (~500 Hz) to ~sample_rate Hz for the "
        "regression samples",
    )
    ap.add_argument(
        "--time-coeffs",
        "-t",
        type=float,
        default=1.0,
        help="[sim] Fourier-trajectory playback factor (>1 speeds up, <1 slows "
        "down; default 1.0 = nominal speed, physical period = "
        "TRAJ_PERIOD/time_coeffs)",
    )
    ap.add_argument(
        "--tau-delay",
        type=float,
        default=TAU_DELAY,
        metavar="SECONDS",
        help="[meas] Delay compensation of the measured torque channel: "
        "torque(t) is assumed to describe the state at t−τ, so the regressor "
        "is evaluated at (t−τ).  >0 = the torque lags the state (the usual "
        "transport/filter delay), negative = the torque leads the state.  A "
        "pure time shift turns the armature column into cos(ωτ)·q̈ + "
        "ω·sin(ωτ)·q̇, i.e. it trades weight between armature and damping.  "
        "Default 0 = no compensation",
    )

    # ---- model / environment ----
    ap.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Gravity vector in the base frame ([meas] default: negated mean "
        "of the bag IMU readings; [sim] default: (-9.81, 0, 0) for the MuJoCo "
        "lying-pose model)",
    )
    ap.add_argument(
        "--waist-offset",
        type=float,
        default=None,
        help="Fixed angle of the waist joint J12_WAIST_YAW (rad) ([meas] "
        "default: mean of joint_state position_12; [sim] default: 0.0 for the "
        "lying-pose model)",
    )

    # ---- validation / plotting ----
    ap.add_argument(
        "--val-yaml",
        "-v-y",
        default=None,
        nargs="+",
        metavar="NAME",
        help="Trajectory YAMLs for cross-validation (**several** may be "
        "given, space separated; each must be a different trajectory from "
        "--yaml):\n"
        "  [sim]  each is treated as a held-out trajectory after "
        "identification: the true torque is recomputed with the same "
        "pi_true, and the prior vs identified RMSE improvement is printed "
        "to show whether the identification generalises; with several "
        "yamls a single cross-yaml table is summarised at the end.  "
        "Omitted → no cross-comparison\n"
        "  [meas] each yaml is validated on the measured data of the bag "
        "named by its _meta.source_bag (each using its **own** bag's IMU "
        "gravity / waist angle); when omitted, the VAL_YAML_PATH constant "
        "of config.py is used",
    )
    ap.add_argument(
        "-w",
        "--twin",
        default=None,
        metavar="START:END",
        help="[plot] time-window zoom of the comparison figures, e.g. "
        "'0:13.4' (either end may be omitted; None = full length)",
    )
    ap.add_argument(
        "--twin-id",
        default=None,
        metavar="START:END",
        help="Time window used for identification: manually pick a section "
        "with good sampling quality (works for both sim and meas; the "
        "trajectory is periodic, so one good period is enough, e.g. "
        "'10:16.667'; None = full length)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the torque comparison figures",
    )
    ap.add_argument(
        "--static-test",
        "-static",
        action="store_true",
        help="[--sim only] additional static-pose gravity test: sample N "
        "random collision-free, within-limit static poses (v = a = 0 ⇒ only "
        "the gravity term survives) and compare the joint torques of the true "
        "URDF with those of the identified parameters, drawn as a scatter "
        "whose y axis is centred on 0 (one column per pose, one point per "
        "joint)",
    )
    ap.add_argument(
        "--static-poses",
        type=int,
        default=STATIC_TEST_POSES,
        metavar="N",
        help="[--static-test] number of random static poses",
    )
    ap.add_argument(
        "--static-seed",
        type=int,
        default=STATIC_TEST_SEED,
        metavar="SEED",
        help="[--static-test] RNG seed of the pose sampling (fixed ⇒ the same "
        "poses every run)",
    )

    # ---- result export ----
    ap.add_argument(
        "--no-save-urdf",
        "-no",
        action="store_true",
        help="Do not write the identified parameters out as a URDF (default: "
        "write)",
    )
    ap.add_argument(
        "--urdf-out-dir",
        default=None,
        metavar="DIR",
        help="Output directory of the identified URDF; defaults to the "
        "directory of --urdf (i.e. the prior URDF path), file name = "
        "<prior stem>_<YYMMDD_HHMMSS>.urdf (a name collision appends "
        "_1/_2…)",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the CLI and run the identification pipeline step by step.

    The steps themselves live in :mod:`identification.sdp_ridge.pipeline`; this
    function only resolves the mode and calls them in order.
    """
    args = build_parser().parse_args(argv)

    is_sim = args.sim
    if args.static_test and not is_sim:
        raise SystemExit(
            "--static-test is only available in --sim mode (the meas mode "
            "has no pi_true, so the joint torques of the \"actual URDF\" "
            "cannot be computed)"
        )

    # Timestamp of the result URDF (one per run so the banner and the file
    # agree; a name collision appends _1/_2…).
    urdf_ts = datetime.now().strftime("%y%m%d_%H%M%S")

    traj_yaml, quality_yaml = resolve_yamls(args, is_sim)
    gravity, waist_yaw, setup_src = resolve_setup(args, is_sim, traj_yaml)
    print_run_config(
        args,
        is_sim,
        traj_yaml,
        quality_yaml,
        gravity,
        waist_yaw,
        setup_src,
        urdf_ts,
    )
    val_yamls = resolve_val_yamls(args, is_sim)

    data = load_identification_data(args, is_sim, traj_yaml, gravity, waist_yaw)
    quality_map, freeze_mask, ridge_weights, dof = configure_ridge(
        args, data, quality_yaml
    )
    shape_weight, shape_ref = configure_shape(args, data, dof)
    solver, result = run_solver(
        data, ridge_weights, freeze_mask, shape_ref, shape_weight
    )
    report_results(solver, result, data, quality_map, is_sim, shape_ref)
    export_identified_urdf(
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
    )
    run_static_test(args, result, data)
    run_validation_plots(args, result, data, is_sim, val_yamls, gravity, waist_yaw)
    cross_validate_sim_heldout(
        args, result, data, is_sim, val_yamls, gravity, waist_yaw
    )
    print_finished()
