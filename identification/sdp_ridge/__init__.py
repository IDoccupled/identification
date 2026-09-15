"""Global SDP identification for serial robot limbs (weighted ridge variant).

This package is the former single-file ``sdp_solver_alljoints_ridge.py``
(~3700 lines), split into focused modules.  Architecture (unchanged)::

  ┌─────────────────────────────────────────────────────┐
  │  Data preparation (data.py / pipeline.py)           │
  │  · FourierTrajectory  →  q(t), v(t), a(t)           │
  │  · TargetLimbRegressor  →  Y_aug, pi_prior          │
  │  · tau_measured = Y_stack @ pi_true (or real data)  │
  └──────────────────────┬──────────────────────────────┘
                         │  Y_stack, tau, pi_prior,
                         │  subtree_mask, joint_order
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  SDPSolver (solver.py — pure, stateless)            │
  │  · Single all-joints SOCP/SDP (all joints at once)  │
  │  · 4×4 pseudo-inertia LMI + soft shape hinge (lmi)  │
  │  · Quality-weighted ridge toward the URDF prior     │
  └─────────────────────────────────────────────────────┘

Module map
----------
``config``        paths / global constants (URDF, YAML, SIM_GRAVITY, TAU_DELAY …)
``params``        13-parameter layout + ``IdentificationResult``
``lmi``           4×4 pseudo-inertia LMI, triangle-inequality slack, soft
                  shape hinge, feasibility report
``ridge``         quality-label YAML → ridge weights c and hard-freeze mask
``solver``        ``SDPSolver`` (pure solver; no URDF / Pinocchio)
``metrics``       torque RMSE tables + multi-yaml cross-validation summary
``data``          URDF / bag → ``Y_stack``, ``tau``, ``pi_prior`` (sim/meas)
``plotting``      torque-comparison figures (train / meas / sim held-out)
``static_test``   ``--static-test`` static-pose gravity test
``urdf_export``   write the identified parameters back out as a URDF
``pipeline``      the step functions of ``main`` (data → ridge → solve →
                  report → export)
``cli``           argparse parser + ``main()`` orchestration

Run ``python -m identification.sdp_ridge --help`` (needs ``source
install/setup.bash`` and an interpreter with cvxpy + MOSEK).

NOTE: the tunables now live in the module that uses them (``RIDGE_*`` in
``ridge.py``, ``LMI_SHAPE_*`` in ``lmi.py``, paths in ``config.py``).
Rebinding ``identification.sdp_ridge.RIDGE_LAMBDA`` only changes the copy
re-exported below — patch ``identification.sdp_ridge.ridge.RIDGE_LAMBDA``.
"""

from __future__ import annotations

from .cli import build_parser, main
from .config import (
    DEFAULT_URDF_PATH,
    IMU_CSV_TOPIC,
    PKG_ROOT,
    QUALITY_YAML_PATH,
    SIM_GRAVITY,
    SIM_WAIST_YAW_OFFSET,
    TAU_DELAY,
    TRAJ_YAML_PATH,
    TRUE_URDF_PATH,
    VAL_YAML_PATH,
)
from .data import (
    data_from_measurement,
    load_measurement_csv,
    prepare_data_from_urdf,
    read_setup_from_bag,
    recovered_at_times,
    resolve_bag_dir,
    reorder_y_aug,
    stack_regressor,
    yaml_source_bag,
)
from .lmi import (
    LMI_EPS,
    LMI_SHAPE_FRAC,
    LMI_SHAPE_WEIGHT,
    build_pseudo_inertia_LMI,
    build_shape_reference,
    check_lmi_feasibility,
    print_lmi_feasibility,
    relative_flatness,
    second_moment_about_com,
    triangle_slack,
)
from .metrics import print_cv_summary, print_rmse_comparison
from .params import (
    N_PER_JOINT,
    PARAM_LABELS,
    IdentificationResult,
    join_joint_params,
    split_joint_params,
)
from .plotting import (
    plot_torque_comparison_measured,
    plot_torque_comparison_simulated,
    plot_torque_comparison_simulated_validation,
)
from .ridge import (
    DEFAULT_QUALITY,
    RIDGE_FREEZE_LOCAL_INDICES,
    RIDGE_FREEZE_QUALITIES,
    RIDGE_LAMBDA,
    RIDGE_QUALITY_WEIGHTS,
    RIDGE_SCALE_FLOOR,
    build_freeze_mask,
    build_ridge_weights,
    load_yaml_param_quality,
)
from .solver import SDPSolver
from .static_test import (
    STATIC_TEST_POSES,
    STATIC_TEST_SEED,
    run_static_pose_test,
    sample_static_poses,
)
from .urdf_export import (
    URDF_NUM_DECIMALS,
    dynamic_params_to_inertial,
    write_identified_urdf,
)

__all__ = [
    # config
    "DEFAULT_URDF_PATH",
    "IMU_CSV_TOPIC",
    "PKG_ROOT",
    "QUALITY_YAML_PATH",
    "SIM_GRAVITY",
    "SIM_WAIST_YAW_OFFSET",
    "TAU_DELAY",
    "TRAJ_YAML_PATH",
    "TRUE_URDF_PATH",
    "VAL_YAML_PATH",
    # params
    "N_PER_JOINT",
    "PARAM_LABELS",
    "IdentificationResult",
    "join_joint_params",
    "split_joint_params",
    # lmi
    "LMI_EPS",
    "LMI_SHAPE_FRAC",
    "LMI_SHAPE_WEIGHT",
    "build_pseudo_inertia_LMI",
    "build_shape_reference",
    "check_lmi_feasibility",
    "print_lmi_feasibility",
    "relative_flatness",
    "second_moment_about_com",
    "triangle_slack",
    # ridge
    "DEFAULT_QUALITY",
    "RIDGE_FREEZE_LOCAL_INDICES",
    "RIDGE_FREEZE_QUALITIES",
    "RIDGE_LAMBDA",
    "RIDGE_QUALITY_WEIGHTS",
    "RIDGE_SCALE_FLOOR",
    "build_freeze_mask",
    "build_ridge_weights",
    "load_yaml_param_quality",
    # solver / metrics
    "SDPSolver",
    "print_cv_summary",
    "print_rmse_comparison",
    # data
    "data_from_measurement",
    "load_measurement_csv",
    "prepare_data_from_urdf",
    "read_setup_from_bag",
    "recovered_at_times",
    "reorder_y_aug",
    "resolve_bag_dir",
    "stack_regressor",
    "yaml_source_bag",
    # plotting
    "plot_torque_comparison_measured",
    "plot_torque_comparison_simulated",
    "plot_torque_comparison_simulated_validation",
    # static test
    "STATIC_TEST_POSES",
    "STATIC_TEST_SEED",
    "run_static_pose_test",
    "sample_static_poses",
    # urdf export
    "URDF_NUM_DECIMALS",
    "dynamic_params_to_inertial",
    "write_identified_urdf",
    # cli
    "build_parser",
    "main",
]
