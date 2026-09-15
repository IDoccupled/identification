"""URDF / rosbag → ``Y_stack``, ``tau_measured``, ``pi_prior`` (sim *and* measured).

Two entry points produce the same dictionary::

    prepare_data_from_urdf(...)   # [sim]  tau = Y_stack @ pi_true (synthetic)
    data_from_measurement(...)    # [meas] tau read from the bag CSV

plus the bag/YAML plumbing they share (``resolve_bag_dir``,
``load_measurement_csv``, ``read_setup_from_bag``, ``recovered_at_times``, …).
This is the only module that touches Pinocchio / pandas / rosbag extracts; the
solver stays pure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import IMU_CSV_TOPIC, PKG_ROOT
from .params import N_PER_JOINT
from .ridge import load_yaml_param_quality


# ============================================================================
# Data preparation
# ============================================================================
def reorder_y_aug(Y: np.ndarray, dof: int) -> np.ndarray:
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


def stack_regressor(reg, q_arm, v_arm, a_arm) -> np.ndarray:
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
        Y_list.append(reorder_y_aug(Y_aug, dof))
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
        tw0, tw1 = parse_window(twin, float(ft.t_array[-1]))
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
    # (``reorder_y_aug`` is defined at module level and shared with
    #  ``data_from_measurement``.)
    Y_list, tau_list = [], []
    for k in range(N):
        Y_aug, *_ = reg.compute_regressor(
            q_traj[:, k],
            v_traj[:, k],
            a_traj[:, k],
            print_info=False,
        )
        Y_aug = reorder_y_aug(Y_aug, dof)
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
def resolve_bag_dir(bag_name: str) -> Path:
    """Locate ``bag_data/<bag_name>`` (accepts a short fragment)."""
    bag_root = PKG_ROOT / "bag_data"
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


def load_measurement_csv(
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

    bag_dir = resolve_bag_dir(bag_name)
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
    """Average the bag readings to get the base-frame gravity vector and the
    fixed waist-joint angle.

    * ``gravity = -mean(IMU linear_acceleration.x/y/z)``: an accelerometer
      measures specific force, so a stationary robot reads ≈ -g (pointing
      along the supporting force) and the gravity direction is the
      **opposite** of the IMU reading.  The whole robot is held still while
      the bag is recorded, so averaging over all samples is enough.
    * ``waist_yaw_offset = mean(joint_state position_12)``: J12_WAIST_YAW is
      held still during the measurement.  Note that this is the mean over
      the **whole recording** — if the waist joint really moved while the
      bag was recorded, the mean is only an approximation; pass
      ``--waist-offset`` explicitly in that case.

    Returns ``(gravity (3,), waist_yaw_offset)``.
    """
    import pandas as pd

    from identification.target_limb_regressor import WAIST_Q_INDICES

    bag_dir = resolve_bag_dir(bag_name)

    # --- gravity: mean of the IMU linear acceleration, negated ---
    imu_path = bag_dir / "csv" / f"{IMU_CSV_TOPIC}.csv"
    if not imu_path.is_file():
        raise FileNotFoundError(
            f"IMU CSV not found: {imu_path} → pass --gravity manually"
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
    gravity = -acc_mean  # gravity points opposite to the IMU reading

    # --- waist joint: mean of the joint_state position column ---
    waist_idx = WAIST_Q_INDICES[0]
    _, q, _, _ = load_measurement_csv(bag_name, verbose=False)
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


def latest_recovered_yaml(exclude: str | None = None) -> str:
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


def yaml_source_bag(trajectory_yaml: str) -> str:
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


def latest_pso_yaml_for_group(group: str, exclude: str | None = None) -> str | None:
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


def yaml_has_quality(yaml_name: str) -> bool:
    """True if a coeffs YAML carries ``_diagnostics.per_param`` quality labels."""
    from identification.fourier_trajectory import FourierTrajectory

    path = FourierTrajectory._coeffs_dir / yaml_name
    if not path.is_file():
        return False
    return bool(load_yaml_param_quality(path))


def recovered_at_times(
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
    ``recovered_at_times``).

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
        trajectory_yaml = latest_recovered_yaml()
    if limb_group is None:
        limb_group = FourierTrajectory.load_group(trajectory_yaml, default="left_arm")
    if bag_name is None:
        bag_name = yaml_source_bag(trajectory_yaml)
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

    # Measurement mode has no ground truth: pi_true stays None (unknown) and
    # only the measured tau drives the identification.
    pi_true = None

    # --- Measurement: read CSV — only the time axis and measured tau ---
    t_meas, q_meas, v_meas, tau_meas = load_measurement_csv(
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
        tw0, tw1 = parse_window(twin, t_sel[-1])
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
    #     at the CSV time samples (phase-aligned, see recovered_at_times).
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
    q_arm, v_arm, a_arm = recovered_at_times(
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
    Y_stack = stack_regressor(reg, q_arm, v_arm, a_arm)  # (N*dof, 13*dof)
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


def parse_window(s: str | None, t_end: float) -> tuple[float, float]:
    """Parse a ``'START:END'`` time window (either end optional); None = full."""
    if s is None:
        return 0.0, t_end
    parts = str(s).split(":")
    if len(parts) > 2:
        raise ValueError(f"Time window should be 'start:end', got: '{s}'")
    t0 = float(parts[0]) if parts[0].strip() else 0.0
    t1 = float(parts[1].strip()) if len(parts) == 2 and parts[1].strip() else t_end
    return t0, t1
