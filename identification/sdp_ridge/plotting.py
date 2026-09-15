"""Torque-comparison figures (training / measured validation / sim held-out).

Every entry point returns ``{"stats": …, "figures": […]}"`` (or a figure list)
and draws **one figure per joint** with a residual sub-panel; ``twin`` is an
optional ``'START:END'`` time window for the plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import (
    load_measurement_csv,
    parse_window,
    recovered_at_times,
    stack_regressor,
    yaml_source_bag,
)
from .metrics import print_rmse_comparison
from .params import IdentificationResult


# ============================================================================
# Torque comparison plot
# ============================================================================


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
    t0, t1 = parse_window(twin, t[-1])
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
    waist_yaw_offset: float | None = None,
    grid_sample_rate: float = 500.0,
    tau_delay: float = 0.0,
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
    #         print(f"{PARAM_LABELS[i]:<10s} {pr:>12.6g} {idn:>12.6g} {dp:>8.2f}%")

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
        val_bag_name = yaml_source_bag(val_yaml)
        if verbose:
            print(f"  [validation] val_bag from yaml _meta.source_bag: {val_bag_name}")
    t_val, _, _, tau_val = load_measurement_csv(
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
    q_arm, v_arm, a_arm = recovered_at_times(
        t_sel - float(tau_delay), dof, val_yaml, grid_sample_rate=grid_sample_rate
    )
    if verbose:
        print(f"  [validation] q/v/a from Fourier trajectory {val_yaml}")

    # --- 5) Regressor + predicted torques (prior & identified) ---
    Y_val = stack_regressor(reg, q_arm, v_arm, a_arm)  # (N*dof, 13*dof)
    tau_measured = tau_arm.reshape(-1)  # (N*dof,)
    tau_prior = Y_val @ result.pi_prior
    tau_ident = Y_val @ result.pi_identified
    if verbose:
        print(
            f"  [validation] Y_val: {Y_val.shape}, tau (measured): {tau_measured.shape}"
        )

    # --- 6) RMSE table on the held-out data ---
    stats = print_rmse_comparison(result, joint_names, Y_val, tau_measured)

    # --- 7) Plot ---
    figures = _plot_torque_comparison_panels(
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
    return {"stats": stats, "figures": figures}


def plot_torque_comparison_simulated_validation(
    result: IdentificationResult,
    urdf_path: str | Path,
    val_yaml: str,
    pi_true: np.ndarray,
    joint_names: list[str] | None = None,
    limb_group: str | None = None,
    sample_rate: float = 100.0,
    time_coeffs: float = 1.0,
    gravity: np.ndarray | None = None,
    waist_yaw_offset: float | None = None,
    offset: float | None = None,
    show_residual: bool = True,
    verbose: bool = True,
    twin: str | None = None,
    plot: bool = True,
):
    """Held-out cross-validation of sim identification on a different YAML.

    The sim identification was fit on one (training) excitation YAML whose
    synthetic torque came from the true URDF.  This stage is the sim analogue
    of ``plot_torque_comparison_measured`` (which validates on a different
    measured bag): it re-runs **both** the URDF prior and the identified
    parameters on a *different* (manually specified, e.g. ``verify_*.yaml``)
    trajectory, and compares their predicted torques against the true torque
    ``tau_true = Y_val @ pi_true`` synthesised from the **same true model** on
    that new trajectory:

        1. Build the URDF regressor (prior model) at the validation
           trajectory's q/v/a (same sample_rate / time_coeffs playback as
           identification).
        2. ``tau_true  = Y_val @ pi_true``
           ``tau_prior = Y_val @ pi_prior``
           ``tau_ident = Y_val @ pi_identified``.
        3. Per-joint RMSE table (prior vs identified) on the held-out data and
           per-joint plots — so you can see whether identification actually
           *generalises* (does it still beat the URDF prior on a trajectory it
           never saw?), as opposed to merely memorising the training data.

    ``plot=False`` prints only the RMSE table (no figures).
    """
    from identification.fourier_trajectory import FourierTrajectory
    from identification.target_limb_regressor import TargetLimbRegressor

    # The validation regressor must belong to the identified limb (same dof).
    # If not given, it is read from the validation yaml's _meta.group.
    if limb_group is None:
        limb_group = FourierTrajectory.load_group(val_yaml, default="left_arm")

    dof = len(result.joint_order)
    if joint_names is None:
        joint_names = [f"joint_{d}" for d in range(dof)]

    # --- 1) Regressor: same prior (URDF) model as identification ---
    reg = TargetLimbRegressor(
        urdf_path=Path(urdf_path),
        group_to_identify=limb_group,
        print_info=False,
        gravity=gravity,
        waist_yaw_offset=waist_yaw_offset,
    )
    assert reg.dof == dof, (
        f"limb_group '{limb_group}' dof={reg.dof} != identification dof={dof} "
        f"— the validation yaml must belong to the same limb group"
    )

    # --- 2) Generate the held-out trajectory (same playback convention as the
    #        identification data, i.e. same sample_rate / time_coeffs). ---
    ft = FourierTrajectory(dim=dof, sample_rate=sample_rate, time_coeffs=time_coeffs)
    q_val, v_val, a_val = ft.generate_trajectory_from_yaml(Path(val_yaml).name)
    N = q_val.shape[1]
    if verbose:
        print(
            f"  [sim validation] held-out yaml={Path(val_yaml).name}  "
            f"N={N} steps  (time_coeffs={time_coeffs}, "
            f"period={ft.duration:.3f}s)"
        )

    # --- 3) Stack regressor + reference true torque on the new trajectory.
    #        Y depends only on kinematics/geometry, so the true torque is
    #        Y_val @ pi_true (pi_true from the true URDF) — same as the
    #        training-data preparation. ---
    Y_val = stack_regressor(reg, q_val.T, v_val.T, a_val.T)  # (N*dof, 13*dof)
    tau_true = Y_val @ np.asarray(pi_true)
    if verbose:
        print(f"  [sim validation] Y_val: {Y_val.shape}, tau_true: {tau_true.shape}")

    # --- 4) RMSE table on the held-out data (prior vs identified) ---
    stats = print_rmse_comparison(result, joint_names, Y_val, tau_true)

    # --- 5) Optional plot ---
    figures = []
    if plot:
        figures = _plot_torque_comparison_panels(
            tau_true,
            Y_val @ result.pi_prior,
            Y_val @ result.pi_identified,
            joint_names,
            result.joint_order,
            dof,
            sample_rate=sample_rate,
            offset=offset,
            show_residual=show_residual,
            true_label="true",
            title=(
                "Joint Torque Comparison (sim held-out validation, different yaml): "
                "true vs prior vs identified\n"
                f"held-out yaml={Path(val_yaml).name}   "
                "(identified on a different training yaml)"
            ),
            twin=twin,
        )
    return {"stats": stats, "figures": figures}
