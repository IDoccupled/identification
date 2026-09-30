"""Torque-comparison figures (training / measured validation / sim held-out).

Every entry point returns ``{"stats": …, "figures": […]}"`` (or a figure list)
and draws the torque + residual panels; ``twin`` is an optional ``'START:END'``
time window for the plots (the plotted time axis is re-based on the window
start, so a zoomed segment starts at 0 s).

``layout`` selects the arrangement of the panels:

* ``"per-joint"`` — one figure per joint (default; unchanged old behaviour);
* ``"olympic"``   — all joints in one figure, 3-over-2 stagger for a 5-DoF arm
  (like the Olympic rings) or a plain 2×3 grid for a 6-DoF leg;
* ``"row"``       — all joints in one figure, one column per joint.
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
from .metrics import _rmse_comparison, print_rmse_comparison
from .params import IdentificationResult


# ============================================================================
# Torque comparison plot
# ============================================================================


def _torque_layout_axes(fig, dof: int, show_residual: bool, layout: str) -> list:
    """Build the sub-axes grid for the combined torque-comparison layouts.

    Returns ``(ax, ax_r)`` pairs — one per joint, in the same order as
    ``joint_order`` (``ax_r`` is ``None`` when ``show_residual`` is false).

    * ``"row"``     — one column per joint: torque row on top, Δτ row below.
      Very wide and short: ideal for a full **landscape** LaTeX page.
    * ``"olympic"`` — always two rows:

      * **odd** ``dof`` (5-DoF arm) → 3-over-2 **stagger**: the top row takes
        ``ceil(dof/2)`` panels and the bottom row is shifted right by half a
        panel width, so it nests under the gaps of the top row, exactly like
        the Olympic rings.  (The rings interlock in reality; here the panels
        only *interlock visually* — nothing is drawn on top of a neighbour, so
        no curve is ever hidden.)
      * **even** ``dof`` (6-DoF leg) → plain ``2 × dof/2`` grid, i.e. **2×3**
        for a leg; no half-panel shift exists, so both rows line up.
    """
    if layout == "row":
        nrows = 2 if show_residual else 1
        outer = fig.add_gridspec(nrows, dof, hspace=0.28, wspace=0.30)
        pairs = []
        for j in range(dof):
            ax = fig.add_subplot(outer[0, j])
            ax_r = fig.add_subplot(outer[1, j], sharex=ax) if show_residual else None
            pairs.append((ax, ax_r))
        return pairs

    # --- "olympic" ---
    n_top = (dof + 1) // 2  # dof=5 -> 3 on top ; dof=6 -> 3 on top
    n_bottom = dof - n_top  # dof=5 -> 2 below   ; dof=6 -> 3 below
    if n_bottom < n_top:
        # Half-panel stagger ("rings"): every panel spans 2 columns, so half a
        # panel = 1 GridSpec column.  wspace stays large because each panel
        # needs room for its own y-label plus the tick labels of the *next* one.
        ncols, wspace = 2 * n_top, 0.55
        slots = [(0, 2 * i, 2 * i + 2) for i in range(n_top)]
        slots += [(1, 2 * j + 1, 2 * j + 3) for j in range(n_bottom)]
    else:
        # Even dof (6-DoF leg -> 2x3): both rows share the same columns.
        ncols, wspace = n_top, 0.30
        slots = [(0, i, i + 1) for i in range(n_top)]
        slots += [(1, j, j + 1) for j in range(n_bottom)]

    outer = fig.add_gridspec(2 if n_bottom else 1, ncols, hspace=0.26, wspace=wspace)

    pairs = []
    for row_i, c0, c1 in slots:
        cell = outer[row_i, c0:c1]
        if show_residual:
            inner = cell.subgridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.10)
            ax = fig.add_subplot(inner[0])
            ax_r = fig.add_subplot(inner[1], sharex=ax)
        else:
            ax = fig.add_subplot(cell)
            ax_r = None
        pairs.append((ax, ax_r))
    return pairs


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
    layout: str = "per-joint",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> list:
    """Torque comparison plots — one figure per joint, or every joint in one.

    ``layout`` selects the arrangement:

    * ``"per-joint"`` (default, unchanged behaviour): **one figure per joint**,
      each with the main torque panel and (optionally) a residual panel below.
    * ``"olympic"``: **one single figure** containing every joint — a 3-over-2
      stagger for an odd ``dof`` (the bottom row is shifted half a panel width
      so it nests under the gaps of the top row) or a plain 2×3 grid for an even
      ``dof`` (6-DoF leg).  Each joint keeps its own torque + Δτ pair; the three
      curves share **one** figure-level legend and every Δτ panel shows its two
      RMSE values in a corner box.  Default size ≈ 1.9:1 (landscape).
    * ``"row"``: one single figure, one column per joint (torque row + Δτ row) —
      the widest, flattest variant.

    ``twin`` is an optional ``'START:END'`` time window (seconds) applied to the
    time axis (either end may be omitted; ``None`` = full), matching
    ``compare_torque.py -w/--twin``.  The plotted time axis is re-based on the
    window start, so a zoomed segment is always drawn from 0 s.

    The three curves often nearly overlap when identification is good, so this
    version improves readability via:

    * **Residual panels** (default on): a sub-panel below each torque plot
      shows ``τ_true − τ_prior`` and ``τ_true − τ_identified`` with per-curve
      RMSE — even sub-0.1 Nm gaps become clearly visible.
    * **offset** (optional, Nm): shift the curves vertically
      (prior +offset, identified −offset) to fully separate them.
      Residuals are always computed from the un-shifted data.

    ``figsize`` overrides the per-layout default; ``show=False`` skips the
    interactive window (useful for batch/headless runs).  Saving is left to the
    caller (interactive window → manual export).

    Returns the list of created ``matplotlib.figure.Figure`` objects.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    N_total = len(tau_true)
    N = N_total // dof
    t = np.arange(N) / sample_rate

    # --- Optional time window (like compare_torque.py -w/--twin) ---
    t0, t1 = parse_window(twin, t[-1])
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        raise ValueError(f"Time window [{t0:.3g}, {t1:.3g}] contains no samples")
    # Re-base the plotted time axis on the window start: a ``--twin`` segment is
    # always drawn from 0 s, so the x numbers stay small and comparable between
    # figures (the absolute window is still in the figure title / the CLI call).
    # Without ``twin`` this is a no-op (t[0] is already 0).
    t = t[mask] - t[mask][0]

    combined = layout != "per-joint"
    if combined:
        if layout not in ("olympic", "row"):
            raise ValueError(
                f"layout must be 'per-joint', 'olympic' or 'row', got {layout!r}"
            )
        if figsize is None:
            figsize = (
                (14.5, 7.8) if layout == "olympic" else (max(11.0, 3.4 * dof), 4.9)
            )
        fig = plt.figure(figsize=figsize)
        axes_pairs = _torque_layout_axes(fig, dof, show_residual, layout)
        figs: list = [fig]
    else:
        axes_pairs = None
        figs = []

    for idx, d in enumerate(joint_order):
        row = np.arange(d, N_total, dof)[mask]
        y_true = tau_true[row]
        y_prior = tau_prior[row]
        y_ident = tau_ident[row]

        if offset:
            y_prior = y_prior + offset
            y_ident = y_ident - offset

        if combined:
            ax, ax_r = axes_pairs[idx]
            if ax_r is not None:
                # sharex() does not hide the upper panel's labels by itself.
                ax.tick_params(labelbottom=False)
        elif show_residual:
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
        # In the combined layout the panels are ~3x smaller, so the curves and
        # the (shared) legend are thinned down accordingly.
        lw = 0.85 if combined else 1.0
        ax.plot(t, y_true, "r-", linewidth=2.2 * lw, label=true_label)
        ax.plot(t, y_prior, "b--", linewidth=1.6 * lw, alpha=0.9, label="prior")
        ax.plot(t, y_ident, "g-", linewidth=2.0 * lw, label="identified")
        if combined:
            ax.set_title(joint_names[d], fontsize=10, pad=3)
            ax.set_ylabel("[Nm]", fontsize=9)
            ax.tick_params(labelsize=8)
        else:
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
            if combined:
                ax_r.tick_params(labelsize=8)
                ax_r.text(
                    0.99,
                    0.94,
                    f"RMSE prior {rmse_prior:.3g}\nRMSE ident {rmse_ident:.3g}",
                    transform=ax_r.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    linespacing=1.3,
                    bbox={
                        "boxstyle": "round,pad=0.22",
                        "fc": "white",
                        "ec": "0.75",
                        "alpha": 0.9,
                    },
                )
            else:
                ax_r.legend(loc="upper right", fontsize=7)
            ax_r.grid(True, alpha=0.3)
            ax_r.set_xlim(t[0], t[-1])
            ax_r.set_xlabel("Time [s]", fontsize=9 if combined else None)
        else:
            ax.set_xlabel("Time [s]", fontsize=9 if combined else None)

        if not combined:
            figs.append(fig)

    if combined:
        # One shared legend instead of five identical ones.
        fig.suptitle(title, fontsize=13, y=0.99)
        handles = [
            Line2D([], [], color="r", linewidth=2.2 * 0.85, label=true_label),
            Line2D(
                [], [], color="b", linestyle="--", linewidth=1.6 * 0.85, label="prior"
            ),
            Line2D([], [], color="g", linewidth=2.0 * 0.85, label="identified"),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=3,
            frameon=False,
            fontsize=11,
        )
        # Explicit margins instead of tight_layout(): the nested GridSpec used by
        # these layouts is not tight_layout-compatible, so spacing is chosen here
        # (the per-panel wspace/hspace live in _torque_layout_axes).
        # ``top`` must clear suptitle *and* the shared legend above the panels.
        if layout == "olympic":
            fig.subplots_adjust(left=0.045, right=0.995, top=0.87, bottom=0.075)
        else:
            fig.subplots_adjust(left=0.045, right=0.995, top=0.82, bottom=0.13)
    else:
        for fig in figs:
            fig.suptitle(title, fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

    if show:
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
    layout: str = "per-joint",
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
        layout=layout,
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
    layout: str = "per-joint",
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
        layout=layout,
    )
    return {"stats": stats, "figures": figures}


def plot_torque_comparison_measured_train(
    result: IdentificationResult,
    Y_stack: np.ndarray,
    tau_measured: np.ndarray,
    joint_names: list[str] | None = None,
    bag_name: str | None = None,
    trajectory_yaml: str | None = None,
    sample_rate: float = 100.0,
    offset: float | None = None,
    show_residual: bool = True,
    twin: str | None = None,
    layout: str = "per-joint",
):
    """Training-bag torque comparison — the **identification** bag (meas mode).

    Same figure as ``plot_torque_comparison_measured`` (the held-out
    validation), but drawn on the very samples the solver was fitted on:
    ``Y_stack`` / ``tau_measured`` come straight from
    ``data_from_measurement`` (identification bag, after the ``--twin-id``
    window and the ~``sample_rate`` decimation), so

        ``tau_prior = Y_stack @ pi_prior`` and
        ``tau_ident = Y_stack @ pi_identified``

    are compared against that bag's measured torque.  This shows the
    **achieved training fit** — the prior error the identification removed on
    the data it saw — and is therefore *not* evidence of generalisation
    (that is what the held-out validation plot is for).  Because the ridge is
    pulled toward the prior, the training fit is normally the best case; the
    gap to the held-out numbers is the fit / generalisation split.

    No RMSE table is printed here (the identification report already prints
    the prior-vs-identified comparison on the identification data); the same
    numbers are returned as ``stats``.
    """
    dof = len(result.joint_order)
    if joint_names is None:
        joint_names = [f"joint_{d}" for d in range(dof)]
    if bag_name is None and trajectory_yaml is not None:
        try:
            bag_name = yaml_source_bag(trajectory_yaml)
        except (FileNotFoundError, ValueError):
            bag_name = None

    tau_prior = Y_stack @ result.pi_prior
    tau_ident = Y_stack @ result.pi_identified
    stats = _rmse_comparison(result, result.joint_order, Y_stack, tau_measured)

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
            f"Joint Torque Comparison (identification bag): "
            f"measured vs prior vs identified\n"
            f"bag={bag_name or '?'}  yaml={trajectory_yaml or '?'}"
        ),
        twin=twin,
        layout=layout,
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
    layout: str = "per-joint",
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
            layout=layout,
        )
    return {"stats": stats, "figures": figures}
