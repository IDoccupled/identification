"""``--static-test``: static-pose gravity check (sim only).

Samples random collision-free / within-limit static poses (``v = a = 0`` ⇒ only
the gravity term survives) and compares the joint torques of the identified
parameters against the true URDF — same regressor for both, so the difference
isolates the *parameter* error.  Tolerance of the pose sampler, the RNG seed and
the plot layout all live here.
"""

from __future__ import annotations

import numpy as np

from .data import reorder_y_aug


# --- Static-pose gravity test (--static-test, sim only) ---
# Sample N random static poses (v = a = 0 ⇒ only the gravity term survives)
# and compare the joint torques of the true URDF against those computed from
# the identified parameters.
STATIC_TEST_POSES = 20  # number of poses (--static-poses)
STATIC_TEST_SEED = 0  # sampling seed (--static-seed)


# ============================================================================
# Static-pose gravity test (--static-test, sim only)
# ============================================================================


def sample_static_poses(
    reg,
    pi_true: np.ndarray,
    n_poses: int = STATIC_TEST_POSES,
    seed: int = STATIC_TEST_SEED,
    max_tries: int = 50,
    verbose: bool = True,
) -> dict:
    """Sample static poses that are collision-free and within the joint limits.

    A *static pose* is a joint configuration with ``v = a = 0``, so the only
    joint torque left is the gravity term.  Poses are drawn uniformly inside the
    **effective** position limits (``q_margin`` already applied) through
    ``TargetLimbRegressor.sample_state`` — the position limits therefore hold by
    construction — and then kept only if, using the regressor's own machinery:

    * ``collided == False`` (self-collision pairs + table pairs for this limb),
    * ``q_excess == 0`` (belt and braces against the effective limits),
    * ``|tau_gravity,i| <= effortLimit_i`` for every identified joint: a pose the
      joint could not hold statically is not a usable test pose.

    The global numpy RNG is reseeded with ``seed`` (mirroring
    ``target_limb_regressor``, which also samples through the global RNG), so a
    given seed always reproduces the same poses.

    Returns
    -------
    dict with ``q`` (n, dof), ``Y`` (n*dof, 13*dof) — the stacked joint-major
    regressors, so any parameter vector can be evaluated — ``tau_true``
    (n, dof), ``dof`` and the rejection counts.
    """
    dof = reg.dof
    np.random.seed(int(seed))
    q_poses: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    tau_list: list[np.ndarray] = []
    n_try = n_rej_coll = n_rej_tau = 0
    pi_true = np.asarray(pi_true, dtype=float).reshape(-1)

    while len(q_poses) < n_poses and n_try < n_poses * max_tries:
        n_try += 1
        q_full, _, _ = reg.sample_state([])  # no v/a requested => static pose
        q_g = np.asarray([q_full[i] for i in reg.group_to_identify], dtype=float)
        res = reg.compute_regressor(q_g, np.zeros(dof), np.zeros(dof), print_info=False)
        if bool(res[18]) or float(res[12]) > 0.0:  # collision / outside limits
            n_rej_coll += 1
            continue
        Y_g = reorder_y_aug(res[0], dof)  # (dof, 13*dof)
        tau_g = Y_g @ pi_true
        if any(abs(tau_g[i]) > reg.tau_limit[i] for i in range(dof)):
            n_rej_tau += 1
            continue
        q_poses.append(q_g)
        Y_list.append(Y_g)
        tau_list.append(tau_g)

    n_found = len(q_poses)
    if n_found == 0:
        raise RuntimeError(
            f"no valid static pose found in {n_try} tries "
            f"(rejected: {n_rej_coll} collision/limits, {n_rej_tau} torque limit)"
        )
    if verbose:
        note = (
            "" if n_found >= n_poses else f"  WARNING: fewer than requested ({n_poses})"
        )
        print(
            f"  [static] {n_found} static poses, seed={seed}, {n_try} tries "
            f"(rejected: {n_rej_coll} collision/limits, {n_rej_tau} torque limit)."
            f"{note}"
        )

    return {
        "q": np.asarray(q_poses),
        "Y": np.vstack(Y_list),  # (n*dof, 13*dof), row = pose k, joint d
        "tau_true": np.asarray(tau_list),
        "dof": dof,
        "n_tries": n_try,
        "n_rejected_collision": n_rej_coll,
        "n_rejected_torque": n_rej_tau,
    }


def plot_static_pose_errors(
    err: np.ndarray,
    err_prior: np.ndarray | None,
    tau_true: np.ndarray | None,
    joint_names: list[str],
    joint_order: list[int],
    title: str = "Static-pose gravity test",
):
    """Diverging scatter of the static-pose torque error.

    x = one column per static pose (left → right), y = ``tau − tau_true`` with
    **0 exactly in the vertical middle** (the y limits are forced symmetric about
    zero, covering *both* series), one marker per joint —
    **filled = identified, hollow = prior (URDF)** — plus a thin vertical whisker
    joining the two at each pose.  The joints are **fanned out horizontally**
    inside each pose column (identified and prior for the same joint share the
    same x offset, so the whisker stays vertical) instead of being stacked on one
    line, and the pose columns are separated by **alternating white / light-grey
    background bands**.  A marker on the centre line means that model reproduces
    the true gravity torque for that joint at that pose; above / below means
    over- / under-estimation.  The whisker length is the change the
    identification made, and "shorter and closer to the centre line" = better.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_poses = err.shape[0]
    x = np.arange(1, n_poses + 1).astype(float)
    # Symmetric limits around 0 (covering BOTH series) → centre is exactly y = 0.
    spread = [float(np.max(np.abs(err)))]
    if err_prior is not None:
        spread.append(float(np.max(np.abs(err_prior))))
    ymax = 1.15 * max(max(spread), 1e-9)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, 10))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    # Fan the joints of one pose out horizontally so they do not stack on a
    # single vertical line; identified / prior of the same joint share the
    # offset, so the comparison whisker stays vertical.
    n_j = len(joint_order)
    half = 0.34 if n_j > 1 else 0.0
    off_map = {
        d: float(off)
        for d, off in zip(sorted(joint_order), np.linspace(-half, half, n_j))
    }

    fig, ax = plt.subplots(figsize=(max(9.5, 0.55 * n_poses + 4.5), 6.2))
    for j, d in enumerate(joint_order):
        name = joint_names[d] if joint_names else f"joint_{d}"
        color = colors[j % 10]
        marker = markers[j % len(markers)]
        xj = x + off_map[d]  # <- horizontal fan inside the pose column
        rms = float(np.sqrt(np.mean(err[:, j] ** 2)))
        t_rms = (
            float(np.sqrt(np.mean(tau_true[:, j] ** 2)))
            if tau_true is not None
            else float("nan")
        )
        if err_prior is not None:
            ax.vlines(
                xj,
                err_prior[:, j],
                err[:, j],
                color=color,
                linewidth=0.9,
                alpha=0.35,
                zorder=1,
            )
            ax.scatter(
                xj,
                err_prior[:, j],
                s=46,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=1.3,
                alpha=0.9,
                zorder=2,
            )
            rms_p = float(np.sqrt(np.mean(err_prior[:, j] ** 2)))
            label = (
                f"{name}: ident RMS {rms:.3g} / prior {rms_p:.3g} Nm (true {t_rms:.3g})"
            )
        else:
            label = f"{name}: err RMS {rms:.3g} Nm  (true RMS {t_rms:.3g})"
        ax.scatter(
            xj,
            err[:, j],
            s=68,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=1.4, zorder=2)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(0.35, n_poses + 0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("Static pose #   (each pose's joints are fanned out horizontally)")
    ax.set_ylabel(
        "identified / prior − true   [Nm]"
        if err_prior is not None
        else "identified − true   [Nm]"
    )
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # Alternating white / light-grey bands separate the pose columns (the band
    # edges fall exactly between two poses).
    for i in range(1, n_poses + 1):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="0.93", zorder=0, linewidth=0)

    handles, labels = ax.get_legend_handles_labels()
    if err_prior is not None:
        gray = "0.35"
        handles += [
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=9,
                color=gray,
                label="filled = identified",
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor=gray,
                markeredgewidth=1.3,
                label="hollow = prior (URDF)",
            ),
        ]
    ax.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig


def run_static_pose_test(
    reg,
    pi_true: np.ndarray,
    pi_identified: np.ndarray,
    joint_names: list[str],
    joint_order: list[int],
    pi_prior: np.ndarray | None = None,
    n_poses: int = STATIC_TEST_POSES,
    seed: int = STATIC_TEST_SEED,
    plot: bool = True,
    verbose: bool = True,
):
    """Static-pose gravity test: identified / prior vs true joint torques.

    All torques use the **same** regressor (the prior URDF kinematics — exactly
    how the training torque was synthesised), so the differences isolate the
    *parameter* error:

        tau_true  = Y_static @ pi_true
        tau_ident = Y_static @ pi_identified
        tau_prior = Y_static @ pi_prior          (if pi_prior is given)
        err       = tau − tau_true   [Nm]

    Every pose has ``v = a = 0``, so only the gravity term is exercised: this
    checks the mass / CoM / inertia part of the model, which armature / damping
    / friction cannot mask.  With ``pi_prior`` the plot/table also show how much
    the identification improved the static gravity prediction over the raw URDF.
    Returns ``(err, err_prior_or_None, fig_or_None)``.
    """
    dof = reg.dof
    sampled = sample_static_poses(
        reg, pi_true, n_poses=n_poses, seed=seed, verbose=verbose
    )
    Y_static = sampled["Y"]  # (n*dof, 13*dof)
    tau_true = sampled["tau_true"]  # (n, dof)
    n = int(tau_true.shape[0])
    tau_ident = (Y_static @ np.asarray(pi_identified, dtype=float).reshape(-1)).reshape(
        n, dof
    )
    err = tau_ident - tau_true
    err_prior = None
    if pi_prior is not None:
        tau_prior = (Y_static @ np.asarray(pi_prior, dtype=float).reshape(-1)).reshape(
            n, dof
        )
        err_prior = tau_prior - tau_true

    print(
        f"\nStatic-pose gravity test — {n} poses, seed={seed}  "
        f"(err = model − tau_true;  improve = 1 − ident RMS / prior RMS):"
    )
    if err_prior is None:
        hdr = (
            f"{'Joint':<22s} {'errRMS':>9s} {'errMax':>9s} {'errMean':>9s} "
            f"{'trueRMS':>9s} {'err/true':>8s}"
        )
    else:
        hdr = (
            f"{'Joint':<22s} {'identRMS':>9s} {'identMax':>9s} {'priorRMS':>9s} "
            f"{'priorMax':>9s} {'impr':>8s} {'trueRMS':>8s}"
        )
    print(hdr)
    print("-" * len(hdr))
    for j, d in enumerate(joint_order):
        name = joint_names[d] if joint_names else f"joint_{d}"
        rms = float(np.sqrt(np.mean(err[:, j] ** 2)))
        rms_max = float(np.abs(err[:, j]).max())
        t_rms = float(np.sqrt(np.mean(tau_true[:, j] ** 2)))
        if err_prior is None:
            rel = rms / t_rms * 100 if t_rms > 1e-12 else float("nan")
            print(
                f"{name:<22s} {rms:>9.4f} {rms_max:>9.4f} "
                f"{float(err[:, j].mean()):>9.4f} {t_rms:>9.4f} {rel:>7.1f}%"
            )
        else:
            rms_p = float(np.sqrt(np.mean(err_prior[:, j] ** 2)))
            imp = (1 - rms / rms_p) * 100 if rms_p > 1e-12 else float("nan")
            print(
                f"{name:<22s} {rms:>9.4f} {rms_max:>9.4f} {rms_p:>9.4f} "
                f"{float(np.abs(err_prior[:, j]).max()):>9.4f} {imp:>7.1f}% "
                f"{t_rms:>8.4f}"
            )
    print("-" * len(hdr))
    rms_all = float(np.sqrt(np.mean(err**2)))
    rms_all_p = float(np.sqrt(np.mean(err_prior**2))) if err_prior is not None else None
    if rms_all_p is None:
        print(f"{'ALL':<22s} {rms_all:>9.4f}")
    else:
        imp = (1 - rms_all / rms_all_p) * 100 if rms_all_p > 1e-12 else float("nan")
        print(
            f"{'ALL':<22s} {rms_all:>9.4f} {'':>9s} {rms_all_p:>9.4f} "
            f"{'':>9s} {imp:>7.1f}%"
        )

    if not plot:
        return err, err_prior, None
    fig = plot_static_pose_errors(
        err,
        err_prior,
        tau_true,
        joint_names,
        joint_order,
        title=(
            "Static-pose gravity test — model − true joint torque\n"
            f"{n} random collision-free static poses (seed={seed})  |  "
            f"RMS: identified {rms_all:.4f} Nm"
            + (f" vs prior {rms_all_p:.4f} Nm" if rms_all_p is not None else "")
        ),
    )
    return err, err_prior, fig
