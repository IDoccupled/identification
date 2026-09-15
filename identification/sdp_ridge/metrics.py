"""Torque-RMSE tables and the multi-yaml cross-validation summary.

Pure reporting helpers shared by the solver (``print_results``), the plotting
module and the pipeline (``print_cv_summary``).
"""

from __future__ import annotations

import numpy as np

from .params import IdentificationResult


# ============================================================================
# Torque RMSE helpers
# ============================================================================
def _joint_rmse(
    pi: np.ndarray,
    joint_order: list[int],
    Y_stack: np.ndarray,
    tau_measured: np.ndarray,
) -> np.ndarray:
    """Per-joint RMSE of ``Y_stack @ pi`` vs ``tau_measured``.

    Samples are laid out row-major by joint (sample k, joint d → row k·dof + d),
    matching ``_plot_torque_comparison_panels``.
    """
    dof = len(joint_order)
    n_total = len(tau_measured)
    tau_pred = Y_stack @ pi
    rmse = np.empty(dof)
    for idx, d in enumerate(joint_order):
        row = np.arange(d, n_total, dof)
        rmse[idx] = np.sqrt(np.mean((tau_pred[row] - tau_measured[row]) ** 2))
    return rmse


def _rmse_comparison(
    result: IdentificationResult,
    joint_order: list[int],
    Y_stack: np.ndarray,
    tau_measured: np.ndarray,
) -> dict:
    """Per-joint and overall torque RMSE (prior vs identified); no printing.

    Returns ``{"prior": (dof,), "ident": (dof,), "prior_all": float,
    "ident_all": float}``; the two arrays are indexed by *position in
    ``joint_order``* (same convention as ``_joint_rmse``).
    """
    return {
        "prior": _joint_rmse(result.pi_prior, joint_order, Y_stack, tau_measured),
        "ident": _joint_rmse(result.pi_identified, joint_order, Y_stack, tau_measured),
        "prior_all": float(
            np.sqrt(np.mean((Y_stack @ result.pi_prior - tau_measured) ** 2))
        ),
        "ident_all": float(
            np.sqrt(np.mean((Y_stack @ result.pi_identified - tau_measured) ** 2))
        ),
    }


def _improve_pct(a: float, b: float) -> float:
    """1 − b/a in percent (nan when a ≈ 0)."""
    return (1 - b / a) * 100 if a > 1e-12 else float("nan")


def print_rmse_comparison(
    result: IdentificationResult,
    joint_names: list[str] | None,
    Y_stack: np.ndarray,
    tau_measured: np.ndarray,
) -> dict:
    """Print torque RMSE before (prior) vs after (identified) identification.

    Returns the same numbers as ``_rmse_comparison`` (the multi-yaml
    cross-validation summary aggregates them).
    """
    stats = _rmse_comparison(result, result.joint_order, Y_stack, tau_measured)
    rmse_prior = stats["prior"]
    rmse_ident = stats["ident"]

    def _improve(a: float, b: float) -> float:
        return (1 - b / a) * 100 if a > 1e-12 else float("nan")

    print("\nTorque RMSE comparison (prior vs identified):")
    print(
        f"{'Joint':<20s} {'Prior [Nm]':>12s} {'Identified [Nm]':>15s} {'Improve %':>10s}"
    )
    print("-" * 53)
    for idx, d in enumerate(result.joint_order):
        name = joint_names[d] if joint_names else f"joint_{d}"
        print(
            f"{name:<20s} {rmse_prior[idx]:>12.6g} {rmse_ident[idx]:>15.6g} "
            f"{_improve(rmse_prior[idx], rmse_ident[idx]):>9.2f}%"
        )
    print("-" * 53)
    rp_all = stats["prior_all"]
    ri_all = stats["ident_all"]
    print(
        f"{'ALL':<20s} {rp_all:>12.6g} {ri_all:>15.6g} "
        f"{_improve(rp_all, ri_all):>9.2f}%"
    )
    return stats


def print_cv_summary(
    rows: list[dict],
    joint_names: list[str] | None = None,
    joint_order: list[int] | None = None,
) -> None:
    """One-line-per-yaml summary of a multi-yaml cross-validation run.

    ``rows`` entries: ``{"yaml": str, "note": str, "stats": dict|None,
    "error": str (only when the yaml failed)}``.  Per-joint columns are the
    improvement %% of that joint on that held-out trajectory, followed by the
    mean over all yamls — i.e. the "does the identified URDF generalise"
    verdict, aggregated instead of one table per trajectory.
    """
    if not rows:
        return
    order = sorted(joint_order) if joint_order is not None else []
    shorts = [
        (joint_names[d] if joint_names else f"joint_{d}").split("_")[0] for d in order
    ]
    w_yaml = max(20, max(len(f"{r['yaml']}") for r in rows) + 2)
    w_note = max(10, max(len(f"{r.get('note', '-')}") for r in rows) + 2)
    wj = max(9, max((len(s) for s in shorts), default=0) + 4)
    hdr = (
        f"{'yaml':<{w_yaml}}{'note':<{w_note}}{'prior ALL':>12}{'ident ALL':>12}"
        f"{'ALL imp%':>9}" + "".join(f"{s:>{wj}}" for s in shorts)
    )
    print("\n" + "=" * len(hdr))
    print(f"CROSS-VALIDATION SUMMARY ({len(rows)} held-out yaml)".center(len(hdr)))
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    per_joint_imp: dict[int, list[float]] = {d: [] for d in order}
    all_imp: list[float] = []
    for r in rows:
        st = r.get("stats")
        row_note = f"{r.get('note', '-')}"
        if not st:
            row_err = f"{r.get('error', '')}"[:44]
            print(
                f"{r['yaml']:<{w_yaml}}{row_note:<{w_note}}{'FAILED':>33}   {row_err}"
            )
            continue
        by_d = {d: i for i, d in enumerate(joint_order)}
        imps = [_improve_pct(st["prior"][by_d[d]], st["ident"][by_d[d]]) for d in order]
        imp_all = _improve_pct(st["prior_all"], st["ident_all"])
        all_imp.append(imp_all)
        for d, v in zip(order, imps):
            per_joint_imp[d].append(v)
        print(
            f"{r['yaml']:<{w_yaml}}{row_note:<{w_note}}"
            f"{st['prior_all']:>12.5g}{st['ident_all']:>12.5g}{imp_all:>8.2f}%"
            + "".join(f"{v:>{wj}.1f}" for v in imps)
        )
    if all_imp:
        print("-" * len(hdr))

        def _mean(xs: list[float]) -> float:
            return float(np.mean(xs)) if xs else float("nan")

        print(
            f"{'MEAN over yamls':<{w_yaml}}{'':<{w_note}}"
            f"{'':>12}{'':>12}{_mean(all_imp):>8.2f}%"
            + "".join(f"{_mean(per_joint_imp[d]):>{wj}.1f}" for d in order)
        )
