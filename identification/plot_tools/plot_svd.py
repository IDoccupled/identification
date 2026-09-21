#!/usr/bin/env python3
r"""Plot the singular-value spectrum of an excitation trajectory.

The augmented regressor ``Y`` is rebuilt from a trajectory YAML (all samples of one
period, stacked over the limb's joints), its SVD is taken and the **full** spectrum
``\sigma_1 >= ... >= \sigma_n`` is plotted on a log scale together with

  * the numerical floor ``\sigma_floor = 10^{-6}\sigma_1`` (dashed line),
  * the numerical rank ``r_eff = #{\sigma_i > \sigma_floor}`` (dotted line),
  * a box with ``\sigma_1``, ``\sigma_{r_eff}``, ``\kappa = \sigma_1/\sigma_{r_eff}``
    and the column counts,

i.e. exactly the quantities the D-optimality term of the PSO fitness uses (Chapter 3).
Recomputing is necessary because the YAML itself stores only the first ``r_eff + 5``
singular values, rounded to three decimals -- the nullspace tail (which is what makes
the plot interesting) is not in the file.  When the YAML does carry
``_diagnostics.regression``, the recomputed values are cross-checked against the
recorded ones and the deviation is reported.

Nothing is written to disk: the figure is shown with matplotlib, save it from the
window if you want a file.

Needs the regressor (pinocchio + the ament index), so run it from the package root
with the workspace sourced and the identify venv:

    cd src/identification
    source ../../install/setup.bash
    /home/xiaoran/venv/venv_identify/bin/python identification/thesis_tools/plot_svd.py \
        trajectory_coefficients/excite_left_arm.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Kept standalone on purpose (no `identification` import), so the YAML handling does
# not depend on the package being installed.  make_traj_table.py has the same resolver.
COEFFS_DIR = Path(__file__).resolve().parents[2] / "trajectory_coefficients"

try:
    from identification.fourier_trajectory import FourierTrajectory
except ImportError as e:  # pragma: no cover - import-time environment check
    raise SystemExit(f"error: cannot import the identification package ({e})")

try:
    from identification.target_limb_regressor import (
        VALID_LIMB_GROUPS,
        TargetLimbRegressor,
    )
except ImportError as e:  # pragma: no cover - import-time environment check
    raise SystemExit(
        f"error: the regressor needs the ROS environment ({e})\n"
        "       run from the package root after `source ../../install/setup.bash`, with\n"
        "       an interpreter that has pinocchio (e.g. "
        "/home/xiaoran/venv/venv_identify/bin/python)"
    )

SIGMA_FLOOR_REL = 1e-6  # sigma_floor = 1e-6 * sigma_1 (chapter 3)
ZERO_COL_TOL = 1e-12  # a column with max|Y_ij| below this is structurally zero


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def resolve_yaml(arg: str | None) -> Path:
    """Resolve the argument: existing path, name in trajectory_coefficients/, latest."""
    if arg:
        p = Path(arg)
        if p.is_file():
            return p.resolve()
        cand = COEFFS_DIR / arg
        if cand.is_file():
            return cand.resolve()
        raise SystemExit(
            f"error: trajectory YAML not found: {arg!r} (also tried {cand})"
        )
    matches = sorted(COEFFS_DIR.glob("pso_unified_*.yaml"))
    if not matches:
        raise SystemExit(f"error: no pso_unified_*.yaml in {COEFFS_DIR}")
    return matches[-1].resolve()


def load_coeffs(path: Path) -> tuple[list[dict], dict]:
    """Return (per-joint dicts in joint order, _meta) -- same layout as the other tools."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    keys = [k for k in data if isinstance(k, str) and k.startswith("joint_")]
    keys.sort(key=lambda k: int(k.split("_", 1)[1]))  # numeric, not lexicographic
    if not keys:
        raise SystemExit(f"error: {path.name} has no joint_* entries")
    joints = [
        {"a": list(data[k]["a"]), "b": list(data[k]["b"]), "q0": float(data[k]["q0"])}
        for k in keys
    ]
    return joints, dict(data.get("_meta") or {})


def recorded_regression(path: Path) -> dict:
    """The ``_diagnostics.regression`` block (plus cond_penalty), empty if absent."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    diag = data.get("_diagnostics") or {}
    return {
        "regression": dict(diag.get("regression") or {}),
        "cond": dict((diag.get("reward_breakdown") or {}).get("cond_penalty") or {}),
    }


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------
def build_spectrum(yaml_path: Path, group: str, sample_rate: float, time_coeffs: float):
    """Stack ``Y_aug`` over one period.

    Returns ``(Y, sigma, n_dropped, collisions, n_nonzero)``: the full regressor, the
    singular values of its nonzero columns, how many columns were dropped as
    structurally zero, and how many samples collided.
    """
    joints, _ = load_coeffs(yaml_path)
    dof = len(joints)
    n_group = len(VALID_LIMB_GROUPS[group])
    if dof != n_group:
        raise SystemExit(
            f"error: {yaml_path.name} has {dof} joints but group {group!r} has "
            f"{n_group} -- wrong --group?"
        )

    ft = FourierTrajectory(dim=dof, sample_rate=sample_rate, time_coeffs=time_coeffs)
    flat: list[
        float
    ] = []  # same order as FourierTrajectory.load_coeffs: a1, b1, ..., q0
    for j in joints:
        for ai, bi in zip(j["a"], j["b"]):
            flat += [ai, bi]
        flat.append(j["q0"])
    q, v, a = ft.generate_trajectory(np.asarray(flat))

    # the regressor echoes gravity on stdout; keep this script's stdout for --print
    with contextlib.redirect_stdout(io.StringIO()):
        reg = TargetLimbRegressor(
            group_to_identify=group, print_info=False, gravity=None
        )
        rows = []
        collisions = 0
        for t in range(q.shape[1]):
            res = reg.compute_regressor(q=q[:, t], v=v[:, t], a=a[:, t])
            if res[18]:
                collisions += 1
            rows.append(res[0])  # Y_aug
    Y = np.vstack(rows)

    col_max = np.abs(Y).max(axis=0)
    keep = col_max > ZERO_COL_TOL
    sigma = np.linalg.svd(Y[:, keep], compute_uv=False)
    return Y, sigma, int((~keep).sum()), collisions, int(keep.sum())


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_spectrum(
    sigma: np.ndarray,
    rel: bool,
    title: str,
    annotation: str,
    note: str = "",
) -> None:
    """Draw the recomputed spectrum into the current figure."""
    n = len(sigma)
    idx = np.arange(1, n + 1)
    r_eff = int(np.sum(sigma > SIGMA_FLOOR_REL * sigma[0]))
    floor = SIGMA_FLOOR_REL * sigma[0]
    y = sigma / sigma[0] if rel else sigma
    floor_line = SIGMA_FLOOR_REL if rel else floor

    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=110)
    ax.semilogy(idx, y, marker="o", ms=3.2, lw=1.1, color="C0", label=r"$\sigma_i$")
    ax.axhline(
        floor_line,
        color="C3",
        ls="--",
        lw=1.1,
        label=r"$\sigma_{\mathrm{floor}}=10^{-6}\,\sigma_1$",
    )
    ax.axvline(
        r_eff + 0.5,
        color="0.4",
        ls=":",
        lw=1.1,
        label=rf"$r_{{\mathrm{{eff}}}}={r_eff}$",
    )

    ax.set_ylim(y.min() * 0.3, y.max() * 3.0)
    ax.set_xlim(0, n + 1)
    ax.set_xlabel(r"singular-value index $i$")
    ax.set_ylabel(r"$\sigma_i/\sigma_1$" if rel else r"$\sigma_i$")

    kappa = sigma[0] / sigma[r_eff - 1]
    lines = [
        rf"$\sigma_1={sigma[0]:.4g}$",
        rf"$\sigma_{{r_{{\mathrm{{eff}}}}}}={sigma[r_eff - 1]:.4g}$",
        rf"$\kappa={kappa:.1f}$",
        annotation,
    ]
    ax.text(
        0.985,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9),
    )
    if note:
        ax.text(
            0.5,
            -0.30,
            note,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color="0.35",
        )

    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot the singular-value spectrum rebuilt from a trajectory YAML "
        "(nothing is saved to disk).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Needs the workspace sourced (ament index) and an interpreter with "
        "pinocchio, e.g. /home/xiaoran/venv/venv_identify/bin/python.",
    )
    ap.add_argument(
        "yaml",
        nargs="?",
        default=None,
        help="trajectory YAML: path, name inside trajectory_coefficients/, "
        "or omitted = latest pso_unified_*.yaml",
    )
    ap.add_argument(
        "--group", default=None, help="limb group (default: _meta.group of the YAML)"
    )
    ap.add_argument(
        "--sample-rate",
        type=float,
        default=50.0,
        help="trajectory sample rate in Hz (default 50, the PSO value)",
    )
    ap.add_argument(
        "--time-coeffs",
        type=float,
        default=1.0,
        help="playback time scaling of the trajectory (default 1.0)",
    )
    ap.add_argument(
        "--rel",
        action="store_true",
        help=r"plot $\sigma_i/\sigma_1$ instead of absolute values",
    )
    ap.add_argument(
        "--print",
        dest="dump",
        action="store_true",
        help="also print all recomputed singular values",
    )
    args = ap.parse_args()

    yaml_path = resolve_yaml(args.yaml)
    _, meta = load_coeffs(yaml_path)
    group = args.group or meta.get("group")
    if not group:
        raise SystemExit(
            "error: the YAML has no _meta.group, so the limb is unknown; pass --group"
        )

    Y, sigma, n_dropped, collisions, n_nonzero = build_spectrum(
        yaml_path, group, args.sample_rate, args.time_coeffs
    )
    floor = SIGMA_FLOOR_REL * sigma[0]
    r_eff = int(np.sum(sigma > floor))
    kappa = sigma[0] / sigma[r_eff - 1]
    n_samples = Y.shape[0] // len(VALID_LIMB_GROUPS[group])

    rec = recorded_regression(yaml_path)
    rec_reg, rec_cond = rec["regression"], rec["cond"]

    print(
        f"[plot_svd] {yaml_path.name}  group={group} "
        f"({len(VALID_LIMB_GROUPS[group])} joints)\n"
        f"  samples        {n_samples} per period, sample_rate {args.sample_rate:g} Hz, "
        f"time_coeffs {args.time_coeffs:g}\n"
        f"  regressor      Y: {Y.shape[0]} x {Y.shape[1]} "
        f"({n_nonzero} nonzero columns, {n_dropped} structurally zero)\n"
        f"  sigma_1        {sigma[0]:.6g}\n"
        f"  sigma_r_eff    {sigma[r_eff - 1]:.6g}\n"
        f"  sigma_min      {sigma[-1]:.3g}\n"
        f"  sigma_floor    {floor:.6g}  ({SIGMA_FLOOR_REL:g} * sigma_1)\n"
        f"  r_eff          {r_eff}   (nullspace dim {n_nonzero - r_eff})\n"
        f"  kappa          {kappa:.1f}\n"
        f"  collisions     {collisions}/{n_samples}"
        + (
            "  (informational only; the other limbs sit at their neutral\n"
            "                  pose here, so this is not the PSO's own collision check)"
            if collisions
            else ""
        )
    )

    if rec_reg:
        rec_sigma = np.asarray(rec_reg.get("singular_values", []), dtype=float)
        lines = [
            f"  recorded       r_eff {rec_reg.get('eff_rank')}, "
            f"kappa {rec_cond.get('cond', rec_reg.get('cond'))}, "
            f"sigma_floor {float(rec_reg.get('sigma_floor', 'nan')):.6g}"
        ]
        if rec_sigma.size:
            k = min(rec_sigma.size, len(sigma))
            nonzero = rec_sigma[:k] > 0.0
            dev = (
                np.abs(rec_sigma[:k][nonzero] - sigma[:k][nonzero]) / sigma[:k][nonzero]
            )
            lines.append(
                f"  cross-check    max relative deviation over the "
                f"{int(nonzero.sum())} nonzero recorded values: {dev.max():.2e}"
            )
            lines.append(
                f"                 the YAML stores only min(r_eff+5, n) = "
                f"{rec_sigma.size} values, rounded to 3 decimals"
                + (
                    f"; {int((~nonzero).sum())} of them round to 0"
                    if (~nonzero).any()
                    else ""
                )
            )
        print("\n".join(lines))

    if args.dump:
        for i, s in enumerate(sigma, start=1):
            print(f"{i:4d}  {s:.9e}")

    plot_spectrum(
        sigma,
        args.rel,
        title=f"{yaml_path.stem}  ({group}, {Y.shape[0]} regressor rows)",
        annotation=rf"$n={n_nonzero}$ ({Y.shape[1]} columns)"
        "\n"
        rf"nullspace $={n_nonzero - r_eff}$",
        # note=(
        #     "; ".join(
        #         filter(
        #             None,
        #             [
        #                 f"{n_dropped} structurally zero column(s) dropped"
        #                 if n_dropped
        #                 else "",
        #                 f"{collisions} colliding sample(s)" if collisions else "",
        #             ],
        #         )
        #     )
        # ),
    )

    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        print(
            "[plot_svd] note: the active matplotlib backend is Agg (no display), so no "
            "window can open;\n"
            "           run it in the desktop session or set MPLBACKEND=TkAgg.",
            file=sys.stderr,
        )
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
