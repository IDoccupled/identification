#!/usr/bin/env python3
r"""Plot the nullspace share of the trajectory parameters.

Reads ``_diagnostics.per_param`` (written by the PSO run) and shows, for every
parameter, the nullspace weight

    omega_j = sum_{i > r_eff} V_ji^2   in [0, 1]

i.e. the share of parameter ``j``'s influence that falls into the numerically
unobservable directions of the regressor (Chapter 3).  ``omega_j`` above the
threshold (recorded as ``_meta.reward_config.nullspace_threshold``, 0.3 in this work)
is what makes a parameter "rank-deficient" and therefore impossible to identify
independently -- those parameters are pulled back to the prior by the ridge in the
identification step.

Everything is taken from the YAML (all ``total_cols`` parameters are recorded, and the
weight is not rounded), so nothing has to be recomputed and no ROS / pinocchio is
needed.  Nothing is written to disk -- the figure is shown with matplotlib.

Usage (any Python with matplotlib + PyYAML)
-------------------------------------------
    python3 identification/thesis_tools/plot_nullspace.py excite_left_arm.yaml
    python3 identification/thesis_tools/plot_nullspace.py excite_left_arm.yaml \
        excite_left_leg.yaml --all
    python3 identification/thesis_tools/plot_nullspace.py excite_left_arm.yaml --type hist

Options
-------
``--all``     also draw the parameters below the threshold (grey), for context;
              by default only the rank-deficient ones (omega > threshold) are shown.
``--top N``   show only the N largest.
``--type``    ``bar`` (default) or ``hist`` (distribution over all parameters).
``--tall``    parameters on the y axis (tall figure); the default is landscape, with
              one vertical bar per parameter and omega on the y axis.
``--index``   keep the parameter order of the YAML instead of sorting by omega.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Kept standalone on purpose (no `identification` import, no ROS), so this runs with
# any Python that has matplotlib + PyYAML.  Same resolver as the other tools.
COEFFS_DIR = Path(__file__).resolve().parents[2] / "trajectory_coefficients"

ABOVE_COLOR = "C1"  # rank-deficient (omega > threshold)
BELOW_COLOR = "0.65"  # everything else


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


def load_nullspace(path: Path, thr_override: float | None):
    """Return (group, entries, threshold, total).

    ``entries`` = [{'name', 'omega', 'quality'}] for every parameter whose nullspace
    weight is recorded (the structurally zero ones are excluded before the SVD, so
    they have none).
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    diag = data.get("_diagnostics") or {}
    per_param = diag.get("per_param")
    if not per_param:
        raise SystemExit(
            f"error: {path.name} has no _diagnostics.per_param -- only PSO trajectory "
            "YAMLs (pso_unified_*.yaml) record the nullspace weights; "
            "recovered_*.yaml from fourier_fit.py do not."
        )
    meta = data.get("_meta") or {}
    threshold = (
        float(thr_override)
        if thr_override is not None
        else float((meta.get("reward_config") or {}).get("nullspace_threshold", 0.3))
    )
    entries = [
        {
            "name": p["name"],
            "omega": float(p["nullspace_weight"]),
            "quality": p.get("quality", "?"),
        }
        for p in per_param
        if "nullspace_weight" in p
    ]
    total = int((diag.get("regression") or {}).get("total_cols", len(per_param)))
    return meta.get("group", "?"), entries, threshold, total


def short_name(name: str) -> str:
    """``J13_SHOULDER_PITCH_L/Izz`` -> ``J13/Izz`` (the joint number identifies it)."""
    joint, _, param = name.partition("/")
    return f"{joint.split('_')[0]}/{param}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def draw_bar_wide(ax, entries, threshold, names_mode, title):
    """Vertical bars: one per parameter on x (names rotated), omega on y.

    Landscape shape -- far less vertical space than the ``--tall`` variant, with the
    threshold as a horizontal line.
    """
    x = np.arange(len(entries))
    vals = [e["omega"] for e in entries]
    above = [v > threshold for v in vals]
    ax.bar(
        x, vals, width=0.72, color=[ABOVE_COLOR if a else BELOW_COLOR for a in above]
    )
    for xi, v in zip(x, vals):
        if v > 0.05:  # keep the rotated values readable
            ax.text(
                xi,
                v + 0.02,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=8,
            )
    # thin the tick labels when there are many bars
    step = max(1, int(np.ceil(len(entries) / 34.0)))
    labels = [
        (e["name"] if names_mode == "full" else short_name(e["name"]))
        if xi % step == 0
        else ""
        for xi, e in enumerate(entries)
    ]
    ax.set_xticks(x, labels, rotation=45, fontsize=8)
    ax.set_xlim(-0.8, len(entries) - 0.2)
    ax.set_ylim(0, 1.20)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(r"nullspace weight $\omega_j$")
    ax.axhline(threshold, color="k", ls="--", lw=1.1, alpha=0.8)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    draw_bar_legend(ax, threshold, above)


def draw_bar_tall(ax, entries, threshold, names_mode, title):
    """Horizontal bars of omega, largest on top, threshold marked (`--tall`)."""
    y = np.arange(len(entries))[::-1]
    vals = [e["omega"] for e in entries]
    above = [v > threshold for v in vals]
    ax.barh(
        y, vals, height=0.68, color=[ABOVE_COLOR if a else BELOW_COLOR for a in above]
    )
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=7)
    ax.set_yticks(
        y,
        [e["name"] if names_mode == "full" else short_name(e["name"]) for e in entries],
        fontsize=7,
    )
    ax.axvline(threshold, color="k", ls="--", lw=1.1)
    ax.set_xlim(0, 1.14)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(r"nullspace weight $\omega_j=\sum_{i>r_{\mathrm{eff}}}V_{ji}^2$")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    draw_bar_legend(ax, threshold, above)


def draw_bar_legend(ax, threshold, above):
    """Shared legend explaining the threshold line and the two bar colours."""
    if len(above) < 2:
        return
    handles = [
        plt.Line2D(
            [],
            [],
            color="k",
            ls="--",
            lw=1.1,
            label=rf"threshold $\omega={threshold:g}$",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=ABOVE_COLOR, label="rank-deficient (above threshold)"
        ),
    ]
    if any(not a for a in above):
        handles.append(
            plt.Rectangle((0, 0), 1, 1, color=BELOW_COLOR, label="below threshold")
        )
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)


def draw_hist(ax, entries, threshold, title):
    """Distribution of omega over all recorded parameters (bimodal: ~0 or ~1)."""
    vals = np.array([e["omega"] for e in entries])
    ax.hist(vals, bins=np.linspace(0, 1, 21), color=BELOW_COLOR, edgecolor="white")
    n_above = int((vals > threshold).sum())
    ax.hist(
        vals[vals > threshold],
        bins=np.linspace(0, 1, 21),
        color=ABOVE_COLOR,
        edgecolor="white",
    )
    ax.axvline(threshold, color="k", ls="--", lw=1.1)
    ax.annotate(
        rf"$\omega={threshold:g}$",
        xy=(threshold, 0),
        xytext=(threshold + 0.03, 1),
        fontsize=8,
    )
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(r"nullspace weight $\omega_j$")
    ax.set_ylabel("number of parameters")
    ax.set_title(
        f"{title}   --   {n_above} of {len(vals)} above the threshold", fontsize=10
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot the nullspace weight of the trajectory parameters recorded "
        "in a trajectory YAML (nothing is saved to disk).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "yaml",
        nargs="*",
        default=None,
        help="one or more trajectory YAMLs (path, name inside "
        "trajectory_coefficients/, or omitted = latest pso_unified)",
    )
    ap.add_argument(
        "--type",
        choices=["bar", "hist"],
        default="bar",
        help="bar = per-parameter weights (default), hist = distribution",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="also show the parameters below the threshold",
    )
    ap.add_argument(
        "--top", type=int, default=None, help="show only the N largest weights"
    )
    ap.add_argument(
        "--tall",
        action="store_true",
        help="old layout: parameters on the y axis (tall figure); the "
        "default is landscape, one vertical bar per parameter",
    )
    ap.add_argument(
        "--index",
        action="store_true",
        help="keep the YAML parameter order instead of sorting by omega",
    )
    ap.add_argument(
        "--names",
        choices=["short", "full"],
        default="short",
        help="y labels: J13/Izz (default) or the full recorded name",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the threshold (default: the value recorded in _meta)",
    )
    args = ap.parse_args()

    paths = [resolve_yaml(a) for a in args.yaml] if args.yaml else [resolve_yaml(None)]

    for path in paths:
        group, entries, threshold, total = load_nullspace(path, args.threshold)
        n_above = sum(e["omega"] > threshold for e in entries)
        labelled_rd = sum(e["quality"] == "rank_deficient" for e in entries)
        other = sorted(
            {
                e["quality"]
                for e in entries
                if e["omega"] > threshold and e["quality"] != "rank_deficient"
            }
        )

        print(
            f"[plot_nullspace] {path.name}  group={group}\n"
            f"  parameters         {total} total, {len(entries)} with a recorded "
            f"nullspace weight ({total - len(entries)} structurally zero)\n"
            f"  threshold          omega > {threshold:g}\n"
            f"  above threshold    {n_above} of {len(entries)} "
            f"({100 * n_above / len(entries):.1f} %)\n"
            f"  labelled           {labelled_rd} rank-deficient"
            + (
                f"; {n_above - labelled_rd} of the above are labelled {', '.join(other)} "
                "(the 'small' label takes priority)"
                if other
                else ""
            )
        )

        sel = entries
        if not args.all and args.type == "bar":
            sel = [e for e in entries if e["omega"] > threshold]
        if not args.index:
            sel = sorted(sel, key=lambda e: -e["omega"])
        if args.top:
            sel = sel[: args.top]

        if args.type == "bar":
            print(
                f"  shown              {len(sel)} parameter(s), "
                f"{'YAML order' if args.index else 'largest first'}"
            )
            for e in sel:
                print(
                    f"    {short_name(e['name']):<14} omega {e['omega']:.6f}"
                    f"   {e['quality']}"
                )
            title = f"{path.stem}  ({group}): nullspace share of the parameters"
            if args.tall:
                fig, ax = plt.subplots(figsize=(6.6, 1.5 + 0.185 * len(sel)), dpi=110)
                draw_bar_tall(ax, sel, threshold, args.names, title)
            else:
                fig, ax = plt.subplots(figsize=(6.6, 2.6), dpi=110)
                draw_bar_wide(ax, sel, threshold, args.names, title)
            fig.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=110)
            draw_hist(ax, entries, threshold, f"{path.stem}  ({group})")
            fig.tight_layout()

    if matplotlib.get_backend().lower() == "agg":
        print(
            "[plot_nullspace] note: the active matplotlib backend is Agg (no display), "
            "so no window can open;\n"
            "                 run it in the desktop session or set MPLBACKEND=TkAgg.",
            file=sys.stderr,
        )
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
