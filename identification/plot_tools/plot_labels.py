#!/usr/bin/env python3
r"""Plot the per-quality-label distribution of a trajectory YAML.

Reads ``_diagnostics.quality_summary`` (written by the PSO run) and shows how the
parameters split into the labels null / small / rank-deficient / good / ok / bad --
the classification of Chapter 3, whose thresholds are enforced downstream by the
identification (null and small are hard-frozen, rank-deficient is pulled back to the
prior by the ridge, the rest are rated).

Three chart types are available:

  ``--type bar``      (default) horizontal bars with count *and* percentage --
                      the most readable option: exact values, no angle judgement.
  ``--type stacked``  one 100 %-stacked horizontal bar per YAML; the compact way to
                      compare limbs (e.g. arm vs leg) in a single figure.
  ``--type pie``      pie chart; only sensible when the label set is small, so pair
                      it with ``--merge`` (e.g. ``--merge frozen`` or
                      ``--merge frozen_rated``) to cut the six labels down to 4-5.

``--merge`` aggregates labels: ``frozen`` = null + small (both are hard-frozen in the
identification), ``frozen_rated`` additionally merges good + ok into ``rated``.

Nothing is written to disk -- the figure is shown with matplotlib, save it from the
window if you want a file.

Usage (any Python with matplotlib + PyYAML; no ROS / pinocchio needed)
----------------------------------------------------------------------
    python3 identification/thesis_tools/plot_labels.py excite_left_arm.yaml
    python3 identification/thesis_tools/plot_labels.py excite_left_arm.yaml \
        excite_left_leg.yaml --type stacked
    python3 identification/thesis_tools/plot_labels.py excite_left_arm.yaml \
        --type pie --merge frozen_rated
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

CANONICAL = ["null", "small", "rank_deficient", "good", "ok", "bad"]

# Label aggregation, and how each label is spelled / coloured in the figure.
MERGES = {
    "none": {},
    "frozen": {"null": "frozen", "small": "frozen"},
    "frozen_rated": {
        "null": "frozen",
        "small": "frozen",
        "good": "rated",
        "ok": "rated",
    },
}
DISPLAY = {
    "null": "null",
    "small": "small",
    "frozen": "frozen (null+small)",
    "rank_deficient": "rank-deficient",
    "rated": "rated (good+ok)",
    "good": "good",
    "ok": "ok",
    "bad": "bad",
}
COLORS = {
    "null": "0.35",
    "small": "0.65",
    "frozen": "0.55",
    "rank_deficient": "C1",
    "rated": "C2",
    "good": "C2",
    "ok": "C0",
    "bad": "C3",
}


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


def load_labels(path: Path, merge_map: dict) -> tuple[str, dict, int]:
    """Return (group, {label: count} after merging, total parameter count)."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    diag = data.get("_diagnostics") or {}
    quality = diag.get("quality_summary")
    if not quality:
        raise SystemExit(
            f"error: {path.name} has no _diagnostics.quality_summary -- only PSO "
            "trajectory YAMLs (pso_unified_*.yaml) record the labels; "
            "recovered_*.yaml from fourier_fit.py do not."
        )
    total = int((diag.get("regression") or {}).get("total_cols", 0))
    counts: dict[str, int] = {}
    for label in CANONICAL:
        entry = quality.get(label) or {}
        n = int(entry.get("count", 0))
        name = merge_map.get(label, label)
        counts[name] = counts.get(name, 0) + n
    if total and sum(counts.values()) != total:
        print(
            f"[plot_labels] warning: label counts sum to {sum(counts.values())} but "
            f"total_cols is {total} in {path.name}",
            file=sys.stderr,
        )
    group = (data.get("_meta") or {}).get("group", "?")
    return group, counts, total or sum(counts.values())


def label_order(merge_map: dict) -> list[str]:
    """Labels to show, in canonical order, after merging."""
    out: list[str] = []
    for label in CANONICAL:
        name = merge_map.get(label, label)
        if name not in out:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def draw_bar(ax, counts: dict, total: int, order: list[str], title: str):
    """Horizontal bars: count and percentage next to each bar (the y axis names them,
    so no legend is needed)."""
    y = np.arange(len(order))[::-1]  # first label on top
    vals = [counts.get(k, 0) for k in order]
    ax.barh(y, vals, color=[COLORS[k] for k in order], height=0.65)
    for yi, v in zip(y, vals):
        ax.text(
            v + total * 0.012,
            yi,
            f"{v}  ({100 * v / total:.1f} %)",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y, [DISPLAY[k] for k in order])
    ax.set_xlim(0, max(vals) * 1.35 if max(vals) else 1)
    ax.set_xlabel(f"number of parameters (of {total})")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", alpha=0.5)
    ax.set_axisbelow(True)


def draw_pie(ax, counts: dict, total: int, order: list[str], title: str):
    """Pie with percentage labels; slices below 4 % are left to the legend."""
    labels = [k for k in order if counts.get(k, 0) > 0]
    vals = [counts[k] for k in labels]
    wedges, _, autotexts = ax.pie(
        vals,
        colors=[COLORS[k] for k in labels],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=1.0),
        autopct=lambda p: f"{p:.1f} %" if p >= 8.0 else "",
        pctdistance=0.72,
        textprops=dict(fontsize=8),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.legend(
        wedges,
        [
            f"{DISPLAY[k]}: {counts[k]} ({100 * counts[k] / total:.1f} %)"
            for k in labels
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(3, len(labels)),
        fontsize=8,
        frameon=False,
    )
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def draw_stacked(ax, rows: list[tuple[str, dict, int]], order: list[str]):
    """One 100 %-stacked horizontal bar per YAML (rows = [(title, counts, total)]).

    The legend is left to the caller (it needs figure coordinates).
    """
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for label in order:
        widths = np.array(
            [100.0 * counts.get(label, 0) / total for _, counts, total in rows]
        )
        ax.barh(
            y,
            widths,
            left=left,
            color=COLORS[label],
            height=0.55,
            edgecolor="white",
            linewidth=0.8,
        )
        for yi, w, li in zip(y, widths, left):
            if w >= 8.0:
                ax.text(
                    li + w / 2,
                    yi,
                    f"{w:.0f} %",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        left += widths
    ax.set_yticks(y, [t for t, _, _ in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the parameter set [%]")
    ax.grid(True, axis="x", alpha=0.5)
    ax.set_axisbelow(True)


def legend_handles(order: list[str]):
    """Patch handles for the shared label legend."""
    return [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[k], label=DISPLAY[k]) for k in order
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot the quality-label distribution recorded in trajectory YAMLs "
        "(nothing is saved to disk).",
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
        choices=["bar", "stacked", "pie"],
        default="bar",
        help="chart type (default bar; stacked compares several YAMLs)",
    )
    ap.add_argument(
        "--merge",
        choices=list(MERGES),
        default="none",
        help="merge labels: frozen = null+small, frozen_rated also merges "
        "good+ok into rated (default none)",
    )
    args = ap.parse_args()

    merge_map = MERGES[args.merge]
    order = label_order(merge_map)
    paths = [resolve_yaml(a) for a in args.yaml] if args.yaml else [resolve_yaml(None)]
    rows = []
    for p in paths:
        group, counts, total = load_labels(p, merge_map)
        rows.append((f"{p.stem}\n({group}, {total} params)", counts, total))

    # ---- summary table on stdout ----
    for title, counts, total in rows:
        head = title.replace("\n", "  ")
        print(f"[plot_labels] {head}")
        for label in order:
            n = counts.get(label, 0)
            print(f"    {DISPLAY[label]:<22} {n:4d}  {100 * n / total:5.1f} %")
        print(f"    {'total':<22} {total:4d}  100.0 %")

    titles = [t.split("\n")[0] for t, _, _ in rows]
    groups = [t.split("(")[1].split(",")[0] for t, _, _ in rows]
    n = len(rows)

    if args.type == "stacked":
        height_in = 1.15 + 0.5 * n
        fig, ax = plt.subplots(figsize=(6.6, height_in), dpi=110)
        draw_stacked(
            ax,
            [
                (f"{t}  ({g})", c, tot)
                for t, g, (_, c, tot) in zip(titles, groups, rows)
            ],
            order,
        )
        ax.set_title("Quality labels per limb", fontsize=10, pad=6)
        # explicit margins: the shared legend sits under the x label
        fig.subplots_adjust(
            left=0.30, right=0.98, top=1 - 0.5 / height_in, bottom=1.05 / height_in
        )
        fig.legend(
            handles=legend_handles(order),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(order),
            fontsize=8,
            frameon=False,
        )
    else:
        if args.type == "bar":
            # sharex so the bars of several limbs stay directly comparable
            fig, axes = plt.subplots(
                n, 1, figsize=(6.6, 1.6 + 2.0 * n), dpi=110, sharex=True
            )
        else:  # pie
            fig, axes = plt.subplots(n, 1, figsize=(6.6, 3.1 * n), dpi=110)
            if len(order) >= 6:
                print(
                    "[plot_labels] tip: a 6-slice pie is hard to read -- "
                    "`--merge frozen` or `--merge frozen_rated` gives 5 or 4 slices",
                    file=sys.stderr,
                )
        for ax, name, group, (_, counts, total) in zip(
            np.atleast_1d(axes), titles, groups, rows
        ):
            title = f"{name}  ({group})"
            if args.type == "bar":
                draw_bar(ax, counts, total, order, title)
            else:
                draw_pie(ax, counts, total, order, title)
        fig.tight_layout()

    if matplotlib.get_backend().lower() == "agg":
        print(
            "[plot_labels] note: the active matplotlib backend is Agg (no display), so "
            "no window can open;\n"
            "             run it in the desktop session or set MPLBACKEND=TkAgg.",
            file=sys.stderr,
        )
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
