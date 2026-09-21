#!/usr/bin/env python3
r"""Plot the identifiability of the *rated* parameters (good / ok / bad).

Reads ``_diagnostics.per_param`` (written by the PSO run) and shows, for every
parameter that is rated -- i.e. not excluded as null / small / rank-deficient -- either

  ``--x rho``    (default) the relative identification uncertainty
                 ``rho_j = sqrt(RIU_j)/|pi_0,j|`` on a log axis, or
  ``--x score``  the score ``s_j = tanh(k (1 - rho_j/rho_0))`` that the label is
                 actually binned from.

Two thresholds are drawn, taken from the recorded ``tanh_k`` / ``tanh_rel_zero`` and
from the binning rule of ``pso_traj_excitation.classify_quality``
(``s > 0.5`` good, ``0 <= s <= 0.5`` ok, ``s < 0`` bad):

    rho_good = 1 - artanh(0.5)/k = 0.451      (s = +0.5)
    rho_bad  = 1 - artanh(0.0)/k = 1.000      (s = 0, the tanh's zero crossing)

so the ok/bad edge sits exactly where the parameter stops being rewarded and
starts being penalised (``rho > rho_0`` = relative uncertainty above 100 %).

Parameters whose *absolute* uncertainty is below ``abs_std_good`` (0.01) have their
score floored at >= 0 and therefore can never be rated ``bad``; those markers are
drawn hollow, because their large relative uncertainty is harmless (they are simply
small nominal values).  In the ``score`` view every marker's colour matches its label
region exactly -- in the ``rho`` view a hollow marker may sit to the right of the
``rho_bad`` line while still being labelled ``ok``.

Everything is read from the YAML (nothing is rounded), so no recomputation and no
ROS / pinocchio are needed.  Nothing is written to disk.

Usage (any Python with matplotlib + PyYAML)
-------------------------------------------
    python3 identification/thesis_tools/plot_identifiability.py excite_left_arm.yaml
    python3 identification/thesis_tools/plot_identifiability.py excite_left_arm.yaml --type hist
    python3 identification/thesis_tools/plot_identifiability.py excite_left_arm.yaml --x score
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Kept standalone on purpose (no `identification` import, no ROS), so this runs with
# any Python that has matplotlib + PyYAML.  Same resolver as the other tools.
COEFFS_DIR = Path(__file__).resolve().parents[2] / "trajectory_coefficients"

SCORE_GOOD = 0.5  # both mirrors of pso_traj_excitation.classify_quality()
SCORE_BAD = 0.0  # score = 0 is the tanh's zero crossing, i.e. rho = rho_0
QUALITY_COLOR = {"good": "C2", "ok": "C0", "bad": "C3"}
EXCLUDED_COLOR = "0.75"


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


def load_params(path: Path):
    """Return (group, entries, config, total).

    ``entries`` = [{'name','rho','score','abs_std','quality'}] for every parameter
    whose RIU is recorded (the structurally zero ones are dropped before the SVD).
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    diag = data.get("_diagnostics") or {}
    per_param = diag.get("per_param")
    if not per_param:
        raise SystemExit(
            f"error: {path.name} has no _diagnostics.per_param -- only PSO trajectory "
            "YAMLs (pso_unified_*.yaml) record the RIU; recovered_*.yaml do not."
        )
    meta = data.get("_meta") or {}
    cfg = dict(meta.get("reward_config") or {})
    entries = [
        {
            "name": p["name"],
            "rho": float(p["rel_std"]),
            "score": float(p["score"]),
            "abs_std": float(p["abs_std"]),
            "quality": p.get("quality", "?"),
        }
        for p in per_param
        if "rel_std" in p
    ]
    total = int((diag.get("regression") or {}).get("total_cols", len(per_param)))
    return meta.get("group", "?"), entries, cfg, total


def short_name(name: str) -> str:
    """``J13_SHOULDER_PITCH_L/Izz`` -> ``J13/Izz`` (the joint number identifies it)."""
    joint, _, param = name.partition("/")
    return f"{joint.split('_')[0]}/{param}"


def thresholds(cfg: dict) -> tuple[float, float, float, float]:
    """(rho_good, rho_bad, k, rho0) from the recorded tanh configuration.

    ``score = tanh(k(1 - rho/rho0))`` ⇒ ``rho = rho0 (1 - artanh(score)/k)``, so the
    two score boundaries of ``classify_quality`` map to these rho values.
    """
    k = float(cfg.get("tanh_k", 1.0))
    rho0 = float(cfg.get("tanh_rel_zero", 1.0))
    return (
        rho0 * (1.0 - math.atanh(SCORE_GOOD) / k),
        rho0 * (1.0 - math.atanh(SCORE_BAD) / k),
        k,
        rho0,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def draw_dot(ax, entries, xmode, cfg, names_mode, title, show_excluded):
    """One marker per parameter, x = rho (log) or score; threshold lines drawn."""
    rho_good, rho_bad, _, _ = thresholds(cfg)
    vals = [e["rho"] if xmode == "rho" else e["score"] for e in entries]
    y = np.arange(len(entries))[::-1]  # worst on top
    floor = float(cfg.get("abs_std_good", 0.01))
    x_lo = min(vals) * 0.6 if xmode == "rho" else -1.12

    for yi, e, v in zip(y, entries, vals):
        color = QUALITY_COLOR.get(e["quality"], EXCLUDED_COLOR)
        hollow = e["abs_std"] < floor
        ax.plot(
            [x_lo, v],
            [yi, yi],
            lw=0.8,
            color=color if not hollow else EXCLUDED_COLOR,
            alpha=0.5,
            zorder=1,
        )
        ax.plot(
            [v],
            [yi],
            marker="o",
            ms=4.5,
            zorder=2,
            color=color if not hollow else "white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            fillstyle="none" if hollow else "full",
        )
    if xmode == "rho":
        ax.set_xscale("log")
        for x, lab in ((rho_good, "good / ok"), (rho_bad, "ok / bad")):
            ax.axvline(x, color="k", ls="--", lw=1.0, alpha=0.5)
            ax.text(
                x, ax.get_ylim()[0] + 0.1, f" {x:.2f}\n {lab}", va="bottom", fontsize=7
            )
        ax.set_xlabel(
            r"relative identification uncertainty "
            r"$\rho_j=\sqrt{\mathrm{RIU}_j}/|\pi_{0,j}|$"
        )
    else:
        for x, lab in ((SCORE_GOOD, "good / ok"), (SCORE_BAD, "ok / bad")):
            ax.axvline(x, color="k", ls="--", lw=1.0, alpha=0.5)
            ax.text(x, ax.get_ylim()[0], f" {x:g}\n {lab}", va="bottom", fontsize=7)
        ax.set_xlabel(r"score $s_j=\tanh(k\,(1-\rho_j/\rho_0))$")
    ax.set_yticks(
        y,
        [e["name"] if names_mode == "full" else short_name(e["name"]) for e in entries],
        fontsize=8,
    )
    ax.set_xlim(x_lo, max(vals) * 1.5 if xmode == "rho" else 1.12)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", which="minor", alpha=0.5, lw=0.5)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=QUALITY_COLOR[q], label=q)
        for q in ("good", "ok", "bad")
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="white",
            mec=QUALITY_COLOR["bad"],
            label=rf"abs. unc. $<{floor:g}$ (score floored)",
        )
    )
    if show_excluded:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="",
                mfc="white",
                mec="0.4",
                label="excluded (small / rank-deficient)",
            )
        )
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)


def draw_hist(ax, entries, xmode, cfg, title):
    """Stacked histogram of rho (log bins) or score, coloured by label."""
    rho_good, rho_bad, _, _ = thresholds(cfg)
    if xmode == "rho":
        vals = np.array([e["rho"] for e in entries])
        vals = np.clip(vals, 1e-3, None)
        bins = np.logspace(np.log10(max(vals.min(), 1e-3)), np.log10(vals.max()), 16)
        edges = [rho_good, rho_bad]
        ax.set_xscale("log")
        ax.set_xlabel(r"relative identification uncertainty $\rho_j$")
        ax.set_xlim(bins[0] * 0.8, bins[-1] * 1.2)
    else:
        vals = np.array([e["score"] for e in entries])
        bins = np.linspace(-1, 1, 21)
        edges = [SCORE_GOOD, SCORE_BAD]
        ax.set_xlabel(r"score $s_j$")
        ax.set_xlim(-1.05, 1.05)

    q = np.array([e["quality"] for e in entries])
    data = [vals[(q == lab)] for lab in ("good", "ok", "bad")]
    ax.hist(
        [d if len(d) else np.array([np.nan]) for d in data],
        bins=bins,
        stacked=True,
        color=[QUALITY_COLOR[lab] for lab in ("good", "ok", "bad")],
        edgecolor="white",
        linewidth=0.5,
        label=["good", "ok", "bad"],
    )
    for x in edges:
        ax.axvline(x, color="k", ls="--", lw=1.0)
    ax.set_ylabel("number of parameters")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot the identifiability of the rated parameters recorded in a "
        "trajectory YAML (nothing is saved to disk).",
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
        "--x",
        choices=["rho", "score"],
        default="rho",
        help="x axis: the relative uncertainty rho (log, default) or the "
        "score the label is binned from",
    )
    ap.add_argument(
        "--type",
        choices=["dot", "hist"],
        default="dot",
        help="dot = one marker per parameter (default), hist = distribution",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="also show the excluded parameters (small / rank-deficient)",
    )
    ap.add_argument("--top", type=int, default=None, help="show only the N largest rho")
    ap.add_argument(
        "--index",
        action="store_true",
        help="keep the YAML order instead of sorting by rho",
    )
    ap.add_argument(
        "--names",
        choices=["short", "full"],
        default="short",
        help="y labels: J13/Izz (default) or the full recorded name",
    )
    args = ap.parse_args()

    paths = [resolve_yaml(a) for a in args.yaml] if args.yaml else [resolve_yaml(None)]

    for path in paths:
        group, entries, cfg, total = load_params(path)
        rho_good, rho_bad, k, rho0 = thresholds(cfg)
        floor = float(cfg.get("abs_std_good", 0.01))
        rated = [e for e in entries if e["quality"] in QUALITY_COLOR]
        sel = rated if not args.all else entries
        if args.top:
            sel = sorted(sel, key=lambda e: -e["rho"])[: args.top]
        if not args.index:
            sel = sorted(sel, key=lambda e: -e["rho"])
        n_floor = sum(e["abs_std"] < floor for e in rated)
        n_bad_zone = sum(e["rho"] > rho_bad for e in rated)
        n_floored_bad_zone = sum(
            e["rho"] > rho_bad and e["abs_std"] < floor for e in rated
        )

        print(
            f"[plot_identifiability] {path.name}  group={group}\n"
            f"  parameters      {total} total, {len(entries)} with a recorded RIU, "
            f"{len(rated)} rated "
            f"(good {sum(e['quality'] == 'good' for e in rated)}, "
            f"ok {sum(e['quality'] == 'ok' for e in rated)}, "
            f"bad {sum(e['quality'] == 'bad' for e in rated)})\n"
            f"  scored with     score = tanh(k(1 - rho/rho0)), k={k:g}, rho0={rho0:g}\n"
            f"  label rule      good: score > {SCORE_GOOD:g} (rho < {rho_good:.3f}), "
            f"ok: {SCORE_BAD:g} <= score <= {SCORE_GOOD:g} "
            f"(rho <= {rho_bad:.3f}), bad: score < {SCORE_BAD:g}\n"
            f"  absolute floor  abs_std < {floor:g} -> score floored at >= 0 "
            f"({n_floor} of {len(rated)} rated parameters)\n"
            f"  rho > {rho_bad:.2f}      {n_bad_zone} rated parameters, of which "
            f"{n_floored_bad_zone} are floor-protected and hence *not* rated bad\n"
            f"  shown           {len(sel)} parameter(s), "
            f"{'YAML order' if args.index else 'largest rho first'}"
        )
        if args.type == "dot":
            for e in sel[:40]:
                print(
                    f"    {short_name(e['name']):<14} rho {e['rho']:10.4f}"
                    f"   score {e['score']:+.3f}   abs {e['abs_std']:.3g}"
                    f"   {e['quality']}"
                )

        title = (
            f"{path.stem}  ({group}): identifiability of the rated parameters"
            f"   [{len(rated)} of {len(entries)}]"
        )
        if args.type == "hist":
            fig, ax = plt.subplots(figsize=(6.6, 2.8), dpi=110)
            draw_hist(ax, entries, args.x, cfg, title)
        else:
            fig, ax = plt.subplots(figsize=(6.6, 2.2 + 0.115 * len(sel)), dpi=110)
            draw_dot(ax, sel, args.x, cfg, args.names, title, args.all)
        fig.tight_layout()

    if matplotlib.get_backend().lower() == "agg":
        print(
            "[plot_identifiability] note: the active matplotlib backend is Agg (no "
            "display), so no window can open;\n"
            "                       run it in the desktop session or set MPLBACKEND=TkAgg.",
            file=sys.stderr,
        )
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
