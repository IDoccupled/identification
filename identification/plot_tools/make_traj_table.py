#!/usr/bin/env python3
r"""Emit a LaTeX table of excitation-trajectory Fourier coefficients from a YAML.

Layout
------
Row 1 carries the joint names on one line, one ``\multicolumn{2}{c}`` cell per joint.
Under each name two sub-columns hold that joint's ``a`` and ``b`` coefficients, stacked
over the harmonic index ``l = 1..N_i``.  A final row holds the bias term ``q0`` of every
joint, centred over that joint's two columns.  The horizontal rules are plain
``\midrule``s: one under the names, one under the ``a_l``/``b_l`` labels, one above the
``q0`` row.

The joint count is read from the YAML itself, so the left arm (5 joints) and the left
leg (6 joints) both go through the same script.

Usage (from the package root ``src/identification``)
----------------------------------------------------
    python3 identification/thesis_tools/make_traj_table.py     # latest pso_unified_*.yaml
    python3 identification/thesis_tools/make_traj_table.py excite_left_arm.yaml
    python3 identification/thesis_tools/make_traj_table.py excite_left_leg.yaml \
        --font footnotesize --names short

The LaTeX snippet goes to stdout -- copy-paste it as-is.  Advisories (input summary,
estimated table width, overflow hints) go to stderr, so they never end up in the
pasted text.

Width (measured by compiling inside the thesis class, \textwidth = 483.7pt, 4 decimals)
----------------------------------------------------------------------------------------
    arm, footnotesize, names full, tabcolsep 3   529.7pt  too wide
    arm, scriptsize,   names full, tabcolsep 2   464.0pt  fits (19.7pt slack)  <-- default
    leg, footnotesize, names full, tabcolsep 3   516.5pt  too wide
    leg, scriptsize,   names full, tabcolsep 2   451.0pt  fits (32.7pt slack)

With the full names the header, not the numbers, sets the width -- extra decimals are
nearly free, while shortening the names is what buys room: ``--names short`` fits the
arm at footnotesize (438.0pt) and the leg at 446.6pt, and ``--names index`` fits the arm
at ``small`` (383.5pt).  ``--rotate`` turns the table 90 degrees, so the page height
(~660pt) becomes the limit instead of the text width.

Dependencies: standard library + PyYAML only -- deliberately no ROS / pinocchio, so
the script runs with any Python that can read the trajectory YAML.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]
COEFFS_DIR = PKG_ROOT / "trajectory_coefficients"
URDF_PATH = PKG_ROOT / "resource" / "robot" / "urdf" / "serial_pm_v2_identify.urdf"

# Mirror of identification/target_limb_regressor.VALID_LIMB_GROUPS (q-index lists).
# Duplicated on purpose: this script must stay importable without ROS / pinocchio.
GROUP_Q_INDICES = {
    "left_leg": [0, 1, 2, 3, 4, 5],
    "right_leg": [6, 7, 8, 9, 10, 11],
    "waist": [12],
    "left_arm": [13, 14, 15, 16, 17],
    "right_arm": [18, 19, 20, 21, 22],
    "neck": [23],
}

# Text width / text height of the TUM thesis page (a4paper, thesis=student), in
# points, measured by compiling the generated table inside the tumbook class.
# Only used for the width advisory; override with --textwidth.
TEXT_WIDTH_PT = 483.7
TEXT_HEIGHT_PT = 660.0

# Relative glyph size of each font command, for the width estimate.
FONT_SCALE = {
    "normalsize": 1.00,
    "small": 0.90,
    "footnotesize": 0.80,
    "scriptsize": 0.70,
    "tiny": 0.60,
}
# Mean advance width of one character at normalsize, in points.  Calibrated by
# compiling the generated tables inside the thesis class: digits vs the uppercase
# names differ noticeably, so the two are fitted separately.
NUM_CHAR_PT = 5.25
NAME_CHAR_PT = 6.70


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


def load_coeffs(path: Path) -> tuple[list[dict], dict, int]:
    """Return (per-joint dicts in joint order, _meta, number of harmonics)."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"error: {path.name} is not a mapping")

    # Sort numerically -- sorted() on the key strings would put joint_10 before joint_2.
    keys = [k for k in data if isinstance(k, str) and k.startswith("joint_")]
    keys.sort(key=lambda k: int(k.split("_", 1)[1]))
    if not keys:
        raise SystemExit(f"error: {path.name} has no joint_* entries")
    expected = [f"joint_{i}" for i in range(len(keys))]
    if keys != expected:
        raise SystemExit(f"error: unexpected joint keys {keys} (expected {expected})")

    joints = []
    for k in keys:
        j = data[k]
        a, b = list(j["a"]), list(j["b"])
        if len(a) != len(b):
            raise SystemExit(
                f"error: {path.name}:{k}: len(a)={len(a)} != len(b)={len(b)}"
            )
        joints.append({"a": a, "b": b, "q0": float(j["q0"])})

    n_h = len(joints[0]["a"])
    if any(len(j["a"]) != n_h for j in joints):
        raise SystemExit(f"error: {path.name}: joints disagree on the harmonic count")
    return joints, dict(data.get("_meta", {}) or {}), n_h


def urdf_joint_names() -> list[str]:
    """Revolute joint names in q-index order, checked against the J<nn>_<name> convention."""
    if not URDF_PATH.is_file():
        raise SystemExit(f"error: URDF not found: {URDF_PATH}")
    root = ET.parse(str(URDF_PATH)).getroot()
    names = [
        j.get("name", "")
        for j in root.findall("joint")
        if j.get("type") not in ("fixed", None)
    ]
    for i, n in enumerate(names):
        if not n.startswith(f"J{i:02d}_"):
            raise SystemExit(
                f"error: URDF joint order does not match the q-index convention "
                f"(index {i} is {n!r}); pass --joints explicitly"
            )
    return names


def joint_names(group: str | None, n_joints: int, override: str | None) -> list[str]:
    """Joint names for the header; the group decides how many (arm 5, leg 6)."""
    if override:
        names = [s.strip() for s in override.split(",") if s.strip()]
    else:
        if not group:
            raise SystemExit(
                "error: the YAML has no _meta.group, so the joint names are unknown; "
                'pass --joints "J13_...,J14_...,..."'
            )
        if group not in GROUP_Q_INDICES:
            raise SystemExit(
                f"error: unknown group {group!r}; known: {sorted(GROUP_Q_INDICES)}"
            )
        all_names = urdf_joint_names()
        names = [all_names[i] for i in GROUP_Q_INDICES[group]]
    if len(names) != n_joints:
        raise SystemExit(
            f"error: {len(names)} joint names but {n_joints} joints in the YAML ({names})"
        )
    return names


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------
def esc(s: str) -> str:
    """Escape the characters that occur in joint names / file stems."""
    for ch in ("_", "%", "&", "#"):
        s = s.replace(ch, "\\" + ch)
    return s


def display_name(name: str, mode: str) -> str:
    """Joint name as it appears in the header row (unescaped).

    full  -> J13_SHOULDER_PITCH_L   (default)
    short -> SHOULDER_PITCH_L
    index -> J13
    """
    tok = name.split("_")
    if mode == "index":
        return tok[0]
    if mode == "short":
        return "_".join(tok[1:])
    return name


def fmt_num(v: float, fmt: str) -> str:
    """Format a coefficient; a value that rounds to zero is printed without a sign."""
    s = fmt % v
    return "0" if float(s) == 0.0 else s


def estimate_width_pt(
    names: list[str],
    joints: list[dict],
    fmt: str,
    font: str,
    tabcolsep: float,
    mode: str,
) -> float:
    """Rough rendered width in points -- advisory only, to warn about page overflow."""
    scale = FONT_SCALE[font]
    values = [fmt_num(v, fmt) for j in joints for v in j["a"] + j["b"] + [j["q0"]]]
    col = max(len(s) for s in values) * NUM_CHAR_PT * scale + 2.0 * tabcolsep
    label_col = 2.0 * NUM_CHAR_PT * scale + 2.0 * tabcolsep  # '$l$' / '$q_{i0}$'
    blocks = []
    for n in names:
        # the header cell spans the joint's two columns and can be wider than them
        name_pt = len(display_name(n, mode)) * NAME_CHAR_PT * scale + 2.0 * tabcolsep
        blocks.append(max(2.0 * col, name_pt))
    return label_col + sum(blocks)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def build_table(
    yaml_path: Path,
    joints: list[dict],
    meta: dict,
    n_h: int,
    names: list[str],
    args: argparse.Namespace,
) -> str:
    """Assemble the LaTeX snippet (float wrapper plus tabular)."""
    group = meta.get("group")
    label = args.label or "tab:traj_coeffs"
    if args.caption is None:
        limb = group.replace("_", " ") if group else "limb"
        caption = (
            f"Fourier coefficients of the {limb} excitation trajectory "
            f"(\\texttt{{{esc(yaml_path.stem)}}}). "
            f"$l$ is the harmonic index, $a_l$ and $b_l$ the sine and cosine "
            f"coefficients and $q_{{i0}}$ the constant term of joint $i$."
        )
    else:
        caption = args.caption

    out = []
    if not args.bare:
        out.append("\\begin{table}[htbp]")
        out.append("  \\centering")
    out.append(f"  \\setlength{{\\tabcolsep}}{{{args.tabcolsep:g}pt}}")
    out.append(f"  \\{args.font}")
    if not args.bare:
        if caption:
            out.append(f"  \\caption{{{caption}}}")
        if label:
            out.append(f"  \\label{{{label}}}")
    if args.rotate and not args.bare:
        out.append("  \\rotatebox{90}{%")
    out.append("  \\begin{tabular}{c " + " ".join(["cc"] * len(joints)) + "}")
    out.append("    \\toprule")

    # header row 1: one single-line name per joint, spanning its two sub-columns
    cells = [" "]
    for name in names:
        cells.append(
            f"\\multicolumn{{2}}{{c}}{{{esc(display_name(name, args.names))}}}"
        )
    out.append("    " + " & ".join(cells) + " \\\\")
    out.append("    \\midrule")

    # header row 2: the two sub-columns, repeated under every joint
    cells = ["$l$"]
    for _ in names:
        cells += ["$a_l$", "$b_l$"]
    out.append("    " + " & ".join(cells) + " \\\\")
    out.append("    \\midrule")

    # one row per harmonic
    for h in range(n_h):
        cells = [str(h + 1)]
        for j in joints:
            cells += [fmt_num(j["a"][h], args.fmt), fmt_num(j["b"][h], args.fmt)]
        out.append("    " + " & ".join(cells) + " \\\\")
    out.append("    \\midrule")

    # bias term: one number per joint, centred over that joint's two columns
    cells = ["$q_{i0}$"]
    for j in joints:
        cells.append(f"\\multicolumn{{2}}{{c}}{{{fmt_num(j['q0'], args.fmt)}}}")
    out.append("    " + " & ".join(cells) + " \\\\")
    out.append("    \\bottomrule")
    out.append("  \\end{tabular}")
    if args.rotate and not args.bare:
        out.append("  }%")
    if not args.bare:
        out.append("\\end{table}")
    return "\n".join(out) + "\n"


def main() -> int:
    """Parse the command line, print the table to stdout, advisories to stderr."""
    ap = argparse.ArgumentParser(
        description="Print a LaTeX coefficient table for an excitation-trajectory YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The LaTeX snippet is written to stdout; advisories go to stderr.",
    )
    ap.add_argument(
        "yaml",
        nargs="?",
        default=None,
        help="trajectory YAML: path, name inside trajectory_coefficients/, "
        "or omitted = latest pso_unified_*.yaml",
    )
    ap.add_argument(
        "--names",
        choices=["full", "short", "index"],
        default="full",
        help="header style: J13/SHOULDER/PITCH_L (default) | SHOULDER/PITCH_L | J13",
    )
    ap.add_argument(
        "--fmt",
        default="%.4f",
        help="number format (default %%.4f; with full names the header, not the "
        "numbers, sets the width, so digits are almost free)",
    )
    ap.add_argument(
        "--font",
        choices=list(FONT_SCALE),
        default="scriptsize",
        help="font size command inside the table (default: scriptsize)",
    )
    ap.add_argument(
        "--tabcolsep",
        type=float,
        default=2.0,
        help="\\tabcolsep in pt; smaller = narrower table (default 2.0)",
    )
    ap.add_argument(
        "--rotate",
        action="store_true",
        help="rotate the table 90 degrees (uses the page height instead of the "
        "text width, so a bigger font / more decimals fit)",
    )
    ap.add_argument(
        "--caption",
        default=None,
        help="caption text; empty string = no caption; default = generated",
    )
    ap.add_argument("--label", default=None, help="\\label (default: tab:traj_coeffs)")
    ap.add_argument(
        "--bare",
        action="store_true",
        help="print only the tabular (no table float / caption / label)",
    )
    ap.add_argument(
        "--joints",
        default=None,
        help="comma-separated joint names, overrides the URDF lookup",
    )
    ap.add_argument(
        "--textwidth",
        type=float,
        default=TEXT_WIDTH_PT,
        help=f"page text width in pt for the width advisory (default {TEXT_WIDTH_PT:g})",
    )
    args = ap.parse_args()

    try:
        args.fmt % 0.12345
    except (TypeError, ValueError) as e:
        raise SystemExit(f"error: bad --fmt {args.fmt!r} ({e})")

    yaml_path = resolve_yaml(args.yaml)
    joints, meta, n_h = load_coeffs(yaml_path)
    names = joint_names(meta.get("group"), len(joints), args.joints)

    sys.stdout.write(build_table(yaml_path, joints, meta, n_h, names, args))

    width = estimate_width_pt(
        names, joints, args.fmt, args.font, args.tabcolsep, args.names
    )
    n_cols = 1 + 2 * len(joints)
    limit = TEXT_HEIGHT_PT if args.rotate else args.textwidth
    what = "page height" if args.rotate else "text width"
    sys.stderr.write(
        f"\n[make_traj_table] {yaml_path.name}: group={meta.get('group')}, "
        f"{len(joints)} joints ({names[0]} ... {names[-1]}), {n_h} harmonics, "
        f"{n_cols} tabular columns\n"
        f"[make_traj_table] font={args.font}, fmt={args.fmt}, "
        f"tabcolsep={args.tabcolsep:g}pt -> rough width ~{width:.0f}pt, "
        f"{what} ~{limit:.0f}pt, slack ~{limit - width:.0f}pt\n"
    )
    if width > limit:
        sys.stderr.write(
            "[make_traj_table] WARNING: table too wide -- try, in this order:  "
            "--tabcolsep 1.5  |  --fmt '%.2f'  |  --font tiny  |  "
            "--names index  |  --rotate (90 deg)  |  split into one table per limb\n"
        )
    elif args.rotate:
        sys.stderr.write(
            "[make_traj_table] note: rotated tables need \\rotatebox (graphicx) and "
            "must be read sideways; keep the un-rotated variant if it fits\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
