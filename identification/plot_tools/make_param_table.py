#!/usr/bin/env python3
r"""Emit the LaTeX parameter table of a **simulation** identification run.

Runs the SDP identification in simulation mode (prior URDF vs. the generated
ground-truth URDF) on one excitation YAML and renders the
Prior / True / Identified table of Chapter~\ref{chap:case}: per joint one block of
six rows -- the three parameter sets, the signed relative error of the prior and
of the identified set, and the identifiability label of every parameter.

Numbers are printed with 4 significant digits in *positional* notation
(``0.00007650``, never ``7.65e-05``); the ``S`` column widths of the ``siunitx``
table are derived from the data, so the same script serves the arm (5 joints),
the leg (6 joints) or any other limb group.

Usage (from the package root ``src/identification``; needs the ROS env with
pinocchio + cvxpy/MOSEK, and ``MPLBACKEND=Agg`` if you also want the console run
to be headless)
----------------------------------------------------------------------------
    source ../../install/setup.bash
    python3 -m identification.plot_tools.make_param_table --yaml excite_left_leg.yaml \
        --label tab:leg-results --out thesis/tables/sim_param_leg_results.tex

    # re-render from a previously dumped CSV (no solver, no ROS needed)
    python3 -m identification.plot_tools.make_param_table --from-csv /tmp/leg.csv ...

The LaTeX snippet goes to stdout (or to ``--out``); the run summary goes to stderr.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
COEFFS_DIR = PKG_ROOT / "trajectory_coefficients"
DEFAULT_OUT_DIR = PKG_ROOT / "thesis" / "tables"

# The 13-scalar per-joint layout of identification/sdp_ridge/params.py, duplicated
# on purpose so that ``--from-csv`` stays importable with a bare Python.
PARAM_LABELS = [
    "mass",
    "mc_x",
    "mc_y",
    "mc_z",
    "Ixx",
    "Ixy",
    "Iyy",
    "Ixz",
    "Iyz",
    "Izz",
    "armature",
    "damping",
    "friction",
]
N_PER_JOINT = 13

# Column header of the tables (matches Chapter 4's notation).
COLUMN_HEADERS = [
    r"$m$",
    r"$c_x$",
    r"$c_y$",
    r"$c_z$",
    r"$I_{xx}$",
    r"$I_{xy}$",
    r"$I_{yy}$",
    r"$I_{xz}$",
    r"$I_{yz}$",
    r"$I_{zz}$",
    r"$k_a$",
    r"$d$",
    r"$f$",
]

# Quality label -> (letter in the table, colour name).
QUALITY_LETTER = {
    "good": ("G", "qGood"),
    "ok": ("O", "qOk"),
    "bad": ("B", "qBad"),
    "rank_deficient": ("R", "qRank"),
    "small": ("S", "qSmall"),
    "null": ("--", "qNull"),
}

COLOR_DEFS = r"""\definecolor{qGood}{RGB}{198, 232, 198}
\definecolor{qOk}{RGB}{214, 230, 248}
\definecolor{qBad}{RGB}{252, 232, 196}
\definecolor{qRank}{RGB}{255, 240, 255}
\definecolor{qSmall}{RGB}{232, 232, 232}
\definecolor{qNull}{RGB}{250, 250, 250}
\definecolor{impGood}{RGB}{224, 246, 224}
\definecolor{impBad}{RGB}{252, 226, 226}"""

SIG_DIGITS = 4  # 4 significant digits, positional (see fmt_value)
PCT_DECIMALS = 2  # percent rows: "+239.76"
NEGLIGIBLE_ABS = (
    1e-3  # |pi_ident - pi_true| below this -> grey "numerically negligible"
)


# ============================================================================
# Number formatting (must match the printed solver report)
# ============================================================================
def fmt_value(v: float, sig: int = SIG_DIGITS) -> str:
    """4 significant digits in positional notation (no scientific notation).

    ``0.3 -> 0.3000``, ``7.65e-05 -> 0.00007650``, ``5.17663 -> 5.177``.
    """
    v = float(v)
    if v == 0.0:
        return "0"
    exp = math.floor(math.log10(abs(v)))
    # Guard against log10 rounding exactly at a power of ten.
    while abs(v) >= 10.0 ** (exp + 1):
        exp += 1
    while abs(v) < 10.0**exp:
        exp -= 1
    decimals = max(0, sig - 1 - exp)
    return f"{v:.{decimals}f}"


def _signed_pct(p: float, ref: float) -> float:
    """Signed relative error of ``p`` w.r.t. ``ref``, in percent.

    Same convention as the solver report: the denominator is ``|ref|`` and falls
    back to 1.0 when the reference is (numerically) zero.
    """
    denom = abs(ref) if abs(ref) > 1e-12 else 1.0
    return (p - ref) / denom * 100.0


def fmt_pct(v: float) -> str:
    """``+239.76`` / ``-45.14`` -- explicit sign, two decimals."""
    return f"{v:+.{PCT_DECIMALS}f}"


def latex_escape_underscores(s: str) -> str:
    return s.replace("_", r"\_")


def short_joint_name(name: str) -> str:
    """``J13_SHOULDER_PITCH_L`` -> ``\\textbf{J13}\\quad\\texttt{SHOULDER\\_PITCH}``."""
    parts = name.split("_")
    # Trailing "_L" / "_R" side suffix is dropped; the joint id keeps its own column.
    rest = parts[1:-1] if len(parts) > 2 and parts[-1] in ("L", "R") else parts[1:]
    return (
        r"\textbf{"
        + parts[0]
        + r"}\quad\texttt{"
        + latex_escape_underscores("_".join(rest))
        + "}"
    )


# ============================================================================
# Running the simulation identification
# ============================================================================
def run_sim_identification(yaml_name: str, sample_rate: float = 100.0) -> list[dict]:
    """Identify the limb of ``yaml_name`` in sim mode; return one row per parameter.

    Uses exactly the module defaults of ``identification.sdp_ridge`` -- i.e. the
    configuration of a plain ``python -m identification.sdp_ridge --sim
    --yaml <yaml_name>`` run -- so the table matches the console report.
    """
    from identification.sdp_ridge import (
        DEFAULT_URDF_PATH,
        LMI_SHAPE_FRAC,
        LMI_SHAPE_WEIGHT,
        RIDGE_LAMBDA,
        SDPSolver,
        SIM_GRAVITY,
        SIM_WAIST_YAW_OFFSET,
        TRUE_URDF_PATH,
        build_freeze_mask,
        build_ridge_weights,
        build_shape_reference,
        load_yaml_param_quality,
        prepare_data_from_urdf,
    )

    print(f"[table] sim identification on {yaml_name} ...", file=sys.stderr)
    data = prepare_data_from_urdf(
        urdf_path=DEFAULT_URDF_PATH,  # prior model
        yaml_filename=yaml_name,
        limb_group=None,  # from the YAML _meta.group
        sample_rate=sample_rate,
        time_coeffs=1.0,
        urdf_true_path=TRUE_URDF_PATH,  # ground truth of the virtual robot
        gravity=SIM_GRAVITY,
        waist_yaw_offset=SIM_WAIST_YAW_OFFSET,
        verbose=False,
    )
    dof = int(data["dof"])
    joint_names = list(data["joint_names"])

    # Quality labels of the excitation trajectory: the YAML carries them itself,
    # hence `--quality-yaml` defaults to the trajectory YAML in sim mode.
    quality_map = load_yaml_param_quality(COEFFS_DIR / yaml_name)
    freeze_mask = build_freeze_mask(quality_map, dof=dof)
    ridge_weights = build_ridge_weights(
        quality_map,
        data["pi_prior"],
        ridge_lambda=RIDGE_LAMBDA,
        freeze_mask=freeze_mask,
    )
    shape_ref = build_shape_reference(data["pi_prior"], dof, frac=LMI_SHAPE_FRAC)

    result = SDPSolver(solver_name="MOSEK", verbose=False).solve(
        Y_stack=data["Y_stack"],
        tau_measured=data["tau_measured"],
        pi_prior=data["pi_prior"],
        joint_order=data["joint_order"],
        ridge_weights=ridge_weights,
        freeze_mask=freeze_mask,
        shape_ref=shape_ref,
        shape_weight=LMI_SHAPE_WEIGHT,
        joint_names=joint_names,
    )

    pi_prior = result.pi_prior
    pi_true = data["pi_true"]
    pi_ident = result.pi_identified
    if pi_true is None:
        raise SystemExit("sim mode produced no pi_true -- is the true URDF present?")

    rows: list[dict] = []
    for d in range(dof):
        for i in range(N_PER_JOINT):
            rows.append(
                {
                    "joint": joint_names[d],
                    "param": PARAM_LABELS[i],
                    "prior": float(pi_prior[d * N_PER_JOINT + i]),
                    "true": float(pi_true[d * N_PER_JOINT + i]),
                    "identified": float(pi_ident[d * N_PER_JOINT + i]),
                    "quality": quality_map.get(d * N_PER_JOINT + i, "ok"),
                }
            )

    # Per-joint torque RMSE, for the run summary on stderr.
    rmse = _per_joint_rmse(data, result)
    print("[table] torque RMSE (prior -> identified):", file=sys.stderr)
    for d in range(dof):
        print(
            f"[table]   {joint_names[d]:<22s} {rmse['prior'][d]:9.5f} -> "
            f"{rmse['ident'][d]:9.7f}  ({rmse['improve'][d]:6.2f} %)",
            file=sys.stderr,
        )
    print(
        f"[table]   {'ALL':<22s} {rmse['prior_all']:9.5f} -> "
        f"{rmse['ident_all']:9.7f}  ({rmse['improve_all']:6.2f} %)",
        file=sys.stderr,
    )
    return rows


def _per_joint_rmse(data: dict, result) -> dict:
    """Per-joint torque RMSE of prior/identified vs the (synthetic) measurement."""
    import numpy as np

    Y = data["Y_stack"]
    tau = data["tau_measured"]
    dof = int(data["dof"])

    def _rmse(pi):
        pred = Y @ pi
        return np.array(
            [
                float(np.sqrt(np.mean((pred[d::dof] - tau[d::dof]) ** 2)))
                for d in range(dof)
            ]
        )

    prior, ident = _rmse(result.pi_prior), _rmse(result.pi_identified)
    p_all = float(np.sqrt(np.mean((Y @ result.pi_prior - tau) ** 2)))
    i_all = float(np.sqrt(np.mean((Y @ result.pi_identified - tau) ** 2)))
    return {
        "prior": prior,
        "ident": ident,
        "improve": (1.0 - ident / prior) * 100.0,
        "prior_all": p_all,
        "ident_all": i_all,
        "improve_all": (1.0 - i_all / p_all) * 100.0,
    }


# ============================================================================
# CSV round-trip (so the LaTeX can be re-rendered without re-solving)
# ============================================================================
CSV_FIELDS = ["joint", "param", "prior", "true", "identified", "quality"]


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("prior", "true", "identified"):
            r[k] = float(r[k])
    return rows


# ============================================================================
# LaTeX rendering
# ============================================================================
def _column_formats(rows: list[dict]) -> list[tuple[int, int]]:
    """(integer digits, decimals) per parameter column, over every printed cell.

    Includes the percent rows -- they are what makes e.g. the mass column two
    integer digits wide.
    """
    ints = [0] * N_PER_JOINT
    decs = [0] * N_PER_JOINT
    for r in rows:
        j = PARAM_LABELS.index(r["param"])
        cells = [
            fmt_value(r["prior"]),
            fmt_value(r["true"]),
            fmt_value(r["identified"]),
        ]
        if r["quality"] != "null":
            cells.append(fmt_pct(_signed_pct(r["prior"], r["true"])))
            cells.append(fmt_pct(_signed_pct(r["identified"], r["true"])))
        for c in cells:
            head, _, tail = c.lstrip("+-").partition(".")
            ints[j] = max(ints[j], len(head))
            decs[j] = max(decs[j], len(tail))
    return list(zip(ints, decs))


def _error_cell(row: dict, which: str) -> str:
    """One shaded percent cell (``impGood`` / ``impBad`` / ``qSmall``)."""
    true = row["true"]
    if row["quality"] == "null":
        return "{--}"
    if which == "prior":
        # The prior is the reference: no shading, only a number.
        return fmt_pct(_signed_pct(row["prior"], true))
    eps_prior = abs(_signed_pct(row["prior"], true))
    eps_ident = abs(_signed_pct(row["identified"], true))
    if eps_ident < eps_prior:
        colour = "impGood"
    elif abs(row["identified"] - true) < NEGLIGIBLE_ABS:
        colour = "qSmall"
    else:
        colour = "impBad"
    return rf"\cellcolor{{{colour}}}{fmt_pct(_signed_pct(row['identified'], true))}"


def render_latex(
    rows: list[dict],
    caption: str,
    label: str,
    note: str | None = None,
    regenerate_cmd: str | None = None,
    resize_frac: float = 0.0,
    notation_ref: str | None = None,
) -> str:
    """Render the full ``sidewaystable`` snippet.

    ``resize_frac`` > 0 wraps the tabular in ``\\resizebox{<frac>\\textheight}{!}``.
    A ``sidewaystable`` is typeset in a box of width ``\\textheight``, so a table
    wider than that (which happens as soon as a limb has parameters two orders of
    magnitude below the rest, e.g. the tiny ankle-pitch link) would otherwise
    stick out of the float; the tabular is then scaled down instead.

    ``notation_ref`` replaces the three-paragraph legend with a one-line pointer
    to a table that already spells it out (frees ~35pt of the float height).
    """
    joints: list[str] = []
    for r in rows:
        if r["joint"] not in joints:
            joints.append(r["joint"])

    formats = _column_formats(rows)
    spec = " ".join(f"S[table-format=-{i}.{d}]" for i, d in formats)

    out: list[str] = []
    out.append("% " + "-" * 75)
    out.append(
        "% AUTO-GENERATED by identification/plot_tools/make_param_table.py "
        "- do not edit by hand."
    )
    if regenerate_cmd:
        out.append("% Regenerate with:")
        out.append(f"%   {regenerate_cmd}")
    out.append("% " + "-" * 75)
    out.append(COLOR_DEFS)
    out.append(r"\begin{sidewaystable}[htbp]")
    out.append(r"  \centering")
    out.append(r"  \caption{" + caption + "}")
    out.append(r"  \label{" + label + "}")
    if resize_frac > 0:
        out.append(r"  \resizebox{" + f"{resize_frac:g}" + r"\textheight}{!}{%")
    out.append(r"  \scriptsize")
    out.append(r"  \setlength{\tabcolsep}{3pt}")
    out.append(
        r"  \sisetup{group-digits=none, input-decimal-markers={.}, "
        r"output-decimal-marker={.}}"
    )
    out.append(r"  \begin{tabular}{@{}l " + spec + r"@{}}")
    out.append(
        r"    \multicolumn{1}{c}{} & \multicolumn{4}{c}{Body} & "
        r"\multicolumn{6}{c}{Inertia tensor (CoM frame)} & "
        r"\multicolumn{3}{c}{Joint} \\"
    )
    out.append(r"    \cmidrule(lr){2-5} \cmidrule(lr){6-11} \cmidrule(lr){12-14}")
    out.append(
        r"    \multicolumn{1}{c}{} & "
        + " & ".join(r"\multicolumn{1}{c}{" + h + "}" for h in COLUMN_HEADERS)
        + r" \\"
    )
    out.append(r"    \midrule")

    for j_idx, joint in enumerate(joints):
        block = [r for r in rows if r["joint"] == joint]
        out.append(r"    \multicolumn{14}{@{}l}{" + short_joint_name(joint) + r"} \\")
        for key, name in (
            ("prior", "Prior"),
            ("true", "True"),
            ("identified", "Identified"),
        ):
            out.append(
                r"    "
                + name
                + " & "
                + " & ".join(fmt_value(r[key]) for r in block)
                + r" \\"
            )
        out.append(
            r"    $\varepsilon_{\mathrm{prior}}$ [\%] & "
            + " & ".join(_error_cell(r, "prior") for r in block)
            + r" \\"
        )
        out.append(
            r"    $\varepsilon_{\mathrm{ident}}$ [\%] & "
            + " & ".join(_error_cell(r, "ident") for r in block)
            + r" \\"
        )
        qual_cells = []
        for r in block:
            letter, colour = QUALITY_LETTER.get(r["quality"], ("?", "qOk"))
            qual_cells.append(r"{\cellcolor{" + colour + "}" + letter + "}")
        out.append(r"    Quality & " + " & ".join(qual_cells) + r" \\")
        out.append(r"    \bottomrule" if j_idx == len(joints) - 1 else r"    \midrule")

    out.append(r"  \end{tabular}")
    if resize_frac > 0:
        out.append(r"  }%")
    out.append(r"  \par\medskip")
    out.append(r"  \begin{minipage}{\textwidth}")
    out.append(r"    \footnotesize")
    if notation_ref:
        out.append(
            r"    The table follows the notation of Table~\ref{" + notation_ref + r"}."
        )
    else:
        out.append(
            r"    $\varepsilon_{\mathrm{prior}}$, $\varepsilon_{\mathrm{ident}}$: "
            r"signed relative error $\varepsilon = 100\,(\pi - "
            r"\pi_{\mathrm{true}})/|\pi_{\mathrm{true}}|$ of the URDF and of the "
            r"identified parameters; the shaded cell is "
            r"\protect\cellcolor{impGood}green when $|\varepsilon|$ decreased during "
            r"identification and \protect\cellcolor{impBad}red when it increased. A "
            r"\protect\cellcolor{qSmall}grey cell marks a deterioration that is "
            r"numerically negligible, i.e.\ one where the parameter still agrees with "
            r"its true value to better than $10^{-3}$."
        )
        out.append(
            r"    \texttt{--} marks parameters with a \texttt{null} label, whose value "
            r"is frozen and whose error is therefore not defined."
        )
        out.append(
            r"    Quality letters (cf.\ Table~\ref{tab:param_class}): "
            r"\protect\cellcolor{qGood}G\,=\,\texttt{good}, "
            r"\protect\cellcolor{qOk}O\,=\,\texttt{ok}, "
            r"\protect\cellcolor{qBad}B\,=\,\texttt{bad}, "
            r"\protect\cellcolor{qRank}R\,=\,\texttt{rank\_deficient}, "
            r"\protect\cellcolor{qSmall}S\,=\,\texttt{small}, "
            r"\protect\cellcolor{qNull}--\,=\,\texttt{null}."
        )
    if note:
        out.append("    " + note)
    out.append(r"  \end{minipage}")
    out.append(r"\end{sidewaystable}")
    return "\n".join(out) + "\n"


# ============================================================================
# CLI
# ============================================================================
GROUP_LIMB_NAME = {
    "left_arm": "left arm",
    "right_arm": "right arm",
    "left_leg": "left leg",
    "right_leg": "right leg",
    "waist": "waist",
    "neck": "neck",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="LaTeX Prior/True/Identified table of a sim identification run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--yaml",
        default="excite_left_leg.yaml",
        help="excitation YAML in trajectory_coefficients/ (also the quality YAML)",
    )
    ap.add_argument(
        "--from-csv",
        default=None,
        help="render from a CSV written by --csv instead of running the solver",
    )
    ap.add_argument(
        "--group",
        default=None,
        help="limb group, e.g. left_leg (default: the YAML's _meta.group)",
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="also dump the extracted parameters to this CSV",
    )
    ap.add_argument(
        "--sample-rate",
        type=float,
        default=100.0,
        help="regression sample rate of the sim run",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="write the LaTeX here (default: stdout)",
    )
    ap.add_argument(
        "--label", default=None, help="LaTeX label (default: tab:<group>-results)"
    )
    ap.add_argument("--caption", default=None, help="override the caption")
    ap.add_argument(
        "--note",
        default=None,
        help="extra footnote sentence appended inside the minipage",
    )
    ap.add_argument(
        "--notation-ref",
        default=None,
        help="replace the legend with a one-line pointer to this table label",
    )
    ap.add_argument(
        "--resize",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="wrap the tabular in \\resizebox{FRAC\\textheight}{!}{} (0 = off); "
        "needed when a limb has parameters two orders of magnitude below the "
        "rest, which makes the natural table wider than the rotated float",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if args.from_csv:
        rows = read_csv(Path(args.from_csv))
        print(f"[table] rows from {args.from_csv}", file=sys.stderr)
    else:
        rows = run_sim_identification(args.yaml, sample_rate=args.sample_rate)
        if args.csv:
            write_csv(rows, Path(args.csv))
            print(f"[table] CSV -> {args.csv}", file=sys.stderr)

    # Limb group: explicit, else the YAML's _meta.group (also in --from-csv mode).
    group = args.group
    if group is None:
        yaml_path = COEFFS_DIR / args.yaml
        if yaml_path.is_file():
            import yaml  # PyYAML is already a dependency of the package

            group = (yaml.safe_load(yaml_path.read_text()).get("_meta") or {}).get(
                "group"
            )
    group = group or "left_leg"
    limb = GROUP_LIMB_NAME.get(group, str(group).replace("_", " "))

    label = args.label or f"tab:{group}-results"
    caption = args.caption or (
        r"Original URDF (\textit{Prior}), ground truth of the virtual robot "
        r"(\textit{True}) and identified (\textit{Identified}) inertial and joint "
        r"parameters of the " + limb + r", together with the per-parameter "
        r"identifiability labels of the excitation trajectory."
    )
    regenerate = (
        "python3 -m identification.plot_tools.make_param_table "
        f"--yaml {args.yaml} --label {label}"
    )

    latex = render_latex(
        rows,
        caption=caption,
        label=label,
        note=args.note,
        regenerate_cmd=regenerate,
        resize_frac=args.resize,
        notation_ref=args.notation_ref,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex, encoding="utf-8")
        print(f"[table] LaTeX -> {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
