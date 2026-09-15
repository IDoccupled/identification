"""Write the identified parameters back out as a URDF.

Link inertials (mass / CoM / inertia-about-CoM) and joint dynamics
(armature / damping / friction) replace their prior values in a copy of the
prior URDF; everything else is preserved verbatim.  The output name carries a
``<YYMMDD_HHMMSS>`` timestamp and lands next to the prior URDF unless
``--urdf-out-dir`` says otherwise.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

from .lmi import check_lmi_feasibility
from .params import N_PER_JOINT, split_joint_params


# ============================================================================
# URDF export — write the identified parameters back out as a URDF
# ============================================================================
# Per-joint parameter layout (Pinocchio ``toDynamicParameters()`` order):
#   [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, arm, damp, fric]
# The 6 inertia entries are the inertia **about the joint/link frame origin**
# (I_O).  A URDF ``<inertial>`` stores mass + CoM + inertia **about the CoM**
# (I_C), so the parallel-axis theorem is applied in reverse:
#
#   I_C = I_O − m·(‖c‖²·I₃ − c·cᵀ) ,   c = mc / m
#
# (verified numerically against ``pin.Inertia.FromDynamicParameters`` and
# against the URDF values read by ``pin.buildModelFromUrdf`` — round-trip
# exact to ~1e-19).
#
# ``pin.Inertia`` for joint ``j`` == the URDF inertial of joint j's *child*
# link (Pinocchio's URDF convention: the joint frame IS the child link frame),
# and the URDF inertia component order is
#   [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
# i.e. the same physical tensor, just laid out differently from π.
URDF_NUM_DECIMALS = 8


def _fmt_urdf(v: float) -> str:
    """URDF-style decimal formatting (fixed decimals, no exponent)."""
    return f"{float(v):.{URDF_NUM_DECIMALS}f}"


def dynamic_params_to_inertial(
    pi_joint: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One joint's 13 params → ``(mass, com, I_about_com)``.

    ``pi_joint`` = [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, arm,
    damp, fric] (Pinocchio ordering; the 6 inertia entries are about the
    frame origin).  Returns the mass, the CoM ``c = mc/m`` and the inertia
    tensor about the CoM — exactly what a URDF ``<inertial>`` needs.
    """
    p = np.asarray(pi_joint, dtype=float).reshape(-1)
    if p.size != N_PER_JOINT:
        raise ValueError(f"expected {N_PER_JOINT} params, got {p.size}")
    m = float(p[0])
    if m <= 0.0:
        raise ValueError(f"identified mass must be > 0 (got {m:.6g})")
    mc = p[1:4]
    # Pinocchio inertia layout: [Ixx, Ixy, Iyy, Ixz, Iyz, Izz] about origin
    I_O = np.array(
        [
            [p[4], p[5], p[7]],
            [p[5], p[6], p[8]],
            [p[7], p[8], p[9]],
        ]
    )
    com = mc / m
    I_C = I_O - m * (float(com @ com) * np.eye(3) - np.outer(com, com))
    I_C = 0.5 * (I_C + I_C.T)  # guard tiny numeric asymmetry
    return m, com, I_C


def _urdf_child_link_map(root: ET.Element) -> dict[str, str]:
    """joint name → child link name (URDF has one child link per joint)."""
    out: dict[str, str] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        child = joint.find("child")
        if name and child is not None:
            out[name] = child.attrib.get("link", "")
    return out


def _inertial_block_lines(indent: str, upd: dict) -> list[str]:
    """Render a fresh ``<inertial>`` block (used for links that had none)."""
    ine = upd["inertia"]
    return [
        f"{indent}<inertial>",
        f'{indent}    <origin xyz="{upd["xyz"]}" rpy="0 0 0"/>',
        f'{indent}    <mass value="{upd["mass"]}"/>',
        (
            f'{indent}    <inertia ixx="{ine["ixx"]}" ixy="{ine["ixy"]}" '
            f'ixz="{ine["ixz"]}" iyy="{ine["iyy"]}" iyz="{ine["iyz"]}" '
            f'izz="{ine["izz"]}"/>'
        ),
        f"{indent}</inertial>",
    ]


def _rewrite_urdf_text(
    text: str,
    link_updates: dict[str, dict],
    joint_updates: dict[str, dict],
) -> str:
    """Rewrite only the targeted ``<inertial>`` / ``<dynamics>`` values.

    Everything else (header comments, mesh paths, environment links, tag
    layout, ``/>`` style) is preserved verbatim — this file's layout keeps one
    element per line.  Target links that have no ``<inertial>`` at all get one
    inserted at the end of the ``<link>`` block.
    """
    lines = text.splitlines()
    out: list[str] = []
    cur_link: str | None = None
    cur_joint: str | None = None
    in_inertial = False
    link_had_inertial: set[str] = set()

    for line in lines:
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        lm = re.match(r'<link\s+name="([^"]+)"', stripped)
        if lm:
            cur_link, cur_joint = lm.group(1), None
        jm = re.match(r'<joint\s+name="([^"]+)"', stripped)
        if jm:
            cur_joint, cur_link = jm.group(1), None

        if stripped.startswith("<inertial"):
            in_inertial = True
            link_had_inertial.add(cur_link or "")

        upd = link_updates.get(cur_link) if cur_link else None
        if in_inertial and upd is not None:
            if stripped.startswith("<origin"):
                line = re.sub(r'xyz="[^"]*"', f'xyz="{upd["xyz"]}"', line, count=1)
            elif stripped.startswith("<mass"):
                line = re.sub(r'value="[^"]*"', f'value="{upd["mass"]}"', line, count=1)
            elif stripped.startswith("<inertia"):
                for key, val in upd["inertia"].items():
                    line = re.sub(rf'{key}="[^"]*"', f'{key}="{val}"', line, count=1)

        if stripped.startswith("</inertial>"):
            in_inertial = False

        jupd = joint_updates.get(cur_joint) if cur_joint else None
        if jupd is not None and stripped.startswith("<dynamics"):
            line = (
                f'{indent}<dynamics armature="{jupd["armature"]}" '
                f'damping="{jupd["damping"]}" friction="{jupd["friction"]}"/>'
            )

        if stripped == "</link>":
            # Link without <inertial> (e.g. a bare frame link) → add one.
            lupd = link_updates.get(cur_link) if cur_link else None
            if lupd is not None and cur_link not in link_had_inertial:
                out.extend(_inertial_block_lines(indent + "    ", lupd))
            cur_link = None
        elif stripped == "</joint>":
            cur_joint = None

        out.append(line)
    return "\n".join(out) + "\n"


def _provenance_comment(prov: dict) -> list[str]:
    """XML comment block recording where the identified URDF came from."""
    body = [f"  · {k}: {v}" for k, v in prov.items() if v not in (None, "")]
    if not body:
        return []
    header = "    identified model — generated by sdp_solver_alljoints_ridge.py"
    return ["    <!--", header, *body, "    -->"]


def _unique_path(path: Path) -> Path:
    """Append ``_1``, ``_2``, … until the path does not exist (uniqueness)."""
    if not path.exists():
        return path
    for k in range(1, 1000):
        cand = path.with_name(f"{path.stem}_{k}{path.suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"cannot find a free filename for {path}")


def write_identified_urdf(
    pi_full: np.ndarray,
    joint_names: list[str],
    prior_urdf: str | Path,
    out_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    provenance: dict | None = None,
    timestamp: str | None = None,
    verbose: bool = True,
) -> Path:
    """Write the identified parameters into a copy of the prior URDF.

    The identified link inertials (mass / CoM / inertia-about-CoM) and joint
    dynamics (armature / damping / friction) replace their prior values; every
    other element of the prior URDF is copied verbatim (mesh paths, limits,
    the simulation environment, comments).

    Parameters
    ----------
    pi_full : (dof*13,) ndarray
        Identified parameters, joint-major (same layout as ``data['pi_prior']``
        / ``IdentificationResult.pi_identified``).
    joint_names : list[str]
        URDF joint names, index-aligned with the joints in ``pi_full``
        (``data['joint_names']``).
    prior_urdf : str or Path
        The prior/initial URDF the identification started from.  Its
        directory is the default output directory ("write back to the prior URDF path").
    out_path : str or Path or None
        Explicit output file.  ``None`` (default) → ``<prior_stem>_<timestamp>
        .urdf`` inside ``out_dir``.
    out_dir : str or Path or None
        Output directory; ``None`` → the prior URDF's directory.
    provenance : dict or None
        Optional key/value pairs written as a comment block right after the
        ``<robot>`` tag (mode, trajectory yaml, ridge lambda, …).
    timestamp : str or None
        Timestamp suffix (``YYMMDD_HHMMSS``); ``None`` → now.

    Returns
    -------
    Path
        The written file.
    """
    prior = Path(prior_urdf).resolve()
    if not prior.is_file():
        raise FileNotFoundError(f"prior URDF not found: {prior}")
    text = prior.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    child_of = _urdf_child_link_map(root)

    pi_list = split_joint_params(np.asarray(pi_full, dtype=float))
    if len(joint_names) != len(pi_list):
        raise ValueError(
            f"joint_names has {len(joint_names)} entries but pi_full encodes "
            f"{len(pi_list)} joints"
        )

    link_updates: dict[str, dict] = {}
    joint_updates: dict[str, dict] = {}
    skipped: list[str] = []
    infeasible: list[str] = []
    for jname, pi_j in zip(joint_names, pi_list):
        ok, eig_min, _ = check_lmi_feasibility(pi_j)
        if not ok:
            infeasible.append(f"{jname} (min_eig={eig_min:.3g})")
        link = child_of.get(jname)
        if not link:
            skipped.append(jname)
            continue
        m, com, I_C = dynamic_params_to_inertial(pi_j)
        link_updates[link] = {
            "mass": _fmt_urdf(m),
            "xyz": " ".join(_fmt_urdf(x) for x in com),
            "inertia": {
                key: _fmt_urdf(val)
                for key, val in zip(
                    ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"),
                    (I_C[0, 0], I_C[0, 1], I_C[0, 2], I_C[1, 1], I_C[1, 2], I_C[2, 2]),
                )
            },
        }
        joint_updates[jname] = {
            "armature": _fmt_urdf(pi_j[10]),
            "damping": _fmt_urdf(pi_j[11]),
            "friction": _fmt_urdf(pi_j[12]),
        }

    new_text = _rewrite_urdf_text(text, link_updates, joint_updates)

    if provenance:
        lines = new_text.splitlines()
        insert_at = next(
            (k + 1 for k, ln in enumerate(lines) if re.match(r"<robot\b", ln.strip())),
            0,
        )
        lines[insert_at:insert_at] = _provenance_comment(provenance)
        new_text = "\n".join(lines) + "\n"

    # ---- output path: prior URDF directory + timestamped name ----
    if out_path is not None:
        dest = Path(out_path)
    else:
        ts = timestamp or datetime.now().strftime("%y%m%d_%H%M%S")
        directory = Path(out_dir) if out_dir is not None else prior.parent
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / f"{prior.stem}_{ts}{prior.suffix or '.urdf'}"
        dest = _unique_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(new_text, encoding="utf-8")

    # ---- self-check: re-parse and compare against the identified values ----
    max_err = 0.0
    if not skipped:
        chk = ET.parse(str(dest)).getroot()
        chk_links = {elem.attrib["name"]: elem for elem in chk.findall("link")}
        for link, upd in link_updates.items():
            iel = chk_links[link].find("inertial")
            max_err = max(
                max_err,
                abs(float(iel.find("mass").attrib["value"]) - float(upd["mass"])),
            )
            max_err = max(
                max_err,
                max(
                    abs(float(x) - float(y))
                    for x, y in zip(
                        iel.find("origin").attrib["xyz"].split(), upd["xyz"].split()
                    )
                ),
            )
            ine = iel.find("inertia")
            max_err = max(
                max_err,
                max(
                    abs(float(ine.attrib[k]) - float(v))
                    for k, v in upd["inertia"].items()
                ),
            )
    if verbose:
        print(f"  [urdf] wrote {dest}")
        print(
            f"  [urdf] updated {len(link_updates)} link inertials, "
            f"{len(joint_updates)} joint dynamics "
            f"(max format round-trip error {max_err:.2e})"
        )
        if skipped:
            print(f"  [urdf] WARNING: joints not found in URDF, skipped: {skipped}")
        if infeasible:
            print(
                "  [urdf] WARNING: non physically-consistent inertia (the LMI in "
                "the solver should have prevented this; MuJoCo may reject the "
                "file): " + ", ".join(infeasible)
            )
    return dest
