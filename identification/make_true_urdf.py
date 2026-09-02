#!/usr/bin/env python3
"""Generate ``serial_pm_v2_identify_true.urdf`` — a physically-plausible "real
robot" version of the nominal CAD model (``serial_pm_v2_identify.urdf``).

The true URDF is the ground-truth model used to *simulate* measured data for the
identification algorithms (they try to recover its 13 params/joint:
10 inertial + armature + damping + friction).  This tool:

1. Normalises every ``<dynamics>`` tag to the standard URDF attributes
   ``armature`` + ``damping`` + ``friction`` (MuJoCo-style ``frictionloss`` is
   removed).  ``armature`` defaults match the pattern used in ``identify.urdf`` /
   ``resource/robot/xml/serial_links.xml``: ``0.045325`` for the large leg
   joints (damping 0.12), ``0.039175`` elsewhere.
2. Perturbs the nominal values to mimic a real robot: mass stays close to CAD,
   while COM and the inertia *shape/orientation* deviate clearly.  By default
   (all relative to nominal):
   - mass  : +1 % .. +5 %  (slightly heavier than CAD — small gap)
   - COM   : shift scaled with each link's own radius of gyration (COM_REL),
             so physically bigger links shift more
   - inertia: principal moments change per axis (INERTIA_SHAPE_STD → shape) and
             the ellipsoid's principal axes get a small rotation
             (INERTIA_ORIENT_STD → direction micro-adjustment, not just size),
             then projected so the tensor stays positive-definite AND obeys the
             triangle inequality
   - dynamics: small relative perturbation of armature / damping / friction per
             actuator class (+ tiny per-joint jitter, so same-model joints —
             mirrored L/R included — stay close).
3. Copies the simulation-experiment environment (the fixed ``lay_down`` /
   ``table`` / ``table2`` links & joints the simulator expects, e.g. the table
   the robot lies on) from the nominal URDF, so the true model carries the same
   environment.  Elements already present are left untouched; disable with
   ``--no-env``.

Reproducible: fixed default RNG seed (override with ``--seed``).

Run from ``src/identification``:
    python -m identification.make_true_urdf            # overwrite true URDF
    python -m identification.make_true_urdf --dry-run  # preview only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOMINAL = (
    REPO_ROOT / "resource" / "robot" / "urdf" / "serial_pm_v2_identify.urdf"
)
DEFAULT_TRUE = (
    REPO_ROOT / "resource" / "robot" / "urdf" / "serial_pm_v2_identify_true.urdf"
)

INERTIA_KEYS = ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]  # URDF order
# Dummy links (fixed-joint feet, ~1e-3 kg) are left untouched.
MIN_GENERATED_MASS = 0.01  # kg — links below this are not perturbed


# ---------------------------------------------------------------------------
# Physical / perturbation defaults (tunable via CLI where marked)
# ---------------------------------------------------------------------------
DEFAULT_SEED = 233
# "真实机分布" preset (2026-09-02): observed on the real robot —
#   * mass stays close to CAD (small gap),
#   * COM position and inertia SHAPE/ORIENTATION deviate clearly.
MASS_LO, MASS_HI = 0.01, 0.05  # +1%..+5% heavier (small)
MASS_JIT_STD = 0.002  # per-side mass jitter (L/R stay close)
COM_REL = 0.04  # COM shift std = COM_REL * radius-of-gyration of the link
INERTIA_SHAPE_STD = 0.05  # per-axis principal-moment noise -> ellipsoid shape
INERTIA_JIT_STD = 0.01  # extra per-principal-moment jitter
INERTIA_ORIENT_STD = 0.01  # rad; mean principal-axis tilt ≈ 1.6×this (0.01≈0.9°)
TRIANGLE_SAFETY = 0.02  # keep a >=2% triangle margin after projection
# Dynamics relative perturbation (per-group ~N(0,s) + per-joint jitter).
# *_JIT_STD  sets how far true dynamics drift from nominal (whole actuator class);
# *_SIDE_STD is deliberately small so same-model joints (L/R, same actuator)
#            stay close to each other.
ARM_JIT_STD = 0.03
DAMP_JIT_STD = 0.05
FRIC_JIT_STD = 0.05
ARM_SIDE_STD = 0.004
DAMP_SIDE_STD = 0.008
FRIC_SIDE_STD = 0.008


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------
@dataclass
class LinkInertial:
    """Nominal inertial of one link, expressed about its COM in the link frame."""

    name: str
    mass: float
    com: np.ndarray  # (3,)
    I: np.ndarray  # (3,3) symmetric, about COM


@dataclass
class JointDynamics:
    name: str
    armature: float
    damping: float
    friction: float


def _mat3(ixx, ixy, ixz, iyy, iyz, izz) -> np.ndarray:
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)


def _fmt(v: float) -> str:
    """URDF-style decimal formatting (8 decimals, no exponent)."""
    s = f"{v:.8f}"
    return s


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _load_nominal(nominal_urdf: Path):
    """Return (link_inertials, joint_dynamics) keyed by name from nominal URDF."""
    root = ET.parse(str(nominal_urdf)).getroot()

    links: dict[str, LinkInertial] = {}
    for link in root.findall("link"):
        name = link.attrib.get("name")
        iel = link.find("inertial")
        if not name or iel is None:
            continue
        mass = float(iel.find("mass").attrib["value"])
        ine = iel.find("inertia")
        I6 = {k: float(ine.attrib[k]) for k in INERTIA_KEYS}
        o = iel.find("origin")
        com = (
            np.array([float(x) for x in o.attrib["xyz"].split()])
            if o is not None
            else np.zeros(3)
        )
        links[name] = LinkInertial(name, mass, com, _mat3(**I6))

    joints: dict[str, JointDynamics] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if joint.attrib.get("type") == "fixed" or not name:
            continue
        d = joint.find("dynamics")
        if d is None:
            continue
        joints[name] = JointDynamics(
            name=name,
            armature=float(d.attrib.get("armature", "0.0")),
            damping=float(d.attrib.get("damping", "0.0")),
            friction=float(d.attrib.get("friction", "0.0")),
        )
    return links, joints


def _mirror_groups(names: list[str]) -> list[list[str]]:
    """Group names into mirrored same-part sets (``*_L``/``*_R``) + singletons."""
    name_set = set(names)
    used: set[str] = set()
    groups: list[list[str]] = []
    for n in sorted(names):
        if n in used:
            continue
        if n.endswith(("_L", "_R")) and not n.startswith("J_"):
            base = n[:-2]
            lft, rgt = base + "_L", base + "_R"
            if lft in name_set and rgt in name_set:
                groups.append([lft, rgt])
                used.update((lft, rgt))
                continue
        groups.append([n])
        used.add(n)
    return groups


def _joint_classes(
    joints: dict[str, JointDynamics],
) -> dict[tuple[float, float, float], list[str]]:
    """Group joints by their *identical nominal* dynamics (same actuator model).

    Joints that share the same armature/damping/friction in the nominal URDF are
    the same part/motor class (e.g. the six large leg joints at
    armature=0.045325/damping=0.12/friction=0.55, and the small joints at
    0.039175/0.08/0.30).  They share one perturbation so their true values stay
    close (constraint: same-model joints must not drift apart).
    """
    classes: dict[tuple[float, float, float], list[str]] = {}
    for name, j in sorted(joints.items()):
        key = (round(j.armature, 6), round(j.damping, 6), round(j.friction, 6))
        classes.setdefault(key, []).append(name)
    return classes


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------
def _project_physical(w: np.ndarray, safety: float) -> np.ndarray:
    """Enforce positive principal moments + triangle inequality (in place).

    Order-independent: whichever of the three moments is largest is clamped to
    ``(sum of the other two) * (1 - safety)`` so the tensor stays physical even
    when per-axis noise has reordered the moments.  The value stays paired with
    its own eigenvector (no re-labelling).
    """
    w = np.maximum(w, 1e-12)
    total = w.sum()
    A = w.max()
    if A >= total - A:  # A >= B + C
        w[np.argmax(w)] = (total - A) * (1.0 - safety)
    return w


def _rot_matrix(ang: np.ndarray) -> np.ndarray:
    """Rotation matrix from small body angles [ax, ay, az] (R = Rz @ Ry @ Rx)."""
    ax, ay, az = ang
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def _perturb_link(
    link: LinkInertial, fm: float, rng: np.random.Generator
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (mass, com, I) true values from a nominal link.

    Real-robot behaviour vs CAD: mass stays close (``fm`` ≈ 1.01..1.05) while
    COM and the inertia *ellipsoid* deviate clearly:
      - COM shifts scaled with the link's own radius of gyration (COM_REL);
      - principal moments change independently per axis (INERTIA_SHAPE_STD)
        → different ellipsoid shape (not just an overall scale);
      - the ellipsoid's principal axes get a small rotation (INERTIA_ORIENT_STD)
        → direction micro-adjustment, so off-diagonal terms change too.
    The result is projected onto the physical set (PD + triangle inequality).
    """
    # --- mass (small gap) ---
    mass = link.mass * fm * (1.0 + rng.normal(0.0, MASS_JIT_STD))

    # --- COM: scaled with the link's own size (radius of gyration) ---
    rg = float(np.sqrt(np.trace(link.I) / max(link.mass, 1e-9)))
    com = link.com + rng.normal(0.0, COM_REL * rg, 3)

    # --- inertia ellipsoid ---
    w, V = np.linalg.eigh(link.I)  # ascending eigenvalues, columns = axes

    # shape: independent per-axis change (ellipsoid proportions) + jitter
    per_axis = (1.0 + rng.normal(0.0, INERTIA_SHAPE_STD, 3)) * (
        1.0 + rng.normal(0.0, INERTIA_JIT_STD, 3)
    )
    w_true = _project_physical(w * fm * per_axis, TRIANGLE_SAFETY)

    # orientation: small rotation of the principal axes (direction micro-adjust)
    R = _rot_matrix(rng.normal(0.0, INERTIA_ORIENT_STD, 3))
    Vn = V @ R

    I = (Vn * w_true) @ Vn.T  # sum_k w_k v_k v_k^T
    I = (I + I.T) / 2.0  # guard tiny numeric asymmetry
    return mass, com, I


def _perturb_dynamics(
    joint: JointDynamics, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return (armature, damping, friction) true values for one joint."""
    arm = joint.armature * (1.0 + rng.normal(0.0, ARM_SIDE_STD))
    damp = joint.damping * (1.0 + rng.normal(0.0, DAMP_SIDE_STD))
    fric = joint.friction * (1.0 + rng.normal(0.0, FRIC_SIDE_STD))
    return max(arm, 1e-6), max(damp, 1e-6), max(fric, 1e-6)


def _round_sym(M: np.ndarray) -> np.ndarray:
    return np.round(M, 8)


# ---------------------------------------------------------------------------
# Formatting-preserving text rewrite
# ---------------------------------------------------------------------------
def _rewrite_preserving(
    text: str,
    link_updates: dict[str, dict],
    joint_updates: dict[str, dict],
) -> str:
    """Rewrite only the targeted inertial/dynamics values, leaving everything
    else (header comments, tag layout, ``/>`` style, non-target links) untouched.

    Operates line-by-line on the original URDF text; the file layout used here
    keeps every element on a single line.
    """
    lines = text.splitlines()
    out: list[str] = []
    cur_link: str | None = None
    cur_joint: str | None = None
    in_inertial = False

    for line in lines:
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        lm = re.match(r"<link\s+name=\"([^\"]+)\"", stripped)
        if lm:
            cur_link, cur_joint = lm.group(1), None
        jm = re.match(r"<joint\s+name=\"([^\"]+)\"", stripped)
        if jm:
            cur_joint, cur_link = jm.group(1), None

        if stripped.startswith("<inertial"):
            in_inertial = True

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

        if stripped.startswith("</joint>"):
            cur_joint = None
        elif stripped.startswith("</link>"):
            cur_link = None

        out.append(line)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Environment (simulation props) sync — copy from the nominal URDF
# ---------------------------------------------------------------------------
def _ensure_env_in_text(nominal_text: str, true_text: str) -> tuple[str, list[str]]:
    """Return ``true_text`` with any missing simulation-environment elements
    copied verbatim from ``nominal_text``.

    The environment = top-level links with no ``<inertial>`` (``lay_down``,
    ``table``, ``table2``) plus the fixed joints connecting them — i.e. the
    table props the simulator expects.  Elements already present in the true
    model are untouched (idempotent across runs); missing ones are inserted as
    raw text blocks just before the first robot link (``<link name="LINK_...>``)
    so formatting matches the rest of the file.
    """
    nroot = ET.fromstring(nominal_text)
    env_links = {
        l.attrib["name"] for l in nroot.findall("link") if l.find("inertial") is None
    }
    env_names: set[str] = set(env_links)
    for j in nroot.findall("joint"):
        if j.attrib.get("type") != "fixed":
            continue
        parent = j.find("parent")
        child = j.find("child")
        pn = parent.attrib.get("link") if parent is not None else None
        cn = child.attrib.get("link") if child is not None else None
        if pn in env_links or cn in env_links:
            env_names.add(j.attrib["name"])

    present = set(re.findall(r"<(?:link|joint)\s+name=\"([^\"]+)\"", true_text))
    order: list[str] = []
    for el in list(nroot):
        nm = el.attrib.get("name")
        if nm in env_names:
            order.append(nm)
    missing = [nm for nm in order if nm not in present]
    if not missing:
        return true_text, []

    want = set(missing)
    lines = nominal_text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m = re.match(r"<(link|joint)\s+name=\"([^\"]+)\"", stripped)
        if m and m.group(2) in want:
            tag = m.group(1)
            if stripped.endswith("/>"):
                blocks.append(lines[i])
                i += 1
                continue
            depth, j = 1, i + 1
            while j < len(lines) and depth > 0:
                s = lines[j].strip()
                if s == f"</{tag}>":
                    depth -= 1
                elif s.startswith(f"<{tag}") and not s.endswith("/>"):
                    depth += 1
                j += 1
            blocks.append("\n".join(lines[i:j]))
            i = j
            continue
        i += 1

    tl = true_text.splitlines()
    anchor = next(
        (k for k, ln in enumerate(tl) if ln.strip().startswith('<link name="LINK_')),
        None,
    )
    insert: list[str] = [""]
    insert.extend(blocks)
    insert.append("")
    if anchor is None:
        anchor = len(tl)  # fall back: append at end
    new_tl = tl[:anchor] + insert + tl[anchor:]
    return "\n".join(new_tl), list(missing)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _generate(
    nominal_urdf: Path,
    true_urdf: Path,
    seed: int,
    dry_run: bool,
    verbose: bool = True,
    with_env: bool = True,
) -> None:
    links, joints = _load_nominal(nominal_urdf)
    if verbose:
        print(f"nominal : {nominal_urdf}")
        print(f"true    : {true_urdf}  (seed={seed}, dry_run={dry_run})")
        print(
            f"links={len(links)}  joints(dynamics)={len(joints)}  "
            f"mirror_link_groups={len(_mirror_groups(list(links)))}"
        )

    original_text = true_urdf.read_text(encoding="utf-8")
    nominal_text = nominal_urdf.read_text(encoding="utf-8")
    tree = ET.parse(str(true_urdf))
    root = tree.getroot()

    rng = np.random.default_rng(seed)

    link_updates: dict[str, dict] = {}
    joint_updates: dict[str, dict] = {}

    # ---------------- links: inertial perturbation ----------------
    link_elems = {l.attrib["name"]: l for l in root.findall("link")}
    gen = [lk for lk in links.values() if lk.mass >= MIN_GENERATED_MASS]
    gen_names = sorted(g.name for g in gen)
    n_links = 0
    for group in _mirror_groups(gen_names):
        fm = 1.0 + rng.uniform(MASS_LO, MASS_HI)  # shared small mass gain
        for name in group:
            lk = links[name]
            mass, com, I = _perturb_link(lk, fm, rng)
            elem = link_elems.get(name)
            if elem is None:
                continue
            I = _round_sym(I)
            link_updates[name] = {
                "mass": _fmt(mass),
                "xyz": " ".join(_fmt(x) for x in com),
                "inertia": {
                    key: _fmt(I[np.triu_indices(3)][idx])
                    for idx, key in enumerate(INERTIA_KEYS)
                },
            }
            n_links += 1
            if verbose:
                print(
                    f"  [link ] {name:<22s} m {lk.mass:.5f}->{mass:.5f} "
                    f"(+{(mass / lk.mass - 1) * 100:+.2f}%)  "
                    f"|com| {np.linalg.norm(lk.com):.4f}->{np.linalg.norm(com):.4f}"
                )

    # ---------------- joints: dynamics perturbation ----------------
    joint_elems = {
        j.attrib["name"]: j
        for j in root.findall("joint")
        if j.attrib.get("type") != "fixed"
    }
    n_joints = 0
    for members in _joint_classes(joints).values():
        # shared dynamics perturbation per mirrored part
        g_arm = 1.0 + rng.normal(0.0, ARM_JIT_STD)
        g_damp = 1.0 + rng.normal(0.0, DAMP_JIT_STD)
        g_fric = 1.0 + rng.normal(0.0, FRIC_JIT_STD)
        for jname in members:
            nom = joints[jname]
            elem = joint_elems.get(jname)
            if elem is None:
                continue
            arm = max(
                nom.armature * g_arm * (1.0 + rng.normal(0.0, ARM_SIDE_STD)), 1e-6
            )
            damp = max(
                nom.damping * g_damp * (1.0 + rng.normal(0.0, DAMP_SIDE_STD)), 1e-6
            )
            fric = max(
                nom.friction * g_fric * (1.0 + rng.normal(0.0, FRIC_SIDE_STD)), 1e-6
            )
            joint_updates[jname] = {
                "armature": _fmt(arm),
                "damping": _fmt(damp),
                "friction": _fmt(fric),
            }
            n_joints += 1
            if verbose:
                print(
                    f"  [joint] {jname:<22s} fric {nom.friction:.3f}->{fric:.3f} "
                    f"damp {nom.damping:.3f}->{damp:.3f} "
                    f"arm {nom.armature:.5f}->{arm:.5f}"
                )

    if dry_run:
        text = original_text
        added_env: list[str] = []
        if with_env:
            text, added_env = _ensure_env_in_text(nominal_text, text)
        extra = f"  env to add: {added_env}" if added_env else ""
        print(
            f"\nDRY RUN — would update {n_links} links, {n_joints} joints."
            f"{extra}  Nothing written."
        )
        return

    text = original_text
    added_env: list[str] = []
    if with_env:
        text, added_env = _ensure_env_in_text(nominal_text, text)

    new_text = _rewrite_preserving(text, link_updates, joint_updates)
    true_urdf.write_text(new_text, encoding="utf-8")
    if verbose:
        env_note = f"  (env added: {added_env})" if added_env else ""
        print(f"\nWrote {true_urdf}  ({n_links} links, {n_joints} joints{env_note}).")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(true_urdf: Path) -> None:
    """Re-parse the written URDF and verify physical consistency."""
    root = ET.parse(str(true_urdf)).getroot()

    problems: list[str] = []
    revolute = 0
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        revolute += 1
        d = joint.find("dynamics")
        if d is None or d.attrib.get("friction") is None:
            problems.append(f"{joint.attrib['name']}: missing friction")
        if d is None or d.attrib.get("armature") is None:
            problems.append(f"{joint.attrib['name']}: missing armature")
        if d is not None and d.attrib.get("frictionloss") is not None:
            problems.append(f"{joint.attrib['name']}: still has frictionloss")

    checked = 0
    for link in root.findall("link"):
        iel = link.find("inertial")
        if iel is None:
            continue
        mass = float(iel.find("mass").attrib["value"])
        if mass < MIN_GENERATED_MASS:
            continue  # dummy feet
        ine = iel.find("inertia")
        I6 = {k: float(ine.attrib[k]) for k in INERTIA_KEYS}
        M = _mat3(**I6)
        w = np.sort(np.linalg.eigvalsh(M))
        checked += 1
        if w[0] <= 0:
            problems.append(f"{link.attrib['name']}: not positive-definite")
        if w[2] >= w[0] + w[1]:
            problems.append(
                f"{link.attrib['name']}: triangle violated "
                f"{w[2]:.6f} >= {w[0]:.6f}+{w[1]:.6f}"
            )

    print(f"Validation: {revolute} revolute joints, {checked} physical links")
    if problems:
        print("FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(
        "OK — all joints use armature+damping+friction; all inertias are "
        "positive-definite and satisfy the triangle inequality."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate serial_pm_v2_identify_true.urdf (real-robot "
        "ground truth) from the nominal identify.urdf."
    )
    p.add_argument("--nominal", type=Path, default=DEFAULT_NOMINAL)
    p.add_argument(
        "--true", type=Path, default=DEFAULT_TRUE, help="output true URDF (overwritten)"
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--mass-lo",
        type=float,
        default=MASS_LO,
        help="lower mass gain, e.g. 0.01 = +1%",
    )
    p.add_argument(
        "--mass-hi",
        type=float,
        default=MASS_HI,
        help="upper mass gain, e.g. 0.05 = +5%",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print planned changes without writing"
    )
    p.add_argument(
        "--no-env",
        action="store_true",
        help="do NOT copy the simulation environment (tables) from the nominal URDF",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true", help="suppress per-link/joint lines"
    )
    return p


def main(argv: list[str] | None = None) -> None:
    global MASS_LO, MASS_HI
    args = _build_parser().parse_args(argv)
    MASS_LO, MASS_HI = args.mass_lo, args.mass_hi
    if not args.nominal.is_file():
        raise FileNotFoundError(f"nominal URDF not found: {args.nominal}")
    if not args.true.is_file():
        raise FileNotFoundError(f"true URDF not found: {args.true}")
    _generate(
        args.nominal,
        args.true,
        args.seed,
        args.dry_run,
        verbose=not args.quiet,
        with_env=not args.no_env,
    )
    if not args.dry_run:
        _validate(args.true)


if __name__ == "__main__":
    main()
