#!/usr/bin/env python3
"""Sync MuJoCo joint damping/frictionloss from URDF joint dynamics tags.

This tool reads a URDF file and updates one MJCF file (and optionally included MJCF
files) by matching joint names:
- URDF <dynamics damping="..." friction="..."/>
- MJCF <joint damping="..." frictionloss="..."/>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import xml.etree.ElementTree as ET


JointDynamics = Tuple[str | None, str | None]


def _parse_urdf_dynamics(urdf_file: Path) -> Dict[str, JointDynamics]:
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    result: Dict[str, JointDynamics] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        joint_type = joint.attrib.get("type", "")
        if not name or joint_type == "fixed":
            continue

        dynamics = joint.find("dynamics")
        if dynamics is None:
            continue

        damping = dynamics.attrib.get("damping")
        friction = dynamics.attrib.get("friction")
        if damping is None and friction is None:
            continue

        result[name] = (damping, friction)

    return result


def _collect_mjcf_files(root_mjcf: Path, recursive_includes: bool) -> List[Path]:
    to_visit = [root_mjcf.resolve()]
    visited: Set[Path] = set()
    ordered: List[Path] = []

    while to_visit:
        current = to_visit.pop(0)
        if current in visited:
            continue
        visited.add(current)
        ordered.append(current)

        if not recursive_includes:
            continue

        try:
            tree = ET.parse(current)
            root = tree.getroot()
        except ET.ParseError as exc:
            raise RuntimeError(f"Failed to parse MJCF file: {current} ({exc})") from exc

        for include in root.findall(".//include"):
            include_file = include.attrib.get("file")
            if not include_file:
                continue
            child = (current.parent / include_file).resolve()
            if child.exists() and child not in visited:
                to_visit.append(child)

    return ordered


def _update_one_mjcf(
    mjcf_file: Path, dynamics_map: Dict[str, JointDynamics], dry_run: bool
) -> Tuple[int, int, Set[str]]:
    tree = ET.parse(mjcf_file)
    root = tree.getroot()

    found_joint_names: Set[str] = set()
    changed_joint_count = 0
    field_change_count = 0

    for joint in root.findall(".//joint"):
        joint_name = joint.attrib.get("name")
        if not joint_name or joint_name not in dynamics_map:
            continue

        found_joint_names.add(joint_name)
        damping, friction = dynamics_map[joint_name]
        changed_this_joint = False

        if damping is not None and joint.attrib.get("damping") != damping:
            joint.set("damping", damping)
            field_change_count += 1
            changed_this_joint = True

        if friction is not None and joint.attrib.get("frictionloss") != friction:
            joint.set("frictionloss", friction)
            field_change_count += 1
            changed_this_joint = True

        if changed_this_joint:
            changed_joint_count += 1

    if changed_joint_count > 0 and not dry_run:
        ET.indent(tree, space="    ")
        tree.write(mjcf_file, encoding="utf-8")

    return changed_joint_count, field_change_count, found_joint_names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync MJCF joint damping/frictionloss from URDF joint dynamics.",
    )
    parser.add_argument(
        "--urdf",
        required=True,
        type=Path,
        help="Path to URDF file (source of dynamics).",
    )
    parser.add_argument(
        "--mjcf",
        required=True,
        type=Path,
        help="Path to root MJCF file to update.",
    )
    parser.add_argument(
        "--no-recursive-includes",
        action="store_true",
        help="Do not traverse <include file=...> files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing files.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    urdf_file: Path = args.urdf.resolve()
    mjcf_file: Path = args.mjcf.resolve()
    recursive_includes = not args.no_recursive_includes
    dry_run = args.dry_run

    if not urdf_file.is_file():
        parser.error(f"URDF file does not exist: {urdf_file}")
    if not mjcf_file.is_file():
        parser.error(f"MJCF file does not exist: {mjcf_file}")

    dynamics_map = _parse_urdf_dynamics(urdf_file)
    if not dynamics_map:
        print(f"No URDF dynamics found in: {urdf_file}")
        return 0

    mjcf_files = _collect_mjcf_files(mjcf_file, recursive_includes=recursive_includes)

    total_changed_joints = 0
    total_changed_fields = 0
    matched_joints: Set[str] = set()

    for file_path in mjcf_files:
        changed_joints, changed_fields, found_in_file = _update_one_mjcf(
            file_path,
            dynamics_map=dynamics_map,
            dry_run=dry_run,
        )
        total_changed_joints += changed_joints
        total_changed_fields += changed_fields
        matched_joints.update(found_in_file)

        if changed_joints > 0:
            action = "would update" if dry_run else "updated"
            print(f"{action} {changed_joints} joints in {file_path}")

    unmatched = sorted(set(dynamics_map.keys()) - matched_joints)

    mode = "DRY RUN" if dry_run else "DONE"
    print(
        f"[{mode}] changed joints: {total_changed_joints}, changed fields: {total_changed_fields}"
    )
    print(f"Matched joints: {len(matched_joints)}/{len(dynamics_map)}")
    if unmatched:
        print("URDF joints not found in MJCF (first 20):")
        for name in unmatched[:20]:
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
