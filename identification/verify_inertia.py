#!/usr/bin/env python3
"""Verify URDF inertia values vs Pinocchio transformed values."""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pinocchio as pin
import sys


def main():
    urdf = (
        Path(__file__).resolve().parent.parent
        / "resource"
        / "robot"
        / "urdf"
        / "serial_pm_v2_identify_nominal.urdf"
    ).resolve()
    print(f"URDF: {urdf}  (exists={urdf.is_file()})")
    if not urdf.is_file():
        sys.exit(1)

    # 1. Parse URDF — capture inertial origin + inertia, plus joint tree
    tree = ET.parse(str(urdf))
    root = tree.getroot()

    urdf_data = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        iel = link.find("inertial")
        if iel is None:
            continue
        mass = float(iel.find("mass").attrib["value"])
        ine = iel.find("inertia")
        I = [float(ine.attrib[k]) for k in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]]
        orig_el = iel.find("origin")
        if orig_el is not None:
            xyz = [float(x) for x in orig_el.attrib.get("xyz", "0 0 0").split()]
        else:
            xyz = [0.0, 0.0, 0.0]
        urdf_data[name] = {"mass": mass, "I": I, "inertial_xyz": np.array(xyz)}

    # Build parent→child mapping for fixed joints (Pinocchio merges these)
    fixed_children = {}
    fixed_joint_origin = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type", "revolute") == "fixed":
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            fixed_children.setdefault(parent, []).append(child)
            orig_el = joint.find("origin")
            if orig_el is not None:
                xyz = [float(x) for x in orig_el.attrib.get("xyz", "0 0 0").split()]
            else:
                xyz = [0.0, 0.0, 0.0]
            fixed_joint_origin[(parent, child)] = np.array(xyz)

    def to_mat(I6):
        """URDF order: [ixx, ixy, ixz, iyy, iyz, izz]"""
        return np.array(
            [
                [I6[0], I6[1], I6[2]],
                [I6[1], I6[3], I6[4]],
                [I6[2], I6[4], I6[5]],
            ]
        )

    def to_mat_pin(I6):
        """Pinocchio order: [ixx, ixy, iyy, ixz, iyz, izz] (row-major upper triangle)"""
        return np.array(
            [
                [I6[0], I6[1], I6[3]],
                [I6[1], I6[2], I6[4]],
                [I6[3], I6[4], I6[5]],
            ]
        )

    def parallel_axis(I_com, mass, d):
        """Transform inertia from COM frame to a frame offset by vector d."""
        return I_com + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    def get_merged_inertia(link_name):
        """Return (mass, I_matrix_at_merged_COM, merged_COM_in_link_frame).
        Recursively merges fixed-joint children (like Pinocchio does)."""
        d = urdf_data[link_name]
        m_total = d["mass"]
        I_com = to_mat(d["I"])
        c_com = d["inertial_xyz"].copy()
        # sum of (mass * COM) for weighted average
        mc_sum = m_total * c_com.astype(float)

        for child in fixed_children.get(link_name, []):
            child_m, child_I, child_c_in_child = get_merged_inertia(child)
            # child COM in link frame = joint_origin + child_inertial_xyz
            joint_orig = fixed_joint_origin.get((link_name, child), np.zeros(3))
            child_c = joint_orig + child_c_in_child
            mc_sum += child_m * child_c
            m_total += child_m

        c_merged = mc_sum / m_total

        # Transform each body's inertia from its COM to merged COM
        I_merged = parallel_axis(I_com, d["mass"], c_merged - c_com)
        for child in fixed_children.get(link_name, []):
            child_m, child_I, child_c_in_child = get_merged_inertia(child)
            joint_orig = fixed_joint_origin.get((link_name, child), np.zeros(3))
            child_c = joint_orig + child_c_in_child
            I_merged += parallel_axis(child_I, child_m, c_merged - child_c)

        return m_total, I_merged, c_merged

    # 2. Pinocchio
    model = pin.buildModelFromUrdf(str(urdf))
    # URDF uses LINK_* for <link> names, Pinocchio uses J* (joint names) in model.names
    targets = {
        "LINK_SHOULDER_PITCH_L": "J13_SHOULDER_PITCH_L",
        "LINK_SHOULDER_ROLL_L": "J14_SHOULDER_ROLL_L",
        "LINK_SHOULDER_YAW_L": "J15_SHOULDER_YAW_L",
        "LINK_ELBOW_PITCH_L": "J16_ELBOW_PITCH_L",
        "LINK_ELBOW_YAW_L": "J17_ELBOW_YAW_L",
    }

    print(
        f"\n{'Link':<28s} {'URDF eigvals (→joint)':<40s} {'Pin eigvals (joint frame)':<40s} Match?"
    )
    print("-" * 120)

    all_ok = True
    for link_name, joint_name in targets.items():
        mass, I_com, c_com = get_merged_inertia(link_name)
        I_joint = parallel_axis(I_com, mass, c_com)  # COM → joint (link) frame

        eu = np.sort(np.linalg.eigvalsh(I_joint))

        jid = next(
            (j for j in range(1, model.njoints) if model.names[j] == joint_name), None
        )
        if jid is None:
            print(f"{link_name:<28s} NOT IN MODEL")
            continue

        pin_inertia = model.inertias[jid]
        dp = pin_inertia.toDynamicParameters()
        Ip = to_mat_pin(dp[4:10])  # Pinocchio uses row-major upper-triangle order
        ep = np.sort(np.linalg.eigvalsh(Ip))

        eu_s = "[" + ",".join(f"{e: .4e}" for e in eu) + "]"
        ep_s = "[" + ",".join(f"{e: .4e}" for e in ep) + "]"
        ok = np.allclose(eu, ep, rtol=1e-10)
        all_ok &= ok
        print(f"{link_name:<28s} {eu_s:<40s} {ep_s:<40s} {'OK' if ok else 'DIFF'}")

    print("-" * 120)
    print(f"All eigenvalues match: {all_ok}")

    # 3. TargetLimbRegressor check (requires ROS 2 workspace to be sourced)
    print("\n--- TargetLimbRegressor indexing ---")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from identification.target_limb_regressor import TargetLimbRegressor

        reg = TargetLimbRegressor(
            urdf_path=urdf, group_to_identify="left_arm", print_info=False
        )
        for idx, jid in enumerate(reg.group_to_identify):
            pj = jid + 1
            dp = reg.model.inertias[pj].toDynamicParameters()
            Ip = to_mat_pin(dp[4:10])
            ep = np.sort(np.linalg.eigvalsh(Ip))
            print(
                f"  [{idx}] {reg.target_joint_infos[idx]['name']}: "
                f"child_link={model.names[pj]}  m={dp[0]:.6g}  "
                f"eig={[f'{e:.4e}' for e in ep]}"
            )
    except Exception as e:
        print(f"  (skipped — {e})")


if __name__ == "__main__":
    print("=== Verify URDF inertia values vs Pinocchio transformed values ===")
    main()
