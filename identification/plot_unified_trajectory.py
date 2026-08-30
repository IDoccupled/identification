#!/usr/bin/env python3
"""
Plot unified PSO excitation trajectory from a YAML file.

Usage:
  python3 -m identification.plot_unified_trajectory
  python3 -m identification.plot_unified_trajectory path/to/pso_unified.yaml
"""

import sys

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from pathlib import Path

from identification.fourier_trajectory import FourierTrajectory

SAMPLE_RATE = 500


def load_trajectory(yaml_path):
    """Load (q, dq, ddq, tau) from a YAML file using Pinocchio inverse dynamics.

    The limb group is read from the YAML ``_meta.group``, so this script works
    for any identified limb without hard-coded joint indices or names.
    """
    from identification.target_limb_regressor import (
        TargetLimbRegressor,
        VALID_LIMB_GROUPS,
    )

    # Limb group + DOF come from the YAML itself.
    group = FourierTrajectory.load_group(yaml_path)
    dim = len(VALID_LIMB_GROUPS[group])

    ft = FourierTrajectory(dim=dim, sample_rate=SAMPLE_RATE)
    ft.omega_f = 2.0 * np.pi / 5.0
    ft.t_array = np.linspace(0, 5.0, int(5.0 * SAMPLE_RATE), endpoint=False)

    q_traj, dq_traj, ddq_traj = ft.generate_trajectory_from_yaml(yaml_path)

    # Build Pinocchio model via TargetLimbRegressor (URDF-based)
    reg = TargetLimbRegressor(group_to_identify=group, print_info=False)
    joint_names = [reg.target_joint_infos[d]["name"] for d in range(dim)]

    tau = np.zeros((len(ft.t_array), dim))
    for k in range(len(ft.t_array)):
        q_limb = q_traj[:, k]
        dq_limb = dq_traj[:, k]
        ddq_limb = ddq_traj[:, k]

        # Form full 24-DoF state vectors (non-target joints stay at zero)
        q_full, v_full, a_full = reg.state_size_check_and_form(
            q_limb, dq_limb, ddq_limb
        )

        # Inertial torque via Pinocchio Recursive Newton-Euler Algorithm
        tau_inertial = pin.rnea(reg.model, reg.data, q_full, v_full, a_full)
        tau_inertial_limb = tau_inertial[reg.group_to_identify]

        # Armature torque: armature * acceleration
        tau_armature = np.array(
            [reg.target_joint_infos[i]["armature"] * ddq_limb[i] for i in range(dim)]
        )

        # Friction torque: damping * v + friction * tanh(v * 100)
        tau_friction = np.array(
            [
                reg.target_joint_infos[i]["damping"] * dq_limb[i]
                + reg.target_joint_infos[i]["friction"] * np.tanh(dq_limb[i] * 1e2)
                for i in range(dim)
            ]
        )

        tau[k] = tau_inertial_limb + tau_armature + tau_friction

    return q_traj.T, dq_traj.T, ddq_traj.T, tau, joint_names


def main():
    if len(sys.argv) > 1:
        yaml_name = sys.argv[1]
    else:
        try:
            yaml_name = FourierTrajectory.find_latest_yaml()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(
                "Usage: python -m identification.plot_unified_trajectory "
                "[path/to/pso_unified.yaml]"
            )
            sys.exit(1)

    name = Path(yaml_name).name
    print(f"Loading {name} …")
    q, dq, ddq, tau, joint_names = load_trajectory(name)
    dof = q.shape[1]

    ylabels = [
        "Position q (rad)",
        "Velocity dq (rad/s)",
        "Acceleration ddq (rad/s²)",
        "Torque τ (Nm)",
    ]
    joint_colors = plt.cm.tab10(np.linspace(0, 1, dof))

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    data = [q, dq, ddq, tau]
    t = np.arange(len(q)) * (1.0 / SAMPLE_RATE)

    for row in range(4):
        ax = axes[row]
        for j in range(dof):
            ax.plot(t, data[row][:, j], color=joint_colors[j], lw=0.8, alpha=0.85)

        ax.set_ylabel(ylabels[row], fontsize=10)
        ax.grid(True, alpha=0.3)

        if row == 3:
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.axhline(61, color="gray", ls="--", lw=0.5, alpha=0.5)
            ax.axhline(-61, color="gray", ls="--", lw=0.5, alpha=0.5)
            vmin, vmax = np.percentile(data[3], [0, 100])
            margin = 0.1 * (vmax - vmin) if vmax > vmin else 1.0
            ax.set_ylim(vmin - margin, vmax + margin)

    fig.legend(joint_names, loc="upper right", fontsize=8, ncol=1, framealpha=0.9)
    fig.suptitle(f"Unified Excitation Trajectory — {name}", fontsize=13, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
