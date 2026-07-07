#!/usr/bin/env python3
"""
Plot per-stage excitation trajectories from PSO YAML files.

Usage:
  python3 -m identification.plot_trajectories
  python3 -m identification.plot_trajectories path/to/balance.yaml path/to/armature.yaml path/to/friction.yaml
"""

import sys
import os

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import matplotlib.pyplot as plt
from pathlib import Path

from identification.fourier_trajectory import FourierTrajectory
from identification.sysid_balance_weight import LEFT_ARM_XML

# Default: use latest PSO YAMLs
_YAML_DIR = FourierTrajectory._coeffs_dir


def _latest(pattern):
    files = sorted(Path(_YAML_DIR).glob(pattern))
    return str(files[-1]) if files else None


DEFAULT_PATHS = [
    _latest("pso_balance_*.yaml"),
    _latest("pso_armature_*.yaml"),
    _latest("pso_friction_*.yaml"),
]

LABELS = ["Balance (inertia)", "Armature", "Friction"]
COLORS = ["#3498db", "#e74c3c", "#2ecc71"]
JOINT_NAMES = [
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
]

SAMPLE_RATE = 500


def load_trajectory(yaml_path):
    """Load (q, dq, ddq, tau) from a YAML file using the TRUE model."""
    ft = FourierTrajectory(dim=5, sample_rate=SAMPLE_RATE)
    ft.omega_f = 2.0 * np.pi / 5.0
    ft.t_array = np.linspace(0, 5.0, int(5.0 * SAMPLE_RATE), endpoint=False)

    q_traj, dq_traj, ddq_traj = ft.generate_trajectory_from_yaml(yaml_path)

    true_spec = mujoco.MjSpec.from_string(LEFT_ARM_XML)
    model = true_spec.compile()
    data = mujoco.MjData(model)
    tau_true = np.zeros_like(q_traj.T)
    for k in range(len(ft.t_array)):
        data.qpos[:] = q_traj[:, k]
        data.qvel[:] = dq_traj[:, k]
        data.qacc[:] = ddq_traj[:, k]
        mujoco.mj_inverse(model, data)
        tau_true[k] = data.qfrc_inverse.copy()

    return q_traj.T, dq_traj.T, ddq_traj.T, tau_true


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PATHS

    if any(p is None for p in paths):
        print("Error: could not find all YAML files.")
        print(
            "Usage: python -m identification.plot_trajectories [bal.yaml arm.yaml fric.yaml]"
        )
        sys.exit(1)

    # Load all three trajectories
    trajs = []
    for p in paths:
        name = Path(p).stem
        print(f"Loading {name} …")
        trajs.append(load_trajectory(p))

    # Create 4×3 grid
    n_stages = 3
    fig, axes = plt.subplots(
        4, n_stages, figsize=(5 * n_stages + 1, 10), sharex="col", squeeze=False
    )

    ylabels = [
        "Position q (rad)",
        "Velocity dq (rad/s)",
        "Acceleration ddq (rad/s²)",
        "Torque τ (Nm)",
    ]
    joint_colors = plt.cm.tab10(np.linspace(0, 1, 5))

    for col in range(n_stages):
        q, dq, ddq, tau = trajs[col]
        data = [q, dq, ddq, tau]
        t = np.arange(len(q)) * 0.002  # 500 Hz → 2ms

        for row in range(4):
            ax = axes[row, col]
            for j in range(5):
                ax.plot(t, data[row][:, j], color=joint_colors[j], lw=0.8, alpha=0.85)

            if row == 0:
                ax.set_title(LABELS[col], fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabels[row], fontsize=9)
            if row == 3:
                ax.set_xlabel("Time (s)", fontsize=9)
                # Motor torque limit reference lines
                ax.axhline(61, color="gray", ls="--", lw=0.5, alpha=0.5)
                ax.axhline(-61, color="gray", ls="--", lw=0.5, alpha=0.5)
                # Use percentile-based y-limits to avoid extreme outliers stretching the view
                tau_data = data[3]
                vmin, vmax = np.percentile(tau_data, [0, 100])
                margin = 0.1 * (vmax - vmin) if vmax > vmin else 1.0
                ax.set_ylim(vmin - margin, vmax + margin)
            ax.grid(True, alpha=0.3)

    # Legend (top-right corner of figure)
    fig.legend(JOINT_NAMES, loc="upper right", fontsize=8, ncol=1, framealpha=0.9)

    fig.suptitle(
        "Per-Stage Excitation Trajectories — PSO Optimized", fontsize=13, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
