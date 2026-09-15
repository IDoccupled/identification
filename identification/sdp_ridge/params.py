"""Per-joint parameter layout (13 scalars) and the result container.

Pinocchio ``Inertia::toDynamicParameters()`` ordering::

    [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, armature, damping, friction]
     0    1     2     3     4    5    6    7    8    9     10        11       12

Everything else in :mod:`identification.sdp_ridge` addresses parameters through
this 13-block layout (``N_PER_JOINT`` / ``PARAM_LABELS``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


N_PER_JOINT = 13

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


def split_joint_params(pi_full: np.ndarray) -> list[np.ndarray]:
    """(dof*13,) → list of dof × (13,)."""
    dof = len(pi_full) // N_PER_JOINT
    return [pi_full[i * N_PER_JOINT : (i + 1) * N_PER_JOINT] for i in range(dof)]


def join_joint_params(pi_list: list[np.ndarray]) -> np.ndarray:
    """list of dof × (13,) → (dof*13,)."""
    return np.concatenate(pi_list)


@dataclass
class IdentificationResult:
    pi_identified: np.ndarray  # (dof*13,)
    pi_prior: np.ndarray  # (dof*13,)  — from URDF
    pi_reference: np.ndarray | None  # (dof*13,)  — ground-truth (None if unknown)
    joint_solve_times: list[float]
    joint_objectives: list[float]
    joint_order: list[int]
