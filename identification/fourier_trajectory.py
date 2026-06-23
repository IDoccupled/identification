#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import yaml

# ==============================
# Tunable trajectory configuration
# ==============================

# Series expansion parameters
N_HARMONICS = 5
TRAJ_PERIOD = 5.0
SAMPLE_RATE = 50.0


class FourierTrajectory:
    def __init__(
        self,
        dim: int,
        traj_period=TRAJ_PERIOD,
        sample_rate=SAMPLE_RATE,
    ):

        self.omega_f = 2.0 * np.pi / traj_period
        self.t_array = np.linspace(
            0,
            traj_period,
            int(traj_period * sample_rate),
            endpoint=False,
        )
        self.n_harmonics = N_HARMONICS
        self.dim = dim

    def generate_trajectory(self, coeffs: np.ndarray):
        """
        Generate joint trajectories (q, v, a) from Fourier coefficients.

        :param coeffs: Flattened array of shape (dim * (n_harmonics * 2 + 2 ),) containing [A1_1, B1_1, A2_1, B2_1, ..., A_N_1, B_N_1, q_1,
                                                                                 A1_2, B1_2, A2_2, B2_2, ..., A_N_2, B_N_2, q_2,
                                                                                 ...] for each joint.
        :return: Tuple of (q_traj, v_traj, a_traj), each of shape (len(t_array), dim)
        """
        params = coeffs.reshape(self.dim, self.n_harmonics * 2 + 1)
        q_traj = np.zeros((self.dim, len(self.t_array)))
        v_traj = np.zeros_like(q_traj)
        a_traj = np.zeros_like(q_traj)

        for i in range(self.dim):
            q0 = params[i, -1]
            a = params[i, 0 : self.n_harmonics * 2 : 2]
            b = params[i, 1 : self.n_harmonics * 2 : 2]
            harmonics = np.arange(1, self.n_harmonics + 1, dtype=float)
            w = self.omega_f * harmonics

            sin_wt = np.sin(np.outer(self.t_array, w))
            cos_wt = np.cos(np.outer(self.t_array, w))

            q_traj[i, :] = q0 + sin_wt @ (a / w) - cos_wt @ (b / w)
            v_traj[i, :] = cos_wt @ a + sin_wt @ b
            a_traj[i, :] = -sin_wt @ (a * w) + cos_wt @ (b * w)
        return q_traj, v_traj, a_traj


def load_coeffs_from_yaml(yaml_path: str) -> np.ndarray:
    """
    Load Fourier coefficients from a YAML file and convert to the flat array
    format expected by FourierTrajectory.generate_trajectory().

    YAML format (per joint):
      joint_0:
        a: [a1, a2, ..., aN]
        b: [b1, b2, ..., bN]
        q0: <offset>

    :param yaml_path: Path to the YAML file.
    :return: Flattened coefficient array of shape (dim * (n_harmonics * 2 + 1),).
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    coeffs_list = []
    for joint_key in sorted(data.keys()):
        joint = data[joint_key]
        a = joint["a"]
        b = joint["b"]
        q0 = joint["q0"]
        # Interleave a and b: [a1, b1, a2, b2, ..., aN, bN, q0]
        joint_coeffs = []
        for ai, bi in zip(a, b):
            joint_coeffs.append(ai)
            joint_coeffs.append(bi)
        joint_coeffs.append(q0)
        coeffs_list.extend(joint_coeffs)

    return np.array(coeffs_list)


def get_package_coeffs_path(filename: str = "exciting_trajectory.yaml") -> Path:
    """
    Get the path to a built-in coefficient YAML file inside the
    trajectory_coefficients package directory.

    :param filename: Name of the YAML file (default: 'exciting_trajectory.yaml').
    :return: Absolute Path to the YAML file.
    """
    return Path(__file__).resolve().parent / ".." / "trajectory_coefficients" / filename


def main():
    dim = 5
    yaml_path = get_package_coeffs_path("exciting_trajectory.yaml")
    coeffs = load_coeffs_from_yaml(str(yaml_path))
    print(f"Loaded {len(coeffs)} coefficients from {yaml_path}")

    q_traj, v_traj, a_traj = FourierTrajectory(
        dim=dim, sample_rate=100
    ).generate_trajectory(coeffs=coeffs)
    print("生成的轨迹形状:", q_traj.shape)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    for i in range(q_traj.shape[0]):
        plt.subplot(q_traj.shape[0], 1, i + 1)
        plt.plot(q_traj[i, :])
        plt.title(f"Joint {i}")
        plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
