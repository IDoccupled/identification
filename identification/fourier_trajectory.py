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
    # Directory where all coefficient YAML files are stored
    _coeffs_dir = Path(__file__).resolve().parent / ".." / "trajectory_coefficients"

    def __init__(
        self,
        dim: int,
        sample_rate=SAMPLE_RATE,
    ):

        self.omega_f = 2.0 * np.pi / TRAJ_PERIOD
        self.t_array = np.linspace(
            0,
            TRAJ_PERIOD,
            int(TRAJ_PERIOD * sample_rate),
            endpoint=False,
        )
        self.n_harmonics = N_HARMONICS
        self.dim = dim

    @staticmethod
    def load_coeffs(yaml_filename: str) -> np.ndarray:
        """
        Load Fourier coefficients from a YAML file in the trajectory_coefficients
        directory and convert to the flat array format.

        :param yaml_filename: Name of the YAML file (e.g. 'exciting_trajectory.yaml').
        :return: Flattened coefficient array of shape (dim * (n_harmonics * 2 + 1),).
        """
        yaml_path = FourierTrajectory._coeffs_dir / yaml_filename
        assert yaml_path.is_file(), f"YAML file not found: {yaml_path}"
        with open(str(yaml_path), "r") as f:
            data = yaml.safe_load(f)

        coeffs_list = []
        for joint_key in sorted(data.keys()):
            joint = data[joint_key]
            a = joint["a"]
            b = joint["b"]
            q0 = joint["q0"]
            joint_coeffs = []
            for ai, bi in zip(a, b):
                joint_coeffs.append(ai)
                joint_coeffs.append(bi)
            joint_coeffs.append(q0)
            coeffs_list.extend(joint_coeffs)

        return np.array(coeffs_list)

    def generate_trajectory_from_yaml(self, yaml_filename: str):
        """
        Load coefficients from a YAML file and generate joint trajectories.

        :param yaml_filename: Name of the YAML file in trajectory_coefficients/.
        :return: Tuple of (q_traj, v_traj, a_traj), each of shape (len(t_array), dim).
        """
        coeffs = self.load_coeffs(yaml_filename)
        return self.generate_trajectory(coeffs)

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


def get_package_coeffs_path(filename: str = "exciting_trajectory.yaml") -> Path:
    """Convenience: get the full path to a coefficient YAML file."""
    return FourierTrajectory._coeffs_dir / filename


def main():
    dim = 5
    traj = FourierTrajectory(dim=dim, sample_rate=200)
    q_traj, v_traj, a_traj = traj.generate_trajectory_from_yaml("0724_1.yaml")
    print("生成的轨迹形状:", q_traj.shape)

    import matplotlib.pyplot as plt

    n_joints = q_traj.shape[0]
    t = traj.t_array

    # Position
    fig_q, axes_q = plt.subplots(n_joints, 1, figsize=(10, 8), sharex=True)
    fig_q.suptitle("Joint Positions (q)")
    for i in range(n_joints):
        axes_q[i].plot(t, q_traj[i, :], "b", label="q")
        axes_q[i].set_ylabel(f"Joint {i}")
        axes_q[i].grid()
    axes_q[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    # Velocity
    fig_v, axes_v = plt.subplots(n_joints, 1, figsize=(10, 8), sharex=True)
    fig_v.suptitle("Joint Velocities (v)")
    for i in range(n_joints):
        axes_v[i].plot(t, v_traj[i, :], "r", label="v")
        axes_v[i].set_ylabel(f"Joint {i}")
        axes_v[i].grid()
    axes_v[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    # Acceleration
    fig_a, axes_a = plt.subplots(n_joints, 1, figsize=(10, 8), sharex=True)
    fig_a.suptitle("Joint Accelerations (a)")
    for i in range(n_joints):
        axes_a[i].plot(t, a_traj[i, :], "g", label="a")
        axes_a[i].set_ylabel(f"Joint {i}")
        axes_a[i].grid()
    axes_a[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
