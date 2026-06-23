#!/usr/bin/env python3

import numpy as np

# ==============================
# Tunable trajectory configuration
# ==============================

# Series expansion parameters
N_HARMONICS = 5
TRAJ_PERIOD = 10.0
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


def main():
    dim = 5
    q_traj, v_traj, a_traj = FourierTrajectory(
        dim=dim, sample_rate=100
    ).generate_trajectory(
        coeffs=np.array(
            [
                1.62747080e-01,
                -1.35994919e-01,
                -9.75755707e-02,
                -3.15167643e-01,
                -5.23979956e-01,
                4.63712076e-01,
                6.98639941e-01,
                2.35656746e-01,
                -8.73299926e-01,
                -2.69208837e-01,
                -9.21240000e-01,
                -5.42230741e-02,
                -7.01016614e-02,
                -1.50428188e-01,
                1.27949633e-01,
                1.87742830e-01,
                1.26563514e-01,
                -3.47711475e-01,
                -3.47711475e-01,
                2.67178020e-01,
                3.98071414e-02,
                4.57650000e-01,
                -4.90937107e-02,
                -1.49872408e-01,
                3.16421212e-01,
                3.04698778e-01,
                -4.73053389e-01,
                4.15575151e-01,
                -2.28100405e-01,
                2.69538005e-01,
                7.91053030e-01,
                7.91053030e-01,
                2.77387025e-01,
                -8.58345945e-02,
                6.92765476e-02,
                -1.54501135e-01,
                1.06842117e-01,
                1.61672021e-01,
                2.57503783e-01,
                -3.43338378e-01,
                5.45056611e-02,
                4.29172972e-01,
                -3.32355829e-01,
                -6.88704113e-01,
                -1.19099379e-03,
                1.58210606e-01,
                3.16421212e-01,
                2.33777711e-01,
                -4.74631818e-01,
                -4.74631818e-01,
                6.32842424e-01,
                5.79828861e-01,
                3.33286477e-01,
                7.91053030e-01,
                -7.55400000e-01,
            ]  # -4400, 88 strong
        )
    )
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
