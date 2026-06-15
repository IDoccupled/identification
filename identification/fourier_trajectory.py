#!/usr/bin/env python3

import numpy as np

# ==============================
# Tunable trajectory configuration
# ==============================

# Series expansion parameters
N_HARMONICS = 5
TRAJ_PERIOD = 10.0
SAMPLE_RATE   = 50.0


class FourierTrajectory:
    def __init__(
            self,
            dim: int
    ):        

        self.omega_f      = 2.0 * np.pi / TRAJ_PERIOD
        self.t_array     = np.linspace(0, TRAJ_PERIOD, int(TRAJ_PERIOD * SAMPLE_RATE), endpoint=False)
        self.n_harmonics = N_HARMONICS
        self.dim = dim

    def generate_trajectory(self, coeffs: np.ndarray):
        """
        Generate joint trajectories (q, v, a) from Fourier coefficients.

        :param coeffs: Flattened array of shape (dim * n_harmonics * 2,) containing [A1_1, B1_1, A2_1, B2_1, ..., A_N_1, B_N_1, q_1, t_1,
                                                                                 A1_2, B1_2, A2_2, B2_2, ..., A_N_2, B_N_2, q_2, t_2,
                                                                                 ...] for each joint.
        :return: Tuple of (q_traj, v_traj, a_traj), each of shape (len(t_array), dim)
        """
        params = coeffs.reshape(self.dim, self.n_harmonics * 2 + 2)
        q_traj = np.zeros((self.dim, len(self.t_array)))
        v_traj = np.zeros_like(q_traj)
        a_traj = np.zeros_like(q_traj)

        for i in range(self.dim):
            q0 = params[i, -2]
            t0 = params[i, -1]
            a = params[i, 0 : self.n_harmonics * 2 : 2]
            b = params[i, 1 : self.n_harmonics * 2 : 2]
            harmonics = np.arange(1, self.n_harmonics + 1, dtype=float)
            w = self.omega_f * harmonics

            sin_wt = np.sin(np.outer(self.t_array + t0, w))
            cos_wt = np.cos(np.outer(self.t_array + t0, w))

            q_traj[i, :] = (
                q0 
                + sin_wt @ (a / w) 
                - cos_wt @ (b / w)
            )
            v_traj[i, :] = (
                cos_wt @ a 
                + sin_wt @ b
            )
            a_traj[i, :] = (
                - sin_wt @ (a * w) 
                + cos_wt @ (b * w)
            )
        return q_traj, v_traj, a_traj
    
def main():
    dim = 5
    q_traj, v_traj, a_traj = FourierTrajectory(dim=dim).generate_trajectory(
        coeffs=np.random.uniform(-1, 1, size=(dim * (N_HARMONICS * 2 + 2)))
    )
    print("生成的轨迹形状:", q_traj.shape)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    for i in range(q_traj.shape[0]):
        plt.subplot(q_traj.shape[0], 1, i + 1)
        plt.plot(q_traj[i, :])
        plt.title(f"Joint {i}")
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()