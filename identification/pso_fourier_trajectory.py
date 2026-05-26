#!/usr/bin/env python3

import numpy as np
import pinocchio as pin
import math
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header
import yaml # type: ignore

from sko.PSO import PSO

from target_limb_regressor import TargetLimbRegressor as Regressor

# ==============================
# File saving related constants
# ==============================

YAML_FILE_NAME = "opt_trajectory_4_14_1.yaml"

YAML_PATH = (
    Path(__file__).resolve().parent / ".." / "config" / YAML_FILE_NAME
).resolve()

# ==============================
# Tunable trajectory configuration
# ==============================

# Series expansion parameters
N_HARMONICS = 5
TRAJ_PERIOD = 10.0
N_SAMPLES   = 1000

# Soft constraint parameters
REG_EPS = 1e-6
RANK_REL_TOL = 1e-4
RANK_ABS_TOL = 1e-10

# Normalized penalty weights
PENALTY_W_Q = 2.0
PENALTY_W_V = 1.0
PENALTY_W_TAU = 2.0
PENALTY_W_MAX = 5.0

# Progressive penalty schedule: lambda(k) = lambda0 * (1 + alpha * progress)
PENALTY_LAMBDA0 = 400.0
PENALTY_LAMBDA_ALPHA = 2.0

# Limit buffers
Q_LIMIT_BUFFER = 0.1
V_LIMIT_BUFFER = 0.15
EFFORT_LIMIT_BUFFER = 0.15

# ==============================
# PSO parameters
# ==============================
RNG_SEED = 47
# 6: 221, 46:225
# new: 224

POP_SIZE = 10
MAX_ITER = 600
PSO_W    = 0.7
PSO_C1   = 1.5
PSO_C2   = 1.5

class PSOFourierTrajectory:
    def __init__(self):
        self.regressor = Regressor(
            group_to_identify='left_arm',
        )
        self.nq = np.size(self.regressor.target_v_indices)
        
        self.limits = self.regressor.limits
        self.joint_names = [
            str(info.get("name", "unknown")) for info in self.regressor.joint_infos
        ]
        self.q_lower_limits = self.limits["q_lower"] + Q_LIMIT_BUFFER
        self.q_upper_limits = self.limits["q_upper"] - Q_LIMIT_BUFFER
        self.v_limits = self.limits["v_limit"] - V_LIMIT_BUFFER
        self.effort_limits = self.limits["effort_limit"] - EFFORT_LIMIT_BUFFER

        self.omega0 = 2.0 * np.pi / TRAJ_PERIOD
        self.t_array = np.linspace(0.0, TRAJ_PERIOD, N_SAMPLES)
        self.n_harmonics = N_HARMONICS
        self.dim = self.nq * (2 * self.n_harmonics + 1)  # q0 + a_k + b_k for each joint

        self.eval_count = 0
        self.max_expected_evals = max(1, POP_SIZE * MAX_ITER)
        self._reset_iter_log_state()

        np.random.seed(RNG_SEED)


    def _reset_iter_log_state(self):
        self._iter_eval_counter = 0
        self._iter_best_objective = np.inf
        self._iter_best_info = np.nan
        self._iter_best_info_eff = np.nan
        self._iter_best_penalty = np.nan
        self._iter_best_weighted_penalty = np.nan
        self._iter_best_max_violation = np.nan
        self._iter_best_rank = -1
        self._iter_best_sval_count = 0
        self._iter_best_sigma_r = np.nan
        self._iter_best_kappa_eff = np.nan
        self._iter_best_q_over_max = np.nan
        self._iter_best_v_over_max = np.nan
        self._iter_best_tau_over_max = np.nan


    def _effective_identifiability_metrics(self, svals):
        sigma_max = float(svals[0])
        rank_tol = max(RANK_ABS_TOL, RANK_REL_TOL * sigma_max)
        rank = int(np.sum(svals > rank_tol))

        if rank > 0:
            sigma_r = float(svals[rank - 1])
            kappa_eff = np.inf if sigma_r <= 1e-12 else sigma_max / sigma_r
            info_eff = float(np.sum(np.log(svals[:rank] ** 2 + REG_EPS)))
        else:
            sigma_r = 0.0
            kappa_eff = np.inf
            info_eff = float("-inf")

        return rank, sigma_r, kappa_eff, info_eff


    def _log_once_per_iteration(
        self,
        objective,
        info_term,
        info_eff,
        penalty,
        weighted_penalty,
        max_violation,
        rank,
        sval_count,
        sigma_r,
        kappa_eff,
        q_over_max,
        v_over_max,
        tau_over_max,
    ):
        self._iter_eval_counter += 1

        if objective < self._iter_best_objective:
            self._iter_best_objective = float(objective)
            self._iter_best_info = float(info_term)
            self._iter_best_info_eff = float(info_eff)
            self._iter_best_penalty = float(penalty)
            self._iter_best_weighted_penalty = float(weighted_penalty)
            self._iter_best_max_violation = float(max_violation)
            self._iter_best_rank = int(rank)
            self._iter_best_sval_count = int(sval_count)
            self._iter_best_sigma_r = float(sigma_r)
            self._iter_best_kappa_eff = float(kappa_eff)
            self._iter_best_q_over_max = float(q_over_max)
            self._iter_best_v_over_max = float(v_over_max)
            self._iter_best_tau_over_max = float(tau_over_max)

        if self._iter_eval_counter >= POP_SIZE:
            batch_idx = self.eval_count // POP_SIZE
            phase = "Init" if batch_idx <= 1 else f"Iter {batch_idx - 2}"
            kappa_text = (
                "inf"
                if not np.isfinite(self._iter_best_kappa_eff)
                else f"{self._iter_best_kappa_eff:.6f}"
            )
            sigma_r_text = (
                "nan"
                if not np.isfinite(self._iter_best_sigma_r)
                else f"{self._iter_best_sigma_r:.3e}"
            )
            info_eff_text = (
                "-inf"
                if np.isneginf(self._iter_best_info_eff)
                else f"{self._iter_best_info_eff:.6f}"
            )
            rank_text = (
                "NA"
                if self._iter_best_rank < 0 or self._iter_best_sval_count <= 0
                else f"{self._iter_best_rank}/{self._iter_best_sval_count}"
            )
            q_over_text = (
                "nan"
                if not np.isfinite(self._iter_best_q_over_max)
                else f"{self._iter_best_q_over_max:.6f}"
            )
            v_over_text = (
                "nan"
                if not np.isfinite(self._iter_best_v_over_max)
                else f"{self._iter_best_v_over_max:.6f}"
            )
            tau_over_text = (
                "nan"
                if not np.isfinite(self._iter_best_tau_over_max)
                else f"{self._iter_best_tau_over_max:.6f}"
            )
            limit_exceeded = (
                np.isfinite(self._iter_best_q_over_max)
                and np.isfinite(self._iter_best_v_over_max)
                and np.isfinite(self._iter_best_tau_over_max)
                and max(
                    self._iter_best_q_over_max,
                    self._iter_best_v_over_max,
                    self._iter_best_tau_over_max,
                )
                > 1e-12
            )
            exceed_tag = " LIMIT_EXCEEDED" if limit_exceeded else ""
            print(
                f"{phase}: Obj={self._iter_best_objective:.6f}, "
                f"Info={self._iter_best_info:.6f}, InfoEff={info_eff_text}, "
                f"Rank={rank_text}, SigmaR={sigma_r_text}, KappaEff={kappa_text}, "
                f"Penalty={self._iter_best_penalty:.6f}, "
                f"LambdaPenalty={self._iter_best_weighted_penalty:.6f}, "
                f"MaxViolation={self._iter_best_max_violation:.6f}, "
                f"OverQ={q_over_text}, OverV={v_over_text}, OverTau={tau_over_text}"
                f"{exceed_tag}"
            )
            self._reset_iter_log_state()


    def _progressive_penalty_lambda(self):
        progress = min(self.eval_count / float(self.max_expected_evals), 1.0)
        return PENALTY_LAMBDA0 * (1.0 + PENALTY_LAMBDA_ALPHA * progress)


    def fitness_function(self, x):
        self.eval_count += 1
        q, dq, ddq = self.generate_trajectory(x)

        q_violation, v_violation = self.normalized_violations(
            q,
            dq,
            self.q_lower_limits,
            self.q_upper_limits,
            self.v_limits,
        )
        q_penalty = np.mean(q_violation ** 2)
        v_penalty = np.mean(v_violation ** 2)
        q_low_over = np.maximum(self.q_lower_limits - q, 0.0)
        q_high_over = np.maximum(q - self.q_upper_limits, 0.0)
        q_over_max = float(np.max(np.maximum(q_low_over, q_high_over)))
        v_over_max = float(np.max(np.maximum(np.abs(dq) - self.v_limits, 0.0)))

        W_list = []
        tau_violation_list = []
        tau_excess_list = []
        for qi, dqi, ddqi in zip(q, dq, ddq):
            W_cur, tau_cur = self.regressor.compute_regressor(qi, dqi, ddqi)
            if not np.all(np.isfinite(W_cur)):
                objective = 1e9
                self._log_once_per_iteration(
                    objective=objective,
                    info_term=np.nan,
                    info_eff=np.nan,
                    penalty=np.nan,
                    weighted_penalty=np.nan,
                    max_violation=np.nan,
                    rank=-1,
                    sval_count=0,
                    sigma_r=np.nan,
                    kappa_eff=np.inf,
                    q_over_max=np.nan,
                    v_over_max=np.nan,
                    tau_over_max=np.nan,
                )
                return objective
            if not np.all(np.isfinite(tau_cur)):
                objective = 1e9
                self._log_once_per_iteration(
                    objective=objective,
                    info_term=np.nan,
                    info_eff=np.nan,
                    penalty=np.nan,
                    weighted_penalty=np.nan,
                    max_violation=np.nan,
                    rank=-1,
                    sval_count=0,
                    sigma_r=np.nan,
                    kappa_eff=np.inf,
                    q_over_max=np.nan,
                    v_over_max=np.nan,
                    tau_over_max=np.nan,
                )
                return objective

            tau_violation = np.maximum(
                np.abs(tau_cur) / np.maximum(self.effort_limits, 1e-6) - 1.0,
                0.0,
            )
            tau_excess = np.maximum(np.abs(tau_cur) - self.effort_limits, 0.0)
            tau_violation_list.append(tau_violation)
            tau_excess_list.append(tau_excess)
            W_list.append(W_cur)

        tau_violation_array = np.vstack(tau_violation_list)
        tau_excess_array = np.vstack(tau_excess_list)
        tau_penalty = np.mean(tau_violation_array ** 2)
        tau_over_max = float(np.max(tau_excess_array))

        max_violation = max(
            float(np.max(q_violation)),
            float(np.max(v_violation)),
            float(np.max(tau_violation_array)),
        )
        penalty = (
            PENALTY_W_Q * q_penalty
            + PENALTY_W_V * v_penalty
            + PENALTY_W_TAU * tau_penalty
            + PENALTY_W_MAX * (max_violation ** 2)
        )

        W_total = np.vstack(W_list)

        # Normalize columns before evaluating excitation index to reduce scaling artifacts.
        col_norm = np.linalg.norm(W_total, axis=0)
        valid_cols = col_norm > 1e-12
        if not np.any(valid_cols):
            objective = 1e9
            self._log_once_per_iteration(
                objective=objective,
                info_term=np.nan,
                info_eff=np.nan,
                penalty=np.nan,
                weighted_penalty=np.nan,
                max_violation=np.nan,
                rank=-1,
                sval_count=0,
                sigma_r=np.nan,
                kappa_eff=np.inf,
                q_over_max=np.nan,
                v_over_max=np.nan,
                tau_over_max=np.nan,
            )
            return objective

        W_scaled = W_total[:, valid_cols] / col_norm[valid_cols]

        try:
            svals = np.linalg.svd(W_scaled, compute_uv=False)
        except np.linalg.LinAlgError:
            objective = 1e9
            self._log_once_per_iteration(
                objective=objective,
                info_term=np.nan,
                info_eff=np.nan,
                penalty=np.nan,
                weighted_penalty=np.nan,
                max_violation=np.nan,
                rank=-1,
                sval_count=0,
                sigma_r=np.nan,
                kappa_eff=np.inf,
                q_over_max=np.nan,
                v_over_max=np.nan,
                tau_over_max=np.nan,
            )
            return objective

        info_score = np.sum(np.log(svals ** 2 + REG_EPS))
        rank, sigma_r, kappa_eff, info_eff = self._effective_identifiability_metrics(svals)

        penalty_lambda = self._progressive_penalty_lambda()
        weighted_penalty = penalty_lambda * penalty
        objective = -info_score + weighted_penalty

        self._log_once_per_iteration(
            objective=objective,
            info_term=-info_score,
            info_eff=info_eff,
            penalty=penalty,
            weighted_penalty=weighted_penalty,
            max_violation=max_violation,
            rank=rank,
            sval_count=int(svals.size),
            sigma_r=sigma_r,
            kappa_eff=kappa_eff,
            q_over_max=q_over_max,
            v_over_max=v_over_max,
            tau_over_max=tau_over_max,
        )

        return objective
    

    def generate_trajectory(self, x):
        """Generate q, dq, ddq from truncated Fourier coefficients."""
        q = np.zeros((len(self.t_array), self.nq))
        dq = np.zeros((len(self.t_array), self.nq))
        ddq = np.zeros((len(self.t_array), self.nq))

        params = x.reshape(self.nq, 1 + 2 * self.n_harmonics)
        harmonics = np.arange(1, self.n_harmonics + 1, dtype=float)
        w = self.omega0 * harmonics

        sin_wt = np.sin(np.outer(self.t_array, w))
        cos_wt = np.cos(np.outer(self.t_array, w))

        for i in range(self.nq):
            q0 = params[i, 0]
            a = params[i, 1:self.n_harmonics + 1]
            b = params[i, self.n_harmonics + 1:]

            q[:, i] = q0 + sin_wt @ (a / w) - cos_wt @ (b / w)
            dq[:, i] = cos_wt @ a + sin_wt @ b
            ddq[:, i] = -sin_wt @ (a * w) + cos_wt @ (b * w)

        return q, dq, ddq
    

    def normalized_violations(self, q, dq, q_min, q_max, v_max):
        """Compute normalized position/velocity violation ratios over the full trajectory."""
        q_span = np.maximum(q_max - q_min, 1e-6)
        q_low_violation = np.maximum((q_min - q) / q_span, 0.0)
        q_high_violation = np.maximum((q - q_max) / q_span, 0.0)
        q_violation = q_low_violation + q_high_violation

        v_violation = np.maximum(np.abs(dq) / np.maximum(v_max, 1e-6) - 1.0, 0.0)
        return q_violation, v_violation


    def build_search_bounds(self):
        """Build PSO bounds from joint limits and velocity limits."""
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)

        for j in range(self.nq):
            offset = j * (1 + 2 * self.n_harmonics)

            span = self.q_upper_limits[j] - self.q_lower_limits[j]
            if span > 1e-6:
                margin = 0.2 * span
                q0_low = self.q_lower_limits[j] + margin
                q0_high = self.q_upper_limits[j] - margin
                if q0_low >= q0_high:
                    center = 0.5 * (self.q_lower_limits[j] + self.q_upper_limits[j])
                    q0_low = center - 0.05 * span
                    q0_high = center + 0.05 * span
            else:
                q0_low, q0_high = -0.5, 0.5

            # Conservative coefficient limit so sum(|a_k|+|b_k|) roughly stays under v_max.
            coeff_max = 0.7 * self.v_limits[j] / (2.0 * self.n_harmonics)
            coeff_max = float(np.clip(coeff_max, 0.03, 0.2))

            lb[offset] = q0_low
            ub[offset] = q0_high
            lb[offset + 1:offset + 1 + 2 * self.n_harmonics] = -coeff_max
            ub[offset + 1:offset + 1 + 2 * self.n_harmonics] = coeff_max

        return lb, ub
    

    def save_coeffs_as_yaml(self,coeffs, fitness_value, yaml_path = YAML_PATH):
        """Save flattened PSO coefficient vector as readable YAML."""
        coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
        fitness_value = float(fitness_value)

        params = coeffs.reshape(self.nq, self.n_harmonics * 2 + 1)

        data = {
            "training_hyperparameters": {
                "population": int(POP_SIZE),
                "max_iterations": int(MAX_ITER),
                "random_seed": int(RNG_SEED),
                "period": float(TRAJ_PERIOD),
                "w": float(PSO_W),
                "c1": float(PSO_C1),
                "c2": float(PSO_C2),
            },
            "best_pso_coeffs": {
                "d": int(self.nq),
                "n_harmonics": int(self.n_harmonics),
                "dim": int(coeffs.size),
                "fitness": fitness_value,
                "coeff_vector": [float(v) for v in coeffs],
                "joints": [],
            },
        }

        joints = data["best_pso_coeffs"]["joints"]
        for i in range(self.nq):
            name = self.joint_names[i] if i < len(self.joint_names) else f"q[{i}]"
            q0 = float(params[i, 0])
            a = [float(v) for v in params[i, 1 : 1 + self.n_harmonics]]
            b = [float(v) for v in params[i, 1 + self.n_harmonics :]]

            joints.append(
                {
                    "index": int(i),
                    "name": name,
                    "q0": q0,
                    "a": a,
                    "b": b,
                }
            )

        yaml_path = Path(yaml_path)
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

    def _array_to_text(arr):
        """Serialize a 1D array into compact space-separated text."""
        flat = np.asarray(arr, dtype=float).reshape(-1)
        return " ".join(f"{v:.9g}" for v in flat)
    
def main():
    pso_traj = PSOFourierTrajectory()
    lb, ub = pso_traj.build_search_bounds()
    
    pso = PSO(
        func=lambda x: pso_traj.fitness_function(x), 
        dim=pso_traj.dim, 
        pop=POP_SIZE, 
        max_iter=MAX_ITER, 
        w=PSO_W, 
        c1=PSO_C1, 
        c2=PSO_C2, 
        lb=lb, 
        ub=ub,
        verbose=True)
    best_x, best_fitness = pso.run()
    print(f"Best fitness: {best_fitness}")
    # pso_traj.save_coeffs_as_yaml(best_x, best_fitness)
    
if __name__ == "__main__":    
    main()
