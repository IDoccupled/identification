#!/usr/bin/env python3
"""
Two-round iterative identification for PM-V2 Left Arm.

  Round 1: Balance → Armature → Frictionloss (wide bounds, URDF prior)
  Round 2: Balance → Armature → Frictionloss (R1 prior, tight bounds)

Each stage uses a PSO-optimized Fourier trajectory tailored to its parameters.
All 5 joints share the same motor → armature/damping/frictionloss are TIED.
"""

import os
import pathlib
from pathlib import Path
import numpy as np
import mujoco
from mujoco import sysid
import matplotlib.pyplot as plt
from identification.fourier_trajectory import FourierTrajectory

os.environ["MUJOCO_GL"] = "egl"

BODY_NAMES = [
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
]

JOINT_NAMES = [
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
]
# Auto-discover latest PSO YAML per stage
_YAML_DIR = FourierTrajectory._coeffs_dir

_RESOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "resource"


def _latest_yaml(stage: str) -> str:
    """Find the most recent pso_{stage}_*.yaml file."""
    import glob

    pattern = str(_YAML_DIR / f"pso_{stage}_*.yaml")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No YAML found for stage '{stage}': {pattern}")
    return files[-1]


STAGE_YAMLS = {
    "balance": _latest_yaml("balance"),
    "armature": _latest_yaml("armature"),
    "friction": _latest_yaml("friction"),
}


def _load_xml(filename):
    """Read an XML model file from the resource directory."""
    return (_RESOURCE_DIR / filename).read_text()


LEFT_ARM_XML = _load_xml("left_arm_true.xml")
NOMINAL_LEFT_ARM_XML = _load_xml("left_arm_nominal.xml")
LEFT_ARM_XML_20PCT = _load_xml("left_arm_20pct.xml")
# USED_XML = LEFT_ARM_XML_20PCT
USED_XML = NOMINAL_LEFT_ARM_XML


def _read_true_joint_params():
    spec = mujoco.MjSpec.from_string(LEFT_ARM_XML)
    out = {}
    for jn in JOINT_NAMES:
        j = spec.joint(jn)
        out[jn] = {
            "armature": float(j.armature),
            "damping": float(j.damping[0]),
            "frictionloss": float(j.frictionloss),
        }
    return out


def _read_nominal_joint_params():
    spec = mujoco.MjSpec.from_string(USED_XML)
    out = {}
    for jn in JOINT_NAMES:
        j = spec.joint(jn)
        out[jn] = {
            "armature": float(j.armature),
            "damping": float(j.damping[0]),
            "frictionloss": float(j.frictionloss),
        }
    return out


def box_inertia(m, sx, sy, sz):
    """Rotational inertia of a box (mass m, size sx×sy×sz) about its own CoM.
    Returns [Ixx, Iyy, Izz]."""
    return [
        m / 12 * (sy**2 + sz**2),
        m / 12 * (sx**2 + sz**2),
        m / 12 * (sx**2 + sy**2),
    ]


def parallel_axis_shift(I_diag, mass, old_com, new_com):
    """Shift diagonal inertia tensor I_diag from old_com to new_com.
    Returns new diagonal [Ixx, Iyy, Izz] (drops off-diagonals)."""
    d = np.array(old_com) - np.array(new_com)
    # Parallel axis theorem: I_new = I_old + mass * (dᵀd·I - ddᵀ)
    # For diagonal: I_ii_new = I_ii_old + mass * (sum(d²) - d_i²)
    d2_sum = np.dot(d, d)
    return np.array(I_diag) + mass * (d2_sum - d**2)


def apply_balances_to_spec(spec, balance_vectors):
    """Apply balance weights: adjust mass, CoM, and rotational inertia.
    balance_vectors: dict body_name -> [mass, px, py, pz, sx, sy, sz]

    Computes the combined inertia of the original body + a box-shaped balance weight
    at position (px,py,pz).  Uses parallel axis theorem to shift both inertias to
    the new combined CoM, then sums diagonals and enforces triangle inequality.
    """
    for bn in BODY_NAMES:
        m, px, py, pz, sx, sy, sz = balance_vectors[bn]
        if abs(m) < 1e-9:
            continue
        b = spec.body(bn)
        old_mass = b.mass
        old_ipos = np.array(b.ipos)
        old_I = np.array(b.inertia)
        balance_pos = np.array([px, py, pz])

        # Combined mass and CoM
        new_mass = old_mass + m
        new_ipos = (old_mass * old_ipos + m * balance_pos) / new_mass

        # Box inertia about its own CoM
        box_I = np.array(box_inertia(m, sx, sy, sz))

        # Shift both inertias to the new combined CoM
        old_I_shifted = parallel_axis_shift(old_I, old_mass, old_ipos, new_ipos)
        box_I_shifted = parallel_axis_shift(box_I, m, balance_pos, new_ipos)

        # Combine — the balance weight is an optimization variable so the
        # combined result may violate physics.  Enforce triangle inequality
        # while preserving the inertia as faithfully as possible.
        combined = old_I_shifted + box_I_shifted
        arr = np.array(combined, dtype=float)
        arr = np.maximum(arr, 1e-10)  # positive diagonal
        for _ in range(20):
            changed = False
            for i, j, k in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
                deficit = arr[k] - (arr[i] + arr[j])
                if deficit > 1e-12:
                    scale = (arr[k] + 1e-10) / max(arr[i] + arr[j], 1e-15)
                    arr[i] *= scale
                    arr[j] *= scale
                    changed = True
            if not changed:
                break

        b.mass = new_mass
        b.ipos = new_ipos
        b.inertia = arr


def load_trajectory(yaml_name):
    """Load (q, dq, ddq, tau_true) from a Fourier YAML, using the TRUE model."""
    ft = FourierTrajectory(dim=5, sample_rate=500)
    ft.t_array = np.linspace(0, 5.0, int(5.0 * 500), endpoint=False)

    q_traj, dq_traj, ddq_traj = ft.generate_trajectory_from_yaml(yaml_name)

    # Compute true torques via mj_inverse
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


def make_residual_fn(init_base, q, dq, ddq, tau_true):
    """Build residual: τ_pred(params) - τ_true."""

    def fn(x, pd):
        if x.ndim == 1:
            pd.update_from_vector(x)
            spec = init_base.copy()
            sysid.apply_param_modifiers_spec(pd, spec)
            bv = {bn: list(pd[f"balance_{bn}"].value) for bn in BODY_NAMES}
            try:
                apply_balances_to_spec(spec, bv)
                model = spec.compile()
            except (ValueError, Exception):
                return [np.full(len(q) * 5, 1e6)], None, None  # huge penalty
            data = mujoco.MjData(model)
            res = []
            for k in range(len(q)):
                data.qpos[:] = q[k]
                data.qvel[:] = dq[k]
                data.qacc[:] = ddq[k]
                mujoco.mj_inverse(model, data)
                res.append(data.qfrc_inverse - tau_true[k])
            return [np.array(res).ravel()], None, None
        else:
            res_list = []
            for ki in range(x.shape[1]):
                pd.update_from_vector(x[:, ki])
                spec = init_base.copy()
                sysid.apply_param_modifiers_spec(pd, spec)
                bv = {bn: list(pd[f"balance_{bn}"].value) for bn in BODY_NAMES}
                try:
                    apply_balances_to_spec(spec, bv)
                    model = spec.compile()
                except (ValueError, Exception):
                    res_list.append(np.full(len(q) * 5, 1e6))
                    continue
                data = mujoco.MjData(model)
                rk = []
                for k in range(len(q)):
                    data.qpos[:] = q[k]
                    data.qvel[:] = dq[k]
                    data.qacc[:] = ddq[k]
                    mujoco.mj_inverse(model, data)
                    rk.append(data.qfrc_inverse - tau_true[k])
                res_list.append(np.array(rk).ravel())
            return [np.column_stack(res_list)], None, None

    return fn


def compute_rmse(init_base, params, q, dq, ddq, tau_true):
    spec = init_base.copy()
    sysid.apply_param_modifiers_spec(params, spec)
    bv = {bn: list(params[f"balance_{bn}"].value) for bn in BODY_NAMES}
    try:
        apply_balances_to_spec(spec, bv)
        model = spec.compile()
    except (ValueError, Exception):
        return np.full(5, 1e6)
    data = mujoco.MjData(model)
    tau_pred = np.zeros_like(tau_true)
    for k in range(len(q)):
        data.qpos[:] = q[k]
        data.qvel[:] = dq[k]
        data.qacc[:] = ddq[k]
        mujoco.mj_inverse(model, data)
        tau_pred[k] = data.qfrc_inverse.copy()
    return np.sqrt(np.mean((tau_pred - tau_true) ** 2, axis=0))


# =========================================================================
def main():
    true_joint = _read_true_joint_params()
    nom_joint = _read_nominal_joint_params()
    init_base = mujoco.MjSpec.from_string(USED_XML)

    a_true = true_joint[JOINT_NAMES[0]]["armature"]
    d_true = true_joint[JOINT_NAMES[0]]["damping"]
    f_true = true_joint[JOINT_NAMES[0]]["frictionloss"]
    a_nom = nom_joint[JOINT_NAMES[0]]["armature"]
    d_nom = nom_joint[JOINT_NAMES[0]]["damping"]
    f_nom = nom_joint[JOINT_NAMES[0]]["frictionloss"]

    # ---- Load per-stage trajectories ----
    print("Loading per-stage trajectories …")
    trajs = {}
    for stage_key, yaml_name in STAGE_YAMLS.items():
        msg = f"{stage_key}: {yaml_name}"
        if stage_key == "balance":
            print(f"\033[92m{msg}\033[0m")
        elif stage_key == "armature":
            print(f"\033[93m{msg}\033[0m")
        else:
            print(f"\033[94m{msg}\033[0m")
        q, dq, ddq, tau = load_trajectory(yaml_name)
        trajs[stage_key] = (q, dq, ddq, tau)
        print(
            f"{len(q)} steps, |ddq|max={np.abs(ddq).max():.1f}, "
            f"|dq|max={np.abs(dq).max():.1f}"
        )

    # ---- Build parameters ----
    params = sysid.ParameterDict()

    # def _damp_shared_mod(s, p):
    #     for jn in JOINT_NAMES:
    #         s.joint(jn).damping = np.array([[p.value[0]], [0.0], [0.0]])

    # Shared joint parameters (all 5 joints tied)
    params.add(
        sysid.Parameter(
            "armature",
            nominal=a_true,
            min_value=0.02,
            max_value=0.06,
            modifier=lambda s, p: [
                setattr(s.joint(jn), "armature", p.value[0]) for jn in JOINT_NAMES
            ],
        )
    )
    params["armature"].value[:] = a_nom
    params["armature"].frozen = True

    params.add(
        sysid.Parameter(
            "damping",
            nominal=d_true,
            min_value=0.04,
            max_value=0.12,
            modifier=lambda s, p: [
                setattr(s.joint(jn), "damping", np.array([[p.value[0]], [0.0], [0.0]]))
                for jn in JOINT_NAMES
            ],
        )
    )
    params["damping"].value[:] = d_nom
    params["damping"].frozen = True

    params.add(
        sysid.Parameter(
            "frictionloss",
            nominal=f_true,
            min_value=0.2,
            max_value=0.4,
            modifier=lambda s, p: [
                setattr(s.joint(jn), "frictionloss", p.value[0]) for jn in JOINT_NAMES
            ],
        )
    )
    params["frictionloss"].value[:] = f_nom
    params["frictionloss"].frozen = True

    # ---- Per-body balance weight bounds ----
    # Each body: [mass_min, mass_max, pos_min_x, pos_max_x, pos_min_y, pos_max_y,
    #             pos_min_z, pos_max_z, size_min_x, size_max_x, size_min_y, size_max_y,
    #             size_min_z, size_max_z, init_size_x, init_size_y, init_size_z]
    # mass bounds are relative to the true mass gap
    # pos bounds are in the body's local frame (meters)
    # size bounds define the box dimensions (meters)
    BALANCE_CFG = {
        "LINK_SHOULDER_PITCH_L": {
            "mass_scale": (0.0, 0.2),
            "mass_init": 0.1,
            "pos_x_range": (-0.027 * 1.5, -0.027 * 0.5),
            "pos_y_range": (0.13 * 0.5, 0.13 * 1.5),
            "pos_z_range": (0.22 * 0.5, 0.22 * 1.5),
            "pos_init": (-0.027, 0.13, 0.22),
            "size_x_range": (0.0, 0.027 * 2),
            "size_y_range": (0.0, 0.13 * 2),
            "size_z_range": (0.0, 0.22 * 2),
            "size_init": (0.0027, 0.13, 0.22),
        },
        "LINK_SHOULDER_ROLL_L": {
            "mass_scale": (0.0, 0.1),
            "mass_init": 0.05,
            "pos_x_range": (-0.037 * 1.5, -0.037 * 0.5),
            "pos_y_range": (0.067 * 0.5, 0.067 * 1.5),
            "pos_z_range": (-0.02 * 1.5, -0.02 * 0.5),
            "pos_init": (-0.037, 0.067, -0.02),
            "size_x_range": (0.0, 0.037 * 2),
            "size_y_range": (0.0, 0.067 * 2),
            "size_z_range": (0.0, 0.02 * 2),
            "size_init": (0.037, 0.067, 0.02),
        },
        "LINK_SHOULDER_YAW_L": {
            "mass_scale": (0.0, 0.2),
            "mass_init": 0.1,
            "pos_x_range": (0.037 * 0.5, 0.037 * 1.5),
            "pos_y_range": (0.0176 * 0.5, 0.0176 * 1.5),
            "pos_z_range": (-0.12 * 1.5, -0.12 * 0.5),
            "pos_init": (0.037, 0.0176, -0.12),
            "size_x_range": (0.0, 0.037 * 2),
            "size_y_range": (0.0, 0.0176 * 2),
            "size_z_range": (0.0, 0.12 * 2),
            "size_init": (0.037, 0.0176, 0.12),
        },
        "LINK_ELBOW_PITCH_L": {
            "mass_scale": (0.0, 0.28),
            "mass_init": 0.14,
            "pos_x_range": (0.0025 * 0.5, 0.0025 * 1.5),
            "pos_y_range": (0.01 * 0.5, 0.01 * 1.5),
            "pos_z_range": (-0.16 * 1.5, -0.16 * 0.5),
            "pos_init": (0.0025, 0.01, -0.16),
            "size_x_range": (0.0, 0.0025 * 2),
            "size_y_range": (0.0, 0.01 * 2),
            "size_z_range": (0.0, 0.16 * 2),
            "size_init": (0.0025, 0.01, 0.16),
        },
        "LINK_ELBOW_YAW_L": {
            "mass_scale": (0.0, 0.1),
            "mass_init": 0.05,
            "pos_x_range": (0.034 * 0.5, 0.034 * 1.5),
            "pos_y_range": (0.1 * 0.5, 0.1 * 1.5),
            "pos_z_range": (-0.2447 * 1.5, -0.2447 * 0.5),
            "pos_init": (0.034, 0.1, -0.2447),
            "size_x_range": (0.0, 0.034 * 2),
            "size_y_range": (0.0, 0.1 * 2),
            "size_z_range": (0.0, 0.2447 * 2),
            "size_init": (0.034, 0.1, 0.2447),
        },
    }

    def _build_balance_params(cfg_dict):
        """Build/add balance parameters from a config dict."""
        for bn in BODY_NAMES:
            cfg = cfg_dict[bn]
            ms_lo, ms_hi = cfg["mass_scale"]
            px_lo, px_hi = cfg["pos_x_range"]
            py_lo, py_hi = cfg["pos_y_range"]
            pz_lo, pz_hi = cfg["pos_z_range"]
            sx_lo, sx_hi = cfg["size_x_range"]
            sy_lo, sy_hi = cfg["size_y_range"]
            sz_lo, sz_hi = cfg["size_z_range"]
            isx, isy, isz = cfg["size_init"]
            pxi, pyi, pzi = cfg["pos_init"]
            if f"balance_{bn}" in params:
                p = params[f"balance_{bn}"]
                p.min_value[:] = [ms_lo, px_lo, py_lo, pz_lo, sx_lo, sy_lo, sz_lo]
                p.max_value[:] = [ms_hi, px_hi, py_hi, pz_hi, sx_hi, sy_hi, sz_hi]
                p.value[:] = [cfg["mass_init"], pxi, pyi, pzi, isx, isy, isz]
            else:
                p = sysid.Parameter(
                    f"balance_{bn}",
                    nominal=np.zeros(7),
                    min_value=np.array(
                        [ms_lo, px_lo, py_lo, pz_lo, sx_lo, sy_lo, sz_lo]
                    ),
                    max_value=np.array(
                        [ms_hi, px_hi, py_hi, pz_hi, sx_hi, sy_hi, sz_hi]
                    ),
                )
                p.value[:] = [cfg["mass_init"], pxi, pyi, pzi, isx, isy, isz]
                p.frozen = True
                params.add(p)

    _build_balance_params(BALANCE_CFG)

    x_scale_7 = np.array([0.1, 0.06, 0.06, 0.06, 0.12, 0.12, 0.12])

    rmse_history = {"Stage": [], "Joint": [], "RMSE": []}

    def record_rmse(label, pd):
        rmse = compute_rmse(init_base, pd, *trajs["balance"])
        for j in range(5):
            rmse_history["Stage"].append(label)
            rmse_history["Joint"].append(f"J{13 + j}")
            rmse_history["RMSE"].append(rmse[j])
        return rmse

    # ---- Baseline: nominal model RMSE before any identification ----
    rmse_nom = record_rmse("0.Nominal", params)
    print(
        f"\n  [Baseline] Nominal model RMSE: {[round(float(x), 3) for x in rmse_nom]}"
    )

    def run_round(round_num):
        print(f"\n{'#' * 60}\n  ROUND {round_num}\n{'#' * 60}")

        # --- Balance ---
        print(f"\n  [R{round_num}] Stage: Balance Weights")
        qb, dqb, ddqb, taub = trajs["balance"]
        for bn in reversed(BODY_NAMES):
            params[
                f"balance_{bn}"
            ].frozen = False  # unfreeze current + keep all downstream unfrozen
            n_bal = sum(1 for b in BODY_NAMES if not params[f"balance_{b}"].frozen)
            rf = make_residual_fn(init_base, qb, dqb, ddqb, taub)
            print(f"\n  --- {bn} ({n_bal} body(s) free, {7 * n_bal} active params) ---")
            opt_p, _ = sysid.optimize(
                params,
                rf,
                optimizer="mujoco",
                verbose=True,
                eps=0.005,
                x_scale=np.tile(x_scale_7, n_bal),
                max_iters=50,
            )
            for b in BODY_NAMES:
                params[f"balance_{b}"].value[:] = opt_p[f"balance_{b}"].value
            # DON'T freeze — keep all downstream bodies free for next iteration
            v = params[f"balance_{bn}"].value
            tm = float(mujoco.MjSpec.from_string(LEFT_ARM_XML).body(bn).mass)
            nm = float(init_base.body(bn).mass)
            print(f"  -> mass={v[0]:+.4f} (true={tm:.4f}, nom={nm:.4f})")
        # Freeze all after final iteration
        for bn in BODY_NAMES:
            params[f"balance_{bn}"].frozen = True
        rmse = record_rmse(f"R{round_num}.Balance", params)
        print(f"  RMSE: {[round(float(x), 4) for x in rmse]}")

        # --- Armature ---
        print(f"\n  [R{round_num}] Stage: Armature")
        qa, dqa, ddqa, taua = trajs["armature"]
        rf = make_residual_fn(init_base, qa, dqa, ddqa, taua)
        params["armature"].frozen = False
        opt_p, _ = sysid.optimize(
            params, rf, optimizer="mujoco", verbose=True, eps=0.005, max_iters=20
        )
        params["armature"].value[:] = opt_p["armature"].value
        params["armature"].frozen = True
        a_val = params["armature"].value[0]
        print(f"  -> armature = {a_val:.6f} (true = {a_true:.6f})")
        rmse = record_rmse(f"R{round_num}.Armature", params)
        print(f"  RMSE: {[round(float(x), 4) for x in rmse]}")

        # --- Damping + Frictionloss ---
        print(f"\n  [R{round_num}] Stage: Damping + Frictionloss")
        qf, dqf, ddqf, tauf = trajs["friction"]
        rf = make_residual_fn(init_base, qf, dqf, ddqf, tauf)
        params["damping"].frozen = False
        params["frictionloss"].frozen = False
        opt_p, _ = sysid.optimize(
            params, rf, optimizer="mujoco", verbose=True, eps=0.005, max_iters=50
        )
        params["damping"].value[:] = opt_p["damping"].value
        params["frictionloss"].value[:] = opt_p["frictionloss"].value
        params["damping"].frozen = True
        params["frictionloss"].frozen = True
        d_val = params["damping"].value[0]
        f_val = params["frictionloss"].value[0]
        print(f"  -> damping = {d_val:.6f} (true = {d_true:.6f})")
        print(f"  -> frictionloss = {f_val:.6f} (true = {f_true:.6f})")
        rmse = record_rmse(f"R{round_num}.Friction", params)
        print(f"  RMSE: {[round(float(x), 4) for x in rmse]}")

        return (
            params["armature"].value[0],
            params["damping"].value[0],
            params["frictionloss"].value[0],
        )

    # ---- Round 1 ----
    a1, d1, f1 = run_round(1)

    # ---- Round 2 ----
    print(f"\n{'#' * 60}\n  ROUND 2 \n{'#' * 60}")
    # Build config from R1 results: init=R1 value, range=±30% of R1 value
    r2_cfg = {}
    for bn in BODY_NAMES:
        v = params[f"balance_{bn}"].value
        m1, px1, py1, pz1, sx1, sy1, sz1 = v

        def _tight(v_lo, v_hi, ctr):
            return (min(v_lo, ctr * 0.7), max(v_hi, ctr * 1.3))

        r2_cfg[bn] = {
            "mass_scale": _tight(
                BALANCE_CFG[bn]["mass_scale"][0], BALANCE_CFG[bn]["mass_scale"][1], m1
            ),
            "mass_init": float(m1),
            "pos_x_range": _tight(
                BALANCE_CFG[bn]["pos_x_range"][0],
                BALANCE_CFG[bn]["pos_x_range"][1],
                px1,
            ),
            "pos_y_range": _tight(
                BALANCE_CFG[bn]["pos_y_range"][0],
                BALANCE_CFG[bn]["pos_y_range"][1],
                py1,
            ),
            "pos_z_range": _tight(
                BALANCE_CFG[bn]["pos_z_range"][0],
                BALANCE_CFG[bn]["pos_z_range"][1],
                pz1,
            ),
            "pos_init": (float(px1), float(py1), float(pz1)),
            "size_x_range": _tight(
                BALANCE_CFG[bn]["size_x_range"][0],
                BALANCE_CFG[bn]["size_x_range"][1],
                sx1,
            ),
            "size_y_range": _tight(
                BALANCE_CFG[bn]["size_y_range"][0],
                BALANCE_CFG[bn]["size_y_range"][1],
                sy1,
            ),
            "size_z_range": _tight(
                BALANCE_CFG[bn]["size_z_range"][0],
                BALANCE_CFG[bn]["size_z_range"][1],
                sz1,
            ),
            "size_init": (float(sx1), float(sy1), float(sz1)),
        }
    _build_balance_params(r2_cfg)

    print("\n  [R2] Stage: Balance Weights (R1 prior, tight ranges)")
    qb, dqb, ddqb, taub = trajs["balance"]
    for bn in reversed(BODY_NAMES):
        params[f"balance_{bn}"].frozen = False
        n_bal = sum(1 for b in BODY_NAMES if not params[f"balance_{b}"].frozen)
        rf = make_residual_fn(init_base, qb, dqb, ddqb, taub)
        print(f"\n  --- {bn} ({n_bal} body(s) free, {7 * n_bal} active params) ---")
        opt_p, _ = sysid.optimize(
            params,
            rf,
            optimizer="mujoco",
            verbose=True,
            eps=0.005,
            x_scale=np.tile(x_scale_7, n_bal),
            max_iters=50,
        )
        for b in BODY_NAMES:
            params[f"balance_{b}"].value[:] = opt_p[f"balance_{b}"].value
        v = params[f"balance_{bn}"].value
        tm = float(mujoco.MjSpec.from_string(LEFT_ARM_XML).body(bn).mass)
        nm = float(init_base.body(bn).mass)
        print(f"  -> mass={v[0]:+.4f} (true={tm:.4f}, nom={nm:.4f})")
    for bn in BODY_NAMES:
        params[f"balance_{bn}"].frozen = True
    rmse_r2 = record_rmse("R2.Balance", params)
    print(f"  RMSE: {[round(float(x), 4) for x in rmse_r2]}")

    # R2 Armature: tighten bounds around R1 result
    params["armature"].min_value[:] = max(0.02, a1 * 0.7)
    params["armature"].max_value[:] = min(0.06, a1 * 1.3)
    params["armature"].value[:] = a1
    print("\n  [R2] Stage: Armature (R1 prior)")
    qa, dqa, ddqa, taua = trajs["armature"]
    rf = make_residual_fn(init_base, qa, dqa, ddqa, taua)
    params["armature"].frozen = False
    opt_p, _ = sysid.optimize(
        params, rf, optimizer="mujoco", verbose=True, eps=0.005, max_iters=20
    )
    params["armature"].value[:] = opt_p["armature"].value
    params["armature"].frozen = True
    a2 = params["armature"].value[0]
    print(f"  -> armature = {a2:.6f} (true = {a_true:.6f})")
    rmse_r2 = record_rmse("R2.Armature", params)
    print(f"  RMSE: {[round(float(x), 4) for x in rmse_r2]}")

    # R2 Frictionloss: tighten bounds around R1 result
    params["damping"].min_value[:] = max(0.04, d1 * 0.7)
    params["damping"].max_value[:] = min(0.12, d1 * 1.3)
    params["damping"].value[:] = d1
    params["frictionloss"].min_value[:] = max(0.2, f1 * 0.7)
    params["frictionloss"].max_value[:] = min(0.4, f1 * 1.3)
    params["frictionloss"].value[:] = f1
    print("\n  [R2] Stage: Damping + Frictionloss (R1 prior)")
    qf, dqf, ddqf, tauf = trajs["friction"]
    rf = make_residual_fn(init_base, qf, dqf, ddqf, tauf)
    params["damping"].frozen = False
    params["frictionloss"].frozen = False
    opt_p, _ = sysid.optimize(
        params, rf, optimizer="mujoco", verbose=True, eps=0.005, max_iters=50
    )
    params["damping"].value[:] = opt_p["damping"].value
    params["frictionloss"].value[:] = opt_p["frictionloss"].value
    params["damping"].frozen = True
    params["frictionloss"].frozen = True
    d2 = params["damping"].value[0]
    f2 = params["frictionloss"].value[0]
    print(f"  -> damping = {d2:.6f} (true = {d_true:.6f})")
    print(f"  -> frictionloss = {f2:.6f} (true = {f_true:.6f})")
    rmse_r2 = record_rmse("R2.Friction", params)
    print(f"  RMSE: {[round(float(x), 4) for x in rmse_r2]}")

    # ---- Plot RMSE progression ----
    stage_rmse = {}
    for st, j, r in zip(
        rmse_history["Stage"], rmse_history["Joint"], rmse_history["RMSE"]
    ):
        stage_rmse.setdefault(st, {})[j] = r
    all_stages = list(dict.fromkeys(rmse_history["Stage"]))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(5)
    n = len(all_stages)
    w = 0.8 / n
    # First bar (Nominal) in gray, rest in viridis
    colors = [(0.5, 0.5, 0.5)] + [plt.cm.viridis(i / (n - 2)) for i in range(n - 1)]
    for i, st in enumerate(all_stages):
        vals = [stage_rmse[st].get(f"J{13 + j}", 0) for j in range(5)]
        bars = ax.bar(x + i * w, vals, w, color=colors[i], alpha=0.85, label=st)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(x + (n - 1) * w / 2)
    ax.set_xticklabels([f"J{13 + j}" for j in range(5)])
    ax.set_ylabel("Torque RMSE (Nm)")
    ax.set_title(
        f"RMSE Progression: {'20pct' if USED_XML == LEFT_ARM_XML_20PCT else 'Nominal'}"
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    # ---- Final summary ----
    print("\n" + "=" * 66)
    print("  Final Results  (R1 → R2)")
    print("=" * 66)
    print(f"  {'Param':<16s} {'R1':>10s} {'R2':>10s} {'True':>10s} {'R2 Err%':>10s}")
    print(f"  {'-' * 56}")
    print(
        f"  {'armature':<16s} {a1:>10.6f} {a2:>10.6f} {a_true:>10.6f} {abs(a2 - a_true) / a_true * 100:>9.1f}%"
    )
    print(
        f"  {'damping':<16s} {d1:>10.6f} {d2:>10.6f} {d_true:>10.6f} {abs(d2 - d_true) / d_true * 100:>9.1f}%"
    )
    print(
        f"  {'frictionloss':<16s} {f1:>10.6f} {f2:>10.6f} {f_true:>10.6f} {abs(f2 - f_true) / f_true * 100:>9.1f}%"
    )
    print(
        f"\n{'Body':<28s} {'Balance':>8s} {'+Nominal':>10s} {'=Total':>10s} {'True':>10s} {'Err%':>8s}"
    )
    print("-" * 76)
    for bn in BODY_NAMES:
        v = params[f"balance_{bn}"].value
        nm = float(init_base.body(bn).mass)
        tm = float(mujoco.MjSpec.from_string(LEFT_ARM_XML).body(bn).mass)
        total = nm + v[0]
        err = abs(total - tm) / tm * 100
        print(
            f"{bn:<28s} {v[0]:>+7.4f}  {nm:>8.4f}  {total:>8.4f}  {tm:>8.4f}  {err:>6.1f}%"
        )
    # ---- Test trajectory: compare True vs Nominal vs Optimized ----
    print("\nRunning test trajectory comparison …")
    import glob as _glob

    _test_files = sorted(_glob.glob(str(_YAML_DIR / "test_trajectory_*.yaml")))
    if _test_files:
        _test_yaml = _test_files[-2]
        print(f"  Test trajectory: {Path(_test_yaml).name}")
        qt, dqt, ddqt = load_trajectory(_test_yaml)[:3]  # only q,dq,ddq
        # Recompute tau_true from true model for the test trajectory
        _tspec = mujoco.MjSpec.from_string(LEFT_ARM_XML)
        _tmodel = _tspec.compile()
        _tdata = mujoco.MjData(_tmodel)
        tau_true_test = np.zeros_like(qt)
        for k in range(len(qt)):
            _tdata.qpos[:] = qt[k]
            _tdata.qvel[:] = dqt[k]
            _tdata.qacc[:] = ddqt[k]
            mujoco.mj_inverse(_tmodel, _tdata)
            tau_true_test[k] = _tdata.qfrc_inverse.copy()

        # Nominal (pre-ID) torques
        _nspec = init_base.copy()
        _nmodel = _nspec.compile()
        _ndata = mujoco.MjData(_nmodel)
        tau_nom = np.zeros_like(qt)
        for k in range(len(qt)):
            _ndata.qpos[:] = qt[k]
            _ndata.qvel[:] = dqt[k]
            _ndata.qacc[:] = ddqt[k]
            mujoco.mj_inverse(_nmodel, _ndata)
            tau_nom[k] = _ndata.qfrc_inverse.copy()

        # Optimized (post-ID) torques
        # tau_opt = compute_rmse(init_base, params, qt, dqt, ddqt, tau_true_test)
        # Actually compute_rmse returns RMSE vector, we need the full torque
        _ospec = init_base.copy()
        sysid.apply_param_modifiers_spec(params, _ospec)
        _bv = {bn: list(params[f"balance_{bn}"].value) for bn in BODY_NAMES}
        apply_balances_to_spec(_ospec, _bv)
        _omodel = _ospec.compile()
        _odata = mujoco.MjData(_omodel)
        tau_opt_full = np.zeros_like(qt)
        for k in range(len(qt)):
            _odata.qpos[:] = qt[k]
            _odata.qvel[:] = dqt[k]
            _odata.qacc[:] = ddqt[k]
            mujoco.mj_inverse(_omodel, _odata)
            tau_opt_full[k] = _odata.qfrc_inverse.copy()

        # Plot 5×1 torque comparison
        _fig2, _axes2 = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        _t_test = np.arange(len(qt)) * 0.002
        _colors2 = {"True": "green", "Nominal": "red", "Optimized": "blue"}
        _styles2 = {"True": "-", "Nominal": "--", "Optimized": "-."}
        for j in range(5):
            _ax = _axes2[j]
            for _label, _tau in [
                ("True", tau_true_test),
                ("Nominal", tau_nom),
                ("Optimized", tau_opt_full),
            ]:
                _ax.plot(
                    _t_test,
                    _tau[:, j],
                    color=_colors2[_label],
                    ls=_styles2[_label],
                    lw=1.2,
                    label=_label,
                    alpha=0.85,
                )
            _ax.set_ylabel(f"J{13 + j} torque (Nm)")
            _ax.legend(fontsize=7)
            _ax.grid(True, alpha=0.3)
        _axes2[-1].set_xlabel("Time (s)")
        _fig2.suptitle("Test Trajectory: Inverse Dynamics Torque Comparison", y=1.01)
        plt.tight_layout()

        # Test trajectory RMSE bar chart: Nominal vs Optimized
        _rmse_nom_test = np.sqrt(np.mean((tau_nom - tau_true_test) ** 2, axis=0))
        _rmse_opt_test = np.sqrt(np.mean((tau_opt_full - tau_true_test) ** 2, axis=0))
        _fig3, _ax3 = plt.subplots(figsize=(8, 4))
        _x3 = np.arange(5)
        _w3 = 0.35
        _b1 = _ax3.bar(
            _x3 - _w3 / 2, _rmse_nom_test, _w3, color="red", alpha=0.7, label="Nominal"
        )
        _b2 = _ax3.bar(
            _x3 + _w3 / 2,
            _rmse_opt_test,
            _w3,
            color="blue",
            alpha=0.7,
            label="Optimized",
        )
        for _b, _v in [(_b1, _rmse_nom_test), (_b2, _rmse_opt_test)]:
            for _bar, _val in zip(_b, _v):
                _ax3.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _bar.get_height() + 0.01,
                    f"{_val:.3f}",
                    ha="center",
                    fontsize=9,
                )
        _ax3.set_xticks(_x3)
        _ax3.set_xticklabels([f"J{13 + j}" for j in range(5)])
        _ax3.set_ylabel("Torque RMSE (Nm)")
        _ax3.set_title("Test Trajectory RMSE: Nominal vs Optimized")
        _ax3.legend()
        _ax3.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

    print("\nDone. Showing plots …")
    plt.show()


if __name__ == "__main__":
    main()
