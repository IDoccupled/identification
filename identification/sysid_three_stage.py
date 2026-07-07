#!/usr/bin/env python3
"""
Two-round iterative identification for PM-V2 Left Arm.

  Round 1: Balance → Armature → Damping+Frictionloss
  Round 2: Same, but subtracts round-1 joint estimates from torque residual

Each stage uses a PSO-optimized Fourier trajectory tailored to its parameters.
All 5 joints share the same motor → armature/damping/frictionloss are TIED.
"""

import os
import pathlib
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

TRUE_INERTIA = {
    "LINK_SHOULDER_PITCH_L": {
        "mass": 0.932155,
        "ipos": [-0.0068739, 0.0562468, -0.0156415],
    },
    "LINK_SHOULDER_ROLL_L": {
        "mass": 0.510813,
        "ipos": [0.037412, 0.00677175, -0.0267801],
    },
    "LINK_SHOULDER_YAW_L": {
        "mass": 0.909138,
        "ipos": [-0.00084545, 0.00219391, -0.045665],
    },
    "LINK_ELBOW_PITCH_L": {
        "mass": 1.38061,
        "ipos": [0.00298237, 0.00186711, -0.0651846],
    },
    "LINK_ELBOW_YAW_L": {
        "mass": 0.467519,
        "ipos": [0.0201851, 0.000217483, -0.0900144],
    },
}

# Nominal (perturbed) inertia values — must match NOMINAL_LEFT_ARM_XML exactly.
# mass × 0.75, ipos × 1.25 (adjust MASS_PERTURB / IPOS_PERTURB below to
# re-generate the nominal XML when needed).
NOMINAL_INERTIA = {
    "LINK_SHOULDER_PITCH_L": {
        "mass": 0.699116,
        "ipos": [-0.0085924, 0.0703085, -0.0195519],
    },
    "LINK_SHOULDER_ROLL_L": {
        "mass": 0.383110,
        "ipos": [0.0467650, 0.0084647, -0.0334751],
    },
    "LINK_SHOULDER_YAW_L": {
        "mass": 0.681854,
        "ipos": [-0.0010568, 0.0027424, -0.0570813],
    },
    "LINK_ELBOW_PITCH_L": {
        "mass": 1.035457,
        "ipos": [0.0037280, 0.0023339, -0.0814807],
    },
    "LINK_ELBOW_YAW_L": {
        "mass": 0.350639,
        "ipos": [0.0252314, 0.0002719, -0.1125180],
    },
}

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
    spec = mujoco.MjSpec.from_string(NOMINAL_LEFT_ARM_XML)
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


def make_valid_inertia(Ixx, Iyy, Izz):
    """Ensure positive-definite inertia satisfying triangle inequality."""
    arr = np.array([Ixx, Iyy, Izz], dtype=float)
    # MuJoCo requires positive diagonal elements and A+B >= C for all permutations.
    arr = np.maximum(arr, 1e-10)
    for _ in range(10):
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
    return arr


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

        # Combine, make physically valid, and ensure strictly positive
        combined = old_I_shifted + box_I_shifted
        valid_I = make_valid_inertia(*combined)
        valid_I = np.maximum(valid_I, 1e-10)

        b.mass = new_mass
        b.ipos = new_ipos
        b.inertia = valid_I


def load_trajectory(yaml_name):
    """Load (q, dq, ddq, tau_true) from a Fourier YAML, using the TRUE model."""
    ft = FourierTrajectory(dim=5, sample_rate=500)
    ft.omega_f = 2.0 * np.pi / 5.0
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


def make_residual_fn(
    nominal_base,
    params,
    q,
    dq,
    ddq,
    tau_true,
    subtract_armature=0.0,
    subtract_damping=0.0,
    subtract_fric=0.0,
):
    """Build residual function that optionally subtracts known joint contributions."""
    tau_target = tau_true.copy()
    if subtract_armature != 0.0:
        tau_target = tau_target - subtract_armature * ddq
    if subtract_damping != 0.0:
        tau_target = tau_target - subtract_damping * dq
    if subtract_fric != 0.0:
        tau_target = tau_target - subtract_fric * np.sign(dq)

    def fn(x, pd):
        if x.ndim == 1:
            pd.update_from_vector(x)
            spec = nominal_base.copy()
            sysid.apply_param_modifiers_spec(pd, spec)
            bv = {bn: list(pd[f"balance_{bn}"].value) for bn in BODY_NAMES}
            apply_balances_to_spec(spec, bv)
            model = spec.compile()
            data = mujoco.MjData(model)
            res = []
            for k in range(len(q)):
                data.qpos[:] = q[k]
                data.qvel[:] = dq[k]
                data.qacc[:] = ddq[k]
                mujoco.mj_inverse(model, data)
                res.append(data.qfrc_inverse - tau_target[k])
            return [np.array(res).ravel()], None, None
        else:
            res_list = []
            for ki in range(x.shape[1]):
                pd.update_from_vector(x[:, ki])
                spec = nominal_base.copy()
                sysid.apply_param_modifiers_spec(pd, spec)
                bv = {bn: list(pd[f"balance_{bn}"].value) for bn in BODY_NAMES}
                apply_balances_to_spec(spec, bv)
                model = spec.compile()
                data = mujoco.MjData(model)
                rk = []
                for k in range(len(q)):
                    data.qpos[:] = q[k]
                    data.qvel[:] = dq[k]
                    data.qacc[:] = ddq[k]
                    mujoco.mj_inverse(model, data)
                    rk.append(data.qfrc_inverse - tau_target[k])
                res_list.append(np.array(rk).ravel())
            return [np.column_stack(res_list)], None, None

    return fn


def compute_rmse(nominal_base, params, q, dq, ddq, tau_true):
    spec = nominal_base.copy()
    sysid.apply_param_modifiers_spec(params, spec)
    bv = {bn: list(params[f"balance_{bn}"].value) for bn in BODY_NAMES}
    apply_balances_to_spec(spec, bv)
    model = spec.compile()
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
    nominal_base = mujoco.MjSpec.from_string(NOMINAL_LEFT_ARM_XML)

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
        q, dq, ddq, tau = load_trajectory(yaml_name)
        trajs[stage_key] = (q, dq, ddq, tau)
        print(
            f"  {stage_key}: {len(q)} steps, |ddq|max={np.abs(ddq).max():.1f}, "
            f"|dq|max={np.abs(dq).max():.1f}"
        )

    # ---- Build parameters ----
    params = sysid.ParameterDict()

    def _damp_shared_mod(s, p):
        for jn in JOINT_NAMES:
            s.joint(jn).damping = np.array([[p.value[0]], [0.0], [0.0]])

    params.add(
        sysid.Parameter(
            "armature",
            nominal=a_true,
            min_value=0.001,
            max_value=0.2,
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
            min_value=0.001,
            max_value=0.5,
            modifier=_damp_shared_mod,
        )
    )
    params["damping"].value[:] = d_nom
    params["damping"].frozen = True

    params.add(
        sysid.Parameter(
            "frictionloss",
            nominal=f_true,
            min_value=0.001,
            max_value=1.0,
            modifier=lambda s, p: [
                setattr(s.joint(jn), "frictionloss", p.value[0]) for jn in JOINT_NAMES
            ],
        )
    )
    params["frictionloss"].value[:] = f_nom
    params["frictionloss"].frozen = True

    for bn in BODY_NAMES:
        true_m = float(mujoco.MjSpec.from_string(LEFT_ARM_XML).body(bn).mass)
        nom_m = float(nominal_base.body(bn).mass)
        gap = true_m - nom_m
        p = sysid.Parameter(
            f"balance_{bn}",
            nominal=np.zeros(7),
            min_value=np.array([-0.15, -0.1, -0.1, -0.1, 0.005, 0.005, 0.005]),
            max_value=np.array([0.5, 0.1, 0.1, 0.1, 0.25, 0.25, 0.25]),
        )
        p.value[:] = [gap * 1.05, 0, 0, 0, 0.06, 0.06, 0.06]
        p.frozen = True
        params.add(p)

    x_scale_7 = np.array([0.1, 0.06, 0.06, 0.06, 0.12, 0.12, 0.12])

    rmse_history = {"Stage": [], "Joint": [], "RMSE": []}

    def record_rmse(label, pd):
        rmse = compute_rmse(nominal_base, pd, *trajs["balance"])
        for j in range(5):
            rmse_history["Stage"].append(label)
            rmse_history["Joint"].append(f"J{13 + j}")
            rmse_history["RMSE"].append(rmse[j])
        return rmse

    # ---- Helper: run one round of Balance → Armature → Friction ----
    def run_round(round_num, sub_a=0.0, sub_d=0.0, sub_f=0.0):
        print(f"\n{'#' * 60}\n  ROUND {round_num}\n{'#' * 60}")

        # --- Balance ---
        print(f"\n  [R{round_num}] Stage: Balance Weights")
        qb, dqb, ddqb, taub = trajs["balance"]
        rf = make_residual_fn(
            nominal_base, params, qb, dqb, ddqb, taub, sub_a, sub_d, sub_f
        )
        for bn in reversed(BODY_NAMES):
            params[f"balance_{bn}"].frozen = False
            n_bal = sum(1 for b in BODY_NAMES if not params[f"balance_{b}"].frozen)
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
            params[f"balance_{bn}"].frozen = True
            v = params[f"balance_{bn}"].value
            tm = float(mujoco.MjSpec.from_string(LEFT_ARM_XML).body(bn).mass)
            nm = float(nominal_base.body(bn).mass)
            print(f"  -> mass={v[0]:+.4f} (true={tm:.4f}, nom={nm:.4f})")
        rmse = record_rmse(f"R{round_num}.Balance", params)
        print(f"  RMSE: {[round(float(x), 4) for x in rmse]}")

        # --- Armature ---
        print(f"\n  [R{round_num}] Stage: Armature")
        qa, dqa, ddqa, taua = trajs["armature"]
        rf = make_residual_fn(
            nominal_base, params, qa, dqa, ddqa, taua, sub_a, sub_d, sub_f
        )
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
        rf = make_residual_fn(
            nominal_base, params, qf, dqf, ddqf, tauf, sub_a, sub_d, sub_f
        )
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

    # ---- Round 2: subtract round-1 joint estimates ----
    a2, d2, f2 = run_round(2, sub_a=a1, sub_d=d1, sub_f=f1)

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
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for i, st in enumerate(all_stages):
        vals = [stage_rmse[st].get(f"J{13 + j}", 0) for j in range(5)]
        ax.bar(x + i * w, vals, w, color=colors[i], alpha=0.85, label=st)
    ax.set_xticks(x + (n - 1) * w / 2)
    ax.set_xticklabels([f"J{13 + j}" for j in range(5)])
    ax.set_ylabel("Torque RMSE (Nm)")
    ax.set_title("2-Round Iterative ID: RMSE Progression")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("three_stage_rmse.png", dpi=150)
    plt.close()
    print("\n  -> saved three_stage_rmse.png")

    # ---- Final summary ----
    print("\n" + "=" * 66)
    print("  Final Results")
    print("=" * 66)
    print(
        f"  armature     = {a2:.6f}  (true = {a_true:.6f}, err = {abs(a2 - a_true) / a_true * 100:.1f}%)"
    )
    print(
        f"  damping      = {d2:.6f}  (true = {d_true:.6f}, err = {abs(d2 - d_true) / d_true * 100:.1f}%)"
    )
    print(
        f"  frictionloss = {f2:.6f}  (true = {f_true:.6f}, err = {abs(f2 - f_true) / f_true * 100:.1f}%)"
    )
    print(
        f"\n{'Body':<28s} {'Balance':>8s} {'+Nominal':>10s} {'=Total':>10s} {'True':>10s} {'Err%':>8s}"
    )
    print("-" * 76)
    for bn in BODY_NAMES:
        v = params[f"balance_{bn}"].value
        nm = float(nominal_base.body(bn).mass)
        tm = float(mujoco.MjSpec.from_string(LEFT_ARM_XML).body(bn).mass)
        total = nm + v[0]
        err = abs(total - tm) / tm * 100
        print(
            f"{bn:<28s} {v[0]:>+7.4f}  {nm:>8.4f}  {total:>8.4f}  {tm:>8.4f}  {err:>6.1f}%"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
