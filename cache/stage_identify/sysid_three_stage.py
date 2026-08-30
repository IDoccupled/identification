#!/usr/bin/env python3
"""
Two-round iterative identification for PM-V2 Left Arm.

  Round 1: Balance → Frictionloss → Armature+Damping (wide bounds, URDF prior)
  Round 2: Balance → Frictionloss → Armature+Damping (R1 prior, tight bounds)

Each stage uses a PSO-optimized Fourier trajectory tailored to its parameters.
All 5 joints share the same motor → armature/damping/frictionloss are TIED.

Balance weights: each BODY has a child body `bal_{NAME}` with a box geom and
mass.  Parameter modifiers set body.mass, body.pos, body.geoms[0].size directly.
MuJoCo handles the full-6DoF rigid-body inertia computation natively.
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


# ---------------------------------------------------------------------------
# MuJoCo-native balance weight: add a child body + box geom to each link body
# ---------------------------------------------------------------------------
def _add_balance_child_body(spec, body_name: str):
    """Add a zero-mass child body `bal_{body_name}` with a tiny box geom.
    The child body's quaternion is fixed to the parent's inertial quaternion
    so the box aligns with the inertia principal axes. Only mass, pos, and
    geom size are fitted by the Parameter modifier.
    """
    parent = spec.body(body_name)
    child_name = f"bal_{body_name}"
    child = parent.add_body(name=child_name, pos=[0.0, 0.0, 0.0])
    # Fix orientation to parent's inertial quaternion
    child.quat = parent.iquat.copy()
    child.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.001, 0.001, 0.001],
    )
    child.mass = 0.0  # zero by default; modifier sets actual mass
    return child_name


def _make_balance_modifier(child_name: str):
    """Return a modifier that sets mass, pos, and box size on a balance child body.
    Parameter layout: [mass, px, py, pz, sx, sy, sz].
    """

    def mod(s, p):
        v = p.value
        b = s.body(child_name)
        b.mass = max(v[0], 0.0)
        b.pos = v[1:4]
        b.geoms[0].size = np.maximum(np.abs(v[4:7]), 1e-9)

    return mod


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
    """Build residual: τ_pred(params) - τ_true.
    All modifiers (joint params + balance bodies) are applied via
    sysid.apply_param_modifiers_spec — no manual post-processing needed.
    """

    def _eval_one(xi, pd):
        """Compute residual for a single 1D parameter vector, return as (m, 1)."""
        pd.update_from_vector(xi)
        spec = init_base.copy()
        sysid.apply_param_modifiers_spec(pd, spec)
        spec.compiler.balanceinertia = True
        model = spec.compile()
        data = mujoco.MjData(model)
        y = np.zeros(len(q) * 5)
        for k in range(len(q)):
            data.qpos[:] = q[k]
            data.qvel[:] = dq[k]
            data.qacc[:] = ddq[k]
            mujoco.mj_inverse(model, data)
            y[k * 5 : (k + 1) * 5] = data.qfrc_inverse.copy()
        return (y.ravel() - tau_true.ravel()).reshape(-1, 1)

    def fn(x, pd):
        """Residual compatible with sysid.optimize.
        Handles single (1D or (n,1)) and batch ((n,n) FD Jacobian) inputs.
        Single: returns [(m, 1)] so r is column vector expected by jacobian_fd.
        Batch: returns [(m, n)] directly for the Jacobian.
        """
        if x.ndim == 2 and x.shape[1] > 1:
            # Batch: (n_params, n_params) from finite-difference Jacobian
            cols = [_eval_one(x[:, i], pd) for i in range(x.shape[1])]
            # Each col is (m, 1); hstack gives (m, n)
            return [np.hstack(cols)], None, None
        else:
            # Single: 1D or (n_params, 1) → column vector (m, 1)
            return [_eval_one(x.ravel(), pd)], None, None

    return fn


def compute_rmse(init_base, q, dq, ddq, tau_true, pd, nominal: bool = False):
    """Compute per-joint RMSE. If nominal=True, uses a fresh parse from XML."""
    if nominal:
        spec = mujoco.MjSpec.from_string(USED_XML)
    else:
        spec = init_base.copy()
        sysid.apply_param_modifiers_spec(pd, spec)
    spec.compiler.balanceinertia = True
    model = spec.compile()
    data = mujoco.MjData(model)
    y = np.zeros_like(tau_true)
    for k in range(len(q)):
        data.qpos[:] = q[k]
        data.qvel[:] = dq[k]
        data.qacc[:] = ddq[k]
        mujoco.mj_inverse(model, data)
        y[k] = data.qfrc_inverse.copy()
    err = y - tau_true
    return np.sqrt(np.mean(err**2, axis=0))


# =========================================================================
def main():
    true_joint = _read_true_joint_params()
    nom_joint = _read_nominal_joint_params()
    init_base = mujoco.MjSpec.from_string(USED_XML)
    init_base.compiler.balanceinertia = True

    a_true = true_joint[JOINT_NAMES[0]]["armature"]
    d_true = true_joint[JOINT_NAMES[0]]["damping"]
    f_true = true_joint[JOINT_NAMES[0]]["frictionloss"]
    a_nom = nom_joint[JOINT_NAMES[0]]["armature"]
    d_nom = nom_joint[JOINT_NAMES[0]]["damping"]
    f_nom = nom_joint[JOINT_NAMES[0]]["frictionloss"]

    # ---- Add balance weight child bodies to init_base ----
    bal_child_names = {}
    for bn in BODY_NAMES:
        child_name = _add_balance_child_body(init_base, bn)
        bal_child_names[bn] = child_name
    print(f"Added balance child bodies: {list(bal_child_names.values())}")

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

    # ---- Per-body balance weight config ----
    # Each body: [mass, px, py, pz, sx, sy, sz]
    # mass bounds are absolute (kg)
    # pos/size bounds are in the body's local frame (meters)
    BALANCE_CFG = {
        "LINK_SHOULDER_PITCH_L": {
            "mass_scale": (0.168 * 0.5, 0.168 * 1.5),
            "mass_init": 0.168,
            "pos_x_range": (-0.034 * 1.5, -0.034 * 0.5),
            "pos_y_range": (0.186 * 0.5, 0.186 * 1.5),
            "pos_z_range": (0.2 * 0.5, 0.2 * 1.5),
            "pos_init": (-0.034, 0.186, 0.2),
            "size_x_range": (0.0, 0.034 * 2),
            "size_y_range": (0.0, 0.186 * 2),
            "size_z_range": (0.0, 0.2 * 2),
            "size_init": (0.034, 0.186, 0.2),
        },
        "LINK_SHOULDER_ROLL_L": {
            "mass_scale": (0.1 * 0.5, 0.1 * 1.5),
            "mass_init": 0.1,
            "pos_x_range": (-0.0003 * 1.5, -0.0003 * 0.5),
            "pos_y_range": (0.0737 * 0.5, 0.0737 * 1.5),
            "pos_z_range": (-0.0476 * 1.5, -0.0476 * 0.5),
            "pos_init": (0.0003, 0.0737, -0.0476),
            "size_x_range": (0.0, 0.037 * 2),
            "size_y_range": (0.0, 0.0737 * 2),
            "size_z_range": (0.0, 0.0476 * 2),
            "size_init": (0.0003, 0.0737, 0.0476),
        },
        "LINK_SHOULDER_YAW_L": {
            "mass_scale": (0.16 * 0.5, 0.16 * 1.5),
            "mass_init": 0.16,
            "pos_x_range": (0.037 * 0.5, 0.037 * 1.5),
            "pos_y_range": (0.01985 * 0.5, 0.01985 * 1.5),
            "pos_z_range": (-0.12 * 1.5, -0.12 * 0.5),
            "pos_init": (0.037, 0.01985, -0.12),
            "size_x_range": (0.0, 0.037 * 2),
            "size_y_range": (0.0, 0.01985 * 2),
            "size_z_range": (0.0, 0.12 * 2),
            "size_init": (0.037, 0.01985, 0.12),
        },
        "LINK_ELBOW_PITCH_L": {
            "mass_scale": (0.29 * 0.5, 0.29 * 1.5),
            "mass_init": 0.29,
            "pos_x_range": (0.003 * 0.5, 0.003 * 1.5),
            "pos_y_range": (0.085 * 0.5, 0.085 * 1.5),
            "pos_z_range": (-0.17 * 1.5, -0.17 * 0.5),
            "pos_init": (0.003, 0.085, -0.17),
            "size_x_range": (0.0, 0.003 * 2),
            "size_y_range": (0.0, 0.085 * 2),
            "size_z_range": (0.0, 0.17 * 2),
            "size_init": (0.003, 0.085, 0.17),
        },
        "LINK_ELBOW_YAW_L": {
            "mass_scale": (0.1 * 0.5, 0.1 * 1.5),
            "mass_init": 0.1,
            "pos_x_range": (0.038 * 0.5, 0.038 * 1.5),
            "pos_y_range": (0.01 * 0.5, 0.01 * 1.5),
            "pos_z_range": (-0.26 * 1.5, -0.26 * 0.5),
            "pos_init": (0.038, 0.01, -0.26),
            "size_x_range": (0.0, 0.038 * 2),
            "size_y_range": (0.0, 0.01 * 2),
            "size_z_range": (0.0, 0.26 * 2),
            "size_init": (0.038, 0.01, 0.26),
        },
    }

    def _build_balance_params(cfg_dict):
        """Build/add balance parameters from a config dict.
        Each parameter controls a child body bal_{NAME}: [mass, px, py, pz, sx, sy, sz].
        Mass bounds are from cfg mass_scale (absolute), pos/size from cfg ranges.
        """
        for bn in BODY_NAMES:
            cfg = cfg_dict[bn]
            ms_lo, ms_hi = cfg["mass_scale"]
            px_lo, px_hi = cfg["pos_x_range"]
            py_lo, py_hi = cfg["pos_y_range"]
            pz_lo, pz_hi = cfg["pos_z_range"]
            sx_lo, sx_hi = cfg["size_x_range"]
            sy_lo, sy_hi = cfg["size_y_range"]
            sz_lo, sz_hi = cfg["size_z_range"]

            nominal = np.array(
                [
                    cfg["mass_init"],
                    cfg["pos_init"][0],
                    cfg["pos_init"][1],
                    cfg["pos_init"][2],
                    cfg["size_init"][0],
                    cfg["size_init"][1],
                    cfg["size_init"][2],
                ]
            )
            lower = np.array([ms_lo, px_lo, py_lo, pz_lo, sx_lo, sy_lo, sz_lo])
            upper = np.array([ms_hi, px_hi, py_hi, pz_hi, sx_hi, sy_hi, sz_hi])

            child_name = bal_child_names[bn]
            name = f"balance_{bn}"
            p = sysid.Parameter(
                name=name,
                nominal=nominal,
                min_value=lower,
                max_value=upper,
                modifier=_make_balance_modifier(child_name),
            )
            p.value[:] = nominal.copy()
            params.add(p)

    _build_balance_params(BALANCE_CFG)
    # All balance params start frozen; we unfreeze per-round
    for bn in BODY_NAMES:
        params[f"balance_{bn}"].frozen = True

    # ---- Build residual functions ----
    res_fns = {}
    for stage_key in STAGE_YAMLS:
        q, dq, ddq, tau = trajs[stage_key]
        res_fns[stage_key] = make_residual_fn(init_base, q, dq, ddq, tau)

    # ---- Round 1: wide bounds ----
    print("\n" + "=" * 60)
    print("ROUND 1: Balance → Frictionloss → Armature+Damping (wide bounds)")
    print("=" * 60)

    r1_rmse_by_stage = {}

    # ---- Stage 1: Balance (cumulatively unfreezing) ----
    print("\n--- Stage 1: Balance (cumulatively unfreezing) ---")

    # Build cumulative unfreeze order
    cumul_groups = []
    for i in range(len(BODY_NAMES)):
        group = [f"balance_{BODY_NAMES[j]}" for j in range(i + 1)]
        cumul_groups.append(group)

    for group_idx, group in enumerate(cumul_groups):
        # Freeze all, then unfreeze this group
        for bn in BODY_NAMES:
            params[f"balance_{bn}"].frozen = True
        for g in group:
            params[g].frozen = False

        qb, dqb, ddqb, taub = trajs["balance"]
        print(f"\n  Cumulative group {group_idx + 1}/{len(cumul_groups)}: {group}")
        params, _ = sysid.optimize(
            params,
            res_fns["balance"],
            max_iters=10,
        )
        # Collect per-body fitted vectors
        for bn_i in range(group_idx + 1):
            bn = BODY_NAMES[bn_i]
            p = params[f"balance_{bn}"]
            print(
                f"    {bn}: mass={p.value[0]:.4f} kg, "
                f"pos=({p.value[1]:.4f}, {p.value[2]:.4f}, {p.value[3]:.4f}), "
                f"size=({p.value[4]:.4f}, {p.value[5]:.4f}, {p.value[6]:.4f})"
            )

    # Compute balance-stage RMSE
    q_bal, dq_bal, ddq_bal, tau_bal = trajs["balance"]
    r1_rmse_by_stage["balance"] = compute_rmse(
        init_base, q_bal, dq_bal, ddq_bal, tau_bal, params
    )
    print("\n  Balance RMSE after R1 balance:")
    for jn, e in zip(JOINT_NAMES, r1_rmse_by_stage["balance"]):
        print(f"    {jn}: {e:.4f}")

    # ---- Stage 2: Armature + Damping ----
    print("\n--- Stage 2: Armature + Damping ---")
    params["armature"].frozen = False
    params["damping"].frozen = False
    q_arm, dq_arm, ddq_arm, tau_arm = trajs["armature"]
    params, _ = sysid.optimize(
        params,
        res_fns["armature"],
        max_iters=10,
    )
    print(f"  armature = {params['armature'].value[0]:.6f}  (true={a_true})")
    print(f"  damping  = {params['damping'].value[0]:.6f}  (true={d_true})")
    r1_rmse_by_stage["armature"] = compute_rmse(
        init_base, q_arm, dq_arm, ddq_arm, tau_arm, params
    )
    print("  Armature+Damping RMSE after R1:")
    for jn, e in zip(JOINT_NAMES, r1_rmse_by_stage["armature"]):
        print(f"    {jn}: {e:.4f}")

    # ---- Stage 3: Frictionloss ----
    print("\n--- Stage 3: Frictionloss ---")
    params["frictionloss"].frozen = False
    q_fr, dq_fr, ddq_fr, tau_fr = trajs["friction"]
    params, _ = sysid.optimize(
        params,
        res_fns["friction"],
        max_iters=10,
    )
    print(f"  frictionloss = {params['frictionloss'].value[0]:.6f}  (true={f_true})")
    r1_rmse_by_stage["friction"] = compute_rmse(
        init_base, q_fr, dq_fr, ddq_fr, tau_fr, params
    )
    print("  Friction RMSE after R1 friction:")
    for jn, e in zip(JOINT_NAMES, r1_rmse_by_stage["friction"]):
        print(f"    {jn}: {e:.4f}")

    # ---- Round 2: tight bounds using R1 results as priors ----
    print("\n" + "=" * 60)
    print("ROUND 2: Balance → Frictionloss → Armature+Damping (tight bounds)")
    print("=" * 60)

    # Tighten balance bounds: ±20% around R1 values
    for bn in BODY_NAMES:
        p = params[f"balance_{bn}"]
        r1_val = p.value.copy()
        delta = np.abs(r1_val) * 0.2 + 1e-6
        lo = r1_val - delta
        hi = r1_val + delta
        # mass (idx 0) and sizes (idx 4-6) must be ≥ 0
        lo[0] = max(lo[0], 0.0)
        lo[4:7] = np.maximum(lo[4:7], 0.0)
        hi[0] = max(hi[0], 1e-6)
        hi[4:7] = np.maximum(hi[4:7], 1e-9)
        # Ensure lo < hi strictly for all components
        for i in range(7):
            if lo[i] >= hi[i]:
                hi[i] = lo[i] + 1e-6
        p.min_value = lo
        p.max_value = hi
        p.nominal = r1_val.copy()
        p.frozen = True

    # Tighten friction bounds
    p_fric = params["frictionloss"]
    r1_fric = p_fric.value[0]
    p_fric.min_value = np.array([r1_fric * 0.8])
    p_fric.max_value = np.array([r1_fric * 1.2])
    p_fric.nominal = np.array([r1_fric])
    p_fric.frozen = True

    # Tighten armature bounds
    p_arm = params["armature"]
    r1_arm = p_arm.value[0]
    p_arm.min_value = np.array([r1_arm * 0.8])
    p_arm.max_value = np.array([r1_arm * 1.2])
    p_arm.nominal = np.array([r1_arm])
    p_arm.frozen = True

    # Tighten damping bounds
    p_dmp = params["damping"]
    r1_dmp = p_dmp.value[0]
    p_dmp.min_value = np.array([r1_dmp * 0.8])
    p_dmp.max_value = np.array([r1_dmp * 1.2])
    p_dmp.nominal = np.array([r1_dmp])
    p_dmp.frozen = True

    # ---- R2 Stage 1: Balance (tight bounds) ----
    print("\n--- R2 Stage 1: Balance ---")

    # Cumulative unfreeze again
    for group_idx, group in enumerate(cumul_groups):
        for bn in BODY_NAMES:
            params[f"balance_{bn}"].frozen = True
        for g in group:
            params[g].frozen = False

        print(f"\n  R2 Cumulative group {group_idx + 1}/{len(cumul_groups)}: {group}")
        params, _ = sysid.optimize(
            params,
            res_fns["balance"],
            max_iters=10,
        )
        for bn_i in range(group_idx + 1):
            bn = BODY_NAMES[bn_i]
            p = params[f"balance_{bn}"]
            print(
                f"    {bn}: mass={p.value[0]:.4f}, "
                f"pos=({p.value[1]:.4f}, {p.value[2]:.4f}, {p.value[3]:.4f}), "
                f"size=({p.value[4]:.4f}, {p.value[5]:.4f}, {p.value[6]:.4f})"
            )

    r2_rmse_by_stage = {}
    r2_rmse_by_stage["balance"] = compute_rmse(
        init_base, q_bal, dq_bal, ddq_bal, tau_bal, params
    )

    # ---- R2 Stage 2: Armature + Damping (tight bounds) ----
    print("\n--- R2 Stage 2: Armature + Damping ---")
    params["armature"].frozen = False
    params["damping"].frozen = False
    params, _ = sysid.optimize(
        params,
        res_fns["armature"],
        max_iters=10,
    )
    print(f"  armature = {params['armature'].value[0]:.6f}  (true={a_true})")
    print(f"  damping  = {params['damping'].value[0]:.6f}  (true={d_true})")
    r2_rmse_by_stage["armature"] = compute_rmse(
        init_base, q_arm, dq_arm, ddq_arm, tau_arm, params
    )

    # ---- R2 Stage 3: Frictionloss (tight bounds) ----
    print("\n--- R2 Stage 3: Frictionloss ---")
    params["frictionloss"].frozen = False
    params, _ = sysid.optimize(
        params,
        res_fns["friction"],
        max_iters=10,
    )
    print(f"  frictionloss = {params['frictionloss'].value[0]:.6f}  (true={f_true})")
    r2_rmse_by_stage["friction"] = compute_rmse(
        init_base, q_fr, dq_fr, ddq_fr, tau_fr, params
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("IDENTIFICATION SUMMARY")
    print("=" * 60)
    print(f"\n  armature:     {params['armature'].value[0]:.6f}  (true={a_true})")
    print(f"  damping:      {params['damping'].value[0]:.6f}  (true={d_true})")
    print(f"  frictionloss: {params['frictionloss'].value[0]:.6f}  (true={f_true})")
    for bn in BODY_NAMES:
        v = params[f"balance_{bn}"].value
        print(
            f"  balance_{bn}: "
            f"mass={v[0]:.4f}, pos=({v[1]:.4f},{v[2]:.4f},{v[3]:.4f}), "
            f"size=({v[4]:.4f},{v[5]:.4f},{v[6]:.4f})"
        )

    # ---- Nominal RMSE ----
    print("\n--- Nominal RMSE (fresh parse from XML) ---")
    nom_rmse = {}
    for stage_key in STAGE_YAMLS:
        q, dq, ddq, tau = trajs[stage_key]
        nom_rmse[stage_key] = compute_rmse(
            init_base, q, dq, ddq, tau, params, nominal=True
        )
        print(f"  {stage_key}:")
        for jn, e in zip(JOINT_NAMES, nom_rmse[stage_key]):
            print(f"    {jn}: {e:.4f}")

    # ---- RMSE Progression Bar Chart ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, stage in enumerate(["balance", "armature", "friction"]):
        ax = axes[idx]
        categories = ["Nominal", "R1", "R2"]
        rmses = np.array(
            [
                nom_rmse[stage],
                r1_rmse_by_stage[stage],
                r2_rmse_by_stage[stage],
            ]
        )
        x = np.arange(len(categories))
        width = 0.15
        for j in range(5):
            bars = ax.bar(x + j * width, rmses[:, j], width, label=JOINT_NAMES[j])
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )
        ax.set_title(f"{stage} trajectory")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(categories)
        ax.set_ylabel("RMSE (Nm)")
        ax.legend(fontsize=6)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.suptitle("RMSE Progression: Nominal → Round 1 → Round 2")
    plt.tight_layout()
    # plt.savefig("rmse_progression.png", dpi=150)
    # print("\nSaved rmse_progression.png")

    # ---- Torque comparison plots for each stage ----
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    for row, stage in enumerate(["balance", "armature", "friction"]):
        q, dq, ddq, tau_true = trajs[stage]
        # Optimized (R2) torque
        spec_tmp = init_base.copy()
        sysid.apply_param_modifiers_spec(params, spec_tmp)
        spec_tmp.compiler.balanceinertia = True
        model = spec_tmp.compile()
        data = mujoco.MjData(model)
        tau_opt = np.zeros_like(tau_true)
        for k in range(len(q)):
            data.qpos[:] = q[k]
            data.qvel[:] = dq[k]
            data.qacc[:] = ddq[k]
            mujoco.mj_inverse(model, data)
            tau_opt[k] = data.qfrc_inverse.copy()
        # Nominal torque (fresh parse from XML)
        spec_nom = mujoco.MjSpec.from_string(USED_XML)
        model_nom = spec_nom.compile()
        data_nom = mujoco.MjData(model_nom)
        tau_nom = np.zeros_like(tau_true)
        for k in range(len(q)):
            data_nom.qpos[:] = q[k]
            data_nom.qvel[:] = dq[k]
            data_nom.qacc[:] = ddq[k]
            mujoco.mj_inverse(model_nom, data_nom)
            tau_nom[k] = data_nom.qfrc_inverse.copy()

        t = np.arange(len(q)) / 500.0
        for col in range(5):
            ax = axes[row, col]
            ax.plot(t, tau_true[:, col], "k-", label="true", lw=1.5)
            ax.plot(t, tau_nom[:, col], "b--", label="nominal", lw=1.0, alpha=0.7)
            ax.plot(t, tau_opt[:, col], "r--", label="optimized", lw=1.5)
            ax.set_title(f"{stage} | {JOINT_NAMES[col]}")
            if row == 2:
                ax.set_xlabel("t (s)")
            if col == 0:
                ax.set_ylabel("τ (Nm)")
            ax.legend(fontsize=6)
            ax.grid(axis="both", linestyle="--", alpha=0.5)
    fig.suptitle("Torque Comparison: True vs Nominal vs Optimized (after R2)")
    plt.tight_layout()
    # plt.savefig("torque_comparison.png", dpi=150)
    # print("Saved torque_comparison.png")

    # ---- Test Trajectory RMSE Bar Chart (Nominal vs R1 vs R2) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, stage in enumerate(["balance", "armature", "friction"]):
        ax = axes[idx]
        q, dq, ddq, tau = trajs[stage]
        rmses = np.array(
            [
                nom_rmse[stage],
                r1_rmse_by_stage[stage],
                compute_rmse(init_base, q, dq, ddq, tau, params),
            ]
        )
        x = np.arange(len(JOINT_NAMES))
        width = 0.25
        colors = ["gray", "orange", "green"]
        for i, (label, color) in enumerate(zip(["Nominal", "R1", "R2"], colors)):
            bars = ax.bar(x + i * width, rmses[i], width, label=label, color=color)
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )
        ax.set_title(f"{stage} trajectory")
        ax.set_xticks(x + width)
        ax.set_xticklabels(JOINT_NAMES, fontsize=7)
        ax.set_ylabel("RMSE (Nm)")
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.suptitle("Test Trajectory RMSE: Nominal → R1 → R2")
    plt.tight_layout()
    # plt.savefig("test_rmse_bar.png", dpi=150)
    # print("Saved test_rmse_bar.png")
    plt.show()


if __name__ == "__main__":
    main()
