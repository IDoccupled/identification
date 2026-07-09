#!/usr/bin/env python3
"""
Parameter Identification for PM-V2 Left Arm (J13–J17).

Three-stage approach with configurable parameter sets:
  Stage 1a — Joint armature (5 params, highly identifiable ~3-8% error).
  Stage 1b — Joint damping + frictionloss (10 params, armature frozen).
             NOTE: damping and frictionloss are partially collinear;
             frictionloss often collapses to bounds with joint-only data.
  Stage 2  — Link inertial parameters with joint params frozen.
             Supports InertiaType.Pseudo (mass+CoM+rot.inertia, 10/link)
             and InertiaType.MassIpos (mass+CoM, 4/link).
             Optionally freeze the last link (FROZEN_BODY) to break
             identifiability degeneracy.  Multiple payload trajectories
             (PAYLOAD_MASSES) provide structural perturbations.

KNOWN LIMITATIONS:
  - Full Pseudo-inertia (50 params) is severely underdetermined with
    joint position/velocity/torque sensors alone.
  - Damping and frictionloss trade off against each other.
  - Best practical result: MassIpos + frozen last link → ~30% mass error.
  - For full inertia identification, add base force/torque sensors or
    use diverse known payloads (see MuJoCo sysid notebook Section 4).

Usage:
    python3 -m identification.sysid_left_arm
"""

from __future__ import annotations

import base64
import os

os.environ["MUJOCO_GL"] = "egl"

import numpy as np

import mujoco
import mujoco.rollout as rollout
from mujoco import sysid

import matplotlib.pyplot as plt
import mediapy as media
from IPython.display import IFrame

# ---------------------------------------------------------------------------
# 1.  Build a standalone left-arm MJCF
# ---------------------------------------------------------------------------
# Kinematic chain:  LINK_TORSO_YAW → SHOULDER_PITCH_L → SHOULDER_ROLL_L
#                   → SHOULDER_YAW_L → ELBOW_PITCH_L → ELBOW_YAW_L → end.
# Joints: J13–J17.  Parameters taken verbatim from serial_links.xml.

LEFT_ARM_XML = """\
<mujoco model="pm_v2_left_arm">
  <compiler angle="radian" autolimits="true"/>
  <option integrator="implicitfast" timestep="0.002">
    <flag contact="disable"/>
  </option>

  <worldbody>
    <body name="torso" pos="0 0 0.9">
      <inertial pos="0 0 0" mass="1e-3" diaginertia="1e-12 1e-12 1e-12"/>

      <body name="LINK_SHOULDER_PITCH_L" pos="-0.027105 0.12916 0.21549">
        <inertial
            pos="-0.0068739 0.0562468 -0.0156415"
            quat="0.526158 0.836720 -0.143374 0.0500143"
            mass="0.932155"
            diaginertia="0.00125198 0.00108471 0.000622336"/>
        <joint name="J13_SHOULDER_PITCH_L"
               type="hinge"
               axis="0 0.998027 0.0627908"
               range="-2.9671 2.7925"
               armature="0.039175" damping="0.07" frictionloss="0.29"/>
        <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.04"
              rgba="0.2 0.4 0.8 1"/>

        <body name="LINK_SHOULDER_ROLL_L" pos="-0.0371 0.066941 -0.020838">
          <inertial
              pos="0.037412 0.00677175 -0.0267801"
              quat="0.694988 -0.115505 0.0954646 0.703233"
              mass="0.510813"
              diaginertia="0.0013316 0.000992823 0.000901346"/>
          <joint name="J14_SHOULDER_ROLL_L"
                 type="hinge"
                 axis="1 0 0"
                 range="-0.6108 2.3562"
                 armature="0.039175" damping="0.08" frictionloss="0.30"/>
          <geom type="capsule" fromto="0 0 0 0.06 0 0" size="0.035"
                rgba="0.2 0.5 0.9 1"/>

          <body name="LINK_SHOULDER_YAW_L" pos="0.0371 0.017645 -0.070132">
            <inertial
                pos="-0.00084545 0.00219391 -0.045665"
                quat="0.999922 -0.00834297 -0.00593057 0.00715565"
                mass="0.909138"
                diaginertia="0.00162177 0.00150888 0.000787082"/>
            <joint name="J15_SHOULDER_YAW_L"
                   type="hinge"
                   axis="0 -0.0628027 0.998026"
                   range="-2.618 2.618"
                   armature="0.039175" damping="0.08" frictionloss="0.30"/>
            <geom type="capsule" fromto="0 0 -0.04 0 0 0.04" size="0.03"
                  rgba="0.3 0.6 1.0 1"/>

            <body name="LINK_ELBOW_PITCH_L" pos="0 0.0065994 -0.10487">
              <inertial
                  pos="0.00298237 0.00186711 -0.0651846"
                  quat="0.850602 -0.0227837 0.0480929 0.52311"
                  mass="1.38061"
                  diaginertia="0.00600679 0.00586987 0.000776466"/>
              <joint name="J16_ELBOW_PITCH_L"
                     type="hinge"
                     axis="0.00272431 0.998022 0.0628031"
                     range="-2.1948 0.7374"
                     armature="0.039175" damping="0.08" frictionloss="0.30"/>
              <geom type="capsule" fromto="0 0 0 0 0 -0.12" size="0.035"
                    rgba="0.4 0.7 1.0 1"/>

              <body name="LINK_ELBOW_YAW_L" pos="0.013817 0.0097723 -0.1547">
                <inertial
                    pos="0.0201851 0.000217483 -0.0900144"
                    quat="0.993617 -0.0142345 -0.111507 -0.00947408"
                    mass="0.467519"
                    diaginertia="0.00174612 0.00173484 0.000322726"/>
                <joint name="J17_ELBOW_YAW_L"
                       type="hinge"
                       axis="-0.214789 -0.0619207 0.974696"
                       range="-2.618 2.618"
                       armature="0.039175" damping="0.08" frictionloss="0.30"/>
                <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.025"
                      rgba="0.5 0.8 1.0 1"/>

                <body name="LINK_ELBOW_END_L">
                  <inertial pos="0 0 0" mass="1e-3"
                            diaginertia="1e-12 1e-12 1e-12"/>
                  <geom type="sphere" size="0.02" rgba="0.9 0.3 0.3 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor_J13" joint="J13_SHOULDER_PITCH_L"
           ctrllimited="true" ctrlrange="-61 61"/>
    <motor name="motor_J14" joint="J14_SHOULDER_ROLL_L"
           ctrllimited="true" ctrlrange="-61 61"/>
    <motor name="motor_J15" joint="J15_SHOULDER_YAW_L"
           ctrllimited="true" ctrlrange="-61 61"/>
    <motor name="motor_J16" joint="J16_ELBOW_PITCH_L"
           ctrllimited="true" ctrlrange="-61 61"/>
    <motor name="motor_J17" joint="J17_ELBOW_YAW_L"
           ctrllimited="true" ctrlrange="-61 61"/>
  </actuator>

  <sensor>
    <jointpos name="J13_pos" joint="J13_SHOULDER_PITCH_L"/>
    <jointvel name="J13_vel" joint="J13_SHOULDER_PITCH_L"/>
    <jointpos name="J14_pos" joint="J14_SHOULDER_ROLL_L"/>
    <jointvel name="J14_vel" joint="J14_SHOULDER_ROLL_L"/>
    <jointpos name="J15_pos" joint="J15_SHOULDER_YAW_L"/>
    <jointvel name="J15_vel" joint="J15_SHOULDER_YAW_L"/>
    <jointpos name="J16_pos" joint="J16_ELBOW_PITCH_L"/>
    <jointvel name="J16_vel" joint="J16_ELBOW_PITCH_L"/>
    <jointpos name="J17_pos" joint="J17_ELBOW_YAW_L"/>
    <jointvel name="J17_vel" joint="J17_ELBOW_YAW_L"/>
  </sensor>
</mujoco>
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
]

BODY_NAMES = [
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
]

# Payload masses for structural perturbation (kg).  Use 2-3 different masses
# to break identifiability.  Each payload trajectory adds independent equations.
PAYLOAD_MASSES = [0.5, 1.5]  # kg, added to LINK_ELBOW_END_L

TRUE_ARMATURE = {n: 0.039175 for n in JOINT_NAMES}

TRUE_DAMPING = {
    "J13_SHOULDER_PITCH_L": 0.07,
    "J14_SHOULDER_ROLL_L": 0.08,
    "J15_SHOULDER_YAW_L": 0.08,
    "J16_ELBOW_PITCH_L": 0.08,
    "J17_ELBOW_YAW_L": 0.08,
}

TRUE_FRICTIONLOSS = {
    "J13_SHOULDER_PITCH_L": 0.29,
    "J14_SHOULDER_ROLL_L": 0.30,
    "J15_SHOULDER_YAW_L": 0.30,
    "J16_ELBOW_PITCH_L": 0.30,
    "J17_ELBOW_YAW_L": 0.30,
}

# True inertial parameters (from serial_links.xml <inertial> tags)
TRUE_INERTIA: dict[str, dict] = {
    "LINK_SHOULDER_PITCH_L": {
        "mass": 0.932155,
        "ipos": [-0.0068739, 0.0562468, -0.0156415],
        "diaginertia": [0.00125198, 0.00108471, 0.000622336],
    },
    "LINK_SHOULDER_ROLL_L": {
        "mass": 0.510813,
        "ipos": [0.037412, 0.00677175, -0.0267801],
        "diaginertia": [0.0013316, 0.000992823, 0.000901346],
    },
    "LINK_SHOULDER_YAW_L": {
        "mass": 0.909138,
        "ipos": [-0.00084545, 0.00219391, -0.045665],
        "diaginertia": [0.00162177, 0.00150888, 0.000787082],
    },
    "LINK_ELBOW_PITCH_L": {
        "mass": 1.38061,
        "ipos": [0.00298237, 0.00186711, -0.0651846],
        "diaginertia": [0.00600679, 0.00586987, 0.000776466],
    },
    "LINK_ELBOW_YAW_L": {
        "mass": 0.467519,
        "ipos": [0.0201851, 0.000217483, -0.0900144],
        "diaginertia": [0.00174612, 0.00173484, 0.000322726],
    },
}

# Choose inertia type:
#   Pseudo   = 10 params/link (mass, CoM, rot. inertia) — physically plausible
#   MassIpos =  4 params/link (mass, CoM)                — more identifiable
INERTIA_TYPE = sysid.InertiaType.Pseudo

# Freeze the last link to break identifiability degeneracy.
FROZEN_BODY = "LINK_ELBOW_YAW_L"
FROZEN_JOINT = "J17_ELBOW_YAW_L"

# Initial perturbation
MASS_SCALE = 1.3  # start 30% too heavy
IPOS_OFFSET = 0.02  # CoM offset 2 cm
INIT_ARMATURE = 0.005  # armature ~8× too small
INIT_DAMPING = 0.02  # damping ~3-4× too small
INIT_FRICTION = 0.05  # frictionloss ~6× too small

# ---------------------------------------------------------------------------
# 2.  Generate "measured" data
# ---------------------------------------------------------------------------


def generate_traj_n(
    spec: mujoco.MjSpec,
    index: int,
    duration: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, mujoco.MjModel]:
    """Generate trajectory #index with varied start pose."""
    model = spec.compile()
    data = mujoco.MjData(model)

    start_poses = [
        [0.0, 0.0, 0.0, 0.0, 0.0],  # 0: neutral
        [0.3, -0.5, 0.2, -0.8, 0.1],  # 1: arm bent
        [-0.2, 0.6, -0.3, -0.4, 0.5],  # 2: another bent
    ]
    data.qpos[:] = start_poses[index % len(start_poses)]

    n_steps = int(duration / model.opt.timestep)
    t = np.arange(n_steps) * model.opt.timestep
    rng = np.random.default_rng(42 + index * 100)

    ctrl = np.column_stack(
        [
            5.0 * np.sin(2 * np.pi * 0.5 * t),
            4.0 * np.sin(2 * np.pi * 0.7 * t + 0.5),
            3.0 * np.sin(2 * np.pi * 0.4 * t + 1.0),
            2.0 * np.sin(2 * np.pi * 0.9 * t + 1.5),
            1.0 * np.sin(2 * np.pi * 0.6 * t + 2.0),
        ]
    )

    init_state = sysid.create_initial_state(model, data.qpos, data.qvel, data.act)
    state, sensor = rollout.rollout(model, data, init_state, ctrl[:-1])
    sensor = np.squeeze(sensor, axis=0)

    noise_std = np.zeros(sensor.shape[1])
    noise_std[:5] = 0.0005
    noise_std[5:] = 0.002
    sensor_noisy = sensor + rng.normal(scale=noise_std, size=sensor.shape)

    return t, ctrl, sensor_noisy, init_state, model


def make_payload_spec(base_spec: mujoco.MjSpec, payload_mass: float) -> mujoco.MjSpec:
    """Return a copy of base_spec with payload mass added to the end-effector body."""
    s = base_spec.copy()
    s.body("LINK_ELBOW_END_L").mass += payload_mass
    return s


# ---------------------------------------------------------------------------
# 3.  Parameter definition
# ---------------------------------------------------------------------------

# (definitions moved inline in main() for the two-stage approach)


# ---------------------------------------------------------------------------
# 4.  Utilities
# ---------------------------------------------------------------------------


def display_report(report):
    html_b64 = base64.b64encode(report.build().encode()).decode()
    return IFrame(
        src=f"data:text/html;base64,{html_b64}",
        width="100%",
        height=800,
    )


def set_body_rgba(body: mujoco.MjSpecBody, rgba: list[float]):
    for geom in body.geoms:
        geom.rgba = rgba
    for child in body.bodies:
        set_body_rgba(child, rgba)


def make_colored_model(
    base_spec: mujoco.MjSpec,
    rgba: list[float],
    opt_params: sysid.ParameterDict | None = None,
) -> mujoco.MjModel:
    """Compile a coloured model, optionally with optimized parameters applied."""
    s = base_spec.copy()
    if opt_params is not None:
        sysid.apply_param_modifiers_spec(opt_params, s)
    set_body_rgba(s.worldbody, rgba)
    return s.compile()


def print_inertia_results(opt_params: sysid.ParameterDict):
    """Print identified vs true mass, ipos, and diaginertia for each body."""
    print("\n" + "=" * 72)
    print("  Body Inertial Parameters  (Pseudo-inertia → mass / ipos / diaginertia)")
    print("=" * 72)

    for body_name in BODY_NAMES:
        true = TRUE_INERTIA[body_name]
        pname = f"{body_name}_inertia"
        val = opt_params[pname].value

        if INERTIA_TYPE == sysid.InertiaType.Pseudo:
            mass_est = val[-1]
            ipos_est = val[-4:-1]
        else:
            mass_est = val[0]
            ipos_est = val[1:4]
        mass_err = (mass_est - true["mass"]) / true["mass"] * 100

        print(f"\n  [{body_name}]")
        print(
            f"    mass       : {mass_est:10.6f}  (true: {true['mass']:10.6f})  "
            f"{mass_err:+.2f}%"
        )
        for i, axis in enumerate(["x", "y", "z"]):
            err = ipos_est[i] - true["ipos"][i]
            print(
                f"    ipos_{axis}     : {ipos_est[i]:10.6f}  "
                f"(true: {true['ipos'][i]:10.6f})  Δ={err:+.4f} m"
            )

        # Read back diaginertia from a copy spec (only for Pseudo type)
        if INERTIA_TYPE == sysid.InertiaType.Pseudo:
            tmp = mujoco.MjSpec.from_string(LEFT_ARM_XML)
            sysid.apply_param_modifiers_spec(opt_params, tmp)
            diag = tmp.body(body_name).inertia  # (3,) ndarray
            print(f"    diaginertia: [{diag[0]:.6e}, {diag[1]:.6e}, {diag[2]:.6e}]")
            print(
                f"    (true)     : [{true['diaginertia'][0]:.6e}, "
                f"{true['diaginertia'][1]:.6e}, {true['diaginertia'][2]:.6e}]"
            )
        else:
            print("    diaginertia: (fixed at nominal — use MassIpos type)")


# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------


def main():
    spec_base = mujoco.MjSpec.from_string(LEFT_ARM_XML)

    # Build payload specs
    payload_specs = [make_payload_spec(spec_base, m) for m in PAYLOAD_MASSES]
    all_specs = [spec_base] + payload_specs
    n_traj = len(all_specs)

    # ---- Generate trajectories: baseline + one per payload ----
    print(
        f"Generating {n_traj} trajectories (baseline + {len(PAYLOAD_MASSES)} payloads) …"
    )
    for i, (sp, label) in enumerate(
        zip(all_specs, ["baseline"] + [f"+{m}kg" for m in PAYLOAD_MASSES])
    ):
        print(f"  Traj {i}: {label}")

    model_seqs = []
    for idx, sp in enumerate(all_specs):
        t_i, ctrl_i, sens_i, init_i, model_i = generate_traj_n(
            sp, index=idx, duration=2.0
        )
        times_i = t_i[:-1]
        ctrl_ts = sysid.TimeSeries(t_i, ctrl_i)
        sens_ts = sysid.TimeSeries.from_names(times_i, sens_i, model_i)
        ms = sysid.ModelSequences(
            "pm_v2_left_arm", sp, f"traj{idx}", init_i, ctrl_ts, sens_ts
        )
        model_seqs.append(ms)
        print(f"    {len(times_i)} steps, {sens_i.shape[1]} sensor channels")

    # Quick plot
    ncols = min(n_traj, 3)
    nrows = (n_traj + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False
    )
    for idx, sp in enumerate(all_specs):
        _t, _c, _s, _i, _m = generate_traj_n(sp, index=idx, duration=2.0)
        ax = axes[idx // ncols][idx % ncols]
        for j in range(5):
            ax.plot(_t[:-1], _s[:, j], lw=0.6, alpha=0.7, label=f"J{13 + j}")
        label = "baseline" if idx == 0 else f"+{PAYLOAD_MASSES[idx - 1]}kg"
        ax.set_title(f"Traj {idx}: {label}")
        ax.set_ylabel("Pos (rad)")
        ax.legend(fontsize=6, ncol=5)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("measured_data.png", dpi=150)
    plt.close()
    print("  → saved measured_data.png")

    # =====================================================================
    # Stage 1a: Armature only (5 params) — highly identifiable
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Stage 1a: Armature Identification (5 params)")
    print("=" * 60)

    arm_params = sysid.ParameterDict()
    for name in JOINT_NAMES:
        arm_params.add(
            sysid.Parameter(
                f"{name}_armature",
                nominal=TRUE_ARMATURE[name],
                min_value=0.001,
                max_value=0.2,
                modifier=lambda s, p, n=name: setattr(
                    s.joint(n), "armature", p.value[0]
                ),
            )
        )
        arm_params[f"{name}_armature"].value[:] = INIT_ARMATURE

    residual_arm = sysid.build_residual_fn(models_sequences=model_seqs)
    opt_arm, _ = sysid.optimize(
        arm_params, residual_arm, optimizer="mujoco", verbose=True
    )

    print("\n  Armature results:")
    for name in JOINT_NAMES:
        est = opt_arm[f"{name}_armature"].value[0]
        true = TRUE_ARMATURE[name]
        print(
            f"    {name:30s}  {est:.6f}  (true: {true:.6f},  {(est - true) / true * 100:+.2f}%)"
        )

    # =====================================================================
    # Stage 1b: Damping + Frictionloss (10 params), armature frozen
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Stage 1b: Damping + Frictionloss (10 params, armature frozen)")
    print("=" * 60)

    joint_params = sysid.ParameterDict()
    for name in JOINT_NAMES:
        # frozen armature
        joint_params.add(
            sysid.Parameter(
                f"{name}_armature",
                nominal=TRUE_ARMATURE[name],
                min_value=0.001,
                max_value=0.2,
                frozen=True,
                modifier=lambda s, p, n=name: setattr(
                    s.joint(n), "armature", p.value[0]
                ),
            )
        )
        joint_params[f"{name}_armature"].value[:] = opt_arm[f"{name}_armature"].value[0]

        # damping
        def _damp_mod(spec, param, n=name):
            spec.joint(n).damping = np.array([[param.value[0]], [0.0], [0.0]])

        joint_params.add(
            sysid.Parameter(
                f"{name}_damping",
                nominal=TRUE_DAMPING[name],
                min_value=0.001,
                max_value=0.5,
                modifier=_damp_mod,
            )
        )
        joint_params[f"{name}_damping"].value[:] = INIT_DAMPING

        # frictionloss
        def _fric_mod(spec, param, n=name):
            spec.joint(n).frictionloss = param.value[0]

        joint_params.add(
            sysid.Parameter(
                f"{name}_frictionloss",
                nominal=TRUE_FRICTIONLOSS[name],
                min_value=0.001,
                max_value=1.0,
                modifier=_fric_mod,
            )
        )
        joint_params[f"{name}_frictionloss"].value[:] = INIT_FRICTION

    residual_joint = sysid.build_residual_fn(models_sequences=model_seqs)
    opt_joint, _ = sysid.optimize(
        joint_params, residual_joint, optimizer="mujoco", verbose=True
    )

    print("\n  Joint parameter results:")
    for name in JOINT_NAMES:
        for key, true_dict in [
            ("armature", TRUE_ARMATURE),
            ("damping", TRUE_DAMPING),
            ("frictionloss", TRUE_FRICTIONLOSS),
        ]:
            est = opt_joint[f"{name}_{key}"].value[0]
            true = true_dict[name]
            pct = (est - true) / true * 100
            print(
                f"    {name:30s} {key:12s} = {est:.6f}  (true: {true:.6f},  {pct:+.2f}%)"
            )

    # =====================================================================
    # Stage 2: Inertia Identification with frozen joint params
    # =====================================================================
    print("\n" + "=" * 60)
    inertia_type_name = (
        "Pseudo (mass+CoM+rot.inertia)"
        if INERTIA_TYPE == sysid.InertiaType.Pseudo
        else "MassIpos (mass+CoM)"
    )
    n_bodies_free = sum(
        1 for b in BODY_NAMES if FROZEN_BODY is None or b != FROZEN_BODY
    )
    params_per_body = 10 if INERTIA_TYPE == sysid.InertiaType.Pseudo else 4
    print(
        f"  Stage 2: {inertia_type_name} ({n_bodies_free} free links × {params_per_body} params)"
    )
    if FROZEN_BODY:
        print(f"           Frozen body: {FROZEN_BODY}")
    print("=" * 60)

    ref_model = spec_base.compile()
    params = sysid.ParameterDict()

    for body_name in BODY_NAMES:
        freeze_this = FROZEN_BODY is not None and body_name == FROZEN_BODY
        extra_kw = {}
        if INERTIA_TYPE == sysid.InertiaType.Pseudo:
            extra_kw = dict(
                stretch_bound_mult=np.array([0.1, 5.0]),
                shear_bound_off=np.array([-0.01, 0.01]),
            )
        p = sysid.body_inertia_param(
            spec_base,
            ref_model,
            body_name,
            inertia_type=INERTIA_TYPE,
            scale_rot_inertia=True,
            mass_bound_mult=np.array([0.1, 5.0]),
            ipos_bound_off=np.array([-0.15, 0.15]),
            **extra_kw,
        )
        if freeze_this:
            p.frozen = True
        params.add(p)

        true = TRUE_INERTIA[body_name]
        nominal_mass = true["mass"]
        nominal_ipos = np.array(true["ipos"])

        if INERTIA_TYPE == sysid.InertiaType.Pseudo:
            if freeze_this:
                p.value[-1] = nominal_mass
                p.value[-4:-1] = nominal_ipos
            else:
                p.value[-1] = nominal_mass * MASS_SCALE
                p.value[-4:-1] = nominal_ipos + IPOS_OFFSET
        else:
            if freeze_this:
                p.value[0] = nominal_mass
                p.value[1:4] = nominal_ipos
            else:
                p.value[0] = nominal_mass * MASS_SCALE
                p.value[1:4] = nominal_ipos + IPOS_OFFSET

    # Add frozen joint parameters
    for name in JOINT_NAMES:
        # armature
        params.add(
            sysid.Parameter(
                f"{name}_armature",
                nominal=0.04,
                min_value=0.001,
                max_value=0.2,
                frozen=True,
                modifier=lambda s, p, n=name: setattr(
                    s.joint(n), "armature", p.value[0]
                ),
            )
        )
        params[f"{name}_armature"].value[:] = opt_joint[f"{name}_armature"].value[0]

        # damping
        def _damp_frozen(spec, param, n=name):
            spec.joint(n).damping = np.array([[param.value[0]], [0.0], [0.0]])

        params.add(
            sysid.Parameter(
                f"{name}_damping",
                nominal=0.04,
                min_value=0.001,
                max_value=0.5,
                frozen=True,
                modifier=_damp_frozen,
            )
        )
        params[f"{name}_damping"].value[:] = opt_joint[f"{name}_damping"].value[0]

        # frictionloss
        def _fric_frozen(spec, param, n=name):
            spec.joint(n).frictionloss = param.value[0]

        params.add(
            sysid.Parameter(
                f"{name}_frictionloss",
                nominal=0.04,
                min_value=0.001,
                max_value=1.0,
                frozen=True,
                modifier=_fric_frozen,
            )
        )
        params[f"{name}_frictionloss"].value[:] = opt_joint[
            f"{name}_frictionloss"
        ].value[0]

    n_free = len(params.as_vector())
    print(
        f"  Free parameters: {n_free}  ({n_bodies_free} bodies × {params_per_body} inertia, joint params frozen)"
    )

    residual_inertia = sysid.build_residual_fn(models_sequences=model_seqs)
    opt_params, opt_result = sysid.optimize(
        initial_params=params,
        residual_fn=residual_inertia,
        optimizer="mujoco",
        verbose=True,
    )

    # Merge joint results
    for name in JOINT_NAMES:
        for key in ["armature", "damping", "frictionloss"]:
            opt_params[f"{name}_{key}"].value[:] = opt_joint[f"{name}_{key}"].value

    # ---- Results ----
    print_inertia_results(opt_params)

    print("\n--- Joint Parameters (from Stage 1a+1b) ---")
    for name in JOINT_NAMES:
        for key, true_dict in [
            ("armature", TRUE_ARMATURE),
            ("damping", TRUE_DAMPING),
            ("frictionloss", TRUE_FRICTIONLOSS),
        ]:
            est = opt_joint[f"{name}_{key}"].value[0]
            true = true_dict[name]
            print(
                f"  {name:30s} {key:12s} = {est:.6f}  (true: {true:.6f},  {(est - true) / true * 100:+.2f}%)"
            )

    # ---- HTML Report ----
    print("\nBuilding HTML report …")
    report = sysid.default_report(
        models_sequences=model_seqs,
        initial_params=params,
        opt_params=opt_params,
        residual_fn=residual_inertia,
        opt_result=opt_result,
        title=f"PM-V2 Left Arm — {inertia_type_name} + Joint Params (Payloads: "
        + ", ".join(f"{m}kg" for m in PAYLOAD_MASSES)
        + ")",
        generate_videos=False,
    )
    html_path = "left_arm_sysid_report.html"
    with open(html_path, "w") as f:
        f.write(report.build())
    print(f"  → saved {html_path}")

    # ---- Comparison rendering ----
    print("\nRendering comparison …")
    green = [0.2, 0.8, 0.2, 0.7]
    red = [1.0, 0.2, 0.2, 0.7]
    blue = [0.2, 0.4, 1.0, 0.7]
    truth_model = make_colored_model(spec_base, green, None)
    init_model = make_colored_model(spec_base, red, params)
    opt_model = make_colored_model(spec_base, blue, opt_params)

    fps = 30
    all_frames = []
    _t0, ctrl0, _s0, init0, _m0 = generate_traj_n(spec_base, index=0, duration=2.0)
    for m_c in [init_model, opt_model]:
        models_c = [m_c, truth_model]
        datas_c = [mujoco.MjData(m) for m in models_c]
        state_c, _ = rollout.rollout(models_c, datas_c, init0, ctrl0[:-1])
        all_frames.extend(
            sysid.render_rollout(
                models_c, datas_c[0], state_c, framerate=fps, height=480, width=640
            )
        )

    try:
        media.write_video("left_arm_comparison.mp4", all_frames, fps=fps, qp=23)
        print("  → saved left_arm_comparison.mp4")
    except RuntimeError:
        np.save("frames_comparison.npy", np.array(all_frames))
        print("  ⚠ ffmpeg not found → saved frames_comparison.npy")

    print("\nDone.")


if __name__ == "__main__":
    main()
