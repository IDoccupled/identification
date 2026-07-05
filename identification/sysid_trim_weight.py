#!/usr/bin/env python3
"""
Trim-weight approach v2 — fully manual residual function.

Each link gets a small box-shaped "trim weight" (7 params: mass, pos×3, size×3)
to compensate for CAD-to-real gap.  Mass can be negative.

Test: TRUE model generates data; NOMINAL model (perturbed masses/CoMs) + trim
weights is optimized to match TRUE dynamics.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco, mujoco.rollout as rollout
from mujoco import sysid

# ---------------------------------------------------------------------------
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

MASS_PERTURB = 1.25
IPOS_PERTURB = 0.03

# ---------------------------------------------------------------------------
LEFT_ARM_XML = """\
<mujoco model="pm_v2_left_arm">
  <compiler angle="radian" autolimits="true"/>
  <option integrator="implicitfast" timestep="0.002"><flag contact="disable"/></option>
  <worldbody>
    <body name="torso" pos="0 0 0.9"><inertial pos="0 0 0" mass="1e-3" diaginertia="1e-12 1e-12 1e-12"/>
      <body name="LINK_SHOULDER_PITCH_L" pos="-0.027105 0.12916 0.21549">
        <inertial pos="-0.0068739 0.0562468 -0.0156415" quat="0.526158 0.836720 -0.143374 0.0500143" mass="0.932155" diaginertia="0.00125198 0.00108471 0.000622336"/>
        <joint name="J13_SHOULDER_PITCH_L" type="hinge" axis="0 0.998027 0.0627908" range="-2.9671 2.7925" armature="0.039175" damping="0.07" frictionloss="0.29"/>
        <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.04" rgba="0.2 0.4 0.8 1"/>
        <body name="LINK_SHOULDER_ROLL_L" pos="-0.0371 0.066941 -0.020838">
          <inertial pos="0.037412 0.00677175 -0.0267801" quat="0.694988 -0.115505 0.0954646 0.703233" mass="0.510813" diaginertia="0.0013316 0.000992823 0.000901346"/>
          <joint name="J14_SHOULDER_ROLL_L" type="hinge" axis="1 0 0" range="-0.6108 2.3562" armature="0.039175" damping="0.08" frictionloss="0.30"/>
          <geom type="capsule" fromto="0 0 0 0.06 0 0" size="0.035" rgba="0.2 0.5 0.9 1"/>
          <body name="LINK_SHOULDER_YAW_L" pos="0.0371 0.017645 -0.070132">
            <inertial pos="-0.00084545 0.00219391 -0.045665" quat="0.999922 -0.00834297 -0.00593057 0.00715565" mass="0.909138" diaginertia="0.00162177 0.00150888 0.000787082"/>
            <joint name="J15_SHOULDER_YAW_L" type="hinge" axis="0 -0.0628027 0.998026" range="-2.618 2.618" armature="0.039175" damping="0.08" frictionloss="0.30"/>
            <geom type="capsule" fromto="0 0 -0.04 0 0 0.04" size="0.03" rgba="0.3 0.6 1.0 1"/>
            <body name="LINK_ELBOW_PITCH_L" pos="0 0.0065994 -0.10487">
              <inertial pos="0.00298237 0.00186711 -0.0651846" quat="0.850602 -0.0227837 0.0480929 0.52311" mass="1.38061" diaginertia="0.00600679 0.00586987 0.000776466"/>
              <joint name="J16_ELBOW_PITCH_L" type="hinge" axis="0.00272431 0.998022 0.0628031" range="-2.1948 0.7374" armature="0.039175" damping="0.08" frictionloss="0.30"/>
              <geom type="capsule" fromto="0 0 0 0 0 -0.12" size="0.035" rgba="0.4 0.7 1.0 1"/>
              <body name="LINK_ELBOW_YAW_L" pos="0.013817 0.0097723 -0.1547">
                <inertial pos="0.0201851 0.000217483 -0.0900144" quat="0.993617 -0.0142345 -0.111507 -0.00947408" mass="0.467519" diaginertia="0.00174612 0.00173484 0.000322726"/>
                <joint name="J17_ELBOW_YAW_L" type="hinge" axis="-0.214789 -0.0619207 0.974696" range="-2.618 2.618" armature="0.039175" damping="0.08" frictionloss="0.30"/>
                <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.025" rgba="0.5 0.8 1.0 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor_J13" joint="J13_SHOULDER_PITCH_L" ctrllimited="true" ctrlrange="-61 61"/>
    <motor name="motor_J14" joint="J14_SHOULDER_ROLL_L" ctrllimited="true" ctrlrange="-61 61"/>
    <motor name="motor_J15" joint="J15_SHOULDER_YAW_L" ctrllimited="true" ctrlrange="-61 61"/>
    <motor name="motor_J16" joint="J16_ELBOW_PITCH_L" ctrllimited="true" ctrlrange="-61 61"/>
    <motor name="motor_J17" joint="J17_ELBOW_YAW_L" ctrllimited="true" ctrlrange="-61 61"/>
  </actuator>
  <sensor>
    <jointpos name="J13_pos" joint="J13_SHOULDER_PITCH_L"/><jointvel name="J13_vel" joint="J13_SHOULDER_PITCH_L"/>
    <jointpos name="J14_pos" joint="J14_SHOULDER_ROLL_L"/><jointvel name="J14_vel" joint="J14_SHOULDER_ROLL_L"/>
    <jointpos name="J15_pos" joint="J15_SHOULDER_YAW_L"/><jointvel name="J15_vel" joint="J15_SHOULDER_YAW_L"/>
    <jointpos name="J16_pos" joint="J16_ELBOW_PITCH_L"/><jointvel name="J16_vel" joint="J16_ELBOW_PITCH_L"/>
    <jointpos name="J17_pos" joint="J17_ELBOW_YAW_L"/><jointvel name="J17_vel" joint="J17_ELBOW_YAW_L"/>
  </sensor>
</mujoco>
"""


# ---------------------------------------------------------------------------
def box_inertia(m, sx, sy, sz):
    return [
        m / 12 * (sy**2 + sz**2),
        m / 12 * (sx**2 + sz**2),
        m / 12 * (sx**2 + sy**2),
    ]


def build_nominal_spec(base_spec):
    """Copy spec, perturb masses/CoMs, return it (no trim bodies)."""
    s = base_spec.copy()
    for bn in BODY_NAMES:
        b = s.body(bn)
        b.mass = b.mass * MASS_PERTURB
        old = np.array(b.ipos) if b.ipos is not None else np.zeros(3)
        b.ipos = old + IPOS_PERTURB * np.array([1.0, -0.7, 0.5])
    return s


def apply_trims_to_spec(spec, trim_vectors):
    """Apply trim weights: adjust mass and CoM. Rotational inertia scales with mass."""
    for bn in BODY_NAMES:
        m, px, py, pz = trim_vectors[bn]
        if abs(m) < 1e-9:
            continue
        b = spec.body(bn)
        old_mass = b.mass
        old_ipos = np.array(b.ipos)
        trim_pos = np.array([px, py, pz])
        new_mass = old_mass + m
        new_ipos = (old_mass * old_ipos + m * trim_pos) / new_mass
        b.mass = new_mass
        b.ipos = new_ipos
        b.inertia = np.array(b.inertia) * (new_mass / old_mass)


def rollout_spec(spec, init_state, ctrl):
    """Compile spec and rollout, return sensor data."""
    model = spec.compile()
    data = mujoco.MjData(model)
    state, sensor = rollout.rollout(model, data, init_state, ctrl)
    return np.squeeze(sensor, axis=0)


# ---------------------------------------------------------------------------
def main():
    true_spec = mujoco.MjSpec.from_string(LEFT_ARM_XML)
    nominal_base = build_nominal_spec(true_spec)

    # Generate measured data from TRUE model
    print("Generating measured data …")
    true_model = true_spec.compile()
    true_data = mujoco.MjData(true_model)
    n_steps = int(2.0 / true_model.opt.timestep)
    t = np.arange(n_steps) * true_model.opt.timestep
    rng = np.random.default_rng(42)
    ctrl = np.column_stack(
        [
            5.0 * np.sin(2 * np.pi * 0.5 * t),
            4.0 * np.sin(2 * np.pi * 0.7 * t + 0.5),
            3.0 * np.sin(2 * np.pi * 0.4 * t + 1.0),
            2.0 * np.sin(2 * np.pi * 0.9 * t + 1.5),
            1.0 * np.sin(2 * np.pi * 0.6 * t + 2.0),
        ]
    )
    init_state = sysid.create_initial_state(
        true_model, true_data.qpos, true_data.qvel, true_data.act
    )
    sens_true = rollout_spec(true_spec, init_state, ctrl[:-1])
    noise = np.zeros(sens_true.shape[1])
    noise[:5] = 0.0005
    noise[5:] = 0.002
    sens_true += rng.normal(scale=noise, size=sens_true.shape)
    print(f"  {len(t[:-1])} steps, {sens_true.shape[1]} sensors")

    # ---- Build trim-weight parameters (5 bodies × 7 = 35) ----
    params = sysid.ParameterDict()
    for bn in BODY_NAMES:
        p = sysid.Parameter(
            f"trim_{bn}",
            nominal=np.zeros(4),  # [mass, px, py, pz]
            min_value=np.array([-0.5, -0.05, -0.05, -0.05]),
            max_value=np.array([0.5, 0.05, 0.05, 0.05]),
        )
        p.value[:] = [-0.05, 0.0, 0.0, 0.0]
        params.add(p)

    print(f"Parameters: {len(params.as_vector())}  (5 bodies × 4 trim params)")

    # ---- Manual residual function (supports batch evaluation) ----
    def residual_fn(x, param_dict):
        if x.ndim == 1:
            # Single evaluation
            param_dict.update_from_vector(x)
            tv = {bn: list(param_dict[f"trim_{bn}"].value) for bn in BODY_NAMES}
            spec_copy = nominal_base.copy()
            apply_trims_to_spec(spec_copy, tv)
            sens_pred = rollout_spec(spec_copy, init_state, ctrl[:-1])
            return [(sens_pred - sens_true).ravel()], None, None
        else:
            # Batch evaluation (2D: each column is a parameter vector)
            results = []
            for k in range(x.shape[1]):
                xk = x[:, k]
                param_dict.update_from_vector(xk)
                tv = {bn: list(param_dict[f"trim_{bn}"].value) for bn in BODY_NAMES}
                spec_copy = nominal_base.copy()
                apply_trims_to_spec(spec_copy, tv)
                sens_pred = rollout_spec(spec_copy, init_state, ctrl[:-1])
                results.append((sens_pred - sens_true).ravel())
            return [np.column_stack(results)], None, None

    # Sanity check
    x0 = params.as_vector().copy()
    r0, _, _ = residual_fn(x0, params)
    c0 = sum(np.sum(r**2) for r in r0)
    x1 = x0.copy()
    x1[0] += 0.02
    r1, _, _ = residual_fn(x1, params)
    c1 = sum(np.sum(r**2) for r in r1)
    print(f"Sanity: cost(x0)={c0:.4f}, cost(x0+Δmass)={c1:.4f}, Δ={c1 - c0:+.4f}")
    if abs(c1 - c0) < 1e-6:
        print("⚠ FAIL: trim mass has no effect. Aborting.")
        return

    # ---- Optimize ----
    print("\nOptimizing trim weights …")
    opt_params, opt_result = sysid.optimize(
        initial_params=params,
        residual_fn=residual_fn,
        optimizer="mujoco",
        verbose=True,
        eps=0.001,
    )

    # ---- Results ----
    print("\n" + "=" * 65)
    print("  Trim Weight Results  (nominal + trim → true)")
    print("=" * 65)
    for bn in BODY_NAMES:
        val = opt_params[f"trim_{bn}"].value
        m, px, py, pz = val
        true_m = TRUE_INERTIA[bn]["mass"]
        nominal_m = true_m * MASS_PERTURB
        print(f"\n  [{bn}]  true_m={true_m:.4f}, nominal_m={nominal_m:.4f}")
        print(f"    trim mass={m:+.4f}  (expected ≈{true_m - nominal_m:+.4f})")
        print(f"    trim pos=({px:+.4f}, {py:+.4f}, {pz:+.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
