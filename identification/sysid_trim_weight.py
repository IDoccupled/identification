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

MASS_PERTURB = 0.75  # CAD model is LIGHTER than reality (non-neg trim compensates)
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
    """Ensure triangle inequality on the ABSOLUTE values, then restore signs."""
    arr = np.array([Ixx, Iyy, Izz], dtype=float)
    signs = np.sign(arr)
    pos = np.abs(arr)
    pos = np.maximum(pos, 1e-12)
    for _ in range(5):
        changed = False
        for largest_idx in range(3):
            others = [i for i in range(3) if i != largest_idx]
            deficit = pos[largest_idx] - (pos[others[0]] + pos[others[1]])
            if deficit > 1e-15:
                scale = (pos[largest_idx] + 1e-10) / max(
                    pos[others[0]] + pos[others[1]], 1e-15
                )
                pos[others[0]] *= scale
                pos[others[1]] *= scale
                changed = True
        if not changed:
            break
    return pos * signs


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
    """Apply trim weights: adjust mass, CoM, and rotational inertia.
    trim_vectors: dict body_name -> [mass, px, py, pz, sx, sy, sz]

    Computes the combined inertia of the original body + a box-shaped trim weight
    at position (px,py,pz).  Uses parallel axis theorem to shift both inertias to
    the new combined CoM, then sums diagonals and enforces triangle inequality.
    """
    for bn in BODY_NAMES:
        m, px, py, pz, sx, sy, sz = trim_vectors[bn]
        if abs(m) < 1e-9:
            continue
        b = spec.body(bn)
        old_mass = b.mass
        old_ipos = np.array(b.ipos)
        old_I = np.array(b.inertia)
        trim_pos = np.array([px, py, pz])

        # Combined mass and CoM
        new_mass = old_mass + m
        new_ipos = (old_mass * old_ipos + m * trim_pos) / new_mass

        # Box inertia about its own CoM
        box_I = np.array(box_inertia(m, sx, sy, sz))

        # Shift both inertias to the new combined CoM
        old_I_shifted = parallel_axis_shift(old_I, old_mass, old_ipos, new_ipos)
        box_I_shifted = parallel_axis_shift(box_I, m, trim_pos, new_ipos)

        # Combine, make physically valid, and ensure strictly positive
        combined = old_I_shifted + box_I_shifted
        valid_I = make_valid_inertia(*combined)
        valid_I = np.maximum(valid_I, 1e-10)

        b.mass = new_mass
        b.ipos = new_ipos
        b.inertia = valid_I


def rollout_spec(spec, init_state, ctrl):
    """Compile spec and rollout, return sensor data."""
    try:
        model = spec.compile()
    except ValueError as e:
        # Print debug info on compile failure
        for bn in BODY_NAMES:
            b = spec.body(bn)
            I = np.array(b.inertia)
            ok = (
                "OK"
                if (I[0] + I[1] >= I[2] and I[0] + I[2] >= I[1] and I[1] + I[2] >= I[0])
                else "BAD"
            )
            print(
                f"  [FAIL] {bn}: mass={b.mass:.4f}, inertia={I}, valid={ok}", flush=True
            )
        raise
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

    # Start from a non-trivial pose to cover more configuration space
    true_data.qpos[:] = [0.4, -0.6, 0.3, -0.9, 0.2]

    # Longer, richer excitation: 4 seconds, multi-sine with harmonics
    duration = 4.0
    n_steps = int(duration / true_model.opt.timestep)
    t = np.arange(n_steps) * true_model.opt.timestep
    rng = np.random.default_rng(42)

    # Multi-sine with 3-5 frequencies per joint, up to ±40 Nm
    ctrl = np.column_stack(
        [
            10.0 * np.sin(2 * np.pi * 0.3 * t)
            + 8.0 * np.sin(2 * np.pi * 0.7 * t)
            + 6.0 * np.sin(2 * np.pi * 1.3 * t)
            + 4.0 * np.sin(2 * np.pi * 2.1 * t),
            12.0 * np.sin(2 * np.pi * 0.5 * t + 0.8)
            + 9.0 * np.sin(2 * np.pi * 1.1 * t)
            + 7.0 * np.sin(2 * np.pi * 1.9 * t)
            + 5.0 * np.sin(2 * np.pi * 2.7 * t),
            14.0 * np.sin(2 * np.pi * 0.4 * t + 1.5)
            + 10.0 * np.sin(2 * np.pi * 0.9 * t)
            + 8.0 * np.sin(2 * np.pi * 1.6 * t)
            + 6.0 * np.sin(2 * np.pi * 2.4 * t),
            11.0 * np.sin(2 * np.pi * 0.6 * t + 2.0)
            + 9.0 * np.sin(2 * np.pi * 1.2 * t)
            + 7.0 * np.sin(2 * np.pi * 2.0 * t)
            + 5.0 * np.sin(2 * np.pi * 3.1 * t),
            13.0 * np.sin(2 * np.pi * 0.35 * t + 0.5)
            + 10.0 * np.sin(2 * np.pi * 0.85 * t)
            + 8.0 * np.sin(2 * np.pi * 1.5 * t)
            + 6.0 * np.sin(2 * np.pi * 2.6 * t),
        ]
    )
    # Clip to actuator limits
    ctrl = np.clip(ctrl, -60, 60)

    init_state = sysid.create_initial_state(
        true_model, true_data.qpos, true_data.qvel, true_data.act
    )
    sens_true = rollout_spec(true_spec, init_state, ctrl[:-1])
    noise = np.zeros(sens_true.shape[1])
    noise[:5] = 0.0005
    noise[5:] = 0.002
    sens_true += rng.normal(scale=noise, size=sens_true.shape)
    print(f"  {len(t[:-1])} steps, {sens_true.shape[1]} sensors, duration={duration}s")

    # ---- Sequential identification: end-effector → base ----
    # Start with known mass differences as initial guesses.
    params = sysid.ParameterDict()
    for bn in BODY_NAMES:
        true_m = TRUE_INERTIA[bn]["mass"]
        nominal_m = true_m * MASS_PERTURB
        known_dm = true_m - nominal_m  # known mass gap
        p = sysid.Parameter(
            f"trim_{bn}",
            nominal=np.zeros(7),
            min_value=np.array([0.0, -0.10, -0.10, -0.10, 0.005, 0.005, 0.005]),
            max_value=np.array([1.0, 0.10, 0.10, 0.10, 0.250, 0.250, 0.250]),
        )
        # Initial: known mass diff + small perturbation, pos at origin, moderate size
        perturb = 1.0 + 0.12 * (abs(hash(bn)) % 10) / 10.0  # ±12% perturbation
        p.value[:] = [known_dm * perturb, 0.0, 0.0, 0.0, 0.06, 0.06, 0.06]
        p.frozen = True
        params.add(p)

    def make_residual_fn():
        """Closure over current params state."""

        def residual_fn(x, param_dict):
            if x.ndim == 1:
                param_dict.update_from_vector(x)
                tv = {bn: list(param_dict[f"trim_{bn}"].value) for bn in BODY_NAMES}
                spec_copy = nominal_base.copy()
                apply_trims_to_spec(spec_copy, tv)
                sens_pred = rollout_spec(spec_copy, init_state, ctrl[:-1])
                return [(sens_pred - sens_true).ravel()], None, None
            else:
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

        return residual_fn

    x_scale_7 = np.array([0.3, 0.06, 0.06, 0.06, 0.12, 0.12, 0.12])

    # Identify from last link to first
    for round_idx, bn in enumerate(reversed(BODY_NAMES)):
        print(f"\n{'=' * 60}")
        print(f"  Round {round_idx + 1}/{len(BODY_NAMES)}: identifying trim_{bn}")
        print(f"{'=' * 60}")

        # Unfreeze this body's trim, using known mass diff as initial guess
        true_m = TRUE_INERTIA[bn]["mass"]
        known_dm = true_m - true_m * MASS_PERTURB
        params[f"trim_{bn}"].frozen = False
        params[f"trim_{bn}"].value[:] = [
            known_dm * 1.05,
            0.0,
            0.0,
            0.0,
            0.06,
            0.06,
            0.06,
        ]

        n_free = len(params.as_vector())
        print(f"  Free params: {n_free} (only {bn})")

        residual_fn = make_residual_fn()

        # Sanity
        x0 = params.as_vector().copy()
        r0, _, _ = residual_fn(x0, params)
        c0 = sum(np.sum(r**2) for r in r0)
        if n_free > 0:
            x1 = x0.copy()
            x1[0] += 0.02
            r1, _, _ = residual_fn(x1, params)
            c1 = sum(np.sum(r**2) for r in r1)
            print(f"  Sanity: cost(x0)={c0:.4f}, cost(x0+Δ)={c1:.4f}, Δ={c1 - c0:+.4f}")

        # Build per-round x_scale (only for the free body)
        x_scales = []
        for b in BODY_NAMES:
            if not params[f"trim_{b}"].frozen:
                x_scales.extend(x_scale_7)

        opt_params, opt_result = sysid.optimize(
            initial_params=params,
            residual_fn=residual_fn,
            optimizer="mujoco",
            verbose=True,
            eps=0.005,
            x_scale=np.array(x_scales) if x_scales else None,
            max_iters=50,
        )
        # Copy optimized values back and freeze
        for b in BODY_NAMES:
            params[f"trim_{b}"].value[:] = opt_params[f"trim_{b}"].value
        params[f"trim_{bn}"].frozen = True

        val = params[f"trim_{bn}"].value
        print(
            f"  → {bn}: mass={val[0]:+.4f}, pos=({val[1]:+.3f},{val[2]:+.3f},{val[3]:+.3f}), "
            f"size=({val[4]:.4f},{val[5]:.4f},{val[6]:.4f})"
        )

    # ---- Final Results ----
    print("\n" + "=" * 65)
    print("  Sequential Trim Weight Results  (end→base)")
    print("=" * 65)
    for bn in BODY_NAMES:
        val = params[f"trim_{bn}"].value
        m, px, py, pz, sx, sy, sz = val
        true_m = TRUE_INERTIA[bn]["mass"]
        nominal_m = true_m * MASS_PERTURB
        print(f"\n  [{bn}]  true={true_m:.4f}, nominal={nominal_m:.4f}")
        print(f"    mass={m:+.4f} kg  (expect ≈{true_m - nominal_m:+.4f})")
        print(f"    pos=({px:+.4f}, {py:+.4f}, {pz:+.4f}) m")
        print(f"    size=({sx:.4f}, {sy:.4f}, {sz:.4f}) m")

    print("\nDone.")


if __name__ == "__main__":
    main()
