#!/usr/bin/env python3
"""
Balance-weight approach v2 — fully manual residual function.

Each link gets a small box-shaped "balance weight" (7 params: mass, pos×3, size×3)
to compensate for CAD-to-real gap.  Mass can be negative.

Test: TRUE model generates data; NOMINAL model (perturbed masses/CoMs) + balance
weights is optimized to match TRUE dynamics.
"""

import os
import time

os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
import mujoco.rollout as rollout
from mujoco import sysid
import matplotlib.pyplot as plt

import pathlib

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

MASS_PERTURB = 0.75  # CAD is 25% lighter
IPOS_PERTURB = 1.25  # CoM scaled by 25% (preserves sign direction)
INERTIA_PERTURB = 1.0  # rotational inertia unchanged
SHOW_GUI = False  # Set False for headless batch runs

_RESOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "resource"


def _load_xml(filename):
    """Read an XML model file from the resource directory."""
    return (_RESOURCE_DIR / filename).read_text()


LEFT_ARM_XML = _load_xml("left_arm_true.xml")

# Nominal (perturbed) version of the same XML for optimization input.
# Masses are scaled by MASS_PERTURB=0.75, CoM positions by IPOS_PERTURB=1.25.
NOMINAL_LEFT_ARM_XML = _load_xml("left_arm_nominal.xml")


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


def rollout_spec(spec, init_state, ctrl):
    """Compile spec and rollout, return sensor data."""
    try:
        model = spec.compile()
    except ValueError:
        # Print debug info on compile failure
        for bn in BODY_NAMES:
            b = spec.body(bn)
            i = np.array(b.inertia)
            ok = (
                "OK"
                if (i[0] + i[1] >= i[2] and i[0] + i[2] >= i[1] and i[1] + i[2] >= i[0])
                else "BAD"
            )
            print(
                f"  [FAIL] {bn}: mass={b.mass:.4f}, inertia={i}, valid={ok}", flush=True
            )
        raise
    data = mujoco.MjData(model)
    state, sensor = rollout.rollout(model, data, init_state, ctrl)
    return np.squeeze(sensor, axis=0)


def run_with_viewer(model, data, ctrl, title="Excitation Trajectory"):
    """Replay a pre-computed control trajectory with an interactive viewer.

    Steps through `ctrl` in real-time using the model's timestep.  The viewer
    window stays open until the user closes it or the trajectory finishes.
    """
    import mujoco.viewer  # lazy import so headless runs don't fail early

    # Reset to a well-defined initial pose
    data.qpos[:] = [0.4, -0.6, 0.3, -0.9, 0.2]
    data.qvel[:] = 0.0
    data.act[:] = 0.0
    mujoco.mj_forward(model, data)

    print(f"\nLaunching viewer: {title} …")
    print("  (close the window or press Esc to continue)\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Optional: adjust camera
        # viewer.cam.lookat[:] = [0, 0, 0.5]
        # viewer.cam.distance = 1.5

        idx = 0
        while viewer.is_running() and idx < len(ctrl):
            step_start = time.time()

            data.ctrl[:] = ctrl[idx]
            mujoco.mj_step(model, data)
            viewer.sync()

            idx += 1

            # Real-time synchronisation
            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Brief pause so the user can see the final pose
        if viewer.is_running():
            time.sleep(1.0)


# ---------------------------------------------------------------------------
def main():
    true_spec = mujoco.MjSpec.from_string(LEFT_ARM_XML)
    nominal_base = mujoco.MjSpec.from_string(NOMINAL_LEFT_ARM_XML)

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

    # ---- Visualise the excitation trajectory on the true model ----
    if SHOW_GUI:
        run_with_viewer(true_model, true_data, ctrl, title="True Model — Excitation")

    # ---- Sequential identification: end-effector → base ----
    # Start with known mass differences as initial guesses.
    params = sysid.ParameterDict()
    for bn in BODY_NAMES:
        true_m = TRUE_INERTIA[bn]["mass"]
        nominal_m = NOMINAL_INERTIA[bn]["mass"]
        known_dm = true_m - nominal_m  # known mass gap
        p = sysid.Parameter(
            f"balance_{bn}",
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
                tv = {bn: list(param_dict[f"balance_{bn}"].value) for bn in BODY_NAMES}
                spec_copy = nominal_base.copy()
                apply_balances_to_spec(spec_copy, tv)
                sens_pred = rollout_spec(spec_copy, init_state, ctrl[:-1])
                return [(sens_pred - sens_true).ravel()], None, None
            else:
                results = []
                for k in range(x.shape[1]):
                    xk = x[:, k]
                    param_dict.update_from_vector(xk)
                    tv = {
                        bn: list(param_dict[f"balance_{bn}"].value) for bn in BODY_NAMES
                    }
                    spec_copy = nominal_base.copy()
                    apply_balances_to_spec(spec_copy, tv)
                    sens_pred = rollout_spec(spec_copy, init_state, ctrl[:-1])
                    results.append((sens_pred - sens_true).ravel())
                return [np.column_stack(results)], None, None

        return residual_fn

    x_scale_7 = np.array([0.3, 0.06, 0.06, 0.06, 0.12, 0.12, 0.12])

    # Identify from last link to first
    for round_idx, bn in enumerate(reversed(BODY_NAMES)):
        print(f"\n{'=' * 60}")
        print(f"  Round {round_idx + 1}/{len(BODY_NAMES)}: identifying balance_{bn}")
        print(f"{'=' * 60}")

        # Unfreeze this body's balance, using known mass diff as initial guess
        true_m = TRUE_INERTIA[bn]["mass"]
        nominal_m = NOMINAL_INERTIA[bn]["mass"]
        known_dm = true_m - nominal_m
        params[f"balance_{bn}"].frozen = False
        params[f"balance_{bn}"].value[:] = [
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
            if not params[f"balance_{b}"].frozen:
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
            params[f"balance_{b}"].value[:] = opt_params[f"balance_{b}"].value
        params[f"balance_{bn}"].frozen = True

        val = params[f"balance_{bn}"].value
        print(
            f"  → {bn}: mass={val[0]:+.4f}, pos=({val[1]:+.3f},{val[2]:+.3f},{val[3]:+.3f}), "
            f"size=({val[4]:.4f},{val[5]:.4f},{val[6]:.4f})"
        )

    # ---- Final Results ----
    print("\n" + "=" * 65)
    print("  Sequential Balance Weight Results  (end→base)")
    print("=" * 65)
    for bn in BODY_NAMES:
        val = params[f"balance_{bn}"].value
        m, px, py, pz, sx, sy, sz = val
        true_m = TRUE_INERTIA[bn]["mass"]
        nominal_m = NOMINAL_INERTIA[bn]["mass"]
        print(f"\n  [{bn}]  true={true_m:.4f}, nominal={nominal_m:.4f}")
        print(f"    mass={m:+.4f} kg  (expect ≈{true_m - nominal_m:+.4f})")
        print(f"    pos=({px:+.4f}, {py:+.4f}, {pz:+.4f}) m")
        print(f"    size=({sx:.4f}, {sy:.4f}, {sz:.4f}) m")

    print("\nDone.")

    # ---- Inverse dynamics torque comparison ----
    # Use mj_inverse: same (q,dq,ddq) on each model → directly tests inverse dynamics.
    print("\nRunning inverse dynamics comparison …")

    true_model_c = mujoco.MjSpec.from_string(LEFT_ARM_XML).compile()
    tv_zero = {bn: [0.0] * 7 for bn in BODY_NAMES}
    tv_opt = {bn: list(params[f"balance_{bn}"].value) for bn in BODY_NAMES}

    def make_model(base_spec, bv):
        s = base_spec.copy()
        if bv is not None:
            apply_balances_to_spec(s, bv)
        return s.compile()

    init_model_c = make_model(nominal_base, tv_zero)
    opt_model_c = make_model(nominal_base, tv_opt)

    # Generate reference (q, dq) from true model under gentle excitation
    duration_vid = 2.0
    n_steps = int(duration_vid / true_model.opt.timestep)
    t_vid = np.arange(n_steps) * true_model.opt.timestep
    # Start from mid-range to avoid immediately hitting limits
    q_mid = 0.5 * (
        true_model_c.jnt_range[: true_model_c.nv, 0]
        + true_model_c.jnt_range[: true_model_c.nv, 1]
    )

    ctrl_ref = np.column_stack(
        [
            2.0 * np.sin(2 * np.pi * 0.5 * t_vid),
            1.5 * np.sin(2 * np.pi * 0.7 * t_vid + 0.5),
            1.2 * np.sin(2 * np.pi * 0.4 * t_vid + 1.0),
            0.8 * np.sin(2 * np.pi * 0.9 * t_vid + 1.5),
            0.5 * np.sin(2 * np.pi * 0.6 * t_vid + 2.0),
        ]
    )

    d_true = mujoco.MjData(true_model_c)
    d_true.qpos[:] = q_mid
    q_ref = np.zeros((n_steps, true_model_c.nv))
    dq_ref = np.zeros((n_steps, true_model_c.nv))
    ddq_ref = np.zeros((n_steps, true_model_c.nv))
    q_ref[0] = q_mid
    for k in range(n_steps - 1):
        d_true.ctrl[:] = ctrl_ref[k]
        mujoco.mj_step(true_model_c, d_true)
        q_ref[k + 1] = d_true.qpos.copy()
        dq_ref[k + 1] = d_true.qvel.copy()
        ddq_ref[k + 1] = d_true.qacc.copy()

    # Trim boundary steps + validate joint limits
    trim = max(n_steps // 20, 1)
    q_ref = q_ref[trim:-trim]
    dq_ref = dq_ref[trim:-trim]
    ddq_ref = ddq_ref[trim:-trim]
    t_vid = t_vid[trim:-trim]
    n_steps = len(t_vid)

    # Clip to limits and skip boundary steps where inverse dynamics is ill-defined
    q_min = true_model_c.jnt_range[: true_model_c.nv, 0] + 0.08
    q_max = true_model_c.jnt_range[: true_model_c.nv, 1] - 0.08
    at_limit = (q_ref <= q_min) | (q_ref >= q_max)
    q_ref = np.clip(q_ref, q_min, q_max)
    dq_ref[at_limit] = 0.0
    ddq_ref[at_limit] = 0.0
    keep = ~at_limit.any(axis=1)
    q_ref = q_ref[keep]
    dq_ref = dq_ref[keep]
    ddq_ref = ddq_ref[keep]
    t_vid = t_vid[keep]
    n_steps = len(t_vid)
    print(f"  Reference: {n_steps} steps, q=[{q_ref.min():.2f},{q_ref.max():.2f}]")

    # mj_inverse on each model
    models_id = [
        ("True", true_model_c),
        ("Nominal", init_model_c),
        ("Optimized", opt_model_c),
    ]
    torque_id = {}

    for label, model in models_id:
        data = mujoco.MjData(model)
        tau = np.zeros((n_steps, model.nv))
        for k in range(n_steps):
            data.qpos[:] = q_ref[k]
            data.qvel[:] = dq_ref[k]
            data.qacc[:] = ddq_ref[k]
            mujoco.mj_inverse(model, data)
            tau[k] = data.qfrc_inverse.copy()
        torque_id[label] = tau

    # Use common valid steps
    common_len = min(t.shape[0] for t in torque_id.values())
    for label in torque_id:
        torque_id[label] = torque_id[label][:common_len]
    t_tau = t_vid[:common_len]

    # ---- Plot: inverse dynamics torque per joint ----
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    colors = {"True": "green", "Nominal": "red", "Optimized": "blue"}
    for j in range(5):
        ax = axes[j]
        for label in ["True", "Nominal", "Optimized"]:
            ax.plot(
                t_tau,
                torque_id[label][:, j],
                color=colors[label],
                lw=1.0 if label != "True" else 1.5,
                ls="-" if label == "True" else ("--" if label == "Nominal" else "-."),
                label=label,
            )
        ax.set_ylabel(f"J{13 + j} torque (Nm)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")

    import time

    plt.suptitle("Inverse Dynamics Torque — Same (q,dq,ddq) on Each Model", y=1.01)
    plt.tight_layout()
    plt.savefig(f"torque_tracking_{time.time():.0f}.png", dpi=150)
    plt.close()
    print(f"  → saved torque_tracking_{time.time():.0f}.png")

    # ---- Torque RMSE bar chart ----
    rmse_t_init = np.sqrt(
        np.mean((torque_id["Nominal"] - torque_id["True"]) ** 2, axis=0)
    )
    rmse_t_opt = np.sqrt(
        np.mean((torque_id["Optimized"] - torque_id["True"]) ** 2, axis=0)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(5)
    w = 0.35
    ax.bar(x - w / 2, rmse_t_init, w, color="red", alpha=0.7, label="Nominal")
    ax.bar(x + w / 2, rmse_t_opt, w, color="blue", alpha=0.7, label="Optimized")
    for i in range(5):
        ax.text(
            i - w / 2,
            rmse_t_init[i] + 0.02,
            f"{rmse_t_init[i]:.2f}",
            ha="center",
            fontsize=7,
        )
        ax.text(
            i + w / 2,
            rmse_t_opt[i] + 0.02,
            f"{rmse_t_opt[i]:.2f}",
            ha="center",
            fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"J{13 + j}" for j in range(5)])
    ax.set_ylabel("Torque RMSE (Nm)")
    ax.set_title("Tracking Torque RMSE vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"torque_rmse_{time.time():.0f}.png", dpi=150)
    plt.close()
    print(f"  → saved torque_rmse_{time.time():.0f}.png")


if __name__ == "__main__":
    main()
