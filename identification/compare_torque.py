#!/usr/bin/env python3
"""
compare_torque.py — 把恢复出的实际运动轨迹（傅里叶系数）作用到机器人 URDF 上，
用 TargetLimbRegressor 算出"理论上需要的关节力矩"，与实测力矩画在同一张图里对比。

背景 / 约定
----------
* 轨迹来源：fourier_fit.py 从 bag 的位置反馈恢复出的 5 次傅里叶系数
  （trajectory_coefficients/recovered_*.yaml）。回放 time_coeffs 默认从实测
  位置自动估计（实际 f0 ≈ 0.150025 Hz -> time_coeffs ≈ 0.750125，不要手动拍死
  0.75），也可用 --time-coeffs 手动覆盖。
* 实测力矩：bag 的 hardware_joint_state.torque（实际关节力矩，无丢包）。
  注意：不要用 hardware_joint_command_feedback.torque——那是控制器 PD/指令
  力矩（左臂 stiffness=100、damping=1），数值偏大且 ~35% 丢包为 0，不是实际
  关节力矩。joint_state 相对指令反馈约有 ~22 ms 滞后（控制环延迟），画图时
  已按相位自动对齐。
* 测量时机器人并非站立：机体系下的重力向量为 [x,y,z]=[-9.712746, 0.390467, -1.393897]，
  腰关节 J12 角度 = 0.076485634 rad（固定，不参与辨识），这两个参数传给
  TargetLimbRegressor(gravity=..., waist_yaw_offset=...)。
* 理论力矩 = 惯性项(pin.rnea) + armature·a + damping·v + frictionloss·tanh(v·1e2)，
  与 regressor 内部 tau_aug 的口径一致（见 plot_unified_trajectory.py）。

用法（需先 source install/setup.bash，让 ament 找到 identification 包）：
    python -m identification.compare_torque
    python -m identification.compare_torque --bag 13_55_31 --yaml recovered_260813_131930.yaml
    python -m identification.compare_torque --save /tmp/torque_compare.png   # 存图不弹窗
"""

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

from identification.fourier_trajectory import FourierTrajectory, TRAJ_PERIOD
from identification.target_limb_regressor import TargetLimbRegressor

PKG_DIR = Path(__file__).resolve().parent.parent
EXTRACTED = PKG_DIR / "extracted"
COEFFS_DIR = PKG_DIR / "trajectory_coefficients"

CMD_KEY = "hardware_joint_command_feedback"
STATE_KEY = "hardware_joint_state"
DEFAULT_BAG = "13_55_31"
DEFAULT_YAML = "recovered_260813_131930.yaml"
DEFAULT_JOINTS = [13, 14, 15, 16, 17]
JOINT_NAMES = [
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
]

# 测量时机器人并非站立：机体系下的重力向量（长度≈9.82 m/s²）。
GRAVITY = np.array([-9.712746, 0.390467, -1.393897])
# 腰关节 J12 角度（rad），测量时固定在此值。
WAIST_YAW_OFFSET = 0.076485634

ZERO_TOL = 1e-9  # 兜底：低于该绝对值视为无效样本，画图时 mask


def _estimate_state_lag(t_cmd, q_cmd, t_state, q_state, joint=13, max_lag_s=0.05):
    """用左臂关节 13 的位置交叉相关，估计 joint_state 相对指令反馈的滞后（秒）。

    joint_state 是实测状态，滞后于命令反馈约 ~22 ms。理论轨迹按 q_cmd 拟合，
    实测力矩对应的是滞后 ~22ms 后的实际状态，因此画图时按该滞后对齐相位。
    """
    j = min(joint, q_cmd.shape[1] - 1)
    q_cmd_i = np.interp(t_state, t_cmd, q_cmd[:, j])
    dt = np.median(np.diff(t_state))
    n_max = int(max_lag_s / dt)
    n = len(t_state)
    lo, hi = n_max, n - n_max
    best_n, best_c = 0, -1.0
    for k in range(-n_max, n_max + 1):
        c = np.corrcoef(np.roll(q_cmd_i, k)[lo:hi], q_state[lo:hi, j])[0, 1]
        if c > best_c:
            best_c, best_n = c, k
    return best_n * dt


def load_bag(bag_name: str):
    """从 extracted/<bag>/data.npz 读取实测数据。

    返回:
        t, q, v  : 时间轴 / 位置 / 速度（joint_command_feedback，t 归零）
        tau      : 实测关节力矩 (N,24) = joint_state.torque 插值到 t 上
        lag      : joint_state 相对指令反馈的滞后秒数（相位对齐用）
    """
    bag_dir = EXTRACTED / bag_name
    if not (bag_dir / "data.npz").is_file():
        matches = sorted(EXTRACTED.glob(f"*{bag_name}*"))
        if len(matches) == 1:
            bag_dir = matches[0]
        elif len(matches) > 1:
            raise FileNotFoundError(
                f"短名 '{bag_name}' 匹配到多个 bag: {[m.name for m in matches]}"
            )
    npz_path = bag_dir / "data.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"data.npz not found: {npz_path}")
    d = np.load(npz_path)
    t = d[f"{CMD_KEY}.t_s"].astype(float)
    q = d[f"{CMD_KEY}.position"].astype(float)  # (N, 24)
    v = d[f"{CMD_KEY}.velocity"].astype(float)  # (N, 24)
    t = t - t[0]  # 归零

    # 实测力矩用 joint_state.torque（实际关节力矩），插值到 t 上。
    t_state = d[f"{STATE_KEY}.t_s"].astype(float)
    t_state = t_state - t_state[0]
    q_state = d[f"{STATE_KEY}.position"].astype(float)
    tau_state = d[f"{STATE_KEY}.torque"].astype(float)
    tau = np.column_stack(
        [np.interp(t, t_state, tau_state[:, j]) for j in range(tau_state.shape[1])]
    )
    lag = _estimate_state_lag(t, q, t_state, q_state)

    print(f"  数据: {npz_path}")
    print(
        f"  N={len(t)}  时长={t[-1]:.2f}s  dt~{np.median(np.diff(t)) * 1e3:.2f}ms  "
        f"关节数={q.shape[1]}  joint_state 滞后≈{lag * 1e3:.0f}ms"
    )
    return t, q, v, tau, lag


def compute_theoretical_torque(reg, q_th, v_th, a_th) -> np.ndarray:
    """在恢复出的一周期轨迹上算理论力矩 (dof, N)。

    与 TargetLimbRegressor 内部 tau_aug 口径一致：
        tau = tau_inertia(rnea, 含重力) + armature·a + damping·v + friction·tanh(v·1e2)
    """
    dof = reg.dof
    n = q_th.shape[1]
    tau_th = np.zeros((dof, n))
    for k in range(n):
        q_full, v_full, a_full = reg.state_size_check_and_form(
            q_th[:, k], v_th[:, k], a_th[:, k]
        )
        tau_rnea = pin.rnea(reg.model, reg.data, q_full, v_full, a_full)[
            reg.group_to_identify
        ]
        for i in range(dof):
            info = reg.target_joint_infos[i]
            tau_th[i, k] = (
                tau_rnea[i]
                + info["armature"] * a_th[i, k]
                + info["damping"] * v_th[i, k]
                + info["friction"] * np.tanh(v_th[i, k] * 1e2)
            )
    return tau_th


def plot_compare(t_raw, tau_meas, tau_th_folded, joints, joint_names, out_png=None):
    """一张图 5 个子图（每个左臂关节一行）：实测 vs 理论力矩。"""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(
        len(joints), 1, figsize=(13, 12), sharex=True, squeeze=False
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(joints)))

    for row, (j, name) in enumerate(zip(joints, joint_names)):
        ax = axs[row][0]
        ax.plot(
            t_raw,
            tau_meas[:, j],
            lw=0.8,
            color=colors[row],
            alpha=0.85,
            label=f"measured tau_{j} (joint_state, actual)",
        )
        ax.plot(
            t_raw,
            tau_th_folded[:, row],
            lw=1.4,
            color="k",
            label="theoretical (URDF + recovered traj)",
        )
        ax.set_ylabel(f"τ_{j} (Nm)")
        ax.set_title(f"{name}  (joint {j})", fontsize=11, loc="left")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axs[-1][0].set_xlabel("t (s)")

    fig.suptitle(
        "Measured (joint_state) vs theoretical (URDF) joint torque (left arm) — "
        f"gravity={GRAVITY.tolist()}, waist={WAIST_YAW_OFFSET:.4f} rad",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_png:
        fig.savefig(out_png, dpi=150)
        print(f"对比图已保存: {out_png}")
    plt.show()
    return fig


def _latest_recovered_yaml() -> str:
    matches = sorted(COEFFS_DIR.glob("recovered_*.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"trajectory_coefficients 下没有 recovered_*.yaml: {COEFFS_DIR}"
        )
    return matches[-1].name


def _yaml_meta(yaml_name: str) -> dict:
    """读取 trajectory_coefficients 下 YAML 的 _meta（fourier_fit 保存的备忘）。"""
    path = COEFFS_DIR / yaml_name
    if not path.is_file():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("_meta", {}) if isinstance(data, dict) else {}


def _estimate_f0_from_data(t, q, joint=13) -> float:
    """从实测位置用 fourier_fit 的方式（FFT 粗扫 + 最小二乘细扫）估计基频 f0 (Hz)。"""
    from identification.fourier_fit import estimate_f0

    j = min(joint, q.shape[1] - 1)
    return float(estimate_f0(t, q[:, j]))


def _parse_window(s: str | None, t_end: float) -> tuple[float, float]:
    """解析 '-w/--twin' 时间窗 'start:end'（可省略任一端）-> (t0, t1)。None=全程。"""
    if s is None:
        return 0.0, t_end
    parts = s.split(":")
    if len(parts) > 2:
        raise ValueError(f"时间窗格式应为 'start:end'，得到: '{s}'")
    t0 = float(parts[0]) if parts[0].strip() else 0.0
    t1 = float(parts[1].strip()) if len(parts) == 2 and parts[1].strip() else t_end
    return t0, t1


def main():
    ap = argparse.ArgumentParser(description="实测 vs 理论（URDF）关节力矩对比图")
    ap.add_argument(
        "--bag",
        default=None,
        help="extracted 下的 bag 名（默认跟随 YAML _meta.source_bag）",
    )
    ap.add_argument(
        "--yaml",
        default=None,
        help="trajectory_coefficients 下的轨迹系数 YAML（默认取最新的 recovered_*.yaml）",
    )
    ap.add_argument(
        "--time-coeffs",
        type=float,
        default=None,
        help="回放时间倍率；默认从实测轨迹自动估计（f0*TRAJ_PERIOD，例如 0.750125）",
    )
    ap.add_argument(
        "--sample-rate", type=float, default=500.0, help="生成理论轨迹的采样率 Hz"
    )
    ap.add_argument(
        "--joints", nargs="+", type=int, default=DEFAULT_JOINTS, help="左臂关节号"
    )
    ap.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        default=GRAVITY.tolist(),
        metavar=("X", "Y", "Z"),
        help="机体系重力向量",
    )
    ap.add_argument(
        "--waist-offset",
        type=float,
        default=WAIST_YAW_OFFSET,
        help="腰关节 J12 角度 rad",
    )
    ap.add_argument(
        "-w",
        "--twin",
        default=None,
        metavar="START:END",
        help="只画时间窗，如 '0:13.4'（两端可省略，默认全程）",
    )
    ap.add_argument("--save", default=None, help="保存对比图到该路径（默认弹窗显示）")
    args = ap.parse_args()

    yaml_name = args.yaml or _latest_recovered_yaml()
    meta = _yaml_meta(yaml_name)
    meta_bag = meta.get("source_bag")
    meta_f0 = meta.get("estimated_f0_hz")

    # 0) bag 与回放 time_coeffs 优先跟随 YAML _meta（fourier_fit 保存的来源
    #    bag 与基频 f0），避免轨迹系数和实测数据来源不一致（例如 57_28 的
    #    YAML 配上 13_55_31 的 bag）。--bag / --time-coeffs 显式指定时覆盖。
    bag_name = args.bag or (meta_bag or DEFAULT_BAG)
    if args.time_coeffs is not None:
        time_coeffs = float(args.time_coeffs)
        f0 = time_coeffs / TRAJ_PERIOD
        tc_src = "手动 --time-coeffs"
    elif meta_f0:
        f0 = float(meta_f0)
        time_coeffs = f0 * TRAJ_PERIOD
        tc_src = "YAML _meta"
    else:
        time_coeffs = None
        f0 = None

    # 1) 实测数据（位置/速度/力矩）
    t, q_meas, v_meas, tau_meas, state_lag = load_bag(bag_name)

    # 2) 回放 time_coeffs：YAML 没有 _meta.f0 时才从实测位置估计。
    if time_coeffs is None:
        f0 = _estimate_f0_from_data(t, q_meas, args.joints[0])
        time_coeffs = f0 * TRAJ_PERIOD
        tc_src = "实测轨迹估计"
    print(
        f"\n[对齐] bag={bag_name}  f0={f0:.6f} Hz -> time_coeffs={time_coeffs:.6f}"
        f"（来源: {tc_src}）"
    )

    # 3) 恢复出的一周期理论轨迹
    ft = FourierTrajectory(
        dim=len(args.joints), sample_rate=args.sample_rate, time_coeffs=time_coeffs
    )
    q_th, v_th, a_th = ft.generate_trajectory_from_yaml(yaml_name)
    period = ft.duration
    print(
        f"\n轨迹: {yaml_name}  time_coeffs={time_coeffs:.6f}  "
        f"f0={f0:.6f} Hz  周期={period:.4f}s  采样点={q_th.shape[1]}"
    )

    # 4) TargetLimbRegressor：非站立重力 + 固定腰关节
    reg = TargetLimbRegressor(
        group_to_identify="left_arm",
        gravity=np.asarray(args.gravity, dtype=float),
        waist_yaw_offset=float(args.waist_offset),
        print_info=False,
    )
    print(
        f"重力向量: {reg.model.gravity.linear}   腰关节偏移: {reg.waist_yaw_offset:.6f} rad"
    )

    # 5) 一周期理论力矩，再按相位折叠到实测时间轴上
    #    joint_state 滞后指令反馈约 state_lag，实测力矩对应的是滞后后的实际
    #    状态，因此相位里减掉该滞后再对齐。
    tau_th = compute_theoretical_torque(reg, q_th, v_th, a_th)  # (dof, N_th)
    phase = np.mod(t - state_lag, period)
    tau_th_folded = np.stack(
        [
            np.interp(phase, ft.t_array, tau_th[i], period=period)
            for i in range(reg.dof)
        ],
        axis=1,  # (N_raw, dof)
    )

    # 6) 兜底：接近 0 的力矩样本 -> NaN（joint_state 正常无丢包，一般不会触发）
    n_zero = int(np.sum(np.abs(tau_meas[:, args.joints]) < ZERO_TOL))
    tau_meas_plot = tau_meas.copy()
    tau_meas_plot[np.abs(tau_meas_plot) < ZERO_TOL] = np.nan
    print(f"\n实测力矩(joint_state)接近0被mask样本数: {n_zero}")

    # 7) 时间窗裁剪（-w/--twin，格式 'start:end'）
    t0, t1 = _parse_window(args.twin, t[-1])
    mask_t = (t >= t0) & (t <= t1)
    if not np.any(mask_t):
        raise ValueError(f"时间窗 [{t0}, {t1}] 内没有数据点")
    t = t[mask_t]
    tau_meas_plot = tau_meas_plot[mask_t]
    tau_th_folded = tau_th_folded[mask_t]

    # 8) 画图
    plot_compare(
        t,
        tau_meas_plot,
        tau_th_folded,
        joints=args.joints,
        joint_names=JOINT_NAMES[: len(args.joints)],
        out_png=args.save,
    )


if __name__ == "__main__":
    main()
