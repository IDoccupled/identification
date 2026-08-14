#!/usr/bin/env python3
"""
compare_torque.py — 把恢复出的实际运动轨迹（傅里叶系数）作用到机器人 URDF 上，
用 TargetLimbRegressor 算出"理论上需要的关节力矩"，与实测数据对比。

每个关节单独一张图，4 个子图分别对比：
    ① 位置：实际(joint_state) vs 还原
    ② 速度：实际(joint_state) vs 还原
    ③ 加速度：仅还原（解析求导，bag 无实测加速度）
    ④ 力矩：实际(joint_state) vs 还原(URDF+轨迹)

背景 / 约定
----------
* 轨迹来源：fourier_fit.py 从 bag 的位置反馈恢复出的 5 次傅里叶系数
  （trajectory_coefficients/recovered_*.yaml）。回放 time_coeffs 从该 YAML 的
  _meta 读取（fourier_fit 拟合时写入，源于 bag summary.json / --time-coeffs），
  也可 --time-coeffs 覆盖（如 0.75 -> f0 = 0.15 Hz）；不再从实测位置估计 f0。
* 轨迹拟合与实测对比：全部用 bag 的 hardware_joint_state（无丢包）——
  fourier_fit.py 用 joint_state 位置拟合恢复轨迹，本脚本也用 joint_state 的
  位置/速度/力矩做实测对比，二者同源，还原量直接按 phase=mod(t) 对齐，无需
  滞后补偿。
  注意：不要用 hardware_joint_command_feedback——它的 torque 是控制器 PD/指令
  力矩（左臂 stiffness=100、damping=1），数值偏大且 ~35% 丢包为 0，不是实际
  力矩；位置/速度也只是指令反馈而非实测状态。
* 测量时机器人并非站立：机体系下的重力向量为 [x,y,z]=[-9.712746, 0.390467, -1.393897]，
  腰关节 J12 角度 = 0.076485634 rad（固定，不参与辨识），这两个参数传给
  TargetLimbRegressor(gravity=..., waist_yaw_offset=...)。
* 理论力矩 = 惯性项(pin.rnea) + armature·a + damping·v + frictionloss·tanh(v·1e2)，
  与 regressor 内部 tau_aug 的口径一致（见 plot_unified_trajectory.py）。

用法（需先 source install/setup.bash，让 ament 找到 identification 包）：
    python -m identification.compare_torque
    python -m identification.compare_torque --bag 13_55_31 --yaml recovered_260813_131930.yaml
    python -m identification.compare_torque --save /tmp/torque_compare.png
        # 存图不弹窗；每个关节一张 -> /tmp/torque_compare_J13.png ... J17.png
    python -m identification.compare_torque -w 0:13.4     # 只看时间窗
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin
import yaml

from identification.fourier_trajectory import FourierTrajectory, TRAJ_PERIOD
from identification.target_limb_regressor import TargetLimbRegressor

PKG_DIR = Path(__file__).resolve().parent.parent
BAG_DATA = PKG_DIR / "bag_data"
COEFFS_DIR = PKG_DIR / "trajectory_coefficients"

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


def load_bag(bag_name: str):
    """从 <bag>/csv/hardware_joint_state.csv 读取实测数据。

    返回:
        t, q, v, tau : 时间轴 / 位置 / 速度 / 力矩（joint_state，t 归零，
                       形状均为 (N, 24)）
    """
    bag_dir = BAG_DATA / bag_name
    if not (bag_dir / "csv").is_dir():
        matches = sorted(BAG_DATA.glob(f"*{bag_name}*"))
        if len(matches) == 1:
            bag_dir = matches[0]
        elif len(matches) > 1:
            raise FileNotFoundError(
                f"短名 '{bag_name}' 匹配到多个 bag: {[m.name for m in matches]}"
            )
    csv_path = bag_dir / "csv" / f"{STATE_KEY}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    n_joints = sum(c.startswith("position_") for c in df.columns)
    t = df["t_s"].to_numpy(dtype=float)
    q = df[[f"position_{i}" for i in range(n_joints)]].to_numpy(dtype=float)
    v = df[[f"velocity_{i}" for i in range(n_joints)]].to_numpy(dtype=float)
    tau = df[[f"torque_{i}" for i in range(n_joints)]].to_numpy(dtype=float)
    t = t - t[0]  # 归零

    print(f"  数据: {csv_path}")
    print(
        f"  N={len(t)}  时长={t[-1]:.2f}s  dt~{np.median(np.diff(t)) * 1e3:.2f}ms  "
        f"关节数={q.shape[1]}"
    )
    return t, q, v, tau


def _yaml_time_coeffs(meta) -> float | None:
    """从轨迹 YAML 的 _meta 读取 time_coeffs（fourier_fit 拟合时写入）。

    兼容新格式的 f0_hz（= time_coeffs / TRAJ_PERIOD）。旧格式只有
    estimated_f0_hz（扫描噪声值），这里不使用，返回 None 由调用方报错。
    """
    tc = meta.get("time_coeffs")
    if tc is not None:
        return float(tc)
    f0 = meta.get("f0_hz")
    if f0 is not None:
        return float(f0) * TRAJ_PERIOD
    return None


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


def _setup_cjk_font():
    """尽量使用中文字体渲染，避免缺字；无 CJK 字体时静默回退默认字体。"""
    import matplotlib

    for f in [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
    ]:
        if f in matplotlib.font_manager.get_font_names():
            matplotlib.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


def plot_joint_compare(
    t,
    q_meas,
    v_meas,
    tau_meas,
    q_rec,
    v_rec,
    a_rec,
    tau_rec,
    joint_number,
    joint_name,
    out_png=None,
):
    """单个关节一张图：4 个子图 —— 位置 / 速度 / 加速度 / 力矩，实际 vs 还原。"""
    import matplotlib.pyplot as plt

    C_ACT = "C0"  # 实际
    C_REC = "C3"  # 还原

    fig, axs = plt.subplots(4, 1, figsize=(13, 13), sharex=True)

    # ① 位置：实际 vs 还原
    axs[0].plot(
        t,
        q_meas,
        lw=0.8,
        color=C_ACT,
        label=f"actual q_{joint_number} (joint_state)",
    )
    axs[0].plot(
        t, q_rec, lw=1.4, color=C_REC, alpha=0.5, label=f"recovered q_{joint_number}"
    )
    axs[0].set_ylabel("q (rad)")
    axs[0].set_title("Position", loc="left", fontsize=11)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    # ② 速度：实际 vs 还原
    axs[1].plot(
        t,
        v_meas,
        lw=0.8,
        color=C_ACT,
        label=f"actual v_{joint_number} (joint_state)",
    )
    axs[1].plot(
        t, v_rec, lw=1.4, color=C_REC, alpha=0.5, label=f"recovered v_{joint_number}"
    )
    axs[1].set_ylabel("v (rad/s)")
    axs[1].set_title("Velocity", loc="left", fontsize=11)
    axs[1].legend(loc="upper right", fontsize=8)
    axs[1].grid(alpha=0.3)

    # ③ 加速度：仅还原（解析求导，bag 无实测加速度）
    axs[2].plot(
        t,
        a_rec,
        lw=1.2,
        color=C_REC,
        label=f"recovered a_{joint_number} (analytic)",
    )
    axs[2].set_ylabel("a (rad/s²)")
    axs[2].set_title(
        "Acceleration (recovered only, no measured)", loc="left", fontsize=11
    )
    axs[2].legend(loc="upper right", fontsize=8)
    axs[2].grid(alpha=0.3)

    # ④ 力矩：实际 vs 还原
    axs[3].plot(
        t,
        tau_meas,
        lw=0.8,
        color=C_ACT,
        label=f"actual tau_{joint_number} (joint_state)",
    )
    axs[3].plot(
        t,
        tau_rec,
        lw=1.4,
        alpha=0.5,
        color=C_REC,
        label=f"recovered tau_{joint_number} (URDF + recovered traj)",
    )
    axs[3].set_ylabel("tau (Nm)")
    axs[3].set_title("Torque", loc="left", fontsize=11)
    axs[3].legend(loc="upper right", fontsize=8)
    axs[3].grid(alpha=0.3)

    axs[3].set_xlabel("t (s)")
    fig.suptitle(
        f"{joint_name} (joint {joint_number}) — actual vs recovered\n"
        f"gravity={GRAVITY.tolist()}, waist={WAIST_YAW_OFFSET:.4f} rad",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out_png:
        fig.savefig(out_png, dpi=150)
        print(f"对比图已保存: {out_png}")
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
        help="bag_data 下的 bag 名（默认跟随 YAML _meta.source_bag）",
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
        help="回放时间倍率（默认从 YAML _meta 读取，可覆盖；例如 0.75）",
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
    ap.add_argument(
        "--save",
        default=None,
        help="保存对比图到该路径（每个关节一张，自动加 _J# 后缀）",
    )
    args = ap.parse_args()

    yaml_name = args.yaml or _latest_recovered_yaml()
    meta = _yaml_meta(yaml_name)
    meta_bag = meta.get("source_bag")

    # 0) bag 跟随 YAML _meta.source_bag（fourier_fit 保存的来源），避免轨迹
    #    系数和实测数据来源不一致（例如 57_28 的 YAML 配上 13_55_31 的 bag）；
    #    --bag 显式指定时覆盖。time_coeffs 默认从 bag summary.json 读取
    #    （extract 时写入），也可 --time-coeffs 覆盖，f0 = time_coeffs / TRAJ_PERIOD。
    bag_name = args.bag or (meta_bag or DEFAULT_BAG)

    # 1) 实测数据（位置/速度/力矩，全部 joint_state）
    t, q_meas, v_meas, tau_meas = load_bag(bag_name)

    # 2) time_coeffs 只从 YAML _meta 读取（fit 时写入），可 --time-coeffs 覆盖；不再读 bag summary.json
    time_coeffs = args.time_coeffs
    tc_src = "手动 --time-coeffs" if time_coeffs is not None else "YAML _meta"
    if time_coeffs is None:
        time_coeffs = _yaml_time_coeffs(meta)
    if time_coeffs is None:
        raise SystemExit(
            f"YAML {yaml_name} 的 _meta 里没有 time_coeffs（旧格式只有 estimated_f0_hz），"
            "请用 --time-coeffs 指定，或重新用 fourier_fit.py 拟合"
        )
    time_coeffs = float(time_coeffs)
    f0 = time_coeffs / TRAJ_PERIOD

    print(
        f"\n[对齐] bag={bag_name}  f0={f0:.2f} Hz -> time_coeffs={time_coeffs:.2f}"
        f"（来源: {tc_src}）"
    )

    # 3) 恢复出的一周期理论轨迹
    ft = FourierTrajectory(
        dim=len(args.joints), sample_rate=args.sample_rate, time_coeffs=time_coeffs
    )
    q_th, v_th, a_th = ft.generate_trajectory_from_yaml(yaml_name)
    period = ft.duration
    print(
        f"\n轨迹: {yaml_name}  time_coeffs={time_coeffs:.2f}  "
        f"f0={f0:.2f} Hz  周期={period:.4f}s  采样点={q_th.shape[1]}"
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

    # 5) 一周期理论量，按相位折叠到实测时间轴上
    #    轨迹拟合(joint_state 位置) 与实测(joint_state) 同源，还原量统一按
    #    phase=mod(t, period) 对齐，无需滞后补偿。
    tau_th = compute_theoretical_torque(reg, q_th, v_th, a_th)  # (dof, N_th)
    phase = np.mod(t, period)
    q_rec = np.stack(
        [np.interp(phase, ft.t_array, q_th[i], period=period) for i in range(reg.dof)],
        axis=1,
    )
    v_rec = np.stack(
        [np.interp(phase, ft.t_array, v_th[i], period=period) for i in range(reg.dof)],
        axis=1,
    )
    a_rec = np.stack(
        [np.interp(phase, ft.t_array, a_th[i], period=period) for i in range(reg.dof)],
        axis=1,
    )
    tau_rec = np.stack(
        [
            np.interp(phase, ft.t_array, tau_th[i], period=period)
            for i in range(reg.dof)
        ],
        axis=1,
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
    q_meas = q_meas[mask_t]
    v_meas = v_meas[mask_t]
    tau_meas_plot = tau_meas_plot[mask_t]
    q_rec = q_rec[mask_t]
    v_rec = v_rec[mask_t]
    a_rec = a_rec[mask_t]
    tau_rec = tau_rec[mask_t]

    # 8) 每个关节单独一张 4 子图（位置/速度/加速度/力矩），收集后一起 show
    joint_names = JOINT_NAMES[: len(args.joints)]
    figs = []
    for i, (j, name) in enumerate(zip(args.joints, joint_names)):
        out_path = None
        if args.save:
            p = Path(args.save)
            out_path = str(p.with_name(f"{p.stem}_J{j}{p.suffix}"))
        figs.append(
            plot_joint_compare(
                t,
                q_meas[:, j],
                v_meas[:, j],
                tau_meas_plot[:, j],
                q_rec[:, i],
                v_rec[:, i],
                a_rec[:, i],
                tau_rec[:, i],
                joint_number=j,
                joint_name=name,
                out_png=out_path,
            )
        )
    if figs:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
