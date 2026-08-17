#!/usr/bin/env python3
"""
plot_residual.py — 直接从已保存的轨迹系数 YAML 画出与 fourier_fit --plot 相同的
"原始 vs 理论 q/v + 残差小图"，不需要重新跑一遍拟合。

流程：确定 bag → 从轨迹 YAML 的 _meta 读取 time_coeffs（fourier_fit 拟合时写入，
可 --time-coeffs 覆盖）→ fourier_trajectory 从 YAML 生成理论 q/v/a → 按相位折叠
到实测时间轴 → 每关节一张 4 子图（位置、速度、位置残差、速度残差）。
（time_coeffs 只在 fit 时读 bag 的 summary.json，这里只用 YAML 里存的。）

用法（需先 source install/setup.bash）：
    python -m identification.plot_residual
    python -m identification.plot_residual --yaml recovered_260813_173014.yaml
    python -m identification.plot_residual --yaml X.yaml --bag 13_57_28
    python -m identification.plot_residual --yaml X.yaml --time-coeffs 0.75 -w 0:15 --save /tmp/resid.png
        # --save 时每关节一张 -> /tmp/resid_J13.png ... J17.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from identification.fourier_trajectory import FourierTrajectory, TRAJ_PERIOD

PKG_DIR = Path(__file__).resolve().parent.parent
BAG_DATA = PKG_DIR / "bag_data"
COEFFS_DIR = PKG_DIR / "trajectory_coefficients"

TOPIC_KEY = "hardware_joint_state"  # 与 fourier_fit.py 一致（实测状态）
DEFAULT_JOINTS = [13, 14, 15, 16, 17]
JOINT_NAMES = [
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
]


def _yaml_meta(yaml_name: str) -> dict:
    """读取 trajectory_coefficients 下 YAML 的 _meta（fourier_fit 保存的备忘）。"""
    path = COEFFS_DIR / yaml_name
    if not path.is_file():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("_meta", {}) if isinstance(data, dict) else {}


def _latest_recovered_yaml() -> str:
    matches = sorted(COEFFS_DIR.glob("recovered_*.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"trajectory_coefficients 下没有 recovered_*.yaml: {COEFFS_DIR}"
        )
    return matches[-1].name


def load_bag(bag_name: str):
    """从 <bag>/csv/hardware_joint_state.csv 读取 (t, position, velocity)，短名自动匹配。"""
    bag_dir = BAG_DATA / bag_name
    if not (bag_dir / "csv").is_dir():
        matches = sorted(BAG_DATA.glob(f"*{bag_name}*"))
        if len(matches) == 1:
            bag_dir = matches[0]
        elif len(matches) > 1:
            raise FileNotFoundError(
                f"短名 '{bag_name}' 匹配到多个 bag: {[m.name for m in matches]}"
            )
    csv_path = bag_dir / "csv" / f"{TOPIC_KEY}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    n_joints = sum(c.startswith("position_") for c in df.columns)
    t = df["t_s"].to_numpy(dtype=float)
    q = df[[f"position_{i}" for i in range(n_joints)]].to_numpy(dtype=float)
    v = df[[f"velocity_{i}" for i in range(n_joints)]].to_numpy(dtype=float)
    t = t - t[0]  # 归零
    print(f"  数据: {csv_path}")
    print(
        f"  N={len(t)}  时长={t[-1]:.2f}s  dt~{np.median(np.diff(t)) * 1e3:.2f}ms  "
        f"关节数={q.shape[1]}"
    )
    return t, q, v


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


def _plot_residual_spectrum(ax, t, rq, rv, f0):
    """在 ax 上画位置/速度残差的频谱（|FFT| vs Hz），竖虚线标出 f0 的整数倍。"""
    dt = np.median(np.diff(t))
    freq = np.fft.rfftfreq(len(rq), dt)
    mag_q = np.maximum(np.abs(np.fft.rfft(rq)), 1e-15)
    mag_v = np.maximum(np.abs(np.fft.rfft(rv)), 1e-15)
    ax.semilogy(freq, mag_q, lw=0.9, color="C2", label="q residual spectrum")
    ax.semilogy(freq, mag_v, lw=0.9, color="C4", label="v residual spectrum")
    fmax = freq.max()
    k = 1
    while k * f0 <= fmax and k <= 25:
        ax.axvline(k * f0, color="gray", lw=0.6, ls="--", alpha=0.6)
        k += 1
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|FFT(residual)|")
    ax.set_title("Residual Spectrum (q & v)", loc="left", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def plot_joint_residual(
    t,
    q_raw,
    v_raw,
    q_th,
    v_th,
    t_th,
    period,
    f0,
    tc,
    joint_number,
    joint_name,
    out_png=None,
):
    """单个关节一张图：5 子图 —— 位置/速度、位置/速度残差、残差频谱。"""
    import matplotlib.pyplot as plt

    C_ACT = "C0"  # 实际
    C_REC = "C3"  # 还原

    phase = np.mod(t, period)
    q_t = np.interp(phase, t_th, q_th, period=period)
    v_t = np.interp(phase, t_th, v_th, period=period)
    rq, rv = q_raw - q_t, v_raw - v_t

    fig, axs = plt.subplots(
        5,
        1,
        figsize=(12, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1.5, 1.5, 3.5]},
    )
    axs[4].get_shared_x_axes().remove(axs[4])  # 频谱子图不共享时间轴

    # ① 位置：原始 vs 理论
    axs[0].plot(
        t,
        q_raw,
        lw=0.5,
        alpha=0.7,
        color=C_ACT,
        label=f"actual q_{joint_number} (joint_state)",
    )
    axs[0].plot(t, q_t, lw=1.2, color=C_REC, label="recovered q (fourier_trajectory)")
    axs[0].set_ylabel("q (rad)")
    axs[0].set_title(
        f"Position — {joint_name} (joint {joint_number})  f0={f0:.2f} Hz  "
        f"time_coeffs={tc:.2f}  q resid RMS {np.sqrt(np.mean(rq**2)):.5f} rad",
        loc="left",
        fontsize=10,
    )
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    # ② 速度：原始 vs 理论
    axs[1].plot(
        t, v_raw, lw=0.5, alpha=0.7, color=C_ACT, label=f"actual v_{joint_number}"
    )
    axs[1].plot(t, v_t, lw=1.2, color=C_REC, alpha=0.8, label="recovered v")
    axs[1].set_ylabel("v (rad/s)")
    axs[1].set_title(
        f"Velocity — v resid RMS {np.sqrt(np.mean(rv**2)):.4f} = "
        f"{100 * np.sqrt(np.mean(rv**2)) / np.std(v_raw):.2f}% of v std",
        loc="left",
        fontsize=10,
    )
    axs[1].legend(loc="upper right", fontsize=8)
    axs[1].grid(alpha=0.3)

    # ③ 位置残差
    axs[2].plot(t, rq, lw=0.6, color="C2")
    axs[2].set_ylabel("q residual (rad)")
    axs[2].axhline(0, color="gray", lw=0.6)
    axs[2].grid(alpha=0.3)

    # ④ 速度残差
    axs[3].plot(t, rv, lw=0.6, color="C4")
    axs[3].set_ylabel("v residual (rad/s)")
    axs[3].set_xlabel("t (s)")
    axs[3].axhline(0, color="gray", lw=0.6)
    axs[3].grid(alpha=0.3)

    # ⑤ 残差频谱
    _plot_residual_spectrum(axs[4], t, rq, rv, f0)

    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
        print(f"残差图已保存: {out_png}")
    return fig


def main():
    ap = argparse.ArgumentParser(
        description="从 YAML 直接画 原始 vs 还原 q/v + 残差（无需重新拟合）"
    )
    ap.add_argument(
        "--yaml",
        default=None,
        help="trajectory_coefficients 下的 YAML（默认最新 recovered_*.yaml）",
    )
    ap.add_argument(
        "--bag",
        default=None,
        help="bag_data 下的 bag 名（默认跟随 YAML _meta.source_bag）",
    )
    ap.add_argument(
        "--time-coeffs",
        type=float,
        default=None,
        help="时间系数（默认从 YAML _meta 读取；fourier_fit 拟合时写入）",
    )
    ap.add_argument(
        "--joints",
        nargs="+",
        type=int,
        default=DEFAULT_JOINTS,
        help="YAML joint_i 对应的机器人关节号",
    )
    ap.add_argument(
        "--sample-rate", type=float, default=500.0, help="生成理论轨迹的采样率 Hz"
    )
    ap.add_argument(
        "-w",
        "--twin",
        default=None,
        metavar="START:END",
        help="只画时间窗，如 '0:13.4'（两端可省略）",
    )
    ap.add_argument(
        "--save",
        default=None,
        help="保存到该路径（每关节一张，自动加 _J# 后缀）",
    )
    args = ap.parse_args()

    yaml_name = args.yaml or _latest_recovered_yaml()
    meta = _yaml_meta(yaml_name)
    bag_name = args.bag or meta.get("source_bag")
    if not bag_name:
        raise ValueError(f"YAML {yaml_name} 没有 _meta.source_bag，请用 --bag 指定")

    t, q, v = load_bag(bag_name)

    # time_coeffs 只从 YAML _meta 读取（fit 时写入），可 --time-coeffs 覆盖；不再读 bag summary.json
    tc = args.time_coeffs
    if tc is None:
        tc = _yaml_time_coeffs(meta)
    if tc is None:
        raise SystemExit(
            f"YAML {yaml_name} 的 _meta 里没有 time_coeffs（旧格式只有 estimated_f0_hz），"
            "请用 --time-coeffs 指定，或重新用 fourier_fit.py 拟合"
        )
    tc = float(tc)
    f0 = tc / TRAJ_PERIOD

    print(
        f"\nYAML: {yaml_name}   bag: {bag_name}   f0={f0:.2f} Hz -> time_coeffs={tc:.2f}"
    )

    traj = FourierTrajectory(
        dim=len(args.joints), sample_rate=args.sample_rate, time_coeffs=tc
    )
    q_th, v_th, _ = traj.generate_trajectory_from_yaml(yaml_name)
    t_th = traj.t_array
    period = traj.duration
    print(f"  理论轨迹: 采样点={q_th.shape[1]}  周期={period:.4f}s")

    # 时间窗裁剪
    t0w, t1w = _parse_window(args.twin, t[-1])
    mask_t = (t >= t0w) & (t <= t1w)
    if not np.any(mask_t):
        raise ValueError(f"时间窗 [{t0w}, {t1w}] 内没有数据点")
    t, q, v = t[mask_t], q[mask_t], v[mask_t]

    joint_names = JOINT_NAMES[: len(args.joints)]
    figs = []
    for i, (j, name) in enumerate(zip(args.joints, joint_names)):
        out_path = None
        if args.save:
            p = Path(args.save)
            out_path = str(p.with_name(f"{p.stem}_J{j}{p.suffix}"))
        figs.append(
            plot_joint_residual(
                t,
                q[:, j],
                v[:, j],
                q_th[i],
                v_th[i],
                t_th,
                period,
                f0,
                tc,
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
