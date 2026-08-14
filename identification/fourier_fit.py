#!/usr/bin/env python3
"""
fourier_fit.py — 只用位置(编码器)最小二乘拟合 5 次傅里叶轨迹参数，
                然后由同一组参数解析求导得到理论速度 / 加速度，
                与实测速度比对，验证编码器位置 / 速度的一致性。

为什么只拟合位置？
-----------------
采集的关节反馈只有 position / velocity，没有 acceleration。
轨迹本身是 5 次谐波傅里叶形式，编码器对位置的测量几乎无噪声，
因此位置是"信噪比最高"的量。先用最小二乘从位置恢复出
    q(t) = q0 + Σ_n [ s_n·sin(n·ω0·t) + c_n·cos(n·ω0·t) ]   (n=1..5)
再由同一组系数解析求导：
    v(t) = Σ_n n·ω0·[ s_n·cos(n·ω0·t) − c_n·sin(n·ω0·t) ]
    a(t) = −Σ_n (n·ω0)²·[ s_n·sin(n·ω0·t) + c_n·cos(n·ω0·t) ]
得到的 v、a 与拟合用的位置完全自洽，且天然无噪声放大的问题。

关键结论（已在本包实际数据上验证，bag 13_55_31，关节 13）：
    * f0 直接由录制时的 time_coeffs 决定：f0 = time_coeffs / TRAJ_PERIOD
      （该 bag 用 time_coeffs=0.75 跑，周期 = 5/0.75 = 6.667 s -> f0 = 0.15 Hz）。
      因此不再从数据估计 f0：time_coeffs 在 extract_bag_data.py 提取时写入
      <bag>/summary.json，本脚本默认从 summary.json 读取，也可 --time-coeffs 覆盖，
      避免扫描 f0 引入的人为误差。
    * 位置 5 谐波拟合可解释 ~99.996% 方差，残差 RMS ~0.007 rad；
    * 由位置系数解析求导的理论速度 vs 实测速度残差 ~0.83% (v 的 std)。

本脚本流程：拟合 -> 保存 YAML -> 可选本地画图对比（--plot）。
画图时用 fourier_trajectory 从保存的 YAML 生成理论 q/v/a（与真实回放同一代码路径），
叠加 bag 原始位置 / 速度并画残差小图；理论轨迹按手动指定的 time_coeffs 对齐录制频率。

用法（--time-coeffs 可省略，默认从 bag 的 summary.json 读取；在包根目录 src/identification 下运行）：
    python -m identification.fourier_fit --bag 13_55_31
    python -m identification.fourier_fit --bag 13_55_31 --joints 13 14 15
    python -m identification.fourier_fit --bag 13_55_31 --time-coeffs 0.75 --plot   # 手动覆盖 + 画图

输出：
    拟合结果按 fourier_trajectory.py 的 YAML 格式保存到
    trajectory_coefficients/ 目录（joint_0..joint_{k-1}，含 a / b / q0）。
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from identification.fourier_trajectory import FourierTrajectory, TRAJ_PERIOD

N_HARMONICS = 5  # 与 fourier_trajectory.py 一致

COEFFS_DIR = Path(__file__).resolve().parent / ".." / "trajectory_coefficients"

BAGS_ROOT = Path(__file__).resolve().parent / ".." / "bag_data"
TOPIC_KEY = "hardware_joint_state"  # 对应 /hardware/joint_state
DEFAULT_BAG = "rosbag2_1970_01_01-13_55_31"
DEFAULT_JOINTS = [13, 14, 15, 16, 17]


# ---------------------------------------------------------------------------
# 1) 数据加载
# ---------------------------------------------------------------------------
def load_bag(bag_name: str):
    """从 <bag>/csv/hardware_joint_state.csv 读取 (t, position, velocity)。

    支持短名（如 "13_55_31"）：自动按 glob 匹配到完整 bag 目录名。
    """
    bag_dir = BAGS_ROOT / bag_name
    if not (bag_dir / "csv").is_dir():
        # 尝试短名模糊匹配，例如 "13_55_31" -> "rosbag2_1970_01_01-13_55_31"
        matches = sorted(BAGS_ROOT.glob(f"*{bag_name}*"))
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
    t = t - t[0]  # 归零，便于分析
    print(f"  数据: {csv_path}")
    print(
        f"  N={len(t)}  时长={t[-1]:.2f}s  dt~{np.median(np.diff(t)) * 1e3:.2f}ms  "
        f"关节数={q.shape[1]}"
    )
    return t, q, v, bag_dir


def _read_summary_time_coeffs(bag_dir) -> float | None:
    """从 <bag>/summary.json 读取录制时的 time_coeffs（extract_bag_data.py 写入）。

    没有该字段（例如旧数据）时返回 None，由调用方用 --time-coeffs 覆盖或报错。
    """
    p = bag_dir / "summary.json"
    if not p.is_file():
        return None
    with open(p) as f:
        data = json.load(f)
    tc = data.get("time_coeffs")
    return float(tc) if tc is not None else None


# ---------------------------------------------------------------------------
# 2) 位置拟合 + 解析求导（f0 由 time_coeffs 决定，不再估计）
# ---------------------------------------------------------------------------
def fit_position(t, y, f0, n_harm=N_HARMONICS):
    """对位置做 5 谐波最小二乘拟合，返回各项与诊断。"""
    w = 2.0 * np.pi * f0
    cols = [np.sin(n * w * t) for n in range(1, n_harm + 1)]
    cols += [np.cos(n * w * t) for n in range(1, n_harm + 1)]
    cols.append(np.ones_like(t))
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)

    s = c[:n_harm]  # sin 系数
    cn = c[n_harm : 2 * n_harm]  # cos 系数
    dc = c[2 * n_harm]  # 直流偏置 q0

    q_fit = A @ c
    res = y - q_fit
    explained = (
        1.0 - float(np.var(res)) / float(np.var(y)) if np.var(y) > 0 else float("nan")
    )

    # 解析求导（由位置系数直接得到，自洽、无噪声放大）
    v_th = np.zeros_like(y)
    a_th = np.zeros_like(y)
    for n in range(1, n_harm + 1):
        wn = n * w
        v_th += wn * (s[n - 1] * np.cos(wn * t) - cn[n - 1] * np.sin(wn * t))
        a_th += -(wn**2) * (s[n - 1] * np.sin(wn * t) + cn[n - 1] * np.cos(wn * t))

    return {
        "f0": f0,
        "s": s,
        "c": cn,
        "dc": dc,
        "q_fit": q_fit,
        "v_th": v_th,
        "a_th": a_th,
        "res_rms": float(np.sqrt(np.mean(res**2))),
        "explained": explained,
    }


def compare_v(v_meas, v_th):
    """理论速度 vs 实测速度 的比对统计。"""
    res = v_meas - v_th
    rms = float(np.sqrt(np.mean(res**2)))
    std = float(np.std(v_meas))
    corr = float(np.corrcoef(v_meas, v_th)[0, 1]) if std > 0 else float("nan")
    return {
        "v_rms": rms,
        "v_std": std,
        "v_pct": 100.0 * rms / std if std > 0 else float("nan"),
        "corr": corr,
    }


# ---------------------------------------------------------------------------
# 3) 主流程
# ---------------------------------------------------------------------------
def analyze_joint(t, q, v, joint, f0):
    """对单个关节做完整分析，返回统计 dict（不含任何画图）。

    :param f0: 手动指定的基频（Hz），由 --time-coeffs / TRAJ_PERIOD 得到。
    """
    qj, vj = q[:, joint], v[:, joint]
    if np.std(qj) < 1e-6:
        print(f"  关节 {joint:2d}: 数据全 0 / 静止，跳过")
        return None

    fit = fit_position(t, qj, f0)
    cmp_ = compare_v(vj, fit["v_th"])

    print(f"\n=== 关节 {joint} ===")
    print(f"  f0 手动指定   : {fit['f0']:.2f} Hz  (周期 {1 / fit['f0']:.4f} s)")
    print(
        f"  位置拟合      : 解释方差 {fit['explained'] * 100:.4f}%  残差RMS {fit['res_rms']:.5f} rad"
    )
    print(
        f"  理论速度比对  : 残差RMS {cmp_['v_rms']:.4f}  = {cmp_['v_pct']:.2f}% (v std)"
        f"  相关系数 {cmp_['corr']:.6f}"
    )
    print(
        f"  解析加速度    : std {np.std(fit['a_th']):.3f}  max {np.abs(fit['a_th']).max():.3f} rad/s^2"
    )

    return {"joint": joint, "f0": f0, **fit, **cmp_}


# ---------------------------------------------------------------------------
# 4) 保存恢复出的轨迹系数（fourier_trajectory.py 的 YAML 格式）
# ---------------------------------------------------------------------------
def save_coeffs_yaml(
    results: list, out_name: str, bag: str = "", time_coeffs: float | None = None
) -> Path:
    """把恢复出的轨迹系数按 fourier_trajectory.py 的 YAML 格式保存。

    拟合得到的位置系数  q = dc + Σ_n [ s_n·sin(n·ω0·t) + c_n·cos(n·ω0·t) ]，
    换算成 fourier_trajectory.py 的 (a, b, q0) 约定：
        a_n = n·ω_nominal·s_n ,   b_n = −n·ω_nominal·c_n ,   q0 = dc
    其中 ω_nominal = 2π/TRAJ_PERIOD（标称角频率）。
    这样回放时只要传入与录制一致的 time_coeffs（--time-coeffs 手动指定）
    即可复现采集到的轨迹；YAML 本身格式不变。

    :param results: analyze_joint 返回的 dict 列表（按关节顺序）。
    :param out_name: 输出文件名，如 "recovered_260813_180000.yaml"。
    :param bag: bag 名，写进 _meta 备忘。
    :param time_coeffs: 辨识时手动提供的时间系数，写进 _meta 备忘。
    :return: 保存后的完整路径。
    """
    w_nominal = 2.0 * np.pi / TRAJ_PERIOD
    data = {}
    for i, r in enumerate(results):
        n = np.arange(1, N_HARMONICS + 1, dtype=float)
        data[f"joint_{i}"] = {
            "a": (n * w_nominal * r["s"]).tolist(),
            "b": (-n * w_nominal * r["c"]).tolist(),
            "q0": float(r["dc"]),
        }
    # 备忘信息（load_coeffs 只读 joint_ 开头的键，_meta 会被跳过，不影响加载）
    data["_meta"] = {
        "source_bag": bag,
        "time_coeffs": float(time_coeffs) if time_coeffs is not None else None,
        "f0_hz": float(results[0]["f0"]),
        "note": "recovered by fourier_fit.py; time_coeffs from bag summary.json (or --time-coeffs) at fit time",
    }

    COEFFS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COEFFS_DIR / out_name
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return out_path


# ---------------------------------------------------------------------------
# 5) 本地画图：用 fourier_trajectory 生成理论 q/v/a，与 bag 原始数据叠加对比
# ---------------------------------------------------------------------------
def plot_compare(
    yaml_name: str,
    dim: int,
    robot_joints: list,
    t_raw: np.ndarray,
    q_raw: np.ndarray,
    v_raw: np.ndarray,
    f0: float,
    sample_rate: float = 500.0,
):
    """画每个关节的对比图：原始 vs 理论 q/v + 底部残差小图。

    理论 q/v 由 fourier_trajectory 从保存的 YAML 生成，并按手动指定的 f0
    对齐录制频率（time_coeffs = TRAJ_PERIOD × f0），一周期信号按相位折叠
    延拓到整个原始时间轴后与 bag 原始数据叠加。

    :param yaml_name: trajectory_coefficients 下的 YAML 文件名。
    :param dim: 关节数。
    :param robot_joints: 各 YAML 关节对应的机器人关节号列表。
    :param t_raw, q_raw, v_raw: bag 原始时间 / 位置 / 速度。
    :param f0: 手动指定的基频（= time_coeffs / TRAJ_PERIOD），用于对齐录制频率。
    :param sample_rate: 生成理论轨迹的采样率。
    :return: Figure 列表（由调用方统一 plt.show()）。
    """
    import matplotlib.pyplot as plt

    tc = TRAJ_PERIOD * f0  # 与录制频率对齐的 time_coeffs
    traj = FourierTrajectory(dim=dim, sample_rate=sample_rate, time_coeffs=tc)
    q_th, v_th, _ = traj.generate_trajectory_from_yaml(yaml_name)
    t_th = traj.t_array
    period = TRAJ_PERIOD / tc  # 一周期时长（秒）

    figs = []
    for i, rj in enumerate(robot_joints):
        qr, vr = q_raw[:, rj], v_raw[:, rj]
        # 一周期理论值按相位折叠延拓到整个原始时间轴
        phase = np.mod(t_raw, period)
        q_t = np.interp(phase, t_th, q_th[i], period=period)
        v_t = np.interp(phase, t_th, v_th[i], period=period)
        rq, rv = qr - q_t, vr - v_t

        fig, axs = plt.subplots(
            4,
            1,
            figsize=(12, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 3, 1.3, 1.3]},
        )

        # q: 原始 vs 理论
        axs[0].plot(t_raw, qr, lw=0.5, alpha=0.7, label="raw")
        axs[0].plot(t_raw, q_t, lw=1.2, label="theoretical (fourier_trajectory)")
        axs[0].set_ylabel("q (rad)")
        axs[0].set_title(
            f"Joint {rj}  f0={f0:.2f} Hz  time_coeffs={tc:.2f}  "
            f"q resid RMS {np.sqrt(np.mean(rq**2)):.5f} rad"
        )
        axs[0].legend(loc="upper right", fontsize=8)
        axs[0].grid(alpha=0.3)

        # v: 原始 vs 理论
        axs[1].plot(t_raw, vr, lw=0.5, alpha=0.7, label="raw")
        axs[1].plot(t_raw, v_t, lw=1.2, label="theoretical")
        axs[1].set_ylabel("v (rad/s)")
        axs[1].set_title(
            f"v resid RMS {np.sqrt(np.mean(rv**2)):.4f} = "
            f"{100 * np.sqrt(np.mean(rv**2)) / np.std(vr):.2f}% of v std"
        )
        axs[1].legend(loc="upper right", fontsize=8)
        axs[1].grid(alpha=0.3)

        # 残差小图
        axs[2].plot(t_raw, rq, lw=0.6)
        axs[2].set_ylabel("q resid (rad)")
        axs[2].grid(alpha=0.3)

        axs[3].plot(t_raw, rv, lw=0.6)
        axs[3].set_ylabel("v resid (rad/s)")
        axs[3].set_xlabel("t (s)")
        axs[3].grid(alpha=0.3)

        fig.tight_layout()
        figs.append(fig)
    return figs


def main():
    ap = argparse.ArgumentParser(
        description="用位置最小二乘拟合傅里叶轨迹，并把恢复出的系数保存到 trajectory_coefficients/"
    )
    ap.add_argument("--bag", default=DEFAULT_BAG, help="bag_data 下的 bag 名")
    ap.add_argument(
        "--joints", nargs="+", type=int, default=DEFAULT_JOINTS, help="要分析的关节号"
    )
    ap.add_argument(
        "--time-coeffs",
        type=float,
        default=None,
        help="时间系数（默认从 bag summary.json 读取；extract_bag_data.py --time-coeffs 写入）",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="输出 YAML 文件名（默认 recovered_YYMMDD_HHMMSS.yaml）",
    )
    ap.add_argument(
        "--plot",
        default=True,
        action="store_true",
        help="保存后本地画图：对比理论 q/v 与 bag 原始数据并画残差小图",
    )
    args = ap.parse_args()

    t, q, v, bag_dir = load_bag(args.bag)

    # time_coeffs 默认从 bag summary.json 读取（extract 时写入），可 --time-coeffs 覆盖
    time_coeffs = args.time_coeffs
    if time_coeffs is None:
        time_coeffs = _read_summary_time_coeffs(bag_dir)
    if time_coeffs is None:
        raise SystemExit(
            f"bag {args.bag} 的 summary.json 里没有 time_coeffs，请用 --time-coeffs "
            "指定，或重新用 extract_bag_data.py --time-coeffs <值> 提取"
        )
    time_coeffs = float(time_coeffs)
    f0 = time_coeffs / TRAJ_PERIOD
    print(
        f"\nf0 = time_coeffs / TRAJ_PERIOD = {time_coeffs:.2f} / {TRAJ_PERIOD}"
        f" = {f0:.2f} Hz"
    )

    print("\n===== 逐个关节分析 =====")
    results = []
    for j in args.joints:
        r = analyze_joint(t, q, v, j, f0)
        if r is not None:
            results.append(r)

    # 汇总表
    if results:
        print("\n===== 汇总 =====")
        print(
            f"{'关节':>4} {'f0(Hz)':>9} {'位置解释%':>10} {'v残差RMS':>10} {'v残差%std':>9} {'相关系数':>10} {'a_max':>9}"
        )
        for r in results:
            print(
                f"{r['joint']:>4} {r['f0']:>9.2f} {r['explained'] * 100:>9.3f}% "
                f"{r['v_rms']:>10.4f} {r['v_pct']:>8.2f}% {r['corr']:>10.6f} "
                f"{np.abs(r['a_th']).max():>9.3f}"
            )

    if not results:
        return

    # 保存恢复出的轨迹系数（格式与 fourier_trajectory.py 一致）
    out_name = args.out or f"recovered_{datetime.datetime.now():%y%m%d_%H%M%S}.yaml"
    out_path = save_coeffs_yaml(
        results, out_name, bag=args.bag, time_coeffs=time_coeffs
    )
    print(f"\n恢复的轨迹系数已保存: {out_path}")
    print(
        "  关节映射 (YAML joint_i -> 机器人关节): "
        + ", ".join(f"joint_{i}->{r['joint']}" for i, r in enumerate(results))
    )
    print(
        f"  也可单独画图: python identification/fourier_trajectory.py "
        f"--yaml {out_name} --time-coeffs <录制时的值>"
    )

    # 本地画图：fourier_trajectory 生成理论 q/v/a，叠加 bag 原始数据 + 残差小图
    if args.plot:
        import matplotlib.pyplot as plt

        sample_rate = 1.0 / float(np.median(np.diff(t)))
        figs = plot_compare(
            yaml_name=out_name,
            dim=len(results),
            robot_joints=[r["joint"] for r in results],
            t_raw=t,
            q_raw=q,
            v_raw=v,
            f0=results[0]["f0"],
            sample_rate=sample_rate,
        )
        if figs:
            plt.show()


if __name__ == "__main__":
    main()
