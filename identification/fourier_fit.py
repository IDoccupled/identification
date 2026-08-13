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
    * 真实基频 f0 不是标称 0.2 Hz！该 bag 是用 time_coeffs=0.75 跑的，
      周期 = 5 / 0.75 = 6.667 s  ->  f0 = 0.15 Hz。
      因此必须先估计 f0（FFT 粗扫 + 最小二乘细扫），不能直接套标称值。
    * 位置 5 谐波拟合可解释 ~99.996% 方差，残差 RMS ~0.007 rad；
    * 由位置系数解析求导的理论速度 vs 实测速度残差 ~0.83% (v 的 std)。

本脚本只负责"拟合 + 保存 + 打印诊断"，不含任何画图代码。
画图统一由 fourier_trajectory.py 的 plot_from_yaml() 从保存的 YAML 完成
（可传 --time-coeffs 手动指定时间倍率以复现采集轨迹）。

用法：
    python identification/fourier_fit.py --bag 13_55_31
    python identification/fourier_fit.py --bag 13_55_31 --joints 13 14 15
    python identification/fourier_fit.py --bag 13_55_31 --joints 13 --f0 0.15   # 手动指定 f0
    python identification/fourier_fit.py --bag 13_55_31 --time-coeffs 0.75     # 保存后自动画图

输出：
    拟合结果按 fourier_trajectory.py 的 YAML 格式保存到
    trajectory_coefficients/ 目录（joint_0..joint_{k-1}，含 a / b / q0）。
    保存后如需画图：
        python identification/fourier_trajectory.py --yaml recovered_xxx.yaml --time-coeffs 0.75
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import yaml

try:
    # 以包内模块方式导入（安装后 / colcon build）
    from .fourier_trajectory import TRAJ_PERIOD, plot_from_yaml
except ImportError:  # 以脚本方式直接运行
    from fourier_trajectory import TRAJ_PERIOD, plot_from_yaml

N_HARMONICS = 5  # 与 fourier_trajectory.py 一致
F0_MIN, F0_MAX = 0.05, 0.5  # f0 搜索范围（time_coeffs 归一化后约 0.1..0.4）

COEFFS_DIR = Path(__file__).resolve().parent / ".." / "trajectory_coefficients"

BAGS_ROOT = Path(__file__).resolve().parent / ".." / "extracted"
TOPIC_KEY = "hardware_joint_command_feedback"  # 对应 /hardware/joint_command_feedback
DEFAULT_BAG = "rosbag2_1970_01_01-13_55_31"
DEFAULT_JOINTS = [13, 14, 15, 16, 17]


# ---------------------------------------------------------------------------
# 1) 数据加载
# ---------------------------------------------------------------------------
def load_bag(bag_name: str):
    """从 extracted/<bag>/data.npz 读取 (t, position, velocity)。

    支持短名（如 "13_55_31"）：自动按 glob 匹配到完整 bag 目录名。
    """
    bag_dir = BAGS_ROOT / bag_name
    if not (bag_dir / "data.npz").is_file():
        # 尝试短名模糊匹配，例如 "13_55_31" -> "rosbag2_1970_01_01-13_55_31"
        matches = sorted(BAGS_ROOT.glob(f"*{bag_name}*"))
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
    t = d[f"{TOPIC_KEY}.t_s"].astype(float)
    q = d[f"{TOPIC_KEY}.position"].astype(float)  # (N, n_joints)
    v = d[f"{TOPIC_KEY}.velocity"].astype(float)  # (N, n_joints)
    t = t - t[0]  # 归零，便于分析
    print(f"  数据: {npz_path}")
    print(
        f"  N={len(t)}  时长={t[-1]:.2f}s  dt~{np.median(np.diff(t)) * 1e3:.2f}ms  "
        f"关节数={q.shape[1]}"
    )
    return t, q, v, bag_dir


# ---------------------------------------------------------------------------
# 2) 基频 f0 估计（粗扫 + 细扫，最小二乘残差最小处即真实 f0）
# ---------------------------------------------------------------------------
def _ls_residual(t, y, f0, n_harm=N_HARMONICS):
    """在给定 f0 下，用 5 谐波 + 直流基做最小二乘，返回 (残差RMS, 系数)。"""
    w = 2.0 * np.pi * f0
    cols = [np.sin(n * w * t) for n in range(1, n_harm + 1)]
    cols += [np.cos(n * w * t) for n in range(1, n_harm + 1)]
    cols.append(np.ones_like(t))
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    rms = float(np.sqrt(np.mean((y - A @ c) ** 2)))
    return rms, c


def estimate_f0(t, y, f0_lo=F0_MIN, f0_hi=F0_MAX, n_harm=N_HARMONICS):
    """粗网格 + 细网格扫描，找让 5 谐波拟合残差最小的 f0。

    注意不能只靠 FFT 找峰：某次谐波可能才是最大峰（例如关节 13 的
    3 次谐波最大），直接取峰再除以谐波次数会得到非整数倍的错误结果。
    """
    # 粗扫
    f_coarse = np.linspace(f0_lo, f0_hi, 401)
    res = np.array([_ls_residual(t, y, f)[0] for f in f_coarse])
    f_best = f_coarse[int(np.argmin(res))]
    # 细扫（围绕粗扫最优 ±0.005 Hz）
    f_fine = np.linspace(max(f0_lo, f_best - 0.005), min(f0_hi, f_best + 0.005), 401)
    res_f = np.array([_ls_residual(t, y, f)[0] for f in f_fine])
    f_final = f_fine[int(np.argmin(res_f))]
    return f_final


# ---------------------------------------------------------------------------
# 3) 位置拟合 + 解析求导
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
# 4) 主流程
# ---------------------------------------------------------------------------
def analyze_joint(t, q, v, joint, f0_override=None):
    """对单个关节做完整分析，返回统计 dict（不含任何画图）。"""
    qj, vj = q[:, joint], v[:, joint]
    if np.std(qj) < 1e-6:
        print(f"  关节 {joint:2d}: 数据全 0 / 静止，跳过")
        return None

    f0 = f0_override if f0_override is not None else estimate_f0(t, qj)
    fit = fit_position(t, qj, f0)
    cmp_ = compare_v(vj, fit["v_th"])

    print(f"\n=== 关节 {joint} ===")
    print(f"  f0 估计       : {fit['f0']:.5f} Hz  (周期 {1 / fit['f0']:.4f} s)")
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
# 5) 保存恢复出的轨迹系数（fourier_trajectory.py 的 YAML 格式）
# ---------------------------------------------------------------------------
def save_coeffs_yaml(results: list, out_name: str, bag: str = "") -> Path:
    """把恢复出的轨迹系数按 fourier_trajectory.py 的 YAML 格式保存。

    拟合得到的位置系数  q = dc + Σ_n [ s_n·sin(n·ω0·t) + c_n·cos(n·ω0·t) ]，
    换算成 fourier_trajectory.py 的 (a, b, q0) 约定：
        a_n = n·ω_nominal·s_n ,   b_n = −n·ω_nominal·c_n ,   q0 = dc
    其中 ω_nominal = 2π/TRAJ_PERIOD（标称角频率）。
    这样回放时只要传入与录制一致的 time_coeffs（如 0.75 -> f0=0.15Hz）
    即可复现采集到的轨迹；YAML 本身格式不变。

    :param results: analyze_joint 返回的 dict 列表（按关节顺序）。
    :param out_name: 输出文件名，如 "recovered_260813_180000.yaml"。
    :param bag: bag 名，写进 _meta 备忘。
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
        "estimated_f0_hz": float(results[0]["f0"]),
        "note": "recovered by fourier_fit.py; replay with the recording time_coeffs",
    }

    COEFFS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COEFFS_DIR / out_name
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="用位置最小二乘拟合傅里叶轨迹，并把恢复出的系数保存到 trajectory_coefficients/"
    )
    ap.add_argument("--bag", default=DEFAULT_BAG, help="extracted 下的 bag 名")
    ap.add_argument(
        "--joints", nargs="+", type=int, default=DEFAULT_JOINTS, help="要分析的关节号"
    )
    ap.add_argument("--f0", type=float, default=None, help="手动指定基频 f0 (Hz)")
    ap.add_argument(
        "--out",
        default=None,
        help="输出 YAML 文件名（默认 recovered_YYMMDD_HHMMSS.yaml）",
    )
    ap.add_argument(
        "--time-coeffs",
        type=float,
        default=None,
        help="若指定，保存后调用 fourier_trajectory.plot_from_yaml 画图，"
        "并按此时间倍率回放（复现采集轨迹一般用 0.75）",
    )
    args = ap.parse_args()

    t, q, v, _ = load_bag(args.bag)

    print("\n===== 逐个关节分析 =====")
    results = []
    for j in args.joints:
        r = analyze_joint(t, q, v, j, f0_override=args.f0)
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
                f"{r['joint']:>4} {r['f0']:>9.5f} {r['explained'] * 100:>9.3f}% "
                f"{r['v_rms']:>10.4f} {r['v_pct']:>8.2f}% {r['corr']:>10.6f} "
                f"{np.abs(r['a_th']).max():>9.3f}"
            )

    if not results:
        return

    # 保存恢复出的轨迹系数（格式与 fourier_trajectory.py 一致）
    out_name = args.out or f"recovered_{datetime.datetime.now():%y%m%d_%H%M%S}.yaml"
    out_path = save_coeffs_yaml(results, out_name, bag=args.bag)
    print(f"\n恢复的轨迹系数已保存: {out_path}")
    print(
        "  关节映射 (YAML joint_i -> 机器人关节): "
        + ", ".join(f"joint_{i}->{r['joint']}" for i, r in enumerate(results))
    )
    print(
        f"  复现/画图:  python identification/fourier_trajectory.py "
        f"--yaml {out_name} --time-coeffs <录制时的值>"
    )

    # 画图全部委托给 fourier_trajectory.plot_from_yaml（本脚本无任何绘图代码）
    if args.time_coeffs is not None:
        plot_from_yaml(out_name, dim=len(results), time_coeffs=args.time_coeffs)


if __name__ == "__main__":
    main()
