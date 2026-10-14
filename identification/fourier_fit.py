#!/usr/bin/env python3
"""
fourier_fit.py — 只用位置(编码器)最小二乘拟合傅里叶轨迹参数（阶数由
                --n-harmonics 指定，默认 N_HARMONICS_FIT = 15），
                然后由同一组参数解析求导得到理论速度 / 加速度，
                与实测速度比对，验证编码器位置 / 速度的一致性。

为什么只拟合位置？
-----------------
采集的关节反馈只有 position / velocity，没有 acceleration。
激励轨迹本身是 5 次谐波傅里叶形式，但实测位置里还叠着更高次谐波；
编码器对位置的测量几乎无噪声，因此位置是"信噪比最高"的量。
先用最小二乘从位置恢复出
    q(t) = q0 + Σ_n [ s_n·sin(n·ω0·t) + c_n·cos(n·ω0·t) ]   (n=1..N)
其中 N = --n-harmonics（默认 N_HARMONICS_FIT = 15，刻意高于激励的 5 次）。
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
    * 位置 15 阶（默认）拟合可解释 ~99.996% 方差，残差 RMS ~0.007 rad；
    * 由位置系数解析求导的理论速度 vs 实测速度残差 ~0.83% (v 的 std)。

本脚本流程：拟合 -> 保存 YAML -> 可选本地画图对比（--plot）。
画图时用 fourier_trajectory 从保存的 YAML 生成理论 q/v/a（与真实回放同一代码路径），
叠加 bag 原始位置 / 速度并画残差小图；理论轨迹按手动指定的 time_coeffs 对齐录制频率。

用法（--time-coeffs 可省略，默认从 bag 的 summary.json 读取；在包根目录 src/identification 下运行）：
    python -m identification.fourier_fit --bag 13_55_31
    python -m identification.fourier_fit --bag 13_55_31 --joints 13 14 15
    python -m identification.fourier_fit --bag 13_55_31 --time-coeffs 0.75 --plot   # 手动覆盖 + 画图

肢体选择（--group，默认 left_arm）：
    left_arm / right_arm / left_leg / right_leg / waist / neck。
    选定 group 后，默认关节号 = 该肢体的全部关节，默认 bag 也换成该肢体的
    bag（--joints / --bag 显式给出时覆盖）。还原腿轨迹就用 --group left_leg：
        python -m identification.fourier_fit --group left_leg                       # bag exc_leg_1，关节 0..5
        python -m identification.fourier_fit --group left_leg --bag exc_leg_0.75    # 换 bag
        python -m identification.fourier_fit --group left_leg --joints 0 1 2 3      # 只拟合部分关节

时间窗（--twin START:END，两端可省略，单位秒；默认全程）：
    只用这段样本做最小二乘，跟 compare_torque / plot_residual 的 -w/--twin 和
    sdp_ridge 的 --twin-id 同一套写法。与那两处一样，**时间轴不重定基准**：
    相位始终以 bag 起点（t=0）为准，所以窗口拟合出的系数与全程拟合在同一
    相位约定下，可直接回放、直接叠加 bag 数据看残差。
        python -m identification.fourier_fit --group left_leg --twin 0:13.4         # 只拟合前 13.4 s
        python -m identification.fourier_fit --group left_leg --twin 20:            # 从 20 s 到末尾

默认时间窗（自动跳过录制开头的瞬变段）：
    不给 --twin 时，自动用"周期折叠"判据找"轨迹已稳定"的起点：录制开头常常
    有掉帧、时间戳跳变、或从保持位姿切入轨迹，这些样本偏差远大于正常建模残差
    （~3%），自动截掉，拟合到 bag 末尾。--no-auto-twin 可关闭、改回全程。
        python -m identification.fourier_fit --group left_leg                     # 自动截断
        python -m identification.fourier_fit --group left_leg --no-auto-twin      # 全程

谐波阶数（--n-harmonics，默认 N_HARMONICS_FIT = 15）：
    激励轨迹本身只有 5 次谐波，但实测位置里还含有 5 次以上、f0 整数倍的高次
    谐波（摩擦/回差等非线性在周期运动下"固定相位"的特征），只用激励的 5 阶
    拟合吸收不了，表现为"每个周期同一位置"的残差。这些都是 f0 的整数倍谐波，
    提高阶数就能拟合进去：
        python -m identification.fourier_fit --group left_leg --n-harmonics 30
    默认 15（刻意高于激励设计的 5 次，用来吸收上述高次谐波）；>5 时下游
    fourier_trajectory.generate_trajectory 会按 YAML 系数个数自动推断阶数。
    阶数并非越大越好：实测（exc_leg_0.75）残差 RMS 在 5 / 15 / 30 阶约为
    2.8e-2 / 7.2e-3 / 4.0e-3 rad，30 阶以后完全饱和。饱和后剩下的残差已经不是
    f0 的整数倍谐波（那是逐周期不可重复的跟踪误差，多高阶都吸收不了），
    两者用 plot_residual 的相位折叠 / 残差频谱可以区分。

输出：
    拟合结果按 fourier_trajectory.py 的 YAML 格式保存到
    trajectory_coefficients/ 目录（joint_0..joint_{k-1}，含 a / b / q0），
    并在 _meta 里写入 group / source_bag / time_coeffs / twin_s，供 sdp_ridge、
    plot_trajectory、compare_torque 等按肢体自动推出关节映射。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from identification.fourier_trajectory import FourierTrajectory, TRAJ_PERIOD

N_HARMONICS_FIT = 15

COEFFS_DIR = Path(__file__).resolve().parent / ".." / "trajectory_coefficients"

BAGS_ROOT = Path(__file__).resolve().parent / ".." / "bag_data"
TOPIC_KEY = "hardware_joint_state"  # 对应 /hardware/joint_state

# 肢体组 -> 机器人关节号（镜像 identification/target_limb_regressor.VALID_LIMB_GROUPS）。
# 刻意复制一份（而不是 import target_limb_regressor）：那个模块会拉起
# pinocchio / ament_index_python，需要先 source ROS 环境，而本脚本要保持
# "有 numpy/pandas/yaml 就能跑"。（identification/plot_tools/make_traj_table.py
# 出于同样理由也镜像了这张表。）
GROUP_Q_INDICES = {
    "left_leg": [0, 1, 2, 3, 4, 5],
    "right_leg": [6, 7, 8, 9, 10, 11],
    "waist": [12],
    "left_arm": [13, 14, 15, 16, 17],
    "right_arm": [18, 19, 20, 21, 22],
    "neck": [23],
}

DEFAULT_GROUP = "left_arm"
DEFAULT_BAG = "rosbag2_1970_01_01-13_55_31"  # left_arm 的默认 bag
# 其他肢体在不给 --bag 时用的默认 bag（没有记录的组回退到 DEFAULT_BAG）。
# left_leg 用 exc_leg_0.75：exc_leg_1 的录制开头有问题（数据不太准确），
# 0.75 作辨识、0.5 作交叉验证。
DEFAULT_BAG_BY_GROUP = {
    "left_arm": DEFAULT_BAG,
    "left_leg": "exc_leg_0.75",
    "right_leg": "exc_leg_1",
}


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
        raise FileNotFoundError(
            f"bag {bag_dir.name} 没有 summary.json，请用 --time-coeffs 指定"
        )
    with open(p) as f:
        data = json.load(f)
    tc = data.get("time_coeffs")
    if tc is None:
        raise ValueError(f"bag {bag_dir.name} 的 summary.json 中没有 time_coeffs")
    return float(tc)


def _parse_window(s: str | None, t_end: float) -> tuple[float, float]:
    """解析 '--twin START:END' 时间窗（可省略任一端）-> (t0, t1)。None = 全程。

    与 compare_torque / plot_residual 的 _parse_window 同义（小函数刻意复制，
    避免 import 那些脚本 / sdp_ridge）。
    """
    if s is None:
        return 0.0, t_end
    parts = s.split(":")
    if len(parts) > 2:
        raise ValueError(f"时间窗格式应为 'start:end'，得到: '{s}'")
    t0 = float(parts[0]) if parts[0].strip() else 0.0
    t1 = float(parts[1].strip()) if len(parts) == 2 and parts[1].strip() else t_end
    return t0, t1


# ---------------------------------------------------------------------------
# 2) 位置拟合 + 解析求导
# ---------------------------------------------------------------------------
def fit_position(t, y, f0, n_harm=N_HARMONICS_FIT):
    """对位置做 n_harm 次谐波最小二乘拟合，返回各项与诊断。"""
    w = 2.0 * np.pi * f0
    cols = [np.sin(n * w * t) for n in range(1, n_harm + 1)] + [
        np.cos(n * w * t) for n in range(1, n_harm + 1)
    ]
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


def detect_stable_start(t, q, period, tol_frac=0.15, n_bins=240):
    """自动检测"轨迹已稳定"的起点（秒）：跳过录制开头的瞬变段。

    轨迹本身是周期的，所以用"第一周期之后"的样本按相位折叠出参考轮廓 r(φ)，
    再算每个样本相对 r(φ(t)) 的偏差。录制开头若掉帧、时间戳/位置跳变、或从
    保持位姿切入轨迹，偏差会远大于正常值（正常只剩建模残差 ~3%）。取第一周期
    内最后一个越界样本的下一时刻作为起点。

    :param t: 时间（秒，已归零，`t[0] == 0`）。
    :param q: 位置 (N, n_joints)。
    :param period: 一个轨迹周期（秒），= TRAJ_PERIOD / time_coeffs。
    :param tol_frac: 偏差阈值，相对各关节幅值（0.15 = 15%）。
    :param n_bins: 相位分箱数。
    :return: (t_start, info)：t_start 为起点秒数（0.0 = 没检测到瞬变）；
        info 为一行说明，便于打印。
    """
    n_j = q.shape[1]
    span = float(t[-1] - t[0])
    if span < 2.0 * period:
        return 0.0, f"数据只有 {span / period:.2f} 周期(<2)，跳过自动截断"

    ref_mask = t >= t[0] + period  # 第一周期之后 = 已锁定到周期轨道
    if int(ref_mask.sum()) < 4 * n_bins:
        return 0.0, "参考样本太少，跳过自动截断"

    ph = np.mod(t - t[0], period)
    refph = np.mod(t[ref_mask] - t[0], period)
    bins = np.clip((refph / period * n_bins).astype(int), 0, n_bins - 1)
    centre = (np.arange(n_bins) + 0.5) / n_bins * period
    ref = np.full((n_bins, n_j), np.nan)
    for j in range(n_j):
        qj = q[ref_mask, j]
        for b in np.unique(bins):
            ref[b, j] = np.median(qj[bins == b])
    for j in range(n_j):  # 空箱按相位循环插值补齐
        good = np.isfinite(ref[:, j])
        if good.sum() < 2:
            return 0.0, "参考轮廓相位覆盖不足，跳过自动截断"
        ref[:, j] = np.interp(centre, centre[good], ref[good, j], period=period)

    ref_at_t = np.column_stack(
        [np.interp(ph, centre, ref[:, j], period=period) for j in range(n_j)]
    )
    amp = np.nanstd(q[ref_mask], axis=0)
    # 只统计"明显在动"的关节：包里的静止关节（幅值 ~1e-4 的编码器噪声）若也参与
    # 归一化，偏差会被放大到无穷大，把整段都判成瞬变。
    moving = amp > max(1e-6, 0.02 * float(np.max(amp)))
    if not np.any(moving):
        return 0.0, "没有明显运动的关节，跳过自动截断"
    safe_amp = np.where(moving, amp, np.inf)
    score = np.max(np.abs(q - ref_at_t) / safe_amp, axis=1)

    within_first = int(np.searchsorted(t, t[0] + period))
    bad = np.where(score[:within_first] > tol_frac)[0]
    if not len(bad):
        return 0.0, f"第一周期内无瞬变（阈值 {tol_frac:.0%}·幅值）"
    k = int(bad[-1]) + 1
    t_start = float(t[k]) if k < len(t) else float(t[-1])
    return t_start, (
        f"瞬变段结束于 t={t[int(bad[-1])]:.3f} s（阈值 {tol_frac:.0%}·幅值），"
        f"从 {t_start:.3f} s 起拟合"
    )


# ---------------------------------------------------------------------------
# 3) 主流程
# ---------------------------------------------------------------------------
def analyze_joint(t, q, v, joint, f0, n_harm=N_HARMONICS_FIT):
    """对单个关节做完整分析，返回统计 dict（不含任何画图）。

    :param f0: 手动指定的基频（Hz），由 --time-coeffs / TRAJ_PERIOD 得到。
    :param n_harm: 拟合的傅里叶谐波阶数（默认 N_HARMONICS_FIT = 15；
        激励设计本身只有 5 次）。
    """
    qj, vj = q[:, joint], v[:, joint]
    if np.std(qj) < 1e-6:
        print(f"  关节 {joint:2d}: 数据全 0 / 静止，跳过")
        return None

    fit = fit_position(t, qj, f0, n_harm=n_harm)
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
    results: list,
    out_name: str,
    bag: str = "",
    time_coeffs: float | None = None,
    group: str | None = None,
    twin: tuple[float, float] | None = None,
    n_harmonics: int = N_HARMONICS_FIT,
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
    :param group: 肢体组名（left_leg / left_arm / ...），写进 _meta.group；
        下游 sdp_ridge / plot_trajectory / compare_torque / plot_residual
        都从这里取关节映射，所以腿的 YAML 一定要带这个字段。
    :param twin: 实际使用的拟合时间窗 (t0, t1)（秒，绝对时间）；全程拟合给 None。
    :return: 保存后的完整路径。
    """
    w_nominal = 2.0 * np.pi / TRAJ_PERIOD
    n = np.arange(1, int(n_harmonics) + 1, dtype=float)
    data = {}
    for i, r in enumerate(results):
        data[f"joint_{i}"] = {
            "a": (n * w_nominal * r["s"]).tolist(),
            "b": (-n * w_nominal * r["c"]).tolist(),
            "q0": float(r["dc"]),
        }
    # 备忘信息（load_coeffs 只读 joint_ 开头的键，_meta 会被跳过，不影响加载）
    data["_meta"] = {
        "source_bag": bag,
        "group": group,
        "time_coeffs": float(time_coeffs) if time_coeffs is not None else None,
        "f0_hz": float(results[0]["f0"]),
        "n_harmonics": int(n_harmonics),
        "twin_s": [float(twin[0]), float(twin[1])] if twin is not None else None,
        "note": "recovered by fourier_fit.py; order = n_harmonics (len(a)=n); time_coeffs from bag summary.json (or --time-coeffs) at fit time",
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
    ap.add_argument(
        "--group",
        "-g",
        default=DEFAULT_GROUP,
        choices=sorted(GROUP_Q_INDICES),
        help=(
            "肢体组，决定默认关节号与默认 bag（默认 %(default)s）: "
            + " / ".join(sorted(GROUP_Q_INDICES))
        ),
    )
    ap.add_argument(
        "--bag", "-b", default=None, help="bag_data 下的 bag 名（默认按 --group 选择）"
    )
    ap.add_argument(
        "--joints",
        nargs="+",
        type=int,
        default=None,
        help="要分析的关节号（默认取 --group 该肢体的全部关节）",
    )
    ap.add_argument(
        "--time-coeffs",
        type=float,
        default=None,
        help="时间系数（默认从 bag summary.json 读取；extract_bag_data.py --time-coeffs 写入）",
    )
    ap.add_argument(
        "-w",
        "--twin",
        default=None,
        metavar="START:END",
        help="只拟合时间段（秒，两端可省略），如 '0:13.4' / '20:'；默认全程",
    )
    ap.add_argument(
        "--n-harmonics",
        "-N",
        type=int,
        default=N_HARMONICS_FIT,
        help="傅里叶谐波阶数（默认 %(default)s；激励设计本身只有 5 次）。实测位置"
        "含 5 次以上、f0 整数倍的高次谐波（摩擦等非线性），可提高到 ~30 阶，"
        "实测 30 阶后饱和",
    )
    ap.add_argument(
        "--no-auto-twin",
        action="store_true",
        help="关闭默认的自动跳过开头瞬变段，改用全程（--twin 显式给出时本项无效）",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="输出 YAML 文件名（默认 recovered_<bag>.yaml）",
    )
    ap.add_argument(
        "--plot",
        default=True,
        action="store_true",
        help="保存后本地画图：对比理论 q/v 与 bag 原始数据并画残差小图",
    )
    args = ap.parse_args()
    n_harm = int(args.n_harmonics)

    # 肢体组决定默认关节集合；--joints / --bag 显式给出时覆盖。
    # 腿/臂走的是同一条拟合代码路径，唯一区别就是这里的关节号（腿 6 DoF，臂 5 DoF）。
    joints = (
        args.joints if args.joints is not None else list(GROUP_Q_INDICES[args.group])
    )
    bag_name = args.bag or DEFAULT_BAG_BY_GROUP.get(args.group, DEFAULT_BAG)
    print(f"肢体组: {args.group}   关节: {joints}   (--joints 可覆盖)")
    print(f"谐波阶数   : {n_harm}")

    t, q, v, bag_dir = load_bag(bag_name)

    # time_coeffs 默认从 bag summary.json 读取（extract 时写入），可 --time-coeffs 覆盖。
    # 先读：周期 = TRAJ_PERIOD / time_coeffs 决定自动截断与"窗口短于一周期"的告警。
    time_coeffs = args.time_coeffs
    if time_coeffs is None:
        time_coeffs = _read_summary_time_coeffs(bag_dir)
    if time_coeffs is None:
        raise SystemExit(
            f"bag {bag_name} 的 summary.json 里没有 time_coeffs，请用 --time-coeffs "
            "指定，或重新用 extract_bag_data.py --time-coeffs <值> 提取"
        )
    time_coeffs = float(time_coeffs)
    f0 = time_coeffs / TRAJ_PERIOD
    period = TRAJ_PERIOD / time_coeffs
    print(
        f"\nf0 = time_coeffs / TRAJ_PERIOD = {time_coeffs:.2f} / {TRAJ_PERIOD}"
        f" = {f0:.2f} Hz  (周期 {period:.3f} s)"
    )

    # 时间窗（--twin START:END，两端可省略）：只用这段样本做最小二乘，
    # 与 sdp_ridge --twin-id / plot_residual -w 同义（轨迹是周期的，取一个干净
    # 周期就够，也能把坏段、静止段排除掉）。注意**时间轴不重定基准**：phase 始终
    # 以 bag 起点为准，所以窗口拟合的系数与全程拟合在同一相位约定下可直接比。
    #
    # 默认（不给 --twin）：自动跳过录制开头的瞬变段（掉帧 / 时间戳与位置跳变 /
    # 从保持位姿切入轨迹），从"轨迹已稳定"的时刻开始拟合。
    twin = None
    if args.twin is not None:
        t0, t1 = _parse_window(args.twin, float(t[-1]))
        source = f"--twin {args.twin}"
    elif args.no_auto_twin:
        t0, t1 = float(t[0]), float(t[-1])
        source = "--no-auto-twin（全程）"
    else:
        # 只用本次拟合的关节判瞬变：包里的其他关节（腰/另一条腿等）即使轻微动
        # 一下，也可能主导归一化后的偏差，把正常段误判成瞬变。
        t0, info = detect_stable_start(t, q[:, joints], period)
        t1 = float(t[-1])
        source = f"自动截断开头瞬变段（{info}）"

    if t0 > float(t[0]) or t1 < float(t[-1]):
        mask = (t >= t0) & (t <= t1)
        if not np.any(mask):
            raise SystemExit(
                f"时间窗 [{t0:.3g}, {t1:.3g}] 内没有样本（bag 时间轴 [0, {t[-1]:.3f}] s）"
            )
        t, q, v = t[mask], q[mask], v[mask]
        twin = (float(t[0]), float(t[-1]))
        print(
            f"时间窗     : {source}\n"
            f"             ->  [{t[0]:.3f}, {t[-1]:.3f}] s  {len(t)} 样本 "
            f"({t[-1] - t[0]:.3f} s = {(t[-1] - t[0]) / period:.2f} 周期)"
        )
    else:
        print(f"时间窗     : {source}（未截断，全程 {len(t)} 样本）")

    if len(t) < 2 * n_harm + 1:
        raise SystemExit(
            f"时间窗内只有 {len(t)} 个样本，少于 {2 * n_harm + 1} 个未知系数，无法拟合"
        )

    # 窗口短于一个周期时，窗口内各次谐波几乎共线，系数不可靠（拟合值仍能贴合
    # 该段数据，但当做轨迹回放/外推就没意义了）。
    win = float(t[-1] - t[0])
    if win < period:
        print(
            f"[警告] 拟合窗口 {win:.2f}s 短于轨迹周期 {period:.2f}s：窗口内 {n_harm} "
            f"次谐波（最短周期 {period / n_harm:.2f}s）高度相关，系数不可靠，"
            "建议窗口至少取一个完整周期。"
        )

    print("\n===== 逐个关节分析 =====")
    results = []
    for j in joints:
        r = analyze_joint(t, q, v, j, f0, n_harm=n_harm)
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

    # 静态关节会被 analyze_joint 跳过，导致 YAML 的 joint_i 数目少于该肢体 DoF，
    # 下游 sdp_ridge / plot_trajectory 会直接对不上号，这里先报警。
    if len(results) != len(joints):
        print(
            f"\n[警告] {args.group} 应有 {len(joints)} 个关节，实际只拟合出 "
            f"{len(results)} 个（其余关节数据静止被跳过）。\n"
            "        该 YAML 的关节数与该肢体 DoF 不一致，请检查 bag / --joints。"
        )

    # 保存恢复出的轨迹系数（格式与 fourier_trajectory.py 一致）
    out_name = args.out or f"recovered_{bag_name}.yaml"
    out_path = save_coeffs_yaml(
        results,
        out_name,
        bag=bag_name,
        time_coeffs=time_coeffs,
        group=args.group,
        twin=twin,
        n_harmonics=n_harm,
    )
    print(f"\n恢复的轨迹系数已保存: {out_path}")
    print(
        f"  _meta.group = {args.group}    _meta.source_bag = {bag_name}    "
        f"_meta.n_harmonics = {n_harm}"
    )
    if twin is not None:
        print(
            f"  _meta.twin_s = [{twin[0]:.3f}, {twin[1]:.3f}]（绝对时间，相位仍以 bag 起点为准）"
        )
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
