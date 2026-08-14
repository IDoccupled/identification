#!/usr/bin/env python3
"""
panda_learn.py — 用 pandas 读取 / 分析 ROS2 导出的关节 CSV，并画图。

目标：读取 joint command 和 joint command state 两个 CSV，
     取出指定关节的角度(position)、速度(velocity)、力矩(torque)，画图。

CSV 结构说明（两个文件列结构相同，共 149 列）：
    t_recv_ns, t_s, t_stamp_ns, frame_id, parallel_parser_type
    然后对每个关节 N = 0..23 依次出现：
        damping_N, feed_forward_torque_N, position_N,
        stiffness_N, torque_N, velocity_N
    所以某个关节的列名是固定的，例如关节 13：
        'position_13', 'velocity_13', 'torque_13'

用法：
    python identification/panda_learn.py                          # 默认 bag 和默认关节
    python identification/panda_learn.py --bag 13_57_28           # 换一个 bag
    python identification/panda_learn.py --joints 13 14 15        # 指定关节
    python identification/panda_learn.py --save                   # 顺便存 PNG

本脚本刻意写成了“教学风格”：每个 pandas 用法都用打印输出 + 注释讲清楚。
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# 默认 bag（当前打开的目录），改成你的目标 bag 名即可
# DEFAULT_BAG = "rosbag2_1970_01_01-13_51_06"
DEFAULT_BAG = "rosbag2_1970_01_01-13_55_31"
# 这个 bag 里真正有动作的关节（其它关节数据全是 0）
DEFAULT_JOINTS = [13, 14, 15, 16, 17]


# ---------------------------------------------------------------------------
# 1) 读取
# ---------------------------------------------------------------------------
def read_csvs(bag_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 joint command 与 joint command state 两个 CSV。"""
    cmd_path = os.path.join(bag_dir, "csv", "hardware_joint_command.csv")
    ste_path = os.path.join(bag_dir, "csv", "hardware_joint_state.csv")
    # ste_path = os.path.join(bag_dir, "csv", "hardware_joint_command_feedback.csv")

    print("=" * 70)
    print("① pd.read_csv —— 把 CSV 读成 DataFrame")
    print("=" * 70)

    # read_csv 直接读文件；comment/na_values 等参数可按需加
    cmd = pd.read_csv(cmd_path)  # 指令（目标位置、前馈力矩、刚度、阻尼）
    ste = pd.read_csv(ste_path)  # 反馈（实际位置、速度、力矩）

    print(f"   command   : {cmd_path}")
    print(f"   state  : {ste_path}")
    return cmd, ste


# ---------------------------------------------------------------------------
# 2) 基本侦察
# ---------------------------------------------------------------------------
def explore(df: pd.DataFrame, name: str) -> None:
    """演示 pandas 的常用“体检”方法。"""
    print()
    print("-" * 70)
    print(f"② DataFrame 体检 —— {name}")
    print("-" * 70)

    # shape: (行数, 列数)
    print(f"  .shape            -> 行×列 = {df.shape}")

    # dtypes: 每列的类型（这里的数值列应是 float64）
    print(f"  .dtypes 中前 8 列 :\n{df.dtypes.head(8)}")

    # head / tail: 看头尾几行
    print(f"  .head(2) 前两行:\n{df.head(2).to_string()}\n")

    # 时间列的信息
    print(
        f"  时间列 t_s        : min={df['t_s'].min():.3f}s, "
        f"max={df['t_s'].max():.3f}s, 点数={len(df)}"
    )

    # isna().sum(): 统计缺失值（本数据一般没有 NaN）
    na = df.isna().sum().sum()
    print(f"  缺失值总数        : {na}")

    # describe(): 数值列的统计摘要（count/mean/std/min/分位数/max）
    print(f"  .describe() 摘要（关节 13 的 position/torque/velocity）:")
    print(
        df[["position_13", "velocity_13", "torque_13"]].describe().round(3).to_string()
    )


# ---------------------------------------------------------------------------
# 3) 选列（取“指定关节”的数据）
# ---------------------------------------------------------------------------
def select_joint(df: pd.DataFrame, joint: int) -> pd.DataFrame:
    """
    取出某个关节的 位置/速度/力矩 三列 + 时间列。
    演示 pandas 的“列名用 f-string 拼接 + 多列切片选择”。
    """
    cols = {
        "t": df["t_s"],
        "pos": df[f"position_{joint}"],
        "vel": df[f"velocity_{joint}"],
        "torque": df[f"torque_{joint}"],
    }
    # 用 pd.DataFrame({...}) 拼成一个新的小表
    return pd.DataFrame(cols)


def demo_column_selection(df: pd.DataFrame) -> None:
    """展示几种常用的选列方式。"""
    print()
    print("-" * 70)
    print("③ 选列 —— 三种常见写法")
    print("-" * 70)

    # 方式 A：按列名列表直接选（多个列）
    a = df[["position_13", "velocity_13"]]
    print(f"  A. df[['position_13','velocity_13']]  -> {a.shape}")

    # 方式 B：按前缀模糊匹配（一次性捞出所有 position_* 列）
    b = df.filter(like="position_")
    print(f"  B. df.filter(like='position_')        -> {b.shape}（24 个关节的位置列）")

    # 方式 C：布尔掩码过滤行（比如只看 t_s 在 0~1 秒之间）
    c = df[(df["t_s"] >= 0) & (df["t_s"] <= 1)]
    print(f"  C. 布尔掩码过滤 t∈[0,1]s              -> {c.shape} 行")


# ---------------------------------------------------------------------------
# 4) 时间对齐：把 command 和 state 合到同一个时间轴上
# ---------------------------------------------------------------------------
def align_cmd_ste(cmd: pd.DataFrame, ste: pd.DataFrame, joint: int) -> pd.DataFrame:
    """
    把某个关节的 command 和 state 按时间对齐（merge_asof 最近邻对齐）。

    为什么不能直接 merge：
        command 和 state 的采样时刻不同（t_s 是两套时间点），
        直接按 t_s 相等 merge 会得到几乎为空的表。
    merge_asof 的做法：以 state 的时间为基准，为每一行找到
    时间最接近（且 <= 它）的 command 行，把两条拼成一行。
    """
    print()
    print("-" * 70)
    print(f"④ merge_asof 时间对齐 —— 关节 {joint}")
    print("-" * 70)

    # command 表里 torque 全是 0（指令不下发力矩），但列名会和 state 撞，
    # 所以这里把 command 的三个量都加上 cmd_ 前缀区分开
    c = select_joint(cmd, joint).rename(
        columns={"pos": "cmd_pos", "vel": "cmd_vel", "torque": "cmd_torque"}
    )
    f = select_joint(ste, joint)

    # 两表都要先按时间排序，merge_asof 才能工作
    c = c.sort_values("t")
    f = f.sort_values("t")

    aligned = pd.merge_asof(f, c, on="t", direction="backward")
    print(f"  merge_asof 后: {aligned.shape}（行数跟 state 一致）")
    print(f"  前 3 行:\n{aligned.head(3).round(4).to_string()}")
    return aligned


# ---------------------------------------------------------------------------
# 5) 画图
# ---------------------------------------------------------------------------
def plot_joint(
    aligned: pd.DataFrame, joint: int, save: bool = False, out_dir: str | None = None
) -> None:
    """
    画一个关节的 3 联图：
        上：位置 position   —— command(指令) vs state(实际)
        中：速度 velocity   —— 只有 state 有实测速度
        下：力矩 torque     —— 只有 state 有实测力矩
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, constrained_layout=True
    )
    t = aligned["t"]

    # 上：位置对比（指令 vs 实际）
    axes[0].plot(t, aligned["cmd_pos"], label="command", lw=1.0, color="tab:blue")
    axes[0].plot(t, aligned["pos"], label="state", lw=1.0, color="tab:orange")
    axes[0].set_ylabel("position [rad]")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].grid(alpha=0.3)

    # 中：速度（指令 vs 实际）
    axes[1].plot(t, aligned["cmd_vel"], label="command", lw=1.0, color="tab:blue")
    axes[1].plot(t, aligned["vel"], label="state", lw=1.0, color="tab:green")
    axes[1].set_ylabel("velocity [rad/s]")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    # 下：力矩（实际）
    axes[2].plot(t, aligned["torque"], label="state", lw=1.0, color="tab:red")
    axes[2].set_ylabel("torque [Nm]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Joint {joint}  position / velocity / torque")

    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"joint_{joint}.png")
        fig.savefig(path, dpi=150)
        print(f"  已保存图片 -> {path}")
    # plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="pandas 学习脚本：读关节 CSV 并画图")
    parser.add_argument("--bag", default=DEFAULT_BAG, help="bag 目录名")
    parser.add_argument(
        "--joints",
        nargs="+",
        type=int,
        default=DEFAULT_JOINTS,
        help="要分析的关节号，如: --joints 13 14 15",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="同时把图保存为 PNG",
    )
    args = parser.parse_args()

    # bag 目录：脚本位于 identification/ 下，数据在 ../bag_data/<bag>/
    here = os.path.dirname(os.path.abspath(__file__))
    bag_dir = os.path.join(os.path.dirname(here), "bag_data", args.bag)
    out_dir = os.path.join(here, "..", "plots", args.bag)

    cmd, ste = read_csvs(bag_dir)
    explore(cmd, "hardware_joint_command")
    explore(ste, "hardware_joint_command_state")
    demo_column_selection(ste)

    print()
    print("=" * 70)
    print("⑤ 逐关节：对齐 + 画图")
    print("=" * 70)
    for joint in args.joints:
        aligned = align_cmd_ste(cmd, ste, joint)
        plot_joint(aligned, joint, save=args.save, out_dir=out_dir)
    plt.show()

    # 额外演示：GroupBy 思维 —— 把 24 个关节的“平均位置幅度”算出来看谁在动
    print()
    print("=" * 70)
    print("⑥ 小彩蛋：一键找出“真正在动的关节”")
    print("=" * 70)
    moving = ste.filter(like="position_").std().sort_values(ascending=False).head(5)
    print("  position 标准差最大的 5 个关节（越大说明动得越明显）:")
    for col, val in moving.items():
        print(f"    {col:12s} std = {val:.4f}")


if __name__ == "__main__":
    main()
