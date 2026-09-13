#!/usr/bin/env python3
"""读 bag_data/<bag>/csv/hardware_imu_info.csv，画 IMU 三轴线加速度并打印均值。

用法：
    python -m identification.bag_imu_plot
    python -m identification.bag_imu_plot --bag rosbag2_1970_01_01-13_57_28
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def read_csvs(bag_dir: str) -> pd.DataFrame:
    imu_path = os.path.join(bag_dir, "csv", "hardware_imu_info.csv")
    imu = pd.read_csv(imu_path)  # IMU 数据
    print(f"   imu  : {imu_path}")
    return imu


def columns_select(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "t": df["t_s"],
        "x": df["linear_acceleration.x"],
        "y": df["linear_acceleration.y"],
        "z": df["linear_acceleration.z"],
    }
    return pd.DataFrame(cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="读 IMU CSV 并画三轴线加速度")
    parser.add_argument(
        "--bag",
        "-b",
        help="bag_data/ 下的 bag 目录名（默认: %(default)s）",
    )
    args = parser.parse_args()

    # bag 目录：脚本位于 identification/ 下，数据在 ../bag_data/<bag>/
    here = os.path.dirname(os.path.abspath(__file__))
    bag_dir = os.path.join(os.path.dirname(here), "bag_data", args.bag)
    imu = read_csvs(bag_dir)
    imu_selected = columns_select(imu)
    # calculate mean for each, x,y,z
    imu_mean = imu_selected.mean()
    print(f"Mean of IMU data:\n{imu_mean}")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = imu_selected["t"]
    for i, axis in enumerate(["x", "y", "z"]):
        axes[i].plot(t, imu_selected[axis], label=f"{axis} acceleration", lw=1.0)
        axes[i].axhline(imu_mean[axis], color="r", linestyle="--", label="mean")
        axes[i].set_ylabel(f"{axis} [m/s^2]")
        axes[i].legend(loc="upper right")
        axes[i].grid(alpha=0.3)
    axes[2].set_xlabel("time [s]")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
