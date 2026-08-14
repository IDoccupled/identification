import os
import matplotlib.pyplot as plt
import pandas as pd

# DEFAULT_BAG = "rosbag2_1970_01_01-13_51_06"
DEFAULT_BAG = "rosbag2_1970_01_01-13_55_31"
DEFAULT_JOINTS = [13, 14, 15, 16, 17]


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


def main():
    bag_dir = os.path.join(os.path.dirname(__file__), "..", "bag_data", DEFAULT_BAG)
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
