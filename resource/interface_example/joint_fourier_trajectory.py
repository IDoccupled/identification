import numpy as np

import sys
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header

from interface_protocol.msg import JointCommand, JointState, MotionState  # type: ignore
from identification.fourier_trajectory import FourierTrajectory

LEFT_LEG_Q_INDICES = [0, 1, 2, 3, 4, 5]
RIGHT_LEG_Q_INDICES = [6, 7, 8, 9, 10, 11]
WAIST_Q_INDICES = [12]
LEFT_ARM_Q_INDICES = [13, 14, 15, 16, 17]
RIGHT_ARM_Q_INDICES = [18, 19, 20, 21, 22]
NECK_Q_INDICES = [23]

VALID_LIMB_GROUPS = {
    "left_leg": LEFT_LEG_Q_INDICES,
    "right_leg": RIGHT_LEG_Q_INDICES,
    "left_arm": LEFT_ARM_Q_INDICES,
    "right_arm": RIGHT_ARM_Q_INDICES,
    "waist": WAIST_Q_INDICES,
    "neck": NECK_Q_INDICES,
}

DEFAULT_GROUP_TO_IDENTIFY = "left_arm"
REPEAT_TRAJ = 1
SOFT_START_DURATION = 3.0  # seconds

CONTROL_FREQUENCY = 500.0
CONTROL_PERIOD = 1.0 / CONTROL_FREQUENCY

CONFIG_PATH = (
    Path(__file__).resolve().parent / ".." / ".." / "config" / "joint_sine.yaml"
).resolve()  # For PD controller parameters


def _flatten_groups(value):
    if value is None:
        return None
    if isinstance(value, list) and value and isinstance(value[0], list):
        flat = []
        for group in value:
            flat.extend(group)
        return flat
    return value


def _expand_or_default(value, num_joints, default_value):
    if isinstance(value, list):
        if len(value) == num_joints:
            return value
        return None
    return [default_value] * num_joints if value is None else [value] * num_joints


def _require_list(name, value, num_joints):
    if not isinstance(value, list) or len(value) != num_joints:
        raise ValueError(f"{name} must be a list with length {num_joints}")
    return value


def _to_float_list(name, value):
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc


class FourierTrajectoryNode(Node):
    def __init__(
        self,
        coeff: list,
        config_path: Path = CONFIG_PATH,
        group: str = DEFAULT_GROUP_TO_IDENTIFY,
    ):
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        super().__init__("fourier_trajectory_node")

        self.num_joints = 24
        self.target_joint_indices = VALID_LIMB_GROUPS[group]
        self.dim = len(self.target_joint_indices)

        kp_cfg = _flatten_groups(cfg.get("kp"))
        kd_cfg = _flatten_groups(cfg.get("kd"))
        self.kp_list = _expand_or_default(kp_cfg, self.num_joints, 100.0)
        self.kd_list = _expand_or_default(kd_cfg, self.num_joints, 1.0)
        self.kp_list = _require_list("kp", self.kp_list, self.num_joints)
        self.kd_list = _require_list("kd", self.kd_list, self.num_joints)
        self.kp_list = _to_float_list("kp", self.kp_list)
        self.kd_list = _to_float_list("kd", self.kd_list)

        # Create publishers and subscribers
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.joint_command_pub = self.create_publisher(
            JointCommand,
            "/hardware/joint_command",
            qos,
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/hardware/joint_state",
            self.joint_state_callback,
            qos,
        )
        self._motion_state_sub = self.create_subscription(
            MotionState,
            "/motion/motion_state",
            self.motion_state_callback,
            qos,
        )

        # Initialize trajectory
        self.q_traj, self.v_traj, _ = FourierTrajectory(
            dim=self.dim, repeat=REPEAT_TRAJ, sample_rate=CONTROL_FREQUENCY
        ).generate_trajectory(coeffs=np.array(coeff))
        self.total_samples = self.q_traj.shape[1]
        self.sample_per_traj = self.total_samples // REPEAT_TRAJ

        # Control loop state variables
        self.start_time = None
        self.timer = None
        self.latest_joint_state = None
        self.should_exit = False
        self.last_log_time = 0.0

        self.get_logger().info("FourierTrajectoryNode created")

    def joint_state_callback(self, msg: JointState):
        """Store latest joint state."""
        self.latest_joint_state = msg

    def motion_state_callback(self, msg: MotionState):
        """Check that we are in joint_bridge motion task."""
        if msg.current_motion_task != "joint_bridge":
            self.get_logger().error(
                f"Not in joint_bridge state (current: {msg.current_motion_task}), exiting"
            )
            self.should_exit = True

    def initialize(self) -> bool:
        """Wait for first joint state, then start the control timer."""
        try:
            # Wait for first joint state
            while self.latest_joint_state is None and not self.should_exit:
                rclpy.spin_once(self, timeout_sec=0.1)
                current_time = time.time()
                if current_time - self.last_log_time >= 1.0:
                    self.get_logger().info("Waiting for first joint state...")
                    self.last_log_time = current_time

            if self.should_exit:
                return False

            initial_positions = list(self.latest_joint_state.position)
            self.print_positions("Initial positions", initial_positions)

            # Save initial positions of target joints for soft-start blending
            self.initial_target_positions = [
                initial_positions[j] for j in self.target_joint_indices
            ]

            # Record start time and begin control loop
            self.start_time = self.get_clock().now()
            self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)

            self.get_logger().info(
                f"Fourier trajectory control started "
                f"(soft-start: {SOFT_START_DURATION:.1f}s)"
            )
            return True

        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")
            return False

    def control_loop(self):
        """Timer callback: publish joint command following Fourier trajectory."""
        if self.should_exit or self.latest_joint_state is None:
            return

        # Compute the trajectory sample index from elapsed time
        elapsed_sec = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        sample_idx = int(elapsed_sec * CONTROL_FREQUENCY) % self.total_samples

        # Build joint command for all 24 joints
        joint_command = JointCommand()
        joint_command.header = Header()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.header.frame_id = ""

        joint_command.position = [0.0] * self.num_joints
        joint_command.velocity = [0.0] * self.num_joints
        joint_command.feed_forward_torque = [0.0] * self.num_joints
        joint_command.torque = [0.0] * self.num_joints
        joint_command.stiffness = list(self.kp_list)
        joint_command.damping = list(self.kd_list)

        # Soft-start blending factor: 0 → 1 over SOFT_START_DURATION seconds
        alpha = min(elapsed_sec / SOFT_START_DURATION, 1.0)

        # Fill in target joint positions/velocities from Fourier trajectory,
        # and hold current position for non-target joints
        current_positions = list(self.latest_joint_state.position)
        for j in range(self.num_joints):
            if j in self.target_joint_indices:
                traj_idx = self.target_joint_indices.index(j)
                traj_pos = float(self.q_traj[traj_idx, sample_idx])
                traj_vel = float(self.v_traj[traj_idx, sample_idx])
                init_pos = self.initial_target_positions[traj_idx]
                # Blend from initial position to trajectory
                joint_command.position[j] = init_pos + alpha * (traj_pos - init_pos)
                joint_command.velocity[j] = alpha * traj_vel
            else:
                # Hold current position for joints not participating in identification
                joint_command.position[j] = current_positions[j]
                joint_command.velocity[j] = 0.0

        self.joint_command_pub.publish(joint_command)

    def print_positions(self, title: str, positions: list):
        """Print formatted position list."""
        ss = f"\n{title}:\n["
        for i, pos in enumerate(positions):
            ss += f"{pos:.3f}"
            if i < len(positions) - 1:
                ss += ", "
                if (i + 1) % 6 == 0:
                    ss += "\n "
        ss += "]\n"
        self.get_logger().info(ss)


def main(argv=None):
    rclpy.init(args=argv)
    coeff = [
        1.62747080e-01,
        -1.35994919e-01,
        -9.75755707e-02,
        -3.15167643e-01,
        -5.23979956e-01,
        4.63712076e-01,
        6.98639941e-01,
        2.35656746e-01,
        -8.73299926e-01,
        -2.69208837e-01,
        -9.21240000e-01,
        9.81766220e-06,
        -5.42230741e-02,
        -7.01016614e-02,
        -1.50428188e-01,
        1.27949633e-01,
        1.87742830e-01,
        1.26563514e-01,
        -3.47711475e-01,
        -3.47711475e-01,
        2.67178020e-01,
        3.98071414e-02,
        4.57650000e-01,
        0.00000000e00,
        -4.90937107e-02,
        -1.49872408e-01,
        3.16421212e-01,
        3.04698778e-01,
        -4.73053389e-01,
        4.15575151e-01,
        -2.28100405e-01,
        2.69538005e-01,
        7.91053030e-01,
        7.91053030e-01,
        2.77387025e-01,
        4.99636528e-06,
        -8.58345945e-02,
        6.92765476e-02,
        -1.54501135e-01,
        1.06842117e-01,
        1.61672021e-01,
        2.57503783e-01,
        -3.43338378e-01,
        5.45056611e-02,
        4.29172972e-01,
        -3.32355829e-01,
        -6.88704113e-01,
        1.00000000e-05,
        -1.19099379e-03,
        1.58210606e-01,
        3.16421212e-01,
        2.33777711e-01,
        -4.74631818e-01,
        -4.74631818e-01,
        6.32842424e-01,
        5.79828861e-01,
        3.33286477e-01,
        7.91053030e-01,
        -7.55400000e-01,
        0.00000000e00,
    ]
    node = FourierTrajectoryNode(coeff=coeff, group="left_arm")

    if not node.initialize():
        node.get_logger().error("Failed to initialize, exiting")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("User interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
