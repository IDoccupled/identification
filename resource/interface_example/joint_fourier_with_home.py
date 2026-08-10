#!/usr/bin/env python3
"""
Combined node: first move all joints to a manually-set home (target) position,
then start the Fourier trajectory on selected joints while holding others.

Usage:
    python joint_fourier_with_home.py --type unified --group left_arm
    python joint_fourier_with_home.py --type unified --group left_arm --config_file /path/to/joint_test.yaml

Features:
    Phase 1 — Go to target positions defined in the YAML config (joint_test.yaml).
    Phase 2 — Start the Fourier trajectory on the selected limb group,
              while non-selected joints stay at their target positions.
"""

import argparse
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
SOFT_START_DURATION = 3.0  # seconds
CONTROL_FREQUENCY = 500.0
CONTROL_PERIOD = 1.0 / CONTROL_FREQUENCY
NUM_JOINTS = 24

# ---------------------------------------------------------------------------
# Home position for all 24 joints (set to 0.0 to keep the joint relaxed/uncontrolled)
# Joint layout: left_leg[0-5], right_leg[6-11], waist[12],
#               left_arm[13-17], right_arm[18-22], neck[23]
# ---------------------------------------------------------------------------
HOME_POS = [
    # left leg (0-5)
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    # right leg (6-11)
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    # waist (12)
    0.0,
    # left arm (13-17)
    0.01,
    0.1,
    0.01,
    0.01,
    0.01,
    # right arm (18-22)
    0.0,
    -0.0,
    0.0,
    0.0,
    0.0,
    # neck (23)
    0.0,
]

# Default path to the joint_test.yaml (used for num_steps, kp/kd, etc.)
DEFAULT_TARGET_CONFIG = (
    Path(__file__).resolve().parent / ".." / ".." / "config" / "joint_test.yaml"
).resolve()

# Default path to the PD config (kp/kd)
PD_CONFIG_PATH = (
    Path(__file__).resolve().parent / ".." / ".." / "config" / "joint_sine.yaml"
).resolve()


# ---------------------------------------------------------------------------
# Helpers (from both original files)
# ---------------------------------------------------------------------------
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


def _load_grouped_config(config_dict, key):
    """Load a key from config that has a grouped structure (list of lists)."""
    raw = config_dict.get(key, [])
    result = []
    for group in raw:
        result.extend(group)
    return result


# ---------------------------------------------------------------------------
# Combined Node
# ---------------------------------------------------------------------------
class FourierWithHomeNode(Node):
    """Node that first homes to target positions, then runs Fourier trajectory."""

    def __init__(
        self,
        yaml_name: str,
        target_config_path: Path = DEFAULT_TARGET_CONFIG,
        pd_config_path: Path = PD_CONFIG_PATH,
        group: str = DEFAULT_GROUP_TO_IDENTIFY,
        time_coeffs: float = 1.0,
        dry_run: bool = True,
    ):
        assert target_config_path.exists(), (
            f"Target config not found: {target_config_path}"
        )
        assert pd_config_path.exists(), f"PD config not found: {pd_config_path}"
        assert group in VALID_LIMB_GROUPS, (
            f"Invalid group '{group}', must be one of: {list(VALID_LIMB_GROUPS.keys())}"
        )

        super().__init__("fourier_with_home_node")

        # --- Load target-position config (Phase 1) ---
        with target_config_path.open("r", encoding="utf-8") as f:
            target_cfg = yaml.safe_load(f)

        self.target_positions = list(HOME_POS)
        self.num_steps_list = [
            int(s) for s in _load_grouped_config(target_cfg, "num_steps")
        ]
        # Identify joints that should remain relaxed (home pos == 0)
        self.relaxed_joints = [i for i, p in enumerate(HOME_POS) if p == 0.0]

        # Phase-1 PD gains (from target config, or fallback to pd_config)
        kp_phase1_raw = _load_grouped_config(target_cfg, "kp") or [100.0] * NUM_JOINTS
        kd_phase1_raw = _load_grouped_config(target_cfg, "kd") or [1.0] * NUM_JOINTS
        self.kp_phase1 = (
            kp_phase1_raw[:NUM_JOINTS]
            if len(kp_phase1_raw) >= NUM_JOINTS
            else kp_phase1_raw + [100.0] * (NUM_JOINTS - len(kp_phase1_raw))
        )
        self.kd_phase1 = (
            kd_phase1_raw[:NUM_JOINTS]
            if len(kd_phase1_raw) >= NUM_JOINTS
            else kd_phase1_raw + [1.0] * (NUM_JOINTS - len(kd_phase1_raw))
        )

        # --- Load PD config (Phase 2) ---
        with pd_config_path.open("r", encoding="utf-8") as f:
            pd_cfg = yaml.safe_load(f)

        kp_cfg = _flatten_groups(pd_cfg.get("kp"))
        kd_cfg = _flatten_groups(pd_cfg.get("kd"))
        self.kp_list = _expand_or_default(kp_cfg, NUM_JOINTS, 100.0)
        self.kd_list = _expand_or_default(kd_cfg, NUM_JOINTS, 1.0)
        self.kp_list = _require_list("kp", self.kp_list, NUM_JOINTS)
        self.kd_list = _require_list("kd", self.kd_list, NUM_JOINTS)
        self.kp_list = _to_float_list("kp", self.kp_list)
        self.kd_list = _to_float_list("kd", self.kd_list)

        # --- Limb group settings ---
        self.target_joint_indices = VALID_LIMB_GROUPS[group]
        self.dim = len(self.target_joint_indices)
        self.group = group
        self.dry_run = dry_run
        if self.dry_run:
            self.get_logger().warn(
                "DRY RUN mode — no joint commands will be published. "
                "Use --no-dry_run to send commands to the robot."
            )

        # --- Prepare Phase 2: Fourier trajectory ---
        self.q_traj, self.v_traj, _ = FourierTrajectory(
            dim=self.dim, sample_rate=CONTROL_FREQUENCY, time_coeffs=time_coeffs
        ).generate_trajectory_from_yaml(yaml_name)
        self.total_samples = self.q_traj.shape[1]

        # --- ROS2 pub/sub ---
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.joint_command_pub = self.create_publisher(
            JointCommand, "/hardware/joint_command", qos
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/hardware/joint_state", self.joint_state_callback, qos
        )
        self._motion_state_sub = self.create_subscription(
            MotionState, "/motion/motion_state", self.motion_state_callback, qos
        )

        # --- State variables ---
        self.latest_joint_state = None
        self.should_exit = False
        self.last_log_time = 0.0
        self.timer = None

        # Phase 1 state
        self.phase = "homing"  # "homing" → "holding" → "fourier"
        self.interpolated_positions = []  # interpolated trajectory for homing
        self.current_steps = []
        self.reached_targets = []
        self.all_homed = False

        # Phase 2 state
        self.fourier_start_time = None
        self.final_positions = None  # positions to hold after homing (target positions)

        self.get_logger().info(
            f"FourierWithHomeNode created: group={group}, phase={self.phase}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def motion_state_callback(self, msg: MotionState):
        if msg.current_motion_task != "joint_bridge":
            self.get_logger().error(
                f"Not in joint_bridge state (current: {msg.current_motion_task}), exiting"
            )
            self.should_exit = True

    # ------------------------------------------------------------------
    # Phase 1: generate interpolated homing trajectories
    # ------------------------------------------------------------------
    def _generate_homing_trajectories(self, initial_positions):
        """Create interpolated paths from current positions to target positions.
        Joints with HOME_POS == 0 are skipped (marked as already reached)."""
        self.interpolated_positions = []
        self.current_steps = [0] * NUM_JOINTS
        self.reached_targets = [False] * NUM_JOINTS

        for i in range(NUM_JOINTS):
            if i in self.relaxed_joints:
                # Don't generate trajectory for relaxed joints
                self.interpolated_positions.append([initial_positions[i]])
                self.reached_targets[i] = True
                continue
            num_steps_i = max(self.num_steps_list[i], 2)  # at least 2 steps
            start = initial_positions[i]
            end = self.target_positions[i]
            step_size = (end - start) / (num_steps_i - 1)
            interpolated = [start + step_size * j for j in range(num_steps_i)]
            self.interpolated_positions.append(interpolated)

        self.get_logger().info(f"Homing trajectories generated for {NUM_JOINTS} joints")

    # ------------------------------------------------------------------
    # Phase 2: start Fourier trajectory
    # ------------------------------------------------------------------
    def _start_fourier_phase(self):
        """Transition from homing to Fourier trajectory."""
        self.phase = "fourier"
        self.fourier_start_time = self.get_clock().now()

        # Save the final homed positions as the baseline for non-selected joints
        self.final_positions = list(self.latest_joint_state.position)

        # Save initial positions of *target* joints for soft-start blending
        self.initial_target_positions = [
            self.final_positions[j] for j in self.target_joint_indices
        ]

        self.get_logger().info(
            f"Phase 2 — Fourier trajectory started "
            f"(soft-start: {SOFT_START_DURATION:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Timer callback  (shared by both phases)
    # ------------------------------------------------------------------
    def control_loop(self):
        if self.should_exit or self.latest_joint_state is None:
            return

        if self.phase == "homing":
            self._control_homing()
        elif self.phase == "holding":
            self._control_holding()
        elif self.phase == "fourier":
            self._control_fourier()

    def _control_homing(self):
        """Phase 1: interpolate joints toward target positions.
        Joints with HOME_POS == 0 are left relaxed (kp=kd=0, no position command)."""
        joint_command = JointCommand()
        joint_command.header = Header()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.header.frame_id = ""

        joint_command.position = [0.0] * NUM_JOINTS
        joint_command.velocity = [0.0] * NUM_JOINTS
        joint_command.feed_forward_torque = [0.0] * NUM_JOINTS
        joint_command.torque = [0.0] * NUM_JOINTS
        joint_command.stiffness = list(self.kp_phase1)
        joint_command.damping = list(self.kd_phase1)

        # Relax joints that have HOME_POS == 0
        for i in self.relaxed_joints:
            joint_command.stiffness[i] = 0.0
            joint_command.damping[i] = 0.0

        all_reached = True
        for i in range(NUM_JOINTS):
            if i in self.relaxed_joints:
                # Don't control relaxed joints
                continue
            if not self.reached_targets[i]:
                if self.current_steps[i] < len(self.interpolated_positions[i]):
                    joint_command.position[i] = self.interpolated_positions[i][
                        self.current_steps[i]
                    ]
                    self.current_steps[i] += 1
                else:
                    joint_command.position[i] = self.target_positions[i]
                    self.reached_targets[i] = True
                all_reached = False
            else:
                joint_command.position[i] = self.target_positions[i]

        joint_command.velocity = [0.0] * NUM_JOINTS
        if not self.dry_run:
            self.joint_command_pub.publish(joint_command)

        if all_reached and not self.all_homed:
            self.get_logger().info(
                "All joints reached target positions! Holding briefly before Fourier..."
            )
            self.all_homed = True
            # Switch to holding phase briefly, then start Fourier
            self.phase = "holding"
            self._hold_start_time = self.get_clock().now()

    def _control_holding(self):
        """Brief hold at target position before starting Fourier.
        Joints with HOME_POS == 0 are left relaxed."""
        hold_elapsed = (
            self.get_clock().now() - self._hold_start_time
        ).nanoseconds * 1e-9

        # Publish hold command
        joint_command = JointCommand()
        joint_command.header = Header()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.header.frame_id = ""
        joint_command.position = [float(p) for p in self.target_positions]
        joint_command.velocity = [0.0] * NUM_JOINTS
        joint_command.feed_forward_torque = [0.0] * NUM_JOINTS
        joint_command.torque = [0.0] * NUM_JOINTS
        joint_command.stiffness = list(self.kp_phase1)
        joint_command.damping = list(self.kd_phase1)

        # Relax joints that have HOME_POS == 0
        for i in self.relaxed_joints:
            joint_command.stiffness[i] = 0.0
            joint_command.damping[i] = 0.0

        if not self.dry_run:
            self.joint_command_pub.publish(joint_command)

        # After a short hold (1 second), start Fourier
        if hold_elapsed >= 1.0:
            self._start_fourier_phase()

    def _control_fourier(self):
        """Phase 2: run Fourier trajectory on selected joints, hold others.
        Joints with HOME_POS == 0 that are NOT in the target group stay relaxed."""
        elapsed_sec = (
            self.get_clock().now() - self.fourier_start_time
        ).nanoseconds * 1e-9
        sample_idx = int(elapsed_sec * CONTROL_FREQUENCY) % self.total_samples

        joint_command = JointCommand()
        joint_command.header = Header()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.header.frame_id = ""

        joint_command.position = [0.0] * NUM_JOINTS
        joint_command.velocity = [0.0] * NUM_JOINTS
        joint_command.feed_forward_torque = [0.0] * NUM_JOINTS
        joint_command.torque = [0.0] * NUM_JOINTS
        joint_command.stiffness = list(self.kp_list)
        joint_command.damping = list(self.kd_list)

        # Relax joints that have HOME_POS == 0 and are NOT in the target group
        for i in self.relaxed_joints:
            if i not in self.target_joint_indices:
                joint_command.stiffness[i] = 0.0
                joint_command.damping[i] = 0.0

        # Soft-start blending factor
        alpha = min(elapsed_sec / SOFT_START_DURATION, 1.0)

        for j in range(NUM_JOINTS):
            if j in self.target_joint_indices:
                traj_idx = self.target_joint_indices.index(j)
                traj_pos = float(self.q_traj[traj_idx, sample_idx])
                traj_vel = float(self.v_traj[traj_idx, sample_idx])
                init_pos = self.initial_target_positions[traj_idx]
                joint_command.position[j] = init_pos + alpha * (traj_pos - init_pos)
                joint_command.velocity[j] = alpha * traj_vel
            else:
                # Hold at the home (target) position
                joint_command.position[j] = self.final_positions[j]
                joint_command.velocity[j] = 0.0

        if not self.dry_run:
            self.joint_command_pub.publish(joint_command)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """Wait for first joint state, generate homing trajectories, start timer."""
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

            # Generate homing trajectories from current positions to targets
            self._generate_homing_trajectories(initial_positions)

            # Start the shared control timer
            self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)

            self.get_logger().info("Homing phase started — moving to target positions")
            return True

        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def print_positions(self, title: str, positions: list):
        ss = f"\n{title}:\n["
        for i, pos in enumerate(positions):
            ss += f"{pos:.3f}"
            if i < len(positions) - 1:
                ss += ", "
                if (i + 1) % 6 == 0:
                    ss += "\n "
        ss += "]\n"
        self.get_logger().info(ss)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Home to target position, then run Fourier trajectory."
    )
    parser.add_argument(
        "--type",
        type=str,
        default="unified",
        choices=list(FourierTrajectory.VALID_TYPES.keys()),
        help="Identification type: balance, armature, friction, or unified.",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="Directly specify a YAML filename from trajectory_coefficients/.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=DEFAULT_GROUP_TO_IDENTIFY,
        help=f"Limb group to identify (default: {DEFAULT_GROUP_TO_IDENTIFY}).",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=str(DEFAULT_TARGET_CONFIG),
        help="Path to joint_test.yaml with target positions (Phase 1).",
    )
    parser.add_argument(
        "--time_coeffs",
        type=float,
        default=1.0,
        help="Time scaling coefficient for the trajectory (default: 1.0).",
    )
    parser.add_argument(
        "--dry_run",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Dry run mode: skip publishing joint commands (default: True). Use '--dry_run False' to send commands.",
    )

    parsed_args, unknown_args = parser.parse_known_args(argv)

    # Resolve Fourier YAML
    if parsed_args.yaml:
        yaml_path = FourierTrajectory._coeffs_dir / parsed_args.yaml
        if not yaml_path.is_file():
            print(f"ERROR: YAML file not found: {yaml_path}")
            return 1
        yaml_name = parsed_args.yaml
    elif parsed_args.type:
        yaml_name = FourierTrajectory.find_latest_yaml(parsed_args.type)
    else:
        parser.error("Either --type or --yaml must be specified.")

    print(f"Target config: {parsed_args.config_file}")
    print(f"Fourier YAML : {yaml_name}")
    print(f"Limb group   : {parsed_args.group}")

    print(f"Dry run      : {parsed_args.dry_run}")

    rclpy.init(args=unknown_args)
    node = FourierWithHomeNode(
        yaml_name=yaml_name,
        target_config_path=Path(parsed_args.config_file),
        group=parsed_args.group,
        time_coeffs=parsed_args.time_coeffs,
        dry_run=parsed_args.dry_run,
    )

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
