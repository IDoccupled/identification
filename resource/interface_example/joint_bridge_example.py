#!/usr/bin/env python3
"""
Joint Test Example - Python version
Read YAML configuration file, generate joint trajectories and send joint commands
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header

from interface_protocol.msg import MotionState
from interface_protocol.msg import JointCommand, JointState  # type: ignore


class JointTestExample(Node):
    """Joint Test Example Node"""

    # Control frequency constant
    CONTROL_FREQUENCY = 500.0  # Hz
    CONTROL_PERIOD = 1.0 / CONTROL_FREQUENCY  # seconds

    def __init__(self, config_file_path: str):
        """
        Initialize node

        Args:
            config_file_path: Full path to configuration file
        """
        super().__init__("joint_test_example")

        # Configuration file path
        self.joint_test_config_path = config_file_path

        # Load joint test parameters
        self.load_joint_test_params()

        # Initialize state variables
        self.initial_positions = None
        self.current_joint_positions = None
        self.interpolated_positions = []
        self.current_steps = []
        self.reached_targets = []
        self.latest_joint_state = None
        self.all_reached = False
        self.last_log_time = 0.0  # For log throttling
        self.should_exit = False

        # Create subscriber - use same QoS settings as C++ version
        qos = QoSProfile(depth=3)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE

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

        # Create publisher
        self.joint_cmd_pub = self.create_publisher(
            JointCommand,
            "/hardware/joint_command",
            qos,
        )

        # Create timer
        self.timer = None

        self.get_logger().info("JointTestExample node created")

    def load_joint_test_params(self):
        """Load joint test parameters from YAML file"""
        try:
            with open(self.joint_test_config_path, "r") as f:
                config = yaml.safe_load(f)

            # Load parameters
            target_pos = config["target_position"]
            steps = config["num_steps"]
            kp = config["kp"]
            kd = config["kd"]

            # Clear vectors
            self.target_positions = []
            self.num_steps = []
            self.kp = []
            self.kd = []

            # Concatenate all groups into a single vector
            for pos_group in target_pos:
                self.target_positions.extend(pos_group)

            for steps_group in steps:
                self.num_steps.extend([int(s) for s in steps_group])

            for kp_group in kp:
                self.kp.extend(kp_group)

            for kd_group in kd:
                self.kd.extend(kd_group)

            # Print loaded parameters for verification
            self.get_logger().info("Joint test parameters loaded:")
            self.get_logger().info(f"Number of joints: {len(self.target_positions)}")

        except Exception as e:
            self.get_logger().error(f"Failed to load joint test parameters: {e}")
            raise

    def joint_state_callback(self, msg: JointState):
        """Joint state callback function"""
        self.latest_joint_state = msg

    def motion_state_callback(self, msg: MotionState):
        if msg.current_motion_task != "joint_bridge":
            self.get_logger().error("Not in joint_bridge state, exiting")
            self.should_exit = True

    def initialize(self):
        """Initialize node"""
        try:
            # Wait for first joint state
            while self.latest_joint_state is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                # Throttle log output (print at most once per second)
                current_time = time.time()
                if current_time - self.last_log_time >= 1.0:
                    self.get_logger().info("Waiting for first joint state...")
                    self.last_log_time = current_time

            # Get initial joint state
            self.initial_positions = list(self.latest_joint_state.position)

            # Print initial positions
            self.print_positions("Initial positions", self.initial_positions)

            # Generate trajectories
            self.generate_trajectories()

            # Create timer for periodic execution
            self.timer = self.create_timer(self.CONTROL_PERIOD, self.timer_callback)

            return True

        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")
            return False

    def generate_trajectories(self):
        """Generate interpolated trajectories"""
        # Convert initial motor positions to joint positions
        num_joints = len(self.initial_positions)
        self.current_joint_positions = np.array(self.initial_positions)

        # Initialize interpolation for each joint
        self.interpolated_positions = []
        self.current_steps = [0] * num_joints
        self.reached_targets = [False] * num_joints

        # Generate interpolated trajectory for each joint
        for i in range(num_joints):
            interpolated = []
            step_size = (self.target_positions[i] - self.current_joint_positions[i]) / (
                self.num_steps[i] - 1
            )
            for j in range(self.num_steps[i]):
                pos = self.current_joint_positions[i] + step_size * j
                interpolated.append(pos)
            self.interpolated_positions.append(interpolated)

        self.get_logger().info(f"Generated trajectories for {num_joints} joints")

    def print_positions(self, title: str, positions: list):
        """Print position information"""
        ss = f"\n{title}:\n["
        for i, pos in enumerate(positions):
            ss += f"{pos:.3f}"
            if i < len(positions) - 1:
                ss += ", "
                # Newline every 6 elements for readability
                if (i + 1) % 6 == 0:
                    ss += "\n "
        ss += "]\n"
        self.get_logger().info(ss)

    def timer_callback(self):
        """Timer callback function"""
        if self.all_reached:
            return

        joint_command = JointCommand()
        joint_command.header = Header()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.header.frame_id = ""

        num_joints = len(self.initial_positions)
        joint_command.position = [0.0] * num_joints
        joint_command.velocity = [0.0] * num_joints
        joint_command.feed_forward_torque = [0.0] * num_joints
        joint_command.torque = [0.0] * num_joints
        joint_command.stiffness = [400.0] * num_joints
        joint_command.damping = [5.0] * num_joints

        all_reached = True
        for i in range(num_joints):
            if not self.reached_targets[i]:
                if self.current_steps[i] < self.num_steps[i]:
                    joint_pos = self.interpolated_positions[i][self.current_steps[i]]
                    joint_command.position[i] = joint_pos
                    joint_command.velocity[i] = (
                        0.0  # Velocity is zero for position control
                    )
                    self.current_steps[i] += 1
                else:
                    # Keep at target position
                    joint_pos = self.target_positions[i]
                    joint_command.position[i] = joint_pos
                    joint_command.velocity[i] = 0.0
                    self.reached_targets[i] = True
                all_reached = False
            else:
                # Continue sending target position
                joint_pos = self.target_positions[i]
                joint_command.position[i] = joint_pos
                joint_command.velocity[i] = 0.0

        if all_reached and not self.all_reached:
            self.get_logger().info("All joints have reached target positions!")
            self.all_reached = True
            # Stop timer
            if self.timer is not None:
                self.timer.cancel()
            return

        # Publish joint command
        if not self.all_reached:
            self.joint_cmd_pub.publish(joint_command)


def main(args=None):
    """Main function"""
    default_config = (
        Path(__file__).resolve().parent / ".." / ".." / "config" / "joint_test.yaml"
    ).resolve()

    parser = argparse.ArgumentParser(
        description="Joint bridge example - Load and execute joint command queue from configuration file"
    )
    parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        default=str(default_config),
        help="Full path to joint command queue configuration file (YAML format)",
    )

    # Only parse known arguments, ignore ROS2 arguments
    parsed_args, unknown = parser.parse_known_args()
    config_file = parsed_args.config_file

    # Initialize ROS2
    rclpy.init(args=args)

    node = JointTestExample(config_file)
    if not node.initialize():
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        return 1

    try:
        # Continue running until all joints reach target positions or user interrupts
        while rclpy.ok() and not node.all_reached and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.all_reached:
            node.get_logger().info("Task completed, program exiting")
    except KeyboardInterrupt:
        node.get_logger().info("User interrupted")
    except Exception as e:
        node.get_logger().error(f"Error occurred: {e}")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
