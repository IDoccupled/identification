#!/usr/bin/env python3
"""
Publish per-joint sine-wave commands from YAML.
"""

import math
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header
import yaml

from interface_protocol.msg import JointCommand, JointState  # type: ignore


CONFIG_PATH = (
    Path(__file__).resolve().parent / ".." / "config" / "pm01" / "joint_sine.yaml"
).resolve()


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


class JointSineExample(Node):
    def __init__(self, config):
        super().__init__("joint_sine_example")

        sine_cfg = config.get("sine", config)

        self.num_joints = int(sine_cfg.get("num_joints", config.get("num_joints", 24)))

        amplitude_cfg = _flatten_groups(sine_cfg.get("target_position", sine_cfg.get("amplitude")))
        frequency_cfg = _flatten_groups(sine_cfg.get("frequency"))
        phase_cfg = _flatten_groups(sine_cfg.get("phase"))
        offset_cfg = _flatten_groups(sine_cfg.get("offset"))

        self.amplitude_list = _expand_or_default(amplitude_cfg, self.num_joints, 0.0)
        self.frequency_list = _expand_or_default(frequency_cfg, self.num_joints, 0.0)
        self.phase_list = _expand_or_default(phase_cfg, self.num_joints, 0.0)
        self.offset_list = _expand_or_default(offset_cfg, self.num_joints, 0.0)

        self.amplitude_list = _require_list("sine.target_position", self.amplitude_list, self.num_joints)
        self.frequency_list = _require_list("sine.frequency", self.frequency_list, self.num_joints)
        self.phase_list = _require_list("sine.phase", self.phase_list, self.num_joints)
        self.offset_list = _require_list("sine.offset", self.offset_list, self.num_joints)
        self.amplitude_list = _to_float_list("sine.target_position", self.amplitude_list)
        self.frequency_list = _to_float_list("sine.frequency", self.frequency_list)
        self.phase_list = _to_float_list("sine.phase", self.phase_list)
        self.offset_list = _to_float_list("sine.offset", self.offset_list)

        transition_cfg = sine_cfg.get("transition", {})
        self.transition_enabled = bool(transition_cfg.get("enabled", True))
        self.transition_center_time = float(transition_cfg.get("center_time", 1.0))
        self.transition_sharpness = float(transition_cfg.get("sharpness", 4.0))
        self.transition_settle_time = float(transition_cfg.get("settle_time", 3.0))

        self.initial_positions = None
        self.latest_joint_state = None
        self.last_wait_log_time = self.get_clock().now()

        kp_cfg = _flatten_groups(config.get("kp"))
        kd_cfg = _flatten_groups(config.get("kd"))
        self.kp_list = _expand_or_default(kp_cfg, self.num_joints, 100.0)
        self.kd_list = _expand_or_default(kd_cfg, self.num_joints, 1.0)
        self.kp_list = _require_list("kp", self.kp_list, self.num_joints)
        self.kd_list = _require_list("kd", self.kd_list, self.num_joints)
        self.kp_list = _to_float_list("kp", self.kp_list)
        self.kd_list = _to_float_list("kd", self.kd_list)

        command_topic = sine_cfg.get("command_topic", config.get("command_topic", "/hardware/joint_command"))
        self.pub = self.create_publisher(JointCommand, command_topic, 10)

        qos = QoSProfile(depth=3)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/hardware/joint_state",
            self.joint_state_callback,
            qos,
        )

        self.start_time = self.get_clock().now()

        publish_rate = float(sine_cfg.get("publish_rate", config.get("publish_rate", 500.0)))
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)
        self.get_logger().info(f"Publishing sine on all {self.num_joints} joints from YAML")

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg
        if self.initial_positions is None and len(msg.position) >= self.num_joints:
            self.initial_positions = [float(p) for p in msg.position[: self.num_joints]]
            self.get_logger().info("Captured initial joint positions for smooth transition")

    def transition_weight(self, t: float) -> float:
        if not self.transition_enabled:
            return 1.0
        if t >= self.transition_settle_time:
            return 1.0
        return 0.5 * (1.0 + math.tanh(self.transition_sharpness * (t - self.transition_center_time)))

    def timer_callback(self):
        if self.initial_positions is None:
            now = self.get_clock().now()
            if (now - self.last_wait_log_time).nanoseconds > 1_000_000_000:
                self.get_logger().info("Waiting for /hardware/joint_state to capture initial positions...")
                self.last_wait_log_time = now
            return

        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9
        w = self.transition_weight(t)
        msg = JointCommand()
        msg.header = Header()
        msg.header.stamp = now.to_msg()

        msg.position = [0.0] * self.num_joints
        msg.velocity = [0.0] * self.num_joints
        msg.feed_forward_torque = [0.0] * self.num_joints
        msg.torque = [0.0] * self.num_joints
        msg.stiffness = self.kp_list
        msg.damping = self.kd_list

        for i in range(self.num_joints):
            phase = self.phase_list[i]
            sine_target = self.offset_list[i] + self.amplitude_list[i] * math.sin(
                2.0 * math.pi * self.frequency_list[i] * t + phase
            )
            msg.position[i] = self.initial_positions[i] + w * (sine_target - self.initial_positions[i])

        self.pub.publish(msg)


def main(argv=None):
    config_path = CONFIG_PATH
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    rclpy.init(args=argv)
    node = JointSineExample(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
