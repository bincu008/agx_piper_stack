"""Replay a recorded hardware snapshot using the production topic names."""

from __future__ import annotations

import copy
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy

from .snapshot_io import load_messages, MESSAGE_SPECS, prefixed_topic


class MockHardware(Node):
    """Publish a frozen RGB-D scene and robot feedback with fresh stamps."""

    def __init__(self) -> None:
        super().__init__('mock_hardware')
        default_snapshot = str(
            Path(get_package_share_directory('mock_hardware_node')) /
            'snapshots' / 'current')
        self.declare_parameter('snapshot_directory', default_snapshot)
        self.declare_parameter('topic_prefix', '')
        self.declare_parameter('allow_hardware_conflict', False)
        self.declare_parameter('camera_rate_hz', 10.0)
        self.declare_parameter('feedback_rate_hz', 50.0)
        self.snapshot_directory = Path(str(
            self.get_parameter('snapshot_directory').value)).expanduser()
        self.topic_prefix = str(self.get_parameter('topic_prefix').value)
        self.allow_conflict = bool(
            self.get_parameter('allow_hardware_conflict').value)
        self.messages, self.metadata = load_messages(self.snapshot_directory)
        self._mock_publishers = {}
        self.started = False
        self.camera_rate = max(
            0.5, float(self.get_parameter('camera_rate_hz').value))
        feedback_rate = max(
            1.0, float(self.get_parameter('feedback_rate_hz').value))
        self.camera_accumulator = 0.0
        self.period = 1.0 / feedback_rate
        self.startup_checks = 0
        self.timer = self.create_timer(self.period, self._tick)

    def _start_publishers(self) -> None:
        topics = {
            key: prefixed_topic(self.topic_prefix, spec[0])
            for key, spec in MESSAGE_SPECS.items()
        }
        if not self.allow_conflict:
            occupied = [
                topic for topic in topics.values()
                if self.get_publishers_info_by_topic(topic)
            ]
            if occupied:
                raise RuntimeError(
                    'real or other mock publishers already exist: ' +
                    ', '.join(occupied))
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)
        for key, (_, message_type, _) in MESSAGE_SPECS.items():
            self._mock_publishers[key] = self.create_publisher(
                message_type, topics[key], qos)
        self.started = True
        self.get_logger().info(
            'Mock hardware replay started from %s prefix=%s RGB=%sx%s cloud=%sx%s' % (
                self.snapshot_directory,
                self.topic_prefix or '<hardware topics>',
                self.metadata['rgb']['width'], self.metadata['rgb']['height'],
                self.metadata['cloud']['width'],
                self.metadata['cloud']['height']))

    @staticmethod
    def _with_stamp(message, stamp):
        replay = copy.deepcopy(message)
        if hasattr(replay, 'header'):
            replay.header.stamp = stamp
        return replay

    def _publish(self, key, stamp) -> None:
        self._mock_publishers[key].publish(
            self._with_stamp(self.messages[key], stamp))

    def _tick(self) -> None:
        if not self.started:
            # Give DDS graph discovery time before checking for real drivers.
            self.startup_checks += 1
            if self.startup_checks < max(2, int(round(1.0 / self.period))):
                return
            try:
                self._start_publishers()
            except RuntimeError as exception:
                self.get_logger().fatal(str(exception))
                self.timer.cancel()
            return
        stamp = self.get_clock().now().to_msg()
        for key in ('joint_state', 'tcp_pose', 'gripper_status', 'arm_status'):
            self._publish(key, stamp)
        self.camera_accumulator += self.period
        if self.camera_accumulator + 1.0e-9 >= 1.0 / self.camera_rate:
            self.camera_accumulator = 0.0
            for key in ('camera_info', 'rgb', 'cloud'):
                self._publish(key, stamp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockHardware()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
