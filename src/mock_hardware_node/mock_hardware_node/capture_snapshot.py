"""Capture one complete registered RGB-D and Piper feedback snapshot."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy

from .snapshot_io import MESSAGE_SPECS, save_messages


def select_synchronized_samples(buffers, tolerance):
    """Match buffered camera/joint samples to the newest viable cloud stamp."""
    def stamp(message):
        return message.header.stamp.sec + message.header.stamp.nanosec / 1e9

    best_span = float('inf')
    for cloud, receipt in reversed(buffers['cloud']):
        anchor = stamp(cloud)
        selected = {'cloud': (cloud, receipt)}
        for key in ('rgb', 'depth', 'joint_state'):
            selected[key] = min(buffers[key], key=lambda sample: abs(stamp(sample[0]) - anchor))
        stamps = [stamp(sample[0]) for sample in selected.values()]
        span = max(stamps) - min(stamps)
        best_span = min(best_span, span)
        if span <= tolerance:
            return selected, span
    return None, best_span


class SnapshotCapture(Node):
    """Wait for one message on each required hardware topic and save it."""

    def __init__(self) -> None:
        super().__init__('mock_hardware_snapshot_capture')
        self.declare_parameter('output_directory', '')
        self.declare_parameter('timeout_sec', 30.0)
        self.declare_parameter('synchronization_tolerance_sec', 0.05)
        output = str(self.get_parameter('output_directory').value)
        if not output:
            raise RuntimeError('output_directory parameter is required')
        self.output_directory = Path(output).expanduser().resolve()
        self.deadline = time.monotonic() + float(
            self.get_parameter('timeout_sec').value)
        self.messages = {}
        self.received_monotonic = {}
        self.buffers = {key: deque(maxlen=64 if key == "joint_state" else 6)
                        for key in MESSAGE_SPECS}
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)
        self._snapshot_subscriptions = []
        for key, (topic, message_type, _) in MESSAGE_SPECS.items():
            self._snapshot_subscriptions.append(self.create_subscription(
                message_type, topic,
                lambda message, item=key: self._callback(item, message), qos))
        self.timer = self.create_timer(0.05, self._check_complete)
        self.exit_code = 1
        self.done = False
        self.get_logger().info(
            f'Capturing hardware snapshot into {self.output_directory}')

    @staticmethod
    def _stamp(message) -> dict | None:
        if not hasattr(message, 'header'):
            return None
        return {
            'sec': int(message.header.stamp.sec),
            'nanosec': int(message.header.stamp.nanosec),
            'frame_id': str(message.header.frame_id),
        }

    def _callback(self, key, message) -> None:
        self.messages[key] = message
        self.received_monotonic[key] = time.monotonic()
        self.buffers[key].append((message, self.received_monotonic[key]))

    def _check_complete(self) -> None:
        if self.done:
            return
        missing = sorted(set(MESSAGE_SPECS) - set(self.messages))
        if not missing:
            tolerance = float(self.get_parameter('synchronization_tolerance_sec').value)
            selected, span = select_synchronized_samples(self.buffers, tolerance)
            if selected is None:
                if time.monotonic() >= self.deadline:
                    self.get_logger().error(f'Snapshot synchronization timed out: span={span:.6f}s')
                    self.done = True
                return
            for key, (message, receipt) in selected.items():
                self.messages[key] = message
                self.received_monotonic[key] = receipt
            image = self.messages['rgb']
            cloud = self.messages['cloud']
            joints = self.messages['joint_state']
            metadata = {
                'format_version': 2,
                'synchronization_span_sec': span,
                'depth': {
                    'width': int(self.messages['depth'].width),
                    'height': int(self.messages['depth'].height),
                    'encoding': str(self.messages['depth'].encoding),
                },
                'captured_unix_time': time.time(),
                'topics': {
                    key: {
                        'topic': MESSAGE_SPECS[key][0],
                        'type': MESSAGE_SPECS[key][1].__name__,
                        'stamp': self._stamp(message),
                    }
                    for key, message in self.messages.items()
                },
                'rgb': {
                    'width': int(image.width),
                    'height': int(image.height),
                    'encoding': str(image.encoding),
                },
                'cloud': {
                    'width': int(cloud.width),
                    'height': int(cloud.height),
                    'point_step': int(cloud.point_step),
                    'row_step': int(cloud.row_step),
                    'fields': [field.name for field in cloud.fields],
                },
                'joint_names': list(joints.name),
                'joint_positions': [float(value) for value in joints.position],
                'receipt_span_sec': (
                    max(self.received_monotonic.values()) -
                    min(self.received_monotonic.values())),
            }
            save_messages(self.output_directory, self.messages, metadata)
            try:
                preview = CvBridge().imgmsg_to_cv2(image, 'bgr8')
                cv2.imwrite(
                    str(self.output_directory / 'rgb_preview.png'), preview)
            except Exception as exception:  # Preview is not replay data.
                self.get_logger().warn(f'Could not write RGB preview: {exception}')
            self.get_logger().info(
                'Snapshot saved: RGB=%dx%d cloud=%dx%d joints=%d span=%.3fs' % (
                    image.width, image.height, cloud.width, cloud.height,
                    len(joints.position), metadata['receipt_span_sec']))
            self.exit_code = 0
            self.done = True
            return
        if time.monotonic() >= self.deadline:
            self.get_logger().error(
                'Snapshot capture timed out; missing: ' + ', '.join(missing))
            self.done = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SnapshotCapture()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        exit_code = node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)
