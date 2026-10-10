"""Message definitions and serialization helpers for hardware snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from agx_arm_msgs.msg import AgxArmStatus, GripperStatus
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2


MESSAGE_SPECS = {
    'rgb': ('/camera/color/image_raw', Image, 'rgb_image.cdr'),
    'depth': ('/camera/depth/image_raw', Image, 'depth_image.cdr'),
    'cloud': (
        '/camera/depth_registered/points', PointCloud2,
        'registered_cloud.cdr'),
    'camera_info': (
        '/camera/color/camera_info', CameraInfo, 'camera_info.cdr'),
    'joint_state': (
        '/feedback/joint_states', JointState, 'joint_state.cdr'),
    'tcp_pose': ('/feedback/tcp_pose', PoseStamped, 'tcp_pose.cdr'),
    'gripper_status': (
        '/feedback/gripper_status', GripperStatus, 'gripper_status.cdr'),
    'arm_status': (
        '/feedback/arm_status', AgxArmStatus, 'arm_status.cdr'),
}


def save_messages(directory: Path, messages: dict, metadata: dict) -> None:
    """Serialize a complete snapshot without changing ROS message content."""
    directory.mkdir(parents=True, exist_ok=True)
    for key, (_, _, filename) in MESSAGE_SPECS.items():
        if key == 'depth' and key not in messages:
            (directory / filename).unlink(missing_ok=True)
            continue
        (directory / filename).write_bytes(bytes(serialize_message(messages[key])))
    (directory / 'manifest.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')


def load_messages(directory: Path) -> tuple[dict, dict]:
    """Load and deserialize every message in a snapshot directory."""
    messages = {}
    for key, (_, message_type, filename) in MESSAGE_SPECS.items():
        path = directory / filename
        if key == 'depth' and not path.is_file():
            continue  # Older snapshots contain registered depth only.
        if not path.is_file():
            raise FileNotFoundError(f'missing snapshot message: {path}')
        messages[key] = deserialize_message(path.read_bytes(), message_type)
    manifest_path = directory / 'manifest.json'
    metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
    return messages, metadata


def prefixed_topic(prefix: str, topic: str) -> str:
    """Apply an optional namespace prefix to an absolute topic."""
    clean = prefix.strip('/')
    return topic if not clean else '/' + clean + topic
