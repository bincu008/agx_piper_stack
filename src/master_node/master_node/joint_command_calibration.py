"""Apply an operator-selected joint-6 correction only at command publication."""

import copy
import json
import math

from scipy.spatial.transform import Rotation


def load_command_offset(path):
    """Require a finite raw-radian command offset in the calibration file."""
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    value = data.get('joint6_command_offset_rad')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('joint6_command_offset_rad must be a finite number in radians')
    return float(value)


def driver_joint_target(positions, offset):
    """Copy the nominal target and add the correction once, without wrapping or clipping."""
    target = [float(value) for value in positions]
    if len(target) != 6 or not all(math.isfinite(value) for value in target):
        raise ValueError('Six finite nominal joints required')
    target[5] += offset
    return target


def feedback_reference(node, expected):
    """Compare raw feedback with the last sent command when expecting its nominal target."""
    nominal = getattr(node, '_last_nominal_arm_command', None)
    sent = getattr(node, '_last_sent_arm_command', None)
    if nominal is not None and sent is not None and len(expected) == len(nominal) and all(
            abs(a-b) < 1.e-6 for a, b in zip(expected, nominal)):
        return list(sent)
    return list(expected)


def command_pose_reference(pose, offset):
    """Predict the sent joint-6 roll for endpoint verification, leaving raw pose untouched."""
    if pose is None or offset == 0.:
        return pose
    result = copy.deepcopy(pose)
    q = pose.pose.orientation
    rotation = Rotation.from_quat([q.x, q.y, q.z, q.w]) * Rotation.from_rotvec([0., 0., offset])
    q = rotation.as_quat()
    result.pose.orientation.x, result.pose.orientation.y = float(q[0]), float(q[1])
    result.pose.orientation.z, result.pose.orientation.w = float(q[2]), float(q[3])
    return result
