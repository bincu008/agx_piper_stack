"""Manual Piper FK and calibrated camera transforms (no TF lookup)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _urdf_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def load_hand_eye_matrix(path: str) -> np.ndarray:
    """Load and validate the calibrated end-effector-to-camera X matrix."""
    calibration_path = Path(path).expanduser()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"hand-eye calibration not found: {calibration_path}")
    with calibration_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    matrix = np.asarray(data.get("hand_eye_transform"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("hand_eye_transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("hand_eye_transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("hand_eye_transform rotation determinant is not +1")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("hand_eye_transform has an invalid homogeneous row")
    return matrix


def piper_base_to_camera(joints, hand_eye_matrix: np.ndarray) -> np.ndarray:
    """Match ActiveVisionServer's FK(joints) * calibrated-X calculation."""
    q = np.asarray(joints, dtype=np.float64).reshape(-1)
    if q.size < 6 or not np.isfinite(q[:6]).all():
        raise ValueError("six finite Piper joint positions are required")

    transforms = [np.eye(4, dtype=np.float64) for _ in range(6)]
    transforms[0][2, 3] = 0.123
    transforms[0][:3, :3] = _urdf_rotation(0.0, 0.0, 0.0) @ _rotation_z(q[0])
    transforms[1][:3, :3] = (
        _urdf_rotation(1.5707963, -0.1357866, -3.1415926) @ _rotation_z(q[1]))
    transforms[2][0, 3] = 0.28503
    transforms[2][:3, :3] = (
        _urdf_rotation(0.0, 0.0, -1.7938494) @ _rotation_z(q[2]))
    transforms[3][0, 3] = -0.02198
    transforms[3][1, 3] = -0.25075
    transforms[3][:3, :3] = (
        _urdf_rotation(1.5707963, 0.0, 0.0) @ _rotation_z(q[3]))
    transforms[4][:3, :3] = (
        _urdf_rotation(-1.5707963, 0.0, 0.0) @ _rotation_z(q[4]))
    transforms[5][0, 3] = 8.8259e-05
    transforms[5][1, 3] = -0.091
    transforms[5][:3, :3] = (
        _urdf_rotation(1.5707963, 0.0, 0.0) @ _rotation_z(q[5]))

    base_to_end = np.eye(4, dtype=np.float64)
    for transform in transforms:
        base_to_end = base_to_end @ transform
    return base_to_end @ np.asarray(hand_eye_matrix, dtype=np.float64)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform row-vector XYZ points with a homogeneous transform."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (xyz @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def transform_pose(rotation: np.ndarray, translation: np.ndarray,
                   transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform an object pose from camera coordinates into base coordinates."""
    base_rotation = transform[:3, :3] @ np.asarray(rotation, dtype=np.float64)
    base_translation = (
        transform[:3, :3] @ np.asarray(translation, dtype=np.float64)
        + transform[:3, 3])
    return base_rotation.astype(np.float32), base_translation.astype(np.float32)


def pose_matrix(position, quaternion) -> np.ndarray:
    """Build a finite rigid pose from XYZ and XYZW components."""
    xyz = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion, dtype=np.float64)
    if xyz.shape != (3,) or q.shape != (4,) or not np.isfinite(xyz).all() or not np.isfinite(q).all():
        raise ValueError("pose components must be finite XYZ and XYZW")
    length = np.linalg.norm(q)
    if length < 1e-8:
        raise ValueError("pose quaternion is zero")
    x, y, z, w = q / length
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]
    result[:3, 3] = xyz
    return result


def attached_object_delta(source_grasp, placement_grasp) -> np.ndarray:
    """Keep the original object-to-gripper transform fixed at placement."""
    return np.asarray(placement_grasp) @ np.linalg.inv(np.asarray(source_grasp))
