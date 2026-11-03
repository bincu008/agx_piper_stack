"""Consolidated calibration for camera scaling, grasp offsets, and joint commands."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import List

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose
from my_actions.srv import CaptureObservation, GroundObject, SegmentGroundedImage
import numpy as np
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.serialization import serialize_message
from scipy.spatial.transform import Rotation
from sensor_msgs_py import point_cloud2

from task_pipeline.registry import endpoint


def load_metric_ratio(path: str) -> float:
    """Load and validate perception_measurement_ratio from calibration json."""
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    ratio = data.get('perception_measurement_ratio')
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError('Calibration requires a numeric perception_measurement_ratio')
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('perception_measurement_ratio must be finite and positive')
    return float(ratio)


def load_grasp_offset(path: str) -> List[float]:
    """Read a Cartesian command correction expressed in base_link metres."""
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    if data.get('frame_id') != 'base_link':
        raise ValueError('Grasp offset must be expressed in base_link')
    values = data.get('grasp_position_offset_base_m')
    if not isinstance(values, list) or len(values) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values):
        raise ValueError('grasp_position_offset_base_m requires three finite metres')
    return [float(v) for v in values]


def load_command_offset(path: str) -> float:
    """Require a finite raw-radian command offset in the calibration file."""
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    value = data.get('joint6_command_offset_rad')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('joint6_command_offset_rad must be a finite number in radians')
    return float(value)


def driver_joint_target(positions: List[float], offset: float) -> List[float]:
    """Copy nominal target and add correction once, without wrapping or clipping."""
    target = [float(value) for value in positions]
    if len(target) != 6 or not all(math.isfinite(value) for value in target):
        raise ValueError('Six finite nominal joints required')
    target[5] += offset
    return target


def feedback_reference(node, expected: List[float]) -> List[float]:
    """Compare raw feedback with last sent command when expecting nominal target."""
    nominal = getattr(node, '_last_nominal_arm_command', None)
    sent = getattr(node, '_last_sent_arm_command', None)
    if nominal is not None and sent is not None and len(expected) == len(nominal) and all(
            abs(a - b) < 1.e-6 for a, b in zip(expected, nominal)):
        return list(sent)
    return list(expected)


def command_pose_reference(pose, offset: float):
    """Predict sent joint-6 roll for endpoint verification, leaving raw pose untouched."""
    if pose is None or offset == 0.:
        return pose
    result = copy.deepcopy(pose)
    q = pose.pose.orientation
    rotation = Rotation.from_quat([q.x, q.y, q.z, q.w]) * Rotation.from_rotvec([0., 0., offset])
    quat = rotation.as_quat()
    result.pose.orientation.x, result.pose.orientation.y = float(quat[0]), float(quat[1])
    result.pose.orientation.z, result.pose.orientation.w = float(quat[2]), float(quat[3])
    return result


def pose_matrix(pose):
    """Use the planner's base-frame pose directly without consulting TF."""
    q = pose.orientation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def transform_pose(matrix, pose):
    """Apply a rigid transform to an existing planner finger pose."""
    result = matrix @ pose_matrix(pose)
    output = Pose()
    output.position = Point(**dict(zip(('x', 'y', 'z'), result[:3, 3].tolist())))
    q = Rotation.from_matrix(result[:3, :3]).as_quat()
    output.orientation.x, output.orientation.y, output.orientation.z, output.orientation.w = q
    return output


def measure_plate(camera_xyz, mask, diameter_m, current_ratio):
    """Fit rim plane robustly, intersect silhouette rays, measure planar widths."""
    if diameter_m <= 0 or current_ratio <= 0:
        raise ValueError('Positive diameter and current ratio required')
    valid = np.isfinite(camera_xyz).all(axis=2) & (camera_xyz[:, :, 2] > .05)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('No segmented plate')
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if (contour[:, 0].min() <= 0 or contour[:, 1].min() <= 0 or
            contour[:, 0].max() >= mask.shape[1] - 1 or
            contour[:, 1].max() >= mask.shape[0] - 1):
        raise ValueError('Plate silhouette clipped by image boundary')
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    points = camera_xyz[(distances >= 3) & (distances <= 8) & valid]
    if len(points) < 100:
        raise ValueError('Insufficient valid rim samples')
    rng, best = np.random.default_rng(1234), None
    for _ in range(300):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1.e-10:
            continue
        normal /= length
        selected = np.abs((points - a) @ normal) < .0015
        if best is None or selected.sum() > best.sum():
            best = selected
    if best is None or best.mean() < .65:
        raise ValueError('Rim plane unreliable; do not change calibration')
    center = points[best].mean(axis=0)
    _, _, basis = np.linalg.svd(points[best] - center, full_matrices=False)
    normal = basis[2]
    rays = camera_xyz[contour[:, 1], contour[:, 0]]
    usable = np.isfinite(rays).all(axis=1) & (rays[:, 2] > .05)
    if usable.mean() < .9:
        raise ValueError('Insufficient valid silhouette rays')
    rays = rays[usable] / rays[usable, 2, None]
    denominator = rays @ normal
    if np.min(np.abs(denominator)) < .1:
        raise ValueError('Plate plane too oblique')
    rim = rays * ((center @ normal) / denominator)[:, None]
    xy = (rim - center) @ basis[:2].T
    angles = np.arange(0., 180., 15.)
    directions = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    widths = np.ptp(xy @ directions.T, axis=0)
    if np.ptp(widths) / np.median(widths) > .05:
        raise ValueError('Directional spread exceeds 5%; uniform scale not justified')
    raw_widths = widths * current_ratio
    axis_widths = {}
    for axis, name in enumerate(('x', 'y', 'z')):
        projected = basis[:2, axis]
        length = np.linalg.norm(projected)
        axis_widths[name] = (float(np.ptp(xy @ (projected / length)))
                             if length > .1 else None)
    return {
        'angles_degrees': angles.tolist(),
        'diameters_current_m': widths.tolist(),
        'camera_axis_projected_direction_diameters_m': axis_widths,
        'diameters_unscaled_m': raw_widths.tolist(),
        'perception_measurement_ratio': float(np.median(raw_widths) / diameter_m),
        'rim_plane_inlier_fraction': float(best.mean()),
        'camera_plane_basis': basis[:2].tolist(),
        'camera_plane_normal': normal.tolist(),
        'axis_calibration_independent': False,
        'method': 'rim-plane RANSAC and silhouette-ray planar Feret widths',
    }


def calibrate_plate_main():
    """Calculate calibration without commanding the arm or changing live parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--diameter', type=float, default=.174)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 20:
        parser.error('samples must be between 1 and 20')
    rclpy.init()
    node = rclpy.create_node('plate_scale_calibration')
    bridge = CvBridge()
    directory = args.output.parent / (args.output.stem + '_captures')
    directory.mkdir(parents=True, exist_ok=True)

    def call(schema, address, request):
        client = node.create_client(schema, address)
        try:
            if not client.wait_for_service(timeout_sec=15):
                raise RuntimeError('Unavailable: ' + address)
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=130)
            if not future.done():
                raise RuntimeError('Timeout: ' + address)
            response = future.result()
            if hasattr(response, 'success') and not response.success:
                raise RuntimeError(response.status)
            return response
        finally:
            node.destroy_client(client)

    try:
        params = call(GetParameters, '/active_vision_server/get_parameters',
                      GetParameters.Request(names=['perception_measurement_ratio']))
        ratio = params.values[0].double_value
        frames = []
        for index in range(args.samples):
            obs = call(CaptureObservation, endpoint('capture_observation'),
                       CaptureObservation.Request())
            ground = call(GroundObject, endpoint('ground_object'), GroundObject.Request(
                image=obs.image, target='the entire round plate, including its outer rim',
                verify_presence=False))
            segmented = call(SegmentGroundedImage, endpoint('segment_object'),
                             SegmentGroundedImage.Request(image=obs.image,
                                                          bbox=ground.pixel_box))
            prefix = directory / f'frame_{index:02d}'
            prefix.with_suffix('.cdr').write_bytes(serialize_message(obs))
            cv2.imwrite(str(prefix.with_suffix('.png')),
                        bridge.imgmsg_to_cv2(obs.image, 'bgr8'))
            mask = bridge.imgmsg_to_cv2(segmented.mask).astype(bool)
            cv2.imwrite(str(directory / f'mask_{index:02d}.png'),
                        mask.astype(np.uint8) * 255)
            world = point_cloud2.read_points_numpy(
                obs.cloud, field_names=('x', 'y', 'z'), skip_nans=False).reshape(
                    obs.cloud.height, obs.cloud.width, 3)
            transform = obs.camera_transform.transform
            rotation = Rotation.from_quat([transform.rotation.x, transform.rotation.y,
                                           transform.rotation.z, transform.rotation.w]).as_matrix()
            translation = np.array([transform.translation.x, transform.translation.y,
                                    transform.translation.z])
            camera = (world - translation) @ rotation
            measured = measure_plate(camera, mask, args.diameter, ratio)
            measured['source_stamp'] = {'sec': obs.cloud.header.stamp.sec,
                                        'nanosec': obs.cloud.header.stamp.nanosec}
            measured['grounding_response'] = ground.text
            frames.append(measured)
            np.savez_compressed(directory / f'plate_{index:02d}.npz',
                                camera_xyz=camera, mask=mask)
            print(index + 1, measured['perception_measurement_ratio'], flush=True)
        ratios = np.array([f['perception_measurement_ratio'] for f in frames])
        if np.ptp(ratios) / np.median(ratios) > .02:
            raise RuntimeError('Capture ratios vary by more than 2%; calibration not saved')
        result = {'schema_version': 1,
                  'perception_measurement_ratio': float(np.median(ratios)),
                  'known_diameter_m': args.diameter, 'previous_ratio': ratio,
                  'status': 'plate_in_plane_uniform_scale; absolute_depth_unverified',
                  'axis_calibration_independent': False, 'frames': frames}
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print('Saved:', args.output, flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
