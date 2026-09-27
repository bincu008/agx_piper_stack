"""ROS 2 node exposing isolated AnyPlace placement inference for RViz."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import threading
import time
from collections import deque
from urllib import error as url_error
from urllib import request as url_request

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header, String
from std_srvs.srv import Trigger
from my_actions.msg import PlacementCandidate as PlacementCandidateMessage
from my_actions.srv import FindPlacementCandidates
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from .inference import infer_placements, make_dummy_box_cloud
from .model import load_anyplace_checkpoint
from .packing import search_heightmap_placements
from .perception import (
    construct_support_bottom, estimate_support_translation,
    extract_destination_cloud, parse_grounding_box)
from .transforms import (
    load_hand_eye_matrix, piper_base_to_camera, transform_points,
    transform_pose)


class AnyPlaceNode(Node):
    """Ground a destination and publish AnyPlace candidate poses without motion."""

    def __init__(self):
        super().__init__("anyplace_cpu_node")
        default_checkpoint = ""
        parameters = {
            "checkpoint_path": default_checkpoint,
            "rgb_topic": "/camera/color/image_raw",
            "point_cloud_topic": "/camera/depth_registered/points",
            "joint_state_topic": "/feedback/joint_states",
            "joint_sync_tolerance_sec": 0.20,
            "hand_eye_calibration_path": "",
            "api_server_url": "http://10.211.55.2:1234",
            "model_name": "robobrain2.5-4b",
            "grounding_target": "the entire visible plate surface",
            "request_timeout_sec": 20.0,
            "sync_tolerance_sec": 0.20,
            "candidate_count": 8,
            "refinement_steps": 50,
            "placement_backend": "heightmap",
            "random_seed": 13,
            "surface_neighbor_distance_m": 0.02,
            "minimum_destination_points": 80,
            "dummy_object_dimensions_m": [0.04, 0.03, 0.02],
            "dummy_object_point_count": 200,
            "enable_object_bottom_completion": True,
            "bottom_support_context_margin_m": 0.04,
            "bottom_source_observation_tolerance_sec": 30.0,
            "bottom_support_plane_distance_m": 0.004,
            "bottom_support_maximum_tilt_deg": 15.0,
            "bottom_footprint_band_m": 0.025,
            "bottom_grid_spacing_m": 0.004,
            "bottom_maximum_points": 1500,
            "enable_upright_support_alignment": True,
            "placement_support_grid_size_m": 0.004,
            "placement_support_lookup_radius_m": 0.015,
            "placement_support_quantile": 0.95,
            "placement_support_clearance_m": 0.002,
            "placement_support_maximum_shift_m": 0.15,
            "packing_grid_resolution_m": 0.005,
            "packing_upright_yaw_step_deg": 15.0,
            "packing_free_yaw_step_deg": 30.0,
            "packing_tilt_angles_deg": [-20.0, -10.0, 0.0, 10.0, 20.0],
            "packing_clearance_m": 0.002,
            "packing_contact_tolerance_m": 0.006,
            "packing_minimum_support_area_m2": 0.0002,
            "packing_support_margin_m": 0.002,
            "packing_maximum_pile_height_m": 0.30,
        }
        for name, value in parameters.items():
            self.declare_parameter(name, value)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._recent_images = deque(maxlen=20)
        self._recent_clouds = deque(maxlen=5)
        self._recent_joints = deque(maxlen=200)
        self._observation_clouds = deque(maxlen=8)
        self._cached_object_cloud = None
        self._cached_completed_object_points = None
        self._cached_bottom_points = None
        self._cached_collision_envelope_points = None
        self._completed_object_stamp = None
        self._bottom_attempt_stamp = None
        self._model = None
        self._model_path = None
        self._hand_eye_matrix = load_hand_eye_matrix(
            self.get_parameter("hand_eye_calibration_path").value)

        transient = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._destination_pub = self.create_publisher(
            PointCloud2, "/anyplace/destination_cloud", transient)
        self._object_pub = self.create_publisher(
            PointCloud2, "/anyplace/best_placed_object", transient)
        self._completed_object_pub = self.create_publisher(
            PointCloud2, "/anyplace/completed_object_cloud", transient)
        self._bottom_pub = self.create_publisher(
            PointCloud2, "/anyplace/artificial_bottom_cloud", transient)
        self._marker_pub = self.create_publisher(
            MarkerArray, "/anyplace/placement_candidates", transient)
        self._debug_pub = self.create_publisher(
            Image, "/anyplace/grounding_debug", transient)
        self._status_pub = self.create_publisher(
            String, "/anyplace/status", transient)
        self.create_subscription(
            Image, self.get_parameter("rgb_topic").value,
            self._image_callback, qos_profile_sensor_data)
        self.create_subscription(
            PointCloud2, self.get_parameter("point_cloud_topic").value,
            self._cloud_callback, qos_profile_sensor_data)
        self.create_subscription(
            JointState, self.get_parameter("joint_state_topic").value,
            self._joint_callback, 50)
        self.create_subscription(
            PointCloud2, "/active_vision/observation_cloud",
            self._observation_cloud_callback, transient)
        self.create_subscription(
            PointCloud2, "/vamp_planner/segmented_target_cloud",
            self._object_cloud_callback, transient)
        self.create_service(Trigger, "/anyplace/infer_placement", self._infer_callback)
        self.create_service(
            FindPlacementCandidates,
            "/anyplace/find_placement_candidates",
            self._placement_callback)
        self.get_logger().info(
            "Placement node ready (backend=%s; manual FK*X output in "
            "base_link; no TF lookup and no motion publishers). "
            "Call /anyplace/infer_placement after RGB and registered depth "
            "arrive."
            % self.get_parameter("placement_backend").value)

    def _observation_cloud_callback(self, message):
        if message.height <= 1 or message.header.frame_id != "base_link":
            return
        with self._lock:
            self._observation_clouds.append(message)
        self._try_complete_cached_object()

    def _object_cloud_callback(self, message):
        if message.header.frame_id != "base_link" or message.width == 0:
            return
        with self._lock:
            self._cached_object_cloud = message
            self._cached_completed_object_points = None
            self._cached_bottom_points = None
            self._cached_collision_envelope_points = None
            self._completed_object_stamp = None
            self._bottom_attempt_stamp = None
        self.get_logger().info(
            f"[ANYPLACE] Cached segmented object cloud: {message.width} points")
        self._try_complete_cached_object()

    @staticmethod
    def _stamp_key(header):
        return int(header.stamp.sec), int(header.stamp.nanosec)

    def _try_complete_cached_object(self):
        """Construct and publish one source-support bottom for a cached object."""
        if not bool(self.get_parameter("enable_object_bottom_completion").value):
            return
        with self._lock:
            object_message = self._cached_object_cloud
            observations = list(self._observation_clouds)
            completed_stamp = self._completed_object_stamp
            attempt_stamp = self._bottom_attempt_stamp
        if object_message is None:
            return
        stamp = self._stamp_key(object_message.header)
        if completed_stamp == stamp or attempt_stamp == stamp:
            return
        if not observations:
            return
        object_time = self._stamp_seconds(object_message.header)
        observation = min(
            observations,
            key=lambda item: abs(
                self._stamp_seconds(item.header) - object_time))
        observation_offset = abs(
            self._stamp_seconds(observation.header) - object_time)
        observation_tolerance = float(self.get_parameter(
            "bottom_source_observation_tolerance_sec").value)
        if observation_offset > observation_tolerance:
            return
        with self._lock:
            if self._cached_object_cloud is not object_message:
                return
            self._bottom_attempt_stamp = stamp
        try:
            object_points = self._cloud_xyz(object_message)
            scene_points = self._cloud_xyz(observation)
            completed, bottom, collision_envelope, diagnostics = (
                construct_support_bottom(
                    object_points, scene_points,
                    float(self.get_parameter(
                        "bottom_support_context_margin_m").value),
                    float(self.get_parameter(
                        "bottom_support_plane_distance_m").value),
                    float(self.get_parameter(
                        "bottom_support_maximum_tilt_deg").value),
                    float(self.get_parameter(
                        "bottom_footprint_band_m").value),
                    float(self.get_parameter("bottom_grid_spacing_m").value),
                    int(self.get_parameter("bottom_maximum_points").value)))
            header = Header()
            header.stamp = object_message.header.stamp
            header.frame_id = "base_link"
            completed_message = point_cloud2.create_cloud_xyz32(
                header, completed.tolist())
            bottom_message = point_cloud2.create_cloud_xyz32(
                header, bottom.tolist())
            with self._lock:
                if (self._cached_object_cloud is object_message and
                        self._completed_object_stamp is None):
                    self._cached_completed_object_points = completed
                    self._cached_bottom_points = bottom
                    self._cached_collision_envelope_points = collision_envelope
                    self._completed_object_stamp = stamp
                else:
                    return
            self._completed_object_pub.publish(completed_message)
            self._bottom_pub.publish(bottom_message)
            normal = diagnostics["plane_normal"]
            self.get_logger().info(
                "[ANYPLACE FOOTPRINT] contour=%d envelope=%d model=%s "
                "support_inliers=%d footprint_samples=%d area=%.1fcm2 "
                "radius=%.1fmm source_offset=%.3fs normal=(%.3f,%.3f,%.3f)"
                % (diagnostics["bottom_points"],
                   diagnostics["collision_envelope_points"],
                   diagnostics["footprint_model"], diagnostics["support_points"],
                   diagnostics["footprint_points"],
                   diagnostics["footprint_area_m2"] * 1.0e4,
                   diagnostics["fitted_radius_m"] * 1000.0,
                   observation_offset,
                   normal[0], normal[1], normal[2]))
        except Exception as exception:
            self.get_logger().warn(
                f"[ANYPLACE BOTTOM] completion unavailable: {exception}")

    def _image_callback(self, message):
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exception:
            self.get_logger().error(f"RGB decode failed: {exception}")
            return
        with self._lock:
            self._recent_images.append((message.header, image.copy()))

    def _cloud_callback(self, message):
        if message.height <= 1:
            self.get_logger().warn(
                "AnyPlace requires an organized registered cloud; received height <= 1",
                throttle_duration_sec=5.0)
            return
        with self._lock:
            self._recent_clouds.append(message)

    def _joint_callback(self, message):
        if len(message.position) < 6:
            return
        positions = None
        names = list(message.name)
        if names and all(f"joint{index}" in names for index in range(1, 7)):
            positions = [
                message.position[names.index(f"joint{index}")]
                for index in range(1, 7)]
        elif len(message.position) >= 6:
            positions = list(message.position[:6])
        if positions is None or not np.isfinite(positions).all():
            return
        with self._lock:
            self._recent_joints.append((message.header, np.asarray(positions)))

    @staticmethod
    def _stamp_seconds(header):
        return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9

    def _load_model(self):
        path = os.path.expanduser(self.get_parameter("checkpoint_path").value)
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"checkpoint not found: {path}; run ros2 run anyplace_ros "
                "download_anyplace_checkpoint")
        if self._model is None or self._model_path != path:
            started = time.perf_counter()
            self._model = load_anyplace_checkpoint(path, device="cpu")
            self._model_path = path
            self.get_logger().info(
                f"Loaded AnyPlace low-level checkpoint in "
                f"{time.perf_counter() - started:.2f}s: {path}")
        return self._model

    def _call_robobrain(self, image: np.ndarray, target: str) -> tuple[str, tuple[int, int, int, int]]:
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise RuntimeError("failed to JPEG-encode RGB frame")
        prompt = (
            "Please provide one tight bounding box enclosing the complete region "
            f"described here: {target}. Use normalized image coordinates from 0 "
            "to 1000 and respond with exactly [x1, y1, x2, y2].")
        payload = {
            "model": self.get_parameter("model_name").value,
            "temperature": 0.1,
            "max_tokens": 32,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url":
                        "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")}},
                ],
            }],
        }
        base_url = self.get_parameter("api_server_url").value.rstrip("/")
        request = url_request.Request(
            base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with url_request.urlopen(
                    request, timeout=float(self.get_parameter("request_timeout_sec").value)) as result:
                response = json.loads(result.read().decode("utf-8"))
        except (url_error.URLError, TimeoutError, json.JSONDecodeError) as exception:
            raise RuntimeError(f"RoboBrain request failed: {exception}") from exception
        try:
            reply = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exception:
            raise RuntimeError(f"unexpected RoboBrain response: {response}") from exception
        box = parse_grounding_box(str(reply), image.shape[1], image.shape[0])
        self.get_logger().info(f'[ROBOBRAIN] target="{target}" response={reply}')
        return str(reply), box

    def _publish_debug(self, image, bbox, foreground, cloud_box, header):
        debug = image.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Overlay the selected depth component inside the RGB grounding box.
        _ = cloud_box
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        resized = cv2.resize(
            foreground.astype(np.uint8) * 255, (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST)
        mask[y1:y2, x1:x2] = resized
        overlay = debug.copy()
        overlay[mask > 0] = (0, 180, 255)
        debug = cv2.addWeighted(debug, 0.65, overlay, 0.35, 0)
        message = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        message.header = header
        self._debug_pub.publish(message)

    @staticmethod
    def _point(xyz) -> Point:
        point = Point()
        point.x, point.y, point.z = map(float, xyz)
        return point

    @staticmethod
    def _quaternion_from_rotation(rotation):
        """Convert a proper 3x3 rotation matrix to ROS xyzw order."""
        matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.array([
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale])
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                quaternion = np.array([
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale])
            elif axis == 1:
                scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                quaternion = np.array([
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale])
            else:
                scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                quaternion = np.array([
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale])
        quaternion /= np.linalg.norm(quaternion)
        return quaternion

    def _publish_candidates(self, header, candidates, object_points, dimensions):
        _ = dimensions
        markers = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)
        palette = [
            (0.15, 0.85, 1.0), (1.0, 0.45, 0.1), (0.7, 0.25, 1.0),
            (0.2, 1.0, 0.35), (1.0, 0.85, 0.1), (1.0, 0.2, 0.6)]
        axis_colors = [ColorRGBA(r=1.0, a=1.0), ColorRGBA(g=1.0, a=1.0), ColorRGBA(b=1.0, a=1.0)]
        axis_length = 0.04
        for index, candidate in enumerate(candidates):
            color = palette[index % len(palette)]
            candidate_namespace = f"placement_candidate_{index + 1}"
            box = Marker()
            box.header = header
            box.ns = candidate_namespace
            box.id = 0
            box.type = Marker.POINTS
            box.action = Marker.ADD
            box.pose.orientation.w = 1.0
            box.scale.x = 0.0035
            box.scale.y = 0.0035
            box.color = ColorRGBA(
                r=color[0], g=color[1], b=color[2], a=0.45)
            displayed_object = (
                object_points @ candidate.rotation.T + candidate.translation)
            if len(displayed_object) > 1500:
                displayed_object = displayed_object[np.linspace(
                    0, len(displayed_object) - 1, 1500, dtype=np.int32)]
            box.points = [self._point(point) for point in displayed_object]
            markers.markers.append(box)

            axes = Marker()
            axes.header = header
            axes.ns = candidate_namespace
            axes.id = 1
            axes.type = Marker.LINE_LIST
            axes.action = Marker.ADD
            axes.scale.x = 0.0025
            for axis in range(3):
                axes.points.extend((
                    self._point(candidate.translation),
                    self._point(candidate.translation + candidate.rotation[:, axis] * axis_length)))
                axes.colors.extend((axis_colors[axis], axis_colors[axis]))
            markers.markers.append(axes)

            label = Marker()
            label.header = header
            label.ns = candidate_namespace
            label.id = 2
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = self._point(candidate.translation + np.array([0, 0, 0.025]))
            label.pose.orientation.w = 1.0
            label.scale.z = 0.018
            label.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
            label.text = f"#{index + 1}"
            markers.markers.append(label)
        self._marker_pub.publish(markers)

        best = candidates[0]
        placed = object_points @ best.rotation.T + best.translation
        self._object_pub.publish(point_cloud2.create_cloud_xyz32(header, placed.tolist()))

    @staticmethod
    def _cloud_xyz(message, organized=False):
        points = point_cloud2.read_points_numpy(
            message, field_names=["x", "y", "z"], skip_nans=False,
            reshape_organized_cloud=organized)
        points = np.asarray(points, dtype=np.float32)
        if organized:
            return points.reshape(message.height, message.width, 3)
        return points.reshape(-1, 3)

    def _placement_callback(self, request, response):
        with self._lock:
            clouds = list(self._observation_clouds)
            object_message = self._cached_object_cloud
            completed_object = self._cached_completed_object_points
            bottom_points = self._cached_bottom_points
            collision_envelope = self._cached_collision_envelope_points
        try:
            if object_message is None:
                raise RuntimeError(
                    "NO_CACHED_OBJECT_CLOUD: run [FIND_GRASP] for the complete "
                    "object first")
            request_stamp = self._stamp_seconds(request.observation_header)
            if not clouds:
                raise RuntimeError("NO_ACTIVE_VISION_OBSERVATION_CLOUD")
            cloud = min(
                clouds,
                key=lambda item: abs(
                    self._stamp_seconds(item.header) - request_stamp))
            offset = abs(self._stamp_seconds(cloud.header) - request_stamp)
            if offset > 1.0e-6:
                raise RuntimeError(
                    f"PLACEMENT_OBSERVATION_TIMESTAMP_MISMATCH: {offset:.6f}s")
            if cloud.header.frame_id != "base_link":
                raise RuntimeError(
                    f"PLACEMENT_CLOUD_FRAME_INVALID: {cloud.header.frame_id}")
            if (request.destination_mask.width != cloud.width or
                    request.destination_mask.height != cloud.height):
                raise RuntimeError("PLACEMENT_MASK_CLOUD_DIMENSION_MISMATCH")
            mask = self._bridge.imgmsg_to_cv2(
                request.destination_mask, desired_encoding="mono8") > 0
            organized = self._cloud_xyz(cloud, organized=True)
            valid_destination = mask & np.isfinite(organized).all(axis=2)
            destination = organized[valid_destination]
            if completed_object is not None:
                display_points = completed_object.copy()
            else:
                display_points = self._cloud_xyz(object_message)
                display_points = display_points[
                    np.isfinite(display_points).all(axis=1)]
            backend = str(self.get_parameter("placement_backend").value).lower()
            if backend == "heightmap" and collision_envelope is not None:
                object_points = np.vstack((
                    display_points, collision_envelope)).astype(np.float32)
            else:
                object_points = display_points
            minimum = int(self.get_parameter("minimum_destination_points").value)
            if len(destination) < minimum:
                raise RuntimeError(
                    f"INSUFFICIENT_DESTINATION_POINTS: {len(destination)} < {minimum}")
            if len(object_points) < 80:
                raise RuntimeError(
                    f"INSUFFICIENT_OBJECT_POINTS: {len(object_points)} < 80")

            policy = request.orientation_policy.strip().lower()
            if policy not in ("upright", "free"):
                raise RuntimeError(f"INVALID_ORIENTATION_POLICY: {policy}")
            object_center = object_points.mean(axis=0)
            object_centered = object_points - object_center
            displayed_object_centered = display_points - object_center
            dimensions = np.maximum(
                displayed_object_centered.max(axis=0) -
                displayed_object_centered.min(axis=0),
                0.002)
            packing_diagnostics = None
            if backend == "heightmap":
                candidates, packing_diagnostics = search_heightmap_placements(
                    object_points, organized.reshape(-1, 3), destination,
                    policy,
                    int(self.get_parameter("candidate_count").value),
                    float(self.get_parameter("packing_grid_resolution_m").value),
                    float(self.get_parameter(
                        "packing_upright_yaw_step_deg").value),
                    float(self.get_parameter(
                        "packing_free_yaw_step_deg").value),
                    self.get_parameter("packing_tilt_angles_deg").value,
                    float(self.get_parameter("packing_clearance_m").value),
                    float(self.get_parameter(
                        "packing_contact_tolerance_m").value),
                    float(self.get_parameter(
                        "packing_minimum_support_area_m2").value),
                    float(self.get_parameter(
                        "packing_support_margin_m").value),
                    float(self.get_parameter(
                        "packing_maximum_pile_height_m").value),
                    bottom_points=bottom_points)
                inference_seconds = packing_diagnostics.elapsed_seconds
            elif backend == "anyplace":
                candidates, inference_seconds = infer_placements(
                    self._load_model(), destination, object_centered,
                    int(self.get_parameter("candidate_count").value),
                    int(self.get_parameter("refinement_steps").value),
                    int(self.get_parameter("random_seed").value), "cpu", policy)
            else:
                raise RuntimeError(f"INVALID_PLACEMENT_BACKEND: {backend}")
            if not candidates:
                details = ""
                if packing_diagnostics is not None:
                    details = (
                        f" tested={packing_diagnostics.tested} "
                        f"contained={packing_diagnostics.contained} "
                        f"stable={packing_diagnostics.stable} "
                        f"support_tilt_deg="
                        f"{packing_diagnostics.support_tilt_deg:.1f} "
                        f"support_rejected={{contacts:"
                        f"{packing_diagnostics.support_contacts_rejected},"
                        f"area:{packing_diagnostics.support_area_rejected},"
                        f"com:{packing_diagnostics.support_com_rejected}}}")
                raise RuntimeError(
                    f"PLACEMENT_BACKEND_RETURNED_NO_CANDIDATES:{details}")

            alignment_shifts = []
            if (backend == "anyplace" and policy == "upright" and
                    bottom_points is not None and
                    bool(self.get_parameter(
                        "enable_upright_support_alignment").value)):
                bottom_centered = bottom_points - object_center
                maximum_shift = float(self.get_parameter(
                    "placement_support_maximum_shift_m").value)
                for candidate in candidates:
                    transformed_bottom = (
                        bottom_centered @ candidate.rotation.T +
                        candidate.translation)
                    try:
                        shift, matches = estimate_support_translation(
                            transformed_bottom, destination,
                            float(self.get_parameter(
                                "placement_support_grid_size_m").value),
                            float(self.get_parameter(
                                "placement_support_lookup_radius_m").value),
                            float(self.get_parameter(
                                "placement_support_quantile").value),
                            float(self.get_parameter(
                                "placement_support_clearance_m").value))
                    except ValueError:
                        continue
                    if abs(shift) > maximum_shift:
                        continue
                    candidate.translation = candidate.translation.copy()
                    candidate.translation[2] += shift
                    alignment_shifts.append((shift, matches))

            header = Header()
            header.stamp = cloud.header.stamp
            header.frame_id = "base_link"
            self._destination_pub.publish(
                point_cloud2.create_cloud_xyz32(header, destination.tolist()))
            self._publish_candidates(
                header, candidates, displayed_object_centered, dimensions)
            support_status = (
                f"support_aligned={len(alignment_shifts)}/{len(candidates)} "
                f"support_shift_mm="
                f"{[round(item[0] * 1000, 1) for item in alignment_shifts]}"
                if backend == "anyplace" else
                f"support_aligned={len(candidates)}/{len(candidates)} "
                "support_shift_mm=integrated_heightmap")
            status = (
                f"OK backend={backend} policy={policy} object={request.object_name} "
                f"destination={request.destination_name} "
                f"candidates={len(candidates)} object_points={len(object_points)} "
                f"bottom_completion={'yes' if completed_object is not None else 'no'} "
                f"{support_status} "
                f"destination_points={len(destination)} "
                f"planning={inference_seconds:.2f}s "
                f"best_score={candidates[0].score:.4f}")
            if packing_diagnostics is not None:
                status += (
                    f" packing={{orientations:{packing_diagnostics.orientations},"
                    f"tested:{packing_diagnostics.tested},"
                    f"contained:{packing_diagnostics.contained},"
                    f"stable:{packing_diagnostics.stable},"
                    f"support_tilt_deg:"
                    f"{packing_diagnostics.support_tilt_deg:.1f},"
                    f"support_contacts_rejected:"
                    f"{packing_diagnostics.support_contacts_rejected},"
                    f"support_area_rejected:"
                    f"{packing_diagnostics.support_area_rejected},"
                    f"support_com_rejected:"
                    f"{packing_diagnostics.support_com_rejected}}}")
            response.success = True
            response.status = status
            response.source_object_center = self._point(object_center)
            response.candidates = []
            for index, candidate in enumerate(candidates):
                message = PlacementCandidateMessage()
                message.id = index + 1
                message.object_pose.header = header
                message.object_pose.pose.position = self._point(
                    candidate.translation)
                quaternion = self._quaternion_from_rotation(
                    candidate.rotation)
                message.object_pose.pose.orientation.x = float(quaternion[0])
                message.object_pose.pose.orientation.y = float(quaternion[1])
                message.object_pose.pose.orientation.z = float(quaternion[2])
                message.object_pose.pose.orientation.w = float(quaternion[3])
                message.score = float(candidate.score)
                message.orientation_policy = policy
                response.candidates.append(message)
            self._status_pub.publish(String(data=status))
            self.get_logger().info(f"[ANYPLACE] {status}")
        except Exception as exception:
            response.success = False
            response.status = f"PLACEMENT_FAILED: {exception}"
            response.candidates = []
            self._status_pub.publish(String(data=response.status))
            self.get_logger().error(response.status)
        return response

    def _infer_callback(self, _request, response):
        with self._lock:
            images = list(self._recent_images)
            clouds = list(self._recent_clouds)
            joints = list(self._recent_joints)
        try:
            if not images or not clouds or not joints:
                raise RuntimeError(
                    "NO_SYNCHRONIZED_OBSERVATION: waiting for RGB, registered "
                    "cloud, and joint feedback")
            image_header, image, cloud = min(
                ((image_header, image, cloud) for image_header, image in images
                 for cloud in clouds),
                key=lambda item: abs(
                    self._stamp_seconds(item[0]) - self._stamp_seconds(item[2].header)))
            offset = abs(self._stamp_seconds(image_header) - self._stamp_seconds(cloud.header))
            tolerance = float(self.get_parameter("sync_tolerance_sec").value)
            if offset > tolerance:
                raise RuntimeError(
                    f"RGBD_TIMESTAMP_MISMATCH: offset={offset:.3f}s tolerance={tolerance:.3f}s")
            joint_header, joint_positions = min(
                joints,
                key=lambda item: abs(
                    self._stamp_seconds(item[0]) - self._stamp_seconds(cloud.header)))
            joint_offset = abs(
                self._stamp_seconds(joint_header) - self._stamp_seconds(cloud.header))
            joint_tolerance = float(
                self.get_parameter("joint_sync_tolerance_sec").value)
            if joint_offset > joint_tolerance:
                raise RuntimeError(
                    f"JOINT_TIMESTAMP_MISMATCH: offset={joint_offset:.3f}s "
                    f"tolerance={joint_tolerance:.3f}s")
            base_to_camera = piper_base_to_camera(
                joint_positions, self._hand_eye_matrix)
            target = self.get_parameter("grounding_target").value
            _, bbox = self._call_robobrain(image, target)
            xyz = point_cloud2.read_points_numpy(
                cloud, field_names=["x", "y", "z"], skip_nans=False,
                reshape_organized_cloud=True)
            xyz = np.asarray(xyz, dtype=np.float32).reshape(cloud.height, cloud.width, 3)
            destination, foreground, cloud_box = extract_destination_cloud(
                xyz, image.shape[:2], bbox,
                float(self.get_parameter("surface_neighbor_distance_m").value),
                int(self.get_parameter("minimum_destination_points").value))
            model = self._load_model()
            dimensions = np.asarray(
                self.get_parameter("dummy_object_dimensions_m").value,
                dtype=np.float32)
            object_points = make_dummy_box_cloud(
                dimensions, int(self.get_parameter("dummy_object_point_count").value),
                int(self.get_parameter("random_seed").value))
            candidates, inference_seconds = infer_placements(
                model, destination, object_points,
                int(self.get_parameter("candidate_count").value),
                int(self.get_parameter("refinement_steps").value),
                int(self.get_parameter("random_seed").value), "cpu")
            destination = transform_points(destination, base_to_camera)
            for candidate in candidates:
                candidate.rotation, candidate.translation = transform_pose(
                    candidate.rotation, candidate.translation, base_to_camera)
            header = Header()
            header.stamp = cloud.header.stamp
            header.frame_id = "base_link"
            self._destination_pub.publish(
                point_cloud2.create_cloud_xyz32(header, destination.tolist()))
            self._publish_candidates(header, candidates, object_points, dimensions)
            self._publish_debug(image, bbox, foreground, cloud_box, image_header)
            status = (
                f"OK candidates={len(candidates)} destination_points={len(destination)} "
                f"rgbd_offset={offset:.3f}s joint_offset={joint_offset:.3f}s "
                f"inference={inference_seconds:.2f}s "
                f"frame={header.frame_id} "
                f"best=({candidates[0].translation[0]:.3f},"
                f"{candidates[0].translation[1]:.3f},"
                f"{candidates[0].translation[2]:.3f}) "
                f"surface_score={candidates[0].score:.4f}")
            self._status_pub.publish(String(data=status))
            self.get_logger().info(f"[ANYPLACE] {status}")
            response.success = True
            response.message = status
        except Exception as exception:
            response.success = False
            response.message = f"ANYPLACE_FAILED: {exception}"
            self._status_pub.publish(String(data=response.message))
            self.get_logger().error(response.message)
        return response


def main():
    rclpy.init()
    node = AnyPlaceNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
