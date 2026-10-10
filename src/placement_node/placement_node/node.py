"""ROS 2 placement calculation using height maps or an optional AnyPlace model."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import threading
import time
from collections import deque
from dataclasses import replace
import copy
from urllib import error as url_error
from urllib import request as url_request

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header, String
from std_srvs.srv import Trigger
from my_actions.msg import PlacementCandidate as PlacementCandidateMessage
from my_actions.srv import (
    FindGraspCandidates, FindPlacementCandidates, PinLayoutObservation,
    PlanObjectLayout, PlanPlacement, SearchPlacements, CommitFloorSlots)
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from .grounding import grounding_request
from .inference import infer_placements, make_dummy_box_cloud
from .model import load_anyplace_checkpoint
from task_pipeline.timing import publish_timing
from my_actions.msg import PipelineTiming
from task_pipeline.grasp_input import affordance_from_description
from task_pipeline.registry import endpoint
from task_pipeline.placement import search_placements, validate_placements

from .floor_slots import FloorSlots, layout_key
from .packing import search_heightmap_placements
from .packing_api import make_search_request, decode_search_response, diagnostics_json
from .perception import (
    construct_support_bottom, estimate_support_translation,
    extract_destination_cloud, parse_grounding_box, recover_depth_connected_boundary,
    recover_semantic_source_mask, active_source_support_labels, exposed_stack_support_labels)
from .transforms import (
    load_hand_eye_matrix, piper_base_to_camera, transform_points,
    transform_pose, pose_matrix, attached_object_delta)


def robobrain_chat_endpoint(base_url: str) -> str:
    """Return the chat endpoint for either a host URL or a /v1 base URL."""
    base = str(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def parse_robobrain_content(response: dict) -> str:
    """Extract OpenAI-compatible assistant text with a useful error."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exception:
        raise ValueError("missing choices[0].message.content") from exception
    if not isinstance(content, str):
        raise ValueError("choices[0].message.content is not text")
    return content


class PlacementNode(Node):
    """Calculate placement poses with height-map or optional AnyPlace backends."""

    def __init__(self):
        super().__init__("placement_node")
        default_checkpoint = ""
        parameters = {
            "checkpoint_path": default_checkpoint,
            "rgb_topic": "/camera/color/image_raw",
            "point_cloud_topic": "/camera/depth_registered/points",
            "joint_state_topic": "/feedback/joint_states",
            "joint_sync_tolerance_sec": 0.20,
            "hand_eye_calibration_path": "",
            "api_server_url": "http://10.10.10.2:18091/v1",
            "model_name": "robobrain2.5-4b",
            "grounding_target": "the entire visible plate surface",
            "request_timeout_sec": 120.0,
            "sync_tolerance_sec": 0.20,
            # A short edge-only shortlist can exclude every reachable wrist yaw.
            # Keep the ranked spatial/yaw alternatives for downstream IK batches.
            "candidate_count": 16,
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
            "layout_debug_directory": "",
            "layout_candidate_count": 16,
            "layout_support_mode": "floor_slots",
            "floor_slot_share_unused_space": False,
            "floor_slot_adaptive": True,
            "floor_slot_max_size_m": .125,
            "floor_slot_min_size_m": .06,
            "packing_adjacency_weight": 0.004,
            "packing_grid_resolution_m": 0.005,
            "packing_stack_minimum_coverage": 0.6,
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

        self._pipeline_timing_pub = self.create_publisher(PipelineTiming, '/pipeline/timings', 100)
        self._bridge = CvBridge()
        self._service_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._raw_capture_lock = threading.Lock()
        self._raw_ready = threading.Event()
        self._recent_images = deque(maxlen=20)
        self._recent_clouds = deque(maxlen=5)
        self._recent_joints = deque(maxlen=200)
        self._observation_clouds = deque(maxlen=8)
        self._pending_layout_stamp = None
        self._pinned_layout_cloud = None
        self._cached_object_cloud = None
        self._cached_completed_object_points = None
        self._cached_bottom_points = None
        self._cached_collision_envelope_points = None
        self._completed_object_stamp = None
        self._bottom_attempt_stamp = None
        self._slot_lock = threading.Lock()
        self._slot_layouts = {}
        self._slot_pending = deque(maxlen=32)
        self._model = None
        self._model_path = None
        self._hand_eye_matrix = load_hand_eye_matrix(
            self.get_parameter("hand_eye_calibration_path").value)

        transient = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._destination_pub = self.create_publisher(
            PointCloud2, "/placement/destination_cloud", transient)
        self._object_pub = self.create_publisher(
            PointCloud2, "/placement/best_placed_object", transient)
        self._completed_object_pub = self.create_publisher(
            PointCloud2, "/placement/completed_object_cloud", transient)
        self._bottom_pub = self.create_publisher(
            PointCloud2, "/placement/artificial_bottom_cloud", transient)
        self._slot_pub = self.create_publisher(MarkerArray, '/placement/floor_slots', transient)
        self._marker_pub = self.create_publisher(
            MarkerArray, "/placement/placement_candidates", transient)
        self._debug_pub = self.create_publisher(
            Image, "/placement/grounding_debug", transient)
        self._status_pub = self.create_publisher(
            String, "/placement/status", transient)
        self.create_subscription(
            PointCloud2, "/active_vision/observation_cloud",
            self._observation_cloud_callback, transient)
        self.create_subscription(
            PointCloud2, "/vamp_planner/segmented_target_cloud",
            self._object_cloud_callback, transient)
        self.create_service(
            Trigger, "/placement/infer_placement", self._infer_callback,
            callback_group=self._service_group)
        self.create_service(
            FindPlacementCandidates,
            "/placement/find_placement_candidates",
            self._placement_callback, callback_group=self._service_group)
        self.create_service(
            PlanObjectLayout,
            "/placement/plan_object_layout",
            self._layout_callback, callback_group=self._service_group)
        self.create_service(
            PinLayoutObservation,
            "/placement/pin_layout_observation",
            self._pin_layout_observation_callback,
            callback_group=self._service_group)
        self.create_service(CommitFloorSlots, '/placement/commit_floor_slots',
                            self._commit_floor_slots, callback_group=self._service_group)
        self.create_service(
            SearchPlacements, '/placement/search_placements', self._search_callback,
            callback_group=self._service_group)
        self._commit_slots_client = self.create_client(
            CommitFloorSlots, endpoint('commit_floor_slots'), callback_group=self._service_group)
        self._search_client = self.create_client(
            SearchPlacements, endpoint('search_placements'), callback_group=self._service_group)
        self._grasp_candidates_client = self.create_client(
            FindGraspCandidates, endpoint("calculate_grasps"),
            callback_group=self._service_group)
        self._plan_placement_client = self.create_client(
            PlanPlacement, endpoint("plan_placement"),
            callback_group=self._service_group)
        self.get_logger().info(
            "Placement node ready (backend=%s; manual FK*X output in "
            "base_link; no TF lookup and no motion publishers). "
            "Use /placement/find_placement_candidates or "
            "/placement/plan_object_layout for object placement."
            % self.get_parameter("placement_backend").value)

    def _observation_cloud_callback(self, message):
        if message.height <= 1 or message.header.frame_id != "base_link":
            return
        with self._lock:
            self._observation_clouds.append(message)
            if self._pending_layout_stamp == self._stamp_key(message.header):
                self._pinned_layout_cloud = message
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
            f"[PLACEMENT] Cached segmented object cloud: {message.width} points")
        self._try_complete_cached_object()

    @staticmethod
    def _stamp_key(header):
        return int(header.stamp.sec), int(header.stamp.nanosec)

    def _pin_layout_observation_callback(self, request, response):
        """Reserve exactly one organized cloud while VLM grounding runs."""
        requested_stamp = self._stamp_key(request.observation_header)
        with self._lock:
            self._pending_layout_stamp = requested_stamp
            self._pinned_layout_cloud = None
            for cloud in self._observation_clouds:
                if self._stamp_key(cloud.header) == requested_stamp:
                    self._pinned_layout_cloud = cloud
                    break
            pinned = self._pinned_layout_cloud is not None
        response.success = pinned
        response.status = (
            "LAYOUT_OBSERVATION_PINNED" if pinned else
            "LAYOUT_OBSERVATION_PENDING")
        return response

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
                "[PLACEMENT FOOTPRINT] contour=%d envelope=%d model=%s "
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
                f"[PLACEMENT BOTTOM] completion unavailable: {exception}")

    def _image_callback(self, message):
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exception:
            self.get_logger().error(f"RGB decode failed: {exception}")
            return
        with self._lock:
            self._recent_images.append((message.header, image.copy()))
        self._raw_ready.set()

    def _cloud_callback(self, message):
        if message.height <= 1:
            self.get_logger().warn(
                "Placement requires an organized registered cloud; received height <= 1",
                throttle_duration_sec=5.0)
            return
        with self._lock:
            self._recent_clouds.append(message)
        self._raw_ready.set()

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
        self._raw_ready.set()

    def _capture_legacy_inputs(self):
        """Subscribe only while the legacy raw-input trigger needs a snapshot."""
        if not self._raw_capture_lock.acquire(blocking=False):
            raise RuntimeError('LEGACY_CAPTURE_BUSY')
        subscriptions = []
        try:
            with self._lock:
                self._recent_images.clear()
                self._recent_clouds.clear()
                self._recent_joints.clear()
            self._raw_ready.clear()
            for kind, parameter, callback, qos in (
                    (Image, 'rgb_topic', self._image_callback, qos_profile_sensor_data),
                    (PointCloud2, 'point_cloud_topic', self._cloud_callback, qos_profile_sensor_data),
                    (JointState, 'joint_state_topic', self._joint_callback, 50)):
                subscriptions.append(self.create_subscription(
                    kind, self.get_parameter(parameter).value, callback, qos,
                    callback_group=self._service_group))
            deadline = time.monotonic() + min(
                5.0, float(self.get_parameter('request_timeout_sec').value))
            while rclpy.ok() and time.monotonic() < deadline:
                with self._lock:
                    snapshot = (list(self._recent_images), list(self._recent_clouds),
                                list(self._recent_joints))
                if all(snapshot):
                    images, clouds, joints = snapshot
                    pair = min(((image_header, cloud.header) for image_header, _ in images
                                for cloud in clouds), key=lambda item: abs(
                                    self._stamp_seconds(item[0]) - self._stamp_seconds(item[1])))
                    image_time, cloud_time = map(self._stamp_seconds, pair)
                    joint_offset = min(abs(self._stamp_seconds(header) - cloud_time)
                                       for header, _ in joints)
                    if (abs(image_time-cloud_time) <= self.get_parameter('sync_tolerance_sec').value
                            and joint_offset <= self.get_parameter('joint_sync_tolerance_sec').value):
                        return snapshot
                self._raw_ready.wait(0.02)
                self._raw_ready.clear()
            raise RuntimeError('NO_SYNCHRONIZED_OBSERVATION: raw input capture timed out')
        finally:
            for subscription in subscriptions:
                self.destroy_subscription(subscription)
            self._raw_capture_lock.release()

    @staticmethod
    def _stamp_seconds(header):
        return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9

    def _load_model(self):
        path = os.path.expanduser(self.get_parameter("checkpoint_path").value)
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"checkpoint not found: {path}; run ros2 run placement_node "
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
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("RoboBrain is busy; stale inference is not queued")
        try:
            return self._request_robobrain(image.copy(), target)
        finally:
            self._inference_lock.release()

    def _request_robobrain(self, image: np.ndarray, target: str):
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise RuntimeError("failed to JPEG-encode RGB frame")
        payload = grounding_request(
            self.get_parameter("model_name").value,
            "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
            target)
        request = url_request.Request(
            robobrain_chat_endpoint(
                self.get_parameter("api_server_url").value),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with url_request.urlopen(
                    request, timeout=float(self.get_parameter("request_timeout_sec").value)) as result:
                response = json.loads(result.read().decode("utf-8"))
        except (url_error.URLError, TimeoutError, json.JSONDecodeError) as exception:
            raise RuntimeError(f"RoboBrain request failed: {exception}") from exception
        try:
            reply = parse_robobrain_content(response)
        except ValueError as exception:
            raise RuntimeError(
                f"unexpected RoboBrain response: {exception}") from exception
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

    def _publish_layout_markers(self, header, placed_objects):
        """Keep all accepted objects and their sequence labels visible in RViz."""
        markers = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)
        palette = [(0.2, 1.0, 0.35), (0.15, 0.85, 1.0), (1.0, 0.6, 0.1)]
        for index, placed in enumerate(placed_objects):
            name, points, candidate = placed[:3]
            plan = placed[3] if len(placed) > 3 else None
            color = palette[index % len(palette)]
            marker = Marker()
            marker.header = header
            marker.ns = "accepted_layout"
            marker.id = index * 2
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.004
            marker.scale.y = 0.004
            marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
            displayed = points[np.linspace(
                0, len(points) - 1, min(3000, len(points)), dtype=np.int32)]
            marker.points = [self._point(point) for point in displayed]
            markers.markers.append(marker)
            label = Marker()
            label.header = header
            label.ns = marker.ns
            label.id = index * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.orientation.w = 1.0
            label.pose.position = self._point([
                candidate.translation[0], candidate.translation[1],
                float(points[:, 2].max()) + 0.03])
            label.scale.z = 0.025
            label.color = marker.color
            label.text = f"#{index + 1} {name}"
            markers.markers.append(label)
            if plan is not None:
                for finger_index, pose in enumerate(plan.placement_finger_poses):
                    finger = Marker()
                    finger.header = header
                    finger.ns = "layout_placement_grippers"
                    finger.id = index * 10 + finger_index
                    finger.type = Marker.CUBE
                    finger.action = Marker.ADD
                    finger.pose = pose
                    finger.scale = plan.placement_finger_dimensions
                    finger.color = copy.deepcopy(marker.color)
                    finger.color.a = 0.75
                    markers.markers.append(finger)
                approach = Marker()
                approach.header = header
                approach.ns = "layout_placement_grippers"
                approach.id = index * 10 + 2
                approach.type = Marker.ARROW
                approach.action = Marker.ADD
                approach.pose.orientation.w = 1.0
                approach.points = [plan.preplace_ee_pose.pose.position,
                                   plan.placement_ee_pose.pose.position]
                approach.scale.x, approach.scale.y, approach.scale.z = 0.005, 0.010, 0.015
                approach.color = marker.color
                markers.markers.append(approach)
        self._marker_pub.publish(markers)

    @staticmethod
    def _pose_matrix(pose):
        return pose_matrix(
            [pose.position.x, pose.position.y, pose.position.z],
            [pose.orientation.x, pose.orientation.y, pose.orientation.z,
             pose.orientation.w])

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

    @staticmethod
    def _wait_for_future(future, timeout_sec):
        completed = threading.Event()
        future.add_done_callback(lambda _: completed.set())
        return completed.wait(timeout_sec)

    def _layout_callback(self, request, response):
        """Plan an ordered, calculation-only layout with one active cloud."""
        layout_started = time.perf_counter()
        try:
            if not request.object_names:
                raise RuntimeError("LAYOUT_HAS_NO_OBJECTS")
            if len(request.object_names) != len(request.object_masks):
                raise RuntimeError("LAYOUT_OBJECT_MASK_COUNT_MISMATCH")
            policy = request.orientation_policy.strip().lower()
            if policy not in ("upright", "free"):
                raise RuntimeError(f"INVALID_ORIENTATION_POLICY: {policy}")
            layout_support_mode = str(self.get_parameter('layout_support_mode').value)
            if policy == 'free' and layout_support_mode == 'floor_slots':
                layout_support_mode = 'auto'
            if layout_support_mode not in ('auto', 'edge', 'stack', 'floor_slots'):
                raise RuntimeError('INVALID_LAYOUT_SUPPORT_MODE')

            with self._lock:
                clouds = list(self._observation_clouds)
                pinned_cloud = self._pinned_layout_cloud
            if (pinned_cloud is not None and
                    self._stamp_key(pinned_cloud.header) ==
                    self._stamp_key(request.observation_header)):
                cloud = pinned_cloud
            else:
                if not clouds:
                    raise RuntimeError("NO_ACTIVE_VISION_OBSERVATION_CLOUD")
                request_stamp = self._stamp_seconds(request.observation_header)
                cloud = min(
                    clouds,
                    key=lambda item: abs(
                        self._stamp_seconds(item.header) - request_stamp))
            request_stamp = self._stamp_seconds(request.observation_header)
            offset = abs(self._stamp_seconds(cloud.header) - request_stamp)
            if offset > 1.0e-6:
                raise RuntimeError(
                    f"PLACEMENT_OBSERVATION_TIMESTAMP_MISMATCH: {offset:.6f}s")
            if cloud.header.frame_id != "base_link" or cloud.height <= 1:
                raise RuntimeError("LAYOUT_REQUIRES_ORGANIZED_BASE_LINK_CLOUD")

            debug_directory = self.get_parameter("layout_debug_directory").value
            if debug_directory:
                from rclpy.serialization import serialize_message
                try:
                    directory = Path(debug_directory).expanduser()
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "layout_request.cdr").write_bytes(serialize_message(request))
                    (directory / "layout_observation.cdr").write_bytes(serialize_message(cloud))
                except OSError as exception:
                    self.get_logger().warn(f"[LAYOUT] request debug save failed: {exception}")
            organized = self._cloud_xyz(cloud, organized=True)
            destination_mask = self._bridge.imgmsg_to_cv2(
                request.destination_mask, desired_encoding="mono8") > 0
            if destination_mask.shape != organized.shape[:2]:
                raise RuntimeError("LAYOUT_DESTINATION_MASK_DIMENSION_MISMATCH")
            destination_valid = (
                destination_mask & np.isfinite(organized).all(axis=2))
            destination = organized[destination_valid]
            minimum = int(self.get_parameter(
                "minimum_destination_points").value)
            if len(destination) < minimum:
                raise RuntimeError(
                    f"INSUFFICIENT_DESTINATION_POINTS: {len(destination)} < {minimum}")

            # Publish the segmented destination before candidate generation so
            # RViz still shows the planner input when packing or validation
            # rejects every candidate.
            header = Header()
            header.stamp = cloud.header.stamp
            header.frame_id = "base_link"
            self._publish_layout_markers(header, [])
            self._destination_pub.publish(
                point_cloud2.create_cloud_xyz32(
                    header, destination.tolist()))

            # This scene is private to the request. Every scheduled source is
            # assumed to be picked during execution, so remove all of their
            # measured source clouds before packing. Their accepted poses are
            # then inserted sequentially; unrelated obstacles remain intact.
            object_masks = []
            scheduled_sources = np.zeros(organized.shape[:2], dtype=bool)
            for name, mask_message in zip(
                    request.object_names, request.object_masks):
                object_mask = self._bridge.imgmsg_to_cv2(
                    mask_message, desired_encoding="mono8") > 0
                if object_mask.shape != organized.shape[:2]:
                    raise RuntimeError(
                        f"LAYOUT_OBJECT_MASK_DIMENSION_MISMATCH: {name}")
                object_masks.append(object_mask)
                scheduled_sources |= object_mask
            # Recover bounded, measured SAM seams before both grasping and
            # placement. Future object masks remain protected collision geometry.
            for object_index, object_mask in enumerate(object_masks):
                object_masks[object_index] = recover_depth_connected_boundary(
                    organized, object_mask, scheduled_sources & ~object_mask)
                scheduled_sources |= object_masks[object_index]
            scheduled_sources = np.logical_or.reduce(object_masks)
            # A 2.5-D destination map is image-visible geometry inside the
            # grounded container silhouette. Selecting the full camera cloud
            # merely by base-frame XY can project unrelated foreground or
            # background geometry into the tray. Fill the destination mask's
            # image-space hull so occluding objects already in the tray remain
            # represented while surroundings are excluded.
            destination_rows, destination_columns = np.nonzero(destination_mask)
            destination_region = np.zeros(destination_mask.shape, np.uint8)
            if len(destination_columns) >= 3:
                destination_hull = cv2.convexHull(np.column_stack((
                    destination_columns, destination_rows)).astype(np.int32))
                cv2.fillConvexPoly(destination_region, destination_hull, 1)
            else:
                destination_region[destination_mask] = 1
            virtual_scene = organized.reshape(-1, 3).copy()
            virtual_scene[np.flatnonzero(
                destination_region.reshape(-1) == 0)] = np.nan
            virtual_scene[np.flatnonzero(scheduled_sources.reshape(-1))] = np.nan
            # Collision checks retain all surrounding geometry and future
            # source objects; remove only the object currently being picked.
            collision_scene = organized.reshape(-1, 3).copy()
            support_plane_z = float(np.percentile(destination[:, 2], 20))
            support_valid = destination_valid & (
                organized[:, :, 2] <= support_plane_z + 0.003)
            collision_support = support_valid.reshape(-1).astype(np.uint8)
            # Label known source objects independently of support height, so
            # departure noise tolerance cannot mistake another object's base
            # for the table. Successfully removed samples become NaN below.
            collision_support[scheduled_sources.reshape(-1)] = 3
            destination_support = organized[support_valid]
            measured_scene = organized.reshape(-1, 3)
            placed_objects = []
            placement_messages = []
            diagnostics = []
            if (not self._grasp_candidates_client.wait_for_service(
                    timeout_sec=5.0) or
                    not self._plan_placement_client.wait_for_service(
                    timeout_sec=5.0)):
                raise RuntimeError("LAYOUT_GRASP_OR_PLACEMENT_SERVICE_UNAVAILABLE")
            for index, (name, mask_message, object_mask) in enumerate(zip(
                    request.object_names, request.object_masks, object_masks)):
                source_indices = np.flatnonzero(object_mask.reshape(-1))
                saved_source = collision_scene[source_indices].copy()
                committed_count = len(placed_objects)
                try:
                    object_valid = object_mask & np.isfinite(organized).all(axis=2)
                    source_points = organized[object_valid]
                    if len(source_points) < 80:
                        raise RuntimeError(
                            f"INSUFFICIENT_OBJECT_POINTS: {name}={len(source_points)}")

                    # The median of a segmentation mask is not a reliable grasp
                    # seed for hollow/concave objects: it can lie in a bowl's
                    # empty interior and then snap to a small depth fragment. Pick
                    # the deepest interior pixel of the largest valid-depth image
                    # component instead.  The complete EdgeSAM mask is still sent
                    # to VAMP; this only gives its shared object filter a stable
                    # anchor on measured object material.
                    valid_u8 = object_valid.astype(np.uint8)
                    component_count, component_labels, component_stats, _ = (
                        cv2.connectedComponentsWithStats(valid_u8, connectivity=8))
                    if component_count <= 1:
                        raise RuntimeError(f"OBJECT_HAS_NO_VALID_DEPTH_COMPONENT: {name}")
                    largest_label = 1 + int(np.argmax(
                        component_stats[1:, cv2.CC_STAT_AREA]))
                    largest_component = (
                        component_labels == largest_label).astype(np.uint8)
                    interior_distance = cv2.distanceTransform(
                        largest_component, cv2.DIST_L2, 5)
                    seed_v, seed_u = np.unravel_index(
                        int(np.argmax(interior_distance)),
                        interior_distance.shape)
                    grasp_request = FindGraspCandidates.Request()
                    grasp_request.observation_header = request.observation_header
                    grasp_request.seed_u = int(seed_u)
                    grasp_request.seed_v = int(seed_v)
                    grasp_request.image_width = int(object_mask.shape[1])
                    grasp_request.image_height = int(object_mask.shape[0])
                    grasp_request.target_mask = self._bridge.cv2_to_imgmsg(
                        object_mask.astype(np.uint8) * 255, encoding="mono8")
                    grasp_request.target_mask.header = mask_message.header
                    grasp_request.target_description = "layout: " + name
                    grasp_request.affordance_type = affordance_from_description(name)
                    grasp_request.collision_scene = point_cloud2.create_cloud_xyz32(
                        header, collision_scene.tolist())
                    grasp_started = time.perf_counter()
                    grasp_future = self._grasp_candidates_client.call_async(
                        grasp_request)
                    if not self._wait_for_future(
                            grasp_future,
                            float(self.get_parameter("request_timeout_sec").value)):
                        raise RuntimeError(f"LAYOUT_GRASP_TIMEOUT: {name}")
                    grasp_result = grasp_future.result()
                    if debug_directory:
                        try:
                            (directory / f"layout_object_{index + 1}_grasp_request.cdr").write_bytes(
                                serialize_message(grasp_request))
                            if grasp_result is not None:
                                (directory / f"layout_object_{index + 1}_grasp_response.cdr").write_bytes(
                                    serialize_message(grasp_result))
                        except OSError as exception:
                            self.get_logger().warn(f"[LAYOUT] grasp debug save failed: {exception}")
                    if grasp_result is None or not grasp_result.success or not grasp_result.candidates:
                        status = (
                            grasp_result.status if grasp_result is not None else
                            "no response")
                        raise RuntimeError(
                            f"LAYOUT_GRASP_FAILED: {name}: {status}")

                    # Use the same depth-connected geometry that the grasp engine
                    # validated, carried in the response for this exact observation.
                    # A raw SAM silhouette can include disconnected floor returns.
                    validated_mask = grasp_result.validated_target_mask
                    if (self._stamp_key(validated_mask.header) !=
                            self._stamp_key(request.observation_header)):
                        raise RuntimeError(f"LAYOUT_GRASP_MASK_TIMESTAMP_MISMATCH: {name}")
                    connected_mask = self._bridge.imgmsg_to_cv2(
                        validated_mask, desired_encoding="mono8") > 0
                    if (connected_mask.shape != object_mask.shape or
                            np.any(connected_mask & ~object_mask)):
                        raise RuntimeError(f"LAYOUT_GRASP_MASK_INVALID: {name}")
                    source_points = organized[
                        connected_mask & np.isfinite(organized).all(axis=2)]
                    if len(source_points) < 80:
                        raise RuntimeError(f"LAYOUT_GRASP_MASK_EMPTY: {name}")

                    grasp_seconds = time.perf_counter() - grasp_started
                    bottom_points = None
                    display_points = source_points
                    packing_points = source_points
                    finite_scene = measured_scene[
                        np.isfinite(measured_scene).all(axis=1)]
                    try:
                        completed, bottom_points, envelope, _ = (
                            construct_support_bottom(source_points, finite_scene,
                                float(self.get_parameter(
                                    "bottom_support_context_margin_m").value),
                                float(self.get_parameter(
                                    "bottom_support_plane_distance_m").value),
                                float(self.get_parameter(
                                    "bottom_support_maximum_tilt_deg").value),
                                float(self.get_parameter(
                                    "bottom_footprint_band_m").value),
                                float(self.get_parameter(
                                    "bottom_grid_spacing_m").value),
                                int(self.get_parameter(
                                    "bottom_maximum_points").value)))
                        display_points = completed
                        packing_points = np.vstack((completed, envelope))
                    except ValueError as exception:
                        self.get_logger().warn(
                            f"[LAYOUT] {name}: support contour unavailable: {exception}")

                    # Associate the complete rigid source after completion.
                    # Otherwise a discarded image component on the same cup
                    # rim can collide with its own completed body envelope.
                    # Semantic ownership, measured XYZ proximity and the bounded
                    # seam protect other objects and already-placed geometry.
                    associated_source = recover_semantic_source_mask(
                        organized, connected_mask, object_mask,
                        surface_points=packing_points)
                    # Do not assign a lower stand or table to an elevated body.
                    lower_geometry = (organized[..., 2] <
                                      float(np.percentile(source_points[:, 2], 3)) - 0.003)
                    source_region = recover_depth_connected_boundary(
                        organized, associated_source,
                        (scheduled_sources & ~object_mask) | lower_geometry,
                        distance_m=0.012, pixel_radius=10)
                    source_indices = np.flatnonzero(source_region.reshape(-1))
                    saved_source = collision_scene[source_indices].copy()

                    # VAMP's placed-support visualization consumes a contour in
                    # the SOURCE pose. Publish this object's synchronized contour
                    # before validation rather than leaving the previous object's
                    # already-placed contour on the shared visualization topic.
                    self._bottom_pub.publish(point_cloud2.create_cloud_xyz32(
                        header, bottom_points.tolist() if bottom_points is not None else []))

                    # Plan the pose after the active object has been picked. Its
                    # measured source cloud must therefore not remain as an
                    # obstacle in the placement heightmap. Keep all other measured
                    # geometry, including previously placed virtual objects.
                    planning_scene = virtual_scene
                    object_center = packing_points.mean(axis=0)
                    candidates, packing_diagnostics = self._search_placements_api(header, name,
                        packing_points, planning_scene, destination, policy,
                        int(self.get_parameter("layout_candidate_count").value),
                        float(self.get_parameter(
                            "packing_grid_resolution_m").value),
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
                        bottom_points=bottom_points,
                        support_mode=layout_support_mode,
                        floor_slot_adaptive=bool(self.get_parameter('floor_slot_adaptive').value),
                        floor_slot_max_size_m=float(self.get_parameter('floor_slot_max_size_m').value),
                        floor_slot_min_size_m=float(self.get_parameter('floor_slot_min_size_m').value),

                        floor_slot_share_unused_space=bool(self.get_parameter(
                            "floor_slot_share_unused_space").value),
                        stack_minimum_coverage=float(self.get_parameter(
                            "packing_stack_minimum_coverage").value),
                        adjacency_weight=float(self.get_parameter(
                            "packing_adjacency_weight").value),
                        candidate_yaw_variants=True,
                        grasp_reference_positions=[[
                            grasp.grasp_pose.pose.position.x,
                            grasp.grasp_pose.pose.position.y,
                            grasp.grasp_pose.pose.position.z]
                            for grasp in grasp_result.candidates],
                        reserve_near_space=bool(index == 0 and len(object_masks) > 1 and
                            np.prod(np.ptp(packing_points[:, :2], axis=0)) >=
                            0.2 * np.prod(np.ptp(destination[:, :2], axis=0))),
                        timing_sink=lambda seconds, success, status: publish_timing(
                            getattr(self, '_pipeline_timing_pub', None), 'search_placements',
                            seconds, header, name, success, status))
                    if not candidates:
                        raise RuntimeError(
                            f"LAYOUT_NO_PLACEMENT: {name} tested="
                            f"{packing_diagnostics.tested} contained="
                            f"{packing_diagnostics.contained} stable="
                            f"{packing_diagnostics.stable} stack_maximum_coverage="
                            f"{packing_diagnostics.stack_maximum_coverage:.3f} support_tilt_deg="
                            f"{packing_diagnostics.support_tilt_deg:.1f} "
                            f"support_rejected={{contacts:"
                            f"{packing_diagnostics.support_contacts_rejected},"
                            f"area:{packing_diagnostics.support_area_rejected},"
                            f"com:{packing_diagnostics.support_com_rejected}}}")
                    candidate_messages = []
                    for candidate_index, candidate in enumerate(candidates):
                        message = PlacementCandidateMessage()
                        message.id = candidate_index + 1
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
                        candidate_messages.append(message)

                    if debug_directory:
                        try:
                            np.savez_compressed(
                                directory / f"layout_object_{index + 1}_input.npz",
                                name=name, destination=destination, scene=planning_scene,
                                object_points=packing_points, display_points=display_points,
                                bottom_points=(bottom_points if bottom_points is not None
                                               else np.empty((0, 3), np.float32)),
                                rotations=np.asarray([item.rotation for item in candidates]),
                                translations=np.asarray([item.translation for item in candidates]),
                                scores=np.asarray([item.score for item in candidates]),
                                packing_seconds=packing_diagnostics.elapsed_seconds,
                                grasp_seconds=grasp_seconds)
                        except (OSError, ValueError) as exception:
                            self.get_logger().warn(f"[LAYOUT] input debug save failed: {exception}")

                    # Reuse the normal placement debug topic while VAMP validates
                    # the end-effector pose derived from the accepted HGGD grasp.
                    self._publish_candidates(
                        header, candidates, packing_points - object_center,
                        np.ptp(display_points, axis=0))
                    plan_request = PlanPlacement.Request()
                    plan_request.grasp_candidate = grasp_result.candidates[0]
                    plan_request.source_object_center = self._point(object_center)
                    plan_request.candidates = candidate_messages
                    plan_request.object_name = name
                    plan_request.destination_name = request.destination_name
                    plan_request.orientation_policy = policy
                    collision_scene[source_indices] = np.nan
                    plan_request.use_layout_scene = True
                    plan_request.support_mode = ('edge' if layout_support_mode == 'floor_slots'
                                                 else layout_support_mode)
                    plan_request.source_object_cloud = point_cloud2.create_cloud_xyz32(
                        header, packing_points.tolist())
                    if bottom_points is not None and len(bottom_points) >= 3:
                        bottom_center = bottom_points.mean(axis=0)
                        _, _, axes = np.linalg.svd(bottom_points - bottom_center,
                                                   full_matrices=False)
                        bottom_normal = axes[-1]
                        bottom_flags = (np.abs((packing_points - bottom_center) @
                                               bottom_normal) <= 0.003).astype(np.uint8)
                        # Value 2 identifies the exact inferred contour/cap;
                        # noisy measured samples may be bottom contacts (1),
                        # but must not bias its source support-plane fit.
                        bottom_flags[len(source_points):] *= 2
                        bottom_fields = [
                            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                            PointField(name='bottom', offset=12, datatype=PointField.UINT8, count=1)]
                        plan_request.source_object_cloud = point_cloud2.create_cloud(
                            header, bottom_fields,
                            [(*point, int(flag)) for point, flag in zip(
                                packing_points, bottom_flags)])
                    finite_collision = np.isfinite(collision_scene).all(axis=1)
                    # Disconnected floor returns rejected from this object's
                    # semantic mask are still environment, but are not OTHER
                    # objects. Preserve protection for every overlapping mask
                    # and for appended, already-placed geometry.
                    active_support = active_source_support_labels(
                        collision_support, object_masks, index)
                    active_support = exposed_stack_support_labels(
                        collision_scene, active_support, active_support == 2,
                        resolution=float(self.get_parameter(
                            'packing_grid_resolution_m').value))
                    scene_fields = [
                        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                        PointField(name='support', offset=12, datatype=PointField.UINT8, count=1)]
                    collision_rows = [(*point, int(support)) for point, support in zip(
                        collision_scene[finite_collision], active_support[finite_collision])]
                    plan_request.placement_scene = point_cloud2.create_cloud(
                        header, scene_fields, collision_rows)
                    plan_request.destination_support_cloud = point_cloud2.create_cloud_xyz32(
                        header, destination_support.tolist())
                    if debug_directory:
                        try:
                            (directory / f"layout_object_{index + 1}_plan_request.cdr").write_bytes(
                                serialize_message(plan_request))
                        except OSError as exception:
                            self.get_logger().warn(f"[LAYOUT] plan debug save failed: {exception}")
                    validation_started = time.perf_counter()
                    plan_result = None
                    plan_request.grasp_candidate = grasp_result.candidates[0]
                    plan_request.alternative_grasps = grasp_result.candidates[1:]
                    plan_request.validation_batch_size = 8
                    plan_request.retry_motion_failures = True
                    plan_request.plan_pick_place_motion = True

                    def call_placement_backend(query):
                        future = self._plan_placement_client.call_async(query)
                        if not self._wait_for_future(future, float(
                                self.get_parameter("request_timeout_sec").value)):
                            raise RuntimeError(f"LAYOUT_PLACEMENT_TIMEOUT_STATE_UNKNOWN: {name}")
                        return future.result()

                    plan_result = call_placement_backend(plan_request)
                    if plan_result.success:
                        plan_request.grasp_candidate = plan_result.selected_grasp_candidate
                    if debug_directory:
                        try:
                            (directory / f"layout_object_{index + 1}_plan_request.cdr").write_bytes(
                                serialize_message(plan_request))
                        except OSError as exception:
                            self.get_logger().warn(f"[LAYOUT] final request debug save failed: {exception}")
                    if plan_result is None or not plan_result.success:
                        status = (
                            plan_result.status if plan_result is not None else
                            "no response")
                        raise RuntimeError(
                            f"LAYOUT_PLACEMENT_IK_FAILED: {name}: {status}")
                    if debug_directory:
                        try:
                            (directory / f"layout_object_{index + 1}_plan_response.cdr").write_bytes(
                                serialize_message(plan_result))
                        except OSError as exception:
                            self.get_logger().warn(f"[LAYOUT] response debug save failed: {exception}")
                    selected_id = int(plan_result.selected_candidate.id)
                    if selected_id < 1 or selected_id > len(candidates):
                        raise RuntimeError(
                            f"LAYOUT_SELECTED_CANDIDATE_INVALID: {name} #{selected_id}")
                    validation_seconds = time.perf_counter() - validation_started
                    accepted = candidates[selected_id - 1]
                    delta = attached_object_delta(
                        self._pose_matrix(plan_request.grasp_candidate.grasp_pose.pose),
                        self._pose_matrix(plan_result.placement_ee_pose.pose))
                    # Commit the actual accepted FK pose, not the ideal packing
                    # pose. Later objects and RViz must agree with the held cloud.
                    accepted = replace(accepted, rotation=delta[:3, :3],
                        translation=delta[:3, :3] @ object_center + delta[:3, 3])
                    debug_directory = self.get_parameter("layout_debug_directory").value
                    if debug_directory:
                        try:
                            directory = Path(debug_directory).expanduser()
                            directory.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(
                                directory / f"layout_object_{index + 1}.npz",
                                name=name, destination=destination,
                                scene=planning_scene, object_points=packing_points,
                                display_points=display_points,
                                bottom_points=(bottom_points if bottom_points is not None
                                               else np.empty((0, 3), np.float32)),
                                rotations=np.asarray([item.rotation for item in candidates]),
                                translations=np.asarray([item.translation for item in candidates]),
                                scores=np.asarray([item.score for item in candidates]),
                                selected_id=selected_id,
                                packing_seconds=packing_diagnostics.elapsed_seconds,
                                grasp_seconds=grasp_seconds,
                                validation_seconds=validation_seconds)
                        except (OSError, ValueError) as exception:
                            self.get_logger().warn(f"[LAYOUT] debug save failed: {exception}")
                    centered_display = display_points - object_center
                    placed_display = (
                        centered_display @ accepted.rotation.T +
                        accepted.translation)
                    centered_packing = packing_points - object_center
                    placed_packing = (
                        centered_packing @ accepted.rotation.T +
                        accepted.translation)

                    if layout_support_mode == 'floor_slots':
                        query = CommitFloorSlots.Request(
                            selected_pose=plan_result.selected_candidate.object_pose)
                        if not self._commit_slots_client.wait_for_service(timeout_sec=3.0):
                            raise RuntimeError('FLOOR_SLOT_COMMIT_API_UNAVAILABLE')
                        committed = self._commit_slots_client.call_async(query)
                        if not self._wait_for_future(committed, 10.0):
                            raise RuntimeError('FLOOR_SLOT_COMMIT_TIMEOUT_STATE_UNKNOWN')
                        if not committed.result().success:
                            raise RuntimeError(committed.result().status)
                    # Commit the accepted virtual pose. The next object sees the
                    # destination plus this placed object, not its old source pose.
                    virtual_scene = np.vstack((planning_scene, placed_packing))
                    collision_scene = np.vstack((collision_scene, placed_packing))
                    # Only exposed placed tops permit final bottom contact.
                    # Sides and bottoms remain blocking and are never tray floor.
                    placed_support = exposed_stack_support_labels(
                        placed_packing, np.full(len(placed_packing), 3, dtype=np.uint8),
                        np.ones(len(placed_packing), dtype=bool),
                        resolution=float(self.get_parameter(
                            'packing_grid_resolution_m').value))
                    collision_support = np.concatenate((collision_support, placed_support))
                    if bottom_points is not None:
                        active_footprint = (
                            (bottom_points - object_center) @ accepted.rotation.T +
                            accepted.translation)
                    else:
                        low = np.percentile(placed_display[:, 2], 10.0)
                        active_footprint = placed_display[
                            placed_display[:, 2] <= low]
                    committed_message = copy.deepcopy(plan_result.selected_candidate)
                    committed_message.object_pose.pose.position = self._point(accepted.translation)
                    resolved_quaternion = self._quaternion_from_rotation(accepted.rotation)
                    orientation = committed_message.object_pose.pose.orientation
                    orientation.x, orientation.y, orientation.z, orientation.w = (
                        float(value) for value in resolved_quaternion)
                    placement_messages.append(committed_message)
                    placed_objects.append((name, placed_display, accepted, plan_result))
                    diagnostics.append(
                        f"#{index + 1}:{name}[tested={packing_diagnostics.tested},"
                        f"stable={packing_diagnostics.stable},"
                        f"grasp={plan_request.grasp_candidate.id},"
                        "grasp_backend=hggd,"
                        f"placement={selected_id},"
                        f"grasp_seconds={grasp_seconds:.3f},"
                        f"packing_seconds={packing_diagnostics.elapsed_seconds:.3f},"
                        f"validation_seconds={validation_seconds:.3f}]")

                    # Reuse the existing single-object visualization topics. The
                    # destination accumulates committed objects, while object_world,
                    # grasp markers, placed object and support show this iteration.
                    accumulated = [destination]
                    accumulated.extend(item[1] for item in placed_objects)
                    self._destination_pub.publish(
                        point_cloud2.create_cloud_xyz32(
                            header, np.vstack(accumulated).tolist()))
                    self._object_pub.publish(
                        point_cloud2.create_cloud_xyz32(
                            header, placed_display.tolist()))
                    self._bottom_pub.publish(
                        point_cloud2.create_cloud_xyz32(
                            header, active_footprint.tolist()))
                    self._publish_layout_markers(header, placed_objects)

                except Exception as exception:
                    if len(placed_objects) != committed_count:
                        raise
                    # A failed object was never picked. Restore it for collision
                    # checks and retain its visible destination-region geometry.
                    collision_scene[source_indices] = saved_source
                    failed_region = object_mask & (destination_region > 0)
                    failed_points = organized[failed_region & np.isfinite(organized).all(axis=2)]
                    if len(failed_points):
                        virtual_scene = np.vstack((virtual_scene, failed_points))
                    diagnostics.append(f'#{index + 1}:{name}[FAILED:{exception}]')
                    self.get_logger().warn(f'[LAYOUT SKIPPED] {name}: {exception}')

            response.success = len(placed_objects) == len(request.object_names)
            response.object_names = [item[0] for item in placed_objects]
            response.placements = placement_messages
            response.status = (
                ("LAYOUT_PLAN_READY" if response.success else "LAYOUT_PARTIAL") +
                " calculation_only=true destination="
                f"{request.destination_name} objects={len(placed_objects)} "
                f"failed={len(request.object_names) - len(placed_objects)} "
                f"calculation_seconds={time.perf_counter() - layout_started:.3f} " +
                " ".join(diagnostics))
            self._status_pub.publish(String(data=response.status))
            self.get_logger().info(f"[LAYOUT] {response.status}")
        except Exception as exception:
            response.success = False
            response.status = f"LAYOUT_FAILED: {exception}"
            self._status_pub.publish(String(data=response.status))
            self.get_logger().error(response.status)
        return response

    def _search_placements_api(self, header, name, *args, **kwargs):
        request = make_search_request(header, name, *args, **kwargs)
        if not self._search_client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('SEARCH_API_UNAVAILABLE')
        future = self._search_client.call_async(request)
        if not self._wait_for_future(future, self.get_parameter('request_timeout_sec').value):
            raise RuntimeError('SEARCH_API_TIMEOUT_STATE_UNKNOWN')
        return decode_search_response(future.result())

    def _publish_floor_slots(self, header, layout, blocked=(), selected=(), z=0.0, occupied_xy=()):
        """Display reusable cells and only the committed selected group."""
        grid = FloorSlots.from_dict(layout)
        blocked, selected = set(map(tuple, blocked)), set(map(tuple, selected))
        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for index, cell in enumerate(map(tuple, grid.cells)):
            marker = Marker(header=header, ns='floor_slots', id=index,
                            type=Marker.LINE_STRIP, action=Marker.ADD)
            marker.pose.orientation.w = 1.0
            marker.scale.x = .002
            marker.color = (ColorRGBA(r=.2, g=1.0, b=.3, a=1.0) if cell in selected else
                            ColorRGBA(r=1.0, g=.4, b=.1, a=1.0) if cell in blocked else
                            ColorRGBA(r=.15, g=.7, b=1.0, a=.9))
            vertices = grid.vertices(cell)
            marker.points = [self._point([xy[0], xy[1], z+.003])
                             for xy in np.vstack((vertices, vertices[0]))]
            markers.markers.append(marker)
            label = Marker(header=header, ns='floor_slot_ids', id=index,
                           type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            label.pose.orientation.w = 1.0
            xy = vertices.mean(axis=0)
            label.pose.position = self._point([xy[0], xy[1], z+.008])
            label.scale.z = .012
            label.color = copy.deepcopy(marker.color)
            label.text = f'{cell[0]},{cell[1]}'
            markers.markers.append(label)
        if occupied_xy:
            occupied = Marker(header=header, ns='floor_slot_occupied_area', id=0,
                              type=Marker.POINTS, action=Marker.ADD)
            occupied.pose.orientation.w = 1.0
            occupied.scale.x = occupied.scale.y = .005
            occupied.color = ColorRGBA(r=1.0, g=.75, b=.05, a=.65)
            occupied.points = [self._point([(x+.5)*.005, (y+.5)*.005, z+.005])
                               for x, y in sorted(occupied_xy)]
            markers.markers.append(occupied)
        self._slot_pub.publish(markers)

    def _commit_floor_slots(self, request, response):
        """Commit a source-associated validated pose, or reset reservations."""
        with self._slot_lock:
            if request.reset:
                self._slot_layouts.clear()
                self._slot_pending.clear()
                self._slot_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
                response.success, response.status = True, 'FLOOR_SLOTS_RESET'
                return response
            pose = self._pose_matrix(request.selected_pose.pose)
            for pending in reversed(self._slot_pending):
                header, key, candidates, cells, z = pending[:5]
                shared = pending[5] if len(pending) > 5 else False
                masks = pending[6] if len(pending) > 6 else None
                if header != request.selected_pose.header:
                    continue
                for candidate_index, (candidate, used) in enumerate(zip(candidates, cells)):
                    if (np.linalg.norm(candidate.translation-pose[:3, 3]) < 1e-5 and
                            np.linalg.norm(candidate.rotation-pose[:3, :3]) < 1e-4):
                        entry = self._slot_layouts.get(key)
                        if entry is None:
                            break
                        fingerprint = (header.stamp.sec, header.stamp.nanosec,
                                       tuple(np.round(pose[:3, 3], 5)),
                                       tuple(np.round(pose[:3, :3], 5).ravel()))
                        if fingerprint not in entry['commits']:
                            occupied = entry.setdefault('occupied_xy', set())
                            footprint_mask = masks[candidate_index] if masks else set()
                            if not masks and len(pending) > 7:
                                footprint_mask = FloorSlots.reservation_mask(
                                    pending[7], candidate.rotation, candidate.translation)
                            if ((not shared and set(map(tuple, used)) & entry['reserved']) or
                                    shared and footprint_mask & occupied):
                                response.status = 'FLOOR_SLOT_RESERVATION_CONFLICT'
                                return response
                            occupied.update(footprint_mask)
                            entry['reserved'].update(map(tuple, used))
                            entry['commits'].add(fingerprint)
                        self._publish_floor_slots(header, entry['layout'], entry['reserved'], used, z,
                                                 entry.get('occupied_xy', ()))
                        response.success = True
                        response.status = 'FLOOR_SLOTS_COMMITTED'
                        response.occupied_slots = len(entry['reserved'])
                        return response
            response.status = 'FLOOR_SLOT_SOURCE_OR_POSE_MISMATCH'
        return response

    def _search_callback(self, request, response):
        try:
            if request.observation_header.frame_id != 'base_link':
                raise ValueError('SEARCH_REQUIRES_MANUAL_BASE_LINK_CLOUD')
            cloud_fields = ('object_cloud', 'scene_cloud', 'destination_cloud')
            for name in cloud_fields + ('bottom_cloud', 'stack_support_cloud'):
                cloud = getattr(request, name)
                if cloud.width and cloud.header != request.observation_header:
                    raise ValueError('SEARCH_OBSERVATION_ASSOCIATION_MISMATCH: ' + name)
            options = json.loads(request.options_json or '{}')
            if not isinstance(options, dict):
                raise ValueError('SEARCH_OPTIONS_MUST_BE_AN_OBJECT')
            forbidden = {'floor_slot_occupied_xy', 'floor_slot_layout', 'floor_slot_reserved_cells', 'object_points', 'organized_scene', 'destination_points',
                         'orientation_policy', 'bottom_points', 'stack_support_points',
                         'support_mode'}
            if forbidden.intersection(options):
                raise ValueError('SEARCH_OPTIONS_CANNOT_OVERRIDE_CLOUD_INPUTS')
            bottom = self._cloud_xyz(request.bottom_cloud) if request.bottom_cloud.width else None
            top = self._cloud_xyz(request.stack_support_cloud) if request.stack_support_cloud.width else None
            clouds = {name: self._cloud_xyz(getattr(request, name)) for name in cloud_fields}
            support_mode = request.support_mode or str(self.get_parameter('layout_support_mode').value)
            slot_key = None
            if support_mode == 'floor_slots':
                destination = clouds['destination_cloud']
                for name in ('floor_slot_adaptive', 'floor_slot_max_size_m',
                             'floor_slot_min_size_m'):
                    options.setdefault(name, self.get_parameter(name).value)
                slot_key = layout_key(destination) + json.dumps([
                    options.get(name, default) for name, default in (
                        ("floor_slot_adaptive", True), ("floor_slot_max_size_m", .125),
                        ("floor_slot_min_size_m", .06))])
                with self._slot_lock:
                    entry = self._slot_layouts.get(slot_key)
                    if entry:
                        options['floor_slot_layout'] = entry['layout']
                        options['floor_slot_reserved_cells'] = [list(c) for c in entry['reserved']]
                        options['floor_slot_occupied_xy'] = [list(c) for c in entry.get('occupied_xy', ())]
            candidates, diagnostics = search_placements(
                search_heightmap_placements,
                *[clouds[name] for name in cloud_fields],
                request.orientation_policy, bottom_points=bottom,
                stack_support_points=top, support_mode=support_mode,
                timing_sink=lambda seconds, success, status: publish_timing(
                    self._pipeline_timing_pub, 'search_placements', seconds,
                    request.observation_header, request.object_name, success, status),
                **options)
            if slot_key is not None and diagnostics.slot_layout:
                with self._slot_lock:
                    if slot_key not in self._slot_layouts:
                        if len(self._slot_layouts) >= 8:
                            self._slot_layouts.pop(next(iter(self._slot_layouts)))
                        self._slot_layouts[slot_key] = {
                            'layout': diagnostics.slot_layout, 'reserved': set(), 'commits': set()}
                    source = clouds['object_cloud']
                    centered_source = source-source.mean(axis=0)
                    shared = bool(options.get('floor_slot_share_unused_space', False))
                    # Whole-cell search needs no per-candidate occupancy masks.
                    # Compute only the selected mask at commit, retaining the
                    # identical occupancy geometry and idempotence behavior.
                    footprint_masks = ([FloorSlots.reservation_mask(
                        centered_source, candidate.rotation, candidate.translation)
                        for candidate in candidates] if shared else [])
                    self._slot_pending.append((request.observation_header, slot_key, candidates,
                                               diagnostics.slot_candidate_cells,
                                               diagnostics.heightmap_base_m, shared,
                                               footprint_masks, centered_source))
                    self._publish_floor_slots(request.observation_header, diagnostics.slot_layout,
                                              diagnostics.slot_blocked_cells,
                                              z=diagnostics.heightmap_base_m)
            response.diagnostics_json = diagnostics_json(diagnostics)
            response.search_seconds = diagnostics.elapsed_seconds
            for index, candidate in enumerate(candidates):
                message = PlacementCandidateMessage(id=index + 1, score=float(candidate.score),
                                                    orientation_policy=request.orientation_policy)
                message.object_pose.header = request.observation_header
                message.object_pose.pose.position = self._point(candidate.translation)
                q = self._quaternion_from_rotation(candidate.rotation)
                (message.object_pose.pose.orientation.x, message.object_pose.pose.orientation.y,
                 message.object_pose.pose.orientation.z, message.object_pose.pose.orientation.w) = map(float, q)
                response.candidates.append(message)
            response.success = bool(candidates)
            response.status = 'CANDIDATES_READY' if candidates else 'NO_STABLE_PACKING_CANDIDATE'
        except Exception as error:
            response.status = str(error)
        return response

    def _placement_callback(self, request, response):
        with self._lock:
            clouds = list(self._observation_clouds)
            pinned = getattr(self, '_pinned_layout_cloud', None)
            if (pinned is not None and self._stamp_key(pinned.header) ==
                    self._stamp_key(request.observation_header)):
                clouds.append(pinned)
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
            support_mode = request.support_mode or str(self.get_parameter('layout_support_mode').value)
            stack_top = None
            if support_mode not in ('auto', 'edge', 'stack', 'floor_slots'):
                raise RuntimeError('INVALID_SUPPORT_MODE')
            if backend != 'heightmap' and support_mode != 'auto':
                raise RuntimeError('SUPPORT_POLICY_REQUIRES_HEIGHTMAP')
            scene_points = organized.reshape(-1, 3)
            additional_obstacle_count = 0
            extra = request.additional_obstacles
            if extra.width or extra.data:
                xyz_fields = {field.name for field in extra.fields
                              if field.datatype == PointField.FLOAT32 and field.count == 1
                              and field.offset + 4 <= extra.point_step}
                if (backend != 'heightmap' or extra.header.frame_id != 'base_link'
                        or extra.is_bigendian or extra.height < 1 or extra.point_step < 12
                        or extra.row_step != extra.width * extra.point_step
                        or len(extra.data) != extra.row_step * extra.height
                        or len(extra.data) > 8 * 1024 * 1024
                        or not {'x', 'y', 'z'} <= xyz_fields):
                    raise RuntimeError('INVALID_ADDITIONAL_OBSTACLES')
                extra_points = self._cloud_xyz(extra)
                extra_points = extra_points[np.isfinite(extra_points).all(axis=1)]
                if not len(extra_points):
                    raise RuntimeError('EMPTY_ADDITIONAL_OBSTACLES')
                additional_obstacle_count = len(extra_points)
                scene_points = np.vstack((scene_points, extra_points))
            if backend == "heightmap":
                candidates, packing_diagnostics = self._search_placements_api(cloud.header, request.object_name,
                    object_points, scene_points, destination,
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
                    bottom_points=bottom_points,
                    support_mode=support_mode, stack_support_points=stack_top,
                    floor_slot_adaptive=bool(self.get_parameter('floor_slot_adaptive').value),
                    floor_slot_max_size_m=float(self.get_parameter('floor_slot_max_size_m').value),
                    floor_slot_min_size_m=float(self.get_parameter('floor_slot_min_size_m').value),

                    stack_minimum_coverage=float(self.get_parameter(
                        "packing_stack_minimum_coverage").value),
                    adjacency_weight=float(self.get_parameter(
                        "packing_adjacency_weight").value),
                    timing_sink=lambda seconds, success, status: publish_timing(
                        getattr(self, '_pipeline_timing_pub', None), 'search_placements',
                        seconds, cloud.header, request.object_name, success, status))
                inference_seconds = packing_diagnostics.elapsed_seconds
                self.get_logger().info(
                    f'[PLACEMENT TIMING] packing_seconds={inference_seconds:.6f} '
                    f'candidates={len(candidates)}')
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
                        f"stack_maximum_coverage="
                        f"{packing_diagnostics.stack_maximum_coverage:.3f} "
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
                f"support_mode={support_mode} "
                f"candidates={len(candidates)} object_points={len(object_points)} "
                f"additional_obstacles={additional_obstacle_count} "
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
            self.get_logger().info(f"[PLACEMENT] {status}")
        except Exception as exception:
            response.success = False
            response.status = f"PLACEMENT_FAILED: {exception}"
            response.candidates = []
            self._status_pub.publish(String(data=response.status))
            self.get_logger().error(response.status)
        return response

    def _infer_callback(self, _request, response):
        try:
            images, clouds, joints = self._capture_legacy_inputs()
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
            self.get_logger().info(f"[PLACEMENT] {status}")
            response.success = True
            response.message = status
        except Exception as exception:
            response.success = False
            response.message = f"PLACEMENT_FAILED: {exception}"
            self._status_pub.publish(String(data=response.message))
            self.get_logger().error(response.message)
        return response


def main():
    rclpy.init()
    node = PlacementNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
