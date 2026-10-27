"""Master-owned, empty-hand next-view planning and revision-bound execution."""

import copy
import math
import threading
import time
import traceback

from geometry_msgs.msg import PoseStamped

from my_actions import srv

import numpy as np

from rclpy.qos import DurabilityPolicy, QoSProfile

from scipy.spatial.transform import Rotation

from sensor_msgs.msg import PointField

from sensor_msgs_py import point_cloud2

from visualization_msgs.msg import Marker, MarkerArray

from .gripper_widths import grasp_gripper_widths
from .prepared_plan_review import endpoint_markers, placed_clouds, pose_matrix, transform_pose


class NextBestView:
    """Use shared perception, IK, VAMP and trajectory APIs without private engines."""

    def __init__(self, execution):
        """Register a calculation API and a separate guarded execution API."""
        self.execution, self.node = execution, execution.node
        self.lock = threading.Lock()
        self.record, self.revision = None, 0
        self.preview = self.node.create_publisher(
            MarkerArray,
            '/pipeline/next_view_endpoint',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.publish_preview()
        self.node.create_service(
            srv.PlanNextBestView,
            '/master/plan_next_best_view',
            self.plan,
            callback_group=execution.group,
        )
        self.node.create_service(
            srv.HeldMotionStage,
            '/master/execute_next_best_view',
            self.execute,
            callback_group=execution.group,
        )

    def publish_preview(self, plan=None, prepared=None):
        """Publish the desired final TCP only; visualization cannot fail a plan."""
        try:
            markers = (
                endpoint_markers(plan.tcp_pose,
                                 '4 Clearance retreat (TCP)' if 'CLEARANCE' in plan.status
                                 else '4 Tray-view retreat (TCP)', 'tray_view_endpoint')
                if plan is not None
                else MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
            )
            if plan is not None and prepared is not None:
                placement = prepared['lift']
                candidate = prepared['candidate']
                target = pose_matrix(plan.tcp_pose.pose)
                origin = pose_matrix(placement.placement_ee_pose.pose)
                opening, _ = grasp_gripper_widths(candidate.jaw_opening_m)
                for index, pose in enumerate(placement.placement_finger_poses):
                    local = transform_pose(np.linalg.inv(origin), pose)
                    side = 1.0 if local.position.y >= 0.0 else -1.0
                    local.position.y += side * (opening - candidate.jaw_opening_m) * 0.5
                    finger = Marker(
                        header=plan.tcp_pose.header, ns='retreat_open_fingers',
                        id=10 + index, type=Marker.CUBE,
                        pose=transform_pose(target, local),
                        scale=copy.deepcopy(placement.placement_finger_dimensions),
                    )
                    finger.color.r, finger.color.g = 1.0, 0.7
                    finger.color.a = 0.85
                    markers.markers.append(finger)
            self.preview.publish(markers)
        except Exception as error:
            self.node.get_logger().error('Next-view preview unavailable: ' + str(error))

    def prepare_retreat(self, geometry, plan, request):
        """Plan against the predicted released scene before any actuator moves."""
        _, placed, _ = placed_clouds(
            geometry.source_object_cloud,
            plan.selected_grasp_candidate,
            plan,
            plan.selected_candidate.object_pose,
            request.source_object_center,
        )

        def xyz(cloud):
            fields = point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'))
            return np.column_stack([fields[a].reshape(-1) for a in ('x', 'y', 'z')])

        groups = []
        for cloud, flag in ((geometry.collision_scene, 0), (placed, 1)):
            points = xyz(cloud)
            points = points[np.isfinite(points).all(axis=1)]
            _, indices = np.unique(
                np.floor(points / 0.005).astype(np.int64), axis=0, return_index=True
            )
            groups.extend((*map(float, point), flag) for point in points[indices])
        if not groups:
            raise ValueError('EMPTY_PREDICTED_RETREAT_SCENE')
        fields = [
            PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
            for i, name in enumerate(('x', 'y', 'z'))
        ]
        fields.append(
            PointField(name='released_object', offset=12, datatype=PointField.UINT8, count=1)
        )
        scene = point_cloud2.create_cloud(placed.header, fields, groups)
        start = list(plan.descent_path_flat[-6:])
        opening, _ = grasp_gripper_widths(plan.selected_grasp_candidate.jaw_opening_m)
        withdrawal = self.call(
            'plan_lift',
            srv.PlanPlacement.Request(
                empty_hand=True,
                start_joints=start,
                jaw_opening_m=opening,
                lift_distance_m=0.03,
                lift_only=True,
                use_layout_scene=True,
                placement_scene=scene,
            ),
        )
        if not withdrawal.success or not withdrawal.motion_success:
            raise RuntimeError('PREFLIGHT_WITHDRAWAL_REJECTED: ' + withdrawal.status)
        if withdrawal.lift_ee_pose.header != scene.header:
            raise RuntimeError('WITHDRAWAL_SOURCE_MISMATCH')
        withdrawal_path = self.execution.held_motion.path(withdrawal.lift_path_flat)
        if (
            not withdrawal_path
            or max(abs(a - b) for a, b in zip(withdrawal_path[0], start)) > 1.0e-4
        ):
            raise RuntimeError('WITHDRAWAL_START_DISCONTINUITY')
        self.execution.prepared_grasp_lift.validate_commands(
            [('withdrawal', withdrawal_path)], start
        )
        start = withdrawal_path[-1]
        result = self.call(
            'plan_camera_retreat',
            srv.PlanCameraRetreat.Request(
                collision_scene=scene,
                start_joints=start,
                clearance_only=True,
                jaw_opening_m=opening,
            ),
        )
        if not result.success:
            raise RuntimeError('PREFLIGHT_RETREAT_REJECTED: ' + result.status)
        if result.tcp_pose.header != scene.header:
            raise RuntimeError('RETREAT_SOURCE_OR_CANDIDATE_MISMATCH')
        path = self.execution.held_motion.path(result.path_flat)
        if not path or max(abs(a - b) for a, b in zip(path[0], start)) > 1.0e-4:
            raise RuntimeError('RETREAT_START_DISCONTINUITY')
        self.execution.prepared_grasp_lift.validate_commands([('retreat', path)], start)
        return withdrawal, result

    def call(self, name, request):
        """Keep backend timing and routing in the shared catalog."""
        return self.node.step_runtime._call(name, request)

    def target(self, request):
        """Resolve a named location hint or an explicit base-frame target."""
        if request.target.header.frame_id:
            if request.target.header.frame_id != 'base_link':
                raise ValueError('VIEW_TARGET_REQUIRES_BASE_LINK')
            point = request.target.point
            name = request.target_name or 'explicit_view_target'
            target = np.array([point.x, point.y, point.z])
        else:
            memory = self.node.manipulation_memory.snapshot()
            records = memory['active_objects'] + list(memory['directory'].values())
            destination = memory['active_destination']
            if destination:
                records.insert(0, destination)
            name = request.target_name or (destination['name'] if destination else '')
            key = ' '.join(name.casefold().split())
            match = next(
                (r for r in records if ' '.join(r['name'].casefold().split()) == key), None
            )
            if match is None:
                prepared = self.execution.prepared_grasp_lift.record
                if (
                    not prepared
                    or prepared['phase'] != 'RELEASED'
                    or prepared.get('destination_name', '').casefold() != name.casefold()
                ):
                    raise ValueError('VIEW_TARGET_UNKNOWN: resolve_target or supply target')
                point = prepared['lift'].selected_candidate.object_pose.pose.position
                target = np.array([point.x, point.y, point.z])
            else:
                name, target = match['name'], np.asarray(match['position'], dtype=float)
        if not np.isfinite(target).all():
            raise ValueError('FINITE_VIEW_TARGET_REQUIRED')
        return name, target

    def idle_empty_hand(self):
        """Reject camera control during a held-object session."""
        if self.node.state != 'IDLE' or self.node._held_object is not None:
            raise RuntimeError('NEXT_VIEW_REQUIRES_IDLE_EMPTY_HAND')
        prepared = self.execution.prepared_grasp_lift.record
        if prepared and prepared['phase'] not in ('PREPARED', 'RELEASED'):
            raise RuntimeError('PREPARED_EXECUTION_SESSION_OWNS_ARM')

    def plan(self, request, response):
        """Capture fresh geometry and test bounded far-view alternatives without motion."""
        admitted = self.lock.acquire(blocking=False)
        actuator_lock = False
        try:
            if not admitted:
                raise RuntimeError('NEXT_VIEW_BUSY')
            actuator_lock = self.execution._lock.acquire(blocking=False)
            if not actuator_lock:
                raise RuntimeError('EXECUTION_API_BUSY')
            self.idle_empty_hand()
            self.record = None
            self.publish_preview()
            name, target = self.target(request)
            distance = float(request.camera_distance_m or 0.45)
            if not math.isfinite(distance) or not 0.30 <= distance <= 0.65:
                raise ValueError('VIEW_DISTANCE_REQUIRES_0.30_TO_0.65_METRES')
            observation = self.call('capture_observation', srv.CaptureObservation.Request())
            if not observation.success:
                raise RuntimeError('VIEW_CAPTURE_FAILED: ' + observation.status)
            joints = self.node.current_joint_states
            if joints is None or len(joints.position) < 6:
                raise RuntimeError('ARM_STATE_UNAVAILABLE')
            start = list(joints.position[:6])
            self.execution._joints(start, command_feedback=True)
            if request.allow_current_view:
                memory=self.node.manipulation_memory.snapshot()
                role=('destination' if memory['active_destination'] and
                      memory['active_destination']['name'].casefold()==name.casefold() else 'object')
                visibility=self.call('check_target_view',srv.CheckTargetView.Request(
                    name=name,role=role,observation_cloud=observation.cloud,
                    camera_transform=observation.camera_transform))
                if not visibility.success:
                    raise RuntimeError('CURRENT_VIEW_CHECK_FAILED: '+visibility.status)
                if visibility.status=='READY':
                    self.revision+=1
                    response.success=True
                    response.status='CURRENT_VIEW_ALREADY_SUITABLE'
                    response.plan_revision=self.revision
                    response.observation_header=observation.cloud.header
                    response.tcp_pose=copy.deepcopy(self.node.current_tcp_pose)
                    response.tcp_pose.header=observation.cloud.header
                    response.expected_start_joints=start
                    response.path_flat=start
                    self.record={'name':name,'revision':self.revision,
                                 'created':time.monotonic(),'result':copy.deepcopy(response)}
                    self.publish_preview(response)
                    return response
            started = time.monotonic()
            radii = list(
                dict.fromkeys((distance, max(0.30, distance - 0.05), max(0.30, distance - 0.10)))
            )
            for radius in radii:
                views = self.call(
                    'get_nbv_candidates',
                    srv.GetNBVPose.Request(
                        target_x=float(target[0]),
                        target_y=float(target[1]),
                        target_z=float(target[2]),
                        radius=radius,
                    ),
                )
                if not views.success:
                    continue
                if not len(views.candidate_x) == len(views.candidate_y) == len(views.candidate_z):
                    raise RuntimeError('VIEW_CANDIDATE_ARRAY_MISMATCH')
                for camera in list(zip(views.candidate_x, views.candidate_y, views.candidate_z))[
                    :8
                ]:
                    if time.monotonic() - started > 15.0:
                        raise RuntimeError('VIEW_PLANNING_BUDGET_EXHAUSTED')
                    response.candidates_tested += 1
                    query = srv.LookAt.Request(current_joints=start)
                    if not self.node.populate_look_at_ik_request(query, camera, target):
                        continue
                    query.look_at_x, query.look_at_y, query.look_at_z = map(float, target)
                    query.current_pose_x, query.current_pose_y, query.current_pose_z = map(
                        float, camera
                    )
                    solved = self.call('solve_pose', query)
                    if not solved.success or len(solved.joint_angles) != 6:
                        continue
                    route = self.call(
                        'plan_motion',
                        srv.CheckAndReplan.Request(
                            current_joints=start, remaining_path_flat=list(solved.joint_angles)
                        ),
                    )
                    if not route.success:
                        continue
                    path = self.execution.held_motion.path(route.path_flat)
                    if not path or len(path) > 4000:
                        continue
                    if max(abs(a - b) for a, b in zip(path[-1], solved.joint_angles)) > 1.0e-4:
                        continue
                    if max(abs(a - b) for a, b in zip(path[0], start)) > 1.0e-5:
                        path.insert(0, start)
                    self.execution.prepared_grasp_lift.validate_commands(
                        [('prepath', path)], start
                    )
                    response.observation_header = observation.cloud.header
                    response.tcp_pose = PoseStamped(
                        header=observation.cloud.header, pose=copy.deepcopy(query.tcp_pose)
                    )
                    q = query.tcp_pose.orientation
                    tcp = np.eye(4)
                    tcp[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                    tcp[:3, 3] = [
                        query.tcp_pose.position.x,
                        query.tcp_pose.position.y,
                        query.tcp_pose.position.z,
                    ]
                    optical = tcp @ self.node.hand_eye_matrix
                    camera_pose = PoseStamped(header=observation.cloud.header)
                    (
                        camera_pose.pose.position.x,
                        camera_pose.pose.position.y,
                        camera_pose.pose.position.z,
                    ) = optical[:3, 3].tolist()
                    quaternion = Rotation.from_matrix(optical[:3, :3]).as_quat()
                    (
                        camera_pose.pose.orientation.x,
                        camera_pose.pose.orientation.y,
                        camera_pose.pose.orientation.z,
                        camera_pose.pose.orientation.w,
                    ) = quaternion
                    response.camera_pose = camera_pose
                    self.revision += 1
                    response.plan_revision = self.revision
                    response.camera_distance_m = float(radius)
                    response.expected_start_joints = start
                    response.path_flat = [float(value) for waypoint in path for value in waypoint]
                    response.success, response.status = True, 'NEXT_VIEW_PLANNED_EMPTY_HAND'
                    self.record = {
                        'name': name,
                        'revision': self.revision,
                        'created': time.monotonic(),
                        'result': copy.deepcopy(response),
                    }
                    self.publish_preview(response)
                    return response
            raise RuntimeError('NO_REACHABLE_NEXT_VIEW')
        except Exception as error:
            self.node.get_logger().error('Next-view planning failed: ' + traceback.format_exc())
            response.status = str(error)
            return response
        finally:
            if actuator_lock:
                self.execution._lock.release()
            if admitted:
                self.lock.release()

    def execute(self, request, response):
        """Execute through the existing empty-hand API, without actuator retries."""
        admitted = self.lock.acquire(blocking=False)
        dispatched = False
        try:
            if not admitted:
                raise RuntimeError('NEXT_VIEW_BUSY')
            self.idle_empty_hand()
            self.execution._guard()
            record = self.record
            if (
                record is None
                or request.plan_revision != record['revision']
                or (
                    request.object_name
                    and request.object_name.casefold() != record['name'].casefold()
                )
            ):
                raise RuntimeError('NEXT_VIEW_PLAN_REVISION_MISMATCH')
            if time.monotonic() - record['created'] > 60.0:
                raise RuntimeError('NEXT_VIEW_PLAN_EXPIRED')
            plan = record['result']
            for field in ('joint_feedback_time', 'pose_feedback_time', 'gripper_feedback_time'):
                if time.monotonic() - getattr(self.node, field, 0.0) > 1.0:
                    raise RuntimeError('STALE_HARDWARE_FEEDBACK: ' + field)
            current = list(self.node.current_joint_states.position[:6])
            if (
                len(current) != 6
                or max(abs(a - b) for a, b in zip(current, plan.expected_start_joints)) > 0.02
            ):
                raise RuntimeError('NEXT_VIEW_START_STATE_CHANGED')
            self.record = None
            if plan.status=='CURRENT_VIEW_ALREADY_SUITABLE':
                response.success=response.motion_completed=True
                response.status='CURRENT_VIEW_ALREADY_SUITABLE'
                return response
            dispatched = True
            result = self.call(
                'execute_trajectory',
                srv.ExecuteTrajectory.Request(
                    path_flat=plan.path_flat, expected_start_joints=plan.expected_start_joints
                ),
            )
            response.success = response.motion_completed = (
                result.success and result.motion_completed
            )
            response.status = (
                'NEXT_VIEW_REACHED' if response.success else 'NEXT_VIEW_' + result.status
            )
            if not response.success and any(
                word in result.status.casefold()
                for word in ('execution_failed', 'state_unknown', 'timeout', 'timed out')
            ):
                self.node.state = 'STATE_UNKNOWN'
                response.status = 'STATE_UNKNOWN: ' + response.status
        except Exception as error:
            if dispatched:
                self.node.state = 'STATE_UNKNOWN'
            response.status = ('STATE_UNKNOWN: ' if dispatched else '') + str(error)
        finally:
            if admitted:
                self.lock.release()
        return response
