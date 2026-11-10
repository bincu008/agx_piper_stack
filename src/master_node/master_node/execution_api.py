"""Explicit master state and execution microsteps using existing safeguards."""

import copy
import math
import threading
import time

from my_actions.srv import (
    CacheGraspPlan, CheckAndReplan, ExecuteTrajectory, GetArmState, GetCachedGrasp,
    GripperCommand, MoveTo, PrepareGraspInput, SelectGrasp, ToggleContinuousNavigation,
)

from rclpy.callback_groups import ReentrantCallbackGroup

from task_pipeline.recipes import affordance_from_description

from .grasp_cache import grasp_cache_ttl
from .grasp_stages import GraspStages
from .held_motion import HeldMotion
from .next_best_view import NextBestView
from .observe_view import ObserveView
from .prepared_grasp_lift import PreparedGraspLiftExecutor


class ExecutionAPI:
    """Expose execution without granting permission to bypass the hardware gate."""

    def __init__(self, node):
        """Keep cache and actuator ownership in the existing master process."""
        self.node = node
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache = {}
        self._current_pool = None
        self.group = ReentrantCallbackGroup()
        self.node._held_object = None
        self.grasp_stages = GraspStages(self)
        self.held_motion = HeldMotion(self)
        self.prepared_grasp_lift = PreparedGraspLiftExecutor(self)
        self.next_best_view = NextBestView(self)
        self.observe_view = ObserveView(self)
        for kind, name, callback in (
                (PrepareGraspInput, 'prepare_grasp_input', self._prepare),
                (MoveTo, 'move_to', self._move),
                (ExecuteTrajectory, 'execute_trajectory', self._trajectory),
                (GripperCommand, 'set_gripper', self._gripper),
                (GripperCommand, 'open_empty_hand', self._open_empty_hand),
                (SelectGrasp, 'select_grasp', self._select),
                (CacheGraspPlan, 'cache_grasp_plan', self._save),
                (GetCachedGrasp, 'get_cached_grasp', self._cached),
                (GetArmState, 'get_arm_state', self._state)):
            node.create_service(kind, '/master/' + name, callback, callback_group=self.group)

    def _prepare(self, request, response):
        """Validate observation masks and calculate seed coordinates via /master/prepare_grasp_input."""
        image, mask = request.image, request.mask
        box = [int(value) for value in request.pixel_box]
        if (image.width != mask.width or image.height != mask.height or
                mask.header.stamp != image.header.stamp or
                request.observation_header.stamp != image.header.stamp):
            response.status = 'GRASP_INPUT_OBSERVATION_MISMATCH'
            return response
        if not (0 <= box[0] < box[2] <= image.width and
                0 <= box[1] < box[3] <= image.height):
            response.status = 'INVALID_PIXEL_BOX'
            return response
        response.seed_u = int((box[0] + box[2]) // 2)
        response.seed_v = int((box[1] + box[3]) // 2)
        response.image_width, response.image_height = image.width, image.height
        response.affordance_type = affordance_from_description(request.target_description)
        response.success = True
        response.status = 'GRASP_INPUT_READY'
        return response

    def _guard(self, prepared=False):
        session = getattr(self, 'prepared_grasp_lift', None)
        record = getattr(session, 'record', None)
        if not prepared and record and record['phase'] not in ('PREPARED', 'RELEASED'):
            raise RuntimeError('PREPARED_EXECUTION_SESSION_OWNS_ARM')
        if not self.node.allow_physical_grasp_execution:
            raise RuntimeError('PHYSICAL_EXECUTION_DISABLED')
        if self.node.state != 'IDLE':
            raise RuntimeError('MASTER_NODE_BUSY')
        if self.node.physical_collision_detected or self.node.collision_detected:
            raise RuntimeError('ACTIVE_COLLISION')

    def _joints(self, values, *, command_feedback=False):
        if len(values) != 6 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError('SIX_FINITE_JOINTS_REQUIRED')
        for index, (value, (lower, upper)) in enumerate(zip(values, self.node.joint_limits), 1):
            if command_feedback and index == 6:
                # Raw feedback may represent an uncorrected starting state or
                # the command-only calibrated endpoint. Do not change stored
                # feedback, nominal planning bounds, or driver commands here.
                offset = getattr(self.node, 'joint6_command_offset_rad', 0.)
                lower += min(0., offset)
                upper += max(0., offset)
            if not lower <= float(value) <= upper:
                raise ValueError(
                    f'JOINT_LIMIT_VIOLATION: joint{index}={float(value):.6f} rad '
                    f'({math.degrees(float(value)):.3f} deg), '
                    f'allowed=[{lower:.6f}, {upper:.6f}] rad '
                    f'([{math.degrees(lower):.3f}, {math.degrees(upper):.3f}] deg)')

    def _move(self, request, response):
        """Command empty-hand arm motion to specified target joints via /master/move_to."""
        admitted = self._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            self._guard()
            if self.node._held_object is not None:
                raise RuntimeError('HELD_OBJECT_REQUIRES_PAYLOAD_VALIDATED_MOTION')
            self._joints(request.target_joints)
            backend = self.node.move_to_joints_callback(
                ToggleContinuousNavigation.Request(target_joints=list(request.target_joints)),
                ToggleContinuousNavigation.Response())
            response.success = backend.success
            response.status = 'TARGET_REACHED' if backend.success else 'MOVE_FAILED'
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self._lock.release()
        return response

    def _trajectory(self, request, response):
        """Validate and execute empty-hand joint trajectory via /master/execute_trajectory."""
        admitted = self._lock.acquire(blocking=False)
        claimed_state = False
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            self._guard()
            if self.node._held_object is not None:
                raise RuntimeError('HELD_OBJECT_REQUIRES_PAYLOAD_VALIDATED_MOTION')
            self._joints(request.expected_start_joints)
            values = list(request.path_flat)
            if not values or len(values) % 6:
                raise ValueError('INVALID_TRAJECTORY_SHAPE')
            for index in range(0, len(values), 6):
                self._joints(values[index:index + 6])
            current = self.node.current_joint_states
            if current is None or len(current.position) < 6:
                raise RuntimeError('ARM_STATE_UNAVAILABLE')
            if max(abs(a - b) for a, b in zip(
                    current.position[:6], request.expected_start_joints)) > .03:
                raise RuntimeError('TRAJECTORY_START_STATE_MISMATCH')
            self.node.state = 'BUSY'
            claimed_state = True
            # Ordinary empty-hand trajectories are revalidated by the existing
            # VAMP service. The bounded final grasp approach retains its dedicated
            # fixed-orientation executor; it is never sent through this API.
            query = CheckAndReplan.Request(
                current_joints=list(current.position[:6]), remaining_path_flat=values)
            plan = self.node.step_runtime._call('plan_motion', query)
            if not plan.success or not plan.path_flat or len(plan.path_flat) % 6:
                raise RuntimeError('TRAJECTORY_REVALIDATION_FAILED')
            for index in range(0, len(plan.path_flat), 6):
                self._joints(plan.path_flat[index:index + 6])
            self.node.state = 'MOVING'
            for index in range(0, len(plan.path_flat), 6):
                # A full RRTC path begins with measured feedback, not a new
                # command target. Do not apply command calibration to it again.
                if index == 0 and len(plan.path_flat) > 6 and max(abs(a-b) for a, b in zip(
                        plan.path_flat[:6], request.expected_start_joints)) < 1.e-5:
                    continue
                if not self.node.move_arm_to(list(plan.path_flat[index:index + 6])):
                    raise RuntimeError('TRAJECTORY_EXECUTION_FAILED')
            response.success = response.motion_completed = True
            response.status = 'TRAJECTORY_COMPLETE'
        except Exception as error:
            response.status = str(error)
        finally:
            if claimed_state:
                self.node.state = 'IDLE'
            if admitted:
                self._lock.release()
        return response

    def _open_empty_hand(self, request, response):
        """Open only, using the existing gripper executor and independent physical gate."""
        return self._gripper(request, response, opening_only=True)

    def _gripper(self, request, response, *, opening_only=False):
        """Command Piper gripper opening width and force via /master/set_gripper."""
        admitted = self._lock.acquire(blocking=False)
        claimed_state = False
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            self._guard()
            if (not math.isfinite(request.width_m) or not 0 <= request.width_m <= .10 or
                    not math.isfinite(request.force_n) or
                    not self.node.GRIPPER_MIN_FORCE_N <= request.force_n <=
                    self.node.GRIPPER_MAX_FORCE_N):
                raise ValueError('INVALID_GRIPPER_COMMAND')
            if request.width_m <= .001:
                raise RuntimeError('CLOSE_REQUIRES_VALIDATED_GRASP_STAGE')
            if self.node._held_object is not None:
                raise RuntimeError('RELEASE_REQUIRES_VALIDATED_PLACEMENT')
            if opening_only:
                if not request.wait_for_opening:
                    raise RuntimeError('OPEN_REQUIRES_POST_COMMAND_FEEDBACK')
                joints = getattr(self.node, 'current_joint_states', None)
                status = getattr(self.node, 'current_gripper_status', None)
                if joints is None or status is None:
                    raise RuntimeError('OPEN_REQUIRES_CURRENT_FEEDBACK')
                self.held_motion.feedback(joints.position[:6], contact=False)
                if (not math.isfinite(status.width) or not 0. <= status.width <= .10 or
                        request.width_m < status.width):
                    raise RuntimeError('EMPTY_HAND_OPEN_CANNOT_NARROW_GRIPPER')
            self.node.state = 'BUSY'
            claimed_state = True
            if not self.node._command_gripper(request.width_m, request.force_n):
                raise RuntimeError('GRIPPER_COMMAND_REJECTED')
            if request.wait_for_opening:
                response.feedback_confirmed = self.node._wait_for_gripper_opening(request.width_m)
                response.success = response.feedback_confirmed
                response.status = (
                    'OPENING_CONFIRMED' if response.success else 'OPENING_UNCONFIRMED')
            else:
                response.success = True
                response.status = 'COMMAND_ACCEPTED_CONTACT_NOT_CONFIRMED'
        except Exception as error:
            response.status = str(error)
        finally:
            if claimed_state:
                self.node.state = 'IDLE'
            if admitted:
                self._lock.release()
        return response

    @staticmethod
    def _safe(candidate):
        return (candidate.ik_valid and candidate.collision_checked and
                candidate.grasp_collision_valid and candidate.pregrasp_collision_valid and
                not candidate.pregrasp_collision_overridden and candidate.score > 0 and
                (candidate.measured_contacts_valid or
                 candidate.validation_status.startswith('CUP_BODY_MODEL_')))

    def _select(self, request, response):
        """Select highest-ranking safe grasp proposal via /master/select_grasp."""
        valid = [candidate for candidate in request.candidates if self._safe(candidate)]
        if valid:
            # Keep the engine's existing ranking order.
            response.candidate = copy.deepcopy(valid[0])
            response.success = True
            response.status = 'RANKED_GRASP_SELECTED'
        else:
            response.status = 'NO_VALID_GRASP'
        return response

    def _save(self, request, response):
        """Cache validated candidate grasp plans in the master store via /master/cache_grasp_plan."""
        admitted = self._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            session = getattr(self, 'prepared_grasp_lift', None)
            record = getattr(session, 'record', None)
            if record and record['phase'] != 'PREPARED':
                raise RuntimeError('PREPARED_EXECUTION_SESSION_OWNS_ARM')
            if not request.object_name or not request.candidates:
                raise ValueError('OBJECT_AND_VALID_GRASPS_REQUIRED')
            if (request.observation_header.stamp.sec == 0 and
                    request.observation_header.stamp.nanosec == 0):
                raise ValueError('SOURCE_OBSERVATION_REQUIRED')
            if not all(self._safe(candidate) for candidate in request.candidates):
                raise ValueError('UNVALIDATED_GRASP_CANNOT_BE_CACHED')
            key = request.object_name.casefold()
            if self.node._held_object == key:
                raise RuntimeError('CANNOT_REPLACE_HELD_ATTACHMENT')
            cloud, mask = request.observation_cloud, request.source_mask
            if cloud.width or mask.width:
                if (cloud.header != request.observation_header or cloud.height < 2 or
                        mask.header.stamp != cloud.header.stamp or
                        mask.width != cloud.width or mask.height != cloud.height or
                        not cloud.data or not mask.data or len(cloud.data) > 16*1024*1024):
                    raise ValueError('PAYLOAD_OBSERVATION_ASSOCIATION_MISMATCH')
            if request.semantic_mask.width and (
                    request.semantic_mask.header.stamp != request.observation_header.stamp or
                    request.semantic_mask.width != cloud.width or
                    request.semantic_mask.height != cloud.height):
                raise ValueError('SEMANTIC_MASK_OBSERVATION_MISMATCH')
            src_stamp = request.observation_header.stamp
            src_sec = getattr(src_stamp, 'sec', 0)
            src_nano = getattr(src_stamp, 'nanosec', 0)
            # ROS stamps and monotonic deadlines have different epochs.
            # Measure source age with the node's ROS clock, then retain it.
            source_age = None
            if hasattr(self.node, 'get_clock'):
                ros_now = self.node.get_clock().now().nanoseconds * 1.e-9
                delta = ros_now - (src_sec + src_nano * 1.e-9)
                if math.isfinite(delta) and delta >= 0.:
                    source_age = delta
            if source_age is None or source_age >= grasp_cache_ttl(request):
                raise ValueError('REFRESH_REQUIRED_SOURCE_OBSERVATION_AGE')
            with self._cache_lock:
                now = time.monotonic()
                # Only one unconsumed pool is public. Held attachment state is separate.
                for old_key in list(self._cache):
                    if old_key != self.node._held_object:
                        self._cache.pop(old_key)
                        self.node.cached_grasp_plans.pop(old_key, None)
                        self.node.refined_grasp_plans.pop(old_key, None)
                        self.grasp_stages.invalidate(old_key)
                self._cache[key] = (now-source_age, copy.deepcopy(request))
                self.node.cached_grasp_plans[key] = {
                    'name': request.object_name,
                    'header': copy.deepcopy(request.observation_header),
                    'candidate': copy.deepcopy(request.candidates[0])}
                self.node.refined_grasp_plans.pop(key, None)
                self.grasp_stages.invalidate(key)
                from .current_workspace import calibration_fingerprint
                joints = getattr(self.node, 'current_joint_states', None)
                self._current_pool = {
                    'key': key,
                    'object_name': request.object_name,
                    'created': now,
                    'source_stamp': copy.deepcopy(src_stamp),
                    'source_age_at_creation': source_age,
                    'ttl_seconds': grasp_cache_ttl(request),
                    'candidates_count': len(request.candidates),
                    'consumed': False,
                    'claimed_by': None,
                    'invalidated': False,
                    'baseline_joints': list(joints.position[:6]) if joints is not None else None,
                    'baseline_tcp': copy.deepcopy(getattr(self.node, 'current_tcp_pose', None)),
                    'calibration': calibration_fingerprint(self.node),
                }
            response.success = True
            response.status = 'GRASP_PLAN_CACHED_IN_MASTER'
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self._lock.release()
        return response

    def _cached(self, request, response):
        """Retrieve the freshest cached grasp plan for an object via /master/get_cached_grasp."""
        with self._cache_lock:
            self._cache = {key: value for key, value in self._cache.items()
                           if time.monotonic() - value[0] <= grasp_cache_ttl(value[1])}
            key = request.object_name.casefold()
            if not key:
                if len(self._cache) != 1:
                    response.status = 'AMBIGUOUS_OR_EMPTY_GRASP_CACHE'
                    return response
                key = next(iter(self._cache))
            record = self._cache.get(key)
            if record is None:
                response.status = 'NO_FRESH_CACHED_GRASP'
                return response
            source = record[1]
            response.object_name = source.object_name
            response.observation_header = copy.deepcopy(source.observation_header)
            response.candidates = copy.deepcopy(source.candidates)
            response.success = True
            response.status = 'CACHED_GRASP_READY'
        return response

    def claim_current_pool(self, session_id):
        """Atomically claim readiness for execution; make unconsumed pool unavailable for other tasks."""
        with self._cache_lock:
            pool = getattr(self, '_current_pool', None)
            if pool and pool['key'] == session_id:
                pool['consumed'] = True
                self._cache.pop(pool['key'], None)
                pool['claimed_by'] = session_id

    def observe_pool_feedback(self, kind, message):
        """Invalidate only an unconsumed source; active sessions keep their own evidence."""
        with self._cache_lock:
            pool = getattr(self, '_current_pool', None)
            if not pool or pool['consumed'] or pool['invalidated']:
                return
            elapsed = time.monotonic() - pool['created']
            initial_age = pool.get('source_age_at_creation')
            if (elapsed < 0. or initial_age is None or
                    elapsed + initial_age >= pool.get('ttl_seconds', 600.)):
                self._invalidate_pool_locked(pool, 'REFRESH_REQUIRED_EXPIRED')
                return
            reason = None
            if kind == 'calibration' and pool.get('calibration') != message:
                reason = 'CALIBRATION_CHANGED'
            if kind == 'joints' and pool.get('baseline_joints') is not None:
                values, reference = list(message.position[:6]), pool['baseline_joints']
                if (len(values) != 6 or len(reference) != 6 or
                        not all(math.isfinite(x) for x in values+reference) or
                        max(abs(a-b) for a, b in zip(values, reference)) > .005):
                    reason = 'ARM_MOVED_SINCE_POOL_CREATION'
            if kind == 'tcp' and pool.get('baseline_tcp') is not None:
                pose, original = message.pose, pool['baseline_tcp'].pose
                xyz = [getattr(pose.position, k) for k in 'xyz']
                ref_xyz = [getattr(original.position, k) for k in 'xyz']
                quat = [getattr(pose.orientation, k) for k in 'xyzw']
                ref_quat = [getattr(original.orientation, k) for k in 'xyzw']
                if (not all(math.isfinite(x) for x in xyz+quat+ref_xyz+ref_quat) or
                        abs(sum(x*x for x in quat)-1.) > .001 or
                        math.dist(xyz, ref_xyz) > .002 or
                        abs(sum(a*b for a, b in zip(quat, ref_quat))) < .99999):
                    reason = 'CAMERA_MOVED_SINCE_POOL_CREATION'
            if reason:
                self._invalidate_pool_locked(pool, reason)

    def _invalidate_pool_locked(self, pool, reason):
        pool['invalidated'] = True
        pool['invalidation_reason'] = reason
        self._cache.pop(pool['key'], None)
        self.node.cached_grasp_plans.pop(pool['key'], None)
        self.node.refined_grasp_plans.pop(pool['key'], None)
        self.grasp_stages.invalidate(pool['key'])

    def invalidate_current_pool(self, reason='SOURCE_INVALIDATED'):
        """Invalidate the current pool upon scene alteration or camera motion."""
        with self._cache_lock:
            pool = getattr(self, '_current_pool', None)
            if pool:
                self._invalidate_pool_locked(pool, reason)

    def get_current_pool_summary(self, now=None):
        """Read metadata of the single current pool using 600s freshness policy."""
        with self._cache_lock:
            pool = getattr(self, '_current_pool', None)
            if not pool:
                return {'available': False, 'fresh': False, 'reason': 'NO_CURRENT_GRASP_POOL',
                        'object_name': None, 'expires_in_seconds': 0.}
            now = time.monotonic() if now is None else now
            age = now - pool['created']
            initial_age = pool.get('source_age_at_creation')
            source_age = age + initial_age if initial_age is not None else None
            ttl = pool.get('ttl_seconds', 600.)
            expires_in = max(0., ttl - source_age) if source_age is not None else 0.
            if pool['consumed']:
                return {'available': False, 'fresh': False, 'reason': 'ALREADY_CONSUMED',
                        'object_name': pool['object_name'], 'consumed': True,
                        'candidates_count': pool['candidates_count'], 'expires_in_seconds': 0.}
            if pool['invalidated']:
                return {'available': False, 'fresh': False,
                        'reason': pool.get('invalidation_reason', 'SOURCE_INVALIDATED'),
                        'object_name': pool['object_name'], 'invalidated': True,
                        'candidates_count': pool['candidates_count'], 'expires_in_seconds': 0.}
            if source_age is None or age < 0.:
                return {'available': False, 'fresh': False,
                        'reason': 'SOURCE_CLOCK_UNVERIFIED',
                        'object_name': pool['object_name'], 'expires_in_seconds': 0.}
            if age >= ttl or source_age >= ttl:
                return {'available': False, 'fresh': False, 'reason': 'REFRESH_REQUIRED_EXPIRED',
                        'object_name': pool['object_name'], 'expires_in_seconds': 0.,
                        'candidates_count': pool['candidates_count']}
            return {'available': True, 'fresh': True, 'reason': 'FRESH_FOR_REUSE_ONLY',
                    'object_name': pool['object_name'], 'expires_in_seconds': expires_in,
                    'candidates_count': pool['candidates_count'], 'consumed': False}

    def _state(self, request, response):
        """Query current joint states, TCP pose, and gripper opening via /master/get_arm_state."""
        if self.node.current_joint_states is None or self.node.current_tcp_pose is None:
            response.status = 'ARM_STATE_UNAVAILABLE'
            return response
        if (len(self.node.current_joint_states.position) < 6 or
                not all(math.isfinite(float(value)) for value in
                        self.node.current_joint_states.position[:6])):
            response.status = 'INVALID_ARM_STATE'
            return response
        response.joints = copy.deepcopy(self.node.current_joint_states)
        response.tcp_pose = copy.deepcopy(self.node.current_tcp_pose)
        status = self.node.current_gripper_status
        response.gripper_width_m = 0.0 if status is None else float(status.width)
        response.success = True
        response.status = 'ARM_STATE_READY'
        return response
