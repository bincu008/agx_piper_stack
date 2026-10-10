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

from task_pipeline.grasp_input import affordance_from_description

from .grasp_stages import GraspStages


class ExecutionAPI:
    """Expose execution without granting permission to bypass the hardware gate."""

    def __init__(self, node):
        """Keep cache and actuator ownership in the existing master process."""
        self.node = node
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache = {}
        self.group = ReentrantCallbackGroup()
        self.node._held_object = None
        self.grasp_stages = GraspStages(self)
        for kind, name, callback in (
                (PrepareGraspInput, 'prepare_grasp_input', self._prepare),
                (MoveTo, 'move_to', self._move),
                (ExecuteTrajectory, 'execute_trajectory', self._trajectory),
                (GripperCommand, 'set_gripper', self._gripper),
                (SelectGrasp, 'select_grasp', self._select),
                (CacheGraspPlan, 'cache_grasp_plan', self._save),
                (GetCachedGrasp, 'get_cached_grasp', self._cached),
                (GetArmState, 'get_arm_state', self._state)):
            node.create_service(kind, '/master/' + name, callback, callback_group=self.group)

    def _prepare(self, request, response):
        image, mask = request.image, request.mask
        box = list(request.pixel_box)
        if (image.width != mask.width or image.height != mask.height or
                mask.header.stamp != image.header.stamp or
                request.observation_header.stamp != image.header.stamp):
            response.status = 'GRASP_INPUT_OBSERVATION_MISMATCH'
            return response
        if not (0 <= box[0] < box[2] <= image.width and
                0 <= box[1] < box[3] <= image.height):
            response.status = 'INVALID_PIXEL_BOX'
            return response
        response.seed_u = (box[0] + box[2]) // 2
        response.seed_v = (box[1] + box[3]) // 2
        response.image_width, response.image_height = image.width, image.height
        response.affordance_type = affordance_from_description(request.target_description)
        response.success = True
        response.status = 'GRASP_INPUT_READY'
        return response

    def _guard(self):
        if not self.node.allow_physical_grasp_execution:
            raise RuntimeError('PHYSICAL_EXECUTION_DISABLED')
        if self.node.state != 'IDLE':
            raise RuntimeError('MASTER_NODE_BUSY')
        if self.node.physical_collision_detected or self.node.collision_detected:
            raise RuntimeError('ACTIVE_COLLISION')

    def _joints(self, values):
        if len(values) != 6 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError('SIX_FINITE_JOINTS_REQUIRED')
        for value, (lower, upper) in zip(values, self.node.joint_limits):
            if not lower <= float(value) <= upper:
                raise ValueError('JOINT_LIMIT_VIOLATION')

    def _move(self, request, response):
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

    def _gripper(self, request, response):
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
        admitted = self._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            if not request.object_name or not request.candidates:
                raise ValueError('OBJECT_AND_VALID_GRASPS_REQUIRED')
            if (request.observation_header.stamp.sec == 0 and
                    request.observation_header.stamp.nanosec == 0):
                raise ValueError('SOURCE_OBSERVATION_REQUIRED')
            if not all(self._safe(candidate) for candidate in request.candidates):
                raise ValueError('UNVALIDATED_GRASP_CANNOT_BE_CACHED')
            key = request.object_name.casefold()
            with self._cache_lock:
                self._cache[key] = (time.monotonic(), copy.deepcopy(request))
                self.node.cached_grasp_plans[key] = {
                    'name': request.object_name,
                    'header': copy.deepcopy(request.observation_header),
                    'candidate': copy.deepcopy(request.candidates[0])}
                self.node.refined_grasp_plans.pop(key, None)
                self.grasp_stages.invalidate(key)
            response.success = True
            response.status = 'GRASP_PLAN_CACHED_IN_MASTER'
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self._lock.release()
        return response

    def _cached(self, request, response):
        with self._cache_lock:
            self._cache = {key: value for key, value in self._cache.items()
                           if time.monotonic() - value[0] <= min(60, value[1].ttl_seconds or 60)}
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

    def _state(self, request, response):
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
