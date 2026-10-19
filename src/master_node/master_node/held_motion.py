"""Master-owned, ordered execution of freshly validated payload paths."""

from master_node.gripper_widths import grasp_gripper_widths

import copy
import math
import time

from my_actions.srv import HeldMotionStage, PlanPlacement, PrepareHeldMotion, PrepareHeldScene

from rclpy.serialization import serialize_message


from master_node.joint_command_calibration import command_pose_reference, feedback_reference


class HeldMotion:
    """Keep one revision-bound plan; a failed actuator call never permits retry."""

    def __init__(self, execution):
        """Share the existing actuator lock and physical permission gate."""
        self.execution, self.node = execution, execution.node
        self.record = None
        self.revision = 0
        self.node.create_service(PrepareHeldMotion, '/master/prepare_held_motion', self.prepare,
                                 callback_group=execution.group)
        for name in ('execute_lift', 'execute_placement_transfer',
                     'execute_placement_descent', 'release_placed_object'):
            self.node.create_service(
                HeldMotionStage, '/master/'+name,
                lambda req, res, operation=name: self.stage(operation, req, res),
                callback_group=execution.group)

    @staticmethod
    def signature(refined):
        """Bind every path to the actual refined attachment geometry."""
        return (tuple(refined['grasp_joints']), refined['width'],
                repr(refined['grasp_pose']), repr(refined['observation_header']),
                tuple(refined['seed_base']), tuple(refined['closing_axis']))

    def prepare(self, request, response):
        """Plan before closing, or at confirmed contact; never trust supplied paths."""
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            key = request.object_name.casefold()
            if self.node.state != 'IDLE' or self.node._held_object not in (None, key):
                raise RuntimeError('MASTER_NODE_BUSY_OR_DIFFERENT_PAYLOAD')
            if self.record and self.record['phase'] not in ('PREPARED', 'RELEASED'):
                raise RuntimeError('HELD_MOTION_ALREADY_STARTED')
            if self.execution.grasp_stages.phases.get(key) not in ('PLANNED', 'CONTACT_CONFIRMED'):
                raise RuntimeError('REFINED_APPROACH_PLAN_REQUIRED')
            with self.execution._cache_lock:
                cached = self.execution._cache.get(key)
                if not cached or time.monotonic()-cached[0] > min(60, cached[1].ttl_seconds or 60):
                    raise RuntimeError('NO_FRESH_SOURCE_OBSERVATION')
                source = copy.deepcopy(cached[1])
            refined = self.node.refined_grasp_plans[key]
            if not source.observation_cloud.width or not source.source_mask.width:
                raise RuntimeError('CACHED_PAYLOAD_OBSERVATION_REQUIRED')
            distance = request.lift_distance_m or .05
            if not math.isfinite(distance) or distance <= 0:
                raise ValueError('LIFT_DISTANCE_MUST_BE_FINITE_AND_POSITIVE')
            geometry = self.node.step_runtime._call('prepare_held_scene', PrepareHeldScene.Request(
                observation_header=source.observation_header, cloud=source.observation_cloud,
                source_mask=source.source_mask, semantic_mask=source.semantic_mask,
                destination_mask=request.destination_mask))
            if not geometry.success:
                raise RuntimeError(geometry.status)
            candidate = copy.deepcopy(self.node.cached_grasp_plans[key]['candidate'])
            candidate.grasp_pose = copy.deepcopy(refined['grasp_pose'])
            candidate.grasp_pose.header = copy.deepcopy(source.observation_header)
            candidate.grasp_joints = list(refined['grasp_joints'])
            candidate.jaw_opening_m = float(refined['width'])
            candidate.pregrasp_joints = list(refined['pregrasp_joints'])
            plan = self.node.step_runtime._call('plan_placement', PlanPlacement.Request(
                object_name=request.object_name, grasp_candidate=candidate,
                source_object_center=geometry.source_object_center,
                source_object_cloud=geometry.source_object_cloud,
                placement_scene=geometry.collision_scene,
                destination_support_cloud=geometry.destination_support_cloud,
                candidates=request.candidates, orientation_policy='upright', support_mode='edge',
                use_layout_scene=True, plan_pick_place_motion=True,
                lift_only=request.lift_only, lift_distance_m=float(distance)))
            if not plan.success or not plan.motion_success:
                raise RuntimeError('HELD_MOTION_PLAN_REJECTED: '+plan.status)
            for flat in ([plan.lift_path_flat] if request.lift_only else
                         [plan.lift_path_flat, plan.transfer_path_flat, plan.descent_path_flat]):
                self.path(flat)
            if plan.selected_grasp_candidate.id != candidate.id:
                raise RuntimeError('HELD_PLAN_GRASP_ASSOCIATION_MISMATCH')
            paths = [self.path(plan.lift_path_flat)]
            if not request.lift_only:
                paths += [self.path(plan.transfer_path_flat), self.path(plan.descent_path_flat)]
            expected = list(candidate.grasp_joints)
            for path in paths:
                if max(abs(a-b) for a, b in zip(path[0], expected)) > .02:
                    raise RuntimeError('HELD_SEGMENT_START_DISCONTINUITY')
                expected = path[-1]
            self.revision += 1
            self.record = {'key': key, 'revision': self.revision, 'plan': plan,
                           'prepared': time.monotonic(), 'phase': 'PREPARED',
                           'signature': self.signature(refined), 'lift_only': request.lift_only,
                           'expected': list(candidate.grasp_joints)}
            response.success = True
            response.plan_revision = self.revision
            response.plan_cdr = serialize_message(plan)
            response.status = 'HELD_PLAN_PREPARED: '+plan.motion_status
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self.execution._lock.release()
        return response

    def path(self, values):
        """Reject malformed or out-of-limit paths before any actuator command."""
        values = list(values)
        if not values or len(values) % 6:
            raise ValueError('INVALID_HELD_TRAJECTORY')
        result = [values[i:i+6] for i in range(0, len(values), 6)]
        for joints in result:
            self.execution._joints(joints)
        return result

    def feedback(self, expected, contact=True, settle_sec=0.):
        """Validate live state; permit bounded contact settling only while stationary."""
        deadline = time.monotonic()+settle_sec
        while True:
            now = time.monotonic()
            for name in ('joint_feedback_time', 'pose_feedback_time', 'gripper_feedback_time'):
                if now-getattr(self.node, name, 0) > 1.0:
                    raise RuntimeError('STALE_HARDWARE_FEEDBACK: '+name)
            if self.node.physical_collision_detected or self.node.collision_detected:
                raise RuntimeError('ACTIVE_COLLISION')
            actual = self.node.current_joint_states.position[:6]
            try:
                self.execution._joints(actual, command_feedback=True)
            except ValueError as error:
                raise ValueError('CURRENT_JOINT_FEEDBACK: '+str(error)) from error
            if max(abs(a-b) for a, b in zip(actual, feedback_reference(self.node, expected))) > .02:
                raise RuntimeError('HELD_PATH_START_STATE_MISMATCH')
            status = self.node.current_gripper_status
            valid = (status is not None and math.isfinite(status.width) and
                     math.isfinite(status.force) and status.width > .001 and
                     abs(status.force) >= .2)
            if not contact or valid:
                return
            if now >= deadline:
                detail = ('feedback unavailable' if status is None else
                          f'width={status.width:.6f}m force={status.force:.6f}N')
                raise RuntimeError('HELD_CONTACT_LOST: '+detail)
            time.sleep(min(.02, deadline-now))

    def stage(self, name, request, response):
        """Advance one microstep and leave the object held after lift."""
        session = getattr(self.execution, 'prepared_grasp_lift', None)
        if session and session.record and session.record.get('pick_place') and session.record['key'] == request.object_name.casefold():
            stage = {'execute_lift': 'lift', 'execute_placement_transfer': 'transfer',
                     'execute_placement_descent': 'descent', 'release_placed_object': 'release'}[name]
            return session.run_stages(request, response, (stage,))
        admitted = self.execution._lock.acquire(blocking=False)
        started = False
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            self.execution._guard()
            record = self.record
            key = request.object_name.casefold()
            if not record or record['key'] != key or record['revision'] != request.plan_revision:
                raise RuntimeError('HELD_PLAN_REVISION_MISMATCH')
            if self.node._held_object != key:
                raise RuntimeError('CONFIRMED_HELD_OBJECT_REQUIRED')
            if time.monotonic()-record['prepared'] > 120:
                raise RuntimeError('HELD_PLAN_EXPIRED')
            refined = self.node.refined_grasp_plans.get(key)
            if not refined or self.signature(refined) != record['signature']:
                raise RuntimeError('REFINED_ATTACHMENT_CHANGED')
            expected_phase = {'execute_lift': 'PREPARED',
                              'execute_placement_transfer': 'LIFTED',
                              'execute_placement_descent': 'AT_PREPLACE',
                              'release_placed_object': 'PLACED'}[name]
            if record['phase'] != expected_phase:
                raise RuntimeError('HELD_STAGE_ORDER_VIOLATION: expected '+expected_phase)
            if record['lift_only'] and name != 'execute_lift':
                raise RuntimeError('LIFT_TEST_STOPS_WITH_OBJECT_HELD')
            self.feedback(record['expected'], settle_sec=.3)
            plan = record['plan']
            if name == 'execute_lift':
                error = self.node._pose_error(command_pose_reference(
                    refined['grasp_pose'], getattr(self.node, 'joint6_command_offset_rad', 0.)),
                    self.node.current_tcp_pose)
                if error is None or error[0] > .003 or error[1] > .03:
                    raise RuntimeError('NOT_AT_REFINED_ATTACHMENT_POSE')
            if name == 'release_placed_object':
                error = self.node._pose_error(command_pose_reference(
                    plan.placement_ee_pose, getattr(self.node, 'joint6_command_offset_rad', 0.)),
                    self.node.current_tcp_pose)
                if error is None or error[0] > .003 or error[1] > .03:
                    raise RuntimeError('NOT_AT_VALIDATED_PLACEMENT_POSE')
                record['phase'] = 'RELEASE_PENDING'
                started = True
                self.node.state = 'MOVING'
                width = grasp_gripper_widths(refined['width'])[0]
                before_width = self.node.current_gripper_status.width
                if not self.node._command_gripper(width, self.node.gripper_open_force_n):
                    raise RuntimeError('RELEASE_COMMAND_FAILED')
                if not self.node._wait_for_gripper_release(before_width):
                    raise RuntimeError('RELEASE_FEEDBACK_UNCONFIRMED')
                self.node._held_object = None
                record['phase'] = 'RELEASED'
                response.gripper_released = True
            else:
                field, phase = {'execute_lift': ('lift_path_flat', 'LIFTED'),
                                'execute_placement_transfer':
                                    ('transfer_path_flat', 'AT_PREPLACE'),
                                'execute_placement_descent':
                                    ('descent_path_flat', 'PLACED')}[name]
                path = self.path(getattr(plan, field))
                record['phase'] = 'MOTION_PENDING'
                started = True
                self.node.state = 'MOVING'
                if not self.node.set_arm_speed(5):
                    raise RuntimeError('HELD_SPEED_COMMAND_FAILED')
                for joints in path:
                    self.feedback(record['expected'])
                    if self.node.physical_collision_detected or self.node.collision_detected:
                        raise RuntimeError('ACTIVE_COLLISION')
                    if not self.node.move_arm_to(joints, tolerance=.01):
                        raise RuntimeError('HELD_WAYPOINT_FAILED')
                    record['expected'] = list(joints)
                self.feedback(record['expected'])
                record['phase'] = phase
            response.success = response.motion_completed = True
            response.status = record['phase']+';'+plan.motion_status
        except Exception as error:
            if started:
                self.record['phase'] = 'STATE_UNKNOWN'
                self.node.state = 'ERROR'
            response.status = ('STATE_UNKNOWN: ' if started else '')+str(error)
        finally:
            if started:
                if not self.node.set_arm_speed(20):
                    self.record['phase'] = 'STATE_UNKNOWN'
                    self.node.state = 'ERROR'
                    response.success = False
                    response.status = 'STATE_UNKNOWN: ARM_SPEED_RESTORE_FAILED'
                elif self.node.state != 'ERROR':
                    self.node.state = 'IDLE'
            if admitted:
                self.execution._lock.release()
        return response
