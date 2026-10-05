"""Expose ordered grasp stages while retaining strict contact and motion checks."""

import copy
import math

from my_actions.srv import (
    CacheGraspRefinement, GraspStage, PlanCartesianGraspApproach, RefineGrasp,
)

from rclpy.serialization import deserialize_message


class GraspStages:
    """One explicit API per task stage; protective speed changes stay atomic."""

    def __init__(self, execution):
        """Share the master's actuator lock and authoritative cached geometry."""
        self.execution = execution
        self.node = execution.node
        self.phases = {}
        self.paths = {}
        self.node.create_service(
            CacheGraspRefinement, '/master/cache_grasp_refinement', self._cache,
            callback_group=execution.group)
        for name in ('move_pregrasp', 'plan_grasp_approach', 'open_grasp_gripper',
                     'execute_grasp_approach', 'close_grasp_gripper', 'verify_grasp_contact'):
            self.node.create_service(
                GraspStage, '/master/' + name,
                lambda req, res, operation=name: self._stage(operation, req, res),
                callback_group=execution.group)

    def invalidate(self, key):
        """Invalidate every dependent execution stage when a grasp is replaced."""
        self.phases.pop(key, None)
        self.paths.pop(key, None)

    def _cache(self, request, response):
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            key = request.object_name.casefold()
            source = self.node.cached_grasp_plans.get(key)
            refined = deserialize_message(bytes(request.refinement_cdr), RefineGrasp.Response)
            if (not refined.valid or source is None or
                    refined.candidate_id != source['candidate'].id):
                raise ValueError('REFINED_GRASP_ASSOCIATION_MISMATCH')
            values = list(refined.grasp_joints) + [refined.grasp_width_m]
            if (not all(math.isfinite(float(value)) for value in values) or
                    not 0 < refined.grasp_width_m <= .10):
                raise ValueError('INVALID_REFINED_GRASP_GEOMETRY')
            candidate = source['candidate']
            plan = {
                'candidate_id': int(refined.candidate_id),
                'observation_header': copy.deepcopy(refined.observation_header),
                'grasp_pose': copy.deepcopy(refined.grasp_pose),
                'grasp_joints': list(refined.grasp_joints),
                'pregrasp_pose': copy.deepcopy(candidate.pregrasp_pose),
                'pregrasp_joints': list(candidate.pregrasp_joints),
                'seed_base': [refined.corrected_seed_base.x, refined.corrected_seed_base.y,
                              refined.corrected_seed_base.z],
                'closing_axis': [refined.closing_axis_base.x, refined.closing_axis_base.y,
                                 refined.closing_axis_base.z],
                'approach_axis': [refined.approach_axis_base.x, refined.approach_axis_base.y,
                                  refined.approach_axis_base.z],
                'width': float(refined.grasp_width_m),
            }
            valid, reason = self.node._verify_refined_pregrasp(plan)
            if not valid:
                raise RuntimeError(reason)
            self.node.refined_grasp_plans[key] = plan
            self.invalidate(key)
            self.phases[key] = 'REFINED'
            response.success = True
            response.status = 'REFINED_GRASP_CACHED_IN_MASTER'
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self.execution._lock.release()
        return response

    def _stage(self, name, request, response):
        key = request.object_name.casefold()
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            self.execution._guard()
            if self.node._held_object is not None and name != 'verify_grasp_contact':
                raise RuntimeError('OBJECT_ALREADY_HELD')
            source = self.node.cached_grasp_plans.get(key)
            if source is None:
                raise RuntimeError('NO_CACHED_GRASP')
            candidate = source['candidate']
            if not self.execution._safe(candidate):
                raise RuntimeError('CACHED_GRASP_UNSAFE')
            if (candidate.validation_status.startswith('CUP_BODY_MODEL_') and
                    not self.node.allow_model_based_body_grasp_execution):
                raise RuntimeError('MODEL_DERIVED_BODY_GRASP_EXECUTION_DISABLED')
            if name == 'move_pregrasp':
                self.invalidate(key)
                radius = self.node.pregrasp_target_exclusion_radius_m
                center = None
                if radius > 0:
                    center = copy.deepcopy(candidate.contact_left)
                    center.x = (candidate.contact_left.x + candidate.contact_right.x) / 2
                    center.y = (candidate.contact_left.y + candidate.contact_right.y) / 2
                    center.z = (candidate.contact_left.z + candidate.contact_right.z) / 2
                if not self.node._run_grasp_joint_target(
                        candidate.pregrasp_joints, center, radius):
                    raise RuntimeError('PREGRASP_MOTION_FAILED')
                self.phases[key] = 'AT_PREGRASP'
                response.motion_completed = True
            else:
                refined = self.node.refined_grasp_plans.get(key)
                if refined is None:
                    raise RuntimeError('REFINED_GRASP_REQUIRED')
                if name in ('plan_grasp_approach', 'open_grasp_gripper', 'execute_grasp_approach'):
                    valid, reason = self.node._verify_refined_pregrasp(refined)
                    if not valid:
                        raise RuntimeError(reason)
                if name == 'plan_grasp_approach':
                    self._plan(key, source, refined)
                elif name == 'open_grasp_gripper':
                    self._phase(key, 'PLANNED')
                    width = min(.10, refined['width'] + .01)
                    if not self.node._command_gripper(width, self.node.gripper_open_force_n):
                        raise RuntimeError('GRIPPER_OPEN_COMMAND_FAILED')
                    if not self.node._wait_for_gripper_opening(width):
                        raise RuntimeError('GRIPPER_OPEN_FEEDBACK_FAILED')
                    self.phases[key] = 'OPENED'
                elif name == 'execute_grasp_approach':
                    self._phase(key, 'OPENED')
                    try:
                        if not self.node.set_arm_speed(5):
                            raise RuntimeError('GRASP_SPEED_COMMAND_FAILED')
                        valid, reason = self.node._execute_validated_final_approach(
                            self.paths[key])
                        if not valid:
                            raise RuntimeError(reason)
                        self.phases[key] = 'APPROACH_COMPLETE'
                        response.motion_completed = True
                    finally:
                        if not self.node.set_arm_speed(20):
                            self.phases.pop(key, None)
                            raise RuntimeError('ARM_SPEED_RESTORE_20_PERCENT_FAILED')
                elif name == 'close_grasp_gripper':
                    self._phase(key, 'APPROACH_COMPLETE')
                    pose_error = self.node._pose_error(
                        refined['grasp_pose'], self.node.current_tcp_pose)
                    if pose_error is None or pose_error[0] > .010 or pose_error[1] > .15:
                        raise RuntimeError('NOT_AT_REFINED_FINAL_GRASP')
                    if not self.node._command_gripper(0.0, 1.0):
                        raise RuntimeError('GRIPPER_CLOSE_COMMAND_FAILED')
                    self.phases[key] = 'CLOSE_COMMAND_SENT'
                elif name == 'verify_grasp_contact':
                    self._phase(key, 'CLOSE_COMMAND_SENT')
                    if not self.node._wait_for_gripper_contact():
                        raise RuntimeError('GRIPPER_CONTACT_UNCONFIRMED')
                    self.phases[key] = 'CONTACT_CONFIRMED'
                    self.node._held_object = key
                    response.gripper_contact = True
                    response.motion_completed = True
            response.success = True
            response.status = self.phases[key]
        except Exception as error:
            if admitted:
                self.invalidate(key)
            response.status = str(error)
        finally:
            if admitted:
                self.execution._lock.release()
        return response

    def _phase(self, key, expected):
        if self.phases.get(key) != expected:
            raise RuntimeError('GRASP_STAGE_ORDER_VIOLATION: expected ' + expected)

    def _plan(self, key, source, refined):
        request = PlanCartesianGraspApproach.Request(
            current_pregrasp_pose=self.node.current_tcp_pose,
            current_joints=list(self.node.current_joint_states.position[:6]),
            refined_grasp_pose=refined['grasp_pose'],
            grasp_width_m=refined['width'], target_observation_header=source['header'])
        (request.refined_contact_center.x, request.refined_contact_center.y,
         request.refined_contact_center.z) = refined['seed_base']
        (request.closing_axis.x, request.closing_axis.y,
         request.closing_axis.z) = refined['closing_axis']
        result = self.node.step_runtime._call('plan_final_approach', request)
        path = list(result.path_flat)
        if (not result.success or not result.waypoint_count or
                len(path) != result.waypoint_count * 6 or
                not all(math.isfinite(float(value)) for value in path)):
            raise RuntimeError('FINAL_APPROACH_PLAN_REJECTED: ' + result.failure_reason)
        self.paths[key] = [path[index:index + 6] for index in range(0, len(path), 6)]
        self.phases[key] = 'PLANNED'
