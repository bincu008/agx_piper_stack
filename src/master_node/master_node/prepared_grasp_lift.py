"""Plan a complete initial-observation grasp/lift before any actuator command."""

from master_node.gripper_widths import grasp_gripper_widths
from master_node.prepared_plan_review import PreparedPlanReview, endpoint_markers

import copy
import math
import time

from my_actions.srv import (
    ExecutePlan, GetExecutionState, PreparedGripperCommand,
    CheckAndReplan, HeldMotionStage, PlanCartesianGraspApproach, PlanPlacement,
    PrepareHeldScene, PreparedGraspLift, PreparePickPlace, CommitFloorSlots, PlanNextBestView,
)

from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import deserialize_message

from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import SetBool, Trigger


from master_node.joint_command_calibration import command_pose_reference, feedback_reference


from .destination_memory import mapped_inputs, check_capacity


class PreparedGraspLiftExecutor:
    """Freeze one complete plan; execution cannot invoke IK or replanning."""

    def __init__(self, execution):
        """Share actuator ownership with the existing master APIs."""
        self.execution, self.node = execution, execution.node
        self.record, self.revision = None, 0
        self.completed_record = None
        timeout = self.node.declare_parameter('prepared_execution_timeout_sec', 300.0).value
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('prepared_execution_timeout_sec must be finite and positive')
        self.node.create_service(HeldMotionStage, '/master/execute_withdrawal',
                                 self.execute_withdrawal, callback_group=execution.group)
        self.node.create_service(HeldMotionStage, '/master/execute_placement_retreat',
                                 self.execute_retreat, callback_group=execution.group)
        self.preview = self.node.create_publisher(
            MarkerArray, '/pipeline/prepared_grasp',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.review = PreparedPlanReview(self.node, self.preview)
        for name, callback in (('prepare_grasp_lift', self.prepare),
                               ('get_prepared_grasp_lift', self.get),
                               ('validate_prepared_grasp_lift', self.validate)):
            self.node.create_service(PreparedGraspLift, '/master/'+name, callback,
                                     callback_group=execution.group)
        for schema, name, callback in (
                (ExecutePlan, 'execute_plan', self.execute_segment),
                (PreparedGripperCommand, 'gripper_open', self.open_gripper),
                (PreparedGripperCommand, 'gripper_close', self.close_gripper),
                (GetExecutionState, 'get_execution_state', self.execution_state),
                (PlanPlacement, 'plan_lift', self.plan_lift)):
            self.node.create_service(schema, '/master/'+name, callback,
                                     callback_group=execution.group)
        for name, callback in (('prepare_pick_place', self.prepare),
                               ('get_prepared_pick_place', self.get_pick_place),
                               ('validate_prepared_pick_place', self.validate_pick_place)):
            self.node.create_service(PreparePickPlace, '/master/'+name, callback,
                                     callback_group=execution.group)
        self.node.create_service(SetBool, '/master/reconcile_empty_hand',
                                 self.reconcile_empty_hand, callback_group=execution.group)
        self.node.create_service(Trigger, '/master/reset_execution_session',
                                 self.reset_session, callback_group=execution.group)
        self.node.create_service(HeldMotionStage, '/master/execute_prepared_grasp_lift',
                                 self.execute, callback_group=execution.group)

    def source(self, key):
        """Require an exact, fresh source observation owned by the master."""
        with self.execution._cache_lock:
            cached = self.execution._cache.get(key)
            if not cached or time.monotonic()-cached[0] > min(60, cached[1].ttl_seconds or 60):
                raise RuntimeError('NO_FRESH_CACHED_GRASP')
            owner = self.node.cached_grasp_plans.get(key)
            if (not owner or owner['header'] != cached[1].observation_header or
                    owner['candidate'] != cached[1].candidates[0]):
                raise RuntimeError('SOURCE_CACHE_CHANGED')
            return copy.deepcopy(cached[1]), cached[0]

    def get(self, request, response):
        """Return readiness without silently calculating a replacement plan."""
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            record = self.record
            if not record or record['key'] != request.object_name.casefold():
                raise RuntimeError('NO_PREPARED_GRASP_LIFT')
            source, source_version = self.source(record['key'])
            if (record['phase'] != 'PREPARED' or time.monotonic()-record['created'] > 60 or
                    source_version != record['source_version'] or
                    source.observation_header != record['header'] or
                    source.candidates[0].id != record['source_candidate_id']):
                raise RuntimeError('PREPARED_GRASP_LIFT_STALE_OR_CONSUMED')
            response.success = True
            response.plan_revision = record['revision']
            response.status = ('ALL_MOTION_PLANNED;' + record['lift'].motion_status +
                               ';'+record['approach_status'] +
                               f";GRASP_CANDIDATES_TRIED={record.get('candidate_attempts', 1)}")
            if record.get('pick_place'):
                response.status += (';'+record['withdrawal_plan'].motion_status+
                                    ';RETREAT_FROZEN_SCENE_VALIDATED')
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self.execution._lock.release()
        return response

    def pick_place_result(self, response):
        """Expose only the selected pose from this master-owned complete plan."""
        if response.success:
            if not self.record or not self.record.get('pick_place'):
                response.success = False
                response.status = 'NO_PREPARED_PICK_PLACE'
            else:
                response.selected_pose = copy.deepcopy(self.record['lift'].selected_candidate.object_pose)
        return response

    def pick_place_request_matches(self, request, response):
        if response.success and (
                request.destination_name and request.destination_name != self.record['destination_name'] or
                request.support_mode and request.support_mode != self.record['support_mode']):
            response.success, response.status = False, 'PREPARED_DESTINATION_OR_POLICY_MISMATCH'
        return response

    def get_pick_place(self, request, response):
        return self.pick_place_request_matches(request, self.pick_place_result(self.get(request, response)))

    def validate_pick_place(self, request, response):
        response = self.pick_place_request_matches(request, self.pick_place_result(self.validate(request, response)))
        if response.success and self.record['support_mode'] == 'floor_slots':
            checked = self.node.step_runtime._call('commit_floor_slots', CommitFloorSlots.Request(
                selected_pose=response.selected_pose, validate_only=True))
            if not checked.success:
                response.success, response.status = False, 'PLACEMENT_SLOT_REJECTED: '+checked.status
        return response

    def validate(self, request, response):
        """Check cached handoff readiness without consuming it or commanding hardware."""
        response = self.get(request, response)
        if not response.success:
            return response
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            response.success = False
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            record = self.record
            if (not record or record['revision'] != response.plan_revision or
                    record['phase'] != 'PREPARED'):
                raise RuntimeError('PREPARED_GRASP_LIFT_STALE_OR_CONSUMED')
            if self.node.state != 'IDLE' or self.node._held_object is not None:
                raise RuntimeError('EMPTY_HAND_IDLE_REQUIRED')
            self.execution.held_motion.feedback(record['start_joints'], contact=False)
            self.check_pose(record['start_pose'])
            self.validate_commands(
                ((stage, record[stage]) for stage in self.motion_fields(record)),
                record['start_joints'])
            response.success = True
            response.status += ';CACHED_HANDOFF_VALIDATED_NO_ACTUATION'
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self.execution._lock.release()
        return response

    def prepare(self, request, response):
        """Validate approach, lift and pre-grasp transfer without moving hardware."""
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            if self.node.state != 'IDLE' or self.node._held_object is not None:
                raise RuntimeError('EMPTY_HAND_IDLE_REQUIRED')
            if self.record and self.record['phase'] not in ('PREPARED', 'RELEASED'):
                raise RuntimeError('PREVIOUS_GRASP_LIFT_ALREADY_STARTED')
            # A failed preparation must never expose an older successful plan.
            self.record = None
            self.publish_preview(None)
            key = request.object_name.casefold()
            source, source_version = self.source(key)
            self.execution.held_motion.feedback(self.node.current_joint_states.position[:6],
                                                contact=False)
            start_joints = list(self.node.current_joint_states.position[:6])
            start_pose = copy.deepcopy(self.node.current_tcp_pose)
            pick_place = isinstance(request, PreparePickPlace.Request)
            if pick_place:
                if (not request.destination_name.strip() or not request.candidates or request.observation_header != source.observation_header or
                        request.destination_mask.header.stamp != source.observation_header.stamp or
                        not all(math.isfinite(getattr(request.source_object_center, a)) for a in ('x', 'y', 'z'))):
                    raise RuntimeError('INVALID_PICK_PLACE_SOURCE_ASSOCIATION')
                if request.support_mode not in ('auto', 'edge', 'stack', 'floor_slots'):
                    raise RuntimeError('INVALID_PLACEMENT_SUPPORT_MODE')
            geometry = self.node.step_runtime._call('prepare_held_scene', mapped_inputs(
                getattr(self.node, 'manipulation_memory', None), PrepareHeldScene.Request(
                observation_header=source.observation_header, cloud=source.observation_cloud,
                source_mask=source.source_mask, semantic_mask=source.semantic_mask,
                destination_mask=request.destination_mask if pick_place else type(source.source_mask)()),
                request.destination_name if pick_place else ''))
            if not geometry.success:
                raise RuntimeError(geometry.status)
            if pick_place:
                memory = getattr(self.node, 'manipulation_memory', None)
                check_capacity(memory, geometry.source_object_cloud, request.destination_name)
            look_plan=None
            planning_start=start_joints
            if pick_place and request.look_plan_cdr:
                if len(request.look_plan_cdr)>1024*1024:
                    raise RuntimeError('LOOK_PLAN_TOO_LARGE')
                look_plan=deserialize_message(bytes(request.look_plan_cdr),PlanNextBestView.Response)
                owner=getattr(self.execution.next_best_view,'record',None)
                if (not owner or not look_plan.success or owner['result']!=look_plan or
                        owner['name']!=request.object_name or time.monotonic()-owner['created']>60 or
                        max(abs(a-b) for a,b in zip(start_joints,look_plan.expected_start_joints))>.02):
                    raise RuntimeError('LOOK_PLAN_SOURCE_OR_START_CHANGED')
                look_path=self.execution.held_motion.path(look_plan.path_flat)
                self.validate_commands([('look',look_path)],start_joints)
                planning_start=look_path[-1]
            failures = []
            planned = False
            budget_started = time.monotonic()
            for candidate_index, proposal in enumerate(source.candidates[:8]):
                if time.monotonic()-budget_started > 30:
                    failures.append('PREPARATION_SEARCH_BUDGET_EXHAUSTED')
                    break
                self.node.get_logger().info(
                    f"[PREPARED GRASP TRIAL] object={request.object_name!r} "
                    f"candidate={proposal.id} skill={proposal.affordance_family} "
                    f"opening={proposal.jaw_opening_m*1000:.1f}mm "
                    f"attempt={candidate_index+1}/{min(8, len(source.candidates))}")
                try:
                    candidate, approach, path, lift, lift_path, prepath = self.plan_candidate(
                        proposal, request, source, geometry, planning_start)
                    withdrawal, retreat = (
                        self.execution.next_best_view.prepare_retreat(geometry, lift, request)
                        if pick_place else (None, None))
                    planned = True
                    break
                except (RuntimeError, ValueError) as error:
                    failures.append(f'candidate={proposal.id}: {error}')
                    self.node.get_logger().info(
                        f"[PREPARED GRASP REJECTED] candidate={proposal.id} reason={error}")
            if not planned:
                raise RuntimeError('NO_COMPLETE_GRASP_LIFT_PLAN: '+' | '.join(failures))
            self.revision += 1
            self.record = {
                'key': key, 'revision': self.revision, 'created': time.monotonic(),
                'phase': 'PREPARED', 'header': copy.deepcopy(source.observation_header),
                'source_version': source_version, 'source_candidate_id': source.candidates[0].id,
                'candidate': lift.selected_grasp_candidate,
                'start_joints': start_joints, 'start_pose': start_pose,
                'prepath': prepath, 'approach': path, 'lift': lift, 'lift_path': lift_path,
                'alignment_waypoint_count': approach.alignment_waypoint_count,
                'approach_status': approach.failure_reason,
                'candidate_attempts': candidate_index+1, 'pick_place': pick_place}
            if look_plan is not None:
                self.record['look']=look_path
                self.record['look_plan']=look_plan
            if pick_place:
                self.record['withdrawal_plan'] = withdrawal
                self.record['withdrawal'] = self.execution.held_motion.path(withdrawal.lift_path_flat)
                self.record['retreat_plan'] = retreat
                self.record['retreat'] = self.execution.held_motion.path(retreat.path_flat)
                self.record['transfer'] = self.execution.held_motion.path(lift.transfer_path_flat)
                self.record['descent'] = self.execution.held_motion.path(lift.descent_path_flat)
                self.record['destination_name'] = request.destination_name
                if memory is not None and memory.destination is not None:
                    self.record['destination_id'] = memory.destination['object_id']
                self.record['support_mode'] = request.support_mode
                self.record['review_source'] = geometry.source_object_cloud
                self.record['review_center'] = request.source_object_center
                self.record['review_nominal'] = copy.deepcopy(next(
                    item.object_pose for item in request.candidates
                    if item.id == lift.selected_candidate.id))
            self.publish_preview(self.record['candidate'])
            if pick_place:
                self.execution.next_best_view.publish_preview(retreat, self.record)
            self.node.get_logger().info(
                f"[PREPARED GRASP SELECTED] object={request.object_name!r} "
                f"candidate={self.record['candidate'].id} "
                f"skill={self.record['candidate'].affordance_family} "
                f"opening={self.record['candidate'].jaw_opening_m*1000:.1f}mm "
                f"status={self.record['candidate'].validation_status}")
            response.success = True
            response.plan_revision = self.revision
            response.status = ('ALL_MOTION_PLANNED;' + lift.motion_status +
                               ';' + approach.failure_reason +
                               f';GRASP_CANDIDATES_TRIED={candidate_index+1}')
            if pick_place:
                response.selected_pose = copy.deepcopy(lift.selected_candidate.object_pose)
                response.status += ';'+withdrawal.motion_status+';RETREAT_FROZEN_SCENE_VALIDATED'
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self.execution._lock.release()
        return response

    def publish_preview(self, candidate):
        """Display only the complete prepared plan, clearing earlier attempts."""
        if not hasattr(self, 'preview'):
            return
        if hasattr(self, 'review'):
            if candidate is None:
                self.review.clear()
                return
            if self.record and self.record.get('pick_place'):
                try:
                    self.review.publish(self.record)
                except Exception as error:
                    self.review.clear()
                    logger = self.node.get_logger()
                    getattr(logger, 'warning', logger.info)(
                        f'[PREPARED REVIEW] unavailable: {error}')
                return
        if candidate is not None and hasattr(self, 'review') and self.record:
            try:
                self.review.lift.publish(endpoint_markers(
                    self.record['lift'].lift_ee_pose, '2 Lift', 'lift_endpoint'))
            except Exception as error:
                self.node.get_logger().warning(f'[LIFT PREVIEW] unavailable: {error}')
        markers = [Marker(action=Marker.DELETEALL)]
        if candidate is not None:
            for index, (name, pose, color) in enumerate((
                    ('prepared_pregrasp', candidate.pregrasp_pose, (0., .5, 1.)),
                    ('prepared_grasp', candidate.grasp_pose, (0., 1., .2)))):
                marker = Marker(header=pose.header, ns=name, id=index, type=Marker.ARROW,
                                action=Marker.ADD, pose=pose.pose)
                marker.scale.x, marker.scale.y, marker.scale.z = .06, .006, .006
                marker.color.r, marker.color.g, marker.color.b = color
                marker.color.a = 1.
                markers.append(marker)
        self.preview.publish(MarkerArray(markers=markers))

    def plan_candidate(self, candidate, request, source, geometry, start_joints):
        """Try one complete plan; a failed candidate cannot command actuators."""
        candidate = copy.deepcopy(candidate)
        if not self.execution._safe(candidate):
            raise RuntimeError('CACHED_GRASP_UNSAFE')
        query = PlanCartesianGraspApproach.Request(
            current_pregrasp_pose=candidate.pregrasp_pose,
            current_joints=list(candidate.pregrasp_joints),
            refined_grasp_pose=candidate.grasp_pose,
            final_grasp_joints=list(candidate.grasp_joints),
            grasp_width_m=float(candidate.jaw_opening_m),
            target_observation_header=source.observation_header,
            source_object_cloud=geometry.source_object_cloud,
            collision_scene=geometry.collision_scene)
        left, right = candidate.contact_left, candidate.contact_right
        center = [(getattr(left, n)+getattr(right, n))/2 for n in ('x', 'y', 'z')]
        closing = [getattr(left, n)-getattr(right, n) for n in ('x', 'y', 'z')]
        (query.refined_contact_center.x, query.refined_contact_center.y,
         query.refined_contact_center.z) = center
        (query.closing_axis.x, query.closing_axis.y, query.closing_axis.z) = closing
        approach = self.node.step_runtime._call('plan_final_approach', query)
        if not approach.success:
            raise RuntimeError('PREFLIGHT_APPROACH_REJECTED: '+approach.failure_reason)
        self.execution._joints(approach.planned_pregrasp_joints)
        candidate.pregrasp_joints = list(approach.planned_pregrasp_joints)
        candidate.pregrasp_pose = copy.deepcopy(approach.planned_pregrasp_pose)
        path = self.execution.held_motion.path(approach.path_flat)
        if len(path) != approach.waypoint_count:
            raise RuntimeError('INVALID_APPROACH_WAYPOINT_COUNT')
        if approach.alignment_waypoint_count >= approach.waypoint_count:
            raise RuntimeError('INVALID_ALIGNMENT_WAYPOINT_COUNT')
        # Freeze the attachment at the actual final Cartesian joint solution.
        candidate.grasp_joints = path[-1]
        full = isinstance(request, PreparePickPlace.Request)
        motion_request = PlanPlacement.Request(
            object_name=request.object_name, grasp_candidate=candidate,
            source_object_center=request.source_object_center if full else geometry.source_object_center,
            source_object_cloud=geometry.source_object_cloud,
            placement_scene=geometry.collision_scene,
            destination_support_cloud=geometry.destination_support_cloud,
            candidates=request.candidates if full else [],
            destination_name=request.destination_name if full else '',
            support_mode=request.support_mode if full else 'edge',
            validation_batch_size=8 if full else 0, retry_motion_failures=full,
            use_layout_scene=True, plan_pick_place_motion=True, lift_only=not full,
            lift_distance_m=float(request.lift_distance_m or .05))
        # Confirm departure through the same public lift API before asking for
        # destination motion. The complete placement backend also revalidates
        # its own lift/escape; no unchecked path can be supplied by a caller.
        lift = self.node.step_runtime._call('plan_lift', motion_request)
        if not lift.success or not lift.motion_success:
            raise RuntimeError('PREFLIGHT_LIFT_REJECTED: '+lift.status)
        if full:
            lift = self.node.step_runtime._call('plan_placement', motion_request)
            if not lift.success or not lift.motion_success:
                raise RuntimeError('PREFLIGHT_PLACEMENT_REJECTED: '+lift.status)
        if (lift.lift_ee_pose.pose.position.z-candidate.grasp_pose.pose.position.z <
                float(request.lift_distance_m or .05)-.002):
            raise RuntimeError('LIFT_CLEARANCE_SHORTFALL')
        lift_path = self.execution.held_motion.path(lift.lift_path_flat)
        if max(abs(a-b) for a, b in zip(lift_path[0], path[-1])) > .02:
            raise RuntimeError('APPROACH_LIFT_DISCONTINUITY')
        transfer = self.node.step_runtime._call('plan_motion', CheckAndReplan.Request(
            current_joints=start_joints, remaining_path_flat=list(candidate.pregrasp_joints),
            grasp_motion=True))
        if not transfer.success:
            raise RuntimeError('PREFLIGHT_PREGRASP_REJECTED')
        prepath = self.execution.held_motion.path(transfer.path_flat)
        if max(abs(a-b) for a, b in zip(prepath[-1], candidate.pregrasp_joints)) > .01:
            raise RuntimeError('PREGRASP_PATH_ENDPOINT_MISMATCH')
        stages = [('pregrasp', prepath), ('approach', path), ('lift', lift_path)]
        if full:
            transfer_path = self.execution.held_motion.path(lift.transfer_path_flat)
            descent_path = self.execution.held_motion.path(lift.descent_path_flat)
            for previous, following in ((lift_path, transfer_path), (transfer_path, descent_path)):
                if max(abs(a-b) for a, b in zip(previous[-1], following[0])) > .02:
                    raise RuntimeError('PLACEMENT_SEGMENT_DISCONTINUITY')
            stages.extend((('transfer', transfer_path), ('descent', descent_path)))
        self.validate_commands(stages, start_joints)
        if full and request.support_mode == 'floor_slots':
            checked = self.node.step_runtime._call('commit_floor_slots', CommitFloorSlots.Request(
                selected_pose=lift.selected_candidate.object_pose, validate_only=True))
            if not checked.success:
                raise RuntimeError('PLACEMENT_SLOT_REJECTED: '+checked.status)
        return candidate, approach, path, lift, lift_path, prepath

    def validate_commands(self, stages, start_joints):
        """Use one nominal command validator for preparation and cached handoff."""
        # Feedback tolerance must not enlarge actuator target bounds.
        command_limits = getattr(self.node, 'command_joint_limits', self.node.joint_limits)
        for stage, waypoints in stages:
            for index, joints in enumerate(waypoints):
                # An initial observation is not a new requested endpoint.
                if stage in ('look', 'pregrasp', 'prepath') and index == 0 and max(
                        abs(a-b) for a, b in zip(joints, start_joints)) < 1.e-5:
                    continue
                for joint, (value, bounds) in enumerate(zip(joints, command_limits)):
                    if not bounds[0]-1.e-5 <= value <= bounds[1]+1.e-5:
                        raise RuntimeError(
                            f'COMMAND_LIMIT_VIOLATION stage={stage} waypoint={index+1} '
                            f'joint{joint+1} target={value:.6f} bounds={bounds}')

    STAGES = ('pregrasp', 'gripper_open', 'approach', 'gripper_close', 'lift')
    FULL_STAGES = STAGES + ('transfer', 'descent', 'release')
    PHASES = ('PREPARED', 'AT_PREGRASP', 'OPENED', 'AT_GRASP', 'CONTACT_CONFIRMED',
              'LIFTED', 'AT_PREPLACE', 'PLACED', 'RELEASED')

    def stages(self, record):
        return (('look',) if record.get('look') else ()) + (
            self.FULL_STAGES + ('withdrawal', 'retreat') if record.get('pick_place') else self.STAGES)

    def motion_fields(self, record):
        return (('look',) if record.get('look') else ()) + ('prepath', 'approach', 'lift_path') + (('transfer', 'descent', 'withdrawal', 'retreat') if record.get('pick_place') else ())

    def plan_lift(self, request, response):
        """Plan either a held-object lift or a released empty-hand withdrawal."""
        query = copy.deepcopy(request)
        query.lift_only = query.plan_pick_place_motion = True
        # Lift has no destination candidates: placement batching would skip the
        # backend entirely and report NO_LIFT_FEASIBLE_GRASP.
        query.validation_batch_size = 0
        query.retry_motion_failures = False
        return self.node.step_runtime._call('plan_placement', query)

    def execute_withdrawal(self, request, response):
        """Execute the cached empty-hand lift after confirmed release."""
        return self.execute_released_segment(request, response, 'withdrawal')

    def execute_retreat(self, request, response):
        """Execute the cached distant view only after withdrawal finishes."""
        return self.execute_released_segment(request, response, 'retreat')

    def execute_released_segment(self, request, response, segment):
        """Follow one released-hand segment without capture, IK or replanning."""
        admitted = self.execution._lock.acquire(blocking=False)
        started = False
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            self.execution._guard(prepared=True)
            record = self.record
            if (not record or not record.get('pick_place') or
                    request.plan_revision != record['revision'] or
                    request.object_name not in ('', record['destination_name'], record['key'])):
                raise RuntimeError('PREPARED_'+segment.upper()+'_REVISION_MISMATCH')
            if (record['phase'] != 'RELEASED' or record.get(segment+'_completed') or
                    (segment == 'retreat' and not record.get('withdrawal_completed'))):
                raise RuntimeError(segment.upper()+'_REQUIRES_RELEASE_AND_ORDERED_PREDECESSOR')
            if self.node._held_object is not None:
                raise RuntimeError(segment.upper()+'_REQUIRES_EMPTY_HAND')
            if time.monotonic()-record.get('execution_started',record['created']) > self.execution_timeout():
                raise RuntimeError('EXECUTION_SESSION_TIMED_OUT')
            with self.execution._cache_lock:
                cached = self.execution._cache.get(record['key'])
            if cached and cached[0] != record['source_version']:
                raise RuntimeError('PREPARED_SOURCE_CHANGED')
            expected = record['descent'][-1] if segment == 'withdrawal' else record['withdrawal'][-1]
            self.execution.held_motion.feedback(expected, contact=False)
            opening, _ = grasp_gripper_widths(record['candidate'].jaw_opening_m)
            if abs(self.node.current_gripper_status.width-opening) > .005:
                if not self.node._wait_for_gripper_opening(opening):
                    raise RuntimeError(segment.upper()+'_RELEASE_OPENING_NOT_REACHED')
                self.execution.held_motion.feedback(expected, contact=False)
            start_pose = (record['lift'].placement_ee_pose if segment == 'withdrawal'
                          else record['withdrawal_plan'].lift_ee_pose)
            self.check_pose(start_pose, commanded=True)
            self.validate_commands([(segment,record[segment])],expected)
            self.node.state = 'MOVING'
            started = True
            if not self.node.set_arm_speed(8 if segment == 'withdrawal' else 20):
                raise RuntimeError('ARM_SPEED_COMMAND_FAILED')
            self.follow(record[segment], expected, False, segment)
            end_pose = (record['withdrawal_plan'].lift_ee_pose if segment == 'withdrawal'
                        else record['retreat_plan'].tcp_pose)
            self.check_pose(end_pose, commanded=True, settle=True)
            record[segment+'_completed'] = True
            record.setdefault('completed_stages',[]).append(segment)
            response.success = response.motion_completed = True
            response.status = 'PREPARED_'+segment.upper()+'_COMPLETE'
        except Exception as error:
            response.status = str(error)
            if started:
                self.record['phase'] = 'STATE_UNKNOWN'
                self.node.state = 'ERROR'
                response.status = 'STATE_UNKNOWN: '+response.status
        finally:
            if started:
                if not self.node.set_arm_speed(20):
                    self.record['phase'] = 'STATE_UNKNOWN'
                    self.node.state = 'ERROR'
                    response.success = False
                    response.status = 'STATE_UNKNOWN: ARM_SPEED_RESTORE_FAILED'
                elif self.record['phase'] != 'STATE_UNKNOWN':
                    self.node.state = 'IDLE'
                if response.success and segment == 'retreat':
                    try:
                        self.complete_locked()
                    except Exception as error:
                        response.success = False
                        response.status = 'SESSION_CLEANUP_FAILED: '+str(error)
            if admitted:
                self.execution._lock.release()
        return response

    def complete_locked(self):
        """Retire executable data after confirmed empty-hand retreat, not lift."""
        record = self.record
        if (not record or record['phase'] != 'RELEASED' or
                not record.get('withdrawal_completed') or not record.get('retreat_completed') or
                self.node._held_object is not None or self.node.state != 'IDLE'):
            raise RuntimeError('COMPLETE_EMPTY_HAND_SESSION_REQUIRED')
        self.completed_record = {
            'key': record['key'], 'revision': record['revision'],
            'header': copy.deepcopy(record['header']),
            'completed_stages': list(record.get('completed_stages', []))}
        with self.execution._cache_lock:
            keys = set(self.execution._cache)
            keys.update(self.node.cached_grasp_plans)
            keys.update(getattr(self.node, 'refined_grasp_plans', {}))
            self.execution._cache.clear()
            self.node.cached_grasp_plans.clear()
            getattr(self.node, 'refined_grasp_plans', {}).clear()
            stages = getattr(self.execution, 'grasp_stages', None)
            if stages:
                for key in keys:
                    stages.invalidate(key)
        self.execution.held_motion.record = None
        view = getattr(self.execution, 'next_best_view', None)
        if view is not None:
            view.record = None
        self.node.gripper_contact_evidence = None
        self.record = None
        # Destination/placed clouds, exclusive cells and API history are not
        # executable caches and must survive the next object's preparation.

    def execution_state(self, request, response):
        """Inspect the authoritative execution session without actuator commands."""
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            record = self.record
            if not record or record['key'] != request.object_name.casefold():
                completed = getattr(self, 'completed_record', None)
                if completed and completed['key'] == request.object_name.casefold():
                    response.success = True
                    response.status = 'EXECUTION_COMPLETE_CACHE_CLEARED'
                    response.phase = 'COMPLETE'
                    response.plan_revision = completed['revision']
                    response.observation_header = completed['header']
                    response.completed_stages = completed['completed_stages']
                    response.waiting_stages = []
                    response.expires_in_seconds = 0.
                    response.physical_execution_enabled = self.node.allow_physical_grasp_execution
                    return response
                raise RuntimeError('NO_PREPARED_GRASP_LIFT')
            response.plan_revision = record['revision']
            response.phase = record['phase']
            response.observation_header = record['header']
            start = record.get('execution_started', record['created'])
            limit = self.execution_timeout() if record.get('execution_started') else 60.
            response.expires_in_seconds = max(0., limit-(time.monotonic()-start))
            if not record.get('execution_started'):
                with self.execution._cache_lock:
                    cached = self.execution._cache.get(record['key'])
                    if cached:
                        response.expires_in_seconds = min(response.expires_in_seconds,
                            max(0., min(60., cached[1].ttl_seconds or 60.)-(time.monotonic()-cached[0])))
            response.completed_stages = list(record.get('completed_stages', []))
            response.waiting_stages = [s for s in self.stages(record)
                                       if s not in response.completed_stages]
            response.physical_execution_enabled = self.node.allow_physical_grasp_execution
            response.success = True
            response.status = 'STATE_UNKNOWN_REQUIRES_RECONCILIATION' if (
                record['phase'] == 'STATE_UNKNOWN') else 'EXECUTION_SESSION_READY'
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self.execution._lock.release()
        return response

    def execute_look(self, request, response):
        """Execute the frozen first view under the same prepared revision and gate."""
        admitted=self.execution._lock.acquire(blocking=False)
        started=False
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            self.execution._guard(prepared=True)
            record=self.record
            if (not record or not record.get('look') or record['key']!=request.object_name.casefold()
                    or request.plan_revision!=record['revision'] or record['phase']!='PREPARED'
                    or record.get('look_completed')):
                raise RuntimeError('PREPARED_LOOK_REVISION_OR_PHASE_MISMATCH')
            source, version=self.source(record['key'])
            if version!=record['source_version'] or time.monotonic()-record['created']>60:
                raise RuntimeError('PREPARED_SOURCE_CHANGED')
            if self.node._held_object is not None or self.node.state!='IDLE':
                raise RuntimeError('LOOK_REQUIRES_IDLE_EMPTY_HAND')
            self.execution.held_motion.feedback(record['start_joints'],contact=False)
            self.check_pose(record['start_pose'])
            self.validate_commands([('look',record['look'])],record['start_joints'])
            record['execution_started']=time.monotonic()
            record['execution_source']=source
            if record['look_plan'].status!='CURRENT_VIEW_ALREADY_SUITABLE':
                self.node.state='MOVING'
                started=True
                if not self.node.set_arm_speed(20):
                    raise RuntimeError('ARM_SPEED_COMMAND_FAILED')
                self.follow(record['look'],record['start_joints'],False,'look')
                self.check_pose(record['look_plan'].tcp_pose,commanded=True,settle=True)
            record['look_completed']=True
            record.setdefault('completed_stages',[]).append('look')
            response.success=response.motion_completed=True
            response.status='PREPARED_LOOK_COMPLETE'
            response.phase=record['phase']
        except Exception as error:
            response.status=str(error)
            if started:
                self.record['phase']='STATE_UNKNOWN'
                self.node.state='ERROR'
                response.status='STATE_UNKNOWN: '+response.status
        finally:
            if started:
                if not self.node.set_arm_speed(20):
                    self.record['phase']='STATE_UNKNOWN'
                    self.node.state='ERROR'
                    response.success=False
                    response.status='STATE_UNKNOWN: ARM_SPEED_RESTORE_FAILED'
                elif self.record['phase']!='STATE_UNKNOWN':
                    self.node.state='IDLE'
            if admitted:
                self.execution._lock.release()
        return response

    def execute_segment(self, request, response):
        """Execute exactly one complete frozen motion segment."""
        if request.segment == 'look':
            return self.execute_look(request,response)
        if request.segment not in ('pregrasp', 'approach', 'lift', 'transfer', 'descent'):
            response.status = 'INVALID_SEGMENT'
            return response
        return self.run_stages(request, response, (request.segment,))

    def open_gripper(self, request, response):
        """Open to the prepared jaw clearance with confirmed feedback."""
        return self.run_stages(request, response, ('gripper_open',))

    def close_gripper(self, request, response):
        """Close only at the validated grasp and confirm contact before attachment."""
        return self.run_stages(request, response, ('gripper_close',))

    def execute(self, request, response):
        """Compatibility composite uses the identical public stage implementations."""
        return self.run_stages(request, response, self.STAGES)

    def run_stages(self, request, response, stages):
        """Serialize stage execution; uncertain actuator outcomes prohibit retries."""
        admitted = self.execution._lock.acquire(blocking=False)
        started = False
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            self.execution._guard(prepared=True)
            record = self.record
            key = request.object_name.casefold()
            if (not record or record['key'] != key or
                    record['revision'] != request.plan_revision or
                    (not record.get('execution_started') and
                     time.monotonic()-record['created'] > 60)):
                raise RuntimeError('PREPARED_GRASP_LIFT_STALE_OR_CONSUMED')
            active_started = record.get('execution_started')
            if active_started:
                timeout = self.execution_timeout()
                if time.monotonic()-active_started > timeout:
                    raise RuntimeError('EXECUTION_SESSION_TIMED_OUT')
                with self.execution._cache_lock:
                    cached = self.execution._cache.get(key)
                if cached and cached[0] != record['source_version']:
                    raise RuntimeError('PREPARED_SOURCE_CHANGED')
                source, source_version = record['execution_source'], record['source_version']
            else:
                source, source_version = self.source(key)
            if (source_version != record['source_version'] or
                    source.observation_header != record['header'] or
                    source.candidates[0].id != record['source_candidate_id']):
                raise RuntimeError('PREPARED_SOURCE_CHANGED')
            candidate = record['candidate']
            if (candidate.validation_status.startswith('CUP_BODY_MODEL_') and
                    not self.node.allow_model_based_body_grasp_execution):
                raise RuntimeError('MODEL_DERIVED_BODY_GRASP_EXECUTION_DISABLED')
            for stage in stages:
                if stage not in self.stages(record):
                    raise RuntimeError('SEGMENT_NOT_IN_PREPARED_PLAN')
                index = self.FULL_STAGES.index(stage)
                if record['phase'] != self.PHASES[index]:
                    raise RuntimeError('WRONG_EXECUTION_PHASE: '+record['phase'])
                if index == 0 and record.get('look') and not record.get('look_completed'):
                    raise RuntimeError('PREGRASP_REQUIRES_COMPLETED_LOOK')
                expected = ((record['look'][-1] if record.get('look') else record['start_joints']) if index == 0 else
                            record['prepath'][-1] if index <= 2 else
                            record['approach'][-1] if index <= 4 else
                            record['lift_path'][-1] if index == 5 else
                            record['transfer'][-1] if index == 6 else record['descent'][-1])
                self.execution.held_motion.feedback(
                    expected, contact=index >= 4, settle_sec=.3 if index >= 4 else 0.)
                if index == 0:
                    if self.node._held_object is not None:
                        raise RuntimeError('EMPTY_HAND_REQUIRED')
                    if (record.get('look') and
                            record['look_plan'].status!='CURRENT_VIEW_ALREADY_SUITABLE'):
                        self.check_pose(record['look_plan'].tcp_pose,commanded=True)
                    else:
                        self.check_pose(record['start_pose'])
                elif index <= 2:
                    self.check_pose(candidate.pregrasp_pose, .010, .15, commanded=True)
                elif index <= 4:
                    self.check_pose(candidate.grasp_pose, commanded=True)
                else:
                    self.check_pose(record['lift'].lift_ee_pose if index == 5 else
                                    record['lift'].preplace_ee_pose if index == 6 else
                                    record['lift'].placement_ee_pose, commanded=True)
                if index >= 4 and self.node._held_object != key:
                    raise RuntimeError('HELD_OBJECT_MISMATCH')
                open_width, close_width = grasp_gripper_widths(candidate.jaw_opening_m)
                width = open_width if index in (1, 7) else close_width
                force = float(self.node.gripper_open_force_n) if index in (1, 7) else 1.
                if hasattr(request, 'width_m') and (
                        not math.isfinite(request.width_m) or
                        not math.isfinite(request.force_n) or
                        (request.width_m and abs(request.width_m-width) > 1.e-6) or
                        (request.force_n and abs(request.force_n-force) > 1.e-6)):
                    raise RuntimeError('GRIPPER_COMMAND_DOES_NOT_MATCH_PREPARED_PLAN')
                record.setdefault('execution_started', time.monotonic())
                if 'execution_source' not in record:
                    record['execution_source'] = source
                started = True
                self.node.state = 'MOVING'
                if stage in ('gripper_open', 'gripper_close', 'release'):
                    before_width = self.node.current_gripper_status.width
                    if not self.node._command_gripper(width, force):
                        raise RuntimeError('GRIPPER_COMMAND_FAILED')
                    confirmed = (self.node._wait_for_gripper_opening(width) if index == 1 else
                                 self.node._wait_for_gripper_release(before_width) if index == 7 else
                                 self.node._wait_for_gripper_contact())
                    if not confirmed:
                        raise RuntimeError('GRIPPER_OPEN_FEEDBACK_FAILED' if index == 1
                                           else 'GRIPPER_CONTACT_UNCONFIRMED')
                    if index == 3:
                        self.node._held_object = key
                    elif index == 7:
                        self.node._held_object = None
                else:
                    if not self.node.set_arm_speed(20 if index in (0, 5) else 8):
                        raise RuntimeError('ARM_SPEED_COMMAND_FAILED')
                    path = record[{'pregrasp': 'prepath', 'approach': 'approach',
                                   'lift': 'lift_path', 'transfer': 'transfer',
                                   'descent': 'descent'}[stage]]
                    self.follow(path, expected, index >= 4, stage)
                    self.check_pose(candidate.pregrasp_pose if index == 0 else
                                    candidate.grasp_pose if index == 2 else
                                    record['lift'].lift_ee_pose if index == 4 else
                                    record['lift'].preplace_ee_pose if index == 5 else
                                    record['lift'].placement_ee_pose,
                                    .010 if index == 0 else .003,
                                    .15 if index == 0 else .03, commanded=True, settle=True)
                record['phase'] = self.PHASES[index+1]
                record.setdefault('completed_stages', []).append(stage)
            response.success = True
            if hasattr(response, 'motion_completed'):
                response.motion_completed = True
            if hasattr(response, 'contact_confirmed'):
                response.contact_confirmed = self.node._held_object == key
            if hasattr(response, 'gripper_released'):
                response.gripper_released = record['phase'] == 'RELEASED'
            response.status = record['phase']+';'+record['lift'].motion_status
        except Exception as error:
            if admitted and self.record:
                self.record['last_failure'] = str(error)
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
            if hasattr(response, 'phase') and self.record:
                response.phase = self.record['phase']
            if admitted:
                self.execution._lock.release()
        return response

    def execution_timeout(self):
        """Return the bounded session timeout independently of preparation freshness."""
        return (self.node.get_parameter('prepared_execution_timeout_sec').value
                if hasattr(self.node, 'get_parameter') else 300.)

    def reset_session(self, request, response):
        """Flush a known empty-hand failed session without commanding actuators."""
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            key = self.recover_locked()
            response.success = True
            response.message = 'SESSION_RESET_READY_FOR_NEW_PLAN' if key else 'NO_SESSION'
        except Exception as error:
            response.message = str(error)
        finally:
            if admitted:
                self.execution._lock.release()
        return response

    def reconcile_empty_hand(self, request, response):
        """Accept explicit operator confirmation of a stationary empty physical hand."""
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            if not request.data:
                raise RuntimeError('OPERATOR_EMPTY_HAND_CONFIRMATION_REQUIRED')
            key = self.recover_locked(confirmed_empty=True)
            response.success = True
            response.message = 'SESSION_RESET_READY_FOR_NEW_PLAN' if key else 'NO_SESSION'
        except Exception as error:
            response.message = str(error)
        finally:
            if admitted:
                self.execution._lock.release()
        return response

    def recover_locked(self, confirmed_empty=False):
        """Clear only known idle empty-hand state; retain uncertain/held sessions."""
        record = self.record
        if not record:
            return None
        if not confirmed_empty and (
                record['phase'] not in ('PREPARED', 'AT_PREGRASP', 'OPENED', 'AT_GRASP') or
                self.node._held_object is not None or self.node.state != 'IDLE'):
            raise RuntimeError('STATE_RECONCILIATION_REQUIRED_SESSION_NOT_RESET')
        if self.node.physical_collision_detected or self.node.collision_detected:
            raise RuntimeError('ACTIVE_COLLISION_SESSION_NOT_RESET')
        self.execution.held_motion.feedback(
            self.node.current_joint_states.position[:6], contact=False)
        self.check_pose(self.node.current_tcp_pose)
        runtime = getattr(self.node, 'step_runtime', None)
        if runtime and hasattr(runtime, 'store'):
            import json
            if any(ctx['busy'] for ctx in json.loads(runtime.store.snapshot()[0])):
                raise RuntimeError('PIPELINE_BUSY_SESSION_NOT_RESET')
        if confirmed_empty:
            velocities = list(getattr(self.node.current_joint_states, 'velocity', []))[:6]
            if len(velocities) != 6 or not all(
                    math.isfinite(v) and abs(v) < .02 for v in velocities):
                raise RuntimeError('STATIONARY_ARM_FEEDBACK_REQUIRED')
            if self.node.state not in ('IDLE', 'ERROR'):
                raise RuntimeError('MASTER_NODE_BUSY')
        key = record['key']
        with self.execution._cache_lock:
            self.execution._cache.pop(key, None)
            self.node.cached_grasp_plans.pop(key, None)
            getattr(self.node, 'refined_grasp_plans', {}).pop(key, None)
        stages = getattr(self.execution, 'grasp_stages', None)
        if stages:
            stages.invalidate(key)
        runtime = getattr(self.node, 'step_runtime', None)
        if runtime and hasattr(runtime, 'store'):
            runtime.store.flush_execution_queues(key)
        self.record = None
        if confirmed_empty:
            self.node._held_object = None
            self.node.state = 'IDLE'
        self.publish_preview(None)
        return key

    def recover_after_failure(self, object_name=''):
        """Recover at an action boundary only; never retry an actuator operation."""
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            if not admitted:
                return 'EXECUTION_API_BUSY_SESSION_NOT_RESET'
            if object_name and self.record and self.record['key'] != object_name.casefold():
                return 'OTHER_EXECUTION_SESSION_PRESERVED'
            runtime = getattr(self.node, 'step_runtime', None)
            if self.record and runtime and hasattr(runtime, 'store'):
                runtime.store.flush_execution_queues(self.record['key'])
            key = self.recover_locked()
            return 'SESSION_RESET_READY_FOR_NEW_PLAN' if key else ''
        except Exception as error:
            return str(error)
        finally:
            if admitted:
                self.execution._lock.release()

    def check_pose(self, pose, position=.003, orientation=.03, commanded=False, settle=False):
        """Allow bounded endpoint feedback settling without resending commands."""
        if commanded:
            pose = command_pose_reference(pose, getattr(self.node, 'joint6_command_offset_rad', 0.))
        deadline = time.monotonic()+(2.0 if settle else 0.0)
        while True:
            now = time.monotonic()
            if settle:
                # Monitor through the final accepted feedback sample as well
                # as the wait; never resend commands while settling.
                self.execution.held_motion.feedback(
                    self.node.current_joint_states.position[:6],
                    contact=self.node._held_object is not None)
            error = self.node._pose_error(pose, self.node.current_tcp_pose)
            age = now-getattr(self.node, 'pose_feedback_time', 0.0)
            fresh = not settle or 0.0 <= age <= .25
            if (error is not None and all(math.isfinite(v) for v in error) and
                    error[0] <= position and error[1] <= orientation and fresh):
                return
            if not settle or now >= deadline:
                detail = ('pose feedback unavailable' if error is None else
                          f'position_error_m={error[0]:.6f} '
                          f'orientation_error_rad={error[1]:.6f}')
                raise RuntimeError('PREPARED_MOTION_POSE_MISMATCH: '+detail+
                                   f' feedback_age_sec={age:.3f}')
            time.sleep(.02)

    def follow(self, path, expected, contact, stage):
        """Issue the frozen joint waypoints without invoking any planning API."""
        if stage in ('look', 'approach', 'lift', 'transfer', 'descent', 'withdrawal', 'retreat'):
            return self.stream_approach(path, expected, contact=contact, stage=stage)
        for index, joints in enumerate(path):
            # The planner's first sample records the measured start, including
            # accepted encoder tolerance. It is not an actuator target.
            if index == 0 and max(abs(a-b) for a, b in zip(joints, expected)) < 1.e-5:
                continue
            self.execution.held_motion.feedback(expected, contact=contact)
            if self.node.physical_collision_detected or self.node.collision_detected:
                raise RuntimeError('ACTIVE_COLLISION')
            if not self.node.move_arm_to(joints, tolerance=.01):
                raise RuntimeError(
                    f'PREPARED_WAYPOINT_FAILED stage={stage} waypoint={index+1}: ' +
                    getattr(self.node, 'last_motion_failure', 'arrival not confirmed'))
            expected = list(joints)
        self.execution.held_motion.feedback(expected, contact=contact)
        return expected


    def stream_approach(self, path, expected, contact=False, stage='approach'):
        """Stream frozen approach/lift edges; wait only at the endpoint."""
        self.execution.held_motion.feedback(expected, contact=contact)
        # 100 Hz commands, bounded to 0.1575 rad/s per joint.
        # Interpolation remains on prevalidated joint edges; no new IK poses.
        period, velocity = 1./100., .1575

        def monitor():
            if self.node.physical_collision_detected or self.node.collision_detected:
                raise RuntimeError('ACTIVE_COLLISION')
            self.execution.held_motion.feedback(
                self.node.current_joint_states.position[:6], contact=contact)

        for target in path:
            count = max(1, math.ceil(max(abs(a-b) for a, b in zip(target, expected)) /
                                     (velocity*period)))
            start = list(expected)
            for index in range(1, count+1):
                deadline = time.monotonic()+period
                monitor()
                joints = (list(target) if index == count else
                          [a+(b-a)*index/count for a, b in zip(start, target)])
                if not self.node._publish_joint_state_command(arm_positions=joints):
                    raise RuntimeError(stage.upper()+'_COMMAND_PUBLICATION_REJECTED')
                time.sleep(max(0., deadline-time.monotonic()))
            expected = list(target)
        deadline = time.monotonic()+5.
        while True:
            monitor()
            actual = self.node.current_joint_states.position[:6]
            sent = feedback_reference(self.node, expected)
            if max(abs(a-b) for a, b in zip(actual, sent)) <= .01:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(stage.upper()+'_FINAL_ARRIVAL_NOT_CONFIRMED')
            time.sleep(period)
        self.execution.held_motion.feedback(expected, contact=contact)
        return expected
