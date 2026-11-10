"""Declarative calculation microsteps; the master resolves every result binding."""

from collections import deque
import copy
import json
import math
import time

from my_actions import srv
from my_actions.msg import GraspCandidate, PipelineStep, StepBinding
from my_actions.srv import PlanPlacement

from rclpy.serialization import serialize_message


def step(step_id, function, request, bindings=(), failure_policy='stop'):
    """Build a typed step with explicit predecessor fields."""
    return PipelineStep(
        step_id=step_id, function_name=function, request_cdr=serialize_message(request),
        bindings=[StepBinding(target_field=target, source_step_id=source, source_field=field)
                  for target, source, field in bindings], failure_policy=failure_policy)


def cup_to_tray(object_name='brown cup', destination_name='tray', include_placement=True,
                additional_obstacles=None, support_mode='floor_slots'):
    """Build the existing image-based grasp and pose-only placement calculation."""
    steps = [
        step('observe', 'capture_observation', srv.CaptureObservation.Request()),
        step('pin_grasp', 'pin_grasp_observation', srv.PinLayoutObservation.Request(),
             [('observation_header', 'observe', 'cloud.header')]),
        step('pin_placement', 'pin_placement_observation', srv.PinLayoutObservation.Request(),
             [('observation_header', 'observe', 'cloud.header')]),
        step('presence_cup', 'verify_presence', srv.VerifyPresence.Request(target=object_name),
             [('image', 'observe', 'image')]),
        step('ground_cup', 'ground_object', srv.GroundObject.Request(target=object_name),
             [('image', 'observe', 'image')]),
        step('segment_cup', 'segment_object', srv.SegmentGroundedImage.Request(),
             [('image', 'observe', 'image'), ('bbox', 'ground_cup', 'pixel_box')]),
        step('prepare_grasp', 'prepare_grasp_input',
             srv.PrepareGraspInput.Request(target_description=object_name),
             [('image', 'observe', 'image'), ('mask', 'segment_cup', 'mask'),
              ('pixel_box', 'ground_cup', 'pixel_box'),
              ('observation_header', 'observe', 'cloud.header')]),
        step('grasps', 'calculate_grasps', srv.FindGraspCandidates.Request(
            target_description=object_name),
            [('observation_header', 'observe', 'cloud.header'),
             ('target_mask', 'segment_cup', 'mask'),
             ('seed_u', 'prepare_grasp', 'seed_u'), ('seed_v', 'prepare_grasp', 'seed_v'),
             ('image_width', 'prepare_grasp', 'image_width'),
             ('image_height', 'prepare_grasp', 'image_height'),
             ('affordance_type', 'prepare_grasp', 'affordance_type')]),
        step('register_object', 'update_object', srv.UpdateObjectObservation.Request(
            name=object_name, detected=True),
            [('position.point', 'grasps', 'corrected_seed_base'),
             ('position.header', 'observe', 'cloud.header')]),
        step('select_grasp', 'select_grasp', srv.SelectGrasp.Request(),
             [('candidates', 'grasps', 'candidates')]),
        step('cache_grasp', 'cache_grasp_plan', srv.CacheGraspPlan.Request(
            object_name=object_name, candidates=[GraspCandidate()], ttl_seconds=600),
            [('candidates[0]', 'select_grasp', 'candidate'),
             ('observation_header', 'observe', 'cloud.header'),
             ('observation_cloud', 'observe', 'cloud'),
             ('source_mask', 'grasps', 'validated_target_mask'),
             ('semantic_mask', 'segment_cup', 'mask')]),
    ]
    if include_placement:
        from sensor_msgs.msg import PointCloud2
        obstacles = additional_obstacles if additional_obstacles is not None else PointCloud2()
        steps.extend([
            step('presence_tray', 'verify_presence', srv.VerifyPresence.Request(target=destination_name),
                 [('image', 'observe', 'image')]),
            step('ground_tray', 'ground_object', srv.GroundObject.Request(
                target='the complete visible placement surface of the ' + destination_name),
                [('image', 'observe', 'image')]),
            step('segment_tray', 'segment_object', srv.SegmentGroundedImage.Request(),
                 [('image', 'observe', 'image'), ('bbox', 'ground_tray', 'pixel_box')]),
            step('placements', 'find_placements', srv.FindPlacementCandidates.Request(
                object_name=object_name, destination_name=destination_name,
                orientation_policy='upright', additional_obstacles=obstacles,
                support_mode=support_mode),
                [('observation_header', 'observe', 'cloud.header'),
                 ('destination_mask', 'segment_tray', 'mask')]),
            step('placement_ik', 'plan_placement', srv.PlanPlacement.Request(
                object_name=object_name, destination_name=destination_name,
                orientation_policy='upright', validation_batch_size=8,
                retry_motion_failures=True, additional_obstacles=obstacles),
                [('grasp_candidate', 'select_grasp', 'candidate'),
                 ('source_object_center', 'placements', 'source_object_center'),
                 ('candidates', 'placements', 'candidates')]),
        ])
    if include_placement and support_mode == 'floor_slots':
        steps.append(step('commit_slots', 'commit_floor_slots', srv.CommitFloorSlots.Request(),
                          [('selected_pose', 'placement_ik', 'selected_candidate.object_pose')]))
    return steps


def prepare_grasp_lift(object_name, lift_distance_m=.05):
    """Calculate every path from the initial observation without moving."""
    if not math.isfinite(lift_distance_m) or lift_distance_m <= 0:
        raise ValueError('Lift distance must be finite and positive')
    steps = cup_to_tray(object_name, include_placement=False)
    # Preserve the validated pool for complete-plan selection, not just endpoint rank.
    cache = next(item for item in steps if item.step_id == 'cache_grasp')
    cache.bindings[0] = StepBinding(target_field='candidates',
                                    source_step_id='grasps', source_field='candidates')
    steps.append(step('prepare_motion', 'prepare_grasp_lift', srv.PreparedGraspLift.Request(
        object_name=object_name, lift_distance_m=float(lift_distance_m))))
    return steps


def grasp_and_lift(object_name, lift_distance_m=.05, use_cached=False):
    """Execute only a complete prepared plan, stopping with the object held."""
    if use_cached:
        steps = [step('prepare_motion', 'get_prepared_grasp_lift',
                      srv.PreparedGraspLift.Request(object_name=object_name))]
    else:
        steps = prepare_grasp_lift(object_name, lift_distance_m)
    steps.append(step('check_motion', 'validate_prepared_grasp_lift',
                      srv.PreparedGraspLift.Request(object_name=object_name)))
    steps.append(step('execution_state', 'get_execution_state',
                      srv.GetExecutionState.Request(object_name=object_name)))
    revision = [('plan_revision', 'prepare_motion', 'plan_revision')]
    for stage in ('pregrasp', 'gripper_open', 'approach', 'gripper_close', 'lift'):
        if stage.startswith('gripper_'):
            steps.append(step(stage, stage, srv.PreparedGripperCommand.Request(
                object_name=object_name), revision))
        else:
            steps.append(step(stage, 'execute_plan', srv.ExecutePlan.Request(
                object_name=object_name, segment=stage), revision))
    return steps



def prepare_pick_place(object_name, destination_name='tray', lift_distance_m=.05,
                       support_mode='floor_slots'):
    """Freeze a complete pick/place plan without consuming slots or actuating."""
    if not math.isfinite(lift_distance_m) or lift_distance_m <= 0:
        raise ValueError('Lift distance must be finite and positive')
    steps = cup_to_tray(object_name, destination_name, support_mode=support_mode)
    steps = [s for s in steps if s.step_id not in ('placement_ik', 'commit_slots')]
    cache = next(s for s in steps if s.step_id == 'cache_grasp')
    cache.bindings[0] = StepBinding(target_field='candidates', source_step_id='grasps',
                                   source_field='candidates')
    steps.append(step('plan_look', 'plan_next_best_view',
                      srv.PlanNextBestView.Request(target_name=object_name, allow_current_view=True),
                      [('target.header', 'observe', 'cloud.header'),
                       ('target.point', 'grasps', 'corrected_seed_base')]))
    steps.append(step('prepare_motion', 'prepare_pick_place', srv.PreparePickPlace.Request(
        object_name=object_name, destination_name=destination_name,
        support_mode=support_mode, lift_distance_m=float(lift_distance_m)),
        [('observation_header', 'observe', 'cloud.header'),
         ('source_object_center', 'placements', 'source_object_center'),
         ('candidates', 'placements', 'candidates'),
         ('destination_mask', 'segment_tray', 'mask'),
         ('look_plan_cdr', 'plan_look', '@response_cdr')]))
    steps.append(step('execution_state', 'get_execution_state', srv.GetExecutionState.Request(object_name=object_name)))
    return steps


def execute_cached_pick_place(object_name, destination_name='tray', support_mode='floor_slots'):
    """Execute only a reviewed complete plan through the shared stage APIs."""
    steps = [step('prepare_motion', 'get_prepared_pick_place', srv.PreparePickPlace.Request(
        object_name=object_name, destination_name=destination_name, support_mode=support_mode))]
    steps.append(step('check_motion', 'validate_prepared_pick_place',
                      srv.PreparePickPlace.Request(object_name=object_name)))
    revision = [('plan_revision', 'prepare_motion', 'plan_revision')]
    for stage in ('look', 'pregrasp', 'gripper_open', 'approach', 'gripper_close', 'lift'):
        schema = srv.PreparedGripperCommand if stage.startswith('gripper') else srv.ExecutePlan
        query = schema.Request(object_name=object_name)
        if schema is srv.ExecutePlan:
            query.segment = stage
        steps.append(step(stage, stage if stage.startswith('gripper') else 'execute_plan',
                          query, revision))
    for stage, function in (('transfer', 'execute_placement_transfer'),
                            ('descent', 'execute_placement_descent'),
                            ('release', 'release_placed_object')):
        steps.append(step(stage, function, srv.HeldMotionStage.Request(object_name=object_name),
                          revision))
    if support_mode == 'floor_slots':
        steps.append(step('commit_slots', 'commit_floor_slots', srv.CommitFloorSlots.Request(),
                          [('selected_pose', 'prepare_motion', 'selected_pose')]))
    # Execute frozen clearance after release; the next object gets its own look stage.
    steps.extend([
        step('withdrawal', 'execute_withdrawal', srv.HeldMotionStage.Request(
            object_name=destination_name), revision),
        step('move_tray_view', 'execute_placement_retreat', srv.HeldMotionStage.Request(
            object_name=destination_name), revision),
        step('observe_tray', 'capture_observation', srv.CaptureObservation.Request()),
    ])
    return steps


def pick_and_place(object_name, destination_name='tray', lift_distance_m=.05,
                   support_mode='floor_slots'):
    """Queue calculation and execution against one immutable prepared revision."""
    return (prepare_pick_place(object_name, destination_name, lift_distance_m, support_mode)
            + execute_cached_pick_place(object_name, destination_name, support_mode)[1:])


def with_active_memory(steps, object_name, destination_name='tray', scene_objects=(),
                       allow_view_motion=False, refresh_scene=False):
    """Acquire the selected view, then refresh its geometry before grasp calculation."""
    import copy
    result = [
        step('observe_initial', 'capture_observation', srv.CaptureObservation.Request()),
        step('scene_memory', 'perceive_scene' if refresh_scene else 'ensure_scene_memory',
             srv.PerceiveScene.Request(active_object_name=object_name,
                 object_names=list(scene_objects), destination_name=destination_name),
             [('image', 'observe_initial', 'image'),
              ('observation_cloud', 'observe_initial', 'cloud'),
              ('camera_transform', 'observe_initial', 'camera_transform')]),
        step('acquire_object_view', 'ensure_target_view' if allow_view_motion else 'require_target_view',
             srv.CheckTargetView.Request(name=object_name, role='object'),
             [('observation_cloud', 'observe_initial', 'cloud'),
              ('camera_transform', 'observe_initial', 'camera_transform')]),
        step('observe', 'capture_observation', srv.CaptureObservation.Request()),
        step('verify_object_view', 'require_target_view',
             srv.CheckTargetView.Request(name=object_name, role='object'),
             [('observation_cloud', 'observe', 'cloud'),
              ('camera_transform', 'observe', 'camera_transform')]),
    ]
    for item in steps:
        if item.step_id in ('observe', 'presence_cup', 'ground_cup', 'presence_tray', 'ground_tray'):
            continue
        item = copy.deepcopy(item)
        if item.step_id in ('segment_cup', 'segment_tray'):
            destination = item.step_id == 'segment_tray'
            item = step(item.step_id, 'resolve_target' if destination else 'refresh_active_target',
                        srv.ResolveManipulationTarget.Request(
                name=destination_name if destination else object_name,
                role='destination' if destination else 'object'),
                [('image', 'observe', 'image'),
                 ('observation_cloud', 'observe', 'cloud'),
                 ('camera_transform', 'observe', 'camera_transform')])
        elif item.step_id == 'prepare_grasp':
            for binding in item.bindings:
                if binding.source_step_id == 'ground_cup':
                    binding.source_step_id = 'segment_cup'
        elif item.step_id == 'register_object':
            for binding in item.bindings:
                if binding.target_field == 'position.point':
                    binding.source_step_id = 'segment_cup'
                    binding.source_field = 'position'
        elif item.step_id == 'grasps':
            item.function_name = 'calculate_active_grasps'
        elif item.step_id == 'placements':
            item.function_name = 'find_active_placements'
        result.append(item)
    return result


def affordance_from_description(description):
    """Select the requested contact region without assuming an object class."""
    target = description.lower()
    if 'handle' in target:
        return 'HANDLE'
    if any(word in target for word in ('rim', 'lip', 'edge')):
        return 'RIM'
    if any(word in target for word in ('body', 'side')):
        return 'BODY'
    return 'AUTO'


def calculate_grasps_with_recovery(request, call):
    """Retry an empty bowl AUTO result once at measured rim contacts."""
    original = call(request)
    if (original.success or (request.affordance_type or 'AUTO').upper() != 'AUTO' or
            'bowl' not in request.target_description.casefold() or
            not original.status.startswith('NO_VALID_GEOMETRIC_GRASP')):
        return original
    query = copy.deepcopy(request)
    query.affordance_type = 'RIM'
    recovered = call(query)
    if recovered.success:
        recovered.status += ';AUTO_TO_RIM_RECOVERY'
        return recovered
    recovered.status = 'AUTO_FAILED: ' + original.status + '; RIM_FAILED: ' + recovered.status
    return recovered


def pin_observation_with_recovery(request, call, wait=None):
    """Wait briefly for delivery of the exact captured frame, without recapture."""
    import time
    if wait is None:
        wait = time.sleep
    response = call(request)
    for delay in (0.05, 0.10, 0.20, 0.40):
        if response.success or not response.status.startswith((
                'LAYOUT_OBSERVATION_PENDING',
                'LAYOUT_SYNCHRONIZED_OBSERVATION_NOT_AVAILABLE')):
            break
        wait(delay)
        response = call(request)
    return response


def retain_captured_observation(response, call):
    """Retain the exact captured source before a remote client can introduce delays.

    Capture remains usable for vision if grasp retention is unavailable; its status
    explicitly reports that distinction. Only idempotent pin delivery is retried.
    """
    if not response.success:
        return response
    query = srv.PinLayoutObservation.Request(observation_header=response.cloud.header)
    try:
        retained = pin_observation_with_recovery(query, call)
        marker = ('GRASP_OBSERVATION_PINNED' if retained.success else
                  'GRASP_RETENTION_FAILED:' + retained.status)
    except Exception as error:
        marker = 'GRASP_RETENTION_FAILED:' + str(error)
    response.status += ';' + marker
    return response


def retry_preparation(result, attempt, maximum_attempts):
    """Request fresh sensor inputs only after a definite geometric rejection."""
    if result.success or attempt >= maximum_attempts or not result.results:
        return False
    failed = result.results[-1]
    if failed.function_name == 'calculate_grasps':
        return (failed.status.startswith('NO_VALID_GEOMETRIC_GRASP') or
                failed.status.startswith('AUTO_FAILED: NO_VALID_GEOMETRIC_GRASP'))
    return (failed.function_name == 'prepare_grasp_lift' and
            failed.status.startswith('NO_COMPLETE_GRASP_LIFT_PLAN:'))


def search_placements(search, *args, timing_sink=None, **kwargs):
    """Use the same height-map engine for every input adapter."""
    import time
    started = time.monotonic()
    try:
        result = search(*args, **kwargs)
    except Exception:
        if timing_sink:
            timing_sink(time.monotonic() - started, False, 'SEARCH_FAILED')
        raise
    if timing_sink:
        timing_sink(time.monotonic() - started, bool(result[0]),
                    'CANDIDATES_READY' if result[0] else 'NO_STABLE_PACKING_CANDIDATE')
    return result


def validate_placements(request, call_backend, check_cancel=lambda: None, timing_sink=None,
                        validate_selected=None):
    """Validate through one shared policy and measure exceptional exits too."""
    import time
    started = time.monotonic()
    try:
        return _validate_placements(request, call_backend, check_cancel, timing_sink, validate_selected)
    except Exception as error:
        if timing_sink:
            timing_sink(time.monotonic() - started, False, str(error))
        raise


def _validate_placements(request, call_backend, check_cancel, timing_sink, validate_selected):
    """Stop on full success, preserving alternatives after motion failures."""
    started = time.monotonic()
    candidates = list(request.candidates)
    grasps = [request.grasp_candidate, *request.alternative_grasps]
    batch_size = request.validation_batch_size or max(1, len(candidates))
    blocked = set(request.blocked_lift_grasp_ids)
    calls = []
    submitted = 0
    result = PlanPlacement.Response(status='NO_LIFT_FEASIBLE_GRASP')
    for index in range(0, len(candidates), batch_size):
        pending = deque([(candidates[index:index + batch_size], grasps, request.lift_distance_m)])
        while pending:
            check_cancel()
            goals, active, lift_distance = pending.popleft()
            if request.retry_motion_failures:
                active = [g for g in active if g.id not in blocked]
            if not active:
                continue
            query = copy.copy(request)
            query.candidates = goals
            query.lift_distance_m = lift_distance
            query.grasp_candidate = active[0]
            query.alternative_grasps = active[1:]
            query.validation_batch_size = 0
            query.retry_motion_failures = False
            query.blocked_lift_grasp_ids = []
            submitted += len(goals)
            result = call_backend(query)
            check_cancel()
            if result.success or (request.retry_motion_failures and result.motion_status):
                if (result.selected_candidate.id not in {g.id for g in goals} or
                        result.selected_grasp_candidate.id not in {g.id for g in active}):
                    result.success = False
                    result.status = 'PLACEMENT_RESPONSE_ASSOCIATION_MISMATCH'
                    result.motion_status = ''
                elif (result.success and request.plan_pick_place_motion and
                      not result.motion_success):
                    result.success = False
                    result.status = 'PLACEMENT_MOTION_NOT_VALIDATED'
            if result.success and validate_selected is not None:
                checked = validate_selected(result)
                if not checked.success:
                    result.success = result.motion_success = False
                    result.status = result.motion_status = 'PLACEMENT_SLOT_REJECTED: ' + checked.status
            calls.append({
                'success': result.success, 'status': result.status,
                'motion_status': result.motion_status,
                'grasp_id': result.selected_grasp_candidate.id,
                'candidate_id': result.selected_candidate.id,
                'lift_distance_m': float(lift_distance or .10),
                'pose_validation_seconds': result.pose_validation_seconds,
                'descent_search_seconds': result.descent_search_seconds,
                'lift_seconds': result.lift_seconds,
                'transfer_seconds': result.transfer_seconds,
            })
            if result.success or not request.retry_motion_failures:
                break
            if not result.motion_status:
                continue
            rejected = result.selected_grasp_candidate.id
            nominal_lift = float(request.lift_distance_m or .10)
            current_lift = float(lift_distance or .10)
            if (request.plan_pick_place_motion and
                    result.motion_status.startswith('TRANSFER_LIFT_ENDPOINT_REJECTED:') and
                    current_lift < nominal_lift + .02 - 1e-6):
                pending.appendleft((goals, active, min(current_lift + .01, nominal_lift + .02)))
                continue
            remaining = [g for g in active if g.id != rejected]
            if remaining:
                pending.appendleft((goals, remaining, lift_distance))
            if result.motion_status.startswith('LIFT_'):
                blocked.add(rejected)
            else:
                other_goals = [g for g in goals if g.id != result.selected_candidate.id]
                if other_goals:
                    pending.append((other_goals, [g for g in active if g.id == rejected], lift_distance))
        if result.success:
            break
    result.validation_seconds = time.monotonic() - started
    result.validated_candidate_count = submitted
    result.blocked_lift_grasp_ids = sorted(blocked)
    if not result.success:
        motion_failures = list(dict.fromkeys(
            call['motion_status'] for call in calls
            if not call['success'] and call['motion_status']))
        if motion_failures and not result.motion_status:
            result.status += '; earlier_motion_failures=' + '|'.join(
                failure[:240] for failure in motion_failures[:6])
    result.validation_calls_json = json.dumps(calls)
    if timing_sink:
        timing_sink(result.validation_seconds, result.success, result.status)
    return result

