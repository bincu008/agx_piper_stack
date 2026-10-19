"""Declarative calculation microsteps; the master resolves every result binding."""

import math

from my_actions import srv
from my_actions.msg import GraspCandidate, PipelineStep, StepBinding

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
            object_name=object_name, candidates=[GraspCandidate()], ttl_seconds=60),
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
    steps.append(step('prepare_motion', 'prepare_pick_place', srv.PreparePickPlace.Request(
        object_name=object_name, destination_name=destination_name,
        support_mode=support_mode, lift_distance_m=float(lift_distance_m)),
        [('observation_header', 'observe', 'cloud.header'),
         ('source_object_center', 'placements', 'source_object_center'),
         ('candidates', 'placements', 'candidates'),
         ('destination_mask', 'segment_tray', 'mask')]))
    steps.append(step('execution_state', 'get_execution_state', srv.GetExecutionState.Request(object_name=object_name)))
    return steps


def execute_cached_pick_place(object_name, destination_name='tray', support_mode='floor_slots'):
    """Execute only a reviewed complete plan through the shared stage APIs."""
    steps = [step('prepare_motion', 'get_prepared_pick_place', srv.PreparePickPlace.Request(
        object_name=object_name, destination_name=destination_name, support_mode=support_mode))]
    steps.append(step('check_motion', 'validate_prepared_pick_place',
                      srv.PreparePickPlace.Request(object_name=object_name)))
    revision = [('plan_revision', 'prepare_motion', 'plan_revision')]
    for stage in ('pregrasp', 'gripper_open', 'approach', 'gripper_close', 'lift'):
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
    return steps


def pick_and_place(object_name, destination_name='tray', lift_distance_m=.05,
                   support_mode='floor_slots'):
    """Queue calculation and execution against one immutable prepared revision."""
    return (prepare_pick_place(object_name, destination_name, lift_distance_m, support_mode)
            + execute_cached_pick_place(object_name, destination_name, support_mode)[1:])


def with_active_memory(steps, object_name, destination_name='tray'):
    """Reuse master target memory while retaining the existing typed recipe."""
    import copy
    result = []
    for item in steps:
        if item.step_id in ('presence_cup', 'ground_cup', 'presence_tray', 'ground_tray'):
            continue
        item = copy.deepcopy(item)
        if item.step_id in ('segment_cup', 'segment_tray'):
            destination = item.step_id == 'segment_tray'
            item = step(item.step_id, 'resolve_target', srv.ResolveManipulationTarget.Request(
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
