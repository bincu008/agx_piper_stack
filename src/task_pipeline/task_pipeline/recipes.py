"""Declarative calculation microsteps; the master resolves every result binding."""

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
             ('observation_header', 'observe', 'cloud.header')]),
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
