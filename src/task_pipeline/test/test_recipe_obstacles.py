"""Frozen scene overlays must reach search and validation in the same recipe."""

from my_actions.srv import FindPlacementCandidates, PlanPlacement
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

from task_pipeline.recipes import cup_to_tray


def test_same_frozen_obstacles_reach_both_placement_apis():
    cloud = PointCloud2(width=1, height=1, point_step=12, row_step=12)
    cloud.header.frame_id = 'base_link'
    cloud.data = bytes(12)
    steps = {step.step_id: step for step in cup_to_tray(additional_obstacles=cloud)}
    search = deserialize_message(bytes(steps['placements'].request_cdr),
                                 FindPlacementCandidates.Request)
    validation = deserialize_message(bytes(steps['placement_ik'].request_cdr),
                                     PlanPlacement.Request)
    assert search.additional_obstacles == cloud
    assert validation.additional_obstacles == cloud
    assert validation.validation_batch_size == 8
    assert not validation.plan_pick_place_motion


def test_default_recipe_has_no_simulation_overlay():
    steps = {step.step_id: step for step in cup_to_tray()}
    request = deserialize_message(bytes(steps['placements'].request_cdr),
                                  FindPlacementCandidates.Request)
    assert request.additional_obstacles.width == 0
    assert not request.additional_obstacles.data


def test_universal_stack_uses_scene_without_named_support():
    steps = {step.step_id: step for step in cup_to_tray(support_mode='stack')}
    request = deserialize_message(bytes(steps['placements'].request_cdr),
                                  FindPlacementCandidates.Request)
    assert request.support_mode == 'stack'
    assert not request.support_name
    assert not request.stack_support_cloud.width


def test_presence_checks_are_separate_and_precede_each_grounding():
    steps = cup_to_tray()
    names = [s.step_id for s in steps]
    assert len(steps) == 17
    assert names.index('presence_cup') + 1 == names.index('ground_cup')
    assert names.index('presence_tray') + 1 == names.index('ground_tray')
    from my_actions.srv import VerifyPresence
    for step_id, target in [('presence_cup', 'brown cup'), ('presence_tray', 'tray')]:
        entry = next(s for s in steps if s.step_id == step_id)
        request = deserialize_message(bytes(entry.request_cdr), VerifyPresence.Request)
        assert request.target == target
        assert entry.failure_policy == 'stop'
        assert entry.bindings[0].source_step_id == 'observe'


def test_default_recipe_uses_exclusive_grid_and_commits_after_validation():
    steps = cup_to_tray()
    search = deserialize_message(bytes(next(s for s in steps if s.step_id == 'placements').request_cdr),
                                 FindPlacementCandidates.Request)
    assert search.support_mode == 'floor_slots'
    assert steps[-1].function_name == 'commit_floor_slots'
    assert steps[-1].bindings[0].source_step_id == 'placement_ik'
    assert steps[-1].failure_policy == 'stop'



def test_all_calculation_precedes_public_frozen_execution_steps():
    """Keep every planning failure before actuator execution."""
    from task_pipeline.recipes import grasp_and_lift, prepare_grasp_lift
    calculation = prepare_grasp_lift('small white bowl at bottom left')
    names = [s.function_name for s in calculation]
    assert names[-1] == 'prepare_grasp_lift'
    assert 'move_pregrasp' not in names and 'execute_prepared_grasp_lift' not in names
    full = grasp_and_lift('small white bowl at bottom left')
    assert [s.function_name for s in full[-5:]] == [
        'execute_plan', 'gripper_open', 'execute_plan', 'gripper_close', 'execute_plan']
    assert [s.step_id for s in full[-5:]] == [
        'pregrasp', 'gripper_open', 'approach', 'gripper_close', 'lift']
    assert full[-1].bindings[0].source_step_id == 'prepare_motion'


def test_reviewed_execution_contains_no_calculation_or_refinement():
    """Reject absent prepared plans rather than silently recalculate after movement."""
    from task_pipeline.recipes import grasp_and_lift
    steps = grasp_and_lift('small white bowl at bottom left', use_cached=True)
    assert [s.function_name for s in steps] == [
        'get_prepared_grasp_lift', 'validate_prepared_grasp_lift', 'get_execution_state',
        'execute_plan', 'gripper_open', 'execute_plan', 'gripper_close', 'execute_plan']
    assert all(s.failure_policy == 'stop' for s in steps)


def test_pick_place_recipe_plans_first_and_executes_cached_public_stages_only():
    from task_pipeline.recipes import prepare_pick_place, execute_cached_pick_place
    planned = prepare_pick_place('beige cup, grasp its body')
    functions = [s.function_name for s in planned]
    assert functions[-2:] == ['prepare_pick_place', 'get_execution_state']
    assert 'commit_floor_slots' not in functions
    assert not any(f.startswith('execute') for f in functions)
    cache = next(s for s in planned if s.step_id == 'cache_grasp')
    assert cache.bindings[0].source_field == 'candidates'
    execute = execute_cached_pick_place('beige cup, grasp its body')
    assert [s.step_id for s in execute] == ['prepare_motion', 'check_motion', 'pregrasp',
        'gripper_open', 'approach', 'gripper_close', 'lift', 'transfer', 'descent',
        'release', 'commit_slots']
    assert all(s.function_name not in ('ground_object', 'calculate_grasps', 'plan_placement')
               for s in execute)


def test_queued_pick_place_binds_execution_to_calculation_revision():
    from task_pipeline.recipes import pick_and_place
    steps = pick_and_place('striped white plate')
    ids = [s.step_id for s in steps]
    assert len(ids) == len(set(ids))
    boundary = ids.index('pregrasp')
    assert ids[boundary:] == ['pregrasp', 'gripper_open', 'approach', 'gripper_close',
                              'lift', 'transfer', 'descent', 'release', 'commit_slots']
    assert steps[boundary - 1].function_name == 'validate_prepared_pick_place'
    assert all(s.failure_policy == 'stop' for s in steps)
    for s in steps[boundary:-1]:
        assert any(b.target_field == 'plan_revision' and b.source_step_id == 'prepare_motion'
                   for b in s.bindings)
