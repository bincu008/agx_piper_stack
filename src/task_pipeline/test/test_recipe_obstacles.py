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
