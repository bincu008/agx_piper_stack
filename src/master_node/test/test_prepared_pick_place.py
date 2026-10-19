"""Complete prepared sessions with fake actuators only; no ROS hardware calls."""

import copy
import time

import pytest
from my_actions.msg import PlacementCandidate
from my_actions.srv import (PreparePickPlace, CommitFloorSlots, GetExecutionState,
                            ExecutePlan, PreparedGripperCommand, HeldMotionStage)
from test_prepared_grasp_lift import fixture


def full_fixture():
    trial, events, calls = fixture()
    trial.execution.prepared_grasp_lift = trial
    source = trial.execution._cache['bowl'][1]
    pose = copy.deepcopy(source.candidates[0].grasp_pose)
    pose.pose.position.z = .12
    req = PreparePickPlace.Request(object_name='bowl', destination_name='tray',
        support_mode='floor_slots', lift_distance_m=.05,
        observation_header=source.observation_header,
        candidates=[PlacementCandidate(id=1, object_pose=pose, orientation_policy='upright')])
    req.destination_mask.header.stamp = source.observation_header.stamp
    original = trial.node.step_runtime._call

    def backend(name, query):
        if name == 'commit_floor_slots':
            calls.append(name)
            assert query.validate_only
            return CommitFloorSlots.Response(success=True, status='FLOOR_SLOTS_AVAILABLE')
        result = original(name, query)
        if name == 'plan_placement' and not query.lift_only:
            result.selected_candidate = query.candidates[0]
            result.preplace_ee_pose = copy.deepcopy(pose)
            result.preplace_ee_pose.pose.position.z = .22
            result.placement_ee_pose = copy.deepcopy(pose)
            result.transfer_path_flat = [.003]*6+[.004]*6
            result.descent_path_flat = [.004]*6+[.005]*6
        return result

    trial.node.step_runtime._call = backend
    base_move = trial.node.move_arm_to

    def move(q, **kwargs):
        result = base_move(q, **kwargs)
        if q[0] >= .0035:
            trial.node.current_tcp_pose.pose.position.z = .22 if q[0] < .0045 else .12
        return result

    trial.node.move_arm_to = move
    trial.node._wait_for_gripper_release = lambda width: True
    return trial, req, events, calls


def run_to_release(trial):
    for stage in trial.FULL_STAGES:
        query = HeldMotionStage.Request(object_name='bowl', plan_revision=trial.revision)
        result = trial.run_stages(query, HeldMotionStage.Response(), (stage,))
        assert result.success, result.status
    return result


def test_all_paths_are_prepared_before_motion_and_shared_session_releases():
    trial, req, events, calls = full_fixture()
    prepared = trial.prepare(req, PreparePickPlace.Response())
    assert prepared.success and not events
    assert prepared.selected_pose == req.candidates[0].object_pose
    state = trial.execution_state(GetExecutionState.Request(object_name='bowl'), GetExecutionState.Response())
    assert state.waiting_stages == list(trial.FULL_STAGES)
    trial.node.step_runtime._call = lambda *a: pytest.fail('Planner invoked during actuation')
    result = run_to_release(trial)
    assert result.gripper_released and trial.node._held_object is None
    assert trial.record['phase'] == 'RELEASED' and trial.node.state == 'IDLE'
    assert trial.record['completed_stages'] == list(trial.FULL_STAGES)
    trial.execution._guard()  # Released session no longer owns the empty arm.


def test_partial_placement_path_cannot_be_cached():
    trial, req, events, calls = full_fixture()
    original = trial.node.step_runtime._call

    def broken(name, query):
        result = original(name, query)
        if name == 'plan_placement':
            result.transfer_path_flat = []
        return result

    trial.node.step_runtime._call = broken
    assert not trial.prepare(req, PreparePickPlace.Response()).success
    assert trial.record is None and not events


def test_destination_mismatch_and_transfer_before_lift_are_rejected():
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    assert not trial.get_pick_place(PreparePickPlace.Request(
        object_name='bowl', destination_name='other'), PreparePickPlace.Response()).success
    result = trial.execution.held_motion.stage('execute_placement_transfer',
        HeldMotionStage.Request(object_name='bowl', plan_revision=trial.revision), HeldMotionStage.Response())
    assert not result.success and not events and trial.record['phase'] == 'PREPARED'


def test_failed_release_retains_attachment_and_unknown_state():
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    trial.node._wait_for_gripper_release = lambda width: False
    for stage in trial.FULL_STAGES[:-1]:
        assert trial.run_stages(HeldMotionStage.Request(object_name='bowl', plan_revision=trial.revision),
            HeldMotionStage.Response(), (stage,)).success
    result = trial.run_stages(HeldMotionStage.Request(object_name='bowl', plan_revision=trial.revision),
        HeldMotionStage.Response(), ('release',))
    assert not result.success and result.status.startswith('STATE_UNKNOWN')
    assert trial.node._held_object == 'bowl' and trial.record['phase'] == 'STATE_UNKNOWN'


def test_physical_gate_blocks_entire_prepared_session():
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    trial.node.allow_physical_grasp_execution = False
    result = trial.execute_segment(ExecutePlan.Request(object_name='bowl', plan_revision=trial.revision,
        segment='pregrasp'), ExecutePlan.Response())
    assert not result.success and not events


def test_placement_failure_tries_next_grasp_and_stops_at_complete_plan():
    """Keep the same scene and never execute a rejected or later candidate."""
    trial, req, events, _ = full_fixture()
    source = trial.execution._cache['bowl'][1]
    for identifier in (8, 9):
        alternate = copy.deepcopy(source.candidates[0])
        alternate.id = identifier
        source.candidates.append(alternate)
    original = trial.node.step_runtime._call
    tried = []

    def backend(name, query):
        if name == 'plan_placement':
            tried.append(query.grasp_candidate.id)
            if query.grasp_candidate.id == 7:
                from my_actions.srv import PlanPlacement
                return PlanPlacement.Response(status='PLACEMENT_NO_REACHABLE_CANDIDATE')
        return original(name, query)

    trial.node.step_runtime._call = backend
    prepared = trial.prepare(req, PreparePickPlace.Response())
    assert prepared.success and not events
    assert tried == [7, 8]
    assert trial.record['candidate'].id == 8
    assert trial.record['phase'] == 'PREPARED'
    assert 'GRASP_CANDIDATES_TRIED=2' in prepared.status
    assert trial.record['source_candidate_id'] == 7


def test_all_pick_place_candidates_fail_without_retaining_old_readiness():
    """A pool is bounded recovery, not permission to cache a partial plan."""
    from my_actions.srv import PlanPlacement
    trial, req, events, _ = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    source = trial.execution._cache['bowl'][1]
    alternate = copy.deepcopy(source.candidates[0])
    alternate.id = 8
    source.candidates.append(alternate)
    original = trial.node.step_runtime._call
    tried = []

    def backend(name, query):
        if name == 'plan_placement':
            tried.append(query.grasp_candidate.id)
            return PlanPlacement.Response(status='PLACEMENT_NO_REACHABLE_CANDIDATE')
        return original(name, query)

    trial.node.step_runtime._call = backend
    prepared = trial.prepare(req, PreparePickPlace.Response())
    assert not prepared.success and not events and trial.record is None
    assert tried == [7, 8]
    assert 'candidate=7' in prepared.status and 'candidate=8' in prepared.status


def test_lift_api_runs_before_destination_motion_without_actuation():
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    assert calls.index('plan_lift') < calls.index('plan_placement')
    assert not events
