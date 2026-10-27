"""Complete prepared sessions with fake actuators only; no ROS hardware calls."""

import copy
import time
from types import SimpleNamespace

import pytest
from my_actions.msg import PlacementCandidate
from my_actions.srv import (PlanPlacement, PlanCameraRetreat, PreparePickPlace, CommitFloorSlots, GetExecutionState,
                            ExecutePlan, PreparedGripperCommand, HeldMotionStage)
from test_prepared_grasp_lift import fixture


def full_fixture():
    trial, events, calls = fixture()
    trial.execution.prepared_grasp_lift = trial
    trial.execution.next_best_view = SimpleNamespace(
        prepare_retreat=lambda *args: (PlanPlacement.Response(
            success=True, motion_success=True, motion_status='EMPTY_HAND_WITHDRAWAL_VALIDATED',
            lift_path_flat=[.005]*6+[.006]*6,
            lift_ee_pose=copy.deepcopy(args[1].placement_ee_pose)),
            PlanCameraRetreat.Response(success=True, status='RETREAT_FROZEN_SCENE_VALIDATED',
                path_flat=[.006]*6+[.007]*6,
                tcp_pose=copy.deepcopy(args[1].placement_ee_pose))),
        publish_preview=lambda *args: None)
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
    assert state.waiting_stages == list(trial.FULL_STAGES)+['withdrawal', 'retreat']
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


def test_retreat_is_cached_before_execution_and_runs_without_backend_calls():
    """Release uses an immutable retreat, not a close-up camera observation."""
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    assert 'retreat' in trial.record and not events
    run_to_release(trial)
    assert trial.execute_withdrawal(HeldMotionStage.Request(
        object_name='tray', plan_revision=trial.revision), HeldMotionStage.Response()).success
    trial.node.step_runtime._call = lambda *args: pytest.fail('No backend during retreat')
    result = trial.execute_retreat(HeldMotionStage.Request(
        object_name='tray', plan_revision=trial.revision), HeldMotionStage.Response())
    assert result.success, result.status
    assert trial.record is None
    assert trial.completed_record['completed_stages'][-1] == 'retreat'
    assert not trial.execute_retreat(HeldMotionStage.Request(
        object_name='tray', plan_revision=trial.revision), HeldMotionStage.Response()).success


def test_failed_retreat_invalidates_whole_preparation_before_actuation():
    """Do not cache an executable pickup when its retreat is missing."""
    trial, req, events, calls = full_fixture()
    def reject(*args):
        raise RuntimeError('PREFLIGHT_RETREAT_REJECTED: no path')
    trial.execution.next_best_view.prepare_retreat = reject
    result = trial.prepare(req, PreparePickPlace.Response())
    assert not result.success and trial.record is None and not events
    assert 'PREFLIGHT_RETREAT_REJECTED' in result.status


@pytest.mark.parametrize('failure', ['revision', 'before_release', 'timeout', 'held'])
def test_retreat_rejects_invalid_session_before_commands(failure):
    """No capture or motion can hide a stale or out-of-order release state."""
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    if failure != 'before_release':
        run_to_release(trial)
        assert trial.execute_withdrawal(HeldMotionStage.Request(
            object_name='tray', plan_revision=trial.revision), HeldMotionStage.Response()).success
    revision = trial.revision
    if failure == 'revision':
        revision += 1
    elif failure == 'timeout':
        trial.record['execution_started'] -= 301
    elif failure == 'held':
        trial.node._held_object = 'bowl'
    events.clear()
    result = trial.execute_retreat(HeldMotionStage.Request(
        object_name='tray', plan_revision=revision), HeldMotionStage.Response())
    assert not result.success and not events


def test_retreat_actuator_failure_is_unknown_and_not_retried():
    """A frozen path does not authorize retrying uncertain physical motion."""
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    run_to_release(trial)
    assert trial.execute_withdrawal(HeldMotionStage.Request(
        object_name='tray', plan_revision=trial.revision), HeldMotionStage.Response()).success
    def fail(*args):
        events.append('failed_motion')
        raise RuntimeError('TEST_ACTUATOR_FAILURE')
    trial.follow = fail
    query = HeldMotionStage.Request(object_name='tray', plan_revision=trial.revision)
    result = trial.execute_retreat(query,HeldMotionStage.Response())
    assert not result.success and 'STATE_UNKNOWN' in result.status
    assert not trial.execute_retreat(query,HeldMotionStage.Response()).success
    assert events.count('failed_motion') == 1


def test_retreat_requires_completed_empty_hand_withdrawal():
    """Release alone cannot start the distant trajectory through the bowl."""
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    run_to_release(trial)
    events.clear()
    result = trial.execute_retreat(HeldMotionStage.Request(
        object_name='tray', plan_revision=trial.revision), HeldMotionStage.Response())
    assert not result.success and not events
    assert not trial.record.get('withdrawal_completed')


def test_empty_hand_lift_uses_same_public_planner_without_payload():
    """The lift API retains its existing held mode and exposes empty mode explicitly."""
    trial, req, events, calls = full_fixture()
    seen=[]
    def plan(name, request):
        seen.append((name,request))
        return PlanPlacement.Response(success=True,motion_success=True)
    trial.node.step_runtime._call=plan
    result=trial.plan_lift(PlanPlacement.Request(empty_hand=True, start_joints=[.1]*6,
                                              jaw_opening_m=.06),PlanPlacement.Response())
    assert result.success and seen[0][0]=='plan_placement'
    query=seen[0][1]
    assert query.empty_hand and query.lift_only and query.plan_pick_place_motion
    assert not query.source_object_cloud.width and not events


def test_cached_pick_place_reports_withdrawal_and_retreat_readiness():
    """The second prompt reports the same validated segments as preparation."""
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    result = trial.get_pick_place(req,PreparePickPlace.Response())
    assert result.success and 'EMPTY_HAND_WITHDRAWAL_VALIDATED' in result.status
    assert 'RETREAT_FROZEN_SCENE_VALIDATED' in result.status
    assert not events


def test_compulsory_look_is_frozen_and_pregrasp_starts_at_its_endpoint():
    """Fake execution follows look before pickup without invoking a planner."""
    from my_actions.srv import PlanNextBestView, CheckAndReplan
    from rclpy.serialization import serialize_message
    trial, req, events, calls = full_fixture()
    view = PlanNextBestView.Response(
        success=True, plan_revision=3, expected_start_joints=[0.]*6,
        path_flat=[0.]*6+[.0005]*6, tcp_pose=copy.deepcopy(trial.node.current_tcp_pose))
    trial.execution.next_best_view.record = {
        'name': 'bowl', 'created': time.monotonic(), 'result': view}
    req.look_plan_cdr = serialize_message(view)
    original = trial.node.step_runtime._call

    def backend(name, query):
        if name == 'plan_motion':
            assert list(query.current_joints) == pytest.approx([.0005]*6)
            return CheckAndReplan.Response(
                success=True, path_flat=list(query.current_joints)+[.001]*6)
        return original(name, query)

    trial.node.step_runtime._call = backend
    prepared = trial.prepare(req, PreparePickPlace.Response())
    assert prepared.success, prepared.status
    assert not events
    state = trial.execution_state(GetExecutionState.Request(object_name='bowl'),
                                  GetExecutionState.Response())
    assert state.waiting_stages[0] == 'look'
    query = ExecutePlan.Request(object_name='bowl', plan_revision=prepared.plan_revision,
                                segment='pregrasp')
    result = trial.execute_segment(query, ExecutePlan.Response())
    assert not result.success and 'COMPLETED_LOOK' in result.status
    assert not events
    query.segment = 'look'
    trial.node.step_runtime._call = lambda *args: pytest.fail('No planner during look execution')
    result = trial.execute_segment(query, ExecutePlan.Response())
    assert result.success, result.status
    assert trial.record['completed_stages'] == ['look']
    assert not trial.execute_segment(query, ExecutePlan.Response()).success
    query.segment = 'pregrasp'
    assert trial.execute_segment(query, ExecutePlan.Response()).success
    result=trial.run_stages(HeldMotionStage.Request(object_name='bowl',
        plan_revision=prepared.plan_revision),HeldMotionStage.Response(),trial.FULL_STAGES[1:])
    assert result.success and result.gripper_released
    for method in (trial.execute_withdrawal,trial.execute_retreat):
        result=method(HeldMotionStage.Request(object_name='tray',
            plan_revision=prepared.plan_revision),HeldMotionStage.Response())
        assert result.success,result.status
    assert trial.record is None
    assert trial.completed_record['completed_stages']==['look']+list(trial.FULL_STAGES)+['withdrawal','retreat']
    assert trial.node._held_object is None


def test_unowned_look_response_cannot_be_cached():
    """Caller-supplied motion is never substituted for a master-owned view plan."""
    from my_actions.srv import PlanNextBestView
    from rclpy.serialization import serialize_message
    trial, req, events, calls = full_fixture()
    req.look_plan_cdr = serialize_message(PlanNextBestView.Response(success=True))
    result = trial.prepare(req, PreparePickPlace.Response())
    assert not result.success and 'LOOK_PLAN_SOURCE' in result.status
    assert trial.record is None and not events


def test_satisfied_look_sends_no_actuator_commands():
    """A visible target still has an ordered stage without streaming raw feedback."""
    from my_actions.srv import PlanNextBestView
    from rclpy.serialization import serialize_message
    trial, req, events, calls = full_fixture()
    view = PlanNextBestView.Response(
        success=True, status='CURRENT_VIEW_ALREADY_SUITABLE', plan_revision=4,
        expected_start_joints=[0.]*6, path_flat=[0.]*6,
        tcp_pose=copy.deepcopy(trial.node.current_tcp_pose))
    trial.execution.next_best_view.record = {
        'name': 'bowl', 'created': time.monotonic(), 'result': view}
    req.look_plan_cdr = serialize_message(view)
    prepared = trial.prepare(req, PreparePickPlace.Response())
    assert prepared.success, prepared.status
    events.clear()
    trial.follow = lambda *args: pytest.fail('No trajectory for a satisfied look')
    result = trial.execute_segment(ExecutePlan.Request(
        object_name='bowl', plan_revision=prepared.plan_revision, segment='look'),
        ExecutePlan.Response())
    assert result.success and result.phase == 'PREPARED'
    assert not events and trial.record['look_completed']


def test_established_thin_rim_contact_survives_lift_transfer_and_descent():
    """Exercise the real held gate across all stages after a lower settled load."""
    from master_node.contact_evidence import SettledContact
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success

    def close_contact():
        gate = SettledContact()
        now = time.monotonic()
        sample = trial.node.current_gripper_status
        sample.width, sample.force = .00056, -.25
        for i in range(11):
            stamp = now-.6+i*.02
            gate.update(sample, stamp, stamp)
        assert gate.established
        sample.force = -.15
        for i in range(1, 21):
            stamp = now-.4+i*.02
            gate.update(sample, stamp, stamp)
        assert gate.holding() and not gate.confirmed()
        trial.node.gripper_contact_evidence = gate
        return True

    trial.node._wait_for_gripper_contact = close_contact
    original_move = trial.node.move_arm_to

    def move(joints, **kwargs):
        result = original_move(joints, **kwargs)
        gate = getattr(trial.node, 'gripper_contact_evidence', None)
        if gate is not None:
            stamp = time.monotonic()
            gate.update(trial.node.current_gripper_status, stamp, stamp)
        return result

    trial.node.move_arm_to = move
    result = run_to_release(trial)
    assert result.success and result.gripper_released, result.status
    assert trial.record['completed_stages'] == list(trial.FULL_STAGES)


def test_complete_bowl_clears_motion_caches_and_allows_next_cup():
    """Completed ownership must not block a new object or discard occupied slots."""
    from my_actions.srv import CacheGraspPlan
    trial, req, events, calls = full_fixture()
    invalidated = []
    trial.execution.grasp_stages = SimpleNamespace(invalidate=invalidated.append)
    trial.node.refined_grasp_plans = {'bowl': {'temporary': True}}
    source = copy.deepcopy(trial.execution._cache['bowl'][1])
    assert trial.prepare(req, PreparePickPlace.Response()).success
    assert run_to_release(trial).success
    destination = {'name': 'tray', 'occupied_slots': ['0,0'], 'placed_cloud': object()}
    trial.node.manipulation_memory = destination
    trial.execution.held_motion.record = {'key': 'bowl', 'phase': 'RELEASED'}
    query = HeldMotionStage.Request(object_name='tray', plan_revision=trial.revision)
    assert trial.execute_withdrawal(query, HeldMotionStage.Response()).success
    assert trial.execute_retreat(query, HeldMotionStage.Response()).success
    assert trial.record is None and trial.execution.held_motion.record is None
    assert not trial.execution._cache and not trial.node.cached_grasp_plans
    assert not trial.node.refined_grasp_plans and invalidated == ['bowl']
    assert trial.node.manipulation_memory is destination
    assert destination['occupied_slots'] == ['0,0']
    state = trial.execution_state(GetExecutionState.Request(object_name='bowl'),
                                  GetExecutionState.Response())
    assert state.success and state.phase == 'COMPLETE' and not state.waiting_stages
    assert state.completed_stages[-1] == 'retreat' and state.expires_in_seconds == 0.
    trial.execution._guard()
    source.object_name = 'cup'
    events.clear()
    result = trial.execution._save(source, CacheGraspPlan.Response())
    assert result.success, result.status
    assert 'cup' in trial.execution._cache and not events
    assert not trial.get_pick_place(req, PreparePickPlace.Response()).success


def test_release_alone_does_not_clear_unfinished_withdrawal_session():
    """An empty hand still owes its cached withdrawal/retreat before retirement."""
    from my_actions.srv import CacheGraspPlan
    trial, req, events, calls = full_fixture()
    source = copy.deepcopy(trial.execution._cache['bowl'][1])
    assert trial.prepare(req, PreparePickPlace.Response()).success
    assert run_to_release(trial).success
    source.object_name = 'cup'
    result = trial.execution._save(source, CacheGraspPlan.Response())
    assert not result.success and result.status == 'PREPARED_EXECUTION_SESSION_OWNS_ARM'
    assert trial.record is not None and 'bowl' in trial.execution._cache


def test_retreat_speed_restore_failure_keeps_unknown_session_owned():
    """A failure after movement cannot be hidden by automatic cleanup."""
    trial, req, events, calls = full_fixture()
    assert trial.prepare(req, PreparePickPlace.Response()).success
    assert run_to_release(trial).success
    query = HeldMotionStage.Request(object_name='tray', plan_revision=trial.revision)
    assert trial.execute_withdrawal(query, HeldMotionStage.Response()).success
    commands = []

    def speed(value):
        commands.append(value)
        return len(commands) == 1

    trial.node.set_arm_speed = speed
    result = trial.execute_retreat(query, HeldMotionStage.Response())
    assert not result.success and result.status == 'STATE_UNKNOWN: ARM_SPEED_RESTORE_FAILED'
    assert trial.record['phase'] == 'STATE_UNKNOWN' and 'bowl' in trial.execution._cache
    assert trial.node.state == 'ERROR'
