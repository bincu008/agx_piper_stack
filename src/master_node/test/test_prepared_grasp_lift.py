"""Validate calculation-before-motion and execution without planning calls."""

import threading
import time
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped

from master_node.execution_api import ExecutionAPI
from master_node.held_motion import HeldMotion
from master_node.prepared_grasp_lift import PreparedGraspLiftExecutor

from my_actions.msg import GraspCandidate
from my_actions.srv import (
    CacheGraspPlan, CheckAndReplan, HeldMotionStage, PlanCartesianGraspApproach,
    PlanPlacement, PreparedGraspLift, PrepareHeldScene,
)

import pytest


def fixture(fail_at=None):
    """Provide deterministic planners and fake arm/gripper feedback."""
    events, calls = [], []

    def pose(z):
        value = PoseStamped()
        value.header.frame_id = 'base_link'
        value.header.stamp.sec = 10
        value.pose.position.z = z
        value.pose.orientation.w = 1.
        return value

    candidate = GraspCandidate(
        id=7, grasp_pose=pose(.18), pregrasp_pose=pose(.2), jaw_opening_m=.04,
        grasp_joints=[.002]*6, pregrasp_joints=[.001]*6, score=1.,
        ik_valid=True, collision_checked=True, grasp_collision_valid=True,
        pregrasp_collision_valid=True, measured_contacts_valid=True)
    candidate.contact_left.x, candidate.contact_right.x = .02, -.02
    source = CacheGraspPlan.Request(
        object_name='bowl', observation_header=pose(.18).header,
        candidates=[candidate], ttl_seconds=60)
    now = time.monotonic()
    node = SimpleNamespace(
        state='IDLE', _held_object=None, allow_physical_grasp_execution=True,
        physical_collision_detected=False, collision_detected=False,
        allow_model_based_body_grasp_execution=False, gripper_open_force_n=1.,
        joint_limits=[(-2., 2.)]*6,
        joint_feedback_time=now, pose_feedback_time=now, gripper_feedback_time=now,
        current_joint_states=SimpleNamespace(position=[0.]*6), current_tcp_pose=pose(.25),
        current_gripper_status=SimpleNamespace(width=.02, force=0.))

    def backend(name, query):
        calls.append(name)
        success = name != fail_at and not (name == 'plan_lift' and fail_at == 'plan_placement')
        if name == 'prepare_held_scene':
            return PrepareHeldScene.Response(success=success, status='GEOMETRY_FAILED')
        if name == 'plan_final_approach':
            assert list(query.final_grasp_joints) == list(candidate.grasp_joints)
            return PlanCartesianGraspApproach.Response(
                success=success, failure_reason='IK_FAILED_AT_WAYPOINT_1',
                waypoint_count=1, path_flat=[.002]*6,
                planned_pregrasp_joints=[.001]*6, planned_pregrasp_pose=pose(.2))
        if name in ('plan_lift', 'plan_placement'):
            return PlanPlacement.Response(
                success=success, motion_success=success, status='LIFT_FAILED',
                selected_grasp_candidate=query.grasp_candidate,
                lift_path_flat=[.002]*6+[.003]*6, lift_ee_pose=pose(.23))
        assert name == 'plan_motion'
        assert query.grasp_motion
        return CheckAndReplan.Response(success=success, path_flat=[0.]*6+[.001]*6)

    def move(joints, **_kwargs):
        events.append('move')
        node.current_joint_states.position = list(joints)
        positions = {0.: .25, .001: .2, .002: .18, .003: .23}
        node.current_tcp_pose = pose(positions[min(positions, key=lambda q: abs(q-joints[0]))])
        node.joint_feedback_time = node.pose_feedback_time = time.monotonic()
        return True

    def gripper(width, force):
        events.append('gripper')
        node.current_gripper_status.width = max(width, .04)
        node.current_gripper_status.force = force if width < .04 else 0.
        node.gripper_feedback_time = time.monotonic()
        return True

    node.step_runtime = SimpleNamespace(_call=backend)
    node.get_logger = lambda: SimpleNamespace(info=lambda message: None)
    node.move_arm_to, node._command_gripper = move, gripper
    node._publish_joint_state_command = lambda *, arm_positions: node.move_arm_to(arm_positions)
    node._wait_for_gripper_opening = lambda _: True
    node._wait_for_gripper_contact = lambda: True
    node.set_arm_speed = lambda speed: (events.append(('speed', speed)), True)[1]
    node._pose_error = lambda ref, measured: (
        abs(ref.pose.position.z-measured.pose.position.z), 0.)
    node.cached_grasp_plans = {
        'bowl': {'header': source.observation_header, 'candidate': candidate}}
    execution = object.__new__(ExecutionAPI)
    execution.node, execution._lock = node, threading.Lock()
    execution._cache_lock = threading.Lock()
    execution._cache = {'bowl': (now, source)}
    held = object.__new__(HeldMotion)
    held.node, held.execution = node, execution
    execution.held_motion = held
    trial = object.__new__(PreparedGraspLiftExecutor)
    trial.node, trial.execution, trial.record, trial.revision = node, execution, None, 0
    return trial, events, calls


def prepare(trial):
    """Prepare a five-centimeter lift without actuator commands."""
    return trial.prepare(PreparedGraspLift.Request(object_name='bowl', lift_distance_m=.05),
                         PreparedGraspLift.Response())


def execute(trial, revision=1):
    """Execute the exact prepared revision on fake hardware."""
    return trial.execute(HeldMotionStage.Request(object_name='bowl', plan_revision=revision),
                         HeldMotionStage.Response())


def test_cached_check_has_no_actuation_and_does_not_consume_plan():
    """Offline second prompt preserves the exact prepared revision."""
    trial, events, calls = fixture()
    assert prepare(trial).success
    count = len(calls)
    trial.node.allow_physical_grasp_execution = False
    result = trial.validate(PreparedGraspLift.Request(object_name='bowl'),
                            PreparedGraspLift.Response())
    assert result.success and result.plan_revision == 1
    assert 'CACHED_HANDOFF_VALIDATED_NO_ACTUATION' in result.status
    assert trial.record['phase'] == 'PREPARED' and not events and len(calls) == count


def test_cached_check_rejects_changed_feedback_without_actuation():
    """A replay start mismatch stops the handoff instead of replanning."""
    trial, events, _ = fixture()
    assert prepare(trial).success
    trial.node.current_joint_states.position[0] = .5
    result = trial.validate(PreparedGraspLift.Request(object_name='bowl'),
                            PreparedGraspLift.Response())
    assert not result.success and not events
    assert trial.record['phase'] == 'PREPARED'


def test_command_limit_roundoff_is_identical_for_cache_handoff():
    """Roundoff at a nominal boundary passes; a real excursion does not."""
    trial, events, _ = fixture()
    assert prepare(trial).success
    trial.record['approach'][-1][0] = 2.0 + 1.e-7
    trial.node.command_joint_limits = [(-2., 2.)]*6
    result = trial.validate(PreparedGraspLift.Request(object_name='bowl'),
                            PreparedGraspLift.Response())
    assert result.success and not events
    trial.record['approach'][-1][0] = 2.001
    result = trial.validate(PreparedGraspLift.Request(object_name='bowl'),
                            PreparedGraspLift.Response())
    assert not result.success and not events
    assert 'COMMAND_LIMIT_VIOLATION' in result.status


@pytest.mark.parametrize(
    'failure', ['prepare_held_scene', 'plan_final_approach', 'plan_placement', 'plan_motion'])
def test_every_planning_failure_happens_before_actuation(failure):
    """Reject the entire motion without opening or moving when any planner fails."""
    trial, events, _ = fixture(failure)
    assert not prepare(trial).success and not events
    assert trial.record is None
    assert not execute(trial).success and not events


def test_prepared_execution_calls_no_planners_and_stops_holding():
    """Run only cached joint paths, leaving the bowl held after lifting."""
    trial, events, calls = fixture()
    prepared = prepare(trial)
    assert prepared.success and not events
    count = len(calls)
    trial.node.step_runtime._call = lambda *_: pytest.fail('Planner called after motion began')
    result = execute(trial, prepared.plan_revision)
    assert result.success and result.motion_completed and len(calls) == count
    assert trial.node._held_object == 'bowl' and trial.record['phase'] == 'LIFTED'
    assert events[-1] == ('speed', 20)
    assert not execute(trial).success


def test_start_state_changed_rejects_without_motion():
    """Do not silently replan when the arm moved after calculation."""
    trial, events, _ = fixture()
    assert prepare(trial).success
    trial.node.current_joint_states.position = [.1]*6
    assert not execute(trial).success and not events


def test_transfer_uses_resolved_pregrasp_returned_by_approach_planner():
    """Transfer to the returned start pose before its rotation/approach path."""
    trial, events, _ = fixture()
    backend = trial.node.step_runtime._call

    def aligned_backend(name, query):
        response = backend(name, query)
        if name == 'plan_final_approach':
            response.planned_pregrasp_joints = [.0015]*6
            response.planned_pregrasp_pose.pose.position.z = .245
        if name == 'plan_motion':
            assert list(query.remaining_path_flat) == pytest.approx([.0015]*6)
            response.path_flat = [0.]*6+[.0015]*6
        return response

    trial.node.step_runtime._call = aligned_backend
    assert prepare(trial).success and not events
    assert trial.record['candidate'].pregrasp_pose.pose.position.z == .245


def test_rotation_prefix_is_retained_in_complete_prepared_path():
    """Do not drop rotation waypoints when caching the final translation."""
    trial, events, _ = fixture()
    backend = trial.node.step_runtime._call

    def rotating_backend(name, query):
        response = backend(name, query)
        if name == 'plan_final_approach':
            response.path_flat = [.0015]*6+[.002]*6
            response.waypoint_count = 2
            response.alignment_waypoint_count = 1
        return response

    trial.node.step_runtime._call = rotating_backend
    assert prepare(trial).success and not events
    assert trial.record['alignment_waypoint_count'] == 1
    assert trial.record['approach'][0] == pytest.approx([.0015]*6)
    assert trial.record['approach'][-1] == pytest.approx([.002]*6)


def test_approach_bypass_marker_is_visible_in_prepared_status():
    """Do not hide disabled finger/scene validation behind a success result."""
    trial, events, _ = fixture()
    backend = trial.node.step_runtime._call

    def bypass_backend(name, query):
        response = backend(name, query)
        if name == 'plan_final_approach':
            response.failure_reason = 'OK;FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED'
        return response

    trial.node.step_runtime._call = bypass_backend
    result = prepare(trial)
    assert result.success and not events
    assert 'FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED' in result.status
    ready = trial.get(PreparedGraspLift.Request(object_name='bowl'), PreparedGraspLift.Response())
    assert ready.success and 'FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED' in ready.status


def test_runtime_hardware_failure_stops_and_prevents_retry():
    """Keep live fault stops despite completing all planning in advance."""
    trial, events, _ = fixture()
    assert prepare(trial).success
    trial.node.move_arm_to = lambda *_args, **_kwargs: False
    result = execute(trial)
    assert not result.success and result.status.startswith('STATE_UNKNOWN')
    assert events[-1] == ('speed', 20) and trial.node.state == 'ERROR'
    count = len(events)
    assert not execute(trial).success and len(events) == count


def test_replacing_source_cache_invalidates_prepared_readiness():
    """Reject replacement even when its observation header and ID are unchanged."""
    trial, events, _ = fixture()
    assert prepare(trial).success
    created, source = trial.execution._cache['bowl']
    trial.execution._cache['bowl'] = (created+.000001, source)
    ready = trial.get(PreparedGraspLift.Request(object_name='bowl'), PreparedGraspLift.Response())
    assert not ready.success and not execute(trial).success and not events


def test_failed_repreparation_clears_old_plan():
    """Do not execute an older success after a replacement planning failure."""
    trial, events, _ = fixture()
    assert prepare(trial).success
    trial.node.step_runtime._call = lambda *_: PrepareHeldScene.Response(success=False)
    assert not prepare(trial).success and trial.record is None and not events


def test_feedback_margin_cannot_enlarge_command_limits():
    """Reject otherwise valid planner output outside actuator bounds before motion."""
    trial, events, _ = fixture()
    trial.node.command_joint_limits = [(-.0005, .0005)]*6
    result = prepare(trial)
    assert not result.success and 'COMMAND_LIMIT_VIOLATION' in result.status
    assert not events and trial.record is None


def test_failed_lift_tries_next_candidate_before_execution():
    """Endpoint ranking cannot prevent a later complete candidate from winning."""
    import copy
    trial, events, _ = fixture()
    source = trial.execution._cache['bowl'][1]
    second = copy.deepcopy(source.candidates[0])
    second.id = 8
    second.validation_status = 'ANGLE_SELECTION=ORIGINAL (tuning rejected)'
    source.candidates.append(second)
    messages = []
    trial.node.get_logger = lambda: SimpleNamespace(info=messages.append)
    original = trial.node.step_runtime._call

    def backend(name, query):
        if name == 'plan_lift' and query.grasp_candidate.id == 7:
            return PlanPlacement.Response(success=False, status='LIFT_IK_FAILED')
        return original(name, query)

    trial.node.step_runtime._call = backend
    result = prepare(trial)
    assert result.success and not events
    assert 'GRASP_CANDIDATES_TRIED=2' in result.status
    assert trial.record['candidate'].id == 8
    selected = [m for m in messages if m.startswith('[PREPARED GRASP SELECTED]')]
    assert len(selected) == 1
    assert 'candidate=8' in selected[0] and 'ANGLE_SELECTION=ORIGINAL' in selected[0]
    assert any('candidate=7' in m for m in messages if 'REJECTED' in m)
    assert trial.get(PreparedGraspLift.Request(object_name='bowl'),
                     PreparedGraspLift.Response()).success
    assert execute(trial).success


def test_lift_fallback_cannot_shortchange_requested_clearance():
    """A successful backend response still must provide the requested lift."""
    trial, events, _ = fixture()
    original = trial.node.step_runtime._call

    def backend(name, query):
        result = original(name, query)
        if name in ('plan_lift', 'plan_placement'):
            result.lift_ee_pose.pose.position.z = .20
        return result

    trial.node.step_runtime._call = backend
    result = prepare(trial)
    assert not result.success and 'LIFT_CLEARANCE_SHORTFALL' in result.status
    assert not events and trial.record is None


def test_public_stages_share_session_and_never_replan():
    """Agent-sized calls follow the same validated frozen paths as the composite."""
    from my_actions.srv import ExecutePlan, GetExecutionState, PreparedGripperCommand
    trial, events, calls = fixture()
    assert prepare(trial).success
    planning_calls = len(calls)
    state = trial.execution_state(GetExecutionState.Request(object_name='bowl'),
                                  GetExecutionState.Response())
    assert state.success and state.waiting_stages == list(trial.STAGES)
    for stage in trial.STAGES:
        if stage.startswith('gripper'):
            query = PreparedGripperCommand.Request(object_name='bowl', plan_revision=1)
            result = (trial.open_gripper if stage == 'gripper_open' else trial.close_gripper)(
                query, PreparedGripperCommand.Response())
        else:
            result = trial.execute_segment(ExecutePlan.Request(
                object_name='bowl', plan_revision=1, segment=stage), ExecutePlan.Response())
        assert result.success, result.status
    assert trial.record['phase'] == 'LIFTED'
    assert trial.record['completed_stages'] == list(trial.STAGES)
    assert trial.node._held_object == 'bowl' and len(calls) == planning_calls
    with pytest.raises(RuntimeError, match='OWNS_ARM'):
        trial.execution.prepared_grasp_lift = trial
        trial.execution._guard()


def test_public_lift_cannot_skip_contact_or_move_disabled_hardware():
    """Wrong phase and disabled execution reject before sending any command."""
    from my_actions.srv import ExecutePlan
    trial, events, calls = fixture()
    assert prepare(trial).success
    query = ExecutePlan.Request(object_name='bowl', plan_revision=1, segment='lift')
    result = trial.execute_segment(query, ExecutePlan.Response())
    assert not result.success and 'WRONG_EXECUTION_PHASE' in result.status
    trial.node.allow_physical_grasp_execution = False
    query.segment = 'pregrasp'
    result = trial.execute_segment(query, ExecutePlan.Response())
    assert not result.success and result.status == 'PHYSICAL_EXECUTION_DISABLED'
    assert not events and trial.record['phase'] == 'PREPARED'


def test_unconfirmed_public_close_blocks_lift_and_retries():
    """Uncertain gripper contact cannot create a held object or advance the session."""
    from my_actions.srv import ExecutePlan, PreparedGripperCommand
    trial, events, calls = fixture()
    assert prepare(trial).success
    for stage in ('pregrasp', 'gripper_open', 'approach'):
        if stage == 'gripper_open':
            assert trial.open_gripper(PreparedGripperCommand.Request(
                object_name='bowl', plan_revision=1), PreparedGripperCommand.Response()).success
        else:
            assert trial.execute_segment(ExecutePlan.Request(
                object_name='bowl', plan_revision=1, segment=stage), ExecutePlan.Response()).success
    trial.node._wait_for_gripper_contact = lambda: False
    result = trial.close_gripper(PreparedGripperCommand.Request(
        object_name='bowl', plan_revision=1), PreparedGripperCommand.Response())
    assert not result.success and 'STATE_UNKNOWN' in result.status
    assert trial.node._held_object is None and trial.record['phase'] == 'STATE_UNKNOWN'
    before = len(events)
    assert not trial.execute_segment(ExecutePlan.Request(
        object_name='bowl', plan_revision=1, segment='lift'), ExecutePlan.Response()).success
    assert len(events) == before


def test_plan_lift_adapter_forces_lift_only_without_mutating_request():
    """Public lift planning delegates the same solver and cannot request transfer."""
    trial, _, calls = fixture()
    query = PlanPlacement.Request(lift_only=False, plan_pick_place_motion=False)
    observed = []
    trial.node.step_runtime._call = lambda name, request: (
        observed.append((name, request.lift_only, request.plan_pick_place_motion)),
        PlanPlacement.Response(success=True))[1]
    assert trial.plan_lift(query, PlanPlacement.Response()).success
    assert observed == [('plan_placement', True, True)]
    assert not query.lift_only and not query.plan_pick_place_motion


@pytest.mark.parametrize('field,value', [('width_m', float('nan')), ('force_n', -1.)])
def test_public_gripper_rejects_invalid_override_before_actuation(field, value):
    """Defaulted commands still reject nonfinite and nonmatching values."""
    from my_actions.srv import ExecutePlan, PreparedGripperCommand
    trial, events, _ = fixture()
    assert prepare(trial).success
    assert trial.execute_segment(ExecutePlan.Request(
        object_name='bowl', plan_revision=1, segment='pregrasp'), ExecutePlan.Response()).success
    query = PreparedGripperCommand.Request(object_name='bowl', plan_revision=1)
    setattr(query, field, value)
    before = len(events)
    result = trial.open_gripper(query, PreparedGripperCommand.Response())
    assert not result.success and 'DOES_NOT_MATCH' in result.status
    assert len(events) == before and trial.record['phase'] == 'AT_PREGRASP'


def test_started_public_session_survives_preparation_and_source_expiry():
    """Accepted execution keeps immutable source rather than expiring between stages."""
    from my_actions.srv import ExecutePlan, PreparedGripperCommand
    trial, events, _ = fixture()
    assert prepare(trial).success
    assert trial.execute_segment(ExecutePlan.Request(
        object_name='bowl', plan_revision=1, segment='pregrasp'), ExecutePlan.Response()).success
    trial.record['created'] -= 90
    trial.execution._cache.clear()  # Ordinary source cache pruning cannot destroy active execution.
    result = trial.open_gripper(PreparedGripperCommand.Request(
        object_name='bowl', plan_revision=1), PreparedGripperCommand.Response())
    assert result.success and trial.record['phase'] == 'OPENED'


def test_empty_hand_recovery_clears_session_cache_and_all_matching_queues():
    """Failed physical gate leaves a clean new-plan state without actuator commands."""
    from my_actions.srv import ExecutePlan
    from task_pipeline.context import StepStore
    from task_pipeline.recipes import grasp_and_lift
    trial, events, calls = fixture()
    store = StepStore(lambda *_: None)
    trial.node.step_runtime.store = store
    context, _ = store.submit('', grasp_and_lift('bowl', use_cached=True))
    assert prepare(trial).success
    trial.node.allow_physical_grasp_execution = False
    result = trial.execute_segment(ExecutePlan.Request(
        object_name='bowl', plan_revision=1, segment='pregrasp'), ExecutePlan.Response())
    assert not result.success
    assert trial.recover_after_failure() == 'SESSION_RESET_READY_FOR_NEW_PLAN'
    import json
    state = json.loads(store.snapshot(context)[0])[0]
    assert not state['pending'] and state['status'] == 'FLUSHED_AFTER_EXECUTION_FAILURE'
    assert trial.record is None and not trial.execution._cache
    assert not trial.node.cached_grasp_plans and not events


def test_known_pregrasp_failure_can_reset_without_restarting_node():
    """A previous known empty-hand endpoint does not hold the planning session forever."""
    from my_actions.srv import ExecutePlan
    trial, events, _ = fixture()
    assert prepare(trial).success
    assert trial.execute_segment(ExecutePlan.Request(
        object_name='bowl', plan_revision=1, segment='pregrasp'), ExecutePlan.Response()).success
    count = len(events)
    assert trial.recover_after_failure() == 'SESSION_RESET_READY_FOR_NEW_PLAN'
    assert trial.record is None and trial.node.state == 'IDLE' and len(events) == count


def test_unknown_contact_cannot_be_reset_to_empty_hand():
    """A failed close retains reconciliation state; reset is not contact evidence."""
    from std_srvs.srv import Trigger
    trial, events, _ = fixture()
    assert prepare(trial).success
    trial.record['phase'] = 'STATE_UNKNOWN'
    trial.node.state = 'ERROR'
    result = trial.reset_session(Trigger.Request(), Trigger.Response())
    assert not result.success and 'RECONCILIATION_REQUIRED' in result.message
    assert trial.record['phase'] == 'STATE_UNKNOWN' and trial.node.state == 'ERROR'
    assert not events


def test_active_session_timeout_remains_bounded_and_does_not_command_hardware():
    """Separate execution timeout still stops stale active sessions."""
    from my_actions.srv import ExecutePlan, PreparedGripperCommand
    trial, events, _ = fixture()
    assert prepare(trial).success
    assert trial.execute_segment(ExecutePlan.Request(
        object_name='bowl', plan_revision=1, segment='pregrasp'), ExecutePlan.Response()).success
    trial.record['execution_started'] -= 301
    count = len(events)
    result = trial.open_gripper(PreparedGripperCommand.Request(
        object_name='bowl', plan_revision=1), PreparedGripperCommand.Response())
    assert not result.success and result.status == 'EXECUTION_SESSION_TIMED_OUT'
    assert len(events) == count


def test_uncertain_session_requires_explicit_stationary_empty_hand_confirmation():
    """Manual reconciliation changes bookkeeping only after operator confirmation."""
    from std_srvs.srv import SetBool
    trial, events, _ = fixture()
    assert prepare(trial).success
    trial.record['phase'], trial.node.state = 'STATE_UNKNOWN', 'ERROR'
    trial.node.current_joint_states.velocity = [0.]*6
    refused = trial.reconcile_empty_hand(SetBool.Request(data=False), SetBool.Response())
    assert not refused.success and trial.record
    trial.node.current_joint_states.velocity[0] = .1
    refused = trial.reconcile_empty_hand(SetBool.Request(data=True), SetBool.Response())
    assert not refused.success and 'STATIONARY' in refused.message
    trial.node.current_joint_states.velocity[0] = 0.
    result = trial.reconcile_empty_hand(SetBool.Request(data=True), SetBool.Response())
    assert result.success and trial.record is None and trial.node.state == 'IDLE'
    assert not events and trial.node._held_object is None


def test_unknown_failure_flushes_waiting_tasks_without_claiming_reset():
    """Discard future actuation while retaining uncertainty and failed-step history."""
    from task_pipeline.context import StepStore
    from task_pipeline.recipes import grasp_and_lift
    import json
    trial, events, _ = fixture()
    assert prepare(trial).success
    store = StepStore(lambda *_: None)
    trial.node.step_runtime.store = store
    context, _ = store.submit('', grasp_and_lift('bowl', use_cached=True))
    trial.record['phase'], trial.node.state = 'STATE_UNKNOWN', 'ERROR'
    status = trial.recover_after_failure()
    assert status == 'STATE_RECONCILIATION_REQUIRED_SESSION_NOT_RESET'
    assert not json.loads(store.snapshot(context)[0])[0]['pending']
    assert trial.record['phase'] == 'STATE_UNKNOWN' and not events


def test_other_object_failure_cannot_reset_existing_execution_session():
    """Fresh planning for another object must not erase a different owner's session."""
    trial, events, _ = fixture()
    assert prepare(trial).success
    result = trial.recover_after_failure('cup')
    assert result == 'OTHER_EXECUTION_SESSION_PRESERVED'
    assert trial.record['key'] == 'bowl' and not events


def test_approach_stream_does_not_wait_for_intermediate_joint_arrival(monkeypatch):
    """Lagging feedback must not prevent the next prevalidated command."""
    trial, _, _ = fixture()
    sent = []
    clock = [time.monotonic()]
    monkeypatch.setattr('master_node.prepared_grasp_lift.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('master_node.prepared_grasp_lift.time.sleep',
                        lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    trial.node.move_arm_to = lambda *_args, **_kw: pytest.fail('Blocking move used')

    def publish(*, arm_positions):
        sent.append(list(arm_positions))
        # Feedback remains at start until final command is issued.
        if arm_positions == [.003]*6:
            trial.node.current_joint_states.position = list(arm_positions)
        return True

    trial.node._publish_joint_state_command = publish
    target = [.003]*6
    assert trial.stream_approach([target], [0.]*6) == target
    assert len(sent) == 2
    assert sent[-1] == target
    assert clock[0] >= trial.node.joint_feedback_time+.019


def test_approach_stream_stops_before_sending_on_fault():
    """Collision monitoring stays active without per-waypoint arrival waits."""
    trial, _, _ = fixture()
    trial.node.physical_collision_detected = True
    trial.node._publish_joint_state_command = lambda **_: pytest.fail('Command after fault')
    with pytest.raises(RuntimeError, match='ACTIVE_COLLISION'):
        trial.stream_approach([[.001]*6], [0.]*6)


def test_lift_stream_monitors_contact_without_waypoint_arrival_wait(monkeypatch):
    """Lift streams prevalidated targets and stops on measured contact loss."""
    trial, _, _ = fixture()
    trial.node.current_gripper_status.force = -1.
    sent = []
    trial.node.move_arm_to = lambda *_a, **_kw: pytest.fail('Blocking waypoint move')
    monkeypatch.setattr('master_node.prepared_grasp_lift.time.sleep', lambda _: None)

    def publish(*, arm_positions):
        sent.append(list(arm_positions))
        trial.node.current_gripper_status.force = 0.
        return True

    trial.node._publish_joint_state_command = publish
    with pytest.raises(RuntimeError, match='HELD_CONTACT_LOST'):
        trial.follow([[.003]*6], [0.]*6, True, 'lift')
    assert len(sent) == 1


def test_pregrasp_does_not_command_initial_out_of_bounds_feedback():
    trial, _, _ = fixture()
    measured = [0., -.01, -.6, 0., 1.23, 0.]
    entry = [0., 0., -.6, 0., 1.22, 0.]
    calls = []
    trial.execution.held_motion.feedback = lambda *a, **kw: None
    trial.node.physical_collision_detected = trial.node.collision_detected = False
    trial.node.move_arm_to = lambda joints, **kw: calls.append(list(joints)) or True
    assert trial.follow([measured, entry], measured, False, 'pregrasp') == entry
    assert calls == [entry]


@pytest.mark.parametrize('stage', ['approach', 'lift', 'transfer', 'descent', 'withdrawal', 'retreat'])
def test_faster_stream_uses_fewer_commands_on_same_validated_edge(monkeypatch, stage):
    trial, _, _ = fixture()
    clock = [0.]
    sent = []
    trial.execution.held_motion.feedback = lambda *a, **kw: None
    trial.node.physical_collision_detected = trial.node.collision_detected = False
    monkeypatch.setattr('master_node.prepared_grasp_lift.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('master_node.prepared_grasp_lift.time.sleep',
                        lambda seconds: clock.__setitem__(0, clock[0]+seconds))

    def publish(*, arm_positions):
        sent.append(list(arm_positions))
        trial.node.current_joint_states.position = list(arm_positions)
        return True

    trial.node._publish_joint_state_command = publish
    target = [.09]*6
    assert trial.stream_approach([target], [0.]*6, stage=stage) == target
    assert len(sent) == 58  # 100 Hz at 0.1575 rad/s along the same edge.
    assert clock[0] == pytest.approx(.58)  # Previously .4 seconds.
    assert sent[-1] == target
    assert all(0. <= q <= .09 for sample in sent for q in sample)
    previous = [0.]*6
    for sample in sent:
        assert max(abs(a-b) for a, b in zip(sample, previous)) <= .1575*.01+1.e-9
        previous = sample


def test_plan_lift_clears_placement_batch_policy_before_dispatch():
    trial, events, calls = fixture()
    observed = []
    def backend(name, query):
        observed.append((name, query.lift_only, query.validation_batch_size,
                         query.retry_motion_failures))
        return PlanPlacement.Response(success=True, motion_success=True)
    trial.node.step_runtime._call = backend
    query = PlanPlacement.Request(validation_batch_size=8, retry_motion_failures=True)
    assert trial.plan_lift(query, PlanPlacement.Response()).success
    assert observed == [('plan_placement', True, 0, False)]
    assert query.validation_batch_size == 8 and query.retry_motion_failures
    assert not events


def settling_trial(monkeypatch):
    """Simulate independent TCP feedback delay without publishing commands."""
    import copy
    trial, _, _ = fixture()
    clock = [time.monotonic()]
    target = copy.deepcopy(trial.node.current_tcp_pose)
    target.pose.position.z -= .02
    trial.node._held_object = 'bowl'
    trial.execution.held_motion.feedback = lambda *a, **kw: None
    trial.node._publish_joint_state_command = lambda **kw: pytest.fail('Command while settling')
    monkeypatch.setattr('master_node.prepared_grasp_lift.time.monotonic', lambda: clock[0])
    return trial, clock, target


def test_endpoint_waits_for_delayed_pose_with_contact_monitoring(monkeypatch):
    trial, clock, target = settling_trial(monkeypatch)
    start = clock[0]
    contacts = []
    trial.execution.held_motion.feedback = lambda *a, **kw: contacts.append(kw['contact'])

    def sleep(seconds):
        clock[0] += seconds
        trial.node.pose_feedback_time = clock[0]
        if clock[0]-start >= .3:
            trial.node.current_tcp_pose = target

    monkeypatch.setattr('master_node.prepared_grasp_lift.time.sleep', sleep)
    trial.check_pose(target, settle=True)
    assert .3 <= clock[0]-start < .35
    assert contacts and all(contacts)


def test_endpoint_settling_is_bounded_and_reports_error(monkeypatch):
    trial, clock, target = settling_trial(monkeypatch)
    start = clock[0]

    def sleep(seconds):
        clock[0] += seconds
        trial.node.pose_feedback_time = clock[0]

    monkeypatch.setattr('master_node.prepared_grasp_lift.time.sleep', sleep)
    with pytest.raises(RuntimeError, match='position_error_m=0.020000'):
        trial.check_pose(target, settle=True)
    assert 2. <= clock[0]-start < 2.05


@pytest.mark.parametrize('fault', ['ACTIVE_COLLISION', 'HELD_CONTACT_LOST', 'STALE_HARDWARE_FEEDBACK'])
def test_endpoint_settling_keeps_safety_monitoring(monkeypatch, fault):
    trial, _, target = settling_trial(monkeypatch)

    def feedback(*args, **kwargs):
        raise RuntimeError(fault)

    trial.execution.held_motion.feedback = feedback
    with pytest.raises(RuntimeError, match=fault):
        trial.check_pose(target, settle=True)


def test_matching_but_old_tcp_feedback_must_refresh(monkeypatch):
    trial, clock, target = settling_trial(monkeypatch)
    trial.node.current_tcp_pose = target
    trial.node.pose_feedback_time = clock[0]-.5
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds
        trial.node.pose_feedback_time = clock[0]

    monkeypatch.setattr('master_node.prepared_grasp_lift.time.sleep', sleep)
    trial.check_pose(target, settle=True)
    assert slept


def test_start_pose_checks_do_not_wait_or_relax(monkeypatch):
    trial, _, target = settling_trial(monkeypatch)
    monkeypatch.setattr('master_node.prepared_grasp_lift.time.sleep',
                        lambda _: pytest.fail('Start-state check waited'))
    with pytest.raises(RuntimeError, match='PREPARED_MOTION_POSE_MISMATCH'):
        trial.check_pose(target)
