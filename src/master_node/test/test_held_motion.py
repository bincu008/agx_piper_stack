"""Guard payload execution using fake hardware; never move a physical arm."""

import threading
import time
from types import SimpleNamespace

from master_node.execution_api import ExecutionAPI
from master_node.held_motion import HeldMotion

from my_actions.srv import HeldMotionStage, PlanPlacement

import pytest


def fixture():
    """Create an already-confirmed grasp and bounded lift with fake feedback."""
    events = []
    now = time.monotonic()
    refined = {'grasp_joints': [0.] * 6, 'width': .04, 'grasp_pose': None,
               'observation_header': None, 'seed_base': [0.] * 3, 'closing_axis': [0., 1., 0.]}
    node = SimpleNamespace(
        allow_physical_grasp_execution=True, state='IDLE', _held_object='bowl',
        joint_limits=[(-3., 3.)] * 6,
        physical_collision_detected=False, collision_detected=False,
        joint_feedback_time=now, pose_feedback_time=now, gripper_feedback_time=now,
        current_joint_states=SimpleNamespace(position=[0.] * 6),
        current_gripper_status=SimpleNamespace(width=.04, force=1.),
        current_tcp_pose=None, refined_grasp_plans={'bowl': refined},
        _pose_error=lambda *_: (0., 0.),
        set_arm_speed=lambda v: (events.append(('speed', v)), True)[1])

    def move(joints, **_):
        events.append(('move', joints))
        node.current_joint_states.position = list(joints)
        return True

    node.move_arm_to = move
    execution = object.__new__(ExecutionAPI)
    execution.node, execution._lock = node, threading.Lock()
    held = object.__new__(HeldMotion)
    held.node, held.execution = node, execution
    plan = PlanPlacement.Response(lift_path_flat=[0.] * 6 + [.001] * 6,
                                  motion_status='LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED')
    held.record = {'key': 'bowl', 'revision': 1, 'plan': plan, 'prepared': now,
                   'phase': 'PREPARED', 'signature': held.signature(refined),
                   'lift_only': True, 'expected': [0.] * 6}
    return held, events


def call(held, operation='execute_lift', revision=1):
    """Call one stage against fake feedback."""
    return held.stage(operation, HeldMotionStage.Request(
                          object_name='bowl', plan_revision=revision),
                      HeldMotionStage.Response())


def test_lift_stops_holding_and_cannot_be_repeated():
    """Preserve the payload and reject repeated lift."""
    held, events = fixture()
    result = call(held)
    assert result.success and result.motion_completed
    assert held.node._held_object == 'bowl' and held.record['phase'] == 'LIFTED'
    assert events[0] == ('speed', 5) and events[-1] == ('speed', 20)
    assert 'BYPASSED' in result.status
    previous = list(events)
    assert not call(held).success
    assert not call(held, 'execute_placement_transfer').success
    assert events == previous


@pytest.mark.parametrize(
    'problem', ['gate', 'revision', 'stale', 'contact', 'attachment', 'start'])
def test_reject_before_actuation(problem):
    """Reject invalid inputs before publishing commands."""
    held, events = fixture()
    revision = 1
    if problem == 'gate':
        held.node.allow_physical_grasp_execution = False
    elif problem == 'revision':
        revision = 2
    elif problem == 'stale':
        held.node.gripper_feedback_time = 0
    elif problem == 'contact':
        held.node.current_gripper_status.force = 0.
    elif problem == 'attachment':
        held.node.refined_grasp_plans['bowl']['width'] = .05
    else:
        held.node.current_joint_states.position = [.1] * 6
    assert not call(held, revision=revision).success
    assert not events


def test_failed_waypoint_is_unknown_and_not_retried():
    """Latch unknown state after attempted execution."""
    held, events = fixture()
    held.node.move_arm_to = lambda *_args, **_kw: False
    result = call(held)
    assert not result.success and result.status.startswith('STATE_UNKNOWN')
    assert held.record['phase'] == 'STATE_UNKNOWN' and held.node.state == 'ERROR'
    assert events[-1] == ('speed', 20)
    count = len(events)
    assert not call(held).success and len(events) == count


def test_release_requires_completed_descent_and_confirmed_opening():
    """Retain held state until release feedback confirms opening."""
    held, events = fixture()
    held.record['lift_only'] = False
    held.record['phase'] = 'AT_PREPLACE'
    assert not call(held, 'release_placed_object').success and not events
    held.record['phase'] = 'PLACED'
    held.node.gripper_open_force_n = 1.
    held.node._command_gripper = lambda *_: True
    held.node._wait_for_gripper_release = lambda *_: False
    result = call(held, 'release_placed_object')
    assert not result.success and not result.gripper_released
    assert held.node._held_object == 'bowl' and held.record['phase'] == 'STATE_UNKNOWN'


@pytest.mark.parametrize('height', [.001, .05, .15])
def test_preparation_uses_refinement_and_rejects_discontinuous_backend_path(height):
    """Bind authoritative refinement and segment continuity before actuator calls."""
    from geometry_msgs.msg import PoseStamped
    from my_actions.msg import GraspCandidate
    from my_actions.srv import CacheGraspPlan, PrepareHeldMotion, PrepareHeldScene
    from sensor_msgs.msg import Image, PointCloud2
    held, events = fixture()
    held.node._held_object = None
    candidate = GraspCandidate(id=7)
    pose = PoseStamped()
    pose.header.frame_id = 'base_link'
    pose.header.stamp.sec = 10
    pose.pose.orientation.w = 1.
    refined = held.node.refined_grasp_plans['bowl']
    refined.update(grasp_pose=pose, pregrasp_joints=[0.] * 6)
    source = CacheGraspPlan.Request(
        observation_header=pose.header, ttl_seconds=60,
        observation_cloud=PointCloud2(width=20), source_mask=Image(width=20))
    held.node.cached_grasp_plans = {'bowl': {'candidate': candidate}}
    held.execution._cache_lock = threading.Lock()
    held.execution._cache = {'bowl': (time.monotonic(), source)}
    held.execution.grasp_stages = SimpleNamespace(phases={'bowl': 'PLANNED'})
    held.record = None
    held.revision = 0
    requests = []

    def backend(name, query):
        requests.append((name, query))
        if name == 'prepare_held_scene':
            return PrepareHeldScene.Response(success=True)
        return PlanPlacement.Response(success=True, motion_success=True,
                                      selected_grasp_candidate=query.grasp_candidate,
                                      lift_path_flat=[0.] * 6 + [.001] * 6)

    held.node.step_runtime = SimpleNamespace(_call=backend)
    query = PrepareHeldMotion.Request(object_name='bowl', lift_only=True, lift_distance_m=height)
    result = held.prepare(query, PrepareHeldMotion.Response())
    assert result.success and result.plan_revision == 1 and not events
    plan_request = requests[-1][1]
    assert plan_request.grasp_candidate.grasp_pose == pose
    assert plan_request.lift_only and plan_request.lift_distance_m == pytest.approx(height)

    def discontinuous(name, query):
        response = backend(name, query)
        if name == 'plan_placement':
            response.lift_path_flat = [.5] * 6
        return response

    held.node.step_runtime._call = discontinuous
    result = held.prepare(query, PrepareHeldMotion.Response())
    assert not result.success and result.status == 'HELD_SEGMENT_START_DISCONTINUITY'
    assert held.record['revision'] == 1 and not events


def test_full_ordered_payload_motion_and_confirmed_release():
    """Execute every stage on fake hardware and clear payload only at release."""
    held, events = fixture()
    held.record['lift_only'] = False
    plan = held.record['plan']
    plan.transfer_path_flat = [.001] * 6 + [.002] * 6
    plan.descent_path_flat = [.002] * 6 + [.003] * 6
    for stage in ('execute_lift', 'execute_placement_transfer', 'execute_placement_descent'):
        assert call(held, stage).success
        assert held.node._held_object == 'bowl'
    held.node.gripper_open_force_n = 1.
    held.node._command_gripper = lambda *_: True
    held.node._wait_for_gripper_release = lambda *_: True
    response = call(held, 'release_placed_object')
    assert response.success and response.gripper_released
    assert held.node._held_object is None and held.record['phase'] == 'RELEASED'
    assert events[-1] == ('speed', 20)


def test_held_contact_accepts_negative_gripping_force():
    """Keep a signed measured contact while rejecting insufficient magnitude."""
    held, _ = fixture()
    held.node.current_gripper_status.force = -1.043
    held.feedback(held.node.current_joint_states.position[:6])
    held.node.current_gripper_status.force = -.1
    with pytest.raises(RuntimeError, match='HELD_CONTACT_LOST'):
        held.feedback(held.node.current_joint_states.position[:6])


def test_stationary_contact_settles_without_actuator_retry(monkeypatch):
    """A brief force dip before lift can recover within a fixed deadline."""
    held, events = fixture()
    clock = [time.monotonic()]
    held.node.current_gripper_status.force = 0.
    monkeypatch.setattr('master_node.held_motion.time.monotonic', lambda: clock[0])

    def sleep(seconds):
        clock[0] += seconds
        held.node.current_gripper_status.force = -.95
        held.node.gripper_feedback_time = clock[0]

    monkeypatch.setattr('master_node.held_motion.time.sleep', sleep)
    held.feedback(held.node.current_joint_states.position[:6], settle_sec=.3)
    assert not events


def test_contact_settling_does_not_accept_persistent_loss(monkeypatch):
    """Persistent loss produces useful measurements and stays bounded."""
    held, events = fixture()
    clock = [time.monotonic()]
    held.node.current_gripper_status.force = -.05
    monkeypatch.setattr('master_node.held_motion.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('master_node.held_motion.time.sleep',
                        lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    with pytest.raises(RuntimeError, match='force=-0.050000N'):
        held.feedback(held.node.current_joint_states.position[:6], settle_sec=.3)
    assert not events


def test_thin_rim_remains_valid_during_held_lift():
    """The post-close checker uses the same thin-rim geometry as confirmation."""
    held, events = fixture()
    held.node.current_gripper_status = SimpleNamespace(width=.00056, force=-.24)
    result = call(held)
    assert result.success, result.status


def test_held_feedback_uses_sustained_force_evidence_for_brief_dip():
    """An isolated low reading does not contradict the confirmed stable grasp."""
    from master_node.contact_evidence import SettledContact
    held, events = fixture()
    gate = SettledContact()
    now = time.monotonic()
    sample = SimpleNamespace(width=.00056, force=-.25)
    for i in range(11):
        stamp = now-.2+i*.02
        sample.force = -.12 if i == 10 else -.25
        gate.update(sample, stamp, stamp)
    held.node.current_gripper_status = sample
    held.node.gripper_contact_evidence = gate
    held.feedback([0.]*6)
    assert not events
    # Replay confirmed contact followed by sustained load loss, ending now.
    loss = SettledContact()
    for i in range(11):
        stamp = now-.4+i*.02
        loss.update(SimpleNamespace(width=.00056, force=-.25), stamp, stamp)
    for i in range(1, 11):
        stamp = now-.2+i*.02
        loss.update(SimpleNamespace(width=.00056, force=0.), stamp, stamp)
    held.node.gripper_contact_evidence = loss
    with pytest.raises(RuntimeError, match='CONTACT_BELOW_HOLDING_THRESHOLD'):
        held.feedback([0.]*6)
