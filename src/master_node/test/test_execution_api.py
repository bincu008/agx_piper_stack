"""Exercise new execution boundaries with fake callbacks and no hardware."""

import threading
from types import SimpleNamespace

from master_node.execution_api import ExecutionAPI
from master_node.grasp_stages import GraspStages

from my_actions.msg import GraspCandidate
from my_actions.srv import CheckAndReplan, ExecuteTrajectory, GraspStage, GripperCommand, MoveTo

import pytest


def test_joint_limit_error_identifies_measured_joint_without_clamping():
    """Explain the live start-state failure and preserve strict limits."""
    api = object.__new__(ExecutionAPI)
    api.node = SimpleNamespace(joint_limits=[(-3., 3.)]*4 + [(-1.2217304, 1.2217304)]
                               + [(-3., 3.)])
    joints = [0., 0., 0., 0., 1.2389568761132146, 0.]
    with pytest.raises(ValueError, match='JOINT_LIMIT_VIOLATION: joint5=.*70.987'):
        api._joints(joints)
    assert joints[4] == 1.2389568761132146
    joints[4] = 1.2217304
    api._joints(joints)


def test_master_loads_two_degree_margin_for_all_joints():
    """Use unchanged measured joints and expand each model boundary equally."""
    import math
    from master_node.piper_master_control import PiperMasterControl
    logger = SimpleNamespace(info=lambda *_: None, error=lambda *_: None)
    node = SimpleNamespace(
        declare_parameter=lambda *_: None,
        get_parameter=lambda _: SimpleNamespace(value=2.0),
        get_logger=lambda: logger)
    PiperMasterControl.load_joint_limits(node)
    nominal = [(-2.6179938, 2.6179938), (0., 3.1415926), (-2.9670597, 0.),
               (-1.7453292, 1.7453292), (-1.2217304, 1.2217304), (-2.0943951, 2.0943951)]
    margin = math.radians(2.)
    for actual, (lower, upper) in zip(node.joint_limits, nominal):
        assert actual == pytest.approx((lower-margin, upper+margin))
    api = object.__new__(ExecutionAPI)
    api.node = node
    api._joints([0., -.005969, -.52, 0., 1.238956876, 0.])
    with pytest.raises(ValueError, match='joint2'):
        api._joints([0., -math.radians(2.1), -.52, 0., 0., 0.])


@pytest.mark.parametrize('name,query,response', [
    ('_move', MoveTo.Request(), MoveTo.Response()),
    ('_trajectory', ExecuteTrajectory.Request(), ExecuteTrajectory.Response()),
    ('_gripper', GripperCommand.Request(), GripperCommand.Response()),
])
def test_all_new_actuator_apis_require_existing_physical_gate(name, query, response):
    """Reject before accessing a planner, publishing commands or setting speed."""
    api = object.__new__(ExecutionAPI)
    api._lock = threading.Lock()
    api.node = SimpleNamespace(allow_physical_grasp_execution=False)
    result = getattr(api, name)(query, response)
    assert not result.success and result.status == 'PHYSICAL_EXECUTION_DISABLED'


def fixture():
    """Supply only fake actuator methods and a valid ordered grasp context."""
    commands, speeds = [], []
    candidate = GraspCandidate(ik_valid=True, collision_checked=True,
                               grasp_collision_valid=True, pregrasp_collision_valid=True,
                               score=1., measured_contacts_valid=True)
    refined = {'width': .04, 'grasp_pose': None}
    node = SimpleNamespace(
        allow_physical_grasp_execution=True, state='IDLE', physical_collision_detected=False,
        collision_detected=False, allow_model_based_body_grasp_execution=False,
        cached_grasp_plans={'cup': {'candidate': candidate}},
        refined_grasp_plans={'cup': refined}, current_tcp_pose=None,
        gripper_open_force_n=1., _held_object=None,
        _verify_refined_pregrasp=lambda _: (True, 'OK'),
        _pose_error=lambda *args: (0., 0.),
        _command_gripper=lambda *args: (commands.append(args), True)[1],
        _wait_for_gripper_opening=lambda _: True,
        _wait_for_gripper_contact=lambda: True,
        set_arm_speed=lambda value: (speeds.append(value), True)[1],
        _execute_validated_final_approach=lambda _: (True, 'OK'))
    execution = object.__new__(ExecutionAPI)
    execution._lock, execution.node = threading.Lock(), node
    stage = object.__new__(GraspStages)
    stage.execution, stage.node = execution, node
    stage.phases, stage.paths = {'cup': 'PLANNED'}, {'cup': [[0.] * 6]}
    return stage, commands, speeds


def test_gripper_close_is_rejected_before_validated_approach():
    """Do not publish a close command when callers skip required stages."""
    stages, commands, _ = fixture()
    result = stages._stage('close_grasp_gripper', GraspStage.Request(object_name='cup'),
                           GraspStage.Response())
    assert not result.success and result.status.startswith('GRASP_STAGE_ORDER_VIOLATION')
    assert not commands


def test_busy_call_does_not_invalidate_another_active_grasp():
    """Reject concurrent requests without deleting the active operation's plan."""
    stages, _, _ = fixture()
    stages.execution._lock.acquire()
    result = stages._stage('open_grasp_gripper', GraspStage.Request(object_name='cup'),
                           GraspStage.Response())
    stages.execution._lock.release()
    assert result.status == 'EXECUTION_API_BUSY'
    assert stages.phases['cup'] == 'PLANNED'


def test_approach_failure_always_restores_speed_and_prevents_close():
    """Retain fixed-path execution and restore 20 percent on its failure path."""
    stages, commands, speeds = fixture()
    stages.phases['cup'] = 'OPENED'
    stages.node._execute_validated_final_approach = lambda _: (False, 'COLLISION')
    result = stages._stage('execute_grasp_approach', GraspStage.Request(object_name='cup'),
                           GraspStage.Response())
    assert not result.success and result.status == 'COLLISION'
    assert speeds == [5, 20] and not commands and 'cup' not in stages.phases


def test_ordered_grasp_stages_confirm_contact_before_marking_object_held():
    """Command completion alone never changes authoritative held-object state."""
    stages, commands, speeds = fixture()
    request = GraspStage.Request(object_name='cup')
    for name in ('open_grasp_gripper', 'execute_grasp_approach', 'close_grasp_gripper'):
        assert stages._stage(name, request, GraspStage.Response()).success
        assert stages.node._held_object is None
    result = stages._stage('verify_grasp_contact', request, GraspStage.Response())
    assert result.success and result.gripper_contact and stages.node._held_object == 'cup'
    assert commands[0] == pytest.approx((.06, 1.))
    assert commands[1] == pytest.approx((.02, 1.))
    assert speeds == [5, 20]


def test_trajectory_claims_master_state_and_restores_it_after_failure():
    """Legacy motion callbacks cannot start during API planning or execution."""
    api = object.__new__(ExecutionAPI)
    api._lock = threading.Lock()
    seen = []
    node = SimpleNamespace(
        allow_physical_grasp_execution=True, state='IDLE', _held_object=None,
        physical_collision_detected=False, collision_detected=False,
        joint_limits=[(-1., 1.)] * 6,
        current_joint_states=SimpleNamespace(position=[0.] * 6))

    def plan(name, request):
        seen.append(node.state)
        return CheckAndReplan.Response(success=True, path_flat=[0.] * 6)

    def move(joints):
        seen.append(node.state)
        return False

    node.step_runtime = SimpleNamespace(_call=plan)
    node.move_arm_to = move
    api.node = node
    result = api._trajectory(
        ExecuteTrajectory.Request(path_flat=[0.] * 6, expected_start_joints=[0.] * 6),
        ExecuteTrajectory.Response())
    assert not result.success and result.status == 'TRAJECTORY_EXECUTION_FAILED'
    assert seen == ['BUSY', 'MOVING'] and node.state == 'IDLE'


def test_trajectory_does_not_recommand_initial_measured_sample():
    """A measured start anchors the path but is not an actuator target."""
    api = object.__new__(ExecutionAPI)
    api._lock = threading.Lock()
    commanded = []
    api.node = SimpleNamespace(
        allow_physical_grasp_execution=True, state='IDLE', _held_object=None,
        physical_collision_detected=False, collision_detected=False,
        joint_limits=[(-1., 1.)] * 6,
        current_joint_states=SimpleNamespace(position=[0.] * 6),
        step_runtime=SimpleNamespace(_call=lambda *args: CheckAndReplan.Response(
            success=True, path_flat=[0.] * 6 + [.1] * 6)),
        move_arm_to=lambda joints: (commanded.append(joints), True)[1])
    result = api._trajectory(ExecuteTrajectory.Request(
        path_flat=[0.] * 6 + [.1] * 6, expected_start_joints=[0.] * 6),
        ExecuteTrajectory.Response())
    assert result.success and len(commanded) == 1
    assert commanded[0] == pytest.approx([.1] * 6)


def test_grasp_seed_numpy_box_serializes_with_native_python_integer_fields():
    """Exercise the actual ROS CDR conversion that aborted the mock master."""
    import numpy as np
    from my_actions.srv import PrepareGraspInput
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header
    header = Header(frame_id='base_link')
    header.stamp.sec = 1
    query = PrepareGraspInput.Request(
        observation_header=header, image=Image(header=header, width=640, height=480),
        mask=Image(header=header, width=640, height=480),
        pixel_box=np.array([10, 100, 262, 400], dtype=np.int32), target_description='cup')
    result = ExecutionAPI._prepare(object.__new__(ExecutionAPI), query,
                                   PrepareGraspInput.Response())
    assert result.success and type(result.seed_u) is int and type(result.seed_v) is int
    assert (result.seed_u, result.seed_v) == (136, 250)
    assert serialize_message(result)


def pool_api():
    from my_actions.srv import CacheGraspPlan
    api = object.__new__(ExecutionAPI)
    api._lock, api._cache_lock = threading.Lock(), threading.Lock()
    api._cache, api._current_pool = {}, None
    api.node = SimpleNamespace(
        _held_object=None, cached_grasp_plans={}, refined_grasp_plans={},
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            nanoseconds=1_800_000_000_000_000_000)))
    api.grasp_stages = SimpleNamespace(invalidate=lambda _: None)
    request = CacheGraspPlan.Request(object_name='cup', candidates=[GraspCandidate(
        ik_valid=True, collision_checked=True, grasp_collision_valid=True,
        pregrasp_collision_valid=True, score=1., measured_contacts_valid=True)])
    request.observation_header.stamp.sec = 1_799_999_990
    return api, request, CacheGraspPlan


def test_current_pool_uses_ros_source_age_and_shorter_deadline(monkeypatch):
    api, request, schema = pool_api()
    monkeypatch.setattr('master_node.execution_api.time.monotonic', lambda: 100.)
    request.ttl_seconds = 30.
    assert api._save(request, schema.Response()).success
    assert api.get_current_pool_summary(now=105.)['expires_in_seconds'] == 15.
    assert not api.get_current_pool_summary(now=120.)['fresh']


def test_current_pool_replaces_previous_source_and_consumes_exact_owner(monkeypatch):
    from my_actions.srv import GetCachedGrasp
    api, request, schema = pool_api()
    monkeypatch.setattr('master_node.execution_api.time.monotonic', lambda: 100.)
    assert api._save(request, schema.Response()).success
    request.object_name = 'bowl'
    assert api._save(request, schema.Response()).success
    assert set(api._cache) == {'bowl'}
    assert set(api.node.cached_grasp_plans) == {'bowl'}
    api.claim_current_pool('cup')
    assert api.get_current_pool_summary(now=101.)['fresh']
    api.invalidate_current_pool('CAMERA_MOVED')
    assert not api._cached(GetCachedGrasp.Request(object_name='bowl'),
                           GetCachedGrasp.Response()).success
    assert api.get_current_pool_summary(now=101.)['reason'] == 'CAMERA_MOVED'


def test_cache_rejects_old_source_without_replacing_current_pool(monkeypatch):
    api, request, schema = pool_api()
    monkeypatch.setattr('master_node.execution_api.time.monotonic', lambda: 100.)
    assert api._save(request, schema.Response()).success
    request.observation_header.stamp.sec -= 600
    request.object_name = 'bowl'
    result = api._save(request, schema.Response())
    assert not result.success
    assert result.status == 'REFRESH_REQUIRED_SOURCE_OBSERVATION_AGE'
    assert set(api._cache) == {'cup'}


@pytest.mark.parametrize('kind,changed,reason', [
    ('joints', .006, 'ARM_MOVED_SINCE_POOL_CREATION'),
    ('tcp', .003, 'CAMERA_MOVED_SINCE_POOL_CREATION'),
    ('calibration', 'different', 'CALIBRATION_CHANGED'),
])
def test_measured_change_invalidates_only_unconsumed_pool(monkeypatch, kind, changed, reason):
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    api, request, schema = pool_api()
    monkeypatch.setattr('master_node.execution_api.time.monotonic', lambda: 100.)
    api.node.current_joint_states = JointState(position=[0.]*6)
    api.node.current_tcp_pose = PoseStamped()
    api.node.current_tcp_pose.pose.orientation.w = 1.
    assert api._save(request, schema.Response()).success
    if kind == 'joints':
        message = JointState(position=[changed]+[0.]*5)
    elif kind == 'tcp':
        message = PoseStamped()
        message.pose.orientation.w = 1.
        message.pose.position.x = changed
    else:
        message = changed
    api.observe_pool_feedback(kind, message)
    assert api.get_current_pool_summary(now=101.)['reason'] == reason
    assert not api._cache
    assert api._save(request, schema.Response()).success
    api.claim_current_pool('cup')
    api.observe_pool_feedback(kind, message)
    assert api.get_current_pool_summary(now=101.)['reason'] == 'ALREADY_CONSUMED'


def test_stationary_feedback_does_not_renew_or_discard_pool(monkeypatch):
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    api, request, schema = pool_api()
    monkeypatch.setattr('master_node.execution_api.time.monotonic', lambda: 100.)
    api.node.current_joint_states = JointState(position=[0.]*6)
    api.node.current_tcp_pose = PoseStamped()
    api.node.current_tcp_pose.pose.orientation.w = 1.
    assert api._save(request, schema.Response()).success
    for _ in range(3):
        api.observe_pool_feedback('joints', api.node.current_joint_states)
        api.observe_pool_feedback('tcp', api.node.current_tcp_pose)
        api.observe_pool_feedback('calibration', None)
    assert api.get_current_pool_summary(now=101.)['fresh']
    assert api.get_current_pool_summary(now=690.)['reason'] == 'REFRESH_REQUIRED_EXPIRED'
    monkeypatch.setattr('master_node.execution_api.time.monotonic', lambda: 690.)
    api.observe_pool_feedback('joints', api.node.current_joint_states)
    assert not api._cache and not api.node.cached_grasp_plans
    assert api.get_current_pool_summary(now=690.)['reason'] == 'REFRESH_REQUIRED_EXPIRED'


@pytest.mark.parametrize('condition,expected', [
    ('disabled', 'PHYSICAL_EXECUTION_DISABLED'),
    ('held', 'RELEASE_REQUIRES_VALIDATED_PLACEMENT'),
    ('stale', 'STALE_HARDWARE_FEEDBACK'),
    ('narrow', 'EMPTY_HAND_OPEN_CANNOT_NARROW_GRIPPER'),
    ('no_wait', 'OPEN_REQUIRES_POST_COMMAND_FEEDBACK'),
    ('valid', 'OPENED'),
])
def test_open_only_master_wrapper_preserves_gates_and_feedback(condition, expected):
    commands = []
    api = object.__new__(ExecutionAPI)
    api._lock = threading.Lock()
    api.node = SimpleNamespace(
        allow_physical_grasp_execution=condition != 'disabled', state='IDLE',
        physical_collision_detected=False, collision_detected=False,
        _held_object='cup' if condition == 'held' else None,
        GRIPPER_MIN_FORCE_N=.5, GRIPPER_MAX_FORCE_N=3.,
        current_joint_states=SimpleNamespace(position=[0.]*6),
        current_gripper_status=SimpleNamespace(width=.04),
        _command_gripper=lambda *args: (commands.append(args), True)[1],
        _wait_for_gripper_opening=lambda _: True)

    def feedback(*args, **kwargs):
        if condition == 'stale':
            raise RuntimeError('STALE_HARDWARE_FEEDBACK')

    api.held_motion = SimpleNamespace(feedback=feedback)
    result = api._open_empty_hand(GripperCommand.Request(
        width_m=.03 if condition == 'narrow' else .07, force_n=1.,
        wait_for_opening=condition != 'no_wait'), GripperCommand.Response())
    if condition == 'valid':
        assert result.success and result.feedback_confirmed
        assert result.status == 'OPENING_CONFIRMED'
        assert len(commands) == 1
    else:
        assert not result.success and result.status == expected
        assert not commands
