"""Exercise new execution boundaries with fake callbacks and no hardware."""

import threading
from types import SimpleNamespace

from master_node.execution_api import ExecutionAPI
from master_node.grasp_stages import GraspStages

from my_actions.msg import GraspCandidate
from my_actions.srv import CheckAndReplan, ExecuteTrajectory, GraspStage, GripperCommand, MoveTo

import pytest


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
    assert commands == [(.05, 1.), (0., 1.)] and speeds == [5, 20]


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
