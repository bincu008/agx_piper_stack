from types import SimpleNamespace

from rclpy.time import Time

from master_node.piper_master_control import PiperMasterControl
from my_actions.msg import GraspCandidate
from my_actions.srv import ExecuteGrasp


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Clock:
    @staticmethod
    def now():
        return Time()


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)


class _CommandTarget:
    ARM_JOINT_NAMES = PiperMasterControl.ARM_JOINT_NAMES
    GRIPPER_MIN_FORCE_N = PiperMasterControl.GRIPPER_MIN_FORCE_N
    GRIPPER_MAX_FORCE_N = PiperMasterControl.GRIPPER_MAX_FORCE_N

    def __init__(self):
        self.joint_pub = _Publisher()
        self.logger = _Logger()

    @staticmethod
    def get_clock():
        return _Clock()

    def get_logger(self):
        return self.logger


class _GraspExecutionGateTarget:
    def __init__(self):
        self.cached_grasp_plans = {}
        self.allow_physical_grasp_execution = True
        self.require_grasp_collision_validation = True
        self.allow_pregrasp_collision_override = False
        self.require_measured_contacts_for_grasp = True
        self.allow_model_based_body_grasp_execution = False
        self.require_positive_grasp_score_for_grasp = True
        self.logger = _Logger()

    def get_logger(self):
        return self.logger


def _publish(target, **kwargs):
    return PiperMasterControl._publish_joint_state_command(
        target, **kwargs)


def test_combined_arm_and_gripper_command_matches_driver_contract():
    target = _CommandTarget()

    assert _publish(
        target,
        arm_positions=[0.0, 0.1, -0.1, 0.2, -0.2, 0.3],
        gripper_width=0.05,
        gripper_force=1.5,
    )

    message = target.joint_pub.messages[0]
    assert list(message.name) == [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
        'gripper',
    ]
    assert list(message.position) == [
        0.0, 0.1, -0.1, 0.2, -0.2, 0.3, 0.05,
    ]
    assert list(message.effort) == [0.0] * 6 + [1.5]


def test_gripper_only_command_and_force_validation():
    target = _CommandTarget()

    assert _publish(
        target, gripper_width=0.04, gripper_force=3.0)
    message = target.joint_pub.messages[0]
    assert list(message.name) == ['gripper']
    assert list(message.position) == [0.04]
    assert list(message.effort) == [3.0]

    assert not _publish(
        target, gripper_width=0.02, gripper_force=15.0)
    assert len(target.joint_pub.messages) == 1
    assert target.logger.errors


def _grasp_request(measured_contacts, score):
    request = ExecuteGrasp.Request()
    request.object_name = 'cup'
    request.execute = True
    request.candidate = GraspCandidate()
    request.candidate.ik_valid = True
    request.candidate.collision_checked = True
    request.candidate.grasp_collision_valid = True
    request.candidate.pregrasp_collision_valid = True
    request.candidate.measured_contacts_valid = measured_contacts
    request.candidate.score = score
    return request


def test_physical_grasp_rejects_missing_measured_contacts():
    target = _GraspExecutionGateTarget()
    response = ExecuteGrasp.Response()

    PiperMasterControl.execute_grasp_callback(
        target, _grasp_request(False, 1.0), response)

    assert response.success is False
    assert 'measured opposing contacts' in response.status


def test_physical_grasp_requires_explicit_body_model_gate():
    target = _GraspExecutionGateTarget()
    request = _grasp_request(False, 1.0)
    request.candidate.validation_status = (
        'CUP_BODY_MODEL_GRASP_AND_PREGRASP_SAFE')
    response = ExecuteGrasp.Response()

    PiperMasterControl.execute_grasp_callback(target, request, response)

    assert response.success is False
    assert 'allow_model_based_body_grasp_execution=true' in response.status


def test_physical_grasp_score_gate_remains_explicit():
    target = _GraspExecutionGateTarget()
    response = ExecuteGrasp.Response()

    PiperMasterControl.execute_grasp_callback(
        target, _grasp_request(True, -1.0), response)

    assert response.success is False
    assert 'grasp proposal score is not positive' in response.status


def test_gripper_contact_requires_feedback_newer_than_close_command(monkeypatch):
    """A stale force reading cannot authorize a payload lift."""
    import time
    import master_node.piper_master_control as module
    node = SimpleNamespace(current_gripper_status=SimpleNamespace(width=.02, force=1.),
                           gripper_command_time=time.monotonic(), gripper_feedback_time=0.)
    node.get_logger = lambda: SimpleNamespace(error=lambda _: None)
    monkeypatch.setattr(module.rclpy, 'ok', lambda: True)
    assert not PiperMasterControl._wait_for_gripper_contact(node, timeout_sec=.001)
    node.gripper_feedback_time = time.monotonic()
    original_sleep = time.sleep

    def fresh_sample(seconds):
        original_sleep(seconds)
        node.gripper_feedback_time = time.monotonic()

    monkeypatch.setattr(module.time, 'sleep', fresh_sample)
    assert PiperMasterControl._wait_for_gripper_contact(node, timeout_sec=.4)



def test_release_requires_observed_opening_and_loss_of_force(monkeypatch):
    """Do not clear payload state merely because old contact width is positive."""
    import time
    import master_node.piper_master_control as module
    node = SimpleNamespace(current_gripper_status=SimpleNamespace(width=.04, force=1.),
                           gripper_command_time=time.monotonic(),
                           gripper_feedback_time=time.monotonic())
    node.get_logger = lambda: SimpleNamespace(error=lambda _: None)
    monkeypatch.setattr(module.rclpy, 'ok', lambda: True)
    assert not PiperMasterControl._wait_for_gripper_release(node, .04, timeout_sec=.001)
    node.current_gripper_status.width = .05
    node.current_gripper_status.force = 0.
    assert PiperMasterControl._wait_for_gripper_release(node, .04, timeout_sec=.001)


def test_signed_force_contact_and_release(monkeypatch):
    """Negative gripping force confirms contact but must not authorize release."""
    import time
    import master_node.piper_master_control as module
    node = SimpleNamespace(current_gripper_status=SimpleNamespace(width=.0056, force=-1.043),
                           gripper_command_time=time.monotonic()-1.,
                           gripper_feedback_time=time.monotonic())
    node.get_logger = lambda: SimpleNamespace(error=lambda _: None)
    monkeypatch.setattr(module.rclpy, 'ok', lambda: True)
    original_sleep = time.sleep

    def fresh_sample(seconds):
        original_sleep(seconds)
        node.gripper_feedback_time = time.monotonic()

    monkeypatch.setattr(module.time, 'sleep', fresh_sample)
    assert PiperMasterControl._wait_for_gripper_contact(node, timeout_sec=.4)
    node.current_gripper_status.width = .05
    assert not PiperMasterControl._wait_for_gripper_release(node, .0056, timeout_sec=.001)
    for force in (float('nan'), float('inf'), -float('inf'), -.1):
        node.current_gripper_status.force = force
        assert not PiperMasterControl._wait_for_gripper_contact(node, timeout_sec=.001)
