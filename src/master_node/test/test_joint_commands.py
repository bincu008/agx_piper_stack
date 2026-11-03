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


def test_offset_applies_strictly_to_joint6_once():
    from master_node.calibration import driver_joint_target
    nominal = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    sent = driver_joint_target(nominal, 0.05)
    assert sent[:5] == nominal[:5]
    assert abs(sent[5] - (-0.55)) < 1.e-6


def test_invalid_nominal_target_rejected():
    import pytest
    from master_node.calibration import driver_joint_target
    for bad in ([0.0] * 5, [0.0] * 7, [0.0, float('nan'), 0.0, 0.0, 0.0, 0.0]):
        with pytest.raises(ValueError):
            driver_joint_target(bad, 0.0)


def test_load_command_offset_requires_finite_number(tmp_path):
    import json
    import pytest
    from master_node.calibration import load_command_offset
    path = tmp_path / 'cal.json'
    path.write_text(json.dumps({'joint6_command_offset_rad': -0.035}))
    assert abs(load_command_offset(str(path)) - (-0.035)) < 1.e-6
    for bad in ({}, {'joint6_command_offset_rad': '0.1'}, {'joint6_command_offset_rad': True}):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_command_offset(str(path))


def test_feedback_reference_matches_nominal_and_sent():
    from master_node.calibration import feedback_reference
    node = SimpleNamespace(_last_nominal_arm_command=[0.0] * 6,
                           _last_sent_arm_command=[0.0, 0.0, 0.0, 0.0, 0.0, 0.05])
    assert feedback_reference(node, [0.0] * 6) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.05]
    assert feedback_reference(node, [0.1] * 6) == [0.1] * 6


def test_command_pose_reference_rotates_tool_yaw_without_mutating_source():
    from geometry_msgs.msg import PoseStamped
    from scipy.spatial.transform import Rotation
    from master_node.calibration import command_pose_reference
    source = PoseStamped()
    source.pose.orientation.w = 1.0
    corrected = command_pose_reference(source, 0.1)
    assert source.pose.orientation.z == 0.0
    rot = Rotation.from_quat([corrected.pose.orientation.x, corrected.pose.orientation.y,
                              corrected.pose.orientation.z, corrected.pose.orientation.w])
    angle = rot.as_rotvec()
    assert abs(angle[2] - 0.1) < 1.e-6


def test_tilted_circle_recovers_uniform_ratio_and_preserves_axis_uncertainty():
    import numpy as np
    import pytest
    from master_node.calibration import measure_plate
    yy, xx = np.mgrid[:320, :400]
    rays = np.stack(((xx - 200) / 500., (yy - 160) / 500., np.ones_like(xx)), axis=2)
    normal = np.array([0., .4, .916515])
    center = np.array([0., 0., .5])
    points = rays * ((normal @ center) / (rays @ normal))[:, :, None]
    mask = np.linalg.norm(points - center, axis=2) <= .087
    report = measure_plate(points * 1.05 / 1.036, mask, .174, 1.036)
    assert report['perception_measurement_ratio'] == pytest.approx(1.05, abs=.015)
    assert report['rim_plane_inlier_fraction'] > .98
    assert not report['axis_calibration_independent']


def test_invalid_json_ratio_is_rejected(tmp_path):
    import json
    import pytest
    from master_node.calibration import load_metric_ratio
    path = tmp_path / 'scale.json'
    for ratio in [0., -1., '1.03', True, float('nan')]:
        path.write_text(json.dumps({'perception_measurement_ratio': ratio}))
        with pytest.raises(ValueError):
            load_metric_ratio(str(path))


def test_valid_json_ratio(tmp_path):
    import json
    from master_node.calibration import load_metric_ratio
    path = tmp_path / 'scale.json'
    path.write_text(json.dumps({'perception_measurement_ratio': .98}))
    assert load_metric_ratio(str(path)) == .98


def test_grasp_offset_uses_explicit_base_frame_and_metres(tmp_path):
    import json
    import pytest
    from master_node.calibration import load_grasp_offset
    path = tmp_path / 'offset.json'
    path.write_text(json.dumps({'frame_id': 'base_link',
                                'grasp_position_offset_base_m': [-.005, .005, .005]}))
    assert load_grasp_offset(str(path)) == [-.005, .005, .005]
    path.write_text(json.dumps({'frame_id': 'tool',
                                'grasp_position_offset_base_m': [-.005, .005, .005]}))
    with pytest.raises(ValueError):
        load_grasp_offset(str(path))
