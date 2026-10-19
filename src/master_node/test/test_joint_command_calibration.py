"""Verify command-only calibration without altering perception or feedback."""

import json

from geometry_msgs.msg import PoseStamped
import pytest
from sensor_msgs.msg import JointState

from master_node.joint_command_calibration import (
    command_pose_reference, driver_joint_target, feedback_reference, load_command_offset,
)
from master_node.piper_master_control import PiperMasterControl
from test_joint_commands import _CommandTarget, _publish


OFFSET = .23499113048851653


def test_final_publication_adds_offset_once_without_mutating_nominal_or_gripper():
    """Every publication starts from nominal targets, not the previously shifted command."""
    target = _CommandTarget()
    target.joint6_command_offset_rad = OFFSET
    nominal = [0., .1, -.2, .3, -.4, .5]
    for _ in range(2):
        assert _publish(target, arm_positions=nominal, gripper_width=.03, gripper_force=1.)
        sent = target.joint_pub.messages[-1]
        assert list(sent.position[:5]) == nominal[:5]
        assert sent.position[5] == .5+OFFSET
        assert sent.position[6] == .03
    assert nominal[5] == .5
    assert feedback_reference(target, nominal)[5] == .5+OFFSET
    raw = list(target._last_sent_arm_command)
    assert feedback_reference(target, raw) == raw
    assert _publish(target, gripper_width=.02)
    assert target._last_sent_arm_command == raw
    assert list(target.joint_pub.messages[-1].name) == ['gripper']
    assert _publish(target, arm_positions=raw, apply_joint6_offset=False)
    assert list(target.joint_pub.messages[-1].position) == raw


def test_limits_are_checked_before_offset_and_arrival_uses_sent_raw_command(monkeypatch):
    """Clamping nominal joint6 does not clamp the subsequently added command correction."""
    target = _CommandTarget()
    target.joint6_command_offset_rad = OFFSET
    target.joint_limits = [(-1., 1.)]*6
    target.torque_history = [[] for _ in range(6)]
    target.enable_camera_collision_monitor = False
    target.joint_state_check_delay = 0.
    target.collision_monitor_pub = target.joint_pub
    target.logger.debug = target.logger.warn = lambda *_: None
    target.current_joint_states = JointState(position=[0.]*6)
    original = target.joint_pub.publish

    def publish(message):
        original(message)
        if isinstance(message, JointState):
            target.current_joint_states = JointState(position=list(message.position))

    target.joint_pub.publish = publish
    target._publish_joint_state_command = lambda **kw: _publish(target, **kw)
    monkeypatch.setattr('master_node.piper_master_control.time.sleep', lambda _: None)
    assert PiperMasterControl.move_arm_to(target, [0.]*5+[1.2])
    assert target._last_nominal_arm_command[5] == 1.
    assert target._last_sent_arm_command[5] == 1.+OFFSET
    assert target.current_joint_states.position[5] == 1.+OFFSET


def test_endpoint_reference_predicts_joint6_roll_without_rewriting_raw_pose():
    """Pose arrival checks remain active for the rotation of the sent command."""
    nominal = PoseStamped()
    nominal.header.frame_id = 'base_link'
    nominal.pose.position.x = .3
    nominal.pose.orientation.w = 1.
    sent = command_pose_reference(nominal, OFFSET)
    position, rotation = PiperMasterControl._pose_error(sent, sent)
    assert position == pytest.approx(0.) and rotation == pytest.approx(0.)
    assert nominal.pose.orientation.z == 0.
    assert sent.pose.position.x == .3
    assert PiperMasterControl._pose_error(nominal, sent)[1] == pytest.approx(OFFSET)


@pytest.mark.parametrize('value', [None, True, '0.23', float('nan'), float('inf')])
def test_bad_radian_config_is_rejected(tmp_path, value):
    """Reject malformed settings rather than silently publishing unexpected commands."""
    path = tmp_path/'joint.json'
    path.write_text(json.dumps({'joint6_command_offset_rad': value}))
    with pytest.raises(ValueError):
        load_command_offset(path)


def test_load_raw_radians_and_pure_target_conversion(tmp_path):
    """Configuration retains the recorded raw measurement without degree conversion."""
    path = tmp_path/'joint.json'
    path.write_text(json.dumps({'joint6_command_offset_rad': OFFSET}))
    assert load_command_offset(path) == OFFSET
    nominal = [0.]*6
    assert driver_joint_target(nominal, OFFSET) == [0.]*5+[OFFSET]
    assert nominal == [0.]*6


def test_complete_prepared_execution_verifies_raw_shifted_feedback_without_replanning():
    """All prepared stages accept the sent roll while cached nominal joints remain unchanged."""
    from test_prepared_grasp_lift import execute, fixture, prepare
    trial, events, calls = fixture()
    node = trial.node
    node.joint6_command_offset_rad = OFFSET
    # Every sent endpoint exceeds the nominal roll bound, as in the live failure.
    node.joint_limits[5] = (-.01, .01)
    nominal_move = node.move_arm_to

    def move(joints, **kwargs):
        result = nominal_move(joints, **kwargs)
        node._last_nominal_arm_command = list(joints)
        node._last_sent_arm_command = driver_joint_target(joints, OFFSET)
        node.current_joint_states.position = list(node._last_sent_arm_command)
        node.current_tcp_pose = command_pose_reference(node.current_tcp_pose, OFFSET)
        return result

    node.move_arm_to = move
    node._pose_error = PiperMasterControl._pose_error
    assert prepare(trial).success
    count = len(calls)
    response = execute(trial)
    assert response.success, response.status
    assert trial.record['phase'] == 'LIFTED'
    assert node.current_joint_states.position[5] == pytest.approx(.003+OFFSET)
    assert trial.record['lift_path'][-1][5] == pytest.approx(.003)
    assert len(calls) == count


def test_raw_feedback_limit_accepts_command_only_offset_but_nominal_limit_does_not():
    """Reproduce pre-grasp rejection without relaxing planner or other joints."""
    from types import SimpleNamespace
    from master_node.execution_api import ExecutionAPI
    # Access the validator without constructing services or actuators.
    node = SimpleNamespace(joint_limits=[(-2.129302, 2.129302)]*6,
                           joint6_command_offset_rad=OFFSET)
    validator = SimpleNamespace(node=node)
    actual = [0.]*5+[2.264093]
    ExecutionAPI._joints(validator, actual, command_feedback=True)
    assert actual[5] == 2.264093
    with pytest.raises(ValueError, match='joint6'):
        ExecutionAPI._joints(validator, actual)
    with pytest.raises(ValueError, match='joint6'):
        ExecutionAPI._joints(validator, [0.]*5+[2.5], command_feedback=True)
    with pytest.raises(ValueError, match='joint1'):
        ExecutionAPI._joints(validator, [2.264093]+[0.]*5, command_feedback=True)
    node.joint6_command_offset_rad = -OFFSET
    ExecutionAPI._joints(validator, [0.]*5+[-2.264093], command_feedback=True)
