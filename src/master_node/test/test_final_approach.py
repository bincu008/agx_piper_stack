from master_node.piper_master_control import PiperMasterControl


class _FinalApproachTarget:
    def __init__(self):
        self.state = 'IDLE'
        self.physical_collision_detected = False
        self.collision_detected = False
        self.gripper_open_force_n = 1.0
        self.events = []
        self.speed_results = {5: True, 20: True}
        self.open_feedback = True
        self.contact_feedback = True

    def _command_gripper(self, width, force):
        self.events.append(('gripper', width, force))
        return True

    def _wait_for_gripper_opening(self, width):
        self.events.append(('open_feedback', width))
        return self.open_feedback

    def _wait_for_gripper_contact(self):
        self.events.append(('contact_feedback',))
        return self.contact_feedback

    def set_arm_speed(self, speed):
        self.events.append(('speed', speed))
        return self.speed_results[speed]

    def move_arm_to(self, waypoint, disable_collision_monitor, tolerance):
        self.events.append(('move', list(waypoint), disable_collision_monitor,
                            tolerance))
        return True

    def _execute_validated_final_approach(self, path):
        return PiperMasterControl._execute_validated_final_approach(self, path)


def test_final_approach_command_ordering_and_contact_feedback():
    target = _FinalApproachTarget()

    result = PiperMasterControl._execute_refined_final_approach(
        target, [[0.0, 0.8, -1.4, 0.0, 0.0, 0.0]], 0.04)

    assert result == (
        True, True, True, 'FINAL_APPROACH_COMPLETE_CONTACT_CONFIRMED')
    assert [event[0] for event in target.events] == [
        'gripper', 'open_feedback', 'speed', 'move', 'gripper',
        'contact_feedback', 'speed']
    assert target.events[0] == ('gripper', 0.05, 1.0)
    assert target.events[2] == ('speed', 5)
    assert target.events[4] == ('gripper', 0.0, 1.0)
    assert target.events[-1] == ('speed', 20)


def test_final_approach_does_not_move_when_open_feedback_fails():
    target = _FinalApproachTarget()
    target.open_feedback = False

    result = PiperMasterControl._execute_refined_final_approach(
        target, [[0.0, 0.8, -1.4, 0.0, 0.0, 0.0]], 0.04)

    assert result[0] is False
    assert result[1] is False
    assert result[2] is False
    assert 'OPEN_FEEDBACK_FAILED' in result[3]
    assert [event[0] for event in target.events] == [
        'gripper', 'open_feedback', 'speed']
    assert target.events[-1] == ('speed', 20)


def test_final_approach_does_not_move_when_slow_speed_fails():
    target = _FinalApproachTarget()
    target.speed_results[5] = False

    result = PiperMasterControl._execute_refined_final_approach(
        target, [[0.0, 0.8, -1.4, 0.0, 0.0, 0.0]], 0.04)

    assert result[0] is False
    assert result[1] is False
    assert result[2] is False
    assert 'SPEED_5_PERCENT_FAILED' in result[3]
    assert [event[0] for event in target.events] == [
        'gripper', 'open_feedback', 'speed', 'speed']
    assert target.events[-1] == ('speed', 20)


def test_final_approach_reports_unconfirmed_contact_after_close():
    target = _FinalApproachTarget()
    target.contact_feedback = False

    result = PiperMasterControl._execute_refined_final_approach(
        target, [[0.0, 0.8, -1.4, 0.0, 0.0, 0.0]], 0.04)

    assert result[0] is False
    assert result[1] is True
    assert result[2] is False
    assert 'GRIPPER_CONTACT_UNCONFIRMED' in result[3]
    assert target.events[-2] == ('contact_feedback',)
    assert target.events[-1] == ('speed', 20)


def test_validated_final_approach_rejects_empty_path_without_motion():
    target = _FinalApproachTarget()

    result = PiperMasterControl._execute_validated_final_approach(target, [])

    assert result == (False, 'FINAL_APPROACH_EMPTY_VALIDATED_PATH')
    assert target.events == []


def test_validated_final_approach_stops_before_next_waypoint_on_collision():
    target = _FinalApproachTarget()
    target.collision_detected = True

    result = PiperMasterControl._execute_validated_final_approach(
        target, [[0.0, 0.8, -1.4, 0.0, 0.0, 0.0]])

    assert result == (False, 'FINAL_APPROACH_COLLISION_BEFORE_WAYPOINT_1')
    assert target.events == []
