"""Recovery never turns an actuator failure into an automatic retry."""
from types import SimpleNamespace

from task_pipeline.preparation_recovery import pin_observation_with_recovery
from task_pipeline.preparation_recovery import retry_preparation


def result(function, status, success=False):
    """Build a completed action result without ROS transport."""
    return SimpleNamespace(success=success, results=[SimpleNamespace(
        function_name=function, status=status)])


def test_geometric_failure_is_bounded():
    """Retry only within the configured calculation attempt limit."""
    failed = result('calculate_grasps', 'NO_VALID_GEOMETRIC_GRASP')
    assert retry_preparation(failed, 1, 3)
    assert not retry_preparation(failed, 3, 3)
    failed_lift = result('prepare_grasp_lift', 'NO_COMPLETE_GRASP_LIFT_PLAN: lift IK')
    assert retry_preparation(failed_lift, 1, 3)


def test_actuator_unknown_and_invalid_inputs_are_not_retried():
    """Preserve unknown-state and input-error stop behavior."""
    for function, status in [('execute_prepared_grasp_lift', 'STATE_UNKNOWN'),
                             ('calculate_grasps', 'OBSERVATION_MISMATCH'),
                             ('prepare_grasp_lift', 'CURRENT_JOINT_FEEDBACK: invalid')]:
        assert not retry_preparation(result(function, status), 1, 3)
    passed = result('calculate_grasps', 'NO_VALID_GEOMETRIC_GRASP', True)
    assert not retry_preparation(passed, 1, 3)


def test_pin_waits_only_for_same_frame_and_is_bounded():
    """Pending delivery retries never replace the source observation."""
    request = object()
    calls = []
    waits = []

    def backend(query):
        calls.append(query)
        return SimpleNamespace(success=len(calls) == 3,
                               status='LAYOUT_OBSERVATION_PENDING')

    assert pin_observation_with_recovery(request, backend, waits.append).success
    assert len(calls) == 3 and all(query is request for query in calls)
    assert waits == [0.05, 0.10]
    calls.clear()

    def pending(query):
        return SimpleNamespace(success=False, status='LAYOUT_OBSERVATION_PENDING')

    waits.clear()
    assert not pin_observation_with_recovery(request, pending, waits.append).success
    assert sum(waits) == 0.75
