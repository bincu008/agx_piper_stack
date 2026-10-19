"""Bounded calculation recovery; actuator and input failures are never retried."""

import time


def pin_observation_with_recovery(request, call, wait=time.sleep):
    """Wait briefly for delivery of the exact captured frame, without recapture."""
    response = call(request)
    for delay in (0.05, 0.10, 0.20, 0.40):
        if response.success or response.status != 'LAYOUT_OBSERVATION_PENDING':
            break
        wait(delay)
        response = call(request)
    return response


def retry_preparation(result, attempt, maximum_attempts):
    """Request fresh sensor inputs only after a definite geometric rejection."""
    if result.success or attempt >= maximum_attempts or not result.results:
        return False
    failed = result.results[-1]
    if failed.function_name == 'calculate_grasps':
        return (failed.status.startswith('NO_VALID_GEOMETRIC_GRASP') or
                failed.status.startswith('AUTO_FAILED: NO_VALID_GEOMETRIC_GRASP'))
    return (failed.function_name == 'prepare_grasp_lift' and
            failed.status.startswith('NO_COMPLETE_GRASP_LIFT_PLAN:'))
