"""Explicit contact intent is shared by image and layout recipes."""
from task_pipeline.grasp_input import affordance_from_description


def test_contact_intent():
    assert affordance_from_description('white bowl') == 'AUTO'
    assert affordance_from_description('upper RIM of bowl') == 'RIM'
    assert affordance_from_description('outer edge of plate') == 'RIM'
    assert affordance_from_description('cup handle') == 'HANDLE'
    assert affordance_from_description('cup body') == 'BODY'


def test_empty_auto_bowl_gets_one_rim_recovery_on_same_observation():
    from my_actions.srv import FindGraspCandidates
    from task_pipeline.grasp_input import calculate_grasps_with_recovery
    query = FindGraspCandidates.Request(target_description='white bowl', affordance_type='AUTO')
    query.observation_header.stamp.sec = 42
    calls = []

    def backend(request):
        calls.append(request)
        return FindGraspCandidates.Response(success=request.affordance_type == 'RIM',
                                           status='NO_VALID_GEOMETRIC_GRASP')

    result = calculate_grasps_with_recovery(query, backend)
    assert result.success and 'AUTO_TO_RIM_RECOVERY' in result.status
    assert len(calls) == 2 and calls[1].observation_header == query.observation_header
    assert calls[1].target_description == query.target_description
    assert query.affordance_type == 'AUTO'


def test_recovery_does_not_loop_or_retry_invalid_inputs():
    from my_actions.srv import FindGraspCandidates
    from task_pipeline.grasp_input import calculate_grasps_with_recovery
    for intent, status, expected in [('RIM', 'NO_VALID_GEOMETRIC_GRASP', 1),
                                     ('AUTO', 'OBSERVATION_MISMATCH', 1),
                                     ('AUTO', 'NO_VALID_GEOMETRIC_GRASP', 2)]:
        calls = []

        def backend(request):
            calls.append(request)
            return FindGraspCandidates.Response(success=False, status=status)

        result = calculate_grasps_with_recovery(FindGraspCandidates.Request(
            target_description='white bowl', affordance_type=intent), backend)
        assert not result.success and len(calls) == expected
