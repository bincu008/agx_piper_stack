"""Exercise shared validation, bounded dispatch, and capture association."""

import threading
from types import SimpleNamespace

from geometry_msgs.msg import TransformStamped

from my_actions.msg import GraspCandidate, PlacementCandidate
from my_actions.srv import CaptureObservation, PlanPlacement

import pytest

import rclpy

from sensor_msgs.msg import Image, PointCloud2

from task_pipeline.metrics import summarize
from task_pipeline.node import PipelineNode
from task_pipeline.placement import search_placements, validate_placements
from task_pipeline.registry import FUNCTIONS, catalog


def request(grasp_ids=(1, 2)):
    """Request."""
    query = PlanPlacement.Request()
    query.grasp_candidate = GraspCandidate(id=grasp_ids[0])
    query.alternative_grasps = [GraspCandidate(id=i) for i in grasp_ids[1:]]
    query.candidates = [PlacementCandidate(id=i) for i in (1, 2, 3)]
    query.retry_motion_failures = True
    query.validation_batch_size = 2
    query.plan_pick_place_motion = True
    return query


@pytest.mark.parametrize('failure,ids,expected', [
    ('LIFT_IK_FAILED_AT_WAYPOINT_2', (1, 2), [(1, [1, 2]), (2, [1, 2])]),
    ('TRANSFER_RRTC_NO_PATH', (1,), [(1, [1, 2]), (1, [2])]),
])
def test_retries_preserve_grasp_goal_combinations(failure, ids, expected):
    """Test retries preserve grasp goal combinations."""
    query = request(ids)
    calls = []

    def backend(item):
        calls.append((item.grasp_candidate.id, [c.id for c in item.candidates]))
        assert not item.retry_motion_failures and item.validation_batch_size == 0
        result = PlanPlacement.Response()
        result.selected_grasp_candidate = item.grasp_candidate
        result.selected_candidate = item.candidates[0]
        result.success = result.motion_success = len(calls) > 1
        result.motion_status = 'LIFT_TRANSFER_DESCENT_VALIDATED' if result.success else failure
        return result

    result = validate_placements(query, backend)
    assert calls == expected
    assert result.success and result.motion_success
    assert [c.id for c in query.candidates] == [1, 2, 3]
    assert query.retry_motion_failures
    assert result.validated_candidate_count == sum(len(c) for _, c in expected)


def test_wrong_response_association_is_rejected_without_retry_loop():
    """Test wrong response association is rejected without retry loop."""
    query = request()

    def backend(item):
        return PlanPlacement.Response(success=True, motion_success=True,
                                      selected_grasp_candidate=GraspCandidate(id=999),
                                      selected_candidate=PlacementCandidate(id=999))

    result = validate_placements(query, backend)
    assert not result.success
    assert result.status == 'PLACEMENT_RESPONSE_ASSOCIATION_MISMATCH'
    assert result.validated_candidate_count == 3


def test_final_ik_rejection_preserves_earlier_lift_failure():
    """Keep the actual motion blocker when later alternatives fail IK."""
    calls = []

    def backend(item):
        calls.append(item)
        if len(calls) == 1:
            return PlanPlacement.Response(
                status='LIFT_INITIAL_HELD_STATE_COLLISION:ATTACHED_OBJECT',
                motion_status='LIFT_INITIAL_HELD_STATE_COLLISION:ATTACHED_OBJECT',
                selected_grasp_candidate=item.grasp_candidate,
                selected_candidate=item.candidates[0])
        return PlanPlacement.Response(status='PLACEMENT_NO_REACHABLE_CANDIDATE')

    result = validate_placements(request(), backend)
    assert not result.success
    assert result.status.startswith('PLACEMENT_NO_REACHABLE_CANDIDATE')
    assert 'earlier_motion_failures=LIFT_INITIAL_HELD_STATE_COLLISION' in result.status


def test_motion_success_required_for_full_motion_scope():
    """Test motion success required for full motion scope."""
    query = request()
    result = validate_placements(query, lambda item: PlanPlacement.Response(
        success=True, selected_grasp_candidate=item.grasp_candidate,
        selected_candidate=item.candidates[0]))
    assert not result.success
    assert result.status == 'PLACEMENT_MOTION_NOT_VALIDATED'


def test_cancel_before_backend_never_submits_request():
    """Test cancel before backend never submits request."""
    submitted = []

    def cancel():
        raise RuntimeError('CANCELLED')

    with pytest.raises(RuntimeError, match='CANCELLED'):
        validate_placements(request(), submitted.append, cancel)
    assert not submitted


def test_search_reports_failure_timing():
    """Test search reports failure timing."""
    samples = []

    def search():
        raise ValueError('invalid geometry')

    with pytest.raises(ValueError):
        search_placements(search, timing_sink=lambda *args: samples.append(args))
    assert len(samples) == 1
    assert samples[0][0] >= 0 and samples[0][1:] == (False, 'SEARCH_FAILED')


def test_busy_dispatch_is_rejected_and_measured():
    """Test busy dispatch is rejected and measured."""
    samples = []
    node = SimpleNamespace(_active=threading.BoundedSemaphore(0),
                           _publish=lambda *args: samples.append(args))
    response = PipelineNode._dispatch(node, 'plan_placement', request(), PlanPlacement.Response())
    assert not response.success and response.status.startswith('PIPELINE_BUSY')
    assert len(samples) == 1


def test_colocated_dispatch_uses_master_store_without_rpc():
    """The public typed API must keep master execution while eliminating the proxy hop."""
    calls, samples = [], []
    runtime = SimpleNamespace(run_typed=lambda name, query: (
        calls.append((name, query)), (PlanPlacement.Response(success=True), (True, 'OK')))[1])
    node = SimpleNamespace(_active=threading.BoundedSemaphore(1), _runtime=runtime,
                           _publish=lambda *args: samples.append(args),
                           _rpc=lambda *args: pytest.fail('unnecessary ROS proxy hop'))
    query = request()
    response = PipelineNode._dispatch(node, 'plan_placement', query, PlanPlacement.Response())
    assert response.success and calls == [('plan_placement', query)]
    assert len(samples) == 1 and samples[0][-1] == (True, 'OK')


@pytest.mark.parametrize('matching', [True, False])
def test_capture_requires_matching_manual_observation(matching):
    """Test capture requires matching manual observation."""
    image, cloud, transform = Image(), PointCloud2(), TransformStamped()
    image.header.stamp.sec = cloud.header.stamp.sec = 10
    transform.header.stamp.sec = 10 if matching else 11
    cloud.header.frame_id = transform.header.frame_id = 'base_link'
    observations = {'image': [], 'cloud': [], 'transform': []}
    samples = []

    def trigger(*args):
        observations['image'].append(image)
        observations['cloud'].append(cloud)
        observations['transform'].append(transform)
        return SimpleNamespace(success=True)

    node = SimpleNamespace(_capture_lock=threading.Lock(), _lock=threading.Lock(),
                           _observations=observations, _scan=None, _rpc=trigger,
                           _stamp=PipelineNode._stamp,
                           get_parameter=lambda _: SimpleNamespace(value=.03),
                           _publish=lambda *args: samples.append(args))
    import rclpy
    from unittest.mock import patch
    with patch.object(rclpy, 'ok', return_value=True):
        response = PipelineNode._capture(node, None, CaptureObservation.Response())
    assert response.success == matching
    if matching:
        assert response.image.header.stamp == response.cloud.header.stamp
        assert response.camera_transform.header.stamp == response.cloud.header.stamp
    else:
        assert response.status == 'SYNCHRONIZED_CAPTURE_TIMEOUT'
    assert len(samples) == 1


def test_catalog_distinguishes_known_cloud_from_hggd_input():
    """Test catalog distinguishes known cloud from hggd input."""
    functions = {item['name']: item for item in catalog()}
    assert functions['calculate_grasps']['input_mode'] == 'rgbd'
    assert functions['calculate_cloud_grasps']['input_mode'] == 'known_cloud'
    assert functions['execute_grasp']['backend'] == '/master/execute_grasp'


def test_real_node_constructs_without_clobbering_rclpy_clients():
    """Construct the dispatcher with real ROS clients and no actuator commands."""
    rclpy.init()
    node = None
    try:
        node = PipelineNode()
        assert set(node._backend_clients) == {
            'run_step', 'submit_steps', 'get_step_context', 'clear_step_context',
            'store_step_result', 'append_task', 'execute_pipeline', 'flush_pipeline'}
        assert isinstance(node._clients, list)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_metrics_keep_input_modes_and_nested_scopes_separate():
    """Do not combine fixture grasps, HGGD or nested validation totals."""
    samples = [SimpleNamespace(
        function_name='plan_placement', input_mode=mode, scope=scope,
        full_motion=motion, wall_seconds=seconds, success=success)
        for mode, scope, motion, seconds, success in [
            ('frozen_scene', 'api_inclusive', True, 2.0, True),
            ('frozen_scene', 'api_inclusive', True, 4.0, False),
            ('frozen_scene', 'orchestration_inclusive', True, 6.0, True),
            ('legacy_pose_only', 'api_inclusive', False, 1.0, True),
        ]]
    result = summarize(samples)
    assert len(result) == 3
    full_api = next(row for row in result if row['full_motion'] and
                    row['scope'] == 'api_inclusive')
    assert full_api['calls'] == 2 and full_api['failed_calls'] == 1
    assert full_api['median_seconds'] == 3.0


def test_achieved_slot_rejection_tries_next_goal_without_actuation():
    """A full motion result outside its reserved cells must not end the search."""
    query = request((1,))
    calls = []

    def backend(item):
        calls.append(item.candidates[0].id)
        return PlanPlacement.Response(success=True, motion_success=True,
                                      selected_grasp_candidate=item.grasp_candidate,
                                      selected_candidate=item.candidates[0])

    def slot_check(result):
        return SimpleNamespace(success=result.selected_candidate.id != 1,
                               status='FLOOR_SLOT_SOURCE_OR_POSE_MISMATCH')

    result = validate_placements(query, backend, validate_selected=slot_check)
    assert result.success and calls == [1, 2]
    assert result.selected_candidate.id == 2


def test_blocked_transfer_start_retries_bounded_higher_lift():
    """Require a fully checked transfer handoff without actuator retries."""
    query = request((1,))
    query.lift_distance_m = .05
    heights = []

    def backend(item):
        heights.append(item.lift_distance_m)
        success = item.lift_distance_m > .055
        return PlanPlacement.Response(success=success, motion_success=success,
                                      selected_grasp_candidate=item.grasp_candidate,
                                      selected_candidate=item.candidates[0],
                                      motion_status='OK' if success else
                                      'TRANSFER_LIFT_ENDPOINT_REJECTED:ATTACHED_OBJECT')

    result = validate_placements(query, backend)
    assert result.success and heights == pytest.approx([.05, .06])
    assert query.lift_distance_m == pytest.approx(.05)


def test_transfer_start_recovery_is_bounded():
    """Exhaust clearance alternatives without relaxing collision checks."""
    query = request((1,))
    query.candidates = query.candidates[:1]
    query.lift_distance_m = .05
    heights = []

    def backend(item):
        heights.append(item.lift_distance_m)
        return PlanPlacement.Response(selected_grasp_candidate=item.grasp_candidate,
                                      selected_candidate=item.candidates[0],
                                      motion_status='TRANSFER_LIFT_ENDPOINT_REJECTED:ATTACHED_OBJECT')

    result = validate_placements(query, backend)
    assert not result.success
    assert heights == pytest.approx([.05, .06, .07])


@pytest.mark.parametrize('height', [.001, .15, .30])
def test_lift_recipes_accept_positive_heights_without_ten_cm_cap(height):
    """Both public recipes preserve requests outside the former fixed range."""
    from rclpy.serialization import deserialize_message
    from task_pipeline.recipes import prepare_grasp_lift, prepare_pick_place
    from my_actions.srv import PreparedGraspLift, PreparePickPlace
    for build, schema in ((prepare_grasp_lift, PreparedGraspLift),
                          (prepare_pick_place, PreparePickPlace)):
        steps = build('bowl', lift_distance_m=height)
        motion = next(s for s in steps if s.step_id == 'prepare_motion')
        request_data = deserialize_message(bytes(motion.request_cdr), schema.Request)
        assert request_data.lift_distance_m == pytest.approx(height)


@pytest.mark.parametrize('height', [-.1, 0., float('nan'), float('inf')])
def test_lift_recipes_reject_invalid_physical_distances(height):
    """Removing the height ceiling must not admit invalid numeric requests."""
    from task_pipeline.recipes import prepare_grasp_lift, prepare_pick_place
    for build in (prepare_grasp_lift, prepare_pick_place):
        with pytest.raises(ValueError, match='finite and positive'):
            build('bowl', lift_distance_m=height)


def test_transfer_clearance_recovery_also_works_above_ten_cm():
    """Recovery stays bounded relative to the requested height, without clipping."""
    query = request((1,))
    query.candidates = query.candidates[:1]
    query.lift_distance_m = .15
    heights = []

    def backend(item):
        heights.append(item.lift_distance_m)
        success = item.lift_distance_m > .155
        return PlanPlacement.Response(success=success, motion_success=success,
                                      selected_grasp_candidate=item.grasp_candidate,
                                      selected_candidate=item.candidates[0],
                                      motion_status='OK' if success else
                                      'TRANSFER_LIFT_ENDPOINT_REJECTED:ATTACHED_OBJECT')

    assert validate_placements(query, backend).success
    assert heights == pytest.approx([.15, .16])


def test_active_recipe_uses_shared_memory_and_valid_predecessors():
    from task_pipeline.recipes import with_active_memory, prepare_pick_place
    from task_pipeline.context import get_field
    recipe = with_active_memory(prepare_pick_place('bowl','tray'),'bowl','tray')
    seen = {}
    for item in recipe:
        schema = FUNCTIONS[item.function_name].service_type
        for binding in item.bindings:
            assert binding.source_step_id in seen
            if binding.source_field != '@response_cdr':
                get_field(seen[binding.source_step_id].Response(),binding.source_field)
        seen[item.step_id] = schema
    assert next(s for s in recipe if s.step_id=='segment_cup').function_name=='refresh_active_target'
    assert next(s for s in recipe if s.step_id=='placements').function_name=='find_active_placements'
    assert all(s.function_name!='ground_object' for s in recipe)


def test_in_process_proxy_does_not_duplicate_master_parameter_services():
    """Only the actual master answers parameter reads under launch name remapping."""
    rclpy.init(args=['--ros-args', '-r', '__node:=piper_master_control'])
    node = None
    try:
        node = PipelineNode(runtime=SimpleNamespace())
        assert node.get_name() == 'piper_master_control'
        names = {service.srv_name for service in node.services}
        assert '/pipeline/list_functions' in names
        assert '/piper_master_control/get_parameters' not in names
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
