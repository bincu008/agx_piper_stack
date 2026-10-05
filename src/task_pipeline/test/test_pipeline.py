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
from task_pipeline.registry import catalog


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
        assert len(node._backend_clients) > 20
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
