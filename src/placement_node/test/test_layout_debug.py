"""Check that ordered layout visualization retains both accepted objects."""

from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from placement_node.node import PlacementNode


def test_active_mask_floor_does_not_disable_other_or_placed_object_protection():
    from placement_node.perception import active_source_support_labels
    active = np.array([[True, True, False, False]])
    neighbor = np.array([[False, True, True, False]])
    # Pixel 0 is an active-mask floor return rejected by grasp segmentation;
    # pixel 1 overlaps another mask. Appended geometry is already placed.
    labels = np.array([3, 3, 3, 1, 3], dtype=np.uint8)
    result = active_source_support_labels(labels, [active, neighbor], 0)
    np.testing.assert_array_equal(result, [0, 3, 3, 1, 3])
    np.testing.assert_array_equal(labels, [3, 3, 3, 1, 3])


def test_stack_support_labels_preserve_bodies_and_revoke_occluded_tops():
    from placement_node.perception import exposed_stack_support_labels
    # Same XY: table, placed body, placed top, then an unplaced blocking object.
    points = np.array([[0, 0, 0], [0, 0, .02], [0, 0, .08],
                       [.02, 0, .08], [.02, 0, .12], [np.nan, 0, 0]])
    labels = np.array([1, 3, 2, 2, 3, 3], dtype=np.uint8)
    result = exposed_stack_support_labels(points, labels, labels == 2)
    np.testing.assert_array_equal(result, [1, 3, 2, 3, 3, 3])
    np.testing.assert_array_equal(labels, [1, 3, 2, 2, 3, 3])


def test_new_placed_cloud_exposes_tops_on_multiple_surfaces():
    from placement_node.perception import exposed_stack_support_labels
    points = np.array([[0, 0, .01], [0, 0, .08], [.02, 0, .01], [.02, 0, .12]])
    np.testing.assert_array_equal(exposed_stack_support_labels(
        points, np.full(4, 3), np.ones(4, dtype=bool)), [3, 2, 3, 2])


def test_ordered_layout_markers_keep_frame_order_and_both_objects():
    published = []
    node = SimpleNamespace(
        _marker_pub=SimpleNamespace(publish=published.append),
        _point=PlacementNode._point)
    header = Header(frame_id='base_link')
    header.stamp.sec = 123
    cup = np.array([[0.1, 0.2, 0.01], [0.1, 0.2, 0.08]], dtype=np.float32)
    plate = np.array([[0.2, 0.2, 0.01], [0.3, 0.2, 0.02]], dtype=np.float32)
    candidate = SimpleNamespace(translation=np.array([0.2, 0.2, 0.01]))
    PlacementNode._publish_layout_markers(
        node, header, [('cup', cup, candidate), ('plate', plate, candidate)])
    markers = published[0].markers
    assert markers[0].action == Marker.DELETEALL
    assert [marker.text for marker in markers if marker.type == Marker.TEXT_VIEW_FACING] == (
        ['#1 cup', '#2 plate'])
    clouds = [marker for marker in markers if marker.type == Marker.POINTS]
    assert len(clouds) == 2
    assert clouds[0].color != clouds[1].color
    assert len({marker.id for marker in markers[1:]}) == 4
    for marker in markers[1:]:
        assert marker.header == header
    assert abs(clouds[0].points[1].z - 0.08) < 1e-6


def test_three_object_layout_keeps_each_actual_placement_gripper():
    from geometry_msgs.msg import Pose, PoseStamped, Vector3
    published = []
    node = SimpleNamespace(
        _marker_pub=SimpleNamespace(publish=published.append),
        _point=PlacementNode._point)
    objects = []
    for index, name in enumerate(['cup', 'plate', 'apple']):
        pose = Pose()
        pose.orientation.w = 1.0
        pose.position.x = index * 0.1
        plan = SimpleNamespace(
            placement_finger_poses=[pose, pose],
            placement_finger_dimensions=Vector3(x=0.01, y=0.02, z=0.07),
            placement_ee_pose=PoseStamped(pose=pose),
            preplace_ee_pose=PoseStamped(pose=pose))
        candidate = SimpleNamespace(translation=np.array([index*0.1, 0, 0]))
        objects.append((name, np.array([[index*0.1, 0, 0.01]]), candidate, plan))
    PlacementNode._publish_layout_markers(node, Header(frame_id='base_link'), objects)
    markers = published[0].markers
    fingers = [item for item in markers if item.type == Marker.CUBE]
    assert len(fingers) == 6
    assert [item.pose.position.x for item in fingers] == [0, 0, 0.1, 0.1, 0.2, 0.2]
    assert len({(item.ns, item.id) for item in markers[1:]}) == len(markers)-1
    assert [item.text for item in markers if item.type == Marker.TEXT_VIEW_FACING] == [
        '#1 cup', '#2 plate', '#3 apple']


def test_rigid_attachment_preserved_with_rotated_placement():
    from placement_node.transforms import attached_object_delta, pose_matrix
    grasp = pose_matrix([0.2, 0.1, 0.3], [0, 0, 0, 1])
    placed = pose_matrix([0.4, -0.1, 0.2], [0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    source_point = np.array([0.25, 0.12, 0.31, 1])
    delta = attached_object_delta(grasp, placed)
    np.testing.assert_allclose(
        np.linalg.inv(placed) @ delta @ source_point,
        np.linalg.inv(grasp) @ source_point, atol=1e-12)


@pytest.mark.parametrize('names, expected_intents', [
    (['cup', 'plate'], ['AUTO', 'AUTO']),
    (['cup body', 'plate rim'], ['BODY', 'RIM']),
])
def test_template_continues_after_failed_grasp_for_each_object(names, expected_intents):
    import threading
    from cv_bridge import CvBridge
    from sensor_msgs_py import point_cloud2
    from my_actions.srv import PlanObjectLayout

    bridge = CvBridge()
    header = Header(frame_id='base_link')
    header.stamp.sec = 123
    yy, xx = np.mgrid[:20, :20]
    xyz = np.stack((xx*.002, yy*.002, np.zeros_like(xx)), axis=-1).astype(np.float32)
    cloud = point_cloud2.create_cloud_xyz32(header, xyz.reshape(-1, 3).tolist())
    cloud.height, cloud.width, cloud.row_step = 20, 20, 20*12
    masks = []
    for start in [0, 10]:
        mask = np.zeros((20, 20), np.uint8)
        mask[:, start:start+10] = 255
        masks.append(bridge.cv2_to_imgmsg(mask, 'mono8'))
    request = PlanObjectLayout.Request(
        observation_header=header, object_names=names, object_masks=masks,
        destination_name='tray', orientation_policy='upright',
        destination_mask=bridge.cv2_to_imgmsg(np.ones((20, 20), np.uint8)*255, 'mono8'))
    calls, logs, affordances, collision_inputs = [], [], [], []

    def call(request):
        calls.append(request.target_description)
        affordances.append(request.affordance_type)
        collision_inputs.append(request.collision_scene)
        return SimpleNamespace(result=lambda: SimpleNamespace(
            success=False, candidates=[], status='test no grasp'))

    client = SimpleNamespace(wait_for_service=lambda **k: True, call_async=call)
    pub = SimpleNamespace(publish=lambda _: None)
    params = dict(layout_debug_directory='', layout_support_mode='auto',
                  layout_debug_stop_on_failure=False,
                  minimum_destination_points=80,
                  request_timeout_sec=1.)
    node = SimpleNamespace(
        _lock=threading.Lock(), _observation_clouds=[cloud], _pinned_layout_cloud=None,
        _stamp_key=PlacementNode._stamp_key, _stamp_seconds=PlacementNode._stamp_seconds,
        _cloud_xyz=lambda *a, **k: xyz, _bridge=bridge,
        get_parameter=lambda k: SimpleNamespace(value=params[k]),
        get_logger=lambda: SimpleNamespace(info=logs.append, warn=logs.append, error=logs.append),
        _publish_layout_markers=lambda *a: None,
        _destination_pub=pub, _status_pub=pub,
        _grasp_candidates_client=client, _plan_placement_client=client,
        _wait_for_future=lambda *a: True)
    response = PlacementNode._layout_callback(node, request, PlanObjectLayout.Response())
    assert calls == ['layout: ' + name for name in names]
    assert affordances == expected_intents
    assert len(collision_inputs) == len(names)
    for frozen_scene in collision_inputs:
        assert frozen_scene.header == header
        assert frozen_scene.width * frozen_scene.height == xyz.shape[0] * xyz.shape[1]
        np.testing.assert_array_equal(
            point_cloud2.read_points_numpy(frozen_scene, field_names=('x', 'y', 'z')),
            xyz.reshape(-1, 3))
    assert not response.success
    assert response.object_names == [] and response.placements == []
    assert 'failed=2' in response.status
    assert all(name + '[FAILED:' in response.status for name in names)


def test_single_object_placement_uses_exact_pinned_frame_after_ring_eviction():
    """Do not associate a slow inference result with a newer camera cloud."""
    import threading
    from cv_bridge import CvBridge
    from my_actions.srv import FindPlacementCandidates
    from sensor_msgs_py import point_cloud2

    header = Header(frame_id='base_link')
    header.stamp.sec = 123
    yy, xx = np.mgrid[:10, :10]
    xyz = np.stack((xx * .003, yy * .003, np.zeros_like(xx)), axis=-1).astype(np.float32)
    pinned = point_cloud2.create_cloud_xyz32(header, xyz.reshape(-1, 3).tolist())
    pinned.height, pinned.width, pinned.row_step = 10, 10, 120
    newer_header = Header(frame_id='base_link')
    newer_header.stamp.sec = 130
    newer = point_cloud2.create_cloud_xyz32(newer_header, xyz.reshape(-1, 3).tolist())
    bridge = CvBridge()
    mask = bridge.cv2_to_imgmsg(np.ones((10, 10), np.uint8), 'mono8')
    mask.header = header
    request = FindPlacementCandidates.Request(
        observation_header=header, destination_mask=mask, orientation_policy='upright', support_mode='auto')

    def search(*args, **kwargs):
        raise RuntimeError('SEARCH_REACHED_WITH_ORIGINAL_FRAME')

    node = SimpleNamespace(
        _lock=threading.Lock(), _observation_clouds=[newer], _pinned_layout_cloud=pinned,
        _cached_object_cloud=pinned, _cached_completed_object_points=xyz.reshape(-1, 3),
        _cached_bottom_points=None, _cached_collision_envelope_points=None,
        _stamp_key=PlacementNode._stamp_key, _stamp_seconds=PlacementNode._stamp_seconds,
        _cloud_xyz=lambda *args, **kwargs: xyz, _bridge=bridge,
        _search_placements_api=search,
        get_parameter=lambda name: SimpleNamespace(value=(
            'heightmap' if name == 'placement_backend' else 80 if
            name == 'minimum_destination_points' else .005)),
        _status_pub=SimpleNamespace(publish=lambda _: None),
        get_logger=lambda: SimpleNamespace(error=lambda _: None))
    result = PlacementNode._placement_callback(node, request, FindPlacementCandidates.Response())
    assert not result.success
    assert result.status == 'PLACEMENT_FAILED: SEARCH_REACHED_WITH_ORIGINAL_FRAME'


def test_mapped_destination_requires_same_verified_observation():
    from sensor_msgs.msg import PointCloud2
    from my_actions.srv import ResolveManipulationTarget
    from std_msgs.msg import Header
    header=Header(frame_id='base_link')
    header.stamp.sec=42
    current=PointCloud2(header=header)
    remembered=ResolveManipulationTarget.Response(success=True,observation_header=header)
    marker=object()
    client=SimpleNamespace(wait_for_service=lambda **k:True,
                           call_async=lambda q:SimpleNamespace(result=lambda:remembered))
    node=SimpleNamespace(_active_target_client=client,_wait_for_future=lambda *a:True,
                         _cloud_xyz=lambda cloud:marker)
    fallback=object()
    assert PlacementNode._mapped_destination(node,'tray',current,fallback) is marker
    remembered.observation_header.stamp.sec=41
    # ROS messages can share assigned Header objects; replace current explicitly.
    current.header=Header(frame_id='base_link');current.header.stamp.sec=42
    assert PlacementNode._mapped_destination(node,'tray',current,fallback) is fallback


@pytest.mark.parametrize('height', [-.2, float('nan'), float('inf')])
def test_invalid_layout_lift_is_rejected_before_scene_search(height):
    """Bad requested heights must not start packing or change reservations."""
    from my_actions.srv import PlanObjectLayout
    from sensor_msgs.msg import Image
    published = []
    node = SimpleNamespace(
        _status_pub=SimpleNamespace(publish=published.append),
        get_logger=lambda: SimpleNamespace(error=lambda *args: None))
    query = PlanObjectLayout.Request(
        object_names=['bowl'], object_masks=[Image()], orientation_policy='upright',
        lift_distance_m=height)
    result = PlacementNode._layout_callback(node, query, PlanObjectLayout.Response())
    assert not result.success and 'INVALID_LAYOUT_LIFT_DISTANCE' in result.status
    assert len(published) == 1
