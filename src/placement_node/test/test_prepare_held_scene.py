"""Exercise real organized-cloud conversion without ROS nodes or actuators."""

import threading
from types import SimpleNamespace

from cv_bridge import CvBridge

from my_actions.srv import PrepareHeldScene

import numpy as np

from placement_node.perception import prepare_held_scene

from sensor_msgs_py import point_cloud2

from std_msgs.msg import Header


def test_organized_source_and_destination_produce_frozen_scene():
    """Read source XYZ and serialize payload/environment/support in one callback."""
    header = Header(frame_id='base_link')
    header.stamp.sec = 1
    y, x = np.mgrid[:30, :30]
    source = (x >= 10) & (x < 20) & (y >= 10) & (y < 20)
    points = np.stack((x*.005, y*.005, np.where(source, .04, 0.)), axis=-1)
    cloud = point_cloud2.create_cloud_xyz32(header, points.reshape(-1, 3))
    cloud.height, cloud.width = 30, 30
    cloud.row_step = cloud.point_step*cloud.width
    mask = CvBridge().cv2_to_imgmsg(source.astype(np.uint8)*255, encoding='mono8')
    mask.header = header
    destination = CvBridge().cv2_to_imgmsg((~source).astype(np.uint8)*255, encoding='mono8')
    destination.header = header
    result = prepare_held_scene(PrepareHeldScene.Request(
        cloud=cloud, observation_header=header, source_mask=mask,
        destination_mask=destination), PrepareHeldScene.Response())
    assert result.success, result.status
    assert result.status == 'IMMUTABLE_HELD_SCENE_READY'
    assert result.source_object_cloud.width > 0 and result.collision_scene.width > 0
    assert result.destination_support_cloud.width > 0
    assert result.source_object_cloud.header == header
    assert result.collision_scene.header == header
    assert 'bottom' in {field.name for field in result.source_object_cloud.fields}
    assert 'support' in {field.name for field in result.collision_scene.fields}


def test_disabled_completed_cloud_preview_preserves_bottom_cache(monkeypatch):
    """A disabled debug publisher must not raise or duplicate cloud publication."""
    from placement_node import node

    header = Header(frame_id='base_link')
    header.stamp.sec = 1
    message = point_cloud2.create_cloud_xyz32(header, [[0., 0., .04]])
    points = np.array([[0., 0., 0.]])
    diagnostics = dict.fromkeys((
        'bottom_points', 'collision_envelope_points', 'footprint_model', 'support_points',
        'footprint_points', 'footprint_area_m2', 'fitted_radius_m'), 0)
    diagnostics['plane_normal'] = [0., 0., 1.]
    monkeypatch.setattr(node, 'construct_support_bottom',
                        lambda *args: (points, points, points, diagnostics))
    published, warnings = [], []
    fake = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=True if name ==
                                                   'enable_object_bottom_completion' else 1.),
        _lock=threading.Lock(), _cached_object_cloud=message,
        _observation_clouds=[message], _completed_object_stamp=None,
        _bottom_attempt_stamp=None, _stamp_key=lambda header: (header.stamp.sec, 0),
        _stamp_seconds=lambda header: float(header.stamp.sec),
        _cloud_xyz=lambda cloud: points,
        _completed_object_pub=node._NOOP_PUB,
        _bottom_pub=SimpleNamespace(publish=published.append),
        get_logger=lambda: SimpleNamespace(warn=warnings.append, info=lambda text: None))
    node.PlacementNode._try_complete_cached_object(fake)
    assert not warnings
    assert fake._completed_object_stamp == (1, 0)
    assert len(published) == 1
