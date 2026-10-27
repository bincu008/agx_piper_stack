"""Payload scene construction preserves adjacent obstacle samples."""

import copy

from cv_bridge import CvBridge

from my_actions.srv import PrepareHeldScene

import numpy as np

from placement_node import held_scene

from sensor_msgs_py import point_cloud2

from std_msgs.msg import Header


def test_payload_mask_removes_only_its_own_points(monkeypatch):
    """Retain neighboring obstacles and reject mask/frame mismatches."""
    header = Header(frame_id='base_link')
    header.stamp.sec = 10
    xyz = np.zeros((20, 20, 3), np.float32)
    xyz[..., 0], xyz[..., 1] = np.meshgrid(np.arange(20)*.003, np.arange(20)*.003)
    xyz[..., 2] = .10
    cloud = point_cloud2.create_cloud_xyz32(header, xyz.reshape(-1, 3).tolist())
    cloud.height, cloud.width, cloud.row_step = 20, 20, cloud.point_step*20
    mask = np.zeros((20, 20), np.uint8)
    mask[:, :5] = 255
    image = CvBridge().cv2_to_imgmsg(mask, encoding='mono8')
    image.header = copy.deepcopy(header)
    monkeypatch.setattr(held_scene, 'construct_support_bottom',
                        lambda measured, _: (measured, measured[:10], measured[:0], None))
    response = held_scene.prepare_held_scene(PrepareHeldScene.Request(
        observation_header=header, cloud=cloud, source_mask=image), PrepareHeldScene.Response())
    assert response.success
    remaining = point_cloud2.read_points_numpy(response.collision_scene,
                                               field_names=('x', 'y', 'z'))
    assert len(remaining) == 300
    assert np.all(remaining[:, 0] >= .015-.000001)
    image.header.stamp.sec = 11
    mismatch = held_scene.prepare_held_scene(PrepareHeldScene.Request(
        observation_header=header, cloud=cloud, source_mask=image), PrepareHeldScene.Response())
    assert not mismatch.success and 'MISMATCH' in mismatch.status


def test_seam_recovery_preserves_table_and_neighbor_outside_semantic_mask(monkeypatch):
    """Recover source edges only within semantic and measured height bounds."""
    header = Header(frame_id='base_link')
    header.stamp.sec = 12
    xyz = np.zeros((20, 20, 3), np.float32)
    xyz[..., 0], xyz[..., 1] = np.meshgrid(np.arange(20)*.003, np.arange(20)*.003)
    xyz[..., 2] = .1
    xyz[19, 5, 2] = .08  # Table return inside the image silhouette remains blocking.
    cloud = point_cloud2.create_cloud_xyz32(header, xyz.reshape(-1, 3).tolist())
    cloud.height, cloud.width, cloud.row_step = 20, 20, cloud.point_step*20
    bridge = CvBridge()
    source = np.zeros((20, 20), np.uint8)
    source[:, :5] = 255
    semantic = source.copy()
    semantic[:, 5] = 255
    mask, semantic_mask = [bridge.cv2_to_imgmsg(p, encoding='mono8')
                           for p in (source, semantic)]
    mask.header = copy.deepcopy(header)
    semantic_mask.header = copy.deepcopy(header)
    monkeypatch.setattr(held_scene, 'construct_support_bottom',
                        lambda measured, _: (measured, measured[:10], measured[:0], None))
    response = held_scene.prepare_held_scene(PrepareHeldScene.Request(
        observation_header=header, cloud=cloud, source_mask=mask,
        semantic_mask=semantic_mask), PrepareHeldScene.Response())
    assert response.success
    remaining = point_cloud2.read_points_numpy(response.collision_scene,
                                               field_names=('x', 'y', 'z'))
    assert len(remaining) == 281
    assert np.count_nonzero(remaining[:, 0] >= .018-.000001) == 280
    assert np.any(remaining[:, 2] < .09)


def test_bounded_registered_depth_seam_moves_with_payload_not_table_or_tray(monkeypatch):
    header = Header(frame_id='base_link')
    header.stamp.sec = 15
    xyz = np.zeros((20, 20, 3), np.float32)
    xyz[..., 0], xyz[..., 1] = np.meshgrid(np.arange(20)*.003, np.arange(20)*.003)
    xyz[..., 2] = .05 + np.arange(20)[:, None]*.004
    xyz[19, 5, 2] = .03  # Table stays outside payload.
    xyz[12, 5, 0] = .10  # Depth-discontinuous boundary stays blocking.
    xyz[10, 5, 0] = .0155  # 3.5 mm registered-depth seam belongs to payload.
    xyz[13, 5, 0] = .0165  # Beyond 4 mm remains an obstacle.
    cloud = point_cloud2.create_cloud_xyz32(header, xyz.reshape(-1, 3).tolist())
    cloud.height, cloud.width, cloud.row_step = 20, 20, cloud.point_step*20
    source = np.zeros((20, 20), np.uint8)
    source[:, :5] = 255
    semantic = source.copy()
    semantic[3, 5] = 255  # Curved silhouette bounds include the missing seam.
    destination = np.zeros_like(source)
    destination[:10, 10:] = 255
    destination[11, 5] = 255  # Explicit destination sample cannot be recovered.
    images = [CvBridge().cv2_to_imgmsg(m, encoding='mono8')
              for m in (source, semantic, destination)]
    for image in images:
        image.header = copy.deepcopy(header)
    monkeypatch.setattr(held_scene, 'construct_support_bottom',
        lambda measured, _: (measured, measured[:10], measured[:0], None))
    response = held_scene.prepare_held_scene(PrepareHeldScene.Request(
        observation_header=header, cloud=cloud, source_mask=images[0],
        semantic_mask=images[1], destination_mask=images[2]), PrepareHeldScene.Response())
    assert response.success, response.status
    payload = point_cloud2.read_points_numpy(response.source_object_cloud, field_names=('x', 'y', 'z'))
    obstacles = point_cloud2.read_points_numpy(response.collision_scene, field_names=('x', 'y', 'z'))
    assert np.any(np.all(np.isclose(payload, xyz[10, 5]), axis=1))
    for point in (xyz[19, 5], xyz[12, 5], xyz[11, 5], xyz[10, 6], xyz[13, 5]):
        assert np.any(np.all(np.isclose(obstacles, point), axis=1))


def test_mapped_tray_and_placed_objects_survive_out_of_view(monkeypatch):
    """A zero current tray mask retains mapped support and placed obstacles."""
    header = Header(frame_id='base_link')
    header.stamp.sec = 50
    x, y = np.meshgrid(np.linspace(0, .06, 20), np.linspace(0, .06, 20))
    xyz = np.stack((x, y, np.full_like(x, .1)), axis=2).astype(np.float32)
    cloud = point_cloud2.create_cloud_xyz32(header, xyz.reshape(-1, 3))
    cloud.height, cloud.width, cloud.row_step = 20, 20, cloud.point_step*20
    mask = CvBridge().cv2_to_imgmsg(np.full((20, 20), 255, np.uint8), 'mono8')
    mask.header = copy.deepcopy(header)
    missing = CvBridge().cv2_to_imgmsg(np.zeros((20, 20), np.uint8), 'mono8')
    missing.header = copy.deepcopy(header)
    old = copy.deepcopy(header)
    old.stamp.sec = 20
    base = xyz.reshape(-1, 3).copy()
    base[:, 0] += .4
    base[:, 2] = .02
    placed = base[:100].copy()
    placed[:, 2] += .06
    base_cloud = point_cloud2.create_cloud_xyz32(old, base)
    placed_cloud = point_cloud2.create_cloud_xyz32(old, placed)
    monkeypatch.setattr(held_scene, 'construct_support_bottom',
                        lambda measured, _: (measured, measured[:10], measured[:0], None))
    reply = held_scene.prepare_held_scene(PrepareHeldScene.Request(
        observation_header=header, cloud=cloud, source_mask=mask, destination_mask=missing,
        destination_base_cloud=base_cloud, placed_scene_cloud=placed_cloud),
        PrepareHeldScene.Response())
    assert reply.success, reply.status
    result = point_cloud2.read_points(reply.collision_scene)
    assert len(result) == len(base)+len(placed)
    assert np.all(result['support'][-len(placed):] == 0)
    support = point_cloud2.read_points_numpy(reply.destination_support_cloud,
                                            field_names=('x', 'y', 'z'))
    np.testing.assert_allclose(support, base)
    assert base_cloud.header == old and placed_cloud.header == old
    assert reply.collision_scene.header == header
