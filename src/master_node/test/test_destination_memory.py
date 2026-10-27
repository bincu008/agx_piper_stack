"""Persistent tray geometry from confirmed releases, without actuator commands."""

import copy
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped

from master_node.destination_memory import (
    check_capacity, cloud_xyz, grasp_scene, mapped_inputs, remember_release,
)
from master_node.prepared_plan_review import pose_matrix

from my_actions import srv

import numpy as np

import pytest

from scipy.spatial.transform import Rotation

from sensor_msgs_py import point_cloud2

from test_manipulation_memory import fixture, resolve


def setup():
    """Ground a source and tray, then provide a simulated confirmed release."""
    memory, observation, calls, _, _ = fixture()
    assert resolve(memory, 'bowl').success
    assert resolve(memory, 'tray', 'destination').success
    source = memory.objects['bowl']['cloud']
    grasp = PoseStamped(header=source.header)
    grasp.pose.orientation.w = 1.
    grasp.pose.position.x = .08
    actual = copy.deepcopy(grasp)
    actual.pose.position.x, actual.pose.position.y, actual.pose.position.z = .24, .16, .04
    q = Rotation.from_euler('z', .3).as_quat()
    actual.pose.orientation.x, actual.pose.orientation.y = map(float, q[:2])
    actual.pose.orientation.z, actual.pose.orientation.w = map(float, q[2:])
    plan = {
        'phase': 'RELEASED', 'pick_place': True, 'key': 'bowl', 'destination_name': 'tray',
        'destination_id': memory.destination['object_id'], 'revision': 1,
        'header': copy.deepcopy(source.header), 'review_source': source,
        'candidate': SimpleNamespace(grasp_pose=grasp),
        'lift': SimpleNamespace(placement_ee_pose=actual)}
    request = srv.HeldMotionStage.Request(object_name='bowl', plan_revision=1)
    memory.destination['grid'] = {'unchanged': True}
    memory.destination['grid_state'] = {'reserved_cells': [[0, 0]]}
    return memory, observation, calls, request, plan


def test_release_merges_actual_fk_geometry_without_changing_base_or_grid():
    """Use achieved gripper attachment geometry, not the ideal slot center."""
    memory, _, _, request, plan = setup()
    destination = memory.destination
    base = destination['base_cloud']
    grid, state = destination['grid'], destination['grid_state']
    original = cloud_xyz(plan['review_source'])
    transform = (pose_matrix(plan['lift'].placement_ee_pose.pose)
                 @ np.linalg.inv(pose_matrix(plan['candidate'].grasp_pose.pose)))
    expected = original @ transform[:3, :3].T+transform[:3, 3]
    assert remember_release(memory, request, plan)
    assert destination['base_cloud'] is base
    assert destination['grid'] is grid and destination['grid_state'] is state
    np.testing.assert_allclose(destination['placed_points'], expected, atol=1.e-7)
    np.testing.assert_allclose(cloud_xyz(destination['cloud']),
                               np.vstack((cloud_xyz(base), expected)), atol=1.e-7)
    assert destination['cloud'].header == destination['geometry_header'] == base.header
    stored = next(iter(destination['placements'].values()))
    assert stored['source_header'] == plan['review_source'].header
    position = memory.directory[stored['object_id']]['position']
    np.testing.assert_allclose(position, expected.mean(axis=0), atol=1.e-7)
    assert 'bowl' not in memory.objects
    assert memory.snapshot()['active_destination']['placed_objects'] == 1


def test_release_update_is_idempotent_and_failed_release_does_not_merge():
    """Never double append on a repeated response or merge unconfirmed motion."""
    memory, _, _, request, plan = setup()
    memory.node.execution_api = SimpleNamespace(
        prepared_grasp_lift=SimpleNamespace(record=plan))
    memory.observe_result('release_placed_object', request,
                          srv.HeldMotionStage.Response(success=False))
    assert not len(memory.destination['placed_points'])
    memory.observe_result('release_placed_object', request,
                          srv.HeldMotionStage.Response(success=True))
    revision = memory.revision
    cloud = memory.destination['cloud']
    memory.observe_result('release_placed_object', request,
                          srv.HeldMotionStage.Response(success=True))
    assert memory.revision == revision and memory.destination['cloud'] is cloud


def test_cached_destination_survives_old_timestamp_occlusion_and_view_change():
    """Changing the current view must not erase the merged inventory/grid."""
    memory, observation, calls, request, plan = setup()
    assert remember_release(memory, request, plan)
    destination = memory.destination
    cloud, grid = destination['cloud'], destination['grid']
    destination['verified'] -= 1000
    next_view = copy.deepcopy(observation)
    points = np.full((32*32, 3), np.nan, np.float32)
    next_view.cloud = point_cloud2.create_cloud_xyz32(next_view.cloud.header, points)
    next_view.cloud.height, next_view.cloud.width = 32, 32
    next_view.cloud.row_step = 32*next_view.cloud.point_step
    next_view.camera_transform.transform.translation.x += .4
    before = calls.count('ground_object')
    reply = resolve(memory, 'tray', 'destination', next_view)
    assert reply.success and reply.reused
    assert calls.count('ground_object') == before
    assert memory.destination['cloud'] is cloud and memory.destination['grid'] is grid
    assert not np.any(memory.bridge.imgmsg_to_cv2(reply.mask, 'mono8'))


def test_mapped_planning_inputs_and_grasp_pixel_prefix_are_preserved():
    """Append remembered geometry without changing camera pixels or provenance."""
    memory, observation, _, request, plan = setup()
    assert remember_release(memory, request, plan)
    query = mapped_inputs(memory, srv.PrepareHeldScene.Request(), 'tray')
    assert query.destination_base_cloud == memory.destination['base_cloud']
    np.testing.assert_allclose(cloud_xyz(query.placed_scene_cloud),
                               memory.destination['placed_points'])
    overlay = grasp_scene(memory, observation)
    values = point_cloud2.read_points(overlay, field_names=('x', 'y', 'z'))
    points = np.column_stack([values[k] for k in ('x', 'y', 'z')])
    original = cloud_xyz(observation.cloud)
    np.testing.assert_equal(points[:len(original)], original)
    np.testing.assert_equal(points[len(original):], memory.destination['points'])
    assert overlay.header == observation.cloud.header


def test_switch_destination_discards_composite_and_grid_but_keeps_directory():
    """The next destination starts with its own clean geometry and slots."""
    memory, _, _, request, plan = setup()
    assert remember_release(memory, request, plan)
    previous = memory.destination
    assert resolve(memory, 'other tray', 'destination').success
    assert memory.destination['cloud'] is not previous['cloud']
    assert memory.destination['grid'] is None
    assert not memory.destination['placements']
    assert not len(memory.destination['placed_points'])
    assert any(x['name'] == 'tray' for x in memory.directory.values())


def test_capacity_and_destination_identity_are_checked_before_use():
    """Do not silently evict placed obstacles or attach to another tray."""
    memory, _, _, request, plan = setup()
    plan['destination_id'] = 'different destination'
    with pytest.raises(RuntimeError, match='DESTINATION_ID_CHANGED'):
        remember_release(memory, request, plan)
    memory.destination['placements'] = dict.fromkeys(range(128))
    with pytest.raises(RuntimeError, match='CAPACITY_REACHED'):
        check_capacity(memory, plan['review_source'], 'tray')


def test_multiple_releases_append_both_sources_and_keep_original_tray():
    """A subsequent object uses the same map and adds its own achieved cloud."""
    memory, _, _, request, plan = setup()
    assert remember_release(memory, request, plan)
    previous = memory.destination['placed_points'].copy()
    base = memory.destination['base_cloud']
    assert resolve(memory, 'second bowl').success
    second = copy.deepcopy(plan)
    second['revision'] += 1
    second['review_source'] = memory.objects['second bowl']['cloud']
    second['lift'].placement_ee_pose.pose.position.y += .12
    query = srv.HeldMotionStage.Request(object_name='second bowl', plan_revision=2)
    assert remember_release(memory, query, second)
    points = memory.destination['placed_points']
    np.testing.assert_array_equal(points[:len(previous)], previous)
    assert len(points) == 2*len(previous)
    assert memory.destination['base_cloud'] is base
    assert memory.snapshot()['active_destination']['placed_objects'] == 2
    values = point_cloud2.read_points(memory.destination['cloud'], field_names=('placed',))
    assert np.count_nonzero(values['placed']) == len(points)


def test_release_rejects_wrong_revision_or_still_held_state():
    """A successful RPC alone cannot substitute for a matching confirmed release."""
    memory, _, _, request, plan = setup()
    request.plan_revision += 1
    with pytest.raises(RuntimeError, match='REVISION_MISMATCH'):
        remember_release(memory, request, plan)
    request.plan_revision = plan['revision']
    memory.node._held_object = object()
    with pytest.raises(RuntimeError, match='CONFIRMED_RELEASE'):
        remember_release(memory, request, plan)
    assert not len(memory.destination['placed_points'])


def test_refined_held_release_uses_its_matching_memory_record():
    """The older held-motion path also updates the same destination map."""
    memory, _, _, request, plan = setup()
    memory.node.execution_api = SimpleNamespace(
        prepared_grasp_lift=SimpleNamespace(record=None),
        held_motion=SimpleNamespace(record=plan))
    memory.observe_result('release_placed_object', request,
                          srv.HeldMotionStage.Response(success=True))
    assert memory.snapshot()['active_destination']['placed_objects'] == 1
