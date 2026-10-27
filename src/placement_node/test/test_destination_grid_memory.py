"""Production search/commit callbacks retain a grid as destination content grows."""

import copy
import json
import threading
from collections import deque
from types import SimpleNamespace

from my_actions.srv import CommitFloorSlots, SearchPlacements

import numpy as np

from placement_node.node import PlacementNode

from sensor_msgs_py import point_cloud2

from std_msgs.msg import Header


def test_two_searches_keep_grid_and_exclude_committed_cells():
    """Run real heightmap searches and commits without ROS actuator services."""
    params = {'candidate_count': 16, 'packing_grid_resolution_m': .005,
              'packing_upright_yaw_step_deg': 15., 'packing_free_yaw_step_deg': 30.,
              'packing_tilt_angles_deg': [0.], 'packing_clearance_m': .002,
              'packing_contact_tolerance_m': .006, 'packing_minimum_support_area_m2': .0002,
              'packing_support_margin_m': .002, 'packing_maximum_pile_height_m': .3,
              'packing_stack_minimum_coverage': .6, 'packing_adjacency_weight': .004,
              'floor_slot_adaptive': True, 'floor_slot_max_size_m': .125,
              'floor_slot_min_size_m': .06, 'floor_slot_large_square_scale': 1.5}
    publisher = SimpleNamespace(publish=lambda _: None)
    node = SimpleNamespace(
        get_parameter=lambda key: SimpleNamespace(value=params[key]),
        _destination_pub=publisher, _pipeline_timing_pub=publisher,
        _slot_lock=threading.Lock(), _slot_layouts={}, _slot_pending=deque(),
        _publish_floor_slots=lambda *args, **kwargs: None)
    for name in ('_cloud_xyz', '_point', '_quaternion_from_rotation', '_pose_matrix'):
        setattr(node, name, getattr(PlacementNode, name))
    header = Header(frame_id='base_link')
    header.stamp.sec = 30
    x, y = np.meshgrid(np.linspace(0, .34, 70), np.linspace(0, .26, 55))
    base = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
    x, y = np.meshgrid(np.linspace(-.035, .035, 20), np.linspace(-.035, .035, 20))
    source = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .04))).astype(np.float32)
    immutable = point_cloud2.create_cloud_xyz32(header, base)
    request = SearchPlacements.Request(
        observation_header=header, object_name='first bowl', orientation_policy='upright',
        support_mode='floor_slots', destination_base_cloud=immutable,
        object_cloud=point_cloud2.create_cloud_xyz32(header, source),
        destination_cloud=point_cloud2.create_cloud_xyz32(header, base),
        scene_cloud=point_cloud2.create_cloud_xyz32(header, base))
    first = PlacementNode._search_callback(node, request, SearchPlacements.Response())
    assert first.success, first.status
    pose = first.candidates[0].object_pose
    first_cells = set(map(tuple, json.loads(first.diagnostics_json)['slot_candidate_cells'][0]))
    commit = PlacementNode._commit_floor_slots(
        node, CommitFloorSlots.Request(selected_pose=pose), CommitFloorSlots.Response())
    assert commit.success, commit.status
    key, entry = next(iter(node._slot_layouts.items()))
    layout = copy.deepcopy(entry['layout'])
    # Recorded actual placement geometry becomes part of the destination map.
    matrix = PlacementNode._pose_matrix(pose.pose)
    placed = (source-source.mean(axis=0)) @ matrix[:3, :3].T+matrix[:3, 3]
    composite = np.vstack((base, placed))
    newer = copy.deepcopy(header)
    newer.stamp.sec += 10
    request.observation_header = newer
    request.object_name = 'second bowl'
    request.object_cloud = point_cloud2.create_cloud_xyz32(newer, source)
    request.destination_cloud = point_cloud2.create_cloud_xyz32(newer, composite)
    request.scene_cloud = point_cloud2.create_cloud_xyz32(newer, base)
    second = PlacementNode._search_callback(node, request, SearchPlacements.Response())
    assert second.success, second.status
    assert len(node._slot_layouts) == 1 and node._slot_layouts[key] is entry
    assert entry['layout'] == layout and first_cells <= entry['reserved']
    for cells in json.loads(second.diagnostics_json)['slot_candidate_cells']:
        assert not first_cells & set(map(tuple, cells))
    second_commit = PlacementNode._commit_floor_slots(
        node, CommitFloorSlots.Request(selected_pose=second.candidates[0].object_pose),
        CommitFloorSlots.Response())
    assert second_commit.success, second_commit.status
    assert second_commit.occupied_slots > commit.occupied_slots
    assert immutable.header == header
