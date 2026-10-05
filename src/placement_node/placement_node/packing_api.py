"""Typed transport for the same heightmap search in all controllers."""

import inspect
import json
from dataclasses import fields

from my_actions.srv import SearchPlacements

import numpy as np

from sensor_msgs_py import point_cloud2

from .inference import PlacementCandidate
from .packing import PackingDiagnostics, search_heightmap_placements
from .transforms import pose_matrix


def make_search_request(header, name, *args, **kwargs):
    """Preserve existing search options and exact cloud dimensions in transport."""
    kwargs.pop('timing_sink', None)
    bound = inspect.signature(search_heightmap_placements).bind(*args, **kwargs).arguments
    request = SearchPlacements.Request(
        observation_header=header, object_name=name,
        orientation_policy=bound.pop('orientation_policy'),
        support_mode=bound.pop('support_mode', 'auto'))
    for source, target in (
            ('object_points', 'object_cloud'), ('organized_scene', 'scene_cloud'),
            ('destination_points', 'destination_cloud'), ('bottom_points', 'bottom_cloud'),
            ('stack_support_points', 'stack_support_cloud')):
        points = bound.pop(source, None)
        if points is not None:
            setattr(request, target, point_cloud2.create_cloud_xyz32(
                header, np.asarray(points).reshape(-1, 3).tolist()))
    request.options_json = json.dumps(bound)
    return request


def decode_search_response(response):
    """Restore planner-independent poses and diagnostics for visualization."""
    if not response.diagnostics_json:
        raise RuntimeError(response.status)
    data = json.loads(response.diagnostics_json)
    for key in ('heightmap', 'heightmap_mask'):
        if data.get(key) is not None:
            data[key] = np.asarray(data[key])
    diagnostics = PackingDiagnostics(**data)
    candidates = []
    for message in response.candidates:
        pose = message.object_pose.pose
        q = pose.orientation
        p = pose.position
        matrix = pose_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        candidates.append(PlacementCandidate(
            matrix[:3, :3].astype(np.float32), matrix[:3, 3].astype(np.float32),
            message.score))
    return candidates, diagnostics


def diagnostics_json(diagnostics):
    """Keep debug grids without placing Python objects on the ROS wire."""
    data = {item.name: getattr(diagnostics, item.name) for item in fields(diagnostics)}
    return json.dumps(data, default=lambda value: value.tolist())
