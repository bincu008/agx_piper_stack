"""Persistent destination base, actual placed geometry and stable grid identity."""

import copy
from collections import OrderedDict

import numpy as np

from sensor_msgs.msg import PointField

from sensor_msgs_py import point_cloud2

from .prepared_plan_review import pose_matrix


def cloud_xyz(cloud):
    """Read finite mapped geometry without changing its source header."""
    if cloud.header.frame_id != 'base_link':
        raise ValueError('DESTINATION_MEMORY_REQUIRES_BASE_LINK')
    values = point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=False)
    points = np.column_stack([values[key].reshape(-1) for key in ('x', 'y', 'z')])
    return points[np.isfinite(points).all(axis=1)]


def mapped_inputs(memory, request, destination_name=''):
    """Bind known destination geometry explicitly into a held-scene request."""
    if memory is None:
        return request
    with memory.lock:
        destination = memory.destination
        if destination is None or (
                destination_name and destination['key'] != memory.key(destination_name)):
            return request
        if destination['calibration'] != memory.calibration():
            raise RuntimeError('DESTINATION_CALIBRATION_CHANGED: reset destination explicitly')
        request.destination_base_cloud = destination.get('base_cloud', destination['cloud'])
        placed = destination.get('placed_points')
        if placed is not None and len(placed):
            request.placed_scene_cloud = point_cloud2.create_cloud_xyz32(
                destination['geometry_header'], placed.astype(np.float32))
    return request


def remember_release(memory, request, plan):
    """Merge once, only after a confirmed prepared release at the actual FK pose."""
    if not plan or plan.get('phase') != 'RELEASED' or not plan.get('pick_place'):
        return False
    if request.plan_revision and request.plan_revision != plan['revision']:
        raise RuntimeError('PLACED_PLAN_REVISION_MISMATCH')
    if memory.node._held_object is not None:
        raise RuntimeError('PLACED_MEMORY_REQUIRES_CONFIRMED_RELEASE')
    key = memory.key(request.object_name)
    with memory.lock:
        destination = memory.destination
        if destination is None or destination['key'] != memory.key(plan['destination_name']):
            raise RuntimeError('PLACED_DESTINATION_MEMORY_MISMATCH')
        if (plan.get('destination_id')
                and destination['object_id'] != plan['destination_id']):
            raise RuntimeError('PLACED_DESTINATION_ID_CHANGED')
        token = (plan['revision'], key, plan['header'].stamp.sec, plan['header'].stamp.nanosec)
        placements = destination.setdefault('placements', OrderedDict())
        if token in placements:
            return False
        record = memory.objects.get(key)
        if record is None:
            raise RuntimeError('PLACED_SOURCE_MEMORY_MISSING')
        source = plan['review_source']
        points = cloud_xyz(source)
        if not len(points):
            raise RuntimeError('PLACED_SOURCE_GEOMETRY_EMPTY')
        matrix = pose_matrix(plan['lift'].placement_ee_pose.pose) @ np.linalg.inv(
            pose_matrix(plan['candidate'].grasp_pose.pose))
        actual = points @ matrix[:3, :3].T + matrix[:3, 3]
        placed = destination.get('placed_points', np.empty((0, 3), np.float32))
        placed = np.vstack((placed, actual)).astype(np.float32)
        base = destination.setdefault('base_cloud', destination['cloud'])
        base_points = destination.setdefault('base_points', cloud_xyz(base))
        merged = np.vstack((base_points, placed)).astype(np.float32)
        fields = [PointField(name=axis, offset=i*4, datatype=PointField.FLOAT32, count=1)
                  for i, axis in enumerate(('x', 'y', 'z'))]
        fields.append(PointField(name='placed', offset=12, datatype=PointField.UINT8, count=1))
        flags = np.concatenate((np.zeros(len(base_points), np.uint8),
                                np.ones(len(placed), np.uint8)))
        rows = [(*point, int(flag)) for point, flag in zip(merged, flags)]
        cloud = point_cloud2.create_cloud(destination['geometry_header'], fields, rows)
        position = actual.mean(axis=0).tolist()
        memory.node.world_state.upsert_object(
            record['object_id'], record['name'], position, 'placed', True)
        memory.directory[record['object_id']] = {'name': record['name'], 'position': position}
        while len(memory.directory) > 256:
            memory.directory.popitem(last=False)
        placements[token] = {
            'object_id': record['object_id'], 'name': record['name'],
            'source_header': copy.deepcopy(source.header),
            'placement_pose': copy.deepcopy(plan['lift'].placement_ee_pose),
            'points': len(actual),
        }
        destination['placed_points'] = placed
        destination['points'], destination['cloud'] = merged, cloud
        memory.revision += 1
        destination['revision'] = memory.revision
        memory.objects.pop(key)
    return True


def check_capacity(memory, source, destination_name=''):
    """Reject map storage overflow before planning/actuation, never at release."""
    if memory is None:
        return
    with memory.lock:
        destination = memory.destination
        if destination is None or (
                destination_name and destination['key'] != memory.key(destination_name)):
            return
        if (len(destination.get('placements', ())) >= 128
                or len(destination['points'])+source.width*source.height > 500000):
            raise RuntimeError('DESTINATION_MEMORY_CAPACITY_REACHED')


def grasp_scene(memory, observation):
    """Keep the immutable pixel prefix and append mapped collision geometry."""
    with memory.lock:
        destination = memory.destination
        if destination is None:
            return None
        if destination['calibration'] != memory.calibration():
            raise RuntimeError('DESTINATION_CALIBRATION_CHANGED: reset destination explicitly')
        mapped = destination['points']
        # Do not filter/reorder NaNs or pixels: VAMP verifies this exact prefix.
        values = point_cloud2.read_points(observation.cloud, field_names=('x', 'y', 'z'))
        prefix = np.column_stack([values[k].reshape(-1) for k in ('x', 'y', 'z')])
        if len(prefix)+len(mapped) > 1000000:
            raise RuntimeError('MAPPED_GRASP_SCENE_TOO_LARGE')
        return point_cloud2.create_cloud_xyz32(
            observation.cloud.header, np.vstack((prefix, mapped)).astype(np.float32))
