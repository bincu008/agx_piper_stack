"""Freeze payload and environmental geometry for physical motion planning."""

from cv_bridge import CvBridge

import numpy as np

from scipy.spatial import cKDTree

from sensor_msgs.msg import PointField

from sensor_msgs_py import point_cloud2

from .packing import destination_support_mask
from .perception import (construct_support_bottom, recover_semantic_source_mask,
                         recover_depth_connected_boundary)


def prepare_held_scene(request, response):
    """Use exact synchronized cloud/masks; never remove neighboring geometry."""
    try:
        cloud, header = request.cloud, request.observation_header
        if (header.frame_id != 'base_link' or cloud.header != header or
                (not header.stamp.sec and not header.stamp.nanosec) or
                not cloud.width or cloud.height < 2 or len(cloud.data) > 16*1024*1024):
            raise ValueError('INVALID_HELD_OBSERVATION')
        bridge = CvBridge()
        mask = bridge.imgmsg_to_cv2(request.source_mask, desired_encoding='mono8') > 0
        if (request.source_mask.header.stamp != header.stamp or
                mask.shape != (cloud.height, cloud.width)):
            raise ValueError('HELD_MASK_OBSERVATION_MISMATCH')
        organized = point_cloud2.read_points_numpy(
            cloud, field_names=('x', 'y', 'z'), skip_nans=False).reshape(
                cloud.height, cloud.width, 3)
        finite = np.isfinite(organized).all(axis=2)
        measured = organized[mask & finite]
        if len(measured) < 80:
            raise ValueError('HELD_SOURCE_TOO_SMALL')
        completed, bottom, envelope, _ = construct_support_bottom(measured, organized[finite])
        body = np.vstack((completed, envelope))
        semantic = request.semantic_mask
        associated = mask.copy()
        if semantic.width:
            semantic_pixels = bridge.imgmsg_to_cv2(semantic, desired_encoding='mono8') > 0
            if (semantic.header.stamp != header.stamp or semantic_pixels.shape != mask.shape):
                raise ValueError('SEMANTIC_MASK_OBSERVATION_MISMATCH')
            # Reuse layout source ownership: bounded distance inside the semantic
            # target only, with lower stands/table protected from recovery.
            semantic_pixels &= organized[..., 2] >= np.percentile(measured[:, 2], 3)-.003
            associated = recover_semantic_source_mask(
                organized, mask, semantic_pixels, surface_points=body, distance_m=.012)
            body = np.vstack((body, organized[associated & ~mask & finite]))
        # Registered-depth/color seams can leave the same cup surface just
        # outside SAM. Recover only measured neighbors in a bounded 5-pixel
        # image band, above the bottom/table, never the tray. A 4 mm spatial
        # bound and original-mask seeds prevent iterative region growth.
        if semantic.width:
            rows, columns = np.nonzero(semantic_pixels)
            protected = np.ones(mask.shape, bool)
            if len(rows):
                protected[max(0, rows.min()-5):rows.max()+6,
                          max(0, columns.min()-5):columns.max()+6] = False
            protected |= organized[..., 2] <= float(np.median(bottom[:, 2]))+.005
            if request.destination_mask.width:
                destination_pixels = bridge.imgmsg_to_cv2(request.destination_mask, desired_encoding='mono8') > 0
                if destination_pixels.shape != mask.shape:
                    raise ValueError('DESTINATION_MASK_OBSERVATION_MISMATCH')
                protected |= destination_pixels
            recovered = recover_depth_connected_boundary(
                organized, associated, protected, distance_m=.004, pixel_radius=5)
            body = np.vstack((body, organized[recovered & ~associated & finite]))
            associated = recovered
        flags = (body[:, 2] <= np.percentile(bottom[:, 2], 50)+.003).astype(np.uint8)
        points = np.vstack((body, bottom))
        flags = np.concatenate((flags, np.full(len(bottom), 2, np.uint8)))
        fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(('x', 'y', 'z'))]
        response.source_object_cloud = point_cloud2.create_cloud(
            header, fields+[PointField(name='bottom', offset=12,
                                       datatype=PointField.UINT8, count=1)],
            [(*p, int(f)) for p, f in zip(points, flags)])
        support = np.zeros(mask.shape, np.uint8)
        destination = request.destination_mask
        mapped_base = request.destination_base_cloud
        base_points = None
        base_support = None
        if mapped_base.width:
            if mapped_base.header.frame_id != 'base_link' or len(mapped_base.data) > 16*1024*1024:
                raise ValueError('INVALID_MAPPED_DESTINATION')
            base_points = point_cloud2.read_points_numpy(
                mapped_base, field_names=('x', 'y', 'z'), skip_nans=True).reshape(-1, 3)
            if len(base_points) < 80:
                raise ValueError('MAPPED_DESTINATION_TOO_SMALL')
            distance, _ = cKDTree(base_points).query(
                organized[finite], distance_upper_bound=.01)
            valid = np.zeros(mask.shape, bool)
            valid[finite] = distance <= .01
            support = destination_support_mask(
                organized, valid & ~associated, base_points).astype(np.uint8)
            base_support = destination_support_mask(
                base_points, np.ones(len(base_points), bool), base_points).astype(np.uint8)
        elif destination.width:
            dmask = bridge.imgmsg_to_cv2(destination, desired_encoding='mono8') > 0
            if destination.header.stamp != header.stamp or dmask.shape != mask.shape:
                raise ValueError('DESTINATION_MASK_OBSERVATION_MISMATCH')
            valid = dmask & finite & ~associated
            if valid.sum() < 80:
                raise ValueError('DESTINATION_SUPPORT_TOO_SMALL')
            support = destination_support_mask(organized, valid, organized[valid]).astype(np.uint8)
        environment = finite & ~associated
        scene_points, scene_support = organized[environment], support[environment]
        support_points = organized[support > 0]
        if base_points is not None:
            scene_points = np.vstack((scene_points, base_points))
            scene_support = np.concatenate((scene_support, base_support))
            support_points = base_points[base_support > 0]
        placed = request.placed_scene_cloud
        if placed.width:
            if placed.header.frame_id != 'base_link' or len(placed.data) > 16*1024*1024:
                raise ValueError('INVALID_MAPPED_PLACED_SCENE')
            placed_points = point_cloud2.read_points_numpy(
                placed, field_names=('x', 'y', 'z'), skip_nans=True).reshape(-1, 3)
            scene_points = np.vstack((scene_points, placed_points))
            scene_support = np.concatenate((scene_support, np.zeros(len(placed_points), np.uint8)))
        response.collision_scene = point_cloud2.create_cloud(
            header, fields+[PointField(name='support', offset=12,
                                       datatype=PointField.UINT8, count=1)],
            [(*p, int(f)) for p, f in zip(scene_points, scene_support)])
        response.destination_support_cloud = point_cloud2.create_cloud_xyz32(header, support_points)
        center = points.mean(axis=0)
        (response.source_object_center.x, response.source_object_center.y,
         response.source_object_center.z) = map(float, center)
        response.success = True
        response.status = 'IMMUTABLE_HELD_SCENE_READY'
    except (ValueError, TypeError, RuntimeError) as error:
        response.status = str(error)
    return response
