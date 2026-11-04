"""Master-owned, bounded target geometry and reusable destination memory."""

from collections import OrderedDict
import copy
import json
import math
from pathlib import Path
import re
import threading
import time

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from my_actions import srv
import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
import yaml

from .calibration import pose_matrix


def cloud_xyz(cloud):
    """Read finite mapped geometry without changing its source header."""
    if cloud.header.frame_id != 'base_link':
        raise ValueError('DESTINATION_MEMORY_REQUIRES_BASE_LINK')
    points = point_cloud2.read_points_numpy(cloud, field_names=['x', 'y', 'z'], skip_nans=False)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
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
        fields = [PointField(name=axis, offset=i * 4, datatype=PointField.FLOAT32, count=1)
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
                or len(destination['points']) + source.width * source.height > 500000):
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
        if len(prefix) + len(mapped) > 1000000:
            raise RuntimeError('MAPPED_GRASP_SCENE_TOO_LARGE')
        return point_cloud2.create_cloud_xyz32(
            observation.cloud.header, np.vstack((prefix, mapped)).astype(np.float32))


def check_target_view(memory, request, response):
    """Resolve expected visibility without inference, memory mutation or motion."""
    try:
        role = request.role or 'object'
        if role not in ('object', 'destination') or not request.name.strip():
            raise ValueError('INVALID_TARGET_VIEW_REQUEST')
        with memory.lock:
            record = memory._record(request.name, role)
            if record is None:
                records = ([memory.destination] if role == 'destination'
                           else list(memory.objects.values()))
                record = next((r for r in records if r and r['object_id'] == request.name), None)
            response.memory_revision = memory.revision
            if record is not None:
                if record['calibration'] != memory.calibration():
                    response.success, response.status = True, 'LOCATION_UNKNOWN'
                    return response
                position = list(record['position'])
                response.object_id = record['object_id']
                target_cloud = record['cloud']
                response.memory_revision = record['revision']
            else:
                entries = [(key, item) for key, item in memory.directory.items()
                           if key == request.name or
                           memory.key(item['name']) == memory.key(request.name)]
                if len(entries) != 1:
                    response.success, response.status = True, 'LOCATION_UNKNOWN'
                    return response
                response.object_id, entry = entries[0]
                position = list(entry['position'])
                target_cloud = None
                response.location_only = True
        response.position = Point(**dict(zip(('x', 'y', 'z'), map(float, position))))
        if request.observation_cloud.width:
            cloud, transform = request.observation_cloud, request.camera_transform
        else:
            if request.camera_transform.header.frame_id:
                raise ValueError('TARGET_VIEW_OBSERVATION_PAIR_REQUIRED')
            observation = memory.call('capture_observation', srv.CaptureObservation.Request())
            if not observation.success:
                raise RuntimeError('TARGET_VIEW_CAPTURE_FAILED: ' + observation.status)
            cloud, transform = observation.cloud, observation.camera_transform
        info = copy.deepcopy(getattr(memory.node, 'current_camera_info', None))
        if info is None:
            raise RuntimeError('TARGET_VIEW_CAMERA_INFO_UNAVAILABLE')
        query = srv.CheckTargetVisibility.Request(
            scene_cloud=cloud, camera_transform=transform,
            camera_info=info, target_position=response.position)
        if target_cloud is not None:
            query.target_cloud = target_cloud
        result = memory.call('check_target_visibility', query)
        for field in ('success', 'status', 'visible', 'observation_header',
                      'sample_count', 'in_view_count', 'clear_count', 'visible_fraction',
                      'target_cell_id'):
            setattr(response, field, getattr(result, field))
    except Exception as error:
        response.success, response.visible = False, False
        response.status = str(error)
    return response


def parse_mapping(text, keys):
    """Validate every requested key before changing memory."""
    begin, end = text.find('{'), text.rfind('}')
    if begin < 0 or end < begin:
        raise ValueError('SCENE_INFERENCE_JSON_MISSING')
    pairs = json.loads(text[begin:end + 1], object_pairs_hook=lambda items: items)
    if not isinstance(pairs, list) or any(not isinstance(p, tuple) for p in pairs):
        raise ValueError('SCENE_INFERENCE_REQUIRES_OBJECT')
    if len(pairs) != len(keys) or {k for k, _ in pairs} != set(keys):
        raise ValueError('SCENE_INFERENCE_TARGET_KEYS_MISMATCH')
    return dict(pairs)


def pixel_box(box, width, height):
    """Require an explicit normalized box, never guess coordinate conventions."""
    if (not isinstance(box, list) or len(box) != 4 or
            any(type(x) not in (int, float) or not np.isfinite(x) for x in box) or
            not all(0 <= x <= 1000 for x in box) or
            not box[0] < box[2] or not box[1] < box[3]):
        raise ValueError('SCENE_GROUNDING_INVALID_BOX')
    result = [min((width if i % 2 == 0 else height) - 1,
                  math.floor(box[i] * (width if i % 2 == 0 else height) / 1000 + .5))
              for i in range(4)]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError('SCENE_GROUNDING_DEGENERATE_PIXEL_BOX')
    return result


class ScenePerception:
    """Ask about the entire target set; keep segmentation and memory per object."""

    def __init__(self, memory):
        """Share the bounded master memory and catalog-backed dispatcher."""
        self.memory = memory

    def targets(self, request):
        """Use the configured inventory plus the requested manipulation target."""
        objects = list(request.object_names)
        if not objects:
            path = (Path(get_package_share_directory('master_node')) /
                    'config/perception/scene_targets.yaml')
            objects = yaml.safe_load(path.read_text())['objects']
        active = request.active_object_name.strip()
        if active:
            root = self.memory.key(active.split(',')[0])
            index = next((i for i, name in enumerate(objects)
                          if self.memory.key(name) in (root, self.memory.key(active))), None)
            if index is None:
                objects.append(active)
            else:
                objects[index] = active
        names = []
        for name in objects:
            if not isinstance(name, str) or not name.strip() or len(name) > 256:
                raise ValueError('SCENE_TARGET_NAME_INVALID')
            if self.memory.key(name) not in {self.memory.key(n) for n in names}:
                names.append(name.strip())
        destination = request.destination_name.strip()
        if not destination or len(destination) > 256:
            raise ValueError('SCENE_DESTINATION_REQUIRED')
        if self.memory.key(destination) in {self.memory.key(n) for n in names}:
            raise ValueError('SCENE_OBJECT_DESTINATION_MUST_DIFFER')
        if not 1 <= len(names) <= 15:
            raise ValueError('SCENE_REQUIRES_1_TO_15_OBJECTS')
        return names + [destination], ['object'] * len(names) + ['destination']

    def infer(self, observation, prompt, tokens):
        """Keep both VLM requests individually timed by the shared dispatcher."""
        result = self.memory.call('infer_vision', srv.VisionInference.Request(
            image=observation.image, user_prompt=prompt, max_tokens=tokens))
        if not result.success:
            raise RuntimeError('SCENE_INFERENCE_FAILED: ' + result.status)
        self.memory.node.get_logger().info('[SCENE INFERENCE] ' + json.dumps(
            {'prompt': prompt, 'response': result.text}))
        return result.text

    def perceive(self, request, response):
        """Never actuate or replace cached destination geometry/reservations."""
        memory = self.memory
        admitted = memory.operation.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('ACTIVE_MEMORY_BUSY')
            memory.guard_mutation()
            names, roles = self.targets(request)
            if bool(request.image.data) != bool(request.observation_cloud.data):
                raise ValueError('SYNCHRONIZED_IMAGE_AND_CLOUD_REQUIRED')
            observation = (srv.CaptureObservation.Response(
                success=True, image=request.image, cloud=request.observation_cloud,
                camera_transform=request.camera_transform) if request.image.data else
                memory.call('capture_observation', srv.CaptureObservation.Request()))
            if not observation.success:
                raise RuntimeError(observation.status)
            cloud, image = observation.cloud, observation.image
            transform = observation.camera_transform
            if (cloud.header.frame_id != 'base_link' or cloud.height < 2 or
                    image.header.stamp != cloud.header.stamp or
                    (image.height, image.width) != (cloud.height, cloud.width) or
                    transform.header != cloud.header or
                    abs(sum(getattr(transform.transform.rotation, k)**2
                            for k in ('x', 'y', 'z', 'w')) - 1) > .001):
                raise ValueError('SCENE_OBSERVATION_ASSOCIATION_MISMATCH')
            keys = [f't{i}' for i in range(len(names))]
            targets = [{'key': key, 'description': name, 'role': role}
                       for key, name, role in zip(keys, names, roles)]
            listing = json.dumps(targets)
            presence_prompt = (
                'SCENE PRESENCE. Check ALL listed targets in this image: ' + listing +
                '. Return one JSON object mapping EVERY key to true if clearly visible, '
                'false if absent or uncertain. No extra keys or explanation.')
            presence = parse_mapping(self.infer(
                observation, presence_prompt, max(128, 24 * len(names))), keys)
            if any(type(value) is not bool for value in presence.values()):
                raise ValueError('SCENE_PRESENCE_REQUIRES_BOOLEANS')
            descriptions = {}
            for key, name, role in zip(keys, names, roles):
                if role == 'destination':
                    descriptions[key] = 'the complete visible placement surface of the ' + name
                else:
                    whole_object = re.sub(r'\s*,\s*grasp\s+its.*$', '', name, flags=re.IGNORECASE)
                    descriptions[key] = (
                        'the entire ' + whole_object +
                        ', including the full outer rim, outer body and handle if present, '
                        'not just its inner surface')
            grounding_prompt = (
                'Please provide the bounding box coordinate of the region EACH sentence '
                'describes: ' + json.dumps(descriptions) +
                '. Return ONLY one JSON object with keys ' + ', '.join(keys) +
                '. Each value MUST contain FOUR numbers [x1, y1, x2, y2] '
                'for the top-left and bottom-right corners of that target bounding box, '
                'in normalized 0-1000 coordinates. A point with only two numbers is NOT '
                'a bounding box. Use null for a target that is not visible. Cover the '
                'WHOLE visible extent of each object; do not box only a small patch '
                'of its surface. Do not return center points.')
            boxes = parse_mapping(self.infer(
                observation, grounding_prompt, max(512, 80 * len(names))), keys)
            pixels = {key: pixel_box(boxes[key], image.width, image.height)
                      for key in keys if presence[key] and boxes[key] is not None}
            response.observation_header = copy.deepcopy(cloud.header)
            response.names, response.roles = names, roles
            for key, name, role in zip(keys, names, roles):
                record = memory._record(name, role)
                observed, status = False, 'NOT_VISIBLE_IN_CURRENT_SCENE'
                if role == 'destination' and record is not None:
                    result = memory._resolve(srv.ResolveManipulationTarget.Request(
                        name=name, role=role, image=image, observation_cloud=cloud,
                        camera_transform=transform), srv.ResolveManipulationTarget.Response())
                    observed = bool(result.success and presence[key] and key in pixels)
                    status = result.status
                elif presence[key] and key in pixels:
                    segmented = memory.call('segment_object', srv.SegmentGroundedImage.Request(
                        image=image, bbox=pixels[key]))
                    if segmented.success:
                        mask = memory.bridge.imgmsg_to_cv2(segmented.mask, 'mono8') > 0
                        if (mask.shape != (image.height, image.width) or
                                segmented.mask.header.stamp != image.header.stamp):
                            raise ValueError('SCENE_MASK_DIMENSIONS_MISMATCH')
                        result = memory._resolve(srv.ResolveManipulationTarget.Request(
                            name=name, role=role, refresh=True, image=image,
                            observation_cloud=cloud, camera_transform=transform),
                            srv.ResolveManipulationTarget.Response(), grounded_mask=mask)
                        observed, status = result.success, result.status
                    else:
                        status = 'SCENE_SEGMENTATION_FAILED: ' + segmented.status
                record = memory._record(name, role)
                hint = next((entry for entry in memory.directory.values()
                             if memory.key(entry['name']) == memory.key(name)), None)
                position = (record['position'] if record else
                            hint['position'] if hint else [0., 0., 0.])
                if role == 'object' and not observed:
                    with memory.lock:
                        memory.objects.pop(memory.key(name), None)
                response.positions.append(
                    Point(**dict(zip(('x', 'y', 'z'), map(float, position)))))
                response.observed.append(observed)
                response.target_statuses.append(status)
            response.memory_revision = memory.revision
            response.success = True
            response.status = (f'SCENE_PERCEPTION_READY observed={sum(response.observed)}'
                               f'/{len(names)}')
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                memory.operation.release()
            memory.print_state('perceive_scene')
        return response


def stamp_ns(header):
    """Compare ROS capture times without changing their frame association."""
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


class PerceptionAcquisition:
    """Compose shared memory, visibility and guarded empty-hand view APIs."""

    def __init__(self, memory):
        """Keep only one bounded post-view capture requirement."""
        self.memory = memory
        self.lock = threading.Lock()
        self.after_view = None

    def guard(self):
        """Do not acquire a new object while a prior physical task is incomplete."""
        self.memory.guard_mutation()
        if self.memory.node.state != 'IDLE':
            raise RuntimeError('ACQUISITION_REQUIRES_IDLE_EMPTY_HAND')
        execution = getattr(self.memory.node, 'execution_api', None)
        session = getattr(execution, 'prepared_grasp_lift', None)
        if session and session.record and session.record['phase'] != 'PREPARED':
            raise RuntimeError('PREVIOUS_PICK_PLACE_NOT_COMPLETE')

    def ensure_memory(self, request, response):
        """Use known locations; batch-perceive only when inventory is incomplete."""
        memory = self.memory
        try:
            self.guard()
            names, roles = memory.scene_perception.targets(request)
            with memory.lock:
                records = []
                for name, role in zip(names, roles):
                    record = memory._record(name, role)
                    hint = next((item for item in memory.directory.values()
                                 if memory.key(item['name']) == memory.key(name)), None)
                    if role == 'destination':
                        known = (record if record and record['calibration'] == memory.calibration()
                                 else None)
                    else:
                        known = (record if record and record['calibration'] == memory.calibration()
                                 else hint)
                    records.append(known)
            if not all(records):
                return memory.call('perceive_scene', request)
            response.success = True
            response.status = 'SCENE_MEMORY_READY_NO_INFERENCE'
            response.names, response.roles = names, roles
            response.observation_header = copy.deepcopy(request.observation_cloud.header)
            response.positions = [Point(**dict(zip(('x', 'y', 'z'), map(float, r['position']))))
                                  for r in records]
            response.observed = [False] * len(records)
            response.target_statuses = ['LOCATION_REMEMBERED'] * len(records)
            response.memory_revision = memory.revision
        except Exception as error:
            response.status = str(error)
        return response

    def ensure_view(self, request, response, allow_motion=False):
        """Move only when geometry requires it; never retry an actuator call."""
        admitted = self.lock.acquire(blocking=False)
        dispatched = False
        arrived = False
        try:
            if not admitted:
                raise RuntimeError('TARGET_VIEW_ACQUISITION_BUSY')
            memory = self.memory
            self.guard()
            marker = self.after_view
            if (marker and marker['key'] == memory.key(request.name) and
                    request.observation_cloud.data and
                    stamp_ns(request.observation_cloud.header) <= marker['stamp_ns']):
                raise RuntimeError('POST_VIEW_CAPTURE_STALE: capture a new synchronized frame')
            checked = memory.call('check_target_view', request)
            if not checked.success:
                return checked
            if checked.status == 'LOCATION_UNKNOWN':
                checked.success = False
                checked.status = 'TARGET_LOCATION_UNKNOWN: acquire initial scene inventory'
                return checked
            if checked.status == 'READY':
                return checked
            if checked.status != 'LOOK_REQUIRED':
                raise RuntimeError('TARGET_VIEW_STATUS_UNEXPECTED: ' + checked.status)
            if not allow_motion:
                checked.success = False
                checked.status = 'LOOK_REQUIRED: motion-free calculation cannot acquire this view'
                return checked
            execution = memory.node.execution_api
            execution._guard()
            planned = memory.call('plan_next_best_view', srv.PlanNextBestView.Request(
                target_position=checked.position,
                standoff_distance=0.42,
                elevation_angle_degrees=60.0,
                base_rotation_degrees=0.0,
                preference='base_facing_slanted_down'))
            if not planned.success:
                raise RuntimeError('NEXT_VIEW_PLAN_FAILED: ' + planned.status)
            if len(planned.plan.joint_names) != 6:
                raise RuntimeError('NEXT_VIEW_PLAN_INVALID: expected 6 joints')
            dispatched = True
            started = time.monotonic()
            if not memory.node.move_arm_to(list(planned.plan.joint_trajectory.points[-1].positions),
                                           tolerance=0.01):
                raise RuntimeError('NEXT_VIEW_MOVE_FAILED')
            arrived = True
            memory.node.get_logger().info(
                f'[NEXT VIEW ARRIVAL] target={request.name} '
                f'duration={time.monotonic() - started:.3f}s')
            time.sleep(0.3)
            after = memory.call('capture_observation', srv.CaptureObservation.Request())
            if not after.success:
                raise RuntimeError('POST_VIEW_CAPTURE_FAILED: ' + after.status)
            self.after_view = {
                'key': memory.key(request.name),
                'stamp_ns': stamp_ns(after.cloud.header),
            }
            verification = memory.call('check_target_view', srv.CheckTargetView.Request(
                name=request.name, role=request.role,
                observation_cloud=after.cloud,
                camera_transform=after.camera_transform))
            if not verification.success:
                return verification
            if verification.status != 'READY':
                verification.success = False
                verification.status = (
                    'POST_VIEW_TARGET_NOT_READY: ' + verification.status +
                    f' (visible={verification.visible} '
                    f'fraction={verification.visible_fraction:.2f})')
                return verification
            checked.success = True
            checked.status = 'READY'
            checked.visible = verification.visible
            checked.visible_fraction = verification.visible_fraction
            checked.in_view_count = verification.in_view_count
            checked.clear_count = verification.clear_count
            checked.sample_count = verification.sample_count
            checked.observation_header = copy.deepcopy(verification.observation_header)
        except Exception as error:
            checked = srv.CheckTargetView.Response(
                success=False, visible=False, status=str(error))
        finally:
            if admitted:
                self.lock.release()
            memory.print_state('ensure_target_view')
            if dispatched and not arrived:
                memory.node.emergency_stop()
        return checked


def xyz(cloud, organized=False):
    """Read existing manual-base-frame geometry without applying another transform."""
    points = point_cloud2.read_points_numpy(cloud, field_names=['x', 'y', 'z'], skip_nans=False)
    points = np.asarray(points, dtype=np.float32)
    return points.reshape(cloud.height, cloud.width, 3) if organized else points.reshape(-1, 3)


def observed_mask(points, mapped, minimum=80, destination=False):
    """Verify a mapped target against measured depth, without changing its size."""
    finite = np.isfinite(points).all(axis=2)
    mask = np.zeros(finite.shape, dtype=bool)
    if len(mapped) < minimum or finite.sum() < minimum:
        return None
    measured = points[finite]
    distance, _ = cKDTree(mapped).query(measured, distance_upper_bound=0.008)
    mask[finite] = distance <= 0.008
    if mask.sum() < minimum:
        return None
    # A moved/occluded target must retain spatial evidence across its old extent.
    reverse, _ = cKDTree(measured).query(mapped, distance_upper_bound=0.008)
    # Tray rims distinguish an unchanged tray from bare table. Check each
    # evidenced side, not just aggregate overlap: a translated rectangle may
    # still share most of its two long sides with the old rectangle.
    rim = mapped[:, 2] > np.percentile(mapped[:, 2], 35) + 0.012
    rim_verified = False
    if destination and rim.sum() >= minimum:
        if (reverse[rim] <= 0.008).mean() < 0.5:
            return None
        centered = mapped[:, :2] - mapped[:, :2].mean(axis=0)
        axes = np.linalg.eigh(centered.T @ centered)[1]
        coordinates = centered @ axes
        sides = 0
        for axis in range(2):
            values = coordinates[:, axis]
            lo, hi = np.percentile(values, [1, 99])
            for side in (values <= lo + 0.03 * (hi - lo), values >= hi - 0.03 * (hi - lo)):
                patch = side & rim
                if patch.sum() >= 20:
                    sides += 1
                    if (reverse[patch] <= 0.008).mean() < 0.5:
                        return None
        rim_verified = sides >= 3
    required_coverage = 0.1 if rim_verified else 0.35
    if (reverse <= 0.008).mean() < required_coverage:
        return None
    return mask


class ManipulationMemory:
    """Hold geometry once; expose small identities and coordinates for agents."""

    def __init__(self, node):
        """Register shared memory endpoints and bounded stores."""
        self.node = node
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.operation = threading.Lock()
        self.objects = OrderedDict()
        self.destination = None
        self.directory = OrderedDict()
        self.revision = 0
        self.group = ReentrantCallbackGroup()
        self.scene_perception = ScenePerception(self)
        self.acquisition = PerceptionAcquisition(self)
        for schema, name, callback in (
            (srv.PerceiveScene, 'perceive_scene', self.scene_perception.perceive),
            (srv.PerceiveScene, 'ensure_scene_memory', self.acquisition.ensure_memory),
            (srv.CheckTargetView, 'ensure_target_view',
             lambda q, r: self.acquisition.ensure_view(q, r, allow_motion=True)),
            (srv.CheckTargetView, 'require_target_view', self.acquisition.ensure_view),
            (srv.ResolveManipulationTarget, 'resolve_target', self.resolve),
            (srv.ResolveManipulationTarget, 'refresh_active_target', self.refresh),
            (srv.ResolveManipulationTarget, 'forget_active_target', self.forget),
            (srv.ResolveManipulationTarget, 'get_active_target', self.get_target),
            (srv.StoreSegmentedTarget, 'store_segmented_target', self.store_segmented),
            (srv.GetManipulationMemory, 'get_manipulation_memory', self.inspect),
            (srv.GetSpatialContext, 'get_spatial_context', self.spatial_context),
            (srv.CheckTargetView, 'check_target_view',
             lambda q, r: check_target_view(self, q, r)),
            (srv.FindGraspCandidates, 'calculate_active_grasps', self.grasps),
            (srv.FindPlacementCandidates, 'find_active_placements', self.placements),
        ):
            node.create_service(schema, '/master/' + name, callback, callback_group=self.group)

    @staticmethod
    def key(name):
        """Normalize caller names without guessing semantic aliases."""
        return ' '.join(name.casefold().split())

    def calibration(self):
        """Identify the manual calibration used by cached geometry."""
        return np.asarray(getattr(self.node, 'X_matrix', []), dtype=float).tobytes()

    def call(self, name, request):
        """Dispatch through the authoritative shared API runtime."""
        return self.node.step_runtime._call(name, request)

    def _record(self, name, role):
        """Find an active record by explicit role and name."""
        key = self.key(name)
        if role == 'destination':
            return (
                self.destination if self.destination and self.destination['key'] == key else None
            )
        return self.objects.get(key)

    @staticmethod
    def same_view(left, right):
        """Require a matching wrist-camera viewpoint for RGB-D grasp reuse."""
        a, b = left.transform, right.transform
        t = sum(
            (getattr(a.translation, k) - getattr(b.translation, k)) ** 2 for k in ('x', 'y', 'z')
        )
        qa = np.array([getattr(a.rotation, k) for k in ('x', 'y', 'z', 'w')])
        qb = np.array([getattr(b.rotation, k) for k in ('x', 'y', 'z', 'w')])
        return t <= 0.002**2 and abs(float(qa @ qb)) >= 0.99999

    def _resolve(self, request, response, grounded_mask=None):
        """Validate mapped geometry or ground it on the synchronized observation."""
        self.guard_mutation()
        name, role = request.name.strip(), request.role or 'object'
        if not name or role not in ('object', 'destination'):
            raise ValueError('TARGET_NAME_AND_ROLE_REQUIRED')
        with self.lock:
            record = self._record(name, role)
            if role == 'destination' and self.destination and record is None:
                # The directory retains the hint; active geometry is released.
                self.destination = None
                self.revision += 1
                self.invalidate_destination()
        if bool(request.image.data) != bool(request.observation_cloud.data):
            raise ValueError('SYNCHRONIZED_IMAGE_AND_CLOUD_REQUIRED')
        if request.image.data:
            observation = srv.CaptureObservation.Response(
                success=True,
                image=request.image,
                cloud=request.observation_cloud,
                camera_transform=request.camera_transform,
            )
        else:
            observation = self.call('capture_observation', srv.CaptureObservation.Request())
            if not observation.success:
                raise RuntimeError(observation.status)
        cloud, image = observation.cloud, observation.image
        if (
            cloud.header.frame_id != 'base_link'
            or cloud.height < 2
            or image.header.stamp != cloud.header.stamp
            or (image.height, image.width) != (cloud.height, cloud.width)
        ):
            raise ValueError('TARGET_OBSERVATION_ASSOCIATION_MISMATCH')
        transform = observation.camera_transform
        if (
            transform.header.frame_id != 'base_link'
            or transform.header.stamp != cloud.header.stamp
            or abs(
                sum(getattr(transform.transform.rotation, k) ** 2 for k in ('x', 'y', 'z', 'w'))
                - 1.0
            )
            > 0.001
        ):
            raise ValueError('TARGET_MANUAL_TRANSFORM_ASSOCIATION_MISMATCH')
        points = xyz(cloud, organized=True)
        cached_destination = role == 'destination' and record is not None
        if cached_destination and record['calibration'] != self.calibration():
            raise RuntimeError('DESTINATION_CALIBRATION_CHANGED: reset destination explicitly')
        mask = None
        if (
            record
            and not cached_destination
            and not request.refresh
            and record['calibration'] == self.calibration()
            and time.monotonic() - record['verified'] < 300
            and (
                role == 'destination'
                or self.same_view(
                    record['observation'].camera_transform, observation.camera_transform
                )
            )
        ):
            mask = observed_mask(points, record['points'], destination=role == 'destination')
            if mask is not None:
                # With an unchanged calibrated view, preserve semantic pixel
                # boundaries. The 8 mm evidence gate may otherwise grow a
                # bowl mask onto adjacent table points on every reuse.
                previous = record['observation'].camera_transform.transform
                current = observation.camera_transform.transform
                stationary = (
                    max(abs(getattr(previous.translation, axis)-
                            getattr(current.translation, axis)) for axis in ('x', 'y', 'z')) < 1.e-6
                    and abs(sum(getattr(previous.rotation, axis)*
                                getattr(current.rotation, axis)
                                for axis in ('x', 'y', 'z', 'w'))) > 1.-1.e-12)
                semantic = self.bridge.imgmsg_to_cv2(record['semantic_mask'], 'mono8') > 0
                if stationary and semantic.shape == mask.shape:
                    mask &= semantic
                    if mask.sum() < 80:
                        mask = None
        if cached_destination:
            # Camera movement/temporary occlusion does not erase the inventory.
            # Current pixels are optional; mapped base/placed clouds feed planning.
            mask = observed_mask(points, record.get('base_points', record['points']),
                                 destination=True)
            if mask is None:
                mask = np.zeros(points.shape[:2], dtype=bool)
        reused = mask is not None
        if not reused:
            if role == 'object' and record:
                with self.lock:
                    self.objects.pop(record['key'], None)
            if role == 'destination' and record:
                with self.lock:
                    self.destination = None
                self.invalidate_destination()
            if grounded_mask is not None:
                mask = grounded_mask
            else:
                present = self.call(
                    'verify_presence', srv.VerifyPresence.Request(image=image, target=name)
                )
                if not present.success or not present.present:
                    raise RuntimeError('TARGET_NOT_PRESENT: ' + present.status)
                target = (
                    'the complete visible placement surface of the '
                    if role == 'destination'
                    else 'the complete visible object '
                ) + name
                ground = self.call(
                    'ground_object', srv.GroundObject.Request(image=image, target=target)
                )
                if not ground.success:
                    raise RuntimeError('TARGET_GROUNDING_FAILED: ' + ground.status)
                segmented = self.call(
                    'segment_object',
                    srv.SegmentGroundedImage.Request(image=image, bbox=ground.pixel_box),
                )
                if not segmented.success:
                    raise RuntimeError('TARGET_SEGMENTATION_FAILED: ' + segmented.status)
                mask = self.bridge.imgmsg_to_cv2(segmented.mask, 'mono8') > 0
        finite = mask & np.isfinite(points).all(axis=2)
        selected = points[finite]
        if len(selected) < 80 and not cached_destination:
            raise ValueError('INSUFFICIENT_TARGET_GEOMETRY')
        ys, xs = np.nonzero(finite)
        mask_message = self.bridge.cv2_to_imgmsg((finite * 255).astype(np.uint8), 'mono8')
        mask_message.header = copy.deepcopy(image.header)
        if not reused:
            center = np.median(selected, axis=0)
            object_id = self.node.world_state.associate_observation(name, center.tolist())
            self.node.world_state.upsert_object(object_id, name, center.tolist(), role, True)
            with self.lock:
                self.directory[object_id] = {'name': name, 'position': center.tolist()}
                while len(self.directory) > 256:
                    self.directory.popitem(last=False)
            self.revision += 1
            record = {
                'key': self.key(name),
                'name': name,
                'object_id': object_id,
                'role': role,
                'revision': self.revision,
                'points': selected.copy(),
                'position': center.tolist(),
                'cloud': point_cloud2.create_cloud_xyz32(cloud.header, selected),
                'geometry_header': copy.deepcopy(cloud.header),
                'grid': None,
                'calibration': self.calibration(),
            }
        if role == 'destination' and not cached_destination:
            record['base_cloud'] = record['cloud']
            record['base_points'] = record['points'].copy()
            record['placed_points'] = np.empty((0, 3), np.float32)
            record['placements'] = OrderedDict()
        record = dict(
            record,
            observation=observation,
            mask=mask_message,
            semantic_mask=mask_message,
            verified=time.monotonic(),
            pixel_box=([int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
                       if len(xs) else [0, 0, 0, 0]),
        )
        with self.lock:
            if role == 'destination':
                self.destination = record
            else:
                self.objects[record['key']] = record
                self.objects.move_to_end(record['key'])
                while len(self.objects) > 8:
                    self.objects.popitem(last=False)
        response.success, response.reused = True, reused
        response.status = 'ACTIVE_TARGET_REUSED' if reused else 'ACTIVE_TARGET_GROUNDED'
        response.object_id, response.memory_revision = record['object_id'], record['revision']
        response.position = Point(
            x=float(record['position'][0]),
            y=float(record['position'][1]),
            z=float(record['position'][2]),
        )
        response.observation_header = cloud.header
        response.image, response.observation_cloud = image, cloud
        response.camera_transform = observation.camera_transform
        response.mask, response.pixel_box = mask_message, record['pixel_box']
        response.object_cloud, response.geometry_source_header = (
            record['cloud'],
            record['geometry_header'],
        )
        return response

    def resolve(self, request, response):
        """Resolve one target while excluding concurrent memory mutation."""
        admitted = self.operation.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('ACTIVE_MEMORY_BUSY')
            return self._resolve(request, response)
        except Exception as error:
            response.status = str(error)
            return response
        finally:
            if admitted:
                self.operation.release()
            self.print_state('resolve_target')

    def spatial_context(self, request, response):
        """Read the authoritative spatial map and assign remembered locations to it."""
        try:
            spatial = self.call('get_spatial_map', srv.GetSpatialMap.Request())
            if not spatial.success:
                raise RuntimeError(spatial.status)
            response.map = spatial.map
            pose = getattr(self.node, 'current_tcp_pose', None)
            if pose is None:
                raise RuntimeError('ARM_STATE_UNAVAILABLE')
            response.tcp_pose = copy.deepcopy(pose)
            memory = self.snapshot()
            records = list(memory['directory'].values()) + memory['active_objects']
            if memory['active_destination']:
                records.append(memory['active_destination'])
            unique = {self.key(record['name']): record for record in records}
            for record in unique.values():
                position = list(map(float, record['position']))
                if not np.isfinite(position).all():
                    continue
                cell_id = ''
                point = np.asarray(position, dtype=np.float32)
                for cell in spatial.map.cells:
                    low = np.array([getattr(cell.minimum, k) for k in ('x', 'y', 'z')],
                                   dtype=np.float32)
                    high = np.array([getattr(cell.maximum, k) for k in ('x', 'y', 'z')],
                                    dtype=np.float32)
                    if np.all(point >= low) and np.all(point < high):
                        cell_id = cell.id
                        break
                response.object_names.append(record['name'])
                response.object_positions.append(Point(**dict(zip(('x', 'y', 'z'), position))))
                response.object_cell_ids.append(cell_id)
            response.success, response.status = True, 'SPATIAL_CONTEXT_READY'
        except Exception as error:
            response.status = str(error)
        return response

    def store_segmented(self, request, response):
        """Extract measured geometry from an existing mask without VLM inference."""
        admitted = self.operation.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('ACTIVE_MEMORY_BUSY')
            if (not request.image.data or not request.observation_cloud.data or
                    request.mask.header != request.image.header or
                    request.mask.encoding != 'mono8' or
                    (request.mask.height, request.mask.width) !=
                    (request.image.height, request.image.width)):
                raise ValueError('SEGMENTED_TARGET_OBSERVATION_MISMATCH')
            cloud, image, transform = (request.observation_cloud, request.image,
                                       request.camera_transform)
            if not request.name.strip() or request.role not in ('object', 'destination'):
                raise ValueError('TARGET_NAME_AND_ROLE_REQUIRED')
            if (cloud.header.frame_id != 'base_link' or cloud.height < 2 or
                    image.header.stamp != cloud.header.stamp or
                    (image.height, image.width) != (cloud.height, cloud.width) or
                    transform.header.frame_id != 'base_link' or
                    transform.header.stamp != cloud.header.stamp or
                    not all(np.isfinite(getattr(transform.transform.translation, k))
                            for k in ('x', 'y', 'z')) or
                    not np.isfinite(sum(getattr(transform.transform.rotation, k)**2
                                        for k in ('x', 'y', 'z', 'w'))) or
                    abs(sum(getattr(transform.transform.rotation, k)**2
                            for k in ('x', 'y', 'z', 'w'))-1.) > .001):
                raise ValueError('TARGET_OBSERVATION_ASSOCIATION_MISMATCH')
            mask = self.bridge.imgmsg_to_cv2(request.mask, 'mono8') > 0
            if np.count_nonzero(mask) < 80:
                raise ValueError('INSUFFICIENT_TARGET_MASK')
            query = srv.ResolveManipulationTarget.Request(
                name=request.name, role=request.role, refresh=True, image=request.image,
                observation_cloud=request.observation_cloud,
                camera_transform=request.camera_transform)
            result = self._resolve(
                query, srv.ResolveManipulationTarget.Response(), grounded_mask=mask)
            for key in response.get_fields_and_field_types():
                setattr(response, key, getattr(result, key))
            return response
        except Exception as error:
            response.status = str(error)
            return response
        finally:
            if admitted:
                self.operation.release()
            self.print_state('store_segmented_target')

    def get_target(self, request, response):
        """Read the active typed snapshot without refreshing it."""
        with self.lock:
            record = self._record(request.name, request.role or 'object')
            if not record:
                response.status = 'ACTIVE_TARGET_REQUIRED'
                return response
            observation = record['observation']
            if (request.observation_cloud.data and
                    observation.cloud.header != request.observation_cloud.header):
                response.status = 'TARGET_NOT_VISIBLE_IN_CURRENT_SCENE: look at target and recapture'
                return response
            response.success, response.reused = True, True
            response.status = 'ACTIVE_TARGET_SNAPSHOT_READY'
            response.object_id, response.memory_revision = record['object_id'], record['revision']
            response.observation_header = observation.cloud.header
            response.image, response.observation_cloud = observation.image, observation.cloud
            response.camera_transform = observation.camera_transform
            response.mask, response.pixel_box = record['mask'], record['pixel_box']
            response.object_cloud, response.geometry_source_header = (
                record['cloud'],
                record['geometry_header'],
            )
            response.position = Point(**dict(zip(('x', 'y', 'z'), map(float, record['position']))))
        return response

    def refresh(self, request, response):
        """Force inference using a fresh or explicitly supplied observation."""
        request = copy.copy(request)
        request.refresh = True
        return self.resolve(request, response)

    def forget(self, request, response):
        """Release active geometry while retaining its minimal location hint."""
        admitted = self.operation.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('ACTIVE_MEMORY_BUSY')
            self.guard_mutation()
            with self.lock:
                record = self._record(request.name, request.role or 'object')
                if record:
                    if request.role == 'destination':
                        self.destination = None
                    else:
                        self.objects.pop(record['key'], None)
                    self.revision += 1
            if record and request.role == 'destination':
                self.invalidate_destination()
            response.success, response.status = True, 'ACTIVE_GEOMETRY_RELEASED_HINT_RETAINED'
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self.operation.release()
            self.print_state('forget_active_target')
        return response

    def active(self, name):
        """Require a fresh source record before pose calculation."""
        with self.lock:
            record = self.objects.get(self.key(name))
        if not record:
            raise ValueError('ACTIVE_OBJECT_REQUIRED: resolve_target first')
        if (
            time.monotonic() - record['verified'] > 60
            or record['calibration'] != self.calibration()
        ):
            raise ValueError('ACTIVE_OBSERVATION_STALE: resolve_target to refresh')
        return record

    def grasps(self, request, response):
        """Calculate grasps on retained inputs without repeating inference."""
        admitted = self.operation.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('ACTIVE_MEMORY_BUSY')
            self.guard_mutation()
            record = self.active(request.target_description)
            observation = record['observation']
            pinned = self.call(
                'pin_grasp_observation',
                srv.PinLayoutObservation.Request(observation_header=observation.cloud.header),
            )
            if not pinned.success:
                raise RuntimeError('ACTIVE_OBSERVATION_UNAVAILABLE: refresh_active_target')
            prepared = self.call(
                'prepare_grasp_input',
                srv.PrepareGraspInput.Request(
                    observation_header=observation.cloud.header,
                    image=observation.image,
                    mask=record['mask'],
                    pixel_box=record['pixel_box'],
                    target_description=record['name'],
                ),
            )
            if not prepared.success:
                raise RuntimeError(prepared.status)
            query = srv.FindGraspCandidates.Request(
                observation_header=observation.cloud.header,
                seed_u=prepared.seed_u,
                seed_v=prepared.seed_v,
                image_width=prepared.image_width,
                image_height=prepared.image_height,
                target_mask=record['mask'],
                target_description=record['name'],
                affordance_type=request.affordance_type or prepared.affordance_type,
            )
            scene = grasp_scene(self, observation)
            if scene is not None:
                query.collision_scene = scene
            result = self.call('calculate_grasps', query)
            if result.success:
                with self.lock:
                    record['mask'] = result.validated_target_mask
                    # A grasp mask may represent only contact/refined geometry.
                    # Keep the semantic cloud immutable for later reobservation.
            return result
        except Exception as error:
            response.status = str(error)
            return response
        finally:
            if admitted:
                self.operation.release()
            self.print_state('calculate_active_grasps')

    def placements(self, request, response):
        """Search the verified destination using retained source geometry."""
        admitted = self.operation.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('ACTIVE_MEMORY_BUSY')
            if request.additional_obstacles.data:
                raise ValueError('ACTIVE_PLACEMENT_OVERLAY_UNSUPPORTED: use find_placements')
            record = self.active(request.object_name)
            observation = record['observation']
            target = self._resolve(
                srv.ResolveManipulationTarget.Request(
                    name=request.destination_name
                    or (self.destination['name'] if self.destination else ''),
                    role='destination',
                    image=observation.image,
                    observation_cloud=observation.cloud,
                    camera_transform=observation.camera_transform,
                ),
                srv.ResolveManipulationTarget.Response(),
            )
            destination = self.destination
            geometry = self.call(
                'prepare_held_scene',
                mapped_inputs(self, srv.PrepareHeldScene.Request(
                    observation_header=observation.cloud.header,
                    cloud=observation.cloud,
                    source_mask=record['mask'],
                    semantic_mask=record['semantic_mask'],
                    destination_mask=target.mask,
                ), destination['name']),
            )
            if not geometry.success:
                raise RuntimeError(geometry.status)
            check_capacity(self, geometry.source_object_cloud, destination['name'])
            # Explicit mapped-scene snapshot: original destination provenance
            # stays in geometry_source_header/memory, not relabeled as a capture.
            mapped_destination = copy.deepcopy(destination['cloud'])
            mapped_destination.header = copy.deepcopy(observation.cloud.header)
            bottom_cloud = type(geometry.source_object_cloud)()
            if 'bottom' in {f.name for f in geometry.source_object_cloud.fields}:
                flags = point_cloud2.read_points(
                    geometry.source_object_cloud, field_names=('bottom',), skip_nans=False
                )['bottom'].reshape(-1)
                bottom = xyz(geometry.source_object_cloud)[flags >= 2]
                if len(bottom) >= 3:
                    bottom_cloud = point_cloud2.create_cloud_xyz32(
                        observation.cloud.header, bottom
                    )
            search = self.call(
                'search_placements',
                srv.SearchPlacements.Request(
                    observation_header=observation.cloud.header,
                    object_name=request.object_name,
                    object_cloud=geometry.source_object_cloud,
                    scene_cloud=geometry.collision_scene,
                    destination_cloud=mapped_destination,
                    destination_base_cloud=destination['base_cloud'],
                    bottom_cloud=bottom_cloud,
                    orientation_policy=request.orientation_policy or 'upright',
                    support_mode=request.support_mode or 'floor_slots',
                    options_json=json.dumps({'candidate_yaw_variants': True}),
                ),
            )
            with self.lock:
                destination['grid'] = json.loads(search.diagnostics_json or '{}').get(
                    'slot_layout'
                )
                record['geometry'] = geometry
            response.success, response.status = search.success, search.status
            response.source_object_center, response.candidates = (
                geometry.source_object_center,
                search.candidates,
            )
        except Exception as error:
            response.status = str(error)
        finally:
            if admitted:
                self.operation.release()
            self.print_state('find_active_placements')
        return response

    def snapshot(self):
        """Expose bounded metadata without transporting dense geometry."""

        def small(record):
            """Extract compact metadata from one geometry record."""
            if not record:
                return None
            return {
                'name': record['name'],
                'position': record['position'],
                'object_id': record['object_id'],
                'memory_revision': record['revision'],
                'geometry_points': len(record['points']),
                'placed_objects': len(record.get('placements', ())),
                'placed_points': len(record.get('placed_points', ())),
                'placed_sources': [
                    {'name': item['name'], 'object_id': item['object_id'],
                     'source_stamp': [item['source_header'].stamp.sec,
                                      item['source_header'].stamp.nanosec]}
                    for item in record.get('placements', {}).values()],
                'grid': record.get('grid'),
                'grid_state': record.get('grid_state'),
            }

        with self.lock:
            return {
                'revision': self.revision,
                'directory': copy.deepcopy(dict(self.directory)),
                'active_objects': [small(r) for r in self.objects.values()],
                'active_destination': small(self.destination),
            }

    def guard_mutation(self):
        """Reject changes while the physical state is held or uncertain."""
        if (
            getattr(self.node, '_held_object', None) is not None
            or getattr(self.node, 'state', '') == 'STATE_UNKNOWN'
        ):
            raise RuntimeError('MEMORY_MUTATION_REQUIRES_KNOWN_EMPTY_HAND')
        session = getattr(getattr(self.node, 'execution_api', None), 'prepared_grasp_lift', None)
        if (
            session
            and session.record
            and session.record.get('phase') not in ('PREPARED', 'RELEASED')
        ):
            raise RuntimeError('MEMORY_MUTATION_BLOCKED_DURING_EXECUTION')

    def invalidate_destination(self):
        """Discard idle prepared geometry and old destination reservations."""
        session = getattr(getattr(self.node, 'execution_api', None), 'prepared_grasp_lift', None)
        if session and session.record and session.record.get('phase') == 'PREPARED':
            session.record = None
            session.publish_preview(None)
        result = self.call('commit_floor_slots', srv.CommitFloorSlots.Request(reset=True))
        if not result.success:
            raise RuntimeError('DESTINATION_GRID_RESET_FAILED: ' + result.status)

    def sync_grid(self):
        """Read current reservation metadata from its placement owner."""
        with self.lock:
            destination = self.destination
        if not destination or not destination.get('grid'):
            return
        result = self.call('get_floor_slots', srv.GetManipulationMemory.Request())
        if result.success:
            with self.lock:
                for entry in json.loads(result.state_json):
                    if entry['layout'] == destination['grid']:
                        destination['grid_state'] = entry
                        break

    def observe_result(self, name, request, response):
        """Update mapped metadata only after confirmed shared API results."""
        if name == 'search_placements' and response.diagnostics_json:
            with self.lock:
                if self.destination and bytes(request.destination_cloud.data) == bytes(
                    self.destination['cloud'].data
                ):
                    self.destination['grid'] = json.loads(response.diagnostics_json).get(
                        'slot_layout'
                    )
        if name == 'commit_floor_slots' and response.success:
            self.sync_grid()
        if name == 'release_placed_object' and response.success:
            session = self.node.execution_api.prepared_grasp_lift
            plan = session.record
            if not (plan and plan.get('pick_place') and
                    plan['key'] == request.object_name.casefold()):
                held = getattr(self.node.execution_api, 'held_motion', None)
                plan = held.record if held else None
            if not plan or plan.get('phase') != 'RELEASED':
                return
            remember_release(self, request, plan)

    def inspect(self, request, response):
        """Return active metadata and the actual pending and completed tasks."""
        try:
            self.sync_grid()
            warning = ''
        except Exception as error:
            warning = ';GRID_REFRESH_UNAVAILABLE: ' + str(error)
        state = self.snapshot()
        state['pipelines'] = json.loads(self.node.step_runtime.store.snapshot('')[0])
        response.success, response.status = True, 'MANIPULATION_MEMORY_READY' + warning
        response.state_json = json.dumps(state)
        return response

    def print_state(self, task):
        """Log names and coordinates alongside the current task."""
        state = self.snapshot()
        objects = [
            (r['name'], [round(v, 4) for v in r['position']]) for r in state['active_objects']
        ]
        destination = state['active_destination']
        dest = (
            None
            if destination is None
            else (destination['name'], [round(v, 4) for v in destination['position']])
        )
        self.node.get_logger().info(
            f'[PIPELINE STATE] task={task} active_objects={objects} destination={dest}'
        )
