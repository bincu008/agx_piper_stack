"""Master-owned, bounded target geometry and reusable destination memory."""

import copy
import json
import threading
import time
from collections import OrderedDict

from cv_bridge import CvBridge

from geometry_msgs.msg import Point

from my_actions import srv

import numpy as np

from rclpy.callback_groups import ReentrantCallbackGroup

from scipy.spatial import cKDTree

from sensor_msgs_py import point_cloud2


from .target_view import check_target_view
from .scene_perception import ScenePerception
from .perception_acquisition import PerceptionAcquisition
from .destination_memory import mapped_inputs, remember_release, check_capacity, grasp_scene


def xyz(cloud, organized=False):
    """Read existing manual-base-frame geometry without applying another transform."""
    fields = point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=False)
    points = np.column_stack([fields[name].reshape(-1) for name in ('x', 'y', 'z')])
    return points.reshape(cloud.height, cloud.width, 3) if organized else points


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
            (srv.GetManipulationMemory, 'get_manipulation_memory', self.inspect),
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
