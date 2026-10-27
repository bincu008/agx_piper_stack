"""Batched VLM inventory on one immutable synchronized RGB-D observation."""

import copy
import json
import math
from pathlib import Path
import re

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from my_actions import srv
import numpy as np
import yaml


def parse_mapping(text, keys):
    """Validate every requested key before changing memory."""
    begin, end = text.find('{'), text.rfind('}')
    if begin < 0 or end < begin:
        raise ValueError('SCENE_INFERENCE_JSON_MISSING')
    pairs = json.loads(text[begin:end+1], object_pairs_hook=lambda items: items)
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
    result = [min((width if i % 2 == 0 else height)-1,
                  math.floor(box[i]*(width if i % 2 == 0 else height)/1000+.5))
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
            # A grasp instruction qualifies an object, rather than adding another object.
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
        return names+[destination], ['object']*len(names)+['destination']

    def infer(self, observation, prompt, tokens):
        """Keep both VLM requests individually timed by the shared dispatcher."""
        result = self.memory.call('infer_vision', srv.VisionInference.Request(
            image=observation.image, user_prompt=prompt, max_tokens=tokens))
        if not result.success:
            raise RuntimeError('SCENE_INFERENCE_FAILED: '+result.status)
        self.memory.node.get_logger().info('[SCENE INFERENCE] '+json.dumps(
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
                            for k in ('x', 'y', 'z', 'w'))-1) > .001):
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
                observation, presence_prompt, max(128, 24*len(names))), keys)
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
                observation, grounding_prompt, max(512, 80*len(names))), keys)
            pixels = {key: pixel_box(boxes[key], image.width, image.height)
                      for key in keys if presence[key] and boxes[key] is not None}
            response.observation_header = copy.deepcopy(cloud.header)
            response.names, response.roles = names, roles
            for key, name, role in zip(keys, names, roles):
                record = memory._record(name, role)
                observed, status = False, 'NOT_VISIBLE_IN_CURRENT_SCENE'
                if role == 'destination' and record is not None:
                    # Frozen base/grid survive looking away or placing objects.
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
                        status = 'SCENE_SEGMENTATION_FAILED: '+segmented.status
                record = memory._record(name, role)
                hint = next((entry for entry in memory.directory.values()
                             if memory.key(entry['name']) == memory.key(name)), None)
                position = (record['position'] if record else
                            hint['position'] if hint else [0., 0., 0.])
                if role == 'object' and not observed:
                    # Keep the directory hint; never grasp an unobserved old cloud.
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
