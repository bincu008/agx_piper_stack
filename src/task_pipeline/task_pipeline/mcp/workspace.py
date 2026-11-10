"""Opt-in read-only current-workspace tools; never dispatch physical operations."""

import base64
import copy
import json
import math

from mcp import types

from .tasks import OUTPUT


NAMES = ('get_camera_image', 'list_objects', 'check_freshness')


def definitions():
    """Publish small flat schemas independently of diagnostic contracts."""
    descriptions = {
        'get_camera_image': 'Read the latest eligible synchronized wrist-camera image and '
        'source pose/time from the automatic idle scan. Returns actual MCP ImageContent. '
        'No inference or motion. REFRESH_REQUIRED means capture is not yet usable; '
        'stored pixels do not establish current object presence or execution readiness.',
        'list_objects': 'Read known object/destination names and positions with independent '
        'geometry/verification ages. Memory is not fresh visual evidence. Call observe_scene '
        'to refresh perception. Missing entries mean UNKNOWN, not absent.',
        'check_freshness': 'Read a Boolean fresh result for the current image, named object, '
        'single current grasp pool, or prepared plan. '
        'No capture, inference, motion or uncertainty reset. Missing/unverified evidence is '
        'false. Source age and existing shorter backend checks both apply. '
        'Execution readiness is not certified.'}
    descriptions['get_camera_image'] += (' If stale, request at most one bounded synchronized '
        'scan while idle and stationary. Scan acknowledgement is not image completion. '
        'Never returns stale pixels as fresh; no VLM or motion.')
    for name in NAMES:
        schema = {'type': 'object', 'properties': {}, 'additionalProperties': False}
        if name == 'check_freshness':
            schema['properties'] = {
                'resource': {'type': 'string', 'enum': ['image', 'object', 'grasp_pool', 'plan'],
                             'description': 'image uses current RGB-D metadata; object requires '
                             'object_name; grasp_pool checks single current grasp pool; '
                             'plan checks current prepared plan. This does not authorize execution.'},
                'object_name': {'type': 'string', 'minLength': 1, 'maxLength': 256,
                                'description': 'Exact remembered name, required for object.'}}
            schema['required'] = ['resource']
        yield types.Tool(name=name, description=descriptions[name], inputSchema=schema,
                         outputSchema=copy.deepcopy(OUTPUT), annotations=types.ToolAnnotations(
                             readOnlyHint=name != 'get_camera_image', destructiveHint=False,
                             idempotentHint=name != 'get_camera_image',
                             openWorldHint=True), _meta={'physicalDispatch': False})


def encode_image(image):
    """Bound and encode pixels for MCP image content, never textual tool arguments."""
    if not image.width or not image.height or not image.data:
        return None
    if (image.width > 4096 or image.height > 4096 or len(image.data) > 16*1024*1024 or
            image.encoding not in ('bgr8', 'rgb8')):
        raise ValueError('UNSUPPORTED_OR_OVERSIZED_WORKSPACE_IMAGE')
    import cv2
    try:
        from cv_bridge import CvBridge
        frame = CvBridge().imgmsg_to_cv2(image, desired_encoding='bgr8')
    except (ImportError, ModuleNotFoundError):
        import numpy as np
        frame = np.frombuffer(bytes(image.data), dtype=np.uint8).reshape((image.height, image.width, -1))
        if image.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    scale = min(1., 960./max(image.width, image.height))
    if scale < 1:
        frame = cv2.resize(frame, (round(image.width*scale), round(image.height*scale)))
    for _ in range(3):
        ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError('WORKSPACE_IMAGE_ENCODING_FAILED')
        if len(data) <= 512*1024:
            return {'data': base64.b64encode(data).decode(), 'mimeType': 'image/jpeg',
                    'width': frame.shape[1], 'height': frame.shape[0]}
        frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    raise ValueError('WORKSPACE_IMAGE_TOO_LARGE')


class WorkspaceTools:
    """Read the ROS-owned workspace; never maintain a second state or calculation job."""

    def __init__(self, bridge, gateway):
        self.bridge, self.gateway = bridge, gateway

    async def call(self, name, arguments):
        if name == 'check_freshness':
            resource = arguments['resource']
            if resource == 'object' and not arguments.get('object_name'):
                raise ValueError('OBJECT_NAME_REQUIRED')
            if resource in ('image', 'grasp_pool', 'plan') and 'object_name' in arguments:
                raise ValueError(f'{resource.upper()}_FRESHNESS_REJECTS_OBJECT_NAME')
        image_request = name == 'get_camera_image'
        read = await self.bridge.workspace(include_image=image_request)
        if read.get('transport_outcome') != 'response':
            return {'success': False, 'state': 'UNKNOWN', 'status': read.get('detail'),
                    'execution_allowed': False}
        response = read['response']
        state = json.loads(response['state_json'])
        result = {'success': True, 'state': 'COMPLETED', 'status': response['status'],
                  'source_mode': state['source_mode'], 'execution_allowed': False}
        if image_request:
            media = response.pop('_image_content', None)
            if not state['freshness']['fresh'] or not media:
                return dict(result, success=False, state='BLOCKED', result=state)
            result['result'] = {'workspace': state, 'observation_header':
                                response['observation_header'], 'camera_transform':
                                response['camera_transform'], 'source_tcp': response['source_tcp'],
                                'rendered_width': media.pop('width'),
                                'rendered_height': media.pop('height')}
            result['_image_content'] = media
            return result
        if name == 'check_freshness' and arguments['resource'] == 'image':
            fresh = state['freshness']['fresh']
            evidence = dict(state['freshness'])
            if self.gateway.execution.load('preparation_barrier'):
                fresh = False
                evidence.update(fresh=False, reason='OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION')
            return dict(result, fresh=fresh, status='FRESHNESS_READ',
                        result={'fresh': fresh, 'evidence': evidence})
        if name == 'check_freshness' and arguments['resource'] == 'grasp_pool':
            barrier = self.gateway.execution.load('preparation_barrier')
            pool = state.get('current_pool')
            if barrier:
                return dict(result, fresh=False, status='FRESHNESS_READ',
                            result={'fresh': False, 'reason': 'OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION',
                                    'pool': pool})
            if not pool or not pool.get('available'):
                reason = pool.get('reason', 'NO_CURRENT_GRASP_POOL') if pool else 'NO_CURRENT_GRASP_POOL'
                return dict(result, fresh=False, status='FRESHNESS_READ',
                            result={'fresh': False, 'reason': reason, 'pool': pool})
            fresh = bool(pool.get('fresh', False) and not pool.get('consumed', False))
            reason = 'FRESH_FOR_REUSE_ONLY' if fresh else pool.get('reason', 'POOL_NOT_FRESH')
            return dict(result, fresh=fresh, status='FRESHNESS_READ',
                        result={'fresh': fresh, 'reason': reason, 'pool': pool})
        if name == 'check_freshness' and arguments['resource'] == 'plan':
            barrier = self.gateway.execution.load('preparation_barrier')
            plan = state.get('current_plan')
            if barrier:
                return dict(result, fresh=False, status='FRESHNESS_READ',
                            result={'fresh': False, 'reason': 'OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION',
                                    'plan': plan})
            if not plan or not plan.get('available'):
                reason = plan.get('reason', 'NO_CURRENT_PREPARED_PLAN') if plan else 'NO_CURRENT_PREPARED_PLAN'
                return dict(result, fresh=False, status='FRESHNESS_READ',
                            result={'fresh': False, 'reason': reason, 'plan': plan})
            fresh = bool(plan.get('fresh', False) and plan.get('phase') == 'PREPARED')
            reason = 'FRESH_FOR_REUSE_ONLY' if fresh else plan.get('reason', 'PLAN_NOT_FRESH')
            return dict(result, fresh=fresh, status='FRESHNESS_READ',
                        result={'fresh': fresh, 'reason': reason, 'plan': plan})
        read = await self.bridge.call('get_manipulation_memory', {})
        if read.get('transport_outcome') != 'response' or not read['response'].get('success'):
            return dict(result, success=False, state='UNKNOWN', status='MEMORY_UNAVAILABLE')
        memory = json.loads(read['response']['state_json'])
        objects = list(memory['active_objects'])
        if memory.get('active_destination'):
            objects.append(memory['active_destination'])
        fields = {'name', 'position', 'geometry_age_seconds', 'verification_age_seconds',
                  'calibration_matches', 'geometry_source_stamp', 'placed_objects'}
        objects = [dict({k: v for k, v in obj.items() if k in fields},
                        evidence='remembered', visible_now='UNKNOWN') for obj in objects]
        if name == 'list_objects':
            return dict(result, status='OBJECT_MEMORY_READ', result={'objects': objects,
                        'frame': 'base_link', 'position_units': 'metres',
                        'directory': list(memory.get('directory', {}).values()),
                        'fresh_perception_performed': False})
        matches = [obj for obj in objects if obj['name'].casefold() ==
                   arguments['object_name'].casefold()]
        fresh = False
        if len(matches) == 1:
            obj = matches[0]
            ages = (obj.get('geometry_age_seconds'), obj.get('verification_age_seconds'))
            fresh = (all(isinstance(age, (int, float)) and math.isfinite(age) and age >= 0
                         for age in ages) and ages[0] < 600 and ages[1] < 60 and
                     obj.get('calibration_matches') is True and
                     not self.gateway.execution.load('preparation_barrier'))
        reason = ('EXISTING_GEOMETRY_AND_VERIFICATION_LIMITS' if fresh else
                  'OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION' if
                  self.gateway.execution.load('preparation_barrier') else
                  'MISSING_AMBIGUOUS_OR_STALE_OBJECT')
        return dict(result, status='FRESHNESS_READ', fresh=bool(fresh),
                    result={'fresh': bool(fresh), 'reason': reason, 'matches': len(matches)})
