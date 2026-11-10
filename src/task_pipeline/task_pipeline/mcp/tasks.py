"""Task-sized, non-actuating facades over existing recipes and queue receipts."""

import asyncio
import copy
import hashlib
import json
import math
import re
import time

import jsonschema
from mcp import types

from .examples import workflow_example
from .preparation import IDENTIFIER, PREPARATION


STATES = ['QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'BLOCKED',
          'CANCEL_PENDING', 'CANCELLED', 'UNKNOWN']
OPERATIONS = ['grasp', 'grasp_lift', 'pick_place']
NAMES = ['inspect_robot', 'observe_scene', 'plan_camera_view', 'prepare_manipulation',
         'validate_plan', 'get_task_status', 'cancel_task', 'describe_workflow']
TERMINAL = {'COMPLETED', 'FAILED', 'BLOCKED', 'CANCELLED', 'UNKNOWN'}
OUTPUT = {'type': 'object', 'required': ['success', 'state', 'status'],
          'properties': {'success': {'type': 'boolean'}, 'state': {'enum': STATES},
                         'status': {'type': 'string'}, 'task_id': {'type': 'string'},
                         'result': {'type': 'object'}, 'blocker': {'type': 'object'},
                         'plan': {'type': 'object'}, 'source_mode': {'type': 'string'},
                         'current_step': {'type': ['string', 'null']},
                         'current_step_id': {'type': ['string', 'null']},
                         'last_successful_step': {'type': ['string', 'null']},
                         'failed_step': {'type': ['string', 'null']},
                         'execution_allowed': {'const': False},
                         'validation': {'type': 'object', 'required': [
                             'source_mode', 'checks_passed', 'checks_bypassed',
                             'execution_allowed', 'outcome', 'scene_freshness_confirmed',
                             'physical_readiness_confirmed'], 'properties': {
                                 'source_mode': {'type': ['string', 'null']},
                                 'checks_passed': {'type': 'array', 'items': {'type': 'string'}},
                                 'checks_bypassed': {'type': 'array', 'items': {
                                     'type': 'object', 'required': ['check', 'reason'],
                                     'properties': {'check': {'type': 'string'},
                                                    'reason': {'type': ['string', 'null']}}}},
                                 'execution_allowed': {'const': False},
                                 'outcome': {'type': 'string'},
                                 'scene_freshness_confirmed': {'const': False},
                                 'physical_readiness_confirmed': {'const': False}}},
                         'SIMULATED': {'type': 'boolean'}, 'input_mode': {'type': 'string'},
                         'input_provenance': {'type': 'object'}}, 'additionalProperties': True}


def definitions():
    """Return the separate eight-tool Nanobot profile; no generic calls are inputs."""
    text = {'type': 'string', 'minLength': 1, 'maxLength': 256}
    wait = {'type': 'number', 'minimum': 0, 'maximum': 120, 'default': 0,
            'description': 'Optional bounded wait for this existing calculation; no polling loop.'}
    req = dict(IDENTIFIER, description='Unique ID for a NEW task. Retry identical inputs with '
               'this same ID to find the same task; never replay unknown work with a new ID.')
    props = {
        'inspect_robot': {},
        'observe_scene': {'request_id': req, 'targets': {'type': 'array', 'minItems': 1,
                          'maxItems': 15, 'uniqueItems': True, 'items': text},
                          'detail': {'enum': ['presence', 'location', 'mask'],
                                     'default': 'location'},
                          'destination': text, 'wait_seconds': wait},
        'prepare_manipulation': {'request_id': req, 'operation': {'enum': OPERATIONS},
                                 'object_name': text, 'destination': text,
                                 'lift_distance_m': {'type': 'number', 'exclusiveMinimum': 0,
                                                     'maximum': .30, 'default': .05},
                                 'wait_seconds': wait},
        'plan_camera_view': {'request_id': req, 'mode': {
            'type': 'string', 'enum': ['relative', 'target'],
            'description': 'relative: supply frame and both vectors, no target selectors. '
            'target: supply exactly one target_name or target_cell_id, no relative fields.'},
            'translation_m': {'type': 'array', 'minItems': 3, 'maxItems': 3,
                              'items': {'type': 'number', 'minimum': -.20, 'maximum': .20},
                              'description': 'Required in relative mode only: camera-optical '
                              '[right,down,forward] metres; vector norm <=0.20m. '
                              'Use [0,0,0] for rotation only.'},
            'rotation_rad': {'type': 'array', 'minItems': 3, 'maxItems': 3,
                             'items': {'type': 'number', 'minimum': -math.pi,
                                       'maximum': math.pi},
                             'description': 'Required in relative mode only: '
                             '[roll,pitch,yaw] radians, each within +/-pi. '
                             'Use [0,0,0] for translation only.'},
            'frame': {'type': 'string', 'enum': ['camera_optical'],
                      'description': 'Required in relative mode: camera_optical. '
                      'Forbidden in target mode.'},
            'target_name': dict(text, description='Target mode only: exact remembered target '
                                'name. Supply this OR target_cell_id, never both.'),
            'target_cell_id': dict(text, description='Target mode only: existing spatial cell '
                                   'ID. Supply this OR target_name, never both.'),
            'camera_distance_m': {'type': 'number', 'minimum': .1, 'maximum': 1.,
                                  'default': .45, 'description': 'Optional target-view distance '
                                  'in metres; used only in target mode.'},
            'wait_seconds': wait},
        'validate_plan': {'request_id': req, 'plan_id': text,
                          'revision': {'type': 'integer', 'minimum': 1}},
        'get_task_status': {'task_id': text, 'wait_seconds': wait,
                            'include_diagnostics': {'type': 'boolean', 'default': False}},
        'cancel_task': {'task_id': text},
        'describe_workflow': {'operation': {'enum': ['observe_scene', *OPERATIONS,
                                                     'plan_camera_view']}}}
    required = {'inspect_robot': [], 'observe_scene': ['request_id', 'targets'],
                'prepare_manipulation': ['request_id', 'operation', 'object_name'],
                'plan_camera_view': ['request_id', 'mode'],
                'validate_plan': ['request_id', 'plan_id', 'revision'],
                'get_task_status': [], 'cancel_task': [],
                'describe_workflow': ['operation']}
    descriptions = {
        'inspect_robot': 'Read compact current feedback, independent timestamps and state. '
        'No scene inference. Feedback ages are evidence, not a motion-readiness certificate.',
        'observe_scene': 'Observe named targets: server owns fresh capture, exact retention and '
        'same-context perception. presence uses verify_presence; location/mask resolves geometry; '
        'destination enables batch perception. Returns task_id and explicit state; RUNNING/QUEUED '
        'are NOT success. get_task_status retrieves the result. UNKNOWN is not absent.',
        'prepare_manipulation': 'Calculate only. grasp generates/selects/caches the full pool '
        'and confirms cache; grasp_lift additionally prepares/validates lift; pick_place requires '
        'destination and prepares/validates complete placement. Server owns all prerequisites. '
        'No physical execution. RUNNING requires get_task_status. Plan ID/revision exist only '
        'after an actual prepared plan is created. Retry same request_id to inspect same task.',
        'plan_camera_view': 'Calculate only; never move. relative requires frame=camera_optical, '
        'translation_m=[right,down,forward], rotation_rad=[roll,pitch,yaw]. Delegates loose '
        'get_observe_view endpoint checking; PARTIAL/fallback does not prove a safe full path. '
        'target requires exactly one target_name or target_cell_id, rejects frame/both vectors, '
        'and uses plan_next_best_view. relative rejects target selectors. Supply request_id '
        'and mode on every call. Read result.status and achieved_fraction: '
        'NO_REACHABLE_CHANGE with 0.0 means no changed viewpoint was found.',
        'validate_plan': 'Validate this exact server-issued plan_id/revision against current '
        'prepared source/feedback; stale plans block. No motion or cache consumption. '
        'A historical successful task is not current plan validity.',
        'get_task_status': 'Default to the current task when task_id is omitted. Read the same '
        'calculation task and original blocker/results; optional '
        'bounded wait. Never starts/replays work. After restart/disconnect unknown outcomes stay '
        'unknown; diagnostic mode reveals the existing ROS context/job and producer receipts.',
        'cancel_task': 'Default to the current task when task_id is omitted. Request cancellation '
        'of the existing calculation action between steps. '
        'CANCEL_PENDING is not confirmed cancellation and cannot interrupt an in-flight ROS '
        'service. It is not robot stop. Completed/unknown tasks are never replayed.',
        'describe_workflow': 'Local read-only description of a supported route, inputs, '
        'prerequisites, freshness and outputs. No robot or scene calls. No internal IDs required.'}
    tools = []
    for name in NAMES:
        schema = {'type': 'object', 'properties': props[name], 'required': required[name],
                  'additionalProperties': False}
        if name == 'prepare_manipulation':
            schema['allOf'] = [{'if': {'properties': {'operation': {'const': 'pick_place'}}},
                                'then': {'required': ['destination']}}]
        tools.append(types.Tool(name=name, description=descriptions[name], inputSchema=schema,
                                outputSchema=copy.deepcopy(OUTPUT),
                                annotations=types.ToolAnnotations(
                                    readOnlyHint=name in ('inspect_robot', 'get_task_status',
                                                          'describe_workflow'),
                                    destructiveHint=False, idempotentHint=True,
                                    openWorldHint=name not in ('describe_workflow',)),
                                _meta={'taskInterface': True, 'physicalDispatch': False}))
    return tools



def gaze_labels(forward):
    """Describe the C++ camera-forward vector in base_link; visualization only."""
    values = [forward.get(k) for k in 'xyz']
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return {'status': 'UNKNOWN'}
    x, y, z = values
    if not .9 <= math.sqrt(x*x+y*y+z*z) <= 1.1:
        return {'status': 'UNKNOWN'}
    return {'status': 'DIRECTION_READ', 'frame': 'base_link',
            'horizontal': 'ambiguous_vertical_view' if abs(z) > .9 else
            'left' if y > .25 else 'right' if y < -.25 else 'centered',
            'vertical': 'up' if z > .25 else 'down' if z < -.25 else 'level',
            'fore_aft': 'forward' if x > .25 else 'backward' if x < -.25 else
            'sideways_or_vertical',
            'meaning': 'Base axes only; these labels do not establish visibility.'}


def validate_camera_view(arguments):
    """Enforce mode contracts before any task identity, receipt or ROS work is created."""
    schema = next(tool.inputSchema for tool in definitions() if tool.name == 'plan_camera_view')
    jsonschema.validate(arguments, schema)
    relative = {'frame', 'translation_m', 'rotation_rad'}
    selectors = {'target_name', 'target_cell_id'}
    if arguments['mode'] == 'relative':
        missing = relative - arguments.keys()
        if missing:
            raise ValueError('RELATIVE_VIEW_FIELDS_REQUIRED: ' + ', '.join(sorted(missing)))
        if selectors & arguments.keys():
            raise ValueError('RELATIVE_VIEW_REJECTS_TARGET_SELECTORS')
        translation, rotation = arguments['translation_m'], arguments['rotation_rad']
        if (not all(math.isfinite(v) for v in translation + rotation) or
                math.hypot(*translation) > .20 or any(abs(v) > math.pi for v in rotation)):
            raise ValueError('VIEW_DELTA_REQUIRES_FINITE_0.20M_AND_PI_RAD_BOUNDS')
    else:
        if len(selectors & arguments.keys()) != 1:
            raise ValueError('TARGET_VIEW_REQUIRES_EXACTLY_ONE_TARGET_NAME_OR_CELL_ID')
        if relative & arguments.keys():
            raise ValueError('TARGET_VIEW_REJECTS_RELATIVE_FIELDS')
        if not math.isfinite(arguments.get('camera_distance_m', .45)):
            raise ValueError('TARGET_VIEW_DISTANCE_MUST_BE_FINITE')


# These are explanations of existing backend markers, not new policy decisions.
BYPASS_REASONS = {
    'LIFT_ROBOT_ENVIRONMENT_CHECK_BYPASSED':
        'Backend placement_bypass_robot_environment_during_lift=true skips robot/finger '
        'versus scene checking during lift.',
    'LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED':
        'Backend placement_bypass_attached_collision_during_lift=true skips attached-object '
        'versus environment checking during lift.',
    'WITHDRAWAL_ROBOT_ENVIRONMENT_CHECK_BYPASSED':
        'Backend placement_bypass_robot_environment_during_lift=true also skips '
        'robot versus scene checking during empty-hand withdrawal.',
    'FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED':
        'Backend final_approach_check_finger_environment=false skips final-approach '
        'finger-body/pad versus scene checks; other configured checks remain active.'}


def validation_summary(record, rows):
    """Expose only backend-reported evidence; never infer unreported checks passed."""
    passed, bypassed = set(), set()
    prepared = False
    if (not rows and record.get('kind') == 'validate_plan' and
            (record.get('result') or {}).get('validated') is True):
        # Older direct validation receipts kept status, not rows. Preserve that evidence.
        kind = (record.get('plan') or {}).get('kind')
        name = {'prepare_pick_place': 'validate_prepared_pick_place',
                'prepare_grasp_lift': 'validate_prepared_grasp_lift'}.get(kind)
        rows = [{'success': True, 'function_name': name, 'status': record['status']}]
    for row in rows:
        output = row.get('output') or {}
        messages = [row.get('status', ''), output.get('status', ''),
                    output.get('motion_status', '')]
        text = ';'.join(str(v) for v in messages)
        markers = set(re.findall(r'\b[A-Z][A-Z0-9_]+\b', text))
        bypassed.update(m for m in markers if m.endswith('_BYPASSED'))
        if row.get('success'):
            passed.update(m for m in markers if m.endswith('_VALIDATED') or
                          m == 'CACHED_HANDOFF_VALIDATED_NO_ACTUATION')
            prepared |= row.get('function_name') in (
                'prepare_grasp_lift', 'prepare_pick_place',
                'validate_prepared_grasp_lift', 'validate_prepared_pick_place')
    succeeded = record['state'] == 'COMPLETED' and prepared
    return {'source_mode': record.get('source_mode'),
            'backend_validation_succeeded': succeeded,
            'outcome': ('PASSED_WITH_BYPASSES' if bypassed else 'PASSED_REPORTED_CHECKS')
            if succeeded else 'UNKNOWN' if record['state'] == 'UNKNOWN' else 'NOT_VALIDATED',
            'checks_passed': sorted(passed),
            'checks_bypassed': [{'check': m, 'reason': BYPASS_REASONS.get(m),
                                'reason_source': 'backend_marker_contract' if m in
                                 BYPASS_REASONS else 'not_reported'} for m in sorted(bypassed)],
            'scope': 'Only backend-reported checks that ran; bypasses are not passes. '
                     'No comprehensive scene-collision guarantee is inferred.',
            'execution_allowed': False, 'execution_block_reason': (
                'PREPARATION_ONLY_MODE' if record.get('preparation_only', True) else
                'CALCULATION_RESULT_NOT_EXECUTION_AUTHORIZATION'),
            'historical_result_not_current_readiness': True,
            'scene_freshness_confirmed': False, 'physical_readiness_confirmed': False}


def route_labels(steps, destination=None):
    """Label semantic roles from the existing arguments/bindings without changing routes."""
    result, roles, current_role = [], {}, 'object'
    for index, step in enumerate(steps):
        name, arguments = step['function_name'], step['arguments']
        if name == 'verify_presence':
            # Destination presence is named exactly in both reviewed route families.
            current_role = 'destination' if (destination is not None and
                                             arguments.get('target') == destination) else 'object'
        role = current_role if name in ('verify_presence', 'ground_object',
                                        'segment_object', 'store_segmented_target') else 'shared'
        if arguments.get('role') in ('object', 'destination'):
            role = arguments['role']
        if name == 'segment_object':
            source = next((b['source_step_id'] for b in step['bindings']
                           if b['target_field'] == 'bbox'), None)
            role = roles.get(source, role)
        if name == 'capture_observation':
            role = 'scene'
        roles[step['step_id']] = role
        result.append({'index': index+1, 'function': name, 'role': role,
                       'label': name + ' (' + role + ')'})
    return result


def failure(status, step=None):
    """Keep original blocker wording and give a bounded, non-replaying recovery."""
    hint = ('Inspect this task and diagnostic receipts. Reconcile locally if outcome is '
            'unknown; never replay it. A definite stale-input failure requires a NEW '
            'fresh-observation task, not a getter.')
    if 'REQUEST_ID_REUSED' in status:
        hint = 'Reuse an ID only for identical inputs; a NEW operation needs a fresh request_id.'
    elif 'DEPENDENCY' in status or 'CONTEXT' in status:
        hint = ('An internal same-context producer is missing/failed. Inspect task diagnostics; '
                'do not supply bindings yourself. Start a NEW task only after a definite failure.')
    elif 'TARGET' in status and any(s in status for s in ('UNKNOWN', 'REQUIRED', 'LOCATION')):
        hint = 'Observe the named target first, or choose a known spatial cell for view planning.'
    elif 'SCENE_INFERENCE_UNKNOWN' in status or 'PRESENCE_UNKNOWN' in status:
        hint = ('Perception did not establish the target. This is not absence. Check the image '
                'or lighting and request a NEW observation; preserve the failed task.')
    refresh = ('STALE', 'EXPIRED', 'NOT_IN_CACHE', 'SOURCE_CHANGED',
               'REVISION_CHANGED', 'FEEDBACK')
    return {'step': step, 'status': status,
            'refresh_required': any(s in status for s in refresh),
            'recovery': hint}


class TaskInterface:
    """Store task aliases in the existing ledger; ROS/gateway own all calculation jobs."""

    def __init__(self, gateway, bridge):
        self.gateway, self.bridge, self.ledger = gateway, bridge, gateway.execution
        self.direct_active = set()
        for key, raw in self.ledger.db.execute(
                "SELECT id,data FROM execution WHERE id LIKE 'task:%'").fetchall():
            record = json.loads(raw)
            if record['state'] not in TERMINAL and not self.ledger.load(
                    'job:' + record.get('job_id', '')):
                record.update(state='UNKNOWN', status='TASK_ADMISSION_UNKNOWN_AFTER_RESTART')
                self.save(record)
                if record['kind'] in ('move_camera', 'control_gripper', 'execute_manipulation'):
                    execution = self.ledger
                    execution.save('preparation_barrier', {'control_task': record['task_id']})

    def source(self):
        info = self.bridge.input_provenance()
        return {'source_mode': ('mock_replay' if info['classification'] ==
                                'mock_replay_publishers' else 'live' if
                                info['classification'] == 'live_driver_publishers' else 'unknown'),
                'input_provenance': info}

    def load(self, task_id):
        record = self.ledger.load('task:' + task_id)
        if record is None:
            raise ValueError('TASK_NOT_FOUND: use the task_id returned by submission')
        return record

    def save(self, record):
        record.setdefault('preparation_only', self.gateway.require_physical_disabled)
        record['updated_at'] = time.time()
        self.ledger.save('task:' + record['task_id'],
                         {key: value for key, value in record.items() if key != 'rows'})

    @staticmethod
    def envelope(name, step_id, arguments=None, bindings=()):
        return {'function_name': name, 'step_id': step_id, 'arguments': arguments or {},
                'bindings': [{'target_field': target, 'source_step_id': source,
                              'source_field': field} for target, source, field in bindings]}

    def route(self, kind, args):
        if kind == 'prepare_manipulation':
            operation = args['operation']
            if operation not in OPERATIONS:
                raise ValueError('UNSUPPORTED_MANIPULATION_OPERATION')
            example = workflow_example('pick_place' if operation == 'pick_place' else
                                       'grasp_cache', args['object_name'],
                                       destination_name=args.get('destination', 'tray'))
            steps = [dict(item['input'], function_name=item['tool']) for item in example['steps']
                     if item['tool'] in PREPARATION]
            for item in steps:
                for field in ('context_id', 'request_id'):
                    item.pop(field, None)
                if item['function_name'] == 'prepare_pick_place':
                    item['arguments']['lift_distance_m'] = args.get('lift_distance_m', .05)
            if operation == 'pick_place':
                # Keep source/destination geometry in the existing master memory before
                # planning. World-name registration alone cannot support release bookkeeping.
                capture_id = next(s['step_id'] for s in steps if
                                  s['function_name'] == 'capture_observation')
                mask_sources = {}
                for item in steps:
                    if item['function_name'] == 'prepare_grasp_input':
                        mask_sources['object'] = next(b['source_step_id'] for b in
                            item['bindings'] if b['target_field'] == 'mask')
                    if item['function_name'] == 'prepare_pick_place':
                        mask_sources['destination'] = next(b['source_step_id'] for b in
                            item['bindings'] if b['target_field'] == 'destination_mask')
                augmented = []
                for item in steps:
                    # store_segmented_target already upserts the observed object center.
                    # A grasp's seed/contact point is not a second object observation.
                    if item['function_name'] == 'update_object':
                        continue
                    augmented.append(item)
                    for role, producer in mask_sources.items():
                        if item['step_id'] == producer:
                            augmented.append(self.envelope(
                                'store_segmented_target', 'remember_' + role,
                                {'name': args['object_name'] if role == 'object' else
                                 args['destination'], 'role': role}, [
                                    ('image', capture_id, 'image'),
                                    ('mask', producer, 'mask'),
                                    ('observation_cloud', capture_id, 'cloud'),
                                    ('camera_transform', capture_id, 'camera_transform')]))
                steps = augmented
                index = next(i for i, item in enumerate(steps)
                             if item['function_name'] == 'cache_grasp_plan')
                steps.insert(index+1, self.envelope('get_cached_grasp', 'confirm_cache', {
                    'object_name': args['object_name']}))
            if operation == 'grasp_lift':
                steps += [self.envelope('prepare_grasp_lift', 'prepare_motion', {
                    'object_name': args['object_name'],
                    'lift_distance_m': args.get('lift_distance_m', .05)})]
            if operation != 'grasp':
                steps += [self.envelope('validate_prepared_pick_place' if operation ==
                                        'pick_place' else 'validate_prepared_grasp_lift',
                                        'validate_motion', {'object_name': args['object_name']})]
            return steps
        if kind == 'plan_camera_view':
            if args['mode'] == 'relative':
                values = dict(zip(('dx', 'dy', 'dz', 'roll', 'pitch', 'yaw'),
                                  args['translation_m'] + args['rotation_rad']))
                return [self.envelope('get_observe_view', 'relative_view', values)]
            return [self.envelope('plan_next_best_view', 'target_view', {
                k: v for k, v in args.items() if k in ('target_name', 'target_cell_id',
                                                       'camera_distance_m')})]
        if kind != 'observe_scene':
            raise ValueError('UNSUPPORTED_TASK_OPERATION')
        targets = args['targets']
        if args.get('destination') in targets:
            raise ValueError('TARGET_AND_DESTINATION_MUST_DIFFER')
        capture = self.envelope('capture_observation', 'capture')
        if args.get('destination') and args.get('detail', 'location') == 'location':
            return [capture, self.envelope('perceive_scene', 'scene', {
                'object_names': targets, 'destination_name': args['destination']},
                [('image', 'capture', 'image'), ('observation_cloud', 'capture', 'cloud'),
                 ('camera_transform', 'capture', 'camera_transform')])]
        steps = [capture, self.envelope('pin_grasp_observation', 'pin', bindings=[
            ('observation_header', 'capture', 'cloud.header')])]
        names = [(name, 'object') for name in targets]
        if args.get('destination'):
            names.append((args['destination'], 'destination'))
        for index, (name, role) in enumerate(names):
            key = 'target_' + str(index)
            steps.append(self.envelope('verify_presence', key + '_presence', {'target': name}, [
                ('image', 'capture', 'image')]))
            if args.get('detail', 'location') != 'presence':
                steps += [self.envelope('ground_object', key + '_ground', {'target': name}, [
                    ('image', 'capture', 'image')]),
                    self.envelope('segment_object', key + '_segment', bindings=[
                        ('image', 'capture', 'image'), ('bbox', key + '_ground', 'pixel_box')]),
                    self.envelope('store_segmented_target', key + '_store', {
                        'name': name, 'role': role}, [
                            ('image', 'capture', 'image'), ('mask', key + '_segment', 'mask'),
                            ('observation_cloud', 'capture', 'cloud'),
                            ('camera_transform', 'capture', 'camera_transform')])]
        return steps

    async def submit(self, kind, args):
        if kind == 'plan_camera_view':
            validate_camera_view(args)
        fingerprint = json.dumps([kind, {k: v for k, v in args.items() if k != 'wait_seconds'}],
                                 sort_keys=True, allow_nan=False)
        task_id = 'task_' + hashlib.sha256(args['request_id'].encode()).hexdigest()[:32]
        prior = self.ledger.load('task:' + task_id)
        if prior:
            if prior['fingerprint'] != fingerprint:
                return {'success': False, 'state': 'BLOCKED', 'status': 'REQUEST_ID_REUSED',
                        'task_id': task_id, 'blocker': failure('REQUEST_ID_REUSED')}
            return await self.status(task_id, args.get('wait_seconds', 0))
        if self.ledger.db.execute(
                "SELECT COUNT(*) FROM execution WHERE id LIKE 'task:%'").fetchone()[0] >= 1000:
            return {'success': False, 'state': 'BLOCKED',
                    'status': 'TASK_HISTORY_CAPACITY_REQUIRES_LOCAL_ARCHIVE'}
        record = {'task_id': task_id, 'kind': kind, 'arguments': copy.deepcopy(args),
                  'fingerprint': fingerprint, 'created_at': time.time(),
                  'state': 'QUEUED', 'status': 'SUBMISSION_PENDING', **self.source()}
        self.save(record)  # Durable identity BEFORE any context/ROS submission.
        try:
            steps = self.route(kind, args)
            if any(s['function_name'] not in set(PREPARATION)
                   | {'get_observe_view'} for s in steps):
                raise ValueError('TASK_ROUTE_CONTAINS_UNSUPPORTED_FUNCTION')
            prefix = task_id + '_'
            for item in steps:
                item['step_id'] = prefix + item['step_id']
                for binding in item['bindings']:
                    binding['source_step_id'] = prefix + binding['source_step_id']
            record['context_id'] = self.gateway.own('', create=True)
            record['job_id'] = task_id + '_run'
            record['steps'] = steps
            self.save(record)
            admitted = await self.gateway.submit({
                'context_id': record['context_id'], 'request_id': task_id + '_submit',
                'steps': steps}, task_level=True)
            if not admitted['success']:
                record.update(state='BLOCKED', status=admitted['status'])
                self.save(record)
                return self.compact(record)
            record['revision'] = admitted['revision']
            self.save(record)
            started = await self.gateway.run({
                'operation': 'start', 'context_id': record['context_id'],
                'request_id': record['job_id'], 'expected_revision': admitted['revision']},
                task_level=True)
            if started.get('status') in ('STARTING', 'RUNNING') or 'UNKNOWN' in started['status']:
                self.ledger.save('current_task', {'task_id': task_id})
            record.update(state='RUNNING' if started.get('status') in ('RUNNING', 'STARTING')
                          else 'BLOCKED', status=started['status'])
            self.save(record)
            return await self.status(task_id, args.get('wait_seconds', 0))
        except BaseException as error:
            record.update(state='UNKNOWN' if isinstance(error, (TimeoutError,
                          asyncio.CancelledError)) else 'BLOCKED', status=str(error))
            self.save(record)
            if isinstance(error, asyncio.CancelledError):
                raise
            return self.compact(record)

    async def status(self, task_id, wait_seconds=0, diagnostics=False):
        record = self.load(task_id)
        if record['kind'] in ('move_camera', 'control_gripper', 'execute_manipulation'):
            result = self.compact(record)
            result['dispatched'] = record.get('dispatched', False)
            if diagnostics:
                result['diagnostics'] = {'steps': record.get('steps', [])}
            return result
        if record['state'] not in TERMINAL or (record['state'] == 'UNKNOWN' and
                                               record.get('job_id')):
            local = self.gateway.jobs.get(record.get('job_id'), {})
            future = local.get('task')
            if future and wait_seconds:
                try:
                    await asyncio.wait_for(asyncio.shield(future), timeout=wait_seconds)
                except asyncio.TimeoutError:
                    pass  # A waiting client timeout does not cancel or replay the ROS action.
            job = self.ledger.load('job:' + record.get('job_id', ''))
            if job:
                record['status'] = job['status']
                if job.get('completion') == 'FINAL_ACTION_RESULT':
                    record['state'] = ('CANCELLED' if job.get('ros_action_status') == 5 else
                                       'COMPLETED' if job.get('success') else 'FAILED')
                    rows = job.get('results', [])
                    record['rows'] = rows
                elif 'UNKNOWN' in job['status']:
                    record['state'] = 'UNKNOWN'
                elif job['status'] not in ('STARTING', 'RUNNING'):
                    record['state'] = 'BLOCKED'
            elif not future and record['status'] not in ('SUBMISSION_PENDING', 'QUEUED_ONLY'):
                if task_id not in self.direct_active:
                    record.update(state='UNKNOWN', status='SUBMISSION_OUTCOME_UNKNOWN')
            if (record['state'] in ('RUNNING', 'CANCEL_PENDING') and
                    record['kind'] != 'validate_plan'):
                record['current_step'] = record['current_step_id'] = None
                try:
                    snapshot = await self.gateway.inspect(record['context_id'], True)
                    state = json.loads(snapshot['state_json'])[0]
                    record['rows'] = snapshot['results']
                    active = state['pending'][0] if state['busy'] and state['pending'] else {}
                    record['current_step'] = active.get('function')
                    record['current_step_id'] = active.get('step_id')
                except Exception as error:
                    record['progress_unavailable'] = str(error)
                # An action can finish during the snapshot read. Terminal receipts win.
                latest = self.ledger.load('job:' + record.get('job_id', '')) or {}
                if latest.get('completion') == 'FINAL_ACTION_RESULT':
                    record['status'] = latest['status']
                    record['state'] = ('CANCELLED' if latest.get('ros_action_status') == 5 else
                                       'COMPLETED' if latest.get('success') else 'FAILED')
                    record['rows'] = latest.get('results', [])
            if record['kind'] != 'validate_plan':
                self.summarize(record)
            self.save(record)
        result = self.compact(record)
        if diagnostics:
            result['diagnostics'] = {'context_id': record.get('context_id'),
                                     'job_id': record.get('job_id'),
                                     'steps': record.get('steps'),
                                     'results': record.get('rows') or (self.ledger.load(
                                         'job:' + record.get('job_id', '')) or {}).get(
                                         'results', [])}
        return result

    def summarize(self, record):
        rows = record.get('rows', [])
        successful = [row for row in rows if row['success']]
        record['last_successful_step'] = successful[-1]['function_name'] if successful else None
        failed = next((row for row in rows if not row['success']), None)
        if failed:
            record['failed_step'] = failed['function_name']
            record['failed_step_id'] = failed['step_id']
            record['blocker'] = failure(failed['status'], failed['function_name'])
            if any(s in failed['status'] for s in ('STATE_UNKNOWN', 'OUTCOME_UNKNOWN', 'TIMEOUT')):
                record['state'] = 'UNKNOWN'
            elif any(s in failed['status'] for s in ('STALE', 'REQUIRED', 'NOT_IN_CACHE')):
                record['state'] = 'BLOCKED'
        if record['state'] in TERMINAL:
            record['current_step'] = record['current_step_id'] = None
        capture = next(
            (r for r in successful if r['function_name'] == 'capture_observation'), None)
        if capture:
            header = (capture.get('output') or {}).get('cloud', {}).get('header')
            if header:
                record['observation_time'] = header
        for row in successful:
            if row['function_name'] in ('prepare_grasp_lift', 'prepare_pick_place'):
                revision = row['output'].get('plan_revision', 0)
                if revision:
                    plan = {'plan_id': record['task_id'] + '_plan_' + str(revision),
                            'revision': revision, 'kind': row['function_name'],
                            'object_name': record['arguments']['object_name'],
                            'observation_header': record.get('observation_time'),
                            'ros_domain_id': record['input_provenance'].get('ros_domain_id'),
                            'execution_allowed': False}
                    record['plan'] = plan
                    self.ledger.save('plan:' + plan['plan_id'], dict(
                        plan, task_id=record['task_id'],
                                     source_mode=record['source_mode']))
        if record['kind'] == 'observe_scene':
            record['result'] = self.observed(record, rows)
        elif record['kind'] == 'prepare_manipulation':
            cache = next((r for r in successful if r['function_name'] == 'get_cached_grasp'), None)
            record['result'] = {'operation': record['arguments']['operation'],
                                'cache_confirmed': bool(cache),
                                'cache_created': any(r['function_name'] == 'cache_grasp_plan'
                                                     for r in successful),
                                'candidate_count': len(cache['output']['candidates'])
                                if cache and 'candidates' in (cache.get('output') or {}) else None,
                                'historical_result_not_current_readiness': True}
        elif record['kind'] == 'plan_camera_view':
            record['result'] = copy.deepcopy(successful[-1]['output']) if successful else {}
            if record['result']:
                record['result']['full_path_validated'] = (
                    record['arguments']['mode'] == 'target' and
                    record['result'].get('status') != 'CURRENT_VIEW_ALREADY_SUITABLE' and
                    len(record['result'].get('path_flat', [])) >= 12)
                record['result'].pop('path_flat', None)
                record['observation_time'] = record['result'].get('observation_header')
                if record['arguments']['mode'] == 'relative':
                    record['result']['planned_joint_change_rad'] = [
                        target - source for source, target in zip(
                            record['result'].get('source_joints', []),
                            record['result'].get('target_joints', []))]
                    record['result']['achieved_change_is_planned_not_executed'] = True

    def observed(self, record, rows):
        names = list(record['arguments']['targets'])
        if record['arguments'].get('destination'):
            names.append(record['arguments']['destination'])
        targets = {name: {'name': name, 'status': 'UNKNOWN'} for name in names}
        for row in rows:
            output = row.get('output') or {}
            if row['function_name'] == 'perceive_scene' and row['success']:
                for name, observed, position, status in zip(output['names'], output['observed'],
                                                            output['positions'],
                                                            output['target_statuses']):
                    targets[name].update(status='PRESENT' if observed else 'NOT_CONFIRMED',
                                         position_m=position, underlying_status=status,
                                         geometry_source='remembered' if 'REUSED' in status
                                         else 'current_observation')
            index = next((i for i, s in enumerate(record.get('steps', []))
                          if s['step_id'] == row['step_id']), None)
            if index is None:
                continue
            spec = record['steps'][index]
            name = spec['arguments'].get('target') or spec['arguments'].get('name')
            if row['function_name'] == 'segment_object':
                name = record['steps'][index - 1]['arguments']['target']
            if name not in targets:
                continue
            if row['function_name'] == 'verify_presence':
                targets[name].update(status='PRESENT' if row['success'] and output.get('present')
                                     else 'NOT_CONFIRMED' if row['status'] == 'NOT_PRESENT'
                                     else 'UNKNOWN', underlying_status=row['status'])
            if row['function_name'] == 'store_segmented_target' and row['success']:
                targets[name].update(position_m=output['position'],
                                     geometry_source_header=output['geometry_source_header'])
            if row['function_name'] == 'segment_object' and row['success']:
                # Source step carries its semantic name in the adjacent grounding request.
                name = record['steps'][index - 1]['arguments']['target']
                targets[name]['mask'] = output['mask']['result_ref']
        return {'targets': list(targets.values()), 'frame': 'base_link'}

    def compact(self, record):
        keys = ('task_id', 'state', 'status', 'source_mode', 'created_at', 'updated_at',
                'current_step', 'current_step_id', 'last_successful_step', 'failed_step',
                'failed_step_id', 'observation_time', 'result', 'plan',
                'blocker', 'cancellation')
        result = {key: record[key] for key in keys if key in record}
        if record['state'] not in ('RUNNING', 'CANCEL_PENDING'):
            result['current_step'] = result['current_step_id'] = None
        result['success'] = record['state'] == 'COMPLETED'
        if record['kind'] in ('prepare_manipulation', 'validate_plan'):
            rows = record.get('rows') or (self.ledger.load(
                'job:' + record.get('job_id', '')) or {}).get('results', [])
            result['validation'] = record.get('validation') or validation_summary(record, rows)
            result['execution_allowed'] = False
        if record['state'] in ('FAILED', 'BLOCKED', 'UNKNOWN') and 'blocker' not in result:
            result['blocker'] = failure(record['status'])
        return result

    async def cancel(self, task_id):
        record = self.load(task_id)
        if record['state'] in TERMINAL:
            return self.compact(record)
        if not record.get('job_id'):
            return {'success': False, 'state': 'BLOCKED', 'task_id': task_id,
                    'status': 'TASK_ADMISSION_NOT_YET_CANCELLABLE; inspect the same task'}
        result = await self.gateway.run({'operation': 'cancel', 'context_id': record['context_id'],
                                         'request_id': record['job_id']})
        record['cancellation'] = {'accepted': bool(result.get('success')),
                                  'status': result['status'],
                                  'inflight_service_interruptible': False}
        self.save(record)
        if result.get('success'):
            record.update(state='CANCEL_PENDING', status=result['status'])
            self.save(record)
        return await self.status(task_id)

    def describe(self, operation):
        if operation in OPERATIONS:
            kind, args = 'prepare_manipulation', {
                'operation': operation, 'object_name': '<exact object>',
                'destination': '<exact destination>'}
        elif operation == 'observe_scene':
            kind, args = operation, {'targets': ['<exact target>'], 'detail': 'location'}
        elif operation == 'plan_camera_view':
            kind, args = operation, {'mode': 'relative', 'translation_m': [0., 0., 0.],
                                     'rotation_rad': [0., 0., 0.]}
        else:
            raise ValueError('UNSUPPORTED_TASK_OPERATION')
        steps = self.route(kind, args)
        labels = route_labels(steps, args.get('destination'))
        return {'success': True, 'state': 'COMPLETED', 'status': 'WORKFLOW_DESCRIPTION',
                'result': {'operation': operation, 'prerequisites': [
                    'Verified source mode and physical gate disabled; existing route-specific '
                    'state checks (including empty hand for view planning) remain authoritative.',
                    'Server captures fresh data; failed/unknown producers stop dependents.'],
                    'route': [s['function_name'] for s in steps],
                    'step_count': len(steps), 'steps': labels,
                    'completion': 'QUEUED/RUNNING are acknowledgements, not completion.',
                    'outputs': 'task_id, source mode, observation time, partial result and '
                    'blocker; plan_id/revision only after actual motion preparation.',
                    'freshness': 'Capture requires new synchronized RGB-D/manual FK*X; active '
                    'object verification is 60s, grasp/prepared cache max 600s. Existing backend '
                    'feedback/pose checks enforce stage-specific limits; retention is not '
                    'proof the scene is unchanged. No one-second Qwen polling is used.',
                    'physical_execution': False}}

    async def inspect_robot(self):
        raw = await self.bridge.call('get_arm_state', {})
        world = await self.bridge.call('get_world_state', {})
        if raw.get('transport_outcome') != 'response' or not (raw.get('response') or {}).get(
                'success') or world.get('transport_outcome') != 'response':
            return {'success': False, 'state': 'UNKNOWN', 'status': 'ROBOT_FEEDBACK_UNAVAILABLE',
                    'blocker': failure(raw.get('detail', '') + ';' + world.get('detail', ''))}
        arm = raw['response']
        robot = json.loads(world['response']['state_json'])['robot']
        now = (self.bridge.node.get_clock().now().nanoseconds * 1.e-9 if hasattr(
            self.bridge, 'node') else time.time())
        workspace = None
        if hasattr(self.bridge, 'workspace'):
            read = await self.bridge.workspace()
            if read.get('transport_outcome') == 'response':
                workspace = json.loads(read['response']['state_json'])
                workspace['observation_header'] = read['response'].get('observation_header')
                workspace['camera_transform'] = read['response'].get('camera_transform')
                spatial = await self.bridge.call('get_spatial_context', {})
                mapped = spatial.get('response') or {}
                if spatial.get('transport_outcome') == 'response' and mapped.get('success'):
                    view = mapped['map']
                    workspace['camera'] = {
                        'observation_header': view['header'], 'pose': view['camera_pose'],
                        'forward': view['camera_forward'], 'up': view['camera_up'],
                        'direction': gaze_labels(view['camera_forward']),
                        'observed_cell_ids': view['observed_cell_ids'],
                        'cell_size_m': view['cell_size_m'],
                        'note': 'Independent spatial observation time; not a fresh VLM result.'}

            else:
                workspace = {'available': False, 'status': read.get('detail'),
                             'execution_allowed': False}
        stamps = {'joints': arm['joints']['header']['stamp'],
                  'tcp': arm['tcp_pose']['header']['stamp'],
                  'gripper': (robot.get('gripper') or {}).get('timestamp')}
        ages = {key: now - (stamp['sec'] + stamp['nanosec'] * 1.e-9) if stamp and
                (stamp['sec'] or stamp['nanosec']) else None for key, stamp in stamps.items()}
        return {'success': True, 'state': 'COMPLETED', 'status': arm['status'], **self.source(),
                'result': {'tcp': arm['tcp_pose'], 'joints_rad': arm['joints']['position'][:6],
                           'joint_names': arm['joints']['name'][:6],
                           'gripper_width_m': arm['gripper_width_m'], 'timestamps': stamps,
                           'feedback_age_seconds': ages,
                           'freshness': 'Read does NOT certify feedback freshness or safety.',
                           'robot': {k: robot.get(k) for k in ('state', 'holding',
                                     'pipeline_executing', 'physical_grasp_execution_enabled')},
                           'scene_memory_used': False, 'workspace': workspace}}

    async def validate(self, args):
        task_id = 'task_' + hashlib.sha256(args['request_id'].encode()).hexdigest()[:32]
        fingerprint = json.dumps(['validate_plan', args], sort_keys=True)
        prior = self.ledger.load('task:' + task_id)
        if prior:
            if prior['fingerprint'] != fingerprint:
                return {'success': False, 'state': 'BLOCKED', 'status': 'REQUEST_ID_REUSED',
                        'task_id': task_id}
            return await self.status(task_id)
        count = self.ledger.db.execute(
            "SELECT COUNT(*) FROM execution WHERE id LIKE 'task:%'").fetchone()[0]
        if count >= 1000:
            return {'success': False, 'state': 'BLOCKED',
                    'status': 'TASK_HISTORY_CAPACITY_REQUIRES_LOCAL_ARCHIVE'}
        record = {'task_id': task_id, 'kind': 'validate_plan', 'arguments': args,
                  'fingerprint': fingerprint, 'created_at': time.time(), 'state': 'RUNNING',
                  'status': 'VALIDATION_PENDING', **self.source()}
        self.save(record)
        self.direct_active.add(task_id)
        rows = []
        try:
            plan = self.ledger.load('plan:' + args['plan_id'])
            if not plan or plan['revision'] != args['revision']:
                raise ValueError(
                    'PLAN_ID_OR_REVISION_MISMATCH: use an actual prepared-plan result')
            if (plan['source_mode'] != record['source_mode'] or
                    plan.get('ros_domain_id') != record['input_provenance'].get('ros_domain_id')):
                raise ValueError('PLAN_SOURCE_MODE_MISMATCH')
            if self.ledger.lock.locked():
                raise ValueError('MCP_PIPELINE_OR_EXECUTION_BUSY')
            async with self.ledger.lock:
                parent = self.load(plan['task_id'])
                context = parent['context_id']
                get = ('get_prepared_pick_place' if plan['kind'] == 'prepare_pick_place'
                       else 'get_prepared_grasp_lift')
                validator = ('validate_prepared_pick_place' if plan['kind'] == 'prepare_pick_place'
                             else 'validate_prepared_grasp_lift')
                record['context_id'] = context
                record['current_step'] = 'get_execution_state'
                self.save(record)
                current = await self.bridge.call('get_execution_state', {
                    'object_name': plan['object_name']})
                state = current.get('response') or {}
                if (current.get('transport_outcome') != 'response' or not state.get('success') or
                        state.get('plan_revision') != args['revision'] or
                        state.get('observation_header') != plan.get('observation_header')):
                    raise ValueError('PREPARED_PLAN_SOURCE_OR_REVISION_CHANGED')
                for function in (get, validator):
                    record['current_step'] = function
                    self.save(record)
                    result = await self.gateway._call_locked(function, {
                        'context_id': context, 'step_id': task_id + '_' + function,
                        'request_id': task_id + '_' + function,
                        'arguments': {'object_name': plan['object_name']}, 'bindings': []})
                    rows.append(result)
                    if not result['success']:
                        record['failed_step'] = function
                        raise ValueError(result['status'])
                    record['last_successful_step'] = function
                    if (result.get('output') or {}).get('plan_revision') != args['revision']:
                        raise ValueError('PREPARED_PLAN_REVISION_CHANGED: prepare a new task')
                record.update(state='COMPLETED', status=result['status'], plan=plan,
                              result={'validated': True, 'execution_allowed': False,
                                      'historical_result_not_current_readiness': True,
                                      'scene_currentness': 'Immutable source checked, not a fresh '
                                      'visual confirmation. Refresh if the scene changed.'})
        except BaseException as error:
            status = str(error)
            record.update(state='UNKNOWN' if isinstance(error, (TimeoutError,
                          asyncio.CancelledError)) or 'UNKNOWN' in status else 'BLOCKED',
                          status=status, blocker=failure(status))
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            self.direct_active.discard(task_id)
            record['validation'] = validation_summary(record, rows)
            record['current_step'] = record['current_step_id'] = None
            self.save(record)
        return self.compact(record)

    async def call(self, name, args):
        if name == 'describe_workflow':
            return self.describe(args['operation'])
        if name == 'inspect_robot':
            return await self.inspect_robot()
        if name in ('get_task_status', 'cancel_task') and not args.get('task_id'):
            current = self.ledger.load('current_task')
            if not current:
                raise ValueError('NO_CURRENT_TASK: submit an operation first')
            args = dict(args, task_id=current['task_id'])
        if name == 'get_task_status':
            return await self.status(args['task_id'], args.get('wait_seconds', 0),
                                     args.get('include_diagnostics', False))
        if name == 'cancel_task':
            record = self.load(args['task_id'])
            if args['task_id'] in self.direct_active:
                return {'success': False, 'state': 'BLOCKED', 'status':
                        'INFLIGHT_SERVICE_CANNOT_BE_CANCELLED; inspect the same task',
                        'task_id': record['task_id']}
            return await self.cancel(args['task_id'])
        if name == 'validate_plan':
            return await self.validate(args)
        if name in ('observe_scene', 'prepare_manipulation', 'plan_camera_view'):
            return await self.submit(name, args)
        raise ValueError('UNSUPPORTED_TASK_TOOL')
