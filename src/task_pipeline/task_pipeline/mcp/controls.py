"""Small control facades over the existing master and durable execution admission."""

import asyncio
import copy
import hashlib
import json
import math
import time

import jsonschema

from mcp import types

from .execution import reference
from .preparation import IDENTIFIER
from .tasks import definitions as task_definitions, validate_camera_view

NAMES = ('move_camera', 'control_gripper', 'execute_manipulation')
OUTPUT = {'type': 'object', 'required': ['success', 'state', 'status', 'dispatched'],
          'properties': {'success': {'type': 'boolean'}, 'state': {'enum': [
              'RUNNING', 'COMPLETED', 'BLOCKED', 'UNKNOWN']}, 'status': {'type': 'string'},
              'dispatched': {'type': 'boolean'}, 'task_id': {'type': 'string'},
              'execution_allowed': {'type': 'boolean'}}, 'additionalProperties': True}


def definitions(enabled=False):
    """Flat model-compatible schemas; annotations never replace dispatch guards."""
    camera = copy.deepcopy(next(t.inputSchema for t in task_definitions()
                                if t.name == 'plan_camera_view'))
    camera['properties'].pop('wait_seconds')
    plan = {'plan_id': {'type': 'string', 'minLength': 1, 'maxLength': 192,
                        'description': 'Exact plan_id from the reviewed preparation result.'},
            'revision': {'type': 'integer', 'minimum': 1,
                         'description': 'Exact revision paired with that reviewed plan_id.'}}
    gripper = {'type': 'object', 'additionalProperties': False,
               'required': ['request_id', 'action', 'mode'], 'properties': {
                   'request_id': IDENTIFIER, 'action': {'enum': ['open', 'close', 'release']},
                   'mode': {'enum': ['prepared', 'empty_hand'], 'description':
                            'prepared requires plan_id/revision; frozen widths/forces only. '
                            'empty_hand supports open only, requiring width_m and force_n.'},
                   **plan, 'width_m': {'type': 'number', 'exclusiveMinimum': .001,
                                       'maximum': .10,
                                       'description': 'Empty-hand opening only, metres.'},
                   'force_n': {'type': 'number', 'minimum': .5, 'maximum': 3.,
                               'description': 'Empty-hand opening only, newtons.'}}}
    execute = {'type': 'object', 'additionalProperties': False,
               'required': ['request_id', 'plan_id', 'revision'], 'properties': {
                   'request_id': IDENTIFIER, **plan, 'stage': {
                       'enum': ['all', 'next'], 'default': 'all', 'description':
                       'all executes the remaining supported prepared stages in order. '
                       'next executes one stage, including gripper/slot bookkeeping boundaries.'}}}
    purposes = {
        'move_camera': 'Move an empty-hand camera using relative optical metres/radians or '
        'one target selector, with the same mode rules as plan_camera_view. Existing planner '
        'and master/VAMP full-path checks run before movement. NO_REACHABLE_CHANGE is a '
        'calculation outcome with no movement. Never use while holding.',
        'control_gripper': 'Open/close/release through master safeguards. Prepared operations '
        'require the exact reviewed plan reference and next valid stage. Close confirms '
        'fresh contact; release requires validated placement. Empty-hand open requires '
        'fresh feedback and cannot narrow the jaws. No free closing or held release. '
        'Opening feedback proves open detection, not calibrated target aperture.',
        'execute_manipulation': 'Execute the exact reviewed grasp_lift or pick_place plan. '
        'No latest-plan guessing. Server follows master waiting stages, verifies contact '
        'before held motion and commits slots after confirmed release. grasp_lift stops '
        'holding the object and keeps ownership. No physical retries or alternatives.'}
    availability = ('Guarded control enabled; master physical permission still required.'
                    if enabled else
                    'DISABLED: preparation/read-only mode cannot dispatch this tool.')
    for name, schema in zip(NAMES, (camera, gripper, execute)):
        yield types.Tool(name=name, inputSchema=schema, outputSchema=copy.deepcopy(OUTPUT),
                         description=purposes[name] + ' Durable request_id: identical retry '
                         'returns the receipt, never reexecutes. Final completion or explicit '
                         'UNKNOWN; inspect get_task_status after timeout/disconnection. Local '
                         'stop/watchdog remains independent. ' + availability,
                         annotations=types.ToolAnnotations(
                             readOnlyHint=False, destructiveHint=True, idempotentHint=False,
                             openWorldHint=False),
                         _meta={'availability': 'guarded_execution' if enabled else 'disabled'})


class ControlInterface:
    """No second planner/queue: one ledger lock and existing execution stage guards."""

    def __init__(self, execution, tasks, bridge, enabled=False):
        """Bind the existing task/ledger owners and independent dispatch mode."""
        self.execution, self.tasks, self.bridge, self.enabled = execution, tasks, bridge, enabled

    def validate(self, name, args):
        """Validate flat mode-specific inputs before any durable task creation."""
        schema = next(t.inputSchema for t in definitions() if t.name == name)
        jsonschema.validate(args, schema)
        if name == 'move_camera':
            validate_camera_view(args)
        if name == 'control_gripper':
            if args['mode'] == 'prepared':
                if not {'plan_id', 'revision'} <= args.keys():
                    raise ValueError('REVIEWED_PLAN_REFERENCE_REQUIRED')
                if {'width_m', 'force_n'} & args.keys():
                    raise ValueError('PREPARED_GRIPPER_USES_FROZEN_WIDTH_AND_FORCE')
            elif (args['action'] != 'open' or not {'width_m', 'force_n'} <= args.keys() or
                  {'plan_id', 'revision'} & args.keys()):
                raise ValueError('EMPTY_HAND_MODE_SUPPORTS_EXPLICIT_OPEN_ONLY')
            if any(not math.isfinite(args[k]) for k in ('width_m', 'force_n') if k in args):
                raise ValueError('FINITE_GRIPPER_COMMAND_REQUIRED')

    async def call(self, name, args):
        """Deduplicate and serialize one fixed control operation through existing guards."""
        self.validate(name, args)  # Reject malformed inputs before task admission.
        if not self.enabled or self.execution.transport is None:
            return {'success': False, 'state': 'BLOCKED', 'status': 'PHYSICAL_CONTROLS_DISABLED',
                    'dispatched': False, 'execution_allowed': False}
        key = 'request:' + args['request_id']
        fingerprint = json.dumps([name, args], sort_keys=True, allow_nan=False)
        old = self.execution.load(key)
        if old:
            if old['fingerprint'] != fingerprint:
                raise ValueError('REQUEST_ID_REUSED_WITH_DIFFERENT_ARGUMENTS')
            result = old.get('result')
            if result:
                return dict(result, dispatched=False, receipt_replayed=True)
            return {'success': False, 'state': 'UNKNOWN', 'status':
                    'CONTROL_OUTCOME_UNKNOWN_INSPECT_SAME_TASK', 'dispatched': False,
                    'task_id': old['task_id']}
        if self.execution.lock.locked():
            return {'success': False, 'state': 'BLOCKED', 'status': 'MCP_EXECUTION_BUSY',
                    'dispatched': False}
        async with self.execution.lock:
            if self.execution.load('preparation_barrier'):
                raise ValueError('PREPARATION_OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION')
            current = self.execution.load('current')
            if current and current.get('unknown'):
                raise ValueError('MCP_STATE_UNKNOWN_REQUIRES_LOCAL_RECONCILIATION')
            source = self.tasks.source()
            if source['source_mode'] != 'live':
                raise ValueError('PHYSICAL_CONTROLS_REQUIRE_LIVE_INPUT_PROVENANCE')
            if self.execution.db.execute('SELECT COUNT(*) FROM execution').fetchone()[0] >= 9900:
                raise ValueError('EXECUTION_HISTORY_CAPACITY_REQUIRES_LOCAL_ARCHIVE')
            task_id = 'control_' + hashlib.sha256(args['request_id'].encode()).hexdigest()[:32]
            record = {'task_id': task_id, 'kind': name, 'state': 'RUNNING',
                      'status': 'CONTROL_ADMITTED', 'arguments': args,
                      'created_at': time.time(), 'steps': [], 'dispatched': False,
                      **source}
            self.tasks.save(record)
            self.tasks.direct_active.add(task_id)
            self.execution.save(key, {'fingerprint': fingerprint, 'task_id': task_id})
            self.execution.save('current_task', {'task_id': task_id})
            try:
                if name == 'move_camera':
                    result = await self.move_camera(record)
                elif name == 'control_gripper' and args['mode'] == 'empty_hand':
                    result = await self.open_empty_hand(record)
                else:
                    result = await self.manipulation(record)
                record.update(state='COMPLETED', status=result.pop('status'), result=result)
            except BaseException as error:
                uncertain = isinstance(error, (TimeoutError, asyncio.CancelledError)) or record.get(
                    'uncertain', False)
                record.update(state='UNKNOWN' if uncertain else 'BLOCKED',
                              status=('CONTROL_OUTCOME_UNKNOWN: ' if uncertain else '') +
                              (str(error) or type(error).__name__),
                              failed_step=record.get('current_step'))
                if uncertain:
                    self.execution.save('preparation_barrier', {'control_task': task_id})
                if isinstance(error, asyncio.CancelledError):
                    raise
            finally:
                record['current_step'] = record['current_step_id'] = None
                self.tasks.direct_active.discard(task_id)
                self.tasks.save(record)
                keys = ('task_id', 'state', 'status', 'source_mode', 'dispatched',
                        'current_step', 'last_successful_step', 'failed_step', 'blocker', 'result')
                result = {k: record[k] for k in keys if k in record}
                result['success'] = record['state'] == 'COMPLETED'
                self.execution.save(key, {'fingerprint': fingerprint, 'task_id': task_id,
                                          'result': result})
            return result

    def progress(self, record, stage):
        """Expose the actual active public stage while waiting for final ROS completion."""
        record['current_step'] = stage
        self.tasks.save(record)

    def completed(self, record, stage):
        """Retain the last master-confirmed successful stage."""
        record['last_successful_step'] = stage
        self.tasks.save(record)

    async def child(self, record, name, arguments, kind='prepare_pick_place'):
        """Run a fixed guarded stage while retaining shared ledger admission."""
        self.progress(record, name)
        arguments = dict(arguments, request_id=record['task_id'] + '_' +
                         str(len(record['steps'])))
        record['steps'].append(name)
        response = (await self.execution._move(arguments) if name == 'move_to' else
                    await self.execution._open(arguments) if name == 'open_empty_hand' else
                    await self.execution._call(name, arguments, prepared_kind=kind))
        record['dispatched'] |= response['dispatched']
        if response['transport_outcome'] != 'response':
            record['uncertain'] = response['transport_outcome'] == 'unknown'
            record['blocker'] = {'step': name, 'status': (response.get('response') or {}).get(
                'status', response['detail']), 'transport_outcome': response['transport_outcome'],
                'recovery': 'Inspect this same task. Unknown actuator outcomes require local '
                            'reconciliation; never replay physical work.'}
            raise RuntimeError(response['detail'])
        self.completed(record, name)
        return response['response']

    async def move_camera(self, record):
        """Plan a view and let the master validate/execute the complete empty-hand path."""
        current = self.execution.load('current')
        if current and (current.get('unknown') or not current.get('complete')):
            raise ValueError('OTHER_OR_UNKNOWN_EXECUTION_SESSION_OWNS_ARM')
        await self.require_empty_hand()
        step = self.tasks.route('plan_camera_view', record['arguments'])[0]
        self.progress(record, step['function_name'])
        plan = await self.execution.transport.request(step['function_name'], step['arguments'], 60.)
        if not plan.get('success'):
            raise ValueError(plan.get('status', 'CAMERA_PLANNING_FAILED'))
        self.completed(record, step['function_name'])
        if record['arguments']['mode'] == 'relative':
            changed = plan.get('moved_target', False)
            if changed and not plan.get('endpoint_safe'):
                raise ValueError('VIEW_ENDPOINT_NOT_CHECKED')
            start, target = plan['source_joints'], plan['target_joints']
        else:
            changed = plan['status'] != 'CURRENT_VIEW_ALREADY_SUITABLE'
            start = plan['expected_start_joints']
            target = plan['path_flat'][-6:]
        summary = {k: plan[k] for k in ('status', 'achieved_fraction', 'endpoint_safe',
                   'translation_error_m', 'orientation_error_rad') if k in plan}
        if not changed:
            return {'status': plan['status'], 'movement_completed': False,
                    'planner_result': summary, 'full_path_executed': False}
        state = await self.execution.transport.request('get_arm_state', {}, 5.)
        measured = state.get('joints', {}).get('position', [])[:6]
        if (not state.get('success') or len(measured) != 6 or len(start) != 6 or
                not all(math.isfinite(v) for v in measured+list(start)) or
                max(abs(a-b) for a, b in zip(measured, start)) > .02):
            raise ValueError('VIEW_START_CHANGED_REPLAN_REQUIRED')
        await self.child(record, 'move_to', {'target_joints': target})
        # The existing ROS idle scanner owns capture; do not create a VLM polling loop.
        try:
            observation = await self.bridge.workspace(include_image=True)
            metadata = observation.get('response') or {}
        except Exception:
            metadata = {}  # Movement is confirmed; sensing remains explicitly unconfirmed.
        return {'status': 'CAMERA_MOVE_COMPLETED', 'movement_completed': True,
                'planner_result': summary, 'full_path_executed': True,
                'observation_status': metadata.get('status', 'REFRESH_REQUIRED'),
                'observation_refresh_required':
                    metadata.get('status') != 'CURRENT_OBSERVATION_READY'}

    async def require_empty_hand(self):
        """Require master permission and a known idle empty-hand state."""
        world = await self.execution.transport.request('get_world_state', {}, 5.)
        robot = json.loads(world['state_json'])['robot']
        if not robot.get('physical_grasp_execution_enabled'):
            raise ValueError('PHYSICAL_EXECUTION_DISABLED_BY_MASTER')
        if (robot.get('state') != 'IDLE' or robot.get('holding') is not None or
                robot.get('pipeline_executing')):
            raise ValueError('CONTROL_REQUIRES_IDLE_EMPTY_HAND')

    async def open_empty_hand(self, record):
        """Use the fixed opening-only master callback and fresh feedback result."""
        await self.require_empty_hand()
        args = record['arguments']
        result = await self.child(record, 'open_empty_hand', {
            'width_m': args['width_m'], 'force_n': args['force_n'], 'wait_for_opening': True})
        return {'status': result['status'], 'opening_feedback_confirmed': True,
                'target_aperture_verified': False}

    async def manipulation(self, record):
        """Execute only stages from the exact reviewed prepared plan."""
        args = record['arguments']
        plan = self.execution.load('plan:' + args['plan_id'])
        if (not plan or plan['revision'] != args['revision'] or
                plan['kind'] not in ('prepare_grasp_lift', 'prepare_pick_place')):
            raise ValueError('REVIEWED_PLAN_ID_OR_REVISION_MISMATCH')
        if (plan['source_mode'] != record['source_mode'] or
                plan['ros_domain_id'] != record['input_provenance']['ros_domain_id']):
            raise ValueError('PLAN_SOURCE_MODE_MISMATCH')
        parent = self.tasks.load(plan['task_id'])
        if parent['state'] != 'COMPLETED':
            raise ValueError('PREPARATION_DID_NOT_COMPLETE')
        state = await self.execution.transport.request('get_execution_state', {
            'object_name': plan['object_name']}, 5.)
        if (not state.get('success') or state['plan_revision'] != plan['revision'] or
                state['observation_header'] != plan['observation_header']):
            raise ValueError('PREPARED_PLAN_SOURCE_OR_REVISION_CHANGED')
        ref = {'object_name': plan['object_name'], 'plan_revision': plan['revision'],
               'attempt': 1, 'task_id': reference(plan['object_name'], state)}
        gripper = record['kind'] == 'control_gripper'
        wanted = {'open': 'gripper_open', 'close': 'gripper_close', 'release': 'release'}.get(
            args.get('action'))
        for _ in range(16):  # Bounded public stages, never micro-waypoints or retries.
            state = await self.execution.transport.request('get_execution_state', {
                'object_name': plan['object_name']}, 5.)
            current = self.execution.load('current') or {}
            if (not state.get('success') or state.get('plan_revision') != plan['revision'] or
                    state.get('observation_header') != plan['observation_header']):
                raise ValueError('PREPARED_PLAN_SOURCE_OR_REVISION_CHANGED')
            if 'UNKNOWN' in state.get('status', '') or current.get('unknown'):
                record['uncertain'] = True
                raise RuntimeError('STATE_UNKNOWN_REQUIRES_LOCAL_RECONCILIATION')
            waiting = state.get('waiting_stages', [])
            if (state.get('phase') == 'RELEASED' and
                    'commit_slots' not in current.get('operations', [])):
                stage = 'commit_slots'
            elif waiting:
                stage = waiting[0]
            else:
                if gripper:
                    raise ValueError('WRONG_GRIPPER_STAGE: no pending prepared gripper action')
                if (state.get('phase') != 'COMPLETE' and not (
                        plan['kind'] == 'prepare_grasp_lift' and state.get('phase') == 'LIFTED')):
                    raise ValueError('NO_VALID_TERMINAL_EXECUTION_STATE')
                return {'status': 'MANIPULATION_COMPLETED', 'phase': state.get('phase'),
                        'holding': state.get('phase') == 'LIFTED',
                        'held_session_remains_owned': state.get('phase') == 'LIFTED'}
            if gripper and stage != wanted:
                raise ValueError('WRONG_GRIPPER_STAGE: next=' + stage)
            function = {'transfer': 'execute_placement_transfer',
                        'descent': 'execute_placement_descent', 'release': 'release_placed_object',
                        'withdrawal': 'execute_withdrawal', 'retreat': 'execute_placement_retreat',
                        'gripper_open': 'gripper_open', 'gripper_close': 'gripper_close',
                        'commit_slots': 'commit_floor_slots'}.get(stage, 'execute_plan')
            if stage == 'commit_slots':
                query = {'selected_pose': current.get('selected_pose'),
                         'reset': False, 'validate_only': False}
            else:
                query = {'object_name': parent['arguments']['destination'] if stage in (
                         'withdrawal', 'retreat') else plan['object_name'],
                         'plan_revision': plan['revision']}
                if function == 'execute_plan':
                    query['segment'] = stage
                if stage in ('gripper_open', 'gripper_close'):
                    query.update(width_m=0., force_n=0.)
            response = await self.child(
                record, function, dict(query, execution_ref=ref), plan['kind'])
            if gripper or args.get('stage') == 'next':
                return {'status': response['status'], 'completed_stage': stage,
                        'phase': response.get('phase', state['phase'])}
        raise RuntimeError('UNEXPECTED_PREPARED_STAGE_SEQUENCE_REQUIRES_INSPECTION')
