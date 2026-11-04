"""Agent-composed calculation steps using the master's typed StepStore.

Images, masks, clouds and CDR stay in ROS-side results. This adapter only builds
requests and bindings; it does not implement another planner or robot queue.
Queued physical calls are refused: the agent invokes the guarded execution tools
individually after preparation, preserving their durable admission/contact checks.
"""

import asyncio
import json
import re
import uuid

from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.set_message import set_message_fields

from task_pipeline.context import fields
from task_pipeline.registry import FUNCTIONS

from .schema import message_schema


PREPARATION = {
    'capture_observation': ('Capture synchronized RGB-D and manual FK*X; returns '
                            'Ubuntu-side handles.'),
    'verify_presence': ('Check target on a same-context captured image. Only success AND '
                        'present establish PRESENT. Missing memory, inference failure, '
                        'timeout, empty/uncertain output mean UNKNOWN, not absent. '
                        'present=false alone is not proof of visual absence.'),
    'ground_object': 'Ground a target bounding box on the bound image.',
    'segment_object': 'Generate an EdgeSAM pixel mask using the bound image and bbox.',
    'store_segmented_target': ('Extract mask-selected XYZ and store named '
                               'object/destination memory.'),
    'resolve_target': 'Reuse target memory or capture/ground/segment/extract/store it.',
    'refresh_active_target': 'Refresh selected target geometry using a synchronized observation.',
    'get_active_target': 'Read stored target image/mask/cloud with source association.',
    'perceive_scene': 'Batch-perceive configured objects and destination on one observation.',
    'ensure_scene_memory': 'Reuse known target locations or populate scene memory if missing.',
    'check_target_view': 'Read geometric visibility using memory and current camera geometry.',
    'require_target_view': 'Require visibility without moving the camera; reject LOOK_REQUIRED.',
    'pin_grasp_observation': 'Pin the grasp observation by its exact source header.',
    'pin_placement_observation': 'Pin the placement observation by its exact source header.',
    'prepare_grasp_input': 'Associate image/mask/box and derive HGGD inputs and affordance.',
    'calculate_grasps': 'Generate and validate HGGD grasp candidates on the bound observation.',
    'calculate_active_grasps': 'Calculate grasps from the active object memory and source image.',
    'update_object': 'Register semantic object identity and measured base-link position.',
    'select_grasp': 'Provisionally rank validated candidates; not whole-task readiness.',
    'cache_grasp_plan': 'Cache the source-associated grasp pool for complete-plan preparation.',
    'get_cached_grasp': 'Read the source-associated cached candidate pool.',
    'find_placements': 'Calculate destination footprint/slots and object placement candidates.',
    'find_active_placements': ('Search slots using active object/destination geometry and '
                               'occupancy.'),
    'search_placements': 'Search placements on bound immutable prepared geometry.',
    'prepare_held_scene': ('Freeze measured source, support and environment geometry '
                           'without motion.'),
    'refine_grasp': ('Refine registered contact geometry without motion; provides closing axis '
                     'and pose for plan_final_approach; may need a closer observation.'),
    'plan_final_approach': 'Calculate and validate the complete pregrasp-to-grasp transition.',
    'plan_lift': 'Calculate held lift or empty-hand withdrawal; no physical execution.',
    'plan_placement': 'Calculate full lift/transfer/descent validation on bound candidates.',
    'validate_placements': 'Validate placement candidates and motion without executing.',
    'plan_motion': 'Calculate a VAMP joint-space path; does not execute it.',
    'solve_pose': 'Solve a target pose; ignore_collision=true is forbidden remotely.',
    'plan_camera_retreat': 'Calculate frozen-scene retreat without actuation.',
    'plan_next_best_view': ('Calculate object/point or target_cell_id viewpoint using the '
                            'shared map; targets are exclusive. No actuation.'),
    'prepare_pick_place': ('Choose a candidate with complete '
                           'approach/lift/place/withdrawal/retreat.'),
    'get_prepared_pick_place': ('Read the frozen complete plan revision and selected '
                                'placement pose.'),
    'validate_prepared_pick_place': ('Check complete cached handoff, no actuation or cache '
                                     'consumption.'),
    'prepare_grasp_lift': ('Calculate grasp/lift only; current remote payload execution '
                           'uses pick/place.'),
    'get_prepared_grasp_lift': 'Read the frozen grasp/lift readiness record.',
    'validate_prepared_grasp_lift': 'Validate cached grasp/lift without actuation.',
}

IDENTIFIER = {'type': 'string', 'minLength': 1, 'maxLength': 128,
              'pattern': '^[A-Za-z0-9_-]+$'}
BINDING_SCHEMA = {'type': 'object', 'additionalProperties': False,
                  'required': ['target_field', 'source_step_id', 'source_field'],
                  'properties': {key: {'type': 'string', 'minLength': 1, 'maxLength': 256}
                                 for key in ('target_field', 'source_step_id', 'source_field')}}
OUTPUT_SCHEMA = {'type': 'object', 'required': ['success', 'status'],
                 'properties': {'success': {'type': 'boolean'}, 'status': {'type': 'string'}},
                 'additionalProperties': True}


def argument_schema(name):
    """Literal arguments are small scalar fields; message geometry uses bindings."""
    request = FUNCTIONS[name].service_type.Request
    schema = message_schema(request)
    properties = {key: schema['properties'][key] for key, kind in
                  request.get_fields_and_field_types().items() if '/' not in kind}
    for value in properties.values():
        if value.get('type') == 'array':
            value['maxItems'] = min(value.get('maxItems', 1024), 1024)
        if value.get('type') == 'string':
            value['maxLength'] = min(value.get('maxLength', 8192), 8192)
    return {'type': 'object', 'properties': properties, 'additionalProperties': False}


def step_schema(name=None):
    """Use actual catalog names and typed binding references, never caller CDR."""
    properties = {'step_id': IDENTIFIER,
                  'arguments': argument_schema(name) if name else {'type': 'object'},
                  'bindings': {'type': 'array', 'items': BINDING_SCHEMA, 'maxItems': 64}}
    required = ['step_id', 'arguments', 'bindings']
    if name is None:
        properties['function_name'] = {'enum': sorted(PREPARATION)}
        required.append('function_name')
    return {'type': 'object', 'required': required, 'properties': properties,
            'additionalProperties': False}


def call_schema(name=None):
    """Add context/revision/idempotency to each individually exposed function."""
    schema = step_schema(name)
    schema['properties'].update(
        context_id={'type': 'string', 'maxLength': 128}, request_id=IDENTIFIER,
        expected_revision={'type': 'integer', 'minimum': 0, 'maximum': 2**64-1})
    schema['required'] += ['context_id', 'request_id']
    return schema


def result_summary(message, context_id, step_id, path=''):
    """Publish metadata/handles, never image, mask, cloud or serialized buffers."""
    ref = {'context_id': context_id, 'step_id': step_id,
           'source_field': path or '@response_cdr'}
    if hasattr(message, 'get_fields_and_field_types'):
        names = message.get_fields_and_field_types()
        if 'data' in names and ('encoding' in names or 'point_step' in names):
            return {'result_ref': ref, 'type': type(message).__name__,
                    'width': message.width, 'height': message.height,
                    'header': result_summary(message.header, context_id, step_id,
                                             path+'.header' if path else 'header')}
        return {key: result_summary(getattr(message, key), context_id, step_id,
                                    path+'.'+key if path else key) for key in names}
    if hasattr(message, 'item') and not hasattr(message, '__len__'):
        message = message.item()
    if isinstance(message, (str, bool, int, float)):
        if isinstance(message, str) and len(message) > 8192:
            return {'result_ref': ref, 'characters': len(message), 'omitted': True}
        return message
    if hasattr(message, '__len__'):
        if len(message) > 32 or isinstance(message, (bytes, bytearray)):
            return {'result_ref': ref, 'count': len(message), 'omitted': True}
        return [result_summary(value, context_id, step_id, path+f'[{index}]')
                for index, value in enumerate(message)]
    raise TypeError('UNSUPPORTED_RESULT_TYPE')


def binding_type(message_class, path):
    """Resolve ROS field types without constructing caller-supplied messages."""
    from rosidl_runtime_py.utilities import get_message
    current = message_class
    parts = fields(path)
    for offset, (name, index) in enumerate(parts):
        kind = current.get_fields_and_field_types().get(name)
        if kind is None:
            raise ValueError('UNKNOWN_BINDING_FIELD: '+name)
        element = re.fullmatch(r'(?:bounded_)?sequence<([^,>]+)(?:,.*)?>', kind)
        fixed = re.fullmatch(r'(.+)\[([0-9]*)\]', kind)
        if index is not None:
            if element:
                kind = element[1]
            elif fixed:
                if fixed[2] and index >= int(fixed[2]):
                    raise ValueError('BINDING_INDEX_OUT_OF_RANGE')
                kind = fixed[1]
            else:
                raise ValueError('BINDING_INDEX_REQUIRES_ARRAY')
        if offset == len(parts)-1:
            return kind
        if '/' not in kind:
            raise ValueError('BINDING_SUBFIELD_REQUIRES_MESSAGE')
        names = kind.split('/')
        current = get_message('/'.join(names if len(names) == 3 else
                                       (names[0], 'msg', names[1])))


def compatible_binding(source, target):
    """Allow existing numeric bindings, require exact ROS message identities."""
    def shape(kind):
        match = re.fullmatch(r'(?:bounded_)?sequence<([^,>]+)(?:,.*)?>', kind)
        fixed = re.fullmatch(r'(.+)\[([0-9]*)\]', kind)
        return (True, match[1]) if match else (True, fixed[1]) if fixed else (False, kind)
    sa, source = shape(source)
    ta, target = shape(target)
    if sa != ta:
        return False
    integers = {'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'byte'}
    floats = {'float', 'double', 'float32', 'float64'}
    return (source == target or source in integers and target in integers or
            source in floats and target in floats)


class PreparationGateway:
    """Resolve all substantive data inside the existing master context."""

    def __init__(self, transport, execution, require_physical_disabled=False):
        """Share admission serialization/persistence; do not create another queue."""
        self.transport, self.execution = transport, execution
        self.require_physical_disabled = require_physical_disabled
        self.jobs = {}
        for key, data in execution.db.execute(
                "SELECT id,data FROM execution WHERE id LIKE 'request:%'").fetchall():
            value = json.loads(data)
            if 'result' not in value and 'context_id' in value:
                execution.save('preparation_barrier', {'request': key})
        for key, data in execution.db.execute(
                "SELECT id,data FROM execution WHERE id LIKE 'job:%'").fetchall():
            value = json.loads(data)
            if value.get('status') in ('STARTING', 'RUNNING'):
                value.update(success=False, status='ACTION_OUTCOME_UNKNOWN_AFTER_RESTART')
                execution.save(key, value)
                execution.save('preparation_barrier', {'job': key, 'status': value['status']})

    def make_step(self, name, values):
        """Construct one typed request from bounded primitives and source handles."""
        import jsonschema
        from my_actions.msg import PipelineStep, StepBinding
        if name not in PREPARATION:
            raise ValueError('FUNCTION_NOT_ALLOWED_IN_CALCULATION_PIPELINE')
        arguments = values['arguments']
        jsonschema.validate(arguments, argument_schema(name))
        if name == 'solve_pose' and arguments.get('ignore_collision'):
            raise ValueError('COLLISION_BYPASS_FORBIDDEN')
        request = FUNCTIONS[name].service_type.Request()
        set_message_fields(request, arguments)
        bindings = []
        for item in values['bindings']:
            if name == 'solve_pose' and item['target_field'] == 'ignore_collision':
                raise ValueError('COLLISION_BYPASS_FORBIDDEN')
            fields(item['target_field'])
            if item['source_field'] != '@response_cdr':
                fields(item['source_field'])
            bindings.append(StepBinding(**item))
        return PipelineStep(step_id=values['step_id'], function_name=name,
                            request_cdr=serialize_message(request), bindings=bindings,
                            failure_policy='stop')

    def guard(self):
        """Prevent planning from replacing an executing/uncertain task's geometry."""
        if self.execution.db.execute('SELECT COUNT(*) FROM execution').fetchone()[0] >= 10000:
            raise ValueError('EXECUTION_HISTORY_CAPACITY_REQUIRES_LOCAL_ARCHIVE')
        if self.execution.load('preparation_barrier'):
            raise ValueError('PREPARATION_OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION')
        current = self.execution.load('current')
        if current and (current.get('unknown') or not current.get('complete')):
            raise ValueError('ACTIVE_OR_UNKNOWN_EXECUTION_REQUIRES_COMPLETION')

    async def current_state(self):
        """Do not let calculations replace geometry during held/unknown execution."""
        from my_actions.srv import GetWorldState
        response = await self.transport.typed('get_world_state', GetWorldState.Request(), 5.)
        robot = json.loads(response.state_json)['robot']
        if (self.require_physical_disabled and
                robot.get('physical_grasp_execution_enabled') is not False):
            raise ValueError('PREPARATION_MODE_REQUIRES_PHYSICAL_GATE_DISABLED')
        if (robot.get('state') != 'IDLE' or robot.get('holding') is not None or
                robot.get('pipeline_executing')):
            raise ValueError('PREPARATION_REQUIRES_IDLE_KNOWN_EMPTY_HAND')

    def own(self, context_id, create=False):
        """Only access contexts created by this authenticated gateway."""
        if not context_id and create:
            context_id = 'mcp_'+uuid.uuid4().hex
            self.execution.save('context:'+context_id, {'context_id': context_id})
        if not self.execution.load('context:'+context_id):
            raise ValueError('MCP_CONTEXT_OWNER_REQUIRED')
        return context_id

    async def validate_bindings(self, context_id, steps, queued=False):
        """Reject missing/wrong-typed predecessors before master backend dispatch."""
        from my_actions.srv import GetStepContext
        local, external = set(), set()
        for step in steps:
            external.update(
                b.source_step_id for b in step.bindings if b.source_step_id not in local)
            local.add(step.step_id)
        sources = {}
        if external:
            response = await self.transport.typed('get_step_context', GetStepContext.Request(
                context_id=context_id, include_results=False), 30.)
            if not response.success:
                raise ValueError('DEPENDENCY_CONTEXT_UNAVAILABLE: '+response.status)
            state = json.loads(response.state_json)[0]
            sources = {item['step_id']: item['function'] for item in state['results']
                       if item['success']}
            if queued:
                sources.update({item['step_id']: item['function'] for item in state['pending']})
        for step in steps:
            for binding in step.bindings:
                source_name = sources.get(binding.source_step_id)
                if source_name not in FUNCTIONS:
                    raise ValueError('DEPENDENCY_FAILED: '+binding.source_step_id)
                source = ('sequence<uint8>' if binding.source_field == '@response_cdr' else
                          binding_type(FUNCTIONS[source_name].service_type.Response,
                                       binding.source_field))
                target = binding_type(FUNCTIONS[step.function_name].service_type.Request,
                                      binding.target_field)
                if not compatible_binding(source, target):
                    raise ValueError('BINDING_TYPE_MISMATCH: '+binding.target_field)
            sources[step.step_id] = step.function_name

    async def call(self, name, values):
        """Retain opaque artifacts in StepStore and return only a bounded projection."""
        if self.execution.lock.locked():
            return {'success': False, 'status': 'MCP_PIPELINE_OR_EXECUTION_BUSY'}
        async with self.execution.lock:
            key = 'request:'+values['request_id']
            fingerprint = json.dumps([name, values], sort_keys=True, allow_nan=False)
            previous = self.execution.load(key)
            if previous:
                if previous['fingerprint'] != fingerprint:
                    return {'success': False, 'status': 'REQUEST_ID_REUSED'}
                return previous.get('result') or {
                    'success': False, 'status': 'REQUEST_OUTCOME_UNKNOWN',
                    'context_id': previous['context_id']}
            context_id = ''
            try:
                self.guard()
                await self.current_state()
                context_id = self.own(values['context_id'], create=True)
                self.execution.save(key, {'fingerprint': fingerprint, 'context_id': context_id})
                spec = self.make_step(name, values)
                await self.validate_bindings(context_id, [spec])
                from my_actions.srv import RunPipelineStep
                response = await self.transport.typed('run_step', RunPipelineStep.Request(
                    context_id=context_id, step_id=spec.step_id, function_name=name,
                    request_cdr=spec.request_cdr, bindings=spec.bindings,
                    expected_revision=values.get('expected_revision', 0)), 600.)
                result = self.project(response.result, context_id)
                result.update(success=response.success, status=response.status,
                              context_id=context_id, revision=response.revision)
                self.execution.save(key, {'fingerprint': fingerprint, 'result': result})
                return result
            except asyncio.CancelledError:
                self.execution.save('preparation_barrier', {
                    'context_id': context_id, 'request_id': values['request_id']})
                raise  # ROS calculation may still mutate caches; block concurrent execution.
            except Exception as error:
                if isinstance(error, TimeoutError):
                    self.execution.save('preparation_barrier', {'context_id': context_id})
                result = {'success': False, 'status': str(error), 'context_id': context_id}
                self.execution.save(key, {'fingerprint': fingerprint, 'result': result})
                return result

    def project(self, item, context_id):
        """Decode trusted master CDR locally; CDR never becomes an LLM argument."""
        result = {'step_id': item.step_id, 'function_name': item.function_name,
                  'success': item.success, 'status': item.status, 'seconds': item.wall_seconds,
                  'result_ref': {'context_id': context_id, 'step_id': item.step_id,
                                 'source_field': '@response_cdr'}, 'output': None}
        if item.response_cdr and item.function_name in FUNCTIONS:
            decoded = deserialize_message(bytes(item.response_cdr),
                                          FUNCTIONS[item.function_name].service_type.Response)
            output = result_summary(decoded, context_id, item.step_id)
            if len(json.dumps(output, allow_nan=False)) > 65536:
                output = {'result_ref': result['result_ref'], 'omitted': True}
            if (item.function_name == 'plan_next_best_view' and decoded.success and
                    len(decoded.path_flat) >= 6 and len(decoded.path_flat) % 6 == 0):
                result['endpoint_joints'] = list(decoded.path_flat[-6:])
            result['output'] = output
        return result

    async def submit(self, values):
        """Queue calculation-only steps; execution remains per-stage guarded MCP."""
        from my_actions.srv import SubmitPipelineSteps
        if self.execution.lock.locked():
            return {'success': False, 'status': 'MCP_PIPELINE_OR_EXECUTION_BUSY'}
        async with self.execution.lock:
            key = 'request:'+values['request_id']
            fingerprint = json.dumps(['submit_steps', values], sort_keys=True)
            previous = self.execution.load(key)
            if previous:
                if previous['fingerprint'] != fingerprint:
                    return {'success': False, 'status': 'REQUEST_ID_REUSED'}
                return previous.get('result') or {'success': False, 'status': 'SUBMIT_UNKNOWN',
                                                  'context_id': previous['context_id']}
            context_id = ''
            try:
                self.guard()
                await self.current_state()
                context_id = self.own(values['context_id'], create=True)
                steps = [self.make_step(item['function_name'], item) for item in values['steps']]
                await self.validate_bindings(context_id, steps, queued=True)
                self.execution.save(key, {'fingerprint': fingerprint, 'context_id': context_id})
                result = await self.transport.typed('submit_steps', SubmitPipelineSteps.Request(
                    context_id=context_id, steps=steps,
                    expected_revision=values.get('expected_revision', 0)), 10.)
                summary = {'success': result.success, 'status': result.status,
                           'context_id': result.context_id, 'revision': result.revision,
                           'completion': 'QUEUED_ONLY'}
                self.execution.save(key, {'fingerprint': fingerprint, 'result': summary})
                return summary
            except Exception as error:
                if isinstance(error, TimeoutError):
                    self.execution.save('preparation_barrier', {'context_id': context_id})
                result = {'success': False, 'status': str(error), 'context_id': context_id}
                self.execution.save(key, {'fingerprint': fingerprint, 'result': result})
                return result

    async def inspect(self, context_id, include_results):
        """Read original context state and safe result projections for this owner."""
        from my_actions.srv import GetStepContext
        self.own(context_id)
        response = await self.transport.typed('get_step_context', GetStepContext.Request(
            context_id=context_id, include_results=include_results), 30.)
        results = [self.project(item, context_id) for item in response.results]
        if len(json.dumps(results, allow_nan=False)) > 524288:
            results = [dict(item, output={'omitted': True, 'result_ref': item['result_ref']})
                       for item in results]
        return {'success': response.success, 'status': response.status,
                'state_json': response.state_json, 'results': results}

    async def run(self, values):
        """Expose the real ROS action as start/status/cancel, with explicit outcomes."""
        context_id = self.own(values['context_id'])
        operation = values['operation']
        job_key = 'job:'+values['request_id']
        fingerprint = json.dumps({k: v for k, v in values.items() if k != 'operation'},
                                 sort_keys=True)
        saved = self.execution.load(job_key)
        if saved and saved['context_id'] != context_id:
            return {'success': False, 'status': 'JOB_CONTEXT_MISMATCH'}
        if operation == 'status':
            return saved or {'success': False, 'status': 'JOB_UNKNOWN'}
        if operation == 'cancel':
            job = self.jobs.get(values['request_id'])
            if not job or not job.get('goal'):
                return {'success': False, 'status': 'NO_LOCAL_ACTIVE_GOAL',
                        'context_id': context_id}
            accepted = await self.transport.cancel(job['goal'])
            return {'success': accepted, 'status': 'CANCEL_REQUESTED' if accepted else
                    'CANCEL_NOT_ACCEPTED', 'context_id': context_id,
                    'completion': 'NOT_A_ROBOT_STOP'}
        if saved:
            if saved.get('fingerprint') != fingerprint:
                return {'success': False, 'status': 'REQUEST_ID_REUSED'}
            return saved  # An existing goal is never submitted again.
        if self.execution.lock.locked():
            return {'success': False, 'status': 'MCP_PIPELINE_OR_EXECUTION_BUSY'}
        await self.execution.lock.acquire()
        try:
            self.guard()
            await self.current_state()
            snapshot = await self.inspect(context_id, False)
            if not snapshot['success']:
                raise ValueError(snapshot['status'])
            state = json.loads(snapshot['state_json'])[0]
            if state['busy'] or state['revision'] != values['expected_revision']:
                raise ValueError('BUSY_OR_STALE_CONTEXT_REVISION')
            if any(item['function'] not in PREPARATION for item in state['pending']):
                raise ValueError('UNGUARDED_PHYSICAL_QUEUE_FORBIDDEN')
            job = {'success': True, 'status': 'STARTING', 'context_id': context_id,
                   'completion': 'ASYNC_ACK_ONLY', 'fingerprint': fingerprint}
            self.execution.save(job_key, job)
            goal = await self.transport.start(context_id, state['revision'],
                                              values.get('stop_before_step_id', ''))
            self.jobs[values['request_id']] = {'goal': goal}
            job['status'] = 'RUNNING'
            self.execution.save(job_key, job)
            self.jobs[values['request_id']]['task'] = asyncio.create_task(
                self._finish(values['request_id'], context_id, goal, fingerprint))
            return job
        except BaseException as error:
            if isinstance(error, (TimeoutError, asyncio.CancelledError)):
                self.execution.save('preparation_barrier', {'context_id': context_id})
            self.execution.lock.release()
            if isinstance(error, asyncio.CancelledError):
                raise
            result = {'success': False, 'status': str(error), 'context_id': context_id}
            self.execution.save(job_key, result)
            return result

    async def _finish(self, job_id, context_id, goal, fingerprint):
        try:
            response, action_status = await self.transport.finish(goal)
            result = {'success': response.success, 'status': response.status,
                      'context_id': context_id, 'completion': 'FINAL_ACTION_RESULT',
                      'ros_action_status': action_status, 'seconds': response.wall_seconds,
                      'results': [self.project(item, context_id) for item in response.results]}
        except BaseException as error:
            self.execution.save('preparation_barrier', {'context_id': context_id, 'job': job_id})
            result = {'success': False, 'status': 'ACTION_OUTCOME_UNKNOWN: '+str(error),
                      'context_id': context_id}
        finally:
            self.execution.lock.release()
        result['fingerprint'] = fingerprint
        self.execution.save('job:'+job_id, result)
        self.jobs.pop(job_id, None)


class RosPreparationTransport:
    """Only StepStore services and its completion action, no actuator clients."""

    def __init__(self, node):
        from my_actions.action import RunPipelineSteps
        from rclpy.action import ActionClient
        self.clients = {name: node.create_client(FUNCTIONS[name].service_type,
                                                 FUNCTIONS[name].endpoint) for name in (
            'run_step', 'submit_steps', 'get_step_context', 'get_world_state')}
        self.action = ActionClient(node, RunPipelineSteps, '/pipeline/run_steps')

    async def typed(self, name, query, timeout):
        return await asyncio.to_thread(self._typed, name, query, timeout)

    def _typed(self, name, query, timeout):
        client = self.clients[name]
        if not client.wait_for_service(timeout_sec=3.):
            raise RuntimeError('BACKEND_UNAVAILABLE: '+name)
        return self.wait(client.call_async(query), timeout)

    @staticmethod
    def wait(future, timeout):
        import threading
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            future.cancel()  # Abandon local wait only, never imply ROS cancellation.
            raise TimeoutError('ROS_OUTCOME_UNKNOWN')
        return future.result()

    async def start(self, context_id, revision, boundary):
        from my_actions.action import RunPipelineSteps
        if not await asyncio.to_thread(self.action.wait_for_server, timeout_sec=3.):
            raise RuntimeError('RUN_STEPS_ACTION_UNAVAILABLE')
        goal = await asyncio.to_thread(self.wait, self.action.send_goal_async(
            RunPipelineSteps.Goal(context_id=context_id, expected_revision=revision,
                                  stop_before_step_id=boundary)), 10.)
        if not goal.accepted:
            raise RuntimeError('ACTION_GOAL_REJECTED')
        return goal

    async def finish(self, goal):
        result = await asyncio.to_thread(self.wait, goal.get_result_async(), 600.)
        return result.result, result.status

    async def cancel(self, goal):
        response = await asyncio.to_thread(self.wait, goal.cancel_goal_async(), 10.)
        return bool(response.goals_canceling)
