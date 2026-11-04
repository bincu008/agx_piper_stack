"""Authenticated read-only MCP over loopback or private-network TLS."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import hmac
import ipaddress
import json
from pathlib import Path
import threading

import jsonschema

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings

from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

from task_pipeline.registry import catalog, FUNCTIONS

from .contracts import (
    ALLOWLIST, CALCULATIONS, COMMON, DISABLED_COMMON, MANIPULATION, READS)
from .preparation import (
    call_schema, IDENTIFIER, OUTPUT_SCHEMA, PREPARATION, step_schema)
from .schema import message_schema


def execution_reference_schema():
    """Describe gateway correlation without modifying the ROS request type."""
    return {'type': 'object', 'additionalProperties': False,
            'required': ['task_id', 'object_name', 'plan_revision', 'attempt'],
            'properties': {'task_id': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                           'object_name': {'type': 'string', 'minLength': 1, 'maxLength': 256},
                           'plan_revision': {'type': 'integer', 'minimum': 1,
                                             'maximum': 2**64 - 1},
                           'attempt': {'const': 1}}}


def definitions(execution_enabled=False, preparation_enabled=False):
    """Describe readable tools and blocked manipulation APIs separately."""
    tools = []
    for name, description in {**READS, **CALCULATIONS}.items():
        function = FUNCTIONS[name]
        output = {'type': 'object', 'additionalProperties': False,
                  'required': ['transport_outcome', 'response', 'detail'],
                  'properties': {
                      'transport_outcome': {'enum': ['response', 'unavailable', 'unknown']},
                      'response': {'anyOf': [message_schema(function.service_type.Response),
                                             {'type': 'null'}]},
                      'detail': {'type': 'string'}}}
        if name == 'get_execution_state' and execution_enabled:
            output['properties']['execution_ref'] = {
                'anyOf': [execution_reference_schema(), {'type': 'null'}]}
            output['required'].append('execution_ref')
            output['properties']['selected_pose'] = {'anyOf': [message_schema(
                FUNCTIONS['commit_floor_slots'].service_type.Request)['properties'][
                    'selected_pose'], {'type': 'null'}]}
            output['properties']['gateway_status'] = {'type': 'string'}
            output['required'] += ['selected_pose', 'gateway_status']
        tools.append(types.Tool(name=name,
                                description=description + (COMMON if name in READS else ''),
                                inputSchema=message_schema(function.service_type.Request),
                                outputSchema=output,
                                annotations=types.ToolAnnotations(
                                    readOnlyHint=name in READS, destructiveHint=False,
                                    idempotentHint=name in READS, openWorldHint=False)))
    for name, description in MANIPULATION.items():
        function = FUNCTIONS[name]
        request_schema = message_schema(function.service_type.Request)
        if name == 'execute_plan':
            request_schema['properties']['segment']['enum'] = [
                'look', 'pregrasp', 'approach', 'lift', 'transfer', 'descent']
        output = {
            'type': 'object', 'additionalProperties': False,
            'required': ['transport_outcome', 'dispatched', 'response', 'detail'],
            'properties': {'transport_outcome': {'const': 'disabled'},
                           'dispatched': {'const': False}, 'response': {'type': 'null'},
                           'detail': {'type': 'string'}}}
        if execution_enabled:
            request_schema['properties']['request_id'] = {
                'type': 'string', 'minLength': 1, 'maxLength': 128}
            request_schema['required'].append('request_id')
            if name != 'move_to':
                request_schema['properties']['execution_ref'] = execution_reference_schema()
                request_schema['required'].append('execution_ref')
            output['properties'].update(
                transport_outcome={'enum': ['response', 'rejected', 'unknown']},
                dispatched={'type': 'boolean'},
                response={'anyOf': [message_schema(function.service_type.Response),
                                    {'type': 'null'}]})
        enabled_description = (
            ' MCP prepared execution is enabled, subject to the master physical gate. '
            'First call get_execution_state and use its exact execution_ref. Supply '
            'a unique durable request_id for each operation; repeat only the same '
            'request ID to retrieve its stored outcome without reexecution. One '
            'grasp attempt per prepared task; no autonomous physical retry or '
            'alternative after failure. Unknown outcomes block all continuation. '
            'Only complete prepared pick/place sessions are supported. See workflow '
            'resources for ordering, 300s backend timeout and local recovery.')
        tools.append(types.Tool(
            name=name, title=name + (
                ' (guarded execution)' if execution_enabled else ' (execution disabled)'),
            description=description + (
                enabled_description if execution_enabled and name != 'move_to' else
                ' Guarded empty-hand execution enabled; use a durable request_id.'
                if execution_enabled else DISABLED_COMMON),
            inputSchema=request_schema,
            outputSchema=output,
            annotations=types.ToolAnnotations(
                readOnlyHint=False, destructiveHint=True,
                idempotentHint=False, openWorldHint=False),
            _meta={'rosPipeline': {
                'availability': 'guarded_execution' if execution_enabled else 'disabled',
                'endpoint': function.endpoint,
                'serviceType': 'my_actions/srv/' + function.service_type.__name__,
                'requestSchema': message_schema(function.service_type.Request),
                'responseSchema': message_schema(function.service_type.Response)}}))
    if preparation_enabled:
        for name, purpose in PREPARATION.items():
            function = FUNCTIONS[name]
            tools.append(
                types.Tool(
                    name=name,
                    description=purpose + (
                        ' Non-actuating; may update master memory/caches. arguments contains '
                        'only primitive ROS fields; bind messages through same-context prior '
                        'step outputs. Empty context_id creates a gateway-owned context. '
                        'Use unique step_id/request_id; duplicate request_id returns its receipt. '
                        'Final service result, 600s wait; timeout is unknown, never auto-replay. '
                        'Check success/status and output; downstream steps require successful '
                        'sources and matching observation headers. Units: metres/radians, '
                        'geometry base_link unless specified by the exact ROS field/header. '
                        'See workflow resources for dependencies and recovery.'),
                    inputSchema=call_schema(name),
                    outputSchema=OUTPUT_SCHEMA,
                    annotations=types.ToolAnnotations(
                        readOnlyHint=name in (
                            'get_active_target',
                            'get_cached_grasp',
                            'get_prepared_pick_place',
                            'get_prepared_grasp_lift',
                            'check_target_view'),
                        destructiveHint=False,
                        idempotentHint=False,
                        openWorldHint=False),
                    _meta={
                        'rosPipeline': {
                            'availability': 'calculation_only',
                            'endpoint': function.endpoint,
                            'requestSchema': message_schema(
                                function.service_type.Request),
                            'responseSchema': message_schema(
                                function.service_type.Response)}}))
        queue_schemas = {
            'run_step': call_schema(),
            'submit_steps': {
                'type': 'object', 'additionalProperties': False,
                'required': ['context_id', 'request_id', 'steps'],
                'properties': {'context_id': {'type': 'string', 'maxLength': 128},
                               'request_id': IDENTIFIER,
                               'expected_revision': {'type': 'integer', 'minimum': 0},
                               'steps': {'type': 'array', 'minItems': 1, 'maxItems': 64,
                                         'items': step_schema()}}},
            'run_steps': {
                'type': 'object', 'additionalProperties': False,
                'required': ['operation', 'context_id', 'request_id'],
                'properties': {'operation': {'enum': ['start', 'status', 'cancel']},
                               'context_id': {'type': 'string', 'minLength': 1},
                               'request_id': IDENTIFIER,
                               'expected_revision': {'type': 'integer', 'minimum': 1},
                               'stop_before_step_id': {'type': 'string', 'maxLength': 128}},
                'allOf': [{'if': {'properties': {'operation': {'const': 'start'}}},
                           'then': {'required': ['expected_revision']}}]},
        }
        for name, schema in queue_schemas.items():
            tools.append(types.Tool(
                name=name, description=(
                    'Master-owned calculation-only pipeline. Physical functions are rejected; '
                    'use guarded per-stage execution tools after complete preparation. '
                    'run_step returns final service completion (600s). submit_steps only queues '
                    '(10s). run_steps start acknowledges asynchronous ROS action; poll status '
                    'with same context/request_id until FINAL_ACTION_RESULT; cancel requests '
                    'between-step cancellation and is not robot stop. Start requires exact '
                    'submit/context revision. Disconnect never replays a goal. '
                    'get_step_context reads retained source handles and results.'),
                inputSchema=schema, outputSchema=OUTPUT_SCHEMA,
                annotations=types.ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                                  idempotentHint=False, openWorldHint=False)))
        for tool in tools:
            if tool.name == 'get_step_context':
                tool.outputSchema = OUTPUT_SCHEMA
                tool.description = ('Inspect a gateway-owned master context; returns final '
                                    'state_json and optional bounded result metadata/handles, '
                                    'never binary image/cloud CDR. No mutation; 30s wait.')
    tools.append(
        types.Tool(
            name='list_functions',
            description=(
                'Read local authoritative ROS catalog plus exact MCP contracts. '
                'Empty input, no ROS calls, final snapshot. ROS catalog entries are NOT '
                'enabled MCP tools. allowlist identifies callable tools; disabled_tools '
                'identifies discoverable blocked APIs. No mutation/retry/cancellation '
                'effects. Next read '
                'ros-pipeline://workflow and ros-pipeline://nanobot.'),
            inputSchema={
                'type': 'object',
                'properties': {},
                'additionalProperties': False},
            outputSchema={
                'type': 'object',
                'required': [
                        'mode',
                        'allowlist',
                        'ros_catalog',
                        'mcp_tools',
                        'disabled_tools'],
                'additionalProperties': False,
                'properties': {
                    'input_provenance': {'type': 'object'},
                    'mode': {
                        'enum': [
                            'read_only',
                            'prepared_execution',
                            'agent_pipeline',
                            'perception_only',
                            'preparation_only']},
                    'disabled_tools': {
                        'type': 'array',
                        'items': {
                            'type': 'string'}},
                        'allowlist': {
                        'type': 'array',
                                'items': {
                                    'type': 'string'}},
                        'ros_catalog': {
                                'type': 'array',
                                'items': {
                                    'type': 'object'}},
                        'mcp_tools': {
                                'type': 'array',
                                'items': {
                                    'type': 'object'}}}},
            annotations=types.ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False)))
    return tools


def create_app(
        bridge,
        token,
        documents,
        port=8765,
        host='127.0.0.1',
        execution=None,
        preparation=None,
        perception_only=False,
        require_live_inputs=False,
        preparation_only=False):
    """Construct transport with an injected read bridge, never a mutation router."""
    if preparation_only and not require_live_inputs:
        raise ValueError('PREPARATION_MODE_REQUIRES_LIVE_INPUT_GUARD')
    if require_live_inputs and not (perception_only or preparation_only):
        raise ValueError('LIVE_INPUT_GUARD_REQUIRES_PERCEPTION_ONLY_MODE')
    if len(token) < 32:
        raise ValueError('MCP bearer token must contain at least 32 characters')
    server = Server('ros-pipeline' if execution else 'ros-pipeline-read-only', version='3.2.0',
                    instructions='Read ros-pipeline://workflow and ros-pipeline://nanobot first. '
                    'Use exact execution references and request IDs. '
                    'Never infer physical readiness.')
    tools = {tool.name: tool for tool in definitions(
        execution is not None, preparation is not None)}
    unavailable = []
    if perception_only or preparation_only:
        if preparation is None:
            raise ValueError('PERCEPTION_MODE_REQUIRES_TYPED_PREPARATION_ADAPTER')
        allow = {'list_functions', 'get_world_state', 'get_arm_state',
                 'get_spatial_context', 'get_spatial_map', 'get_manipulation_memory',
                 'get_step_context', 'capture_observation', 'verify_presence',
                 'ground_object', 'segment_object'}
        if preparation_only:
            allow = set(READS) | set(CALCULATIONS) | set(PREPARATION) | {
                'list_functions', 'run_step', 'submit_steps', 'run_steps'}
        unavailable = sorted(set(tools)-allow)
        tools = {name: tool for name, tool in tools.items() if name in allow}
    documents = Path(documents)
    resources = {'ros-pipeline://preparation': ('Non-actuating preparation and verification',
                                                'PREPARATION_ONLY.md'),
                 'ros-pipeline://live-perception': ('Live Qwen perception and provenance',
                                                    'LIVE_PERCEPTION.md'),
                 'ros-pipeline://spatial-map': ('Shared grid layout and camera coverage',
                                                'SPATIAL_MAP.md'),
                 'ros-pipeline://workflow': ('Workflow and dependencies', 'WORKFLOW.md'),
                 'ros-pipeline://nanobot': ('Reusable Nanobot instructions', 'NANOBOT.md')}
    if preparation is not None:
        resources['ros-pipeline://agent-pipeline'] = (
            'Agent-composed manipulation and result bindings', 'AGENT_PIPELINE.md')

    @server.list_tools()
    async def list_tools():
        return list(tools.values())

    @server.call_tool()
    async def call_tool(name, arguments):
        # Independent enforcement: no metadata/annotation is trusted as a gate.
        if (perception_only or preparation_only) and (name in MANIPULATION or any(
                item['name'] == name and item.get('effect') == 'actuation'
                for item in catalog())):
            raise ValueError('NON_ACTUATING_MODE_PHYSICAL_DISPATCH_FORBIDDEN')
        if name not in tools:
            raise ValueError('TOOL_DISABLED_OR_UNKNOWN_FOR_SERVER_MODE')
        guarded_calculation = (name in PREPARATION or name in CALCULATIONS or
                               name in {'run_step', 'submit_steps'} or
                               (name == 'run_steps' and arguments.get('operation') == 'start'))
        if require_live_inputs and guarded_calculation:
            provenance = bridge.input_provenance()
            if provenance['classification'] != 'live_driver_publishers':
                result = {'success': False, 'status': 'LIVE_INPUT_PROVENANCE_REQUIRED',
                          'input_provenance': provenance}
                return types.CallToolResult(
                    content=[types.TextContent(type='text', text=json.dumps(result))],
                    structuredContent=result, isError=True)

        if preparation_only and name == 'get_observe_view':
            jsonschema.validate(arguments, tools[name].inputSchema)
            if execution.lock.locked():
                result = {'transport_outcome': 'unavailable', 'response': None,
                          'detail': 'MCP_PIPELINE_OR_EXECUTION_BUSY'}
            else:
                async with execution.lock:
                    try:
                        preparation.guard()
                        await preparation.current_state()
                        result = await bridge.call(name, arguments)
                        if result['transport_outcome'] == 'unknown':
                            execution.save('preparation_barrier', {'operation': name})
                    except BaseException as error:
                        if isinstance(error, asyncio.CancelledError):
                            execution.save('preparation_barrier', {'operation': name})
                            raise
                        result = {'transport_outcome': 'unavailable', 'response': None,
                                  'detail': str(error)}
            jsonschema.validate(result, tools[name].outputSchema)
            return types.CallToolResult(
                content=[types.TextContent(type='text', text=json.dumps(result))],
                structuredContent=result,
                isError=(result['transport_outcome'] != 'response' or
                         (result.get('response') or {}).get('success') is False))

        if preparation is not None and name in (
                set(PREPARATION) | {'run_step', 'submit_steps', 'run_steps', 'get_step_context'}):
            jsonschema.validate(arguments, tools[name].inputSchema)
            try:
                if name == 'get_step_context':
                    result = await preparation.inspect(arguments['context_id'],
                                                       arguments['include_results'])
                elif name == 'submit_steps':
                    result = await preparation.submit(arguments)
                elif name == 'run_steps':
                    result = await preparation.run(arguments)
                else:
                    function_name = arguments.get('function_name') if name == 'run_step' else name
                    result = await preparation.call(function_name, arguments)
            except Exception as error:
                result = {'success': False, 'status': str(error)}
            jsonschema.validate(result, tools[name].outputSchema)
            return types.CallToolResult(
                content=[types.TextContent(type='text', text=json.dumps(result, allow_nan=False))],
                structuredContent=result, isError=not result['success'])
        if name in MANIPULATION:
            if execution is not None:
                jsonschema.validate(arguments, tools[name].inputSchema)
                result = await execution.call(name, arguments)
                jsonschema.validate(result, tools[name].outputSchema)
                return types.CallToolResult(
                    content=[types.TextContent(type='text', text=json.dumps(result))],
                    structuredContent=result,
                    isError=result['transport_outcome'] != 'response')
            return types.CallToolResult(
                content=[types.TextContent(type='text', text='MCP_EXECUTION_DISABLED')],
                structuredContent={
                    'transport_outcome': 'disabled', 'dispatched': False, 'response': None,
                    'detail': 'MCP_EXECUTION_DISABLED: discoverable only; no ROS call or '
                    'pipeline task submitted. Inspect execution state and workflow resources.'},
                isError=True)
        if name not in ALLOWLIST or name not in tools:
            raise ValueError('READ_ONLY_SERVER: state-changing/unknown tool disabled')
        jsonschema.validate(arguments, tools[name].inputSchema)
        if name == 'list_functions':
            result = {'mode': ('preparation_only' if preparation_only else
                               'perception_only' if perception_only else
                               'agent_pipeline' if preparation else
                               'prepared_execution' if execution else 'read_only'),
                      'allowlist': sorted(tools if execution else ALLOWLIST),
                      'disabled_tools': (unavailable if (perception_only or preparation_only) else
                                         [] if execution else sorted(MANIPULATION)),
                      'ros_catalog': catalog(),
                      'mcp_tools': [tool.model_dump(by_alias=True, exclude_none=True)
                                    for tool in tools.values()]}
        else:
            result = await bridge.call(name, arguments)
            if name == 'get_execution_state' and execution is not None:
                from .execution import reference
                state = result.get('response') or {}
                result['execution_ref'] = ({
                    'object_name': arguments['object_name'],
                    'plan_revision': state['plan_revision'],
                    'task_id': reference(arguments['object_name'], state), 'attempt': 1}
                    if state.get('success') and state.get('plan_revision') else None)
                current = execution.load('current')
                result['gateway_status'] = (
                    'MCP_STATE_UNKNOWN_REQUIRES_LOCAL_RECONCILIATION'
                    if current and current.get('unknown') else 'MASTER_CONTROLLED')
                ref = result['execution_ref']
                result['selected_pose'] = (
                    current['selected_pose'] if current and ref and
                    current['task_id'] == ref['task_id'] else None)
        if name == 'list_functions' and hasattr(bridge, 'input_provenance'):
            result['input_provenance'] = bridge.input_provenance()
        jsonschema.validate(result, tools[name].outputSchema)
        raw = result.get('response') or {}
        error = (result.get('transport_outcome', 'response') != 'response' or
                 result.get('gateway_status', '').startswith('MCP_STATE_UNKNOWN') or
                 raw.get('success') is False or 'STATE_UNKNOWN' in raw.get('status', ''))
        return types.CallToolResult(
            content=[types.TextContent(type='text', text=json.dumps(result, allow_nan=False))],
            structuredContent=result, isError=error)

    @server.list_resources()
    async def list_resources():
        return [types.Resource(uri=uri, name=title, mimeType='text/markdown')
                for uri, (title, _) in resources.items()] + [
                    types.Resource(uri='ros-pipeline://catalog', name='Exact MCP tool schemas',
                                   mimeType='application/json')]

    @server.read_resource()
    async def read_resource(uri):
        from mcp.server.lowlevel.helper_types import ReadResourceContents
        if str(uri) == 'ros-pipeline://catalog':
            text = json.dumps([tool.model_dump(by_alias=True, exclude_none=True)
                               for tool in tools.values()], indent=2)
            return [ReadResourceContents(content=text, mime_type='application/json')]
        if str(uri) not in resources:
            raise ValueError('Unknown resource')
        return [ReadResourceContents(content=(documents / resources[str(uri)][1]).read_text(),
                                     mime_type='text/markdown')]

    manager = StreamableHTTPSessionManager(
        app=server, json_response=True, stateless=True,
        max_request_body_size=65536,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[f'127.0.0.1:{port}', f'localhost:{port}', f'{host}:{port}'],
            allowed_origins=[f'http://127.0.0.1:{port}', f'http://localhost:{port}',
                             f'https://{host}:{port}']))

    # ASGI endpoint avoids Starlette Request internals for streaming responses.
    class AuthenticatedEndpoint:
        async def __call__(self, scope, receive, send):
            headers = dict(scope.get('headers', []))
            if not hmac.compare_digest(headers.get(b'authorization', b''),
                                       ('Bearer ' + token).encode()):
                await Response('Unauthorized', status_code=401)(scope, receive, send)
                return
            await manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    return Starlette(routes=[Route('/mcp', AuthenticatedEndpoint(),
                                   methods=['GET', 'POST', 'DELETE'])], lifespan=lifespan)


def main():
    """Start guarded prepared execution; master physical permission is independent."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1', help='Loopback or explicit private IPv4')
    parser.add_argument('--tls-cert')
    parser.add_argument('--tls-key')
    parser.add_argument('--token-file', required=True)
    parser.add_argument('--documents', help='Override installed workflow reference directory')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--read-only', action='store_true', help='Disable execution dispatch')
    parser.add_argument('--require-live-inputs', action='store_true',
                        help='Require single live camera/arm publishers')
    mode.add_argument('--preparation-only', action='store_true',
                      help='Expose non-actuating preparation; requires live guard and false gate')
    mode.add_argument('--perception-only', action='store_true',
                      help='Expose only live capture/inference and inspection tools')
    parser.add_argument('--execution-ledger', default=str(
        Path.home() / '.local/state/agx-piper-mcp/execution.sqlite'))
    args = parser.parse_args()
    validate_bind(args.host, args.tls_cert, args.tls_key)
    token_path = Path(args.token_file)
    if token_path.stat().st_mode & 0o077:
        raise ValueError('Token file must be private (chmod 600)')
    token = token_path.read_text().strip()
    if args.documents is None:
        from ament_index_python.packages import get_package_share_directory
        args.documents = str(Path(get_package_share_directory('task_pipeline')) / 'mcp')
    import rclpy
    import uvicorn
    from .bridge import ReadBridge
    from .execution import PreparedExecution, RosExecutionTransport
    from .preparation import PreparationGateway, RosPreparationTransport
    rclpy.init()
    node = rclpy.create_node('pipeline_mcp_read_only')
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        execution = None if args.read_only else PreparedExecution(
            None if (args.perception_only or args.preparation_only) else
            RosExecutionTransport(node), args.execution_ledger)
        uvicorn.run(create_app(ReadBridge(node), token, args.documents, args.port, args.host,
                               execution=execution, perception_only=args.perception_only,
                               preparation_only=args.preparation_only,
                               require_live_inputs=args.require_live_inputs,
                               preparation=(
                                   PreparationGateway(
                                       RosPreparationTransport(node), execution,
                                       require_physical_disabled=args.preparation_only)
                                   if execution else None)),
                    host=args.host, port=args.port, access_log=False,
                    ssl_certfile=args.tls_cert, ssl_keyfile=args.tls_key)
    finally:
        rclpy.shutdown()
        spin.join(timeout=2)
        node.destroy_node()


def validate_bind(host, cert, key):
    """Never expose plaintext/private credentials on wildcard or public listeners."""
    address = ipaddress.IPv4Address(host)
    if address.is_loopback:
        return
    networks = ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '100.64.0.0/10')
    if not any(address in ipaddress.ip_network(network) for network in networks):
        raise ValueError('Only explicit loopback/private IPv4 binding is supported')
    if not cert or not key:
        raise ValueError('Private-network binding requires TLS certificate and key')


if __name__ == '__main__':
    main()
