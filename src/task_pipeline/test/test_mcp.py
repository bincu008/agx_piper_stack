"""SIMULATED fixtures only: no ROS node, client, driver or actuator is created."""

from pathlib import Path

import jsonschema

import pytest

pytest.importorskip('mcp', reason='Optional MCP transport dependencies not installed')

from starlette.testclient import TestClient  # noqa: E402

from task_pipeline.mcp.admission import AdmissionLedger  # noqa: E402
from task_pipeline.mcp.contracts import (  # noqa: E402
    ALLOWLIST, DISCOVERABLE, MANIPULATION, READS)
from task_pipeline.mcp.server import create_app, definitions  # noqa: E402


class FakeReads:
    """Cannot reach ROS; records every inspection request."""

    def __init__(self):
        """Initialize isolated state."""
        self.calls = []

    async def call(self, name, arguments):
        """Read an allowlisted snapshot."""
        self.calls.append((name, arguments))
        return {'transport_outcome': 'response',
                'response': {'state_json': '{"SIMULATED":true}'}, 'detail': 'SIMULATED'}


TOKEN = 'simulated-test-token-never-use-live-123456'
DOCS = Path(__file__).resolve().parents[3]/'docs'/'mcp'


def rpc(client, method, params=None, authorized=True, identifier=1):
    """Send a synthetic protocol request."""
    headers = {'Accept': 'application/json, text/event-stream',
               'MCP-Protocol-Version': '2025-06-18'}
    if authorized:
        headers['Authorization'] = 'Bearer '+TOKEN
    return client.post('/mcp', headers=headers, json={
        'jsonrpc': '2.0', 'id': identifier, 'method': method, 'params': params or {}})


def test_discovery_schemas_auth_resources_and_deny_writes():
    """Test discovery schemas auth resources and deny writes."""
    fake = FakeReads()
    with TestClient(create_app(fake, TOKEN, DOCS, port=8765),
                    base_url='http://127.0.0.1:8765') as client:
        assert rpc(client, 'tools/list', authorized=False).status_code == 401
        response = rpc(client, 'tools/list').json()['result']
        assert {t['name'] for t in response['tools']} == DISCOVERABLE
        assert all(t['annotations']['readOnlyHint'] == (t['name'] in READS or
                                                       t['name'] == 'list_functions')
                   for t in response['tools'])
        for tool in response['tools']:
            jsonschema.Draft202012Validator.check_schema(tool['inputSchema'])
            jsonschema.Draft202012Validator.check_schema(tool['outputSchema'])
        for name in ('execute_plan', 'gripper_close', 'run_step', 'submit_steps',
                     'release_placed_object', 'reconcile_empty_hand', 'unknown'):
            result = rpc(client, 'tools/call', {'name': name, 'arguments': {}}).json()
            assert result['result']['isError']
        assert fake.calls == []
        result = rpc(client, 'tools/call', {'name': 'get_world_state',
                                            'arguments': {}}).json()['result']
        assert result['structuredContent']['response']['state_json']
        assert not result['isError']
        result = rpc(client, 'tools/call', {'name': 'get_world_state',
                                            'arguments': {'endpoint': '/master/move_to'}}).json()
        assert result['result']['isError']
        assert len(fake.calls) == 1
        resources = rpc(client, 'resources/list').json()['result']['resources']
        assert len(resources) == 6
        for resource in resources:
            content = rpc(client, 'resources/read', {'uri': resource['uri']}).json()
            assert content['result']['contents'][0]['text']
        catalog = rpc(client, 'tools/call', {'name': 'list_functions',
                                             'arguments': {}}).json()['result']['structuredContent']
        assert catalog['mode'] == 'read_only'
        assert set(catalog['allowlist']) == ALLOWLIST
        assert set(catalog['disabled_tools']) == set(MANIPULATION)


def test_bad_token_and_host():
    """Test bad token and host."""
    with pytest.raises(ValueError):
        create_app(FakeReads(), 'short', DOCS)
    with TestClient(create_app(FakeReads(), TOKEN, DOCS),
                    base_url='http://untrusted.example') as client:
        assert rpc(client, 'tools/list').status_code == 421


def ledger(tmp_path):
    """Create a temporary persistent policy fixture."""
    value = AdmissionLedger(tmp_path/'policy.sqlite')
    value.create('task', 'bowl')
    return value


def begin(value, n=1, **kwargs):
    """Reserve a synthetic grasp attempt."""
    return value.begin('task', 'bowl', f'request{n}', n, state_ready=True,
                       candidate=f'candidate{n}', alternative=n > 1, **kwargs)


def finish(value, n, outcome, **kwargs):
    """Record a correlated synthetic outcome."""
    value.finish('task', 'bowl', f'request{n}', n, n, outcome, **kwargs)


def test_valid_transition_verification_and_async_ack(tmp_path):
    """Test valid transition verification and async ack."""
    value = ledger(tmp_path)
    assert begin(value)['outcome'] == 'running'
    finish(value, 1, 'running')  # asynchronous ACK must not permit placement
    with pytest.raises(ValueError):
        value.placement_allowed('task', 'bowl', 1, 1, state_ready=True)
    finish(value, 1, 'success')
    with pytest.raises(ValueError):
        value.placement_allowed('task', 'bowl', 1, 1, state_ready=True)
    value.verify('task', 'bowl', 1, 1, success=True, contact=True)
    assert value.placement_allowed('task', 'bowl', 1, 1, state_ready=True)
    with pytest.raises(ValueError):
        value.placement_allowed('task', 'bowl', 1, 1, state_ready=False)


def test_failure_supported_alternative_and_retry_exhaustion(tmp_path):
    """Test failure supported alternative and retry exhaustion."""
    value = ledger(tmp_path)
    begin(value)
    finish(value, 1, 'recoverable_failure', alternatives=['candidate2'])
    assert begin(value, 2)['dispatch']
    finish(value, 2, 'recoverable_failure', alternatives=['candidate3'])
    assert begin(value, 3)['dispatch']
    finish(value, 3, 'recoverable_failure', alternatives=['candidate4'])
    value.db.close()
    value = AdmissionLedger(tmp_path/'policy.sqlite')
    value.create('task', 'bowl')  # reconnect cannot reset count
    with pytest.raises(ValueError, match='EXHAUSTED'):
        begin(value, 4)


@pytest.mark.parametrize('outcome', ['failure', 'cancelled', 'unknown'])
def test_no_automatic_alternative_after_unsafe_outcome(tmp_path, outcome):
    """Test no automatic alternative after unsafe outcome."""
    value = ledger(tmp_path)
    begin(value)
    finish(value, 1, outcome)
    with pytest.raises(ValueError):
        begin(value, 2)


def test_stale_mismatch_and_missing_alternative(tmp_path):
    """Test stale mismatch and missing alternative."""
    value = ledger(tmp_path)
    begin(value)
    for args in [('task', 'cup', 'request1', 1, 1),
                 ('task', 'bowl', 'request1', 2, 1),
                 ('task', 'bowl', 'request1', 1, 2),
                 ('task', 'bowl', 'wrong', 1, 1)]:
        with pytest.raises(ValueError):
            value.finish(*args, 'success')
    finish(value, 1, 'success')
    for obj, attempt, revision in [('cup', 1, 1), ('bowl', 2, 1), ('bowl', 1, 2)]:
        with pytest.raises(ValueError):
            value.verify('task', obj, attempt, revision, success=True, contact=True)
    with pytest.raises(ValueError):
        value.verify('task', 'bowl', 1, 1, success=True, contact=False)
    with pytest.raises(ValueError):
        begin(value, 2)


def test_duplicate_disconnect_timeout_and_restart(tmp_path):
    """Test duplicate disconnect timeout and restart."""
    value = ledger(tmp_path)
    assert begin(value)['dispatch']
    assert not begin(value)['dispatch']
    with pytest.raises(ValueError, match='REQUEST_ID_REUSED'):
        value.begin('task', 'cup', 'request1', 1, state_ready=True, candidate='candidate1')
    value.interrupted('task')  # timeout/disconnect; never claims motion stopped
    assert value.read('task')['outcome'] == 'unknown'
    assert not begin(value)['dispatch']
    with pytest.raises(ValueError):
        finish(value, 1, 'success')  # late response requires reconciliation
    with pytest.raises(ValueError):
        begin(value, 2)
    value.db.close()
    restored = AdmissionLedger(tmp_path/'policy.sqlite')
    assert restored.read('task')['outcome'] == 'unknown'


def test_crash_running_becomes_unknown(tmp_path):
    """Test crash running becomes unknown."""
    value = ledger(tmp_path)
    begin(value)
    value.db.close()
    restored = AdmissionLedger(tmp_path/'policy.sqlite')
    assert restored.read('task')['outcome'] == 'unknown'


def test_mutation_catalog_entries_never_authorized():
    """Visible action cards never expand the read-only dispatch allowlist."""
    from task_pipeline.registry import FUNCTIONS
    names = {tool.name for tool in definitions()}
    assert names == DISCOVERABLE
    assert not set(MANIPULATION) & ALLOWLIST
    assert len(set(FUNCTIONS)-names) > 50


def test_bridge_timeout_unavailable_and_read_only_boundary():
    """Test bridge timeout unavailable and read only boundary."""
    import asyncio
    from task_pipeline.mcp.bridge import ReadBridge

    class Future:
        cancelled = False

        def add_done_callback(self, callback):
            pass  # deliberately incomplete synthetic service

        def cancel(self):
            self.cancelled = True

    class Client:
        available = True

        def __init__(self):
            self.future = Future()

        def wait_for_service(self, timeout_sec):
            return self.available

        def call_async(self, request):
            return self.future

    class Node:
        def __init__(self):
            self.created = []

        def create_client(self, kind, endpoint):
            self.created.append(endpoint)
            return Client()

    node = Node()  # no rclpy.init(), actual ROS node or DDS connection
    bridge = ReadBridge(node, timeout=.001)
    assert len(node.created) == len(ALLOWLIST)-1
    assert set(bridge.clients) == ALLOWLIST-{'list_functions'}
    with pytest.raises(ValueError, match='READ_ONLY'):
        asyncio.run(bridge.call('execute_plan', {}))
    result = asyncio.run(bridge.call('get_world_state', {}))
    assert result['transport_outcome'] == 'unknown' and result['response'] is None
    assert bridge.clients['get_world_state'].future.cancelled
    bridge.clients['get_world_state'].available = False
    assert asyncio.run(bridge.call('get_world_state', {}))['transport_outcome'] == 'unavailable'


def test_failed_or_unknown_backend_response_retains_structure():
    """Test failed or unknown backend response retains structure."""
    class FailedReads:
        async def call(self, name, arguments):
            return {'transport_outcome': 'response', 'response': {
                'success': True, 'status': 'STATE_UNKNOWN_REQUIRES_RECONCILIATION',
                'plan_revision': 42, 'phase': 'STATE_UNKNOWN',
                'observation_header': {'stamp': {'sec': 1, 'nanosec': 2}, 'frame_id': 'base_link'},
                'expires_in_seconds': 0., 'completed_stages': [], 'waiting_stages': [],
                'physical_execution_enabled': False}, 'detail': 'SIMULATED'}

    with TestClient(create_app(FailedReads(), TOKEN, DOCS),
                    base_url='http://127.0.0.1:8765') as client:
        result = rpc(client, 'tools/call', {'name': 'get_execution_state',
                                            'arguments': {'object_name': 'bowl'}}).json()['result']
        assert result['isError']
        assert result['structuredContent']['response']['plan_revision'] == 42


def test_missing_inputs_and_output_validation_fail_closed():
    """Test missing inputs and output validation fail closed."""
    fake = FakeReads()
    with TestClient(create_app(fake, TOKEN, DOCS),
                    base_url='http://127.0.0.1:8765') as client:
        result = rpc(client, 'tools/call', {'name': 'get_execution_state',
                                            'arguments': {}}).json()['result']
        assert result['isError'] and not fake.calls
        # FakeReads deliberately returns the wrong ROS response schema for arm state.
        result = rpc(client, 'tools/call', {'name': 'get_arm_state',
                                            'arguments': {}}).json()['result']
        assert result['isError']


def test_private_network_requires_tls_no_public_or_wildcard_bind():
    """Test private network requires tls no public or wildcard bind."""
    from task_pipeline.mcp.server import validate_bind
    validate_bind('127.0.0.1', None, None)
    validate_bind('10.211.55.3', 'certificate', 'key')
    for host in ('0.0.0.0', '8.8.8.8', '10.211.55.3'):
        with pytest.raises(ValueError):
            validate_bind(host, None, None)


def test_sdk_client_initialize_discovery_and_reconnection():
    """Test sdk client initialize discovery and reconnection."""
    import asyncio
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def run():
        """Run."""
        fake = FakeReads()
        app = create_app(fake, TOKEN, DOCS)
        async with app.router.lifespan_context(app):
            for _ in range(2):  # new MCP connection never enables a mutation tool
                async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        headers={'Authorization': 'Bearer '+TOKEN}) as http:
                    async with streamable_http_client(
                            'http://127.0.0.1:8765/mcp', http_client=http) as (read, write, _):
                        async with ClientSession(read, write) as session:
                            initialized = await session.initialize()
                            assert initialized.serverInfo.name == 'ros-pipeline-read-only'
                            listed = await session.list_tools()
                            assert {t.name for t in listed.tools} == DISCOVERABLE
                            result = await session.call_tool('get_world_state', {})
                            assert result.structuredContent['response']['state_json']
                            denied = await session.call_tool('execute_plan', {})
                            assert denied.isError
                            assert len((await session.list_resources()).resources) == 6
        assert len(fake.calls) == 2

    asyncio.run(run())


def test_cancelling_read_wait_does_not_cancel_ros_service_or_replay():
    """Test cancelling read wait does not cancel ros service or replay."""
    import asyncio
    import threading
    from task_pipeline.mcp.bridge import ReadBridge

    class Pending:
        cancelled = False

        def add_done_callback(self, callback):
            pass

        def cancel(self):
            self.cancelled = True

    class Client:
        calls = 0
        started = threading.Event()
        future = Pending()

        def wait_for_service(self, timeout_sec):
            return True

        def call_async(self, request):
            self.calls += 1
            self.started.set()
            return self.future

    class Node:
        def create_client(self, kind, endpoint):
            return Client()

    async def run():
        """Run."""
        bridge = ReadBridge(Node(), timeout=.05)
        task = asyncio.create_task(bridge.call('get_world_state', {}))
        client = bridge.clients['get_world_state']
        await asyncio.to_thread(client.started.wait)
        task.cancel()  # Simulates disconnected/cancelled MCP caller
        with pytest.raises(asyncio.CancelledError):
            await task
        # The in-flight ROS service has no cancellation interface. This is only
        # a locally abandoned read; no robot-stop or final-response claim.
        assert client.calls == 1

    asyncio.run(run())


def test_schema_bounds_and_no_silent_field_loss():
    """Test schema bounds and no silent field loss."""
    from task_pipeline.mcp.schema import field_schema
    fixed = field_schema('double[6]')
    assert fixed['minItems'] == fixed['maxItems'] == 6
    bounded = field_schema('sequence<uint8, 4>')
    assert bounded['maxItems'] == 4 and bounded['items']['maximum'] == 255
    assert field_schema('uint64')['maximum'] == 2**64-1
    with pytest.raises(ValueError, match='UNSUPPORTED'):
        field_schema('invented_type')


@pytest.mark.parametrize('name', sorted(MANIPULATION))
def test_visible_manipulation_tool_returns_typed_rejection_without_dispatch(name):
    """Real ROS-shaped requests cannot reach a mutation backend through discovery."""
    from rosidl_runtime_py.convert import message_to_ordereddict
    from task_pipeline.mcp.schema import message_schema
    from task_pipeline.registry import FUNCTIONS

    fake = FakeReads()
    arguments = dict(message_to_ordereddict(FUNCTIONS[name].service_type.Request()))
    if name == 'execute_plan':
        arguments['segment'] = 'pregrasp'
    tool = next(item for item in definitions() if item.name == name)
    assert tool.meta['rosPipeline']['availability'] == 'disabled'
    assert tool.meta['rosPipeline']['responseSchema'] == message_schema(
        FUNCTIONS[name].service_type.Response)
    with TestClient(create_app(fake, TOKEN, DOCS),
                    base_url='http://127.0.0.1:8765') as client:
        for _ in range(2):
            result = rpc(client, 'tools/call', {'name': name, 'arguments': arguments}).json()
            result = result['result']
            assert result['isError']
            assert result['structuredContent']['transport_outcome'] == 'disabled'
            assert result['structuredContent']['dispatched'] is False
            assert result['structuredContent']['response'] is None
            jsonschema.validate(result['structuredContent'], tool.outputSchema)
    assert fake.calls == []


def test_enabled_agent_pipeline_discovery_and_typed_routing():
    """The remote caller sees named preparation tools without arbitrary dispatch."""
    class FakePreparation:
        async def call(self, name, arguments):
            return dict(success=True, status='SIMULATED_TYPED_STEP', function_name=name)
    class NeverExecute:
        async def call(self, *args):
            pytest.fail('No physical operation permitted in fixture')
    from task_pipeline.mcp.preparation import PREPARATION
    with TestClient(create_app(FakeReads(), TOKEN, DOCS, execution=NeverExecute(),
                               preparation=FakePreparation()),
                    base_url='http://127.0.0.1:8765') as client:
        listed = rpc(client, 'tools/list').json()['result']['tools']
        assert len(listed) == 63
        names = {tool['name'] for tool in listed}
        assert set(PREPARATION) <= names
        response = rpc(client, 'tools/call', dict(name='capture_observation', arguments={
            'context_id': '', 'step_id': 'observe', 'request_id': 'observe',
            'arguments': {}, 'bindings': []})).json()['result']
        assert response['structuredContent']['function_name'] == 'capture_observation'
        assert not response['isError']
        denied = rpc(client, 'tools/call', dict(name='run_step', arguments={
            'context_id': '', 'step_id': 'illegal', 'request_id': 'illegal',
            'function_name': 'gripper_close', 'arguments': {}, 'bindings': []})).json()
        assert denied['result']['isError']
        resources = rpc(client, 'resources/list').json()['result']['resources']
        assert any(item['uri'] == 'ros-pipeline://agent-pipeline' for item in resources)


def test_perception_only_blocks_motion_and_generic_queue_dispatch():
    """Live inference mode cannot route an actuator or queued shortcut."""
    class FakePreparation:
        async def call(self, name, arguments):
            return dict(success=True, status='SIMULATED_CAPTURE')

    class NeverExecute:
        async def call(self, *args):
            pytest.fail('No actuator backend may be invoked')

    with TestClient(create_app(FakeReads(), TOKEN, DOCS, execution=NeverExecute(),
                               preparation=FakePreparation(), perception_only=True),
                    base_url='http://127.0.0.1:8765') as client:
        catalog = rpc(client, 'tools/call', {'name': 'list_functions',
                                            'arguments': {}}).json()['result']['structuredContent']
        assert catalog['mode'] == 'perception_only'
        assert len(catalog['allowlist']) == 11
        assert {'capture_observation', 'verify_presence', 'ground_object',
                'segment_object'} <= set(catalog['allowlist'])
        for name in ('move_to', 'gripper_close', 'run_step', 'submit_steps', 'run_steps'):
            result = rpc(client, 'tools/call', {'name': name, 'arguments': {}}).json()['result']
            assert result['isError']
        result = rpc(client, 'tools/call', {'name': 'capture_observation', 'arguments': {
            'context_id': '', 'step_id': 'capture', 'request_id': 'capture',
            'arguments': {}, 'bindings': []}}).json()['result']
        assert result['structuredContent']['success']


def test_memory_descriptions_do_not_claim_visual_absence():
    from task_pipeline.mcp.preparation import PREPARATION
    assert 'not fresh visual evidence' in READS['get_spatial_context']
    assert 'UNKNOWN' in READS['get_spatial_context']
    assert 'not visually absent' in READS['get_manipulation_memory']
    assert 'UNKNOWN, not absent' in PREPARATION['verify_presence']


def test_live_input_guard_rejects_duplicate_or_replay_sources():
    """Publisher provenance is enforced before a calculation can be dispatched."""
    class GuardedReads(FakeReads):
        def input_provenance(self):
            return {'classification': 'replay_or_mixed', 'ros_domain_id': 0}

    class NeverPrepare:
        async def call(self, *args):
            pytest.fail('Mixed-source perception must not dispatch')

    with TestClient(create_app(GuardedReads(), TOKEN, DOCS, execution=object(),
                               preparation=NeverPrepare(), perception_only=True,
                               require_live_inputs=True),
                    base_url='http://127.0.0.1:8765') as client:
        result = rpc(client, 'tools/call', {'name': 'capture_observation', 'arguments': {
            'context_id': '', 'step_id': 'capture', 'request_id': 'capture',
            'arguments': {}, 'bindings': []}}).json()['result']
        assert result['structuredContent']['status'] == 'LIVE_INPUT_PROVENANCE_REQUIRED'
        assert result['isError']


def test_publisher_provenance_requires_one_live_owner_per_topic(monkeypatch):
    from types import SimpleNamespace
    from task_pipeline.mcp.bridge import ReadBridge
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    entries = {}
    for topic in ('/camera/color/image_raw', '/camera/depth_registered/points'):
        entries[topic] = [SimpleNamespace(node_name='camera', node_namespace='/camera')]
    for topic in ('/feedback/joint_states', '/feedback/tcp_pose'):
        entries[topic] = [SimpleNamespace(node_name='agx_arm_ctrl_single_node', node_namespace='/')]
    bridge = ReadBridge.__new__(ReadBridge)
    bridge.node = SimpleNamespace(get_publishers_info_by_topic=lambda topic: entries[topic])
    assert bridge.input_provenance()['classification'] == 'live_driver_publishers'
    entries['/camera/color/image_raw'].append(
        SimpleNamespace(node_name='mock_hardware', node_namespace='/'))
    assert bridge.input_provenance()['classification'] == 'replay_or_mixed'
    entries['/camera/color/image_raw'].pop()
    entries['/feedback/joint_states'] *= 2
    assert bridge.input_provenance()['classification'] == 'unverified'
    entries['/feedback/joint_states'] = []
    assert bridge.input_provenance()['classification'] == 'unverified'


def test_preparation_mode_lists_calculations_but_independently_denies_actuation():
    """The mode is a dispatch boundary, not just hidden action cards."""
    from task_pipeline.mcp.preparation import PREPARATION
    from task_pipeline.registry import catalog

    class LiveReads(FakeReads):
        def input_provenance(self):
            return {'classification': 'live_driver_publishers', 'ros_domain_id': 0}

    class NeverActuate:
        async def call(self, *args):
            pytest.fail('Physical admission must not be reached')

    class FakePreparation:
        async def call(self, name, args):
            return {'success': True, 'status': 'SIMULATED_CALCULATION'}

    with TestClient(create_app(LiveReads(), TOKEN, DOCS, execution=NeverActuate(),
                               preparation=FakePreparation(), preparation_only=True,
                               require_live_inputs=True),
                    base_url='http://127.0.0.1:8765') as client:
        listing = rpc(client, 'tools/call', {'name': 'list_functions',
                                            'arguments': {}}).json()['result']['structuredContent']
        assert listing['mode'] == 'preparation_only'
        assert len(listing['allowlist']) == 53
        assert set(PREPARATION) <= set(listing['allowlist'])
        assert {'get_observe_view', 'run_step', 'submit_steps', 'run_steps'} <= set(
            listing['allowlist'])
        forbidden = {item['name'] for item in catalog() if item.get('effect') == 'actuation'}
        forbidden |= {'execute_pipeline', 'reset_execution_session', 'set_gripper'}
        for name in forbidden:
            response = rpc(client, 'tools/call', {'name': name,
                                                 'arguments': {}}).json()['result']
            assert response['isError'], name
        response = rpc(client, 'tools/call', {'name': 'prepare_pick_place', 'arguments': {
            'context_id': '', 'step_id': 'prepare', 'request_id': 'prepare',
            'arguments': {}, 'bindings': []}}).json()['result']
        assert response['structuredContent']['status'] == 'SIMULATED_CALCULATION'


def test_preparation_mode_cannot_disable_live_guard():
    with pytest.raises(ValueError, match='REQUIRES_LIVE_INPUT_GUARD'):
        create_app(FakeReads(), TOKEN, DOCS, preparation_only=True)
