"""Simulated controls with fake transport only; no ROS clients or actuator publishers."""

import asyncio
import copy
import json

import jsonschema
import pytest
pytest.importorskip('mcp')
from starlette.testclient import TestClient  # noqa: E402
from task_pipeline.mcp.controls import ControlInterface, NAMES, definitions  # noqa: E402
from task_pipeline.mcp.server import create_app  # noqa: E402
from test_mcp_execution import FakeMaster, STAGES  # noqa: E402
from test_mcp_tasks import make_facade, ReplayReads, DOCS, TOKEN  # noqa: E402


class Reads(ReplayReads):
    def input_provenance(self):
        return {'classification': 'live_driver_publishers', 'ros_domain_id': 0}

    async def workspace(self, include_image=False):
        return {'transport_outcome': 'response', 'response': {'status': 'CURRENT_OBSERVATION_READY'}}


class Master(FakeMaster):
    def __init__(self):
        super().__init__()
        self.joints = [0.] * 6
        self.no_change, self.gate = False, True
        self.contact_fail, self.open_fail = False, False

    def now(self):
        return 100.

    async def request(self, name, arguments, timeout):
        if name == 'get_world_state':
            return {'state_json': json.dumps({'robot': {'state': 'IDLE', 'holding': None,
                    'pipeline_executing': False, 'physical_grasp_execution_enabled': self.gate}})}
        if name == 'get_arm_state':
            return {'success': True, 'joints': {'header': {'stamp': {'sec': 100, 'nanosec': 0}},
                    'position': list(self.joints)}}
        if name == 'get_observe_view':
            return {'success': True, 'status': 'NO_REACHABLE_CHANGE' if self.no_change else 'PARTIAL',
                    'moved_target': not self.no_change, 'endpoint_safe': not self.no_change,
                    'achieved_fraction': 0. if self.no_change else .5,
                    'source_joints': list(self.joints), 'target_joints': [.02] * 6}
        if name == 'move_to':
            self.calls.append(name)
            if self.fail == 'timeout':
                raise TimeoutError('SIMULATED_TIMEOUT')
            self.joints = arguments['target_joints']
            return {'success': True, 'status': 'TARGET_REACHED'}
        if name == 'open_empty_hand':
            self.calls.append(name)
            return {'success': not self.open_fail, 'feedback_confirmed': not self.open_fail,
                    'status': 'OPENING_CONFIRMED' if not self.open_fail else 'OPENING_UNCONFIRMED'}
        if name == 'validate_prepared_grasp_lift':
            return {'success': True, 'plan_revision': 42, 'status': 'VALIDATED'}
        if name == 'gripper_close' and self.contact_fail:
            self.calls.append(name)
            return {'success': False, 'contact_confirmed': False, 'status': 'CONTACT_UNCONFIRMED'}
        return await super().request(name, arguments, timeout)


def fixture(tmp_path, kind='prepare_pick_place'):
    facade, calc, gateway, ledger = make_facade(tmp_path)
    facade.bridge = Reads()
    master = Master()
    if kind == 'prepare_grasp_lift':
        master.state['waiting_stages'] = list(STAGES[1:6])
    ledger.transport = master
    parent = {'task_id': 'reviewed_task', 'state': 'COMPLETED', 'kind': 'prepare_manipulation',
              'arguments': {'object_name': 'bowl', 'destination': 'tray'}}
    facade.save(parent)
    ledger.save('plan:reviewed_plan', {'plan_id': 'reviewed_plan', 'revision': 42,
                'kind': kind, 'object_name': 'bowl', 'task_id': 'reviewed_task',
                'source_mode': 'live', 'ros_domain_id': 0,
                'observation_header': copy.deepcopy(master.state['observation_header'])})
    return ControlInterface(ledger, facade, facade.bridge, enabled=True), master, facade, calc


def execute(**kwargs):
    return dict(request_id='execute', plan_id='reviewed_plan', revision=42, **kwargs)


def camera(**kwargs):
    return dict(request_id='camera', mode='relative', frame='camera_optical',
                translation_m=[.02, 0, 0], rotation_rad=[0, 0, 0], **kwargs)


def test_complete_pick_place_uses_existing_contact_and_commit_order_and_receipts(tmp_path):
    async def run():
        controls, master, tasks, _ = fixture(tmp_path)
        result = await controls.call('execute_manipulation', execute())
        assert result['state'] == 'COMPLETED', result
        assert result['result']['phase'] == 'COMPLETE'
        assert master.calls.index('gripper_close') < master.calls.index('execute_placement_transfer')
        assert master.calls.index('release_placed_object') < master.calls.index('commit_floor_slots')
        assert master.calls.index('commit_floor_slots') < master.calls.index('execute_withdrawal')
        assert controls.execution.load('current')['complete']
        repeated = await controls.call('execute_manipulation', execute())
        assert repeated['receipt_replayed'] and not repeated['dispatched']
        assert len(master.calls) == 12
        state = await tasks.call('get_task_status', {})
        assert state['current_step'] is None and state['last_successful_step'] == 'execute_placement_retreat'
        assert state['task_id'] == result['task_id']
    asyncio.run(run())


def test_grasp_lift_stops_holding_and_retains_arm_ownership(tmp_path):
    async def run():
        controls, master, _, _ = fixture(tmp_path, 'prepare_grasp_lift')
        result = await controls.call('execute_manipulation', execute())
        assert result['success'] and result['result']['holding'], result
        assert controls.execution.load('current')['verified']
        assert not controls.execution.load('current')['complete']
        assert 'execute_placement_transfer' not in master.calls
        blocked = await controls.call('move_camera', camera())
        assert blocked['state'] == 'BLOCKED'
        assert 'move_to' not in master.calls
        count = len(master.calls)
        release = await controls.call('control_gripper', dict(
            execute(), request_id='cannot_release_held', mode='prepared', action='release'))
        assert release['state'] == 'BLOCKED' and len(master.calls) == count
    asyncio.run(run())


def test_stale_reviewed_plan_and_unconfirmed_contact_never_advance(tmp_path):
    async def run():
        controls, master, _, _ = fixture(tmp_path)
        master.state['plan_revision'] = 43
        result = await controls.call('execute_manipulation', execute())
        assert result['state'] == 'BLOCKED' and not master.calls
        master.state['plan_revision'] = 42
        master.contact_fail = True
        result = await controls.call('execute_manipulation', dict(execute(), request_id='contact'))
        assert result['state'] == 'UNKNOWN', result
        assert result['blocker']['status'] == 'CONTACT_UNCONFIRMED'
        assert 'execute_placement_transfer' not in master.calls
        count = len(master.calls)
        with pytest.raises(ValueError, match='RECONCILIATION'):
            await controls.call('execute_manipulation', dict(execute(), request_id='cannot_retry'))
        assert len(master.calls) == count
    asyncio.run(run())


def test_next_stage_then_prepared_gripper_is_same_owned_session(tmp_path):
    async def run():
        controls, master, _, _ = fixture(tmp_path)
        for request_id in ('look', 'pregrasp'):
            result = await controls.call('execute_manipulation', dict(
                execute(stage='next'), request_id=request_id))
            assert result['success'], result
        result = await controls.call('control_gripper', dict(execute(), request_id='open',
            action='open', mode='prepared'))
        assert result['success']
        assert master.calls == ['execute_plan', 'execute_plan', 'gripper_open']
    asyncio.run(run())


def test_camera_distinguishes_no_change_and_checked_motion(tmp_path):
    async def run():
        controls, master, _, _ = fixture(tmp_path)
        master.no_change = True
        result = await controls.call('move_camera', camera())
        assert result['status'] == 'NO_REACHABLE_CHANGE' and not result['dispatched']
        assert result['result']['planner_result']['achieved_fraction'] == 0.
        assert not master.calls
        master.no_change = False
        result = await controls.call('move_camera', dict(camera(), request_id='changed'))
        assert result['success'] and result['result']['movement_completed'], result
        assert result['result']['planner_result']['status'] == 'PARTIAL'
        assert master.calls == ['move_to']
    asyncio.run(run())


def test_empty_hand_open_is_fixed_api_with_confirmed_feedback(tmp_path):
    async def run():
        controls, master, _, _ = fixture(tmp_path)
        result = await controls.call('control_gripper', {'request_id': 'open', 'mode': 'empty_hand',
                                    'action': 'open', 'width_m': .07, 'force_n': 1.})
        assert result['success'], result
        assert not result['result']['target_aperture_verified']
        assert master.calls == ['open_empty_hand']
    asyncio.run(run())


@pytest.mark.parametrize('name,args', [
    ('move_camera', {'request_id': 'bad', 'mode': 'relative'}),
    ('move_camera', dict(camera(), target_name='bowl')),
    ('control_gripper', {'request_id': 'bad', 'mode': 'empty_hand', 'action': 'close'}),
    ('control_gripper', dict(execute(), mode='prepared', action='close', force_n=1.)),
])
def test_malformed_controls_rejected_before_task_creation(tmp_path, name, args):
    async def run():
        controls, master, _, _ = fixture(tmp_path)
        count = controls.execution.db.execute('select count(*) from execution').fetchone()[0]
        with pytest.raises((ValueError, jsonschema.ValidationError)):
            await controls.call(name, args)
        assert not master.calls
        assert count == controls.execution.db.execute('select count(*) from execution').fetchone()[0]
    asyncio.run(run())


def test_timeout_reconnection_never_replays_move(tmp_path):
    async def run():
        controls, master, _, _ = fixture(tmp_path)
        master.fail = 'timeout'
        result = await controls.call('move_camera', camera())
        assert result['state'] == 'UNKNOWN'
        recreated = ControlInterface(controls.execution, controls.tasks, controls.bridge, True)
        again = await recreated.call('move_camera', camera())
        assert again['state'] == 'UNKNOWN' and not again['dispatched']
        assert master.calls == ['move_to']
    asyncio.run(run())


def test_native_discovery_and_preparation_gate_are_independent_of_master_gate(tmp_path):
    facade, calc, gateway, ledger = make_facade(tmp_path)
    app = create_app(ReplayReads(), TOKEN, DOCS, execution=ledger, preparation=gateway,
                     preparation_only=True, require_mock_inputs=True, task_tools=True,
                     workspace_tools=True, control_tools=True)
    def rpc(client, method, params):
        return client.post('/mcp', headers={'Authorization': 'Bearer ' + TOKEN,
            'Accept': 'application/json, text/event-stream', 'MCP-Protocol-Version': '2025-06-18'},
            json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}).json()['result']
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        tools = {t['name']: t for t in rpc(client, 'tools/list', {})['tools']}
        assert len(tools) == 68 and set(NAMES) <= tools.keys()
        for tool in definitions():
            jsonschema.Draft202012Validator.check_schema(tool.inputSchema)
        for name, args in [('execute_manipulation', execute()), ('move_camera', camera()),
                           ('control_gripper', dict(execute(), mode='prepared', action='close'))]:
            result = rpc(client, 'tools/call', {'name': name, 'arguments': args})
            data = result['structuredContent']
            jsonschema.validate(data, tools[name]['outputSchema'])
            assert result['isError'] and data['status'] == 'PHYSICAL_CONTROLS_DISABLED'
        listing = rpc(client, 'tools/call', {'name': 'list_functions', 'arguments': {}})['structuredContent']
        assert set(NAMES) <= set(listing['disabled_tools'])
        assert not set(NAMES) & set(listing['allowlist'])
        assert not calc.calls


def test_native_guarded_control_mode_completes_only_in_fake_transport(tmp_path):
    controls, master, facade, calc = fixture(tmp_path)
    app = create_app(Reads(), TOKEN, DOCS, execution=controls.execution,
                     preparation=facade.gateway, require_live_inputs=True,
                     task_tools=True, workspace_tools=True, control_tools=True)
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        def rpc(method, params):
            return client.post('/mcp', headers={'Authorization': 'Bearer ' + TOKEN,
                'Accept': 'application/json, text/event-stream',
                'MCP-Protocol-Version': '2025-06-18'}, json={'jsonrpc': '2.0', 'id': 1,
                'method': method, 'params': params}).json()['result']
        tools = {t['name']: t for t in rpc('tools/list', {})['tools']}
        assert all(tools[n]['_meta']['availability'] == 'guarded_execution' for n in NAMES)
        result = rpc('tools/call', {'name': 'execute_manipulation', 'arguments': execute()})
        assert not result['isError'], result
        data = result['structuredContent']
        jsonschema.validate(data, tools['execute_manipulation']['outputSchema'])
        assert data['state'] == 'COMPLETED'
        assert master.calls[-1] == 'execute_placement_retreat'
        assert not calc.calls  # No actual ROS clients/queue exist in this fixture.


def test_cancelled_physical_service_marks_parent_unknown_and_prevents_replay(tmp_path):
    async def run():
        controls, master, tasks, _ = fixture(tmp_path)
        master.fail = 'disconnect'
        with pytest.raises(asyncio.CancelledError):
            await controls.call('execute_manipulation', execute())
        saved = await tasks.call('get_task_status', {})
        assert saved['state'] == 'UNKNOWN' and saved['current_step'] is None
        assert controls.execution.load('preparation_barrier')
        count = len(master.calls)
        again = await controls.call('execute_manipulation', execute())
        assert again['state'] == 'UNKNOWN' and len(master.calls) == count
    asyncio.run(run())


def test_master_gate_false_never_dispatches_camera_or_gripper(tmp_path):
    async def run():
        controls, master, _, _ = fixture(tmp_path)
        master.gate = False
        for name, args in [('move_camera', camera()), ('control_gripper', {
                'request_id': 'open', 'mode': 'empty_hand', 'action': 'open',
                'width_m': .07, 'force_n': 1.})]:
            result = await controls.call(name, args)
            assert result['state'] == 'BLOCKED' and not result['dispatched']
        assert not master.calls
    asyncio.run(run())


def test_generic_queue_rejects_new_physical_names(tmp_path):
    _, _, facade, calc = fixture(tmp_path)
    for name in (*NAMES, 'open_empty_hand'):
        with pytest.raises(ValueError, match='FUNCTION_NOT_ALLOWED'):
            facade.gateway.make_step(name, {'step_id': 'bad', 'arguments': {}, 'bindings': []},
                                     task_level=True)
    assert not calc.calls


def test_validation_never_claims_execution_authorization_in_control_mode():
    from task_pipeline.mcp.tasks import validation_summary
    result = validation_summary({'state': 'COMPLETED', 'preparation_only': False}, [])
    assert result['execution_allowed'] is False
    assert result['execution_block_reason'] == 'CALCULATION_RESULT_NOT_EXECUTION_AUTHORIZATION'
