"""Synthetic master only: these tests cannot construct a ROS client or move hardware."""

import asyncio
import copy
import json

import pytest

pytest.importorskip('mcp', reason='Optional MCP transport dependencies not installed')

from task_pipeline.mcp.execution import PreparedExecution, reference  # noqa: E402


STAGES = ['look', 'pregrasp', 'gripper_open', 'approach', 'gripper_close', 'lift',
          'transfer', 'descent', 'release', 'withdrawal', 'retreat']
FUNCTION = {'transfer': 'execute_placement_transfer', 'descent': 'execute_placement_descent',
            'release': 'release_placed_object', 'withdrawal': 'execute_withdrawal',
            'retreat': 'execute_placement_retreat', 'commit_slots': 'commit_floor_slots',
            'gripper_open': 'gripper_open', 'gripper_close': 'gripper_close'}


class FakeMaster:
    """Model public response shapes; never import rclpy or actual controllers."""

    def __init__(self):
        self.state = dict(success=True, status='EXECUTION_SESSION_READY', plan_revision=42,
                          observation_header={'frame_id': 'base_link',
                                              'stamp': {'sec': 123, 'nanosec': 0}},
                          phase='PREPARED', expires_in_seconds=60.,
                          completed_stages=[], waiting_stages=list(STAGES),
                          physical_execution_enabled=True)
        self.calls = []
        self.fail = None
        self.selected = {'header': self.state['observation_header'], 'pose': {
            'position': {'x': .3, 'y': .1, 'z': .02},
            'orientation': {'x': 0., 'y': 0., 'z': 0., 'w': 1.}}}

    async def request(self, name, arguments, timeout):
        if name == 'get_execution_state':
            return copy.deepcopy(self.state)
        if name == 'validate_prepared_pick_place':
            return dict(success=True, status='VALIDATED', plan_revision=42,
                        selected_pose=copy.deepcopy(self.selected))
        self.calls.append(name)
        if self.fail == 'timeout':
            raise TimeoutError('SIMULATED_TIMEOUT')
        if self.fail == 'disconnect':
            raise asyncio.CancelledError()
        if self.fail == 'contact':
            return dict(success=True, status='SIMULATED_NO_CONTACT', contact_confirmed=False)
        stage = PreparedExecution.stage(name, arguments)
        if stage != 'commit_slots':
            assert self.state['waiting_stages'].pop(0) == stage
            self.state['completed_stages'].append(stage)
        phases = dict(look='PREPARED', pregrasp='AT_PREGRASP', gripper_open='OPENED',
                      approach='AT_GRASP', gripper_close='CONTACT_CONFIRMED', lift='LIFTED',
                      transfer='AT_PREPLACE', descent='PLACED', release='RELEASED',
                      commit_slots='RELEASED', withdrawal='RELEASED', retreat='COMPLETE')
        self.state['phase'] = phases[stage]
        from task_pipeline.registry import FUNCTIONS
        values = dict(success=True, motion_completed=True, contact_confirmed=True,
                      gripper_released=stage in ('release', 'withdrawal', 'retreat'),
                      phase=self.state['phase'], status='SIMULATED_COMPLETED', occupied_slots=1)
        return {key: values[key] for key in
                FUNCTIONS[name].service_type.Response.get_fields_and_field_types()}


def arguments(master, stage, identifier=None):
    ref = dict(object_name='bowl', plan_revision=42, attempt=1,
               task_id=reference('bowl', master.state))
    result = dict(execution_ref=ref, request_id=identifier or stage)
    if stage == 'commit_slots':
        result.update(selected_pose=copy.deepcopy(master.selected), reset=False, validate_only=False)
    else:
        result.update(object_name='tray' if stage in ('withdrawal', 'retreat') else 'bowl',
                      plan_revision=42)
        if stage not in FUNCTION:
            result['segment'] = stage
        if stage.startswith('gripper'):
            result.update(width_m=0., force_n=0.)
    return result


def function(stage):
    return FUNCTION.get(stage, 'execute_plan')


def test_complete_pipeline_and_duplicate_persistence(tmp_path):
    async def run():
        master = FakeMaster()
        adapter = PreparedExecution(master, tmp_path/'ledger.sqlite')
        for stage in STAGES[:9]+['commit_slots']+STAGES[9:]:
            query = arguments(master, stage)
            result = await adapter.call(function(stage), query)
            assert result['transport_outcome'] == 'response', result
            repeated = await adapter.call(function(stage), query)
            assert not repeated['dispatched']
        assert len(master.calls) == 12
        assert adapter.load('current')['complete']
        assert adapter.load('current')['closes'] == 1
        adapter.db.close()
        adapter.file_lock.close()
        reopened = PreparedExecution(master, tmp_path/'ledger.sqlite')
        repeated = await reopened.call('gripper_close', arguments(master, 'gripper_close'))
        assert not repeated['dispatched'] and len(master.calls) == 12
    asyncio.run(run())


@pytest.mark.parametrize('mutation', ['revision', 'object', 'attempt', 'source', 'gate', 'expiry'])
def test_identity_and_master_readiness_before_dispatch(tmp_path, mutation):
    async def run():
        master = FakeMaster()
        query = arguments(master, 'look')
        if mutation == 'revision':
            query['plan_revision'] = 41
        elif mutation == 'object':
            query['object_name'] = 'cup'
        elif mutation == 'attempt':
            query['execution_ref']['attempt'] = 2
        elif mutation == 'source':
            master.state['observation_header']['stamp']['sec'] += 1
        elif mutation == 'gate':
            master.state['physical_execution_enabled'] = False
        else:
            master.state['expires_in_seconds'] = 0.
        adapter = PreparedExecution(master, tmp_path/'ledger.sqlite')
        assert (await adapter.call('execute_plan', query))['transport_outcome'] == 'rejected'
        assert not master.calls
    asyncio.run(run())


def test_placement_before_verification_rejected(tmp_path):
    async def run():
        master = FakeMaster()
        adapter = PreparedExecution(master, tmp_path/'ledger.sqlite')
        result = await adapter.call('execute_placement_transfer', arguments(master, 'transfer'))
        assert result['transport_outcome'] == 'rejected' and not master.calls
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['timeout', 'disconnect', 'contact'])
def test_uncertain_outcome_stops_all_actions_and_survives_restart(tmp_path, failure):
    async def run():
        master = FakeMaster()
        adapter = PreparedExecution(master, tmp_path/'ledger.sqlite')
        for stage in STAGES[:4]:
            assert (await adapter.call(function(stage), arguments(master, stage)))[
                'transport_outcome'] == 'response'
        master.fail = failure
        if failure == 'disconnect':
            with pytest.raises(asyncio.CancelledError):
                await adapter.call('gripper_close', arguments(master, 'gripper_close'))
        else:
            assert (await adapter.call('gripper_close', arguments(master, 'gripper_close')))[
                'transport_outcome'] == 'unknown'
        count = len(master.calls)
        adapter.db.close()
        adapter.file_lock.close()
        restored = PreparedExecution(master, tmp_path/'ledger.sqlite')
        assert restored.load('current')['unknown']
        result = await restored.call('execute_plan', arguments(master, 'lift'))
        assert result['transport_outcome'] == 'rejected'
        assert len(master.calls) == count
    asyncio.run(run())


def test_request_id_change_and_slot_reset_never_dispatch(tmp_path):
    async def run():
        master = FakeMaster()
        adapter = PreparedExecution(master, tmp_path/'ledger.sqlite')
        first = arguments(master, 'look')
        await adapter.call('execute_plan', first)
        changed = copy.deepcopy(first)
        changed['segment'] = 'pregrasp'
        assert (await adapter.call('execute_plan', changed))['transport_outcome'] == 'rejected'
        assert len(master.calls) == 1
        reset = arguments(master, 'commit_slots')
        reset['reset'] = True
        assert (await adapter.call('commit_floor_slots', reset))['transport_outcome'] == 'rejected'
        assert len(master.calls) == 1
    asyncio.run(run())


def test_enabled_mcp_protocol_uses_ref_and_preserves_raw_response(tmp_path):
    from pathlib import Path
    from starlette.testclient import TestClient
    from task_pipeline.mcp.server import create_app

    master = FakeMaster()
    adapter = PreparedExecution(master, tmp_path/'ledger.sqlite')

    class Reads:
        async def call(self, name, query):
            return {'transport_outcome': 'response', 'response': copy.deepcopy(master.state),
                    'detail': 'SIMULATED'}

    token = 'simulated-execution-test-token-1234567890'
    headers = {'Authorization': 'Bearer '+token,
               'Accept': 'application/json, text/event-stream'}
    app = create_app(Reads(), token, Path(__file__).resolve().parents[3]/'docs/mcp',
                     execution=adapter)
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        def call(name, query):
            return client.post('/mcp', headers=headers, json={
                'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                'params': {'name': name, 'arguments': query}}).json()['result']
        state = call('get_execution_state', {'object_name': 'bowl'})
        assert not state['isError']
        assert state['structuredContent']['execution_ref'] == arguments(master, 'look')[
            'execution_ref']
        discovered = call('list_functions', {})['structuredContent']
        assert discovered['mode'] == 'prepared_execution' and not discovered['disabled_tools']
        for stage in STAGES[:9]+['commit_slots']+STAGES[9:]:
            result = call(function(stage), arguments(master, stage))
            assert not result['isError'], result
            assert result['structuredContent']['response']['success']
            if stage == 'release':
                read = call('get_execution_state', {'object_name': 'bowl'})['structuredContent']
                assert read['selected_pose'] == master.selected
        assert len(master.calls) == 12


def test_persisted_budget_and_crash_marker(tmp_path):
    async def run():
        master = FakeMaster()
        adapter = PreparedExecution(master, tmp_path/'ledger.sqlite')
        for stage in STAGES[:4]:
            await adapter.call(function(stage), arguments(master, stage))
        current = adapter.load('current')
        current['closes'] = 3  # Simulate an existing durable exhausted budget.
        adapter.save('current', current)
        count = len(master.calls)
        result = await adapter.call('gripper_close', arguments(master, 'gripper_close'))
        assert result['transport_outcome'] == 'rejected'
        assert len(master.calls) == count
        current['running'] = True  # Simulate process death before a final response.
        adapter.save('current', current)
        adapter.db.close()
        adapter.file_lock.close()
        reopened = PreparedExecution(master, tmp_path/'ledger.sqlite')
        assert reopened.load('current')['unknown']
    asyncio.run(run())


class FakeJointMaster:
    """Exercise joint-target admission without ROS or actuator access."""

    def __init__(self):
        self.robot = dict(physical_grasp_execution_enabled=True, state='IDLE', holding=None,
                          pipeline_executing=False)
        self.positions = [0.]*6
        self.calls = []
        self.fail = False

    def now(self):
        return 100.1

    async def request(self, name, query, timeout):
        if name == 'get_world_state':
            return {'state_json': json.dumps({'robot': self.robot})}
        if name == 'get_arm_state':
            return dict(success=True, status='ARM_STATE_READY', joints={
                'header': {'stamp': {'sec': 100, 'nanosec': 0}}, 'position': self.positions})
        assert name == 'move_to'
        self.calls.append(query)
        if self.fail:
            raise TimeoutError('SIMULATED_MOVE_TIMEOUT')
        self.positions = list(query['target_joints'])
        return dict(success=True, status='TARGET_REACHED')


def test_move_to_uses_joint_target_once_and_retains_uncertainty(tmp_path):
    async def run():
        master = FakeJointMaster()
        gateway = PreparedExecution(master, tmp_path/'moves.sqlite')
        request = dict(target_joints=[.1]*6, request_id='move-1')
        result = await gateway.call('move_to', request)
        assert result['transport_outcome'] == 'response'
        assert len(master.calls) == 1
        assert not (await gateway.call('move_to', request))['dispatched']
        master.fail = True
        request = dict(target_joints=[.2]*6, request_id='move-2')
        assert (await gateway.call('move_to', request))['transport_outcome'] == 'unknown'
        assert not (await gateway.call('move_to', request))['dispatched']
        assert (await gateway.call('move_to', dict(
            target_joints=[.3]*6, request_id='move-3')))['transport_outcome'] == 'rejected'
        assert len(master.calls) == 2
    asyncio.run(run())


@pytest.mark.parametrize('field,value', [('physical_grasp_execution_enabled', False),
                                       ('state', 'ERROR'), ('holding', 'cup'),
                                       ('pipeline_executing', True)])
def test_move_to_does_not_dispatch_when_master_is_not_ready(tmp_path, field, value):
    async def run():
        master = FakeJointMaster()
        master.robot[field] = value
        gateway = PreparedExecution(master, tmp_path/'moves.sqlite')
        result = await gateway.call('move_to', dict(target_joints=[.1]*6, request_id='move-1'))
        assert result['transport_outcome'] == 'rejected' and not master.calls
    asyncio.run(run())
