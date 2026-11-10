"""Task facade fixtures: real typed StepStore/action adapter, no hardware clients."""

import asyncio
import copy
import json
from pathlib import Path

from geometry_msgs.msg import Point
import jsonschema
from my_actions import srv
from my_actions.msg import GraspCandidate, PlacementCandidate
import pytest
from sensor_msgs.msg import Image, PointCloud2

pytest.importorskip('mcp', reason='Optional MCP transport dependencies not installed')

from starlette.testclient import TestClient  # noqa: E402
from std_msgs.msg import Header  # noqa: E402

from task_pipeline.mcp.execution import PreparedExecution  # noqa: E402
from task_pipeline.mcp.preparation import PreparationGateway  # noqa: E402
from task_pipeline.mcp.server import create_app  # noqa: E402
from task_pipeline.mcp.tasks import definitions, NAMES, TaskInterface  # noqa: E402
from test_mcp_preparation import FixtureTransport  # noqa: E402

DOCS = Path(__file__).resolve().parents[3] / 'docs/mcp'
TOKEN = 'task_fixture_token_012345678901234567890'


class ReplayReads:
    def __init__(self, transport=None):
        self.transport = transport

    def input_provenance(self):
        return {'classification': 'mock_replay_publishers', 'ros_domain_id': 89}

    async def call(self, name, arguments):
        if name == 'get_execution_state' and self.transport:
            return {'transport_outcome': 'response', 'response': {
                'success': True, 'plan_revision': self.transport.revision,
                'observation_header': {'frame_id': 'base_link',
                                       'stamp': {'sec': 123, 'nanosec': 0}}}}
        raise AssertionError('Unexpected read: ' + name)


class TaskTransport(FixtureTransport):
    def __init__(self, fail=None):
        self.fail = fail or {}
        self.cached, self.prepared, self.revision = False, False, 7
        self.candidates = [GraspCandidate(id=1), GraspCandidate(id=2)]
        self.header = Header(frame_id='base_link')
        self.header.stamp.sec = 123
        self.image = Image(header=copy.deepcopy(self.header), width=2, height=2,
                           encoding='mono8', step=2, data=[1, 2, 3, 4])
        self.cloud = PointCloud2(header=self.header, width=2, height=2)
        super().__init__()

    def backend(self, name, request):
        from task_pipeline.registry import FUNCTIONS
        self.calls.append(name)
        result = FUNCTIONS[name].service_type.Response()
        result.success = name not in self.fail
        if hasattr(result, 'status'):
            result.status = self.fail.get(name, 'SIMULATED_OK')
        else:
            result.message = self.fail.get(name, 'SIMULATED_OK')
        if not result.success:
            return result
        if name == 'capture_observation':
            result.status = 'SYNCHRONIZED_OBSERVATION_READY;GRASP_OBSERVATION_PINNED'
            result.image, result.cloud = self.image, self.cloud
            result.camera_transform.header = self.header
        elif name == 'verify_presence':
            result.present, result.status = True, 'PRESENT'
        elif name == 'ground_object':
            result.pixel_box = [0, 0, 1, 1]
        elif name == 'segment_object':
            result.mask = self.image
        elif name in ('resolve_target', 'store_segmented_target'):
            result.image, result.mask, result.observation_cloud = (
                self.image, self.image, self.cloud)
            result.observation_header = self.header
            result.geometry_source_header = self.header
            result.camera_transform.header = self.header
        elif name in ('calculate_active_grasps', 'calculate_grasps'):
            result.candidates = self.candidates
            result.validated_target_mask = self.image
        elif name == 'select_grasp':
            result.candidate = request.candidates[0]
        elif name == 'cache_grasp_plan':
            assert [c.id for c in request.candidates] == [1, 2]
            assert request.observation_header == self.header
            self.cached = True
        elif name == 'get_cached_grasp':
            assert self.cached
            result.candidates, result.observation_header = self.candidates, self.header
        elif name in ('prepare_grasp_lift', 'prepare_pick_place'):
            assert self.cached
            self.prepared = True
            result.plan_revision = self.revision
        elif name in ('get_prepared_grasp_lift', 'validate_prepared_grasp_lift',
                      'get_prepared_pick_place', 'validate_prepared_pick_place'):
            assert self.prepared
            result.plan_revision = self.revision
        elif name == 'prepare_grasp_input':
            result.image_width = result.image_height = 2
        elif name == 'update_object':
            pass
        elif name == 'find_placements':
            result.candidates = [PlacementCandidate(id=1)]
        elif name == 'plan_next_best_view':
            result.path_flat = [0.]*12
            result.observation_header = self.header
        elif name == 'get_observe_view':
            result.status = 'PARTIAL'
            result.moved_target = result.endpoint_safe = True
            result.achieved_fraction = .5
        elif name == 'perceive_scene':
            result.names = list(request.object_names) + [request.destination_name]
            result.roles = ['object'] * len(request.object_names) + ['destination']
            result.observed = [True] * len(result.names)
            result.target_statuses = ['ACTIVE_TARGET_GROUNDED'] * len(result.names)
            result.positions = [Point(x=.2)] * len(result.names)
            result.observation_header = self.header
        elif name not in ('pin_grasp_observation', 'pin_placement_observation'):
            raise AssertionError('Unexpected/physical backend: ' + name)
        return result

    async def typed(self, name, request, timeout):
        if name == 'get_world_state':
            return srv.GetWorldState.Response(state_json=json.dumps({'robot': {
                'state': 'IDLE', 'holding': None, 'pipeline_executing': False,
                'physical_grasp_execution_enabled': False}}))
        return await super().typed(name, request, timeout)


def make_facade(tmp_path, fail=None):
    transport = TaskTransport(fail)
    execution = PreparedExecution(None, tmp_path / 'tasks.sqlite')
    gateway = PreparationGateway(transport, execution, require_physical_disabled=True)
    return TaskInterface(gateway, ReplayReads(transport)), transport, gateway, execution


@pytest.mark.parametrize('kind,args', [
    ('observe_scene', {'targets': ['cup'], 'detail': 'mask'}),
    ('observe_scene', {'targets': ['cup'], 'destination': 'tray', 'detail': 'location'}),
    ('prepare_manipulation', {'operation': 'grasp', 'object_name': 'cup'}),
    ('prepare_manipulation', {'operation': 'grasp_lift', 'object_name': 'cup'}),
    ('prepare_manipulation', {'operation': 'pick_place', 'object_name': 'cup',
                              'destination': 'tray'}),
])
def test_task_route_reuses_queue_and_returns_compact_results(tmp_path, kind, args):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path)
        result = await facade.submit(kind, dict(args, request_id='same_task', wait_seconds=1))
        assert result['state'] == 'COMPLETED', result
        assert result['success'] and result['observation_time']['stamp']['sec'] == 123
        assert transport.calls[0] == 'capture_observation'
        count = len(transport.calls)
        again = await facade.submit(kind, dict(args, request_id='same_task', wait_seconds=0))
        assert again['task_id'] == result['task_id'] and len(transport.calls) == count
        if kind == 'observe_scene':
            assert result['result']['targets'][0]['status'] == 'PRESENT'
            if args['detail'] == 'mask':
                assert result['result']['targets'][0]['mask']['source_field'] == 'mask'
        else:
            assert result['result']['candidate_count'] == 2
            if args['operation'] == 'grasp':
                assert 'plan' not in result
            else:
                assert result['plan']['revision'] == 7
                checked = await facade.validate({'request_id': 'check',
                                                 'plan_id': result['plan']['plan_id'],
                                                 'revision': 7})
                assert checked['success'], checked
                transport.revision = 8
                invalid = await facade.validate({'request_id': 'wrong_revision',
                                                 'plan_id': result['plan']['plan_id'],
                                                 'revision': 7})
                assert invalid['state'] == 'BLOCKED'
                assert 'REVISION_CHANGED' in invalid['status']
        evidence = await facade.status(result['task_id'], diagnostics=True)
        assert evidence['diagnostics']['context_id']
        if kind == 'prepare_manipulation':
            cache = next(s for s in evidence['diagnostics']['steps']
                         if s['function_name'] == 'cache_grasp_plan')
            binding = next(b for b in cache['bindings'] if b['target_field'] == 'candidates')
            assert binding['source_field'] == 'candidates'
            assert any(s['step_id'] == binding['source_step_id'] and
                       s['function_name'] in ('calculate_active_grasps', 'calculate_grasps')
                       for s in evidence['diagnostics']['steps'])
    asyncio.run(run())


def test_failure_stops_dependents_and_uncertain_perception_is_not_absence(tmp_path):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path, {'verify_presence': 'PRESENCE_UNKNOWN'})
        result = await facade.submit('observe_scene', {
            'request_id': 'unknown', 'targets': ['cup'], 'detail': 'mask', 'wait_seconds': 1})
        assert not result['success'] and result['blocker']['status'] == 'PRESENCE_UNKNOWN'
        assert result['result']['targets'][0]['status'] == 'UNKNOWN'
        assert 'ground_object' not in transport.calls
    asyncio.run(run())


def test_cancellation_uses_existing_goal_and_never_replays(tmp_path):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path)
        result = await facade.submit('observe_scene', {'request_id': 'cancel', 'targets': ['cup']})
        assert result['state'] == 'RUNNING'
        pending = await facade.cancel(result['task_id'])
        assert pending['state'] in ('CANCEL_PENDING', 'CANCELLED')
        final = await facade.status(result['task_id'], 1)
        assert final['state'] == 'CANCELLED'
        assert 'capture_observation' not in transport.calls
        again = await facade.submit('observe_scene', {'request_id': 'cancel', 'targets': ['cup']})
        assert again['task_id'] == result['task_id'] and again['state'] == 'CANCELLED'
    asyncio.run(run())


def test_wait_timeout_and_disconnection_do_not_cancel_or_duplicate_job(tmp_path):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path)
        original = transport.finish
        event = asyncio.Event()

        async def delayed(goal):
            await event.wait()
            return await original(goal)

        transport.finish = delayed
        request = {'request_id': 'disconnect', 'targets': ['cup'], 'detail': 'presence',
                   'wait_seconds': 1}
        caller = asyncio.create_task(facade.submit('observe_scene', request))
        await asyncio.sleep(.01)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        record = next(json.loads(row[0]) for row in facade.ledger.db.execute(
            "SELECT data FROM execution WHERE id LIKE 'task:%'"))
        status = await facade.status(record['task_id'], .001)
        assert status['state'] == 'UNKNOWN'
        event.set()
        final = await facade.status(record['task_id'], 1)
        assert final['state'] == 'COMPLETED'
        assert transport.calls.count('capture_observation') == 1
        again = await facade.submit('observe_scene', request)
        assert again['task_id'] == final['task_id']
    asyncio.run(run())


def test_unknown_action_after_restart_retains_task_identity_and_barrier(tmp_path):
    async def run():
        facade, transport, gateway, ledger = make_facade(tmp_path)

        async def unknown(goal):
            raise TimeoutError('SIMULATED_RESULT_TIMEOUT')

        transport.finish = unknown
        request = {'request_id': 'timeout', 'targets': ['cup'], 'wait_seconds': 1}
        result = await facade.submit('observe_scene', request)
        assert result['state'] == 'UNKNOWN'
        assert ledger.load('preparation_barrier')
        reconstructed = TaskInterface(PreparationGateway(transport, ledger), ReplayReads())
        again = await reconstructed.submit('observe_scene', request)
        assert again['task_id'] == result['task_id'] and again['state'] == 'UNKNOWN'
        assert not transport.calls
    asyncio.run(run())


def test_stale_missing_and_wrong_context_stop_without_claiming_completion(tmp_path):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path, {
            'calculate_active_grasps': 'ACTIVE_OBSERVATION_STALE: refresh required'})
        result = await facade.submit('prepare_manipulation', {
            'request_id': 'stale', 'operation': 'grasp', 'object_name': 'cup', 'wait_seconds': 1})
        assert result['state'] == 'BLOCKED' and result['blocker']['refresh_required']
        assert 'cache_grasp_plan' not in transport.calls
        with pytest.raises(ValueError, match='TASK_NOT_FOUND'):
            await facade.status('made_up_context')
        original = facade.route

        def wrong(kind, args):
            route = original(kind, args)
            route[1]['bindings'][0]['source_step_id'] = 'wrong_context_capture'
            return route

        facade.route = wrong
        failed = await facade.submit('observe_scene', {
            'request_id': 'wrong', 'targets': ['cup'], 'detail': 'presence', 'wait_seconds': 1})
        assert failed['state'] == 'BLOCKED'
        assert any(s in failed['status'] for s in ('DEPENDENCY_FAILED', 'UNKNOWN_CONTEXT'))
    asyncio.run(run())


def test_relative_view_uses_internal_calculation_exception_not_generic_execution(tmp_path):
    async def run():
        facade, transport, gateway, _ = make_facade(tmp_path)
        result = await facade.submit('plan_camera_view', {
            'request_id': 'view', 'mode': 'relative', 'frame': 'camera_optical',
            'translation_m': [.02, 0, 0], 'rotation_rad': [0, 0, 0], 'wait_seconds': 1})
        assert result['success'], result
        assert result['result']['status'] == 'PARTIAL'
        assert result['result']['endpoint_safe']
        assert not result['result']['full_path_validated']
        with pytest.raises(ValueError, match='FUNCTION_NOT_ALLOWED'):
            gateway.make_step('gripper_close', {
                'step_id': 'forbidden', 'arguments': {}, 'bindings': []}, task_level=True)
    asyncio.run(run())


def test_native_task_tools_are_opt_in_and_preserve_all_diagnostic_schemas(tmp_path):
    facade, transport, gateway, ledger = make_facade(tmp_path)

    def rpc(client, method, params):
        headers = {'Authorization': 'Bearer '+TOKEN,
                   'Accept': 'application/json, text/event-stream',
                   'MCP-Protocol-Version': '2025-06-18'}
        body = {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}
        result = client.post('/mcp', headers=headers, json=body).json()
        return result['result']

    ordinary = create_app(ReplayReads(), TOKEN, DOCS, execution=ledger, preparation=gateway,
                          preparation_only=True, require_mock_inputs=True)
    enhanced = create_app(ReplayReads(), TOKEN, DOCS, execution=ledger, preparation=gateway,
                          preparation_only=True, require_mock_inputs=True, task_tools=True)
    with TestClient(ordinary, base_url='http://127.0.0.1:8765') as old, TestClient(
            enhanced, base_url='http://127.0.0.1:8765') as new:
        old_tools = {t['name']: t for t in rpc(old, 'tools/list', {})['tools']}
        new_tools = {t['name']: t for t in rpc(new, 'tools/list', {})['tools']}
        assert len(old_tools) == 54 and len(new_tools) == 62
        assert set(new_tools) - set(old_tools) == set(NAMES)
        for name, tool in old_tools.items():
            assert new_tools[name] == tool
        described = rpc(new, 'tools/call', {'name': 'describe_workflow',
                                            'arguments': {'operation': 'grasp'}})
        assert described['structuredContent']['success'] and not transport.calls
        rejected = rpc(new, 'tools/call', {'name': 'prepare_manipulation', 'arguments': {
            'request_id': 'physical', 'operation': 'execute_plan', 'object_name': 'cup'}})
        assert rejected['isError'] and not transport.calls
        before = ledger.db.execute('SELECT COUNT(*) FROM execution').fetchone()[0]
        invalid_view = rpc(new, 'tools/call', {'name': 'plan_camera_view', 'arguments': {
            'request_id': 'bad_view', 'mode': 'relative'}})
        assert invalid_view['isError']
        assert 'RELATIVE_VIEW_FIELDS_REQUIRED' in invalid_view['structuredContent']['status']
        assert not transport.calls
        assert ledger.db.execute('SELECT COUNT(*) FROM execution').fetchone()[0] == before
        for tool in definitions():
            jsonschema.Draft202012Validator.check_schema(tool.inputSchema)


def test_task_profile_requires_preparation_only_before_transport_creation(monkeypatch, tmp_path):
    from task_pipeline.mcp.server import main
    monkeypatch.setattr('sys.argv', ['pipeline_mcp', '--task-tools', '--token-file',
                                     str(tmp_path/'unused')])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2


def test_checked_in_task_schemas_match_native_factory():
    schemas = json.loads((DOCS/'TASK_TOOLS.json').read_text())
    assert schemas == [tool.model_dump(by_alias=True, exclude_none=True) for tool in definitions()]


@pytest.mark.parametrize('values', [
    {'mode': 'relative', 'frame': 'camera_optical',
     'translation_m': [0, 0, 0], 'rotation_rad': [0, .05235987755982989, 0]},
    {'mode': 'target', 'target_name': 'cup'},
    {'mode': 'target', 'target_cell_id': '0,0,0', 'camera_distance_m': .45},
])
def test_flat_camera_view_valid_modes(tmp_path, values):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path)
        result = await facade.submit('plan_camera_view', dict(
            values, request_id='valid_view', wait_seconds=1))
        assert result['state'] == 'COMPLETED', result
        expected = 'get_observe_view' if values['mode'] == 'relative' else 'plan_next_best_view'
        assert expected in transport.calls
    asyncio.run(run())


@pytest.mark.parametrize('values', [
    {}, {'mode': 'relative'}, {'mode': 'sideways'},
    {'mode': 'relative', 'frame': 'base_link',
     'translation_m': [0, 0, 0], 'rotation_rad': [0, 0, 0]},
    {'mode': 'relative', 'frame': 'camera_optical', 'rotation_rad': [0, 0, 0]},
    {'mode': 'relative', 'frame': 'camera_optical', 'translation_m': [0, 0, 0]},
    {'mode': 'relative', 'frame': 'camera_optical',
     'translation_m': [0, 0], 'rotation_rad': [0, 0, 0]},
    {'mode': 'relative', 'frame': 'camera_optical',
     'translation_m': [.15, .15, 0], 'rotation_rad': [0, 0, 0]},
    {'mode': 'relative', 'frame': 'camera_optical',
     'translation_m': [.21, 0, 0], 'rotation_rad': [0, 0, 0]},
    {'mode': 'relative', 'frame': 'camera_optical',
     'translation_m': [0, 0, 0], 'rotation_rad': [0, 4, 0]},
    {'mode': 'relative', 'frame': 'camera_optical',
     'translation_m': [0, 0, 0], 'rotation_rad': [0, float('nan'), 0]},
    {'mode': 'relative', 'frame': 'camera_optical', 'target_name': 'cup',
     'translation_m': [0, 0, 0], 'rotation_rad': [0, 0, 0]},
    {'mode': 'relative', 'frame': 'camera_optical', 'target_cell_id': '0,0,0',
     'translation_m': [0, 0, 0], 'rotation_rad': [0, 0, 0]},
    {'mode': 'target'}, {'mode': 'target', 'target_name': 'cup', 'target_cell_id': '0,0,0'},
    {'mode': 'target', 'target_name': ''},
    {'mode': 'target', 'target_name': 'cup', 'frame': 'camera_optical'},
    {'mode': 'target', 'target_name': 'cup', 'translation_m': [0, 0, 0]},
    {'mode': 'target', 'target_name': 'cup', 'rotation_rad': [0, 0, 0]},
    {'mode': 'target', 'target_name': 'cup', 'camera_distance_m': 1.1},
    {'mode': 'target', 'target_name': 'cup', 'camera_distance_m': float('nan')},
    {'mode': 'target', 'target_name': 'cup', 'extra': 'execute_plan'},
])
def test_camera_view_rejects_invalid_before_task_creation(tmp_path, values):
    async def run():
        facade, transport, _, ledger = make_facade(tmp_path)
        before = ledger.db.execute('SELECT COUNT(*) FROM execution').fetchone()[0]
        with pytest.raises((ValueError, jsonschema.ValidationError)):
            await facade.submit('plan_camera_view', dict(values, request_id='invalid_view'))
        assert not transport.calls
        assert ledger.db.execute('SELECT COUNT(*) FROM execution').fetchone()[0] == before
    asyncio.run(run())


def test_published_camera_view_schema_is_flat():
    tool = next(t for t in definitions() if t.name == 'plan_camera_view')

    def check(value):
        if isinstance(value, dict):
            assert not set(value) & {'oneOf', 'anyOf', 'allOf', 'if', 'then', 'not'}
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(tool.inputSchema)
    assert tool.inputSchema['required'] == ['request_id', 'mode']
    assert all(p.get('description') for p in tool.inputSchema['properties'].values())


def test_unchanged_view_result_is_not_a_changed_view_and_retry_is_same_task(tmp_path):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path)
        original = transport.backend

        def unchanged(name, request):
            result = original(name, request)
            if name == 'get_observe_view':
                result.status = 'NO_REACHABLE_CHANGE'
                result.moved_target = result.endpoint_safe = False
                result.achieved_fraction = 0.
            return result
        transport.store._call = unchanged
        arguments = {'request_id': 'unchanged', 'mode': 'relative', 'frame': 'camera_optical',
                     'translation_m': [0, 0, 0], 'rotation_rad': [0, .05235987755982989, 0],
                     'wait_seconds': 1}
        result = await facade.submit('plan_camera_view', arguments)
        assert result['state'] == 'COMPLETED'
        assert result['result']['status'] == 'NO_REACHABLE_CHANGE'
        assert result['result']['achieved_fraction'] == 0.
        assert not result['result']['moved_target']
        assert not result['result']['endpoint_safe']
        assert not result['result']['full_path_validated']
        again = await facade.submit('plan_camera_view', arguments)
        assert again['task_id'] == result['task_id']
        assert transport.calls.count('get_observe_view') == 1
    asyncio.run(run())


def test_validation_reports_only_backend_evidence_and_known_bypass_reasons(tmp_path):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path)
        original = transport.store._call

        def reported(name, request):
            response = original(name, request)
            if name in ('prepare_pick_place', 'validate_prepared_pick_place'):
                response.status = ('ALL_MOTION_PLANNED;LIFT_TRANSFER_DESCENT_VALIDATED;'
                                   'LIFT_ROBOT_ENVIRONMENT_CHECK_BYPASSED;'
                                   'LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED;'
                                   'WITHDRAWAL_ROBOT_ENVIRONMENT_CHECK_BYPASSED;'
                                   'FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED;'
                                   'UNSPECIFIED_CHECK_BYPASSED')
            return response
        transport.store._call = reported
        prepared = await facade.submit('prepare_manipulation', {
            'request_id': 'qualified', 'operation': 'pick_place', 'object_name': 'cup',
            'destination': 'tray', 'wait_seconds': 1})
        assert prepared['current_step'] is None
        assert prepared['last_successful_step'] == 'validate_prepared_pick_place'
        for result in (prepared, await facade.validate({
                'request_id': 'qualify_again', 'plan_id': prepared['plan']['plan_id'],
                'revision': 7})):
            evidence = result['validation']
            assert evidence['source_mode'] == 'mock_replay'
            assert evidence['outcome'] == 'PASSED_WITH_BYPASSES'
            assert evidence['checks_passed'] == ['LIFT_TRANSFER_DESCENT_VALIDATED']
            assert not result['execution_allowed'] and not evidence['execution_allowed']
            assert not evidence['scene_freshness_confirmed']
            assert not evidence['physical_readiness_confirmed']
            for item in evidence['checks_bypassed']:
                assert (item['reason'] is None) == (item['check'] == 'UNSPECIFIED_CHECK_BYPASSED')
            assert result['current_step'] is None
        transport.revision = 8
        rejected = await facade.validate({'request_id': 'superseded',
                                         'plan_id': prepared['plan']['plan_id'], 'revision': 7})
        assert rejected['status'] == 'PREPARED_PLAN_SOURCE_OR_REVISION_CHANGED'
        assert not rejected['validation']['backend_validation_succeeded']
        assert not rejected['execution_allowed']
    asyncio.run(run())


def test_running_progress_and_terminal_historical_progress_are_consistent(tmp_path):
    async def run():
        facade, transport, gateway, ledger = make_facade(tmp_path)
        original_finish = transport.finish
        gate = asyncio.Event()

        async def delayed(goal):
            await gate.wait()
            return await original_finish(goal)
        transport.finish = delayed
        result = await facade.submit('prepare_manipulation', {
            'request_id': 'progress', 'operation': 'pick_place', 'object_name': 'cup',
            'destination': 'tray'})
        original_inspect = gateway.inspect

        async def progress(context, include):
            snapshot = await original_inspect(context, include)
            state = json.loads(snapshot['state_json'])
            state[0]['busy'] = True
            active = next(s for s in state[0]['pending'] if s['function'] == 'ground_object')
            state[0]['pending'] = [active]
            snapshot['state_json'] = json.dumps(state)
            return snapshot
        gateway.inspect = progress
        running = await facade.status(result['task_id'])
        assert running['current_step'] == 'ground_object'
        assert running['current_step_id']
        gateway.inspect = original_inspect
        gate.set()
        complete = await facade.status(result['task_id'], 1)
        assert complete['state'] == 'COMPLETED'
        assert complete['current_step'] is None and complete['current_step_id'] is None
        assert complete['last_successful_step'] == 'validate_prepared_pick_place'
        # Old completed aliases may contain stale progress; do not alter their job receipts.
        record = facade.load(result['task_id'])
        record['current_step'] = 'ground_object'
        ledger.save('task:' + result['task_id'], record)
        assert (await facade.status(result['task_id']))['current_step'] is None
    asyncio.run(run())


def test_workflow_count_and_repeated_perception_roles_derive_from_route(tmp_path):
    facade, transport, _, _ = make_facade(tmp_path)
    for operation in ('grasp', 'grasp_lift', 'pick_place', 'observe_scene', 'plan_camera_view'):
        description = facade.describe(operation)['result']
        assert description['step_count'] == len(description['route']) == len(description['steps'])
        assert [s['function'] for s in description['steps']] == description['route']
        if operation == 'pick_place':
            for function in ('verify_presence', 'ground_object', 'segment_object'):
                assert [s['role'] for s in description['steps'] if s['function'] == function] == [
                    'object', 'destination']
    assert not transport.calls


def test_failed_step_and_unknown_task_are_explicit_without_creating_work(tmp_path):
    async def run():
        facade, transport, _, ledger = make_facade(
            tmp_path, {'verify_presence': 'PRESENCE_UNKNOWN'})
        count = ledger.db.execute('SELECT COUNT(*) FROM execution').fetchone()[0]
        with pytest.raises(ValueError, match='TASK_NOT_FOUND'):
            await facade.status('task_missing')
        assert ledger.db.execute('SELECT COUNT(*) FROM execution').fetchone()[0] == count
        result = await facade.submit('observe_scene', {'request_id': 'failed_progress',
                                                       'targets': ['cup'], 'wait_seconds': 1})
        assert result['failed_step'] == 'verify_presence'
        assert result['current_step'] is None
        assert result['last_successful_step'] == 'pin_grasp_observation'
        assert result['result']['targets'][0]['status'] == 'UNKNOWN'
    asyncio.run(run())


def test_historical_validation_keeps_evidence_without_claiming_current_readiness(tmp_path):
    facade, _, _, ledger = make_facade(tmp_path)
    ledger.save('task:old_validation', {
        'task_id': 'old_validation', 'kind': 'validate_plan', 'state': 'COMPLETED',
        'status': 'LIFT_TRANSFER_DESCENT_VALIDATED;LIFT_ROBOT_ENVIRONMENT_CHECK_BYPASSED',
        'source_mode': 'mock_replay', 'result': {'validated': True},
        'plan': {'kind': 'prepare_pick_place'}, 'current_step': 'ground_object'})
    result = asyncio.run(facade.status('old_validation'))
    assert result['current_step'] is None
    assert result['validation']['outcome'] == 'PASSED_WITH_BYPASSES'
    assert result['validation']['backend_validation_succeeded']
    assert not result['validation']['scene_freshness_confirmed']
    assert not result['execution_allowed']


def test_pick_place_stores_both_roles_from_exact_capture_before_planning(tmp_path):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path)
        original, stored = transport.backend, []

        def backend(name, query):
            if name == 'store_segmented_target':
                assert query.image.header == transport.header
                assert query.mask.header == transport.header
                assert query.observation_cloud.header == transport.header
                assert query.camera_transform.header == transport.header
                stored.append((query.role, query.name))
            if name == 'prepare_pick_place':
                assert stored == [('object', 'cup'), ('destination', 'tray')]
            return original(name, query)

        transport.store._call = backend
        result = await facade.submit('prepare_manipulation', {
            'operation': 'pick_place', 'object_name': 'cup', 'destination': 'tray',
            'request_id': 'store_both', 'wait_seconds': 1})
        assert result['state'] == 'COMPLETED', result
        assert stored == [('object', 'cup'), ('destination', 'tray')]
        assert transport.calls.count('capture_observation') == 1
        assert 'update_object' not in transport.calls
    asyncio.run(run())


def test_pick_place_stops_before_grasp_if_target_memory_store_fails(tmp_path):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path, fail={
            'store_segmented_target': 'SEGMENTED_TARGET_OBSERVATION_MISMATCH'})
        result = await facade.submit('prepare_manipulation', {
            'operation': 'pick_place', 'object_name': 'cup', 'destination': 'tray',
            'request_id': 'invalid_memory', 'wait_seconds': 1})
        assert result['state'] == 'FAILED'
        assert result['failed_step'] == 'store_segmented_target'
        assert 'cache_grasp_plan' not in transport.calls
        assert 'prepare_pick_place' not in transport.calls
    asyncio.run(run())
