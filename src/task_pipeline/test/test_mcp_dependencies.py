"""Dependency discovery/recovery fixtures; never create ROS/hardware clients."""

import asyncio
import copy
import json
from pathlib import Path

import jsonschema

from my_actions import srv
from my_actions.msg import GraspCandidate, PlacementCandidate

import pytest

pytest.importorskip('mcp', reason='Optional MCP dependencies are not installed')


from sensor_msgs.msg import Image  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from task_pipeline.context import StepStore  # noqa: E402
from task_pipeline.mcp.dependencies import (  # noqa: E402
    dependency_contract, recipe_example, recovery_hint, RESOURCE)
from task_pipeline.mcp.execution import PreparedExecution  # noqa: E402
from task_pipeline.mcp.preparation import (  # noqa: E402
    binding_type, compatible_binding, PREPARATION, PreparationGateway)
from task_pipeline.mcp.server import create_app, definitions  # noqa: E402
from task_pipeline.registry import FUNCTIONS  # noqa: E402

DOCS = Path(__file__).resolve().parents[3] / 'docs/mcp'
TOKEN = 'dependency_fixture_token_01234567890123456789'


def rpc(client, method, params):
    return client.post('/mcp', headers={
        'Authorization': 'Bearer ' + TOKEN,
        'Accept': 'application/json, text/event-stream', 'MCP-Protocol-Version': '2025-06-18'},
        json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}).json()


class NoHardwareReads:
    def input_provenance(self):
        return {'classification': 'live_driver_publishers', 'SIMULATED': True}

    async def call(self, *args):
        pytest.fail('Dependency discovery must not call ROS')


def test_every_tool_has_dependency_metadata_and_valid_field_paths():
    for tool in definitions(True, True):
        contract = tool.meta['rosPipeline']
        assert contract['prerequisites']['context_state']
        assert contract['execution_sequence'][-1] == tool.name
        assert 'Prerequisites:' in tool.description
        assert 'Execution sequence:' in tool.description
        assert contract['resource'] == RESOURCE
        for item in contract['prerequisites']['bindings']:
            source = ('sequence<uint8>' if item['source_field'] == '@response_cdr' else
                      binding_type(FUNCTIONS[item['source_tool']].service_type.Response,
                                   item['source_field']))
            target = binding_type(FUNCTIONS[tool.name].service_type.Request, item['target_field'])
            assert compatible_binding(source, target), (tool.name, item)
        if contract['example']:
            jsonschema.validate(contract['example']['input'], tool.inputSchema)


def test_presence_ordering_is_not_an_invented_presence_result_binding():
    ground = dependency_contract('ground_object')
    assert any(p['tool'] == 'verify_presence' and p['requirement'] == 'recommended'
               for p in ground['prerequisites']['prior_calls'])
    assert [b['source_tool'] for b in ground['prerequisites']['bindings']] == [
        'capture_observation']
    resolve = dependency_contract('resolve_target')
    assert resolve['execution_sequence'] == ['resolve_target']
    assert all(p['requirement'] != 'required' for p in resolve['prerequisites']['prior_calls'])
    assert len(resolve['prerequisites']['bindings']) == 3
    assert dependency_contract('plan_motion')['prerequisites']['bindings'] == []


@pytest.mark.parametrize('name,status,producer', [
    ('validate_prepared_grasp_lift', 'NO_PREPARED_GRASP_LIFT', 'prepare_grasp_lift'),
    ('get_cached_grasp', 'NO_FRESH_CACHED_GRASP', 'cache_grasp_plan'),
    ('calculate_active_grasps', 'ACTIVE_TARGET_REQUIRED', 'store_segmented_target'),
    ('resolve_target', 'SYNCHRONIZED_IMAGE_AND_CLOUD_REQUIRED', 'capture_observation'),
])
def test_failure_guidance_names_actual_missing_producer(name, status, producer):
    original = {'success': False, 'status': status}
    saved = copy.deepcopy(original)
    hint = recovery_hint(name, original)
    assert any(producer in value for value in hint['missing_prerequisites'])
    assert producer in hint['execution_sequence']
    assert original == saved
    assert not hint['automatic_retry_allowed']


def test_unknown_hint_preserves_blocking_and_real_caller_step_ids():
    bindings = [{'target_field': 'image', 'source_step_id': 'my_actual_capture',
                 'source_field': 'image'}]
    hint = recovery_hint('ground_object', {'success': False, 'status': 'ROS_OUTCOME_UNKNOWN'},
                         {'bindings': bindings})
    assert hint['requires_local_reconciliation']
    assert hint['bindings'] == bindings
    assert 'Do not replay' in hint['guidance']
    assert recovery_hint('ground_object', {'success': True, 'status': 'GROUNDING_READY'}) is None


def test_graph_resource_is_authenticated_and_keeps_blocked_actions_disabled():
    class NoHardwarePreparation:
        async def call(self, *args):
            return {'success': False, 'status': 'NO_PREPARED_GRASP_LIFT'}

    class NoPhysicalExecution:
        async def call(self, *args):
            pytest.fail('Physical dispatch must remain blocked')

    with TestClient(create_app(
            NoHardwareReads(), TOKEN, DOCS, execution=NoPhysicalExecution(),
            preparation=NoHardwarePreparation(), preparation_only=True,
            require_live_inputs=True), base_url='http://127.0.0.1:8765') as client:
        graph = rpc(client, 'resources/read', {'uri': RESOURCE})
        data = json.loads(graph['result']['contents'][0]['text'])
        assert data['documentation_only']
        nodes = {node['name']: node for node in data['nodes']}
        assert nodes['execute_plan']['availability'] == 'disabled_for_current_server_mode'
        assert nodes['validate_prepared_grasp_lift']['prerequisites']['prior_calls'][0][
            'tool'] == 'prepare_grasp_lift'
        response = rpc(client, 'tools/call', {
            'name': 'validate_prepared_grasp_lift', 'arguments': {
                'context_id': '', 'step_id': 'check_lift', 'request_id': 'check_lift',
                'arguments': {'object_name': 'brown cup'}, 'bindings': []}})['result']
        assert response['isError']
        body = response['structuredContent']
        assert body['status'] == 'NO_PREPARED_GRASP_LIFT'
        assert body['recovery']['missing_prerequisites'] == ['prepare_grasp_lift']
        assert rpc(client, 'tools/call', {'name': 'execute_plan', 'arguments': {}})[
            'result']['isError']
        malformed = rpc(client, 'tools/call', {
            'name': 'segment_object', 'arguments': {'bad': 'schema'}})['result']
        assert malformed['isError']
        assert 'Input validation error' in malformed['content'][0]['text']


class SyntheticPreparationTransport:
    """Run generated examples through the actual gateway/StepStore, with fake engines."""

    def __init__(self):
        self.calls = []
        self.cached = False
        self.prepared = False
        self.store = StepStore(self.backend)

    def backend(self, name, query):
        self.calls.append(name)
        response = FUNCTIONS[name].service_type.Response()
        if hasattr(response, 'success'):
            response.success = True
        if hasattr(response, 'status'):
            response.status = 'SIMULATED_' + name.upper()
        if name == 'capture_observation':
            response.image = Image(width=2, height=2, encoding='mono8', step=2,
                                   data=[1, 2, 3, 4])
            response.image.header.stamp.sec = 42
            response.cloud.header.stamp.sec = 42
            response.cloud.header.frame_id = 'base_link'
        elif name == 'verify_presence':
            assert query.image.header.stamp.sec == 42
            response.present = True
        elif name == 'ground_object':
            assert list(query.image.data) == [1, 2, 3, 4]
            response.pixel_box = [0, 0, 2, 2]
        elif name == 'segment_object':
            assert list(query.bbox) == [0, 0, 2, 2]
            response.mask = copy.deepcopy(query.image)
        elif name == 'prepare_grasp_input':
            assert query.observation_header.stamp.sec == 42
            response.image_width = response.image_height = 2
        elif name == 'calculate_grasps':
            response.candidates = [GraspCandidate()]
            response.validated_target_mask = query.target_mask
        elif name == 'select_grasp':
            assert len(query.candidates) == 1
            response.candidate = query.candidates[0]
        elif name == 'cache_grasp_plan':
            assert len(query.candidates) == 1
            self.cached = True
        elif name == 'find_placements':
            assert self.cached
            response.candidates = [PlacementCandidate()]
        elif name == 'plan_next_best_view':
            pass
        elif name == 'prepare_pick_place':
            assert self.cached and len(query.candidates) == 1
            assert query.destination_mask.header.stamp.sec == 42
            self.prepared = True
        elif name == 'validate_prepared_pick_place':
            assert self.prepared
        elif name == 'get_cached_grasp':
            response.success = self.cached
            response.status = 'SIMULATED_CACHE_READY' if self.cached else 'NO_FRESH_CACHED_GRASP'
            if self.cached:
                response.candidates = [GraspCandidate()]
        elif name == 'prepare_grasp_lift':
            assert self.cached
            self.prepared = True
        elif name == 'validate_prepared_grasp_lift':
            assert self.prepared
        elif name not in ('pin_grasp_observation', 'pin_placement_observation', 'update_object'):
            pytest.fail('Unexpected/physical backend: ' + name)
        return response

    async def typed(self, name, query, timeout):
        if name == 'get_world_state':
            return srv.GetWorldState.Response(state_json=json.dumps({'robot': {
                'state': 'IDLE', 'holding': None, 'pipeline_executing': False,
                'physical_grasp_execution_enabled': False}}))
        if name == 'run_step':
            ctx, rev, item = self.store.run_one(
                query.context_id, query.step_id, query.function_name, query.request_cdr,
                query.expected_revision, query.bindings)
            return srv.RunPipelineStep.Response(success=item.success, status=item.status,
                                                context_id=ctx, revision=rev, result=item)
        if name == 'get_step_context':
            state, results = self.store.snapshot(query.context_id, query.include_results)
            return srv.GetStepContext.Response(success=True, state_json=state, results=results)
        pytest.fail('Unexpected control endpoint: ' + name)


@pytest.mark.parametrize('recipe_kind', ['grasp_lift', 'pick_place'])
def test_complete_recipe_predecessors_without_actuation(tmp_path, recipe_kind):
    async def run():
        transport = SyntheticPreparationTransport()
        ledger = PreparedExecution(None, tmp_path / 'simulated_dependencies.sqlite')
        gateway = PreparationGateway(transport, ledger, require_physical_disabled=True)
        context = ''
        for example in recipe_example(recipe_kind):
            if example['tool'] not in PREPARATION:
                continue
            query = copy.deepcopy(example['input'])
            query['context_id'] = context
            result = await gateway.call(example['tool'], query)
            assert result['success'], result
            context = result['context_id']
        validator = 'validate_prepared_' + recipe_kind
        example = copy.deepcopy(dependency_contract(validator)['example'])
        example['input']['context_id'] = context
        result = await gateway.call(example['tool'], example['input'])
        assert result['success'], result
        assert transport.prepared
        assert validator in transport.calls
        ledger.db.close()
    asyncio.run(run())


def test_asynchronous_failed_step_keeps_status_and_gains_its_own_guidance():
    class SyntheticJob:
        async def run(self, arguments):
            return {'success': False, 'status': 'STOPPED_ON_FAILURE',
                    'completion': 'FINAL_ACTION_RESULT', 'results': [{
                        'function_name': 'get_cached_grasp', 'step_id': 'my_cached_read',
                        'success': False, 'status': 'NO_FRESH_CACHED_GRASP'}]}

    with TestClient(create_app(NoHardwareReads(), TOKEN, DOCS, preparation=SyntheticJob()),
                    base_url='http://127.0.0.1:8765') as client:
        result = rpc(client, 'tools/call', {'name': 'run_steps', 'arguments': {
            'operation': 'status', 'context_id': 'simulated_context',
            'request_id': 'same_job_id'}})['result']['structuredContent']
        assert result['completion'] == 'FINAL_ACTION_RESULT'
        assert result['status'] == 'STOPPED_ON_FAILURE'
        failed = result['results'][0]
        assert failed['status'] == 'NO_FRESH_CACHED_GRASP'
        assert failed['recovery']['missing_prerequisites'] == ['cache_grasp_plan']


def test_cache_recovery_uses_real_fields_and_same_failed_context(tmp_path):
    async def run():
        transport = SyntheticPreparationTransport()
        ledger = PreparedExecution(None, tmp_path/'cache_recovery_fixture.sqlite')
        gateway = PreparationGateway(transport, ledger, require_physical_disabled=True)
        first = await gateway.call('capture_observation', {
            'context_id': '', 'step_id': 'old_capture', 'request_id': 'old_capture',
            'arguments': {}, 'bindings': []})
        context = first['context_id']
        failed_query = {'context_id': context, 'step_id': 'old_failed_cache',
                        'request_id': 'old_failed_cache',
                        'arguments': {'object_name': 'fixture cup'}, 'bindings': []}
        failed = await gateway.call('get_cached_grasp', failed_query)
        assert not failed['success'] and failed['status'] == 'NO_FRESH_CACHED_GRASP'
        hint = recovery_hint('get_cached_grasp', failed, failed_query)
        assert hint['recovery_steps'] == recovery_hint(
            'get_cached_grasp', failed, failed_query)['recovery_steps']
        for operation in hint['recovery_steps']:
            query = copy.deepcopy(operation['input'])
            assert query['context_id'] == context
            assert query['step_id'] != 'old_failed_cache'
            result = await gateway.call(operation['tool'], query)
            assert result['success'], result
        assert len(result['output']['candidates']) == 1
        assert transport.cached
        ledger.db.close()
    asyncio.run(run())


@pytest.mark.parametrize('tool,invalid_field', [
    ('calculate_active_grasps', 'grasp_input'),
    ('select_grasp', 'calculated_grasps'),
    ('cache_grasp_plan', 'selected_grasp'),
    ('get_cached_grasp', 'cached_grasp_plan'),
])
def test_invented_fields_rejected_with_real_field_list(tmp_path, tool, invalid_field):
    transport = SyntheticPreparationTransport()
    ledger = PreparedExecution(None, tmp_path/(tool + '.sqlite'))
    gateway = PreparationGateway(transport, ledger)
    with pytest.raises(ValueError, match='valid_target_fields=') as error:
        gateway.make_step(tool, {'step_id': 'bad_binding',
                                 'arguments': {'target_description': 'cup'}
                                 if tool == 'calculate_active_grasps' else {}, 'bindings': [{
                                     'target_field': invalid_field,
                                     'source_step_id': 'earlier_step',
                                     'source_field': 'some_field'}]})
    assert invalid_field in str(error.value)
    assert not transport.calls
    ledger.db.close()


def test_active_grasp_contract_explains_internal_preparation():
    contract = dependency_contract('calculate_active_grasps')
    assert contract['example']['input']['bindings'] == []
    internal = {item['tool'] for item in contract['prerequisites']['prior_calls']
                if item['requirement'] == 'internal'}
    assert {'pin_grasp_observation', 'prepare_grasp_input', 'calculate_grasps'} <= internal
    assert 'grasp_input' not in contract['valid_binding_targets']


def test_cache_error_reports_called_but_failed_producers():
    class ExistingFailedContext:
        async def call(self, *args):
            return {'success': False, 'status': 'NO_FRESH_CACHED_GRASP',
                    'context_id': 'existing_context'}

        async def inspect(self, context, include_results):
            assert context == 'existing_context' and not include_results
            return {'success': True, 'state_json': json.dumps([{
                'revision': 49, 'results': [
                    {'step_id': 'bad_inputs', 'function': 'prepare_grasp_input',
                     'success': False, 'status': 'INVALID_PIXEL_BOX'},
                    {'step_id': 'bad_cache', 'function': 'cache_grasp_plan',
                     'success': False, 'status': 'OBJECT_AND_VALID_GRASPS_REQUIRED'}]}])}

    with TestClient(create_app(NoHardwareReads(), TOKEN, DOCS,
                               preparation=ExistingFailedContext()),
                    base_url='http://127.0.0.1:8765') as client:
        result = rpc(client, 'tools/call', {'name': 'get_cached_grasp', 'arguments': {
            'context_id': 'existing_context', 'step_id': 'read_cache',
            'request_id': 'read_cache', 'arguments': {'object_name': 'cup'},
            'bindings': []}})['result']['structuredContent']
        assert result['status'] == 'NO_FRESH_CACHED_GRASP'
        hint = result['recovery']
        assert hint['observed_context_revision'] == 49
        assert [row['status'] for row in hint['observed_prerequisite_results']] == [
            'INVALID_PIXEL_BOX', 'OBJECT_AND_VALID_GRASPS_REQUIRED']
        assert all(step['input']['context_id'] == 'existing_context'
                   for step in hint['recovery_steps'])


def test_active_grasp_name_is_required_before_backend_dispatch(tmp_path):
    transport = SyntheticPreparationTransport()
    ledger = PreparedExecution(None, tmp_path/'active_name.sqlite')
    gateway = PreparationGateway(transport, ledger)
    for arguments in ({}, {'target_description': ''}):
        with pytest.raises(jsonschema.ValidationError):
            gateway.make_step('calculate_active_grasps', {
                'step_id': 'active_grasp', 'arguments': arguments, 'bindings': []})
    assert not transport.calls
    valid = gateway.make_step('calculate_active_grasps', {
        'step_id': 'named_active_grasp', 'arguments': {'target_description': 'cup'},
        'bindings': []})
    assert valid.function_name == 'calculate_active_grasps'
    ledger.db.close()


def test_observation_cache_recovery_explains_pinning_not_extra_seed_fields():
    hint = recovery_hint('calculate_grasps', {
        'success': False, 'status': 'OBSERVATION_NOT_IN_CACHE'})
    assert any('pin_grasp_observation' in item for item in hint['missing_prerequisites'])
    assert 'before inference' in hint['guidance']
    assert 'change stamps' in hint['guidance']
    assert not hint['automatic_retry_allowed']


@pytest.mark.parametrize('workflow', ['grasp', 'grasp_cache', 'pick_place'])
def test_small_model_examples_match_schemas_and_keep_pinning_immediate(workflow):
    from task_pipeline.mcp.examples import workflow_example
    tools = {t.name: t for t in definitions(True, True)}
    example = workflow_example(workflow, 'cup')
    assert not example['execution_performed']
    assert example['steps'][0]['tool'] == 'capture_observation'
    assert example['steps'][1]['tool'] == 'pin_grasp_observation'
    assert example['steps'][1]['input']['bindings'][0] == {
        'target_field': 'observation_header',
        'source_step_id': example['steps'][0]['input']['step_id'], 'source_field': 'cloud.header'}
    for step in example['steps']:
        jsonschema.validate(step['input'], tools[step['tool']].inputSchema)
        assert step['tool'] not in ('execute_plan', 'move_to', 'gripper_close', 'gripper_open')
    again = workflow_example(workflow, 'cup')
    assert again['steps'][0]['input']['request_id'] != example['steps'][0]['input']['request_id']


def test_example_tool_and_summary_do_not_call_ros_or_mutate_state():
    with TestClient(create_app(NoHardwareReads(), TOKEN, DOCS),
                    base_url='http://127.0.0.1:8765') as client:
        result = rpc(client, 'tools/call', {'name': 'get_workflow_example', 'arguments': {
            'workflow': 'grasp', 'object_name': 'cup'}})['result']['structuredContent']
        assert len(result['steps']) == 5
        assert not result['execution_performed']
        assert not result['callable_in_current_mode']  # Read-only mode cannot run preparation.
        listing = rpc(client, 'tools/call', {'name': 'list_functions', 'arguments': {
            'summary_only': True}})['result']['structuredContent']
        assert listing['summary_only'] and not listing['mcp_tools'] and not listing['ros_catalog']
        assert 'get_workflow_example' in listing['allowlist']
        assert 'NOT necessarily' in listing['discovery_notes']['ros_catalog']
        assert 'Native MCP tools/list' in listing['discovery_notes']['mcp_tools']


def test_core_descriptions_fit_small_model_and_keep_full_reference_metadata():
    tools = definitions(True, True)
    assert sum(len(t.description) for t in tools) < 40000
    pin = next(t for t in tools if t.name == 'pin_grasp_observation')
    assert 'cloud.header' in pin.description and 'Never bind image' in pin.description
    assert 'NOT a special scene-layout' in pin.description
    assert pin.meta['rosPipeline']['prerequisites']['bindings']
    assert pin.meta['rosPipeline']['reference_description']


@pytest.mark.parametrize('workflow', ['grasp', 'grasp_cache'])
def test_small_model_active_workflows_execute_in_isolated_gateway_fixture(tmp_path, workflow):
    from task_pipeline.mcp.examples import workflow_example

    class ActiveFixture(SyntheticPreparationTransport):
        def backend(self, name, query):
            if name == 'resolve_target':
                self.calls.append(name)
                assert query.name == 'cup'
                assert query.image.header.stamp == query.observation_cloud.header.stamp
                self.active_name = query.name
                self.mask = copy.deepcopy(query.image)
                return srv.ResolveManipulationTarget.Response(
                    success=True, status='SIMULATED_ACTIVE_READY', image=query.image,
                    observation_cloud=query.observation_cloud,
                    observation_header=query.observation_cloud.header,
                    camera_transform=query.camera_transform, mask=self.mask)
            if name == 'calculate_active_grasps':
                self.calls.append(name)
                assert query.target_description == self.active_name
                return srv.FindGraspCandidates.Response(
                    success=True, status='SIMULATED_GRASPS_READY',
                    candidates=[GraspCandidate()], validated_target_mask=self.mask)
            return super().backend(name, query)

    async def run():
        transport = ActiveFixture()
        ledger = PreparedExecution(None, tmp_path/(workflow + '.sqlite'))
        gateway = PreparationGateway(transport, ledger, require_physical_disabled=True)
        example = workflow_example(workflow, 'cup')
        context = ''
        for step in example['steps']:
            query = copy.deepcopy(step['input'])
            query['context_id'] = context
            result = await gateway.call(step['tool'], query)
            assert result['success'], result
            context = result['context_id']
        assert transport.calls == [step['tool'] for step in example['steps']]
        assert transport.cached == (workflow == 'grasp_cache')
        ledger.db.close()
    asyncio.run(run())


def test_legacy_layout_pin_error_explains_source_recovery():
    hint = recovery_hint(
        'pin_grasp_observation',
        {'success': False,
         'response': {'status': 'LAYOUT_SYNCHRONIZED_OBSERVATION_NOT_AVAILABLE'}})
    assert 'cloud.header' in hint['guidance']
    assert 'legacy status prefix' in hint['guidance']
    assert hint['automatic_retry_allowed'] is False


def test_placement_pin_is_independent_of_grasp_pin():
    for name in ('pin_grasp_observation', 'pin_placement_observation'):
        contract = dependency_contract(name)
        assert contract['execution_sequence'] == ['capture_observation', name]
        required = [p['tool'] for p in contract['prerequisites']['prior_calls']
                    if p['requirement'] == 'required']
        assert required == ['capture_observation']


def test_scene_example_exposes_exact_context_and_triplet_without_resource_reads():
    from task_pipeline.mcp.examples import workflow_example
    example = workflow_example('perceive_scene', 'beige cup', destination_name='tray')
    capture, scene = example['steps']
    assert capture['tool'] == 'capture_observation' and capture['input']['context_id'] == ''
    assert scene['tool'] == 'perceive_scene'
    assert example['context_id_source']['native_mcp_response_path'] == (
        'result.structuredContent.context_id')
    assert scene['input']['context_id'] == 'CONTEXT_ID_FROM_CAPTURE'
    assert scene['input']['bindings'] == [
        {'target_field': target, 'source_step_id': capture['input']['step_id'],
         'source_field': field} for target, field in (
             ('image', 'image'), ('observation_cloud', 'cloud'),
             ('camera_transform', 'camera_transform'))]
    assert scene['input']['request_id'] != capture['input']['request_id']
    assert scene['input']['request_id'] != workflow_example(
        'perceive_scene', 'beige cup')['steps'][1]['input']['request_id']
    for tool in definitions(False, True):
        if tool.name in ('capture_observation', 'perceive_scene'):
            assert 'result.structuredContent.context_id' in tool.description
            assert 'description' in tool.inputSchema['properties']['context_id']
            assert 'context_id' in tool.outputSchema['properties']
    hint = recovery_hint('perceive_scene', {'success': False, 'status': 'MCP_CONTEXT_OWNER_REQUIRED'})
    assert 'context_id=""' in hint['guidance']
    hint = recovery_hint('perceive_scene', {'success': False,
                                          'status': 'SCENE_INFERENCE_UNKNOWN: bad JSON'})
    assert hint['perception_outcome'] == 'unknown'
