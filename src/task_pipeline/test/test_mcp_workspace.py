"""Workspace reads, native image preservation and guard fixtures; no hardware."""

import asyncio
import base64
import json

import pytest
pytest.importorskip('mcp')
from starlette.testclient import TestClient  # noqa: E402
from task_pipeline.mcp.server import create_app  # noqa: E402
from task_pipeline.mcp.workspace import encode_image, WorkspaceTools  # noqa: E402
from test_mcp_tasks import DOCS, make_facade, ReplayReads, TOKEN  # noqa: E402


class Reads(ReplayReads):
    async def workspace(self, include_image=False):
        response = {'success': True, 'status': 'CURRENT_OBSERVATION_READY',
                    'state_json': json.dumps({'source_mode': 'mock_replay',
                                              'freshness': {'fresh': True}}),
                    'observation_header': {}, 'camera_transform': {}, 'source_tcp': {}}
        if include_image:
            response['_image_content'] = {'data': base64.b64encode(b'fixture').decode(),
                                          'mimeType': 'image/jpeg', 'width': 2, 'height': 2}
        return {'transport_outcome': 'response', 'response': response}

    async def call(self, name, arguments):
        assert name == 'get_manipulation_memory'
        obj = {'name': 'cup', 'position': {}, 'geometry_age_seconds': 10,
               'verification_age_seconds': 10, 'calibration_matches': True,
               'object_id': 'internal'}
        memory = {'active_objects': [obj], 'active_destination': None, 'directory': {}}
        return {'transport_outcome': 'response', 'response': {
            'success': True, 'state_json': json.dumps(memory)}}


def test_workspace_reads_and_unknown_barrier(tmp_path):
    async def run():
        _, transport, gateway, _ = make_facade(tmp_path)
        tools = WorkspaceTools(Reads(), gateway)
        image = await tools.call('get_camera_image', {})
        assert image['_image_content']['mimeType'] == 'image/jpeg'
        listed = await tools.call('list_objects', {})
        assert listed['result']['objects'][0]['visible_now'] == 'UNKNOWN'
        assert 'object_id' not in listed['result']['objects'][0]
        object_check = {'resource': 'object', 'object_name': 'cup'}
        assert (await tools.call('check_freshness', object_check))['fresh']
        with pytest.raises(ValueError, match='OBJECT_NAME_REQUIRED'):
            await tools.call('check_freshness', {'resource': 'object'})
        gateway.execution.save('preparation_barrier', {'status': 'UNKNOWN'})
        assert not (await tools.call('check_freshness', {'resource': 'image'}))['fresh']
        assert not (await tools.call('check_freshness', {'resource': 'object',
                                                         'object_name': 'cup'}))['fresh']
        assert not transport.calls
    asyncio.run(run())


def test_native_workspace_tools_keep_image_blocks_and_old_surface(tmp_path):
    _, _, gateway, ledger = make_facade(tmp_path)
    app = create_app(Reads(), TOKEN, DOCS, execution=ledger, preparation=gateway,
                     preparation_only=True, require_mock_inputs=True, task_tools=True,
                     workspace_tools=True)
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        def rpc(method, params):
            body = {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}
            return client.post('/mcp', headers={
                'Authorization': 'Bearer '+TOKEN, 'Accept': 'application/json, text/event-stream',
                'MCP-Protocol-Version': '2025-06-18'}, json=body).json()['result']
        assert len(rpc('tools/list', {})['tools']) == 65
        result = rpc('tools/call', {'name': 'get_camera_image', 'arguments': {}})
        assert 'structuredContent' in result, result
        assert result['structuredContent']['SIMULATED']
        assert result['content'][1]['type'] == 'image'
        assert '_image_content' not in result['structuredContent']
        assert 'Zml4dHVyZQ==' not in result['content'][0]['text']


def test_codec_bounds_and_actual_jpeg():
    from sensor_msgs.msg import Image
    assert encode_image(Image()) is None
    image = Image(width=2, height=2, step=6, encoding='bgr8', data=[100]*12)
    encoded = encode_image(image)
    assert base64.b64decode(encoded['data']).startswith(b'\xff\xd8')
    image.width = 5000
    with pytest.raises(ValueError, match='OVERSIZED'):
        encode_image(image)


def test_status_default_current_task_and_no_duplicate_dispatch(tmp_path):
    async def run():
        facade, transport, _, _ = make_facade(tmp_path)
        with pytest.raises(ValueError, match='NO_CURRENT_TASK'):
            await facade.call('get_task_status', {})
        result = await facade.submit('observe_scene', {'request_id': 'current',
                                                       'targets': ['cup'], 'wait_seconds': 1})
        current = await facade.call('get_task_status', {})
        assert result['task_id'] == current['task_id']
        count = len(transport.calls)
        assert (await facade.call('cancel_task', {}))['task_id'] == current['task_id']
        assert len(transport.calls) == count
    asyncio.run(run())


@pytest.mark.parametrize('forward,horizontal,vertical', [
    ({'x': 0., 'y': 1., 'z': 0.}, 'left', 'level'),
    ({'x': 0., 'y': -1., 'z': 0.}, 'right', 'level'),
    ({'x': 0., 'y': 0., 'z': -1.}, 'ambiguous_vertical_view', 'down'),
])
def test_camera_direction_uses_existing_base_frame_vectors(forward, horizontal, vertical):
    from task_pipeline.mcp.tasks import gaze_labels
    result = gaze_labels(forward)
    assert result['horizontal'] == horizontal and result['vertical'] == vertical


def test_unknown_camera_axis_is_not_given_a_direction():
    from task_pipeline.mcp.tasks import gaze_labels
    assert gaze_labels({'x': 0, 'y': 0, 'z': 0})['status'] == 'UNKNOWN'
    assert gaze_labels({})['status'] == 'UNKNOWN'


def test_stale_image_cannot_leak_cached_pixels(tmp_path):
    class Stale(Reads):
        async def workspace(self, include_image=False):
            value = await super().workspace(include_image)
            state = json.loads(value['response']['state_json'])
            state['freshness'] = {'fresh': False, 'reason': 'STREAM_UNAVAILABLE'}
            value['response']['state_json'] = json.dumps(state)
            value['response']['status'] = 'REFRESH_REQUIRED:STREAM_UNAVAILABLE'
            return value

    async def run():
        _, transport, gateway, _ = make_facade(tmp_path)
        result = await WorkspaceTools(Stale(), gateway).call('get_camera_image', {})
        assert result['state'] == 'BLOCKED'
        assert result['status'] == 'REFRESH_REQUIRED:STREAM_UNAVAILABLE'
        assert '_image_content' not in result
        assert not transport.calls
    asyncio.run(run())


def test_workspace_mode_cannot_construct_a_physical_server(tmp_path):
    _, _, gateway, ledger = make_facade(tmp_path)
    with pytest.raises(ValueError, match='WORKSPACE_TOOLS_REQUIRE_TASK_PREPARATION_MODE'):
        create_app(Reads(), TOKEN, DOCS, execution=ledger, preparation=gateway,
                   workspace_tools=True)


def test_live_guard_rejects_mock_pixels_before_workspace_read(tmp_path):
    _, transport, gateway, ledger = make_facade(tmp_path)
    app = create_app(Reads(), TOKEN, DOCS, execution=ledger, preparation=gateway,
                     preparation_only=True, require_live_inputs=True, task_tools=True,
                     workspace_tools=True)
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        response = client.post('/mcp', headers={
            'Authorization': 'Bearer '+TOKEN, 'Accept': 'application/json, text/event-stream',
            'MCP-Protocol-Version': '2025-06-18'}, json={'jsonrpc': '2.0', 'id': 1,
            'method': 'tools/call', 'params': {'name': 'get_camera_image', 'arguments': {}}})
        result = response.json()['result']
        assert result['structuredContent']['status'] == 'LIVE_INPUT_PROVENANCE_REQUIRED'
        assert all(block['type'] != 'image' for block in result['content'])
        assert not transport.calls



def test_only_image_tool_declares_sensing_refresh_side_effect():
    from task_pipeline.mcp.workspace import definitions
    tools = {tool.name: tool for tool in definitions()}
    assert tools['get_camera_image'].annotations.readOnlyHint is False
    assert tools['get_camera_image'].annotations.destructiveHint is False
    assert 'one bounded synchronized' in tools['get_camera_image'].description
    assert tools['check_freshness'].annotations.readOnlyHint is True
    assert tools['list_objects'].annotations.readOnlyHint is True
