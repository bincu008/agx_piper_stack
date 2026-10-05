import pytest

from placement_node.node import parse_robobrain_content, robobrain_chat_endpoint


def test_chat_endpoint_accepts_host_or_v1_base_url():
    assert robobrain_chat_endpoint('http://10.10.10.2:1234') == (
        'http://10.10.10.2:1234/v1/chat/completions')
    assert robobrain_chat_endpoint('http://10.10.10.2:1234/v1/') == (
        'http://10.10.10.2:1234/v1/chat/completions')


def test_openai_response_content_is_extracted():
    response = {'choices': [{'message': {'content': 'a brown mug'}}]}
    assert parse_robobrain_content(response) == 'a brown mug'


@pytest.mark.parametrize('response', [{}, {'choices': []}, {
    'choices': [{'message': {'content': ['not', 'text']}}]}])
def test_invalid_openai_response_is_rejected(response):
    with pytest.raises(ValueError):
        parse_robobrain_content(response)


def test_selected_grounding_request():
    from placement_node.grounding import grounding_request
    payload = grounding_request('robobrain2.5-4b', 'original-image', 'the cup')
    assert payload['temperature'] == 0
    assert payload['seed'] == 1234
    assert payload['max_tokens'] == 48
    assert payload['ignore_eos'] is False
    assert payload['cache_prompt'] is False
    assert payload['stream'] is False
    content = payload['messages'][0]['content']
    assert content[0] == {'type': 'image_url', 'image_url': {'url': 'original-image'}}
    assert content[1]['text'] == (
        'Please provide the bounding box coordinate of the region this sentence describes: '
        'the cup. Return only [x1, y1, x2, y2] in normalized 0-1000 coordinates. '
        'Cover the full visible extent of the object, including its handle if present.')


@pytest.mark.parametrize('reply', ['', '[0,0,1]', '[0.5,0,2,2]',
    '[-1,0,2,2]', '[0,0,1001,1000]', '[10,0,1,1]', '[0,1,1,1]',
    '[0,0,1,1] [1,1,2,2]', 'box: [0,0,1,1]'])
def test_malformed_box_rejected(reply):
    from placement_node.perception import parse_grounding_box
    with pytest.raises(ValueError):
        parse_grounding_box(reply, 1280, 720)


def test_response_keeps_original_image_and_target(monkeypatch):
    import json
    import threading
    from types import SimpleNamespace
    import cv2
    import numpy as np
    import base64
    from placement_node.node import PlacementNode

    original = np.zeros((720, 1280, 3), dtype=np.uint8)
    parameters = {'model_name': 'robobrain2.5-4b',
                  'api_server_url': 'http://remote:18091/v1',
                  'request_timeout_sec': 120}
    client = SimpleNamespace(
        _inference_lock=threading.Lock(), latest_image=original,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: SimpleNamespace(info=lambda text: None))
    client._request_robobrain = lambda image, target: PlacementNode._request_robobrain(
        client, image, target)

    def respond(request, timeout):
        payload = json.loads(request.data)
        content = payload['messages'][0]['content']
        sent = cv2.imdecode(np.frombuffer(base64.b64decode(
            content[0]['image_url']['url'].split(',')[1]), dtype=np.uint8), 1)
        assert sent.shape == (720, 1280, 3)
        assert 'the original cup' in content[1]['text']
        assert timeout == 120
        # A newer frame arrives during inference, with different dimensions.
        client.latest_image = np.zeros((100, 100, 3), dtype=np.uint8)
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps({'choices': [{'message': {
                    'content': '[250, 100, 750, 900]'}}]}).encode()
        return Response()

    monkeypatch.setattr('placement_node.node.url_request.urlopen', respond)
    reply, box = PlacementNode._call_robobrain(client, original, 'the original cup')
    assert reply == '[250, 100, 750, 900]'
    assert box == (320, 72, 960, 648)
    assert not client._inference_lock.locked()
