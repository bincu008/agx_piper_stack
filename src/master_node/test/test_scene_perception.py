"""Batch presence/grounding, per-object geometry and retained destination memory."""

import copy
import json

from my_actions import srv
import numpy as np
import pytest

from master_node.scene_perception import parse_mapping, pixel_box
from test_manipulation_memory import fixture


def scene_fixture(missing=()):
    memory, observation, calls, _, _ = fixture()
    original = memory.call
    prompts = []
    hidden = set(missing)

    def backend(name, query):
        if name == 'infer_vision':
            calls.append(name)
            prompts.append(query.user_prompt)
            if 'SCENE PRESENCE.' in query.user_prompt:
                targets, _ = json.JSONDecoder().raw_decode(
                    query.user_prompt[query.user_prompt.index('['):])
                values = {item['key']: item['description'] not in hidden for item in targets}
            else:
                descriptions, _ = json.JSONDecoder().raw_decode(
                    query.user_prompt[query.user_prompt.index('{'):])
                targets = [{'key': key, 'description': next(name for name in ('bowl', 'cup', 'tray')
                                               if name in description)}
                           for key, description in descriptions.items()]
                assert 'FOUR numbers' in query.user_prompt
                boxes = {'bowl': [0, 0, 300, 1000], 'cup': [330, 0, 620, 1000],
                         'tray': [650, 0, 1000, 1000]}
                values = {item['key']: None if item['description'] in hidden
                          else boxes[item['description']] for item in targets}
            assert query.image == observation.image
            return srv.VisionInference.Response(success=True, text=json.dumps(values))
        if name == 'segment_object':
            calls.append(name)
            mask = np.zeros((32, 32), np.uint8)
            x0, y0, x1, y1 = query.bbox
            mask[y0:y1+1, x0:x1+1] = 255
            image = memory.bridge.cv2_to_imgmsg(mask, 'mono8')
            image.header = observation.image.header
            return srv.SegmentGroundedImage.Response(success=True, mask=image)
        if name in ('verify_presence', 'ground_object'):
            pytest.fail('Single-target VLM inference during batch perception')
        return original(name, query)

    memory.call = backend
    request = srv.PerceiveScene.Request(
        object_names=['bowl', 'cup'], active_object_name='bowl', destination_name='tray',
        image=observation.image, observation_cloud=observation.cloud,
        camera_transform=observation.camera_transform)
    return memory, observation, calls, prompts, hidden, request


def test_all_targets_use_two_batch_prompts_and_one_supplied_observation():
    memory, observation, calls, prompts, hidden, query = scene_fixture()
    raw_cloud = bytes(observation.cloud.data)
    result = memory.scene_perception.perceive(query, srv.PerceiveScene.Response())
    assert result.success, result.status
    assert set(result.names) == {'bowl', 'cup', 'tray'} and all(result.observed)
    assert len(prompts) == 2 and calls.count('infer_vision') == 2
    assert calls.count('segment_object') == 3 and 'capture_observation' not in calls
    assert all(all(name in prompt for name in ('bowl', 'cup', 'tray')) for prompt in prompts)
    assert set(memory.objects) == {'bowl', 'cup'}
    assert memory.destination['name'] == 'tray'
    assert all(record['observation'].cloud.header == observation.cloud.header
               for record in memory.objects.values())
    assert bytes(observation.cloud.data) == raw_cloud


def test_missing_object_keeps_location_hint_without_graspable_stale_geometry():
    memory, observation, calls, prompts, hidden, query = scene_fixture()
    assert memory.scene_perception.perceive(query, srv.PerceiveScene.Response()).success
    hidden.add('cup')
    result = memory.scene_perception.perceive(query, srv.PerceiveScene.Response())
    assert result.success and not result.observed[result.names.index('cup')]
    assert 'cup' not in memory.objects
    assert any(row['name'] == 'cup' for row in memory.directory.values())
    stale = memory.get_target(srv.ResolveManipulationTarget.Request(name='cup'),
                              srv.ResolveManipulationTarget.Response())
    assert not stale.success
    assert 'bowl' in memory.objects and memory.destination is not None


def test_destination_base_and_occupied_grid_survive_repeated_batch_queries():
    memory, observation, calls, prompts, hidden, query = scene_fixture()
    assert memory.scene_perception.perceive(query, srv.PerceiveScene.Response()).success
    base = memory.destination['base_cloud']
    occupied = {'occupied': ['0,0']}
    memory.destination['grid'] = occupied
    memory.destination['placements']['bowl-id'] = {
        'pose': 'accepted', 'name': 'bowl', 'object_id': 'bowl-id',
        'source_header': observation.cloud.header}
    hidden.add('tray')
    result = memory.scene_perception.perceive(query, srv.PerceiveScene.Response())
    assert result.success
    assert memory.destination['base_cloud'] is base
    assert memory.destination['grid'] is occupied
    assert memory.destination['placements']['bowl-id']['pose'] == 'accepted'
    assert memory.destination['observation'].cloud.header == observation.cloud.header
    assert calls.count('ground_object') == 0


@pytest.mark.parametrize('text', ['{}', '{"t0":true,"t0":false}',
                                  '{"t0":true,"extra":false}', 'no JSON'])
def test_malformed_batch_does_not_mutate_memory(text):
    memory, observation, calls, prompts, hidden, query = scene_fixture()
    original = memory.call
    memory.call = lambda name, q: (srv.VisionInference.Response(success=True, text=text)
                                  if name == 'infer_vision' else original(name, q))
    result = memory.scene_perception.perceive(query, srv.PerceiveScene.Response())
    assert not result.success and not memory.objects and memory.destination is None
    assert 'segment_object' not in calls


def test_bad_geometry_association_fails_before_inference():
    memory, observation, calls, prompts, hidden, query = scene_fixture()
    query = copy.deepcopy(query)
    query.camera_transform.header.stamp.sec += 1
    result = memory.scene_perception.perceive(query, srv.PerceiveScene.Response())
    assert not result.success and not prompts


def test_normalized_box_bounds_and_duplicate_keys_are_strict():
    assert pixel_box([0, 0, 1000, 1000], 32, 32) == [0, 0, 31, 31]
    for bad in ([0, 0, 0, 20], [-1, 0, 20, 20], [True, 0, 20, 20], [0, 0, 1001, 20]):
        with pytest.raises(ValueError):
            pixel_box(bad, 32, 32)
    with pytest.raises(ValueError):
        parse_mapping('{"t0":true,"t0":false}', ['t0', 't1'])


def test_default_inventory_contains_both_task_objects_and_requested_destination():
    memory, observation, calls, prompts, hidden, query = scene_fixture()
    names, roles = memory.scene_perception.targets(srv.PerceiveScene.Request(
        active_object_name='beige cup', destination_name='dark rectangular tray'))
    assert set(names) == {'beige cup', 'small white bowl at bottom left', 'dark rectangular tray'}
    assert roles[-1] == 'destination'
    other, _ = memory.scene_perception.targets(srv.PerceiveScene.Request(
        active_object_name='small white bowl at bottom left',
        destination_name='dark rectangular tray'))
    assert names == other  # Switching active objects must not reorder VLM keys.
    names, _ = memory.scene_perception.targets(srv.PerceiveScene.Request(
        active_object_name='beige cup, grasp its rim', destination_name='tray'))
    assert names.count('beige cup') == 0 and 'beige cup, grasp its rim' in names


def test_absent_nonselected_target_cannot_block_valid_selected_geometry():
    memory, observation, calls, prompts, hidden, query = scene_fixture(missing=('cup',))
    original = memory.call

    def backend(name, request):
        result = original(name, request)
        if name == 'infer_vision' and 'FOUR numbers' in request.user_prompt:
            values = json.loads(result.text)
            values['t1'] = [10, 20]  # Ignored because cup presence was false.
            result.text = json.dumps(values)
        return result

    memory.call = backend
    result = memory.scene_perception.perceive(query, srv.PerceiveScene.Response())
    assert result.success, result.status
    assert 'bowl' in memory.objects and 'cup' not in memory.objects
