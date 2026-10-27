"""Memory-first acquisition and camera-only execution with fake actuator APIs."""

import copy
from types import SimpleNamespace

from my_actions import srv
import pytest

from test_scene_perception import scene_fixture


def test_unknown_inventory_is_perceived_once_then_location_memory_is_used():
    memory, observation, calls, prompts, hidden, query = scene_fixture()
    original = memory.call

    def backend(name, request):
        if name == 'perceive_scene':
            calls.append(name)
            return memory.scene_perception.perceive(request, srv.PerceiveScene.Response())
        return original(name, request)

    memory.call = backend
    result = memory.acquisition.ensure_memory(query, srv.PerceiveScene.Response())
    assert result.success and set(memory.objects) == {'bowl', 'cup'}
    assert calls.count('perceive_scene') == 1 and len(prompts) == 2
    memory.objects.pop('bowl')  # A directory hint is sufficient to locate its next view.
    before = list(calls)
    result = memory.acquisition.ensure_memory(query, srv.PerceiveScene.Response())
    assert result.success and result.status == 'SCENE_MEMORY_READY_NO_INFERENCE'
    assert calls == before and not any(result.observed)
    assert result.positions[result.names.index('bowl')].x != 0.


def view_fixture(monkeypatch, status='LOOK_REQUIRED', gate=True, fail=''):
    memory, observation, calls, prompts, hidden, _ = scene_fixture()
    state = {'status': status}
    guards = []

    def guard():
        guards.append(True)
        if not gate:
            raise RuntimeError('PHYSICAL_EXECUTION_DISABLED')

    memory.node.execution_api = SimpleNamespace(
        _guard=guard, prepared_grasp_lift=SimpleNamespace(record=None))
    memory.node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=200_000_000_000))
    monkeypatch.setattr('master_node.perception_acquisition.time.sleep', lambda _: None)

    def backend(name, query):
        calls.append(name)
        if name == 'check_target_view':
            return srv.CheckTargetView.Response(
                success=True, status=state['status'], visible=state['status'] == 'READY',
                observation_header=query.observation_cloud.header)
        if name == 'plan_next_best_view':
            assert not query.allow_current_view and query.target_name == 'bowl'
            return srv.PlanNextBestView.Response(success=fail != 'plan', status='NO_VIEW',
                                                 plan_revision=7)
        if name == 'execute_next_best_view':
            assert query.plan_revision == 7 and query.object_name == 'bowl'
            state['status'] = 'READY'
            return srv.HeldMotionStage.Response(success=fail != 'execute',
                                                motion_completed=fail != 'execute', status='FAILED')
        pytest.fail('Unexpected acquisition call: '+name)

    memory.call = backend
    query = srv.CheckTargetView.Request(name='bowl', role='object',
        observation_cloud=observation.cloud, camera_transform=observation.camera_transform)
    return memory, calls, guards, query


def test_ready_view_skips_all_camera_commands(monkeypatch):
    memory, calls, guards, query = view_fixture(monkeypatch, status='READY', gate=False)
    result = memory.acquisition.ensure_view(query, srv.CheckTargetView.Response(), allow_motion=True)
    assert result.success and result.visible
    assert calls == ['check_target_view'] and not guards


def test_moved_view_requires_new_post_arrival_capture(monkeypatch):
    memory, calls, guards, query = view_fixture(monkeypatch)
    result = memory.acquisition.ensure_view(query, srv.CheckTargetView.Response(), allow_motion=True)
    assert result.success and result.status == 'OBJECT_VIEW_REACHED_RECAPTURE_REQUIRED'
    assert not result.visible and calls == [
        'check_target_view', 'plan_next_best_view', 'execute_next_best_view']
    before = list(calls)
    old = memory.acquisition.ensure_view(query, srv.CheckTargetView.Response())
    assert not old.success and 'POST_VIEW_CAPTURE_STALE' in old.status
    assert calls == before
    query = copy.deepcopy(query)
    query.observation_cloud.header.stamp.sec = 201
    query.camera_transform.header = query.observation_cloud.header
    fresh = memory.acquisition.ensure_view(query, srv.CheckTargetView.Response())
    assert fresh.success and fresh.status == 'READY'
    assert calls.count('execute_next_best_view') == 1


@pytest.mark.parametrize('allowed,gate', [(False, True), (True, False)])
def test_motion_free_or_disabled_acquisition_cannot_move(monkeypatch, allowed, gate):
    memory, calls, guards, query = view_fixture(monkeypatch, gate=gate)
    result = memory.acquisition.ensure_view(query, srv.CheckTargetView.Response(),
                                             allow_motion=allowed)
    assert not result.success and calls == ['check_target_view']
    assert memory.node.state == 'IDLE'


def test_unknown_location_does_not_trigger_blind_camera_search(monkeypatch):
    memory, calls, guards, query = view_fixture(monkeypatch, status='LOCATION_UNKNOWN')
    result = memory.acquisition.ensure_view(query, srv.CheckTargetView.Response(), allow_motion=True)
    assert not result.success and result.status.startswith('TARGET_LOCATION_UNKNOWN')
    assert calls == ['check_target_view']


def test_view_planning_failure_never_dispatches_actuation(monkeypatch):
    memory, calls, guards, query = view_fixture(monkeypatch, fail='plan')
    result = memory.acquisition.ensure_view(query, srv.CheckTargetView.Response(), allow_motion=True)
    assert not result.success and result.status.startswith('OBJECT_VIEW_PLAN_FAILED')
    assert 'execute_next_best_view' not in calls and memory.node.state == 'IDLE'


def test_uncertain_view_execution_is_not_retried(monkeypatch):
    memory, calls, guards, query = view_fixture(monkeypatch, fail='execute')
    result = memory.acquisition.ensure_view(query, srv.CheckTargetView.Response(), allow_motion=True)
    assert not result.success and result.status.startswith('STATE_UNKNOWN')
    assert memory.node.state == 'STATE_UNKNOWN'
    assert not memory.acquisition.ensure_view(
        query, srv.CheckTargetView.Response(), allow_motion=True).success
    assert calls.count('execute_next_best_view') == 1


def test_pending_retreat_blocks_next_object_acquisition(monkeypatch):
    memory, calls, guards, query = view_fixture(monkeypatch)
    memory.node.execution_api.prepared_grasp_lift.record = {'phase': 'RELEASED'}
    result = memory.acquisition.ensure_view(query, srv.CheckTargetView.Response(), allow_motion=True)
    assert not result.success and result.status == 'PREVIOUS_PICK_PLACE_NOT_COMPLETE'
    assert not calls
