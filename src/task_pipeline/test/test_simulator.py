"""Persistent mock queue tests; all backends are isolated, non-actuating fixtures."""

import json
import threading
from types import SimpleNamespace

from my_actions import srv
from my_actions.msg import PipelineStep

import pytest

from rclpy.serialization import deserialize_message

from task_pipeline.context import StepStore
from task_pipeline.simulator import (
    mock_sources, OutcomeUnknown, simulator_domain, simulator_steps, SimulatorPipeline,
    SimulatorTransport, validate_steps)


class FixtureTransport:
    """Run the real typed StepStore with synthetic capture/pin responses only."""

    def __init__(self, fail=False):
        self.calls = []
        self.context = 'simulator_service_fixture'
        self.exists = False
        self.fail = fail
        self.store = StepStore(self.backend)

    def backend(self, name, request):
        self.calls.append(name)
        if name == 'capture_observation':
            return srv.CaptureObservation.Response(success=not self.fail,
                                                   status='SIMULATED_CAPTURE')
        return srv.PinLayoutObservation.Response(success=True, status='SIMULATED_PIN')

    def execute(self, tasks, feedback):
        context, revision = self.store.submit(self.context, tasks)
        self.exists = True
        status, results = self.store.run_queue(context, expected_revision=revision)
        state = json.loads(self.store.snapshot(context)[0])[0]
        return all(item.success for item in results), {
            'status': status, 'results': state['results']}

    def clear(self):
        if self.exists:
            self.store.clear(self.context)
            self.exists = False


def test_typed_reference_recipe_cannot_dispatch_physical_commands():
    from task_pipeline.registry import FUNCTIONS
    recipe = simulator_steps('cup', 'tray')
    assert len(recipe) == 21
    assert recipe[-1].function_name == 'get_execution_state'
    for item in recipe:
        deserialize_message(bytes(item.request_cdr),
                            FUNCTIONS[item.function_name].service_type.Request)
    for name in ('move_to', 'execute_plan', 'gripper_close', 'commit_floor_slots'):
        with pytest.raises(ValueError, match='FORBIDS_FUNCTION'):
            validate_steps([PipelineStep(step_id='bad', function_name=name)])


def test_append_inserts_only_named_tasks_without_executing():
    transport = FixtureTransport()
    pipeline = SimulatorPipeline(transport)
    assert pipeline.display()['tasks'] == []
    assert pipeline.append('cup', 'tray', .05, ['observe']) == 1
    assert pipeline.append('', '', 0., ['pin_grasp', 'pin_placement']) == 3
    assert [item['id'] for item in pipeline.display()['tasks']] == [
        'observe', 'pin_grasp', 'pin_placement']
    assert transport.calls == []
    with pytest.raises(ValueError, match='OUT_OF_ORDER_INSERT'):
        pipeline.append('', '', 0., ['grasps'])
    assert len(pipeline.tasks) == 3


@pytest.mark.parametrize('fail', [False, True])
def test_whole_queue_flushes_on_completion_or_failure_and_accepts_next_batch(fail):
    transport = FixtureTransport(fail)
    pipeline = SimulatorPipeline(transport)
    pipeline.append('cup', 'tray', .05, ['observe', 'pin_grasp', 'pin_placement'])
    success, report = pipeline.execute()
    assert success is not fail
    assert transport.calls == (['capture_observation'] if fail else [
        'capture_observation', 'pin_grasp_observation', 'pin_placement_observation'])
    assert pipeline.display()['tasks'] == []
    assert pipeline.display()['last_run'] == report
    assert not transport.exists
    pipeline.append('', '', 0., ['observe'])
    assert len(pipeline.tasks) == 1


def test_explicit_flush_discards_queue_without_execution():
    transport = FixtureTransport()
    pipeline = SimulatorPipeline(transport)
    pipeline.append('cup', 'tray', .05, ['observe'])
    pipeline.flush()
    assert pipeline.display()['tasks'] == []
    assert transport.calls == []
    with pytest.raises(ValueError, match='EMPTY'):
        pipeline.execute()


@pytest.mark.parametrize('condition', ['busy', 'unknown'])
def test_busy_or_unknown_queue_cannot_append_execute_or_flush(condition):
    pipeline = SimulatorPipeline(FixtureTransport())
    setattr(pipeline, condition, True)
    for action in (lambda: pipeline.append('cup', 'tray', .05, ['observe']),
                   pipeline.execute, pipeline.flush):
        with pytest.raises(RuntimeError, match='BUSY_OR_UNKNOWN'):
            action()
    assert pipeline.display()[condition]


def test_timeout_is_unknown_and_is_never_automatically_replayed_or_flushed():
    class TimeoutTransport:
        def execute(self, tasks, feedback):
            raise OutcomeUnknown('WAIT_TIMEOUT')

        def clear(self):
            pytest.fail('Unknown calculations must not be cleared')

    pipeline = SimulatorPipeline(TimeoutTransport())
    pipeline.append('cup', 'tray', .05, ['observe'])
    success, report = pipeline.execute()
    assert not success and 'UNKNOWN' in report['status']
    assert pipeline.unknown and len(pipeline.tasks) == 1
    with pytest.raises(RuntimeError):
        pipeline.execute()


def test_service_wait_returns_final_result_or_unknown_timeout():
    class Future:
        def add_done_callback(self, callback):
            callback(self)

        def result(self):
            return 'FINAL_COMPLETION'

    assert SimulatorTransport.wait(Future(), .1) == 'FINAL_COMPLETION'
    future = SimpleNamespace(add_done_callback=lambda callback: None)
    with pytest.raises(OutcomeUnknown):
        SimulatorTransport.wait(future, .001)


def test_concurrent_display_works_but_flush_cannot_race_execution():
    started, finish = threading.Event(), threading.Event()

    class WaitingTransport(FixtureTransport):
        def execute(self, tasks, feedback):
            started.set()
            assert finish.wait(2.)
            return super().execute(tasks, feedback)

    pipeline = SimulatorPipeline(WaitingTransport())
    pipeline.append('cup', 'tray', .05, ['observe'])
    worker = threading.Thread(target=pipeline.execute)
    worker.start()
    try:
        assert started.wait(1.)
        assert pipeline.display()['busy']
        with pytest.raises(RuntimeError, match='BUSY'):
            pipeline.flush()
    finally:
        finish.set()
        worker.join(2.)
    assert not worker.is_alive()
    assert not pipeline.tasks


@pytest.mark.parametrize('domain', ['0', '-1', '233', 'bad'])
def test_live_or_invalid_domain_is_rejected(domain):
    with pytest.raises(ValueError):
        simulator_domain(domain)


def test_mixed_or_real_publishers_are_rejected():
    publishers = [SimpleNamespace(node_namespace='/', node_name='mock_hardware')]
    node = SimpleNamespace(get_publishers_info_by_topic=lambda _: publishers)
    assert mock_sources(node)[0]
    publishers.append(SimpleNamespace(node_namespace='/camera', node_name='camera'))
    assert not mock_sources(node)[0]


def test_master_physical_gate_blocks_calculation_admission(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '89')
    transport = SimulatorTransport.__new__(SimulatorTransport)
    transport.node = SimpleNamespace(get_publishers_info_by_topic=lambda _: [
        SimpleNamespace(node_namespace='/', node_name='mock_hardware')])
    transport.request = lambda *a: srv.GetWorldState.Response(state_json=json.dumps({
        'robot': {'physical_grasp_execution_enabled': True, 'state': 'IDLE', 'holding': None}}))
    with pytest.raises(RuntimeError, match='PHYSICAL_EXECUTION_DISABLED'):
        transport.guard()


def test_invalid_targets_or_changed_targets_do_not_mutate_the_queue():
    pipeline = SimulatorPipeline(FixtureTransport())
    with pytest.raises(ValueError, match='OBJECT_AND_DESTINATION_REQUIRED'):
        pipeline.append('', '', .05, ['observe'])
    pipeline.append('cup', 'tray', .05, ['observe'])
    with pytest.raises(ValueError, match='FLUSH_BEFORE_CHANGING_TARGETS'):
        pipeline.append('bowl', '', .05, ['pin_grasp'])
    assert len(pipeline.tasks) == 1


def test_selected_grasp_preview_has_source_pose_and_survives_queue_cleanup():
    from my_actions.msg import GraspCandidate
    from task_pipeline.simulator import selected_grasp_markers
    candidate = GraspCandidate()
    candidate.grasp_pose.header.frame_id = 'base_link'
    candidate.pregrasp_pose.header.frame_id = 'base_link'
    candidate.grasp_pose.pose.orientation.w = 1.
    candidate.pregrasp_pose.pose.orientation.w = 1.
    candidate.grasp_pose.pose.position.x = .3
    preview = selected_grasp_markers(candidate)
    assert len(preview.markers) == 5
    assert preview.markers[1].pose.position.x == .3
    assert 'calculation only' in preview.markers[2].text
    assert len(selected_grasp_markers().markers) == 1


def test_selected_finger_bars_match_vamp_dimensions_opening_and_tcp_offset():
    from my_actions.msg import GraspCandidate
    from task_pipeline.simulator import selected_grasp_markers
    candidate = GraspCandidate()
    candidate.grasp_pose.header.frame_id = 'base_link'
    candidate.pregrasp_pose.header.frame_id = 'base_link'
    candidate.grasp_pose.pose.orientation.w = 1.
    candidate.pregrasp_pose.pose.orientation.w = 1.
    candidate.jaw_opening_m = .04
    bars = [m for m in selected_grasp_markers(candidate, (.01, .02, .07, -.0615)).markers
            if m.ns == 'simulator_fingers']
    assert len(bars) == 2
    assert bars[0].scale.x == .01 and bars[0].scale.z == .07
    assert bars[0].pose.position.z == pytest.approx(.0965)
    assert bars[0].pose.position.y == pytest.approx(.025)
    assert bars[1].pose.position.y == pytest.approx(-.025)
    assert (bars[0].pose.position.y - bars[1].pose.position.y - bars[0].scale.x
            == pytest.approx(candidate.jaw_opening_m))


def test_pregrasp_bars_use_the_pregrasp_pose_and_blue_color():
    from my_actions.msg import GraspCandidate
    from task_pipeline.simulator import selected_grasp_markers
    candidate = GraspCandidate()
    for pose in (candidate.grasp_pose, candidate.pregrasp_pose):
        pose.header.frame_id = 'base_link'
        pose.pose.orientation.w = 1.
    # A quarter-turn around Z rotates the finger separation onto X.
    candidate.pregrasp_pose.pose.position.x = .2
    candidate.pregrasp_pose.pose.position.z = .1
    candidate.pregrasp_pose.pose.orientation.z = 2 ** -.5
    candidate.pregrasp_pose.pose.orientation.w = 2 ** -.5
    candidate.jaw_opening_m = .04
    markers = selected_grasp_markers(candidate, (.01, .02, .07, -.0615)).markers
    bars = [m for m in markers if m.ns == 'simulator_pregrasp_fingers']
    assert len(bars) == 2
    assert bars[0].pose.position.x == pytest.approx(.175)
    assert bars[1].pose.position.x == pytest.approx(.225)
    assert bars[0].pose.position.z == pytest.approx(.1965)
    assert bars[0].color.b == 1. and bars[0].color.g == .5
    assert len([m for m in markers if m.ns == 'simulator_fingers']) == 2
