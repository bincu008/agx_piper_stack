"""Replay benchmark rejects partial results and never calls actuator operations."""

import time
from types import SimpleNamespace

from my_actions import srv
from my_actions.msg import PipelineStepResult

import pytest

from rclpy.serialization import serialize_message

from task_pipeline.mock_bowl_cup import CALCULATION_APIS, MockBowlCup, OBJECTS
from task_pipeline.registry import catalog


def runner(tmp_path):
    """Build a non-actuating benchmark with a fresh report directory."""
    node = SimpleNamespace(create_subscription=lambda *args: None)
    result = MockBowlCup(node, tmp_path)
    result.started = time.monotonic()
    return result


def test_runner_forbids_actuator_calls(tmp_path):
    """The fixed recipe rejects actuator functions before dispatch."""
    bench = runner(tmp_path)
    with pytest.raises(RuntimeError, match='FORBIDS_API'):
        bench.call('move', 'execute_plan', srv.ExecutePlan.Request())
    assert not bench.rows
    descriptions = {entry['name']: entry for entry in catalog()}
    for name in CALCULATION_APIS:
        assert descriptions[name]['effect'] != 'actuation'


def test_partial_layout_saves_failure_response_and_stops(tmp_path):
    """Retain failed response CDR and report the whole run as failed."""
    bench = runner(tmp_path)
    partial = srv.PlanObjectLayout.Response(
        success=False, status='LAYOUT_PARTIAL cup lift failed', object_names=[OBJECTS[0]])
    payload = serialize_message(partial)
    bench.request = lambda *args, **kwargs: srv.RunPipelineStep.Response(
        success=False, status=partial.status,
        result=PipelineStepResult(success=False, status=partial.status, response_cdr=payload))
    with pytest.raises(RuntimeError, match='LAYOUT_PARTIAL'):
        bench.call('layout', 'plan_layout', srv.PlanObjectLayout.Request())
    assert not bench.rows[0]['success']
    assert (tmp_path/'layout_response.cdr').read_bytes() == payload


def test_missing_second_object_cannot_be_reported_as_complete(tmp_path):
    """A partial object list is never interpreted as full completion."""
    bench = runner(tmp_path)
    with pytest.raises(RuntimeError, match='requested order'):
        bench.inspect_layout(srv.PlanObjectLayout.Response(
            success=True, object_names=[OBJECTS[0]]), [])
    assert not bench.checks[-1]['passed']


def test_old_report_directory_is_rejected(tmp_path):
    """Previous successful artifacts cannot mask a later failed run."""
    (tmp_path/'report.json').write_text('{}')
    with pytest.raises(ValueError, match='fresh output directory'):
        runner(tmp_path)


def test_lift_height_is_explicit_and_invalid_values_are_rejected(tmp_path):
    """Replay requests preserve 20 cm rather than silently using 5 cm."""
    bench = runner(tmp_path)
    assert bench.lift_distance == .20
    assert bench.report['lift_distance_m'] == .20
    with pytest.raises(ValueError, match='finite and positive'):
        MockBowlCup(bench.node, tmp_path/'invalid', float('nan'))
