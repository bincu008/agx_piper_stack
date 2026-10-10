"""Check API phase accounting, failure visibility and clock assumptions."""

from types import SimpleNamespace

import pytest

from task_pipeline.api_metrics import summarize_api_calls


def test_phases_cover_request_to_response_without_double_counting():
    """All seven intervals cover the complete client call exactly once."""
    call = dict(operation_id='one', function='search', started_ns=0, sent_ns=10,
                response_received_ns=90, finished_ns=100, success=True)
    stages = {'one': {
        'dispatcher': SimpleNamespace(received_ns=20, finished_ns=80),
        'master': SimpleNamespace(backend_started_ns=30, backend_finished_ns=70)}}
    result = summarize_api_calls([call], stages)
    phases = result['calls'][0]['phases_seconds']
    assert result['complete_traces'] == 1
    assert phases['backend'] == 40 / 1e9
    assert sum(phases.values()) == pytest.approx(result['total_api_seconds'])


@pytest.mark.parametrize('stages', [{}, {'one': {
    'dispatcher': SimpleNamespace(received_ns=200, finished_ns=80),
    'master': SimpleNamespace(backend_started_ns=30, backend_finished_ns=70)}}])
def test_missing_or_inconsistent_stamps_are_reported_incomplete(stages):
    """Do not invent transport intervals or subtract clocks from other hosts."""
    call = dict(operation_id='one', function='search', started_ns=0, sent_ns=10,
                response_received_ns=90, finished_ns=100, success=False)
    result = summarize_api_calls([call], stages)
    assert result['complete_traces'] == 0
    assert result['functions'][0]['calls'] == 1
    assert result['calls'][0]['success'] is False
    assert result['total_api_seconds'] == 100 / 1e9
