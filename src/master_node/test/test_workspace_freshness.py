"""Fake-clock freshness tests; no ROS clients, drivers or actuator calls."""

from master_node.workspace_freshness import WorkspaceFreshnessPolicy

import pytest


def check(**changes):
    values = {'now': 1000., 'source_time': 900., 'source_verified': True,
              'calibration_matches': True, 'generation_matches': True,
              'feedback_valid': True, 'stream_healthy': True}
    values.update(changes)
    return WorkspaceFreshnessPolicy().check(**values)


@pytest.mark.parametrize('age,expected', [(599, True), (600, False), (601, False)])
def test_expiry_boundary(age, expected):
    assert check(source_time=1000.-age).fresh is expected


@pytest.mark.parametrize('changes,reason', [
    ({'source_time': None}, 'SOURCE_MISSING'),
    ({'source_time': 1001.}, 'INVALID_SOURCE_TIME'),
    ({'source_time': float('nan')}, 'INVALID_SOURCE_TIME'),
    ({'source_verified': False}, 'SOURCE_UNVERIFIED'),
    ({'calibration_matches': False}, 'CALIBRATION_CHANGED'),
    ({'generation_matches': False}, 'SOURCE_GENERATION_CHANGED'),
    ({'source_invalidated': True}, 'SOURCE_INVALIDATED'),
    ({'consumed': True}, 'ALREADY_CONSUMED'),
    ({'feedback_valid': False}, 'FEEDBACK_STALE_OR_UNAVAILABLE'),
    ({'stream_healthy': False}, 'STREAM_UNAVAILABLE'),
    ({'additional_deadline': float('inf')}, 'INVALID_DEADLINE'),
])
def test_missing_or_invalid_evidence_cannot_be_overridden_by_age(changes, reason):
    result = check(**changes)
    assert not result.fresh and result.reason == reason


def test_unknown_outcome_remains_unknown_after_expiry():
    result = check(source_time=1., outcome_unknown=True)
    assert not result.fresh
    assert result.reason == 'OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION'


def test_shorter_backend_limit_is_not_relaxed():
    assert check(additional_deadline=999.).reason == 'REFRESH_REQUIRED_EXPIRED'
    assert check(additional_deadline=1001.).expires_in_seconds == 1.


def test_inference_does_not_renew_source_and_backend_deadline_can_be_shorter():
    policy = WorkspaceFreshnessPolicy()
    assert policy.pool_deadline(source_time=100., created_time=650.) == 700.
    assert policy.pool_deadline(source_time=100., created_time=650.,
                                backend_deadline=680.) == 680.
    result = check(source_time=100., now=700.)
    assert not result.fresh


@pytest.mark.parametrize('ttl', [0, -1, 601, float('inf'), float('nan')])
def test_invalid_ttl_rejected(ttl):
    with pytest.raises(ValueError):
        WorkspaceFreshnessPolicy(ttl)


def test_policy_can_be_tightened():
    policy = WorkspaceFreshnessPolicy(60.)
    assert policy.pool_deadline(source_time=100., created_time=101.) == 160.
