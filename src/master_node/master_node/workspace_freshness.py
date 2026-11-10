"""
Master-owned freshness policy for the proposed current workspace.

This policy does not replace driver/contact/synchronization checks or authorize
execution. Callers supply monotonic source times, not timestamps from another
host. Adoption by public tools is staged separately from the existing cache TTL.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FreshnessResult:
    """Explain a Boolean reuse decision without claiming physical readiness."""

    fresh: bool
    reason: str
    age_seconds: float | None
    expires_in_seconds: float


@dataclass(frozen=True)
class WorkspaceFreshnessPolicy:
    """Evaluate immutable source age; reading or new unrelated images never renew it."""

    reuse_ttl_seconds: float = 600.

    def __post_init__(self):
        if (not math.isfinite(self.reuse_ttl_seconds) or
                not 0 < self.reuse_ttl_seconds <= 600.):
            raise ValueError('WORKSPACE_TTL_REQUIRES_0_TO_600_SECONDS')

    def check(self, *, now, source_time, source_verified, calibration_matches,
              generation_matches, feedback_valid, stream_healthy, consumed=False,
              outcome_unknown=False, source_invalidated=False,
              additional_deadline=None):
        """
        Require all applicable evidence and use the earliest applicable deadline.

        Strict backend-specific checks remain authoritative. Unknown outcomes
        outrank expiry: a TTL must never be interpreted as a recovery/reset.
        A false result explains why reuse is unavailable; it does not delete data.
        """
        age, remaining = None, 0.
        if not math.isfinite(now):
            return FreshnessResult(False, 'INVALID_CLOCK', age, remaining)
        if outcome_unknown:
            return FreshnessResult(
                False, 'OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION', age, remaining)
        if source_time is None:
            return FreshnessResult(False, 'SOURCE_MISSING', age, remaining)
        if not math.isfinite(source_time) or source_time > now:
            return FreshnessResult(False, 'INVALID_SOURCE_TIME', age, remaining)
        age = now - source_time
        deadline = source_time + self.reuse_ttl_seconds
        if additional_deadline is not None:
            if not math.isfinite(additional_deadline):
                return FreshnessResult(False, 'INVALID_DEADLINE', age, remaining)
            deadline = min(deadline, additional_deadline)
        remaining = max(0., deadline - now)
        for condition, reason in (
                (not source_verified, 'SOURCE_UNVERIFIED'),
                (not calibration_matches, 'CALIBRATION_CHANGED'),
                (not generation_matches, 'SOURCE_GENERATION_CHANGED'),
                (source_invalidated, 'SOURCE_INVALIDATED'),
                (consumed, 'ALREADY_CONSUMED'),
                (not feedback_valid, 'FEEDBACK_STALE_OR_UNAVAILABLE'),
                (not stream_healthy, 'STREAM_UNAVAILABLE'),
                (now >= deadline, 'REFRESH_REQUIRED_EXPIRED')):
            if condition:
                return FreshnessResult(False, reason, age, remaining)
        return FreshnessResult(True, 'FRESH_FOR_REUSE_ONLY', age, remaining)

    def pool_deadline(self, *, source_time, created_time, backend_deadline=None):
        """Long inference never extends a pool beyond its original source expiry."""
        if (not all(math.isfinite(t) for t in (source_time, created_time)) or
                created_time < source_time):
            raise ValueError('INVALID_POOL_SOURCE_TIME')
        deadline = min(source_time + self.reuse_ttl_seconds,
                       created_time + self.reuse_ttl_seconds)
        if backend_deadline is not None:
            if not math.isfinite(backend_deadline):
                raise ValueError('INVALID_BACKEND_DEADLINE')
            deadline = min(deadline, backend_deadline)
        return deadline
