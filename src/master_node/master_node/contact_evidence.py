"""Bounded fresh-sample evidence for a settled gripping contact."""

from collections import deque
import math
from statistics import median
import time


MIN_CONTACT_WIDTH_M = 0.0001
MIN_CONTACT_FORCE_N = 0.2
HOLDING_FORCE_N = 0.1
CONTACT_LOSS_DELAY_SEC = 0.15


def contact_present(status, minimum_force=MIN_CONTACT_FORCE_N):
    """Accept thin rims, excluding fully closed jaws and absent force."""
    return (status is not None and math.isfinite(status.width) and
            math.isfinite(status.force) and status.width > MIN_CONTACT_WIDTH_M and
            abs(status.force) >= minimum_force)


class SettledContact:
    """Require stable aperture and sustained force, tolerating short force dips."""

    def __init__(self):
        """Keep at most 250 ms of distinct feedback samples."""
        self.last_sample = None
        self.started = None
        self.samples = 0
        self.width = None
        self.history = deque(maxlen=128)
        self.established = False
        self.last_feedback = None
        self.loss_since = None

    def confirmed(self, now=None):
        """Read one snapshot while the feedback callback updates the window."""
        history = tuple(self.history)
        if now is None:
            now = time.monotonic()
        return (len(history) >= 3 and 0 <= now-history[-1][0] <= .1 and
                history[-1][0]-history[0][0] >= .15 and
                median(force for _, force in history) >= MIN_CONTACT_FORCE_N)

    def holding(self, now=None):
        """Maintain an established grip without reacquiring its initial load."""
        return not self.holding_failure(now)

    def holding_failure(self, now=None):
        """Explain sustained loss independently of initial acquisition timing."""
        established, last_feedback, loss_since = (
            self.established, self.last_feedback, self.loss_since)
        if now is None:
            now = time.monotonic()
        if not established:
            return 'CONTACT_NOT_ESTABLISHED'
        if last_feedback is None or not 0 <= now-last_feedback <= .1:
            return 'CONTACT_FEEDBACK_STALE'
        if loss_since is not None and now-loss_since >= CONTACT_LOSS_DELAY_SEC:
            return 'CONTACT_BELOW_HOLDING_THRESHOLD_OR_JAWS_CLOSED'
        return ''

    def update(self, status, stamp, now):
        """Acquire stable contact and separately monitor its continued presence."""
        if math.isfinite(stamp) and 0 <= now-stamp <= .1:
            self.last_feedback = stamp
            if contact_present(status, HOLDING_FORCE_N):
                self.loss_since = None
            elif self.loss_since is None:
                self.loss_since = stamp
        if (not contact_present(status, MIN_CONTACT_FORCE_N / 2) or
                not math.isfinite(stamp) or not 0 <= now-stamp <= .1):
            self.started, self.samples, self.width = None, 0, None
            self.history.clear()
            return False
        if stamp != self.last_sample:
            self.last_sample = stamp
            if self.started is None or abs(status.width-self.width) > .0005:
                self.started, self.samples, self.width = stamp, 0, status.width
                self.history.clear()
            self.history.append((stamp, abs(status.force)))
            while self.history and stamp-self.history[0][0] > .25:
                self.history.popleft()
            self.samples = len(self.history)
        ready = self.confirmed(now)
        self.established = self.established or ready
        return ready
