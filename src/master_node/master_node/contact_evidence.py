"""Bounded fresh-sample evidence for a settled gripping contact."""

import math


class SettledContact:
    """Require 150 ms of contact and at least three distinct feedback samples."""

    def __init__(self):
        self.last_sample = None
        self.started = None
        self.samples = 0
        self.width = None

    def update(self, status, stamp, now):
        """Reset evidence on contact loss or continued aperture movement."""
        if (status is None or not math.isfinite(status.width) or
                not math.isfinite(status.force) or status.width <= .001 or
                abs(status.force) < .2 or now-stamp > .1):
            self.started, self.samples, self.width = None, 0, None
            return False
        if stamp != self.last_sample:
            self.last_sample = stamp
            if self.started is None or abs(status.width-self.width) > .0005:
                self.started, self.samples, self.width = stamp, 1, status.width
            else:
                self.samples += 1
        return (self.started is not None and self.samples >= 3 and
                stamp-self.started >= .15)
