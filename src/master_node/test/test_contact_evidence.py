"""Regression for force spikes while closing and settled rim contact."""

from types import SimpleNamespace

from master_node.contact_evidence import SettledContact


def test_closing_spike_cannot_authorize_lift():
    gate = SettledContact()
    sample = SimpleNamespace(width=.01148, force=-1.)
    assert not gate.update(sample, 1., 1.)
    sample.force = .07
    assert not gate.update(sample, 1.04, 1.04)
    sample.width, sample.force = .00567, -1.026
    assert not gate.update(sample, 1.1, 1.1)
    assert not gate.update(sample, 1.2, 1.2)
    assert gate.update(sample, 1.3, 1.3)


def test_repeated_or_stale_sample_is_not_sustained_contact():
    gate = SettledContact()
    sample = SimpleNamespace(width=.00567, force=-1.)
    for now in (1., 1.02, 1.04, 1.08):
        assert not gate.update(sample, 1., now)
    assert not gate.update(sample, 1., 1.3)


def test_aperture_movement_resets_contact_window():
    gate = SettledContact()
    sample = SimpleNamespace(width=.011, force=-1.)
    assert not gate.update(sample, 1., 1.)
    sample.width = .008
    assert not gate.update(sample, 1.2, 1.2)
    sample.width = .0056
    assert not gate.update(sample, 1.4, 1.4)


def test_thin_rim_with_signed_noisy_force_confirms_without_opening_floor_rejection():
    """Submillimetre rim contact tolerates short dips, not absent load."""
    gate = SettledContact()
    sample = SimpleNamespace(width=.00056, force=-.24)
    for i, force in enumerate((-.24, -.26, -.121, -.28, -.24, -.26,
                               -.389, -.24, -.23, -.25, -.12)):
        sample.force = force
        ready = gate.update(sample, 1.+i*.02, 1.+i*.02)
    assert ready
    assert not gate.confirmed(1.4)  # Feedback has stopped.
    for i in range(1, 20):
        sample.force = -.15
        ready = gate.update(sample, 1.2+i*.02, 1.2+i*.02)
    assert not ready  # Sustained force below the threshold is still rejected.


def test_fully_closed_or_unloaded_jaws_do_not_confirm_contact():
    """A force spike at the closed stop or an open unloaded jaw is insufficient."""
    for width, force in ((0., -1.), (.00005, -.3), (.00056, 0.)):
        gate = SettledContact()
        sample = SimpleNamespace(width=width, force=force)
        for i in range(20):
            assert not gate.update(sample, 1.+i*.02, 1.+i*.02)


def test_live_callback_continues_evidence_without_changing_raw_feedback():
    """Held checks retain the rolling evidence after close confirmation."""
    import time
    from master_node.piper_master_control import PiperMasterControl
    gate = SettledContact()
    now = time.monotonic()
    for i in range(11):
        stamp = now-.2+i*.02
        gate.update(SimpleNamespace(width=.00056, force=-.25), stamp, stamp)
    node = SimpleNamespace(gripper_contact_evidence=gate)
    raw = SimpleNamespace(width=.00056, force=-.121)
    PiperMasterControl.gripper_callback(node, raw)
    assert node.current_gripper_status is raw
    assert raw.force == -.121 and raw.width == .00056
    assert node.gripper_contact_evidence.confirmed()
    raw.force = 0.
    PiperMasterControl.gripper_callback(node, raw)
    assert not node.gripper_contact_evidence.confirmed()


def test_holding_does_not_repeat_close_acquisition_after_load_settles():
    """Established thin-rim contact survives lower holding load without reclose."""
    gate = SettledContact()
    sample = SimpleNamespace(width=.00056, force=-.25)
    for i in range(11):
        gate.update(sample, 1.+i*.02, 1.+i*.02)
    assert gate.established
    for i in range(1, 31):
        sample.width, sample.force = .00035, -.15
        stamp = 1.2+i*.02
        gate.update(sample, stamp, stamp)
        assert gate.holding(stamp)
    assert not gate.confirmed(1.8)  # Initial close load is not required again.
    sample.force = 0.
    gate.update(sample, 1.82, 1.82)
    assert gate.holding(1.82)  # One brief dropout is tolerated.
    gate.update(sample, 1.90, 1.90)
    assert gate.holding(1.90)
    gate.update(sample, 2.0, 2.0)
    assert not gate.holding(2.0)
    assert 'THRESHOLD' in gate.holding_failure(2.0)
    assert 'STALE' in gate.holding_failure(2.3)


def test_unconfirmed_low_force_cannot_authorize_a_lift():
    """Holding hysteresis is available only after successful initial closing."""
    gate = SettledContact()
    sample = SimpleNamespace(width=.00056, force=-.15)
    for i in range(31):
        gate.update(sample, 1.+i*.02, 1.+i*.02)
        assert not gate.holding(1.+i*.02)
