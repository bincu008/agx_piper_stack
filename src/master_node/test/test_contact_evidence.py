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
