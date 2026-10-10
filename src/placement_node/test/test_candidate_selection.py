"""Compare accelerated candidate diversity selection against exhaustive ranking."""

import math

import numpy as np
import pytest

from placement_node.inference import PlacementCandidate
from placement_node.packing import _select_diverse_candidates, _select_ranked_candidates


def reference(candidates, limit):
    selected, deferred = [], []
    for candidate in candidates:
        if any(np.linalg.norm(candidate.translation[:2]-other.translation[:2]) < 0.025
               for other in selected):
            deferred.append(candidate)
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for candidate in deferred:
            if any(np.linalg.norm(candidate.translation[:2]-other.translation[:2]) < 0.012
                   and abs(math.atan2(candidate.rotation[1, 0], candidate.rotation[0, 0]) -
                           math.atan2(other.rotation[1, 0], other.rotation[0, 0])) <
                   math.radians(10) for other in selected):
                continue
            selected.append(candidate)
            if len(selected) >= limit:
                break
    return selected


@pytest.mark.parametrize('limit', [1, 8, 128, 512])
def test_spatial_index_retains_exhaustive_candidate_order(limit):
    rng = np.random.default_rng(723)
    candidates = []
    for i in range(2000):
        yaw = float(rng.uniform(-math.pi, math.pi))
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                             [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]], np.float32)
        translation = np.r_[rng.uniform(-0.15, 0.15, 2), 0].astype(np.float32)
        candidates.append(PlacementCandidate(rotation, translation, -float(i)))
    actual = _select_ranked_candidates(candidates, limit)
    expected = reference(candidates, limit)
    assert [id(c) for c in actual] == [id(c) for c in expected]


def test_empty_candidate_selection():
    assert _select_diverse_candidates([], 128) == []


def test_wider_budget_preserves_lower_ranked_wrist_yaws():
    # All poses fit the same footprint; a fixed grasp can make only a lower
    # ranked object yaw reachable. Ranking must survive the wider shortlist.
    candidates = []
    for index in range(64):
        yaw = index * math.pi / 32
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                             [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        candidates.append(PlacementCandidate(
            rotation, np.array([0.3, 0.1, 0.02]), -float(index)))
    short = _select_ranked_candidates(candidates, 8)
    wide = _select_ranked_candidates(candidates, 128)
    assert len(wide) > len(short)
    assert any(item is candidates[57] for item in wide)
    assert not any(item is candidates[57] for item in short)
    assert [item.score for item in wide] == sorted(
        [item.score for item in wide], reverse=True)


def test_coarse_selection_spreads_positions_and_keeps_score_order():
    candidates = []
    for yaw in (0, math.pi / 2, math.pi):
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                             [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        for x in np.linspace(0, 0.3, 31):
            for y in np.linspace(0, 0.2, 21):
                candidates.append(PlacementCandidate(
                    rotation, np.array([x, y, 0.02]), -float(len(candidates))))
    selected = _select_diverse_candidates(candidates, 16)
    assert len(selected) == 16
    positions = np.array([item.translation[:2] for item in selected])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    assert distances[np.triu_indices(16, 1)].min() >= 0.04 - 1e-9
    assert np.ptp(positions, axis=0).min() >= 0.19
    assert len({round(math.atan2(c.rotation[1, 0], c.rotation[0, 0]), 3)
                for c in selected}) > 1
    assert selected[0] is candidates[0]
    assert [c.score for c in selected] == sorted([c.score for c in selected], reverse=True)


def test_coarse_selection_does_not_fill_with_near_duplicates():
    candidates = [PlacementCandidate(np.eye(3), np.array([x, 0, 0]), -x)
                  for x in (0, 0.01, 0.02)]
    assert _select_diverse_candidates(candidates, 16) == [candidates[0]]


def test_ordered_shortlist_keeps_wrist_yaw_alternatives_at_coarse_locations():
    candidates = []
    for x in (0.3, 0.5):
        for angle in np.arange(8) * math.pi / 4:
            rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                                 [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
            candidates.append(PlacementCandidate(
                rotation, np.array([x, 0.1, 0.02]), -float(len(candidates))))
    selected = _select_diverse_candidates(candidates, 8, preserve_yaw=True)
    assert len(selected) == 8
    assert {item.translation[0] for item in selected} == {0.3, 0.5}
    for x in (0.3, 0.5):
        orientations = [item for item in selected if item.translation[0] == x]
        assert len(orientations) >= 2
    assert [item.score for item in selected] == sorted(
        [item.score for item in selected], reverse=True)


@pytest.mark.parametrize('threshold,covered,accepted', [
    (.8, 8, True), (.8, 7, False), (.6, 6, True), (.6, 5, False)])
def test_combined_surface_coverage_is_footprint_fraction(threshold, covered, accepted):
    from placement_node.packing import _heightmap_trial_block
    offsets = np.column_stack((np.arange(10), np.zeros(10, dtype=int)))
    pile = np.full((1, 10), .08, dtype=np.float32)
    top = np.full((1, 10), np.nan, dtype=np.float32)
    top[:, :covered] = .08
    args = (np.array([[0, 0]]), offsets, np.zeros(10), pile,
            np.zeros((1, 10)), top, np.ones((1, 10)),
            .001, .003, .006, 'stack')
    *_, support, valid = _heightmap_trial_block(*args, stack_minimum_coverage=threshold)
    assert bool(valid[0]) == accepted
    assert support.sum() == covered
    assert not _heightmap_trial_block(*args)[-1][0]


def test_partial_support_does_not_ignore_higher_obstacle():
    from placement_node.packing import _heightmap_trial_block
    offsets = np.column_stack((np.arange(10), np.zeros(10, dtype=int)))
    pile = np.full((1, 10), .08, dtype=np.float32)
    pile[0, -1] = .10
    top = np.full((1, 10), .08, dtype=np.float32)
    args = (np.array([[0, 0]]), offsets, np.zeros(10), pile,
            np.zeros((1, 10)), top, np.ones((1, 10)),
            .001, .003, .006, 'stack')
    assert not _heightmap_trial_block(*args, stack_minimum_coverage=.8)[-1][0]


def test_coverage_measures_projected_area_not_contact_fraction():
    from placement_node.packing import _heightmap_trial_block
    offsets = np.column_stack((np.arange(10), np.zeros(10, dtype=int)))
    top = np.full((1, 10), .08, dtype=np.float32)
    bottom = np.zeros(10)
    bottom[-2:] = .02  # A raised rim/handle shadow need not touch the box.
    args = (np.array([[0, 0]]), offsets, bottom, top,
            np.zeros((1, 10)), top, np.ones((1, 10)),
            .001, .003, .006, 'stack')
    *_, support, valid = _heightmap_trial_block(*args, stack_minimum_coverage=1.0)
    assert valid[0]
    assert support.sum() == 8
