"""Compare accelerated candidate diversity selection against exhaustive ranking."""

import math

import numpy as np
import pytest

from placement_node.inference import PlacementCandidate
from placement_node.packing import _select_diverse_candidates


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
    actual = _select_diverse_candidates(candidates, limit)
    expected = reference(candidates, limit)
    assert [id(c) for c in actual] == [id(c) for c in expected]


def test_empty_candidate_selection():
    assert _select_diverse_candidates([], 128) == []
