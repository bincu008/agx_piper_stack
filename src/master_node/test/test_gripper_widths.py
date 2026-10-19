"""Check shared jaw targets, including physical aperture limits."""

import pytest

from master_node.gripper_widths import grasp_gripper_widths


@pytest.mark.parametrize('width,opening,closing', [
    (.05, .07, .03), (.01, .03, 0.), (.09, .10, .06)])
def test_operator_gripper_targets(width, opening, closing):
    actual = grasp_gripper_widths(width)
    assert actual == pytest.approx((opening, closing))


@pytest.mark.parametrize('width', [-.01, .11, float('nan'), float('inf')])
def test_invalid_grasp_width_rejected(width):
    with pytest.raises(ValueError, match='INVALID_GRASP_WIDTH'):
        grasp_gripper_widths(width)
