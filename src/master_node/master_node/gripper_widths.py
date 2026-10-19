"""Shared opening and closing targets for measured grasp widths, in metres."""

import math


def grasp_gripper_widths(object_width):
    """Apply operator clearance and closing travel within the 10 cm aperture."""
    if not math.isfinite(object_width) or not 0. <= object_width <= .10:
        raise ValueError('INVALID_GRASP_WIDTH')
    opening = min(.10, object_width+.02)
    return opening, max(0., opening-.04)
