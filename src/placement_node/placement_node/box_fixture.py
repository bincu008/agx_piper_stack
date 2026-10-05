"""Deterministic complete-surface boxes for a calculation-only benchmark."""

import numpy as np


def box_surface(center, size=0.08, spacing=0.002):
    """Sample all six surfaces; the lowest face lies on z=0."""
    values = np.linspace(-size / 2, size / 2,
                         int(np.ceil(size / spacing)) + 1)
    aa, bb = np.meshgrid(values, values)
    faces = []
    for axis in range(3):
        for side in (-1, 1):
            points = np.empty((aa.size, 3))
            points[:, axis] = side * size / 2
            remaining = [i for i in range(3) if i != axis]
            points[:, remaining[0]] = aa.ravel()
            points[:, remaining[1]] = bb.ravel()
            faces.append(points)
    return (np.unique(np.vstack(faces), axis=0) + center).astype(np.float32)


def box_fixture(count=10, size=0.08, gap=0.04, radius=0.5,
                occupied_points=None):
    """Put separated boxes on the left (+Y), fully inside the radial bound."""
    if count < 1 or size <= 0 or gap <= 0:
        raise ValueError('positive box count, size and gap are required')
    pitch = size + gap
    centers = []
    for y in (0.295, 0.295 + pitch, 0.295 - pitch):
        for x in (2*pitch, pitch, 0, -pitch, -2*pitch,
                  3*pitch, -3*pitch):
            farthest = np.array([abs(x) + size/2, y + size/2])
            if np.linalg.norm(farthest) <= radius + 1e-9:
                center = np.array([x, y, size/2])
                if occupied_points is not None:
                    delta = np.abs(np.asarray(occupied_points) - center)
                    overlap = ((delta[:, 0] < size/2 + 0.005) &
                               (delta[:, 1] < size/2 + 0.005) &
                               (occupied_points[:, 2] > 0.01) &
                               (occupied_points[:, 2] < size + 0.01))
                    if np.any(overlap):
                        continue
                centers.append(center)
    if count > len(centers):
        raise ValueError(f'fixture fits only {len(centers)} separated boxes')
    return [box_surface(center, size) for center in centers[:count]]


def box_top(points, tolerance=0.0001):
    """Select the source cube top before applying its rigid transform."""
    # Original complete surfaces use a dense top; small IK tilt is allowed.
    points = np.asarray(points)
    return points[points[:, 2] >= points[:, 2].max() - tolerance]


def box_contact_seeds(points, approach_tilt_deg=0.0):
    """Known opposed wall centers, supplied to unchanged grasp validation."""
    lower, upper = points.min(0), points.max(0)
    center = (lower + upper) / 2
    center[2] = lower[2] + 0.65 * (upper[2] - lower[2])
    seeds = []
    for axis in (0, 1):
        for sign in (1, -1):
            left, right = center.copy(), center.copy()
            left[axis], right[axis] = upper[axis], lower[axis]
            if sign < 0:
                left, right = right, left
            tilt = np.deg2rad(float(approach_tilt_deg))
            approach = np.array([0., 0., -np.cos(tilt)])
            approach[1-axis] = np.sin(tilt)
            seeds.append((left, right, approach))
    return seeds


def tray_floor_mask(points):
    """Label the measured tray floor by distance to its dominant plane."""
    from .packing import _estimate_support_frame
    points = np.asarray(points)
    _, normal, _ = _estimate_support_frame(points, 60.0)
    projected = points @ normal
    level = np.median(projected)
    return np.abs(projected - level) <= 0.003


def replace_fixture_floor_noise(original, boxes, return_mask=False):
    """Remove duplicate noisy table samples under the known virtual supports."""
    original = np.asarray(original)
    replace = np.zeros(len(original), dtype=bool)
    for box in boxes:
        minimum, maximum = box.min(axis=0), box.max(axis=0)
        replace |= ((abs(original[:, 2]) <= 0.005) &
                    (original[:, 0] >= minimum[0] - 0.01) &
                    (original[:, 0] <= maximum[0] + 0.01) &
                    (original[:, 1] >= minimum[1] - 0.01) &
                    (original[:, 1] <= maximum[1] + 0.01))
    keep = ~replace
    return (original[keep], keep) if return_mask else original[keep]
