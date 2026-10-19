"""Estimate a uniform optical scale from a segmented circular plate rim."""

import cv2
import numpy as np


def measure_plate(camera_xyz, mask, diameter_m, current_ratio):
    """Fit rim plane robustly, intersect silhouette rays, measure planar widths."""
    if diameter_m <= 0 or current_ratio <= 0:
        raise ValueError('Positive diameter and current ratio required')
    valid = np.isfinite(camera_xyz).all(axis=2) & (camera_xyz[:, :, 2] > .05)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('No segmented plate')
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if (contour[:, 0].min() <= 0 or contour[:, 1].min() <= 0 or
            contour[:, 0].max() >= mask.shape[1]-1 or
            contour[:, 1].max() >= mask.shape[0]-1):
        raise ValueError('Plate silhouette clipped by image boundary')
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    points = camera_xyz[(distances >= 3) & (distances <= 8) & valid]
    if len(points) < 100:
        raise ValueError('Insufficient valid rim samples')
    rng, best = np.random.default_rng(1234), None
    for _ in range(300):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1.e-10:
            continue
        normal /= length
        selected = np.abs((points-a) @ normal) < .0015
        if best is None or selected.sum() > best.sum():
            best = selected
    if best is None or best.mean() < .65:
        raise ValueError('Rim plane unreliable; do not change calibration')
    center = points[best].mean(axis=0)
    _, _, basis = np.linalg.svd(points[best]-center, full_matrices=False)
    normal = basis[2]
    rays = camera_xyz[contour[:, 1], contour[:, 0]]
    usable = np.isfinite(rays).all(axis=1) & (rays[:, 2] > .05)
    if usable.mean() < .9:
        raise ValueError('Insufficient valid silhouette rays')
    rays = rays[usable] / rays[usable, 2, None]
    denominator = rays @ normal
    if np.min(np.abs(denominator)) < .1:
        raise ValueError('Plate plane too oblique')
    rim = rays * ((center @ normal)/denominator)[:, None]
    xy = (rim-center) @ basis[:2].T
    angles = np.arange(0., 180., 15.)
    directions = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    widths = np.ptp(xy @ directions.T, axis=0)
    if np.ptp(widths)/np.median(widths) > .05:
        raise ValueError('Directional spread exceeds 5%; uniform scale not justified')
    raw_widths = widths * current_ratio
    axis_widths = {}
    for axis, name in enumerate(('x', 'y', 'z')):
        projected = basis[:2, axis]
        length = np.linalg.norm(projected)
        axis_widths[name] = (float(np.ptp(xy @ (projected/length)))
                             if length > .1 else None)
    return {
        'angles_degrees': angles.tolist(),
        'diameters_current_m': widths.tolist(),
        'camera_axis_projected_direction_diameters_m': axis_widths,
        'diameters_unscaled_m': raw_widths.tolist(),
        'perception_measurement_ratio': float(np.median(raw_widths)/diameter_m),
        'rim_plane_inlier_fraction': float(best.mean()),
        'camera_plane_basis': basis[:2].tolist(),
        'camera_plane_normal': normal.tolist(),
        'axis_calibration_independent': False,
        'method': 'rim-plane RANSAC and silhouette-ray planar Feret widths',
    }
