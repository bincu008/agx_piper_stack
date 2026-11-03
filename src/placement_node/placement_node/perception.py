"""ROS-independent grounding and registered-cloud helpers."""

import re

import cv2
import numpy as np
from scipy.spatial import cKDTree


def exposed_stack_support_labels(points, labels, eligible, resolution=0.005,
                                 height_tolerance=0.002):
    """Label only exposed tops of eligible geometry; keep its body blocking."""
    points = np.asarray(points).reshape(-1, 3)
    result = np.asarray(labels, dtype=np.uint8).copy()
    eligible = np.asarray(eligible, dtype=bool)
    if len(result) != len(points) or eligible.shape != result.shape:
        raise ValueError('support labels and eligibility must match the scene')
    if resolution <= 0 or height_tolerance < 0:
        raise ValueError('invalid support grid settings')
    finite = np.isfinite(points).all(axis=1)
    # Former tops can become occluded after later virtual placements.
    result[result == 2] = 3
    if not finite.any():
        return result
    cells = np.floor(points[finite, :2] / resolution).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    heights = np.full(inverse.max() + 1, -np.inf)
    np.maximum.at(heights, inverse, points[finite, 2])
    top = np.zeros(len(points), dtype=bool)
    top[finite] = points[finite, 2] >= heights[inverse] - height_tolerance
    result[eligible & top] = 2
    return result


def active_source_support_labels(labels, object_masks, active_index):
    """Keep other/placed objects protected without mislabeling active-mask floor."""
    result = np.asarray(labels, dtype=np.uint8).copy()
    active = np.asarray(object_masks[active_index], dtype=bool)
    other_sources = np.zeros_like(active)
    for index, mask in enumerate(object_masks):
        if index != active_index:
            other_sources |= mask
    active_only = (active & ~other_sources).reshape(-1)
    result[:len(active_only)][active_only] = 0
    return result


def _resample_closed_contour(contour: np.ndarray, spacing: float) -> np.ndarray:
    """Return approximately uniform samples along a closed 2D polygon."""
    samples = []
    for index, start in enumerate(contour):
        end = contour[(index + 1) % len(contour)]
        length = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(length / spacing)))
        for fraction in np.linspace(0.0, 1.0, count, endpoint=False):
            samples.append(start + fraction * (end - start))
    return np.asarray(samples, dtype=np.float32)


def _fit_circular_footprint(
        points: np.ndarray, spacing: float,
        rng: np.random.Generator) -> tuple[np.ndarray, float] | None:
    """Robustly fit a circular body footprint while ignoring appendages."""
    if len(points) < 30:
        return None
    best = None
    tolerance = max(0.0035, spacing)
    maximum_radius = min(
        0.12, max(0.025, 0.65 * float(np.ptp(points, axis=0).max())))
    for _ in range(256):
        sample = points[rng.choice(len(points), 3, replace=False)]
        matrix = 2.0 * np.asarray([
            sample[1] - sample[0], sample[2] - sample[0]])
        right = np.asarray([
            np.dot(sample[1], sample[1]) - np.dot(sample[0], sample[0]),
            np.dot(sample[2], sample[2]) - np.dot(sample[0], sample[0])])
        if abs(float(np.linalg.det(matrix))) < 1.0e-8:
            continue
        center = np.linalg.solve(matrix, right)
        radius = float(np.linalg.norm(sample[0] - center))
        if radius < 0.008 or radius > maximum_radius:
            continue
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        inliers = residual <= tolerance
        if np.count_nonzero(inliers) < 24:
            continue
        angles = np.sort(np.arctan2(
            points[inliers, 1] - center[1],
            points[inliers, 0] - center[0]))
        gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
        coverage = 2.0 * np.pi - float(np.max(gaps))
        if coverage < np.deg2rad(100.0):
            continue
        score = int(np.count_nonzero(inliers)) - 20.0 * float(
            np.mean(residual[inliers]))
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None:
        return None
    fitted = points[best[1]]
    matrix = np.column_stack((2.0 * fitted, np.ones(len(fitted))))
    right = np.sum(fitted * fitted, axis=1)
    solution = np.linalg.lstsq(matrix, right, rcond=None)[0]
    center = solution[:2]
    radius_squared = float(solution[2] + np.dot(center, center))
    if radius_squared <= 0.0:
        return None
    fitted_radius = float(np.sqrt(radius_squared))
    fitted_residual = np.abs(
        np.linalg.norm(fitted - center, axis=1) - fitted_radius)
    all_residual = np.abs(
        np.linalg.norm(points - center, axis=1) - fitted_radius)
    consistent = all_residual <= tolerance
    consistent_fraction = float(np.count_nonzero(consistent)) / len(points)
    consistent_angles = np.sort(np.arctan2(
        points[consistent, 1] - center[1],
        points[consistent, 0] - center[0]))
    if len(consistent_angles) < 3:
        return None
    angle_gaps = np.diff(np.r_[
        consistent_angles, consistent_angles[0] + 2.0 * np.pi])
    angular_coverage = 2.0 * np.pi - float(np.max(angle_gaps))
    # A rectangular face can approximate a short circular arc. Require most
    # persistent wall samples and at least 150 degrees of the circumference to
    # agree before selecting the cylindrical model.
    if (fitted_radius > maximum_radius or
            float(np.mean(fitted_residual)) > tolerance or
            consistent_fraction < 0.75 or
            angular_coverage < np.deg2rad(150.0)):
        return None
    radius = fitted_radius + spacing * 0.5
    count = max(24, int(np.ceil(2.0 * np.pi * radius / spacing)))
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    contour = center + radius * np.column_stack((
        np.cos(angles), np.sin(angles)))
    return contour.astype(np.float32), radius


def construct_support_bottom(
        object_points: np.ndarray, scene_points: np.ndarray,
        context_margin_m: float = 0.04,
        plane_distance_m: float = 0.004,
        maximum_plane_tilt_deg: float = 15.0,
        bottom_band_m: float = 0.025,
        grid_spacing_m: float = 0.004,
        maximum_bottom_points: int = 1500,
        seed: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Complete a supported object's unseen bottom with a gravity-projected cap at measured support height.

    Both input clouds must already be in ``base_link``. The scene is used only
    to fit the source support plane in an XY annulus around the segmented
    object. Only object samples near that plane define the footprint, so raised
    geometry such as a cup handle is not projected into the bottom.
    """
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    scene_points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
    object_points = object_points[np.isfinite(object_points).all(axis=1)]
    scene_points = scene_points[np.isfinite(scene_points).all(axis=1)]
    if len(object_points) < 80:
        raise ValueError("bottom completion needs at least 80 object points")

    xy_min = object_points[:, :2].min(axis=0)
    xy_max = object_points[:, :2].max(axis=0)
    margin = max(float(context_margin_m), 0.005)
    outer = (
        (scene_points[:, 0] >= xy_min[0] - margin) &
        (scene_points[:, 0] <= xy_max[0] + margin) &
        (scene_points[:, 1] >= xy_min[1] - margin) &
        (scene_points[:, 1] <= xy_max[1] + margin))
    guard = min(0.008, margin * 0.25)
    inside_guarded_object = (
        (scene_points[:, 0] >= xy_min[0] - guard) &
        (scene_points[:, 0] <= xy_max[0] + guard) &
        (scene_points[:, 1] >= xy_min[1] - guard) &
        (scene_points[:, 1] <= xy_max[1] + guard))
    low_reference = float(np.percentile(object_points[:, 2], 3.0))
    height_window = (
        (scene_points[:, 2] >= low_reference - 0.05) &
        (scene_points[:, 2] <= low_reference + 0.04))
    support_candidates = scene_points[
        outer & ~inside_guarded_object & height_window]
    if len(support_candidates) < 60:
        raise ValueError(
            f"support annulus has only {len(support_candidates)} points")

    rng = np.random.default_rng(seed)
    fit_points = support_candidates
    if len(fit_points) > 5000:
        fit_points = fit_points[rng.choice(len(fit_points), 5000, replace=False)]
    cosine_limit = np.cos(np.deg2rad(maximum_plane_tilt_deg))
    best_inliers = None
    best_count = 0
    for _ in range(96):
        sample = fit_points[rng.choice(len(fit_points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-8:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < cosine_limit:
            continue
        offset = -float(np.dot(normal, sample[0]))
        inliers = np.abs(fit_points @ normal + offset) <= plane_distance_m
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 40:
        raise ValueError("no gravity-consistent source support plane found")

    plane_points = fit_points[best_inliers]
    centroid = plane_points.mean(axis=0)
    covariance = (plane_points - centroid).T @ (plane_points - centroid)
    _, _, vh = np.linalg.svd(covariance, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    if normal[2] < cosine_limit:
        raise ValueError("refined source support plane exceeds tilt limit")
    offset = -float(np.dot(normal, centroid))

    # Use the fitted surface only to estimate support height. Upright object
    # footprints are gravity projections: retain measured XY, never shear the
    # object's silhouette along a noisy tilted annulus-plane normal.
    measured_support_normal = normal.copy()
    source_xy = np.median(object_points[:, :2], axis=0)
    support_z = -(float(normal[:2] @ source_xy)+offset)/float(normal[2])
    support_z = min(support_z, low_reference)
    centroid = np.r_[source_xy, support_z]
    normal = np.array([0., 0., 1.])
    offset = -float(support_z)

    heights = object_points @ normal + offset
    # A surrounding table is not evidence for the underside of an elevated
    # object. Completing down to it would attach its stand or empty air.
    elevated_planar = False
    if float(np.percentile(heights, 3.0)) > float(bottom_band_m):
        # A thin elevated body has its own underside; the distant table must
        # never become its base. Infer only a bounded thin-body lower plane.
        thickness = float(np.percentile(object_points[:, 2], 97)-
                          np.percentile(object_points[:, 2], 3))
        extent = np.ptp(object_points[:, :2], axis=0)
        if thickness > .035 or np.min(extent) < max(.04, 3*thickness):
            raise ValueError("source support plane is below the observed object; bottom completion refused")
        normal = np.array([0., 0., 1.])
        centroid = np.r_[np.median(object_points[:, :2], axis=0),
                         np.percentile(object_points[:, 2], 3)]
        offset = -float(centroid[2])
        heights = object_points[:, 2]-centroid[2]
        elevated_planar = True
    footprint_points = object_points[
        (heights >= -plane_distance_m * 2.0) &
        (heights <= float(bottom_band_m))]
    if len(footprint_points) < 20:
        cutoff = float(np.percentile(heights, 8.0))
        footprint_points = object_points[heights <= cutoff]
    if len(footprint_points) < 12:
        raise ValueError("insufficient near-support object points for footprint")

    axis_u = np.array(
        [1.0, 0.0, -normal[0] / normal[2]], dtype=np.float64)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    spacing = max(float(grid_spacing_m), 0.001)
    projected_all = object_points - np.outer(heights, normal)
    projected_uv = np.column_stack((
        (projected_all - centroid) @ axis_u,
        (projected_all - centroid) @ axis_v)).astype(np.float32)

    # A rigid body's side wall persists through multiple height slices at the
    # same projected location. Sparse appendages and depth outliers generally
    # do not, so vertical persistence extracts the principal body without an
    # object-class rule.
    usable = heights >= max(0.003, float(np.percentile(heights, 3.0)))
    usable &= heights <= float(np.percentile(heights, 92.0))
    side_uv = projected_uv[usable]
    side_heights = heights[usable]
    cells = np.floor(side_uv / spacing).astype(np.int32)
    height_cells = np.floor(
        side_heights / max(spacing * 1.5, 0.006)).astype(np.int32)
    cell_height = np.column_stack((cells, height_cells))
    unique_cell_height = np.unique(cell_height, axis=0)
    unique_cells, persistence = np.unique(
        unique_cell_height[:, :2], axis=0, return_counts=True)
    persistent_cells = unique_cells[persistence >= 3]
    persistent_keys = {tuple(cell) for cell in persistent_cells}
    persistent_mask = np.asarray(
        [tuple(cell) in persistent_keys for cell in cells], dtype=bool)
    body_uv = side_uv[persistent_mask]
    if len(body_uv) < 30:
        body_uv = side_uv

    circular = None if elevated_planar else _fit_circular_footprint(body_uv, spacing, rng)
    if not elevated_planar and circular is None and len(side_uv) >= 30:
        # A tapered wall spans several radii, so its dense interior samples
        # need not lie on one circle. Test the observed outer contour instead;
        # the same circle-consistency and angular-coverage gates still apply.
        outer_hull = cv2.convexHull(side_uv.astype(np.float32)).reshape(-1, 2)
        if len(outer_hull) >= 3:
            outer_contour = _resample_closed_contour(outer_hull, spacing)
            circular = _fit_circular_footprint(outer_contour, spacing, rng)
    footprint_model = "circle" if circular is not None else "oriented_box"
    if elevated_planar:
        hull = cv2.convexHull(projected_uv).reshape(-1, 2)
        contour_uv = _resample_closed_contour(hull, spacing)
        fitted_radius = 0.0
        footprint_model = "elevated_planar_hull"
    elif circular is not None:
        contour_uv, fitted_radius = circular
    else:
        # Conservative fallback for boxes and irregular objects. An oriented
        # rectangle encloses all mid-height side-wall projections, including a
        # partially observed rear face, instead of treating it as free space.
        rectangle = cv2.minAreaRect(side_uv.reshape(-1, 1, 2))
        hull = cv2.boxPoints(rectangle).astype(np.float32)
        hull_center = hull.mean(axis=0)
        directions = hull - hull_center
        lengths = np.linalg.norm(directions, axis=1, keepdims=True)
        hull = hull + directions / np.maximum(lengths, 1.0e-6) * spacing * 0.5
        contour_uv = _resample_closed_contour(hull, spacing)
        fitted_radius = 0.0

    hull = cv2.convexHull(contour_uv.reshape(-1, 1, 2)).reshape(-1, 2)
    if len(hull) < 3 or abs(float(cv2.contourArea(hull))) < 1.0e-5:
        raise ValueError("estimated bottom footprint is degenerate")

    if len(contour_uv) > maximum_bottom_points:
        indices = np.linspace(
            0, len(contour_uv) - 1, maximum_bottom_points, dtype=np.int32)
        contour_uv = contour_uv[indices]
    bottom_points = (
        centroid + np.outer(contour_uv[:, 0], axis_u) +
        np.outer(contour_uv[:, 1], axis_v)).astype(np.float32)

    # The contour is the only artificial support geometry exposed to RViz.
    # A filled base/top pair is retained separately as a conservative packing
    # envelope, preventing a box edge from entering unseen rear geometry.
    uv_min = hull.min(axis=0)
    uv_max = hull.max(axis=0)
    grid_u = np.arange(uv_min[0], uv_max[0] + spacing * 0.5, spacing)
    grid_v = np.arange(uv_min[1], uv_max[1] + spacing * 0.5, spacing)
    envelope_uv = []
    contour = hull.reshape(-1, 1, 2)
    for coordinate_u in grid_u:
        for coordinate_v in grid_v:
            if cv2.pointPolygonTest(
                    contour, (float(coordinate_u), float(coordinate_v)),
                    False) >= 0:
                envelope_uv.append((coordinate_u, coordinate_v))
    envelope_uv = np.asarray(envelope_uv, dtype=np.float32)
    if len(envelope_uv) == 0:
        raise ValueError("collision envelope rasterization produced no points")
    envelope_base = (
        centroid + np.outer(envelope_uv[:, 0], axis_u) +
        np.outer(envelope_uv[:, 1], axis_v))
    envelope_height = max(spacing, float(np.percentile(heights, 98.0)))
    envelope_top = envelope_base + envelope_height * normal
    collision_envelope = np.vstack((
        envelope_base, envelope_top)).astype(np.float32)
    completed = np.vstack((object_points, bottom_points)).astype(np.float32)
    diagnostics = {
        "support_points": int(best_count),
        "footprint_points": int(len(footprint_points)),
        "bottom_points": int(len(bottom_points)),
        "collision_envelope_points": int(len(collision_envelope)),
        "plane_normal": normal.astype(np.float32),
        "measured_support_normal": measured_support_normal.astype(np.float32),
        "plane_offset": float(offset),
        "footprint_area_m2": abs(float(cv2.contourArea(hull))),
        "footprint_model": footprint_model,
        "fitted_radius_m": float(fitted_radius),
    }
    return completed, bottom_points, collision_envelope, diagnostics


def estimate_support_translation(
        transformed_bottom: np.ndarray, destination_points: np.ndarray,
        grid_size_m: float = 0.004, lookup_radius_m: float = 0.015,
        support_quantile: float = 0.95, clearance_m: float = 0.002,
        minimum_matches: int = 10) -> tuple[float, int]:
    """Return the gravity-axis shift aligning a bottom cap to a surface.

    A compact XY height map avoids a quadratic cloud-to-cloud search. The
    quantile is a robust approximation of the non-penetrating maximum: it
    ignores isolated depth spikes while lifting the cap above almost all of
    the observed destination surface beneath its footprint.
    """
    bottom = np.asarray(transformed_bottom, dtype=np.float32).reshape(-1, 3)
    destination = np.asarray(
        destination_points, dtype=np.float32).reshape(-1, 3)
    bottom = bottom[np.isfinite(bottom).all(axis=1)]
    destination = destination[np.isfinite(destination).all(axis=1)]
    if len(bottom) < minimum_matches or len(destination) < minimum_matches:
        raise ValueError("insufficient bottom or destination support points")
    grid_size = max(float(grid_size_m), 0.001)
    radius = max(float(lookup_radius_m), grid_size)
    quantile = float(np.clip(support_quantile, 0.5, 1.0))

    cells = np.floor(destination[:, :2] / grid_size).astype(np.int64)
    # Keep the highest observed surface point per XY cell. This is the safe
    # support height when the plate has thickness or multiple returns.
    height_map = {}
    for cell, height in zip(cells, destination[:, 2]):
        key = (int(cell[0]), int(cell[1]))
        previous = height_map.get(key)
        if previous is None or height > previous:
            height_map[key] = float(height)

    radius_cells = int(np.ceil(radius / grid_size))
    maximum_distance_squared = radius * radius
    required_shifts = []
    for point in bottom:
        center = np.floor(point[:2] / grid_size).astype(np.int64)
        best_distance_squared = maximum_distance_squared
        best_height = None
        for offset_x in range(-radius_cells, radius_cells + 1):
            for offset_y in range(-radius_cells, radius_cells + 1):
                key = (int(center[0] + offset_x),
                       int(center[1] + offset_y))
                height = height_map.get(key)
                if height is None:
                    continue
                cell_xy = (np.asarray(key, dtype=np.float32) + 0.5) * grid_size
                distance_squared = float(np.sum((cell_xy - point[:2]) ** 2))
                if distance_squared <= best_distance_squared:
                    best_distance_squared = distance_squared
                    best_height = height
        if best_height is not None:
            required_shifts.append(best_height - float(point[2]))
    if len(required_shifts) < minimum_matches:
        raise ValueError(
            f"only {len(required_shifts)} bottom points overlap destination")
    shift = float(np.quantile(required_shifts, quantile) + clearance_m)
    return shift, len(required_shifts)


def parse_grounding_box(reply: str, width: int, height: int) -> tuple[int, int, int, int]:
    """Parse RoboBrain's normalized 0..1000 box into image pixels."""
    match = re.fullmatch(
        r"\s*\[\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*,\s*"
        r"([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\]\s*", reply)
    if not match:
        raise ValueError(f"expected exactly one integer bounding box: {reply!r}")
    values = [int(item) for item in match.groups()]
    if (width <= 0 or height <= 0 or any(item > 1000 for item in values)
            or values[0] >= values[2] or values[1] >= values[3]):
        raise ValueError(f"invalid normalized bounding box: {reply!r}")
    x1 = int(round(values[0] * width / 1000.0))
    y1 = int(round(values[1] * height / 1000.0))
    x2 = int(round(values[2] * width / 1000.0))
    y2 = int(round(values[3] * height / 1000.0))
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(1, min(width, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(1, min(height, y2))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError(f"grounding box is empty after conversion: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def extract_destination_cloud(
        organized_xyz: np.ndarray, image_shape: tuple[int, int],
        bbox: tuple[int, int, int, int], neighbor_distance_m: float,
        minimum_points: int, coordinates_are_base_link: bool = False
        ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop a grounded ROI and keep its dominant foreground depth component."""
    if organized_xyz.ndim != 3 or organized_xyz.shape[-1] != 3:
        raise ValueError("registered point cloud must be organized as H x W x 3")
    image_height, image_width = image_shape
    cloud_height, cloud_width = organized_xyz.shape[:2]
    x1, y1, x2, y2 = bbox
    cloud_box = (
        int(np.floor(x1 * cloud_width / image_width)),
        int(np.floor(y1 * cloud_height / image_height)),
        int(np.ceil(x2 * cloud_width / image_width)),
        int(np.ceil(y2 * cloud_height / image_height)),
    )
    cx1, cy1, cx2, cy2 = cloud_box
    cx1, cx2 = max(0, cx1), min(cloud_width, cx2)
    cy1, cy2 = max(0, cy1), min(cloud_height, cy2)
    roi = organized_xyz[cy1:cy2, cx1:cx2]
    valid = np.isfinite(roi).all(axis=2)
    if not coordinates_are_base_link:
        valid &= (roi[:, :, 2] > 0.05) & (roi[:, :, 2] < 2.0)
    if np.count_nonzero(valid) < minimum_points:
        raise ValueError(
            f"grounded ROI has only {np.count_nonzero(valid)} valid depth points; "
            f"need {minimum_points}")

    # Seed at the valid pixel nearest the grounding-box center. Grow over the
    # organized 3-D surface using only short neighbor-to-neighbor steps. Unlike
    # a fixed depth band, this follows a curved/slanted plate whose total depth
    # range may be large while stopping at its silhouette discontinuity.
    valid_y, valid_x = np.nonzero(valid)
    center_x = (roi.shape[1] - 1) * 0.5
    center_y = (roi.shape[0] - 1) * 0.5
    seed_index = np.argmin((valid_x - center_x) ** 2 + (valid_y - center_y) ** 2)
    seed = (int(valid_y[seed_index]), int(valid_x[seed_index]))
    foreground = np.zeros(valid.shape, dtype=np.bool_)
    foreground[seed] = True
    queue = [seed]
    cursor = 0
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1))
    threshold_squared = float(neighbor_distance_m) ** 2
    while cursor < len(queue):
        row, column = queue[cursor]
        cursor += 1
        point = roi[row, column]
        for row_delta, column_delta in neighbors:
            next_row = row + row_delta
            next_column = column + column_delta
            if (next_row < 0 or next_row >= roi.shape[0] or
                    next_column < 0 or next_column >= roi.shape[1] or
                    foreground[next_row, next_column] or
                    not valid[next_row, next_column]):
                continue
            difference = roi[next_row, next_column] - point
            if float(np.dot(difference, difference)) <= threshold_squared:
                foreground[next_row, next_column] = True
                queue.append((next_row, next_column))
    points = roi[foreground]
    if len(points) < minimum_points:
        raise ValueError(
            f"foreground component has only {len(points)} points; need {minimum_points}")
    return points.astype(np.float32), foreground, (cx1, cy1, cx2, cy2)


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Return ROS-order x,y,z,w quaternion for a 3x3 rotation matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            x, y, z, w = 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            x, y, z, w = (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            x, y, z, w = (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale
    return float(x), float(y), float(z), float(w)


def extract_segmented_destination_cloud(
        organized_xyz: np.ndarray, mask: np.ndarray, minimum_points: int
        ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Select measured geometry only inside the synchronized RGB silhouette."""
    xyz = np.asarray(organized_xyz)
    mask = np.asarray(mask)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("registered point cloud must be organized as H x W x 3")
    if mask.shape != xyz.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("destination mask must be boolean and match organized cloud")
    foreground = mask & np.isfinite(xyz).all(axis=2)
    points = xyz[foreground]
    if len(points) < minimum_points:
        raise ValueError("insufficient segmented destination depth points")
    return points.astype(np.float32), foreground, (0, 0, xyz.shape[1], xyz.shape[0])


def recover_semantic_source_mask(organized, connected_mask, semantic_mask,
                                 distance_m=0.012, surface_points=None):
    """Associate disconnected depth duplicates only inside the same SAM target.

    The grasp filter may discard image components even though their measured
    XYZ samples coincide with the validated surface. Preserve their ownership
    without growing through another object's mask or unrelated background.
    """
    points = np.asarray(organized, dtype=np.float32)
    connected = np.asarray(connected_mask, dtype=bool)
    semantic = np.asarray(semantic_mask, dtype=bool)
    if points.shape != connected.shape + (3,) or semantic.shape != connected.shape:
        raise ValueError('source association input dimensions differ')
    finite = np.isfinite(points).all(axis=2)
    result = connected.copy()
    support = (points[connected & finite] if surface_points is None else
               np.asarray(surface_points, dtype=np.float32).reshape(-1, 3))
    support = support[np.isfinite(support).all(axis=1)]
    candidates = semantic & ~connected & finite
    if len(support) and candidates.any():
        distances, _ = cKDTree(support).query(points[candidates])
        result[candidates] = distances <= distance_m
    return result


def recover_depth_connected_boundary(organized, mask, protected=None, distance_m=0.010,
                                     pixel_radius=3):
    """Recover a bounded SAM seam using original-mask measured XYZ support.

    Search a bounded image band and 10 mm from the ORIGINAL surface,
    protecting other objects and depth discontinuities. No iterative growth.
    """
    points = np.asarray(organized, dtype=np.float32)
    original = np.asarray(mask, dtype=bool)
    protected = np.zeros_like(original) if protected is None else np.asarray(protected, bool)
    if points.shape != original.shape + (3,) or protected.shape != original.shape:
        raise ValueError('boundary recovery input dimensions differ')
    valid = np.isfinite(points).all(axis=2)
    recovered = original.copy()
    radius = max(0, min(10, int(pixel_radius)))
    band = cv2.dilate(original.astype(np.uint8),
                      np.ones((2*radius + 1, 2*radius + 1), np.uint8)) > 0
    rows, columns = np.nonzero(band & ~original & ~protected & valid)
    height, width = original.shape
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            if dv == 0 and du == 0:
                continue
            neighbours_v, neighbours_u = rows + dv, columns + du
            inside = ((neighbours_v >= 0) & (neighbours_v < height) &
                      (neighbours_u >= 0) & (neighbours_u < width))
            v, u = rows[inside], columns[inside]
            nv, nu = neighbours_v[inside], neighbours_u[inside]
            supported = original[nv, nu] & valid[nv, nu]
            close = np.sum((points[v, u] - points[nv, nu])**2, axis=1) <= distance_m**2
            recovered[v, u] |= supported & close
    return recovered


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _urdf_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def load_hand_eye_matrix(path: str) -> np.ndarray:
    """Load and validate the calibrated end-effector-to-camera X matrix."""
    from pathlib import Path
    import yaml
    calibration_path = Path(path).expanduser()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"hand-eye calibration not found: {calibration_path}")
    with calibration_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    matrix = np.asarray(data.get("hand_eye_transform"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("hand_eye_transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("hand_eye_transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("hand_eye_transform rotation determinant is not +1")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("hand_eye_transform has an invalid homogeneous row")
    return matrix


def piper_base_to_camera(joints, hand_eye_matrix: np.ndarray) -> np.ndarray:
    """Match ActiveVisionServer's FK(joints) * calibrated-X calculation."""
    q = np.asarray(joints, dtype=np.float64).reshape(-1)
    if q.size < 6 or not np.isfinite(q[:6]).all():
        raise ValueError("six finite Piper joint positions are required")

    transforms = [np.eye(4, dtype=np.float64) for _ in range(6)]
    transforms[0][2, 3] = 0.123
    transforms[0][:3, :3] = _urdf_rotation(0.0, 0.0, 0.0) @ _rotation_z(q[0])
    transforms[1][:3, :3] = (
        _urdf_rotation(1.5707963, -0.1357866, -3.1415926) @ _rotation_z(q[1]))
    transforms[2][0, 3] = 0.28503
    transforms[2][:3, :3] = (
        _urdf_rotation(0.0, 0.0, -1.7938494) @ _rotation_z(q[2]))
    transforms[3][0, 3] = -0.02198
    transforms[3][1, 3] = -0.25075
    transforms[3][:3, :3] = (
        _urdf_rotation(1.5707963, 0.0, 0.0) @ _rotation_z(q[3]))
    transforms[4][:3, :3] = (
        _urdf_rotation(-1.5707963, 0.0, 0.0) @ _rotation_z(q[4]))
    transforms[5][0, 3] = 8.8259e-05
    transforms[5][1, 3] = -0.091
    transforms[5][:3, :3] = (
        _urdf_rotation(1.5707963, 0.0, 0.0) @ _rotation_z(q[5]))

    base_to_end = np.eye(4, dtype=np.float64)
    for transform in transforms:
        base_to_end = base_to_end @ transform
    return base_to_end @ np.asarray(hand_eye_matrix, dtype=np.float64)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform row-vector XYZ points with a homogeneous transform."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (xyz @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def transform_pose(rotation: np.ndarray, translation: np.ndarray,
                   transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform an object pose from camera coordinates into base coordinates."""
    base_rotation = transform[:3, :3] @ np.asarray(rotation, dtype=np.float64)
    base_translation = (
        transform[:3, :3] @ np.asarray(translation, dtype=np.float64)
        + transform[:3, 3])
    return base_rotation.astype(np.float32), base_translation.astype(np.float32)


def pose_matrix(position, quaternion) -> np.ndarray:
    """Build a finite rigid pose from XYZ and XYZW components."""
    xyz = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion, dtype=np.float64)
    if xyz.shape != (3,) or q.shape != (4,) or not np.isfinite(xyz).all() or not np.isfinite(q).all():
        raise ValueError("pose components must be finite XYZ and XYZW")
    length = np.linalg.norm(q)
    if length < 1e-8:
        raise ValueError("pose quaternion is zero")
    x, y, z, w = q / length
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]
    result[:3, 3] = xyz
    return result


def attached_object_delta(source_grasp, placement_grasp) -> np.ndarray:
    """Keep the original object-to-gripper transform fixed at placement."""
    return np.asarray(placement_grasp) @ np.linalg.inv(np.asarray(source_grasp))


def prepare_held_scene(request, response):
    """Freeze payload and environmental geometry for physical motion planning."""
    from cv_bridge import CvBridge
    from scipy.spatial import cKDTree
    from sensor_msgs.msg import PointField
    from sensor_msgs_py import point_cloud2
    from .packing import destination_support_mask

    try:
        cloud, header = request.cloud, request.observation_header
        if (header.frame_id != 'base_link' or cloud.header != header or
                (not header.stamp.sec and not header.stamp.nanosec) or
                not cloud.width or cloud.height < 2 or len(cloud.data) > 16*1024*1024):
            raise ValueError('INVALID_HELD_OBSERVATION')
        bridge = CvBridge()
        mask = bridge.imgmsg_to_cv2(request.source_mask, desired_encoding='mono8') > 0
        if (request.source_mask.header.stamp != header.stamp or
                mask.shape != (cloud.height, cloud.width)):
            raise ValueError('HELD_MASK_OBSERVATION_MISMATCH')
        organized = point_cloud2.read_points_numpy(
            cloud, field_names=('x', 'y', 'z'), skip_nans=False).reshape(
                cloud.height, cloud.width, 3)
        finite = np.isfinite(organized).all(axis=2)
        measured = organized[mask & finite]
        if len(measured) < 80:
            raise ValueError('HELD_SOURCE_TOO_SMALL')
        completed, bottom, envelope, _ = construct_support_bottom(measured, organized[finite])
        body = np.vstack((completed, envelope))
        semantic = request.semantic_mask
        associated = mask.copy()
        if semantic.width:
            semantic_pixels = bridge.imgmsg_to_cv2(semantic, desired_encoding='mono8') > 0
            if (semantic.header.stamp != header.stamp or semantic_pixels.shape != mask.shape):
                raise ValueError('SEMANTIC_MASK_OBSERVATION_MISMATCH')
            semantic_pixels &= organized[..., 2] >= np.percentile(measured[:, 2], 3)-.003
            associated = recover_semantic_source_mask(
                organized, mask, semantic_pixels, surface_points=body, distance_m=.012)
            body = np.vstack((body, organized[associated & ~mask & finite]))
        if semantic.width:
            rows, columns = np.nonzero(semantic_pixels)
            protected = np.ones(mask.shape, bool)
            if len(rows):
                protected[max(0, rows.min()-5):rows.max()+6,
                          max(0, columns.min()-5):columns.max()+6] = False
            protected |= organized[..., 2] <= float(np.median(bottom[:, 2]))+.005
            if request.destination_mask.width:
                destination_pixels = bridge.imgmsg_to_cv2(request.destination_mask, desired_encoding='mono8') > 0
                if destination_pixels.shape != mask.shape:
                    raise ValueError('DESTINATION_MASK_OBSERVATION_MISMATCH')
                protected |= destination_pixels
            recovered = recover_depth_connected_boundary(
                organized, associated, protected, distance_m=.004, pixel_radius=5)
            body = np.vstack((body, organized[recovered & ~associated & finite]))
            associated = recovered
        flags = (body[:, 2] <= np.percentile(bottom[:, 2], 50)+.003).astype(np.uint8)
        points = np.vstack((body, bottom))
        flags = np.concatenate((flags, np.full(len(bottom), 2, np.uint8)))
        fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(('x', 'y', 'z'))]
        response.source_object_cloud = point_cloud2.create_cloud(
            header, fields+[PointField(name='bottom', offset=12,
                                       datatype=PointField.UINT8, count=1)],
            [(*p, int(f)) for p, f in zip(points, flags)])
        support = np.zeros(mask.shape, np.uint8)
        destination = request.destination_mask
        mapped_base = request.destination_base_cloud
        base_points = None
        base_support = None
        if mapped_base.width:
            if mapped_base.header.frame_id != 'base_link' or len(mapped_base.data) > 16*1024*1024:
                raise ValueError('INVALID_MAPPED_DESTINATION')
            base_points = point_cloud2.read_points_numpy(
                mapped_base, field_names=('x', 'y', 'z'), skip_nans=True).reshape(-1, 3)
            if len(base_points) < 80:
                raise ValueError('MAPPED_DESTINATION_TOO_SMALL')
            distance, _ = cKDTree(base_points).query(
                organized[finite], distance_upper_bound=.01)
            valid = np.zeros(mask.shape, bool)
            valid[finite] = distance <= .01
            support = destination_support_mask(
                organized, valid & ~associated, base_points).astype(np.uint8)
            base_support = destination_support_mask(
                base_points, np.ones(len(base_points), bool), base_points).astype(np.uint8)
        elif destination.width:
            dmask = bridge.imgmsg_to_cv2(destination, desired_encoding='mono8') > 0
            if destination.header.stamp != header.stamp or dmask.shape != mask.shape:
                raise ValueError('DESTINATION_MASK_OBSERVATION_MISMATCH')
            valid = dmask & finite & ~associated
            if valid.sum() < 80:
                raise ValueError('DESTINATION_SUPPORT_TOO_SMALL')
            support = destination_support_mask(organized, valid, organized[valid]).astype(np.uint8)
        environment = finite & ~associated
        scene_points, scene_support = organized[environment], support[environment]
        support_points = organized[support > 0]
        if base_points is not None:
            scene_points = np.vstack((scene_points, base_points))
            scene_support = np.concatenate((scene_support, base_support))
            support_points = base_points[base_support > 0]
        placed = request.placed_scene_cloud
        if placed.width:
            if placed.header.frame_id != 'base_link' or len(placed.data) > 16*1024*1024:
                raise ValueError('INVALID_MAPPED_PLACED_SCENE')
            placed_points = point_cloud2.read_points_numpy(
                placed, field_names=('x', 'y', 'z'), skip_nans=True).reshape(-1, 3)
            scene_points = np.vstack((scene_points, placed_points))
            scene_support = np.concatenate((scene_support, np.zeros(len(placed_points), np.uint8)))
        response.collision_scene = point_cloud2.create_cloud(
            header, fields+[PointField(name='support', offset=12,
                                       datatype=PointField.UINT8, count=1)],
            [(*p, int(f)) for p, f in zip(scene_points, scene_support)])
        response.destination_support_cloud = point_cloud2.create_cloud_xyz32(header, support_points)
        center = points.mean(axis=0)
        (response.source_object_center.x, response.source_object_center.y,
         response.source_object_center.z) = map(float, center)
        response.success = True
        response.status = 'IMMUTABLE_HELD_SCENE_READY'
    except (ValueError, TypeError, RuntimeError) as error:
        response.status = str(error)
    return response


