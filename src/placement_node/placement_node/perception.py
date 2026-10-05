"""ROS-independent grounding and registered-cloud helpers."""

import re

import cv2
import numpy as np


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
        seed: int = 17) -> tuple[np.ndarray, np.ndarray, dict]:
    """Complete a supported object's unseen bottom with a measured-plane cap.

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

    heights = object_points @ normal + offset
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

    circular = _fit_circular_footprint(body_uv, spacing, rng)
    footprint_model = "circle" if circular is not None else "oriented_box"
    if circular is not None:
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
