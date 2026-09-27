"""Deterministic heightmap-minimization placement search.

This module follows the constructive search described by Wang and Hauser,
"Stable Bin Packing of Non-convex 3D Objects with a Robot Manipulator".  It is
ROS independent so geometry and stability behavior can be unit tested.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .inference import PlacementCandidate


@dataclass
class PackingDiagnostics:
    tested: int = 0
    contained: int = 0
    stable: int = 0
    support_contacts_rejected: int = 0
    support_area_rejected: int = 0
    support_com_rejected: int = 0
    orientations: int = 0
    elapsed_seconds: float = 0.0
    support_tilt_deg: float = 0.0
    support_normal: tuple = (0.0, 0.0, 1.0)
    # Exact destination top-height grid used by the packing search. These
    # debug fields remain ROS independent; the node decides whether and where
    # to render them.
    heightmap: np.ndarray | None = None
    heightmap_mask: np.ndarray | None = None
    heightmap_base_m: float = 0.0
    heightmap_resolution_m: float = 0.0


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float32)


def _finite(points: np.ndarray) -> np.ndarray:
    clean = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return clean[np.isfinite(clean).all(axis=1)]


def _align_normal_to_gravity(bottom_points: np.ndarray) -> np.ndarray:
    """Return a rotation which makes the measured object bottom horizontal."""
    bottom = _finite(bottom_points)
    if len(bottom) < 12:
        return np.eye(3, dtype=np.float32)
    centered = bottom - np.median(bottom, axis=0)
    _, _, vh = np.linalg.svd(centered.T @ centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    normal /= max(float(np.linalg.norm(normal)), 1.0e-9)
    gravity = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(normal, gravity)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(normal, gravity), -1.0, 1.0))
    if sine < 1.0e-8:
        return np.eye(3, dtype=np.float32)
    axis = cross / sine
    skew = np.asarray([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]], dtype=np.float64)
    rotation = (
        np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew))
    return rotation.astype(np.float32)


def _estimate_support_frame(
        destination_points: np.ndarray,
        maximum_tilt_deg: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate a dominant destination plane and map it onto horizontal.

    The returned rotation maps base_link coordinates into the packing frame.
    Plane tilt is bounded by the placement policy, so scan noise cannot turn an
    upright request into a strongly leaning placement.
    """
    points = _finite(destination_points)
    if len(points) < 50:
        return np.eye(3, dtype=np.float32), np.array([0, 0, 1], np.float32), 0.0
    rng = np.random.default_rng(29)
    fit = points
    if len(fit) > 5000:
        fit = fit[rng.choice(len(fit), 5000, replace=False)]
    best = None
    best_count = 0
    for _ in range(96):
        sample = fit[rng.choice(len(fit), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length < 1.0e-8:
            continue
        normal /= length
        if normal[2] < 0.0:
            normal = -normal
        # Reject nearly vertical walls but allow the free-placement policy to
        # discover substantially inclined support surfaces.
        if normal[2] < math.cos(math.radians(60.0)):
            continue
        offset = -float(np.dot(normal, sample[0]))
        inliers = np.abs(fit @ normal + offset) <= 0.005
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best = inliers
            best_count = count
    if best is None or best_count < 40:
        return np.eye(3, dtype=np.float32), np.array([0, 0, 1], np.float32), 0.0
    plane = fit[best]
    centered = plane - plane.mean(axis=0)
    _, _, vh = np.linalg.svd(centered.T @ centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    normal /= max(float(np.linalg.norm(normal)), 1.0e-9)
    measured_tilt = math.degrees(math.atan2(
        float(np.linalg.norm(normal[:2])), float(normal[2])))
    allowed_tilt = min(measured_tilt, max(float(maximum_tilt_deg), 0.0))
    if measured_tilt > 1.0e-8 and allowed_tilt < measured_tilt:
        horizontal = normal[:2] / max(float(np.linalg.norm(normal[:2])), 1.0e-9)
        angle = math.radians(allowed_tilt)
        normal = np.asarray([
            horizontal[0] * math.sin(angle),
            horizontal[1] * math.sin(angle),
            math.cos(angle)], dtype=np.float64)
    # Map the accepted support normal to +Z using Rodrigues' formula.
    gravity = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(normal, gravity)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(normal, gravity), -1.0, 1.0))
    if sine < 1.0e-8:
        frame = np.eye(3, dtype=np.float32)
    else:
        axis = cross / sine
        skew = np.asarray([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]], dtype=np.float64)
        frame = (
            np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
        ).astype(np.float32)
    return frame, normal.astype(np.float32), allowed_tilt


def _make_container_grid(
        organized_scene: np.ndarray, destination_points: np.ndarray,
        resolution: float, maximum_pile_height: float):
    destination = _finite(destination_points)
    scene = _finite(organized_scene)
    if len(destination) < 50:
        raise ValueError("destination surface has fewer than 50 points")
    resolution = max(float(resolution), 0.002)
    xy_min = destination[:, :2].min(axis=0) - resolution
    xy_max = destination[:, :2].max(axis=0) + resolution
    size = np.ceil((xy_max - xy_min) / resolution).astype(int) + 1
    if np.any(size < 3) or np.any(size > 400):
        raise ValueError(f"destination grid dimensions are invalid: {size}")

    destination_cells = np.floor(
        (destination[:, :2] - xy_min) / resolution).astype(np.int32)
    destination_cells[:, 0] = np.clip(destination_cells[:, 0], 0, size[0] - 1)
    destination_cells[:, 1] = np.clip(destination_cells[:, 1], 0, size[1] - 1)
    hull = cv2.convexHull(destination_cells.reshape(-1, 1, 2))
    container = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.fillConvexPoly(container, hull, 1)
    container = cv2.morphologyEx(
        container, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    plate_height = np.full(container.shape, -np.inf, dtype=np.float32)
    np.maximum.at(
        plate_height,
        (destination_cells[:, 1], destination_cells[:, 0]),
        destination[:, 2])
    # Spread sparse registered-depth samples by one grid cell. Keep the
    # sentinel out of plate_height: assigning -1e6 to a missing cell makes it
    # finite on the next iteration and later corrupts both the support baseline
    # and placement score by roughly one million metres.
    missing_sentinel = np.float32(-1.0e20)
    for _ in range(3):
        missing = container.astype(bool) & ~np.isfinite(plate_height)
        if not np.any(missing):
            break
        dilated = cv2.dilate(
            np.where(
                np.isfinite(plate_height), plate_height,
                missing_sentinel).astype(np.float32),
            np.ones((3, 3), np.uint8))
        fillable = missing & (dilated > missing_sentinel * 0.5)
        if not np.any(fillable):
            break
        plate_height[fillable] = dilated[fillable]
    finite_plate = plate_height[np.isfinite(plate_height) & (container > 0)]
    if not len(finite_plate):
        raise ValueError("destination heightmap contains no finite cells")
    fallback_height = float(np.percentile(destination[:, 2], 10.0))
    plate_height[container.astype(bool) & ~np.isfinite(plate_height)] = fallback_height

    pile_height = plate_height.copy()
    base = float(np.percentile(destination[:, 2], 10.0))
    inside_bounds = (
        (scene[:, 0] >= xy_min[0]) & (scene[:, 0] <= xy_max[0]) &
        (scene[:, 1] >= xy_min[1]) & (scene[:, 1] <= xy_max[1]) &
        (scene[:, 2] >= base - 0.03) &
        (scene[:, 2] <= base + maximum_pile_height))
    local_scene = scene[inside_bounds]
    scene_cells = np.floor(
        (local_scene[:, :2] - xy_min) / resolution).astype(np.int32)
    valid_cells = (
        (scene_cells[:, 0] >= 0) & (scene_cells[:, 0] < size[0]) &
        (scene_cells[:, 1] >= 0) & (scene_cells[:, 1] < size[1]))
    local_scene = local_scene[valid_cells]
    scene_cells = scene_cells[valid_cells]
    allowed = container[scene_cells[:, 1], scene_cells[:, 0]] > 0
    local_scene = local_scene[allowed]
    scene_cells = scene_cells[allowed]
    np.maximum.at(
        pile_height, (scene_cells[:, 1], scene_cells[:, 0]),
        local_scene[:, 2])
    edge_distance = cv2.distanceTransform(
        container, cv2.DIST_L2, 3).astype(np.float32) * resolution
    return (
        container.astype(bool), pile_height, plate_height, xy_min,
        edge_distance)


def _object_heightmaps(points: np.ndarray, rotation: np.ndarray, resolution: float):
    rotated = points @ rotation.T
    cells = np.rint(rotated[:, :2] / resolution).astype(np.int32)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    bottom = np.full(len(unique), np.inf, dtype=np.float32)
    top = np.full(len(unique), -np.inf, dtype=np.float32)
    np.minimum.at(bottom, inverse, rotated[:, 2])
    np.maximum.at(top, inverse, rotated[:, 2])
    return rotated, unique, bottom, top


def _support_is_stable(
        offsets: np.ndarray, support: np.ndarray, resolution: float,
        minimum_area: float, margin: float,
        center_of_mass_xy: np.ndarray) -> tuple[bool, float, str]:
    contacts = offsets[support]
    if len(contacts) < 3:
        return False, 0.0, "contacts"
    polygon = cv2.convexHull(
        (contacts.astype(np.float32) * resolution).reshape(-1, 1, 2))
    area = abs(float(cv2.contourArea(polygon)))
    if area < minimum_area:
        return False, area, "area"
    point = tuple(float(value) for value in center_of_mass_xy)
    distance = cv2.pointPolygonTest(polygon, point, True)
    return distance >= margin, area, "com"


def search_heightmap_placements(
        object_points: np.ndarray, organized_scene: np.ndarray,
        destination_points: np.ndarray, orientation_policy: str,
        candidate_count: int = 8, resolution: float = 0.005,
        upright_yaw_step_deg: float = 15.0,
        free_yaw_step_deg: float = 30.0,
        tilt_angles_deg=(-20.0, -10.0, 0.0, 10.0, 20.0),
        clearance_m: float = 0.002,
        contact_tolerance_m: float = 0.006,
        minimum_support_area_m2: float = 0.0002,
        support_margin_m: float = 0.002,
        maximum_pile_height_m: float = 0.30,
        bottom_points: np.ndarray | None = None,
        upright_support_tilt_deg: float = 10.0,
        free_support_tilt_deg: float = 45.0):
    """Search compact stable placements using top/bottom heightmaps."""
    started = time.perf_counter()
    points = _finite(object_points)
    if len(points) < 80:
        raise ValueError("object cloud has fewer than 80 finite points")
    policy = str(orientation_policy).lower()
    if policy not in ("upright", "free"):
        raise ValueError(f"unknown packing orientation policy: {policy}")
    support_frame, support_normal, support_tilt = _estimate_support_frame(
        destination_points,
        upright_support_tilt_deg if policy == "upright" else free_support_tilt_deg)
    scene_in_support_frame = _finite(organized_scene) @ support_frame.T
    destination_in_support_frame = _finite(destination_points) @ support_frame.T

    center = points.mean(axis=0)
    local = points - center
    # The segmented cloud is a partial single-view observation. Its arithmetic
    # center is not a reliable mass center (a cup handle is a common bias).
    # A coordinate-wise median is a better model-free stability reference.
    center_of_mass_local = np.median(points, axis=0) - center
    leveling = (
        _align_normal_to_gravity(bottom_points)
        if bottom_points is not None else np.eye(3, dtype=np.float32))
    if len(local) > 4000:
        indices = np.linspace(0, len(local) - 1, 4000, dtype=np.int32)
        local = local[indices]
    container, pile, plate, origin, edge_distance = _make_container_grid(
        scene_in_support_frame, destination_in_support_frame, resolution,
        maximum_pile_height_m)

    orientations = []
    if policy == "upright":
        yaw_values = np.arange(0.0, 360.0, upright_yaw_step_deg)
        orientations = [(0.0, 0.0, yaw) for yaw in yaw_values]
    else:
        yaw_values = np.arange(0.0, 360.0, free_yaw_step_deg)
        tilts = [(0.0, 0.0)]
        for angle in tilt_angles_deg:
            if abs(angle) > 1.0e-6:
                tilts.extend(((angle, 0.0), (0.0, angle)))
        orientations = [
            (roll, pitch, yaw) for roll, pitch in tilts for yaw in yaw_values]

    diagnostics = PackingDiagnostics(
        orientations=len(orientations), support_tilt_deg=support_tilt,
        support_normal=tuple(float(value) for value in support_normal),
        heightmap=pile.copy(), heightmap_mask=container.copy(),
        heightmap_base_m=float(np.nanmin(plate[container])),
        heightmap_resolution_m=float(resolution))
    raw_candidates = []
    occupied = np.zeros(container.shape, dtype=bool)
    occupied[container] = (
        pile[container] - plate[container]) > 0.012
    occupied_indices = np.argwhere(occupied)
    occupied_centroid = (
        occupied_indices[:, ::-1].mean(axis=0)
        if len(occupied_indices) else
        np.asarray([pile.shape[1] * 0.5, pile.shape[0] * 0.5]))

    valid_centers = np.argwhere(container)
    for roll_deg, pitch_deg, yaw_deg in orientations:
        policy_rotation = _rotation(
            math.radians(roll_deg), math.radians(pitch_deg),
            math.radians(yaw_deg))
        rotation = policy_rotation @ leveling
        center_of_mass_xy = (rotation @ center_of_mass_local)[:2]
        _, offsets, bottom, top = _object_heightmaps(local, rotation, resolution)
        min_offset = offsets.min(axis=0)
        max_offset = offsets.max(axis=0)
        for row, column in valid_centers:
            diagnostics.tested += 1
            if (column + min_offset[0] < 0 or
                    column + max_offset[0] >= container.shape[1] or
                    row + min_offset[1] < 0 or
                    row + max_offset[1] >= container.shape[0]):
                continue
            columns = column + offsets[:, 0]
            rows = row + offsets[:, 1]
            if not np.all(container[rows, columns]):
                continue
            diagnostics.contained += 1
            supporting_height = pile[rows, columns]
            if not np.isfinite(supporting_height).all():
                continue
            z = float(np.max(supporting_height - bottom) + clearance_m)
            gap = z + bottom - supporting_height
            support = gap <= contact_tolerance_m
            stable, support_area, rejection = _support_is_stable(
                offsets, support, resolution, minimum_support_area_m2,
                support_margin_m, center_of_mass_xy)
            if not stable:
                if rejection == "contacts":
                    diagnostics.support_contacts_rejected += 1
                elif rejection == "area":
                    diagnostics.support_area_rejected += 1
                else:
                    diagnostics.support_com_rejected += 1
                continue
            diagnostics.stable += 1
            placed_top = z + top
            increase = np.maximum(placed_top - pile[rows, columns], 0.0)
            volume_increase = float(np.sum(increase) * resolution * resolution)
            maximum_height = float(np.max(placed_top) - np.nanmin(plate[container]))
            compact_distance = float(np.linalg.norm(
                np.asarray([column, row]) - occupied_centroid)) * resolution
            # Explicit edge-first organization: among equally low and stable
            # placements, prefer a footprint close to the destination border.
            edge_clearance = float(np.min(edge_distance[rows, columns]))
            tilt_penalty = 0.00002 * (abs(roll_deg) + abs(pitch_deg))
            score = -(
                volume_increase + 0.0002 * maximum_height +
                0.00002 * compact_distance +
                0.00020 * edge_clearance + tilt_penalty)
            translation_support = np.asarray([
                origin[0] + column * resolution,
                origin[1] + row * resolution,
                z], dtype=np.float32)
            rotation_base = support_frame.T @ rotation
            translation_base = support_frame.T @ translation_support
            raw_candidates.append((
                PlacementCandidate(
                    rotation_base.astype(np.float32),
                    translation_base.astype(np.float32), score),
                support_area))

    raw_candidates.sort(key=lambda item: item[0].score, reverse=True)
    selected = []
    for candidate, _ in raw_candidates:
        if any(
                np.linalg.norm(candidate.translation[:2] - other.translation[:2]) < 0.012 and
                abs(math.atan2(candidate.rotation[1, 0], candidate.rotation[0, 0]) -
                    math.atan2(other.rotation[1, 0], other.rotation[0, 0])) < math.radians(10)
                for other in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max(1, int(candidate_count)):
            break
    diagnostics.elapsed_seconds = time.perf_counter() - started
    return selected, diagnostics
