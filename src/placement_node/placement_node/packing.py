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
from .floor_slots import FloorSlots


@dataclass
class PackingDiagnostics:
    slot_layout: dict | None = None
    slot_candidate_cells: list | None = None
    slot_blocked_cells: list | None = None
    tested: int = 0
    contained: int = 0
    stable: int = 0
    support_contacts_rejected: int = 0
    support_area_rejected: int = 0
    support_com_rejected: int = 0
    stack_maximum_coverage: float = 0.0
    orientations: int = 0
    elapsed_seconds: float = 0.0
    support_tilt_deg: float = 0.0
    support_normal: tuple = (0.0, 0.0, 1.0)
    flattened_support: bool = False
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


def destination_support_mask(organized, destination_valid, destination_points,
                             maximum_tilt_deg=10.0, contact_band=.003):
    """Label floor contacts relative to the measured plane, retaining raised obstacles."""
    _, normal, _ = _estimate_support_frame(destination_points, maximum_tilt_deg)
    heights = _finite(destination_points) @ normal
    baseline = float(np.percentile(heights, 20))
    return destination_valid & ((organized @ normal) <= baseline+contact_band)


def _make_container_grid(
        organized_scene: np.ndarray, destination_points: np.ndarray,
        resolution: float, maximum_pile_height: float,
        flatten_planar_support: bool = False):
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

    # Distance to a simplified boundary vertex approximates simultaneous
    # proximity to two non-parallel edges, which gives corner-first packing.
    perimeter = cv2.arcLength(hull, True)
    corners = cv2.approxPolyDP(
        hull, max(1.0, 0.025 * perimeter), True).reshape(-1, 2)
    grid_y, grid_x = np.indices(container.shape)
    corner_distance_cells = np.full(container.shape, np.inf, np.float32)
    for corner_x, corner_y in corners:
        corner_distance_cells = np.minimum(
            corner_distance_cells,
            np.hypot(grid_x - corner_x, grid_y - corner_y))
    corner_distance = corner_distance_cells * resolution

    plate_height = np.full(container.shape, -np.inf, dtype=np.float32)
    flattened = False
    flattened_height = 0.0
    if flatten_planar_support:
        # Find the densest horizontal band. A physical flat tray normally
        # dominates rim pixels and isolated reflective-depth outliers.
        sorted_heights = np.sort(destination[:, 2])
        band_width = 0.010
        left = 0
        best_left = 0
        best_right = 0
        for right in range(len(sorted_heights)):
            while sorted_heights[right] - sorted_heights[left] > band_width:
                left += 1
            if right - left > best_right - best_left:
                best_left, best_right = left, right
        band = sorted_heights[best_left:best_right + 1]
        if len(band) >= max(50, int(0.55 * len(sorted_heights))):
            flattened = True
            flattened_height = float(np.median(band))
            plate_height[container.astype(bool)] = flattened_height
    if not flattened:
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
    if flattened:
        # Suppress base-surface depth noise without removing genuinely
        # elevated geometry such as an existing object or food pile.
        elevated = local_scene[:, 2] > flattened_height + 0.012
        local_scene = local_scene[elevated]
        scene_cells = scene_cells[elevated]
    np.maximum.at(
        pile_height, (scene_cells[:, 1], scene_cells[:, 0]),
        local_scene[:, 2])
    edge_distance = cv2.distanceTransform(
        container, cv2.DIST_L2, 3).astype(np.float32) * resolution
    return (
        container.astype(bool), pile_height, plate_height, xy_min,
        edge_distance, corner_distance, flattened)


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


def _select_ranked_candidates(candidates, candidate_count):
    """Keep existing spatial/yaw ranking using bounded local neighbor queries."""
    selected, deferred, cells = [], [], {}
    limit = max(1, int(candidate_count))

    def cell(candidate):
        return tuple(np.floor(candidate.translation[:2] / 0.025).astype(int))

    def neighbors(candidate):
        x, y = cell(candidate)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield from cells.get((x+dx, y+dy), ())

    def append(candidate):
        selected.append(candidate)
        cells.setdefault(cell(candidate), []).append(candidate)

    for candidate in candidates:
        if any(np.linalg.norm(candidate.translation[:2] - other.translation[:2]) < 0.025
               for other in neighbors(candidate)):
            deferred.append(candidate)
            continue
        append(candidate)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for candidate in deferred:
            yaw = math.atan2(candidate.rotation[1, 0], candidate.rotation[0, 0])
            if any(np.linalg.norm(candidate.translation[:2] - other.translation[:2]) < 0.012
                   and abs(yaw - math.atan2(other.rotation[1, 0], other.rotation[0, 0]))
                   < math.radians(10) for other in neighbors(candidate)):
                continue
            append(candidate)
            if len(selected) >= limit:
                break
    return selected


def _select_diverse_candidates(candidates, candidate_count, preserve_yaw=False):
    """Cover the support with coarse positions and different fixed-grasp yaws.

    Inputs already passed containment/support checks and are score-ranked.
    Keep the best packing pose, then greedily cover XY and yaw. Never relax
    geometric feasibility to fill a shortlist; return fewer poses if necessary.
    """
    if not candidates:
        return []
    limit = max(1, int(candidate_count))
    xy = np.asarray([item.translation[:2] for item in candidates], dtype=np.float64)
    yaw = np.asarray([math.atan2(item.rotation[1, 0], item.rotation[0, 0])
                      for item in candidates])
    # A fixed grasp rotates with the object. Give yaw comparable weight to a
    # 10 cm translation so a small shortlist also explores arm orientations,
    # while retaining the independent 4 cm positional separation below.
    features = np.column_stack((xy, 0.10 * np.cos(yaw), 0.10 * np.sin(yaw)))
    nearest = np.full(len(candidates), np.inf)
    available = np.ones(len(candidates), dtype=bool)
    chosen = []
    index = 0
    spatial_limit = max(1, min(limit, max(4, limit // 2))) if preserve_yaw else limit
    while len(chosen) < spatial_limit:
        chosen.append(index)
        # Every distinct returned position is at least 4 cm from every other.
        separation = np.sum((xy - xy[index]) ** 2, axis=1)
        available &= separation >= 0.040 ** 2 - 1e-12
        nearest = np.minimum(nearest, np.sum((features - features[index]) ** 2, axis=1))
        if not available.any():
            break
        index = int(np.argmax(np.where(available, nearest, -np.inf)))
    if preserve_yaw:
        # Broad yaw coverage alone can skip a narrow reachable wrist window.
        # Reserve at most a quarter of the budget for the ranked pose's nearby
        # rotations, retaining the same containment-tested proposals and scores.
        focused_count = 0
        best_angles = np.abs(np.arctan2(np.sin(yaw-yaw[0]), np.cos(yaw-yaw[0])))
        focused = ((np.sum((xy-xy[0]) ** 2, axis=1) <= 0.0251 ** 2) &
                   (best_angles <= math.radians(20)+1e-6))
        for index in np.flatnonzero(focused):
            if len(chosen) >= limit or focused_count >= limit // 4:
                break
            angle = np.abs(np.arctan2(np.sin(yaw[chosen]-yaw[index]),
                                     np.cos(yaw[chosen]-yaw[index])))
            if np.all(angle >= math.radians(4.9)):
                chosen.append(index)
                focused_count += 1
                nearest = np.minimum(nearest, np.sum((features-features[index]) ** 2, axis=1))
        anchors = xy[chosen].copy()
        # Rotating a partial footprint recenters its bounding box. Retain
        # nearby yaw alternatives despite that shift; containment and exclusive
        # slot checks have already validated each proposal independently.
        near_anchor = np.min(np.sum((xy[:, None] - anchors) ** 2, axis=2), axis=1) <= 0.0251 ** 2
        while len(chosen) < limit:
            valid = near_anchor.copy()
            for selected in chosen:
                angle = np.abs(np.arctan2(np.sin(yaw-yaw[selected]), np.cos(yaw-yaw[selected])))
                valid &= ((np.sum((xy-xy[selected]) ** 2, axis=1) > 0.012 ** 2) |
                          (angle >= math.radians(45)))
            if not valid.any():
                break
            index = int(np.argmax(np.where(valid, nearest, -np.inf)))
            chosen.append(index)
            nearest = np.minimum(nearest, np.sum((features-features[index]) ** 2, axis=1))
    # Preserve the original score ordering used by downstream validation.
    return [candidates[index] for index in sorted(chosen)]


def _heightmap_trial_block(positions, offsets, bottom, pile, plate, stack_grid,
                          obstacle_distance, clearance, contact_tolerance,
                          obstacle_clearance, support_mode, stack_minimum_coverage=0.0):
    """Evaluate unchanged height/contact/mode filters in bounded batches."""
    rows = positions[:, 0, None] + offsets[None, :, 1]
    columns = positions[:, 1, None] + offsets[None, :, 0]
    heights = pile[rows, columns]
    valid = np.isfinite(heights).all(axis=1)
    z = np.max(heights - bottom, axis=1).astype(np.float64) + clearance
    # The scalar path adds a Python float to float32 arrays. Match that
    # rounding before contact tests and scoring, rather than promoting
    # the batched arrays to float64 and changing threshold decisions.
    support = z.astype(np.float32)[:, None] + bottom - heights <= contact_tolerance
    elevated = np.any(support & (heights > plate[rows, columns] + 0.012), axis=1)
    if support_mode == 'edge':
        valid &= ~elevated
    valid &= elevated | ~np.any(
        obstacle_distance[rows, columns] < max(0.0, obstacle_clearance), axis=1)
    if support_mode in ('auto', 'stack'):
        selected_top = stack_grid[rows, columns]
        covered = np.isfinite(selected_top) & (
            np.abs(z.astype(np.float32)[:, None] + np.min(bottom) - clearance - selected_top) <= contact_tolerance)
        matched = support & covered
        if stack_minimum_coverage > 0.0:
            supported = np.any(matched, axis=1) & (covered.mean(axis=1) >= stack_minimum_coverage)
            valid &= (elevated & supported) if support_mode == "stack" else (~elevated | supported)
            # Contacts across all exposed support surfaces share the hull check.
            support = np.where(elevated[:, None], matched, support)
        elif support_mode == "stack":
            valid &= (np.any(support, axis=1) & elevated & ~np.any(support & ~matched, axis=1))
    return rows, columns, heights, z, support, valid


def _exhaustive_orientation(centers, offsets, bottom, top, pile, plate,
                            stack_grid, obstacle_distance, edge_distance,
                            corner_distance, occupied_distance, occupied_centroid,
                            origin, resolution, clearance, contact_tolerance,
                            obstacle_clearance, support_mode, minimum_area,
                            support_margin, center_of_mass, maximum_pile_height,
                            support_baseline, adjacency_weight, tilt_penalty,
                            rotation_base, support_frame, diagnostics, stack_minimum_coverage=0.0):
    """Batch the scalar trial equations, retaining their precision and order."""
    candidates = []
    stability_cache = {}
    # Bound temporary memory independently of the destination size.
    batch_size = max(1, min(256, 262144 // max(1, len(offsets))))
    for start in range(0, len(centers), batch_size):
        positions = centers[start:start + batch_size]
        rows, columns, heights, z, support, valid = _heightmap_trial_block(
            positions, offsets, bottom, pile, plate, stack_grid, obstacle_distance,
            clearance, contact_tolerance, obstacle_clearance, support_mode, stack_minimum_coverage)
        if support_mode in ('auto', 'stack') and len(positions):
            exposed_top = stack_grid[rows, columns]
            covered = np.isfinite(exposed_top) & (
                np.abs(z.astype(np.float32)[:, None] + np.min(bottom) - clearance - exposed_top) <= contact_tolerance)
            raised = np.any((heights > plate[rows, columns] + 0.012) & (
                z.astype(np.float32)[:, None] + bottom - heights <= contact_tolerance), axis=1)
            if raised.any():
                diagnostics.stack_maximum_coverage = max(
                    diagnostics.stack_maximum_coverage, float(covered[raised].mean(axis=1).max()))
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        # Stability depends on local contact cells, not their translated XY.
        # Identical contact masks share the exact same hull and COM test.
        packed = np.packbits(support[indices], axis=1)
        masks, first, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
        for group, (mask, representative) in enumerate(zip(masks, first)):
            key = mask.tobytes()
            result = stability_cache.get(key)
            if result is None:
                result = _support_is_stable(
                    offsets, support[indices[representative]], resolution,
                    minimum_area, support_margin, center_of_mass)
                if len(stability_cache) < 1024:
                    stability_cache[key] = result
            stable, _, rejection = result
            members = indices[inverse == group]
            if stable:
                diagnostics.stable += len(members)
            else:
                valid[members] = False
                attribute = {'contacts': 'support_contacts_rejected',
                             'area': 'support_area_rejected',
                             'com': 'support_com_rejected'}[rejection]
                setattr(diagnostics, attribute, getattr(diagnostics, attribute) + len(members))
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        placed_top = z[indices].astype(np.float32)[:, None] + top
        maximum_height = np.max(placed_top, axis=1).astype(np.float64) - support_baseline
        if support_mode != 'auto':
            below_limit = maximum_height <= maximum_pile_height
            indices = indices[below_limit]
            placed_top = placed_top[below_limit]
            maximum_height = maximum_height[below_limit]
        if not len(indices):
            continue
        trial_rows, trial_columns = rows[indices], columns[indices]
        increase = np.maximum(placed_top - heights[indices], 0.0)
        volume = np.sum(increase, axis=1).astype(np.float64) * resolution * resolution
        compact = np.linalg.norm(positions[indices, ::-1] - occupied_centroid, axis=1) * resolution
        edges = np.min(edge_distance[trial_rows, trial_columns], axis=1).astype(np.float64)
        corners = np.min(corner_distance[trial_rows, trial_columns], axis=1).astype(np.float64)
        adjacency = (np.min(occupied_distance[trial_rows, trial_columns], axis=1).astype(np.float64)
                     if occupied_distance is not None else corners)
        scores = -(volume + 0.0002 * maximum_height + 0.00001 * compact +
                   0.00005 * edges + adjacency_weight * adjacency + tilt_penalty)
        translations = np.column_stack((
            origin + positions[indices, ::-1] * resolution, z[indices])).astype(np.float32)
        # Keep scalar matrix/vector multiplication, including its rounding.
        for translation, score in zip(translations, scores):
            candidates.append((PlacementCandidate(
                rotation_base.astype(np.float32),
                (support_frame.T @ translation).astype(np.float32), float(score)), 0.0))
    return candidates


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
        free_support_tilt_deg: float = 45.0,
        adjacency_weight: float = 0.004,
        obstacle_clearance_m: float = 0.006,
        support_mode: str = "auto",
        stack_support_points: np.ndarray | None = None,
        early_stop_candidate_count: int = 0,
        stack_minimum_coverage: float = 0.6,
        candidate_yaw_variants: bool = False,
        reserve_near_space: bool = False,
        grasp_reference_positions=None,
        flatten_planar_support: bool = False,
        floor_slot_layout=None, floor_slot_reserved_cells=(),
        floor_slot_yaw_step_deg=5.0, floor_slot_share_unused_space=False, floor_slot_occupied_xy=(),
        floor_slot_adaptive=True, floor_slot_max_size_m=.125,
        floor_slot_min_size_m=.06, floor_slot_large_square_scale=1.5):
    """Search stable floor or combined-surface stack placements.

    stack_support_points is deprecated and ignored; scene heights are authoritative.
    Preserve measured floor heights by default so packing and collision validation
    agree. Flattening is an explicit option for prefiltered plane observations.
    """
    started = time.perf_counter()
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("heightmap resolution must be positive and finite")
    resolution = max(float(resolution), .002)
    if not np.isfinite(stack_minimum_coverage) or not 0 <= stack_minimum_coverage <= 1:
        raise ValueError("stack minimum coverage must be in [0, 1] (0 disables footprint coverage)")
    grid_policy = support_mode == "floor_slots"
    if grid_policy:
        if orientation_policy != "upright":
            raise ValueError("floor_slots requires upright orientation")
        support_mode = "edge"
    if support_mode not in ("auto", "edge", "stack"):
        raise ValueError(f"unknown support mode: {support_mode}")
    points = _finite(object_points)
    if len(points) < 80:
        raise ValueError("object cloud has fewer than 80 finite points")
    policy = str(orientation_policy).lower()
    if policy not in ("upright", "free"):
        raise ValueError(f"unknown packing orientation policy: {policy}")
    support_frame, support_normal, support_tilt = _estimate_support_frame(
        destination_points,
        upright_support_tilt_deg if policy == "upright" else free_support_tilt_deg)
    if policy == "upright":
        # UPRIGHT is gravity-referenced. Keep the measured normal only for
        # diagnostics; RGB-D plane noise must not lean the placed object.
        support_frame = np.eye(3, dtype=np.float32)
    scene_in_support_frame = _finite(organized_scene) @ support_frame.T
    destination_in_support_frame = _finite(destination_points) @ support_frame.T

    center = points.mean(axis=0)
    local = points - center
    # The segmented cloud is a partial single-view observation. Its arithmetic
    # center is not a reliable mass center (a cup handle is a common bias).
    # A coordinate-wise median is a better model-free stability reference.
    center_of_mass_local = np.median(points, axis=0) - center
    measured_bottom = np.empty((0, 3), dtype=np.float32)
    if bottom_points is not None:
        measured_bottom = _finite(bottom_points)
        if len(measured_bottom) >= 3:
            # The completed support contour is intentionally symmetric around
            # the principal body and excludes handles/appendages. Its center
            # is therefore a better gravity projection than a partial-view
            # point-cloud median.
            center_of_mass_local[:2] = (
                measured_bottom[:, :2].mean(axis=0) - center[:2])
    leveling = (
        _align_normal_to_gravity(bottom_points)
        if bottom_points is not None else np.eye(3, dtype=np.float32))
    # Aggregate every measured point into the rotated footprint. Uniform
    # point decimation can lose a rim/handle extremum and admit edge overhang.
    # The completed bottom contour is not only a visualization aid. Include it
    # in the object's bottom heightmap so stability is evaluated against the
    # inferred complete support boundary rather than only the visible wall.
    # This is especially important for a cup observed from one side.
    support_geometry_local = local
    if len(measured_bottom) >= 3:
        support_geometry_local = np.vstack((
            local, measured_bottom.astype(np.float32) - center))
    (container, pile, plate, origin, edge_distance, corner_distance,
     flattened_support) = _make_container_grid(
        scene_in_support_frame, destination_in_support_frame, resolution,
        maximum_pile_height_m,
        flatten_planar_support=(policy == "upright" and flatten_planar_support))

    orientations = []
    if policy == "upright":
        yaw_step = floor_slot_yaw_step_deg if grid_policy else upright_yaw_step_deg
        if not np.isfinite(yaw_step) or yaw_step <= 0:
            raise ValueError("yaw step must be positive and finite")
        yaw_values = np.arange(0.0, 360.0, yaw_step)
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
        flattened_support=flattened_support,
        heightmap=pile.copy(), heightmap_mask=container.copy(),
        heightmap_base_m=float(np.nanmin(plate[container])),
        heightmap_resolution_m=float(resolution))
    raw_candidates = []
    # The support baseline is invariant across every orientation and XY trial.
    # Avoid rescanning the entire grid for each stable candidate.
    support_baseline = float(np.nanmin(plate[container]))
    # The measured tray rim is part of the destination, not an already placed
    # object. Compare against its measured height before choosing adjacency
    # instead of corner preference; keep the full rim in collision/support maps.
    adjacency_reference = plate.copy()
    destination_cells = np.floor(
        (destination_in_support_frame[:, :2] - origin) / resolution).astype(np.int32)
    np.maximum.at(adjacency_reference,
        (destination_cells[:, 1], destination_cells[:, 0]),
        destination_in_support_frame[:, 2])
    adjacency_reference = cv2.dilate(adjacency_reference, np.ones((3, 3), np.uint8))
    occupied = np.zeros(container.shape, dtype=bool)
    occupied[container] = (
        pile[container] - adjacency_reference[container]) > 0.012
    obstacle_distance = cv2.distanceTransform(
        (~occupied).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * resolution
    occupied_indices = np.argwhere(occupied)
    occupied_centroid = (
        occupied_indices[:, ::-1].mean(axis=0)
        if len(occupied_indices) else
        np.asarray([pile.shape[1] * 0.5, pile.shape[0] * 0.5]))
    occupied_distance = (
        cv2.distanceTransform(
            (~occupied).astype(np.uint8), cv2.DIST_L2, 3
        ).astype(np.float32) * resolution
        if len(occupied_indices) else None)

    # Raster containment can admit a sub-cell overhang on an oblique edge.
    # Retain a metric convex boundary and require the complete footprint to
    # clear it by the configured clearance, independently of grid resolution.
    boundary = cv2.convexHull(
        destination_in_support_frame[:, :2].astype(np.float32)).reshape(-1, 2)
    edges = np.roll(boundary, -1, axis=0) - boundary
    normals = np.column_stack((edges[:, 1], -edges[:, 0])).astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    boundary_offsets = np.sum(normals * boundary, axis=1)
    # Exposed raised cells come from the scene, without object identity labels.
    stack_grid = np.where(pile > plate + 0.012, pile, -np.inf)
    slot_grid = None
    slot_groups_by_pose = {}
    slot_reserved = (set() if floor_slot_share_unused_space else
                     set(map(tuple, floor_slot_reserved_cells)))
    if grid_policy:
        slot_grid = (FloorSlots.from_dict(floor_slot_layout) if floor_slot_layout else
                     FloorSlots.build_adaptive(
                         boundary, clearance_m, floor_slot_max_size_m,
                         floor_slot_min_size_m,
                         surface_mask=(container & (pile <= plate+.012)),
                         surface_origin=origin, surface_resolution=resolution,
                         large_square_scale=floor_slot_large_square_scale)
                     if floor_slot_adaptive else FloorSlots.build(boundary, clearance_m))
        # Sample relative to the frozen grid, so a slightly rotated tray
        # does not lose an axis-aligned tight fit between global yaw steps.
        grid_heading = math.degrees(math.atan2(slot_grid.axes[0, 1], slot_grid.axes[0, 0]))
        orientations = [(roll, pitch, yaw+grid_heading) for roll, pitch, yaw in orientations]
        # Existing scene geometry blocks whole slots, not just the footprint.
        raised = np.argwhere((pile > plate + .012) & container)
        if len(raised) and not floor_slot_share_unused_space:
            slot_reserved |= slot_grid.touched(origin + raised[:, ::-1]*resolution)
        diagnostics.slot_layout = slot_grid.to_dict()
        diagnostics.slot_blocked_cells = [list(cell) for cell in sorted(slot_reserved)]
        early_stop_candidate_count = max(1, candidate_count*4)
    grid_free_base = container.astype(np.uint8)
    slot_cell_masks = {}
    slot_free_radius_cache = {}
    slot_available = set(map(tuple, slot_grid.cells))-slot_reserved if slot_grid else set()
    if slot_grid is not None:
        usable = np.zeros(container.shape, np.uint8)
        for cell in slot_grid.cells:
            world = slot_grid.territory_vertices(cell)
            vertices = np.rint((world-origin)/resolution).astype(np.int32)
            cv2.fillConvexPoly(usable, vertices, 1)
            if not floor_slot_share_unused_space:
                cell_mask = np.zeros(container.shape, np.uint8)
                cv2.fillConvexPoly(cell_mask, vertices, 1)
                slot_cell_masks[tuple(cell)] = cell_mask.astype(bool)
        grid_free_base &= usable
        if not floor_slot_share_unused_space:
            for cell in slot_reserved:
                mask = slot_cell_masks.get(cell)
                if mask is not None:
                    grid_free_base[mask] = 0
    reservation_free = None
    if grid_policy and floor_slot_share_unused_space and len(floor_slot_occupied_xy):
        occupied_xy = (np.asarray(floor_slot_occupied_xy, np.float64)+.5)*.005
        indices = np.rint((occupied_xy-origin)/resolution).astype(int)
        inside = ((indices[:, 0]>=0) & (indices[:, 0]<container.shape[1]) &
                  (indices[:, 1]>=0) & (indices[:, 1]<container.shape[0]))
        indices = indices[inside]
        reserved_grid = np.zeros(container.shape, np.uint8)
        reserved_grid[indices[:, 1], indices[:, 0]] = 1
        # Stored footprints already include one 5 mm dilation. Add the new
        # object's allowance before shortlisting, matching commit semantics.
        reservation_free = 1-cv2.dilate(reserved_grid, np.ones((3, 3), np.uint8))
    valid_centers = np.argwhere(container)
    # Complete the cheaper single-slot tier before evaluating combinations.
    # A fitting rectangle alone is insufficient: only fully validated raw
    # candidates can suppress the fallback pass. Shared sub-cell reservations
    # and legacy grids retain their existing exhaustive behavior.
    single_first = (slot_grid is not None and slot_grid.bounds is not None and
                    not floor_slot_share_unused_space)
    search_order = ([(True, *angles) for angles in orientations] +
                    [(False, *angles) for angles in orientations] if single_first else
                    [(False, *angles) for angles in orientations])
    diagnostics.orientations = 0
    for single_only, roll_deg, pitch_deg, yaw_deg in search_order:
        if (single_first and not single_only and
                diagnostics.orientations == len(orientations) and raw_candidates):
            break
        diagnostics.orientations += 1
        policy_rotation = _rotation(
            math.radians(roll_deg), math.radians(pitch_deg),
            math.radians(yaw_deg))
        rotation = policy_rotation @ leveling
        center_of_mass_xy = (rotation @ center_of_mass_local)[:2]
        rotated_points, offsets, bottom, top = _object_heightmaps(
            support_geometry_local, rotation, resolution)
        if policy == "upright" and len(measured_bottom) >= 3:
            # The completed contact plane is a measured physical lower bound.
            # RGB-D samples a few mm below it must not lift the whole object
            # onto a single noisy pixel and make its real support look absent.
            support_z = float(np.median(
                ((measured_bottom - center) @ rotation.T)[:, 2]))
            bottom = np.maximum(bottom, support_z)
            top = np.maximum(top, bottom)
        min_offset = offsets.min(axis=0)
        max_offset = offsets.max(axis=0)
        # Evaluate containment for all centers in native OpenCV code. Kernel
        # offsets and anchor use the exact same cells as the heightmap lookup.
        kernel_size = max_offset - min_offset + 1
        footprint = np.zeros((kernel_size[1], kernel_size[0]), np.uint8)
        footprint[offsets[:, 1] - min_offset[1], offsets[:, 0] - min_offset[0]] = 1
        contained = cv2.erode(
            container.astype(np.uint8), footprint,
            anchor=(-int(min_offset[0]), -int(min_offset[1])),
            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if reservation_free is not None:
            contained &= cv2.erode(reservation_free, footprint,
                                  anchor=(-int(min_offset[0]), -int(min_offset[1])),
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        contained_centers = np.argwhere(contained & container)
        if slot_grid is not None:
            phases = (tuple(np.arange(-.04, .0401, .01)) if floor_slot_share_unused_space
                      else (0., -.01, .01))
            # Keep exact continuous slot centers. The heightmap is sampled at
            # the nearest raster center, just as object points are rasterized;
            # do not let raster rounding change the reserved inventory cells.
            exact_xy = rotated_points[:, :2]
            trials = slot_grid.trials(exact_xy, slot_reserved, clearance_m, phases,
                                     single_only=single_only)
            continuous_centers = {}
            continuous_groups = {}
            for xy, intended in trials:
                pixel = np.rint((xy-origin)/resolution).astype(int)
                column, row = map(int, pixel)
                if not (0 <= column < container.shape[1] and 0 <= row < container.shape[0]):
                    continue
                if not contained[row, column]:
                    continue
                if not intended <= slot_available:
                    continue
                if (row, column) not in continuous_centers:
                    continuous_centers[(row, column)] = xy
                    continuous_groups[(row, column)] = intended
            contained_centers = np.asarray(sorted(continuous_centers), np.int32).reshape(-1, 2)
        footprint_hull = cv2.convexHull(
            rotated_points[:, :2].astype(np.float32)).reshape(-1, 2)
        footprint_limits = np.max(footprint_hull @ normals.T, axis=0)
        centers_xy = (np.asarray([continuous_centers[tuple(c)] for c in contained_centers])
                      .reshape(-1, 2) if slot_grid is not None else
                      origin + contained_centers[:, ::-1] * resolution)
        exact_containment = np.all(
            centers_xy @ normals.T + footprint_limits <=
            boundary_offsets - max(0.0, float(clearance_m)) + 1.0e-7,
            axis=1)
        contained_centers = contained_centers[exact_containment]
        if not grid_policy and early_stop_candidate_count > 0 and len(contained_centers):
            # Corner/adjacency order is only a search priority. Every returned
            # candidate still passes the unchanged exact support/collision tests.
            proximity = occupied_distance if occupied_distance is not None else corner_distance
            priority = proximity[contained_centers[:, 0], contained_centers[:, 1]]
            contained_centers = contained_centers[np.argsort(priority, kind='stable')]
        diagnostics.tested += len(contained_centers) if grid_policy else len(valid_centers)
        diagnostics.contained += len(contained_centers)
        if early_stop_candidate_count <= 0:
            raw_candidates.extend(_exhaustive_orientation(
                contained_centers, offsets, bottom, top, pile, plate,
                stack_grid, obstacle_distance, edge_distance, corner_distance,
                occupied_distance, occupied_centroid, origin, resolution,
                clearance_m, contact_tolerance_m, obstacle_clearance_m,
                support_mode, minimum_support_area_m2, support_margin_m,
                center_of_mass_xy, maximum_pile_height_m, support_baseline,
                float(adjacency_weight), 0.00002 * (abs(roll_deg) + abs(pitch_deg)),
                support_frame.T @ rotation, support_frame, diagnostics, stack_minimum_coverage))
            continue
        # Fast searches keep their original priority, early-stop and diversity
        # decisions. Batch only the exact rejection filters, which also makes
        # exhausted/blocked supports cheap to reject before the Python loop.
        filtered = []
        batch_size = max(1, min(256, 262144 // max(1, len(offsets))))
        for start in range(0, len(contained_centers), batch_size):
            positions = contained_centers[start:start + batch_size]
            *_, valid = _heightmap_trial_block(
                positions, offsets, bottom, pile, plate, stack_grid, obstacle_distance,
                clearance_m, contact_tolerance_m, obstacle_clearance_m, support_mode, stack_minimum_coverage)
            filtered.append(positions[valid])
        contained_centers = np.vstack(filtered) if filtered else contained_centers
        orientation_positions = []
        for row, column in contained_centers:
            if not grid_policy and early_stop_candidate_count > 0 and any(
                    (row-prior_row)**2 + (column-prior_column)**2 < (.025/resolution)**2
                    for prior_row, prior_column in orientation_positions):
                continue
            columns = column + offsets[:, 0]
            rows = row + offsets[:, 1]
            supporting_height = pile[rows, columns]
            if not np.isfinite(supporting_height).all():
                continue
            z = float(np.max(supporting_height - bottom) + clearance_m)
            gap = z + bottom - supporting_height
            support = gap <= contact_tolerance_m
            # Mode filters share exactly the same heightmap, stability test,
            # score and orientation search. They only select contact support.
            if support_mode == "edge" and np.any(
                    support & (supporting_height > plate[rows, columns] + 0.012)):
                continue
            # A footprint can clear occupied cells yet sit inside the 3 mm
            # collision band. Reserve a metric gap for pose error as well.
            floor_contacts = not np.any(
                support & (supporting_height > plate[rows, columns] + 0.012))
            if floor_contacts and np.any(
                    obstacle_distance[rows, columns] < max(0.0, obstacle_clearance_m)):
                continue
            if support_mode == "stack" or (support_mode == "auto" and not floor_contacts):
                selected_top = stack_grid[rows, columns]
                covered = np.isfinite(selected_top) & (
                    np.abs(z + np.min(bottom) - clearance_m - selected_top) <= contact_tolerance_m)
                diagnostics.stack_maximum_coverage = max(
                    diagnostics.stack_maximum_coverage, float(covered.mean()))
                matched = support & covered
                if not np.any(matched & (supporting_height > plate[rows, columns] + 0.012)):
                    continue
                if stack_minimum_coverage > 0.0:
                    if covered.mean() < stack_minimum_coverage:
                        continue
                    support = matched
                elif not np.any(support) or np.any(support & ~matched):
                    continue
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
            maximum_height = float(np.max(placed_top) - support_baseline)
            if support_mode != "auto" and maximum_height > maximum_pile_height_m:
                continue
            compact_distance = float(np.linalg.norm(
                np.asarray([column, row]) - occupied_centroid)) * resolution
            # Empty destination: put a footprint near a polygon corner (two
            # edges). Occupied destination: pack beside the closest elevated
            # region. Height/volume terms still discourage premature stacking.
            edge_clearance = float(np.min(edge_distance[rows, columns]))
            corner_clearance = float(np.min(corner_distance[rows, columns]))
            adjacency_clearance = (
                float(np.min(occupied_distance[rows, columns]))
                if occupied_distance is not None else corner_clearance)
            tilt_penalty = 0.00002 * (abs(roll_deg) + abs(pitch_deg))
            score = -(
                volume_increase + 0.0002 * maximum_height +
                0.00001 * compact_distance +
                0.00005 * edge_clearance +
                float(adjacency_weight) * adjacency_clearance + tilt_penalty)
            if slot_grid is not None:
                used = continuous_groups[(row, column)]
                score -= .01*len(used)
                if floor_slot_share_unused_space:
                    score -= .001*len(used-set(map(tuple, floor_slot_reserved_cells)))
                # Preserve a contiguous opening for later objects rather than
                # greedily scattering individually reachable placements.
                if floor_slot_share_unused_space:
                    free_after = grid_free_base.copy()
                    if reservation_free is not None:
                        free_after &= reservation_free
                    free_after[rows, columns] = 0
                    free_radius = float(cv2.distanceTransform(
                        free_after, cv2.DIST_L2, 3).max())*resolution
                else:
                    # Exclusive inventory slots consume whole cells. Equivalent
                    # groups share the same remaining-space score across yaws.
                    key = frozenset(used)
                    free_radius = slot_free_radius_cache.get(key)
                    if free_radius is None:
                        free_after = grid_free_base.copy()
                        for cell in used:
                            free_after[slot_cell_masks[cell]] = 0
                        free_radius = float(cv2.distanceTransform(
                            free_after, cv2.DIST_L2, 3).max())*resolution
                        slot_free_radius_cache[key] = free_radius
                score += .01*free_radius
            translation_support = np.asarray([
                origin[0] + column * resolution,
                origin[1] + row * resolution,
                z], dtype=np.float32)
            if slot_grid is not None:
                translation_support[:2] = continuous_centers[(row, column)]
            rotation_base = support_frame.T @ rotation
            translation_base = support_frame.T @ translation_support
            if grid_policy and floor_slot_share_unused_space and len(floor_slot_occupied_xy):
                exact_mask = FloorSlots.reservation_mask(points-center, rotation_base, translation_base)
                if exact_mask & set(map(tuple, floor_slot_occupied_xy)):
                    continue
            if slot_grid is not None:
                slot_groups_by_pose[(rotation_base.astype(np.float32).tobytes(),
                                     translation_base.astype(np.float32).tobytes())] = used
            raw_candidates.append((
                PlacementCandidate(
                    rotation_base.astype(np.float32),
                    translation_base.astype(np.float32), score),
                support_area))
            orientation_positions.append((row, column))
            if not grid_policy and early_stop_candidate_count > 0 and (
                    len(raw_candidates) >= early_stop_candidate_count or
                    len(orientation_positions) >= min(4, early_stop_candidate_count)):
                break
        if (not grid_policy and early_stop_candidate_count > 0 and
                len(raw_candidates) >= early_stop_candidate_count):
            break

    raw_candidates.sort(key=lambda item: item[0].score, reverse=True)
    # Preserve fast/refined benchmark policies; coarse coverage targets small
    # exhaustive shortlists where ranked neighbors otherwise exhaust the budget.
    use_ranked = (early_stop_candidate_count > 0 or candidate_count > 16
                  or support_mode == 'stack')
    selector = _select_ranked_candidates if use_ranked else _select_diverse_candidates
    pool = [candidate for candidate, _ in raw_candidates]
    if grid_policy and grasp_reference_positions is None:
        grasp_reference_positions = [center]
    if grasp_reference_positions is not None and len(grasp_reference_positions):
        references = np.asarray(grasp_reference_positions, dtype=np.float32).reshape(-1, 3)
        if not np.isfinite(references).all():
            raise ValueError('grasp reference positions must be finite')
        source_bearings = np.arctan2(references[:, 1], references[:, 0])
        def heading_error(candidate):
            placed_grasps = (references-center) @ candidate.rotation.T + candidate.translation
            bearing_change = np.arctan2(placed_grasps[:, 1], placed_grasps[:, 0])-source_bearings
            yaw = math.atan2(candidate.rotation[1, 0], candidate.rotation[0, 0])
            return float(np.min(np.abs(np.arctan2(
                np.sin(yaw-bearing_change), np.cos(yaw-bearing_change)))))
        # Rank yaw proposals by preservation of the grasp's radial heading.
        # This is only a shortlist heuristic; unchanged IK/motion checks decide.
        pool = [PlacementCandidate(item.rotation, item.translation,
                                   item.score - (.0001 if grid_policy else .002) * heading_error(item)) for item in pool]
        pool.sort(key=lambda item: item.score, reverse=True)
    if slot_grid is not None:
        # Exhaust single slots first, then groups using the fewest slots.
        # Keep spatial/yaw diversity inside equal-area tiers only.
        groups = {}
        for candidate in pool:
            used = slot_groups_by_pose[(candidate.rotation.tobytes(),
                                        candidate.translation.tobytes())]
            groups.setdefault(slot_grid.fit_priority(used), []).append(candidate)
        selected = []
        for tier, priority in enumerate(sorted(groups)):
            proposals = _select_diverse_candidates(
                groups[priority], candidate_count-len(selected), preserve_yaw=True)
            # VAMP also compares scores, so encode the same strict tier ordering
            # while retaining geometry/heading preference within each tier.
            selected.extend(PlacementCandidate(
                item.rotation, item.translation, -float(tier)+.1*math.atan(item.score))
                for item in proposals)
            if len(selected) >= candidate_count:
                break
    else:
        selected = (_select_diverse_candidates(pool, candidate_count, candidate_yaw_variants)
                    if selector is _select_diverse_candidates else selector(pool, candidate_count))
    if reserve_near_space and not grid_policy:
        # A large first object may reach farther via its off-center grasp.
        # Preserve floor priority, then reserve nearer workspace for later objects.
        def priority(candidate):
            placed = (points - center) @ candidate.rotation.T + candidate.translation
            bottom_height = float(np.min((placed @ support_frame.T)[:, 2]))
            floor = bottom_height <= support_baseline + clearance_m + 0.015
            return floor, float(np.linalg.norm(candidate.translation[:2]))
        selected.sort(key=priority, reverse=True)
        selected = [PlacementCandidate(item.rotation, item.translation, -float(rank))
                    for rank, item in enumerate(selected)]
    if slot_grid is not None:
        diagnostics.slot_candidate_cells = []
        for candidate in selected:
            used = slot_groups_by_pose[(candidate.rotation.tobytes(),
                                        candidate.translation.tobytes())]
            diagnostics.slot_candidate_cells.append([list(cell) for cell in sorted(used)])
    diagnostics.elapsed_seconds = time.perf_counter() - started
    return selected, diagnostics
