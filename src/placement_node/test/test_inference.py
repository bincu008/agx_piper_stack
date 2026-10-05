import numpy as np

from placement_node.inference import (
    diffusion_timestep_schedule, downsample_points, infer_placements,
    make_dummy_box_cloud)
from placement_node.packing import search_heightmap_placements
from placement_node.perception import (
    construct_support_bottom, estimate_support_translation,
    extract_destination_cloud, parse_grounding_box)
from placement_node.transforms import (
    piper_base_to_camera, transform_points, transform_pose)


def test_grounding_box_normalized_conversion():
    assert parse_grounding_box("[250, 100, 750, 900]", 640, 480) == (
        160, 48, 480, 432)


def test_extract_dominant_foreground_component():
    xyz = np.full((40, 60, 3), np.nan, dtype=np.float32)
    yy, xx = np.mgrid[:40, :60]
    xyz[:, :, 0] = (xx - 30) * 0.001
    xyz[:, :, 1] = (yy - 20) * 0.001
    xyz[:, :, 2] = 0.8
    xyz[10:30, 15:45, 2] = 0.45
    points, mask, _ = extract_destination_cloud(
        xyz, (40, 60), (10, 5, 50, 35), 0.03, 50)
    assert 500 <= len(points) <= 600
    assert np.allclose(np.median(points[:, 2]), 0.45)
    assert mask.dtype == np.bool_


def test_connected_surface_follows_curvature_not_fixed_depth():
    height, width = 80, 100
    yy, xx = np.mgrid[:height, :width]
    xyz = np.empty((height, width, 3), dtype=np.float32)
    xyz[:, :, 0] = (xx - width / 2) * 0.001
    xyz[:, :, 1] = (yy - height / 2) * 0.001
    xyz[:, :, 2] = 0.8
    plate = ((xx - 50) / 35) ** 2 + ((yy - 40) / 28) ** 2 <= 1.0
    # A smooth bowl spans 12 cm in depth, exceeding the old fixed band.
    radial = np.sqrt(((xx - 50) / 35) ** 2 + ((yy - 40) / 28) ** 2)
    xyz[:, :, 2][plate] = 0.42 + 0.12 * np.clip(radial[plate], 0, 1)
    points, mask, _ = extract_destination_cloud(
        xyz, (height, width), (10, 5, 90, 75), 0.02, 100)
    assert len(points) > 2500
    assert points[:, 2].max() - points[:, 2].min() > 0.10
    assert np.count_nonzero(mask) == len(points)


def test_construct_support_bottom_uses_low_object_band_not_raised_handle():
    rng = np.random.default_rng(9)
    table_xy = rng.uniform(-0.12, 0.12, size=(5000, 2))
    table = np.column_stack((
        table_xy, rng.normal(0.0, 0.0005, size=len(table_xy))))
    angles = np.linspace(-np.pi, np.pi, 1200, endpoint=False)
    heights = rng.uniform(0.0, 0.10, size=len(angles))
    cup = np.column_stack((
        0.04 * np.cos(angles), 0.04 * np.sin(angles), heights))
    handle = np.column_stack((
        rng.uniform(0.04, 0.08, 200), rng.normal(0.0, 0.004, 200),
        rng.uniform(0.04, 0.08, 200)))
    object_points = np.vstack((cup, handle)).astype(np.float32)
    completed, bottom, envelope, diagnostics = construct_support_bottom(
        object_points, table.astype(np.float32), seed=3)
    assert len(completed) == len(object_points) + len(bottom)
    assert len(bottom) > 40
    assert len(envelope) > len(bottom)
    assert np.max(np.abs(bottom[:, 2])) < 0.003
    assert np.percentile(np.linalg.norm(bottom[:, :2], axis=1), 99) < 0.05
    assert diagnostics["plane_normal"][2] > 0.99
    assert diagnostics["footprint_area_m2"] > 0.003
    assert diagnostics["footprint_model"] == "circle"


def test_partial_cylinder_sidewall_produces_closed_full_contour():
    rng = np.random.default_rng(21)
    table_xy = rng.uniform(-0.12, 0.12, size=(5000, 2))
    table = np.column_stack((
        table_xy, rng.normal(0.0, 0.0004, size=len(table_xy))))
    angles = np.deg2rad(np.linspace(-105.0, 105.0, 180))
    heights = np.linspace(0.0, 0.09, 35)
    partial_wall = np.array([
        [0.04 * np.cos(angle), 0.04 * np.sin(angle), height]
        for angle in angles for height in heights], dtype=np.float32)
    _, contour, envelope, diagnostics = construct_support_bottom(
        partial_wall, table.astype(np.float32), seed=5)
    assert diagnostics["footprint_model"] == "circle"
    assert 0.038 <= diagnostics["fitted_radius_m"] <= 0.045
    assert np.all(np.ptp(contour[:, :2], axis=0) > 0.075)
    assert len(envelope) > len(contour)


def test_box_sidewalls_select_oriented_rectangular_contour():
    rng = np.random.default_rng(33)
    table_xy = rng.uniform(-0.12, 0.12, size=(5000, 2))
    table = np.column_stack((
        table_xy, rng.normal(0.0, 0.0004, size=len(table_xy))))
    horizontal = np.linspace(-0.03, 0.03, 50)
    heights = np.linspace(0.0, 0.08, 30)
    front = np.array([
        [x, -0.025, z] for x in horizontal for z in heights])
    side = np.array([
        [0.03, y, z] for y in np.linspace(-0.025, 0.025, 40)
        for z in heights])
    partial_rear = np.array([
        [x, 0.025, z] for x in np.linspace(0.0, 0.03, 25)
        for z in heights])
    box = np.vstack((front, side, partial_rear)).astype(np.float32)
    _, contour, envelope, diagnostics = construct_support_bottom(
        box, table.astype(np.float32), seed=7)
    assert diagnostics["footprint_model"] == "oriented_box"
    extents = np.sort(np.ptp(contour[:, :2], axis=0))
    assert extents[0] > 0.045
    assert extents[1] > 0.060
    assert len(envelope) > len(contour)


def test_support_translation_lifts_penetrating_bottom_to_surface():
    yy, xx = np.mgrid[-5:6, -5:6]
    destination = np.column_stack((
        xx.reshape(-1) * 0.004,
        yy.reshape(-1) * 0.004,
        np.full(xx.size, 0.03)))
    bottom = np.array([
        [-0.01, -0.01, -0.04], [0.01, -0.01, -0.04],
        [-0.01, 0.01, -0.04], [0.01, 0.01, -0.04],
        [0.0, 0.0, -0.04], [0.004, 0.0, -0.04],
        [-0.004, 0.0, -0.04], [0.0, 0.004, -0.04],
        [0.0, -0.004, -0.04], [0.008, 0.008, -0.04]])
    shift, matches = estimate_support_translation(
        bottom, destination, clearance_m=0.002)
    assert matches == len(bottom)
    assert np.isclose(shift, 0.072, atol=0.001)


def test_support_translation_can_lower_floating_bottom():
    destination = np.array([
        [x, y, 0.01]
        for x in (-0.01, 0.0, 0.01)
        for y in (-0.01, 0.0, 0.01)], dtype=np.float32)
    bottom = np.repeat([[0.0, 0.0, 0.05]], 10, axis=0)
    shift, _ = estimate_support_translation(
        bottom, destination, clearance_m=0.002, minimum_matches=5)
    assert np.isclose(shift, -0.038, atol=0.001)


def test_heightmap_packing_stacks_upright_cup_on_small_box():
    axis = np.linspace(-0.06, 0.06, 25)
    plate = np.array([[x, y, 0.0] for x in axis for y in axis], dtype=np.float32)
    box_axis = np.linspace(-0.026, 0.026, 12)
    box_top = np.array(
        [[x, y, 0.05] for x in box_axis for y in box_axis], dtype=np.float32)
    scene = np.vstack((plate, box_top))
    angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    heights = np.linspace(0.0, 0.08, 12)
    wall = np.array([
        [0.043 * np.cos(angle), 0.043 * np.sin(angle), height]
        for angle in angles for height in heights], dtype=np.float32)
    radial = np.linspace(0.0, 0.043, 12)
    bottom = np.array([
        [radius * np.cos(angle), radius * np.sin(angle), 0.0]
        for radius in radial for angle in angles], dtype=np.float32)
    cup = np.vstack((wall, bottom))
    candidates, diagnostics = search_heightmap_placements(
        cup, scene, plate, "upright", candidate_count=4,
        resolution=0.005, upright_yaw_step_deg=45.0)
    assert candidates
    best = candidates[0]
    placed = (cup - cup.mean(axis=0)) @ best.rotation.T + best.translation
    assert np.percentile(placed[:, 2], 1) >= 0.049
    assert np.linalg.norm(best.translation[:2]) < 0.025
    assert diagnostics.stable > 0
    assert diagnostics.heightmap is not None
    assert diagnostics.heightmap_mask is not None
    assert diagnostics.heightmap.shape == diagnostics.heightmap_mask.shape
    assert np.max(
        diagnostics.heightmap[diagnostics.heightmap_mask] -
        diagnostics.heightmap_base_m) >= 0.049
    assert np.isclose(diagnostics.heightmap_resolution_m, 0.005)


def test_heightmap_missing_destination_cells_use_measured_fallback():
    axis = np.linspace(-0.06, 0.06, 25)
    complete_plate = np.array(
        [[x, y, 0.0] for x in axis for y in axis], dtype=np.float32)
    # Model the part of the plate hidden from the camera by a box.
    visible_plate = complete_plate[
        (np.abs(complete_plate[:, 0]) > 0.025) |
        (np.abs(complete_plate[:, 1]) > 0.025)]
    box_axis = np.linspace(-0.026, 0.026, 12)
    box_top = np.array(
        [[x, y, 0.05] for x in box_axis for y in box_axis], dtype=np.float32)
    scene = np.vstack((visible_plate, box_top))
    object_axis = np.linspace(-0.02, 0.02, 9)
    bottom = np.array(
        [[x, y, 0.0] for x in object_axis for y in object_axis],
        dtype=np.float32)
    object_points = np.vstack((bottom, bottom + [0.0, 0.0, 0.06]))

    candidates, diagnostics = search_heightmap_placements(
        object_points, scene, visible_plate, "upright", candidate_count=2,
        resolution=0.005, upright_yaw_step_deg=45.0,
        bottom_points=bottom)

    assert candidates
    assert diagnostics.heightmap_base_m > -0.01
    relative = (
        diagnostics.heightmap[diagnostics.heightmap_mask] -
        diagnostics.heightmap_base_m)
    assert np.min(relative) > -0.01
    assert 0.045 <= np.max(relative) <= 0.055
    assert candidates[0].score > -1.0


def test_upright_packing_levels_measured_bottom_before_yaw_search():
    axis = np.linspace(-0.08, 0.08, 33)
    plate = np.array(
        [[x, y, 0.0] for x in axis for y in axis], dtype=np.float32)
    object_axis = np.linspace(-0.025, 0.025, 11)
    bottom = np.array(
        [[x, y, 0.0] for x in object_axis for y in object_axis],
        dtype=np.float32)
    top = bottom + np.array([0.0, 0.0, 0.06], dtype=np.float32)
    box = np.vstack((bottom, top))
    angle = np.deg2rad(12.0)
    tilt = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)]], dtype=np.float32)
    tilted_box = box @ tilt.T
    tilted_bottom = bottom @ tilt.T
    candidates, _ = search_heightmap_placements(
        tilted_box, plate, plate, "upright", candidate_count=1,
        resolution=0.005, upright_yaw_step_deg=45.0,
        bottom_points=tilted_bottom)
    assert candidates
    transformed_bottom = (
        (tilted_bottom - tilted_box.mean(axis=0)) @
        candidates[0].rotation.T)
    normal = np.linalg.svd(
        (transformed_bottom - transformed_bottom.mean(axis=0)).T @
        (transformed_bottom - transformed_bottom.mean(axis=0)),
        full_matrices=False)[2][-1]
    assert abs(float(normal[2])) > 0.999


def test_upright_packing_remains_gravity_aligned_on_tilted_measurement():
    axis = np.linspace(-0.08, 0.08, 33)
    flat_plate = np.array(
        [[x, y, 0.0] for x in axis for y in axis], dtype=np.float32)
    object_axis = np.linspace(-0.02, 0.02, 9)
    bottom = np.array(
        [[x, y, 0.0] for x in object_axis for y in object_axis],
        dtype=np.float32)
    box = np.vstack((bottom, bottom + [0.0, 0.0, 0.05])).astype(np.float32)
    angle = np.deg2rad(8.0)
    surface_rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)]], dtype=np.float32)
    plate = flat_plate @ surface_rotation.T
    candidates, diagnostics = search_heightmap_placements(
        box, plate, plate, "upright", candidate_count=1,
        resolution=0.005, upright_yaw_step_deg=45.0,
        bottom_points=bottom)
    assert candidates
    placed_bottom = (
        (bottom - box.mean(axis=0)) @ candidates[0].rotation.T)
    placed_normal = np.linalg.svd(
        (placed_bottom - placed_bottom.mean(axis=0)).T @
        (placed_bottom - placed_bottom.mean(axis=0)),
        full_matrices=False)[2][-1]
    assert abs(float(placed_normal[2])) > 0.999
    assert 7.0 < diagnostics.support_tilt_deg < 9.0


def test_upright_packing_flattens_noisy_planar_destination():
    rng = np.random.default_rng(17)
    axis = np.linspace(-0.08, 0.08, 33)
    plate = np.array(
        [[x, y, rng.normal(0.0, 0.0015)] for x in axis for y in axis],
        dtype=np.float32)
    plate[::13, 2] += 0.025
    object_axis = np.linspace(-0.02, 0.02, 9)
    bottom = np.array(
        [[x, y, 0.0] for x in object_axis for y in object_axis],
        dtype=np.float32)
    box = np.vstack((bottom, bottom + [0.0, 0.0, 0.05]))
    candidates, diagnostics = search_heightmap_placements(
        box, plate, plate, "upright", candidate_count=1,
        resolution=0.005, upright_yaw_step_deg=45.0,
        bottom_points=bottom)
    assert candidates
    assert diagnostics.flattened_support
    assert abs(float(candidates[0].rotation[2, 2])) > 0.999


def test_support_contour_centers_partial_view_stability_reference():
    axis = np.linspace(-0.10, 0.10, 41)
    tray = np.array(
        [[x, y, 0.0] for x in axis for y in axis], dtype=np.float32)
    angles = np.linspace(-0.7, 0.7, 48)
    heights = np.linspace(0.0, 0.08, 10)
    partial_wall = np.array([
        [0.04 * np.cos(angle), 0.04 * np.sin(angle), height]
        for angle in angles for height in heights], dtype=np.float32)
    handle_bias = np.array([
        [0.065 + x, 0.0 + y, 0.04 + z]
        for x in np.linspace(0.0, 0.02, 5)
        for y in np.linspace(-0.01, 0.01, 5)
        for z in np.linspace(-0.02, 0.02, 5)], dtype=np.float32)
    partial_mug = np.vstack((partial_wall, handle_bias))
    contour_angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    bottom_contour = np.column_stack((
        0.04 * np.cos(contour_angles),
        0.04 * np.sin(contour_angles),
        np.zeros_like(contour_angles))).astype(np.float32)
    candidates, diagnostics = search_heightmap_placements(
        partial_mug, tray, tray, "upright", candidate_count=1,
        resolution=0.005, upright_yaw_step_deg=45.0,
        bottom_points=bottom_contour)
    assert candidates
    assert diagnostics.stable > 0


def test_empty_destination_uses_corner_first_organization():
    axis = np.linspace(-0.10, 0.10, 41)
    plate = np.array(
        [[x, y, 0.0] for x in axis for y in axis], dtype=np.float32)
    object_axis = np.linspace(-0.015, 0.015, 7)
    bottom = np.array(
        [[x, y, 0.0] for x in object_axis for y in object_axis],
        dtype=np.float32)
    object_points = np.vstack((
        bottom, bottom + np.array([0.0, 0.0, 0.04], dtype=np.float32)))
    candidates, _ = search_heightmap_placements(
        object_points, plate, plate, "upright", candidate_count=1,
        resolution=0.005, upright_yaw_step_deg=45.0,
        bottom_points=bottom)
    assert candidates
    assert np.min(np.abs(candidates[0].translation[:2])) > 0.06


def test_candidate_budget_contains_spatially_distinct_corners():
    axis = np.linspace(-0.10, 0.10, 41)
    plate = np.array(
        [[x, y, 0.0] for x in axis for y in axis], dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    bottom = np.column_stack((
        0.02 * np.cos(angles), 0.02 * np.sin(angles),
        np.zeros_like(angles))).astype(np.float32)
    object_points = np.vstack((bottom, bottom + [0.0, 0.0, 0.05]))
    candidates, _ = search_heightmap_placements(
        object_points, plate, plate, "upright", candidate_count=4,
        resolution=0.005, upright_yaw_step_deg=15.0,
        bottom_points=bottom)
    assert len(candidates) == 4
    translations = np.asarray([candidate.translation[:2] for candidate in candidates])
    separations = np.linalg.norm(
        translations[:, None, :] - translations[None, :, :], axis=2)
    assert np.min(separations[np.triu_indices(4, 1)]) >= 0.025 - 1.0e-6


def test_ordered_layout_rebuilds_heightmap_after_each_object():
    axis = np.linspace(-0.10, 0.10, 41)
    plate = np.array(
        [[x, y, 0.0] for x in axis for y in axis], dtype=np.float32)

    mug_axis = np.linspace(-0.018, 0.018, 9)
    mug_bottom = np.array(
        [[x, y, 0.0] for x in mug_axis for y in mug_axis],
        dtype=np.float32)
    mug = np.vstack((mug_bottom, mug_bottom + [0.0, 0.0, 0.07]))
    mug_candidates, _ = search_heightmap_placements(
        mug, plate, plate, "upright", candidate_count=1,
        resolution=0.005, upright_yaw_step_deg=45.0,
        bottom_points=mug_bottom)
    assert mug_candidates
    mug_centered = mug - mug.mean(axis=0)
    placed_mug = (
        mug_centered @ mug_candidates[0].rotation.T +
        mug_candidates[0].translation)

    bowl_axis = np.linspace(-0.024, 0.024, 11)
    bowl_bottom = np.array(
        [[x, y, 0.0] for x in bowl_axis for y in bowl_axis],
        dtype=np.float32)
    bowl = np.vstack((bowl_bottom, bowl_bottom + [0.0, 0.0, 0.035]))
    bowl_candidates, diagnostics = search_heightmap_placements(
        bowl, np.vstack((plate, placed_mug)), plate, "upright",
        candidate_count=1, resolution=0.005,
        upright_yaw_step_deg=45.0, bottom_points=bowl_bottom)

    assert bowl_candidates
    assert diagnostics.stable > 0
    assert np.linalg.norm(
        bowl_candidates[0].translation[:2] -
        mug_candidates[0].translation[:2]) < 0.09
    # The second search saw the virtually committed mug in its rebuilt map.
    assert np.max(
        diagnostics.heightmap[diagnostics.heightmap_mask] -
        diagnostics.heightmap_base_m) >= 0.065


def test_point_helpers():
    cloud = make_dummy_box_cloud(point_count=200)
    assert cloud.shape == (200, 3)
    assert downsample_points(cloud, 100).shape == (100, 3)
    schedule = diffusion_timestep_schedule(50)
    assert len(schedule) == 50
    assert schedule[:4] == [4, 3, 2, 1]
    assert schedule[-1] == 0


def test_manual_transform_points_and_pose():
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    points = transform_points(np.array([[0.1, 0.2, 0.3]]), transform)
    assert np.allclose(points, [[1.1, 2.2, 3.3]])
    rotation, translation = transform_pose(
        np.eye(3), np.array([0.1, 0.2, 0.3]), transform)
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, [1.1, 2.2, 3.3])


def test_piper_fk_returns_rigid_transform():
    transform = piper_base_to_camera(np.zeros(6), np.eye(4))
    assert transform.shape == (4, 4)
    assert np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1.0)


class _PitchDisturbingModel:
    @staticmethod
    def pos_emb(timestep):
        return timestep

    def __call__(self, model_input):
        import torch
        count = model_input["child_start_pcd"].shape[0]
        angle = torch.full((count,), 0.3)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotation = torch.zeros((count, 1, 3, 3))
        rotation[:, 0, 0, 0] = cosine
        rotation[:, 0, 0, 2] = sine
        rotation[:, 0, 1, 1] = 1.0
        rotation[:, 0, 2, 0] = -sine
        rotation[:, 0, 2, 2] = cosine
        translation = torch.zeros((count, 1, 3))
        return {"rot_mat": rotation, "trans": translation}


def test_upright_policy_removes_roll_and_pitch():
    destination = make_dummy_box_cloud([0.2, 0.2, 0.01], 400, 4)
    child = make_dummy_box_cloud([0.04, 0.04, 0.08], 200, 5)
    candidates, _ = infer_placements(
        _PitchDisturbingModel(), destination, child,
        candidate_count=3, refinement_steps=2, seed=7,
        orientation_policy="upright")
    assert candidates
    for candidate in candidates:
        assert np.allclose(
            candidate.rotation[2], [0.0, 0.0, 1.0], atol=1e-5)
        assert np.allclose(
            candidate.rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-5)


def test_full_footprint_keeps_rare_appendage_inside_destination():
    axis = np.linspace(-0.1, 0.1, 41)
    destination = np.array([[x, y, 0.0] for x in axis for y in axis], np.float32)
    object_axis = np.linspace(-0.015, 0.015, 7)
    bottom = np.array([[x, y, 0.0] for x in object_axis for y in object_axis], np.float32)
    body = np.vstack((bottom, bottom + [0.0, 0.0, 0.04]))
    points = np.tile(body, (70, 1)).astype(np.float32)
    # An isolated measured handle extremum between uniform decimation indices.
    points[123] = [0.07, 0.03, 0.02]
    candidates, _ = search_heightmap_placements(
        points, destination, destination, 'upright', candidate_count=4,
        resolution=0.005, upright_yaw_step_deg=90, bottom_points=bottom)
    assert candidates
    for candidate in candidates:
        placed = (points - points.mean(axis=0)) @ candidate.rotation.T + candidate.translation
        assert np.all(placed[:, :2] >= -0.098 - 1e-6)
        assert np.all(placed[:, :2] <= 0.098 + 1e-6)


def test_tray_rim_does_not_replace_corner_preference_with_object_adjacency():
    axis = np.linspace(-0.1, 0.1, 41)
    tray = np.array([[x, y, 0.02 if max(abs(x), abs(y)) > 0.095 else 0.0]
                     for x in axis for y in axis], dtype=np.float32)
    object_axis = np.linspace(-0.015, 0.015, 7)
    bottom = np.array([[x, y, 0.0] for x in object_axis for y in object_axis], np.float32)
    points = np.vstack((bottom, bottom + [0.0, 0.0, 0.04]))
    candidates, _ = search_heightmap_placements(
        points, tray, tray, 'upright', candidate_count=1,
        resolution=0.005, upright_yaw_step_deg=90, bottom_points=bottom)
    assert candidates
    assert np.min(np.abs(candidates[0].translation[:2])) > 0.06


def test_measured_support_plane_rejects_depth_noise_below_object_bottom():
    axis = np.linspace(-0.1, 0.1, 41)
    tray = np.array([[x, y, 0.0] for x in axis for y in axis], np.float32)
    object_axis = np.linspace(-0.02, 0.02, 9)
    bottom = np.array([[x, y, 0.0] for x in object_axis for y in object_axis], np.float32)
    points = np.vstack((bottom, bottom + [0.0, 0.0, 0.04], [[0.018, 0.018, -0.02]]))
    candidates, diagnostics = search_heightmap_placements(
        points, tray, tray, 'upright', candidate_count=1,
        resolution=0.005, upright_yaw_step_deg=90, bottom_points=bottom)
    assert candidates
    assert diagnostics.stable > 0
    placed_bottom = (bottom - points.mean(axis=0)) @ candidates[0].rotation.T + candidates[0].translation
    np.testing.assert_allclose(placed_bottom[:, 2], 0.002, atol=1e-6)


def test_metric_edge_clearance_on_rotated_destination():
    angle = np.deg2rad(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    xs, ys = np.linspace(-0.103, 0.097, 41), np.linspace(-0.082, 0.118, 41)
    destination_xy = np.array([[x, y] for x in xs for y in ys]) @ rotation.T
    tray = np.column_stack((destination_xy, np.zeros(len(destination_xy)))).astype(np.float32)
    object_axis = np.linspace(-0.02, 0.02, 9)
    bottom = np.array([[x, y, 0.0] for x in object_axis for y in object_axis], np.float32)
    points = np.vstack((bottom, bottom + [0.0, 0.0, 0.04]))
    candidates, _ = search_heightmap_placements(
        points, tray, tray, 'upright', candidate_count=4,
        resolution=0.005, upright_yaw_step_deg=90, bottom_points=bottom)
    assert candidates
    for candidate in candidates:
        placed = (points - points.mean(0)) @ candidate.rotation.T + candidate.translation
        xy = placed[:, :2] @ rotation
        assert np.all(xy >= np.array([-0.101, -0.080]) - 1e-6)
        assert np.all(xy <= np.array([0.095, 0.116]) + 1e-6)
