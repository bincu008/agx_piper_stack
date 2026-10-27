"""Floor grid grouping, gap handling and immutable reservation regression tests."""

import numpy as np

from placement_node.floor_slots import FloorSlots
from placement_node.packing import search_heightmap_placements


def grid():
    return FloorSlots.build(
        np.array([[0, 0], [.34, 0], [.34, .34], [0, .34]], np.float32), size=.07, gap=.01)


def filled_rectangle(width, height):
    x, y = np.meshgrid(np.linspace(-width/2, width/2, 40),
                       np.linspace(-height/2, height/2, 40))
    return np.column_stack((x.ravel(), y.ravel()))


def test_grid_dimensions_and_single_cell_fit():
    g = grid()
    assert len(g.cells) == 16
    assert np.isclose(np.linalg.norm(g.vertices((0, 0))[1]-g.vertices((0, 0))[0]), .07)
    trials = g.trials(filled_rectangle(.04, .04))
    assert len(trials) == 16 and all(len(cells) == 1 for _, cells in trials)
    blocked = trials[0][1]
    assert all(not cells & blocked for _, cells in g.trials(filled_rectangle(.04, .04), blocked))


def test_rectangular_and_square_groups_span_gap():
    g = grid()
    assert all(len(cells) == 2 for _, cells in g.trials(filled_rectangle(.12, .04)))
    assert all(len(cells) == 4 for _, cells in g.trials(filled_rectangle(.12, .12)))


def test_l_shape_reserves_actual_pattern_and_roundtrip():
    g = grid()
    p = filled_rectangle(.12, .12)
    p = p[(p[:, 0] < -.005) | (p[:, 1] < -.005)]
    trials = g.trials(p, clearance=0)
    assert trials and len(trials[0][1]) == 3
    assert g.to_dict() == FloorSlots.from_dict(g.to_dict()).to_dict()


def test_no_fit_and_fully_reserved():
    g = grid()
    assert not g.trials(filled_rectangle(.4, .4))
    assert not g.trials(filled_rectangle(.04, .04), g.cells)


def test_grid_search_keeps_floor_support_checks():
    x, y = np.meshgrid(np.linspace(0, .34, 70), np.linspace(0, .34, 70))
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
    footprint = filled_rectangle(.03, .03)
    obj = np.column_stack((footprint, np.ones(len(footprint))*.04)).astype(np.float32)
    candidates, diagnostic = search_heightmap_placements(
        obj, floor, floor, 'upright', support_mode='floor_slots', candidate_count=16)
    assert candidates and diagnostic.slot_layout and diagnostic.slot_candidate_cells
    assert all(len(cells) == 1 for cells in diagnostic.slot_candidate_cells)
    blocked = diagnostic.slot_layout['cells']
    candidates, _ = search_heightmap_placements(
        obj, floor, floor, 'upright', support_mode='floor_slots',
        floor_slot_reserved_cells=blocked)
    assert not candidates


def test_rotated_destination_keeps_grid_aligned_and_contained():
    angle = np.deg2rad(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    boundary = np.array([[0, 0], [.34, 0], [.34, .26], [0, .26]]) @ rotation.T + [.3, -.2]
    g = FloorSlots.build(boundary, size=.07, gap=.01)
    assert len(g.cells) == 12
    for cell in g.cells:
        for vertex in g.vertices(cell):
            import cv2
            assert cv2.pointPolygonTest(boundary.astype(np.float32), tuple(vertex), True) >= .0019


def test_reservations_commit_selected_pose_only_and_are_idempotent():
    from collections import deque
    from types import SimpleNamespace
    import threading
    from my_actions.srv import CommitFloorSlots
    from std_msgs.msg import Header
    from geometry_msgs.msg import PoseStamped
    from placement_node.inference import PlacementCandidate
    from placement_node.node import PlacementNode
    header = Header(frame_id='base_link')
    header.stamp.sec = 13
    candidate = PlacementCandidate(np.eye(3), np.array([.1, .2, .03]), 1)
    entry = {'layout': grid().to_dict(), 'reserved': set(), 'commits': set()}
    published = []
    node = SimpleNamespace(_slot_lock=threading.Lock(), _slot_layouts={'key': entry},
                           _slot_pending=deque([(header, 'key', [candidate], [[[0, 0]]], 0)]),
                           _pose_matrix=PlacementNode._pose_matrix,
                           _publish_floor_slots=lambda *args: published.append(args))
    pose = PoseStamped(header=header)
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = candidate.translation
    pose.pose.orientation.w = 1.0
    request = CommitFloorSlots.Request(selected_pose=pose, validate_only=True)
    preview = PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response())
    assert preview.success and preview.status == 'FLOOR_SLOTS_AVAILABLE'
    assert not entry['reserved'] and not entry['commits']
    request.validate_only = False
    for _ in range(2):
        response = PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response())
        assert response.success and response.occupied_slots == 1
    assert entry['reserved'] == {(0, 0)}
    rotated = PlacementCandidate(np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
                                 candidate.translation.copy(), 1)
    node._slot_pending.append((header, 'key', [rotated], [[[0, 0], [1, 0]]], 0))
    request.selected_pose.pose.orientation.z = float(np.sqrt(.5))
    request.selected_pose.pose.orientation.w = float(np.sqrt(.5))
    response = PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response())
    assert not response.success and response.status == 'FLOOR_SLOT_RESERVATION_CONFLICT'

    request.selected_pose.header.stamp.sec += 1
    response = PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response())
    assert not response.success


def test_layout_key_accepts_xyz_cloud_slices():
    from placement_node.floor_slots import layout_key
    cloud = np.array([[0, 0, 0], [.34, 0, 0], [.34, .34, 0], [0, .34, 0]], np.float64)
    assert layout_key(cloud) == layout_key(cloud.astype(np.float32))


def test_fast_stencil_encoding_preserves_negative_coordinates():
    points = np.array([[-23, -45], [23, 45], [-23, -45], [-23, 45], [0, 0]])
    assert FloorSlots.unique_pairs(points) == set(map(tuple, points))


def test_phase_offsets_add_positions_without_losing_grid_containment():
    g = grid()
    footprint = filled_rectangle(.12, .10)
    centered = g.trials(footprint)
    shifted = g.trials(footprint, phase_offsets=(0., -.02, .02))
    assert len(shifted) > len(centered)
    assert all(cells <= set(map(tuple, g.cells)) for _, cells in shifted)


def test_shared_cells_require_disjoint_footprint_reservations():
    from collections import deque
    from types import SimpleNamespace
    import threading
    from my_actions.srv import CommitFloorSlots
    from std_msgs.msg import Header
    from geometry_msgs.msg import PoseStamped
    from placement_node.inference import PlacementCandidate
    from placement_node.node import PlacementNode
    h = Header(frame_id='base_link')
    a = PlacementCandidate(np.eye(3), np.array([.1, .2, .03]), 1)
    b = PlacementCandidate(np.eye(3), np.array([.14, .2, .03]), 1)
    c = PlacementCandidate(np.eye(3), np.array([.15, .2, .03]), 1)
    entry = {'layout': grid().to_dict(), 'reserved': set(), 'commits': set()}
    pending = deque([(h, 'key', [a, b, c], [[[0, 0]]]*3, 0, True,
                      [{(1, 1)}, {(2, 2)}, {(1, 1)}])])
    node = SimpleNamespace(_slot_lock=threading.Lock(), _slot_layouts={'key': entry},
                           _slot_pending=pending, _pose_matrix=PlacementNode._pose_matrix,
                           _publish_floor_slots=lambda *args: None)
    outcomes = []
    for candidate in (a, b, c):
        pose = PoseStamped(header=h)
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = candidate.translation
        pose.pose.orientation.w = 1.0
        outcomes.append(PlacementNode._commit_floor_slots(
            node, CommitFloorSlots.Request(selected_pose=pose),
            CommitFloorSlots.Response()).success)
    assert outcomes == [True, True, False]
    assert entry['reserved'] == {(0, 0)} and entry['occupied_xy'] == {(1, 1), (2, 2)}


def test_shared_search_rejects_committed_footprint_before_shortlisting():
    x, y = np.meshgrid(np.linspace(0, .34, 70), np.linspace(0, .34, 70))
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
    footprint = filled_rectangle(.04, .04)
    obj = np.column_stack((footprint, np.ones(len(footprint))*.04)).astype(np.float32)
    first, _ = search_heightmap_placements(
        obj, floor, floor, 'upright', support_mode='floor_slots')

    def mask(candidate):
        xy = ((obj-obj.mean(axis=0)) @ candidate.rotation.T+candidate.translation)[:, :2]
        keys = FloorSlots.unique_pairs(np.floor(xy/.005).astype(int))
        return {(x+dx, y+dy) for x, y in keys for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    occupied = mask(first[0])
    candidates, _ = search_heightmap_placements(
        obj, floor, floor, 'upright', support_mode='floor_slots',
        floor_slot_share_unused_space=True, floor_slot_occupied_xy=list(occupied))
    assert candidates and all(not mask(c) & occupied for c in candidates)


def test_default_twelve_point_five_cm_grid_keeps_whole_cell_exclusivity():
    boundary = np.array([[0, 0], [.34, 0], [.34, .34], [0, .34]], np.float32)
    g = FloorSlots.build(boundary)
    assert g.size == .125 and g.gap == .005
    assert len(g.cells) == 4
    footprint = filled_rectangle(.08, .08)
    trials = g.trials(footprint)
    assert trials
    reserved = trials[0][1]
    assert len(reserved) == 1
    remaining = g.trials(footprint, reserved)
    assert remaining and all(not cells & reserved for _, cells in remaining)
    assert np.isclose(np.linalg.norm(g.vertices((0, 0))[1]-g.vertices((0, 0))[0]), .125)


def test_twelve_point_five_cm_search_prefers_one_slot_and_falls_back_to_two():
    x, y = np.meshgrid(np.linspace(0, .34, 70), np.linspace(0, .34, 70))
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
    for width, height, expected in ((.08, .08, 1), (.16, .055, 2)):
        footprint = filled_rectangle(width, height)
        obj = np.column_stack((footprint, np.ones(len(footprint))*.04)).astype(np.float32)
        candidates, diagnostic = search_heightmap_placements(
            obj, floor, floor, 'upright', support_mode='floor_slots', candidate_count=16,
            floor_slot_large_square_scale=0)
        assert candidates
        counts = [len(cells) for cells in diagnostic.slot_candidate_cells]
        assert counts[0] == expected
        assert counts == sorted(counts)


def test_area_alone_does_not_allow_single_slot_overhang():
    boundary = np.array([[0, 0], [.34, 0], [.34, .34], [0, .34]], np.float32)
    g = FloorSlots.build(boundary)
    # Less area than a square, but its long side does not fit the square.
    footprint = filled_rectangle(.16, .055)
    assert .16*.055 < g.size*g.size
    trials = g.trials(footprint)
    assert trials and min(len(cells) for _, cells in trials) == 2


def test_slot_yaw_variants_survive_small_recentering_shift():
    from placement_node.packing import _select_diverse_candidates, _rotation
    from placement_node.inference import PlacementCandidate
    proposals = [PlacementCandidate(_rotation(0, 0, np.deg2rad(angle)),
                                    np.array([.3 + .012*np.sin(np.deg2rad(angle)),
                                              .1, .04], np.float32), -float(i))
                 for i, angle in enumerate((0, 60, 120, 180, 240, 300))]
    selected = _select_diverse_candidates(proposals, 16, preserve_yaw=True)
    assert len(selected) == 6


def test_exclusive_commit_lazily_keeps_selected_geometry_only():
    from collections import deque
    from types import SimpleNamespace
    import threading
    from my_actions.srv import CommitFloorSlots
    from std_msgs.msg import Header
    from geometry_msgs.msg import PoseStamped
    from placement_node.inference import PlacementCandidate
    from placement_node.node import PlacementNode
    h = Header(frame_id='base_link')
    g = grid()
    a = PlacementCandidate(np.eye(3), np.r_[g.vertices((0, 0)).mean(axis=0), .03], 1)
    b = PlacementCandidate(np.eye(3), np.r_[g.vertices((1, 0)).mean(axis=0), .03], 1)
    source = np.array([[0, 0, 0], [.005, .005, 0]])
    entry = {'layout': grid().to_dict(), 'reserved': set(), 'commits': set()}
    node = SimpleNamespace(_slot_lock=threading.Lock(), _slot_layouts={'key': entry},
                           _slot_pending=deque([(h, 'key', [a, b], [[[0, 0]], [[1, 0]]],
                                                0, False, [], source)]),
                           _pose_matrix=PlacementNode._pose_matrix,
                           _publish_floor_slots=lambda *args: None)
    pose = PoseStamped(header=h)
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = a.translation
    pose.pose.orientation.w = 1.
    request = CommitFloorSlots.Request(selected_pose=pose)
    for _ in range(2):
        result = PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response())
        assert result.success and result.occupied_slots == 1
    assert entry['reserved'] == {(0, 0)}
    assert entry['occupied_xy'] == FloorSlots.reservation_mask(source, a.rotation, a.translation)
    assert not entry['occupied_xy'] & FloorSlots.reservation_mask(
        source, b.rotation, b.translation)


def test_adaptive_two_rows_and_small_surface_single_row():
    for short, expected_rows in ((.254, 2), (.09, 1)):
        boundary = np.array([[0, 0], [.254, 0], [.254, short], [0, short]], np.float32)
        g = FloorSlots.build_adaptive(boundary)
        assert len(np.unique(g.cells[:, 1])) == expected_rows
        dimensions = g.bounds[:, 2:]-g.bounds[:, :2]
        assert np.all(dimensions >= .06-1e-8) and np.all(dimensions <= .125+1e-8)
        assert np.all(g.cell_gaps >= 0)
        assert np.allclose(dimensions[:-1, 0], dimensions[:-1, 1])
        assert g.to_dict() == FloorSlots.from_dict(g.to_dict()).to_dict()


def test_adaptive_l_surface_adds_nonoverlapping_extension():
    import cv2
    boundary = np.array([[0, 0], [.4, 0], [.4, .15], [.25, .15],
                         [.25, .3], [0, .3]], np.float32)
    g = FloorSlots.build_adaptive(boundary)
    assert np.max(g.bounds[:, 2]) > .35
    assert np.max(g.bounds[:, 3]) > .25
    for i, cell in enumerate(g.cells):
        assert all(cv2.pointPolygonTest(boundary, tuple(map(float, v)), True) >= .001
                   for v in g.vertices(cell))
        for j in range(i):
            overlap = np.minimum(g.bounds[i, 2:], g.bounds[j, 2:])-np.maximum(
                g.bounds[i, :2], g.bounds[j, :2])
            assert not np.all(overlap > 1e-8)


def test_adaptive_occupied_bowl_keeps_exactly_three_free_slots():
    boundary = np.array([[0, 0], [.254, 0], [.254, .254], [0, .254]], np.float32)
    g = FloorSlots.build_adaptive(boundary)
    assert len(g.cells) == 4
    footprint = filled_rectangle(.116, .116)
    initial = g.trials(footprint)
    assert initial and len(initial[0][1]) == 1
    reserved = initial[0][1]
    frozen = g.to_dict()
    restored = FloorSlots.from_dict(frozen)
    remaining = restored.trials(footprint, reserved)
    assert restored.to_dict() == frozen
    assert len(set(map(tuple, g.cells))-reserved) == 3
    assert {next(iter(cells)) for _, cells in remaining} == set(map(tuple, g.cells))-reserved


def test_adaptive_heightmap_feedback_preserves_three_free_slots():
    x, y = np.meshgrid(np.linspace(0, .254, 100), np.linspace(0, .254, 100))
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
    xy = filled_rectangle(.116, .116)
    xy = xy[np.linalg.norm(xy, axis=1) <= .058]
    bowl = np.vstack((np.column_stack((xy, np.zeros(len(xy)))),
                      np.column_stack((xy, np.full(len(xy), .03))))).astype(np.float32)
    candidates, first = search_heightmap_placements(
        bowl, floor, floor, 'upright', support_mode='floor_slots')
    assert candidates and len(first.slot_layout['cells']) == 4
    assert len(first.slot_candidate_cells[0]) == 1
    chosen = candidates[0]
    placed = (bowl-bowl.mean(axis=0)) @ chosen.rotation.T+chosen.translation
    probe_xy = filled_rectangle(.03, .03)
    probe = np.column_stack((probe_xy, np.full(len(probe_xy), .03))).astype(np.float32)
    candidates, second = search_heightmap_placements(
        probe, np.vstack((floor, placed)), floor, 'upright', support_mode='floor_slots',
        floor_slot_layout=first.slot_layout)
    assert candidates and second.slot_layout == first.slot_layout
    all_cells = set(map(tuple, second.slot_layout['cells']))
    blocked = all_cells & set(map(tuple, second.slot_blocked_cells))
    assert len(blocked) == 1 and len(all_cells-blocked) == 3
    assert all(not set(map(tuple, cells)) & blocked for cells in second.slot_candidate_cells)


def test_square_rows_only_add_full_size_rows_and_dynamic_gaps():
    for depth, rows, size, gap in ((.28, 2, .125, .03),
                                   (.20, 2, .10, 0),
                                   (.119, 1, .119, 0),
                                   (.40, 3, .125, .0125),
                                   (.52, 4, .125, .0066666667)):
        widths, count, side, gaps = FloorSlots._square_region(.40, depth, .125, .06)
        assert count == rows and np.isclose(side, size)
        assert np.isclose(gaps[1], gap)
        assert all(np.isclose(w, side) for w in widths[:-1])
        assert not widths or widths[-1] >= .06


def test_square_rows_keep_rectangular_remainder_only():
    widths, rows, side, gaps = FloorSlots._square_region(.342, .266, .125, .06)
    assert rows == 2 and side == .125
    assert np.allclose(widths, [.125, .125, .092])
    assert np.allclose(gaps, [0, .016])
    widths, rows, side, gaps = FloorSlots._square_region(.29, .266, .125, .06)
    assert widths == [.125, .125] and np.isclose(gaps[0], .04)


def test_contained_footprint_keeps_its_slot_when_neighbor_gap_extends_inside():
    """Gap territories must not steal an otherwise fully contained physical slot."""
    grid = FloorSlots(np.zeros(2), np.eye(2), np.array([[0, 0], [0, 1]]),
                      bounds=np.array([[0, 0, .125, .125], [.127, 0, .252, .125]]),
                      cell_gaps=np.array([[0, 0], [.02, 0]]))
    footprint = filled_rectangle(.116, .116)
    trials = grid.trials(footprint, clearance=.002)
    assert trials and all(len(used) == 1 for _, used in trials)
    remaining = grid.trials(footprint, reserved=((0, 1),), clearance=.002)
    assert remaining and all(used == {(0, 0)} for _, used in remaining)
    spanning = filled_rectangle(.14, .1)
    assert not grid.trials(spanning, reserved=((0, 1),), clearance=.002)


def test_large_square_preserves_two_normal_slots_and_exclusive_reservation():
    boundary = np.array([[0, 0], [.35, 0], [.35, .274], [0, .274]], np.float32)
    g = FloorSlots.build_adaptive(boundary)
    assert g.large_cell is not None
    dimensions = g.bounds[:, 2:]-g.bounds[:, :2]
    large_index = list(map(tuple, g.cells)).index(g.large_cell)
    assert np.allclose(dimensions[large_index], .1875)
    assert np.count_nonzero(np.all(np.isclose(dimensions, .125), axis=1)) >= 2
    for i, box in enumerate(g.bounds):
        for other in g.bounds[:i]:
            assert not np.all(np.minimum(box[2:], other[2:]) -
                              np.maximum(box[:2], other[:2]) > 1e-8)
    restored = FloorSlots.from_dict(g.to_dict())
    assert restored.to_dict() == g.to_dict()
    trials = g.trials(filled_rectangle(.16, .16), clearance=.002)
    assert any(cells == {g.large_cell} for _, cells in trials)
    assert all(g.large_cell not in cells for _, cells in g.trials(
        filled_rectangle(.16, .16), {g.large_cell}, clearance=.002))


def test_large_square_scale_and_no_fit_fallback():
    boundary = np.array([[0, 0], [.35, 0], [.35, .274], [0, .274]], np.float32)
    ordinary = FloorSlots.build_adaptive(boundary, large_square_scale=0)
    oversized = FloorSlots.build_adaptive(boundary, large_square_scale=3)
    assert oversized.to_dict() == ordinary.to_dict()
    smaller = FloorSlots.build_adaptive(boundary, large_square_scale=1.25)
    index = list(map(tuple, smaller.cells)).index(smaller.large_cell)
    assert np.allclose(smaller.bounds[index, 2:]-smaller.bounds[index, :2], .15625)
    small = np.array([[0, 0], [.254, 0], [.254, .254], [0, .254]], np.float32)
    assert FloorSlots.build_adaptive(small).large_cell is None


def test_large_square_scale_rejects_invalid_values():
    import pytest
    boundary = np.array([[0, 0], [.35, 0], [.35, .274], [0, .274]], np.float32)
    for scale in (-1, .5, 1, np.nan, np.inf):
        with pytest.raises(ValueError):
            FloorSlots.build_adaptive(boundary, large_square_scale=scale)


def test_narrow_extension_keeps_squares_inside_region():
    widths, rows, side, gaps = FloorSlots._square_region(.07, .5, .125, .06)
    assert rows == 2 and side == .07
    assert sum(widths)+(len(widths)-1)*gaps[0] <= .07+1e-8


def test_single_slot_precedes_smaller_area_groups_then_minimum_slot_count():
    cells = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    g = FloorSlots(np.zeros(2), np.eye(2), cells, bounds=np.array([
        [0, 0, .125, .125], [.125, 0, .205, .08],
        [.205, 0, .285, .08], [.3, 0, .4875, .1875]]),
        cell_gaps=np.zeros((4, 2)), large_cell=(3, 0))
    assert g.fit_priority({(0, 0)}) < g.fit_priority({(1, 0), (2, 0)})
    assert g.fit_priority({(3, 0)}) < g.fit_priority({(0, 0), (1, 0), (2, 0)})
    assert g.fit_priority({(1, 0), (2, 0)}) < g.fit_priority({(0, 0), (1, 0), (2, 0)})
    assert g.fit_priority({(1, 0)}) < g.fit_priority({(3, 0)})
    trials = g.trials(filled_rectangle(.14, .06))
    assert trials and trials[0][1] == {(3, 0)}
    single = g.trials(filled_rectangle(.14, .06), single_only=True)
    assert single and all(len(used) == 1 for _, used in single)
    expected_single = [(xy, used) for xy, used in trials if len(used) == 1]
    assert len(single) == len(expected_single)
    for actual, expected in zip(single, expected_single):
        assert np.array_equal(actual[0], expected[0]) and actual[1] == expected[1]
    priorities = [g.fit_priority(used) for _, used in trials]
    assert priorities == sorted(priorities)
    combined = g.trials(filled_rectangle(.14, .06), {g.large_cell})
    assert combined and len(combined[0][1]) == 2
    assert all(len(used) >= 2 for _, used in combined)
    fallback = g.trials(filled_rectangle(.14, .1), {(0, 0), (1, 0), (2, 0)})
    assert fallback and all(used == {(3, 0)} for _, used in fallback)


def test_heightmap_single_slot_count_uses_area_only_to_break_ties():
    x, y = np.meshgrid(np.linspace(0, .35, 85), np.linspace(0, .274, 70))
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
    footprint = filled_rectangle(.09, .09)
    obj = np.column_stack((footprint, np.ones(len(footprint))*.03)).astype(np.float32)
    candidates, diagnostic = search_heightmap_placements(
        obj, floor, floor, 'upright', support_mode='floor_slots')
    assert candidates
    g = FloorSlots.from_dict(diagnostic.slot_layout)
    assert g.large_cell is not None
    assert all(g.large_cell not in set(map(tuple, used))
               for used in diagnostic.slot_candidate_cells)
    reserved = set(map(tuple, g.cells))-{g.large_cell}
    candidates, fallback = search_heightmap_placements(
        obj, floor, floor, 'upright', support_mode='floor_slots',
        floor_slot_layout=g.to_dict(), floor_slot_reserved_cells=reserved)
    assert candidates
    assert all(set(map(tuple, used)) == {g.large_cell}
               for used in fallback.slot_candidate_cells)


def test_single_slot_pass_falls_back_to_fully_checked_combinations():
    x, y = np.meshgrid(np.linspace(0, .30, 70), np.linspace(0, .15, 40))
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
    xy = filled_rectangle(.16, .06)
    obj = np.column_stack((xy, np.full(len(xy), .04))).astype(np.float32)
    g = FloorSlots(np.array([.01, .01]), np.eye(2), np.array([[0, 0], [1, 0]]),
                   bounds=np.array([[0, 0, .125, .125], [.135, 0, .26, .125]]),
                   cell_gaps=np.full((2, 2), .01))
    candidates, diagnostic = search_heightmap_placements(
        obj, floor, floor, 'upright', support_mode='floor_slots',
        floor_slot_layout=g.to_dict(), candidate_count=16)
    assert candidates and diagnostic.orientations == 144
    assert all(len(cells) == 2 for cells in diagnostic.slot_candidate_cells)
    assert diagnostic.stable > 0


def test_continuous_single_slot_fit_preserves_exact_center_without_extra_cells():
    """Raster sampling must not displace a tightly contained inventory fit."""
    x, y = np.meshgrid(np.linspace(-.03, .17, 90), np.linspace(-.03, .17, 90))
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
    g = FloorSlots(np.zeros(2), np.eye(2), np.array([[0, 0]]),
                   bounds=np.array([[.0005, .005, .1255, .13]]),
                   cell_gaps=np.zeros((1, 2)))
    xy = filled_rectangle(.1195, .11)
    obj = np.column_stack((xy, np.full(len(xy), .04))).astype(np.float32)
    candidates, diagnostic = search_heightmap_placements(
        obj, floor, floor, 'upright', support_mode='floor_slots', floor_slot_layout=g.to_dict())
    assert candidates
    assert diagnostic.slot_layout == g.to_dict()
    assert diagnostic.heightmap_resolution_m == .005
    assert all(used == [[0, 0]] for used in diagnostic.slot_candidate_cells)

    for candidate in candidates:
        placed = (obj-obj.mean(axis=0)) @ candidate.rotation.T + candidate.translation
        local = (placed[:, :2]-g.origin) @ g.axes.T
        assert np.all(local.min(axis=0) >= g.bounds[0, :2]+.002-1e-7)
        assert np.all(local.max(axis=0) <= g.bounds[0, 2:]-.002+1e-7)


def test_wide_leftover_gap_does_not_consume_neighbor_of_contained_footprint():
    """Physical containment takes precedence over overlapping gap territories."""
    g = FloorSlots(np.zeros(2), np.eye(2), np.array([[0, 0], [0, 1]]),
                   bounds=np.array([[0, 0, .125, .125], [.125, 0, .195, .07]]),
                   cell_gaps=np.array([[0., 0.], [.06, 0.]]))
    footprint = filled_rectangle(.112, .112)+[.0625, .0625]
    assert g.touched(footprint, .002) == {(0, 0)}
    assert g.touched(np.empty((0, 2)), .002) == set()
    trials = g.trials(footprint-[.0625, .0625])
    assert trials and trials[0][1] == {(0, 0)}


def test_grid_local_membership_matches_world_membership_with_gap_spanning():
    """Coordinate reuse must retain every point, clearance corner and outside cell."""
    angle = .37
    axes = np.array([[np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]])
    g = FloorSlots(np.array([.3, -.2]), axes, np.array([[0, 0], [1, 0], [0, 1]]),
                   bounds=np.array([[0., 0., .1, .1], [.11, 0., .21, .1],
                                    [0., .11, .1, .21]]),
                   cell_gaps=np.full((3, 2), .01))
    rng = np.random.default_rng(77)
    for clearance in (0., .002):
        for _ in range(20):
            local = rng.uniform(-.02, .23, (180, 2))
            world = local @ axes + g.origin
            recovered = (world-g.origin) @ axes.T
            points = np.vstack([recovered+d for d in
                                ((0, 0), (-clearance, -clearance), (-clearance, clearance),
                                 (clearance, -clearance), (clearance, clearance))])
            expected, covered = set(), np.zeros(len(points), bool)
            for cell, box, gap in zip(g.cells, g.bounds, g.cell_gaps):
                inside = (np.all(points >= box[:2]-gap/2-1e-8, axis=1) &
                          np.all(points <= box[2:]+gap/2+1e-8, axis=1))
                if inside.any():
                    expected.add(tuple(map(int, cell)))
                    covered |= inside
            if not covered.all():
                expected.add((-1, -1))
            assert g.touched(world, clearance) == expected
            assert g._touched_local(local, clearance) == expected


def test_achieved_pose_uses_slot_space_instead_of_exact_original_footprint():
    g = grid()
    source = np.column_stack((filled_rectangle(.04, .04), np.zeros(1600)))
    center = np.r_[g.vertices((0, 0)).mean(axis=0), .03]
    angle = np.deg2rad(10)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    assert g.contains_in_cells(source, rotation, center+[.003, 0, 0], {(0, 0)})
    assert not g.contains_in_cells(source, rotation, center+[.03, 0, 0], {(0, 0)})


def test_adjusted_slot_commit_preserves_source_exclusivity_and_dry_run():
    from collections import deque
    from types import SimpleNamespace
    import threading
    from my_actions.srv import CommitFloorSlots
    from std_msgs.msg import Header
    from geometry_msgs.msg import PoseStamped
    from placement_node.inference import PlacementCandidate
    from placement_node.node import PlacementNode
    g = grid()
    header = Header(frame_id='base_link')
    header.stamp.sec = 91
    center = np.r_[g.vertices((0, 0)).mean(axis=0), .03]
    candidate = PlacementCandidate(np.eye(3), center, 1)
    source = np.column_stack((filled_rectangle(.04, .04), np.zeros(1600)))
    entry = {'layout': g.to_dict(), 'reserved': set(), 'commits': set()}
    node = SimpleNamespace(_slot_lock=threading.Lock(), _slot_layouts={'key': entry},
                           _slot_pending=deque([(header, 'key', [candidate],
                                                 [[(0, 0)]], 0, False, [], source)]),
                           _pose_matrix=PlacementNode._pose_matrix,
                           _publish_floor_slots=lambda *args: None)
    pose = PoseStamped(header=header)
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = center+[.003, 0, .003]
    pose.pose.orientation.z = float(np.sin(np.deg2rad(5)))
    pose.pose.orientation.w = float(np.cos(np.deg2rad(5)))
    request = CommitFloorSlots.Request(selected_pose=pose, validate_only=True)
    assert PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response()).success
    assert not entry['reserved']
    request.validate_only = False
    assert PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response()).success
    assert PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response()).success
    assert entry['reserved'] == {(0, 0)}
    request.selected_pose.pose.position.x += .002
    assert not PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response()).success
    request.selected_pose.header.stamp.sec += 1
    assert not PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response()).success


def test_committed_numpy_cell_indices_have_json_safe_memory():
    """Exercise the real post-commit memory response that crashed after release."""
    import json
    import threading
    from collections import deque
    from types import SimpleNamespace
    from geometry_msgs.msg import PoseStamped
    from my_actions.srv import CommitFloorSlots, GetManipulationMemory
    from placement_node.node import PlacementNode
    from placement_node.inference import PlacementCandidate
    from std_msgs.msg import Header

    header = Header(frame_id='base_link')
    header.stamp.sec = 123
    candidate = PlacementCandidate(np.eye(3), np.array([.1, .2, .03]), 1)
    entry = {'layout': grid().to_dict(), 'reserved': set(), 'commits': set()}
    used = [[np.int64(0), np.int64(1)]]
    node = SimpleNamespace(
        _slot_lock=threading.Lock(), _slot_layouts={'key': entry},
        _slot_pending=deque([(header, 'key', [candidate], [used], 0)]),
        _pose_matrix=PlacementNode._pose_matrix, _publish_floor_slots=lambda *args: None)
    pose = PoseStamped(header=header)
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = candidate.translation
    pose.pose.orientation.w = 1.0
    request = CommitFloorSlots.Request(selected_pose=pose)
    for _ in range(2):
        result = PlacementNode._commit_floor_slots(node, request, CommitFloorSlots.Response())
        assert result.success
        response = PlacementNode._get_floor_slots(
            node, GetManipulationMemory.Request(), GetManipulationMemory.Response())
        assert json.loads(response.state_json)[0]['reserved_cells'] == [[0, 1]]
    assert all(type(value) is int for cell in entry['reserved'] for value in cell)
    # Existing cache entries from before normalization must also be readable.
    entry['reserved'] = {(np.int64(0), np.int64(1))}
    response = PlacementNode._get_floor_slots(
        node, GetManipulationMemory.Request(), GetManipulationMemory.Response())
    assert json.loads(response.state_json)[0]['reserved_cells'] == [[0, 1]]
