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
    request = CommitFloorSlots.Request(selected_pose=pose)
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
            obj, floor, floor, 'upright', support_mode='floor_slots', candidate_count=16)
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
    a = PlacementCandidate(np.eye(3), np.array([.1, .2, .03]), 1)
    b = PlacementCandidate(np.eye(3), np.array([.3, .2, .03]), 1)
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


def test_contained_footprint_still_reserves_overlapping_neighbor_territory():
    """A neighboring region's gap can extend inside a physical cell."""
    grid = FloorSlots(np.zeros(2), np.eye(2), np.array([[0, 0], [0, 1]]),
                      bounds=np.array([[0, 0, .125, .125], [.127, 0, .252, .125]]),
                      cell_gaps=np.array([[0, 0], [.02, 0]]))
    footprint = filled_rectangle(.116, .116)
    trials = grid.trials(footprint, clearance=.002)
    assert any(used == {(0, 0), (0, 1)} for _, used in trials)
    assert not grid.trials(footprint, reserved=((0, 1),), clearance=.002)
