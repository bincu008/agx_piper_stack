"""Verify fixture geometry and edge-to-stack support selection."""

import numpy as np
import pytest

from placement_node.box_fixture import box_fixture, box_surface, box_top, box_contact_seeds, tray_floor_mask
from placement_node.packing import search_heightmap_placements
from placement_node.perception import extract_destination_cloud


def tray_surface():
    xy = np.linspace(0, 0.10, 21)
    xx, yy = np.meshgrid(xy, xy)
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))


def test_repeated_runs_release_completed_contexts_but_preserve_busy_context():
    """Do not exhaust master storage or discard a still-running operation."""
    from types import SimpleNamespace
    from placement_node.box_benchmark import BoxBenchmark
    from my_actions.srv import ClearStepContext

    cleared = []

    def clear(key, request):
        assert key == 'clear_context'
        cleared.append(request.context_id)
        return ClearStepContext.Response(
            success=request.context_id != 'busy', status='CONTEXT_BUSY')

    node = SimpleNamespace(_step_contexts=['finished', 'busy', 'later'], _call=clear)
    with pytest.raises(RuntimeError, match='BENCHMARK_CONTEXT_CLEANUP_FAILED'):
        BoxBenchmark._release_step_contexts(node)
    assert cleared == ['finished', 'busy']
    assert node._step_contexts == ['busy', 'later']


def test_ten_boxes_are_separated_complete_and_within_radius():
    boxes = box_fixture()
    assert len(boxes) == 10
    for box in boxes:
        np.testing.assert_allclose(np.ptp(box, axis=0), [0.08]*3, atol=1e-7)
        assert np.max(np.linalg.norm(box[:, :2], axis=1)) <= 0.500001
        assert abs(box[:, 2].min()) < 1e-7
        assert box[:, 1].min() > 0
        assert len(box_top(box)) >= 250
    for index, box in enumerate(boxes):
        for other in boxes[index+1:]:
            gap = np.maximum(other.min(0)-box.max(0), box.min(0)-other.max(0))
            assert max(gap[:2]) >= 0.039999


@pytest.mark.parametrize('yaw_step', [90, 15, 5])
def test_full_floor_rejects_edge_and_accepts_named_stack_top(yaw_step):
    tray = tray_surface()
    source = box_fixture(1)[0]
    previous = box_surface([0.05, 0.05, 0.04])
    scene = np.vstack((tray, previous))
    kwargs = dict(candidate_count=8, upright_yaw_step_deg=yaw_step,
                  bottom_points=source[source[:, 2] < 1e-6])
    edge, _ = search_heightmap_placements(
        source, scene, tray, 'upright', support_mode='edge', **kwargs)
    assert not edge
    stacked, _ = search_heightmap_placements(
        source, scene, tray, 'upright', support_mode='stack',
        stack_support_points=box_top(previous), **kwargs)
    assert stacked
    assert all(candidate.translation[2] >= 0.12 for candidate in stacked)


def test_stack_requires_explicit_support_and_respects_height_limit():
    tray = tray_surface()
    source = box_fixture(1)[0]
    previous = box_surface([0.05, 0.05, 0.04])
    with pytest.raises(ValueError, match='explicit support'):
        search_heightmap_placements(source, tray, tray, 'upright', support_mode='stack')
    candidates, _ = search_heightmap_placements(
        source, np.vstack((tray, previous)), tray, 'upright',
        support_mode='stack', stack_support_points=box_top(previous),
        maximum_pile_height_m=0.10, upright_yaw_step_deg=90)
    assert not candidates


def test_base_frame_tray_at_zero_height_is_valid():
    yy, xx = np.mgrid[:10, :10]
    xyz = np.stack((xx*0.005, yy*0.005, np.zeros_like(xx)), axis=-1)
    points, _, _ = extract_destination_cloud(
        xyz, (10, 10), (0, 0, 10, 10), 0.02, 80,
        coordinates_are_base_link=True)
    assert len(points) == 100


def test_known_box_seeds_are_flat_wall_contacts_with_orthogonal_approach():
    source = box_fixture(1)[0]
    for left, right, approach in box_contact_seeds(source):
        span = left - right
        assert np.linalg.norm(span) == pytest.approx(0.08, abs=1e-7)
        assert float(span @ approach) == pytest.approx(0)
        assert min(np.linalg.norm(source-left, axis=1)) < 0.001
        assert min(np.linalg.norm(source-right, axis=1)) < 0.001
        assert left[2] > 0.05


def test_tray_support_labels_follow_slope_and_exclude_raised_surfaces():
    yy, xx = np.mgrid[:20, :20]
    floor = np.column_stack((xx.ravel()*0.01, yy.ravel()*0.01,
                             xx.ravel()*0.001))
    raised = floor[::5] + [0, 0, 0.04]
    mask = tray_floor_mask(np.vstack((floor, raised)))
    assert mask[:len(floor)].all()
    assert not mask[len(floor):].any()


def test_floor_placements_reserve_gap_from_existing_box():
    xy = np.arange(0, 0.251, 0.005)
    xx, yy = np.meshgrid(xy, xy)
    tray = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    source = box_fixture(1)[0]
    previous = box_surface([0.05, 0.05, 0.04])
    candidates, _ = search_heightmap_placements(
        source, np.vstack((tray, previous)), tray, 'upright',
        candidate_count=8, support_mode='edge', obstacle_clearance_m=0.006,
        upright_yaw_step_deg=90, bottom_points=source[source[:, 2] < 1e-6])
    assert candidates
    for candidate in candidates:
        placed = (source-source.mean(0)) @ candidate.rotation.T + candidate.translation
        gap = np.maximum(placed.min(0)-previous.max(0), previous.min(0)-placed.max(0))
        assert np.linalg.norm(np.maximum(gap[:2], 0)) >= 0.006 - 1e-6


@pytest.mark.parametrize('tilt', [15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0])
def test_tilted_seeds_keep_closing_axis_normal_to_flat_walls(tilt):
    source = box_fixture(1)[0]
    for left, right, approach in box_contact_seeds(source, tilt):
        closing = (left-right) / np.linalg.norm(left-right)
        assert abs(float(closing @ approach)) < 1e-7
        assert np.linalg.norm(approach) == pytest.approx(1)
        assert approach[2] < 0
        # Tangential approach tilt cannot introduce an oblique jaw closing axis.
        assert np.count_nonzero(abs(closing) > 1e-6) == 1


def test_segmented_tray_excludes_connected_table_and_preserves_measured_points():
    from placement_node.perception import extract_segmented_destination_cloud
    yy, xx = np.mgrid[:12, :16]
    xyz = np.stack((xx*0.001, yy*0.001, np.zeros_like(xx)), axis=-1)
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:10, 4:12] = True
    xyz[4, 6] = np.nan
    points, foreground, crop = extract_segmented_destination_cloud(xyz, mask, 60)
    assert len(points) == 63
    assert crop == (0, 0, 16, 12)
    assert not foreground[0, 0]
    np.testing.assert_allclose(points, xyz[foreground], atol=1e-8)
    with pytest.raises(ValueError, match='match organized'):
        extract_segmented_destination_cloud(xyz, mask[:8], 1)
    with pytest.raises(ValueError, match='insufficient'):
        extract_segmented_destination_cloud(xyz, np.zeros_like(mask), 1)


@pytest.mark.parametrize('stage', ['COMPLETE', 'COMPLETE_WITH_FAILURES', 'FAILED', 'CANCELLED'])
def test_run_service_waits_for_worker_and_returns_terminal_result(stage):
    import threading
    from types import SimpleNamespace
    from std_srvs.srv import Trigger
    from placement_node.box_benchmark import BoxBenchmark

    entered, finish, returned = (threading.Event() for _ in range(3))
    node = SimpleNamespace(_lock=threading.Lock(), _cancel=threading.Event(),
                           _worker=None, _run_service_active=False, _report={})

    def run():
        entered.set()
        assert finish.wait(timeout=5)
        node._report = dict(stage=stage, accepted=10 if stage == 'COMPLETE' else 2,
                            elapsed_seconds=1.0)
        if stage != 'COMPLETE':
            node._report['failure'] = 'test termination'

    node._run = run
    response = Trigger.Response()

    def call():
        BoxBenchmark._start(node, Trigger.Request(), response)
        returned.set()

    caller = threading.Thread(target=call)
    caller.start()
    try:
        assert entered.wait(timeout=2)
        assert not returned.is_set()
        busy = BoxBenchmark._start(node, Trigger.Request(), Trigger.Response())
        assert not busy.success and 'BUSY' in busy.message
        BoxBenchmark._stop(node, Trigger.Request(), Trigger.Response())
        assert node._cancel.is_set()
    finally:
        finish.set()
        caller.join(timeout=5)
    assert returned.is_set()
    assert response.success == (stage == 'COMPLETE')
    assert stage in response.message
    assert not node._run_service_active


@pytest.mark.parametrize('failure_phase', ['grasp', 'placement', 'cancel'])
def test_failed_box_is_retained_and_later_stack_uses_actual_support_name(
        monkeypatch, tmp_path, failure_phase):
    import threading
    import time
    from types import SimpleNamespace
    from cv_bridge import CvBridge
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
    from my_actions.msg import GraspCandidate
    import placement_node.box_benchmark as module

    header = Header(frame_id='base_link')
    header.stamp.sec = 123
    cloud = point_cloud2.create_cloud_xyz32(header, np.zeros((256, 3)).tolist())
    cloud.height, cloud.width, cloud.row_step = 16, 16, 16*12
    bridge = CvBridge()
    image = bridge.cv2_to_imgmsg(np.zeros((16, 16, 3), np.uint8), 'bgr8')
    image.header = header
    mask = bridge.cv2_to_imgmsg(np.ones((16, 16), np.uint8)*255, 'mono8')
    mask.header = header
    boxes = [box_surface([0.1*i, 0.3, 0.04], spacing=0.04) for i in range(3)]
    monkeypatch.setattr(module, 'box_fixture', lambda *a, **k: boxes)
    monkeypatch.setattr(module, 'tray_floor_mask', lambda p: np.ones(len(p), bool))
    monkeypatch.setattr(module.PlacementNode, '_publish_layout_markers', lambda *a: None)
    params = dict(box_count=3, box_size_m=.08, box_gap_m=.04, source_radius_m=.5,
                  candidate_count=128, grid_resolution_m=.005, yaw_step_deg=15.,
                  max_pile_height_m=.3, placement_policy='stack',
                  grasp_approach_tilts_deg=[0.], floor_refinement_candidate_count=0,
                  floor_refinement_yaw_step_deg=5., output_directory=str(tmp_path),
                  destination='tray', plan_pick_place_motion=False, optimize_pipeline=False,
                  grasp_early_stop_count=4, incremental_grasp_fallback=False, placement_early_stop_count=32,
                  placement_validation_batch_size=8)
    messages, plans = [], []
    publisher = SimpleNamespace(publish=lambda _: None)
    node = SimpleNamespace(
        _value=params.__getitem__, _capture=lambda: (cloud, image),
        _cancel=threading.Event(), _bridge=bridge, _point=module.PlacementNode._point,
        _pose_matrix=lambda _: np.eye(4), _check_cancel=lambda: None,
        _cloud=lambda h, p: point_cloud2.create_cloud_xyz32(h, np.asarray(p).tolist()),
        get_logger=lambda: SimpleNamespace(info=messages.append, warn=messages.append),
        _tray_pub=publisher, _fused_pub=publisher, _scene_pub=publisher,
        _boxes_pub=publisher)

    def status(stage, **data):
        node._report.update(data, stage=stage,
                            elapsed_seconds=time.perf_counter()-node._started)

    node._status = status
    node._log_box_timing = lambda row, elapsed: module.BoxBenchmark._log_box_timing(node, row, elapsed)
    node._print_final_summary = lambda: module.BoxBenchmark._print_final_summary(node)

    def call(key, request):
        if key == 'gate':
            return SimpleNamespace(values=[SimpleNamespace(type=1, bool_value=False)])
        if key == 'presence':
            return SimpleNamespace(success=True, present=True, status='PRESENT')
        if key == 'ground':
            return SimpleNamespace(success=True, text='[0, 0, 1000, 1000]')
        if key == 'segment':
            return SimpleNamespace(success=True, mask=mask)
        assert key == 'grasp'
        if failure_phase == 'cancel':
            node._cancel.set()
            raise RuntimeError('BENCHMARK_CANCELLED')
        failed = request.object_name == 'box_01' and failure_phase == 'grasp'
        return SimpleNamespace(success=not failed, status='test grasp failure',
                               candidates=[] if failed else [GraspCandidate()])

    node._call = call

    def plan(source, grasps, scene, flags, tray, top, mode, h, name, **kwargs):
        plans.append((name, mode, scene.copy()))
        attempt = dict(mode=mode, calculation_seconds=.1)
        if name == 'box_01':
            attempt['failure'] = 'test placement failure'
            return None, attempt
        result = SimpleNamespace(selected_candidate=SimpleNamespace(id=1),
                                 position_error_m=0., descent_waypoint_count=0,
                                 descent_path_flat=[], lift_path_flat=[], transfer_path_flat=[])
        candidate = SimpleNamespace(rotation=np.eye(3), translation=source.mean(0))
        return (result, np.eye(4), candidate), attempt

    node._plan = plan
    module.BoxBenchmark._run(node)
    if failure_phase == 'cancel':
        assert node._report['stage'] == 'CANCELLED'
        assert node._report['accepted'] == 0
        assert len(node._report['objects']) == 1
        return
    assert node._report['stage'] == 'COMPLETE_WITH_FAILURES'
    assert node._report['accepted'] == 2 and node._report['failed'] == 1
    assert node._report['objects'][0]['result'] == 'FAILED'
    assert node._report['objects'][2]['support_object'] == 'box_02'
    assert node._report['objects'][2]['mode'] == 'STACK'
    for name, mode, scene in plans:
        if name != 'box_01':
            assert all(np.any(np.all(np.isclose(scene, p), axis=1)) for p in boxes[0])
    summaries = [m for m in messages if m.startswith('[BOX SUMMARY]')]
    assert len(summaries) == 1
    assert all(name in summaries[0] for name in ['box_01', 'box_02', 'box_03'])


def test_motion_failure_is_not_accepted_as_placement(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from my_actions.msg import GraspCandidate
    from my_actions.srv import PlanPlacement
    from std_msgs.msg import Header
    import placement_node.box_benchmark as module

    candidate = SimpleNamespace(score=1.0, translation=np.array([.2, 0., .04]),
                                rotation=np.eye(3))
    monkeypatch.setattr(module, 'search_heightmap_placements',
                        lambda *a, **k: ([candidate], SimpleNamespace(elapsed_seconds=.1)))
    parameters = dict(candidate_count=1, yaw_step_deg=90., grid_resolution_m=.005,
                      max_pile_height_m=.3, plan_pick_place_motion=True, optimize_pipeline=False)
    calls = []
    response = PlanPlacement.Response()
    response.success = False
    response.motion_status = response.status = 'TRANSFER_RRTC_NO_PATH'
    response.lift_seconds, response.transfer_seconds = .25, .75
    response.lift_path_flat = [0.] * 12
    response.descent_waypoint_count = 10

    def call(name, request):
        calls.append(request)
        return response

    node = SimpleNamespace(
        _value=parameters.__getitem__, _directory=tmp_path,
        _search_placements_api=lambda h, n, *a, **k: module.search_heightmap_placements(*a, **k),
        _call=lambda key, req: module.validate_placements(req, lambda query: call(key, query)),
        _point=module.PlacementNode._point,
        _cloud=lambda h, p: module.BoxBenchmark._cloud(None, h, p),
        _scene_cloud=lambda h, p, f: module.BoxBenchmark._scene_cloud(None, h, p, f),
        _finish_placement_timing=lambda *a: None,
        get_logger=lambda: SimpleNamespace(info=lambda _: None))
    source = box_fixture(1)[0]
    header = Header(frame_id='base_link')
    accepted, attempt = module.BoxBenchmark._plan_impl(
        node, source, [GraspCandidate()], source, np.zeros(len(source), np.uint8),
        tray_surface(), tray_surface(), 'edge', header, 'box_01')
    assert calls[0].plan_pick_place_motion
    assert accepted is None
    assert attempt['failure'] == 'TRANSFER_RRTC_NO_PATH'
    assert not attempt['motion_success']
    assert attempt['lift_seconds'] == .25
    assert attempt['transfer_seconds'] == .75
    assert attempt['lift_waypoints'] == 2
    assert attempt['descent_waypoints'] == 10


def test_virtual_support_replaces_only_duplicate_floor_noise():
    from placement_node.box_fixture import replace_fixture_floor_noise
    box = box_surface([.24, .29, .04])
    original = np.array([[.22, .27, .0014], [.22, .27, -.002],
                         [.22, .27, .008], [.40, .27, .0014]])
    filtered, keep = replace_fixture_floor_noise(original, [box], return_mask=True)
    np.testing.assert_array_equal(filtered, original[2:])
    np.testing.assert_array_equal(keep, [False, False, True, True])
    flags = np.array([0, 0, 1, 2], np.uint8)
    np.testing.assert_array_equal(flags[keep], [1, 2])


def test_fast_failure_tries_complete_placement_grid_before_expanding_grasps():
    from types import SimpleNamespace
    from placement_node.box_benchmark import BoxBenchmark
    calls = []
    node = SimpleNamespace(
        _value=lambda _: True, _report={'objects': [{'attempts': []}]},
        _expand_grasps=lambda *a: calls.append('expand'))

    def plan(*args, **kwargs):
        calls.append(kwargs.get('exhaustive', False))
        return ('accepted' if kwargs.get('exhaustive') else None,
                {'mode': 'edge', 'calculation_seconds': .1})

    node._plan_impl = plan
    accepted, _ = BoxBenchmark._plan(node, None, [], None, None, None, None,
                                    'edge', None, 'box_01')
    assert accepted == 'accepted'
    assert calls == [False, True]
    assert len(node._report['objects'][0]['attempts']) == 1


def test_fast_heightmap_stops_after_bounded_stable_candidates():
    source = box_fixture(1)[0]
    yy, xx = np.mgrid[:51, :51]
    tray = np.column_stack((xx.ravel()*.005, yy.ravel()*.005, np.zeros(xx.size)))
    candidates, diagnostics = search_heightmap_placements(
        source, tray, tray, 'upright', candidate_count=128,
        bottom_points=source[source[:, 2] <= .0001],
        support_mode='edge', early_stop_candidate_count=8)
    assert 0 < len(candidates) <= 8
    assert diagnostics.stable == 8
    for candidate in candidates:
        placed = (source-source.mean(axis=0)) @ candidate.rotation.T + candidate.translation
        assert placed[:, :2].min() >= -.001
        assert placed[:, :2].max() <= .251


@pytest.mark.parametrize('failure_status, grasp_ids, expected_calls', [
    ('LIFT_INITIAL_HELD_STATE_COLLISION', [1, 2], [(1, [1, 2]), (2, [1, 2])]),
    ('TRANSFER_RRTC_NO_PATH', [1], [(1, [1, 2]), (1, [2])]),
])
def test_motion_failure_retries_grasp_and_stops_after_full_success(
        monkeypatch, tmp_path, failure_status, grasp_ids, expected_calls):
    from types import SimpleNamespace
    from my_actions.msg import GraspCandidate
    from my_actions.srv import PlanPlacement
    from std_msgs.msg import Header
    import placement_node.box_benchmark as module

    candidates = [SimpleNamespace(score=1.-i*.01, translation=np.array([.2, 0., .04]),
                                  rotation=np.eye(3)) for i in range(6)]
    monkeypatch.setattr(module, 'search_heightmap_placements',
                        lambda *a, **k: (candidates, SimpleNamespace(elapsed_seconds=.1)))
    parameters = dict(candidate_count=6, yaw_step_deg=90., grid_resolution_m=.005,
                      max_pile_height_m=.3, plan_pick_place_motion=True, optimize_pipeline=True,
                      placement_early_stop_count=32, placement_validation_batch_size=2)
    grasps = [GraspCandidate(id=i) for i in grasp_ids]
    for grasp in grasps:
        grasp.grasp_pose.pose.orientation.w = 1.
    calls = []

    def call(name, request):
        calls.append((request.grasp_candidate.id, [c.id for c in request.candidates]))
        result = PlanPlacement.Response()
        result.selected_candidate.id = request.candidates[0].id
        result.selected_grasp_candidate = request.grasp_candidate
        result.placement_ee_pose.pose.orientation.w = 1.
        if len(calls) == 1:
            result.motion_status = result.status = failure_status
        else:
            result.success = result.motion_success = True
            result.motion_status = result.status = 'LIFT_TRANSFER_DESCENT_VALIDATED'
        return result

    node = SimpleNamespace(
        _value=parameters.__getitem__, _directory=tmp_path,
        _search_placements_api=lambda h, n, *a, **k: module.search_heightmap_placements(*a, **k),
        _call=lambda key, req: module.validate_placements(req, lambda query: call(key, query)),
        _point=module.PlacementNode._point, _pose_matrix=module.PlacementNode._pose_matrix,
        _cloud=lambda h, p: module.BoxBenchmark._cloud(None, h, p),
        _scene_cloud=lambda h, p, f: module.BoxBenchmark._scene_cloud(None, h, p, f),
        _finish_placement_timing=lambda *a: None,
        get_logger=lambda: SimpleNamespace(info=lambda _: None))
    source = box_fixture(1)[0]
    accepted, attempt = module.BoxBenchmark._plan_impl(
        node, source, grasps, source, np.zeros(len(source), np.uint8),
        tray_surface(), tray_surface(), 'edge', Header(frame_id='base_link'), 'box_01')
    assert accepted is not None
    assert calls == expected_calls
    assert attempt['motion_success']
    assert attempt['validated_candidate_count'] == sum(len(ids) for _, ids in expected_calls)


def test_empty_fast_search_does_not_repeat_exhaustive_grid_or_expand_grasps():
    from types import SimpleNamespace
    from placement_node.box_benchmark import BoxBenchmark
    calls = []
    node = SimpleNamespace(
        _value=lambda _: True, _report={'objects': [{'attempts': []}]},
        _expand_grasps=lambda *a: calls.append('expand'))

    def plan(*args, **kwargs):
        calls.append('search')
        return None, {'failure': 'NO_STABLE_PACKING_CANDIDATE'}

    node._plan_impl = plan
    accepted, attempt = BoxBenchmark._plan(node, None, [], None, None, None, None,
                                          'edge', None, 'box_01')
    assert accepted is None
    assert attempt['failure'] == 'NO_STABLE_PACKING_CANDIDATE'
    assert calls == ['search']


def test_grasp_pool_expands_incrementally_and_reuses_frozen_packing():
    from types import SimpleNamespace
    from placement_node.box_benchmark import BoxBenchmark
    calls, grasps = [], ['first']
    cached = object()
    node = SimpleNamespace(
        _value={'optimize_pipeline': True, 'incremental_grasp_fallback': True,
                'grasp_early_stop_count': 4}.__getitem__,
        _report={'objects': [{'attempts': []}]})

    def plan(*args, **kwargs):
        calls.append(('plan', kwargs.get('exhaustive', False), kwargs.get('packing_result')))
        if kwargs.get('exhaustive'):
            node._last_packing = cached
        return ('accepted' if args[1] == ['second'] else None,
                {'mode': 'edge', 'calculation_seconds': .1})

    def expand(limit):
        calls.append(('expand', limit))
        grasps.append('second')

    node._plan_impl, node._expand_grasps = plan, expand
    accepted, _ = BoxBenchmark._plan(node, None, grasps, None, None, None, None,
                                    'edge', None, 'box_01')
    assert accepted == 'accepted'
    assert calls == [('plan', False, None), ('plan', True, None),
                     ('expand', 5), ('plan', True, cached)]
    assert len(node._report['objects'][0]['attempts']) == 2
