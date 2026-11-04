"""Memory provenance, reuse and failure recovery without actuator calls."""

import copy
import json
from types import SimpleNamespace

from cv_bridge import CvBridge

from geometry_msgs.msg import TransformStamped

from master_node.manipulation_memory import ManipulationMemory, observed_mask, xyz
from master_node.world_state import WorldStateManager

from my_actions import srv

import numpy as np

from sensor_msgs_py import point_cloud2

from std_msgs.msg import Header


def fixture():
    """Create synchronized immutable inputs and non-actuating API fakes."""
    calls, logs = [], []
    bridge = CvBridge()
    header = Header(frame_id='base_link')
    header.stamp.sec = 100
    image = bridge.cv2_to_imgmsg(np.zeros((32, 32, 3), np.uint8), 'bgr8')
    image.header = copy.deepcopy(header)
    xx, yy = np.meshgrid(np.arange(32) * 0.01, np.arange(32) * 0.01)
    pts = np.stack((xx, yy, np.where(xx < 0.16, 0.1, 0.01)), axis=2).astype(np.float32)
    cloud = point_cloud2.create_cloud_xyz32(header, pts.reshape(-1, 3))
    cloud.height, cloud.width, cloud.row_step = 32, 32, 32 * cloud.point_step
    transform = TransformStamped(
        header=copy.deepcopy(header), child_frame_id='camera_optical_frame'
    )
    transform.transform.rotation.w = 1.0
    observation = srv.CaptureObservation.Response(
        success=True, image=image, cloud=cloud, camera_transform=transform
    )
    node = SimpleNamespace(
        world_state=WorldStateManager(),
        X_matrix=np.eye(4),
        state='IDLE',
        _held_object=None,
        create_service=lambda *a, **k: None,
        get_logger=lambda: SimpleNamespace(info=logs.append),
    )
    fail_grasp = [False]
    current_name = ['']

    def call(name, q):
        """Dispatch through the authoritative shared API runtime."""
        calls.append(name)
        if name == 'capture_observation':
            return observation
        if name == 'verify_presence':
            current_name[0] = q.target
            return srv.VerifyPresence.Response(success=True, present=True, status='PRESENT')
        if name == 'ground_object':
            return srv.GroundObject.Response(success=True, pixel_box=[0, 0, 32, 32])
        if name == 'segment_object':
            mask = (xx >= 0.16 if 'tray' in current_name[0] else xx < 0.16).astype(np.uint8) * 255
            result = bridge.cv2_to_imgmsg(mask, 'mono8')
            result.header = copy.deepcopy(header)
            return srv.SegmentGroundedImage.Response(success=True, mask=result)
        if name == 'commit_floor_slots':
            return srv.CommitFloorSlots.Response(success=True)
        if name == 'pin_grasp_observation':
            return srv.PinLayoutObservation.Response(success=True)
        if name == 'prepare_grasp_input':
            return srv.PrepareGraspInput.Response(
                success=True,
                seed_u=4,
                seed_v=4,
                image_width=32,
                image_height=32,
                affordance_type='AUTO',
            )
        if name == 'calculate_grasps':
            return srv.FindGraspCandidates.Response(
                success=not fail_grasp[0],
                status='NO_VALID_GEOMETRIC_GRASP' if fail_grasp[0] else 'GRASPS_READY',
                validated_target_mask=q.target_mask,
            )
        raise AssertionError(name)

    memory = ManipulationMemory(node)
    memory.call = call
    return memory, observation, calls, logs, fail_grasp


def resolve(memory, name, role='object', observation=None):
    """Resolve one target while excluding concurrent memory mutation."""
    q = srv.ResolveManipulationTarget.Request(name=name, role=role)
    if observation:
        q.image, q.observation_cloud, q.camera_transform = (
            observation.image,
            observation.cloud,
            observation.camera_transform,
        )
    return memory.resolve(q, srv.ResolveManipulationTarget.Response())


def test_unchanged_destination_skips_inference_and_keeps_original_geometry_header():
    """Unchanged destination skips inference and keeps original geometry header."""
    memory, observation, calls, _, _ = fixture()
    first = resolve(memory, 'tray', role='destination')
    assert first.success and not first.reused
    second = copy.deepcopy(observation)
    second.cloud.header.stamp.sec = 101
    second.image.header.stamp.sec = 101
    second.camera_transform.header.stamp.sec = 101
    count = calls.count('ground_object')
    reused = resolve(memory, 'tray', 'destination', second)
    assert reused.success and reused.reused
    assert calls.count('ground_object') == count
    assert reused.geometry_source_header == first.geometry_source_header
    assert reused.observation_header.stamp.sec == 101
    assert reused.object_cloud.header.stamp.sec == 100


def test_switch_destination_releases_old_geometry_but_retains_name_location():
    """Switch destination releases old geometry but retains name location."""
    memory, _, calls, _, _ = fixture()
    assert resolve(memory, 'tray', 'destination').success
    old = memory.destination
    assert resolve(memory, 'other tray', 'destination').success
    state = memory.snapshot()
    assert state['active_destination']['name'] == 'other tray'
    assert any(r['name'] == 'tray' for r in state['directory'].values())
    assert old not in memory.objects.values()
    assert 'commit_floor_slots' in calls
    assert all(set(r) == {'name', 'position'} for r in state['directory'].values())


def test_failed_grasp_preserves_source_for_retry_without_grounding():
    """Failed grasp preserves source for retry without grounding."""
    memory, _, calls, _, fail = fixture()
    assert resolve(memory, 'bowl').success
    record = memory.objects['bowl']
    count = calls.count('ground_object')
    fail[0] = True
    bad = memory.grasps(
        srv.FindGraspCandidates.Request(target_description='bowl'),
        srv.FindGraspCandidates.Response(),
    )
    assert not bad.success and memory.objects['bowl'] is record
    fail[0] = False
    good = memory.grasps(
        srv.FindGraspCandidates.Request(target_description='bowl', affordance_type='RIM'),
        srv.FindGraspCandidates.Response(),
    )
    assert good.success and calls.count('ground_object') == count


def test_active_geometry_is_bounded_while_hints_remain():
    """Active geometry is bounded while hints remain."""
    memory, _, _, _, _ = fixture()
    for i in range(10):
        assert resolve(memory, f'bowl {i}').success
    assert len(memory.objects) == 8 and len(memory.directory) == 10


def test_unknown_or_held_state_cannot_mutate_memory():
    """Unknown or held state cannot mutate memory."""
    memory, _, calls, _, _ = fixture()
    memory.node._held_object = 'bowl'
    result = resolve(memory, 'tray', 'destination')
    assert not result.success and not calls
    assert memory.inspect  # Read-only inspection remains exposed.


def test_moved_target_does_not_reuse_old_geometry():
    """Moved target does not reuse old geometry."""
    xx, yy = np.meshgrid(np.arange(16) * 0.01, np.arange(16) * 0.01)
    mapped = np.stack((xx, yy, xx * 0 + 0.1), axis=2)
    assert observed_mask(mapped, mapped.reshape(-1, 3)) is not None
    moved = mapped.copy()
    moved[:, :, 0] += 0.5
    assert observed_mask(moved, mapped.reshape(-1, 3)) is None


def test_calculation_commit_does_not_mark_object_physically_placed():
    """Calculation commit does not mark object physically placed."""
    memory, _, _, _, _ = fixture()
    assert resolve(memory, 'bowl').success
    before = memory.snapshot()
    memory.observe_result(
        'commit_floor_slots',
        srv.CommitFloorSlots.Request(),
        srv.CommitFloorSlots.Response(success=True),
    )
    assert memory.snapshot() == before


def test_mixed_field_cloud_reads_xyz_without_changing_coordinates():
    """Mixed field cloud reads xyz without changing coordinates."""
    from sensor_msgs.msg import PointField
    from master_node.manipulation_memory import xyz

    fields = [
        PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
        for i, n in enumerate(('x', 'y', 'z'))
    ]
    fields.append(PointField(name='bottom', offset=12, datatype=PointField.UINT8, count=1))
    cloud = point_cloud2.create_cloud(
        Header(frame_id='base_link'), fields, [(0.1, 0.2, 0.3, 2), (0.4, 0.5, 0.6, 0)]
    )
    np.testing.assert_allclose(xyz(cloud), [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def test_bad_manual_transform_is_rejected_before_inference():
    """Bad manual transform is rejected before inference."""
    memory, observation, calls, _, _ = fixture()
    observation.camera_transform.header.stamp.sec = 99
    result = resolve(memory, 'bowl', observation=observation)
    assert not result.success and 'TRANSFORM_ASSOCIATION' in result.status
    assert 'ground_object' not in calls


def test_changed_camera_view_forces_object_reobservation():
    """Changed camera view forces object reobservation."""
    memory, observation, calls, _, _ = fixture()
    assert resolve(memory, 'bowl').success
    observation = copy.deepcopy(observation)
    observation.camera_transform.transform.translation.x = 0.01
    before = calls.count('ground_object')
    result = resolve(memory, 'bowl', observation=observation)
    assert result.success and not result.reused
    assert calls.count('ground_object') == before + 1


def test_changed_calibration_invalidates_destination_geometry():
    """Changed calibration blocks reuse without silently discarding inventory."""
    memory, _, calls, _, _ = fixture()
    first = resolve(memory, 'tray', 'destination')
    memory.node.X_matrix[0, 3] = 0.005
    second = resolve(memory, 'tray', 'destination')
    assert not second.success and 'CALIBRATION_CHANGED' in second.status
    assert memory.destination['geometry_header'] == first.geometry_source_header
    assert calls.count('ground_object') == 1


def test_grasp_success_keeps_semantic_geometry_for_future_reobservation():
    """Keep contact refinement separate from the stored whole-object geometry."""
    memory, _, _, _, _ = fixture()
    assert resolve(memory, 'bowl').success
    original = memory.objects['bowl']['cloud']
    result = memory.grasps(
        srv.FindGraspCandidates.Request(target_description='bowl'),
        srv.FindGraspCandidates.Response(),
    )
    assert result.success and memory.objects['bowl']['cloud'] is original


def test_active_placements_reject_unsupported_overlay_instead_of_ignoring_it():
    """Require the explicit frozen-scene path for added synthetic obstacles."""
    memory, observation, _, _, _ = fixture()
    result = memory.placements(
        srv.FindPlacementCandidates.Request(
            object_name='bowl', additional_obstacles=observation.cloud
        ),
        srv.FindPlacementCandidates.Response(),
    )
    assert not result.success and 'OVERLAY_UNSUPPORTED' in result.status


def test_stationary_reuse_does_not_expand_semantic_mask_into_nearby_surface():
    """The 8 mm evidence radius must not turn adjacent table into the object."""
    memory, observation, calls, _, _ = fixture()
    assert resolve(memory, 'bowl').success
    record = memory.objects['bowl']
    semantic = memory.bridge.imgmsg_to_cv2(record['semantic_mask'], 'mono8') > 0
    # Move an outside pixel within the geometric evidence radius of the rim.
    points = xyz(observation.cloud, organized=True).copy()
    points[:, 16] = points[:, 15] + np.array([.002, 0., 0.])
    cloud = point_cloud2.create_cloud_xyz32(observation.cloud.header, points.reshape(-1, 3))
    cloud.height, cloud.width, cloud.row_step = 32, 32, 32 * cloud.point_step
    changed = copy.deepcopy(observation)
    changed.cloud = cloud
    grounded = calls.count('ground_object')
    result = resolve(memory, 'bowl', observation=changed)
    assert result.success and result.reused
    mask = memory.bridge.imgmsg_to_cv2(result.mask, 'mono8') > 0
    assert not np.any(mask & ~semantic)
    assert calls.count('ground_object') == grounded


def destination_setup():
    """Ground a source and tray, then provide a simulated confirmed release."""
    from master_node.manipulation_memory import check_capacity, cloud_xyz, grasp_scene, mapped_inputs, remember_release
    from master_node.prepared_grasp_lift import pose_matrix
    from geometry_msgs.msg import PoseStamped
    from scipy.spatial.transform import Rotation
    memory, observation, calls, _, _ = fixture()
    assert resolve(memory, 'bowl').success
    assert resolve(memory, 'tray', 'destination').success
    source = memory.objects['bowl']['cloud']
    grasp = PoseStamped(header=source.header)
    grasp.pose.orientation.w = 1.
    grasp.pose.position.x = .08
    actual = copy.deepcopy(grasp)
    actual.pose.position.x, actual.pose.position.y, actual.pose.position.z = .24, .16, .04
    q = Rotation.from_euler('z', .3).as_quat()
    actual.pose.orientation.x, actual.pose.orientation.y = map(float, q[:2])
    actual.pose.orientation.z, actual.pose.orientation.w = map(float, q[2:])
    plan = {
        'phase': 'RELEASED', 'pick_place': True, 'key': 'bowl', 'destination_name': 'tray',
        'destination_id': memory.destination['object_id'], 'revision': 1,
        'header': copy.deepcopy(source.header), 'review_source': source,
        'candidate': SimpleNamespace(grasp_pose=grasp),
        'lift': SimpleNamespace(placement_ee_pose=actual)}
    request = srv.HeldMotionStage.Request(object_name='bowl', plan_revision=1)
    memory.destination['grid'] = {'unchanged': True}
    memory.destination['grid_state'] = {'reserved_cells': [[0, 0]]}
    return memory, observation, calls, request, plan


def test_release_merges_actual_fk_geometry_without_changing_base_or_grid():
    from master_node.manipulation_memory import cloud_xyz, remember_release
    from master_node.prepared_grasp_lift import pose_matrix
    memory, _, _, request, plan = destination_setup()
    destination = memory.destination
    base = destination['base_cloud']
    grid, state = destination['grid'], destination['grid_state']
    original = cloud_xyz(plan['review_source'])
    transform = (pose_matrix(plan['lift'].placement_ee_pose.pose)
                 @ np.linalg.inv(pose_matrix(plan['candidate'].grasp_pose.pose)))
    expected = original @ transform[:3, :3].T + transform[:3, 3]
    assert remember_release(memory, request, plan)
    assert destination['base_cloud'] is base
    assert destination['grid'] is grid and destination['grid_state'] is state
    np.testing.assert_allclose(destination['placed_points'], expected, atol=1.e-7)
    np.testing.assert_allclose(cloud_xyz(destination['cloud']),
                               np.vstack((cloud_xyz(base), expected)), atol=1.e-7)
    assert destination['cloud'].header == destination['geometry_header'] == base.header
    stored = next(iter(destination['placements'].values()))
    assert stored['source_header'] == plan['review_source'].header
    position = memory.directory[stored['object_id']]['position']
    np.testing.assert_allclose(position, expected.mean(axis=0), atol=1.e-7)
    assert 'bowl' not in memory.objects
    assert memory.snapshot()['active_destination']['placed_objects'] == 1


def test_release_update_is_idempotent_and_failed_release_does_not_merge():
    from master_node.manipulation_memory import remember_release
    memory, _, _, request, plan = destination_setup()
    memory.node.execution_api = SimpleNamespace(
        prepared_grasp_lift=SimpleNamespace(record=plan))
    memory.observe_result('release_placed_object', request,
                          srv.HeldMotionStage.Response(success=False))
    assert not len(memory.destination['placed_points'])
    memory.observe_result('release_placed_object', request,
                          srv.HeldMotionStage.Response(success=True))
    revision = memory.revision
    cloud = memory.destination['cloud']
    memory.observe_result('release_placed_object', request,
                          srv.HeldMotionStage.Response(success=True))
    assert memory.revision == revision and memory.destination['cloud'] is cloud


def test_cached_destination_survives_old_timestamp_occlusion_and_view_change():
    from master_node.manipulation_memory import remember_release
    memory, observation, calls, request, plan = destination_setup()
    assert remember_release(memory, request, plan)
    destination = memory.destination
    cloud, grid = destination['cloud'], destination['grid']
    destination['verified'] -= 1000
    next_view = copy.deepcopy(observation)
    points = np.full((32 * 32, 3), np.nan, np.float32)
    next_view.cloud = point_cloud2.create_cloud_xyz32(next_view.cloud.header, points)
    next_view.cloud.height, next_view.cloud.width = 32, 32
    next_view.cloud.row_step = 32 * next_view.cloud.point_step
    next_view.camera_transform.transform.translation.x += .4
    before = calls.count('ground_object')
    reply = resolve(memory, 'tray', 'destination', next_view)
    assert reply.success and reply.reused
    assert calls.count('ground_object') == before
    assert memory.destination['cloud'] is cloud and memory.destination['grid'] is grid
    assert not np.any(memory.bridge.imgmsg_to_cv2(reply.mask, 'mono8'))


def test_mapped_planning_inputs_and_grasp_pixel_prefix_are_preserved():
    from master_node.manipulation_memory import cloud_xyz, grasp_scene, mapped_inputs, remember_release
    memory, observation, _, request, plan = destination_setup()
    assert remember_release(memory, request, plan)
    query = mapped_inputs(memory, srv.PrepareHeldScene.Request(), 'tray')
    assert query.destination_base_cloud == memory.destination['base_cloud']
    np.testing.assert_allclose(cloud_xyz(query.placed_scene_cloud),
                               memory.destination['placed_points'])
    overlay = grasp_scene(memory, observation)
    values = point_cloud2.read_points(overlay, field_names=('x', 'y', 'z'))
    points = np.column_stack([values[k] for k in ('x', 'y', 'z')])
    original = cloud_xyz(observation.cloud)
    np.testing.assert_equal(points[:len(original)], original)
    np.testing.assert_equal(points[len(original):], memory.destination['points'])
    assert overlay.header == observation.cloud.header


def test_capacity_and_destination_identity_are_checked_before_use():
    import pytest
    from master_node.manipulation_memory import check_capacity, remember_release
    memory, _, _, request, plan = destination_setup()
    plan['destination_id'] = 'different destination'
    with pytest.raises(RuntimeError, match='DESTINATION_ID_CHANGED'):
        remember_release(memory, request, plan)
    memory.destination['placements'] = dict.fromkeys(range(128))
    with pytest.raises(RuntimeError, match='CAPACITY_REACHED'):
        check_capacity(memory, plan['review_source'], 'tray')


def test_multiple_releases_append_both_sources_and_keep_original_tray():
    from master_node.manipulation_memory import remember_release
    memory, _, _, request, plan = destination_setup()
    assert remember_release(memory, request, plan)
    previous = memory.destination['placed_points'].copy()
    base = memory.destination['base_cloud']
    assert resolve(memory, 'second bowl').success
    second = copy.deepcopy(plan)
    second['revision'] += 1
    second['review_source'] = memory.objects['second bowl']['cloud']
    second['lift'].placement_ee_pose.pose.position.y += .12
    query = srv.HeldMotionStage.Request(object_name='second bowl', plan_revision=2)
    assert remember_release(memory, query, second)
    points = memory.destination['placed_points']
    np.testing.assert_array_equal(points[:len(previous)], previous)
    assert len(points) == 2 * len(previous)
    assert memory.destination['base_cloud'] is base
    assert memory.snapshot()['active_destination']['placed_objects'] == 2
    values = point_cloud2.read_points(memory.destination['cloud'], field_names=('placed',))
    assert np.count_nonzero(values['placed']) == len(points)


def test_release_rejects_wrong_revision_or_still_held_state():
    import pytest
    from master_node.manipulation_memory import remember_release
    memory, _, _, request, plan = destination_setup()
    request.plan_revision += 1
    with pytest.raises(RuntimeError, match='REVISION_MISMATCH'):
        remember_release(memory, request, plan)
    request.plan_revision = plan['revision']
    memory.node._held_object = object()
    with pytest.raises(RuntimeError, match='CONFIRMED_RELEASE'):
        remember_release(memory, request, plan)
    assert not len(memory.destination['placed_points'])


def target_view_setup():
    from sensor_msgs.msg import CameraInfo
    memory, observation, calls, _, _ = fixture()
    assert resolve(memory, 'bowl').success
    assert resolve(memory, 'tray', 'destination').success
    memory.node.current_camera_info = CameraInfo(
        width=32, height=32, k=[50., 0., 16., 0., 50., 16., 0., 0., 1.])
    memory.node.current_camera_info.header.frame_id = 'camera_optical_frame'
    native = memory.call
    seen = []

    def call(name, query):
        if name == 'check_target_visibility':
            calls.append(name)
            seen.append(query)
            return srv.CheckTargetVisibility.Response(
                success=True, status='READY', visible=True,
                observation_header=query.scene_cloud.header,
                sample_count=64, in_view_count=64, clear_count=48, visible_fraction=.75)
        return native(name, query)

    memory.call = call
    calls.clear()
    return memory, observation, calls, seen


def test_known_object_captures_and_checks_geometry_without_vlm():
    from master_node.manipulation_memory import check_target_view
    memory, observation, calls, seen = target_view_setup()
    before = (memory.revision, copy.deepcopy(memory.directory))
    reply = check_target_view(memory, srv.CheckTargetView.Request(name='bowl'),
                              srv.CheckTargetView.Response())
    assert reply.success and reply.visible and reply.status == 'READY'
    assert reply.sample_count == 64 and reply.visible_fraction == .75
    assert calls == ['capture_observation', 'check_target_visibility']
    assert seen[0].scene_cloud == observation.cloud
    assert seen[0].camera_transform == observation.camera_transform
    assert seen[0].target_cloud == memory.objects['bowl']['cloud']
    assert not reply.location_only
    assert (memory.revision, memory.directory) == before


def test_supplied_observation_avoids_another_capture_and_destination_is_reused():
    from master_node.manipulation_memory import check_target_view
    memory, observation, calls, seen = target_view_setup()
    query = srv.CheckTargetView.Request(
        name='tray', role='destination', observation_cloud=observation.cloud,
        camera_transform=observation.camera_transform)
    reply = check_target_view(memory, query, srv.CheckTargetView.Response())
    assert reply.success and calls == ['check_target_visibility']
    assert seen[0].target_cloud == memory.destination['cloud']


def test_unknown_name_or_ambiguous_directory_requires_location_not_vlm():
    from master_node.manipulation_memory import check_target_view
    memory, _, calls, _ = target_view_setup()
    for name in ('unknown', ''):
        reply = check_target_view(memory, srv.CheckTargetView.Request(name=name),
                                  srv.CheckTargetView.Response())
        assert not reply.visible
        assert reply.status == ('LOCATION_UNKNOWN' if name else 'INVALID_TARGET_VIEW_REQUEST')
    assert calls == []
    memory.objects.clear()
    memory.directory['duplicate'] = {'name': 'bowl', 'position': [0., 0., .3]}
    reply = check_target_view(memory, srv.CheckTargetView.Request(name='bowl'),
                              srv.CheckTargetView.Response())
    assert reply.success and reply.status == 'LOCATION_UNKNOWN' and calls == []


def scene_fixture(missing=()):
    import pytest
    memory, observation, calls, _, _ = fixture()
    original = memory.call
    prompts = []
    hidden = set(missing)

    def backend(name, query):
        if name == 'infer_vision':
            calls.append(name)
            prompts.append(query.user_prompt)
            if 'SCENE PRESENCE.' in query.user_prompt:
                targets, _ = json.JSONDecoder().raw_decode(
                    query.user_prompt[query.user_prompt.index('['):])
                values = {item['key']: item['description'] not in hidden for item in targets}
            else:
                descriptions, _ = json.JSONDecoder().raw_decode(
                    query.user_prompt[query.user_prompt.index('{'):])
                targets = [{'key': key, 'description': next(name for name in ('bowl', 'cup', 'tray')
                                                             if name in description)}
                           for key, description in descriptions.items()]
                boxes = {'bowl': [0, 0, 300, 1000], 'cup': [330, 0, 620, 1000],
                         'tray': [650, 0, 1000, 1000]}
                values = {item['key']: None if item['description'] in hidden
                          else boxes[item['description']] for item in targets}
            return srv.VisionInference.Response(success=True, text=json.dumps(values))
        if name == 'segment_object':
            calls.append(name)
            mask = np.zeros((32, 32), np.uint8)
            x0, y0, x1, y1 = query.bbox
            mask[y0:y1 + 1, x0:x1 + 1] = 255
            image = memory.bridge.cv2_to_imgmsg(mask, 'mono8')
            image.header = observation.image.header
            return srv.SegmentGroundedImage.Response(success=True, mask=image)
        return original(name, query)

    memory.call = backend
    request = srv.PerceiveScene.Request(
        object_names=['bowl', 'cup'], active_object_name='bowl', destination_name='tray',
        image=observation.image, observation_cloud=observation.cloud,
        camera_transform=observation.camera_transform)
    return memory, observation, calls, prompts, hidden, request


def test_all_targets_use_two_batch_prompts_and_one_supplied_observation():
    memory, observation, calls, prompts, hidden, query = scene_fixture()
    raw_cloud = bytes(observation.cloud.data)
    result = memory.scene_perception.perceive(query, srv.PerceiveScene.Response())
    assert result.success, result.status
    assert set(result.names) == {'bowl', 'cup', 'tray'} and all(result.observed)
    assert len(prompts) == 2 and calls.count('infer_vision') == 2
    assert calls.count('segment_object') == 3 and 'capture_observation' not in calls
    assert set(memory.objects) == {'bowl', 'cup'}
    assert memory.destination['name'] == 'tray'
    assert bytes(observation.cloud.data) == raw_cloud


def test_missing_object_keeps_location_hint_without_graspable_stale_geometry():
    memory, observation, calls, prompts, hidden, query = scene_fixture()
    assert memory.scene_perception.perceive(query, srv.PerceiveScene.Response()).success
    hidden.add('cup')
    result = memory.scene_perception.perceive(query, srv.PerceiveScene.Response())
    assert result.success and not result.observed[result.names.index('cup')]
    assert 'cup' not in memory.objects
    assert any(row['name'] == 'cup' for row in memory.directory.values())
    stale = memory.get_target(srv.ResolveManipulationTarget.Request(name='cup'),
                              srv.ResolveManipulationTarget.Response())
    assert not stale.success
    assert 'bowl' in memory.objects and memory.destination is not None


def test_normalized_box_bounds_and_duplicate_keys_are_strict():
    import pytest
    from master_node.manipulation_memory import parse_mapping, pixel_box
    assert pixel_box([0, 0, 1000, 1000], 32, 32) == [0, 0, 31, 31]
    for bad in ([0, 0, 0, 20], [-1, 0, 20, 20], [True, 0, 20, 20], [0, 0, 1001, 20]):
        with pytest.raises(ValueError):
            pixel_box(bad, 32, 32)
    with pytest.raises(ValueError):
        parse_mapping('{"t0":true,"t0":false}', ['t0', 't1'])


def test_unknown_inventory_is_perceived_once_then_location_memory_is_used():
    memory, observation, calls, prompts, hidden, query = scene_fixture()
    original = memory.call

    def backend(name, request):
        if name == 'perceive_scene':
            calls.append(name)
            return memory.scene_perception.perceive(request, srv.PerceiveScene.Response())
        return original(name, request)

    memory.call = backend
    result = memory.acquisition.ensure_memory(query, srv.PerceiveScene.Response())
    assert result.success and set(memory.objects) == {'bowl', 'cup'}
    assert calls.count('perceive_scene') == 1 and len(prompts) == 2
    memory.objects.pop('bowl')
    before = list(calls)
    result = memory.acquisition.ensure_memory(query, srv.PerceiveScene.Response())
    assert result.success and result.status == 'SCENE_MEMORY_READY_NO_INFERENCE'
    assert calls == before and not any(result.observed)
    assert result.positions[result.names.index('bowl')].x != 0.


def test_store_existing_segmentation_without_vlm():
    """A bound mask creates active geometry without repeated grounding inference."""
    memory, observation, calls, _, _ = fixture()
    mask = np.zeros((32, 32), np.uint8)
    mask[:, :16] = 255
    message = memory.bridge.cv2_to_imgmsg(mask, 'mono8')
    message.header = copy.deepcopy(observation.image.header)
    result = memory.store_segmented(srv.StoreSegmentedTarget.Request(
        name='bowl', role='object', image=observation.image, mask=message,
        observation_cloud=observation.cloud, camera_transform=observation.camera_transform),
        srv.StoreSegmentedTarget.Response())
    assert result.success, result.status
    assert result.object_cloud.width > 80
    assert not calls
    target = memory.get_target(srv.ResolveManipulationTarget.Request(name='bowl', role='object'),
                               srv.ResolveManipulationTarget.Response())
    assert target.success and target.object_cloud.width == result.object_cloud.width


def test_invalid_segmented_observation_does_not_replace_destination():
    """Reject stale masks before changing the retained destination/grid association."""
    memory, observation, _, _, _ = fixture()
    original = resolve(memory, 'tray', 'destination', observation)
    assert original.success
    destination = memory.destination
    mask = memory.bridge.cv2_to_imgmsg(np.ones((32, 32), np.uint8)*255, 'mono8')
    mask.header = copy.deepcopy(observation.image.header)
    transform = copy.deepcopy(observation.camera_transform)
    transform.header.stamp.sec += 1
    result = memory.store_segmented(srv.StoreSegmentedTarget.Request(
        name='other tray', role='destination', image=observation.image, mask=mask,
        observation_cloud=observation.cloud, camera_transform=transform),
        srv.StoreSegmentedTarget.Response())
    assert not result.success and 'ASSOCIATION_MISMATCH' in result.status
    assert memory.destination is destination


def test_spatial_context_reuses_authoritative_layout_without_capture():
    """Object cell membership comes from the returned map, with no inference."""
    from geometry_msgs.msg import Point, PoseStamped
    from my_actions.msg import SpatialCell, SpatialMap
    memory, _, calls, _, _ = fixture()
    assert resolve(memory, 'bowl').success
    memory.node.current_tcp_pose = PoseStamped()
    original = memory.call
    cell = SpatialCell(id='x0_y2_z0', minimum=Point(x=0., y=0., z=0.),
                       maximum=Point(x=.3, y=.3, z=.3))

    def call(name, query):
        if name == 'get_spatial_map':
            return srv.GetSpatialMap.Response(success=True, map=SpatialMap(cells=[cell]))
        return original(name, query)

    memory.call = call
    calls.clear()
    result = memory.spatial_context(srv.GetSpatialContext.Request(),
                                    srv.GetSpatialContext.Response())
    assert result.success, result.status
    assert result.object_names == ['bowl']
    assert result.object_cell_ids == ['x0_y2_z0']
    assert not calls
