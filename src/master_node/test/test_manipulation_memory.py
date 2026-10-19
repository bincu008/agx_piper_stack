"""Memory provenance, reuse and failure recovery without actuator calls."""

import copy
from types import SimpleNamespace

from cv_bridge import CvBridge

from geometry_msgs.msg import TransformStamped

from master_node.manipulation_memory import ManipulationMemory, observed_mask
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
    """Changed calibration invalidates destination geometry."""
    memory, _, calls, _, _ = fixture()
    first = resolve(memory, 'tray', 'destination')
    memory.node.X_matrix[0, 3] = 0.005
    second = resolve(memory, 'tray', 'destination')
    assert second.success and not second.reused
    assert second.memory_revision > first.memory_revision
    assert calls.count('ground_object') == 2


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
