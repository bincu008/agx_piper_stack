"""Visibility reads existing memory and never invokes inference or actuation."""

import copy

from master_node.target_view import check_target_view

from my_actions import srv

import numpy as np

from sensor_msgs.msg import CameraInfo

from test_manipulation_memory import fixture, resolve


def setup():
    """Create grounded-memory fixtures and a non-actuating geometry backend."""
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
    """Use existing target geometry without another inference request."""
    memory, observation, calls, seen = setup()
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
    """Reuse explicit observations and the active destination cloud."""
    memory, observation, calls, seen = setup()
    query = srv.CheckTargetView.Request(
        name='tray', role='destination', observation_cloud=observation.cloud,
        camera_transform=observation.camera_transform)
    reply = check_target_view(memory, query, srv.CheckTargetView.Response())
    assert reply.success and calls == ['check_target_visibility']
    assert seen[0].target_cloud == memory.destination['cloud']


def test_unknown_name_or_ambiguous_directory_requires_location_not_vlm():
    """Report absent or ambiguous memory without implicit grounding."""
    memory, _, calls, _ = setup()
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


def test_directory_only_position_and_object_id_are_supported():
    """Expose the weaker center-only visibility check for directory entries."""
    memory, _, calls, seen = setup()
    object_id = memory.objects['bowl']['object_id']
    memory.objects.clear()
    reply = check_target_view(memory, srv.CheckTargetView.Request(name=object_id),
                              srv.CheckTargetView.Response())
    assert reply.success and reply.location_only and reply.object_id == object_id
    assert not seen[0].target_cloud.width
    assert seen[0].target_position == reply.position
    assert calls == ['capture_observation', 'check_target_visibility']


def test_calibration_change_does_not_reuse_stale_geometry():
    """Reject remembered geometry after the manual calibration changes."""
    memory, _, calls, _ = setup()
    memory.node.X_matrix = np.eye(4)*2.
    reply = check_target_view(memory, srv.CheckTargetView.Request(name='bowl'),
                              srv.CheckTargetView.Response())
    assert reply.success and reply.status == 'LOCATION_UNKNOWN' and not calls


def test_missing_intrinsics_and_unpaired_inputs_are_errors():
    """Fail incomplete camera inputs instead of pretending a clear view."""
    memory, observation, calls, _ = setup()
    memory.node.current_camera_info = None
    reply = check_target_view(memory, srv.CheckTargetView.Request(name='bowl'),
                              srv.CheckTargetView.Response())
    assert not reply.success and reply.status == 'TARGET_VIEW_CAMERA_INFO_UNAVAILABLE'
    assert 'check_target_visibility' not in calls
    reply = check_target_view(memory, srv.CheckTargetView.Request(
        name='bowl', camera_transform=observation.camera_transform),
        srv.CheckTargetView.Response())
    assert not reply.success
    assert reply.status == 'TARGET_VIEW_OBSERVATION_PAIR_REQUIRED'


def test_backend_errors_are_not_reported_as_visible():
    """Preserve geometry backend errors through the memory API."""
    memory, _, _, _ = setup()
    native = memory.call

    def fail(name, query):
        if name == 'check_target_visibility':
            return srv.CheckTargetVisibility.Response(status='VISIBILITY_CAMERA_INFO_MISMATCH')
        return native(name, query)

    memory.call = fail
    reply = check_target_view(memory, srv.CheckTargetView.Request(name='bowl'),
                              srv.CheckTargetView.Response())
    assert not reply.success and not reply.visible
    assert reply.status == 'VISIBILITY_CAMERA_INFO_MISMATCH'
