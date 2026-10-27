"""Exercise next-view planning and post-release sequencing using fake actuators."""

import threading
import time
from types import SimpleNamespace

from geometry_msgs.msg import PointStamped

from master_node.next_best_view import NextBestView
from master_node.piper_master_control import PiperMasterControl

from my_actions import srv

import numpy as np

import pytest

from std_msgs.msg import Header

from task_pipeline.context import StepStore
from task_pipeline.recipes import execute_cached_pick_place
from task_pipeline.registry import FUNCTIONS, catalog


def fixture():
    """Provide shared fake services and calibrated camera conversion without ROS nodes."""
    calls = []
    now = time.monotonic()
    node = SimpleNamespace(
        state='IDLE',
        _held_object=None,
        allow_physical_grasp_execution=False,
        hand_eye_matrix=np.eye(4),
        current_joint_states=SimpleNamespace(position=[0.0] * 6),
        joint_feedback_time=now,
        pose_feedback_time=now,
        gripper_feedback_time=now,
        create_service=lambda *a, **k: None,
        create_publisher=lambda *a, **k: SimpleNamespace(publish=lambda *args: None),
        get_logger=lambda: SimpleNamespace(error=lambda *a: None),
        manipulation_memory=SimpleNamespace(
            snapshot=lambda: {
                'active_objects': [],
                'directory': {},
                'active_destination': {'name': 'tray', 'position': [0.35, -0.1, 0.01]},
            }
        ),
    )
    node.hand_eye_matrix[:3, 3] = [0.04, 0.01, 0.02]
    node.populate_look_at_ik_request = lambda query, camera, target: (
        PiperMasterControl.populate_look_at_ik_request(node, query, camera, target)
    )
    node.command_joint_limits = [(-1.0, 1.0)] * 6
    execution = SimpleNamespace(node=node, group=None, _lock=threading.Lock())

    def guard():
        """Mirror only the hardware gate in this non-actuating fixture."""
        if not node.allow_physical_grasp_execution:
            raise RuntimeError('PHYSICAL_EXECUTION_DISABLED')

    def validate(stages, start):
        """Verify the retained target sequence stays within command bounds."""
        for _, path in stages:
            assert all(len(q) == 6 and all(-1.0 <= a <= 1.0 for a in q) for q in path)

    execution._guard = guard
    execution._joints = lambda *a, **k: None
    execution.prepared_grasp_lift = SimpleNamespace(record=None, validate_commands=validate)
    execution.held_motion = SimpleNamespace(
        path=lambda flat: [list(flat[i:i + 6]) for i in range(0, len(flat), 6)]
    )

    def backend(name, query):
        """Record calls, with no driver transport or physical motion."""
        calls.append((name, query))
        if name == 'capture_observation':
            response = srv.CaptureObservation.Response(success=True)
            response.cloud.header = Header(frame_id='base_link')
            response.cloud.header.stamp.sec = 123
            return response
        if name == 'get_nbv_candidates':
            return srv.GetNBVPose.Response(
                success=True,
                candidate_x=[query.target_x - 0.2 * query.radius],
                candidate_y=[query.target_y],
                candidate_z=[query.target_z + np.sqrt(0.96) * query.radius],
            )
        if name == 'solve_pose':
            assert query.use_tcp_pose and not query.ignore_collision
            return srv.LookAt.Response(success=True, joint_angles=[0.1] * 6)
        if name == 'plan_motion':
            assert len(query.remaining_path_flat) == 6
            assert not query.grasp_motion and not query.allow_target_contact
            return srv.CheckAndReplan.Response(
                success=True, path_flat=list(query.current_joints) + [0.1] * 6
            )
        if name == 'execute_trajectory':
            assert not execution._lock.locked()
            return srv.ExecuteTrajectory.Response(success=True, motion_completed=True)
        raise AssertionError(name)

    node.step_runtime = SimpleNamespace(_call=backend)
    return NextBestView(execution), calls


def plan(view):
    """Calculate a tray view without permitting execution."""
    return view.plan(
        srv.PlanNextBestView.Request(target_name='tray'), srv.PlanNextBestView.Response()
    )


def execute(view, revision):
    """Call the revision-bound API on the fake execution service only."""
    return view.execute(
        srv.HeldMotionStage.Request(object_name='tray', plan_revision=revision),
        srv.HeldMotionStage.Response(),
    )


def test_plan_is_calibrated_far_view_without_actuation():
    """Recover the desired optical pose after manual optical/TCP conversion."""
    view, calls = fixture()
    result = plan(view)
    assert result.success, result.status
    assert [name for name, _ in calls] == [
        'capture_observation',
        'get_nbv_candidates',
        'solve_pose',
        'plan_motion',
    ]
    assert result.camera_distance_m == pytest.approx(0.45)
    assert result.camera_pose.pose.position.x < 0.35
    assert result.camera_pose.pose.position.z > 0.44
    assert result.observation_header.stamp.sec == 123
    assert len(result.path_flat) == 12
    assert not np.allclose(
        [result.tcp_pose.pose.position.x, result.tcp_pose.pose.position.z],
        [result.camera_pose.pose.position.x, result.camera_pose.pose.position.z],
    )


def test_hardware_gate_and_single_use_revision_are_preserved():
    """Only a reviewed fresh plan may reach the fake actuator service once."""
    view, calls = fixture()
    result = plan(view)
    assert not execute(view, result.plan_revision).success
    assert view.record is not None and all(name != 'execute_trajectory' for name, _ in calls)
    view.node.allow_physical_grasp_execution = True
    assert execute(view, result.plan_revision).success
    assert not execute(view, result.plan_revision).success
    assert sum(name == 'execute_trajectory' for name, _ in calls) == 1


@pytest.mark.parametrize('failure', ['revision', 'expired', 'moved', 'stale', 'held', 'unknown'])
def test_stale_or_wrong_execution_cannot_actuate(failure):
    """Reject invalid handoffs before submitting any trajectory command."""
    view, calls = fixture()
    result = plan(view)
    view.node.allow_physical_grasp_execution = True
    revision = result.plan_revision
    if failure == 'revision':
        revision += 1
    elif failure == 'expired':
        view.record['created'] -= 61.0
    elif failure == 'moved':
        view.node.current_joint_states.position = [0.2] * 6
    elif failure == 'stale':
        view.node.joint_feedback_time -= 2.0
    elif failure == 'held':
        view.node._held_object = 'bowl'
    else:
        view.node.state = 'STATE_UNKNOWN'
    assert not execute(view, revision).success
    assert all(name != 'execute_trajectory' for name, _ in calls)


def test_actuator_failure_consumes_plan_and_marks_unknown_without_retry():
    """An uncertain execution cannot become an automatic motion retry."""
    view, calls = fixture()
    result = plan(view)
    view.node.allow_physical_grasp_execution = True
    original = view.call

    def fail(name, request):
        """Fail only the fake actuator backend."""
        if name == 'execute_trajectory':
            return srv.ExecuteTrajectory.Response(
                success=False, status='TRAJECTORY_EXECUTION_FAILED'
            )
        return original(name, request)

    view.call = fail
    assert not execute(view, result.plan_revision).success
    assert view.node.state == 'STATE_UNKNOWN' and view.record is None
    assert not execute(view, result.plan_revision).success


def test_explicit_target_requires_base_frame_and_distance_bounds():
    """Camera hints never introduce a computational TF conversion."""
    view, _ = fixture()
    target = PointStamped(header=Header(frame_id='camera_link'))
    result = view.plan(
        srv.PlanNextBestView.Request(target=target), srv.PlanNextBestView.Response()
    )
    assert not result.success and 'BASE_LINK' in result.status
    result = view.plan(
        srv.PlanNextBestView.Request(target_name='tray', camera_distance_m=0.1),
        srv.PlanNextBestView.Response(),
    )
    assert not result.success and 'DISTANCE' in result.status


def test_release_and_slot_commit_precede_next_view_and_observation():
    """Keep successful placement committed even if the later camera plan fails."""
    recipe = execute_cached_pick_place('bowl', 'tray')
    ids = [s.step_id for s in recipe]
    assert ids.index('release') < ids.index('commit_slots') < ids.index('move_tray_view')
    assert ids.index('release') < ids.index('move_tray_view') < ids.index('observe_tray')
    calls = []

    def backend(name, request):
        """Run the public typed sequence with fake success before view failure."""
        calls.append(name)
        response = FUNCTIONS[name].service_type.Response()
        response.success = name != 'execute_placement_retreat'
        if hasattr(response, 'status'):
            response.status = 'DONE' if response.success else 'NO_REACHABLE_NEXT_VIEW'
        if hasattr(response, 'plan_revision'):
            response.plan_revision = 7
        return response

    store = StepStore(backend)
    context, _ = store.submit('view-test', recipe)
    _, results = store.run_queue(context)
    assert any(r.function_name == 'release_placed_object' and r.success for r in results)
    assert any(r.function_name == 'commit_floor_slots' and r.success for r in results)
    assert results[-1].function_name == 'execute_placement_retreat' and not results[-1].success
    assert 'plan_next_best_view' not in calls
    assert 'capture_observation' not in calls
    assert (
        next(c for c in catalog() if c['name'] == 'execute_placement_retreat')['effect']
        == 'actuation'
    )


def test_shared_timing_handles_point_target_without_rejecting_valid_plan():
    """A PointStamped target is geometry, not a string timing label."""
    from master_node.step_runtime import MasterStepRuntime

    runtime = object.__new__(MasterStepRuntime)
    runtime._trace_local = threading.local()
    runtime._native = lambda *args: srv.PlanNextBestView.Response(success=True, status='PLANNED')
    runtime.node = SimpleNamespace()
    published = []
    runtime._timings = SimpleNamespace(publish=published.append)
    result = runtime._call(
        'plan_next_best_view',
        srv.PlanNextBestView.Request(target_name='tray', target=PointStamped()),
    )
    assert result.success and published[0].object_name == 'tray'


def test_released_plan_provides_view_hint_without_active_memory():
    """Fresh-perception recipes can still look at their released destination."""
    view, _ = fixture()
    node = view.node
    node.manipulation_memory.snapshot = lambda: {
        'active_objects': [],
        'directory': {},
        'active_destination': None,
    }
    pose = srv.PlanNextBestView.Response().tcp_pose
    pose.pose.position.x = 0.35
    view.execution.prepared_grasp_lift.record = {
        'phase': 'RELEASED',
        'destination_name': 'tray',
        'lift': SimpleNamespace(selected_candidate=SimpleNamespace(object_pose=pose)),
    }
    name, point = view.target(srv.PlanNextBestView.Request(target_name='tray'))
    assert name == 'tray' and point[0] == pytest.approx(0.35)


def test_next_view_preview_shows_only_final_tcp_and_clears_failed_replacement():
    """Keep the view endpoint separate from its complete trajectory samples."""
    view, _ = fixture()
    sent = []
    view.preview = SimpleNamespace(publish=sent.append)
    result = plan(view)
    assert result.success
    markers = sent[-1].markers
    assert len(markers) == 5
    assert all(m.points[0] == result.tcp_pose.pose.position for m in markers[1:4])
    assert markers[-1].text == '4 Tray-view retreat (TCP)'
    rejected = view.plan(
        srv.PlanNextBestView.Request(camera_distance_m=0.1), srv.PlanNextBestView.Response()
    )
    assert not rejected.success and sent[-1].markers[0].action == 3


def test_frozen_retreat_scene_contains_achieved_object_not_source():
    """Predict release geometry without taking another camera observation."""
    from test_prepared_plan_review import fixture as review_fixture
    from sensor_msgs_py import point_cloud2

    source, candidate, motion, nominal, center, points = review_fixture()
    motion.selected_grasp_candidate = candidate
    motion.selected_candidate.object_pose = nominal
    motion.descent_path_flat = [0.0] * 6 + [0.2] * 6
    obstacle = point_cloud2.create_cloud_xyz32(
        source.header, np.array([[0.8, 0.8, 0.8]], np.float32)
    )
    geometry = SimpleNamespace(
        source_object_cloud=source, collision_scene=obstacle, destination_support_cloud=source
    )
    view, calls = fixture()
    original = view.call
    sent = []

    def backend(name, request):
        if name == 'plan_lift':
            assert request.empty_hand and not request.source_object_cloud.width
            assert list(request.start_joints) == pytest.approx([0.2] * 6)
            return srv.PlanPlacement.Response(
                success=True,
                motion_success=True,
                lift_path_flat=[0.2] * 6 + [0.25] * 6,
                lift_ee_pose=motion.placement_ee_pose,
            )
        if name == 'plan_camera_retreat':
            assert request.clearance_only and not request.candidate_tcp_poses
            sent.append(request)
            return srv.PlanCameraRetreat.Response(
                success=True, path_flat=[0.25] * 6 + [0.3] * 6, tcp_pose=motion.placement_ee_pose
            )
        assert name == 'get_nbv_candidates'
        return original(name, request)

    view.call = backend
    withdrawal, result = view.prepare_retreat(
        geometry, motion, srv.PreparePickPlace.Request(source_object_center=center)
    )
    assert withdrawal.success and result.success
    assert list(sent[0].start_joints) == pytest.approx([0.25] * 6)
    cloud = point_cloud2.read_points_numpy(sent[0].collision_scene, field_names=('x', 'y', 'z'))
    assert any(np.allclose(p, [0.8, 0.8, 0.8]) for p in cloud)
    assert not any(np.allclose(p, points[0]) for p in cloud)
    assert all(name != 'capture_observation' for name, _ in calls)
    assert sent[0].collision_scene.header == source.header


def test_suitable_current_view_requires_no_camera_motion():
    """Prepared look is a read-only completion when target visibility is READY."""
    from geometry_msgs.msg import PoseStamped
    view, calls = fixture()
    view.node.current_tcp_pose=PoseStamped()
    view.node.current_tcp_pose.pose.orientation.w=1.
    original=view.call

    def backend(name, request):
        if name=='check_target_view':
            return srv.CheckTargetView.Response(success=True,status='READY',visible=True)
        assert name not in ('solve_pose','plan_motion','execute_trajectory','get_nbv_candidates')
        return original(name,request)

    view.call=backend
    response=view.plan(srv.PlanNextBestView.Request(target_name='tray',allow_current_view=True),
                       srv.PlanNextBestView.Response())
    assert response.success and response.status=='CURRENT_VIEW_ALREADY_SUITABLE'
    assert list(response.path_flat)==[0.]*6
