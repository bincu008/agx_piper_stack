"""Check requested/achieved visualization without planners or actuator calls."""

import copy
from pathlib import Path
from types import SimpleNamespace

from geometry_msgs.msg import Point, PoseStamped, Vector3

from master_node.prepared_plan_review import (
    PreparedPlanReview,
    placed_clouds,
    pose_matrix,
    review_markers,
)

from my_actions.msg import GraspCandidate
from my_actions.srv import PlanPlacement

import numpy as np

from scipy.spatial.transform import Rotation

from sensor_msgs_py import point_cloud2

from std_msgs.msg import Header

from visualization_msgs.msg import Marker

import yaml


def fixture():
    """Use an off-origin source and a relaxed placement with rotation and drift."""
    header = Header(frame_id='base_link')
    header.stamp.sec = 123
    points = np.array([[0.3, 0.1, 0.02], [0.32, 0.1, 0.02], [0.3, 0.12, 0.05]], np.float32)
    source = point_cloud2.create_cloud_xyz32(header, points)
    grasp = PoseStamped(header=header)
    grasp.pose.position = Point(x=0.28, y=0.08, z=0.08)
    grasp.pose.orientation.w = 1.0
    candidate = GraspCandidate(id=1, grasp_pose=grasp, jaw_opening_m=0.02)
    nominal = PoseStamped(header=header)
    nominal.pose.position = Point(x=0.5, y=-0.1, z=0.04)
    nominal.pose.orientation.w = 1.0
    placement = copy.deepcopy(nominal)
    placement.pose.position.x += 0.015
    (
        placement.pose.orientation.x,
        placement.pose.orientation.y,
        placement.pose.orientation.z,
        placement.pose.orientation.w,
    ) = Rotation.from_euler('z', 10, degrees=True).as_quat()
    plan = PlanPlacement.Response(success=True, motion_success=True, placement_ee_pose=placement)
    # Different geometry than the defaults proves no mesh dimensions are guessed.
    plan.placement_finger_dimensions = Vector3(x=0.012, y=0.025, z=0.08)
    for side in (-1, 1):
        finger = copy.deepcopy(placement.pose)
        finger.position.y += side * 0.016
        plan.placement_finger_poses.append(finger)
    center = Point(x=0.31, y=0.11, z=0.03)
    return source, candidate, plan, nominal, center, points


def xyz(cloud):
    """Extract fields for independent expected-coordinate comparisons."""
    values = point_cloud2.read_points(cloud)
    return np.column_stack([values[name] for name in ('x', 'y', 'z')])


def test_actual_cloud_follows_gripper_attachment_instead_of_requested_slot():
    """Show the actual shift even when the selected slot pose stays nominal."""
    source, candidate, plan, nominal, center, points = fixture()
    plan.selected_candidate.object_pose = nominal
    requested, actual, achieved = placed_clouds(source, candidate, plan, nominal, center)
    delta = pose_matrix(plan.placement_ee_pose.pose) @ np.linalg.inv(
        pose_matrix(candidate.grasp_pose.pose)
    )
    expected = np.array([(delta @ np.r_[p, 1.0])[:3] for p in points])
    np.testing.assert_allclose(xyz(actual), expected, atol=1.0e-7)
    np.testing.assert_allclose(
        xyz(requested), points - [0.31, 0.11, 0.03] + [0.5, -0.1, 0.04], atol=1.0e-7
    )
    np.testing.assert_allclose(achieved, (delta @ [0.31, 0.11, 0.03, 1.0])[:3])
    assert not np.allclose(xyz(requested), xyz(actual))
    assert actual.header == requested.header == source.header


def test_rotated_source_grasp_is_removed_from_attachment_exactly_once():
    """Carry the held object through arbitrary grasp orientation, not world yaw alone."""
    source, candidate, plan, nominal, center, points = fixture()
    rotation = Rotation.from_euler('xyz', [0.3, -0.2, 0.8])
    q = rotation.as_quat()
    (
        candidate.grasp_pose.pose.orientation.x,
        candidate.grasp_pose.pose.orientation.y,
        candidate.grasp_pose.pose.orientation.z,
        candidate.grasp_pose.pose.orientation.w,
    ) = q
    _, actual, _ = placed_clouds(source, candidate, plan, nominal, center)
    local = rotation.inv().apply(points - [0.28, 0.08, 0.08])
    expected = Rotation.from_euler('z', 10, degrees=True).apply(local) + [0.515, -0.1, 0.04]
    np.testing.assert_allclose(xyz(actual), expected, atol=1.0e-7)


def test_review_reuses_actual_finger_poses_and_mesh_dimensions():
    """Draw exact returned placement fingers and map them back to the grasp."""
    source, candidate, plan, nominal, center, _ = fixture()
    _, _, achieved = placed_clouds(source, candidate, plan, nominal, center)
    grasp, placement = review_markers(candidate, plan, source.header, nominal, achieved)
    fingers = [m for m in placement.markers if m.type == Marker.CUBE]
    assert len(fingers) == 2
    for marker, finger in zip(fingers, plan.placement_finger_poses):
        assert marker.pose == finger
        assert marker.scale == plan.placement_finger_dimensions
    back = pose_matrix(candidate.grasp_pose.pose) @ np.linalg.inv(
        pose_matrix(plan.placement_ee_pose.pose)
    )
    grasp_fingers = [m for m in grasp.markers if m.type == Marker.CUBE]
    np.testing.assert_allclose(
        pose_matrix(grasp_fingers[0].pose), back @ pose_matrix(fingers[0].pose), atol=1.0e-7
    )
    assert any('Achieved vs slot' in m.text for m in placement.markers)


def test_clear_removes_all_latched_final_outputs():
    """A new failed preparation must not leave an old successful view visible."""
    sent = {}

    def publisher(kind, topic, qos):
        """Record visualization output only."""
        sent[topic] = []
        return SimpleNamespace(publish=sent[topic].append)

    grasp = publisher(None, 'grasp', None)
    review = PreparedPlanReview(SimpleNamespace(create_publisher=publisher), grasp)
    review.clear()
    assert sent['grasp'][-1].markers[0].action == Marker.DELETEALL
    assert sent['/pipeline/prepared_placement'][-1].markers[0].action == Marker.DELETEALL
    assert not sent['/pipeline/prepared_placed_object_cloud'][-1].data
    assert not sent['/pipeline/prepared_nominal_object_cloud'][-1].data


def test_focused_rviz_has_final_topics_and_no_benchmark_or_trial_displays():
    """Keep standard review separate from the full candidate diagnostic view."""
    path = Path(__file__).resolve().parents[2] / 'vamp_node/rviz/grasp_debug.rviz'
    displays = yaml.safe_load(path.read_text())['Visualization Manager']['Displays']
    topics = [item['Topic']['Value'] for item in displays if 'Topic' in item]
    assert not any('/box_benchmark/' in topic for topic in topics)
    assert not any(
        'candidates_markers' in topic or 'trial_candidates' in topic for topic in topics
    )
    assert '/pipeline/prepared_grasp' in topics
    assert '/pipeline/prepared_placement' in topics
    assert '/pipeline/prepared_placed_object_cloud' in topics
    assert '/pipeline/prepared_nominal_object_cloud' in topics


def test_only_complete_preparation_publishes_final_review():
    """Clear trials first and publish once after every motion stage passes."""
    from test_prepared_pick_place import full_fixture
    from my_actions.srv import PreparePickPlace

    trial, request, _, _ = full_fixture()
    seen = []
    trial.preview = object()
    trial.review = SimpleNamespace(
        clear=lambda: seen.append('clear'), publish=lambda record: seen.append(record['phase'])
    )
    response = trial.prepare(request, PreparePickPlace.Response())
    assert response.success and seen == ['clear', 'PREPARED']


def test_failed_preparation_clears_old_review_without_publishing_trials():
    """Endpoint grasp candidates must not masquerade as a final complete plan."""
    from test_prepared_pick_place import full_fixture
    from my_actions.srv import PreparePickPlace

    trial, request, _, _ = full_fixture()
    seen = []
    trial.preview = object()
    trial.review = SimpleNamespace(
        clear=lambda: seen.append('clear'), publish=lambda record: seen.append('published')
    )
    original = trial.node.step_runtime._call

    def reject(name, query):
        """Reject planning without any actuator calls."""
        if name == 'plan_final_approach':
            from my_actions.srv import PlanCartesianGraspApproach

            return PlanCartesianGraspApproach.Response(success=False, failure_reason='TEST')
        return original(name, query)

    trial.node.step_runtime._call = reject
    response = trial.prepare(request, PreparePickPlace.Response())
    assert not response.success and seen == ['clear']


def test_visualization_error_does_not_invalidate_a_complete_motion_plan():
    """Report a failed review publication separately from planning success."""
    from test_prepared_pick_place import full_fixture
    from my_actions.srv import PreparePickPlace

    trial, request, _, _ = full_fixture()
    seen = []
    trial.preview = object()

    def broken(record):
        """Simulate a visualization-only failure."""
        raise ValueError('TEST_PREVIEW_FAILURE')

    trial.review = SimpleNamespace(clear=lambda: seen.append('clear'), publish=broken)
    response = trial.prepare(request, PreparePickPlace.Response())
    assert response.success and trial.record['phase'] == 'PREPARED'
    assert seen == ['clear', 'clear']


def test_milestones_use_final_stage_poses_without_intermediate_samples():
    """Publish lift once at its planned endpoint, with distinct stage labels."""
    source, candidate, plan, nominal, center, _ = fixture()
    plan.lift_ee_pose = copy.deepcopy(candidate.grasp_pose)
    plan.lift_ee_pose.pose.position.z += 0.05
    sent = {}

    def publisher(kind, topic, qos):
        """Capture only marker and cloud publications."""
        sent[topic] = []
        return SimpleNamespace(publish=sent[topic].append)

    review = PreparedPlanReview(
        SimpleNamespace(create_publisher=publisher), publisher(None, 'grasp', None)
    )
    review.publish(
        {
            'review_source': source,
            'review_nominal': nominal,
            'review_center': center,
            'candidate': candidate,
            'lift': plan,
        }
    )
    lift = sent['/pipeline/prepared_lift'][-1].markers
    assert len(lift) == 7  # Clear, three axes, label, two endpoint fingers.
    assert all(m.points[0] == plan.lift_ee_pose.pose.position for m in lift[1:4])
    assert lift[4].text == '2 Lift'
    assert len([m for m in lift if m.ns == 'lift_fingers']) == 2
    assert sent['grasp'][-1].markers[-1].text == '1 Grasp'
    assert sent['/pipeline/prepared_placement'][-1].markers[-1].text == '3 Placement'
    review.clear()
    assert sent['/pipeline/prepared_lift'][-1].markers[0].action == Marker.DELETEALL
