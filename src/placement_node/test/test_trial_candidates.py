"""Trial-only visualization retains proposed poses without affecting motion."""

from types import SimpleNamespace
import numpy as np
from std_msgs.msg import Header
from my_actions.msg import GraspCandidate
from placement_node.node import PlacementNode


def test_noncontiguous_source_cloud_and_gripper_attachment():
    messages = []
    node = SimpleNamespace(_point=PlacementNode._point,
                           _pose_matrix=PlacementNode._pose_matrix,
                           _trial_candidates_pub=SimpleNamespace(publish=messages.append))
    points = np.array([[0., 0., 0.], [.1, 0., 0.], [.1, .1, 0.], [0., .1, 0.]], np.float32)
    center = np.array([.05, .05, 0.])
    grasp = GraspCandidate()
    grasp.grasp_pose.pose.orientation.w = 1.
    grasp.grasp_pose.pose.position.x = .05
    candidate = SimpleNamespace(rotation=np.eye(3), translation=np.array([.3, .2, .1]))
    PlacementNode._publish_trial_candidates(node, Header(frame_id='base_link'),
                                            'bowl', [candidate], center, points, grasp)
    markers = messages[0].markers
    axes = [m for m in markers if m.ns == 'trial_gripper_axes']
    assert len(axes) == 3
    assert np.allclose([axes[0].points[0].x, axes[0].points[0].y, axes[0].points[0].z],
                       [.3, .15, .1])
    assert any('proposed' in m.text for m in markers)


def test_tilted_source_bottom_becomes_horizontal_after_support_alignment():
    from scipy.spatial.transform import Rotation
    messages = []
    node = SimpleNamespace(_point=PlacementNode._point,
                           _pose_matrix=PlacementNode._pose_matrix,
                           _trial_candidates_pub=SimpleNamespace(publish=messages.append))
    flat = np.array([[0., 0., 0.], [.1, 0., 0.], [.1, .1, 0.], [0., .1, 0.]])
    tilted = flat @ Rotation.from_euler('x', 15, degrees=True).as_matrix().T
    candidate = SimpleNamespace(rotation=Rotation.from_euler('x', -15, degrees=True).as_matrix(),
                                translation=np.array([.3, .2, .1]))
    grasp = GraspCandidate()
    grasp.grasp_pose.pose.orientation.w = 1.
    PlacementNode._publish_trial_candidates(node, Header(frame_id='base_link'), 'bowl',
                                            [candidate], np.zeros(3), tilted, grasp, tilted)
    outline = next(m for m in messages[0].markers if m.ns == 'trial_footprints')
    assert np.ptp([p.z for p in outline.points]) < 1.e-6
