"""Display only a complete prepared plan, with requested and achieved geometry."""

import copy

from geometry_msgs.msg import Point, Pose

import numpy as np

from rclpy.qos import DurabilityPolicy, QoSProfile

from scipy.spatial.transform import Rotation

from sensor_msgs.msg import PointCloud2

from sensor_msgs_py import point_cloud2

from std_msgs.msg import Header

from visualization_msgs.msg import Marker, MarkerArray


def pose_matrix(pose):
    """Use the planner's base-frame pose directly without consulting TF."""
    q = pose.orientation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def transform_pose(matrix, pose):
    """Apply a rigid transform to an existing planner finger pose."""
    result = matrix @ pose_matrix(pose)
    output = Pose()
    output.position = Point(**dict(zip(('x', 'y', 'z'), result[:3, 3].tolist())))
    q = Rotation.from_matrix(result[:3, :3]).as_quat()
    output.orientation.x, output.orientation.y, output.orientation.z, output.orientation.w = q
    return output


def placed_clouds(source, candidate, plan, nominal, center):
    """Compare ideal slot geometry with the immutable attachment's achieved pose."""
    if source.header.frame_id != 'base_link':
        raise ValueError('PREVIEW_REQUIRES_BASE_LINK_SOURCE')
    fields = point_cloud2.read_points(source, field_names=('x', 'y', 'z'), skip_nans=False)
    points = np.column_stack([fields[name].reshape(-1) for name in ('x', 'y', 'z')])
    points = points[np.isfinite(points).all(axis=1)]
    actual = pose_matrix(plan.placement_ee_pose.pose) @ np.linalg.inv(
        pose_matrix(candidate.grasp_pose.pose)
    )
    requested = pose_matrix(nominal.pose)
    source_center = np.array([center.x, center.y, center.z])
    requested[:3, 3] -= requested[:3, :3] @ source_center
    # Both clouds use the exact same immutable source and source header.
    clouds = [
        point_cloud2.create_cloud_xyz32(
            source.header, (points @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
        )
        for matrix in (requested, actual)
    ]
    achieved_center = actual[:3, :3] @ source_center + actual[:3, 3]
    return clouds[0], clouds[1], achieved_center


def axes(header, pose, namespace, first_id):
    """Draw the actual three tool axes, rather than one ambiguous pose arrow."""
    matrix = pose_matrix(pose)
    result = []
    for i, color in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
        marker = Marker(header=header, ns=namespace, id=first_id + i, type=Marker.ARROW)
        end = matrix[:3, 3] + matrix[:3, i] * 0.05
        marker.points = [
            copy.deepcopy(pose.position),
            Point(**dict(zip(('x', 'y', 'z'), end.tolist()))),
        ]
        marker.scale.x, marker.scale.y, marker.scale.z = 0.003, 0.006, 0.009
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        result.append(marker)
    return result


def endpoint_markers(pose, label, namespace):
    """Show one stage endpoint and its tool axes without trajectory samples."""
    markers = axes(pose.header, pose.pose, namespace, 0)
    text = Marker(header=pose.header, ns=namespace, id=3, type=Marker.TEXT_VIEW_FACING)
    text.pose = copy.deepcopy(pose.pose)
    text.pose.position.z += 0.07
    text.scale.z = 0.018
    text.color.r = text.color.g = text.color.b = text.color.a = 1.0
    text.text = label
    markers.append(text)
    return MarkerArray(markers=[Marker(action=Marker.DELETEALL)] + markers)


def review_markers(candidate, plan, header, nominal, center):
    """Reuse VAMP's exact finger dimensions and poses for both endpoint displays."""
    grasp = axes(header, candidate.grasp_pose.pose, 'final_grasp_axes', 0)
    placement = axes(header, plan.placement_ee_pose.pose, 'actual_placement_axes', 0)
    # Map the returned placement fingers back to the selected grasp. This avoids
    # duplicating or hard-coding VAMP's TCP offset and finger mesh dimensions.
    back = pose_matrix(candidate.grasp_pose.pose) @ np.linalg.inv(
        pose_matrix(plan.placement_ee_pose.pose)
    )
    for i, pose in enumerate(plan.placement_finger_poses):
        for output, transformed, namespace, rgb in (
            (grasp, transform_pose(back, pose), 'final_grasp_fingers', (0.0, 1.0, 0.2)),
            (placement, pose, 'actual_placement_fingers', (0.0, 1.0, 1.0)),
        ):
            marker = Marker(
                header=header,
                ns=namespace,
                id=10 + i,
                type=Marker.CUBE,
                pose=copy.deepcopy(transformed),
                scale=copy.deepcopy(plan.placement_finger_dimensions),
            )
            marker.color.r, marker.color.g, marker.color.b = rgb
            marker.color.a = 0.7
            output.append(marker)
    label = Marker(header=header, ns='placement_shift', id=20, type=Marker.TEXT_VIEW_FACING)
    label.pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2] + 0.07))
    label.pose.orientation.w = 1.0
    label.scale.z = 0.012
    label.color.r = label.color.g = label.color.b = label.color.a = 1.0
    offset = center - np.array(
        [nominal.pose.position.x, nominal.pose.position.y, nominal.pose.position.z]
    )
    actual_rotation = (
        pose_matrix(plan.placement_ee_pose.pose)[:3, :3]
        @ pose_matrix(candidate.grasp_pose.pose)[:3, :3].T
    )
    angle = Rotation.from_matrix(actual_rotation @ pose_matrix(nominal.pose)[:3, :3].T)
    label.text = 'Achieved vs slot: %.1f mm / %.1f deg' % (
        np.linalg.norm(offset) * 1000.0,
        np.degrees(angle.magnitude()),
    )
    placement.append(label)
    if np.linalg.norm(offset) > 1.0e-6:
        marker = Marker(header=header, ns='placement_shift', id=21, type=Marker.ARROW)
        marker.points = [
            copy.deepcopy(nominal.pose.position),
            Point(**dict(zip(('x', 'y', 'z'), center.tolist()))),
        ]
        marker.scale.x, marker.scale.y, marker.scale.z = 0.002, 0.005, 0.008
        marker.color.r = marker.color.g = marker.color.a = 1.0
        placement.append(marker)
    return MarkerArray(
        markers=[Marker(header=Header(frame_id='base_link'), action=Marker.DELETEALL)] + grasp
    ), MarkerArray(
        markers=[Marker(header=Header(frame_id='base_link'), action=Marker.DELETEALL)] + placement
    )


class PreparedPlanReview:
    """Publish final-only views; failed/intermediate candidates never enter them."""

    def __init__(self, node, grasp_publisher):
        """Keep visualization separate from execution and planner state."""
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node = node
        self.withdrawal = node.create_publisher(MarkerArray, '/pipeline/prepared_withdrawal', qos)
        self.lift = node.create_publisher(MarkerArray, '/pipeline/prepared_lift', qos)
        self.grasp = grasp_publisher
        self.placement = node.create_publisher(MarkerArray, '/pipeline/prepared_placement', qos)
        self.nominal = node.create_publisher(
            PointCloud2, '/pipeline/prepared_nominal_object_cloud', qos
        )
        self.actual = node.create_publisher(
            PointCloud2, '/pipeline/prepared_placed_object_cloud', qos
        )
        self.clear()

    def clear(self):
        """Remove the previous final review before a new calculation or reset."""
        empty = point_cloud2.create_cloud_xyz32(
            Header(frame_id='base_link'), np.empty((0, 3), dtype=np.float32)
        )
        self.grasp.publish(
            MarkerArray(
                markers=[Marker(header=Header(frame_id='base_link'), action=Marker.DELETEALL)]
            )
        )
        self.placement.publish(
            MarkerArray(
                markers=[Marker(header=Header(frame_id='base_link'), action=Marker.DELETEALL)]
            )
        )
        self.lift.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        self.withdrawal.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        view = getattr(getattr(self.node, 'execution_api', None), 'next_best_view', None)
        if view is not None:
            view.publish_preview()
        self.nominal.publish(empty)
        self.actual.publish(empty)

    def publish(self, record):
        """Render the selected complete plan, including achieved object drift."""
        source, nominal, center = (
            record['review_source'],
            record['review_nominal'],
            record['review_center'],
        )
        candidate, plan = record['candidate'], record['lift']
        requested, actual, achieved = placed_clouds(source, candidate, plan, nominal, center)
        grasp, placement = review_markers(candidate, plan, source.header, nominal, achieved)
        grasp.markers += endpoint_markers(candidate.grasp_pose, '1 Grasp', 'grasp_label').markers[
            -1:
        ]
        placement.markers += endpoint_markers(
            plan.placement_ee_pose, '3 Placement', 'placement_label'
        ).markers[-1:]
        lift_markers = endpoint_markers(plan.lift_ee_pose, '2 Lift', 'lift_endpoint')
        placement_to_lift = pose_matrix(plan.lift_ee_pose.pose) @ np.linalg.inv(
            pose_matrix(plan.placement_ee_pose.pose)
        )
        for index, finger_pose in enumerate(plan.placement_finger_poses):
            finger = Marker(
                header=plan.lift_ee_pose.header,
                ns='lift_fingers',
                id=10 + index,
                type=Marker.CUBE,
                pose=transform_pose(placement_to_lift, finger_pose),
                scale=copy.deepcopy(plan.placement_finger_dimensions),
            )
            finger.color.r, finger.color.g, finger.color.b, finger.color.a = 0.2, 0.5, 1.0, 0.85
            lift_markers.markers.append(finger)
        self.lift.publish(lift_markers)
        if record.get('withdrawal_plan') is not None:
            self.withdrawal.publish(endpoint_markers(record['withdrawal_plan'].lift_ee_pose,
                                    '3b Withdrawal (empty)', 'withdrawal_endpoint'))
        self.grasp.publish(grasp)
        self.placement.publish(placement)
        self.nominal.publish(requested)
        self.actual.publish(actual)
