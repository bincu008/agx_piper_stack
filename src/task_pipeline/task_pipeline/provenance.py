"""Shared raw-input publisher identity evidence; never physical readiness."""

import os


def input_provenance(node):
    """Require exactly one expected raw publisher; distinguish replay from live."""
    expected = {
        '/camera/color/image_raw': '/camera/camera',
        '/camera/depth_registered/points': '/camera/camera',
        '/feedback/joint_states': '/agx_arm_ctrl_single_node',
        '/feedback/tcp_pose': '/agx_arm_ctrl_single_node',
    }
    sources, verified, replay, mock_only = {}, True, False, True
    for topic, owner in expected.items():
        publishers = node.get_publishers_info_by_topic(topic)
        names = [p.node_namespace.rstrip('/')+'/'+p.node_name for p in publishers]
        sources[topic] = {'publisher_count': len(names), 'publisher_nodes': names}
        verified = verified and names == [owner]
        mock_only = mock_only and names == ['/mock_hardware']
        replay = replay or any('mock_hardware' in name for name in names)
    return {'ros_domain_id': int(os.environ.get('ROS_DOMAIN_ID', '0')),
            'classification': ('live_driver_publishers' if verified else
                               'mock_replay_publishers' if mock_only else
                               'replay_or_mixed' if replay else 'unverified'),
            'SIMULATED': bool(mock_only),
            'topics': sources,
            'detail': 'DDS publisher identity snapshot; capture validates fresh sync. '
                      'This is not an actuator readiness certificate.'}
