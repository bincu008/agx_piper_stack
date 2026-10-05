from mock_hardware_node.snapshot_io import prefixed_topic


def test_prefixed_topic():
    assert prefixed_topic('', '/feedback/joint_states') == '/feedback/joint_states'
    assert prefixed_topic('/mock/', '/feedback/joint_states') == (
        '/mock/feedback/joint_states')
