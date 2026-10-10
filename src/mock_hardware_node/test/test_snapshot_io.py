from mock_hardware_node.snapshot_io import prefixed_topic


def test_prefixed_topic():
    assert prefixed_topic('', '/feedback/joint_states') == '/feedback/joint_states'
    assert prefixed_topic('/mock/', '/feedback/joint_states') == (
        '/mock/feedback/joint_states')


def test_depth_roundtrip_and_legacy_snapshot_loading(tmp_path):
    from mock_hardware_node.snapshot_io import MESSAGE_SPECS, save_messages, load_messages
    messages = {key: schema() for key, (_, schema, _) in MESSAGE_SPECS.items()}
    messages['depth'].width = 2
    messages['depth'].height = 1
    messages['depth'].encoding = '16UC1'
    messages['depth'].step = 4
    messages['depth'].data = bytes([1, 0, 2, 0])
    save_messages(tmp_path, messages, {'format_version': 2})
    restored, _ = load_messages(tmp_path)
    assert restored['depth'] == messages['depth']
    del messages['depth']
    save_messages(tmp_path, messages, {'format_version': 1})
    restored, _ = load_messages(tmp_path)
    assert 'depth' not in restored
    assert len(restored) == 7


def test_matching_uses_joint_history_instead_of_newest_arrival():
    from types import SimpleNamespace
    from mock_hardware_node.capture_snapshot import select_synchronized_samples

    def sample(seconds):
        return (SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
            sec=10, nanosec=int(seconds * 1e9)))), seconds)

    buffers = {'cloud': [sample(.10)], 'rgb': [sample(.10), sample(.20)],
               'depth': [sample(.10), sample(.20)],
               'joint_state': [sample(.105), sample(.19)]}
    selected, span = select_synchronized_samples(buffers, .05)
    assert selected is not None
    assert abs(span - .005) < 1e-6
    assert selected['joint_state'][1] == .105
    assert select_synchronized_samples(buffers, .001)[0] is None
