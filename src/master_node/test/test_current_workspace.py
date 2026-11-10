"""One-observation source association with synthetic feedback, never hardware."""

from copy import deepcopy

from geometry_msgs.msg import PoseStamped, TransformStamped
from master_node.current_workspace import CurrentObservation
import pytest
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import Header


def header(seconds, frame='base_link'):
    result = Header(frame_id=frame)
    result.stamp.sec = int(seconds)
    result.stamp.nanosec = round((seconds-int(seconds))*1.e9)
    return result


def fixture():
    clock = [100.]
    store = CurrentObservation(clock=lambda: clock[0])
    for stamp in (99.85, 99.95, 100.):
        clock[0] = stamp
        joint = JointState(header=header(stamp), position=[0.]*6)
        tcp = PoseStamped(header=header(stamp))
        tcp.pose.orientation.w = 1.
        store.feedback_sample('joints', joint)
        store.feedback_sample('tcp', tcp)
    image = Image(header=header(100., 'optical'), height=2, width=2,
                  encoding='bgr8', step=6, data=[1]*12)
    cloud = PointCloud2(header=header(100.), height=2, width=2)
    transform = TransformStamped(header=header(100.), child_frame_id='optical')
    transform.transform.rotation.w = 1.
    return store, clock, {'image': image, 'cloud': cloud, 'transform': transform}


def admit(store, parts):
    for kind, message in parts.items():
        store.component(kind, message, ros_now=100., calibration='calibration_A',
                        provenance={'classification': 'mock_replay_publishers',
                                    'ros_domain_id': 89})


def snapshot(store, **changes):
    arguments = {'ros_now': 100., 'calibration': 'calibration_A', 'idle': True,
                 'provenance': {'classification': 'mock_replay_publishers', 'ros_domain_id': 89}}
    arguments.update(changes)
    return store.snapshot(**arguments)


def test_exact_source_bundle_and_metadata_only_cloud_storage():
    store, _, parts = fixture()
    admit(store, parts)
    state, record = snapshot(store)
    assert state['freshness']['fresh']
    assert state['source_mode'] == 'mock_replay'
    assert not state['execution_allowed'] and not state['scene_freshness_confirmed']
    assert record['image'].data == parts['image'].data
    assert record['transform'].header == record['header']
    assert record['joints'].header.stamp == record['header'].stamp
    assert 'cloud' not in record and not store.pending


@pytest.mark.parametrize('bad', ['stamp', 'dimensions', 'frame', 'rotation', 'missing_rgb'])
def test_incomplete_or_mismatched_components_do_not_replace_current(bad):
    store, _, parts = fixture()
    if bad == 'stamp':
        parts['image'].header = header(99.9)
    elif bad == 'dimensions':
        parts['image'].width = 3
    elif bad == 'frame':
        parts['transform'].header.frame_id = 'map'
    elif bad == 'rotation':
        parts['transform'].transform.rotation.w = float('nan')
    else:
        parts['image'].data = []
    admit(store, parts)
    assert store.latest is None
    assert not snapshot(store)[0]['freshness']['fresh']


def test_duplicate_or_delayed_frames_cannot_renew_source_time():
    store, clock, parts = fixture()
    admit(store, parts)
    clock[0] += 100.
    admit(store, parts)
    state, record = snapshot(store, ros_now=200.)
    assert record['source_time'] == 100.
    assert state['freshness']['age_seconds'] == 100.
    assert not state['freshness']['fresh']
    assert state['generation'] == 1


@pytest.mark.parametrize('changes,reason', [
    ({'calibration': 'calibration_B'}, 'CALIBRATION_CHANGED'),
    ({'calibration': None}, 'CALIBRATION_CHANGED'),
    ({'idle': False}, 'FEEDBACK_STALE_OR_UNAVAILABLE'),
    ({'outcome_unknown': True}, 'OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION'),
    ({'provenance': {'classification': 'unverified'}}, 'SOURCE_UNVERIFIED'),
    ({'provenance': {'classification': 'mock_replay_publishers', 'ros_domain_id': 0}},
     'SOURCE_UNVERIFIED'),
])
def test_reuse_checks_do_not_relax_existing_state_or_source_guards(changes, reason):
    store, _, parts = fixture()
    admit(store, parts)
    result = snapshot(store, **changes)[0]['freshness']
    assert not result['fresh'] and result['reason'] == reason


def test_same_tcp_does_not_hide_joint_movement():
    store, clock, parts = fixture()
    admit(store, parts)
    clock[0] = 100.1
    store.feedback_sample('joints', JointState(header=header(100.1), position=[.02]*6))
    state, _ = snapshot(store, ros_now=100.1)
    assert not state['stationary'] and not state['freshness']['fresh']


def test_feedback_must_correspond_to_the_source_stamp():
    store, _, parts = fixture()
    store.feedback['tcp'].clear()
    admit(store, parts)
    assert store.latest is None


def test_pending_mismatches_stay_bounded_and_newer_bundle_replaces_only_latest():
    store, _, parts = fixture()
    admit(store, parts)
    original = store.latest
    for second in range(101, 110):
        image = deepcopy(parts['image'])
        image.header = header(float(second), 'optical')
        store.component('image', image, ros_now=float(second), calibration='calibration_A')
    assert len(store.pending) <= 4
    assert store.latest is original


def test_current_live_publishers_cannot_relabel_a_replayed_observation():
    store, _, parts = fixture()
    admit(store, parts)
    state, _ = snapshot(store, provenance={
        'classification': 'live_driver_publishers', 'ros_domain_id': 0})
    assert not state['freshness']['fresh']
    assert state['freshness']['reason'] == 'SOURCE_UNVERIFIED'


def test_tcp_change_blocks_reuse_even_if_joint_feedback_stays_constant():
    store, clock, parts = fixture()
    admit(store, parts)
    clock[0] = 100.1
    tcp = PoseStamped(header=header(100.1))
    tcp.pose.orientation.w = 1.
    tcp.pose.position.x = .01
    store.feedback_sample('tcp', tcp)
    state, _ = snapshot(store, ros_now=100.1)
    assert not state['stationary'] and not state['freshness']['fresh']


def test_processing_latency_is_not_stream_interruption_or_semantic_renewal():
    store, clock, parts = fixture()
    clock[0] = 105.
    # Simulate a five-second processing lag with still-current stationary feedback.
    joint = JointState(header=header(105.), position=[0.]*6)
    tcp = PoseStamped(header=header(105.))
    tcp.pose.orientation.w = 1.
    for time in (104.85, 105.):
        clock[0] = time
        joint.header = header(time)
        tcp.header = header(time)
        store.feedback_sample('joints', deepcopy(joint))
        store.feedback_sample('tcp', deepcopy(tcp))
    for kind, message in parts.items():
        store.component(kind, message, ros_now=105., calibration='calibration_A',
                        provenance={'classification': 'mock_replay_publishers',
                                    'ros_domain_id': 89})
    state, _ = snapshot(store, ros_now=105.)
    assert state['freshness']['fresh']
    assert state['freshness']['age_seconds'] == 5.
    assert state['processing_latency_seconds'] == 5.
    assert state['observation_received_age_seconds'] == 0.


def test_read_includes_current_pool_and_plan_summary():
    import json
    from master_node.current_workspace import CurrentWorkspace

    class MockExecutionApi:
        def get_current_pool_summary(self):
            return {
                'available': True, 'fresh': True, 'reason': 'FRESH_FOR_REUSE_ONLY',
                'object_name': 'beige cup', 'expires_in_seconds': 550.0,
                'candidates_count': 6, 'consumed': False
            }

    class MockNode:
        def __init__(self):
            import time
            self.state = 'IDLE'
            self.hand_eye_matrix = None
            self.execution_api = MockExecutionApi()
            import threading
            self.execution_api._cache_lock = threading.Lock()
            self.execution_api._cache = {'beige cup': (123., None)}
            self.execution_api.prepared_grasp_lift = type('obj', (), {
                'record': {
                    'key': 'beige cup', 'revision': 2, 'phase': 'PREPARED',
                    'source_version': 123.,
                    'created': time.monotonic() - 50.0, 'pick_place': True,
                    'destination_name': 'dark rectangular tray'
                }
            })()

        def declare_parameter(self, *a, **k): pass
        def get_parameter(self, name):
            return type('obj', (), {'value': 600.0 if 'ttl' in name else 3.0})()
        def create_subscription(self, *a, **k): pass
        def create_service(self, *a, **k): pass
        def get_clock(self):
            return type('obj', (), {'now': lambda: type('obj', (), {'nanoseconds': 100 * int(1e9)})()})()

    cw = CurrentWorkspace.__new__(CurrentWorkspace)
    cw.node = MockNode()
    cw.ros_now = lambda: 100.0
    cw.calibration = lambda: 'calib_123'
    cw.provenance = lambda n: {'classification': 'mock_replay_publishers', 'ros_domain_id': 89}

    class MockStore:
        def snapshot(self, **kw):
            return {'freshness': {'fresh': True, 'reason': 'FRESH_FOR_REUSE_ONLY'}}, None

    cw.store = MockStore()

    class MockReq:
        include_image = False

    class MockResp:
        success = False
        status = ''
        state_json = ''

    resp = MockResp()
    cw.read(MockReq(), resp)
    assert resp.success is True
    data = json.loads(resp.state_json)
    assert data['current_pool']['available'] is True
    assert data['current_pool']['object_name'] == 'beige cup'
    assert data['current_plan']['available'] is True
    assert data['current_plan']['revision'] == 2
    assert data['current_plan']['destination_name'] == 'dark rectangular tray'
    cw.node.execution_api._cache['beige cup'] = (456., None)
    cw.read(MockReq(), resp)
    changed = json.loads(resp.state_json)['current_plan']
    assert not changed['fresh'] and not changed['available']
    assert changed['expires_in_seconds'] == 0.
    assert changed['reason'] == 'PLAN_SOURCE_NOT_READY'



@pytest.mark.parametrize('completed', [True, False])
def test_image_refresh_requires_completed_synchronized_observation(completed):
    from master_node.current_workspace import CurrentWorkspace
    from threading import Lock
    from types import SimpleNamespace
    calls = []
    stale = {'freshness': {'fresh': False, 'reason': 'STREAM_UNAVAILABLE'},
             'idle': True, 'stationary': True, 'feedback_current': True,
             'input_provenance': {'classification': 'mock_replay_publishers', 'ros_domain_id': 89}}
    fresh = dict(stale, freshness={'fresh': True, 'reason': 'FRESH_FOR_REUSE_ONLY'})
    adapter = CurrentWorkspace.__new__(CurrentWorkspace)
    adapter.refresh_lock = Lock()
    adapter.store = SimpleNamespace(lock=Lock(), generation=1)

    class Changed:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait_for(self, predicate, timeout):
            calls.append(('wait_for_complete', timeout))
            return predicate()

    adapter.store.changed = Changed()

    def trigger(**kwargs):
        calls.append(('scan', kwargs))
        if completed:
            adapter.store.generation += 1
        return True  # This acknowledgement is deliberately insufficient on its own.

    adapter.node = SimpleNamespace(trigger_single_scan=trigger)
    adapter.snapshot = lambda: (fresh if completed else stale, None)
    result, _, evidence = adapter.recover_image(stale, None)
    assert evidence['attempted']
    assert result['freshness']['fresh'] == completed
    assert evidence['status'] == ('SYNCHRONIZED_REFRESH_READY' if completed else
                                  'REFRESH_NOT_CONFIRMED')
    assert len([c for c in calls if c[0] == 'scan']) == 1
    assert not adapter.refresh_lock.locked()


@pytest.mark.parametrize('change', [
    {'idle': False}, {'stationary': False}, {'feedback_current': False},
    {'freshness': {'fresh': False, 'reason': 'OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION'}},
    {'input_provenance': {'classification': 'unverified'}},
])
def test_image_refresh_never_tries_to_repair_missing_safety_evidence(change):
    from master_node.current_workspace import CurrentWorkspace
    stale = {'freshness': {'fresh': False, 'reason': 'SOURCE_MISSING'},
             'idle': True, 'stationary': True, 'feedback_current': True,
             'input_provenance': {'classification': 'live_driver_publishers'}}
    stale.update(change)
    adapter = CurrentWorkspace.__new__(CurrentWorkspace)
    # No node/trigger/lock: any attempted scan fails this test.
    result, _, evidence = adapter.recover_image(stale, None)
    assert not result['freshness']['fresh'] and not evidence['attempted']


@pytest.mark.parametrize('completed', [True, False])
def test_typed_image_read_never_returns_old_pixels_after_ack_only(completed):
    import json
    from types import SimpleNamespace
    from threading import Lock
    from my_actions.srv import GetCurrentWorkspace
    from rclpy.serialization import serialize_message
    from master_node.current_workspace import CurrentWorkspace
    store, _, parts = fixture()
    admit(store, parts)
    record = store.latest
    stale = {'freshness': {'fresh': False, 'reason': 'STREAM_UNAVAILABLE'},
             'idle': True, 'stationary': True, 'feedback_current': True,
             'input_provenance': {'classification': 'mock_replay_publishers', 'ros_domain_id': 89}}
    fresh = dict(stale, freshness={'fresh': True, 'reason': 'FRESH_FOR_REUSE_ONLY'})
    adapter = CurrentWorkspace.__new__(CurrentWorkspace)
    adapter.store, adapter.refresh_lock = store, Lock()
    snapshots = iter([(stale, record), (fresh if completed else stale, record)])
    adapter.snapshot = lambda: next(snapshots)
    adapter.node = SimpleNamespace(execution_api=SimpleNamespace(
        prepared_grasp_lift=SimpleNamespace(record=None)))

    def trigger(**kwargs):
        if completed:
            with store.changed:
                store.generation += 1
                store.changed.notify_all()
        return completed

    adapter.node.trigger_single_scan = trigger
    result = adapter.read(GetCurrentWorkspace.Request(include_image=True),
                          GetCurrentWorkspace.Response())
    assert serialize_message(result)
    assert bool(result.image.data) == completed
    assert json.loads(result.state_json)['image_refresh']['attempted']
    assert (result.status == 'CURRENT_OBSERVATION_READY') == completed
