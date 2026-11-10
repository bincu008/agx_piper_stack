"""One current observation, owned by master; no task scheduler or actuator paths."""

from collections import deque, OrderedDict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import threading
import time

from master_node.workspace_freshness import WorkspaceFreshnessPolicy


def stamp_seconds(header):
    """Read original ROS time, without substituting arrival time."""
    return header.stamp.sec + header.stamp.nanosec * 1.e-9


def calibration_fingerprint(node):
    """Identify only the manual hand-eye matrix; never use computational TF."""
    value = getattr(node, 'hand_eye_matrix', None)
    return hashlib.sha256(value.tobytes()).hexdigest() if value is not None else None


class CurrentObservation:
    """Bounded join of ActiveVision RGB/cloud metadata/manual transform and feedback."""

    def __init__(self, clock=time.monotonic, ttl=600., max_stream_age=3.):
        self.clock = clock
        self.policy = WorkspaceFreshnessPolicy(ttl)
        self.max_stream_age = max_stream_age
        if not math.isfinite(max_stream_age) or max_stream_age <= 0:
            raise ValueError('INVALID_WORKSPACE_STREAM_AGE')
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.pending = OrderedDict()
        self.feedback = {'joints': deque(maxlen=64), 'tcp': deque(maxlen=64)}
        self.latest = None
        self.generation = 0

    def feedback_sample(self, kind, message):
        with self.lock:
            self.feedback[kind].append((self.clock(), message))

    def component(self, kind, message, ros_now, calibration, provenance=None):
        """Admit only exact same-stamp RGB/organized cloud/manual FK*X bundles."""
        stamp = stamp_seconds(message.header)
        if not math.isfinite(stamp) or stamp <= 0 or stamp > ros_now:
            return
        with self.lock:
            if self.latest and stamp <= self.latest['stamp']:
                return  # Replayed/duplicate stamps never renew source age.
            parts = self.pending.setdefault(stamp, {})
            # Dense cloud storage already belongs to ActiveVision. Retain metadata only.
            parts[kind] = (deepcopy(message.header), message.width, message.height) if (
                kind == 'cloud') else message
            while len(self.pending) > 4:
                self.pending.popitem(last=False)
            if set(parts) != {'image', 'cloud', 'transform'}:
                return
            image, cloud, transform = parts['image'], parts['cloud'], parts['transform']
            xyz = transform.transform.translation
            quat = transform.transform.rotation
            finite_transform = all(math.isfinite(getattr(value, key))
                                   for value, keys in ((xyz, 'xyz'), (quat, 'xyzw'))
                                   for key in keys)
            if (not finite_transform or not image.data or
                    cloud[1:] != (image.width, image.height) or
                    cloud[2] < 2 or cloud[0].frame_id != 'base_link' or
                    transform.header != cloud[0] or
                    abs(sum(getattr(transform.transform.rotation, k)**2
                            for k in ('x', 'y', 'z', 'w')) - 1.) > .001):
                self.pending.pop(stamp)
                return
            nearest = {}
            for kind in self.feedback:
                candidates = [m for _, m in self.feedback[kind]
                              if abs(stamp_seconds(m.header) - stamp) <= .20]
                if not candidates:
                    return
                nearest[kind] = min(candidates, key=lambda m: abs(stamp_seconds(m.header)-stamp))
            if (len(nearest['joints'].position) < 6 or
                    nearest['tcp'].header.frame_id != 'base_link'):
                return
            self.generation += 1
            self.latest = {'stamp': stamp, 'source_time': self.clock()-(ros_now-stamp),
                           'image': image, 'header': cloud[0], 'transform': transform,
                           'joints': nearest['joints'], 'tcp': nearest['tcp'],
                           'received_time': self.clock(),
                           'calibration': calibration, 'generation': self.generation,
                           'provenance': deepcopy(provenance or {})}
            self.pending.clear()
            self.changed.notify_all()

    def snapshot(self, *, ros_now, calibration, provenance, idle, outcome_unknown=False):
        """Report sensor reuse evidence only; no claim that semantic memory was refreshed."""
        with self.lock:
            now, record = self.clock(), self.latest
            samples = {k: list(v) for k, v in self.feedback.items()}
            healthy_feedback = all(v and 0 <= now-v[-1][0] <= 1. and
                                   0 <= ros_now-stamp_seconds(v[-1][1].header) <= 1.
                                   for v in samples.values())
            recent = [m for received, m in samples['joints'] if now-received <= .25]
            stable = bool(len(recent) >= 2 and len(recent[0].position) >= 6 and
                          stamp_seconds(recent[-1].header)-stamp_seconds(recent[0].header) >= .10)
            if stable:
                stable = all(len(m.position) >= 6 and all(math.isfinite(x) for x in m.position[:6])
                             and max(abs(a-b) for a, b in zip(
                                 recent[0].position[:6], m.position[:6])) <= .005 for m in recent)
            if stable and record:
                stable = max(abs(a-b) for a, b in zip(
                    record['joints'].position[:6], recent[-1].position[:6])) <= .005
            if stable and record:
                poses = [m.pose for received, m in samples['tcp'] if now-received <= .25]
                poses.insert(0, record['tcp'].pose)
                stable = bool(len(poses) >= 2)
                for pose in poses:
                    values = [getattr(pose.position, k) for k in 'xyz']
                    rotation = [getattr(pose.orientation, k) for k in 'xyzw']
                    reference = [getattr(poses[0].orientation, k) for k in 'xyzw']
                    stable &= (all(math.isfinite(v) for v in values+rotation) and
                               abs(sum(v*v for v in rotation)-1.) <= .001 and
                               math.dist(values, [getattr(poses[0].position, k)
                                                  for k in 'xyz']) <= .002 and
                               abs(sum(a*b for a, b in zip(reference, rotation))) >= .99999)
            same_source = bool(record and all(
                record['provenance'].get(key) == provenance.get(key)
                for key in ('classification', 'ros_domain_id')))
            verified_source = (provenance['classification'] == 'live_driver_publishers' or
                               provenance['classification'] == 'mock_replay_publishers' and
                               provenance.get('ros_domain_id', 0) != 0)
            result = self.policy.check(
                now=now, source_time=record['source_time'] if record else None,
                source_verified=same_source and verified_source,
                calibration_matches=bool(record and calibration is not None and
                                         record['calibration'] == calibration),
                generation_matches=True, feedback_valid=healthy_feedback and stable and idle,
                stream_healthy=bool(record and
                                    0 <= now-record['received_time'] <= self.max_stream_age),
                outcome_unknown=outcome_unknown)
            original_source = record['provenance'] if record else provenance
            original_mode = original_source.get('classification')
            summary = {'generation': record['generation'] if record else 0,
                       'freshness': asdict(result), 'stationary': stable,
                       'idle': idle, 'feedback_current': healthy_feedback,
                       'input_provenance': provenance,
                       'observation_provenance': original_source,
                       'source_mode': {'live_driver_publishers': 'live',
                                       'mock_replay_publishers': 'mock_replay'}.get(
                                           original_mode, 'unverified'),
                       'reuse_ttl_seconds': self.policy.reuse_ttl_seconds,
                       'observation_received_age_seconds': now-record['received_time']
                       if record else None,
                       'processing_latency_seconds': record['received_time']-record['source_time']
                       if record else None,
                       'stream_max_age_seconds': self.max_stream_age,
                       'scene_freshness_confirmed': False, 'execution_allowed': False}
            return summary, record


class CurrentWorkspace:
    """ROS read adapter. Existing master/memory/execution remain authoritative."""

    def __init__(self, node):
        from my_actions.srv import GetCurrentWorkspace
        from geometry_msgs.msg import PoseStamped, TransformStamped
        from rcl_interfaces.msg import ParameterDescriptor
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image, JointState, PointCloud2
        from task_pipeline.provenance import input_provenance

        self.node, self.provenance = node, input_provenance
        self.refresh_lock = threading.Lock()
        descriptor = ParameterDescriptor(read_only=True)
        node.declare_parameter('workspace_reuse_ttl_seconds', 600., descriptor)
        node.declare_parameter('workspace_stream_max_age_seconds', 3., descriptor)
        self.store = CurrentObservation(
            ttl=node.get_parameter('workspace_reuse_ttl_seconds').value,
            max_stream_age=node.get_parameter('workspace_stream_max_age_seconds').value)
        group = ReentrantCallbackGroup()
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.subscriptions = []
        for kind, schema, topic in (
                ('image', Image, '/active_vision/observation_image'),
                ('cloud', PointCloud2, '/active_vision/observation_cloud'),
                ('transform', TransformStamped, '/active_vision/observation_camera_transform')):
            self.subscriptions.append(node.create_subscription(
                schema, topic, lambda m, k=kind: self.component(k, m),
                qos, callback_group=group))
        for kind, schema, topic in (('joints', JointState, '/feedback/joint_states'),
                                    ('tcp', PoseStamped, '/feedback/tcp_pose')):
            self.subscriptions.append(node.create_subscription(
                schema, topic, lambda m, k=kind: self.feedback(k, m),
                20, callback_group=group))
        node.create_service(GetCurrentWorkspace, '/master/get_current_workspace', self.read,
                            callback_group=group)

    def ros_now(self):
        return self.node.get_clock().now().nanoseconds * 1.e-9

    def calibration(self):
        return calibration_fingerprint(self.node)

    def component(self, kind, message):
        """Update sensor evidence and revoke pools whose manual calibration changed."""
        calibration = self.calibration()
        self.node.execution_api.observe_pool_feedback('calibration', calibration)
        self.store.component(kind, message, self.ros_now(), calibration,
                             self.provenance(self.node))

    def feedback(self, kind, message):
        """Revoke unconsumed pool readiness after measured camera/arm displacement."""
        self.store.feedback_sample(kind, message)
        self.node.execution_api.observe_pool_feedback(kind, message)

    def snapshot(self):
        """Read current evidence; historical planning never establishes currentness."""
        session = getattr(self.node.execution_api.prepared_grasp_lift, 'record', None)
        return self.store.snapshot(
            ros_now=self.ros_now(), calibration=self.calibration(),
            provenance=self.provenance(self.node), idle=self.node.state == 'IDLE',
            outcome_unknown=bool(session and session.get('phase') == 'STATE_UNKNOWN'))

    def recover_image(self, summary, record):
        """One bounded sensing-only refresh; acknowledgement alone never means captured."""
        reason = summary['freshness']['reason']
        evidence = {'attempted': False, 'status': 'NOT_NEEDED_OR_NOT_ELIGIBLE'}
        if (reason not in ('SOURCE_MISSING', 'STREAM_UNAVAILABLE', 'REFRESH_REQUIRED_EXPIRED')
                or not summary.get('idle') or not summary.get('stationary')
                or not summary.get('feedback_current')
                or summary.get('input_provenance', {}).get('classification') not in
                ('mock_replay_publishers', 'live_driver_publishers')
                or (summary.get('input_provenance', {}).get('classification') ==
                    'mock_replay_publishers' and
                    summary.get('input_provenance', {}).get('ros_domain_id', 0) == 0)):
            return summary, record, evidence
        if not self.refresh_lock.acquire(blocking=False):
            evidence['status'] = 'REFRESH_ALREADY_IN_PROGRESS'
            return summary, record, evidence
        try:
            with self.store.lock:
                generation = self.store.generation
            evidence['attempted'] = True
            acknowledged = self.node.trigger_single_scan(wait_for_result=True, timeout_sec=1.)
            if acknowledged:
                with self.store.changed:
                    self.store.changed.wait_for(lambda: self.store.generation > generation,
                                                timeout=1.)
            summary, record = self.snapshot()
            evidence['status'] = ('SYNCHRONIZED_REFRESH_READY' if
                                  summary['freshness']['fresh'] else 'REFRESH_NOT_CONFIRMED')
            return summary, record, evidence
        except Exception as error:
            evidence.update(status='REFRESH_NOT_CONFIRMED', error=type(error).__name__)
            return summary, record, evidence
        finally:
            self.refresh_lock.release()

    def read(self, request, response):
        session = getattr(self.node.execution_api.prepared_grasp_lift, 'record', None)
        summary, record = self.snapshot()
        if request.include_image:
            summary, record, refresh = self.recover_image(summary, record)
            summary['image_refresh'] = refresh
        response.success = True  # A read completed; inspect freshness, not this flag.
        response.status = 'CURRENT_OBSERVATION_READY' if summary['freshness']['fresh'] else (
            'REFRESH_REQUIRED:' + summary['freshness']['reason'])
        if record:
            response.observation_header = record['header']
            response.camera_transform = record['transform']
            response.source_joints, response.source_tcp = record['joints'], record['tcp']
            if request.include_image and summary['freshness']['fresh']:
                response.image = record['image']
        exec_api = getattr(self.node, 'execution_api', None)
        if exec_api and hasattr(exec_api, 'get_current_pool_summary'):
            summary['current_pool'] = exec_api.get_current_pool_summary()
        else:
            summary['current_pool'] = None
        if session:
            now_mono = time.monotonic()
            plan_age = now_mono - session.get('created', now_mono)
            pool = summary.get('current_pool') or {}
            source_matches = False
            if hasattr(exec_api, '_cache_lock'):
                with exec_api._cache_lock:
                    cached = exec_api._cache.get(session.get('key'))
                    source_matches = bool(cached and cached[0] == session.get('source_version'))
            eligible = bool(0. <= plan_age < 600. and session.get('phase') == 'PREPARED'
                            and source_matches and pool.get('fresh'))
            expires_in = min(max(0., 600. - plan_age),
                             pool.get('expires_in_seconds', 0.)) if eligible else 0.
            summary['current_plan'] = {
                'available': eligible,
                'reason': 'FRESH_FOR_REUSE_ONLY' if eligible else 'PLAN_SOURCE_NOT_READY',
                'object_name': session.get('key'),
                'revision': session.get('revision'),
                'phase': session.get('phase'),
                'pick_place': session.get('pick_place', False),
                'destination_name': session.get('destination_name'),
                'expires_in_seconds': expires_in,
                'fresh': eligible,
            }
        else:
            summary['current_plan'] = None
        response.state_json = json.dumps(summary, allow_nan=False)
        return response
