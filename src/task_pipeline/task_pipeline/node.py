"""Typed, bounded API dispatcher shared by deployment and benchmarks."""

import json
import threading
import time
import uuid
from collections import deque

from geometry_msgs.msg import TransformStamped

from my_actions.msg import ApiCallTrace, PipelineTiming
from my_actions.srv import (
    CaptureObservation, GetPipelineFunctions, GetPipelineMetrics, RunPipelineStep,
)

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message, serialize_message

from sensor_msgs.msg import Image, PointCloud2

from std_srvs.srv import Trigger

from .registry import FUNCTIONS, catalog


_LOCAL_CONTROLS = {
    'run_step': '_run_step', 'submit_steps': '_submit', 'get_step_context': '_get',
    'clear_step_context': '_clear', 'store_step_result': '_import'}
_CONTROL_FUNCTIONS = {*_LOCAL_CONTROLS, 'append_task', 'execute_pipeline', 'flush_pipeline'}


def response_success(response):
    """Read the existing typed status without parsing human-readable logs."""
    if hasattr(response, 'success'):
        return response.success
    if hasattr(response, 'valid'):
        return response.valid
    if hasattr(response, 'state_json'):
        try:
            return bool(response.state_json) and 'error' not in json.loads(response.state_json)
        except (ValueError, TypeError):
            return False
    return True


def response_status(response):
    """Preserve backend failure details."""
    for name in ('status', 'failure_reason', 'message'):
        value = getattr(response, name, '')
        if value:
            return value
    return 'OK' if response_success(response) else 'FAILED'


def set_failure(response, reason):
    """Return a failure using the operation's existing response schema."""
    for field in ('success', 'valid'):
        if hasattr(response, field):
            setattr(response, field, False)
    for field in ('status', 'failure_reason', 'message'):
        if hasattr(response, field):
            setattr(response, field, reason)
            return response
    if hasattr(response, 'state_json'):
        response.state_json = json.dumps({'error': reason})
    return response


class PipelineNode(Node):
    """Dispatch to existing engines; never publish actuator commands."""

    def __init__(self, runtime=None):
        """Create bounded clients and observation subscriptions."""
        super().__init__('pipeline_node')
        self._runtime = runtime
        self.declare_parameter('request_timeout_sec', 600.0)
        self.declare_parameter('capture_timeout_sec', 30.0)
        self._group = ReentrantCallbackGroup()
        self._active = threading.BoundedSemaphore(8)
        self._capture_lock = threading.Lock()
        self._lock = threading.Lock()
        self._samples = deque(maxlen=512)
        self._observations = {kind: deque(maxlen=8) for kind in ('image', 'cloud', 'transform')}
        self._timing_pub = self.create_publisher(PipelineTiming, '/pipeline/timings', 100)
        self._trace_pub = self.create_publisher(ApiCallTrace, '/pipeline/api_traces', 100)
        self.create_subscription(PipelineTiming, '/pipeline/timings', self._record, 100,
                                 callback_group=self._group)
        self._backend_clients = {}
        for name, function in FUNCTIONS.items():
            # Numerical operations are dispatched by the master runtime. Do
            # not add unused clients (and QoS waitables) to this executor.
            if name in _CONTROL_FUNCTIONS and (runtime is None or name not in _LOCAL_CONTROLS):
                self._backend_clients[name] = self.create_client(
                    function.service_type, function.backend, callback_group=self._group)
            self.create_service(function.service_type, function.endpoint,
                                lambda req, res, key=name: self._dispatch(key, req, res),
                                callback_group=self._group)
        self._master_step = (self.create_client(
            RunPipelineStep, '/master/run_step', callback_group=self._group)
            if runtime is None else None)
        self._scan = self.create_client(Trigger, '/active_vision/trigger_scan',
                                        callback_group=self._group)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        for kind, message_type, topic in (
                ('image', Image, '/active_vision/observation_image'),
                ('cloud', PointCloud2, '/active_vision/observation_cloud'),
                ('transform', TransformStamped, '/active_vision/observation_camera_transform')):
            self.create_subscription(message_type, topic,
                                     lambda msg, key=kind: self._observe(key, msg), qos,
                                     callback_group=self._group)
        self.create_service(CaptureObservation, '/pipeline/_capture_observation', self._capture,
                            callback_group=self._group)
        self.create_service(GetPipelineFunctions, '/pipeline/list_functions',
                            self._catalog, callback_group=self._group)
        self.create_service(GetPipelineMetrics, '/pipeline/get_metrics',
                            self._metrics, callback_group=self._group)
        self.get_logger().info('Shared task API ready at /pipeline/list_functions')

    def _record(self, sample):
        with self._lock:
            self._samples.append(sample)

    def _observe(self, kind, message):
        with self._lock:
            self._observations[kind].append(message)

    def _catalog(self, request, response):
        response.functions_json = json.dumps(catalog())
        return response

    def _metrics(self, request, response):
        with self._lock:
            response.samples = list(self._samples)
        return response

    def _rpc(self, client, request, deadline=None):
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('BACKEND_UNAVAILABLE: ' + client.srv_name)
        future = client.call_async(request)
        complete = threading.Event()
        future.add_done_callback(lambda _: complete.set())
        if deadline is None:
            deadline = time.monotonic() + self.get_parameter('request_timeout_sec').value
        while not complete.wait(0.02):
            if not rclpy.ok():
                raise RuntimeError('PIPELINE_SHUTDOWN')
            if time.monotonic() >= deadline:
                # A ROS service timeout does not stop its backend. Report this
                # explicitly; never treat it as permission to retry execution.
                raise RuntimeError('BACKEND_TIMEOUT_STATE_UNKNOWN: ' + client.srv_name)
        if future.exception():
            raise RuntimeError(str(future.exception()))
        return future.result()

    def _publish(self, name, request, response, started, backend, input_mode, outcome=None):
        if (name == 'run_step' and request.function_name in FUNCTIONS and
                response.result.response_cdr):
            spec = FUNCTIONS[request.function_name]
            inner_request = deserialize_message(
                bytes(request.request_cdr), spec.service_type.Request)
            inner_response = deserialize_message(
                bytes(response.result.response_cdr), spec.service_type.Response)
            return self._publish(request.function_name, inner_request, inner_response,
                                 started, spec.backend, spec.input_mode,
                                 (response.result.success, response.result.status))
        sample = PipelineTiming(
            operation_id=uuid.uuid4().hex, function_name=name,
            backend=backend, input_mode=input_mode, scope='api_inclusive',
            wall_seconds=time.monotonic() - started,
            success=response_success(response) if outcome is None else outcome[0],
            status=response_status(response) if outcome is None else outcome[1],
            full_motion=bool(getattr(request, 'plan_pick_place_motion', False)))
        if name == 'capture_observation_backend':
            sample.scope = 'capture_delivery'
        if name == 'ground_object':
            sample.input_mode = (
                'grounding_with_presence' if request.verify_presence else 'grounding_only')
            sample.object_name = request.target
        elif name == 'infer_vision':
            prompt = request.user_prompt
            sample.input_mode = ('box' if prompt.startswith('Please provide the bounding box') else
                                 'inventory' if prompt.startswith('Locate every') else
                                 'presence' if request.max_tokens == 16 else 'general')
        if name == 'plan_layout':
            sample.full_motion = True
        if name in ('plan_placement', 'validate_placements'):
            sample.input_mode = 'frozen_scene' if request.use_layout_scene else 'legacy_pose_only'
        header = getattr(request, 'observation_header', None)
        if header is None:
            header = getattr(request, 'target_observation_header', None) or getattr(
                request, 'previous_observation_header', None)
        if header is None and hasattr(request, 'image'):
            header = request.image.header
        if header is None and getattr(request, 'use_layout_scene', False):
            header = request.source_object_cloud.header
        if name == 'capture_observation' and response.success:
            header = response.cloud.header
        if header is not None:
            sample.observation_header = header
        sample.object_name = sample.object_name or getattr(request, 'object_name', '') or getattr(
            request, 'target_description', '')
        self._timing_pub.publish(sample)
        self.get_logger().info('[PIPELINE TIMING] ' + json.dumps({
            'operation_id': sample.operation_id, 'function': name,
            'seconds': sample.wall_seconds, 'input_mode': sample.input_mode,
            'full_motion': sample.full_motion, 'success': sample.success,
        }))

    def _dispatch(self, name, request, response):
        received_ns = time.monotonic_ns()
        started = time.monotonic()
        outcome = None
        published = False
        spec = FUNCTIONS[name]
        admitted = self._active.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('PIPELINE_BUSY: bounded active operations, no queued requests')
            if name in _CONTROL_FUNCTIONS:
                runtime = getattr(self, '_runtime', None)
                if runtime is not None and name in _LOCAL_CONTROLS:
                    if name == 'run_step':
                        def completed(inner_request, inner_response, inner_outcome):
                            nonlocal published
                            inner_spec = FUNCTIONS[request.function_name]
                            self._publish(request.function_name, inner_request, inner_response,
                                          started, inner_spec.backend, inner_spec.input_mode,
                                          inner_outcome)
                            published = True
                        response = runtime._run_step(request, response, on_complete=completed)
                    else:
                        response = getattr(runtime, _LOCAL_CONTROLS[name])(request, response)
                else:
                    response = self._rpc(self._backend_clients[name], request)
            else:
                runtime = getattr(self, '_runtime', None)
                if runtime is not None:
                    response, outcome = runtime.run_typed(name, request)
                else:
                    envelope = RunPipelineStep.Request(
                        function_name=name, request_cdr=serialize_message(request))
                    result = self._rpc(self._master_step, envelope)
                    if not result.result.response_cdr:
                        raise RuntimeError(result.status)
                    response = deserialize_message(
                        bytes(result.result.response_cdr), spec.service_type.Response)
                    outcome = (result.result.success, result.result.status)
        except Exception as error:
            outcome = (False, str(error))
            response = set_failure(response, str(error))
        finally:
            if admitted:
                self._active.release()
            if not published:
                self._publish(name, request, response, started, spec.backend, spec.input_mode, outcome)
            if name == 'run_step':
                self._trace_pub.publish(ApiCallTrace(
                    operation_id=request.step_id, function_name=request.function_name,
                    component='dispatcher', received_ns=received_ns,
                    finished_ns=time.monotonic_ns(), success=response.success))
        return response

    @staticmethod
    def _stamp(message):
        stamp = message.header.stamp
        return stamp.sec * 1000000000 + stamp.nanosec

    def _capture(self, request, response):
        started = time.monotonic()
        admitted = self._capture_lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('CAPTURE_BUSY')
            with self._lock:
                previous = max(
                    (self._stamp(msg) for msg in self._observations['image']), default=0)
            deadline = time.monotonic() + self.get_parameter('capture_timeout_sec').value
            ack = self._rpc(self._scan, Trigger.Request(), deadline)
            if not ack.success:
                raise RuntimeError('CAPTURE_TRIGGER_FAILED: ' + ack.message)
            while rclpy.ok() and time.monotonic() < deadline:
                with self._lock:
                    maps = {key: {self._stamp(msg): msg for msg in messages}
                            for key, messages in self._observations.items()}
                common = set(maps['image']) & set(maps['cloud']) & set(maps['transform'])
                fresh = [stamp for stamp in common if stamp > previous]
                if fresh:
                    stamp = max(fresh)
                    cloud = maps['cloud'][stamp]
                    transform = maps['transform'][stamp]
                    if (cloud.header.frame_id != 'base_link' or
                            transform.header.frame_id != 'base_link'):
                        raise RuntimeError('CAPTURE_MANUAL_FRAME_MISMATCH')
                    response.image = maps['image'][stamp]
                    response.cloud = cloud
                    response.camera_transform = transform
                    response.success = True
                    response.status = 'SYNCHRONIZED_OBSERVATION_READY'
                    return response
                time.sleep(0.02)
            raise RuntimeError('SYNCHRONIZED_CAPTURE_TIMEOUT')
        except Exception as error:
            response = set_failure(response, str(error))
            return response
        finally:
            if admitted:
                self._capture_lock.release()
            self._publish('capture_observation_backend', request, response, started,
                          '/active_vision/trigger_scan', 'rgbd')


def main():
    """Start the non-actuating task dispatcher."""
    rclpy.init()
    node = PipelineNode()
    executor = MultiThreadedExecutor(num_threads=16)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
