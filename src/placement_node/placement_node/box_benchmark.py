"""Fuse known boxes into a mock scene and benchmark edge/stack placement."""

import copy
from collections import deque
from contextlib import nullcontext
import json
import uuid
from pathlib import Path
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from geometry_msgs.msg import Vector3
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray
from my_actions.msg import PlacementCandidate, StepBinding
from my_actions.srv import CaptureObservation, GroundObject, FindCloudGrasps, PinLayoutObservation, PlanPlacement, SegmentGroundedImage, RunPipelineStep, VerifyPresence
from my_actions.srv import ClearStepContext, PrepareCloudScene

from .box_fixture import replace_fixture_floor_noise, box_fixture, box_top, box_contact_seeds, tray_floor_mask
from .node import PlacementNode
from task_pipeline.metrics import summarize
from task_pipeline.timing import publish_timing
from my_actions.msg import ApiCallTrace, PipelineTiming
from task_pipeline.api_metrics import summarize_api_calls
from task_pipeline.registry import endpoint, FUNCTIONS
from task_pipeline.placement import search_placements, validate_placements

from .packing import search_heightmap_placements
from .packing_api import make_search_request, decode_search_response
from .perception import extract_segmented_destination_cloud, parse_grounding_box
from .transforms import attached_object_delta, transform_points


class BoxBenchmark(Node):
    """A bounded worker owns the frozen scene; ROS callbacks remain responsive."""

    _point = staticmethod(PlacementNode._point)
    _pose_matrix = staticmethod(PlacementNode._pose_matrix)

    def __init__(self):
        super().__init__('box_benchmark')
        params = {
            'box_count': 10, 'box_size_m': 0.08, 'box_gap_m': 0.04,
            'source_radius_m': 0.5, 'placement_policy': 'edge_then_stack',
            'candidate_count': 128, 'grid_resolution_m': 0.005,
            'floor_refinement_candidate_count': 0, 'floor_refinement_yaw_step_deg': 5.0,
            'yaw_step_deg': 15.0, 'max_pile_height_m': 0.30,
            'stack_minimum_coverage': 0.6,
            'grasp_approach_tilts_deg': [0.0, 15.0, -15.0, 30.0, -30.0,
                                         45.0, -45.0, 60.0, -60.0],
            'request_timeout_sec': 600.0, 'auto_run': False,
            'plan_pick_place_motion': True,
            'optimize_pipeline': True, 'incremental_grasp_fallback': False,
            'grasp_early_stop_count': 4,
            'placement_early_stop_count': 32, 'placement_validation_batch_size': 8,
            'destination': 'the complete visible placement surface of the tray',
            'output_directory': str(Path.home() / 'agx_piper_data' / 'reports' / 'box_benchmark'),
        }
        for name, value in params.items():
            self.declare_parameter(name, value)
        self._group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._worker = None
        self._run_service_active = False
        self._pipeline_timing_pub = self.create_publisher(PipelineTiming, '/pipeline/timings', 100)
        self._bridge = CvBridge()
        self._report = {}
        self._function_samples = deque(maxlen=2048)
        self._api_calls = []
        self._api_traces = {}
        self.create_subscription(ApiCallTrace, '/pipeline/api_traces',
                                 self._observe_api_trace, 100, callback_group=self._group)
        self.create_subscription(PipelineTiming, '/pipeline/timings',
                                 self._observe_function_timing, 100, callback_group=self._group)
        self.add_on_set_parameters_callback(self._validate_parameters)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._qos = qos
        self._fused_pub = self.create_publisher(PointCloud2,
            '/box_benchmark/fused_cloud', qos)
        self._scene_pub = self.create_publisher(PointCloud2,
            '/box_benchmark/scene_cloud', qos)
        self._boxes_pub = self.create_publisher(PointCloud2,
            '/box_benchmark/boxes_cloud', qos)
        self._marker_pub = self.create_publisher(MarkerArray,
            '/box_benchmark/placements', qos)
        self._tray_pub = self.create_publisher(PointCloud2,
            '/box_benchmark/tray_cloud', qos)
        self._status_pub = self.create_publisher(String,
            '/box_benchmark/status', qos)
        self._step_client = self.create_client(RunPipelineStep, endpoint('run_step'),
            callback_group=self._group)
        self._step_context = ''
        self._step_ids = {}
        self._step_contexts = []
        self._benchmark_clients = {
            'clear_context': self.create_client(ClearStepContext, endpoint('clear_step_context'),
                callback_group=self._group),
            'gate': self.create_client(GetParameters,
                '/piper_master_control/get_parameters', callback_group=self._group),

        }
        self.create_service(Trigger, '/box_benchmark/run', self._start,
                            callback_group=self._group)
        self.create_service(Trigger, '/box_benchmark/cancel', self._stop,
                            callback_group=self._group)
        self.create_service(Trigger, '/box_benchmark/reset', self._reset,
                            callback_group=self._group)
        self._auto_pending = bool(self.get_parameter('auto_run').value)
        self.create_timer(1.0, self._auto, callback_group=self._group)
        self.get_logger().info('Box benchmark ready: /box_benchmark/run; no motion execution')

    def _observe_function_timing(self, sample):
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                self._function_samples.append(sample)

    def _observe_api_trace(self, sample):
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                if len(self._api_traces) < 2048 or sample.operation_id in self._api_traces:
                    self._api_traces.setdefault(sample.operation_id, {})[sample.component] = sample

    def _validate_parameters(self, parameters):
        _ = parameters
        busy = self._worker is not None and self._worker.is_alive()
        return SetParametersResult(successful=not busy,
            reason='Cancel the running benchmark before changing parameters' if busy else '')

    def _value(self, key):
        return self.get_parameter(key).value

    def _auto(self):
        if (self._auto_pending and self._step_client.service_is_ready() and
                all(c.service_is_ready() for c in self._benchmark_clients.values())):
            self._auto_pending = False
            self._start(Trigger.Request(), Trigger.Response())

    def _start(self, request, response):
        _ = request
        with self._lock:
            if self._run_service_active or (self._worker is not None and self._worker.is_alive()):
                response.success = False
                response.message = 'BENCHMARK_BUSY: no queued runs'
                return response
            self._cancel.clear()
            if hasattr(self, '_function_samples'):
                self._function_samples.clear()
            self._api_calls = []
            self._api_traces = {}
            self._run_service_active = True
            self._report = {}
            worker = threading.Thread(target=self._run, daemon=True)
            self._worker = worker
            worker.start()
        try:
            # Only this service callback waits. The worker and the other
            # executor threads continue handling status, cancellation and RPCs.
            worker.join()
            report = self._report.copy()
            if report.get('stage') not in ('COMPLETE', 'COMPLETE_WITH_FAILURES', 'FAILED', 'CANCELLED'):
                report.update(stage='FAILED', failure='WORKER_EXITED_WITHOUT_FINAL_STATUS')
            response.success = report.get('stage') == 'COMPLETE'
            response.message = json.dumps({key: report[key] for key in (
                'stage', 'accepted', 'failed', 'edge_count', 'stack_count',
                'elapsed_seconds', 'failure') if key in report})
            return response
        finally:
            with self._lock:
                self._run_service_active = False

    def _stop(self, request, response):
        _ = request
        self._cancel.set()
        response.success = True
        response.message = 'Cancellation requested; pending calculations are discarded'
        return response

    def _reset(self, request, response):
        _ = request
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                response.success = False
                response.message = 'Cancel and wait for the running worker before reset'
                return response
        PlacementNode._publish_layout_markers(self, Header(frame_id='base_link'), [])
        response.success = True
        response.message = 'Cleared placement markers; next run starts from the mock observation'
        return response

    def _check_cancel(self):
        if self._cancel.is_set():
            raise RuntimeError('BENCHMARK_CANCELLED')

    def _call(self, key, request, bindings=()):
        started_ns = time.monotonic_ns()
        self._check_cancel()
        names = {'capture': 'capture_observation', 'pin': 'pin_grasp_observation',
                 'ground': 'ground_object', 'presence': 'verify_presence', 'segment': 'segment_object',
                 'grasp': 'calculate_cloud_grasps', 'place': 'validate_placements',
                 'search': 'search_placements'}
        names['scene'] = 'prepare_cloud_scene'
        function = names.get(key)
        client = self._step_client if function else self._benchmark_clients[key]
        if function:
            request = RunPipelineStep.Request(
                context_id=self._step_context, function_name=function,
                step_id=uuid.uuid4().hex, request_cdr=serialize_message(request),
                bindings=list(bindings))
        call = {'operation_id': request.step_id, 'function': function,
                'started_ns': started_ns, 'success': False} if function else None
        try:
            if not client.wait_for_service(timeout_sec=10):
                raise RuntimeError(f'SERVICE_UNAVAILABLE: {key}')
            if call is not None:
                call['sent_ns'] = time.monotonic_ns()
            future = client.call_async(request)
            done = threading.Event()

            def received(_):
                if call is not None:
                    call['response_received_ns'] = time.monotonic_ns()
                done.set()

            future.add_done_callback(received)
            deadline = time.monotonic() + float(self._value('request_timeout_sec'))
            while not done.wait(0.1):
                self._check_cancel()
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'SERVICE_TIMEOUT: {key}')
            self._check_cancel()
            if future.exception() is not None:
                raise RuntimeError(f'SERVICE_FAILED: {key}: {future.exception()}')
            result = future.result()
            if function:
                if not result.result.response_cdr:
                    raise RuntimeError(result.status)
                decoded = deserialize_message(bytes(result.result.response_cdr),
                                              FUNCTIONS[function].service_type.Response)
                call['success'] = result.result.success
                call['status'] = result.result.status
                self._step_ids[key] = request.step_id
                return decoded
            return result
        except Exception as error:
            if call is not None:
                call['status'] = str(error)
            raise
        finally:
            if call is not None:
                call['finished_ns'] = time.monotonic_ns()
                # Missing delivery stamps denote an unavailable/timed-out service.
                call.setdefault('sent_ns', call['finished_ns'])
                call.setdefault('response_received_ns', call['finished_ns'])
                with self._lock:
                    self._api_calls.append(call)

    def _status(self, stage, **data):
        self._report.update(data)
        self._report['stage'] = stage
        self._report['elapsed_seconds'] = time.perf_counter() - self._started
        text = json.dumps(self._report)
        self._status_pub.publish(String(data=text))
        self.get_logger().info(text)
        (self._directory / 'report.json').write_text(json.dumps(self._report, indent=2))

    def _capture(self):
        result = self._call('capture', CaptureObservation.Request())
        if not result.success:
            raise RuntimeError(result.status)
        cloud, image = result.cloud, result.image
        if cloud.header.frame_id != 'base_link' or cloud.height <= 1:
            raise RuntimeError('REQUIRES_ACTIVE_VISION_MANUAL_BASE_CLOUD')
        for _ in range(30):
            self._check_cancel()
            request = PinLayoutObservation.Request(observation_header=cloud.header)
            if self._call('pin', request).success:
                return cloud, image
            self._cancel.wait(0.1)
        raise RuntimeError('OBSERVATION_PIN_FAILED')

    def _cloud(self, header, points):
        return point_cloud2.create_cloud_xyz32(header, np.asarray(points, dtype=np.float32).reshape(-1, 3))

    def _scene_cloud(self, header, points, flags):
        fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(['x', 'y', 'z'])]
        fields.append(PointField(name='support', offset=12,
                                 datatype=PointField.UINT8, count=1))
        rows = np.empty(len(points), dtype=point_cloud2.dtype_from_fields(fields))
        for axis, field in enumerate(('x', 'y', 'z')):
            rows[field] = np.asarray(points)[:, axis]
        rows['support'] = np.asarray(flags, dtype=np.uint8)
        return point_cloud2.create_cloud(header, fields, rows)

    def _finish_placement_timing(self, name, attempt, started):
        attempt['calculation_seconds'] = time.perf_counter() - started
        result = 'FAILED' if 'failure' in attempt else 'ACCEPTED'
        mode = 'FLOOR' if attempt['mode'] == 'edge' else 'STACK'
        self.get_logger().info(
            f'[BOX TIMING] {name} {mode} '
            f'{attempt["search_stage"]}: {attempt["calculation_seconds"]:.2f}s '
            f'(packing={attempt["packing_seconds"]:.2f}s, '
            f'validation={attempt["validation_seconds"]:.2f}s), result={result}')

    def _log_box_timing(self, row, elapsed):
        floor = sum(a['calculation_seconds'] for a in row['attempts'] if a['mode'] == 'edge')
        stack = sum(a['calculation_seconds'] for a in row['attempts'] if a['mode'] == 'stack')
        row.update(floor_placement_seconds=floor, stack_placement_seconds=stack,
                   placement_seconds=floor + stack, total_seconds=elapsed)
        result = 'FLOOR' if row.get('mode') == 'EDGE' else row.get('mode', 'FAILED')
        row['result'] = result

    def _print_final_summary(self):
        with getattr(self, '_lock', nullcontext()):
            samples = list(getattr(self, '_function_samples', []))
        header = getattr(self, '_run_observation_header', None)
        samples = [sample for sample in samples if header is not None and
                   sample.observation_header.stamp == header.stamp]
        self._report['function_summary'] = summarize(samples)
        with getattr(self, '_lock', nullcontext()):
            self._report['api_timing'] = summarize_api_calls(
                list(getattr(self, '_api_calls', [])), dict(getattr(self, '_api_traces', {})))
        self._report['api_timing']['non_api_seconds'] = (
            self._report.get('elapsed_seconds', 0.0) -
            self._report['api_timing']['total_api_seconds'])
        for entry in self._report['api_timing']['functions']:
            phases = entry['median_phases_seconds']
            self.get_logger().info(
                f"[API END-TO-END] {entry['function']} calls={entry['calls']} "
                f"total={entry['median_total_seconds']:.4f}s phases=" + json.dumps(phases))
        self._report['timing_scope_note'] = (
            'API and orchestration times include child operations; do not sum nested scopes. '
            'Known-cloud grasp calls omit HGGD; this is not a full deployment runtime estimate.')
        directory = getattr(self, '_directory', None)
        if directory is not None and hasattr(directory, 'is_dir') and directory.is_dir():
            (directory / 'report.json').write_text(json.dumps(self._report, indent=2))
        lines = ['[BOX SUMMARY] All calculation results (seconds)',
                 'Object   Result   Grasp   Floor   Stack   Placement   Total']
        for row in self._report.get('objects', []):
            lines.append(
                f'{row["object"]:8} {row.get("result", "INCOMPLETE"):8} '
                f'{row.get("grasp_seconds", 0):7.2f} '
                f'{row.get("floor_placement_seconds", 0):7.2f} '
                f'{row.get("stack_placement_seconds", 0):7.2f} '
                f'{row.get("placement_seconds", 0):9.2f} '
                f'{row.get("total_seconds", 0):7.2f}')
            for attempt in row.get('attempts', []):
                if 'motion_status' in attempt:
                    lines.append(
                        f'  {attempt["mode"]}: '
                        f'lift={attempt["lift_seconds"]:.3f}s '
                        f'transfer={attempt["transfer_seconds"]:.3f}s '
                        f'descent search={attempt["descent_search_seconds"]:.3f}s '
                        f'status={attempt["motion_status"]}')
            if row.get('failure'):
                lines.append(f'  {row["object"]}: {row["failure"]}')
        lines.append(
            f'Stage={self._report.get("stage", "FAILED")} '
            f'accepted={self._report.get("accepted", 0)} '
            f'failed={self._report.get("failed", 0)} '
            f'floor={self._report.get("edge_count", 0)} '
            f'stack={self._report.get("stack_count", 0)} '
            f'pipeline={self._report.get("elapsed_seconds", 0):.2f}s '
            f'tray_grounding={self._report.get("grounding_seconds", 0):.2f}s')
        if self._report['function_summary']:
            lines.append('Shared functions: calls / median seconds / input mode / scope')
            for sample in self._report['function_summary']:
                lines.append(f"  {sample['function']}: {sample['calls']} / "
                             f"{sample['median_seconds']:.3f}s / {sample['input_mode']} / "
                             f"{sample['scope']}")
        self.get_logger().info('\n'.join(lines))

    def _plan(self, source, grasps, scene, flags, tray, top, mode, header, name, refine=False):
        started = time.perf_counter()
        try:
            accepted, attempt = self._plan_impl(source, grasps, scene, flags, tray, top,
                                                mode, header, name, refine=refine)
            if (accepted is None and bool(self._value('optimize_pipeline')) and not refine and
                    attempt.get('failure') != 'NO_STABLE_PACKING_CANDIDATE'):
                self._report['objects'][-1]['attempts'].append(attempt)
                cached_packing = None
                incremental = bool(self._value('incremental_grasp_fallback'))
                if not incremental:
                    self._expand_grasps()
                current_grasps = grasps
                while True:
                    # First try the complete placement grid with the existing
                    # grasp pool. More grasps are needed only if that fails.
                    accepted, attempt = self._plan_impl(
                        source, current_grasps, scene, flags, tray, top, mode, header, name,
                        exhaustive=True, packing_result=cached_packing)
                    cached_packing = getattr(self, '_last_packing', None)
                    if (accepted is not None or not incremental or
                            attempt.get('failure') == 'NO_STABLE_PACKING_CANDIDATE'):
                        break
                    previous_count = len(grasps)
                    self._expand_grasps(previous_count + max(1, int(self._value('grasp_early_stop_count'))))
                    if len(grasps) == previous_count:
                        break
                    current_grasps = grasps[previous_count:]
                    self._report['objects'][-1]['attempts'].append(attempt)
            return accepted, attempt
        except Exception as exception:
            self._report['objects'][-1]['attempts'].append({
                'mode': mode, 'search_stage': 'refined' if refine else 'primary',
                'calculation_seconds': time.perf_counter() - started,
                'failure': str(exception)})
            raise

    def _search_placements_api(self, header, name, *args, **kwargs):
        return decode_search_response(self._call(
            'search', make_search_request(header, name, *args, **kwargs)))

    def _plan_impl(self, source, grasps, scene, flags, tray, top, mode, header, name, refine=False, exhaustive=False, packing_result=None):
        calculation_started = time.perf_counter()
        candidate_count = int(self._value(
            'floor_refinement_candidate_count' if refine else 'candidate_count'))
        yaw_step = float(self._value(
            'floor_refinement_yaw_step_deg' if refine else 'yaw_step_deg'))
        fast = bool(self._value('optimize_pipeline')) and not refine and not exhaustive
        file_mode = mode + ('_refined' if refine else '_fast' if fast else '')
        bottom = source[source[:, 2] <= source[:, 2].min() + 0.0001]
        reused_packing = packing_result is not None
        if reused_packing:
            candidates, diagnostics = packing_result
        else:
            candidates, diagnostics = self._search_placements_api(header, name,
                source, scene, tray, 'upright',
                candidate_count=candidate_count,
                resolution=float(self._value('grid_resolution_m')),
                upright_yaw_step_deg=yaw_step,
                maximum_pile_height_m=float(self._value('max_pile_height_m')),
                bottom_points=bottom, support_mode=mode,
                stack_minimum_coverage=float(self._value('stack_minimum_coverage')),
                early_stop_candidate_count=int(self._value('placement_early_stop_count')) if fast else 0,
                timing_sink=lambda seconds, success, status: publish_timing(
                    getattr(self, '_pipeline_timing_pub', None), 'search_placements',
                    seconds, header, name, success, status))
        self._last_packing = (candidates, diagnostics)
        attempt = {'mode': mode, 'search_stage': 'refined' if refine else 'fast' if fast else 'exhaustive',
                   'yaw_step_deg': yaw_step, 'packing_seconds': 0.0 if reused_packing else diagnostics.elapsed_seconds,
                   'packing_reused': reused_packing,
                   'candidate_count': len(candidates), 'validation_seconds': 0.0}
        if not candidates:
            attempt['failure'] = 'NO_STABLE_PACKING_CANDIDATE'
            self._finish_placement_timing(name, attempt, calculation_started)
            return None, attempt
        request = PlanPlacement.Request()
        request.source_object_center = self._point(source.mean(axis=0))
        request.object_name, request.destination_name = name, 'tray'
        request.orientation_policy = 'upright'
        request.use_layout_scene, request.support_mode = True, mode
        request.plan_pick_place_motion = bool(self._value('plan_pick_place_motion'))
        request.source_object_cloud = self._cloud(header, source)
        request.placement_scene = self._scene_cloud(header, scene, flags)
        request.destination_support_cloud = self._cloud(header, top)
        for index, candidate in enumerate(candidates):
            msg = PlacementCandidate(id=index+1, score=float(candidate.score),
                                     orientation_policy='upright')
            msg.object_pose.header = header
            msg.object_pose.pose.position = self._point(candidate.translation)
            q = PlacementNode._quaternion_from_rotation(candidate.rotation)
            msg.object_pose.pose.orientation.x, msg.object_pose.pose.orientation.y = map(float, q[:2])
            msg.object_pose.pose.orientation.z, msg.object_pose.pose.orientation.w = map(float, q[2:])
            request.candidates.append(msg)
        started = time.perf_counter()
        attempt['grasp_attempts'] = []
        request.grasp_candidate = grasps[0]
        request.alternative_grasps = grasps[1:]
        request.validation_batch_size = max(1, int(self._value(
            'placement_validation_batch_size'))) if bool(self._value('optimize_pipeline')) else 0
        request.retry_motion_failures = bool(self._value('optimize_pipeline'))
        request.blocked_lift_grasp_ids = sorted(getattr(self, '_unliftable_grasp_ids', set()))
        call_number = 0

        def call_backend(query):
            nonlocal call_number
            prefix = f'{name}_{file_mode}_{call_number}'
            call_number += 1
            (self._directory / f'{prefix}_request.cdr').write_bytes(serialize_message(query))
            response = self._call('place', query)
            (self._directory / f'{prefix}_response.cdr').write_bytes(serialize_message(response))
            return response

        getattr(self, '_check_cancel', lambda: None)()
        result = call_backend(request)
        getattr(self, '_check_cancel', lambda: None)()
        self._unliftable_grasp_ids = set(result.blocked_lift_grasp_ids)
        attempt['validation_seconds'] = time.perf_counter() - started
        attempt['validated_candidate_count'] = result.validated_candidate_count
        attempt['validation_calls'] = json.loads(result.validation_calls_json)
        if request.plan_pick_place_motion:
            attempt.update(motion_success=result.motion_success, motion_status=result.motion_status,
                           lift_seconds=result.lift_seconds, transfer_seconds=result.transfer_seconds,
                           pose_validation_seconds=result.pose_validation_seconds,
                           descent_search_seconds=result.descent_search_seconds,
                           transfer_collision_samples=result.transfer_collision_samples,
                           lift_waypoints=len(result.lift_path_flat) // 6,
                           transfer_waypoints=len(result.transfer_path_flat) // 6,
                           descent_waypoints=result.descent_waypoint_count)
            self.get_logger().info(
                f'[BOX MOTION] {name}: lift={result.lift_seconds:.3f}s '
                f'transfer={result.transfer_seconds:.3f}s '
                f'descent_search={result.descent_search_seconds:.3f}s '
                f'status={result.motion_status}')
        attempt['grasp_attempts'].append({'success': result.success, 'status': result.status,
                                         'batch_grasp_count': len(grasps)})
        if result.success:
            grasp = result.selected_grasp_candidate
            delta = attached_object_delta(self._pose_matrix(grasp.grasp_pose.pose),
                                         self._pose_matrix(result.placement_ee_pose.pose))
            self._finish_placement_timing(name, attempt, calculation_started)
            return (result, delta, candidates[result.selected_candidate.id-1]), attempt
        attempt['failure'] = result.status
        self._finish_placement_timing(name, attempt, calculation_started)
        return None, attempt

    def _release_step_contexts(self):
        """Release the previous run's retained results before creating new contexts."""
        contexts = getattr(self, '_step_contexts', [])
        while contexts:
            response = self._call('clear_context', ClearStepContext.Request(context_id=contexts[0]))
            if not response.success and not response.status.startswith('UNKNOWN_CONTEXT'):
                raise RuntimeError('BENCHMARK_CONTEXT_CLEANUP_FAILED: ' + response.status)
            contexts.pop(0)

    def _run(self):
        self._started = time.perf_counter()
        self._directory = Path(str(self._value('output_directory'))).expanduser()
        self._report = {'calculation_only': True,
                        'api_version': 1, 'input_mode': 'synthetic_known_cloud',
                        'bypassed_functions': ['object_grounding', 'object_segmentation', 'hggd_detection', 'current_to_pregrasp_motion', 'wrist_refinement',
                                               'physical_execution'],
                        'objects': [], 'accepted': 0,
                        'edge_count': 0, 'stack_count': 0, 'failed': 0,
                        'configuration': {name: self._value(name) for name in (
                            'box_count', 'box_size_m', 'box_gap_m', 'source_radius_m',
                            'candidate_count', 'grid_resolution_m', 'yaw_step_deg',
                            'max_pile_height_m', 'placement_policy', 'grasp_approach_tilts_deg',
                            'floor_refinement_candidate_count', 'floor_refinement_yaw_step_deg',
                            'plan_pick_place_motion', 'optimize_pipeline', 'incremental_grasp_fallback',
                            'grasp_early_stop_count',
                            'placement_early_stop_count', 'placement_validation_batch_size')}}
        try:
            BoxBenchmark._release_step_contexts(self)
            self._step_context = 'benchmark_' + uuid.uuid4().hex
            self._step_ids = {}
            self._step_contexts = [self._step_context]
            self._report['step_contexts'] = self._step_contexts
            self._directory.mkdir(parents=True, exist_ok=True)
            gate = self._call('gate', GetParameters.Request(names=['allow_physical_grasp_execution']))
            if not gate.values or gate.values[0].type != 1 or gate.values[0].bool_value:
                raise RuntimeError('PHYSICAL_EXECUTION_MUST_BE_DISABLED')
            policy = str(self._value('placement_policy'))
            if policy not in ('edge', 'stack', 'edge_then_stack'):
                raise RuntimeError('INVALID_PLACEMENT_POLICY')
            tilts = list(self._value('grasp_approach_tilts_deg'))
            if (not tilts or len(tilts) > 19 or
                    any(not np.isfinite(tilt) or abs(tilt) > 60 for tilt in tilts)):
                raise RuntimeError('INVALID_GRASP_APPROACH_TILTS')
            cloud, image = self._capture()
            header = cloud.header
            self._run_observation_header = header
            PlacementNode._publish_layout_markers(self, header, [])
            xyz = point_cloud2.read_points_numpy(cloud, field_names=['x', 'y', 'z'],
                                                skip_nans=False).reshape(cloud.height, cloud.width, 3)
            original = xyz[np.isfinite(xyz).all(axis=2)]
            boxes = box_fixture(int(self._value('box_count')),
                float(self._value('box_size_m')), float(self._value('box_gap_m')),
                float(self._value('source_radius_m')), occupied_points=original)
            # A known floor patch under the fixtures prevents below-table grasps.
            floor = []
            for box in boxes:
                xs = np.arange(box[:, 0].min()-0.01, box[:, 0].max()+0.011, 0.005)
                ys = np.arange(box[:, 1].min()-0.01, box[:, 1].max()+0.011, 0.005)
                xx, yy = np.meshgrid(xs, ys)
                floor.append(np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size))))
            original, original_keep = replace_fixture_floor_noise(original, boxes, return_mask=True)
            original = np.vstack((original, *floor)).astype(np.float32)
            initial = np.vstack((original, *boxes))
            self._fused_pub.publish(self._cloud(header, initial))
            self._scene_pub.publish(self._cloud(header, np.vstack(boxes)))
            self._boxes_pub.publish(self._cloud(header, np.vstack(boxes)))
            PlacementNode._publish_layout_markers(self, header, [])
            np.savez_compressed(self._directory / 'fixture.npz',
                                original=original, boxes=np.asarray(boxes))
            self._status('GROUNDING_TRAY', box_count=len(boxes), policy=policy,
                         cloud_surfaces='complete six-face synthetic boxes')
            rgb = self._bridge.imgmsg_to_cv2(image, 'bgr8')
            path = self._directory / 'source_image.png'
            if not cv2.imwrite(str(path), rgb):
                raise RuntimeError('IMAGE_SAVE_FAILED')
            started = time.perf_counter()
            presence = self._call('presence', VerifyPresence.Request(
                target=str(self._value('destination'))), [StepBinding(
                    target_field='image', source_step_id=self._step_ids['capture'],
                    source_field='image')])
            if not presence.success or not presence.present:
                raise RuntimeError('TRAY_NOT_CONFIRMED_PRESENT: ' + presence.status)
            grounding = self._call('ground', GroundObject.Request(
                target=str(self._value('destination')), verify_presence=False), [StepBinding(
                    target_field='image', source_step_id=self._step_ids['capture'],
                    source_field='image')])
            if not grounding.success:
                raise RuntimeError('TRAY_GROUNDING_FAILED: ' + grounding.status)
            reply = grounding.text
            bbox = parse_grounding_box(reply, image.width, image.height)
            segmentation = self._call('segment', SegmentGroundedImage.Request(
                bbox=list(bbox)), [StepBinding(
                    target_field='image', source_step_id=self._step_ids['capture'],
                    source_field='image')])
            if not segmentation.success:
                raise RuntimeError('TRAY_SEGMENTATION_FAILED: ' + segmentation.status)
            if segmentation.mask.header != image.header:
                raise RuntimeError('TRAY_SEGMENTATION_OBSERVATION_MISMATCH')
            silhouette = self._bridge.imgmsg_to_cv2(segmentation.mask, 'mono8') > 0
            tray, foreground, crop = extract_segmented_destination_cloud(xyz, silhouette, 80)
            cv2.imwrite(str(self._directory / 'tray_mask.png'), silhouette.astype(np.uint8)*255)
            overlay = rgb.copy()
            overlay[silhouette] = (0.55*rgb[silhouette] + 0.45*np.array([0, 255, 0])).astype(np.uint8)
            cv2.imwrite(str(self._directory / 'tray_segmentation.png'), overlay)
            floor_mask = np.zeros(xyz.shape[:2], dtype=bool)
            x1, y1, x2, y2 = crop
            floor_mask[y1:y2, x1:x2] = foreground
            floor_mask[floor_mask] = tray_floor_mask(tray)
            measured_flags = floor_mask[np.isfinite(xyz).all(axis=2)].astype(np.uint8)
            measured_flags = measured_flags[original_keep]
            original_flags = np.r_[measured_flags, np.zeros(sum(len(p) for p in floor), np.uint8)]
            tray_floor = xyz[floor_mask]
            self._tray_pub.publish(self._cloud(header, tray))
            np.savez_compressed(self._directory / 'destination.npz', tray=tray, bbox=bbox, mask=silhouette)
            self._status('TRAY_READY', grounding_seconds=time.perf_counter()-started,
                         grounding_reply=reply)
            placed, tops, markers = [], [], []
            placed_sources, placed_names, failed_boxes = [], [], []
            for index, source in enumerate(boxes):
                self._step_context = 'box_' + uuid.uuid4().hex
                self._step_contexts.append(self._step_context)
                self._check_cancel()
                box_started = time.perf_counter()
                name = f'box_{index+1:02d}'
                self._unliftable_grasp_ids = set()
                row = {'object': name, 'attempts': []}
                self._report['objects'].append(row)
                grasp_started = None
                committed_count = len(placed)
                try:
                    scene = np.vstack((original, *failed_boxes, *boxes[index+1:], *placed))
                    flags = np.r_[original_flags, np.zeros(len(scene)-len(original), np.uint8)]
                    self._status('FINDING_GRASP', current_object=name)
                    started = time.perf_counter()
                    grasp_started = started
                    grasp_candidates = []
                    grasp_failures = []
                    # Separate wall-axis queries prevent the quick validation pool
                    # from filling with variants of just one closing direction.
                    seed_groups = []
                    query_tilts = sorted(tilts, key=lambda angle: (angle < 0, -abs(angle))) if bool(
                        self._value('optimize_pipeline')) else tilts
                    for tilt in query_tilts:
                        seeds = box_contact_seeds(source, tilt)
                        seed_groups.extend((seeds[:2], seeds[2:]))
                    pending_groups = iter(seed_groups)
                    def collect_grasps(limit):
                        collection_started = time.perf_counter()
                        for wall_seeds in pending_groups:
                            row['grasp_service_calls'] = row.get('grasp_service_calls', 0) + 1
                            grasp = self._call('grasp', FindCloudGrasps.Request(
                                observation_header=header, object_name=name,
                                contact_left=[self._point(l) for l, _, _ in wall_seeds],
                                contact_right=[self._point(r) for _, r, _ in wall_seeds],
                                approaches=[Vector3(x=float(a[0]), y=float(a[1]), z=float(a[2]))
                                            for _, _, a in wall_seeds]), [
                                StepBinding(target_field='object_cloud',
                                            source_step_id=self._scene_step,
                                            source_field='object_cloud'),
                                StepBinding(target_field='scene_cloud',
                                            source_step_id=self._scene_step,
                                            source_field='scene_cloud')])
                            if grasp.success:
                                for candidate in grasp.candidates[:2]:
                                    if not any(np.allclose(self._pose_matrix(candidate.grasp_pose.pose),
                                                           self._pose_matrix(other.grasp_pose.pose),
                                                           rtol=0, atol=1e-6) and
                                               np.allclose(candidate.grasp_joints, other.grasp_joints,
                                                           rtol=0, atol=1e-4)
                                               for other in grasp_candidates):
                                        grasp_candidates.append(candidate)
                            else:
                                grasp_failures.append(grasp.status)
                            if limit > 0 and len(grasp_candidates) >= limit:
                                break
                        for number, candidate in enumerate(grasp_candidates, 1):
                            candidate.id = number
                        row['grasp_seconds'] = row.get('grasp_seconds', 0) + time.perf_counter()-collection_started
                        row['grasp_count'] = len(grasp_candidates)
                    object_message = self._cloud(header, source)
                    scene_message = self._cloud(header, scene)
                    prepared = self._call('scene', PrepareCloudScene.Request(
                        observation_header=header, object_cloud=object_message,
                        scene_cloud=scene_message))
                    if not prepared.success:
                        raise RuntimeError(prepared.status)
                    self._scene_step = self._step_ids['scene']
                    self._expand_grasps = lambda limit=0: collect_grasps(limit)
                    collect_grasps(int(self._value('grasp_early_stop_count')) if bool(
                        self._value('optimize_pipeline')) else 0)
                    row['grasp_seconds'] = time.perf_counter()-started
                    row['grasp_count'] = len(grasp_candidates)
                    self.get_logger().info(
                        f'[BOX TIMING] {name} GRASP: {row["grasp_seconds"]:.2f}s, '
                        f'valid_candidates={len(grasp_candidates)}')
                    if not grasp_candidates:
                        raise RuntimeError(f'INVALID_GRASP: {name}: {grasp_failures}')
                    accepted = None
                    if policy != 'stack' or not placed:
                        self._status('EDGE_SEARCH')
                        accepted, attempt = self._plan(source, grasp_candidates, scene, flags,
                            tray, tray_floor, 'edge', header, name)
                        row['attempts'].append(attempt)
                        if accepted is None and int(self._value('floor_refinement_candidate_count')) > 0:
                            self._status('FLOOR_REFINEMENT')
                            accepted, attempt = self._plan(source, grasp_candidates, scene, flags,
                                tray, tray_floor, 'edge', header, name, refine=True)
                            row['attempts'].append(attempt)
                    mode, support_name = 'EDGE', 'tray'
                    if accepted is None and policy != 'edge':
                        self._status('STACK_SEARCH', fallback_reason=row['attempts'][-1].get('failure')
                                     if row['attempts'] else 'STACK_ONLY')
                        mode = 'STACK'
                        # All exposed tops contribute; no per-object support selection.
                        selected_flags = flags.copy()
                        offset = len(original) + sum(len(box) for box in (*failed_boxes, *boxes[index+1:]))
                        for points, source_points in zip(placed, placed_sources):
                            top_mask = source_points[:, 2] >= source_points[:, 2].max() - 0.0001
                            selected_flags[offset:offset+len(points)][top_mask] = 2
                            offset += len(points)
                        support_top = np.vstack(tops) if tops else tray_floor[:0]
                        accepted, attempt = self._plan(source, grasp_candidates, scene,
                            selected_flags, tray, support_top, 'stack', header, name)
                        row['attempts'].append(attempt)
                        support_name = 'heightmap'
                    if accepted is None:
                        raise RuntimeError(f'NO_VALID_PLACEMENT: {name}: {row["attempts"]}')
                    result, delta, candidate = accepted
                    self._check_cancel()
                    points = transform_points(source, delta)
                    placed.append(points)
                    placed_sources.append(source)
                    placed_names.append(name)
                    tops.append(transform_points(box_top(source), delta))
                    # Markers and future heightmaps use actual FK, preserving attachment.
                    candidate = copy.copy(candidate)
                    candidate.rotation = delta[:3, :3]
                    candidate.translation = delta[:3, :3] @ source.mean(axis=0) + delta[:3, 3]
                    markers.append((f'{name} [{mode} on {support_name}]', points, candidate, result))
                    PlacementNode._publish_layout_markers(self, header, markers)
                    self._scene_pub.publish(self._cloud(header,
                        np.vstack((*failed_boxes, *boxes[index+1:], *placed))))
                    self._fused_pub.publish(self._cloud(header,
                        np.vstack((original, *failed_boxes, *boxes[index+1:], *placed))))
                    self._boxes_pub.publish(self._cloud(header,
                        np.vstack((*failed_boxes, *boxes[index+1:], *placed))))
                    row.update(mode=mode, support_object=support_name,
                               selected_candidate=result.selected_candidate.id,
                               placement_position_error_m=result.position_error_m,
                               descent_waypoint_count=result.descent_waypoint_count)
                    self._report['accepted'] += 1
                    self._report['edge_count' if mode == 'EDGE' else 'stack_count'] += 1
                    np.savez_compressed(self._directory / f'{name}.npz',
                                        source=source, placed=points, delta=delta,
                                        descent_joints=np.asarray(result.descent_path_flat).reshape(-1, 6),
                                        lift_joints=np.asarray(result.lift_path_flat).reshape(-1, 6),
                                        transfer_joints=np.asarray(result.transfer_path_flat).reshape(-1, 6))
                    self._log_box_timing(row, time.perf_counter() - box_started)
                    self._status('OBJECT_ACCEPTED')
                except Exception as exception:
                    if self._cancel.is_set() or len(placed) != committed_count:
                        raise
                    row.update(mode='FAILED', failure=str(exception))
                    if 'grasp_seconds' not in row:
                        row['grasp_seconds'] = (
                            time.perf_counter() - grasp_started if grasp_started else 0.0)
                    self._log_box_timing(row, time.perf_counter() - box_started)
                    failed_boxes.append(source)
                    self._report['failed'] += 1
                    self.get_logger().warn(f'[BOX SKIPPED] {name}: {exception}')
                    visible = np.vstack((*failed_boxes, *boxes[index+1:], *placed))
                    self._boxes_pub.publish(self._cloud(header, visible))
                    self._scene_pub.publish(self._cloud(header, visible))
                    self._fused_pub.publish(self._cloud(header, np.vstack((original, visible))))
                    self._status('OBJECT_FAILED', current_object=name)
            self._status('COMPLETE_WITH_FAILURES' if self._report['failed'] else 'COMPLETE')
        except Exception as exception:
            self._status('CANCELLED' if self._cancel.is_set() else 'FAILED',
                         failure=str(exception))
        finally:
            self._print_final_summary()

    def shutdown_worker(self):
        self._cancel.set()
        if self._worker is not None:
            self._worker.join(timeout=2)


def main(args=None):
    rclpy.init(args=args)
    node = BoxBenchmark()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_worker()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
