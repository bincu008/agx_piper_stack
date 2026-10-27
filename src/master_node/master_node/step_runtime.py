"""Host the common API microstep queue and typed result memory in the master."""

import copy
import threading
import time
import uuid

from my_actions.action import RunPipelineSteps
from my_actions.msg import ApiCallTrace, PipelineTiming
from my_actions.srv import (
    ClearStepContext, CommitFloorSlots, GetStepContext, PrepareCloudScene, RunPipelineStep,
    StorePipelineResult, SubmitPipelineSteps, VerifyPresence,
)

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.serialization import deserialize_message, serialize_message

from task_pipeline.context import StepStore
from task_pipeline.grasp_input import calculate_grasps_with_recovery
from task_pipeline.preparation_recovery import pin_observation_with_recovery
from task_pipeline.node import response_status, response_success, set_failure
from task_pipeline.placement import validate_placements
from task_pipeline.registry import FUNCTIONS


class MasterStepRuntime:
    """Keep orchestration memory beside arm state, not in each API proxy."""

    def __init__(self, node):
        """Register bounded services and an action for queued completion."""
        self.node = node
        self.group = ReentrantCallbackGroup()
        self._active = threading.BoundedSemaphore(8)
        self._clients = {}
        self._goals = set()
        self._goal_lock = threading.Lock()
        node.declare_parameter('pipeline_step_timeout_sec', 600.0)
        self.store = StepStore(self._call)
        self._timings = node.create_publisher(PipelineTiming, '/pipeline/timings', 100)
        self._traces = node.create_publisher(ApiCallTrace, '/pipeline/api_traces', 100)
        self._trace_local = threading.local()
        for service_type, name, callback in (
                (RunPipelineStep, 'run_step', self._run_step),
                (SubmitPipelineSteps, 'submit_steps', self._submit),
                (GetStepContext, 'get_step_context', self._get),
                (ClearStepContext, 'clear_step_context', self._clear),
                (StorePipelineResult, 'store_step_result', self._import)):
            node.create_service(service_type, '/master/' + name, callback,
                                callback_group=self.group)
        node.create_service(PrepareCloudScene, '/master/prepare_cloud_scene',
                            self._prepare_cloud_scene, callback_group=self.group)
        self._action = ActionServer(
            node, RunPipelineSteps, '/pipeline/run_steps', self._execute,
            goal_callback=self._goal, cancel_callback=lambda _: CancelResponse.ACCEPT,
            handle_accepted_callback=self._accepted, callback_group=self.group)

    def _native(self, name, request):
        if name == 'prepare_cloud_scene':
            return self._prepare_cloud_scene(request, PrepareCloudScene.Response())
        spec = FUNCTIONS[name]
        if name not in self._clients:
            self._clients[name] = self.node.create_client(
                spec.service_type, spec.backend, callback_group=self.group)
        client = self._clients[name]
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('BACKEND_UNAVAILABLE: ' + spec.backend)
        future = client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        deadline = time.monotonic() + self.node.get_parameter('pipeline_step_timeout_sec').value
        missing_since = None
        while not done.wait(.02):
            if not rclpy.ok():
                raise RuntimeError('MASTER_SHUTDOWN')
            now = time.monotonic()
            if client.service_is_ready():
                missing_since = None
            elif missing_since is None:
                missing_since = now
            elif now - missing_since >= 2.0 and not done.is_set():
                # Cancel only this local wait, never retry a possibly completed
                # operation or infer that the remote operation did not run.
                future.cancel()
                raise RuntimeError('BACKEND_LOST_STATE_UNKNOWN: ' + spec.backend)
            if now >= deadline:
                future.cancel()
                raise RuntimeError('BACKEND_TIMEOUT_STATE_UNKNOWN: ' + spec.backend)
        if future.exception():
            raise RuntimeError(str(future.exception()))
        return future.result()

    @staticmethod
    def _prepare_cloud_scene(request, response):
        """Validate metadata and register source geometry without altering it."""
        header = request.observation_header
        if header.frame_id != 'base_link':
            response.status = 'SCENE_MANUAL_FRAME_MISMATCH'
            return response
        for cloud in (request.object_cloud, request.scene_cloud):
            if cloud.header != header:
                response.status = 'SCENE_OBSERVATION_MISMATCH'
                return response
            if (not {'x', 'y', 'z'} <= {field.name for field in cloud.fields} or
                    cloud.point_step <= 0 or cloud.height <= 0 or
                    cloud.row_step < cloud.width * cloud.point_step or
                    len(cloud.data) != cloud.row_step * cloud.height):
                response.status = 'SCENE_CLOUD_MALFORMED'
                return response
        response.success = True
        response.status = 'IMMUTABLE_CLOUD_SCENE_READY'
        response.observation_header = header
        response.object_cloud = request.object_cloud
        response.scene_cloud = request.scene_cloud
        return response

    def _call(self, name, request):
        trace = getattr(self._trace_local, 'current', None)
        if trace is not None and not trace.backend_started_ns:
            trace.backend_started_ns = time.monotonic_ns()
        started = time.monotonic()
        failure = None
        try:
            if name == 'ground_object' and request.verify_presence:
                query = VerifyPresence.Request(image=request.image, target=request.target)
                _, _, item = self.store.run_one(
                    '', uuid.uuid4().hex, 'verify_presence', serialize_message(query))
                presence = deserialize_message(bytes(item.response_cdr), VerifyPresence.Response)
                if not presence.success or not presence.present:
                    raise RuntimeError('NOT_FOUND_OR_PRESENCE_FAILED: ' + presence.status)
                query = copy.deepcopy(request)
                query.verify_presence = False
                response = self._native(name, query)
            elif name == 'calculate_grasps':
                response = calculate_grasps_with_recovery(
                    request, lambda query: self._native(name, query))
            elif name == 'pin_placement_observation':
                response = pin_observation_with_recovery(
                    request, lambda query: self._native(name, query))
            elif name in ('plan_placement', 'validate_placements') and (
                    request.validation_batch_size or request.retry_motion_failures):
                check_selected = None
                if request.support_mode == 'floor_slots':
                    def check_selected(result):
                        return self._call('commit_floor_slots', CommitFloorSlots.Request(
                            selected_pose=result.selected_candidate.object_pose, validate_only=True))
                response = validate_placements(request, lambda query: self._native(name, query),
                                               validate_selected=check_selected)
            else:
                response = self._native(name, request)
        except Exception as error:
            failure = str(error)
            response = set_failure(FUNCTIONS[name].service_type.Response(), str(error))
        memory = getattr(self.node, 'manipulation_memory', None)
        if memory is not None and failure is None:
            try:
                memory.observe_result(name, request, response)
            except Exception as error:
                self.node.get_logger().warning(f'[ACTIVE MEMORY] update requires refresh: {error}')
        if trace is not None:
            trace.backend_finished_ns = time.monotonic_ns()
            if trace.function_name == name:
                self._trace_local.public_call = (request, response)
        header = getattr(request, 'observation_header', None)
        if header is None and hasattr(request, 'image'):
            header = request.image.header
        if header is None and getattr(request, 'use_layout_scene', False):
            header = request.source_object_cloud.header
        if name == 'capture_observation' and response.success:
            header = response.cloud.header
        if header is None:
            header = getattr(request, 'target_observation_header', None) or getattr(
                request, 'previous_observation_header', None)
        mode = FUNCTIONS[name].input_mode
        if name == 'ground_object':
            mode = 'grounding_with_presence' if request.verify_presence else 'grounding_only'
        if name in ('plan_placement', 'validate_placements'):
            mode = 'frozen_scene' if request.use_layout_scene else 'legacy_pose_only'
        kwargs = {} if header is None else {'observation_header': header}
        self._timings.publish(PipelineTiming(
            function_name=name, backend=FUNCTIONS[name].backend,
            input_mode=mode, scope='backend_inclusive',
            object_name=(getattr(request, 'object_name', '') or
                         getattr(request, 'target_name', '') or
                         (request.target if isinstance(getattr(request, 'target', None), str)
                          else '')),
            full_motion=bool(getattr(request, 'plan_pick_place_motion', False)),
            wall_seconds=time.monotonic() - started,
            success=failure is None and response_success(response),
            status=failure if failure is not None else response_status(response), **kwargs))
        if failure is not None and not any(hasattr(response, field) for field in (
                'success', 'valid', 'state_json')):
            # Older response schemas without a status flag cannot carry failure.
            # The structured run_step result still must reject their empty payload.
            raise RuntimeError(failure)
        return response

    def _run_step(self, request, response, on_complete=None):
        trace = ApiCallTrace(
            operation_id=request.step_id, function_name=request.function_name,
            component='master', received_ns=time.monotonic_ns())
        self._trace_local.current = trace
        admitted = self._active.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('MASTER_STEP_BUSY')
            response.context_id, response.revision, response.result = self.store.run_one(
                request.context_id, request.step_id, request.function_name,
                request.request_cdr, request.expected_revision, request.bindings)
            response.success = response.result.success
            response.status = response.result.status
        except Exception as error:
            response.success = False
            response.status = str(error)
        finally:
            if admitted:
                self._active.release()
            public_call = getattr(self._trace_local, 'public_call', None)
            if hasattr(self._trace_local, 'public_call'):
                del self._trace_local.public_call
            if on_complete is not None and public_call is not None:
                on_complete(*public_call, (response.success, response.status))
            trace.finished_ns = time.monotonic_ns()
            trace.success = response.success
            self._trace_local.current = None
            self._traces.publish(trace)
        return response

    def run_typed(self, name, request):
        """Execute a colocated API call through the same bounded result store."""
        admitted = self._active.acquire(blocking=False)
        if not admitted:
            raise RuntimeError('MASTER_STEP_BUSY')
        try:
            _, _, item, response = self.store.run_typed('', uuid.uuid4().hex, name, request)
            return response, (item.success, item.status)
        finally:
            self._active.release()

    def _submit(self, request, response):
        try:
            response.context_id, response.revision = self.store.submit(
                request.context_id, request.steps, request.expected_revision)
            response.success = True
            response.status = 'MICROSTEPS_QUEUED'
        except Exception as error:
            response.status = str(error)
        return response

    def _import(self, request, response):
        try:
            response.context_id, response.revision = self.store.store(
                request.context_id, request.step_id, request.function_name,
                request.response_cdr, request.expected_revision)
            response.success = True
            response.status = 'TYPED_RESULT_STORED'
        except Exception as error:
            response.status = str(error)
        return response

    def _get(self, request, response):
        try:
            response.state_json, response.results = self.store.snapshot(
                request.context_id, request.include_results)
            response.success = True
            response.status = 'CONTEXT_READY'
        except Exception as error:
            response.status = str(error)
        return response

    def _clear(self, request, response):
        try:
            self.store.clear(request.context_id, request.expected_revision)
            response.success = True
            response.status = 'CONTEXT_RELEASED'
        except Exception as error:
            response.status = str(error)
        return response

    def _goal(self, goal):
        with self._goal_lock:
            if len(self._goals) >= 4 or goal.context_id in self._goals:
                return GoalResponse.REJECT
            try:
                self.store.snapshot(goal.context_id)
            except ValueError:
                return GoalResponse.REJECT
            self._goals.add(goal.context_id)
            return GoalResponse.ACCEPT

    def _accepted(self, handle):
        handle.execute()

    def _execute(self, handle):
        started = time.monotonic()
        result = RunPipelineSteps.Result(context_id=handle.request.context_id)
        attempted = {}

        def progress(step, status, completed, remaining):
            memory = getattr(self.node, 'manipulation_memory', None)
            if memory is not None:
                state, _ = self.store.snapshot(handle.request.context_id)
                import json
                pending = [s['step_id']+':'+s['function'] for s in json.loads(state)[0]['pending']]
                memory.print_state(f'{step.step_id}:{step.function_name}:{status} pending={pending}')
            if status == 'RUNNING' and 'object_name' in (
                    FUNCTIONS[step.function_name].service_type.Request.get_fields_and_field_types()):
                query = deserialize_message(bytes(step.request_cdr),
                                            FUNCTIONS[step.function_name].service_type.Request)
                attempted['object'] = getattr(query, 'object_name', '')
                attempted['function'] = step.function_name
            handle.publish_feedback(RunPipelineSteps.Feedback(
                context_id=handle.request.context_id, step_id=step.step_id,
                function_name=step.function_name, status=status,
                completed_steps=completed, remaining_steps=remaining))

        try:
            result.status, result.results = self.store.run_queue(
                handle.request.context_id, lambda: handle.is_cancel_requested, progress,
                stop_before_step_id=handle.request.stop_before_step_id)
            result.success = result.status in ('COMPLETE', 'PAUSED_AT_STEP_BOUNDARY')
            if result.status == 'STOPPED_ON_FAILURE':
                relevant = {'execute_plan', 'gripper_open', 'gripper_close',
                            'execute_prepared_grasp_lift', 'cache_grasp_plan',
                            'prepare_grasp_lift', 'get_prepared_grasp_lift',
                            'validate_prepared_grasp_lift'}
                state, _ = self.store.snapshot(handle.request.context_id)
                import json
                prepared_waiting = any(step['function'] in relevant
                                       for step in json.loads(state)[0]['pending'])
                failed = next((item for item in reversed(result.results) if not item.success), None)
                if prepared_waiting or (failed and failed.function_name in relevant):
                    self.store.flush_failed_queue(handle.request.context_id)
                    if (failed and attempted.get('function') == failed.function_name and
                            attempted.get('object')):
                        recovery = self.node.execution_api.prepared_grasp_lift.recover_after_failure(
                            attempted['object'])
                        if recovery:
                            result.status += ';'+recovery

            if handle.is_cancel_requested:
                result.success = False
                result.status = 'CANCELLED_AT_STEP_BOUNDARY'
                handle.canceled()
            elif result.success:
                handle.succeed()
            else:
                handle.abort()
        except Exception as error:
            result.status = str(error)
            handle.abort()
        finally:
            result.wall_seconds = time.monotonic() - started
            with self._goal_lock:
                self._goals.discard(handle.request.context_id)
        return result
