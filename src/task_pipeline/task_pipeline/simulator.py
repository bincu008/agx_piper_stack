"""One mock-only pipeline: append, display, execute all and flush."""

import copy
import json
import math
import os
import threading
import time
import uuid

from my_actions import srv
from my_actions.action import RunPipelineSteps

import numpy as np

from rcl_interfaces.srv import GetParameters

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import deserialize_message

from std_srvs.srv import Trigger

from visualization_msgs.msg import Marker, MarkerArray

from .recipes import prepare_pick_place, step
from .registry import catalog, FUNCTIONS


CALCULATIONS = frozenset({
    'capture_observation', 'pin_grasp_observation', 'pin_placement_observation',
    'verify_presence', 'ground_object', 'segment_object', 'store_segmented_target',
    'prepare_grasp_input', 'calculate_grasps', 'update_object', 'select_grasp',
    'cache_grasp_plan', 'find_placements', 'plan_next_best_view',
    'prepare_pick_place', 'validate_prepared_pick_place', 'get_execution_state',
})
RAW_TOPICS = ('/camera/color/image_raw', '/camera/depth_registered/points',
              '/feedback/joint_states', '/feedback/tcp_pose')


def simulator_steps(object_name, destination, lift_distance=.05):
    """Reuse the production recipe, adding memory and a final no-motion check."""
    recipe = copy.deepcopy(prepare_pick_place(object_name, destination, lift_distance))
    names = {'presence_cup': 'presence_object', 'ground_cup': 'ground_object',
             'segment_cup': 'segment_object', 'presence_tray': 'presence_destination',
             'ground_tray': 'ground_destination', 'segment_tray': 'segment_destination',
             'placements': 'find_slots', 'prepare_motion': 'prepare_pick_place'}
    cache = next(item for item in recipe if item.function_name == 'cache_grasp_plan')
    recipe.remove(cache)
    recipe.insert(next(i for i, item in enumerate(recipe)
                       if item.function_name == 'find_placements'), cache)
    result = []
    for item in recipe:
        item.step_id = names.get(item.step_id, item.step_id)
        for binding in item.bindings:
            binding.source_step_id = names.get(binding.source_step_id, binding.source_step_id)
        if item.function_name == 'get_execution_state':
            result.append(step('check_pick_place', 'validate_prepared_pick_place',
                               srv.PreparePickPlace.Request(object_name=object_name)))
        result.append(item)
        if item.step_id in ('segment_object', 'segment_destination'):
            target = item.step_id == 'segment_destination'
            result.append(step('remember_destination' if target else 'remember_object',
                               'store_segmented_target', srv.StoreSegmentedTarget.Request(
                                   name=destination if target else object_name,
                                   role='destination' if target else 'object'), [
                                   ('image', 'observe', 'image'), ('mask', item.step_id, 'mask'),
                                   ('observation_cloud', 'observe', 'cloud'),
                                   ('camera_transform', 'observe', 'camera_transform')]))
    validate_steps(result)
    return result


def validate_steps(steps):
    """Refuse any recipe that could dispatch physical operations or commit slots."""
    physical = {item['name'] for item in catalog() if item.get('effect') == 'actuation'}
    seen = set()
    for item in steps:
        if item.function_name not in CALCULATIONS or item.function_name in physical:
            raise ValueError('SIMULATOR_FORBIDS_FUNCTION: ' + item.function_name)
        if item.step_id in seen or any(b.source_step_id not in seen for b in item.bindings):
            raise ValueError('SIMULATOR_INVALID_STEP_DEPENDENCY: ' + item.step_id)
        if item.failure_policy not in ('', 'stop'):
            raise ValueError('SIMULATOR_REQUIRES_STOP_ON_FAILURE')
        seen.add(item.step_id)


def mock_sources(node):
    """Require exclusively the replay publisher, independently of domain number."""
    observed = {}
    for topic in RAW_TOPICS:
        observed[topic] = [p.node_namespace.rstrip('/') + '/' + p.node_name
                           for p in node.get_publishers_info_by_topic(topic)]
    return all(names == ['/mock_hardware'] for names in observed.values()), observed


def simulator_domain(value):
    """Keep the student replay separate from the usual live domain zero."""
    number = int(value)
    if not 1 <= number <= 232:
        raise ValueError('SIMULATOR_REQUIRES_DEDICATED_NONZERO_ROS_DOMAIN')
    return number


def selected_grasp_markers(candidate=None, finger_geometry=None):
    """Preview a selected calculation result without claiming a prepared motion."""
    markers = [Marker(action=Marker.DELETEALL)]
    if candidate is not None:
        for index, (name, pose, color) in enumerate((
                ('Selected grasp', candidate.grasp_pose, (0., 1., .2)),
                ('Pregrasp', candidate.pregrasp_pose, (0., .5, 1.)))):
            if pose.header.frame_id != 'base_link':
                raise ValueError('GRASP_PREVIEW_REQUIRES_BASE_LINK')
            arrow = Marker(header=pose.header, ns='simulator_grasp', id=index,
                           type=Marker.ARROW, pose=copy.deepcopy(pose.pose))
            arrow.scale.x, arrow.scale.y, arrow.scale.z = .06, .006, .009
            arrow.color.r, arrow.color.g, arrow.color.b = color
            arrow.color.a = 1.
            markers.append(arrow)
            label = Marker(header=pose.header, ns='simulator_grasp', id=index + 2,
                           type=Marker.TEXT_VIEW_FACING, pose=copy.deepcopy(pose.pose))
            label.pose.position.z += .035
            label.scale.z = .012
            label.color.r = label.color.g = label.color.b = label.color.a = 1.
            label.text = name + ' (calculation only)'
            markers.append(label)
    if candidate is not None and finger_geometry is not None:
        # Match VAMP PlanPlacement: X=closing, Y=height, Z=approach for each bar.
        width, height, length, tcp_offset = finger_geometry
        for stamped, namespace, color in (
                (candidate.grasp_pose, 'simulator_fingers', (0., 1., .2)),
                (candidate.pregrasp_pose, 'simulator_pregrasp_fingers', (0., .5, 1.))):
            pose = stamped.pose
            q = np.array([pose.orientation.x, pose.orientation.y,
                          pose.orientation.z, pose.orientation.w], dtype=float)
            q /= np.linalg.norm(q)
            for index, sign in enumerate((1., -1.)):
                local = np.array([0., sign * (candidate.jaw_opening_m + width) / 2.,
                                  length / 2. - tcp_offset])
                shifted = local + 2. * np.cross(q[:3], np.cross(q[:3], local) + q[3] * local)
                bar = Marker(header=stamped.header, ns=namespace, id=index,
                             type=Marker.CUBE, pose=copy.deepcopy(pose))
                bar.pose.position.x += float(shifted[0])
                bar.pose.position.y += float(shifted[1])
                bar.pose.position.z += float(shifted[2])
                factor = math.sqrt(.5)
                bar.pose.orientation.x = float(factor * (q[0] + q[1]))
                bar.pose.orientation.y = float(factor * (q[1] - q[0]))
                bar.pose.orientation.z = float(factor * (q[2] + q[3]))
                bar.pose.orientation.w = float(factor * (q[3] - q[2]))
                bar.scale.x, bar.scale.y, bar.scale.z = width, height, length
                bar.color.r, bar.color.g, bar.color.b = color
                bar.color.a = .9
                markers.append(bar)
    return MarkerArray(markers=markers)


class OutcomeUnknown(RuntimeError):
    """An in-flight calculation must not be replayed or cleared automatically."""


class SimulatorTransport:
    """Delegate all calculation and result binding to the existing master queue."""

    def __init__(self, node, group):
        """Create fixed control clients; never construct actuator clients."""
        self.node = node
        self.clients = {name: node.create_client(FUNCTIONS[name].service_type,
                                                 FUNCTIONS[name].endpoint,
                                                 callback_group=group)
                        for name in ('get_world_state', 'submit_steps', 'get_step_context',
                                     'clear_step_context')}
        self.action = ActionClient(node, RunPipelineSteps, '/pipeline/run_steps',
                                   callback_group=group)
        self.geometry_client = node.create_client(
            GetParameters, '/vamp_planner_server/get_parameters', callback_group=group)
        self.context = 'simulator_service_' + uuid.uuid4().hex
        self.has_context = False
        self.revision = 0

    @staticmethod
    def wait(future, timeout):
        """Wait while the multithreaded executor delivers responses."""
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        if not ready.wait(timeout):
            raise OutcomeUnknown('OUTCOME_UNKNOWN: calculation still may be running')
        try:
            return future.result()
        except Exception as error:
            raise OutcomeUnknown('OUTCOME_UNKNOWN: ' + str(error)) from error

    def request(self, name, query):
        """Call only a fixed catalogued control endpoint."""
        client = self.clients[name]
        if not client.wait_for_service(timeout_sec=5.):
            raise RuntimeError('SERVICE_UNAVAILABLE: ' + FUNCTIONS[name].endpoint)
        return self.wait(client.call_async(query), 10.)

    def guard(self):
        """Require isolated replay, idle empty hand and a disabled physical gate."""
        simulator_domain(os.environ.get('ROS_DOMAIN_ID', '0'))
        deadline = time.monotonic() + 8.
        while True:
            valid, observed = mock_sources(self.node)
            if valid:
                break
            if any(name != '/mock_hardware' and '_UNKNOWN_' not in name
                   for names in observed.values() for name in names):
                raise RuntimeError('SIMULATOR_REPLAY_SOURCES_REQUIRED: ' + json.dumps(observed))
            if time.monotonic() >= deadline:
                raise RuntimeError('SIMULATOR_REPLAY_SOURCES_REQUIRED: ' + json.dumps(observed))
            threading.Event().wait(.1)
        world = self.request('get_world_state', srv.GetWorldState.Request())
        robot = json.loads(world.state_json)['robot']
        if robot.get('physical_grasp_execution_enabled') is not False:
            raise RuntimeError('SIMULATOR_REQUIRES_PHYSICAL_EXECUTION_DISABLED')
        if robot.get('state') != 'IDLE' or robot.get('holding') is not None:
            raise RuntimeError('SIMULATOR_REQUIRES_IDLE_EMPTY_HAND')

    def execute(self, steps, feedback):
        """Wait for final queue completion; report exact master results and timings."""
        validate_steps(steps)
        self.guard()
        self.node.grasp_preview.publish(selected_grasp_markers())
        response = self.request('submit_steps', srv.SubmitPipelineSteps.Request(
            context_id=self.context, steps=steps))
        if not response.success:
            raise RuntimeError(response.status)
        self.has_context = True
        self.revision = response.revision
        if not self.action.wait_for_server(timeout_sec=5.):
            raise RuntimeError('SERVICE_UNAVAILABLE: /pipeline/run_steps')
        goal = self.wait(self.action.send_goal_async(RunPipelineSteps.Goal(
            context_id=self.context, expected_revision=self.revision),
            feedback_callback=feedback), 10.)
        if not goal.accepted:
            raise RuntimeError('SIMULATOR_QUEUE_GOAL_REJECTED')
        result = self.wait(goal.get_result_async(), 600.)
        response = self.request('get_step_context', srv.GetStepContext.Request(
            context_id=self.context, include_results=False))
        if not response.success:
            raise OutcomeUnknown(response.status)
        state = json.loads(response.state_json)[0]
        if state['busy']:
            raise OutcomeUnknown('OUTCOME_UNKNOWN: master still busy')
        self.revision = state['revision']
        if ('STATE_UNKNOWN' in result.result.status or
                any('STATE_UNKNOWN' in item['status'] for item in state['results'])):
            raise OutcomeUnknown('OUTCOME_UNKNOWN: ' + result.result.status)
        for item in result.result.results:
            if item.function_name == 'select_grasp' and item.success:
                selected = deserialize_message(bytes(item.response_cdr), srv.SelectGrasp.Response)
                if selected.success:
                    geometry = None
                    try:
                        if not self.geometry_client.wait_for_service(timeout_sec=2.):
                            raise RuntimeError('VAMP geometry parameters unavailable')
                        response = self.wait(self.geometry_client.call_async(GetParameters.Request(
                            names=['finger_width_m', 'finger_height_m', 'finger_length_m',
                                   'grasp_tcp_offset_m'])), 5.)
                        if len(response.values) != 4 or any(v.type != 3 for v in response.values):
                            raise ValueError('Invalid VAMP geometry parameter types')
                        geometry = tuple(v.double_value for v in response.values)
                        if not all(math.isfinite(v) for v in geometry) or min(geometry[:3]) <= 0:
                            raise ValueError('Invalid VAMP finger geometry')
                    except Exception as error:
                        self.node.get_logger().warning('Finger preview unavailable: ' + str(error))
                    self.node.grasp_preview.publish(
                        selected_grasp_markers(selected.candidate, geometry))
        return result.result.success, {'status': result.result.status, 'results': state['results']}

    def clear(self):
        """Clear only this simulator's idle calculation context, never an execution ledger."""
        if not self.has_context:
            return
        response = self.request('get_step_context', srv.GetStepContext.Request(
            context_id=self.context, include_results=False))
        if not response.success:
            raise OutcomeUnknown('CLEANUP_UNKNOWN: ' + response.status)
        state = json.loads(response.state_json)[0]
        if state['busy'] or any('STATE_UNKNOWN' in item['status'] for item in state['results']):
            raise OutcomeUnknown('CLEANUP_UNKNOWN: context busy or unresolved')
        response = self.request('clear_step_context', srv.ClearStepContext.Request(
            context_id=self.context, expected_revision=state['revision']))
        if not response.success:
            raise OutcomeUnknown('CLEANUP_UNKNOWN: ' + response.status)
        self.has_context = False
        self.revision = 0


class SimulatorPipeline:
    """One in-memory task queue with no sessions, stage runner or automatic insertion."""

    def __init__(self, transport):
        """Inject the master transport for replay or isolated fixture validation."""
        self.transport = transport
        self.lock = threading.RLock()
        self.tasks = []
        self.object_name = ''
        self.destination_name = ''
        self.lift_distance = .05
        self.busy = False
        self.unknown = False
        self.current_task = ''
        self.last_run = None

    def append(self, object_name, destination_name, lift_distance, task_ids):
        """Append exactly the requested consecutive reference stages, without executing."""
        with self.lock:
            if self.busy or self.unknown:
                raise RuntimeError('PIPELINE_BUSY_OR_UNKNOWN')
            if not task_ids:
                raise ValueError('TASKS_REQUIRED')
            target = object_name or self.object_name
            destination = destination_name or self.destination_name
            distance = lift_distance or self.lift_distance
            if not target.strip() or not destination.strip():
                raise ValueError('OBJECT_AND_DESTINATION_REQUIRED')
            if not math.isfinite(distance) or distance <= 0:
                raise ValueError('LIFT_DISTANCE_MUST_BE_POSITIVE_METRES')
            if self.tasks and (target != self.object_name or
                               destination != self.destination_name or
                               distance != self.lift_distance):
                raise ValueError('FLUSH_BEFORE_CHANGING_TARGETS')
            recipe = simulator_steps(target, destination, distance)
            selected = recipe[len(self.tasks):len(self.tasks) + len(task_ids)]
            if list(task_ids) != [item.step_id for item in selected]:
                next_id = recipe[len(self.tasks)].step_id if len(
                    self.tasks) < len(recipe) else 'none'
                raise ValueError('OUT_OF_ORDER_INSERT: next task is ' + next_id)
            self.object_name, self.destination_name = target, destination
            self.lift_distance = distance
            self.tasks.extend(selected)
            return len(self.tasks)

    def display(self):
        """Show only this persistent pipeline and the previous run's final result."""
        with self.lock:
            return {'busy': self.busy, 'unknown': self.unknown,
                    'object': self.object_name, 'destination': self.destination_name,
                    'current_task': self.current_task,
                    'tasks': [{'id': item.step_id, 'api': item.function_name}
                              for item in self.tasks],
                    'last_run': copy.deepcopy(self.last_run)}

    def feedback(self, message):
        """Expose the actual active master task while execute is waiting."""
        with self.lock:
            self.current_task = message.feedback.step_id

    def execute(self):
        """Execute the entire inserted queue once, then flush on success or known failure."""
        with self.lock:
            if self.busy or self.unknown:
                raise RuntimeError('PIPELINE_BUSY_OR_UNKNOWN')
            if not self.tasks:
                raise ValueError('PIPELINE_EMPTY')
            tasks = list(self.tasks)
            self.busy = True
        try:
            try:
                success, report = self.transport.execute(tasks, self.feedback)
            except OutcomeUnknown:
                raise
            except Exception as error:
                success, report = False, {'status': str(error), 'results': []}
            with self.lock:
                self.last_run = report
            self.transport.clear()
            with self.lock:
                self.tasks.clear()
                self.current_task = ''
            return success, report
        except Exception as error:
            with self.lock:
                self.unknown = True
                self.last_run = {'status': 'OUTCOME_UNKNOWN: ' + str(error), 'results': []}
            return False, self.last_run
        finally:
            with self.lock:
                self.busy = False

    def flush(self):
        """Discard idle queued calculations; never clear an unresolved operation."""
        with self.lock:
            if self.busy or self.unknown:
                raise RuntimeError('PIPELINE_BUSY_OR_UNKNOWN: cannot flush')
            try:
                self.transport.clear()
            except Exception:
                self.unknown = True
                raise
            self.tasks.clear()
            self.current_task = ''


class SimulatorPipelineNode(Node):
    """Expose four simulator-only services; physical APIs never enter its queue."""

    def __init__(self):
        """Launch once and keep one queue available until shutdown."""
        super().__init__('simulator_pipeline')
        simulator_domain(os.environ.get('ROS_DOMAIN_ID', '0'))
        group = ReentrantCallbackGroup()
        self.grasp_preview = self.create_publisher(
            MarkerArray, '/simulator/selected_grasp',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pipeline = SimulatorPipeline(SimulatorTransport(self, group))
        self.create_service(srv.AppendSimulatorTasks, '/simulator/append_tasks', self.append,
                            callback_group=group)
        for name, handler in (('display_pipeline', self.display),
                              ('execute_pipeline', self.execute), ('flush_pipeline', self.flush)):
            self.create_service(Trigger, '/simulator/' + name, handler, callback_group=group)
        self.get_logger().info('Mock pipeline ready: append_tasks, display_pipeline, '
                               'execute_pipeline, flush_pipeline; no physical motion')

    def append(self, request, response):
        """Validate input and insert only explicitly named tasks."""
        try:
            response.queued_tasks = self.pipeline.append(
                request.object_name, request.destination_name,
                request.lift_distance_m, request.tasks)
            response.success, response.status = True, 'TASKS_INSERTED_NO_EXECUTION'
        except Exception as error:
            response.success, response.status = False, str(error)
        return response

    def display(self, request, response):
        """Return queue JSON; this does not capture new camera data."""
        response.success = True
        response.message = json.dumps(self.pipeline.display())
        return response

    def execute(self, request, response):
        """Return final calculation completion, not an asynchronous acknowledgement."""
        try:
            response.success, report = self.pipeline.execute()
            response.message = json.dumps(report)
            self.get_logger().info('Simulator result: ' + response.message)
        except Exception as error:
            response.success, response.message = False, str(error)
        return response

    def flush(self, request, response):
        """Empty only the idle simulator calculation pipeline."""
        try:
            self.pipeline.flush()
            response.success, response.message = True, 'PIPELINE_FLUSHED'
        except Exception as error:
            response.success, response.message = False, str(error)
        return response


def main():
    """Spin the persistent service node with callbacks available during calculation."""
    rclpy.init()
    node = SimulatorPipelineNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
