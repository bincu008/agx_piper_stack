"""Fixed bowl-then-cup replay benchmark using the shared API/result store."""

import argparse
import json
import time
import uuid
from pathlib import Path

from cv_bridge import CvBridge

from my_actions import srv
from my_actions.msg import ApiCallTrace

import numpy as np

from rcl_interfaces.srv import GetParameters, SetParameters

import rclpy
from rclpy.parameter import Parameter
from rclpy.serialization import deserialize_message, serialize_message

from sensor_msgs_py import point_cloud2

from task_pipeline.registry import FUNCTIONS


OBJECTS = ('small white bowl at bottom left', 'beige cup')
DESTINATION = 'dark rectangular tray'
CALCULATION_APIS = {
    'commit_floor_slots', 'capture_observation', 'pin_grasp_observation',
    'pin_placement_observation', 'resolve_target', 'check_target_view',
    'plan_layout', 'get_floor_slots', 'perceive_scene', 'get_active_target',
}


class MockBowlCup:
    """Call only calculation APIs; virtual placement is owned by plan_layout."""

    def __init__(self, node, directory, lift_distance=.20):
        """Create a fresh report and subscribe to nested shared API timings."""
        if not np.isfinite(lift_distance) or lift_distance <= 0:
            raise ValueError('Lift distance must be finite and positive')
        self.lift_distance = float(lift_distance)
        self.node, self.directory = node, directory
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory/'report.json').exists():
            raise ValueError('Use a fresh output directory for each benchmark run')
        self.context = 'mock_bowl_cup_'+uuid.uuid4().hex
        self.rows, self.traces, self.checks = [], [], []
        self.subscription = node.create_subscription(
            ApiCallTrace, '/pipeline/api_traces', self.trace, 1000)
        self.report = {'mode': 'calculation_only_virtual_layout', 'success': False,
                       'context_id': self.context, 'objects': list(OBJECTS),
                       'destination': DESTINATION, 'lift_distance_m': self.lift_distance,
                       'api_calls': self.rows,
                       'api_traces': self.traces, 'checks': self.checks,
                       'physical_motion_executed': False}

    def trace(self, message):
        """Retain nested API timings separately from their inclusive parent."""
        from rosidl_runtime_py.convert import message_to_ordereddict
        self.traces.append(dict(message_to_ordereddict(message)))

    def check(self, label, condition):
        """Fail at the first unexpected response; preserve diagnostics."""
        self.checks.append({'check': label, 'passed': bool(condition)})
        if not condition:
            raise RuntimeError('UNEXPECTED_API_DATA: '+label)

    def request(self, schema, endpoint, query, timeout=30.):
        """Wait for one response without automatically retrying the call."""
        client = self.node.create_client(schema, endpoint)
        try:
            if not client.wait_for_service(timeout_sec=10.):
                raise RuntimeError('SERVICE_UNAVAILABLE: '+endpoint)
            future = client.call_async(query)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
            if not future.done():
                raise RuntimeError('RESPONSE_TIMEOUT: '+endpoint)
            return future.result()
        finally:
            self.node.destroy_client(client)

    def call(self, label, name, query):
        """Run a typed shared step and record start/end before the next call."""
        if name not in CALCULATION_APIS:
            raise RuntimeError('MOCK_PIPELINE_FORBIDS_API: '+name)
        started = time.monotonic()
        row = {'step': label, 'api': name, 'start_seconds': started-self.started}
        self.rows.append(row)
        print(f'{len(self.rows):02d} {label}: {name} RUNNING', flush=True)
        result = self.request(srv.RunPipelineStep, FUNCTIONS['run_step'].endpoint,
                              srv.RunPipelineStep.Request(
                                  context_id=self.context, step_id=label, function_name=name,
                                  request_cdr=serialize_message(query)), timeout=600.)
        row.update(end_seconds=time.monotonic()-self.started,
                   runtime_seconds=time.monotonic()-started,
                   success=result.success and result.result.success,
                   status=result.result.status or result.status)
        print(f"{label}: {row['status']} ({row['runtime_seconds']:.3f}s)", flush=True)
        payload = bytes(result.result.response_cdr)
        (self.directory/(label+'_response.cdr')).write_bytes(payload)
        if not result.success or not result.result.success:
            raise RuntimeError(label+': '+row['status'])
        return deserialize_message(payload, FUNCTIONS[name].service_type.Response)

    def configure(self):
        """Enable diagnostics only for this isolated replay's placement node."""
        keys = ['layout_debug_directory', 'layout_debug_stop_on_failure']
        old = self.request(GetParameters, '/placement_node/get_parameters',
                           GetParameters.Request(names=keys)).values
        self.old_parameters = (keys, old)
        values = [Parameter(keys[0], value=str(self.directory)).to_parameter_msg(),
                  Parameter(keys[1], value=True).to_parameter_msg()]
        result = self.request(SetParameters, '/placement_node/set_parameters',
                              SetParameters.Request(parameters=values))
        self.check('benchmark diagnostics configured', all(x.successful for x in result.results))

    def verify_target(self, label, target, header):
        """Check point/mask/header association before passing to the planner."""
        self.check(label+' observation association', target.observation_header == header)
        self.check(label+' geometry frame', target.object_cloud.header.frame_id == 'base_link')
        self.check(label+' measured geometry', target.object_cloud.width >= 80)
        mask = CvBridge().imgmsg_to_cv2(target.mask, 'mono8')
        self.check(label+' nonempty mask', np.count_nonzero(mask) >= 80)
        self.check(label+' mask stamp', target.mask.header.stamp == header.stamp)

    def inspect_layout(self, response, masks):
        """Check full held paths, actual FK, grid exclusivity and updated scene."""
        self.check('both objects planned in requested order',
                   list(response.object_names) == list(OBJECTS) and len(response.placements) == 2)
        for index in range(1, 3):
            prefix = self.directory/f'layout_object_{index}'
            request = deserialize_message(
                prefix.with_name(prefix.name+'_plan_request.cdr').read_bytes(),
                srv.PlanPlacement.Request)
            plan = deserialize_message(
                prefix.with_name(prefix.name+'_plan_response.cdr').read_bytes(),
                srv.PlanPlacement.Response)
            self.check(f'object {index} full held motion',
                       plan.success and plan.motion_success and request.plan_pick_place_motion)
            for stage in ('lift', 'transfer', 'descent'):
                path = np.array(getattr(plan, stage+'_path_flat'))
                self.check(f'object {index} {stage} path',
                           len(path) >= 6 and len(path) % 6 == 0 and np.isfinite(path).all())
            achieved_lift = (plan.lift_ee_pose.pose.position.z
                             - plan.selected_grasp_candidate.grasp_pose.pose.position.z)
            self.check(f'object {index} requested lift achieved',
                       achieved_lift >= self.lift_distance-.002)
            self.check(f'object {index} selected grasp', plan.selected_grasp_candidate.id > 0)
            self.check(f'object {index} base placement',
                       plan.placement_ee_pose.header.frame_id == 'base_link')
        second = deserialize_message(
            (self.directory/'layout_object_2_grasp_request.cdr').read_bytes(),
            srv.FindGraspCandidates.Request)
        points = point_cloud2.read_points_numpy(
            second.collision_scene, field_names=('x', 'y', 'z'))
        pixel_count = masks[0].height*masks[0].width
        self.check('cup scene includes appended placed bowl', len(points) > pixel_count)
        mask = CvBridge().imgmsg_to_cv2(masks[0], 'mono8').reshape(-1) > 0
        self.check('bowl removed from its old source pixels',
                   np.isfinite(points[:pixel_count][mask]).all(axis=1).sum() < mask.sum()/2)
        slots = self.call('final_slots', 'get_floor_slots', srv.GetManipulationMemory.Request())
        grids = json.loads(slots.state_json)
        self.check('one retained grid', len(grids) == 1)
        self.check('occupied slots retained', bool(grids[0]['reserved_cells']))
        self.report['grid'] = grids
        self.report['note'] = (
            'Full held lift/transfer/descent calculated sequentially with virtual scene updates. '
            'No physical tracking/contact, camera movement, or confirmed release was emulated. '
            'This benchmark does not create an executable multi-object session.')

    def collect_objects(self):
        """Retain successes and failures without presenting a partial run as complete."""
        objects = []
        for index, name in enumerate(OBJECTS, 1):
            row = {'name': name, 'physical_execution': 'not_executed'}
            for kind, schema in (('grasp', srv.FindGraspCandidates),
                                 ('plan', srv.PlanPlacement)):
                path = self.directory/f'layout_object_{index}_{kind}_response.cdr'
                if path.is_file():
                    result = deserialize_message(path.read_bytes(), schema.Response)
                    row[kind] = {'success': result.success, 'status': result.status}
                    if kind == 'grasp':
                        row[kind]['candidates'] = len(result.candidates)
                    else:
                        grasp_z = result.selected_grasp_candidate.grasp_pose.pose.position.z
                        row[kind].update(
                            motion_success=result.motion_success,
                            executable=False,
                            motion_status=result.motion_status,
                            grasp_id=result.selected_grasp_candidate.id,
                            placement_id=result.selected_candidate.id,
                            requested_lift_m=self.lift_distance,
                            actual_lift_m=(result.lift_ee_pose.pose.position.z-grasp_z
                                           if result.motion_success else None),
                            lift_seconds=result.lift_seconds,
                            transfer_seconds=result.transfer_seconds,
                            validation_seconds=result.validation_seconds,
                            validation_calls=json.loads(result.validation_calls_json or '[]'),
                            waypoint_counts={stage: len(getattr(result, stage+'_path_flat'))//6
                                             for stage in ('lift', 'transfer', 'descent')})
            objects.append(row)
        self.report['object_results'] = objects
        source = self.directory/'resolve_0_response.cdr'
        scene = self.directory/'layout_object_2_grasp_request.cdr'
        if source.is_file() and scene.is_file():
            target = deserialize_message(
                source.read_bytes(), srv.ResolveManipulationTarget.Response)
            query = deserialize_message(scene.read_bytes(), srv.FindGraspCandidates.Request)
            points = point_cloud2.read_points_numpy(
                query.collision_scene, field_names=('x', 'y', 'z'))
            mask = CvBridge().imgmsg_to_cv2(target.mask, 'mono8').reshape(-1) > 0
            pixels = len(mask)
            remaining = int(np.isfinite(points[:pixels][mask]).all(axis=1).sum())
            appended = len(points)-pixels
            self.report['virtual_scene_check'] = {
                'bowl_source_pixels': int(mask.sum()), 'remaining_bowl_source_pixels': remaining,
                'appended_placed_points': appended,
                'passed': remaining < mask.sum()/2 and appended > 0}

    def run(self):
        """Ground once, check visibility, then plan bowl followed by cup."""
        self.started = time.monotonic()
        try:
            deadline = time.monotonic()+10.
            while time.monotonic() < deadline:
                nodes = [name for name, _ in self.node.get_node_names_and_namespaces()]
                if 'mock_hardware' in nodes:
                    break
                rclpy.spin_once(self.node, timeout_sec=.1)
            self.check('mock node present', 'mock_hardware' in nodes)
            self.check('no hardware driver nodes',
                       not any('agx_arm' in name or 'orbbec' in name for name in nodes))
            gate = self.request(GetParameters, '/piper_master_control/get_parameters',
                                GetParameters.Request(names=['allow_physical_grasp_execution']))
            self.check('physical execution disabled',
                       len(gate.values) == 1 and gate.values[0].type == 1
                       and not gate.values[0].bool_value)
            self.configure()
            self.call('reset_slots', 'commit_floor_slots',
                      srv.CommitFloorSlots.Request(reset=True))
            observation = self.call('observe', 'capture_observation',
                                    srv.CaptureObservation.Request())
            header = observation.cloud.header
            for role in ('grasp', 'placement'):
                self.call('pin_'+role, 'pin_'+role+'_observation',
                          srv.PinLayoutObservation.Request(observation_header=header))
            inventory = self.call('perceive_scene', 'perceive_scene', srv.PerceiveScene.Request(
                object_names=list(OBJECTS), destination_name=DESTINATION,
                image=observation.image, observation_cloud=observation.cloud,
                camera_transform=observation.camera_transform))
            self.check('all scene targets observed together',
                       len(inventory.observed) == 3 and all(inventory.observed))
            targets = []
            for index, name in enumerate((*OBJECTS, DESTINATION)):
                role = 'destination' if index == 2 else 'object'
                target = self.call('resolve_'+str(index), 'get_active_target',
                                   srv.ResolveManipulationTarget.Request(
                                       name=name, role=role, image=observation.image,
                                       observation_cloud=observation.cloud,
                                       camera_transform=observation.camera_transform))
                self.verify_target(name, target, header)
                targets.append(target)
            self.check('distinct source identities', targets[0].object_id != targets[1].object_id)
            for index, name in enumerate(OBJECTS):
                view = self.call('visibility_'+str(index), 'check_target_view',
                                 srv.CheckTargetView.Request(
                                     name=name, role='object', observation_cloud=observation.cloud,
                                     camera_transform=observation.camera_transform))
                self.check(name+' visible for this frozen replay', view.status == 'READY')
            masks = [target.mask for target in targets[:2]]
            result = self.call('bowl_then_cup', 'plan_layout', srv.PlanObjectLayout.Request(
                observation_header=header, destination_mask=targets[2].mask,
                object_masks=masks, object_names=list(OBJECTS), destination_name=DESTINATION,
                orientation_policy='upright', lift_distance_m=self.lift_distance))
            self.inspect_layout(result, masks)
            tray = self.call('reuse_tray', 'resolve_target', srv.ResolveManipulationTarget.Request(
                name=DESTINATION, role='destination', image=observation.image,
                observation_cloud=observation.cloud,
                camera_transform=observation.camera_transform))
            self.check('tray reused with identical base geometry',
                       tray.reused and tray.object_cloud == targets[2].object_cloud)
            self.report['success'] = True
        except Exception as error:
            self.report['failure'] = str(error)
            if hasattr(self, 'old_parameters'):
                try:
                    slots = self.call('slots_after_failure', 'get_floor_slots',
                                      srv.GetManipulationMemory.Request())
                    self.report['grid'] = json.loads(slots.state_json)
                except Exception as diagnostic_error:
                    self.report['diagnostic_error'] = str(diagnostic_error)
            raise
        finally:
            self.report['wall_seconds'] = time.monotonic()-self.started
            try:
                self.collect_objects()
            except Exception as error:
                self.report['artifact_inspection_error'] = str(error)
                self.report['success'] = False
            (self.directory/'report.json').write_text(json.dumps(self.report, indent=2))
            if hasattr(self, 'old_parameters'):
                keys, values = self.old_parameters
                from rcl_interfaces.msg import Parameter as ParameterMessage
                self.request(SetParameters, '/placement_node/set_parameters',
                             SetParameters.Request(parameters=[
                                 ParameterMessage(name=k, value=v) for k, v in zip(keys, values)]))


def main():
    """Run on an isolated replay domain; never invoke actuator APIs."""
    parser = argparse.ArgumentParser()
    default_output = (Path.home() / 'agx_piper_data/reports/mock_bowl_cup'
                      / time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--output', type=Path, default=default_output)
    parser.add_argument('--lift-distance', type=float, default=.20)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('mock_bowl_cup_pipeline')
    try:
        MockBowlCup(node, args.output.expanduser().resolve(), args.lift_distance).run()
        print('MOCK BOWL THEN CUP PASSED:', args.output, flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
