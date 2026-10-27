"""Memory-first camera acquisition before single-object grasp perception."""

import copy
import threading
import time

from geometry_msgs.msg import Point
from my_actions import srv


def stamp_ns(header):
    """Compare ROS capture times without changing their frame association."""
    return header.stamp.sec*1_000_000_000 + header.stamp.nanosec


class PerceptionAcquisition:
    """Compose shared memory, visibility and guarded empty-hand view APIs."""

    def __init__(self, memory):
        """Keep only one bounded post-view capture requirement."""
        self.memory = memory
        self.lock = threading.Lock()
        self.after_view = None

    def guard(self):
        """Do not acquire a new object while a prior physical task is incomplete."""
        self.memory.guard_mutation()
        if self.memory.node.state != 'IDLE':
            raise RuntimeError('ACQUISITION_REQUIRES_IDLE_EMPTY_HAND')
        execution = getattr(self.memory.node, 'execution_api', None)
        session = getattr(execution, 'prepared_grasp_lift', None)
        if session and session.record and session.record['phase'] != 'PREPARED':
            raise RuntimeError('PREVIOUS_PICK_PLACE_NOT_COMPLETE')

    def ensure_memory(self, request, response):
        """Use known locations; batch-perceive only when inventory is incomplete."""
        memory = self.memory
        try:
            self.guard()
            names, roles = memory.scene_perception.targets(request)
            with memory.lock:
                records = []
                for name, role in zip(names, roles):
                    record = memory._record(name, role)
                    hint = next((item for item in memory.directory.values()
                                 if memory.key(item['name']) == memory.key(name)), None)
                    if role == 'destination':
                        known = record if record and record['calibration'] == memory.calibration() else None
                    else:
                        known = record if record and record['calibration'] == memory.calibration() else hint
                    records.append(known)
            if not all(records):
                return memory.call('perceive_scene', request)
            response.success = True
            response.status = 'SCENE_MEMORY_READY_NO_INFERENCE'
            response.names, response.roles = names, roles
            response.observation_header = copy.deepcopy(request.observation_cloud.header)
            response.positions = [Point(**dict(zip(('x', 'y', 'z'), map(float, r['position']))))
                                  for r in records]
            response.observed = [False]*len(records)  # Location is not fresh visual evidence.
            response.target_statuses = ['LOCATION_REMEMBERED']*len(records)
            response.memory_revision = memory.revision
        except Exception as error:
            response.status = str(error)
        return response

    def ensure_view(self, request, response, allow_motion=False):
        """Move only when geometry requires it; never retry an actuator call."""
        admitted = self.lock.acquire(blocking=False)
        dispatched = False
        arrived = False
        try:
            if not admitted:
                raise RuntimeError('TARGET_VIEW_ACQUISITION_BUSY')
            memory = self.memory
            self.guard()
            marker = self.after_view
            if (marker and marker['key'] == memory.key(request.name) and
                    request.observation_cloud.data and
                    stamp_ns(request.observation_cloud.header) <= marker['stamp_ns']):
                raise RuntimeError('POST_VIEW_CAPTURE_STALE: capture a new synchronized frame')
            checked = memory.call('check_target_view', request)
            if not checked.success:
                return checked
            if checked.status == 'LOCATION_UNKNOWN':
                checked.success = False
                checked.status = 'TARGET_LOCATION_UNKNOWN: acquire initial scene inventory'
                return checked
            if checked.status == 'READY':
                return checked
            if checked.status != 'LOOK_REQUIRED':
                raise RuntimeError('TARGET_VIEW_STATUS_UNEXPECTED: '+checked.status)
            if not allow_motion:
                checked.success = False
                checked.status = 'LOOK_REQUIRED: motion-free calculation cannot acquire this view'
                return checked
            # This is an explicit actuator boundary, before any grasp calculation.
            execution = memory.node.execution_api
            execution._guard()
            planned = memory.call('plan_next_best_view', srv.PlanNextBestView.Request(
                target_name=request.name, camera_distance_m=.45, allow_current_view=False))
            if not planned.success:
                raise RuntimeError('OBJECT_VIEW_PLAN_FAILED: '+planned.status)
            dispatched = True
            moved = memory.call('execute_next_best_view', srv.HeldMotionStage.Request(
                object_name=request.name, plan_revision=planned.plan_revision))
            if not moved.success or not moved.motion_completed:
                memory.node.state = 'STATE_UNKNOWN'
                response.status = 'STATE_UNKNOWN: OBJECT_VIEW_EXECUTION_FAILED: '+moved.status
                return response
            arrived = True
            self.after_view = {
                'key': memory.key(request.name),
                'stamp_ns': memory.node.get_clock().now().nanoseconds}
            # Allow the wrist stream to settle; the following capture is still
            # explicit and its timestamp must be after confirmed arrival.
            time.sleep(.25)
            checked.status = 'OBJECT_VIEW_REACHED_RECAPTURE_REQUIRED'
            checked.visible = False
            checked.sample_count = checked.in_view_count = checked.clear_count = 0
            checked.visible_fraction = 0.
            return checked
        except Exception as error:
            response.status = str(error)
            if dispatched and not arrived:
                self.memory.node.state = 'STATE_UNKNOWN'
                response.status = 'STATE_UNKNOWN: '+response.status
            return response
        finally:
            if admitted:
                self.lock.release()
