"""Capture one immutable camera observation and delegate view geometry to C++."""

import math
import time

from my_actions import srv

import numpy as np


class ObserveView:
    """Calculate nearby empty-hand views, never execute them."""

    def __init__(self, execution):
        """Register the six-input public API on the existing master owner."""
        self.execution, self.node = execution, execution.node
        self.node.create_service(srv.GetObserveView, '/master/get_observe_view', self.calculate,
                                 callback_group=execution.group)

    def calculate(self, request, response):
        """Freeze synchronized manual FK*X and return a solved or unchanged target."""
        admitted = self.execution._lock.acquire(blocking=False)
        try:
            if not admitted:
                raise RuntimeError('EXECUTION_API_BUSY')
            self.execution.next_best_view.idle_empty_hand()
            values = [getattr(request, key) for key in ('dx', 'dy', 'dz', 'roll', 'pitch', 'yaw')]
            if (not all(math.isfinite(value) for value in values) or
                    np.linalg.norm(values[:3]) > .20 or any(abs(v) > math.pi for v in values[3:])):
                raise ValueError('VIEW_DELTA_REQUIRES_FINITE_0.20M_AND_PI_RAD_BOUNDS')
            calibration = self.node.hand_eye_matrix
            if calibration is None or np.shape(calibration) != (4, 4):
                raise RuntimeError('HAND_EYE_CALIBRATION_UNAVAILABLE')
            initial = self.joints()
            capture = self.node.step_runtime._call('capture_observation',
                                                   srv.CaptureObservation.Request())
            if not capture.success:
                raise RuntimeError('VIEW_CAPTURE_FAILED: '+capture.status)
            source = self.joints()
            if max(abs(a-b) for a, b in zip(initial, source)) > .005:
                raise RuntimeError('ARM_MOVED_DURING_VIEW_CAPTURE')
            stamp = self.node.current_joint_states.header.stamp
            cloud_stamp = capture.cloud.header.stamp
            if abs((stamp.sec-cloud_stamp.sec)+(stamp.nanosec-cloud_stamp.nanosec)*1.e-9) > .20:
                raise RuntimeError('CAMERA_JOINT_TIMESTAMP_MISMATCH')
            width = float(self.node.current_gripper_status.width)
            if not math.isfinite(width) or not -.001 <= width <= .101:
                raise RuntimeError('INVALID_VIEW_GRIPPER_FEEDBACK')
            # Closed-jaw encoder roundoff is not a negative geometric opening.
            # Keep the raw feedback unchanged; only the collision geometry is bounded.
            opening = min(.10, max(0., width))
            query = srv.SolveObserveView.Request(
                source_joints=source, camera_transform=capture.camera_transform,
                observation_cloud=capture.cloud,
                hand_eye_matrix=np.asarray(calibration).reshape(-1).tolist(),
                jaw_opening_m=opening)
            for key, value in zip(('dx', 'dy', 'dz', 'roll', 'pitch', 'yaw'), values):
                setattr(query, key, value)
            result = self.node.step_runtime._call('solve_observe_view', query)
            if max(abs(a-b) for a, b in zip(self.joints(), source)) > .005:
                raise RuntimeError('ARM_MOVED_DURING_VIEW_CALCULATION')
            for key in response.get_fields_and_field_types():
                setattr(response, key, getattr(result, key))
        except Exception as error:
            response.success = False
            response.status = str(error)
        finally:
            if admitted:
                self.execution._lock.release()
        return response

    def joints(self):
        """Reject missing or stale feedback instead of inventing a fallback."""
        now = time.monotonic()
        if any(now-getattr(self.node, field, 0.) > 1. for field in (
                'joint_feedback_time', 'pose_feedback_time', 'gripper_feedback_time')):
            raise RuntimeError('STALE_VIEW_FEEDBACK')
        state = self.node.current_joint_states
        if state is None or len(state.position) < 6 or self.node.current_gripper_status is None:
            raise RuntimeError('VIEW_FEEDBACK_UNAVAILABLE')
        result = list(state.position[:6])
        self.execution._joints(result, command_feedback=True)
        return result
