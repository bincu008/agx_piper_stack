"""Calculate-only orchestration fixtures; no ROS node or actuator clients."""

from types import SimpleNamespace
import threading
import time

from my_actions import srv
import numpy as np
import pytest
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from master_node.observe_view import ObserveView


def fixture(status='EXACT', moved=True):
    header=Header(frame_id='base_link')
    header.stamp.sec=100
    now=time.monotonic()
    node=SimpleNamespace(state='IDLE', _held_object=None, hand_eye_matrix=np.eye(4),
        joint_feedback_time=now, pose_feedback_time=now, gripper_feedback_time=now,
        current_joint_states=JointState(header=header,position=[0.]*6),
        current_gripper_status=SimpleNamespace(width=.05))
    calls=[]
    def call(name, query):
        calls.append((name,query))
        if name=='capture_observation':
            response=srv.CaptureObservation.Response(success=True)
            response.cloud.header=header
            response.camera_transform.header=header
            return response
        assert name=='solve_observe_view'
        return srv.SolveObserveView.Response(
            success=True,status=status,moved_target=moved,endpoint_safe=moved,
            source_joints=list(query.source_joints),
            target_joints=[.02]*6 if moved else list(query.source_joints))
    node.step_runtime=SimpleNamespace(_call=call)
    execution=SimpleNamespace(node=node,_lock=threading.Lock(),_joints=lambda *a,**kw: None,
        group=None,next_best_view=SimpleNamespace(idle_empty_hand=lambda: None))
    node.create_service=lambda *a,**kw: None
    return ObserveView(execution),calls


@pytest.mark.parametrize('status,moved',[('EXACT',True),('TARGET_COLLISION',False),
                                      ('NO_REACHABLE_CHANGE',False)])
def test_result_uses_frozen_capture_and_no_motion_calls(status,moved):
    api,calls=fixture(status,moved)
    response=api.calculate(srv.GetObserveView.Request(dx=.02,dy=-.02),
                           srv.GetObserveView.Response())
    assert response.success and response.moved_target==moved
    assert response.endpoint_safe==moved and response.status==status
    assert [name for name,_ in calls]==['capture_observation','solve_observe_view']
    query=calls[-1][1]
    assert query.dx==.02 and query.dy==-.02
    assert list(query.hand_eye_matrix)==np.eye(4).reshape(-1).tolist()
    if not moved:
        assert list(response.target_joints)==list(response.source_joints)


def test_stale_feedback_and_invalid_delta_are_errors_not_guessed_fallbacks():
    api,calls=fixture()
    api.node.joint_feedback_time-=2.
    response=api.calculate(srv.GetObserveView.Request(dx=.02),srv.GetObserveView.Response())
    assert not response.success and 'STALE' in response.status and not calls
    api,calls=fixture()
    response=api.calculate(srv.GetObserveView.Request(dx=float('nan')),
                           srv.GetObserveView.Response())
    assert not response.success and not calls


def test_closed_jaw_roundoff_changes_geometry_only():
    api,calls=fixture()
    api.node.current_gripper_status.width=-.00028
    result=api.calculate(srv.GetObserveView.Request(dx=.02),srv.GetObserveView.Response())
    assert result.success
    assert calls[-1][1].jaw_opening_m==0.
    assert api.node.current_gripper_status.width==-.00028
