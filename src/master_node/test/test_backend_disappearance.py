"""A crashed service must end its caller without automatic operation retries."""

from concurrent.futures import Future
from types import SimpleNamespace

from master_node import step_runtime

from my_actions.srv import CommitFloorSlots

import pytest


def runtime(monkeypatch, ready, completion=None):
    """Advance a deterministic clock without ROS executors or real waiting."""
    clock = [0.0]
    future = Future()
    calls = []

    class Event:
        def __init__(self):
            self.done = False

        def set(self):
            self.done = True

        def is_set(self):
            return self.done

        def wait(self, duration):
            clock[0] += duration
            if completion is not None and clock[0] >= completion and not future.done():
                future.set_result(CommitFloorSlots.Response(success=True))
            return self.done

    def call(request):
        calls.append(request)
        return future

    monkeypatch.setattr(step_runtime.threading, 'Event', Event)
    monkeypatch.setattr(step_runtime.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(step_runtime.rclpy, 'ok', lambda: True)
    client = SimpleNamespace(
        wait_for_service=lambda **kw: True, call_async=call,
        service_is_ready=lambda: ready(clock[0]))
    node = SimpleNamespace(get_parameter=lambda _: SimpleNamespace(value=600.0))
    owner = SimpleNamespace(node=node, _clients={'commit_floor_slots': client})
    return owner, future, calls, clock


def test_lost_backend_terminates_wait_without_retry(monkeypatch):
    owner, future, calls, clock = runtime(monkeypatch, lambda _: False)
    with pytest.raises(RuntimeError, match='BACKEND_LOST_STATE_UNKNOWN'):
        step_runtime.MasterStepRuntime._native(owner, 'commit_floor_slots',
                                               CommitFloorSlots.Request())
    assert 2.0 <= clock[0] < 2.1
    assert future.cancelled()
    assert len(calls) == 1


def test_brief_discovery_gap_does_not_cancel_success(monkeypatch):
    owner, future, calls, _ = runtime(monkeypatch, lambda t: t >= 1.0, completion=1.5)
    result = step_runtime.MasterStepRuntime._native(
        owner, 'commit_floor_slots', CommitFloorSlots.Request())
    assert result.success and not future.cancelled()
    assert len(calls) == 1


def test_available_slow_backend_retains_normal_timeout_budget(monkeypatch):
    owner, future, calls, _ = runtime(monkeypatch, lambda _: True, completion=3.0)
    assert step_runtime.MasterStepRuntime._native(
        owner, 'commit_floor_slots', CommitFloorSlots.Request()).success
    assert not future.cancelled() and len(calls) == 1
