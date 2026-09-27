import threading

from master_node.task_manager import canonical_task_type
from master_node.task_manager import Task
from master_node.task_manager import TaskManager
from master_node.task_manager import TaskQueue
from master_node.task_manager import TaskState
from master_node.world_state import WorldStateManager


class RecordingTask(Task):
    def __init__(self, task_type, target, calls, result=True):
        super().__init__(task_type, target)
        self.calls = calls
        self.result = result

    def execute(self):
        self.calls.append(self.task_type)
        return self.result


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(('info', message))

    def error(self, message):
        self.messages.append(('error', message))


def test_queue_append_prepend_insert_replace_complete_and_flush():
    state = WorldStateManager()
    queue = TaskQueue(state)
    observe = Task('OBSERVE_OBJECT', 'obj_001')
    grasp = Task('GRASP', 'obj_001')
    align = Task('ALIGN_GRIPPER', 'obj_001')

    queue.append(grasp)
    queue.prepend(observe)
    queue.insert(1, align)
    replacement = Task('APPROACH_OBJECT', 'obj_001')
    assert queue.replace(1, replacement) is align
    assert queue.snapshot() == [
        {'type': 'OBSERVE_OBJECT', 'target': 'obj_001'},
        {'type': 'APPROACH_OBJECT', 'target': 'obj_001'},
        {'type': 'GRASP', 'target': 'obj_001'},
    ]

    assert queue.remove_completed(observe) is False
    observe.state = TaskState.SUCCESS
    assert queue.remove_completed(observe) is True
    assert state.snapshot()['task_queue'][0]['type'] == 'APPROACH_OBJECT'

    queue.flush()
    assert queue.snapshot() == []
    assert state.snapshot()['task_queue'] == []


def test_successful_execution_self_depletes_queue():
    calls = []
    manager = TaskManager()
    manager.append(RecordingTask('OBSERVE_OBJECT', 'obj_001', calls))
    manager.append(RecordingTask('GRASP', 'obj_001', calls))

    assert manager.execute() is True
    manager._worker_thread.join(timeout=1.0)

    assert calls == ['OBSERVE_OBJECT', 'GRASP']
    assert manager.queue.snapshot() == []


def test_obstacle_event_inserts_recovery_before_interrupted_task():
    started = threading.Event()
    release = threading.Event()

    class InterruptedTask(Task):
        def execute(self):
            started.set()
            release.wait(timeout=1.0)
            return False

    manager = TaskManager()
    original = InterruptedTask('MOVE_TO_OBJECT', 'obj_002')
    manager.append(original)
    assert manager.execute() is True
    assert started.wait(timeout=1.0)

    assert manager.insert_obstacle_recovery('obj_002') is True
    assert manager.queue.snapshot() == [
        {'type': 'RETREAT', 'target': 'obj_002'},
        {'type': 'REPLAN', 'target': 'obj_002'},
        {'type': 'MOVE_TO_OBJECT', 'target': 'obj_002'},
    ]

    manager.flush()
    release.set()
    manager._worker_thread.join(timeout=1.0)


def test_legacy_look_at_maps_to_observe_object():
    state = WorldStateManager()
    state.upsert_object(
        'obj_001',
        name='apple',
        position=[0.4, 0.1, 0.2],
        role='target',
        visible=True,
    )
    task = Task('LOOK_AT', 'obj_001')

    assert canonical_task_type('LOOK_AT') == 'OBSERVE_OBJECT'
    assert task.as_dict() == {
        'type': 'OBSERVE_OBJECT',
        'target': 'obj_001',
    }


def test_pipeline_is_logged_on_append_execution_and_task_start():
    calls = []
    logger = RecordingLogger()
    manager = TaskManager(logger=logger)

    manager.append(RecordingTask('GRASP', 'obj_001', calls))
    assert manager.execute() is True
    manager._worker_thread.join(timeout=1.0)

    output = '\n'.join(message for _, message in logger.messages)
    assert 'FULL PIPELINE (appended GRASP)' in output
    assert 'FULL PIPELINE (execution begins)' in output
    assert 'TASK BEGIN: GRASP target=obj_001' in output
