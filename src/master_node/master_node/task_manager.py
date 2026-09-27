"""Generic, thread-safe task queue and self-depleting task manager."""

from dataclasses import dataclass, field
from enum import Enum
import threading


class TaskState(Enum):
    """Internal execution state; it is not part of the public task schema."""

    PENDING = 0
    RUNNING = 1
    FAILED = 2
    SUCCESS = 3


def canonical_task_type(task_type):
    """Map legacy names onto the generic high-level task vocabulary."""
    normalized = task_type.strip().upper()
    if normalized == 'LOOK_AT':
        return 'OBSERVE_OBJECT'
    return normalized


@dataclass(eq=False)
class Task:
    """A high-level task that deliberately contains no joint states."""

    task_type: str
    target: str = None
    state: TaskState = field(default=TaskState.PENDING, init=False)

    def __post_init__(self):
        self.task_type = canonical_task_type(self.task_type)
        if not self.task_type:
            raise ValueError('task_type must be a non-empty string')
        if self.target is not None and not isinstance(self.target, str):
            raise ValueError('target must be an object id or None')

    def as_dict(self):
        """Return the minimal public task representation."""
        item = {'type': self.task_type}
        if self.target is not None:
            item['target'] = self.target
        return item

    def execute(self):
        """Execute the task in a concrete motion/perception adapter."""
        raise NotImplementedError


class TaskQueue:
    """Own queue mutation and keep world-state snapshots synchronized."""

    def __init__(self, world_state=None):
        self._items = []
        self._lock = threading.RLock()
        self._world_state = world_state
        self._sync_world_state()

    def _sync_world_state(self):
        if self._world_state is not None:
            self._world_state.replace_task_queue(
                [task.as_dict() for task in self._items]
            )

    def append(self, task):
        with self._lock:
            self._items.append(task)
            self._sync_world_state()

    def prepend(self, task):
        self.insert(0, task)

    def prepend_many(self, tasks):
        with self._lock:
            self._items[0:0] = list(tasks)
            self._sync_world_state()

    def insert(self, index, task):
        with self._lock:
            self._items.insert(index, task)
            self._sync_world_state()

    def insert_before(self, existing_task, tasks):
        """Insert tasks immediately before an existing queued task."""
        with self._lock:
            index = self._identity_index(existing_task)
            self._items[index:index] = list(tasks)
            self._sync_world_state()

    def replace(self, index, task):
        with self._lock:
            previous = self._items[index]
            self._items[index] = task
            self._sync_world_state()
            return previous

    def remove_completed(self, task):
        """Remove a specific task only after it reports success."""
        if task.state is not TaskState.SUCCESS:
            return False
        with self._lock:
            try:
                index = self._identity_index(task)
            except ValueError:
                return False
            self._items.pop(index)
            self._sync_world_state()
            return True

    def flush(self):
        with self._lock:
            self._items.clear()
            self._sync_world_state()

    def peek(self):
        with self._lock:
            return self._items[0] if self._items else None

    def has_tasks_before(self, task):
        with self._lock:
            try:
                return self._identity_index(task) > 0
            except ValueError:
                return False

    def snapshot(self):
        with self._lock:
            return [task.as_dict() for task in self._items]

    def __len__(self):
        with self._lock:
            return len(self._items)

    def __bool__(self):
        return len(self) > 0

    def _identity_index(self, task):
        for index, queued_task in enumerate(self._items):
            if queued_task is task:
                return index
        raise ValueError('task is not in the queue')


class TaskManager:
    """Execute the queue head and remove tasks only after success."""

    def __init__(self, world_state=None, logger=None, cancel_current=None):
        self.queue = TaskQueue(world_state)
        self._logger = logger
        self._cancel_current = cancel_current
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._is_executing = False
        self._current_task = None

    @property
    def is_executing(self):
        with self._state_lock:
            return self._is_executing

    @property
    def stop_requested(self):
        return self._stop_event.is_set()

    def append(self, task):
        self.queue.append(task)
        self.log_pipeline(f'appended {task.task_type}')

    def prepend(self, task):
        self.queue.prepend(task)
        self.log_pipeline(f'prepended {task.task_type}')

    def insert(self, index, task):
        self.queue.insert(index, task)
        self.log_pipeline(f'inserted {task.task_type} at index {index}')

    def replace(self, index, task):
        previous = self.queue.replace(index, task)
        self.log_pipeline(
            f'replaced {previous.task_type} with {task.task_type}'
        )
        return previous

    def flush(self):
        self._stop_event.set()
        self.queue.flush()
        if self._cancel_current is not None:
            self._cancel_current()
        self._log('info', '[TASK MANAGER] Queue flushed')

    def execute(self):
        with self._state_lock:
            if self._is_executing or not self.queue:
                return False
            self._stop_event.clear()
            self._is_executing = True
            self.log_pipeline('execution begins')
            self._worker_thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
            )
            self._worker_thread.start()
            return True

    def insert_obstacle_recovery(self, target=None):
        """Handle an obstacle event without encoding joint states as tasks."""
        recovery = [Task('RETREAT', target), Task('REPLAN', target)]
        with self._state_lock:
            current = self._current_task
            if current is not None:
                existing_types = [
                    item['type'] for item in self.queue.snapshot()
                ]
                if 'RETREAT' in existing_types or 'REPLAN' in existing_types:
                    return False
                self.queue.insert_before(current, recovery)
            else:
                self.queue.prepend_many(recovery)

        self.log_pipeline('obstacle recovery inserted')
        if self._cancel_current is not None:
            self._cancel_current()
        return True

    def _run_loop(self):
        completed_any = False
        try:
            while not self._stop_event.is_set():
                task = self.queue.peek()
                if task is None:
                    break

                with self._state_lock:
                    self._current_task = task
                task.state = TaskState.RUNNING
                target = f' target={task.target}' if task.target else ''
                self._log(
                    'info',
                    f'[TASK MANAGER] TASK BEGIN: {task.task_type}{target}',
                )

                try:
                    success = bool(task.execute())
                except Exception as exc:  # Keep the worker state consistent.
                    self._log(
                        'error',
                        f'[TASK MANAGER] {task.task_type} raised: {exc}',
                    )
                    success = False

                if success:
                    task.state = TaskState.SUCCESS
                    self.queue.remove_completed(task)
                    completed_any = True
                elif self.queue.has_tasks_before(task):
                    # An event inserted recovery work ahead of the interrupted task.
                    task.state = TaskState.PENDING
                else:
                    task.state = TaskState.FAILED
                    self._log(
                        'error',
                        f'[TASK MANAGER] Task {task.task_type} failed; '
                        'aborting queue',
                    )
                    self.queue.flush()
                    break
        finally:
            with self._state_lock:
                self._current_task = None
                self._is_executing = False
            if completed_any and not self._stop_event.is_set() and not self.queue:
                self._log('info', '[TASK MANAGER] Queue execution successful')

    def _log(self, level, message):
        if self._logger is not None:
            getattr(self._logger, level)(message)

    def log_pipeline(self, reason):
        """Print a stable full-queue snapshot for operator inspection."""
        snapshot = self.queue.snapshot()
        lines = [
            f'[TASK MANAGER] FULL PIPELINE ({reason}) — {len(snapshot)} task(s)'
        ]
        if not snapshot:
            lines.append('  <empty>')
        for index, task in enumerate(snapshot, start=1):
            target = f" target={task['target']}" if 'target' in task else ''
            lines.append(f"  {index}. {task['type']}{target}")
        self._log('info', '\n'.join(lines))
