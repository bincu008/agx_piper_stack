"""Minimal, thread-safe world state for manipulation orchestration."""

from copy import deepcopy
import math
import threading


class WorldStateManager:
    """Own the small persistent state shared by orchestration components."""

    def __init__(self):
        self._lock = threading.RLock()
        self._observation_meta = {}
        self._next_object_index = 1
        self._state = {
            'robot': {
                'state': 'IDLE',
                'holding': None,
            },
            'objects': {},
            'task_queue': [],
        }

    def snapshot(self):
        """Return an isolated copy suitable for readers and future reasoning."""
        with self._lock:
            return deepcopy(self._state)

    @property
    def robot_state(self):
        """Return the current high-level robot state."""
        with self._lock:
            return self._state['robot']['state']

    def set_robot_state(self, state):
        """Update the high-level robot state."""
        if not isinstance(state, str) or not state:
            raise ValueError('robot state must be a non-empty string')
        with self._lock:
            self._state['robot']['state'] = state

    def set_holding(self, object_id):
        """Record the held object id, or None when the gripper is empty."""
        if object_id is not None and not isinstance(object_id, str):
            raise ValueError('holding must be an object id or None')
        with self._lock:
            self._state['robot']['holding'] = object_id

    def upsert_object(self, object_id, name, position, role, visible):
        """Insert or replace a minimal object record in the world frame."""
        if not isinstance(object_id, str) or not object_id:
            raise ValueError('object_id must be a non-empty string')
        if not isinstance(name, str) or not name:
            raise ValueError('name must be a non-empty string')
        if not isinstance(role, str) or not role:
            raise ValueError('role must be a non-empty string')
        if len(position) != 3:
            raise ValueError('position must contain exactly three values')

        normalized_position = [float(value) for value in position]
        if not all(math.isfinite(value) for value in normalized_position):
            raise ValueError('position values must be finite')

        record = {
            'name': name,
            'position': normalized_position,
            'role': role,
            'visible': bool(visible),
        }
        with self._lock:
            self._state['objects'][object_id] = record
            return deepcopy(record)

    def get_object(self, object_id):
        """Return a copy of an object record, or None when it is unknown."""
        with self._lock:
            record = self._state['objects'].get(object_id)
            return deepcopy(record) if record is not None else None

    def set_object_visibility(self, object_id, visible):
        """Update visibility without discarding the stored world position."""
        with self._lock:
            if object_id not in self._state['objects']:
                raise KeyError(object_id)
            self._state['objects'][object_id]['visible'] = bool(visible)

    def associate_observation(self, name, position, tolerance=0.05):
        """Associate a detection by name and distance, or create a stable id."""
        normalized_position = [float(value) for value in position]
        with self._lock:
            best_id = None
            best_distance = float('inf')
            for object_id, record in self._state['objects'].items():
                if record['name'].casefold() != name.casefold():
                    continue
                distance = math.dist(record['position'], normalized_position)
                if distance <= tolerance and distance < best_distance:
                    best_id = object_id
                    best_distance = distance
            if best_id is None:
                best_id = f'obj_{self._next_object_index:03d}'
                self._next_object_index += 1
            self.upsert_object(
                best_id, name, normalized_position, 'target', True)
            self._observation_meta[best_id] = {'misses': 0}
            return best_id

    def objects_named(self, name):
        """Return copies of all records with the requested semantic name."""
        with self._lock:
            return [
                (object_id, deepcopy(record))
                for object_id, record in self._state['objects'].items()
                if record['name'].casefold() == name.casefold()
            ]

    def record_expected_miss(self, name, removal_threshold=3):
        """Remove objects only after repeated misses from an expected view."""
        removed = []
        with self._lock:
            matching_ids = [
                object_id
                for object_id, record in self._state['objects'].items()
                if record['name'].casefold() == name.casefold()
            ]
            for object_id in matching_ids:
                metadata = self._observation_meta.setdefault(
                    object_id, {'misses': 0})
                metadata['misses'] += 1
                self._state['objects'][object_id]['visible'] = False
                if metadata['misses'] >= removal_threshold:
                    del self._state['objects'][object_id]
                    del self._observation_meta[object_id]
                    removed.append(object_id)
        return removed

    def replace_task_queue(self, tasks):
        """Synchronize the public task-queue view with the task manager."""
        normalized = []
        for task in tasks:
            item = {'type': task['type']}
            if task.get('target') is not None:
                item['target'] = task['target']
            normalized.append(item)
        with self._lock:
            self._state['task_queue'] = normalized
