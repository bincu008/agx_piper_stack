"""Minimal, thread-safe world state for manipulation orchestration."""

import copy
import math
import re
import threading

# Strip only an explicitly delimited instruction or a pronoun-qualified imperative.
_REGION = r'(?:body|rim|handle|side|edge|lip|top|base)'
_INSTRUCTION = (r'(?:please\s+)?grasp\s+'
                r'(?:it\s+by\s+(?:its\s+|the\s+)?|(?:its|the)\s+)?' + _REGION)
_PLAIN_INSTRUCTION = (r'(?:please\s+)?grasp\s+'
                      r'(?:it\s+by\s+(?:its\s+|the\s+)?|(?:its|the)\s+)' + _REGION)
_GRASP_INSTRUCTION_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r'^(?P<base>.+?)(?:\s*[,;]\s*|\s+-\s+)(?P<instruction>' + _INSTRUCTION + r')\s*\.?$',
    r'^(?P<base>.+?)\s*\((?P<instruction>' + _INSTRUCTION + r')\)\s*\.?$',
    r'^(?P<base>.+?)\s+(?P<instruction>' + _PLAIN_INSTRUCTION + r')\s*\.?$',
))

MAX_ALIASES_PER_OBJECT = 16


def normalize_name(name: str) -> str:
    """Normalize repeated, leading, and trailing whitespace."""
    if not isinstance(name, str):
        return ''
    return ' '.join(name.strip().split())


def parse_object_description(description: str):
    """
    Conservatively separate explicit trailing grasp instructions.

    Preserves meaningful descriptors (e.g. 'small', 'left', 'white').
    Returns (base_name, instruction_or_none).
    """
    clean = normalize_name(description)
    if not clean:
        return '', None
    for pattern in _GRASP_INSTRUCTION_PATTERNS:
        match = pattern.fullmatch(clean)
        if match:
            base = normalize_name(match.group('base'))
            instruction = normalize_name(match.group('instruction'))
            if base:
                return base, instruction
    return clean, None


def name_matches(target_name: str, candidate_name: str, aliases=None) -> bool:
    """Check if target_name matches candidate_name or its aliases."""
    t_clean = normalize_name(target_name)
    c_clean = normalize_name(candidate_name)
    if not t_clean or not c_clean:
        return False
    t_base, _ = parse_object_description(t_clean)
    c_base, _ = parse_object_description(c_clean)

    t_cf, c_cf = t_clean.casefold(), c_clean.casefold()
    t_base_cf, c_base_cf = t_base.casefold(), c_base.casefold()

    if t_cf == c_cf or t_base_cf == c_base_cf:
        return True
    if t_cf == c_base_cf or t_base_cf == c_cf:
        return True

    if aliases:
        for alias in aliases:
            a_clean = normalize_name(alias)
            a_base, _ = parse_object_description(a_clean)
            a_cf = a_clean.casefold()
            a_base_cf = a_base.casefold()
            if t_cf in (a_cf, a_base_cf) or t_base_cf in (a_cf, a_base_cf):
                return True
    return False


class WorldStateManager:
    """Own the small persistent state shared by orchestration components."""

    def __init__(self):
        """Initialize thread-safe world state store."""
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
        """Return an isolated copy suitable for readers and reasoning."""
        with self._lock:
            return copy.deepcopy(self._state)

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
            previous = self._state['robot']['holding']
            if previous != object_id and previous in self._state['objects']:
                record = self._state['objects'][previous]
                if record.get('role') == 'held':
                    # Empty hand is not evidence of a successful placement.
                    record['role'] = 'target'
                    record['visible'] = False
            self._state['robot']['holding'] = object_id
            if object_id is not None and object_id in self._state['objects']:
                self._state['objects'][object_id]['role'] = 'held'

    def is_held(self, object_id):
        """Return whether the object is currently held."""
        with self._lock:
            if self._state['robot']['holding'] == object_id:
                return True
            record = self._state['objects'].get(object_id)
            return record is not None and record.get('role') == 'held'

    def is_placed(self, object_id):
        """Return whether the object has been placed."""
        with self._lock:
            record = self._state['objects'].get(object_id)
            return record is not None and record.get('role') == 'placed'

    def get_aliases(self, object_id):
        """Return known aliases and descriptions for an object."""
        with self._lock:
            meta = self._observation_meta.get(object_id)
            if meta and 'aliases' in meta:
                return copy.deepcopy(meta['aliases'])
            record = self._state['objects'].get(object_id)
            if record and 'aliases' in record:
                return copy.deepcopy(record['aliases'])
            if record and 'name' in record:
                return [record['name']]
            return []

    def get_object_metadata(self, object_id):
        """Return metadata (aliases, misses, display_name) for an object."""
        with self._lock:
            meta = self._observation_meta.get(object_id)
            return copy.deepcopy(meta) if meta is not None else None

    def upsert_object(self, object_id, name, position, role, visible,
                      aliases=None, force_role=False):
        """Insert or replace a minimal object record in the world frame."""
        if not isinstance(object_id, str) or not object_id:
            raise ValueError('object_id must be a non-empty string')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('name must be a non-empty string')
        if not isinstance(role, str) or not role:
            raise ValueError('role must be a non-empty string')
        if len(position) != 3:
            raise ValueError('position must contain exactly three values')

        normalized_position = [float(value) for value in position]
        if not all(math.isfinite(value) for value in normalized_position):
            raise ValueError('position values must be finite')

        clean_name = normalize_name(name)
        base_name, instruction = parse_object_description(clean_name)

        with self._lock:
            existing = self._state['objects'].get(object_id)
            effective_role = role
            # Preserve held/placed state against ordinary observation updates
            if not force_role:
                if role not in ('held', 'placed') and existing is not None:
                    if existing.get('role') in ('held', 'placed'):
                        effective_role = existing.get('role')
                    elif self._state['robot']['holding'] == object_id:
                        effective_role = 'held'
                elif (self._state['robot']['holding'] == object_id and
                      role not in ('held', 'placed')):
                    effective_role = 'held'

            record = {
                'name': base_name or clean_name,
                'position': normalized_position,
                'role': effective_role,
                'visible': bool(visible),
            }

            meta = self._observation_meta.setdefault(object_id, {
                'misses': 0,
                'aliases': [],
                'display_name': base_name or clean_name,
                'last_instruction': None,
            })
            if instruction:
                meta['last_instruction'] = instruction

            all_aliases = []
            if existing and 'aliases' in existing:
                all_aliases.extend(existing['aliases'])
            all_aliases.extend(meta.get('aliases', []))
            if aliases:
                all_aliases.extend(aliases)
            all_aliases.append(clean_name)
            if base_name and base_name != clean_name:
                all_aliases.append(base_name)

            unique_aliases = []
            seen_cf = set()
            for a in all_aliases:
                cf = a.casefold()
                if cf not in seen_cf:
                    seen_cf.add(cf)
                    unique_aliases.append(a)
            if len(unique_aliases) > MAX_ALIASES_PER_OBJECT:
                unique_aliases = unique_aliases[-MAX_ALIASES_PER_OBJECT:]
            meta['aliases'] = unique_aliases

            if aliases is not None or (existing and 'aliases' in existing):
                record['aliases'] = list(unique_aliases)

            self._state['objects'][object_id] = record
            return copy.deepcopy(record)

    def get_object(self, object_id):
        """Return a copy of an object record, or None when it is unknown."""
        with self._lock:
            record = self._state['objects'].get(object_id)
            return copy.deepcopy(record) if record is not None else None

    def set_object_visibility(self, object_id, visible):
        """Update visibility without discarding the stored world position."""
        with self._lock:
            if object_id not in self._state['objects']:
                raise KeyError(object_id)
            self._state['objects'][object_id]['visible'] = bool(visible)

    def associate_observation(self, name, position, tolerance=0.05):
        """Associate detection by name and distance, or create stable id."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError('name must be a non-empty string')
        if len(position) != 3:
            raise ValueError('position must contain exactly three values')
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('association tolerance must be finite and nonnegative')
        normalized_position = [float(value) for value in position]
        if not all(math.isfinite(value) for value in normalized_position):
            raise ValueError('position values must be finite')

        clean_name = normalize_name(name)
        base_name, instruction = parse_object_description(clean_name)

        with self._lock:
            best_id = None
            best_distance = float('inf')
            for object_id, record in self._state['objects'].items():
                meta = self._observation_meta.get(object_id, {})
                aliases = meta.get('aliases', [])
                if not name_matches(clean_name, record['name'], aliases):
                    continue
                distance = math.dist(record['position'], normalized_position)
                if distance <= tolerance and distance < best_distance:
                    best_id = object_id
                    best_distance = distance

            if best_id is None:
                while f'obj_{self._next_object_index:03d}' in self._state['objects']:
                    self._next_object_index += 1
                best_id = f'obj_{self._next_object_index:03d}'
                self._next_object_index += 1
                canonical_name = base_name if base_name else clean_name
                aliases = [clean_name]
                if base_name and base_name != clean_name:
                    aliases.insert(0, base_name)
                self._observation_meta[best_id] = {
                    'misses': 0,
                    'aliases': aliases,
                    'display_name': canonical_name,
                    'last_instruction': instruction,
                }
                self.upsert_object(
                    best_id, canonical_name, normalized_position, 'target',
                    True, aliases=aliases)
            else:
                existing_record = self._state['objects'][best_id]
                existing_role = existing_record.get('role', 'target')
                if (self._state['robot']['holding'] == best_id or
                        existing_role in ('held', 'placed')):
                    target_role = (
                        existing_role if existing_role in ('held', 'placed')
                        else 'held')
                else:
                    target_role = 'target'

                current_name = existing_record.get('name', '')
                curr_base, _ = parse_object_description(current_name)
                canonical_name = (
                    curr_base if curr_base
                    else (base_name if base_name else clean_name))

                meta = self._observation_meta.setdefault(best_id, {
                    'misses': 0,
                    'aliases': [],
                    'display_name': canonical_name,
                    'last_instruction': None,
                })
                meta['misses'] = 0
                if instruction:
                    meta['last_instruction'] = instruction

                existing_aliases = meta.setdefault('aliases', [])
                for candidate in (clean_name, base_name):
                    if (candidate and not any(
                            candidate.casefold() == a.casefold()
                            for a in existing_aliases)):
                        existing_aliases.append(candidate)
                if len(existing_aliases) > MAX_ALIASES_PER_OBJECT:
                    meta['aliases'] = (
                        existing_aliases[-MAX_ALIASES_PER_OBJECT:])

                self.upsert_object(
                    best_id, canonical_name, normalized_position, target_role,
                    True, aliases=meta['aliases'])

            self._observation_meta[best_id]['misses'] = 0
            return best_id

    def objects_named(self, name):
        """Return copies of all records matching the requested name."""
        with self._lock:
            return [
                (object_id, copy.deepcopy(record))
                for object_id, record in self._state['objects'].items()
                if name_matches(
                    name, record['name'],
                    self._observation_meta.get(object_id, {}).get('aliases'))
            ]

    def record_expected_miss(self, name, removal_threshold=3):
        """Remove objects only after repeated misses from an expected view."""
        removed = []
        with self._lock:
            matching_ids = [
                object_id
                for object_id, record in self._state['objects'].items()
                if name_matches(
                    name, record['name'],
                    self._observation_meta.get(object_id, {}).get('aliases'))
            ]
            for object_id in matching_ids:
                is_held = (
                    self._state['robot']['holding'] == object_id or
                    self._state['objects'][object_id].get('role') == 'held')
                if is_held:
                    continue
                metadata = self._observation_meta.setdefault(
                    object_id, {'misses': 0})
                metadata['misses'] += 1
                self._state['objects'][object_id]['visible'] = False
                if metadata['misses'] >= removal_threshold:
                    del self._state['objects'][object_id]
                    if object_id in self._observation_meta:
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
