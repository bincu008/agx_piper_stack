"""Durable admission policy tested in isolation; NOT a robot execution adapter.

No ROS imports, callbacks or dispatch. Live admission uses execution.py instead.
Attempt/verification identifiers are adapter identities, not new ROS API semantics.
They cannot be populated safely from current ROS interfaces automatically.
"""

import json
import sqlite3


class AdmissionLedger:
    """Persist attempts and ambiguity across client reconnects/process restarts."""

    def __init__(self, path):
        """Initialize isolated state."""
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, data TEXT)')
        # Interrupted dispatch cannot be assumed complete or safe to replay.
        for key, raw in self.db.execute('SELECT id,data FROM tasks').fetchall():
            value = json.loads(raw)
            if value['outcome'] == 'running':
                value['outcome'] = 'unknown'
                self._save(key, value)

    def _save(self, task, value):
        self.db.execute('INSERT OR REPLACE INTO tasks VALUES (?,?)',
                        (task, json.dumps(value)))

    def create(self, task, object_name):
        """Retain an existing task identity."""
        try:
            self.db.execute('INSERT INTO tasks VALUES (?,?)', (task, json.dumps({
                'object': object_name, 'attempt': 0, 'revision': 0, 'verified': False,
                'outcome': 'idle', 'requests': {}, 'alternatives': []})))
        except sqlite3.IntegrityError:
            if self.read(task)['object'] != object_name:
                raise ValueError('TASK_OBJECT_MISMATCH')

    def read(self, task):
        """Read the durable task state."""
        row = self.db.execute('SELECT data FROM tasks WHERE id=?', (task,)).fetchone()
        if not row:
            raise ValueError('TASK_UNKNOWN')
        return json.loads(row[0])

    def begin(self, task, object_name, request_id, revision, *, state_ready,
              candidate, alternative=False):
        """Reserve one grasp attempt before dispatch; never dispatch here."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            value = self.read(task)
            fingerprint = [object_name, revision, candidate, alternative]
            previous = value['requests'].get(request_id)
            if previous:
                if previous['fingerprint'] != fingerprint:
                    raise ValueError('REQUEST_ID_REUSED')
                return {'dispatch': False, 'attempt': previous['attempt'],
                        'outcome': previous['outcome']}
            if value['object'] != object_name:
                raise ValueError('TASK_OBJECT_MISMATCH')
            if not state_ready or value['outcome'] in ('unknown', 'running', 'cancelled'):
                raise ValueError('CONTINUATION_NOT_ESTABLISHED')
            if value['attempt'] >= 3:
                raise ValueError('GRASP_ATTEMPTS_EXHAUSTED')
            if value['attempt'] and (not alternative or value['outcome'] != 'recoverable_failure'
                                     or candidate not in value['alternatives']):
                raise ValueError('SUPPORTED_RECOVERY_REQUIRED')
            if revision <= value['revision']:
                raise ValueError('STALE_PLAN_REVISION')
            value.update(attempt=value['attempt']+1, revision=revision, verified=False,
                         outcome='running', alternatives=[])
            value['requests'][request_id] = {'fingerprint': fingerprint,
                                             'attempt': value['attempt'], 'outcome': 'running'}
            self._save(task, value)
            return {'dispatch': True, 'attempt': value['attempt'], 'outcome': 'running'}
        finally:
            self.db.execute('COMMIT')

    def finish(self, task, object_name, request_id, attempt, revision, outcome,
               *, alternatives=()):
        """Accept only an exact final correlation; acknowledgment is not success."""
        if outcome not in ('running', 'success', 'recoverable_failure', 'failure',
                           'cancelled', 'unknown'):
            raise ValueError('UNDEFINED_OUTCOME')
        value = self.read(task)
        request = value['requests'].get(request_id)
        if (value['object'] != object_name or value['attempt'] != attempt or
                value['revision'] != revision or not request or request['attempt'] != attempt):
            raise ValueError('STALE_OR_MISMATCHED_RESULT')
        if value['outcome'] != 'running':
            raise ValueError('OUTCOME_REQUIRES_OPERATOR_RECONCILIATION')
        value['outcome'] = request['outcome'] = outcome
        value['alternatives'] = list(alternatives) if outcome == 'recoverable_failure' else []
        self._save(task, value)

    def verify(self, task, object_name, attempt, revision, *, success, contact):
        """Only the exact current successful attempt can receive verification."""
        value = self.read(task)
        if (value['object'] != object_name or value['attempt'] != attempt or
                value['revision'] != revision or value['outcome'] != 'success' or
                not success or not contact):
            raise ValueError('CURRENT_GRASP_NOT_VERIFIED')
        value['verified'] = True
        self._save(task, value)

    def placement_allowed(self, task, object_name, attempt, revision, *, state_ready):
        """Require current verified contact before placement."""
        value = self.read(task)
        if (value['object'] != object_name or value['attempt'] != attempt or
                value['revision'] != revision or not state_ready or
                not value['verified'] or value['outcome'] != 'success'):
            raise ValueError('CURRENT_GRASP_NOT_VERIFIED')
        return True

    def interrupted(self, task):
        """Transport loss/timeout blocks continuation; not a robot stop command."""
        value = self.read(task)
        value['outcome'], value['verified'] = 'unknown', False
        for request in value['requests'].values():
            if request['outcome'] == 'running':
                request['outcome'] = 'unknown'
        self._save(task, value)
