"""
Administrative CLI to reconcile an interrupted non-actuating preparation calculation.

Never resets physical actuator uncertainty. Preserves all historical receipts.
Requires explicit operator confirmation.
"""

import argparse
import json
from pathlib import Path
import sqlite3
import time
import uuid


def reconcile_connection(conn, *, task_id=None, job_id=None, context_id=None,
                         confirmed_operator=False, notes=''):
    """Reconcile exact interrupted calculation records atomically; never running jobs."""
    if not confirmed_operator:
        raise ValueError('OPERATOR_CONFIRMATION_REQUIRED')
    with conn:
        conn.execute('BEGIN IMMEDIATE')
        records = {key: json.loads(data) for key, data in conn.execute(
            'SELECT id, data FROM execution')}
        barrier = records.get('preparation_barrier')
        if not barrier:
            return {'reconciled': False, 'status': 'NO_PREPARATION_BARRIER'}
        physical = records.get('current')
        if physical and (physical.get('unknown') or not physical.get('complete')):
            raise ValueError('ACTIVE_OR_UNKNOWN_PHYSICAL_EXECUTION_REQUIRES_LOCAL_RECONCILIATION')
        jobs = {key: value for key, value in records.items() if key.startswith('job:')}
        if any(value.get('status') in ('STARTING', 'RUNNING') for value in jobs.values()):
            raise ValueError('RUNNING_PREPARATION_CANNOT_BE_RECONCILED')
        barrier_key = barrier.get('job')
        if barrier_key and not barrier_key.startswith('job:'):
            barrier_key = 'job:' + barrier_key
        selected = {}
        for key, value in jobs.items():
            if task_id and key != 'job:' + task_id + '_run':
                continue
            if job_id and key != (job_id if job_id.startswith('job:') else 'job:' + job_id):
                continue
            if context_id and value.get('context_id') != context_id:
                continue
            if barrier_key and key != barrier_key:
                continue
            if barrier.get('context_id') and value.get('context_id') != barrier['context_id']:
                continue
            if str(value.get('status', '')).startswith('ACTION_OUTCOME_UNKNOWN'):
                selected[key] = value
        if not selected:
            raise ValueError('MATCHING_UNKNOWN_PREPARATION_REQUIRED_BARRIER_PRESERVED')
        now = time.time()
        rec_id = f'reconciliation:{int(now)}_{uuid.uuid4().hex[:8]}'
        for key, value in selected.items():
            value.update(reconciled=True, reconciled_at=now,
                         reconciliation_notes=notes,
                         prior_status=value['status'],
                         success=False, status='ABORTED_BY_OPERATOR_RECONCILIATION')
            conn.execute('UPDATE execution SET data=? WHERE id=?',
                         (json.dumps(value, allow_nan=False), key))
        unresolved = any(str(value.get('status', '')).startswith('ACTION_OUTCOME_UNKNOWN')
                         for value in jobs.values())
        unresolved |= any(key.startswith('request:') and 'context_id' in value
                          and 'result' not in value for key, value in records.items())
        # Request-only or unassociated barriers cannot be resolved by guessing a job.
        associated = bool(barrier_key or barrier.get('context_id')) and not barrier.get('request')
        cleared = associated and not unresolved
        if cleared:
            conn.execute("DELETE FROM execution WHERE id='preparation_barrier'")
        audit = {'reconciliation_id': rec_id, 'timestamp': now, 'confirmed_by': 'operator',
                 'reconciled_records': sorted(selected), 'prior_barrier': barrier, 'notes': notes}
        conn.execute('INSERT INTO execution VALUES (?,?)',
                     (rec_id, json.dumps(audit, allow_nan=False)))
        return {'reconciled': True, 'barrier_cleared': cleared,
                'reconciled_records': sorted(selected), 'reconciliation_id': rec_id,
                'status': 'RECONCILIATION_COMPLETED' if cleared else
                          'PARTIALLY_RECONCILED_BARRIER_RETAINS_OTHER_UNKNOWN_JOBS'}


def reconcile_ledger(ledger_path, **kwargs):
    """Offline-only entrypoint; share the server's exclusive ledger admission lock."""
    import fcntl
    path = Path(ledger_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Ledger not found: {path}')
    with open(str(path) + '.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('STOP_MCP_BEFORE_LOCAL_RECONCILIATION') from error
        conn = sqlite3.connect(str(path))
        try:
            return reconcile_connection(conn, **kwargs)
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', required=True, help='Path to execution.sqlite')
    parser.add_argument('--task', help='Task ID to reconcile (e.g. task_4c518ac7d9022810acb1039fe978d0cf)')
    parser.add_argument('--job', help='Job ID to reconcile')
    parser.add_argument('--context', help='Context ID to reconcile')
    parser.add_argument('--notes', default='', help='Operator reconciliation reason')
    parser.add_argument('--confirm-operator', action='store_true',
                        help='Explicit operator confirmation that this was a non-actuating calculation')
    args = parser.parse_args()

    result = reconcile_ledger(
        args.ledger, task_id=args.task, job_id=args.job, context_id=args.context,
        confirmed_operator=args.confirm_operator, notes=args.notes)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
