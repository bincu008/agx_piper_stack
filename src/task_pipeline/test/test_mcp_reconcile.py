"""Tests for administrative reconciliation of interrupted preparation calculations."""

import json
from pathlib import Path
import sqlite3
import pytest
from task_pipeline.mcp.reconcile import reconcile_ledger


def create_test_ledger(path):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE execution (id TEXT PRIMARY KEY, data TEXT)")
    conn.commit()
    conn.close()


def save_record(path, key, data):
    conn = sqlite3.connect(str(path))
    conn.execute("INSERT OR REPLACE INTO execution VALUES (?, ?)",
                 (key, json.dumps(data, allow_nan=False)))
    conn.commit()
    conn.close()


def load_record(path, key):
    conn = sqlite3.connect(str(path))
    row = conn.execute("SELECT data FROM execution WHERE id=?", (key,)).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def test_reconcile_without_operator_confirmation_raises(tmp_path):
    ledger_path = tmp_path / "mock-execution.sqlite"
    create_test_ledger(ledger_path)
    save_record(ledger_path, 'preparation_barrier', {'job': 'job:task_123_run'})

    with pytest.raises(ValueError, match="OPERATOR_CONFIRMATION_REQUIRED"):
        reconcile_ledger(ledger_path, task_id="task_123", confirmed_operator=False)


def test_reconcile_preserves_records_and_clears_barrier(tmp_path):
    ledger_path = tmp_path / "mock-execution.sqlite"
    create_test_ledger(ledger_path)

    job_id = "task_4c518ac7d9022810acb1039fe978d0cf_run"
    task_id = "task_4c518ac7d9022810acb1039fe978d0cf"
    context_id = "mcp_7a2a19cb419d4485bd8b4ef22efbf174"

    # Seed ledger with the interrupted state exactly matching the blocked mock task
    save_record(ledger_path, f'job:{job_id}', {
        'job_id': job_id, 'task_id': task_id, 'context_id': context_id,
        'status': 'ACTION_OUTCOME_UNKNOWN', 'success': False
    })
    save_record(ledger_path, f'task:{task_id}', {
        'task_id': task_id, 'context_id': context_id, 'state': 'UNKNOWN'
    })
    save_record(ledger_path, 'preparation_barrier', {'job': f'job:{job_id}', 'status': 'ACTION_OUTCOME_UNKNOWN'})

    assert load_record(ledger_path, 'preparation_barrier') is not None

    # Reconcile via reconcile_ledger
    result = reconcile_ledger(
        ledger_path, task_id=task_id, confirmed_operator=True,
        notes="Operator confirmed mock serialization crash aborted without motion"
    )

    assert result['reconciled'] is True
    assert result['barrier_cleared'] is True
    assert result['status'] == 'RECONCILIATION_COMPLETED'

    # Verify barrier is cleared in SQLite
    assert load_record(ledger_path, 'preparation_barrier') is None

    # Verify job record is preserved and marked reconciled
    job_data = load_record(ledger_path, f'job:{job_id}')
    assert job_data['reconciled'] is True
    assert job_data['status'] == 'ABORTED_BY_OPERATOR_RECONCILIATION'

    # Verify task record is preserved
    task_data = load_record(ledger_path, f'task:{task_id}')
    assert task_data['state'] == 'UNKNOWN'
    assert 'reconciled' not in task_data  # Historical task receipt is never rewritten.

    # Verify audit record was created
    audit = load_record(ledger_path, result['reconciliation_id'])
    assert audit['confirmed_by'] == 'operator'
    assert f'job:{job_id}' in audit['reconciled_records']


def test_reconcile_leaves_barrier_if_multiple_unknown_jobs_exist(tmp_path):
    ledger_path = tmp_path / "mock-execution.sqlite"
    create_test_ledger(ledger_path)

    save_record(ledger_path, 'job:task_1_run', {'status': 'ACTION_OUTCOME_UNKNOWN'})
    save_record(ledger_path, 'job:task_2_run', {'status': 'ACTION_OUTCOME_UNKNOWN'})
    save_record(ledger_path, 'preparation_barrier', {'job': 'job:task_1_run'})

    # Reconcile only task 1
    result = reconcile_ledger(ledger_path, task_id="task_1", confirmed_operator=True)
    assert result['reconciled'] is True
    assert result['barrier_cleared'] is False
    assert result['status'] == 'PARTIALLY_RECONCILED_BARRIER_RETAINS_OTHER_UNKNOWN_JOBS'

    # Barrier still exists because task_2 is unresolved
    assert load_record(ledger_path, 'preparation_barrier') is not None


def test_reconcile_when_no_barrier_returns_status(tmp_path):
    ledger_path = tmp_path / "mock-execution.sqlite"
    create_test_ledger(ledger_path)

    result = reconcile_ledger(ledger_path, task_id="task_1", confirmed_operator=True)
    assert result['reconciled'] is False
    assert result['status'] == 'NO_PREPARATION_BARRIER'


def test_gateway_reconcile_preparation(tmp_path):
    from task_pipeline.mcp.preparation import PreparationGateway

    class MockExecution:
        def __init__(self, p):
            self.db = sqlite3.connect(str(p))
            self.db.execute('CREATE TABLE execution (id TEXT PRIMARY KEY, data TEXT)')
            self.db.commit()
        def load(self, key):
            r = self.db.execute('SELECT data FROM execution WHERE id=?', (key,)).fetchone()
            return json.loads(r[0]) if r else None
        def save(self, key, val):
            self.db.execute('INSERT OR REPLACE INTO execution VALUES (?,?)',
                            (key, json.dumps(val, allow_nan=False)))
            self.db.commit()

    ledger_path = tmp_path / "gateway_exec.sqlite"
    mock_exec = MockExecution(ledger_path)
    mock_exec.save('job:task_4c518ac7d9022810acb1039fe978d0cf_run',
                   {'status': 'ACTION_OUTCOME_UNKNOWN', 'job_id': 'task_4c518ac7d9022810acb1039fe978d0cf_run'})
    mock_exec.save('preparation_barrier', {'job': 'job:task_4c518ac7d9022810acb1039fe978d0cf_run'})

    gw = PreparationGateway.__new__(PreparationGateway)
    gw.execution = mock_exec

    res = gw.reconcile_preparation(task_id='task_4c518ac7d9022810acb1039fe978d0cf', confirmed_operator=True)
    assert res['reconciled'] is True
    assert res['barrier_cleared'] is True
    assert mock_exec.load('preparation_barrier') is None



@pytest.mark.parametrize('status', ['RUNNING', 'STARTING', 'COMPLETE'])
def test_reconciliation_rejects_non_unknown_jobs(tmp_path, status):
    path = tmp_path / 'ledger.sqlite'
    create_test_ledger(path)
    save_record(path, 'job:task_1_run', {'status': status})
    save_record(path, 'preparation_barrier', {'job': 'task_1_run'})
    with pytest.raises(ValueError):
        reconcile_ledger(path, task_id='task_1', confirmed_operator=True)
    assert load_record(path, 'preparation_barrier') == {'job': 'task_1_run'}


def test_reconciliation_rejects_wrong_task_and_preserves_physical_uncertainty(tmp_path):
    path = tmp_path / 'ledger.sqlite'
    create_test_ledger(path)
    save_record(path, 'job:task_1_run', {'status': 'ACTION_OUTCOME_UNKNOWN: timeout'})
    save_record(path, 'preparation_barrier', {'job': 'task_1_run'})
    with pytest.raises(ValueError, match='MATCHING_UNKNOWN_PREPARATION_REQUIRED'):
        reconcile_ledger(path, task_id='other', confirmed_operator=True)
    save_record(path, 'current', {'unknown': True})
    with pytest.raises(ValueError, match='PHYSICAL_EXECUTION'):
        reconcile_ledger(path, task_id='task_1', confirmed_operator=True)
    assert load_record(path, 'current') == {'unknown': True}
    assert load_record(path, 'preparation_barrier') is not None


def test_reconciliation_uses_same_exclusive_ledger_lock(tmp_path):
    import fcntl
    path = tmp_path / 'ledger.sqlite'
    create_test_ledger(path)
    with open(str(path) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='STOP_MCP_BEFORE_LOCAL_RECONCILIATION'):
            reconcile_ledger(path, confirmed_operator=True)


def test_reconciliation_preserves_unfinished_request_barrier(tmp_path):
    path = tmp_path / 'ledger.sqlite'
    create_test_ledger(path)
    save_record(path, 'job:task_1_run', {'status': 'ACTION_OUTCOME_UNKNOWN: timeout'})
    save_record(path, 'preparation_barrier', {'job': 'task_1_run'})
    save_record(path, 'request:unfinished', {'context_id': 'other_context'})
    result = reconcile_ledger(path, task_id='task_1', confirmed_operator=True)
    assert not result['barrier_cleared']
    assert load_record(path, 'preparation_barrier') is not None
    assert load_record(path, 'request:unfinished') == {'context_id': 'other_context'}
