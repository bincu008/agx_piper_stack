"""Durable transport admission for existing master-owned prepared stages.

This module plans nothing and never assumes contact from a command. The master
remains authoritative for phases, feedback, contact, limits and driver commands.
"""

import asyncio
import fcntl
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path

from .contracts import MANIPULATION


def reference(object_name, state):
    """Bind remote requests to the exact master's plan and source observation."""
    identity = [object_name, state['plan_revision'], state['observation_header']]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def outcome(kind, detail, response=None, dispatched=False):
    """Keep transport admission distinct from raw ROS success/completion."""
    return {'transport_outcome': kind, 'dispatched': dispatched,
            'response': response, 'detail': detail}


class PreparedExecution:
    """Serialize, correlate and deduplicate actions without a second robot queue."""

    def __init__(self, transport, path):
        """Open one private durable ledger; prevent concurrent gateway processes."""
        self.transport = transport
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.file_lock = open(str(path)+'.lock', 'a')
        os.chmod(str(path)+'.lock', 0o600)
        fcntl.flock(self.file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(path, check_same_thread=False)
        os.chmod(path, 0o600)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS execution (id TEXT PRIMARY KEY, data TEXT)')
        self.lock = asyncio.Lock()
        current = self.load('current')
        if current and current.get('running'):
            current.update(unknown=True, running=False)
            self.save('current', current)

    def load(self, key):
        """Read a persisted correlation or final operation response."""
        row = self.db.execute('SELECT data FROM execution WHERE id=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, key, value):
        """Commit admission before dispatch; never rely on client connection state."""
        self.db.execute('INSERT OR REPLACE INTO execution VALUES (?,?)',
                        (key, json.dumps(value, allow_nan=False)))
        self.db.commit()

    async def call(self, name, arguments):
        """Execute at most once, refusing ambiguity and unsupported recovery."""
        if self.lock.locked():
            return outcome('rejected', 'MCP_EXECUTION_BUSY; no request dispatched')
        async with self.lock:
            if self.load('preparation_barrier'):
                return outcome('rejected', 'PREPARATION_OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION')
            if name == 'move_to':
                return await self._move(arguments)
            return await self._call(name, arguments)

    async def _move(self, arguments):
        """Guard a joint-target move with the persistent uncertainty barrier."""
        key = 'request:'+arguments['request_id']
        fingerprint = json.dumps(['move_to', arguments], sort_keys=True, allow_nan=False)
        dispatched = False
        current = self.load('current')
        try:
            previous = self.load(key)
            if previous:
                if previous['fingerprint'] != fingerprint:
                    raise ValueError('REQUEST_ID_REUSED_WITH_DIFFERENT_ARGUMENTS')
                result = previous.get('result') or outcome('unknown', 'MOVE_OUTCOME_UNKNOWN')
                return dict(result, dispatched=False,
                            detail='CACHED_RECEIPT_NOT_CURRENT_READINESS: '+result['detail'])
            if current and (current.get('unknown') or not current.get('complete')):
                raise ValueError('OTHER_OR_UNKNOWN_EXECUTION_SESSION_OWNS_ARM')
            if self.db.execute('SELECT COUNT(*) FROM execution').fetchone()[0] >= 10000:
                raise ValueError('EXECUTION_HISTORY_CAPACITY_REQUIRES_LOCAL_ARCHIVE')
            target = arguments['target_joints']
            if len(target) != 6 or not all(math.isfinite(value) for value in target):
                raise ValueError('SIX_FINITE_JOINTS_REQUIRED')
            world = await self.transport.request('get_world_state', {}, 5.)
            robot = json.loads(world['state_json'])['robot']
            if not robot.get('physical_grasp_execution_enabled'):
                raise ValueError('PHYSICAL_EXECUTION_DISABLED_BY_MASTER')
            if (robot.get('state') != 'IDLE' or robot.get('holding') is not None or
                    robot.get('pipeline_executing')):
                raise ValueError('MOVE_REQUIRES_IDLE_EMPTY_HAND')
            state = await self.transport.request('get_arm_state', {}, 5.)
            if not state['success']:
                raise ValueError('ARM_STATE_UNAVAILABLE')
            stamp = state['joints']['header']['stamp']
            age = self.transport.now() - (stamp['sec']+stamp['nanosec']*1.e-9)
            if not 0. <= age <= 1.:
                raise ValueError('STALE_ARM_STATE')
            current = {'task_id': 'move_to:'+arguments['request_id'], 'running': True,
                       'complete': False, 'unknown': False, 'selected_pose': None}
            self.save('current', current)
            self.save(key, {'fingerprint': fingerprint})
            dispatched = True
            result = await self.transport.request('move_to', {'target_joints': target}, 300.)
            if not result.get('success'):
                raise RuntimeError('MOVE_COMPLETION_NOT_CONFIRMED: '+result.get('status', ''))
            post = await self.transport.request('get_arm_state', {}, 5.)
            positions = post['joints']['position'][:6]
            stamp = post['joints']['header']['stamp']
            age = self.transport.now() - (stamp['sec']+stamp['nanosec']*1.e-9)
            if (not post['success'] or len(positions) != 6 or
                    not 0. <= age <= 1. or
                    not all(math.isfinite(value) for value in positions) or
                    max(abs(a-b) for a, b in zip(positions, target)) > .03):
                raise RuntimeError('MOVE_TARGET_FEEDBACK_NOT_CONFIRMED')
            current.update(running=False, complete=True)
            self.save('current', current)
            response = outcome('response', 'Master confirmed joint target reached', result, True)
            self.save(key, {'fingerprint': fingerprint, 'result': response})
            return response
        except asyncio.CancelledError:
            if dispatched:
                current.update(running=False, unknown=True)
                self.save('current', current)
            raise
        except Exception as error:
            response = outcome('unknown' if dispatched else 'rejected', str(error),
                               dispatched=dispatched)
            if dispatched:
                current.update(running=False, unknown=True)
                self.save('current', current)
                self.save(key, {'fingerprint': fingerprint, 'result': response})
            return response

    async def _call(self, name, arguments):
        current = None
        dispatched = False
        request_key = 'request:'+arguments['request_id']
        fingerprint = json.dumps([name, arguments], sort_keys=True, allow_nan=False)
        try:
            previous = self.load(request_key)
            if previous:
                if previous['fingerprint'] != fingerprint:
                    raise ValueError('REQUEST_ID_REUSED_WITH_DIFFERENT_ARGUMENTS')
                result = previous.get('result') or outcome(
                    'unknown', 'REQUEST_OUTCOME_UNKNOWN; never replay this action')
                return dict(result, dispatched=False,
                            detail='CACHED_RECEIPT_NOT_CURRENT_READINESS: '+result['detail'])
            ref = arguments['execution_ref']
            current = self.load('current')
            if current and current.get('unknown'):
                raise ValueError('MCP_STATE_UNKNOWN_REQUIRES_LOCAL_RECONCILIATION')
            if self.db.execute('SELECT COUNT(*) FROM execution').fetchone()[0] >= 10000:
                raise ValueError('EXECUTION_HISTORY_CAPACITY_REQUIRES_LOCAL_ARCHIVE')
            state = await self.transport.request('get_execution_state',
                                                 {'object_name': ref['object_name']}, 5.)
            if (not state['success'] or state['phase'] == 'STATE_UNKNOWN' or
                    'STATE_UNKNOWN' in state['status']):
                raise ValueError('MASTER_SESSION_NOT_READY: '+state['status'])
            if (ref['task_id'] != reference(ref['object_name'], state) or
                    ref['plan_revision'] != state['plan_revision'] or ref['attempt'] != 1):
                raise ValueError('TASK_OBJECT_ATTEMPT_OR_SOURCE_MISMATCH')
            if not state['physical_execution_enabled']:
                raise ValueError('PHYSICAL_EXECUTION_DISABLED_BY_MASTER')
            if state['expires_in_seconds'] <= 0:
                raise ValueError('PREPARED_PLAN_OR_EXECUTION_SESSION_EXPIRED')
            if current and current['task_id'] != ref['task_id'] and not current.get('complete'):
                raise ValueError('OTHER_EXECUTION_SESSION_OWNS_ARM')
            query = {key: value for key, value in arguments.items()
                     if key not in ('execution_ref', 'request_id')}
            if name != 'commit_floor_slots' and query['plan_revision'] != state['plan_revision']:
                raise ValueError('PLAN_REVISION_MISMATCH')
            if (name not in ('commit_floor_slots', 'execute_withdrawal',
                             'execute_placement_retreat') and
                    query['object_name'] != ref['object_name']):
                raise ValueError('OBJECT_MISMATCH')
            if not current or current['task_id'] != ref['task_id']:
                if state['phase'] != 'PREPARED' or state['completed_stages']:
                    raise ValueError('MCP_MUST_OWN_EXECUTION_FROM_PREPARED_START')
                # This is a handoff check, not planning or actuation. It captures
                # the source-associated selected pose before any physical stage.
                checked = await self.transport.request('validate_prepared_pick_place',
                                                       {'object_name': ref['object_name']}, 15.)
                if (not checked['success'] or
                        checked['plan_revision'] != state['plan_revision']):
                    raise ValueError('COMPLETE_PICK_PLACE_HANDOFF_REQUIRED: '+checked['status'])
                current = dict(ref, unknown=False, running=False, complete=False,
                               verified=False, closes=0, operations=[],
                               selected_pose=checked['selected_pose'])
            stage = self.stage(name, query)
            if stage in current['operations']:
                raise ValueError('STAGE_ALREADY_COMPLETED; use the original request_id')
            if stage == 'commit_slots':
                if (state['phase'] != 'RELEASED' or 'release' not in current['operations'] or
                        query['reset'] or query['validate_only'] or
                        query['selected_pose'] != current['selected_pose']):
                    raise ValueError('COMMIT_REQUIRES_CONFIRMED_RELEASE_AND_FROZEN_POSE')
            elif not state['waiting_stages'] or state['waiting_stages'][0] != stage:
                raise ValueError('WRONG_EXECUTION_ORDER')
            if stage in ('lift', 'transfer', 'descent', 'release') and not current['verified']:
                raise ValueError('CURRENT_ATTEMPT_CONTACT_NOT_VERIFIED')
            if stage == 'withdrawal' and 'commit_slots' not in current['operations']:
                raise ValueError('WITHDRAWAL_REQUIRES_SLOT_COMMIT')
            if stage == 'gripper_close':
                if current['closes'] >= 3:
                    raise ValueError('GRASP_ATTEMPTS_EXHAUSTED')
                current['closes'] += 1
            # First persist the in-flight marker. Cancellation, timeout or crash
            # after this point conservatively blocks the session, never retries.
            current['running'] = True
            self.save('current', current)
            self.save(request_key, {'fingerprint': fingerprint})
            dispatched = True
            response = await self.transport.request(name, query, 300.)
            if (not response.get('success') or 'STATE_UNKNOWN' in response.get('status', '') or
                    (stage not in ('commit_slots', 'gripper_open', 'gripper_close') and
                     not response.get('motion_completed')) or
                    (stage == 'gripper_close' and not response.get('contact_confirmed')) or
                    (stage == 'release' and not response.get('gripper_released'))):
                current['unknown'] = True
                result = outcome('unknown', 'EXECUTION_NOT_CONFIRMED; local reconciliation '
                                 'required, no actuator retry', response, True)
            else:
                post = await self.transport.request('get_execution_state',
                                                    {'object_name': ref['object_name']}, 5.)
                if (not post['success'] or post['plan_revision'] != ref['plan_revision'] or
                        reference(ref['object_name'], post) != ref['task_id'] or
                        post['phase'] == 'STATE_UNKNOWN' or
                        (stage != 'commit_slots' and stage not in post['completed_stages'])):
                    raise RuntimeError('POST_EXECUTION_STATE_NOT_CONFIRMED')
                current['operations'].append(stage)
                if stage == 'gripper_close':
                    current['verified'] = True
                if stage == 'retreat':
                    current['complete'] = post['phase'] == 'COMPLETE'
                    if not current['complete']:
                        raise RuntimeError('EMPTY_HAND_COMPLETION_NOT_CONFIRMED')
                result = outcome('response', 'Master-confirmed stage completion', response, True)
            current['running'] = False
            self.save('current', current)
            self.save(request_key, {'fingerprint': fingerprint, 'result': result})
            return result
        except asyncio.CancelledError:
            if dispatched:
                current.update(unknown=True, running=False)
                self.save('current', current)
            raise
        except Exception as error:
            if dispatched:
                current.update(unknown=True, running=False)
                self.save('current', current)
                result = outcome('unknown', str(error)+'; never replay the action', dispatched=True)
                self.save(request_key, {'fingerprint': fingerprint, 'result': result})
                return result
            return outcome('rejected', str(error))

    @staticmethod
    def stage(name, query):
        """Map public operations to master-declared boundaries, not waypoints."""
        if name == 'execute_plan':
            return query['segment']
        return {'gripper_open': 'gripper_open', 'gripper_close': 'gripper_close',
                'execute_placement_transfer': 'transfer',
                'execute_placement_descent': 'descent', 'release_placed_object': 'release',
                'commit_floor_slots': 'commit_slots', 'execute_withdrawal': 'withdrawal',
                'execute_placement_retreat': 'retreat'}[name]


class RosExecutionTransport:
    """Construct only fixed catalog clients; rely on the master's physical guard."""

    def __init__(self, node):
        """Keep actuator clients separate from the read-only bridge."""
        from task_pipeline.registry import FUNCTIONS
        self.node = node
        names = set(MANIPULATION) | {'get_execution_state', 'validate_prepared_pick_place',
                                     'get_arm_state', 'get_world_state'}
        self.clients = {name: node.create_client(FUNCTIONS[name].service_type,
                                                 FUNCTIONS[name].endpoint) for name in names}

    def now(self):
        """Use ROS time for feedback age, matching the existing master."""
        return self.node.get_clock().now().nanoseconds*1.e-9

    async def request(self, name, arguments, timeout):
        """Wait for final ROS completion; local cancellation cannot stop services."""
        return await asyncio.to_thread(self._request, name, arguments, timeout)

    def _request(self, name, arguments, timeout):
        import threading
        from rosidl_runtime_py.convert import message_to_ordereddict
        from rosidl_runtime_py.set_message import set_message_fields
        from task_pipeline.registry import FUNCTIONS
        client = self.clients[name]
        if not client.wait_for_service(timeout_sec=1.):
            raise RuntimeError('ROS_SERVICE_UNAVAILABLE: '+name)
        query = FUNCTIONS[name].service_type.Request()
        set_message_fields(query, arguments)
        future = client.call_async(query)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            future.cancel()  # Cancels waiting locally, not the ROS command.
            raise TimeoutError('ROS_SERVICE_OUTCOME_UNKNOWN: '+name)
        return dict(message_to_ordereddict(future.result()))
