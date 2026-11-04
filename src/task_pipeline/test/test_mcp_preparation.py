"""Isolated typed StepStore fixtures: no ROS clients or actuator connection."""

import asyncio
import json

from my_actions import srv
from my_actions.action import RunPipelineSteps

import pytest

from sensor_msgs.msg import Image

from task_pipeline.context import StepStore
from task_pipeline.mcp.execution import PreparedExecution
from task_pipeline.mcp.preparation import PreparationGateway, result_summary


class FixtureTransport:
    """Run real StepStore dependency/revision logic against synthetic perception."""

    def __init__(self):
        self.calls = []
        self.store = StepStore(self.backend)
        self.cancelled = False

    def backend(self, name, query):
        self.calls.append(name)
        if name == 'capture_observation':
            return srv.CaptureObservation.Response(
                success=True, status='SIMULATED_CAPTURE', image=Image(
                    width=2, height=2, encoding='mono8', step=2, data=[
                        1, 2, 3, 4]))
        if name == 'ground_object':
            assert list(query.image.data) == [1, 2, 3, 4]
            return srv.GroundObject.Response(success=query.target != 'absent', status='SIMULATED',
                                             pixel_box=[0, 0, 2, 2])
        raise AssertionError('No physical/unknown backend allowed: ' + name)

    async def typed(self, name, query, timeout):
        if name == 'get_world_state':
            return srv.GetWorldState.Response(state_json=json.dumps(
                {'robot': {'state': 'IDLE', 'holding': None, 'pipeline_executing': False}}))
        if name == 'run_step':
            ctx, rev, item = self.store.run_one(
                query.context_id, query.step_id, query.function_name, query.request_cdr,
                query.expected_revision, query.bindings)
            return srv.RunPipelineStep.Response(success=item.success, status=item.status,
                                                context_id=ctx, revision=rev, result=item)
        if name == 'submit_steps':
            ctx, rev = self.store.submit(query.context_id, query.steps, query.expected_revision)
            return srv.SubmitPipelineSteps.Response(success=True, status='QUEUED',
                                                    context_id=ctx, revision=rev)
        if name == 'get_step_context':
            state, results = self.store.snapshot(query.context_id, query.include_results)
            return srv.GetStepContext.Response(success=True, state_json=state, results=results)
        raise AssertionError(name)

    async def start(self, ctx, revision, boundary):
        return (ctx, revision, boundary)

    async def finish(self, goal):
        await asyncio.sleep(.01)
        ctx, rev, boundary = goal
        status, results = self.store.run_queue(ctx, cancelled=lambda: self.cancelled,
                                               stop_before_step_id=boundary, expected_revision=rev)
        return RunPipelineSteps.Result(success=status in ('COMPLETE', 'PAUSED_AT_STEP_BOUNDARY'),
                                       status=status, context_id=ctx, results=results), (
                                           5 if self.cancelled else 4)

    async def cancel(self, goal):
        self.cancelled = True
        return True


def make_gateway(tmp_path):
    transport = FixtureTransport()
    execution = PreparedExecution(None, tmp_path / 'simulated.sqlite')
    return PreparationGateway(transport, execution), transport, execution


def query(step, context='', **arguments):
    return {
        'context_id': context,
        'step_id': step,
        'request_id': step,
        'arguments': arguments,
        'bindings': []}


def test_capture_bind_ground_and_duplicate(tmp_path):
    async def run():
        gateway, transport, _ = make_gateway(tmp_path)
        capture = await gateway.call('capture_observation', query('capture'))
        assert capture['success'], capture
        assert 'data' not in capture['output']['image']
        context = capture['context_id']
        ground = query('ground', context, target='bowl')
        ground['bindings'] = [{'target_field': 'image',
                               'source_step_id': 'capture', 'source_field': 'image'}]
        result = await gateway.call('ground_object', ground)
        assert result['success'], result
        assert result['output']['pixel_box'] == [0, 0, 2, 2]
        assert await gateway.call('ground_object', ground) == result
        assert transport.calls == ['capture_observation', 'ground_object']
        ground['arguments']['target'] = 'cup'
        assert (await gateway.call('ground_object', ground))['status'] == 'REQUEST_ID_REUSED'
        snapshot = await gateway.inspect(context, True)
        assert len(snapshot['results']) == 2
        assert '"response_cdr":' not in json.dumps(snapshot)
    asyncio.run(run())


@pytest.mark.parametrize('name', ['execute_plan', 'gripper_close', 'move_to', 'reset_target'])
def test_no_actuator_or_reset_through_generic_route(tmp_path, name):
    async def run():
        gateway, transport, _ = make_gateway(tmp_path)
        result = await gateway.call(name, query('blocked'))
        assert not result['success'] and 'NOT_ALLOWED' in result['status']
        assert not transport.calls
    asyncio.run(run())


def test_foreign_context_stale_revision_and_failed_binding(tmp_path):
    async def run():
        gateway, transport, execution = make_gateway(tmp_path)
        foreign = await gateway.call('capture_observation', query('foreign', 'other'))
        assert not foreign['success']
        cap = await gateway.call('capture_observation', query('capture'))
        stale = query('stale', cap['context_id'])
        stale['expected_revision'] = cap['revision'] + 1
        assert 'STALE' in (await gateway.call('capture_observation', stale))['status']
        missing = query('missing', cap['context_id'], target='bowl')
        missing['bindings'] = [{'target_field': 'image',
                                'source_step_id': 'other_context_step', 'source_field': 'image'}]
        assert 'DEPENDENCY_FAILED' in (await gateway.call('ground_object', missing))['status']
        execution.save('current', {'unknown': True})
        assert 'UNKNOWN_EXECUTION' in (await gateway.call('capture_observation',
                                                          query('uncertain')))['status']
        assert transport.calls == ['capture_observation']
    asyncio.run(run())


@pytest.mark.parametrize('cancel', [False, True])
def test_async_queue_completion_cancellation_and_duplicate(tmp_path, cancel):
    async def run():
        gateway, transport, _ = make_gateway(tmp_path)
        item = query('capture')
        item.pop('context_id')
        item.pop('request_id')
        item['function_name'] = 'capture_observation'
        submission = {'context_id': '', 'request_id': 'submit', 'steps': [item]}
        queued = await gateway.submit(submission)
        assert queued['success'] and queued['completion'] == 'QUEUED_ONLY', queued
        assert await gateway.submit(submission) == queued
        args = {'context_id': queued['context_id'], 'request_id': 'run',
                'operation': 'start', 'expected_revision': queued['revision']}
        ack = await gateway.run(args)
        assert ack['status'] == 'RUNNING'
        assert not transport.calls
        assert await gateway.run(args) == ack
        if cancel:
            result = await gateway.run(dict(args, operation='cancel'))
            assert result['status'] == 'CANCEL_REQUESTED'
        await asyncio.sleep(.03)
        final = await gateway.run(dict(args, operation='status'))
        assert final['completion'] == 'FINAL_ACTION_RESULT', final
        assert ('CANCELLED' in final['status']) == cancel
        assert not gateway.execution.lock.locked()
        assert len(transport.calls) == (0 if cancel else 1)
        reused = await gateway.run(dict(args, expected_revision=99))
        assert reused['status'] == 'REQUEST_ID_REUSED'
    asyncio.run(run())


def test_action_revision_checked_atomically(tmp_path):
    gateway, transport, _ = make_gateway(tmp_path)
    item = gateway.make_step('capture_observation', query('capture'))
    ctx, rev = transport.store.submit('synthetic', [item])
    with pytest.raises(ValueError, match='STALE_CONTEXT_REVISION'):
        transport.store.run_queue(ctx, expected_revision=rev + 1)
    assert not transport.calls


def test_restart_marks_active_goal_unknown(tmp_path):
    gateway, _, execution = make_gateway(tmp_path)
    execution.save('job:interrupted', {'success': True, 'status': 'RUNNING', 'context_id': 'mock'})
    reopened = PreparationGateway(None, execution)
    assert execution.load('job:interrupted')['status'] == 'ACTION_OUTCOME_UNKNOWN_AFTER_RESTART'
    assert not reopened.jobs


def test_binary_projection_and_collision_bypass_blocked(tmp_path):
    image = Image(width=2, height=2, encoding='mono8', step=2, data=[1, 2, 3, 4])
    assert 'data' not in result_summary(image, 'ctx', 'step')
    gateway, _, _ = make_gateway(tmp_path)
    with pytest.raises(ValueError, match='COLLISION_BYPASS_FORBIDDEN'):
        gateway.make_step('solve_pose', query('unsafe', ignore_collision=True))


def test_preparation_timeout_blocks_later_execution(tmp_path):
    async def run():
        gateway, transport, execution = make_gateway(tmp_path)

        async def timeout(*args):
            raise TimeoutError('SIMULATED_ROS_OUTCOME_UNKNOWN')
        transport.typed = timeout
        result = await gateway.call('capture_observation', query('timeout'))
        assert not result['success'] and execution.load('preparation_barrier')
        blocked = await execution.call('move_to', {'request_id': 'never_sent',
                                                   'target_joints': [0.] * 6})
        assert blocked['transport_outcome'] == 'rejected'
        assert 'PREPARATION_OUTCOME_UNKNOWN' in blocked['detail']
    asyncio.run(run())


def test_preparation_mode_requires_master_physical_gate_false(tmp_path):
    async def run():
        gateway, transport, _ = make_gateway(tmp_path)
        gateway.require_physical_disabled = True
        original = transport.typed

        async def enabled(name, query, timeout):
            if name == 'get_world_state':
                return srv.GetWorldState.Response(state_json=json.dumps({'robot': {
                    'state': 'IDLE', 'holding': None, 'pipeline_executing': False,
                    'physical_grasp_execution_enabled': True}}))
            return await original(name, query, timeout)

        transport.typed = enabled
        result = await gateway.call('capture_observation', query('must_not_dispatch'))
        assert result['status'] == 'PREPARATION_MODE_REQUIRES_PHYSICAL_GATE_DISABLED'
        assert not transport.calls
    asyncio.run(run())


def test_binding_validation_and_physical_queue_rejection(tmp_path):
    async def run():
        gateway, transport, _ = make_gateway(tmp_path)
        cap = await gateway.call('capture_observation', query('capture'))
        malformed = query('bad_field', cap['context_id'], target='bowl')
        malformed['bindings'] = [{'target_field': '__class__', 'source_step_id': 'capture',
                                   'source_field': 'image'}]
        result = await gateway.call('ground_object', malformed)
        assert not result['success'] and 'INVALID_BINDING' in result['status']
        wrong_type = query('bad_type', cap['context_id'], target='bowl')
        wrong_type['bindings'] = [{'target_field': 'image', 'source_step_id': 'capture',
                                  'source_field': 'status'}]
        assert not (await gateway.call('ground_object', wrong_type))['success']
        bad = {'step_id': 'move', 'function_name': 'move_to',
               'arguments': {'target_joints': [0.]*6}, 'bindings': []}
        result = await gateway.submit({'context_id': cap['context_id'],
                                       'request_id': 'bad_queue', 'steps': [bad]})
        assert not result['success'] and 'NOT_ALLOWED' in result['status']
        assert transport.calls == ['capture_observation']
    asyncio.run(run())


def test_queue_start_rechecks_catalog_after_local_queue_injection(tmp_path):
    async def run():
        from my_actions.msg import PipelineStep
        from rclpy.serialization import serialize_message
        gateway, transport, _ = make_gateway(tmp_path)
        cap = await gateway.call('capture_observation', query('capture'))
        injected = PipelineStep(step_id='external_physical', function_name='move_to',
            request_cdr=serialize_message(srv.MoveTo.Request(target_joints=[0.]*6)))
        ctx, revision = transport.store.submit(cap['context_id'], [injected])

        async def forbidden_start(*args):
            pytest.fail('An injected physical queue must never start')

        transport.start = forbidden_start
        result = await gateway.run({'operation': 'start', 'context_id': ctx,
                                    'request_id': 'blocked_start', 'expected_revision': revision})
        assert result['status'] == 'UNGUARDED_PHYSICAL_QUEUE_FORBIDDEN'
        assert transport.calls == ['capture_observation']
    asyncio.run(run())
