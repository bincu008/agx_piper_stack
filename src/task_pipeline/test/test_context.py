"""Verify typed handoffs, queue atomicity, revisions and bounded result memory."""

import json

from my_actions.srv import (
    CacheGraspRefinement, CaptureObservation, GetObstacleLocation, GroundObject,
    RefineGrasp, SegmentGroundedImage,
)

import pytest

from rclpy.serialization import serialize_message

from task_pipeline.context import StepStore, fields
from task_pipeline.recipes import step
from my_actions.msg import StepBinding


def test_queue_binds_exact_observation_and_depletes_successful_steps():
    """Deliver the source image and box without consulting newer camera frames."""
    observed = CaptureObservation.Response(success=True)
    observed.image.width, observed.image.height = 640, 480
    observed.image.header.stamp.sec = 123
    observed.cloud.header.stamp.sec = 123
    observed.cloud.header.frame_id = 'base_link'
    calls = []

    def backend(name, request):
        calls.append(name)
        if name == 'capture_observation':
            return observed
        if name == 'ground_object':
            assert request.image.header.stamp.sec == 123
            return GroundObject.Response(success=True, pixel_box=[10, 20, 50, 60])
        assert request.image.header.stamp.sec == 123
        assert list(request.bbox) == [10, 20, 50, 60]
        return SegmentGroundedImage.Response(success=True)

    store = StepStore(backend)
    context, revision = store.submit('', [
        step('observe', 'capture_observation', CaptureObservation.Request()),
        step('ground', 'ground_object', GroundObject.Request(target='cup'),
             [('image', 'observe', 'image')]),
        step('mask', 'segment_object', SegmentGroundedImage.Request(),
             [('image', 'observe', 'image'), ('bbox', 'ground', 'pixel_box')]),
    ])
    status, results = store.run_queue(context)
    assert status == 'COMPLETE' and len(results) == 3
    assert calls == ['capture_observation', 'ground_object', 'segment_object']
    snapshot = json.loads(store.snapshot(context)[0])[0]
    assert snapshot['pending'] == [] and snapshot['revision'] > revision
    assert store.snapshot(context)[1] == []


def test_atomic_append_rejects_forward_dependencies_without_mutating_queue():
    """Validate all dependencies before accepting any step in the batch."""
    store = StepStore(lambda *args: None)
    invalid = step('mask', 'segment_object', SegmentGroundedImage.Request(),
                   [('image', 'future', 'image')])
    with pytest.raises(ValueError, match='MISSING_OR_FORWARD_DEPENDENCY'):
        store.submit('test', [invalid])
    assert json.loads(store.snapshot('test')[0])[0]['pending'] == []


def test_failed_dependency_skips_backend_and_continues_independent_object():
    """A failed object cannot supply inputs to its dependent placement."""
    calls = []

    def backend(name, request):
        calls.append(request.target)
        return GroundObject.Response(success=request.target == 'plate')

    store = StepStore(backend)
    context, _ = store.submit('', [
        step('cup', 'ground_object', GroundObject.Request(target='cup'),
             failure_policy='continue'),
        step('cup_mask', 'segment_object', SegmentGroundedImage.Request(),
             [('bbox', 'cup', 'pixel_box')], failure_policy='continue'),
        step('plate', 'ground_object', GroundObject.Request(target='plate')),
    ])
    status, results = store.run_queue(context)
    assert status == 'PARTIAL' and calls == ['cup', 'plate']
    assert results[1].status == 'DEPENDENCY_FAILED: cup'
    assert results[2].success


def test_stop_policy_leaves_later_tasks_pending():
    """Stopping on a failure preserves later tasks for explicit controller decisions."""
    store = StepStore(lambda *args: GroundObject.Response(success=False))
    context, _ = store.submit('', [
        step('cup', 'ground_object', GroundObject.Request()),
        step('plate', 'ground_object', GroundObject.Request()),
    ])
    status, results = store.run_queue(context)
    assert status == 'STOPPED_ON_FAILURE' and len(results) == 1
    assert json.loads(store.snapshot(context)[0])[0]['pending'][0]['step_id'] == 'plate'


def test_step_id_is_idempotent_and_conflicting_payload_is_rejected():
    """Do not execute a motion twice after a caller retries its request ID."""
    calls = []
    store = StepStore(lambda *args: (calls.append(args), GroundObject.Response(success=True))[1])
    payload = serialize_message(GroundObject.Request(target='cup'))
    context, revision, result = store.run_one('test', 'one', 'ground_object', payload)
    _, _, duplicate = store.run_one(context, 'one', 'ground_object', payload)
    assert len(calls) == 1 and duplicate == result
    with pytest.raises(ValueError, match='STEP_ID_PAYLOAD_CONFLICT'):
        store.run_one(context, 'one', 'ground_object',
                      serialize_message(GroundObject.Request(target='tray')))
    with pytest.raises(ValueError, match='STALE_CONTEXT_REVISION'):
        store.run_one(context, 'two', 'ground_object', payload, expected=revision - 1)


def test_bound_call_reuses_exact_image_and_keeps_stored_data_immutable():
    """Sparse calls bind stored observations without resending pixels."""
    observed = CaptureObservation.Response(success=True)
    observed.image.header.stamp.sec = 123
    observed.image.data = bytes([1, 2, 3])
    calls = []

    def backend(name, request):
        assert request.image.header.stamp.sec == 123
        assert list(request.image.data) == [1, 2, 3]
        calls.append(name)
        request.image.data[0] = 99
        return GroundObject.Response(success=True)

    store = StepStore(backend)
    context, revision = store.store('test', 'observation', 'capture_observation',
                                    serialize_message(observed))
    bindings = [StepBinding(target_field='image', source_step_id='observation', source_field='image')]
    payload = serialize_message(GroundObject.Request(target='cup'))
    first = store.run_one(context, 'ground', 'ground_object', payload, revision, bindings)[2]
    assert store.run_one(context, 'ground', 'ground_object', payload, bindings=bindings)[2] == first
    store.run_one(context, 'second', 'ground_object', payload, bindings=bindings)
    assert len(calls) == 2
    with pytest.raises(ValueError, match='UNKNOWN_CONTEXT'):
        store.run_one('missing', 'bad', 'ground_object', payload, bindings=bindings)


def test_typed_fast_path_keeps_idempotence_and_storage_limits():
    """Colocated calls retain the same state checks as serialized calls."""
    calls = []
    store = StepStore(lambda *args: (calls.append(args), GroundObject.Response(success=True))[1])
    request = GroundObject.Request(target='cup')
    context, _, item, response = store.run_typed('test', 'ground', 'ground_object', request)
    assert item.success and response.success
    store.run_typed(context, 'ground', 'ground_object', request)
    assert len(calls) == 1
    with pytest.raises(TypeError, match='REQUEST_TYPE_MISMATCH'):
        store.run_typed(context, 'invalid', 'ground_object', CaptureObservation.Request())


def test_multiple_bindings_decode_predecessor_once(monkeypatch):
    """Reading several observation fields must not decode its whole cloud repeatedly."""
    import task_pipeline.context as module
    from my_actions.srv import PrepareCloudScene

    observation = CaptureObservation.Response(success=True)
    observation.cloud.header.stamp.sec = 123
    store = StepStore(lambda name, request: PrepareCloudScene.Response(success=True))
    context, revision = store.store('test', 'observe', 'capture_observation',
                                    serialize_message(observation))
    store.submit(context, [step('scene', 'prepare_cloud_scene', PrepareCloudScene.Request(), [
        ('observation_header', 'observe', 'cloud.header'),
        ('object_cloud', 'observe', 'cloud'), ('scene_cloud', 'observe', 'cloud')])], revision)
    original = module.deserialize_message
    decodes = []

    def counted(payload, kind):
        if kind is CaptureObservation.Response:
            decodes.append(kind)
        return original(payload, kind)

    monkeypatch.setattr(module, 'deserialize_message', counted)
    assert store.run_queue(context)[0] == 'COMPLETE'
    assert len(decodes) == 1


def test_unknown_backend_completion_is_not_reexecuted_on_retry():
    """Remember timeout failures rather than repeating potentially completed actions."""
    calls = []

    def backend(*args):
        calls.append(args)
        raise RuntimeError('BACKEND_TIMEOUT_STATE_UNKNOWN')

    store = StepStore(backend)
    payload = serialize_message(GroundObject.Request())
    context, _, first = store.run_one('test', 'one', 'ground_object', payload)
    _, _, second = store.run_one(context, 'one', 'ground_object', payload)
    assert len(calls) == 1 and not first.success and second == first


def test_backend_exception_cannot_succeed_with_a_legacy_flagless_response():
    """The typed envelope preserves failure even if the old service lacks flags."""
    def backend(*args):
        raise RuntimeError('BACKEND_UNAVAILABLE')

    store = StepStore(backend)
    _, _, result = store.run_one('', 'obstacle', 'get_obstacle_location',
                                 serialize_message(GetObstacleLocation.Request()))
    assert not result.success and result.status == 'BACKEND_UNAVAILABLE'


def test_imported_result_can_feed_a_new_typed_step():
    """Support calculate, explicitly store result, then append its consumer."""
    store = StepStore(lambda name, request: SegmentGroundedImage.Response(success=True))
    context, revision = store.store('', 'ground', 'ground_object', serialize_message(
        GroundObject.Response(success=True, pixel_box=[1, 2, 3, 4])))
    consumer = step('mask', 'segment_object', SegmentGroundedImage.Request(),
                    [('bbox', 'ground', 'pixel_box')])
    store.submit(context, [consumer], revision)
    assert store.run_queue(context)[0] == 'COMPLETE'


def test_complete_refinement_response_can_feed_cache_without_external_handoff():
    """Keep the serialized refinement inside master memory between queued steps."""
    refined = RefineGrasp.Response(valid=True, candidate_id=42, grasp_width_m=.04)

    def backend(name, request):
        if name == 'refine_grasp':
            return refined
        assert name == 'cache_grasp_refinement'
        assert bytes(request.refinement_cdr) == serialize_message(refined)
        return CacheGraspRefinement.Response(success=True)

    store = StepStore(backend)
    context, _ = store.submit('', [
        step('refine', 'refine_grasp', RefineGrasp.Request()),
        step('cache', 'cache_grasp_refinement',
             CacheGraspRefinement.Request(object_name='cup'),
             [('refinement_cdr', 'refine', '@response_cdr')]),
    ])
    assert store.run_queue(context)[0] == 'COMPLETE'


def test_cancel_at_boundary_preserves_unstarted_tasks():
    """Cancel before the next RPC without claiming an in-flight service was stopped."""
    store = StepStore(lambda *args: pytest.fail('backend should not run'))
    context, _ = store.submit('', [step('ground', 'ground_object', GroundObject.Request())])
    assert store.run_queue(context, cancelled=lambda: True)[0] == 'CANCELLED_AT_STEP_BOUNDARY'
    assert len(json.loads(store.snapshot(context)[0])[0]['pending']) == 1
    store.clear(context)
    with pytest.raises(ValueError, match='UNKNOWN_CONTEXT'):
        store.snapshot(context)


def test_automatic_history_is_bounded_without_evicting_explicit_task_context():
    """Evict automatic completed calls, while preserving controller-owned state."""
    store = StepStore(lambda *args: GroundObject.Response(success=True), max_contexts=2)
    payload = serialize_message(GroundObject.Request())
    store.run_one('owned', 'one', 'ground_object', payload)
    for index in range(10):
        store.run_one('', str(index), 'ground_object', payload)
    assert len(json.loads(store.snapshot()[0])) == 2
    assert json.loads(store.snapshot('owned')[0])[0]['results'][0]['step_id'] == 'one'


@pytest.mark.parametrize('path', ['__dict__', 'image._header', 'candidates[-1]', 'image;print(1)'])
def test_binding_paths_cannot_execute_python_or_access_private_fields(path):
    """Keep bindings restricted to ROS fields and nonnegative indices."""
    with pytest.raises(ValueError):
        fields(path)


def test_registry_rejects_queue_control_functions_as_executable_steps():
    """Prevent recursive pipelines and arbitrary service dispatch."""
    with pytest.raises(ValueError, match='UNKNOWN_OR_CONTROL_FUNCTION'):
        StepStore.spec('run_step')
    with pytest.raises(ValueError, match='UNKNOWN_OR_CONTROL_FUNCTION'):
        StepStore.spec('/emergency_stop')


def test_fingerprint_preserves_wire_digest_for_buffer_and_sequence_inputs():
    """Buffer hashing preserves replay/conflict detection without concatenation."""
    import array
    import hashlib
    from task_pipeline.context import fingerprint

    payload = serialize_message(GroundObject.Request(target='brown cup'))
    expected = hashlib.sha256(b'ground_object' + payload).hexdigest()
    for value in (payload, bytearray(payload), array.array('B', payload), list(payload)):
        assert fingerprint('ground_object', value) == expected


def test_pause_keeps_physical_tasks_pending_and_resume_uses_bound_results():
    """A successful planning prefix leaves genuine queued actuator tasks untouched."""
    from my_actions.srv import ExecutePlan, PreparedGraspLift
    calls = []

    def backend(name, request):
        calls.append(name)
        if name == 'prepare_grasp_lift':
            return PreparedGraspLift.Response(success=True, plan_revision=17)
        assert request.plan_revision == 17 and request.segment == 'lift'
        return ExecutePlan.Response(success=True, motion_completed=True)

    store = StepStore(backend)
    context, _ = store.submit('', [
        step('prepare', 'prepare_grasp_lift', PreparedGraspLift.Request()),
        step('lift', 'execute_plan', ExecutePlan.Request(segment='lift'),
             [('plan_revision', 'prepare', 'plan_revision')]),
    ])
    status, results = store.run_queue(context, stop_before_step_id='lift')
    assert status == 'PAUSED_AT_STEP_BOUNDARY' and len(results) == 1
    state = json.loads(store.snapshot(context)[0])[0]
    assert state['pending'] == [{'step_id': 'lift', 'function': 'execute_plan'}]
    assert calls == ['prepare_grasp_lift']
    status, results = store.run_queue(context)
    assert status == 'COMPLETE' and len(results) == 1
    assert calls == ['prepare_grasp_lift', 'execute_plan']


def test_invalid_pause_boundary_does_not_execute_or_lock_context():
    """Reject misspelled pause boundaries before any work starts."""
    store = StepStore(lambda *_: pytest.fail('Must not execute'))
    context, _ = store.submit('', [step('observe', 'capture_observation',
                                       CaptureObservation.Request())])
    with pytest.raises(ValueError, match='UNKNOWN_PAUSE_BOUNDARY'):
        store.run_queue(context, stop_before_step_id='missing')
    assert not json.loads(store.snapshot(context)[0])[0]['busy']


def test_failure_flush_only_discards_idle_queues_for_matching_object():
    """Recovery cannot erase another object's tasks or a busy owner."""
    from my_actions.srv import ExecutePlan
    store = StepStore(lambda *_: None)
    ids = []
    for name in ('bowl', 'cup', 'bowl'):
        context, _ = store.submit('', [step('lift', 'execute_plan', ExecutePlan.Request(
            object_name=name, segment='lift'))])
        ids.append(context)
    store._contexts[ids[2]].busy = True
    store.flush_execution_queues('bowl')
    states = {s['context_id']: s for s in json.loads(store.snapshot()[0])}
    assert not states[ids[0]]['pending']
    assert states[ids[1]]['pending'] and states[ids[2]]['pending']


def test_failed_queue_flush_retains_completed_results_and_other_contexts():
    """Planning rejection also discards future physical tasks without global erasure."""
    from my_actions.srv import ExecutePlan
    store = StepStore(lambda *_: CaptureObservation.Response(success=True))
    context, _ = store.submit('', [
        step('observe', 'capture_observation', CaptureObservation.Request()),
        step('lift', 'execute_plan', ExecutePlan.Request(object_name='bowl', segment='lift'))])
    other, _ = store.submit('', [step('observe', 'capture_observation', CaptureObservation.Request())])
    store.run_queue(context, stop_before_step_id='lift')
    store.flush_failed_queue(context)
    states = {s['context_id']: s for s in json.loads(store.snapshot()[0])}
    assert not states[context]['pending'] and len(states[context]['results']) == 1
    assert states[other]['pending']


def test_typed_fingerprint_ignores_cdr_padding_and_detects_semantic_changes():
    from task_pipeline.context import request_fingerprint
    first=GroundObject.Request(target='cup')
    same=GroundObject.Request(target='cup')
    assert request_fingerprint('ground_object',first)==request_fingerprint('ground_object',same)
    same.target='bowl'
    assert request_fingerprint('ground_object',first)!=request_fingerprint('ground_object',same)
    first.image.data=bytes([1,2,3,4])
    same.target='cup';same.image.data=bytes([1,2,3,5])
    assert request_fingerprint('ground_object',first)!=request_fingerprint('ground_object',same)
