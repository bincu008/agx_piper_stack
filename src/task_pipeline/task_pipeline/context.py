"""Master-owned typed artifacts and a bounded queue of API microsteps."""

import copy
import hashlib
import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from my_actions.msg import PipelineStepResult

from rclpy.serialization import deserialize_message, serialize_message

from .node import response_status, response_success, set_failure
from .registry import FUNCTIONS


_FIELD = re.compile(r'([a-z][a-z0-9_]*)(?:\[([0-9]+)\])?')


def fields(path):
    """Accept public message fields and explicit nonnegative array indices."""
    parts = path.split('.')
    matches = [_FIELD.fullmatch(part) for part in parts]
    if not path or not all(matches):
        raise ValueError('INVALID_BINDING_FIELD: ' + path)
    return [(match[1], None if match[2] is None else int(match[2])) for match in matches]


def get_field(message, path):
    """Read a typed field without eval or Python private attribute access."""
    for name, index in fields(path):
        if name not in message.get_fields_and_field_types():
            raise ValueError('UNKNOWN_MESSAGE_FIELD: ' + name)
        message = getattr(message, name)
        if index is not None:
            message = message[index]
    return message


def set_field(message, path, value):
    """Copy a predecessor result into a typed request field."""
    parts = fields(path)
    for name, index in parts[:-1]:
        if name not in message.get_fields_and_field_types():
            raise ValueError('UNKNOWN_MESSAGE_FIELD: ' + name)
        message = getattr(message, name)
        if index is not None:
            message = message[index]
    name, index = parts[-1]
    if name not in message.get_fields_and_field_types():
        raise ValueError('UNKNOWN_MESSAGE_FIELD: ' + name)
    if index is None:
        setattr(message, name, copy.deepcopy(value))
    else:
        getattr(message, name)[index] = copy.deepcopy(value)


def fingerprint(name, payload):
    """Hash the same wire bytes without concatenating a large second buffer."""
    digest = hashlib.sha256(name.encode())
    try:
        view = memoryview(payload)
    except TypeError:
        view = memoryview(bytes(payload))
    digest.update(view)
    return digest.hexdigest()


@dataclass
class Context:
    """Own immutable result payloads; pending tasks reference them by step ID."""

    context_id: str
    revision: int = 1
    pending: list = field(default_factory=list)
    results: OrderedDict = field(default_factory=OrderedDict)
    fingerprints: dict = field(default_factory=dict)
    busy: bool = False
    status: str = 'READY'
    automatic: bool = False


class StepStore:
    """Atomically append tasks and execute each result dependency once."""

    def __init__(self, call_backend, max_contexts=32, max_steps=256,
                 max_bytes=256 * 1024 * 1024):
        """Inject only the catalogued backend caller, never arbitrary endpoints."""
        self._call = call_backend
        self._contexts = OrderedDict()
        self._lock = threading.RLock()
        self._max_contexts = max_contexts
        self._max_steps = max_steps
        self._max_bytes = max_bytes

    @staticmethod
    def spec(name):
        """Restrict step execution to engine functions, not queue administration."""
        if name not in FUNCTIONS or name in {
                'run_step', 'submit_steps', 'get_step_context', 'clear_step_context',
                'store_step_result', 'append_task', 'execute_pipeline', 'flush_pipeline'}:
            raise ValueError('UNKNOWN_OR_CONTROL_FUNCTION: ' + name)
        return FUNCTIONS[name]

    def _context(self, context_id, create=False, automatic=False):
        if context_id in self._contexts:
            return self._contexts[context_id]
        if not create:
            raise ValueError('UNKNOWN_CONTEXT: ' + context_id)
        self._make_room()
        context_id = context_id or uuid.uuid4().hex
        context = Context(context_id, automatic=automatic)
        self._contexts[context_id] = context
        return context

    def _bytes(self):
        return sum(sum(len(item.request_cdr) for item in ctx.pending) +
                   sum(len(item.response_cdr) for item in ctx.results.values())
                   for ctx in self._contexts.values())

    def _make_room(self, extra_bytes=0, protect=None):
        while ((len(self._contexts) >= self._max_contexts if protect is None else
                len(self._contexts) > self._max_contexts) or
               self._bytes() + extra_bytes > self._max_bytes):
            victim = next((key for key, ctx in self._contexts.items()
                           if key != protect and ctx.automatic and
                           not ctx.busy and not ctx.pending), None)
            if victim is None:
                raise RuntimeError('PIPELINE_STORAGE_FULL')
            del self._contexts[victim]

    @staticmethod
    def _revision(context, expected):
        if expected and expected != context.revision:
            raise ValueError('STALE_CONTEXT_REVISION')

    def submit(self, context_id, steps, expected=0):
        """Validate a complete append before changing state or queue order."""
        steps = copy.deepcopy(list(steps))
        if not steps:
            raise ValueError('EMPTY_STEP_SEQUENCE')
        for step in steps:
            spec = self.spec(step.function_name)
            deserialize_message(bytes(step.request_cdr), spec.service_type.Request)
            if not step.step_id:
                raise ValueError('STEP_ID_REQUIRED')
            if step.failure_policy not in ('', 'stop', 'continue'):
                raise ValueError('INVALID_FAILURE_POLICY')
            for binding in step.bindings:
                fields(binding.target_field)
                if binding.source_field != '@response_cdr':
                    fields(binding.source_field)
        with self._lock:
            ctx = self._context(context_id, create=True)
            self._revision(ctx, expected)
            if ctx.busy:
                raise RuntimeError('CONTEXT_BUSY')
            if any('STATE_UNKNOWN' in item.status for item in ctx.results.values()):
                raise RuntimeError('CONTEXT_REQUIRES_STATE_RECONCILIATION')
            if len(ctx.pending) + len(ctx.results) + len(steps) > self._max_steps:
                raise RuntimeError('CONTEXT_STEP_LIMIT')
            available = set(ctx.results) | {item.step_id for item in ctx.pending}
            for step in steps:
                if step.step_id in available:
                    raise ValueError('DUPLICATE_STEP_ID: ' + step.step_id)
                if any(binding.source_step_id not in available for binding in step.bindings):
                    raise ValueError('MISSING_OR_FORWARD_DEPENDENCY: ' + step.step_id)
                available.add(step.step_id)
            self._make_room(sum(len(step.request_cdr) for step in steps), ctx.context_id)
            ctx.pending.extend(steps)
            ctx.revision += 1
            ctx.status = 'QUEUED'
            return ctx.context_id, ctx.revision

    def store(self, context_id, step_id, function_name, response_cdr, expected=0):
        """Import a successful typed result before atomically appending its consumer."""
        spec = self.spec(function_name)
        response = deserialize_message(bytes(response_cdr), spec.service_type.Response)
        if not response_success(response):
            raise ValueError('FAILED_RESULT_CANNOT_BE_IMPORTED')
        if not step_id:
            raise ValueError('STEP_ID_REQUIRED')
        item = PipelineStepResult(
            step_id=step_id, function_name=function_name, success=True,
            status=response_status(response), response_type=self.response_type(spec),
            response_cdr=bytes(response_cdr))
        with self._lock:
            ctx = self._context(context_id, create=True)
            self._revision(ctx, expected)
            if ctx.busy or ctx.pending:
                raise RuntimeError('CONTEXT_BUSY_OR_ALREADY_QUEUED')
            if step_id in ctx.results:
                raise ValueError('DUPLICATE_STEP_ID')
            if len(ctx.results) >= self._max_steps:
                raise RuntimeError('CONTEXT_STEP_LIMIT')
            self._make_room(len(response_cdr), ctx.context_id)
            ctx.results[step_id] = item
            ctx.revision += 1
            return ctx.context_id, ctx.revision

    @staticmethod
    def response_type(spec):
        """Identify the response schema without transporting Python objects."""
        module = spec.service_type.__module__.split('.')[0]
        return f'{module}/srv/{spec.service_type.__name__}_Response'

    def run_one(self, context_id, step_id, function_name, request_cdr, expected=0, bindings=()):
        """Run a public call and retain its typed result in master memory."""
        spec = self.spec(function_name)
        request = deserialize_message(bytes(request_cdr), spec.service_type.Request)
        if bindings:
            # Reference names are part of idempotence. Completed results are
            # immutable within their context; no large resolved re-encoding.
            refs = [(b.target_field, b.source_step_id, b.source_field) for b in bindings]
            request_cdr = bytes(request_cdr) + json.dumps(refs).encode()
        return self._run_one(
            context_id, step_id, spec, request, request_cdr, expected, bindings)[:3]

    def _bind(self, ctx, request, bindings):
        """Resolve immutable artifacts once per predecessor, with isolated fields."""
        decoded = {}
        for binding in bindings:
            source = ctx.results.get(binding.source_step_id)
            if source is None or not source.success:
                raise ValueError('DEPENDENCY_FAILED: ' + binding.source_step_id)
            if binding.source_field == '@response_cdr':
                value = bytes(source.response_cdr)
            else:
                if binding.source_step_id not in decoded:
                    source_spec = self.spec(source.function_name)
                    decoded[binding.source_step_id] = deserialize_message(
                        bytes(source.response_cdr), source_spec.service_type.Response)
                value = get_field(decoded[binding.source_step_id], binding.source_field)
            set_field(request, binding.target_field, value)

    def run_typed(self, context_id, step_id, function_name, request, expected=0):
        """Keep a local typed request without an envelope encode/decode round trip."""
        spec = self.spec(function_name)
        if not isinstance(request, spec.service_type.Request):
            raise TypeError('REQUEST_TYPE_MISMATCH')
        return self._run_one(context_id, step_id, spec, request,
                             serialize_message(request), expected)

    def _run_one(self, context_id, step_id, spec, request, request_cdr, expected, bindings=()):
        """Share revision, idempotence and bounded storage checks for both paths."""
        function_name = spec.name
        digest = fingerprint(function_name, request_cdr)
        with self._lock:
            ctx = self._context(context_id, create=not bindings, automatic=not context_id)
            self._revision(ctx, expected)
            step_id = step_id or uuid.uuid4().hex
            if step_id in ctx.results:
                if ctx.fingerprints.get(step_id) != digest:
                    raise ValueError('STEP_ID_PAYLOAD_CONFLICT')
                item = copy.deepcopy(ctx.results[step_id])
                return ctx.context_id, ctx.revision, item, deserialize_message(
                    bytes(item.response_cdr), spec.service_type.Response)
            if ctx.busy or ctx.pending:
                raise RuntimeError('CONTEXT_BUSY_OR_ALREADY_QUEUED')
            if any('STATE_UNKNOWN' in item.status for item in ctx.results.values()):
                raise RuntimeError('CONTEXT_REQUIRES_STATE_RECONCILIATION')
            if len(ctx.results) >= self._max_steps:
                raise RuntimeError('CONTEXT_STEP_LIMIT')
            self._bind(ctx, request, bindings)
            ctx.busy = True
        try:
            item, response = self._execute(ctx, step_id, spec, request, digest)
            return ctx.context_id, ctx.revision, copy.deepcopy(item), response
        finally:
            with self._lock:
                ctx.busy = False

    def _execute(self, ctx, step_id, spec, request, digest):
        started = time.monotonic()
        failure = None
        try:
            response = self._call(spec.name, request)
            if not isinstance(response, spec.service_type.Response):
                raise TypeError('BACKEND_RESPONSE_TYPE_MISMATCH')
        except Exception as error:
            failure = str(error)
            response = set_failure(spec.service_type.Response(), str(error))
        item = PipelineStepResult(
            step_id=step_id, function_name=spec.name,
            success=failure is None and response_success(response),
            status=failure if failure is not None else response_status(response),
            wall_seconds=time.monotonic() - started,
            response_type=self.response_type(spec), response_cdr=serialize_message(response))
        with self._lock:
            try:
                self._make_room(len(item.response_cdr), ctx.context_id)
            except RuntimeError:
                # Never repeat a potentially executed operation because its output
                # did not fit. Retain a small typed failure and its fingerprint.
                response = set_failure(
                    spec.service_type.Response(), 'RESULT_STORAGE_FULL_STATE_UNKNOWN')
                item.response_cdr = serialize_message(response)
                item.success = False
                item.status = 'RESULT_STORAGE_FULL_STATE_UNKNOWN'
            ctx.results[step_id] = item
            ctx.fingerprints[step_id] = digest
            ctx.revision += 1
            ctx.status = 'SUCCEEDED' if item.success else 'FAILED'
        return item, response

    def run_queue(self, context_id, cancelled=lambda: False, feedback=lambda *args: None):
        """Deplete each attempted task; never consume failed dependency results."""
        with self._lock:
            ctx = self._context(context_id)
            if ctx.busy:
                raise RuntimeError('CONTEXT_BUSY')
            if any('STATE_UNKNOWN' in item.status for item in ctx.results.values()):
                raise RuntimeError('CONTEXT_REQUIRES_STATE_RECONCILIATION')
            ctx.busy = True
            ctx.status = 'RUNNING'
        completed = []
        try:
            while True:
                with self._lock:
                    if cancelled():
                        ctx.status = 'CANCELLED_AT_STEP_BOUNDARY'
                        break
                    if not ctx.pending:
                        ctx.status = (
                            'COMPLETE' if all(item.success for item in completed) else 'PARTIAL')
                        break
                    step = ctx.pending[0]
                spec = self.spec(step.function_name)
                request = deserialize_message(bytes(step.request_cdr), spec.service_type.Request)
                try:
                    self._bind(ctx, request, step.bindings)
                    digest = fingerprint(spec.name, serialize_message(request))
                    feedback(step, 'RUNNING', len(completed), len(ctx.pending))
                    item, _ = self._execute(ctx, step.step_id, spec, request, digest)
                except Exception as error:
                    response = set_failure(spec.service_type.Response(), str(error))
                    item = PipelineStepResult(
                        step_id=step.step_id, function_name=spec.name, success=False,
                        status=str(error), response_type=self.response_type(spec),
                        response_cdr=serialize_message(response))
                    with self._lock:
                        ctx.results[step.step_id] = item
                        ctx.revision += 1
                completed.append(copy.deepcopy(item))
                with self._lock:
                    ctx.pending.pop(0)
                    ctx.revision += 1
                feedback(step, item.status, len(completed), len(ctx.pending))
                if not item.success and step.failure_policy != 'continue':
                    ctx.status = 'STOPPED_ON_FAILURE'
                    break
            return ctx.status, completed
        finally:
            with self._lock:
                ctx.busy = False

    def clear(self, context_id, expected=0):
        """Release completed artifacts explicitly; reject clearing an active call."""
        with self._lock:
            ctx = self._context(context_id)
            self._revision(ctx, expected)
            if ctx.busy:
                raise RuntimeError('CONTEXT_BUSY')
            del self._contexts[context_id]

    def snapshot(self, context_id='', include_results=False):
        """Expose queue state without serializing large images into JSON."""
        with self._lock:
            contexts = [self._context(context_id)] if context_id else list(self._contexts.values())
            state = [{
                'context_id': ctx.context_id, 'revision': ctx.revision, 'busy': ctx.busy,
                'status': ctx.status,
                'pending': [{'step_id': step.step_id, 'function': step.function_name}
                            for step in ctx.pending],
                'results': [{'step_id': item.step_id, 'function': item.function_name,
                             'success': item.success, 'status': item.status,
                             'seconds': item.wall_seconds} for item in ctx.results.values()],
            } for ctx in contexts]
            results = ([copy.deepcopy(item) for ctx in contexts for item in ctx.results.values()]
                       if include_results else [])
            return json.dumps(state), results if include_results else []
