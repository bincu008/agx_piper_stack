"""Consolidated pipeline performance metrics and timing diagnostics."""

from collections import defaultdict
from statistics import median
import uuid

from my_actions.msg import PipelineTiming


def publish_timing(publisher, name, seconds, header, object_name, success, status,
                   scope='compute', full_motion=False):
    """Emit a typed sample for a shared local calculation."""
    if publisher is None:
        return
    publisher.publish(PipelineTiming(
        operation_id=uuid.uuid4().hex, function_name=name,
        backend='heightmap' if name == 'search_placements' else 'vamp',
        input_mode='prepared_geometry', scope=scope,
        observation_header=header, object_name=object_name,
        wall_seconds=seconds, success=success, status=status,
        full_motion=full_motion))


def summarize(samples):
    """Report call counts and distributions for each input/validation mode."""
    grouped = defaultdict(list)
    for sample in samples:
        key = (sample.function_name, sample.input_mode, sample.scope, sample.full_motion)
        grouped[key].append(sample)
    result = []
    for (name, mode, scope, motion), values in sorted(grouped.items()):
        seconds = sorted(value.wall_seconds for value in values)
        result.append({
            'function': name, 'input_mode': mode, 'scope': scope,
            'full_motion': motion, 'calls': len(values),
            'failed_calls': sum(not value.success for value in values),
            'median_seconds': median(seconds), 'minimum_seconds': seconds[0],
            'maximum_seconds': seconds[-1],
        })
    return result


def summarize_api_calls(calls, traces):
    """Correlate client and server timings for APIs running on the same host."""
    rows = []
    for call in calls:
        row = dict(call)
        stages = traces.get(call['operation_id'], {})
        api = stages.get('dispatcher')
        master = stages.get('master')
        row['total_seconds'] = (call['finished_ns'] - call['started_ns']) / 1e9
        row['trace_complete'] = False
        if api and master and master.backend_started_ns and master.backend_finished_ns:
            stamps = [call['started_ns'], call['sent_ns'], api.received_ns,
                      master.backend_started_ns, master.backend_finished_ns,
                      api.finished_ns, call['response_received_ns'], call['finished_ns']]
            if all(right >= left for left, right in zip(stamps, stamps[1:])):
                names = ('client_request', 'request_delivery', 'server_request',
                         'backend', 'server_response', 'response_delivery', 'client_response')
                row['phases_seconds'] = {
                    name: (right - left) / 1e9
                    for name, left, right in zip(names, stamps, stamps[1:])}
                row['trace_complete'] = True
        rows.append(row)
    functions = []
    for name in sorted({row['function'] for row in rows}):
        group = [row for row in rows if row['function'] == name]
        complete = [row for row in group if row['trace_complete']]
        phases = {phase: median(row['phases_seconds'][phase] for row in complete)
                  for phase in complete[0]['phases_seconds']} if complete else {}
        functions.append({
            'function': name, 'calls': len(group), 'complete_traces': len(complete),
            'median_total_seconds': median(row['total_seconds'] for row in group),
            'median_phases_seconds': phases,
            'total_seconds': sum(row['total_seconds'] for row in group),
            'backend_seconds': sum(row['phases_seconds']['backend'] for row in complete),
        })
    return {'calls': rows, 'functions': functions,
            'complete_traces': sum(row['trace_complete'] for row in rows),
            'total_api_seconds': sum(row['total_seconds'] for row in rows),
            'clock': 'same-host monotonic_ns',
            'note': 'Phases sum per call. Medians of phases need not sum to median total. '
                    'Backend includes its ROS transport and computation, including remote VLM.'}
