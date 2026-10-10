"""Correlate client and server timings for APIs running on the same host."""

from statistics import median


def summarize_api_calls(calls, traces):
    """Return disjoint phases; inclusive API and backend scopes are not added."""
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
