"""Group comparable samples without adding nested timing scopes."""

from collections import defaultdict
from statistics import median


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
