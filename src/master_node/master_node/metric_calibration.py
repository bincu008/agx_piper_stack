"""Load validated metric-scale configuration without changing robot geometry."""

import json
import math


def load_metric_ratio(path):
    """Reject malformed/nonfinite calibration instead of silently using a default."""
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    ratio = data.get('perception_measurement_ratio')
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError('Calibration requires a numeric perception_measurement_ratio')
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('perception_measurement_ratio must be finite and positive')
    return float(ratio)


def load_grasp_offset(path):
    """Read a Cartesian command correction expressed in base_link metres."""
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    if data.get('frame_id') != 'base_link':
        raise ValueError('Grasp offset must be expressed in base_link')
    values = data.get('grasp_position_offset_base_m')
    if not isinstance(values, list) or len(values) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values):
        raise ValueError('grasp_position_offset_base_m requires three finite metres')
    return [float(v) for v in values]
