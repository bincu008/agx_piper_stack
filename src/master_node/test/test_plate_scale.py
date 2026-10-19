"""Check tilted-plate measurement and strict JSON scale loading."""

import json

import numpy as np
import pytest

from master_node.metric_calibration import load_metric_ratio
from master_node.plate_scale import measure_plate


def test_tilted_circle_recovers_uniform_ratio_and_preserves_axis_uncertainty():
    """Perspective projection must not be mistaken for anisotropic scale."""
    yy, xx = np.mgrid[:320, :400]
    rays = np.stack(((xx-200)/500., (yy-160)/500., np.ones_like(xx)), axis=2)
    normal = np.array([0., .4, .916515])
    center = np.array([0., 0., .5])
    points = rays * ((normal@center)/(rays@normal))[:, :, None]
    mask = np.linalg.norm(points-center, axis=2) <= .087
    report = measure_plate(points*1.05/1.036, mask, .174, 1.036)
    assert report['perception_measurement_ratio'] == pytest.approx(1.05, abs=.015)
    assert report['rim_plane_inlier_fraction'] > .98
    assert not report['axis_calibration_independent']


@pytest.mark.parametrize('ratio', [0., -1., '1.03', True, float('nan')])
def test_invalid_json_ratio_is_rejected(tmp_path, ratio):
    """Invalid persisted configuration must fail rather than corrupt perception."""
    path = tmp_path/'scale.json'
    path.write_text(json.dumps({'perception_measurement_ratio': ratio}))
    with pytest.raises(ValueError):
        load_metric_ratio(path)


def test_valid_json_ratio(tmp_path):
    """Load only the explicit measurement-to-real convention."""
    path = tmp_path/'scale.json'
    path.write_text(json.dumps({'perception_measurement_ratio': .98}))
    assert load_metric_ratio(path) == .98


def test_grasp_offset_uses_explicit_base_frame_and_metres(tmp_path):
    """Position correction is separate from scale and preserves the requested signs."""
    from master_node.metric_calibration import load_grasp_offset
    path = tmp_path/'offset.json'
    path.write_text(json.dumps({'frame_id': 'base_link',
                                'grasp_position_offset_base_m': [-.005, .005, .005]}))
    assert load_grasp_offset(path) == [-.005, .005, .005]
    path.write_text(json.dumps({'frame_id': 'tool',
                                'grasp_position_offset_base_m': [-.005, .005, .005]}))
    with pytest.raises(ValueError):
        load_grasp_offset(path)
