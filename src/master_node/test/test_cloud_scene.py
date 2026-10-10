"""Check that reusable geometry remains associated with its original observation."""

import copy

from master_node.step_runtime import MasterStepRuntime
from my_actions.srv import PrepareCloudScene
import numpy as np
import pytest
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header



def scene_request():
    """Create geometry in the computational base frame with a shared timestamp."""
    header = Header(frame_id='base_link')
    header.stamp.sec = 123
    cloud = point_cloud2.create_cloud_xyz32(header, np.zeros((80, 3), np.float32))
    return PrepareCloudScene.Request(observation_header=copy.deepcopy(header),
                                     object_cloud=copy.deepcopy(cloud),
                                     scene_cloud=copy.deepcopy(cloud))


def test_prepared_scene_preserves_exact_cloud_and_observation():
    """Registration must not resize, transform or filter the submitted geometry."""
    request = scene_request()
    result = MasterStepRuntime._prepare_cloud_scene(request, PrepareCloudScene.Response())
    assert result.success
    assert result.object_cloud == request.object_cloud
    assert result.scene_cloud == request.scene_cloud
    assert result.observation_header == request.observation_header


@pytest.mark.parametrize('fault', ['frame', 'stamp', 'data', 'fields'])
def test_prepared_scene_rejects_mismatched_or_malformed_clouds(fault):
    """Different frames/times and broken layouts cannot enter the result cache."""
    request = scene_request()
    if fault == 'frame':
        request.observation_header.frame_id = 'camera'
    elif fault == 'stamp':
        request.scene_cloud.header.stamp.sec = 124
    elif fault == 'data':
        request.scene_cloud.data = bytes([1])
    else:
        request.scene_cloud.fields = []
    result = MasterStepRuntime._prepare_cloud_scene(request, PrepareCloudScene.Response())
    assert not result.success
