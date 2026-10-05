"""Common timing samples; inclusive scopes must not be added together."""

import uuid

from my_actions.msg import PipelineTiming


def publish_timing(publisher, name, seconds, header, object_name, success, status,
                   scope='compute', full_motion=False):
    """Emit a typed sample for a shared local calculation."""
    if publisher is None:
        return
    publisher.publish(PipelineTiming(
        operation_id=uuid.uuid4().hex, function_name=name, backend='heightmap' if
        name == 'search_placements' else 'vamp', input_mode='prepared_geometry',
        scope=scope, observation_header=header, object_name=object_name,
        wall_seconds=seconds, success=success, status=status, full_motion=full_motion))
