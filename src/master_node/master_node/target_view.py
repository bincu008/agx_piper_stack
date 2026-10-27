"""Read-only target visibility from memory and a synchronized manual camera pose."""

import copy

from geometry_msgs.msg import Point

from my_actions import srv


def check_target_view(memory, request, response):
    """Resolve expected visibility without inference, memory mutation or motion."""
    try:
        role = request.role or 'object'
        if role not in ('object', 'destination') or not request.name.strip():
            raise ValueError('INVALID_TARGET_VIEW_REQUEST')
        with memory.lock:
            record = memory._record(request.name, role)
            if record is None:
                records = ([memory.destination] if role == 'destination'
                           else list(memory.objects.values()))
                record = next((r for r in records if r and r['object_id'] == request.name), None)
            response.memory_revision = memory.revision
            if record is not None:
                if record['calibration'] != memory.calibration():
                    response.success, response.status = True, 'LOCATION_UNKNOWN'
                    return response
                position = list(record['position'])
                response.object_id = record['object_id']
                target_cloud = record['cloud']
                response.memory_revision = record['revision']
            else:
                entries = [(key, item) for key, item in memory.directory.items()
                           if key == request.name or
                           memory.key(item['name']) == memory.key(request.name)]
                if len(entries) != 1:
                    response.success, response.status = True, 'LOCATION_UNKNOWN'
                    return response
                response.object_id, entry = entries[0]
                position = list(entry['position'])
                target_cloud = None
                response.location_only = True
        response.position = Point(**dict(zip(('x', 'y', 'z'), map(float, position))))
        if request.observation_cloud.width:
            cloud, transform = request.observation_cloud, request.camera_transform
        else:
            if request.camera_transform.header.frame_id:
                raise ValueError('TARGET_VIEW_OBSERVATION_PAIR_REQUIRED')
            observation = memory.call('capture_observation', srv.CaptureObservation.Request())
            if not observation.success:
                raise RuntimeError('TARGET_VIEW_CAPTURE_FAILED: ' + observation.status)
            cloud, transform = observation.cloud, observation.camera_transform
        info = copy.deepcopy(getattr(memory.node, 'current_camera_info', None))
        if info is None:
            raise RuntimeError('TARGET_VIEW_CAMERA_INFO_UNAVAILABLE')
        query = srv.CheckTargetVisibility.Request(
            scene_cloud=cloud, camera_transform=transform,
            camera_info=info, target_position=response.position)
        if target_cloud is not None:
            query.target_cloud = target_cloud
        result = memory.call('check_target_visibility', query)
        for field in ('success', 'status', 'visible', 'observation_header',
                      'sample_count', 'in_view_count', 'clear_count', 'visible_fraction'):
            setattr(response, field, getattr(result, field))
    except Exception as error:
        response.success, response.visible = False, False
        response.status = str(error)
    return response
