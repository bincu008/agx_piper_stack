"""Only inspection/calculation clients: no action client or actuator dispatcher."""

import asyncio
import json
import os
import threading
import time

from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields

from task_pipeline.registry import FUNCTIONS

from .contracts import ALLOWLIST


class ReadBridge:
    """Bound ROS reads; an MCP cancellation never claims to stop ROS motion."""

    def __init__(self, node, timeout=5.):
        """Initialize isolated state."""
        self.node, self.timeout = node, timeout
        self.clients = {name: node.create_client(FUNCTIONS[name].service_type,
                                                 FUNCTIONS[name].endpoint)
                        for name in ALLOWLIST if name != 'list_functions'}

    def input_provenance(self):
        """Inspect raw publisher identities; capture still checks freshness/sync."""
        expected = {
            '/camera/color/image_raw': '/camera/camera',
            '/camera/depth_registered/points': '/camera/camera',
            '/feedback/joint_states': '/agx_arm_ctrl_single_node',
            '/feedback/tcp_pose': '/agx_arm_ctrl_single_node',
        }
        sources, verified, replay = {}, True, False
        for topic, owner in expected.items():
            publishers = self.node.get_publishers_info_by_topic(topic)
            names = [p.node_namespace.rstrip('/')+'/'+p.node_name for p in publishers]
            sources[topic] = {'publisher_count': len(names), 'publisher_nodes': names}
            verified = verified and names == [owner]
            replay = replay or any('mock_hardware' in name for name in names)
        return {'ros_domain_id': int(os.environ.get('ROS_DOMAIN_ID', '0')),
                'classification': ('live_driver_publishers' if verified else
                                   'replay_or_mixed' if replay else 'unverified'),
                'topics': sources,
                'detail': 'DDS publisher identity snapshot; capture validates fresh sync. '
                          'This is not an actuator readiness certificate.'}

    async def call(self, name, arguments):
        """Read an allowlisted snapshot."""
        if name not in self.clients or name not in ALLOWLIST:
            raise ValueError('READ_ONLY_SERVER: operation is not allowed')
        return await asyncio.to_thread(self._call, name, arguments)

    def _call(self, name, arguments):
        client = self.clients[name]
        timeout = 30. if name == 'get_observe_view' else self.timeout
        deadline = time.monotonic()+timeout
        if not client.wait_for_service(timeout_sec=min(1., timeout)):
            return {'transport_outcome': 'unavailable', 'response': None,
                    'detail': 'ROS_SERVICE_UNAVAILABLE; no robot readiness inferred'}
        request = FUNCTIONS[name].service_type.Request()
        set_message_fields(request, arguments)
        future = client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(max(0., deadline-time.monotonic())):
            future.cancel()  # Local wait only; cannot cancel an in-flight ROS service.
            return {'transport_outcome': 'unknown', 'response': None,
                    'detail': 'READ_TIMEOUT; ROS operation may still be running'}
        result = message_to_ordereddict(future.result())
        # Refuse non-JSON feedback or oversized state rather than truncate evidence.
        if len(json.dumps(result, allow_nan=False).encode()) > 1024*1024:
            raise ValueError('READ_RESULT_TOO_LARGE')
        return {'transport_outcome': 'response', 'response': result,
                'detail': 'Raw ROS response; inspect success/status/phase separately'}
