"""Only inspection/calculation clients: no action client or actuator dispatcher."""

import asyncio
import json
import threading
import time

from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields

from task_pipeline.registry import FUNCTIONS

from .contracts import ALLOWLIST, LOCAL_DOCUMENTATION


class ReadBridge:
    """Bound ROS reads; an MCP cancellation never claims to stop ROS motion."""

    def __init__(self, node, timeout=5.):
        """Initialize isolated state."""
        self.node, self.timeout = node, timeout
        self._workspace_client = None
        self._workspace_lock = threading.Lock()
        self.clients = {name: node.create_client(FUNCTIONS[name].service_type,
                                                 FUNCTIONS[name].endpoint)
                        for name in ALLOWLIST if name not in LOCAL_DOCUMENTATION}

    def input_provenance(self):
        """Inspect raw publisher identities; capture still checks freshness/sync."""
        from task_pipeline.provenance import input_provenance
        return input_provenance(self.node)

    async def call(self, name, arguments):
        """Read an allowlisted snapshot."""
        if name not in self.clients or name not in ALLOWLIST:
            raise ValueError('READ_ONLY_SERVER: operation is not allowed')
        return await asyncio.to_thread(self._call, name, arguments)

    async def workspace(self, include_image=False):
        """Read internal typed metadata without adding a generic/native dispatch tool."""
        def read():
            with self._workspace_lock:
                if self._workspace_client is None:
                    spec = FUNCTIONS['get_current_workspace']
                    self._workspace_client = self.node.create_client(
                        spec.service_type, spec.endpoint)
            return self._call('get_current_workspace', {'include_image': include_image},
                              client=self._workspace_client)
        return await asyncio.to_thread(read)

    def _call(self, name, arguments, client=None):
        client = self.clients[name] if client is None else client
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
        message = future.result()
        image_content = None
        if name == 'get_current_workspace' and arguments.get('include_image'):
            from .workspace import encode_image
            image_content = encode_image(message.image)
            from sensor_msgs.msg import Image
            message.image = Image()  # Do not turn raw pixels into a JSON integer array.
        result = message_to_ordereddict(message)
        if image_content:
            result['_image_content'] = image_content
        # Refuse non-JSON feedback or oversized state rather than truncate evidence.
        if len(json.dumps(result, allow_nan=False).encode()) > 1024*1024:
            raise ValueError('READ_RESULT_TOO_LARGE')
        return {'transport_outcome': 'response', 'response': result,
                'detail': 'Raw ROS response; inspect success/status/phase separately'}
