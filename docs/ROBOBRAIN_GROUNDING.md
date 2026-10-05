# RoboBrain grounding client

Grounding sends the original image before the target prompt, with temperature 0,
seed 1234, max_tokens 48, ignore_eos false and cache_prompt false. Existing
non-streaming mode is retained. Resizing and image-token budgets belong to the
server. Non-grounding VLM modes retain their existing generation settings.

Responses must contain only one four-integer box within 0–1000 with increasing
corners. C++ keeps its existing inclusive pixel convention; placement Python
keeps its existing exclusive crop upper bounds. Existing synchronized image
headers remain attached to downstream requests and masks. Inference rejects
concurrent requests instead of queueing stale camera frames.

The default optimized endpoint is `http://10.10.10.2:18091/v1`. Configure the remote
host with `api_server_ip` and `api_server_port`, or override `api_server_url`.
The default `request_timeout_sec` is 120. `vlm_max_tokens` and `vlm_temperature`
continue to configure other VLM modes; grounding uses the selected fixed values.
`vlm_image_max_dimension` applies only to non-grounding VLM requests.

```bash
ros2 launch master_node reasoning_stack.launch.py \
  api_server_ip:=10.10.10.2 api_server_port:=18091 \
  request_timeout_sec:=120 allow_physical_grasp_execution:=false
```

For inference performance only, start no ROS nodes or hardware drivers. Export
the bundled replay image first; the derived preview PNG is not committed:

```bash
source install/setup.bash
python3 - <<'PY'
from pathlib import Path
import cv2
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from mock_hardware_node.snapshot_io import load_messages
snapshot = Path(get_package_share_directory('mock_hardware_node')) / 'snapshots/current'
messages, _ = load_messages(snapshot)
image = CvBridge().imgmsg_to_cv2(messages['rgb'], 'bgr8')
cv2.imwrite(str(Path.home() / 'grounding_input.png'), image)
PY
```

```bash
source install/setup.bash
ros2 run placement_node benchmark_grounding \
  --api-server-url http://10.10.10.2:18091/v1 \
  --image "$HOME/grounding_input.png" \
  --target 'the cup' --timeout 120
```

The optimized server listens at `10.10.10.2:18091`. Localhost
on this ROS client does not address the remote PC. The benchmark prints elapsed
response time, normalized box, pixel box, usage and server timings when supplied.
Non-streaming requests measure complete-response latency; they cannot measure
first-token latency.

## Initial validation on 2026-10-04 (before optimized deployment)

- Builds passed for vlm_node, placement_node and master_node.
- AnyPlace tests: 40 passed. C++ grounding tests: 2 passed.
- Master tests: 20 passed, 1 skipped; existing flake8 check failed (503 findings).
- Local HTTP fixture exercised the rebuilt ROS Chat service with original
  1280×720 images, image-first content, selected settings, IP/port configuration,
  valid-box acceptance and invalid-box rejection. No hardware was started.
- Remote `/v1/models` and `/health` succeeded at 10.10.10.2:1234.
- One original snapshot request targeting 'the cup' timed out after 120.118 s
  without a response. The server slot showed seed 1234, max_tokens 48,
  ignore_eos false, stream false and zero decoded tokens when inspected.
- Port 18091 refused connections on the documented candidate hosts. A second
  request was not queued because the remote slot was still processing.

## Optimized deployment validation on 2026-10-04

The optimized endpoint is now the default in the VLM node, placement node,
reasoning stack launch, and standalone benchmark.

- `GET http://10.10.10.2:18091/health`: HTTP 200, `{"status":"ok"}`.
- Original image: `src/mock_hardware_node/snapshots/current/rgb_preview.png`,
  640×480, JPEG encoded without resizing. Target: `the cup`.
- Returned normalized box: `[200, 18, 400, 272]`.
- Placement crop pixel box: `(128, 9, 256, 131)`.
- Measured complete-response latency: **21.248 seconds**.
- Server timings: prompt processing 13.723 seconds, decoding 7.332 seconds;
  259 prompt tokens, 20 completion tokens, zero cached tokens.
- Non-streaming mode was retained, so first-token latency was not measured.
- Builds passed for vlm_node, placement_node and master_node.

- Regression checks: 40 AnyPlace tests, 2 C++ grounding tests, and 19 master
  functional tests passed. Python compilation and documentation whitespace
  checks passed.
- Local HTTP/ROS integration passed with original 1280×720 images and both
  valid-box acceptance and out-of-range-box rejection.

No remote server configuration, authentication, or hardware behavior was changed.
