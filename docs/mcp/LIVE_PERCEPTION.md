# Live perception through Qwen — MCP 3.1.1

## Scope and observed deployment

The live-perception deployment uses ROS_DOMAIN_ID=0 for the existing Orbbec camera,
existing AGX arm feedback, reasoning stack and MCP. Domain-89 mock replay and its
reasoning/MCP processes were stopped. Hardware drivers were already running; this
procedure does NOT start controllers, enable motors, home, move or grip.

The master physical gate is false. MCP runs --perception-only --require-live-inputs:
only 11 capture/inference/inspection tools are callable. Motion/gripper tools and
generic queues are absent and independently refused. Default full-agent MCP mode
still exists for other workflows; do not use it for this non-actuating validation.

list_functions.input_provenance reports the actual ROS domain and raw DDS publisher
identities. Before capture/presence/grounding/segmentation, the live guard requires
one expected publisher per raw topic:

| Topic | Expected publisher |
|---|---|
| /camera/color/image_raw | /camera/camera |
| /camera/depth_registered/points | /camera/camera |
| /feedback/joint_states | /agx_arm_ctrl_single_node |
| /feedback/tcp_pose | /agx_arm_ctrl_single_node |

Duplicate, missing, replayed or unexpected publishers return
LIVE_INPUT_PROVENANCE_REQUIRED before dispatch. The graph check is publisher
identity evidence, not a cryptographic device certificate or freshness guarantee;
the existing capture path still checks synchronized fresh samples and manual FK*X.
If the driver node names intentionally change, review the expected identities
rather than disabling the live guard. Mock replay's existing conflict guard stays
active. Never label a mode or timestamp alone as proof that data is live.

## Ubuntu-authenticated SSH tunnel

The Qwen host keeps inference bound to its own 127.0.0.1:18080. In an Ubuntu
terminal, run and leave open:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18092:127.0.0.1:18080 \
  bincu008@192.168.50.128
```

Authenticate interactively; no password is requested/stored by this integration.
Verify a new host-key fingerprint through trusted server access. Do not disable
host-key verification. A working -NT session normally remains silent. This binds
only Ubuntu loopback and carries inference over authenticated/encrypted SSH.
The Mac's localhost forward is a different host and is not assumed reachable.

Verify FROM UBUNTU:

```bash
curl --fail http://127.0.0.1:18092/health
curl --fail http://127.0.0.1:18092/v1/models
```

Verified endpoint: http://127.0.0.1:18092/v1. Verified model ID:
Qwen/Qwen3.5-9B. Both health and model discovery returned HTTP 200 through the
operator-authenticated tunnel on 2026-11-03. Recheck after every tunnel/server
change; TCP availability alone does not verify the model or inference.

## Restart without bringing up hardware

Leave the operator's existing camera/arm-feedback drivers alone. Stop previous
reasoning/MCP/mock processes you own before launching replacements; do not start
another mock/live publisher for these topics. Source ROS/workspace in each terminal.

Terminal 2 (Ubuntu, camera and feedback already running in domain 0):

```bash
cd ~/agx_piper_stack
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=qwen_tunnel \
  placement_backend:=heightmap \
  allow_physical_grasp_execution:=false
```

The new qwen_tunnel profile selects Ubuntu loopback port 18092, model
Qwen/Qwen3.5-9B and timeout 120s. Existing server/lmstudio profiles and explicit
endpoint/model overrides retain their original semantics. No credentials are in
the profile. Inspect the VLM startup endpoint/model before running perception.

Terminal 3 (Ubuntu):

```bash
cd ~/agx_piper_stack
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
export PYTHONPATH="$HOME/agx_piper_data/deps/ros_mcp:$PYTHONPATH"
ros2 run task_pipeline pipeline_mcp \
  --perception-only --require-live-inputs \
  --host 10.211.55.3 --port 8765 \
  --token-file "$HOME/.config/agx-piper-mcp/token" \
  --tls-cert "$HOME/.config/agx-piper-mcp/server.crt" \
  --tls-key "$HOME/.config/agx-piper-mcp/server.key"
```

Keep the existing private ledger; do not delete it or reset uncertainty to change
modes/domains. New master sessions need fresh context IDs/captures; old receipts
are historical evidence, not fresh image data. Ctrl+C stops the stack/server you
launched; it is not an emergency stop. The operator owns the interactive SSH
session. Do not copy the TLS private key to the Mac or disable TLS verification.

## Minimal Nanobot tools and dependencies

For the full perception validation, enable these eight tools in the Mac client:

1. list_functions — mode/provenance/actual schemas.
2. get_world_state — verify physical execution is disabled.
3. get_arm_state — inspect current feedback and timestamp.
4. capture_observation — retain synchronized image/cloud/manual transform.
5. verify_presence — fresh visual presence on the bound captured image.
6. ground_object — bounding box on that SAME image.
7. segment_object — mask from the same image and grounded box.
8. get_step_context — inspect retained results and source associations.

Optional read-only tools: get_spatial_map, get_spatial_context,
get_manipulation_memory. These complete the server's 11-tool allowlist. They do
not provide fresh semantic visual evidence. Generic run_step/submit_steps/run_steps,
move_to, execute_plan and gripper operations are not enabled in this mode.

The Mac needs its actual Streamable HTTP MCP client, trusted public server.crt,
bearer token delivered via the existing secure channel, and these resource/tool
permissions. It does not need ROS, camera drivers, a Mac VLM tunnel, cloud bytes,
or the Ubuntu TLS private key. Preserve structuredContent and exact schemas.
Allow >=180s tool-call waits: the measured remote presence and grounding calls
were ~25.8s and ~21.5s; short client timeouts create unknown outcomes, not absence.
Ubuntu needs the existing ROS/EdgeSAM stack and optional MCP Python dependencies
from docs/mcp/requirements.txt. HGGD and physical APIs are not called in this test.

## Same-context test and evidence

Call capture_observation with an empty context_id, a unique step_id such as
capture, and a durable request_id. Reuse its returned context ID. All preparation
tools still take arguments/bindings, not raw ROS image/CDR literals.

For verify_presence and ground_object bind:

```json
{"target_field":"image","source_step_id":"capture","source_field":"image"}
```

For segment_object bind the same capture.image and ground.pixel_box→bbox. Do not
recapture between presence, grounding and segmentation when validating one image.
Keep timestamps/frame conventions: optical image coordinates; organized cloud and
manual camera_transform header in base_link; mask header matches the source image.
Grounding remains normalized [x1,y1,x2,y2] on 0..1000, converted through the existing
pixel convention. No TF lookup or new calibration conversion was introduced.

Verified on the actual live 640x480 image: Qwen returned PRESENT, normalized box
[234,300,468,625], pixel box [150,144,300,300]. EdgeSAM returned OK, 14,295 mask
pixels and score ~0.993. The memory initially contained no named objects while
fresh image-bound presence succeeded. This directly demonstrates that missing
memory is NOT visual absence. Images/clouds/source headers were retained in one
master context; no arm/gripper commands were sent.

Evidence outside the source tree, under ~/agx_piper_data:
reports/live_input_provenance.json, live_qwen_perception.json,
live_qwen_source_evidence.json, live_qwen_capture.png, live_qwen_grounding.png,
live_qwen_mask.png; logs/live_qwen_stack.log and live_qwen_perception.log.
Physical grasp/placement and native Nanobot client behavior are NOT verified.

## Unknown is not absent

get_spatial_context and get_manipulation_memory read stored locations, not the
current image. Empty names mean unknown/unregistered; an empty cube ID means
outside the summary box, not absent. Old entries are only location hints.

Only a successful verify_presence with present=true and a valid image association
establishes positive visual evidence. present=false or success=false alone does
not establish absence. PRESENCE_UNKNOWN covers empty/unparseable/uncertain
responses (including backend failures returning no text). The prompt now asks
UNKNOWN for uncertainty instead of instructing the model to call it ABSENT.
Explicit ABSENT retains the existing NOT_CONFIRMED_PRESENT failure semantics.
Timeout/unavailable/unknown transport outcomes must not be rewritten as ABSENT.
No schema or execution precondition was relaxed.
