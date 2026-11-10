## MCP 3.6.0 — optional task-level profile

Add `--task-tools` only in preparation_only mode to expose eight intent tools
alongside the unchanged 54 diagnostics tools. Capture, producer bindings and
workflow order are server-owned. No physical dispatch is enabled.
See [task interface and Mac allowlist](TASK_INTERFACE.md) and
[exact native schemas](TASK_TOOLS.json). Mock results remain explicitly simulated.

## MCP 3.5.1 — explicit recorded-scene preparation

Current replay validation runs on isolated domain 89 with real drivers stopped.
`--preparation-only --require-mock-inputs` requires mock-only publishers and a
dedicated mock ledger. The production live guard, TLS/auth and physical gates
are unchanged. Results are explicitly SIMULATED=true / input_mode=mock_replay.
Authenticated cup+tray and bowl+cup+tray perception passed (2/2 and 3/3), including
MCP/ROS structured validation. This is recorded perception, not physical execution.
See [startup and source guard](PREPARATION_ONLY.md) and [evidence](MOCK_VALIDATION.md).

## MCP 3.5.0 — scene perception

The scene parser now handles the complete Qwen JSON response, including exact
key/value record arrays, and rejects ambiguous/conflicting output as unknown.
`get_workflow_example(workflow="perceive_scene", object_name="cup",
destination_name="tray")` provides a schema-only capture->perception example.
Initial context_id is empty; later calls use result.structuredContent.context_id.
Native mode remains preparation_only with 54 tools and physical gate false.
Live result was SCENE_INFERENCE_UNKNOWN: PRESENCE_GROUNDING_CONFLICT on a nearly
black RGB frame; structural validation passed, positive recognition did not.
See the current validation report and journal; older deployment sections below
remain historical evidence. Refresh native/profile discovery and resource caches.

# Small-model workflow support — MCP 3.4.0

For Qwen3.5-9B/Nanobot, call the new local read-only `get_workflow_example` tool
first. It supplies exact ordered grasp/cache/pick-place request templates, with
fresh IDs and the capture-to-pin binding. It executes nothing. Model-visible
preparation descriptions are about 81% shorter; full metadata remains available.
`list_functions(summary_only=true)` returns a small mode/allowlist snapshot while
its default full response remains unchanged. Preparation-only mode now has 54
tools: the existing 53 plus this documentation helper. Physical access is unchanged.

Start with [Nanobot quickstart](NANOBOT.md) and the
[adapter instructions](NANOBOT_ADAPTATION_PROMPT.md). Earlier deployment reports
below are historical; query the actual mode and tool availability.

# Source-retention and active-name guidance — MCP 3.3.2

Pin the captured header before long inference calls and keep all seed/mask/header
bindings on one source. Active grasp calculation explicitly requires the exact
stored name in target_description. See the [source-cache recovery guide](TOOL_DEPENDENCIES.md#observation-cache-errors-pinning-and-source-identity).
No cache size, planner/collision setting, physical gate or actuator path changed.
Refresh discovery after updating MCP.

# Cache-recovery guidance — MCP 3.3.1

Missing-cache errors now include exact valid field names, same-context proposed
recovery requests and observed producer results when available. Active-grasp
calculation's internal input preparation is documented. Read
[the recovery section](TOOL_DEPENDENCIES.md#grasp-cache-recovery-use-actual-fields-not-inferred-names)
and refresh tool discovery. Input names, cache TTL, mode gates and durable receipt
semantics are unchanged.

# Dependency-aware discovery — MCP 3.3.0

Tool schemas, callable allowlists and safety semantics are unchanged. Each tool
now documents prerequisites, real producer bindings and a recommended sequence.
Read [the dependency guide](TOOL_DEPENDENCIES.md), available through MCP as
`ros-pipeline://tool-dependencies`; the generated structured graph is
`ros-pipeline://tool-dependency-graph`. Restart only MCP to load this update;
keep its existing mode, ROS domain, TLS/authentication and execution ledger.
Earlier deployment notes below are historical observations, not live status.

# Current deployment: non-actuating preparation

MCP 3.2.0 is running in preparation_only mode with 53 tools, live-input provenance
checks and the master physical gate disabled. Read
[preparation-only tools and verification](PREPARATION_ONLY.md), available through
ros-pipeline://preparation. Physical dispatch is independently blocked, including
generic queues. Current complete-preparation tests are blocked by stale feedback;
see the report rather than treating transport discovery as motion readiness.

The earlier perception-only and full-agent configurations below remain separate
modes; neither is enabled by this current startup command.

# Current deployment: live Qwen perception only

MCP 3.1.1 now uses live domain-0 camera/arm feedback with physical execution
DISABLED. It exposes 11 perception/inspection tools and requires verified single
live input publishers. Read [live-perception setup and evidence](LIVE_PERCEPTION.md)
(or ros-pipeline://live-perception). Full manipulation mode below is a separate
configuration; it is not enabled in this session.

# MCP 3.1.0 — remote agent manipulation

The remote agent can now compose the complete supported pick/place workflow.
**62 tools** expose perception, mask-to-cloud memory, grasp/slot calculation,
whole-plan preparation, and individually guarded execution. Images/clouds remain
on Ubuntu and use same-context typed result bindings. No autonomous physical retry
or uncertainty reset is exposed.

Read [Agent pipeline and examples](AGENT_PIPELINE.md),
[Execution rules](WORKFLOW.md), [Nanobot instructions](NANOBOT.md),
[Native-agent adaptation prompt](NANOBOT_ADAPTATION_PROMPT.md), and
[Mock validation](MOCK_VALIDATION.md). Exact current schemas are generated by
list_functions and ros-pipeline://catalog.

## Connection and guarded execution

The server retains seven inspection tools, get_observe_view and ten guarded motion APIs,
alongside the 42 preparation/queue tools described above. It calls the existing master services; it does not calculate new paths or
replace the master queue/planner. Physical operation still requires the master's
`allow_physical_grasp_execution=true` and a complete fresh prepared pick/place.
The currently running deployment uses **mock ROS domain 89**, physical gate false.
No real motion is performed as part of development or validation.

- [Exact workflow, inputs, outcomes and recovery](WORKFLOW.md)
- [Placement pipeline coverage and missing MCP APIs](PLACEMENT_API_AUDIT.md)
- [Reusable Nanobot instructions](NANOBOT.md)
- [Native macOS adaptation prompt](NANOBOT_ADAPTATION_PROMPT.md)
- [Earlier calculation-only mock validation](MOCK_VALIDATION.md)
- Exact live schemas: `ros-pipeline://catalog` or `list_functions`.

## Connection and startup

Private TLS endpoint: `https://10.211.55.3:8765/mcp`.
Server name `ros-pipeline`, version `3.1.0`, mode `agent_pipeline`.
IP is environment-specific. Keep the bearer token/private key secret; trust the
public certificate on the client and never disable certificate verification.

```bash
cd ~/agx_piper_stack
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export PYTHONPATH="$HOME/agx_piper_data/deps/ros_mcp:$PYTHONPATH"
export ROS_DOMAIN_ID=89  # current mock; deliberately choose real stack's domain later
ros2 run task_pipeline pipeline_mcp \
  --host 10.211.55.3 --port 8765 \
  --token-file "$HOME/.config/agx-piper-mcp/token" \
  --tls-cert "$HOME/.config/agx-piper-mcp/server.crt" \
  --tls-key "$HOME/.config/agx-piper-mcp/server.key"
```

This command starts no hardware, camera, controller or reasoning stack and sends
no motion on startup. All actuation is operator-task-driven. Use Ctrl+C to stop
MCP; that does not stop an in-flight ROS service or the arm. The existing local
emergency-stop/watchdog paths retain control. `--read-only` restores ten non-actuating tools (nine reads and view calculation) plus ten disabled action cards. No private/public wildcard or
plaintext private-network listener is accepted.

Optional dependencies: `python3 -m pip install --target
"$HOME/agx_piper_data/deps/ros_mcp" -r docs/mcp/requirements.txt`.
Token/public cert/private key are in `~/.config/agx-piper-mcp/`; token and key
remain mode 600. Configure `Authorization: Bearer <token>` on the client.

## Execution boundary

`get_execution_state` returns the exact `execution_ref` to attach to every action,
plus gateway_status and the frozen selected_pose once a session has been admitted.
Every action needs a unique durable request_id. move_to is an empty-hand joint-target
operation without execution_ref; it cannot interrupt an active/unknown task. The gateway persists admission
and results in `~/.local/state/agx-piper-mcp/execution.sqlite`. Never delete or
change this ledger to retry a possibly executed command. Reconnect/process restart
cannot reset its ownership, count or uncertainty. One gateway process owns it.

Prepared payload actions support only a complete prepared pick/place. The
separate move_to operation is empty-hand only. Start at its first waiting
stage; you cannot adopt a partially executed local session. Master feedback,
contact, source/revision/expiry, collision and driver checks remain active.
Placement requires contact confirmed by that task's gripper_close response.
Duplicate IDs return historical receipts without dispatch; changed arguments
under the same ID are rejected. Any dispatched failure, timeout, cancellation or
incomplete response blocks further actions until local administrative review.

Currently **one physical grasp attempt per prepared task**, stricter than the
three-attempt ceiling. No automatic alternative/retry is enabled because the
underlying APIs do not supply an unambiguous recoverable physical failure. Remote
reset/reconciliation and generic movement are unavailable. All history is bounded
at 10,000 records; reaching capacity fails closed instead of erasing evidence.

## Relative camera view and move_to

Call get_observe_view with six inputs: dx,dy,dz in metres and roll,pitch,yaw in
radians. Axes are the current optical camera frame: +X right, +Y down, +Z forward.
Rotation is Rcamera*Rz(yaw)*Ry(pitch)*Rx(roll). The function captures synchronized
manual FK*X, then C++ calculates the camera/TCP target, solves bounded relaxed IK
and checks the full arm/gripper endpoint against a frozen scene and self-collision.

Example: 2 cm up and right, no rotation:

```json
{"dx":0.02,"dy":-0.02,"dz":0.0,"roll":0.0,"pitch":0.0,"yaw":0.0}
```

Only when response.success, moved_target and endpoint_safe are all true, pass its
six target_joints to move_to with a new request_id. move_to asks VAMP to plan and
execute the complete path; no RPC per microstep. It still requires the real
master's physical gate, idle empty hand and no active/unknown gateway task.

TARGET_COLLISION or NO_REACHABLE_CHANGE returns current source joints, unchanged,
with moved_target=false and endpoint_safe=false. Skip movement. This does not
prove the current pose is safe. Missing/stale feedback or calibration is an error,
not a guessed fallback. See WORKFLOW.md for bounds and achieved-error reporting.

## Validation

`test_mcp_execution.py` uses a synthetic master only. It tests the full prepared
sequence, real MCP input/output schema round-trips, duplicate persistence,
contact/order gates, source/object/revision/attempt mismatch, timeout,
cancellation, restart and exhausted budgets. It cannot create ROS clients.
Real mock validation checks physical-gate rejection without invoking an actuator
API. Mock success and a writable gateway do not establish physical readiness.
Reports/logs stay under `~/agx_piper_data/{reports,logs}/mcp_execution*`.


Shared spatial sensing is documented in [SPATIAL_MAP.md](SPATIAL_MAP.md), exposed
as ros-pipeline://spatial-map. Read get_spatial_context/get_spatial_map before
reasoning about cell scan age or camera coverage. The fixed 30cm summary grid has
16 cells in base_link; observed empty cells are included through bounded depth
rays. plan_next_best_view now accepts target_cell_id instead of a named/point
target; existing target modes remain unchanged. No view command moves the arm
without separately authorized guarded execution.
