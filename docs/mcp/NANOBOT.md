Task responses (MCP 3.6.2): read validation.outcome, checks_passed and
checks_bypassed; PASSED_WITH_BYPASSES certifies only reported checks that ran.
execution_allowed stays false. Historical results do not confirm scene freshness
or physical readiness. Terminal current_step is null; use last_successful_step
and failed_step. describe_workflow returns a derived step_count and labelled
object/destination steps alongside its existing route. Refresh output-schema caches.

Camera-view schema update (MCP 3.6.1): refresh tools/list, task-catalog and the
Mac translated/profile schema caches. plan_camera_view now uses a flat schema;
mode requirements are described per field and enforced server-side. Read
result.status/achieved_fraction: NO_REACHABLE_CHANGE with zero fraction is a
finished calculation with no changed view, not movement success.

## Optional task profile — MCP 3.6.0

With `--preparation-only --task-tools`, prefer the eight task tools in
[TASK_INTERFACE.md](TASK_INTERFACE.md). Choose intent with `observe_scene` or
`prepare_manipulation`; the server owns capture, pinning, bindings and sequencing.
Use one fresh request_id per new operation, reuse it only with identical inputs,
and inspect RUNNING/UNKNOWN tasks with get_task_status. Never manufacture success
from queue admission. These tools calculate only; they cannot manipulate hardware.
The existing 54-tool diagnostic surface remains unchanged and separately selectable.
Exact native schemas: [TASK_TOOLS.json](TASK_TOOLS.json).

# Nanobot quickstart — MCP 3.5.1

For a new grasp task, **call get_workflow_example first**. The helper is read-only:
it returns proposed calls with exact fields and fresh IDs; it executes nothing.
This page is available as `ros-pipeline://nanobot`. Read it before capturing.

## Scene-perception example

For batch scene understanding call `get_workflow_example` with
`{"workflow":"perceive_scene","object_name":"beige cup","destination_name":"tray"}`.
It returns two proposed calls, without execution: capture with `context_id:""`,
then `perceive_scene`. Copy capture's top-level `context_id` (native JSON-RPC
`result.structuredContent.context_id`) into the second call. Bind its image,
observation_cloud and camera_transform from the SAME capture step's `image`,
`cloud` and `camera_transform` outputs. The helper generates fresh IDs for new
operations; use old IDs only to retrieve identical receipts, never to replay an
unknown action. No external instruction file is needed to read these schemas.

`SCENE_INFERENCE_UNKNOWN` means the VLM reply could not establish a trustworthy
scene result; it does not mean objects are absent or prerequisites were missing.
The parser accepts one complete keyed object or an exact key/value record array,
with optional single JSON fence. Duplicate/missing/extra keys, conflicting or
multiple documents and invalid value types fail before scene-memory updates.

## Capture retention: check the evidence

Shared `capture_observation` now attempts to retain its exact RGB/cloud/manual-FK*X
bundle in VAMP **before returning**. MCP reports
`observation_retention.grasp_retained`: true=confirmed, false=failed,
null=not reported by that backend. Capture success alone means image capture,
not grasp readiness. The explicit pin in examples is an idempotent confirmation;
keep it immediately after capture and proceed only after it succeeds.

Pin operations require a typed `observation_header <- capture.cloud.header`
binding. The server rejects missing bindings before planner dispatch. Grasp and
placement pins are independent, each referring to the same capture when needed.
The master retries only known missing-frame/pending pin replies at bounded
50/100/200/400 ms delays, without recapturing or changing the source timestamp.
An unresolved pin reports requested timestamp and per-component presence/counts.
Do not label this eviction, model delay, or a robot planning failure without evidence.
Do not mark downstream tools failed when their prerequisites were unavailable.

## First call

Call `get_workflow_example` with:

```json
{"workflow":"grasp","object_name":"cup","affordance_type":"AUTO"}
```

Use `grasp_cache` to include caching/checking, or `pick_place` to calculate the
complete cup-to-destination plan. These examples contain **no physical dispatch**.
Optional `context_id` reuses an existing gateway-owned context. Optional
`destination_name` selects the destination for pick/place.

Follow returned `steps` in order:

1. Call `step.tool` with exactly `step.input`. Do not invent or rename fields.
2. If the first capture uses empty context_id, copy its returned context_id over
   `CONTEXT_ID_FROM_CAPTURE` in every later input. Use the generated step/request
   IDs and source_step_id values unchanged.
3. Pin **immediately** after capture. Do not perform catalog discovery, resource
   reading or long reasoning between capture and pin.
4. Require success=true before proceeding. Stop and preserve the full failure
   result otherwise. Unknown outcomes never permit automatic replay or reset.

The `grasp` example is exactly:

**capture_observation -> pin_grasp_observation -> resolve_target ->
calculate_active_grasps -> select_grasp**

`resolve_target` uses the full captured image/cloud/transform triplet. Active
calculation uses the exact stored target_description and empty bindings, preparing
its own grasp inputs internally. The pin binds observation_header from the
capture's cloud.header, never from image. Same context does not imply same frame.

## Choose the requested task only

- **Grasp preview:** use the `grasp` example; stop after `select_grasp`.
- **Pick and lift:** use `grasp_cache`, then `prepare_grasp_lift` and
  `validate_prepared_grasp_lift`. No destination or placement search is required.
- **Pick and place:** use `pick_place`; it includes destination/slot calculation,
  `prepare_pick_place` and `validate_prepared_pick_place`.

For either preparation route, bind `cache_grasp_plan.candidates` from the full
`calculate_active_grasps.candidates` response, not only `select_grasp.candidate`.
Selection is provisional: preparation tries up to eight cached candidates, with
30 seconds checked between attempts, and chooses a complete successful plan.
Placement rejection can advance to the next cached candidate during calculation.
No automatic alternative is permitted after physical failure/unknown outcome.
A cached grasp is not a prepared motion. Never claim placement was checked by
`prepare_grasp_lift`, or movement occurred from a preparation success.

## Keep the model context small

For routine status use `list_functions` with `{"summary_only":true}`. Its default
full response is a large developer reference and is not needed during a normal
workflow. Keep full schemas/metadata in the adapter, not repeatedly in the model's
conversation. Tool input definitions need only name, short description and
inputSchema; retain outputSchema/_meta for validation and on-demand reference.

For a grasp-only task, Nanobot should expose this small set to Qwen:

```text
get_workflow_example
list_functions
get_world_state
get_step_context
capture_observation
pin_grasp_observation
resolve_target
calculate_active_grasps
select_grasp
```

This is a suggested **client-side context filter**, not a server safety boundary.
The server remains authoritative for availability, provenance, state and dispatch.
Expand the client tool set only when the chosen workflow requires it. Preserve
any existing connector namespace mapping, e.g. ros2::get_workflow_example; binding
field names and source_step_id values are not namespaced.

## Error meanings

| Error | Meaning / response |
| --- | --- |
| MCP_CONTEXT_OWNER_REQUIRED | Do not invent context IDs. New capture uses empty context_id; later calls use its returned ID. |
| REQUEST_ID_REUSED / STEP_ID_PAYLOAD_CONFLICT | A changed operation needs a new ID. For an uncertain operation retain its original ID and inspect the receipt, not a fresh replay. |
| LAYOUT_SYNCHRONIZED_OBSERVATION_NOT_AVAILABLE | The exact RGB-D triplet cannot be pinned. LAYOUT is a legacy API name, not a special scene requirement. Stop; do not guess another field. |
| OBSERVATION_NOT_IN_CACHE | The requested source is no longer retained. Extra seed bindings cannot restore it. A new explicit capture/pin/perception attempt needs safe known state and new IDs. |
| ACTIVE_OBJECT_REQUIRED | Check the exact target_description against the stored name; mask/header bindings do not select the object. |
| NO_FRESH_CACHED_GRASP | Preparation/cache may be missing, failed, expired or invalidated. Read observed producer results and proposed recovery_steps. A getter creates no cache. |
| STATE_UNKNOWN / unresolved timeout | Stop. Preserve receipt/ledger and require local reconciliation; do not reprepare to bypass the barrier. |

If an immediate correct pin fails, report its capture ID/header, exact binding and
status. This may be ROS delivery/synchronization, not an agent-argument error.

## Limits and full reference

A valid selected grasp is not a validated lift/placement or permission for motion.
Preparation-only mode never permits arm/gripper execution. Read actual server
availability rather than old deployment notes. Template steps marked
callable_in_current_mode=false must not be dispatched.

For uncommon calls read `ros-pipeline://tool-dependencies`,
`ros-pipeline://tool-dependency-graph`, `ros-pipeline://catalog`,
`ros-pipeline://agent-pipeline` and `ros-pipeline://workflow` on demand. Required,
recommended, internal and alternative dependencies are different. Images/clouds
stay on Ubuntu; preserve source headers, revisions, TLS/authentication, durable
IDs, local cancellation/watchdog behavior and unknown-state barriers.

## Grasp cache lifetime

Grasp caches and unstarted prepared plans default to 15 minutes (900 seconds).
Use `ttl_seconds: 900` or omit/zero that field for the default; shorter explicit
values are honored. Preparation and validation do not consume the grasp. Once
prepared execution is admitted, the public source cache is removed; the frozen
execution session retains its source privately. Never retry an unknown action.
The separate active-execution watchdog and all live-state checks still apply.

## Mac profile and discovery refresh

Add native tool name `get_workflow_example` to the ROS MCP server's
`all-calculations` allowlist. If that profile stores connector-qualified names,
the existing `ros2::` namespace gives `ros2::get_workflow_example`. This is a
local read-only MCP helper, not a ROS catalog function; do not require a
`/pipeline/get_workflow_example` endpoint to discover it.

Reconnect and refresh initialize/server metadata, native `tools/list` (including
input/output schemas, annotations and descriptions), the profile-filtered and
model-facing translated tool snapshots, `list_functions` snapshots, and cached
`ros-pipeline://nanobot` instructions. Do not reset execution receipts/ledger or
existing task IDs. `ros_catalog` is reference documentation, `mcp_tools` mirrors
native discovery, `allowlist` names enabled tools, and `disabled_tools` names
blocked APIs. Native discovery is authoritative for MCP tool existence.

## Recorded replay mode

The operator may start MCP with --preparation-only --require-mock-inputs on an
isolated nonzero ROS domain and a dedicated mock ledger. It verifies one
/mock_hardware publisher for every required camera/feedback topic; mixed,
live-only, missing and domain-0 sources are rejected. All results carry
SIMULATED=true, input_mode=mock_replay and input_provenance. They are recorded
perception/planning, not live observations, physics or physical execution.
The normal --require-live-inputs guard remains unchanged. On a source-mode
switch refresh discovery and start a fresh context; do not reuse live task IDs
or reset the live execution ledger. No motion/gripper dispatch is exposed.
