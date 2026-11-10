## Observation-retention update — 3.4.2

Shared capture attempts exact VAMP source retention before responding. Inspect
`observation_retention.grasp_retained`: true=confirmed, false=failed,
null=unreported. Explicit pins in examples remain idempotent confirmations.
Pin requires `observation_header <- capture.cloud.header`; grasp/placement pins
are independent. Missing components get bounded idempotent delivery recovery;
failures report component presence/counts. Capture success is not grasp readiness.
Physical queues/gates, source association and unknown-state barriers are unchanged.

## Dependency discovery update — MCP 3.3.0, 2026-11-09

Read `ros-pipeline://tool-dependencies` and the structured
`ros-pipeline://tool-dependency-graph`. Every tool's `_meta.rosPipeline` and
model-visible description now include prerequisites, execution sequence and
binding examples. Required, recommended, internal and alternative routes differ.
Failures retain their original status and gain recovery guidance; unknown-state
blocking and the actual mode allowlist remain unchanged. Refresh tool discovery
on reconnect. See [dependency guide](TOOL_DEPENDENCIES.md).

Current deployment uses preparation_only (53 tools), not the full physical mode
below. Read ros-pipeline://preparation first. No execution stage in this reference
is callable in preparation mode; fresh feedback is currently missing.

# Agent-built manipulation pipeline — MCP 3.2.0

Full agent mode exposes 63 tools: the previous 20 plus 40 calculation/perception/memory
functions and three master pipeline operations. Default mode is `agent_pipeline`.
The master remains the only planner, task store and robot executor. No physical
commands are sent on connection. `--read-only` retains the previous ten callable
inspection/view-calculation tools and disables physical action cards; preparation
mutation tools are absent in that mode.

Use `list_functions` and `ros-pipeline://catalog` for the actual running allowlist,
exact JSON schemas and `_meta.rosPipeline` request/response schemas. A ROS catalog
entry alone does not grant invocation. Legacy actuator shortcuts, arbitrary
trajectory/CDR imports, resets and physical alternatives remain unavailable.

## Complete pick/place sequence for one object

Use this same route for each object. Retain the destination's geometry and occupied
tray slots between objects; do not reset the tray for a new source object.

| Order | Agent calls / decision | Required result before continuing |
|---|---|---|
| 1 | list_functions, get_world_state, get_arm_state, get_spatial_context | Correct mode; known robot state. If no spatial observation exists, capture first. |
| 2 | capture_observation → ensure_scene_memory | Bind image/cloud/manual transform from the capture. Supply object_names, active_object_name and destination_name; inspect each observed/target_statuses entry. |
| 3 | check_target_view for the source | READY permits reuse of the current view. LOCATION_UNKNOWN requires perception; LOOK_REQUIRED requires a new view. |
| 4 | When needed: plan_next_best_view → guarded move_to | Plan only first. After an authorized, confirmed empty-hand move, capture a NEW frame; do not reuse pre-move image/mask geometry. |
| 5 | Pin fresh grasp/placement observation; ground_object → segment_object → store_segmented_target(role=object) | The same synchronized source image, mask, cloud and manual camera transform. Combined refresh_active_target is an alternative. |
| 6 | Reuse valid active destination; otherwise ground/segment/store it with role=destination | Keep existing placed geometry and slot reservations. Source and destination masks must match the observation used by planning. |
| 7 | prepare_grasp_input → calculate_grasps or calculate_active_grasps → update_object → select_grasp → cache_grasp_plan | Validated candidate pool, exact source header/cloud and masks. Cache the whole pool, not only the preliminary winner. |
| 8 | find_placements or find_active_placements | Placement candidates from available tray slots. No cell commitment yet. |
| 9 | plan_next_best_view for the prepared source look stage → prepare_pick_place → validate_prepared_pick_place | Complete frozen grasp/lift/place/withdrawal/retreat plan; no actuator calls. |
| 10 | get_execution_state; review plan | Valid execution_ref/revision/source, physical gate enabled, no unknown state. Preparation alone is not permission to execute. |
| 11 | execute_plan(look when listed, then pregrasp) → gripper_open → execute_plan(approach) | Confirm each completed stage in order. |
| 12 | gripper_close → execute_plan(lift) | success AND contact_confirmed for this attempt before lifting. |
| 13 | execute_placement_transfer → execute_placement_descent | Completed transfer and descent; stop on any uncertain outcome. |
| 14 | release_placed_object → commit_floor_slots | Confirm release; commit exact frozen selected_pose, reset=false, validate_only=false. |
| 15 | execute_withdrawal → execute_placement_retreat | Use destination name with original execution_ref/revision. Confirm completion before starting the next object. |

Do not run both alternatives in a row redundantly. Exact field names and bindings
are described below and in the catalog. get_execution_state.waiting_stages is the
runtime authority; never execute a guessed list against a different session.

There are two distinct grids: **30cm spatial cubes describe the robot workspace**;
**floor_slots describe placement reservations on the tray**. A spatial cube ID is
a view target, never a tray slot reservation or an object placement pose.

## Calling a calculation API

All 40 named preparation tools take this envelope:

```json
{
  "context_id": "",
  "step_id": "observe",
  "request_id": "task1_observe",
  "arguments": {},
  "bindings": [],
  "expected_revision": 0
}
```

An empty context creates an owner-scoped master context. Copy returned context_id
into every subsequent call. Use a new step_id per calculation and a unique durable
request_id per operation. expected_revision=0 uses current revision; a nonzero
revision must match. Pass primitive ROS request fields inside arguments. Message
fields, images, masks, clouds, candidate pools and source headers MUST come from
prior successful results in this context through typed bindings. Never send raw
CDR, invented clouds/poses, or copy model-sized byte arrays into the prompt.

For example, after capture_observation with step_id=observe, ground_object is:

```json
{
  "context_id": "COPY_RETURNED_CONTEXT_ID",
  "step_id": "ground_bowl",
  "request_id": "task1_ground_bowl",
  "arguments": {"target": "small white bowl at bottom left"},
  "bindings": [{"target_field": "image", "source_step_id": "observe", "source_field": "image"}]
}
```

segment_object binds image from observe.image and bbox from ground_bowl.pixel_box.
store_segmented_target takes arguments name and role (`object` or `destination`),
binds image/observation_cloud/camera_transform from the SAME capture and mask from
the segment result. It extracts organized mask-selected measured XYZ and stores
active memory without another VLM call. It requires mono8 mask, >=80 selected
pixels, matching dimensions/stamps, base_link cloud and synchronized manual FK*X.
It preserves existing destination occupancy/placed geometry when reusing that
destination; switching destinations follows existing memory semantics.

Outputs contain success, status, context_id, revision, step_id, function_name,
seconds, result_ref and output (actual ROS response projected to JSON). Image/cloud
fields expose dimensions/header/result_ref instead of data. Large arrays (>32)
and large outputs expose count/omitted/ref. A result_ref is a binding source, not
an actuator permission. source_field=`@response_cdr` means the full trusted ROS
response retained on Ubuntu; use it only where an API explicitly expects that
response, e.g. prepare_pick_place.look_plan_cdr from plan_next_best_view.

## Exact ordered dependencies

Alternative APIs within a row are routes, not instructions to call everything.
All these calls finish their ROS service operation before returning. No movement
occurs in this table. get_active_target/check_target_view and cached-plan getters
read current geometry; the adapter still stores their typed response as a step.

| Tool(s) | Prerequisite / output used next |
|---|---|
| capture_observation | Camera, joints and calibration available; produces image, cloud and manual camera_transform. |
| pin_grasp_observation, pin_placement_observation | Bind observation_header from capture.cloud.header. |
| verify_presence | Bind capture.image; target string; require success and present. |
| ground_object | Same image and target; returns pixel_box. |
| segment_object | Same image plus ground.pixel_box→bbox; returns mask. |
| store_segmented_target | Same capture image/cloud/transform + segment.mask; name/role; returns named cloud, position, mask, source headers. |
| resolve_target, refresh_active_target | Combined memory/perception alternative; name, role, optional synchronized capture bindings; returns identical target data. |
| get_active_target | Existing exact name/role; read target handles and geometry_source_header. |
| perceive_scene, ensure_scene_memory | Configured object_names/destination_name and optional capture; batch grounding or reuse; inspect actual response for targets found/missing. |
| check_target_view, require_target_view | Known target location + camera geometry; visibility/FOV/raycast, no VLM or motion. LOOK_REQUIRED is not permission to skip view acquisition. |
| plan_next_best_view | Known target_name or bound target PointStamped; returns camera/TCP pose and path_flat. For an operator-authorized view move, pass the projection's endpoint_joints (the final six path_flat values) to guarded empty-hand move_to; then capture/refresh again. Never use stale pre-view geometry. |
| prepare_grasp_input | Capture image/header + source mask + box; returns seed_u/v, image_width/height, affordance_type. |
| calculate_grasps, calculate_active_grasps | Same header, source mask, seeds/dimensions/affordance; target_description; returns validated candidates, corrected_seed_base and validated_target_mask. |
| update_object | name/detected + position.header from capture, position.point from grasp.corrected_seed_base. |
| select_grasp | Bind entire grasp candidates; provisional ranking, not full-task readiness. |
| cache_grasp_plan | object_name, ttl_seconds; bind whole candidate pool, capture header/cloud, validated source_mask and original semantic_mask. |
| get_cached_grasp | Existing object cache; inspect expiry/source; does not prepare paths. |
| find_placements, find_active_placements | Object cache and destination mask/header; object/destination names, support_mode=floor_slots, orientation_policy=upright; returns source_object_center and placement candidates, builds grid without committing occupancy. |
| prepare_pick_place | object/destination names, support_mode, lift_distance_m; bind header, placement source center/candidates, destination_mask and optional look_plan_cdr. Returns selected_pose and plan_revision only after complete approach/lift/transfer/descent/withdrawal/retreat validation. |
| get_prepared_pick_place, validate_prepared_pick_place | Exact object_name; validate cached source/start/readiness; no actuation/consumption. |
| prepare_grasp_lift, get_prepared_grasp_lift, validate_prepared_grasp_lift | Lift-only preparation/inspection, not a complete remote pick/place session. Current guarded payload execution requires the complete pick/place route. |

These lower-level non-actuating APIs are also exposed for bounded adjustment;
prefer prepare_pick_place for a consistent complete plan rather than manually
rebuilding its internal validation sequence:

| Tool | Required predecessor data / meaning |
|---|---|
| prepare_held_scene | Source observation, grasp and object geometry; immutable held/environment scene. |
| search_placements | Bound prepared source/destination geometry; candidate search only. |
| validate_placements | Bound candidates/source; full validation via existing recipe. |
| plan_final_approach | Frozen observation + selected grasp; returns complete validated approach. |
| plan_lift | Frozen held scene and start/grasp joints; lift or empty-hand withdrawal calculation. |
| plan_placement | Frozen held scene, source grasp and placement candidates; validates existing lift/transfer/descent. |
| plan_motion | Measured start and requested goal joints, frozen scene; VAMP path calculation. |
| solve_pose | Bound PoseStamped, seed joints/scene; endpoint IK. ignore_collision=true is forbidden, including via bindings. |
| plan_camera_retreat | Bound frozen scene/release state and destination; retreat calculation. |

Exact fields, defaults and response schemas are in each tool's metadata. Width,
translation and distance are metres; joints, yaw/pitch/roll are radians. Affordance
is AUTO/HANDLE/RIM/BODY. Support modes are auto/edge/stack/floor_slots; orientation
upright/free. Respect explicit message headers (manipulation geometry base_link,
pixels in the associated image); never introduce TF or another scale correction.
Invalid geometry/IK/path remains a reported failure, not fabricated action success.

## Execution after preparation

Read get_execution_state(object_name). Require a complete prepared plan, nonempty
execution_ref, matching revision/source, no unknown state and physical gate true.
The agent invokes EACH guarded action itself in first-waiting-stage order:

1. execute_plan(segment=look), when present, then pregrasp.
2. gripper_open → execute_plan(approach) → gripper_close.
3. Require success AND contact_confirmed, then execute_plan(lift).
4. execute_placement_transfer → execute_placement_descent.
5. release_placed_object; require success AND gripper_released.
6. commit_floor_slots with exact frozen selected_pose, reset=false, validate_only=false.
7. execute_withdrawal → execute_placement_retreat (destination name, original ref).

Every action requires request_id and exact execution_ref, except empty-hand
move_to (target_joints[6] + request_id). Preparation does not authorize automatic
execution. Execution calls preserve existing ROS-shaped arguments and structured
transport receipts. A remote model cannot skip/reorder contact checks. Destination
geometry/reservations survive completed placement for the next object. No physical
retry/alternative is available after an uncertain action. See WORKFLOW.md.

## Queue, cancellation and recovery

run_step takes the same envelope plus function_name from the preparation allowlist.
submit_steps takes context_id, request_id, expected_revision and an ordered steps
array (step_id, function_name, arguments, bindings). It only queues: QUEUED_ONLY.
run_steps(operation=start) needs context_id, durable request_id and exact queued
expected_revision; optional stop_before_step_id pauses before that step. It returns
RUNNING/ASYNC_ACK_ONLY, not completion. Poll operation=status with same context/id
until FINAL_ACTION_RESULT and inspect ros_action_status and every result. Status
4=ROS succeeded, 5=cancelled, 6=aborted; a step-boundary pause is not full completion.
operation=cancel requests local ROS action cancellation between steps, not emergency
stop or interruption of an in-flight service. Physical calls in generic run_step
or queues are independently rejected. Use guarded action tools individually.

A definite calculation failure permits an explicit new step/request, using valid
unchanged source bindings or a fresh capture after source expiry. Same-ID repeats
retrieve receipts, never redispatch. Different arguments under an existing ID are
rejected. Generic queues stop on failure. Do not blindly restart failed physical
steps. Async disconnect does not stop the Ubuntu job: reconnect and poll the same
job; never create a replacement. Restart during an active calculation/action or
an ambiguous timeout establishes a persisted preparation barrier until local
reconciliation, preventing stale background cache writes racing later actuation.
It has no remote reset. Master cancellation/watchdogs remain independent.

Service waits: named/run_step calculations 600s, submit 10s, context inspection
30s; action final wait 600s. Context storage and request history are bounded; keep
one context per object calculation, reuse it for related retries, and arrange
local cleanup of completed calculation contexts before store capacity is reached.
Do not clear active contexts or replace the execution ledger. Stored images/clouds
remain on Ubuntu; current projection caps are 64 KiB per step, 512 KiB per context.
A successful build/mock plan does not validate physical tracking, contact or safety.


Shared spatial sensing is documented in [SPATIAL_MAP.md](SPATIAL_MAP.md), exposed
as ros-pipeline://spatial-map. Read get_spatial_context/get_spatial_map before
reasoning about cell scan age or camera coverage. The fixed 30cm summary grid has
16 cells in base_link; observed empty cells are included through bounded depth
rays. plan_next_best_view now accepts target_cell_id instead of a named/point
target; existing target modes remain unchanged. No view command moves the arm
without separately authorized guarded execution.
