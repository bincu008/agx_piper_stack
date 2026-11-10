Optional MCP 3.7.0 workspace profile: see [WORKSPACE_TOOLS.md](WORKSPACE_TOOLS.md).
It adds image/freshness/object-list reads and current-task defaults. The new Mac
adapter hides IDs while retaining private receipts. Existing profiles remain.
Some redesign stages remain incomplete; do not infer physical readiness.

# Nanobot task interface — MCP 3.6.2

Use the **eight task tools only** in the Mac task profile. They select fixed
server-side routes; Nanobot supplies intent/parameters and interprets results.
It does not construct contexts, bindings, step IDs or queue operations.
No task tool executes motion, actuates a gripper, commits placement occupancy,
or resets safety/uncertainty. Existing 54 diagnostic tools remain available,
unchanged; enabling this surface adds eight native tools (62 total).

## Task profile

```json
["inspect_robot", "observe_scene", "plan_camera_view", "prepare_manipulation",
 "validate_plan", "get_task_status", "cancel_task", "describe_workflow"]
```

These are native names. With the existing connector namespace, calls appear as
`ros2::observe_scene`, etc. Do not include the 54 diagnostic operations in the
model-facing task profile; keep that separate for explicit debugging.
Exact input/output schemas: MCP resource `ros-pipeline://task-catalog` and
[TASK_TOOLS.json](TASK_TOOLS.json). Human reference: `ros-pipeline://tasks`.

## Calls and semantics

`plan_camera_view` publishes a flat object schema for llama.cpp/Qwen tool calling:
only request_id and mode are globally required. Mode-specific requirements remain
strict server checks before task/receipt creation. relative requires frame and
both three-element vectors and rejects either target selector; target requires
exactly one selector and rejects frame/translation_m/rotation_rad. Finite relative
translation norm <=.20m and angles <=pi, target distance .1–1m, field types, vector
lengths and extra-field rejection remain enforced. Read result.status and
achieved_fraction: COMPLETED means the calculation finished; NO_REACHABLE_CHANGE
with achieved_fraction=0 and moved_target=false found no changed viewpoint.
It does not prove that the unchanged endpoint is safe. Reconnect and refresh
native tools/list, task-catalog and the Mac's translated/profile schemas.

| Tool | Required inputs | Result |
|---|---|---|
| inspect_robot | `{}` | TCP/joints/jaw width, original independent feedback timestamps/ages and robot state. No memory is used. Read success is not a freshness or safety certificate. |
| observe_scene | request_id, targets | detail=presence/location/mask; optional destination and wait_seconds. Fresh capture, retention and successful producer bindings are owned by the server. Returns per-target PRESENT, NOT_CONFIRMED or UNKNOWN, geometry/source time and mask handle when requested/available. |
| plan_camera_view | request_id, mode | relative: frame=camera_optical, translation_m=[right,down,forward], rotation_rad=[roll,pitch,yaw]. target: exactly one target_name or target_cell_id. Delegates the existing endpoint/next-best-view planner. Calculate only. |
| prepare_manipulation | request_id, operation, object_name | grasp caches the full validated pool and confirms it; grasp_lift also prepares/validates approach and lift; pick_place requires destination and prepares/validates complete placement. lift_distance_m defaults .05. No plan ID exists for grasp-only. |
| validate_plan | request_id, plan_id, revision | Validates that exact actual prepared plan. Checks stored source mode/domain/header and current revision before invoking existing validators, then rechecks the returned revision. Does not consume/execute it. |
| get_task_status | task_id | Read completion/progress/blocker. Optional wait_seconds (0–120) waits on the existing action completion, not a Qwen polling loop. include_diagnostics exposes original context/job/steps and receipts. |
| cancel_task | task_id | Requests existing action cancellation between steps. CANCEL_PENDING is not confirmation and cannot stop an in-flight service. Completed/unknown tasks are never replayed. Direct in-flight validation services cannot be interrupted; inspect the same task. |
| describe_workflow | operation | Local explanation derived from the actual reviewed routes; no ROS/scene calls. |

All task inputs reject arbitrary nested calls, raw CDR, cloud/image bytes and
extra fields. Names are exact semantic descriptions, not IDs guessed by the model.
Relative view translation norm is <=.20m and each angle <=pi; the existing backend
enforces its bounds. It can return PARTIAL or the exact current joints as fallback.
`endpoint_safe=false` does not establish current-pose safety. A relative endpoint
is not a full-path plan. A target next-best-view result retains the planner status;
CURRENT_VIEW_ALREADY_SUITABLE does not imply a movement path was planned.

## Natural requests and example calls

- “Where is the arm now?” -> `inspect_robot({})`.
- “Find the beige cup and its outline.” ->
  `observe_scene({"request_id":"new-observation-01","targets":["beige cup"],"detail":"mask"})`.
- “Find a grasp for the cup.” ->
  `prepare_manipulation({"request_id":"new-grasp-01","operation":"grasp","object_name":"beige cup"})`.
- “Pick up the cup and lift it 5 cm.” ->
  `prepare_manipulation({"request_id":"new-lift-01","operation":"grasp_lift","object_name":"beige cup","lift_distance_m":0.05})`.
- “Put the cup in the tray.” ->
  `prepare_manipulation({"request_id":"new-place-01","operation":"pick_place","object_name":"beige cup","destination":"dark rectangular tray"})`.
- “Look 2 cm to the right.” ->
  `plan_camera_view({"request_id":"new-view-01","mode":"relative","frame":"camera_optical","translation_m":[0.02,0,0],"rotation_rad":[0,0,0]})`.

IDs above are illustrations: generate a fresh request_id for every NEW operation.
All movement requests above **calculate only** in preparation_only mode. Do not
claim the arm picked/placed/looked merely because a plan was prepared.

## State, identity, cancellation and reconnection

Preparation and explicit validation responses include top-level `validation`:
source_mode, backend_validation_succeeded, outcome, checks_passed (exact backend
markers), checks_bypassed (check/reason/reason_source), execution_allowed=false
and execution_block_reason=PREPARATION_ONLY_MODE. Reasons are included only when
an existing backend marker contract identifies the configuration; unreported
reasons are null. PASSED_WITH_BYPASSES means only the configured checks that ran
passed; bypasses are not passes and no comprehensive collision guarantee follows.
No listed marker does not prove a check ran. Grasp-cache-only has NOT_VALIDATED
until a motion preparation/validator actually succeeds.

These are historical calculation receipts: scene_freshness_confirmed=false and
physical_readiness_confirmed=false. A successful retained receipt does not prove
the scene is still current. New validation still uses the existing exact source,
revision and live-feedback checks; unchanged request IDs retrieve the old receipt.

While RUNNING, current_step/current_step_id identify the actual first pending
step only when the existing ROS context reports busy. Missing progress is not
inferred from old data. Terminal responses set both fields to null (including
old stored receipts), retain last_successful_step and include failed_step/ID when
an actual step failed. UNKNOWN has no confirmed active step; null is not proof
that remote work or hardware stopped. Terminal action receipts take precedence
if an action finishes during a progress snapshot.

describe_workflow retains its existing function-name route and adds step_count
computed from that route plus indexed steps with function/role/label. Repeated
verify_presence, ground_object and segment_object distinguish object from
destination, following reviewed arguments and predecessor bindings.

Submission returns a stable task_id and QUEUED/RUNNING or a final state. Those
acknowledgements have success=false but are not MCP errors. Only COMPLETED is
successful completion. FAILED preserves an original step failure; BLOCKED means
a prerequisite/freshness/admission/revision condition prevented continuation.
UNKNOWN means the result cannot establish an outcome (or work was lost on
restart). CANCEL_PENDING requires a later CANCELLED or other final outcome.
Dependent steps never execute after their producer fails. Useful partial results
remain; untested targets remain UNKNOWN, not absent.

Repeat an identical submission with the same request_id to find the same task;
changing its payload is rejected. Waiting longer does not create a new task.
get_task_status reads existing work and never submits/replays it. Disconnecting
a wait does not cancel the ROS action. Gateway action receipts persist across
reconnections; a server restart marks unreconciled active jobs unknown and blocks
continuation through the existing preparation barrier. Do not reset ledgers or
reissue an unknown task with a new ID. An unknown task may reconcile to a known
final result only if the original action's real final receipt arrives.
Task aliases/plan references are in that same private ledger, not a new scheduler.
The existing ROS queue/gateway owns execution, revisions, cancellation and jobs.
Task alias history is bounded; capacity requires local archive, never remote reset.

Completed results are historical, including successful validations. They do not
promise a cache/plan remains current. A new validation check needs a NEW request_id.
New task requests always capture when the route requires it. A validation request
uses the exact frozen plan; it does not reobserve objects or prove the scene has
not changed. Refresh the preparation task when objects/scene have changed.

## Freshness policies actually enforced

- Capture: shared fresh synchronized RGB/cloud/manual FK*X; exact source headers.
  Capture waits up to its configured timeout (default 30s) and reports retention.
  Pinning is exact and idempotent; retention is not scene-currentness evidence.
- Active-object grasp route: existing manipulation-memory verification age <=60s
  and calibration association; stale inputs stop the route rather than loop VLM.
- Grasp/source and unstarted prepared plan: existing default/max 900s cache life;
  shorter explicit request TTLs still apply. Same source version/revision is required.
- Prepared validation and view calculation use existing stage-specific current
  feedback checks (typically <=1s for joint/TCP/gripper); relative view additionally
  checks camera/joint timestamp difference <=.20s and movement during capture.
- Next-best-view records retain their existing 60s lifetime; no universal one-second
  scene threshold or one-second Qwen polling is introduced.
- Backend start-pose/joint/collision/source checks remain authoritative. Existing
  operator-configured collision bypass markers are preserved, not hidden or weakened.
  An accepted plan carrying bypass markers is not fully scene-collision validated.
- Batch destination positions can use existing remembered destination geometry;
  the result labels that source. The batch API does not provide its geometry age.
  Mask-detail routes return the actual geometry source header and fresh mask handle.

## Startup and Mac handoff

Keep the current mock replay domain, false physical gate, TLS/authentication and
**same existing mock ledger**. Add only `--task-tools` to its MCP startup:

```bash
ros2 run task_pipeline pipeline_mcp \
  --preparation-only --require-mock-inputs --task-tools \
  --execution-ledger "$HOME/.local/state/agx-piper-mcp/mock-execution.sqlite" \
  --host 10.211.55.3 --port 8765 \
  --token-file "$HOME/.config/agx-piper-mcp/token" \
  --tls-cert "$HOME/.config/agx-piper-mcp/server.crt" \
  --tls-key "$HOME/.config/agx-piper-mcp/server.key"
```

Source ROS/workspace and set ROS_DOMAIN_ID=89 in that terminal. No drivers or live
stack changes are needed. The source guards still reject mixed/unverified inputs.
Results retain SIMULATED=true / input_mode=mock_replay and actual provenance.

Mac-side agent: reconnect to the existing endpoint/certificate/auth, refresh
initialize/native tools/list, task-catalog, task resources and translated/profile
snapshots, then create a separate task profile with exactly the eight names above.
Leave all-calculations/diagnostics separate. Do not change execution ledgers or task
IDs. Ubuntu changes do not edit or verify the native Mac configuration.
