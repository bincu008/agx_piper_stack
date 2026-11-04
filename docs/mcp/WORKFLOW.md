# ROS MCP workflow — version 3.2.0, 2026-11-03

## Current mode: non-actuating preparation

Read `ros-pipeline://preparation` first for the exact 53-tool allowlist, startup
command, dependency rules and per-tool verification results. The current server
uses preparation_only with --require-live-inputs; master physical execution is
false. This is a separate mode from perception_only and full agent_pipeline.
No physical transport is constructed and no physical tools/queues are permitted.
Current registered arm publishers are not providing fresh joint/TCP feedback;
complete preparation remains blocked. Do not remove freshness checks or infer
readiness from successful grasp/slot searches. Drivers and Qwen tunnel are intact.

## Current live-perception configuration

Read `ros-pipeline://live-perception` for the verified domain-0 sources,
operator-authenticated Ubuntu SSH tunnel, actual Qwen endpoint/model, restart
commands and minimal Mac tool allowlist. The prior perception_only mode exposes 11
tools, with the master physical gate false and no motion tools/queues exposed.
The full agent_pipeline mode described below remains available separately.
list_functions.input_provenance reports the live DDS publisher snapshot; the
--require-live-inputs guard rejects perception for mock/duplicate/missing sources.
Memory reads are not current visual evidence: missing entries or inference errors
mean UNKNOWN, never absent. Use image-bound verify_presence for visual evidence.

## Start here: documents the agent must read

1. `ros-pipeline://nanobot`: operating instructions and the complete pick/place order.
2. `ros-pipeline://agent-pipeline`: exact predecessor bindings, preparation envelopes,
   calculation queues and guarded execution stages.
3. `ros-pipeline://spatial-map`: cube layout, scan age, camera orientation and view targeting.
4. `ros-pipeline://catalog`: live tool schemas; call `list_functions` for availability.

These resources are served by this MCP server, not instructions that require access
to the Ubuntu filesystem. Use resources/list and resources/read. Read them before
constructing tasks. Descriptions guide sequencing; the master and gateway enforce
prerequisites, revisions, contact and unknown-state blocking independently.

The complete supported route is: establish view → capture → perceive/store source
and destination → calculate grasps → search tray slots → prepare/validate every
motion → look/pregrasp → open → approach → close/verify → lift → transfer → descend
→ release → commit slots → withdraw → retreat. View acquisition may itself move
an empty hand and therefore requires separate guarded execution authorization.
Calculation queues cannot contain those physical stages.

## Ownership and discovery

The gateway now forwards nine prepared action APIs plus empty-hand move_to through the existing public
ROS catalog, with durable transport admission. It implements no motion solver,
waypoint slicing or second robot queue. The master still owns preparation,
execution phases, feedback/contact checks, attachments and completion.

Read ros-pipeline://catalog for generated input/output schemas and
_meta.rosPipeline for exact ROS endpoint/type/request/response schemas. In action
inputs, execution_ref and request_id are MCP-only correlation fields and are
removed before forwarding. Existing ROS service semantics have not changed.
list_functions returns mode, allowlist, disabled_tools, ros_catalog and mcp_tools.
The full ROS catalog is documentation, not a general-purpose dispatch surface.

Default mode is agent_pipeline, server ros-pipeline version 3.2.0. --read-only
runs ros-pipeline-read-only and leaves the ten action cards discoverable but
blocked. Inspect actual mode/availability, not annotations, for authorization.
Private-network TLS/bearer authentication and exact client allowlists remain.

## Inspection tools

| Tool | Exact input | Raw response fields |
|---|---|---|
| list_functions | {} | mode, allowlist, disabled_tools, ros_catalog, mcp_tools (local, no ROS) |
| get_arm_state | {} | success, status, joints, tcp_pose, gripper_width_m |
| get_execution_state | object_name:string | success, status, plan_revision, phase, observation_header, expires_in_seconds, completed_stages, waiting_stages, physical_execution_enabled |
| get_manipulation_memory | {} | success, status, state_json |
| get_floor_slots | {} | success, status, state_json |
| get_step_context | context_id:string, include_results:boolean | success, status, state_json, results |
| get_world_state | {} | state_json; there is no ROS success flag |

Reads return {transport_outcome,response,detail}; outcomes response, unavailable,
unknown refer to transport, not robot success. get_execution_state in enabled
mode additionally returns execution_ref, gateway_status and selected_pose.
The last is the frozen commit pose once the gateway has admitted the task.
A success=true master response can still report STATE_UNKNOWN; inspect status,
phase, gateway_status, expiry and physical gate. Reads do not authorize actuation.

Exact nested schemas are generated from installed ROS messages. Joint positions
are radians, velocity radians/s; effort depends on the driver. Pose XYZ and jaw
width are metres, force newtons; pose orientation is quaternion XYZW. Headers
carry ROS sec/nanosec and explicit frame_id. Manipulation geometry uses base_link;
never substitute the optical frame or transform through a new TF fallback.
State_json is opaque version-dependent JSON. PipelineStepResult includes step_id,
function_name, success, status, wall_seconds, response_type and response_cdr bytes;
only decode using the matching ROS type. Keep uint64 IDs exact.

Reads have five-second bounded waits; >1-MiB/nonfinite responses are refused
rather than truncated. Use include_results=false for task summaries; image/cloud
CDRs may exceed this limit. Read retries give a new snapshot and move nothing.

## Agent-composed preparation

Default startup now exposes 62 callable tools in agent_pipeline mode. Read
[AGENT_PIPELINE.md](AGENT_PIPELINE.md), available as ros-pipeline://agent-pipeline,
for exact step bindings, full perception/memory/planning dependency tables,
calculation queues and asynchronous completion. Individual preparation tools and
run_step/submit_steps/run_steps reuse the master's StepStore. Images/clouds stay
in ROS-side results; clients supply references, never raw CDR. The new
store_segmented_target extracts and stores an existing segmentation without a
second inference. Preparation checks current idle known-empty-hand state.

Physical functions are rejected from generic step/queue dispatch. The remote
agent invokes the guarded execution tools individually, preserving the existing
durable execution/contact/source/phase checks. See the prepared stages below.

Review the complete frozen grasp/lift/place/withdrawal/retreat before actuation.
The gateway's first action independently calls validate_prepared_pick_place and
captures its selected_pose. It only supports complete prepared pick/place plans,
not legacy staged grasp/held records, lift-only plans or adoption halfway through
someone else's execution. Planning candidates tried are not physical attempts.

## Action inputs and dependencies

Prepared pick/place actions require the following fields. move_to instead takes
only target_joints plus request_id; it does not require an object or execution_ref:


- execution_ref: copy get_execution_state.execution_ref exactly. It contains
  task_id (SHA-256 of object/revision/source-header identity), object_name,
  plan_revision:uint64, attempt=1. It is a correlation handle, not a safety token.
- request_id: a new durable string of 1–128 characters for each operation. Never
  reuse it for another request. Keep it across reconnect and lost responses.

| API | ROS arguments (in addition to MCP fields) | Required predecessor / next |
|---|---|---|
| execute_plan | object_name, plan_revision, segment | segment look/pregrasp/approach/lift/transfer/descent; follow first waiting_stages entry |
| gripper_open | object_name, plan_revision, width_m, force_n | AT_PREGRASP -> OPENED; next approach |
| gripper_close | object_name, plan_revision, width_m, force_n | AT_GRASP -> CONTACT_CONFIRMED; require success AND contact_confirmed |
| execute_placement_transfer | object_name, plan_revision | verified contact and LIFTED -> AT_PREPLACE |
| execute_placement_descent | object_name, plan_revision | verified contact and AT_PREPLACE -> PLACED |
| release_placed_object | object_name, plan_revision | verified contact and PLACED -> RELEASED; require gripper_released |
| commit_floor_slots | selected_pose, reset, validate_only | confirmed release; exact get_execution_state.selected_pose, reset=false, validate_only=false |
| execute_withdrawal | object_name, plan_revision | RELEASED, committed slots and empty hand; use destination name |
| execute_placement_retreat | object_name, plan_revision | completed withdrawal; use destination name; final COMPLETE |

Normal order: look (when in plan), pregrasp, gripper_open, approach, gripper_close,
lift, transfer, descent, release, slot commit, withdrawal, retreat. After complete
empty-hand retreat, master retires transient execution caches while retaining
object/destination memory, placed geometry, reservations and result history. The
next object requires fresh preparation. No API is automatically invoked by connect.

object_name is the exact prepared source name, except withdrawal/retreat use the
prepared destination name. Keep the original execution_ref for those operations.
Width/force zero selects frozen defaults; nonzero must match. selected_pose is a
source-associated base_link PoseStamped; do not change its header, frame or pose.
Only standard commit is enabled; resetting or previewing reservations is rejected.

## Outcomes, idempotency and recovery

Action envelope: {transport_outcome, dispatched:boolean, response, detail}.
- response: raw ROS response preserved; completion and post-state confirmed.
- rejected: pre-dispatch failure; response=null, dispatched=false. Fix the exact
  missing prerequisite, not random retries. No alternative candidate is executed.
- unknown: command may have executed, or completion/contact/post-state is not
  established. Stop all further actions; never automatically replay it.

All failures set MCP isError. Exact ROS response schemas remain in discovery.
ExecutePlan returns success/motion_completed/phase/status; prepared gripper calls
return success/contact_confirmed/phase/status. HeldMotionStage returns success,
motion_completed/gripper_released/status. Commit returns success/status/
occupied_slots. A command ACK is never treated as final contact/completion.

Admission and final receipts are committed to SQLite before/after dispatch. Same
request_id+arguments returns the stored historical receipt with dispatched=false,
never another command; this is not current readiness. Changed arguments with the
same ID are rejected. One gateway process owns the ledger via an OS file lock.
One active task owns admission until confirmed COMPLETE. A 10,000-record capacity
limit fails closed instead of dropping safety/idempotency evidence.

Current policy is one physical grasp attempt per task, stricter than the maximum
three-attempt ceiling. A persisted close count of three is rejected. The master
interfaces still lack explicit safe physical-failure/alternative handoff, so no
alternative/retry after a dispatched failure is supported. A new task is admitted
only after the previous task completes. Reconnection never resets this ownership.

Any dispatched failure, timeout, cancelled wait, crash while running, unconfirmed
contact or mismatched post-state marks the gateway unknown and blocks all new
actions. Backend waits are up to 300 seconds; master active-session timeout is
independent (default 300 s); initial preparation lifetime is 60 s. First handoff
validation has a 15 s wait. Prepared execution does not perform IK, refine or replan. Empty-hand move_to
retains its existing VAMP planning/arrival cycles inside the master.

There is no automatic or remote unknown-state reset. Local administrative review
must reconcile actual arm/hand state through the existing operator-confirmed
master recovery before changing gateway records. Do not delete/replace the ledger
or restart to manufacture permission. This delivery has no administrative ledger
reconciliation tool: uncertainty deliberately fails closed. Request IDs retrieve
known final receipts but never clear uncertainty.

MCP cancellation/disconnection cannot stop an in-flight ROS service. If a final
response is obtained it is retained; if the gateway loses its wait, the outcome
is unknown. Local ROS RunPipelineSteps cancellation stays between steps, not
inside an actuator service; existing emergency/driver/watchdog paths remain
independent. Stop MCP is not emergency stop. Do not bypass configured collision
checks, speed restoration, joint bounds, held-object checks or the physical gate.
Existing reported lift/withdrawal/approach bypasses remain explicit and unchanged.

## Relative-camera calculation and empty-hand move

get_observe_view is non-actuating but captures/caches a fresh observation. It is
available in both modes; --read-only blocks motion while retaining this calculation.
These previous 20 tools remain available alongside 42 new preparation/queue tools.

Exactly six inputs: dx,dy,dz metres; roll,pitch,yaw radians, in the CURRENT OPTICAL
CAMERA frame (+X right, +Y down, +Z forward). Translation norm <=0.20 m and each
rotation magnitude <=pi. Rnew=Rcamera*Rz(yaw)*Ry(pitch)*Rx(roll). Inputs remain
camera-relative even when the wrist rotates. Hand-eye calibration converts the
desired camera target to TCP; only synchronized manual FK*X is consumed, no TF.

Master requires fresh, stationary, idle empty-hand feedback and calibration. A
frozen registered cloud, measured joints and manual camera transform feed the C++
solver. It prefers 2 mm TCP/0.025 rad orientation tolerance, then permits 0.35 rad
orientation tolerance; achieved camera position remains within 5 mm of the attempted
target. Bounded half/quarter requests are tried if IK fails. Search budget is 3 s
between solves; the inclusive API wait is 30 s. Full endpoint self/arm/wrist/palm/
finger-environment checks are retained, with no grasp/approach collision bypass.
Current-robot self returns inside sphere geometry plus 2.5 mm shell are excluded
from the immutable cloud; this is self filtering, not arbitrary obstacle deletion.
Raw gripper feedback remains unchanged; closed-jaw roundoff from -1 mm is treated
as zero geometric opening, up to 101 mm as at most 100 mm geometric opening.

Response: success, status, moved_target, endpoint_safe, source_joints[6],
target_joints[6], observation_header, requested_camera_pose, achieved_camera_pose,
translation_error_m, orientation_error_rad, achieved_fraction, candidates_tested.
Poses are base_link PoseStamped, joints radians, errors metres/radians. EXACT,
RELAXED or PARTIAL reports a new safe endpoint; achieved_fraction is 1, .5 or .25.
Inspect achieved errors, particularly when relaxing orientation. These are predicted
FK poses, not measured physical motion. A collision immediately returns source
joints with TARGET_COLLISION. Exhausted IK returns NO_REACHABLE_CHANGE. Zero input
returns NO_CHANGE. All unchanged decisions have moved_target=false and
endpoint_safe=false; they do not certify current-pose safety. Source/transform,
calibration or feedback errors return success=false instead of fabricated joints.

move_to accepts the existing ROS target_joints[6] in radians plus the MCP-only
request_id. No execution_ref. Call it once, only after a useful checked view result.
Gateway checks world/feedback readiness, false physical gate before dispatch,
active/unknown ownership, and persists request/admission. Master then performs its
existing empty-hand VAMP path planning, trajectory execution and endpoint check.
It does not receive raw microstep RPCs. A collision-free get_observe_view endpoint
is not permission to bypass path checking. Master success plus fresh final joint
feedback confirms completion. Timeout/failure/cancellation marks unknown, with
no automatic retry and the same durable receipt semantics as prepared operations.

No legacy actuator shortcut, reset/reconcile or arbitrary execute_trajectory is exposed.
Generic run_step/submit_steps/run_steps are calculation-only; see the agent pipeline resource. The gateway is not a global cap on
unrelated ROS clients. Unit fixtures cannot move hardware. Mock tests calculate
views and verify false-gate rejection only. Earlier calculation/placement results
are in MOCK_VALIDATION.md; reports for view calculation are under
~/agx_piper_data/reports/observe_view_validation.json, observe_view_rotations.json,
and observe_mcp_validation.json. Physical tracking and native macOS execution
remain untested. Reconnect Nanobot to refresh 3.1.0 schemas.


Shared spatial sensing is documented in [SPATIAL_MAP.md](SPATIAL_MAP.md), exposed
as ros-pipeline://spatial-map. Read get_spatial_context/get_spatial_map before
reasoning about cell scan age or camera coverage. The fixed 30cm summary grid has
16 cells in base_link; observed empty cells are included through bounded depth
rays. plan_next_best_view now accepts target_cell_id instead of a named/point
target; existing target modes remain unchanged. No view command moves the arm
without separately authorized guarded execution.
