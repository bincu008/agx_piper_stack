# Current native-agent adaptation task: preparation only

Refresh the existing authenticated Streamable HTTP connection to ros-pipeline
3.2.0 at https://10.211.55.3:8765/mcp. Read ros-pipeline://preparation FIRST, then
catalog/workflow/spatial-map. Current mode is preparation_only, 53 callable tools,
ROS domain 0, live publisher guard enabled, master physical gate false. Keep
TLS/bearer authentication, structured outputs, durable request IDs, exact source
headers/revisions and typed same-context bindings. No motor/controller changes.

Enable only the minimal task-specific allowlists in the preparation resource.
State/perception, nearby/cube/object view calculations, grasp calculation and
complete pick/place preparation are available. get_observe_view keeps its six
scalar inputs; the other named preparations keep their existing step envelope.
refine_grasp is the existing non-actuating contact-refinement API supplying the
closing axis/pose for plan_final_approach. Never invent a closing axis or poses.
Generic queues accept calculations only and recheck function names and revisions.
No physical, gripper, navigation, reset or legacy execution API is callable.

Report actual per-call outcomes. Current tests found batch Qwen perception parse
failures (single-target resolution works), stale arm feedback blocking view/lift/
complete preparation, and some standalone IK/packing rejections. Do not retry
through a different actuator or weaken prerequisites. Getter/validator failure
means no prepared motion exists. Unknown transport outcomes remain blocked; keep
request IDs and retrieve receipts, never replay a potentially executed operation.
Do not replace the execution ledger. The driver/tunnel processes must stay intact.

The older mode-specific references below are historical context; they do not
change the current preparation_only allowlist or authorize physical execution.

# Current task for the native Mac Nanobot agent

Refresh discovery from ros-pipeline MCP 3.1.1 using the existing private TLS/bearer
connection. Read ros-pipeline://live-perception FIRST. The current server is on
ROS domain 0 with live Orbbec/actual arm-feedback sources, mode perception_only,
11 tools, physical execution disabled. Do not start hardware or call motion tools.
The ROS VLM now reaches Qwen/Qwen3.5-9B through Ubuntu's authenticated SSH forward
http://127.0.0.1:18092/v1; do not assume the Mac localhost tunnel serves Ubuntu.
This is ROS perception inference, separate from your own reasoning-model setup.

Enable only list_functions, get_world_state, get_arm_state, capture_observation,
verify_presence, ground_object, segment_object and get_step_context for this test.
Optional memory/spatial reads are allowed, but empty memory means UNKNOWN, not
visually absent. No motion/gripper/generic queue tools are available in this mode.
Use >=180s tool-call timeout, preserve structuredContent and bindings, and verify
list_functions.input_provenance reports domain 0/live_driver_publishers. Capture
one image; bind that same context/step image into presence and grounding, then
image+bbox into segmentation. Report false/empty/failed/timed-out inference as
UNKNOWN, never ABSENT. Do not rewrite an unregistered object into a visual absence.
Do not automatically replay any operation with an unknown outcome. Inspect your
real client implementation before changing configuration keys.

The rest of this file is the full-manipulation adaptation reference for the
separate full-agent mode. Its physical instructions do NOT apply to this current
non-actuating validation session.

# Prompt for the native macOS Nanobot coding agent

Adapt your existing client to Ubuntu `ros-pipeline` MCP **3.1.0**, Streamable HTTP,
`https://10.211.55.3:8765/mcp`. Inspect your actual client code/config first; do not
invent configuration keys. Keep authenticated private-network TLS. Obtain the
bearer token through the trusted operator channel; trust the public server.crt
after fingerprint verification. Never copy server.key, log credentials or disable
TLS verification. Current verification endpoint uses mock domain 89 with physical
execution disabled. Do not start hardware/controllers or execute physical actions
as an integration test.

Refresh discovery: 62 tools in default agent_pipeline mode. Preserve structuredContent,
outputSchema and tool metadata. Read resources ros-pipeline://catalog,
ros-pipeline://workflow, ros-pipeline://agent-pipeline, ros-pipeline://spatial-map
and ros-pipeline://nanobot.
Use list_functions.mode/allowlist/disabled_tools to determine actual availability.
Catalogued ROS functions are documentation unless also callable MCP tools.

The preparation gap is closed. Build your pipeline using actual named tools:

capture_observation → pin_grasp_observation/pin_placement_observation →
verify_presence/ground_object → segment_object → store_segmented_target(role=object),
then ground/segment/store destination, prepare_grasp_input → calculate_grasps
(or calculate_active_grasps) → update_object → select_grasp → cache_grasp_plan →
find_placements (or find_active_placements) → plan_next_best_view → prepare_pick_place
→ validate_prepared_pick_place → get_execution_state.

Follow resource examples rather than inventing argument names. Every preparation
call takes {context_id,step_id,request_id,arguments,bindings,expected_revision?}.
Empty context_id creates a gateway-owned context; retain it. Bind image/cloud/mask/
header/candidates through {target_field,source_step_id,source_field}. Only primitive
ROS fields belong in arguments. Never transfer cloud/image bytes or raw CDR through
the model. Copy result handles; do not confuse result_ref with execution_ref.
Same request ID returns its stored receipt without redispatch; changed arguments
are rejected. Calculate a new explicit step/request after a definite geometric
failure; never loop blindly or replay uncertain work.

For composition use run_step or submit_steps + run_steps. Generic dispatch accepts
only non-actuating preparation functions. submit_steps only queues; run_steps start
acknowledges a real asynchronous ROS action. Poll status with the same context/job
request_id until final completion. Start requires the returned exact queue revision.
Cancel requests between-step cancellation, not robot stop. Reconnect polls the
existing job; never restart it under a new ID. get_step_context returns retained
metadata/handles without binary payloads. Unknown background calculations block
new preparation/execution pending local reconciliation.

After an operator-authorized review, execute each physical stage with its own
guarded tool: look/pregrasp, gripper_open, approach, gripper_close, lift, transfer,
descent, release, commit_floor_slots, withdrawal, retreat. execute_plan supplies
look/pregrasp/approach/lift; use exact tool schemas for remaining named operations.
Use get_execution_state.execution_ref unchanged and a durable request_id. Require
fresh successful gripper_close.contact_confirmed before lift/placement; never infer
contact from a command. Use exact frozen selected_pose for commit after confirmed
release. Withdrawal/retreat use destination name with the original execution_ref.
The master enforces ordering, source/revision/phase/contact and physical gate.
No potentially executed action may be automatically retried. Only one physical
grasp attempt per prepared task is supported; no alternative after failure.

For viewing adjustments get_observe_view accepts dx/dy/dz metres and roll/pitch/yaw
radians in current optical-camera axes (+X right,+Y down,+Z forward). If success,
moved_target and endpoint_safe are true, an explicitly authorized move_to accepts
six returned target_joints plus request_id. move_to is empty-hand only and plans
its full VAMP path; do not stream microstep RPCs. No duplicated planner, transforms,
fake IK fallback or client-side safety state machine.

Test the client against synthetic responses first, then use read/preparation calls
against mock only. Show discovered tools, correct result bindings, asynchronous
completion, duplicate handling and uncertainty blocking. No physical readiness
claim follows from discovery or a mock calculation pass. Report actual client
changes, redacted startup settings and remaining limitations.


Shared spatial sensing is documented in [SPATIAL_MAP.md](SPATIAL_MAP.md), exposed
as ros-pipeline://spatial-map. Read get_spatial_context/get_spatial_map before
reasoning about cell scan age or camera coverage. The fixed 30cm summary grid has
16 cells in base_link; observed empty cells are included through bounded depth
rays. plan_next_best_view now accepts target_cell_id instead of a named/point
target; existing target modes remain unchanged. No view command moves the arm
without separately authorized guarded execution.


Make resource reading an explicit part of the client startup/task preparation.
The server's nanobot resource now includes the complete pick/place order, view
acquisition decisions and queue versus physical-execution rules. The spatial-map
resource explains the fixed cube coordinates and how to read camera_forward,
camera_up and TCP/camera quaternions. Use base_link axes for scene reasoning and
optical axes for get_observe_view deltas; never conflate them. Workspace cubes and
tray floor_slots serve different purposes. Do not embed raw clouds or reconstruct
another planner/grid in the model. Verify resources/read and structured outputs
in your client, rather than relying only on tool-name discovery.
