## Task-profile update — MCP 3.6.0

Create a separate Nanobot task profile; leave diagnostics/all-calculations intact.
Enable exactly: inspect_robot, observe_scene, plan_camera_view,
prepare_manipulation, validate_plan, get_task_status, cancel_task,
describe_workflow. Use the native tool schemas, not manual ROS bindings.
Reconnect and refresh initialize, tools/list, translated tool/profile snapshots,
ros-pipeline://tasks and ros-pipeline://task-catalog. Preserve endpoint, TLS,
authentication, task IDs and durable request receipts. On RUNNING use status;
on UNKNOWN inspect/reconcile, never resubmit with a new ID. All preparation-only
results are non-actuating. See [TASK_INTERFACE.md](TASK_INTERFACE.md) for examples,
startup, schemas and explicit limits. Mac changes must be made by the Mac agent.

# Adapt Nanobot for the small-model MCP workflow — 3.5.0

Update the existing Nanobot integration, not the robot APIs. Qwen3.5-9B has been
wasting context on the full catalog and guessing tool bindings. Keep the existing
authenticated connection to https://10.211.55.3:8765/mcp, TLS verification,
request/receipt handling and safety gates.

1. Reconnect, refresh tools/list, and honor the server initialization instructions.
   Retain complete schemas/_meta in the adapter, but present only tool name,
   compact description and inputSchema to the model. Do not dump the entire ROS
   catalog, tool output schemas or dependency graph into every conversation.
2. For a new grasp task, make the model call get_workflow_example with
   {"workflow":"grasp","object_name":"cup","affordance_type":"AUTO"}.
   This helper only returns proposed requests. It never captures, creates a
   context, inserts a queue or moves anything. Prefer its returned exact JSON
   over free-text reconstruction. Preserve the existing connector tool namespace.
3. Follow its five returned calls in order: capture -> pin -> resolve -> named
   active calculation -> select. Programmatically replace only the later
   CONTEXT_ID_FROM_CAPTURE placeholders with the successful capture context_id.
   Keep generated step/request IDs and all binding paths unchanged. Copy template
   inputs, not prose guesses. Stop at any failure.
4. Preload the example BEFORE capture. Dispatch its pin immediately after the
   capture response; do not perform discovery/resource reading or ask the model
   to reconstruct a binding between these calls. The model chooses the task;
   deterministic JSON substitution does not add robot APIs or weaken checks.
5. For routine mode/allowlist checks call list_functions(summary_only=true).
   Preserve full-reference discovery only as an explicit on-demand operation.
   For grasp tasks expose only get_workflow_example, list_functions,
   get_world_state, get_step_context, capture_observation,
   pin_grasp_observation, resolve_target, calculate_active_grasps, select_grasp.
   Client-side filtering reduces context; it is not authorization/enforcement.
6. Use grasp_cache or pick_place examples only when requested. Expand the client
   tool set to the returned tool names that the server marks callable. Never
   automatically continue from grasp selection to placement or physical motion.
7. Preserve original structured errors and receipts. Do not turn a failed producer
   into 'never called', memory absence into visual absence, or a pin failure into
   a special-layout requirement. LAYOUT_* is a legacy pinning status name. If a
   fresh, correctly bound pin fails immediately, report source/header evidence.
8. Retrying a possibly executed/unknown operation uses its existing ID to retrieve
   its receipt. Do not create new IDs to replay it. Corrected new operations need
   fresh IDs; get_workflow_example generates a new batch of proposed IDs. It is
   not permission to bypass an unknown-state barrier. Never reset/replace ledgers.
9. Current server is preparation_only with physical gate false. No motor,
   controller, arm/gripper dispatch or safety setting changes are authorized.

Acceptance checks: the example helper is discoverable; routine list_functions
response stays small; a grasp template binds cloud.header to pin; context
substitution preserves IDs; named active grasp has no invented grasp_input
binding; failures halt progress; no physical tool is called. Test transport and
JSON sequencing with synthetic fixtures first. Do not claim end-to-end Nanobot
reliability from MCP discovery or a single model-only smoke check.

Read ros-pipeline://nanobot for the short operator workflow and
ros-pipeline://tool-dependencies for detailed field/reference examples.

Current cache/task rules: use `grasp` for preview, `grasp_cache` followed by
`prepare_grasp_lift`/`validate_prepared_grasp_lift` for lift-only, or `pick_place`
for full placement preparation. Cache the full calculated candidates pool;
`select_grasp` is provisional. Preparation may try up to eight candidates within
a budget checked at 30 seconds between attempts. Grasp/source and unstarted
prepared-plan lifetimes are now 900 seconds (shorter explicit TTLs still apply).
Preparation/validation do not consume the cache; admitted prepared execution
removes public reuse while retaining its frozen private session source. Refresh
discovery/resources after reconnect. Current launch remains non-actuating.

For the Mac `all-calculations` profile, add native `get_workflow_example`
(`ros2::get_workflow_example` if the profile uses the current connector namespace).
Discover this helper from native MCP `tools/list`, not the ROS-only catalog.
Refresh raw discovery, profile-filtered/model-facing tool snapshots, initialize
metadata, list_functions cache and ros-pipeline://nanobot after reconnect.
Keep execution ledgers and task IDs intact.

Observation-retention update: shared capture attempts exact VAMP retention before
returning. Inspect observation_retention.grasp_retained; true is confirmed, false
is failed, null is unreported. Keep the example's explicit pin as an idempotent
confirmation, bound to that capture.cloud.header; a missing binding is rejected
server-side. Placement pin does not require grasp pin. On failure report the
original request ID, context/step IDs, source header, timing and diagnostic
component counts. Do not infer eviction from a missing-bundle status, and label
dependent calls blocked/not tested. Refresh 3.4.2 discovery and resource caches.

Scene workflow: `get_workflow_example(workflow="perceive_scene",
object_name="beige cup", destination_name="tray")` returns exact two-call inputs.
Use context_id="" for initial capture; copy native result.structuredContent.context_id
(the returned top-level context_id) to subsequent calls, preserving same-context
image/cloud/transform bindings. Do not guess output.context_id. Scene parser
failures use SCENE_INFERENCE_UNKNOWN, not object absence. Fresh corrected
operations need NEW IDs; unknown actions must never be replayed. Refresh 3.5.0
tool schemas/descriptions, examples and resource snapshots.
