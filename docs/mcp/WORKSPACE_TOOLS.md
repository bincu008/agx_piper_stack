# Current-workspace tools — MCP 3.7.0

Implemented opt-in preparation extension. These tools do not actuate the arm or
gripper. Enable with --preparation-only --task-tools --workspace-tools and the
appropriate unchanged live or isolated-mock source guard. Original 54 diagnostic
tools and eight-tool profile remain available. This extension adds three tools:
65 native tools total, with nine selected for the small-model workspace profile.

## Public profile

inspect_robot, get_camera_image, check_freshness, list_objects, observe_scene,
plan_camera_view, prepare_manipulation, get_task_status, cancel_task.

- get_camera_image({}): latest eligible synchronized RGB, source pose/time and
  actual JPEG MCP ImageContent. Pixels are not placed in textual JSON. No VLM or
  motion call occurs. If stale, one bounded scan may be requested while idle,
  stationary and supplied with current feedback. The tool waits for an actual
  synchronized observation, not just a scan acknowledgement. Failed refresh
  returns REFRESH_REQUIRED or UNKNOWN without stale pixels. `check_freshness`
  remains a pure read. Sensor refresh is reflected in tool annotations.
- list_objects({}): remembered object/destination names, base_link positions in
  metres, geometry/verification ages and explicit UNKNOWN current visibility.
  Reading memory is not a fresh perception query. Use observe_scene for that.
- check_freshness({resource:"image"}) or
  check_freshness({resource:"object",object_name:"beige cup"}): Boolean fresh field
  and compact evidence. No capture/inference/reset. Image and object are the only
  implemented scopes; pool/plan scopes await single-pool migration. Object checks
  retain the shorter existing 60-second verification check as well as a 600-second
  geometry cap and calibration consistency. This does not authorize execution.
- get_task_status()/cancel_task(): omitting task_id resolves the durable current
  task. Explicit IDs remain supported for older clients. A rejected new submission
  cannot replace the admitted current task. Terminal results remain historical.
- inspect_robot additionally reports the original manual camera transform and
  existing C++ spatial-map camera axes, coverage cells and base_link direction
  labels. A vertical view has ambiguous horizontal heading. Labels do not prove
  visibility and independent source timestamps remain exposed.

Exact new schemas: WORKSPACE_TOOLS.json and native tools/list. Task-catalog includes
all eleven task/help/workspace tools; the Mac workspace profile selects nine.
Source guards, physical dispatch rejection, typed bindings, calculation jobs,
durable receipts and UNKNOWN barriers remain independent of profile filtering.

## Mac adaptation

The new opt-in profile is workspace-tasks; robot-tasks and diagnostic profiles
are retained. Local workspace_arguments.py hides request/task IDs from model
schemas/results. TaskCheckpointHook generates a new request ID once before durable
submission recording. SDK retries preserve it; original receipts remain private.
The model uses current-task defaults rather than carrying references. No unconfirmed
submission or UNKNOWN task is automatically replayed.

camera_visuals.py registers images returned by the trusted MCP camera storage path
and attaches only the latest requested successful camera image in the current user
turn as an actual image_url provider block. The cache is bounded to four images.
No image is read from an arbitrary file path, and base64 is not inserted into
textual conversation history/checkpoints. A failed new read cannot resurrect an
old picture. Legacy profiles retain their original handling. Artifact paths remain
available for user display; automatic native CLI image display is not implemented.

Portable adapter tests pass. A direct real Qwen/Qwen3.5-9B test using the nine
model-facing schemas selected get_camera_image({}) and described actual mock pixels
through the same image-request assembly function. This does not establish that the
native Mac launcher/provider/compaction chain passed; run the Mac integration test
there after refreshing dedicated workspace discovery snapshots.

## Startup and deployment boundaries

Test build is isolated under ~/agx_piper_data/builds/current_workspace/install.
Source ROS, the original workspace and then that overlay, in this order. The
new GetCurrentWorkspace interface requires rebuilt my_actions and consumers.
The existing live reasoning process has not been restarted with these changes.
A newly built MCP process alone cannot add the missing backend to an old master.

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/agx_piper_stack/install/setup.bash"
source "$HOME/agx_piper_data/builds/current_workspace/install/setup.bash"
export ROS_DOMAIN_ID=89
ros2 run task_pipeline pipeline_mcp \
  --preparation-only --require-mock-inputs --task-tools --workspace-tools \
  --execution-ledger "$HOME/.local/state/agx-piper-mcp/mock-execution.sqlite" \
  --host 10.211.55.3 --port 8766 \
  --token-file "$HOME/.config/agx-piper-mcp/token" \
  --tls-cert "$HOME/.config/agx-piper-mcp/server.crt" \
  --tls-key "$HOME/.config/agx-piper-mcp/server.key"
```

This requires the matching isolated mock reasoning stack and false master physical
gate. Do not replace a ledger or clear an unknown task to make the example succeed.
The live MCP server on 8765 was stopped at the operator's request and was not
restarted. Existing live drivers/reasoning and Qwen tunnel were left alone.

## Verification and current blocker

Authenticated mock image delivery, source/time metadata, remembered objects,
Boolean freshness, default-current status and camera direction passed. Real Qwen
selected the image tool with empty arguments and received pixels; its caption is
approximate visual evidence, not grounded object inventory or motion success.

The subsequent full pick/place recheck aborted in prepare_grasp_input: NumPy
integer seeds failed generated ROS PyLong_Check during CDR response conversion.
Fixed by converting pixel-box values/seeds to native Python ints. Regression tests
include actual serialization and an isolated domain-90 ROS request/response,
which returned GRASP_INPUT_READY and seeds 136/250 without actuators.
The interrupted calculation was subsequently marked aborted by explicit local
administrative reconciliation, with its original UNKNOWN status and audit history
preserved. No ledger replacement or automatic replay occurred. Fresh authenticated
mock pick/place tasks then completed, including the validator; see the latest
checkpoint below for scope and bypasses.

The single unconsumed pool and bounded stale-image refresh are implemented.
Identity/alias refinements are assigned to Gemini and await combined review.
Native Mac execution and the three physical facades remain outside verified scope.

## Review corrections

The current source enforces one public unconsumed pool and a maximum 600-second
source-based lifetime, including shorter caller TTLs. Source age is measured
using the ROS clock; elapsed reuse is measured monotonically. Replacing or
invalidating the pool revokes the old cache. Prepared-plan freshness requires
an exact current source version; a recent plan alone is insufficient.
Automatic invalidation on all camera/calibration changes is not yet verified.

Mac `workspace-tasks` now requires its dedicated discovery snapshot pair. Refresh
with `tests/discover_workspace_mcp.py --url https://10.211.55.3:8766/mcp
--expect-source mock_replay_publishers` on the Mac using its existing authenticated
TLS setup. This does not modify legacy discovery files or perform robot actions.
Local reconciliation remains an explicit administrative operation, rejects active
jobs and physical uncertainty, and requires operator confirmation. The historical
interrupted task has not been reconciled by this review.


## Latest verified checkpoint — 2026-11-10

Two fresh authenticated mock pick/place preparations completed with
`PASSED_WITH_BYPASSES`, including `validate_prepared_pick_place`. The final route
stores object and destination geometry via existing `store_segmented_target`
with same-capture bindings, and avoids registering the grasp contact seed as a
second object observation. Current view, pool and plan reads passed their schemas;
`execution_allowed` remains false. Full execution/physics were not tested.

A float32 ROS wire-rounding mismatch previously caused
`LOOK_PLAN_SOURCE_OR_START_CHANGED`. Owned next-view responses now use the exact
serialized ROS representation; the ownership check itself remains strict.

Configured bypasses remain visible: final-approach fingers/environment,
lift robot/environment, lift attached-object/environment, and withdrawal
robot/environment. Preparation is not proof of complete scene collision checking
or current physical readiness. No safety settings were changed.

Evidence: ~/agx_piper_data/reports/workspace_memory/preparation_result.json and
read_results.json. Mock MCP is on port8766/domain89 with its original ledger.
Live MCP remains stopped; live drivers/reasoning/tunnel were not changed.
