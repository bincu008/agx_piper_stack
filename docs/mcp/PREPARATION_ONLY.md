## Explicit mock source mode — MCP 3.5.1

For recorded-scene tests, stop real camera/arm drivers and launch mock hardware
and reasoning with ROS_DOMAIN_ID=89 and physical execution false. MCP has a
separate explicit replay guard; it does not disable the production live guard.

```bash
export ROS_DOMAIN_ID=89
ros2 run task_pipeline pipeline_mcp \
  --preparation-only --require-mock-inputs \
  --execution-ledger "$HOME/.local/state/agx-piper-mcp/mock-execution.sqlite" \
  --host 10.211.55.3 --port 8765 \
  --token-file "$HOME/.config/agx-piper-mcp/token" \
  --tls-cert "$HOME/.config/agx-piper-mcp/server.crt" \
  --tls-key "$HOME/.config/agx-piper-mcp/server.key"
```

Source guard requires exactly one /mock_hardware publisher on every raw camera/
arm-feedback source and a nonzero domain. The CLI requires a dedicated mock
ledger, retaining live receipts unchanged. Every result is marked SIMULATED=true
and input_mode=mock_replay with DDS provenance. The 54 preparation/inspection
tools remain; physical APIs/queues are still rejected independently.
Replay does not emulate motion or prove physical manipulation.

# Current small-model entry point — MCP 3.4.0

The callable preparation-mode count is now 54: the 53 tools below plus the local
read-only `get_workflow_example`. No physical API was enabled. Prefer its exact
grasp example and `list_functions(summary_only=true)` for normal agent use.
Read [Nanobot quickstart](NANOBOT.md). Earlier verification facts below apply to
their recorded runs, not current sensor freshness or tool readiness.

## Dependency discovery update — MCP 3.3.0, 2026-11-09

Read `ros-pipeline://tool-dependencies` and the structured
`ros-pipeline://tool-dependency-graph`. Every tool's `_meta.rosPipeline` and
model-visible description now include prerequisites, execution sequence and
binding examples. Required, recommended, internal and alternative routes differ.
Failures retain their original status and gain recovery guidance; unknown-state
blocking and the actual mode allowlist remain unchanged. Refresh tool discovery
on reconnect. See [dependency guide](TOOL_DEPENDENCIES.md).

# Non-actuating preparation — MCP 3.2.0

Current mode is preparation_only, ROS domain 0, with live publisher provenance
checks and the master physical gate false. This mode allows state changes to
perception memory, grid/search caches and prepared records, but NO actuation.
Do not confuse non-actuating with read-only or successful planning with permission
to execute. Current arm/TCP feedback is stale; complete preparation is blocked.

## Startup (restart MCP only)

Leave the existing reasoning stack, hardware drivers and operator-authenticated
Qwen SSH tunnel intact. Stop only the previous MCP process you own, then:

```bash
cd ~/agx_piper_stack
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
export PYTHONPATH="$HOME/agx_piper_data/deps/ros_mcp:$PYTHONPATH"
ros2 run task_pipeline pipeline_mcp \
  --preparation-only --require-live-inputs \
  --host 10.211.55.3 --port 8765 \
  --token-file "$HOME/.config/agx-piper-mcp/token" \
  --tls-cert "$HOME/.config/agx-piper-mcp/server.crt" \
  --tls-key "$HOME/.config/agx-piper-mcp/server.key"
```

The live-input flag is mandatory for this mode. It does not turn off
--perception-only: the modes are separate and mutually exclusive. Keep the same
execution ledger. Never replace it, reset unknown state, enable motors or change
the master's safety settings to make a test pass. The SSH endpoint/model remain
http://127.0.0.1:18092/v1 and Qwen/Qwen3.5-9B as described by live-perception.

## Exact callable tools (53)

**State / discovery (9)**

```text
list_functions
get_world_state
get_arm_state
get_execution_state
get_manipulation_memory
get_floor_slots
get_step_context
get_spatial_map
get_spatial_context
```

**View calculations (2)**

```text
get_observe_view
plan_next_best_view
```

**Perception / memory / source association (14)**

```text
capture_observation
verify_presence
ground_object
segment_object
store_segmented_target
resolve_target
refresh_active_target
get_active_target
perceive_scene
ensure_scene_memory
check_target_view
require_target_view
pin_grasp_observation
pin_placement_observation
```

**Grasp calculation / contact refinement (9)**

```text
prepare_grasp_input
calculate_grasps
calculate_active_grasps
update_object
select_grasp
cache_grasp_plan
get_cached_grasp
refine_grasp
plan_final_approach
```

refine_grasp is an existing ROS catalog operation exposed because its actual
closing_axis_base, corrected_seed_base, grasp_pose/joints and grasp_width_m supply
plan_final_approach's contact inputs. Its raw response uses valid/failure_reason;
the master StepStore projects its existing success interpretation. No new robot
solver or message schema was introduced. See catalog metadata for exact fields.

**Geometry / standalone planning (10)**

```text
prepare_held_scene
search_placements
find_placements
find_active_placements
solve_pose
plan_motion
plan_lift
plan_placement
validate_placements
plan_camera_retreat
```

**Complete prepared calculations (6)**

```text
prepare_grasp_lift
get_prepared_grasp_lift
validate_prepared_grasp_lift
prepare_pick_place
get_prepared_pick_place
validate_prepared_pick_place
```

**Calculation-only composition (3)**

```text
run_step
submit_steps
run_steps
```

list_functions and ros-pipeline://catalog are authoritative for the running
allowlist, exact schemas and disabled tools. ROS catalog documentation is not
a general-purpose dispatch grant. No motion, gripper, slot commit, execution,
reset, navigation toggle or legacy composite dispatcher is callable in this mode.

## Enforcement and unchanged contracts

- Each named preparation tool keeps its existing context_id/step_id/request_id/
  arguments/bindings envelope; get_observe_view retains its exact six scalar inputs.
  ROS names and message/service schemas are unchanged.
- Preparation mode never constructs the physical ROS execution transport.
  Physical catalog operations are independently rejected before routing even if
  called by name. Generic run_step/submit_steps use a preparation-only enum and
  adapter allowlist; run_steps rechecks every pending function and exact revision
  immediately before starting the master action. No arbitrary endpoint/CDR import.
- Preparation and view calculations share the existing durable admission lock and
  ledger/unknown-state barriers. Preparation requires a known idle empty hand and
  an explicitly false master physical gate. It never changes that gate.
- Before calculation admission (including generic queue submission/start), verify
  the same raw live publisher identities used by perception-only mode. Publication
  identity is NOT message freshness. Frozen headers/source association and the
  master feedback-age checks remain independent. State inspection and action
  status/cancellation remain available when sources are missing.
- Queued work still executes in the master StepStore, not another robot queue.
  Live graph admission is not a lock on external publishers for the duration of a
  running queue. Inspect source headers/provenance again after source changes;
  failed/unknown source state is never evidence of live, fresh robot feedback.
- Malformed binding syntax, unknown/failed/out-of-order predecessors and incompatible
  ROS field types are refused before forwarding the backend call. Numeric field
  widths retain existing compatible integer/float binding conventions; message
  identities remain exact. This additional check is in MCP because ROS generated
  Python setters may postpone type validation until serialization.
- Same request ID retrieves its durable receipt; changed arguments are rejected.
  A duplicate receipt is historical, not current readiness. Async run_steps start
  is ACK_ONLY; poll the same job to final completion. Cancellation is between
  steps, not emergency stop. Timeout/restart ambiguity preserves unknown barriers;
  no remote reconciliation or physical retry was added.

## Minimal Nanobot allowlists grouped by task

Start every task with list_functions and read ros-pipeline://preparation. Confirm
mode=preparation_only, domain=0 and the false master gate; preserve TLS/bearer auth.
Do not enable all documented ROS functions indiscriminately.

| Task | Small useful allowlist (in addition to list_functions/get_world_state/get_arm_state) |
|---|---|
| Visual presence/segmentation | capture_observation, verify_presence, ground_object, segment_object, get_step_context |
| Memory and geometric visibility | capture_observation, resolve_target, refresh_active_target, get_active_target, get_manipulation_memory, check_target_view, require_target_view |
| Nearby/cube/object view calculation | get_observe_view, plan_next_best_view, get_spatial_map, get_spatial_context; add resolve_target/check_target_view for named objects |
| Grasp and lift preparation | capture_observation, pin_grasp_observation, resolve_target, prepare_grasp_input, calculate_active_grasps, update_object, select_grasp, cache_grasp_plan, get_cached_grasp, prepare_grasp_lift, get_prepared_grasp_lift, validate_prepared_grasp_lift, get_execution_state, get_step_context |
| Complete pick/place preparation | Grasp/source tools above plus pin_placement_observation, find_active_placements, plan_next_best_view, prepare_pick_place, get_prepared_pick_place, validate_prepared_pick_place, get_floor_slots; resolve_target(role=destination) supplies tray geometry |
| Low-level planning diagnostics | prepare_held_scene, search_placements, solve_pose, plan_motion, refine_grasp, plan_final_approach, plan_lift, plan_placement, validate_placements, plan_camera_retreat; require real successful grasp/scene predecessors |
| Ordered calculation composition | run_step, submit_steps, run_steps, get_step_context plus the desired named calculation tools |

Use either explicit ground/segment/store or combined resolve_target, not both
redundantly. Missing memory is UNKNOWN, never absence. LOOK_REQUIRED may block a
workflow: this mode cannot physically acquire a different view. get_observe_view
is a standalone calculation, not a generic queued operation. State reads are
called directly; run_step's function enum contains only the named preparation
operations. Keep cloud/image/CDR data in Ubuntu and bind same-context results.

## Actual functional results and limitations

The live tests used existing camera images and stored, source-associated geometry;
no hardware was moved and no feedback was fabricated. Successes included source
capture/pinning, single-target perception/memory, visibility, pose/path/view-plan
calculation, both grasp generators, ranking/caching, immutable held geometry,
placement candidates, real grasp refinement/final approach, empty-hand clearance
retreat and the real master calculation queue. Rejection tests covered direct
physical API names, legacy dispatchers, physical queue insertion, wrong binding
syntax/types and missing predecessors. See the per-tool verification report for
all results rather than interpreting this paragraph as every tool succeeding.

Observed blockers retained unchanged:

- perceive_scene: Qwen echoed a target-description array instead of the expected
  keyed presence mapping. The existing ROS parser returned Extra data. After
  resolving targets individually, ensure_scene_memory reused memory successfully.
- get_observe_view: STALE_VIEW_FEEDBACK. prepare_grasp_lift/prepare_pick_place:
  STALE_HARDWARE_FEEDBACK: joint_feedback_time. A read-only subscriber received
  zero joint/TCP samples in six seconds despite registered publishers. Their
  prepared getters/validators therefore correctly report NO_PREPARED_GRASP_LIFT;
  successful end-to-end prepared-motion handoff remains BLOCKED/UNVERIFIED.
- One standalone lift failed IK at waypoint 10; standalone placement validation
  found no reachable candidate. Some low-level support-cloud searches found no
  stable packing. Higher-level placement search returned valid candidates.

Restore fresh feedback through the operator's normal process before retrying
complete preparation. This change does not restart drivers, change safety limits,
weaken collision checks or repair/tune existing ROS planning/model behavior.
The working single-target route is available while batch inference is blocked.

Detailed evidence is under ~/agx_piper_data/reports/preparation_mode_*.json and
logs/preparation_mode_*.log. This is functional validation, not a performance
benchmark. Native Nanobot adoption and physical execution remain untested.


Camera image/cloud headers can be consistent while the arm feedback is stale.
A captured or retained observation is not a current robot-readiness certificate.
The independent fresh-feedback requirement still blocks complete preparation.
No poses, feedback, contact flags or successful prepared records were fabricated
for the missing success-path tests.

## Per-tool verification results

The 53 tools all have actual call evidence. Forty had successful calls; thirteen
returned calculation failures or were blocked by prerequisites. The latter are
NOT claimed as successful positive-path verification. The original negative-only
binding tests are retained alongside later valid calls. State snapshot success
is not a motion-safety or freshness claim.

# Per-tool functional verification — preparation-only mode

Live-origin immutable observations; no actuation. Tests were not a performance benchmark.
A failed or blocked call is not positive functional verification of the success path.

| Tool | Successful call observed | Actual statuses |
|---|---|---|
| cache_grasp_plan | Yes | GRASP_PLAN_CACHED_IN_MASTER |
| calculate_active_grasps | Yes | FEWER_THAN_MAXIMUM_VALID |
| calculate_grasps | Yes | FEWER_THAN_MAXIMUM_VALID |
| capture_observation | Yes | SYNCHRONIZED_OBSERVATION_READY |
| check_target_view | Yes | READY |
| ensure_scene_memory | Yes | Extra data: line 1 column 82 (char 81); SCENE_MEMORY_READY_NO_INFERENCE |
| find_active_placements | Yes | CANDIDATES_READY |
| find_placements | Yes | OK backend=heightmap policy=upright object=light-colored mug with a handle destination=dark rectangular tray support_mode=floor_slots candidates=16 object_points=9144 additional_obstacles=0 bottom_completion=no support_aligned=16/16 support_shift_mm=integrated_heightmap destination_points=48276 planning=0.59s best_score=-0.0010 packing={orientations:72,tested:432,contained:432,stable:313,support_tilt_deg:1.0,support_contacts_rejected:2,support_area_rejected:6,support_com_rejected:111} |
| get_active_target | Yes | ACTIVE_TARGET_REQUIRED; ACTIVE_TARGET_SNAPSHOT_READY |
| get_arm_state | Yes | ARM_STATE_READY |
| get_cached_grasp | Yes | NO_FRESH_CACHED_GRASP; CACHED_GRASP_READY |
| get_execution_state | No — failed/blocked | NO_PREPARED_GRASP_LIFT |
| get_floor_slots | Yes | FLOOR_SLOT_MEMORY_READY |
| get_manipulation_memory | Yes | MANIPULATION_MEMORY_READY |
| get_observe_view | No — failed/blocked | STALE_VIEW_FEEDBACK |
| get_prepared_grasp_lift | No — failed/blocked | NO_PREPARED_GRASP_LIFT |
| get_prepared_pick_place | No — failed/blocked | NO_PREPARED_GRASP_LIFT |
| get_spatial_context | Yes | SPATIAL_CONTEXT_READY |
| get_spatial_map | Yes | SPATIAL_MAP_SNAPSHOT |
| get_step_context | Yes | CONTEXT_READY |
| get_world_state | Yes | response |
| ground_object | Yes | INVALID_BINDING_FIELD: __class__; BINDING_TYPE_MISMATCH: image; GROUNDING_READY |
| list_functions | Yes | preparation_only |
| perceive_scene | No — failed/blocked | Extra data: line 1 column 82 (char 81) |
| pin_grasp_observation | Yes | LAYOUT_OBSERVATION_PINNED |
| pin_placement_observation | Yes | LAYOUT_OBSERVATION_PINNED |
| plan_camera_retreat | Yes | CLEARANCE_RETREAT_VALIDATED |
| plan_final_approach | Yes | OK;NEAR_GRASP_USED;FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED |
| plan_lift | No — failed/blocked | LIFT_IK_FAILED_AT_WAYPOINT_10;LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED;LIFT_ROBOT_ENVIRONMENT_CHECK_BYPASSED |
| plan_motion | Yes | OK |
| plan_next_best_view | Yes | CURRENT_VIEW_ALREADY_SUITABLE |
| plan_placement | No — failed/blocked | PLACEMENT_NO_REACHABLE_CANDIDATE tested=10 rejected={pose:0,ik:4,upright_tilt:20,collision:1,preplace:5,static_collision:0,finger_collision:0,attached_collision:1,object_arm_collision:0,descent_arm_collision:0,final_pose_collision:1,descent_ik:0,descent_continuity:0} |
| prepare_grasp_input | Yes | GRASP_INPUT_READY |
| prepare_grasp_lift | No — failed/blocked | NO_FRESH_CACHED_GRASP; STALE_HARDWARE_FEEDBACK: joint_feedback_time |
| prepare_held_scene | Yes | IMMUTABLE_HELD_SCENE_READY |
| prepare_pick_place | No — failed/blocked | STALE_HARDWARE_FEEDBACK: joint_feedback_time |
| refine_grasp | Yes | OK |
| refresh_active_target | Yes | ACTIVE_TARGET_GROUNDED |
| require_target_view | Yes | READY |
| resolve_target | Yes | ACTIVE_TARGET_GROUNDED |
| run_step | Yes | ACTIVE_TARGET_SNAPSHOT_READY; DEPENDENCY_FAILED: not_yet_calculated; MCP_ERROR |
| run_steps | Yes | RUNNING; COMPLETE |
| search_placements | No — failed/blocked | NO_STABLE_PACKING_CANDIDATE |
| segment_object | Yes | OK |
| select_grasp | Yes | RANKED_GRASP_SELECTED |
| solve_pose | Yes | OK |
| store_segmented_target | Yes | ACTIVE_TARGET_GROUNDED; ACTIVE_TARGET_REUSED |
| submit_steps | Yes | MICROSTEPS_QUEUED; MCP_ERROR |
| update_object | Yes | UPDATED obj_001 |
| validate_placements | No — failed/blocked | PLACEMENT_NO_REACHABLE_CANDIDATE tested=10 rejected={pose:0,ik:4,upright_tilt:20,collision:1,preplace:5,static_collision:0,finger_collision:0,attached_collision:1,object_arm_collision:0,descent_arm_collision:0,final_pose_collision:1,descent_ik:0,descent_continuity:0} |
| validate_prepared_grasp_lift | No — failed/blocked | NO_PREPARED_GRASP_LIFT |
| validate_prepared_pick_place | No — failed/blocked | NO_PREPARED_GRASP_LIFT |
| verify_presence | Yes | PRESENT |

Failures of complete preparation/getters/validators require fresh arm feedback and a
successful full preparation before their success paths can be verified.
Details: ~/agx_piper_data/reports/preparation_mode_summary.json and preparation_mode_*.json.
