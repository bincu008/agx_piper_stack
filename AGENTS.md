# Repository Guidelines

## Project Structure & Module Organization

This is a ROS 2/colcon workspace. Production packages live under `src/`:

- `master_node`: Python orchestration and hardware execution.
- `task_pipeline`: shared public API catalog, dispatcher, typed step/result storage and recipes. `MasterStepRuntime` hosts the queue/result store inside `master_node`.
- `vision_node`, `vamp_node`, and `vlm_node`: C++ active perception, VAMP motion planning, and VLM integration.
- `vamp_mvt`: separate VAMP C++ planning/collision library with the generated Piper model.
- `placement_node`: deterministic height-map placement and the optional AnyPlace backend; it is launched by the reasoning stack.
- `my_actions`: shared ROS services/actions; update these before changing cross-node APIs.
- `mock_hardware_node`: recorded RGB-D and arm-feedback replay for calculation-only development. It does not emulate physics or execute arm commands.
- `agx_arm_ros/src/agx_arm_ctrl` and `agx_arm_msgs`: Piper CAN control and arm messages.
- `OrbbecSDK_ROS2/orbbec_camera` and `orbbec_camera_msgs`: RGB-D driver and interfaces.

Keep runtime assets beside their owning package: `vision_node/models`, `vlm_node/models`, and `vamp_node/models`. ONNX Runtime is an independently installed C/C++ dependency selected through `ONNXRUNTIME_ROOT`. Generated `build/`, `install/`, and `log/` directories are not source.

## Build, Test, and Development Commands

From the workspace root:

```bash
export ONNXRUNTIME_ROOT="$HOME/.local/onnxruntime-1.15.1"
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select task_pipeline placement_node master_node vamp_node vlm_node mock_hardware_node agx_arm_ctrl
colcon test-result --verbose
```

The official application launch is `master_node/reasoning_stack.launch.py`. It starts composed ActiveVision/VAMP, the standalone multithreaded VLM node, `PiperMasterControl`, and `placement_node`. `pipeline_node` and the microstep runtime run inside the master process by default (`api_dispatch_in_master:=false` restores the separate API proxy). Camera and AGX hardware drivers are started separately. The placement package and Python module are named `placement_node`, with `/placement/*` backend endpoints and deterministic height-map packing by default.

Safe planning launch, with physical grasp execution disabled:

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio grasp_candidate_backend:=hggd \
  placement_backend:=heightmap allow_physical_grasp_execution:=false
```

`inference_profile:=server` selects the optimized remote server. Profile defaults live in `master_node/config/inference/profiles.yaml`; explicit endpoint/model/timeout launch arguments override them.

Multi-object templates use `[PLAN_UPRIGHT]` or `[PLAN]` with two or more ordered object descriptions (commas, `then` or `and`); the parser routes to layout calculation. Explicit `[PLAN_LAYOUT_UPRIGHT]` / `[PLAN_LAYOUT]` aliases remain. Document only tags implemented by the VLM parser.

## Policy reference

See [README: Available policies and backend choices](README.md#available-policies-and-backend-choices)
for supported values and their public API/launch controls. Preserve the distinction
between support modes (`auto`, `edge`, `stack`, `floor_slots`), benchmark sequencing
(`edge_then_stack`), orientation (`upright`, `free`), search effort and engine choices.
Deployment grasp proposals use HGGD only and target masks use EdgeSAM only.
Do not restore analytical/GPD/hybrid selection or box-only/YOLO-intersection
target masks. Known-cloud benchmark contact generation remains internal, and
YOLO world inventory remains a separate supported feature.
Universal stacking uses exposed scene heightmaps and defaults to 60% projected
footprint coverage across one or multiple surfaces, with existing stability and
collision checks. Do not reintroduce named-support restrictions. IK seed modes
use the same solver; the application motion planner remains RRT-Connect.
Attached-object lift/descent permit a bounded sparse-contact budget (32 unique
obstacle points and 10 occupied 5 mm cells by default), accumulated
across a complete segment. Never reset it per waypoint or use it for arm/gripper
or transfer collision checks. `placement_noise_max_points:=0` disables this
sparse-contact allowance. Bounded separating departure is enabled by default. Small real contacts can also meet these limits; do not
describe the budget as proof that samples are noise.

## Shared API and Microstep Rules

- Accepted next architecture: read [Agent-controlled arm API design](docs/AGENT_CONTROL_ARCHITECTURE.md)
  before MCP or execution-API work. It is a documented plan, not implemented functionality.
  Expose prepared motion segments and gripper operations through the shared catalog;
  unify the prepared/staged/held paths on one master-owned execution session.
  Keep composite recipes and benchmarks on those same operations. The agent chooses
  tasks and adjustments; feedback/contact checks, speed restoration and attachment
  updates remain automatic. Do not create one RPC per waypoint/IK iteration or
  a separate MCP state/planner implementation. The 2026-10-13 update authorizes
  documentation only; follow later user instructions for implementation scope.

- `task_pipeline/task_pipeline/registry.py` is the authoritative function-name/type/backend catalog. Production controllers and benchmarks must use it rather than introduce private engine-call sequences or duplicate endpoints.
- Public services use `/pipeline/<function>`; list them with `/pipeline/list_functions` (`my_actions/srv/GetPipelineFunctions`). Declare message/service/action schemas in `my_actions` and rebuild consumers after changing them.
- The master-owned `StepStore` executes typed `PipelineStep` requests, resolves predecessor result bindings, records per-step status/timing and depletes attempted tasks. `@response_cdr` binds a complete serialized response. Never use eval or an arbitrary endpoint supplied by a caller.
- Use `submit_steps` plus `/pipeline/run_steps` (`my_actions/action/RunPipelineSteps`) when callers need final completion and feedback. `run_step` executes one operation. The legacy `execute_pipeline` service only acknowledges queue start.
- Retain stop/continue policies, source observation association, revision checks, bounded storage, and idempotent context/step IDs. Explicit contexts must be cleared by their owner; the box benchmark clears its previous run's contexts before rerunning.
- Cancellation occurs between steps; it cannot stop an in-flight ROS service. A timeout with `STATE_UNKNOWN` must never trigger automatic actuator retries. Emergency stop and drivers retain their direct paths.
- `prepare_cloud_scene` registers immutable known-cloud inputs in the bounded master result store. `RunPipelineStep.bindings` and queued step bindings refer to results in the same context; preserve exact source timestamps/frame IDs and revision/idempotence checks.
- Publish shared function timing on `/pipeline/timings`. Correlate per-call request/backend/response timestamps through `/pipeline/api_traces`; monotonic timestamp differences are valid only on the same host. API, backend and engine scopes overlap: do not add them as disjoint stages. Compare deployed RGB-D/HGGD and synthetic known-cloud modes separately.
- General `move_to`/`execute_trajectory` remain empty-hand only. Payload execution uses `prepare_held_motion` followed by revision-bound `execute_lift`, `execute_placement_transfer`, `execute_placement_descent`, and `release_placed_object`. Gripper closing requires the ordered refined grasp stages and fresh post-command contact feedback. `pipeline_task --prepare-grasp-lift` validates all pre-grasp/approach/lift paths without motion; `--execute-cached-grasp-lift` executes that prepared plan within its 60-second cache lifetime and stops holding after lift; it must never automatically continue to placement. See README for the physical trial procedure.

## Mock and Benchmark Workflow

Source ROS and `install/setup.bash` in every terminal. Keep real camera/arm drivers stopped and physical execution false during replay. Start each command in its own terminal:

```bash
ros2 launch mock_hardware_node mock_hardware.launch.py
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio grasp_candidate_backend:=hggd \
  placement_backend:=heightmap pipeline_profiling:=true \
  allow_physical_grasp_execution:=false
ros2 launch placement_node box_benchmark.launch.py \
  box_count:=10 optimize_pipeline:=true plan_pick_place_motion:=true
ros2 launch vamp_node grasp_debug_rviz.launch.py
```

Trigger the benchmark with `ros2 service call /box_benchmark/run std_srvs/srv/Trigger '{}'`. It returns after worker exit, prints a final per-box summary, and saves artifacts outside the repository (default `~/agx_piper_data/reports/box_benchmark`). `/box_benchmark/cancel` requests cancellation; `/box_benchmark/reset` clears placement markers. Keep the completed benchmark node alive for its latched RViz outputs.

The benchmark grounds/segments the tray using the configured real VLM server, creates known 8 cm box clouds, then calls the public cloud-grasp/search/validation APIs sequentially. It tries edge placement before universal heightmap stacking. Failed objects are skipped and remain collision obstacles. Full-motion mode validates lift, transfer and descent; it never executes these paths. This is planning simulation, not physics or hardware verification.

For the deployed image-based path on the mock cup:

```bash
ros2 run task_pipeline pipeline_task --object 'brown cup' --destination tray
```

Single-object placement keeps up to 16 coarse spatial/yaw candidates by default (`placement_candidate_count`). For a cup calculation after synthetic placements, keep the benchmark alive and add `--obstacle-cloud-topic /box_benchmark/scene_cloud` to `pipeline_task`; otherwise benchmark geometry is not part of the unchanged mock camera observation. The optional frozen XYZ overlay reaches both public placement search and pose collision validation, not RGB-D/HGGD input or full held-object motion.

The default recipe waits for its 17-step result (including floor-slot reservation) and returns a nonzero exit code on failure. It calculates only; placement IK failure must not be described as a successful placement.

See [README.md](README.md) for onboarding commands, [docs/CODE_MAP.md](docs/CODE_MAP.md) for the launched file hierarchy and API ownership, and [docs/STANDARD_PIPELINE_API.md](docs/STANDARD_PIPELINE_API.md) for schemas and completion rules.

## Engineering Journal

For calculation-only two-prompt grasp/lift tests, use `--prepare-grasp-lift`
then `--check-cached-grasp-lift`. The second calls the centralized
`validate_prepared_grasp_lift` API without actuation or consuming the cache.
Never substitute physical execution for this offline check or describe it as
verified actuator tracking/contact. Preparation and cache checks share nominal
waypoint-limit validation, including only the existing 1e-5 rad roundoff epsilon.

Maintain [docs/ENGINEERING_JOURNAL.md](docs/ENGINEERING_JOURNAL.md) whenever work
encounters a hard blocker. Record the date, a brief system-level problem,
the resolution or attempted approach, and its verified outcome. Update an
existing entry when the blocker is resolved rather than logging every retry.
Mark open blockers and untested changes explicitly; a successful build is not
proof that the runtime issue is fixed. Keep descriptions concise and preserve
earlier results as historical evidence. Before reporting completion, ensure
the journal reflects any major blockers encountered and their current status.
Keep detailed logs, captures and benchmark outputs outside the source tree;
link reusable project reports or summarize the relevant validation instead.

## Repository Output Hygiene

Keep the bundled eight CDR messages and `manifest.json` under `mock_hardware_node/snapshots/current`: they are replay inputs installed by the package, not build artifacts. `rgb_preview.png` is a derived local inspection image and is ignored. Keep reusable benchmark/RViz entrypoints and regression tests as source. Write run reports, captured debug images, CDR request/response dumps, rosbag recordings and custom snapshots outside the source tree or in an explicitly ignored output directory. Do not ignore all images/CDR/model files or remove assets required by a fresh clone.

The change inventory and review boundaries are in [docs/GIT_CHANGE_AUDIT.md](docs/GIT_CHANGE_AUDIT.md). Do not discard accumulated feature changes to reduce file counts.

Hardware drivers and RViz:

```bash
ros2 launch orbbec_camera gemini_330_series.launch.py
ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
  can_port:=can0 arm_type:=piper effector_type:=agx_gripper \
  auto_enable:=true control_enabled:=true speed_percent:=20 pub_rate:=50
ros2 launch vamp_node grasp_debug_rviz.launch.py
```

## Coding Style & Naming Conventions

Use 4 spaces in Python and C++; follow existing ROS 2 patterns: `snake_case` for Python files, functions, topics, and services; `PascalCase` for C++ classes and ROS message/service types. Keep computational mapping and planning in C++; keep orchestration in Python. Declare cross-node interfaces in `my_actions` rather than inventing ad-hoc topic contracts.

Respect existing ament linting (`ament_flake8`, `ament_pep257`, `ament_lint_auto`). Avoid committing debug images, local models, build output, or `.bak` files.

## Testing Guidelines

Place Python tests in a package's `test/` directory as `test_*.py`. Run affected package tests before submitting changes. For service or planner changes, exercise the relevant ROS service/action manually with camera and arm hardware disabled when possible; document hardware-only validation in the PR.

## Commit & Pull Request Guidelines

Use Conventional Commit-style subjects already used by the arm repository, for example `feat(vision_node): add voxel filter` or `fix(master_node): handle planner timeout`. Keep commits focused. PRs should explain the affected nodes/interfaces, provide build and test commands/results, link relevant issues, and include logs or screenshots for behavior or visualization changes.

## Safety & Configuration

`joint_limit_tolerance_degrees` is an operator-requested startup configuration
in the range 0--2 degrees (default 2). Apply it equally to all six master,
numerical IK, Cartesian validation and VAMP sampling limits. Preserve raw joint
feedback, FK and collision geometry; do not clamp measured joints or call this
an encoder calibration. Physical driver/controller limits are unchanged.

Do not hard-code machine-specific paths, CAN interfaces, credentials, or calibration values in code. The default hand-eye calibration is installed from `master_node/config/calibration/hand_eye_calibration.yaml` and may be overridden with the `hand_eye_calibration_path` launch argument. Changes to calibration, motion, collision, or emergency-stop behavior require explicit review.

Coordinate-transform invariant: perception and grasp computation must use only the synchronized manual `FK(joints) * X_matrix` result produced by `ActiveVisionServer`. Do not add TF lookup/listener fallbacks. Publishing that same matrix to a visualization-only TF frame for RViz is allowed, but computational nodes must never consume it.

## Current Grasp Pipeline Handoff

The approach is pre-grasp -> optional near-grasp -> final grasp. Near-grasp
is tried 2 cm then 1 cm before the final pose along the original retreat direction,
with final grasp orientation. Its joint-space transition is sampled at <=0.005 rad
and <=2 mm adjacent TCP displacement, with limits and collision checks; the final
short segment is fixed-orientation Cartesian IK. If either near route fails,
a direct joint-space transition to strict final IK is fully checked instead.
No fixed-position rotation is required. No failed or partial path is executable.
`NEAR_GRASP_USED` / `NEAR_GRASP_SKIPPED` identifies the selected route;
`alignment_waypoint_count` is zero (retained for interface compatibility).
The existing explicit approach finger-scene bypass applies to this approach;
arm/palm/self checks and complete preplanning remain mandatory.

Operator-selected `final_approach_check_finger_environment` defaults to false:
final-approach waypoint validation skips finger-body/pad versus scene checks
(spheres 82--95), retaining arm/wrist, palm/base and self-collision checks.
This exception applies only during approach, not grasp proposal validation,
pre-grasp validation, lift, transfer or descent. Successful plans expose
`FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED`; do not describe them as
fully finger/environment collision validated. Set the launch argument true
to restore the complete final-approach checker.

`[FIND_GRASP]` grounds the requested region, runs HGGD, validates and ranks its candidates, and caches the selected candidate. `[PRE_GRASP]` uses VAMP to reach the candidate's standoff pose, waits for the wrist camera to settle, captures synchronized RGB-D, and calls `/vamp_planner/refine_grasp`. The refiner caches and merges 5 cm-radius contact regions from the original and pre-grasp observations in `base_link`, then calculates an adjusted contact center, closing axis, minimum valid jaw opening, and strict final IK. Debug imagery is published on ROS topics and is not written to the repository.

RGB optical-flow/visual servoing and the old single-joint-target jump from pre-grasp to final grasp have intentionally been removed. The replacement is a bounded Cartesian final approach: it keeps the refined orientation fixed, interpolates 1–2 mm translation waypoints, seeds each IK solve from the preceding solution, rejects discontinuous joint changes, and prevalidates the complete sequence. Detailed finger/palm geometry remains active for swept-volume checks. The environment is blocking, while only the intended inner-pad contact region may be excluded near final contact. Do not add RGB tracking, computational TF lookup, APF steering, or unconstrained replanning to this final segment.

`[PLAN]` and `[PLAN_UPRIGHT]` use the separate placement node to generate object poses, then VAMP derives and validates the corresponding end-effector placement pose. The default backend is deterministic height-map packing; AnyPlace is retained as an optional backend.

Physical execution requirements: retain the existing `allow_physical_grasp_execution` gate; verify the arm is still at the cached refined pre-grasp; open to `min(0.10, refined_width + 0.01)` at the configured force; use 5% speed for the final approach; close to 0 m at 1 N only after the final waypoint; and restore 20% speed on every exit path. Never assume gripper contact from a command alone. Development and automated tests must not move the real arm or gripper.

## Ordered mock scene validation

The user-selected default `placement_bypass_attached_collision_during_lift`
is true. Set it false to validate attached-object/environment contact during lift.
Development tests must keep physical execution disabled. The bypass skips checking
only while lifting; preserve robot, finger, object-arm, transfer and descent
checks. Report the `LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED` response marker and
do not describe its lift as collision validated. When the user requests stopping
at a new blocker, report that blocker before attempting further tuning.
The placement backend may use bounded RRT-Connect retreat to validated pre-grasp
joints after ordinary lift routes fail. Preserve its local spatial/orientation,
joint-limit, midpoint and 2 mm TCP-step checks; this fallback must not change
the fixed-orientation final grasp approach.

The current replay capture contains a beige cup, two white bowls, an elevated
striped white plate and a dark tray. Use distinct spatial descriptions for the
bowls; do not treat an inferred plate stand as part of the plate. Avoid `and`
inside an object description because the template parser treats it as a task
separator. Layout calls preserve the original RGB-D for HGGD and pass the updated
immutable `FindGraspCandidates.collision_scene` through the public API for each
subsequent object. Successful source clouds are removed, virtually placed clouds
are added, and failures stay blocking. Optional source-bottom labels are described
in `docs/STANDARD_PIPELINE_API.md`. Keep attempts bounded and report full-motion
validation failures as failures, even when grasp or placement IK succeeded.

## Floor-slot placement

`floor_slots` uses upright floor-only square main slots (6–12.5 cm sides).
Rows follow the longest destination axis. Prefer two rows, scaling squares down
to 6 cm; if two minimum-size rows cannot fit, use one row and recalculate size.
Allow three or more rows only with full 12.5 cm squares. Compute inter-row and
inter-column gaps from available space rather than a percentage. Only leftover
strips may use rectangular slots. Use the largest inscribed usable rectangle
and non-overlapping extensions. Freeze geometry and IDs for occupied-surface
updates. `floor_slot_adaptive:=false` restores the legacy fixed grid.
Larger footprints reserve adjacent touched pitch cells, spanning the gaps.
Keep support/collision validation active. The public `commit_floor_slots` API
commits a matched source-header/pose only after successful placement validation;
failed searches or IK must not consume cells. Independent trials must reset
reservations. `/placement/floor_slots` shows blue free, orange reserved and green
selected cells in RViz. Bounded geometry/pending caches live in placement_node;
scene obstacles also block cells. Source timestamps and manual FK/X remain intact.

For ordered compact layouts, `layout_support_mode:=floor_slots` and
`floor_slot_share_unused_space:=true` enable non-overlapping 5 mm footprint
reservations within the visible adaptive grid. Whole-cell reservation stays default.
Search must reject reserved-mask overlap before IK and commit must recheck it;
full frozen-scene collision validation remains mandatory. Keep the 16-candidate
budget, within-tier spatial/yaw diversity, 1 cm phases and 5° yaw sampling.
Retain fine yaw alternatives near the best-ranked floor-slot pose (up to one
quarter of the same budget, within 20 degrees) alongside broad coverage; coarse
yaw separation alone can omit a narrow reachable placement orientation.
Document the extra search-time cost. The layout controller must remove accepted
source geometry and add its actual placed geometry before the next grasp/search.
Pinned grasp fallback must use the original synchronized cloud, RGB and manual
FK/X transform whenever any component is missing from its rolling cache.

Current user-selected floor-slot behavior is whole-cell exclusivity:
`floor_slot_share_unused_space:=false`. Treat cells like inventory slots: after
an accepted object commits them, no other object may use any part of those cells.
Do not enable optional sharing during tuning without an explicit user request.

Upright placement defaults to `floor_slots` in the recipe, CLI, VLM single-object
plan and ordered layout. Use the smallest validated exclusive cell group;
never reduce measured geometry to force a fit. Free-orientation VLM/layout
requests retain `auto`, and explicitly selected other support modes remain
available. The box benchmark keeps its explicit edge/stack sequencing.
The legacy single-object VLM plan must commit the selected grid pose through
`/pipeline/commit_floor_slots` after validation; candidate-only queries do not
consume slots. Exclusive search retains one frozen centered source cloud and
builds its selected occupancy mask lazily at idempotent commit. Remaining-space
ranking must exclude whole reserved cells, and equivalent groups share cached
scores. Keep source association, full projected containment and collision checks.

### Large floor square

Adaptive floor grids preserve at least two ordinary squares before forming at
most one larger square in spare space. `floor_slot_large_square_scale` multiplies
ordinary side length (default 1.5, 0 disables); no-fit falls back to ordinary
slots. Remaining space uses ordinary squares and rectangular remainders. Keep
whole-cell exclusivity, cached configuration separation and validation-before-commit.

Floor-slot matching tries one fitting slot first, including the large square,
then combinations using the fewest slots. Break equal-count ties by total reserved
area; do not prefer ordinary groups over a fitting single large square. Preserve
this priority in search order and scores consumed by VAMP, retaining spatial/yaw
diversity within equal-count/area tiers. This supersedes the previous ordinary-
group-first rule and is built-in behavior, not a policy option.

## Shared perception scale

The launch default ratio is loaded from
`master_node/config/calibration/perception_scale.json` (plate-derived uniform
ratio 0.976819, optical measurement / real dimension). A custom JSON can be
selected with `perception_scale_path`; explicit ratio overrides take precedence. ActiveVision divides optical XYZ once by `perception_measurement_ratio`
at ingress, before manual FK*X. Never scale transformed base-link coordinates,
robot geometry, joints, fixed motion distances or known-cloud synthetic geometry
again. HGGD derives depth from corrected observation geometry; optional
`/active_vision/corrected_registered_depth` uses 32FC1 metres and the raw camera
header. Raw camera topics/snapshots remain unchanged. The parameter is read-only
and requires stack restart to avoid mixed-scale cached frames. Replay success
verifies implementation, not physical calibration accuracy.

For tight inventory fits, preserve continuous slot centers and use grid-relative
yaw samples; the heightmap raster must not move the proposed pose into another
slot. Physical single-slot containment takes precedence over wide neighboring
gap territories. Commit still rechecks the accepted FK pose. Classify destination
floor contact relative to its measured normal, protecting scheduled objects and
raised geometry. Thin elevated plate/dish completion must use its local underside
and exclude lower stand geometry, never extrude it to a distant table.


## Plan-before-execution physical trial

The operator requested completing every calculation before moving. Use
`prepare_grasp_lift` / `get_prepared_grasp_lift` / `execute_prepared_grasp_lift`.
The default two-command trial is `pipeline_task --prepare-grasp-lift`, then
`--execute-cached-grasp-lift` for the same object. `--grasp-only` remains endpoint
preview and is insufficient to authorize prepared execution. Preparation uses
initial-observation HGGD contact geometry and immutable source/environment clouds
in `plan_final_approach`, validates the lift and pre-grasp transfer, and caches one
revision-bound plan. It performs no actuator calls. Execution performs no IK,
refinement or replanning; live start-state, pose, collision and contact checks
remain mandatory. A source-cache replacement, expiry or attempted execution
invalidates readiness. Any actuator failure becomes `STATE_UNKNOWN` with no retry.
The legacy close-up-refined workflow remains separate; do not insert it midway
into the prepared trial. Preserve the optional near-grasp transition, fixed-orientation final short
segment when used, joint continuity, configured collision checks, intended
inner-pad allowances and the explicit physical gate. Automated tests must not move hardware; isolate mock replay from
running real drivers using a different ROS domain.


The operator-selected `placement_bypass_robot_environment_during_lift` defaults
true. It skips robot/finger versus scene checks at the planned closed-gripper
initial state and throughout lift only; self-collision, object-arm, IK and joint
limits remain active. Transfer and descent restore scene checks. The independent
attached-object lift bypass remains configurable. Results expose
`LIFT_ROBOT_ENVIRONMENT_CHECK_BYPASSED`; this is not a fully scene-collision-
validated lift. Set both lift bypasses false for complete scene checking.


Commanded planning and measured feedback limits are separate. The master
`joint_limit_tolerance_degrees` defaults 2 degrees for feedback acceptance;
`planning_joint_limit_tolerance_degrees` defaults 0 and is passed to VAMP.
Prepared execution checks every requested waypoint against nominal URDF command
bounds before caching. Do not enlarge actuator target bounds to accommodate
encoder deviations: a 72-degree joint-5 target stalled at its 70-degree limit.
Execution failures identify stage, waypoint and worst joint target/feedback error.


Prepared grasp/lift selection retains the validated HGGD candidate pool.
Try at most eight candidates with a 30-second between-attempt search budget;
cache only a complete approach/lift/pre-grasp plan. Geometry and source revision
remain shared and immutable, and runtime execution never retries candidates.
Fixed-orientation lift routes remain first. A final bounded joint-space lift
fallback may relax orientation by up to 20 degrees, use <=4 cm endpoint lateral
offset (<=6 cm swept offset), and achieves the requested vertical clearance.
It samples <=0.005 rad joint changes and <=2 mm TCP steps, retaining configured
collision checks and nominal joint bounds. It may not descend >1 mm below the
initial height. No all-scene/all-object success guarantee follows from one test.


`/pipeline/prepared_grasp` publishes the actual complete prepared plan as a
latched MarkerArray (pre-grasp blue, grasp green). The default debug RViz
configuration includes Prepared Grasp and Pregrasp. A new preparation clears
the previous preview; inspect this preview when an alternate candidate wins.

Calculation-only grasp/lift recovery: `pipeline_task --prepare-grasp-lift` permits
one attempt by default for bounded latency. `--preparation-attempts 2` or `3`
opts into fresh-observation recovery after definite geometric rejection. Preserve per-attempt timings and clear
failed contexts before recapture. Never apply this retry to actuator execution,
unknown states, or invalid observations. Empty AUTO bowl grasps may try measured
RIM contacts once on the same immutable inputs through the shared API.

## Prepared-session public execution stages

Use `execute_plan` (`pregrasp`, `approach`, `lift`), `gripper_open` and
`gripper_close` for the standard prepared grasp/lift recipe. They share the
master-owned prepared record with the compatibility composite and require its
object, revision and ordered phase. `get_execution_state` exposes completed and
waiting stages; `plan_lift` delegates the existing solver in lift-only mode.
Do not invent actuator retries or expose arbitrary waypoint slicing.
`pipeline_task --queue-grasp-lift` queues the whole recipe and pauses before
`pregrasp`; report the actual pending master tasks rather than a proposed list.
An action with `stop_before_step_id` may succeed with
`PAUSED_AT_STEP_BOUNDARY`; this does not mean pending physical motion executed.

Plate calibration uses `ros2 run master_node calibrate_plate --diameter 0.174
--samples 5 --output "$HOME/agx_piper_data/calibration/plate_scale_calibration.json"`. It captures through
the shared observation/grounding/segmentation APIs, fits a robust rim plane and
measures multiple planar silhouette widths. A single plate view does not identify
independent XYZ factors or absolute depth bias; preserve that limitation in reports.

## Prepared execution recovery

Preparation expires after 60 seconds only before execution starts. Active
prepared execution freezes its source and uses `prepared_execution_timeout_sec`
(default 300 seconds) at stage boundaries. Never reapply observation TTL between
physical stages. On prepared action failure, flush matching idle object queues;
clear session/source/refinement caches only for a known fresh idle empty hand.
Keep failure results and other/busy contexts. Do not automatically retry commands.
`reset_execution_session` is non-actuating and rejects unknown/held state.
`reconcile_empty_hand` requires explicit operator confirmation plus fresh
stationary feedback, no active collision and no running pipeline. Never invoke
it automatically to erase `STATE_UNKNOWN` or assume a failed close missed the
object. A reset leaves the arm where it is; new preparation uses fresh feedback.

## Base-frame physical grasp trial offset

Load `config/calibration/grasp_position_offset.json` with `grasp_offset_path`.
`grasp_position_offset_base_m` defaults to [-0.005, +0.005, +0.005] metres in
base_link. Apply it once to newly constructed image-based grasp/refinement
Cartesian targets before IK and collision validation; downstream approach,
lift and cached execution consume corrected targets without reapplying it.
Do not shift scene geometry, measured contact samples, joint angles or X-matrix
translation. Known-cloud synthetic proposals remain unoffset. This is empirical
command correction, not verified hand-eye/TCP calibration. Restart and reprepare
after changing the read-only setting; preserve safety gates and IK tolerances.

## Joint-6 command-only compensation

The user-selected convention is command-only: preserve raw feedback throughout
planning/FK/perception and check nominal joint limits before adding the offset.
Load `config/calibration/joint_command_offset.json` by default with no new launch
arguments. Its `joint6_command_offset_rad` is 0.23499113048851653, in raw radians.
Add once in the common master publisher. Never subtract it from feedback or
change X-matrix/FK/model bounds. Arrival/handoff verification tracks the sent
raw command; gripper-only and raw-position brake commands do not add it. Cached
paths remain nominal. Collision/limit checks do not validate the added rotation;
no post-addition wrapping/clamping. Driver-native homing/direct commands are
outside this compensation. Physical results remain operator-test-only.

## Local output organization

Use `~/agx_piper_data/{calibration,logs,reports,captures,tools}` for local task
outputs. Do not scatter new logs/images/CDR/reports into the home root. Installed
source calibration files remain beside master_node; user override JSON copies
are in `~/agx_piper_data/calibration`. Standard ROS logging stays in `~/.ros/log`.
Resolve older journal artifact paths with
`~/agx_piper_data/relocation_manifest.json`; retain historical evidence. Keep
unrelated home files/source repositories untouched and do not delete diagnostic
artifacts merely to reduce clutter.

## Perception translation calibration

The existing grasp-offset JSON now supplies a global perception translation in
base_link via ActiveVision. Compose it once with synchronized manual FK*X; leave
raw optical data unchanged. Map downsampling precedes corrected transformation;
organized HGGD observations retain pixel correspondence and use the same matrix.
The reasoning launch must keep VAMP target offset zero to avoid double correction.
Restart and regenerate plans after changes.

Initial image-based grasp pose collision validation uses the coarse obstacle
world only, per operator request. Do not reintroduce the candidate-centric
detailed local-cloud pose check. Measured contacts/finger closing sweep, IK,
pre-grasp and motion checks remain active.

Initial final-grasp environment checks omit gripper spheres 74..95 by default,
checking arm/wrist 0..73 against the coarse obstacle world. This applies only
to candidate IK acceptance and candidate final-pose validation; full robot
self-collision and other stage policies remain unchanged. No launch switch is
required. Do not infer that gripper/environment clearance was validated.

Approach planning requires final_grasp_joints from the validated candidate or
refinement. Do not re-solve this final endpoint from optional intermediate seeds.
Keep reachable optional intermediates and validate the connection to the supplied
endpoint; skip optional intermediates when needed. No unchecked physical jumps.

Final approach environment checks omit palm sphere 80 by explicit operator request; matching diagnostics use the same exclusion. Other callers retain sphere 80, and self-collision remains full.

Sphere-80 policy update: approach checks include sphere 80 again. Initial grasp
calculation detects sphere-80 environment collision explicitly and attempts
bounded 2..10 mm whole-gripper retreat with preserved measured contacts, IK,
actual finger-pad reach and arm clearance validation. Trigger only on sphere-80
collision; reject when bounded recovery cannot clear it. This supersedes the
earlier sphere-80 approach exclusion.

Palm recovery trigger list is now spheres 80 and 81; accepted recovery must clear both. Other stage policies are unchanged.

Initial grasp-pose environment checks are now limited to sphere indices 0..67
(inclusive), per operator request. Palm spheres 80/81 retain their separate
bounded contact-preserving recovery. Other collision stages and full self-collision
remain unchanged. This supersedes the earlier initial 0..73 scope.

Grasp motion scope: approach interpolation and grasp-tagged VAMP pre-grasp
planning use only environment spheres 0..67. CheckAndReplan.grasp_motion defaults
false; grasp controllers set true explicitly. Empty-hand general moves and held
placement transfer keep full environment sphere checks. Full self-collision is
retained. Grasp RRTC uses the per-lane scoped validator, not obstacle deletion.

Obstacle-world publication applies one-pass 26-neighbor filtering: keep a voxel
only with at least two distinct occupied neighbor cells at obstacle resolution.
Count neighbors before pruning; never recursively erode surfaces. This does not
filter organized object clouds or prove removed samples were noise.

Upright footprint completion projects along base_link gravity, retaining measured
XY. Source support fitting estimates height; preserve elevated underside/stand
separation and physical dimensions. Do not use a noisy tilted source-plane normal
to shear the footprint.

### Pre-grasp robustness update (2026-10-14)

Operator-selected pre-grasp orientation tolerance defaults to 16 degrees;
preferred standoff is 0.156 m and allowed upper standoff is 0.195 m. Retain
shorter target-visible alternatives and complete approach/lift preplanning.
Grasp RRTC admits raw start feedback within the master's existing 0--2 degree
feedback tolerance only through a fully collision-checked transition to a
separate in-bounds entry configuration. Do not modify raw feedback/FK, enlarge
command/IK/sampling bounds, or send the initial measured sample as a command.
Reject larger deviations and any collision along the entry transition.

### Prepared approach/lift speed (2026-10-14)

Operator update (2026-10-20): prepared streaming uses 100 Hz and 0.1575 rad/s
joint timing (30% below the previous 0.225 rad/s bound). This applies
to approach, lift, transfer, descent, withdrawal and retreat along unchanged
validated edges. Driver approach/lift cap is 8%
(the integer setting rounds 7.5% upward); pre-grasp and exit restoration stay
20%. Preserve collision/contact/fresh-feedback monitoring, exact frozen endpoints
and final arrival verification. Post-motion TCP checks allow up to 2 seconds
for fresh feedback to settle, with unchanged pose tolerances and continuous
contact/collision/freshness monitoring; start-state checks stay immediate.
This supersedes the earlier prepared 5% cap;
do not change geometric validation spacing or skip unvalidated path corners.

### Gripper targets update (2026-10-14)

Operator-selected jaw commands supersede the previous +1 cm/open, zero/close:
use shared `grasp_gripper_widths` for prepared, staged and compatibility execution
and release. Opening = min(0.10, grasp width + 0.02) m; closing = max(0, opening
- 0.04) m, with existing 1 N closing force and measured contact confirmation.
Default pre-grasp/approach opening validation also uses +0.02 m. Preserve measured
contact geometry and held-object aperture assumptions; actual aperture is limited
by the object, not assumed equal to the commanded closing target. No actuator
calls in automated tests.

### Object-upright held-motion policy (2026-10-14)

`placement_motion_orientation_policy` defaults `keep_upright`; `free` explicitly
restores unconstrained held orientation. `placement_motion_upright_tolerance_rad`
defaults 0.05235988 (3 degrees), >0 and <=0.17453293. Treat source upright object
+Z as base_link gravity; derive its current axis from FK times inverse fixed grasp
attachment, not tool roll/pitch. Yaw remains free. Enforce before lift bypasses,
in all lift fallback checks, every transfer RRTC validation lane, descent and
returned-edge densification (<=0.005 rad and <=2 mm TCP). Never silently relax
upright when constrained planning fails. Success reports OBJECT_UPRIGHT_VALIDATED;
existing collision-bypass markers remain independent. Public APIs/schema and
master-owned sessions are unchanged; both deployment and benchmark use this same
planner policy. Restart and prepare fresh; tests must not actuate hardware.

### Hollow-object foreground recovery (2026-10-14)

Do not treat median camera range as complete object membership. Keep the existing
foreground band and anchor selection, then recover substantial semantic-mask
components that connect to accepted measured foreground in 3-D (existing neighbor
radius, at least three supported samples; existing component size/extent gates).
Use the same recovered validated_target_mask for HGGD, object world and layout;
do not globally widen the depth band, include tiny/distant islands or synthesize
hidden exterior surfaces. Exact source association and raw measurement geometry
remain unchanged.

## Bounded outward grasp bend

Image-based measured-contact grasps use best-effort outward bending from the
original refined grasp about its local tool X axis through the original contact
region. Estimate one local contact-patch surface normal at the measured pair
midpoint; do not align the two contact normals independently or add a free twist.
Try a maximum finger-long-axis angle to that surface of 10°, then 30°, then 50°;
the larger target angles require less aggressive bending. Recompute jaw opening
by projecting the same measured contact pair onto the rotated closing axis;
never resize object geometry or invent contacts. Verify actual IK/FK angle,
original hinge axis, measured contact reach, finger sweep, palm/collision and
pregrasp checks. Proposed and actual hand-body motion must remain outward from
the measured object center. The original validated pose is the fallback.
Accepted results retain PAD_SURFACE_ALIGNMENT_VALIDATED or
ANGLE_TUNING_FAILED_ORIGINAL_USED markers. The former refers to this local
finger-to-surface angle, not proof of a physically stable pinch. Known-cloud
benchmark generation is unchanged. Preview markers show up to three actual
10°/30°/50° trials associated with the selected grasp, including rejected trials;
never draw an unrelated best-fit rotation as a requested-angle trial.

Outward angle sweeps are dynamic: try only surface-angle targets strictly below
the actual current angle, in order 10°, 30°, 50°, stopping at the first valid
result. Thus 90° tries three, 45° tries two, 20° tries one, and <=10° does not
sweep. An already-good pose reports ANGLE_ALREADY_WITHIN_BAND and no trial
markers. The original validated fallback and ordinary validation still apply.

A tilted grasp whose measured contacts exceed the usable finger depth may
advance along its approach by at most 10 mm total, with at most two seeded IK
retries. Preserve the existing physical pad bounds (including the 2 mm tip
margin), contact pair, jaw and height checks. Revalidate actual IK/FK angle,
original hinge, outward hand position, sweep, collision and pregrasp; recheck
pad reach after palm recovery. Accepted corrected results expose
CONTACT_DEPTH_RECENTERED. A failed recenter retains the original validated
fallback, never a partial or invalid corrected pose.


## Complete prepared placement (2026-10-17)

`pipeline_task --queue-pick-place --object 'striped white plate' --destination tray`
calculates the full image-based pick/place plan and pauses before `pregrasp`.
The same master queue retains pregrasp, open, approach, close, lift, transfer,
descent, release and floor-slot commit, bound to its prepared revision.
`--check-cached-pick-place` validates readiness without actuation;
`--execute-cached-pick-place` is explicit gated execution of the reviewed cache.
Keep the existing source/cache expiry and no-replanning/no-actuator-retry rules.
Do not describe a paused calculation as executed motion or consume slots during
preparation. Full prepared transfer/descent/release share the existing prepared
owner; tests use fake actuators and mock replay only.

When fixed-orientation lift fails, bounded yaw-only endpoints may turn about
base_link Z (up to 20 degrees) and move inward up to 4 cm while preserving the
requested rise. Every sampled transition retains upright, nominal-limit,
continuity and configured collision checks. This does not relax object tilt.
Frozen held-scene construction may associate an original-mask-connected RGB/depth
seam within 4 mm and a non-iterative 5-pixel band/rectangle margin; retain bottom,
destination and depth-discontinuity protections. Scene-2 plate validation and
runtime measurements are recorded in the engineering journal.

### Placement search/collision performance

For exclusive adaptive floor slots, fully validate the single-slot tier first;
only if it returns no candidates run combination matching. Preserve every yaw,
physical containment, support and obstacle check. Shared sub-cell and legacy
searches keep their original behavior. Candidate count is an upper bound, not a
requirement to fill the shortlist with worse slot-count tiers.

Attached-object scene checking may skip an entire source/block only when a
conservative transformed bounding box is disjoint from occupied coarse cells.
Occupied regions still use the original ordered 3 mm point tests and all support,
departure and segment-wide noise-budget rules. Bounds cache geometry only; require
an exact immutable source snapshot match and update occupancy for every added
scene point. Never cache a collision decision or use broad-phase occupancy as a
replacement for the narrow-phase test.
Exact point-index queries may clip loops to the global occupied CELL envelope.
Keep original cell-boundary padding and predicate order; metric point-AABB
pruning is not equivalent for arbitrary cell predicates.

### Practical placement tolerance (2026-10-19)

Prepared placement allows 5 mm TCP position error and 10 degree target orientation
error by default (`placement_position_tolerance_m:=0.005`,
`placement_target_orientation_tolerance_rad:=0.17453293`). Object upright tilt
is limited to 5 degrees throughout held motion; yaw remains unconstrained by
that upright check. Local descent tracks the accepted FK orientation within
0.01 rad. Preserve nominal joint limits and collision validation; lift solves
retain their previous 1 mm position accuracy. Floor-slot targets aim 8 mm above
the measured floor (3 mm release clearance plus the 5 mm IK position allowance)
rather than requiring exact noisy surface contact. Descent tracks position within
1 mm; final target acceptance includes that local tracking allowance.

The placement response carries the actual achieved object pose with the exact
source header. Slot commit/dry-run may associate it with a retained candidate
within 25 mm object-center displacement and 12 degrees; these are association
bounds, not IK acceptance bounds. The complete achieved projected footprint
must fit the same exclusive slot group, including physical single-slot bounds.
Reject stale sources, occupied cells and geometry outside that group. Use actual
geometry for occupancy and preserve idempotence. Do not compare against the exact
original footprint as an additional placement constraint. Restart and reprepare;
no physical success is implied by calculation-only validation.
When a completed lift endpoint fails the full transfer checker, shared
calculation-only validation may retry with 1 cm additional lift, bounded to
2 cm above the requested minimum. Every recovery reruns the
normal lift/transfer/descent checks; no transfer collision bypass or actuator
retry is permitted. Per-backend timing records include the attempted lift height.

### Multiple-grasp complete-plan selection (2026-10-19)

Image-based HGGD keeps up to six geometrically validated, diverse ranked grasps
for single objects and layouts; fewer may exist. Do not truncate single-object
responses to one pose. Keep proposal, IK, collision, diversity and pre-grasp path
checks. Known-cloud benchmark pool behavior is unchanged. The shared prepared
session tries its cached candidates in rank order (at most eight, existing
30-second between-attempt budget), validating approach, lift, placement, transfer,
descent and pre-grasp travel before caching the first complete plan. No actuator
calls or retries occur during this search. Failed candidates do not reserve slots.
Logs `[PREPARED GRASP TRIAL]`, `[PREPARED GRASP REJECTED]` and
`[PREPARED GRASP SELECTED]` identify attempts and the actual RViz/execution winner.
Existing APIs and physical execution gates are unchanged. Building a larger
validated pool can cost more than the former first-valid-grasp shortcut.

### Lift distance and avoidance scope (2026-10-19)

Lift requests have no fixed 10 cm maximum or 5 mm minimum. CLI recipes accept
finite positive metres; the engine retains zero as its existing default sentinel
and rejects negative/nonfinite values. Actual robot reach, nominal joint limits,
upright policy, configured collision checks and planning budgets still apply.
Higher-clearance recovery stays bounded to +1/+2 cm above the request, without
clipping to 10 cm. Cartesian planning checks its 4096-waypoint resource budget
before allocation; removing the request ceiling must not create unbounded memory.

The existing `plan_apf` API plans the bare arm from joint targets and must not
be substituted directly for held-object departure. The shared `plan_lift` and
full `plan_placement` backend now use the payload-aware Cartesian escape below;
this does not change the bare-arm API. Existing inward/retreat lift fallbacks
remain before the endpoint escape.

### Shallow measured-contact grasps (2026-10-19)

Final measured HGGD routes, including original-angle fallback, must keep both
contacts inside the physical pads and open-finger geometry at least 3 mm above
the reliable local table plane fitted from the immutable synchronized scene, not
the floor-filtered coarse collision map. Try bounded contact-centered targets with
4/7/10 mm tip engagement (three seeded strict IK trials, <=20 mm TCP shift,
<=10 mm table-normal correction), preserving measured contacts, nominal limits,
palm/coarse collision, angle and pre-grasp validation. Reject missing plane or
unrepairable contacts; never fall back to a table-penetrating original pose.
Expose SHALLOW_CONTACT_VALIDATED / FINGER_TABLE_CLEARANCE_VALIDATED.
Known-cloud generation and configured approach/lift bypasses are unchanged.

### Stack-wide robust API outcome policy (2026-10-19; implementation direction)

This policy applies to every public API and composite recipe, not only lift.
The operator reports roughly 80% calculation failures in the bowl pick/place
workflow; this is a reported live reliability problem, not a measured benchmark.
Optimize for completed user tasks in realistic noisy/crowded scenes, rather than
fast first-candidate rejection or success on one easy recorded frame.

- Recover from definite calculation failures within bounded per-call and whole-
  task budgets: try alternative candidates, IK seeds, reachable approach routes,
  placement positions/yaws or the documented relaxed tolerances before stopping.
  An optional intermediate waypoint must not invalidate an otherwise validated
  complete route when a checked direct transition is available.
- Treat sparse unsupported depth samples as noise using explicit, tested spatial
  support/contact rules. Do not let an isolated point terminate a task. Preserve
  real object extent, source association and consistent object/obstacle geometry;
  do not label every inconvenient sample noise or erase supported obstacles.
  Contact tolerance is phase-specific; do not propagate departure allowances to
  transfer, descent or all robot collisions without explicit authorization.
- Grounding/segmentation recovery may refine the description or request a fresh
  observation when available. Never invent a detection or use mismatched/stale
  RGB-D. With disconnected hardware, use the existing valid recorded inputs and
  report when new evidence is required instead of waiting indefinitely.
- Return an actionable outcome: a complete validated result, a bounded recovery
  operation routed through the same catalog, or a specific operator-required
  action when autonomous recovery is exhausted. Preserve truthful success/status
  semantics; a suggested action is not a successful grasp or placement. Do not
  silently replace the requested task with a different object or destination.
- Keep recovery ownership in the shared API/master session, so deployment,
  agent callers and benchmarks receive the same behavior. Avoid nested retry
  explosions, duplicated private planner paths and extra arguments for routine
  recovery. Record attempted alternatives, selected fallback and total API time.
- Planning recovery must finish before execution. Preserve complete-plan
  validation, nominal command limits, self-collision, explicit execution gates,
  revision checks and cancellation. Never retry actuation after STATE_UNKNOWN
  or clear an unknown/held state automatically to manufacture readiness.
- Validate repeated full recipes on multiple recorded scenes, including the
  actual failing bowl/cup scene and crowded cases. Target at least 90% complete
  calculation success, reporting attempts, failures, runtime distribution and
  active bypasses. A recovery suggestion is not counted as task success. Mock
  success does not establish physical contact/tracking reliability.

Implement incrementally from measured failure categories and journal verified
outcomes. This policy does not claim these recovery mechanisms are all present
or that a valid physical action can always exist.

### Contact-tolerant lift endpoint escape (implemented 2026-10-19)

The operator accepts slight contact with nearby objects during departure and
wants useful recovery in crowded or noisy scenes. Do not treat every obstacle
sample or shallow contact as an immediate terminal failure. Sparse contacts are
not proof of noise, and an executable motion cannot be guaranteed in every scene.

Implement this sequence through the shared APIs and master-owned prepared session:

1. Calculate the initial 5 cm lift using the operator-selected lift collision
   bypasses. Preserve nominal joint limits, self-collision and existing object-arm
   checks. Report the bypasses; this lift is not scene-collision validated.
2. At its predicted endpoint, evaluate both the arm and attached object against
   the frozen obstacle scene. If needed, use payload-aware APF to propose a short
   upward/sideways escape while retaining the object's upright constraint.
3. Validate the entire escape with bounded contact tolerance and an overlap
   reduction criterion. Allow shallow separating contact; reject substantial
   worsening penetration. Define and test explicit contact, displacement,
   iteration and time budgets rather than silently disabling all checks. Existing
   segment-wide sparse-contact accounting must not reset at each waypoint.
4. Once clear, restore normal scene checks and plan transfer, descent and release
   from the escaped state. Calculate all stages before any physical execution;
   never lift first and defer placement planning until afterward.
5. Cache only a complete accepted plan. If escape or transfer fails, try bounded
   alternatives or another validated grasp during calculation. Otherwise return
   a clear recovery-required result with the blocking stage and suggested next
   action. Never fabricate a successful plan, execute a partial path or retry an
   actuator automatically after an unknown-state failure.

The payload-aware APF is implemented in `vamp_node/payload_escape.hpp`, inside
the same held-motion backend used by `plan_lift` and `plan_placement`. It uses
exact immutable obstacle/source points and arm/finger geometry; 5 mm cells count
contact extent rather than discard points. The initial lift retains its explicit
bypasses. APF runs only when that endpoint fails the ordinary full checker.
A successful escape must end in strict full clearance. Its returned waypoints
are part of `lift_path_flat`, and `lift_ee_pose` is the actual escaped endpoint.
`LIFT_APF_ESCAPE_VALIDATED` identifies recovery; failure returns no partial path.

Escape bounds: 32 iterations, up to five direction alternatives with 5/1 mm
backtracking trials per iteration, 4 seconds, bounded proposed steps, <=10 cm lateral/rise and <=12 cm total endpoint displacement.
Permit at most 10 mm padded penetration, no more than the initial penetration
plus 0.5 mm; overlap energy must decrease within a bounded 25%/roundoff allowance relative
to both the best/current segment state and its initial value; the allowance
must not accumulate at successive waypoints.
Accumulate at most 128 unique 5 mm contact cells over the complete escape,
with no more than four additional currently overlapping cells over the initial
count. This departure-only budget is independent of the existing 32-point/10-cell
attached-object sparse allowance; do not apply it to ordinary transfer/descent.
Preserve nominal limits, self/object-arm checks, upright policy and sampled
<=0.005 rad / <=2 mm TCP joint edges. The full prepared recipe calls public
`plan_lift` before `plan_placement`; complete placement revalidates departure
against the same immutable scene rather than trusting a caller-supplied path.
Ordered layouts explicitly request 5 cm lifts. No new launch arguments or
actuator retries are introduced. APF can get stuck; do not claim universal success.
Preserve source revisions, bounded recovery and non-actuating validation.

## Master-owned active manipulation memory

`master_node/manipulation_memory.py` owns bounded active object geometry (eight),
one active destination/grid and minimal name/base-link-position hints (256).
Use catalog APIs `resolve_target`, `refresh_active_target`, `get_active_target`,
`forget_active_target`, `get_manipulation_memory`, `calculate_active_grasps` and
`find_active_placements`; `get_floor_slots` reads placement-owned reservations.
Default CLI recipes use memory; `--fresh-perception` retains explicit inference.
Keep source headers, current-mask association, depth/view/calibration/age checks
and frozen destination grid identity. Invalid geometry must be reobserved;
name/location hints alone never authorize manipulation. Failures preserve valid
source data for a new calculation step, not automatic actuator retries. Print
actual pending/completed tasks and active names/XYZ at step boundaries. Only
confirmed release updates physical location; planning/slot commits do not imply
motion. Switching destinations releases active geometry and idle prepared/grid
state, retaining hints. Held/unknown execution state blocks memory mutation.
See the memory section in docs/STANDARD_PIPELINE_API.md for commands and limits.

## Final grasp skill consistency

Preserve final measured-contact checks after angle refinement and actual FK:
contacts must be within 3 mm of opposite inner finger faces. Reclassify an
initial PINCH as POWER_GRASP when final jaw opening or full measured contact
span exceeds the existing pinch_max_opening_m limit (50 mm default). Rank
validated pinches before other skills, preserving quality/path ranking within
each group and bounded complete-plan fallback. Do not treat an early local-skill
label as evidence of final pinch geometry. Prepared logs expose final skill and
opening. A power fallback must not be described as a successful rim pinch.

## Final prepared-plan RViz review

Default grasp_debug_rviz uses `/pipeline/prepared_grasp`, `prepared_placement`,
`prepared_nominal_object_cloud` (grey requested slot) and
`prepared_placed_object_cloud` (cyan achieved object). Master publishes final
endpoints only after complete preparation and clears them on startup/new plans/
resets/failure. Reuse VAMP's returned finger dimensions and poses. Achieved
geometry is placement FK times inverse selected grasp FK applied to the exact
immutable source cloud; retain source headers and compare against the requested
slot pose/center. Never show nominal geometry as achieved after relaxed IK or
intermediate successes as a complete accepted plan. Visualization errors must
not change execution readiness. Actual here means predicted FK, not measured
physical object motion. Benchmark and candidate trials stay out of this focused
config; arm collision spheres remain optional and disabled by default.

## Post-release next best view

Prefer steep top-down camera views from the target's base-facing side. Public
plan_next_best_view uses fresh shared capture, manual calibrated optical-to-TCP
conversion, existing pose IK and VAMP with full empty-hand collision checking;
execute_next_best_view delegates the guarded execute_trajectory API. Keep their
revision/expiry/start checks and no actuator retry on unknown outcome. Default
camera-to-target distance is 45 cm with bounded shorter reachable alternatives.
Prepared pick/place must plan and validate the retreat before any actuation,
against the predicted scene after placement, under the same prepared revision.
After confirmed release and floor-slot commit, execute_placement_retreat follows
the cached path without capture, IK or replanning; recapture only at the distant
viewpoint. The wrist camera is too close to the object immediately after release. View failure must preserve completed
placement and reservations; never retry physical placement because its later
camera task failed. Capture does not clear occupied slots. Test on isolated mock
ROS domains with physical execution false, without moving real hardware.


## Empty-hand release withdrawal

Operator update (2026-10-20): the existing `placement_bypass_robot_environment_during_lift` flag (default true) also bypasses robot/finger-versus-scene checks for the initial released state and the short empty-hand withdrawal. Self collision, nominal joint limits, IK, no-descending and dense edge checks remain active. Results explicitly expose `WITHDRAWAL_ROBOT_ENVIRONMENT_CHECK_BYPASSED`; this is not scene-collision validated withdrawal. The subsequent distant retreat restores full scene checking and must pass before accepting a complete plan. Set the existing flag false to restore the contact-separation policy described below. No new launch argument is introduced.


`plan_lift` supports explicit `empty_hand=true`, frozen placement_scene,
start_joints and jaw_opening_m, with no source payload cloud. Held mode stays
unchanged. Prepared pick/place plans a 3 cm withdrawal before the distant retreat;
execute_withdrawal follows confirmed release/slot commit and precedes
execute_placement_retreat. Both are cached/revision-bound before any actuation,
and execute without capture/IK/replanning. Capture only at the far view.
Only samples labelled released_object=1 in the initial finger contact region
(including a 2 mm sphere/mesh proximity shell) may receive contact separation
allowance. Freeze the original overlap-energy budget; do not reset it per waypoint.
Keep aggregate squared penetration within its original budget, with bounded
numerical slack, a 3 cm contact window and fully clear terminal pose. Tangential
sliding is allowed; do not demand an arbitrary decrease per millimetre. Initial padded-sphere overlap
is capped at 30 mm; the current maximum may exceed its original value by at most
2 mm. These geometric proxies are not physical penetration or evidence of noise.
Keep neighboring obstacles, arm/palm/self checks and nominal joint/dense edge
validation active. This is independent of attached-object sparse contact budgets.
Withdrawal tries vertical/reverse-approach paths first, then bounded two-leg escape paths: 1.5 or 2.5 cm along the reversed approach or initial contact normals, followed by the requested rise. No route may descend; the same frozen contact allowance and fully clear endpoint apply. `OUTWARD_WITHDRAWAL_USED` identifies a two-leg route.

Even with the scene bypass, never bypass self-collision or joint-limit checks. Preview on /pipeline/prepared_withdrawal.

## Geometric target visibility

`check_target_view` is a read-only memory-aware public API. It may capture via
capture_observation and call check_target_visibility, but must never call VLM,
segmentation or motion APIs. Reuse the synchronized manual FK*X transform and
registered-camera intrinsics. No computational TF fallback. READY/LOOK_REQUIRED/
LOCATION_UNKNOWN are successful decisions; errors are failures. Callers must
inspect status. Current-view ray casting uses a frozen 5 mm tree and bounded
surface samples, excludes target-occupied cells and stops 10 mm short of target
samples. Unknown space is transparent: READY is expected visibility, not object
presence, depth sufficiency or collision safety. Directory-only center checks
expose location_only=true. Do not silently make this API move the camera, alter
memory or consume tray slots. Existing next-view and motion policies stay intact.


## Persistent destination geometry

The active destination retains an immutable tray base, composite cloud and frozen
grid IDs until switching destinations or explicit forget/reset. Camera movement,
occlusion, refresh or age must not silently erase this inventory. Calibration
mismatch requires explicit reset. Object observation freshness checks still apply.
After confirmed prepared release, append exact source geometry transformed by
actual achieved EE FK and cached grasp attachment, once per plan/source revision.
Failed release/calculation must not append. Preserve original map/source headers;
the composite is derived geometry (`placed`: base=0, placed object=1).
Pass immutable base and placed obstacles via PrepareHeldScene; SearchPlacements
uses destination_base_cloud for stable support/grid identity and merged geometry
for obstacles. Keep placed points out of floor-support exemptions. Preserve
whole-cell reservations and explicit commit validation. Capacity checks (128
placements / 500,000 mapped points) happen before preparation; never silently
remove obstacles. Automated validation simulates release bookkeeping only and
must not call physical gripper/motion APIs to update memory.

## Guided joint lift and whole-plan grasp selection

Prepared pick/place must accept a grasp only after its approach, lift, transfer,
descent, withdrawal and retreat are all prepared. A definite planning failure
tries the next retained grasp within the existing eight-candidate/30-second
budget; execution never retries candidates or uses a partial path.

Held lifting first tries differential joint-space ascent: requested height and
object upright tolerance remain constraints, while XY and yaw may vary. APF
scene guidance and joint-margin centering act in the free task directions. Keep
nominal command limits, self/object-arm checks, midpoint checks, <=0.004 rad joint
steps and <=2 mm adjacent TCP steps. The lateral envelope is bounded by
min(25 cm, requested rise + 6 cm), with a 1.5-second total guided-attempt budget;
failed routes return no executable prefix. Existing Cartesian/bounded fallbacks
remain available. All configured lift bypasses remain explicit in responses;
transfer restores full scene checks. `GUIDED_JOINT_LIFT_VALIDATED` identifies this
route. Do not apply APF or free-orientation joint ascent to final grasp approach,
transfer, descent, or empty-hand withdrawal. Never shorten a requested lift or
relax the object upright tolerance to force success.


## Compulsory per-object prepared sequence (2026-10-27)

Standard pick/place preparation includes look, grasp, lift, place and retreat.
Bind the owner-validated `PlanNextBestView` response into preparation and plan
pre-grasp from its endpoint. Geometric READY may satisfy look without motion;
otherwise execute the frozen look through the same prepared session before
pre-grasp. Queue-only must pause before look. Default post-release retreat is a
bounded free-orientation clearance path, not a mandatory far tray camera pose.
Withdrawal must end at a full-scene-clear state before retreat; retain explicit
withdrawal/lift bypass markers and all self/joint checks. The next object checks
visibility independently. Frozen replay cannot verify moved-view acquisition or
physical actuator/contact behavior; report those limits separately.

Successful complete pick/place execution must retire transient execution caches
and ownership after empty-hand retreat and speed restoration. Preserve destination
memory, placed geometry, occupied slots and history; keep only a bounded completion
summary for inspection. Do not retire held, incomplete or unknown sessions. Test
completion of one object followed by caching the next, not only independent trials.

## All-target perception inventory (2026-10-28)

Standard image-based pipeline recipes must call the shared `perceive_scene`
API on one synchronized observation before HGGD. Both batch presence and batch
bounding-box prompts include the complete configured target list and destination;
segment each observed target and select it through `get_active_target`, rather
than issuing separate single-object VLM requests. Preserve stable target order
when the active object changes. Defaults live in
`master_node/config/perception/scene_targets.yaml`; explicit scene object lists
override them, with the requested active object always included. Keep single-
object APIs available for explicit callers. Never treat returned center points
as boxes or infer a missing object's grasp cloud from its old location. Retain
unseen-object directory hints and frozen destination/grid/placed geometry.
Inventory is non-actuating and does not implement automatic camera look recovery.
