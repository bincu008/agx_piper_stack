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

- `task_pipeline/task_pipeline/registry.py` is the authoritative function-name/type/backend catalog. Production controllers and benchmarks must use it rather than introduce private engine-call sequences or duplicate endpoints.
- Public services use `/pipeline/<function>`; list them with `/pipeline/list_functions` (`my_actions/srv/GetPipelineFunctions`). Declare message/service/action schemas in `my_actions` and rebuild consumers after changing them.
- The master-owned `StepStore` executes typed `PipelineStep` requests, resolves predecessor result bindings, records per-step status/timing and depletes attempted tasks. `@response_cdr` binds a complete serialized response. Never use eval or an arbitrary endpoint supplied by a caller.
- Use `submit_steps` plus `/pipeline/run_steps` (`my_actions/action/RunPipelineSteps`) when callers need final completion and feedback. `run_step` executes one operation. The legacy `execute_pipeline` service only acknowledges queue start.
- Retain stop/continue policies, source observation association, revision checks, bounded storage, and idempotent context/step IDs. Explicit contexts must be cleared by their owner; the box benchmark clears its previous run's contexts before rerunning.
- Cancellation occurs between steps; it cannot stop an in-flight ROS service. A timeout with `STATE_UNKNOWN` must never trigger automatic actuator retries. Emergency stop and drivers retain their direct paths.
- `prepare_cloud_scene` registers immutable known-cloud inputs in the bounded master result store. `RunPipelineStep.bindings` and queued step bindings refer to results in the same context; preserve exact source timestamps/frame IDs and revision/idempotence checks.
- Publish shared function timing on `/pipeline/timings`. Correlate per-call request/backend/response timestamps through `/pipeline/api_traces`; monotonic timestamp differences are valid only on the same host. API, backend and engine scopes overlap: do not add them as disjoint stages. Compare deployed RGB-D/HGGD and synthetic known-cloud modes separately.
- General motion APIs are currently limited to empty-hand execution. Gripper closing requires the ordered validated grasp stages; held-object placement/release execution has not been implemented by the calculation API.

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

Trigger the benchmark with `ros2 service call /box_benchmark/run std_srvs/srv/Trigger '{}'`. It returns after worker exit, prints a final per-box summary, and saves artifacts outside the repository (default `~/box_benchmark`). `/box_benchmark/cancel` requests cancellation; `/box_benchmark/reset` clears placement markers. Keep the completed benchmark node alive for its latched RViz outputs.

The benchmark grounds/segments the tray using the configured real VLM server, creates known 8 cm box clouds, then calls the public cloud-grasp/search/validation APIs sequentially. It tries edge placement before universal heightmap stacking. Failed objects are skipped and remain collision obstacles. Full-motion mode validates lift, transfer and descent; it never executes these paths. This is planning simulation, not physics or hardware verification.

For the deployed image-based path on the mock cup:

```bash
ros2 run task_pipeline pipeline_task --object 'brown cup' --destination tray
```

Single-object placement keeps up to 16 coarse spatial/yaw candidates by default (`placement_candidate_count`). For a cup calculation after synthetic placements, keep the benchmark alive and add `--obstacle-cloud-topic /box_benchmark/scene_cloud` to `pipeline_task`; otherwise benchmark geometry is not part of the unchanged mock camera observation. The optional frozen XYZ overlay reaches both public placement search and pose collision validation, not RGB-D/HGGD input or full held-object motion.

The default recipe waits for its 17-step result (including floor-slot reservation) and returns a nonzero exit code on failure. It calculates only; placement IK failure must not be described as a successful placement.

See [README.md](README.md) for onboarding commands, [docs/CODE_MAP.md](docs/CODE_MAP.md) for the launched file hierarchy and API ownership, and [docs/STANDARD_PIPELINE_API.md](docs/STANDARD_PIPELINE_API.md) for schemas and completion rules.

## Engineering Journal

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

Do not hard-code machine-specific paths, CAN interfaces, credentials, or calibration values in code. The default hand-eye calibration is installed from `master_node/config/calibration/hand_eye_calibration.yaml` and may be overridden with the `hand_eye_calibration_path` launch argument. Changes to calibration, motion, collision, or emergency-stop behavior require explicit review.

Coordinate-transform invariant: perception and grasp computation must use only the synchronized manual `FK(joints) * X_matrix` result produced by `ActiveVisionServer`. Do not add TF lookup/listener fallbacks. Publishing that same matrix to a visualization-only TF frame for RViz is allowed, but computational nodes must never consume it.

## Current Grasp Pipeline Handoff

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
