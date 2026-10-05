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

The official application launch is `master_node/reasoning_stack.launch.py`. It starts composed ActiveVision/VAMP, the standalone multithreaded VLM node, `pipeline_node`, `PiperMasterControl`, and `placement_node`. The master owns the microstep runtime; it is not another process. Camera and AGX hardware drivers are started separately. The placement package and Python module are named `placement_node`, with `/placement/*` backend endpoints and deterministic height-map packing by default.

Safe planning launch, with physical grasp execution disabled:

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio grasp_candidate_backend:=hggd \
  placement_backend:=heightmap allow_physical_grasp_execution:=false
```

`inference_profile:=server` selects the optimized remote server. Profile defaults live in `master_node/config/inference/profiles.yaml`; explicit endpoint/model/timeout launch arguments override them.

Multi-object templates use `[PLAN_UPRIGHT]` or `[PLAN]` with two or more ordered object descriptions (commas, `then` or `and`); the parser routes to layout calculation. Explicit `[PLAN_LAYOUT_UPRIGHT]` / `[PLAN_LAYOUT]` aliases remain. Document only tags implemented by the VLM parser.

## Shared API and Microstep Rules

- `task_pipeline/task_pipeline/registry.py` is the authoritative function-name/type/backend catalog. Production controllers and benchmarks must use it rather than introduce private engine-call sequences or duplicate endpoints.
- Public services use `/pipeline/<function>`; list them with `/pipeline/list_functions` (`my_actions/srv/GetPipelineFunctions`). Declare message/service/action schemas in `my_actions` and rebuild consumers after changing them.
- The master-owned `StepStore` executes typed `PipelineStep` requests, resolves predecessor result bindings, records per-step status/timing and depletes attempted tasks. `@response_cdr` binds a complete serialized response. Never use eval or an arbitrary endpoint supplied by a caller.
- Use `submit_steps` plus `/pipeline/run_steps` (`my_actions/action/RunPipelineSteps`) when callers need final completion and feedback. `run_step` executes one operation. The legacy `execute_pipeline` service only acknowledges queue start.
- Retain stop/continue policies, source observation association, revision checks, bounded storage, and idempotent context/step IDs. Explicit contexts must be cleared by their owner; the box benchmark clears its previous run's contexts before rerunning.
- Cancellation occurs between steps; it cannot stop an in-flight ROS service. A timeout with `STATE_UNKNOWN` must never trigger automatic actuator retries. Emergency stop and drivers retain their direct paths.
- Publish shared function timing on `/pipeline/timings`. API, backend and engine scopes overlap: do not add them as disjoint stages. Compare deployed RGB-D/HGGD and synthetic known-cloud modes separately.
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

The benchmark grounds/segments the tray using the configured real VLM server, creates known 8 cm box clouds, then calls the public cloud-grasp/search/validation APIs sequentially. It tries edge placement before named-support stacking. Failed objects are skipped and remain collision obstacles. Full-motion mode validates lift, transfer and descent; it never executes these paths. This is planning simulation, not physics or hardware verification.

For the deployed image-based path on the mock cup:

```bash
ros2 run task_pipeline pipeline_task --object 'brown cup' --destination tray
```

The recipe waits for its 14-step result and returns a nonzero exit code on failure. It calculates only; placement IK failure must not be described as a successful placement.

See [README.md](README.md) for onboarding commands, [docs/CODE_MAP.md](docs/CODE_MAP.md) for the launched file hierarchy and API ownership, and [docs/STANDARD_PIPELINE_API.md](docs/STANDARD_PIPELINE_API.md) for schemas and completion rules.

## Repository Output Hygiene

Keep the bundled seven CDR messages and `manifest.json` under `mock_hardware_node/snapshots/current`: they are replay inputs installed by the package, not build artifacts. `rgb_preview.png` is a derived local inspection image and is ignored. Keep reusable benchmark/RViz entrypoints and regression tests as source. Write run reports, captured debug images, CDR request/response dumps, rosbag recordings and custom snapshots outside the source tree or in an explicitly ignored output directory. Do not ignore all images/CDR/model files or remove assets required by a fresh clone.

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
