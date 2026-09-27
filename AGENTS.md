# Repository Guidelines

## Project Structure & Module Organization

This is a ROS 2/colcon workspace. Production packages live under `src/`:

- `master_node`: Python orchestration and hardware execution.
- `vision_node`, `vamp_node`, and `vlm_node`: C++ active perception, VAMP motion planning, and VLM integration.
- `vamp_mvt`: separate VAMP C++ planning/collision library with the generated Piper model.
- `anyplace_ros`: deterministic height-map placement and the optional AnyPlace backend; it is launched by the reasoning stack.
- `my_actions`: shared ROS services/actions; update these before changing cross-node APIs.
- `agx_arm_ros/src/agx_arm_ctrl` and `agx_arm_msgs`: Piper CAN control and arm messages.
- `OrbbecSDK_ROS2/orbbec_camera` and `orbbec_camera_msgs`: RGB-D driver and interfaces.

Keep runtime assets beside their owning package: `vision_node/models`, `vlm_node/models`, and `vamp_node/models`. ONNX Runtime is an independently installed C/C++ dependency selected through `ONNXRUNTIME_ROOT`. Generated `build/`, `install/`, and `log/` directories are not source.

## Build, Test, and Development Commands

From the workspace root:

```bash
export ONNXRUNTIME_ROOT="$HOME/.local/onnxruntime-1.15.1"
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select anyplace_ros master_node vamp_node agx_arm_ctrl
colcon test-result --verbose
```

The official application launch is `master_node/reasoning_stack.launch.py`. It starts the composed perception/VLM/VAMP container, `PiperMasterControl`, and the placement node. Camera and AGX hardware drivers are started separately.

Safe planning launch, with physical grasp execution disabled:

```bash
ros2 launch master_node reasoning_stack.launch.py \
  api_server_ip:=10.211.55.2 api_server_port:=1234 \
  model_name:=robobrain2.5-4b grasp_candidate_backend:=hggd \
  placement_backend:=heightmap allow_physical_grasp_execution:=false
```

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
