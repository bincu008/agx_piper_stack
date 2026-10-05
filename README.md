# AGX Piper Stack

ROS 2 Jazzy workspace for a Piper arm with a wrist RGB-D camera. It combines
RoboBrain visual grounding, EdgeSAM segmentation, HGGD grasp proposals,
VAMP IK/collision/motion planning, height-map placement and AGX hardware control.
The supported development environment is Ubuntu 24.04 ARM64.

Start here:

- [Launched nodes and file hierarchy](docs/CODE_MAP.md): which files own each node, API and service.
- [Shared microstep API](docs/STANDARD_PIPELINE_API.md): typed calls, result bindings, queue control and timing.
- [Git change audit](docs/GIT_CHANGE_AUDIT.md): why the current changes span many files and what belongs in Git.
- [Dependencies](DEPENDENCIES.md): non-ROS libraries and model assets.

## Build and shell setup

```bash
cd ~/agx_piper_stack
source /opt/ros/jazzy/setup.bash
export ONNXRUNTIME_ROOT="$HOME/.local/onnxruntime-1.15.1"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

Run this setup in **every terminal** used below:

```bash
cd ~/agx_piper_stack
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export LD_LIBRARY_PATH="$HOME/.local/onnxruntime-1.15.1/lib:$LD_LIBRARY_PATH"
```

Adjust the ONNX Runtime path to your installation. Rebuild shared interfaces and
consumers together when changing `my_actions`; source the rebuilt workspace and
restart existing nodes afterward.

## What gets launched

`master_node/reasoning_stack.launch.py` starts ActiveVision and VAMP in a composed
container, plus the standalone VLM, public API dispatcher, master controller and
placement node. The microstep queue runs inside the master process. Camera/arm
drivers, mock replay, RViz and the box benchmark are launched separately.

Controllers choose task order, then call the same `/pipeline/*` APIs. The master
stores typed intermediate results and executes queued steps. Existing numerical
engines remain behind these interfaces. The ROS package, Python module and
executable are all named `placement_node`; its `/placement/*` backend uses
**deterministic height-map packing** by default.

## Mock scene and cup calculation

Keep real camera/arm drivers stopped. Start one instance of each node, using
separate terminals.

**Terminal 1 — recorded scene and arm feedback:**

```bash
ros2 launch mock_hardware_node mock_hardware.launch.py
```

The mock republishes a bundled 640×480 RGB-D scene, camera intrinsics and recorded
joint/TCP/gripper/arm feedback with fresh stamps. It does not move an arm,
update feedback in response to commands, or simulate physics. Its default topic
names match the real drivers; startup checks reject conflicting publishers.
The bundled CDR snapshot is required replay data, not a generated build artifact.

**Terminal 2 — safe reasoning stack:**

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio grasp_candidate_backend:=hggd \
  placement_backend:=heightmap pipeline_profiling:=true \
  allow_physical_grasp_execution:=false
```

Use `inference_profile:=server` for the optimized RoboBrain server. Defaults in
`src/master_node/config/inference/profiles.yaml` are:

| Profile | Base URL | Model |
|---|---|---|
| `lmstudio` | `http://10.211.55.2:1234/v1` | `robobrain2.5-4b` |
| `server` (launch default) | `http://10.10.10.2:18091/v1` | `robobrain2.5-4b` |

Select the server that is actually running; there is no automatic failover.
Override `api_server_ip`, `api_server_port`, `api_server_url`, `model_name` and
`request_timeout_sec` as needed. The inference timeout defaults to 120 seconds.
Mock replay still needs the external VLM server for image grounding.

**Terminal 3 — RViz:**

```bash
ros2 launch vamp_node grasp_debug_rviz.launch.py fixed_frame:=base_link
```

**Terminal 4 — deployed grasp/placement calculations on the recorded cup:**

```bash
ros2 run task_pipeline pipeline_task --object 'brown cup' --destination tray \
  --output "$HOME/cup_pipeline.json"
```

This runs a 14-step API recipe and waits for its final result. Use `--grasp-only`
for the first ten steps. It sends the actual scene image, uses HGGD, and calculates
poses without executing motion. A failed step returns a nonzero exit code.
The tested brown cup has a valid grasp; its upright placement candidates currently
fail IK in the recorded scene. Candidate generation is not placement success.

Legacy prompts are also supported:

```bash
ros2 service call /vlm_reasoning_node/chat_prompt my_actions/srv/Chat \
  "{prompt: '[FIND_GRASP] brown cup'}"
ros2 service call /vlm_reasoning_node/chat_prompt my_actions/srv/Chat \
  "{prompt: '[PLAN_UPRIGHT] place brown cup onto tray'}"
```

`[PLAN_UPRIGHT]` with two or more objects automatically selects ordered layout
planning. `[PLAN_LAYOUT_UPRIGHT]` is an explicit alias:

```bash
ros2 service call /vlm_reasoning_node/chat_prompt my_actions/srv/Chat \
  "{prompt: '[PLAN_UPRIGHT] place brown cup, white plate onto tray'}"
```

These placement prompts calculate results. Physical pick/transfer/release is not
performed. See [multi-object planning](docs/MULTI_OBJECT_PLACEMENT.md) for ordering
and partial-failure behavior.

## Ten-box planning benchmark

Keep the mock and reasoning stack above running. In another terminal:

```bash
ros2 launch placement_node box_benchmark.launch.py \
  box_count:=10 optimize_pipeline:=true plan_pick_place_motion:=true \
  output_directory:="$HOME/box_benchmark/run_01"
```

Start it from a separate terminal:

```bash
ros2 service call /box_benchmark/run std_srvs/srv/Trigger '{}'
```

The service waits until the worker exits and returns the final summary. The node
also prints `[BOX TIMING]` progress and a final `[BOX SUMMARY]` table. A second run
request while busy is rejected. `auto_run:=true` launches and starts automatically.

The benchmark grounds and segments the tray using the VLM, adds ten separated
8 cm boxes on the arm's supporting surface within 50 cm of `base_link`, and
calculates each box sequentially. Box grasp inputs come from known cloud geometry;
box RGB grounding and HGGD are bypassed. It calls the shared grasp validation,
height-map search and placement/motion validation APIs, tries edge placement
first, then named-support stacking. Failed boxes remain collision obstacles and
the worker proceeds to the next object.

`plan_pick_place_motion:=true` validates held-object lift, VAMP transfer and
Cartesian descent. It does not execute these paths or model physical contact.
The last verified API run accepted all ten boxes (six floor, four stacked) in
230.08 seconds. This synthetic timing is not the full deployed HGGD pick-and-place
latency. Refined floor search remains available with
`floor_refinement_candidate_count:=384 floor_refinement_yaw_step_deg:=5.0`.

In RViz, enable **Box Benchmark Tray**, **Box Benchmark Scene** (boxes only),
and **Box Benchmark Placements**. The tray uses `/box_benchmark/tray_cloud`,
not the ordinary single-object `/placement/destination_cloud`. Keep the completed
benchmark node alive to retain its latched outputs for newly opened RViz clients.

```bash
ros2 topic echo /box_benchmark/status
ros2 service call /box_benchmark/cancel std_srvs/srv/Trigger '{}'
ros2 service call /box_benchmark/reset std_srvs/srv/Trigger '{}'
```

Reports, accepted clouds, paths and request/response dumps go into the configured
output directory. Use a new directory to preserve each run. Details are in the
[benchmark guide](docs/BOX_BENCHMARK.md).

## Public API and performance measurements

```bash
ros2 service call /pipeline/list_functions my_actions/srv/GetPipelineFunctions '{}'
ros2 topic echo /pipeline/timings
ros2 service call /pipeline/get_metrics my_actions/srv/GetPipelineMetrics '{}'
```

The catalog currently lists 54 services and the `run_steps` action, with schemas
and backend mappings. `run_step` calls one function; `submit_steps` plus
`/pipeline/run_steps` executes a sequence with final completion and feedback.
`get_step_context` reads retained results; `clear_step_context` releases them.
Step bindings pass exact prior outputs to later requests. Stop/continue policies
handle failures; cancellation stops at step boundaries. The legacy
`execute_pipeline` service only acknowledges starting work.

Timings distinguish API, backend and numerical-engine scopes. They overlap;
do not sum them. Report input mode and full-motion scope when comparing runs.
See [profiling details](docs/INFERENCE_PROFILES_AND_TIMING.md).

## Real hardware and checks

Real hardware uses the same reasoning stack, with camera and arm drivers in
place of the mock:

```bash
ros2 launch orbbec_camera gemini_330_series.launch.py
ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
  can_port:=can0 arm_type:=piper effector_type:=agx_gripper \
  auto_enable:=true control_enabled:=true speed_percent:=20 pub_rate:=50
```

These are real hardware commands. Keep physical execution disabled for planning
and review calibration/motion behavior before enabling it. Computational
transforms use ActiveVision's manual FK × hand-eye calibration, never TF lookup.
General motion APIs currently permit empty-hand execution; gripper closing uses
validated grasp stages. Held-object placement/release execution still needs an
implementation. Physical behavior was not validated by the mock benchmark.

```bash
colcon test --packages-select task_pipeline placement_node master_node \
  vamp_node vlm_node mock_hardware_node
colcon test-result --verbose
```

[Baseline validation](docs/VALIDATION.md) records inherited lint/test debt;
[current API validation](docs/STANDARD_PIPELINE_API.md#timing-and-validation)
records the latest software-only checks. Commit reusable source, tests, launch
files and the replay fixture. Keep build/install/log output and run artifacts
out of Git; see the [change audit](docs/GIT_CHANGE_AUDIT.md) before staging.
