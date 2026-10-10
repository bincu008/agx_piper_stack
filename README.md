# AGX Piper Stack

ROS 2 Jazzy workspace for a Piper arm with a wrist RGB-D camera. It combines
RoboBrain visual grounding, EdgeSAM segmentation, HGGD grasp proposals,
VAMP IK/collision/motion planning, height-map placement and AGX hardware control.
The supported development environment is Ubuntu 24.04 ARM64.

Start here:

- [Available policies and backend choices](#available-policies-and-backend-choices): selectable behavior, engine choices and configuration interfaces.

- [Launched nodes and file hierarchy](docs/CODE_MAP.md): which files own each node, API and service.
- [Shared microstep API](docs/STANDARD_PIPELINE_API.md): typed calls, result bindings, queue control and timing.
- [Git change audit](docs/GIT_CHANGE_AUDIT.md): why the current changes span many files and what belongs in Git.
- [Engineering journal](docs/ENGINEERING_JOURNAL.md): major roadblocks, resolutions and remaining open issues.
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

## Available policies and backend choices

HGGD and EdgeSAM are the only supported deployment grasp/mask engines.
The old GPD vendor package and detector adapter have been removed. Historical
shared geometry parameters now use `grasp_` names (for example
`grasp_tcp_offset_m`); their defaults and safety checks are unchanged.

Policies select behavior; backends select an engine; inference profiles select
server configuration. These options do not all select different algorithms.

| Component | Setting / interface | Options and behavior |
| --- | --- | --- |
| Placement support | `pipeline_task --placement-policy`; public `support_mode` | `auto`: search floor and elevated surfaces together; `edge`: floor-only with corner/edge/adjacency priority; `floor_slots` (default for upright placement): cached adaptive square slots, 6–12.5 cm sides with dynamic gaps and rectangular remainder slots, reserving adjacent cells for larger footprints; `stack`: elevated supports only. Elevated placements require 60% projected footprint coverage by default, plus stability and collision checks. Multiple exposed surfaces may contribute; no object names are needed. |
| Benchmark placement sequence | Benchmark launch `placement_policy` | `edge_then_stack` (default): try edge placement, then stack on failure; `edge` or `stack`: use only that mode. This ordered fallback differs from `auto`. |
| Placement orientation | Public `orientation_policy` | `upright`: gravity-aligned with yaw search; `free`: also search configured roll/pitch angles. The cup CLI currently requests `upright`. |
| Source departure contact | Stack launch `placement_allow_separating_contact_departure`, `placement_departure_contact_tolerance_m` | Enabled by default: small, checked separation before lifting when needed. Contact allowance defaults to 1 mm and is bounded to 0–2 mm; zero restores strict initial-clearance checks, subject to the separate sparse-contact budget below. Disable departure with `false`. |
| Sparse attached-object contacts | Stack launch `placement_noise_max_points`, `placement_noise_max_voxels` | Lift/descent tolerate at most 32 unique obstacle points across 10 cells of 5 mm, accumulated over the whole segment. Separate contacts have no spread/distance threshold. Transfer and arm/gripper checks stay strict. Set max points to 0 to disable the sparse-contact allowance. This tolerates small noise clusters but does not prove they are noise. |
| Placement search effort | Benchmark `optimize_pipeline`, `placement_early_stop_count` | Fast search prioritizes promising cells and stops after enough candidates. Exhaustive search explores the configured grid/orientations. `floor_refinement_candidate_count` (default 0, disabled) and `floor_refinement_yaw_step_deg` enable a finer floor fallback. These are search controls, not extra support modes. |
| Placement backend | Stack launch `placement_backend` | `heightmap` (default): deterministic packing; `anyplace`: retained optional diffusion backend requiring its model/checkpoint. Heightmap support policies and synthetic overlays require `heightmap`. |
| Grasp proposals | Stack launch `grasp_candidate_backend` | `hggd` only: deployed RGB-D learned proposals. Synthetic known-cloud fixtures retain an internal geometric contact path because they have no RGB image; this is not a selectable deployment backend. |
| Target mask | Stack launch `grasp_target_mask_backend` | `edgesam` only: RoboBrain grounding followed by EdgeSAM segmentation. Box-only and YOLO-intersection target-mask paths have been removed. EdgeSAM models are required. |
| World inventory | Stack launch `world_perception_backend` | `robobrain` (default): VLM inventory; `yolo`: detector-based asynchronous inventory. |
| IK seed search | Stack launch `deterministic_ik` | `true` (default): deterministic seed bank, sized by `deterministic_ik_seed_count`; `false`: legacy limited seed set. Both use the same numerical IK solver; neither selects a different solver. |
| Pipeline failure handling | `PipelineStep.failure_policy` | `stop`: stop the queue on failure; `continue`: attempt later independent steps. Failed results cannot satisfy dependent inputs. |
| Inference connection | Stack launch `inference_profile` | `lmstudio` or `server`: configurable endpoint/model/timeout defaults from `master_node/config/inference/profiles.yaml`; these are connection profiles. |

The stack's VAMP motion planner uses **RRT-Connect**; no alternative planner is
exposed by the application launch. Side, rim, thin-object and top-surface grasp
modes are selected internally from geometry rather than a public policy flag.

Configure footprint coverage through stack launch `stack_minimum_coverage:=0.6`
(placement parameter `packing_stack_minimum_coverage`). The box benchmark uses
0.6 through the same public heightmap search API. The reserved deprecated
named-support fields are ignored; `--support-object` is no longer supported.

Example universal stacking after a completed box benchmark:

```bash
ros2 run task_pipeline pipeline_task --object 'brown cup' --destination tray \
  --placement-policy stack --obstacle-cloud-topic /box_benchmark/scene_cloud
```

Keep the benchmark alive for its latched cloud. For real RGB-D observations,
omit the obstacle-cloud option. Calculation success is not physical execution:
the cup recipe validates placement/pre-place poses; full-motion benchmark mode
also validates lift, transfer and descent, without executing them.

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

This runs a 17-step API recipe (including slot reservation) and waits for its final result. Use `--grasp-only`
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
first, then universal heightmap stacking. Failed boxes remain collision obstacles and
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

### API performance reports

The API dispatcher now runs in the master process by default; use
`api_dispatch_in_master:=false` to restore the separate proxy. Public
`/pipeline/*` names stay unchanged. Each completed box benchmark writes
`api_timing` (request preparation/delivery, backend, response preparation/delivery
and client decoding) to `~/box_benchmark/report.json` and prints
`[API END-TO-END]` rows. See
[API transport performance](docs/STANDARD_PIPELINE_API.md#api-transport-performance)
for timing boundaries and reusable cloud-scene references. Physical execution
must remain disabled for these benchmarks.

Single-object placement keeps 16 ranked spatial/yaw candidates by default. The previous
eight-pose shortlist missed reachable cup poses in the recorded scene. Override with
`placement_candidate_count:=16` on `reasoning_stack.launch.py`; the old
`anyplace_candidate_count` argument remains a compatibility alias. Validation still
uses bounded batches and retains the existing IK, collision and pre-place checks.
The cup calculation recipe validates poses; it does not certify a held-object
transfer trajectory or execute placement.

To include completed benchmark boxes in the cup calculation, keep the benchmark
node running and use:

```bash
ros2 run task_pipeline pipeline_task --object "brown cup" --destination tray \
  --obstacle-cloud-topic /box_benchmark/scene_cloud
```

The client
freezes the latched boxes-only cloud and supplies it to both placement search and
IK/collision validation. Without this option, the mock camera scene is unchanged
by benchmark calculations. This remains pose-only planning, without hardware
execution or held-object transfer certification. Rebuild interface consumers after
changes to `FindPlacementCandidates` and `PlanPlacement`.

Exhaustive heightmap searches use coarse XY/yaw coverage for the 16-candidate
shortlist: keep the best packing pose, then choose alternatives far from those
already selected, with a 4 cm minimum center spacing. The recorded simple cup
scene passes with this shortlist. Fast early-stop searches and larger refined-search budgets retain
their existing ranked selector. If the support is too small, fewer candidates
are returned rather than filling the list with near-duplicates. `pipeline_task` reports master `seconds` and `client_wall_seconds`
(submission through final result delivery; excludes ROS startup/input discovery).

Universal heightmap stacking on an existing benchmark scene:

```bash
ros2 run task_pipeline pipeline_task --object 'brown cup' --destination tray \
  --placement-policy stack --obstacle-cloud-topic /box_benchmark/scene_cloud \
  --output "$HOME/cup_stack.json"
```

Keep the benchmark node alive for its latched geometry. No box name or saved
benchmark directory is required. Real observations need no overlay option.
Stacking combines exposed support across surfaces and requires at least 60%
projected footprint coverage at the placement height, plus contact area, COM
and collision checks. `auto` also enforces this threshold for elevated poses;
`edge` requests floor-only placement. Cup calculations validate poses, not full
held-object transfer or physical execution.

On the 8 GiB development VM, build large C++ packages sequentially to avoid
compiler memory pressure taking down VS Code:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1 \
  colcon build --symlink-install --executor sequential
```

Avoid concurrent full builds while the stack and editor are running.

The coverage threshold is configurable with `stack_minimum_coverage:=0.6`
(placement ROS parameter `packing_stack_minimum_coverage`, default 0.6).
The benchmark uses the same heightmap policy and threshold; configure its launch
with `stack_minimum_coverage:=0.6` as well when overriding the default. It searches all
raised supports together after edge placement fails, preserving yaw alternatives.

`placement_bypass_attached_collision_during_lift` now defaults to true, as
requested for departure from closely packed scenes. Set it to false to enable
attached-object/environment validation during lift. Development runs must use
`allow_physical_grasp_execution:=false`. The bypass skips only lift checks. Robot, gripper,
object-arm, transfer and descent checks remain active. Motion responses carry
`LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED`; such results do not establish a
collision-free lift. Alternative bounded departure paths remain available.
If ordinary lift routes fail, VAMP can search a bounded retreat to the validated
pre-grasp configuration using the same RRT-Connect engine. The fallback retains
joint limits and stage collision checks, revalidates interpolated samples, and
limits TCP steps to 2 mm; it does not change the final grasp approach.

Cup-to-tray recipes issue four separate RoboBrain requests in order: cup presence,
cup grounding, tray presence, tray grounding. Each presence failure stops the
recipe. `[ROBOBRAIN TX]` logs the exact prompt for each HTTP request independently;
`[VLM TIMING]` logs that request's duration. `pipeline_task` reports separate
step IDs, targets, responses and durations for all four inference APIs.

### Grid floor placement

`floor_slots` is an upright floor-only policy. Rows follow the longest destination
axis. Main slots are squares, up to 12.5 cm per side. Two rows are preferred,
scaling squares down to 6 cm; below that, use one row and recalculate square size.
Three or more rows are allowed only when all those rows retain 12.5 cm squares.
Unused space between rows/columns becomes dynamically calculated gaps; a
remaining strip at least 6 cm wide can form rectangular remainder slots.
The largest usable inscribed rectangle and non-overlapping extensions define the
working regions. Footprint stencils select the
smallest compatible group; gaps between squares remain usable tray support.
Actual touched-cell patterns can be rectangular or irregular. Existing heightmap
support, clearance and pose validation still apply. Scene obstacles block cells.
The cache is bounded and invalidates when the measured destination boundary changes.

Use the recorded scene with the normal mock and reasoning stack running:

```bash
ros2 run task_pipeline pipeline_task \
  --object 'beige cup at far left' --destination 'dark rectangular tray' \
  --placement-policy floor_slots --output "$HOME/cup_grid.json"
ros2 launch vamp_node grasp_debug_rviz.launch.py
```

RViz `/placement/floor_slots` shows free slots in blue, previous reservations in
orange, and the selected group in green. Selected cup geometry and gripper poses
remain on the existing VAMP placement displays. Reload the RViz preset to enable
the new grid display. The recipe adds `commit_floor_slots` after successful
`plan_placement`; unsuccessful validation never consumes cells. Reservations
persist for later tasks until reset, geometry invalidation or node restart.
A calculation commit reserves space but does not execute hardware or add an
object to the camera point cloud. Explicitly clear reservations before repeating
an independent trial:

```bash
ros2 service call /pipeline/commit_floor_slots my_actions/srv/CommitFloorSlots \
  '{reset: true}'
```

The existing single-object recipe measures placement IK and descent, not a
complete held-object transfer. Compare `find_placements`, `plan_placement` and
reservation cost together, and keep each RoboBrain request separate. Full-motion
planning callers still use the public `plan_placement` frozen-scene inputs.

Measured cup results and the full per-API runtime table are in
[Grid floor policy profiling](docs/FLOOR_SLOTS_PROFILE.md).

### Compact multi-object grid placement

Whole-cell reservation remains the default. For better utilization, the ordered
layout controller can reserve 5 mm footprint masks inside the same visible grid:

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio placement_backend:=heightmap \
  layout_support_mode:=floor_slots floor_slot_share_unused_space:=true \
  layout_candidate_count:=16 allow_physical_grasp_execution:=false
```

Only non-overlapping reserved footprint portions may share a grid square.
Search and commit use the same bounded mask state; full-scene collision and motion
validation remain active. This mode samples 1 cm position shifts and 5° yaws,
prefers fewer newly consumed cells and preserves a contiguous opening for later
objects. Spatial/yaw diversity applies within each cell-count tier, preventing
larger groups from displacing smaller alternatives. It costs more search time
than centered whole-cell matching. Cached footprint stencils avoid recomputing
object heightmaps per position.

Submit ordered `[PLAN_UPRIGHT]` descriptions using the existing VLM Chat API.
The layout controller calls the shared search, motion-validation and slot-commit
APIs, removes successfully moved source geometry, and adds its actual accepted
pose for subsequent objects. Failed objects remain obstacles. The ordinary
single-object recipe retains whole-cell behavior; direct shared-search callers
must supply a complete updated frozen scene and perform the normal validation.

RViz grid outlines show free/occupied/current cells. Yellow 5 mm points show the
reserved portions, including clearance, so sharing a cell does not imply an
object collision. The selected layout topic contains accepted objects only.
Results and launch details: [Grid space optimization](docs/GRID_SPACE_OPTIMIZATION.md).

Floor-slot matching prioritizes one-cell candidates before larger groups. Fit
uses the complete projected object (including rim/handle) and clearance, not
footprint area alone. A small-area but wide shape may require multiple cells.

The upright placement default is exclusive `floor_slots` across the CLI recipe,
ordered layouts and single-object VLM plans. Each object uses the smallest
validated group; a used cell is unavailable to other objects. Slot fit uses the
full projected geometry, including handles/rims, not area alone. The current
slots use the adaptive square/remainder dimensions described above. Whole-cell remaining-space scores
are cached across equivalent rotations, and only the accepted pose's occupancy
mask is built at commit. Free-orientation VLM/layout requests retain `auto`;
explicit public `floor_slots` requests require upright orientation. `edge`,
`stack` and `auto` remain explicit alternatives. The box benchmark retains its
explicit `edge_then_stack` test sequence.

Verified mock cup → lower bowl → upper bowl uses one slot per object and
adjacent slots for the bowls. See [grid placement report](docs/GRID_SPACE_OPTIMIZATION.md).

Adaptive grid launch controls are `floor_slot_adaptive:=true`,
`floor_slot_max_size_m:=0.125` and `floor_slot_min_size_m:=0.06`.
Gaps are computed from the working region and square size. Setting `floor_slot_adaptive:=false` restores the
legacy fixed 12.5 cm squares with 0.5 cm gaps. Occupied-surface updates block
existing cells without moving their boundaries or IDs. See
[adaptive grid validation](docs/ADAPTIVE_FLOOR_GRID.md) for measured results.
