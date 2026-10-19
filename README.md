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
Grasp angle tuning is best effort: bend the original refined pose outward about
its local tool X axis, keeping the contact region and hinge axis fixed. Try a
finger-to-local-surface angle of 10°, then 30°, then 50°; each larger angle bends
less aggressively. One measured midpoint patch defines the surface. The same
contact pair determines the jaw opening projected onto the new closing axis.
Actual IK, contact, collision and pre-grasp checks still apply; failure retains
the original validated pose. `PAD_SURFACE_ALIGNMENT_VALIDATED` identifies a
passing angle; `ANGLE_TUNING_FAILED_ORIGINAL_USED` identifies the fallback.
No additional launch argument is needed.

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
| Held-motion orientation | Launch `placement_motion_orientation_policy` | `keep_upright` (default): constrain the attached object's vertical axis throughout lift, transfer and descent; yaw is free. `free`: retain collision checks without this orientation constraint. |
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
  output_directory:="$HOME/agx_piper_data/reports/box_benchmark/run_01"
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
General motion APIs permit empty-hand execution; payload motion uses the guarded held-motion APIs below. Gripper closing uses
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
and client decoding) to `~/agx_piper_data/reports/box_benchmark/report.json` and prints
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

Floor-slot shortlists also keep up to one quarter of their candidate budget for
fine yaw alternatives within 20 degrees of the best-ranked pose. These use the
existing containment-tested 5-degree samples, alongside coarse position/yaw
coverage. This preserves narrow reachable wrist orientations without increasing
the 16-candidate budget or relaxing placement validation.

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

Slot selection tries a single fitting slot first, including the large square.
When a combination is needed, prefer the fewest occupied slots. Equal-count
choices use the least reserved area, then spatial/yaw preference. This ranking
is automatic; support, exclusivity, IK and motion checks remain active.

The adaptive grid also reserves at most one larger square after preserving two
ordinary squares, when the spare region can contain it.
`floor_slot_large_square_scale:=1.5` multiplies the ordinary **side length**;
`0` disables it. Oversized squares fall back to the ordinary grid. Remaining
space is divided into ordinary squares and remainder rectangles. RViz labels
the larger cell with its measured side length. Changing the factor creates
a separate grid configuration; reset reservations for independent trials.

Adaptive grid launch controls are `floor_slot_adaptive:=true`,
`floor_slot_max_size_m:=0.125` and `floor_slot_min_size_m:=0.06`.
Gaps are computed from the working region and square size. Setting `floor_slot_adaptive:=false` restores the
legacy fixed 12.5 cm squares with 0.5 cm gaps. Occupied-surface updates block
existing cells without moving their boundaries or IDs. See
[adaptive grid validation](docs/ADAPTIVE_FLOOR_GRID.md) for measured results.

### Shared metric correction

`master_node/config/calibration/perception_scale.json` stores the global
`perception_measurement_ratio` (currently 0.976819, measured distance / real distance).
The reasoning stack passes it to ActiveVision, which divides raw optical XYZ
once before manual `FK * X`. Observation clouds, maps, YOLO world inventory,
HGGD-derived depth, grasp refinement, placement and collision checking therefore
use corrected geometry. Robot link dimensions, arm feedback, joint commands,
fixed lift distances, RGB pixels and camera intrinsics remain unchanged.
Known-cloud benchmark objects already expressed in metres are not rescaled.

Override at launch:

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio perception_scale_path:="$HOME/agx_piper_data/calibration/plate_scale_calibration.json" \
  allow_physical_grasp_execution:=false
```

Use ratio 1.0 to disable correction. The parameter is read-only while running;
restart the stack to change it so cached observations cannot mix scales.
Raw camera topics and recorded snapshots remain raw. Corrected registered depth
is available on `/active_vision/corrected_registered_depth` as `32FC1` metres,
with the original camera header, and is reconstructed from the same corrected
XYZ. It is published when subscribed. The current factor is fitted to a known
17.4 cm plate across five captures. Absolute depth bias and independent camera
XYZ scale are unverified; this does not calibrate hand-eye or physical TCP.

Plate-first replay validation and tight-slot behavior are documented in
[the adaptive grid guide](docs/ADAPTIVE_FLOOR_GRID.md#tight-fits-and-elevated-plates).
Continuous centers preserve single-slot fits; tilted floor contact labels avoid
misclassifying the tray itself as an obstacle. Measured cloud extents remain
conservative and must not be reported as calibrated physical diameters.


### Guarded physical bowl grasp-and-lift trial

The shared APIs now separate payload planning from execution:

| API | Purpose |
|---|---|
| `prepare_held_scene` | Freeze source payload and environment from the synchronized cloud/mask; retain neighboring objects. |
| `prepare_held_motion` | Plan from the cached refined attachment; return a plan revision and complete serialized `PlanPlacement.Response`. With `lift_only=true`, no tray or placement candidates are required. |
| `execute_lift` | Follow the cached lift at 5% speed; leave the object held. |
| `execute_placement_transfer` | Follow the cached transfer after lift. |
| `execute_placement_descent` | Follow the cached descent after transfer. |
| `release_placed_object` | Open only after descent, at the validated pose; clear held state after new opening movement and loss-of-force feedback. |

Each execution request contains `object_name` and `plan_revision`. These APIs
reject stale feedback, changed attachment geometry, wrong order/revision and
unconfirmed gripper contact. Failed actuator execution marks `STATE_UNKNOWN` and
cannot be automatically retried. Speed is restored to 20% on every motion exit.
A lift-only plan cannot execute transfer, descent or release. General empty-hand
motion APIs remain unavailable while an object is held.

**Current validation blocker (2026-10-12):** both recorded bowls produce grasps,
but the 5 cm lift planner rejects gripper/environment contact at the starting
pose. Physical motion is untested; resolve this modeled contact before the trial.
See the engineering journal for exact results.

For the first **operator-run physical trial once that check passes**, stop mock replay and benchmark
nodes; use the real camera and arm driver commands above. In another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio grasp_candidate_backend:=hggd \
  placement_backend:=heightmap allow_physical_grasp_execution:=true \
  placement_bypass_attached_collision_during_lift:=true
```

First calculate and preview the complete grasp-and-lift plan without motion:

The operator-selected `joint_limit_tolerance_degrees` defaults to `2.0`: each
of the six nominal joint ranges expands by 2 degrees at both ends. Master
feedback/path checks, numerical IK and VAMP sampling use the same ranges.
Set the launch argument to `0.0` to restore nominal limits. This changes
planning limits; it does not calibrate encoders or modify driver/controller limits.

`final_approach_check_finger_environment:=false` is the operator-selected
default. It skips finger-versus-scene checking during final approach only;
arm, palm and self-collision checks remain. The plan reports
`FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED`. Set it to `true` to restore
full finger/environment validation. A successful calculation with this bypass
does not establish finger clearance along the approach.

```bash
ros2 run task_pipeline pipeline_task \
  --object 'small white bowl at bottom left' --prepare-grasp-lift --lift-distance 0.05
```

For a calculation-only second prompt on replay, check the cached handoff:

```bash
ros2 run task_pipeline pipeline_task \
  --object 'small white bowl at bottom left' --check-cached-grasp-lift
```

It checks cache freshness, starting joint/TCP feedback and command limits without
moving or consuming the plan. It does not test physical tracking or contact.

Execute only after the first command reports `ALL_MOTION_PLANNED`. Inspect
`/vamp_planner/valid_grasp_candidates_markers` in RViz. Then, within
60 seconds, execute the reviewed cached grasp (an expired cache stops the recipe):

```bash
ros2 run task_pipeline pipeline_task \
  --object 'small white bowl at bottom left' \
  --execute-cached-grasp-lift --lift-distance 0.05 \
  --output "$HOME/agx_piper_data/reports/bowl_grasp_lift.json"
```

The first command validates and caches pre-grasp transfer, optional near-grasp
approach, and the 5 cm lift from the original synchronized observation.
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
It reports success only when every path is ready. The second command executes
those frozen paths, opens at pre-grasp, closes after approach, confirms fresh
contact, lifts, then **stops holding**. It runs no IK, replanning or close-up
refinement. New close-up refinement requires preliminary movement and remains
available only through the separate legacy staged workflow. Inspect the bowl before
requesting any placement. The existing physical gate remains false by default.
The lift bypass affects only attached-object/environment checks; robot, fingers,
object-arm, transfer and descent checks remain active. Results expose
`LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED`; lift is not environment-collision validated.
Automated tests do not move hardware. Physical calibration/contact and the
operator-run trial must be verified separately.


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

Single-object HGGD search stops at the first collision-safe endpoint. It uses
up to nine measured variants per seed and can continue an empty search under the
configured 600-attempt hard cap. Layout and known-cloud searches retain their
backup pools. Deployment IK uses the existing solver in first-valid mode with
coarse and precise collision predicates; rejected seeds continue through the
deterministic bank. Returned HGGD grasp poses are the FK of their validated
joints, so preview, strict final-approach planning and cached execution use
the same endpoint. Complete approach/lift validation remains required. Empty
AUTO bowl results receive one measured RIM retry on the same immutable observation.
The local lift retreat resamples its RRT path with bounded joint and TCP steps
instead of imposing excessive angular subdivisions.

`pipeline_task --prepare-grasp-lift` defaults to one calculation attempt for
bounded latency. Opt into fresh-observation recovery with
`--preparation-attempts 2` or `3` after definite geometric rejection. Failed contexts are released
before recapture; JSON `attempts` retains each attempt's status and step timings.
`seconds` sums backend attempts and `client_wall_seconds` includes recovery and
response delivery. Physical execution, unknown states, and invalid inputs are
never automatically retried. Review the final prepared preview before execution.

The next agent/MCP architecture is documented in
[Agent-controlled arm API design](docs/AGENT_CONTROL_ARCHITECTURE.md): shared
execution state, public prepared segments and gripper operations, with composite
recipes using the same calls. This is an accepted design plan, not implemented APIs.

Lower-bowl replay validation (2026-10-12): ten calculation-only grasp/lift
preparations passed in 2.99–4.34 s (mean 3.36 s), with one observation attempt
each. See `docs/ENGINEERING_JOURNAL.md` for the endpoint handoff fix and limits
of this recorded-scene result.

Paired lower-bowl replay validation (2026-10-12): 20/20 preparations followed
by `--check-cached-grasp-lift` passed. Mean API duration 3.36 s; both command
processes including ROS startup averaged 8.09 s. No physical motion was executed.

To inspect a fully constructed grasp/lift pipeline without moving hardware, run
`ros2 run task_pipeline pipeline_task --object 'small white bowl at bottom left'
--queue-grasp-lift --lift-distance 0.05 --output "$HOME/bowl_queued_pipeline.json"`
with the mock and reasoning nodes running and physical execution disabled.
Planning and readiness checks execute; the master retains pregrasp, gripper
opening, approach, gripper closing and lift as pending public API tasks. See
[prepared-stage API documentation](docs/STANDARD_PIPELINE_API.md#public-prepared-grasplift-stages-2026-10-13).

### Plate scale calibration

Keep physical execution disabled and the plate fully visible and stationary:

```bash
ros2 run master_node calibrate_plate --diameter 0.174 --samples 5 \
  --output "$HOME/agx_piper_data/calibration/plate_scale_calibration.json"
```

The tool captures source-associated RGB-D, grounds and segments the plate in
each frame, fits its outer rim plane robustly, then measures 12 directions. It
rejects clipped silhouettes, unreliable planes, directional spread over 5% or
inter-frame factor variation over 2%. Captures/CDR/masks remain beside the output
outside the source tree. It saves a uniform optical measurement-to-real ratio;
one circular plane does not independently identify X/Y/Z or absolute depth.
Restart the stack with `perception_scale_path` pointing to that JSON to apply it.
The installed JSON is the portable default; `perception_measurement_ratio` is
an explicit override. Raw drivers, robot geometry and joint commands are unchanged.

### Recovering a failed prepared grasp/lift

Restart the reasoning stack once after updating the code. Thereafter a failed
known empty-hand prepared run automatically clears its session, grasp cache and
remaining tasks, returning `SESSION_RESET_READY_FOR_NEW_PLAN`. Run the preparation
command again; no restart or automatic arm motion is required. The preparation
freshness limit remains 60 seconds; accepted execution has a separate 300-second
stage-boundary timeout (`prepared_execution_timeout_sec`).

If a close/motion outcome is unknown, queued tasks are discarded but the unknown
or held state is preserved. After physically checking that the hand is empty
and the arm is stationary, explicitly reconcile it:

```bash
ros2 service call /pipeline/reconcile_empty_hand std_srvs/srv/SetBool '{data: true}'
```

For a known empty-hand session, `/pipeline/reset_execution_session`
(`std_srvs/srv/Trigger`) can reset it manually. Neither API commands actuators.
See [failure recovery semantics](docs/STANDARD_PIPELINE_API.md#prepared-session-failure-recovery-2026-10-13).

### Perception XYZ translation

`grasp_offset_path` retains its existing JSON schema for compatibility:
`frame_id: "base_link"` and `grasp_position_offset_base_m` (metres).
The reasoning stack now routes this vector to ActiveVision's read-only
`perception_position_offset_base_m`, and sets VAMP's grasp-target offset to zero.
The correction is composed into synchronized manual `FK(joints) * X` translation;
raw camera XYZ, RGB, optical depth, scale ratio and stored hand-eye calibration
remain unchanged. The obstacle map transforms only surviving downsampled points.
The organized observation retains pixel correspondence for segmentation/HGGD and
uses the same corrected transform in its existing conversion, without an extra
per-point pass. Thus contacts, obstacles, placement inputs and RViz agree, and IK
never adds the correction twice. Synthetic clouds already supplied in base_link
are unchanged. Restart the stack and prepare fresh plans after editing the JSON;
do not reuse old observations/reservations with a different correction.

### Joint-6 command-only correction

`master_node/config/calibration/joint_command_offset.json` is loaded by default
when the master starts, with `joint6_command_offset_rad` currently
`0.23499113048851653`. Edit that file and restart the reasoning stack; there is
no additional launch argument. All values are raw radians.

Raw joint/TCP feedback, IK, FK, RViz planning poses, camera X-matrix and collision
planning remain unchanged. After the existing nominal target limit checks, the
master publishes `joint6_sent = joint6_nominal + offset`, once per arm command.
Other joints and gripper-only commands are unchanged. Arrival/handoff checks
compare raw feedback with the sent command; endpoint verification accounts for
the commanded local joint-6 roll. The offset is not accumulated in cached paths.
Emergency raw-position holding commands bypass it. Driver-native `/move_home`
and commands sent directly to the driver bypass the master's correction.

This is the operator-requested command compensation, not a joint-zero model
calibration. Planner collision/limit checks cover the nominal path. The added
rotation is outside those checks, and the driver can reject targets pushed past
its physical range. No post-addition clamping is performed by this compensation.
Use a fresh preparation before operator-run physical validation.

### Local output locations

Keep project outputs under `~/agx_piper_data/`:

- `calibration/`: user-selected scale and grasp-offset JSON files.
- `logs/`: run/build/test logs; ROS also keeps its standard `~/.ros/log` folder.
- `reports/`: pipeline JSON summaries and benchmark outputs.
- `captures/`: RGB-D/CDR/images and replay snapshots.
- `tools/`: archived local diagnostic helpers, outside the source repository.

The 2026-10-13 cleanup mapping is `~/agx_piper_data/relocation_manifest.json`.
Historical journal paths can be resolved through that manifest. Installed default
calibrations remain under `master_node/config/calibration`; moving user overrides
does not change their values. On the next launch, explicit user paths are:

```bash
perception_scale_path:="$HOME/agx_piper_data/calibration/plate_scale_calibration.json" \
grasp_offset_path:="$HOME/agx_piper_data/calibration/perception_offset_X_Y_Z_m.json"
```

The box benchmark defaults to `~/agx_piper_data/reports/box_benchmark`. For manual
pipeline trials use `--output "$HOME/agx_piper_data/reports/bowl_prepare.json"`.
Create the folders on a fresh workstation with
`mkdir -p "$HOME/agx_piper_data"/{calibration,logs,reports,captures,tools}`.

Command-only joint-6 calibration: execution checks raw feedback against the
envelope of nominal limits and their offset-translated limits. This does not
expand nominal IK/command bounds, wrap angles, change raw feedback, or bypass
driver limit enforcement.

Prepared final approach streams prevalidated joint edges at 60 Hz with a
0.075 rad/s joint timing bound (50% faster than the previous default), rather than waiting at each microstep.
Start-state, live feedback/fault and final joint/TCP arrival checks remain active;
closing is permitted only after final verification. Driver-side smoothing is
hardware-dependent and has not been verified by offline tests.

Gripper closing confirms contact only after three fresh samples spanning 150 ms
with a settled aperture (within 0.5 mm) and force magnitude at least 0.2 N. A
transient closing motor load cannot alone authorize lifting.

Prepared lift uses the same timed streaming as final approach, retaining live
held-contact monitoring and final joint/TCP verification. Pre-grasp retains
its existing execution behavior.

Initial grasp pose collision checks use the coarse obstacle world; the detailed
candidate-local cloud pose check is disabled. Measured object contact and finger
closing-sweep checks remain active, along with IK and subsequent motion checks.

Initial final-grasp candidate environment validation checks only arm/wrist
spheres (0..73) against the coarse world; palm/fingers (74..95) are excluded
by default. Full self-collision and existing pre-grasp/approach/lift/placement
policies are unchanged. No extra launch argument is needed.

Prepared approach preserves the validated final-grasp joints. Optional near-grasp
solutions are intermediate targets only; a failed intermediate cannot force a
new final-grasp IK solve. Reachable prefixes and their connections to the fixed
endpoint are checked before execution; direct fallback is also fully validated.
Restart all reasoning components after the approach service-schema update and
prepare a fresh plan.

Pre-grasp defaults allow 16 degrees of orientation difference from the final
grasp (previously 20), with a preferred 15.6 cm standoff and a maximum 19.5 cm
standoff. Shorter, target-visible alternatives remain available. A good endpoint
alone is insufficient: VAMP must also validate a path from the measured arm state.
Small feedback deviations within `joint_limit_tolerance_degrees` are handled by
a collision-checked transition into nominal planning bounds. Feedback and FK stay
unchanged; IK/sampling and commanded waypoint limits are not enlarged. The initial
measured path sample is never sent as a new actuator command.

### Live collision-sphere view

With the arm driver or mock feedback running, launch:

```bash
ros2 launch vamp_node grasp_debug_rviz.launch.py
```

This starts a read-only viewer and RViz with base_link as fixed frame. It
subscribes to /feedback/joint_states (joint1..joint6) and gripper_status;
/vamp_debug/current_collision_spheres publishes all 96 VAMP sphere markers
and index labels. Arm/wrist is blue, palm orange, fingers green, and sphere 80
magenta. Fresh gripper width adjusts finger centers; otherwise generated 70 mm
opening is shown. Markers expire after 0.5 s without joint updates. The viewer
displays complete raw-feedback geometry, regardless of checker-specific sphere
exclusions. It does not issue commands or use computational TF.

Sphere 80 is again active during approach validation. Initial grasp calculation
can recover a sphere-80 collision by retreating the whole gripper 2..10 mm,
keeping measured contacts inside the actual finger-pad reach and rechecking IK
and arm clearance. This supersedes the earlier sphere-80 exclusion. Unresolved
collisions remain failures; no new launch option is required.

Grasp-pose environment checking now uses spheres 0..67 inclusive, both initially and at the fixed final endpoint recheck. Palm 80/81 remain separate adjustment triggers. Other motion checks and full self-collision retain existing policies.

Grasp-tagged VAMP planning and the complete final approach use environment spheres 0..67. Post-lift placement transfer retains full geometry and self-collision remains full in all stages. No additional launch argument is needed.

The published coarse obstacle world removes voxels with fewer than two occupied
neighbors (26-cell adjacency, one simultaneous pass). Object clouds are unchanged.
This heuristic can remove sparse real geometry as well as depth noise.

Mock diagnostic layout inspection can opt into placement_node parameters
`layout_debug_stop_on_failure` and `layout_debug_show_candidates` (both default
false). The trial-only `/placement/trial_candidates` MarkerArray shows proposed
footprints and gripper XYZ targets from the first ranked grasp, not accepted
IK solutions. Debug RViz includes this display. Normal layouts continue past
failed objects.

Upright source footprints use gravity projection in base_link. The surrounding
support-plane fit supplies height only; it must not shear measured XY dimensions.
Measured source points are preserved. This applies to the actual packing geometry,
not only debug rendering.

### Gripper opening and closing targets

All grasp execution paths share `master_node.gripper_widths.grasp_gripper_widths`:
open to measured grasp width + 2 cm (capped at 10 cm), then command closure to
that capped opening minus 4 cm (floored at zero). A 5 cm grasp opens to 7 cm and
commands closure to 3 cm at 1 N; contact feedback must still confirm the grasp.
The approach/pre-grasp geometry uses the same 2 cm opening clearance by default.
Restart reasoning and prepare fresh after changing this behavior.

### Keep-object-upright motion

Held-object planning defaults to `placement_motion_orientation_policy:=keep_upright`.
The object's source vertical axis is gravity-aligned, and its rigid attachment to
the gripper maps that axis at every FK sample. Tool roll/pitch may therefore differ
from the object's; yaw is unconstrained. The default angular tolerance is
`placement_motion_upright_tolerance_rad:=0.05235988` (3 degrees).

This constraint applies before lift collision bypasses, during VAMP RRT-Connect
transfer sampling and to descent. Returned segments are also densely checked
between waypoints (at most 0.005 rad joint increments and 2 mm TCP steps). All
lift fallbacks obey it; a rejected route never falls back to unconstrained motion.
`OBJECT_UPRIGHT_VALIDATED` identifies successful constrained plans. Use
`placement_motion_orientation_policy:=free` explicitly to restore unconstrained
held orientation. Endpoint `orientation_policy` and packing support modes remain
separate choices. Restart reasoning and prepare fresh after changing this policy.
It is a planning constraint; physical tracking remains to be validated.

### Saved grounding and segmentation previews

Shared grounding/segmentation calls save the latest previews in
`~/robobrain_inference/`: `object_bounding_box.png`, `destination_bounding_box.png`,
`object_segmented.png` and `destination_segmented.png`. Bounding-box files show
RoboBrain's pixel box on the source RGB; segmented files retain only EdgeSAM's
masked pixels against black. `object_input.png`/`destination_input.png` preserve
source RGB, and the corresponding `_mask.png` files show binary masks. Association
uses source frame/timestamp/dimensions and the grounded box in a bounded cache;
segmentation without matching grounding is not mislabeled. Files represent the
latest successful calls, not a history of every object. Restart the reasoning
stack after updating this logging behavior; saved images are outside the repo.

### Hollow-object cloud completeness

The foreground filter keeps its 3 cm camera-range band for initial component
selection. It additionally retains substantial components within the original
EdgeSAM mask when at least three measured samples connect within the existing
2 cm 3-D neighborhood of selected foreground. This recovers cup rear/interior
surfaces split by depth gaps without widening the range band globally or inventing
hidden surfaces. Tiny and distant unconnected components remain excluded. The
same validated mask feeds HGGD, object-world publication and subsequent geometry.

The outward bend keeps the original hinge axis instead of rotating freely toward
two separate normals. It compares physical hand-body displacement against the
measured object center before IK, after actual IK/FK and after palm recovery.
Ambiguous geometry retains the original validated fallback. These checks do not
certify improved physical contact or holding force.

### Accepted grasp angle review in RViz

With the reasoning stack running, open the focused visualization:

```bash
ros2 launch vamp_node grasp_angles_rviz.launch.py
```

Run a normal calculation-only `pipeline_task --object 'beige cup, grasp its body'
--grasp-only` query. The **Accepted Grasps - Actual Angles and Tuning Status**
display uses `/vamp_planner/grasp_candidates_markers` with transient-local QoS,
so it also shows the latest result when RViz is opened after calculation.
Finger meshes and RGB tool axes use the accepted IK/FK pose; labels give final
roll/pitch/yaw in base_link degrees. These are absolute tool angles, not the
10°/30°/50° pad-alignment tolerance or the amount of correction. **Surface aligned**
means the alignment check passed; **Original pose (angle fallback)** means the
original validated pose was retained. Rejected tuning attempts are shown separately in **Angle Attempts - Preview Only
Including Rejected** on `/vamp_planner/grasp_angle_trials`; pre-ranking proposals
remain hidden. The preview shows up to three actual 10°/30°/50° trials from
the selected grasp, with the requested finger-to-surface angle and rejection
stage. Trials are geometric proposals, not executable poses. Unrelated best-fit
rotations are not shown as requested-angle trials. Disable that display to
see only accepted grasps. This launch only starts visualization; it does not calculate
a new grasp or execute motion. The existing `grasp_debug_rviz.launch.py` remains
available and accepts an optional `rviz_config` override.

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

Measured HGGD grasps also undergo final physical finger checks, including original-angle
fallbacks. The planner prefers 4 mm tip engagement, then tries 7/10 mm if needed,
retaining the measured contact pair. Bounded contact-centered IK adjustments must
keep both contacts inside the usable pads and the open fingers at least 3 mm
above the locally measured table plane. Unsafe original fallbacks are rejected;
a missing reliable plane is reported as a rejection rather than assumed clear.
Accepted results expose `SHALLOW_CONTACT_VALIDATED` and
`FINGER_TABLE_CLEARANCE_VALIDATED`; `[SHALLOW GRASP]` logs adjusted distance
and table clearance. These endpoint checks do not remove existing explicit
approach/lift collision bypasses or verify physical contact.

Complete plate pick/place preview, with every execution stage queued and paused:

```bash
ros2 run task_pipeline pipeline_task \
  --object 'striped white plate' --destination tray \
  --queue-pick-place --lift-distance 0.05 \
  --output "$HOME/agx_piper_data/reports/plate_pick_place_queue.json"
```

Keep `allow_physical_grasp_execution:=false` for mock development. The report
lists completed calculations and the actual pending grasp, gripper, lift,
transfer, descent, release and slot-commit tasks. A paused queue has not moved
the arm. Use `--check-cached-pick-place` for a non-actuating readiness check
before the existing 60-second source/plan expiry. See the
[complete prepared pick/place API](docs/STANDARD_PIPELINE_API.md#complete-prepared-pickplace)
for the execution contract and its physical gate.

Grasp-angle selection appears in the stack terminal at normal log verbosity:
`[GRASP ANGLE SELECTED] ... ANGLE_SELECTION=ADJUSTED target=50deg`, or
`ANGLE_SELECTION=ORIGINAL (...)`. The target is the finger-to-contact-surface
angle, not a 50-degree rotation from the original pose. Individual trial logs do
not identify the final winner. For complete preparation, use
`[PREPARED GRASP SELECTED]` to see the actual candidate retained after all motion
checks, which can differ from the initial HGGD ranking.

Placement performance keeps the same validation rules. Exclusive adaptive grids
validate single-slot fits before spending time on combinations; combinations
remain the fallback when no single-slot search result passes. Held-object
collision checking uses conservative whole-object/block bounds to skip empty
regions, then runs the unchanged point checks near occupied regions. No geometry
is downsampled by this optimization and no clearance or path sampling is relaxed.

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

The existing `plan_apf` API plans the bare arm from joint targets. It has no
payload attachment or object-upright input and must not be substituted directly
for held-object departure. A payload-aware obstacle-directed departure followed
by the existing checked VAMP transfer is a proposed extension, not implemented
by the lift-distance change. Existing inward/retreat lift fallbacks remain.


Payload-aware lift recovery: public `plan_lift` now checks the lifted endpoint
and, if blocked, calculates a short upright APF escape for the arm and attached
object before transfer. The existing lift collision bypasses remain explicit;
normal full checks must pass at the escaped endpoint and during transfer/descent.
Prepared pick/place calculates every segment before execution, using the same
shared backend. Ordered layouts request 5 cm lifts. `LIFT_APF_ESCAPE_VALIDATED`
marks recovery; its waypoints and actual endpoint are included in the lift result.
See [lift/escape API details](docs/STANDARD_PIPELINE_API.md#payload-aware-lift-and-escape).

### Reusable manipulation memory

The master now keeps up to eight active object clouds and one active destination
with its grid, plus a small name/location directory. Default `pipeline_task`
recipes reuse a destination after checking it against current depth. Missing,
moved or stale geometry is grounded again; calculation failures keep valid data
available for a quick API retry. No actuator retries are added.

The master log prints each pipeline step, pending tasks, active object names/XYZ
and destination name/XYZ. Inspect the same state with:

```bash
ros2 service call /pipeline/get_manipulation_memory my_actions/srv/GetManipulationMemory '{}'
```

Use `--queue-pick-place` to calculate and leave the physical tasks waiting.
`--fresh-perception` selects the previous explicit perception recipe.
See [memory APIs and retry examples](docs/STANDARD_PIPELINE_API.md#master-owned-manipulation-memory-2026-10-19).
