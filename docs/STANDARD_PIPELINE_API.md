# Shared microstep API

The official reasoning stack hosts `pipeline_node` in the master process by default. `api_dispatch_in_master:=false` restores the separate dispatcher process for comparison. Its catalog in
`task_pipeline/registry.py` is the common source for production controllers,
declarative recipes and the synthetic-box benchmark. Public engine calls at
`/pipeline/*` run through `MasterStepRuntime` inside `PiperMasterControl`.
The master owns the queue, typed results, grasp caches and arm state. Existing
numerical engines and their solvers are retained.

## Public operations

| Purpose | Canonical functions (prefix `/pipeline/`) |
|---|---|
| Observe | `capture_observation`, `pin_grasp_observation`, `pin_placement_observation`, `get_arm_state` |
| Interpret | `verify_presence`, `ground_object`, `segment_object`, `infer_vision`, `infer_yolo` |
| Find a grasp | `prepare_cloud_scene`, `prepare_grasp_input`, `calculate_grasps`, `calculate_cloud_grasps`, `select_grasp` |
| Store grasp state | `cache_grasp_plan`, `get_cached_grasp`, `cache_grasp_refinement` |
| Refine and approach | `move_pregrasp`, `refine_grasp`, `plan_grasp_approach`, `plan_final_approach` |
| Complete grasp | `open_grasp_gripper`, `execute_grasp_approach`, `close_grasp_gripper`, `verify_grasp_contact` |
| Place | `find_placements`, `search_placements`, `validate_placements`, `plan_placement` |
| General motion | `solve_pose`, `plan_motion`, `move_to`, `execute_trajectory`, `set_gripper` |
| Object state | `update_object`, `resolve_object`, `get_world_state` |
| Microstep control | `run_step`, `store_step_result`, `submit_steps`, `get_step_context`, `clear_step_context` |
| Wait for queued work | `/pipeline/run_steps` (`my_actions/action/RunPipelineSteps`) |

The catalog also retains navigation/recovery functions and old task interfaces.
`execute_grasp`, `plan_layout` and `look_at_target` remain compatibility facades;
they construct sequences of shared calls. Engine internals such as individual
IK iterations stay inside their numerical operation rather than becoming RPCs.

Discover all 56 service/action entries, request/response schemas and completion
semantics with:

```bash
ros2 service call /pipeline/list_functions my_actions/srv/GetPipelineFunctions '{}'
ros2 topic echo /pipeline/timings
ros2 service call /pipeline/get_metrics my_actions/srv/GetPipelineMetrics '{}'
```

## Constructing a sequence

Each `PipelineStep` has a unique ID, catalogued function name, CDR-serialized typed
ROS request, predecessor field bindings, and a `stop` or `continue` failure policy.
Bindings copy typed fields from successful earlier responses; they do not evaluate
Python expressions or accept arbitrary backend endpoints. A failed predecessor
cannot supply a result to another step.
The reserved source `@response_cdr` binds a complete serialized response, for
example from `refine_grasp` directly into `cache_grasp_refinement.refinement_cdr`.

1. Build steps with `task_pipeline.recipes.step` and existing `my_actions` types.
2. Call `submit_steps`; retain the returned context ID and revision.
3. Send that ID to the `run_steps` action. Its result waits for completion, failure
   or cancellation at a step boundary; feedback identifies the current function.
4. Query `get_step_context` for queue metadata and optionally typed response CDR.
5. Call `clear_step_context` when the results are no longer needed.

For example, the calculation-only cup recipe binds `observe.image` to
`ground_cup.image`, `ground_cup.pixel_box` to `segment_cup.bbox`, and the selected
grasp and generated placements to `placement_ik`. The master resolves these
bindings when each step runs. Controllers choose the task order; they need not
carry their own copies of every intermediate image or pose.

`run_step` executes one typed request and retains its response. With the same
context, step ID and request payload, a retry returns the retained result instead
of executing again. A different payload under the same ID is rejected.
`store_step_result` imports a successful typed response for later consumers; it
does not execute that response or authorize motion.

Appending is atomic, rejects forward/missing dependencies and checks an optional
expected revision. Attempted steps leave the pending queue; results remain until
cleared. `continue` permits independent later objects to run after a failure.
Defaults bound storage to 32 contexts, 256 steps per context and 256 MiB of CDR
payloads. Completed automatic single-call contexts may be evicted; explicitly
named or submitted contexts must be cleared by their owner.
The box benchmark retains its latest run's context IDs in the report and releases
them before its next run, so repeated runs do not accumulate stale result memory.

## Launch and run

Source the environment in each terminal:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export LD_LIBRARY_PATH="$HOME/.local/onnxruntime-1.15.1/lib:$LD_LIBRARY_PATH"
```

Start the mock and safe stack in separate terminals:

```bash
ros2 launch mock_hardware_node mock_hardware.launch.py
```

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio placement_backend:=heightmap \
  grasp_candidate_backend:=hggd pipeline_profiling:=true \
  allow_physical_grasp_execution:=false
```

Use `inference_profile:=server` for the optimized remote RoboBrain server.
Vision and VAMP remain composed; the VLM runs in its own process so network
callbacks do not occupy the perception/planner executor. Its multithread executor
also permits API responses to arrive while a legacy command callback waits.
Camera and arm drivers
remain separate and should be stopped for mock-only validation.

Run a 17-step cup calculation (including slot reservation) and wait for its final result:

```bash
ros2 run task_pipeline pipeline_task --object 'brown cup' --destination tray \
  --output "$HOME/cup_pipeline.json"
```

Use `--grasp-only` for its first ten steps. This recipe calculates poses and updates
the grasp cache; it contains no actuator commands. For the calculation-only box
benchmark, start its node and call its blocking trigger:

```bash
ros2 launch placement_node box_benchmark.launch.py \
  box_count:=10 optimize_pipeline:=true plan_pick_place_motion:=true
ros2 service call /box_benchmark/run std_srvs/srv/Trigger '{}'
ros2 launch vamp_node grasp_debug_rviz.launch.py
```

## Execution and interrupt boundaries

New actuator APIs retain `allow_physical_grasp_execution`, collision checks and
master busy-state checks. General `move_to` and `execute_trajectory` are limited
to empty-hand motion and use VAMP validation. General `set_gripper` permits
empty-hand opening; closing requires the ordered validated grasp stages.
Releasing a held object requires a validated placement execution implementation,
which is not supplied by these calculation APIs.

The grasp stages preserve the fixed-orientation, bounded Cartesian approach,
measured pre-grasp checks, minimum valid opening, 5% approach speed and restoration
to 20%. Closing occurs only after the final waypoint; feedback must confirm
contact before master state marks an object held. Protective speed changes stay
atomic inside the approach execution rather than becoming optional queued steps.

Action cancellation stops between steps. It cannot forcibly cancel an in-flight
ROS service. Emergency-stop and hardware-driver paths remain direct. A timeout
reports `BACKEND_TIMEOUT_STATE_UNKNOWN`; do not automatically retry an actuator
operation. Such a context rejects further work until actual state is reconciled
and the owner explicitly starts a new context. Existing legacy task interrupts
remain available; this change does not replace the hardware stop mechanism.

Master parameter `pipeline_step_timeout_sec` and dispatcher `request_timeout_sec`
default to 600 s; capture timeout is 30 s. RoboBrain inference keeps its separate
configurable 120 s timeout. Synchronized source images, clouds and manual
FK × calibration transforms retain their association; no computational TF lookup
was introduced. Placement can use the exact pinned observation after newer
frames have evicted it from the rolling buffer.

## Timing and validation

`PipelineTiming` separates `api_inclusive`, `backend_inclusive`, engine compute
and validation scopes. These are nested measurements: do not add them together.
Use action/run elapsed time for total completion; compare repeated functions
with the same input mode and full-motion flag.

Synthetic boxes use `calculate_cloud_grasps` with known geometry rather than HGGD.
They share public placement search, VAMP validation and motion planning with
deployment. Tray grounding/segmentation still uses real image APIs. Legacy
single-object placement is labelled `legacy_pose_only`; the full-motion benchmark
uses a frozen scene and requires valid lift, transfer and descent paths.

Software-only validation on 2026-10-05 accepted all ten boxes: six on the floor,
four stacked, zero failed, in 230.08 s. Median added API transport overhead was
about 24 ms per call when comparing API/backend medians. Placement-search compute was slower than the prior run;
this refactor does not claim a pipeline speedup. The 14-step brown-cup run found
a safe HGGD grasp and eight height-map candidates in 21.47 s, then correctly
rejected all eight at placement IK. That reachability limitation remains unresolved.
The legacy `[FIND_GRASP] brown cup` command also returned one safe, master-cached
candidate in 12.58 s after the standalone VLM executor fix.

Affected package checks passed: task_pipeline 30, placement_node 94, vlm_node 7,
vamp_node 48, plus 31 master-node functional tests. New API modules passed flake8
and docstring checks. Live actuator calls were rejected by the disabled physical
gate; live context checks verified idempotent retry and explicit result cleanup.

No physical execution was tested. Local build/test logs, per-step cup results and
the ten-box report are under `~/microstep_pipeline_validation/`. Rebuild and source
the workspace before restarting clients because shared ROS interfaces changed.

## API transport performance

The public names and master-owned result store are unchanged. Colocated dispatch
calls the runtime directly; typed public calls avoid the internal CDR envelope
round trip. Queued recipes submit once with `submit_steps` and `run_steps`.
The conditional box benchmark keeps individual `run_step` calls because each
placement result determines the next scene and edge/stack fallback.

`RunPipelineStep.bindings` accepts the same `StepBinding` references as queued
steps. A sparse request can bind `image` from an earlier `capture_observation`
response, or bind `object_cloud`/`scene_cloud` from `prepare_cloud_scene`.
Predecessors must be successful and in the same context. The master decodes each
predecessor once per step and copies selected fields so backend mutation cannot
change stored artifacts. No persistent duplicate decoded cloud cache is added.
Geometry registration preserves all points and only validates layout and source
headers. The benchmark registers each box scene once and reuses it for grasp
queries. Failed-box obstacles and motion validation remain enabled.

The benchmark prints `[API END-TO-END]` rows and saves `api_timing` in its final
`report.json`. Client times start on entry to the API wrapper and end after the
typed response is decoded. They exclude construction of the caller's input
message before entering that wrapper. Seven disjoint intervals are recorded:

1. `client_request`: envelope preparation and service readiness.
2. `request_delivery`: client send to API handler entry (DDS and scheduling).
3. `server_request`: API entry to backend invocation, including result bindings.
4. `backend`: backend ROS transport and computation, including remote VLM time.
5. `server_response`: backend completion to API handler completion.
6. `response_delivery`: API completion to client observation of future completion.
7. `client_response`: client wake-up and typed response decoding.

Intervals sum to each call's total; medians of intervals need not sum to the
median total. `ApiCallTrace` messages on `/pipeline/api_traces` correlate the
`run_step` ID across dispatcher and master. They use host monotonic clocks;
never subtract timestamps across hosts. Missing, timed-out or cache-hit backend
traces are marked incomplete rather than assigned invented durations. These
measurements do not replace `/pipeline/timings` engine scopes.

`numeric_threads:=1` limits BLAS/OpenMP worker counts for Python processes;
override it in the reasoning and benchmark launches together when comparing
profiles. It does not change the IK solver or VAMP algorithm.

Height-map search evaluates bounded batches of cells and reuses identical
support-mask checks; the scoring, clearance and edge/stack policies are unchanged.
VAMP reuses an exact spatial index for unchanged environment points. Every
candidate still passes the original narrow collision and intended-contact tests;
IK solutions and collision decisions are not cached.

The height-map node consumes synchronized ActiveVision geometry. The optional
legacy `/placement/infer_placement` trigger subscribes to raw RGB, cloud and
joint topics only while collecting a fresh synchronized snapshot, then releases
those subscriptions. Capture is bounded by the smaller of five seconds and the
request timeout; unavailable or unsynchronized inputs return an error.

Performance changes preserve search effort: difficult RGB-D/HGGD objects may
require more candidate validation than known synthetic boxes. The box fixture
must not be used to justify reducing deployed grasp checks. The benchmark takes
its source observation through the public capture API; unused image/cloud topic
subscriptions and unused engine clients are not created. Frozen-scene placement
skips building a live obstacle list that would immediately be discarded. Held
object collision queries prune impossible spatial cells while retaining the
same 3 mm narrow test and support contact rules.

Frozen placement geometry is cached only when the entire cloud (including
metadata, bytes and support labels) and support mode match. The planning node
retains at most one scene with at most 8 MiB of input cloud data; larger inputs
still undergo validation but are not retained. Changes to the source object or
gripper still receive fresh attachment, IK and collision checks. No successful
plan or collision decision is cached.

Dense organized-cloud scans resolve XYZ field offsets once per scan. The reader
retains the existing pixel indexing, native float values, finite-point rejection
and base-frame/optical-depth validity rules. It neither downsamples geometry nor
changes target-region growth, surface checks, grasp generation or search effort.

### Frozen simulation obstacles

`FindPlacementCandidates.Request.additional_obstacles` and
`PlanPlacement.Request.additional_obstacles` accept an optional immutable, tightly
packed little-endian FLOAT32 XYZ cloud in `base_link` (maximum 8 MiB). Empty
clouds preserve the existing deployed path. The heightmap merges these points
as blocking geometry; single-object VAMP collision validation keeps every
finite overlay point blocking without applying source-object removal to it.
The overlay is not a replacement RGB-D observation and is not a computational
TF input. Layout callers already supply a complete frozen scene and must leave
this field empty. Optional AnyPlace inference does not support overlays.

`pipeline_task --obstacle-cloud-topic /box_benchmark/scene_cloud` freezes the
latched simulation boxes before submitting the normal cup recipe. No benchmark
setup time is included in the cup action wall time. This recipe validates
placement and pre-place poses only; it does not validate held-object transfer
trajectories or synthesize a new camera image.

`find_placements.support_mode` selects `auto`, `edge` or `stack`.
Stack support is inferred from exposed scene heightmap cells, without object IDs.
The old `support_name` and `stack_support_cloud` fields remain reserved for wire
compatibility and are ignored; named support is no longer a policy.
CLI: `--placement-policy stack`; a synthetic scene additionally uses
`--obstacle-cloud-topic /box_benchmark/scene_cloud`. No saved geometry is needed.

`search_placements` option `stack_minimum_coverage` defaults to 0.6. Image-based
search reads `packing_stack_minimum_coverage` (launch `stack_minimum_coverage`).
Coverage measures projected footprint area at the placement support height;
combined exposed surfaces may satisfy it. Lower cells do not count. Existing
contact hull/COM, area, containment and collision checks remain mandatory.
Auto mode applies the same coverage rule to elevated placements. Benchmark
stack fallback searches all support surfaces in one call using this same engine.

### Ordered RGB-D layout collision inputs

`calculate_grasps` (`FindGraspCandidates`) accepts an optional immutable
`collision_scene`. HGGD still uses the original synchronized image, target mask
and registered cloud. The override changes collision geometry only, allowing a
layout caller to remove successfully picked source samples and append virtually
placed objects before calculating the next grasp. Failed objects remain obstacles.

The scene must use `base_link` and the exact observation timestamp. Its first
image-sized XYZ prefix retains the original pixel order: each point is unchanged
or replaced by NaN. Append placed geometry after that prefix. Removing the active
target, altering finite prefix points, mismatched observations, malformed fields,
more than one million points or more than 16 MiB is rejected. Empty input retains
the existing image-based behavior.

Frozen `PlanPlacement.source_object_cloud` may contain an optional UINT8 `bottom`
field: 0 means ordinary geometry, 1 means measured bottom contact, and 2 identifies
the inferred contour/cap plane. This preserves support contact on tilted source
surfaces. Initial lift allows at most 3 mm of fitted support depth noise for bottom
points during the first 6 mm (up to 15 mm with the departure allowance enabled);
terminal placement retains its 1 mm penetration
limit. Other objects and non-bottom contact remain blocking. These APIs calculate
paths; they do not execute held-object placement or release.

Ordered RGB-D layouts associate discarded depth components with the same source
only when they lie inside its original semantic mask and within 12 mm of its
validated/completed surface. A bounded 10-pixel, 12 mm registration seam excludes
other scheduled object masks and geometry below the observed body, preserving
an elevated object's stand. This changes source ownership, not collision radii.
The strict initial support-plane sample band is 4 mm. With sparse-contact
tolerance enabled, or a positive departure allowance, it includes the 3 mm point
neighborhood, the existing 3 mm bottom-contact
margin and that allowance, capped at 12 mm. The bottom-only penetration limits
above remain unchanged. Scene
`support=3` labels known object geometry and forbids treating its base as source
table support. Ordered layouts label only exposed tops of accepted objects as
`support=2`; their sides and bottoms remain blocking. `PlanPlacement.support_mode`
is `auto` for ordered layouts, allowing final bottom contact on floor (`1`) or
exposed tops (`2`). These tops remain object obstacles during source departure. Rejected floor returns in the active object's mask remain
environment rather than being labeled as another object; overlapping masks and
already-placed clouds retain protection. Labels 0/1/2 retain their existing
obstacle/floor/stack semantics.

`placement_allow_separating_contact_departure:=true` enables the default bounded
departure for existing padded source contacts: 3–4 mm sideways with 0.5 or 1 mm
upward. `placement_departure_contact_tolerance_m` permits a small loss of initial
clearance (default 0.001 m, bounded to 0–0.002 m) during the first 6 mm of departure.
Bottom/table contact may persist for the support band plus the point neighborhood
(at most 15 mm of initial travel), so contact is not revoked while clearing noisy
support samples. Every swept sample still rejects new neighboring contacts, and the endpoint must
pass ordinary clearance before lifting, subject to the separate sparse-contact
budget below. Transfer and robot/finger checks retain normal clearance. Set the
departure allowance and sparse-contact budget to zero for strict previous behavior.
Set `placement_allow_separating_contact_departure:=false` to disable departure.

Attached-object lift and descent checks additionally use a segment-wide sparse
contact budget. Defaults are `placement_noise_max_points:=32`,
`placement_noise_max_voxels:=10` (5 mm cells). There is no spread/distance threshold
between separate contact points. Count unique
environment XYZ samples, not the number of held-object/sample pairs. All accepted
waypoints and midpoint checks share the same budget, so moving through a wall
cannot reset the allowance at every sample. Failed path trials do not consume the
next trial's budget. Points are preserved in the scene; this is bounded contact
tolerance, not proof that those points are noise. Sparse real contacts satisfying
the same limits can also pass. Arm, palm/fingers, object-to-arm and VAMP transfer
checks remain strict. Set `placement_noise_max_points:=0` to disable the budget.
`[ATTACHED NOISE BUDGET]` logs report the tolerated point/voxel count.

The user-selected launch default
`placement_bypass_attached_collision_during_lift:=true` (default) skips only the
attached-object/environment check during the source lift. Set it false to
restore that validation.
Robot, finger and object-arm checks still run during lift; attached-object
environment checking resumes for transfer and remains active during descent.
Motion results include `LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED`, so a successful
calculation with this option does not claim a collision-free lift. Use it with
mock inputs and `allow_physical_grasp_execution:=false`.

The existing `plan_placement` backend includes a last-resort bounded RRT-Connect
retreat to the selected grasp's validated pre-grasp configuration. Search is
bounded to 2 seconds/1000 iterations, within 18 cm of the source TCP, no more
than 1 mm below its starting height, and within 0.40 radians of its starting
orientation; the goal must rise at least 4 cm. Returned paths retain joint-limit
and stage collision checks, midpoint validation, TCP steps at most 2 mm, and
a 4096-waypoint cap. Ordinary lift routes are tried first. This uses the existing
planner and public API, and does not alter the final grasp approach.

## Floor-slot policy and reservation API

`find_placements` / `search_placements` accept `support_mode: floor_slots` with
`orientation_policy: upright`. This searches a cached adaptive 6–12.5 cm square floor
grid, with dynamic gaps and rectangular remainder slots, and retains the normal support and collision tests. It never stacks.
`pipeline_task --placement-policy floor_slots` uses the same public search and
validation functions, then appends one `commit_floor_slots` microstep.

`commit_floor_slots` (`my_actions/srv/CommitFloorSlots`) calls
`/placement/commit_floor_slots`. Its `selected_pose` must match a retained grid
candidate including the exact source header. Call it only after successful
placement validation. It commits the selected footprint's cells, rejects occupied
cell conflicts or unmatched/stale results, and is idempotent for the same source
and pose. `reset: true` clears all cached layouts, pending matches and reservations;
use this for independent calculation trials. It does not modify scene clouds.
Eight layouts and 32 pending search results are retained at most. Different
measured destination boundaries or adaptive-grid configurations produce separate
grids. The first search uses the available floor heightmap to build the largest
inscribed rectangle plus usable extensions. Subsequent occupied-scene feedback
blocks frozen cells without shifting their geometry or IDs. The public search
options and placement-node launch parameters are `floor_slot_adaptive`,
`floor_slot_max_size_m` and `floor_slot_min_size_m`. Reset or node restart
clears the calculation reservations. Physical release acknowledgement is outside
this calculation-only API.

Latched MarkerArray `/placement/floor_slots` uses the source timestamp and
`base_link`: blue free, orange occupied, green selected. Timing reports include
the reservation microstep separately; grid-generation and matching are included
in `find_placements` / `search_placements` and must not be added again.

For frozen-scene held-object validation, use `plan_placement.support_mode: edge`
with grid-generated candidates. `floor_slots` selects search positions; `edge`
expresses the floor-only contact semantics to the existing motion backend.

### Shared-cell footprint reservation

The optional ordered-layout setting `floor_slot_share_unused_space` passes that
option through public `search_placements`. The visible lattice stays unchanged,
but non-overlapping 5 mm footprint masks may share cells. Masks include a one-cell
clearance dilation; search applies reservation exclusion before shortlisting and
commit rechecks exact mask overlap atomically. A shared cell is not a collision
exception. The controller supplies updated frozen scene geometry to normal
`plan_placement` validation, uses `edge` contact semantics, and commits only after
successful complete motion calculation. Ordinary single-object recipes retain
whole-cell reservations. Cached layout/occupied-mask fields cannot be overridden
by external search options. Yellow grid points render reserved mask portions.

`layout_support_mode` selects the same public search for ordered VLM templates.
No new private engine sequence is introduced. Compact ranking favors small groups,
few newly consumed cells, and a contiguous remaining opening; final shortlist
diversity is applied within each size tier. Search uses cached stencils, 1 cm phase
steps and 5° yaws with 16 returned candidates. This is a heuristic and does not
prove globally minimum packing. Reset reservations before independent trials.

Upright placement defaults to `floor_slots`: the CLI/recipe and legacy VLM plan
commit after successful validation, while ordered layout commits each accepted
object before continuing. Empty backend support mode resolves to the configured
layout support mode. Explicit modes remain available; free-orientation VLM and
layout calls use `auto` when the configured default is upright-only grid mode.
Candidate-only queries never consume slots. Whole-cell exclusivity is the
selected policy. The benchmark's explicit edge/stack sequence remains unchanged.
