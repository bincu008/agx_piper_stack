# Shared microstep API

For the accepted, not-yet-implemented MCP and execution-session design, see
[Agent-controlled arm API design](AGENT_CONTROL_ARCHITECTURE.md). It distinguishes
current APIs from proposed segment/gripper/TCP controls and documents the planned
grounding-to-lift sequence. Runtime API documentation below describes current code.

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
| Held payload | `prepare_held_scene`, `prepare_held_motion`, `execute_lift`, `execute_placement_transfer`, `execute_placement_descent`, `release_placed_object` |
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
the ten-box report are under `~/agx_piper_data/reports/microstep_pipeline_validation/`. Rebuild and source
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

Matching prefers one fitting slot, including a large square. Combinations are
ranked by the fewest occupied slots; equal-count alternatives use the least
reserved area. Search and downstream candidate scores preserve this ordering.
Measured footprint containment, support, reservation and motion validation still
decide whether a candidate is valid.
Exclusive adaptive grids evaluate the fully checked single-slot tier first and
evaluate combinations only when that tier produces no valid search result.
The candidate count remains an upper bound of 16 by default; it need not be
filled with lower-priority groups. Shared sub-cell and legacy grids retain their
existing exhaustive search.

`commit_floor_slots` (`my_actions/srv/CommitFloorSlots`) calls
`/placement/commit_floor_slots`. Its `selected_pose` uses the exact retained
source header and either the original candidate or its bounded achieved FK pose.
Adjusted poses must stay within 25 mm object-center displacement/12 degrees of a
retained candidate and their full projected geometry must fit that same free
slot group (physical bounds for one slot). These are association bounds, not IK
acceptance tolerances. Call it only after successful placement validation. It commits the selected footprint's cells, rejects occupied
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


## Payload execution microsteps

The authoritative catalog includes `prepare_held_scene` (`PrepareHeldScene`),
`prepare_held_motion` (`PrepareHeldMotion`) and the four `HeldMotionStage` APIs:
`execute_lift`, `execute_placement_transfer`, `execute_placement_descent`,
`release_placed_object`. Public endpoints are `/pipeline/<function>`; executors
live in `master_node/held_motion.py` and geometry preparation in
`placement_node/held_scene.py`.

`CacheGraspPlan` optionally carries the original organized cloud and validated
source mask, plus the original semantic mask for bounded seam recovery. All must match its source header/dimensions; held planning requires
both. Refinement remains tied to the selected candidate. `PrepareHeldMotion`
uses the authoritative cached refinement, derives immutable payload/obstacle
geometry, and calls the existing `plan_placement` engine. Full planning additionally
needs placement candidates and a destination mask from the same observation.
`PlanPlacement.lift_only` validates a bounded lift without destination candidates;
`lift_distance_m` defaults to 10 cm in the engine and 5 cm for held preparation.
Its response adds `lift_ee_pose` and `lift_joints` alongside existing path arrays.

The master keeps one revision-bound plan. No caller-supplied trajectory can bypass
planning. A sequence is PREPARED → LIFTED → AT_PREPLACE → PLACED → RELEASED.
Joint/TCP/gripper feedback must be newer than one second; gripper confirmation
must follow the relevant command. Post-actuation errors become `STATE_UNKNOWN`.
Cancellation is still between microsteps; emergency stop retains its direct path.
The `grasp_and_lift` recipe stops at LIFTED and deliberately omits release and
placement. Physical execution remains off unless explicitly enabled by an operator.


## Complete grasp-and-lift preparation before movement

`prepare_grasp_lift` (`PreparedGraspLift`) validates every path without commands;
`get_prepared_grasp_lift` returns readiness/revision without recalculating; and
`execute_prepared_grasp_lift` (`HeldMotionStage`) executes those cached paths.
`validate_prepared_grasp_lift` (`PreparedGraspLift`) is the calculation-only
second-prompt check: it checks freshness, revision, empty-hand idle state,
starting feedback and the same nominal command bounds as preparation. It
preserves `PREPARED`, returns `CACHED_HANDOFF_VALIDATED_NO_ACTUATION`, and neither
plans nor commands hardware. `pipeline_task --check-cached-grasp-lift` calls it.
This check does not emulate actuator tracking, contact or grasp success.
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
A reachable endpoint alone does not guarantee a continuous
Cartesian approach; all intermediate IK and collision checks remain mandatory.

The authoritative master implementation is `prepared_grasp_lift.py`. Preparation
uses the initial HGGD candidate, not a close-up observation requiring prior motion.
`PlanCartesianGraspApproach` accepts optional immutable `source_object_cloud` and
`collision_scene` from `prepare_held_scene`. With these present, it retains the
source as blocking geometry, with only the existing final inner-pad contact
allowance, and validates against the actual commanded open width. Both clouds
must match the original base-frame observation stamp. The legacy refiner cache
path remains when both clouds are absent.

The first CLI command is `pipeline_task --prepare-grasp-lift --object ...`.
It returns success only after approach, lift and pre-grasp paths are all ready.
The second is `--execute-cached-grasp-lift` for the same object. It contains only
`get_prepared_grasp_lift` and `execute_prepared_grasp_lift`; no motion-time planning
calls occur. Failed preparation clears old readiness. Readiness lasts at most
60 seconds and is tied to the exact cache version, start joints/TCP and revision.
Live hardware faults still stop execution and mark unknown state; completing
planning does not guarantee actuator or contact feedback success.


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

### Public prepared grasp/lift stages (2026-10-13)

`prepare_grasp_lift` calculates and freezes all paths before movement. Its
`plan_revision` binds every subsequent command to the same observation and plan.
`plan_lift` is a public lift-only adapter to the existing `plan_placement` solver;
it forces `lift_only` and full-motion planning, without changing IK or VAMP.

| API | Function / required session phase |
| --- | --- |
| `validate_prepared_grasp_lift` | Check cached paths, source and current feedback; no actuation |
| `get_execution_state` | Read revision, source header, expiry, completed/waiting stages and physical gate |
| `execute_plan(segment=pregrasp)` | Frozen transfer; `PREPARED` → `AT_PREGRASP` |
| `gripper_open` | Confirm prepared jaw clearance; `AT_PREGRASP` → `OPENED` |
| `execute_plan(segment=approach)` | Frozen near/final grasp path at 5% speed; `OPENED` → `AT_GRASP` |
| `gripper_close` | Close at 1 N and confirm fresh contact; `AT_GRASP` → `CONTACT_CONFIRMED` |
| `execute_plan(segment=lift)` | Frozen lift with confirmed held object; `CONTACT_CONFIRMED` → `LIFTED` |

Gripper requests include object and revision. Zero width/force selects the
prepared default; nonzero values must match it. These operations do not permit
arbitrary gripper overrides. Near-grasp waypoints remain inside the validated
approach segment. An actuator or feedback failure marks `STATE_UNKNOWN` and
blocks automatic retry. Every actuated stage restores 20% speed on exit.
`execute_prepared_grasp_lift` remains a compatibility composite using these
same stage implementations. Older refined-grasp and held-placement services
retain their separate protocols; they cannot execute over an active prepared
session. Held-object placement/release and an MCP adapter are still separate
future work.

`RunPipelineSteps.Goal.stop_before_step_id` defaults to empty (run all).
An explicit queued ID pauses before that task with `PAUSED_AT_STEP_BOUNDARY`;
remaining tasks and predecessor bindings stay in the master store. Action
success means the requested prefix succeeded, **not** that physical tasks ran.
A misspelled boundary is rejected before execution. Resume the same context
using the action after reviewing feedback and enabling physical execution;
source/plan freshness is still required (60 seconds).

For a full queued mock trial:

```bash
ros2 run task_pipeline pipeline_task \
  --object 'small white bowl at bottom left' \
  --queue-grasp-lift --lift-distance 0.05 \
  --output "$HOME/bowl_queued_pipeline.json"
```

The CLI prints completed API timings and the actual master `task_pipeline`
snapshot, including the five waiting physical tasks. Mock feedback is recorded
replay, not actuator tracking: it cannot execute or confirm these physical tasks.

### Prepared-session failure recovery (2026-10-13)

The 60-second prepared/source freshness limit applies **before the first physical
stage starts**. Once accepted, execution retains an immutable source and uses
`prepared_execution_timeout_sec` (default 300 seconds) at stage boundaries.
Observation/cache age cannot expire an otherwise valid active session between
pregrasp and gripper opening. Revision, phase, feedback and collision gates
remain active; timeouts never trigger automatic actuator retries.

On an action failure involving prepared execution/caching, the master flushes
idle pending queues for that object. A known idle empty-hand session
(`PREPARED`, `AT_PREGRASP`, `OPENED`, `AT_GRASP`) is cleared together with its
source/refinement caches and preview, provided hardware feedback is fresh and
no collision is active. The failed action remains unsuccessful and returns
`SESSION_RESET_READY_FOR_NEW_PLAN`; rerun preparation with fresh inputs.
The arm stays at its physical position. Completed/failed results remain for
inspection; unrelated object queues and busy contexts are preserved. Context
owners still clear completed result storage through `clear_step_context`.
Cancellation retains its existing between-step behavior.

Public `/pipeline/reset_execution_session` (`std_srvs/srv/Trigger`) provides
this same non-actuating reset explicitly. It does not clear held or uncertain
state. A failed actuator/contact confirmation retains `STATE_UNKNOWN` and
requires physical inspection. **Only after the operator confirms the arm is
stationary and the hand is empty**, use:

```bash
ros2 service call /pipeline/reconcile_empty_hand std_srvs/srv/SetBool '{data: true}'
```

This explicit reconciliation also requires fresh feedback, low measured joint
velocities, no active collision and no running pipeline. It clears bookkeeping,
not physical geometry, and sends no motion/gripper/home command. Never call it
automatically or claim empty-hand confirmation from a failed closing command.

`plan_final_approach` now requires `final_grasp_joints`: six previously validated
joints associated with `refined_grasp_pose`. It retains the supplied endpoint and
only solves optional near-grasp poses. Missing/mismatched endpoint data is rejected;
all consumers must rebuild against the changed my_actions schema.

`plan_motion` (`CheckAndReplan`) includes `grasp_motion`, default false. Grasp
controllers set it for 0..67 environment-sphere checks across RRT-Connect edges;
general and held-transfer paths retain full sphere geometry. Self-collision
remains full. Rebuild consumers and restart the reasoning stack after this change.

## Complete prepared pick/place

`prepare_pick_place` uses the same source-linked grasp pool and public
`prepare_held_scene`, `plan_final_approach`, `plan_placement` (full-motion mode)
and `plan_motion` operations. It caches only a complete pregrasp, approach,
lift, transfer and descent plan after command-limit/continuity checks and a
non-consuming floor-slot commit check. No actuator calls or slot reservations
occur during preparation. `get_prepared_pick_place` and
`validate_prepared_pick_place` expose that same revision and selected object pose.

`pipeline_task --queue-pick-place --object 'striped white plate' --destination tray`
constructs the complete master queue and pauses before `pregrasp`. Its remaining
operations are, in order: `execute_plan(pregrasp)`, `gripper_open`,
`execute_plan(approach)`, `gripper_close`, `execute_plan(lift)`,
`execute_placement_transfer`, `execute_placement_descent`, `release_placed_object`,
then `commit_floor_slots`. All physical stages share the same prepared revision
and ordered session; they never run IK or replan. `PAUSED_AT_STEP_BOUNDARY`
means calculation completed and pending stages were retained, not executed.

`--prepare-pick-place` calculates without queuing physical stages;
`--check-cached-pick-place` checks readiness without actuation.
`--execute-cached-pick-place` explicitly submits the execution recipe for a
reviewed fresh plan and requires the physical execution gate. Preparation has
the existing 60-second pre-execution/source-cache lifetime; restarting or
changing source geometry invalidates readiness. Active execution uses the
existing execution-session timeout. Contact/arrival and failure/unknown-state
checks remain automatic. A mock plan does not authorize live hardware movement.

### Relaxed prepared placement and lift handoff

Prepared placement defaults to 5 mm target position and 10 degree target
orientation matching, with a 5 degree object-upright limit. Local descent keeps
1 mm position and 0.01 rad orientation tracking around the accepted FK endpoint.
Floor targets include 3 mm release clearance plus the target IK position allowance
(8 mm total by default). Nominal joint limits and transfer/descent collision
checks remain. Floor-slot dry-run checks the actual achieved pose; a failed
slot match causes shared validation to try another candidate before caching.

If the completed lift endpoint fails the full transfer checker, shared
`plan_placement` validation can retry calculation at +1/+2 cm above the requested
minimum, with no fixed total-height ceiling. Per-backend call records include `lift_distance_m`.
No motion command is retried, and the transfer collision checker is never bypassed.

### Multiple validated grasps for prepared motion

`calculate_grasps` retains up to six diverse validated HGGD grasps for single
objects instead of truncating to one. The existing `cache_grasp_plan` binding
preserves the full returned array. `prepare_pick_place` / `prepare_grasp_lift`
try ranked candidates through all required planning checks, stop at the first
complete plan and cache only its revision-bound paths. Limits remain eight
candidate trials and a 30-second between-attempt budget. No new API or actuator
retry is introduced. `GRASP_CANDIDATES_TRIED` and the prepared selection log
identify the winner; RViz prepared markers show that actual candidate. Fewer
than six candidates may exist, and all-candidate failure remains a failure.

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


## Payload-aware lift and escape

`/pipeline/plan_lift` (`my_actions/srv/PlanPlacement`, lift-only adapter) calculates
the requested initial lift, then tests its predicted endpoint with full arm,
finger and payload collision checks. If blocked, bounded upright Cartesian APF
moves away from obstacles. It never actuates. Placement-only batching flags are
cleared by the adapter so an empty destination list still reaches the engine.

The accepted escape is appended to `lift_path_flat`; `lift_joints` and
`lift_ee_pose` describe the escaped endpoint. `LIFT_APF_ESCAPE_VALIDATED` appears
in `motion_status`. `lift_seconds` includes escape calculation, while shared API
traces measure complete request-to-response time. Initial lift bypass markers
remain explicit; escape endpoint clearance does not certify the bypassed lift.

Prepared pick/place calls `plan_lift` before destination `plan_placement`, which
revalidates its own departure, transfer and descent before a complete session can
be cached. Ordered layout requests use the same backend and explicitly request
5 cm lifts. Execution uses the existing prepared lift segment; no APF/IK or
placement replanning runs after physical lifting starts. Failed escape returns
`LIFT_APF_ESCAPE_EXHAUSTED` without an executable partial path. Contact tolerance
is limited to escape and normal transfer/descent checks remain active. Bounds
and geometry rules are documented in AGENTS.md.

## Master-owned manipulation memory (2026-10-19)

`master_node/manipulation_memory.py` owns bounded active geometry. The shared
registry remains the API catalog; the memory is not another planner. The default
`pipeline_task` calculation recipes use this memory; `--fresh-perception` retains
the explicit presence/ground/segment recipe for comparison. Synthetic overlay
recipes retain their explicit frozen-scene path.

| API (`/pipeline/` prefix) | Function |
| --- | --- |
| `resolve_target` | Resolve a name with role `object` or `destination`; capture if no synchronized inputs supplied. Reuse verified data or separately call presence, grounding and segmentation. |
| `refresh_active_target` | Force fresh inference for that name/role. |
| `get_active_target` | Read the typed active snapshot, including original geometry source header. Does not refresh or authorize execution. |
| `forget_active_target` | Drop active geometry; retain the name/location hint. |
| `get_manipulation_memory` | Read metadata JSON: active names/XYZ, directory, grid/reservations and actual master pipeline tasks/results. No image/cloud arrays. |
| `calculate_active_grasps` | Run the existing shared grasp API on the named active source; request another affordance after definite calculation failure. |
| `find_active_placements` | Search using the active object and destination (or resolve the requested destination), through shared held-scene/search APIs. |
| `get_floor_slots` | Read bounded placement grid layouts and reservation metadata. |

Memory is process-local and clears on master restart. Directory entries retain
only name and `base_link` location (metres), keyed by
world object ID; at most 256 hints and eight active objects are retained. One
active destination retains its source geometry and grid. Switching destinations
releases old active geometry and unexecuted prepared/grid state, retaining its
hint. Hints guide identity; they are not trusted manipulation geometry.

Every normal resolve validates cached geometry against a fresh synchronized
manual-FK/X observation. Unchanged destinations keep their original geometry and
grid IDs; the current mask belongs to the current observation. Old geometry is
explicitly mapped into the current logical planning scene, not claimed as a new
camera capture. Depth evidence, camera viewpoint (for grasp RGB-D), calibration
or age mismatch triggers inference. Location matching cannot prove semantic
identity after an identical-looking replacement; use `refresh_active_target`
when external changes are known. Reuse is bounded to five minutes since the
last successful resolve; direct grasp/search retries require source verification
within 60 seconds. Backend source/cache expiry can still require refresh.

Calculation failures retain active geometry for another API call. Queued retry
steps must use a new step ID/context because completed step IDs are idempotent.
They do not automatically retry actuators, invalidate another execution, or
turn missing IK/path results into successful actions. Held/unknown execution
state blocks memory mutation. Only confirmed release updates the physical object
location; a calculated/committed slot alone does not mean an object moved.

Example, with the mock node and stack on the same ROS domain:

```bash
ros2 run task_pipeline pipeline_task \
  --object 'small white bowl at bottom left' --destination 'dark rectangular tray' \
  --queue-pick-place
# Pauses before motion; master prints completed/current tasks, pending tasks,
# active object names/XYZ and destination name/XYZ at each step boundary.
ros2 service call /pipeline/get_manipulation_memory my_actions/srv/GetManipulationMemory '{}'
# Retry geometry only, while the active observation is fresh:
ros2 service call /pipeline/calculate_active_grasps my_actions/srv/FindGraspCandidates \
  "{target_description: 'small white bowl at bottom left', affordance_type: 'RIM'}"
ros2 service call /pipeline/find_active_placements my_actions/srv/FindPlacementCandidates \
  "{object_name: 'small white bowl at bottom left', destination_name: 'dark rectangular tray', orientation_policy: upright, support_mode: floor_slots}"
```

The retry returns proposals, not an executable prepared session. Selection,
`cache_grasp_plan` and complete motion preparation must follow before execution.
A second CLI command for another object reuses the verified tray. VLM object and
destination resolution uses the same public memory APIs; its legacy visibility/
NBV behavior remains separate. Presence/ground/segment backend calls retain
individual timing/traces even though `resolve_target` is a composite API.
