# Inference profiles and calculation profiling

The official stack supports two selectable endpoint defaults:

```bash
ros2 launch master_node reasoning_stack.launch.py inference_profile:=lmstudio \
  allow_physical_grasp_execution:=false
ros2 launch master_node reasoning_stack.launch.py inference_profile:=server \
  allow_physical_grasp_execution:=false
```

Defaults are stored in `master_node/config/inference/profiles.yaml`. The server
profile remains the default. Explicit `api_server_ip`, `api_server_port`,
`api_server_url`, `model_name`, and `request_timeout_sec` launch arguments
continue to override the profile. Existing request settings and authentication
behavior are unchanged. Endpoints are defaults, not discovery or automatic
failover; select the profile that is actually running.

Add `pipeline_profiling:=true` to collect `[PIPELINE PROFILE]` JSON logs from
VAMP's grasp and placement requests. Measurements include numerical IK time,
pose-IK call count, numerical iterations and FK evaluation count, instrumented
collision-check time/calls, HGGD detection time/calls, and RRT-Connect solver
time/calls. HGGD already logs AnchorNet/LocalNet inference times separately.
Collision timing covers the instrumented candidate/descent checks, not every
geometry helper. RRTC time includes its own internal collision checks. Different
categories can overlap inside RRTC; do not sum all categories as disjoint time.
Same-category nested checks are timed only once. Metrics are request-local and
thread-local, disabled by default, and do not change solver/collision policies.

VLM logs `[VLM TIMING]` request wall time (including image preparation and HTTP),
and EdgeSAM encoder/decoder times. The placement node logs `[PLACEMENT TIMING]`
heightmap search seconds. Client service wall time includes all orchestration,
preprocessing, logging and transport overhead; the difference from measured
components must not be attributed to IK without further profiling.

Safe deployed-pipeline test with the recorded mock scene (no synthetic boxes):

```bash
ros2 launch mock_hardware_node mock_hardware.launch.py
ros2 launch master_node reasoning_stack.launch.py inference_profile:=lmstudio \
  pipeline_profiling:=true allow_physical_grasp_execution:=false
ros2 service call /vlm_reasoning_node/chat_prompt my_actions/srv/Chat \
  "{prompt: '[FIND_GRASP] cup'}"
ros2 service call /vlm_reasoning_node/chat_prompt my_actions/srv/Chat \
  "{prompt: '[PLAN_UPRIGHT] place cup onto tray'}"
```

These calls calculate grasp and placement only. They do not move the arm or
simulate executing the pick/transfer/release. The placement reuses the cached
validated grasp; call it promptly before the configured cache expires.

## Recorded cup measurement

The LM Studio mock-scene run saved artifacts under `~/cup_pipeline_profile/`.
`[FIND_GRASP] cup` took 14.010 s end to end. Inside the grasp computation,
3.375 s total comprised 2.903 s numerical IK, 0.362 s HGGD detection (0.127 s
AnchorNet/LocalNet inference), 0.020 s instrumented collision validation, and
0.000255 s across three RRT-Connect solver calls. Entire pre-grasp path requests
including setup and IK took 0.510 s, which overlaps the IK measurement.
There were 125 pose-IK calls, 4,562 numerical solver/refinement invocations,
2,991,222 iterations and 20,933,658 instrumented `eefk` evaluations. The FK
counter covers `eefk` calls inside these numerical solvers, not every FK call
or sphere-FK evaluation in the system.

`[PLAN_UPRIGHT] place cup onto tray` took 10.922 s end to end: heightmap search
8.461 s, placement validation 0.239 s including 0.238 s numerical IK. All eight
placement candidates failed IK before collision checking. A second body-grasp
prompt found another valid grasp, but its eight upright placement candidates
also failed IK; packing took 10.223 s. No physical grasp/placement was executed.
This is evidence that numerical IK dominates grasp computation in this scene,
while heightmap search dominates the attempted placement; it is not a benchmark
of successful placement descent or a guarantee about other scenes.

## Single-box placement motion measurement

The original placement service checked destination/hover IK and the layout
Cartesian descent, but did not plan the held-object transfer from the source.
The optional `PlanPlacement.plan_pick_place_motion` request now adds a fixed
orientation 10 cm Cartesian lift and VAMP RRT-Connect transfer to the selected
hover. Arm, gripper, immutable environment, and attached cloud are checked along
transfer edges. Only the original bottom support in the 3 mm collision neighborhood of the
source footprint can permit initial lift contact. The mock fixture replaces
duplicate noisy table points only within its known virtual support patches;
other environment geometry remains blocking. Bottom-only contact and the
existing 1 mm penetration limit remain active. This mode never executes hardware motion.

With the mock camera already running:

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio pipeline_profiling:=true \
  allow_physical_grasp_execution:=false
ros2 launch placement_node box_benchmark.launch.py \
  box_count:=1 plan_pick_place_motion:=true \
  output_directory:=~/single_box_motion_profile
ros2 service call /box_benchmark/run std_srvs/srv/Trigger '{}'
```

`[BOX MOTION]` logs and `report.json` contain lift/transfer durations and waypoint
counts. `descent_search_seconds` includes all attempted descent paths, including
rejected candidates; it is part of `pose_validation_seconds`, not an additional
cost. The nested `placement_motion` profile reports its own numerical IK,
collision, and RRTC costs. Collision time within RRTC overlaps RRTC time.

VAMP node parameters `placement_transfer_max_iterations` (5000) and
`placement_transfer_timeout_sec` (30 seconds) bound transfer search. Exhaustion,
collision, lift failure, and timeout are reported as failures; a valid destination
pose alone no longer counts as benchmark success when motion calculation is
requested. The default request flag remains false for existing clients.

This measures motion after the cached grasp. The synthetic benchmark does not
execute pre-grasp, acquire a refined wrist observation, close the gripper, or
calculate a post-release retreat. These stages must not be included in claims
about the measured complete manipulation runtime.

### Recorded one-box run

The live LM Studio/mock run accepted one floor placement with 16 validated grasp
alternatives, candidate #100, and 1.0 mm final position error. Physical execution
was disabled throughout.

| Stage | Seconds |
| --- | ---: |
| Tray grounding and segmentation | 2.297 |
| Grasp requests and validation (18 RPCs) | 6.023 |
| Height-map footprint search (128 candidates) | 4.612 |
| Placement IK/collision validation, including descent searches | 1.009 |
| Cartesian lift (50 waypoints) | 0.142 |
| Held-object VAMP transfer (2 waypoints, 42 state checks) | 0.071 |
| Descent searches, already included in placement validation | 0.304 |
| Full benchmark | 16.245 |

Measured numerical IK was 2.083 seconds in grasp validation, 0.546 seconds in
placement validation, and 0.00014 seconds in the seeded lift. The RRTC solve was
0.0675 seconds; its direct interpolation passed, so no tree expansion was needed.
This is not evidence of performance on a difficult obstructed transfer.

The parent placement profile's wall time includes the nested motion profile;
its numerical IK/collision counters exclude the nested profile. Do not sum parent
and nested wall times, or collision and RRTC times. Grasp service wall time also
includes serialization, orchestration, and work outside the instrumented engine.

The first attempt exposed noisy mock floor points penetrating the virtual cube;
its logs remain in `~/single_box_motion_profile/initial_attempt`. The virtual
support patch correction preserves all non-floor points and applies the same
filter to their support labels. Final `report.json`, `component_report.json`,
serialized service request/response, source/tray imagery, and `box_01.npz` with
lift/transfer/descent joint arrays are in `~/single_box_motion_profile`.
