# Current deployment note

Read ros-pipeline://preparation before using this server. Current mode is
preparation_only: 53 non-actuating state/perception/calculation tools, mandatory
live-input guard and a false master physical gate. Full physical instructions in
this reference apply only to a separately enabled execution mode. Do NOT call
motion/gripper tools in this session. Current arm feedback is stale; complete
pick/place preparation is blocked despite successful grasp and slot calculations.
Use the actual tool allowlist and per-tool verification report. Never present a
failed calculation, stale input or missing memory as successful manipulation.

# Nanobot operating instructions — MCP 3.2.0

Read these server resources before composing a manipulation task:

- ros-pipeline://catalog — actual tool input/output schemas and ROS metadata.
- ros-pipeline://agent-pipeline — complete ordered pipeline and result bindings.
- ros-pipeline://workflow — execution admission, outcomes and recovery.
- ros-pipeline://spatial-map — cube layout, scan times and camera orientation.

Use resources/list and resources/read; no Ubuntu filesystem access is needed.
Refresh discovery on reconnect/version changes. list_functions reports the actual
mode/allowlist/disabled_tools. Default agent_pipeline exposes 62 tools; read-only
startup disables physical dispatch and omits preparation mutation tools. A tool
being discoverable does not establish physical readiness. The current deployment is domain-0 LIVE PERCEPTION ONLY, not the previous
domain-89 replay. Read ros-pipeline://live-perception and list_functions.mode.
This mode exposes 11 tools, with physical execution false; do not attempt the
manipulation sequence below in this session. The full 62-tool mode remains a
separate configuration.

## Compose the same pipeline for every object

1. Read robot state and spatial context. Capture a synchronized observation.
2. Call ensure_scene_memory with requested object_names, active_object_name and
   destination_name, binding that capture. Populate missing locations; inspect
   observed/target_statuses, not only the top-level success flag.
3. Call check_target_view. If LOOK_REQUIRED, plan_next_best_view for the object,
   then an explicitly authorized guarded empty-hand move_to. Confirm completion,
   capture again and refresh the source object. No VLM is required for visibility.
4. Pin the observation. Ground/segment/store the source, or use the combined
   refresh_active_target route. Reuse valid destination memory/grid; otherwise
   ground/segment/store the destination. Never mix old masks with a new image.
5. prepare_grasp_input → calculate_grasps/calculate_active_grasps → update_object
   → select_grasp → cache_grasp_plan (whole candidate pool).
6. find_placements/find_active_placements → plan_next_best_view for the prepared
   look stage → prepare_pick_place → validate_prepared_pick_place.
7. Read get_execution_state and review the complete plan before physical motion.
8. When authorized and the physical gate is true: execute_plan(look if listed,
   pregrasp) → gripper_open → execute_plan(approach) → gripper_close.
9. Require successful contact_confirmed, then execute_plan(lift) →
   execute_placement_transfer → execute_placement_descent.
10. release_placed_object; require gripper_released → commit_floor_slots with
    exact frozen selected_pose, reset=false, validate_only=false →
    execute_withdrawal → execute_placement_retreat.
11. Confirm COMPLETE, then start the next object. Keep destination geometry,
    placed clouds and occupied tray slots; do not reset them between objects.

The agent-pipeline resource supplies exact arguments and predecessor bindings.
Choose one alternative where a slash is shown, not both. Follow the master's
first waiting_stages entry; do not infer readiness from this explanatory list.
The prepared look stage is not a substitute for acquiring a useful view before
source perception/calculation when the current camera cannot see the object.

## Queue and result rules

Preparation calls take context_id, step_id, request_id, arguments and bindings.
Use one gateway-owned context for related calculations. Keep images, masks,
clouds, candidate arrays and source headers in Ubuntu; pass same-context typed
bindings. Never invent CDR, poses, observation headers or source revisions.

run_step executes one allowed calculation. submit_steps only queues; run_steps
start requires the exact returned queue revision and only acknowledges async
start. Poll the same context/job ID until FINAL_ACTION_RESULT, inspecting every
step and action status. Cancellation is between steps, not emergency stop.
Physical functions are forbidden in generic queues; invoke guarded tools directly.

Physical calls use exact execution_ref plus a unique durable request_id (move_to
uses target_joints/request_id, no execution_ref). Withdrawal/retreat use destination
name with the original reference/revision. Copy commit selected_pose from the
admitted session; never shift or replace it. A command ACK is not grasp contact.

Same-ID duplicates retrieve historical receipts without redispatch; they are not
current readiness. Never change arguments under an existing ID. Unknown outcomes
block continuation across reconnect/restart; no remote reset or automatic physical
retry/alternative is supported. One physical grasp attempt per prepared task is
supported, within the ceiling of three. Never erase/replace the ledger to recover.
Local cancellation, emergency and driver watchdog paths remain independent.

## Use spatial knowledge before choosing a view

get_spatial_context combines the shared map, TCP pose and remembered object
coordinates/cube IDs. The workspace has 16 cubes of side .30m in base_link:
X forward [0,.6), Y left [-.6,.6), Z up [0,.6). IDs are x0..1_y0..3_z0..1;
Y0 is rightmost, Y3 leftmost, Z0 lower and Z1 upper. Read returned bounds rather
than recreating another grid. The box is not a guarantee of IK reachability.

last_scanned=zero means never observed. Compare timestamps using the robot's ROS
clock; observed_cell_ids are from the returned map observation, not live gaze.
Empty cells before a depth hit are included; occluded/invalid-depth areas are not.
A timestamp represents sampled partial coverage, not proof a whole cube is empty.
The 30cm workspace grid is separate from the tray's fine floor-slot reservations.

Camera pose differs from TCP pose. camera_forward and camera_up are base_link
vectors. Positive forward-vector X means looking forward, positive Y left,
negative Y right, negative Z down. Several components can be present together.
Use the camera's FOV-derived observed_cell_ids for coverage, not one center ray.
For a stored object, prefer its exact position; for exploration/rescan, call
plan_next_best_view with target_cell_id. Do not combine cell/name/explicit targets.
This only plans; an authorized guarded move_to is a separate physical call.

Relative adjustment uses get_observe_view in CURRENT OPTICAL camera axes:
+X right, +Y down, +Z forward; roll/pitch/yaw radians. Up/right 2cm is
{dx:.02,dy:-.02,dz:0,roll:0,pitch:0,yaw:0}. Send target_joints to move_to only
when success,moved_target,endpoint_safe are true. The master checks the full path.
Do not send unchanged fallback joints or bypass unknown/empty-hand checks.


## Visual evidence versus memory

Missing entries in get_spatial_context/get_manipulation_memory mean UNKNOWN, not
visually absent. Empty cube IDs mean outside bounds. For current visual evidence,
capture and bind that exact image into verify_presence, then ground/segment on
the same observation. Inference errors/timeouts/uncertain text are UNKNOWN, never
ABSENT. A successful image-bound PRESENT can coexist with an empty memory.
Inspect list_functions.input_provenance; live mode refuses capture/inference for
replay, duplicate or unverified raw publishers. Do not infer source type from the
word 'perception' or from timestamps alone. Follow the live-perception resource's
minimal tool allowlist and 180s client timeout guidance.
