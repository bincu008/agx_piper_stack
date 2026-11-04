# Mock MCP integration validation — 2026-11-02

This is real ROS replay + real RoboBrain/EdgeSAM/HGGD inference and planning,
with **no physical actuation or physics emulation**. Mock scene 1 is the packaged
`snapshots/current` capture. All development ROS nodes run on isolated domain 89;
no camera/arm driver or controller was started. Physical execution is false.
The authenticated private MCP endpoint now reads this mock domain.

## Verified results

- All seven exposed MCP tools returned typed structured results against actual
  ROS services, including a prepared bowl session, destination slots and the
  queued tasks. Initial missing-plan failure was also preserved as a failure.
- Bowl: full calculation + cached handoff succeeded twice, pausing before `look`
  and every physical stage. Runs: 23.286 s and 19.690 s.
  Final prepare_pick_place: 2.997 s.
  Result: `PAUSED_AT_STEP_BOUNDARY`, phase `PREPARED`, physical gate false.
  Planned pending stages: look, pregrasp, gripper_open, approach, gripper_close,
  lift, transfer, descent, release, withdrawal, retreat (plus queued slot commit
  and post-retreat observation). None of those physical APIs was dispatched.
- Actual perception/calculation calls used the existing ROS recipe; these have
  NOT been added as MCP tools. Remote mutation tools remain blocked.
- The cup passed perception, HGGD and placement candidate search, but its full
  route did NOT pass (43.945 s). Candidate failures include
  lift IK, bounded APF escape and placement/transfer reachability. This is an
  OPEN motion-planning blocker, not successful placement or a transport failure.
- 33 isolated regressions passed: 18 MCP fixtures plus 15 placement callback/
  visualization regressions. task_pipeline and placement_node builds passed.

## Bugs found and fixed

1. `prepare_held_scene` referenced point_cloud2 without importing it after module
   consolidation. The callback crashed the placement process. Added a local ROS
   cloud-helper import; an organized source/destination regression now exercises
   payload, environment and support-cloud conversion. Actual retry reached
   placement candidate search instead of losing the backend.
2. Bottom-completion debug code published completed_message twice, including an
   unconditional use when the preview publisher was disabled. Removed the
   redundant publish. Regression verifies the no-preview branch retains its
   bottom cache; final replay emits a normal footprint result without that error.

## Final bowl API runtime (seconds, inclusive scopes)

Child backend/engine times overlap these rows; do not add overlapping scopes.

| # | Actual public ROS API | Runtime s |
|---|---|---:|
| 1 | `capture_observation` | 0.144 |
| 2 | `ensure_scene_memory` | 0.012 |
| 3 | `require_target_view` | 0.084 |
| 4 | `capture_observation` | 1.186 |
| 5 | `require_target_view` | 0.061 |
| 6 | `pin_grasp_observation` | 0.002 |
| 7 | `pin_placement_observation` | 0.002 |
| 8 | `refresh_active_target` | 2.148 |
| 9 | `prepare_grasp_input` | 0.004 |
| 10 | `calculate_active_grasps` | 10.754 |
| 11 | `update_object` | 0.002 |
| 12 | `select_grasp` | 0.004 |
| 13 | `cache_grasp_plan` | 0.007 |
| 14 | `resolve_target` | 0.135 |
| 15 | `find_active_placements` | 1.336 |
| 16 | `plan_next_best_view` | 0.266 |
| 17 | `prepare_pick_place` | 2.997 |
| 18 | `get_execution_state` | 0.002 |
| 19 | `validate_prepared_pick_place` | 0.014 |

The final run reuses the valid destination/memory inventory. These two runs are
not a reliability benchmark and do not validate feedback tracking or real contact.
Configured lift/withdrawal and final-approach collision bypasses remain active and
explicit in the result. Successful preparation is not fully scene-collision-
validated motion or permission to execute on hardware.

## Reproduce without motion

In separate terminals, source ROS and the workspace, then use domain 89:

```bash
export ROS_DOMAIN_ID=89
ros2 launch mock_hardware_node mock_hardware.launch.py
```

```bash
export ROS_DOMAIN_ID=89
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio grasp_candidate_backend:=hggd \
  placement_backend:=heightmap pipeline_profiling:=true \
  allow_physical_grasp_execution:=false
```

```bash
export ROS_DOMAIN_ID=89
ros2 run task_pipeline pipeline_task \
  --object 'small white bowl at bottom left, grasp its rim' \
  --destination 'dark rectangular tray' --queue-pick-place --lift-distance 0.05 \
  --output "$HOME/agx_piper_data/reports/mcp_mock_bowl_pipeline.json"
```

The queue must pause before `look`; do not run cached physical execution. Use the
MCP startup in README with ROS_DOMAIN_ID=89, then read the seven tools. For
get_step_context use the returned context_id and include_results=false; image/
cloud CDR payloads may exceed the transport's intentional 1-MiB response bound.

Reports remain outside source:
`~/agx_piper_data/reports/mcp_mock_validation.json`,
`mcp_mock_bowl_final.json`, `mcp_mock_cup_pipeline_retry.json`.
Logs: `~/agx_piper_data/logs/mcp_mock_*.log`, `mcp_held_scene_regression.log`,
`mcp_mock_regressions.log`. Prepared plans expire normally; this recorded success
is not a current executable readiness token. Native macOS connectivity is still
for the Nanobot agent/operator to verify.

## MCP 3.0.0 agent-composed pipeline (2026-11-03)

On isolated ROS domain 89, recorded scene 1 and physical gate false, a client used
TLS-authenticated MCP calls for the complete lower-left rim-bowl preparation:
capture/pin → RoboBrain presence+grounding → EdgeSAM → new store_segmented_target
for bowl and tray → HGGD → object registration/ranking/cache → floor-slot search →
view plan → complete pick/place preparation → cached handoff validation. No arm or
gripper commands were sent. No simulated inference replaced the live RoboBrain
or HGGD calls. Raw camera inputs were recorded mock RGB-D/joint feedback.

The measured 26-call workflow took about 22.0s including inspection. HGGD 10.84s,
placement search 0.94s, complete preparation 2.63s, final validation 0.05s. It selected
one of three tried grasp candidates. Final status included ALL_MOTION_PLANNED,
LIFT_TRANSFER_DESCENT_VALIDATED, OBJECT_UPRIGHT_VALIDATED and
CACHED_HANDOFF_VALIDATED_NO_ACTUATION. Existing explicit lift/withdrawal/approach
bypasses were reported unchanged; this is not complete collision validation.

Evidence: ~/agx_piper_data/reports/mcp_agent_pipeline_validation.json,
mcp_tool_catalog.json, and logs/mcp_agent_pipeline_validation.log. These are
calculation results, not actuator/contact/tracking verification. The external
Nanobot native client still needs its own adaptation and verification.

Final repeated run after the admission checks: **20.23s / 26 calls**, HGGD 11.03s,
placement search .80s, full preparation 2.75s. Package tests reported task_pipeline
148 and master_node 222 tests, zero errors/failures, one existing master skip.
Real TLS calculation queue also completed across HTTP disconnect/reconnect with
same-ID duplicate receipts and stale-revision rejection. Generic dispatch rejected
physical function injection before any backend call. Summary:
~/agx_piper_data/reports/mcp_integration_summary.json.

## Shared spatial map — MCP 3.1.0 (2026-11-03)

Mock scene 1/domain 89, physical gate false: authenticated MCP returned all 16
30cm cells and eight currently observed cells. get_spatial_context mapped the
remembered lower-left bowl to x0_y2_z0; check_target_view returned READY with the
same target_cell_id. plan_next_best_view rejected one unreachable cube target,
then produced NEXT_VIEW_PLANNED_EMPTY_HAND for x1_y1_z0. No movement was sent.

Builds passed; package tests: vision_node 15, task_pipeline 148, master_node 224,
zero errors/failures, one existing skip. Regression cases cover observed empty
space, occluded cells, invalid rays, minimum sample count, boundary hits and map
membership reuse. The new ros-pipeline://spatial-map resource and 62-tool catalog
were verified over TLS. Evidence: ~/agx_piper_data/reports/spatial_map_validation.json
and spatial_integration_summary.json. This does not validate actual camera motion,
physical tracking or complete coverage of any cube.
