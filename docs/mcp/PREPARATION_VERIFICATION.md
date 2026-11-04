# Per-tool functional verification — preparation-only mode

Live-origin immutable observations; no actuation. Tests were not a performance benchmark.
A failed or blocked call is not positive functional verification of the success path.

| Tool | Successful call observed | Actual statuses |
|---|---|---|
| cache_grasp_plan | Yes | GRASP_PLAN_CACHED_IN_MASTER |
| calculate_active_grasps | Yes | FEWER_THAN_MAXIMUM_VALID |
| calculate_grasps | Yes | FEWER_THAN_MAXIMUM_VALID |
| capture_observation | Yes | SYNCHRONIZED_OBSERVATION_READY |
| check_target_view | Yes | READY |
| ensure_scene_memory | Yes | Extra data: line 1 column 82 (char 81); SCENE_MEMORY_READY_NO_INFERENCE |
| find_active_placements | Yes | CANDIDATES_READY |
| find_placements | Yes | OK backend=heightmap policy=upright object=light-colored mug with a handle destination=dark rectangular tray support_mode=floor_slots candidates=16 object_points=9144 additional_obstacles=0 bottom_completion=no support_aligned=16/16 support_shift_mm=integrated_heightmap destination_points=48276 planning=0.59s best_score=-0.0010 packing={orientations:72,tested:432,contained:432,stable:313,support_tilt_deg:1.0,support_contacts_rejected:2,support_area_rejected:6,support_com_rejected:111} |
| get_active_target | Yes | ACTIVE_TARGET_REQUIRED; ACTIVE_TARGET_SNAPSHOT_READY |
| get_arm_state | Yes | ARM_STATE_READY |
| get_cached_grasp | Yes | NO_FRESH_CACHED_GRASP; CACHED_GRASP_READY |
| get_execution_state | No — failed/blocked | NO_PREPARED_GRASP_LIFT |
| get_floor_slots | Yes | FLOOR_SLOT_MEMORY_READY |
| get_manipulation_memory | Yes | MANIPULATION_MEMORY_READY |
| get_observe_view | No — failed/blocked | STALE_VIEW_FEEDBACK |
| get_prepared_grasp_lift | No — failed/blocked | NO_PREPARED_GRASP_LIFT |
| get_prepared_pick_place | No — failed/blocked | NO_PREPARED_GRASP_LIFT |
| get_spatial_context | Yes | SPATIAL_CONTEXT_READY |
| get_spatial_map | Yes | SPATIAL_MAP_SNAPSHOT |
| get_step_context | Yes | CONTEXT_READY |
| get_world_state | Yes | response |
| ground_object | Yes | INVALID_BINDING_FIELD: __class__; BINDING_TYPE_MISMATCH: image; GROUNDING_READY |
| list_functions | Yes | preparation_only |
| perceive_scene | No — failed/blocked | Extra data: line 1 column 82 (char 81) |
| pin_grasp_observation | Yes | LAYOUT_OBSERVATION_PINNED |
| pin_placement_observation | Yes | LAYOUT_OBSERVATION_PINNED |
| plan_camera_retreat | Yes | CLEARANCE_RETREAT_VALIDATED |
| plan_final_approach | Yes | OK;NEAR_GRASP_USED;FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED |
| plan_lift | No — failed/blocked | LIFT_IK_FAILED_AT_WAYPOINT_10;LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED;LIFT_ROBOT_ENVIRONMENT_CHECK_BYPASSED |
| plan_motion | Yes | OK |
| plan_next_best_view | Yes | CURRENT_VIEW_ALREADY_SUITABLE |
| plan_placement | No — failed/blocked | PLACEMENT_NO_REACHABLE_CANDIDATE tested=10 rejected={pose:0,ik:4,upright_tilt:20,collision:1,preplace:5,static_collision:0,finger_collision:0,attached_collision:1,object_arm_collision:0,descent_arm_collision:0,final_pose_collision:1,descent_ik:0,descent_continuity:0} |
| prepare_grasp_input | Yes | GRASP_INPUT_READY |
| prepare_grasp_lift | No — failed/blocked | NO_FRESH_CACHED_GRASP; STALE_HARDWARE_FEEDBACK: joint_feedback_time |
| prepare_held_scene | Yes | IMMUTABLE_HELD_SCENE_READY |
| prepare_pick_place | No — failed/blocked | STALE_HARDWARE_FEEDBACK: joint_feedback_time |
| refine_grasp | Yes | OK |
| refresh_active_target | Yes | ACTIVE_TARGET_GROUNDED |
| require_target_view | Yes | READY |
| resolve_target | Yes | ACTIVE_TARGET_GROUNDED |
| run_step | Yes | ACTIVE_TARGET_SNAPSHOT_READY; DEPENDENCY_FAILED: not_yet_calculated; MCP_ERROR |
| run_steps | Yes | RUNNING; COMPLETE |
| search_placements | No — failed/blocked | NO_STABLE_PACKING_CANDIDATE |
| segment_object | Yes | OK |
| select_grasp | Yes | RANKED_GRASP_SELECTED |
| solve_pose | Yes | OK |
| store_segmented_target | Yes | ACTIVE_TARGET_GROUNDED; ACTIVE_TARGET_REUSED |
| submit_steps | Yes | MICROSTEPS_QUEUED; MCP_ERROR |
| update_object | Yes | UPDATED obj_001 |
| validate_placements | No — failed/blocked | PLACEMENT_NO_REACHABLE_CANDIDATE tested=10 rejected={pose:0,ik:4,upright_tilt:20,collision:1,preplace:5,static_collision:0,finger_collision:0,attached_collision:1,object_arm_collision:0,descent_arm_collision:0,final_pose_collision:1,descent_ik:0,descent_continuity:0} |
| validate_prepared_grasp_lift | No — failed/blocked | NO_PREPARED_GRASP_LIFT |
| validate_prepared_pick_place | No — failed/blocked | NO_PREPARED_GRASP_LIFT |
| verify_presence | Yes | PRESENT |

Failures of complete preparation/getters/validators require fresh arm feedback and a
successful full preparation before their success paths can be verified.
Details: ~/agx_piper_data/reports/preparation_mode_summary.json and preparation_mode_*.json.
