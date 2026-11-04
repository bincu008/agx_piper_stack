# Placement API coverage audit — 2026-11-03

**Resolved in MCP 3.0.0:** all preparation stages listed below are callable via
named tools and master-owned result bindings. store_segmented_target now supplies
the missing mask-to-memory extraction API. The complete mock bowl pipeline passed
through authenticated TLS MCP without actuator calls; see AGENT_PIPELINE.md and
MOCK_VALIDATION.md. Generic queues remain calculation-only, physical stages are
called individually through existing guards.

The remainder is the historical 2.1.0 audit, retained as evidence of the gap.

Checked the running MCP 2.1.0 tool list, authoritative ROS registry, recipes,
active manipulation memory and prepared executor. No perception inference,
planning or physical operation was executed in this audit.

**The external agent cannot yet perform autonomous end-to-end pick/place through
MCP.** There are 18 MCP tools and 97 catalogued ROS service functions, plus the
ROS run_steps action. list_functions.ros_catalog lets the agent read ROS API names
and field definitions, but does not authorize invoking them as MCP tools.
Most preparation is still local/ROS-only; the execution side is already exposed.

## Requested sequence mapped to actual APIs

"ROS only" means the function exists but is not an MCP tool. Alternatives in a
row are different supported routes; do not call all of them redundantly.

| Stage | Existing public ROS API | MCP availability |
|---|---|---|
| Capture synchronized RGB-D/manual transform | capture_observation | ROS only |
| Preserve the shared source frame | pin_grasp_observation, pin_placement_observation | ROS only |
| Ground source object | verify_presence -> ground_object | ROS only |
| Segment source pixels | segment_object (returns a 2D mask, not a 3D cloud) | ROS only |
| Extract source cloud and store object memory | resolve_target(role=object), refresh_active_target; combined inference/extraction/storage | ROS only; no standalone existing-mask insertion API |
| Ground destination | verify_presence -> ground_object, destination description | ROS only |
| Segment destination pixels | segment_object | ROS only |
| Extract/store destination cloud and persistent grid owner | resolve_target(role=destination) | ROS only; combined route |
| Read active target geometry | get_active_target | ROS only; get_manipulation_memory is only a small metadata snapshot |
| Prepare and calculate grasps | prepare_grasp_input -> calculate_active_grasps (or calculate_grasps) | ROS only |
| Register, provisionally rank and cache grasp pool | update_object, select_grasp, cache_grasp_plan; get_cached_grasp reads it | ROS only |
| Build grid, match footprint, find slot candidates | find_active_placements (or find_placements) | ROS only; get_floor_slots only reads existing grid |
| Validate whole task and choose a complete safe candidate | prepare_pick_place | ROS only |
| Read selected prepared pose and validate handoff | get_prepared_pick_place, validate_prepared_pick_place | ROS only as tools; gateway internally calls validation when admitting execution |
| Optional semantic look/view planning | check_target_view, plan_next_best_view | ROS only; get_observe_view/move_to support relative empty-hand viewing instead |
| Move to prepared pre-grasp | execute_plan(segment=pregrasp) | Exposed |
| Open gripper before approach | gripper_open | Exposed |
| Approach final grasp | execute_plan(segment=approach) | Exposed |
| Close and verify contact | gripper_close (verification included) | Exposed |
| Lift | execute_plan(segment=lift) | Exposed |
| Transfer above chosen slot | execute_placement_transfer | Exposed |
| Descend to placement | execute_placement_descent | Exposed |
| Release | release_placed_object | Exposed |
| Commit occupied slots | commit_floor_slots | Exposed; after release, frozen selected pose only |
| Withdraw released gripper | execute_withdrawal | Exposed |
| Retreat | execute_placement_retreat | Exposed |
| Inspect completion | get_execution_state; get_step_context for an existing queue | Exposed reads |

Batch perception is another existing ROS-only route: ensure_scene_memory /
perceive_scene can populate multiple objects plus the destination. resolve_target
may reuse valid cached data or capture/ground/segment when necessary. These are
conveniences over the same memory, not an explicit "insert this existing mask"
contract. Avoid duplicated inference if exposing both routes.

## Internal stages do not all require separate agent calls

prepare_pick_place already invokes prepare_held_scene, plan_final_approach,
plan_lift, plan_placement and plan_motion; its retreat path uses plan_lift in
empty-hand mode and plan_camera_retreat. These have catalogued ROS names but are
not individually MCP tools. Exposing the complete preparation operation closes
this dependency without making the agent coordinate every IK/waypoint iteration.
The master still validates the whole approach/lift/transfer/descent/withdrawal/
retreat before accepting a plan. It tries retained grasp candidates within its
bounds. select_grasp is preliminary ranking, not complete pick/place validation.

The standard recipe adds gripper_open, slot commit and short withdrawal to the
user's sequence. Contact verification belongs to prepared gripper_close; do not
mix the separate legacy verify_grasp_contact phase record into this path.
After confirmed release, the shared API result hook updates the destination's
placed geometry; successful slot commit synchronizes reservation metadata.
Do not expose a cloud-merge command that can pretend a failed placement succeeded.

## Actual blockers for the outside agent

1. It cannot obtain/correlate capture, mask, source/destination geometry and grasp
   results through callable MCP tools. Relative-view calculation is not object
   perception, and metadata inspection does not insert geometry.
2. It cannot create source/candidate caches, build placement candidates or call
   prepare_pick_place. get_execution_state only reads a plan created elsewhere.
   Without that plan, execution_ref cannot authorize a manipulation sequence.
3. There is no callable MCP submit_steps/run_step/run_steps. Existing ROS queue
   insertion/action execution works locally, but the agent cannot construct a
   master-owned pipeline remotely through the current adapter.
4. For explicit ground -> segment -> extract/store composition, a public API
   accepting the existing synchronized mask/observation and semantic role is
   missing. The private grounded_mask argument is not a ROS/MCP input.

## Next implementation scope

Expose the public perception/memory/grasp/search/preparation operations above,
plus guarded master pipeline submission/completion, through the same registry.
Keep images/clouds/CDRs on Ubuntu in bounded StepStore results; return handles and
bind later inputs to earlier outputs in the SAME context. Do not stream point
arrays or image buffers through an LLM to assemble requests.

Provide an explicit validated mask-to-memory operation when the agent has already
performed grounding/segmentation. Preserve name, role, source header, calibration,
manual transform, observation/mask association, memory revision and destination
inventory. A supplied arbitrary cloud is not automatically trusted perception.

Queue execution must use the same durable admission, contact/order, physical gate,
unknown-state and no-replay checks as the exposed action tools. A generic run_step
that accepts an arbitrary function/CDR would otherwise bypass the MCP guard. Keep
legacy/engine/reset operations discoverable as documentation where appropriate;
do not silently make every catalogue entry an unrestricted callable endpoint.

Evidence: ~/agx_piper_data/reports/placement_mcp_api_audit.json. This audit adds no
new callable APIs, changes no safety checks, and is not physical validation.
