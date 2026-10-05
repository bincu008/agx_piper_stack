# Multi-object placement and grasp validation

This update supersedes the earlier footprint-only tuning result. The earlier plate grasp is now rejected; the old successful layout must not be treated as a validated grasp-and-placement result under the current checks.

## Templates

Use the existing action-specific template for one or many objects:

```text
[PLAN_UPRIGHT] place the cup and the white plate onto the tray
[PLAN_UPRIGHT] place cup, white plate, and bowl onto tray
[PLAN] place cup then plate then bowl into tray
```

Two or more objects automatically route to ordered layout calculation. Single-object behavior is preserved. Explicit `[PLAN_LAYOUT_UPRIGHT]` and `[PLAN_LAYOUT]` aliases remain available. Lists accept `then`, commas and `and`, and preserve their written order. The service uses arrays rather than a fixed two-object limit. Three-object parsing and persistent visualization are tested; the current live scene contains only the two requested objects.

## Grasp finding

The earlier tuning changed placement ranking and placement IK accuracy, not the pre-grasp refiner or final grasp approach. However, the existing layout analytical fallback accepted a generic PCA extent pair for the plate without applying the strict pad-normal check to that generic skill. Two points inside the jaws and collision freedom do not establish surface alignment.

Analytical candidates now require measured opposed contact normals aligned with the closing axis using the existing `antipodal_normal_cosine` parameter (default 0.90). The actual FK closing axis is checked after IK as well. Missing or misaligned normals reject the candidate. HGGD refinement, bounded final grasp approach, execution gates, force/speed settings and calibration remain unchanged.

In the live scene, HGGD produced no valid plate grasp. Of the 100 analytical proposals, 72 failed pad-surface alignment and 28 failed jaw/contact-pair consistency. No plate placement is accepted. A better observation of usable opposed contact surfaces or a separately validated plate grasp strategy is required to complete this layout. The checks were not relaxed.

## Fixed attachment and scene validation

The grasp-to-object transform remains fixed:

```text
object_delta = actual_placement_EE * inverse(original_grasp_EE)
placed_point = object_delta * source_point
```

The algorithm evaluates packing poses against that fixed grasp. If multiple already-validated grasp candidates are available, it may try another whole fixed grasp; it never changes the attachment while evaluating a placement.

Layout requests now explicitly carry the source object and complete immutable collision scene in `my_actions/PlanPlacement`. The scene retains surroundings and future source objects, removes the current picked source, and inserts each previously placed object at its actual accepted IK pose. Tray-floor support samples have explicit labels; placed object bottoms remain blocking. Scene timestamps and frames must match the original grasp observation.

Checks include robot self/environment collision, explicit finger boxes at the planned jaw opening, held-object collision against scene and arm, and 21 samples of the preplace-to-place joint interpolation. Bottom contact is allowed only against named tray-floor samples at the final sample. This is calculation-only pose and sampled-approach validation, not validation of a complete execution/transfer trajectory.

## RViz

The existing `/placement/placement_candidates` MarkerArray retains each accepted object, sequence label, two actual IK placement fingers and approach arrow. Finger geometry comes from VAMP configuration, rather than duplicated constants. `/vamp_planner/placement_plan_markers` also shows the current end-effector axes and fingers.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch vamp_node grasp_debug_rviz.launch.py
```

Enable Ordered Placement Layout, or add the MarkerArray topic with Reliable/Transient Local QoS and fixed frame `base_link`. The final running stack retains only the validated cup placement. The plate source observation remains available for grasp inspection.

## Live result and timing

Input: `[PLAN_UPRIGHT] place the cup and the white plate onto the tray`

| Measurement | Seconds |
| --- | ---: |
| Whole request, ending in plate rejection | 125.229 |
| Three live RoboBrain requests | 68.496 |
| Cup grasp calculation | 3.651 |
| Cup packing | 4.295 |
| Cup placement/approach validation | 40.912 |

RoboBrain returned tray `[436,77,976,712]`, cup `[200,21,400,266]`, white plate `[72,336,404,677]`, all normalized 0–1000.

The 16-candidate budget yielded no feasible cup placement under the stricter checks. A 64-candidate budget found candidate #63: actual footprint clearance 2.31 mm, nearest corner distance 49.89 mm, EE position error 3.88 mm and orientation error 1.47 degrees. It favors an edge while allowing the gripper and attached object to clear the scene. Many closer corner poses failed the held-object descent check. The default layout candidate budget is now 64.

The overall result is `LAYOUT_PLAN_FAILED`, because the plate has no valid grasp. This is not a completed two-object placement plan. No arm or gripper motion was commanded.

## Build and launch

The updated service definition requires rebuilding the affected consumers:

```bash
export ONNXRUNTIME_ROOT="$HOME/.local/onnxruntime-1.15.1"
colcon build --symlink-install --packages-select \
  my_actions placement_node vamp_node vlm_node master_node
source install/setup.bash
ros2 launch master_node reasoning_stack.launch.py \
  allow_physical_grasp_execution:=false \
  layout_candidate_count:=64 \
  placement_position_tolerance_m:=0.005
```

Relevant AnyPlace parameters retain the 5 mm packing grid, 15 degree yaw step and adjacency weight 0.004. Its configurable request timeout covers the fuller placement validation, and VLM scales its layout-service wait with object count. Network endpoint/model/authentication remain unchanged.

## Modified files and checks

- `src/vlm_node/include/vlm_node/layout_prompt.hpp`, `src/vlm_node/src/vlm_reasoning_node.cpp`: arbitrary ordered lists and automatic action-template routing.
- `src/my_actions/srv/PlanPlacement.srv`: immutable layout inputs and configured placement-finger output geometry; old single-object callers retain defaults.
- `src/vamp_node/include/vamp_node/placement_geometry.hpp`, `src/vamp_node/src/vamp_planner_server.cpp`: analytical pad alignment, attached-cloud/finger/arm checks, sampled descent and rejection diagnostics.
- `src/placement_node/placement_node/node.py`, `src/placement_node/placement_node/transforms.py`: actual fixed-attachment transforms, sequential collision scenes, ranked grasp alternatives and persistent placement grippers.
- `src/master_node/launch/reasoning_stack.launch.py`: 64-candidate default.
- Focused tests: `src/vlm_node/test/test_layout_prompt.cpp`, `src/vamp_node/test/test_placement_geometry.cpp`, `src/placement_node/test/test_layout_debug.py`; CMake test registrations.

Affected builds passed. Tests: AnyPlace 47, VAMP 28, VLM 5 and master functional 19, all passing. Python compilation and targeted whitespace checks passed. A live timestamp-mismatch service test returned `PLACEMENT_LAYOUT_OBSERVATION_MISMATCH` as expected. Full live grounding and the stricter pipeline were exercised with execution disabled.

Artifacts under `/home/parallels/layout_tuning`: `multi_object_summary.json`, `multi_object_final_live.json`, `multi_object_final_live/`, `multi_object_final_markers.cdr`, and build/test logs. The remaining blocker is a valid plate grasp, not connectivity or layout template support.
