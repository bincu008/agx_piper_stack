# anyplace_ros

Isolated, CPU-only ROS 2 evaluation of the AnyPlace low-level placement
network. The node never publishes arm or gripper commands.

Inputs:

- `/camera/color/image_raw`
- `/camera/depth_registered/points` (organized and RGB-registered)
- RoboBrain-compatible `/v1/chat/completions` server for a tight destination box

Outputs:

- `/anyplace/destination_cloud`
- `/anyplace/best_placed_object`
- `/anyplace/completed_object_cloud` (measured object plus synthetic bottom)
- `/anyplace/artificial_bottom_cloud` (synthetic debug points only)
- `/anyplace/placement_candidates`
- `/anyplace/grounding_debug`
- `/anyplace/status`

Placement inference runs in the registered camera-cloud frame. Published
clouds and poses are converted to `base_link` with the synchronized arm joint
feedback and the same manual `FK(joints) * X_matrix` calculation used by the
active-vision stack. No TF lookup or TF-based computation is used.

## Setup and run

```bash
colcon build --packages-select anyplace_ros
source install/setup.bash
ros2 launch master_node reasoning_stack.launch.py \
  api_server_ip:=127.0.0.1 \
  placement_backend:=heightmap \
  allow_physical_grasp_execution:=false
```

In another terminal:

```bash
source install/setup.bash
ros2 launch vamp_node grasp_debug_rviz.launch.py fixed_frame:=base_link
```

The default production stack uses the deterministic height-map packing backend.
The retained AnyPlace diffusion backend can be enabled explicitly after its
checkpoint has been downloaded:

```bash
ros2 run anyplace_ros download_anyplace_checkpoint -- \
  --output /path/to/anyplace_diffusion.pt
ros2 launch master_node reasoning_stack.launch.py \
  placement_backend:=anyplace \
  anyplace_checkpoint_path:=/path/to/anyplace_diffusion.pt \
  allow_physical_grasp_execution:=false
```

The node rejects inference when synchronized joint feedback or the calibrated
X matrix installed from
`src/master_node/config/calibration/hand_eye_calibration.yaml` is unavailable.

When the segmented object and its matching organized source observation are
received, the node fits the source support plane once and fills only the
near-plane object footprint. The artificial bottom is cached and used as the
AnyPlace object input. This stage does not perform placement-height correction
or general collision rejection. For `PLACE_UPRIGHT`, a lightweight post-step
uses the transformed synthetic bottom and a local destination height map to
translate each candidate along `base_link` gravity until the cap rests 2 mm
above the observed support surface. XY and network-predicted orientation are
unchanged. `PLACE_FREE` is not adjusted because its contact face may not be the
original object bottom.

## Integrated grasp-to-placement workflow

The normal reasoning stack caches `/vamp_planner/segmented_target_cloud` after
`[FIND_GRASP]`, then sends an EdgeSAM destination mask to AnyPlace. Both inputs
are already expressed/indexed through the synchronized ActiveVision manual
`FK(joints) * X_matrix` path; no TF lookup is used.

```text
[FIND_GRASP] find grasping pose on the cup
[PLACE_UPRIGHT] place the cup onto the plate
[PLACE_FREE] place the cup onto the plate
[PLAN_UPRIGHT] place the cup onto the plate
[PLAN] place the cup onto the plate
```

The default `placement_backend:=heightmap` bypasses the AnyPlace network.
`PLACE_UPRIGHT` searches yaw while preserving gravity alignment. `PLACE` and
`PLACE_FREE` additionally search bounded roll/pitch tilts. The retained
`placement_backend:=anyplace` mode restores the original diffusion behavior.
The exact destination top-height grid is available as a ROS debug image topic;
normal operation does not write debug images to disk.
Each result appears under `placement_candidate_N` on
`/anyplace/placement_candidates`; the marker contains the transformed object
point cloud rather than a generic box.

`PLAN_UPRIGHT` and `PLAN` continue past packing without moving hardware. They
use the cached final grasp as the rigid object-to-gripper attachment, derive
the required placement end-effector pose for every packing candidate, and run
relaxed placement/pre-place IK plus robot collision validation. The selected
actual (IK-resolved) object cloud is published on
`/vamp_planner/refined_placed_object_cloud`; the placement TCP axes and the
10 cm pre-place approach are published on
`/vamp_planner/placement_plan_markers`. Placement IK defaults to 15 mm position
and 0.35 rad orientation tolerance, and the result remains calculation-only.
