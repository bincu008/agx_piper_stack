# Code map and operating guide

## Runtime topology

```text
Orbbec camera driver                         AGX Piper driver
  RGB + registered depth + camera info         joint/TCP/gripper feedback
                  |                                      |
                  +---------- ActiveVisionServer --------+
                               | synchronized observation
                               | obstacle world
                               v
RoboBrain ---> VLMReasoningNode + EdgeSAM ---> segmented target
                               |                    |
                               |                    v
                               +------------> VampPlannerServer
                                               HGGD proposals
                                               local PCA refinement
                                               IK + collision validation
                                                      |
Height-map / optional AnyPlace node ---> placement ----+
                                                      |
                                                      v
                                              PiperMasterControl
                                                      |
                                              /control/joint_states
```

Computational perception uses only the synchronized manual
`FK(joints) * hand_eye_transform` result from `ActiveVisionServer`. TF is
published only for visualization and must not be used as a computational
fallback.

## Packages

| Package | Responsibility | Primary implementation |
|---|---|---|
| `master_node` | Pipeline state, physical execution gates, pre-grasp/final approach | `master_node/piper_master_control.py` |
| `vision_node` | RGB-D synchronization, gripper mask, obstacle world, observation cache | `src/active_vision_server.cpp` |
| `vlm_node` | RoboBrain requests, grounding parsing, EdgeSAM segmentation, prompt dispatch | `src/vlm_reasoning_node.cpp` |
| `vamp_node` | HGGD, PCA refinement, IK, collision, Cartesian approach and placement validation | `src/vamp_planner_server.cpp` |
| `vamp` (`src/vamp_mvt`) | Generic VAMP planning/collision library and generated Piper sphere/FK model | `src/impl/vamp/` |
| `gpd_vendor` | Pinned legacy GPD fallback library; linked into `vamp_node`, not launched | `CMakeLists.txt`, `config/`, `models/` |
| `anyplace_ros` | Height-map placement proposals and optional AnyPlace CPU inference | `anyplace_ros/node.py`, `packing.py` |
| `my_actions` | Cross-node messages, actions, and services | `msg/`, `srv/` |
| `agx_arm_ctrl` | CAN hardware driver | `agx_arm_ctrl_single_node.py` |
| `orbbec_camera` | Orbbec RGB-D driver | upstream driver with Gemini 330 launch defaults |

Cricket and Foam are retained under `tools/robot_geometry/` only to regenerate
the Piper sphere/FK model. Their unrelated demo robot datasets and generated
build artifacts are intentionally excluded from the baseline.

## Grasp flow

1. `[FIND_GRASP]` grounds and segments the requested object or part.
2. HGGD produces spatial seeds from the organized RGB-D observation.
3. `vamp_node` aligns/refines the seeds, measures jaw width, runs IK and checks
   grasp/pre-grasp collisions.
4. The ranked accepted candidate is cached and published for RViz.
5. `[PRE_GRASP]` moves only when the master execution gate and hardware are
   available, settles the camera, captures a new observation, and refines the
   local contact region.
6. `[GRASP]` prevalidates the bounded Cartesian approach before any command is
   published.

## Placement flow

1. `[PLAN]` or `[PLAN_UPRIGHT]` grounds the destination.
2. `anyplace_ros` extracts the destination cloud and runs the default
   deterministic height-map packing search.
3. `vamp_node` derives the required end-effector pose from the cached grasp
   attachment and tests reachable orientation variants.
4. The accepted placed-object cloud and end-effector marker are published.

## Launch commands

Terminal 1 — camera:

```bash
source install/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py
```

Terminal 2 — Piper hardware driver:

```bash
source install/setup.bash
ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
  can_port:=can0 arm_type:=piper effector_type:=agx_gripper \
  auto_enable:=true control_enabled:=true speed_percent:=20 pub_rate:=50
```

Terminal 3 — safe planning stack:

```bash
source install/setup.bash
ros2 launch master_node reasoning_stack.launch.py \
  api_server_ip:=10.211.55.2 api_server_port:=1234 \
  model_name:=robobrain2.5-4b \
  grasp_candidate_backend:=hggd placement_backend:=heightmap \
  enable_grasp_collision_check:=true \
  allow_pregrasp_collision_override:=false \
  allow_physical_grasp_execution:=false
```

Terminal 4 — RViz:

```bash
source install/setup.bash
ros2 launch vamp_node grasp_debug_rviz.launch.py
```

Set `allow_physical_grasp_execution:=true` only for an explicitly reviewed
physical run. This flag permits motion but does not bypass IK or collision
validation.

## Debug entry points

| Symptom | Start here |
|---|---|
| No synchronized scan | `/active_vision/observation_cloud`, camera and joint timestamps |
| Bad gripper masking | `/active_vision/gripper_self_mask`, `gripper_self_filter.yaml` |
| Incorrect target | EdgeSAM mask and `/vamp_planner/segmented_target_cloud` |
| HGGD has no proposals | `/vamp_planner/hggd_heatmap`, HGGD summary log |
| Refined grasp rejected | per-seed HGGD validation log and accepted candidate marker |
| Placement rejected | destination cloud, support surface and placement validation log |
| Motion unavailable | master execution gate and AGX feedback topics |
