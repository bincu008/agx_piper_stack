> Historical tuning result. Superseded by [multi-object grasp and placement validation](MULTI_OBJECT_PLACEMENT.md); the plate grasp is rejected by the current checks.

# Sequential layout tuning

Command: `[PLAN_LAYOUT_UPRIGHT] place the cup then the white plate onto the tray`

The live mock-scene pipeline succeeded with RoboBrain at `10.10.10.2:18091`. Physical execution was disabled throughout. The accepted cup is near a tray corner; the plate follows beside the cup near the edge. Both placements passed IK and collision validation. The plate used the existing analytical grasp fallback after HGGD found no collision-valid grasp.

## Measured timing

| Stage | Seconds |
| --- | ---: |
| Complete text-to-layout pipeline | 86.170 |
| Three live RoboBrain grounding requests | 68.592 |
| Grasp and placement calculation | 15.542 |
| Cup packing | 5.120 |
| Plate packing | 2.100 |

The final visualization replay reproduced the same candidates and centers in 16.389 seconds using segmentation reused only after exact image and organized-cloud comparison.

## Result quality

Metrics use full object geometry transformed by the actual accepted IK pose, rather than just requested centers. Cup: 7.66 mm from the nearest tray corner, 1.78 mm minimum boundary clearance. Plate: 7.21 mm minimum boundary clearance. Nearest projected XY gap between objects: 5.67 mm. These are mock-scene calculations, not measured physical placement accuracy. The requested footprint clearance is 2 mm; the resolved cup pose is slightly closer due to allowed IK error.

This is the best validated configuration among the tested sampling budgets, not a proof of a continuous global optimum. A 3 mm grid took 36.27 seconds for little placement improvement; the chosen 5 mm grid and 16 candidates provide the better tested tradeoff.

## Configuration and launch

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch master_node reasoning_stack.launch.py \
  allow_physical_grasp_execution:=false \
  layout_candidate_count:=16 \
  placement_position_tolerance_m:=0.005
```

AnyPlace parameters: `packing_grid_resolution_m=0.005`, `packing_upright_yaw_step_deg=15.0`, `packing_adjacency_weight=0.004`. Optional `layout_debug_directory` saves inspection artifacts. RoboBrain endpoint, model and timeout retain their configurable parameters and optimized defaults.

RViz: use the updated `ros2 launch vamp_node grasp_debug_rviz.launch.py`, or add MarkerArray `/placement/placement_candidates` with Transient Local durability. Green `#1 cup` and cyan `#2 white plate` show the accepted ordered layout. The current stack is left running with the final result.

## Changes

- `src/placement_node/placement_node/packing.py`: complete footprints, exact metric tray containment, measured support plane, rim-aware adjacency, efficient containment prefilter and configurable adjacency weight.
- `src/placement_node/placement_node/node.py`: 16-candidate layout budget, stage timings, optional debug capture and persistent ordered layout markers/current source contours.
- `src/vamp_node/src/vamp_planner_server.cpp`: bounded exact-observation retention, upright IK tolerance and packing-first ranking.
- `src/vamp_node/include/vamp_node/placement_ranking.hpp`: ranking helper.
- `src/vlm_node/src/vlm_reasoning_node.cpp`: pin the original observation in both calculation nodes before long grounding requests; accurate request-budget logging.
- `src/master_node/launch/reasoning_stack.launch.py`: layout candidate and placement position tolerance arguments.
- `src/vamp_node/rviz/grasp_debug.rviz`: ordered layout display.
- Focused tests in AnyPlace and VAMP, with their CMake registration.

## Validation

Affected packages built successfully. AnyPlace: 45 tests; VAMP: 23 tests; VLM: 2 tests; master functional tests: 19. Existing master lint findings remain outside this tuning scope. Live grounding and full calculation succeeded. No hardware execution validation was performed.

Artifacts: `/home/parallels/layout_tuning/summary.json`, `final_full_pipeline.json`, and `final_full_pipeline/` containing raw image, top-down plot, serialized service exchanges, object geometry and quality metrics.
