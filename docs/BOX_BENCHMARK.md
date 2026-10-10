# Box planning benchmark

This is a calculation-only controller launched beside the mock and reasoning
stack. The [README](../README.md#ten-box-planning-benchmark) gives the complete
terminal-by-terminal setup; [CODE_MAP.md](CODE_MAP.md) maps the implementation.

## Inputs and calculation

`box_benchmark.py` captures an exact RGB/cloud/manual-transform observation via
`/pipeline/capture_observation`. It creates complete six-face 8 cm box clouds on
the base supporting plane, within 50 cm of `base_link`, separated by a 4 cm gap
by default. The left side of the arm corresponds to positive base Y. The fixture
adds known supporting floor patches and replaces noise only under those patches;
it retains surrounding geometry for collision checks.

It uses the configured real VLM server for tray presence/grounding, then EdgeSAM
for the tray silhouette. Only measured points inside that mask form the tray.
It does not invent tray geometry if inference/segmentation fails.

For each box, the controller calls the same registered functions as deployment:

1. `calculate_cloud_grasps`: known flat-wall contact seeds, existing IK/collision
   validation and ranking. Synthetic objects skip RGB grounding and HGGD.
2. `search_placements`: the deterministic height-map engine, initially edge/floor
   search; optional refined floor retry, then stacking on a named accepted top.
3. `validate_placements`: shared batching/retry policy and VAMP attachment/IK/
   collision validation. Full-motion mode also validates lift, transfer and descent.
4. Commit the accepted rigid cloud transform to the virtual scene and publish it.

Each accepted object keeps its grasp-to-object attachment for motion calculation.
A failed object remains a source obstacle, cannot become a stack support, and is
skipped so later boxes still run. The environment remains part of collision
calculation even when RViz displays only boxes.

This is not physics simulation: it does not execute arm/gripper commands,
verify a physical pickup, model contact dynamics or validate real grasp refinement.
Known-cloud timing is not a deployed HGGD pipeline timing.

## Launch and control

Start one mock and one safe reasoning stack as documented in README. Source
ROS and `install/setup.bash` in every terminal. Then:

```bash
ros2 launch placement_node box_benchmark.launch.py \
  box_count:=10 optimize_pipeline:=true plan_pick_place_motion:=true \
  output_directory:="$HOME/box_benchmark/run_01"
```

From another terminal:

```bash
ros2 service call /box_benchmark/run std_srvs/srv/Trigger '{}'
```

The service waits for worker exit and returns JSON counts/time/failure details.
Success is true only for `COMPLETE`; partial object failure reports
`COMPLETE_WITH_FAILURES`. Shared setup failures stop the run. Another run while
busy returns `BENCHMARK_BUSY`. Use `auto_run:=true` to trigger at startup.

```bash
ros2 topic echo /box_benchmark/status
ros2 service call /box_benchmark/cancel std_srvs/srv/Trigger '{}'
ros2 service call /box_benchmark/reset std_srvs/srv/Trigger '{}'
```

Cancellation is checked between calculations; it cannot stop an already-running
backend ROS service. Reset clears placement markers while idle. Parameters cannot
be changed during a run. The benchmark retains the latest run's master context
IDs in its report and releases them before rerunning.

## Search profiles

| Setting | Default | Effect |
|---|---|---|
| `box_count` | 10 | Number of known synthetic boxes |
| `box_gap_m` | 0.04 | Source separation |
| `placement_policy` | `edge_then_stack` | Prefer floor/edge, then universal heightmap stacking |
| `stack_minimum_coverage` | `0.6` | Minimum projected footprint fraction supported at the stacking height |
| `optimize_pipeline` | true | Stop grasp/search work once a useful set is available |
| `grasp_early_stop_count` | 4 | Initial validated grasp target |
| `placement_early_stop_count` | 32 | Initial packing candidate target |
| `placement_validation_batch_size` | 8 | Shared validation batch size |
| `candidate_count` / `yaw_step_deg` | 128 / 15° | Primary search limits |
| `floor_refinement_candidate_count` | 0 | Fast default; no fine floor retry |
| `floor_refinement_yaw_step_deg` | 5° | Yaw sampling when floor refinement is enabled |
| `plan_pick_place_motion` | true | Require lift/transfer/descent validation |
| `output_directory` | `~/box_benchmark` | Generated evidence; reruns overwrite files |

To retain refined floor search:

```bash
ros2 launch placement_node box_benchmark.launch.py \
  floor_refinement_candidate_count:=384 floor_refinement_yaw_step_deg:=5.0 \
  output_directory:="$HOME/box_benchmark/refined_01"
```

Use one benchmark node at a time. Early search controls candidate effort; it does
not disable collision checks or change the IK solver. Motion-failure retry policy
is shared with deployed API calls; see [optimization details](BOX_PIPELINE_OPTIMIZATION.md).

## RViz and reports

```bash
ros2 launch vamp_node grasp_debug_rviz.launch.py fixed_frame:=base_link
```

| Display | Topic | Content |
|---|---|---|
| Box Benchmark Tray | `/box_benchmark/tray_cloud` | Grounded, segmented measured tray |
| Box Benchmark Scene | `/box_benchmark/boxes_cloud` | Source/failed/accepted boxes, without environment |
| Box Benchmark Placements | `/box_benchmark/placements` | Ordered accepted placements and gripper geometry |

The ordinary `/placement/destination_cloud` display belongs to the other placement
workflow. Empty results can mean the worker is still grounding; inspect
`/box_benchmark/status` before assuming a visualization problem. Keep the completed
node alive so late RViz subscribers receive transient-local outputs.

`[BOX TIMING]` reports progress. `[BOX SUMMARY]` prints once at worker exit with
per-object grasp, floor search, stack search and total times. Failed attempts are
included. Shared tray setup is measured separately.

The output directory contains `report.json`, `source_image.png`, tray mask/
segmentation images, fixture/destination arrays, accepted `box_XX.npz` clouds and
joint paths, and request/response CDR dumps. These are generated run artifacts;
keep them outside Git. A new output directory preserves each run.

The latest full shared-API run on 2026-10-05 accepted **10/10 boxes: 6 floor and
4 stacked, in 230.08 s**, with all lift/transfer/descent paths validated. Older
capacity/timing figures used different masks, search profiles or API scopes and
should not be treated as current results. No global packing optimum or hardware
feasibility is claimed. [Current API validation](STANDARD_PIPELINE_API.md#timing-and-validation)
records checks and the unresolved brown-cup placement IK limitation.
