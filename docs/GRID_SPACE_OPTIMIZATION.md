## Default exclusive-grid validation — 2026-10-10

Upright placement now defaults to `floor_slots` in the CLI/recipe, single-object
VLM plan and ordered layout. Slots are **12.5 cm squares with 0.5 cm gaps**;
whole-cell reservations are exclusive. Prefer the smallest validated group,
including the full projected geometry. The 16-candidate limit is the default.
Free-orientation VLM/layout requests retain `auto`; explicit edge/stack/auto
policies and the benchmark's edge-then-stack sequence remain available.

The rebuilt stack completed **cup → lower bowl → upper bowl**, all on the floor,
using three of four slots. Independent geometry checks found each full projected
object inside exactly one square, with no slot reuse; the bowls are adjacent.

| Object | Cell | HGGD + validation (s) | Placement search (s) | Placement motion validation (s) |
| --- | --- | ---: | ---: | ---: |
| Beige cup | 2,1 | 2.230 | 0.419 | 0.252 |
| Lower bowl | 1,1 | 4.033 | 0.450 | 0.412 |
| Upper bowl | 1,0 | 3.289 | 0.300 | 0.219 |

Total request-to-result **20.502 s**, layout calculation **14.599 s**. Two earlier
successful runs took 22.169 and 20.085 s. API and engine scopes overlap; do not
sum these numbers as disjoint pipeline stages. RoboBrain and segmentation were
real calls on mock RGB-D; no synthetic grasp generator was used.

Transfer/descent, robot/finger and object-arm checks remained active. The existing
requested mock diagnostic lift-only bypass was enabled. All saved motion statuses
include `LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED`; this is not a claim of
collision-validated source lift. Physical execution remained false. This run
used the existing 90-degree aligned-grasp yaw test setting.

Cleanup decodes each public search cloud once, retains one centered source for
exclusive pending results, and constructs only the selected occupancy mask at
commit. Idempotent commits retain the original mask geometry. Whole-cell
remaining-space scores exclude consumed cells and are cached per equivalent
cell group across rotations. Removed unused container-helper grid arguments.
A same-cloud microbenchmark measured mask construction **18.04 → 1.20 ms**
(16 masks versus one). Packing-only medians were **378.4 → 374.6 ms**, within run
variation; a substantial overall speedup is not established. Grasp/IK remains
the larger measured cost. The legacy VLM plan now commits the selected grid
pose after validation; this branch was built and unit-checked, while the live
end-to-end validation exercised the ordered layout path.

Checks: **177 placement/pipeline tests**, **7 VLM tests**, VLM build, Python
compilation and whitespace checks passed. Evidence outside Git:
`~/cup_bowls_grid_rebuilt.json`, `~/cup_bowls_grid_rebuilt/`,
`~/cup_bowls_grid_rebuilt_validation.json`, `~/cup_bowls_grid_rebuilt_clouds.npz`,
`~/grid_default_tests.log`, `~/grid_default_vlm_build.log`,
`~/grid_default_vlm_tests.log`, `~/grid_reservation_mask_comparison.json`,
`~/grid_search_cleanup_comparison.json`.

Repeat with the mock in one terminal and the stack in another:

```bash
ros2 launch mock_hardware_node mock_hardware.launch.py
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio aligned_zero_pose_yaw_limit_degrees:=90 \
  placement_bypass_attached_collision_during_lift:=true \
  pipeline_profiling:=true allow_physical_grasp_execution:=false
```

Reset independent trials, then request the ordered layout:

```bash
ros2 service call /pipeline/commit_floor_slots my_actions/srv/CommitFloorSlots '{reset: true}'
ros2 service call /vlm_reasoning_node/chat_prompt my_actions/srv/Chat \
  "{prompt: '[PLAN_UPRIGHT] place beige cup at far left, small white bowl at bottom left, small white bowl in middle left above the other bowl onto dark rectangular tray'}"
```

Start `ros2 launch vamp_node grasp_debug_rviz.launch.py` for grid and selected
poses. The diagnostic lift bypass is for this explicitly requested mock test;
it remains false by default. Agent-launched ROS processes were stopped after
validation; publishers must be relaunched to review a new result.

Previous experiments below are historical.

## Current verified configuration — two bowls, 2026-10-09

**12.5 cm squares, 0.5 cm gaps, whole-cell exclusivity.** Lower bowl then upper
bowl passed, each occupying one distinct adjacent slot: (1,1) and (1,0).
Full projected geometry was independently checked inside the selected squares.
The shortlist now retains yaw alternatives whose recentered footprint shifts
up to 25 mm from an anchor; its 16-candidate budget is unchanged. Every proposal
still undergoes support, containment, collision and exclusive-reservation checks.

| Object | Reserved slots | Search (s) | Motion validation (s) |
| --- | ---: | ---: | ---: |
| Lower bowl | 1 | 0.905 | 0.878 |
| Upper bowl | 1 | 0.616 | 0.396 |

End-to-end 31.805 s, layout 20.903 s. HGGD calls took 8.221/6.270 s.
Both saved PlanPlacement responses have success and motion_success true.
The existing mock lift-only bypass was enabled: transfer/descent checks remain
active, but lift is not claimed collision validated. Physical execution false.
All 133 placement tests passed. Intermediate grid/clearance trials did not
complete both objects; the successful configuration uses original 2 mm packing
clearance and 6 mm contact tolerance. Evidence: `~/bowls125_yaw_layout.json`,
`~/bowls125_yaw_layout/`, `~/bowls125_yaw_verification.json`,
`~/bowls125_yaw_tests.log`. Stack remains alive for RViz.

Previous experiments below are historical.

Previous defaults (2026-10-09): **10 cm squares, 1 cm gaps, exclusive cells**.
The ordered rerun accepted the cup and lower bowl; the upper bowl had no free
footprint-matching group. RViz confirms six usable slots, all six reserved by
the first two objects. End-to-end 25.159 s; layout calculation 15.972 s.
All 130 placement tests passed. Evidence: `~/grid10_exclusive_layout.json`,
`~/grid10_exclusive_validation.json`, `~/grid10_exclusive_stack.log`.
Earlier size experiments below are historical.

Previous defaults (2026-10-09): **4 cm squares, 1 cm gaps, exclusive cells**.
The rerun accepted cup and lower bowl, but the upper bowl had no available
footprint-matching slot group. Total 21.236 s; layout 14.319 s. RViz shows
36 cells, 17 reserved and 19 free. All 130 placement tests passed. The earlier
7 cm and shared-slot experiments below are historical.

# Compact grid placement: cup and two bowls

Historical shared-slot experiment (superseded by exclusive-slot use): the best tested configuration places **cup → lower bowl → upper bowl**, all on
the tray floor. It uses the original 7 cm squares/1 cm gaps, with optional finer
reservations for disjoint portions of shared cells. This is a tested heuristic
result, not a proof of globally minimum packing.

| Object | Cells touched | Newly consumed cells | Search (s) | Pose/motion validation (s) |
| --- | ---: | ---: | ---: | ---: |
| Beige cup | 4 | 4 | 7.548 | 0.179 |
| Lower white bowl | 4 | 2 | 4.389 | 0.393 |
| Upper white bowl | 4 | 4 | 2.523 | 0.243 |
| Union | — | **10 of 12** | — | — |

The cup and lower bowl use different portions of two common cells. Their actual
reconstructed projected clouds have **21.1 mm** minimum separation. The other
pairwise gaps are 43.7 and 111.4 mm. Reservation masks include clearance and are
disjoint. Ten touched cells correspond to 490 cm² of nominal square area; the
5 mm reservation mask, including its dilation, covers approximately 392.75 cm².
The two upper-right cells remain unused. Each object still touches four cells;
sharing unused portions reduces the union compared with twelve exclusive cell
allocations. Earlier whole-cell and coarse shared-space trials placed only two.

Total request-to-result time: **33.003 s**. Layout calculation: **27.213 s**.
Search times are engine scopes; pose/motion validation includes its transport and
substeps. These overlap public API/engine timings and must not be added again.
The more thorough compact search is slower than centered grid matching.

## Current exclusive-slot control

Use `floor_slot_share_unused_space:=false` (the default). Each successfully
validated object reserves every touched cell; another object cannot reuse any
portion of those cells. Reset reservations before rerunning. The shared-slot
results below are historical and are not the requested inventory-slot policy.

## Exclusive-slot rerun — 2026-10-08

Restarted mock and reasoning stack with sharing explicitly false, reset all
reservations, and reran cup → lower bowl → upper bowl. The result was
`LAYOUT_PARTIAL`: cup accepted, both bowls rejected by final attached-object
collision/IK validation, not by reservation commit. Request-to-result time was
19.203 s; layout calculation was 12.906 s. Cup search took 0.411 s and validation
0.256 s. The latched grid contains twelve cells, four reserved exclusively by
the cup and eight free. Failed bowls consumed no cells. This does not reproduce
the historical two-object strict result; candidate/grasp selection differs.
No collision constraints were relaxed to force completion. The existing mock
lift-only bypass remained enabled; transfer and descent checks remained active.
Twelve floor-slot regression tests passed, including atomic conflicts and
reserved-cell exclusion. Evidence: `~/grid_exclusive_layout.json`,
`~/grid_exclusive_layout/`, `~/grid_exclusive_validation.json`, and
`~/grid_exclusive_stack.log`. The stack remains running for RViz inspection.

## Historical shared-slot controls and validation

- `layout_support_mode:=floor_slots`
- `floor_slot_share_unused_space:=true` (default false)
- `layout_candidate_count:=16`
- 1 cm position phases, 5° yaw samples; unchanged IK solver and motion planner.
- Prefer smaller groups and fewer new cells, preserve a contiguous opening, and
  retain spatial/yaw diversity within each group-size tier.
- Ordered layout updates the immutable collision scene after every accepted
  object; failed objects remain obstacles. Grounding and HGGD use the original
  synchronized observation.

Start the mock separately, then launch the tested configuration:

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio placement_backend:=heightmap \
  layout_support_mode:=floor_slots floor_slot_share_unused_space:=true \
  layout_candidate_count:=16 aligned_zero_pose_yaw_limit_degrees:=90 \
  placement_bypass_attached_collision_during_lift:=true \
  pipeline_profiling:=true allow_physical_grasp_execution:=false
```

The lift-only bypass is the previously agreed diagnostic setting. Robot, fingers,
object-arm, transfer/descent and floor support checks remain active. All saved
responses have successful complete motion calculation with that bypass marker;
this does not claim collision-free source lift.

Reset before independent runs and submit the ordered template:

```bash
ros2 service call /pipeline/commit_floor_slots my_actions/srv/CommitFloorSlots \
  '{reset: true}'
ros2 service call /vlm_reasoning_node/chat_prompt my_actions/srv/Chat \
  "{prompt: '[PLAN_UPRIGHT] place beige cup at far left, small white bowl at bottom left, small white bowl in middle left above the other bowl onto dark rectangular tray'}"
```

Local RViz review:

```bash
rviz2 -d "$HOME/grid_cup_bowls.rviz"
```

`/placement/placement_candidates` contains the three accepted selected objects
and their placement grippers. `/placement/floor_slots` shows blue free cells,
orange previously touched cells, green current cells and yellow reserved areas.
Markers were verified to contain all three labels, ten used cells and 1571
reservation-mask samples. This preset is a local output; the reusable RViz
entrypoint remains `vamp_node/grasp_debug_rviz.launch.py`.

## Checks and artifacts

Constrained build passed. **136 placement tests**, **78 VAMP tests**, and ten
focused master unit tests passed. The full master test attempt reports its
existing flake8 failure; unrelated lint cleanup was excluded. Whitespace and
Python compilation checks passed.

Independent checks of the saved responses verified successful full motion,
nonempty lift/transfer/descent paths, finite Piper-limited joint values, disjoint
ideal reservation masks and reconstructed pairwise separation. The longer
sequence also passed after fixing pinned RGB/cloud/transform fallback.

Raw artifacts remain outside Git: `~/grid_compact_layout7.json`,
`~/grid_compact_layout7/`, `~/grid_compact_best_validation.json`,
`~/grid_compact_best_clouds.npz`, `~/grid_compact_rviz_validation.json`,
`~/grid_compact_final_tests.log` and `~/grid_compact_stack7.log`.
