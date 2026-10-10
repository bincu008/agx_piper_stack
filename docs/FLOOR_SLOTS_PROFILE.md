# Grid floor policy profiling — 2026-10-08

Recorded beige cup at far left into the dark rectangular tray. Existing LM Studio
RoboBrain profile, real RGB grounding, EdgeSAM and HGGD; 16 placement candidates.
Three sequential paired edge/grid trials, all successful. Times cover each public
API from request through returned response via the shared master microstep runtime.
The recipe validates placement IK and Cartesian descent; these numbers exclude
held-object lift/transfer motion. Grounding/presence requests are separate.

| Scope | Existing edge (s) | Floor slots (s) |
| --- | ---: | ---: |
| `find_placements` median | 1.949 | 0.809 |
| `plan_placement` median | 0.172 | 0.621 |
| `commit_floor_slots` median | — | 0.005 |
| Complete placement portion, median per-run sum | 2.120 | 1.423 |
| Complete recipe, median wall time | 10.598 | 10.313 |

Grid search was **58% faster**, while the complete placement portion was **33%
faster**. Grid pose validation took longer because retained coarse candidates
required more IK attempts. End-to-end median decreased about 3%; RoboBrain and
segmentation variation affects the whole recipe. These are three scene-specific
measurements, not a general speed guarantee. Separate row medians need not sum
to the median per-run total. The older screenshot describes a different cup
recording and is not a controlled baseline for this scene.

The tray grid retained 12 usable squares. A geometrically smaller two-cell group
was proposed, but the validated selected cup placement used a four-cell group.
The choice therefore satisfies both footprint fit and pose feasibility. A second
calculation without reset also passed and committed different free cells, checking
reuse. Reservations are reset before the final single-cup RViz view.

| # | API | Purpose | Median runtime (s) |
| --- | --- | --- | ---: |
| 1 | `capture_observation` | Capture synchronized RGB-D | 0.355 |
| 2 | `pin_grasp_observation` | Preserve grasp frame | 0.002 |
| 3 | `pin_placement_observation` | Preserve placement frame | 0.003 |
| 4 | `verify_presence` | RoboBrain presence — cup | 0.560 |
| 5 | `ground_object` | RoboBrain box — cup | 1.278 |
| 6 | `segment_object` | EdgeSAM cup mask | 1.016 |
| 7 | `prepare_grasp_input` | Prepare HGGD input | 0.009 |
| 8 | `calculate_grasps` | HGGD generation and grasp validation | 1.968 |
| 9 | `update_object` | Register cup state | 0.004 |
| 10 | `select_grasp` | Select ranked grasp | 0.005 |
| 11 | `cache_grasp_plan` | Cache selected grasp | 0.005 |
| 12 | `verify_presence` | RoboBrain presence — tray | 0.660 |
| 13 | `ground_object` | RoboBrain box — tray | 1.230 |
| 14 | `segment_object` | EdgeSAM tray mask | 0.819 |
| 15 | `find_placements` | Grid generation, matching and stable placements | 0.809 |
| 16 | `plan_placement` | Placement/pre-place IK, collisions and descent | 0.621 |
| 17 | `commit_floor_slots` | Reserve selected grid cells | 0.005 |

Artifacts outside Git: `~/cup_grid_compare.log`,
`~/cup_grid_compare_{edge,floor_slots}_{1,2,3}.json`,
`~/cup_floor_slots_reuse.json`, `~/cup_grid_final.json`.
RViz: `/placement/floor_slots` (blue free, orange occupied, green selected),
`/vamp_planner/refined_placed_object_cloud` and
`/vamp_planner/placement_plan_markers`.

Final RViz validation confirmed the selected 2×2 group at cells `(3,0)`, `(4,0)`,
`(3,1)`, `(4,1)`; all grid marker headers matched the selected pose.
Build, 132 placement tests and 47 pipeline tests passed. The final run after
restart took 15.154 s (startup/cold inference variation); use the three paired
medians above for the controlled policy comparison.
Local review preset: `rviz2 -d "$HOME/cup_floor_grid.rviz"`.
