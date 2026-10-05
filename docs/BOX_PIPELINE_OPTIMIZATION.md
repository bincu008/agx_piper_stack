# Box benchmark pipeline optimization

This change tunes the calculation-only synthetic box benchmark. It does not
replace or modify IK, VAMP RRT-Connect, collision geometry, clearance, jaw/contact
rules, or the physical-execution gate. Synthetic boxes supply known contact
seeds; HGGD detection and current-arm-to-pregrasp motion are not measured here.
Production callers retain the exhaustive packing behavior unless they explicitly
supply the new optional packing budget.

## Search policy

- Collect a small, deduplicated grasp pool first (at least four by default).
  Prioritize the configured fixture approach angles, retaining both closing-axis
  families and the complete original seed set for fallback. Build each source
  and scene message once per object.
- Try up to 32 stable packing candidates, with corner/adjacency priority and
  spatial/yaw diversity. Exact containment, support, stability, and clearance
  tests are unchanged. Candidate generation stops early, not just result truncation.
- Validate placement candidates in batches of eight. Stop only after placement,
  hover, descent, source lift, and held-object transfer all pass.
- A failed source lift excludes that grasp only within this object's immutable
  scene. A failed transfer excludes the selected grasp/goal combination; other
  goals for the same grasp and other grasps at that goal remain eligible.
- On fast-search failure, restore all remaining grasp seeds and exhaustive
  placement search before switching from floor to stack. If no stable footprint
  exists, no early candidate quota was reached, so the configured grid was already
  searched completely; repeating it or expanding grasps cannot create a footprint.
- Keep the existing optional refined floor search. Failure handling continues to
  retain rejected boxes as obstacles and proceed to later objects.

The experimental incremental fallback tries the full placement grid with the
small grasp pool, then adds grasp candidates in small groups. It reuses packing
results only within the same frozen object/support scene and tests only newly
added grasps against previously exhausted candidates. It is retained as an option,
but the broader grasp fallback was faster and packed more boxes onto the floor
in the measured scene.

## Control

With mock hardware and the safe reasoning stack running:

```bash
ros2 launch placement_node box_benchmark.launch.py \
  box_count:=10 optimize_pipeline:=true \
  incremental_grasp_fallback:=false plan_pick_place_motion:=true
ros2 service call /box_benchmark/run std_srvs/srv/Trigger '{}'
```

The service waits for the complete benchmark. `/box_benchmark/status`, the final
console summary, and `report.json` record results. `/box_benchmark/cancel` remains
available. Do not launch a second benchmark node while one is running.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `optimize_pipeline` | `true` | Enable early search, candidate batches, and motion-aware retries. |
| `grasp_early_stop_count` | `4` | Initial validated, deduplicated grasp pool target. |
| `placement_early_stop_count` | `32` | Initial stable packing candidate budget; zero disables the budget. |
| `placement_validation_batch_size` | `8` | Placement candidates submitted per VAMP request. |
| `incremental_grasp_fallback` | `false` | Experimental incremental pool growth and frozen packing reuse. |
| `plan_pick_place_motion` | `true` | Calculate lift, held-object transfer, and descent; no execution. |
| `floor_refinement_candidate_count` | `0` | Existing refined floor fallback, disabled by default. |

Use `optimize_pipeline:=false plan_pick_place_motion:=true` for the comparable
exhaustive baseline. The reasoning stack selects LM Studio with
`inference_profile:=lmstudio`; it must retain `allow_physical_grasp_execution:=false`.

## Measured comparisons

The initial scene contains ten 8 cm cubes and a tray grounded through LM Studio.
The baseline and optimized modes both require full placement motion validation.
Later scenes differ as each policy accepts, rejects, or positions objects.

| Policy | Full benchmark | Accepted | Floor | Stack | Failed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exhaustive baseline | 190.35 s | 6 | 2 | 4 | 4 |
| Recommended fast search, broader grasp fallback | 127.2 s | 10 | 6 | 4 | 0 |
| Incremental grasp fallback, with packing reuse | 147.0 s | 10 | 5 | 5 | 0 |
| Recommended profile, confirmation run | 125.31 s | 10 | 6 | 4 | 0 |

The final confirmation was approximately 34% shorter and accepted all ten objects.
The baseline failures were failed source lifts; the optimized policy tried other
validated grasps instead of treating a destination-compatible grasp as sufficient.
These are benchmark observations, not guaranteed runtimes for other scenes.
LM Studio latency varies; use per-object calculation times to compare planning.

Artifacts are in `~/box_pipeline_optimization/`: `baseline10`, `optimized10`,
`incremental10`, and the final `recommended10` verification. Each run records
stage times, grasp service call counts, candidate submissions, motion validation
attempts, source/tray images, fixture clouds, accepted poses, and joint paths.

## Checks

```bash
colcon build --symlink-install --packages-select placement_node
colcon test --packages-select placement_node
colcon test-result --test-result-base build/placement_node --verbose
```

Focused tests cover the bounded height-map search, full-search fallback,
no redundant search for an empty grid, motion-aware alternative grasp/goal
retries, stopping after full success, incremental pool growth and frozen packing
reuse. Existing exact packing, collision/support, cancellation, failure-continuation,
and end-of-pipeline summary checks are retained.

Final verification: the package built successfully; all 92 package tests passed.
The confirmation completed in 125.31 seconds (2.08 seconds tray grounding),
with ten accepted objects, six on the floor and four stacked. All accepted
results recorded complete lift/transfer/descent validation and finite joint paths.
