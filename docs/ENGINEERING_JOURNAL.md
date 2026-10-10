# Engineering journal

Brief record of hard blockers, how we addressed them, and what remains open.
This is a history of observed results, not a promise that every scene now works.
Older entries are reconstructed from project reports and the task history;
dates are omitted where they cannot be verified. Historical timings and results
apply to their recorded scene and configuration.

## Earlier work

| Roadblock | Resolution | Outcome / evidence |
| --- | --- | --- |
| Grounding used the old CPU server, which timed out on an image request. | Added the optimized endpoint and validated original-image requests, strict box parsing and image association. Later added selectable LM Studio/server profiles. | Optimized live request completed in 21.248 s. See [grounding report](ROBOBRAIN_GROUNDING.md) and [inference profiles](INFERENCE_PROFILES_AND_TIMING.md). |
| Long inference could associate a result with a newer camera frame. | Pin the synchronized observation and preserve source timestamps, frame IDs and immutable scene inputs. | Association and mismatch checks are documented in the [API reference](STANDARD_PIPELINE_API.md). |
| Separate controller and benchmark sequences made performance comparisons unreliable. | Introduced a shared public API catalog, typed microsteps, master-owned result storage and correlated timing traces. | Deployment and benchmarks use the shared catalog. API/backend/engine timing scopes overlap and must not be added together. See [API reference](STANDARD_PIPELINE_API.md). |
| Benchmark requests returned before completion, and failures obscured the final result. | Wait for worker completion, report a final per-object summary, and skip failed objects while retaining them as obstacles. | Supported workflow is in [box benchmark guide](BOX_BENCHMARK.md). |
| Synthetic box results were mistaken for deployed HGGD performance. | Keep known-cloud contact generation separate from RGB-D/HGGD detection while sharing downstream APIs. | Benchmark timings describe their input mode explicitly. See [optimization report](BOX_PIPELINE_OPTIMIZATION.md). |
| Search spent too long exploring grasp and placement candidates. | Added bounded initial searches, candidate batches, frozen-scene reuse and fallback searches without replacing IK or collision checks. | One documented ten-box comparison improved from 190.35 s with six accepted boxes to 125.31 s with all ten accepted. Scene outcomes differ; this is not a universal speedup. See [optimization report](BOX_PIPELINE_OPTIMIZATION.md). |
| Placement IK succeeded but the held-object motion could still fail. | Validate lift, transfer and descent, then try other validated grasp/goal combinations when motion fails. | A valid final pose alone is no longer reported as a successful full-motion plan. See [API reference](STANDARD_PIPELINE_API.md). |
| A small candidate pool repeatedly sampled similar placement positions. | Added coarse spatial/yaw diversity while retaining bounded candidate counts. | Diversity has focused regression coverage; reachability still requires motion validation. |
| Stacking tied to a named simulated box could not generalize to real observations. | Replaced named-support restrictions with scene-heightmap search, allowing multiple supporting surfaces with at least 80% projected footprint coverage. | Public `stack` policy retains stability and collision checks. See [policy reference](../README.md#available-policies-and-backend-choices). |
| The AnyPlace package name suggested the wrong deployed algorithm. | Renamed the active package/module to `placement_node` and documented deterministic heightmap placement as the default. | AnyPlace remains an optional backend. See [code map](CODE_MAP.md). |
| The large Git diff mixed many accumulated features and was difficult to review. | Audited source changes, ignored generated outputs and retained replay inputs needed by a fresh clone. | Review boundaries and required assets are listed in [change audit](GIT_CHANGE_AUDIT.md). |

## Current recorded scene — 2026-10-07

Scene: elevated striped plate, two white bowls, beige cup and dark tray.
Requested order: plate, lower bowl, upper bowl, cup.

| Roadblock | Resolution / current approach | Status |
| --- | --- | --- |
| Grounding could confuse the two bowls or include the plate stand. | Use separate spatial descriptions for the bowls and explicitly exclude the stand from the plate description. | All four objects have produced usable grasp candidates in the recorded scene; this does not guarantee placement motion. |
| Estimated geometry could extend an elevated plate down to the table. | Avoid table-bottom extrusion when the observed object is elevated. | Geometry correction applied; the plate has subsequently passed full placement-motion validation. |
| Gripper geometry did not always reflect the planned jaw opening. | Apply the actual jaw opening to finger collision geometry. | Correction applied; detailed finger checks remain active. |
| Source-support labels could treat the active object's own region as another obstacle. | Separate the active source mask from other source objects and virtually placed clouds. | Regression coverage added; neighboring objects remain blocking. |
| Sparse table-depth contacts rejected the cup's departure. | Add bounded separating-contact handling and a cumulative unique-point/voxel budget during lift and descent. | Cup full-motion validation passed. Defaults are 32 points and 10 voxels; transfer and other collision checks remain strict. |
| Widely separated sparse contacts failed the spread criterion; the upper bowl also exceeded the previous eight-voxel limit. | Removed the spread criterion and raised the voxel limit to ten as requested. | Changes applied. Other grasp, IK and placement failures still prevent claiming all four objects pass. |
| Parallel compilation exhausted VM resources and disrupted VS Code. | Use one compiler job and a memory-limited build process, stopping the owned reasoning stack before heavy builds. | Subsequent constrained builds completed successfully. |
| A NumPy boolean could not be serialized in a placement request. | Convert the computed option to a Python boolean before serialization. | Subsequent ordered runs passed request construction. |
| Greedy plate placement consumed floor positions needed by later objects. | Test reserving nearer space and retaining more diverse yaw choices within the candidate budget. | **Open.** Best recent ordered run accepted plate, lower bowl and cup; upper bowl found no stable placement. Later ranking changes accepted only plate and cup. |
| A farther plate goal had valid IK, but its source lift failed near a wrist limit. | Test bounded departure alternatives and a checked retreat toward the validated pre-grasp configuration. | **Open.** Latest clearance-before-retreat change built successfully but has not yet been live validated. |
| A suspected background seam could not be reliably separated from plate geometry. | Investigated planar filtering, then removed it when the geometry did not support that classification. | No seam filter retained; the original source cloud remains intact. |
| Placing the plate first did not let the remaining bowl stack on it. | Reviewed the ordered run and stacking data flow. Heightmap search already uses `auto` (floor or stack), but returned no stable candidate for the upper bowl. A plate's projected size alone does not establish 80% support at a compatible height. | **Open.** Run 40 failed at search, before placement IK or motion. Existing counters do not identify whether coverage, height/contact filtering or another prefilter removed the stacking trials; insufficient tray area is not a proven diagnosis. |
| Ordered layout did not carry stacking support consistently into motion validation. | Set ordered `PlanPlacement` mode to `auto`; mark only exposed placed tops as support (`2`) and keep sides/bottoms blocking (`3`). Motion validation now recognizes floor and top contact in auto mode; tops stay object obstacles during departure. | **Handoff fixed.** Regression tests passed, and live captured requests contain `auto` with 7,780 top-support points. A complete stacked bowl motion has not passed; this fix did not cause or resolve Run 40's earlier search failure. |

### Latest verified checkpoint

- Run 40: **3/4 accepted**, 26.778 s end to end; layout calculation 16.734 s.
  Upper bowl failed placement search after the plate and lower bowl were accepted.
- Run 41: **2/4 accepted**, 23.306 s end to end; layout calculation 16.193 s.
  Both bowls failed placement validation under the revised ranking.
- All-four success and all-four floor placement remain **unverified**.
- Coverage was reduced from 80% to **60%** across heightmap defaults, ordered
  layouts, single-object placement and the configurable box benchmark.
  Ordered layouts now forward the configured threshold instead of relying on
  the engine default.
- Run 60 after the handoff/coverage fixes: **2/4 accepted** (plate and cup),
  27.368 s end to end; layout calculation 18.231 s. The lower bowl failed
  lift/descent validation; its recorded lift rejection reached 33 points against
  the unchanged 32-point allowance. The upper bowl failed placement IK.
- Separate public stacking-only searches on those frozen bowl inputs found
  maximum coverage of **26.0%** (lower bowl) and **24.4%** (upper bowl), at both
  80% and 60% thresholds; neither produced a stacking candidate. This identifies
  insufficient height-compatible support in the current heightmap, not simply
  insufficient projected plate area. Whether partial/curved plate reconstruction
  underestimates the usable support surface remains **open**.
- Validation: constrained build passed; 117 placement Python tests and 78 VAMP
  tests passed; Python compilation and whitespace checks passed.
- The failed object was the **third object (upper bowl)**, not the last object;
  the fourth object (cup) passed. Plate-first ordering therefore did not simply
  exhaust all available space. Stacking search rejection and support handoff
  need separate investigation.
- Local evidence: `~/new_scene_plate_first_40.json`,
  `~/new_scene_plate_first_41.json` and
  `~/clearance_pregrasp_retreat_build.log`, `~/new_scene_plate_first_60.json`,
  `~/stack_coverage60_probe.json` and `~/stack_support60_build.log`.
  These run artifacts stay outside Git
  and may not exist on another developer's machine.

## 1% coverage inspection — 2026-10-07

Coverage was reduced to **0.01 at runtime** for RViz diagnosis; the reusable
source defaults remain 0.6. With the original 32-point/10-voxel noise allowance,
the ordered run accepted the plate, lower bowl and cup in **27.629 s**
(layout calculation 20.455 s). The upper bowl passed placement search but failed
the initial held-object check at 33 points/10 voxels. Coverage is therefore no
longer the blocker in this run.

Contact inspection found 33 nearby non-table samples across ten voxels at the
neighboring-object boundary. A separate diagnostic restart raised only the point
allowance to 33, retaining ten voxels. That run still accepted **3/4**, taking
**35.620 s** (layout calculation 25.199 s); subsequent lift checks exceeded the
allowance at 34 points or eleven voxels. No further collision-limit relaxation
was applied. This is an **open departure/contact problem**, not an all-four
success.

To inspect the placement independently, the frozen upper-bowl request was sent
through the same public `plan_placement` API with `plan_pick_place_motion=false`.
Placement IK and descent passed in **0.230 s**. The inspection publisher combines
that pose with the three full-motion accepted poses and explicitly labels the
upper bowl **POSE ONLY - LIFT FAILED**. It does not commit the bowl as a completed
placement or change the original ordered-run outcome.

Inspection topics: `/debug/coverage01_layout` (markers, labels and grippers) and
`/debug/coverage01_placed_cloud` (XYZ only). Local RViz preset:
`~/coverage01_inspection.rviz`. Evidence: `~/new_scene_plate_first_01.json`,
`~/new_scene_plate_first_01_point33.json`, `~/upper_bowl_pose01_result.json` and
`~/upper_bowl_pose01_view.log`. All diagnostic scripts and artifacts stay outside
the source tree. Keep the inspection publisher alive for its latched view.

## Lift-only diagnostic bypass — 2026-10-08

The upper bowl's sparse-contact lift rejection prevented transfer testing.
Added `placement_bypass_attached_collision_during_lift` to VAMP and the official
stack launch, default false. When explicitly enabled, only the held-object versus
environment check is skipped during lift; robot, fingers, object-arm, transfer
and descent checks remain active. Existing bounded departure alternatives remain
available. Motion results carry `LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED`.

The frozen upper-bowl request passed the public `plan_placement` API in
**0.193 s**: lift 0.006 s (41 waypoints), transfer 0.044 s (two waypoints),
and descent 50 waypoints. The retract-then-lift fallback was selected.
This resolves that calculation under the diagnostic bypass; it does not prove
the skipped lift collision check would pass. Point/voxel allowances stayed at
their normal 32/10 defaults; coverage remained 1% for this inspection.

The subsequent full ordered scene hit a **new external blocker before planning**:
LM Studio at `http://10.211.55.2:1234/v1/chat/completions` reported
`connection failed or timed out`, HTTP status 0 (request log 0.006 s).
No HTTP response arrived, leaving tray grounding empty. The client returned
`LAYOUT_SEGMENTATION_FAILED: grounding parse failed for dark rectangular tray`
after 0.496 s. Testing stopped immediately as requested; no server repair,
prompt changes or repeated inference requests were attempted. All-four success
under the lift bypass therefore remained **unverified at that checkpoint**;
see the later ordered-scene completion below.

Constrained build passed, 78 VAMP tests passed, and launch Python compilation
and whitespace checks passed. Modified implementation files:
`src/vamp_node/src/vamp_planner_server.cpp` and
`src/master_node/launch/reasoning_stack.launch.py`; usage documented in README,
AGENTS and the shared API reference. Evidence outside Git:
`~/upper_bowl_lift_bypass_result.json`, `~/upper_bowl_lift_bypass_response.cdr`,
`~/new_scene_plate_first_lift_bypass.json`, `~/lift_only_bypass_stack.log`,
`~/lift_only_bypass_build.log` and `~/lift_only_bypass_cpp_tests.log`.

## Ordered plate/cup/bowls completion — 2026-10-08

| Hard blocker | Resolution and verification |
| --- | --- |
| The lower bowl had a feasible destination but fixed-orientation lift IK and straight retreat failed. | Added a bounded retreat toward validated pre-grasp joints using the existing VAMP RRT-Connect engine. Source association, joint limits and stage collision checks remain active; local search and returned-path size are bounded. The lower bowl passed full motion calculation. |
| The upper bowl's validated retreat rotated 21.1°, beyond the old 20.6° cutoff. | Raised the departure bound to 22.9°, retaining path and collision validation. Its checked reverse-pre-grasp retreat passed. |
| Fine sampling of simultaneous joint travel required 2506 waypoints, exceeding the 2048 cap. | Increased the bounded fallback cap to 4096, retaining actual TCP steps of at most 2 mm and midpoint revalidation. Both bowls passed without increasing placement candidates or changing IK. |

The optimized endpoint on port 18091 did not respond to the health check;
LM Studio at `10.211.55.2:1234` was reachable, so the existing `lmstudio`
profile served the real grounding requests. No server configuration was changed.

The complete ordered run **plate → beige cup → lower bowl → upper bowl**
returned `LAYOUT_PLAN_READY objects=4 failed=0`: **27.137 s end to end**, including
**20.472 s layout calculation**. All proposals used HGGD and the placement budget
remained 16. This diagnostic result uses the previously agreed **1% coverage**
and **lift-only attached-object/environment bypass**; it does not establish a
collision-free lift or demonstrate that all four objects fit on the tray floor.
Robot, finger, object-arm, transfer and descent checks remain active.

Constrained build and **78 VAMP tests** passed. Independent checks of all saved
paths verified source headers, finite joints within Piper limits, nonempty
transfer/descent paths and TCP steps within 2 mm. Lower-bowl bounded-retreat
spatial, height and orientation limits also passed. Evidence outside Git:
`~/plate_cup_bowls_run4.json`, `~/plate_cup_bowls_run4/`,
`~/plate_cup_bowls_path_validation.json` and `~/plate_cup_bowls.rviz`.
The RViz preset shows the four accepted selected poses, replacing the earlier
pose-only inspection view. The final repeat after adding the explicit fallback
joint-limit guard also passed **4/4**, in **28.614 s** end to end and **20.676 s**
calculation. All four saved responses passed joint-limit/path-presence checks,
and the latched RViz topic contained all four selected-object labels. Final
evidence: `~/plate_cup_bowls_final.json`, `~/plate_cup_bowls_final/`,
`~/plate_cup_bowls_final_validation.json`, `~/plate_cup_bowls_stack_final.log`
and `~/bounded_lift_retreat_final_tests.log` (78 tests, zero failures).

## Grid floor policy — 2026-10-08

The first live grid search rejected an XYZ array representation in the geometry
library before matching slots. Normalized that input and added a regression
test; subsequent live searches passed. Fixed 7 cm squares with 1 cm gaps now
reuse a bounded destination grid, with multi-cell footprint matching and a public
source-associated commit/reset API. Failed validation leaves slots available.

Three paired cup trials passed for both edge and grid policies. Median placement
search decreased from 1.949 s to 0.809 s; grid IK validation increased from
0.172 s to 0.621 s because coarse poses required more candidate attempts.
Including reservation, the placement portion decreased from 2.120 s to 1.423 s.
This resolves the requested policy implementation, while pose-validation cost
remains a measured tradeoff rather than a claimed optimization. A subsequent
no-reset calculation successfully reused other free slots. These runs measure
placement IK/descent, not complete lift/transfer motion. See
[FLOOR_SLOTS_PROFILE.md](FLOOR_SLOTS_PROFILE.md) for the full API table.
Build passed; 132 placement tests and 47 shared pipeline tests passed. The final
single-cup run passed and its latched grid headers matched the selected pose.
RViz shows 12 usable slots, with the chosen 2×2 group at cells (3,0), (4,0),
(3,1), (4,1) in green. Evidence: `~/cup_grid_final_validation.json`.

## Compact cup/two-bowl grid layout — 2026-10-08

| Hard blocker | Resolution and verified outcome |
| --- | --- |
| Whole-cell reservations and greedy ranking stranded floor space after two objects. | Kept strict mode and added optional footprint-level sharing, 1 cm phases, remaining-space ranking, and safe reuse of already touched cells. Objects share grid cells only through disjoint reserved areas; measured source geometry is preserved. All three passed using ten unique cells. |
| Search accepted a pose rejected by reservation commit. | Applied the cached occupancy mask before shortlisting/IK and kept the atomic commit check. Shared-mask regression tests and the complete scene passed. |
| The last bowl had geometrically valid floor positions but nearly identical retained wrist approaches failed. | Restored spatial/yaw diversity within each cell-count tier, keeping the 16-pose budget. The upper bowl passed with candidate 3 and HGGD grasp 2. |
| Long planning retained the source cloud while its matching RGB expired from the rolling cache. | Changed VAMP to restore the pinned synchronized cloud/RGB/manual-transform tuple when any component is missing, rather than only when the cloud is missing. The longer ordered run completed with the original frame association. |

The best tested configuration was **cup → lower bowl → upper bowl**, using
`layout_support_mode=floor_slots`, `floor_slot_share_unused_space=true`, 1 cm
position phases, 5° yaw sampling and 16 candidates. All three fit on the floor:
each touched four cells, but cup/lower-bowl reservations shared two cells without
footprint overlap, so the union used **10 of 12 cells**. Closest reconstructed XY
separation was 21.1 mm. Whole-cell and coarse shared-space trials only placed two;
changing to bowls-first did not improve validation. This is the best tested
heuristic configuration, not a proof of global optimality.

The successful run took **33.003 s** end to end and **27.213 s** layout calculation.
Search times were 7.548/4.389/2.523 s; inclusive pose/motion validation times were
0.179/0.393/0.243 s. Finer packing costs more than centered slot search.
The existing agreed lift-only attached-object/environment bypass was enabled;
transfer/descent, robot/finger/object-arm checks and floor support checks remained
active. No stacking or physical execution was used.

Build passed; 136 placement and 78 VAMP tests passed, as did ten focused master
unit tests. The full master test attempt still reports its existing flake8
failure (functional tests passed); broad unrelated lint changes were not made.
Independent saved-response checks verified full-motion success, finite joints
within Piper limits, nonempty lift/transfer/descent, disjoint ideal reservation
masks and reconstructed object separation. See
[GRID_SPACE_OPTIMIZATION.md](GRID_SPACE_OPTIMIZATION.md). Raw evidence:
`~/grid_compact_layout7/`, `~/grid_compact_layout7.json`,
`~/grid_compact_best_validation.json`, `~/grid_compact_best_clouds.npz` and
`~/grid_compact_final_tests.log`.

## Exclusive inventory-slot rerun — 2026-10-08

Shared cells contradicted the requested inventory-slot rule. Disabled sharing,
reset reservations, and documented whole-cell exclusivity as the user-selected
behavior. Verified parameter false, twelve regression tests passed, and only
the accepted cup reserved four of twelve cells; failed bowls consumed none.
The ordered rerun took 19.203 s and accepted one of three objects. Both bowls
remain blocked by final attached-object collision/IK validation; this blocker
is open and was not bypassed. Existing diagnostic lift-only bypass remained
active, with transfer/descent checks preserved. See the exclusive rerun in
[GRID_SPACE_OPTIMIZATION.md](GRID_SPACE_OPTIMIZATION.md).

## Smaller exclusive floor slots — 2026-10-09

Changed default floor squares from 7 cm to 4 cm, retaining 1 cm gaps and
whole-cell exclusivity. Restarted the reasoning stack, reset reservations and
reran cup → lower bowl → upper bowl. Cup and lower bowl passed full placement
calculation; the upper bowl had no available footprint-matching slot group
(`tested=0`), before IK. The 36-cell grid retained 17 reserved and 19 free cells;
free cell count alone does not imply a fitting contiguous group. Three-object
packing remains open; no sharing or collision relaxation was introduced.
The existing diagnostic lift-only bypass remained enabled, so lift is not
claimed collision validated. End-to-end time 21.236 s, layout 14.319 s.
All 130 placement tests passed, including the new 4 cm exclusive-cell regression.
Evidence: `~/grid4_exclusive_layout.json`, `~/grid4_exclusive_layout/`,
`~/grid4_exclusive_validation.json` and `~/grid4_placement_tests.log`.

## Ten-centimetre exclusive floor slots — 2026-10-09

At the user's request, enlarged default slots to 10 cm while keeping 1 cm gaps
and whole-cell exclusivity. Restarted and reset, then reran cup → lower bowl →
upper bowl. Cup and lower bowl passed; their reservations exhausted all six
usable slots. Upper bowl had no available slot group before IK (`tested=0`).
Three-object floor packing remains open; this size did not solve it. Request
25.159 s, layout calculation 15.972 s; all 130 placement tests passed.
Existing diagnostic lift-only bypass remained enabled; transfer/descent checks
were preserved. Evidence: `~/grid10_exclusive_layout.json`,
`~/grid10_exclusive_validation.json`, and `~/grid10_placement_tests.log`.

## Cup-only single-slot fit check — 2026-10-09

The visible bottom footprint suggested one 10 cm slot should suffice, but the
placement search reserves the full projected object, including rim/handle and
completed geometry. On the saved cup input, best sampled aligned extents were
105.9 × 102.5 mm, whereas the bottom contour was about 92.4 × 92.4 mm. Area alone
cannot establish square containment. Existing cell-count-tier ranking already
prioritizes single-slot fits, then larger groups; no geometry was trimmed to
force a one-slot fit. Added regressions for one-slot priority, two-slot fallback
and rejection of area-only overhang. All 15 floor-slot tests passed.
Cup-only public recipe passed in 12.313 s and reserved cells (2,0),(2,1), with
no shared reservations. This recipe validates placement poses/descent, not full
held-object transfer. Evidence: `~/cup10_single.json`, `~/cup10_single_grid.json`,
`~/cup_single_slot_tests.log`. One-slot packing for this full cup remains
geometrically unavailable at the sampled rotations with current clearance.

## Two bowls in adjacent exclusive slots — 2026-10-09

A geometric fit in 12.5 cm squares did not initially yield reachable placements.
The shortlist discarded yaw alternatives because rotating partial footprints
shifted their centers beyond its 5 mm anchor radius. Widened that shortlist-only
radius to 25 mm, retaining the 16-candidate cap, independent containment/support
checks and atomic exclusive commits. Returned to 12.5 cm slots with 5 mm gaps,
2 mm packing clearance and 6 mm contact tolerance. Intermediate 13/13.5/14 cm
and increased-clearance trials did not complete both bowls; they are not the
selected settings. Successful lower-bowl → upper-bowl run accepted both with
full motion calculation in 31.805 s (layout 20.903 s). Saved responses and RViz
geometry verified one slot per bowl, adjacent cells (1,1)/(1,0), no shared cells,
and complete projected geometry inside its selected square.
Existing diagnostic lift-only attached/environment bypass remained enabled;
transfer/descent checks stayed active. This does not claim collision-validated
lift. All 133 placement tests passed, including one-slot priority, two-slot
fallback, exclusivity and recentered yaw alternatives. Evidence:
`~/bowls125_yaw_layout.json`, `~/bowls125_yaw_layout/`,
`~/bowls125_yaw_verification.json`, `~/bowls125_yaw_tests.log`.

## Default exclusive grid and cup/two-bowl cleanup — 2026-10-10

Resolved the three-object floor-grid blocker with whole-cell remaining-space
ranking and retained yaw alternatives. The rebuilt stack accepted cup then both
bowls, each inside one exclusive 12.5 cm slot with 5 mm gaps; bowl slots are
adjacent. End-to-end 20.502 s, layout 14.599 s. No measured geometry or collision
checks were removed. Existing requested mock lift-only bypass remained enabled;
transfer/descent validation passed, so source lift is not called collision
validated. This resolves this cup/two-bowl grid case, not the earlier four-object
plate/cup/bowl scenario.

API cleanup removed duplicate cloud decoding and eager masks for unselected
poses; selected-mask commit remains idempotent and preserves occupancy geometry.
Whole-cell free-space scores are cached by group. Mask work measured 18.04 to
1.20 ms; packing-only medians were essentially unchanged, so a major overall
speedup is not claimed. Defaults now route upright CLI, VLM and ordered plans
through the grid, with atomic commits after validation. Fixed the legacy VLM
plan's missing reservation commit; built/tested this branch, but live validation
used ordered layout. Free orientation and explicit other policies remain.
177 placement/pipeline tests and seven VLM tests passed; VLM build passed.
Mock/stack/test workers were stopped after validation. See
[GRID_SPACE_OPTIMIZATION.md](GRID_SPACE_OPTIMIZATION.md) for timings, repeat
commands and evidence. The old stack did not expose the reset service during
initial discovery; fresh restart completed successfully.

## Adaptive floor-grid sizing and occupancy — 2026-10-10 (resolved)

The initial sizing requirement conflicted with expecting exactly three free
slots. The user clarified that 5 cm is the minimum, not a preferred size.
Implemented 5–12.5 cm rectangular slots, two rows when feasible, longest-axis
alignment, and largest usable rectangle plus non-overlapping extensions.
Differently sized regions required per-cell regional gap metadata; keeping one
global gap could misrepresent their boundaries, so serialization and matching
now preserve those gaps. Occupied-surface feedback blocks the frozen grid rather
than rearranging reserved cells.

Verified the recorded bowl through the public RGB-D/HGGD recipe in 8.908 s on the final rerun,
including placement pose validation and exclusive reservation. Feeding its
placed cloud back through public search preserved all nine cell IDs/bounds;
two were occupied and seven remained free. A separate four-cell regression
verified one occupied and exactly three free. 181 placement/pipeline tests passed.
This live recipe verifies placement poses, not the full held-object transfer.
See [ADAPTIVE_FLOOR_GRID.md](ADAPTIVE_FLOOR_GRID.md) for timings and artifacts.

## Square-slot row policy correction — 2026-10-10 (resolved)

The adaptive grid incorrectly resized every row into rectangles and added a
third row of reduced-size cells. The clarified policy requires square main
slots, two rows scaling only to 6 cm, and additional rows only at full 12.5 cm
size. Replaced equal-depth partitioning with square row selection, rectangular
remainder strips and dynamic gaps. Removed the percentage-gap control; stored
gaps now retain independent row/column dimensions. All 183 placement/pipeline tests passed. Live bowl pose calculation succeeded
in 8.929 s; RViz confirms four 12.5 cm squares and two 9.2×12.5 cm
remainder rectangles in two rows, with a 1.6 cm row gap. Full held-object
transfer was not part of this recipe. Mock and stack were left running at the
user’s request. See ADAPTIVE_FLOOR_GRID.md for the corrected policy.

## Two bowls and cup on square-row grid — 2026-10-10 (resolved with lift bypass)

The new two-row square grid accepted the lower bowl and cup, but the upper
bowl had no stable support candidates at the default 6 mm contact tolerance.
A bounded 8 mm trial produced candidates, exposing a motion blocker: the
upper bowl's attached-object lift accumulated 33 obstacle points, exceeding the
32-point budget (10 occupied cells). Remaining alternatives also failed IK or
collision checks. No collision checks were bypassed. Stopped at this new blocker
and restored the default 6 mm support tolerance; the partial two-object layout
remains visible. Latest trial took 19.841 s (layout 13.719 s). A cup-first retry
also failed the cup's lift check, so ordering alone does not resolve this scene.

The archived plate segmentation has approximately 23.5 cm maximum projected
span; an empty-grid geometry test requires four 12.5 cm cells. This does not
verify plate IK/motion or availability after the other objects are placed.
Artifacts: `~/bowls_cup_square_contact8_layout.json`, its capture directory,
and `~/plate_slot_estimate.json`. The free-space preview publisher was stopped
before rerunning the ordered layout to avoid showing stale overlays. The later
user-requested lift bypass resolved the three-object calculation at 8 mm support
tolerance; see the default-lift-bypass entry. This is not a collision-validated
lift result.

## Default lift bypass — 2026-10-10 (verified)

The user requested skipping attached-object/environment checks during lift by
default for closely packed source scenes. Updated both the reasoning launch and
standalone VAMP parameter defaults to true. Robot, finger, object-arm, transfer
and descent checks remain active, and responses retain the explicit bypass
marker. Rebuilt VAMP and reran the previous 8 mm support-contact trial to isolate
the lift change. All three objects passed in lower-bowl → upper-bowl → cup
order: 36.932 s end-to-end, 25.285 s layout. Saved motion responses all show
`LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED` and successful motion calculation;
transfer/descent checks remain active. Lift is not collision-validated. All nine
VAMP test executables passed after adding the installed ONNX Runtime library
directory to the test environment. The 8 mm support tolerance is a runtime
override; its source default remains 6 mm. Stack and mock remain running.
Artifacts: `~/bowls_cup_square_bypass_layout.json`, its capture directory and
`~/bowls_cup_square_bypass_verification.json`.

## Runtime comparison baseline — 2026-10-10

Successful ordered mock calculation: lower bowl → upper bowl → beige cup,
all three accepted. LMStudio profile with RoboBrain 2.5, EdgeSAM/HGGD,
heightmap `floor_slots`, 16 placement candidates, whole-cell exclusivity,
two rows of 12.5 cm squares plus 9.2×12.5 cm remainder slots, 1.6 cm row gap.
Packing clearance 2 mm, runtime support-contact tolerance 8 mm (source default
6 mm), aligned yaw limit 90°. Lift-only attached/environment bypass enabled;
transfer/descent, robot, finger and object-arm checks active. Physical execution
disabled. This is one run after rebuilding/restarting, not a repeat-run median.

| Measured scope | Seconds |
| --- | ---: |
| Chat submission through complete response | 36.932 |
| Ordered layout callback | 25.285 |
| Outside layout callback, not separately attributed | 11.647 |

Per-object controller measurements, in execution order:

| Object | Grasp call | Placement search | Placement validation call | Sum of these three scopes |
| --- | ---: | ---: | ---: | ---: |
| Lower bowl | 7.519 | 1.706 | 0.441 | 9.666 |
| Upper bowl | 6.577 | 0.821 | 0.434 | 7.832 |
| Beige cup | 1.362 | 0.833 | 0.414 | 2.609 |

These stage sums total 20.107 s. The remaining 5.178 s inside layout includes
operations not individually reported in this response; do not label it solely
transport overhead. The grasp scope includes HGGD and candidate validation,
not just inference. Placement validation includes pose/motion work and response
handling, not just IK.

Saved `PlanPlacement` engine measurements (seconds):

| Object | Pose validation | Descent search | Lift | Transfer | Backend validation scope |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lower bowl | 0.204770 | 0.170538 | 0.008354 | 0.076256 | 0.379681 |
| Upper bowl | 0.105650 | 0.061442 | 0.017380 | 0.216991 | 0.405177 |
| Beige cup | 0.186062 | 0.159162 | 0.009819 | 0.147661 | 0.376930 |

Engine scopes overlap and are nested inside controller calls. Do not sum this
table or add it to the previous table. Every saved response confirms motion
success and `LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED`; lift times do not represent
collision-validated lifts.

The preceding no-bypass trial took 19.841 s (layout 13.719 s) but accepted only
two objects. Its shorter runtime is not a successful speed improvement. The
older 20.502 s three-object result used a different fixed grid and order, so
this run does not establish an improvement over that result either.

For future comparisons, record the same scene, object order, inference profile,
grid/clearance/tolerance, candidate count, bypass setting, accepted/failed counts
and timing scope. Compare runs that complete the same work; preferably report
several repetitions and distinguish cold/warm starts and host load. Keep this
baseline as historical evidence rather than overwrite it. Raw evidence:
`~/bowls_cup_square_bypass_layout.json`, its per-object CDR capture directory,
`~/bowls_cup_square_bypass_verification.json`, and `~/lift_bypass_stack.log`.

## Checkpoint review and grid matching optimization — 2026-10-10

Reviewed the accumulated diff against the API-split checkpoint. No tracked
build/debug outputs or safely disposable implementation files were found.
Retained required replay inputs, coupled schemas/consumers and requested backend
deletions. Updated GIT_CHANGE_AUDIT.md with the current 98-path inventory
(94 staged change records after four renames are detected).

A strictly contained one-cell bounding rectangle now avoids repeatedly scanning
the full footprint, retaining point-level checking for boundaries, neighboring
territories and multi-cell groups. 432 old/new cases have identical ordered
results. Alternating five-repeat matching benchmark (three real footprints,
72 yaws each): median 4.229818 s before, 0.628723 s after, 6.73× faster in this
isolated scope. Do not treat this as an end-to-end speedup.

Successful lower-bowl → upper-bowl → cup rerun: all three accepted, client
18.123958 s, layout 12.629 s. Configuration matches the prior bypass run:
adaptive square slots, 16 candidates, 8 mm runtime support-contact tolerance,
2 mm packing clearance, lift-only bypass enabled and physical execution false.
All saved responses retain the bypass marker and successful motion calculation.

| Object | Grasp call (s) | Placement search (s) | Validation call (s) |
| --- | ---: | ---: | ---: |
| Lower bowl | 3.825 | 0.763 | 0.202 |
| Upper bowl | 3.188 | 0.388 | 0.218 |
| Beige cup | 0.665 | 0.384 | 0.222 |

The prior 36.932 s run also had slower grasp calls; warmed state and host load
are confounding factors. The observed total reduction is not attributable solely
to grid matching. Functional checks: 223 passed, one skipped. New grid Flake8
and docstring checks pass; master retains 508 Flake8 findings (committed version
511), while package-scoped master docstring checks pass. The initial workspace-
root lint invocation included generated/vendor trees and is not a valid package
lint count. No broad controller formatting rewrite was applied.

Evidence: `~/checkpoint_grid_benchmark.json`, `~/checkpoint_layout.json`,
`~/checkpoint_layout_capture/`, `~/checkpoint_functional_tests.log` and
`~/checkpoint_stack.log`. The diagnostic rerun initially reused the previous
capture path; the initial baseline timings remain preserved above and in
`~/lift_bypass_initial_summary.json`, while the newer captures are copied into
the separate checkpoint directory. Do not interpret reused CDR files as the
initial baseline's engine timings.

## Maintenance

Add an entry when a hard blocker is identified, then update it after a fix is
validated. Describe the problem and resolution at the system level in one or
two sentences each. Keep unresolved and build-only results clearly marked.
Do not record every failed candidate, routine check or repeated run as a new
blocker. Link reusable reports or summarize verification; keep raw run artifacts
outside the source tree.
