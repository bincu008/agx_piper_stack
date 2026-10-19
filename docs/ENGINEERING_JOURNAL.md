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

## 2026-10-11 — Configurable large floor square

Added at most one larger square in spare usable space after retaining at least
two ordinary squares. `floor_slot_large_square_scale` defaults to 1.5 and
multiplies side length; 0 disables it. No-fit keeps ordinary subdivision, and
remaining regions become normal squares or remainder rectangles. Geometry,
exclusive reservations and validation-before-commit remain shared API behavior.

A narrow remainder could previously generate squares wider than its working
rectangle. Capping the side by both region dimensions and preventing reduced
three-row layouts fixes this boundary case; a focused regression passes.

Verified: 227 functional tests passed, one skipped, two lint tests deselected;
grid/test Flake8 and docstring checks pass. Live mock lower-bowl recipe completed
in 11.791 s: `find_placements` 1.042 s, `plan_placement` 0.115 s, commit 0.007 s.
RViz markers expose exactly one large square labeled 18.7 cm, two ordinary
12.5 cm squares and three remainder-region slots. This validates placement pose
calculation and reservation, not held-object transfer. Plate fitting was not
retested; the previous estimated 23.5 cm span exceeds the default large square.
Increasing the factor still requires enough spare region after two ordinary
squares, otherwise ordinary multi-cell placement remains available.
Evidence: `~/large_square_tests.log`, `~/large_square_bowl.json`,
`~/large_square_markers.json` and `~/large_square_stack.log`.

## 2026-10-11 — Least-area slot matching, bowls then cup

Problem: cell-count ranking could favor one large slot over a smaller-area
combination, consuming the plate-sized opening prematurely. Search now ranks
ordinary groups by total reserved area and uses large-square groups last.
VAMP receives scores encoding the same priority, preserving geometry preferences
within each tier. This behavior is automatic, with no extra policy selector.

Verified ordered mock layout: lower bowl, upper bowl, beige cup; all three
passed full placement planning and committed without shared reservations.
Matched saved poses against the same public search implementation: lower bowl
used ordinary cell (0,1), 156.25 cm²; upper bowl used (0,0)+(0,3), 224.31 cm²;
cup used large cell (0,2), 351.56 cm². The upper bowl still needed an adjacent
small cell in the validated search; do not claim both bowls consumed one slot.
The remaining small-cell strip cannot contain the cup's short footprint span,
so the cup correctly used the large-square fallback. No-fit never permits
footprint overhang or reservation overlap.

End-to-end request: 20.486 s, layout calculation 13.812 s (overlapping scopes).
Per-object grasp/search/validation seconds: lower bowl 3.935/0.859/0.210;
upper bowl 3.273/0.559/0.300; cup 0.786/0.365/0.249. All three responses include
`LIFT_TRANSFER_DESCENT_VALIDATED;LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED`;
transfer/descent remain checked and lift attached-environment checking uses the
user-requested bypass. Functional checks: 229 passed, one skipped; grid/test
Flake8 and docstring checks pass. Evidence: `~/least_fit_bowls_cup_layout.json`,
`~/least_fit_bowls_cup_layout/`, `~/least_fit_commits.json`,
`~/least_fit_all_tests.log` and `~/least_fit_stack.log`. Stack/replay remain
running for RViz inspection.

## 2026-10-12 — Physical dimensions versus estimated footprints (open)

User measured plate diameter 17.5 cm, bowls 11 cm and tray interior width 26 cm.
Reinspection of saved clouds gives bowl completed-bottom spans 11.60/11.78 cm;
raw plate projected bounding rectangle 20.11×16.79 cm. The previously reported
23.50 cm was maximum projected point-pair distance, not a fitted plate diameter;
it must not be treated as a verified physical diameter. Plate segmentation or
outlier/completion bias remains unverified. Do not tune hardware calibration
from this one footprint statistic.

Replayed first least-area search yields working grid span 34.6×27.0 cm, ordinary
sides 12.5 cm and 2.0 cm between the two rows. Slots reserve more area than the
11 cm bowls, explaining the tight inventory appearance without proving a
camera scale error. The 1 cm tray-width difference still needs validation against
the true inner boundary. Camera XYZ uses SDK depth scaling and mm-to-m conversion;
manual FK/X applies a rigid transform. Next diagnostic is an unmodified camera-
cloud ruler/known-distance check, then separate segmentation, footprint padding
and tray-rim inclusion from depth/intrinsics scale errors. No calibration changed.

## 2026-10-12 — RViz tray working rectangle

Added frozen usable-working-region bounds to floor-grid diagnostics. White
outlines show the planner's estimated inscribed working rectangles; magenta
shows the grid envelope. Both have centimetre labels. They intentionally expose
estimated geometry, not a claim that the physical 26 cm tray interior is 27 cm.
The current labels both read 34.6×27.0 cm: grid and working envelopes coincide,
although the grid contains gaps. Different thicknesses and label heights keep
both visible. No placement or calibration policy changed.

Verified 30 floor-slot tests passed, including geometry serialization. The final
ordered bowls/cup replay passed all three, 18.273 s end to end, 12.724 s layout;
search times 0.748/0.458/0.343 s and validation 0.209/0.203/0.199 s. Requested lift
bypass remains in effect; transfer/descent checks remain active. Marker labels
were read back from the live topic. Evidence: `~/tray_outline_tests.log`,
`~/tray_working_rectangle_markers.json`, `~/tray_outline_final_layout.json` and
`~/tray_outline_final_stack.log`. Stack and replay are running for inspection.

## 2026-10-12 — Shared perception ratio trial

User approved correcting camera-derived measurements globally while preserving
robot dimensions, joints and fixed movement distances. Added a configurable
measured/real ratio in `master_node/config/calibration/perception_scale.yaml`,
currently 1.036. ActiveVision applies its inverse (0.965250965) once to optical
XYZ at ingress before FK*X. Maps, observation geometry, derived HGGD depth,
refinement and placement share this correction. Optional corrected registered
depth is 32FC1 metres. Raw camera messages/snapshots remain unchanged. The
parameter is read-only; changing it requires a stack restart to prevent cached
observations mixing scales. No joint commands or physical movement were scaled.

Verified: vision/master build passed, dedicated C++ scale test passed (XYZ,
RGB/organization/header preservation, depth agreement, identity and invalid
ratios). Live manual-FK/X comparison checked 229,832 points, maximum position
error 0.000000213 m; corrected optical depth maximum error 0.000000060 m.
Both bowls then cup passed full placement planning: request 18.590 s, layout
12.935 s. Grasp/search/validation seconds: lower bowl 3.490/0.726/0.436;
upper bowl 3.412/0.341/0.170; cup 1.183/0.326/0.229. Requested attached-environment
lift bypass remained enabled; transfer/descent checks remain active. RViz now
shows working/grid envelopes 33.0×26.2 cm. The earlier physical-dimension issue
remains unverified: this trial demonstrates consistent correction, not proof
that the camera had a uniform scale error or that physical execution is accurate.

Evidence: `~/metric_scale_build.log`, `~/metric_scale_validation.json`,
`~/metric_scaled_layout.json`, `~/metric_scaled_layout/`,
`~/metric_scaled_markers.json` and `~/metric_scale_stack.log`. Stack and mock
remain running for inspection. Previous unscaled outline label values remain
in `~/tray_working_rectangle_markers.json`.

## 2026-10-12 — Plate-first layout and tight-slot measurements

**Problem:** Plate segmentation included its stand, and underside completion used
its distant table, inflating the footprint. Raster snapping removed reachable
single-slot orientations. Horizontal floor labels misclassified the tilted tray
as an obstacle; wide gap territories also consumed cells beside a contained bowl.

**Resolution:** Filter plate/dish masks against a measured body plane, retain the
raised rim, and infer a local underside only for thin elevated bodies. Preserve
continuous slot centers and use grid-relative yaw; reuse matched groups through
ranking. Label tray support relative to its measured normal with the same 3 mm
band. Physical single-cell containment takes precedence over gap territories.
Commit still checks the accepted FK pose; obstacles and scheduled objects remain
blocking. No physical diameter was substituted for measured geometry.

**Verified:** Plate → lower bowl → upper bowl passed twice consecutively after
all fixes, with zero failures. All three selected candidate #1. RViz reservations
contain exactly the two 12.5 cm squares and the 18.75 cm large square; both spare
cells remain free. Final run: `~/plate_bowls_scaled_run13.json` and its sibling
CDR/NPZ directory; previous confirming run: `~/plate_bowls_scaled_run12.json`.
The first body/grid-only success (`run10`) still needed an extra bowl cell;
the final floor-label and gap fixes resolved that limitation.

| Object | Grasp API (s) | Placement search (s) | Placement validation (s) |
|---|---:|---:|---:|
| Plate | 2.757 | 7.188 | 0.230 |
| Lower bowl | 3.441 | 9.201 | 0.169 |
| Upper bowl | 3.339 | 4.639 | 0.168 |

Final request-to-response: **39.216 s**; layout calculation: **33.687 s**.
These are sequential wall-time scopes; broader API/engine traces overlap and
must not be added. No regression tests ran concurrently during this measurement.
Trial parameters: slot max 0.125 m, large scale 1.5, clearance 0.002 m,
contact tolerance 0.008 m, perception ratio 1.036 and grasp yaw limit 90°.

All placement responses include `LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED`, the
existing user-selected mock exception. Robot/finger/object-arm and transfer/descent
checks remain active; do not claim lift environment collision validation.
Corrected cloud bounds remain conservative: plate 18.25 × 15.72 cm, lower bowl
11.75 × 10.70 cm, upper bowl 11.53 × 11.45 cm. These are XYZ extents, not verified
physical diameters. Raw capture inputs were not modified.

**Checks:** VAMP build and 10 CTest executables passed. The final Python suite
passed 233 functional tests (one skipped; lint suites excluded), including the
wide-gap/empty-footprint regression. Focused tests cover elevated thin/tall bodies, exact slot
containment, tilted support labels, gap ownership and shared reservation overlap.
Stack and mock remain running for RViz inspection.

## 2026-10-12 — Cup appended after plate and two bowls

**Problem:** The remaining floor-slot area has enough total area but is too narrow
for the cup. Combining the two 7.45 cm spare squares produces a 20.5 × 7.45 cm
band; the measured completed cup footprint has a minimum projected width of
8.99 cm before clearance (XYZ bounds 9.03 × 11.33 cm).

**Verified outcome:** Replayed plate → lower bowl → upper bowl → beige cup. The
first three passed and retained their exclusive slots. Cup HGGD returned one
validated grasp; floor search returned zero contained candidates, before placement
IK or motion. Rejection is consistent with the measured geometry, not a planner
failure. Full request took 42.144 s; layout calculation 35.351 s. Artifacts:
`~/cup_after_plate_bowls.json`, `~/cup_after_plate_bowls/cup_fit_report.json` and
its CDR inputs. No algorithm changes or reservation reset after completion;
stack remains running for RViz. Fitting all four on the floor requires a different
slot arrangement or wider free band; this is unresolved under the present grid.

## Maintenance

Add an entry when a hard blocker is identified, then update it after a fix is
validated. Describe the problem and resolution at the system level in one or
two sentences each. Keep unresolved and build-only results clearly marked.
Do not record every failed candidate, routine check or repeated run as a new
blocker. Link reusable reports or summarize verification; keep raw run artifacts
outside the source tree.


## 2026-10-12 — Guarded physical payload execution

**Problem:** Placement paths were calculation-only, and a stale gripper force
reading could be mistaken for contact after closing.

**Approach:** Add shared, revision-bound lift/transfer/descent/release APIs with
fresh state/contact checks, exact refined attachment association and no retry
on unknown actuator state. Add a bowl recipe that stops holding after a 5 cm
lift. Keep the requested lift-only attached/environment bypass and all other
collision checks. Require post-command gripper feedback.

**Verified outcome:** Seven packages rebuilt successfully. The functional suite
passed 244 tests (lint tests excluded); the final focused suite passed 34 tests,
including fake lift/transfer/descent/release, stale-feedback rejection, source
ownership and no retry after unknown actuator state. All ten VAMP CTest
executables passed with ONNX Runtime on the library path. The four live public
actuator endpoints returned `PHYSICAL_EXECUTION_DISABLED`, without commands.
New modules passed scoped lint. Repository-wide legacy lint still fails.
Artifacts: `~/held_motion_final_build.log`, `~/held_motion_final_tests.log`,
`~/held_motion_final_focused.log`, `~/held_motion_final_cpp.log` and
`~/held_motion_execution_gate.log`. Physical motion remains untested.

**Open blocker for the first bowl trial:** Fresh HGGD grasps for both bowls
(two candidates each) pass grasp validation, but a standalone 5 cm lift plan
rejects its initial gripper/environment state. Detailed finger spheres 87,
94 or 95 intersect recorded points at base-frame height 5.2–5.7 cm. This happens
before movement and is separate from the requested attached/environment bypass.
Bounded semantic source-edge recovery did not resolve it; neighboring geometry
and lower table/stand returns remain blocking. Do not infer that these are
confirmed physical collisions from this model-only result. Need to reconcile
grasp and held-path gripper validation before recommending the physical trial.
No additional checker bypass was added. Artifacts:
`~/final_bowl_lift_validation.json`, `~/final_bowl_lift_validation.log`.

## 2026-10-12 — Physical bowl final-approach IK failure

**Problem:** The operator's cached-grasp trial reached pre-grasp and successfully
refined the bowl, but final-approach planning rejected its first waypoint with
`IK_FAILED_AT_WAYPOINT_1`. Endpoint grasp validation did not establish a complete
continuous approach from the actual post-motion state. This is a new blocker,
before held-motion preparation and separate from the earlier lift collision.

**Verified outcome:** Context `40b258184de547b7b6994b2bcd7d0794` stopped after
4.498 s; pre-grasp execution took 4.119 s. Refinement logged a jaw opening change
from the original selected 8 mm to 26.4 mm. The first intermediate pose uses the
refined fixed orientation and a single previous-joint IK seed; the returned
failure does not distinguish solver convergence, pose/feedback mismatch or an
unreachable path. No final approach, gripper-stage execution or lift was reached.
**Open:** compare measured TCP against joint FK and refined orientation, then
inspect the first waypoint's IK diagnostics. No automatic hardware retry or
collision-check changes were made. Operator report: `~/bowl_grasp_lift.json`;
engine log: `~/.ros/log/component_container_mt_1218278_1791784818443.log`.


## 2026-10-12 — Complete planning before physical movement

**Problem:** The old two-command trial previewed endpoints, then moved to
pre-grasp before refining and validating the final approach. Planning could fail
after the arm had already moved.

**Resolution:** Add `prepare_grasp_lift`, `get_prepared_grasp_lift` and
`execute_prepared_grasp_lift`. The preview command validates and freezes the
pre-grasp transfer, fixed-orientation approach and lift before any command.
Execution follows only those cached paths; it invokes no IK, refinement or
replanning. Initial-observation HGGD geometry replaces close-up refinement in
this workflow; the legacy manually staged refiner remains separate. Physical
feedback, contact and collision stops remain active, with no retry on unknown
actuator state. Source replacement/expiry invalidate the prepared plan.

**Verified outcome:** Seven packages rebuilt; 103 master/pipeline functional
tests and all ten VAMP CTest executables passed. New-module scoped lint and
`git diff --check` passed. Mock validation ran on ROS domain 73, isolated from
real drivers, with physical execution disabled. Complete preparation rejected
recorded joint2=-0.01037 rad (minimum 0) and joint5=1.23585 rad (maximum 1.22173)
before commands; the request took 5.067 s. Direct frozen-scene approach validation
of two bowl candidates returned `IK_FAILED_AT_WAYPOINT_1` and
`COLLISION_AT_WAYPOINT_50`, both before any actuator execution. Thus the sequencing
blocker is fixed; bowl feasibility and recorded joint-limit mismatch remain open.
No physical motion was tested and no collision constraints were relaxed. Test
processes were stopped. Artifacts: `~/preflight_grasp_lift_build.log`,
`~/preflight_master_pipeline_tests.log`, `~/preflight_cpp_tests.log`,
`~/bowl_preflight_validation.json`, `~/preflight_approach_validation.json`.

### 2026-10-12 — Live start joint exceeds configured model limit (open)

**Problem:** Restarting the arm did not resolve grasp/lift preparation rejection.
Read-only live feedback repeatedly reported joint5=1.238956876 rad (70.987 degrees),
above the loaded Piper URDF maximum 1.2217304 rad (70 degrees). Other arm joints
were within the URDF limits. RViz geometry cannot establish joint-limit compliance.

**Change:** Joint validation now reports the joint, measured value and bounds;
held-motion feedback errors explicitly identify the current feedback as the source.
Limits and measurements remain unchanged. The generated VAMP Piper model also
has a restrictive joint5 range (approximately +/-1.22 rad), so widening only the
master check would not resolve the shared-model mismatch.

**Verified outcome:** 29 execution/held-motion/preparation tests passed and
`git diff --check` passed. No actuator commands were issued. The physical/model
mismatch remains open: determine encoder calibration versus actual position or
robot variant before changing limits. Restart the reasoning process to load the
improved Python diagnostics; existing driver processes were left running.

### 2026-10-12 — Relaxed pre-grasp conflicts with fixed Cartesian approach (open)

**Problem:** Valid grasp endpoints failed IK at the first Cartesian waypoint.
Inspection found proposal generation permits a relaxed pre-grasp orientation,
whereas the final approach imposes the final grasp orientation immediately.
That first waypoint can therefore demand rotation as well as a small translation.
This identifies a sequencing inconsistency, not proof that every endpoint pair
admits a straight, fixed-orientation path.

**Change:** Frozen-scene offline preparation resolves a collision-checked aligned
pre-grasp first, returns its actual FK pose and joints, and plans the initial
transfer to those resolved joints. A bounded 5/6/8/10 cm standoff fallback along
the original approach direction handles an unreachable relaxed pre-grasp.
Final-approach IK now uses 0.5 mm position and 0.01 rad rotation tolerances,
rather than allowing the generic 4 mm/0.20 rad endpoint tolerances to exceed
the Cartesian step. Collision diagnostics include the actual commanded opening.
All collision, continuity and joint-limit checks remain active; no solver changed.

**Verified outcome:** Seven packages built, 30 focused Python tests and all ten
VAMP CTest executables passed. Physical-disabled mock replay ran on ROS domain
73. Initially both bowl candidates rejected aligned pre-grasp resolution.
After the shorter-standoff fallback, candidate 1 passes alignment and Cartesian
IK but rejects at waypoint 23: finger sphere 88 intersects target point
[0.194593,0.0812052,0.0568237], outside the intended contact patch. Candidate 2
still fails aligned IK, with zero collision rejections. A fresh RoboBrain box,
EdgeSAM mask and HGGD run also failed approach collision validation (waypoint 22);
explicit RIM contact selection did not resolve it. No broad target/contact
exemption was introduced. Full preparation took 5.264 s and rejected recorded
joint2=-0.010367 rad before planning; recorded joint5 also exceeds its limit.
Thus no complete motion plan or successful grasp/lift was obtained. New in-limit
replay feedback and an approach-valid grasp are still required. No actuator
commands were issued. Runtime/build artifacts: `~/bowl_aligned_diagnostics.log`,
`~/bowl_fresh_approach.log`, `~/bowl_offline_prepare_final.json`,
`~/bowl_short_pregrasp_build.log`, `~/bowl_offline_cpp_tests.log`.

### 2026-10-12 — Operator-selected 2 degree joint margin; approach remains blocked

**Problem:** Live bowl preparation rejected joint2=-0.342 degrees despite a
valid grasp candidate. The operator explicitly requested a 2 degree expansion
at both ends of every joint range and calculation-only debugging.

**Change:** Added startup `joint_limit_tolerance_degrees` (0--2, default 2).
Master URDF-derived checks, shared Cartesian/IK bounds, all numerical IK clamps
and deterministic seeds, and generated Piper scalar/SIMD sampling transforms
now use consistent expanded ranges. Raw feedback, FK, collision geometry and
driver/controller limits are unchanged. No actuator API was called. Added
all-joint boundary/sampling regression tests and pre-contact collision details.

**Verified outcome:** `vamp`, `vamp_node` and `master_node` rebuilt; 38 focused
Python tests and all ten VAMP CTest executables passed. A synchronized live
capture was saved outside source at `~/bowl_joint_margin_snapshot`, then replayed
on isolated ROS domain 73 with physical execution false. Joint2=-0.005969 rad
passed preparation with the margin. Original bowl preparation then rejected
`COLLISION_AT_WAYPOINT_18`; shorter standoffs, seed variation and actual-FK target
inspection did not resolve the finger/scene collision. Detailed rejection:
finger sphere95 versus point [0.292313,0.0856673,0.0476146], opening 0.0227152 m.
The RIM-directed calculation took 5.717 s and rejected aligned IK (best rotation
error 0.0456 rad, zero collision rejections), not a joint-limit violation.
No complete grasp/lift plan succeeded; physical execution is not ready.
Artifacts: `~/bowl_margin_prepare.json`, `~/bowl_margin_rim_prepare.json`,
`~/bowl_actual_fk_approach.log`, `~/joint_margin_build.log`,
`~/joint_margin_cpp_tests.log`. Debug replay/stack stopped; real drivers left
under operator control.

### 2026-10-12 — Operator-selected finger/scene approach bypass; complete plan passed

**Problem:** After the joint margin resolved start-state rejection, the original
bowl approach still failed finger-versus-scene checking before intended contact.
The operator explicitly requested disabling that check during approach.

**Change:** Added startup/launch `final_approach_check_finger_environment`
(default false). Only approach waypoint finger spheres 82--95 skip scene
collisions. Arm/wrist 0--73, palm/base 74--81, self-collision, joint limits,
IK and continuity checks remain active. Grasp proposals, aligned pre-grasp,
lift, transfer and descent retain their existing checks. Prepared-plan statuses
preserve `FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED`. Setting the parameter
true restores the complete approach checker. This bypass is not a geometry
correction or evidence of finger clearance. The earlier rim alignment failure
is not claimed resolved; successful validation used the original bowl target.

**Verified outcome:** Two packages rebuilt, 32 focused Python tests and all ten
VAMP CTest executables passed; scoped Python lint and diff checks passed.
The fresh captured scene replayed on domain 73 with physical execution false
returned `ALL_MOTION_PLANNED` in 7.337 s for 'small white bowl at bottom left'
and a 0.05 m lift. The complete calculation also reported
`LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED`, using the previously requested lift
policy. No arm/gripper commands were issued; physical execution remains untested.
Debug processes stopped. Operator must prepare again against fresh live feedback
and execute its own unexpired prepared revision. Artifacts:
`~/bowl_approach_bypass_prepare.json`, `~/approach_bypass_build.log`,
`~/approach_bypass_cpp_tests.log`.

### 2026-10-12 — Relaxed pre-grasp with explicit standoff rotation

**Problem:** A relaxed pre-grasp could not directly start a fixed-orientation
approach. The operator requested gradual orientation adjustment before the
close translation rather than requiring an already-aligned initial pose.

**Change:** Added a fixed-position shortest-arc rotation prefix (at most 20
degrees total, 1 degree per waypoint), using previous-solution IK seeds, joint
continuity, strict pose tolerances and full finger/arm/palm/self collision checks.
The close translation remains straight and orientation fixed. Frozen preparation
may find a different relaxed 5/6/8/10 cm standoff; its actual starting pose/joints
are returned for the initial transfer. The prior aligned-start solution remains a
bounded fallback. `alignment_waypoint_count` identifies the prefix in the full
returned path. The master caches/executes both segments without mid-motion planning.
Failed alignment prefixes expose no partial path. Existing translation/lift
bypass policies remain separate and do not disable standoff-rotation checks.

**Verified outcome:** Seven packages rebuilt; 40 focused Python tests and all
ten VAMP CTest executables passed, including rotation-position continuity,
collision rejection, rotation bounds and prefix preservation. Scoped Python
lint and diff checks passed. Physical-disabled mock replay of the recent capture
returned `ALL_MOTION_PLANNED` for 'small white bowl at bottom left' and a 5 cm
lift in 6.031 s, with one alignment waypoint. The result retains both requested
approach-finger and attached-object/lift bypass markers. The rim-specific target
still rejected `STANDOFF_ALIGNMENT_IK_FAILED_AT_WAYPOINT_1` in a 4.939 s task;
that rim feasibility blocker is open, not claimed resolved. No actuator commands
were issued. Debug replay/stack stopped. Artifacts:
`~/bowl_standoff_alignment_prepare.json`,
`~/bowl_rim_standoff_alignment_prepare.json`,
`~/standoff_alignment_build.log`, `~/standoff_alignment_cpp_tests.log`.


## 2026-10-12 — Replace fixed-position alignment with optional near-grasp

**Problem:** Live bowl preparation passed endpoint IK but failed the first
fixed-position orientation-alignment waypoint. Endpoint feasibility did not
establish a continuous in-place rotation.

**Change:** Operator requested pre-grasp -> optional near-grasp -> final grasp.
Try strict-orientation near poses at 2/1 cm, validate a sampled joint transition
and short Cartesian finish, then fall back to a fully checked direct joint
transition if no near route passes. Never skip path validation or execute a
partial plan. Existing explicit approach finger-scene bypass remains identifiable.

**Verified outcome:** VAMP rebuilt; all 10 CTest executables and 40 focused
Python tests passed. Isolated recorded-mock bowl preparation returned
`ALL_MOTION_PLANNED;NEAR_GRASP_USED` in 7.259 s (motion preparation 0.556 s).
The rim-specific approach also passed, but complete preparation rejected a
separate initial-lift finger/scene collision (sphere 88); that blocker remains
open. Existing approach-finger and attached-object-lift bypasses were active;
these are not fully collision-validated physical trials. No actuators were
commanded. Debug mock/stack stopped. The prior live reasoning stack was stopped
to free compilation memory; camera/arm drivers were left untouched. Artifacts:
`~/bowl_optional_near_prepare.json`, `~/bowl_rim_optional_near_prepare.json`,
`~/near_grasp_build.log`, `~/near_grasp_cpp_tests.log`.


## 2026-10-12 — Closed-gripper lift blocked by finger/scene check

**Problem:** Approach passed under the requested finger-scene bypass, but lift
preflight rechecked the planned closed-gripper pose and rejected finger sphere
88 against a scene sample. This was a modeled collision, not failed lift IK or
an actual actuator failure; sample ownership is not established by that log.

**Change:** Operator requested scene-check bypass from closing through lift,
restoring it for transfer. Added a separate default-enabled robot/finger-scene
lift bypass, retaining self/object-arm/IK/limits and all transfer/descent checks.
Responses explicitly identify bypassed lift checks.

**Verified outcome:** VAMP/master rebuilt. All 10 VAMP CTest executables and
40 focused Python tests passed, including lift-bypass/transfer-restoration policy.
Isolated recorded-mock ordinary and rim bowl preparations both returned
`ALL_MOTION_PLANNED` with `NEAR_GRASP_USED` and both explicit lift-bypass
markers. The previously blocked rim initial state now passes under the requested
policy. No hardware motion occurred; this does not validate physical clearance.
Debug nodes stopped. Artifacts: `~/bowl_lift_scene_bypass_prepare.json`,
`~/bowl_rim_lift_scene_bypass_prepare.json`, `~/lift_scene_bypass_build.log`,
`~/lift_scene_bypass_cpp_tests.log`. Live drivers were untouched; reasoning stack
was stopped for the build and must be manually relaunched.


## 2026-10-12 — Physical pre-grasp arrival not confirmed

**Problem:** Operator execution moved the arm toward pre-grasp, then returned
`STATE_UNKNOWN: PREPARED_WAYPOINT_FAILED` after 10.46 s. Live logs show only
20% speed commands, with no switch to the 5% approach stage. The initial
pre-grasp trajectory did not satisfy the joint-arrival check; the current error
does not record the waypoint index or per-joint residuals.

**Outcome:** Root cause subsequently identified and corrected in the nominal
command-limit entry below. Calculation completed, but physical execution did not confirm
pre-grasp arrival and did not reach gripper close/lift. Read-only inspection:
`move_arm_to` polls 100 times and requires every commanded joint within 0.01 rad
for prepared waypoints. Need target/feedback residual diagnostics to distinguish
tracking error, timeout or feedback issues. No automatic actuator retry performed.


## 2026-10-12 — Planning margin commanded beyond physical joint limit

**Root cause:** Live VAMP logs commanded pre-grasp joint 5 at 1.257 rad (~72°).
Read-only live feedback settled at 1.220579 rad (69.93°); other joints matched,
with no driver fault. The previous 2° planning margin admitted an unreachable
actuator target; extending arrival time or tolerance would conceal the mismatch.

**Change:** Separate feedback tolerance (2°) from commanded IK/sampling margin
(default 0°). Check nominal command bounds for prepared paths before caching.
Add stage/waypoint and worst joint target/error diagnostics for failed arrival.

**Verified outcome:** Master rebuilt; 41 focused Python tests passed, including
nominal-command-bound rejection before any actuation. The same recorded bowl
scene returned `ALL_MOTION_PLANNED`; VAMP selected a different pre-grasp with
joint 5 at its nominal 70° instead of 72°. Live calculation from the arm's
stopped pre-grasp camera view failed earlier at HGGD (six proposals, no valid
geometric grasp), so live physical execution remains unverified. The operator
must restore the observation view and prepare a fresh plan. No arm/gripper motion
commands were issued; drivers stayed running and debug stacks stopped. Artifacts:
`~/live_arrival_world_state.txt`, `~/bowl_live_nominal_prepare.json`,
`~/bowl_mock_nominal_prepare.json`, `~/nominal_limits_mock_stack.log`.


## 2026-10-12 — Complete-plan selection and wrist-limited lift

**Problem:** Another live bowl grasp passed approach but fixed-orientation lift
IK failed at waypoint 9 under nominal actuator limits. Preparation cached only
the top endpoint-ranked grasp, so alternate validated grasps were discarded.

**Change:** Keep the validated pool; try up to eight complete plans with bounded
search effort before any movement. Add a limited relaxed-orientation joint lift
fallback after existing routes fail, preserving requested vertical clearance,
nominal limits, bounded movement and configured checks. Failed/partial paths
remain non-executable; source identity and execution revision remain exact.

**Verified outcome:** Three packages rebuilt. All 50 focused Python tests and
10 VAMP CTest executables passed; scoped prepared-planner lint and diff checks
passed. Five initial and five final-build repeated captured-scene full-stack
runs all returned `ALL_MOTION_PLANNED`. Final five: 6.25–7.26 s (mean 6.62 s),
including separate presence/grounding calls. The same bounding box as the failed
live report was reproduced. Successful lift routes used the validated
reverse-pregrasp retreat; the new relaxed endpoint fallback was not exercised
in these successful runs. A fresh live RGB-D/joint-feedback calculation also
passed (runtime below). No physical arm/gripper motion was commanded, so physical
execution remains operator validation. The previously hardcoded 4 cm retreat
threshold was fixed to honor requested clearance, independently checked by the
master. `/pipeline/prepared_grasp` shows the actual completed selected plan
(pre-grasp blue, grasp green), and is cleared when a new preparation starts.
Artifacts: `~/robust_bowl_prepare_6.json` through `~/robust_bowl_prepare_10.json`,
`~/robust_bowl_live_prepare.json`, `~/robust_grasp_lift_final_build.log`,
`~/robust_grasp_lift_final_cpp_tests.log`.

Live calculation: 8.156 s total; 0.583 s complete motion preparation. Debug stacks stopped after validation.


## 2026-10-12 — Ten live retries expose early grasp-search exhaustion

**Problem:** Ten fresh live calculation-only preparations produced 7 successes
and 3 failures (runs 1/4/9). Failed calls found no geometric grasp: the 100-attempt
fast validation budget was exhausted with later HGGD proposals untested. All
three inferred the same box [143,145,376,389]. These failures preceded lift and
were not repaired by the earlier successful captured-scene lift tests.

**Change:** Retain the 100-attempt successful fast path; when it has no valid
grasp, continue through remaining proposals under `maximum_candidates_to_test`
(default 600). Preserve all geometric, IK, self/scene and pre-grasp checks.

**Further blockers and fixes:** A larger search cap alone still gave 7/10;
valid endpoints also failed lift retreat because a fixed angular subdivision
exhausted the path sample cap. Expand measured HGGD seed variants and retain a
small backup candidate pool; sample RRT retreat by bounded joint/TCP changes.
This produced 9/10, including a successful 699-waypoint retreat. A bounded
same-frame AUTO-to-RIM retry still left one empty-grasp failure (9/10).
Calculation-only CLI preparation now permits three fresh-observation attempts,
releasing failed contexts and preserving all attempt timings. It excludes
actuator/unknown-state/input failures and never forces an invalid plan.

**Outcome:** Final fresh live batch: 10/10 complete grasp/lift preparations
passed; all used one observation attempt, so fresh-frame recovery was not
exercised in that batch (its retry/exclusion policy has regression tests).
Mean 16.13 s, range 11.24–21.61 s, versus baseline
7/10 and 8.67 s mean. Reliability costs additional validation time; this is
batch evidence, not a guarantee for every scene. Lift robot/attached environment
and final-approach finger bypass markers remained present as configured by the
operator; no additional collision bypass was introduced for these retries.
55 focused Python tests and 10 C++ suites pass; scoped lint and diff checks pass.
No physical motion. Reasoning stack remains running with physical execution false.
Artifacts: `~/bowl_consistency_fresh_1.json` through `_10.json`,
`~/bowl_consistency_fresh_stack.log`, `~/bowl_consistency_summary.json`.
Baseline artifacts: `~/bowl_consistency_baseline_1.json` through `_10.json`,
`~/bowl_consistency_baseline_stack.log`, `~/bowl_consistency_current` capture.


## 2026-10-12 — Bowl preparation latency target below ten seconds

**Problem:** The reliable 10/10 batch averaged 16.13 s; expanded grasp
validation averaged 8.43 s versus 3.46 s at baseline. User accepts 9/10
if preparation returns below ten seconds.

**Approach:** Stop single-object validation after the first valid raw grasp,
reduce per-seed variants from 24 to 9, preserve complete motion validation and
bounded calculation-only recovery. Multi-object layout keeps its existing pool.

**Additional finding:** Smaller pools alone passed five completed runs but took
9.15–15.38 s, missing the target. IK still evaluated the complete deterministic
seed bank before collision rejection. Deployment now uses its existing
first-valid mode with the same coarse/precise collision predicates inside seed
selection; rejected branches continue, and downstream validation is retained.

**Further result:** First-safe IK passed 10/10, but one 19 s fresh-observation
retry and one 11.25 s run missed the time target. The retry was caused by a
loose initial grasp endpoint failing strict final-approach IK. Initial deployment
grasp IK now uses the same 0.5 mm / 0.01 rad endpoint tolerances. CLI preparation
uses one observation attempt by default; bounded recapture is opt-in (2 or 3).
Known-cloud and layout backup pools remain unchanged.

**New blocker:** Strict first-safe batch passed 8/10. One empty grasp was a
normal bounded geometric rejection; the next run failed because placement had
not yet received the exact captured frame (`LAYOUT_OBSERVATION_PENDING`). Add
up to 0.75 s of delivery waits using the identical request/header; no recapture
or motion retry. Regression test covers same-frame association and bounded wait.

**Mock steering:** User requested stopping live-data tests and tuning the lower
bowl in recorded replay instead. Live reasoning stopped; mock and reasoning use
isolated ROS domain 86, physical execution false. Four strict-IK mock trials
all rejected geometric grasps (4.87–5.78 s). Revert the tightened initial IK
thresholds. The API published requested poses while collision validation used
solved joint FK; publish solved HGGD endpoint FK instead, preserving the original
IK bounds, source header and downstream strict approach checks.

**Verified outcome:** Recorded lower bowl, isolated ROS domain 86: 10/10
complete grasp/lift preparations passed, all one observation attempt, mean
3.356 s, range 2.985–4.341 s. Every run below ten seconds.
Grasp API mean 0.477 s; full motion preparation
mean 0.555 s. Presence and grounding remain
separate measured API calls. Mock and live timing scopes are different inputs;
do not present their timings as a controlled speedup comparison. No added
collision bypass; previously configured approach/lift bypass markers remain.
No hardware actuation. Final checks: 56 focused Python tests and 10 C++
suites pass; scoped recovery lint and diff checks pass. Mock test processes
stopped after validation.
Artifacts: `~/bowl_mock_endpoint_1.json` through `_10.json`,
`~/bowl_mock_endpoint_stack.log`, `~/bowl_mock_latency_summary.json`.


## 2026-10-12 — Twenty paired preparation/cache-handoff trials

**Requirement:** Lower-bowl mock scene, prepare a 5 cm grasp/lift then issue a
second prompt checking the cached handoff, with no hardware actuation; target
at least 18/20 paired successes.

**Change:** Expose `validate_prepared_grasp_lift` through the shared registry
and `pipeline_task --check-cached-grasp-lift`. It checks cache freshness/revision,
empty-hand idle state, starting joint/TCP feedback and nominal waypoint bounds,
without consuming the plan, invoking planners or commanding actuators. Two
regression tests cover success with physical execution disabled and changed
start feedback rejection.

**Blocker:** First four pairs prepared successfully but the new handoff check
rejected nominal-boundary roundoff. Preparation used a 1e-5 rad numerical epsilon
while the new check used exact bounds. Factor one command validator shared by
both; keep the same tiny epsilon, not the 2-degree feedback margin. Regression
covers tiny boundary roundoff and a real nominal-bound violation.

**Verified outcome:** Complete rerun: 20/20 paired successes, zero
failures. Single-attempt preparation mean 3.342 s;
cached handoff mean 0.020 s; API sum mean
3.362 s (range 3.108–3.672 s).
Both command processes including ROS startup: mean
8.091 s, range
5.583–10.373 s.
No movement/gripper commands; physical execution false, ROS domain 86. Real
RoboBrain presence and grounding plus EdgeSAM/HGGD and full prepared motion
calculation are retained. Cache check is not physical execution or contact/
tracking validation. Original approach/lift bypass markers remain as configured.
Per-run contexts released after both prompts. No physical success guarantee
follows from repeated fixed-scene replay. 59 focused Python tests and 10 C++
suites pass; scoped prepared-executor lint (existing compact-style exclusions)
and diff checks pass. Own mock/reasoning stacks stopped after validation.
Artifacts: `~/bowl_pairs_fixed_summary.json`, `~/bowl_pairs_report.json`,
`~/bowl_pairs_fixed_1_prepare.json` through `_20_check.json`,
`~/bowl_pairs_fixed_stack.log`.


### 2026-10-12 — Ordered full-pipeline profile

Fresh lower-bowl mock preparation 4.146 s; cached readiness
check 0.021 s. Combined command wall time including startup
7.325 s. Both contexts COMPLETE with pending=[]; physical continuation
was documented, not submitted. Nested APIs: held scene 0.244 s, final approach
0.036 s, lift-only planning 0.044 s, pre-grasp VAMP planning 0.186 s; these
are included in the 0.522 s preparation wrapper. Full ordered report and raw
events: `~/bowl_full_pipeline_profile.md` / `.json`. No actuation.


## 2026-10-13 — MCP and public execution architecture review

**Finding:** Grounding and numerical planning already have public APIs. Prepared
grasp/lift, older staged grasp and held-motion execution keep separate phase
records, so exposing every current endpoint to an agent would not unify execution.

**Accepted plan:** One master execution session; public prepared-segment and
gripper operations; composite recipes and MCP use the same registry and state.
Feedback/contact checks, attachment updates and speed restoration stay automatic.
Details: `docs/AGENT_CONTROL_ARCHITECTURE.md`. Documentation only; no runtime
changes, new MCP server or physical validation. Implementation remains pending.

## 2026-10-13 — Public prepared stages and actual pending task queue

**Problem:** the previous offline report listed physical continuation as proposed
internal stages. Those tasks were not in the master queue, and the prepared
executor only exposed a compound execute call.

**Resolution:** expose `execute_plan` (pregrasp/approach/lift), `gripper_open`,
`gripper_close`, `get_execution_state` and lift-only `plan_lift` in the shared
registry. Stage and composite execution use the same prepared session, frozen
paths, revisions and contact/feedback gates. Add an explicit action pause before
a queued task; planning results and waiting actuator tasks remain in StepStore.
The standard recipe now constructs the physical stages as public API tasks.

**Verified:** 3/3 fresh lower-bowl RGB-D mock trials completed 14 calculation/readiness
tasks and retained 5 actual pending physical tasks. Real LM Studio presence and
box prompts were separate requests; EdgeSAM/HGGD and all motion calculations ran.
Mean action runtime 4.141 s (3.063–5.374 s); submission-to-result client runtime
4.236 s (3.139–5.493 s). Mean presence 0.511 s, grounding 0.936 s,
segmentation 0.364 s, HGGD 0.518 s, complete motion preparation 0.571 s.
Nested lift API 0.070 s (includes its 0.058 s solver call); do not add nested
scopes to the parent. This includes observation wait and is not an isolated
execution-overhead comparison with prior runs.

101 focused Python tests and 10 planner tests passed. Fake-actuator tests
executed the five public stages and rejected missing contact, wrong phases and
invalid gripper overrides. Live mock services verified catalog discovery and
`PHYSICAL_EXECUTION_DISABLED` without consuming the prepared session. No real
arm/gripper commands were sent. Existing lift and approach environment bypasses
remain explicitly marked; this is not fully collision-validated physical motion.
Report: `~/bowl_public_api_profile.md`, raw timings/queue snapshots:
`~/bowl_public_api_profile.json`. MCP transport, general TCP adjustment and
unified held-object placement/release remain unimplemented.

Validation limitation: running all master/task Python tests from the workspace
root produced 127 passes, one skip and two lint failures. The legacy master
lint wrappers use the current directory and consequently scan generated build/
install/ output and vendored tools (including a third-party parse error).
Do not treat those workspace-wide lint failures as runtime failures or rewrite
unrelated vendor code. Focused changed-module lint and functional checks are
reported separately.

## 2026-10-13 — Physical bowl pinch misses the rim (open)

**Problem:** the arm completed pregrasp and final approach, but the operator
observed an approximately 3 cm rim miss; gripper closed without confirmed contact
and lift remained pending. The photo shows the visible finger inside the bowl
instead of pinching the rim.

**Read-only diagnosis:** the stored HGGD grasp TCP is
(0.288319, 0.180068, 0.122314) m; subsequent live TCP feedback is
(0.288211, 0.180631, 0.123265) m, approximately 1.1 mm apart. Feedback/model
agreement does not establish physical fingertip accuracy. Live perception ratio
is 1.036, applied by dividing optical XYZ once before synchronized FK*X.
At this observation's contact range (corrected 0.439 m), restoring ratio 1.0
would shift the inferred contact about 15.8 mm, not the reported 30 mm by itself.
Potential remaining contributors are hand-eye extrinsics, TCP/finger contact
geometry and robot kinematic/encoder calibration. No cause is yet verified and
no calibration values or actuator commands were changed. Inspect independent
physical references before replacing the global scale with an arbitrary offset.
Saved pose/feedback/scale evidence: `~/bowl_calibration_inspection.json`.

## 2026-10-13 — Known-plate optical scale calibration

**Problem:** physical bowl grasp missed the rim despite close software pose
agreement. The previous 1.036 scale had been inferred from scene dimensions,
without a controlled metric reference. Direct segmented rim depth measurements
also varied with edge noise and could bias calibration.

**Resolution:** capture a fully visible known 17.4 cm plate through the public
RGB-D, grounding and EdgeSAM APIs. Fit a robust outer-rim plane, project silhouette
rays onto it and measure twelve planar directions. Save source-associated CDR,
RGB, masks and measurement arrays outside source. Five captures gave uniform
optical measured/real ratio 0.976819; replace the prior 1.036 YAML default with
validated JSON loading and an explicit `perception_scale_path` launch argument.
Correction remains once at optical ingress, never on robot commands or geometry.

**Verified:** after restart using the saved JSON, five new captures had median
17.3909 cm diameter versus 17.40 cm ground truth (previous median 16.4166 cm).
Projected camera-axis direction means were X=17.231, Y=17.307, Z=17.481 cm;
these are planar projections, not independent XYZ estimates. 134 functional
master/task Python tests passed, including tilted-circle and invalid-JSON
regressions; master build and focused lint passed. Raw calibration/verification
reports and images: `~/plate_scale_calibration.json`,
`~/plate_scale_verification.json`, `~/plate_calibration_report.md`.

**Open:** a single circular plane does not identify independent XYZ scale,
absolute depth offset, hand-eye or physical TCP error. The uniform correction
is verified for this plate view only. The bowl physical miss remains untested
with the new factor; no arm/gripper motion was performed.

JSON migration briefly failed because the build tree retained a dangling symlink
to the removed legacy YAML. Removed only the stale generated links and the
redundant cv_bridge dependency declaration; the final master rebuild passed.
Calibration-only reasoning processes were then stopped; real camera/arm drivers
were left running without actuator commands.

Farther-view verification, 2026-10-13: retained ratio 0.976819 unchanged. Camera-
reported optical depth increased from 38.80 to 48.83 cm (not independently measured
physical depth). Five captures attempted; four accepted with median planar
plate diameter 17.5348 cm, +1.35 mm/+0.77% versus 17.40 cm. Projected camera-axis
direction means X=17.130, Y=17.583, Z=17.749 cm. One capture exceeded the existing
5% directional-spread gate; original batch stopped and a single additional
capture completed the five attempts, without relaxing limits. Saved factor is
approximately consistent at the farther view, but directional reliability is
worse and independent XYZ/depth calibration remains open. Results:
`~/plate_far_results.json`, `~/plate_far_report.md`. No actuator motion or scale
changes; calibration-only stack stopped after inspection.

## 2026-10-13 — Prepared session expires after movement and blocks new planning

**Problem:** an execution reached pregrasp then gripper opening rejected the plan
as stale. The same 60-second preparation/source limit was reapplied between
physical stages. The known `AT_PREGRASP` session then blocked caching a fresh
plan (`PREPARED_EXECUTION_SESSION_OWNS_ARM`), requiring a process restart.

**Resolution:** retain pre-execution freshness but freeze accepted execution
source and use a separate bounded 300-second stage-boundary timeout. At prepared
action failures discard matching idle pending tasks. Clear a known idle empty-
hand session and source/refinement caches without actuator commands, preserving
failure history and other owners. Uncertain/held state remains blocked; explicit
operator-confirmed empty-hand reconciliation requires fresh stationary feedback
and no running pipeline/collision. No actuator retries or automatic reopening.

**Verified:** an isolated ROS_DOMAIN_ID=86 mock run prepared successfully, then
attempting its first physical task with execution disabled failed as expected.
The master automatically reported `SESSION_RESET_READY_FOR_NEW_PLAN`, discarded
all pending tasks and cleared the session. A fresh full preparation then passed
without restarting the process (6.451 s). Regressions exercise preparation TTL
crossing between public stages, source-cache pruning, active timeout, scoped queue
flush, empty-hand reset and rejection/explicit reconciliation of unknown state.
No real arm/gripper was commanded; existing user hardware processes were untouched.
Evidence: `~/session_recovery_verified.json`, `~/session_recovery_prepare2.json`.

Final checks: 144 functional Python tests passed; focused lint, build and
`git diff --check` passed. Latest-code mock action again verified automatic
failure flush and successful immediate re-preparation. Public reset/reconcile
services and the explicit confirmation gate were verified on the mock ROS
services. A failure for another object preserves the existing session; failed
preparation also discards its own future physical tasks. Hardware execution and
operator reconciliation on the real driver remain untested. One restart of the
user's reasoning process is needed to load the update; camera/arm drivers were
not restarted or commanded.

## 2026-10-13 — Small residual physical grasp-position correction

**Problem:** after plate scale correction, the operator reported successful rim
contact but requested a residual base-frame target adjustment: X=-5 mm,
Y=+5 mm, Z=+5 mm. Uniform scale cannot encode a constant translation; a wrist-
frame X-matrix translation would rotate this base-frame trial correction.

**Implementation:** separate validated grasp-offset JSON and read-only VAMP
parameter. Add the vector once when image-based grasp/refinement Cartesian
targets are constructed, before IK and collision checks. Pregrasp and complete
motion planning inherit the corrected endpoint; cached/approach paths do not add
again. Keep source cloud/contact geometry, scale, X-matrix, joint feedback and
synthetic known-cloud proposals unchanged. This is a physical trial adjustment,
not a verified global calibration. Physical result remains untested.

**Verified:** vamp/master build passed with one worker; 145 functional Python
tests and 10 planner test targets passed. Offset regression checks confirm
base-axis translation despite tool rotation, preserved orientation/input pose,
zero application for synthetic proposals and rejection of malformed/nonfinite
settings. An isolated domain-86 mock full bowl preparation loaded the requested
vector, completed all planning/readiness steps and paused with physical tasks
waiting (5.947 s client time). Physical correction remains operator-test-only;
no real arm/gripper commands or user driver restarts. Test reasoning/mock
processes stopped afterward. Report: `~/grasp_offset_bowl_prepare.json`.

## 2026-10-13 — Joint-6 upright/home alignment offset (open)

**Problem:** the operator observes a tilted gripper at commanded home and has
manually rotated joint 6 until the gripper is upright. Planned motion should
preserve the physical orientation expected in RViz.

**Live observation:** ten fresh `/feedback/joint_states` samples agreed on
joint6 = +0.23499113048851653 rad, after the operator's adjustment.
Joints 2–5 were zero; joint1 was +1.3008811579739734 rad,
so this capture is not an all-zero home configuration. Upright orientation is
operator-reported, not independently verified against the robot/tool model.
Evidence: `~/joint6_upright_calibration_observation.json`.

**Requested next implementation:** store the joint-zero correction in a
calibration JSON, load it by default and permit editing that file; add no new
launch arguments. Treat +0.23499113048851653 rad as a candidate joint-6 driver/model zero
offset pending verification. Use raw radians throughout; no degree conversion.
If the model's upright joint6 is zero, convert
`joint6_model = joint6_driver - offset` for planning/FK/RViz and
`joint6_driver_command = joint6_model_target + offset` at the actuator boundary.
Apply that conversion exactly once and consistently across current feedback,
IK seeds, FK/manual camera FK*X, collision validation, paths and execution.
Do not merely append a rotation to the last waypoint: that would invalidate
planned Cartesian poses and intermediate paths. Recheck limit conventions,
homing behavior and physical TCP feedback versus model FK. Preserve the current
X-matrix and perception scale unless independent validation supports changes.

**Status:** journal entry and live feedback capture completed only. No joint
offset implementation, calibration-file modification or arm/gripper motion
performed. Physical/model agreement and the candidate offset remain unverified.

## 2026-10-13 — Implement joint-6 correction at command publication

**User clarification:** replace the earlier proposed driver/model zero mapping
with command-only compensation. Keep raw joints throughout the pipeline and
perform existing target limit comparison before adding +0.23499113048851653 rad.
No new launch arguments; default JSON is editable.

**Implementation:** master loads `joint_command_offset.json` at startup. The
common publisher copies the nominal arm target and adds the recorded raw-radian
value once to joint 6. Other joints, gripper-only messages, raw feedback, FK,
perception scale and X-matrix are unchanged. Cached paths stay nominal. Arrival
and between-stage checks compare measured raw feedback to the command actually
sent; endpoint checks include that commanded roll. Raw-position collision brakes
bypass the addition. Driver-native homing/direct commands remain unmodified.

**Scope:** this intentionally does not implement feedback subtraction or a
calibrated kinematic model. Existing collision/limit validation is nominal and
does not cover the added rotation; no post-addition wrapping/clamping is done.
Driver physical-range rejection remains possible. Physical result is untested;
no real arm/gripper command was sent during implementation.

**Verified:** master build and 155 functional master/task Python tests passed;
focused changed-module lint and whitespace checks passed. Regressions cover raw-
radian JSON loading, one-time final publication, untouched raw feedback and
nominal caches, gripper-only/raw brake handling, nominal clamping before addition,
arrival at the sent raw target, and complete prepared grasp/lift execution with
fake actuator feedback and no replanning. Installed default JSON loads the exact
0.23499113048851653 rad value. Real physical motion was not tested and no test
stack or actuator commands were started; the existing user stack needs one
restart to load this update.

## 2026-10-13 — Latest bowl selects power grasp; local artifacts reorganized

**Observation:** latest preparation context e7f61247011f4a19b633a5426eea18de
succeeded in 4.972 s client time. Description `small white bowl at bottom left`
selects AUTO. HGGD seed #2 was accepted as POWER_GRASP, jaw opening 0.078207 m;
its measured contacts were around 0.011 m in base-link Z. The one-candidate
ranking pool stopped further search. AUTO-to-RIM recovery only runs on an empty
failed bowl result, so a rim request was not attempted. This is current selection
behavior, not evidence of failed rim IK. Explicit `rim` in the target description
selects RIM; no grasp-selection or solver changes were made in this inspection.
Live arm/pose data were read only; no actuator commands.

**Output hygiene:** moved project artifacts into `~/agx_piper_data` calibration,
logs, reports, captures and tools subfolders; no deletions. 1,733 recorded
relocations include top-level artifacts and nested logs. Calibration values
were checked unchanged. Updated current README/AGENTS/API paths and the box
benchmark default output folder. Historical journal paths resolve through
`~/agx_piper_data/relocation_manifest.json`; standard ROS logs and unrelated
home files/repos remain untouched. Existing processes keep loaded calibration;
the next explicit launch must use the relocated override paths.

Inspection note: the read-only query inadvertently reused the earlier
`bowl_calibration_inspection.json` output name. The new result is archived as
`~/agx_piper_data/reports/latest_bowl_grasp_inspection.json`; earlier numerical
calibration observations remain recorded in their journal/report entries.

### 2026-10-13 — Consistent perception XYZ correction

Problem: correcting only the grasp target left measured contacts and scene geometry
unshifted. Resolution: route the existing calibration vector to ActiveVision's
manual camera-to-base transform and disable the separate VAMP correction in the
reasoning launch. Raw optical data stays unchanged; obstacle points are transformed
after downsampling, while organized HGGD observations preserve pixel association
using the same matrix in the existing conversion. Physical accuracy after this
change remains untested; fresh observations and plans require a stack restart.

Validation: vision_node/master_node build passed; both vision CTest regressions
and eight plate-scale tests passed. No live stack restart or actuator command
was performed. Physical validation remains pending.

### 2026-10-13 — Open: command-only roll offset rejected at pre-grasp

Physical execution reached the pre-grasp stage, then raw joint-6 feedback
(2.264093 rad) exceeded the feedback limit (2.129302 rad). Preparation and
cached handoff validate nominal commands before the command-only 0.234991 rad
offset, but HeldMotion.feedback validates raw measured joints against the same
nominal bounds before its offset-aware endpoint comparison. This leaves a
planning/execution calibration mismatch. No fix or hardware retry performed;
need consistent calibrated feedback bounds while retaining independent physical
actuator limits and preflight checks of offset-applied commands.

Resolution: measured joint-6 feedback now accepts the envelope of nominal limits
and their command-offset translation. Nominal commands and planner limits stay
unchanged; raw states are not rewritten and driver enforcement is untouched.
Regression covers the exact 2.264093 rad failure, both offset signs, rejection
beyond the calibrated envelope, unchanged other-joint limits, and all five
prepared execution stages with sent feedback outside nominal roll bounds.
Offline tests passed; physical retest remains pending. No claim of hardware
360-degree travel is inferred or encoded.

### 2026-10-13 — Final approach stop-and-go execution

Problem: prepared approach called blocking move_arm_to and confirmed arrival at
every tiny waypoint. Resolution: approach-only streaming at 50 Hz, with linear
joint-edge timing bounded to 0.05 rad/s, start-state confirmation, live fresh
feedback/collision checks, and a single bounded final arrival check before pose
verification and gripper closing. Pre-grasp/lift behavior and calculation remain
unchanged; command-only calibration still applies exactly once. Driver forwards
JointState targets to move_j/move_js without a ROS arrival handshake. Offline
regressions cover lagging intermediate feedback and fault suppression of commands.
Physical smoothness remains untested; no actuator commands were sent.

### 2026-10-13 — Open: rim contact rejected by signed force feedback

Latest physical run completed pre-grasp and approach, then closing timed out with
GRIPPER_CONTACT_UNCONFIRMED; lift never began. Read-only live feedback shows
width 0.0056 m and force -1.043 N, enabled driver and no reported fault flags.
Contact gate requires force >= +0.2 N, so negative measured gripping force is
rejected despite nonzero aperture. Need verify the driver's signed-force
convention and apply consistent contact evidence to close and held-motion gates.
No actuator retry or contact-check bypass performed.

Resolution: close confirmation, held-contact monitoring and release detection
now consistently use finite force magnitude. Fresh post-command feedback and
nonzero aperture remain mandatory for contact; negative loaded force cannot
authorize release. Regression reproduces -1.043 N contact and rejects nonfinite
or insufficient force. 160 functional tests passed; physical retest pending.
Commanded closing force remains 1 N; no actuator commands sent during development.

### 2026-10-13 — Contact transient at lift admission

Physical close confirmed contact, but immediate lift admission rejected one
feedback sample as HELD_CONTACT_LOST; later live readings were valid. The exact
failed sample was not recorded, so a transient remains a hypothesis. Added a
300 ms bounded stationary settling window at prepared lift and legacy held-stage
admission, preserving fresh feedback, limits, collision checks and force/aperture
thresholds. During motion, contact loss still fails immediately. Errors now include
measured width/force; close timeout logs its rejected values. Signed force handling
was reviewed across close, hold and release. Regression covers recovering dips and
persistent loss without actuator retries. Physical retest remains pending.

Latest physical retest: close reported CONTACT_CONFIRMED, then lift failed after
27 ms with width=0.011480 m, force=+0.070000 N. Later live feedback settled to
width=0.00567 m, force=-1.026 N. This shows the single-sample confirmation can
accept a transient while the gripper is still closing. The stationary settling
window returns on the first qualifying sample, so it does not guarantee settled
contact. Next fix should require sustained fresh contact before authorizing lift;
no actuator commands or retries were sent during this diagnosis.

Resolution: close confirmation now requires at least three distinct fresh samples
spanning 150 ms, with force magnitude >=0.2 N, opening >1 mm and aperture change
within 0.5 mm. Loss, stale feedback or continued closing resets the window; the
overall two-second timeout stays bounded. All shared close callers inherit this
check. Regression reproduces the brief force spike followed by 0.07 N and then
settled rim contact, and rejects repeated samples and moving aperture. 165 tests
passed. Physical validation remains pending; no actuator command was sent.

### 2026-10-13 — Lift waypoint stutter

User observed the same stop-and-go behavior during lift. Prepared lift now uses
the shared approach streaming loop: 50 Hz, conservative 0.05 rad/s joint-edge
timing, no intermediate arrival waits, and final joint/TCP verification. Lift
contact, fresh feedback, limits and fault monitoring remain active during every
streamed command. The existing calculation/collision policy is unchanged.
Regression verifies lift streams without blocking moves and stops sending when
contact is lost. 166 functional tests and master build passed; physical
smoothness remains unverified. No actuator commands were sent.

### 2026-10-13 — Initial grasp detailed-cloud rejection

User requested removing only the detailed local-cloud grasp-pose collision
check. Removed it from both collision-aware IK selection and final candidate
validation, and removed its unused per-candidate cloud construction. Coarse
obstacle-world checks and measured contact/finger-sweep validation remain.
Nearby contact is not guaranteed safe or accepted by this change. Runtime cup
validation remains pending; no actuator commands were sent.

Validation: VAMP build, 10 CTest targets and 166 Python regressions passed. Live cup retest remains pending.

### 2026-10-13 — Initial grasp coarse gripper-sphere rejection

User explicitly requested arm-only environment checking for initial grasp poses.
Candidate collision-aware IK and final candidate checks now exclude palm/finger
spheres 74..95 while retaining arm/wrist 0..73 and full self-collision. The
internal scope defaults to full checks for other callers; no ROS argument added.
Existing stage-specific bypasses are unchanged. Boundary regression covers every
sphere index. Live cup validation remains pending; no actuator commands sent.

Validation: rebuilt VAMP and all 10 CTest targets passed, including initial-grasp sphere boundaries. Physical cup retest remains pending.

### 2026-10-13 — Optional near-grasp must preserve final endpoint

Problem: approach re-solved an already validated grasp from intermediate seeds
and rejected complete preparation when that extra solve failed. Added explicit
final_grasp_joints to the shared approach schema; both master callers pass their
validated endpoint. VAMP checks its FK/limits/collision, solves optional 2/1 cm
intermediates independently, retains a reachable prefix with a validated fixed
endpoint connection, and falls back to direct validated joint interpolation when
needed. No final endpoint IK rerun. Every segment retains collision, joint limits
and <=2 mm TCP sampling. Build/offline validation pending; no physical commands.

Validation: all six affected interface/consumer packages rebuilt; 166 functional
Python tests and 10 VAMP CTest targets passed. The new optional-prefix regression
verifies fixed endpoint preservation, late intermediate failure, direct fallback
and rejection of a colliding connection. Live/physical cup retest pending.

### 2026-10-13 — Opaque final-grasp collision rejection

The latest fixed-endpoint cup preparation failed FINAL_GRASP_ENDPOINT_COLLISION
without identifying robot geometry. Approach endpoint/path failures now include
the colliding sphere index, arm/wrist versus palm/finger category, base_link
obstacle coordinates and existing contact classification. Self-collision reports
up to four exact sphere pairs. Uses the same finger/contact policy as the check;
extra diagnostic traversal runs only on rejection. Collision behavior unchanged.
Rebuild/testing pending; no physical commands sent.

Validation: VAMP rebuild and all 10 CTest targets passed. The new rejection details still require a live calculation after stack restart.

### 2026-10-13 — Sphere 80 at cup endpoint

Diagnostics identified approach endpoint palm sphere 80 overlapping a cup-labelled
point outside the contact patch. User requested removing this sphere from this
checker. Approach-only environment checks and matching diagnostic reporting now
skip sphere 80; other palm spheres, arm checks and full self-collision stay active.
Other collision callers default to no exclusion. Regression covers all 96 sphere
indices. Build/runtime validation pending; no physical commands sent.

Validation: VAMP build and all 10 CTest targets passed, including sphere-80-only scope regression. Physical retest remains pending.

### 2026-10-13 — Inspect collision spheres in RViz

Added standalone read-only live sphere viewer and dedicated RViz launch/config.
Build passed. Live subscription verified 96 sphere markers plus 96 labels in
base_link with valid diameters and sphere 80 highlighted; test viewer stopped.
No actuator commands or user-node restarts performed.

### 2026-10-13 — Sphere RViz camera controls missing

Dedicated RViz configuration omitted camera tools and explicit Orbit target.
Added Interact/MoveCamera/Select/FocusCamera, base_link Orbit target and window
geometry. Launch adds identity visualization-only base_link child for frame
discovery; no computational TF use. Existing RViz log had no rendering errors.
Configuration syntax verified; user must reload/relaunch the view.

Sphere-view consolidation: integrated read-only live spheres and index markers
into grasp_debug_rviz.launch.py/grasp_debug.rviz; removed dedicated sphere-view
launch/config per user request. Viewer executable remains required. Launch and
RViz configuration syntax checked; no actuator commands sent.

### 2026-10-13 — Contact-preserving palm recovery

User confirmed sphere-80 collision for wide cup grasps can be physically real.
Restored sphere 80 in approach checks. During initial grasp only, explicitly
test sphere 80 against the coarse environment; on collision, try whole-TCP
retreats 2..10 mm, preserving measured contacts, rerunning IK and requiring actual
FK contact depth/height/closing alignment to remain within finger geometry.
Reject unresolved palm collision rather than hiding it. Other sphere policies
are unchanged. Build/checks pending; live recovery effectiveness untested.

Validation: VAMP rebuilt and all 10 CTest targets passed. The palm-recovery branch still needs live cup calculation/physical verification; no runtime success claimed.

### 2026-10-13 — Extend palm recovery to sphere 81

Initial grasp recovery now triggers on sphere 80 OR 81, and every adjusted
configuration must clear BOTH before acceptance. Same 2..10 mm retreat, actual
contact reach and IK/arm validation; approach still includes both spheres.
No new arguments or actuator commands. Build/regression validation pending.

Validation: VAMP rebuilt and all 10 CTest targets passed. Live two-sphere recovery remains untested.

### 2026-10-13 — Narrow initial grasp sphere scope

User requested initial grasp environment validation only through sphere 67.
Updated the candidate predicate to include exactly 0..67; palm 80/81 remain in
separate bounded recovery. No change to other stage collision policies or full
self-collision. Boundary regression updated; build/runtime validation pending.

The exact fixed final-grasp endpoint recheck and diagnostics use the same 0..67 scope. Intermediate/motion checks keep their existing sphere policy.

Validation: VAMP build and all 10 CTest targets passed. Live grasp retest remains pending.

### 2026-10-13 — Grasp approach and pre-grasp motion sphere scope

User requested spheres 0..67 for full pre-grasp-to-grasp approach and VAMP grasp
planning, while post-lift placement transfer retains full geometry. Approach
checks now share this scope at all waypoints. Added grasp_motion to the typed
CheckAndReplan request; prepared/legacy grasp controllers and candidate preplanning
select a scoped RRT-Connect validator using full SIMD self-collision and per-lane
0..67 environment checks. Ordinary motions default to full model; held transfer
is untouched. No physical commands sent. Build/validation pending.

Validation: all six interface/consumer packages rebuilt; 166 Python functional tests and all 10 CTest targets passed. Live grasp/placement validation remains pending; no actuators commanded.

### 2026-10-13 — Sparse obstacle-world voxel filter

User requested quick suppression of isolated obstacle cells near the gripper.
Published obstacle world now retains only voxels with at least two distinct
occupied neighbors among the 26 adjacent cells. Single simultaneous pass avoids
recursive erosion; underlying OctoMap evidence and object observations remain
unchanged. Thin real obstacles can also be removed, so this is a heuristic rather
than proof of noise. Regression covers isolated cells, pairs, lines, diagonals
and duplicate cells. Build/live validation pending; no actuator commands.

Validation: vision_node rebuilt and all three CTest targets passed, including voxel-neighbor regressions. Live scene verification remains pending.

### 2026-10-13 — Mock grid trial: two bowls then plate

Isolated domain 86, real RoboBrain/LMStudio and HGGD, physical execution false.
Normal layout failed all three objects at NO_PATH_REACHABLE_PREGRASP (6.926 s),
preventing grid evaluation. Focused grid test disabled optional grasp path
preplanning only: lower bowl placement validated; upper bowl failed placement
IK/descent collision checks across 8 candidates; plate generated no fitting
placement. Layout computation 22.352 s; this excludes destination setup and is
not actuator time. No algorithm change or successful full-motion claim. Saved
reports under ~/agx_piper_data/reports/grid_bowls_plate*. Mock/focused stack
left running in domain 86 for RViz inspection; user hardware nodes untouched.

### 2026-10-13 — Stop-on-failure mock placement inspection

Added opt-in layout_debug_stop_on_failure and layout_debug_show_candidates, both
false by default, enabled only in isolated mock domain 86. Trial marker topic
shows proposed footprint outlines and rigid attachment-derived gripper axes;
these are not validated poses. Fixed noncontiguous OpenCV hull input encountered
in first debug trial; focused regressions passed (10 tests). Trial lower bowl
validated, upper bowl failed eight placement candidates (IK/descent collision),
then stopped before cup. Computation 30.818 s; sixteen proposed failed-object
footprints and sixteen gripper targets verified live (eight tested by validator). Normal continue-on-failure unchanged.
Mock stack remains running for RViz; no physical commands sent. Detailed outputs
under ~/agx_piper_data/reports/grid_bowls_cup_trial*.

### 2026-10-13 — Tilted debug footprint and crowded grasp display

Trial overlay incorrectly flattened the source hull in XY before applying the
source-to-placement rotation, creating artificial tilt. It now transforms actual
3D bottom-contour vertices. Regression verifies tilted source becoming horizontal;
11 focused tests passed. Refreshed mock run verified all 16 candidate outlines
within 0.005 degrees of horizontal. Source bottom still shows its fitted support
plane (~15 degrees), not a gravity projection. Grasp-pool displays are hidden by
default and named accurately; placement targets remain proposed, not accepted
complete paths. Placement algorithm unchanged; upper bowl still fails IK/descent
collision, trial stops before cup (26.188 s). Initial refresh capture timeouts
resolved by restarting only isolated mock replay. No physical commands sent.

### 2026-10-13 — Gravity footprint used for both bowl placements

Fixed source footprint construction: measured support fit supplies height, while
projection uses base_link gravity and preserves measured XY rather than shearing
along a noisy tilted annulus normal. Thin elevated-object underside protection
remains. Added tilted-support regression. Mock ordered lower bowl then upper bowl
now passes both placement validations and exclusive commits (slots 0,0 and 0,1),
calculation 30.591 s. Final upper contour vertical span 0.192 mm verified live.
155 package tests plus focused gravity regression passed. Diagnostic candidate
overlay hidden by default for accepted-layout inspection; mock stack remains
in domain 86. Optional grasp path preplanning stays disabled for this grid-only
trial, so no complete pickup-motion claim; no actuator commands sent.

### 2026-10-13 — Gravity-layout retry: two bowls then cup

Independent retry initially reused occupied slots from the prior two-bowl run.
Reset reservations through public commit_floor_slots(reset=true), then reran
lower bowl, upper bowl, beige cup. Both bowls validated/committed again; cup
packing produced candidates but placement IK/descent validation failed. Trial
stopped on cup and retained its proposed targets. Computation 28.713 s; domain
86 mock stack remains running, physical execution false and optional grasp
path preplanning false for grid-only inspection. No algorithm changes this run.

### 2026-10-13 — Cup placement after two bowls: narrow reachable yaw

Resolved the previous gravity-layout cup blocker. Coarse shortlist rotations
skipped a reachable body-grasp placement between sampled headings. Diagnostic
frozen-scene search found it without changing footprint, IK tolerance or collision
checks. Floor-slot selection now retains up to one quarter of its existing
16-candidate budget for nearby 5-degree yaw alternatives within 20 degrees of
the best-ranked pose; remaining proposals preserve broad coverage. Vectorized
eligibility avoids a quadratic scan. Least-area tiers and exclusive commits remain.

Five fresh domain-86 mock trials reset reservations, then grounded/segmented and
placed lower bowl, upper bowl and beige cup through shared APIs: **5/5 complete**.
Both bowls occupy ordinary slots; cup uses the large slot with body grasp #2.
Cup lift/transfer/descent returned successful full-motion calculation in every
trial, with the configured robot/environment and attached/environment lift
bypasses explicitly reported. Optional pickup path preplanning remains disabled
for this grid trial; current-to-pregrasp motion and physical execution were not
verified. No actuator calls were made.

| Trial | Full request (s) | Layout calculation (s) | Cup HGGD (s) | Cup search (s) | Cup validation (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 25.260 | 19.694 | 1.332 | 0.405 | 0.236 |
| 2 | 26.354 | 20.022 | 1.358 | 0.414 | 0.238 |
| 3 | 27.339 | 21.631 | 1.456 | 0.436 | 0.252 |
| 4 | 27.237 | 20.705 | 1.428 | 0.389 | 0.256 |
| 5 | 25.564 | 19.644 | 1.340 | 0.470 | 0.257 |

Full-request scope is chat service submission through final response and includes
visual grounding/segmentation; layout and nested timings overlap. Artifacts:
`~/agx_piper_data/reports/cup_fine_yaw_five/run_1` through `run_5`, plus
`summary.json`; log `~/agx_piper_data/logs/cup_fine_yaw_stack.log`.
Placement build succeeded; 50 focused candidate/grid/overlay tests passed,
including the new fine-yaw/broad-coverage/budget regression. Final vectorized
selector independently replayed the frozen cup request successfully (16 proposals,
placement validation 0.325 s). This demonstrates recorded-scene repeatability,
not an all-scenes guarantee. Mock stack remains running for RViz inspection.

### 2026-10-14 — Cup body grasp: false no-path rejection from feedback bounds

Reproduced `NO_PATH_REACHABLE_PREGRASP` using the current mock recording, real
VLM grounding and HGGD body contacts. Grasp endpoints existed, but grasp RRTC
rejected measured start feedback before searching: joint2=-0.010367 rad and
joint5=1.235850 rad slightly exceed nominal command bounds. This was incorrectly
presented as an unreachable pre-grasp, not a failed IK solve.

The planner now accepts a start only within the existing configured feedback
tolerance (0--2 degrees), preserves its raw FK/feedback, collision-checks the
complete short transition to a separate in-bounds entry, then runs RRT-Connect
with unchanged command/sampling limits and existing grasp collision scope.
Larger deviations or any entry collision still reject. Prepared execution skips
the initial measured sample so it cannot become an out-of-bounds actuator target.
Applied requested orientation tolerance 20 -> 16 degrees, preferred standoff
0.120 -> 0.156 m and maximum 0.150 -> 0.195 m; shorter visible alternatives stay.
No collision bypasses, calibration or solver replacements were added.

Ten fresh `--prepare-grasp-lift` requests for `beige cup, grasp its body`, each
followed by `--check-cached-grasp-lift`: **10/10 preparations and 10/10 cache
checks passed** in isolated ROS domain 86, physical gate false. Each complete
plan checks current-to-pregrasp, approach and lift. Existing lift bypass markers
and optional near-grasp behavior remain; this does not verify actuator tracking
or successful physical contact. Single preparation attempt, no retry on failure.

| Trial | Preparation API sequence (s) | Preparation process incl. ROS startup (s) |
| --- | --- | --- |
| 1 | 2.954 | 4.344 |
| 2 | 2.745 | 5.855 |
| 3 | 2.507 | 5.577 |
| 4 | 2.577 | 5.157 |
| 5 | 2.420 | 5.870 |
| 6 | 2.513 | 5.176 |
| 7 | 2.462 | 5.375 |
| 8 | 2.945 | 5.800 |
| 9 | 2.659 | 4.475 |
| 10 | 2.419 | 5.470 |

Mean calculation 2.620 s; VLM presence and grounding calls are separate and
included, as is HGGD. Cache API checks took 0.0035--0.0117 s. Reports/logs:
`~/agx_piper_data/reports/cup_body_pregrasp_ten/` (`run_01`--`run_10`,
`summary.json`, `timing_report.json`), baseline
`~/agx_piper_data/reports/cup_pregrasp_baseline.json` and stack log
`~/agx_piper_data/logs/cup_pregrasp_tuned.log`. 36 prepared-execution Python
regressions passed, including never commanding initial out-of-bounds feedback;
focused Cartesian/payload CTests passed, including feedback acceptance/rejection.
Final vamp_node/master_node build, all 10 VAMP CTest targets, and all 36
prepared-execution Python regressions passed. One additional fresh final-build
preparation/cache check passed (2.745 s preparation, 0.0054 s cache API check).
Ten trials show repeatability of this recorded scene, not guaranteed success on
different scenes/hardware. Our isolated reasoning/mock launch processes stopped
after validation; real hardware nodes were untouched.

### 2026-10-14 — Faster prepared approach and lift streaming

Operator requested 50% higher speed, fewer commands and faster publication.
Prepared approach/lift joint timing increased 0.05 -> 0.075 rad/s and command
rate 50 -> 60 Hz; unchanged validated edges/endpoints remain, with live fault,
contact, fresh-feedback and final arrival checks. Driver cap changed 5 -> 8%
(integer representation of the requested 7.5%); pre-grasp/exit speed remains 20%.
No geometry validation samples were removed. Deterministic non-actuating tests
for both stages show a 0.09 rad edge dropping from 90 commands/1.8 s to 72/1.2 s.
Actual hardware timing/lag is not measured, and per-edge rounding can reduce the
realized speed gain. Build passed and 38 prepared-execution regressions passed.
Live nodes were not restarted or commanded; restart reasoning and prepare fresh
before the operator's physical trial.

### 2026-10-14 — Shared gripper clearance and bounded closing target

Operator requested opening 2 cm wider than measured grasp width, then closing
4 cm narrower than the capped opening. Added one shared width helper used by
prepared, staged, compatibility grasp execution and release. Opening is capped
at the 10 cm aperture and closing floored at zero; closing force remains 1 N
and live contact confirmation is still mandatory. Pre-grasp/approach validation
now uses the same wider opening, preventing a calculation/execution mismatch.
A 5 cm grasp commands 7 cm open and 3 cm close; actual closure stops against the
object. Updated fake-contact fixtures to represent nonzero closing targets.
69 focused width/prepared/staged/held-motion tests passed without actuator calls.
Final master_node/vamp_node build and all 10 VAMP CTest targets passed. Live nodes unchanged;
restart reasoning and calculate fresh before physical testing.

### 2026-10-14 — Upright object constraint throughout held motion

Previously upright packing constrained the destination only: transfer and relaxed
lift fallbacks could tilt the held object. Added keep_upright/free motion policy
(default keep_upright) independently of endpoint orientation/support mode.
Constraint uses the fixed object's source gravity axis through tool attachment,
with yaw free and 3-degree tolerance. Enforced during lift before collision
bypasses, all fallbacks, VAMP transfer lane checks and descent; final returned
segments are densely checked between endpoints. Failures remain non-executable;
no implicit unconstrained fallback or solver replacement.

Offline ROS domain 87 replay through public plan_placement used the frozen cup
request after two bowl placements and passed complete lift/transfer/descent in
0.578 s. Independent generated-Piper FK inspection at <=0.001 rad increments
measured maximum object tilt: lift 0.240 degrees (123 samples), transfer 2.149
(203), descent 0.452 (99). Fresh RGB-D/VLM/HGGD body-grasp preparation also passed
(4.365 s), and cached non-actuating handoff passed (0.0092 s), both reporting
OBJECT_UPRIGHT_VALIDATED. Existing lift collision bypasses remain explicitly
reported; physical tracking/contact not tested. Build and all 10 VAMP CTest targets
passed, including tool/object-axis distinction, free yaw, mid-edge tilt rejection
and collision retention. Artifacts: ~/agx_piper_data/reports/upright_motion/;
stack log ~/agx_piper_data/logs/upright_stack.log. No live arm/gripper commands.

52 prepared/held-motion Python regressions also passed. Isolated mock/reasoning
launches stopped after verification; real hardware nodes left untouched.

### 2026-10-14 — Missing shared-API inference previews

The direct pipeline grounding/segmentation path did not save object previews;
only legacy destination captures and HGGD heatmap were present. Added matching
object/destination bounding-box, source-RGB, mask and segmented-cutout outputs to
the shared VLM service handlers. A bounded 32-entry source-header/box association
keeps cup/tray segmentation separate; native expanded boxes remain supported.
Build and all three VLM CTest targets passed. Live RGB was read without actuation,
then processed through isolated domain 89 public ground_object/segment_object APIs:
beige cup box [226,175,498,539], mask 19762 pixels; tray box [754,193,1000,862],
mask 40209 pixels. All four requested previews were saved to ~/robobrain_inference/.
Real arm/driver nodes were not restarted or commanded. Validation stack stopped.

### 2026-10-14 — Cup rear/interior cloud lost despite correct segmentation

Reproduced using a new synchronized live RGB-D/arm snapshot, replayed with physical
execution disabled in ROS domain 90. EdgeSAM retained the cup silhouette, but
foreground component filtering dropped the entire 6427-point rear/interior region:
its median camera range 0.402 m was 41 mm behind the front region 0.361 m, exceeding
the 30 mm band. The two measured surfaces were only 6.1 mm apart at the rim. This
was an object-cloud membership bug, not missing camera depth or bad segmentation.

Kept existing anchor/band behavior and added bounded substantial-component
recovery inside the semantic mask, using the existing 20 mm 3-D neighborhood and
at least three supported samples. Full measured rear/interior component now
retained: 8656 -> 15083 points; 147 tiny/distant samples remain excluded. No invented
hidden wall, geometry scaling, collision relaxation or solver change. Before/after
replay used the same semantic mask and recorded geometry; HGGD body grasp and
VAMP pre-grasp path passed after recovery. Fresh complete grasp/lift preparation
passed in 4.818 s with OBJECT_UPRIGHT_VALIDATED and existing explicit lift bypass
markers; physical motion/contact was not tested. Build and all 10 VAMP CTest targets
passed, including connected far surface recovery, remote background rejection,
minimum component support and insufficient-neighbor rejection.

Artifacts: ~/agx_piper_data/reports/cup_rear_wall/ (retention_before/after.png,
CDR responses, masks/cloud arrays and stats); snapshot
~/agx_piper_data/captures/cup_rear_wall_debug. Overlay green=retained measured
points, red=excluded measured points. Domain90 mock/reasoning stack remains for
RViz inspection; real hardware nodes untouched and no actuator commands sent.

Verified before/after XYZ arrays and semantic masks are identical, and every
rear-component pixel is present in the recovered validated mask. A delayed cache
check correctly expired at the existing 60-second source lifetime; fresh paired
preparation/cache validation then passed (3.379 s / 0.0051 s), without actuation.

### 2026-10-14 — Plate finger pads not aligned with the contact surface

- Problem: a reachable plate-edge grasp could be accepted with finger pads angled across the measured surface, risking displacement when closing.
- Change: after original grasp IK succeeds, estimate the two local contact normals once and try at most three pad-alignment tolerances: 10°, 15°, then 20°. Rotate about the contact midpoint, preserve both measured contacts, and retain pad reach, sweep, IK, collision and pre-grasp validation. Already aligned poses reuse their original IK. Unreachable alignment rejects the candidate.
- Validation: five geometry regression tests and the VAMP package test suite pass. Live plate and explicit-rim queries reject all candidates. The first rim rejections have original pad errors of 75.9–89.5°; both contact normals are incompatible with a common pad direction within 10°, 15° or 20°, so those trials stop before IK. Open blocker: finding a different, geometrically compatible plate contact pair. The previously accepted edge grasp is not verified under the new requirement. No hardware commands were sent.
- Follow-up: the operator selected best-effort alignment with original-pose fallback. The original IK pose/joints are saved immutably; failed angle trials or later tuned-route validation retry that original once through the existing checks. Accepted candidates report `PAD_SURFACE_ALIGNMENT_VALIDATED` or `ANGLE_TUNING_FAILED_ORIGINAL_USED`. Live plate calculation now returns one valid fallback grasp, including a reachable pre-grasp path. Surface alignment itself remains unverified for that fallback; physical holding has not been tested.
- Follow-up timings (live RGB-D, calculation only): general plate query succeeded in 12.750 s overall / 1.598 s grasp API; explicit-rim query succeeded in 8.165 s overall / 1.430 s grasp API. Both returned one original-pose fallback and a reachable pre-grasp path. VAMP package tests: 11 CTest targets passed, including five pad-alignment cases. Reports: `~/agx_piper_data/reports/live_plate_optional_alignment.json` and `live_plate_optional_rim.json`. The calculation-only stack remains running for RViz; physical execution is disabled.

- 2026-10-14 follow-up: changed the bounded alignment ladder to 10°, 20°, 30° at the operator's request, preserving original-pose fallback. On frozen scene 2, general plate and explicit-rim queries both returned a valid original fallback; no tuned candidate was accepted. First logged candidates reject 10°/20° on incompatible normals and fail the 30° IK trial. Grasp API times: 2.207 s and 2.149 s; complete queries: 11.889 s and 6.700 s. Six alignment tests and all 11 VAMP CTest targets pass. Reports: `~/agx_piper_data/reports/plate_alignment_10_20_30.json` and `plate_rim_alignment_10_20_30.json`. No actuation occurred.

### 2026-10-15 — Angle tuning could cross the original approach side

- Problem: choosing the shortest surface-normal rotation could reverse the lateral tilt of a previously accepted grasp.
- Resolution: reject tuned poses that reverse approach or cross to the opposite horizontal side of the original approach, using base_link gravity. Straightening is allowed; an original tilt below 1° has no defined lateral side. Check the proposed rotation, actual IK/FK result and pose after palm recovery. Retain 10°/20°/30° best-effort tuning and the original validated fallback.
- Verification: build passed; nine alignment regression cases and all 11 VAMP CTest targets passed. Scene-2 plate test through public capture/pin/EdgeSAM/preparation/HGGD APIs returned one original-pose fallback and a reachable pre-grasp path (grasp API 0.791 s). Full VLM recipe stopped at presence: RoboBrain `/v1/chat/completions` connection failed or timed out. Used manual image bounds [12,168,274,423] only for the offline test, not a fake inference result. Saved `~/agx_piper_data/reports/approach_side/result.json`. No hardware moved; isolated test nodes stopped. Physical alignment/holding remains unverified.

- 2026-10-15 outward-direction follow-up: a regression exposed that retaining the lateral approach side still permits moving the hand inward on that same side. Added a separate radial check from the measured target centroid through the contact midpoint, comparing physical hand-base displacement (TCP offset removed), before IK, after IK/FK and after palm recovery. Maximum numerical inward allowance is 0.1 mm; ambiguous radial direction retains only the original validated fallback. Three new cases cover same-side inward versus outward rotation, object-relative direction away from the world origin, and ambiguous direction. Build, all 12 alignment cases and all 11 VAMP CTest targets passed.
- Scene-2 repeatability: 10/10 plate and 10/10 cup-body grasp calculations passed, all with `ANGLE_TUNING_FAILED_ORIGINAL_USED`, zero accepted angle corrections. Therefore these runs verify fallback repeatability, not successful outward-tuned physical contact. Mean grasp API: plate 0.785 s, cup 0.383 s; mean capture-through-grasp sequence: plate 1.457 s, cup 1.119 s. RoboBrain unavailable (`curl: (7) Failed to connect to 10.211.55.2 port 1234`); used manual boxes [12,168,274,423] and [8,20,182,181], real EdgeSAM, HGGD, IK/collision and pre-grasp path checks. Reports/CDR outputs: `~/agx_piper_data/reports/outward_scene2/`. No hardware commands sent; isolated test nodes stopped. A successfully tuned grasp remains unverified on this scene.

- 2026-10-15 RViz review aid: added `grasp_angles_rviz.launch.py` and a focused preset for accepted grasps. Existing accepted marker output now includes actual FK tool axes and final base_link roll/pitch/yaw, plus explicit surface-aligned versus original-fallback labels. Rejected proposals remain hidden; the display does not claim fallback angles were corrected. Launch/config checks, VAMP build and package tests passed. A late transient-local subscriber received the scene-2 accepted fallback label (`RPY 179.8, 1.2, -166.9 deg`) and three correctly coloured axes. Used the existing manual-box offline test because RoboBrain was unavailable; no actuation. Detailed validation: `~/agx_piper_data/reports/grasp_angles_rviz.json`. Test nodes stopped.

- 2026-10-15 rejected-angle visibility: added latched `/vamp_planner/grasp_angle_trials` and enabled it in the focused grasp-angle preset. It renders up to 24 distinct, translucent trial hands with requested limits and rejection stages, separate from accepted candidates. Impossible surface/side cases expose a best-fit geometric preview explicitly labelled rejected; that optional preview output never becomes an executable solver result. Build, 13 alignment cases and package tests passed. Scene-2 manual-box plate calculation retained its valid original fallback; a late subscriber verified 24 trial labels, all three limits and two finger markers per preview. Report: `~/agx_piper_data/reports/rejected_angles_rviz.json`. Stack/mock remain on domain 0 for operator RViz inspection, physical execution disabled, no hardware commands sent.

### 2026-10-15 — Angle previews did not bend from the displayed original grasp

- Problem: free normal alignment and previews from earlier discarded candidates produced an unrelated fan of grippers. That did not represent bending the accepted original grasp outward toward the plate surface.
- Resolution: use one hinge about original local tool X through its contact region. Fit the local midpoint surface patch and solve analytically for finger-to-surface targets 10°, 20°, 30°; the larger target is less aggressive. Keep the original IK/FK pose as the reference, measure the same contacts along the rotated jaw axis, preserve all contact/IK/collision/pregrasp checks, and retain the original fallback. Cache preview association by grasp joints, so only the selected grasp's three trials are shown, including rejected ones. No unrelated best-fit rotation is rendered as a requested-angle trial.
- Preliminary verification: build and all 15 alignment tests passed, including unchanged hinge/contact pivot, outward direction and progressively smaller bends. Scene-2 plate returned its original validated fallback; all three bends failed IK. Marker geometry independently measured 9.427°, 19.427°, 29.427° to the observed surface (0.573° numerical solve margin), sharing one axis with progressively less outward displacement. Final reference fix verified after rebuild: shared hinge differs from the displayed original by less than 0.00001°, and all three trial poses have the same contact pivot within 1 micrometre. The measured hand-base shifts outward from the original by 67.30, 63.16 and 56.25 mm for 10°/20°/30° targets respectively, confirming progressively less aggressive bends. Actual surface angles are 9.427°, 19.427°, 29.427° with the 0.573° solve margin. All three fail IK; the original fallback is retained. Final grasp API 2.041 s; all 15 alignment tests and package checks pass. Report: `~/agx_piper_data/reports/original_hinge_rviz.json`. Mock scene 2 and the calculation-only stack remain running on domain 0 with updated markers; no actuator commands were sent.

### 2026-10-17 — Dynamic outward bend targets

- Request/change: use 10°/30°/50° and try only targets strictly below the actual current finger-to-surface angle. 90° schedules three, 45° two, 20° one, and <=10° none; stop on the first passing trial. An already-good pose is tagged `ANGLE_ALREADY_WITHIN_BAND`; the original validated fallback remains available.
- Verified: build and all 17 alignment regression cases passed, including exact 10°/30°/50° boundaries and invalid-angle scheduling; all VAMP package tests passed. Scene-2 plate grasp API succeeded in 1.331 s using its original fallback. Published trials measured 9.427°, 29.427°, 49.718° to the local surface; outward hand displacements decrease 67.30, 56.25, 33.04 mm. Runtime IK introduces 0.732 mm pivot error in the 50° preview, which shows the actual solver result.
- New blocker: 10° and 30° fail IK; 50° now passes IK but fails measured contact reach on the fingers. No checker was relaxed to force acceptance. Markers explicitly identify the rejection. Stack/mock scene 2 remain running on domain 0 for RViz; physical execution is disabled. RoboBrain remains offline and the same manual plate box was used with real EdgeSAM/HGGD. Report: `~/agx_piper_data/reports/dynamic_bend_rviz.json`. No hardware moved.

- 2026-10-17 contact-reach diagnosis: measured the published 50° actual-IK preview. Contacts lie 73.948 mm and 64.295 mm from the physical finger base; fingers are 70 mm long and the checker reserves a 2 mm tip margin (maximum 68 mm). Height and jaw errors pass (<=0.403 mm height against 13 mm allowance; 0.682 mm jaw error against 4 mm). One contact is therefore 3.948 mm beyond the physical finger, not merely within a conservative margin. Relaxing only the 2 mm margin cannot resolve it. Proposed, unimplemented remedy: advance the hand approximately 6 mm along its approach to bring both contacts inside the usable pad, then revalidate IK, outward displacement and collisions. No checks changed and no hardware moved. Measurements: `~/agx_piper_data/reports/finger_contact_50deg.json`.

### 2026-10-17 — Resolved tilted plate contact beyond fingertip

- Problem: the 50° pose passed IK but placed one measured contact at 73.948 mm on a 70 mm finger; relaxing the 2 mm tip margin would still leave the contact outside the physical finger.
- Resolution: preserve pad bounds and use a bounded approach advance only for depth violations. At most two seeded IK retries, 10 mm total, with 0.5 mm position tolerance and 1 mm pad-depth headroom. Reject lateral errors, excessive advances or loss of either contact. Revalidate actual angle/hinge/outward position, sweep, collision and pregrasp; check contact reach again after palm recovery. Mark accepted corrections `CONTACT_DEPTH_RECENTERED`.
- Verified: the selected 50° grasp is accepted at approximately 49.7° after a 6.95 mm advance. Final accepted-pose contacts measure 67.429 and 57.774 mm (unchanged 2–68 mm usable range); height <=0.520 mm against 13 mm, jaw error <=0.703 mm against 4 mm. Four consecutive scene-2 calculations return `GRASP_AND_PREGRASP_SAFE;PAD_SURFACE_ALIGNMENT_VALIDATED;CONTACT_DEPTH_RECENTERED`, including a VAMP path to pregrasp. Grasp API times 1.402, 1.402, 1.308, 1.325 s. All 21 alignment/contact regression cases and all VAMP package CTest targets pass. Used manual plate bounds with EdgeSAM/HGGD because LMStudio is offline; no physical approach/lift/holding validation or actuator commands. Report: `~/agx_piper_data/reports/accepted_contact_recenter.json`; logs `contact_recenter_*.log` outside source. Stack and scene-2 replay remain running on domain 0 for RViz inspection.


### 2026-10-17 — Full plate pick/place preparation

- Blocker: the real LMStudio/EdgeSAM/HGGD recipe found 16 fitting floor-slot poses, but search took 101.031 s and preparation rejected the expired 60-second grasp cache. No motion was attempted.
- In progress: profile the exact frozen search inputs and remove redundant grid-coordinate transforms and two-column membership reductions, preserving every measured point, slot exclusivity, ranking and collision/support checks. An initial offline optimized replay returned 16 poses in 22.977 s under concurrent diagnostic load. Complete motion planning and pending execution stages are not yet verified.
- Search verification: the deployed optimized search completed in 31.540 s (30.65 s backend). Its 16 selected translations exactly match the frozen pre-optimization response; score differences are below 6e-8 from wire precision. Grid/recipe tests: 41 passed plus a randomized gap/outside-membership regression; prepared execution/recipe tests: 52 passed using fake actuators.
- Next open blocker: full preparation now reaches held-motion planning but rejects the 5 cm lift at IK waypoint 17. The fixed-orientation route and pregrasp retreat cannot satisfy the upright constraint. Implementing bounded yaw-only lift endpoints (yaw is free under upright) with the same nominal limits, sampled paths, upright and held-scene validation. Outcome pending; no actuator calls.
- Lift resolved: a bounded yaw-only endpoint with 20° turn and 4 cm inward translation yields 68 fully sampled upright lift waypoints. The next transfer-boundary rejection was the plate's own right-rim return, 3.14–3.64 mm from validated source points and 3–4 image columns outside SAM. Corrected non-iterative source seam ownership to at most 4 mm / 5 pixels, including the bounded image rectangle margin, with table/bottom and destination protection. Regression retains a 4.5 mm neighbor and depth-discontinuous/table/tray returns.
- Frozen full-motion replay now passes: selected candidate #7, position error 0.8 mm, orientation error 0.1°, 58.4 mm preplace standoff, 68 lift and 194 transfer waypoints, 1.563 s public placement-motion call. Reports `LIFT_TRANSFER_DESCENT_VALIDATED;OBJECT_UPRIGHT_VALIDATED` and the existing robot/attached-environment lift bypass markers. Transfer checks are active. Fresh complete queued preparation and cache handoff remain to be verified; no actuation.
- Fresh end-to-end verification: real LMStudio presence and bounding-box requests for plate and tray, EdgeSAM, HGGD, slot search and complete prepared motion succeed in 32.800 s. Slot API 21.562 s (same 16 translations as before optimization); complete-motion preparation 3.660 s, cached non-actuating readiness 0.079 s. Master status `PAUSED_AT_STEP_BOUNDARY`, phase `PREPARED`, zero completed physical stages, nine actual pending tasks through release and commit. Selected fitting exclusive group: cells (0,0), (0,1), (0,3), (0,4), (0,5), under the existing ordinary-group-before-large-square priority. No claim of a single-large-slot placement.
- Checks: VAMP 120 tests, task_pipeline 56 and placement_node 161 pass. Master functional/package checks pass after updating an outdated jaw-width expectation to the already selected +2 cm/open-minus-4 cm rule; its standalone flake8 check still reports accumulated style debt (578 diagnostics), not a planner regression. A workspace-root pytest invocation also scanned unrelated installed files through pep257; reran with package-scoped colcon instead. Master package run excluding flake8: 129 tests, zero failures, one skip. Physical tracking, contact and placement remain untested. Source/cache expiry remains 60 seconds; reprepare before later execution. Full profiles/results: `~/agx_piper_data/reports/plate_pick_place_profile.md`, `plate_pick_place_queue.json`, `plate_pick_place_check.json`, `plate_prepared_final/summary.json`.
- Clean-restart repeat also passed: 22.191 s end-to-end, 14.580 s slot API and 1.874 s complete-motion preparation. Second cached readiness check passed (0.010 s API). Two fresh full runs succeeded; physical execution remains false and all nine actuator/commit tasks remain pending. Regular scene-2 mock + reasoning launch are left running for RViz; temporary profiling/replay nodes and the abandoned diagnostic client are stopped. Latest report: `~/agx_piper_data/reports/plate_pick_place_repeat.json`, corresponding `_check.json`, and `plate_prepared_repeat/summary.json`.

### 2026-10-18 — Prefer one fitting slot before combinations

- Problem: ordinary-cell-first ranking reserved five cells for the plate while an 18.3 cm large square was available.
- Change: use occupied-slot count as the primary ranking key, then reserved area for equal counts. A single large square now precedes every multi-cell combination. Both grid trial order and heightmap/VAMP scores use the same key; exclusivity, measured geometry and motion validation are unchanged. Added regression cases for single-large versus smaller-area multi-slot groups and minimum-count combinations. Updated README and AGENTS to supersede the earlier priority.
- Verified: 33 grid tests and all 161 placement-package checks pass. Frozen plate search ranks 11 single-large-slot poses before five two-cell alternatives. Fresh real LMStudio/EdgeSAM/HGGD preparation selects only large cell (0,2), measured 18.3 x 18.3 cm, and completes upright lift/transfer/descent planning plus a separate non-actuating cache check. Nine execution/commit tasks remain paused before motion; no hardware moved. Full run 63.306 s under concurrent test/profile load; transfer 16.387 s / 629 returned waypoints to the farther slot, versus the previous nearer five-cell placement. Do not infer a ranking-only timing regression from these different destinations/load. Existing lift collision bypass markers remain active.
- Replay startup initially timed out waiting for synchronized input; restarting the overnight scene-2 mock restored capture. No perception synchronization criteria were relaxed. Artifacts: `~/agx_piper_data/reports/plate_single_slot_queue.json`, `plate_single_slot_check.json`, `plate_single_slot_final/summary.json` and `single_slot_frozen_search.log`. Stack and mock remain running for RViz.

### 2026-10-18 — Make selected grasp-angle logs explicit

- Problem: trial logs and RViz final poses did not clearly identify whether the final winner used an adjusted angle or the original fallback.
- Change: preserve `ANGLE_SELECTION=ADJUSTED target=10deg/30deg/50deg`, `ORIGINAL (already within 10deg)`, or `ORIGINAL (tuning unavailable or rejected)` in the accepted candidate status. Print `[GRASP ANGLE SELECTED]` after final HGGD ranking and `[PREPARED GRASP SELECTED]` for the actual complete-plan winner, including alternate-candidate recovery. No geometry, acceptance or collision rules changed.
- Validation: VAMP/master build and 43 prepared-session tests pass, including a recovery case confirming the log reports the actual second-candidate winner rather than the first-ranked grasp. All 120 VAMP package checks pass. Owned mock and reasoning stack were stopped before editing and remain stopped; no live inference or hardware actuation was rerun for this logging-only change.

### 2026-10-18 — Optimize plate slot search and held-object collision traversal

- Clean baseline, same scene-2 plate and single-large-slot policy: 29.393 s total, 13.273 s slot API, 10.373 s full preparation; transfer 9.733 s / 629 returned waypoints / 5,329 collision samples. This replaces the concurrent-load 63 s run as the comparison baseline.
- Slot optimization: evaluate the complete single-slot tier first using exact extent containment, retaining every yaw and ordinary support/obstacle/footprint check. Only after no fully valid single-slot result exists, run the unchanged combination search. Shared sub-cell and legacy grids retain their original search behavior. Frozen search: 1.670 s, 11 single-slot candidates (maximum budget 16); 34 grid regressions pass, including combination fallback.
- Collision optimization: add immutable exact-source bounds and a conservative 4 cm occupied-cell broad phase. Skip entire payload/block scans only when their transformed, outward-padded bounding box cannot touch any obstacle. Otherwise execute the original ordered 3 mm point checker, labels, departure and segment noise-budget logic. Compare the complete source snapshot before reuse; scene additions update the coarse index. Initial equivalence tests pass for rotated/negative-coordinate inputs, first-hit diagnostics, support/noise state and source changes. No sampling spacing, geometry, collision distance, upright rule or planner limits are reduced.
- Build and final isolated measurements pending. Physical execution disabled throughout.
- First optimized validation: 122 VAMP and 162 placement checks pass. On byte-identical frozen requests, transfer drops from 10.001 s to 4.013 s with exactly identical lift (68), transfer (629) and descent (18) waypoint vectors, same candidate #10 and 5,329 transfer samples. Three isolated full pipelines and cached readiness checks all pass: 12.669 / 12.240 / 12.641 s, versus 29.393 s clean baseline. Reports retained under `placement_optimization/phase1`.
- Additional optimization under validation: clip exact point-index queries to the occupied cell envelope, skipping only empty cell ranges. Preserve original predicate invocation/order even for points outside the metric query box but in an indexed cell. Added a targeted regression for that distinction. No point or collision decision is dropped.

- Final validation: all 123 VAMP and 162 placement-package checks pass (285 total). The final frozen transfer takes 3.671 s versus 10.001 s originally; byte-identical requests produce identical lift/transfer/descent waypoint vectors, same candidate and same 5,329 transfer samples. Final sequential full pipelines: 12.130 / 11.521 / 11.925 s (mean 11.859), 3/3 successful preparations and separate cache checks, no concurrent tests/profilers. Best slot API 1.652 s and complete preparation 4.311 s; clean total baseline 29.393 s (2.55x speedup / 60.8% reduction). No solver, geometry, clearance, path sampling or configured lift bypass changes.
- Full API timings and pending stages: `~/agx_piper_data/reports/placement_optimization/report.md`; individual `run_1.json`–`run_3.json`, `check_1.json`–`check_3.json` and frozen CDR responses are alongside it. Prior run contexts were cleared by the benchmark owner; the last complete queue remains paused with nine execution/commit tasks. Regular scene-2 mock and reasoning stack remain running with physical execution false. Journal baseline measurements are historical; do not use the earlier concurrent-load 63 s as the clean comparison.

### 2026-10-19 — Bowl placement orientation matching

- Live preparation found a grasp and 16 floor placements but every placement IK failed; closest logged endpoint errors were 7–8 degrees against 0.01 rad (~0.6 degree) matching. No actuator stage ran. This is not a VAMP path or collision rejection.
- Operator requested relaxed placement angles. Separate placement endpoint matching (5 degrees) from precise descent tracking (0.01 rad); descend to the accepted FK endpoint. Preserve the 3 degree object-upright bound, 1 mm target position, nominal commands and collision checks. Conservatively reject actual floor-slot footprints outside the searched footprint instead of reserving an unchecked larger footprint.
- Validation: VAMP/master build, launch Python syntax and all 114 VAMP checks pass, including two achieved-footprint regressions (5 degree circular yaw, outward translation/rectangular rotation rejection). Live bowl success remains unverified; a 5 degree matching limit does not automatically admit the earlier 7–8 degree solutions. User-owned live stack is left untouched and must restart to load the change. No hardware moved. Logs: `~/agx_piper_data/logs/placement_angle_relax_{build,tests,results}.log`.

### 2026-10-19 — Make bowl placement practical within its reserved slots

- Follow-up live calculation improved IK after the first relaxation but failed final pose/footprint checks and bowl-bottom contact near the floor; 8.273 s, no motion. The exact searched-footprint guard was stricter than the available slot and rejected useful adjusted poses.
- Replace that guard with achieved-geometry containment in the same frozen free slot group. Return actual FK object geometry while keeping its immutable source header; dry-run and commit validate the adjusted pose, reject occupied cells and preserve idempotence. Defaults: 5 mm placement TCP position, 10 degree matching, 5 degree upright bound, unchanged local descent tracking and collision checks. Aim 8 mm above the measured floor (3 mm release clearance plus 5 mm IK position allowance), retaining 1 mm descent tracking, to avoid precise depth-surface fitting. Lift IK retains 1 mm accuracy.
- Captured current live RGB-D/joints without actuation at `~/agx_piper_data/captures/bowl_placement_relax_20261019`; tests use replay on ROS domain 87 with physical execution disabled. User live stack remains separate. Build, regression and full preparation validation pending; do not treat these changes as a successful physical placement.

- Initial relaxed replay still rejected achieved endpoints after descent and residual table contact. Keep local descent at 1 mm instead of inheriting the 5 mm target-matching tolerance, include that bounded tracking error in endpoint acceptance, and budget the release height for IK uncertainty. Shared placement validation now dry-runs actual slot containment and tries another goal if an achieved pose leaves its slots; no actuator retry is involved. Validation pending.

- Next blocker: the 5 cm lift endpoint overlaps a remaining source-rim sample at pixel (217,102), 3.16 pixels outside both the semantic and validated masks. Do not assert it is noise or delete it. Explicit 6 cm lift passes the entire prepared pick/place pipeline in 6.237 s; 10 cm instead fails lift IK, so unconstrained higher lifting is inappropriate. Added shared, calculation-only handoff recovery at +1/+2 cm (10 cm total cap), preserving complete transfer collision checks and per-attempt height/timing. Focused 17 API tests pass, including recovery bounds. Three standard 5 cm-request preparations and immediate cache checks pending.

- Verified final outcome: the ordinary 5 cm-request queue prepares successfully 3/3 times, choosing the checked 6 cm lift handoff. Full pipeline times 5.854 / 6.393 / 5.357 s (mean 5.868), complete motion-preparation API 0.528 / 0.593 / 0.598 s. All three immediate non-actuating cached-readiness checks pass, with nine physical/commit tasks pending; no arm or gripper commands were issued. Existing explicit lift/approach bypass markers remain reported; transfer/descent checks stay active.
- Regression/build evidence: all affected packages build; 36 slot tests, 59 task/API tests, 43 prepared-session tests and 112 VAMP checks pass. Full Python package run passed 219 tests before adding the final two recovery tests. Report and individual API timings: `~/agx_piper_data/reports/bowl_relaxed_validation/report.md`, `run_1.json`–`run_3.json`, `check_1.json`–`check_3.json`. The earlier standalone 6 cm cache check expired before inspection; final immediate checks verify readiness within its actual lifetime. Hardware execution/contact remains untested.
- Cleanup: owned domain-87 replay and reasoning launch processes were stopped after validation. No live hardware nodes were controlled. Restart the live stack and prepare a fresh reviewed plan before operator execution.

### 2026-10-19 — Preserve multiple grasps for complete pick/place trials

- Blocker: fresh live frames still failed despite relaxed placement matching. Only one image-based grasp reached preparation; it could be graspable yet unable to lift clear or place with acceptable arm/finger geometry. Repeatedly relaxing the destination was not sufficient.
- Root cause: the HGGD quick pool was capped at one for single objects, and a second post-ranking truncation discarded alternatives. Retain up to six validated diverse HGGD grasps and keep the existing master-owned complete-plan loop (maximum eight candidates, 30-second between-attempt budget). No solver, placement tolerance, collision scope, actuator retry or message schema changes. Add per-candidate trial/rejection logs; cache and preview only the first complete winner.
- Validation so far: 45 prepared-session tests pass, including first-grasp placement rejection then second success without actuation, early stop before a third candidate, and complete-pool failure clearing old readiness. Current live RGB-D/joints captured read-only to `~/agx_piper_data/captures/bowl_multigrasp_20261019`; replay uses isolated ROS domain 87 with physical execution disabled. Build/runtime validation pending.

- Verified final result: HGGD retains six diverse grasps on the newly captured scene; 5/5 complete queued preparations and immediate cached-readiness checks pass. Requested lifts: 5 cm three times, 10 cm twice. Times 12.245 / 10.596 / 9.545 / 10.144 / 9.886 s; ordinary runs selected the top-ranked candidate after multi-candidate ranking. Nine physical/commit tasks stayed pending, with physical execution false.
- Real geometry recovery regression: separately test validated HGGD candidate 4 (placement failure) followed by validated candidate 3 (success) on the same immutable source. Pool [4,3] rejects 4 and accepts 3 with GRASP_CANDIDATES_TRIED=2 in 0.475 s; the cache readiness check passes. No synthetic pose or unchecked grasp was inserted. Build plus 112 VAMP and 45 prepared-session checks pass. Report: `~/agx_piper_data/reports/bowl_multigrasp_validation/report.md`, individual runs/checks and `actual_candidate_recovery.json`. Candidate-pool work increases runtime; this is not a hardware success guarantee. Existing explicit collision bypass markers remain disclosed.
- Cleanup: owned domain-87 replay and reasoning stack stopped after validation. User live stack/drivers were not stopped or actuated. Restart the live reasoning stack to load the new pool policy and prepare a fresh plan before reviewing/executing it.

### 2026-10-19 — Remove fixed lift request ceiling; review APF departure

- Blocker: all grasp trials were rejected as INVALID_BOUNDED_LIFT_REQUEST when the operator requested more than 10 cm, before motion planning. The same arbitrary interval existed in the engine, grasp-lift recipe and staged held-motion adapter; higher-clearance recovery also clipped to 10 cm.
- Change: accept finite positive requested heights without the fixed interval in all adapters, preserve the engine zero/default convention, and keep relative +1/+2 cm recovery bounded without a total-height cap. Nominal joint/workspace, orientation and configured collision checks remain. Guard Cartesian waypoint count before integer conversion/allocation to retain bounded resources for extremely large requests.
- APF review: current PlanAPF consumes only arm joint targets and checks robot spheres, without attached bowl geometry or object-upright constraints. Direct substitution would omit payload checks. Recommend payload-aware obstacle-directed departure before the existing VAMP transfer; this extension is not implemented in this change. Existing inward/retreat fallbacks remain.
- Validation pending build/runtime checks. User hardware has not been moved.

- Verified: build succeeds; 67 task/API tests, 61 staged/prepared-motion tests and 114 VAMP checks pass. Isolated domain-87 public plan_lift probes accept 0.001/0.15/0.30 m past height validation (then intentionally stop at a missing grasp-frame check); negative and infinite requests return INVALID_LIFT_REQUEST. This verifies request handling, not reachability or a successful 15/30 cm lift. No actuator calls. Tests cover bounded Cartesian allocation before IK and relative clearance recovery above 10 cm. Artifacts: `~/agx_piper_data/reports/lift_request_range.json`, `~/agx_piper_data/logs/lift_range_*.log`.
- The fixed request-range blocker is resolved. Payload-aware APF/escape planning remains unimplemented; no claim that neighboring-cup collision is solved. Owned probe stack stopped after validation; user live stack/drivers untouched and require restart to load the changed planner.

### 2026-10-19 — Shallow finger engagement and table clearance

- Blocker: the reviewed bowl grasp reached the rim but inserted a finger into the table. Read-only inspection of the original synchronized observation and accepted pool found the selected original-angle fallback about 7.2 mm below the measured table; one contact lay beyond the usable pad tip. Angle-adjusted routes had a final pad check, but original fallbacks did not.
- Change under validation: validate actual FK contact reach and open-finger/table clearance for measured HGGD routes, including original fallbacks. Try at most three contact-centered shallow targets (4/7/10 mm tip engagement), a maximum 20 mm TCP adjustment and 10 mm table-normal correction. Keep the same measured pair, strict seeded IK, nominal limits, palm/coarse collision, angle and pre-grasp checks. Unsafe originals cannot bypass this guard. Known-cloud benchmark generation and jaw-command policy are unchanged.
- Initial geometry checks: 25 alignment/shallow-geometry tests pass. Full build and isolated replay validation pending. Current user live stack and failed execution state have not been reset or actuated. Diagnostic source responses and measurements: `~/agx_piper_data/reports/shallow_grasp/`.
- Replay initially rejected all routes because the coarse collision world had already filtered out the table. Corrected the plane source to the immutable synchronized scene (excluding target-mask points), while preserving collision-world filtering. This is a distinct measurement input, not a collision bypass. Replay revalidation pending.

- Final verified result: after correcting the plane source, 3/3 full RGB-D/HGGD queued pick/place preparations and immediate non-actuating cache checks pass (10.852 / 9.154 / 9.380 s). Each retains six checked grasps and nine pending physical/commit stages. The selected last-run candidate has 3.97/5.86 mm fingertip engagement and 13.55 mm minimum open-finger/table clearance, independently recomputed from its returned pose and source observation. Build and 119 VAMP tests pass, including the actual old table-penetrating fallback regression.
- Report: `~/agx_piper_data/reports/shallow_grasp/report.md`; observations, geometry and run/cache reports stay outside source. Existing lift/approach collision bypass markers remain; physical accuracy/contact is untested. At the user's request the isolated domain-87 replay stack remains running. User live stack/state was not stopped, reset or actuated. The original coarse-plane rejection and endpoint pad/table bug are resolved in recorded replay; no all-scene physical guarantee is implied.

### 2026-10-19 — Payload-aware lift endpoint escape

- Reproduced scene-1 ordered two-bowl failure using real VLM/EdgeSAM/HGGD and frozen updated layout geometry: lower bowl fails lift IK, upper bowl succeeds. The layout used the legacy zero/10 cm lift default; baseline response is saved under `~/agx_piper_data/reports/apf_scene1/baseline.json`.
- Implementing a shared held-motion extension: explicit 5 cm layout lift, then check the predicted endpoint with full transfer geometry; if blocked, APF guides a bounded upright departure considering arm/fingers and attached source geometry. Complete escape must end in ordinary full clearance before transfer. Initial lift bypasses remain explicit. Escape has separate segment-wide contact/displacement/time budgets and cannot become an executable partial plan.
- Build and runtime validation pending. Physical drivers and prior stacks stopped at operator request; tests use recorded scene 1 in isolated domain 87 with physical execution disabled.
- First runtime result: both frozen bowl requests pass public plan_lift followed by plan_placement. The upper bowl uses the APF escape; the lower bowl succeeds with the shorter 5 cm lift. Ordered two-bowl layout also passes (21.751 s total, 16.171 s layout calculation). Final exact-point contact validation and dense joint-edge sampling are being added before repeated validation.
- Additional blocker: plan_lift inherited placement batching flags, so a request with no destination candidates could return NO_LIFT_FEASIBLE_GRASP without invoking the backend. Clear those placement-only controls in the shared lift adapter; regression test plus prepared-session tests pass (47 total). Full prepared pick/place now explicitly calls plan_lift before destination-motion planning; execution remains preplanned and non-retrying.
- Full prepared-motion checks exposed a separate lower-bowl case: completed source geometry has shallow endpoint contact, and broad APF influence from nearby clear surfaces could overpower the actual contact gradient. Prioritize the actual overlap gradient and add 1 mm backtracking plus bounded upward/inward alternatives; retain exact contact bounds and strict final clearance. The upper bowl's complete prepared plan/cache check already passes. Lower-bowl complete preparation is still under validation; ordered-layout success alone is not proof it is resolved.

- Resolved and verified: prioritize actual overlap gradient, allow bounded 25% temporary energy variation without cumulative growth, and retain the 0.5 mm additional-penetration/global contact/displacement limits. The escape must terminate in ordinary full clearance; no transfer/descent bypass was added. Exact prepared source requests now pass for both bowls with APF (0.164 / 0.128 s API round trip). Both complete prepared recipes and immediate cache checks pass (10.989 / 9.441 s), retaining nine unexecuted physical/commit stages each.
- Repeated final ordered scene-1 test: 3/3 full two-bowl layouts, 6/6 accepted placements, using updated immutable collision geometry and exclusive slots after the first bowl. End-to-end 22.812 / 21.774 / 22.348 s; layout calculation 17.617 / 16.951 / 17.165 s. APF engine recovery about 0.02–0.08 s; grasp generation/validation remains dominant. Independent actual-pose source-cloud comparison confirms no placed-cloud overlap (minimum separation 116.2 mm in the final layout).
- Build, 125 VAMP checks and 56 prepared/layout tests pass. Report and detailed API/path artifacts: `~/agx_piper_data/reports/apf_scene1/report.md`. Baseline used 10 cm lifts and the old helper's 8 mm packing tolerance; final uses explicit 5 cm/default 6 mm packing tolerance. Do not present this as a controlled speed benchmark. Initial lift bypasses and approach finger bypass remain explicit; no physical motion/contact was tested. Broad-scene 90% reliability is not established by three repeats of one scene.
- Mock scene 1 and the calculation-only stack remain running in domain 87 with final two-bowl RViz outputs. Real drivers and original stacks were stopped at the operator's request. The stale derived rgb_preview.png was not used: replay RGB was decoded from its actual CDR and verified to contain the beige cup, two bowls, elevated plate and tray.

### 2026-10-19 — Master-owned manipulation memory (verified on mock scene 1)

- Limitation: existing object hints, short grasp caches and placement grids were separate. A new command repeated destination inference, and calculation failure offered no simple source-linked grasp retry. Reusing old pixel masks would misassociate geometry when the wrist camera moves.
- Implementing master-owned bounded active object/destination records plus a minimal name/location directory. Resolve through shared presence/ground/segment APIs only when missing or depth consistency fails; retain original geometry provenance and grid identity. Retries use the same source-linked active data. Destination switching releases old active geometry and invalidates unexecuted prepared/grid state, without erasing hints or unknown physical state.
- Initial validation: default mock grasp/lift recipe with active-memory APIs passes (7.967 s); unit tests cover cached destination inference suppression, provenance, destination switching, bounded storage, failed-grasp retry and no false physical-placement update. VLM integration and repeated destination/grid reuse tests pending. No actuator commands.

- Resolved: shared memory/retry APIs now retain eight active sources, one destination/grid and 256 minimal hints. CLI and VLM resolution use the shared owner; master logs actual task transitions/pending tasks plus active names and base-link XYZ. Depth/view/calibration/age checks prevent blindly reusing pixel masks; hints alone never authorize motion.
- Full mock preparations: lower bowl 12.615 s, upper bowl 7.467 s, both paused before physical execution with nine pending motion/commit tasks. Tray resolution changed from 2.338 s initial inference to 0.133 s verified reuse; grid geometry/revision were identical. These are separate planning trials, not evidence that either object physically moved.
- Quick shared grasp recalculations on the retained upper-bowl source passed in 1.504 s and 1.260 s without another grounding request. Unit tests separately verify preserving a source after a geometric rejection. Switching to plate then back to tray released the old active destination/grid, retained its name/location hint and correctly performed fresh inference.
- Blockers fixed: mixed FLOAT32 XYZ/UINT8 bottom clouds required structured field reads; re-pinning a retained VAMP observation must remain idempotent after rolling-cache eviction (32.24 s old source recheck passed); CDR alignment padding caused a false STEP_ID_PAYLOAD_CONFLICT on identical typed requests. Typed-value fingerprints now ignore padding, include bindings and avoid reserializing large local requests. No actuator retry, collision relaxation or solver change was introduced.
- Validation: interfaces/consumers built; 185 focused Python regression tests passed, all 11 VAMP and three VLM CTest executables passed. New memory module/tests pass targeted flake8. Detailed artifacts and API timings are under ~/agx_piper_data/reports/active_memory_*. Physical commands were disabled in isolated ROS domain 87 throughout. Memory is process-local and resets on stack restart; this does not establish hardware execution reliability.

- Additional repeated cached-source trial reached grasp/search successfully but failed complete placement-motion preparation (unreachable IK goals and one approach self-collision). This remains a planner limitation; active object/destination records were retained. Kept semantic geometry immutable instead of replacing the reobservation source with a grasp-contact mask; regression test passes. A quick explicit RIM/BODY API recovery trial is recorded separately; memory success is not a guarantee that every motion plan exists.

- Explicit recovery verified: RIM recalculation still produced an unreachable complete plan; requesting BODY through the same active API, caching its candidates, searching and preparing again succeeded in 4.972 s with two grasp candidates tried, without another grounding request. The memory feature is validated; broad motion reliability remains unproven. Final two-bowl trial timings were 12.435/7.129 s and tray resolve 2.163/0.116 s (grounded/reused). Report: ~/agx_piper_data/reports/active_memory_report.md. Mock/stack remain in domain 87 with physical execution false.
