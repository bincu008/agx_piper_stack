# Adaptive floor grid

The default upright `floor_slots` policy uses square main slots with side lengths
from 6 cm to 12.5 cm. Rows run along the destination's longest axis.

- Prefer two rows. Scale their squares down only when necessary, to at least 6 cm.
- If two 6 cm rows do not fit, use one row and recalculate its square size.
- Three or more rows require full 12.5 cm squares; extra rows never shrink them.
- Calculate row gaps from the remaining depth, shared between rows. Gaps may be
  wide when two maximum-size rows fit but a third does not.
- Along each row, make as many squares as fit. A remaining strip at least 6 cm
  wide becomes a rectangular slot; smaller strips are distributed as column gaps.
  If only one column fits, an undersized strip stays unused.

After keeping at least two ordinary squares, reserve at most one larger square
at the spare end of a usable rectangle. Its side is the ordinary side multiplied
by `floor_slot_large_square_scale` (default 1.5; 0 disables it). It may replace
prospective ordinary slots; it never overlaps retained slots or extends beyond
the usable region. If it cannot fit, keep the ordinary subdivision. Subdivide
leftovers into squares and remainder rectangles. Existing reservations remain
exclusive and commit only after successful validation.

Slots therefore stay square except at remainder strips. Gaps can be zero where
squares exactly fit; they are not controlled by a percentage. Regions narrower
than 6 cm cannot form a slot. Frozen IDs, reservations and collision checks remain.

The algorithm finds the largest inscribed usable rectangle on a conservative
2 mm raster, then adds non-overlapping rectangular extensions. It redistributes
undersized remainder strips into gaps where possible. Rectangle decomposition is bounded;
this is a practical packing heuristic, not a proof of globally optimal packing.

`/pipeline/search_placements` and `find_placements` use the same implementation.
Search first ranks ordinary cell groups by total reserved area, including
combinations of smaller cells. Groups touching the large square are fallback
candidates after ordinary groups. Equal-area groups retain spatial/yaw diversity
and geometry preferences; VAMP scores preserve the same area-tier ordering.
This is built-in behavior, not a separate selectable policy.
An object reserves its validated touched-cell group, and
`/pipeline/commit_floor_slots` commits only after successful placement validation.
Whole-cell exclusivity remains the default. Frozen geometry and cell IDs survive
occupied-surface updates; supplied scene obstacles block cells. A changed
destination/configuration creates a separate grid. Independent trials must reset
reservations. RViz `/placement/floor_slots` shows free, occupied and selected cells.

Launch parameters:

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio allow_physical_grasp_execution:=false \
  floor_slot_adaptive:=true floor_slot_max_size_m:=0.125 \
  floor_slot_min_size_m:=0.06 floor_slot_large_square_scale:=1.5
```

Start the mock replay separately, then calculate a bowl placement:

```bash
ros2 run task_pipeline pipeline_task \
  --object 'small white bowl at bottom left' \
  --destination 'dark rectangular tray' --placement-policy floor_slots
```

`floor_slot_adaptive:=false` restores the legacy fixed 12.5 cm square / 0.5 cm gap
builder. Support, containment, source association and collision checks remain.

## Earlier rectangular-grid validation — 2026-10-10

These results record the previous all-rectangular sizing rule, superseded by
the square-row rule above.

The live mock recipe used separate presence and grounding requests for the bowl
and tray, EdgeSAM segmentation, HGGD grasp calculation, placement pose validation
and reservation. It completed successfully; physical execution was disabled.
This recipe does not validate the full held-object lift/transfer path.

| Measurement | Seconds |
| --- | ---: |
| Complete client recipe | 8.908 |
| `find_placements` | 1.140 |
| `plan_placement` | 0.191 |
| `commit_floor_slots` | 0.006 |
| Occupied-surface backend search | 0.279 |
| Occupied-surface API client wall time | 0.293 |

The final two measurements overlap; do not add them. Client wall time spans
request submission through receipt of the complete response, excluding subsequent
RViz marker waiting.

The tray produced nine slots in three rows, with approximate dimensions
12.5×8.81 cm, 12.5×8.81 cm and 9.02×8.81 cm per row. The bowl reserved two slots.
Its 11,034 placed visible points were sent back to public search with the tray
cloud. All cell IDs and boundaries stayed unchanged, seven cells remained free,
and returned candidates avoided occupied cells. Exactly three free cells were
verified separately using a four-cell grid with one occupied cell.

181 placement/pipeline tests passed, covering adaptive row sizing, concave
extensions, serialization, exclusive reservation and occupied-heightmap feedback.
Run captures and CDR dumps are outside the source tree:
`~/adaptive_bowl.json`, `~/adaptive_bowl_results/`,
`~/adaptive_occupied_surface_result.json`, and `~/adaptive_grid_tests.log`.

## Square-row validation — 2026-10-10

The corrected live grid has two rows: four 12.5×12.5 cm squares and two
9.2×12.5 cm remainder rectangles. The row gap is approximately 1.60 cm;
column gaps are zero because the residual strip is used as a rectangle.
The public bowl recipe completed successfully in 8.929 s.
Placement search took 1.096 s, pose validation 0.150 s,
and reservation commit 0.006 s. These are pose-calculation
results, not full held-object transfer validation. All 183 placement/pipeline
tests passed, including 1/2/3/4-row selection, square preservation, rectangular
remainders, dynamic gaps and occupied-cell exclusion. Live captures are
`~/square_bowl_grid.json` and `~/square_grid_dimensions.json`.

## Large-square validation — 2026-10-11

The default factor 1.5 produced one 18.7 cm large square on the captured tray,
while preserving two 12.5 cm ordinary squares. The live lower-bowl calculation
and reservation succeeded: 11.791 s end to end, 1.042 s placement search and
0.115 s placement IK. These are pose-calculation timings, not a full transfer
benchmark. 227 functional tests passed, including variable scaling, no-fit
fallback, geometry separation and exclusive large-slot reservations.

Change future grid sizing with:

```bash
ros2 param set /placement_node floor_slot_large_square_scale 1.6
```

The factor applies to side length, not area. A larger factor cannot force a
square beyond the usable spare region. A configuration change creates separate
cached geometry; reset reservations before an independent comparison.

## Least-area ordered validation — 2026-10-11

Two bowls followed by the cup passed the mock full-motion calculation. The
lower bowl reserved one ordinary cell (156.25 cm²), the upper bowl an ordinary
plus small cell (224.31 cm²), and the cup the large fallback cell (351.56 cm²).
The small strip was too narrow for the cup. Request completion took 20.486 s;
layout calculation took 13.812 s. Search times were 0.859/0.559/0.365 s.
The requested lift attached-environment bypass remained enabled; transfer and
descent checks remained active. No hardware execution occurred. 229 functional
tests passed, including ordinary combinations before large-square fallback.

## RViz working-area comparison

The same `/placement/floor_slots` MarkerArray now includes white outlines of
estimated usable working rectangles (largest rectangle plus extensions) and
a magenta grid envelope. Labels show dimensions in centimetres. Both come from
the frozen grid geometry used by search, not from manually entered tray sizes.
An envelope can include gaps and unused areas; it does not mean every enclosed
point is occupied. These outlines expose the planner's estimated interior for
comparison with the physical tray; they are not independently calibrated
measurements of its inner walls. Slot and occupancy colors remain unchanged.

### Tight fits and elevated plates

Slot matching uses the complete measured footprint at continuous positions.
The nearest heightmap sample does not move the proposed pose or expand its
inventory reservation. A footprint fully contained in one physical cell reserves
only that cell, even when a neighboring leftover has a wide gap territory.
Yaw samples follow the frozen grid's axes; exclusive
matching tries centered and ±1 cm phases inside the available group. Matched
cell groups are reused through ranking. Actual accepted FK poses are still
rechecked on commit, and collision/motion validation remains mandatory.

A grounded plate/dish mask is trimmed against its measured body plane to avoid
including a lower stand. A thin elevated body may use its observed underside
plane and projected hull; a tall body may not use this shortcut. No assumed
plate diameter is substituted for measured geometry. Destination floor-contact
labels use the fitted surface normal rather than a horizontal height cut.

Validated ordered prompt (mock replay; physical execution disabled):

```text
[PLAN_UPRIGHT] place striped white plate surface at upper left excluding its stand below, small white bowl at bottom left, small white bowl in middle left above the other bowl onto dark rectangular tray
```

Trial settings: `floor_slot_max_size_m=0.125`,
`floor_slot_large_square_scale=1.5`, `packing_clearance_m=0.002`,
`packing_contact_tolerance_m=0.008`, `perception_measurement_ratio=1.036`,
`aligned_zero_pose_yaw_limit_degrees=90`. Independent trials reset reservations.
Timing and the lift-check exception are recorded in the engineering journal.
