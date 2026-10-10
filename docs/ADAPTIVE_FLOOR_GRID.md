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

Slots therefore stay square except at remainder strips. Gaps can be zero where
squares exactly fit; they are not controlled by a percentage. Regions narrower
than 6 cm cannot form a slot. Frozen IDs, reservations and collision checks remain.

The algorithm finds the largest inscribed usable rectangle on a conservative
2 mm raster, then adds non-overlapping rectangular extensions. It redistributes
undersized remainder strips into gaps where possible. Rectangle decomposition is bounded;
this is a practical packing heuristic, not a proof of globally optimal packing.

`/pipeline/search_placements` and `find_placements` use the same implementation.
An object reserves the smallest validated touched-cell group, and
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
  floor_slot_min_size_m:=0.06
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
