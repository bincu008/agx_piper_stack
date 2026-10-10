# Checkpoint change audit

Reviewed 2026-10-10 against `83b26a9` (`API split`), the latest committed
checkpoint. Earlier counts of 134/147 files referred to the preceding API work;
they are not the current pending diff.

## Current inventory

The reviewed checkpoint contains **98 paths: 61 modified, 25 new, 12 deleted**.
The audit update itself adds one modified documentation path to the initial
97-path inventory. Counts include source, tests, documentation and replay inputs.
After staging, Git recognizes four renames: **94 change records** (61 modified,
21 added, eight deleted, four renamed). The known-cloud generator/header/test
and shared diversity helper retain existing code instead of duplicating it.

| Group | Paths | Purpose |
| --- | ---: | --- |
| VAMP | 23 | HGGD-only deployment, internal known-cloud proposals, contact/attachment checks, lift/transfer/descent planning, geometry helpers and regression tests |
| Placement | 14 | Adaptive square floor grid, exclusive reservations, heightmap support/stacking, scene association, benchmark/API transport and tests |
| Shared API | 14 | Typed bindings and colocated dispatch, tracing, shared grasp-input preparation, recipe changes and tests |
| Mock replay | 14 | Updated captured scene/joints, raw depth replay, synchronized capture and fixture tests |
| Shared ROS interfaces | 8 | API trace, immutable cloud preparation, slot commit and matching caller/response schemas |
| Master | 6 | Colocated API dispatcher, typed step runtime, shared input preparation, launch parameters and tests |
| Retired GPD vendor | 6 | Deletions required by removal of the obsolete engine/dependency |
| Vision / VLM | 2 | Requested gripper mask correction; EdgeSAM-only masks, grid policy and validated slot commits |
| Documentation | 11 | Current commands, API/code map, policies, journal, runtime reports and this audit |

The numerical counts are a review inventory, not a claim that all files are
runtime dependencies. Tests verify behavior; historical reports preserve tuning
evidence. Cross-node interfaces and their consumers must be committed together.

## What can be removed

No pending tracked build output, Python cache, benchmark cloud dump, debug image,
RPC capture or rosbag was found. Generated output is already ignored or stored
under home-directory run folders. No further implementation files were deleted
solely to reduce the count.

The twelve existing deletions remove the requested analytical/GPD grasp paths,
GPD adapter/test and vendor package. Keep those deletions. Their replacement
known-cloud generator is simulation-only; deployed proposals remain HGGD.

Keep the eight `.cdr` replay messages and `manifest.json` under
`mock_hardware_node/snapshots/current`. They contain the newly requested RGB,
registered cloud, raw depth, current joints and feedback. They are installed
runtime inputs, not generated build output. The derived `rgb_preview.png`
remains ignored. RViz launch/configuration and benchmark/debug regression tests
are reusable source, not disposable session logs.

`FLOOR_SLOTS_PROFILE.md` and `GRID_SPACE_OPTIMIZATION.md` are optional historical
reports for users, not runtime code. Retain them because the journal and policy
documentation reference the experiments. Current behavior is described by README,
AGENTS, STANDARD_PIPELINE_API and ADAPTIVE_FLOOR_GRID; historical measurements
must not be presented as current defaults.

## Cleanup and optimization in this review

The square-grid matcher formerly scanned the entire dense footprint for every
contained one-cell trial. A bounding rectangle now proves one-cell membership
when it lies strictly within that cell and cannot intersect any other cell's
gap territory. Boundary cases, multi-cell footprints and overlapping territories
retain the original point-level checks. Per-call cell indices avoid repeated
linear lookups. No geometry, support, collision or candidate budget was relaxed.
The new grid module/test were cleaned to pass their Flake8 and docstring checks.
Touched master imports and launch argument wrapping were cleaned without
reformatting the controller; total master lint findings dropped below HEAD.

432 old/new comparisons produced identical ordered candidates/cell sets, covering
recorded and randomized footprints, rotations, reservations, phases and clearance.
Alternating five-repeat matching benchmark: median **4.230 s → 0.629 s (6.73×)**
for three recorded footprints over 72 yaw angles. This is matching time, not
complete API or pipeline latency. Raw results: `~/checkpoint_grid_benchmark.json`.

The live ordered lower-bowl → upper-bowl → cup calculation accepted all three in
**18.124 s**, layout **12.629 s**. Support-contact tolerance was overridden to
8 mm; its source default remains 6 mm. Lift-only attached/environment bypass was
active and explicitly marked; transfer/descent checks remained active. Results
are in `~/checkpoint_layout.json` and `~/checkpoint_layout_capture/`.

## Checks and limits

- 223 functional tests passed, one skipped, across placement, pipeline, master
  and mock packages. Package lint checks were evaluated separately; running
  lint tests from the workspace root incorrectly scans build/vendor directories.
- The new grid module and test pass Flake8 and docstring checks.
- Master Flake8 reports 508 findings versus 511 in the committed version;
  master docstring checks pass. This review does not claim repository-wide lint
  cleanliness or mechanically rewrite the existing controller.
- Latest VAMP build and nine C++ test executables passed during the preceding
  default-bypass change. This review changes Python grid matching only.
- Live three-object calculation passed; saved motion responses confirm success
  and `LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED` for every object.
- `git diff --check`, fixture eligibility and source/output separation checked.

The requested lift bypass is now a deliberate default; set
`placement_bypass_attached_collision_during_lift:=false` to restore lift checking.
Physical execution remains disabled during development. Current testing does
not validate hardware contact/CAN execution, a collision-free lift, or plate
placement after these three objects. Computational transforms still use manual
FK/X only. The free-space redivision overlay used a home-directory visualization
script; it is a preview, not a newly installed production API.

Compare future performance using the same completed work, settings and timing
scopes. Warm starts, CPU load and remote inference can change latency; the
18.124 s rerun does not isolate the full effect of the matching optimization.
The engineering journal preserves both successful runtime summaries.
