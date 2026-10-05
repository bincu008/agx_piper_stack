# Git change audit

Audit date: 2026-10-05. Comparison base: `dad5a99` (`baseline code`).

## Why the change list is large

Before this documentation/cleanup pass, Git showed **31 modified tracked files
and 103 new files**. The repository has one baseline commit; the working tree
contains the accumulated grounding, simulation, packing, motion-profiling and
shared-API work from multiple tasks. It is not a diff of only the last API change.
There are no intermediate commits from which to reliably assign every line to
an individual request.

The original 103 new files break down as follows:

| Kind | Files | Why they exist |
|---|---:|---|
| ROS interface definitions | 31 | Services, messages and action shared by independently running nodes; ROSIDL requires separate definition files |
| Implementation sources | 26 | New pipeline package, master microstep classes, mock replay, benchmark adapters and planner helpers |
| Regression tests | 15 | API handoffs, cancellation/idempotence, geometry, grounding and motion checks |
| Launch/build/entrypoint/config metadata | 14 | Install and launch the new nodes; register interfaces/tests and inference profiles |
| Documentation | 8 | Seven feature guides and the mock README |
| Recorded mock fixture | 9 | Seven serialized ROS inputs, manifest and optional RGB preview |

`git diff --stat` shows only changes to tracked files. Use
`git status --short --untracked-files=all` to include new files. Directory rows
such as `?? src/task_pipeline/` expand to many real source/interface files.

## Changes grouped by launched component

Counts below are from before this documentation pass; the new audit/guide edits
and ignored RGB preview change the final totals.

| Group | Modified / new | Main files and behavior |
|---|---:|---|
| Shared schemas (`my_actions`) | 3 / 31 | New microstep/result/timing messages, `RunPipelineSteps` action, image/grasp/cache/state/search/execution services; `PlanPlacement.srv` gains frozen-scene inputs, grasp alternatives, motion paths and timings; CMake/package register dependencies |
| Shared API (`task_pipeline`) | 0 / 14 | `registry.py`, `node.py`, `context.py`, `recipes.py`, common placement policy and timing helpers; installed dispatcher/recipe CLI, package metadata and two test modules |
| Master (`master_node`) | 5 / 6 | `step_runtime.py`, `execution_api.py`, `grasp_stages.py`; execution facade refactor and shared client routing; launch/profiles, state handling, install/dependency wiring and tests |
| VLM (`vlm_node`) | 4 / 4 | `vlm_reasoning_node.cpp`: optimized requests, image APIs, multiple-object templates, common API/master-cache routing; strict parsing/template headers and tests; standalone multithread executor; removal of an unnecessary EdgeSAM image copy |
| Placement/benchmark (`placement_node`) | 8 / 12 | `node.py`, `packing.py`, extraction/attachment helpers; edge/stack search and joint-grasp validation; new benchmark/fixture/typed adapters, entrypoints/launch and tests; runtime node renamed `placement_node` |
| Planner (`vamp_node`) | 7 / 9 | `vamp_planner_server.cpp`: fixed-attachment placement collision/descent/lift/transfer validation, candidate alternatives, early stopping and profiling; geometry/index/payload/ranking helpers and tests; HGGD depth/orientation handling; RViz updates |
| Acquisition (`vision_node`) | 2 / 0 | `active_vision_server.cpp`: clock-domain/arrival association, bounded RGB receipt fallback and fewer copies; CMake removes duplicate dependency declarations |
| Mock (`mock_hardware_node`) | 0 / 20 | Replay/capture/serialization code, package/launch/test/docs and recorded input fixture |
| Root documentation | 2 / 7 | README/code map and feature/tuning guides |

The largest tracked changes are in VAMP, VLM and placement node implementations.
They include substantial placement/motion and controller behavior, not just API
renaming. The new `task_pipeline` package and API definitions depend on matching
changes in their backend nodes and launch files. Removing only those backend
changes would leave callers, schemas or installation inconsistent.

Some small diffs are performance/housekeeping carryovers: `world_state.py` removes
a second deep copy after constructing an already-copied queue; `edge_sam_onnx.cpp`
and ActiveVision retain immutable reference-counted image data instead of making
extra clones; Vision CMake removes duplicate `find_package` declarations. These
are not required simply to rename API endpoints. This audit identifies their
purpose; it does not claim that every accumulated change is necessary for the
latest refactor.

## Generated output and removable inspection files

The audit found **no tracked build/install/log files, Python bytecode, object
files or recorded benchmark output** in the pending changes. Build/install/log
and Python cache directories were already ignored. Most local inference and
benchmark reports are under home-directory output folders, outside this Git tree.

This pass expands `.gitignore` for CTest/coverage/object by-products, backups,
root-level run-output folders (`benchmark_results`, `debug_output`,
`runtime_artifacts`), default rosbag folders and the derived mock RGB preview.
Required runtime libraries, robot meshes, model assets, source images and CDR
fixtures are not globally ignored by extension.

| File/group | Decision | Reason |
|---|---|---|
| `mock_hardware_node/snapshots/current/rgb_preview.png` | Ignore; leave the local copy available | Derived inspection image; replay reads `rgb_image.cdr`. It can be regenerated; grounding documentation now explains export from the CDR input |
| Seven `.cdr` files plus `manifest.json` in the bundled snapshot | Keep | Runtime inputs installed with the mock package; a fresh clone needs them for replay. They total about 7.07 MB |
| `vamp_node/rviz/grasp_debug.rviz` and RViz launch | Keep | Reusable operator visualization required to inspect grasp and placement; the name “debug” does not make it generated output |
| `box_benchmark.py`, fixture, entrypoint and launch | Keep | Requested simulation/benchmark functionality, using the shared APIs |
| `scripts/benchmark_grounding` and `grounding.py` | Keep as optional utility | Reusable inference-latency tool; not launched by the arm stack and not used for deployed pipeline latency measurements |
| `test/test_layout_debug.py` and other new tests | Keep | Regression source, including pinned-frame and visualization behavior; not a saved debug session |
| Run reports, saved cloud/path arrays, CDR RPC dumps, PNG debug captures, bags | Exclude from Git | Generated run evidence, not source. Use a home output directory or an ignored runtime directory |
| Historical tuning/performance guides | Keep as supporting documentation | Not runtime dependencies. Current launch instructions are in README/code map/API guide; old timings must not be presented as the current deployed latency |

No planner, solver, hardware source or required replay message was deleted or
reverted in this pass. The optional preview remains on disk but will not be
staged by a normal `git add`. Tests and source utilities were not removed merely
to make the file count smaller.

## Review boundaries before publishing

These groups are useful for reviewing the accumulated diff. They are not
independent folder-only commits: shared definitions and consumers cross groups.

1. **Grounding and inference configuration:** request format/parse rules, endpoint
   profiles, image association, streaming/timeout/authentication behavior.
2. **Mock and benchmark:** recorded inputs, known-cloud grasp mode, sequential
   scene updates, failure continuation, blocking trigger and generated output.
3. **Placement and motion:** height-map edge/stack behavior, fixed grasp attachment,
   collision checks, descent/lift/transfer paths, early stopping and timing.
4. **API/master orchestration:** typed catalog, result store, bindings, idempotence,
   queue cancellation, execution-stage ordering and physical gates.
5. **Documentation and output hygiene:** current commands, source hierarchy,
   fixture/output distinction and ignore rules.

Acquisition now pairs cloud/joints by local arrival and has a bounded RGB receipt
fallback for different clock domains. HGGD depth snapping/orientation retention,
placement early termination, and master execution-stage refactoring also change
behavior. These are real functional diffs that need review, not removable debug
noise. The mock checks do not validate real camera synchronization, contact or CAN
execution. Existing master lint debt is recorded in `docs/VALIDATION.md`.

For this documentation/ignore pass, check Markdown links, launch arguments,
fixture loading, whitespace and ignore behavior. The preceding implementation
checks are recorded in `docs/STANDARD_PIPELINE_API.md`.

Completed checks: all local links in the seven edited/new guides resolve;
documented parameters match all four launch files' `--show-args`; the catalog
contains the documented 54 services and one action; prompt tags match the VLM
parser; the eight-file replay fixture loads in a temporary directory without
the preview PNG; the snapshot helper test and `git diff --check` pass. Required
CDR/API/RViz source remains eligible for Git, while sample run outputs and the
preview are ignored. No runtime algorithms or hardware behavior changed in
this documentation/ignore pass.

## Inspect and stage

The placement package and Python module were subsequently moved to
`src/placement_node/placement_node`, and backend endpoints to `/placement/*`.
Launch, dependency, import, registry, RViz and documentation references were
updated together. The obsolete executable/class aliases were removed. Before
staging, Git can show the directory move as deletions and new files; rename
detection applies when the corresponding paths are staged. The numerical
placement algorithm is unchanged by this rename.

```bash
git status --short --untracked-files=all
git diff --stat
git diff -- src/vamp_node/src/vamp_planner_server.cpp
git diff -- src/vision_node/src/active_vision_server.cpp
git diff -- src/master_node/master_node/piper_master_control.py
git ls-files --others --exclude-standard
git diff --check
```

After review, stage the coupled source/interfaces/tests/docs and inspect
`git diff --cached --stat` before committing. Do not use `git clean` or discard
new files to reduce the count: required untracked package code and ROS schemas
are part of the implementation. This pass does not create commits or push to
`origin`; the working tree remains available for review.
