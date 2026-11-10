# Current workspace / Nanobot implementation handoff

Recorded 2026-11-10 on branch `optimization`. Latest checkpoint: authenticated
mock complete pick/place preparation now passes; see the final section below. This is an implementation checkpoint,
not a declaration of physical readiness. The earlier “60%” was a rough scope
estimate, not a measured completion metric.

## Read first

Read `AGENTS.md`, `docs/mcp/WORKFLOW.md`, `docs/mcp/AGENT_PIPELINE.md`,
`docs/SIMPLE_ROBOT_INTERFACE_PROPOSAL.md`, `docs/mcp/WORKSPACE_TOOLS.md`, and the
2026-11-10 current-workspace entry in `docs/ENGINEERING_JOURNAL.md`.
Inspect current processes and Git changes before doing anything. There are many
accumulated changes: do not discard, stage wholesale, or assume all belong to this
increment. Do not change live drivers, activate hardware, or bypass unknown state.

## Implemented stage

- Master-owned latest synchronized workspace observation, independent source and
  arrival freshness, measured feedback association, calibration/provenance checks.
  Original manual ActiveVision FK*X remains authoritative; no computational TF.
- Typed `GetCurrentWorkspace` service and public catalog registration.
- Opt-in MCP 3.7.0 extension: `get_camera_image`, `list_objects`, `check_freshness`.
  Current-task defaults for status/cancellation and compact camera direction labels.
- Existing 54 diagnostic and eight task tools retained. With the workspace
  extension there are 65 native tools. The small-model profile selects these nine:
  `inspect_robot`, `get_camera_image`, `check_freshness`, `list_objects`,
  `observe_scene`, `plan_camera_view`, `prepare_manipulation`, `get_task_status`,
  `cancel_task`.
- Mounted Mac source has opt-in `workspace-tasks`, model-facing ID hiding,
  server-request ID generation before durable recording, private original receipts,
  and actual image blocks for the model. Legacy profiles remain available.

Important semantics: memory is not current visibility; TTL does not establish
execution readiness. Image/object are the only new freshness scopes. Object checks
still include legacy 60-second verification freshness. Existing execution cache
has not yet migrated to the proposed single 600-second pool.

## Main files

Ubuntu additions: `master_node/current_workspace.py`, `workspace_freshness.py`,
`my_actions/srv/GetCurrentWorkspace.srv`, `task_pipeline/provenance.py`,
`task_pipeline/mcp/workspace.py` (package paths under `src/`). Integration edits
include `piper_master_control.py`, `manipulation_memory.py`, registry, MCP bridge,
server/tasks, package CMake and tests. Exact contracts are in
`docs/mcp/WORKSPACE_TOOLS.json` and `TASK_TOOLS.json`.

Mac root: `/media/psf/Documents-mac/NanoBot`. Changed sources include
`mcp_profiles.py`, `workspace_arguments.py`, `camera_visuals.py`,
`task_checkpoint.py`, `launch_nanobot.py`, workspace instructions, README,
launcher help and `tests/test_workspace_adapters.py`. Credentials/default endpoint
configuration were not changed. Native Mac execution has NOT been verified.

## Actual verification

- task_pipeline: **251 passed**.
- master_node: **285 passed, 1 skipped**.
- portable Mac adapter tests: **8 passed** under Ubuntu Python.
- Authenticated isolated domain-89 MCP returned 65 native schemas. ImageContent,
  source/timestamps, observe_scene cup mask, object freshness, camera labels and
  omitted-task-ID lookup passed.
- Direct Qwen/Qwen3.5-9B selected get_camera_image with `{}` and received actual
  mock pixels using the image assembly helper. This is not native Mac integration.
- Full pick/place recheck crashed at ROS `prepare_grasp_input` response conversion:
  NumPy seed integers failed `PyLong_Check`. `execution_api.py` now explicitly
  converts pixel-box and seed coordinates to Python ints. Actual serialization
  regression and isolated domain-90 ROS roundtrip passed (`GRASP_INPUT_READY`).
  Full pick/place after this fix has NOT passed a new complete recheck.

Evidence outside repository:
`~/agx_piper_data/reports/workspace_tools/` contains native_tools.json,
calls/result.json, qwen_visual_result.json, preparation_calls/result.json and
seed_serialization_fixture.json. Logs are under `~/agx_piper_data/logs/`, including
workspace_tools_pipeline_package.log and workspace_tools_master_package.log.
Temporary verification scripts were `/tmp/verify_workspace_tools.py`,
`/tmp/verify_workspace_qwen.py`, `/tmp/verify_workspace_preparation.py`; inspect
before reuse and do not assume /tmp survives reboot.

## Blocking unknown task and supported reconciliation

The interrupted mock task was `task_4c518ac7d9022810acb1039fe978d0cf`, context
`mcp_7a2a19cb419d4485bd8b4ef22efbf174`, job
`task_4c518ac7d9022810acb1039fe978d0cf_run`.
Its original mock ledger contained `ACTION_OUTCOME_UNKNOWN` and a preparation
barrier. Supported administrative reconciliation has now been implemented:
`task_pipeline.mcp.reconcile` (`reconcile_ledger`) and `PreparationGateway.reconcile_preparation`.
It strictly requires `confirmed_operator=True`, records an audit event in the ledger,
transitions the unknown job to `ABORTED_BY_OPERATOR_RECONCILIATION`, and clears
`preparation_barrier` only when no unresolved unknown jobs remain. Verified by 5 tests
in `test_mcp_reconcile.py`.

The unified 600-second grasp pool is now implemented across `grasp_cache.py`,
`execution_api.py`, and `prepared_grasp_lift.py`, with single pool lifecycle tracking
(`claim_current_pool`, `invalidate_current_pool`, `get_current_pool_summary`) and
extended `check_freshness` tool schemas for `grasp_pool` and `plan`.

## Process/deployment state at handoff

Live MCP on 8765 was stopped as requested. Existing domain-0 drivers/reasoning and
Qwen tunnel were left unchanged; the old live master does not have the new backend.
Our temporary domain-89 mock launchers received SIGINT at wrap-up. Reinspect
processes/listening ports rather than trusting historical PIDs. Do not restart or
replace live reasoning merely to test this work.

## Reproduce automated checks

From the workspace root, source absolute paths:

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/agx_piper_stack/install/setup.bash"
source "$HOME/agx_piper_data/builds/current_workspace/install/setup.bash"
export PYTHONPATH="$HOME/agx_piper_data/deps/ros_mcp:$PYTHONPATH"
python3 -m pytest src/task_pipeline/test -q
python3 -m pytest src/master_node/test -q
python3 -m unittest discover -s /media/psf/Documents-mac/NanoBot/tests \
  -p test_workspace_adapters.py
 git diff --check
```

The targeted native integer regression is in `test_execution_api.py`:
`test_grasp_seed_numpy_box_serializes_with_native_python_integer_fields`.
Run tests in package directories if local test configuration requires it, keeping
all sourced paths absolute. These commands do not authorize hardware execution.

If rebuilding, preserve the separate deployment overlay:

```bash
colcon --log-base "$HOME/agx_piper_data/builds/current_workspace/log" build \
  --build-base "$HOME/agx_piper_data/builds/current_workspace/build" \
  --install-base "$HOME/agx_piper_data/builds/current_workspace/install" \
  --symlink-install --packages-select my_actions task_pipeline master_node \
  --allow-overriding my_actions task_pipeline master_node
```

For functional replay, use domain89 and separate terminals sourced as above:

```bash
export ROS_DOMAIN_ID=89
ros2 launch mock_hardware_node mock_hardware.launch.py
# In another terminal:
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=qwen_tunnel grasp_candidate_backend:=hggd \
  placement_backend:=heightmap allow_physical_grasp_execution:=false
```

Qwen tunnel must already be available at 127.0.0.1:18092/v1. Do not request/store a
password or disturb an existing tunnel. The MCP startup command, TLS/token paths
and source guard are in `docs/mcp/WORKSPACE_TOOLS.md`; use its original mock ledger
and port8766. Read-only verification can inspect the unknown barrier; further
calculation admission must remain blocked until supported reconciliation.
Never remove `--preparation-only` or substitute live sources in these tests.

## Next implementation order

1. Finish Mac discovery/profile wiring: `workspace-tasks` currently reads common
   legacy discovery snapshots. Add dedicated authenticated preparation-only
   workspace discovery snapshots and fail clearly when stale/missing; preserve
   legacy profile snapshots. Verify native Mac provider/image/checkpoint integration
   on macOS; Ubuntu portable tests are insufficient.
2. Establish supported reconciliation for the interrupted calculation without
   weakening unknown barriers. Then rerun complete mock grasp/lift/pick-place and
   report the first real blocker, not merely queue admission.
3. Implement proposal's single current grasp pool, 600-second expiry and explicit
   consume/invalidate semantics, preserving durable identities internally and
   held-object/occupancy state. Add pool freshness only when it is real.
4. Integrate semantic freshness and explicit fresh-perception recovery; refine
   stable object aliases without merging distinct bowls. Retained data must never
   silently become fresh evidence.
5. Separately implement/review any remaining held-placement or physical facades.
   They are NOT part of the verified current surface; keep physical gates disabled
   and never test arm/gripper execution automatically.

Update proposal and engineering journal after each meaningful verified milestone.

## Review continuation — 2026-11-10 (latest checkpoint)

Reviewed the follow-on implementation. Its pool tracking and reconciliation were
not yet sufficient: pool age mixed ROS stamp and monotonic epochs, multiple
public object caches remained, invalidation did not revoke cache access, and
plan metadata ignored replaced source versions. Corrected these issues with
single public pool replacement, source-age admission using the ROS clock,
monotonic lifetime accounting, shorter requested TTLs, exact-owner consumption,
cache invalidation, and plan/source-version freshness matching. Tests cover stale
source admission, shorter lifetimes, replacement, invalidation and superseded plans.
Motion-triggered/calibration-triggered pool invalidation integration remains to
be verified; the presence of an invalidation helper alone is not that guarantee.

Reconciliation now uses one implementation for CLI and gateway, a SQLite
transaction and the existing exclusive ledger lock for offline CLI use. Rejects
running jobs, wrong selectors, unmatched barriers and physical uncertainty;
preserves unknown status prefixes, historical task receipts and unresolved
request barriers. **No real ledger was reconciled in this review.** Operator
confirmation was requested before touching the interrupted saved task.

Mac dedicated discovery wiring is now implemented in `workspace_discovery.py`,
`tests/discover_workspace_mcp.py` and `mcp_profiles.py`. The workspace profile no
longer falls back to legacy discovery files. Pair identity, preparation mode,
source mode, tool presence and object schemas are validated. New native Mac
snapshot refresh command, after the guarded mock server is running:

```bash
cd /path/to/NanoBot
python3 tests/discover_workspace_mcp.py \
  --url https://10.211.55.3:8766/mcp \
  --expect-source mock_replay_publishers
# Add --ca-file /path/to/trusted-ca.pem if needed; never disable TLS verification.
```

This writes only `results/mcp-workspace-tools.json` and
`results/mcp-workspace-status-list_functions.json`. Legacy snapshots remain
untouched. Discovery makes only initialization, tools/list and read-only
list_functions calls. **Actual native Mac refresh/launcher remains unverified.**
Portable Mac workspace tests now total 11 passing tests.

Isolated overlay rebuild of task_pipeline/master_node passed. Pipeline package
suite passed 261 tests. Run master tests from its package directory (the PEP257
test scans the current directory; running it from repository root wrongly scans
unrelated generated/vendor files). Review logs are under
`~/agx_piper_data/logs/workspace_review_*`. Existing live processes were not changed.

## Resume verification — latest 2026-11-10 checkpoint

Implemented automatic revocation of unconsumed pool readiness from the existing
CurrentWorkspace feedback subscriptions. Measured joint movement >0.005 rad,
TCP movement >2 mm, orientation change beyond the existing stationary threshold,
or manual calibration changes invalidate the pool. Stationary feedback never
renews it. Feedback also purges expired unconsumed geometry after the source-based
600-second (or shorter requested) lifetime. Active consumed execution sessions
remain governed by their existing safeguards; this is not a collision monitor or
actuator stop. Baselines describe feedback at pool creation, not an invented
original-source pose. Freshness is still not a safety certificate.

Verification: master package 293 passed/1 skipped; pipeline package 262 passed;
portable Mac workspace tests 12 passed. Latest expiry-event regression also passed
in the 19-test execution API subset. Two-package isolated overlay build passed.
Registered reconciliation regressions in the package CMake test suite.

Authenticated MOCK MCP 3.7.0 on https://10.211.55.3:8766/mcp returned 65 tools,
no native motion/gripper dispatch tools, CURRENT_OBSERVATION_READY with an actual
image, valid typed output schemas and explicit SIMULATED/mock_replay provenance.
Pool/plan freshness returned false with the historical barrier; saved task status
remains ACTION_OUTCOME_UNKNOWN / UNKNOWN. No new calculation was submitted.
Evidence: ~/agx_piper_data/reports/workspace_resume/ and workspace_resume_* logs.

Dedicated workspace snapshots were refreshed from these actual authenticated
Ubuntu calls into the mounted Mac results directory. This does NOT verify native
Mac execution. Workspace profile generation now also rejects discovery/config URL
mismatches: snapshots for mock port8766 must not silently configure live port8765.
No Mac default config, credentials or endpoint were changed.

At this checkpoint, domain89 mock publisher, mock reasoning and MCP8766 are running
for guarded read-only inspection (preparation_only, physical gate false, original
mock ledger/barrier intact). Only the temporary mock reasoning process was
restarted to load final code. Live domain0 reasoning/drivers/tunnel were untouched;
live MCP remains stopped. Full preparation still awaits explicit operator
confirmation for local reconciliation of the saved interrupted mock calculation.
The prior confirmation question remains unanswered; do not infer approval from a
request to resume unrelated implementation.


## Root's two features completed — 2026-11-10

**Stale-image recovery:** `get_camera_image` may request one bounded sensing-only
scan when stale, idle, stationary and supplied with current feedback and verified
provenance. Waits for a completed synchronized observation; acknowledgement alone
never authorizes returning old pixels. No VLM, task contexts or actuation are
introduced for this fallback. Metadata-only/freshness reads do not trigger it.
Sensor-refresh annotations and schemas were updated; typed response regressions
cover success and missing completion. Actual mock image reads pass.

**Complete mock preparation integration:** canonicalize owned NextBestView results
through ROS serialization before exact ownership comparison, avoiding float32
radius rounding failures. Added real serialization regressions at .35/.45/.60m.
Task pick/place routes now store object and destination masks/clouds/transform from
the exact capture before planning; failures stop dependents. Redundant update_object
from grasp seed is removed from this route because contact points are not new
object-center observations. Public tool names/inputs and diagnostic recipes remain
compatible. Cache examples/descriptions now specify600s (shorter lifetimes apply).

The prior crashed mock calculation was reconciled locally after the operator's
instruction to proceed with the disclosed remaining work. Original ledger and
receipts preserved; audit is reconciliation:1794309239_5ddadd6d. This was not an
actuator retry or physical reset. Fresh tasks were then submitted normally.

Actual authenticated results:
- task_2cc0e1eadf3112d3d40b079aa3f00b49: COMPLETED, complete prepared plan and
  validation PASSED_WITH_BYPASSES after wire-format fix.
- task_4018198a617433593ab72bc03928e96d: COMPLETED after the final memory-route
  integration; validate_prepared_pick_place succeeded. Active cup and tray exist;
  look status CURRENT_VIEW_ALREADY_SUITABLE; pool and plan freshness true.
- Both execution_allowed=false. Existing bypasses for approach fingers/environment,
  lift robot/environment and attached-object/environment, and withdrawal
  robot/environment are reported. No physical execution occurred.

Evidence: ~/agx_piper_data/reports/workspace_recovery/ and workspace_memory/;
reconciliation audit also saved under workspace_resume/reconciliation.json.
Latest schemas/discovery snapshots refreshed in the dedicated Mac workspace files.
Native Mac launch/provider integration remains unverified.

Tests: task_pipeline final suite265 passed; next-view/prepared-placement targeted
suite45 passed; workspace/typed-image subset41 passed. Full master suite had
311 passes/1 skip and one PEP257 failure in Gemini-owned world_state.py (D213 at
parse_object_description). Gemini files arrived during testing; root has not edited
those files or claimed their combined review complete. After Gemini's handoff,
review its identity changes and rerun the whole master suite. No subagent is running.

Root-owned files for this increment: current_workspace.py, next_best_view.py,
their tests, task_pipeline/mcp/tasks.py and its tests, workspace tool definitions,
cache examples/descriptions, GetCurrentWorkspace.srv comments and documentation.
Gemini owns world_state.py and test_world_state.py until handoff. Mock stack/MCP8766
remain running in domain89; live processes untouched. Further physical facades and
native Mac validation remain distinct future work, not verified by mock preparation.

## Identity review complete — latest checkpoint

Gemini's identity feature has now been reviewed and corrected; the prior PEP257
failure is resolved. See OBJECT_IDENTITY_REVIEW.md. Full checks: master324 passed,
1 existing skip; task_pipeline265 passed. Typed memory/view regression subset88
passed. Authenticated mock capture/pin and body/rim resolution share obj_001 with
one canonical active/world cup. Code now preserves grasp-region instructions when
normalizing identity and propagates canonical keys through memory/view lookup.
Only domain89 mock reasoning/MCP were restarted, keeping their ledger intact;
live processes untouched. Mock MCP8766 remains preparation_only with no physical
dispatch. Native Mac integration and proposed physical facades remain outside
this verified checkpoint. Existing prepared plans from before the mock master
restart are historical receipts, not current executable readiness.

## Guarded controls implementation — latest checkpoint

Added three explicit control facades, source/docs in task_pipeline/mcp/controls.py
and docs/mcp/CONTROL_TOOLS.md (exact schemas CONTROL_TOOLS.json). Existing prepared
execution now supports the reviewed grasp/lift handoff as well as complete
pick/place. A fixed opening-only master wrapper uses the existing GripperCommand
schema and executor; registry name open_empty_hand. Controls share the durable
ledger and lock, require exact client-reviewed plan references and preserve unknown
barriers and contact ordering. No generic queues contain physical calls.

Main tests passed: master330/1 existing skip, pipeline282, portable Mac14. Synthetic
full pick/place and grasp/lift execution tests cannot reach hardware. Actual
TLS-authenticated mock server3.8.0:68 native cards,65 callable calculation/read APIs,
3 disabled control cards; each control dispatch rejected before ROS. Reports under
~/agx_piper_data/reports/control_tools/. Mock8766/domain89 remains running in
preparation_only, master physical flag false. Live processes unchanged.

Mac source adds opt-in workspace-control (nine workspace tools plus three controls),
strict separate live-control discovery, private reviewed-plan injection, and
600s tool timeout for that profile. Default config/endpoints/credentials untouched;
native Mac/provider execution remains unverified. Read CONTROL_TOOLS.md before
any future operator-supervised live deployment. Hardware execution was not tested
or enabled. No further test expansion after the operator requested conserving usage.
