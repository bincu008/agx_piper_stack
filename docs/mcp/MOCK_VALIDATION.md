## 2026-11-10 — Task response improvements, MCP 3.6.2

Authenticated mock cup-to-tray preparation and exact validation passed (revision
6). Structured outcome PASSED_WITH_BYPASSES lists seven backend validation markers
and four bypasses with contract-derived configuration reasons. Execution remains
false; scene freshness and physical readiness are not confirmed. Running progress
identified verify_presence, then prepare_pick_place. Terminal current_step=null;
last_successful_step=validate_prepared_pick_place. describe_workflow count is 19,
matching the unchanged route, with repeated object/destination perception labelled.

Failure checks: superseded plan -> BLOCKED /
PREPARED_PLAN_SOURCE_OR_REVISION_CHANGED; unknown ID -> BLOCKED /
TASK_NOT_FOUND: use the task_id returned by submission; cancellation acknowledgement
CANCEL_PENDING with cancellation.status=CANCEL_REQUESTED -> CANCELLED /
CANCELLED_AT_STEP_BOUNDARY. Terminal cancellation clears the active step and retains
capture_observation as last success. Unknown-outcome/retry barriers remain covered
by isolated fixtures, without resolving or replaying uncertain actions.
270 package tests, including 159 MCP regression tests, pass. Fresh native rediscovery and post-reload exact
validation passed; original 54 diagnostics and all task input schemas are unchanged.
Receipts: ~/agx_piper_data/reports/task_response_update/result.json and
reconnect_result.json. Only MCP was reloaded; existing replay stack/tunnel/drivers
and physical gates were unchanged. These are simulated planning tests, not physics
or native Mac Nanobot execution tests.

## 2026-11-10 — Flat plan_camera_view schema, MCP 3.6.1

Native authenticated discovery on existing mock domain 89 confirms 62 tools and
flat plan_camera_view schema (no oneOf/anyOf/allOf); request_id/mode required.
Strict mode checks run before task creation. Direct missing-relative and mixed
mode MCP calls returned BLOCKED with no task ID. Valid +3-degree rightward
camera pitch returned NO_REACHABLE_CHANGE, achieved_fraction=0, moved_target=false,
endpoint_safe=false and zero joint changes. Target-name mode returned
NEXT_VIEW_PLANNED_EMPTY_HAND/full_path_validated=true. Identical request retry
retained its task. 154 MCP tests pass. Existing physical gates/planners unchanged.
Evidence: ~/agx_piper_data/reports/flat_camera_schema/result.json and native schema
snapshot in the same directory. These are calculation-only mock results, not
actuator or Mac Qwen/llama.cpp validation. Refresh Mac discovery/profile caches
and retry the model there; no model-side success is claimed here.

## 2026-11-10 — Task-level MCP facade (3.6.0, mock only)

Follow-up end-to-end authenticated MCP recheck: inspect_robot ->
prepare_manipulation(pick_place, beige cup -> dark rectangular tray, 5cm lift) ->
get_task_status -> validate_plan completed successfully. All 19 internal steps
passed; a new prepared plan revision 3 was issued and validated. Fresh capture,
both pins, same-context perception/grasp producers, cache confirmation, slot
search, complete motion preparation and final validator were recorded in
~/agx_piper_data/reports/task_interface/end_to_end_recheck/result.json.
No execution/actuation or occupancy commit was requested; operator-configured
collision bypass markers remain in the successful status.

Resolved the agent-bookkeeping blocker by adding eight opt-in task tools over the
existing PreparationGateway, typed StepStore queue/action jobs, reviewed workflow
examples and same private ledger. No second scheduler or actuator route was added.
`--task-tools` requires preparation_only; original 54 diagnostics remain unchanged.
Normal task schemas need no context/bindings/producer IDs. Durable aliases preserve
identical-submission identity, cancellation and unknown-outcome handling.

Authenticated TLS MCP on domain 89, bundled scene 1, real Qwen inference:
- observe_scene(beige cup, mask): COMPLETE / PRESENT, fresh source header and mask ref;
- grasp-cache: COMPLETE, six candidates, exact full-pool producer binding verified;
- complete cup/tray preparation: COMPLETE, cache confirmed, plan revision 2;
- exact validate_plan: passed existing validator; wrong revision rejected;
- target view: NEXT_VIEW_PLANNED_EMPTY_HAND, full path returned by existing planner;
- relative +2cm view: NO_REACHABLE_CHANGE, endpoint_safe=false (fallback, not movement);
- cancellation: CANCELLED_AT_STEP_BOUNDARY; reconnect read retained completed task.
Structured tool results were validated against native output schemas. Tests: 127
MCP fixture tests pass; colcon task_pipeline reports 238 tests, zero failures/errors;
affected-source lint and package build pass. Fixtures additionally cover stale/
wrong-context inputs, unknown perception, duplicates, timeout/disconnect recovery,
cancellation and independent physical-operation rejection.

Limits: this is replay planning, not physics, live-input or Mac-agent validation.
No physical commands were sent. Existing lift/withdrawal/final-approach collision
bypasses are preserved and exposed; acceptance is not full scene-collision proof.
Retained observations and successful historical receipts do not prove current
scene geometry. Batch remembered destination age is unavailable in the existing
API. Standalone grasp_lift positive route is fixture-tested; the authenticated
positive full-motion route was pick_place. Relative changed-view success remains
unverified. Existing drivers, mock stack, source guards and Qwen tunnel unchanged;
only MCP reloaded with --task-tools and the same mock ledger. See
[TASK_INTERFACE.md](TASK_INTERFACE.md) and exact schemas in [TASK_TOOLS.json](TASK_TOOLS.json).
Detailed receipts: ~/agx_piper_data/reports/task_interface/{stage1_result,
stage2_result,final_result,final_calls}.json and grasp_cache_diagnostics.json.

## 2026-11-10 — Mock-only MCP scene validation (3.5.1)

Operator confirmed room lighting was off and requested stopping live drivers and
switching to replay. Stopped domain-0 reasoning/MCP, Orbbec launch and AGX launch
after zero in-flight receipt checks. Verified no real camera/arm driver processes
remain. Left the SSH Qwen tunnel intact. Started bundled scene 1 (snapshots/current)
mock publisher and physical-gate-false reasoning in ROS_DOMAIN_ID=89.

Added explicit --require-mock-inputs, usable only with --preparation-only, a nonzero
ROS domain and dedicated mock execution ledger. It checks exactly one
/mock_hardware publisher per required RGB/cloud/joint/TCP topic and rejects mixed,
live-only, missing or domain-0 sources. Production --require-live-inputs behavior
is unchanged; live ledger/receipts are preserved. Mock mode constructs no physical
execution transport and independently rejects motion/gripper and physical generic
queues. All returned structured results carry SIMULATED=true, input_mode=mock_replay
and actual DDS provenance, with additive declared output fields. MCP remains TLS/
bearer authenticated on the existing private endpoint, version 3.5.1, 54 tools.

Authenticated mock tests using real remote Qwen inference passed:
- fresh capture -> perceive_scene(cup, tray): SCENE_PERCEPTION_READY observed=2/2;
  both ACTIVE_TARGET_GROUNDED.
- fresh capture -> configured inventory (bowl, cup, tray):
  SCENE_PERCEPTION_READY observed=3/3; bowl/cup grounded and tray memory reused.
Both responses success=true, MCP isError=false, and passed current MCP output and
typed ROS response schema validation. Invalid contexts and partial triplets were
rejected with actionable guidance. This verifies recorded perception/segmentation/
memory, not live perception, physics, grasp planning, actuator tracking or placement.
No motion/gripper/physical queue command was dispatched. Live-lighting limitation
remains historical; positive recorded-scene perception is now verified.

Build passed; 113 isolated MCP tests and 223 task_pipeline package tests passed,
zero failures/errors; affected server/bridge lint and whitespace checks passed.
Fixture checks cover exact mock ownership, wrong domain, mixed/live source
rejection, physical/direct-queue rejection and CLI refusal of the default live
ledger. Mock services remain running with dedicated mock-execution.sqlite;
production execution.sqlite was not reset. Logs: ~/agx_piper_data/logs/mock_scene_perception_*.
Evidence: ~/agx_piper_data/reports/mock_scene_perception and
~/agx_piper_data/reports/mock_scene_perception_inventory (live_result.json is an
actual MCP invocation against MOCK input; SIMULATED=true is explicit).

---

## 2026-11-10 — Scene Qwen JSON contract and schema-only workflow (3.5.0)

Confirmed server-side root cause of perceive_scene Extra data (char 29): raw Qwen
returned a fenced JSON array [{"key":"t0","value":true},{"key":"t1","value":true}].
parse_mapping sliced first '{' through last '}', discarding the enclosing array,
then json.loads rejected the resulting adjacent objects. Correct context/image/
cloud/transform bindings were not responsible. Exact historical raw replies are
saved outside source in reports/perceive_scene_contract/failing_qwen_responses.json.

Parser now consumes one complete JSON document (optionally one JSON fence),
normalizing either a keyed object or exact key/value records. Duplicate/extra/
missing keys, multiple documents, malformed values, and conflicting presence/
grounding are rejected as SCENE_INFERENCE_UNKNOWN before scene-memory updates.
Added a strict JSON system prompt and explicit root-object instruction. Regression
fixtures use the failing raw responses, ambiguity cases, same-source segmentation,
unchanged memory on failure, and conflicting presence/grounding.

get_workflow_example now supports workflow=perceive_scene and returns exact
capture->perception inputs with initial context_id="", the native returned path
result.structuredContent.context_id, same-context triplet bindings and fresh IDs.
Context/request descriptions and the capture context output field are explicit in
tool schemas. Perceive_scene requires nonempty destination_name; partial bound
triplets are rejected before dispatch with missing-field/sequence guidance.
Invalid-context recovery explains how to create/reuse the actual context.
Preparation_only/TLS/auth/ledger/physical guards are unchanged. Reloaded reasoning
and MCP only after idle empty-hand/gate=false and zero in-flight checks; drivers
and SSH tunnel remained untouched. Native MCP reports 3.5.0/54 tools.

Verified: master/task builds passed; package tests 242/215, zero errors/failures,
one existing master skip. Targeted 150 tests passed; affected MCP modules lint
and whitespace checks passed. The authenticated fresh capture->perceive_scene
call no longer returns Extra data. ACTUAL returned status:
SCENE_INFERENCE_UNKNOWN: PRESENCE_GROUNDING_CONFLICT, success=false, MCP isError=true.
Both structured MCP result and typed ROS response validate against current live
schemas. Invalid-context and missing-triplet negative probes were rejected with
actionable errors. No motion/gripper/physical queue operation was called.

OPEN positive-recognition blocker: the exact bound RGB image is nearly black
(mean grayscale 16.88/max 44 out of 255). Qwen said both targets false, then
returned identical nonnull boxes for cup and tray. These inconsistent answers are
correctly unknown, not successful scene grounding or evidence of absence. Useful
live recognition remains unverified until the scene is clearly illuminated.
Do not weaken the parser to accept this contradiction. No Nanobot/model-sequencing
success is claimed by direct MCP verification. Evidence under
~/agx_piper_data/reports/perceive_scene_contract: live_calls.json, live_result.json,
final_discovery_validation.json, native_tools_list.json, live_qwen_responses.json,
bound_capture_rgb.png. Logs: ~/agx_piper_data/logs/perceive_scene_contract_*.

---

## 2026-11-09 — Observation retention no longer depends on agent timing (3.4.2)

An external-agent audit attributed every pin failure to eviction/preparation-only
mode. Durable receipts confirm that test_pin_grasp_003 used the correct capture
header and still received the old undifferentiated missing-observation status.
The receipt does not establish delay, eviction or which component was absent;
the historical root cause remains unknown. Before changes, a fresh authenticated
capture->pin test succeeded, disproving the claim that the mode always fails.

Hardened shared MasterStepRuntime capture to attempt exact VAMP RGB/cloud/manual
FK*X retention before returning. Capture remains usable for vision if retention
fails, with GRASP_RETENTION_FAILED status instead of invented grasp readiness.
MCP exposes observation_retention.grasp_retained (true/false/null), source step
and exact confirmation binding. Explicit pins remain idempotent. The existing
bounded 50/100/200/400 ms delivery recovery now covers grasp pinning and known
missing-bundle replies, without recapture, timestamp substitution, or action retry.
VAMP failures expose requested stamp, cloud/image/transform presence, cache counts
and latest cloud stamp. Missing pin-header bindings are rejected before backend
dispatch. Corrected dependency guidance so placement pin is independent of grasp
pin. Descriptions/recovery no longer assert eviction from a missing bundle alone.

Verified: builds task_pipeline/master_node/vamp_node passed. Package test totals
213/228/165, zero errors/failures, one existing master skip. Targeted fixture suite
141 passed, including delayed-delivery recovery, retention failure reporting,
pre-dispatch binding rejection and independent pin dependencies. MCP lint and
patch whitespace checks pass; larger historical files retain pre-existing lint
issues. After checking idle empty hand/physical gate=false and zero in-flight
ledger operations, reloaded only reasoning stack and MCP with the same domain-0
Qwen profile, gate=false, TLS/authentication, ledger and live-input guard. Camera,
arm driver and SSH tunnel were untouched. Old transient contexts require fresh
capture after restart; durable receipts were not reset.

Authenticated live verification: five captures all confirmed automatic retention;
five explicit pins passed (including a delayed confirmation), and placement pin
passed. Missing/wrong-type bindings were rejected. A synthetic invalid source
probe on the live non-actuating pin API returned component diagnostics as expected.
Native MCP discovery reports 3.4.2/preparation_only/54 tools; motion/gripper tools
remain absent and physical queue rejection remains independently fixture-tested.
No robot action was dispatched and no VLM/HGGD success is claimed by these tests.
Mac Nanobot sequencing itself remains unverified. Reports:
`~/agx_piper_data/reports/mcp_pin_reliability/{live_after_fix,negative_source_diagnostics}.json`;
pre-fix report b7ac4cbf2a.json; logs `~/agx_piper_data/logs/pin_reliability_*`.

---

## 2026-11-09 — Native example-tool discovery (MCP 3.4.1)

The Mac all-calculations profile omitted get_workflow_example, although native
MCP discovery exposes it. Clarified the description to select it for requests
for an example workflow, tool ordering or bindings. Added list_functions
`discovery_notes` (additive output schema) distinguishing ROS reference catalog,
native MCP discovery, enabled allowlist and blocked APIs. Documented the exact
native/connector-qualified profile entries and discovery caches to refresh.

Verified over authenticated TLS on the running preparation_only server:
initialize reports 3.4.1; tools/list contains 54 tools including the helper;
exact input schema captured; one helper call returned the five-stage grasp
example, execution_performed=false, callable_in_current_mode=true. No returned
step was executed. Motion/gripper APIs are absent from native tools/list and
mode allowlist. Isolated MCP tests cover independent direct/queue physical
rejection; no physical rejection probe was sent to live ROS. Restarted MCP only
after confirming zero in-flight receipts/jobs, with the same ledger/TLS/auth,
require-live-inputs and preparation-only flags. Master launch retains its false
physical gate. Native Mac profile could not be edited/verified from Ubuntu.

Discovery snapshots (no credentials):
`~/agx_piper_data/reports/mcp_example_discovery/{initialize,native_tools_list,get_workflow_example_tool,get_workflow_example_result,list_functions_summary}.json`.

---

## 2026-11-09 — Smaller-model MCP examples (3.4.0)

The external Nanobot/Qwen3.5-9B agent guessed pin bindings and treated the legacy
LAYOUT status as a separate workflow. Model-facing descriptions totaled about
138,000 characters. Added local read-only `get_workflow_example` with exact
schema-checked grasp, cache and pick/place inputs, fresh proposed IDs, and explicit
context substitution. Reduced descriptions to about 26,000 characters; retained
full schemas/dependencies in metadata/resources. Added opt-in
`list_functions(summary_only=true)` and documented a nine-tool grasp allowlist.
The legacy pin-cache failure now explains the correct source-header recovery.

Verified: 97 isolated MCP tests and 205 task_pipeline tests passed; affected-file
lint passed. Authenticated TLS discovery reports version 3.4.0, preparation_only,
54 tools; the live helper returned the five-call grasp chain without executing it.
Summary discovery omitted the full catalog and unauthenticated access returned
401. Reloaded only MCP, retaining its ledger, TLS/auth, live-input guard and mode;
drivers and the tunnel were left intact. Initial reload briefly met the old
process's ledger lock; after that process exited, the same ledger opened normally.
No hardware command or controller change was issued.

Open limitation: model-only HTTP tool-selection probes against remote
Qwen/Qwen3.5-9B timed out (54-tool probe at 120 s, nine-tool probe at 60 s).
No returned tool call was dispatched. These fixtures verify API examples and
server behavior, not native Nanobot/Qwen end-to-end reliability. Nanobot must
refresh discovery and apply the updated adaptation instructions; its Mac code
was not changed here. Logs: `~/agx_piper_data/logs/mcp_small_model*` and
`~/agx_piper_data/logs/qwen_small_model*`.

---

# Mock MCP integration validation — 2026-11-02

This is real ROS replay + real RoboBrain/EdgeSAM/HGGD inference and planning,
with **no physical actuation or physics emulation**. Mock scene 1 is the packaged
`snapshots/current` capture. All development ROS nodes run on isolated domain 89;
no camera/arm driver or controller was started. Physical execution is false.
The authenticated private MCP endpoint now reads this mock domain.

## Verified results

- All seven exposed MCP tools returned typed structured results against actual
  ROS services, including a prepared bowl session, destination slots and the
  queued tasks. Initial missing-plan failure was also preserved as a failure.
- Bowl: full calculation + cached handoff succeeded twice, pausing before `look`
  and every physical stage. Runs: 23.286 s and 19.690 s.
  Final prepare_pick_place: 2.997 s.
  Result: `PAUSED_AT_STEP_BOUNDARY`, phase `PREPARED`, physical gate false.
  Planned pending stages: look, pregrasp, gripper_open, approach, gripper_close,
  lift, transfer, descent, release, withdrawal, retreat (plus queued slot commit
  and post-retreat observation). None of those physical APIs was dispatched.
- Actual perception/calculation calls used the existing ROS recipe; these have
  NOT been added as MCP tools. Remote mutation tools remain blocked.
- The cup passed perception, HGGD and placement candidate search, but its full
  route did NOT pass (43.945 s). Candidate failures include
  lift IK, bounded APF escape and placement/transfer reachability. This is an
  OPEN motion-planning blocker, not successful placement or a transport failure.
- 33 isolated regressions passed: 18 MCP fixtures plus 15 placement callback/
  visualization regressions. task_pipeline and placement_node builds passed.

## Bugs found and fixed

1. `prepare_held_scene` referenced point_cloud2 without importing it after module
   consolidation. The callback crashed the placement process. Added a local ROS
   cloud-helper import; an organized source/destination regression now exercises
   payload, environment and support-cloud conversion. Actual retry reached
   placement candidate search instead of losing the backend.
2. Bottom-completion debug code published completed_message twice, including an
   unconditional use when the preview publisher was disabled. Removed the
   redundant publish. Regression verifies the no-preview branch retains its
   bottom cache; final replay emits a normal footprint result without that error.

## Final bowl API runtime (seconds, inclusive scopes)

Child backend/engine times overlap these rows; do not add overlapping scopes.

| # | Actual public ROS API | Runtime s |
|---|---|---:|
| 1 | `capture_observation` | 0.144 |
| 2 | `ensure_scene_memory` | 0.012 |
| 3 | `require_target_view` | 0.084 |
| 4 | `capture_observation` | 1.186 |
| 5 | `require_target_view` | 0.061 |
| 6 | `pin_grasp_observation` | 0.002 |
| 7 | `pin_placement_observation` | 0.002 |
| 8 | `refresh_active_target` | 2.148 |
| 9 | `prepare_grasp_input` | 0.004 |
| 10 | `calculate_active_grasps` | 10.754 |
| 11 | `update_object` | 0.002 |
| 12 | `select_grasp` | 0.004 |
| 13 | `cache_grasp_plan` | 0.007 |
| 14 | `resolve_target` | 0.135 |
| 15 | `find_active_placements` | 1.336 |
| 16 | `plan_next_best_view` | 0.266 |
| 17 | `prepare_pick_place` | 2.997 |
| 18 | `get_execution_state` | 0.002 |
| 19 | `validate_prepared_pick_place` | 0.014 |

The final run reuses the valid destination/memory inventory. These two runs are
not a reliability benchmark and do not validate feedback tracking or real contact.
Configured lift/withdrawal and final-approach collision bypasses remain active and
explicit in the result. Successful preparation is not fully scene-collision-
validated motion or permission to execute on hardware.

## Reproduce without motion

In separate terminals, source ROS and the workspace, then use domain 89:

```bash
export ROS_DOMAIN_ID=89
ros2 launch mock_hardware_node mock_hardware.launch.py
```

```bash
export ROS_DOMAIN_ID=89
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio grasp_candidate_backend:=hggd \
  placement_backend:=heightmap pipeline_profiling:=true \
  allow_physical_grasp_execution:=false
```

```bash
export ROS_DOMAIN_ID=89
ros2 run task_pipeline pipeline_task \
  --object 'small white bowl at bottom left, grasp its rim' \
  --destination 'dark rectangular tray' --queue-pick-place --lift-distance 0.05 \
  --output "$HOME/agx_piper_data/reports/mcp_mock_bowl_pipeline.json"
```

The queue must pause before `look`; do not run cached physical execution. Use the
MCP startup in README with ROS_DOMAIN_ID=89, then read the seven tools. For
get_step_context use the returned context_id and include_results=false; image/
cloud CDR payloads may exceed the transport's intentional 1-MiB response bound.

Reports remain outside source:
`~/agx_piper_data/reports/mcp_mock_validation.json`,
`mcp_mock_bowl_final.json`, `mcp_mock_cup_pipeline_retry.json`.
Logs: `~/agx_piper_data/logs/mcp_mock_*.log`, `mcp_held_scene_regression.log`,
`mcp_mock_regressions.log`. Prepared plans expire normally; this recorded success
is not a current executable readiness token. Native macOS connectivity is still
for the Nanobot agent/operator to verify.

## MCP 3.0.0 agent-composed pipeline (2026-11-03)

On isolated ROS domain 89, recorded scene 1 and physical gate false, a client used
TLS-authenticated MCP calls for the complete lower-left rim-bowl preparation:
capture/pin → RoboBrain presence+grounding → EdgeSAM → new store_segmented_target
for bowl and tray → HGGD → object registration/ranking/cache → floor-slot search →
view plan → complete pick/place preparation → cached handoff validation. No arm or
gripper commands were sent. No simulated inference replaced the live RoboBrain
or HGGD calls. Raw camera inputs were recorded mock RGB-D/joint feedback.

The measured 26-call workflow took about 22.0s including inspection. HGGD 10.84s,
placement search 0.94s, complete preparation 2.63s, final validation 0.05s. It selected
one of three tried grasp candidates. Final status included ALL_MOTION_PLANNED,
LIFT_TRANSFER_DESCENT_VALIDATED, OBJECT_UPRIGHT_VALIDATED and
CACHED_HANDOFF_VALIDATED_NO_ACTUATION. Existing explicit lift/withdrawal/approach
bypasses were reported unchanged; this is not complete collision validation.

Evidence: ~/agx_piper_data/reports/mcp_agent_pipeline_validation.json,
mcp_tool_catalog.json, and logs/mcp_agent_pipeline_validation.log. These are
calculation results, not actuator/contact/tracking verification. The external
Nanobot native client still needs its own adaptation and verification.

Final repeated run after the admission checks: **20.23s / 26 calls**, HGGD 11.03s,
placement search .80s, full preparation 2.75s. Package tests reported task_pipeline
148 and master_node 222 tests, zero errors/failures, one existing master skip.
Real TLS calculation queue also completed across HTTP disconnect/reconnect with
same-ID duplicate receipts and stale-revision rejection. Generic dispatch rejected
physical function injection before any backend call. Summary:
~/agx_piper_data/reports/mcp_integration_summary.json.

## Shared spatial map — MCP 3.1.0 (2026-11-03)

Mock scene 1/domain 89, physical gate false: authenticated MCP returned all 16
30cm cells and eight currently observed cells. get_spatial_context mapped the
remembered lower-left bowl to x0_y2_z0; check_target_view returned READY with the
same target_cell_id. plan_next_best_view rejected one unreachable cube target,
then produced NEXT_VIEW_PLANNED_EMPTY_HAND for x1_y1_z0. No movement was sent.

Builds passed; package tests: vision_node 15, task_pipeline 148, master_node 224,
zero errors/failures, one existing skip. Regression cases cover observed empty
space, occluded cells, invalid rays, minimum sample count, boundary hits and map
membership reuse. The new ros-pipeline://spatial-map resource and 62-tool catalog
were verified over TLS. Evidence: ~/agx_piper_data/reports/spatial_map_validation.json
and spatial_integration_summary.json. This does not validate actual camera motion,
physical tracking or complete coverage of any cube.

## Dependency-aware discovery verification — 2026-11-09

MCP 3.3.0 preserves all existing tool input names/schemas and mode enforcement.
The dependency guide/graph and additive recovery output are documentation, not
a second state machine or a dispatch grant.

- **Synthetic fixtures:** 80 MCP tests passed, including full generated grasp/lift
  and pick/place preparation examples through the real gateway and StepStore
  with fake engines. Verified binding field types/schemas, absent prepared/cache/
  active-state hints, SDK malformed-input rejection, disabled physical dispatch,
  per-step asynchronous failure guidance and preserved unknown-state blocking.
- **Package:** all 188 task_pipeline tests passed; task_pipeline build and targeted
  lint/whitespace checks passed. The legacy test_mcp.py already has unrelated
  baseline lint findings; only its two resource-count expectations changed.
- **Running authenticated TLS server:** initialize reported version 3.3.0; all 53
  preparation-only tools carried metadata. Graph returned 63 nodes/104 labelled
  edges, including disabled actions. Human guide served successfully; a request
  without a token returned HTTP 401. Existing ledger had no incomplete requests
  at restart. Only MCP was restarted with identical mode/domain/auth/TLS/ledger.
- **Limits:** production verification used discovery/resources/list_functions only.
  No live inference, motion planning, actuator commands, motor enables, controller
  or driver changes were performed. Fixture success does not establish physical
  readiness.

Logs: `~/agx_piper_data/logs/mcp_dependency_{tests,package_tests,tls_validation}.log`.

## Cache recovery verification — 2026-11-09, MCP 3.3.1

- **Fixtures:** 87 MCP tests / 195 full task_pipeline tests passed. Added
  same-context recovery after a failed cache getter, exact-field rejection of
  all four invented agent bindings, compound active-grasp documentation and
  history-based cache error guidance. No physical backend exists in fixtures.
- **Live calculation-only check:** With the operator-restored Qwen tunnel, used
  fresh synchronized inputs in the existing MCP context. Presence, grounding,
  segmentation, grasp inputs/calculation/selection, caching and a new cache
  getter all succeeded. One candidate was returned; context revision 49 -> 61.
  No motion/gripper API was called. Cache expiry remains 60 seconds.
- **TLS/receipt check:** Version 3.3.1 serves 53 preparation-only tools. Re-reading
  the original failed request returned its original failure without replaying a
  backend, with added context evidence and 12 proposed same-context recovery
  requests. Old receipts are historical evidence, not a fresh cache/state read.
- Logs: `~/agx_piper_data/logs/mcp_{live_cache_recovery,cache_guidance_tls_validation,cache_recovery_tests,cache_recovery_package_tests}.log`.

## Observation source recovery — 2026-11-09, MCP 3.3.2

All 197 package tests / 89 MCP tests passed. Added omitted-name admission and
source-cache guidance regressions. Live calculation-only verification used the
same affected context: capture -> immediate pin -> resolve cup on that supplied
triplet -> calculate_active_grasps(target_description=cup, bindings=[]) -> select.
The capture and active record headers matched; six valid candidates were returned
and selected after a 41-second resolve. No motion/gripper API was invoked.
TLS discovery confirmed 53 preparation tools and the named-target requirement;
a missing name was rejected before dispatch without changing context revision.
No cache bounds, safety gates, drivers or controller settings changed.
