# Proposal: one current workspace and a small robot interface

Status: staged implementation, 2026-11-10. See CURRENT_WORKSPACE_HANDOFF.md for
verified scope and remaining work; physical facades are not yet implemented.
Historical baseline from the initial design review:
The running live stack, drivers, gates, TLS/authentication and Nanobot configuration
were not changed for this review. MCP discovery confirmed 3.6.2, 62 tools,
preparation_only and live_driver_publishers. This proposal does not enable motion.

## 1. Decision

Expose one current workspace to the human and agent: current observation, object
inventory, active object, active destination, one grasp pool, one prepared plan
and one active task. The server owns all prerequisite sequencing and freshness
checks. The model never supplies a ROS context, producer binding, source stamp,
joint configuration, execution reference or plan revision.

Keep immutable, privately identified snapshots underneath that workspace. A
background frame must not change the image/cloud/transform halfway through a
calculation. A transport receipt must still identify whether an operation ran.
These are internal correctness mechanisms, not concepts the user must manage.

Do not implement a second planner, ROS queue, obstacle map or world-state database
inside MCP. Extend the current master-owned state, StepStore and gateway ledger.
One current workspace is an interface over those owners, not their replacement.

## 2. Findings from the current implementation

| Area | What exists | Gap or implication |
|---|---|---|
| Public task tools | Eight task tools already hide contexts/bindings from normal inputs. | The old 54 diagnostic tools should not be the normal agent profile. Keep them available for maintenance. |
| Background capture | Master scan_routine_loop requests a scan about once per second in IDLE and after returning to IDLE. ActiveVision synchronizes registered depth/joints, RGB and manual FK*X. | The periodic trigger is not proof a valid capture completed; IDLE alone is not measured stability. It does not rerun named-object VLM grounding. |
| Current image | /active_vision/observation_image is published; calculation APIs hold image references. | No task tool currently sends the actual scene image to the agent. |
| Mac image handling | Installed Nanobot MCPToolWrapper saves ImageContent as artifact files and returns text paths. Its image-result instructions suggest the message tool. | Returning an MCP image alone will not make this Qwen agent see its pixels. The task profile also disables built-in message tools. Both server and Mac adapter need an image path. |
| Robot/spatial state | get_arm_state, get_spatial_context and get_spatial_map provide joints/TCP, optical camera axes, shared cells, scan times and object hints. | inspect_robot does not yet combine these into a simple gaze/visibility summary. |
| Object identity | WorldStateManager.associate_observation matches case-insensitive exact names within 5 cm. ManipulationMemory keys normalize case/whitespace only. | A changed description or displacement beyond the gate can create another obj_NNN. This is not proof that two real objects exist. Conversely two real cups must not be merged by name alone. |
| Object state update | update_object and store_segmented_target already update world state/memory. | Reuse them; do not add an unrestricted agent-written XYZ/cloud database. |
| Freshness | Active target verification has a 60-second check; a mapped-view reuse path has 300 seconds; grasp/prepared cache defaults/max are 900 seconds; NBV records have 60 seconds. Feedback/synchronization checks have separate tighter limits. | Unify policy ownership and report distinct ages. A new RGB frame must not renew an old mask, object geometry or plan. |
| Grasp cache | Master stores per-name pools; prepared execution binds source versions and consumes source readiness. | Expose only one selectable current pool, with explicit replacement/expiry/consumption. Do not remove source/attachment data needed by an executing or held session. |
| Destination | Persistent base/composite geometry, placed objects and frozen exclusive slot reservations exist. | Keep occupancy and identity when transient data expires. Forgetting occupied slots would allow overlapping placements. |
| Transport | Same-context typed results, durable request IDs, action receipts, unknown barriers and atomic admission exist. | Hide these from the model; do not delete their protections. |
| Motion/gripper | Empty-hand move_to plans and executes through master/VAMP; prepared stages/contact/release exist. get_observe_view is endpoint calculation only. | Camera endpoint success is not path success. General free closing and release of held objects are intentionally restricted. |

Sources: task_pipeline/mcp/tasks.py, preparation.py, execution.py; master
step_runtime.py, world_state.py, manipulation_memory.py, grasp_cache.py,
prepared_grasp_lift.py, execution_api.py, piper_master_control.py;
vision_node active_vision_server.cpp and the shared spatial grid.

Mac review covered mcp_profiles.py, task_checkpoint.py, launch_nanobot.py,
robot-tasks-workspace/AGENTS.md and installed nanobot/agent/tools/mcp.py under
/media/psf/Documents-mac/NanoBot. No credential values were copied. The profile
generator explicitly selects the eight tools; this review does not establish
which profile an already-open Mac conversation is using.

## 3. Data ownership and lifecycle

### One visible workspace

The master owns the authoritative current workspace summary:

- Latest usable synchronized observation and its original source mode/time,
  RGB/depth/CameraInfo, joints, measured TCP and manual FK(joints)*X camera transform.
- Current measured arm/gripper state, calibration identity and motion state.
- One stable semantic inventory; one active object and destination.
- One current grasp pool and current prepared plan, or none.
- One active job and its stage, or none; a durable unresolved-outcome barrier.

Use existing storage. Add a compact workspace generation/current-pointer contract
through my_actions and the registry. MCP reads it, rather than maintaining a
competing copy. Existing historical task/receipt lookups remain diagnostic tools.

### Background observations versus frozen inputs

Retain only the latest eligible background observation for normal use, plus the
bounded rolling/synchronization buffer already needed by ActiveVision. Capturing
must not create a new StepStore context every second. The existing triggered scan
path updates the latest observation; task admission freezes a reference to it.
In-flight task snapshots and attachment geometry remain retained until their owner
can release them. No unbounded history of dense observations is introduced.

Strengthen stationary eligibility using fresh feedback, no executing motion, and
measured joint/TCP stability across a short settling window. Use the existing
synchronization and pose-comparison mechanisms; thresholds need explicit review.
Do not use `IDLE OR same TCP` as the complete test: TCP can stay fixed while joints
change, and IDLE is a controller state rather than a velocity measurement.

Continue roughly 1 Hz local acquisition while eligible. Do not run Qwen, grounding,
segmentation or grasp calculation at 1 Hz. These run on task demand. Moving the
camera requires settling and a new synchronized observation afterward. This is
not continuous dynamic-obstacle protection; the current idle-only routine does
not monitor an executing path against new obstacles.

### Private snapshots and receipts

For every new calculation the server creates a short-lived internal StepStore
context, resolves typed bindings and freezes one observation. The caller never
receives that context in normal results. Reusing one mutable context for all jobs
would mix old/new producer results and break source association and idempotence.
Clear completed calculation contexts and dense results once no pool, plan,
attachment or unresolved job references them. Keep bounded small receipts.

TTL cleanup is not permission to clear an unknown execution session, erase
contact state, reset attempt counts or replay an operation. Those records remain
in the existing ledger until supported reconciliation establishes the outcome.

## 4. Freshness policy

Introduce one master-owned configuration for semantic reuse and unconsumed pools:
`workspace_reuse_ttl_seconds=600`. Change the public profile to this value in a
later implementation, not by changing live parameters now. Remove scattered
semantic-age constants only after their existing purpose/tests have been mapped.
Keep engine-specific capture/feedback/contact/NBV checks authoritative during
migration. A public fresh=true must not override a shorter applicable check.

| Data | Freshness meaning | On stale/missing/invalid data |
|---|---|---|
| Latest RGB-D view | Actual completed synchronized live observation; source checks, feedback, calibration and measured stability valid; stream health and configured age policy pass. | Capture a new synchronized view for a request, or return a concrete unavailable/unstable status. A timer tick alone does not refresh it. |
| Object/destination observation | Semantic evidence and geometry retain their own source times; at most 600 seconds for reuse, with matching calibration and no detected invalidating event. | Task request reobserves/regrounds automatically when possible. Inventory may retain an explicitly stale location hint. |
| New grasp calculation | Always use a newly completed capture after request admission, even when object memory is fresh. | Capture -> grounding/segmentation -> HGGD; stop on unknown inference. No old masks on new pixels. |
| Current grasp pool | One object/source-linked pool, not consumed, applicable source/calibration/generation valid. Expiry is no later than source capture +600s and pool creation +600s. | Remove it from selectable readiness. A new request calculates a new pool. Reading it or taking unrelated images never extends expiry. |
| Prepared plan | Unconsumed source-linked plan within the pool/source expiry, exact revision and expected start state still applicable. | Before any actuation, a definite stale calculation may be replaced by fresh preparation; report the replacement. Never silently substitute a different plan under an old execution receipt. |
| Physical feedback/contact | Existing stage-specific short deadlines and confirmed contact, not the 600-second reuse limit. | Block dispatch or preserve STATE_UNKNOWN according to existing behavior. |
| Tray inventory/occupied slots | Persistent manipulation bookkeeping, not automatically renewed visual evidence. | Reobserve stale geometry; preserve reservations/placed-object history. A moved/reconfigured tray requires reconciliation, not silently clearing inventory. |

Track capture age, semantic-observation age, pool age, calibration/source generation
and stream/feedback health separately. Use monotonic time for local TTLs; preserve
ROS stamps for source association. Restart does not renew retained sensor/plan
freshness; reacquire live data, while durable receipts remain historical.

A new background image proves only that a new image arrived. It neither establishes
that every remembered object stayed put nor updates their last_seen time. Camera
movement marks previous pixel bindings/view-dependent results inapplicable; stable
world identity and tray occupancy remain. Material object/scene/calibration changes
invalidate affected pools/plans, even below 600 seconds. Change detection needs
noise-tolerant tested thresholds; do not invalidate on every timestamp or claim
that no detected change proves safety in occluded areas.

### Simple freshness tool

Proposed `check_freshness(resource, object_name?)`, with resource one of image,
object, grasp_pool or plan, resolves the current workspace internally. Return
`{"fresh":true}` or `{"fresh":false}` in its normal structured result. It is a
read-only policy check, not a VLM call, a refresh action or movement authorization.
Missing/uncertain evidence gives false. Detailed ages/reasons are available in
inspect_robot/status diagnostics. Other public tools check and refresh their own
inputs automatically; the agent does not have to call this first.

## 5. Public tools for Nanobot

Recommended normal preparation profile: **nine tools**. Future separately enabled
control profile: the same nine plus **three physical tools**. Typical pick/place
requires prepare and execute, not an 18-call chain. Existing names below are
retained; new names are proposals, not currently callable APIs.

| Public tool | Human intent / proposed normal inputs | Internals and response |
|---|---|---|
| inspect_robot (extend) | “Where is the arm; what is it doing?” No IDs. | Combine get_arm_state, execution/world state and spatial context/map. Return TCP, joints/jaw, base_link frame, timestamps/ages, stationary flag, camera position/forward/up, plain gaze labels, visible cells, active object/destination, pool/plan availability and blockers. |
| get_camera_image (new) | “Show me what you see.” Optional overlay flag; no context. | Return latest eligible RGB from the shared 1 Hz observation, exact pose/time/source metadata and MCP image content. If unavailable/stale, obtain a synchronized capture or explain why it cannot. No Qwen call just to fetch an image. |
| check_freshness (new) | “Is the current cup/pool/view still fresh?” Resource plus optional name. | Boolean contract above; inspect/status explains false. Never certifies execution safety. |
| list_objects (new) | “What objects do you know about?” No IDs. | Compact canonical names, locations, last_seen/source age, visible/remembered/unknown/held/placed states and destination occupancy. Does not pretend remembered objects are visible or run VLM. |
| observe_scene (extend) | “Find the cup, bowls and tray.” Names/detail; optionally use configured inventory defaults in the future. | Ensure fresh capture, use existing batch perception or single-target route, upsert identity and return per-object PRESENT/NOT_CONFIRMED/UNKNOWN with ages. This is the tool to refresh semantic inventory. |
| plan_camera_view (keep) | “Can you look 3 degrees right?” Existing flat relative/target input. | Use existing planner, preserve exact status/PARTIAL/NO_REACHABLE_CHANGE and endpoint versus path distinction; no movement. |
| prepare_manipulation (extend) | “Prepare to lift the cup” or “put it in the tray.” Existing operation/object/destination inputs. | Always fresh capture for new grasp; own whole chain and candidate trials. grasp_lift validates grasp/approach/lift; pick_place also validates placement, withdrawal and retreat. Return compact ready/blocker and checks/bypasses. Existing grasp operation stays cache/preview only, not an executable plan. |
| get_task_status (extend) | “How is it going?” Default current task; explicit task_id remains optional for compatibility/diagnostics. | Resolve current task server-side. State/stage/object/last success/failure and freshness actions; no inference or replay. Terminal active step null. |
| cancel_task (extend) | “Cancel that calculation.” Default current job; explicit ID remains compatible. | Existing supported cancellation boundaries. Pending is not stopped; no remote uncertainty reset. |
| move_camera (new, physical profile only) | “Move the camera 2 cm right” or “look at the bowl.” Flat relative six-delta or named-target schema, no joint input. | Capture current geometry -> existing get_observe_view/plan_next_best_view -> full master/VAMP path checks -> guarded empty-hand execution -> settle/capture. Degrees may be accepted at this facade and converted once to existing radian APIs. Endpoint-only success never authorizes motion. Reject while holding; payload movement uses prepared held operations. |
| control_gripper (new, physical profile only) | “Open” or “close.” Action, optional width/force with explicit units. | Empty-hand opening uses existing bounded set_gripper path with a new reviewed MCP adapter/ledger contract. Planned opening/closing uses frozen prepared stages. Close requires AT_GRASP and fresh contact verification; held-object opening requires validated placement release. Arbitrary free closing/releasing is NOT currently supported and must not be enabled by removing these guards. |
| execute_manipulation (new, physical profile only) | “Execute the current prepared action.” Optional supported stage/all. No context or plan ID for the model. | Internally bind the exact plan the client last reviewed to the master execution revision; execute existing prepared stages in order, including contact/attachment, release, commit, withdrawal and retreat. Return real completion/unknown status. Reject replaced or ambiguous plans; never execute whichever plan happens to be newest. |

Keep public schemas flat and descriptions short for Qwen/llama.cpp. Mode
requirements remain strict server checks. get_task_status and cancel_task default
current are backward-compatible extensions, not removal of historical lookups.

Retain describe_workflow and validate_plan as optional help/diagnostic tools;
normal prepare/execute paths own their checks automatically. Existing 54-tool
profiles remain supported for maintainers. Profile filtering is usability, not
security: preparation_only still independently refuses all physical dispatch.
No physical tools are added to the running preparation profile by this proposal.

### Classification of the current diagnostic surface

All **54 existing diagnostic/preparation tools** are accounted for below. Keep
these as internal building blocks for ordinary Nanobot use, and retain their
existing authorized diagnostic profiles for developers. No rename or removal of
the existing contracts is proposed.

| Internal category | Exact current tools |
|---|---|
| Robot/world/grid reads | `get_arm_state`, `get_world_state`, `get_execution_state`, `get_spatial_context`, `get_spatial_map`, `get_manipulation_memory`, `get_floor_slots` |
| Capture/source retention | `capture_observation`, `pin_grasp_observation`, `pin_placement_observation` |
| Perception, identity and visibility | `verify_presence`, `ground_object`, `segment_object`, `perceive_scene`, `ensure_scene_memory`, `resolve_target`, `refresh_active_target`, `get_active_target`, `store_segmented_target`, `update_object`, `check_target_view`, `require_target_view` |
| Grasp pool and local refinement | `prepare_grasp_input`, `calculate_active_grasps`, `calculate_grasps`, `select_grasp`, `cache_grasp_plan`, `get_cached_grasp`, `refine_grasp` |
| Camera solving | `get_observe_view`, `plan_next_best_view`, `plan_camera_retreat` |
| Placement/motion preparation and validators | `find_active_placements`, `find_placements`, `search_placements`, `prepare_held_scene`, `solve_pose`, `plan_motion`, `plan_final_approach`, `plan_lift`, `plan_placement`, `validate_placements`, `prepare_grasp_lift`, `get_prepared_grasp_lift`, `validate_prepared_grasp_lift`, `prepare_pick_place`, `get_prepared_pick_place`, `validate_prepared_pick_place` |
| Queue/binding infrastructure | `run_step`, `submit_steps`, `run_steps`, `get_step_context` |
| Maintainer discovery/examples | `list_functions`, `get_workflow_example` |

Existing physical catalog stages such as move_to, execute_plan, gripper_open,
gripper_close, execute_lift, transfer/descent/release, commit_floor_slots,
execute_withdrawal and execute_placement_retreat also remain implementation
building blocks. They are not part of those 54 preparation-mode tools. Future
public control facades must use the existing guarded physical admission path,
never submit these stages through the generic calculation queue. Local emergency
stop/watchdog mechanisms stay independent of the normal task/LLM queue.

### Further placement while already holding an object

Add a reviewed `place_held` operation to prepare_manipulation in a later stage,
reusing prepare_held_motion and placement search from the master-verified current
attachment. It must not re-grasp or pretend the initial object pose is current.
Do not invent an attachment when the prior grasp/contact outcome is unknown.
Generic arbitrary payload TCP jogging remains outside the existing empty-hand
move_to contract; it needs a separately validated payload-aware route.

## 6. Object identity and updating the world

Keep stable physical-instance IDs internally and human names/aliases publicly.
Separate target identity (“beige cup”) from requested grasp region (“body/rim”).
Normalize case/spacing and store explicit aliases; do not equate descriptions by
unreviewed fuzzy text matching alone.

Extend the existing WorldStateManager association/upsert, not an MCP dictionary:
use explicit existing identity/aliases, simple existing position/geometry gates and
confirmed grasp/release motion lineage. Use appearance evidence only when already
available; a new learned tracker is not required for this change. Refresh that same record
rather than creating cup_2 because the cup moved more than the old 5cm gate.

For genuinely multiple cups, return distinct spatial descriptions such as “cup
on left” and “cup beside plate.” Ambiguous “cup” requests must not silently merge
or select a different physical instance. If identity cannot be established, keep
it uncertain and reobserve/ask for a target distinction. Missing memory or failed
inference is UNKNOWN, not visual absence. Update location/last_seen only from real
synchronized evidence or explicitly labelled confirmed-placement estimates.

No general public write-world-XYZ tool is needed. observe_scene upserts actual
observations; confirmed execution updates held/placed state automatically. Manual
naming/alias changes, if required later, cannot forge visible geometry/contact.

## 7. Single-pool and execution lifecycle

1. A fresh grasp request starts a current job; reject another mutating calculation
   while the workspace owner is busy. Read state/image/status remains available.
2. Freeze fresh synchronized input, perceive/segment the named instance and
   calculate one pool. Background latest images may advance independently.
3. For grasp_lift try the pool against the complete existing grasp/lift preparation;
   for pick_place try it against the complete existing preparation. Failed
   candidates are internal bounded trials, not separate agent calls. Unknown stops.
4. Keep at most one current unconsumed pool/plan alias, expiring within 600s from
   source, with remaining lifetime reported. Check expiry on every access; a bounded
   master cleanup sweep releases expired unowned geometry even if nobody queries it.
   Cleanup does not trigger inference or reset unresolved outcomes. New explicit preparation replaces
   only unclaimed transient readiness and reports the replacement.
5. At first authorized execution dispatch, atomically claim readiness for that
   execution and make the pool unavailable for another task. Move its immutable
   source/plan ownership to the existing execution session; do not free it mid-path.
6. A calculation-only validation does not consume readiness. After confirmed
   terminal completion, retire the pool/plan and dense temporary data. After
   grasp_lift, keep verified attachment/geometry for later held manipulation;
   after placed release, keep destination composite/occupied slots.
7. Definite calculation failure may discard the transient pool with a useful
   blocker. Dispatched failure/timeout/contact uncertainty keeps the session and
   ledger barrier until existing supported reconciliation, regardless of TTL.

Consumption, replacement, expiry and workspace generation must be atomic with
existing master ownership and MCP admission, not separate locks that can disagree.

## 8. Bookkeeping without making the model manage IDs

No public context IDs. The Mac adapter generates a request ID once per NEW tool
operation and persists it before dispatch, preferably keyed by the provider's
stable tool-call identity. Identical recovery reuses it; a new intent gets a new
ID. Do not generate an ID for every network retry, and do not deduplicate distinct
user intentions solely because their arguments match.

Server-generated job/plan/source/revision IDs stay in the current workspace and
durable ledger. Mac task_checkpoint.py stores the exact receipt/plan association
outside the model transcript. Model-facing status can default to current. The
adapter binds execute-current to the exact displayed prepared receipt; another
client replacing the alias must cause PLAN_REPLACED, not execution of the new plan.

One mutating control owner is admitted at a time, even across reconnects/clients.
Reads can coexist. Unknown outcomes survive reconnect and expiry; a client
connection disappearing is not proof motion stopped. Preserve local cancellation,
watchdogs and direct emergency-stop paths independently of Nanobot/Qwen.

## 9. Automatic refresh and compact responses

Ordinary calls explain what happened, not how to repair ROS bindings:

```json
{
  "status": "READY_WITH_BYPASSES",
  "source_mode": "live",
  "target": "beige cup",
  "freshness": {
    "new_capture": true,
    "object_reobserved": true,
    "reason": "NEW_GRASP_REQUIRES_FRESH_OBSERVATION"
  },
  "expires_in_seconds": 510,
  "execution_allowed": false
}
```

This is a proposed display shape, not an existing backend status. Preserve exact
backend statuses, checks_passed/checks_bypassed and their reasons alongside the
summary. Report actual original capture times, geometry ages and refresh actions.

Only automatically refresh before a not-yet-dispatched operation, following a
definite stale/missing-input condition and a bounded route. Do not retry UNKNOWN,
keep regrounding forever, or recapture and silently change the requested physical
plan. If a target is outside the view, report LOOK_REQUIRED or use an explicitly
authorized look workflow; a perception request alone does not authorize movement.

## 10. Mac-specific implementation work

- mcp_profiles.py: keep robot-tasks/planning and a separate opt-in control profile;
  retain diagnostics. Adapt TASK_TOOLS, forbidden/control gates and loader filter
  together. A client profile never overrides Ubuntu authorization.
- task_checkpoint.py: hide request/task/plan IDs from ordinary prompts while keeping
  durable submission/receipt recovery. Default-current lookup is a convenience,
  not permission to overwrite an unresolved task or choose among several plans.
- Installed MCPToolWrapper currently returns image artifact paths, not visual
  input to Qwen. Add a supported multimodal forwarding adapter (or a small upstream
  change if needed) so get_camera_image pixels actually reach the vision-capable
  provider request. Keep binary data out of textual JSON/history and checkpoints.
  Add CLI/user image display independently; disabled message tools cannot be assumed.
- Verify actual model + vision projector/input support through the Mac llama.cpp
  connection. A tools/list success or stored image file is not a visual test.
- Keep one latest image plus requested task evidence; do not send 1 Hz images into
  every model turn. Bound resolution/vision tokens and keep schemas compact.
- Preserve TLS, bearer authentication, flat native schema definitions and the
  64K server context limit. Re-measure real prompt/image budget, do not expand
  context as a substitute for hiding diagnostic workflows.

## 11. Implementation stages and review gates

| Stage | Concrete work / owners | Verification before proceeding |
|---|---|---|
| 1. Current workspace + policy | Extend master world/memory summary and registry/my_actions contracts; keep StepStore contexts private. Introduce shared 600s reuse policy, dependency/pool generation and automatic context/blob retirement. | Fake clock tests at 599/600/601s; idle-but-moving joints; stale feedback; calibration/source change; restart; new frames cannot renew semantic age; no live/mocked mix; unknown records never auto-expire into readiness. |
| 2. Image, gaze, freshness and inventory | ActiveVision exposes latest completed observation metadata/image through a bounded typed read. Extend inspect_robot; add image/freshness/list tools, reuse shared spatial map/manual FK*X. Fix stable world upsert/aliases. | Real pixels/source pose/header correspond; ray/FOV labels use existing grid; distinguish vertical gaze/roll; repeated moved cup updates one instance; two physical cups stay separate; unknown inference cannot overwrite last_seen; tray reservations persist. |
| 3. Automatic calculation + one pool | Extend current TaskInterface routes and master cache owner; always fresh capture for new grasp; operation-specific whole-plan validation; source-bound expiry/claim/flush. Add place_held through existing held-motion preparation. | Mock cup and two distinct bowls; grasp_lift and full pick/place; stale target auto-refresh announced; stop at first real blocker; TTL never extends during inference; alternate-candidate search stays bounded; new task after completion; no deletion while holding. |
| 4. Nanobot usability | Local profile/checkpoint adapter and supported image forwarding/display. No changes to Ubuntu guards. | Actual Qwen receives pixels and answers a fixture image question; natural “show scene/find cup/prepare lift/check status” without IDs/bindings; compaction/reconnect retains one operation; flat schema argument generation tests. |
| 5. Guarded physical facade (separate authorization) | Add move_camera, control_gripper and execute_manipulation adapters over catalogued master operations, same durable ledger/session. Update preparation/control mode allowlists and local physical gate review. | Isolated fixtures only: forbidden dispatch in preparation mode; endpoint success/path failure; stale displayed plan; fresh source replacement; contact loss; release ordering; duplicate/reconnect; bounded grasp attempts; cancellation and watchdog independent of client. No automated hardware motion. |
| 6. Operator physical validation | Only after the above, operator explicitly enables the existing reviewed physical gates and performs staged trials. | Empty-hand camera move, open/validated close/contact, grasp/lift, release/retreat; verify real tracking and scene/currentness limits. Mock replay cannot provide this evidence. |

Likely Ubuntu files: vision_node/src/active_vision_server.cpp and spatial_grid.hpp;
my_actions/srv interfaces; task_pipeline/registry.py, context.py, mcp/tasks.py,
server.py and schemas; master_node world_state.py, manipulation_memory.py,
execution_api.py, grasp_cache.py, prepared_grasp_lift.py, held_motion.py,
step_runtime.py, launch/config and affected package tests. Reuse existing engines.
Detailed file changes must be chosen after stage-specific contract review; do not
implement every row as an independent authoritative cache.

### Required negative tests

- Fresh RGB plus stale cup mask -> refresh that object, not reuse the mask.
- Same TCP but changing joints, stale driver or unknown task -> not eligible.
- No valid depth/sync, missing source, black/uninformative image or Qwen ambiguity
  -> unavailable/UNKNOWN, not fabricated absence or a ready plan.
- Pool used/expired/superseded, wrong displayed plan, moved tray or changed
  calibration -> no physical dispatch; preserve inventory and receipts.
- Opening while holding outside validated release, free close without prepared
  grasp, relative empty-hand movement while holding -> explicit existing guard.
- Disconnect during action -> UNKNOWN until actual receipt/reconciliation; no
  auto-clear or replay even after 600s or a new background observation.
- A local stop is not delayed behind Qwen; between-step cancellation is not
  represented as an emergency stop or proof an in-flight service stopped.

## 12. Human-facing examples after implementation

- “What do you see?” -> get_camera_image. Pixels reach Qwen and the user; pose/time
  are attached. To enumerate object geometry, observe_scene follows on demand.
- “Where is the camera pointing?” -> inspect_robot, with optical/base frame labels
  and actual FOV/occlusion-aware cells, not only a quaternion.
- “Move 2 cm to the right.” -> move_camera in the explicitly enabled control
  profile; server plans, validates, executes, settles and captures. No manual IK.
- “Lift the cup.” -> prepare_manipulation(grasp_lift, cup), then authorized
  execute_manipulation. Fresh capture -> internal perception -> pool -> complete
  lift preparation. User sees ready or the real blocker, not producer bindings.
- “Put it in the tray.” -> pick_place before grasping; place_held if already held
  with confirmed attachment. Destination identity/reservations remain current.
- “Is that data still fresh?” -> check_freshness(resource), a Boolean. It is never
  a prerequisite the agent must manually insert before every operation.

The recommended change is a simpler interface over rigorous internal ownership:
**one current workspace, fresh data when needed, one consumable pool, and no
context bookkeeping for the human or model**.

## 13. Implementation checklist and completion tracking

This is a staged implementation plan, not a completed implementation. All items
below are initially unchecked. Complete each stage's acceptance tests in section
11 before marking it done. Record actual reports/blockers in ENGINEERING_JOURNAL.

### A. Contracts and authoritative state

- [ ] Inventory each existing semantic TTL, source owner and physical freshness
  check; define which becomes 600 seconds and which keeps its tighter limit.
- [ ] Define the typed current-workspace summary in my_actions and register its
  public ROS operation in registry.py. Include independent observation/semantic
  times, current task/pool/plan availability, generations and source mode.
- [ ] Extend master world/memory owners with current aliases; preserve durable
  gateway receipts and private immutable StepStore producer contexts.
- [ ] Implement one freshness-policy helper in the owning ROS layer and expose
  its result to MCP. Do not duplicate policy decisions in the Mac adapter.
- [ ] Implement generation invalidation and bounded cleanup; prove that unrelated
  new captures cannot renew semantic data, and UNKNOWN cannot become ready by TTL.

### B. Sensor and object-facing tools

- [ ] Expose the latest completed synchronized observation through a typed read;
  reuse ActiveVision buffers and original FK*X/source headers. Add measured
  stability and missing-stream diagnostics to periodic capture eligibility.
- [ ] Extend inspect_robot with camera axes, gaze, coverage, freshness and current
  workspace summary using existing C++ spatial/FOV/raycast results.
- [ ] Add get_camera_image, check_freshness and list_objects to the task facade,
  native schemas/resources and a nine-tool preparation profile.
- [ ] Extend WorldStateManager association/upsert with explicit aliases and known
  grasp/release lineage; keep ambiguous or multiple objects distinguishable.
- [ ] Test image-pose association, two cups versus one moved cup, remembered versus
  observed data, unavailable RGB-D and persistent destination reservations.

### C. One-pool preparation and consumption

- [ ] Extend prepare_manipulation to ensure new grasp capture and automatically
  refresh stale prerequisites through the existing reviewed calculation routes.
- [ ] Replace model-visible per-object cache selection with one current pool;
  keep source-bound engine storage internal. Apply the 600-second expiry without
  silently changing existing physical feedback/contact checks.
- [ ] Atomically claim readiness at execution admission and retire selectable
  pool/plan aliases; preserve the session's source and attachment until safe to free.
- [ ] Extend get_task_status/cancel_task with optional default-current resolution,
  while retaining explicit-ID compatibility and all existing receipt semantics.
- [ ] Implement place_held only through verified attachment and existing held-motion
  preparation; report real unsupported cases instead of treating them as pick_place.
- [ ] Run isolated mock cup/bowl end-to-end calculations and negative tests for
  expiry, replacement, stale inputs, capacity, cancellation and unknown outcomes.

### D. Mac Nanobot integration

- [ ] Update local mcp_profiles.py/loader filtering for the reviewed preparation
  profile while preserving existing diagnostic profiles and execution exclusions.
- [ ] Extend task_checkpoint.py to inject/store request and exact reviewed plan
  references outside model inputs, and recover default-current work after compaction.
- [ ] Implement actual MCP-image-to-provider visual forwarding and user image
  display; inspect the provider request and test Qwen against a known mock image.
- [ ] Verify flat schemas and natural-language use without contexts, bindings or
  manual task/plan IDs. Refresh native and translated discovery snapshots.

### E. Physical facade and operator handoff

- [ ] Define/review the three physical-tool schemas and map every action to an
  existing guarded master function; document any missing capability first.
- [ ] Implement move_camera with full-path planning and fresh post-move capture;
  implement control_gripper only within the supported opening/contact/release states.
- [ ] Implement execute_manipulation as orchestration of the existing prepared
  stages and exact displayed-plan binding, not another execution state machine.
- [ ] Prove preparation-mode rejection, duplicate suppression, replacement races,
  contact/feedback failures, watchdog/cancellation independence and UNKNOWN barriers
  in isolated fixtures without dispatching hardware commands.
- [ ] Provide an operator trial procedure. Actual enabling of physical execution
  and physical trials are separate from automated development verification.

### Decisions to finalize before the relevant stage

The architecture and implementation order are specified. A few detailed contracts
still require stage-specific choices: typed message/service names and exact fields,
stationary/stream-health/change-detection thresholds, object alias/disambiguation
rules, the supported Nanobot multimodal forwarding API, physical-tool schema
bounds and precisely supported stage boundaries. These are explicit design work,
not already tested capabilities. No unrestricted free gripper close, arbitrary
payload jog or remote reset is implicitly authorized by this proposal.

## 14. Implementation progress

2026-11-10 — first Stage A increment:

- Implemented master_node/workspace_freshness.py: immutable 600-second policy,
  Boolean/reason/age/remaining-time result, explicit source/calibration/generation/
  feedback/stream evidence, consumption and unknown-outcome rejection, optional
  shorter backend deadline, and source-bounded pool expiry.
- 23 new isolated policy tests pass; together with existing world-state tests,
  26 tests pass. Affected-file ament_flake8 and git diff --check pass.
- This helper is NOT yet connected to runtime storage, public MCP or the current
  900-second cache. No live configuration change, deployment/restart or actuation.
- Next: define the master-owned current-workspace read contract and observation
  receipt/source timing. Only then integrate policy decisions into that contract;
  never substitute arrival/verification time for the original capture age.
- Remaining Stage A items and all later stages remain incomplete. The checklist
  above is not marked complete merely because this helper passes unit tests.

### Working with another coding agent

The plan supports scoped handoff; it is not a safe single prompt to implement the
whole redesign without review. Give each ticket the specific files/contracts,
required tests, prohibited runtime changes and completion evidence. Schema wiring,
read-only tool formatting and documentation can be separate tickets. Source-time
association, current-plan atomic ownership, consumption/retirement, unknown barriers,
image-to-provider integration and physical adapters need careful review. Do not
parallelize edits to the same master state or gateway ownership code.

2026-11-10 — Stage A observation-contract increment:

- Added typed GetCurrentWorkspace.srv and registered get_current_workspace at
  /pipeline/get_current_workspace (backend /master/get_current_workspace).
- Master CurrentWorkspace subscribes to existing ActiveVision synchronized image,
  cloud metadata and original manual camera transform. It retains one admitted
  image, four bounded unmatched bundles and bounded measured feedback samples;
  it does not duplicate dense cloud storage or create a second planner/job queue.
- Joins exact source stamps/dimensions and nearby measured feedback. Checks idle
  state, measured joint/TCP stability, current feedback, original source identity,
  calibration fingerprint, original capture age and independent arrival health.
- Read-only startup settings currently default to 600s reuse and 3s stream-arrival
  gap. Existing cache TTL, contact deadlines and planning/execution guards unchanged.
- Shared task_pipeline.provenance helper preserves the existing live/mock graph
  policy. Original observation identity is retained so a later live publisher
  cannot relabel a recorded replay observation as live.
- inspect_robot can read the workspace metadata through a fixed internal bridge;
  public diagnostic tool inputs and the existing eight-tool profile are unchanged.
- Fixed memory calibration fingerprinting to prefer the runtime hand_eye_matrix,
  with legacy X_matrix support for fixtures. This changes identity validation,
  not calibration values or manual FK*X geometry.
- First mock read exposed source image age incorrectly used as stream gap.
  Corrected it: processing latency, capture age and last-arrival age are separate;
  duplicate stamps never renew original age. Regression coverage added.
- Isolated build of my_actions/task_pipeline/master_node passed. Package tests:
  master_node 284 passed/1 skipped; task_pipeline 240 passed. Targeted lint passed.
  Authenticated domain-89 MCP inspect_robot reported fresh mock workspace metadata;
  the typed read returned 640x480 RGB with matching source timestamps.
- Reports: ~/agx_piper_data/reports/current_workspace/{typed_read,authenticated_read}.json.
  Test build: ~/agx_piper_data/builds/current_workspace/install. Temporary mock
  processes were stopped after verification. Domain-0 live stack/port 8765 and
  drivers were not restarted. The new contract is NOT deployed to them.
- Remaining: public image/freshness/object-list tools, semantic-age/identity and
  single-pool migration, cleanup/claim integration, Mac visual forwarding and
  physical facades. Stage A is partial; this does not complete the whole checklist.

The runtime contract includes nearby source_joints/source_tcp (<=0.20s), not a
new Python reconstruction of the exact C++ synchronization sample. The original
camera_transform remains the authoritative synchronized FK*X. Workspace freshness
is sensor reuse evidence only, not scene-currentness or execution authorization.
Rebuild/source the updated my_actions and consumers before launching new code;
the running deployment intentionally continues using its existing generated types.

### Handoff — 2026-11-10

This increment implements the optional workspace tool surface and staged Mac
Nanobot adapters; see [WORKSPACE_TOOLS.md](mcp/WORKSPACE_TOOLS.md).
Verification: 251 pipeline tests and 285 master tests passed (1 skipped);
8 portable Mac adapter tests passed. Authenticated mock image delivery and
direct Qwen image consumption passed; native Mac launcher remains unverified.
Full preparation was interrupted by a ROS NumPy integer serialization crash.
The fix passed a regression and isolated ROS roundtrip, but the interrupted
job remains UNKNOWN with its existing barrier intact. See the engineering
journal before resuming. The single-pool/cache migration and physical facades
are still outstanding; this is not a completed physical-control redesign.
