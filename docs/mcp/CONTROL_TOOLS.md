# Guarded control facades

Staged implementation. The running mock/preparation server cannot actuate hardware.
`--control-tools --task-tools --workspace-tools --preparation-only` publishes the
three disabled cards for discovery. Preparation-only rejects them independently,
constructs no physical ROS transport and never forwards physical queue functions.

Physical-mode startup additionally requires explicit `--require-live-inputs`,
absence of preparation/perception/read-only flags, and the master's separate
physical permission. No flag enables motors or changes that permission.
Do not enable physical mode as part of development tests.

- `move_camera`: same flat relative optical metre/radian or target selection as
  plan_camera_view. It owns planning, fresh-start checking and master/VAMP path
  execution. Unreachable/no-change returns the actual planner result without
  movement. A collision-checked endpoint alone never authorizes a path.
- `control_gripper`: prepared open/close/release require exact reviewed plan_id
  and revision and the appropriate master stage. Empty-hand mode permits only
  explicit opening width_m/force_n, fresh feedback, no known attachment and no
  narrowing. Open detection does not verify a calibrated requested aperture.
- `execute_manipulation`: exact reviewed plan_id/revision, stage all or next;
  complete grasp_lift and pick_place routes only. Runs existing stage guards,
  contact checks, release/slot commit/withdrawal/retreat. Grasp_lift stops holding
  and retains ownership. It does not invent an unsupported place-held route.

All require durable request_id. Identical retries retrieve the receipt without
reexecution. A request ID cannot change its arguments. No latest-plan guessing:
the exact plan pair received by the reviewing client must be supplied. The model
may have these transport fields hidden by a reviewed client adapter; the server
still requires them. A superseded source/revision is rejected before actuation.
Timeout/cancellation after dispatch is UNKNOWN, blocks further actions, and
requires local reconciliation. Disconnecting is not a stop. get_task_status reads
the same durable task; cancellation cannot stop an in-flight ROS physical service.
Local stop/watchdog remains independent. Generic queues never contain these tools.

The ROS opening wrapper `open_empty_hand` uses the existing GripperCommand schema,
master physical/ownership guards and feedback executor; it adds no motor shortcut.
Existing set_gripper semantics are unchanged. Existing diagnostic API names and
preparation-mode allowlists are preserved unless control cards are explicitly added.

## Startup and client setup

Current verified mock service: MCP3.8.0 on port8766/domain89, **preparation_only**.
With --control-tools it lists68 cards:65 callable non-actuating tools and3 disabled
control cards. Without that optional flag the existing65-tool surface is unchanged.
The old54 diagnostic profiles and nine-tool workspace profile remain supported.
Exact inputs/outputs: CONTROL_TOOLS.json and native control-catalog resource.

To expose disabled cards for mock development, retain the existing source guard,
TLS/token/ledger arguments and add --control-tools to the preparation-only command
in WORKSPACE_TOOLS.md. Do not point a mock server at the live execution ledger.

Future supervised live-mode server command (not executed during development):

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/agx_piper_stack/install/setup.bash"
# Source the separately rebuilt deployment overlay when applicable.
export ROS_DOMAIN_ID=0
ros2 run task_pipeline pipeline_mcp \
  --require-live-inputs --task-tools --workspace-tools --control-tools \
  --execution-ledger "$HOME/.local/state/agx-piper-mcp/execution.sqlite" \
  --host 10.211.55.3 --port 8765 \
  --token-file "$HOME/.config/agx-piper-mcp/token" \
  --tls-cert "$HOME/.config/agx-piper-mcp/server.crt" \
  --tls-key "$HOME/.config/agx-piper-mcp/server.key"
```

This command alone cannot enable the master's physical gate, motors or drivers.
No live startup or master safety-setting change was performed. Physical behavior
is untested. Rebuild master_node/task_pipeline before deployment; the new internal
open_empty_hand service uses the existing GripperCommand schema.

Mac: opt-in profile `workspace-control`, with exactly the nine workspace tools plus
move_camera, control_gripper, execute_manipulation. Default config is unchanged.
Refresh **separate control snapshots** against the intended guarded live endpoint:

```bash
python3 tests/discover_workspace_mcp.py --control \
  --url https://10.211.55.3:8765/mcp --expect-source live_driver_publishers
python3 mcp_profiles.py workspace-control
```

Use the existing trusted CA setup or --ca-file; TLS is never disabled. Profile
creation rejects preparation/mock discovery, disabled cards and URL mismatches.
The client hides request/plan IDs from the model and injects the exact successfully
reviewed plan pair from that conversation's private checkpoint. No global-latest
plan fallback; failed/new incomplete preparation blocks that binding. SDK retries
retain the original request and reviewed pair. Native Mac launch remains unverified.

All control tool calls await final results. The existing ROS stage timeout is300s;
camera planning is bounded at60s and free opening at15s. Client timeout/disconnect
is never proof of a stop. Use get_task_status for the same task and local stop for
an emergency. Never replay unknown work with a new request ID.

## Verification evidence

330 master tests passed (1 existing skip),282 pipeline tests passed,14 portable
Mac adapter tests passed. Synthetic transports exercised full pick/place ordering,
grip failure before transfer, exact reviewed revision, grasp/lift holding ownership,
no-change/partial camera plans, master gate rejection, opening-only behavior,
SDK replay receipts, cancellation/timeout uncertainty and forbidden generic queues.
No fixture can construct an actual actuator transport or publish physical commands.

Authenticated calls against the actual domain89 mock server verified native
MCP3.8.0 discovery, flat schemas and rejection of all three cards with
PHYSICAL_CONTROLS_DISABLED/dispatched=false. The source guard, TLS/authentication
and ledger remain intact. Reports: ~/agx_piper_data/reports/control_tools/.
Successful physical execution is not established by these tests.
