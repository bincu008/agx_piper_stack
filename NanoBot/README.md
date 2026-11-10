# Native Nanobot robot integration

Installed official PyPI nanobot-ai 0.4.0 in this folder's `.venv` using native Python 3.12. No models downloaded and no ROS pipeline changed. Config: `~/.nanobot/config.json`; original onboarding config backed up as `config.json.before-robot`. Agent instructions appended to `~/.nanobot/workspace/AGENTS.md`.

## Start

Open LM Studio. Start its existing server on port 1234. Load the existing model with:

```sh
lms load robobrain2.5-4b -c 32768 --parallel 1 --ttl 3600 -y
cd /Users/huynhnguyentrunghieu/Documents/NanoBot
./start-nanobot.command
```

Use the same `robot-console` session to retain state. Quit with Ctrl-C. Optional UI: `.venv/bin/nanobot webui` (not started or validated here). No background agent service installed.

## Status and limitations

- Native Nanobot smoke test PASSED at 32768 context: returned READY (`results/nanobot-smoke-3.txt`). LM Studio OpenAI-compatible endpoint works. Native Nanobot `lm_studio` provider configured. Serialized provider key is `lmStudio` in release 0.4.0; `lm_studio` in the provider selector.
- No model was initially loaded. Selected existing RoboBrain2.5-4B Q4_K_S (3.44 GB), the only installed model advertising tool training. That metadata did NOT predict correct behavior: synthetic three-step test produced JSON text, zero structured tool calls. Do not use it for robot control until a model/template configuration passes tests.
- 4096 and 16384 context budgets failed Nanobot's own context guard before inference. Its built-in prompt/tools plus robot instructions measured about 23,384 tokens by heuristic. Final context is 32768, output cap 768, one inference at a time; this is below advertised 262144. VM is assigned 8 GB. This remains provisional: sustained memory pressure and long-chain behavior are unverified. No VM allocation changed.
- Shell, web, CLI adapters and self-configuration tools disabled; heartbeat disabled; workspace restriction enabled. Workspace file tools remain for checkpoints. This is not a security sandbox or a safety controller.
- VM read-only inspection via Parallels Tools succeeded: `10.211.55.3`. No MCP listener or running ROS processes appeared in the inspected process/socket snapshot. No MCP implementation found in ROS source tree. No physical robot commands issued; no VM changes made.
- Remote MCP discovery, authenticated connection, read-only ROS queries, actual API schema preservation, grasp retry, cancellation, timeout and disconnection integration tests remain BLOCKED by missing MCP endpoint. No local robot substitutes created.
- Instructions bound grasp attempts to three and require verification before placement, stop on unknown outcome, and checkpoint transitions. These policies require independent server-side enforcement. Native iteration bound is 16; persistent session history is native. Background goals are intentionally not configured for physical tasks.

## Finish MCP integration

Start the existing compatibility server inside Ubuntu separately; provide its actual endpoint and existing authentication. Do not guess API names. Bind it to the VM private interface, restrict inbound access to this Mac, and require authentication. HTTP bearer credentials on the shared private subnet are unencrypted: prefer verified HTTPS if supported.

The helper uses MCP SDK initialize/list_tools only. It never calls a tool. From this folder, supply the existing token through `ROBOT_MCP_TOKEN` without putting it in command history, then:

```sh
.venv/bin/python connect_mcp.py ACTUAL_SERVER_URL
```

It accepts only discovered VM host `10.211.55.3`, times out at 30 seconds, saves advertised schemas to `results/mcp-tools.json`, and merges native remote MCP config with exact `/32` SSRF exception and `enabledTools: []`. Token is stored in mode-600 Nanobot config; do not share it. Review schemas and explicitly add only verified read-only names initially. Nanobot wraps remote names with `mcp_<server>_`; raw schemas remain in discovery output. Confirm all pages of tool listing if server paginates before enabling tools. MCP client timeout never proves physical cancellation.

Then validate read-only status, native multi-call orchestration, server-supported mock/dry-run failure scenarios, cancellation and disconnection before authorizing physical work. No claim of a completed robot integration is made.

## Evidence

`results/tool-chain.json`: synthetic arithmetic test and raw transcript (no robot functions). `results/nanobot-smoke*.txt`: native agent results. `results/installed-packages.txt`: installed dependency versions. Run `.venv/bin/python tests/tool_chain.py` to repeat the structured-calling test.

Official references: https://github.com/HKUDS/nanobot and its docs/configuration.md. Installed release schema was used to validate settings because main-branch docs can differ.

## Native Nanobot VM terminal test — passed

Nanobot CLI session `vm-terminal-test` invoked its own native `exec` tool twice through Parallels Tools:

- `prlctl exec 'Ubuntu 24.04.3 ARM64' 'hostname'` → `ubuntu-gnu-linux-24-04-3`
- `prlctl exec 'Ubuntu 24.04.3 ARM64' 'uname -sm'` → `Linux aarch64`

Evidence: `results/nanobot-vm-terminal.txt` and native session history. The agent received both tool results and interpreted them correctly. Both calls were requested in one model response; this verifies two actual executions, not an adaptive long-horizon chain. The earlier direct synthetic test failure remains relevant, but the native Nanobot path passed this limited terminal test.

Native exec is now enabled with an exact allowlist for these two commands, 20-second runtime and 30-second upper limit. No general terminal access or ROS commands were authorized. Settings before this change are in `~/.nanobot/config.json.before-vm-terminal`. This connection uses Parallels Tools, not SSH or MCP. Robot/MCP integration remains pending. Nanobot is the agent; LM Studio supplies the underlying reasoning model.

Repeat using `./start-nanobot.command -m "Use exec to run prlctl exec 'Ubuntu 24.04.3 ARM64' 'hostname' and report its output."`.

## VM memory inspection

Native terminal allowlist now also includes `prlctl exec 'Ubuntu 24.04.3 ARM64' 'free -h'`. Instructions direct Nanobot to execute memory queries and interpret the observed output. Restart an already-open Nanobot CLI to reload configuration: Ctrl-C, then `./start-nanobot.command`. Evidence: `results/nanobot-vm-memory.txt`. General shell and robot commands remain excluded.

## Interactive launcher correction

The launcher now explicitly selects `--classic`, matching the direct runtime used in the successful tests, instead of the default TUI/gateway path. `./start-nanobot.command --new` opens a fresh session without deleting previous chats. Without `--new`, it resumes `robot-console`. Ctrl-C stops the interface; it does not delete persistent history. This change avoids relying on a separate gateway to refresh configuration. The gateway/TUI path has not been validated for this workflow.

## Remote Qwen startup

Run `./start-remote-nanobot.command` in the Mac Terminal. It creates or reuses the loopback SSH tunnel on port 18080 to `bincu008@192.168.50.128`, checks the exact advertised model ID, then validates and atomically updates Nanobot configuration with a restricted-permission backup. It selects the custom provider, clears an overriding preset, sets 32768 context and 512 output tokens, preserves tool settings, and opens a new classic CLI chat. SSH authentication occurs in your Terminal; no password is saved by the launcher.

The tunnel runs in the background using SSH `-f`; a separate normal SSH shell is not needed. The launcher does not modify server files, download models or start a remote inference service. Model discovery is not proof of generation, vision or tool-call correctness. At creation, the tunnel was closed and non-interactive SSH authentication failed; remote functional validation was unavailable. Do not treat the supplied server handoff's claimed tests as independently verified here.

## ROS MCP connected (2026-11-03)

Verified authenticated Streamable HTTP at `https://10.211.55.3:8765/mcp`, server `ros-pipeline` version 3.1.0. Public self-signed certificate and token were obtained through the existing trusted Parallels guest channel; no private key copied. Certificate SHA256: `70a91d0cc4c206a03ea7a350c96e5c759b79a1c96491add6800b3533899382cd`. Verified TLS without disabling certificate checking; unauthenticated access returned 401. The launch script exports a private CA bundle containing standard roots plus this server certificate. Use the launchers; a direct `nanobot` invocation needs the same SSL_CERT_FILE.

All 62 advertised tools, including exact input/output schemas, were saved to `results/mcp-tools.json`. Five server resources were read and cached in `~/.nanobot/workspace/integrations/ros-mcp/`. Configuration enables exactly seven read-only native MCP tools: get_arm_state, get_spatial_map, get_spatial_context, get_execution_state, get_manipulation_memory, get_floor_slots, get_world_state. Exact private-network exception: `10.211.55.3/32`. All physical and preparation-mutation functions remain excluded. No physical robot commands executed.

Read-only calls passed: get_arm_state returned ARM_STATE_READY; get_spatial_context returned SPATIAL_CONTEXT_READY; get_world_state returned state data. These labels do not prove freshness, physical readiness or a safety condition. Server docs identify current development as mock domain 89 with physical execution disabled; live hardware operation was not verified. Native Nanobot MCPProvider also registered the seven tools and completed get_arm_state; see `results/native-mcp-check.txt`. This does not yet test the Qwen-driven loop with MCP.

Current Nanobot MCP wrappers present the server's text-content JSON to the model. SDK discovery preserves outputSchema and raw structuredContent in evidence files, but this client does not separately forward structuredContent or outputSchema into model tool definitions. Responses tested include matching JSON text. With a restricted enabledTools list, native resources/prompts are skipped; local resource snapshots provide the workflow documentation without enabling all tools. Refresh snapshots after server/version changes. Large state responses may exceed the configured result limit; truncated state is not complete evidence.

Server workflow overrides the earlier generic retry policy: one physical attempt per prepared task; no automatic physical retries or alternative grasps after failure. Instructions updated accordingly. Physical execution, preparation pipelines and their timeout/reconnect behavior remain untested here.

Restart with `./start-remote-nanobot.command` for a new chat. Example: “Use mcp_ros2_get_arm_state to read the Ubuntu robot state. Report the returned status and TCP frame; do not move anything.” Token and trust files are under `.private/` (restricted permissions, git-ignored); token is also in restricted Nanobot config. Never share them.

## MCP natural-language troubleshooting (2026-11-03)

The user's process started before MCP configuration and remained in the same chat. Its guessed ROS shell commands were correctly blocked by the exact shell allowlist; the model had not loaded native MCP tools. Restart the running process, not merely send another message. A fresh-process test then exposed ContextWindowExceededError (35936 estimated tokens versus 31232 input budget). Workspace file tools were disabled to reduce tool-schema/context overhead within the unchanged 32K server limit. The minimal profile retains seven read-only ROS tools and the existing three-command shell allowlist; agent file reads/edits are unavailable. Workflow snapshots remain on disk for operator review. Fresh startup confirmed all seven MCP tools connected; model-loop evidence is in results/nanobot-mcp-natural-language.txt.

## Live perception client (2026-11-03, MCP 3.1.1)

Refreshed authenticated discovery now returns 11 tools in perception_only mode, not the previous 62-tool development mode. list_functions reports ROS domain 0 and one expected camera/feedback publisher per required topic. get_world_state reports robot.physical_grasp_execution_enabled=false. These were independently queried; the Ubuntu handoff's full 389-test suite was not rerun on the Mac.

Nanobot now enables the recommended eight tools: list_functions, get_world_state, get_arm_state, capture_observation, verify_presence, ground_object, segment_object, get_step_context. Tool timeout is 180 seconds, max output 1024 tokens, max tool result 6000 characters, context unchanged at 32768. Instructions were compacted with backup to avoid the previous context overflow. All physical and generic pipeline tools remain excluded; workspace file tools remain disabled. Full live server schemas and current resources are cached under integrations/ros-mcp in the Nanobot workspace.

Nanobot must capture a fresh observation and bind capture.image into verify_presence using the returned same-context ID. User target attributes must remain unchanged. A memory miss or inference failure cannot become an absence claim. ground_object uses the same image; segment_object binds that image and the returned pixel_box. Handles remain inside Ubuntu; the Mac does not copy image/cloud buffers through the model.

Restart an existing CLI to load configuration using ./start-remote-nanobot.command. The Ubuntu tunnel on port 18092 is independently operator-owned and must remain available, as must the live ROS/MCP stack. The Mac launcher only maintains the Mac inference tunnel; it does not restart Ubuntu services. Model-driven presence test evidence: results/nanobot-live-presence.txt. Do not treat successful discovery alone as completed perception validation.

Native Qwen-driven presence test PASSED: ordinary request “Do you see a beige cup?” caused capture_observation followed by verify_presence with target="beige cup", using the actual returned context ID and capture.image binding. Capture returned a current synchronized 640x480 observation; presence returned success=true, present=true, status=PRESENT. Nanobot correctly reported positive presence. End-to-end turn took 117.9 seconds, including reasoning; remote presence stage reported 19.2 seconds. No physical tools were enabled or called. This is one positive presence case, not a grounding-accuracy benchmark or completed robustness evaluation. Native grounding/segmentation, negative/unknown cases and repeated-query freshness remain to be tested. Evidence: results/nanobot-live-presence-transcript.json.

## All currently permitted MCP tools enabled

Nanobot now explicitly enables all 11 tools in the server's perception_only allowlist. This adds get_spatial_map, get_spatial_context and get_manipulation_memory to the eight primary perception tools. All advertised input/output JSON schemas passed schema validation; native MCP registration matched the 11-tool set and get_arm_state passed. This is NOT a claim that all 11 functions or the disabled 51 have passed behavioral tests. Coverage: results/mcp-validation-coverage.json.

Full 62-tool validation requires an Ubuntu-side safe test deployment: preserve live provenance checks for actual observations, isolate mock/synthetic execution tests from hardware, retain the physical gate, and reject physical dispatch. Current server restricts --require-live-inputs to --perception-only; do not simply remove this guard or switch the active live server to full mode. Client '*' would also register extra resources/prompts and increase context; explicit task-scoped allowlists remain preferable. Real motion/contact validation requires a separately authorized operator-supervised procedure. Existing historical mock reports must not be presented as current live validation.

## MCP 3.2 preparation profiles

Authenticated discovery independently confirmed MCP 3.2.0, preparation_only, 53 callable non-actuating tools. Ubuntu's per-tool report records 40 successful calls and 13 blocked/failed success paths; those are not all positively verified. Full preparation remains blocked. All physical tools remain excluded both by server mode and client profiles.

Launch a fresh chat with a task-scoped profile:

```sh
./start-remote-nanobot.command --tools-profile perception
./start-remote-nanobot.command --tools-profile view
./start-remote-nanobot.command --tools-profile memory
./start-remote-nanobot.command --tools-profile grasp
./start-remote-nanobot.command --tools-profile placement
./start-remote-nanobot.command --tools-profile diagnostics
./start-remote-nanobot.command --tools-profile queue
```

Without an explicit profile, the existing default tool list is retained. Profiles generate separate mode-600 configuration files under .private/profiles without replacing the default MCP allowlist. They share the existing workspace and server-side state, but open a new conversation. Do not switch profiles mid-task and expect chat context to transfer; retain real server handles if deliberately continuing. The queue profile exposes only the server's calculation-only dispatcher, not physical calls.

Tool counts are perception 11, view 9, memory 11, grasp 17, placement 21, diagnostics 14, queue 7. Native MCP registration was checked for each; initial prompt estimates after tokenizer initialization were 6.9K–13.3K tokens. This is configuration/registration/context verification, not positive end-to-end verification of each preparation recipe. Test evidence: results/mcp-profile-checks.json.

A calculation-only get_observe_view call with six zero deltas returned success=false, JOINT_LIMIT_VIOLATION for joint2=-0.048328 rad versus allowed lower bound -0.034907, moved_target=false and endpoint_safe=false. No motion dispatched. Failure/status output: results/native-view-calculation.txt. Never adjust limits or use fallback joints to force success. This is the current observed blocker, in addition to the stale feedback and batch parsing problems reported in the Ubuntu handoff.

Launcher fixes: remote startup no longer forces 512 output tokens; it preserves higher values and uses at least 1024. launch_nanobot.py initializes Nanobot's native background token counter before entry, with a bounded 10s wait, preventing startup from using its conservative byte-count fallback as though bytes were tokens. No Nanobot core files changed. The model/server still enforces the real 32K context limit.

## Camera-language axis correction (2026-11-04)

Checked VM source vamp_node/include/vamp_node/observe_view.hpp: relativeCameraView multiplies current optical camera rotation by Rz(yaw)*Ry(pitch)*Rx(roll). Correct single-axis gaze mapping is right=positive pitch, left=negative pitch, down=negative roll, up=positive roll. yaw rotates about optical viewing axis Z; it is not horizontal gaze. Translation uses optical X right/Y down/Z forward. These are calculation semantics, not a physical verification of calibration or achieved viewpoints.

Nanobot workspace instructions now include mappings, degree/radian examples, clarification for vague magnitudes and explicit interpretation of PARTIAL/RELAXED, fraction/errors, moved_target and endpoint_safe. No VM implementation or safety limits changed. A native natural-language request to calculate a rightward 3-degree look emitted get_observe_view with pitch=0.0523598776, yaw=roll=0 and zero translation. Evidence: results/camera-axis-natural-language.txt. No physical motion allowed or issued. Only the argument-selection correction is established by that call; returned IK/endpoint success must be judged separately.

## All-calculations profile

`./start-remote-nanobot.command --tools-profile all-calculations` exposes the reviewed 53 MCP tools from the preparation_only allowlist. Physical execution tools and commit_floor_slots are excluded. Local shell tools are disabled in this profile. The generator requires the cached discovery mode/allowlist to match the reviewed set; it does not dynamically enable arbitrary newly advertised tools. The server independently enforces preparation-only dispatch.

Verified configuration, native registration of all 53 tools and initial estimated input size (28,956 tokens with initialized tiktoken, versus a 32,768 context limit). This leaves limited history/result headroom. Long multi-tool tasks may require compaction or a narrower profile; this check is not an end-to-end verification of all tools or unlimited long-horizon execution. Existing failed/blocked preparation functions remain failed/blocked until their prerequisites are repaired. Default profile is unchanged. Physical execution remains disabled.

## Remote Qwen settings update (2026-11-04)

Backups of config, launchers and generated profiles: .private/backups/20261104-102730 (contains credentials; do not share). Defaults now use custom / Qwen/Qwen3.5-9B / http://127.0.0.1:18080/v1, placeholder key local, context 49152, output 1024, one concurrent subagent, idle compaction disabled (idleCompactAfterMinutes=0), reasoningEffort=none. Thinking is disabled via providers.custom.extraBody.chat_template_kwargs.enable_thinking=false, a supported provider extra-body mechanism rather than an invented root setting. All requested options are supported in installed Nanobot 0.4.0. Explicit context/provider/timeout options remain subject to the remote server's runtime semantics.

Both shell launchers persist NANOBOT_STREAM_IDLE_TIMEOUT_S=600 and NANOBOT_OPENAI_COMPAT_TIMEOUT_S=600 before Python starts. Remote launcher now preserves the 49152 setting rather than reverting to 32768; profile files inherit the updated defaults while retaining their own allowlists. MCP configuration/tools and channel settings were compared unchanged in the main configuration; other provider credentials retained. Existing SSH tunnel was reused once the operator reopened it; no tunnel was terminated or recreated by the agent.

Remote model discovery reports Qwen/Qwen3.5-9B and a loaded context of 65536. Native Nanobot provider tests passed text generation and an actual structured synthetic_add call with 17/25, local synthetic result 42 and final RESULT=42. No robot commands executed. Evidence: results/remote-qwen-verification.json, results/qwen-model-discovery.json, results/qwen-configuration-check.json. The restarted launcher test uses an isolated temporary config with MCP and shell disabled so testing cannot call robot tools; normal configuration remains intact. See results/qwen-launcher-restart.txt.

## Choose a remote Qwen backend

Start only one backend in a separate SSH terminal on `bincu008@192.168.50.128`.
For llama.cpp run `~/qwen35-server/run-console.sh`. For OpenVINO run
`~/qwen35-openvino/run-console.sh`.

On the Mac, run one of these from this directory:

```sh
./start-remote-nanobot-llama.command --tools-profile all-calculations
./start-remote-nanobot-openvino.command --tools-profile all-calculations
```

Both launchers accept the existing leading `--tools-profile` option and
forward Nanobot CLI arguments. A supplied `--config` serves as the base for
the per-launch copy. The launchers select the custom provider and
`Qwen/Qwen3.5-9B` with thinking disabled and 1,024 output tokens. The client context budget is
57,344 tokens for llama.cpp and 49,152 tokens for OpenVINO. They preserve the shared config and remove the private
per-launch copy on exit.

The llama.cpp launcher uses loopback port 18080 on both ends of the SSH tunnel.
OpenVINO uses 18081 on both ends. The launcher checks the exact model ID at
`/v1/models` before starting Nanobot. Add `--check` as the first argument to
run discovery and exit without starting Nanobot:

```sh
./start-remote-nanobot-llama.command --check
./start-remote-nanobot-openvino.command --check
```

A pre-existing listener is reused only when its SSH command line confirms the
matching host and explicit forward. If the launcher reports a port conflict,
inspect the listener before retrying. The launcher closes only its own
tunnel when it exits. Both use the existing 1,800-second Nanobot stream idle
and OpenAI-compatible request timeout settings.

## ROS task profile

For normal robot queries and calculation requests, launch the eight-tool profile
against the existing Ubuntu MCP endpoint:

```sh
cd ~/Documents/NanoBot
./start-remote-nanobot-llama.command --tools-profile robot-tasks
```

Use `start-remote-nanobot-openvino.command` instead if that backend is running.
Exit any existing Nanobot session before switching profiles. Each new session
connects through native MCP and rediscovers its tool definitions. Example request:
“Where is the tool center point of the arm right now?”

`robot-tasks` registers exactly these native ROS MCP tools:

- `inspect_robot`, `observe_scene`, `plan_camera_view`
- `prepare_manipulation`, `validate_plan`
- `get_task_status`, `cancel_task`, `describe_workflow`

The terminal displays the existing `ros2::` namespace. The profile uses
`robot-tasks-workspace/AGENTS.md`, suppresses built-in tools for the task session,
and retains the configured MCP endpoint, credentials, certificate validation, and
180-second MCP call timeout. Detailed MCP resources are cached locally for
inspection and are excluded from the model prompt. The `all-calculations` profile
still selects the original 54 tools for diagnostics and coverage testing.

The task tools handle internal contexts, step IDs, bindings, and sequencing.
For a running job, keep the returned `task_id` for status and cancellation.
Treat simulated results as mock evidence. Planning tools only calculate.

To refresh the checked MCP 3.6.0 mock discovery snapshots and rerun the
profile integration checks:

```sh
SSL_CERT_FILE="$PWD/.private/ca-bundle.pem" .venv/bin/python tests/verify_robot_tasks.py
```

The test verifies mock replay and preparation-only mode before starting any
calculation. It writes native discovery, translated definitions, the task
resources, and structured outcomes under `results/`.

### Task references after compaction

The `robot-tasks` launcher saves one rolling, private record per submission/task
under `.private/robot-task-checkpoints/`. It records the original structured
submission before dispatch, then the server's task ID, last known status and
source, and the exact plan reference when available. It does not store the full
MCP payload or workflow graph. References are scoped to the Nanobot session.
The model gets a bounded selection of recent references when its context is
built or compacted. The server remains the source of truth for status.

Say “continue” in the same session to follow an existing task. To reuse the same
checkpoint scope across launcher restarts, choose a stable session ID:

```sh
./start-remote-nanobot-llama.command --tools-profile robot-tasks --session robot-work
```

Use the same session ID when relaunching. The checkpoint does not start any task
on launch. To exercise the installed compaction path with a synthetic receipt:

```sh
.venv/bin/python tests/check_task_checkpoint.py
```


## Staged workspace profile (MCP 3.7.0)

`./start-remote-nanobot-llama.command --tools-profile workspace-tasks` selects nine
non-actuating tools: inspect_robot, get_camera_image, check_freshness, list_objects,
observe_scene, plan_camera_view, prepare_manipulation, get_task_status, cancel_task.
Do not run until the updated MCP/backend are available and dedicated workspace
snapshots have been refreshed. The live MCP server was stopped on request.
Development verification used mock domain 89 and TLS MCP port 8766; it is not live
hardware verification. Existing connection credentials/default profiles were not
modified. The updated backend requires the tested Ubuntu overlay to be sourced.

workspace_arguments.py hides request/task/plan bookkeeping from this profile;
task_checkpoint.py persists the original receipts. Status/cancel default to the
current server task. camera_visuals.py forwards a successful requested camera
image as visual provider input, not textual base64. It does not attach older images
after a failed read or a new user turn. Existing robot-tasks/diagnostic profiles
remain separate. Physical tools remain forbidden.

Portable tests: `python3 tests/test_workspace_adapters.py`. Real remote Qwen
selected get_camera_image({}) with the nine schemas and described mock pixels in
a direct request-assembly test. Native Mac launcher/provider/compaction integration
has NOT been exercised here. Automatic native CLI image display remains pending;
images are saved as artifacts and passed to the model.

Freshness currently supports image/object only; object verification still uses
the existing shorter 60s limit during migration. Single-pool/10-minute execution
cache changes and physical facades are incomplete. A full preparation regression
found a ROS NumPy integer seed serialization crash; code/isolated transport fix
passed but the interrupted mock task remains UNKNOWN with its ledger barrier.
Do not reset the ledger, replay that task or claim complete preparation passed.

The opt-in `workspace-control` profile adds three guarded action tools to the nine
workspace tools. It requires dedicated control discovery snapshots from an enabled,
live-source server (`tests/discover_workspace_mcp.py --control --expect-source
live_driver_publishers`), and fails closed for mock/preparation discovery. Default
configuration is unchanged. Request IDs and the exact reviewed plan pair are private
checkpoint fields; preparation failure cannot fall back to an older plan. Native
Mac execution remains unverified; portable adapter checks do not establish robot
readiness. See the Ubuntu `docs/mcp/CONTROL_TOOLS.md` handoff before enabling.
