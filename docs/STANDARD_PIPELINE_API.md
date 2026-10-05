# Shared microstep API

The official reasoning stack starts `task_pipeline/pipeline_node`. Its catalog in
`task_pipeline/registry.py` is the common source for production controllers,
declarative recipes and the synthetic-box benchmark. Public engine calls at
`/pipeline/*` run through `MasterStepRuntime` inside `PiperMasterControl`.
The master owns the queue, typed results, grasp caches and arm state. Existing
numerical engines and their solvers are retained.

## Public operations

| Purpose | Canonical functions (prefix `/pipeline/`) |
|---|---|
| Observe | `capture_observation`, `pin_grasp_observation`, `pin_placement_observation`, `get_arm_state` |
| Interpret | `verify_presence`, `ground_object`, `segment_object`, `infer_vision`, `infer_yolo` |
| Find a grasp | `prepare_grasp_input`, `calculate_grasps`, `calculate_cloud_grasps`, `select_grasp` |
| Store grasp state | `cache_grasp_plan`, `get_cached_grasp`, `cache_grasp_refinement` |
| Refine and approach | `move_pregrasp`, `refine_grasp`, `plan_grasp_approach`, `plan_final_approach` |
| Complete grasp | `open_grasp_gripper`, `execute_grasp_approach`, `close_grasp_gripper`, `verify_grasp_contact` |
| Place | `find_placements`, `search_placements`, `validate_placements`, `plan_placement` |
| General motion | `solve_pose`, `plan_motion`, `move_to`, `execute_trajectory`, `set_gripper` |
| Object state | `update_object`, `resolve_object`, `get_world_state` |
| Microstep control | `run_step`, `store_step_result`, `submit_steps`, `get_step_context`, `clear_step_context` |
| Wait for queued work | `/pipeline/run_steps` (`my_actions/action/RunPipelineSteps`) |

The catalog also retains navigation/recovery functions and old task interfaces.
`execute_grasp`, `plan_layout` and `look_at_target` remain compatibility facades;
they construct sequences of shared calls. Engine internals such as individual
IK iterations stay inside their numerical operation rather than becoming RPCs.

Discover all 55 service/action entries, request/response schemas and completion
semantics with:

```bash
ros2 service call /pipeline/list_functions my_actions/srv/GetPipelineFunctions '{}'
ros2 topic echo /pipeline/timings
ros2 service call /pipeline/get_metrics my_actions/srv/GetPipelineMetrics '{}'
```

## Constructing a sequence

Each `PipelineStep` has a unique ID, catalogued function name, CDR-serialized typed
ROS request, predecessor field bindings, and a `stop` or `continue` failure policy.
Bindings copy typed fields from successful earlier responses; they do not evaluate
Python expressions or accept arbitrary backend endpoints. A failed predecessor
cannot supply a result to another step.
The reserved source `@response_cdr` binds a complete serialized response, for
example from `refine_grasp` directly into `cache_grasp_refinement.refinement_cdr`.

1. Build steps with `task_pipeline.recipes.step` and existing `my_actions` types.
2. Call `submit_steps`; retain the returned context ID and revision.
3. Send that ID to the `run_steps` action. Its result waits for completion, failure
   or cancellation at a step boundary; feedback identifies the current function.
4. Query `get_step_context` for queue metadata and optionally typed response CDR.
5. Call `clear_step_context` when the results are no longer needed.

For example, the calculation-only cup recipe binds `observe.image` to
`ground_cup.image`, `ground_cup.pixel_box` to `segment_cup.bbox`, and the selected
grasp and generated placements to `placement_ik`. The master resolves these
bindings when each step runs. Controllers choose the task order; they need not
carry their own copies of every intermediate image or pose.

`run_step` executes one typed request and retains its response. With the same
context, step ID and request payload, a retry returns the retained result instead
of executing again. A different payload under the same ID is rejected.
`store_step_result` imports a successful typed response for later consumers; it
does not execute that response or authorize motion.

Appending is atomic, rejects forward/missing dependencies and checks an optional
expected revision. Attempted steps leave the pending queue; results remain until
cleared. `continue` permits independent later objects to run after a failure.
Defaults bound storage to 32 contexts, 256 steps per context and 256 MiB of CDR
payloads. Completed automatic single-call contexts may be evicted; explicitly
named or submitted contexts must be cleared by their owner.
The box benchmark retains its latest run's context IDs in the report and releases
them before its next run, so repeated runs do not accumulate stale result memory.

## Launch and run

Source the environment in each terminal:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export LD_LIBRARY_PATH="$HOME/.local/onnxruntime-1.15.1/lib:$LD_LIBRARY_PATH"
```

Start the mock and safe stack in separate terminals:

```bash
ros2 launch mock_hardware_node mock_hardware.launch.py
```

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=lmstudio placement_backend:=heightmap \
  grasp_candidate_backend:=hggd pipeline_profiling:=true \
  allow_physical_grasp_execution:=false
```

Use `inference_profile:=server` for the optimized remote RoboBrain server.
Vision and VAMP remain composed; the VLM runs in its own process so network
callbacks do not occupy the perception/planner executor. Its multithread executor
also permits API responses to arrive while a legacy command callback waits.
Camera and arm drivers
remain separate and should be stopped for mock-only validation.

Run a 14-step cup calculation and wait for its final result:

```bash
ros2 run task_pipeline pipeline_task --object 'brown cup' --destination tray \
  --output "$HOME/cup_pipeline.json"
```

Use `--grasp-only` for its first ten steps. This recipe calculates poses and updates
the grasp cache; it contains no actuator commands. For the calculation-only box
benchmark, start its node and call its blocking trigger:

```bash
ros2 launch placement_node box_benchmark.launch.py \
  box_count:=10 optimize_pipeline:=true plan_pick_place_motion:=true
ros2 service call /box_benchmark/run std_srvs/srv/Trigger '{}'
ros2 launch vamp_node grasp_debug_rviz.launch.py
```

## Execution and interrupt boundaries

New actuator APIs retain `allow_physical_grasp_execution`, collision checks and
master busy-state checks. General `move_to` and `execute_trajectory` are limited
to empty-hand motion and use VAMP validation. General `set_gripper` permits
empty-hand opening; closing requires the ordered validated grasp stages.
Releasing a held object requires a validated placement execution implementation,
which is not supplied by these calculation APIs.

The grasp stages preserve the fixed-orientation, bounded Cartesian approach,
measured pre-grasp checks, minimum valid opening, 5% approach speed and restoration
to 20%. Closing occurs only after the final waypoint; feedback must confirm
contact before master state marks an object held. Protective speed changes stay
atomic inside the approach execution rather than becoming optional queued steps.

Action cancellation stops between steps. It cannot forcibly cancel an in-flight
ROS service. Emergency-stop and hardware-driver paths remain direct. A timeout
reports `BACKEND_TIMEOUT_STATE_UNKNOWN`; do not automatically retry an actuator
operation. Such a context rejects further work until actual state is reconciled
and the owner explicitly starts a new context. Existing legacy task interrupts
remain available; this change does not replace the hardware stop mechanism.

Master parameter `pipeline_step_timeout_sec` and dispatcher `request_timeout_sec`
default to 600 s; capture timeout is 30 s. RoboBrain inference keeps its separate
configurable 120 s timeout. Synchronized source images, clouds and manual
FK × calibration transforms retain their association; no computational TF lookup
was introduced. Placement can use the exact pinned observation after newer
frames have evicted it from the rolling buffer.

## Timing and validation

`PipelineTiming` separates `api_inclusive`, `backend_inclusive`, engine compute
and validation scopes. These are nested measurements: do not add them together.
Use action/run elapsed time for total completion; compare repeated functions
with the same input mode and full-motion flag.

Synthetic boxes use `calculate_cloud_grasps` with known geometry rather than HGGD.
They share public placement search, VAMP validation and motion planning with
deployment. Tray grounding/segmentation still uses real image APIs. Legacy
single-object placement is labelled `legacy_pose_only`; the full-motion benchmark
uses a frozen scene and requires valid lift, transfer and descent paths.

Software-only validation on 2026-10-05 accepted all ten boxes: six on the floor,
four stacked, zero failed, in 230.08 s. Median added API transport overhead was
about 24 ms per call when comparing API/backend medians. Placement-search compute was slower than the prior run;
this refactor does not claim a pipeline speedup. The 14-step brown-cup run found
a safe HGGD grasp and eight height-map candidates in 21.47 s, then correctly
rejected all eight at placement IK. That reachability limitation remains unresolved.
The legacy `[FIND_GRASP] brown cup` command also returned one safe, master-cached
candidate in 12.58 s after the standalone VLM executor fix.

Affected package checks passed: task_pipeline 30, placement_node 94, vlm_node 7,
vamp_node 48, plus 31 master-node functional tests. New API modules passed flake8
and docstring checks. Live actuator calls were rejected by the disabled physical
gate; live context checks verified idempotent retry and explicit result cleanup.

No physical execution was tested. Local build/test logs, per-step cup results and
the ten-box report are under `~/microstep_pipeline_validation/`. Rebuild and source
the workspace before restarting clients because shared ROS interfaces changed.
