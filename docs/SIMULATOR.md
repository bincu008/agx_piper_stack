# Student simulator: one pipeline, four services

Launch the simulator once. It keeps **one empty task queue** running. Add the tasks
you want, display them, execute the whole queue, or flush it. There are no session
files or per-stage execution commands.

Recorded RGB-D and arm feedback come only from the mock node. Grounding uses
LM Studio, masks use EdgeSAM, and grasp/placement calculations use the existing
production APIs. This is **calculation replay, not physics or physical motion**.
It does not move an arm or gripper. The recording stays unchanged after a run.

## Build

Use Ubuntu 24.04 and ROS 2 Jazzy; see [Dependencies](../DEPENDENCIES.md) for ONNX
Runtime and model assets. From your workspace root:

```bash
git switch simulator
source /opt/ros/jazzy/setup.bash
export ONNXRUNTIME_ROOT="$HOME/.local/onnxruntime-1.15.1"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Load `robobrain2.5-4b` in LM Studio and start its local server on port 1234.
Download model assets beforehand for a PC without internet access. The bundled
recording contains a cup, two white bowls, a raised plate and a dark tray.

## Launch once

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/agx_piper_stack/install/setup.bash"
export ROS_DOMAIN_ID=89
ros2 launch mock_hardware_node simulator.launch.py rviz:=true
```

The launcher starts recorded sensors, the official calculation stack and the
persistent simulator pipeline. Physical execution is always disabled. No hardware
bringup/controller is started. Keep real drivers out of domain 89. The pipeline
rejects live/mixed publishers and an enabled physical gate before execution.

LM Studio defaults to `http://127.0.0.1:1234/v1`. If ROS runs in Ubuntu VM and
LM Studio runs on its Mac host, use the host address instead:

```bash
ros2 launch mock_hardware_node simulator.launch.py \
  api_server_url:=http://10.211.55.2:1234/v1 \
  model_name:=robobrain2.5-4b rviz:=true
```

Stop your old simulator launch and start this updated launch once after updating.
Stop the launch with Ctrl+C when finished. Queue state lives in memory and is lost
on restart; it is never automatically executed after restart.

## Four service calls

In another terminal, source ROS/workspace and export `ROS_DOMAIN_ID=89` as above.

### 1. Add tasks

This adds **only these nine tasks**, in order, and does not execute them:

```bash
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
"{object_name: 'beige cup, grasp its body', destination_name: 'dark rectangular tray', lift_distance_m: 0.05, tasks: [observe, pin_grasp, pin_placement, presence_object, ground_object, segment_object, remember_object, prepare_grasp, grasps]}"
```

You can add consecutive tasks in multiple service calls before execution. Empty
object/destination fields reuse the previous targets. Zero `lift_distance_m` uses
5 cm initially, or the previous batch's setting. A queued batch cannot change its
targets; flush it first. Missing prerequisites, duplicate tasks, physical functions
and incorrect order are rejected without modifying the queue.

### 2. Display the queue

```bash
ros2 service call /simulator/display_pipeline std_srvs/srv/Trigger '{}'
```

Returns JSON in `message`: `busy`, `unknown`, object/destination names,
`current_task`, queued `tasks` (stage IDs and actual API names), and `last_run`.
This reads pipeline state; it does not capture a camera frame.

### 3. Execute the entire inserted queue

```bash
ros2 service call /simulator/execute_pipeline std_srvs/srv/Trigger '{}'
```

Waits for **final calculation completion**, not just queue-start acknowledgement.
The response contains success/failure, the actual master status and ordered
per-task results/timings. No extra tasks are inserted. With the nine tasks above,
it stops after grasp candidates; destination/placement calculations do not run.

On success the queue becomes empty. On a definite task failure, remaining tasks
are flushed and the service returns `success: false`. `last_run` remains visible.
You can add the same nine tasks again into the same running pipeline.

While executing, display remains available. Add/flush/another execute is rejected
while busy. A timeout or unresolved `STATE_UNKNOWN` blocks further operations;
it is never automatically replayed or erased. A disconnected CLI does not prove
the ROS operation stopped. Physical execution remains disabled independently.

### 4. Flush queued tasks

```bash
ros2 service call /simulator/flush_pipeline std_srvs/srv/Trigger '{}'
```

Clears an idle task queue and its calculation-result context without execution.
It is not cancellation of an in-flight API, nor a robot safety/ledger reset.

## Full example: cup grasp and placement planning

Start the simulator once using [Launch once](#launch-once), and keep it running.
This example plans the complete cup pick/place sequence using the mock recording.
It does not execute arm/gripper motion, simulate physical contact, or change the
recording. Full motion preparation can still fail; inspect the actual result.

In a second terminal:

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/agx_piper_stack/install/setup.bash"
export ROS_DOMAIN_ID=89
```

Start with an empty queue. If you already have unwanted pending tasks, discard
them first with `/simulator/flush_pipeline` as described above. Insert **all 21
tasks before executing**. Each command below inserts exactly one task. Check each
response for `success: true`; stop if an insertion is rejected.

```bash
# 1. Capture synchronized replay RGB-D and arm feedback.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
"{object_name: 'beige cup, grasp its body', destination_name: 'dark rectangular tray', lift_distance_m: 0.05, tasks: [observe]}"

# 2. Preserve the grasp observation.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [pin_grasp]}'

# 3. Preserve the placement observation.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [pin_placement]}'

# 4. Check cup presence through RoboBrain.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [presence_object]}'

# 5. Ground the cup bounding box.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [ground_object]}'

# 6. Segment the cup.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [segment_object]}'

# 7. Extract and store the cup point cloud.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [remember_object]}'

# 8. Prepare grasp inputs.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [prepare_grasp]}'

# 9. Calculate and validate grasp candidates.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [grasps]}'

# 10. Register the cup.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [register_object]}'

# 11. Select a ranked grasp for the green/blue finger preview.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [select_grasp]}'

# 12. Check tray presence through RoboBrain.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [presence_destination]}'

# 13. Ground the tray bounding box.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [ground_destination]}'

# 14. Segment the tray surface.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [segment_destination]}'

# 15. Store the tray point cloud.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [remember_destination]}'

# 16. Cache the selected grasp after inference calls.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [cache_grasp]}'

# 17. Calculate the tray grid and fitting placement slots.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [find_slots]}'

# 18. Calculate a destination viewing candidate.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [plan_look]}'

# 19. Validate complete approach, lift, placement and retreat motion.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [prepare_pick_place]}'

# 20. Check the prepared plan without actuation.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [check_pick_place]}'

# 21. Read the prepared execution state without dispatching motion.
ros2 service call /simulator/append_tasks my_actions/srv/AppendSimulatorTasks \
'{tasks: [execution_state]}'

```

Display the queue and confirm that it contains 21 tasks, ending at `execution_state`:

```bash
ros2 service call /simulator/display_pipeline std_srvs/srv/Trigger '{}'
```

Execute the entire inserted batch once:

```bash
ros2 service call /simulator/execute_pipeline std_srvs/srv/Trigger '{}'
```

The call waits for final completion and reports ordered API results/timings.
`success: true` with all 21 tasks successful means the complete calculation and
prepared-plan check succeeded. It does not mean the arm moved or the cup was
physically placed. If `prepare_pick_place` fails, fitting slots or a selected
grasp alone are not proof of a complete plan.

Inspect RViz after the call. The selected-grasp display shows green grasp fingers
and blue pregrasp fingers. Successful complete preparation additionally publishes
prepared grasp, lift, placement, withdrawal/retreat milestones and the predicted
placed-object cloud. The complete plan may use another grasp after candidate
trials; use its prepared milestone displays to review the final plan.

The queue empties automatically after completion or a definite failure. Read the
previous run and its actual failing task, if any:

```bash
ros2 service call /simulator/display_pipeline std_srvs/srv/Trigger '{}'
```

To repeat, reinsert the 21 tasks into the same running pipeline and execute again.
There is no new session or simulator relaunch. To discard an idle pending batch:

```bash
ros2 service call /simulator/flush_pipeline std_srvs/srv/Trigger '{}'
```

## Reference order

Add only the consecutive prefix you need. Every batch starts at `observe` because
completion/failure flushing clears predecessor bindings. To calculate a full
pick/place plan, explicitly insert all 21 stages before executing. For grasp
candidates only, insert the first nine as shown above.

| Stage ID | Existing public API | Purpose |
| --- | --- | --- |
| observe | capture_observation | Synchronized replay RGB-D and pose |
| pin_grasp | pin_grasp_observation | Preserve grasp observation |
| pin_placement | pin_placement_observation | Preserve placement observation |
| presence_object | verify_presence | VLM object presence |
| ground_object | ground_object | Object bounding box |
| segment_object | segment_object | Object mask |
| remember_object | store_segmented_target | Active object point cloud |
| prepare_grasp | prepare_grasp_input | Prepare grasp inputs |
| grasps | calculate_grasps | HGGD/refined validated candidates |
| register_object | update_object | Register object |
| select_grasp | select_grasp | Choose ranked candidate |
| presence_destination | verify_presence | Tray presence |
| ground_destination | ground_object | Tray bounding box |
| segment_destination | segment_object | Tray mask |
| remember_destination | store_segmented_target | Destination point cloud |
| cache_grasp | cache_grasp_plan | Cache grasp after inference |
| find_slots | find_placements | Tray grid and object slots |
| plan_look | plan_next_best_view | Calculate viewing candidate |
| prepare_pick_place | prepare_pick_place | Complete motion preplanning |
| check_pick_place | validate_prepared_pick_place | Non-actuating cache check |
| execution_state | get_execution_state | Read waiting physical stages only |

`append_tasks` input schema: object/destination strings, `lift_distance_m` in
metres, and `tasks` as an array of stage IDs. Its response is `success`, `status`
and `queued_tasks`. Other services use `std_srvs/Trigger` (`success`, `message`).
Source headers, camera calibration, base-link coordinates, typed bindings,
revision checks and underlying motion validation retain their existing semantics.
These four simulator endpoints do not change production MCP/pipeline services.

## RViz and failures

The grasp preview appears after **`select_grasp`**, the eleventh task. A batch
ending at `grasps` calculates valid candidates but does not publish the selected
pose on this topic. Include `register_object` and `select_grasp` to see it.

The simulator publishes a latched `/simulator/selected_grasp` MarkerArray:

- **Two green bars:** selected-grasp finger bodies, with dimensions/TCP offset
  read from VAMP and separation from the selected candidate's jaw opening.
- **Green arrow:** the selected valid grasp.
- **Two blue bars and blue arrow:** that candidate's pregrasp pose, using the
  same finger dimensions and jaw opening as the selected grasp.

This is the ranked candidate chosen by `select_grasp`, not necessarily the first
candidate generated. It has passed grasp-candidate validation. **It has not yet
passed complete approach, lift, transfer and placement motion validation.** A
visible selected grasp therefore does not mean a complete pick/place plan exists.

Complete motion validation happens later at `prepare_pick_place`. That stage can
reject this candidate and try another grasp. When successful, inspect the separate
prepared grasp/placement milestone displays for the actual complete plan; its
grasp may differ from the earlier selected-grasp preview.

The default debug RViz includes **Simulator Selected Grasp - Calculation Only**.
If adding it manually, use a MarkerArray display on `/simulator/selected_grasp`,
set Durability to **Transient Local**, and use fixed frame **base_link**. The
preview remains after queue cleanup and is cleared at the next calculation start.
No selected-grasp preview after only `grasps` is expected; it does not indicate a
placement rejection. Before this display was added, the prepared-grasp display
alone stayed empty until a complete motion plan was prepared.

RViz uses the existing grasp debug displays for clouds, grid and
prepared milestones. Earlier candidate markers can remain visible after failure;
check `last_run`, not markers alone. No simulated contact/tracking is claimed.

- `PRESENCE_UNKNOWN`: inspect the LM Studio URL/model/response. Unknown is not absent.
- No valid grasp/IK/path: the calculation failed; remaining tasks are flushed.
- Empty queue: add tasks before execute.
- Out-of-order insert: use the next stage in the reference table.
- Unknown outcome: inspect node/master logs; do not replay work or reset a safety ledger.

The removed `simulator_task --session/--next/--until/--stage/--all` interface is no
longer supported. No client session JSON or per-run archive files are created.
The node prints final results to its ROS log and retains the previous run in memory.

## Developer checks

```bash
python3 -m pytest -q src/task_pipeline/test/test_simulator.py
```

Fixtures use the real typed result store with synthetic, non-actuating backends.
They verify insertion order, exact whole-queue execution, failure/completion
flushing, busy/concurrent access and unknown-state blocking. Runtime replay is
not a substitute for physical robot validation.
