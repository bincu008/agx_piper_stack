"""Explicit allowlist; descriptions never grant backend access."""

READS = {
    'get_spatial_map': (
        'Read ActiveVision 30cm grid in base_link: X [0,.6), Y [-.6,.6), Z [0,.6). '
        '16 cells, stable IDs, ROS last_scanned time (zero=never), latest observed '
        'cell IDs and camera pose/forward/up. Sampled valid depth rays mark traversed '
        'empty and hit cells, not cells behind hits. >=3 rays qualify; not full-cell '
        'coverage or collision clearance. header is last observation time, not query time.'),
    'get_spatial_context': (
        'Read shared spatial map plus current TCP and remembered object positions/cell '
        'IDs. This reads stored memory, not fresh visual evidence. Missing object names '
        'mean UNKNOWN, never visually absent. Empty cell ID means outside summary '
        'bounds; it does not mean absent. Objects may be stale hints. '
        'No capture, inference or actuation. Compare independent TCP/map timestamps.'),
    'get_arm_state': (
        'Read joint feedback (radians), TCP PoseStamped (metres/quaternion, header '
        'frame), and jaw width (metres). Empty input. ARM_STATE_READY is not a '
        'freshness or motion-safety certificate.'
    ),
    'get_execution_state': (
        'Read the prepared session for exact object_name. Returns revision, phase, '
        'source header, expiry seconds, completed/waiting stages and physical gate. '
        'STATE_UNKNOWN_REQUIRES_RECONCILIATION prohibits continuation even when '
        'success=true.'
    ),
    'get_manipulation_memory': (
        'Read bounded object/destination names, base_link positions in metres, geometry '
        'handles, revisions and queues as opaque state_json. Empty input. Location '
        'hints alone do not authorize manipulation. Missing memory entries mean UNKNOWN, '
        'not visually absent; capture and run image-bound verify_presence to inspect vision.'
    ),
    'get_floor_slots': (
        'Read destination slot reservations as opaque state_json. Empty input; does not '
        'reserve or commit cells.'
    ),
    'get_step_context': (
        'Read the exact owner-created context_id; include_results selects stored typed '
        'results. Does not submit, run, clear or cancel tasks. A queue snapshot is not '
        'final action completion.'
    ),
    'get_world_state': (
        'Read master world state_json. Empty input. The ROS response has no success '
        'flag: no completion or safety outcome can be inferred.'
    ),
}

COMMON = (
    '  Read-only, final service response/snapshot, never an asynchronous start '
    'acknowledgment. No prerequisites except an available ROS service; missing '
    'session/context is a backend failure. No supplied task/object/attempt identity is '
    'inferred. Inspect raw response and its status; running/cancelled queue states '
    'remain inside state_json/results, not fabricated from transport success. 5 second '
    'timeout; timeout/cancellation/disconnect does not cancel ROS work and gives no '
    'robot-stop guarantee. Read retries are allowed and yield a new snapshot; there is '
    'no actuator retry. Next: inspect related state/resources and tool availability. '
    'ROS strings/enums remain exact; timestamps use ROS sec/nanosec; '
    'revisions are unsigned integers. No default coordinate frame overrides message '
    'headers.'
)

CALCULATIONS = {
    'get_observe_view': (
        'Calculate a nearby camera view without motion. Inputs dx/dy/dz metres in '
        'current optical camera axes (+X right,+Y down,+Z forward); roll/pitch/yaw '
        'radians with Rnew=Rcamera*Rz(yaw)*Ry(pitch)*Rx(roll). Translation norm <=0.20m, '
        'each rotation <=pi. Requires fresh synchronized manual FK*X, idle empty hand. '
        'Returns source_joints/target_joints in radians, requested/achieved camera '
        'poses, errors, fraction and endpoint_safe. EXACT/RELAXED/PARTIAL indicate '
        'a new checked endpoint; TARGET_COLLISION/NO_REACHABLE_CHANGE/NO_CHANGE '
        'return current joints with moved_target=false, endpoint_safe=false. '
        'Missing/stale feedback/calibration is an explicit failure. Call move_to '
        'only when success,moved_target,endpoint_safe are all true; move_to must '
        'still plan the complete path. Calculation may capture/cache observations '
        'but never commands motion. Inclusive timeout 30 seconds.'),
}

ALLOWLIST = frozenset((*READS, *CALCULATIONS, 'list_functions'))

# Discovery is separate from authorization. These names are existing ROS APIs;
# they never enter ALLOWLIST or the ROS bridge's client map.
MANIPULATION = {
    'move_to': (
        'Execute empty-hand motion to six target_joints in radians. One call asks '
        'the master/VAMP to plan and execute the waypoint path, not one RPC per '
        'microstep. Requires idle empty hand, fresh state, physical master gate '
        'and no active/unknown gateway task. Uses request_id but no execution_ref. '
        'Master limits/collision/path/feedback checks remain active. success and '
        'TARGET_REACHED confirm completion; any uncertain dispatched result blocks '
        'all later actions. No automatic actuator retry. Endpoint IK alone is '
        'not a validated path. Master confirmation timeout 300 seconds.'),
    'execute_plan': (
        'Execute one complete prepared segment: look, pregrasp, approach, lift, '
        'transfer or descent. Requires the exact object_name and returned '
        'plan_revision, a complete unexpired prepared plan and the ordered predecessor. '
        'Prepared order: look -> pregrasp -> gripper_open -> approach -> '
        'gripper_close -> lift -> transfer -> descent. ROS returns success, '
        'motion_completed, phase and status after its execution attempt.'),
    'gripper_open': (
        'Open the gripper to the frozen plan width after AT_PREGRASP. Requires '
        'object_name and plan_revision from the prepared plan. width_m is metres, '
        'force_n is newtons; zero selects the prepared default, nonzero must match '
        'it. Next: execute_plan approach. ROS returns success, contact_confirmed, '
        'phase and status; opening does not verify a grasp.'),
    'gripper_close': (
        'Close at the confirmed final grasp and verify fresh contact in the same '
        'prepared session. Requires AT_GRASP, object_name and plan_revision. '
        'width_m is metres, force_n is newtons; zero selects prepared defaults, '
        'nonzero must match. Require success and contact_confirmed with phase '
        'CONTACT_CONFIRMED before execute_plan lift. Command acknowledgment alone '
        'is not verified contact.'),
    'execute_placement_transfer': (
        'Follow the prepared held-object path from LIFTED to AT_PREPLACE. Requires '
        'current verified contact, matching object_name/plan_revision and successful '
        'lift. Next: execute_placement_descent. ROS returns success, motion_completed, '
        'gripper_released and status; inspect completion rather than an ACK.'),
    'execute_placement_descent': (
        'Lower the held object from AT_PREPLACE to PLACED along the cached path. '
        'Requires current verified contact, matching object_name/plan_revision and '
        'completed transfer. Next: release_placed_object. ROS returns success, '
        'motion_completed, gripper_released and status.'),
    'release_placed_object': (
        'Release at the validated placement after completed descent. Requires '
        'matching object_name/plan_revision and PLACED. Confirm success and '
        'gripper_released before committing occupancy and withdrawal. ROS returns '
        'success, motion_completed, gripper_released and status; it updates master '
        'attachment/destination state only through the existing implementation.'),
    'commit_floor_slots': (
        'Record occupied destination grid cells; state-changing but not an actuator. '
        'selected_pose is the source-header-associated PoseStamped from the prepared '
        'selected_pose (metres/quaternion, base_link); do not invent a pose/header. '
        'In the physical recipe it follows confirmed release. reset clears '
        'reservations; validate_only previews without consuming cells. Keep both '
        'false for the ordinary commit. ROS returns success, status and occupied_slots. '
        'A commit does not prove physical placement. Next: execute_withdrawal.'),
    'execute_withdrawal': (
        'Pull the released empty gripper clear along the cached withdrawal. Requires '
        'RELEASED, matching plan_revision and the prepared destination_name as '
        'object_name in the standard recipe. Next: execute_placement_retreat. '
        'ROS returns success, motion_completed, gripper_released and status.'),
    'execute_placement_retreat': (
        'Follow the cached empty-hand retreat after withdrawal completes. Requires '
        'RELEASED, completed withdrawal, matching plan_revision and destination_name '
        'as object_name in the standard recipe. Complete retreat retires executable '
        'ownership after confirmed completion/speed restoration. Next: capture a '
        'new observation and prepare the next object. ROS returns success, '
        'motion_completed, gripper_released and status.'),
}

DISABLED_COMMON = (
    ' DISCOVERABLE ONLY: MCP EXECUTION DISABLED. This tool is not forwarded to ROS, '
    'does not queue work and cannot move the robot or consume cells. The current '
    'MCP result is final immediate rejection: isError=true, '
    'transport_outcome=disabled, dispatched=false, response=null. The exact ROS '
    'request and response schemas are in _meta.rosPipeline; they describe the '
    'underlying API, not an enabled MCP execution path. No backend timeout, '
    'cancellation or actuator retry occurs. Repeated calls remain rejected and '
    'consume no grasp attempts. If future execution becomes available, timeout '
    'or disconnection must be treated as unknown until reconciled; never replay '
    'a potentially executed action. Supported now: inspect get_execution_state, '
    'get_step_context and workflow resources. Do not route around this gate.'
)

DISCOVERABLE = ALLOWLIST | frozenset(MANIPULATION)
