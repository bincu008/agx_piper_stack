"""Generate protected, task-scoped Nanobot configs; no robot calls or server changes."""
import json,pathlib,sys
from nanobot.config.schema import Config
TASK_TOOLS = (
    'inspect_robot', 'observe_scene', 'plan_camera_view', 'prepare_manipulation',
    'validate_plan', 'get_task_status', 'cancel_task', 'describe_workflow',
)
WORKSPACE_TOOLS = ('inspect_robot', 'get_camera_image', 'check_freshness', 'list_objects',
                   'observe_scene', 'plan_camera_view', 'prepare_manipulation',
                   'get_task_status', 'cancel_task')
CONTROL_TOOLS = WORKSPACE_TOOLS + ('move_camera', 'control_gripper', 'execute_manipulation')
BASE=['list_functions','get_world_state','get_arm_state']
P={
 'robot-tasks':TASK_TOOLS,
 'workspace-tasks':WORKSPACE_TOOLS,
 'workspace-control':CONTROL_TOOLS,
 'perception':BASE+['get_spatial_map','get_spatial_context','get_manipulation_memory','capture_observation','verify_presence','ground_object','segment_object','get_step_context'],
 'view':BASE+['get_observe_view','plan_next_best_view','get_spatial_map','get_spatial_context','check_target_view','get_active_target'],
 'memory':BASE+['capture_observation','resolve_target','refresh_active_target','get_active_target','get_manipulation_memory','check_target_view','require_target_view','get_step_context'],
 'grasp':BASE+['capture_observation','pin_grasp_observation','resolve_target','prepare_grasp_input','calculate_active_grasps','update_object','select_grasp','cache_grasp_plan','get_cached_grasp','prepare_grasp_lift','get_prepared_grasp_lift','validate_prepared_grasp_lift','get_execution_state','get_step_context'],
 'placement':BASE+['capture_observation','pin_grasp_observation','pin_placement_observation','resolve_target','prepare_grasp_input','calculate_active_grasps','update_object','select_grasp','cache_grasp_plan','get_cached_grasp','find_active_placements','plan_next_best_view','prepare_pick_place','get_prepared_pick_place','validate_prepared_pick_place','get_execution_state','get_floor_slots','get_step_context'],
 'diagnostics':BASE+['prepare_held_scene','search_placements','solve_pose','plan_motion','refine_grasp','plan_final_approach','plan_lift','plan_placement','validate_placements','plan_camera_retreat','get_step_context'],
 'queue':BASE+['run_step','submit_steps','run_steps','get_step_context'],
 'all-calculations':['get_workflow_example', 'cache_grasp_plan', 'calculate_active_grasps', 'calculate_grasps', 'capture_observation', 'check_target_view', 'ensure_scene_memory', 'find_active_placements', 'find_placements', 'get_active_target', 'get_arm_state', 'get_cached_grasp', 'get_execution_state', 'get_floor_slots', 'get_manipulation_memory', 'get_observe_view', 'get_prepared_grasp_lift', 'get_prepared_pick_place', 'get_spatial_context', 'get_spatial_map', 'get_step_context', 'get_world_state', 'ground_object', 'list_functions', 'perceive_scene', 'pin_grasp_observation', 'pin_placement_observation', 'plan_camera_retreat', 'plan_final_approach', 'plan_lift', 'plan_motion', 'plan_next_best_view', 'plan_placement', 'prepare_grasp_input', 'prepare_grasp_lift', 'prepare_held_scene', 'prepare_pick_place', 'refine_grasp', 'refresh_active_target', 'require_target_view', 'resolve_target', 'run_step', 'run_steps', 'search_placements', 'segment_object', 'select_grasp', 'solve_pose', 'store_segmented_target', 'submit_steps', 'update_object', 'validate_placements', 'validate_prepared_grasp_lift', 'validate_prepared_pick_place', 'verify_presence'],
}
ROOT=pathlib.Path(__file__).resolve().parent
TASK_WORKSPACE = ROOT / 'robot-tasks-workspace'
FORBIDDEN={'move_to','execute_plan','gripper_open','gripper_close','execute_placement_transfer','execute_placement_descent','release_placed_object','commit_floor_slots','execute_withdrawal','execute_placement_retreat','move_camera','control_gripper','execute_manipulation'}
def apply_profile(name, config):
 if name not in P:raise SystemExit('Choose profile: '+', '.join(P))
 if name in ('workspace-tasks', 'workspace-control'):
  from workspace_discovery import load_snapshot, require_endpoint
  available, workspace_snapshot = load_snapshot(ROOT, controls=name == 'workspace-control')
  require_endpoint(workspace_snapshot, config['tools']['mcpServers']['ros2'].get('url'))
 else:
  available={t['name'] for t in json.loads((ROOT/'results/mcp-tools.json').read_text())}
 names=list(dict.fromkeys(P[name]))
 if (FORBIDDEN.intersection(names) - ({'move_camera', 'control_gripper', 'execute_manipulation'}
                                     if name == 'workspace-control' else set())):raise SystemExit('Physical/state-commit tools are forbidden in calculation profiles')
 if name in ('all-calculations', 'robot-tasks', 'workspace-tasks', 'workspace-control'):
  snapshot=(workspace_snapshot if name in ('workspace-tasks', 'workspace-control') else
            json.loads((ROOT/'results/mcp-status-list_functions.json').read_text())['structuredContent'])
  allowed = set(snapshot['allowlist'])
  matches = (set(names) == allowed - set(TASK_TOOLS) - set(WORKSPACE_TOOLS) if name == 'all-calculations'
             else set(names) <= allowed)
  expected_mode = 'agent_pipeline' if name == 'workspace-control' else 'preparation_only'
  if snapshot.get('mode') != expected_mode or not matches:
   raise SystemExit(f'Refresh discovery: {name} requires the reviewed preparation-only catalog')
 missing=set(names)-available
 if missing:raise SystemExit(f'Refresh MCP discovery; missing tools: {sorted(missing)}')
 config['tools']['mcpServers']['ros2'].update(enabledTools=names,
     toolTimeout=600 if name == 'workspace-control' else 180)
 if name in ('all-calculations', 'robot-tasks', 'workspace-tasks', 'workspace-control'):
  config['tools']['exec']['enable']=False
 if name in ('robot-tasks', 'workspace-tasks', 'workspace-control'):
  config['tools']['mcpServers'] = {'ros2': config['tools']['mcpServers']['ros2']}
  config['agents']['defaults']['workspace'] = str(TASK_WORKSPACE)


def install_task_tool_filter():
    # The installed CLI has no global built-in-tool allowlist. Skip its default
    # loader only for this profile; native MCP registration still runs normally.
    from nanobot.agent.tools.loader import ToolLoader

    original_load = ToolLoader.load

    def load(self, context, registry, *, scope="core"):
        servers = context.config.mcp_servers
        from workspace_arguments import activate
        activate(set(servers) == {"ros2"} and
                 set(servers["ros2"].enabled_tools) in (set(WORKSPACE_TOOLS), set(CONTROL_TOOLS)),
                 controls=set(servers) == {"ros2"} and
                 set(servers["ros2"].enabled_tools) == set(CONTROL_TOOLS))
        if set(servers) == {"ros2"} and set(servers["ros2"].enabled_tools) in (set(TASK_TOOLS), set(WORKSPACE_TOOLS), set(CONTROL_TOOLS)):
            return []
        return original_load(self, context, registry, scope=scope)

    ToolLoader.load = load

def generate(name):
 c=json.loads((pathlib.Path.home()/'.nanobot/config.json').read_text())
 apply_profile(name, c)
 c['agents']['defaults']['maxTokens']=1024
 Config.model_validate(c)
 dest=ROOT/'.private/profiles';dest.mkdir(parents=True,exist_ok=True);dest.chmod(0o700);p=dest/(name+'.json');p.write_text(json.dumps(c,indent=2)+'\n');p.chmod(0o600);return p
if __name__=='__main__':
 if len(sys.argv)!=2:raise SystemExit('Usage: mcp_profiles.py '+ '|'.join(P))
 print(generate(sys.argv[1]))
