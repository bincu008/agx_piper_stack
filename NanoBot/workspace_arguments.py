"""Hide transport bookkeeping only in the explicit workspace-tasks profile."""
from copy import deepcopy
import json

_ACTIVE = False
_CONTROL_ACTIVE = False
SUBMISSIONS = {'observe_scene', 'plan_camera_view', 'prepare_manipulation',
               'move_camera', 'control_gripper', 'execute_manipulation'}


MODEL_DESCRIPTIONS = {
    'inspect_robot': 'Read current arm feedback, camera direction and workspace ages. '
    'Reading state does not authorize movement.',
    'get_camera_image': 'See the current wrist-camera image with source/time metadata. '
    'Pixels are attached to the vision input. A blocked read provides no new image.',
    'list_objects': 'Read remembered object/destination names and locations. '
    'Memory is not current visual evidence; observe_scene refreshes it.',
    'check_freshness': 'Read fresh=true/false for the image or a named object. '
    'No inference or motion; currently uses existing shorter verification limits too.',
    'observe_scene': 'Perceive named targets using fresh synchronized capture. '
    'Use targets and detail; follow RUNNING with get_task_status(). UNKNOWN is not absent.',
    'plan_camera_view': 'Calculate only. relative requires frame=camera_optical and '
    'translation_m=[right,down,forward], rotation_rad=[roll,pitch,yaw]. target requires '
    'exactly one target_name or target_cell_id. Do not mix modes. '
    'NO_REACHABLE_CHANGE means no changed view was found; never claim movement.',
    'prepare_manipulation': 'Prepare an object task automatically. grasp previews/cache only; '
    'grasp_lift validates grasp and lift; pick_place also requires destination. '
    'Follow RUNNING with get_task_status(). Report checks and bypasses. No execution.',
    'get_task_status': 'Read the current server task. No identifiers needed. '
    'Preserve UNKNOWN and the actual blocker; never replay work automatically.',
    'cancel_task': 'Request cancellation of the current calculation between steps. '
    'CANCEL_PENDING is not confirmed cancellation or an emergency stop.'}


MODEL_DESCRIPTIONS.update({
    'move_camera': 'Execute a checked empty-hand camera move. Use relative optical metres/radians '
    'or one target. NO_REACHABLE_CHANGE means no movement. No held-object jogging.',
    'control_gripper': 'Prepared open/close/release uses the last reviewed successful plan. '
    'Empty-hand mode supports open only with width_m and force_n. No free close or held release.',
    'execute_manipulation': 'Execute the last reviewed successful prepared manipulation. '
    'Use stage=all or next. No identifiers needed; private receipts bind the exact plan. '
    'UNKNOWN requires local reconciliation, never retry with a new request.'})

def activate(value, controls=False):
    global _ACTIVE, _CONTROL_ACTIVE
    _ACTIVE = bool(value)
    _CONTROL_ACTIVE = bool(value and controls)


def model_schema(name, schema):
    if not _ACTIVE:
        return schema
    result = deepcopy(schema)
    hidden = {'request_id'} if name in SUBMISSIONS else {'task_id'} if name in {
        'get_task_status', 'cancel_task'} else set()
    if _CONTROL_ACTIVE and name in ('execute_manipulation', 'control_gripper'):
        hidden |= {'plan_id', 'revision'}
    for field in hidden:
        result.get('properties', {}).pop(field, None)
        if field in result.get('required', []):
            result['required'].remove(field)
    return result



def model_result(value):
    """Keep IDs in receipts, not in ordinary model-facing JSON."""
    hidden = {'request_id', 'task_id', 'context_id', 'job_id', 'plan_id', 'execution_ref',
              'step_id', 'source_step_id', 'current_step_id', 'failed_step_id',
              'plan_revision', 'revision'}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in hidden:
                continue
            if key == 'text' and isinstance(item, str):
                try:
                    item = json.dumps(model_result(json.loads(item)))
                except ValueError:
                    pass
            result[key] = model_result(item)
        return result
    if isinstance(value, list):
        return [model_result(item) for item in value]
    return value


def install():
    from nanobot.agent.tools.mcp import MCPToolWrapper
    from nanobot.agent.tools.base import ToolResult
    original_description = MCPToolWrapper.description.fget


    def description(self):
        if _ACTIVE and self._server_name == 'ros2' and self._original_name in MODEL_DESCRIPTIONS:
            return MODEL_DESCRIPTIONS[self._original_name]
        return original_description(self)

    original = MCPToolWrapper.parameters.fget
    original_execute = MCPToolWrapper.execute

    def parameters(self):
        schema = original(self)
        return model_schema(self._original_name, schema) if self._server_name == 'ros2' else schema

    async def execute(self, **kwargs):
        result = await original_execute(self, **kwargs)
        if not _ACTIVE or self._server_name != 'ros2':
            return result
        try:
            compact = ToolResult(json.dumps(model_result(json.loads(result))),
                                 is_error=getattr(result, 'is_error', False))
            compact.workspace_receipt = str(result)
            return compact
        except (ValueError, TypeError):
            return result

    MCPToolWrapper.description = property(description)
    MCPToolWrapper.parameters = property(parameters)
    MCPToolWrapper.execute = execute
