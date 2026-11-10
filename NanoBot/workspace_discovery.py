"""Validate dedicated workspace discovery without using legacy diagnostic snapshots."""
import json
from pathlib import Path

WORKSPACE_NAMES = frozenset((
    'inspect_robot', 'get_camera_image', 'check_freshness', 'list_objects',
    'observe_scene', 'plan_camera_view', 'prepare_manipulation',
    'get_task_status', 'cancel_task'))


def validate_snapshot(tools, status, controls=False):
    if (not tools.get('discovery_id') or
            tools.get('discovery_id') != status.get('discovery_id') or
            not str(tools.get('endpoint', '')).startswith('https://') or
            tools.get('endpoint') != status.get('endpoint')):
        raise ValueError('WORKSPACE_DISCOVERY_PAIR_MISMATCH: refresh both snapshots')
    result = status.get('result', {})
    data = result.get('structuredContent') or {}
    expected_mode = 'agent_pipeline' if controls else 'preparation_only'
    if result.get('isError') or data.get('mode') != expected_mode:
        raise ValueError('WORKSPACE_REQUIRES_PREPARATION_ONLY_DISCOVERY')
    source = (data.get('input_provenance') or {}).get('classification')
    if source not in ('mock_replay_publishers', 'live_driver_publishers'):
        raise ValueError('WORKSPACE_DISCOVERY_SOURCE_UNVERIFIED')
    catalog = tools.get('tools', [])
    available = {tool['name'] for tool in catalog}
    if len(available) != len(catalog):
        raise ValueError('WORKSPACE_DISCOVERY_DUPLICATE_TOOLS')
    required = WORKSPACE_NAMES | ({'move_camera', 'control_gripper', 'execute_manipulation'}
                                  if controls else set())
    if not required <= available or not required <= set(data.get('allowlist', [])):
        raise ValueError('WORKSPACE_DISCOVERY_MISSING_TOOLS')
    for tool in catalog:
        if tool['name'] in WORKSPACE_NAMES and tool.get('inputSchema', {}).get('type') != 'object':
            raise ValueError('WORKSPACE_DISCOVERY_INVALID_SCHEMA')
    if controls:
        if source != 'live_driver_publishers':
            raise ValueError('CONTROL_PROFILE_REQUIRES_LIVE_INPUT_DISCOVERY')
        for tool in catalog:
            if tool['name'] in required - WORKSPACE_NAMES and (
                    tool.get('_meta', {}).get('availability') != 'guarded_execution' or
                    tool['name'] in data.get('disabled_tools', [])):
                raise ValueError('CONTROL_PROFILE_REQUIRES_ENABLED_CONTROL_CARDS')
    return available, dict(data, discovery_endpoint=tools['endpoint'])


def load_snapshot(root, controls=False):
    root = Path(root)
    try:
        prefix = 'mcp-control' if controls else 'mcp-workspace'
        tools = json.loads((root / ('results/' + prefix + '-tools.json')).read_text())
        status = json.loads((root / ('results/' + prefix + '-status-list_functions.json')).read_text())
        return validate_snapshot(tools, status, controls)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit('Refresh workspace discovery with tests/discover_workspace_mcp.py: '
                         + str(error)) from error


def require_endpoint(snapshot, configured_url):
    """Do not silently use mock discovery to configure a different live server."""
    if configured_url != snapshot.get('discovery_endpoint'):
        raise SystemExit('WORKSPACE_DISCOVERY_ENDPOINT_MISMATCH: refresh discovery for the '
                         'configured MCP URL, or explicitly select the intended mock endpoint')
