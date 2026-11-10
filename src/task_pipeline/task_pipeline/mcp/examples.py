"""Short model-facing instructions and local, non-executing workflow examples."""

import copy
import uuid

from .contracts import CALCULATIONS, MANIPULATION, READS
from .dependencies import recipe_example
from .preparation import PREPARATION


EXAMPLE_TOOL = 'get_workflow_example'
WORKFLOWS = ('grasp', 'grasp_cache', 'pick_place', 'perceive_scene')


def workflow_example(workflow, object_name, context_id='', destination_name='tray',
                     affordance_type='AUTO'):
    """Return exact existing API calls with fresh IDs; never create state or dispatch."""
    if workflow not in WORKFLOWS:
        raise ValueError('UNKNOWN_WORKFLOW')
    if not object_name.strip():
        raise ValueError('OBJECT_NAME_REQUIRED')
    prefix = 'example_' + uuid.uuid4().hex[:10] + '_'
    context = context_id or 'CONTEXT_ID_FROM_CAPTURE'

    def item(tool, name, arguments, bindings=()):
        return {'tool': tool, 'input': {
            'context_id': context, 'step_id': prefix + name, 'request_id': prefix + name,
            'arguments': arguments,
            'bindings': [{'target_field': target, 'source_step_id': prefix + source,
                          'source_field': field} for target, source, field in bindings]}}

    if workflow == 'perceive_scene':
        steps = [
            item('capture_observation', 'capture', {}),
            item('perceive_scene', 'perceive', {
                'object_names': [object_name], 'active_object_name': object_name,
                'destination_name': destination_name}, [
                    ('image', 'capture', 'image'),
                    ('observation_cloud', 'capture', 'cloud'),
                    ('camera_transform', 'capture', 'camera_transform')]),
        ]
    elif workflow == 'pick_place':
        steps = copy.deepcopy(recipe_example('pick_place', object_name, destination_name))
        for step in steps:
            query = step['input']
            if 'step_id' not in query:  # Direct get_execution_state read has no step envelope.
                continue
            query['context_id'] = context
            query['step_id'] = prefix + query['step_id']
            query['request_id'] = query['step_id']
            for source in query['bindings']:
                source['source_step_id'] = prefix + source['source_step_id']
    else:
        steps = [
            item('capture_observation', 'capture', {}),
            item('pin_grasp_observation', 'pin', {}, [
                ('observation_header', 'capture', 'cloud.header')]),
            item('resolve_target', 'resolve', {'name': object_name, 'role': 'object'}, [
                ('image', 'capture', 'image'), ('observation_cloud', 'capture', 'cloud'),
                ('camera_transform', 'capture', 'camera_transform')]),
            item('calculate_active_grasps', 'grasps', {
                'target_description': object_name, 'affordance_type': affordance_type}),
            item('select_grasp', 'select', {}, [('candidates', 'grasps', 'candidates')]),
        ]
        if workflow == 'grasp_cache':
            steps += [
                item('cache_grasp_plan', 'cache',
                     {'object_name': object_name, 'ttl_seconds': 600}, [
                         ('candidates', 'grasps', 'candidates'),
                         ('observation_header', 'resolve', 'observation_header'),
                         ('observation_cloud', 'resolve', 'observation_cloud'),
                         ('source_mask', 'grasps', 'validated_target_mask'),
                         ('semantic_mask', 'resolve', 'mask')]),
                item('get_cached_grasp', 'cache_check', {'object_name': object_name}),
            ]
    steps[0]['input']['context_id'] = context_id
    return {'workflow': workflow, 'execution_performed': False, 'steps': steps,
            'context_id_source': {
                'step_id': steps[0]['input']['step_id'],
                'response_field': 'context_id',
                'native_mcp_response_path': 'result.structuredContent.context_id',
                'instruction': 'Use that returned string as context_id in every later step; '
                'do not use output.context_id or invent a context ID.'},
            'instructions': [
                'These are proposed calculation calls, NOT executed tasks. Use step.tool and '
                'step.input exactly. Add only the connector namespace required by Nanobot.',
                'If step 1 context_id is empty, capture creates a gateway-owned context. '
                'Replace CONTEXT_ID_FROM_CAPTURE in later inputs with that returned context_id.',
                'Execute in order. Pin immediately after capture; do not insert discovery or '
                'long reasoning between them. Only proceed after success=true.',
                'IDs were generated for this example. Do not change payloads under an existing '
                'request_id. On unknown outcome retrieve that same receipt; never auto-replay.',
                'A pin failure means the exact synchronized source is unavailable; LAYOUT '
                'is an old API name, not a requirement for a special scene layout.',
                'resolve_target may internally ground/segment/store. Do not add invented '
                'grasp_input bindings to calculate_active_grasps.',
                'On failure stop, preserve status and inspect recovery. Grasp success is not '
                'whole-motion readiness. No motion/gripper task is included.',
            ]}


def compact_description(name, original, contract, execution_enabled):
    """Keep essential guidance in model-visible text; full reference stays in metadata."""
    if name == 'list_functions':
        return ('Read mode/allowlist. Use summary_only=true for normal checks; false returns '
                'a LARGE ROS reference catalog, NOT a callable MCP list. allowlist names enabled '
                'tools; mcp_tools mirrors native tools/list; disabled_tools are blocked. '
                'Prefer get_workflow_example for exact task calls. '
                'Read-only, no prerequisites or ROS action. Prerequisites: none. '
                'Execution sequence: list_functions -> inspect mode. '
                'Never infer motion readiness.')
    if name == EXAMPLE_TOOL:
        return (
            'Get an example workflow / sample pipeline with exact copyable tool calls. '
            'Call this when asked for an example, tool order or bindings. READ-ONLY HELP. '
            'workflow=grasp, grasp_cache, pick_place or perceive_scene; '
            'object_name is the exact target name. Returns exact ordered tool inputs with fresh '
            'step/request IDs. For a new task initial capture uses context_id=""; copy its '
            'result.structuredContent.context_id into ALL later calls. It executes nothing. '
            'Copy inputs; after capture, substitute its '
            'returned context_id. Prefer this to guessing a pipeline. No robot state required. '
            'Prerequisites: none. Execution sequence: get_workflow_example -> '
            'follow returned steps.')
    purpose = PREPARATION.get(name) or READS.get(name) or CALCULATIONS.get(name)
    if name in MANIPULATION:
        purpose = MANIPULATION[name] + (
            ' Guarded physical dispatch; exact phase/ref/contact checks apply.'
            if execution_enabled else ' DISABLED: discovery only; no ROS dispatch. Do not bypass.')
    if purpose is None:
        purpose = original
    prior = contract['prerequisites']['prior_calls']
    required = [p['tool'] for p in prior if p['requirement'] == 'required']
    parts = [purpose, 'Prerequisites: ' +
             (', '.join(required) or 'see state/alternatives in metadata')]
    route = contract['execution_sequence']
    parts.append('Execution sequence: ' + ' -> '.join(route[-4:]) +
                 (' (tail; full route in metadata)' if len(route) > 4 else ''))
    if name in PREPARATION:
        parts.append('Use the SAME returned context_id and successful producer step IDs. '
                     'Final result; stop on failure. Unknown outcomes must not be replayed.')
    bindings = contract['prerequisites']['bindings']
    if bindings:
        parts.append('Bind: ' + '; '.join(
            b['target_field'] + ' <- ' + b['source_step_id'] + '.' + b['source_field']
            for b in bindings[:4]) + (
                '; remaining bindings in metadata' if len(bindings) > 4 else ''))
    if name in ('capture_observation', 'pin_grasp_observation', 'resolve_target',
                'calculate_active_grasps', 'select_grasp'):
        parts.append('For exact copyable inputs call get_workflow_example(workflow="grasp", '
                     'object_name="cup").')
    if name == 'pin_grasp_observation':
        parts.append('EXACT binding: target_field="observation_header", '
                     'source_step_id=YOUR_CAPTURE_ID, source_field="cloud.header". '
                     'Never bind image. Capture attempts retention automatically; this confirms '
                     'the exact source. Call immediately after capture. LAYOUT_* is a legacy '
                     'status name, NOT a special scene-layout requirement.')
    return '\n'.join(parts)
