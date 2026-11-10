"""Discoverable API dependencies and recovery advice, never execution authorization."""

import copy
import hashlib

from rclpy.serialization import deserialize_message

from task_pipeline.recipes import prepare_grasp_lift, prepare_pick_place
from task_pipeline.registry import FUNCTIONS

from .contracts import CALCULATIONS, MANIPULATION, READS
from .preparation import PREPARATION


RESOURCE = 'ros-pipeline://tool-dependency-graph'
COMMON_STATE = [
    'Use the actual server allowlist; a dependency description does not enable a blocked tool.',
    'Preparation requires known IDLE empty hand, no unknown execution/preparation barrier, '
    'and the existing mode/provenance checks. preparation_only requires physical gate=false.',
    'Bindings use successful predecessor results in the SAME gateway-owned context; '
    'preserve image/cloud/mask timestamps, source revision and manual FK*X frame association.',
]

# A dependency is dataflow, state, recommended workflow, an alternative, or an internal call.
# These distinctions must not be flattened into mandatory caller-side invocations.
PRIOR = {'capture_observation': [],
         'verify_presence': [('capture_observation',
                              'required',
                              'Bind its image; inspect success/present/status.')],
         'ground_object': [('capture_observation', 'required', 'Bind the captured image.'),
                           ('verify_presence',
                            'recommended',
                            'PRESENT before grounding in the normal recipe; no presence-result '
                            'binding.')],
         'segment_object': [('capture_observation', 'required', 'Bind the original image.'),
                            ('ground_object',
                             'required',
                             'Bind pixel_box to bbox on that same image.')],
         'store_segmented_target': [('capture_observation',
                                     'required',
                                     'Bind image, cloud and camera_transform together.'),
                                    ('segment_object',
                                     'required',
                                     'Bind mask; set name and role=object or destination.')],
         'resolve_target': [('get_manipulation_memory', 'optional',
                             'Inspect reusable target memory.'
                             ),
                            ('capture_observation',
                             'conditional',

                             'Bind image/cloud/transform together, or omit ALL three for an int'
                             'ernal '

                             'capture.')],
         'refresh_active_target': [('resolve_target',
                                    'alternative',
                                    'Existing target may be refreshed; supply its name/role.'),
                                   ('capture_observation',
                                    'conditional',
                                    'Bind a complete fresh observation, or let the API capture.')],
         'get_active_target': [('store_segmented_target',
                                'alternative',
                                'Target must already be stored.'),
                               ('resolve_target',
                                'alternative',
                                'A successful resolve also stores target geometry.')],
         'perceive_scene': [('capture_observation',
                             'conditional',
                             'Complete synchronized inputs or the API internal capture.')],
         'ensure_scene_memory': [('get_manipulation_memory',
                                  'optional',
                                  'Memory reuse is not fresh visual proof.'),
                                 ('capture_observation',
                                  'conditional',
                                  'Complete supplied observation or internal capture.')],
         'check_target_view': [('resolve_target',
                                'alternative',
                                'Requires remembered target position; store_segmented_target also '
                                'supplies it.'),
                               ('capture_observation',
                                'conditional',
                                'Bind scene cloud and camera geometry together when supplied.')],
         'require_target_view': [('check_target_view',
                                  'recommended',
                                  'Inspect visibility; LOOK_REQUIRED cannot be repaired by a '
                                  'calculation-only move.')],
         'calculate_active_grasps': [('get_active_target',
                                      'state',

                                      'Fresh active object image/mask/cloud must exi'
                                      'st; getter alone '

                                      'does not populate it.')],
         'find_active_placements': [('get_active_target',
                                     'state',
                                     'Fresh active object and destination geometry must exist.'),
                                    ('cache_grasp_plan',
                                     'required',
                                     'Retain matching object grasp/source geometry.')],
         'prepare_grasp_lift': [('cache_grasp_plan',
                                 'required',
                                 'A matching valid source-associated grasp pool must already be '
                                 'cached.')],
         'prepare_pick_place': [('cache_grasp_plan',
                                 'required',

                                 'A matching source-associated grasp pool must already be cached.'
                                 ),
                                ('find_placements',
                                 'required',
                                 'Bind placement candidates and source center.'),
                                ('segment_object',
                                 'required',

                                 'Bind destination mask from the destination se'
                                 'gmentation, not the '

                                 'source mask.')],
         'get_cached_grasp': [('cache_grasp_plan',
                               'required',
                               'Matching object cache must exist, be unexpired and '
                               'source-associated.')],
         'get_prepared_grasp_lift': [('prepare_grasp_lift',
                                      'required',

                                      'Successful complete, unexpired grasp/lift rec'
                                      'ord for the exact '

                                      'object_name.')],
         'validate_prepared_grasp_lift': [('prepare_grasp_lift',
                                           'required',

                                           'Successful complete cached record; no actuati'
                                           'on or cache '

                                           'consumption.')],
         'get_prepared_pick_place': [('prepare_pick_place',
                                      'required',
                                      'Successful complete, unexpired pick/place record for this '
                                      'object.')],
         'validate_prepared_pick_place': [('prepare_pick_place',
                                           'required',

                                           'Successful complete cached record, matching s'
                                           'ource and '

                                           'start state.')],
         'get_execution_state': [('prepare_pick_place',
                                  'conditional',

                                  'A prepared record is needed to report actiona'
                                  'ble waiting physical'
                                  ' '

                                  'stages; the read cannot create it.')],
         'get_step_context': [('capture_observation',
                               'alternative',

                               'Use the context_id returned by any gateway-owned preparation call'
                               '.'
                               ),
                              ('submit_steps', 'alternative',
                               'Queued context is also gateway-owned.'
                               )],
         'get_floor_slots': [('find_placements',
                              'conditional',
                              'A prior search populates grid data; empty state is not a search.')],
         'get_observe_view': [('capture_observation',
                               'internal',

                               'Uses existing synchronized geometry; may capture internally. This'
                               ' is '

                               'not movement.')],
         'plan_next_best_view': [('resolve_target',
                                  'alternative',
                                  'Name target uses remembered position.'),
                                 ('get_spatial_map',
                                  'alternative',
                                  'Cell target uses a returned target_cell_id.'),
                                 ('capture_observation',
                                  'alternative',
                                  'Explicit target/header may instead be bound from captured '
                                  'geometry.')],
         'prepare_held_scene': [('capture_observation', 'required',
                                 'Bind organized cloud/header.'
                                 ),
                                ('calculate_grasps',
                                 'required',

                                 'Bind validated_target_mask; semantic mask com'
                                 'es from segmentation'
                                 '.'
                                 )],
         'search_placements': [('prepare_held_scene',
                                'required',
                                'Bind frozen source, collision and destination support clouds.')],
         'refine_grasp': [('capture_observation',
                           'required',
                           'Previous registered header must remain valid; close-up refinement may '
                           'require a fresh view.'),
                          ('select_grasp',
                           'required',

                           'Bind candidate. Preparation-only cannot move to acquire a missing'
                           ' close-up '

                           'view.')],
         'plan_final_approach': [('select_grasp',
                                  'required',
                                  'Validated pregrasp/final joints and poses.'),
                                 ('refine_grasp',
                                  'required',
                                  'Refined pose, joints, center, closing axis and width.'),
                                 ('prepare_held_scene',
                                  'required',
                                  'Bind immutable source/environment clouds.')],
         'plan_lift': [('select_grasp',
                        'conditional',
                        'Held lift uses this grasp; empty_hand withdrawal instead uses explicit '
                        'start_joints/jaw_opening_m.'),
                       ('prepare_held_scene',
                        'conditional',
                        'Bound frozen source/environment geometry for standalone held planning.')],
         'plan_placement': [('select_grasp', 'required', 'Bind grasp_candidate.'),
                            ('find_placements',
                             'alternative',
                             'Bind source_object_center and candidates.'),
                            ('search_placements',
                             'alternative',
                             'Standalone search with matching prepared geometry can supply '
                             'candidates.')],
         'validate_placements': [('plan_placement',
                                  'alternative',
                                  'Uses the same PlanPlacement request, not its response as a '
                                  'request.'),
                                 ('find_placements',
                                  'required',
                                  'Supply original candidates plus selected grasp and matching '
                                  'source.')],
         'plan_motion': [('get_arm_state',
                          'required',

                          'Use current joints and an explicit remaining_path_flat; computes '
                          'a path '

                          'only.')],
         'solve_pose': [('get_arm_state',
                         'required',

                         'Current joint seed and requested look-at/TCP pose; ignore_collisi'
                         'on=true is '

                         'forbidden.')],
         'plan_camera_retreat': [('prepare_held_scene',
                                  'required',
                                  'Frozen collision_scene plus explicit release/start joints and '
                                  'candidate_tcp_poses.')],
         'run_step': [],
         'submit_steps': [],
         'run_steps': [('submit_steps',
                        'required',

                        'Start with exact returned context_id/revision; poll status using '
                        'the SAME job '

                        'request_id.')]}

MEMORY_SEQUENCE = ['capture_observation', 'verify_presence', 'ground_object',
                   'segment_object', 'store_segmented_target']
ACTIVE_SEQUENCE = MEMORY_SEQUENCE + ['get_active_target']
SEQUENCES = {
    'resolve_target': ['resolve_target'],
    'refresh_active_target': ['refresh_active_target'],
    'get_active_target': ACTIVE_SEQUENCE,
    'perceive_scene': ['capture_observation', 'perceive_scene'],
    'ensure_scene_memory': ['capture_observation', 'ensure_scene_memory'],
    'check_target_view': ACTIVE_SEQUENCE + ['check_target_view'],
    'require_target_view': ACTIVE_SEQUENCE + ['check_target_view', 'require_target_view'],
    'calculate_active_grasps': ACTIVE_SEQUENCE + ['calculate_active_grasps'],
    'get_cached_grasp': [],  # Filled from the production grasp preparation recipe below.
    'get_prepared_grasp_lift': [],
    'validate_prepared_grasp_lift': [],
    'get_prepared_pick_place': [],
    'validate_prepared_pick_place': [],
    'run_step': ['run_step'], 'submit_steps': ['submit_steps'],
    'run_steps': ['submit_steps', 'run_steps'],
}


def binding(target, source_tool, source_step, field, requirement='required'):
    """Describe a same-context binding using actual request/response field paths."""
    return {'target_field': target, 'source_tool': source_tool, 'source_step_id': source_step,
            'source_field': field, 'requirement': requirement}


def observation_bindings(cloud_field='observation_cloud'):
    """Keep the synchronized triplet together; none may come from another capture."""
    return [binding('image', 'capture_observation', 'observe', 'image', 'all_or_none'),
            binding(cloud_field, 'capture_observation', 'observe', 'cloud', 'all_or_none'),
            binding('camera_transform', 'capture_observation', 'observe',
                    'camera_transform', 'all_or_none')]


EXTRA_BINDINGS = {
    'store_segmented_target': observation_bindings() + [
        binding('mask', 'segment_object', 'segment_cup', 'mask')],
    'resolve_target': observation_bindings(),
    'refresh_active_target': observation_bindings(),
    'perceive_scene': observation_bindings(), 'ensure_scene_memory': observation_bindings(),
    'check_target_view': [binding('observation_cloud', 'capture_observation', 'observe', 'cloud'),
                          binding(
        'camera_transform', 'capture_observation', 'observe', 'camera_transform')],
    'prepare_held_scene': [binding(
        'observation_header', 'capture_observation', 'observe', 'cloud.header'),
        binding('cloud', 'capture_observation', 'observe', 'cloud'),
        binding(
        'source_mask',
        'calculate_grasps',
        'grasps',
        'validated_target_mask'),
        binding('semantic_mask', 'segment_object', 'segment_cup', 'mask')],
    'search_placements': [binding(
        'object_cloud', 'prepare_held_scene', 'held_scene', 'source_object_cloud'),
        binding(
        'scene_cloud',
        'prepare_held_scene',
        'held_scene',
        'collision_scene'),
        binding(
        'destination_cloud',
        'prepare_held_scene',
        'held_scene',
        'destination_support_cloud'),
        binding('observation_header', 'capture_observation', 'observe', 'cloud.header')],
    'refine_grasp': [binding(
        'previous_observation_header', 'capture_observation', 'observe', 'cloud.header'),
        binding('candidate', 'select_grasp', 'select_grasp', 'candidate')],
    # Read-only get_arm_state returns a snapshot, not a StepStore handle. Copy its
    # primitive joints into arguments.current_joints; never bind a nonexistent step.
    'plan_motion': [], 'solve_pose': [],
    'plan_camera_retreat': [binding(
        'collision_scene', 'prepare_held_scene', 'held_scene', 'collision_scene')],
}
EXTRA_BINDINGS['require_target_view'] = copy.deepcopy(EXTRA_BINDINGS['check_target_view'])


EXTRA_BINDINGS['plan_final_approach'] = [
    binding('current_pregrasp_pose', 'select_grasp', 'select_grasp', 'candidate.pregrasp_pose'),
    binding('current_joints', 'select_grasp', 'select_grasp', 'candidate.pregrasp_joints'),
    binding('refined_grasp_pose', 'refine_grasp', 'refine', 'grasp_pose'),
    binding('final_grasp_joints', 'refine_grasp', 'refine', 'grasp_joints'),
    binding('refined_contact_center', 'refine_grasp', 'refine', 'corrected_seed_base'),
    binding('closing_axis', 'refine_grasp', 'refine', 'closing_axis_base'),
    binding('grasp_width_m', 'refine_grasp', 'refine', 'grasp_width_m'),
    binding('target_observation_header', 'capture_observation', 'observe', 'cloud.header'),
    binding('source_object_cloud', 'prepare_held_scene', 'held_scene', 'source_object_cloud'),
    binding('collision_scene', 'prepare_held_scene', 'held_scene', 'collision_scene'),
]
for _name in ('plan_placement', 'validate_placements', 'plan_lift'):
    EXTRA_BINDINGS[_name] = [
        binding('grasp_candidate', 'select_grasp', 'select_grasp', 'candidate'),
        binding('source_object_center', 'prepare_held_scene',
                'held_scene', 'source_object_center'),
        binding('source_object_cloud', 'prepare_held_scene', 'held_scene', 'source_object_cloud'),
        binding('placement_scene', 'prepare_held_scene', 'held_scene', 'collision_scene'),
        binding('destination_support_cloud', 'prepare_held_scene',
                'held_scene', 'destination_support_cloud'),
    ]
    if _name != 'plan_lift':
        EXTRA_BINDINGS[_name].append(
            binding(
                'candidates',
                'find_placements',
                'placements',
                'candidates'))


def _primitive(value):
    """Use JSON primitives for example literal arguments, never caller binary buffers."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return [_primitive(item) for item in value]


def _recipe_steps(kind, object_name='brown cup', destination='dark rectangular tray'):
    """Take names, argument defaults and binding paths from existing production recipes."""
    recipe = (prepare_grasp_lift(object_name, .05) if kind == 'grasp_lift' else
              prepare_pick_place(object_name, destination, .05))
    return recipe


def recipe_example(kind='pick_place', object_name='brown cup',
                   destination='dark rectangular tray'):
    """Generate a reusable typed example, preserving the actual public recipe order."""
    example = []
    for index, item in enumerate(_recipe_steps(kind, object_name, destination)):
        request = deserialize_message(bytes(item.request_cdr),
                                      FUNCTIONS[item.function_name].service_type.Request)
        arguments = {key: _primitive(getattr(request, key)) for key, value in
                     request.get_fields_and_field_types().items() if '/' not in value}
        if item.function_name not in PREPARATION:
            example.append({'tool': item.function_name, 'input': arguments})
            continue
        example.append({'tool': item.function_name, 'input': {
            'context_id': '' if not index else 'RETURNED_CONTEXT_ID',
            'step_id': item.step_id, 'request_id': 'example_' + kind + '_' + item.step_id,
            'arguments': arguments,
            'bindings': [{'target_field': b.target_field, 'source_step_id': b.source_step_id,
                          'source_field': b.source_field} for b in item.bindings]}})
    return example


def _caller_sequence(name, seen=()):
    """Expand required data producers into a complete recommended caller route."""
    if name in seen:
        return [name]
    if name in SEQUENCES and SEQUENCES[name]:
        return list(SEQUENCES[name])
    for kind in ('grasp_lift', 'pick_place'):
        recipe = _recipe_steps(kind)
        for i, item in enumerate(recipe):
            if item.function_name == name:
                return [step.function_name for step in recipe[:i + 1]]
    result = []
    for tool, requirement, _ in PRIOR.get(name, []):
        if requirement == 'required':
            for producer in _caller_sequence(tool, (*seen, name)):
                if producer not in result:
                    result.append(producer)
    return result + [name]


def dependency_contract(name):
    """Return dependency guidance; it never relaxes or replaces backend enforcement."""
    if name not in {*READS, *CALCULATIONS, *MANIPULATION, *PREPARATION,
                    'list_functions', 'get_workflow_example', 'run_step',
                    'submit_steps', 'run_steps'}:
        raise ValueError('UNKNOWN_TOOL_DEPENDENCY: ' + name)
    calls = [{'tool': tool, 'requirement': kind, 'reason': reason}
             for tool, kind, reason in PRIOR.get(name, [])]
    inputs = copy.deepcopy(EXTRA_BINDINGS.get(name, []))
    example = None
    chain = SEQUENCES.get(name)
    kind = 'grasp_lift' if 'grasp_lift' in name else 'pick_place'
    recipe = _recipe_steps(kind)
    for index, item in enumerate(recipe):
        if item.function_name == name:
            producers = {p.step_id: p.function_name for p in recipe[:index]}
            inputs = [binding(b.target_field, producers[b.source_step_id], b.source_step_id,
                              b.source_field) for b in item.bindings]
            for producer in dict.fromkeys(b['source_tool'] for b in inputs):
                if not any(call['tool'] == producer for call in calls):
                    calls.append({'tool': producer, 'requirement': 'required',
                                  'reason': 'Supplies typed predecessor data.'})
            chain = [step.function_name for step in recipe[:index + 1]]
            example = recipe_example(kind)[index]
            break
    if name in ('pin_grasp_observation', 'pin_placement_observation'):
        chain = ['capture_observation', name]
    for cached, producer, recipe_name in (
            ('get_cached_grasp', 'cache_grasp_plan', 'grasp_lift'),
            ('get_prepared_grasp_lift', 'prepare_grasp_lift', 'grasp_lift'),
            ('validate_prepared_grasp_lift', 'prepare_grasp_lift', 'grasp_lift'),
            ('get_prepared_pick_place', 'prepare_pick_place', 'pick_place'),
            ('validate_prepared_pick_place', 'prepare_pick_place', 'pick_place')):
        if name == cached:
            base = _recipe_steps(recipe_name)
            stop = next(i for i, step in enumerate(base) if step.function_name == producer)
            chain = [step.function_name for step in base[:stop + 1]] + [name]
            example = {'tool': name, 'input': {
                'context_id': 'RETURNED_CONTEXT_ID', 'step_id': name,
                'request_id': 'example_' + name, 'arguments': {'object_name': 'brown cup'},
                'bindings': []}}
    if chain is None:
        chain = _caller_sequence(name)
    state = list(COMMON_STATE) if name in PREPARATION else [

        'An available ROS backend is required for ROS snapshots; reads do '
        'not populate missing state.'
    ]
    if name in MANIPULATION:
        calls = [{'tool': 'prepare_pick_place', 'requirement': 'required',
                  'reason': 'Complete prepared pick/place is required for payload stages.'},
                 {'tool': 'get_execution_state', 'requirement': 'required',
                  'reason': 'Use its exact execution_ref/revision and first waiting stage.'}]
        if name == 'move_to':
            calls = [{'tool': 'get_observe_view', 'requirement': 'recommended',
                      'reason':
                          'Safe endpoint target; move_to independently plans the full path.'
                      }]
        chain = [item['tool'] for item in calls] + [name]
        state = ['Physical dispatch must be enabled by the server AND master; never use a '
                 'calculation to bypass a disabled mode.',
                 'Require matching execution_ref, source/revision, ordered phase and verified '
                 'contact before placement; unknown state forbids continuation.']
        previous = {
            'gripper_open': ('execute_plan', 'pregrasp completed (AT_PREGRASP).'),
            'gripper_close': (
                'execute_plan', 'approach completed (AT_GRASP); verify fresh contact.'),
            'execute_placement_transfer': (
                'execute_plan', 'lift completed for the current held object.'),
            'execute_placement_descent': ('execute_placement_transfer', 'Transfer completed.'),
            'release_placed_object': (
                'execute_placement_descent', 'Descent completed at the placement.'),
            'commit_floor_slots': (
                'release_placed_object', 'Confirmed release; exact frozen selected_pose.'),
            'execute_withdrawal': (
                'release_placed_object', 'Confirmed release; slot commit if required by policy.'),
            'execute_placement_retreat': (
                'execute_withdrawal', 'Withdrawal completed before retreat.'),
        }.get(name)
        if previous:
            calls.append({'tool': previous[0], 'requirement': 'required', 'reason': previous[1]})
            chain = [item['tool'] for item in calls] + [name]
        if name == 'execute_plan':
            state.append('Segment prerequisites: look/pregrasp follow waiting_stages; approach '
                         'requires gripper_open; lift requires gripper_close.contact_confirmed; '
                         'transfer requires completed lift; descent requires completed transfer.')
        if name == 'move_to':
            state = ['Requires enabled physical mode/master gate, fresh idle empty-hand state '
                     'and no active/unknown session. Use a durable request_id. move_to does '
                     'NOT accept execution_ref and independently plans the full path.']
    if name in ('list_functions', 'get_workflow_example'):
        state = ['No ROS call or prior tool is required. This is local authenticated discovery.']
    if name in ('run_step', 'submit_steps', 'run_steps'):
        state = list(COMMON_STATE) + [
            'Only calculation functions from the actual enum are allowed in generic queues.',
            'Use exact revision returned by submit_steps for run_steps start; poll with the '
            'same job request_id/context_id. Acknowledgement is not final completion.']
    if name in ('plan_motion', 'solve_pose'):
        state.append('get_arm_state is a direct read, not a stored step: copy its primitive '

                     'joints into arguments.current_joints; there is no get_arm_state b'
                     'inding handle.'
                     )
    if name == 'verify_presence':
        state.append('Require captured success=true with synchronized image/cloud. PRESENT '

                     'requires presence success=true AND present=true; failure/timeout '
                     'is UNKNOWN.'
                     )
    if name in ('get_cached_grasp', 'get_prepared_grasp_lift', 'validate_prepared_grasp_lift',
                'get_prepared_pick_place', 'validate_prepared_pick_place'):
        state.append('Prior producer must have returned success=true; match exact object_name, '
                     'source revision and cache expiry. An expired/invalidated record must be '
                     'reprepared; its absence does not prove the producer was never called.')
    if example is None and name in PREPARATION:
        arguments = {}
        for key, value in (('name', 'brown cup'), ('role', 'object'), ('object_name',
                           'brown cup'), ('destination_name', 'dark rectangular tray')):
            if key in FUNCTIONS[name].service_type.Request.get_fields_and_field_types():
                arguments[key] = value
        example = {'tool': name, 'input': {
            'context_id': 'RETURNED_CONTEXT_ID' if inputs else '', 'step_id': name,
            'request_id': 'example_' + name, 'arguments': arguments,
            'bindings': [{k: v for k, v in item.items() if k in (
                'target_field', 'source_step_id', 'source_field')} for item in inputs]}}
    if name == 'calculate_grasps':
        calls.append({'tool': 'pin_grasp_observation', 'requirement': 'conditional',
                      'reason': 'Pin the exact captured header before VLM/agent delays so '
                      'rolling-cache eviction cannot remove the source.'})
        state.append('All seed fields, mask and observation_header must refer to the SAME '
                     'capture. prepare_grasp_input does not preserve a rolling VAMP cloud. '
                     'Never bind a newer resolve header to older prepared seeds/mask.')
    if name == 'calculate_active_grasps':
        state.append('Compound active route internally pins the stored observation and calls '
                     'prepare_grasp_input then calculate_grasps. It accepts no grasp_input '
                     'message. Bindings may be empty; set target_description for the object.')
        calls += [{'tool': tool, 'requirement': 'internal', 'reason': 'Performed internally.'}
                  for tool in ('pin_grasp_observation', 'prepare_grasp_input', 'calculate_grasps')]
        example['input']['arguments']['target_description'] = 'brown cup'
    if name == 'resolve_target':
        state.append('Compound operation: can internally capture -> verify_presence -> '
                     'ground_object -> segment_object -> store. Prior single-tool perception '
                     'calls are NOT mandatory; do not bind only image without cloud/transform.')
    return {'prerequisites': {'prior_calls': calls, 'bindings': inputs, 'context_state': state},
            'execution_sequence': chain, 'example': example,
            'example_semantics': 'Illustrative request and predecessor IDs, not a dispatch grant. '
            'Production-recipe examples are in graph.examples; standalone solvers also require '
            'task-specific literal targets/seeds. Replace RETURNED_CONTEXT_ID and source IDs.',
            'sequence_semantics': 'Recommended complete caller route, not a list of universally '
            'mandatory backend gates. Conditional/alternative/internal edges are labelled.',
            'valid_binding_targets': (dict(FUNCTIONS[name].service_type.Request.
                                           get_fields_and_field_types())
                                      if name in FUNCTIONS else {}),
            'response_fields': dict(FUNCTIONS[name].service_type.Response.
                                    get_fields_and_field_types()) if name in FUNCTIONS else {},
            'binding_semantics': 'Example step IDs are arbitrary owner-chosen IDs. Replace them '
            'with your successful predecessor IDs. source_field is the ROS response path '
            '(image, NOT output.image). Reuse returned context_id; never paste a binary result.',
            'resource': RESOURCE}


def dependency_graph(names):
    """Expose every tool and labelled dependency, including blocked/read-only tools."""
    nodes = [{'name': name, **dependency_contract(name)} for name in sorted(names)]
    edges = [{'from': prior['tool'], 'to': node['name'], 'kind': prior['requirement'],
              'reason': prior['reason']} for node in nodes
             for prior in node['prerequisites']['prior_calls']]
    return {'schema_version': 1, 'documentation_only': True, 'nodes': nodes, 'edges': edges,
            'examples': {'grasp_lift': recipe_example('grasp_lift'),
                         'pick_place': recipe_example('pick_place')},
            'instructions': 'Read actual availability from list_functions; graph edges never '
            'enable a tool, synthesize state, execute a task or bypass uncertainty.'}


def cache_recovery_steps(arguments):
    """Offer a same-context explicit route; never dispatch it or assume stale results valid."""
    arguments = arguments or {}
    object_name = arguments.get('arguments', {}).get('object_name') or 'EXACT_OBJECT_NAME'
    context = arguments.get('context_id') or 'RETURNED_CONTEXT_ID'
    identity = str(context) + ':' + str(arguments.get('request_id', 'cache_recovery'))
    prefix = 'repair_' + hashlib.sha256(identity.encode()).hexdigest()[:10] + '_'
    result = []
    for item in recipe_example('grasp_lift', object_name):
        if item['tool'] not in PREPARATION:
            continue
        item = copy.deepcopy(item)
        query = item['input']
        query['context_id'] = context
        query['step_id'] = prefix + query['step_id']
        query['request_id'] = query['step_id']
        for source in query['bindings']:
            source['source_step_id'] = prefix + source['source_step_id']
        result.append(item)
        if item['tool'] == 'cache_grasp_plan':
            break
    result.append({'tool': 'get_cached_grasp', 'input': {
        'context_id': context, 'step_id': prefix + 'get_cached_grasp',
        'request_id': prefix + 'get_cached_grasp',
        'arguments': {'object_name': object_name}, 'bindings': []}})
    return result


def recovery_hint(name, result, arguments=None):
    """Explain an existing failure without changing its status, result or retry semantics."""
    raw = result.get('response') or result.get('output') or {}
    if not isinstance(raw, dict):
        raw = {}
    status = str(result.get('status') or raw.get('status') or raw.get(
        'failure_reason') or result.get('detail') or '')
    failed = (result.get('success') is False or raw.get('success') is False or
              result.get('transport_outcome') in (
                  'unknown', 'unavailable', 'disabled', 'rejected') or
              'STATE_UNKNOWN' in status)
    if not failed:
        return None
    target = name
    if name == 'run_step':
        target = (arguments or {}).get('function_name', name)
    missing = []
    if 'MCP_CONTEXT_OWNER_REQUIRED' in status:
        target = 'capture_observation'
        missing = ['A gateway-owned context is required. Call capture_observation with '
                   'context_id=""; copy result.structuredContent.context_id into later calls.']
    elif 'NO_PREPARED_GRASP_LIFT' in status:
        target, missing = 'validate_prepared_grasp_lift', ['prepare_grasp_lift']
    elif 'NO_PREPARED_PICK_PLACE' in status:
        target, missing = 'validate_prepared_pick_place', ['prepare_pick_place']
    elif any(code in status for code in (
            'NO_CACHED_GRASP', 'NO_FRESH_CACHED_GRASP', 'GRASP_PLAN_REQUIRED')):
        target, missing = 'get_cached_grasp', ['cache_grasp_plan']
    elif any(code in status for code in (
            'OBSERVATION_NOT_IN_CACHE', 'OBSERVATION_UNAVAILABLE',
            'LAYOUT_SYNCHRONIZED_OBSERVATION_NOT_AVAILABLE')):
        missing = ['Pin the exact source with pin_grasp_observation immediately after capture; '
                   'if already evicted, recapture and redo its perception/grasp inputs.']
    elif ('ACTIVE_OBJECT_REQUIRED' in status and name == 'calculate_active_grasps' and
          not (arguments or {}).get('arguments', {}).get('target_description', '').strip()):
        target = 'calculate_active_grasps'
        missing = ['arguments.target_description must equal the stored object name, e.g. cup']
    elif 'ACTIVE_' in status and ('REQUIRED' in status or 'STALE' in status):
        target, missing = 'get_active_target', ['store_segmented_target or resolve_target']
    elif 'SYNCHRONIZED_IMAGE_AND_CLOUD_REQUIRED' in status:
        missing = ['capture_observation: bind image + cloud + camera_transform together']
    elif 'DEPENDENCY_' in status or 'BINDING_' in status:
        missing = ['successful same-context predecessor with matching field type/source header']
    contract = dependency_contract(target)
    unknown = any(key in status for key in ('STATE_UNKNOWN', 'OUTCOME_UNKNOWN', 'TIMEOUT'))
    sequence = list(contract['execution_sequence'])
    if 'SYNCHRONIZED_IMAGE_AND_CLOUD_REQUIRED' in status:
        sequence = ['capture_observation', target]
    actual_name = (arguments or {}).get('function_name', name) if name == 'run_step' else name
    if target != actual_name and sequence[-1] != actual_name:
        sequence.append(actual_name)
    submitted = (arguments or {}).get('bindings', [])
    bindings = list(submitted)
    targets = {item['target_field'] for item in submitted}
    for item in contract['prerequisites']['bindings']:
        if item['target_field'] not in targets:
            bindings.append(dict(item))
    hint = {'failed_tool': actual_name, 'missing_prerequisites': missing,
            'execution_sequence': sequence,
            'bindings': bindings,
            'submitted_bindings': submitted,
            'valid_binding_targets': contract['valid_binding_targets'],
            'response_fields': contract['response_fields'],
            'context_state': contract['prerequisites']['context_state'],
            'guidance': 'Use actual predecessor step IDs from get_step_context. Keep original '
            'status/receipt. A missing cache can mean absent, expired or invalidated data, '
            'not proof its preparation was never called.',
            'automatic_retry_allowed': False, 'requires_local_reconciliation': unknown,
            'resource': RESOURCE}
    if not unknown and any(code in status for code in (
            'NO_CACHED_GRASP', 'NO_FRESH_CACHED_GRASP', 'GRASP_PLAN_REQUIRED')):
        hint['recovery_steps'] = cache_recovery_steps(arguments)
        hint['guidance'] += (
            ' Inspect the current context results and latest revision; a prior call may exist '
            'but have failed. These are proposed same-context calculation steps, NOT executed '
            'tasks. Stop at any failure. Fresh capture is included to avoid mixing stale '
            'image/box/mask sources. Never reissue a cached failed request/step ID to retry.')
    if any(code in status for code in (
            'OBSERVATION_NOT_IN_CACHE', 'OBSERVATION_UNAVAILABLE',
            'LAYOUT_SYNCHRONIZED_OBSERVATION_NOT_AVAILABLE')):
        hint['guidance'] = (
            'The exact RGB-D triplet is unavailable to VAMP; delivery, eviction and restart '
            'are different possible causes, not proven by this status. Inspect component '
            'Inspect observation_retention and component diagnostics. Extra seed bindings cannot '
            'restore it. Call capture_observation -> pin_grasp_observation on its cloud.header '
            'before inference, then redo presence/ground/segment/prepare/calculate from that '
            'one source. Do not substitute the latest frame for an old mask or change stamps. '
            'LAYOUT is a legacy status prefix; it does not require a layout workflow.')
    if 'MCP_CONTEXT_OWNER_REQUIRED' in status:
        hint['guidance'] = (
            'Do not invent a context or use output.context_id. Start capture_observation '
            'with context_id="" and fresh request/step IDs, then use its returned top-level '
            'context_id (native MCP result.structuredContent.context_id) for SAME-context '
            'bindings. get_workflow_example(workflow="perceive_scene", object_name="cup", '
            'destination_name="tray") returns exact proposed inputs.')
    if 'SYNCHRONIZED_IMAGE_AND_CLOUD_REQUIRED' in status:
        hint['guidance'] = (
            'Supply ALL three bindings from one successful capture step in this context: '
            'image<-CAPTURE_STEP_ID.image, observation_cloud<-CAPTURE_STEP_ID.cloud, '
            'camera_transform<-CAPTURE_STEP_ID.camera_transform. Use its returned top-level '
            'context_id. Do not mix captures or bind image alone. APIs supporting internal '
            'capture also allow omitting ALL three, never a partial triplet.')
    if 'SCENE_INFERENCE_UNKNOWN' in status:
        hint['perception_outcome'] = 'unknown'
        hint['guidance'] = (
            'The VLM response failed the scene contract; this is NOT evidence of absence '
            'or a missing pin. Preserve the raw response/status. Do not update conclusions '
            'from partial output. Use a fresh capture and NEW IDs for a deliberate new '
            'perception attempt, only after known-state checks; never replay an unknown action.')
    if name == 'calculate_active_grasps' and 'ACTIVE_OBJECT_REQUIRED' in status:
        hint['guidance'] += (
            ' Set arguments.target_description to the exact stored name. Empty names do not '
            'select an active object; image/mask/header bindings cannot fix the missing name.')
    if unknown:
        hint['guidance'] = 'Outcome is unknown. Do not replay, reset the ledger, or call '
        hint['guidance'] += (
            'preparation to bypass the barrier; inspect receipts and use local'
            ' review.'
        )
    return hint
