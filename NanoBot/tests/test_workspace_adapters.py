"""Portable adapter tests; no network, Nanobot launch, ROS or physical calls."""
import importlib.util
import json
import asyncio
import tempfile
import types
from unittest.mock import patch
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import camera_visuals as visuals
import workspace_arguments as arguments


class WorkspaceAdapters(unittest.TestCase):
    def setUp(self):
        visuals._IMAGES.clear()
        arguments.activate(False)

    def messages(self, success=True):
        return [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'look'},
                {'role': 'assistant', 'tool_calls': [{'id': 'camera', 'function': {
                    'name': 'mcp_ros2_get_camera_image', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'camera', 'content': json.dumps({
                    'text': json.dumps({'success': success, 'source_mode': 'mock_replay'}),
                    'artifacts': [{'path': '/trusted/image.jpg'}]})}]

    def test_actual_image_block_is_attached_without_mutating_textual_history(self):
        visuals.remember('/trusted/image.jpg', 'data:image/jpeg;base64,/9j/AA==')
        history = self.messages()
        request = visuals.with_camera_image(history)
        self.assertEqual(len(request), len(history)+1)
        self.assertEqual(request[-1]['content'][1]['image_url']['url'],
                         'data:image/jpeg;base64,/9j/AA==')
        self.assertNotIn('/9j/', json.dumps(history))
        self.assertEqual(len(visuals.with_camera_image(request)), len(request))

    def test_failed_latest_or_new_turn_never_resurrects_old_picture(self):
        visuals.remember('/trusted/image.jpg', 'data:image/jpeg;base64,/9j/AA==')
        failed = self.messages(False)
        self.assertEqual(visuals.with_camera_image(failed), failed)
        later = self.messages()+[{'role': 'user', 'content': 'what now?'}]
        self.assertEqual(visuals.with_camera_image(later), later)

    def test_unregistered_file_path_does_not_read_local_files(self):
        history = self.messages()
        self.assertEqual(visuals.with_camera_image(history), history)

    def test_image_cache_is_bounded(self):
        for index in range(9):
            visuals.remember(str(index), 'data:image/jpeg;base64,/9j/AA==')
        self.assertEqual(len(visuals._IMAGES), 4)

    def test_only_workspace_model_schema_hides_receipt_fields(self):
        schema = {'type': 'object', 'properties': {'request_id': {}, 'object_name': {}},
                  'required': ['request_id', 'object_name']}
        self.assertIs(arguments.model_schema('prepare_manipulation', schema), schema)
        arguments.activate(True)
        public = arguments.model_schema('prepare_manipulation', schema)
        self.assertNotIn('request_id', public['properties'])
        self.assertEqual(public['required'], ['object_name'])
        self.assertIn('request_id', schema['properties'])
        self.assertIsNot(public, schema)

    def test_model_result_hides_handles_but_original_receipt_is_unchanged(self):
        receipt = {'task_id': 'private_task', 'state': 'RUNNING', 'current_step': 'ground_object',
                   'plan': {'plan_id': 'private_plan', 'revision': 2, 'execution_allowed': False}}
        visible = arguments.model_result(receipt)
        self.assertNotIn('private_task', json.dumps(visible))
        self.assertNotIn('private_plan', json.dumps(visible))
        self.assertEqual(visible['state'], 'RUNNING')
        self.assertEqual(receipt['plan']['revision'], 2)

    def test_status_uses_current_and_other_tool_fields_are_preserved(self):
        arguments.activate(True)
        schema = {'properties': {'task_id': {}, 'wait_seconds': {}}, 'required': []}
        public = arguments.model_schema('get_task_status', schema)
        self.assertEqual(set(public['properties']), {'wait_seconds'})
        self.assertEqual(arguments.model_schema('inspect_robot', schema), schema)


    def test_checkpoint_hook_injects_one_id_before_durable_submission(self):
        fake_profile = types.ModuleType('mcp_profiles')
        fake_profile.ROOT = ROOT
        fake_profile.TASK_WORKSPACE = ROOT / 'robot-tasks-workspace'
        fake_profile.CONTROL_TOOLS = ('execute_manipulation', 'control_gripper', 'move_camera')
        fake_profile.TASK_TOOLS = ('prepare_manipulation', 'get_task_status')
        fake_profile.WORKSPACE_TOOLS = ('prepare_manipulation', 'get_task_status')
        hooks = types.ModuleType('nanobot.agent.hook')
        hooks.AgentHook = object
        runtime = types.ModuleType('nanobot.runtime_context')
        runtime.RUNTIME_CONTEXT_TAG = '[Runtime Context]'
        runtime.wrap_runtime_context_lines = lambda lines: '\n'.join(lines)
        with patch.dict(sys.modules, {'mcp_profiles': fake_profile,
                'nanobot.agent.hook': hooks, 'nanobot.runtime_context': runtime}):
            spec = importlib.util.spec_from_file_location('checkpoint_fixture', ROOT/'task_checkpoint.py')
            checkpoint = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checkpoint)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint.STORE = Path(folder)
            arguments.activate(True)
            hook = checkpoint.TaskCheckpointHook('session')
            call = types.SimpleNamespace(name='mcp_ros2_prepare_manipulation')
            params = {'operation': 'grasp_lift', 'object_name': 'cup'}
            asyncio.run(hook.before_execute_tool(None, call, None, params))
            identity = params['request_id']
            asyncio.run(hook.before_execute_tool(None, call, None, params))
            self.assertEqual(params['request_id'], identity)
            saved = hook.store.records()[0][1]
            self.assertEqual(saved['submission']['request_id'], identity)
            self.assertEqual(saved['state'], 'SUBMISSION_UNCONFIRMED')
            simple = hook.store.context(simple=True)
            self.assertNotIn(identity, simple)
            self.assertIn('SUBMISSION_UNCONFIRMED', simple)
            arguments.activate(True, controls=True)
            physical_call = types.SimpleNamespace(name='mcp_ros2_execute_manipulation')
            with self.assertRaisesRegex(ValueError, 'NO_.*REVIEWED_PLAN'):
                asyncio.run(hook.before_execute_tool(None, physical_call, None, {}))
            hook.store.submission('prepare_manipulation', {'request_id': 'reviewed'})
            hook.store.receipt('prepare_manipulation', {'request_id': 'reviewed'}, json.dumps({
                'task_id': 'reviewed_task', 'state': 'COMPLETED',
                'plan': {'plan_id': 'exact_plan', 'revision': 7, 'execution_allowed': False}}))
            physical = {'stage': 'all'}
            asyncio.run(hook.before_execute_tool(None, physical_call, None, physical))
            self.assertEqual((physical['plan_id'], physical['revision']), ('exact_plan', 7))
            first_id = physical['request_id']
            asyncio.run(hook.before_execute_tool(None, physical_call, None, physical))
            self.assertEqual(physical['request_id'], first_id)
            other = checkpoint.TaskCheckpointHook('other_session')
            with self.assertRaisesRegex(ValueError, 'NO_.*REVIEWED_PLAN'):
                asyncio.run(other.before_execute_tool(None, physical_call, None, {}))
            hook.store.submission('prepare_manipulation', {'request_id': 'failed_prepare'})
            hook.store.receipt('prepare_manipulation', {'request_id': 'failed_prepare'},
                               json.dumps({'task_id': 'failed', 'state': 'FAILED'}))
            with self.assertRaisesRegex(ValueError, 'NO_SUCCESSFULLY_REVIEWED_PLAN'):
                asyncio.run(hook.before_execute_tool(None, physical_call, None, {}))

    def test_control_schema_hides_transport_fields_only_in_control_profile(self):
        schema = {'properties': {k: {} for k in ('request_id', 'plan_id', 'revision', 'stage')},
                  'required': ['request_id', 'plan_id', 'revision']}
        arguments.activate(True, controls=True)
        public = arguments.model_schema('execute_manipulation', schema)
        self.assertEqual(set(public['properties']), {'stage'})
        self.assertEqual(public['required'], [])
        self.assertIn('plan_id', schema['properties'])



if __name__ == '__main__':
    unittest.main()
