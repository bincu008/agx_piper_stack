"""Portable fail-closed snapshot tests; no network or native Nanobot dependencies."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workspace_discovery import WORKSPACE_NAMES, load_snapshot, validate_snapshot, require_endpoint


class WorkspaceDiscovery(unittest.TestCase):
    def fixtures(self):
        return ({'discovery_id': 'one', 'endpoint': 'https://fixture/mcp',
                 'tools': [{'name': name, 'inputSchema': {'type': 'object'}}
                           for name in WORKSPACE_NAMES]},
                {'discovery_id': 'one', 'endpoint': 'https://fixture/mcp', 'result': {
                    'isError': False, 'structuredContent': {'mode': 'preparation_only',
                        'input_provenance': {'classification': 'mock_replay_publishers'},
                        'allowlist': sorted(WORKSPACE_NAMES)}}})

    def test_valid_workspace_pair(self):
        tools, status = self.fixtures()
        self.assertEqual(validate_snapshot(tools, status)[0], WORKSPACE_NAMES)

    def test_mismatched_pair_and_wrong_mode_source_missing_tools_rejected(self):
        tools, status = self.fixtures()
        for change in ('pair', 'mode', 'source', 'missing', 'duplicate', 'schema'):
            t, s = copy.deepcopy(tools), copy.deepcopy(status)
            if change == 'pair':
                s['discovery_id'] = 'other'
            elif change == 'mode':
                s['result']['structuredContent']['mode'] = 'agent_pipeline'
            elif change == 'source':
                s['result']['structuredContent']['input_provenance']['classification'] = 'unverified'
            elif change == 'missing':
                t['tools'].pop()
            elif change == 'duplicate':
                t['tools'].append(t['tools'][0])
            else:
                t['tools'][0]['inputSchema'] = {'type': 'string'}
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_snapshot(t, s)

    def test_no_fallback_to_legacy_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'results'
            out.mkdir()
            (out / 'mcp-tools.json').write_text(json.dumps(self.fixtures()[0]))
            with self.assertRaisesRegex(SystemExit, 'Refresh workspace discovery'):
                load_snapshot(directory)


    def test_discovery_cannot_configure_a_different_server(self):
        _, snapshot = validate_snapshot(*self.fixtures())
        require_endpoint(snapshot, 'https://fixture/mcp')
        with self.assertRaisesRegex(SystemExit, 'ENDPOINT_MISMATCH'):
            require_endpoint(snapshot, 'https://other/mcp')


    def test_control_profile_requires_live_enabled_cards_and_separate_mode(self):
        tools, status = self.fixtures()
        for name in ('move_camera', 'control_gripper', 'execute_manipulation'):
            tools['tools'].append({'name': name, 'inputSchema': {'type': 'object'},
                                   '_meta': {'availability': 'guarded_execution'}})
            status['result']['structuredContent']['allowlist'].append(name)
        with self.assertRaises(ValueError):
            validate_snapshot(tools, status, controls=True)
        data = status['result']['structuredContent']
        data['mode'] = 'agent_pipeline'
        data['input_provenance']['classification'] = 'live_driver_publishers'
        validate_snapshot(tools, status, controls=True)
        tools['tools'][-1]['_meta']['availability'] = 'disabled'
        with self.assertRaisesRegex(ValueError, 'ENABLED_CONTROL_CARDS'):
            validate_snapshot(tools, status, controls=True)
