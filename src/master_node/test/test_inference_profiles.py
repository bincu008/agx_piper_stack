"""Check selectable defaults and explicit overrides without starting nodes."""

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
import pytest


def launch_defaults(profile, overrides=None):
    path = Path(__file__).resolve().parents[1] / 'launch' / 'reasoning_stack.launch.py'
    spec = importlib.util.spec_from_file_location('reasoning_profile_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update(inference_profile=profile)
    context.launch_configurations.update(overrides or {})
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    return context.launch_configurations


@pytest.mark.parametrize('profile,port', [('server', '18091'), ('lmstudio', '1234')])
def test_inference_profile_selects_consistent_endpoint(profile, port):
    defaults = launch_defaults(profile)
    assert defaults['api_server_port'] == port
    assert defaults['api_server_url'] == (
        f'http://{defaults["api_server_ip"]}:{port}/v1')
    assert defaults['model_name'] == 'robobrain2.5-4b'
    assert defaults['request_timeout_sec'] == '120'
    assert defaults['pipeline_profiling'] == 'false'


def test_insid3_profile_selects_local_grounding_server():
    defaults = launch_defaults('insid3')
    assert defaults['api_server_ip'] == '127.0.0.1'
    assert defaults['api_server_port'] == '18092'
    assert defaults['api_server_url'] == 'http://127.0.0.1:18092/v1'
    assert defaults['model_name'] == 'insid3-grounding'


def test_explicit_endpoint_and_model_override_profile():
    defaults = launch_defaults('lmstudio', {
        'api_server_ip': '192.0.2.1', 'api_server_port': '5678', 'model_name': 'custom'})
    assert defaults['api_server_url'] == 'http://192.0.2.1:5678/v1'
    assert defaults['model_name'] == 'custom'


def test_invalid_profile_is_rejected():
    with pytest.raises(RuntimeError):
        launch_defaults('unknown')
