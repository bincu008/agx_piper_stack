"""Exercise the launcher guard and installed compaction path without ROS or inference."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nanobot.agent.context_governance import (
    ContextCompactionState,
    ContextGovernanceConfig,
    ContextGovernor,
    ModelRequestState,
)
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.execution import execute_tool_calls
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMResponse, ToolCallRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repeat_tool_guard import install, stalled_calls

NAME = "mcp_ros2_get_arm_state"


def pair(call_id, stamp, width):
    call = ToolCallRequest(id=call_id, name=NAME, arguments={}).to_openai_tool_call()
    result = {"response": {
        "joints": {"header": {"stamp": stamp}},
        "gripper_width_m": width,
    }}
    return [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call_id, "name": NAME, "content": json.dumps(result)},
    ]


async def main():
    first, second, third = pair("a", 1, 0.03), pair("b", 2, 0.03), pair("c", 3, 0.03)
    proposed = third[0]["tool_calls"]
    assert not stalled_calls(first, proposed)
    assert stalled_calls(first + second, proposed)
    assert not stalled_calls(first + pair("b", 2, 0.04), proposed)
    assert not stalled_calls(first + second + [{"role": "user", "content": "Retry"}], proposed)
    different = ToolCallRequest(id="d", name=NAME, arguments={"other": 1})
    assert not stalled_calls(first + second, [different.to_openai_tool_call()])

    async def model_request(*_args, **_kwargs):
        return LLMResponse(content=None, tool_calls=[ToolCallRequest(id="c", name=NAME, arguments={})]), None

    with patch.object(AgentRunner, "_request_model", model_request):
        install()
        response, _ = await AgentRunner._request_model(None, None, [], None, None, transcript=first + second)
        assert not response.should_execute_tools and "Stopped before a third" in response.content

    async def fake_dispatch(name, params):
        assert name == NAME and params == {}
        return "synthetic status"

    calls = [ToolCallRequest(id="test", name=NAME, arguments={})]
    _, events = await execute_tool_calls(
        SimpleNamespace(execute=fake_dispatch), calls, concurrent=False,
        external_lookup_counts={}, workspace_violation_counts={},
        hook=AgentHook(), context=AgentHookContext(iteration=0, messages=[]),
    )
    assert events[0]["name"] == NAME and events[0]["status"] == "ok"

    prefix = [{"role": "user", "content": "Previous turn"}, *first]
    raw = prefix + second

    async def summarize(*_args):
        return "Prior turn summary"

    governor = ContextGovernor()
    compaction = ContextCompactionState(
        raw_messages=raw,
        accepted_messages=prefix,
        raw_accepted_boundary=len(prefix),
        active_summary=None,
        summary_transcript_builder=lambda summary: [{"role": "system", "content": summary}],
        consolidate_history=summarize,
        consolidate_provider_compaction=None,
    )
    state = ModelRequestState(
        config=ContextGovernanceConfig(None, "test", ToolRegistry(), None, None, 4096),
        conversation=SimpleNamespace(replace_transcript=lambda _: None),
        compaction=compaction,
    )
    prepared = await governor._compact_request_history(
        state, compaction, raw, (5000, "test"), tool_definitions=[],
    )
    assert prepared[-2:] == second
    assert raw == prefix + second
    print("PASS: repeat guard stops before third dispatch; progress and new turns reset it.")
    print("PASS: compaction summarizes accepted history and keeps the pending assistant/result pair intact.")


if __name__ == "__main__":
    asyncio.run(main())
