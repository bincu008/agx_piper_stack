"""Stop stalled tool loops in the native Nanobot CLI before redispatch."""

import json
from functools import wraps


def _call_key(call):
    function = call["function"]
    arguments = function["arguments"]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    return function["name"], json.dumps(arguments, sort_keys=True)


def _result_key(name, content):
    if not isinstance(content, str):
        return content
    try:
        data = json.loads(content)
        if name == "mcp_ros2_get_arm_state":
            response = data.get("response", {})
            for field in ("joints", "tcp_pose"):
                # A new ROS sample timestamp by itself is no task progress.
                response.get(field, {}).get("header", {}).pop("stamp", None)
        return data
    except (ValueError, AttributeError):
        return content


def stalled_calls(transcript, calls):
    """Match the proposed batch against two completed, unchanged batches."""
    if not calls:
        return False
    keys = [_call_key(call) for call in calls]
    batches = []
    end = len(transcript)
    for _ in range(2):
        start = end
        while start and transcript[start - 1].get("role") == "tool":
            start -= 1
        if start == end or not start:
            return False
        assistant = transcript[start - 1]
        prior = assistant.get("tool_calls") or []
        if assistant.get("role") != "assistant" or [_call_key(call) for call in prior] != keys:
            return False
        results = {msg.get("tool_call_id"): msg.get("content") for msg in transcript[start:end]}
        if len(results) != len(prior) or any(call["id"] not in results for call in prior):
            return False
        batches.append([_result_key(name, results[call["id"]])
                        for (name, _), call in zip(keys, prior)])
        end = start - 1
    return batches[0] == batches[1]


def install():
    """Wrap the installed runner's model-request boundary, before tool checkpointing."""
    from nanobot.agent.runner import AgentRunner

    original = AgentRunner._request_model

    @wraps(original)
    async def request(self, spec, messages, hook, context, *, transcript, **kwargs):
        response, usage = await original(
            self, spec, messages, hook, context, transcript=transcript, **kwargs
        )
        if response.should_execute_tools and stalled_calls(
            transcript or [], [call.to_openai_tool_call() for call in response.tool_calls]
        ):
            names = ", ".join(call.name for call in response.tool_calls)
            response.tool_calls = []
            response.finish_reason = "stop"
            response.content = (
                f"Stopped before a third identical tool attempt ({names}). "
                "The previous two attempts returned unchanged results."
            )
        return response, usage

    AgentRunner._request_model = request
