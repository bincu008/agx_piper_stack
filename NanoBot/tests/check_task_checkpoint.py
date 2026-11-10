"""Verify rolling robot-task references survive the installed compaction path."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nanobot.agent.context_governance import (
    ContextCompactionState, ContextGovernanceConfig, ContextGovernor, ModelRequestState,
)
from nanobot.agent.hook import AgentHookContext
from nanobot.providers.base import ToolCallRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import task_checkpoint as checkpoint


async def check():
    with tempfile.TemporaryDirectory() as directory, patch.object(checkpoint, "STORE", Path(directory)):
        checkpoint.install()
        hook = checkpoint.TaskCheckpointHook("cli:checkpoint-test")
        context = AgentHookContext(iteration=0, messages=[])
        submission = {"request_id": "original", "operation": "pick_place", "object_name": "beige cup", "destination": "tray"}
        tool = ToolCallRequest(id="submit", name="mcp_ros2_prepare_manipulation", arguments=submission)
        await hook.before_execute_tool(context, tool, None, submission)
        assert hook.store.records()[0][1]["state"] == "SUBMISSION_UNCONFIRMED"
        first = {"task_id": "task_test", "state": "RUNNING", "updated_at": 1, "source_mode": "mock_replay"}
        await hook.after_execute_tool(context, tool, None, submission, json.dumps(first))
        followup = ToolCallRequest(id="status", name="mcp_ros2_get_task_status", arguments={"task_id": "task_test"})
        completed = {**first, "state": "COMPLETED", "updated_at": 2, "plan": {"plan_id": "plan_test", "revision": 4}}
        await hook.after_execute_tool(context, followup, None, followup.arguments, json.dumps(completed))
        await hook.after_execute_tool(context, followup, None, followup.arguments, json.dumps(first))
        records = hook.store.records()
        assert len(records) == 1 and records[0][1]["state"] == "COMPLETED"
        assert records[0][1]["submission"] == submission
        assert records[0][0].stat().st_mode & 0o777 == 0o600
        assert not checkpoint.Checkpoints("another-session").records()

        async def summarize(*_args):
            return "Older messages summarized without task references."

        prefix = [{"role": "system", "content": "Robot task rules"}, {"role": "user", "content": "Prepare the pick."}]
        raw = [*prefix, {"role": "user", "content": "Continue."}]
        compaction = ContextCompactionState(
            raw_messages=raw, accepted_messages=prefix,
            raw_accepted_boundary=len(prefix), active_summary=None,
            summary_transcript_builder=lambda summary: [{"role": "system", "content": summary}],
            consolidate_history=summarize, consolidate_provider_compaction=None,
        )
        config = ContextGovernanceConfig(
            None, "test", SimpleNamespace(tool_names=checkpoint.TOOL_NAMES),
            checkpoint.TASK_WORKSPACE, "cli:checkpoint-test", 4096,
        )
        state = ModelRequestState(
            config=config, conversation=SimpleNamespace(replace_transcript=lambda _: None),
            compaction=compaction,
        )
        prepared = await ContextGovernor()._compact_request_history(
            state, compaction, raw, (5000, "forced-test"), tool_definitions=[],
        )
        assert "summarized without task references" in prepared[0]["content"]
        assert '"task_id":"task_test"' in prepared[0]["content"]
        assert '"plan_id":"plan_test","revision":4' in prepared[0]["content"]
        assert "task_test" not in json.dumps(raw)
        assert prepared[-1] == raw[-1]
        prepared_again = ContextGovernor().prepare_messages_for_model(config, prepared)
        assert prepared_again[0]["content"].count(checkpoint.CONTEXT_HEADER) == 1
        print("PASS: rolling task reference, exact retry inputs, stale receipt rejection, session isolation and forced compaction.")


if __name__ == "__main__":
    asyncio.run(check())
