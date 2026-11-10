"""Refresh the ROS task catalog and verify the task surface in mock replay."""
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import httpx
from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import Config
from nanobot.security.network import configure_ssrf_whitelist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_profiles import P, ROOT, TASK_TOOLS, apply_profile, install_task_tool_filter

RESULTS = ROOT / "results"
ACTIVE = {"QUEUED", "RUNNING", "CANCEL_PENDING"}


def write(name, value):
    path = RESULTS / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def body(result):
    return result.structuredContent or json.loads(result.content[0].text)


async def discover(server):
    async with httpx.AsyncClient(headers=server["headers"], timeout=15) as client:
        async with streamable_http_client(server["url"], http_client=client) as (read, send, _):
            async with ClientSession(read, send) as session:
                initialized = await session.initialize()
                tools = []
                cursor = None
                while True:
                    page = await session.list_tools(cursor=cursor)
                    tools.extend(page.tools)
                    cursor = page.nextCursor
                    if not cursor:
                        break
                catalog = await session.call_tool("list_functions", {"summary_only": True})
                status = body(catalog)
                expected = set(TASK_TOOLS) | set(P["all-calculations"])
                assert initialized.serverInfo.version == "3.6.0", initialized.serverInfo.version
                assert len(tools) == 62 and {tool.name for tool in tools} == expected
                assert set(status["allowlist"]) == expected and not catalog.isError
                assert status["mode"] == "preparation_only" and status["input_mode"] == "mock_replay"
                resources = {}
                for name in ("tasks", "task-catalog"):
                    result = await session.read_resource("ros-pipeline://" + name)
                    resources[name] = result.model_dump(mode="json")
    write("mcp-initialize.json", initialized.model_dump(mode="json"))
    write("mcp-tools.json", [tool.model_dump(mode="json") for tool in tools])
    write("mcp-status-list_functions.json", catalog.model_dump(mode="json"))
    for name, result in resources.items():
        write(f"mcp-resource-{name}.json", result)
    print("Discovery: MCP 3.6.0, preparation_only/mock_replay, 62 native tools", flush=True)
    return {tool.name: tool.inputSchema for tool in tools}


async def verify():
    logger.remove()
    raw = json.loads((Path.home() / ".nanobot/config.json").read_text())
    schemas = await discover(raw["tools"]["mcpServers"]["ros2"])
    apply_profile("robot-tasks", raw)
    config = Config.model_validate(raw)
    configure_ssrf_whitelist(config.tools.ssrf_whitelist)
    install_task_tool_filter()
    registry = ToolRegistry()
    loop = AgentLoop.from_config(config, tool_registry=registry)
    assert not loop.tools.tool_names
    mcp = MCPProvider.from_config(config, registry)
    report = []
    try:
        await mcp.connect()
        definitions = registry.get_definitions()
        assert {d["function"]["name"] for d in definitions} == {"mcp_ros2_" + n for n in TASK_TOOLS}
        for definition in definitions:
            tool = definition["function"]
            original = schemas[tool["name"].removeprefix("mcp_ros2_")]
            assert tool["parameters"] == original, tool["name"]
        write("mcp-robot-task-definitions.json", definitions)
        print("Translation: all eight native input schemas preserved; eight tools registered", flush=True)

        async def call(name, args):
            result = await registry.execute("mcp_ros2_" + name, args)
            try:
                payload = json.loads(result)
            except ValueError:
                payload = {"unparsed_result": str(result)}
            report.append({"tool": name, "arguments": args, "result": payload})
            write("nanobot-robot-task-verification.json", report)
            print(name, payload.get("state"), payload.get("status"), payload.get("task_id"), flush=True)
            return payload

        async def finish(task):
            if not task.get("task_id"):
                return task
            deadline = time.monotonic() + 330
            while task.get("state") in ACTIVE and time.monotonic() < deadline:
                task = await call("get_task_status", {"task_id": task["task_id"], "wait_seconds": 40})
                if task.get("state") in ACTIVE:
                    await asyncio.sleep(1)
            return task

        workflow = await call("describe_workflow", {"operation": "pick_place"})
        assert workflow.get("status") == "WORKFLOW_DESCRIPTION"
        robot = await call("inspect_robot", {})
        assert robot.get("SIMULATED") is True and robot["result"]["robot"]["physical_grasp_execution_enabled"] is False
        observed = await call("observe_scene", {
            "request_id": "verify-observe-" + uuid.uuid4().hex,
            "targets": ["beige cup"], "detail": "location",
        })
        await finish(observed)
        args = {
            "request_id": "verify-pick-place-" + uuid.uuid4().hex,
            "operation": "pick_place", "object_name": "beige cup",
            "destination": "dark rectangular tray", "lift_distance_m": 0.05,
        }
        first = await call("prepare_manipulation", args)
        finished = await finish(first)
        repeat = await call("prepare_manipulation", args)
        assert first.get("task_id") and first["task_id"] == repeat.get("task_id")
        assert first["created_at"] == repeat["created_at"]
        if finished.get("state") == "COMPLETED":
            assert finished["updated_at"] == repeat["updated_at"]
            assert finished.get("plan") == repeat.get("plan")
        plan = finished.get("plan") or {}
        plan_id, revision = plan.get("plan_id"), plan.get("revision")
        if plan_id is not None and revision is not None:
            await call("validate_plan", {
                "request_id": "verify-plan-" + uuid.uuid4().hex,
                "plan_id": plan_id, "revision": revision,
            })
    finally:
        await mcp.aclose()


if __name__ == "__main__":
    asyncio.run(verify())
