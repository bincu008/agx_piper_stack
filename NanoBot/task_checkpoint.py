"""Keep MCP task references outside the model's compacted transcript."""
import hashlib
import json
import os
import tempfile
import uuid
from functools import wraps
from pathlib import Path

from nanobot.agent.hook import AgentHook
from nanobot.runtime_context import RUNTIME_CONTEXT_TAG, wrap_runtime_context_lines

from mcp_profiles import ROOT, TASK_TOOLS, TASK_WORKSPACE, WORKSPACE_TOOLS, CONTROL_TOOLS

STORE = ROOT / ".private" / "robot-task-checkpoints"
SUBMISSIONS = {"observe_scene", "plan_camera_view", "prepare_manipulation", "validate_plan",
               "move_camera", "control_gripper", "execute_manipulation"}
FOLLOWUPS = {"get_task_status", "cancel_task"}
TOOL_NAMES = {"mcp_ros2_" + name for name in set(TASK_TOOLS) | set(WORKSPACE_TOOLS) | set(CONTROL_TOOLS)}
CONTEXT_HEADER = "Saved ROS task references for this session (last known, server status is authoritative)."


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Checkpoints:
    def __init__(self, session_key, root=None):
        self.directory = (root or STORE) / digest(session_key)

    def records(self):
        paths = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [(path, json.loads(path.read_text())) for path in paths]

    def save(self, path, record):
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(record, stream, ensure_ascii=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def submission(self, name, params):
        request_id = params.get("request_id")
        if not request_id:
            return
        path = self.directory / (digest("request:" + request_id) + ".json")
        if not path.exists():
            self.save(path, {"tool": name, "submission": params, "state": "SUBMISSION_UNCONFIRMED"})

    def receipt(self, name, params, result):
        if not isinstance(result, str):
            return
        try:
            data = json.loads(result)
        except ValueError:
            return
        if not isinstance(data, dict) or not data.get("task_id") or not data.get("state"):
            return
        task_id = data["task_id"]
        path = self.directory / (digest("task:" + task_id) + ".json")
        record = {}
        if name in SUBMISSIONS and params.get("request_id"):
            path = self.directory / (digest("request:" + params["request_id"]) + ".json")
            if path.exists():
                record = json.loads(path.read_text())
        else:
            for candidate, saved in self.records():
                if saved.get("task_id") == task_id:
                    path, record = candidate, saved
                    break
        if record.get("task_id") not in (None, task_id):
            return
        if (isinstance(record.get("updated_at"), (int, float))
                and isinstance(data.get("updated_at"), (int, float))
                and data["updated_at"] < record["updated_at"]):
            return
        for key in ("task_id", "state", "status", "source_mode", "input_mode", "SIMULATED",
                    "current_step", "last_successful_step", "failed_step", "observation_time", "updated_at"):
            if key in data:
                record[key] = data[key][:320] if key == "status" and isinstance(data[key], str) else data[key]
        blocker = data.get("blocker")
        if blocker is not None:
            record["blocker"] = json.dumps(blocker, ensure_ascii=False)[:420]
        else:
            record.pop("blocker", None)
        plan = data.get("plan")
        if isinstance(plan, dict) and plan.get("plan_id") and type(plan.get("revision")) is int:
            record["plan"] = {key: plan[key] for key in ("plan_id", "revision", "execution_allowed") if key in plan}
        self.save(path, record)

    def reviewed_plan(self):
        """Use the latest preparation receipt delivered to THIS session, never global latest."""
        for _, record in self.records():
            if record.get('tool') in ('prepare_manipulation', 'validate_plan'):
                if record.get('state') != 'COMPLETED' or not record.get('plan'):
                    raise ValueError('NO_SUCCESSFULLY_REVIEWED_PLAN: prepare and inspect its result')
                return dict(record['plan'])
        raise ValueError('NO_REVIEWED_PLAN_IN_THIS_SESSION')

    def context(self, simple=False):
        records = [record for _, record in self.records()]
        if not records:
            return ""
        if simple:
            current = records[0]
            compact = {key: current[key] for key in (
                "state", "status", "source_mode", "current_step", "last_successful_step",
                "failed_step", "observation_time") if key in current}
            return wrap_runtime_context_lines([
                CONTEXT_HEADER,
                "Current summary only. Use get_task_status() for current server state.",
                json.dumps(compact, ensure_ascii=False),
            ])
        # Keep prompt cost bounded; prioritize unresolved work and retain the latest plan.
        unresolved = {"SUBMISSION_UNCONFIRMED", "QUEUED", "RUNNING", "CANCEL_PENDING", "UNKNOWN"}
        selected = sorted(records, key=lambda record: record.get("state") not in unresolved)[:5]
        latest_plan = next((record for record in records if record.get("plan")), None)
        if latest_plan is not None and latest_plan not in selected:
            selected.append(latest_plan)
        encoded = "\n".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")).replace(
                "[Runtime Context", "\\u005bRuntime Context"
            ).replace("[/Runtime Context", "\\u005b/Runtime Context")
            for record in selected
        )
        return wrap_runtime_context_lines([
            CONTEXT_HEADER, encoded,
        ])


class TaskCheckpointHook(AgentHook):
    def __init__(self, session_key):
        super().__init__()
        self.store = Checkpoints(session_key)

    async def before_execute_tool(self, context, tool_call, tool, params):
        name = tool_call.name.removeprefix("mcp_ros2_")
        if tool_call.name in TOOL_NAMES and name in SUBMISSIONS:
            from workspace_arguments import _CONTROL_ACTIVE
            if (_CONTROL_ACTIVE and (name == 'execute_manipulation' or
                    name == 'control_gripper' and params.get('mode') == 'prepared')):
                if not {'plan_id', 'revision'} & params.keys():
                    plan = self.store.reviewed_plan()
                    params.update(plan_id=plan['plan_id'], revision=plan['revision'])
                elif not {'plan_id', 'revision'} <= params.keys():
                    raise ValueError('INCOMPLETE_PRIVATE_PLAN_REFERENCE')
            from workspace_arguments import _ACTIVE
            if _ACTIVE and not params.get("request_id"):
                # Once per actual invocation, before durable submission recording.
                # SDK retries keep this same params dictionary/receipt identity.
                params["request_id"] = "nb_" + uuid.uuid4().hex
            self.store.submission(name, params)

    async def after_execute_tool(self, context, tool_call, tool, params, result):
        name = tool_call.name.removeprefix("mcp_ros2_")
        if tool_call.name in TOOL_NAMES and name in SUBMISSIONS | FOLLOWUPS:
            self.store.receipt(name, params, getattr(result, "workspace_receipt", result))

    async def on_execute_tool_error(self, context, tool_call, tool, params, error):
        await self.after_execute_tool(context, tool_call, tool, params, error)


def install():
    from nanobot.agent.context_governance import ContextGovernor
    from nanobot.agent.loop import AgentLoop

    original_init = AgentLoop.__init__
    original_prepare = ContextGovernor.prepare_messages_for_model

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.workspace.resolve() == TASK_WORKSPACE:
            def checkpoint_hook(context):
                if context.session_key and not context.ephemeral:
                    return TaskCheckpointHook(context.session_key)
                return None

            self._hook_factories = [*self._hook_factories, checkpoint_hook]

    @wraps(original_prepare)
    def prepare(self, config, messages):
        prepared = original_prepare(self, config, messages)
        if (not config.session_key or config.workspace is None
                or config.workspace.resolve() != TASK_WORKSPACE
                or set(config.tools.tool_names) not in (
                    {"mcp_ros2_" + name for name in TASK_TOOLS},
                    {"mcp_ros2_" + name for name in WORKSPACE_TOOLS},
                    {"mcp_ros2_" + name for name in CONTROL_TOOLS})):
            return prepared
        saved = Checkpoints(config.session_key).context(simple=set(config.tools.tool_names) in (
            {"mcp_ros2_" + name for name in WORKSPACE_TOOLS},
            {"mcp_ros2_" + name for name in CONTROL_TOOLS}))
        if not saved or not prepared or prepared[0].get("role") != "system":
            return prepared
        system = prepared[0]
        if not isinstance(system.get("content"), str):
            return prepared
        marker = "\n\n" + RUNTIME_CONTEXT_TAG + "\n" + CONTEXT_HEADER
        content = system["content"].partition(marker)[0]
        return [{**system, "content": content + "\n\n" + saved}, *prepared[1:]]

    AgentLoop.__init__ = init
    ContextGovernor.prepare_messages_for_model = prepare
