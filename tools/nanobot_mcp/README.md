# nanobot MCP adapter

MCP server that lets a nanobot agent call this repository's operations.
It is a **thin adapter** (docs/AGENT_CONTROL_ARCHITECTURE.md): every tool
forwards to the owning backend. No solvers, planners, or duplicated state
live here.

## Tools

| Tool | What it does | Backend |
|---|---|---|
| `insid3_list_templates` | List INSID3 reference templates | template dir / server builtins |
| `insid3_infer` | INSID3 segmentation; returns raw mask path, pixel bbox **and normalized 0–1000 bbox** | `/insid3/infer` (ros2) or INSID3 HTTP (http) |
| `insid3_visualize` | Draw the mask on the image | local |
| `pipeline_list_functions` | List every catalogued `/pipeline/*` operation | `registry.py` (AST-parsed, no ROS import needed) |
| `pipeline_call` | Call one catalogued operation by name | `ros2 service call /pipeline/<name>` |

`pipeline_call` rejects any function name not in the catalog and refuses
actuating operations (`execute_plan`, `gripper_close`, …) unless
`MCP_ALLOW_ACTUATION=1` is set — mirroring the stack's
`allow_physical_grasp_execution` gate.

`insid3_infer` accepts either `image_path` or `image_data_url` (the VLM node
sends data URLs). Its result includes `bbox_normalized_1000` so an agent can
answer grounding prompts in the `[x1,y1,x2,y2]` contract the VLM node parses.

## Run

```bash
# inside the Parallels Ubuntu VM (or any ROS 2 Jazzy machine)
python3 -m venv --system-site-packages ~/nanobot_mcp_venv
~/nanobot_mcp_venv/bin/pip install "mcp>=1.0" pillow numpy

# INSID3 via direct HTTP (no ROS needed; for development)
INSID3_TRANSPORT=http INSID3_URL=http://127.0.0.1:18092 \
  INSID3_TEMPLATE_DIR=$HOME/agx_piper_data/templates \
  ~/nanobot_mcp_venv/bin/python tools/nanobot_mcp/mcp_server.py

# INSID3 via the ROS bridge node (production)
INSID3_TRANSPORT=ros2 ~/nanobot_mcp_venv/bin/python tools/nanobot_mcp/mcp_server.py
```

## Wire into nanobot

Add to `~/.nanobot/config.json` under `mcpServers`:

```json
"agx-piper": {
  "command": "/home/<user>/nanobot_mcp_venv/bin/python",
  "args": ["/home/<user>/agx_piper_stack/tools/nanobot_mcp/mcp_server.py"],
  "env": {
    "INSID3_TRANSPORT": "ros2",
    "INSID3_TEMPLATE_DIR": "/home/<user>/agx_piper_data/templates"
  }
}
```

## Adding a tool

1. If it is a pipeline operation, add it to `task_pipeline/task_pipeline/registry.py`
   — `pipeline_list_functions`/`pipeline_call` pick it up automatically.
2. Otherwise add a `@mcp.tool()` function in `mcp_server.py` that forwards to
   the owning backend. Keep per-call timing via `log_call`.

## Logs

Per-call timing: `~/agx_piper_data/mcp_results/mcp_calls.log`
(`MCP_RESULTS_DIR` overrides). Result masks/overlays land beside it.
