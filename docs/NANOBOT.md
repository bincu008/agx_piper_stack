# nanobot integration

nanobot (HKUDS, MIT) is the LLM agent loop for this stack. It talks to the
robot exclusively through the MCP adapter (`tools/nanobot_mcp/`), which
forwards to the catalogued `/pipeline/*` operations and the INSID3 grounding
APIs. The agent never touches drivers directly.

## Install (inside the Parallels Ubuntu VM)

```bash
# 1. nanobot itself
python3 -m venv ~/nanobot_venv
~/nanobot_venv/bin/pip install "nanobot-ai==0.3.5"

# 2. local LLM runtime (llama.cpp, CPU)
mkdir -p ~/llama-bin && cd ~/llama-bin
curl -sL -o llama.tgz \
  https://github.com/ggerganov/llama.cpp/releases/download/b11560/llama-b11560-bin-ubuntu-x64.tar.gz
tar xzf llama.tgz

# 3. agent model (minimum viable for reliable tool calling: 3B)
curl -sL -o ~/models/qwen3b.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf

# 4. MCP adapter venv
python3 -m venv --system-site-packages ~/nanobot_mcp_venv
~/nanobot_mcp_venv/bin/pip install "mcp>=1.0" pillow numpy
```

VM sizing: 6 vCPUs / 8 GB RAM comfortably holds the LLM server (~3 GB),
the INSID3 server (~1 GB) and the ROS stack.

## Configure

`~/.nanobot/config.json` (provider = local llama.cpp server):

```json
{
  "providers": {
    "local": {
      "api_base": "http://127.0.0.1:11435/v1",
      "api_key": "none",
      "models": ["qwen3b"]
    }
  },
  "default_model": "local/qwen3b",
  "mcpServers": {
    "agx-piper": {
      "command": "/home/<user>/nanobot_mcp_venv/bin/python",
      "args": ["/home/<user>/agx_piper_stack/tools/nanobot_mcp/mcp_server.py"],
      "env": {
        "INSID3_TRANSPORT": "ros2",
        "INSID3_TEMPLATE_DIR": "/home/<user>/agx_piper_data/templates"
      }
    }
  }
}
```

Start the LLM server (16K context; the agent's tool schemas need it):

```bash
~/llama-bin/llama-b11560/llama-server -m ~/models/qwen3b.gguf \
  --host 127.0.0.1 --port 11435 -c 16384 --threads 6
```

## Launch modes

**Interactive / one-shot** (development, demos):

```bash
~/nanobot_venv/bin/nanobot agent --classic --no-logs \
  -m "Find the meatball container in /path/to/image.png using your insid3 tools."
```

**Serve mode** (what the VLM node talks to):

```bash
~/nanobot_venv/bin/nanobot serve --host 127.0.0.1 --port 18093 --timeout 600
```

`nanobot serve` exposes OpenAI-compatible `POST /v1/chat/completions` and runs
the full agent loop (including MCP tool calls) per request, returning the
agent's final text.

## Routing the VLM node to nanobot (instead of RoboBrain)

No C++ change is needed: `vlm_reasoning_node` already POSTs OpenAI-format
requests to a configurable endpoint. Select the `nanobot` inference profile:

```bash
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=nanobot allow_physical_grasp_execution:=false
```

This points the node at `127.0.0.1:18093` with a 600 s timeout (agent loops
take minutes on CPU). Grounding prompts (`"Please provide the bounding box
coordinate of the region this sentence describes: ..."`) go to the agent,
which calls `insid3_list_templates` → `insid3_infer` and answers with
`[x1,y1,x2,y2]` in normalized 0–1000 coordinates. `parseGroundingBoxLenient`
(`vlm_node/grounding.hpp`) accepts the box embedded in agent-style prose and
rounds fractional coordinates; the strict exact-match parse remains primary.

For grounding to work the agent needs an instruction to answer grounding
prompts with only the normalized box. Put it in the nanobot workspace
instructions (`~/.nanobot/workspace/AGENTS.md` or the session prompt):

> When asked for a bounding box, first pick the matching template with
> insid3_list_templates, run insid3_infer, then reply with ONLY the
> bbox_normalized_1000 value as [x1,y1,x2,y2] (integers, 0–1000), no other text.

## INSID3 reference images

The server loads templates from `--template-dir <dir>` at startup
(`manifest.json` + per-template `ref.jpg` / `mask.png` / `description.txt`).
Add reference pairs there; the MCP `insid3_list_templates` tool picks them up
automatically. Keep the images out of git unless explicitly approved.

## Reference timings (2-vCPU x86 VM; Apple Silicon VM is ~3–4x faster)

| Step | Time |
|---|---|
| `insid3_infer` (ViT-S @1024px) | ~15–18 s |
| `insid3_visualize` | ~1 s |
| nanobot agent turn (3B, CPU) | minutes; a 3-tool task took ~7.5 min wall, 91% LLM |
