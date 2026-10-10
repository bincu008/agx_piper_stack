# INSID3 grounding server

OpenAI-compatible HTTP server that drop-in replaces the RoboBrain VLM
grounding endpoint (`{api_server_url}/chat/completions`) used by
`vlm_node`'s `vlm_reasoning_node`. Instead of a 4B-parameter VLM, grounding
is done with **INSID3** (training-free in-context segmentation on a frozen
DINOv3 encoder): the target phrase parsed from the VLM prompt selects a
pre-registered reference image+mask, INSID3 segments the live image, and the
predicted mask's bounding box is returned as `[x1, y1, x2, y2]` in normalized
0-1000 coordinates -- exactly the format the C++ client's regex expects.

## Run

From the repo root:

```bash
insid3_eval/server/insid3_server.py --port 18092
```

The shebang points at the INSID3 virtualenv
(`/home/hatch/workspace/insid3-venv/bin/python`); or invoke explicitly:

```bash
/home/hatch/workspace/insid3-venv/bin/python insid3_eval/server/insid3_server.py --port 18092
```

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--host` | 127.0.0.1 | bind address |
| `--port` | 18092 | bind port |
| `--backbone` | vit-s | `vit-s` (82MB) or `vit-b` (327MB) DINOv3 encoder |
| `--image-size` | 1024 | INSID3 model input resolution |
| `--tau` | 0.9 | clustering threshold; 0.9 validated for cup/handle grounding |
| `--weights` | `<INSID3_DIR>/pretrain/<backbone>.bin` | encoder weights |
| `--ref-dir` | `insid3_eval/ref` | reference image/mask directory |
| `--debug-dir` | - | if set, saves per-request target/mask/overlay PNGs |

`INSID3_DIR` env var overrides the INSID3 checkout location
(default `~/workspace/insid3`).

## API

- `GET /health` -> `{"status":"ok"}`
- `GET /v1/models` -> model list (contains `insid3-grounding`)
- `POST /v1/chat/completions` (also `/chat/completions`) -- OpenAI-style
  payload with image-first content array (data-URL JPEG) and the standard
  grounding prompt. Returns a chat-completion JSON whose message content is
  exactly `"[x1, y1, x2, y2]"` (normalized 0-1000), plus an `insid3` object
  with `inference_seconds`, `mask_pixels`, boxes, etc.

## Target mapping

The target is extracted from the prompt with
`this sentence describes: (.*?). Return only`:

| target contains | reference used |
|---|---|
| `handle` | `brown_cup_mask_handle.png` (cup handle only) |
| `cup` | `brown_cup_mask_whole.png` (whole cup) |
| anything else | HTTP 400, clear error message |

Requests are serialized with a lock (one inference at a time), matching the
C++ client's no-concurrent-inference policy. An empty predicted mask returns
HTTP 500 rather than a silently wrong box.
