#!/home/hatch/workspace/insid3-venv/bin/python
"""INSID3 grounding server: an OpenAI-compatible drop-in replacement for the
RoboBrain VLM grounding endpoint used by vlm_node's vlm_reasoning_node.

The C++ client POSTs to ``{api_server_url}/chat/completions`` with an
OpenAI-style payload: an image-first content array (data-URL JPEG) plus the
grounding prompt ``"Please provide the bounding box coordinate of the region
this sentence describes: {target}. Return only [x1, y1, x2, y2] in normalized
0-1000 coordinates..."``. This server parses the target out of that prompt,
runs INSID3 in-context segmentation against a pre-registered reference
image+mask for that target, and returns a chat-completion JSON whose content
is exactly ``"[x1, y1, x2, y2]"`` -- the format vlm_node's regex expects.

Target -> reference mapping:
    * target contains "handle" -> brown_cup_mask_handle.png (cup handle only)
    * target contains "cup"    -> brown_cup_mask_whole.png (whole cup)
    * anything else            -> HTTP 400 with a clear error message

Validated config (mock snapshot, 640x480): --backbone vit-s --image-size 1024
--tau 0.9. The faster vit-b@512/tau=0.6 default merged the white plate into
the cup mask and could not isolate the handle; the validated config segments
the cup cleanly and isolates the handle (see insid3_eval/results/TIMING.md).

Run (from the repo root, after cloning this branch)::

    insid3_eval/server/insid3_server.py --port 18092

The script must run under the INSID3 virtualenv (the shebang above points at
it). Alternatively invoke it explicitly::

    /home/hatch/workspace/insid3-venv/bin/python insid3_eval/server/insid3_server.py

Environment / flags:
    INSID3_DIR   path to the INSID3 checkout (default: ~/workspace/insid3)
    --weights    DINOv3 encoder weights .bin (default: <INSID3_DIR>/pretrain/<backbone>.bin)
    --backbone   vit-b (default) or vit-s  (timm DINOv3 Eva models)
    --image-size model input resolution (default: 512)
    --ref-dir    reference image/mask dir (default: <repo>/insid3_eval/ref)
    --debug-dir  if set, per-request target/mask/overlay PNGs are written here
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

INSID3_DIR = Path(os.environ.get("INSID3_DIR", str(Path.home() / "workspace" / "insid3")))
sys.path.insert(0, str(INSID3_DIR))
os.chdir(INSID3_DIR)

TARGET_RE = re.compile(r"this sentence describes:\s*(.*?)\.\s*Return only", re.DOTALL)

REFS = {
    # key -> (reference image, reference mask, description)
    "cup": ("brown_cup.jpg", "brown_cup_mask_whole.png", "whole brown cup"),
    "handle": ("brown_cup.jpg", "brown_cup_mask_handle.png", "cup handle only"),
}

BACKBONES = {
    "vit-b": ("vit_base_patch16_dinov3", "dinov3_vitb16_timm_lvd1689m.bin"),
    "vit-s": ("vit_small_patch16_dinov3", "dinov3_vits16_timm_lvd1689m.bin"),
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="INSID3 grounding server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18092)
    parser.add_argument("--backbone", choices=list(BACKBONES), default="vit-s")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.9,
                        help="agglomerative-clustering threshold; 0.9 validated "
                             "for cup/handle grounding (finer part separation)")
    parser.add_argument("--weights", default=None,
                        help="DINOv3 weights .bin (default: <INSID3_DIR>/pretrain/<backbone>.bin)")
    parser.add_argument("--ref-dir", default=str(here.parent / "ref"),
                        help="reference image/mask directory")
    parser.add_argument("--debug-dir", default=None,
                        help="write per-request target/mask/overlay PNGs here")
    return parser.parse_args()


ARGS = parse_args()

# ----------------------------------------------------------------------------
# model (loaded once at startup)
# ----------------------------------------------------------------------------

import torch  # noqa: E402
from torch import nn  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

import timm  # noqa: E402
from models.insid3 import INSID3  # noqa: E402


class Dinov3Encoder(nn.Module):
    """Adapter exposing the DINOv3-style get_intermediate_layers API over
    timm's Eva implementation. norm=True is REQUIRED (official DINOv3 applies
    the final norm by default); without it the similarity pipeline degenerates
    into all-foreground masks."""

    def __init__(self, eva_model: nn.Module):
        super().__init__()
        self.net = eva_model

    def get_intermediate_layers(self, x, n=1, reshape=True):
        return self.net.forward_intermediates(
            x, n, intermediates_only=True, norm=True,
            output_fmt="NCHW" if reshape else "NLC",
        )

    def forward(self, x):
        return self.net(x)


def load_model() -> "INSID3":
    timm_name, weights_file = BACKBONES[ARGS.backbone]
    weights = Path(ARGS.weights) if ARGS.weights else INSID3_DIR / "pretrain" / weights_file
    if not weights.exists():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
    t0 = time.time()
    eva = timm.create_model(timm_name, pretrained=False)
    sd = torch.load(str(weights), map_location="cpu", weights_only=True)
    missing, unexpected = eva.load_state_dict(sd, strict=False)
    print(f"[insid3] encoder load_state_dict: missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    model = INSID3(encoder=Dinov3Encoder(eva), image_size=ARGS.image_size,
                   device="cpu", tau=ARGS.tau)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    dt = time.time() - t0
    print(f"[insid3] model loaded in {dt:.1f}s "
          f"(backbone={ARGS.backbone}, image_size={ARGS.image_size}, tau={ARGS.tau})",
          flush=True)
    return model


MODEL = load_model()
INFER_LOCK = threading.Lock()
REQUEST_ID = 0
REQUEST_ID_LOCK = threading.Lock()

REF_DIR = Path(ARGS.ref_dir)
for _key, (img, msk, _desc) in REFS.items():
    assert (REF_DIR / img).exists(), f"missing reference image {REF_DIR / img}"
    assert (REF_DIR / msk).exists(), f"missing reference mask {REF_DIR / msk}"

DEBUG_DIR = Path(ARGS.debug_dir) if ARGS.debug_dir else None
if DEBUG_DIR:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# grounding helpers
# ----------------------------------------------------------------------------

from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402


def extract_target(prompt: str) -> str | None:
    m = TARGET_RE.search(prompt or "")
    return m.group(1).strip() if m else None


def pick_reference(target: str) -> tuple[str, str, str] | None:
    low = target.lower()
    if "handle" in low:
        return REFS["handle"]
    if "cup" in low:
        return REFS["cup"]
    return None


def mask_to_boxes(mask: np.ndarray, width: int, height: int):
    """Bounding box of a binary mask -> (normalized 0-1000 box, pixel box)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x1p, y1p, x2p, y2p = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    # keep the client's strict x1<x2 / y1<y2 invariant even for 1px masks
    if x2p == x1p:
        x2p = min(width - 1, x1p + 1)
    if y2p == y1p:
        y2p = min(height - 1, y1p + 1)
    norm = [round(x1p / width * 1000), round(y1p / height * 1000),
            round(x2p / width * 1000), round(y2p / height * 1000)]
    norm = [max(0, min(1000, v)) for v in norm]
    if norm[2] <= norm[0]:
        norm[2] = min(1000, norm[0] + 1)
    if norm[3] <= norm[1]:
        norm[3] = min(1000, norm[1] + 1)
    return norm, (x1p, y1p, x2p, y2p)


def run_grounding(image: Image.Image, target: str):
    """Full per-request path: reference lookup + set_reference/set_target/segment."""
    ref = pick_reference(target)
    if ref is None:
        raise ValueError(
            f"unsupported grounding target {target!r}: no INSID3 reference registered "
            f"(supported: targets containing 'cup' or 'handle')")
    ref_img, ref_mask, desc = ref
    t0 = time.time()
    with INFER_LOCK:  # serialize: one request at a time, like the C++ client's policy
        MODEL.set_reference(str(REF_DIR / ref_img), str(REF_DIR / ref_mask))
        MODEL.set_target(image)
        pred = MODEL.segment()
    dt = time.time() - t0
    mask = pred.detach().cpu().squeeze().numpy() > 0
    boxes = mask_to_boxes(mask, image.width, image.height)
    if boxes is None:
        raise RuntimeError(f"INSID3 predicted an empty mask for target {target!r}")
    return desc, dt, mask, boxes


# ----------------------------------------------------------------------------
# HTTP layer (OpenAI-style)
# ----------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "INSID3-Grounding/1.0"

    def log_message(self, fmt, *args):  # keep logs on stdout, single line
        sys.stdout.write("[http] " + fmt % args + "\n")
        sys.stdout.flush()

    def _json(self, status: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._json(200, {"status": "ok"})
        elif self.path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": "insid3-grounding", "object": "model",
                 "owned_by": "insid3", "backbone": ARGS.backbone}]})
        else:
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._json(400, {"error": {"message": f"invalid JSON: {exc}"}})
            return

        # find the user message: image-first content array + text prompt
        image_b64, prompt = None, ""
        for msg in payload.get("messages", []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for part in parts:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:image/") and ";base64," in url:
                        image_b64 = url.split(";base64,", 1)[1]
                elif part.get("type") == "text":
                    prompt += part.get("text", "")
        if not image_b64:
            self._json(400, {"error": {"message": "no base64 data-URL image in request"}})
            return
        target = extract_target(prompt)
        if not target:
            self._json(400, {"error": {"message": "could not parse grounding target from prompt"}})
            return

        global REQUEST_ID
        with REQUEST_ID_LOCK:
            REQUEST_ID += 1
            rid = REQUEST_ID
        t_req = time.time()
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        except Exception as exc:
            self._json(400, {"error": {"message": f"could not decode image: {exc}"}})
            return
        try:
            desc, infer_s, mask, (norm_box, pixel_box) = run_grounding(image, target)
        except ValueError as exc:  # unsupported target -> 400
            self._json(400, {"error": {"message": str(exc)}})
            return
        except Exception as exc:  # inference failure -> 500
            self._json(500, {"error": {"message": f"INSID3 inference failed: {exc}"}})
            return
        total_s = time.time() - t_req

        if DEBUG_DIR:
            tag = re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_") or "target"
            stem = DEBUG_DIR / f"req{rid:03d}_{tag}"
            image.save(str(stem.with_suffix(".jpg")))
            Image.fromarray((mask.astype("uint8")) * 255).save(str(stem.with_name(stem.name + "_mask.png")))
            try:
                from utils.visualization import visualize_prediction
                visualize_prediction(
                    str(REF_DIR / REFS["cup" if desc.startswith("whole") else "handle"][0]),
                    str(REF_DIR / REFS["cup" if desc.startswith("whole") else "handle"][1]),
                    image, mask, str(stem.with_name(stem.name + "_overlay.png")))
            except Exception as exc:
                print(f"[insid3] debug overlay failed: {exc}", flush=True)

        x1, y1, x2, y2 = norm_box
        content = f"[{x1}, {y1}, {x2}, {y2}]"
        print(f"[insid3] req={rid} target={target!r} ref={desc} "
              f"infer_s={infer_s:.2f} total_s={total_s:.2f} "
              f"box={content} mask_px={int(mask.sum())}", flush=True)
        self._json(200, {
            "id": f"chatcmpl-insid3-{rid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "insid3-grounding"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "insid3": {
                "request_id": rid,
                "target": target,
                "reference": desc,
                "inference_seconds": round(infer_s, 3),
                "total_seconds": round(total_s, 3),
                "mask_pixels": int(mask.sum()),
                "normalized_box": norm_box,
                "pixel_box": list(pixel_box),
                "image_size": [image.width, image.height],
            },
        })


def main() -> None:
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"[insid3] serving on http://{ARGS.host}:{ARGS.port} "
          f"(POST /v1/chat/completions, GET /health)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
