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

Validated config (current mock snapshot, 640x480): --backbone vit-s
--image-size 1024 --tau 0.9 --merge-threshold 0.35 --threads 2 with
reference-feature caching enabled (default). merge-threshold 0.35 excludes
the white bowls from the whole-cup mask on the current scene (beige cup +
two white bowls); the upstream default 0.2 cross-matches them. Handle
grounding is unaffected (perfect at both values).

Run (from the repo root, after cloning this branch)::

    insid3_eval/server/insid3_server.py --port 18092

The script must run under the INSID3 virtualenv (the shebang above points at
it). Alternatively invoke it explicitly::

    /home/hatch/workspace/insid3-venv/bin/python insid3_eval/server/insid3_server.py

Environment / flags:
    INSID3_DIR   path to the INSID3 checkout (default: ~/workspace/insid3)
    --weights    DINOv3 encoder weights .bin (default: <INSID3_DIR>/pretrain/<backbone>.bin)
    --backbone   vit-b (default) or vit-s  (timm DINOv3 Eva models)
    --image-size model input resolution (default: 768)
    --threads    torch CPU threads (default: half the CPUs)
    --no-ref-cache
                 disable reference-feature caching (slower, bit-identical)
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
    parser.add_argument("--image-size", type=int, default=1024,
                        help="model input resolution (1024 validated for cup/handle "
                             "grounding; 768 is ~3.6x faster but cross-matches white "
                             "bowls/plate on the whole-cup reference)")
    parser.add_argument("--threads", type=int,
                        default=max(1, (os.cpu_count() or 2) // 2),
                        help="torch CPU threads (default: half the CPUs)")
    parser.add_argument("--no-ref-cache", action="store_true",
                        help="disable reference-feature caching (recompute reference "
                             "features per request; slower, bit-identical output)")
    parser.add_argument("--tau", type=float, default=0.9,
                        help="agglomerative-clustering threshold; 0.9 validated "
                             "for cup/handle grounding (finer part separation)")
    parser.add_argument("--merge-threshold", type=float, default=0.35,
                        help="cluster aggregation score cutoff; 0.35 validated on "
                             "the current mock scene (beige cup + white bowls): "
                             "excludes the bowls while keeping the full cup. "
                             "Upstream default 0.2 cross-matches the bowls.")
    parser.add_argument("--weights", default=None,
                        help="DINOv3 weights .bin (default: <INSID3_DIR>/pretrain/<backbone>.bin)")
    parser.add_argument("--ref-dir", default=str(here.parent / "ref"),
                        help="reference image/mask directory")
    parser.add_argument("--template-dir", default=None,
                        help="directory holding manifest.json of extra "
                             "reference templates (name -> ref.jpg/mask.png); "
                             "a template is selected per-request with the "
                             "'insid3_template' payload field")
    parser.add_argument("--debug-dir", default=None,
                        help="write per-request target/mask/overlay PNGs here")
    return parser.parse_args()


ARGS = parse_args()

# ----------------------------------------------------------------------------
# model (loaded once at startup)
# ----------------------------------------------------------------------------

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

torch.set_num_threads(max(1, ARGS.threads))

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
                   device="cpu", tau=ARGS.tau, merge_threshold=ARGS.merge_threshold)
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


def _resolve(p: str | Path) -> Path:
    """Resolve a reference path: absolute stays, relative joins REF_DIR."""
    p = Path(p)
    return p if p.is_absolute() else REF_DIR / p


if ARGS.template_dir:
    # Extra reference templates (e.g. the kitchen container/food pairs):
    # <template-dir>/manifest.json maps name -> {ref_image, ref_mask,
    # description}; names become selectable via the 'insid3_template'
    # request field and are precomputed into the ref-cache like the
    # built-in cup/handle references.
    _tdir = Path(ARGS.template_dir)
    _manifest = json.loads((_tdir / "manifest.json").read_text())
    for _name, _entry in _manifest.items():
        REFS[_name] = (str(_tdir / _entry["ref_image"]),
                       str(_tdir / _entry["ref_mask"]),
                       _entry.get("description", _name))
    print(f"[insid3] loaded {len(_manifest)} templates from {_tdir}",
          flush=True)

for _key, (img, msk, _desc) in REFS.items():
    assert _resolve(img).exists(), f"missing reference image {_resolve(img)}"
    assert _resolve(msk).exists(), f"missing reference mask {_resolve(msk)}"

# Reference-feature cache: one encoder pass per registered reference at
# startup; per-request inference then only encodes the target image.
# (Populated after the helper definitions below, once precompute_ref exists.)
REF_CACHES = {}

DEBUG_DIR = Path(ARGS.debug_dir) if ARGS.debug_dir else None
if DEBUG_DIR:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# grounding helpers
# ----------------------------------------------------------------------------

from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from utils.data import downsample_mask  # noqa: E402


def precompute_ref(model, ref_img_path, ref_mask_path):
    """Run the reference side of INSID3 once; cache features for reuse.

    The reference image+mask never change per registered target, so its ViT
    features, debiased features and foreground prototype are computed once at
    startup instead of on every request. The per-request path then only pays
    for the target image. Verified bit-exact vs the uncached path
    (mask IoU 1.000).
    """
    tf = model._transform
    img = Image.open(ref_img_path).convert("RGB")
    img_t = tf(img).unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
    m = torch.tensor(np.array(Image.open(ref_mask_path).convert("L")) > 0,
                     dtype=torch.bool)
    m = F.interpolate(m.unsqueeze(0).unsqueeze(0).float(),
                      size=(model.image_size, model.image_size),
                      mode="nearest") > 0.5  # (1, 1, H, W)
    with torch.no_grad():
        fmaps = model._extract_features(img_t)          # (1, 1, C, h, w)
        fmaps_norm = F.normalize(fmaps, p=2, dim=2)
        fmaps_deb = model._debias_features(fmaps_norm)  # (1, 1, C, h, w)
        _, _, c, h, w = fmaps_norm.shape
        fg = fmaps_deb[0, 0][:, downsample_mask(m, h, w)]
        prototype = F.normalize(fg.mean(dim=1), p=2, dim=0).unsqueeze(1)
    return {"feat_ref_deb": fmaps_deb, "prototype": prototype, "ref_masks": m}


@torch.no_grad()
def segment_cached(model, cache, image):
    """Target-side-only inference reusing a precomputed reference cache.

    Mirrors INSID3.predict() exactly, but the reference features come from
    ``cache`` instead of a fresh encoder pass.
    """
    tgt = model._transform(image).unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
    fmaps = model._extract_features(tgt)
    fmaps_norm = F.normalize(fmaps, p=2, dim=2)
    _, _, c, h, w = fmaps_norm.shape
    fmaps_deb = model._debias_features(fmaps_norm)
    feat_tgt = fmaps_norm[:, 0]
    feat_tgt_deb = fmaps_deb[:, 0]
    feat_ref_deb = cache["feat_ref_deb"][:, 0]
    prototype = cache["prototype"]
    sim_m = torch.einsum("bchw,bcxy->bhwxy", feat_ref_deb, fmaps_deb[:, 0])
    candidate_mask = model._locate_candidates(
        [sim_m], cache["ref_masks"], feat_tgt_deb, prototype, h, w)
    if candidate_mask.sum() == 0:
        model._orig_tgt_size = (image.height, image.width)
        try:
            return model._finalize_mask(candidate_mask, tgt[0, 0])
        finally:
            model._orig_tgt_size = None
    from utils.clustering import (agglomerative_clustering,
                                  compute_cluster_prototypes)
    feat_tgt_flat = feat_tgt[0].reshape(c, -1).permute(1, 0)
    cluster_labels = agglomerative_clustering(
        feat_tgt_flat, model.tau).reshape(h, w)
    k = int(cluster_labels.max().item()) + 1
    feat_tgt_deb_flat = feat_tgt_deb[0].reshape(c, -1).permute(1, 0)
    cluster_protos = compute_cluster_prototypes(
        feat_tgt_deb_flat, cluster_labels.view(-1), k)
    pred_mask = model._seed_and_aggregate(
        candidate_mask, cluster_labels, cluster_protos, k,
        prototype, feat_tgt, feat_tgt_deb, h, w)
    model._orig_tgt_size = (image.height, image.width)
    try:
        return model._finalize_mask(pred_mask, tgt[0, 0])
    finally:
        model._orig_tgt_size = None


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


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Cosmetic cleanup of the raw INSID3 mask.

    INSID3 reasons at the ViT patch level, so thin structures (e.g. the cup
    handle) come out as disconnected islands with stray specks. This bridges
    the small gaps (morphological closing) and drops only tiny specks. It
    deliberately does NOT keep only the largest connected component: that
    amputated real handle fragments during validation. Verified on the
    mock-scene masks: bounding boxes identical before/after, ~30ms cost.
    """
    closed = ndimage.binary_closing(mask, structure=np.ones((7, 7), dtype=bool))
    labeled, n = ndimage.label(closed)
    if n > 1:
        sizes = np.bincount(labeled.ravel())
        keep = np.zeros(n + 1, dtype=bool)  # index 0 = background: never kept
        keep[1:] = sizes[1:] >= max(1, int(0.001 * mask.sum()))
        return keep[labeled]
    return closed


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


if not ARGS.no_ref_cache:
    for _key, (img, msk, _desc) in REFS.items():
        t0 = time.time()
        REF_CACHES[_key] = precompute_ref(MODEL, str(_resolve(img)), str(_resolve(msk)))
        print(f"[insid3] ref-cache[{_key}] precomputed in {time.time() - t0:.1f}s",
              flush=True)


def run_grounding(image: Image.Image, target: str | None = None,
                  template_key: str | None = None):
    """Per-request path: reference lookup + INSID3 segment (cached or full).

    template_key selects a registered reference directly (used by the
    /insid3 ROS bridge); otherwise the target string is matched against
    the built-in cup/handle references (RoboBrain prompt path).
    """
    if template_key is not None:
        ref = REFS.get(template_key)
        if ref is None:
            raise ValueError(
                f"unknown INSID3 template {template_key!r} "
                f"(registered: {sorted(REFS)})")
        key = template_key
    else:
        key = ("handle" if "handle" in target.lower()
               else "cup" if "cup" in target.lower() else None)
        ref = pick_reference(target)
        if ref is None:
            raise ValueError(
                f"unsupported grounding target {target!r}: no INSID3 reference registered "
                f"(supported: targets containing 'cup' or 'handle', "
                f"or insid3_template in {sorted(REFS)})")
    ref_img, ref_mask, desc = ref
    t0 = time.time()
    with INFER_LOCK:  # serialize: one request at a time, like the C++ client's policy
        if key is not None and key in REF_CACHES:
            pred = segment_cached(MODEL, REF_CACHES[key], image)
        else:  # --no-ref-cache fallback: original set_reference/set_target/segment
            MODEL.set_reference(str(_resolve(ref_img)), str(_resolve(ref_mask)))
            MODEL.set_target(image)
            pred = MODEL.segment()
    dt = time.time() - t0
    mask = pred.detach().cpu().squeeze().numpy() > 0
    mask = clean_mask(mask)  # cosmetic: bridge patch fragments, drop specks
    boxes = mask_to_boxes(mask, image.width, image.height)
    if boxes is None:
        raise RuntimeError(f"INSID3 predicted an empty mask for target {target!r}")
    return key, desc, dt, mask, boxes


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
        template_key = payload.get("insid3_template")
        if template_key is None and not target:
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
            key, desc, infer_s, mask, (norm_box, pixel_box) = run_grounding(
                image, target, template_key)
        except ValueError as exc:  # unsupported target -> 400
            self._json(400, {"error": {"message": str(exc)}})
            return
        except Exception as exc:  # inference failure -> 500
            self._json(500, {"error": {"message": f"INSID3 inference failed: {exc}"}})
            return
        total_s = time.time() - t_req

        if DEBUG_DIR:
            tag = re.sub(r"[^a-z0-9]+", "_", (template_key or target or "").lower()).strip("_") or "target"
            stem = DEBUG_DIR / f"req{rid:03d}_{tag}"
            image.save(str(stem.with_suffix(".jpg")))
            Image.fromarray((mask.astype("uint8")) * 255).save(str(stem.with_name(stem.name + "_mask.png")))
            try:
                from utils.visualization import visualize_prediction
                ref_img_p, ref_mask_p, _ = REFS[key]
                visualize_prediction(
                    str(_resolve(ref_img_p)),
                    str(_resolve(ref_mask_p)),
                    image, mask, str(stem.with_name(stem.name + "_overlay.png")))
            except Exception as exc:
                print(f"[insid3] debug overlay failed: {exc}", flush=True)

        x1, y1, x2, y2 = norm_box
        content = f"[{x1}, {y1}, {x2}, {y2}]"
        print(f"[insid3] req={rid} target={target!r} template={template_key!r} ref={desc} "
              f"infer_s={infer_s:.2f} total_s={total_s:.2f} "
              f"box={content} mask_px={int(mask.sum())}", flush=True)
        _mask_buf = io.BytesIO()
        Image.fromarray((mask.astype("uint8")) * 255).save(_mask_buf, format="PNG")
        _mask_b64 = base64.b64encode(_mask_buf.getvalue()).decode()
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
                # Raw INSID3 mask as base64 PNG (consumed by the
                # /insid3 ROS bridge and the MCP demo transport).
                "mask_png_b64": _mask_b64,
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
