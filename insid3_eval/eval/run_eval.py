#!/usr/bin/env python3
"""End-to-end evaluation of the INSID3 grounding server against mock snapshot RGB.

Runs the real HTTP server (insid3_eval/server/insid3_server.py), sends it
OpenAI-style grounding requests built exactly like vlm_reasoning_node does
(image-first content array, data-URL JPEG, standard grounding prompt), and
records timings/boxes/masks for two cases:

    case A: target "the cup"    -> whole-cup reference mask
    case B: target "cup handle" -> handle-only reference mask

The RGB frame comes from the mock_hardware_node snapshot
(src/mock_hardware_node/snapshots/current/rgb_image.cdr) via snapshot_io --
no ROS graph is launched.

Run with the ROS workspace sourced (for snapshot_io):

    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    python3 insid3_eval/eval/run_eval.py

Outputs (under insid3_eval/results/):
    result_caseA_whole_cup.png / result_caseB_handle.png  (RGB | RGB+mask overlay)
    mask_caseA_whole_cup.png   / mask_caseB_handle.png    (raw predicted masks)
    TIMING.md  (per-run timings, boxes, mask pixel counts, RoboBrain comparison)
"""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "insid3_eval" / "results"
SERVER = REPO / "insid3_eval" / "server" / "insid3_server.py"
VENV_PY = Path.home() / "workspace" / "insid3-venv" / "bin" / "python"
SNAPSHOT = REPO / "src" / "mock_hardware_node" / "snapshots" / "current"

PORT = 18092
RUNS = 3
ROBOBRAIN_BASELINE_S = 21.248  # docs/ROBOBRAIN_GROUNDING.md, 'the cup', 640x480

CASES = [
    ("the cup", "caseA_whole_cup"),
    ("cup handle", "caseB_handle"),
]

PROMPT = ("Please provide the bounding box coordinate of the region this sentence "
         "describes: {target}. Return only [x1, y1, x2, y2] in normalized 0-1000 "
         "coordinates. Cover the full visible extent of the object, including its "
         "handle if present.")
BOX_RE = re.compile(r"^\s*\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]\s*$")


def wait_for_health(timeout: float = 180.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(2.0)
    raise RuntimeError("server did not become healthy in time")


def post_grounding(jpeg_b64: str, target: str) -> tuple[float, dict]:
    payload = {
        "model": "insid3-grounding",
        "temperature": 0,
        "seed": 1234,
        "max_tokens": 48,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64}},
                {"type": "text", "text": PROMPT.format(target=target)},
            ],
        }],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    return time.time() - t0, body


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # --- snapshot RGB ---
    from mock_hardware_node.snapshot_io import load_messages
    messages, _ = load_messages(SNAPSHOT)
    img = messages["rgb"]
    import numpy as np
    rgb = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, 3)
    print(f"[eval] snapshot RGB: {img.width}x{img.height} {img.encoding}", flush=True)
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=92)
    jpeg_b64 = base64.b64encode(buf.getvalue()).decode()

    # --- start server ---
    debug_dir = Path(tempfile.mkdtemp(prefix="insid3_debug_"))
    log_path = RESULTS / "server_stdout.log"
    log_f = open(log_path, "w")
    server = subprocess.Popen(
        [str(VENV_PY), str(SERVER), "--port", str(PORT),
         "--backbone", "vit-s", "--image-size", "1024", "--tau", "0.9",
         "--debug-dir", str(debug_dir)],
        stdout=log_f, stderr=subprocess.STDOUT, text=True)
    try:
        wait_for_health()
        print("[eval] server healthy", flush=True)

        report = {"cases": {}}
        for target, label in CASES:
            runs = []
            for i in range(RUNS):
                wall_s, body = post_grounding(jpeg_b64, target)
                content = body["choices"][0]["message"]["content"]
                m = BOX_RE.match(content)
                assert m, f"server returned unparseable box: {content!r}"
                box = [int(m.group(j)) for j in range(1, 5)]
                extra = body.get("insid3", {})
                runs.append({
                    "wall_s": round(wall_s, 3),
                    "inference_s": extra.get("inference_seconds"),
                    "normalized_box": box,
                    "pixel_box": extra.get("pixel_box"),
                    "mask_pixels": extra.get("mask_pixels"),
                    "request_id": extra.get("request_id"),
                })
                print(f"[eval] {label} run{i + 1}: wall={wall_s:.2f}s "
                      f"infer={extra.get('inference_seconds')}s box={box} "
                      f"mask_px={extra.get('mask_pixels')}", flush=True)
            report["cases"][label] = {"target": target, "runs": runs}

            # --- committed artifacts from the last run's debug files ---
            rid = runs[-1]["request_id"]
            tag = re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_")
            dbg_mask = debug_dir / f"req{rid:03d}_{tag}_mask.png"
            dbg_target = debug_dir / f"req{rid:03d}_{tag}.jpg"
            mask_png = Image.open(dbg_mask).convert("L")
            mask_png.save(RESULTS / f"mask_{label}.png")
            tgt = Image.open(dbg_target).convert("RGB")

            # inline of insid3.utils.visualization._overlay_mask (that module
            # imports torch, which only exists in the INSID3 venv; the eval
            # itself runs under system python with ROS sourced)
            def _overlay_mask(image, mask, color, alpha):
                overlay = image.astype(np.float32).copy()
                color_arr = np.array(color, dtype=np.float32) * 255.0
                overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr
                return np.clip(overlay, 0, 255).astype(np.uint8)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            overlay = _overlay_mask(np.array(tgt),
                                    np.array(mask_png) > 0,
                                    color=(0.15, 0.8, 0.35), alpha=0.45)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            axes[0].imshow(np.array(tgt))
            axes[0].set_title("RGB (mock snapshot)")
            axes[1].imshow(overlay)
            axes[1].set_title(f"Target + Prediction ({target})")
            for ax in axes:
                ax.axis("off")
            fig.savefig(RESULTS / f"result_{label}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    finally:
        server.terminate()
        server.wait(timeout=30)
        log_f.close()

    # --- model load time from server log ---
    load_s = None
    m = re.search(r"model loaded in ([\d.]+)s", log_path.read_text())
    if m:
        load_s = float(m.group(1))
    report["model_load_s"] = load_s

    # --- TIMING.md ---
    lines = ["# INSID3 grounding eval — timing & accuracy",
             "",
             f"Backbone: ViT-S (DINOv3, 82MB) @ image_size=1024, tau=0.9, CPU-only. "
             f"Model load: {load_s:.1f}s." if load_s else
             "Backbone: ViT-S @ 1024, tau=0.9, CPU-only.",
             f"Target image: mock snapshot RGB, 640x480. {RUNS} runs per case.",
             "",
             "Config rationale: ViT-B@512/tau=0.6 was tried first but merged the",
             "white plate into the cup mask and could not isolate the handle;",
             "ViT-S@1024/tau=0.9 segments the cup cleanly and isolates the handle.",
             "",
             "| case | target | run | client wall (s) | server infer (s) | "
             "normalized box | mask px |",
             "|---|---|---|---|---|---|---|"]
    for _target, label in CASES:
        for i, r in enumerate(report["cases"][label]["runs"], 1):
            lines.append(
                f"| {label} | {report['cases'][label]['target']} | {i} | "
                f"{r['wall_s']:.2f} | {r['inference_s']:.2f} | {r['normalized_box']} | "
                f"{r['mask_pixels']} |")
    lines += [
        "",
        "## vs RoboBrain baseline",
        "",
        f"RoboBrain 4B grounding for 'the cup' on the same 640x480 snapshot: "
        f"**{ROBOBRAIN_BASELINE_S:.3f}s** complete-response latency "
        f"(docs/ROBOBRAIN_GROUNDING.md).",
        "",
    ]
    for _target, label in CASES:
        walls = [r["wall_s"] for r in report["cases"][label]["runs"]]
        mean_w = sum(walls) / len(walls)
        lines.append(
            f"- {label} ('{report['cases'][label]['target']}'): mean client wall "
            f"{mean_w:.2f}s vs RoboBrain {ROBOBRAIN_BASELINE_S:.3f}s "
            f"-> **{ROBOBRAIN_BASELINE_S / mean_w:.1f}x faster**.")
    lines += ["",
              "## Notes",
              "",
              "- Server inference is serialized (one request at a time).",
              "- Boxes are the predicted mask's bounding box in normalized 0-1000 coords,",
              "  the exact format vlm_reasoning_node's regex accepts.",
              "- Raw masks: mask_caseA_whole_cup.png, mask_caseB_handle.png.",
              ""]
    (RESULTS / "TIMING.md").write_text("\n".join(lines))
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2))
    print("[eval] wrote results to", RESULTS, flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
