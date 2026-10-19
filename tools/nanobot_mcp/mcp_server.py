#!/usr/bin/env python3
"""MCP adapter for the agx_piper_stack agent control plane.

Thin adapter only: it exposes the operations catalogued in
``task_pipeline/task_pipeline/registry.py`` plus the INSID3 grounding tools.
It implements no solvers, planners, or state of its own -- every call is
forwarded to the owning ROS 2 service (or the INSID3 HTTP server), per
docs/AGENT_CONTROL_ARCHITECTURE.md.

Transports
----------
INSID3_TRANSPORT=ros2  -> call /insid3/infer and /insid3/visualize services
                         (requires insid3_bridge + insid3_bridge_msgs built).
INSID3_TRANSPORT=http   -> talk to the INSID3 OpenAI-compatible server
                         directly (development fallback, no ROS needed).

Pipeline calls always shell out to the ``ros2`` CLI, so this server works
from any Python with the MCP package installed; ROS 2 only needs to be on
PATH when pipeline tools are actually called.

Safety
------
MCP_ALLOW_ACTUATION=1 must be set to call operations that move hardware
(execute_plan, gripper_close, execute_grasp, ...). Read-only and
calculation-only operations are always allowed. This mirrors the stack's
allow_physical_grasp_execution gate; it does not replace it.
"""

import ast
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

try:
    from mcp.server import MCPServer as FastMCP  # MCP SDK v2
except ImportError:  # pragma: no cover - legacy SDK
    from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Catalog loading (AST-parsed so importing ROS messages is not required)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = (
    REPO_ROOT / "src" / "task_pipeline" / "task_pipeline" / "registry.py"
)

# Operations that can move the arm/gripper or change held-object state.
ACTUATING_FUNCTIONS = frozenset({
    "execute_plan", "gripper_open", "gripper_close", "execute_grasp",
    "execute_prepared_grasp_lift", "execute_lift", "execute_placement_transfer",
    "execute_placement_descent", "release_placed_object", "move_to",
    "execute_trajectory", "set_gripper", "move_pregrasp",
    "execute_grasp_approach", "close_grasp_gripper", "reconcile_empty_hand",
})


def load_catalog():
    """Return {name: (service_type, endpoint)} from registry.py via AST."""
    tree = ast.parse(REGISTRY_PATH.read_text())
    catalog = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "Function"):
            continue
        try:
            name = ast.literal_eval(node.args[0])
            svc = ast.unparse(node.args[1]) if len(node.args) > 1 else ""
            endpoint = ast.literal_eval(node.args[2]) if len(node.args) > 2 else ""
        except Exception:
            continue
        if isinstance(name, str) and endpoint:
            # '/pipeline/<name>' public endpoints map to their backend;
            # direct engine endpoints are used as-is.
            catalog[name] = (svc, endpoint)
    # List-comprehension entries (prepare_pick_place etc.) use '/master/'+name.
    for extra in ("prepare_pick_place", "get_prepared_pick_place",
                  "validate_prepared_pick_place", "execute_lift",
                  "execute_placement_transfer", "execute_placement_descent",
                  "release_placed_object", "resolve_target",
                  "refresh_active_target", "forget_active_target",
                  "get_active_target"):
        catalog.setdefault(extra, ("srv.PreparePickPlace", f"/master/{extra}"))
    return catalog


CATALOG = load_catalog()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(os.environ.get(
    "MCP_RESULTS_DIR", str(Path.home() / "agx_piper_data" / "mcp_results")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CALL_LOG = RESULTS_DIR / "mcp_calls.log"

INSID3_TRANSPORT = os.environ.get("INSID3_TRANSPORT", "http").lower()
INSID3_URL = os.environ.get("INSID3_URL", "http://127.0.0.1:18092")
INSID3_TEMPLATE_DIR = os.environ.get("INSID3_TEMPLATE_DIR", "")
ALLOW_ACTUATION = os.environ.get("MCP_ALLOW_ACTUATION", "0") == "1"
# MCP_TOOLSET=insid3  -> only the INSID3 grounding tools (lean agent prompt
#                        for the VLM-node grounding path).
# MCP_TOOLSET=all     -> INSID3 + pipeline catalog tools (default).
MCP_TOOLSET = os.environ.get("MCP_TOOLSET", "all").lower()


def log_call(tool, elapsed_s, ok, extra=""):
    with open(CALL_LOG, "a") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tool": tool, "elapsed_s": round(elapsed_s, 3), "ok": ok,
            "transport": INSID3_TRANSPORT, "extra": extra[:200],
        }) + "\n")


def resolve_image(image_path="", image_data_url=""):
    """Return a local image path, materialising a data URL if needed."""
    if image_path and Path(image_path).exists():
        return image_path
    if image_data_url.startswith("data:image"):
        header, _, b64 = image_data_url.partition(",")
        ext = ".png" if "png" in header else ".jpg"
        tmp = tempfile.NamedTemporaryFile(
            suffix=ext, prefix="mcp_img_", dir=str(RESULTS_DIR),
            delete=False)
        tmp.write(base64.b64decode(b64))
        tmp.close()
        return tmp.name
    raise FileNotFoundError(
        f"image not found: path={image_path!r} "
        f"(data_url={'given' if image_data_url else 'missing'})")


def insid3_http(template_name, image_path):
    img = open(image_path, "rb").read()
    data_url = "data:image/png;base64," + base64.b64encode(img).decode()
    payload = {"model": "insid3-grounding", "insid3_template": template_name,
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": data_url}},
                   {"type": "text", "text": (
                       "Please provide the bounding box coordinate of the "
                       f"region this sentence describes: {template_name}. "
                       "Return only [x1, y1, x2, y2] in normalized 0-1000 "
                       "coordinates.")}]}]}
    t0 = time.time()
    req = urllib.request.Request(
        INSID3_URL + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.load(r)
    total = time.time() - t0
    detail = body.get("insid3", {})
    mask_png = base64.b64decode(detail["mask_png_b64"])
    mask_path = str(RESULTS_DIR / (
        f"mask_{template_name}_{time.strftime('%Y%m%d_%H%M%S')}.png"))
    with open(mask_path, "wb") as f:
        f.write(mask_png)
    return {"box_text": body["choices"][0]["message"]["content"],
            "detail": detail, "mask_path": mask_path,
            "service_total_s": total}


def ros2_service_call(service, srv_type, args_json, timeout=600):
    """Thin forward to `ros2 service call`. No local reimplementation."""
    ros2 = shutil.which("ros2")
    if not ros2:
        return {"ok": False,
                "error": "ros2 CLI not on PATH; cannot reach ROS services"}
    # 'srv.X' -> package-qualified type for the CLI.
    cli_type = srv_type.replace("srv.", "my_actions/srv/", 1) \
        if srv_type.startswith("srv.") else srv_type
    cmd = [ros2, "service", "call", service, cli_type, args_json or "{}"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"service call timed out ({timeout}s)"}
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip()[-500:]}
    return {"ok": True, "raw": proc.stdout.strip()[-4000:]}


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

mcp = FastMCP("agx-piper-mcp")


@mcp.tool()
def insid3_list_templates() -> str:
    """List INSID3 template names."""
    t0 = time.time()
    try:
        if INSID3_TEMPLATE_DIR:
            manifest = Path(INSID3_TEMPLATE_DIR) / "manifest.json"
            names = [t["name"] for t in json.loads(manifest.read_text())["templates"]]
        else:
            names = ["cup", "handle"]  # server built-in references
        out = {"success": True, "templates": names}
        log_call("insid3_list_templates", time.time() - t0, True)
    except Exception as e:  # noqa: BLE001
        out = {"success": False, "message": str(e)}
        log_call("insid3_list_templates", time.time() - t0, False, str(e))
    return json.dumps(out)


@mcp.tool()
def insid3_infer(image_path: str = "", image_data_url: str = "",
                 template_name: str = "") -> str:
    """INSID3 segment image by template. Args: image_path or image_data_url,
    template_name (from insid3_list_templates). Returns mask_path,
    bbox_xyxy_pixels, bbox_normalized_1000."""
    t0 = time.time()
    try:
        if not template_name:
            raise ValueError("template_name is required")
        local_path = resolve_image(image_path, image_data_url)
        if INSID3_TRANSPORT == "ros2":
            from PIL import Image as PILImage  # lazy: only needed here
            img = PILImage.open(local_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            # Route through the /insid3/infer ROS service via CLI.
            payload = {"template_name": template_name,
                       "image_png_b64": base64.b64encode(buf.getvalue()).decode()}
            res = ros2_service_call("/insid3/infer",
                                    "insid3_bridge_msgs/srv/Insid3Infer",
                                    json.dumps(payload))
            if not res["ok"]:
                raise RuntimeError(res["error"])
            out = {"success": True, "transport": "ros2", "raw": res["raw"]}
        else:
            r = insid3_http(template_name, local_path)
            detail = r["detail"]
            # Normalized 0-1000 box for the VLM-node grounding contract.
            box_text = r["box_text"]
            import re
            nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", box_text)
            norm = [round(float(x)) for x in nums[:4]] if len(nums) >= 4 else []
            try:
                from PIL import Image as PILImage
                w, h = PILImage.open(local_path).size
            except Exception:  # noqa: BLE001
                w, h = 0, 0
            out = {
                "success": True,
                "template": template_name,
                "mask_path": r["mask_path"],
                "bbox_xyxy_pixels": detail.get("bbox_xyxy", []),
                "bbox_normalized_1000": norm,
                "image_size_wh": [w, h],
                "mask_pixels": detail.get("mask_pixels", 0),
                "infer_time_s": round(detail.get("inference_seconds",
                                                 detail.get("infer_s", 0.0)), 3),
                "service_total_s": round(r["service_total_s"], 3),
                "message": (f"template={template_name} "
                            f"(transport=http direct)"),
            }
        log_call("insid3_infer", time.time() - t0, True, template_name)
    except Exception as e:  # noqa: BLE001
        out = {"success": False, "message": str(e)}
        log_call("insid3_infer", time.time() - t0, False, str(e))
    return json.dumps(out)


@mcp.tool()
def insid3_visualize(image_path: str = "", image_data_url: str = "",
                     mask_path: str = "") -> str:
    """Draw mask_path onto image_path (or image_data_url). Returns overlay_path."""
    t0 = time.time()
    try:
        local_path = resolve_image(image_path, image_data_url)
        if not mask_path or not Path(mask_path).exists():
            raise FileNotFoundError(f"mask not found: {mask_path!r}")
        from PIL import Image as PILImage
        img = PILImage.open(local_path).convert("RGB")
        mask = PILImage.open(mask_path).convert("L").resize(img.size)
        overlay = img.copy()
        green = PILImage.new("RGB", img.size, (0, 255, 0))
        overlay.paste(green, mask=mask.point(lambda v: v // 3))
        out_path = str(RESULTS_DIR / (
            f"overlay_{time.strftime('%Y%m%d_%H%M%S')}.png"))
        overlay.save(out_path)
        out = {"success": True, "overlay_path": out_path,
               "service_total_s": round(time.time() - t0, 3)}
        log_call("insid3_visualize", time.time() - t0, True)
    except Exception as e:  # noqa: BLE001
        out = {"success": False, "message": str(e)}
        log_call("insid3_visualize", time.time() - t0, False, str(e))
    return json.dumps(out)


# Pipeline catalog tools are omitted in the lean "insid3" toolset so the
# grounding agent's prompt stays small (tool schemas dominate prefill time
# on CPU-hosted models).
if MCP_TOOLSET != "insid3":

    @mcp.tool()
    def pipeline_list_functions() -> str:
        """List the catalogued public pipeline operations (from registry.py)
        with their service types, endpoints, and whether they can actuate."""
        t0 = time.time()
        funcs = [
            {"name": n,
             "service_type": s,
             "endpoint": e,
             "actuating": n in ACTUATING_FUNCTIONS}
            for n, (s, e) in sorted(CATALOG.items())
        ]
        log_call("pipeline_list_functions", time.time() - t0, True)
        return json.dumps({"success": True, "count": len(funcs),
                           "functions": funcs})


    @mcp.tool()
    def pipeline_call(function_name: str, args_json: str = "{}",
                      timeout_sec: int = 300) -> str:
        """Call one catalogued pipeline operation by name. args_json holds the
        service request fields as JSON. Only names from pipeline_list_functions
        are accepted; anything else is rejected. Actuating operations
        (execute_plan, gripper_close, ...) require MCP_ALLOW_ACTUATION=1."""
        t0 = time.time()
        try:
            if function_name not in CATALOG:
                raise ValueError(
                    f"unknown function {function_name!r}; not in the catalog")
            if function_name in ACTUATING_FUNCTIONS and not ALLOW_ACTUATION:
                raise PermissionError(
                    f"{function_name} can actuate hardware; set "
                    "MCP_ALLOW_ACTUATION=1 to enable (mirrors "
                    "allow_physical_grasp_execution)")
            svc_type, backend = CATALOG[function_name]
            # The canonical public contract is /pipeline/<name>; pipeline_node
            # dispatches to the catalogued backend. Never call an arbitrary
            # endpoint supplied by the caller.
            service = f"/pipeline/{function_name}"
            res = ros2_service_call(service, svc_type, args_json,
                                    timeout=timeout_sec)
            out = {"success": res["ok"], **res,
                   "elapsed_s": round(time.time() - t0, 3)}
            log_call("pipeline_call", time.time() - t0, res["ok"], function_name)
        except Exception as e:  # noqa: BLE001
            out = {"success": False, "message": f"{type(e).__name__}: {e}"}
            log_call("pipeline_call", time.time() - t0, False, str(e))
        return json.dumps(out)


def main():
    n_tools = 3 + (2 if MCP_TOOLSET != "insid3" else 0)
    print(f"[agx-piper-mcp] toolset={MCP_TOOLSET} tools={n_tools} "
          f"catalogued functions: {len(CATALOG)}; "
          f"results -> {RESULTS_DIR}; actuation "
          f"{'ENABLED' if ALLOW_ACTUATION else 'disabled'}",
          file=sys.stderr, flush=True)
    mcp.run()


if __name__ == "__main__":
    main()
