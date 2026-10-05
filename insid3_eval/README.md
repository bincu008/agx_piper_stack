# INSID3 grounding evaluation

Experimental replacement of the RoboBrain VLM grounding endpoint with
**INSID3** (training-free in-context segmentation on a frozen DINOv3
encoder), aimed at edge deployment: the VLM's only job in this stack is
object grounding/locating, and INSID3 does that with an 82-327MB encoder
instead of a 4B-parameter VLM.

## Layout

- `server/insid3_server.py` -- OpenAI-compatible HTTP grounding server
  (see `server/README.md` for API, target mapping, flags).
- `ref/` -- reference image + masks: `brown_cup.jpg`,
  `brown_cup_mask_whole.png` (whole cup), `brown_cup_mask_handle.png`
  (handle only).
- `eval/run_eval.py` -- end-to-end eval: starts the server, sends the exact
  grounding requests `vlm_reasoning_node` would send for the mock snapshot
  RGB, records timings/boxes/masks.
- `results/` -- committed eval outputs: per-case overlay PNGs, raw masks,
  `TIMING.md`, `report.json`, server stdout log.

## Pointing the stack at INSID3

The `insid3` inference profile was added to
`src/master_node/config/inference/profiles.yaml`
(`api_server_url` -> `http://127.0.0.1:18092/v1`). Production defaults are
untouched (`server` profile still default). Rebuild `master_node` after
pulling so the installed `profiles.yaml` picks up the new profile
(`colcon build --packages-select master_node`), then:

```bash
# terminal 1: start the INSID3 server (loads DINOv3 once, ~11s on CPU)
insid3_eval/server/insid3_server.py --port 18092

# terminal 2: launch the stack against it (mock hardware, no physical motion)
ros2 launch master_node reasoning_stack.launch.py \
  inference_profile:=insid3 allow_physical_grasp_execution:=false
```

Explicit `api_server_url:=...` launch args still override the profile, as
before.

## Reproducing the eval

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 insid3_eval/eval/run_eval.py
```

The script needs the INSID3 checkout at `~/workspace/insid3`
(`INSID3_DIR` env override) and its virtualenv at
`~/workspace/insid3-venv`. It runs 3 requests per case against the mock
snapshot RGB and writes fresh results into `results/`.

## Status

Evaluation branch work (`muse/insid3-eval`); not production. See
`results/TIMING.md` for the numbers vs the RoboBrain 21.248s baseline.
