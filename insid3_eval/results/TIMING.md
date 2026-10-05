# INSID3 grounding eval — timing & accuracy

Backbone: ViT-S (DINOv3, 82MB) @ image_size=1024, tau=0.9, CPU-only. Model load: 15.1s.
Target image: mock snapshot RGB, 640x480. 3 runs per case.

Config rationale: ViT-B@512/tau=0.6 was tried first but merged the
white plate into the cup mask and could not isolate the handle;
ViT-S@1024/tau=0.9 segments the cup cleanly and isolates the handle.

| case | target | run | client wall (s) | server infer (s) | normalized box | mask px |
|---|---|---|---|---|---|---|
| caseA_whole_cup | the cup | 1 | 31.12 | 28.46 | [203, 15, 405, 279] | 11178 |
| caseA_whole_cup | the cup | 2 | 29.01 | 27.85 | [203, 15, 405, 279] | 11178 |
| caseA_whole_cup | the cup | 3 | 29.13 | 28.01 | [203, 15, 405, 279] | 11178 |
| caseB_handle | cup handle | 1 | 28.90 | 27.80 | [344, 77, 405, 200] | 940 |
| caseB_handle | cup handle | 2 | 29.70 | 28.61 | [344, 77, 405, 200] | 940 |
| caseB_handle | cup handle | 3 | 29.67 | 28.27 | [344, 77, 405, 200] | 940 |

## vs RoboBrain baseline

RoboBrain 4B grounding for 'the cup' on the same 640x480 snapshot: **21.248s** complete-response latency (docs/ROBOBRAIN_GROUNDING.md).

- caseA_whole_cup ('the cup'): mean client wall 29.76s vs RoboBrain 21.248s -> **0.7x faster**.
- caseB_handle ('cup handle'): mean client wall 29.42s vs RoboBrain 21.248s -> **0.7x faster**.

## Notes

- Server inference is serialized (one request at a time).
- Boxes are the predicted mask's bounding box in normalized 0-1000 coords,
  the exact format vlm_reasoning_node's regex accepts.
- Raw masks: mask_caseA_whole_cup.png, mask_caseB_handle.png.
