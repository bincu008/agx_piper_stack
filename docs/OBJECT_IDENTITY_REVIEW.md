# Stable object identity — integrated review

Verified 2026-11-10 in isolated mock replay. Gemini's basic association tests passed,
but additional integration regressions found gaps. These are now corrected.

## Behavior

- Case and repeated whitespace normalize for identity association.
- Explicit suffixes such as `beige cup, grasp its body` and
  `beige cup, grasp its rim` identify the canonical `beige cup`.
- The original grasp instruction remains available as metadata and still reaches
  grasp-input preparation and HGGD. Identity normalization does not turn rim into
  body or make their cached grasp plans interchangeable.
- Existing position association is retained: the default gate is 5 cm. Names alone
  do not merge distant instances. Meaningful size/color/spatial descriptors remain.
- Original descriptions are retained as bounded aliases (16 per object).
- World-state IDs remain internal. Active geometry uses the canonical name so
  changing body/rim wording does not create a second active record for that cup.
- Ordinary memory upserts preserve held/placed classification. Explicit empty-hand
  updates clear a previous held flag without claiming successful placement.

## Corrections made during review

Added regressions and fixed: ordinary memory upserts losing canonical names or
placed state; phantom held flags after clearing holding; generated IDs overwriting
explicit IDs; whitespace-only names and nonfinite/negative association radii;
and overly broad parsing of ordinary descriptors such as `cup with grasp handle`.
Fixed the docstring lint failure. Integrated canonical lookup with manipulation
memory and named camera-view lookup, while retaining the caller's grasp region in
actual grasp-calculation requests. No ROS schemas, physical gates, collision
policies or motor/gripper behavior changed for this feature.

## Verification

- Full master suite: 324 passed, 1 existing skip; lint tests pass.
- Full task_pipeline suite: 265 passed.
- Identity/memory/view subset: 88 passed.
- Authenticated MCP 3.7.0, preparation_only, mock domain89:
  capture -> pin -> resolve body -> resolve rim -> inspect memory/world.
  Body: ACTIVE_TARGET_GROUNDED, obj_001.
  Rim: ACTIVE_TARGET_REUSED, the same obj_001.
  Both active memory and world state contain one canonical beige cup.
  Typed MCP output schemas validate; source is explicitly SIMULATED/mock_replay.
  No actuator command or physical queue was called.

Evidence outside source tree: ~/agx_piper_data/reports/identity_review/{calls.json,
initial_calls.json,result.json}. Logs: identity_review_* under
~/agx_piper_data/logs/. The first verification script had a client-side assertion
assuming every diagnostic read has a top-level success field; corrected to honor
its transport envelope, then repeated verification successfully. This was not a
robot/API failure or actuator retry.

## Limits

This is semantic association, not visual tracking or proof of a grasp. Identical
nearby objects inside the association gate remain difficult to distinguish using
names and centroids alone; use distinct spatial descriptions. In-memory IDs are
not promised to survive a master restart. Previously verified full pick/place
preparation is historical evidence; physical execution and native Mac Nanobot
integration are not verified by these identity tests.
