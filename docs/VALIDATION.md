# Baseline validation

Validated on Ubuntu 24.04 ARM64 with ROS 2 Jazzy on 2026-09-27.

## Results

- Clean `colcon build --symlink-install`: passed, 15 packages.
- Clean rebuild after moving the workspace to `~/agx_piper_stack`: passed.
- AnyPlace/height-map placement tests: 20 passed.
- VAMP/HGGD/local refinement/Cartesian approach tests: 20 passed.
- Master-node behavior tests: 21 passed, 1 skipped.
- Software-only production launch: passed without camera or arm hardware.
- Production nodes/services were present and physical execution was confirmed
  disabled.

## Known non-functional test debt

`master_node/test/test_flake8.py` reports legacy formatting findings in the
large `piper_master_control.py` module. Functional tests pass. This baseline
does not apply a whole-file automatic rewrite to safety-critical motion code;
formatting should be addressed separately in reviewable commits.

The AGX driver's empty upstream unittest discovery exits with status 5 (`no
tests ran`). This is not a runtime failure.

The baseline documentation, dependency manifests, and repository metadata pass
`git diff --check`. A whole-tree check still reports inherited trailing
whitespace in upstream driver code and older planner/orchestrator sources; it
was not mechanically rewritten during this safety-focused cleanup.

## Hardware scope

The hardware was disconnected. Camera acquisition, CAN communication, and
physical motion were not tested. The launch used
`allow_physical_grasp_execution:=false`.
