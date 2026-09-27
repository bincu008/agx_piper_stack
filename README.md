# AGX Piper Stack

ROS 2 workspace for wrist-camera perception, RoboBrain grounding, EdgeSAM
segmentation, HGGD grasp generation, VAMP motion/collision planning, placement
planning, and AGX Piper control.

The supported platform is Ubuntu 24.04 ARM64 with ROS 2 Jazzy. The repository
contains the required robot descriptions, camera SDK, calibration, HGGD
weights, EdgeSAM weights, and optional YOLO segmentation weight. RoboBrain is
provided by an external OpenAI-compatible inference server.

## Build

```bash
cd ~/agx_piper_stack
source /opt/ros/jazzy/setup.bash
export ONNXRUNTIME_ROOT="$HOME/.local/onnxruntime-1.15.1"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

See [DEPENDENCIES.md](DEPENDENCIES.md) for non-ROS dependencies and
[docs/CODE_MAP.md](docs/CODE_MAP.md) for architecture, launch commands, data
flow, and debugging entry points. The last software-only verification is
recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

## Safe software-only launch

This starts planning/perception components without authorizing physical grasp
execution. Camera and arm topics may be absent.

```bash
ros2 launch master_node reasoning_stack.launch.py \
  api_server_ip:=127.0.0.1 \
  api_server_port:=1234 \
  model_name:=robobrain2.5-4b \
  grasp_candidate_backend:=hggd \
  placement_backend:=heightmap \
  allow_physical_grasp_execution:=false
```

Do not enable physical execution until the hardware, calibration, obstacle
world, emergency stop, and candidate visualization have been reviewed.
