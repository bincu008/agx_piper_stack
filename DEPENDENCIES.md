# Dependencies

## Supported environment

- Ubuntu 24.04 ARM64
- ROS 2 Jazzy
- Python 3.12
- C++17 compiler

Install the packages in `apt-packages.txt`. `rosdep-packages.txt` records the
direct rosdep keys declared by this workspace; let rosdep resolve them for the
target ROS distribution:

```bash
sudo rosdep init  # once per machine
rosdep update
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install --user -r python-requirements.txt
```

The deployed grasp engine is HGGD; `vamp_node` no longer depends on GPD.
The benchmark retains internal known-cloud geometry proposals.

## ONNX Runtime

HGGD, EdgeSAM, and optional YOLO inference require the ONNX Runtime C/C++
library. Install an ARM64 CPU build and set its prefix before building:

```bash
export ONNXRUNTIME_ROOT="$HOME/.local/onnxruntime-1.15.1"
```

The prefix must contain `include/onnxruntime_cxx_api.h` and
`lib/libonnxruntime.so`. The current stack was validated with 1.15.1.

## Assets included in the repository

- `src/master_node/config/calibration/hand_eye_calibration.yaml`
- `src/vamp_node/models/hggd_mcu/*.onnx`
- `src/vlm_node/models/edgesam/*.onnx`
- `src/vision_node/models/yolo26n-seg.onnx`
- Piper URDF and meshes under `src/agx_arm_ros`
- Orbbec SDK 2.8.6 libraries for ARM64 and x86-64
- Piper VAMP FK/sphere model in `src/vamp_mvt`

The hand-eye calibration is specific to the current camera bracket and robot.
Recalibrate after changing either mounting geometry or hardware.

## External runtime services

RoboBrain is not stored in this repository. Start an OpenAI-compatible server
that exposes the configured model, then pass its address to
`reasoning_stack.launch.py`.

The AnyPlace checkpoint is optional and is not used by the default height-map
placement backend. Install it only when testing `placement_backend:=anyplace`:

```bash
ros2 run placement_node download_anyplace_checkpoint
```
