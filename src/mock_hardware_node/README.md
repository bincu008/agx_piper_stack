# Mock hardware node

This package records and replays the exact hardware inputs used by the
calculation-only reasoning stack. The bundled `snapshots/current` scene
contains registered RGB-D data and Piper feedback captured together.

The default snapshot contains eight CDR messages and `manifest.json`, including
raw depth (`depth_image.cdr`, `/camera/depth/image_raw`). Older seven-message
snapshots remain supported and replay registered depth through the point cloud.
The derived `rgb_preview.png` is local inspection output and is ignored by Git.
The mock republishes recorded geometry/state with fresh timestamps; it does not
simulate physics or respond to motion commands. Camera data defaults to 10 Hz
and feedback to 50 Hz.

See the workspace [README](../../README.md) for the mock + reasoning + benchmark
launch sequence and [file map](../../docs/CODE_MAP.md) for API/node ownership.

Do not run the default replay while the real camera or arm drivers are active.
The node checks for existing publishers and refuses to start on conflicts.

Offline replay:

```bash
source install/setup.bash
ros2 launch mock_hardware_node mock_hardware.launch.py
```

Safe namespaced inspection while hardware is running:

```bash
ros2 launch mock_hardware_node mock_hardware.launch.py topic_prefix:=mock
```

Capture a replacement snapshot while both hardware drivers are running, with
mock replay stopped. Store new captures outside the installed package:

```bash
ros2 run mock_hardware_node capture_snapshot --ros-args \
  -p output_directory:="$HOME/piper_snapshots/scene_01"
```

Replay that capture with
`ros2 launch mock_hardware_node mock_hardware.launch.py snapshot_directory:="$HOME/piper_snapshots/scene_01"`.

For a source-tree capture that will be committed and installed on rebuild, use
`src/mock_hardware_node/snapshots/current` as the output directory.

Capture buffers a bounded recent history and matches RGB, raw depth and joint
feedback to a registered-cloud timestamp within 50 ms by default
(`synchronization_tolerance_sec`). The manifest records the actual timestamp
spread. Arrival-time differences are reported separately and can be larger due
to cloud processing latency. No capture/replay command moves the arm.
