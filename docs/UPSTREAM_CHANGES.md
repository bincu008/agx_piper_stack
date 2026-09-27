# Upstream integration record

## AGX arm ROS driver

The active driver snapshot is retained. Local functional changes include:

- runtime speed-percentage updates;
- 50 Hz feedback support;
- TCP pose publication in `base_link`;
- orchestration moved out of the driver and into `master_node`.

Driver-local YOLO weights and the obsolete duplicate master controller were
removed. The populated AGX URDF submodule was flattened into the baseline so a
fresh clone contains the Piper description and meshes.

## Orbbec ROS 2 driver

The known-working 2.8.6 snapshot is retained. The Gemini 330 launch defaults
were changed to 640x480 RGB and registered depth at 15 Hz, ordered colored
point clouds, frame synchronization, and optional composition. These changes
are required by the organized RGB-D grasp pipeline.

## VAMP MVT

The library is retained separately under `src/vamp_mvt`. Required changes are:

- generated Piper FK and collision-sphere model;
- denser hierarchy/voxel-pool allocation for the observed obstacle world;
- workspace grid width derived from all three dimensions;
- Piper module registration.

Demo-only Orbbec coupling, PLY dumps, experimental generated Piper variants,
and downloaded build trees were removed.
