# Shared spatial map — MCP 3.1.0

get_spatial_map reads ActiveVision's spatial summary. get_spatial_context adds
current TCP pose and remembered object positions/cell membership. Both are
read-only: they neither capture a frame nor move the arm. Missing observation
returns success=false, SPATIAL_OBSERVATION_REQUIRED. Header/timestamps use ROS
observation time; compare against the same ROS clock, never macOS wall time.
Stored object coordinates are hints, not proof the object is still there.

## Fixed layout

All geometry is base_link, metres. X forward [0,.6), Y left [-.6,.6), Z up [0,.6).
This is a rectangular summary box, NOT a 60cm radial reachability guarantee or
motion constraint. The fine obstacle world outside this box is unchanged.
There are 16 cubes of side .30m. Dimensions in XYZ order are [2,4,2]. Stable IDs
are x{0..1}_y{0..3}_z{0..1}. Y0 is rightmost, Y3 leftmost; Z0 lower, Z1 upper.
The base_link origin is on the layout boundary, not its minimum corner.

| Index | Bounds (metres) |
|---|---|
| X0 / X1 | [0,.3) / [.3,.6) forward |
| Y0 / Y1 / Y2 / Y3 | [-.6,-.3) / [-.3,0) / [0,.3) / [.3,.6) left |
| Z0 / Z1 | [0,.3) / [.3,.6) upward |

Bounds/centers/IDs are derived. Only last_scanned is retained per cell, in memory;
zero time means never observed. It resets on node restart/calibration reload.
No persistent per-cell cloud, occupancy flag or confidence state is added.
Latest-frame observed IDs are a derived view, not additional scan history.

## Observed empty space and occlusion

The same synchronized, calibrated FK(joints)*X base cloud that feeds perception
and fine obstacle construction supplies a bounded image-wide ray sample (about
32x24 positions). Self-masked/invalid/out-of-range depth is skipped. Each valid
ray ends at the measured surface, so every traversed cube before the hit is
observed even when empty. The endpoint cube is included, including boundary hits;
no cube behind the surface is refreshed. Rays entering from outside the box work.
Only cells reached by >=3 sampled rays update their timestamp; a single grazing
ray does not refresh a cube. Timestamps never move backwards on late samples.

This is approximate sampled FOV coverage, not complete voxel coverage. A scan
time means a qualifying portion of a cube was observed; it says nothing about
whether the entire cube is free or how many objects it contains. Foreground depth
stops the ray. Missing depth never establishes clear space. Latest observed IDs
are tied to map.header and the returned camera pose, not instantaneous live gaze.

ActiveVision owns both the fine OctoMap and this coarse index under the same
observation lock. Visibility keeps its existing synchronized fine-cloud FOV/ray
checker; next-view occlusion keeps fine OctoMap ray casting. Both use this shared
cell geometry rather than independent grid definitions. VAMP retains detailed
collision geometry; a recently observed cube never authorizes unsafe movement.

## Camera and remembered objects

get_spatial_map response map contains header, revision, cell_size_m, minimum,
maximum, dimensions, cells[], observed_cell_ids, camera_pose, camera_forward and
camera_up. Each cell has id/minimum/maximum/center/last_scanned. Camera vectors are
base_link unit vectors: optical +Z forward, optical -Y up. Camera pose is the
manual synchronized FK*X observation pose, not a TF lookup or the gripper pose.

get_spatial_context adds tcp_pose and parallel object_names/object_positions/
object_cell_ids arrays. Empty ID means outside the summary box. It uses the map's
returned bounds for membership; there is no second coordinate grid in the master.
Object-center membership does not imply full object containment. TCP feedback
and the map have independent timestamps; inspect both when reasoning about gaze.

check_target_visibility and check_target_view now return target_cell_id (empty
outside). Visibility and source matching retain their existing semantics. The
map cell alone does not establish object visibility: FOV and occlusion still do.

## Looking at a cube

plan_next_best_view accepts target_cell_id as an alternative to target_name or
explicit target PointStamped. Do not combine them. For example:

{"target_cell_id":"x1_y1_z0"}

Use this as arguments in the standard preparation envelope. The planner looks up
the cell center in get_spatial_map, then reuses bounded viewpoint sampling, fine
OctoMap occlusion, calibrated LookAt IK, VAMP path/limits and empty-hand safeguards.
Unknown cell IDs or conflicting targets fail explicitly. A cell may have no
reachable camera viewpoint; no universal reachability or automatic move is implied.
The returned endpoint_joints projection supplies the final planned joints. An
explicitly authorized guarded move_to still plans/checks its complete path.
No physical movement occurs during plan_next_best_view.

Agent sequence: read get_spatial_context → choose old/unseen cell or known object
→ plan_next_best_view → inspect successful result → operator-authorized guarded
move_to → capture_observation → read updated spatial map. Object manipulation
still requires fresh perception and complete prepared pick/place validation.

## Interpret orientation and choose a direction

Keep three different quantities separate:

| Quantity | Interpretation |
|---|---|
| tcp_pose.position | Where the gripper TCP is in its stated frame, metres. |
| tcp_pose.orientation | Gripper rotation as quaternion x,y,z,w; it is not the camera's rotation. |
| map.camera_pose / camera_forward / camera_up | Where the in-hand camera was and how it was oriented at map.header. |

Joint angles are radians describing individual joints; they are not a direct
left/right/up/down heading. Use returned camera vectors to describe gaze. Their
components use the FIXED base_link axes, regardless of wrist rotation:

| Example camera_forward | Gaze description |
|---|---|
| (1,0,0) | Forward along base +X. |
| (0,1,0) | Left along base +Y. |
| (0,-1,0) | Right along base -Y. |
| (0,0,-1) | Down along base -Z. |
| (.707,.707,0) | Forward-left. |
| (.707,0,-.707) | Forward and downward. |

These examples are unit vectors, not Cartesian targets. Interpret dominant
components rather than giving strong direction labels to tiny numerical values.
When looking vertically, horizontal heading can be ambiguous; camera_up and the
full quaternion describe camera roll. A negative X component means backward
along the base axis, not necessarily exactly toward the base origin.

For object position p and camera position c, the desired center-ray direction is
(p-c)/norm(p-c). This describes a desired view; it does not prove visibility or
reachability. Let check_target_view and the existing view planner perform their
FOV/occlusion and path checks. Never convert this vector directly into joint
commands in the outside agent.

Use observed_cell_ids to answer "which cubes were actually observed?" The forward
vector alone only describes central gaze; the FOV covers multiple cubes and rays
stop at obstacles. Objects can be visible off-center. Refresh observation after
an executed view change before reasoning about new coverage.

For relative camera movement, the input frame is DIFFERENT: optical +X right,
+Y down, +Z forward. Thus get_observe_view(dx=.02,dy=-.02,dz=0,roll=0,pitch=0,yaw=0)
requests 2cm camera-right and 2cm camera-up. It does not mean base +X/+Y motion.
The master applies the synchronized camera transform and performs IK/checking.

The agent's normal spatial workflow is:

1. Read get_spatial_context, including map/pose timestamps and object hints.
2. If the object is known, call check_target_view for that object. If its location
   is unknown, capture/perceive to populate memory first.
3. For an unseen/stale region, choose a cell from last_scanned, then plan a view
   with target_cell_id. Prefer exact object targets for manipulation.
4. Inspect success and the planned endpoint. Execute only an authorized guarded
   empty-hand move; a plan or recent cube timestamp does not authorize movement.
5. Capture again; inspect updated observed_cell_ids and last_scanned before
   grounding the source and calculating the manipulation plan.

Pick/place uses a separate tray floor-slot grid: get_floor_slots reports its
reservations. Do not pass a workspace cube ID as a placement pose or slot ID.
