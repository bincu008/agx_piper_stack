std_msgs/Header header
uint64 revision
float32 cell_size_m
geometry_msgs/Point minimum
geometry_msgs/Point maximum
uint32[3] dimensions
my_actions/SpatialCell[] cells
# Cells traversed by >=3 valid sampled depth rays in the latest observation.
string[] observed_cell_ids
geometry_msgs/PoseStamped camera_pose
geometry_msgs/Vector3 camera_forward
geometry_msgs/Vector3 camera_up
