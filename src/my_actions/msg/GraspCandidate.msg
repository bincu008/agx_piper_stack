uint32 id
geometry_msgs/PoseStamped grasp_pose
geometry_msgs/PoseStamped pregrasp_pose
geometry_msgs/Point contact_left
geometry_msgs/Point contact_right
float32 jaw_opening_m
# Backend quality. For analytical proposals this is a deterministic geometry
# quality; for legacy GPD proposals it is the unmodified classifier output.
float32 score
# Composite 0-based ordering score used to choose between IK-valid candidates.
float32 ranking_score
# [proposal_quality, proposal_quality_normalized, wide_opening_penalty,
#  vertical_closing_penalty, seed_distance_penalty,
#  vertical_approach_penalty, joint_travel_penalty, joint_margin_reward,
#  pregrasp_override_penalty]
float32[] score_components
bool ik_valid
# True when the VAMP grasp validator found two local contact surfaces and the
# commanded finger body/closing sweep was clear.
bool measured_contacts_valid
# True when the contact pair came from two registered RGB-D silhouette edges.
bool silhouette_contacts_valid
# True only when both endpoint tests ran and passed.  Kept for compatibility.
bool collision_valid
bool collision_checked
bool grasp_collision_valid
bool pregrasp_collision_valid
bool pregrasp_collision_overridden
string validation_status
# PINCH, POWER_GRASP, or LOCAL_GENERIC: geometry-selected manipulation skill.
string affordance_family
float32[6] grasp_joints
float32[6] pregrasp_joints
geometry_msgs/Point[] projected_geometry
