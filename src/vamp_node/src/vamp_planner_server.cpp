#include "vamp_node/pipeline_timing.hpp"
#include "vamp_node/payload_checked_robot.hpp"
#include "vamp_node/upright_motion.hpp"
#include "vamp_node/foreground_components.hpp"
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <vector>
#include <array>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cctype>
#include <mutex>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>
#include <optional>
#include <filesystem>
#include <fstream>
#include <queue>
#include <deque>
#include <sstream>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "vamp_node/hggd_mcu_adapter.hpp"
#include "vamp_node/grasp_diversity.hpp"
#include "vamp_node/rim_seed.hpp"
#include "vamp_node/piper_finger_geometry.hpp"
#include "vamp_node/known_cloud_grasp_generator.hpp"
#include "vamp_node/planar_body_mask.hpp"
#include "vamp_node/deterministic_ik.hpp"
#include "vamp_node/cartesian_grasp_approach.hpp"
#include "vamp_node/grasp_position_offset.hpp"
#include "vamp_node/shallow_grasp.hpp"
#include "vamp_node/payload_escape.hpp"
#include "vamp_node/contact_surface_alignment.hpp"
#include "vamp_node/local_grasp_refiner.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_srvs/srv/trigger.hpp"

#include "my_actions/srv/check_and_replan.hpp"
#include "my_actions/srv/toggle_continuous_navigation.hpp"
#include "my_actions/srv/calculate_fallback_pose.hpp"
#include "my_actions/srv/plan_apf.hpp"
#include "my_actions/srv/look_at.hpp"
#include "my_actions/srv/find_grasp_candidates.hpp"
#include "my_actions/srv/find_cloud_grasps.hpp"
#include "my_actions/srv/refine_grasp.hpp"
#include "my_actions/srv/plan_cartesian_grasp_approach.hpp"
#include "my_actions/srv/plan_placement.hpp"
#include "my_actions/srv/pin_layout_observation.hpp"
#include "vamp_node/placement_ranking.hpp"
#include "vamp_node/placement_geometry.hpp"
#include "vamp_node/layout_scene_cache.hpp"
#include "vamp_node/organized_xyz_reader.hpp"
#include "vamp_node/placement_point_index.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <vamp/collision/factory.hh>
#include <vamp/collision/filter_centervox.hh>
#include <vamp/robots/piper.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/planning/simplify.hh>
#include <vamp/random/halton.hh>

using Point = std::array<float, 3>;

// Cache only the geometric broad phase, never a collision/IK result. Compare
// complete point bytes on every lookup; stamps or pointer identity cannot
// establish scene equality. Two entries per worker bound retained memory.
static const vamp_node::PlacementPointIndex &exactCollisionIndex(const std::vector<Point> &points)
{
  static thread_local vamp_node::ExactPointIndexCache cache;
  return cache.get(points);
}
struct VoxelHash {
  std::size_t operator()(const std::tuple<int, int, int> &key) const {
    return std::hash<int>{}(std::get<0>(key)) ^
           (std::hash<int>{}(std::get<1>(key)) << 1) ^
           (std::hash<int>{}(std::get<2>(key)) << 2);
  }
};
using Robot = vamp::robots::Piper;
static constexpr const std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;
using RRTC = vamp::planning::RRTC<Robot, rake, Robot::resolution>;

using PayloadPiper = vamp_node::PayloadCheckedRobot<Robot>;

namespace vamp_node
{

class VampPlannerServer : public rclcpp::Node
{
private:
  struct AntipodalContacts {
    Eigen::Vector3f left = Eigen::Vector3f::Zero();
    Eigen::Vector3f right = Eigen::Vector3f::Zero();
    Eigen::Vector3f closing_axis = Eigen::Vector3f::UnitY();
    float opening = 0.0f;
    float quality = std::numeric_limits<float>::infinity();
    bool silhouette_supported = false;
    bool body_model_supported = false;
    LocalGraspSkill grasp_skill = LocalGraspSkill::kUnknown;
  };

  struct PoseIkDiagnostics {
    int attempts = 0;
    int collision_rejections = 0;
    float best_position_error = std::numeric_limits<float>::infinity();
    float best_rotation_error = std::numeric_limits<float>::infinity();

    void keepBest(const PoseIkDiagnostics &other) {
      attempts += other.attempts;
      collision_rejections += other.collision_rejections;
      const float current_cost = best_position_error + 0.05F * best_rotation_error;
      const float other_cost = other.best_position_error + 0.05F * other.best_rotation_error;
      if (other_cost < current_cost) {
        best_position_error = other.best_position_error;
        best_rotation_error = other.best_rotation_error;
      }
    }
  };

  std::mutex cloud_mutex_;
  std::vector<Point> obstacle_cloud_;
  sensor_msgs::msg::PointCloud2::SharedPtr latest_registered_cloud_;
  std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> registered_cloud_cache_;
  sensor_msgs::msg::Image::SharedPtr latest_observation_image_;
  std::deque<sensor_msgs::msg::Image::SharedPtr> observation_image_cache_;
  geometry_msgs::msg::TransformStamped::SharedPtr latest_observation_camera_transform_;
  std::deque<geometry_msgs::msg::TransformStamped::SharedPtr>
    observation_camera_transform_cache_;
  sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;
  size_t grasp_observation_cache_size_ = 1000;
  std::vector<float> latest_joints_;
  // One bounded bundle retains the exact request frame across slow VLM calls.
  sensor_msgs::msg::PointCloud2::SharedPtr pinned_layout_cloud_;
  sensor_msgs::msg::Image::SharedPtr pinned_layout_image_;
  geometry_msgs::msg::TransformStamped::SharedPtr pinned_layout_transform_;
  std::mutex observation_mutex_;
  std::mutex refinement_target_mutex_;
  std::unordered_set<std::tuple<int, int, int>, VoxelHash>
    refinement_target_support_voxels_;
  std::vector<Eigen::Vector3f> refinement_contact_region_;
  int64_t refinement_target_stamp_ns_ = 0;
  float refinement_target_support_voxel_m_ = 0.008F;
  float refinement_contact_region_radius_m_ = 0.05F;
  float refinement_contact_merge_voxel_m_ = 0.001F;
  int maximum_candidates_to_test_ = 600;
  int maximum_valid_candidates_ = 100;
  int maximum_ranking_pool_candidates_ = 300;
  double angular_step_degrees_ = 30.0;
  double local_crop_radius_ = 0.04;
  double minimum_gripper_width_ = 0.005;
  double maximum_gripper_width_ = 0.100;
  double finger_length_m_ = 0.070;
  double finger_width_m_ = 0.010;
  double finger_height_m_ = 0.020;
  double point_correction_limit_ = 24.0;
  double pregrasp_offset_ = 0.156;
  float grasp_feedback_tolerance_rad_ = 0.034906585F;
  double pregrasp_position_tolerance_m_ = 0.008;
  double pregrasp_orientation_tolerance_degrees_ = 16.0;
  double grasp_scene_crop_radius_m_ = 0.20;
  double grasp_target_sample_radius_m_ = 0.14;
  double target_region_neighbor_distance_m_ = 0.020;
  double target_foreground_band_m_ = 0.030;
  double target_collision_clearance_m_ = 0.030;
  bool enable_grasp_collision_check_ = true;
  bool require_measured_antipodal_contacts_ = true;
  bool allow_pregrasp_collision_candidates_ = false;
  double antipodal_normal_cosine_ = 0.90;
  double antipodal_normal_radius_m_ = 0.008;
  int antipodal_min_normal_neighbors_ = 10;
  double antipodal_seed_search_radius_m_ = 0.040;
  int antipodal_maximum_local_points_ = 512;
  double grasp_contact_depth_m_ = 0.060;
  double pregrasp_opening_clearance_m_ = 0.02;
  // Depth/mask registration error allowed immediately outside a measured
  // silhouette contact.  It is not a relaxation of the outer finger body or
  // scene collision checks.
  double target_contact_surface_tolerance_m_ = 0.006;
  double pinch_max_opening_m_ = 0.050;
  double pinch_min_free_gap_m_ = 0.013;
  double aligned_zero_pose_yaw_limit_degrees_ = 10.0;
  double aligned_zero_pose_pitch_limit_degrees_ = 90.0;
  double aligned_zero_pose_roll_limit_degrees_ = 90.0;
  double handle_approach_roll_limit_degrees_ = 100.0;
  double handle_approach_roll_step_degrees_ = 10.0;
  double handle_visible_approach_min_cosine_ = 0.25;
  double handle_approach_dedup_degrees_ = 5.0;
  double silhouette_min_tangent_anisotropy_ = 1.15;
  double natural_proposal_quality_weight_ = 1.0;
  double natural_wide_opening_weight_ = 1.0;
  double natural_vertical_closing_weight_ = 0.35;
  double natural_seed_distance_weight_ = 0.25;
  double natural_vertical_approach_weight_ = 0.15;
  double natural_camera_approach_weight_ = 0.75;
  double natural_joint_travel_weight_ = 0.15;
  double natural_joint_margin_weight_ = 0.10;
  double natural_pregrasp_override_weight_ = 0.25;
  double natural_corner_penalty_weight_ = 2.0;
  bool enable_grasp_path_preplanning_ = false;
  double grasp_path_length_weight_ = 0.50;
  // GPD reports the bottom of its finger closing region.  Piper's
  // gripper_base is 61.5 mm behind the beginning of the physical fingers.
  double grasp_tcp_offset_m_ = -0.0615;
  Eigen::Vector3f grasp_position_offset_base_m_ = Eigen::Vector3f::Zero();
  int grasp_visualization_limit_ = 100;
  double hggd_diversity_position_m_ = 0.030;
  double hggd_diversity_approach_degrees_ = 20.0;
  double hggd_diversity_closing_degrees_ = 30.0;
  std::string grasp_candidate_backend_ = "hggd";
  KnownCloudGraspParameters cloud_parameters_;
  std::unique_ptr<KnownCloudGraspGenerator> cloud_generator_;
  std::unique_ptr<HggdMcuAdapter> hggd_adapter_;
  std::string hggd_initialization_error_;
  std::string hggd_model_directory_;
  int hggd_cpu_threads_ = 4;
  int hggd_maximum_proposals_ = 100;
  double hggd_seed_snap_max_distance_m_ = 0.05;
  LocalGraspRefinerParameters local_refiner_parameters_;
  std::unique_ptr<LocalGraspRefiner> local_grasp_refiner_;

  bool continuous_nav_active_ = false;
  std::vector<float> continuous_target_joints_;
  bool deterministic_ik_ = true;
  int deterministic_ik_seed_count_ = 32;
  std::vector<float> deterministic_ik_home_;
  std::optional<std::array<float, 6>> previous_valid_ik_;
  int lookahead_poses_ = 10;
  float final_approach_translation_step_m_ = 0.002F;
  float final_approach_max_joint_step_rad_ = 0.08F;
  float final_approach_contact_phase_m_ = 0.006F;
  bool final_approach_check_finger_environment_ = false;
  float placement_position_tolerance_m_ = 0.005F;
  float placement_descent_orientation_tolerance_rad_ = 0.01F;
  float placement_descent_step_m_ = 0.002F;
  float placement_descent_max_joint_step_rad_ = 0.05F;
  float placement_contact_distance_m_ = 0.006F;
  float placement_orientation_tolerance_rad_ = 0.35F;
  float placement_target_orientation_tolerance_rad_ = 0.17453293F;
  float placement_upright_tilt_tolerance_rad_ = 0.08726646F;
  float placement_preplace_offset_m_ = 0.10F;
  bool pipeline_profiling_ = false;
  bool placement_fast_ik_ = true;
  int placement_transfer_max_iterations_ = 5000;
  double placement_transfer_timeout_sec_ = 30.0;
  bool keep_object_upright_ = true;
  float object_upright_tolerance_rad_ = 0.08726646F;
  bool placement_allow_separating_contact_departure_ = true;
  bool placement_bypass_attached_collision_during_lift_ = true;
  bool placement_bypass_robot_environment_during_lift_ = true;
  float placement_departure_contact_tolerance_m_ = 0.001F;
  AttachedObjectScene::NoiseBudget placement_noise_policy_;

  float max_workspace_dist_ = 1.5f;
  float ws_x_min_ = -0.8f;
  float ws_x_max_ = 0.8f;
  float ws_y_min_ = -0.8f;
  float ws_y_max_ = 0.8f;
  float ws_z_min_ = -0.4f;
  float ws_z_max_ = 0.8f;

  rclcpp::Service<my_actions::srv::CheckAndReplan>::SharedPtr check_replan_srv_;
  rclcpp::Service<my_actions::srv::CalculateFallbackPose>::SharedPtr calc_fallback_srv_;
  rclcpp::Service<my_actions::srv::ToggleContinuousNavigation>::SharedPtr toggle_nav_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_map_srv_;
  rclcpp::Service<my_actions::srv::PlanAPF>::SharedPtr plan_apf_srv_;
  rclcpp::Service<my_actions::srv::LookAt>::SharedPtr look_at_ik_srv_;
  rclcpp::Service<my_actions::srv::FindGraspCandidates>::SharedPtr grasp_candidates_srv_;
  rclcpp::Service<my_actions::srv::FindCloudGrasps>::SharedPtr cloud_grasps_srv_;
  rclcpp::Service<my_actions::srv::RefineGrasp>::SharedPtr refine_grasp_srv_;
  rclcpp::Service<my_actions::srv::PlanCartesianGraspApproach>::SharedPtr
    plan_cartesian_grasp_approach_srv_;
  rclcpp::Service<my_actions::srv::PlanPlacement>::SharedPtr plan_placement_srv_;
  vamp_node::LayoutSceneCache layout_scene_cache_;
  rclcpp::Service<my_actions::srv::PinLayoutObservation>::SharedPtr pin_layout_srv_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr registered_cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    object_support_surface_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr observation_image_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr
    observation_camera_transform_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_world_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr segmented_target_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmented_target_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmented_target_mask_pub_;
  std::mutex segmented_target_mutex_;
  sensor_msgs::msg::PointCloud2::SharedPtr latest_segmented_target_cloud_;
  sensor_msgs::msg::Image::SharedPtr latest_segmented_target_image_;
  std::mutex object_support_surface_mutex_;
  sensor_msgs::msg::PointCloud2::SharedPtr latest_object_support_surface_;
  std::mutex placement_plan_mutex_;
  std::optional<my_actions::srv::PlanPlacement::Response> cached_placement_plan_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr grasp_markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr angle_trial_markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    valid_grasp_markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr known_cloud_proposals_markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    hggd_raw_proposals_markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr hggd_proposals_markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr grasp_debug_markers_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr hggd_heatmap_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr refined_grasp_debug_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    placement_plan_markers_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
    refined_placed_object_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
    refined_placed_support_surface_pub_;

public:
  explicit VampPlannerServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("vamp_planner_server", options)
  {
    maximum_candidates_to_test_ = this->declare_parameter<int>("maximum_candidates_to_test", 600);
    const double tolerance_degrees = declare_parameter<double>("joint_limit_tolerance_degrees", 2.0);
    if (!std::isfinite(tolerance_degrees) || tolerance_degrees < 0 || tolerance_degrees > 2) {
      throw std::invalid_argument("joint_limit_tolerance_degrees must be between 0 and 2");
    }
    configurePiperJointLimitTolerance(static_cast<float>(tolerance_degrees*M_PI/180.0));
    Robot::configure_joint_limits(kPiperJointLower, kPiperJointUpper);
    const auto feedback_tolerance = declare_parameter<double>("grasp_feedback_tolerance_degrees", 2.0);
    if (!std::isfinite(feedback_tolerance) || feedback_tolerance < 0.0 || feedback_tolerance > 2.0)
      throw std::invalid_argument("grasp_feedback_tolerance_degrees must be between 0 and 2");
    grasp_feedback_tolerance_rad_ = static_cast<float>(feedback_tolerance*M_PI/180.0);
    maximum_valid_candidates_ = this->declare_parameter<int>("maximum_valid_candidates", 100);
    maximum_ranking_pool_candidates_ = this->declare_parameter<int>(
      "maximum_ranking_pool_candidates", 300);
    angular_step_degrees_ = this->declare_parameter<double>("angular_step_degrees", 30.0);
    local_crop_radius_ = this->declare_parameter<double>("local_crop_radius", 0.04);
    minimum_gripper_width_ = this->declare_parameter<double>("minimum_gripper_width", 0.005);
    maximum_gripper_width_ = this->declare_parameter<double>("maximum_gripper_width", 0.100);
    finger_length_m_ = this->declare_parameter<double>("finger_length_m", 0.070);
    finger_width_m_ = this->declare_parameter<double>("finger_width_m", 0.010);
    finger_height_m_ = this->declare_parameter<double>("finger_height_m", 0.020);
    point_correction_limit_ = this->declare_parameter<double>("point_correction_limit", 24.0);
    pregrasp_offset_ = this->declare_parameter<double>("pregrasp_offset_m", 0.156);
    pregrasp_position_tolerance_m_ = this->declare_parameter<double>(
      "pregrasp_position_tolerance_m", 0.008);
    pregrasp_orientation_tolerance_degrees_ = this->declare_parameter<double>(
      "pregrasp_orientation_tolerance_degrees", 16.0);
    grasp_scene_crop_radius_m_ = this->declare_parameter<double>("grasp_scene_crop_radius_m", 0.20);
    grasp_target_sample_radius_m_ = this->declare_parameter<double>("grasp_target_sample_radius_m", 0.14);
    target_region_neighbor_distance_m_ = this->declare_parameter<double>(
      "target_region_neighbor_distance_m", 0.020);
    target_foreground_band_m_ = this->declare_parameter<double>(
      "target_foreground_band_m", 0.030);
    target_collision_clearance_m_ = this->declare_parameter<double>(
      "target_collision_clearance_m", 0.030);
    enable_grasp_collision_check_ = this->declare_parameter<bool>(
      "enable_grasp_collision_check", true);
    require_measured_antipodal_contacts_ = this->declare_parameter<bool>(
      "require_measured_antipodal_contacts", true);
    allow_pregrasp_collision_candidates_ = this->declare_parameter<bool>(
      "allow_pregrasp_collision_candidates", false);
    antipodal_normal_cosine_ = this->declare_parameter<double>(
      "antipodal_normal_cosine", 0.90);
    antipodal_normal_radius_m_ = this->declare_parameter<double>(
      "antipodal_normal_radius_m", 0.008);
    antipodal_min_normal_neighbors_ = this->declare_parameter<int>(
      "antipodal_min_normal_neighbors", 10);
    antipodal_seed_search_radius_m_ = this->declare_parameter<double>(
      "antipodal_seed_search_radius_m", 0.040);
    antipodal_maximum_local_points_ = this->declare_parameter<int>(
      "antipodal_maximum_local_points", 512);
    grasp_contact_depth_m_ = this->declare_parameter<double>(
      "grasp_contact_depth_m", 0.060);
    pregrasp_opening_clearance_m_ = this->declare_parameter<double>(
      "pregrasp_opening_clearance_m", 0.02);
    target_contact_surface_tolerance_m_ = this->declare_parameter<double>(
      "target_contact_surface_tolerance_m", 0.006);
    pinch_max_opening_m_ = this->declare_parameter<double>(
      "pinch_max_opening_m", 0.050);
    pinch_min_free_gap_m_ = this->declare_parameter<double>(
      "pinch_min_free_gap_m", 0.013);
    aligned_zero_pose_yaw_limit_degrees_ = this->declare_parameter<double>(
      "aligned_zero_pose_yaw_limit_degrees", 10.0);
    aligned_zero_pose_pitch_limit_degrees_ = this->declare_parameter<double>(
      "aligned_zero_pose_pitch_limit_degrees", 90.0);
    aligned_zero_pose_roll_limit_degrees_ = this->declare_parameter<double>(
      "aligned_zero_pose_roll_limit_degrees", 90.0);
    handle_approach_roll_limit_degrees_ = this->declare_parameter<double>(
      "handle_approach_roll_limit_degrees", 100.0);
    handle_approach_roll_step_degrees_ = this->declare_parameter<double>(
      "handle_approach_roll_step_degrees", 10.0);
    handle_visible_approach_min_cosine_ = this->declare_parameter<double>(
      "handle_visible_approach_min_cosine", 0.25);
    handle_approach_dedup_degrees_ = this->declare_parameter<double>(
      "handle_approach_dedup_degrees", 5.0);
    silhouette_min_tangent_anisotropy_ = this->declare_parameter<double>(
      "silhouette_min_tangent_anisotropy", 1.15);
    antipodal_normal_cosine_ = std::clamp(
      antipodal_normal_cosine_, 0.0, 0.999);
    antipodal_normal_radius_m_ = std::clamp(
      antipodal_normal_radius_m_, 0.004, 0.030);
    antipodal_min_normal_neighbors_ = std::clamp(
      antipodal_min_normal_neighbors_, 6, 100);
    antipodal_seed_search_radius_m_ = std::clamp(
      antipodal_seed_search_radius_m_, 0.010, 0.100);
    antipodal_maximum_local_points_ = std::clamp(
      antipodal_maximum_local_points_, 64, 1024);
    grasp_contact_depth_m_ = std::clamp(
      grasp_contact_depth_m_, 0.0, finger_length_m_);
    pregrasp_opening_clearance_m_ = std::clamp(
      pregrasp_opening_clearance_m_, 0.002, maximum_gripper_width_);
    pregrasp_offset_ = std::clamp(pregrasp_offset_, 0.05, 0.195);
    pregrasp_position_tolerance_m_ = std::clamp(
      pregrasp_position_tolerance_m_, 0.004, 0.015);
    pregrasp_orientation_tolerance_degrees_ = std::clamp(
      pregrasp_orientation_tolerance_degrees_, 0.0, 20.0);
    target_contact_surface_tolerance_m_ = std::clamp(
      target_contact_surface_tolerance_m_, 0.001, 0.010);
    pinch_max_opening_m_ = std::clamp(
      pinch_max_opening_m_, minimum_gripper_width_, maximum_gripper_width_);
    pinch_min_free_gap_m_ = std::clamp(
      pinch_min_free_gap_m_, finger_width_m_, maximum_gripper_width_);
    aligned_zero_pose_yaw_limit_degrees_ = std::clamp(
      aligned_zero_pose_yaw_limit_degrees_, 1.0, 90.0);
    aligned_zero_pose_pitch_limit_degrees_ = std::clamp(
      aligned_zero_pose_pitch_limit_degrees_, 1.0, 90.0);
    aligned_zero_pose_roll_limit_degrees_ = std::clamp(
      aligned_zero_pose_roll_limit_degrees_, 1.0, 90.0);
    handle_approach_roll_limit_degrees_ = std::clamp(
      handle_approach_roll_limit_degrees_, 0.0, 180.0);
    handle_approach_roll_step_degrees_ = std::clamp(
      handle_approach_roll_step_degrees_, 1.0, 45.0);
    handle_visible_approach_min_cosine_ = std::clamp(
      handle_visible_approach_min_cosine_, 0.0, 1.0);
    handle_approach_dedup_degrees_ = std::clamp(
      handle_approach_dedup_degrees_, 0.1, 45.0);
    silhouette_min_tangent_anisotropy_ = std::clamp(
      silhouette_min_tangent_anisotropy_, 1.01, 10.0);
    natural_proposal_quality_weight_ = this->declare_parameter<double>(
      "natural_proposal_quality_weight", 1.0);
    natural_wide_opening_weight_ = this->declare_parameter<double>(
      "natural_wide_opening_weight", 1.0);
    natural_vertical_closing_weight_ = this->declare_parameter<double>(
      "natural_vertical_closing_weight", 0.35);
    natural_seed_distance_weight_ = this->declare_parameter<double>(
      "natural_seed_distance_weight", 0.25);
    natural_vertical_approach_weight_ = this->declare_parameter<double>(
      "natural_vertical_approach_weight", 0.15);
    natural_camera_approach_weight_ = this->declare_parameter<double>(
      "natural_camera_approach_weight", 0.75);
    natural_joint_travel_weight_ = this->declare_parameter<double>(
      "natural_joint_travel_weight", 0.15);
    natural_joint_margin_weight_ = this->declare_parameter<double>(
      "natural_joint_margin_weight", 0.10);
    natural_pregrasp_override_weight_ = this->declare_parameter<double>(
      "natural_pregrasp_override_weight", 0.25);
    natural_corner_penalty_weight_ = this->declare_parameter<double>(
      "natural_corner_penalty_weight", 2.0);
    enable_grasp_path_preplanning_ = this->declare_parameter<bool>(
      "enable_grasp_path_preplanning", false);
    grasp_path_length_weight_ = this->declare_parameter<double>(
      "grasp_path_length_weight", 0.50);
    grasp_path_length_weight_ = std::clamp(
      grasp_path_length_weight_, 0.0, 5.0);
    grasp_tcp_offset_m_ = this->declare_parameter<double>("grasp_tcp_offset_m", -0.0615);
    rcl_interfaces::msg::ParameterDescriptor offset_descriptor;
    offset_descriptor.read_only = true;
    grasp_position_offset_base_m_ = vamp_node::validatedGraspPositionOffset(
      this->declare_parameter<std::vector<double>>(
        "grasp_position_offset_base_m", std::vector<double>{0., 0., 0.}, offset_descriptor));
    RCLCPP_INFO(this->get_logger(), "Base-frame grasp position offset: [%.4f, %.4f, %.4f] m",
      grasp_position_offset_base_m_.x(), grasp_position_offset_base_m_.y(),
      grasp_position_offset_base_m_.z());
    grasp_visualization_limit_ = this->declare_parameter<int>("grasp_visualization_limit", 100);
    hggd_diversity_position_m_ = this->declare_parameter<double>(
      "hggd_diversity_position_m", 0.030);
    hggd_diversity_approach_degrees_ = this->declare_parameter<double>(
      "hggd_diversity_approach_degrees", 20.0);
    hggd_diversity_closing_degrees_ = this->declare_parameter<double>(
      "hggd_diversity_closing_degrees", 30.0);
    grasp_candidate_backend_ = this->declare_parameter<std::string>(
      "grasp_candidate_backend", "hggd");
    std::transform(grasp_candidate_backend_.begin(), grasp_candidate_backend_.end(),
      grasp_candidate_backend_.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value)); });
    if (grasp_candidate_backend_ != "hggd") {
      throw std::invalid_argument("grasp_candidate_backend only supports hggd");
    }
    cloud_parameters_.maximum_proposals = this->declare_parameter<int>(
      "known_cloud_maximum_proposals", 300);
    cloud_parameters_.voxel_size_m = static_cast<float>(this->declare_parameter<double>(
      "known_cloud_voxel_size_m", 0.005));
    cloud_parameters_.roll_step_degrees = static_cast<float>(this->declare_parameter<double>(
      "known_cloud_roll_step_degrees", 15.0));
    cloud_parameters_.normal_radius_m = static_cast<float>(this->declare_parameter<double>(
      "known_cloud_normal_radius_m", 0.008));
    cloud_parameters_.min_normal_neighbors = this->declare_parameter<int>(
      "known_cloud_min_normal_neighbors", 10);
    cloud_parameters_.dedup_position_m = static_cast<float>(this->declare_parameter<double>(
      "known_cloud_dedup_position_m", 0.010));
    cloud_parameters_.dedup_angle_degrees = static_cast<float>(this->declare_parameter<double>(
      "known_cloud_dedup_angle_degrees", 10.0));
    cloud_parameters_.min_contact_center_separation_m = static_cast<float>(
      this->declare_parameter<double>("known_cloud_min_contact_center_separation_m", 0.025));
    cloud_parameters_.local_pca_searches = this->declare_parameter<int>(
      "known_cloud_local_pca_searches", 50);
    cloud_parameters_.global_pca_searches = this->declare_parameter<int>(
      "known_cloud_global_pca_searches", 50);
    cloud_parameters_.maximum_approach_variants_per_pair = this->declare_parameter<int>(
      "known_cloud_maximum_approach_variants_per_pair", 2);
    deterministic_ik_ = this->declare_parameter<bool>("deterministic_ik", true);
    deterministic_ik_seed_count_ = this->declare_parameter<int>(
      "deterministic_ik_seed_count", 32);
    const auto deterministic_ik_home_d = this->declare_parameter<std::vector<double>>(
      "deterministic_ik_home_joint_state", {0.0, 0.8, -1.4, 0.0, 0.0, 0.0});
    deterministic_ik_home_.assign(
      deterministic_ik_home_d.begin(), deterministic_ik_home_d.end());
    final_approach_translation_step_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "final_approach_translation_step_m", 0.002), 0.0005, 0.002));
    final_approach_check_finger_environment_ = declare_parameter<bool>(
      "final_approach_check_finger_environment", false);
    final_approach_max_joint_step_rad_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "final_approach_max_joint_step_rad", 0.08), 0.01, 0.25));
    final_approach_contact_phase_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "final_approach_contact_phase_m", 0.006), 0.0, 0.010));
    placement_position_tolerance_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "placement_position_tolerance_m", 0.005), 0.0005, 0.025));
    placement_descent_orientation_tolerance_rad_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>("placement_descent_orientation_tolerance_rad", 0.01), 0.002, 0.02));
    placement_descent_step_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>("placement_descent_step_m", 0.002), 0.001, 0.002));
    placement_descent_max_joint_step_rad_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>("placement_descent_max_joint_step_rad", 0.05), 0.01, 0.10));
    placement_contact_distance_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>("placement_contact_distance_m", 0.006), 0.003, 0.008));
    placement_orientation_tolerance_rad_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "placement_orientation_tolerance_rad", 0.35), 0.10, 0.60));
    placement_target_orientation_tolerance_rad_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "placement_target_orientation_tolerance_rad", 0.17453293), 0.01, 0.17453293));
    placement_upright_tilt_tolerance_rad_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "placement_upright_tilt_tolerance_rad", 0.08726646), 0.017, 0.175));
    pipeline_profiling_ = this->declare_parameter<bool>("pipeline_profiling", false);
    placement_fast_ik_ = this->declare_parameter<bool>("placement_fast_ik", true);
    placement_transfer_max_iterations_ = this->declare_parameter<int>("placement_transfer_max_iterations", 5000);
    placement_transfer_timeout_sec_ = this->declare_parameter<double>("placement_transfer_timeout_sec", 30.0);
    const auto motion_orientation = declare_parameter<std::string>("placement_motion_orientation_policy", "keep_upright");
    if (motion_orientation != "keep_upright" && motion_orientation != "free")
      throw std::invalid_argument("placement_motion_orientation_policy must be keep_upright or free");
    keep_object_upright_ = motion_orientation == "keep_upright";
    const auto upright_tolerance = declare_parameter<double>("placement_motion_upright_tolerance_rad", 0.08726646);
    if (!std::isfinite(upright_tolerance) || upright_tolerance <= 0.0 || upright_tolerance > 0.17453293)
      throw std::invalid_argument("placement_motion_upright_tolerance_rad must be >0 and <=0.17453293");
    object_upright_tolerance_rad_ = static_cast<float>(upright_tolerance);
    placement_allow_separating_contact_departure_ = this->declare_parameter<bool>(
      "placement_allow_separating_contact_departure", true);
    placement_bypass_robot_environment_during_lift_ = this->declare_parameter<bool>(
      "placement_bypass_robot_environment_during_lift", true);
    placement_bypass_attached_collision_during_lift_ = this->declare_parameter<bool>(
      "placement_bypass_attached_collision_during_lift", true);
    if (placement_bypass_attached_collision_during_lift_) {
      RCLCPP_WARN(this->get_logger(),
        "Lift bypass enabled: attached-object/environment checks are skipped "
        "during lift only; robot, finger, object-arm, transfer and descent checks remain active.");
    }
    placement_departure_contact_tolerance_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>("placement_departure_contact_tolerance_m", 0.001),
      0.0, 0.002));
    placement_noise_policy_.maximum_points = static_cast<size_t>(std::clamp<int64_t>(
      this->declare_parameter<int>("placement_noise_max_points", 32), 0, 64));
    placement_noise_policy_.maximum_voxels = static_cast<size_t>(std::clamp<int64_t>(
      this->declare_parameter<int>("placement_noise_max_voxels", 10), 0, 16));
    placement_preplace_offset_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "placement_preplace_offset_m", 0.10), 0.05, 0.20));
    grasp_observation_cache_size_ = static_cast<size_t>(std::max<int64_t>(
      2, this->declare_parameter<int>("grasp_observation_cache_size", 1000)));
    cloud_parameters_.maximum_proposals = std::clamp(
      cloud_parameters_.maximum_proposals, 100, 500);
    maximum_candidates_to_test_ = std::clamp(maximum_candidates_to_test_, 1, 2000);
    maximum_ranking_pool_candidates_ = std::clamp(maximum_ranking_pool_candidates_, 1, 1000);
    maximum_valid_candidates_ = std::clamp(maximum_valid_candidates_, 1, 100);
    grasp_visualization_limit_ = std::clamp(grasp_visualization_limit_, 1, 100);
    hggd_model_directory_ = this->declare_parameter<std::string>("hggd_model_directory", "");
    hggd_cpu_threads_ = static_cast<int>(std::clamp<int64_t>(
      this->declare_parameter<int>("hggd_cpu_threads", 4), 1, 16));
    hggd_maximum_proposals_ = static_cast<int>(std::clamp<int64_t>(
      this->declare_parameter<int>("hggd_maximum_proposals", 100), 1, 100));
    hggd_seed_snap_max_distance_m_ = std::clamp(
      this->declare_parameter<double>(
        "hggd_seed_snap_max_distance_m", 0.05),
      0.005, 0.10);
    local_refiner_parameters_.roi_radius_m = static_cast<float>(
      this->declare_parameter<double>("refine_roi_radius_m", 0.10));
    local_refiner_parameters_.snap_radius_px =
      this->declare_parameter<int>("refine_snap_radius_px", 18);
    local_refiner_parameters_.snap_max_distance_m = static_cast<float>(
      this->declare_parameter<double>("refine_snap_max_distance_m", 0.05));
    local_refiner_parameters_.neighbor_distance_m = static_cast<float>(
      this->declare_parameter<double>("refine_neighbor_distance_m", 0.006));
    local_refiner_parameters_.neighbor_depth_change_m = static_cast<float>(
      this->declare_parameter<double>("refine_neighbor_depth_change_m", 0.006));
    local_refiner_parameters_.minimum_normal_cosine = static_cast<float>(
      this->declare_parameter<double>("refine_minimum_normal_cosine", 0.90));
    local_refiner_parameters_.minimum_region_points =
      this->declare_parameter<int>("refine_minimum_region_points", 40);
    local_refiner_parameters_.angle_step_degrees = static_cast<float>(
      this->declare_parameter<double>("refine_angle_step_degrees", 5.0));
    local_refiner_parameters_.minimum_width_m = static_cast<float>(
      this->declare_parameter<double>("refine_minimum_width_m", 0.008));
    local_refiner_parameters_.maximum_width_m = static_cast<float>(
      this->declare_parameter<double>("refine_maximum_width_m", 0.10));
    local_refiner_parameters_.minimum_usable_depth_m = static_cast<float>(
      this->declare_parameter<double>("refine_minimum_usable_depth_m", 0.01));
    local_refiner_parameters_.cross_section_half_length_m = static_cast<float>(
      this->declare_parameter<double>("refine_cross_section_half_length_m", 0.012));
    local_refiner_parameters_.maximum_center_shift_m = static_cast<float>(
      this->declare_parameter<double>("refine_maximum_center_shift_m", 0.03));
    refinement_target_support_voxel_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>("refine_target_support_voxel_m", 0.008),
      0.004, 0.020));
    refinement_contact_region_radius_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>("refine_contact_region_radius_m", 0.05),
      0.01, 0.05));
    refinement_contact_merge_voxel_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>("refine_contact_merge_voxel_m", 0.001),
      0.0005, 0.004));
    local_refiner_parameters_.cross_section_half_length_m = std::clamp(
      local_refiner_parameters_.cross_section_half_length_m, 0.005F, 0.030F);
    local_refiner_parameters_.maximum_center_shift_m = std::clamp(
      local_refiner_parameters_.maximum_center_shift_m, 0.005F, 0.050F);
    local_grasp_refiner_ = std::make_unique<LocalGraspRefiner>(
      local_refiner_parameters_);
    hggd_diversity_position_m_ = std::clamp(
      hggd_diversity_position_m_, 0.001, 0.100);
    hggd_diversity_approach_degrees_ = std::clamp(
      hggd_diversity_approach_degrees_, 1.0, 90.0);
    hggd_diversity_closing_degrees_ = std::clamp(
      hggd_diversity_closing_degrees_, 1.0, 90.0);
    cloud_parameters_.voxel_size_m = std::clamp(
      cloud_parameters_.voxel_size_m, 0.002F, 0.020F);
    cloud_parameters_.roll_step_degrees = std::clamp(
      cloud_parameters_.roll_step_degrees, 5.0F, 45.0F);
    cloud_parameters_.normal_radius_m = std::clamp(
      cloud_parameters_.normal_radius_m, 0.004F, 0.030F);
    cloud_parameters_.min_normal_neighbors = std::clamp(
      cloud_parameters_.min_normal_neighbors, 6, 100);
    cloud_parameters_.dedup_position_m = std::clamp(
      cloud_parameters_.dedup_position_m, 0.002F, 0.050F);
    cloud_parameters_.dedup_angle_degrees = std::clamp(
      cloud_parameters_.dedup_angle_degrees, 1.0F, 45.0F);
    cloud_parameters_.min_contact_center_separation_m = std::clamp(
      cloud_parameters_.min_contact_center_separation_m, 0.005F, 0.100F);
    cloud_parameters_.local_pca_searches = std::clamp(
      cloud_parameters_.local_pca_searches, 1, 100);
    cloud_parameters_.global_pca_searches = std::clamp(
      cloud_parameters_.global_pca_searches, 1, 100);
    cloud_parameters_.maximum_approach_variants_per_pair = std::clamp(
      cloud_parameters_.maximum_approach_variants_per_pair, 1, 5);
    deterministic_ik_seed_count_ = std::clamp(deterministic_ik_seed_count_, 1, 128);
    if (deterministic_ik_home_.size() != 6) {
      RCLCPP_WARN(this->get_logger(), "deterministic_ik_home_joint_state must have six values; using zero home.");
      deterministic_ik_home_ = {0.0, 0.8, -1.4, 0.0, 0.0, 0.0};
    }
    cloud_parameters_.minimum_opening_m = static_cast<float>(minimum_gripper_width_);
    cloud_parameters_.maximum_opening_m = static_cast<float>(maximum_gripper_width_);
    cloud_parameters_.contact_depth_m = static_cast<float>(grasp_contact_depth_m_);
    cloud_parameters_.finger_length_m = static_cast<float>(finger_length_m_);
    cloud_parameters_.finger_width_m = static_cast<float>(finger_width_m_);
    cloud_parameters_.finger_height_m = static_cast<float>(finger_height_m_);
    cloud_generator_ = std::make_unique<KnownCloudGraspGenerator>(cloud_parameters_);

    if (grasp_candidate_backend_ == "hggd") {
      try {
        if (hggd_model_directory_.empty()) {
          hggd_model_directory_ = ament_index_cpp::get_package_share_directory("vamp_node") +
            "/models/hggd_mcu";
        }
        hggd_adapter_ = std::make_unique<HggdMcuAdapter>(
          hggd_model_directory_, hggd_cpu_threads_);
      } catch (const std::exception &error) {
        hggd_initialization_error_ = error.what();
        RCLCPP_ERROR(this->get_logger(), "HGGD-MCU initialization failed: %s", error.what());
      }
    }
    auto visual_qos = rclcpp::QoS(1).reliable().transient_local();
    object_world_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/vamp_planner/object_world", visual_qos);
    segmented_target_cloud_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/vamp_planner/segmented_target_cloud", visual_qos);
    segmented_target_image_pub_ =
      this->create_publisher<sensor_msgs::msg::Image>(
      "/vamp_planner/segmented_target_image", visual_qos);
    segmented_target_mask_pub_ =
      this->create_publisher<sensor_msgs::msg::Image>(
      "/vamp_planner/segmented_target_mask", visual_qos);
    grasp_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vamp_planner/grasp_candidates_markers", visual_qos);
    angle_trial_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/vamp_planner/grasp_angle_trials", visual_qos);
    valid_grasp_markers_pub_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/vamp_planner/valid_grasp_candidates_markers", visual_qos);
    known_cloud_proposals_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vamp_planner/known_cloud_proposals_markers", visual_qos);
    hggd_raw_proposals_markers_pub_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/vamp_planner/hggd_raw_proposals_markers", visual_qos);
    hggd_proposals_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vamp_planner/hggd_proposals_markers", visual_qos);
    grasp_debug_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vamp_planner/grasp_debug_markers", visual_qos);
    hggd_heatmap_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/vamp_planner/hggd_heatmap", visual_qos);
    refined_grasp_debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/vamp_planner/refined_grasp_debug", visual_qos);
    placement_plan_markers_pub_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/vamp_planner/placement_plan_markers", visual_qos);
    refined_placed_object_cloud_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/vamp_planner/refined_placed_object_cloud", visual_qos);
    refined_placed_support_surface_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/vamp_planner/refined_placed_support_surface", visual_qos);
    // Services for Path Planning & Collision Checking
    check_replan_srv_ = this->create_service<my_actions::srv::CheckAndReplan>(
      "/vamp_planner/check_and_replan",
      std::bind(&VampPlannerServer::handleCheckAndReplan, this, std::placeholders::_1, std::placeholders::_2));

    calc_fallback_srv_ = this->create_service<my_actions::srv::CalculateFallbackPose>(
      "/vamp_planner/calculate_fallback_pose",
      std::bind(&VampPlannerServer::handleCalculateFallbackPose, this, std::placeholders::_1, std::placeholders::_2));

    toggle_nav_srv_ = this->create_service<my_actions::srv::ToggleContinuousNavigation>(
      "/vamp_planner/toggle_continuous_navigation",
      std::bind(&VampPlannerServer::handleToggleNav, this, std::placeholders::_1, std::placeholders::_2));

    clear_map_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/vamp_planner/clear_map",
      std::bind(&VampPlannerServer::handleClearMap, this, std::placeholders::_1, std::placeholders::_2));

    plan_apf_srv_ = this->create_service<my_actions::srv::PlanAPF>(
      "/vamp_planner/plan_apf",
      std::bind(&VampPlannerServer::handlePlanAPF, this, std::placeholders::_1, std::placeholders::_2));

    look_at_ik_srv_ = this->create_service<my_actions::srv::LookAt>(
      "/vamp_planner/look_at_ik",
      std::bind(&VampPlannerServer::handleLookAtIK, this, std::placeholders::_1, std::placeholders::_2));
    grasp_candidates_srv_ = this->create_service<my_actions::srv::FindGraspCandidates>(
      "/vamp_planner/find_grasp_candidates",
      std::bind(&VampPlannerServer::handleFindGraspCandidates, this, std::placeholders::_1, std::placeholders::_2));
    cloud_grasps_srv_ = this->create_service<my_actions::srv::FindCloudGrasps>(
      "/vamp_planner/find_cloud_grasps",
      std::bind(&VampPlannerServer::handleFindCloudGrasps, this,
        std::placeholders::_1, std::placeholders::_2));
    refine_grasp_srv_ = this->create_service<my_actions::srv::RefineGrasp>(
      "/vamp_planner/refine_grasp",
      std::bind(&VampPlannerServer::handleRefineGrasp, this,
        std::placeholders::_1, std::placeholders::_2));
    plan_cartesian_grasp_approach_srv_ =
      this->create_service<my_actions::srv::PlanCartesianGraspApproach>(
        "/vamp_planner/plan_cartesian_grasp_approach",
        std::bind(&VampPlannerServer::handlePlanCartesianGraspApproach, this,
          std::placeholders::_1, std::placeholders::_2));
    plan_placement_srv_ = this->create_service<my_actions::srv::PlanPlacement>(
      "/vamp_planner/plan_placement",
      std::bind(&VampPlannerServer::handlePlanPlacement, this,
        std::placeholders::_1, std::placeholders::_2));

    pin_layout_srv_ = this->create_service<my_actions::srv::PinLayoutObservation>(
      "/vamp_planner/pin_layout_observation",
      [this](const std::shared_ptr<my_actions::srv::PinLayoutObservation::Request> request,
        std::shared_ptr<my_actions::srv::PinLayoutObservation::Response> response) {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        const rclcpp::Time stamp(request->observation_header.stamp);
        sensor_msgs::msg::PointCloud2::SharedPtr cloud;
        sensor_msgs::msg::Image::SharedPtr image;
        geometry_msgs::msg::TransformStamped::SharedPtr transform;
        for (const auto &entry : registered_cloud_cache_) {
          if (rclcpp::Time(entry->header.stamp) == stamp) cloud = entry;
        }
        for (const auto &entry : observation_image_cache_) {
          if (rclcpp::Time(entry->header.stamp) == stamp) image = entry;
        }
        for (const auto &entry : observation_camera_transform_cache_) {
          if (rclcpp::Time(entry->header.stamp) == stamp) transform = entry;
        }
        // Re-pinning the retained observation is idempotent even after the
        // rolling caches expire. Never combine components from different frames.
        if ((!cloud || !image || !transform) && pinned_layout_cloud_ &&
            pinned_layout_image_ && pinned_layout_transform_ &&
            rclcpp::Time(pinned_layout_cloud_->header.stamp) == stamp &&
            rclcpp::Time(pinned_layout_image_->header.stamp) == stamp &&
            rclcpp::Time(pinned_layout_transform_->header.stamp) == stamp) {
          cloud = pinned_layout_cloud_;
          image = pinned_layout_image_;
          transform = pinned_layout_transform_;
        }
        response->success = stamp.nanoseconds() != 0 && cloud && image && transform;
        if (response->success) {
          pinned_layout_cloud_ = cloud;
          pinned_layout_image_ = image;
          pinned_layout_transform_ = transform;
        }
        response->status = response->success ? "LAYOUT_OBSERVATION_PINNED" :
          "LAYOUT_SYNCHRONIZED_OBSERVATION_NOT_AVAILABLE";
      });

    // Subscribe directly to 3D Octree Occupied Voxels from Active Vision
    map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/active_vision/global_map", 10,
      std::bind(&VampPlannerServer::mapCallback, this, std::placeholders::_1));
    registered_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/active_vision/observation_cloud", rclcpp::QoS(1).transient_local().reliable(),
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        latest_registered_cloud_ = msg;
        registered_cloud_cache_.push_back(msg);
        while (registered_cloud_cache_.size() > grasp_observation_cache_size_) {
          registered_cloud_cache_.pop_front();
        }
      });
    object_support_surface_sub_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/placement/artificial_bottom_cloud",
        rclcpp::QoS(1).transient_local().reliable(),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
          if (msg->header.frame_id != "base_link" || msg->width == 0) return;
          std::lock_guard<std::mutex> lock(object_support_surface_mutex_);
          latest_object_support_surface_ = msg;
        });
    observation_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/active_vision/observation_image", rclcpp::QoS(1).transient_local().reliable(),
      [this](const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        latest_observation_image_ = msg;
        observation_image_cache_.push_back(msg);
        while (observation_image_cache_.size() > grasp_observation_cache_size_) {
          observation_image_cache_.pop_front();
        }
      });
    observation_camera_transform_sub_ =
      this->create_subscription<geometry_msgs::msg::TransformStamped>(
        "/active_vision/observation_camera_transform",
        rclcpp::QoS(1).transient_local().reliable(),
        [this](const geometry_msgs::msg::TransformStamped::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(observation_mutex_);
          latest_observation_camera_transform_ = msg;
          observation_camera_transform_cache_.push_back(msg);
          while (observation_camera_transform_cache_.size() >
                 grasp_observation_cache_size_) {
            observation_camera_transform_cache_.pop_front();
          }
        });
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera/color/camera_info", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        latest_camera_info_ = msg;
      });
    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/feedback/joint_states", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (msg->position.size() < 6) return;
        std::lock_guard<std::mutex> lock(observation_mutex_);
        latest_joints_.assign(msg->position.begin(), msg->position.begin() + 6);
      });

    RCLCPP_INFO(this->get_logger(), "Dedicated VAMP-MVT Motion Planner & Collision Checker Node Ready.");
  }

private:
  static Eigen::Matrix3f visualGripperRotation(
      const Eigen::Vector3f &closing_hint, const Eigen::Vector3f &approach_hint) {
    const Eigen::Vector3f approach = approach_hint.normalized();
    Eigen::Vector3f closing = closing_hint - approach * closing_hint.dot(approach);
    closing.normalize();
    const Eigen::Vector3f height = approach.cross(closing).normalized();
    Eigen::Matrix3f rotation;
    // The abstract RViz hand uses X=closing, Y=height, Z=approach.
    rotation.col(0) = closing;
    rotation.col(1) = height;
    rotation.col(2) = approach;
    return rotation;
  }

  static Eigen::Matrix3f piperGripperRotation(
      const Eigen::Vector3f &closing_hint, const Eigen::Vector3f &approach_hint,
      bool swap_fingers = false) {
    const Eigen::Vector3f approach = approach_hint.normalized();
    Eigen::Vector3f closing = closing_hint - approach * closing_hint.dot(approach);
    closing.normalize();
    if (swap_fingers) closing = -closing;
    // Piper's fingers separate along gripper_base Y and extend along +Z.
    // X=YxZ completes a proper right-handed gripper_base frame.  Swapping
    // both X/Y is the equivalent 180-degree wrist-roll solution for a
    // symmetric parallel-jaw grasp.
    const Eigen::Vector3f x_axis = closing.cross(approach).normalized();
    Eigen::Matrix3f rotation;
    rotation.col(0) = x_axis;
    rotation.col(1) = closing;
    rotation.col(2) = approach;
    return rotation;
  }

  std::vector<float> pregraspRetreatCandidates() const {
    std::vector<float> distances;
    const auto add = [&distances](float distance) {
        if (std::none_of(
            distances.begin(), distances.end(),
            [distance](float existing) {
              return std::abs(existing - distance) < 1.0e-4F;
            }))
        {
          distances.push_back(distance);
        }
      };
    add(std::clamp(static_cast<float>(pregrasp_offset_), 0.05F, 0.195F));
    for (const float distance : {0.12F, 0.10F, 0.08F, 0.06F, 0.14F, 0.18F, 0.19F}) {
      add(distance);
    }
    return distances;
  }

  static bool finitePoint(const Point &p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
  }

  bool refineHggdOpeningFromTarget(
      const std::vector<Eigen::Vector3f> &target_points,
      GraspProposal &proposal,
      float minimum_contact_z = -std::numeric_limits<float>::infinity()) const {
    Eigen::Vector3f closing = proposal.closing;
    Eigen::Vector3f approach = proposal.approach;
    Eigen::Vector3f height = proposal.height_axis;
    if (closing.squaredNorm() < 1.0e-8F ||
        approach.squaredNorm() < 1.0e-8F ||
        height.squaredNorm() < 1.0e-8F) {
      return false;
    }
    closing.normalize();
    approach.normalize();
    height.normalize();

    // AnchorNet's projected width is only a prior. Reconstruct the physical
    // cross-section from the unvoxelized target cloud. The depth slab spans a
    // finger so a curved body remains continuous from one silhouette side to
    // the other; the mask keeps table/background points out of this test.
    const float depth_half_extent = std::max(
      static_cast<float>(finger_length_m_),
      static_cast<float>(2.0 * target_contact_surface_tolerance_m_));
    const float height_half_extent = static_cast<float>(
      finger_height_m_ * 0.5 + target_contact_surface_tolerance_m_);
    const float lateral_extent = static_cast<float>(maximum_gripper_width_);
    struct CrossSectionSample {
      float lateral = 0.0F;
      Eigen::Vector3f point = Eigen::Vector3f::Zero();
    };
    std::vector<CrossSectionSample> cross_section;
    cross_section.reserve(target_points.size());
    for (const auto &point : target_points) {
      if (point.z() < minimum_contact_z) continue;
      const Eigen::Vector3f relative = point - proposal.sample;
      if (std::abs(relative.dot(approach)) > depth_half_extent ||
          std::abs(relative.dot(height)) > height_half_extent) {
        continue;
      }
      const float lateral = relative.dot(closing);
      if (std::abs(lateral) <= lateral_extent) {
        cross_section.push_back({lateral, point});
      }
    }
    if (cross_section.size() <
        static_cast<size_t>(antipodal_min_normal_neighbors_)) {
      return false;
    }
    std::sort(cross_section.begin(), cross_section.end(),
      [](const CrossSectionSample &left, const CrossSectionSample &right) {
        return left.lateral < right.lateral;
      });
    // Split the projected cross-section into continuous material bands. This
    // is the common skill selector for every object: a narrow band with free
    // finger space is PINCH; otherwise the jaws span the complete section as
    // a POWER_GRASP. No HANDLE/BODY/RIM label participates in this decision.
    struct MaterialInterval {
      size_t begin = 0;
      size_t end = 0;
    };
    constexpr float maximum_material_gap_m = 0.006F;
    std::vector<MaterialInterval> intervals;
    size_t begin = 0;
    for (size_t index = 1; index < cross_section.size(); ++index) {
      if (cross_section[index].lateral - cross_section[index - 1].lateral >
          maximum_material_gap_m) {
        intervals.push_back({begin, index - 1});
        begin = index;
      }
    }
    intervals.push_back({begin, cross_section.size() - 1});

    const auto nearest = std::min_element(
      cross_section.begin(), cross_section.end(),
      [](const CrossSectionSample &left, const CrossSectionSample &right) {
        return std::abs(left.lateral) < std::abs(right.lateral);
      });
    const size_t nearest_index = static_cast<size_t>(
      std::distance(cross_section.begin(), nearest));
    size_t selected_interval_index = 0;
    for (size_t index = 0; index < intervals.size(); ++index) {
      if (nearest_index >= intervals[index].begin &&
          nearest_index <= intervals[index].end) {
        selected_interval_index = index;
        break;
      }
    }
    const auto &local_interval = intervals[selected_interval_index];
    const size_t interval_size = local_interval.end - local_interval.begin + 1;
    if (interval_size < static_cast<size_t>(
          antipodal_min_normal_neighbors_)) {
      return false;
    }

    const float local_minimum = cross_section[local_interval.begin].lateral;
    const float local_maximum = cross_section[local_interval.end].lateral;
    const float local_opening = local_maximum - local_minimum;
    float largest_adjacent_gap = 0.0F;
    if (selected_interval_index > 0) {
      largest_adjacent_gap = std::max(
        largest_adjacent_gap,
        local_minimum - cross_section[
          intervals[selected_interval_index - 1].end].lateral);
    }
    if (selected_interval_index + 1 < intervals.size()) {
      largest_adjacent_gap = std::max(
        largest_adjacent_gap,
        cross_section[intervals[selected_interval_index + 1].begin].lateral -
          local_maximum);
    }

    // An isolated, narrow target band has free space on both mask sides. If
    // other material is present, require an explicit finger-sized gap next to
    // the band before calling it a pinch. This prevents a small patch on the
    // continuous cup wall from masquerading as a thin feature.
    const bool isolated_narrow_band = intervals.size() == 1 &&
      local_opening <= static_cast<float>(pinch_max_opening_m_);
    const bool narrow_band_with_gap =
      local_opening <= static_cast<float>(pinch_max_opening_m_) &&
      largest_adjacent_gap >= static_cast<float>(pinch_min_free_gap_m_);
    const bool use_pinch = isolated_narrow_band || narrow_band_with_gap;

    // Both skills use the material band containing the HGGD seed. A power
    // grasp spans that complete band, but never an unrelated band across a
    // handle hole or background gap.
    const size_t interval_begin = local_interval.begin;
    const size_t interval_end = local_interval.end;
    const size_t selected_size = interval_end - interval_begin + 1;
    // Trim isolated depth-edge pixels without shrinking a real surface band.
    const size_t trim = selected_size >= 40 ? selected_size / 40 : 0;
    if (interval_begin + trim >= interval_end - trim) return false;
    const float minimum_lateral = cross_section[interval_begin + trim].lateral;
    const float maximum_lateral = cross_section[interval_end - trim].lateral;
    const float measured_opening = maximum_lateral - minimum_lateral;
    if (!std::isfinite(measured_opening) ||
        measured_opening < static_cast<float>(minimum_gripper_width_) ||
        measured_opening > static_cast<float>(maximum_gripper_width_)) {
      return false;
    }

    // Keep contacts on real measured samples.  Reconstructing endpoints only
    // from their 1-D lateral coordinates can move a curved rim contact several
    // millimetres away from the surface and falsely fail the downstream
    // contact/collision tests.  Among the points near each trimmed edge,
    // choose the pair whose height/depth coordinates agree most closely.
    const float edge_band = std::max(0.002F, 0.08F * measured_opening);
    size_t best_left = interval_end - trim;
    size_t best_right = interval_begin + trim;
    float best_pair_cost = std::numeric_limits<float>::infinity();
    for (size_t low = interval_begin + trim;
         low <= interval_end - trim; ++low) {
      if (cross_section[low].lateral > minimum_lateral + edge_band) break;
      for (size_t high = interval_end - trim;; --high) {
        if (cross_section[high].lateral < maximum_lateral - edge_band) break;
        const Eigen::Vector3f delta =
          cross_section[high].point - cross_section[low].point;
        const float height_error = delta.dot(height);
        const float depth_error = delta.dot(approach);
        const float cost = height_error * height_error +
          depth_error * depth_error;
        if (cost < best_pair_cost) {
          best_pair_cost = cost;
          best_left = high;
          best_right = low;
        }
        if (high == interval_begin + trim) break;
      }
    }
    proposal.contact_left = cross_section[best_left].point;
    proposal.contact_right = cross_section[best_right].point;
    proposal.sample = 0.5F *
      (proposal.contact_left + proposal.contact_right);
    proposal.hand_base = proposal.sample - approach *
      static_cast<float>(grasp_contact_depth_m_);
    proposal.opening_m = measured_opening;
    proposal.has_contacts = true;
    proposal.grasp_skill = use_pinch ?
      LocalGraspSkill::kPinch : LocalGraspSkill::kPowerGrasp;
    return true;
  }

  bool alignHggdSeedToLocalDominantAxis(
      const std::vector<Eigen::Vector3f> &target_points,
      const Eigen::Vector3f &camera_origin,
      const Eigen::Matrix3f &current_gripper_rotation,
      const AntipodalContacts *silhouette_pair,
      const std::string &requested_affordance,
      GraspProposal &proposal,
      std::vector<GraspProposal> *aligned_hypotheses = nullptr) const {
    Eigen::Vector3f object_centroid = Eigen::Vector3f::Zero();
    std::vector<float> target_heights;
    target_heights.reserve(target_points.size());
    for (const auto &point : target_points) {
      object_centroid += point;
      target_heights.push_back(point.z());
    }
    object_centroid /= static_cast<float>(target_points.size());
    const auto top_index = static_cast<std::size_t>(
      0.95F * static_cast<float>(target_heights.size() - 1));
    std::nth_element(target_heights.begin(),
      target_heights.begin() + top_index, target_heights.end());
    const float robust_top_z = target_heights[top_index];
    constexpr float rim_top_band_m = 0.025F;
    std::optional<Eigen::Vector3f> nearby_rim_seed;
    if ((requested_affordance == "RIM" || requested_affordance == "AUTO") &&
        proposal.sample.z() < robust_top_z - rim_top_band_m) {
      // HGGD often centres a bowl on its visible interior. For a requested
      // rim grasp, move only to a nearby upper surface with local support.
      // AUTO keeps the original seed for body/handle hypotheses and uses the
      // snapped copy only to add a distinct rim hypothesis.
      nearby_rim_seed = vamp_node::supportedRimSeed(
        target_points, proposal.sample, robust_top_z);
    }
    if (requested_affordance == "RIM" && nearby_rim_seed) {
      proposal.sample = *nearby_rim_seed;
    }
    // Preserve a usable LocalNet centre. Only snap to the measured target as
    // a recovery path when its original 3-D offset has too little local
    // support; unconditional snapping biases thin handles toward the much
    // denser cup body at their junction.
    constexpr float local_radius_m = 0.020F;
    constexpr float local_radius_sq = local_radius_m * local_radius_m;
    const auto collect_neighborhood = [&target_points](
        const Eigen::Vector3f &center) {
        std::vector<Eigen::Vector3f> points;
        points.reserve(512);
        for (const auto &point : target_points) {
          if ((point - center).squaredNorm() <= local_radius_sq) {
            points.push_back(point);
          }
        }
        return points;
      };
    std::vector<Eigen::Vector3f> neighborhood =
      collect_neighborhood(proposal.sample);
    if (neighborhood.size() < 30) {
      float nearest_distance_sq = std::numeric_limits<float>::infinity();
      Eigen::Vector3f nearest_surface = Eigen::Vector3f::Zero();
      for (const auto &point : target_points) {
        const float distance_sq = (point - proposal.sample).squaredNorm();
        if (distance_sq < nearest_distance_sq) {
          nearest_distance_sq = distance_sq;
          nearest_surface = point;
        }
      }
      const float maximum_snap_sq = static_cast<float>(
        hggd_seed_snap_max_distance_m_ * hggd_seed_snap_max_distance_m_);
      if (!nearest_surface.allFinite() ||
          nearest_distance_sq > maximum_snap_sq) {
        return false;
      }
      proposal.sample = nearest_surface;
      neighborhood = collect_neighborhood(proposal.sample);
    }
    if (neighborhood.size() < 30) return false;

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto &point : neighborhood) centroid += point;
    centroid /= static_cast<float>(neighborhood.size());
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto &point : neighborhood) {
      const Eigen::Vector3f delta = point - centroid;
      covariance.noalias() += delta * delta.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) return false;
    Eigen::Vector3f dominant = solver.eigenvectors().col(2).normalized();
    Eigen::Vector3f frontal = proposal.sample - camera_origin;
    if (!dominant.allFinite() || frontal.squaredNorm() < 1.0e-8F) return false;
    frontal.normalize();

    const auto wrap_angle = [](float angle) {
        return std::atan2(std::sin(angle), std::cos(angle));
      };
    const Eigen::Vector3f current_approach =
      current_gripper_rotation.col(2).normalized();
    const Eigen::Vector3f current_closing_raw =
      current_gripper_rotation.col(1).normalized();
    const float yaw_limit = static_cast<float>(
      aligned_zero_pose_yaw_limit_degrees_ * M_PI / 180.0);
    const float pitch_limit = static_cast<float>(
      aligned_zero_pose_pitch_limit_degrees_ * M_PI / 180.0);
    const float roll_limit = static_cast<float>(
      aligned_zero_pose_roll_limit_degrees_ * M_PI / 180.0);

    // Apply the limits to physical quantities instead of coupled Euler
    // components: base-frame approach azimuth is yaw, approach elevation is
    // pitch, and closing-axis rotation around that approach is wrist roll.
    const auto bounded_rotation = [&wrap_angle, &current_approach,
        &current_closing_raw, yaw_limit, pitch_limit, roll_limit](
        const Eigen::Vector3f &approach_hint,
        const Eigen::Vector3f &closing_hint, Eigen::Matrix3f &rotation,
        float &adjustment_cost) {
        if (approach_hint.squaredNorm() < 1.0e-8F ||
            closing_hint.squaredNorm() < 1.0e-8F) return false;
        const Eigen::Vector3f desired_approach = approach_hint.normalized();
        const float current_azimuth = std::atan2(
          current_approach.y(), current_approach.x());
        const float current_elevation = std::atan2(
          current_approach.z(), std::hypot(
            current_approach.x(), current_approach.y()));
        const float desired_horizontal = std::hypot(
          desired_approach.x(), desired_approach.y());
        const float desired_azimuth = desired_horizontal > 1.0e-5F ?
          std::atan2(desired_approach.y(), desired_approach.x()) :
          current_azimuth;
        const float desired_elevation = std::atan2(
          desired_approach.z(), desired_horizontal);
        const float yaw_delta = std::clamp(
          wrap_angle(desired_azimuth - current_azimuth),
          -yaw_limit, yaw_limit);
        const float pitch_delta = std::clamp(
          desired_elevation - current_elevation,
          -pitch_limit, pitch_limit);
        const float azimuth = current_azimuth + yaw_delta;
        const float elevation = current_elevation + pitch_delta;
        Eigen::Vector3f approach(
          std::cos(elevation) * std::cos(azimuth),
          std::cos(elevation) * std::sin(azimuth),
          std::sin(elevation));
        approach.normalize();

        Eigen::Vector3f desired_closing = closing_hint -
          approach * closing_hint.dot(approach);
        Eigen::Vector3f current_closing = current_closing_raw -
          approach * current_closing_raw.dot(approach);
        if (desired_closing.squaredNorm() < 1.0e-8F ||
            current_closing.squaredNorm() < 1.0e-8F) return false;
        desired_closing.normalize();
        current_closing.normalize();
        if (desired_closing.dot(current_closing) < 0.0F) {
          desired_closing = -desired_closing;
        }
        float roll_delta = std::atan2(
          approach.dot(current_closing.cross(desired_closing)),
          current_closing.dot(desired_closing));
        roll_delta = std::clamp(
          roll_delta, -roll_limit, roll_limit);
        const Eigen::Vector3f closing = (
          Eigen::AngleAxisf(roll_delta, approach) * current_closing).normalized();
        rotation = piperGripperRotation(closing, approach);
        adjustment_cost = std::abs(yaw_delta) /
            yaw_limit +
          0.25F * std::abs(pitch_delta) /
            pitch_limit +
          0.10F * std::abs(roll_delta) / roll_limit;
        return rotation.allFinite();
      };

    struct ZeroPoseHypothesis {
      GraspProposal grasp;
      float cost = std::numeric_limits<float>::infinity();
      enum class Mode { kSide, kThinPca, kRim, kTopSurface } mode = Mode::kSide;
    };
    std::vector<ZeroPoseHypothesis> hypotheses;
    const float lambda_middle = std::max(solver.eigenvalues()[1], 1.0e-9F);
    const float local_anisotropy = solver.eigenvalues()[2] / lambda_middle;
    const bool seed_is_at_top_edge =
      proposal.sample.z() >= robust_top_z - rim_top_band_m;

    const auto contacts_on_measured_surface = [&target_points](
        const GraspProposal &candidate, const float tolerance) {
        const float tolerance_sq = tolerance * tolerance;
        const auto supported = [&target_points, tolerance_sq](
            const Eigen::Vector3f &contact) {
            float nearest_sq = std::numeric_limits<float>::infinity();
            for (const auto &point : target_points) {
              nearest_sq = std::min(
                nearest_sq, (point - contact).squaredNorm());
            }
            return nearest_sq <= tolerance_sq;
          };
        return supported(candidate.contact_left) &&
          supported(candidate.contact_right);
      };

    const auto add_hypothesis = [&](const Eigen::Vector3f &approach_hint,
        const Eigen::Vector3f &closing_hint,
        ZeroPoseHypothesis::Mode mode, float topology_cost) {
        Eigen::Matrix3f rotation;
        float adjustment_cost = 0.0F;
        if (!bounded_rotation(
            approach_hint, closing_hint, rotation, adjustment_cost)) return;
        GraspProposal candidate = proposal;
        candidate.height_axis = rotation.col(0).normalized();
        candidate.closing = rotation.col(1).normalized();
        candidate.approach = rotation.col(2).normalized();
        candidate.hand_base = candidate.sample - candidate.approach *
          static_cast<float>(grasp_contact_depth_m_);
        candidate.has_contacts = false;
        // The bowl interior below a rim is not the rim's jaw contact section.
        // Restrict contact measurement only; every target sample still participates
        // in the downstream finger sweep and full collision checks.
        const float contact_floor = mode == ZeroPoseHypothesis::Mode::kRim ?
          robust_top_z - rim_top_band_m : -std::numeric_limits<float>::infinity();
        if (!refineHggdOpeningFromTarget(target_points, candidate, contact_floor)) return;
        if (!contacts_on_measured_surface(
            candidate, static_cast<float>(
              target_contact_surface_tolerance_m_))) return;
        // A side grasp must not bridge the rim to a lower body patch. Rim and
        // thin-feature contacts must likewise stay within one local height
        // band rather than connecting two remote seams.
        const float contact_height_difference = std::abs(
          candidate.contact_left.z() - candidate.contact_right.z());
        if (contact_height_difference > 0.015F) return;
        if (mode == ZeroPoseHypothesis::Mode::kRim &&
            (candidate.sample.z() < robust_top_z - rim_top_band_m ||
            candidate.grasp_skill != LocalGraspSkill::kPinch)) return;
        if (mode == ZeroPoseHypothesis::Mode::kSide &&
            candidate.grasp_skill != LocalGraspSkill::kPowerGrasp) {
          topology_cost += 1.0F;
        }
        if (mode == ZeroPoseHypothesis::Mode::kThinPca &&
            (candidate.grasp_skill != LocalGraspSkill::kPinch ||
            local_anisotropy < 2.0F)) {
          topology_cost += 1.0F;
        }
        hypotheses.push_back({candidate,
          topology_cost + 0.05F * adjustment_cost +
            0.10F * candidate.opening_m,
          mode});
      };

    // Body-side solution: infer the hidden opposing wall from the visible
    // depth arc. The generic local-band measurement sees only the near wall
    // and incorrectly calls it a narrow pinch, so body power grasps must use
    // the fitted cross-section contacts directly.
    AntipodalContacts body_pair;
    if (buildCupBodyThicknessPair(
        target_points, proposal.sample, camera_origin, 0.0F, body_pair)) {
      Eigen::Vector3f outward_surface_normal;
      bool have_body_approach = estimateObjectNormal(
        target_points, proposal.sample, outward_surface_normal);
      if (have_body_approach && outward_surface_normal.dot(
          camera_origin - proposal.sample) < 0.0F) {
        outward_surface_normal = -outward_surface_normal;
      }
      // Infer the inward horizontal surface direction, then add only the
      // downward tilt justified by how far the contact lies below the object
      // top. Contacts near the top therefore retain a true horizontal side
      // approach; lower contacts gain progressively more vertical clearance.
      Eigen::Vector3f inward_surface = have_body_approach ?
        -outward_surface_normal : frontal;
      inward_surface.z() = 0.0F;
      if (inward_surface.squaredNorm() < 1.0e-8F) {
        inward_surface = Eigen::Vector3f(frontal.x(), frontal.y(), 0.0F);
      }
      have_body_approach = inward_surface.squaredNorm() >= 1.0e-8F;
      if (have_body_approach) inward_surface.normalize();
      const float body_standoff = std::clamp(
        static_cast<float>(pregrasp_offset_), 0.05F, 0.195F);
      const float depth_below_top = std::max(
        0.0F, robust_top_z - proposal.sample.z());
      const float maximum_vertical_rise = body_standoff *
        std::sin(pitch_limit);
      const float desired_vertical_rise = std::clamp(
        depth_below_top, 0.0F, maximum_vertical_rise);
      const float downward_tilt = std::asin(std::clamp(
        desired_vertical_rise / body_standoff, 0.0F,
        std::sin(pitch_limit)));
      const Eigen::Vector3f body_approach =
        std::cos(downward_tilt) * inward_surface -
        std::sin(downward_tilt) * Eigen::Vector3f::UnitZ();
      Eigen::Matrix3f rotation;
      float adjustment_cost = 0.0F;
      if (have_body_approach && bounded_rotation(
          body_approach, body_pair.closing_axis, rotation,
          adjustment_cost)) {
        ZeroPoseHypothesis body;
        body.mode = ZeroPoseHypothesis::Mode::kSide;
        body.grasp = proposal;
        body.grasp.sample = 0.5F * (body_pair.left + body_pair.right);
        body.grasp.height_axis = rotation.col(0).normalized();
        body.grasp.closing = rotation.col(1).normalized();
        body.grasp.approach = rotation.col(2).normalized();
        body.grasp.hand_base = body.grasp.sample - body.grasp.approach *
          static_cast<float>(grasp_contact_depth_m_);
        body.grasp.opening_m = body_pair.opening;
        body.grasp.contact_left = body.grasp.sample + body.grasp.closing *
          (0.5F * body_pair.opening);
        body.grasp.contact_right = body.grasp.sample - body.grasp.closing *
          (0.5F * body_pair.opening);
        body.grasp.has_contacts = true;
        body.grasp.model_supported_contacts = true;
        body.grasp.grasp_skill = LocalGraspSkill::kPowerGrasp;
        if (std::abs(body.grasp.contact_left.z() -
            body.grasp.contact_right.z()) <= 0.015F) {
          body.cost = 0.05F * adjustment_cost +
            0.10F * body.grasp.opening_m;
          hypotheses.push_back(std::move(body));
        }
      }
    }

    // Broad top-surface solution: approach along gravity and close across the
    // short horizontal PCA axis. Unlike the rim branch, this is a power grasp
    // against two opposing vertical walls and is suitable for boxes and other
    // flat-topped rigid objects without using an object-class label.
    if (seed_is_at_top_edge) {
      constexpr float top_surface_band_m = 0.008F;
      std::vector<Eigen::Vector3f> top_points;
      top_points.reserve(target_points.size() / 4);
      for (const auto &point : target_points) {
        if (point.z() >= robust_top_z - top_surface_band_m) {
          top_points.push_back(point);
        }
      }
      if (top_points.size() >= 50) {
        Eigen::Vector2f top_center = Eigen::Vector2f::Zero();
        float top_mean_z = 0.0F;
        for (const auto &point : top_points) {
          top_center += point.head<2>();
          top_mean_z += point.z();
        }
        top_center /= static_cast<float>(top_points.size());
        top_mean_z /= static_cast<float>(top_points.size());
        Eigen::Matrix2f top_covariance = Eigen::Matrix2f::Zero();
        float top_variance_z = 0.0F;
        for (const auto &point : top_points) {
          const Eigen::Vector2f delta = point.head<2>() - top_center;
          top_covariance.noalias() += delta * delta.transpose();
          const float delta_z = point.z() - top_mean_z;
          top_variance_z += delta_z * delta_z;
        }
        top_variance_z /= static_cast<float>(top_points.size());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> top_solver(
          top_covariance);
        if (top_solver.info() == Eigen::Success &&
            std::sqrt(top_variance_z) <= 0.004F) {
          Eigen::Vector3f closing_hint(
            top_solver.eigenvectors()(0, 0),
            top_solver.eigenvectors()(1, 0), 0.0F);
          Eigen::Matrix3f rotation;
          float adjustment_cost = 0.0F;
          if (bounded_rotation(
              -Eigen::Vector3f::UnitZ(), closing_hint, rotation,
              adjustment_cost)) {
            const Eigen::Vector3f top_closing =
              rotation.col(1).normalized();
            const Eigen::Vector3f top_height =
              rotation.col(0).normalized();
            Eigen::Vector3f nominal_center(
              top_center.x(), top_center.y(),
              robust_top_z - 0.012F);
            // Measure both sides at the HGGD seed's position along the
            // wall. A cup rim curves around the object; using the whole
            // finger-height strip can pair points from different arcs.
            nominal_center += top_height *
              (proposal.sample - nominal_center).dot(top_height);
            constexpr float paired_wall_tangent_band_m = 0.003F;
            struct WallSample {
              Eigen::Vector3f point = Eigen::Vector3f::Zero();
              float lateral = 0.0F;
            };
            std::vector<WallSample> wall_samples;
            wall_samples.reserve(target_points.size() / 4);
            for (const auto &point : target_points) {
              const Eigen::Vector3f relative = point - nominal_center;
              if (point.z() < robust_top_z - 0.040F ||
                  point.z() > robust_top_z + 0.004F ||
                  std::abs(relative.dot(top_height)) >
                    paired_wall_tangent_band_m) {
                continue;
              }
              wall_samples.push_back({point, relative.dot(top_closing)});
            }
            if (wall_samples.size() >= 30) {
              std::vector<float> laterals;
              laterals.reserve(wall_samples.size());
              for (const auto &sample : wall_samples) {
                laterals.push_back(sample.lateral);
              }
              std::sort(laterals.begin(), laterals.end());
              const size_t trim = std::max<size_t>(
                1, laterals.size() / 100);
              const float minimum_lateral = laterals[trim];
              const float maximum_lateral =
                laterals[laterals.size() - 1 - trim];
              const float opening = maximum_lateral - minimum_lateral;
              if (opening >= static_cast<float>(minimum_gripper_width_) &&
                  opening <= static_cast<float>(maximum_gripper_width_)) {
                const Eigen::Vector3f ideal_left = nominal_center +
                  top_closing * maximum_lateral;
                const Eigen::Vector3f ideal_right = nominal_center +
                  top_closing * minimum_lateral;
                Eigen::Vector3f left = Eigen::Vector3f::Zero();
                Eigen::Vector3f right = Eigen::Vector3f::Zero();
                float left_distance_sq = std::numeric_limits<float>::infinity();
                float right_distance_sq = std::numeric_limits<float>::infinity();
                constexpr float wall_edge_band_m = 0.004F;
                for (const auto &sample : wall_samples) {
                  if (sample.lateral >= maximum_lateral - wall_edge_band_m) {
                    const float distance_sq =
                      (sample.point - ideal_left).squaredNorm();
                    if (distance_sq < left_distance_sq) {
                      left_distance_sq = distance_sq;
                      left = sample.point;
                    }
                  }
                  if (sample.lateral <= minimum_lateral + wall_edge_band_m) {
                    const float distance_sq =
                      (sample.point - ideal_right).squaredNorm();
                    if (distance_sq < right_distance_sq) {
                      right_distance_sq = distance_sq;
                      right = sample.point;
                    }
                  }
                }
                if (left.allFinite() && right.allFinite() &&
                    std::isfinite(left_distance_sq) &&
                    std::isfinite(right_distance_sq) &&
                    std::abs(left.z() - right.z()) <= 0.008F) {
                  ZeroPoseHypothesis top;
                  top.mode = ZeroPoseHypothesis::Mode::kTopSurface;
                  top.grasp = proposal;
                  top.grasp.sample = 0.5F * (left + right);
                  top.grasp.height_axis = top_height;
                  top.grasp.closing = top_closing;
                  top.grasp.approach = -Eigen::Vector3f::UnitZ();
                  top.grasp.hand_base = top.grasp.sample -
                    top.grasp.approach *
                    static_cast<float>(grasp_contact_depth_m_);
                  top.grasp.opening_m = opening;
                  top.grasp.contact_left = left;
                  top.grasp.contact_right = right;
                  top.grasp.has_contacts = true;
                  top.grasp.top_surface_grasp = true;
                  top.grasp.grasp_skill = LocalGraspSkill::kPowerGrasp;
                  top.cost = 0.02F + 0.05F * adjustment_cost +
                    0.10F * opening;
                  hypotheses.push_back(std::move(top));
                }
              }
            }
          }
        }
      }
    }

    // Elongated handle/edge solution: prefer the two nearest boundaries of
    // the same contiguous silhouette. Unlike the 1-D projected cross-section,
    // this trace stops at the first empty image interval and therefore cannot
    // jump across a handle hole to the cup body. Fall back to the generic PCA
    // cross-section only when no stable thin silhouette exists.
    bool silhouette_is_local_cross_section = silhouette_pair != nullptr;
    if (silhouette_pair != nullptr) {
      const float upper_contact_z = std::max(
        silhouette_pair->left.z(), silhouette_pair->right.z());
      const float lower_contact_z = std::min(
        silhouette_pair->left.z(), silhouette_pair->right.z());
      const bool mixes_rim_and_body =
        upper_contact_z >= robust_top_z - 0.015F &&
        lower_contact_z < robust_top_z - 0.025F &&
        upper_contact_z - lower_contact_z > 0.020F;
      const float along_dominant = std::abs(
        silhouette_pair->closing_axis.normalized().dot(
          dominant.normalized()));
      // A silhouette thickness must cross a local elongated feature, not run
      // along it. Registered depth can turn a short 2-D mask trace into a
      // diagonal 3-D segment from a mug wall to its upper rim. Such a pair
      // looks narrow enough for the jaws but would tip the mug violently.
      // Dedicated rim hypotheses remain available through kRim.
      constexpr float maximum_along_dominant = 0.55F;
      silhouette_is_local_cross_section =
        !mixes_rim_and_body && along_dominant <= maximum_along_dominant;
    }
    if (silhouette_is_local_cross_section &&
        silhouette_pair->opening >= static_cast<float>(minimum_gripper_width_) &&
        silhouette_pair->opening <= static_cast<float>(maximum_gripper_width_)) {
      Eigen::Matrix3f rotation;
      float adjustment_cost = 0.0F;
      if (bounded_rotation(
          frontal, silhouette_pair->closing_axis, rotation,
          adjustment_cost)) {
        ZeroPoseHypothesis thin;
        thin.mode = ZeroPoseHypothesis::Mode::kThinPca;
        thin.grasp = proposal;
        thin.grasp.sample = 0.5F *
          (silhouette_pair->left + silhouette_pair->right);
        thin.grasp.height_axis = rotation.col(0).normalized();
        thin.grasp.closing = rotation.col(1).normalized();
        thin.grasp.approach = rotation.col(2).normalized();
        thin.grasp.hand_base = thin.grasp.sample - thin.grasp.approach *
          static_cast<float>(grasp_contact_depth_m_);
        thin.grasp.opening_m = silhouette_pair->opening;
        const Eigen::Vector3f contact_half_axis = thin.grasp.closing *
          (0.5F * silhouette_pair->opening);
        thin.grasp.contact_left = thin.grasp.sample + contact_half_axis;
        thin.grasp.contact_right = thin.grasp.sample - contact_half_axis;
        thin.grasp.has_contacts = true;
        thin.grasp.silhouette_supported_contacts = true;
        thin.grasp.grasp_skill = LocalGraspSkill::kPinch;
        thin.cost = 0.05F * adjustment_cost +
          0.10F * thin.grasp.opening_m;
        hypotheses.push_back(std::move(thin));
      }
    } else {
      add_hypothesis(
        frontal, dominant.cross(frontal),
        ZeroPoseHypothesis::Mode::kThinPca,
        local_anisotropy >= 2.0F ? 0.0F : 0.35F);
    }

    // Top-rim solution: approach downward and close radially across a single
    // wall, permitting one pad inside and one outside the rim.
    const Eigen::Vector3f non_rim_sample = proposal.sample;
    if (requested_affordance == "AUTO" && nearby_rim_seed) {
      proposal.sample = *nearby_rim_seed;
    }
    auto rim_seeds = vamp_node::supportedRimSeeds(
      target_points, proposal.sample, robust_top_z);
    rim_seeds.insert(rim_seeds.begin(), proposal.sample);
    for (const auto &rim_seed : rim_seeds) {
      proposal.sample = rim_seed;
      Eigen::Vector3f radial(
        proposal.sample.x() - object_centroid.x(),
        proposal.sample.y() - object_centroid.y(), 0.0F);
      if (radial.squaredNorm() > 1.0e-8F) {
        add_hypothesis(
          -Eigen::Vector3f::UnitZ(), radial,
          ZeroPoseHypothesis::Mode::kRim,
          proposal.sample.z() >= robust_top_z - rim_top_band_m ? 0.0F : 1.0F);
      }
    }
    proposal.sample = non_rim_sample;

    if (hypotheses.empty()) return false;
    const auto matches_requested_rim = [requested_affordance, robust_top_z](
        const ZeroPoseHypothesis &hypothesis) {
        if (requested_affordance != "RIM") return true;
        if (hypothesis.mode != ZeroPoseHypothesis::Mode::kRim &&
            hypothesis.mode != ZeroPoseHypothesis::Mode::kThinPca) {
          return false;
        }
        const float left_z = hypothesis.grasp.contact_left.z();
        const float right_z = hypothesis.grasp.contact_right.z();
        return std::abs(left_z - right_z) <= 0.008F &&
          robust_top_z - std::min(left_z, right_z) <= 0.025F;
      };
    const auto mode_available = [&hypotheses, &matches_requested_rim](
        ZeroPoseHypothesis::Mode mode) {
        return std::any_of(
          hypotheses.begin(), hypotheses.end(),
          [mode, &matches_requested_rim](const auto &hypothesis) {
            return hypothesis.mode == mode && hypothesis.cost < 1.0F &&
              matches_requested_rim(hypothesis);
          });
      };
    // Choose by observed topology, then compare costs only within that mode.
    // A 10 mm rim must never beat a valid 90 mm body grasp merely because its
    // numerical opening is smaller.
    ZeroPoseHypothesis::Mode preferred_mode =
      ZeroPoseHypothesis::Mode::kSide;
    if (requested_affordance == "RIM" && seed_is_at_top_edge &&
        mode_available(ZeroPoseHypothesis::Mode::kRim)) {
      preferred_mode = ZeroPoseHypothesis::Mode::kRim;
    } else if (requested_affordance == "RIM" &&
        mode_available(ZeroPoseHypothesis::Mode::kThinPca)) {
      preferred_mode = ZeroPoseHypothesis::Mode::kThinPca;
    } else if (requested_affordance == "RIM") {
      // A broad top-surface or side grasp does not satisfy a rim request.
      return false;
    } else if (seed_is_at_top_edge &&
        mode_available(ZeroPoseHypothesis::Mode::kTopSurface)) {
      preferred_mode = ZeroPoseHypothesis::Mode::kTopSurface;
    } else if (seed_is_at_top_edge &&
        mode_available(ZeroPoseHypothesis::Mode::kRim)) {
      preferred_mode = ZeroPoseHypothesis::Mode::kRim;
    } else if (silhouette_is_local_cross_section &&
        mode_available(ZeroPoseHypothesis::Mode::kThinPca)) {
      preferred_mode = ZeroPoseHypothesis::Mode::kThinPca;
    } else if (local_anisotropy >= 2.0F &&
        mode_available(ZeroPoseHypothesis::Mode::kThinPca)) {
      preferred_mode = ZeroPoseHypothesis::Mode::kThinPca;
    } else if (!mode_available(ZeroPoseHypothesis::Mode::kSide)) {
      preferred_mode = mode_available(ZeroPoseHypothesis::Mode::kThinPca) ?
        ZeroPoseHypothesis::Mode::kThinPca :
        ZeroPoseHypothesis::Mode::kRim;
    }
    const auto best_for_mode = [&hypotheses, &matches_requested_rim](
        ZeroPoseHypothesis::Mode mode) {
        auto best = hypotheses.end();
        for (auto candidate = hypotheses.begin();
             candidate != hypotheses.end(); ++candidate) {
          if (candidate->mode != mode ||
              !matches_requested_rim(*candidate)) continue;
          if (best == hypotheses.end() || candidate->cost < best->cost) {
            best = candidate;
          }
        }
        return best;
      };
    const auto primary = best_for_mode(preferred_mode);
    if (primary == hypotheses.end()) return false;

    // Preserve one best hypothesis per genuinely different local topology.
    // Previously every HGGD spatial seed collapsed to exactly one mode before
    // validation, so a failed side grasp could hide a reachable rim grasp (or
    // vice versa).  AUTO requests now send these distinct hypotheses through
    // the unchanged sweep, IK, collision, diversity, and ranking stages.
    std::vector<ZeroPoseHypothesis::Mode> modes{preferred_mode};
    if (requested_affordance == "AUTO") {
      for (const auto mode : {
          ZeroPoseHypothesis::Mode::kSide,
          ZeroPoseHypothesis::Mode::kRim,
          ZeroPoseHypothesis::Mode::kThinPca,
          ZeroPoseHypothesis::Mode::kTopSurface}) {
        if (std::find(modes.begin(), modes.end(), mode) == modes.end()) {
          modes.push_back(mode);
        }
      }
    }
    std::vector<GraspProposal> selected;
    selected.reserve(modes.size());
    for (const auto mode : modes) {
      const auto best = best_for_mode(mode);
      if (best == hypotheses.end()) continue;
      GraspProposal candidate = best->grasp;
      candidate.side_surface_grasp =
        mode == ZeroPoseHypothesis::Mode::kSide;
      candidate.rim_grasp = mode == ZeroPoseHypothesis::Mode::kRim;
      candidate.local_dominant_axis = dominant;
      candidate.has_local_dominant_axis = true;
      selected.push_back(std::move(candidate));
    }
    if (requested_affordance == "RIM" || requested_affordance == "AUTO") {
      for (const auto &hypothesis : hypotheses) {
        if (selected.size() >= 6) break;
        if (hypothesis.mode != ZeroPoseHypothesis::Mode::kRim ||
            !matches_requested_rim(hypothesis)) continue;
        bool distinct = true;
        for (const auto &existing : selected) {
          if (existing.rim_grasp &&
              (existing.sample - hypothesis.grasp.sample).norm() < 0.025F)
            distinct = false;
        }
        if (!distinct) continue;
        auto candidate = hypothesis.grasp;
        candidate.rim_grasp = true;
        candidate.side_surface_grasp = false;
        candidate.local_dominant_axis = dominant;
        candidate.has_local_dominant_axis = true;
        selected.push_back(std::move(candidate));
      }
    }
    if (selected.empty()) return false;
    proposal = selected.front();
    if (aligned_hypotheses != nullptr) {
      *aligned_hypotheses = std::move(selected);
    }
    return true;
  }

  bool pointAtPixel(const sensor_msgs::msg::PointCloud2 &cloud, int u, int v, Point &out) const {
    if (cloud.height <= 1 || u < 0 || v < 0 || u >= static_cast<int>(cloud.width) ||
        v >= static_cast<int>(cloud.height)) return false;
    const size_t index = static_cast<size_t>(v) * cloud.width + static_cast<size_t>(u);
    sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z(cloud, "z");
    x += index; y += index; z += index;
    out = {*x, *y, *z};
    if (!finitePoint(out)) return false;
    // Optical-frame Z is forward depth, but observation_cloud is normally
    // transformed into base_link, where Z is height.  Applying a camera-depth
    // threshold to base-frame Z removed the lower part of objects near the
    // table.  Invalid registered pixels are already represented by NaNs.
    if (cloud.header.frame_id == "base_link") return true;
    return out[2] > 0.05f;
  }

  bool maskAllows(const sensor_msgs::msg::Image &mask, int u, int v) const {
    if (mask.data.empty() || mask.width == 0 || mask.height == 0) return true;
    if (u < 0 || v < 0 || u >= static_cast<int>(mask.width) || v >= static_cast<int>(mask.height) ||
        mask.step < mask.width) return false;
    return mask.data[static_cast<size_t>(v) * mask.step + static_cast<size_t>(u)] != 0;
  }

  sensor_msgs::msg::Image buildConnectedTargetMask(
      const sensor_msgs::msg::PointCloud2 &cloud,
      const sensor_msgs::msg::Image &roi_mask,
      const Eigen::Vector3f &camera_origin,
      int &seed_u, int &seed_v, size_t &point_count) const {
    sensor_msgs::msg::Image filtered;
    filtered.header = roi_mask.header;
    filtered.height = cloud.height;
    filtered.width = cloud.width;
    filtered.encoding = "mono8";
    filtered.is_bigendian = false;
    filtered.step = cloud.width;
    filtered.data.assign(static_cast<size_t>(cloud.width) * cloud.height, 0);
    point_count = 0;

    const vamp_node::OrganizedXYZReader reader(cloud);
    const float maximum_step_sq = static_cast<float>(
      target_region_neighbor_distance_m_ * target_region_neighbor_distance_m_);
    int roi_min_u = static_cast<int>(cloud.width);
    int roi_min_v = static_cast<int>(cloud.height);
    int roi_max_u = -1;
    int roi_max_v = -1;
    for (int v = 0; v < static_cast<int>(cloud.height); ++v) {
      for (int u = 0; u < static_cast<int>(cloud.width); ++u) {
        if (!maskAllows(roi_mask, u, v)) continue;
        roi_min_u = std::min(roi_min_u, u);
        roi_min_v = std::min(roi_min_v, v);
        roi_max_u = std::max(roi_max_u, u);
        roi_max_v = std::max(roi_max_v, v);
      }
    }
    if (roi_max_u < roi_min_u || roi_max_v < roi_min_v) return filtered;

    struct Component {
      std::vector<size_t> pixels;
      std::vector<float> camera_ranges;
      int closest_distance_sq = std::numeric_limits<int>::max();
      int closest_u = 0;
      int closest_v = 0;
      int min_u = std::numeric_limits<int>::max();
      int min_v = std::numeric_limits<int>::max();
      int max_u = -1;
      int max_v = -1;
      Eigen::Vector3f minimum_point = Eigen::Vector3f::Constant(
        std::numeric_limits<float>::infinity());
      Eigen::Vector3f maximum_point = Eigen::Vector3f::Constant(
        -std::numeric_limits<float>::infinity());
      float median_camera_range = std::numeric_limits<float>::infinity();
    };
    std::vector<uint8_t> visited(filtered.data.size(), 0);
    std::vector<Component> components;
    constexpr int neighbor_offsets[8][2] = {
      {-1, -1}, {0, -1}, {1, -1}, {-1, 0},
      {1, 0}, {-1, 1}, {0, 1}, {1, 1}
    };
    for (int start_v = roi_min_v; start_v <= roi_max_v; ++start_v) {
      for (int start_u = roi_min_u; start_u <= roi_max_u; ++start_u) {
        const size_t start_index = static_cast<size_t>(start_v) * filtered.step + start_u;
        if (visited[start_index] || !maskAllows(roi_mask, start_u, start_v)) continue;
        Point start;
        if (!reader.read(start_u, start_v, start)) {
          visited[start_index] = 1;
          continue;
        }
        Component component;
        std::queue<std::pair<int, int>> frontier;
        frontier.push({start_u, start_v});
        visited[start_index] = 1;
        while (!frontier.empty()) {
          const auto [u, v] = frontier.front();
          frontier.pop();
          const size_t index = static_cast<size_t>(v) * filtered.step + u;
          component.pixels.push_back(index);
          component.min_u = std::min(component.min_u, u);
          component.min_v = std::min(component.min_v, v);
          component.max_u = std::max(component.max_u, u);
          component.max_v = std::max(component.max_v, v);
          const int pixel_distance_sq =
            (u - seed_u) * (u - seed_u) + (v - seed_v) * (v - seed_v);
          if (pixel_distance_sq < component.closest_distance_sq) {
            component.closest_distance_sq = pixel_distance_sq;
            component.closest_u = u;
            component.closest_v = v;
          }
          Point current;
          if (!reader.read(u, v, current)) continue;
          const Eigen::Vector3f current_point(current[0], current[1], current[2]);
          component.minimum_point = component.minimum_point.cwiseMin(current_point);
          component.maximum_point = component.maximum_point.cwiseMax(current_point);
          component.camera_ranges.push_back(
            (current_point - camera_origin).norm());
          for (const auto &offset : neighbor_offsets) {
            const int next_u = u + offset[0];
            const int next_v = v + offset[1];
            if (next_u < roi_min_u || next_v < roi_min_v ||
                next_u > roi_max_u || next_v > roi_max_v ||
                !maskAllows(roi_mask, next_u, next_v)) continue;
            const size_t next_index = static_cast<size_t>(next_v) * filtered.step + next_u;
            if (visited[next_index]) continue;
            Point next;
            if (!reader.read(next_u, next_v, next)) continue;
            const float dx = next[0] - current[0];
            const float dy = next[1] - current[1];
            const float dz = next[2] - current[2];
            if (dx * dx + dy * dy + dz * dz > maximum_step_sq) continue;
            visited[next_index] = 1;
            frontier.push({next_u, next_v});
          }
        }
        if (!component.camera_ranges.empty()) {
          const auto middle = component.camera_ranges.begin() +
            static_cast<std::ptrdiff_t>(component.camera_ranges.size() / 2);
          std::nth_element(
            component.camera_ranges.begin(), middle, component.camera_ranges.end());
          component.median_camera_range = *middle;
        }
        components.push_back(std::move(component));
      }
    }

    // EdgeSAM supplies an object silhouette, but missing depth can split a
    // curved or hollow object into several 3-D connected components. Prefer
    // the best-supported measured component. The seed remains a secondary
    // tie-breaker; otherwise a seed in a bowl opening can make a small front
    // fragment win over the main measured bowl surface.
    // A cluster that collapses to one 5 mm object-world voxel is depth noise,
    // not a usable 70 mm-long-finger target. Require both image and metric
    // extent before it can compete with a real handle surface.
    constexpr size_t minimum_foreground_component_pixels = 80;
    constexpr int minimum_foreground_component_span_pixels = 12;
    constexpr float minimum_foreground_component_extent_m = 0.020F;
    const auto geometrically_valid = [&](const Component &component) {
      if (!std::isfinite(component.median_camera_range) ||
          component.max_u < component.min_u ||
          component.max_v < component.min_v) {
        return false;
      }
      const int span_u = component.max_u - component.min_u + 1;
      const int span_v = component.max_v - component.min_v + 1;
      const float metric_extent =
        (component.maximum_point - component.minimum_point).norm();
      if ((span_u < minimum_foreground_component_span_pixels &&
           span_v < minimum_foreground_component_span_pixels) ||
          !std::isfinite(metric_extent) ||
          metric_extent < minimum_foreground_component_extent_m) {
        return false;
      }
      return true;
    };

    // Do not let the closest qualifying speckle win merely because it has 80
    // pixels.  Depth holes split glossy and curved objects differently on
    // every frame, so first derive a support threshold from the largest
    // meaningful component in this exact ROI.  Cap the adaptive part so a
    // thin handle can still compete with the background visible through it.
    size_t largest_component_pixels = 0;
    for (const auto &component : components) {
      if (geometrically_valid(component)) {
        largest_component_pixels = std::max(
          largest_component_pixels, component.pixels.size());
      }
    }
    const size_t adaptive_support = std::min<size_t>(
      500, static_cast<size_t>(std::ceil(
        static_cast<double>(largest_component_pixels) * 0.02)));
    const size_t required_component_pixels = std::max(
      minimum_foreground_component_pixels, adaptive_support);

    float nearest_supported_range = std::numeric_limits<float>::infinity();
    for (const auto &component : components) {
      if (geometrically_valid(component) &&
          component.pixels.size() >= required_component_pixels) {
        nearest_supported_range = std::min(
          nearest_supported_range, component.median_camera_range);
      }
    }

    auto anchor = components.end();
    constexpr float foreground_layer_tolerance_m = 0.012F;
    const float roi_diagonal = std::max(
      1.0F, std::hypot(
        static_cast<float>(roi_max_u - roi_min_u + 1),
        static_cast<float>(roi_max_v - roi_min_v + 1)));
    float best_anchor_score = -std::numeric_limits<float>::infinity();
    for (auto component = components.begin(); component != components.end();
         ++component) {
      if (!geometrically_valid(*component) ||
          component->pixels.size() < required_component_pixels) {
        continue;
      }
      const float support = std::sqrt(
        static_cast<float>(component->pixels.size()) /
        std::max(1.0F, static_cast<float>(largest_component_pixels)));
      const float seed_distance = std::sqrt(
        static_cast<float>(component->closest_distance_sq)) / roi_diagonal;
      const float depth_offset = std::max(
        0.0F, component->median_camera_range - nearest_supported_range) /
        foreground_layer_tolerance_m;
      const float score = 3.0F * support - 0.35F * seed_distance -
        0.05F * depth_offset;
      if (score > best_anchor_score) {
        best_anchor_score = score;
        anchor = component;
      }
    }
    if (anchor == components.end() ||
        !std::isfinite(anchor->median_camera_range)) {
      RCLCPP_WARN(
        this->get_logger(),
        "[OBJECT FILTER] No spatially valid foreground component in RoboBrain ROI; "
        "rejecting the crop instead of publishing a table/noise fallback.");
      return filtered;
    }
    const float seeded_surface_range = anchor->median_camera_range;

    seed_u = anchor->closest_u;
    seed_v = anchor->closest_v;
    std::vector<bool> accepted(components.size(),false);
    std::vector<Eigen::Vector3f> foreground_points;
    const auto accept_component = [&](size_t index) {
      accepted[index]=true;
      for (const size_t pixel:components[index].pixels) {
        filtered.data[pixel]=255;
        Point point;
        if (reader.read(static_cast<int>(pixel%cloud.width),static_cast<int>(pixel/cloud.width),point))
          foreground_points.emplace_back(point[0],point[1],point[2]);
      }
      point_count+=components[index].pixels.size();
    };
    for (size_t index=0;index<components.size();++index) {
      const auto &component=components[index];
      if (std::isfinite(component.median_camera_range) &&
          component.median_camera_range-seeded_surface_range<=static_cast<float>(target_foreground_band_m_))
        accept_component(index);
    }
    size_t recovered_components=0, recovered_points=0;
    // Expand only through substantial components inside the original semantic
    // mask. Each accepted addition must touch the measured foreground in 3-D;
    // retain the existing depth band for distant/unconnected background.
    for (size_t pass=0;pass<components.size();++pass) {
      const vamp_node::PlacementPointIndex foreground_index(foreground_points);
      std::vector<size_t> additions;
      for (size_t index=0;index<components.size();++index) {
        if (accepted[index] || !geometrically_valid(components[index]) ||
            components[index].pixels.size()<required_component_pixels) continue;
        std::vector<Eigen::Vector3f> measured;
        measured.reserve(components[index].pixels.size());
        for (const auto pixel:components[index].pixels) {
          Point point;
          if (reader.read(static_cast<int>(pixel%cloud.width),static_cast<int>(pixel/cloud.width),point))
            measured.emplace_back(point[0],point[1],point[2]);
        }
        if (vamp_node::substantialSurfaceConnected(measured,foreground_index,
              required_component_pixels,static_cast<float>(target_region_neighbor_distance_m_)))
          additions.push_back(index);
      }
      if (additions.empty()) break;
      for (const auto index:additions) {
        ++recovered_components;recovered_points+=components[index].pixels.size();accept_component(index);
      }
    }
    if (recovered_components) RCLCPP_INFO(get_logger(),
      "[OBJECT FILTER RECOVERY] spatially_connected_components=%zu measured_points=%zu bridge=%.3fm",
      recovered_components,recovered_points,target_region_neighbor_distance_m_);
    RCLCPP_INFO(
      this->get_logger(),
      "[OBJECT FILTER] components=%zu seeded_component=%zu px "
      "seed_distance=%.1f px range=%.3f m extent=%.3f m selection=foreground_depth "
      "selected=%zu px band=%.3f m support_threshold=%zu largest=%zu score=%.3f",
      components.size(), anchor->pixels.size(),
      std::sqrt(static_cast<float>(anchor->closest_distance_sq)),
      seeded_surface_range,
      (anchor->maximum_point - anchor->minimum_point).norm(), point_count,
      target_foreground_band_m_, required_component_pixels,
      largest_component_pixels, best_anchor_score);
    return filtered;
  }

  bool findCorrectedSeed(const sensor_msgs::msg::PointCloud2 &cloud,
                         const my_actions::srv::FindGraspCandidates::Request &request,
                         int &corrected_u, int &corrected_v, Point &seed, double &distance) const {
    const int max_radius = static_cast<int>(std::ceil(point_correction_limit_));
    for (int radius = 0; radius <= max_radius; ++radius) {
      for (int dy = -radius; dy <= radius; ++dy) for (int dx = -radius; dx <= radius; ++dx) {
        if (std::max(std::abs(dx), std::abs(dy)) != radius) continue;
        const int u = request.seed_u + dx, v = request.seed_v + dy;
        if (!maskAllows(request.target_mask, u, v) || !pointAtPixel(cloud, u, v, seed)) continue;
        Point left, right, up, down;
        const bool neighbors = pointAtPixel(cloud, u - 1, v, left) && pointAtPixel(cloud, u + 1, v, right) &&
                               pointAtPixel(cloud, u, v - 1, up) && pointAtPixel(cloud, u, v + 1, down);
        if (neighbors && (std::abs(left[2] - seed[2]) > 0.03f || std::abs(right[2] - seed[2]) > 0.03f ||
                          std::abs(up[2] - seed[2]) > 0.03f || std::abs(down[2] - seed[2]) > 0.03f)) continue;
        corrected_u = u; corrected_v = v; distance = std::hypot(dx, dy); return true;
      }
    }
    return false;
  }

  bool estimateLocalFrame(const sensor_msgs::msg::PointCloud2 &cloud,
                          const my_actions::srv::FindGraspCandidates::Request &request,
                          int seed_u, int seed_v, const Point &seed,
                          Eigen::Vector3f &normal, Eigen::Vector3f &tangent) const {
    std::vector<Eigen::Vector3f> points;
    const int radius_px = std::max(8, static_cast<int>(point_correction_limit_));
    const float radius_sq = static_cast<float>(local_crop_radius_ * local_crop_radius_);
    for (int v = std::max(0, seed_v - radius_px); v <= std::min(static_cast<int>(cloud.height) - 1, seed_v + radius_px); ++v) {
      for (int u = std::max(0, seed_u - radius_px); u <= std::min(static_cast<int>(cloud.width) - 1, seed_u + radius_px); ++u) {
        Point point;
        if (!maskAllows(request.target_mask, u, v) || !pointAtPixel(cloud, u, v, point)) continue;
        Eigen::Vector3f p(point[0], point[1], point[2]);
        if ((p - Eigen::Vector3f(seed[0], seed[1], seed[2])).squaredNorm() <= radius_sq) points.push_back(p);
      }
    }
    if (points.size() < 30) return false;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto &point : points) centroid += point;
    centroid /= static_cast<float>(points.size());
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto &point : points) { const Eigen::Vector3f d = point - centroid; covariance.noalias() += d * d.transpose(); }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) return false;
    normal = solver.eigenvectors().col(0).normalized();
    tangent = solver.eigenvectors().col(2).normalized();
    // The outward normal should face the camera origin.
    if (normal.dot(Eigen::Vector3f(seed[0], seed[1], seed[2])) > 0.0f) normal = -normal;
    tangent = (tangent - normal * tangent.dot(normal)).normalized();
    return normal.allFinite() && tangent.allFinite();
  }

  void publishObjectWorld(const sensor_msgs::msg::PointCloud2 &cloud,
                          const sensor_msgs::msg::Image &mask) {
    // ActiveVisionServer is the only coordinate transformer. It publishes an
    // organized observation cloud already transformed with FK * calibrated X.
    // Never mix that source with a TF lookup in the planner.
    if (cloud.header.frame_id != "base_link") {
      RCLCPP_ERROR(
        this->get_logger(),
        "[MANUAL X] Object world rejected: observation cloud frame is '%s', "
        "expected base_link.", cloud.header.frame_id.c_str());
      return;
    }
    std::vector<Point> points;
    points.reserve(static_cast<size_t>(cloud.width) * cloud.height / 8);
    size_t mask_pixels = 0;
    size_t valid_depth_pixels = 0;
    for (int v = 0; v < static_cast<int>(cloud.height); ++v) {
      for (int u = 0; u < static_cast<int>(cloud.width); ++u) {
        Point point;
        if (!maskAllows(mask, u, v)) continue;
        ++mask_pixels;
        if (!pointAtPixel(cloud, u, v, point)) continue;
        ++valid_depth_pixels;
        points.push_back(point);
      }
    }
    sensor_msgs::msg::PointCloud2 output;
    output.header.stamp = cloud.header.stamp;
    output.header.frame_id = "base_link";
    sensor_msgs::PointCloud2Modifier modifier(output);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> x(output, "x");
    sensor_msgs::PointCloud2Iterator<float> y(output, "y");
    sensor_msgs::PointCloud2Iterator<float> z(output, "z");
    for (const auto &point : points) {
      *x = point[0]; *y = point[1]; *z = point[2];
      ++x; ++y; ++z;
    }
    {
      std::lock_guard<std::mutex> lock(segmented_target_mutex_);
      latest_segmented_target_cloud_ =
        std::make_shared<sensor_msgs::msg::PointCloud2>(output);
    }
    object_world_pub_->publish(output);
    segmented_target_cloud_pub_->publish(output);
    RCLCPP_DEBUG(this->get_logger(),
                "[OBJECT WORLD] mask=%zu px, valid registered-depth=%zu px, published=%zu raw points; frame=%s. "
                "Missing depth/mask pixels—not a rigid base transform—cause partial geometry.",
                mask_pixels, valid_depth_pixels, points.size(), cloud.header.frame_id.c_str());
  }

  void publishSegmentedTargetImage(
      const sensor_msgs::msg::Image &image,
      const sensor_msgs::msg::Image &mask) {
    cv::Mat source;
    if (image.encoding == "bgr8" || image.encoding == "rgb8") {
      source = cv::Mat(
        static_cast<int>(image.height), static_cast<int>(image.width), CV_8UC3,
        const_cast<uint8_t *>(image.data.data()), image.step);
    } else {
      RCLCPP_ERROR(this->get_logger(),
                   "Segmented target image requires bgr8/rgb8 input, got %s.",
                   image.encoding.c_str());
      return;
    }
    cv::Mat binary(
      static_cast<int>(mask.height), static_cast<int>(mask.width), CV_8UC1,
      const_cast<uint8_t *>(mask.data.data()), mask.step);
    cv::Mat cropped = cv::Mat::zeros(source.size(), source.type());
    source.copyTo(cropped, binary);
    sensor_msgs::msg::Image output;
    output.header = image.header;
    output.height = image.height;
    output.width = image.width;
    output.encoding = image.encoding;
    output.is_bigendian = false;
    output.step = static_cast<uint32_t>(cropped.cols * cropped.elemSize());
    output.data.assign(cropped.datastart, cropped.dataend);
    {
      std::lock_guard<std::mutex> lock(segmented_target_mutex_);
      latest_segmented_target_image_ =
        std::make_shared<sensor_msgs::msg::Image>(output);
    }
    segmented_target_image_pub_->publish(output);

  }

  void publishGraspMarkers(
      const std::vector<my_actions::msg::GraspCandidate> &candidates,
      const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr &publisher = nullptr,
      const std::string &namespace_prefix = "grasp_candidate_",
      const bool pre_ranking = false) {
    visualization_msgs::msg::MarkerArray array;
    visualization_msgs::msg::Marker clear;
    clear.header.frame_id = "base_link";
    clear.header.stamp = this->now();
    clear.ns = "grasp_candidates";
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    array.markers.push_back(clear);
    for (const auto &candidate : candidates) {
      const std::string candidate_ns = namespace_prefix + std::to_string(candidate.id);
      const std::array<std::array<float, 3>, 5> colors{{
        {{0.10f, 0.90f, 0.20f}}, {{0.10f, 0.55f, 1.00f}}, {{1.00f, 0.65f, 0.05f}},
        {{0.85f, 0.15f, 0.90f}}, {{0.10f, 0.85f, 0.85f}}
      }};
      const auto &color = colors[(candidate.id - 1) % colors.size()];
      const Eigen::Quaternionf robot_orientation(
        static_cast<float>(candidate.grasp_pose.pose.orientation.w),
        static_cast<float>(candidate.grasp_pose.pose.orientation.x),
        static_cast<float>(candidate.grasp_pose.pose.orientation.y),
        static_cast<float>(candidate.grasp_pose.pose.orientation.z));
      // Render the actual IK pose. The pregrasp retreat and measured contact
      // line can differ from the final hand axes; using either here makes an
      // unsafe pose appear aligned in RViz.
      const Eigen::Matrix3f grasp_rotation =
        robot_orientation.normalized().toRotationMatrix();
      const Eigen::Vector3f approach_axis = grasp_rotation.col(2);
      const Eigen::Vector3f closing_axis = grasp_rotation.col(1);
      const Eigen::Quaternionf marker_orientation(
        visualGripperRotation(closing_axis, approach_axis));
      const Eigen::Vector3f tcp_position(
        static_cast<float>(candidate.grasp_pose.pose.position.x),
        static_cast<float>(candidate.grasp_pose.pose.position.y),
        static_cast<float>(candidate.grasp_pose.pose.position.z));
      const Eigen::Vector3f hand_base = tcp_position -
        approach_axis * static_cast<float>(grasp_tcp_offset_m_);
      for (int contact = 0; contact < 2; ++contact) {
        const auto &point = contact == 0 ? candidate.contact_left : candidate.contact_right;
        visualization_msgs::msg::Marker finger;
        finger.header = candidate.grasp_pose.header;
        finger.ns = candidate_ns;
        finger.id = static_cast<int>(candidate.id * 100 + contact);
        finger.type = visualization_msgs::msg::Marker::CUBE;
        finger.action = visualization_msgs::msg::Marker::ADD;
        finger.pose = candidate.grasp_pose.pose;
        finger.pose.orientation.x = marker_orientation.x();
        finger.pose.orientation.y = marker_orientation.y();
        finger.pose.orientation.z = marker_orientation.z();
        finger.pose.orientation.w = marker_orientation.w();
        // jaw_opening_m is the inner-face separation. Place each marker's
        // centre another half finger-width outward so the rendered solid does
        // not penetrate the object merely because it is centred on contact.
        const float side = contact == 0 ? 1.0F : -1.0F;
        const Eigen::Vector3f finger_center = hand_base +
          approach_axis * static_cast<float>(finger_length_m_ * 0.5) +
          side * closing_axis * static_cast<float>(
            candidate.jaw_opening_m * 0.5 + finger_width_m_ * 0.5);
        finger.pose.position.x = finger_center.x();
        finger.pose.position.y = finger_center.y();
        finger.pose.position.z = finger_center.z();
        finger.scale.x = finger_width_m_;
        finger.scale.y = finger_height_m_;
        finger.scale.z = finger_length_m_;
        finger.color.r = color[0]; finger.color.g = color[1]; finger.color.b = color[2];
        finger.color.a = pre_ranking ? 0.25F : 0.55F;
        array.markers.push_back(finger);

        visualization_msgs::msg::Marker marker = finger;
        marker.id = static_cast<int>(candidate.id * 100 + 10 + contact);
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.pose.orientation.w = 1.0;
        marker.pose.position = point;
        marker.scale.x = marker.scale.y = marker.scale.z = 0.012;
        marker.color.r = 1.0f; marker.color.g = 0.2f; marker.color.b = 0.1f;
        marker.color.a = pre_ranking ? 0.45F : 1.0F;
        array.markers.push_back(marker);
      }
      // Accepted candidates also show the actual IK-resolved pre-grasp hand.
      // A translucent pair distinguishes it from the solid final grasp. Raw
      // backend proposal layers remain uncluttered because they have their
      // own explicit publisher.
      if (!publisher) {
        const Eigen::Quaternionf pregrasp_robot_orientation(
          static_cast<float>(candidate.pregrasp_pose.pose.orientation.w),
          static_cast<float>(candidate.pregrasp_pose.pose.orientation.x),
          static_cast<float>(candidate.pregrasp_pose.pose.orientation.y),
          static_cast<float>(candidate.pregrasp_pose.pose.orientation.z));
        const Eigen::Matrix3f pregrasp_rotation =
          pregrasp_robot_orientation.normalized().toRotationMatrix();
        const Eigen::Vector3f pregrasp_closing = pregrasp_rotation.col(1);
        const Eigen::Vector3f pregrasp_approach = pregrasp_rotation.col(2);
        const Eigen::Quaternionf pregrasp_marker_orientation(
          visualGripperRotation(pregrasp_closing, pregrasp_approach));
        const Eigen::Vector3f pregrasp_tcp(
          static_cast<float>(candidate.pregrasp_pose.pose.position.x),
          static_cast<float>(candidate.pregrasp_pose.pose.position.y),
          static_cast<float>(candidate.pregrasp_pose.pose.position.z));
        const Eigen::Vector3f pregrasp_hand_base = pregrasp_tcp -
          pregrasp_approach * static_cast<float>(grasp_tcp_offset_m_);
        const float pregrasp_opening = static_cast<float>(std::min(
          maximum_gripper_width_,
          static_cast<double>(candidate.jaw_opening_m) +
            pregrasp_opening_clearance_m_));
        for (int finger_index = 0; finger_index < 2; ++finger_index) {
          const float side = finger_index == 0 ? 1.0F : -1.0F;
          visualization_msgs::msg::Marker pregrasp_finger;
          pregrasp_finger.header = candidate.pregrasp_pose.header;
          pregrasp_finger.ns = candidate_ns;
          pregrasp_finger.id = static_cast<int>(
            candidate.id * 100 + 30 + finger_index);
          pregrasp_finger.type = visualization_msgs::msg::Marker::CUBE;
          pregrasp_finger.action = visualization_msgs::msg::Marker::ADD;
          pregrasp_finger.pose = candidate.pregrasp_pose.pose;
          pregrasp_finger.pose.orientation.x = pregrasp_marker_orientation.x();
          pregrasp_finger.pose.orientation.y = pregrasp_marker_orientation.y();
          pregrasp_finger.pose.orientation.z = pregrasp_marker_orientation.z();
          pregrasp_finger.pose.orientation.w = pregrasp_marker_orientation.w();
          const Eigen::Vector3f center = pregrasp_hand_base +
            pregrasp_approach * static_cast<float>(finger_length_m_ * 0.5) +
            side * pregrasp_closing *
              (pregrasp_opening * 0.5F +
               static_cast<float>(finger_width_m_ * 0.5));
          pregrasp_finger.pose.position.x = center.x();
          pregrasp_finger.pose.position.y = center.y();
          pregrasp_finger.pose.position.z = center.z();
          pregrasp_finger.scale.x = finger_width_m_;
          pregrasp_finger.scale.y = finger_height_m_;
          pregrasp_finger.scale.z = finger_length_m_;
          pregrasp_finger.color.r = 0.75F;
          pregrasp_finger.color.g = 0.90F;
          pregrasp_finger.color.b = 1.00F;
          pregrasp_finger.color.a = 0.30F;
          array.markers.push_back(pregrasp_finger);
        }
      }
      visualization_msgs::msg::Marker approach_marker;
      approach_marker.header = candidate.grasp_pose.header;
      approach_marker.ns = candidate_ns;
      approach_marker.id = static_cast<int>(candidate.id * 100 + 20);
      approach_marker.type = visualization_msgs::msg::Marker::ARROW;
      approach_marker.action = visualization_msgs::msg::Marker::ADD;
      approach_marker.points = {candidate.pregrasp_pose.pose.position, candidate.grasp_pose.pose.position};
      approach_marker.scale.x = 0.006; approach_marker.scale.y = 0.012; approach_marker.scale.z = 0.018;
      approach_marker.color.r = color[0]; approach_marker.color.g = color[1];
      approach_marker.color.b = color[2];
      approach_marker.color.a = pre_ranking ? 0.45F : 1.0F;
      array.markers.push_back(approach_marker);
      visualization_msgs::msg::Marker label;
      label.header = candidate.grasp_pose.header;
      label.ns = candidate_ns;
      label.action = visualization_msgs::msg::Marker::ADD;
      label.pose = candidate.grasp_pose.pose;
      label.id = static_cast<int>(candidate.id * 100 + 21);
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.pose.orientation.x = 0.0;
      label.pose.orientation.y = 0.0;
      label.pose.orientation.z = 0.0;
      label.pose.orientation.w = 1.0;
      label.pose.position.z += 0.045;
      label.scale.z = 0.025;
      label.color.r = color[0]; label.color.g = color[1]; label.color.b = color[2];
      label.color.a = pre_ranking ? 0.65F : 1.0F;
      label.text = (pre_ranking ? "V" : "#") +
        std::to_string(candidate.id);
      if (!publisher) {
        const bool aligned = candidate.validation_status.find(
          "PAD_SURFACE_ALIGNMENT_VALIDATED") != std::string::npos;
        const bool fallback = candidate.validation_status.find(
          "ANGLE_TUNING_FAILED_ORIGINAL_USED") != std::string::npos;
        const float degrees = 180.0F / static_cast<float>(M_PI);
        const float roll = std::atan2(grasp_rotation(2,1), grasp_rotation(2,2));
        const float pitch = std::atan2(-grasp_rotation(2,0),
          std::hypot(grasp_rotation(0,0), grasp_rotation(1,0)));
        const float yaw = std::atan2(grasp_rotation(1,0), grasp_rotation(0,0));
        std::ostringstream text;
        text << label.text << "\n" << (aligned ? "Surface aligned" :
          fallback ? "Original pose (angle fallback)" : "Angle tuning not reported")
          << "\nRPY " << std::fixed << std::setprecision(1)
          << roll*degrees << ", " << pitch*degrees << ", " << yaw*degrees << " deg";
        label.text = text.str();
        label.scale.z = 0.014;
        for (int axis=0; axis<3; ++axis) {
          auto axes = approach_marker;
          axes.id = static_cast<int>(candidate.id*100+40+axis);
          axes.points.clear();
          axes.points.push_back(candidate.grasp_pose.pose.position);
          auto end = candidate.grasp_pose.pose.position;
          const Eigen::Vector3f direction = grasp_rotation.col(axis)*.04F;
          end.x += direction.x(); end.y += direction.y(); end.z += direction.z();
          axes.points.push_back(end);
          axes.scale.x = .002; axes.scale.y = .004; axes.scale.z = .007;
          axes.color.r = axis==0 ? 1.F : 0.F;
          axes.color.g = axis==1 ? 1.F : 0.F;
          axes.color.b = axis==2 ? 1.F : 0.F;
          axes.color.a = 1.F;
          array.markers.push_back(axes);
        }
      }
      if (namespace_prefix == "angle_trial_") {
        label.text = candidate.validation_status;
        label.scale.z = .011;
        label.pose.position.z += .025 * static_cast<double>((candidate.id-1)%3);
      }
      array.markers.push_back(label);
    }
    (publisher ? publisher : grasp_markers_pub_)->publish(array);
  }

  struct HandleOrientation {
    Eigen::Vector3f approach = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f closing = Eigen::Vector3f::UnitY();
    float roll_alignment_cost = 1.0F;
    bool constrained_seed = false;
  };

  // A single RGB-D view does not expose opposing contact normals reliably.
  // For cup-body grasps, fit the visible depth arc to a cylinder cross-section
  // and infer the two side contacts from that conservative primitive.
  struct BodyGraspModel {
    Eigen::Vector3f center;
    Eigen::Vector3f body_axis;
    Eigen::Vector3f closing_axis;
    Eigen::Vector3f surface_normal;
    float radius = 0.0f;
  };

  bool fitCupBodyModel(const std::vector<Eigen::Vector3f> &points,
                       const Eigen::Vector3f &seed,
                       const Eigen::Vector3f &surface_normal,
                       const Eigen::Vector3f &body_axis_hint,
                       BodyGraspModel &model) const {
    model.surface_normal = surface_normal.normalized();
    model.body_axis = (body_axis_hint - model.surface_normal *
        body_axis_hint.dot(model.surface_normal)).normalized();
    model.closing_axis = model.body_axis.cross(model.surface_normal).normalized();
    if (!model.surface_normal.allFinite() || !model.body_axis.allFinite() ||
        !model.closing_axis.allFinite()) return false;

    std::vector<Eigen::Vector2f> arc;
    arc.reserve(points.size());
    constexpr float body_band_half_height_m = 0.018f;
    constexpr float body_search_radius_m = 0.090f;
    for (const auto &point : points) {
      const Eigen::Vector3f delta = point - seed;
      if (std::abs(delta.dot(model.body_axis)) > body_band_half_height_m ||
          delta.norm() > body_search_radius_m) continue;
      arc.emplace_back(delta.dot(model.closing_axis), delta.dot(model.surface_normal));
    }
    if (arc.size() < 40) return false;

    const auto fit_circle = [](const std::vector<Eigen::Vector2f> &samples,
                               Eigen::Vector2f &center, float &radius) {
      Eigen::MatrixXf a(samples.size(), 3);
      Eigen::VectorXf b(samples.size());
      for (size_t i = 0; i < samples.size(); ++i) {
        const float u = samples[i].x(), v = samples[i].y();
        a(i, 0) = u; a(i, 1) = v; a(i, 2) = 1.0f;
        b(i) = -(u * u + v * v);
      }
      const Eigen::Vector3f coeff = a.colPivHouseholderQr().solve(b);
      center = Eigen::Vector2f(-0.5f * coeff.x(), -0.5f * coeff.y());
      const float radius_sq = center.squaredNorm() - coeff.z();
      if (!std::isfinite(radius_sq) || radius_sq <= 0.0f) return false;
      radius = std::sqrt(radius_sq);
      return center.allFinite() && std::isfinite(radius);
    };

    Eigen::Vector2f center_2d;
    float radius = 0.0f;
    if (!fit_circle(arc, center_2d, radius)) return false;
    std::vector<Eigen::Vector2f> inliers;
    inliers.reserve(arc.size());
    for (const auto &sample : arc) {
      if (std::abs((sample - center_2d).norm() - radius) <= 0.008f) inliers.push_back(sample);
    }
    if (inliers.size() < 30 || !fit_circle(inliers, center_2d, radius)) return false;
    if (radius < 0.012f || radius > 0.047f || 2.0f * radius > maximum_gripper_width_) return false;
    model.center = seed + model.closing_axis * center_2d.x() + model.surface_normal * center_2d.y();
    model.radius = radius;
    return model.center.allFinite();
  }

  bool buildCupBodyThicknessPair(
      const std::vector<Eigen::Vector3f> &points,
      const Eigen::Vector3f &seed,
      const Eigen::Vector3f &camera_origin,
      float closing_yaw_radians,
      AntipodalContacts &pair) const {
    Eigen::Vector3f surface_normal;
    if (!estimateObjectNormal(points, seed, surface_normal)) return false;
    // The depth camera observes the near side of the cup.  Keep the fitted
    // cylinder normal on that side; PCA supplies an unsigned normal.
    if (surface_normal.dot(camera_origin - seed) < 0.0F) {
      surface_normal = -surface_normal;
    }
    BodyGraspModel model;
    if (!fitCupBodyModel(
          points, seed, surface_normal, Eigen::Vector3f::UnitZ(), model)) {
      return false;
    }
    // A single near-side depth arc tends to fit slightly inside the physical
    // silhouette.  Expand only this model-derived radius by 5 mm so an
    // observed outer-body point is treated as an intended pad contact rather
    // than a false finger-sweep collision.  The normal gripper-width limit
    // remains the hard safety bound.
    constexpr float body_arc_surface_margin_m = 0.005F;
    const float contact_radius = model.radius + body_arc_surface_margin_m;
    const Eigen::Vector3f closing_axis =
      Eigen::AngleAxisf(closing_yaw_radians, model.body_axis) *
      model.closing_axis;
    if (!closing_axis.allFinite() || closing_axis.squaredNorm() < 1.0e-6F) {
      return false;
    }
    pair.left = model.center + closing_axis * contact_radius;
    pair.right = model.center - closing_axis * contact_radius;
    pair.opening = 2.0F * contact_radius;
    if (!std::isfinite(pair.opening) ||
        pair.opening < minimum_gripper_width_ ||
        pair.opening > maximum_gripper_width_) {
      return false;
    }
    pair.closing_axis = closing_axis.normalized();
    pair.quality = 0.002F;  // Prefer direct silhouette edges when available.
    pair.body_model_supported = true;
    return pair.left.allFinite() && pair.right.allFinite();
  }

  struct RejectedGraspPose {
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    Eigen::Vector3f closing = Eigen::Vector3f::UnitY();
    Eigen::Vector3f approach = Eigen::Vector3f::UnitZ();
    float opening = 0.060f;
    std::string reason;
  };

  void publishRejectedGraspMarkers(const std::vector<RejectedGraspPose> &rejected) {
    visualization_msgs::msg::MarkerArray array;
    visualization_msgs::msg::Marker clear;
    clear.header.frame_id = "base_link";
    clear.header.stamp = this->now();
    clear.ns = "grasp_debug_rejected";
    // RViz checks namespace/ID uniqueness even for DELETEALL entries.
    // Keep this control marker distinct from rejected pose index zero.
    clear.id = -1;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    array.markers.push_back(clear);
    for (size_t index = 0; index < rejected.size(); ++index) {
      const auto &entry = rejected[index];
      const std::string candidate_ns = "grasp_debug_rejected_" + std::to_string(index + 1);
      Eigen::Quaternionf q(visualGripperRotation(entry.closing, entry.approach));
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "base_link";
      marker.header.stamp = this->now();
      marker.ns = candidate_ns;
      marker.id = static_cast<int>(index * 10);
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = entry.pose.translation().x();
      marker.pose.position.y = entry.pose.translation().y();
      marker.pose.position.z = entry.pose.translation().z();
      marker.pose.orientation.x = q.x(); marker.pose.orientation.y = q.y();
      marker.pose.orientation.z = q.z(); marker.pose.orientation.w = q.w();
      marker.scale.x = entry.opening + finger_width_m_ * 2.0;
      marker.scale.y = finger_height_m_;
      marker.scale.z = finger_length_m_;
      marker.color.r = 1.0f; marker.color.g = 0.15f; marker.color.b = 0.05f; marker.color.a = 0.35f;
      array.markers.push_back(marker);
      visualization_msgs::msg::Marker label = marker;
      label.id += 1;
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.pose.orientation.w = 1.0;
      label.pose.position.z += 0.05;
      label.scale.z = 0.022;
      label.color.r = 1.0f; label.color.g = 0.8f; label.color.b = 0.1f; label.color.a = 1.0f;
      label.text = "R" + std::to_string(index + 1) + ": " + entry.reason;
      array.markers.push_back(label);
    }
    grasp_debug_markers_pub_->publish(array);
  }

  std::vector<Eigen::Vector3f> collectMaskedObjectPoints(
      const sensor_msgs::msg::PointCloud2 &cloud,
      const sensor_msgs::msg::Image &mask) const {
    std::vector<Eigen::Vector3f> points;
    points.reserve(static_cast<size_t>(cloud.width) * cloud.height / 8);
    for (int v = 0; v < static_cast<int>(cloud.height); ++v) {
      for (int u = 0; u < static_cast<int>(cloud.width); ++u) {
        Point point;
        if (!maskAllows(mask, u, v) || !pointAtPixel(cloud, u, v, point)) continue;
        points.emplace_back(point[0], point[1], point[2]);
      }
    }
    // The organized cloud is dense enough that this cap preserves 3-D shape
    // while bounding the repeated local-normal calculations below.
    constexpr size_t max_points = 12000;
    if (points.size() > max_points) {
      std::vector<Eigen::Vector3f> reduced;
      reduced.reserve(max_points);
      const double stride = static_cast<double>(points.size()) / max_points;
      for (double index = 0.0; index < points.size(); index += stride) {
        reduced.push_back(points[static_cast<size_t>(index)]);
      }
      return reduced;
    }
    return points;
  }

  bool estimateObjectNormal(const std::vector<Eigen::Vector3f> &points,
                            const Eigen::Vector3f &query,
                            Eigen::Vector3f &normal) const {
    const float normal_radius_m = static_cast<float>(
      antipodal_normal_radius_m_);
    const float radius_sq = normal_radius_m * normal_radius_m;
    std::vector<Eigen::Vector3f> neighborhood;
    neighborhood.reserve(96);
    for (const auto &point : points) {
      if ((point - query).squaredNorm() <= radius_sq) neighborhood.push_back(point);
    }
    if (neighborhood.size() < static_cast<size_t>(
          antipodal_min_normal_neighbors_)) return false;
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto &point : neighborhood) center += point;
    center /= static_cast<float>(neighborhood.size());
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto &point : neighborhood) {
      const Eigen::Vector3f delta = point - center;
      covariance.noalias() += delta * delta.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) return false;
    normal = solver.eigenvectors().col(0).normalized();
    // PCA normals have an arbitrary sign.  Do not orient them from the ROI
    // centroid: an affordance can be non-convex and its centroid can lie in
    // empty space (for example, the hole inside a cup handle).
    return normal.allFinite();
  }

  bool buildSilhouetteThicknessPair(
      const sensor_msgs::msg::PointCloud2 &cloud,
      const sensor_msgs::msg::Image &mask,
      int seed_u, int seed_v,
      AntipodalContacts &pair) const {
    if (!maskAllows(mask, seed_u, seed_v)) return false;

    // Estimate the local 2-D material tangent from mask pixels. The minor PCA
    // axis crosses the narrow branch and therefore intersects its two edges.
    constexpr int pca_radius_px = 20;
    std::vector<Eigen::Vector2f> pixels;
    pixels.reserve(1200);
    Eigen::Vector2f center = Eigen::Vector2f::Zero();
    for (int dv = -pca_radius_px; dv <= pca_radius_px; ++dv) {
      for (int du = -pca_radius_px; du <= pca_radius_px; ++du) {
        if (du * du + dv * dv > pca_radius_px * pca_radius_px) continue;
        const int pixel_u = seed_u + du;
        const int pixel_v = seed_v + dv;
        if (!maskAllows(mask, pixel_u, pixel_v)) continue;
        const Eigen::Vector2f pixel(
          static_cast<float>(pixel_u), static_cast<float>(pixel_v));
        pixels.push_back(pixel);
        center += pixel;
      }
    }
    if (pixels.size() < 30) return false;
    center /= static_cast<float>(pixels.size());
    Eigen::Matrix2f covariance = Eigen::Matrix2f::Zero();
    for (const auto &pixel : pixels) {
      const Eigen::Vector2f delta = pixel - center;
      covariance.noalias() += delta * delta.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(covariance);
    if (solver.info() != Eigen::Success) return false;
    const Eigen::Vector2f eigenvalues = solver.eigenvalues();
    if (eigenvalues.x() <= 1.0e-5F ||
        eigenvalues.y() / eigenvalues.x() <
          static_cast<float>(silhouette_min_tangent_anisotropy_)) {
      // A circular/corner neighborhood has no stable local tangent. Reject it
      // instead of choosing an arbitrary minor axis that can span the handle
      // hole or cup body.
      return false;
    }
    Eigen::Vector2f crossing = solver.eigenvectors().col(0).normalized();
    if (!crossing.allFinite()) return false;

    // Walk from the semantic material point to each edge of the same
    // contiguous mask interval. Two consecutive misses prevent crossing the
    // handle hole and accidentally selecting the other branch.
    const auto trace_edge = [&](float direction, float &last_inside_step) {
      last_inside_step = 0.0F;
      int misses = 0;
      int previous_u = seed_u;
      int previous_v = seed_v;
      constexpr float maximum_trace_px = 40.0F;
      constexpr float trace_step_px = 0.5F;
      for (float step = trace_step_px; step <= maximum_trace_px;
           step += trace_step_px) {
        const int pixel_u = static_cast<int>(std::lround(
          seed_u + direction * crossing.x() * step));
        const int pixel_v = static_cast<int>(std::lround(
          seed_v + direction * crossing.y() * step));
        if (pixel_u == previous_u && pixel_v == previous_v) continue;
        previous_u = pixel_u;
        previous_v = pixel_v;
        if (maskAllows(mask, pixel_u, pixel_v)) {
          last_inside_step = step;
          misses = 0;
        } else if (++misses >= 2) {
          break;
        }
      }
      // A corrected semantic seed may already lie on one silhouette edge;
      // zero travel on that side is valid if the opposite trace establishes
      // the required total material thickness below.
      return true;
    };
    float positive_step = 0.0F;
    float negative_step = 0.0F;
    if (!trace_edge(1.0F, positive_step) ||
        !trace_edge(-1.0F, negative_step) ||
        positive_step + negative_step < 4.0F) return false;

    const auto lift_edge = [&](float direction, float edge_step,
                               Eigen::Vector3f &base_point) {
      // Silhouette depth can be invalid at the final pixel. Search a few
      // pixels inward while remaining on the grounded material mask.
      for (float inward = 0.0F; inward <= 4.0F; inward += 0.5F) {
        const float step = std::max(0.0F, edge_step - inward);
        const int pixel_u = static_cast<int>(std::lround(
          seed_u + direction * crossing.x() * step));
        const int pixel_v = static_cast<int>(std::lround(
          seed_v + direction * crossing.y() * step));
        if (!maskAllows(mask, pixel_u, pixel_v)) continue;
        Point cloud_point;
        if (!pointAtPixel(cloud, pixel_u, pixel_v, cloud_point)) continue;
        base_point = Eigen::Vector3f(
          cloud_point[0], cloud_point[1], cloud_point[2]);
        return base_point.allFinite();
      }
      return false;
    };

    if (!lift_edge(1.0F, positive_step, pair.left) ||
        !lift_edge(-1.0F, negative_step, pair.right)) return false;
    const Eigen::Vector3f delta = pair.left - pair.right;
    pair.opening = delta.norm();
    if (!std::isfinite(pair.opening) ||
        pair.opening < minimum_gripper_width_ ||
        pair.opening > maximum_gripper_width_) return false;
    pair.closing_axis = delta / pair.opening;
    // Both silhouette edges are observed in the registered RGB-D mask. Give
    // this direct local-width measurement priority over noisier PCA pairs;
    // Gripper-frame compatibility and the sweep check still gate acceptance.
    pair.quality = 0.0F;
    pair.silhouette_supported = true;
    return true;
  }

  std::vector<AntipodalContacts> buildLocalAntipodalPairs(
      const std::vector<Eigen::Vector3f> &points,
      const Eigen::Vector3f &semantic_seed,
      size_t &local_surface_count) const {
    struct SurfacePoint {
      Eigen::Vector3f point;
      Eigen::Vector3f normal;
    };
    struct VoxelPoint {
      Eigen::Vector3f point = Eigen::Vector3f::Zero();
      int count = 0;
    };

    // One representative per depth voxel avoids treating dense neighboring
    // pixels on one visible face as many independent contact surfaces.
    constexpr float contact_voxel_m = 0.003F;
    const float search_radius_sq = static_cast<float>(
      antipodal_seed_search_radius_m_ * antipodal_seed_search_radius_m_);
    std::unordered_map<std::tuple<int, int, int>, VoxelPoint, VoxelHash> voxels;
    for (const auto &point : points) {
      if ((point - semantic_seed).squaredNorm() > search_radius_sq) continue;
      const auto key = std::make_tuple(
        static_cast<int>(std::floor(point.x() / contact_voxel_m)),
        static_cast<int>(std::floor(point.y() / contact_voxel_m)),
        static_cast<int>(std::floor(point.z() / contact_voxel_m)));
      auto &voxel = voxels[key];
      voxel.point += point;
      ++voxel.count;
    }

    std::vector<Eigen::Vector3f> representatives;
    representatives.reserve(voxels.size());
    for (const auto &entry : voxels) {
      if (entry.second.count > 0) {
        representatives.push_back(
          entry.second.point / static_cast<float>(entry.second.count));
      }
    }
    // A uniform stride over voxel representatives retains spatial coverage;
    // taking only the nearest pixels can discard the opposite surface.
    if (representatives.size() > static_cast<size_t>(
          antipodal_maximum_local_points_)) {
      std::vector<Eigen::Vector3f> reduced;
      reduced.reserve(static_cast<size_t>(antipodal_maximum_local_points_));
      const double stride = static_cast<double>(representatives.size()) /
        static_cast<double>(antipodal_maximum_local_points_);
      for (double index = 0.0; index < representatives.size(); index += stride) {
        reduced.push_back(representatives[static_cast<size_t>(index)]);
      }
      representatives = std::move(reduced);
    }

    std::vector<SurfacePoint> surfaces;
    surfaces.reserve(representatives.size());
    for (const auto &point : representatives) {
      Eigen::Vector3f normal;
      if (estimateObjectNormal(points, point, normal)) {
        surfaces.push_back({point, normal});
      }
    }
    local_surface_count = surfaces.size();

    std::vector<AntipodalContacts> pairs;
    const float required_cosine = static_cast<float>(antipodal_normal_cosine_);
    // Two contacts closer than the normal-estimation radius use overlapping
    // neighborhoods and cannot prove two independent surfaces. This also
    // rejects RGB-D thickness artifacts near the 3 mm voxel resolution.
    const float minimum_reliable_opening = std::max(
      static_cast<float>(minimum_gripper_width_),
      static_cast<float>(antipodal_normal_radius_m_));
    for (size_t left_index = 0; left_index < surfaces.size(); ++left_index) {
      const auto &left = surfaces[left_index];
      for (size_t right_index = left_index + 1;
           right_index < surfaces.size(); ++right_index) {
        const auto &right = surfaces[right_index];
        const Eigen::Vector3f delta = left.point - right.point;
        const float opening = delta.norm();
        if (opening < minimum_reliable_opening ||
            opening > maximum_gripper_width_) continue;
        const Eigen::Vector3f pair_axis = delta / opening;
        // PCA normal signs are arbitrary, so compare absolute alignment. A
        // true thickness pair has two locally parallel planes whose common
        // normal lies along the segment between the contacts.
        const float left_alignment = std::abs(left.normal.dot(pair_axis));
        const float right_alignment = std::abs(right.normal.dot(pair_axis));
        const float plane_parallelism = std::abs(left.normal.dot(right.normal));
        if (left_alignment < required_cosine ||
            right_alignment < required_cosine ||
            plane_parallelism < required_cosine) continue;
        const Eigen::Vector3f midpoint = (left.point + right.point) * 0.5F;
        const float seed_distance = (midpoint - semantic_seed).norm();
        if (seed_distance > antipodal_seed_search_radius_m_) continue;
        AntipodalContacts pair;
        pair.left = left.point;
        pair.right = right.point;
        pair.closing_axis = pair_axis;
        pair.opening = opening;
        // Prefer a thin local cross-section near the semantic material seed.
        // This is a continuous ranking term rather than an object-specific
        // hard aperture cutoff, so thicker novel objects remain eligible.
        pair.quality = opening + 0.5F * seed_distance +
          0.005F * (3.0F - left_alignment - right_alignment - plane_parallelism);
        pairs.push_back(pair);
      }
    }
    std::stable_sort(
      pairs.begin(), pairs.end(), [](const auto &left, const auto &right) {
        return left.quality < right.quality;
      });
    constexpr size_t maximum_pair_pool = 1024;
    if (pairs.size() > maximum_pair_pool) pairs.resize(maximum_pair_pool);
    return pairs;
  }

  bool selectAntipodalContacts(
      const std::vector<AntipodalContacts> &pairs,
      const Eigen::Vector3f &proposed_closing,
      const Eigen::Vector3f &proposed_approach,
      AntipodalContacts &contacts,
      std::string &failure_reason) const {
    failure_reason.clear();
    if (pairs.empty()) {
      failure_reason = "no_local_thickness_pair";
      return false;
    }
    const float required_cosine = static_cast<float>(antipodal_normal_cosine_);
    constexpr float maximum_approach_alignment = 0.35F;
    float best_quality = std::numeric_limits<float>::infinity();
    bool closing_match_found = false;
    for (const auto &pair : pairs) {
      const float signed_closing_alignment =
        pair.closing_axis.dot(proposed_closing);
      const float closing_alignment = std::abs(signed_closing_alignment);
      if (closing_alignment < required_cosine) continue;
      closing_match_found = true;
      if (!pair.silhouette_supported &&
          std::abs(pair.closing_axis.dot(proposed_approach)) >
          maximum_approach_alignment) continue;
      const float quality = pair.quality + 0.01F * (1.0F - closing_alignment);
      if (quality >= best_quality) continue;
      contacts = pair;
      if (signed_closing_alignment < 0.0F) {
        std::swap(contacts.left, contacts.right);
        contacts.closing_axis = -contacts.closing_axis;
      }
      best_quality = quality;
    }
    if (!std::isfinite(best_quality)) {
      failure_reason = closing_match_found ?
        "approach_not_orthogonal" : "closing_axis_mismatch";
      return false;
    }
    contacts.quality = best_quality;
    return true;
  }

  std::vector<HandleOrientation> buildVisibleHandleApproaches(
      const Eigen::Vector3f &closing_axis,
      const Eigen::Vector3f &visible_approach_hint) const {
    std::vector<HandleOrientation> orientations;
    Eigen::Vector3f visible = visible_approach_hint -
      closing_axis * visible_approach_hint.dot(closing_axis);
    if (visible.squaredNorm() < 1.0e-6F) {
      return orientations;
    }
    visible.normalize();
    Eigen::Vector3f nominal_closing = closing_axis.normalized();
    Eigen::Vector3f nominal_height = visible.cross(nominal_closing);
    if (nominal_height.squaredNorm() < 1.0e-6F) return orientations;
    nominal_height.normalize();

    // Thirteen approach families: frontal, six yaw-only, and six pitch-only.
    // Avoiding the yaw/pitch Cartesian product keeps this bounded.
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> approach_frames;
    approach_frames.emplace_back(visible, nominal_closing);
    for (const float angle_degrees : {-30.0F, -20.0F, -10.0F,
                                      10.0F, 20.0F, 30.0F}) {
      const float radians = angle_degrees * static_cast<float>(M_PI / 180.0);
      const Eigen::AngleAxisf yaw(radians, nominal_height);
      approach_frames.emplace_back(
        (yaw * visible).normalized(), (yaw * nominal_closing).normalized());
      const Eigen::AngleAxisf pitch(radians, nominal_closing);
      approach_frames.emplace_back(
        (pitch * visible).normalized(), nominal_closing);
    }

    for (const auto &[base_approach, base_closing] : approach_frames) {
      std::vector<HandleOrientation> roll_hypotheses;
      roll_hypotheses.reserve(static_cast<size_t>(
        2.0 * handle_approach_roll_limit_degrees_ /
          handle_approach_roll_step_degrees_ + 1.0));
      for (double roll_degrees = -handle_approach_roll_limit_degrees_;
           roll_degrees <= handle_approach_roll_limit_degrees_ + 1.0e-6;
           roll_degrees += handle_approach_roll_step_degrees_) {
        const Eigen::AngleAxisf roll(
          static_cast<float>(roll_degrees * M_PI / 180.0), base_approach);
        Eigen::Vector3f candidate_closing = roll * base_closing;
        candidate_closing -= base_approach *
          candidate_closing.dot(base_approach);
        if (candidate_closing.squaredNorm() < 1.0e-6F) continue;
        candidate_closing.normalize();
        const float alignment = std::abs(
          candidate_closing.dot(nominal_closing));
        roll_hypotheses.push_back({
          base_approach, candidate_closing, 1.0F - alignment});
      }
      std::stable_sort(
        roll_hypotheses.begin(), roll_hypotheses.end(),
        [](const auto &left, const auto &right) {
          return left.roll_alignment_cost < right.roll_alignment_cost;
        });
      const size_t keep = std::min<size_t>(3, roll_hypotheses.size());
      orientations.insert(
        orientations.end(), roll_hypotheses.begin(),
        roll_hypotheses.begin() + keep);
    }
    return orientations;
  }

  std::vector<HandleOrientation> buildDominantAxisOrbitApproaches(
      const Eigen::Vector3f &dominant_axis,
      const Eigen::Vector3f &frontal_approach) const {
    std::vector<HandleOrientation> orientations;
    if (dominant_axis.squaredNorm() < 1.0e-8F ||
        frontal_approach.squaredNorm() < 1.0e-8F) {
      return orientations;
    }
    const Eigen::Vector3f dominant = dominant_axis.normalized();
    Eigen::Vector3f frontal = frontal_approach -
      dominant * frontal_approach.dot(dominant);
    if (frontal.squaredNorm() < 1.0e-8F) return orientations;
    frontal.normalize();

    // Search the aligned zero pose first, then move symmetrically away from
    // the camera-frontal approach. Rotating the whole grasp frame around the
    // local tangent keeps the gripper pad axis parallel to the affordance.
    constexpr std::array<float, 19> orbit_degrees = {
      0.0F, -10.0F, 10.0F, -20.0F, 20.0F,
      -30.0F, 30.0F, -40.0F, 40.0F, -50.0F,
      50.0F, -60.0F, 60.0F, -70.0F, 70.0F,
      -80.0F, 80.0F, -90.0F, 90.0F};
    orientations.reserve(orbit_degrees.size());
    for (const float degrees : orbit_degrees) {
      const Eigen::AngleAxisf rotation(
        degrees * static_cast<float>(M_PI / 180.0), dominant);
      Eigen::Vector3f approach = (rotation * frontal).normalized();
      Eigen::Vector3f closing = dominant.cross(approach);
      if (closing.squaredNorm() < 1.0e-8F) continue;
      closing.normalize();
      orientations.push_back({
        approach, closing, std::abs(degrees) / 90.0F});
    }
    return orientations;
  }

  bool fingerVolumesAndClosingSweepClear(const std::vector<Eigen::Vector3f> &points,
                                         const AntipodalContacts &contacts,
                                         const Eigen::Vector3f &closing,
                                         const Eigen::Vector3f &height_axis,
                                         const Eigen::Vector3f &approach,
                                         std::array<float, 3> *blocked_local = nullptr,
                                         const Eigen::Isometry3f *actual_pose = nullptr) const {
    const Eigen::Vector3f midpoint =
      (contacts.left + contacts.right) * 0.5F;
    const float half_height = static_cast<float>(finger_height_m_ * 0.5);
    const float final_inner_half = contacts.opening * 0.5F;
    // The physical routine opens only a small clearance beyond the measured
    // object thickness; it does not start every grasp at the 100 mm maximum.
    // Validate the volume actually occupied by that commanded motion.
    const float pregrasp_inner_half = static_cast<float>(
      std::min(maximum_gripper_width_,
        static_cast<double>(contacts.opening) + pregrasp_opening_clearance_m_) * 0.5);
    const float pregrasp_outer_half = pregrasp_inner_half +
      static_cast<float>(finger_width_m_);
    // Registered RGB-D edges commonly fluctuate by about a millimetre. This
    // allowance applies only to points from the isolated target; external
    // environment collision remains strict in the downstream VAMP checks.
    constexpr float target_edge_noise_margin_m = 0.001F;
    const float contact_allowance = static_cast<float>(
      target_contact_surface_tolerance_m_) + target_edge_noise_margin_m;
    const float depth_behind_contact = actual_pose ? 0.F : static_cast<float>(grasp_contact_depth_m_);
    const float depth_ahead_contact = actual_pose ? static_cast<float>(finger_length_m_) :
      static_cast<float>(finger_length_m_ - grasp_contact_depth_m_);
    struct BlockingPoint {
      Eigen::Vector3f point = Eigen::Vector3f::Zero();
      std::array<float, 3> local{};
    };
    std::vector<BlockingPoint> blocking_points;
    blocking_points.reserve(64);
    for (const auto &point : points) {
      const Eigen::Vector3f local = point - midpoint;
      Eigen::Vector3f physical=Eigen::Vector3f::Zero();
      if (actual_pose) {
        physical=actual_pose->linear().transpose()*(point-actual_pose->translation());
        physical.z()+=static_cast<float>(grasp_tcp_offset_m_);
      }
      const float height = actual_pose ? std::abs(physical.x()) : std::abs(local.dot(height_axis));
      const float depth = actual_pose ? physical.z() : local.dot(approach);
      if (height > half_height || depth < -depth_behind_contact ||
          depth > depth_ahead_contact) continue;
      const float absolute_side = actual_pose ? std::abs(physical.y()) : std::abs(local.dot(closing));
      // Points at the two final inner pad surfaces are intended contact.
      // Anything farther outward lies in a finger body or in the volume that
      // a finger sweeps through from the commanded pre-grasp aperture.
      if (absolute_side > final_inner_half + contact_allowance &&
          absolute_side <= pregrasp_outer_half) {
        blocking_points.push_back({
          point, {absolute_side, height, depth}});
      }
    }
    // Do not reject a grasp because of one noisy depth-edge sample. Require a
    // compact patch of target points occupying the swept volume. Eight points
    // within a 4 mm neighbourhood remains sensitive to a real handle/body
    // junction while ignoring isolated flying pixels and registration noise.
    constexpr size_t minimum_blocking_cluster = 8;
    constexpr float blocking_neighbor_radius_m = 0.004F;
    constexpr float blocking_neighbor_radius_sq =
      blocking_neighbor_radius_m * blocking_neighbor_radius_m;
    if (blocking_points.size() < minimum_blocking_cluster) return true;
    for (size_t anchor = 0; anchor < blocking_points.size(); ++anchor) {
      size_t local_support = 0;
      for (const auto &candidate : blocking_points) {
        if ((candidate.point - blocking_points[anchor].point).squaredNorm() <=
            blocking_neighbor_radius_sq &&
            ++local_support >= minimum_blocking_cluster) {
          if (blocked_local != nullptr) {
            *blocked_local = blocking_points[anchor].local;
          }
          return false;
        }
      }
    }
    return true;
  }

  bool pinchSectionAwayFromJunction(
      const std::vector<Eigen::Vector3f> &target_points,
      const AntipodalContacts &contacts,
      const Eigen::Vector3f &closing,
      const Eigen::Vector3f &height_axis,
      const Eigen::Vector3f &approach) const {
    if (contacts.grasp_skill != LocalGraspSkill::kPinch) return true;
    const Eigen::Vector3f midpoint =
      (contacts.left + contacts.right) * 0.5F;
    const auto section_width = [&](float height_offset) -> std::optional<float> {
        std::vector<float> lateral;
        lateral.reserve(128);
        constexpr float half_band_m = 0.005F;
        constexpr float half_depth_m = 0.030F;
        for (const auto &point : target_points) {
          const Eigen::Vector3f local = point - midpoint;
          if (std::abs(local.dot(height_axis) - height_offset) > half_band_m ||
              std::abs(local.dot(approach)) > half_depth_m) {
            continue;
          }
          const float side = local.dot(closing);
          if (std::abs(side) <= static_cast<float>(maximum_gripper_width_)) {
            lateral.push_back(side);
          }
        }
        if (lateral.size() < 10) return std::nullopt;
        std::sort(lateral.begin(), lateral.end());
        const size_t trim = lateral.size() >= 40 ? lateral.size() / 40 : 0;
        return lateral[lateral.size() - 1 - trim] - lateral[trim];
      };

    // A usable thin-feature grasp needs material support on both sides along
    // the pad axis. At a handle/body seam either support disappears or its
    // cross-section expands sharply into the body.
    constexpr float probe_offset_m = 0.012F;
    const auto negative = section_width(-probe_offset_m);
    const auto positive = section_width(probe_offset_m);
    if (!negative || !positive) return false;
    const float allowed_neighbor_width = std::min(
      static_cast<float>(maximum_gripper_width_),
      std::max(contacts.opening * 1.8F, contacts.opening + 0.015F));
    return *negative <= allowed_neighbor_width &&
      *positive <= allowed_neighbor_width;
  }

  bool contactSurfaceNormalsSupportPinch(
      const std::vector<Eigen::Vector3f> &target_points,
      const AntipodalContacts &contacts,
      const Eigen::Vector3f &closing, bool strict = false) const {
    if (!strict && contacts.grasp_skill != LocalGraspSkill::kPinch) return true;
    Eigen::Vector3f left_normal;
    Eigen::Vector3f right_normal;
    if (!estimateObjectNormal(target_points, contacts.left, left_normal) ||
        !estimateObjectNormal(target_points, contacts.right, right_normal)) {
      return false;
    }
    const Eigen::Vector3f midpoint =
      0.5F * (contacts.left + contacts.right);
    if (left_normal.dot(contacts.left - midpoint) < 0.0F) {
      left_normal = -left_normal;
    }
    if (right_normal.dot(contacts.right - midpoint) < 0.0F) {
      right_normal = -right_normal;
    }
    const Eigen::Vector3f jaw_axis = closing.normalized();
    // A real pinch needs two locally opposed surfaces. A contact pair drawn
    // diagonally across one continuous mug wall has nearly parallel normals
    // and must not be accepted merely because its endpoints fit between the
    // jaws. Keep a moderate angular allowance for RGB-D normal noise.
    if (strict) return padNormalsAligned(left_normal, right_normal, jaw_axis,
      static_cast<float>(antipodal_normal_cosine_));
    const float minimum_facing_cosine = strict ?
      static_cast<float>(antipodal_normal_cosine_) : 0.65F;
    const float maximum_normal_dot = strict ?
      -static_cast<float>(antipodal_normal_cosine_) : -0.30F;
    return left_normal.dot(jaw_axis) >= minimum_facing_cosine &&
      right_normal.dot(jaw_axis) <= -minimum_facing_cosine &&
      left_normal.dot(right_normal) <= maximum_normal_dot;
  }

  void handleRefineGrasp(
    const std::shared_ptr<my_actions::srv::RefineGrasp::Request> request,
    std::shared_ptr<my_actions::srv::RefineGrasp::Response> response)
  {
    response->candidate_id = request->candidate.id;
    sensor_msgs::msg::PointCloud2::SharedPtr cloud;
    sensor_msgs::msg::Image::SharedPtr image;
    geometry_msgs::msg::TransformStamped::SharedPtr camera_transform;
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
    std::vector<float> joints;
    {
      std::lock_guard<std::mutex> lock(observation_mutex_);
      cloud = latest_registered_cloud_;
      if (cloud) {
        const rclcpp::Time stamp(cloud->header.stamp);
        for (auto entry = observation_image_cache_.rbegin();
          entry != observation_image_cache_.rend(); ++entry)
        {
          if (rclcpp::Time((*entry)->header.stamp) == stamp) {
            image = *entry;
            break;
          }
        }
        for (auto entry = observation_camera_transform_cache_.rbegin();
          entry != observation_camera_transform_cache_.rend(); ++entry)
        {
          if (rclcpp::Time((*entry)->header.stamp) == stamp) {
            camera_transform = *entry;
            break;
          }
        }
      }
      camera_info = latest_camera_info_;
      joints = latest_joints_;
    }
    const auto fail = [&response](const std::string & reason) {
        response->valid = false;
        response->failure_reason = reason;
      };
    if (!cloud || !image || !camera_transform || !camera_info) {
      std::ostringstream reason;
      reason << "SYNCHRONIZED_PREGRASP_RGBD_UNAVAILABLE missing=";
      bool first = true;
      const auto append_missing = [&reason, &first](const char * name) {
          if (!first) reason << ',';
          reason << name;
          first = false;
        };
      if (!cloud) append_missing("cloud");
      if (!image) append_missing("matching_rgb");
      if (!camera_transform) append_missing("matching_manual_x_pose");
      if (!camera_info) append_missing("camera_info");
      fail(reason.str());
      return;
    }
    const rclcpp::Time current_stamp(cloud->header.stamp);
    if (current_stamp == rclcpp::Time(request->previous_observation_header.stamp)) {
      fail("NO_NEW_PREGRASP_OBSERVATION");
      return;
    }
    if (cloud->height <= 1 || cloud->header.frame_id != "base_link" ||
      image->width != cloud->width || image->height != cloud->height ||
      camera_transform->header.frame_id != "base_link" ||
      camera_info->width != cloud->width || camera_info->height != cloud->height)
    {
      fail("PREGRASP_OBSERVATION_CONTRACT_MISMATCH");
      return;
    }
    if (joints.size() != 6) {
      fail("CURRENT_JOINT_STATE_UNAVAILABLE");
      return;
    }

    const auto & candidate = request->candidate;
    const Eigen::Vector3f original_seed(
      static_cast<float>(0.5 * (candidate.contact_left.x + candidate.contact_right.x)),
      static_cast<float>(0.5 * (candidate.contact_left.y + candidate.contact_right.y)),
      static_cast<float>(0.5 * (candidate.contact_left.z + candidate.contact_right.z)));
    if (!original_seed.allFinite()) {
      fail("INVALID_HGGD_CONTACT_CENTER");
      return;
    }
    std::unordered_set<std::tuple<int, int, int>, VoxelHash> target_support;
    std::vector<Eigen::Vector3f> cached_contact_region;
    {
      std::lock_guard<std::mutex> lock(refinement_target_mutex_);
      if (refinement_target_stamp_ns_ ==
        rclcpp::Time(request->previous_observation_header.stamp).nanoseconds())
      {
        target_support = refinement_target_support_voxels_;
        cached_contact_region = refinement_contact_region_;
      }
    }
    if (target_support.empty()) {
      fail("ORIGINAL_GROUNDED_TARGET_SUPPORT_UNAVAILABLE");
      return;
    }
    const auto & transform = camera_transform->transform;
    Eigen::Quaternionf camera_quaternion(
      static_cast<float>(transform.rotation.w),
      static_cast<float>(transform.rotation.x),
      static_cast<float>(transform.rotation.y),
      static_cast<float>(transform.rotation.z));
    if (camera_quaternion.squaredNorm() < 1.0e-8F) {
      fail("INVALID_MANUAL_X_CAMERA_ROTATION");
      return;
    }
    camera_quaternion.normalize();
    Eigen::Isometry3f base_from_camera = Eigen::Isometry3f::Identity();
    base_from_camera.linear() = camera_quaternion.toRotationMatrix();
    base_from_camera.translation() = Eigen::Vector3f(
      static_cast<float>(transform.translation.x),
      static_cast<float>(transform.translation.y),
      static_cast<float>(transform.translation.z));
    CameraIntrinsics intrinsics{
      static_cast<float>(camera_info->k[0]), static_cast<float>(camera_info->k[4]),
      static_cast<float>(camera_info->k[2]), static_cast<float>(camera_info->k[5])};
    int projected_u = -1;
    int projected_v = -1;
    if (!LocalGraspRefiner::projectBasePoint(
        original_seed, base_from_camera, intrinsics,
        static_cast<int>(cloud->width), static_cast<int>(cloud->height),
        projected_u, projected_v))
    {
      fail("HGGD_SEED_OUTSIDE_CURRENT_CAMERA_FRAME");
      return;
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<Eigen::Vector3f> organized(
      static_cast<std::size_t>(cloud->width) * cloud->height,
      Eigen::Vector3f(nan, nan, nan));
    std::vector<Eigen::Vector3f> current_observation_points;
    current_observation_points.reserve(
      static_cast<std::size_t>(cloud->width) * cloud->height);
    std::size_t supported_points = 0;
    for (int v = 0; v < static_cast<int>(cloud->height); ++v) {
      for (int u = 0; u < static_cast<int>(cloud->width); ++u) {
        Point point;
        if (pointAtPixel(*cloud, u, v, point)) {
          current_observation_points.emplace_back(point[0], point[1], point[2]);
          const auto key = std::make_tuple(
            static_cast<int>(std::floor(
              point[0] / refinement_target_support_voxel_m_)),
            static_cast<int>(std::floor(
              point[1] / refinement_target_support_voxel_m_)),
            static_cast<int>(std::floor(
              point[2] / refinement_target_support_voxel_m_)));
          bool target_supported = false;
          for (int dz = -1; dz <= 1 && !target_supported; ++dz) {
            for (int dy = -1; dy <= 1 && !target_supported; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                if (target_support.count(std::make_tuple(
                    std::get<0>(key) + dx, std::get<1>(key) + dy,
                    std::get<2>(key) + dz)) > 0)
                {
                  target_supported = true;
                  break;
                }
              }
            }
          }
          if (target_supported) {
            organized[static_cast<std::size_t>(v * cloud->width + u)] =
              Eigen::Vector3f(point[0], point[1], point[2]);
            ++supported_points;
          }
        }
      }
    }
    // The second view must be allowed to contribute surfaces hidden in the
    // first observation. Extract its local sphere from the complete organized
    // observation, not only from voxels present in the old grounded support.
    // Both clouds are already in base_link via joint FK * calibrated X.
    const auto current_contact_region = LocalGraspRefiner::extractContactRegion(
      current_observation_points, original_seed,
      refinement_contact_region_radius_m_);
    const auto merged_contact_region = LocalGraspRefiner::mergeContactRegions(
      cached_contact_region, current_contact_region,
      refinement_contact_merge_voxel_m_);
    LocalGraspRefinement refinement;
    if (!cached_contact_region.empty() &&
      merged_contact_region.size() >= static_cast<std::size_t>(
        local_refiner_parameters_.minimum_region_points))
    {
      refinement = local_grasp_refiner_->refinePointRegion(
        merged_contact_region, original_seed, base_from_camera.translation());
      refinement.seed_u = projected_u;
      refinement.seed_v = projected_v;
    } else if (supported_points >= static_cast<std::size_t>(
        local_refiner_parameters_.minimum_region_points))
    {
      refinement = local_grasp_refiner_->refine(
        organized, static_cast<int>(cloud->width), static_cast<int>(cloud->height),
        original_seed, projected_u, projected_v, base_from_camera.translation());
    } else {
      fail("INSUFFICIENT_REOBSERVED_GROUNDED_TARGET_SUPPORT");
      return;
    }
    if (refinement.valid) {
      int centered_u = -1;
      int centered_v = -1;
      if (LocalGraspRefiner::projectBasePoint(
          refinement.seed, base_from_camera, intrinsics,
          static_cast<int>(cloud->width), static_cast<int>(cloud->height),
          centered_u, centered_v))
      {
        refinement.seed_u = centered_u;
        refinement.seed_v = centered_v;
      }
    }
    response->observation_header = cloud->header;
    response->seed_u = refinement.seed_u;
    response->seed_v = refinement.seed_v;
    response->corrected_seed_base.x = refinement.seed.x();
    response->corrected_seed_base.y = refinement.seed.y();
    response->corrected_seed_base.z = refinement.seed.z();
    response->closing_axis_base.x = refinement.closing_axis.x();
    response->closing_axis_base.y = refinement.closing_axis.y();
    response->closing_axis_base.z = refinement.closing_axis.z();
    response->approach_axis_base.x = refinement.approach_axis.x();
    response->approach_axis_base.y = refinement.approach_axis.y();
    response->approach_axis_base.z = refinement.approach_axis.z();
    response->grasp_width_m = refinement.width_m;
    response->usable_grasp_depth_m = refinement.usable_depth_m;

    int left_u = -1;
    int left_v = -1;
    int right_u = -1;
    int right_v = -1;
    bool grasp_line_visible = false;
    if (refinement.width_m > 0.0F) {
      const Eigen::Vector3f half_line =
        refinement.closing_axis * (0.5F * refinement.width_m);
      const bool left_projected = LocalGraspRefiner::projectBasePoint(
        refinement.seed + half_line, base_from_camera, intrinsics,
        static_cast<int>(cloud->width), static_cast<int>(cloud->height),
        left_u, left_v);
      const bool right_projected = LocalGraspRefiner::projectBasePoint(
        refinement.seed - half_line, base_from_camera, intrinsics,
        static_cast<int>(cloud->width), static_cast<int>(cloud->height),
        right_u, right_v);
      constexpr int line_margin_px = 8;
      grasp_line_visible = left_projected && right_projected &&
        left_u >= line_margin_px && right_u >= line_margin_px &&
        left_v >= line_margin_px && right_v >= line_margin_px &&
        left_u < static_cast<int>(cloud->width) - line_margin_px &&
        right_u < static_cast<int>(cloud->width) - line_margin_px &&
        left_v < static_cast<int>(cloud->height) - line_margin_px &&
        right_v < static_cast<int>(cloud->height) - line_margin_px;
      if (left_u >= 0 && left_v >= 0 && right_u >= 0 && right_v >= 0) {
        response->closing_angle_rad = std::atan2(
          static_cast<float>(right_v - left_v),
          static_cast<float>(right_u - left_u));
      }
    }
    response->left_u = left_u;
    response->left_v = left_v;
    response->right_u = right_u;
    response->right_v = right_v;
    if (refinement.valid && !grasp_line_visible) {
      refinement.valid = false;
      refinement.failure_reason = "GRASP_LINE_NOT_FULLY_VISIBLE";
    }

    cv::Mat debug;
    if (image->encoding == "bgr8" || image->encoding == "rgb8") {
      cv::Mat wrapped(
        static_cast<int>(image->height), static_cast<int>(image->width), CV_8UC3,
        image->data.data(), image->step);
      debug = wrapped.clone();
      if (image->encoding == "rgb8") {
        cv::cvtColor(debug, debug, cv::COLOR_RGB2BGR);
      }
    }
    if (!refinement.valid) {
      fail(refinement.failure_reason);
    } else {
      const Eigen::Vector3f hand_base = refinement.seed -
        refinement.approach_axis * static_cast<float>(grasp_contact_depth_m_);
      const Eigen::Vector3f tcp = hand_base +
        refinement.approach_axis * static_cast<float>(grasp_tcp_offset_m_);
      Eigen::Isometry3f target_pose = Eigen::Isometry3f::Identity();
      target_pose.translation() = tcp;
      // This is a newly reconstructed RGB-D target, not an already corrected
      // cached pose. Final-approach planning consumes it without adding again.
      target_pose = vamp_node::correctedGraspTarget(target_pose, grasp_position_offset_base_m_);
      std::vector<float> grasp_joints;
      PoseIkDiagnostics best_diagnostics;
      float best_upright_score = -std::numeric_limits<float>::infinity();
      bool found_refined_ik = false;
      // The local grasp line is symmetric. Test both equivalent finger
      // assignments and retain the reachable solution that keeps the
      // wrist-mounted camera most upright.
      for (const bool swap_fingers : {false, true}) {
        Eigen::Isometry3f branch_pose = target_pose;
        branch_pose.linear() = piperGripperRotation(
          refinement.closing_axis, refinement.approach_axis, swap_fingers);
        std::vector<float> branch_joints;
        PoseIkDiagnostics branch_diagnostics;
        if (!solvePiperPoseIK(
            branch_pose, joints, branch_joints, branch_diagnostics))
        {
          best_diagnostics.keepBest(branch_diagnostics);
          continue;
        }
        best_diagnostics.keepBest(branch_diagnostics);
        const float upright_score = cameraUprightScore(
          joints, branch_joints, *camera_transform).value_or(-1.0F);
        if (!found_refined_ik || upright_score > best_upright_score) {
          target_pose = branch_pose;
          grasp_joints = std::move(branch_joints);
          best_upright_score = upright_score;
          found_refined_ik = true;
        }
      }
      if (!found_refined_ik) {
        std::ostringstream reason;
        reason << "STRICT_FINAL_IK_FAILED position="
               << best_diagnostics.best_position_error << " rotation="
               << best_diagnostics.best_rotation_error;
        fail(reason.str());
      } else {
        response->valid = true;
        response->failure_reason = "OK";
        response->grasp_pose.header = cloud->header;
        response->grasp_pose.header.frame_id = "base_link";
        response->grasp_pose.pose.position.x = tcp.x();
        response->grasp_pose.pose.position.y = tcp.y();
        response->grasp_pose.pose.position.z = tcp.z();
        const Eigen::Quaternionf quaternion(target_pose.rotation());
        response->grasp_pose.pose.orientation.x = quaternion.x();
        response->grasp_pose.pose.orientation.y = quaternion.y();
        response->grasp_pose.pose.orientation.z = quaternion.z();
        response->grasp_pose.pose.orientation.w = quaternion.w();
        std::copy_n(grasp_joints.begin(), 6, response->grasp_joints.begin());
        // Retain both synchronized base_link views for the final swept-volume
        // check.  This is the same manual FK * calibrated-X data used by the
        // refiner; no TF lookup or alternate transform path is introduced.
        std::lock_guard<std::mutex> lock(refinement_target_mutex_);
        refinement_contact_region_ = merged_contact_region;
      }
    }

    if (!debug.empty()) {
      const cv::Scalar color = response->valid ?
        cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
      // Yellow cross: the old-view HGGD contact center reprojected through the
      // synchronized manual-X camera pose. Orange dot: the nearest measured
      // surface pixel. The green/red circle and line are the centered local
      // contact geometry used for the refined grasp.
      if (projected_u >= 0 && projected_v >= 0) {
        cv::drawMarker(debug, cv::Point(projected_u, projected_v),
          cv::Scalar(0, 255, 255), cv::MARKER_TILTED_CROSS, 13, 2);
      }
      if (refinement.snapped_u >= 0 && refinement.snapped_v >= 0) {
        cv::circle(debug,
          cv::Point(refinement.snapped_u, refinement.snapped_v), 3,
          cv::Scalar(0, 165, 255), cv::FILLED);
      }
      if (refinement.seed_u >= 0 && refinement.seed_v >= 0) {
        cv::circle(debug, cv::Point(refinement.seed_u, refinement.seed_v), 6, color, 2);
      }
      if (grasp_line_visible) {
        cv::line(debug, cv::Point(left_u, left_v),
          cv::Point(right_u, right_v), color, 3);
      } else if (left_u >= 0 && left_v >= 0 && right_u >= 0 && right_v >= 0) {
        cv::Point clipped_left(left_u, left_v);
        cv::Point clipped_right(right_u, right_v);
        if (cv::clipLine(debug.size(), clipped_left, clipped_right)) {
          cv::line(debug, clipped_left, clipped_right,
            cv::Scalar(0, 0, 255), 2);
        }
      }
      std::ostringstream label;
      label << (response->valid ? "VALID " : "INVALID ")
            << std::fixed << std::setprecision(1)
            << refinement.width_m * 1000.0F << "mm depth="
            << refinement.usable_depth_m * 1000.0F << "mm snap="
            << refinement.snap_distance_m * 1000.0F << "mm";
      cv::putText(debug, label.str(), cv::Point(10, 28),
        cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 2);
      if (!response->valid) {
        cv::putText(debug, response->failure_reason, cv::Point(10, 54),
          cv::FONT_HERSHEY_SIMPLEX, 0.52, color, 2);
      }
      sensor_msgs::msg::Image debug_message;
      debug_message.header = cloud->header;
      debug_message.height = static_cast<uint32_t>(debug.rows);
      debug_message.width = static_cast<uint32_t>(debug.cols);
      debug_message.encoding = "bgr8";
      debug_message.step = static_cast<uint32_t>(debug.cols * 3);
      debug_message.data.assign(debug.datastart, debug.dataend);
      refined_grasp_debug_pub_->publish(debug_message);
    }
    RCLCPP_INFO(
      this->get_logger(),
      "[GRASP REFINE] candidate=%u valid=%s pixel=(%d,%d) region=%zu "
      "support=%zu regions={cached:%zu,current:%zu,merged:%zu} "
      "hggd=(%d,%d) snapped=(%d,%d) snap=%.1fmm search=%s "
      "width=%.1fmm usable_depth=%.1fmm reason=%s",
      candidate.id, response->valid ? "true" : "false",
      response->seed_u, response->seed_v, refinement.region_points,
      supported_points,
      cached_contact_region.size(), current_contact_region.size(),
      merged_contact_region.size(),
      projected_u, projected_v, refinement.snapped_u, refinement.snapped_v,
      refinement.snap_distance_m * 1000.0F,
      refinement.used_global_target_search ? "target-3d" : "local-pixel",
      response->grasp_width_m * 1000.0F,
      response->usable_grasp_depth_m * 1000.0F,
      response->failure_reason.c_str());
  }

  void handlePlanPlacement(
    const std::shared_ptr<my_actions::srv::PlanPlacement::Request> request,
    std::shared_ptr<my_actions::srv::PlanPlacement::Response> response)
  {
    PipelineProfile profile(pipeline_profiling_, "placement", [this](const std::string &data) {
      RCLCPP_INFO(this->get_logger(), "[PIPELINE PROFILE] %s", data.c_str());
    });
    const auto placement_started=std::chrono::steady_clock::now();
    response->success = false;
    response->status = "PLACEMENT_PLAN_FAILED";
    {
      std::lock_guard<std::mutex> lock(placement_plan_mutex_);
      cached_placement_plan_.reset();
    }
    visualization_msgs::msg::MarkerArray clear_markers;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    clear_markers.markers.push_back(clear);
    placement_plan_markers_pub_->publish(clear_markers);
    sensor_msgs::msg::PointCloud2 clear_preview;
    clear_preview.header.frame_id = "base_link";
    clear_preview.header.stamp = now();
    sensor_msgs::PointCloud2Modifier clear_modifier(clear_preview);
    clear_modifier.setPointCloud2FieldsByString(1, "xyz");
    clear_modifier.resize(0);
    refined_placed_object_cloud_pub_->publish(clear_preview);
    refined_placed_support_surface_pub_->publish(clear_preview);

    if (request->candidates.empty() && !request->lift_only) {
      response->status = "PLACEMENT_PLAN_NO_CANDIDATES";
      return;
    }
    if ((request->lift_only && (!request->plan_pick_place_motion || !request->use_layout_scene)) ||
        !validLiftDistance(request->lift_distance_m)) {
      response->status = "INVALID_LIFT_REQUEST";
      return;
    }
    if (request->grasp_candidate.grasp_pose.header.frame_id != "base_link") {
      response->status = "PLACEMENT_GRASP_FRAME_MUST_BE_BASE_LINK";
      return;
    }
    const auto pose_from_message = [](
      const geometry_msgs::msg::PoseStamped &message,
      Eigen::Isometry3f &pose) {
        const auto &orientation = message.pose.orientation;
        Eigen::Quaternionf quaternion(
          static_cast<float>(orientation.w),
          static_cast<float>(orientation.x),
          static_cast<float>(orientation.y),
          static_cast<float>(orientation.z));
        if (!quaternion.coeffs().allFinite() ||
          quaternion.squaredNorm() < 1.0e-8F) return false;
        quaternion.normalize();
        pose = Eigen::Isometry3f::Identity();
        pose.linear() = quaternion.toRotationMatrix();
        pose.translation() = Eigen::Vector3f(
          static_cast<float>(message.pose.position.x),
          static_cast<float>(message.pose.position.y),
          static_cast<float>(message.pose.position.z));
        return pose.matrix().allFinite();
      };
    const auto pose_to_message = [this](
      const Eigen::Isometry3f &pose, geometry_msgs::msg::PoseStamped &message) {
        message.header.frame_id = "base_link";
        message.header.stamp = this->now();
        message.pose.position.x = pose.translation().x();
        message.pose.position.y = pose.translation().y();
        message.pose.position.z = pose.translation().z();
        const Eigen::Quaternionf quaternion(pose.linear());
        message.pose.orientation.x = quaternion.x();
        message.pose.orientation.y = quaternion.y();
        message.pose.orientation.z = quaternion.z();
        message.pose.orientation.w = quaternion.w();
      };

    Eigen::Isometry3f grasp_pose;
    if (!pose_from_message(request->grasp_candidate.grasp_pose, grasp_pose)) {
      response->status = "PLACEMENT_INVALID_GRASP_POSE";
      return;
    }
    std::vector<float> grasp_joints(
      request->grasp_candidate.grasp_joints.begin(),
      request->grasp_candidate.grasp_joints.end());
    if (!jointsWithinPiperLimits(grasp_joints)) {
      response->status = "PLACEMENT_INVALID_GRASP_JOINTS";
      return;
    }

    std::vector<my_actions::msg::GraspCandidate> placement_grasps{request->grasp_candidate};
    for (const auto &alternative : request->alternative_grasps) {
      Eigen::Isometry3f alternative_pose;
      const std::vector<float> joints(alternative.grasp_joints.begin(), alternative.grasp_joints.end());
      if (!request->use_layout_scene ||
          alternative.grasp_pose.header.frame_id != "base_link" ||
          alternative.grasp_pose.header.stamp != request->grasp_candidate.grasp_pose.header.stamp ||
          !pose_from_message(alternative.grasp_pose, alternative_pose) ||
          !jointsWithinPiperLimits(joints) || !std::isfinite(alternative.jaw_opening_m) ||
          alternative.jaw_opening_m < 0.0F || alternative.jaw_opening_m > 0.10F) {
        response->status = "PLACEMENT_INVALID_ALTERNATIVE_GRASP";
        return;
      }
      placement_grasps.push_back(alternative);
    }

    sensor_msgs::msg::PointCloud2::SharedPtr source_cloud;
    if (request->use_layout_scene) {
      const auto &header = request->grasp_candidate.grasp_pose.header;
      const auto matches = [&header](const auto &cloud) {
        return cloud.header.frame_id == "base_link" &&
          cloud.header.stamp == header.stamp && cloud.width > 0;
      };
      if (!matches(request->source_object_cloud) ||
          !matches(request->placement_scene) ||
          (!request->lift_only && !matches(request->destination_support_cloud))) {
        response->status = "PLACEMENT_LAYOUT_OBSERVATION_MISMATCH";
        return;
      }
      // Alias the immutable request for this callback's lifetime; copying the
      // complete cloud is unnecessary before its read-only geometry scan.
      source_cloud = std::shared_ptr<sensor_msgs::msg::PointCloud2>(
        request, &request->source_object_cloud);
    } else {
      std::lock_guard<std::mutex> lock(segmented_target_mutex_);
      source_cloud = latest_segmented_target_cloud_;
    }
    if (!source_cloud || source_cloud->header.frame_id != "base_link" ||
      source_cloud->width == 0) {
      response->status = "PLACEMENT_SOURCE_OBJECT_CLOUD_UNAVAILABLE";
      return;
    }
    std::vector<Eigen::Vector3f> source_points;
    std::vector<uint8_t> source_bottom_flags;
    const bool explicit_bottom = std::any_of(source_cloud->fields.begin(), source_cloud->fields.end(),
      [](const auto &field) { return field.name == "bottom" &&
        field.datatype == sensor_msgs::msg::PointField::UINT8 && field.count == 1; });
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<uint8_t>> bottom_iterator;
    if (explicit_bottom) bottom_iterator = std::make_unique<
      sensor_msgs::PointCloud2ConstIterator<uint8_t>>(*source_cloud, "bottom");
    sensor_msgs::PointCloud2ConstIterator<float> source_x(*source_cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> source_y(*source_cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> source_z(*source_cloud, "z");
    for (; source_x != source_x.end(); ++source_x, ++source_y, ++source_z) {
      if (std::isfinite(*source_x) && std::isfinite(*source_y) &&
        std::isfinite(*source_z)) {
        source_points.emplace_back(*source_x, *source_y, *source_z);
        if (bottom_iterator) source_bottom_flags.push_back(**bottom_iterator);
      }
      if (bottom_iterator) ++*bottom_iterator;
    }
    if (source_points.size() < 20) {
      response->status = "PLACEMENT_SOURCE_OBJECT_CLOUD_TOO_SMALL";
      return;
    }

    // The object is attached after the cached grasp, so its source voxels must
    // not remain as static obstacles during placement IK checks. Remove only
    // voxels close to the segmented source object; all other scene geometry
    // remains blocking.
    std::vector<Point> environment;
    if (!request->use_layout_scene) {
      constexpr float source_voxel_m = 0.015F;
      std::unordered_set<std::tuple<int, int, int>, VoxelHash> source_voxels;
      for (const auto &point : source_points) {
        source_voxels.emplace(
          static_cast<int>(std::floor(point.x() / source_voxel_m)),
          static_cast<int>(std::floor(point.y() / source_voxel_m)),
          static_cast<int>(std::floor(point.z() / source_voxel_m)));
      }
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        environment.reserve(obstacle_cloud_.size());
        for (const auto &point : obstacle_cloud_) {
          const int ix = static_cast<int>(std::floor(point[0] / source_voxel_m));
          const int iy = static_cast<int>(std::floor(point[1] / source_voxel_m));
          const int iz = static_cast<int>(std::floor(point[2] / source_voxel_m));
          bool source_neighbor = false;
          for (int dx = -1; dx <= 1 && !source_neighbor; ++dx) {
            for (int dy = -1; dy <= 1 && !source_neighbor; ++dy) {
              for (int dz = -1; dz <= 1; ++dz) {
                if (source_voxels.count({ix + dx, iy + dy, iz + dz}) != 0) {
                  source_neighbor = true;
                  break;
                }
              }
            }
          }
          if (!source_neighbor) environment.push_back(point);
        }
      }
    }

    // Simulation overlays are immutable request inputs, not a subscription to
    // a changing benchmark scene. Never remove them with source-object voxels.
    const auto &extra = request->additional_obstacles;
    if (extra.width || !extra.data.empty()) {
      bool valid = !request->use_layout_scene && extra.header.frame_id == "base_link" &&
        !extra.is_bigendian && extra.point_step > 0 && extra.height > 0 &&
        extra.data.size() <= 8 * 1024 * 1024 &&
        extra.row_step == extra.width * extra.point_step &&
        extra.data.size() == static_cast<size_t>(extra.row_step) * extra.height;
      for (const auto *name : {"x", "y", "z"}) {
        valid = valid && std::any_of(extra.fields.begin(), extra.fields.end(),
          [&](const auto &field) { return field.name == name && field.count == 1 &&
            field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
            field.offset <= extra.point_step && extra.point_step - field.offset >= 4; });
      }
      if (!valid) {
        response->status = "PLACEMENT_INVALID_ADDITIONAL_OBSTACLES";
        return;
      }
      sensor_msgs::PointCloud2ConstIterator<float> x(extra, "x"), y(extra, "y"), z(extra, "z");
      size_t added = 0;
      for (; x != x.end(); ++x, ++y, ++z) {
        if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z)) {
          environment.push_back(Point{*x, *y, *z});
          ++added;
        }
      }
      if (!added) {
        response->status = "PLACEMENT_EMPTY_ADDITIONAL_OBSTACLES";
        return;
      }
    }

    std::shared_ptr<const vamp_node::LayoutSceneCache::Geometry> layout_geometry;
    if (request->use_layout_scene) {
      const auto support_field = std::find_if(
        request->placement_scene.fields.begin(), request->placement_scene.fields.end(),
        [](const auto &field) { return field.name == "support" &&
          field.datatype == sensor_msgs::msg::PointField::UINT8 && field.count == 1; });
      if (support_field == request->placement_scene.fields.end()) {
        response->status = "PLACEMENT_LAYOUT_SUPPORT_LABELS_MISSING";
        return;
      }
      layout_geometry = layout_scene_cache_.get(request->placement_scene, request->support_mode);
      if (layout_geometry->environment.empty()) {
        response->status = "PLACEMENT_LAYOUT_SCENE_EMPTY";
        return;
      }
    }
    const AttachedObjectScene empty_attached_scene;
    const auto &attached_scene = layout_geometry ? layout_geometry->attached : empty_attached_scene;
    const std::vector<Eigen::Vector3f> empty_environment_points;
    const auto &environment_points = layout_geometry ? layout_geometry->points : empty_environment_points;
    const auto &placement_environment = layout_geometry ? layout_geometry->environment : environment;
    float source_bottom_z = std::numeric_limits<float>::infinity();
    for (const auto &point : source_points) source_bottom_z = std::min(source_bottom_z, point.z());

    const Eigen::Vector3f source_center(
      static_cast<float>(request->source_object_center.x),
      static_cast<float>(request->source_object_center.y),
      static_cast<float>(request->source_object_center.z));
    if (!source_center.allFinite()) {
      response->status = "PLACEMENT_INVALID_SOURCE_CENTER";
      return;
    }

    struct FeasiblePlacement {
      size_t index = 0;
      Eigen::Isometry3f desired = Eigen::Isometry3f::Identity();
      Eigen::Isometry3f resolved = Eigen::Isometry3f::Identity();
      Eigen::Isometry3f preplace = Eigen::Isometry3f::Identity();
      Eigen::Isometry3f object_delta = Eigen::Isometry3f::Identity();
      std::vector<float> joints;
      std::vector<float> preplace_joints;
      float position_error = 0.0F;
      float rotation_error = 0.0F;
      float cost = std::numeric_limits<float>::infinity();
      std::vector<std::vector<float>> descent;
      size_t grasp_index = 0;
    };
    if (request->plan_pick_place_motion) {
      // Freeze the attachment at the actual cached grasp FK, not the nominal
      // IK target. Otherwise solver residuals can translate the source through
      // its support before the first lift waypoint is even calculated.
      for (auto &candidate_grasp:placement_grasps) {
        std::array<float,6> joints{};
        std::copy_n(candidate_grasp.grasp_joints.begin(),6,joints.begin());
        const auto header=candidate_grasp.grasp_pose.header;
        pose_to_message(Robot::eefk(joints),candidate_grasp.grasp_pose);
        candidate_grasp.grasp_pose.header=header;
      }
    }
    const PlacementPointIndex empty_environment_index(empty_environment_points);
    const auto &environment_index = layout_geometry ? *layout_geometry->index : empty_environment_index;
    const PlacementPointIndex source_index(source_points);
    const AttachedObjectScene::SourceBounds source_bounds(source_points);
    std::string indexed_collision_reason;
    const auto indexed_pose_safe = [&](const std::vector<float> &joints, float opening = -1.0F) {
      ComponentTimer collision_timer(PipelineComponent::Collision);
      indexed_collision_reason.clear();
      if (!isCandidatePoseSafe(joints, {})) {
        indexed_collision_reason="SELF_COLLISION";
        return false;
      }
      Robot::ConfigurationBlock<1> configuration;
      for (size_t j=0;j<6;++j) configuration[j][0]=joints[j];
      Robot::Spheres<1> spheres;
      Robot::template sphere_fk<1>(configuration,spheres);
      std::array<float,6> array{};
      std::copy_n(joints.begin(),6,array.begin());
      const auto actual_pose=Robot::eefk(array);
      for (size_t sphere=0;sphere<Robot::n_spheres;++sphere) {
        const auto center=vamp_node::piperFingerSphereCenter(sphere, Eigen::Vector3f(
            spheres.x[{sphere,0}],spheres.y[{sphere,0}],spheres.z[{sphere,0}]), actual_pose, opening);
        const float radius = spheres.r[{sphere,0}] + 0.0025F;
        if (!environment_index.sphereClearWithPredicate(center, radius,
            [&](size_t, const Eigen::Vector3f &point) {
              const bool blocked = (point - center).squaredNorm() < radius * radius &&
                vamp_node::piperFingerMeshBoundsContain(sphere, point, actual_pose, opening);
              if (blocked && indexed_collision_reason.empty()) {
                std::ostringstream detail;
                detail << "sphere=" << sphere << ",point=" << point.transpose();
                indexed_collision_reason=detail.str();
              }
              return blocked;
            })) return false;
      }
      return true;
    };
    const auto pose_safe = [&](const std::vector<float> &joints, float opening = -1.0F) {
      return request->use_layout_scene ? indexed_pose_safe(joints, opening) :
        isCandidatePoseSafe(joints,environment,nullptr,nullptr,nullptr,0.0F,nullptr,opening);
    };
    std::optional<FeasiblePlacement> best;
    if (request->lift_only) best = FeasiblePlacement{};
    int rejected_pose = 0;
    int rejected_ik = 0;
    int rejected_upright_tilt = 0;
    int rejected_collision = 0;
    int rejected_preplace = 0;
    int rejected_static_collision = 0;
    int rejected_finger_collision = 0;
    int rejected_attached_collision = 0;
    std::string attached_first_block;
    int rejected_object_arm_collision = 0;
    int rejected_descent_arm_collision = 0;
    int rejected_final_pose_collision = 0;
    int rejected_descent_ik = 0;
    int rejected_descent_continuity = 0;
    for (size_t index = 0; index < request->candidates.size() && !request->lift_only; ++index) {
      const auto &candidate = request->candidates[index];
      if (candidate.object_pose.header.frame_id != "base_link" ||
        !std::isfinite(candidate.score)) {
        ++rejected_pose;
        continue;
      }
      if (best && candidate.score < request->candidates[best->index].score) continue;
      Eigen::Isometry3f centered_object_pose;
      if (!pose_from_message(candidate.object_pose, centered_object_pose)) {
        ++rejected_pose;
        continue;
      }
      Eigen::Isometry3f object_delta = Eigen::Isometry3f::Identity();
      object_delta.linear() = centered_object_pose.linear();
      object_delta.translation() = centered_object_pose.translation() -
        centered_object_pose.linear() * source_center;
      for (size_t grasp_index = 0; grasp_index < placement_grasps.size(); ++grasp_index) {
        const auto &active_grasp = placement_grasps[grasp_index];
        pose_from_message(active_grasp.grasp_pose, grasp_pose);
        grasp_joints.assign(active_grasp.grasp_joints.begin(), active_grasp.grasp_joints.end());
        Eigen::Isometry3f desired = object_delta * grasp_pose;
        // Release just above the floor instead of fitting noisy depth exactly.
        if (request->use_layout_scene && request->support_mode == "floor_slots")
          desired.translation().z() += 0.003F+placement_position_tolerance_m_;
        // Solve to the acceptance tolerance instead of stopping at a loose pose
        // and subsequently rejecting an otherwise reachable upright placement.
        const float rotation_tolerance = request->use_layout_scene ?
          placement_target_orientation_tolerance_rad_ : candidate.orientation_policy == "upright" ?
          std::min(placement_orientation_tolerance_rad_,
            placement_upright_tilt_tolerance_rad_ * 0.9F) :
          placement_orientation_tolerance_rad_;
        auto descent_noise = placement_noise_policy_;
        bool accumulate_descent_noise = false;
        const auto check_sample = [&](const std::vector<float> &sample_joints) {
          ComponentTimer collision_timer(PipelineComponent::Collision);
          std::array<float,6> sample_array{};
          std::copy_n(sample_joints.begin(),6,sample_array.begin());
          const Eigen::Isometry3f sample_pose=Robot::eefk(sample_array);
          if (keep_object_upright_ && !vamp_node::objectRemainsUpright(sample_pose,grasp_pose,object_upright_tolerance_rad_)) {
            ++rejected_upright_tilt; return false;
          }
          const Eigen::Isometry3f sample_delta=sample_pose*grasp_pose.inverse();
          const Eigen::Isometry3f inverse_delta=sample_delta.inverse();
          Robot::ConfigurationBlock<1> configuration;
          for (size_t j=0;j<6;++j) configuration[j][0]=sample_array[j];
          Robot::Spheres<1> spheres;
          Robot::template sphere_fk<1>(configuration,spheres);
          for (size_t sphere=0;sphere<74;++sphere) {
            if (!source_index.sphereClear(inverse_delta*Eigen::Vector3f(
                spheres.x[{sphere,0}],spheres.y[{sphere,0}],spheres.z[{sphere,0}]),
                spheres.r[{sphere,0}]+0.0025F)) {
              ++rejected_object_arm_collision;
              return false;
            }
          }
          if (!environment_index.fingersClear(sample_pose,
              active_grasp.jaw_opening_m,static_cast<float>(grasp_tcp_offset_m_),
              static_cast<float>(finger_width_m_),static_cast<float>(finger_height_m_),
              static_cast<float>(finger_length_m_))) {
            ++rejected_finger_collision;
            return false;
          }
          if (!pose_safe(sample_joints,active_grasp.jaw_opening_m)) {
            ++rejected_descent_arm_collision;
            return false;
          }
          const bool contact_segment=(sample_pose.translation()-desired.translation()).norm()<=
            placement_contact_distance_m_;
          AttachedObjectScene::CollisionInfo blocked;
          auto trial_noise = accumulate_descent_noise ? descent_noise : placement_noise_policy_;
          if (!attached_scene.clear(source_points,sample_delta,source_bottom_z,contact_segment,&blocked,&source_bottom_flags,
              0.001F,nullptr,0.0F,&trial_noise,&source_bounds)) {
            ++rejected_attached_collision;
            if (attached_first_block.empty()) {
              std::ostringstream detail;
              detail << " held=[" << blocked.held_point.transpose()
                << "] obstacle=[" << blocked.obstacle_point.transpose()
                << "] support=" << blocked.named_support
                << " bottom=" << blocked.source_bottom << " terminal=" << contact_segment;
              attached_first_block = detail.str();
            }
            RCLCPP_DEBUG(this->get_logger(),
              "[PLACEMENT HELD BLOCK] object=%s candidate=%u support=%d bottom=%d terminal=%d",
              request->object_name.c_str(),candidate.id,blocked.named_support,blocked.source_bottom,contact_segment);
            return false;
          }
          if (accumulate_descent_noise) descent_noise = std::move(trial_noise);
          return true;
        };
        PoseIkDiagnostics ik_diagnostics;
        std::vector<float> placement_joints;
        if (!solvePiperPoseIK(
            desired, grasp_joints, placement_joints, ik_diagnostics, false,
            placement_position_tolerance_m_,
            rotation_tolerance, 1000, [&](const auto &joints) {
              return request->use_layout_scene ? check_sample(joints) :
                !enable_grasp_collision_check_ || pose_safe(joints,active_grasp.jaw_opening_m);
            }, request->use_layout_scene && placement_fast_ik_)) {
          if (ik_diagnostics.collision_rejections > 0) {
            ++rejected_collision;
            ++rejected_final_pose_collision;
            continue;
          }
          ++rejected_ik;
          if (rejected_ik <= 3) {
            RCLCPP_INFO(this->get_logger(),
              "[PLACEMENT IK DETAIL] object=%s mode=%s candidate=%u tcp=[%.3f,%.3f,%.3f] "
              "best_error={position:%.1fmm,orientation:%.1fdeg} tolerance={position:%.1fmm,orientation:%.1fdeg}",
              request->object_name.c_str(), request->support_mode.c_str(), candidate.id,
              desired.translation().x(), desired.translation().y(), desired.translation().z(),
              ik_diagnostics.best_position_error*1000.0F,
              ik_diagnostics.best_rotation_error*180.0F/static_cast<float>(M_PI),
              placement_position_tolerance_m_*1000.0F,
              rotation_tolerance*180.0F/static_cast<float>(M_PI));
          }
          continue;
        }
        if (enable_grasp_collision_check_ &&
          !pose_safe(placement_joints,active_grasp.jaw_opening_m)) {
          ++rejected_collision;
          ++rejected_static_collision;
          if (rejected_static_collision == 1) {
            RCLCPP_INFO(this->get_logger(), "[PLACEMENT COLLISION] candidate=%u static: %s",
              candidate.id, describeCandidateCollision(placement_joints, placement_environment, nullptr, nullptr, nullptr, 0.0F, nullptr).c_str());
          }
          continue;
        }
        std::array<float, 6> placement_array{};
        std::copy_n(placement_joints.begin(), 6, placement_array.begin());
        Eigen::Isometry3f resolved = Robot::eefk(placement_array);
        float position_error =
          (resolved.translation() - desired.translation()).norm();
        float rotation_error = std::abs(Eigen::AngleAxisf(
          desired.linear() * resolved.linear().transpose()).angle());
        Eigen::Isometry3f resolved_object_delta =
          resolved * grasp_pose.inverse();
        if (candidate.orientation_policy == "upright") {
          const Eigen::Vector3f desired_up =
            centered_object_pose.linear() * Eigen::Vector3f::UnitZ();
          const Eigen::Vector3f resolved_up =
            resolved_object_delta.linear() * Eigen::Vector3f::UnitZ();
          const float upright_tilt_error = std::acos(std::clamp(
            desired_up.normalized().dot(resolved_up.normalized()), -1.0F, 1.0F));
          if (upright_tilt_error > placement_upright_tilt_tolerance_rad_) {
            ++rejected_upright_tilt;
            continue;
          }
        }

        // Descend to the accepted FK endpoint, not back to the ideal orientation.
        // Tight local tracking stays independent of target matching tolerance.
        const Eigen::Isometry3f descent_target = resolved;
        Eigen::Isometry3f preplace_desired = descent_target;
        std::vector<float> preplace_joints;
        bool hover_found=false;
        Eigen::Isometry3f resolved_preplace;
        std::vector<std::vector<float>> descent;
        for (const float hover : placementHoverDistances(
            placement_preplace_offset_m_, request->use_layout_scene)) {
          preplace_desired.translation()=descent_target.translation();
          preplace_desired.translation().z()+=hover;
          PoseIkDiagnostics preplace_diagnostics;
          if (solvePiperPoseIK(preplace_desired,placement_joints,preplace_joints,
              preplace_diagnostics,false,
              request->use_layout_scene ? placement_position_tolerance_m_ : placement_position_tolerance_m_+0.005F,
              request->use_layout_scene ? placement_descent_orientation_tolerance_rad_ : placement_orientation_tolerance_rad_+0.10F,1000,
              [&](const auto &joints) { return !enable_grasp_collision_check_ || pose_safe(joints,active_grasp.jaw_opening_m); }, request->use_layout_scene && placement_fast_ik_)) {
            hover_found=true;
          } else {
            continue;
          }
          std::array<float, 6> preplace_array{};
          std::copy_n(preplace_joints.begin(), 6, preplace_array.begin());
          resolved_preplace = Robot::eefk(preplace_array);
          if (request->use_layout_scene) {
            descent_noise = placement_noise_policy_;
            accumulate_descent_noise = true;
            if (!check_sample(preplace_joints)) { ++rejected_collision; continue; }
            std::vector<float> previous=preplace_joints;
            const auto descent_started=std::chrono::steady_clock::now();
            const auto path=planCartesianApproach(resolved_preplace,descent_target,preplace_joints,
              placement_descent_step_m_,placement_descent_max_joint_step_rad_,
              [&](const auto &waypoint,const auto &seed,std::vector<float> &solution) {
                PoseIkDiagnostics diagnostics;
                return solvePiperPoseIK(waypoint,seed,solution,diagnostics,true,
                  std::min(placement_position_tolerance_m_,0.001F),placement_descent_orientation_tolerance_rad_,600);
              },
              [&](const auto &joints,size_t,bool) {
                std::vector<float> midpoint(6);
                for (size_t j=0;j<6;++j) midpoint[j]=0.5F*(previous[j]+joints[j]);
                if (!check_sample(midpoint) || !check_sample(joints)) return false;
                previous=joints;
                return true;
              });
            response->descent_search_seconds+=std::chrono::duration<double>(
              std::chrono::steady_clock::now()-descent_started).count();
            if (!path.success) {
              if (path.failure_reason.rfind("IK_FAILED", 0) == 0) {
                ++rejected_descent_ik;
              } else if (path.failure_reason.rfind("JOINT_", 0) == 0) {
                ++rejected_descent_continuity;
              } else {
                ++rejected_collision;
              }
              RCLCPP_INFO(this->get_logger(),"[PLACEMENT DESCENT] object=%s candidate=%u failure=%s",
                request->object_name.c_str(),candidate.id,path.failure_reason.c_str());
              continue;
            }
            if (keep_object_upright_) {
              const auto checked=vamp_node::validateUprightPath(preplace_joints,path.waypoint_joints,
                [](const auto &q) { std::array<float,6> a{};std::copy_n(q.begin(),6,a.begin());return Robot::eefk(a); },
                grasp_pose,object_upright_tolerance_rad_,
                [&](const auto &q,size_t,bool) { return check_sample(q); });
              if (!checked.success) { ++rejected_upright_tilt; continue; }
              descent=checked.waypoint_joints;
            } else descent=path.waypoint_joints;
            RCLCPP_INFO(this->get_logger(),
              "[ATTACHED NOISE BUDGET] stage=descent candidate=%u accepted_points=%zu voxels=%zu",
              candidate.id,descent_noise.points.size(),descent_noise.observed_voxels);
            placement_joints=descent.back();
            std::copy_n(placement_joints.begin(),6,placement_array.begin());
            resolved=Robot::eefk(placement_array);
            resolved_object_delta=resolved*grasp_pose.inverse();
            position_error=(resolved.translation()-desired.translation()).norm();
            rotation_error=std::abs(Eigen::AngleAxisf(desired.linear()*resolved.linear().transpose()).angle());
          }
          break;
        }
        if (!hover_found || (request->use_layout_scene && descent.empty())) {
          ++rejected_preplace;
          continue;
        }
        // Local tracking error must not enlarge the endpoint acceptance band.
        if (request->use_layout_scene &&
          (position_error > placement_position_tolerance_m_+0.001F ||
           rotation_error > rotation_tolerance+placement_descent_orientation_tolerance_rad_)) {
          ++rejected_pose;
          RCLCPP_INFO(this->get_logger(),
            "[PLACEMENT ACHIEVED POSE] candidate=%u position_error=%.1fmm orientation_error=%.1fdeg",
            candidate.id,position_error*1000.0F,rotation_error*180.0F/static_cast<float>(M_PI));
          continue;
        }
        float joint_travel = 0.0F;
        for (size_t joint = 0; joint < 6; ++joint) {
          const float difference = placement_joints[joint] - grasp_joints[joint];
          joint_travel += difference * difference;
        }
        const float cost = -candidate.score + 8.0F * position_error +
          0.25F * rotation_error + 0.02F * std::sqrt(joint_travel);
        if (!best || preferPlacement(candidate.score, cost,
            request->candidates[best->index].score, best->cost)) {
          best = FeasiblePlacement{
            index, desired, resolved, resolved_preplace,
            resolved_object_delta, placement_joints,
            preplace_joints, position_error, rotation_error, cost, descent, grasp_index};
        }
        if (request->use_layout_scene && best && best->index == index) break;
      }
    }
    if (!best) {
      std::ostringstream status;
      status << "PLACEMENT_NO_REACHABLE_CANDIDATE tested="
             << request->candidates.size() << " rejected={pose:"
             << rejected_pose << ",ik:" << rejected_ik << ",upright_tilt:"
             << rejected_upright_tilt << ",collision:"
             << rejected_collision << ",preplace:" << rejected_preplace
             << ",static_collision:" << rejected_static_collision
             << ",finger_collision:" << rejected_finger_collision
             << ",attached_collision:" << rejected_attached_collision
             << ",object_arm_collision:" << rejected_object_arm_collision
             << ",descent_arm_collision:" << rejected_descent_arm_collision
             << ",final_pose_collision:" << rejected_final_pose_collision
             << ",descent_ik:" << rejected_descent_ik
             << ",descent_continuity:" << rejected_descent_continuity << "}";
      if (!attached_first_block.empty()) {
        RCLCPP_WARN(this->get_logger(), "[PLACEMENT FIRST HELD BLOCK] object=%s%s",
          request->object_name.c_str(), attached_first_block.c_str());
      }
      response->status = status.str();
      RCLCPP_WARN(this->get_logger(), "%s", response->status.c_str());
      return;
    }

    response->pose_validation_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-placement_started).count();
    const auto &selected_grasp = placement_grasps[best->grasp_index];
    response->selected_grasp_candidate = selected_grasp;
    if (!request->lift_only) {
    response->selected_candidate = request->candidates[best->index];
    // Reservation and RViz must validate the actual held-object FK pose.
    // Preserve the immutable observation header used by the slot search.
    if (request->use_layout_scene && request->support_mode == "floor_slots") {
      Eigen::Isometry3f actual_object_pose=best->object_delta;
      actual_object_pose.translation()=best->object_delta*source_center;
      const auto source_header=response->selected_candidate.object_pose.header;
      pose_to_message(actual_object_pose,response->selected_candidate.object_pose);
      response->selected_candidate.object_pose.header=source_header;
    }
    pose_to_message(best->resolved, response->placement_ee_pose);
    pose_to_message(best->preplace, response->preplace_ee_pose);
    std::copy_n(best->joints.begin(), 6, response->placement_joints.begin());
    std::copy_n(
      best->preplace_joints.begin(), 6, response->preplace_joints.begin());
    response->descent_waypoint_count=static_cast<uint32_t>(best->descent.size());
    for (const auto &waypoint:best->descent) {
      response->descent_path_flat.insert(response->descent_path_flat.end(),waypoint.begin(),waypoint.end());
    }
    response->position_error_m = best->position_error;
    response->orientation_error_rad = best->rotation_error;
    }

    if (request->plan_pick_place_motion) {
      PipelineProfile motion_profile(pipeline_profiling_, "placement_motion", [this](const auto &line) {
        RCLCPP_INFO(this->get_logger(), "[PIPELINE PROFILE] %s", line.c_str());
      });
      if (!request->use_layout_scene) {
        response->status="MOTION_REQUIRES_FROZEN_LAYOUT_SCENE";
        return;
      }
      const auto seconds_since=[](const auto &started) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
      };
      std::array<float,6> source_joints{};
      std::copy_n(selected_grasp.grasp_joints.begin(),6,source_joints.begin());
      const Eigen::Isometry3f source_pose=Robot::eefk(source_joints);
      const float requested_lift_rise = request->lift_distance_m == 0.0F ?
        0.10F : request->lift_distance_m;
      Eigen::Isometry3f attachment_pose;
      pose_from_message(selected_grasp.grasp_pose,attachment_pose);
      Eigen::Vector3f minimum=source_points.front(),maximum=minimum;
      for (const auto &point:source_points) {
        minimum=minimum.cwiseMin(point); maximum=maximum.cwiseMax(point);
      }
      // Explicit source-bottom labels preserve contact on a tilted support.
      Eigen::Vector3f bottom_normal = Eigen::Vector3f::UnitZ();
      Eigen::Vector3f bottom_center(0, 0, source_bottom_z);
      if (!source_bottom_flags.empty()) {
        std::vector<Eigen::Vector3f> bottom_samples;
        for (size_t i = 0; i < source_points.size(); ++i)
          if (source_bottom_flags[i] == 2) bottom_samples.push_back(source_points[i]);
        if (bottom_samples.size() >= 3) {
          bottom_center.setZero();
          for (const auto &point : bottom_samples) bottom_center += point;
          bottom_center /= static_cast<float>(bottom_samples.size());
          Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
          for (const auto &point : bottom_samples) {
            const auto offset = (point - bottom_center).eval();
            covariance += offset * offset.transpose();
          }
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
          if (solver.info() == Eigen::Success) bottom_normal = solver.eigenvectors().col(0);
        }
      }
      AttachedObjectScene lift_scene;
      const float departure_slack = placement_allow_separating_contact_departure_ ?
        placement_departure_contact_tolerance_m_ : 0.0F;
      // Source-table uncertainty belongs to the noise policy as well as the
      // optional sideways departure; ordinary default launches use it too.
      const float support_allowance = placement_noise_policy_.maximum_points > 0 ?
        0.002F : departure_slack;
      for (size_t point_index = 0; point_index < environment_points.size(); ++point_index) {
        const auto &point = environment_points[point_index];
        // Known objects remain obstacles even if their bases share the table plane.
        const bool protected_object = layout_geometry &&
          layout_geometry->support_labels[point_index] >= 2;
        const bool initial_support = !protected_object && (source_bottom_flags.empty() ?
          initialLiftSupportSample(point,minimum,maximum,source_bottom_z) :
          // The fit band and point neighborhood define the departure-only
          // table uncertainty. Bottom penetration remains separately bounded.
          std::abs(bottom_normal.dot(point - bottom_center)) <=
            vamp_node::initialLiftSupportBand(support_allowance) &&
          // Include the bounded 6 mm departure plus the 3 mm point neighborhood.
          // Otherwise sliding off the original footprint sees the same table
          // as a new obstacle before the bottom has lifted clear of it.
          point.x() >= minimum.x() - 0.009F && point.x() <= maximum.x() + 0.009F &&
          point.y() >= minimum.y() - 0.009F && point.y() <= maximum.y() + 0.009F);
        lift_scene.add(point,initial_support);
      }
      bool lifting=true;
      bool escaping=false;
      std::string escape_status;
      std::function<bool(const std::vector<float> &)> escape_contact_safe;
      bool separating_departure=false;
      auto lift_noise = placement_noise_policy_;
      const auto source_contacts=lift_scene.initialContacts(source_points);
      std::string held_failure_reason;
      const auto held_safe=[&](const std::vector<float> &joints) {
        ComponentTimer collision_timer(PipelineComponent::Collision);
        ++response->transfer_collision_samples;
        std::array<float,6> array{};
        std::copy_n(joints.begin(),6,array.begin());
        const auto pose=Robot::eefk(array);
        if (keep_object_upright_ && !vamp_node::objectRemainsUpright(pose,attachment_pose,object_upright_tolerance_rad_)) {
          held_failure_reason="OBJECT_UPRIGHT_CONSTRAINT"; return false;
        }
        const Eigen::Isometry3f delta=pose*attachment_pose.inverse();
        const Eigen::Isometry3f inverse=delta.inverse();
        Robot::ConfigurationBlock<1> configuration;
        for (size_t j=0;j<6;++j) configuration[j][0]=array[j];
        Robot::Spheres<1> spheres;
        Robot::template sphere_fk<1>(configuration,spheres);
        for (size_t sphere=0;sphere<74;++sphere) {
          if (!source_index.sphereClear(inverse*Eigen::Vector3f(
              spheres.x[{sphere,0}],spheres.y[{sphere,0}],spheres.z[{sphere,0}]),
              spheres.r[{sphere,0}]+0.0025F)) {
            held_failure_reason="OBJECT_ARM"; return false;
          }
        }
        // Closed-gripper initial state and lift only. Self and object-arm
        // remain checked; transfer restores robot/finger environment checks.
        if (!isCandidatePoseSafe(joints, {})) {
          held_failure_reason="SELF_COLLISION"; return false;
        }
        if (escaping) return escape_contact_safe && escape_contact_safe(joints);
        const bool check_robot_environment = checkLiftRobotEnvironment(
          lifting, placement_bypass_robot_environment_during_lift_);
        if (check_robot_environment && !indexed_pose_safe(joints,selected_grasp.jaw_opening_m)) {
          held_failure_reason="ROBOT_ENVIRONMENT:"+indexed_collision_reason; return false;
        }
        if (check_robot_environment && !environment_index.fingersClear(pose,
            selected_grasp.jaw_opening_m,static_cast<float>(grasp_tcp_offset_m_),
            static_cast<float>(finger_width_m_),static_cast<float>(finger_height_m_),
            static_cast<float>(finger_length_m_))) {
          held_failure_reason="FINGER_ENVIRONMENT"; return false;
        }
        if (lifting) {
          // Independent attached-object bypass. Transfer switches lifting off
          // and restores all robot/finger/attached-object scene checks.
          if (placement_bypass_attached_collision_during_lift_) return true;
          const bool initial_contact=(pose.translation()-source_pose.translation()).norm()<=
            vamp_node::initialLiftContactTravel(support_allowance);
          AttachedObjectScene::CollisionInfo blocked;
          const bool clear=lift_scene.clear(source_points,delta,source_bottom_z,initial_contact,&blocked,&source_bottom_flags,
            initial_contact ? 0.003F : 0.001F,
            separating_departure ? &source_contacts : nullptr,
            separating_departure ? departure_slack : 0.0F,&lift_noise,&source_bounds);
          if (!clear) {
            std::ostringstream reason;
            reason << "ATTACHED_OBJECT held=[" << blocked.held_point.transpose()
              << "] obstacle=[" << blocked.obstacle_point.transpose()
              << "] support=" << blocked.named_support << " bottom=" << blocked.source_bottom
              << " noise_points=" << lift_noise.observed_points
              << " noise_voxels=" << lift_noise.observed_voxels;
            held_failure_reason=reason.str();
          }
          return clear;
        }
        AttachedObjectScene::CollisionInfo blocked;
        if (!attached_scene.clear(source_points,delta,source_bottom_z,false,&blocked,&source_bottom_flags,
            0.001F,nullptr,0.0F,nullptr,&source_bounds)) {
          std::ostringstream detail;detail << "ATTACHED_OBJECT held=[" << blocked.held_point.transpose()
            << "] obstacle=[" << blocked.obstacle_point.transpose() << "]";
          held_failure_reason=detail.str();return false;
        }
        return true;
      };
      auto started=std::chrono::steady_clock::now();
      using HeldRRTC=vamp::planning::RRTC<PayloadPiper,rake,Robot::resolution>;
      Eigen::Isometry3f lift_goal=source_pose;
      lift_goal.translation().z() += request->lift_distance_m == 0.0F ?
        0.10F : request->lift_distance_m;
      std::vector<float> previous(source_joints.begin(),source_joints.end());
      CartesianApproachPlan lift;
      std::vector<std::vector<float>> departure_path;
      if (!held_safe(previous) && placement_allow_separating_contact_departure_ &&
          held_failure_reason.rfind("ATTACHED_OBJECT", 0) == 0) {
        // Only admit pre-existing padded contacts when every swept sample
        // preserves their initial clearance and separates from the obstacle.
        // The endpoint must pass ordinary clearance before vertical lifting.
        separating_departure=true;
        if (held_safe(previous)) {
          for (float rise : {0.0005F,0.001F}) {
            for (float distance : {0.003F,0.004F}) {
              for (int direction=0;direction<8 && departure_path.empty();++direction) {
                lift_noise = placement_noise_policy_;
                Eigen::Isometry3f departure_goal=source_pose;
                const float angle=static_cast<float>(direction)*static_cast<float>(M_PI)/4.0F;
                departure_goal.translation() += Eigen::Vector3f(
                  distance*std::cos(angle),distance*std::sin(angle),rise);
                std::vector<float> seed(source_joints.begin(),source_joints.end());
                const auto departure=planCartesianApproach(source_pose,departure_goal,seed,0.0005F,0.05F,
                  [&](const auto &pose,const auto &prior,std::vector<float> &solution) {
                    PoseIkDiagnostics diagnostics;
                    return solvePiperPoseIK(pose,prior,solution,diagnostics,true,
                      // Small source clearances need a genuinely rigid departure;
                      // a loose angular residual moves distant object points
                      // toward neighbors despite a separating TCP translation.
                      std::min(placement_position_tolerance_m_,0.00001F),
                      std::min(placement_descent_orientation_tolerance_rad_,0.0001F),600);
                  },[&](const auto &joints,size_t,bool) {
                    std::vector<float> midpoint(6);
                    for (size_t j=0;j<6;++j) midpoint[j]=(seed[j]+joints[j])*0.5F;
                    if (!held_safe(midpoint) || !held_safe(joints)) return false;
                    seed=joints;return true;
                  });
                if (departure.success) {
                  separating_departure=false;
                  if (held_safe(departure.waypoint_joints.back())) departure_path=departure.waypoint_joints;
                  separating_departure=true;
                }
              }
              if (!departure_path.empty()) break;
            }
            if (!departure_path.empty()) break;
          }
        }
        separating_departure=false;
      }
      Eigen::Isometry3f lift_start=source_pose;
      std::vector<float> lift_start_joints(source_joints.begin(),source_joints.end());
      if (!departure_path.empty()) {
        lift_start_joints=departure_path.back();
        std::array<float,6> array{};
        std::copy_n(lift_start_joints.begin(),6,array.begin());
        lift_start=Robot::eefk(array);
        lift_goal=lift_start;
        lift_goal.translation().z() += request->lift_distance_m == 0.0F ?
        0.10F : request->lift_distance_m;
      }
      previous=lift_start_joints;
      if (departure_path.empty()) lift_noise = placement_noise_policy_;
      if (held_safe(previous)) {
        auto initial_lift_noise = lift_noise;
        const auto calculate_lift = [&](const Eigen::Isometry3f &goal) {
          lift_noise = initial_lift_noise;
          previous=lift_start_joints;
          held_failure_reason.clear();
          return planCartesianApproach(lift_start,goal,previous,0.002F,0.05F,
          [&](const auto &pose,const auto &seed,std::vector<float> &solution) {
            PoseIkDiagnostics diagnostics;
            return solvePiperPoseIK(pose,seed,solution,diagnostics,true,
              std::min(placement_position_tolerance_m_,0.001F),placement_descent_orientation_tolerance_rad_,600);
          },[&](const auto &joints,size_t,bool) {
            std::vector<float> midpoint(6);
            for (size_t j=0;j<6;++j) midpoint[j]=(previous[j]+joints[j])*0.5F;
            if (!held_safe(midpoint) || !held_safe(joints)) return false;
            previous=joints; return true;
          });
        };
        lift=calculate_lift(lift_goal);
        // A vertical departure can leave the arm's reachable workspace near
        // full extension or approach a rear wall. Try one bounded inward/upward departure;
        // preserve fixed orientation, continuity and every held-object check.
        if (!lift.success && (lift.failure_reason.rfind("IK_FAILED_", 0) == 0 ||
                             lift.failure_reason.rfind("COLLISION_", 0) == 0 ||
                             lift.failure_reason.rfind("JOINT_", 0) == 0)) {
          Eigen::Vector2f radial=source_pose.translation().head<2>();
          if (radial.norm() > 0.05F) {
            Eigen::Isometry3f inward_goal=lift_goal;
            inward_goal.translation().head<2>() -= 0.05F * radial.normalized();
            lift=calculate_lift(inward_goal);
            if (!lift.success) {
              // Near extension, rising immediately can leave the workspace.
              // Retract with a small rise first, then establish clearance.
              // Every midpoint/waypoint still checks the complete attachment.
              Eigen::Isometry3f retract_goal=lift_start;
              retract_goal.translation().head<2>() -= 0.04F * radial.normalized();
              retract_goal.translation().z() += 0.01F;
              auto retract=calculate_lift(retract_goal);
              if (retract.success) {
                initial_lift_noise=lift_noise;
                lift_start_joints=retract.waypoint_joints.back();
                std::array<float,6> array{};
                std::copy_n(lift_start_joints.begin(),6,array.begin());
                lift_start=Robot::eefk(array);
                Eigen::Isometry3f clearance_goal=lift_start;
                clearance_goal.translation().z() += std::max(0.0F, requested_lift_rise-0.01F);
                lift=calculate_lift(clearance_goal);
                if (lift.success) {
                  lift.waypoint_joints.insert(lift.waypoint_joints.begin(),
                    retract.waypoint_joints.begin(),retract.waypoint_joints.end()-1);
                  RCLCPP_INFO(this->get_logger(),
                    "[PLACEMENT LIFT] validated retract-then-lift fallback");
                }
              } else {
                lift=std::move(retract);
              }
            }
          }
        }
      } else {
        lift.failure_reason="INITIAL_HELD_STATE_COLLISION:"+held_failure_reason;
      }
      if (!lift.success) {
        // Last bounded departure: reverse the validated grasp's pregrasp
        // configuration. Its wrist may need a small rotation away from a
        // joint limit; fixed-orientation Cartesian lifting cannot do that.
        // This is a sampled, prevalidated path, never a joint-target jump.
        for (int retreat_attempt=0;retreat_attempt<2 && !lift.success;++retreat_attempt) {
        std::vector<float> retreat_start(source_joints.begin(),source_joints.end());
        auto retreat_start_pose=source_pose;
        std::vector<std::vector<float>> retreat_prefix;
        lift_noise=placement_noise_policy_;
        if (retreat_attempt==1) {
          Eigen::Vector2f radial=source_pose.translation().head<2>();
          if (radial.norm()<=0.05F) continue;
          auto goal=source_pose;
          goal.translation().head<2>()-=0.02F*radial.normalized();
          goal.translation().z()+=0.005F;
          auto previous=retreat_start;
          const auto prefix=planCartesianApproach(source_pose,goal,retreat_start,0.002F,0.05F,
            [&](const auto &pose,const auto &seed,std::vector<float> &solution) {
              PoseIkDiagnostics diagnostics;
              return solvePiperPoseIK(pose,seed,solution,diagnostics,true,
                std::min(placement_position_tolerance_m_,0.001F),placement_descent_orientation_tolerance_rad_,600);
            },[&](const auto &joints,size_t,bool) {
              std::vector<float> midpoint(6);
              for (size_t j=0;j<6;++j) midpoint[j]=0.5F*(previous[j]+joints[j]);
              if (!held_safe(midpoint) || !held_safe(joints)) return false;
              previous=joints;return true;
            });
          if (!prefix.success) continue;
          retreat_prefix=prefix.waypoint_joints;
          retreat_start=retreat_prefix.back();
          std::array<float,6> array{};std::copy_n(retreat_start.begin(),6,array.begin());
          retreat_start_pose=Robot::eefk(array);
        }
        std::vector<float> retreat_goal(selected_grasp.pregrasp_joints.begin(),
          selected_grasp.pregrasp_joints.end());
        std::array<float,6> retreat_array{};
        std::copy_n(retreat_goal.begin(),6,retreat_array.begin());
        const auto retreat_pose=Robot::eefk(retreat_array);
        const float retreat_distance=(retreat_pose.translation()-retreat_start_pose.translation()).norm();
        const float retreat_angle=std::abs(Eigen::AngleAxisf(
          retreat_pose.linear()*source_pose.linear().transpose()).angle());
        if (jointsWithinPiperLimits(retreat_goal) && retreat_distance<=0.16F &&
            retreat_pose.translation().z()>=source_pose.translation().z()+requested_lift_rise-0.001F &&
            retreat_angle<=0.40F) {
          const auto &start=retreat_start;
          const size_t count=std::max<size_t>(1,std::max(
            std::ceil(maximumJointDelta(start,retreat_goal)/0.002F),
            std::ceil(retreat_distance/0.001F)));
          std::vector<std::vector<float>> path;
          auto prior=start;
          auto prior_pose=retreat_start_pose;
          bool valid=count<=512 && held_safe(start);
          for (size_t step=1;valid && step<=count;++step) {
            std::vector<float> joints(6),midpoint(6);
            for (size_t j=0;j<6;++j) {
              joints[j]=start[j]+static_cast<float>(step)/count*(retreat_goal[j]-start[j]);
              midpoint[j]=0.5F*(prior[j]+joints[j]);
              retreat_array[j]=joints[j];
            }
            const auto pose=Robot::eefk(retreat_array);
            valid=(pose.translation()-prior_pose.translation()).norm()<=0.002F &&
              pose.translation().z()>=source_pose.translation().z()-0.001F &&
              held_safe(midpoint) && held_safe(joints);
            if (valid) { path.push_back(joints);prior=joints;prior_pose=pose; }
          }
          if (valid) {
            lift.success=true;lift.failure_reason.clear();lift.waypoint_joints=std::move(path);
            if (!retreat_prefix.empty()) lift.waypoint_joints.insert(lift.waypoint_joints.begin(),
              retreat_prefix.begin(),retreat_prefix.end()-1);
            departure_path.clear();
            RCLCPP_INFO(this->get_logger(),
              "[PLACEMENT LIFT] validated reverse-pregrasp retreat waypoints=%zu",count);
          }
        }
        }
      }
      if (!lift.success) {
        // A validated pregrasp can be reachable through a curved joint path
        // even when fixed-orientation lifts and straight joint retreat fail.
        // Bound this local escape to the source region and retain the same
        // held-object validator used by the ordinary placement transfer.
        const auto &goal=selected_grasp.pregrasp_joints;
        std::array<float,Robot::Configuration::num_scalars_rounded> a{},b{};
        std::copy_n(source_joints.begin(),6,a.begin());
        std::copy_n(goal.begin(),6,b.begin());
        std::array<float,6> goal_joints{};std::copy_n(goal.begin(),6,goal_joints.begin());
        const auto retreat_pose=Robot::eefk(goal_joints);
        const auto local_safe=[&](const std::vector<float> &joints) {
          if (!jointsWithinPiperLimits(joints)) return false;
          std::array<float,6> q{};std::copy_n(joints.begin(),6,q.begin());
          const auto pose=Robot::eefk(q);
          return (pose.translation()-source_pose.translation()).norm()<=0.18F &&
            pose.translation().z()>=source_pose.translation().z()-0.001F &&
            std::abs(Eigen::AngleAxisf(pose.linear()*source_pose.linear().transpose()).angle())<=0.40F &&
            held_safe(joints);
        };
        std::vector<float> target(goal.begin(),goal.end());
        std::vector<float> source_start(source_joints.begin(),source_joints.end());
        lift_noise=placement_noise_policy_;
        if (jointsWithinPiperLimits(target) &&
            retreat_pose.translation().z()>=source_pose.translation().z()+requested_lift_rise-0.001F &&
            local_safe(source_start) && local_safe(target)) {
          const auto search_started=std::chrono::steady_clock::now();
          PayloadPiper::ScopedValidator validator([&](const auto &joints) {
            const auto saved=lift_noise;
            const bool clear=seconds_since(search_started)<=2.0 && local_safe(joints);
            lift_noise=saved;
            return clear;
          });
          vamp::planning::RRTCSettings settings;
          settings.max_iterations=std::min(1000,std::max(1,placement_transfer_max_iterations_));
          settings.max_samples=settings.max_iterations;settings.range=0.1F;
          auto rng=std::make_shared<vamp::rng::Halton<PayloadPiper>>();
          EnvironmentVector empty;
          const auto escape=[&]() {
            ComponentTimer retreat_timer(PipelineComponent::RRTC);
            return HeldRRTC::solve(Robot::Configuration(a.data()),
              Robot::Configuration(b.data()),empty,settings,rng);
          }();
          std::vector<std::vector<float>> path;
          auto prior=source_start;
          lift_noise=placement_noise_policy_;
          bool valid=!escape.path.empty() && local_safe(prior);
          for (const auto &waypoint:escape.path) {
            if (!valid) break;
            const auto values=waypoint.to_array();
            std::vector<float> next(values.begin(),values.begin()+6);
            auto checked_previous = prior;
            auto segment = planJointApproach(prior, next,
              [](const auto &q) {
                std::array<float,6> values{}; std::copy_n(q.begin(),6,values.begin());
                return Robot::eefk(values);
              }, [&](const auto &q, size_t, bool) {
                std::vector<float> midpoint(6);
                for (size_t j=0;j<6;++j) midpoint[j]=0.5F*(checked_previous[j]+q[j]);
                if (!local_safe(midpoint) || !local_safe(q)) return false;
                checked_previous=q; return true;
              });
            valid=segment.success && path.size()+segment.waypoint_joints.size() <= 8192;
            if (valid) {
              path.insert(path.end(),segment.waypoint_joints.begin(),segment.waypoint_joints.end());
              prior=next;
            }

          }
          if (valid && !path.empty()) {
            lift.success=true;lift.failure_reason.clear();lift.waypoint_joints=std::move(path);
            departure_path.clear();
            RCLCPP_INFO(this->get_logger(),"[PLACEMENT LIFT] validated bounded RRT retreat waypoints=%zu",
              lift.waypoint_joints.size());
          } else if (pipeline_profiling_) {
            RCLCPP_INFO(this->get_logger(),"[PLACEMENT LIFT] bounded RRT retreat rejected path=%zu resampled=%zu reason=%s",
              escape.path.size(),path.size(),held_failure_reason.c_str());
          }
        } else if (pipeline_profiling_) {
          RCLCPP_INFO(this->get_logger(),"[PLACEMENT LIFT] bounded retreat endpoint rejected reason=%s",
            held_failure_reason.c_str());
        }
      }
      if (!lift.success) {
        // Upright constrains object roll/pitch, not yaw. A fixed-orientation
        // vertical lift can overconstrain a wrist near its joint limit. Try
        // bounded yaw endpoints through the same fully sampled held validator.
        const std::vector<float> start(source_joints.begin(), source_joints.end());
        Eigen::Vector2f radial = source_pose.translation().head<2>();
        if (radial.norm() < 0.001F) radial = Eigen::Vector2f::UnitX();
        radial.normalize();
        for (float lateral : {0.0F, 0.02F, 0.04F}) {
          for (float yaw_degrees : {0.0F, 5.0F, -5.0F, 10.0F, -10.0F, 20.0F, -20.0F}) {
            auto desired = source_pose;
            desired.translation().head<2>() -= lateral * radial;
            desired.translation().z() += requested_lift_rise;
            desired.linear() = Eigen::AngleAxisf(yaw_degrees * static_cast<float>(M_PI)/180.0F,
              Eigen::Vector3f::UnitZ()).toRotationMatrix() * source_pose.linear();
            std::vector<float> endpoint;
            PoseIkDiagnostics diagnostics;
            if (!solvePiperPoseIK(desired, start, endpoint, diagnostics, true,
                std::min(placement_position_tolerance_m_, 0.001F),
                std::min(placement_descent_orientation_tolerance_rad_, 0.01F), 600)) continue;
            lift_noise = placement_noise_policy_;
            auto route = planJointApproach(start, endpoint,
              [](const auto &q) {
                std::array<float,6> values{}; std::copy_n(q.begin(),6,values.begin());
                return Robot::eefk(values);
              }, [&](const auto &q, size_t, bool) {
                std::array<float,6> values{}; std::copy_n(q.begin(),6,values.begin());
                const auto pose = Robot::eefk(values);
                const auto displacement = pose.translation()-source_pose.translation();
                if (displacement.z() < -0.001F || displacement.z() > requested_lift_rise+0.01F ||
                    displacement.head<2>().norm() > 0.06F) return false;
                return held_safe(q);
              });
            if (route.success) {
              lift = std::move(route); departure_path.clear();
              RCLCPP_INFO(this->get_logger(),
                "[PLACEMENT LIFT] bounded upright yaw route accepted yaw=%.1f inward=%.3f",
                yaw_degrees, lateral);
              break;
            }
          }
          if (lift.success) break;
        }
      }
      if (!lift.success) {
        // Fixed orientation may pin a wrist joint at its limit. Try bounded
        // relaxed lift endpoints, then validate every sampled joint transition.
        // Never shorten the requested vertical clearance or relax joint bounds.
        const std::vector<float> start(source_joints.begin(), source_joints.end());
        Eigen::Vector2f radial = source_pose.translation().head<2>();
        if (radial.norm() < 0.001F) radial = Eigen::Vector2f::UnitX();
        radial.normalize();
        const Eigen::Vector2f sideways(-radial.y(), radial.x());
        const float rise = request->lift_distance_m == 0.0F ? 0.10F : request->lift_distance_m;
        for (const Eigen::Vector2f offset : {
            Eigen::Vector2f::Zero().eval(), (-0.01F*radial).eval(),
            (-0.02F*radial).eval(), (-0.04F*radial).eval(),
            (0.02F*sideways).eval(), (-0.02F*sideways).eval()}) {
          auto desired = source_pose;
          desired.translation().head<2>() += offset;
          desired.translation().z() += rise;
          Eigen::Isometry3f resolved;
          std::vector<float> endpoint;
          PoseIkDiagnostics diagnostics;
          if (!solveRelaxedPregraspIK(desired, start, endpoint, diagnostics, resolved)) continue;
          if (std::abs(resolved.translation().z()-desired.translation().z()) > 0.001F ||
              (resolved.translation().head<2>()-desired.translation().head<2>()).norm() > 0.001F ||
              Eigen::AngleAxisf(source_pose.linear().transpose()*resolved.linear()).angle() > 0.35F)
            continue;
          lift_noise = placement_noise_policy_;
          auto route = planJointApproach(start, endpoint,
            [](const auto &q) {
              std::array<float,6> values{}; std::copy_n(q.begin(),6,values.begin());
              return Robot::eefk(values);
            }, [&](const auto &q, size_t, bool) {
              std::array<float,6> values{}; std::copy_n(q.begin(),6,values.begin());
              const auto pose = Robot::eefk(values);
              const auto displacement = pose.translation()-source_pose.translation();
              if (displacement.z() < -0.001F || displacement.z() > rise+0.01F ||
                  displacement.head<2>().norm() > 0.06F ||
                  Eigen::AngleAxisf(source_pose.linear().transpose()*pose.linear()).angle() > 0.35F)
                return false;
              return held_safe(q);
            });
          if (route.success) {
            lift = std::move(route); departure_path.clear();
            RCLCPP_INFO(this->get_logger(), "[PLACEMENT LIFT] bounded relaxed joint route accepted");
            break;
          }
        }
      }
      response->lift_seconds=seconds_since(started);
      RCLCPP_INFO(this->get_logger(),
        "[ATTACHED NOISE BUDGET] stage=lift accepted_points=%zu observed_points=%zu voxels=%zu",
        lift_noise.points.size(),lift_noise.observed_points,lift_noise.observed_voxels);
      if (!lift.success) {
        response->motion_status="LIFT_"+lift.failure_reason;
        if (placement_bypass_attached_collision_during_lift_) {
          response->motion_status += ";LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED";
        }
        if (lift.failure_reason.rfind("COLLISION_", 0) == 0 &&
            !held_failure_reason.empty()) {
          response->motion_status += ":" + held_failure_reason;
        }
        if (placement_bypass_robot_environment_during_lift_)
          response->motion_status += ";LIFT_ROBOT_ENVIRONMENT_CHECK_BYPASSED";
        response->status=response->motion_status;
        RCLCPP_WARN(this->get_logger(),"[PLACEMENT MOTION] lift=%.6fs failure=%s",
          response->lift_seconds,response->motion_status.c_str());
        return;
      }
      if (!departure_path.empty()) {
        lift.waypoint_joints.insert(lift.waypoint_joints.begin(),
          departure_path.begin(),departure_path.end()-1);
      }
      if (keep_object_upright_) {
        const auto checked=vamp_node::validateUprightPath(std::vector<float>(source_joints.begin(),source_joints.end()),
          lift.waypoint_joints,[](const auto &q) { std::array<float,6> a{};std::copy_n(q.begin(),6,a.begin());return Robot::eefk(a); },
          attachment_pose,object_upright_tolerance_rad_,[&](const auto &q,size_t,bool) { return held_safe(q); });
        if (!checked.success) { response->status=response->motion_status="LIFT_UPRIGHT_PATH_REJECTED:"+checked.failure_reason; return; }
        lift.waypoint_joints=checked.waypoint_joints;
      }
      // Restore the complete transfer checker at the predicted lift endpoint.
      // If it is blocked, APF guides a bounded, contact-separating payload escape.
      // The whole route is calculated now, before any hardware execution.
      lifting=false;
      if (!held_safe(lift.waypoint_joints.back())) {
        const auto escape_started=std::chrono::steady_clock::now();
        const vamp_node::PayloadEscapeField field(environment_points);
        const auto probes_at=[&](const std::vector<float> &q) {
          std::array<float,6> values{};std::copy_n(q.begin(),6,values.begin());
          const auto pose=Robot::eefk(values);
          Robot::ConfigurationBlock<1> configuration;
          for (size_t j=0;j<6;++j) configuration[j][0]=q[j];
          Robot::Spheres<1> spheres;
          Robot::template sphere_fk<1>(configuration,spheres);
          std::vector<vamp_node::PayloadEscapeField::Probe> probes;
          probes.reserve(Robot::n_spheres+source_points.size()/4);
          for (size_t sphere=0;sphere<Robot::n_spheres;++sphere) {
            const auto center=vamp_node::piperFingerSphereCenter(sphere,Eigen::Vector3f(
              spheres.x[{sphere,0}],spheres.y[{sphere,0}],spheres.z[{sphere,0}]),
              pose,selected_grasp.jaw_opening_m);
            probes.push_back({center,spheres.r[{sphere,0}]+.0025F,
              [sphere,pose,opening=selected_grasp.jaw_opening_m](const auto &point) {
                return vamp_node::piperFingerMeshBoundsContain(sphere,point,pose,opening);
              }});
          }
          const Eigen::Isometry3f delta=pose*attachment_pose.inverse();
          for (const auto &point:source_points)
            probes.push_back({delta*point,.003F,{}});
          return probes;
        };
        auto current=lift.waypoint_joints.back();
        std::array<float,6> initial_array{};std::copy_n(current.begin(),6,initial_array.begin());
        const auto initial_pose=Robot::eefk(initial_array);
        const auto initial=field.evaluate(probes_at(current));
        auto best=initial;
        std::set<vamp_node::PayloadEscapeField::Cell> segment_contacts=initial.contacts;
        std::vector<std::vector<float>> escape_path;
        bool escaped=false;
        std::string escape_trial_failure;
        for (int iteration=0;iteration<32 && seconds_since(escape_started)<4.;++iteration) {
          std::array<float,6> values{};std::copy_n(current.begin(),6,values.begin());
          const auto current_pose=Robot::eefk(values);
          auto direction=best.force;
          if (!direction.allFinite()) break;
          if (direction.norm()>1.e-6F) direction.normalize();
          direction.z()=std::max(.35F,direction.z());direction.normalize();
          bool advanced=false;
          std::vector<Eigen::Vector3f> directions{direction};
          for (float bias : {.4F,-.4F}) {
            Eigen::Vector3f step=direction;
            step.x()+=bias*direction.y();step.y()-=bias*direction.x();
            directions.push_back(step.normalized());
          }
          directions.push_back(Eigen::Vector3f::UnitZ());
          Eigen::Vector3f inward(-current_pose.translation().x(),-current_pose.translation().y(),0.F);
          if (inward.norm()>1.e-6F) {inward.normalize();inward.z()=.5F;directions.push_back(inward.normalized());}
          for (size_t trial=0;trial<directions.size()*2;++trial) {
            if (seconds_since(escape_started)>=4.) break;
            const auto &step=directions[trial/2];
            const float step_size=trial%2==0 ? .005F : .001F;
            auto desired=current_pose;desired.translation()+=step_size*step;
            auto proposed_best=best;
            auto proposed_contacts=segment_contacts;
            escape_contact_safe=[&](const auto &q) {
              std::array<float,6> a{};std::copy_n(q.begin(),6,a.begin());
              const auto pose=Robot::eefk(a);
              const auto displacement=pose.translation()-initial_pose.translation();
              if (displacement.head<2>().norm()>.10F || displacement.z()<-.001F ||
                  displacement.z()>.10F || displacement.norm()>.12F) return false;
              const auto state=field.evaluate(probes_at(q));
              if (!vamp_node::PayloadEscapeField::admissible(initial,proposed_best,state,proposed_contacts)) {
                std::ostringstream reason;
                reason<<"APF_CONTACT_BUDGET energy="<<state.energy<<" initial="<<initial.energy
                  <<" best="<<proposed_best.energy<<" cells="<<state.contacts.size()
                  <<" initial_cells="<<initial.contacts.size()<<" depth="<<state.penetration;
                held_failure_reason=reason.str();return false;
              }
              proposed_contacts.insert(state.contacts.begin(),state.contacts.end());
              if (state.energy<proposed_best.energy) proposed_best=state;
              return true;
            };
            escaping=true;
            auto prior=current;
            auto route=planCartesianApproach(current_pose,desired,current,.002F,.05F,
              [&](const auto &pose,const auto &seed,std::vector<float> &solution) {
                PoseIkDiagnostics diagnostics;
                return solvePiperPoseIK(pose,seed,solution,diagnostics,true,.0005F,.01F,600);
              },[&](const auto &q,size_t,bool) {
                std::vector<float> midpoint(6);
                for (size_t j=0;j<6;++j) midpoint[j]=(prior[j]+q[j])*.5F;
                if (!held_safe(midpoint) || !held_safe(q)) return false;
                prior=q;return true;
              });
            if (route.success) {
              proposed_best=best;
              proposed_contacts=segment_contacts;
              // Also sample joint transitions at <=0.005 rad / <=2 mm TCP;
              // midpoint-only Cartesian samples are insufficient near a wrist limit.
              auto checked=vamp_node::validateUprightPath(current,route.waypoint_joints,
                [](const auto &q) {std::array<float,6> a{};std::copy_n(q.begin(),6,a.begin());return Robot::eefk(a);},
                attachment_pose,keep_object_upright_ ? object_upright_tolerance_rad_ : static_cast<float>(M_PI),
                [&](const auto &q,size_t,bool){return held_safe(q);});
              if (checked.success) route=std::move(checked);
              else {route.success=false;route.failure_reason=checked.failure_reason;}
            }
            escaping=false;
            escape_contact_safe={};
            if (!route.success) {escape_trial_failure=route.failure_reason;continue;}
            current=route.waypoint_joints.back();
            escape_path.insert(escape_path.end(),route.waypoint_joints.begin(),route.waypoint_joints.end());
            best=field.evaluate(probes_at(current));segment_contacts=std::move(proposed_contacts);
            advanced=true;
            if (held_safe(current)) escaped=true;
            break;
          }
          if (escaped || !advanced) break;
        }
        if (!escaped) {
          response->motion_status="LIFT_APF_ESCAPE_EXHAUSTED:"+held_failure_reason;
          if (placement_bypass_robot_environment_during_lift_)
            response->motion_status+=";LIFT_ROBOT_ENVIRONMENT_CHECK_BYPASSED";
          if (placement_bypass_attached_collision_during_lift_)
            response->motion_status+=";LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED";
          response->status=response->motion_status;
          response->lift_seconds=seconds_since(started);
          RCLCPP_WARN(this->get_logger(),"[PAYLOAD APF] failed seconds=%.3f contact_cells=%zu penetration=%.2fmm trial=%s detail=%s",
            seconds_since(escape_started),segment_contacts.size(),best.penetration*1000.F,
            escape_trial_failure.c_str(),held_failure_reason.c_str());
          return;
        }
        lift.waypoint_joints.insert(lift.waypoint_joints.end(),escape_path.begin(),escape_path.end());
        escape_status=";LIFT_APF_ESCAPE_VALIDATED";
        RCLCPP_INFO(this->get_logger(),"[PAYLOAD APF] accepted seconds=%.3f waypoints=%zu contact_cells=%zu penetration=%.2fmm",
          seconds_since(escape_started),escape_path.size(),segment_contacts.size(),best.penetration*1000.F);
      }
      response->lift_seconds=seconds_since(started);
      for (const auto &waypoint:lift.waypoint_joints)
        response->lift_path_flat.insert(response->lift_path_flat.end(),waypoint.begin(),waypoint.end());
      std::array<float,6> lift_end{};
      std::copy_n(lift.waypoint_joints.back().begin(), 6, lift_end.begin());
      pose_to_message(Robot::eefk(lift_end), response->lift_ee_pose);
      response->lift_ee_pose.header = source_cloud->header;
      response->lift_joints = lift_end;
      if (request->lift_only) {
        response->success = response->motion_success = true;
        response->status = "LIFT_PLAN_READY";
        response->motion_status = keep_object_upright_ ? "LIFT_VALIDATED;OBJECT_UPRIGHT_VALIDATED" : "LIFT_VALIDATED";
        response->motion_status += escape_status;
        if (placement_bypass_robot_environment_during_lift_)
          response->motion_status += ";LIFT_ROBOT_ENVIRONMENT_CHECK_BYPASSED";
        if (placement_bypass_attached_collision_during_lift_)
          response->motion_status += ";LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED";
        return;
      }
      lifting=false;
      response->transfer_collision_samples=0;
      started=std::chrono::steady_clock::now();
      std::array<float,Robot::Configuration::num_scalars_rounded> start_array{},goal_array{};
      std::copy_n(lift.waypoint_joints.back().begin(),6,start_array.begin());
      std::copy_n(best->preplace_joints.begin(),6,goal_array.begin());
      const Robot::Configuration start_configuration(start_array.data()),goal_configuration(goal_array.data());
      EnvironmentVector empty_environment;
      vamp::planning::RRTCSettings settings;
      settings.max_iterations=std::max(1,placement_transfer_max_iterations_);
      settings.max_samples=settings.max_iterations;
      settings.range=0.1F;
      auto rng=std::make_shared<vamp::rng::Halton<PayloadPiper>>();
      bool timed_out=false;
      PayloadPiper::ScopedValidator scoped_validator([&](const auto &joints) {
        if (seconds_since(started)>std::max(0.01,placement_transfer_timeout_sec_)) {
          timed_out=true; return false;
        }
        return held_safe(joints);
      });
      held_failure_reason.clear();
      if (!held_safe(lift.waypoint_joints.back())) {
        response->motion_status="TRANSFER_LIFT_ENDPOINT_REJECTED:"+held_failure_reason;
      } else if (!held_safe(best->preplace_joints)) {
        response->motion_status="TRANSFER_PREPLACE_ENDPOINT_REJECTED:"+held_failure_reason;
      } else {
        ComponentTimer rrtc_timer(PipelineComponent::RRTC);
        const auto result=HeldRRTC::solve(start_configuration,goal_configuration,empty_environment,settings,rng);
        for (const auto &waypoint:result.path) {
          const auto array=waypoint.to_array();
          response->transfer_path_flat.insert(response->transfer_path_flat.end(),array.begin(),array.begin()+6);
        }
        response->motion_success=!result.path.empty();
        response->motion_status=response->motion_success ? "LIFT_TRANSFER_DESCENT_VALIDATED" :
          timed_out ? "TRANSFER_RRTC_TIMEOUT" : "TRANSFER_RRTC_NO_PATH";
      }
      if (keep_object_upright_ && response->motion_success) {
        std::vector<std::vector<float>> path;
        for (size_t i=0;i<response->transfer_path_flat.size();i+=6)
          path.emplace_back(response->transfer_path_flat.begin()+i,response->transfer_path_flat.begin()+i+6);
        const auto checked=vamp_node::validateUprightPath(lift.waypoint_joints.back(),path,
          [](const auto &q) { std::array<float,6> a{};std::copy_n(q.begin(),6,a.begin());return Robot::eefk(a); },
          attachment_pose,object_upright_tolerance_rad_,[&](const auto &q,size_t,bool) { return held_safe(q); });
        response->transfer_path_flat.clear();
        response->motion_success=checked.success;
        response->motion_status=checked.success ? "LIFT_TRANSFER_DESCENT_VALIDATED;OBJECT_UPRIGHT_VALIDATED" : "TRANSFER_UPRIGHT_PATH_REJECTED:"+checked.failure_reason;
        if (checked.success) for (const auto &q:checked.waypoint_joints)
          response->transfer_path_flat.insert(response->transfer_path_flat.end(),q.begin(),q.end());
      }
      response->transfer_seconds=seconds_since(started);
      response->motion_status += escape_status;
      if (placement_bypass_robot_environment_during_lift_)
        response->motion_status += ";LIFT_ROBOT_ENVIRONMENT_CHECK_BYPASSED";
      if (placement_bypass_attached_collision_during_lift_) {
        response->motion_status += ";LIFT_ATTACHED_ENVIRONMENT_CHECK_BYPASSED";
      }
      RCLCPP_INFO(this->get_logger(),"[PLACEMENT MOTION] lift=%.6fs transfer=%.6fs lift_waypoints=%zu transfer_waypoints=%zu collision_samples=%u status=%s",
        response->lift_seconds,response->transfer_seconds,response->lift_path_flat.size()/6,
        response->transfer_path_flat.size()/6,response->transfer_collision_samples,response->motion_status.c_str());
      if (!response->motion_success) { response->status=response->motion_status; return; }
    }

    sensor_msgs::msg::PointCloud2 placed_cloud;
    placed_cloud.header.frame_id = "base_link";
    placed_cloud.header.stamp = this->now();
    sensor_msgs::PointCloud2Modifier modifier(placed_cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(source_points.size());
    sensor_msgs::PointCloud2Iterator<float> placed_x(placed_cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> placed_y(placed_cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> placed_z(placed_cloud, "z");
    for (const auto &source_point : source_points) {
      const Eigen::Vector3f point = best->object_delta * source_point;
      *placed_x = point.x(); *placed_y = point.y(); *placed_z = point.z();
      ++placed_x; ++placed_y; ++placed_z;
    }
    refined_placed_object_cloud_pub_->publish(placed_cloud);

    // Show the inferred source support contour at the accepted placement.
    // Use the same resolved-IK object transform as placed_cloud so this view
    // includes the actual placement offset accepted by the planner.
    sensor_msgs::msg::PointCloud2::SharedPtr source_support_surface;
    {
      std::lock_guard<std::mutex> lock(object_support_surface_mutex_);
      source_support_surface = latest_object_support_surface_;
    }
    if (source_support_surface &&
      source_support_surface->header.frame_id == "base_link" &&
      source_support_surface->width > 0 &&
      source_support_surface->header.stamp.sec == source_cloud->header.stamp.sec &&
      source_support_surface->header.stamp.nanosec == source_cloud->header.stamp.nanosec)
    {
      std::vector<Eigen::Vector3f> support_points;
      sensor_msgs::PointCloud2ConstIterator<float> support_x(
        *source_support_surface, "x");
      sensor_msgs::PointCloud2ConstIterator<float> support_y(
        *source_support_surface, "y");
      sensor_msgs::PointCloud2ConstIterator<float> support_z(
        *source_support_surface, "z");
      for (; support_x != support_x.end();
        ++support_x, ++support_y, ++support_z)
      {
        if (std::isfinite(*support_x) && std::isfinite(*support_y) &&
          std::isfinite(*support_z))
        {
          support_points.emplace_back(*support_x, *support_y, *support_z);
        }
      }
      sensor_msgs::msg::PointCloud2 placed_support;
      placed_support.header = placed_cloud.header;
      sensor_msgs::PointCloud2Modifier support_modifier(placed_support);
      support_modifier.setPointCloud2FieldsByString(1, "xyz");
      support_modifier.resize(support_points.size());
      sensor_msgs::PointCloud2Iterator<float> placed_support_x(
        placed_support, "x");
      sensor_msgs::PointCloud2Iterator<float> placed_support_y(
        placed_support, "y");
      sensor_msgs::PointCloud2Iterator<float> placed_support_z(
        placed_support, "z");
      for (const auto &source_point : support_points) {
        const Eigen::Vector3f point = best->object_delta * source_point;
        *placed_support_x = point.x();
        *placed_support_y = point.y();
        *placed_support_z = point.z();
        ++placed_support_x; ++placed_support_y; ++placed_support_z;
      }
      refined_placed_support_surface_pub_->publish(placed_support);
    } else {
      RCLCPP_WARN(
        this->get_logger(),
        "[PLACEMENT PLAN] Accepted object support contour unavailable or stale; "
        "the placed support surface was not published.");
    }

    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(clear);
    visualization_msgs::msg::Marker approach;
    approach.header = response->placement_ee_pose.header;
    approach.ns = "selected_placement_approach";
    approach.id = 1;
    approach.type = visualization_msgs::msg::Marker::ARROW;
    approach.action = visualization_msgs::msg::Marker::ADD;
    approach.points = {
      response->preplace_ee_pose.pose.position,
      response->placement_ee_pose.pose.position};
    approach.scale.x = 0.007;
    approach.scale.y = 0.014;
    approach.scale.z = 0.020;
    approach.color.r = 0.1F; approach.color.g = 1.0F;
    approach.color.b = 0.2F; approach.color.a = 1.0F;
    markers.markers.push_back(approach);
    const std::array<std::array<float, 3>, 3> colors{{
      {{1.0F, 0.1F, 0.1F}}, {{0.1F, 1.0F, 0.1F}}, {{0.1F, 0.3F, 1.0F}}}};
    for (int axis = 0; axis < 3; ++axis) {
      visualization_msgs::msg::Marker marker;
      marker.header = response->placement_ee_pose.header;
      marker.ns = "selected_placement_ee";
      marker.id = 10 + axis;
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;
      geometry_msgs::msg::Point start = response->placement_ee_pose.pose.position;
      geometry_msgs::msg::Point end = start;
      end.x += best->resolved.linear()(0, axis) * 0.06F;
      end.y += best->resolved.linear()(1, axis) * 0.06F;
      end.z += best->resolved.linear()(2, axis) * 0.06F;
      marker.points = {start, end};
      marker.scale.x = 0.006; marker.scale.y = 0.012; marker.scale.z = 0.016;
      marker.color.r = colors[axis][0]; marker.color.g = colors[axis][1];
      marker.color.b = colors[axis][2]; marker.color.a = 1.0F;
      markers.markers.push_back(marker);
    }
    const Eigen::Vector3f hand_base = best->resolved.translation() -
      best->resolved.linear().col(2) * static_cast<float>(grasp_tcp_offset_m_);
    const Eigen::Quaternionf finger_rotation(visualGripperRotation(
      best->resolved.linear().col(1), best->resolved.linear().col(2)));
    response->placement_finger_dimensions.x = finger_width_m_;
    response->placement_finger_dimensions.y = finger_height_m_;
    response->placement_finger_dimensions.z = finger_length_m_;
    for (int finger = 0; finger < 2; ++finger) {
      visualization_msgs::msg::Marker marker;
      marker.header = response->placement_ee_pose.header;
      marker.ns = "selected_placement_fingers";
      marker.id = 20 + finger;
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      const Eigen::Vector3f center = hand_base +
        best->resolved.linear().col(2) * static_cast<float>(finger_length_m_*0.5) +
        (finger == 0 ? 1.0F : -1.0F)*best->resolved.linear().col(1)*
          static_cast<float>((selected_grasp.jaw_opening_m + finger_width_m_)*0.5);
      marker.pose.position.x = center.x();
      marker.pose.position.y = center.y();
      marker.pose.position.z = center.z();
      marker.pose.orientation.x = finger_rotation.x();
      marker.pose.orientation.y = finger_rotation.y();
      marker.pose.orientation.z = finger_rotation.z();
      marker.pose.orientation.w = finger_rotation.w();
      marker.scale = response->placement_finger_dimensions;
      marker.color.g = 1.0F; marker.color.a = 0.75F;
      response->placement_finger_poses.push_back(marker.pose);
      markers.markers.push_back(marker);
    }
    placement_plan_markers_pub_->publish(markers);

    std::ostringstream status;
    status << std::fixed << std::setprecision(1)
           << "PLACEMENT_PLAN_READY candidate=#"
           << response->selected_candidate.id << " tested="
           << request->candidates.size() << " policy="
           << request->orientation_policy << " error={position:"
           << best->position_error * 1000.0F << "mm,orientation:"
           << best->rotation_error * 180.0F / static_cast<float>(M_PI)
           << "deg} preplace=" << (best->preplace.translation()-best->resolved.translation()).norm()*1000.0F
           << "mm calculation_only=true";
    response->status = status.str();
    response->success = true;
    {
      std::lock_guard<std::mutex> lock(placement_plan_mutex_);
      cached_placement_plan_ = *response;
    }
    RCLCPP_INFO(this->get_logger(), "[PLACEMENT PLAN] %s", response->status.c_str());
  }

  void handlePlanCartesianGraspApproach(
    const std::shared_ptr<my_actions::srv::PlanCartesianGraspApproach::Request> request,
    std::shared_ptr<my_actions::srv::PlanCartesianGraspApproach::Response> response)
  {
    response->success = false;
    response->waypoint_count = 0;
    response->path_flat.clear();
    const auto fail = [&response](const std::string &reason) {
        response->failure_reason = reason;
      };

    if (request->current_pregrasp_pose.header.frame_id != "base_link" ||
      request->refined_grasp_pose.header.frame_id != "base_link")
    {
      fail("FINAL_APPROACH_POSE_FRAME_MUST_BE_BASE_LINK");
      return;
    }
    if (request->current_joints.size() != 6) {
      fail("FINAL_APPROACH_CURRENT_JOINTS_MUST_HAVE_SIX_VALUES");
      return;
    }

    const auto poseFromMessage = [](
      const geometry_msgs::msg::PoseStamped &message,
      Eigen::Isometry3f &pose) {
        const auto &orientation = message.pose.orientation;
        Eigen::Quaternionf quaternion(
          static_cast<float>(orientation.w), static_cast<float>(orientation.x),
          static_cast<float>(orientation.y), static_cast<float>(orientation.z));
        if (!quaternion.coeffs().allFinite() ||
          quaternion.squaredNorm() < 1.0e-8F)
        {
          return false;
        }
        quaternion.normalize();
        pose = Eigen::Isometry3f::Identity();
        pose.linear() = quaternion.toRotationMatrix();
        pose.translation() = Eigen::Vector3f(
          static_cast<float>(message.pose.position.x),
          static_cast<float>(message.pose.position.y),
          static_cast<float>(message.pose.position.z));
        return pose.matrix().allFinite();
      };

    Eigen::Isometry3f current_pose;
    Eigen::Isometry3f grasp_pose;
    if (!poseFromMessage(request->current_pregrasp_pose, current_pose) ||
      !poseFromMessage(request->refined_grasp_pose, grasp_pose))
    {
      fail("FINAL_APPROACH_INVALID_POSE");
      return;
    }
    if (!jointsWithinPiperLimits(request->current_joints)) {
      fail("FINAL_APPROACH_START_JOINT_LIMIT_VIOLATION");
      return;
    }

    const Eigen::Vector3f contact_center(
      static_cast<float>(request->refined_contact_center.x),
      static_cast<float>(request->refined_contact_center.y),
      static_cast<float>(request->refined_contact_center.z));
    Eigen::Vector3f closing_axis(
      static_cast<float>(request->closing_axis.x),
      static_cast<float>(request->closing_axis.y),
      static_cast<float>(request->closing_axis.z));
    const float grasp_width = request->grasp_width_m;
    if (!contact_center.allFinite() || !closing_axis.allFinite() ||
      closing_axis.squaredNorm() < 1.0e-8F ||
      !std::isfinite(grasp_width) ||
      grasp_width < static_cast<float>(minimum_gripper_width_) ||
      grasp_width > static_cast<float>(maximum_gripper_width_))
    {
      fail("FINAL_APPROACH_INVALID_REFINED_CONTACT_GEOMETRY");
      return;
    }
    closing_axis.normalize();
    const Eigen::Vector3f contact_left =
      contact_center + closing_axis * (0.5F * grasp_width);
    const Eigen::Vector3f contact_right =
      contact_center - closing_axis * (0.5F * grasp_width);

    const int64_t target_stamp_ns = rclcpp::Time(
      request->target_observation_header.stamp).nanoseconds();
    std::vector<Point> environment;
    std::vector<uint8_t> target_flags;
    const bool frozen = request->source_object_cloud.width || request->collision_scene.width;
    if (frozen) {
      const auto append_cloud = [&](const sensor_msgs::msg::PointCloud2 &cloud, uint8_t target) {
        bool valid = cloud.header.frame_id == "base_link" &&
          rclcpp::Time(cloud.header.stamp).nanoseconds() == target_stamp_ns &&
          target_stamp_ns > 0 && cloud.width && cloud.height && !cloud.is_bigendian &&
          cloud.point_step > 0 && cloud.row_step == cloud.width * cloud.point_step &&
          cloud.data.size() == static_cast<size_t>(cloud.row_step) * cloud.height &&
          cloud.data.size() <= 16 * 1024 * 1024;
        for (const auto *name : {"x", "y", "z"}) {
          valid = valid && std::any_of(cloud.fields.begin(), cloud.fields.end(),
            [&](const auto &field) { return field.name == name && field.count == 1 &&
              field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
              field.offset <= cloud.point_step && cloud.point_step-field.offset >= 4; });
        }
        if (!valid) return false;
        sensor_msgs::PointCloud2ConstIterator<float> x(cloud,"x"), y(cloud,"y"), z(cloud,"z");
        for (; x != x.end(); ++x,++y,++z) {
          if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z)) {
            environment.push_back(Point{*x,*y,*z});
            target_flags.push_back(target);
          }
        }
        return true;
      };
      if (!append_cloud(request->collision_scene,0) ||
          !append_cloud(request->source_object_cloud,1)) {
        fail("FINAL_APPROACH_INVALID_FROZEN_SCENE");
        return;
      }
    } else {
    std::vector<Eigen::Vector3f> cached_contact_region;
    {
      std::lock_guard<std::mutex> lock(refinement_target_mutex_);
      if (target_stamp_ns <= 0 || refinement_target_stamp_ns_ != target_stamp_ns) {
        fail("FINAL_APPROACH_TARGET_REGION_STALE");
        return;
      }
      cached_contact_region = refinement_contact_region_;
    }
    if (cached_contact_region.empty()) {
      fail("FINAL_APPROACH_TARGET_REGION_UNAVAILABLE");
      return;
    }

    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      environment = getFilteredEnvironment(request->current_joints);
    }
    target_flags.assign(environment.size(), 0);
    const float target_radius_sq = static_cast<float>(
      target_collision_clearance_m_ * target_collision_clearance_m_);
    for (std::size_t index = 0; index < environment.size(); ++index) {
      const Eigen::Vector3f obstacle(
        environment[index][0], environment[index][1], environment[index][2]);
      target_flags[index] = std::any_of(
        cached_contact_region.begin(), cached_contact_region.end(),
        [&obstacle, target_radius_sq](const Eigen::Vector3f &target) {
          return (obstacle - target).squaredNorm() <= target_radius_sq;
        });
    }

    }

    const auto fk = [](const std::vector<float> &joints) {
      std::array<float, 6> values{};
      std::copy_n(joints.begin(), 6, values.begin());
      return Robot::eefk(values);
    };
    const auto actual_start = fk(request->current_joints);
    if ((actual_start.translation()-current_pose.translation()).norm() > 0.004F ||
        Eigen::AngleAxisf(actual_start.linear().transpose()*current_pose.linear()).angle() > 0.03F) {
      fail("PREGRASP_POSE_JOINT_FK_MISMATCH"); return;
    }
    const float open_width = std::min(0.10F, grasp_width+0.02F);
    std::vector<float> last_approach_collision_joints;
    const auto collision = [&](const std::vector<float> &joints, std::size_t, bool) {
      const bool contact = (fk(joints).translation()-grasp_pose.translation()).norm()
        <= final_approach_contact_phase_m_+1.0e-6F;
      const bool safe = isCandidatePoseSafe(joints, environment, contact ? &target_flags : nullptr,
        contact ? &contact_left : nullptr, contact ? &contact_right : nullptr,
        contact ? static_cast<float>(target_contact_surface_tolerance_m_) : 0.0F,
        contact ? &grasp_pose : nullptr, open_width,
        final_approach_check_finger_environment_,
        true);
      if (!safe) last_approach_collision_joints = joints;
      return safe;
    };
    const auto collision_detail = [&](const std::vector<float> &joints) {
      const bool contact = (fk(joints).translation()-grasp_pose.translation()).norm()
        <= final_approach_contact_phase_m_+1.0e-6F;
      return describeCandidateCollision(joints, environment,
        contact ? &target_flags : nullptr, contact ? &contact_left : nullptr,
        contact ? &contact_right : nullptr,
        contact ? static_cast<float>(target_contact_surface_tolerance_m_) : 0.0F,
        contact ? &grasp_pose : nullptr, open_width,
        final_approach_check_finger_environment_, false,
        true);
    };
    const auto endpoint_ik = [&](const Eigen::Isometry3f &pose,
        const std::vector<float> &seed, std::vector<float> &joints) {
      PoseIkDiagnostics diagnostics;
      return solvePiperPoseIK(pose, seed, joints, diagnostics, false,
        0.0005F, 0.01F, 800, [&](const auto &q) { return collision(q, 0, false); }, true);
    };
    const auto &goal = request->final_grasp_joints;
    if (goal.size() != 6 || !jointsWithinPiperLimits(goal)) {
      fail("VALIDATED_FINAL_GRASP_JOINTS_REQUIRED"); return;
    }
    const auto goal_pose = fk(goal);
    if ((goal_pose.translation()-grasp_pose.translation()).norm() > .004F ||
        Eigen::AngleAxisf(goal_pose.linear().transpose()*grasp_pose.linear()).angle() > .20F) {
      fail("FINAL_GRASP_POSE_JOINT_FK_MISMATCH"); return;
    }
    if (!collision(goal, 0, true)) {
      fail("FINAL_GRASP_ENDPOINT_COLLISION: "+collision_detail(goal)); return;
    }
    const Eigen::Vector3f offset = actual_start.translation()-grasp_pose.translation();
    std::vector<std::vector<float>> optional;
    std::vector<float> near_seed = request->current_joints;
    if (offset.norm() > .02F) {
      for (const float distance : {.02F, .01F}) {
        Eigen::Isometry3f near = grasp_pose;
        near.translation() += offset.normalized()*distance;
        std::vector<float> near_joints;
        if (!endpoint_ik(near, near_seed, near_joints)) break;
        optional.push_back(near_joints);
        near_seed = near_joints;
      }
    }
    auto complete = planOptionalJointApproach(
      request->current_joints, goal, optional, fk, collision);
    const bool used_near = complete.failure_reason == "OK;NEAR_GRASP_USED";
    if (!complete.success) {
      fail(complete.failure_reason + (last_approach_collision_joints.empty() ? "" :
        ": "+collision_detail(last_approach_collision_joints)));
      return;
    }
    response->planned_pregrasp_pose = request->current_pregrasp_pose;
    response->planned_pregrasp_joints = request->current_joints;
    response->alignment_waypoint_count = 0;
    for (const auto &joints : complete.waypoint_joints) {
      response->path_flat.insert(response->path_flat.end(), joints.begin(), joints.end());
    }
    response->waypoint_count = static_cast<uint32_t>(complete.waypoint_joints.size());
    response->success = true;
    response->failure_reason = used_near ? "OK;NEAR_GRASP_USED" : "OK;NEAR_GRASP_SKIPPED";
    if (!final_approach_check_finger_environment_)
      response->failure_reason += ";FINAL_APPROACH_FINGER_ENVIRONMENT_CHECK_BYPASSED";
    const float approach_distance = offset.norm();
    RCLCPP_INFO(
      this->get_logger(),
      "[FINAL APPROACH] distance=%.1fmm waypoints=%u step<=%.1fmm "
      "joint_delta<=%.3frad contact_phase=%.1fmm status=VALID",
      approach_distance * 1000.0F, response->waypoint_count,
      final_approach_translation_step_m_ * 1000.0F,
      final_approach_max_joint_step_rad_,
      final_approach_contact_phase_m_ * 1000.0F);
  }

  void handleFindCloudGrasps(
    const std::shared_ptr<my_actions::srv::FindCloudGrasps::Request> request,
    std::shared_ptr<my_actions::srv::FindCloudGrasps::Response> response) {
    const auto matches = [&request](const auto &cloud) {
      return cloud.header.frame_id == "base_link" &&
        cloud.header.stamp == request->observation_header.stamp && cloud.width > 0;
    };
    if (!matches(request->object_cloud) || !matches(request->scene_cloud)) {
      response->status = "CLOUD_GRASP_OBSERVATION_MISMATCH";
      return;
    }
    std::vector<Eigen::Vector3f> points;
    points.reserve(std::min<size_t>(500000,
      static_cast<size_t>(request->object_cloud.width) * request->object_cloud.height +
      static_cast<size_t>(request->scene_cloud.width) * request->scene_cloud.height));
    const auto append = [&points](const auto &cloud) {
      sensor_msgs::PointCloud2ConstIterator<float> x(cloud,"x"), y(cloud,"y"), z(cloud,"z");
      for (; x != x.end(); ++x,++y,++z) {
        if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z)) points.emplace_back(*x,*y,*z);
      }
    };
    append(request->object_cloud);
    const size_t object_count = points.size();
    if (object_count < 80 || object_count > 100000) {
      response->status = "CLOUD_GRASP_INVALID_OBJECT_SIZE";
      return;
    }
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto &point : points) center += point;
    center /= static_cast<float>(object_count);
    size_t seed = 0;
    float distance = std::numeric_limits<float>::infinity();
    for (size_t index=0; index<object_count; ++index) {
      const float cost = (points[index]-center).squaredNorm() +
        (points[index].z() < center.z() ? 1.0F : 0.0F);
      if (cost < distance) { distance=cost; seed=index; }
    }
    append(request->scene_cloud);
    if (points.size() > 500000) { response->status="CLOUD_GRASP_SCENE_TOO_LARGE"; return; }
    // A two-row storage adapter is an index mask, not a camera projection.
    // It supplies known cloud membership to the existing geometric pipeline.
    auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header = request->object_cloud.header;
    sensor_msgs::PointCloud2Modifier modifier(*cloud);
    modifier.setPointCloud2FieldsByString(1,"xyz");
    const size_t width = (points.size()+1)/2;
    modifier.resize(width*2);
    cloud->width=static_cast<uint32_t>(width); cloud->height=2;
    cloud->row_step=cloud->width*cloud->point_step;
    sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
    for (size_t index=0; index<width*2; ++index,++x,++y,++z) {
      const auto point = index<points.size() ? points[index] :
        Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
      *x=point.x(); *y=point.y(); *z=point.z();
    }
    auto adapted = std::make_shared<my_actions::srv::FindGraspCandidates::Request>();
    adapted->observation_header=request->observation_header;
    adapted->image_width=static_cast<int32_t>(width); adapted->image_height=2;
    adapted->seed_u=static_cast<int32_t>(seed%width); adapted->seed_v=static_cast<int32_t>(seed/width);
    adapted->target_mask.header=cloud->header;
    adapted->target_mask.width=cloud->width; adapted->target_mask.height=2;
    adapted->target_mask.step=cloud->width; adapted->target_mask.encoding="mono8";
    adapted->target_mask.data.resize(width*2,0);
    std::fill_n(adapted->target_mask.data.begin(),object_count,255);
    adapted->target_description=request->object_name;
    adapted->affordance_type="AUTO";
    auto result=std::make_shared<my_actions::srv::FindGraspCandidates::Response>();
    std::vector<GraspProposal> fixture_seeds;
    if (request->contact_left.size() != request->contact_right.size() ||
        request->contact_left.size() != request->approaches.size() ||
        request->contact_left.size() > 64) {
      response->status = "CLOUD_GRASP_INVALID_SEED_ARRAYS";
      return;
    }
    for (size_t i=0; i<request->contact_left.size(); ++i) {
      const auto &l=request->contact_left[i], &r=request->contact_right[i];
      const auto &a=request->approaches[i];
      GraspProposal proposal;
      proposal.contact_left=Eigen::Vector3f(l.x,l.y,l.z);
      proposal.contact_right=Eigen::Vector3f(r.x,r.y,r.z);
      proposal.closing=proposal.contact_left-proposal.contact_right;
      proposal.approach=Eigen::Vector3f(a.x,a.y,a.z);
      proposal.opening_m=proposal.closing.norm();
      if (!proposal.contact_left.allFinite() || !proposal.contact_right.allFinite() ||
          !proposal.approach.allFinite() || proposal.opening_m < 0.01F ||
          proposal.opening_m > 0.10F || proposal.approach.norm() < 1e-6F) {
        response->status="CLOUD_GRASP_INVALID_SEED_GEOMETRY";
        return;
      }
      proposal.closing.normalize();
      proposal.approach.normalize();
      if (std::abs(proposal.closing.dot(proposal.approach)) > 0.001F) {
        response->status="CLOUD_GRASP_NONORTHOGONAL_SEED";
        return;
      }
      proposal.sample=(proposal.contact_left+proposal.contact_right)*0.5F;
      proposal.hand_base=proposal.sample-proposal.approach*static_cast<float>(grasp_contact_depth_m_);
      proposal.height_axis=proposal.approach.cross(proposal.closing).normalized();
      proposal.has_contacts=true;
      proposal.grasp_skill=LocalGraspSkill::kPowerGrasp;
      proposal.quality=1.0F;
      fixture_seeds.push_back(proposal);
    }
    computeGraspCandidates(adapted,result,cloud,fixture_seeds);
    response->status=result->status;
    for (const auto &candidate:result->candidates) {
      if (candidate.ik_valid && candidate.measured_contacts_valid &&
          candidate.grasp_collision_valid && candidate.pregrasp_collision_valid &&
          !candidate.pregrasp_collision_overridden) response->candidates.push_back(candidate);
    }
    response->success=!response->candidates.empty();
  }

  void handleFindGraspCandidates(
    const std::shared_ptr<my_actions::srv::FindGraspCandidates::Request> request,
    std::shared_ptr<my_actions::srv::FindGraspCandidates::Response> response) {
    computeGraspCandidates(request,response);
  }

  void computeGraspCandidates(
    const std::shared_ptr<my_actions::srv::FindGraspCandidates::Request> request,
    std::shared_ptr<my_actions::srv::FindGraspCandidates::Response> response,
    const sensor_msgs::msg::PointCloud2::SharedPtr &object_scene = nullptr,
    const std::vector<GraspProposal> &fixture_seeds = {}) {
    PipelineProfile profile(pipeline_profiling_, "grasp", [this](const std::string &data) {
      RCLCPP_INFO(this->get_logger(), "[PIPELINE PROFILE] %s", data.c_str());
    });
    const bool frozen_image_layout = request->target_description.rfind("layout: ", 0) == 0;
    const auto computation_started = std::chrono::steady_clock::now();
    std::vector<my_actions::msg::GraspCandidate> angle_trials;
    std::vector<std::pair<std::array<float,6>,
      std::vector<my_actions::msg::GraspCandidate>>> angle_trials_by_grasp;
    const auto angle_trial_publish = std::shared_ptr<void>(nullptr,
      [this, &angle_trials, &angle_trials_by_grasp, response](void*) {
        if (!response->candidates.empty()) {
          angle_trials.clear();
          for (const auto &record : angle_trials_by_grasp) {
            if (record.first==response->candidates.front().grasp_joints) {
              angle_trials=record.second;break;
            }
          }
        }
        publishGraspMarkers(angle_trials, angle_trial_markers_pub_, "angle_trial_", true);
      });
    publishGraspMarkers({}, angle_trial_markers_pub_, "angle_trial_", true);
    publishGraspMarkers({});
    publishGraspMarkers({}, valid_grasp_markers_pub_, "valid_grasp_candidate_", true);
    publishGraspMarkers({}, known_cloud_proposals_markers_pub_, "known_cloud_proposal_");
    publishGraspMarkers({}, hggd_raw_proposals_markers_pub_, "hggd_raw_proposal_");
    publishGraspMarkers({}, hggd_proposals_markers_pub_, "hggd_proposal_");
    publishRejectedGraspMarkers({});
    sensor_msgs::msg::PointCloud2::SharedPtr cloud;
    sensor_msgs::msg::Image::SharedPtr observation_image;
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
    geometry_msgs::msg::TransformStamped::SharedPtr observation_camera_transform;
    std::vector<float> joints;
    bool selected_from_cache = false;
    const rclcpp::Time requested_stamp(request->observation_header.stamp);
    {
      std::lock_guard<std::mutex> lock(observation_mutex_);
      joints = latest_joints_;
      camera_info = latest_camera_info_;
      if (requested_stamp.nanoseconds() == 0) {
        cloud = latest_registered_cloud_;
        observation_image = latest_observation_image_;
        observation_camera_transform = latest_observation_camera_transform_;
      } else {
        for (auto entry = registered_cloud_cache_.rbegin();
             entry != registered_cloud_cache_.rend(); ++entry) {
          if (rclcpp::Time((*entry)->header.stamp) == requested_stamp) {
            cloud = *entry;
            selected_from_cache = latest_registered_cloud_ &&
              rclcpp::Time(latest_registered_cloud_->header.stamp) !=
              requested_stamp;
            break;
          }
        }
        for (auto entry = observation_image_cache_.rbegin();
             entry != observation_image_cache_.rend(); ++entry) {
          if (rclcpp::Time((*entry)->header.stamp) == requested_stamp) {
            observation_image = *entry;
            break;
          }
        }
        for (auto entry = observation_camera_transform_cache_.rbegin();
             entry != observation_camera_transform_cache_.rend(); ++entry) {
          if (rclcpp::Time((*entry)->header.stamp) == requested_stamp) {
            observation_camera_transform = *entry;
            break;
          }
        }
      }
    }
    if ((!cloud || !observation_image || !observation_camera_transform) &&
        requested_stamp.nanoseconds() != 0) {
      std::lock_guard<std::mutex> lock(observation_mutex_);
      if (pinned_layout_cloud_ &&
        rclcpp::Time(pinned_layout_cloud_->header.stamp) == requested_stamp) {
        cloud = pinned_layout_cloud_;
        observation_image = pinned_layout_image_;
        observation_camera_transform = pinned_layout_transform_;
        selected_from_cache = true;
      }
    }
    if (!cloud) {
      response->status = requested_stamp.nanoseconds() == 0 ?
        "NO_SYNCHRONIZED_OBSERVATION" : "OBSERVATION_NOT_IN_CACHE";
      RCLCPP_WARN(
        this->get_logger(),
        "[GRASP POC] Observation %ld is unavailable in the %zu-frame cache.",
        requested_stamp.nanoseconds(), grasp_observation_cache_size_);
      return;
    }
    if (object_scene) {
      if (cloud->header.stamp != object_scene->header.stamp ||
          object_scene->header.frame_id != "base_link") {
        response->status="CLOUD_GRASP_MANUAL_OBSERVATION_MISMATCH";
        return;
      }
      cloud=object_scene;
    }
    if (cloud->height <= 1 || joints.size() != 6) {
      response->status = "NO_SYNCHRONIZED_OBSERVATION";
      return;
    }
    if (selected_from_cache) {
      RCLCPP_INFO(
        this->get_logger(),
        "[GRASP POC] Using cached observation %ld while newer scans continue.",
        requested_stamp.nanoseconds());
    }
    const rclcpp::Time cloud_stamp(cloud->header.stamp);
    if (requested_stamp.nanoseconds() != 0 && requested_stamp != cloud_stamp) {
      response->status = "OBSERVATION_TIMESTAMP_MISMATCH";
      RCLCPP_WARN(this->get_logger(),
                  "[GRASP POC] Rejected mask/cloud mismatch: request=%ld cloud=%ld",
                  requested_stamp.nanoseconds(), cloud_stamp.nanoseconds());
      return;
    }
    if (request->image_width != static_cast<int>(cloud->width) || request->image_height != static_cast<int>(cloud->height)) {
      response->status = "OBSERVATION_DIMENSION_MISMATCH"; return;
    }
    if (cloud->header.frame_id != "base_link") {
      response->status = "MANUAL_X_OBSERVATION_NOT_IN_BASE_LINK";
      RCLCPP_ERROR(
        this->get_logger(),
        "[MANUAL X] Refusing grasp input in frame '%s'; Active Vision must "
        "publish the FK * calibrated-X observation in base_link.",
        cloud->header.frame_id.c_str());
      return;
    }
    if (!observation_camera_transform ||
        rclcpp::Time(observation_camera_transform->header.stamp) != cloud_stamp ||
        observation_camera_transform->header.frame_id != "base_link") {
      response->status = "MANUAL_X_SYNCHRONIZED_TRANSFORM_UNAVAILABLE";
      RCLCPP_ERROR(
        this->get_logger(),
        "[MANUAL X] Missing synchronized FK * calibrated-X camera transform; "
        "TF fallback is disabled.");
      return;
    }
    if (grasp_candidate_backend_ == "hggd" && !object_scene) {
      if (!hggd_adapter_) {
        response->status = "HGGD_UNAVAILABLE: " + hggd_initialization_error_;
        return;
      }
      if (!observation_image) {
        response->status = "HGGD_RGB_OBSERVATION_NOT_IN_CACHE";
        return;
      }
      if (rclcpp::Time(observation_image->header.stamp) != cloud_stamp) {
        response->status = "HGGD_RGB_CLOUD_TIMESTAMP_MISMATCH";
        return;
      }
      if (observation_image->width != cloud->width ||
          observation_image->height != cloud->height) {
        response->status = "HGGD_RGB_CLOUD_DIMENSION_MISMATCH";
        return;
      }
      if (!camera_info || camera_info->k[0] <= 0.0 || camera_info->k[4] <= 0.0) {
        response->status = "HGGD_CAMERA_INFO_UNAVAILABLE";
        return;
      }
      if (camera_info->width != cloud->width || camera_info->height != cloud->height) {
        response->status = "HGGD_CAMERA_INFO_DIMENSION_MISMATCH";
        return;
      }
    }
    const bool request_uses_hggd = grasp_candidate_backend_ == "hggd" && !object_scene;
    const Eigen::Vector3f camera_origin_in_cloud(
      static_cast<float>(observation_camera_transform->transform.translation.x),
      static_cast<float>(observation_camera_transform->transform.translation.y),
      static_cast<float>(observation_camera_transform->transform.translation.z));
    int u = std::clamp(request->seed_u, 0, static_cast<int>(cloud->width) - 1);
    int v = std::clamp(request->seed_v, 0, static_cast<int>(cloud->height) - 1);
    Point seed_camera{};
    double correction = 0.0;
    // HGGD provides the actual grasp seeds. Its request pixel is only the ROI
    // centre retained for API compatibility, so foreground selection must not
    // require valid depth at that pixel (it may lie in a handle hole).
    if (!request_uses_hggd && !object_scene &&
        !findCorrectedSeed(*cloud, *request, u, v, seed_camera, correction)) {
      response->status = "NO_VALID_DEPTH_NEAR_POINT";
      return;
    }
    size_t connected_target_points = 0;
    sensor_msgs::msg::Image target_mask = object_scene ? request->target_mask :
      buildConnectedTargetMask(*cloud, request->target_mask, camera_origin_in_cloud,
        u, v, connected_target_points);
    std::string surface_description = request->target_description;
    std::transform(surface_description.begin(), surface_description.end(), surface_description.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!object_scene && (surface_description.find("plate") != std::string::npos ||
      surface_description.find("dish") != std::string::npos)) {
      const vamp_node::OrganizedXYZReader reader(*cloud);
      std::vector<Eigen::Vector3f> points;
      std::vector<size_t> pixels;
      for (int v = 0; v < static_cast<int>(cloud->height); ++v) {
        for (int u = 0; u < static_cast<int>(cloud->width); ++u) {
          std::array<float, 3> point;
          if (maskAllows(target_mask, u, v) && reader.read(u, v, point)) {
            points.emplace_back(point[0], point[1], point[2]); pixels.push_back(static_cast<size_t>(v)*target_mask.step+u);
          }
        }
      }
      const auto keep = vamp_node::planar_body_mask(points);
      for (size_t i = 0; i < keep.size(); ++i) if (!keep[i]) target_mask.data[pixels[i]] = 0;
      connected_target_points = static_cast<size_t>(std::count(keep.begin(), keep.end(), true));
      if (!maskAllows(target_mask, u, v) && connected_target_points > 0) {
        int best_distance = std::numeric_limits<int>::max();
        const int old_u = u, old_v = v;
        for (size_t i = 0; i < keep.size(); ++i) {
          if (!keep[i]) continue;
          const int row = static_cast<int>(pixels[i]/target_mask.step);
          const int column = static_cast<int>(pixels[i]%target_mask.step);
          const int distance = (row-old_v)*(row-old_v)+(column-old_u)*(column-old_u);
          if (distance < best_distance) { best_distance = distance; u = column; v = row; }
        }
      }
      RCLCPP_INFO(this->get_logger(), "[PLANAR BODY MASK] retained=%zu removed_stand=%zu",
        connected_target_points, points.size()-connected_target_points);
    }
    if (object_scene) connected_target_points=static_cast<size_t>(std::count(
      target_mask.data.begin(), target_mask.data.end(), uint8_t{255}));
    if (connected_target_points == 0) {
      response->status = "NO_VALID_DEPTH_ON_CONNECTED_TARGET";
      RCLCPP_WARN(
        this->get_logger(),
        "[GRASP POC] Connected target extraction returned no valid depth; "
        "returning the failed request immediately.");
      return;
    }
    if (!pointAtPixel(*cloud, u, v, seed_camera)) {
      response->status = "NO_VALID_DEPTH_ON_CONNECTED_TARGET";
      return;
    }
    correction = std::hypot(u - request->seed_u, v - request->seed_v);
    response->validated_target_mask = target_mask;
    publishObjectWorld(*cloud, target_mask);
    segmented_target_mask_pub_->publish(target_mask);
    if (observation_image &&
        observation_image->width == target_mask.width &&
        observation_image->height == target_mask.height) {
      publishSegmentedTargetImage(*observation_image, target_mask);
    }
    if (connected_target_points < 80) {
      std::ostringstream status;
      status << "INSUFFICIENT_CONNECTED_TARGET points=" << connected_target_points;
      response->status = status.str();
      return;
    }
    geometry_msgs::msg::PointStamped camera_point, base_point;
    camera_point.header = cloud->header;
    camera_point.point.x = seed_camera[0]; camera_point.point.y = seed_camera[1]; camera_point.point.z = seed_camera[2];
    base_point = camera_point;
    response->corrected_seed_camera = camera_point.point;
    response->corrected_seed_base = base_point.point;
    response->correction_distance_px = static_cast<float>(correction);
    RCLCPP_DEBUG(this->get_logger(),
                "[GRASP] seed pixel=(%d,%d), correction=%.1fpx, connected-target=%zu, base=[%.3f %.3f %.3f]",
                u, v, correction, connected_target_points,
                base_point.point.x, base_point.point.y, base_point.point.z);


    Eigen::Vector3f seed(base_point.point.x, base_point.point.y, base_point.point.z);
    Eigen::Vector3f camera_origin = camera_origin_in_cloud;

    const vamp_node::OrganizedXYZReader scene_reader(*cloud);
    std::vector<Eigen::Vector3f> scene_points;
    std::vector<uint8_t> scene_target_flags;
    std::vector<Eigen::Vector3f> target_samples;
    scene_points.reserve(static_cast<size_t>(cloud->width) * cloud->height / 3);
    scene_target_flags.reserve(
      static_cast<size_t>(cloud->width) * cloud->height / 3);
    target_samples.reserve(static_cast<size_t>(cloud->width) * cloud->height / 8);
    const float scene_radius_sq = static_cast<float>(grasp_scene_crop_radius_m_ * grasp_scene_crop_radius_m_);
    const float sample_radius_sq = static_cast<float>(
      grasp_target_sample_radius_m_ * grasp_target_sample_radius_m_);
    for (int pixel_v = 0; pixel_v < static_cast<int>(cloud->height); ++pixel_v) {
      for (int pixel_u = 0; pixel_u < static_cast<int>(cloud->width); ++pixel_u) {
        Point point;
        if (!scene_reader.read(pixel_u, pixel_v, point)) continue;
        const Eigen::Vector3f base(point[0], point[1], point[2]);
        const bool target = maskAllows(target_mask, pixel_u, pixel_v);
        const float seed_distance_sq = (base - seed).squaredNorm();
        if (object_scene || frozen_image_layout || seed_distance_sq <= scene_radius_sq || target) {
          scene_points.push_back(base);
          scene_target_flags.push_back(target ? 1U : 0U);
        }
        if (target && (request_uses_hggd ||
            grasp_target_sample_radius_m_ <= 0.0 ||
            seed_distance_sq <= sample_radius_sq)) {
          target_samples.push_back(base);
        }
      }
    }
    const bool frozen_collision_override = request->collision_scene.width > 0;
    if (frozen_collision_override) {
      const auto &override_cloud = request->collision_scene;
      const size_t original_count = static_cast<size_t>(cloud->width) * cloud->height;
      const size_t override_count = static_cast<size_t>(override_cloud.width) * override_cloud.height;
      const auto xyz_field_valid = [&override_cloud](const std::string &name) {
        return std::any_of(override_cloud.fields.begin(), override_cloud.fields.end(),
          [&](const auto &field) { return field.name == name &&
            field.datatype == sensor_msgs::msg::PointField::FLOAT32 && field.count == 1 &&
            field.offset + sizeof(float) <= override_cloud.point_step; });
      };
      if (override_cloud.height != 1 || override_cloud.is_bigendian ||
          override_cloud.point_step < 12 ||
          override_cloud.row_step != static_cast<size_t>(override_cloud.width) * override_cloud.point_step ||
          override_cloud.data.size() != override_cloud.row_step ||
          !xyz_field_valid("x") || !xyz_field_valid("y") || !xyz_field_valid("z")) {
        response->status = "GRASP_COLLISION_SCENE_MALFORMED_LAYOUT";
        return;
      }
      if (override_cloud.header.frame_id != "base_link" ||
          override_cloud.header.stamp != cloud->header.stamp ||
          override_count < original_count || override_count > 1000000 ||
          override_cloud.data.size() > 16 * 1024 * 1024) {
        response->status = "GRASP_COLLISION_SCENE_OBSERVATION_MISMATCH";
        return;
      }
      scene_points.clear(); scene_target_flags.clear();
      try {
        sensor_msgs::PointCloud2ConstIterator<float> x(override_cloud,"x"),
          y(override_cloud,"y"), z(override_cloud,"z");
        size_t index = 0;
        for (; x != x.end(); ++x, ++y, ++z, ++index) {
          const Eigen::Vector3f point(*x,*y,*z);
          bool target = false;
          if (index < original_count) {
            const int u = static_cast<int>(index % cloud->width);
            const int v = static_cast<int>(index / cloud->width);
            Point original;
            const bool valid_original = scene_reader.read(u,v,original);
            target = maskAllows(target_mask,u,v) && valid_original;
            if (point.allFinite() && (!valid_original ||
                (point - Eigen::Vector3f(original[0],original[1],original[2])).norm() > 0.000001F)) {
              response->status = "GRASP_COLLISION_SCENE_PIXEL_PREFIX_CHANGED";
              return;
            }
            if (target && !point.allFinite()) {
              response->status = "GRASP_COLLISION_SCENE_ACTIVE_TARGET_REMOVED";
              return;
            }
          }
          if (!point.allFinite()) continue;
          scene_points.push_back(point);
          scene_target_flags.push_back(target ? 1U : 0U);
        }
      } catch (const std::exception &error) {
        response->status = std::string("GRASP_COLLISION_SCENE_MALFORMED: ") + error.what();
        return;
      }
    }
    if (scene_points.size() < 200 || target_samples.size() < 80) {
      std::ostringstream status;
      status << "INSUFFICIENT_GRASP_INPUT scene=" << scene_points.size()
             << " target=" << target_samples.size();
      response->status = status.str();
      return;
    }
    Eigen::Vector3f alignment_object_center = Eigen::Vector3f::Zero();
    for (const auto &point : target_samples) alignment_object_center += point;
    alignment_object_center /= static_cast<float>(target_samples.size());
    // Retain the grounded affordance in base_link for the post-motion local
    // lookup. The object is static between FIND_GRASP and PRE_GRASP, while the
    // wrist camera moves. Restricting the new observation to this 3-D support
    // prevents a handle seed from growing through its junction into the cup
    // body or table.
    {
      std::lock_guard<std::mutex> lock(refinement_target_mutex_);
      refinement_target_support_voxels_.clear();
      refinement_contact_region_.clear();
      refinement_target_support_voxels_.reserve(target_samples.size());
      for (const auto & point : target_samples) {
        refinement_target_support_voxels_.insert(std::make_tuple(
          static_cast<int>(std::floor(
            point.x() / refinement_target_support_voxel_m_)),
          static_cast<int>(std::floor(
            point.y() / refinement_target_support_voxel_m_)),
          static_cast<int>(std::floor(
            point.z() / refinement_target_support_voxel_m_))));
      }
      refinement_target_stamp_ns_ = cloud_stamp.nanoseconds();
    }

    std::string affordance = request->affordance_type;
    std::transform(affordance.begin(), affordance.end(), affordance.begin(),
      [](unsigned char value) { return static_cast<char>(std::toupper(value)); });
    if (affordance != "HANDLE" && affordance != "RIM" &&
        affordance != "BODY" && affordance != "AUTO") {
      affordance = "AUTO";
    }
    std::string normalized_description = request->target_description;
    std::transform(
      normalized_description.begin(), normalized_description.end(),
      normalized_description.begin(),
      [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    const bool known_cloud = static_cast<bool>(object_scene);
    const bool pure_hggd = !known_cloud;
    const bool pure_learned = pure_hggd;
    size_t local_surface_count = 0;
    std::vector<AntipodalContacts> local_antipodal_pairs;
    AntipodalContacts silhouette_pair;
    bool silhouette_pair_available = false;
    std::vector<AntipodalContacts> body_model_pairs;
    if (require_measured_antipodal_contacts_ && !pure_learned) {
      local_antipodal_pairs = buildLocalAntipodalPairs(
        target_samples, seed, local_surface_count);
      if (!object_scene && !silhouette_pair_available && buildSilhouetteThicknessPair(
            *cloud, target_mask, u, v, silhouette_pair)) {
        silhouette_pair_available = true;
      }
      if (silhouette_pair_available) {
        local_antipodal_pairs.push_back(silhouette_pair);
        std::stable_sort(
          local_antipodal_pairs.begin(), local_antipodal_pairs.end(),
          [](const auto &left, const auto &right) {
            return left.quality < right.quality;
          });
      }
      // A cup body has a locally round, filled mask neighborhood.  Unlike a
      // handle branch it has no unique 2-D minor axis, so use the visible
      // depth arc to fit its cylindrical cross-section.  This is restricted
      // to body prompts; handle prompts retain the stricter measured-edge
      // requirement above.
      std::string target_description = request->target_description;
      std::transform(
        target_description.begin(), target_description.end(),
        target_description.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
      if (!silhouette_pair_available &&
          target_description.find("handle") == std::string::npos) {
        // Preserve the handle and test body bands above and below it.  The
        // frame is base_link, so +Z is the table-normal for an upright cup.
        // The centre band is deliberately last because it is commonly where
        // the handle occupies the finger-sweep volume.
        constexpr std::array<float, 5> body_band_offsets_m = {
          -0.030F, 0.030F, -0.045F, 0.045F, 0.0F};
        constexpr std::array<float, 7> body_closing_yaws_degrees = {
          0.0F, 30.0F, -30.0F, 60.0F, -60.0F, 90.0F, -90.0F};
        for (const float offset : body_band_offsets_m) {
          for (const float yaw_degrees : body_closing_yaws_degrees) {
            AntipodalContacts candidate_pair;
            if (!buildCupBodyThicknessPair(
                  target_samples,
                  seed + Eigen::Vector3f::UnitZ() * offset,
                  camera_origin,
                  yaw_degrees * static_cast<float>(M_PI / 180.0),
                  candidate_pair)) {
              continue;
            }
            body_model_pairs.push_back(candidate_pair);
            local_antipodal_pairs.push_back(candidate_pair);
          }
        }
      }
    }

    std::vector<GraspProposal> proposals;
    KnownCloudGraspDiagnostics cloud_diagnostics;
    HggdMcuDiagnostics hggd_diagnostics;
    int hggd_aligned_centers = 0;
    int hggd_geometry_rejected = 0;
    int hggd_direct_seeds_tested = 0;
    int hggd_prefilter_ik_rejected = 0;
    int hggd_prefilter_collision_rejected = 0;
    double hggd_seed_selection_ms = 0.0;
    std::string hggd_heatmap_output;
    const auto proposal_started = std::chrono::steady_clock::now();
    if (!fixture_seeds.empty()) {
      proposals = fixture_seeds;
    } else if (known_cloud) {
      proposals = cloud_generator_->generate(
        target_samples, seed, camera_origin, &cloud_diagnostics);
    }

    if (pure_hggd) {
      try {
        const auto &manual_transform = observation_camera_transform->transform;
        const Eigen::Quaternionf calibrated_rotation(
          static_cast<float>(manual_transform.rotation.w),
          static_cast<float>(manual_transform.rotation.x),
          static_cast<float>(manual_transform.rotation.y),
          static_cast<float>(manual_transform.rotation.z));
        if (!std::isfinite(calibrated_rotation.norm()) ||
            calibrated_rotation.norm() < 1.0e-6F) {
          throw std::runtime_error("invalid synchronized calibrated-X rotation");
        }
        Eigen::Isometry3f camera_to_base_eigen = Eigen::Isometry3f::Identity();
        camera_to_base_eigen.linear() =
          calibrated_rotation.normalized().toRotationMatrix();
        camera_to_base_eigen.translation() = Eigen::Vector3f(
          static_cast<float>(manual_transform.translation.x),
          static_cast<float>(manual_transform.translation.y),
          static_cast<float>(manual_transform.translation.z));
        // observation_cloud is already base_link because Active Vision used
        // this exact FK * calibrated-X matrix when publishing it.
        const Eigen::Isometry3f cloud_to_base_eigen =
          Eigen::Isometry3f::Identity();
        const Eigen::Isometry3f cloud_to_camera_eigen =
          camera_to_base_eigen.inverse() * cloud_to_base_eigen;
        HggdCameraIntrinsics intrinsics;
        intrinsics.fx = static_cast<float>(camera_info->k[0]);
        intrinsics.fy = static_cast<float>(camera_info->k[4]);
        intrinsics.cx = static_cast<float>(camera_info->k[2]);
        intrinsics.cy = static_cast<float>(camera_info->k[5]);
        cv::Mat heatmap;
        {
        ComponentTimer hggd_timer(PipelineComponent::HGGD);
        proposals = hggd_adapter_->detect(
          *cloud, *observation_image, target_mask, intrinsics,
          cloud_to_camera_eigen, camera_to_base_eigen,
          static_cast<float>(grasp_contact_depth_m_),
          static_cast<std::size_t>(hggd_maximum_proposals_), heatmap,
          hggd_diagnostics);
        }

        // Prepare one raw marker per spatial family. Publication is delayed
        // until after the quick seed IK/collision prefilter so this RViz layer
        // shows exactly the HGGD priors that were actually offered to local
        // geometry refinement.
        std::unordered_map<std::size_t, my_actions::msg::GraspCandidate>
          raw_hggd_marker_by_family;
        const std::size_t raw_marker_limit = std::min(
          proposals.size(), static_cast<std::size_t>(
            std::max(0, grasp_visualization_limit_)));
        std::unordered_set<std::size_t> displayed_raw_families;
        for (std::size_t index = 0; index < raw_marker_limit; ++index) {
          const auto &proposal = proposals[index];
          // LocalNet emits several orientation hypotheses for every AnchorNet
          // centre. A seed is the spatial centre, so show only its highest
          // ranked raw orientation instead of dozens of overlapping jaws.
          if (!displayed_raw_families.insert(proposal.family_index).second) {
            continue;
          }
          if (!proposal.hand_base.allFinite() || !proposal.approach.allFinite() ||
            !proposal.closing.allFinite() ||
            proposal.approach.squaredNorm() < 1.0e-8F ||
            proposal.closing.squaredNorm() < 1.0e-8F)
          {
            continue;
          }
          const Eigen::Vector3f approach = proposal.approach.normalized();
          const Eigen::Vector3f closing = proposal.closing.normalized();
          const Eigen::Vector3f tcp = proposal.hand_base +
            approach * static_cast<float>(grasp_tcp_offset_m_);
          const Eigen::Quaternionf quaternion(
            piperGripperRotation(closing, approach));
          my_actions::msg::GraspCandidate marker;
          marker.id = static_cast<uint32_t>(raw_hggd_marker_by_family.size() + 1);
          marker.grasp_pose.header = cloud->header;
          marker.grasp_pose.header.frame_id = "base_link";
          marker.grasp_pose.pose.position.x = tcp.x();
          marker.grasp_pose.pose.position.y = tcp.y();
          marker.grasp_pose.pose.position.z = tcp.z();
          marker.grasp_pose.pose.orientation.x = quaternion.x();
          marker.grasp_pose.pose.orientation.y = quaternion.y();
          marker.grasp_pose.pose.orientation.z = quaternion.z();
          marker.grasp_pose.pose.orientation.w = quaternion.w();
          marker.pregrasp_pose = marker.grasp_pose;
          marker.pregrasp_pose.pose.position.x -=
            approach.x() * pregrasp_offset_;
          marker.pregrasp_pose.pose.position.y -=
            approach.y() * pregrasp_offset_;
          marker.pregrasp_pose.pose.position.z -=
            approach.z() * pregrasp_offset_;
          const Eigen::Vector3f left = proposal.has_contacts ?
            proposal.contact_left :
            proposal.sample + closing * (proposal.opening_m * 0.5F);
          const Eigen::Vector3f right = proposal.has_contacts ?
            proposal.contact_right :
            proposal.sample - closing * (proposal.opening_m * 0.5F);
          marker.contact_left.x = left.x();
          marker.contact_left.y = left.y();
          marker.contact_left.z = left.z();
          marker.contact_right.x = right.x();
          marker.contact_right.y = right.y();
          marker.contact_right.z = right.z();
          marker.jaw_opening_m = proposal.opening_m;
          marker.score = proposal.quality;
          raw_hggd_marker_by_family.emplace(
            proposal.family_index, std::move(marker));
        }

        // Heatmap is an inference diagnostic, not a validation result. Save
        // and publish it immediately so PCA/IK failure cannot suppress it.
        if (!heatmap.empty()) {
          sensor_msgs::msg::Image message;
          message.header = observation_image->header;
          message.height = static_cast<uint32_t>(heatmap.rows);
          message.width = static_cast<uint32_t>(heatmap.cols);
          message.encoding = "bgr8";
          message.is_bigendian = false;
          message.step = static_cast<uint32_t>(heatmap.cols * 3);
          message.data.assign(heatmap.datastart, heatmap.dataend);
          hggd_heatmap_pub_->publish(message);
          hggd_heatmap_output = "/vamp_planner/hggd_heatmap";
          const char *home = std::getenv("HOME");
          if (home != nullptr && *home != '\0') {
            const auto capture_dir = std::filesystem::path(home) /
              "robobrain_inference";
            try {
              std::filesystem::create_directories(capture_dir);
              const auto path = capture_dir / "hggd_heatmap.png";
              if (!cv::imwrite(path.string(), heatmap)) {
                RCLCPP_WARN(this->get_logger(),
                  "Could not save HGGD heatmap: %s", path.c_str());
              }
            } catch (const std::exception &error) {
              RCLCPP_WARN(this->get_logger(),
                "Could not save HGGD heatmap: %s", error.what());
            }
          }
        }
        float raw_minimum_opening = std::numeric_limits<float>::infinity();
        float raw_maximum_opening = 0.0F;
        for (const auto &proposal : proposals) {
          if (!std::isfinite(proposal.opening_m)) continue;
          raw_minimum_opening = std::min(
            raw_minimum_opening, proposal.opening_m);
          raw_maximum_opening = std::max(
            raw_maximum_opening, proposal.opening_m);
        }
        RCLCPP_INFO(
          this->get_logger(),
          "[HGGD RAW] decoded=%zu spatial-seeds=%zu learned-width=[%.1f,%.1f]mm heatmap=%s",
          proposals.size(), raw_hggd_marker_by_family.size(),
          std::isfinite(raw_minimum_opening) ?
          raw_minimum_opening * 1000.0F : 0.0F,
          raw_maximum_opening * 1000.0F,
          hggd_heatmap_output.empty() ? "unavailable" :
          hggd_heatmap_output.c_str());
        std::stable_sort(
          proposals.begin(), proposals.end(),
          [](const GraspProposal &left, const GraspProposal &right) {
            return left.quality > right.quality;
          });
        // Quickly reject spatial centres that the current Piper branch cannot
        // reach or that place the robot in the non-target obstacle world.
        // The target itself is excluded only from this coarse seed test: the
        // unchanged downstream validator checks the reconstructed fingers,
        // contacts, target, and full environment strictly.
        std::vector<Point> seed_prefilter_environment;
        if (enable_grasp_collision_check_) {
          std::vector<Point> complete_environment;
          {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            complete_environment = getFilteredEnvironment(joints);
          }
          const float target_clearance_sq = static_cast<float>(
            target_collision_clearance_m_ * target_collision_clearance_m_);
          seed_prefilter_environment.reserve(complete_environment.size());
          for (const auto &obstacle : complete_environment) {
            const Eigen::Vector3f obstacle_point(
              obstacle[0], obstacle[1], obstacle[2]);
            const bool belongs_to_target = std::any_of(
              target_samples.begin(), target_samples.end(),
              [&obstacle_point, target_clearance_sq](
                  const Eigen::Vector3f &target) {
                return (obstacle_point - target).squaredNorm() <=
                  target_clearance_sq;
              });
            if (!belongs_to_target) seed_prefilter_environment.push_back(obstacle);
          }
        }
        std::array<float, 6> current_joint_array{};
        std::copy_n(joints.begin(), 6, current_joint_array.begin());
        const Eigen::Matrix3f current_gripper_rotation =
          Robot::eefk(current_joint_array).linear();

        // LocalNet's width is a learned prior and may exceed the physical
        // 100 mm jaw limit. Do not apply the hardware-width gate until local
        // unvoxelized geometry has replaced it with a measured opening.
        const auto seed_selection_started = std::chrono::steady_clock::now();
        constexpr int direct_seed_limit = 32;
        std::vector<GraspProposal> aligned_seeds;
        aligned_seeds.reserve(static_cast<std::size_t>(direct_seed_limit));
        std::vector<my_actions::msg::GraspCandidate>
          prefiltered_raw_hggd_markers;
        prefiltered_raw_hggd_markers.reserve(
          static_cast<std::size_t>(direct_seed_limit));
        std::unordered_set<std::size_t> prepared_spatial_families;
        for (const auto &direct : proposals) {
          if (hggd_direct_seeds_tested >= direct_seed_limit) break;
          // LocalNet returns several orientation hypotheses for one heatmap
          // centre. Geometry refinement searches orientation itself, so
          // process each spatial seed only once and move on to a genuinely
          // different point when it has no safe variant.
          if (!prepared_spatial_families.insert(direct.family_index).second) {
            continue;
          }
          if (!direct.hand_base.allFinite() || !direct.approach.allFinite() ||
            !direct.closing.allFinite() || !std::isfinite(direct.opening_m) ||
            direct.approach.squaredNorm() < 1.0e-8F ||
            direct.closing.squaredNorm() < 1.0e-8F)
          {
            continue;
          }
          ++hggd_direct_seeds_tested;

          Eigen::Vector3f seed_approach = direct.sample - camera_origin;
          if (seed_approach.squaredNorm() < 1.0e-8F) {
            ++hggd_prefilter_ik_rejected;
            continue;
          }
          seed_approach.normalize();
          Eigen::Vector3f seed_closing = current_gripper_rotation.col(1);
          seed_closing -= seed_approach * seed_closing.dot(seed_approach);
          if (seed_closing.squaredNorm() < 1.0e-8F) {
            seed_closing = direct.closing -
              seed_approach * direct.closing.dot(seed_approach);
          }
          if (seed_closing.squaredNorm() < 1.0e-8F) {
            ++hggd_prefilter_ik_rejected;
            continue;
          }
          seed_closing.normalize();
          Eigen::Isometry3f seed_pose = Eigen::Isometry3f::Identity();
          seed_pose.linear() = piperGripperRotation(
            seed_closing, seed_approach);
          seed_pose.translation() = direct.sample -
            seed_approach * static_cast<float>(grasp_contact_depth_m_) +
            seed_approach * static_cast<float>(grasp_tcp_offset_m_);
          std::vector<float> seed_ik;
          PoseIkDiagnostics seed_ik_diagnostics;
          if (!solvePiperPoseIK(
              seed_pose, joints, seed_ik, seed_ik_diagnostics, true,
              0.012F, 0.35F, 300)) {
            ++hggd_prefilter_ik_rejected;
            continue;
          }
          if (enable_grasp_collision_check_ &&
              !isCandidatePoseSafe(seed_ik, seed_prefilter_environment)) {
            ++hggd_prefilter_collision_rejected;
            continue;
          }

          const auto raw_marker = raw_hggd_marker_by_family.find(
            direct.family_index);
          if (raw_marker != raw_hggd_marker_by_family.end()) {
            auto marker = raw_marker->second;
            marker.id = static_cast<uint32_t>(
              prefiltered_raw_hggd_markers.size() + 1);
            prefiltered_raw_hggd_markers.push_back(std::move(marker));
          }

          // Recover the image-space material pixel belonging to this 3-D
          // HGGD seed. Searching the synchronized organized cloud avoids a
          // second transform/projection convention and uses the same manual
          // FK * calibrated-X observation as the rest of grasp computation.
          int thin_seed_u = -1;
          int thin_seed_v = -1;
          float nearest_target_distance_sq =
            std::numeric_limits<float>::infinity();
          for (int pixel_v = 0;
               pixel_v < static_cast<int>(cloud->height); ++pixel_v) {
            for (int pixel_u = 0;
                 pixel_u < static_cast<int>(cloud->width); ++pixel_u) {
              if (!maskAllows(target_mask, pixel_u, pixel_v)) continue;
              Point target_point;
              if (!pointAtPixel(*cloud, pixel_u, pixel_v, target_point)) {
                continue;
              }
              const Eigen::Vector3f point(
                target_point[0], target_point[1], target_point[2]);
              const float distance_sq =
                (point - direct.sample).squaredNorm();
              if (distance_sq >= nearest_target_distance_sq) continue;
              nearest_target_distance_sq = distance_sq;
              thin_seed_u = pixel_u;
              thin_seed_v = pixel_v;
            }
          }
          AntipodalContacts thin_silhouette_pair;
          const float silhouette_snap_limit_sq = static_cast<float>(
            hggd_seed_snap_max_distance_m_ *
            hggd_seed_snap_max_distance_m_);
          const bool have_thin_silhouette =
            thin_seed_u >= 0 && thin_seed_v >= 0 &&
            nearest_target_distance_sq <= silhouette_snap_limit_sq &&
            buildSilhouetteThicknessPair(
              *cloud, target_mask, thin_seed_u, thin_seed_v,
              thin_silhouette_pair);

          GraspProposal aligned = direct;
          std::vector<GraspProposal> aligned_hypotheses;
          if (!alignHggdSeedToLocalDominantAxis(
              target_samples, camera_origin, current_gripper_rotation,
              have_thin_silhouette ? &thin_silhouette_pair : nullptr,
              affordance,
              aligned, &aligned_hypotheses))
          {
            ++hggd_geometry_rejected;
            continue;
          }
          aligned_seeds.insert(
            aligned_seeds.end(),
            std::make_move_iterator(aligned_hypotheses.begin()),
            std::make_move_iterator(aligned_hypotheses.end()));
          ++hggd_aligned_centers;
        }
        publishGraspMarkers(
          prefiltered_raw_hggd_markers,
          hggd_raw_proposals_markers_pub_, "hggd_raw_proposal_");
        hggd_seed_selection_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - seed_selection_started).count();
        proposals = std::move(aligned_seeds);
        if (proposals.empty()) {
          throw std::runtime_error(
            "no ranked HGGD spatial seed passed local PCA alignment");
        }
        const auto aligned_opening_range = std::minmax_element(
          proposals.begin(), proposals.end(),
          [](const GraspProposal &left, const GraspProposal &right) {
            return left.opening_m < right.opening_m;
          });
        RCLCPP_INFO(
          this->get_logger(),
          "[HGGD ALIGNED] spatial-tested=%d prefilter={ik_rejected:%d,collision_rejected:%d} "
          "zero-poses=%zu measured-width=[%.1f,%.1f]mm contacts=measured",
          hggd_direct_seeds_tested, hggd_prefilter_ik_rejected,
          hggd_prefilter_collision_rejected,
          proposals.size(),
          aligned_opening_range.first->opening_m * 1000.0F,
          aligned_opening_range.second->opening_m * 1000.0F);
        // This is updated to the first seed with accepted variants below.
        response->corrected_seed_base.x = proposals.front().sample.x();
        response->corrected_seed_base.y = proposals.front().sample.y();
        response->corrected_seed_base.z = proposals.front().sample.z();
      } catch (const std::exception &error) {
        response->status = std::string("HGGD_DETECTION_FAILED: ") + error.what();
        RCLCPP_ERROR(this->get_logger(), "%s", response->status.c_str());
        return;
      }
    }
    if (pure_hggd) {
      std::stable_sort(proposals.begin(), proposals.end(),
        [](const GraspProposal &left, const GraspProposal &right) {
          return left.quality > right.quality;
        });
    }
    const auto proposal_finished = std::chrono::steady_clock::now();
    RCLCPP_INFO(this->get_logger(),
      "[KNOWN CLOUD GRASP] backend=%s isolated=%zu downsampled=%zu normals=%zu "
      "pairs=%zu local_pairs=%zu global_pairs=%zu families=%zu local_families=%zu "
      "global_families=%zu generated=%zu deduplicated=%zu total_proposals=%zu generation=%.1fms",
      grasp_candidate_backend_.c_str(), cloud_diagnostics.isolated_points,
      cloud_diagnostics.downsampled_points, cloud_diagnostics.normal_points,
      cloud_diagnostics.contact_pairs, cloud_diagnostics.local_contact_pairs,
      cloud_diagnostics.global_contact_pairs, cloud_diagnostics.selected_contact_families,
      cloud_diagnostics.selected_local_families,
      cloud_diagnostics.selected_global_families,
      cloud_diagnostics.generated,
      cloud_diagnostics.deduplicated, proposals.size(),
      std::chrono::duration<double, std::milli>(proposal_finished - proposal_started).count());



    std::vector<my_actions::msg::GraspCandidate> cloud_markers;
    std::vector<my_actions::msg::GraspCandidate> hggd_markers;
    const size_t visualization_limit =
      static_cast<size_t>(std::max(0, grasp_visualization_limit_));
    for (const auto &proposal : proposals) {
      GraspProposal visual_proposal = proposal;
      // Each HGGD marker is one geometry-aligned topology hypothesis.  The
      // subsequent bounded IK orbit is still deliberately not published on
      // this layer, so RViz shows useful side/rim/top alternatives without
      // showing every small solver perturbation.
      auto *marker_group = &cloud_markers;
      if (proposal.source == GraspProposalSource::kKnownCloud) {
        marker_group = &cloud_markers;
      } else if (proposal.source == GraspProposalSource::kHggd) {
        marker_group = &hggd_markers;
      }
      if (marker_group->size() >= visualization_limit) continue;
      const Eigen::Vector3f tcp = visual_proposal.hand_base +
        visual_proposal.approach * static_cast<float>(grasp_tcp_offset_m_);
      const float proposal_contact_depth = std::clamp(
        (visual_proposal.sample - visual_proposal.hand_base).dot(
          visual_proposal.approach),
        0.0F, static_cast<float>(finger_length_m_));
      const Eigen::Vector3f contact_center = visual_proposal.hand_base +
        visual_proposal.approach * proposal_contact_depth;
      const Eigen::Matrix3f rotation = piperGripperRotation(
        visual_proposal.closing, visual_proposal.approach);
      Eigen::Quaternionf quaternion(rotation);
      my_actions::msg::GraspCandidate marker;
      marker.id = static_cast<uint32_t>(marker_group->size() + 1);
      marker.grasp_pose.header.stamp = cloud->header.stamp;
      marker.grasp_pose.header.frame_id = "base_link";
      marker.grasp_pose.pose.position.x = tcp.x();
      marker.grasp_pose.pose.position.y = tcp.y();
      marker.grasp_pose.pose.position.z = tcp.z();
      marker.grasp_pose.pose.orientation.x = quaternion.x();
      marker.grasp_pose.pose.orientation.y = quaternion.y();
      marker.grasp_pose.pose.orientation.z = quaternion.z();
      marker.grasp_pose.pose.orientation.w = quaternion.w();
      marker.pregrasp_pose = marker.grasp_pose;
      marker.pregrasp_pose.pose.position.x -= visual_proposal.approach.x() * pregrasp_offset_;
      marker.pregrasp_pose.pose.position.y -= visual_proposal.approach.y() * pregrasp_offset_;
      marker.pregrasp_pose.pose.position.z -= visual_proposal.approach.z() * pregrasp_offset_;
      const Eigen::Vector3f left = visual_proposal.has_contacts ? visual_proposal.contact_left :
        contact_center + visual_proposal.closing * (visual_proposal.opening_m * 0.5F);
      const Eigen::Vector3f right = visual_proposal.has_contacts ? visual_proposal.contact_right :
        contact_center - visual_proposal.closing * (visual_proposal.opening_m * 0.5F);
      marker.contact_left.x = left.x(); marker.contact_left.y = left.y(); marker.contact_left.z = left.z();
      marker.contact_right.x = right.x(); marker.contact_right.y = right.y(); marker.contact_right.z = right.z();
      marker.jaw_opening_m = visual_proposal.opening_m;
      marker.score = visual_proposal.quality;
      marker_group->push_back(marker);
    }
    publishGraspMarkers(cloud_markers, known_cloud_proposals_markers_pub_, "known_cloud_proposal_");
    publishGraspMarkers(hggd_markers, hggd_proposals_markers_pub_, "hggd_proposal_");

    std::vector<Point> env;
    std::vector<uint8_t> env_target_flags;
    if (enable_grasp_collision_check_) {
      if (object_scene || frozen_image_layout || frozen_collision_override) {
        for (const auto &point : scene_points) env.push_back({point.x(),point.y(),point.z()});
      } else {
        std::lock_guard<std::mutex> lock(cloud_mutex_); env = getFilteredEnvironment(joints);
      }
      // Preserve the complete low-resolution target in the collision world.
      // Candidate-specific validation below permits only inner-pad contact
      // near the two reconstructed jaw contacts. Arm/wrist contact with the
      // object, and gripper contact elsewhere on it, remain blocking.
      const float target_clearance_sq = static_cast<float>(
        target_collision_clearance_m_ * target_collision_clearance_m_);
      env_target_flags.assign(env.size(), 0);
      if (object_scene || frozen_image_layout || frozen_collision_override) env_target_flags=scene_target_flags;
      for (size_t obstacle_index = 0; !object_scene && !frozen_image_layout && !frozen_collision_override && obstacle_index < env.size();
           ++obstacle_index) {
        const Point &obstacle = env[obstacle_index];
        const Eigen::Vector3f obstacle_point(
          obstacle[0], obstacle[1], obstacle[2]);
        env_target_flags[obstacle_index] = std::any_of(
          target_samples.begin(), target_samples.end(),
          [&obstacle_point, target_clearance_sq](const Eigen::Vector3f &target) {
            return (obstacle_point - target).squaredNorm() <=
              target_clearance_sq;
          });
      }
      RCLCPP_INFO(
        this->get_logger(),
        "[GRASP COLLISION] environment=%zu target_voxels=%zu "
        "contact_radius=%.3fm policy=jaw_contacts_only",
        env.size(), static_cast<size_t>(std::count(
          env_target_flags.begin(), env_target_flags.end(), uint8_t{1})),
        target_collision_clearance_m_);
    }
    std::vector<RejectedGraspPose> rejected_debug;
    int rejected_score = 0, rejected_width = 0, rejected_antipodal = 0;
    int rejected_contacts = 0, rejected_finger_sweep = 0, rejected_ik = 0;
    std::unordered_map<std::string, int> contact_rejection_reasons;
    int measured_contact_valid = 0, silhouette_contact_selected = 0;
    int body_model_contact_selected = 0;

    bool have_target_sweep_block = false;
    int logged_grasp_collision_detail = 0;
    std::array<float, 3> first_target_sweep_block{};
    float first_target_sweep_opening = 0.0F;
    bool first_target_sweep_silhouette = false;
    int rejected_pregrasp_ik = 0, rejected_pregrasp_visibility = 0;
    int rejected_pregrasp_envelope = 0;
    int rejected_grasp_collision = 0, rejected_pregrasp_collision = 0;
    int retained_pregrasp_overrides = 0;
    size_t ranking_pool_candidate_count = 0;
    const auto record_rejection = [&rejected_debug](
        const Eigen::Isometry3f &pose, const Eigen::Vector3f &closing,
        const Eigen::Vector3f &approach, float opening, const std::string &reason) {
      if (rejected_debug.size() < 10) {
        rejected_debug.push_back({pose, closing, approach, opening, reason});
      }
    };
    std::optional<vamp_node::GraspSupportPlane> grasp_support_plane;
    if (pure_hggd) {
      std::vector<Eigen::Vector3f> support_samples;
      // Measure the table from the immutable observation, not the coarse
      // collision map whose floor/self filters may already remove it.
      for (size_t i=0;i<scene_points.size();++i) {
        if (i<scene_target_flags.size() && scene_target_flags[i]) continue;
        support_samples.push_back(scene_points[i]);
      }
      grasp_support_plane=vamp_node::fitGraspSupportPlane(support_samples,alignment_object_center);
    }
    int tested = 0;
    const bool quick_layout_validation =
      pure_hggd || known_cloud;
    const int candidate_test_limit = quick_layout_validation ?
      std::min(maximum_candidates_to_test_,
        known_cloud ? 120 : 100) : maximum_candidates_to_test_;
    const int ranking_pool_limit = quick_layout_validation ?
      std::min(maximum_ranking_pool_candidates_,
        (pure_hggd || request->target_description.rfind("layout: ", 0) == 0) ? 6 : 3) :
      maximum_ranking_pool_candidates_;
    const auto search_budget_exhausted = [&]() {
      return graspSearchBudgetExhausted(tested, candidate_test_limit,
        pure_hggd ? maximum_candidates_to_test_ : candidate_test_limit,
        static_cast<int>(response->candidates.size()) >= std::min(2, ranking_pool_limit));
    };
    int hggd_duplicate_valid_rejected = 0;
    std::unordered_map<uint32_t, std::string> candidate_origins;
    std::unordered_map<uint32_t, size_t> hggd_candidate_seed_indices;
    std::unordered_map<uint32_t, float> hggd_candidate_variant_costs;
    int geometry_approach_variants_generated = 0;
    int hggd_contact_axis_fallback_seeds = 0;
    int hggd_orbit_variants_generated = 0;
    int pinch_candidates_generated = 0;
    int power_candidates_generated = 0;
    int pinch_candidates_accepted = 0;
    int power_candidates_accepted = 0;
    size_t body_model_pair_cursor = 0;
    struct HggdSeedValidationDiagnostics
    {
      int orientations_considered = 0;
      int degenerate_orientations = 0;
      int cross_section_rejected = 0;
      int sweep_rejected = 0;
      int ik_submitted = 0;
      int grasp_ik_rejected = 0;
      int pregrasp_ik_rejected = 0;
      int pregrasp_visibility_rejected = 0;
      int grasp_collision_rejected = 0;
      int pregrasp_collision_rejected = 0;
      int accepted = 0;
      bool ik_budget_exhausted = false;
      bool ranking_pool_full = false;
      std::string terminal_reason;
      PoseIkDiagnostics best_grasp_ik;
      PoseIkDiagnostics best_pregrasp_ik;
    };
    std::vector<HggdSeedValidationDiagnostics> hggd_seed_diagnostics(proposals.size());
    for (size_t proposal_index = 0; proposal_index < proposals.size(); ++proposal_index) {
      const auto &proposal = proposals[proposal_index];
      auto &seed_diagnostics = hggd_seed_diagnostics[proposal_index];
      // Keep a bounded diverse pool so complete approach/lift/placement
      // planning can reject one grasp without restarting perception.
      // Stop the expensive validation path as soon as that pool is full.
      if (search_budget_exhausted() ||
          static_cast<int>(response->candidates.size()) >=
            ranking_pool_limit) {
        if (pure_hggd) {
          seed_diagnostics.ik_budget_exhausted = search_budget_exhausted();
          seed_diagnostics.ranking_pool_full =
            static_cast<int>(response->candidates.size()) >=
            ranking_pool_limit;
          continue;
        }
        break;
      }
      if (!proposal.hand_base.allFinite() || !proposal.approach.allFinite() ||
          !proposal.closing.allFinite() || !proposal.height_axis.allFinite() ||
          !std::isfinite(proposal.opening_m) || !std::isfinite(proposal.quality) ||
          proposal.approach.squaredNorm() < 1.0e-8F ||
          proposal.closing.squaredNorm() < 1.0e-8F) {
        ++rejected_score;
        seed_diagnostics.terminal_reason = "INVALID_PROPOSAL";
        continue;
      }
      Eigen::Vector3f approach = proposal.approach.normalized();
      Eigen::Vector3f closing = proposal.closing.normalized();
      if (proposal.opening_m < minimum_gripper_width_ ||
          proposal.opening_m > maximum_gripper_width_) {
        ++rejected_width;
        seed_diagnostics.terminal_reason = "WIDTH_FILTER";
        continue;
      }

      AntipodalContacts measured_contacts;
      Eigen::Vector3f hand_base = proposal.hand_base;
      Eigen::Vector3f left;
      Eigen::Vector3f right;
      float opening = proposal.opening_m;
      std::vector<HandleOrientation> approach_variants;
      float corner_penalty = 0.0F;
      const int seed_test_start = tested;
      constexpr int hggd_variant_test_limit_per_seed = 9;
      LocalGraspSkill selected_grasp_skill = LocalGraspSkill::kUnknown;
      bool have_constrained_grasp_seed = false;
      std::vector<float> constrained_grasp_seed;
      const bool refine_hggd_geometry = pure_hggd;
      const bool use_measured_contacts = refine_hggd_geometry ||
        proposal.source == GraspProposalSource::kKnownCloud ||
        (require_measured_antipodal_contacts_ && !pure_learned);
      if (use_measured_contacts) {
        std::string contact_failure_reason;
        if (refine_hggd_geometry) {
          if (!proposal.has_contacts ||
              !proposal.has_local_dominant_axis) {
            ++rejected_contacts;
            ++contact_rejection_reasons[
              "hggd_local_axis_or_cross_section_unavailable"];
            seed_diagnostics.terminal_reason =
              "LOCAL_AXIS_OR_CROSS_SECTION_UNAVAILABLE";
            continue;
          }
          measured_contacts.left = proposal.contact_left;
          measured_contacts.right = proposal.contact_right;
          measured_contacts.closing_axis = proposal.closing;
          measured_contacts.opening = proposal.opening_m;
          measured_contacts.quality = 0.0F;
          measured_contacts.grasp_skill = proposal.grasp_skill;
          measured_contacts.silhouette_supported =
            proposal.silhouette_supported_contacts;
        } else if (proposal.has_contacts) {
          measured_contacts.left = proposal.contact_left;
          measured_contacts.right = proposal.contact_right;
          measured_contacts.closing_axis = proposal.closing;
          measured_contacts.opening = proposal.opening_m;
          measured_contacts.quality = 1.0F - proposal.quality;
          measured_contacts.body_model_supported = proposal.model_supported_contacts;
          measured_contacts.grasp_skill = proposal.grasp_skill;
        } else if (silhouette_pair_available) {
          // The two mask boundaries are the directly observed local handle
          // thickness. Do not discard them merely because GPD proposed the
          // loop-spanning closing direction; GPD remains the approach prior.
          measured_contacts = silhouette_pair;
        } else if (body_model_pair_cursor < body_model_pairs.size()) {
          measured_contacts = body_model_pairs[body_model_pair_cursor++];
        } else {
          if (!selectAntipodalContacts(
                  local_antipodal_pairs, closing, approach,
                  measured_contacts, contact_failure_reason)) {
            ++rejected_contacts;
            ++contact_rejection_reasons[contact_failure_reason];
            corner_penalty = 1.0F;

            const Eigen::Vector3f contact_center = proposal.hand_base + approach * static_cast<float>(grasp_contact_depth_m_);
            measured_contacts.left = contact_center + closing * (proposal.opening_m * 0.5F);
            measured_contacts.right = contact_center - closing * (proposal.opening_m * 0.5F);
            measured_contacts.opening = proposal.opening_m;
            measured_contacts.silhouette_supported = false;
            measured_contacts.body_model_supported = false;
            measured_contacts.closing_axis = closing;
          }
        }
        if (measured_contacts.body_model_supported) {
          ++body_model_contact_selected;
        } else {
          ++measured_contact_valid;
        }
        if (measured_contacts.silhouette_supported) {
          ++silhouette_contact_selected;
        }
        closing = measured_contacts.closing_axis;
        if (refine_hggd_geometry) {
          std::array<float, 6> capture_joints{};
          std::copy_n(joints.begin(), 6, capture_joints.begin());
          const Eigen::Matrix3f capture_rotation =
            Robot::eefk(capture_joints).linear();
          if (proposal.top_surface_grasp || proposal.rim_grasp) {
            // Broad top-down and rim grasps already have complete,
            // gravity-aligned poses.  The contact-axis solver is intended to
            // flex side/handle poses around a local PCA axis; applying it to
            // a rim uses the vertical cup-wall PCA axis as a jaw axis and can
            // turn a correct top-down rim pinch into a roll-tilted pose.
            approach_variants.push_back({
              proposal.approach.normalized(),
              proposal.closing.normalized(), 0.0F});
            ++geometry_approach_variants_generated;
            if (proposal.rim_grasp) {
              const Eigen::Vector3f contact_line =
                measured_contacts.left - measured_contacts.right;
              if (contact_line.norm() > 0.001F) {
                const auto measured_closing = contact_line.normalized();
                Eigen::Vector3f measured_approach = -Eigen::Vector3f::UnitZ() -
                  measured_closing * measured_closing.dot(-Eigen::Vector3f::UnitZ());
                if (measured_approach.norm() >= 0.707F &&
                    std::abs(measured_closing.z()) > 0.05F) {
                  approach_variants.push_back({measured_approach.normalized(),
                    measured_closing, 0.05F});
                  ++geometry_approach_variants_generated;
                }
              }
              // Keep the radial jaw line fixed and horizontal, but let the
              // wrist approach the same rim contacts with a modest pitch.
              // This supplies reachable 6-DoF alternatives without accepting
              // a loose-IK pose or rolling the jaws toward the cup wall PCA.
              const Eigen::Vector3f rim_closing =
                proposal.closing.normalized();
              const Eigen::Vector3f rim_approach =
                proposal.approach.normalized();
              for (const float degrees : {
                  -5.0F, 5.0F, -10.0F, 10.0F,
                  -15.0F, 15.0F, -20.0F, 20.0F}) {
                const Eigen::Vector3f tilted_approach =
                  (Eigen::AngleAxisf(
                    degrees * static_cast<float>(M_PI / 180.0),
                    rim_closing) * rim_approach).normalized();
                // This is a bounded pitch about the fixed radial jaw axis,
                // not an unconstrained yaw. Test it even when the generic
                // capture-frame Euler decomposition labels the same motion
                // as exceeding one component limit.
                approach_variants.push_back({
                  tilted_approach, rim_closing,
                  std::abs(degrees) / 90.0F});
                ++geometry_approach_variants_generated;
                ++hggd_orbit_variants_generated;
              }
            }
          } else {
          const Eigen::Vector3f contact_centroid = 0.5F *
            (measured_contacts.left + measured_contacts.right);
          const Eigen::Matrix3f aligned_rotation = piperGripperRotation(
            proposal.closing, proposal.approach);
          Eigen::Matrix3f resolved_rotation = aligned_rotation;
          if (solvePiperContactAxisIK(
              contact_centroid, proposal.local_dominant_axis,
              aligned_rotation, capture_rotation, joints,
              constrained_grasp_seed, resolved_rotation)) {
            have_constrained_grasp_seed = true;
            const Eigen::Vector3f resolved_approach =
              resolved_rotation.col(2).normalized();
            const Eigen::Vector3f resolved_closing =
              resolved_rotation.col(1).normalized();
            const float angular_deviation = std::acos(std::clamp(
              proposal.approach.normalized().dot(resolved_approach),
              -1.0F, 1.0F));
            approach_variants.push_back({
              resolved_approach, resolved_closing,
              angular_deviation / static_cast<float>(M_PI_2), true});
            ++geometry_approach_variants_generated;
          } else {
            ++hggd_contact_axis_fallback_seeds;
          }
          // The contact-axis solver is a preference, not proof that the
          // measured contact pair is unreachable. Try a small, deterministic
          // orbit around that same closing axis. Every variant still passes
          // the measured finger sweep, strict grasp IK, and collision checks.
          const Eigen::Vector3f orbit_axis = proposal.closing.normalized();
          const Eigen::Vector3f nominal_approach =
            proposal.approach.normalized();
          constexpr float orbit_dedup_cosine = 0.9961947F;  // 5 degrees
          for (const float degrees : {0.0F, -10.0F, 10.0F,
                                      -20.0F, 20.0F}) {
            const float radians = degrees * static_cast<float>(M_PI / 180.0);
            const Eigen::Vector3f orbit_approach =
              (Eigen::AngleAxisf(radians, orbit_axis) *
               nominal_approach).normalized();
            const Eigen::Matrix3f orbit_rotation = piperGripperRotation(
              orbit_axis, orbit_approach);
            if (!graspOrientationWithinCaptureLimits(
                capture_rotation, orbit_rotation)) continue;
            const bool duplicate = std::any_of(
              approach_variants.begin(), approach_variants.end(),
              [&orbit_approach, orbit_dedup_cosine](
                  const HandleOrientation &existing) {
                return existing.approach.dot(orbit_approach) >
                  orbit_dedup_cosine;
              });
            if (duplicate) continue;
            approach_variants.push_back({
              orbit_approach, orbit_axis,
              std::abs(degrees) / 90.0F, false});
            ++geometry_approach_variants_generated;
            ++hggd_orbit_variants_generated;
          }
          if (approach_variants.empty()) {
            ++rejected_contacts;
            ++contact_rejection_reasons[
              "contact_axis_and_bounded_orbit_unreachable"];
            seed_diagnostics.terminal_reason =
              "CONTACT_AXIS_AND_ORBIT_UNREACHABLE";
            continue;
          }
          }
        } else if (measured_contacts.silhouette_supported ||
            measured_contacts.body_model_supported) {
          const Eigen::Vector3f contact_midpoint =
            (measured_contacts.left + measured_contacts.right) * 0.5F;
          // Both silhouette contacts were observed along an unobstructed
          // camera ray. Approach from that visible side instead of accepting
          // a GPD direction that may pass through the handle loop or cup.
          Eigen::Vector3f visible_approach = contact_midpoint - camera_origin;
          visible_approach -= closing * visible_approach.dot(closing);
          if (visible_approach.squaredNorm() < 1.0e-6F) {
            ++rejected_contacts;
            ++contact_rejection_reasons["degenerate_visible_approach"];
            continue;
          }
          visible_approach.normalize();
          approach_variants = buildVisibleHandleApproaches(
            closing, visible_approach);
          if (approach_variants.empty()) {
            ++rejected_contacts;
            ++contact_rejection_reasons["no_unique_visible_approach"];
            continue;
          }
          geometry_approach_variants_generated +=
            static_cast<int>(approach_variants.size());
        }
        if (approach_variants.empty()) {
          approach_variants.push_back({approach, closing, 0.0F});
        }
      } else {
        approach_variants.push_back({approach, closing, 0.0F});
      }

      for (const HandleOrientation &orientation : approach_variants) {
        ++seed_diagnostics.orientations_considered;
        closing = orientation.closing.normalized();
        approach = orientation.approach - closing *
          orientation.approach.dot(closing);
        if (approach.squaredNorm() < 1.0e-6F) {
          ++rejected_contacts;
          ++contact_rejection_reasons["degenerate_refined_approach"];
          ++seed_diagnostics.degenerate_orientations;
          continue;
        }
        approach.normalize();
        const Eigen::Vector3f height_axis =
          approach.cross(closing).normalized();
        bool measured_sweep_valid = false;
        if (use_measured_contacts) {
          AntipodalContacts oriented_contacts = measured_contacts;
          if (refine_hggd_geometry && !proposal.top_surface_grasp) {
            // Constrained IK keeps the aligned contact centroid fixed while
            // changing approach continuously. Preserve the aligned opening
            // and reconstruct its contact line in the resolved closing
            // direction; do not run another discrete cross-section sweep.
            const Eigen::Vector3f midpoint = 0.5F *
              (measured_contacts.left + measured_contacts.right);
            oriented_contacts.closing_axis = closing;
            oriented_contacts.left = midpoint + closing *
              (0.5F * measured_contacts.opening);
            oriented_contacts.right = midpoint - closing *
              (0.5F * measured_contacts.opening);
            oriented_contacts.opening = measured_contacts.opening;
            oriented_contacts.grasp_skill = measured_contacts.grasp_skill;
            oriented_contacts.silhouette_supported =
              measured_contacts.silhouette_supported;
          }
          left = oriented_contacts.left;
          right = oriented_contacts.right;
          opening = oriented_contacts.opening;
          const Eigen::Vector3f contact_midpoint = (left + right) * 0.5F;
          // The jaw opening and pose must describe the same measured pair.
          // Top-surface hypotheses choose two observed edge samples; those
          // samples can lie at different positions along a curved cup rim
          // even when their projected PCA width looks plausible. Reject that
          // geometry before the sweep and IK checks grant jaw-pad contact.
          const Eigen::Vector3f contact_span = left - right;
          const float measured_jaw_span = contact_span.dot(closing);
          const float tangent_offset = std::abs(contact_span.dot(height_axis));
          const float span_tolerance = std::max(0.003F, 0.20F * opening);
          const float tangent_tolerance = std::max(0.003F, 0.27F * opening);
          if (measured_jaw_span <= 0.0F ||
              std::abs(measured_jaw_span - opening) > span_tolerance ||
              tangent_offset > tangent_tolerance) {
            ++rejected_contacts;
            ++contact_rejection_reasons["contact_pair_misaligned_with_jaw"];
            ++seed_diagnostics.cross_section_rejected;
            continue;
          }
          // The two-sided section probe protects handle pinches from their
          // junction and tip. A requested upper rim is intentionally at an
          // endpoint; its separate top-band contact test, finger sweep, and
          // full collision checks remain mandatory.
          if (affordance != "RIM" && !proposal.rim_grasp &&
              !pinchSectionAwayFromJunction(
                target_samples, oriented_contacts, closing, height_axis,
                approach)) {
            ++rejected_contacts;
            ++contact_rejection_reasons[
              "pinch_junction_or_endpoint"];
            ++seed_diagnostics.cross_section_rejected;
            continue;
          }
          if (affordance != "RIM" && !proposal.rim_grasp &&
              !oriented_contacts.silhouette_supported &&
              !contactSurfaceNormalsSupportPinch(
                target_samples, oriented_contacts, closing)) {
            ++rejected_contacts;
            ++contact_rejection_reasons[
              "pinch_contact_normals_not_antipodal"];
            ++seed_diagnostics.cross_section_rejected;
            continue;
          }
          // Analytical PCA extent pairs can lie diagonally across a sloped
          // plate rim. Require measured pad-normal alignment even when the
          // proposal has no PINCH skill tag; collision freedom alone is not
          // evidence that closing the jaws will hold the object stationary.
          if (proposal.source == GraspProposalSource::kKnownCloud &&
              !contactSurfaceNormalsSupportPinch(
                target_samples, oriented_contacts, closing, true)) {
            ++rejected_contacts;
            ++contact_rejection_reasons["known_cloud_pad_surface_misalignment"];
            continue;
          }
          // grasp_contact_depth_m is the commanded distance from the physical
          // finger base to the measured contact plane. grasp_tcp_offset_m is a
          // separate, fixed hand-base-to-Piper-TCP transform, so the two are
          // intentionally composed rather than double-counted.
          hand_base = contact_midpoint -
            approach * static_cast<float>(grasp_contact_depth_m_);
          std::array<float, 3> blocked_local{};
          if (!fingerVolumesAndClosingSweepClear(
                target_samples, oriented_contacts, closing, height_axis,
                approach, &blocked_local)) {
            ++rejected_finger_sweep;
            ++seed_diagnostics.sweep_rejected;
            if (!have_target_sweep_block) {
              have_target_sweep_block = true;
              first_target_sweep_block = blocked_local;
              first_target_sweep_opening = opening;
              first_target_sweep_silhouette =
                oriented_contacts.silhouette_supported;
            }
            continue;
          }
          measured_sweep_valid = true;
          if (oriented_contacts.grasp_skill == LocalGraspSkill::kPinch) {
            ++pinch_candidates_generated;
          } else if (oriented_contacts.grasp_skill ==
              LocalGraspSkill::kPowerGrasp) {
            ++power_candidates_generated;
          }
          selected_grasp_skill = oriented_contacts.grasp_skill;
        } else {
          const Eigen::Vector3f contact_center = proposal.hand_base +
            approach * static_cast<float>(grasp_contact_depth_m_);
          left = contact_center + closing * (opening * 0.5F);
          right = contact_center - closing * (opening * 0.5F);
        }
        const Eigen::Vector3f tcp = hand_base +
          approach * static_cast<float>(grasp_tcp_offset_m_);
        const Eigen::Vector3f contact_midpoint = (left + right) * 0.5F;
        Eigen::Matrix3f rotation = piperGripperRotation(closing, approach);
        Eigen::Isometry3f target_pose = Eigen::Isometry3f::Identity();
        target_pose.linear() = rotation;
        target_pose.translation() = tcp;
        // Empirical command-space correction, before IK and collision checks.
        // Preserve measured contacts and all scene geometry. Exact known-cloud
        // benchmark proposals do not inherit a physical observation trial offset.
        target_pose = vamp_node::correctedGraspTarget(target_pose,
          grasp_position_offset_base_m_, proposal.source != GraspProposalSource::kKnownCloud);

        // Initial grasp validation uses the coarse obstacle world only.
        // Detailed target contact/closing-sweep validation remains above.
      // Only classifier-positive, physically legal openings enter the
      // comparatively expensive IK and collision-validation stages.
      if (search_budget_exhausted() ||
          (pure_hggd && tested - seed_test_start >=
            hggd_variant_test_limit_per_seed)) {
        seed_diagnostics.ik_budget_exhausted = true;
        if (seed_diagnostics.terminal_reason.empty()) {
          seed_diagnostics.terminal_reason = "VARIANT_TEST_BUDGET";
        }
        break;
      }
      if (pure_hggd && tested == candidate_test_limit && response->candidates.empty()) {
        RCLCPP_INFO(this->get_logger(),
          "[GRASP SEARCH] no valid fast result; continuing bounded fallback budget=%d",
          maximum_candidates_to_test_);
      }
      ++tested;
      ++seed_diagnostics.ik_submitted;
      bool any_grasp_ik = false;
      bool any_pregrasp_ik = false;
      bool any_pregrasp_visible = !pure_hggd;
      bool any_grasp_collision_free = false;
      bool accepted = false;
      bool accepted_pregrasp_collision_free = !enable_grasp_collision_check_;
      bool have_pregrasp_collision_fallback = false;
      std::string accepted_alignment_status;
      float accepted_opening = opening;
      std::vector<my_actions::msg::GraspCandidate> selected_angle_trials;
      float accepted_camera_upright_score = -std::numeric_limits<float>::infinity();
      float fallback_camera_upright_score = -std::numeric_limits<float>::infinity();
      std::vector<float> ik;
      std::vector<float> pregrasp_ik;
      Eigen::Isometry3f fallback_target = target_pose;
      std::vector<float> fallback_ik;
      std::vector<float> fallback_pregrasp_ik;
      Eigen::Isometry3f accepted_pregrasp_pose = target_pose;
      Eigen::Isometry3f fallback_pregrasp_pose = target_pose;
      PoseIkDiagnostics best_grasp_diagnostics;
      PoseIkDiagnostics best_pregrasp_diagnostics;
      // A parallel-jaw grasp is invariant to a 180-degree roll about its
      // approach direction. Test both finger assignments because only one
      // may fall inside Piper's wrist limits.
      const auto original_variant_target = target_pose;
      const float original_variant_opening = opening;
      for (const bool swap_fingers : {false, true}) {
        opening = original_variant_opening;
        Eigen::Isometry3f branch_target = original_variant_target;
        branch_target.linear() = piperGripperRotation(closing, approach, swap_fingers);
        std::vector<float> branch_ik;
        PoseIkDiagnostics grasp_diagnostics;
        const bool constrained_seed_only =
          have_constrained_grasp_seed && orientation.constrained_seed &&
          !swap_fingers;
        const std::vector<float> &grasp_seed =
          constrained_seed_only ? constrained_grasp_seed : joints;
        const float early_contact_allowance_m =
          (affordance == "RIM" || proposal.rim_grasp) ? std::min(
            0.008F, static_cast<float>(target_collision_clearance_m_)) :
          proposal.top_surface_grasp ? 0.050F :
            static_cast<float>(target_collision_clearance_m_);
        const auto valid_grasp_solution = [&](const std::vector<float> &solution) {
          return !enable_grasp_collision_check_ ||
            isCandidatePoseSafe(solution, env, &env_target_flags, &left, &right,
              early_contact_allowance_m, &branch_target, -1.0F, true, true);
        };
        // Deployment stops at the first collision-safe IK branch. Rejected
        // branches continue through the existing deterministic seed bank;
        // all downstream contact, pregrasp and complete-path checks remain.
        if (!solvePiperPoseIK(
            branch_target, grasp_seed, branch_ik, grasp_diagnostics,
            constrained_seed_only, 0.004F, 0.20F, 800,
            pure_hggd ? valid_grasp_solution :
              std::function<bool(const std::vector<float> &)>{}, pure_hggd)) {
          if (grasp_diagnostics.collision_rejections > 0) any_grasp_ik = true;
          best_grasp_diagnostics.keepBest(grasp_diagnostics);
          continue;
        }
        any_grasp_ik = true;
        std::vector<my_actions::msg::GraspCandidate> branch_angle_trials;
        std::array<float,6> original_resolved_joints{};
        std::copy_n(branch_ik.begin(),6,original_resolved_joints.begin());
        // Bend from the actual IK pose shown in RViz, including its solve error.
        const auto untuned_target = Robot::eefk(original_resolved_joints);
        const auto untuned_joints = branch_ik;
        bool tuned_route_available = false;
        for (int alignment_route = 0; alignment_route < 2; ++alignment_route) {
          if (alignment_route == 1 && !tuned_route_available) break;
          branch_target = untuned_target;
          branch_ik = untuned_joints;
          opening = original_variant_opening;
          bool route_accepted = false;
          std::string alignment_status = pure_hggd && use_measured_contacts ?
            "ANGLE_TUNING_FAILED_ORIGINAL_USED;ANGLE_SELECTION=ORIGINAL (tuning unavailable or rejected)" : "";
        Eigen::Vector3f aligned_surface_normal;
        float accepted_surface_tolerance = 0.0F;
        // Surface alignment is bounded to three trials after original IK succeeds.
        // Keep both measured contacts; never invent a new pair to force a fit.
        if (alignment_route == 0 && pure_hggd && use_measured_contacts) {
          Eigen::Vector3f surface_normal;
          const bool normals_available = estimateObjectNormal(
            target_samples, contact_midpoint, surface_normal);
          if (!normals_available) {
            ++contact_rejection_reasons["pad_surface_normal_unavailable"];
          } else {
          const auto original_target = branch_target;
          const auto original_joints = branch_ik;
          const auto record_angle_trial = [&](const Eigen::Isometry3f &pose,
              float degrees, const std::string &reason) {
            if (branch_angle_trials.size() >= 3) return;
            const Eigen::Quaternionf q(pose.linear());
            for (const auto &existing : branch_angle_trials) {
              const auto &p = existing.grasp_pose.pose;
              const Eigen::Vector3f delta(p.position.x-pose.translation().x(),
                p.position.y-pose.translation().y(),p.position.z-pose.translation().z());
              const Eigen::Quaternionf previous(p.orientation.w,p.orientation.x,
                p.orientation.y,p.orientation.z);
              if (delta.norm()<.001F && q.angularDistance(previous)<.01745F &&
                  existing.validation_status.find(std::to_string(static_cast<int>(degrees))+" deg")!=std::string::npos)
                return;
            }
            my_actions::msg::GraspCandidate trial;
            trial.id=branch_angle_trials.size()+1;
            trial.grasp_pose.header=base_point.header;
            trial.grasp_pose.header.frame_id="base_link";
            trial.grasp_pose.pose.position.x=pose.translation().x();
            trial.grasp_pose.pose.position.y=pose.translation().y();
            trial.grasp_pose.pose.position.z=pose.translation().z();
            trial.grasp_pose.pose.orientation.x=q.x();trial.grasp_pose.pose.orientation.y=q.y();
            trial.grasp_pose.pose.orientation.z=q.z();trial.grasp_pose.pose.orientation.w=q.w();
            trial.pregrasp_pose=trial.grasp_pose;
            trial.contact_left.x=left.x();trial.contact_left.y=left.y();trial.contact_left.z=left.z();
            trial.contact_right.x=right.x();trial.contact_right.y=right.y();trial.contact_right.z=right.z();
            trial.jaw_opening_m=std::max(.002F,
              std::abs((left-right).dot(pose.linear().col(1))));
            trial.validation_status="PREVIEW ONLY - "+std::to_string(static_cast<int>(degrees))+
              " deg to surface\n"+reason;
            branch_angle_trials.push_back(trial);
          };
          const float original_surface_angle = vamp_node::fingerSurfaceAngle(
            original_target.linear(),surface_normal);
          const auto bend_targets = vamp_node::outwardBendTargets(original_surface_angle);
          bool aligned = bend_targets.empty();
          if (aligned) {
            alignment_status = "PAD_SURFACE_ALIGNMENT_VALIDATED;ANGLE_ALREADY_WITHIN_BAND;ANGLE_SELECTION=ORIGINAL (already within 10deg)";
            aligned_surface_normal = surface_normal;
            accepted_surface_tolerance = 10.F*static_cast<float>(M_PI/180.0);
            tuned_route_available = true;
            RCLCPP_INFO(this->get_logger(),
              "[PAD SURFACE ALIGNMENT] original=%.1fdeg already within 10deg; no sweep",
              original_surface_angle*180.F/static_cast<float>(M_PI));
          }
          int align_geometry_fail=0, align_ik_fail=0, align_pad_fail=0, align_sweep_fail=0;
          for (const float degrees : bend_targets) {
            if (search_budget_exhausted()) break;
            const float tolerance = degrees * static_cast<float>(M_PI / 180.0);
            const auto corrected = vamp_node::bendOriginalGraspOutward(
              original_target, surface_normal, contact_midpoint, alignment_object_center,
              grasp_tcp_offset_m_, grasp_contact_depth_m_, tolerance-.01F);
            if (!corrected) {
              ++align_geometry_fail;
              continue;
            }
            auto adjusted = *corrected;
            const float trial_opening=std::max(.002F,
              std::abs((left-right).dot(adjusted.linear().col(1))));
            if (trial_opening>maximum_gripper_width_) continue;
            std::vector<float> adjusted_joints;
            PoseIkDiagnostics diagnostics;
            std::array<float, 6> original_q{};
            std::copy_n(original_joints.begin(), 6, original_q.begin());
            const auto original_actual = Robot::eefk(original_q);
            if (vamp_node::fingerSurfaceAngle(original_actual.linear(),surface_normal) <= tolerance) {
              adjusted_joints = original_joints;
            } else if (!solvePiperPoseIK(adjusted, original_joints, adjusted_joints,
                diagnostics, false, .003F, .01F, 800)) {
              ++align_ik_fail;
              record_angle_trial(adjusted,degrees,"REJECTED IK"); continue;
            }
            std::array<float, 6> q{};
            std::copy_n(adjusted_joints.begin(), 6, q.begin());
            auto actual = Robot::eefk(q);
            if (!vamp_node::handAdjustmentIsOutward(original_target, actual,
                contact_midpoint, alignment_object_center, grasp_tcp_offset_m_) ||
                !vamp_node::sameOriginalHinge(original_target.linear(),actual.linear()) ||
                vamp_node::fingerSurfaceAngle(actual.linear(),surface_normal) > tolerance) {
              record_angle_trial(actual,degrees,"REJECTED actual angle/direction"); continue;
            }
            const std::array<Eigen::Vector3f,2> tilted_pair{{left,right}};
            auto pad_advance = vamp_node::contactPadAdvance(actual,tilted_pair,
              trial_opening,finger_length_m_,finger_height_m_,grasp_tcp_offset_m_);
            float total_advance = 0.F;
            // At most two seeded solves, with no more than 10 mm total advance.
            for (int retry=0; pad_advance && *pad_advance>0.F && retry<2; ++retry) {
              if (search_budget_exhausted() || total_advance+*pad_advance>.010F) break;
              auto recentered=actual;
              recentered.translation() += actual.linear().col(2)* *pad_advance;
              std::vector<float> recentered_joints;
              PoseIkDiagnostics recenter_diagnostics;
              if (!solvePiperPoseIK(recentered,adjusted_joints,recentered_joints,
                  recenter_diagnostics,false,.0005F,.01F,800)) break;
              total_advance += *pad_advance;
              std::copy_n(recentered_joints.begin(),6,q.begin());
              actual=Robot::eefk(q);
              adjusted_joints=recentered_joints;
              pad_advance=vamp_node::contactPadAdvance(actual,tilted_pair,
                trial_opening,finger_length_m_,finger_height_m_,grasp_tcp_offset_m_,
                .010F-total_advance);
            }
            if (!pad_advance || *pad_advance>0.F) {
              ++align_pad_fail;
              record_angle_trial(actual,degrees,"REJECTED contact reach after bounded advance");
              continue;
            }
            if (!vamp_node::handAdjustmentIsOutward(original_target,actual,
                contact_midpoint,alignment_object_center,grasp_tcp_offset_m_) ||
                !vamp_node::sameOriginalHinge(original_target.linear(),actual.linear()) ||
                vamp_node::fingerSurfaceAngle(actual.linear(),surface_normal)>tolerance) {
              record_angle_trial(actual,degrees,"REJECTED recentered angle/direction");continue;
            }
            if (total_advance>0.F) RCLCPP_INFO(this->get_logger(),
              "[PAD CONTACT RECENTER] advance=%.2fmm both measured contacts inside unchanged pad bounds",
              total_advance*1000.F);
            auto tilted_contacts = measured_contacts;
            tilted_contacts.opening = trial_opening;
            if (!fingerVolumesAndClosingSweepClear(target_samples, tilted_contacts,
                actual.linear().col(1), actual.linear().col(0), actual.linear().col(2))) {
              ++align_sweep_fail;
              record_angle_trial(actual,degrees,"REJECTED finger sweep"); continue;
            }
            branch_target = actual;
            if (!valid_grasp_solution(adjusted_joints)) {
              record_angle_trial(actual,degrees,"REJECTED collision");
              branch_target = original_target;
              continue;
            }
            record_angle_trial(actual,degrees,"Alignment passed; trial layer is not acceptance");
            branch_ik = adjusted_joints;
            opening = trial_opening;
            aligned = true;
            tuned_route_available = true;
            alignment_status = "PAD_SURFACE_ALIGNMENT_VALIDATED;ANGLE_SELECTION=ADJUSTED target="+
              std::to_string(static_cast<int>(degrees))+"deg";
            if (total_advance>0.F) alignment_status += ";CONTACT_DEPTH_RECENTERED";
            aligned_surface_normal = surface_normal;
            accepted_surface_tolerance = tolerance;
            RCLCPP_INFO(this->get_logger(),
              "[PAD SURFACE ALIGNMENT] limit=%.0fdeg actual=%.1fdeg outward_hand_delta=%.3fmm measured contacts retained",
              degrees, vamp_node::fingerSurfaceAngle(actual.linear(),surface_normal) *
                180.0F / static_cast<float>(M_PI),
              1000.F * *vamp_node::outwardHandDisplacement(original_target,actual,
                contact_midpoint,alignment_object_center,grasp_tcp_offset_m_));
            break;
          }
          if (!aligned) {
            ++contact_rejection_reasons["pad_surface_alignment_unreachable"];
            if (contact_rejection_reasons["pad_surface_alignment_unreachable"] <= 3)
              RCLCPP_INFO(this->get_logger(),
                "[PAD SURFACE FALLBACK] original=%.1fdeg trials={geometry:%d,ik:%d,pad_reach:%d,sweep:%d}",
                vamp_node::fingerSurfaceAngle(original_target.linear(),surface_normal)*
                  180.F/static_cast<float>(M_PI),
                align_geometry_fail,align_ik_fail,align_pad_fail,align_sweep_fail);
            branch_target = untuned_target;
            branch_ik = untuned_joints;
          }
          }
        }
        // Palm recovery is triggered ONLY by sphere 80 or 81 at the initial grasp.
        // Keep measured contacts fixed; retreat the whole TCP without extending
        // the fingers beyond their measured reach. Other initial gripper-sphere
        // exclusions remain unchanged.
        if (enable_grasp_collision_check_ &&
            !graspPalmRecoveryClear(branch_ik, env, opening)) {
          bool recovered = false;
          const auto original_target = branch_target;
          const auto original_joints = branch_ik;
          for (const float retreat_m : {0.002F, 0.004F, 0.006F, 0.008F, 0.010F}) {
            auto adjusted = original_target;
            adjusted.translation() -= adjusted.linear().col(2)*retreat_m;
            std::vector<float> adjusted_joints;
            PoseIkDiagnostics recovery_diagnostics;
            if (!solvePiperPoseIK(adjusted, original_joints, adjusted_joints,
                recovery_diagnostics, false, .004F, .03F, 800)) continue;
            std::array<float, 6> q{};
            std::copy_n(adjusted_joints.begin(), 6, q.begin());
            const auto actual = Robot::eefk(q);
            bool contacts_reachable = true;
            for (const auto &contact : {left, right}) {
              const Eigen::Vector3f local = actual.linear().transpose()*
                (contact-actual.translation());
              const float depth = local.z()+static_cast<float>(grasp_tcp_offset_m_);
              if (depth < .002F || depth > finger_length_m_-.002F ||
                  std::abs(local.x()) > finger_height_m_*.5+.003F ||
                  std::abs(std::abs(local.y())-opening*.5F) > .004F)
                contacts_reachable = false;
            }
            if (!contacts_reachable || !graspPalmRecoveryClear(adjusted_joints, env, opening) ||
                !valid_grasp_solution(adjusted_joints)) continue;
            branch_target = actual;
            branch_ik = adjusted_joints;
            recovered = true;
            RCLCPP_INFO(this->get_logger(),
              "[PALM80_81 RECOVERY] whole-gripper retreat=%.1fmm contacts preserved", retreat_m*1000.F);
            break;
          }
          if (!recovered) {
            any_grasp_ik = true;
            RCLCPP_INFO(this->get_logger(),
              "[PALM80_81 RECOVERY] rejected: no contact-preserving clearance within 10mm");
            continue;
          }
        }
        // Final measured-pad/table validation applies to the original fallback
        // too. Coarse arm-only checks cannot certify finger/table clearance.
        if (pure_hggd && use_measured_contacts) {
          const std::array<Eigen::Vector3f,2> contact_pair{{left,right}};
          if (!grasp_support_plane) {
            ++contact_rejection_reasons["support_plane_unavailable"];
            continue;
          }
          const auto contact_safe = [&](const Eigen::Isometry3f &pose, float jaw) {
            return vamp_node::measuredPadContactsSupported(pose,contact_pair,jaw,
              finger_length_m_,finger_height_m_,grasp_tcp_offset_m_) &&
              vamp_node::fingerSupportClearance(pose,*grasp_support_plane,
                std::min(.10F,jaw+.02F),finger_length_m_,finger_width_m_,
                finger_height_m_,grasp_tcp_offset_m_)>=.003F;
          };
          float maximum_depth=-std::numeric_limits<float>::infinity();
          for (const auto &contact : contact_pair)
            maximum_depth=std::max(maximum_depth,
              (branch_target.linear().transpose()*(contact-branch_target.translation())).z()+
              static_cast<float>(grasp_tcp_offset_m_));
          bool shallow_ok=contact_safe(branch_target,opening) &&
            maximum_depth>=finger_length_m_-.006F;
          const auto original_contact_pose=branch_target;
          const auto original_contact_joints=branch_ik;
          if (!shallow_ok) for (const auto &target : vamp_node::shallowContactTargets(
              original_contact_pose,contact_pair,opening,finger_length_m_,finger_width_m_,
              finger_height_m_,grasp_tcp_offset_m_,*grasp_support_plane)) {
            std::vector<float> refined_joints;
            PoseIkDiagnostics diagnostics;
            if (!solvePiperPoseIK(target,original_contact_joints,refined_joints,diagnostics,
                false,.0005F,.01F,600)) continue;
            std::array<float,6> q{};std::copy_n(refined_joints.begin(),6,q.begin());
            const auto actual=Robot::eefk(q);
            const float jaw=std::abs((left-right).dot(actual.linear().col(1)));
            if (jaw<minimum_gripper_width_ || jaw>maximum_gripper_width_ ||
                !contact_safe(actual,jaw) ||
                !graspPalmRecoveryClear(refined_joints,env,jaw) ||
                !valid_grasp_solution(refined_joints)) continue;
            auto actual_contacts=measured_contacts;actual_contacts.opening=jaw;
            if (!fingerVolumesAndClosingSweepClear(target_samples,actual_contacts,
                actual.linear().col(1),actual.linear().col(0),actual.linear().col(2),
                nullptr,&actual)) continue;
            branch_target=actual;branch_ik=refined_joints;opening=jaw;
            shallow_ok=true;
            RCLCPP_INFO(this->get_logger(),
              "[SHALLOW GRASP] adjusted=%.2fmm table_clearance=%.2fmm contacts preserved",
              1000.F*(actual.translation()-original_contact_pose.translation()).norm(),
              1000.F*vamp_node::fingerSupportClearance(actual,*grasp_support_plane,
                std::min(.10F,jaw+.02F),finger_length_m_,finger_width_m_,finger_height_m_,grasp_tcp_offset_m_));
            break;
          }
          if (!shallow_ok) {
            ++contact_rejection_reasons["shallow_contact_or_table_clearance"];
            continue;
          }
          alignment_status += ";SHALLOW_CONTACT_VALIDATED;FINGER_TABLE_CLEARANCE_VALIDATED";
        }
        if (accepted_surface_tolerance > 0.F) {
          const auto final_pad = vamp_node::contactPadAdvance(branch_target,
            std::array<Eigen::Vector3f,2>{{left,right}},opening,
            finger_length_m_,finger_height_m_,grasp_tcp_offset_m_);
          if (!final_pad || *final_pad>0.F) {
            ++contact_rejection_reasons["pad_contacts_lost_after_palm_recovery"];
            continue;
          }
        }
        if (accepted_surface_tolerance > 0.0F &&
            (!vamp_node::handAdjustmentIsOutward(untuned_target, branch_target,
               contact_midpoint, alignment_object_center, grasp_tcp_offset_m_) ||
             !vamp_node::sameOriginalHinge(untuned_target.linear(),branch_target.linear()) ||
             vamp_node::fingerSurfaceAngle(branch_target.linear(),aligned_surface_normal) >
               accepted_surface_tolerance)) {
          ++contact_rejection_reasons["pad_surface_alignment_lost_after_palm_recovery"];
          continue;
        }
        if (proposal.source == GraspProposalSource::kKnownCloud) {
          std::array<float, 6> actual_joints{};
          std::copy_n(branch_ik.begin(), 6, actual_joints.begin());
          const Eigen::Vector3f actual_closing =
            Robot::eefk(actual_joints).linear().col(1);
          // Check actual FK, including either finger assignment.
          AntipodalContacts actual_contacts = measured_contacts;
          if (actual_closing.dot(actual_contacts.closing_axis) < 0.0F) {
            std::swap(actual_contacts.left, actual_contacts.right);
          }
          if (!contactSurfaceNormalsSupportPinch(
              target_samples, actual_contacts, actual_closing, true)) {
            ++rejected_contacts;
            ++contact_rejection_reasons["known_cloud_ik_pad_surface_misalignment"];
            continue;
          }
        }
        any_grasp_ik = true;
        best_grasp_diagnostics.keepBest(grasp_diagnostics);
        // A top-down power grasp contacts a flat box wall along the jaw pad,
        // rather than at one mathematical point. Keep the configured
        // allowance for other grasps, but cover the bounded 50 mm cached
        // contact region for this measured topology. The exemption below is
        // still restricted to segmented-target points and inner-pad spheres;
        // palm, finger exterior, arm, table, and clutter remain blocking.
        const float candidate_contact_allowance_m =
          (affordance == "RIM" || proposal.rim_grasp) ? std::min(
            0.008F, static_cast<float>(target_collision_clearance_m_)) :
          proposal.top_surface_grasp ? 0.050F :
            static_cast<float>(target_collision_clearance_m_);
        if (enable_grasp_collision_check_ && !isCandidatePoseSafe(
            branch_ik, env, &env_target_flags, &left, &right,
            candidate_contact_allowance_m,
            &branch_target, -1.0F, true, true))
        {
          if (logged_grasp_collision_detail < 4) {
            RCLCPP_INFO(this->get_logger(),
              "[GRASP COLLISION DETAIL] coarse spheres 0..67 rejection; palm 80/81 checked by recovery");
            ++logged_grasp_collision_detail;
          }
          continue;
        }
        any_grasp_collision_free = true;
        for (const float retreat : pregraspRetreatCandidates()) {
          Eigen::Isometry3f branch_pregrasp = branch_target;
          branch_pregrasp.translation() -= branch_target.linear().col(2) * retreat;
          Eigen::Isometry3f resolved_pregrasp = branch_pregrasp;
          std::vector<float> branch_pregrasp_ik;
          PoseIkDiagnostics pregrasp_diagnostics;
          if (!solveRelaxedPregraspIK(
              branch_pregrasp, branch_ik, branch_pregrasp_ik,
              pregrasp_diagnostics, resolved_pregrasp))
          {
            best_pregrasp_diagnostics.keepBest(pregrasp_diagnostics);
            continue;
          }
          const float actual_standoff =
            (branch_target.translation() -
             resolved_pregrasp.translation()).norm();
          const float actual_orientation_delta = Eigen::AngleAxisf(
            branch_target.linear().transpose() *
            resolved_pregrasp.linear()).angle();
          if (!std::isfinite(actual_standoff) ||
              actual_standoff < 0.05F || actual_standoff > 0.195F ||
              !std::isfinite(actual_orientation_delta) ||
              actual_orientation_delta > static_cast<float>(
                pregrasp_orientation_tolerance_degrees_ * M_PI / 180.0) +
                1.0e-4F) {
            ++rejected_pregrasp_envelope;
            continue;
          }
          any_pregrasp_ik = true;
          best_pregrasp_diagnostics.keepBest(pregrasp_diagnostics);
          if (pure_hggd &&
            (!observation_camera_transform || !camera_info ||
            !pregraspKeepsSeedVisible(
              joints, branch_pregrasp_ik, *observation_camera_transform,
              *camera_info, contact_midpoint)))
          {
            continue;
          }
          any_pregrasp_visible = true;
          const float camera_upright_score = pure_hggd &&
            observation_camera_transform ?
            cameraUprightScore(
              joints, branch_pregrasp_ik,
              *observation_camera_transform).value_or(-1.0F) : 1.0F;
          if (enable_grasp_collision_check_ &&
              !isCandidatePoseSafe(branch_pregrasp_ik, env, nullptr, nullptr, nullptr,
                0.0F, nullptr, std::min(0.10F, opening + 0.02F)))
          {
            if (allow_pregrasp_collision_candidates_ &&
                (!have_pregrasp_collision_fallback ||
                camera_upright_score > fallback_camera_upright_score))
            {
              fallback_target = branch_target;
              fallback_ik = branch_ik;
              fallback_pregrasp_ik = branch_pregrasp_ik;
              fallback_pregrasp_pose = resolved_pregrasp;
              fallback_camera_upright_score = camera_upright_score;
              have_pregrasp_collision_fallback = true;
            }
            continue;
          }
          accepted_pregrasp_collision_free = true;
          route_accepted = true;
          if (!accepted ||
              camera_upright_score > accepted_camera_upright_score) {
            target_pose = branch_target;
            rotation = branch_target.rotation();
            ik = branch_ik;
            pregrasp_ik = std::move(branch_pregrasp_ik);
            accepted_pregrasp_pose = resolved_pregrasp;
            accepted_camera_upright_score = camera_upright_score;
            accepted_alignment_status = alignment_status;
            accepted_opening = opening;
            selected_angle_trials = branch_angle_trials;
            accepted = true;
          }
          // The configured/preferred retreat was feasible for this wrist
          // branch. Remaining distances cannot improve visibility enough to
          // justify additional IK work.
          break;
        }
          // Never return a tuned pose whose ordinary validation failed.
          // Retry the immutable original once; it still needs every old check.
          if (route_accepted) break;
        }
      }
      if (!accepted && have_pregrasp_collision_fallback && !pure_hggd) {
        target_pose = fallback_target;
        rotation = fallback_target.rotation();
        ik = std::move(fallback_ik);
        pregrasp_ik = std::move(fallback_pregrasp_ik);
        accepted_pregrasp_pose = fallback_pregrasp_pose;
        accepted_camera_upright_score = fallback_camera_upright_score;
        accepted = true;
        accepted_pregrasp_collision_free = false;
        ++retained_pregrasp_overrides;
      }
      if (!accepted) {
        std::ostringstream reason;
        if (!any_grasp_ik) {
          ++rejected_ik;
          ++seed_diagnostics.grasp_ik_rejected;
          seed_diagnostics.best_grasp_ik.keepBest(best_grasp_diagnostics);
          reason << "GRASP_IK_FAILED pos=" << best_grasp_diagnostics.best_position_error
                 << " rot=" << best_grasp_diagnostics.best_rotation_error
                 << " attempts=" << best_grasp_diagnostics.attempts;
        } else if (!any_grasp_collision_free) {
          ++rejected_grasp_collision;
          ++seed_diagnostics.grasp_collision_rejected;
          reason << "GRASP_COLLISION";
        } else if (!any_pregrasp_ik) {
          ++rejected_pregrasp_ik;
          ++seed_diagnostics.pregrasp_ik_rejected;
          seed_diagnostics.best_pregrasp_ik.keepBest(best_pregrasp_diagnostics);
          reason << "PREGRASP_IK_FAILED pos=" << best_pregrasp_diagnostics.best_position_error
                 << " rot=" << best_pregrasp_diagnostics.best_rotation_error
                 << " attempts=" << best_pregrasp_diagnostics.attempts;
        } else if (!any_pregrasp_visible) {
          ++rejected_pregrasp_visibility;
          ++seed_diagnostics.pregrasp_visibility_rejected;
          reason << "PREGRASP_SEED_OUTSIDE_CAMERA_FRAME";
        } else {
          ++rejected_pregrasp_collision;
          ++seed_diagnostics.pregrasp_collision_rejected;
          reason << "PREGRASP_COLLISION";
        }
        record_rejection(target_pose, closing, approach, opening, reason.str());
        continue;
      }
      opening = accepted_opening;
      my_actions::msg::GraspCandidate candidate;
      candidate.id = static_cast<uint32_t>(response->candidates.size() + 1);
      candidate.grasp_pose.header = base_point.header; candidate.grasp_pose.header.frame_id = "base_link";
      // Publish the exact endpoint whose joints passed collision validation,
      // not the requested pose within IK tolerances. This keeps final-approach
      // strict IK, RViz and cached joint execution on one immutable endpoint.
      Eigen::Isometry3f published_grasp = target_pose;
      if (pure_hggd) {
        std::array<float, 6> validated_joints{};
        std::copy_n(ik.begin(), 6, validated_joints.begin());
        published_grasp = Robot::eefk(validated_joints);
      }
      candidate.grasp_pose.pose.position.x = published_grasp.translation().x();
      candidate.grasp_pose.pose.position.y = published_grasp.translation().y();
      candidate.grasp_pose.pose.position.z = published_grasp.translation().z();
      Eigen::Quaternionf q(published_grasp.rotation());
      candidate.grasp_pose.pose.orientation.x = q.x();
      candidate.grasp_pose.pose.orientation.y = q.y();
      candidate.grasp_pose.pose.orientation.z = q.z();
      candidate.grasp_pose.pose.orientation.w = q.w();
      candidate.pregrasp_pose = candidate.grasp_pose;
      candidate.pregrasp_pose.pose.position.x = accepted_pregrasp_pose.translation().x();
      candidate.pregrasp_pose.pose.position.y = accepted_pregrasp_pose.translation().y();
      candidate.pregrasp_pose.pose.position.z = accepted_pregrasp_pose.translation().z();
      const Eigen::Quaternionf pregrasp_q(accepted_pregrasp_pose.rotation());
      candidate.pregrasp_pose.pose.orientation.x = pregrasp_q.x();
      candidate.pregrasp_pose.pose.orientation.y = pregrasp_q.y();
      candidate.pregrasp_pose.pose.orientation.z = pregrasp_q.z();
      candidate.pregrasp_pose.pose.orientation.w = pregrasp_q.w();
      candidate.contact_left.x = left.x(); candidate.contact_left.y = left.y(); candidate.contact_left.z = left.z(); candidate.contact_right.x = right.x(); candidate.contact_right.y = right.y(); candidate.contact_right.z = right.z();
      candidate.jaw_opening_m = opening; candidate.score = proposal.quality;
      candidate.affordance_family =
        selected_grasp_skill == LocalGraspSkill::kPinch ? "PINCH" :
        selected_grasp_skill == LocalGraspSkill::kPowerGrasp ?
          "POWER_GRASP" : "LOCAL_GENERIC";
      const float vertical_closing_penalty = std::clamp(
        std::abs(closing.z()), 0.0F, 1.0F);
      const Eigen::Vector3f ranking_seed = pure_hggd ?
        proposal.sample : seed;
      const float seed_distance_penalty = std::clamp(
        (contact_midpoint - ranking_seed).norm() /
          std::max(0.001F, static_cast<float>(grasp_target_sample_radius_m_)),
        0.0F, 1.0F);
      const float vertical_approach_penalty = std::clamp(
        std::abs(approach.z()), 0.0F, 1.0F);
      float camera_approach_penalty = 0.0F;
      const float camera_upright_penalty = pure_hggd ?
        0.5F * (1.0F - std::clamp(
          accepted_camera_upright_score, -1.0F, 1.0F)) : 0.0F;
      // For a side-wall power grasp, a gravity-level jaw line is the natural
      // solution.  Keep tilted solutions as reachable fallbacks, but rank
      // them below an otherwise comparable level grasp instead of treating a
      // solver-induced roll as an equally desirable second candidate.
      const float side_roll_tilt_penalty =
        proposal.side_surface_grasp ? 0.5F * std::clamp(
          std::abs(closing.z()), 0.0F, 1.0F) : 0.0F;
      if (refine_hggd_geometry) {
        Eigen::Vector3f frontal = contact_midpoint - camera_origin;
        frontal -= closing * frontal.dot(closing);
        if (frontal.squaredNorm() > 1.0e-6F) {
          frontal.normalize();
          camera_approach_penalty = 1.0F - std::clamp(
            approach.dot(frontal), 0.0F, 1.0F);
        } else {
          camera_approach_penalty = 1.0F;
        }
      }
      float joint_travel_sq = 0.0F;
      float joint_margin_reward = 1.0F;
      constexpr const auto &lower_limits = kPiperJointLower;
      constexpr const auto &upper_limits = kPiperJointUpper;
      for (size_t index = 0; index < 6; ++index) {
        // The first commanded destination is the pre-grasp, so rank its
        // required joint displacement rather than the later contact pose.
        // This is the inexpensive reachability proxy used when full path
        // preplanning is disabled.
        const float delta = pregrasp_ik[index] - joints[index];
        joint_travel_sq += delta * delta;
        const float range = upper_limits[index] - lower_limits[index];
        const float normalized_margin = 2.0F * std::min(
          (ik[index] - lower_limits[index]) / range,
          (upper_limits[index] - ik[index]) / range);
        joint_margin_reward = std::min(
          joint_margin_reward, std::clamp(normalized_margin, 0.0F, 1.0F));
      }
      const float joint_travel_penalty = std::clamp(
        std::sqrt(joint_travel_sq / 6.0F) / 3.14159265358979323846F,
        0.0F, 1.0F);
      // The normalized proposal-quality component and final ranking score are filled after
      // the ranking pool is complete. Width is also normalized over that
      // valid pool, so no task-specific opening threshold rejects a pose.
      candidate.score_components = {
        proposal.quality, 0.0F, 0.0F,
        vertical_closing_penalty, seed_distance_penalty,
        vertical_approach_penalty, joint_travel_penalty,
        joint_margin_reward,
        accepted_pregrasp_collision_free ? 0.0F : 1.0F,
        corner_penalty + orientation.roll_alignment_cost +
          camera_upright_penalty + side_roll_tilt_penalty,
        camera_approach_penalty};
      float variant_selection_cost =
        orientation.roll_alignment_cost + camera_upright_penalty +
        0.20F * joint_travel_penalty - 0.10F * joint_margin_reward;
      if (proposal.rim_grasp) {
        // Prefer a gravity-top-down approach and a level radial jaw line.
        // IK/collision safety remain hard requirements; this only chooses
        // among variants of the same spatial HGGD seed.
        variant_selection_cost += std::acos(std::clamp(
          approach.dot(-Eigen::Vector3f::UnitZ()), -1.0F, 1.0F));
        variant_selection_cost += 0.75F * std::abs(closing.z());
      } else if (proposal.side_surface_grasp) {
        variant_selection_cost += 1.25F * std::abs(closing.z());
        variant_selection_cost += 0.25F * camera_approach_penalty;
      } else {
        variant_selection_cost += 0.50F * std::abs(closing.z());
        variant_selection_cost += 0.25F * camera_approach_penalty;
      }
      candidate.ik_valid = true;
      // Contact reconstruction is diagnostic metadata only.  Do not claim a
      // measured-contact/sweep validation unless fingerVolumesAndClosingSweepClear
      // has actually run for this exact, unmodified pose.
      candidate.measured_contacts_valid = measured_sweep_valid;
      candidate.silhouette_contacts_valid = measured_sweep_valid &&
        measured_contacts.silhouette_supported;
      candidate.collision_checked = enable_grasp_collision_check_;
      candidate.grasp_collision_valid = enable_grasp_collision_check_;
      candidate.pregrasp_collision_valid =
        enable_grasp_collision_check_ && accepted_pregrasp_collision_free;
      candidate.pregrasp_collision_overridden =
        enable_grasp_collision_check_ && !accepted_pregrasp_collision_free;
      candidate.collision_valid = candidate.grasp_collision_valid &&
        candidate.pregrasp_collision_valid;
      if (!enable_grasp_collision_check_) {
        candidate.validation_status = pure_hggd ?
          (candidate.affordance_family + "_IK_ONLY") :
          (measured_contacts.body_model_supported ?
            "CUP_BODY_MODEL_IK_ONLY" : "IK_ONLY");
      } else if (candidate.pregrasp_collision_overridden) {
        candidate.validation_status = measured_contacts.body_model_supported ?
          "CUP_BODY_MODEL_GRASP_SAFE_PREGRASP_CONTACT_OVERRIDE" :
          "GRASP_SAFE_PREGRASP_CONTACT_OVERRIDE";
      } else {
        candidate.validation_status = measured_contacts.body_model_supported ?
          "CUP_BODY_MODEL_GRASP_AND_PREGRASP_SAFE" :
          "GRASP_AND_PREGRASP_SAFE";
      }
      if (!accepted_alignment_status.empty()) {
        candidate.validation_status += ";" + accepted_alignment_status;
      }
      std::copy_n(ik.begin(), candidate.grasp_joints.size(), candidate.grasp_joints.begin());
      std::copy_n(
        pregrasp_ik.begin(), candidate.pregrasp_joints.size(),
        candidate.pregrasp_joints.begin());
      if (angle_trials_by_grasp.size()<64) {
        angle_trials_by_grasp.emplace_back(candidate.grasp_joints,selected_angle_trials);
      }
      if (pure_hggd) {
        const auto same_seed = std::find_if(
          response->candidates.begin(), response->candidates.end(),
          [&](const my_actions::msg::GraspCandidate &existing) {
            const auto seed = hggd_candidate_seed_indices.find(existing.id);
            return seed != hggd_candidate_seed_indices.end() &&
              seed->second == proposal_index;
          });
        if (same_seed != response->candidates.end()) {
          const uint32_t retained_id = same_seed->id;
          const auto retained_cost =
            hggd_candidate_variant_costs.find(retained_id);
          if (retained_cost == hggd_candidate_variant_costs.end() ||
              variant_selection_cost < retained_cost->second) {
            candidate.id = retained_id;
            *same_seed = candidate;
            hggd_candidate_variant_costs[retained_id] =
              variant_selection_cost;
            seed_diagnostics.terminal_reason = "BEST_SAFE_VARIANT_UPDATED";
          }
          continue;
        }
        const Eigen::Vector3f candidate_approach =
          q * Eigen::Vector3f::UnitZ();
        const Eigen::Vector3f candidate_closing =
          q * Eigen::Vector3f::UnitY();
        const bool duplicate = std::any_of(
          response->candidates.begin(), response->candidates.end(),
          [&](const my_actions::msg::GraspCandidate &existing) {
            const Eigen::Vector3f existing_center(
              static_cast<float>(0.5 *
                (existing.contact_left.x + existing.contact_right.x)),
              static_cast<float>(0.5 *
                (existing.contact_left.y + existing.contact_right.y)),
              static_cast<float>(0.5 *
                (existing.contact_left.z + existing.contact_right.z)));
            const Eigen::Quaternionf existing_orientation(
              static_cast<float>(existing.grasp_pose.pose.orientation.w),
              static_cast<float>(existing.grasp_pose.pose.orientation.x),
              static_cast<float>(existing.grasp_pose.pose.orientation.y),
              static_cast<float>(existing.grasp_pose.pose.orientation.z));
            return gpdPosesAreDuplicates(
              contact_midpoint, candidate_approach, candidate_closing,
              existing_center,
              existing_orientation * Eigen::Vector3f::UnitZ(),
              existing_orientation * Eigen::Vector3f::UnitY(),
              static_cast<float>(hggd_diversity_position_m_),
              hggd_diversity_approach_degrees_,
              hggd_diversity_closing_degrees_);
          });
        if (duplicate) {
          ++hggd_duplicate_valid_rejected;
          seed_diagnostics.terminal_reason = "DUPLICATE_VALID_POSE";
          continue;
        }
        hggd_candidate_seed_indices[candidate.id] = proposal_index;
        hggd_candidate_variant_costs[candidate.id] =
          variant_selection_cost;
      }
      response->candidates.push_back(candidate);
      ++seed_diagnostics.accepted;
      if (candidate.affordance_family == "PINCH") {
        ++pinch_candidates_accepted;
      } else if (candidate.affordance_family == "POWER_GRASP") {
        ++power_candidates_accepted;
      }
      switch (proposal.cloud_origin) {
        case CloudProposalOrigin::kLocalPca:
          candidate_origins[candidate.id] = "local_pca";
          break;
        case CloudProposalOrigin::kGlobalPca:
          candidate_origins[candidate.id] = "global_pca";
          break;
      }
      if (pure_hggd) {
        // Continue through the bounded orientations. If another safe pose
        // exists for this spatial seed, the topology-aware cost above keeps
        // only the best one, so the ranking pool remains spatially diverse.
        seed_diagnostics.terminal_reason = "BEST_SAFE_VARIANT";
      }
    }
    }

    if (pure_hggd) {
      const auto skill_name = [](LocalGraspSkill skill) -> const char * {
          switch (skill) {
            case LocalGraspSkill::kPinch:
              return "PINCH";
            case LocalGraspSkill::kPowerGrasp:
              return "POWER_GRASP";
            case LocalGraspSkill::kUnknown:
            default:
              return "UNKNOWN";
          }
        };
      for (size_t index = 0; index < proposals.size(); ++index) {
        const auto &proposal = proposals[index];
        const auto &diagnostics = hggd_seed_diagnostics[index];
        std::string result = "NO_ACCEPTED_VARIANT";
        if (diagnostics.accepted > 0) {
          result = "ACCEPTED";
        } else if (!diagnostics.terminal_reason.empty()) {
          result = diagnostics.terminal_reason;
        } else if (diagnostics.ranking_pool_full) {
          result = "SKIPPED_RANKING_POOL_FULL";
        } else if (diagnostics.ik_budget_exhausted && diagnostics.ik_submitted == 0) {
          result = "SKIPPED_IK_BUDGET";
        } else if (diagnostics.ik_budget_exhausted) {
          result = "PARTIAL_IK_BUDGET";
        } else if (diagnostics.ik_submitted == 0) {
          result = "GEOMETRY_REJECTED";
        }
        const float best_grasp_position_mm =
          std::isfinite(diagnostics.best_grasp_ik.best_position_error) ?
          diagnostics.best_grasp_ik.best_position_error * 1000.0F : -1.0F;
        const float best_grasp_rotation_deg =
          std::isfinite(diagnostics.best_grasp_ik.best_rotation_error) ?
          diagnostics.best_grasp_ik.best_rotation_error *
          180.0F / 3.14159265358979323846F : -1.0F;
        const float best_pregrasp_position_mm =
          std::isfinite(diagnostics.best_pregrasp_ik.best_position_error) ?
          diagnostics.best_pregrasp_ik.best_position_error * 1000.0F : -1.0F;
        const float best_pregrasp_rotation_deg =
          std::isfinite(diagnostics.best_pregrasp_ik.best_rotation_error) ?
          diagnostics.best_pregrasp_ik.best_rotation_error *
          180.0F / 3.14159265358979323846F : -1.0F;
        RCLCPP_INFO(
          this->get_logger(),
          "[HGGD SEED #%zu] family=%zu center=(%.3f,%.3f,%.3f)m quality=%.3f "
          "skill=%s result=%s variants={seen:%d,degenerate:%d,cross_section:%d,sweep:%d} "
          "IK={submitted:%d,grasp_fail:%d,pregrasp_fail:%d,pregrasp_not_visible:%d,"
          "best_grasp:[%.1fmm,%.1fdeg],"
          "best_pregrasp:[%.1fmm,%.1fdeg]} collision={grasp:%d,pregrasp:%d} accepted=%d",
          index + 1, proposal.family_index, proposal.sample.x(), proposal.sample.y(),
          proposal.sample.z(), proposal.quality, skill_name(proposal.grasp_skill),
          result.c_str(), diagnostics.orientations_considered,
          diagnostics.degenerate_orientations, diagnostics.cross_section_rejected,
          diagnostics.sweep_rejected, diagnostics.ik_submitted,
          diagnostics.grasp_ik_rejected, diagnostics.pregrasp_ik_rejected,
          diagnostics.pregrasp_visibility_rejected,
          best_grasp_position_mm, best_grasp_rotation_deg,
          best_pregrasp_position_mm, best_pregrasp_rotation_deg,
          diagnostics.grasp_collision_rejected,
          diagnostics.pregrasp_collision_rejected, diagnostics.accepted);
      }
    }

    // These poses have completed local contact refinement, strict grasp IK,
    // pre-grasp IK, and the enabled collision checks. Diversity is applied
    // below before publication so RViz does not present near-identical poses
    // as independent choices.
    RCLCPP_INFO(this->get_logger(),
      "[GRASP VALID POOL] before_ranking=%zu cap=%d duplicate_valid_rejected=%d",
      response->candidates.size(), ranking_pool_limit,
      hggd_duplicate_valid_rejected);
    int diversity_rejected = 0;
    if (!response->candidates.empty()) {
      ranking_pool_candidate_count = response->candidates.size();
      const auto [minimum_score, maximum_score] = std::minmax_element(
        response->candidates.begin(), response->candidates.end(),
        [](const auto &left, const auto &right) {return left.score < right.score;});
      const auto [minimum_opening, maximum_opening] = std::minmax_element(
        response->candidates.begin(), response->candidates.end(),
        [](const auto &left, const auto &right) {
          return left.jaw_opening_m < right.jaw_opening_m;
        });
      const float score_span = maximum_score->score - minimum_score->score;
      const float minimum_opening_m = minimum_opening->jaw_opening_m;
      const float maximum_opening_m = maximum_opening->jaw_opening_m;
      const float opening_span = maximum_opening_m - minimum_opening_m;
      for (auto &candidate : response->candidates) {
        const float proposal_quality_normalized = score_span > 1e-6F ?
          (candidate.score - minimum_score->score) / score_span : 1.0F;

        const float relative_opening_penalty = opening_span > 1e-6F ?
          (candidate.jaw_opening_m - minimum_opening_m) /
            opening_span : 0.0F;
        candidate.score_components[1] = proposal_quality_normalized;
        candidate.score_components[2] = std::clamp(
          relative_opening_penalty, 0.0F, 1.0F);
        candidate.ranking_score = static_cast<float>(
          natural_proposal_quality_weight_ * proposal_quality_normalized -
          natural_wide_opening_weight_ * candidate.score_components[2] -
          natural_vertical_closing_weight_ * candidate.score_components[3] -
          natural_seed_distance_weight_ * candidate.score_components[4] -
          natural_vertical_approach_weight_ * candidate.score_components[5] -
          natural_joint_travel_weight_ * candidate.score_components[6] +
          natural_joint_margin_weight_ * candidate.score_components[7] -
          natural_pregrasp_override_weight_ * candidate.score_components[8] -
          natural_corner_penalty_weight_ * candidate.score_components[9] -
          natural_camera_approach_weight_ * candidate.score_components[10]);
      }
      std::stable_sort(
        response->candidates.begin(), response->candidates.end(),
        [](const auto &left, const auto &right) {
          return left.ranking_score > right.ranking_score;
        });
      const auto valid_pool_for_log = response->candidates;
      if (pure_learned) {
        std::vector<my_actions::msg::GraspCandidate> diverse_candidates;
        diverse_candidates.reserve(static_cast<size_t>(maximum_valid_candidates_));
        for (const auto &candidate : response->candidates) {
          const Eigen::Vector3f center(
            static_cast<float>((candidate.contact_left.x + candidate.contact_right.x) * 0.5),
            static_cast<float>((candidate.contact_left.y + candidate.contact_right.y) * 0.5),
            static_cast<float>((candidate.contact_left.z + candidate.contact_right.z) * 0.5));
          const Eigen::Quaternionf orientation(
            static_cast<float>(candidate.grasp_pose.pose.orientation.w),
            static_cast<float>(candidate.grasp_pose.pose.orientation.x),
            static_cast<float>(candidate.grasp_pose.pose.orientation.y),
            static_cast<float>(candidate.grasp_pose.pose.orientation.z));
          const Eigen::Vector3f approach = orientation * Eigen::Vector3f::UnitZ();
          const Eigen::Vector3f closing = orientation * Eigen::Vector3f::UnitY();
          bool duplicate = false;
          for (const auto &selected : diverse_candidates) {
            const Eigen::Vector3f selected_center(
              static_cast<float>((selected.contact_left.x + selected.contact_right.x) * 0.5),
              static_cast<float>((selected.contact_left.y + selected.contact_right.y) * 0.5),
              static_cast<float>((selected.contact_left.z + selected.contact_right.z) * 0.5));
            const Eigen::Quaternionf selected_orientation(
              static_cast<float>(selected.grasp_pose.pose.orientation.w),
              static_cast<float>(selected.grasp_pose.pose.orientation.x),
              static_cast<float>(selected.grasp_pose.pose.orientation.y),
              static_cast<float>(selected.grasp_pose.pose.orientation.z));
            const Eigen::Vector3f selected_approach =
              selected_orientation * Eigen::Vector3f::UnitZ();
            const Eigen::Vector3f selected_closing =
              selected_orientation * Eigen::Vector3f::UnitY();
            duplicate = gpdPosesAreDuplicates(
              center, approach, closing, selected_center, selected_approach,
              selected_closing,
              static_cast<float>(hggd_diversity_position_m_),
              hggd_diversity_approach_degrees_,
              hggd_diversity_closing_degrees_);
            if (duplicate) break;
          }
          if (duplicate) {
            ++diversity_rejected;
            continue;
          }
          diverse_candidates.push_back(candidate);
          if (diverse_candidates.size() >= static_cast<size_t>(maximum_valid_candidates_)) break;
        }
        RCLCPP_INFO(
          this->get_logger(),
          "[%s DIVERSITY] feasible_pool=%zu diversity_rejected=%d returned=%zu "
          "position=%.3fm approach=%.1fdeg closing=%.1fdeg",
          "HGGD", response->candidates.size(),
          diversity_rejected, diverse_candidates.size(),
          hggd_diversity_position_m_,
          hggd_diversity_approach_degrees_,
          hggd_diversity_closing_degrees_);
        response->candidates = std::move(diverse_candidates);
      } else if (response->candidates.size() > static_cast<size_t>(maximum_valid_candidates_)) {
        response->candidates.resize(static_cast<size_t>(maximum_valid_candidates_));
      }
      if (pure_hggd && enable_grasp_path_preplanning_ &&
          !response->candidates.empty()) {
        struct PreplannedCandidate {
          my_actions::msg::GraspCandidate candidate;
          float path_motion_cost = std::numeric_limits<float>::infinity();
          double planning_ms = 0.0;
          size_t waypoints = 0;
        };
        const auto preplanning_started = std::chrono::steady_clock::now();
        std::vector<PreplannedCandidate> feasible;
        feasible.reserve(response->candidates.size());
        for (const auto &candidate : response->candidates) {
          std::vector<float> goal(
            candidate.pregrasp_joints.begin(),
            candidate.pregrasp_joints.end());
          const auto candidate_started = std::chrono::steady_clock::now();
          std::vector<float> path;
          {
            // Keep the obstacle snapshot fixed for the complete RRTC query.
            // This performs planning only; it never publishes motor commands.
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            geometry_msgs::msg::Point unused_contact_center;
            path = replanRRTC(
              joints, goal, false, unused_contact_center, 0.0F, true);
          }
          const double planning_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - candidate_started).count();
          if (path.empty()) {
            RCLCPP_INFO(
              this->get_logger(),
              "[GRASP PREPLAN V%u] reachable=0 planning=%.1fms",
              candidate.id, planning_ms);
            continue;
          }
          float motion_cost = 0.0F;
          std::array<float, 6> previous{};
          std::copy_n(joints.begin(), 6, previous.begin());
          const size_t waypoints = path.size() / 6;
          for (size_t waypoint = 0; waypoint < waypoints; ++waypoint) {
            float synchronized_step = 0.0F;
            for (size_t joint = 0; joint < 6; ++joint) {
              const float value = path[waypoint * 6 + joint];
              synchronized_step = std::max(
                synchronized_step, std::abs(value - previous[joint]));
              previous[joint] = value;
            }
            motion_cost += synchronized_step;
          }
          feasible.push_back({candidate, motion_cost, planning_ms, waypoints});
          RCLCPP_INFO(
            this->get_logger(),
            "[GRASP PREPLAN V%u] reachable=1 planning=%.1fms "
            "waypoints=%zu motion_cost=%.3frad",
            candidate.id, planning_ms, waypoints, motion_cost);
        }
        if (feasible.empty()) {
          response->candidates.clear();
          response->success = false;
          response->status = "NO_PATH_REACHABLE_PREGRASP";
          publishGraspMarkers(
            response->candidates, valid_grasp_markers_pub_,
            "valid_grasp_candidate_", true);
          RCLCPP_WARN(
            this->get_logger(),
            "[GRASP PREPLAN] all %zu collision-safe endpoint candidates "
            "failed full-path planning",
            ranking_pool_candidate_count);
          return;
        }
        const auto [minimum_path, maximum_path] = std::minmax_element(
          feasible.begin(), feasible.end(),
          [](const auto &left, const auto &right) {
            return left.path_motion_cost < right.path_motion_cost;
          });
        const float path_span =
          maximum_path->path_motion_cost - minimum_path->path_motion_cost;
        for (auto &planned : feasible) {
          const float normalized_path_cost = path_span > 1.0e-6F ?
            (planned.path_motion_cost - minimum_path->path_motion_cost) /
              path_span : 0.0F;
          planned.candidate.ranking_score -= static_cast<float>(
            grasp_path_length_weight_) * normalized_path_cost;
        }
        std::stable_sort(
          feasible.begin(), feasible.end(),
          [](const auto &left, const auto &right) {
            return left.candidate.ranking_score >
              right.candidate.ranking_score;
          });
        response->candidates.clear();
        response->candidates.reserve(feasible.size());
        for (auto &planned : feasible) {
          response->candidates.push_back(std::move(planned.candidate));
        }
        const double total_preplanning_ms =
          std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - preplanning_started).count();
        RCLCPP_INFO(
          this->get_logger(),
          "[GRASP PREPLAN] tested=%zu reachable=%zu total=%.1fms "
          "path_weight=%.2f selected=V%u",
          ranking_pool_candidate_count, response->candidates.size(),
          total_preplanning_ms, grasp_path_length_weight_,
          response->candidates.front().id);
      }
      publishGraspMarkers(
        response->candidates, valid_grasp_markers_pub_,
        "valid_grasp_candidate_", true);
      if (pure_hggd) {
        // Preserve ranked, validated alternatives through the shared API.
        // The prepared session selects the first complete motion solution.
        std::vector<int> retained_by_seed(proposals.size(), 0);
        for (const auto &candidate : response->candidates) {
          const auto seed = hggd_candidate_seed_indices.find(candidate.id);
          if (seed != hggd_candidate_seed_indices.end() &&
              seed->second < retained_by_seed.size()) {
            ++retained_by_seed[seed->second];
          }
        }
        std::ostringstream retained;
        retained << "[HGGD FINAL] retained_after_ranking_and_diversity={";
        for (size_t index = 0; index < retained_by_seed.size(); ++index) {
          if (index > 0) retained << ",";
          retained << "#" << index + 1 << ":" << retained_by_seed[index];
        }
        retained << "}";
        RCLCPP_INFO(this->get_logger(), "%s", retained.str().c_str());
      }
      const uint32_t selected_valid_pool_id = response->candidates.front().id;
      if (pure_hggd) {
        const auto selected_seed =
          hggd_candidate_seed_indices.find(selected_valid_pool_id);
        if (selected_seed != hggd_candidate_seed_indices.end() &&
            selected_seed->second < proposals.size()) {
          const auto &sample = proposals[selected_seed->second].sample;
          response->corrected_seed_base.x = sample.x();
          response->corrected_seed_base.y = sample.y();
          response->corrected_seed_base.z = sample.z();
        }
      }
      for (size_t index = 0; index < response->candidates.size(); ++index) {
        response->candidates[index].id = static_cast<uint32_t>(index + 1);
      }
      const auto &best = response->candidates.front();
      const Eigen::Vector3f best_grasp_position(
        static_cast<float>(best.grasp_pose.pose.position.x),
        static_cast<float>(best.grasp_pose.pose.position.y),
        static_cast<float>(best.grasp_pose.pose.position.z));
      const Eigen::Vector3f best_pregrasp_position(
        static_cast<float>(best.pregrasp_pose.pose.position.x),
        static_cast<float>(best.pregrasp_pose.pose.position.y),
        static_cast<float>(best.pregrasp_pose.pose.position.z));
      RCLCPP_INFO(
        this->get_logger(),
        "[GRASP RANK] Selected #1 from_valid_pool=V%u ranking=%.3f proposal-quality=%.3f opening=%.3fm "
        "pregrasp-retreat=%.3fm "
        "relative-width=%.2f pool-width=[%.3f,%.3f]m "
        "vertical-close=%.2f seed=%.2f approach=%.2f "
        "camera-approach=%.2f joint-travel=%.2f joint-margin=%.2f status=%s",
        selected_valid_pool_id, best.ranking_score, best.score, best.jaw_opening_m,
        (best_pregrasp_position - best_grasp_position).norm(),
        best.score_components[2], minimum_opening_m,
        maximum_opening_m, best.score_components[3],
        best.score_components[4], best.score_components[5],
        best.score_components[10], best.score_components[6],
        best.score_components[7],
        best.validation_status.c_str());
      const auto angle_start = best.validation_status.find("ANGLE_SELECTION=");
      const auto angle_summary = angle_start == std::string::npos ?
        std::string("ANGLE_SELECTION=NOT_APPLIED") :
        best.validation_status.substr(angle_start,
          best.validation_status.find(';', angle_start)-angle_start);
      RCLCPP_INFO(this->get_logger(), "[GRASP ANGLE SELECTED] candidate=#1 %s",
        angle_summary.c_str());
      const Eigen::Vector3f best_contact_center(
        static_cast<float>(0.5 *
          (best.contact_left.x + best.contact_right.x)),
        static_cast<float>(0.5 *
          (best.contact_left.y + best.contact_right.y)),
        static_cast<float>(0.5 *
          (best.contact_left.z + best.contact_right.z)));
      auto contact_region = LocalGraspRefiner::extractContactRegion(
        target_samples, best_contact_center,
        refinement_contact_region_radius_m_);
      const std::size_t cached_contact_points = contact_region.size();
      {
        std::lock_guard<std::mutex> lock(refinement_target_mutex_);
        // Cache only the 10 cm-diameter contact neighborhood, not the entire
        // object cloud. Both observations are already in base_link through
        // joint FK * calibrated X, so no TF or additional transform is used.
        refinement_contact_region_ = std::move(contact_region);
        refinement_target_stamp_ns_ = cloud_stamp.nanoseconds();
      }
      RCLCPP_INFO(
        this->get_logger(),
        "[CONTACT REGION] cached=%zu radius=%.1fmm frame=base_link",
        cached_contact_points,
        refinement_contact_region_radius_m_ * 1000.0F);
      if (retained_pregrasp_overrides > 0) {
        RCLCPP_WARN(
          this->get_logger(),
          "[VAMP GRASP] VAMP found collision at the pre-grasp pose for %d "
          "ranking-pool candidate(s); they were retained with an override "
          "penalty and can use only the bounded target-contact path.",
          retained_pregrasp_overrides);
      }
    } else {
    }
    publishGraspMarkers(response->candidates);
    // Keep a bounded sample of feasibility failures visible alongside the
    // final pool; raw GPD proposals use their dedicated marker topic.
    publishRejectedGraspMarkers(rejected_debug);
    response->success = !response->candidates.empty();
    if (response->success) {
      if (enable_grasp_collision_check_) {
        response->status = response->candidates.size() < static_cast<size_t>(maximum_valid_candidates_) ?
          "FEWER_THAN_MAXIMUM_VALID" : "OK";
      } else {
        response->status = response->candidates.size() < static_cast<size_t>(maximum_valid_candidates_) ?
          "FEWER_THAN_MAXIMUM_IK_ONLY" : "OK_IK_ONLY";
      }
    } else {
      std::ostringstream status;
      status << "NO_VALID_GEOMETRIC_GRASP"
             << " backend=" << (known_cloud ? "known_cloud" : grasp_candidate_backend_) << " proposals=" << proposals.size()
             << " score=" << rejected_score
             << " antipodal=" << rejected_antipodal
             << " width=" << rejected_width
             << " contacts=" << rejected_contacts
             << " finger_sweep=" << rejected_finger_sweep
             << " grasp_ik=" << rejected_ik << " pregrasp_ik=" << rejected_pregrasp_ik
             << " pregrasp_not_visible=" << rejected_pregrasp_visibility
             << " grasp_collision=" << rejected_grasp_collision
             << " pregrasp_collision=" << rejected_pregrasp_collision;
      response->status = status.str();
    }

    const auto computation_finished = std::chrono::steady_clock::now();

    if (pure_hggd) {
      RCLCPP_INFO(
        this->get_logger(),
        "[HGGD SUMMARY] target_pixels=%zu heatmap_centers=%zu local_groups=%zu "
        "decoded=%zu geometry={aligned:%d,rejected:%d,orientations:%d} submitted=%zu "
        "seed_prepare={tested:%d,aligned:%d,geometry_rejected:%d,time:%.1fms} "
        "skills={pinch:%d/%d,power:%d/%d} accepted=%zu rejected={width:%d,grasp_ik:%d,"
        "pregrasp_ik:%d,pregrasp_not_visible:%d,grasp_collision:%d,pregrasp_collision:%d} "
        "network={anchor:%.1fms,local:%.1fms} validation=%.1fms total=%.1fms "
        "heatmap=%s axis_fallback_seeds=%d orbit_variants=%d "
        "pregrasp_envelope_rejected=%d",
        hggd_diagnostics.target_pixels, hggd_diagnostics.heatmap_centers,
        hggd_diagnostics.valid_local_groups, hggd_diagnostics.network_grasps,
        hggd_aligned_centers, hggd_geometry_rejected,
        geometry_approach_variants_generated, proposals.size(),
        hggd_direct_seeds_tested, hggd_aligned_centers,
        hggd_geometry_rejected, hggd_seed_selection_ms,
        pinch_candidates_accepted, pinch_candidates_generated,
        power_candidates_accepted, power_candidates_generated,
        response->candidates.size(), rejected_width,
        rejected_ik, rejected_pregrasp_ik, rejected_pregrasp_visibility,
        rejected_grasp_collision,
        rejected_pregrasp_collision, hggd_diagnostics.backbone_ms,
        hggd_diagnostics.local_ms,
        std::chrono::duration<double, std::milli>(
          computation_finished - proposal_finished).count(),
        std::chrono::duration<double, std::milli>(
          computation_finished - computation_started).count(),
        hggd_heatmap_output.empty() ? "<not-written>" : hggd_heatmap_output.c_str(),
        hggd_contact_axis_fallback_seeds, hggd_orbit_variants_generated,
        rejected_pregrasp_envelope);
    }

    RCLCPP_INFO(this->get_logger(),
      "[GRASP VALIDATION] backend=%s generated=%zu tested=%d accepted=%zu "
      "rejected={quality:%d,width:%d,antipodal:%d,contacts:%d,sweep:%d,"
      "grasp_ik:%d,pregrasp_ik:%d,pregrasp_envelope:%d,pregrasp_not_visible:%d,"
      "grasp_collision:%d,pregrasp_collision:%d} "
      "pregrasp={retreat_range:[0.050,0.195]m,preferred:%.3fm,"
      "orientation_tolerance:%.1fdeg} "
      "validation=%.1fms total=%.1fms",
      grasp_candidate_backend_.c_str(), proposals.size(), tested,
      response->candidates.size(), rejected_score, rejected_width,
      rejected_antipodal, rejected_contacts, rejected_finger_sweep,
      rejected_ik, rejected_pregrasp_ik, rejected_pregrasp_envelope,
      rejected_pregrasp_visibility,
      rejected_grasp_collision,
      rejected_pregrasp_collision,
      pregrasp_offset_, pregrasp_orientation_tolerance_degrees_,
      std::chrono::duration<double, std::milli>(
        computation_finished - proposal_finished).count(),
      std::chrono::duration<double, std::milli>(
        computation_finished - computation_started).count());

    if (!contact_rejection_reasons.empty()) {
      std::ostringstream reasons;
      reasons << "[CONTACT FILTER]";
      for (const auto &entry : contact_rejection_reasons) {
        reasons << " " << entry.first << "=" << entry.second;
      }
      RCLCPP_INFO(this->get_logger(), "%s", reasons.str().c_str());
    }
    if (have_target_sweep_block) {
      RCLCPP_INFO(
        this->get_logger(),
        "[SWEEP FIRST BLOCK] source=%s opening=%.1fmm side=%.1fmm "
        "height=%.1fmm depth=%.1fmm",
        first_target_sweep_silhouette ? "silhouette" : "surface_normals",
        first_target_sweep_opening * 1000.0F,
        first_target_sweep_block[0] * 1000.0F,
        first_target_sweep_block[1] * 1000.0F,
        first_target_sweep_block[2] * 1000.0F);
    }
  }

  void mapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    obstacle_cloud_.clear();

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      if (!std::isnan(*iter_x) && !std::isnan(*iter_y) && !std::isnan(*iter_z)) {
        obstacle_cloud_.push_back({*iter_x, *iter_y, *iter_z});
      }
    }
  }

  void handleClearMap(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    (void)request;
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    obstacle_cloud_.clear();
    response->success = true;
    response->message = "Cleared VAMP internal obstacle map.";
    RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
  }

  void handleToggleNav(
    const std::shared_ptr<my_actions::srv::ToggleContinuousNavigation::Request> request,
    std::shared_ptr<my_actions::srv::ToggleContinuousNavigation::Response> response)
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    if (request->target_joints.size() == 6) {
      continuous_target_joints_ = request->target_joints;
      continuous_nav_active_ = true;
      response->success = true;
      RCLCPP_INFO(this->get_logger(), "Continuous navigation target updated.");
    } else if (request->target_joints.empty()) {
      continuous_nav_active_ = false;
      response->success = true;
      RCLCPP_INFO(this->get_logger(), "Continuous navigation disabled.");
    } else {
      response->success = false;
      RCLCPP_ERROR(this->get_logger(), "Invalid target_joints size (expected 6, got %zu)", request->target_joints.size());
    }
  }

  void handleCalculateFallbackPose(
    const std::shared_ptr<my_actions::srv::CalculateFallbackPose::Request> request,
    std::shared_ptr<my_actions::srv::CalculateFallbackPose::Response> response)
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    response->success = false;
    
    if (request->current_joints.size() != 6) {
      RCLCPP_ERROR(this->get_logger(), "Fallback check failed: current_joints != 6");
      return;
    }
    
    Point obs = {request->obstacle_point_base_link[0], request->obstacle_point_base_link[1], request->obstacle_point_base_link[2]};
    Point tcp = {request->tcp_point_base_link[0], request->tcp_point_base_link[1], request->tcp_point_base_link[2]};
    std::vector<float> curr_joints = request->current_joints;
    
    // Vector from Obstacle to TCP
    Point vec = {tcp[0] - obs[0], tcp[1] - obs[1], tcp[2] - obs[2]};
    float dist = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    
    RCLCPP_INFO(this->get_logger(), "  Distance Obs to TCP: %.3fm", dist);
    
    if (dist < 0.01f) {
      vec = {-1.0f, 0.0f, 0.0f};
    } else {
      vec[0] /= dist;
      vec[1] /= dist;
      vec[2] /= dist;
    }
    RCLCPP_INFO(this->get_logger(), "  Retreat Vector: [%.3f, %.3f, %.3f]", vec[0], vec[1], vec[2]);
    
    // Create Valid Env Voxels
    std::vector<Point> valid_env_voxels = getFilteredEnvironment(curr_joints);
    
    // Try distances 30cm, 32cm, ..., 40cm
    for (float d = 0.30f; d <= 0.41f; d += 0.02f) {
      Point target = {obs[0] + vec[0] * d, obs[1] + vec[1] * d, obs[2] + vec[2] * d};
      
      RCLCPP_INFO(this->get_logger(), "  [FALLBACK] Trying d=%.2fm -> target [%.3f, %.3f, %.3f]",
                  d, target[0], target[1], target[2]);
      
      std::vector<float> q_ik;
      // Allow up to 20cm error for fallback pose, keep joint limits during IK search, and ignore convergence
      bool success_ik = solvePiperIK(target, obs, curr_joints, q_ik, 0.04f, true, true);
      
      if (success_ik) {
        bool is_safe = isCandidatePoseSafe(q_ik, valid_env_voxels);
        RCLCPP_INFO(this->get_logger(), "  [FALLBACK]   IK solved: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f] -> collision_check=%s",
                    q_ik[0], q_ik[1], q_ik[2], q_ik[3], q_ik[4], q_ik[5],
                    is_safe ? "SAFE" : "COLLIDING");
        
        if (is_safe) {
          response->fallback_joints = q_ik;
          response->success = true;
          RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE] Found valid fallback pose at distance %.2fm", d);
          return;
        }
      } else {
        RCLCPP_INFO(this->get_logger(), "  [FALLBACK]   IK failed to converge for this target");
      }
    }
    
    RCLCPP_ERROR(this->get_logger(), "[VAMP ENGINE] Failed to find ANY valid fallback pose between 30cm and 40cm!");
  }

  void handleLookAtIK(const std::shared_ptr<my_actions::srv::LookAt::Request> request,
                      std::shared_ptr<my_actions::srv::LookAt::Response> response)
  {
    response->success = false;

    std::vector<float> curr_joints;
    Point current_pos = {request->current_pose_x, request->current_pose_y, request->current_pose_z};
    
    for (size_t i = 0; i < request->current_joints.size(); ++i) {
      curr_joints.push_back(request->current_joints[i]);
    }

    if (curr_joints.size() != 6) {
        RCLCPP_ERROR(this->get_logger(), "LookAtIK requires 6 current joints.");
        return;
    }

    Point look_target = {
      request->look_at_x,
      request->look_at_y,
      request->look_at_z
    };

    bool lock_base = request->lock_base;

    std::vector<float> q_ik;
    bool ik_success = false;

    if (request->use_tcp_pose) {
      const auto &pose = request->tcp_pose;
      Eigen::Quaternionf orientation(
        static_cast<float>(pose.orientation.w),
        static_cast<float>(pose.orientation.x),
        static_cast<float>(pose.orientation.y),
        static_cast<float>(pose.orientation.z));
      if (orientation.norm() < 1.0e-5F) {
        RCLCPP_ERROR(this->get_logger(), "LookAtIK received an invalid TCP quaternion.");
        return;
      }
      orientation.normalize();
      Eigen::Isometry3f desired = Eigen::Isometry3f::Identity();
      desired.linear() = orientation.toRotationMatrix();
      desired.translation() = Eigen::Vector3f(
        static_cast<float>(pose.position.x),
        static_cast<float>(pose.position.y),
        static_cast<float>(pose.position.z));
      PoseIkDiagnostics diagnostics;
      ik_success = solvePiperPoseIK(
        desired, curr_joints, q_ik, diagnostics);
    } else if (solvePiperIK(
        current_pos, look_target, curr_joints, q_ik, 0.02f, true, false,
        lock_base)) {
      // Backward-compatible position/look-vector mode.
      ik_success = true;
    } else {
        // Retry with a fixed seed bank to escape local minima (skip if lock_base is true).
        if (!lock_base) {
            const std::array<std::vector<float>, 3> fixed_seeds{{
              {0.0F, 0.8F, -1.4F, 0.0F, 0.0F, 0.0F},
              {0.6F, 1.4F, -2.1F, 0.0F, 0.0F, 0.0F},
              {-0.6F, 1.4F, -2.1F, 0.0F, 0.0F, 0.0F}}};
            for (const auto &fixed_seed : fixed_seeds) {
                if (solvePiperIK(current_pos, look_target, fixed_seed, q_ik, 0.02f, true, false, lock_base)) {
                    ik_success = true;
                    break;
                }
            }
        }
    }

    if (ik_success) {
      std::vector<Point> env_v;
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        env_v = obstacle_cloud_;
      }
      
      if (request->ignore_collision || isCandidatePoseSafe(q_ik, env_v)) {
        response->joint_angles = q_ik;
        response->success = true;
        RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE] LookAtIK solved and collision-free for target [%.3f, %.3f, %.3f]",
                    look_target[0], look_target[1], look_target[2]);
      } else {
        response->success = false;
        RCLCPP_WARN(this->get_logger(), "[VAMP ENGINE] LookAtIK solved but pose is IN COLLISION for target [%.3f, %.3f, %.3f]",
                    look_target[0], look_target[1], look_target[2]);
      }
    } else {
      RCLCPP_ERROR(this->get_logger(), "[VAMP ENGINE] LookAtIK failed to place camera at candidate [%.3f, %.3f, %.3f] to look at target [%.3f, %.3f, %.3f]",
                  current_pos[0], current_pos[1], current_pos[2],
                  look_target[0], look_target[1], look_target[2]);
    }
  }

  void handlePlanAPF(
    const std::shared_ptr<my_actions::srv::PlanAPF::Request> request,
    std::shared_ptr<my_actions::srv::PlanAPF::Response> response)
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    response->success = false;
    
    if (request->current_joints.size() != 6 || request->target_joints.size() != 6) {
      RCLCPP_ERROR(this->get_logger(), "PlanAPF: joints must have 6 elements");
      return;
    }
    
    std::vector<Point> valid_env_voxels = getFilteredEnvironment(request->current_joints);
    
    std::vector<float> curr_q = request->current_joints;
    std::vector<float> target_q = request->target_joints;
    
    float alpha = 1.0f; // Attractive step size
    float beta = 15.0f; // Repulsive step size
    float safe_dist = 0.12f;
    int max_iters = 1000;
    
    std::vector<float> path_flat;
    
    for (int iter = 0; iter < max_iters; ++iter) {
       for (size_t j = 0; j < 6; ++j) path_flat.push_back(curr_q[j]);
       
       float dist_to_target = 0;
       for (size_t j = 0; j < 6; ++j) dist_to_target += std::pow(target_q[j] - curr_q[j], 2);
       dist_to_target = std::sqrt(dist_to_target);
       
       if (dist_to_target < 0.05f) { // Reached target within 0.05 rad
           response->success = true;
           response->path_flat = path_flat;
           RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE] APF Local Planner reached target in %d steps.", iter);
           return;
       }
       
       std::array<float, 6> F_attract = {0};
       for (size_t j = 0; j < 6; ++j) F_attract[j] = alpha * (target_q[j] - curr_q[j]) / (dist_to_target + 1e-5f);
       
       std::array<float, 6> F_repulse = {0};
       float delta = 0.01f;
       
       Robot::ConfigurationBlock<1> c_curr;
       for (size_t j = 0; j < 6; ++j) c_curr[j][0] = curr_q[j];
       Robot::Spheres<1> out_curr;
       Robot::template sphere_fk<1>(c_curr, out_curr);
       
       float current_min_dist = 1e9f;
       for (size_t s = 0; s < Robot::n_spheres; ++s) {
          float sx = out_curr.x[{s, 0}], sy = out_curr.y[{s, 0}], sz = out_curr.z[{s, 0}], sr = out_curr.r[{s, 0}];
          for (const auto& pt : valid_env_voxels) {
             float dist = std::sqrt(std::pow(pt[0]-sx, 2) + std::pow(pt[1]-sy, 2) + std::pow(pt[2]-sz, 2)) - sr;
             if (dist < current_min_dist) current_min_dist = dist;
          }
       }
       
       if (current_min_dist < safe_dist) {
          for (size_t j = 0; j < 6; ++j) {
             float q_test[6];
             for (size_t k = 0; k < 6; ++k) q_test[k] = curr_q[k];
             q_test[j] += delta;
             
             Robot::ConfigurationBlock<1> c_test;
             for (size_t k = 0; k < 6; ++k) c_test[k][0] = q_test[k];
             Robot::Spheres<1> out_test;
             Robot::template sphere_fk<1>(c_test, out_test);
             
             float test_min_dist = 1e9f;
             for (size_t s = 0; s < Robot::n_spheres; ++s) {
                float sx = out_test.x[{s, 0}], sy = out_test.y[{s, 0}], sz = out_test.z[{s, 0}], sr = out_test.r[{s, 0}];
                for (const auto& pt : valid_env_voxels) {
                   float tdist = std::sqrt(std::pow(pt[0]-sx, 2) + std::pow(pt[1]-sy, 2) + std::pow(pt[2]-sz, 2)) - sr;
                   if (tdist < test_min_dist) test_min_dist = tdist;
                }
             }
             
             float grad = (test_min_dist - current_min_dist) / delta;
             float mag = (safe_dist - current_min_dist);
             F_repulse[j] = beta * mag * grad;
          }
       }
       
       float max_step = 0.05f; // Max step size per tick
       std::vector<float> q_next = curr_q;
       float step_len = 0;
       std::array<float, 6> F_total;
       for (size_t j = 0; j < 6; ++j) {
          F_total[j] = F_attract[j] + F_repulse[j];
          step_len += F_total[j]*F_total[j];
       }
       step_len = std::sqrt(step_len);
       if (step_len < 1e-4f) {
           RCLCPP_ERROR(this->get_logger(), "[VAMP ENGINE] APF Planner trapped in local minima!");
           return;
       }
       if (step_len > max_step) {
           for (size_t j = 0; j < 6; ++j) F_total[j] = F_total[j] * max_step / step_len;
       }
       
       const auto &lower_limits = kPiperJointLower;
       const auto &upper_limits = kPiperJointUpper;
       
       for (size_t j = 0; j < 6; ++j) {
           q_next[j] += F_total[j];
           if (q_next[j] < lower_limits[j] + 0.001f) q_next[j] = lower_limits[j] + 0.001f;
           if (q_next[j] > upper_limits[j] - 0.001f) q_next[j] = upper_limits[j] - 0.001f;
       }
       
       if (!isCandidatePoseSafe(q_next, valid_env_voxels)) {
           RCLCPP_ERROR(this->get_logger(), "[VAMP ENGINE] APF Planner proposed colliding step! Aborting.");
           return;
       }
       
       curr_q = q_next;
    }
    RCLCPP_ERROR(this->get_logger(), "[VAMP ENGINE] APF Planner failed to reach target in %d iterations.", max_iters);
  }

  std::vector<Point> getFilteredEnvironment(const std::vector<float>& current_joints)
  {
    Point origin = {0.0f, 0.0f, 0.0f};
    Point ws_min = {ws_x_min_, ws_y_min_, ws_z_min_};
    Point ws_max = {ws_x_max_, ws_y_max_, ws_z_max_};
    
    std::vector<Point> centervox_cloud = vamp::collision::filter_pointcloud_centervox(
        obstacle_cloud_, 0.04f, 10.0f, origin, ws_min, ws_max);
    
    Robot::ConfigurationBlock<1> c_start_filter;
    for (size_t j = 0; j < 6; ++j) {
      c_start_filter[j][0] = current_joints[j];
    }
    Robot::Spheres<1> out_start_filter;
    Robot::template sphere_fk<1>(c_start_filter, out_start_filter);

    std::vector<Point> valid_env_voxels;
    valid_env_voxels.reserve(centervox_cloud.size());

    for (const auto& pt : centervox_cloud) {
      bool is_self_collision = false;
      for (size_t s = 0; s < Robot::n_spheres; ++s) {
        float sx = out_start_filter.x[{s, 0}];
        float sy = out_start_filter.y[{s, 0}];
        float sz = out_start_filter.z[{s, 0}];
        float sr = out_start_filter.r[{s, 0}];
        
        float dist = std::sqrt(std::pow(pt[0]-sx, 2) + std::pow(pt[1]-sy, 2) + std::pow(pt[2]-sz, 2));
        if (dist <= sr + 0.04f) {
          is_self_collision = true;
          break;
        }
      }
      if (!is_self_collision) {
        valid_env_voxels.push_back(pt);
      }
    }
    RCLCPP_DEBUG(this->get_logger(), "[VAMP ENGINE LOG] Building collision environment with %zu obstacle voxels", valid_env_voxels.size());
    return valid_env_voxels;
  }

  void handleCheckAndReplan(
    const std::shared_ptr<my_actions::srv::CheckAndReplan::Request> request,
    std::shared_ptr<my_actions::srv::CheckAndReplan::Response> response)
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    if (request->current_joints.size() != 6) {
      RCLCPP_ERROR(this->get_logger(), "CheckAndReplan: current_joints must have 6 elements");
      response->success = false;
      return;
    }

    size_t num_poses = request->remaining_path_flat.size() / 6;
    size_t check_count = std::min(static_cast<size_t>(lookahead_poses_), num_poses);

    bool collision_found = false;
    float r_point = 0.0025f;
    const bool allow_target_contact = request->allow_target_contact &&
      request->allowed_contact_radius_m > 0.0F;
    const float allowed_radius_sq = request->allowed_contact_radius_m *
      request->allowed_contact_radius_m;
    const auto is_allowed_contact = [&request, allow_target_contact, allowed_radius_sq](
        const Point &point) {
      if (!allow_target_contact) return false;
      const float dx = point[0] - request->allowed_contact_center.x;
      const float dy = point[1] - request->allowed_contact_center.y;
      const float dz = point[2] - request->allowed_contact_center.z;
      return dx * dx + dy * dy + dz * dz <= allowed_radius_sq;
    };

    // Use the same self-filtered, voxelised environment as RRTC and grasp
    // candidate validation. Checking the raw camera cloud here reintroduced
    // points from the robot/camera that the planner correctly removes, which
    // caused a false collision before RRTC had a chance to assess the path.
    std::vector<Point> execution_env = getFilteredEnvironment(request->current_joints);
    if (allow_target_contact) {
      execution_env.erase(
        std::remove_if(
          execution_env.begin(), execution_env.end(), is_allowed_contact),
        execution_env.end());
    }

    // SIMD Sphere Collision Validation for remaining lookahead path poses
    for (size_t k = 0; k < check_count; ++k) {
      Robot::ConfigurationBlock<1> c_in;
      for (size_t j = 0; j < 6; ++j) {
        c_in[j][0] = request->remaining_path_flat[k * 6 + j];
      }
      Robot::Spheres<1> out;
      Robot::template sphere_fk<1>(c_in, out);

      for (const auto& pt : execution_env) {
        for (size_t s = 0; s < Robot::n_spheres; ++s) {
          if (!initialGraspChecksEnvironmentSphere(s, request->grasp_motion)) continue;
          if (vamp::collision::sphere_sphere_sql2(
                out.x[{s, 0}], out.y[{s, 0}], out.z[{s, 0}], out.r[{s, 0}],
                pt[0], pt[1], pt[2], r_point) < 0)
          {
            collision_found = true;
            break;
          }
        }
        if (collision_found) break;
      }
      if (collision_found) break;
    }

    if (collision_found || num_poses <= 1) {
      if (collision_found) {
        RCLCPP_WARN(this->get_logger(), "[VAMP PLANNER] Collision detected on path! Re-planning with RRTC...");
      } else {
        RCLCPP_INFO(this->get_logger(), "[VAMP PLANNER] Initial path planning requested. Solving RRTC...");
      }
      std::vector<float> path_out = replanRRTC(
        request->current_joints, request->remaining_path_flat,
        allow_target_contact, request->allowed_contact_center,
        request->allowed_contact_radius_m, request->grasp_motion);
      if (path_out.empty()) {
        response->success = false;
      } else {
        response->success = true;
        response->path_flat = path_out;
      }
    } else {
      response->success = true;
      response->path_flat = request->remaining_path_flat;
    }
  }

  static void updatePoseIkDiagnostics(const Eigen::Isometry3f &desired,
                                      const Eigen::Isometry3f &current,
                                      PoseIkDiagnostics &diagnostics) {
    const float position_error = (desired.translation() - current.translation()).norm();
    const Eigen::AngleAxisf rotation_error(desired.linear() * current.linear().transpose());
    const float angle_error = std::abs(rotation_error.angle());
    const float current_cost = diagnostics.best_position_error +
      0.05F * diagnostics.best_rotation_error;
    const float new_cost = position_error + 0.05F * angle_error;
    if (new_cost < current_cost) {
      diagnostics.best_position_error = position_error;
      diagnostics.best_rotation_error = angle_error;
    }
  }

  bool graspOrientationWithinCaptureLimits(
      const Eigen::Matrix3f &capture_rotation,
      const Eigen::Matrix3f &candidate_rotation) const {
    const auto wrap_angle = [](float angle) {
        return std::atan2(std::sin(angle), std::cos(angle));
      };
    const Eigen::Vector3f reference_approach =
      capture_rotation.col(2).normalized();
    const Eigen::Vector3f candidate_approach =
      candidate_rotation.col(2).normalized();
    const float reference_azimuth = std::atan2(
      reference_approach.y(), reference_approach.x());
    const float candidate_horizontal = std::hypot(
      candidate_approach.x(), candidate_approach.y());
    const float candidate_azimuth = candidate_horizontal > 1.0e-5F ?
      std::atan2(candidate_approach.y(), candidate_approach.x()) :
      reference_azimuth;
    const float reference_elevation = std::atan2(
      reference_approach.z(), std::hypot(
        reference_approach.x(), reference_approach.y()));
    const float candidate_elevation = std::atan2(
      candidate_approach.z(), candidate_horizontal);
    const float yaw_delta = std::abs(wrap_angle(
      candidate_azimuth - reference_azimuth));
    const float pitch_delta = std::abs(
      candidate_elevation - reference_elevation);

    Eigen::Vector3f reference_closing = capture_rotation.col(1) -
      candidate_approach * capture_rotation.col(1).dot(candidate_approach);
    Eigen::Vector3f candidate_closing = candidate_rotation.col(1) -
      candidate_approach * candidate_rotation.col(1).dot(candidate_approach);
    if (reference_closing.squaredNorm() < 1.0e-8F ||
        candidate_closing.squaredNorm() < 1.0e-8F) return false;
    reference_closing.normalize();
    candidate_closing.normalize();
    if (reference_closing.dot(candidate_closing) < 0.0F) {
      candidate_closing = -candidate_closing;
    }
    const float roll_delta = std::abs(std::atan2(
      candidate_approach.dot(reference_closing.cross(candidate_closing)),
      reference_closing.dot(candidate_closing)));
    const float radians_per_degree = static_cast<float>(M_PI / 180.0);
    return yaw_delta <= static_cast<float>(
      aligned_zero_pose_yaw_limit_degrees_) * radians_per_degree + 1.0e-4F &&
      pitch_delta <= static_cast<float>(
      aligned_zero_pose_pitch_limit_degrees_) * radians_per_degree + 1.0e-4F &&
      roll_delta <= static_cast<float>(
      aligned_zero_pose_roll_limit_degrees_) * radians_per_degree + 1.0e-4F;
  }

  bool solvePiperContactAxisIK(
      const Eigen::Vector3f &contact_centroid,
      const Eigen::Vector3f &dominant_axis,
      const Eigen::Matrix3f &aligned_rotation,
      const Eigen::Matrix3f &capture_rotation,
      const std::vector<float> &live_seed,
      std::vector<float> &out_joints,
      Eigen::Matrix3f &out_rotation) {
    ComponentTimer numeric_timer(PipelineComponent::NumericIK);
    if (live_seed.size() != 6 || !contact_centroid.allFinite() ||
        dominant_axis.squaredNorm() < 1.0e-8F) return false;
    Eigen::Vector3f desired_height = dominant_axis.normalized();
    if (desired_height.dot(aligned_rotation.col(0)) < 0.0F) {
      desired_height = -desired_height;
    }
    const Eigen::Vector3f preferred_approach =
      aligned_rotation.col(2).normalized();
    constexpr float position_tolerance_m = 0.003F;
    constexpr float axis_tolerance_rad =
      10.0F * static_cast<float>(M_PI / 180.0);
    constexpr float axis_weight = 0.10F;
    constexpr float approach_weight = 0.02F;
    constexpr float finite_difference = 0.001F;
    constexpr int maximum_iterations = 300;
    const float contact_from_tcp = static_cast<float>(
      grasp_contact_depth_m_ - grasp_tcp_offset_m_);
    const auto &lower = kPiperJointLower;
    const auto &upper = kPiperJointUpper;

    std::vector<std::array<float, 6>> seeds;
    std::array<float, 6> current_seed{};
    std::copy_n(live_seed.begin(), 6, current_seed.begin());
    seeds.push_back(current_seed);
    if (previous_valid_ik_) seeds.push_back(*previous_valid_ik_);
    std::array<float, 6> home{};
    std::copy_n(deterministic_ik_home_.begin(), 6, home.begin());
    seeds.push_back(home);

    for (auto seed : seeds) {
      std::vector<float> joints(seed.begin(), seed.end());
      for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
        if (active_pipeline_metrics) ++active_pipeline_metrics->ik_iterations;
        std::array<float, 6> q{};
        std::copy_n(joints.begin(), 6, q.begin());
        if (active_pipeline_metrics) ++active_pipeline_metrics->ik_fk_evaluations;
        const Eigen::Isometry3f current = Robot::eefk(q);
        const Eigen::Vector3f approach = current.linear().col(2).normalized();
        const Eigen::Vector3f height = current.linear().col(0).normalized();
        const Eigen::Vector3f current_contact =
          current.translation() + approach * contact_from_tcp;
        const float position_error =
          (contact_centroid - current_contact).norm();
        const float axis_error = std::acos(std::clamp(
          height.dot(desired_height), -1.0F, 1.0F));
        if (position_error <= position_tolerance_m &&
            axis_error <= axis_tolerance_rad &&
            graspOrientationWithinCaptureLimits(
              capture_rotation, current.linear())) {
          out_joints = joints;
          out_rotation = current.linear();
          return true;
        }

        Eigen::Matrix<float, 9, 1> error;
        error.segment<3>(0) = contact_centroid - current_contact;
        error.segment<3>(3) = axis_weight * (desired_height - height);
        error.segment<3>(6) = approach_weight *
          (preferred_approach - approach);
        Eigen::Matrix<float, 9, 6> jacobian;
        for (size_t column = 0; column < 6; ++column) {
          q[column] += finite_difference;
          if (active_pipeline_metrics) ++active_pipeline_metrics->ik_fk_evaluations;
          const Eigen::Isometry3f perturbed = Robot::eefk(q);
          q[column] -= finite_difference;
          const Eigen::Vector3f perturbed_approach =
            perturbed.linear().col(2).normalized();
          const Eigen::Vector3f perturbed_contact =
            perturbed.translation() +
            perturbed_approach * contact_from_tcp;
          jacobian.block<3, 1>(0, column) =
            (perturbed_contact - current_contact) / finite_difference;
          jacobian.block<3, 1>(3, column) = axis_weight *
            (perturbed.linear().col(0).normalized() - height) /
            finite_difference;
          jacobian.block<3, 1>(6, column) = approach_weight *
            (perturbed_approach - approach) / finite_difference;
        }
        const Eigen::Matrix<float, 9, 9> damped =
          jacobian * jacobian.transpose() +
          0.0025F * Eigen::Matrix<float, 9, 9>::Identity();
        Eigen::Matrix<float, 6, 1> step =
          jacobian.transpose() * damped.ldlt().solve(error);
        const float maximum_step = step.cwiseAbs().maxCoeff();
        if (maximum_step > 0.08F) step *= 0.08F / maximum_step;
        for (size_t joint = 0; joint < 6; ++joint) {
          joints[joint] = std::clamp(
            joints[joint] + 0.45F * step[joint],
            lower[joint], upper[joint]);
        }
      }
    }
    return false;
  }

  // Refine one full SE(3) numerical IK seed. The candidate orientation is
  // verified against Robot::eefk rather than treated as display-only data.
  bool refinePiperPoseIK(const Eigen::Isometry3f &desired,
                         const std::array<float, 6> &seed,
                         std::vector<float> &out_joints,
                         PoseIkDiagnostics &diagnostics,
                         const float position_tolerance_m = 0.004F,
                         const float rotation_tolerance_rad = 0.20F,
                         const int maximum_iterations = 800,
                         bool bounded_stagnation = false) {
    ComponentTimer numeric_timer(PipelineComponent::NumericIK);
    out_joints.assign(seed.begin(), seed.end());
    const auto &lower = kPiperJointLower;
    const auto &upper = kPiperJointUpper;
    constexpr float delta = 0.001f;
    constexpr float orientation_weight = 0.12f;
    IkStagnationGuard stagnation(bounded_stagnation ? 60 : 0);
    for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
      if (active_pipeline_metrics) ++active_pipeline_metrics->ik_iterations;
      std::array<float, 6> q{};
      for (size_t i = 0; i < 6; ++i) q[i] = out_joints[i];
      if (active_pipeline_metrics) ++active_pipeline_metrics->ik_fk_evaluations;
      const Eigen::Isometry3f current = Robot::eefk(q);
      const Eigen::Vector3f position_error = desired.translation() - current.translation();
      const Eigen::AngleAxisf rotation_error(desired.linear() * current.linear().transpose());
      const float rotation_angle = std::abs(rotation_error.angle());
      Eigen::Vector3f angular_error = Eigen::Vector3f::Zero();
      if (rotation_angle >= 1e-7F) {
        angular_error = rotation_error.axis() * rotation_error.angle();
      }
      updatePoseIkDiagnostics(desired, current, diagnostics);
      if (position_error.norm() <= position_tolerance_m &&
          rotation_angle <= rotation_tolerance_rad) return true;
      if (bounded_stagnation && stagnation.stalled(position_error.squaredNorm() +
          orientation_weight*orientation_weight*rotation_angle*rotation_angle)) return false;
      Eigen::Matrix<float, 6, 1> error;
      for (int axis = 0; axis < 3; ++axis) {
        error[axis] = position_error[axis];
        error[axis + 3] = orientation_weight * angular_error[axis];
      }
      Eigen::Matrix<float, 6, 6> jacobian;
      for (size_t column = 0; column < 6; ++column) {
        q[column] += delta;
        if (active_pipeline_metrics) ++active_pipeline_metrics->ik_fk_evaluations;
        const Eigen::Isometry3f perturbed = Robot::eefk(q);
        q[column] -= delta;
        jacobian.block<3, 1>(0, column) = (perturbed.translation() - current.translation()) / delta;
        const Eigen::AngleAxisf delta_rotation(perturbed.linear() * current.linear().transpose());
        jacobian.block<3, 1>(3, column) = orientation_weight * delta_rotation.axis() * delta_rotation.angle() / delta;
      }
      const Eigen::Matrix<float, 6, 6> damped = jacobian * jacobian.transpose() + 0.0025f * Eigen::Matrix<float, 6, 6>::Identity();
      const Eigen::Matrix<float, 6, 1> step = jacobian.transpose() * damped.ldlt().solve(error);
      for (size_t i = 0; i < 6; ++i) out_joints[i] = std::clamp(out_joints[i] + 0.35f * step[i], lower[i], upper[i]);
    }
    return false;
  }

  // Piper has several shoulder/elbow/wrist IK branches. Starting only from
  // the live joint state caused reachable GPD poses to be rejected whenever
  // the numerical solver was trapped in the current branch. Try a small,
  // deterministic set of seeds and stop at the first verified solution.
  bool solvePiperPoseIK(const Eigen::Isometry3f &desired,
                        const std::vector<float> &seed,
                        std::vector<float> &out_joints,
                        PoseIkDiagnostics &diagnostics,
                        bool seed_only = false,
                        const float position_tolerance_m = 0.004F,
                        const float rotation_tolerance_rad = 0.20F,
                        const int maximum_iterations = 800,
                        const std::function<bool(const std::vector<float> &)> &valid_solution = {},
                        bool first_valid_solution = false) {
    if (active_pipeline_metrics) ++active_pipeline_metrics->pose_ik_calls;
    if (seed.size() != 6) return false;
    const auto &lower = kPiperJointLower;
    const auto &upper = kPiperJointUpper;
    std::vector<std::array<float, 6>> seeds;
    const auto add_seed = [&seeds, &lower, &upper](std::array<float, 6> candidate) {
      for (size_t i = 0; i < candidate.size(); ++i) {
        if (!std::isfinite(candidate[i])) return;
        candidate[i] = std::clamp(candidate[i], lower[i], upper[i]);
      }
      for (const auto &existing : seeds) {
        float distance_sq = 0.0F;
        for (size_t i = 0; i < candidate.size(); ++i) {
          const float delta_q = candidate[i] - existing[i];
          distance_sq += delta_q * delta_q;
        }
        if (distance_sq < 0.01F) return;
      }
      seeds.push_back(candidate);
    };

    std::array<float, 6> live_seed{};
    std::copy_n(seed.begin(), 6, live_seed.begin());
    if (seed_only) {
      add_seed(live_seed);
    } else if (!deterministic_ik_) {
      // The default is deterministic. This branch preserves a narrow legacy
      // mode but never introduces a random seed.
      add_seed(live_seed);
      add_seed({0.0F, 0.8F, -1.4F, 0.0F, 0.0F, 0.0F});
    } else {
      std::array<float, 6> home{};
      std::copy_n(deterministic_ik_home_.begin(), 6, home.begin());
      const auto seed_bank = deterministicIkSeedBank(
        live_seed, previous_valid_ik_, home, deterministic_ik_seed_count_);
      for (const auto &candidate : seed_bank) {
        add_seed(candidate);
      }
    }

    bool found = false;
    std::size_t best_seed_index = std::numeric_limits<std::size_t>::max();
    std::vector<float> best_solution;
    std::vector<std::vector<float>> unique_solutions;
    for (std::size_t seed_index = 0; seed_index < seeds.size(); ++seed_index) {
      const auto &candidate_seed = seeds[seed_index];
      ++diagnostics.attempts;
      std::vector<float> candidate_joints;
      PoseIkDiagnostics attempt_diagnostics;
      if (refinePiperPoseIK(desired, candidate_seed, candidate_joints,
                           attempt_diagnostics, position_tolerance_m,
                           rotation_tolerance_rad, maximum_iterations, first_valid_solution)) {
        diagnostics.keepBest(attempt_diagnostics);
        if (valid_solution && !valid_solution(candidate_joints)) {
          ++diagnostics.collision_rejections;
          continue;
        }
        bool duplicate_solution = false;
        for (const auto &existing : unique_solutions) {
          float distance_sq = 0.0F;
          for (size_t i = 0; i < 6; ++i) distance_sq += std::pow(existing[i] - candidate_joints[i], 2);
          if (distance_sq < 1.0e-6F) { duplicate_solution = true; break; }
        }
        if (duplicate_solution) continue;
        if (first_valid_solution) {
          out_joints = candidate_joints;
          std::array<float, 6> cached{};
          std::copy_n(out_joints.begin(), 6, cached.begin());
          previous_valid_ik_ = cached;
          return true;
        }
        unique_solutions.push_back(candidate_joints);
        PiperJointArray candidate_solution{};
        PiperJointArray best_solution_array{};
        std::copy_n(candidate_joints.begin(), 6, candidate_solution.begin());
        if (found) {
          std::copy_n(best_solution.begin(), 6, best_solution_array.begin());
        }
        if (!found || deterministicIkSolutionPreferred(
            candidate_solution, seed_index, best_solution_array, best_seed_index, live_seed)) {
          found = true;
          best_seed_index = seed_index; best_solution = candidate_joints;
        }
      }
      diagnostics.keepBest(attempt_diagnostics);
    }
    if (!found) return false;
    out_joints = best_solution;
    std::array<float, 6> cached{};
    std::copy_n(out_joints.begin(), 6, cached.begin());
    previous_valid_ik_ = cached;
    return true;
  }

  bool solveRelaxedPregraspIK(const Eigen::Isometry3f &exact_pregrasp,
                              const std::vector<float> &seed_ik,
                              std::vector<float> &out_ik,
                              PoseIkDiagnostics &diagnostics,
                              Eigen::Isometry3f &resolved_pregrasp) {
    // The pre-grasp position remains on the final grasp's approach line, but
    // orientation is a bounded soft constraint. The numerical IK therefore
    // adjusts all six joints continuously instead of testing a grid of fixed
    // tilt targets. The caller verifies the resolved 5--15 cm standoff.
    const float orientation_tolerance_rad =
      static_cast<float>(pregrasp_orientation_tolerance_degrees_ *
        M_PI / 180.0);
    constexpr int pregrasp_maximum_iterations = 700;
    if (!solvePiperPoseIK(
        exact_pregrasp, seed_ik, out_ik, diagnostics, false,
        static_cast<float>(pregrasp_position_tolerance_m_),
        orientation_tolerance_rad, pregrasp_maximum_iterations)) {
      return false;
    }
    std::array<float, 6> resolved_joints{};
    std::copy_n(out_ik.begin(), 6, resolved_joints.begin());
    resolved_pregrasp = Robot::eefk(resolved_joints);
    return true;
  }

  bool pregraspKeepsSeedVisible(
    const std::vector<float> & current_joints,
    const std::vector<float> & pregrasp_joints,
    const geometry_msgs::msg::TransformStamped & current_camera_transform,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const Eigen::Vector3f & seed_base) const
  {
    if (current_joints.size() != 6 || pregrasp_joints.size() != 6 ||
      camera_info.width == 0 || camera_info.height == 0)
    {
      return false;
    }
    PiperJointArray current{};
    PiperJointArray pregrasp{};
    std::copy_n(current_joints.begin(), 6, current.begin());
    std::copy_n(pregrasp_joints.begin(), 6, pregrasp.begin());
    const Eigen::Isometry3f current_base_from_hand = Robot::eefk(current);
    const Eigen::Isometry3f pregrasp_base_from_hand = Robot::eefk(pregrasp);

    const auto & transform = current_camera_transform.transform;
    Eigen::Quaternionf quaternion(
      static_cast<float>(transform.rotation.w),
      static_cast<float>(transform.rotation.x),
      static_cast<float>(transform.rotation.y),
      static_cast<float>(transform.rotation.z));
    if (quaternion.squaredNorm() < 1.0e-8F) {
      return false;
    }
    quaternion.normalize();
    Eigen::Isometry3f current_base_from_camera = Eigen::Isometry3f::Identity();
    current_base_from_camera.linear() = quaternion.toRotationMatrix();
    current_base_from_camera.translation() = Eigen::Vector3f(
      static_cast<float>(transform.translation.x),
      static_cast<float>(transform.translation.y),
      static_cast<float>(transform.translation.z));

    // Carry the exact synchronized FK * calibrated-X camera pose to the
    // proposed hand pose. No TF lookup or alternative calibration is used.
    const Eigen::Isometry3f hand_from_camera =
      current_base_from_hand.inverse() * current_base_from_camera;
    const Eigen::Isometry3f pregrasp_base_from_camera =
      pregrasp_base_from_hand * hand_from_camera;
    const CameraIntrinsics intrinsics{
      static_cast<float>(camera_info.k[0]),
      static_cast<float>(camera_info.k[4]),
      static_cast<float>(camera_info.k[2]),
      static_cast<float>(camera_info.k[5])};
    int u = -1;
    int v = -1;
    return LocalGraspRefiner::projectBasePoint(
      seed_base, pregrasp_base_from_camera, intrinsics,
      static_cast<int>(camera_info.width),
      static_cast<int>(camera_info.height), u, v);
  }

  std::optional<float> cameraUprightScore(
    const std::vector<float> & current_joints,
    const std::vector<float> & proposed_joints,
    const geometry_msgs::msg::TransformStamped & current_camera_transform) const
  {
    if (current_joints.size() != 6 || proposed_joints.size() != 6) {
      return std::nullopt;
    }
    PiperJointArray current{};
    PiperJointArray proposed{};
    std::copy_n(current_joints.begin(), 6, current.begin());
    std::copy_n(proposed_joints.begin(), 6, proposed.begin());
    const Eigen::Isometry3f current_base_from_hand = Robot::eefk(current);
    const Eigen::Isometry3f proposed_base_from_hand = Robot::eefk(proposed);

    const auto & transform = current_camera_transform.transform;
    Eigen::Quaternionf quaternion(
      static_cast<float>(transform.rotation.w),
      static_cast<float>(transform.rotation.x),
      static_cast<float>(transform.rotation.y),
      static_cast<float>(transform.rotation.z));
    if (quaternion.squaredNorm() < 1.0e-8F) {
      return std::nullopt;
    }
    quaternion.normalize();
    Eigen::Isometry3f current_base_from_camera = Eigen::Isometry3f::Identity();
    current_base_from_camera.linear() = quaternion.toRotationMatrix();
    current_base_from_camera.translation() = Eigen::Vector3f(
      static_cast<float>(transform.translation.x),
      static_cast<float>(transform.translation.y),
      static_cast<float>(transform.translation.z));

    // Preserve the synchronized manual FK * calibrated-X hand/camera
    // relation. OpenCV camera +Y points down in the image, therefore -Y is
    // image-up. Its alignment with base_link +Z is +1 upright and -1 inverted.
    const Eigen::Isometry3f hand_from_camera =
      current_base_from_hand.inverse() * current_base_from_camera;
    const Eigen::Isometry3f proposed_base_from_camera =
      proposed_base_from_hand * hand_from_camera;
    return std::clamp(
      -proposed_base_from_camera.linear().col(1).dot(Eigen::Vector3f::UnitZ()),
      -1.0F, 1.0F);
  }

  // Numerical Inverse Kinematics for Piper 6-DOF End-Effector Position
  bool solvePiperIK(const Point& target_pos, const Point& look_at_target,
                    const std::vector<float>& seed_joints, std::vector<float>& out_joints,
                    float tolerance_sq = 1e-4f, bool enforce_limits = true, bool ignore_convergence = false, bool lock_base = false)
  {
    ComponentTimer numeric_timer(PipelineComponent::NumericIK);
    out_joints = seed_joints;
    
    float learning_rate = 1.0f;
    float delta = 0.001f;
    int max_iters = 2000;
    
    // Joint limits — must clamp each iteration to prevent divergence
    const auto &lower_limits = kPiperJointLower;
    const auto &upper_limits = kPiperJointUpper;
    
    float q_curr[6];
    for (size_t j = 0; j < 6; ++j) q_curr[j] = out_joints[j];
    
    Robot::ConfigurationBlock<1> c_in;
    size_t ee_idx = Robot::n_spheres - 1;
    
    float final_err_sq = 1e9f;
    float prev_err_sq = 1e9f;
    int stagnation_count = 0;
    
    for (int iter = 0; iter < max_iters; ++iter) {
      if (active_pipeline_metrics) ++active_pipeline_metrics->ik_iterations;
      for (size_t j = 0; j < 6; ++j) c_in[j][0] = q_curr[j];
      Robot::Spheres<1> out;
      Robot::template sphere_fk<1>(c_in, out);
      
      float cx = out.x[{ee_idx, 0}];
      float cy = out.y[{ee_idx, 0}];
      float cz = out.z[{ee_idx, 0}];
      
      float err_x = cx - target_pos[0];
      float err_y = cy - target_pos[1];
      float err_z = cz - target_pos[2];
      
      float err_sq = err_x*err_x + err_y*err_y + err_z*err_z;
      // Calculate Jacobian numerically for position (3x6) and orientation (1x6)
      float J[3][6];
      float J_ang[6];
      
      // Base orientation error calculation
      std::array<float, 6> c_arr_base;
      for (size_t k = 0; k < 6; ++k) c_arr_base[k] = q_curr[k];
      if (active_pipeline_metrics) ++active_pipeline_metrics->ik_fk_evaluations;
      Eigen::Isometry3f ee_pose_base = Robot::eefk(c_arr_base);
      Eigen::Vector3f ee_z_base = ee_pose_base.linear().col(2);
      
      Eigen::Vector3f desired_dir(look_at_target[0] - cx, look_at_target[1] - cy, look_at_target[2] - cz);
      float dir_norm = desired_dir.norm();
      if (dir_norm > 1e-4f) {
        desired_dir /= dir_norm;
      } else {
        desired_dir = ee_z_base;
      }
      float base_angular_err = 1.0f - ee_z_base.dot(desired_dir);

      // Add angular error to total error
      float w_ang = 3.0f; // Increased weight to tighten effector pointing angle
      err_sq += w_ang * base_angular_err;
      final_err_sq = err_sq;
      
      if (err_sq < tolerance_sq) {
        break; // Converged
      }
      
      // Adaptive learning rate
      if (err_sq >= prev_err_sq) {
        learning_rate *= 0.5f;
        stagnation_count++;
        if (learning_rate < 0.01f) learning_rate = 0.01f;
        if (stagnation_count > 100) break; // Stuck in local minimum
      } else {
        learning_rate = std::min(learning_rate * 1.1f, 2.0f);
        stagnation_count = 0;
      }
      prev_err_sq = err_sq;

      for (size_t j = 0; j < 6; ++j) {
        q_curr[j] += delta;
        for (size_t k = 0; k < 6; ++k) c_in[k][0] = q_curr[k];
        Robot::Spheres<1> out_d;
        Robot::template sphere_fk<1>(c_in, out_d);
        J[0][j] = (out_d.x[{ee_idx, 0}] - cx) / delta;
        J[1][j] = (out_d.y[{ee_idx, 0}] - cy) / delta;
        J[2][j] = (out_d.z[{ee_idx, 0}] - cz) / delta;
        
        // Numerical gradient for angular error
        std::array<float, 6> c_arr_d;
        for (size_t k = 0; k < 6; ++k) c_arr_d[k] = q_curr[k];
        if (active_pipeline_metrics) ++active_pipeline_metrics->ik_fk_evaluations;
      Eigen::Isometry3f ee_pose_d = Robot::eefk(c_arr_d);
        Eigen::Vector3f ee_z_d = ee_pose_d.linear().col(2);
        
        // We evaluate direction from the perturbed position to the fixed target
        Eigen::Vector3f desired_dir_d(look_at_target[0] - out_d.x[{ee_idx, 0}], 
                                      look_at_target[1] - out_d.y[{ee_idx, 0}], 
                                      look_at_target[2] - out_d.z[{ee_idx, 0}]);
        if (desired_dir_d.norm() > 1e-4f) desired_dir_d.normalize();
        else desired_dir_d = ee_z_d;
        
        float angular_err_d = 1.0f - ee_z_d.dot(desired_dir_d);
        J_ang[j] = (angular_err_d - base_angular_err) / delta;

        q_curr[j] -= delta; // restore
      }
      
      // Jacobian transpose gradient descent with a regularizer towards seed_joints
      // to keep overall posture similar
      float reg_weight = 0.05f;
      for (size_t j = 0; j < 6; ++j) {
        if (lock_base && j < 3) continue; // Lock the base and shoulder joints for pure wrist twisting
        float pos_grad = J[0][j]*err_x + J[1][j]*err_y + J[2][j]*err_z;
        float ang_grad = w_ang * J_ang[j]; // J_ang is already gradient of the scalar angular error
        float grad = pos_grad + ang_grad;
        float reg_grad = (q_curr[j] - seed_joints[j]) * reg_weight;
        q_curr[j] -= learning_rate * (grad + reg_grad);
        
        // Clamp to joint limits to prevent divergence if enforce_limits is true
        if (enforce_limits) {
          if (q_curr[j] < lower_limits[j]) q_curr[j] = lower_limits[j];
          if (q_curr[j] > upper_limits[j]) q_curr[j] = upper_limits[j];
        }
      }
    }
    
    for (size_t j = 0; j < 6; ++j) {
      out_joints[j] = q_curr[j];
    }
    
    if (final_err_sq >= tolerance_sq) {
      RCLCPP_DEBUG(this->get_logger(), "  [IK] Failed: final_err=%.4fm, target=[%.3f,%.3f,%.3f]",
                   std::sqrt(final_err_sq), target_pos[0], target_pos[1], target_pos[2]);
    }
    
    if (ignore_convergence) return true;
    return final_err_sq < tolerance_sq;
  }

  bool isInnerJawPadSphereContact(
    std::size_t sphere_index, const Eigen::Vector3f &sphere_center,
    const Eigen::Isometry3f &grasp_pose,
    const Eigen::Vector3f &contact_left,
    const Eigen::Vector3f &contact_right,
    float allowed_contact_radius_m) const
  {
    // The generated Piper model uses spheres 74..81 for the gripper palm/base
    // and 82..95 for the detailed finger bodies/pads. The palm/base is never
    // exempted. This local-volume check further limits the exemption to the
    // intended inner pad region around one of the two refined contacts.
    constexpr std::size_t first_finger_sphere = 82;
    if (sphere_index < first_finger_sphere) return false;
    const Eigen::Vector3f local_sphere =
      grasp_pose.linear().transpose() * (sphere_center - grasp_pose.translation());
    const float height_limit = static_cast<float>(0.5 * finger_height_m_) +
      allowed_contact_radius_m;
    const float depth_limit = static_cast<float>(0.5 * finger_length_m_) +
      allowed_contact_radius_m;
    const float side_limit = static_cast<float>(finger_width_m_) +
      allowed_contact_radius_m;
    for (const Eigen::Vector3f &contact : {contact_left, contact_right}) {
      const Eigen::Vector3f local_contact =
        grasp_pose.linear().transpose() * (contact - grasp_pose.translation());
      if (std::abs(local_sphere.x() - local_contact.x()) <= height_limit &&
        std::abs(local_sphere.y() - local_contact.y()) <= side_limit &&
        std::abs(local_sphere.z() - local_contact.z()) <= depth_limit)
      {
        return true;
      }
    }
    return false;
  }

  bool isObstacleInIntendedJawContactPatch(
    const Eigen::Vector3f &obstacle,
    const Eigen::Isometry3f &grasp_pose,
    const Eigen::Vector3f &contact_left,
    const Eigen::Vector3f &contact_right,
    const float surface_tolerance_m) const
  {
    // This is deliberately independent of the segmentation label. It does
    // not grow or relabel the target mask. Instead it describes the small
    // physical patch swept by an inner jaw pad at either computed contact.
    // X/Y/Z in gripper_base are height/closing/approach respectively.
    const Eigen::Vector3f local_obstacle =
      grasp_pose.linear().transpose() *
      (obstacle - grasp_pose.translation());
    const float height_limit = static_cast<float>(0.5 * finger_height_m_) +
      surface_tolerance_m;
    const float closing_limit = static_cast<float>(0.5 * finger_width_m_) +
      surface_tolerance_m;
    const float depth_limit = static_cast<float>(0.5 * finger_length_m_) +
      surface_tolerance_m;
    for (const Eigen::Vector3f &contact : {contact_left, contact_right}) {
      const Eigen::Vector3f local_contact =
        grasp_pose.linear().transpose() *
        (contact - grasp_pose.translation());
      const Eigen::Vector3f delta = local_obstacle - local_contact;
      if (std::abs(delta.x()) <= height_limit &&
        std::abs(delta.y()) <= closing_limit &&
        std::abs(delta.z()) <= depth_limit)
      {
        return true;
      }
    }
    return false;
  }

  // VAMP SIMD Sphere Collision Validation for Candidate Pose
  bool isCandidatePoseSafe(
    const std::vector<float>& q_vec, const std::vector<Point>& env_voxels,
    const std::vector<uint8_t> *target_flags = nullptr,
    const Eigen::Vector3f *contact_left = nullptr,
    const Eigen::Vector3f *contact_right = nullptr,
    float allowed_contact_radius_m = 0.0F,
    const Eigen::Isometry3f *allowed_grasp_pose = nullptr,
    float jaw_opening_m = -1.0F,
    bool check_finger_environment = true,
    bool initial_grasp_arm_only = false,
    bool skip_approach_palm_sphere_80 = false)
  {

    ComponentTimer collision_timer(PipelineComponent::Collision);
    static bool initialized_empty_env = false;
    static EnvironmentVector empty_env_v;
    if (!initialized_empty_env) {
      EnvironmentInput empty_mvt_env;
      empty_env_v = EnvironmentVector(empty_mvt_env);
      initialized_empty_env = true;
    }
    
    Robot::Configuration cand(q_vec);
    Robot::ConfigurationBlock<rake> b;
    for (size_t d = 0; d < Robot::dimension; ++d) b[d] = cand.broadcast(d);
    if (!Robot::fkcc<rake>(empty_env_v, b)) {
      return false; // Self-collision detected
    }
    if (env_voxels.empty()) return true;

    Robot::ConfigurationBlock<1> c_in;
    for (size_t j = 0; j < 6; ++j) c_in[j][0] = q_vec[j];
    Robot::Spheres<1> out;
    Robot::template sphere_fk<1>(c_in, out);

    std::array<float,6> actual_joints{};
    std::copy_n(q_vec.begin(),6,actual_joints.begin());
    const auto actual_pose=Robot::eefk(actual_joints);
    const float opening=jaw_opening_m >= 0.0F ? jaw_opening_m :
      contact_left && contact_right ? (*contact_left-*contact_right).norm() : -1.0F;
    float r_point = 0.0025f;
    const bool allow_jaw_contacts = target_flags != nullptr &&
      target_flags->size() == env_voxels.size() && contact_left != nullptr &&
      contact_right != nullptr && allowed_contact_radius_m > 0.0F;
    const float allowed_contact_radius_sq =
      allowed_contact_radius_m * allowed_contact_radius_m;
    // No contact exemption is ever applied to an arm or wrist sphere (0..73).
    // With an explicit final-approach pose, isInnerJawPadSphereContact also
    // rejects the palm/base spheres and any finger-body point outside the pad.
    constexpr size_t first_gripper_base_sphere = 74;
    const auto &index = exactCollisionIndex(env_voxels);
    for (size_t s = 0; s < Robot::n_spheres; ++s) {
      if (!approachPalmSphereEnabled(s, skip_approach_palm_sphere_80)) continue;
      if (!initialGraspChecksEnvironmentSphere(s, initial_grasp_arm_only)) continue;
      if (!finalApproachChecksEnvironmentSphere(s, check_finger_environment)) continue;
      const Eigen::Vector3f center=vamp_node::piperFingerSphereCenter(s,
        Eigen::Vector3f(out.x[{s,0}], out.y[{s,0}], out.z[{s,0}]), actual_pose, opening);
      if (!index.sphereClearWithPredicate(center, out.r[{s,0}] + r_point,
          [&](size_t obstacle_index, const Eigen::Vector3f &) {
        const auto &pt = env_voxels[obstacle_index];
        if (!vamp_node::piperFingerMeshBoundsContain(s,
            Eigen::Vector3f(pt[0], pt[1], pt[2]), actual_pose, opening)) return false;
        if (vamp::collision::sphere_sphere_sql2(
              center.x(), center.y(), center.z(), out.r[{s, 0}],
              pt[0], pt[1], pt[2], r_point) < 0)
        {
          if (allow_jaw_contacts && s >= first_gripper_base_sphere) {
            const Eigen::Vector3f obstacle(pt[0], pt[1], pt[2]);
            const bool classified_target =
              (*target_flags)[obstacle_index] != 0;
            const bool near_contact =
              (obstacle - *contact_left).squaredNorm() <=
                allowed_contact_radius_sq ||
              (obstacle - *contact_right).squaredNorm() <=
                allowed_contact_radius_sq;
            const bool inner_pad = allowed_grasp_pose == nullptr ||
              isInnerJawPadSphereContact(
                s, center,
                *allowed_grasp_pose, *contact_left, *contact_right,
                allowed_contact_radius_m);
            const float strict_surface_tolerance_m = static_cast<float>(
              std::clamp(target_contact_surface_tolerance_m_, 0.003, 0.008));
            const bool strict_unlabelled_contact =
              allowed_grasp_pose != nullptr &&
              inner_pad &&
              isObstacleInIntendedJawContactPatch(
                obstacle, *allowed_grasp_pose, *contact_left, *contact_right,
                strict_surface_tolerance_m);
            if ((classified_target && near_contact && inner_pad) ||
              strict_unlabelled_contact) {
              return false;
            }
          }
          return true;
        }
        return false;
      })) return false;
    }

    return true;
  }

  bool graspPalmRecoveryClear(const std::vector<float> &joints,
    const std::vector<Point> &environment, float /*opening*/) const
  {
    Robot::ConfigurationBlock<1> input;
    for (size_t j=0; j<6; ++j) input[j][0]=joints[j];
    Robot::Spheres<1> spheres;
    Robot::template sphere_fk<1>(input, spheres);
    for (const size_t sphere : {size_t{80}, size_t{81}}) {
      const Eigen::Vector3f center(spheres.x[{sphere,0}], spheres.y[{sphere,0}], spheres.z[{sphere,0}]);
      if (!exactCollisionIndex(environment).sphereClearWithPredicate(
          center, spheres.r[{sphere,0}]+.0025F,
          [&](size_t index, const Eigen::Vector3f &) {
            const auto &point=environment[index];
            return vamp::collision::sphere_sphere_sql2(center.x(), center.y(), center.z(),
              spheres.r[{sphere,0}], point[0], point[1], point[2], .0025F)<0;
          })) return false;
    }
    return true;
  }

  std::string describeCandidateCollision(
    const std::vector<float> &q_vec, const std::vector<Point> &env_voxels,
    const std::vector<uint8_t> *target_flags,
    const Eigen::Vector3f *contact_left,
    const Eigen::Vector3f *contact_right,
    const float allowed_contact_radius_m,
    const Eigen::Isometry3f *allowed_grasp_pose,
    const float opening_override = -1.0F,
    const bool check_finger_environment = true,
    const bool skip_approach_palm_sphere_80 = false,
    const bool grasp_pose_scope = false)
  {
    if (q_vec.size() != Robot::dimension) return "invalid_joint_vector";
    static bool initialized_empty_env = false;
    static EnvironmentVector empty_env_v;
    if (!initialized_empty_env) {
      EnvironmentInput empty_mvt_env;
      empty_env_v = EnvironmentVector(empty_mvt_env);
      initialized_empty_env = true;
    }
    Robot::Configuration candidate(q_vec);
    Robot::ConfigurationBlock<rake> block;
    for (size_t joint = 0; joint < Robot::dimension; ++joint) {
      block[joint] = candidate.broadcast(joint);
    }
    if (!Robot::fkcc<rake>(empty_env_v, block)) {
      const auto result = Robot::fkcc_debug<rake>(empty_env_v, block);
      std::ostringstream description;
      description << "self_collision";
      size_t reported = 0;
      for (const auto &pair : result.second) {
        description << " sphere_pair=[" << pair.first << ',' << pair.second << ']';
        if (++reported == 4) break;
      }
      return description.str();
    }

    Robot::ConfigurationBlock<1> configuration;
    for (size_t joint = 0; joint < Robot::dimension; ++joint) {
      configuration[joint][0] = q_vec[joint];
    }
    Robot::Spheres<1> spheres;
    Robot::template sphere_fk<1>(configuration, spheres);
    std::array<float, 6> actual_joints{};
    std::copy_n(q_vec.begin(), 6, actual_joints.begin());
    const auto actual_pose = Robot::eefk(actual_joints);
    const float opening = opening_override >= 0.0F ? opening_override :
      contact_left && contact_right ?
      (*contact_left - *contact_right).norm() : -1.0F;
    constexpr float point_radius_m = 0.0025F;
    constexpr size_t first_gripper_base_sphere = 74;
    const bool allow_jaw_contacts = target_flags != nullptr &&
      target_flags->size() == env_voxels.size() && contact_left != nullptr &&
      contact_right != nullptr && allowed_contact_radius_m > 0.0F;
    const float allowed_radius_sq =
      allowed_contact_radius_m * allowed_contact_radius_m;
    for (size_t obstacle_index = 0; obstacle_index < env_voxels.size();
         ++obstacle_index) {
      const auto &point = env_voxels[obstacle_index];
      for (size_t sphere = 0; sphere < Robot::n_spheres; ++sphere) {
        if (!initialGraspChecksEnvironmentSphere(sphere, grasp_pose_scope)) continue;
        if (!approachPalmSphereEnabled(sphere, skip_approach_palm_sphere_80)) continue;
        if (!finalApproachChecksEnvironmentSphere(sphere, check_finger_environment)) continue;
        const auto center = vamp_node::piperFingerSphereCenter(sphere,
          Eigen::Vector3f(spheres.x[{sphere, 0}], spheres.y[{sphere, 0}],
            spheres.z[{sphere, 0}]), actual_pose, opening);
        if (vamp::collision::sphere_sphere_sql2(
            center.x(), center.y(), center.z(), spheres.r[{sphere, 0}],
            point[0], point[1], point[2], point_radius_m) >= 0) {
          continue;
        }
        if (!vamp_node::piperFingerMeshBoundsContain(sphere,
            Eigen::Vector3f(point[0], point[1], point[2]), actual_pose, opening)) continue;
        bool target = false;
        bool near_contact = false;
        bool inner_pad = false;
        bool strict_contact_patch = false;
        if (allow_jaw_contacts) {
          target = (*target_flags)[obstacle_index] != 0;
          const Eigen::Vector3f obstacle(point[0], point[1], point[2]);
          near_contact =
            (obstacle - *contact_left).squaredNorm() <= allowed_radius_sq ||
            (obstacle - *contact_right).squaredNorm() <= allowed_radius_sq;
          inner_pad = sphere >= first_gripper_base_sphere &&
            (allowed_grasp_pose == nullptr || isInnerJawPadSphereContact(
              sphere, center,
              *allowed_grasp_pose, *contact_left, *contact_right,
              allowed_contact_radius_m));
          const float strict_surface_tolerance_m = static_cast<float>(
            std::clamp(target_contact_surface_tolerance_m_, 0.003, 0.008));
          strict_contact_patch = sphere >= first_gripper_base_sphere &&
            allowed_grasp_pose != nullptr &&
            inner_pad &&
            isObstacleInIntendedJawContactPatch(
              obstacle, *allowed_grasp_pose, *contact_left, *contact_right,
              strict_surface_tolerance_m);
          if ((target && near_contact && inner_pad) ||
              strict_contact_patch) continue;
        }
        std::ostringstream description;
        description << "environment sphere=" << sphere
          << " part=" << (sphere < 74 ? "arm_wrist" : sphere < 82 ? "gripper_palm" : "gripper_finger")
          << " point=[" << point[0] << ',' << point[1] << ',' << point[2]
          << "] target=" << target << " near_contact=" << near_contact
          << " inner_pad=" << inner_pad
          << " strict_contact_patch=" << strict_contact_patch
          << " tcp=[" << actual_pose.translation().transpose()
          << "] approach=[" << actual_pose.linear().col(2).transpose()
          << "] opening=" << opening;
        if (contact_left && contact_right) description
          << " contacts=[" << contact_left->transpose() << ";" << contact_right->transpose() << "]";
        return description.str();
      }
    }
    return "unknown";
  }

  // Detailed debug logger for Configuration Validity against workspace bounds & obstacles
  void logConfigValidity(const std::string& label, const Robot::Configuration& cfg, const std::vector<Point>& env_voxels)
  {
    Robot::ConfigurationBlock<1> c_in;
    auto arr = cfg.to_array();
    for (size_t j = 0; j < 6; ++j) c_in[j][0] = arr[j];
    Robot::Spheres<1> out;
    Robot::template sphere_fk<1>(c_in, out);

    float r_point = 0.0025f;
    size_t oob_count = 0;
    size_t coll_count = 0;

    for (size_t s = 0; s < Robot::n_spheres; ++s) {
      float sx = out.x[{s, 0}], sy = out.y[{s, 0}], sz = out.z[{s, 0}], sr = out.r[{s, 0}];
      if (sx < ws_x_min_ || sx > ws_x_max_ ||
          sy < ws_y_min_ || sy > ws_y_max_ ||
          sz < ws_z_min_ || sz > ws_z_max_) {
        oob_count++;
        RCLCPP_WARN(this->get_logger(), "  [VAMP VALIDITY DEBUG] %s Sphere #%zu at [%.3f, %.3f, %.3f] OUT OF WORKSPACE BOUNDS [%.2f..%.2f]",
                    label.c_str(), s, sx, sy, sz, ws_x_min_, ws_x_max_);
      }
      for (const auto& pt : env_voxels) {
        if (vamp::collision::sphere_sphere_sql2(sx, sy, sz, sr, pt[0], pt[1], pt[2], r_point) < 0) {
          coll_count++;
          RCLCPP_WARN(this->get_logger(), "  [VAMP VALIDITY DEBUG] %s Sphere #%zu at [%.3f, %.3f, %.3f] COLLIDED with voxel [%.3f, %.3f, %.3f]",
                      label.c_str(), s, sx, sy, sz, pt[0], pt[1], pt[2]);
          break;
        }
      }
    }
    // RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE LOG] %s Config [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f] Status: %zu OOB Spheres, %zu Collided Spheres",
    //            label.c_str(), arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], oob_count, coll_count);
  }

  void applyElasticBandSmoothing(std::vector<Robot::Configuration>& path, const std::vector<Point>& valid_env_voxels)
  {
    if (path.size() < 3) return;

    int max_iters = 50;
    float alpha = 0.2f;
    float beta = 10.0f;
    float safe_dist = 0.08f;

    for (int iter = 0; iter < max_iters; ++iter) {
      std::vector<Robot::Configuration> new_path = path;
      bool changed = false;

      for (size_t i = 1; i < path.size() - 1; ++i) {
        auto q_prev = path[i-1].to_array();
        auto q_curr = path[i].to_array();
        auto q_next = path[i+1].to_array();

        std::array<float, 6> F_internal = {0};
        for (size_t j = 0; j < 6; ++j) {
          F_internal[j] = alpha * (q_prev[j] - 2.0f * q_curr[j] + q_next[j]);
        }

        std::array<float, 6> F_repulse = {0};
        float delta = 0.01f;
        
        Robot::ConfigurationBlock<1> c_curr;
        for (size_t j = 0; j < 6; ++j) c_curr[j][0] = q_curr[j];
        Robot::Spheres<1> out_curr;
        Robot::template sphere_fk<1>(c_curr, out_curr);
        
        float current_min_dist = 1e9f;
        for (size_t s = 0; s < Robot::n_spheres; ++s) {
          float sx = out_curr.x[{s, 0}], sy = out_curr.y[{s, 0}], sz = out_curr.z[{s, 0}], sr = out_curr.r[{s, 0}];
          for (const auto& pt : valid_env_voxels) {
            float dist = std::sqrt(std::pow(pt[0]-sx, 2) + std::pow(pt[1]-sy, 2) + std::pow(pt[2]-sz, 2)) - sr;
            if (dist < current_min_dist) current_min_dist = dist;
          }
        }
        
        if (current_min_dist < safe_dist) {
          for (size_t j = 0; j < 6; ++j) {
            float q_test[6];
            for (size_t k = 0; k < 6; ++k) q_test[k] = q_curr[k];
            q_test[j] += delta;
            
            Robot::ConfigurationBlock<1> c_test;
            for (size_t k = 0; k < 6; ++k) c_test[k][0] = q_test[k];
            Robot::Spheres<1> out_test;
            Robot::template sphere_fk<1>(c_test, out_test);
            
            float test_min_dist = 1e9f;
            for (size_t s = 0; s < Robot::n_spheres; ++s) {
               float sx = out_test.x[{s, 0}], sy = out_test.y[{s, 0}], sz = out_test.z[{s, 0}], sr = out_test.r[{s, 0}];
               for (const auto& pt : valid_env_voxels) {
                  float tdist = std::sqrt(std::pow(pt[0]-sx, 2) + std::pow(pt[1]-sy, 2) + std::pow(pt[2]-sz, 2)) - sr;
                  if (tdist < test_min_dist) test_min_dist = tdist;
               }
            }
            
            float grad = (test_min_dist - current_min_dist) / delta;
            float mag = (safe_dist - current_min_dist);
            F_repulse[j] = beta * mag * grad;
          }
        }
        
        std::array<float, 6> q_new_arr;
        std::vector<float> q_new_vec(6);
        for (size_t j = 0; j < 6; ++j) {
          q_new_arr[j] = q_curr[j] + F_internal[j] + F_repulse[j];
          q_new_vec[j] = q_new_arr[j];
        }
        
        if (isCandidatePoseSafe(q_new_vec, valid_env_voxels)) {
            Robot::Configuration new_cfg(q_new_arr);
            new_path[i] = new_cfg;
            changed = true;
        }
      }
      if (!changed) break;
      path = new_path;
    }
    // RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE LOG] Elastic Band Smoothing complete.");
  }

  std::vector<float> replanRRTC(
      const std::vector<float>& current_joints,
      const std::vector<float>& remaining_path_flat,
      bool allow_target_contact = false,
      const geometry_msgs::msg::Point &allowed_contact_center = geometry_msgs::msg::Point(),
      float allowed_contact_radius_m = 0.0F,
      bool grasp_motion = false)
  {
    std::vector<float> goal_joints(6, 0.0f);
    if (continuous_nav_active_ && continuous_target_joints_.size() == 6) {
      goal_joints = continuous_target_joints_;
    } else if (remaining_path_flat.size() >= 6) {
      for (size_t j = 0; j < 6; ++j) {
        goal_joints[j] = remaining_path_flat[remaining_path_flat.size() - 6 + j];
      }
    } else {
      RCLCPP_ERROR(this->get_logger(), "Cannot replan: no valid goal configuration found.");
      return {};
    }

    // 1. Forward Kinematics for start and goal TCP poses
    Robot::ConfigurationBlock<1> c_start_in, c_goal_in;
    for (size_t j = 0; j < 6; ++j) {
      c_start_in[j][0] = current_joints[j];
      c_goal_in[j][0] = goal_joints[j];
    }
    Robot::Spheres<1> out_start, out_goal;
    Robot::template sphere_fk<1>(c_start_in, out_start);
    Robot::template sphere_fk<1>(c_goal_in, out_goal);

    size_t ee_idx = Robot::n_spheres - 1;
    // Point p_start = {out_start.x[{ee_idx, 0}], out_start.y[{ee_idx, 0}], out_start.z[{ee_idx, 0}]};
    Point p_goal = {out_goal.x[{ee_idx, 0}], out_goal.y[{ee_idx, 0}], out_goal.z[{ee_idx, 0}]};

    // RCLCPP_INFO(this->get_logger(),
    //  "[VAMP ENGINE LOG] Forward Kinematics: Start TCP [%.3f, %.3f, %.3f] -> Goal TCP [%.3f, %.3f, %.3f]",
    //  p_start[0], p_start[1], p_start[2], p_goal[0], p_goal[1], p_goal[2]);

    // 2. Setup VAMP Environment & Self-Arm Point Filter
    vamp::collision::Point origin = {0.0f, 0.0f, 0.0f};
    vamp::collision::Point ws_min = {ws_x_min_, ws_y_min_, ws_z_min_};
    vamp::collision::Point ws_max = {ws_x_max_, ws_y_max_, ws_z_max_};
    float r_min = 0.008f;
    float r_max = 0.07f;
    float r_point = 0.0025f;

    std::vector<Point> centervox_cloud = vamp::collision::filter_pointcloud_centervox(
        obstacle_cloud_, vamp::robots::Piper::max_radius, max_workspace_dist_, origin, ws_min, ws_max);
    if (centervox_cloud.empty() && !obstacle_cloud_.empty()) {
      centervox_cloud = obstacle_cloud_;
    }
    if (allow_target_contact && allowed_contact_radius_m > 0.0F) {
      const float radius_sq = allowed_contact_radius_m * allowed_contact_radius_m;
      const size_t before = centervox_cloud.size();
      centervox_cloud.erase(
        std::remove_if(
          centervox_cloud.begin(), centervox_cloud.end(),
          [&allowed_contact_center, radius_sq](const Point &point) {
            const float dx = point[0] - allowed_contact_center.x;
            const float dy = point[1] - allowed_contact_center.y;
            const float dz = point[2] - allowed_contact_center.z;
            return dx * dx + dy * dy + dz * dz <= radius_sq;
          }),
        centervox_cloud.end());
      RCLCPP_WARN(
        this->get_logger(),
        "[VAMP GRASP] Allowed-contact volume removed %zu/%zu obstacle voxels "
        "within %.3fm of [%.3f, %.3f, %.3f]; all other collision checks remain enabled.",
        before - centervox_cloud.size(), before, allowed_contact_radius_m,
        allowed_contact_center.x, allowed_contact_center.y,
        allowed_contact_center.z);
    }

    Robot::ConfigurationBlock<1> c_start_filter;
    for (size_t j = 0; j < 6; ++j) {
      c_start_filter[j][0] = current_joints[j];
    }
    Robot::Spheres<1> out_start_filter;
    Robot::template sphere_fk<1>(c_start_filter, out_start_filter);

    std::vector<Point> valid_env_voxels;
    valid_env_voxels.reserve(centervox_cloud.size());

    for (const auto& pt : centervox_cloud) {
      bool is_self_collision = false;
      for (size_t s = 0; s < Robot::n_spheres; ++s) {
        float sx = out_start_filter.x[{s, 0}];
        float sy = out_start_filter.y[{s, 0}];
        float sz = out_start_filter.z[{s, 0}];
        float sr = out_start_filter.r[{s, 0}] + 0.04f;
        float dx = pt[0] - sx, dy = pt[1] - sy, dz = pt[2] - sz;
        if (dx*dx + dy*dy + dz*dz <= sr*sr) {
          is_self_collision = true;
          break;
        }
      }
      if (!is_self_collision) {
        valid_env_voxels.push_back(pt);
      }
    }

    // RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE LOG] Building collision environment with %zu obstacle voxels (cleared %zu self-arm points)...",
    //  valid_env_voxels.size(), centervox_cloud.size() - valid_env_voxels.size());

    if (grasp_motion) {
      const auto entry = vamp_node::nominalFeedbackEntry(current_joints, grasp_feedback_tolerance_rad_);
      if (!entry || !jointsWithinPiperLimits(goal_joints)) {
        RCLCPP_WARN(get_logger(), "[GRASP PATH] Start feedback exceeds tolerance or target exceeds command limits");
        return {};
      }
      // Validate departure from real feedback into command bounds without
      // changing raw feedback/FK or widening IK and RRTC sampling limits.
      float entry_distance = 0.F;
      for (size_t j=0; j<6; ++j) entry_distance=std::max(entry_distance,std::abs((*entry)[j]-current_joints[j]));
      const int entry_steps=std::max(1,static_cast<int>(std::ceil(entry_distance/0.0005F)));
      for (int step=0; step<=entry_steps; ++step) {
        std::vector<float> sample(6);
        for (size_t j=0;j<6;++j) sample[j]=current_joints[j]+((*entry)[j]-current_joints[j])*step/entry_steps;
        if (!isCandidatePoseSafe(sample,valid_env_voxels,nullptr,nullptr,nullptr,0.F,nullptr,-1.F,true,true)) {
          RCLCPP_WARN(get_logger(), "[GRASP PATH] Feedback entry collision: %s", describeCandidateCollision(sample,valid_env_voxels,nullptr,nullptr,nullptr,0.F,nullptr,-1.F,true,false,true).c_str());
          return {};
        }
      }
      using GraspRRTC = vamp::planning::RRTC<PayloadPiper, rake, Robot::resolution>;
      EnvironmentInput empty_input;
      EnvironmentVector empty_environment(empty_input);
      PayloadPiper::ScopedValidator validator([&](const auto &q) {
        return jointsWithinPiperLimits(q) && isCandidatePoseSafe(q, valid_env_voxels,
          nullptr, nullptr, nullptr, 0.F, nullptr, -1.F, true, true);
      });
      if (!PayloadPiper::validator(std::vector<float>(entry->begin(),entry->end())) || !PayloadPiper::validator(goal_joints)) return {};
      Robot::ConfigurationArray start_array, goal_array;
      for (size_t j=0; j<6; ++j) {start_array[j]=(*entry)[j]; goal_array[j]=goal_joints[j];}
      Robot::Configuration start(start_array), goal(goal_array);
      vamp::planning::RRTCSettings settings;
      settings.max_iterations=150000; settings.max_samples=150000; settings.range=.1F;
      auto sampler=std::make_shared<vamp::rng::Halton<PayloadPiper>>();
      ComponentTimer timer(PipelineComponent::RRTC);
      const auto result=GraspRRTC::solve(start, std::vector<Robot::Configuration>{goal},
        empty_environment, settings, sampler);
      std::vector<float> output;
      if (result.path.empty()) return output;
      if (entry_distance > 1.e-5F) output.insert(output.end(),current_joints.begin(),current_joints.end());
      for (const auto &waypoint : result.path) {
        const auto values=waypoint.to_array();
        output.insert(output.end(),values.begin(),values.begin()+6);
      }
      return output;
    }

    // 3. Set the goal directly (Dynamic Cone Algorithm removed in favor of APF)
    std::vector<float> selected_target_joints = goal_joints;

    // 4. Setup VAMP Environment & Solve RRTC Trajectory (Matching original vamp_server.cpp logic but using filtered voxels)
    EnvironmentInput mvt_env;
    mvt_env.mvt_pointclouds.emplace_back(valid_env_voxels, r_min, r_max, ws_min, ws_max, r_point);
    auto env_v = EnvironmentVector(mvt_env);

    auto rng = std::make_shared<vamp::rng::Halton<Robot>>();
    vamp::planning::RRTCSettings rrtc_settings;
    rrtc_settings.max_iterations = 150000;
    rrtc_settings.max_samples = 150000;
    rrtc_settings.range = 0.1; // Reduced step range to prevent large volume sweeps from hitting obstacles

    Robot::ConfigurationArray start_arr;
    Robot::ConfigurationArray goal_arr;
    for (size_t j = 0; j < 6; ++j) {
      start_arr[j] = current_joints[j];
      goal_arr[j] = selected_target_joints[j];
    }
    Robot::Configuration start_cfg(start_arr);
    Robot::Configuration goal_cfg(goal_arr);

    std::vector<Robot::Configuration> valid_goals_vec;

    // Check if the original goal is safe (Environment + Self-collision)
    bool original_safe = isCandidatePoseSafe(selected_target_joints, valid_env_voxels);
    Robot::ConfigurationBlock<rake> block_orig_goal;
    for (size_t i = 0; i < Robot::dimension; ++i) block_orig_goal[i] = goal_cfg.broadcast(i);
    if (original_safe && Robot::fkcc<rake>(env_v, block_orig_goal)) {
        valid_goals_vec.push_back(goal_cfg);
        RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE LOG] Original target pose is safe. Added to candidate goals.");
    } else {
        RCLCPP_WARN(this->get_logger(), "[VAMP ENGINE LOG] Original target pose is IN COLLISION! Generating alternative IK candidates...");
    }

    // Generate deterministic Halton joint-space IK seeds.
    const auto halton = [](int index, int base) {
        float result = 0.0F, factor = 1.0F;
        while (index > 0) { factor /= static_cast<float>(base); result += factor * (index % base); index /= base; }
        return result;
    };
    constexpr std::array<int, 6> halton_bases{{2, 3, 5, 7, 11, 13}};
    for (int i = 1; i <= deterministic_ik_seed_count_; ++i) {
        std::vector<float> seed(6);
        const auto &lower_limits = kPiperJointLower;
        const auto &upper_limits = kPiperJointUpper;
        for (size_t j = 0; j < 6; ++j) {
            float r = halton(i, halton_bases[j]);
            seed[j] = lower_limits[j] + r * (upper_limits[j] - lower_limits[j]);
        }
        
        std::vector<float> ik_out(6);
        if (solvePiperIK(p_goal, p_goal, seed, ik_out)) {
            if (isCandidatePoseSafe(ik_out, valid_env_voxels)) {
                Robot::Configuration cand(ik_out);
                Robot::ConfigurationBlock<rake> b;
                for (size_t d = 0; d < Robot::dimension; ++d) b[d] = cand.broadcast(d);
                if (Robot::fkcc<rake>(env_v, b)) {
                    valid_goals_vec.push_back(cand);
                    RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE LOG] Found valid alternative IK pose #%zu", valid_goals_vec.size());
                }
            }
        }
    }

    if (valid_goals_vec.empty()) {
        RCLCPP_ERROR(this->get_logger(), "[VAMP ENGINE LOG] ALL IK CANDIDATES FAILED VALIDATION! Target is completely blocked.");
        return {};
    }

    // Detailed debug logs for Start configuration validity
    logConfigValidity("Start", start_cfg, centervox_cloud);

    // DEBUG: Check EXACTLY why VAMP validate_vector fails by calling fkcc_debug
    Robot::ConfigurationBlock<rake> block_start;
    for (size_t i = 0; i < Robot::dimension; ++i) block_start[i] = start_cfg.broadcast(i);
    auto debug_res = Robot::fkcc_debug<rake>(env_v, block_start);
    if (!debug_res.second.empty()) {
        RCLCPP_ERROR(this->get_logger(), "[VAMP ENGINE LOG] START POSE IS IN SELF-COLLISION! %zu sphere pairs overlap!", debug_res.second.size());
        for (const auto& pair : debug_res.second) {
            RCLCPP_ERROR(this->get_logger(), "  Sphere %zu overlaps with Sphere %zu", pair.first, pair.second);
        }
        
        RCLCPP_WARN(this->get_logger(), "[VAMP ENGINE LOG] Attempting to auto-nudge start pose out of self-collision...");
        bool found = false;
        for (int i = 0; i < 500; ++i) {
            auto arr = start_cfg.to_array();
            std::vector<float> cand_arr(6);
            for (size_t j = 0; j < 6; ++j) {
                const float fixed_offset = std::sin(
                  static_cast<float>((i + 1) * (j + 2)) * 1.61803398875F);
                cand_arr[j] = arr[j] + fixed_offset * 0.1f; // Max 0.1 rad (~5.7 deg) nudge
            }
            Robot::Configuration cand(cand_arr);
            Robot::ConfigurationBlock<rake> b;
            for (size_t d = 0; d < Robot::dimension; ++d) b[d] = cand.broadcast(d);
            
            if (Robot::fkcc<rake>(env_v, b)) {
                RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE LOG] Found collision-free start pose after %d nudges!", i+1);
                start_cfg = cand;
                for (size_t j = 0; j < 6; ++j) start_arr[j] = cand_arr[j];
                found = true;
                break;
            }
        }
        if (!found) RCLCPP_ERROR(this->get_logger(), "[VAMP ENGINE LOG] Failed to auto-nudge! Planning will likely fail.");
    }
    
    std::vector<Robot::Configuration> final_path;

    for (size_t i = 0; i < valid_goals_vec.size(); ++i) {
        const auto& valid_goal = valid_goals_vec[i];
        std::vector<Robot::Configuration> goals_vec = { valid_goal };
        
        logConfigValidity("Candidate Goal", valid_goal, centervox_cloud);
        
        auto arr = valid_goal.to_array();
        RCLCPP_INFO(this->get_logger(),
          "[VAMP ENGINE LOG] Solving VAMP RRTC trajectory to Candidate %zu: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]...",
          i+1, arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
          
        auto result = [&]() {
          ComponentTimer rrtc_timer(PipelineComponent::RRTC);
          return RRTC::solve(start_cfg, goals_vec, env_v, rrtc_settings, rng);
        }();
        if (!result.path.empty()) {
            vamp::planning::SimplifySettings simplify_settings;
            auto simplify_res = vamp::planning::simplify<Robot, rake, Robot::resolution>(
                result.path, env_v, simplify_settings, rng);
            final_path = simplify_res.path;
            applyElasticBandSmoothing(final_path, valid_env_voxels);
            
            if (!final_path.empty()) {
                RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE LOG] Successfully planned to candidate goal %zu!", i+1);
                break;
            }
        }
    }

    if (final_path.empty()) {
      RCLCPP_ERROR(this->get_logger(), "[VAMP ENGINE LOG] VAMP RRTC failed to find path to ANY candidate goal!");
      return {};
    }

    RCLCPP_INFO(this->get_logger(), "[VAMP ENGINE LOG] RRTC Path Planning Successful! Output Sequence: %zu joint waypoints:", final_path.size());
    std::vector<float> path_out;
    path_out.reserve(final_path.size() * 6);
    size_t wp_idx = 0;
    for (const auto& cfg : final_path) {
      auto arr = cfg.to_array();
      RCLCPP_INFO(this->get_logger(), "  [VAMP OUTPUT SEQUENCE] Waypoint %zu/%zu: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
        ++wp_idx, final_path.size(), arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
      for (size_t j = 0; j < 6; ++j) {
        path_out.push_back(arr[j]);
      }
    }
    return path_out;
  }
};

} // namespace vamp_node

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vamp_node::VampPlannerServer)

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vamp_node::VampPlannerServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
