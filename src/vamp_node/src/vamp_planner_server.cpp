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
#include "vamp_node/gpd_grasp_adapter.hpp"
#include "vamp_node/hggd_mcu_adapter.hpp"
#include "vamp_node/gpd_diversity.hpp"
#include "vamp_node/analytical_grasp_generator.hpp"
#include "vamp_node/deterministic_ik.hpp"
#include "vamp_node/cartesian_grasp_approach.hpp"
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
#include "my_actions/srv/refine_grasp.hpp"
#include "my_actions/srv/plan_cartesian_grasp_approach.hpp"
#include "my_actions/srv/plan_placement.hpp"
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
    float best_position_error = std::numeric_limits<float>::infinity();
    float best_rotation_error = std::numeric_limits<float>::infinity();

    void keepBest(const PoseIkDiagnostics &other) {
      attempts += other.attempts;
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
  double pregrasp_offset_ = 0.20;
  double pregrasp_position_tolerance_m_ = 0.008;
  double pregrasp_orientation_tolerance_degrees_ = 20.0;
  double gpd_scene_crop_radius_m_ = 0.20;
  double gpd_target_sample_radius_m_ = 0.14;
  double target_region_neighbor_distance_m_ = 0.020;
  double target_foreground_band_m_ = 0.030;
  double target_collision_clearance_m_ = 0.030;
  bool enable_grasp_collision_check_ = true;
  bool enable_gpd_score_filter_ = true;
  bool require_gpd_full_antipodal_ = false;
  bool require_measured_antipodal_contacts_ = true;
  bool allow_pregrasp_collision_candidates_ = false;
  double antipodal_normal_cosine_ = 0.90;
  double antipodal_normal_radius_m_ = 0.008;
  int antipodal_min_normal_neighbors_ = 10;
  double antipodal_seed_search_radius_m_ = 0.040;
  int antipodal_maximum_local_points_ = 512;
  double grasp_contact_depth_m_ = 0.060;
  double pregrasp_opening_clearance_m_ = 0.015;
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
  // GPD reports the bottom of its finger closing region.  Piper's
  // gripper_base is 61.5 mm behind the beginning of the physical fingers.
  double gpd_tcp_offset_m_ = -0.0615;
  double gpd_min_score_ = 0.0;
  int gpd_visualization_limit_ = 100;
  double gpd_diversity_position_m_ = 0.015;
  double gpd_diversity_approach_degrees_ = 15.0;
  double gpd_diversity_closing_degrees_ = 15.0;
  std::string gpd_camera_frame_;
  std::string gpd_config_file_;
  std::string grasp_candidate_backend_ = "hggd";
  AnalyticalGraspParameters analytical_parameters_;
  double analytical_accepted_candidate_separation_m_ = 0.025;
  std::string analytical_debug_log_directory_;
  std::unique_ptr<AnalyticalGraspGenerator> analytical_generator_;
  std::unique_ptr<GpdGraspAdapter> gpd_adapter_;
  std::string gpd_initialization_error_;
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
  float placement_position_tolerance_m_ = 0.015F;
  float placement_orientation_tolerance_rad_ = 0.35F;
  float placement_preplace_offset_m_ = 0.10F;

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
  rclcpp::Service<my_actions::srv::RefineGrasp>::SharedPtr refine_grasp_srv_;
  rclcpp::Service<my_actions::srv::PlanCartesianGraspApproach>::SharedPtr
    plan_cartesian_grasp_approach_srv_;
  rclcpp::Service<my_actions::srv::PlanPlacement>::SharedPtr plan_placement_srv_;
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
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gpd_proposals_markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr analytical_proposals_markers_pub_;
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
    pregrasp_offset_ = this->declare_parameter<double>("pregrasp_offset_m", 0.20);
    pregrasp_position_tolerance_m_ = this->declare_parameter<double>(
      "pregrasp_position_tolerance_m", 0.008);
    pregrasp_orientation_tolerance_degrees_ = this->declare_parameter<double>(
      "pregrasp_orientation_tolerance_degrees", 20.0);
    gpd_scene_crop_radius_m_ = this->declare_parameter<double>("gpd_scene_crop_radius_m", 0.20);
    gpd_target_sample_radius_m_ = this->declare_parameter<double>("gpd_target_sample_radius_m", 0.14);
    target_region_neighbor_distance_m_ = this->declare_parameter<double>(
      "target_region_neighbor_distance_m", 0.020);
    target_foreground_band_m_ = this->declare_parameter<double>(
      "target_foreground_band_m", 0.030);
    target_collision_clearance_m_ = this->declare_parameter<double>(
      "target_collision_clearance_m", 0.030);
    enable_grasp_collision_check_ = this->declare_parameter<bool>(
      "enable_grasp_collision_check", true);
    enable_gpd_score_filter_ = this->declare_parameter<bool>(
      "enable_gpd_score_filter", true);
    require_gpd_full_antipodal_ = this->declare_parameter<bool>(
      "require_gpd_full_antipodal", false);
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
      "pregrasp_opening_clearance_m", 0.015);
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
    pregrasp_offset_ = std::clamp(pregrasp_offset_, 0.02, 0.30);
    pregrasp_position_tolerance_m_ = std::clamp(
      pregrasp_position_tolerance_m_, 0.004, 0.015);
    pregrasp_orientation_tolerance_degrees_ = std::clamp(
      pregrasp_orientation_tolerance_degrees_, 0.0, 30.0);
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
    gpd_tcp_offset_m_ = this->declare_parameter<double>("gpd_tcp_offset_m", -0.0615);
    gpd_min_score_ = this->declare_parameter<double>("gpd_min_score", 0.0);
    gpd_visualization_limit_ = this->declare_parameter<int>("gpd_visualization_limit", 100);
    gpd_diversity_position_m_ = this->declare_parameter<double>(
      "gpd_diversity_position_m", 0.015);
    gpd_diversity_approach_degrees_ = this->declare_parameter<double>(
      "gpd_diversity_approach_degrees", 15.0);
    gpd_diversity_closing_degrees_ = this->declare_parameter<double>(
      "gpd_diversity_closing_degrees", 15.0);
    gpd_camera_frame_ = this->declare_parameter<std::string>("gpd_camera_frame", "");
    if (!gpd_camera_frame_.empty()) {
      RCLCPP_WARN(
        this->get_logger(),
        "gpd_camera_frame is ignored: grasp transformations require the "
        "synchronized manual FK * calibrated-X observation.");
    }
    gpd_config_file_ = this->declare_parameter<std::string>("gpd_config_file", "");
    grasp_candidate_backend_ = this->declare_parameter<std::string>(
      "grasp_candidate_backend", "hggd");
    std::transform(grasp_candidate_backend_.begin(), grasp_candidate_backend_.end(),
      grasp_candidate_backend_.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value)); });
    if (grasp_candidate_backend_ != "analytical" &&
        grasp_candidate_backend_ != "gpd" && grasp_candidate_backend_ != "hybrid" &&
        grasp_candidate_backend_ != "hggd") {
      RCLCPP_WARN(this->get_logger(), "Unknown grasp_candidate_backend '%s'; using gpd.",
        grasp_candidate_backend_.c_str());
      grasp_candidate_backend_ = "hggd";
    }
    analytical_parameters_.maximum_proposals = this->declare_parameter<int>(
      "analytical_maximum_proposals", 300);
    analytical_parameters_.voxel_size_m = static_cast<float>(this->declare_parameter<double>(
      "analytical_voxel_size_m", 0.005));
    analytical_parameters_.roll_step_degrees = static_cast<float>(this->declare_parameter<double>(
      "analytical_roll_step_degrees", 15.0));
    analytical_parameters_.normal_radius_m = static_cast<float>(this->declare_parameter<double>(
      "analytical_normal_radius_m", 0.008));
    analytical_parameters_.min_normal_neighbors = this->declare_parameter<int>(
      "analytical_min_normal_neighbors", 10);
    analytical_parameters_.dedup_position_m = static_cast<float>(this->declare_parameter<double>(
      "analytical_dedup_position_m", 0.010));
    analytical_parameters_.dedup_angle_degrees = static_cast<float>(this->declare_parameter<double>(
      "analytical_dedup_angle_degrees", 10.0));
    analytical_parameters_.min_contact_center_separation_m = static_cast<float>(
      this->declare_parameter<double>("analytical_min_contact_center_separation_m", 0.025));
    analytical_parameters_.local_pca_searches = this->declare_parameter<int>(
      "analytical_local_pca_searches", 50);
    analytical_parameters_.global_pca_searches = this->declare_parameter<int>(
      "analytical_global_pca_searches", 50);
    analytical_parameters_.maximum_approach_variants_per_pair = this->declare_parameter<int>(
      "analytical_maximum_approach_variants_per_pair", 2);
    analytical_accepted_candidate_separation_m_ = this->declare_parameter<double>(
      "analytical_accepted_candidate_separation_m", 0.025);
    analytical_debug_log_directory_ = this->declare_parameter<std::string>(
      "analytical_debug_log_directory", "");
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
    final_approach_max_joint_step_rad_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "final_approach_max_joint_step_rad", 0.08), 0.01, 0.25));
    final_approach_contact_phase_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "final_approach_contact_phase_m", 0.006), 0.0, 0.010));
    placement_position_tolerance_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "placement_position_tolerance_m", 0.015), 0.005, 0.025));
    placement_orientation_tolerance_rad_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "placement_orientation_tolerance_rad", 0.35), 0.10, 0.60));
    placement_preplace_offset_m_ = static_cast<float>(std::clamp(
      this->declare_parameter<double>(
        "placement_preplace_offset_m", 0.10), 0.05, 0.20));
    grasp_observation_cache_size_ = static_cast<size_t>(std::max<int64_t>(
      2, this->declare_parameter<int>("grasp_observation_cache_size", 1000)));
    analytical_parameters_.maximum_proposals = std::clamp(
      analytical_parameters_.maximum_proposals, 100, 500);
    maximum_candidates_to_test_ = std::clamp(maximum_candidates_to_test_, 1, 2000);
    maximum_ranking_pool_candidates_ = std::clamp(maximum_ranking_pool_candidates_, 1, 1000);
    maximum_valid_candidates_ = std::clamp(maximum_valid_candidates_, 1, 100);
    gpd_visualization_limit_ = std::clamp(gpd_visualization_limit_, 1, 100);
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
    gpd_diversity_position_m_ = std::clamp(gpd_diversity_position_m_, 0.001, 0.100);
    gpd_diversity_approach_degrees_ = std::clamp(gpd_diversity_approach_degrees_, 1.0, 90.0);
    gpd_diversity_closing_degrees_ = std::clamp(gpd_diversity_closing_degrees_, 1.0, 90.0);
    analytical_parameters_.voxel_size_m = std::clamp(
      analytical_parameters_.voxel_size_m, 0.002F, 0.020F);
    analytical_parameters_.roll_step_degrees = std::clamp(
      analytical_parameters_.roll_step_degrees, 5.0F, 45.0F);
    analytical_parameters_.normal_radius_m = std::clamp(
      analytical_parameters_.normal_radius_m, 0.004F, 0.030F);
    analytical_parameters_.min_normal_neighbors = std::clamp(
      analytical_parameters_.min_normal_neighbors, 6, 100);
    analytical_parameters_.dedup_position_m = std::clamp(
      analytical_parameters_.dedup_position_m, 0.002F, 0.050F);
    analytical_parameters_.dedup_angle_degrees = std::clamp(
      analytical_parameters_.dedup_angle_degrees, 1.0F, 45.0F);
    analytical_parameters_.min_contact_center_separation_m = std::clamp(
      analytical_parameters_.min_contact_center_separation_m, 0.005F, 0.100F);
    analytical_parameters_.local_pca_searches = std::clamp(
      analytical_parameters_.local_pca_searches, 1, 100);
    analytical_parameters_.global_pca_searches = std::clamp(
      analytical_parameters_.global_pca_searches, 1, 100);
    analytical_parameters_.maximum_approach_variants_per_pair = std::clamp(
      analytical_parameters_.maximum_approach_variants_per_pair, 1, 5);
    analytical_accepted_candidate_separation_m_ = std::clamp(
      analytical_accepted_candidate_separation_m_, 0.005, 0.100);
    deterministic_ik_seed_count_ = std::clamp(deterministic_ik_seed_count_, 1, 128);
    if (deterministic_ik_home_.size() != 6) {
      RCLCPP_WARN(this->get_logger(), "deterministic_ik_home_joint_state must have six values; using zero home.");
      deterministic_ik_home_ = {0.0, 0.8, -1.4, 0.0, 0.0, 0.0};
    }
    analytical_parameters_.minimum_opening_m = static_cast<float>(minimum_gripper_width_);
    analytical_parameters_.maximum_opening_m = static_cast<float>(maximum_gripper_width_);
    analytical_parameters_.contact_depth_m = static_cast<float>(grasp_contact_depth_m_);
    analytical_parameters_.finger_length_m = static_cast<float>(finger_length_m_);
    analytical_parameters_.finger_width_m = static_cast<float>(finger_width_m_);
    analytical_parameters_.finger_height_m = static_cast<float>(finger_height_m_);
    analytical_generator_ = std::make_unique<AnalyticalGraspGenerator>(analytical_parameters_);
    if (grasp_candidate_backend_ == "gpd" || grasp_candidate_backend_ == "hybrid") {
      try {
      if (gpd_config_file_.empty()) {
        gpd_config_file_ = ament_index_cpp::get_package_share_directory("gpd_vendor") +
          "/config/piper_eigen_params.cfg";
      }
      gpd_adapter_ = std::make_unique<GpdGraspAdapter>(gpd_config_file_);
      } catch (const std::exception &error) {
        gpd_initialization_error_ = error.what();
        RCLCPP_ERROR(this->get_logger(), "GPD initialization failed: %s", error.what());
      }
    }
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
    gpd_proposals_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vamp_planner/gpd_proposals_markers", visual_qos);
    analytical_proposals_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vamp_planner/analytical_proposals_markers", visual_qos);
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
        "/anyplace/artificial_bottom_cloud",
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
    add(std::clamp(static_cast<float>(pregrasp_offset_), 0.10F, 0.20F));
    for (const float distance : {0.15F, 0.175F, 0.125F, 0.20F, 0.10F}) {
      add(distance);
    }
    return distances;
  }

  static bool finitePoint(const Point &p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
  }

  bool refineHggdOpeningFromTarget(
      const std::vector<Eigen::Vector3f> &target_points,
      GraspProposal &proposal) const {
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
    std::vector<float> lateral_coordinates;
    lateral_coordinates.reserve(target_points.size());
    for (const auto &point : target_points) {
      const Eigen::Vector3f relative = point - proposal.sample;
      if (std::abs(relative.dot(approach)) > depth_half_extent ||
          std::abs(relative.dot(height)) > height_half_extent) {
        continue;
      }
      const float lateral = relative.dot(closing);
      if (std::abs(lateral) <= lateral_extent) {
        lateral_coordinates.push_back(lateral);
      }
    }
    if (lateral_coordinates.size() <
        static_cast<size_t>(antipodal_min_normal_neighbors_)) {
      return false;
    }
    std::sort(lateral_coordinates.begin(), lateral_coordinates.end());
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
    for (size_t index = 1; index < lateral_coordinates.size(); ++index) {
      if (lateral_coordinates[index] - lateral_coordinates[index - 1] >
          maximum_material_gap_m) {
        intervals.push_back({begin, index - 1});
        begin = index;
      }
    }
    intervals.push_back({begin, lateral_coordinates.size() - 1});

    const auto nearest = std::min_element(
      lateral_coordinates.begin(), lateral_coordinates.end(),
      [](const float left, const float right) {
        return std::abs(left) < std::abs(right);
      });
    const size_t nearest_index = static_cast<size_t>(
      std::distance(lateral_coordinates.begin(), nearest));
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

    const float local_minimum = lateral_coordinates[local_interval.begin];
    const float local_maximum = lateral_coordinates[local_interval.end];
    const float local_opening = local_maximum - local_minimum;
    float largest_adjacent_gap = 0.0F;
    if (selected_interval_index > 0) {
      largest_adjacent_gap = std::max(
        largest_adjacent_gap,
        local_minimum - lateral_coordinates[
          intervals[selected_interval_index - 1].end]);
    }
    if (selected_interval_index + 1 < intervals.size()) {
      largest_adjacent_gap = std::max(
        largest_adjacent_gap,
        lateral_coordinates[intervals[selected_interval_index + 1].begin] -
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
    const float minimum_lateral = lateral_coordinates[interval_begin + trim];
    const float maximum_lateral = lateral_coordinates[interval_end - trim];
    const float measured_opening = maximum_lateral - minimum_lateral;
    if (!std::isfinite(measured_opening) ||
        measured_opening < static_cast<float>(minimum_gripper_width_) ||
        measured_opening > static_cast<float>(maximum_gripper_width_)) {
      return false;
    }

    const float center_shift = 0.5F * (minimum_lateral + maximum_lateral);
    const Eigen::Vector3f shift = closing * center_shift;
    proposal.sample += shift;
    proposal.hand_base += shift;
    proposal.opening_m = measured_opening;
    proposal.contact_left = proposal.sample + closing * (0.5F * measured_opening);
    proposal.contact_right = proposal.sample - closing * (0.5F * measured_opening);
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
      GraspProposal &proposal) const {
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
    Eigen::Vector3f object_centroid = Eigen::Vector3f::Zero();
    std::vector<float> target_heights;
    target_heights.reserve(target_points.size());
    for (const auto &point : target_points) {
      object_centroid += point;
      target_heights.push_back(point.z());
    }
    object_centroid /= static_cast<float>(target_points.size());
    std::sort(target_heights.begin(), target_heights.end());
    const float robust_top_z = target_heights[static_cast<std::size_t>(
      0.95F * static_cast<float>(target_heights.size() - 1))];
    constexpr float rim_top_band_m = 0.012F;
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
        if (!refineHggdOpeningFromTarget(target_points, candidate)) return;
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
            (!seed_is_at_top_edge ||
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
        static_cast<float>(pregrasp_offset_), 0.10F, 0.20F);
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
                    static_cast<float>(finger_height_m_ * 0.5 +
                      target_contact_surface_tolerance_m_)) {
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
    if (silhouette_pair != nullptr &&
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
    Eigen::Vector3f radial(
      proposal.sample.x() - object_centroid.x(),
      proposal.sample.y() - object_centroid.y(), 0.0F);
    if (radial.squaredNorm() > 1.0e-8F) {
      add_hypothesis(
        -Eigen::Vector3f::UnitZ(), radial,
        ZeroPoseHypothesis::Mode::kRim,
        seed_is_at_top_edge ? 0.0F : 1.0F);
    }

    if (hypotheses.empty()) return false;
    const auto mode_available = [&hypotheses](ZeroPoseHypothesis::Mode mode) {
        return std::any_of(
          hypotheses.begin(), hypotheses.end(),
          [mode](const auto &hypothesis) {
            return hypothesis.mode == mode && hypothesis.cost < 1.0F;
          });
      };
    // Choose by observed topology, then compare costs only within that mode.
    // A 10 mm rim must never beat a valid 90 mm body grasp merely because its
    // numerical opening is smaller.
    ZeroPoseHypothesis::Mode preferred_mode =
      ZeroPoseHypothesis::Mode::kSide;
    if (seed_is_at_top_edge &&
        mode_available(ZeroPoseHypothesis::Mode::kTopSurface)) {
      preferred_mode = ZeroPoseHypothesis::Mode::kTopSurface;
    } else if (seed_is_at_top_edge &&
        mode_available(ZeroPoseHypothesis::Mode::kRim)) {
      preferred_mode = ZeroPoseHypothesis::Mode::kRim;
    } else if (silhouette_pair != nullptr &&
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
    auto best = hypotheses.end();
    for (auto candidate = hypotheses.begin(); candidate != hypotheses.end();
         ++candidate) {
      if (candidate->mode != preferred_mode) continue;
      if (best == hypotheses.end() || candidate->cost < best->cost) {
        best = candidate;
      }
    }
    if (best == hypotheses.end()) return false;
    best->grasp.local_dominant_axis = dominant;
    best->grasp.has_local_dominant_axis = true;
    proposal = best->grasp;
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
        if (!pointAtPixel(cloud, start_u, start_v, start)) {
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
          if (!pointAtPixel(cloud, u, v, current)) continue;
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
            if (!pointAtPixel(cloud, next_u, next_v, next)) continue;
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

    // A grounding rectangle for a handle necessarily includes the empty loop.
    // Its registered depth is normally the table behind the handle.  The old
    // support/(seed-distance) score therefore selected the table because it
    // has many more pixels than the thin handle.  Select a meaningful
    // foreground layer first, then choose its best-supported component rather
    // than blindly accepting the closest depth fragment. This also remains
    // deterministic when the VLM
    // point lands in the handle hole rather than exactly on material.
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
          component->pixels.size() < required_component_pixels ||
          component->median_camera_range >
            nearest_supported_range + foreground_layer_tolerance_m) {
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
      const float score = 2.0F * support - 0.75F * seed_distance -
        0.25F * depth_offset;
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
    for (const auto &component : components) {
      if (!std::isfinite(component.median_camera_range) ||
          std::abs(component.median_camera_range - seeded_surface_range) >
            static_cast<float>(target_foreground_band_m_)) {
        continue;
      }
      for (const size_t index : component.pixels) filtered.data[index] = 255;
      point_count += component.pixels.size();
    }
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
      const std::string &namespace_prefix = "grasp_candidate_") {
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
      Eigen::Vector3f approach_axis(
        static_cast<float>(candidate.grasp_pose.pose.position.x - candidate.pregrasp_pose.pose.position.x),
        static_cast<float>(candidate.grasp_pose.pose.position.y - candidate.pregrasp_pose.pose.position.y),
        static_cast<float>(candidate.grasp_pose.pose.position.z - candidate.pregrasp_pose.pose.position.z));
      if (approach_axis.squaredNorm() < 1e-8F) {
        approach_axis = robot_orientation.toRotationMatrix().col(2);
      } else {
        approach_axis.normalize();
      }
      Eigen::Vector3f closing_axis(
        static_cast<float>(candidate.contact_left.x - candidate.contact_right.x),
        static_cast<float>(candidate.contact_left.y - candidate.contact_right.y),
        static_cast<float>(candidate.contact_left.z - candidate.contact_right.z));
      if (closing_axis.squaredNorm() < 1e-8F) {
        closing_axis = robot_orientation.toRotationMatrix().col(1);
      } else {
        closing_axis.normalize();
      }
      const Eigen::Quaternionf marker_orientation(
        visualGripperRotation(closing_axis, approach_axis));
      const Eigen::Vector3f tcp_position(
        static_cast<float>(candidate.grasp_pose.pose.position.x),
        static_cast<float>(candidate.grasp_pose.pose.position.y),
        static_cast<float>(candidate.grasp_pose.pose.position.z));
      const Eigen::Vector3f hand_base = tcp_position -
        approach_axis * static_cast<float>(gpd_tcp_offset_m_);
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
        finger.color.r = color[0]; finger.color.g = color[1]; finger.color.b = color[2]; finger.color.a = 0.55f;
        array.markers.push_back(finger);

        visualization_msgs::msg::Marker marker = finger;
        marker.id = static_cast<int>(candidate.id * 100 + 10 + contact);
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.pose.orientation.w = 1.0;
        marker.pose.position = point;
        marker.scale.x = marker.scale.y = marker.scale.z = 0.012;
        marker.color.r = 1.0f; marker.color.g = 0.2f; marker.color.b = 0.1f; marker.color.a = 1.0f;
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
          pregrasp_approach * static_cast<float>(gpd_tcp_offset_m_);
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
      approach_marker.color.r = color[0]; approach_marker.color.g = color[1]; approach_marker.color.b = color[2]; approach_marker.color.a = 1.0f;
      array.markers.push_back(approach_marker);
      visualization_msgs::msg::Marker label;
      label.header = candidate.grasp_pose.header;
      label.ns = candidate_ns;
      label.action = visualization_msgs::msg::Marker::ADD;
      label.pose = candidate.grasp_pose.pose;
      label.id = static_cast<int>(candidate.id * 100 + 21);
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.pose.orientation.w = 1.0;
      label.pose.position.z += 0.045;
      label.scale.z = 0.025;
      label.color.r = color[0]; label.color.g = color[1]; label.color.b = color[2]; label.color.a = 1.0f;
      label.text = "#" + std::to_string(candidate.id);
      array.markers.push_back(label);
    }
    (publisher ? publisher : grasp_markers_pub_)->publish(array);
  }

  void writeAnalyticalDebugLog(
      const rclcpp::Time &stamp, const Eigen::Vector3f &seed,
      const AnalyticalGraspDiagnostics &diagnostics,
      const std::vector<my_actions::msg::GraspCandidate> &valid_pool,
      const std::vector<my_actions::msg::GraspCandidate> &selected_candidates,
      const std::unordered_map<uint32_t, std::string> &candidate_origins) {
    if (grasp_candidate_backend_ != "analytical" && grasp_candidate_backend_ != "hybrid") return;
    std::filesystem::path directory = analytical_debug_log_directory_;
    if (directory.empty()) {
      const char *home = std::getenv("HOME");
      directory = home == nullptr ? std::filesystem::current_path() :
        std::filesystem::path(home) / "analytical_grasp_debug";
    }
    try {
      std::filesystem::create_directories(directory);
      const auto path = directory / ("grasp_" + std::to_string(stamp.nanoseconds()) + ".json");
      std::ofstream output(path);
      if (!output) throw std::runtime_error("cannot open output file");
      output << std::fixed << std::setprecision(6);
      output << "{\n  \"schema_version\": 1,\n";
      output << "  \"observation_stamp_ns\": " << stamp.nanoseconds() << ",\n";
      output << "  \"backend\": \"" << grasp_candidate_backend_ << "\",\n";
      output << "  \"seed_base_link\": [" << seed.x() << ", " << seed.y() << ", " << seed.z() << "],\n";
      output << "  \"generation\": {\"isolated_points\": " << diagnostics.isolated_points
             << ", \"downsampled_points\": " << diagnostics.downsampled_points
             << ", \"normal_points\": " << diagnostics.normal_points
             << ", \"local_pairs\": " << diagnostics.local_contact_pairs
             << ", \"global_pairs\": " << diagnostics.global_contact_pairs
             << ", \"selected_local_families\": " << diagnostics.selected_local_families
             << ", \"selected_global_families\": " << diagnostics.selected_global_families
             << ", \"generated_proposals\": " << diagnostics.generated << "},\n";
      output << "  \"valid_candidates\": [\n";
      for (std::size_t index = 0; index < valid_pool.size(); ++index) {
        const auto &candidate = valid_pool[index];
        const auto origin = candidate_origins.find(candidate.id);
        output << "    {\"id\": " << candidate.id
               << ", \"grasp_position\": [" << candidate.grasp_pose.pose.position.x << ", "
               << candidate.grasp_pose.pose.position.y << ", " << candidate.grasp_pose.pose.position.z << "]"
               << ", \"contact_left\": [" << candidate.contact_left.x << ", "
               << candidate.contact_left.y << ", " << candidate.contact_left.z << "]"
               << ", \"contact_right\": [" << candidate.contact_right.x << ", "
               << candidate.contact_right.y << ", " << candidate.contact_right.z << "]"
               << ", \"jaw_opening_m\": " << candidate.jaw_opening_m
               << ", \"origin\": \""
               << (origin == candidate_origins.end() ? "unknown" : origin->second) << "\""
               << ", \"proposal_quality\": " << candidate.score
               << ", \"ranking_score\": " << candidate.ranking_score
               << ", \"score_components\": {\"proposal_quality\": " << candidate.score_components[0]
               << ", \"normalized_quality\": " << candidate.score_components[1]
               << ", \"wide_opening_penalty\": " << candidate.score_components[2]
               << ", \"vertical_closing_penalty\": " << candidate.score_components[3]
               << ", \"seed_distance_penalty\": " << candidate.score_components[4]
               << ", \"vertical_approach_penalty\": " << candidate.score_components[5]
               << ", \"joint_travel_penalty\": " << candidate.score_components[6]
               << ", \"joint_margin_reward\": " << candidate.score_components[7]
               << ", \"pregrasp_override_penalty\": " << candidate.score_components[8]
               << ", \"corner_penalty\": " << candidate.score_components[9]
               << ", \"camera_approach_penalty\": " << candidate.score_components[10] << "}"
               << ", \"validation_status\": \"" << candidate.validation_status << "\"}"
               << (index + 1 == valid_pool.size() ? "\n" : ",\n");
      }
      output << "  ],\n  \"selected_candidate_ids\": [";
      for (std::size_t index = 0; index < selected_candidates.size(); ++index) {
        output << selected_candidates[index].id << (index + 1 == selected_candidates.size() ? "" : ", ");
      }
      output << "]\n}\n";
      RCLCPP_INFO(this->get_logger(), "[ANALYTICAL DEBUG] Wrote %s", path.c_str());
    } catch (const std::exception &error) {
      RCLCPP_WARN(this->get_logger(), "[ANALYTICAL DEBUG] Log write failed: %s", error.what());
    }
  }

  struct HandleOrientation {
    Eigen::Vector3f approach = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f closing = Eigen::Vector3f::UnitY();
    float roll_alignment_cost = 1.0F;
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
                                         std::array<float, 3> *blocked_local = nullptr) const {
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
    const float depth_behind_contact = static_cast<float>(
      grasp_contact_depth_m_);
    const float depth_ahead_contact = static_cast<float>(
      finger_length_m_ - grasp_contact_depth_m_);
    struct BlockingPoint {
      Eigen::Vector3f point = Eigen::Vector3f::Zero();
      std::array<float, 3> local{};
    };
    std::vector<BlockingPoint> blocking_points;
    blocking_points.reserve(64);
    for (const auto &point : points) {
      const Eigen::Vector3f local = point - midpoint;
      const float height = std::abs(local.dot(height_axis));
      const float depth = local.dot(approach);
      if (height > half_height || depth < -depth_behind_contact ||
          depth > depth_ahead_contact) continue;
      const float absolute_side = std::abs(local.dot(closing));
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
        refinement.approach_axis * static_cast<float>(gpd_tcp_offset_m_);
      Eigen::Isometry3f target_pose = Eigen::Isometry3f::Identity();
      target_pose.translation() = tcp;
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

    if (request->candidates.empty()) {
      response->status = "PLACEMENT_PLAN_NO_CANDIDATES";
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

    sensor_msgs::msg::PointCloud2::SharedPtr source_cloud;
    {
      std::lock_guard<std::mutex> lock(segmented_target_mutex_);
      source_cloud = latest_segmented_target_cloud_;
    }
    if (!source_cloud || source_cloud->header.frame_id != "base_link" ||
      source_cloud->width == 0) {
      response->status = "PLACEMENT_SOURCE_OBJECT_CLOUD_UNAVAILABLE";
      return;
    }
    std::vector<Eigen::Vector3f> source_points;
    sensor_msgs::PointCloud2ConstIterator<float> source_x(*source_cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> source_y(*source_cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> source_z(*source_cloud, "z");
    for (; source_x != source_x.end(); ++source_x, ++source_y, ++source_z) {
      if (std::isfinite(*source_x) && std::isfinite(*source_y) &&
        std::isfinite(*source_z)) {
        source_points.emplace_back(*source_x, *source_y, *source_z);
      }
    }
    if (source_points.size() < 20) {
      response->status = "PLACEMENT_SOURCE_OBJECT_CLOUD_TOO_SMALL";
      return;
    }

    // The object is attached after the cached grasp, so its source voxels must
    // not remain as static obstacles during placement IK checks. Remove only
    // voxels close to the segmented source object; all other scene geometry
    // remains blocking.
    constexpr float source_voxel_m = 0.015F;
    std::unordered_set<std::tuple<int, int, int>, VoxelHash> source_voxels;
    for (const auto &point : source_points) {
      source_voxels.emplace(
        static_cast<int>(std::floor(point.x() / source_voxel_m)),
        static_cast<int>(std::floor(point.y() / source_voxel_m)),
        static_cast<int>(std::floor(point.z() / source_voxel_m)));
    }
    std::vector<Point> environment;
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
    };
    std::optional<FeasiblePlacement> best;
    int rejected_pose = 0;
    int rejected_ik = 0;
    int rejected_collision = 0;
    int rejected_preplace = 0;
    for (size_t index = 0; index < request->candidates.size(); ++index) {
      const auto &candidate = request->candidates[index];
      if (candidate.object_pose.header.frame_id != "base_link") {
        ++rejected_pose;
        continue;
      }
      Eigen::Isometry3f centered_object_pose;
      if (!pose_from_message(candidate.object_pose, centered_object_pose)) {
        ++rejected_pose;
        continue;
      }
      Eigen::Isometry3f object_delta = Eigen::Isometry3f::Identity();
      object_delta.linear() = centered_object_pose.linear();
      object_delta.translation() = centered_object_pose.translation() -
        centered_object_pose.linear() * source_center;
      const Eigen::Isometry3f desired = object_delta * grasp_pose;
      PoseIkDiagnostics ik_diagnostics;
      std::vector<float> placement_joints;
      if (!solvePiperPoseIK(
          desired, grasp_joints, placement_joints, ik_diagnostics, false,
          placement_position_tolerance_m_,
          placement_orientation_tolerance_rad_, 1000)) {
        ++rejected_ik;
        continue;
      }
      if (enable_grasp_collision_check_ &&
        !isCandidatePoseSafe(placement_joints, environment)) {
        ++rejected_collision;
        continue;
      }
      std::array<float, 6> placement_array{};
      std::copy_n(placement_joints.begin(), 6, placement_array.begin());
      const Eigen::Isometry3f resolved = Robot::eefk(placement_array);
      const float position_error =
        (resolved.translation() - desired.translation()).norm();
      const float rotation_error = std::abs(Eigen::AngleAxisf(
        desired.linear() * resolved.linear().transpose()).angle());

      Eigen::Isometry3f preplace_desired = resolved;
      preplace_desired.translation().z() += placement_preplace_offset_m_;
      PoseIkDiagnostics preplace_diagnostics;
      std::vector<float> preplace_joints;
      if (!solvePiperPoseIK(
          preplace_desired, placement_joints, preplace_joints,
          preplace_diagnostics, false,
          placement_position_tolerance_m_ + 0.005F,
          placement_orientation_tolerance_rad_ + 0.10F, 800) ||
        (enable_grasp_collision_check_ &&
          !isCandidatePoseSafe(preplace_joints, environment))) {
        ++rejected_preplace;
        continue;
      }
      std::array<float, 6> preplace_array{};
      std::copy_n(preplace_joints.begin(), 6, preplace_array.begin());
      const Eigen::Isometry3f resolved_preplace = Robot::eefk(preplace_array);
      float joint_travel = 0.0F;
      for (size_t joint = 0; joint < 6; ++joint) {
        const float difference = placement_joints[joint] - grasp_joints[joint];
        joint_travel += difference * difference;
      }
      const float cost = -candidate.score + 8.0F * position_error +
        0.25F * rotation_error + 0.02F * std::sqrt(joint_travel);
      if (!best || cost < best->cost) {
        best = FeasiblePlacement{
          index, desired, resolved, resolved_preplace,
          resolved * grasp_pose.inverse(), placement_joints,
          preplace_joints, position_error, rotation_error, cost};
      }
    }
    if (!best) {
      std::ostringstream status;
      status << "PLACEMENT_NO_REACHABLE_CANDIDATE tested="
             << request->candidates.size() << " rejected={pose:"
             << rejected_pose << ",ik:" << rejected_ik << ",collision:"
             << rejected_collision << ",preplace:" << rejected_preplace << "}";
      response->status = status.str();
      RCLCPP_WARN(this->get_logger(), "%s", response->status.c_str());
      return;
    }

    response->selected_candidate = request->candidates[best->index];
    pose_to_message(best->resolved, response->placement_ee_pose);
    pose_to_message(best->preplace, response->preplace_ee_pose);
    std::copy_n(best->joints.begin(), 6, response->placement_joints.begin());
    std::copy_n(
      best->preplace_joints.begin(), 6, response->preplace_joints.begin());
    response->position_error_m = best->position_error;
    response->orientation_error_rad = best->rotation_error;

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
    placement_plan_markers_pub_->publish(markers);

    std::ostringstream status;
    status << std::fixed << std::setprecision(1)
           << "PLACEMENT_PLAN_READY candidate=#"
           << response->selected_candidate.id << " tested="
           << request->candidates.size() << " policy="
           << request->orientation_policy << " error={position:"
           << best->position_error * 1000.0F << "mm,orientation:"
           << best->rotation_error * 180.0F / static_cast<float>(M_PI)
           << "deg} preplace=" << placement_preplace_offset_m_ * 1000.0F
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

    std::vector<Point> environment;
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      environment = getFilteredEnvironment(request->current_joints);
    }
    std::vector<uint8_t> target_flags(environment.size(), 0);
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

    const float approach_distance =
      (grasp_pose.translation() - current_pose.translation()).norm();
    const float bounded_step =
      std::min(final_approach_translation_step_m_, 0.002F);
    const std::size_t approach_waypoint_count = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(approach_distance / bounded_step)));
    const float actual_step = approach_distance /
      static_cast<float>(approach_waypoint_count);

    const auto plan = planCartesianApproach(
      current_pose, grasp_pose, request->current_joints,
      final_approach_translation_step_m_, final_approach_max_joint_step_rad_,
      [this](const Eigen::Isometry3f &waypoint,
             const std::vector<float> &seed, std::vector<float> &solution) {
        PoseIkDiagnostics diagnostics;
        // This final segment is deliberately seed-local: the previous
        // waypoint is the only IK seed, so a hidden branch jump cannot pass
        // the Cartesian planner's discontinuity gate.
        return solvePiperPoseIK(waypoint, seed, solution, diagnostics, true);
      },
      [this, &environment, &target_flags, &contact_left, &contact_right,
       &grasp_pose, approach_waypoint_count, actual_step](
        const std::vector<float> &joints, std::size_t waypoint_index,
        bool final_waypoint) {
        const std::size_t completed_waypoints = std::min(
          approach_waypoint_count, waypoint_index + 1);
        const float remaining_distance = actual_step * static_cast<float>(
          approach_waypoint_count - completed_waypoints);
        const bool intended_contact_phase = final_waypoint ||
          remaining_distance <= final_approach_contact_phase_m_ + 1.0e-6F;
        if (!intended_contact_phase) {
          return isCandidatePoseSafe(joints, environment);
        }
        return isCandidatePoseSafe(
          joints, environment, &target_flags, &contact_left, &contact_right,
          static_cast<float>(target_contact_surface_tolerance_m_), &grasp_pose);
      });

    if (!plan.success) {
      fail(plan.failure_reason);
      return;
    }
    for (const auto &joints : plan.waypoint_joints) {
      response->path_flat.insert(
        response->path_flat.end(), joints.begin(), joints.end());
    }
    response->waypoint_count =
      static_cast<uint32_t>(plan.waypoint_joints.size());
    response->success = true;
    response->failure_reason = "OK";
    RCLCPP_INFO(
      this->get_logger(),
      "[FINAL APPROACH] distance=%.1fmm waypoints=%u step<=%.1fmm "
      "joint_delta<=%.3frad contact_phase=%.1fmm status=VALID",
      approach_distance * 1000.0F, response->waypoint_count,
      final_approach_translation_step_m_ * 1000.0F,
      final_approach_max_joint_step_rad_,
      final_approach_contact_phase_m_ * 1000.0F);
  }

  void handleFindGraspCandidates(
    const std::shared_ptr<my_actions::srv::FindGraspCandidates::Request> request,
    std::shared_ptr<my_actions::srv::FindGraspCandidates::Response> response) {
    const auto computation_started = std::chrono::steady_clock::now();
    publishGraspMarkers({});
    publishGraspMarkers({}, gpd_proposals_markers_pub_, "gpd_proposal_");
    publishGraspMarkers({}, analytical_proposals_markers_pub_, "analytical_proposal_");
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
    if (!cloud) {
      response->status = requested_stamp.nanoseconds() == 0 ?
        "NO_SYNCHRONIZED_OBSERVATION" : "OBSERVATION_NOT_IN_CACHE";
      RCLCPP_WARN(
        this->get_logger(),
        "[GRASP POC] Observation %ld is unavailable in the %zu-frame cache.",
        requested_stamp.nanoseconds(), grasp_observation_cache_size_);
      return;
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
    if (grasp_candidate_backend_ == "hggd") {
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
    const bool request_uses_hggd = grasp_candidate_backend_ == "hggd";
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
    if (!request_uses_hggd &&
        !findCorrectedSeed(*cloud, *request, u, v, seed_camera, correction)) {
      response->status = "NO_VALID_DEPTH_NEAR_POINT";
      return;
    }
    size_t connected_target_points = 0;
    const sensor_msgs::msg::Image target_mask = buildConnectedTargetMask(
      *cloud, request->target_mask, camera_origin_in_cloud,
      u, v, connected_target_points);
    if (!pointAtPixel(*cloud, u, v, seed_camera)) {
      response->status = "NO_VALID_DEPTH_ON_CONNECTED_TARGET";
      return;
    }
    correction = std::hypot(u - request->seed_u, v - request->seed_v);
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
    if ((grasp_candidate_backend_ == "gpd" || grasp_candidate_backend_ == "hybrid") &&
        !gpd_adapter_) {
      response->status = "GPD_UNAVAILABLE: " + gpd_initialization_error_;
      return;
    }

    Eigen::Vector3f seed(base_point.point.x, base_point.point.y, base_point.point.z);
    Eigen::Vector3f camera_origin = camera_origin_in_cloud;

    std::vector<Eigen::Vector3f> scene_points;
    std::vector<uint8_t> scene_target_flags;
    std::vector<Eigen::Vector3f> target_samples;
    scene_points.reserve(static_cast<size_t>(cloud->width) * cloud->height / 3);
    scene_target_flags.reserve(
      static_cast<size_t>(cloud->width) * cloud->height / 3);
    target_samples.reserve(static_cast<size_t>(cloud->width) * cloud->height / 8);
    const float scene_radius_sq = static_cast<float>(gpd_scene_crop_radius_m_ * gpd_scene_crop_radius_m_);
    const float sample_radius_sq = static_cast<float>(
      gpd_target_sample_radius_m_ * gpd_target_sample_radius_m_);
    for (int pixel_v = 0; pixel_v < static_cast<int>(cloud->height); ++pixel_v) {
      for (int pixel_u = 0; pixel_u < static_cast<int>(cloud->width); ++pixel_u) {
        Point point;
        if (!pointAtPixel(*cloud, pixel_u, pixel_v, point)) continue;
        const Eigen::Vector3f base(point[0], point[1], point[2]);
        const bool target = maskAllows(target_mask, pixel_u, pixel_v);
        const float seed_distance_sq = (base - seed).squaredNorm();
        if (seed_distance_sq <= scene_radius_sq || target) {
          scene_points.push_back(base);
          scene_target_flags.push_back(target ? 1U : 0U);
        }
        if (target && (request_uses_hggd ||
            gpd_target_sample_radius_m_ <= 0.0 ||
            seed_distance_sq <= sample_radius_sq)) {
          target_samples.push_back(base);
        }
      }
    }
    if (scene_points.size() < 200 || target_samples.size() < 80) {
      std::ostringstream status;
      status << "INSUFFICIENT_GRASP_INPUT scene=" << scene_points.size()
             << " target=" << target_samples.size();
      response->status = status.str();
      return;
    }
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
    const bool pure_gpd = grasp_candidate_backend_ == "gpd";
    const bool pure_hggd = grasp_candidate_backend_ == "hggd";
    const bool pure_learned = pure_gpd || pure_hggd;
    size_t local_surface_count = 0;
    std::vector<AntipodalContacts> local_antipodal_pairs;
    AntipodalContacts silhouette_pair;
    bool silhouette_pair_available = false;
    std::vector<AntipodalContacts> body_model_pairs;
    if (require_measured_antipodal_contacts_ && !pure_learned) {
      local_antipodal_pairs = buildLocalAntipodalPairs(
        target_samples, seed, local_surface_count);
      if (!silhouette_pair_available && buildSilhouetteThicknessPair(
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
    AnalyticalGraspDiagnostics analytical_diagnostics;
    HggdMcuDiagnostics hggd_diagnostics;
    int hggd_aligned_centers = 0;
    int hggd_geometry_rejected = 0;
    int hggd_direct_seeds_tested = 0;
    int hggd_prefilter_ik_rejected = 0;
    int hggd_prefilter_collision_rejected = 0;
    double hggd_seed_selection_ms = 0.0;
    std::string hggd_heatmap_output;
    const auto proposal_started = std::chrono::steady_clock::now();
    if (grasp_candidate_backend_ == "analytical" ||
        grasp_candidate_backend_ == "hybrid") {
      proposals = analytical_generator_->generate(
        target_samples, seed, camera_origin, &analytical_diagnostics);
    }
    if (grasp_candidate_backend_ == "gpd" || grasp_candidate_backend_ == "hybrid") {
      try {
        RCLCPP_DEBUG(this->get_logger(),
          "[GPD] Running legacy detector for backend=%s: scene=%zu target=%zu",
          grasp_candidate_backend_.c_str(), scene_points.size(), target_samples.size());
        auto gpd_proposals = gpd_adapter_->detect(
          scene_points, target_samples, camera_origin);
        proposals.insert(proposals.end(), gpd_proposals.begin(), gpd_proposals.end());
      } catch (const std::exception &error) {
        response->status = std::string("GPD_DETECTION_FAILED: ") + error.what();
        RCLCPP_ERROR(this->get_logger(), "%s", response->status.c_str());
        return;
      }
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
        proposals = hggd_adapter_->detect(
          *cloud, *observation_image, target_mask, intrinsics,
          cloud_to_camera_eigen, camera_to_base_eigen,
          static_cast<float>(grasp_contact_depth_m_),
          static_cast<std::size_t>(hggd_maximum_proposals_), heatmap,
          hggd_diagnostics);

        // Prepare one raw marker per spatial family. Publication is delayed
        // until after the quick seed IK/collision prefilter so this RViz layer
        // shows exactly the HGGD priors that were actually offered to local
        // geometry refinement.
        std::unordered_map<std::size_t, my_actions::msg::GraspCandidate>
          raw_hggd_marker_by_family;
        const std::size_t raw_marker_limit = std::min(
          proposals.size(), static_cast<std::size_t>(
            std::max(0, gpd_visualization_limit_)));
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
            approach * static_cast<float>(gpd_tcp_offset_m_);
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
            seed_approach * static_cast<float>(gpd_tcp_offset_m_);
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
          if (!alignHggdSeedToLocalDominantAxis(
              target_samples, camera_origin, current_gripper_rotation,
              have_thin_silhouette ? &thin_silhouette_pair : nullptr,
              aligned))
          {
            ++hggd_geometry_rejected;
            continue;
          }
          aligned_seeds.push_back(std::move(aligned));
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
    if (pure_gpd || pure_hggd) {
      std::stable_sort(proposals.begin(), proposals.end(),
        [](const GraspProposal &left, const GraspProposal &right) {
          return left.quality > right.quality;
        });
    }
    const auto proposal_finished = std::chrono::steady_clock::now();
    RCLCPP_INFO(this->get_logger(),
      "[ANALYTICAL GRASP] backend=%s isolated=%zu downsampled=%zu normals=%zu "
      "pairs=%zu local_pairs=%zu global_pairs=%zu families=%zu local_families=%zu "
      "global_families=%zu generated=%zu deduplicated=%zu total_proposals=%zu generation=%.1fms",
      grasp_candidate_backend_.c_str(), analytical_diagnostics.isolated_points,
      analytical_diagnostics.downsampled_points, analytical_diagnostics.normal_points,
      analytical_diagnostics.contact_pairs, analytical_diagnostics.local_contact_pairs,
      analytical_diagnostics.global_contact_pairs, analytical_diagnostics.selected_contact_families,
      analytical_diagnostics.selected_local_families,
      analytical_diagnostics.selected_global_families,
      analytical_diagnostics.generated,
      analytical_diagnostics.deduplicated, proposals.size(),
      std::chrono::duration<double, std::milli>(proposal_finished - proposal_started).count());



    std::vector<my_actions::msg::GraspCandidate> raw_markers;
    std::vector<my_actions::msg::GraspCandidate> analytical_markers;
    std::vector<my_actions::msg::GraspCandidate> hggd_markers;
    const size_t visualization_limit =
      static_cast<size_t>(std::max(0, gpd_visualization_limit_));
    raw_markers.reserve(std::min(proposals.size(), visualization_limit));
    std::unordered_map<std::size_t, bool> visualized_hggd_families;
    for (const auto &proposal : proposals) {
      GraspProposal visual_proposal = proposal;
      if (proposal.source == GraspProposalSource::kHggd) {
        // Show exactly one geometry-aligned zero frame for each HGGD center.
        // The subsequent dominant-axis orbit is deliberately not published
        // on this debug layer.
        if (visualized_hggd_families[proposal.family_index]) continue;
        visualized_hggd_families[proposal.family_index] = true;
      }
      auto *marker_group = &raw_markers;
      if (proposal.source == GraspProposalSource::kAnalytical) {
        marker_group = &analytical_markers;
      } else if (proposal.source == GraspProposalSource::kHggd) {
        marker_group = &hggd_markers;
      }
      if (marker_group->size() >= visualization_limit) continue;
      const Eigen::Vector3f tcp = visual_proposal.hand_base +
        visual_proposal.approach * static_cast<float>(gpd_tcp_offset_m_);
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
    publishGraspMarkers(raw_markers, gpd_proposals_markers_pub_, "gpd_proposal_");
    publishGraspMarkers(analytical_markers, analytical_proposals_markers_pub_, "analytical_proposal_");
    publishGraspMarkers(hggd_markers, hggd_proposals_markers_pub_, "hggd_proposal_");

    std::vector<Point> env;
    std::vector<uint8_t> env_target_flags;
    if (enable_grasp_collision_check_) {
      { std::lock_guard<std::mutex> lock(cloud_mutex_); env = getFilteredEnvironment(joints); }
      // Preserve the complete low-resolution target in the collision world.
      // Candidate-specific validation below permits only gripper-base contact
      // near the two reconstructed jaw contacts. Arm/wrist contact with the
      // object, and gripper contact elsewhere on it, remain blocking.
      const float target_clearance_sq = static_cast<float>(
        target_collision_clearance_m_ * target_collision_clearance_m_);
      env_target_flags.assign(env.size(), 0);
      for (size_t obstacle_index = 0; obstacle_index < env.size();
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
    bool logged_grasp_collision_detail = false;
    std::array<float, 3> first_target_sweep_block{};
    float first_target_sweep_opening = 0.0F;
    bool first_target_sweep_silhouette = false;
    int rejected_pregrasp_ik = 0, rejected_pregrasp_visibility = 0;
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
    int tested = 0;
    const int candidate_test_limit = pure_hggd ?
      std::min(maximum_candidates_to_test_, 100) : maximum_candidates_to_test_;
    std::unordered_map<uint32_t, std::string> candidate_origins;
    std::unordered_map<uint32_t, size_t> hggd_candidate_seed_indices;
    int geometry_approach_variants_generated = 0;
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
      // Quick exit is seed-scoped, not variant-scoped: finish validating and
      // ranking every purposeful orbit variant for the first successful PCA-
      // aligned seed, then skip all lower-confidence spatial seeds.
      if (pure_hggd && !response->candidates.empty()) {
        seed_diagnostics.terminal_reason = "SKIPPED_AFTER_SUCCESSFUL_SEED";
        continue;
      }
      if (tested >= candidate_test_limit ||
          static_cast<int>(response->candidates.size()) >=
            maximum_ranking_pool_candidates_) {
        if (pure_hggd) {
          seed_diagnostics.ik_budget_exhausted = tested >= candidate_test_limit;
          seed_diagnostics.ranking_pool_full =
            static_cast<int>(response->candidates.size()) >=
            maximum_ranking_pool_candidates_;
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
      // The legacy classifier threshold applies only to legacy GPD proposals.
      // Analytical quality is normalized geometry support, not a classifier logit.
      if (proposal.source == GraspProposalSource::kGpd &&
          enable_gpd_score_filter_ && proposal.quality <= gpd_min_score_) {
        ++rejected_score;
        seed_diagnostics.terminal_reason = "SCORE_FILTER";
        continue;
      }
      if (!pure_learned && proposal.source == GraspProposalSource::kGpd &&
          require_gpd_full_antipodal_ && !proposal.full_antipodal) {
        ++rejected_antipodal;
        seed_diagnostics.terminal_reason = "ANTIPODAL_FILTER";
        continue;
      }
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
      constexpr int hggd_variant_test_limit_per_seed = 2;
      LocalGraspSkill selected_grasp_skill = LocalGraspSkill::kUnknown;
      bool have_constrained_grasp_seed = false;
      std::vector<float> constrained_grasp_seed;
      const bool refine_hggd_geometry = pure_hggd;
      const bool use_measured_contacts = refine_hggd_geometry ||
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
          if (proposal.top_surface_grasp) {
            // A broad top-down grasp already has a complete gravity-aligned
            // pose and two contacts sampled from the observed box walls.
            // The contact-axis solver is intended to flex side/handle poses
            // around a local PCA axis; applying it here can rotate a valid
            // top-down pose back toward a side approach or reject it before
            // the normal full-pose IK solver gets a chance to evaluate it.
            approach_variants.push_back({
              proposal.approach.normalized(),
              proposal.closing.normalized(), 0.0F});
            ++geometry_approach_variants_generated;
          } else {
          const Eigen::Vector3f contact_centroid = 0.5F *
            (measured_contacts.left + measured_contacts.right);
          std::array<float, 6> capture_joints{};
          std::copy_n(joints.begin(), 6, capture_joints.begin());
          const Eigen::Matrix3f capture_rotation =
            Robot::eefk(capture_joints).linear();
          const Eigen::Matrix3f aligned_rotation = piperGripperRotation(
            proposal.closing, proposal.approach);
          Eigen::Matrix3f resolved_rotation = aligned_rotation;
          if (!solvePiperContactAxisIK(
              contact_centroid, proposal.local_dominant_axis,
              aligned_rotation, capture_rotation, joints,
              constrained_grasp_seed, resolved_rotation)) {
            ++rejected_contacts;
            ++contact_rejection_reasons[
              "constrained_contact_axis_ik_failed"];
            seed_diagnostics.terminal_reason =
              "CONSTRAINED_CONTACT_AXIS_IK_FAILED";
            continue;
          }
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
            angular_deviation / static_cast<float>(M_PI_2)});
          ++geometry_approach_variants_generated;
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
          if (!pinchSectionAwayFromJunction(
                target_samples, oriented_contacts, closing, height_axis,
                approach)) {
            ++rejected_contacts;
            ++contact_rejection_reasons[
              "pinch_junction_or_endpoint"];
            ++seed_diagnostics.cross_section_rejected;
            continue;
          }
          // grasp_contact_depth_m is the commanded distance from the physical
          // finger base to the measured contact plane. gpd_tcp_offset_m is a
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
          approach * static_cast<float>(gpd_tcp_offset_m_);
        const Eigen::Vector3f contact_midpoint = (left + right) * 0.5F;
        Eigen::Matrix3f rotation = piperGripperRotation(closing, approach);
        Eigen::Isometry3f target_pose = Eigen::Isometry3f::Identity();
        target_pose.linear() = rotation;
        target_pose.translation() = tcp;

        // Build a candidate-centric high-resolution environment from the
        // synchronized raw observation. The isolated target remains the
        // contact source, while cup body, table, and nearby clutter remain
        // obstacles. This complements the global low-resolution voxel map.
        std::vector<Point> precise_local_environment;
        std::vector<uint8_t> precise_local_target_flags;
        constexpr float precise_local_radius_m = 0.12F;
        constexpr float precise_local_radius_sq =
          precise_local_radius_m * precise_local_radius_m;
        precise_local_environment.reserve(scene_points.size() / 4);
        precise_local_target_flags.reserve(scene_points.size() / 4);
        for (size_t scene_index = 0; scene_index < scene_points.size();
             ++scene_index) {
          if ((scene_points[scene_index] - contact_midpoint).squaredNorm() >
              precise_local_radius_sq) {
            continue;
          }
          const auto &point = scene_points[scene_index];
          precise_local_environment.push_back(
            {point.x(), point.y(), point.z()});
          precise_local_target_flags.push_back(
            scene_target_flags[scene_index]);
        }
      // Only classifier-positive, physically legal openings enter the
      // comparatively expensive IK and collision-validation stages.
      if (tested >= candidate_test_limit ||
          (pure_hggd && tested - seed_test_start >=
            hggd_variant_test_limit_per_seed)) {
        seed_diagnostics.ik_budget_exhausted = true;
        if (seed_diagnostics.terminal_reason.empty()) {
          seed_diagnostics.terminal_reason = "VARIANT_TEST_BUDGET";
        }
        break;
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
      for (const bool swap_fingers : {false, true}) {
        Eigen::Isometry3f branch_target = target_pose;
        branch_target.linear() = piperGripperRotation(closing, approach, swap_fingers);
        std::vector<float> branch_ik;
        PoseIkDiagnostics grasp_diagnostics;
        const std::vector<float> &grasp_seed =
          have_constrained_grasp_seed && !swap_fingers ?
          constrained_grasp_seed : joints;
        const bool constrained_seed_only =
          have_constrained_grasp_seed && !swap_fingers;
        if (!solvePiperPoseIK(
            branch_target, grasp_seed, branch_ik, grasp_diagnostics,
            constrained_seed_only)) {
          best_grasp_diagnostics.keepBest(grasp_diagnostics);
          continue;
        }
        any_grasp_ik = true;
        best_grasp_diagnostics.keepBest(grasp_diagnostics);
        // A top-down power grasp contacts a flat box wall along the jaw pad,
        // rather than at one mathematical point. Keep the generic 30 mm
        // allowance for all other grasps, but cover the bounded 50 mm cached
        // contact region for this measured topology. The exemption below is
        // still restricted to segmented-target points and inner-pad spheres;
        // palm, finger exterior, arm, table, and clutter remain blocking.
        const float candidate_contact_allowance_m =
          proposal.top_surface_grasp ? 0.050F :
          static_cast<float>(target_collision_clearance_m_);
        if (enable_grasp_collision_check_ && !isCandidatePoseSafe(
            branch_ik, env, &env_target_flags, &left, &right,
            candidate_contact_allowance_m,
            &branch_target))
        {
          if (!logged_grasp_collision_detail) {
            RCLCPP_INFO(
              this->get_logger(), "[GRASP COLLISION DETAIL] coarse %s",
              describeCandidateCollision(
                branch_ik, env, &env_target_flags, &left, &right,
                candidate_contact_allowance_m,
                &branch_target).c_str());
            logged_grasp_collision_detail = true;
          }
          continue;
        }
        if (enable_grasp_collision_check_ && !isCandidatePoseSafe(
            branch_ik, precise_local_environment,
            &precise_local_target_flags, &left, &right,
            candidate_contact_allowance_m,
            &branch_target))
        {
          if (!logged_grasp_collision_detail) {
            RCLCPP_INFO(
              this->get_logger(), "[GRASP COLLISION DETAIL] precise %s",
              describeCandidateCollision(
                branch_ik, precise_local_environment,
                &precise_local_target_flags, &left, &right,
                candidate_contact_allowance_m,
                &branch_target).c_str());
            logged_grasp_collision_detail = true;
          }
          continue;
        }
        any_grasp_collision_free = true;
        for (const float retreat : pregraspRetreatCandidates()) {
          Eigen::Isometry3f branch_pregrasp = branch_target;
          branch_pregrasp.translation() -= approach * retreat;
          Eigen::Isometry3f resolved_pregrasp = branch_pregrasp;
          std::vector<float> branch_pregrasp_ik;
          PoseIkDiagnostics pregrasp_diagnostics;
          if (!solveRelaxedPregraspIK(
              branch_pregrasp, branch_ik, branch_pregrasp_ik,
              pregrasp_diagnostics, resolved_pregrasp,
              proposal.top_surface_grasp ? 45.0F : -1.0F))
          {
            best_pregrasp_diagnostics.keepBest(pregrasp_diagnostics);
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
              !isCandidatePoseSafe(branch_pregrasp_ik, env))
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
          if (!accepted ||
              camera_upright_score > accepted_camera_upright_score) {
            target_pose = branch_target;
            rotation = branch_target.rotation();
            ik = branch_ik;
            pregrasp_ik = std::move(branch_pregrasp_ik);
            accepted_pregrasp_pose = resolved_pregrasp;
            accepted_camera_upright_score = camera_upright_score;
            accepted = true;
          }
          // The configured/preferred retreat was feasible for this wrist
          // branch. Remaining distances cannot improve visibility enough to
          // justify additional IK work.
          break;
        }
      }
      if (!accepted && have_pregrasp_collision_fallback) {
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
      my_actions::msg::GraspCandidate candidate;
      candidate.id = static_cast<uint32_t>(response->candidates.size() + 1);
      candidate.grasp_pose.header = base_point.header; candidate.grasp_pose.header.frame_id = "base_link";
      candidate.grasp_pose.pose.position.x = tcp.x(); candidate.grasp_pose.pose.position.y = tcp.y(); candidate.grasp_pose.pose.position.z = tcp.z();
      Eigen::Quaternionf q(rotation); candidate.grasp_pose.pose.orientation.x = q.x(); candidate.grasp_pose.pose.orientation.y = q.y(); candidate.grasp_pose.pose.orientation.z = q.z(); candidate.grasp_pose.pose.orientation.w = q.w();
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
          std::max(0.001F, static_cast<float>(gpd_target_sample_radius_m_)),
        0.0F, 1.0F);
      const float vertical_approach_penalty = std::clamp(
        std::abs(approach.z()), 0.0F, 1.0F);
      float camera_approach_penalty = 0.0F;
      const float camera_upright_penalty = pure_hggd ?
        0.5F * (1.0F - std::clamp(
          accepted_camera_upright_score, -1.0F, 1.0F)) : 0.0F;
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
      constexpr float lower_limits[6] = {
        -2.6179938F, 0.0F, -2.9670597F, -1.7453292F, -1.2217304F, -2.0943951F};
      constexpr float upper_limits[6] = {
        2.6179938F, 3.1415926F, 0.0F, 1.7453292F, 1.2217304F, 2.0943951F};
      for (size_t index = 0; index < 6; ++index) {
        const float delta = ik[index] - joints[index];
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
          camera_upright_penalty,
        camera_approach_penalty};
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
      std::copy_n(ik.begin(), candidate.grasp_joints.size(), candidate.grasp_joints.begin());
      std::copy_n(
        pregrasp_ik.begin(), candidate.pregrasp_joints.size(),
        candidate.pregrasp_joints.begin());
      if (pure_hggd) {
        hggd_candidate_seed_indices[candidate.id] = proposal_index;
        if (seed_diagnostics.accepted == 0) {
          response->corrected_seed_base.x = proposal.sample.x();
          response->corrected_seed_base.y = proposal.sample.y();
          response->corrected_seed_base.z = proposal.sample.z();
        }
      }
      response->candidates.push_back(candidate);
      ++seed_diagnostics.accepted;
      if (candidate.affordance_family == "PINCH") {
        ++pinch_candidates_accepted;
      } else if (candidate.affordance_family == "POWER_GRASP") {
        ++power_candidates_accepted;
      }
      switch (proposal.analytical_origin) {
        case AnalyticalProposalOrigin::kLocalPca:
          candidate_origins[candidate.id] = "local_pca";
          break;
        case AnalyticalProposalOrigin::kGlobalPca:
          candidate_origins[candidate.id] = "global_pca";
          break;
        case AnalyticalProposalOrigin::kLegacyGpd:
          candidate_origins[candidate.id] = "legacy_gpd";
          break;
      }
      if (pure_hggd) {
        seed_diagnostics.terminal_reason = "FIRST_SAFE_VARIANT";
        // HGGD is evaluated seed-by-seed. This candidate has passed measured
        // finger sweep, strict final IK, grasp collision, relaxed pre-grasp
        // IK, visibility, and pre-grasp collision checks. Do not spend the
        // remaining budget producing lower-priority variants.
        break;
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
        if (pure_gpd) {
          // Preserve the detector score and pose. In legacy GPD mode the learned
          // score is primary; only small, deterministic robot-state terms
          // break otherwise similar proposals.
          candidate.score_components[1] = proposal_quality_normalized;
          candidate.score_components[2] = 0.0F;
          candidate.score_components[3] = 0.0F;
          candidate.score_components[4] = 0.0F;
          candidate.score_components[5] = 0.0F;
          candidate.ranking_score = proposal_quality_normalized +
            0.05F * candidate.score_components[7] -
            0.05F * candidate.score_components[6];
          continue;
        }
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
              selected_closing, static_cast<float>(gpd_diversity_position_m_),
              gpd_diversity_approach_degrees_, gpd_diversity_closing_degrees_);
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
          pure_hggd ? "HGGD" : "GPD", response->candidates.size(),
          diversity_rejected, diverse_candidates.size(),
          gpd_diversity_position_m_, gpd_diversity_approach_degrees_,
          gpd_diversity_closing_degrees_);
        response->candidates = std::move(diverse_candidates);
      } else if (grasp_candidate_backend_ == "analytical" ||
          grasp_candidate_backend_ == "hybrid") {
        std::vector<my_actions::msg::GraspCandidate> diverse_candidates;
        diverse_candidates.reserve(static_cast<size_t>(maximum_valid_candidates_));
        for (const auto &candidate : response->candidates) {
          const Eigen::Vector3f center(
            static_cast<float>((candidate.contact_left.x + candidate.contact_right.x) * 0.5),
            static_cast<float>((candidate.contact_left.y + candidate.contact_right.y) * 0.5),
            static_cast<float>((candidate.contact_left.z + candidate.contact_right.z) * 0.5));
          bool too_close = false;
          for (const auto &selected : diverse_candidates) {
            const Eigen::Vector3f selected_center(
              static_cast<float>((selected.contact_left.x + selected.contact_right.x) * 0.5),
              static_cast<float>((selected.contact_left.y + selected.contact_right.y) * 0.5),
              static_cast<float>((selected.contact_left.z + selected.contact_right.z) * 0.5));
            if ((center - selected_center).norm() <
                static_cast<float>(analytical_accepted_candidate_separation_m_)) {
              too_close = true;
              break;
            }
          }
          if (too_close) continue;
          diverse_candidates.push_back(candidate);
          if (diverse_candidates.size() >= static_cast<size_t>(maximum_valid_candidates_)) break;
        }
        RCLCPP_INFO(
          this->get_logger(),
          "[ANALYTICAL DIVERSITY] accepted_pool=%zu spatially_distinct=%zu separation=%.3fm",
          response->candidates.size(), diverse_candidates.size(),
          analytical_accepted_candidate_separation_m_);
        response->candidates = std::move(diverse_candidates);
      } else if (response->candidates.size() > static_cast<size_t>(maximum_valid_candidates_)) {
        response->candidates.resize(static_cast<size_t>(maximum_valid_candidates_));
      }
      if (pure_hggd) {
        // Stage 2 ranks every valid orientation variant of the selected seed,
        // but the public pipeline needs only its single best executable pose.
        if (response->candidates.size() > 1) {
          response->candidates.resize(1);
        }
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
      writeAnalyticalDebugLog(
        cloud_stamp, seed, analytical_diagnostics, valid_pool_for_log,
        response->candidates, candidate_origins);
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
        "[GRASP RANK] Selected #1 ranking=%.3f proposal-quality=%.3f opening=%.3fm "
        "pregrasp-retreat=%.3fm "
        "relative-width=%.2f pool-width=[%.3f,%.3f]m "
        "vertical-close=%.2f seed=%.2f approach=%.2f "
        "camera-approach=%.2f joint-travel=%.2f joint-margin=%.2f status=%s",
        best.ranking_score, best.score, best.jaw_opening_m,
        (best_pregrasp_position - best_grasp_position).norm(),
        best.score_components[2], minimum_opening_m,
        maximum_opening_m, best.score_components[3],
        best.score_components[4], best.score_components[5],
        best.score_components[10], best.score_components[6],
        best.score_components[7],
        best.validation_status.c_str());
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
      writeAnalyticalDebugLog(
        cloud_stamp, seed, analytical_diagnostics, {}, {}, candidate_origins);
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
             << " backend=" << grasp_candidate_backend_ << " proposals=" << proposals.size()
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
    if (pure_gpd) {
      RCLCPP_INFO(
        this->get_logger(),
        "[GPD PIPELINE] raw=%zu score_rejected=%d width_rejected=%d "
        "grasp_ik_rejected=%d pregrasp_ik_rejected=%d "
        "grasp_collision_rejected=%d pregrasp_collision_rejected=%d "
        "feasible_pool=%zu diversity_rejected=%d returned=%zu",
        proposals.size(), rejected_score, rejected_width, rejected_ik,
        rejected_pregrasp_ik, rejected_grasp_collision,
        rejected_pregrasp_collision, ranking_pool_candidate_count,
        diversity_rejected, response->candidates.size());
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
        "network={anchor:%.1fms,local:%.1fms} validation=%.1fms total=%.1fms heatmap=%s",
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
        hggd_heatmap_output.empty() ? "<not-written>" : hggd_heatmap_output.c_str());
    }

    RCLCPP_INFO(this->get_logger(),
      "[GRASP VALIDATION] backend=%s generated=%zu tested=%d accepted=%zu "
      "rejected={quality:%d,width:%d,antipodal:%d,contacts:%d,sweep:%d,"
      "grasp_ik:%d,pregrasp_ik:%d,pregrasp_not_visible:%d,"
      "grasp_collision:%d,pregrasp_collision:%d} "
      "pregrasp={retreat_range:[0.100,0.200]m,preferred:%.3fm,"
      "orientation_tolerance:%.1fdeg} "
      "validation=%.1fms total=%.1fms",
      grasp_candidate_backend_.c_str(), proposals.size(), tested,
      response->candidates.size(), rejected_score, rejected_width,
      rejected_antipodal, rejected_contacts, rejected_finger_sweep,
      rejected_ik, rejected_pregrasp_ik, rejected_pregrasp_visibility,
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
       
       float lower_limits[6] = {-2.6179938f, 0.0f, -2.9670597f, -1.7453292f, -1.2217304f, -2.0943951f};
       float upper_limits[6] = {2.6179938f, 3.1415926f, 0.0f, 1.7453292f, 1.2217304f, 2.0943951f};
       
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
        request->allowed_contact_radius_m);
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
      grasp_contact_depth_m_ - gpd_tcp_offset_m_);
    const float lower[6] = {
      -2.6179938F, 0.0F, -2.9670597F,
      -1.7453292F, -1.2217304F, -2.0943951F};
    const float upper[6] = {
      2.6179938F, 3.1415926F, 0.0F,
      1.7453292F, 1.2217304F, 2.0943951F};

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
        std::array<float, 6> q{};
        std::copy_n(joints.begin(), 6, q.begin());
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
                         const int maximum_iterations = 800) {
    out_joints.assign(seed.begin(), seed.end());
    const float lower[6] = {-2.6179938f, 0.0f, -2.9670597f, -1.7453292f, -1.2217304f, -2.0943951f};
    const float upper[6] = { 2.6179938f, 3.1415926f, 0.0f, 1.7453292f,  1.2217304f,  2.0943951f};
    constexpr float delta = 0.001f;
    constexpr float orientation_weight = 0.12f;
    for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
      std::array<float, 6> q{};
      for (size_t i = 0; i < 6; ++i) q[i] = out_joints[i];
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
      Eigen::Matrix<float, 6, 1> error;
      for (int axis = 0; axis < 3; ++axis) {
        error[axis] = position_error[axis];
        error[axis + 3] = orientation_weight * angular_error[axis];
      }
      Eigen::Matrix<float, 6, 6> jacobian;
      for (size_t column = 0; column < 6; ++column) {
        q[column] += delta;
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
                        const int maximum_iterations = 800) {
    if (seed.size() != 6) return false;
    const float lower[6] = {-2.6179938f, 0.0f, -2.9670597f, -1.7453292f, -1.2217304f, -2.0943951f};
    const float upper[6] = { 2.6179938f, 3.1415926f, 0.0f, 1.7453292f,  1.2217304f,  2.0943951f};
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
                           rotation_tolerance_rad, maximum_iterations)) {
        diagnostics.keepBest(attempt_diagnostics);
        bool duplicate_solution = false;
        for (const auto &existing : unique_solutions) {
          float distance_sq = 0.0F;
          for (size_t i = 0; i < 6; ++i) distance_sq += std::pow(existing[i] - candidate_joints[i], 2);
          if (distance_sq < 1.0e-6F) { duplicate_solution = true; break; }
        }
        if (duplicate_solution) continue;
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
                              Eigen::Isometry3f &resolved_pregrasp,
                              const float orientation_tolerance_override_degrees =
                                -1.0F) {
    // The pre-grasp position remains on the final grasp's approach line, but
    // orientation is a bounded soft constraint. The numerical IK therefore
    // adjusts all six joints continuously instead of testing a grid of fixed
    // 5-degree tilt targets. The caller already tries standoffs from 10--20 cm.
    const float orientation_tolerance_degrees =
      orientation_tolerance_override_degrees >= 0.0F ?
      orientation_tolerance_override_degrees :
      static_cast<float>(pregrasp_orientation_tolerance_degrees_);
    const float orientation_tolerance_rad =
      orientation_tolerance_degrees * static_cast<float>(M_PI / 180.0);
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
    out_joints = seed_joints;
    
    float learning_rate = 1.0f;
    float delta = 0.001f;
    int max_iters = 2000;
    
    // Joint limits — must clamp each iteration to prevent divergence
    float lower_limits[6] = {-2.6179938f, 0.0f, -2.9670597f, -1.7453292f, -1.2217304f, -2.0943951f};
    float upper_limits[6] = {2.6179938f, 3.1415926f, 0.0f, 1.7453292f, 1.2217304f, 2.0943951f};
    
    float q_curr[6];
    for (size_t j = 0; j < 6; ++j) q_curr[j] = out_joints[j];
    
    Robot::ConfigurationBlock<1> c_in;
    size_t ee_idx = Robot::n_spheres - 1;
    
    float final_err_sq = 1e9f;
    float prev_err_sq = 1e9f;
    int stagnation_count = 0;
    
    for (int iter = 0; iter < max_iters; ++iter) {
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
    const Eigen::Isometry3f *allowed_grasp_pose = nullptr)
  {
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

    Robot::ConfigurationBlock<1> c_in;
    for (size_t j = 0; j < 6; ++j) c_in[j][0] = q_vec[j];
    Robot::Spheres<1> out;
    Robot::template sphere_fk<1>(c_in, out);

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
    for (size_t obstacle_index = 0; obstacle_index < env_voxels.size();
         ++obstacle_index) {
      const auto &pt = env_voxels[obstacle_index];
      for (size_t s = 0; s < Robot::n_spheres; ++s) {
        if (vamp::collision::sphere_sphere_sql2(
              out.x[{s, 0}], out.y[{s, 0}], out.z[{s, 0}], out.r[{s, 0}],
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
                s, Eigen::Vector3f(out.x[{s, 0}], out.y[{s, 0}], out.z[{s, 0}]),
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
              continue;
            }
          }
          return false;
        }
      }
    }
    return true;
  }

  std::string describeCandidateCollision(
    const std::vector<float> &q_vec, const std::vector<Point> &env_voxels,
    const std::vector<uint8_t> *target_flags,
    const Eigen::Vector3f *contact_left,
    const Eigen::Vector3f *contact_right,
    const float allowed_contact_radius_m,
    const Eigen::Isometry3f *allowed_grasp_pose)
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
    if (!Robot::fkcc<rake>(empty_env_v, block)) return "self_collision";

    Robot::ConfigurationBlock<1> configuration;
    for (size_t joint = 0; joint < Robot::dimension; ++joint) {
      configuration[joint][0] = q_vec[joint];
    }
    Robot::Spheres<1> spheres;
    Robot::template sphere_fk<1>(configuration, spheres);
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
        if (vamp::collision::sphere_sphere_sql2(
            spheres.x[{sphere, 0}], spheres.y[{sphere, 0}],
            spheres.z[{sphere, 0}], spheres.r[{sphere, 0}],
            point[0], point[1], point[2], point_radius_m) >= 0) {
          continue;
        }
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
              sphere, Eigen::Vector3f(
                spheres.x[{sphere, 0}], spheres.y[{sphere, 0}],
                spheres.z[{sphere, 0}]),
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
        }
        std::ostringstream description;
        description << "environment sphere=" << sphere
          << " point=[" << point[0] << ',' << point[1] << ',' << point[2]
          << "] target=" << target << " near_contact=" << near_contact
          << " inner_pad=" << inner_pad
          << " strict_contact_patch=" << strict_contact_patch;
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
      float allowed_contact_radius_m = 0.0F)
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
        float lower_limits[6] = {-2.617f, 0.0f, -2.967f, -1.745f, -1.221f, -2.094f};
        float upper_limits[6] = {2.617f, 3.141f, 0.0f, 1.745f, 1.221f, 2.094f};
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
          
        auto result = RRTC::solve(start_cfg, goals_vec, env_v, rrtc_settings, rng);
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
