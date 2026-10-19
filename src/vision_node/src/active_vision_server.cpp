#include <onnxruntime_cxx_api.h>
#include "vision_node/metric_scale.hpp"
#include "vision_node/voxel_neighbors.hpp"
#include "vision_node/perception_translation.hpp"
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "my_actions/srv/get_yolo_inference.hpp"
#include "my_actions/srv/update_object_observation.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <tuple>
#include <deque>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <limits>
#include <iomanip>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/empty.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "my_actions/srv/get_obstacle_location.hpp"
#include "my_actions/srv/get3_d_coordinate.hpp"
#include "my_actions/srv/get_nbv_pose.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>

namespace vision_node
{

const std::vector<std::string> COCO_CLASSES = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
  "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
  "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};


struct Point3D {
  float x, y, z;
};

struct VoxelHash {
  std::size_t operator()(const std::tuple<int, int, int>& k) const {
    return std::hash<int>()(std::get<0>(k)) ^
           (std::hash<int>()(std::get<1>(k)) << 1) ^
           (std::hash<int>()(std::get<2>(k)) << 2);
  }
};

class ActiveVisionServer : public rclcpp::Node
{
public:
  ActiveVisionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("active_vision_server", options), scanning_active_(true), collision_monitor_active_(false)
  {
    hand_eye_calibration_path_ = this->declare_parameter<std::string>(
      "hand_eye_calibration_path", "");
    load_calibration();
    rcl_interfaces::msg::ParameterDescriptor scale_descriptor;
    scale_descriptor.read_only = true;  // Restart to avoid mixing pinned observations.
    perception_measurement_ratio_ = this->declare_parameter<double>(
      "perception_measurement_ratio", 1.0, scale_descriptor);
    if (!std::isfinite(perception_measurement_ratio_) || perception_measurement_ratio_ <= 0.) {
      throw std::invalid_argument("perception_measurement_ratio must be finite and positive");
    }
    const auto offset = this->declare_parameter<std::vector<double>>(
      "perception_position_offset_base_m", std::vector<double>{0., 0., 0.}, scale_descriptor);
    if (offset.size() != 3 || !std::all_of(offset.begin(), offset.end(),
        [](double v) {return std::isfinite(v);})) {
      throw std::invalid_argument("perception_position_offset_base_m requires three finite metres");
    }
    perception_position_offset_base_m_ = Eigen::Vector3f(offset[0], offset[1], offset[2]);
    corrected_depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/active_vision/corrected_registered_depth", rclcpp::SensorDataQoS());
    RCLCPP_INFO(this->get_logger(), "[METRIC SCALE] measured/real=%.6f correction=%.6f before FK*X",
      perception_measurement_ratio_, 1. / perception_measurement_ratio_);

    obstacle_voxel_size_ = this->declare_parameter<double>("obstacle_voxel_size_m", 0.05);
    // Reject transient depth noise: a coarse obstacle voxel is occupied only
    // when the current captured frame contributes at least 100 valid samples.
    obstacle_min_points_per_voxel_ = this->declare_parameter<int>("obstacle_min_points_per_voxel", 100);
    // Gemini RGB and registered-depth streams may be timestamped by separate
    // device pipelines.  Choose the nearest buffered RGB frame and report the
    // measured offset instead of silently discarding a valid observation.
    rgb_depth_sync_tolerance_s_ = this->declare_parameter<double>("rgb_depth_sync_tolerance_s", 0.15);
    enable_yolo_inference_ = this->declare_parameter<bool>("enable_yolo_inference", false);
    publish_yolo_world_observations_ = this->declare_parameter<bool>(
      "publish_yolo_world_observations", false);
    yolo_world_classes_ = this->declare_parameter<std::vector<std::string>>(
      "yolo_world_classes", {"cup"});
    enable_gripper_self_filter_ = this->declare_parameter<bool>(
      "enable_gripper_self_filter", true);
    gripper_self_filter_margin_px_ = this->declare_parameter<int>(
      "gripper_self_filter_margin_px", 10);
    gripper_self_filter_vertical_margin_px_ = this->declare_parameter<int>(
      "gripper_self_filter_vertical_margin_px", 8);
    gripper_calibration_width_px_ = this->declare_parameter<int>(
      "gripper_self_filter_calibration_width_px", 640);
    gripper_calibration_height_px_ = this->declare_parameter<int>(
      "gripper_self_filter_calibration_height_px", 480);
    gripper_calibration_widths_m_ = this->declare_parameter<std::vector<double>>(
      "gripper_self_filter_widths_m", {0.01, 0.05, 0.07});
    gripper_left_closed_xy_ = this->declare_parameter<std::vector<int64_t>>(
      "gripper_self_filter_left_closed_xy",
      {355, 319, 402, 319, 435, 480, 285, 480, 335, 410});
    gripper_left_mid_xy_ = this->declare_parameter<std::vector<int64_t>>(
      "gripper_self_filter_left_mid_xy",
      {250, 319, 290, 319, 358, 420, 383, 480, 50, 480, 170, 400});
    gripper_left_open_xy_ = this->declare_parameter<std::vector<int64_t>>(
      "gripper_self_filter_left_open_xy",
      {190, 319, 230, 319, 225, 480, 0, 480, 110, 405});
    gripper_right_closed_xy_ = this->declare_parameter<std::vector<int64_t>>(
      "gripper_self_filter_right_closed_xy",
      {398, 319, 455, 319, 640, 429, 640, 480, 435, 480});
    gripper_right_mid_xy_ = this->declare_parameter<std::vector<int64_t>>(
      "gripper_self_filter_right_mid_xy",
      {525, 319, 640, 319, 640, 480, 500, 480, 495, 420, 516, 390});
    gripper_right_open_xy_ = this->declare_parameter<std::vector<int64_t>>(
      "gripper_self_filter_right_open_xy",
      {585, 319, 640, 319, 640, 480, 625, 480, 572, 390});
    // Collision world is deliberately coarse; object grasp geometry uses its
    // own 5 mm filter in VAMP.
    octree_ = std::make_unique<octomap::OcTree>(obstacle_voxel_size_);
    octree_->setOccupancyThres(0.7);
    octree_->setProbHit(0.7);
    octree_->setProbMiss(0.4);
    octree_->setClampingThresMin(0.12);
    octree_->setClampingThresMax(0.97);

    // Publishers
    auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    map_qos.reliable();
    map_qos.transient_local();
    global_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/active_vision/global_map", map_qos);
    observation_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/active_vision/observation_cloud", map_qos);
    observation_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/active_vision/observation_image", map_qos);
    observation_camera_transform_pub_ =
      this->create_publisher<geometry_msgs::msg::TransformStamped>(
        "/active_vision/observation_camera_transform", map_qos);
    // RViz receives the exact same FK(joints) * calibrated-X transform used
    // to build observation_cloud. TF is output-only here; no computation in
    // this node reads from the TF tree.
    calibrated_camera_tf_broadcaster_ =
      std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    gripper_self_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/active_vision/gripper_self_mask", map_qos);
    gripper_self_filter_debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/active_vision/gripper_self_filter_debug", map_qos);
    collision_alert_pub_ = this->create_publisher<std_msgs::msg::Bool>("/active_vision/collision_alert", 10);

    // Services
    trigger_scan_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/active_vision/trigger_scan",
      std::bind(&ActiveVisionServer::handleTriggerScan, this, std::placeholders::_1, std::placeholders::_2));
    collision_monitor_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "/active_vision/enable_collision_monitor", 10,
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        bool prev = collision_monitor_active_;
        collision_monitor_active_ = msg->data;
        if (prev != msg->data) {
          RCLCPP_INFO(this->get_logger(), "[COLLISION MONITOR] Toggled: %s -> %s",
                      prev ? "ON" : "OFF", msg->data ? "ON" : "OFF");
        }
      });

    reload_calib_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/active_vision/reload_calibration",
      std::bind(&ActiveVisionServer::handleReloadCalibration, this, std::placeholders::_1, std::placeholders::_2));

    get_obstacle_srv_ = this->create_service<my_actions::srv::GetObstacleLocation>(
      "/active_vision/get_obstacle_location",
      std::bind(&ActiveVisionServer::handleGetObstacleLocation, this, std::placeholders::_1, std::placeholders::_2));

    get_3d_coord_srv_ = this->create_service<my_actions::srv::Get3DCoordinate>(
      "/active_vision/get_3d_coordinate",
      std::bind(&ActiveVisionServer::handleGet3DCoordinate, this, std::placeholders::_1, std::placeholders::_2));

    get_nbv_srv_ = this->create_service<my_actions::srv::GetNBVPose>(
      "/active_vision/get_nbv_candidates",
      std::bind(&ActiveVisionServer::handleGetNBVPose, this, std::placeholders::_1, std::placeholders::_2));

    // YOLO is optional. The GRASP path consumes the synchronized raw RGB-D
    // observation directly and therefore remains usable when this is false.
    if (enable_yolo_inference_) {
      env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLO26s-seg");
      session_options_.SetIntraOpNumThreads(2);
      session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      try {
        std::string pkg_share = ament_index_cpp::get_package_share_directory("vision_node");
        std::string model_path = pkg_share + "/models/yolo26n-seg.onnx";
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options_);
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_->GetInputNameAllocated(0, allocator).get();
        output0_name_ = session_->GetOutputNameAllocated(0, allocator).get();
        output1_name_ = session_->GetOutputNameAllocated(1, allocator).get();
        RCLCPP_INFO(this->get_logger(), "Loaded internal ONNX model: %s", model_path.c_str());
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error loading ONNX model: %s", e.what());
      }
    } else {
      RCLCPP_INFO(this->get_logger(), "YOLO inference disabled; synchronized RGB-D and grasp services remain active.");
    }

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/color/image_raw", rclcpp::QoS(10).reliable(),
      std::bind(&ActiveVisionServer::colorImageCallback, this, std::placeholders::_1));

    get_yolo_srv_ = this->create_service<my_actions::srv::GetYoloInference>(
      "/active_vision/get_yolo_inference",
      std::bind(&ActiveVisionServer::handleGetYoloInference, this, std::placeholders::_1, std::placeholders::_2));


    // Clients
    estop_client_ = this->create_client<std_srvs::srv::Empty>("/emergency_stop");
    update_object_client_ =
      this->create_client<my_actions::srv::UpdateObjectObservation>(
      "/master/update_object_observation");

    // The Orbbec and arm drivers can stamp messages from different clock
    // domains, so header-based ApproximateTime cannot pair them reliably.
    // Both subscribers retain their independent callbacks; each cloud is
    // paired below with the latest 50 Hz joint sample by local arrival order.
    pc_sub_.subscribe(this, "/camera/depth_registered/points", rmw_qos_profile_sensor_data);
    pose_sub_.subscribe(this, "/feedback/joint_states");

    pc_sub_.registerCallback(std::bind(&ActiveVisionServer::camCountCallback, this, std::placeholders::_1));
    pose_sub_.registerCallback(std::bind(&ActiveVisionServer::poseCountCallback, this, std::placeholders::_1));

    last_proc_time_ = this->now();

    RCLCPP_INFO(this->get_logger(), "Edge Active Vision & Safety Server ready (50cm depth filter, Octree raycasting).");
  }

private:
  void load_calibration() {
    const std::string calib_file = hand_eye_calibration_path_;
    X_matrix_ = Eigen::Matrix4f::Identity();
    if (calib_file.empty()) {
      RCLCPP_ERROR(this->get_logger(),
        "hand_eye_calibration_path is empty; refusing to use an implicit transform.");
      has_calibration_ = false;
      return;
    }
    try {
      YAML::Node config = YAML::LoadFile(calib_file);
      std::vector<std::vector<float>> mat = config["hand_eye_transform"].as<std::vector<std::vector<float>>>();
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          X_matrix_(i, j) = mat[i][j];
        }
      }
      has_calibration_ = true;
      RCLCPP_INFO(this->get_logger(), "Loaded Hand-Eye Calibration matrix successfully.");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load hand-eye calibration: %s", e.what());
    }
  }



  void colorImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(color_mutex_);
      try {
          latest_color_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
          latest_color_header_ = msg->header;
          // The decoded frame is immutable; cv::Mat retains its pixels while
          // a matching RGB-D observation still references this history entry.
          recent_color_frames_.push_back({
            latest_color_header_, latest_color_image_, this->now()});
          constexpr size_t max_color_history = 90;
          while (recent_color_frames_.size() > max_color_history) recent_color_frames_.pop_front();
      } catch (cv_bridge::Exception& e) {
          RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      }
  }

  std::vector<cv::Point2f> decodeGripperPolygon(
    const std::vector<int64_t> &flattened) const {
    std::vector<cv::Point2f> polygon;
    if (flattened.size() < 6 || flattened.size() % 2 != 0) return polygon;
    polygon.reserve(flattened.size() / 2);
    for (size_t index = 0; index < flattened.size(); index += 2) {
      polygon.emplace_back(
        static_cast<float>(flattened[index]),
        static_cast<float>(flattened[index + 1]));
    }
    return polygon;
  }

  double interpolateGripperAnchorX(
    double gripper_width_m, const std::vector<int64_t> &closed_flat,
    const std::vector<int64_t> &middle_flat,
    const std::vector<int64_t> &open_flat) const {
    if (closed_flat.size() < 2 || middle_flat.size() < 2 || open_flat.size() < 2) {
      return 0.0;
    }
    const double closed_width = gripper_calibration_widths_m_[0];
    const double middle_width = gripper_calibration_widths_m_[1];
    const double open_width = gripper_calibration_widths_m_[2];
    const double clamped_width = std::clamp(
      gripper_width_m, closed_width, open_width);
    const bool lower_segment = clamped_width <= middle_width;
    const double segment_start = lower_segment ? closed_width : middle_width;
    const double segment_end = lower_segment ? middle_width : open_width;
    const double first_x = static_cast<double>(
      lower_segment ? closed_flat[0] : middle_flat[0]);
    const double second_x = static_cast<double>(
      lower_segment ? middle_flat[0] : open_flat[0]);
    const double ratio = (clamped_width - segment_start) /
      (segment_end - segment_start);
    return first_x + ratio * (second_x - first_x);
  }

  std::vector<cv::Point> translateGripperPolygonEnvelope(
    const std::vector<int64_t> &closed_flat,
    const std::vector<int64_t> &middle_flat,
    const std::vector<int64_t> &open_flat, double anchor_x,
    int image_width, int image_height) const {
    const auto closed = decodeGripperPolygon(closed_flat);
    const auto middle = decodeGripperPolygon(middle_flat);
    const auto open = decodeGripperPolygon(open_flat);
    std::vector<cv::Point> result;
    if (closed.empty() || middle.empty() || open.empty()) return result;
    const double scale_x = static_cast<double>(image_width) /
      std::max(1, gripper_calibration_width_px_);
    const double scale_y = static_cast<double>(image_height) /
      std::max(1, gripper_calibration_height_px_);
    const double reference_anchor_x = middle.front().x;
    std::vector<cv::Point2f> aligned_points;
    aligned_points.reserve(closed.size() + middle.size() + open.size());
    const auto append_aligned = [&aligned_points, reference_anchor_x](
      const std::vector<cv::Point2f> &polygon) {
        const double alignment_x = reference_anchor_x - polygon.front().x;
        for (const auto &point : polygon) {
          aligned_points.emplace_back(
            static_cast<float>(point.x + alignment_x), point.y);
        }
      };
    append_aligned(closed);
    append_aligned(middle);
    append_aligned(open);

    // The convex envelope is calculated in the calibration frame, so its
    // shape remains fixed. Live joint feedback only supplies a horizontal
    // translation. Using all calibration silhouettes prevents clipped jaw or
    // wrist regions from becoming exposed near either opening limit.
    std::vector<cv::Point2f> envelope;
    cv::convexHull(aligned_points, envelope);
    const double offset_x = anchor_x - reference_anchor_x;
    result.reserve(envelope.size());
    for (const auto &point : envelope) {
      result.emplace_back(
        static_cast<int>(std::lround((point.x + offset_x) * scale_x)),
        static_cast<int>(std::lround(point.y * scale_y)));
    }
    return result;
  }

  cv::Mat buildGripperSelfMask(
    int image_width, int image_height, double gripper_width_m) const {
    cv::Mat mask(image_height, image_width, CV_8UC1, cv::Scalar(0));
    if (!enable_gripper_self_filter_ ||
        gripper_calibration_widths_m_.size() != 3) {
      return mask;
    }
    const double closed_width = gripper_calibration_widths_m_[0];
    const double middle_width = gripper_calibration_widths_m_[1];
    const double open_width = gripper_calibration_widths_m_[2];
    if (!(closed_width < middle_width && middle_width < open_width)) {
      return mask;
    }
    // Keep each jaw silhouette rigid. Only its horizontal location follows the
    // live gripper opening; changing every vertex had distorted the polygons
    // and exposed jaw pixels at intermediate openings.
    const double left_anchor_x = interpolateGripperAnchorX(
      gripper_width_m, gripper_left_closed_xy_, gripper_left_mid_xy_,
      gripper_left_open_xy_);
    const double right_anchor_x = interpolateGripperAnchorX(
      gripper_width_m, gripper_right_closed_xy_, gripper_right_mid_xy_,
      gripper_right_open_xy_);
    const auto left = translateGripperPolygonEnvelope(
      gripper_left_closed_xy_, gripper_left_mid_xy_, gripper_left_open_xy_,
      left_anchor_x, image_width, image_height);
    const auto right = translateGripperPolygonEnvelope(
      gripper_right_closed_xy_, gripper_right_mid_xy_, gripper_right_open_xy_,
      right_anchor_x, image_width, image_height);
    if (left.empty() || right.empty()) return mask;
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{left, right},
                 cv::Scalar(255));
    const int margin = std::max(0, gripper_self_filter_margin_px_);
    const int vertical_margin = std::max(
      0, gripper_self_filter_vertical_margin_px_);
    if (margin > 0 || vertical_margin > 0) {
      // Horizontal padding suppresses registered-depth edge artifacts. Keep
      // vertical padding small to avoid hiding objects immediately above the
      // jaw tips.
      const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2 * margin + 1, 2 * vertical_margin + 1));
      cv::dilate(mask, mask, kernel);
    }
    return mask;
  }

  void publishGripperSelfFilterDebug(
    const cv::Mat &mask, const cv::Mat &rgb,
    const std_msgs::msg::Header &header, double gripper_width_m) {
    if (mask.empty()) return;
    gripper_self_mask_pub_->publish(
      *cv_bridge::CvImage(header, "mono8", mask).toImageMsg());
    cv::Mat debug;
    if (!rgb.empty() && rgb.size() == mask.size()) {
      cv::Mat red_overlay(rgb.size(), CV_8UC3, cv::Scalar(0, 0, 255));
      debug = rgb.clone();
      cv::Mat blended;
      cv::addWeighted(rgb, 0.45, red_overlay, 0.55, 0.0, blended);
      blended.copyTo(debug, mask);
    } else {
      // A depth scan must always produce an inspectable mask even when no RGB
      // frame satisfies the synchronization tolerance.
      cv::cvtColor(mask, debug, cv::COLOR_GRAY2BGR);
    }
    std::ostringstream label;
    label << "gripper self-mask width=" << std::fixed << std::setprecision(3)
          << gripper_width_m << " m";
    cv::putText(debug, label.str(), cv::Point(10, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
    gripper_self_filter_debug_pub_->publish(
      *cv_bridge::CvImage(header, "bgr8", debug).toImageMsg());
  }

  void runYoloInference(const cv::Mat & captured_image,
                        const std_msgs::msg::Header & observation_header,
                        const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud_msg,
                        const Eigen::Matrix4f & t_base_camera,
                        const cv::Mat & gripper_self_mask,
                        std::shared_ptr<my_actions::srv::GetYoloInference::Response> response)
  {
      const cv::Mat &img = captured_image;
      if (img.empty()) { response->success = false; return; }

      if (!session_) {
          response->success = false;
          RCLCPP_INFO(this->get_logger(), "ONNX session not initialized");
          return;
      }

      // Rotate image upright based on the exact synchronized camera pose.
      float rot_angle = 0.0f;
      Eigen::Vector3f v_up_cam =
        t_base_camera.block<3,3>(0,0).transpose() * Eigen::Vector3f(0, 0, 1);
      if (std::sqrt(v_up_cam(0)*v_up_cam(0) + v_up_cam(1)*v_up_cam(1)) > 0.1f) {
          float current_angle = std::atan2(v_up_cam(1), v_up_cam(0)) * 180.0f / M_PI;
          rot_angle = 90.0f + current_angle;
      }

      pcl::PointCloud<pcl::PointXYZ> organized_cloud;
      pcl::fromROSMsg(*cloud_msg, organized_cloud);
      if (organized_cloud.width != static_cast<uint32_t>(img.cols) ||
          organized_cloud.height != static_cast<uint32_t>(img.rows)) {
        response->success = false;
        RCLCPP_ERROR(
          this->get_logger(),
          "YOLO RGB/cloud organization mismatch: image=%dx%d cloud=%ux%u",
          img.cols, img.rows, organized_cloud.width, organized_cloud.height);
        return;
      }

      cv::Mat upright_image;
      cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
      cv::Mat rot_mat = cv::getRotationMatrix2D(center, rot_angle, 1.0);
      // Rotating a rectangular image into its original-sized canvas clips the
      // corners (and, near 90 degrees, a large strip of the scene).  Grow the
      // inference canvas and retain the complete affine transform so masks can
      // be warped back onto the organized RGB-D pixel grid without data loss.
      const cv::Rect2f rotated_bounds =
        cv::RotatedRect(center, img.size(), rot_angle).boundingRect2f();
      rot_mat.at<double>(0, 2) -= rotated_bounds.x;
      rot_mat.at<double>(1, 2) -= rotated_bounds.y;
      const cv::Size upright_size(
        std::max(1, static_cast<int>(std::ceil(rotated_bounds.width))),
        std::max(1, static_cast<int>(std::ceil(rotated_bounds.height))));
      cv::warpAffine(img, upright_image, rot_mat, upright_size,
                     cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      
      // Preprocess for YOLO
      float ratio = std::min(640.0f / upright_image.cols, 640.0f / upright_image.rows);
      int unpad_w = std::round(upright_image.cols * ratio);
      int unpad_h = std::round(upright_image.rows * ratio);
      int pad_w = (640 - unpad_w) / 2;
      int pad_h = (640 - unpad_h) / 2;

      cv::Mat resized_img, padded_img;
      cv::resize(upright_image, resized_img, cv::Size(unpad_w, unpad_h));
      cv::copyMakeBorder(resized_img, padded_img, pad_h, 640 - unpad_h - pad_h, 
                         pad_w, 640 - unpad_w - pad_w, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
      cv::cvtColor(padded_img, padded_img, cv::COLOR_BGR2RGB);

      std::vector<float> input_tensor_values(1 * 3 * 640 * 640);
      for (int c = 0; c < 3; ++c) {
          for (int i = 0; i < 640 * 640; ++i) {
              input_tensor_values[c * 640 * 640 + i] = padded_img.data[i * 3 + c] / 255.0f;
          }
      }

      std::vector<int64_t> input_shape = {1, 3, 640, 640};
      Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

      const char* input_names[] = {input_name_.c_str()};
      const char* output_names[] = {output0_name_.c_str(), output1_name_.c_str()};

      auto start_time = std::chrono::high_resolution_clock::now();
      auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);
      
      float* floatarr = output_tensors[0].GetTensorMutableData<float>();
      float* protoarr = output_tensors[1].GetTensorMutableData<float>();
      auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
      int num_boxes = output_shape[1]; // 300
      int num_attrs = output_shape[2]; // 38 (x1, y1, x2, y2, conf, class_id, 32 mask_coeffs)
      float conf_threshold = 0.5f;
      cv::Mat protos(32, 25600, CV_32F, protoarr);

      cv::Mat debug_yolo = upright_image.clone();
      cv::Mat inverse_rot_mat;
      cv::invertAffineTransform(rot_mat, inverse_rot_mat);

      const auto median = [](std::vector<float> & values) {
        const size_t middle = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + middle, values.end());
        return values[middle];
      };

      for (int i = 0; i < num_boxes; ++i) {
          float* row = floatarr + i * num_attrs;
          float x1_raw = row[0];
          float y1_raw = row[1];
          float x2_raw = row[2];
          float y2_raw = row[3];
          float conf = row[4];
          int class_id = static_cast<int>(row[5]);

          if (conf > conf_threshold && class_id >= 0 && class_id < static_cast<int>(COCO_CLASSES.size())) {
              float x1 = (x1_raw - pad_w) / ratio;
              float y1 = (y1_raw - pad_h) / ratio;
              float x2 = (x2_raw - pad_w) / ratio;
              float y2 = (y2_raw - pad_h) / ratio;

              int u_min = std::max(0, static_cast<int>(x1));
              int v_min = std::max(0, static_cast<int>(y1));
              int u_max = std::min(upright_image.cols, static_cast<int>(std::ceil(x2)));
              int v_max = std::min(upright_image.rows, static_cast<int>(std::ceil(y2)));
              
              if (u_max <= u_min || v_max <= v_min) continue;

              // Compute Mask
              cv::Mat coeffs(1, 32, CV_32F, row + 6);
              cv::Mat mask = coeffs * protos; // 1 x 25600
              mask = mask.reshape(1, 160); // 160 x 160
              cv::exp(-mask, mask);
              mask = 1.0f / (1.0f + mask);

              int mx1 = std::max(0, static_cast<int>(x1_raw / 4.0f));
              int my1 = std::max(0, static_cast<int>(y1_raw / 4.0f));
              int mx2 = std::min(159, static_cast<int>(x2_raw / 4.0f));
              int my2 = std::min(159, static_cast<int>(y2_raw / 4.0f));
              
              if (mx2 <= mx1 || my2 <= my1) continue;

              cv::Rect m_roi(mx1, my1, mx2 - mx1, my2 - my1);
              cv::Mat mask_roi = mask(m_roi);
              
              // Resize the floating-point mask FIRST to preserve smooth gradients via interpolation
              cv::Mat resized_mask_f;
              cv::resize(mask_roi, resized_mask_f, cv::Size(u_max - u_min, v_max - v_min), 0, 0, cv::INTER_LINEAR);
              
              // Then threshold the high-resolution mask to get crisp, smooth boundaries
              cv::Mat resized_mask;
              cv::threshold(resized_mask_f, resized_mask, 0.5, 255.0, cv::THRESH_BINARY);
              resized_mask.convertTo(resized_mask, CV_8UC1);

              cv::Rect yolo_roi(u_min, v_min, u_max - u_min, v_max - v_min);
              cv::rectangle(debug_yolo, yolo_roi, cv::Scalar(255, 0, 0), 2);
              
              // Overlay Mask onto debug image
              cv::Mat color_mask(resized_mask.size(), CV_8UC3, class_id == 41 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255));
              cv::Mat img_roi = debug_yolo(yolo_roi);
              cv::Mat mask_overlay;
              cv::bitwise_and(color_mask, color_mask, mask_overlay, resized_mask);
              cv::addWeighted(img_roi, 1.0, mask_overlay, 0.5, 0.0, img_roi);

              std::string label = COCO_CLASSES[class_id] + " (" + std::to_string(conf).substr(0, 4) + ")";
              cv::putText(debug_yolo, label, cv::Point(u_min, std::max(15, v_min - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

              cv::Mat full_upright_mask = cv::Mat::zeros(upright_image.size(), CV_8UC1);
              resized_mask.copyTo(full_upright_mask(yolo_roi));
              cv::Mat full_raw_mask;
              cv::warpAffine(full_upright_mask, full_raw_mask, inverse_rot_mat, img.size(),
                             cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
              const cv::Rect raw_bbox = cv::boundingRect(full_raw_mask);
              if (raw_bbox.empty()) continue;

              // Localize the instance from the exact organized depth pixels
              // selected by its mask. A bounding-box center ray frequently
              // lands on a table through a hollow handle or mask gap.
              std::vector<float> base_x;
              std::vector<float> base_y;
              std::vector<float> base_z;
              base_x.reserve(cv::countNonZero(full_raw_mask));
              base_y.reserve(base_x.capacity());
              base_z.reserve(base_x.capacity());
              for (int raw_v = raw_bbox.y;
                   raw_v < raw_bbox.y + raw_bbox.height; ++raw_v) {
                const auto *mask_row = full_raw_mask.ptr<uint8_t>(raw_v);
                const auto *self_row =
                  (!gripper_self_mask.empty() &&
                   gripper_self_mask.size() == full_raw_mask.size()) ?
                  gripper_self_mask.ptr<uint8_t>(raw_v) : nullptr;
                for (int raw_u = raw_bbox.x;
                     raw_u < raw_bbox.x + raw_bbox.width; ++raw_u) {
                  if (mask_row[raw_u] == 0 ||
                      (self_row != nullptr && self_row[raw_u] != 0)) {
                    continue;
                  }
                  const auto &point = organized_cloud.at(raw_u, raw_v);
                  if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                      !std::isfinite(point.z) || point.z < 0.05F ||
                      point.z > 1.0F) {
                    continue;
                  }
                  const Eigen::Vector4f point_base = t_base_camera *
                    Eigen::Vector4f(point.x, point.y, point.z, 1.0F);
                  base_x.push_back(point_base.x());
                  base_y.push_back(point_base.y());
                  base_z.push_back(point_base.z());
                }
              }
              if (base_x.size() < 20) {
                RCLCPP_WARN(
                  this->get_logger(),
                  "YOLO discarded %s instance: only %zu valid masked depth pixels.",
                  COCO_CLASSES[class_id].c_str(), base_x.size());
                continue;
              }
              const float obj_x = median(base_x);
              const float obj_y = median(base_y);
              const float obj_z = median(base_z);

              response->classes.push_back(COCO_CLASSES[class_id]);
              response->confidences.push_back(conf);
              response->bbox_x_min.push_back(raw_bbox.x);
              response->bbox_y_min.push_back(raw_bbox.y);
              response->bbox_x_max.push_back(raw_bbox.x + raw_bbox.width);
              response->bbox_y_max.push_back(raw_bbox.y + raw_bbox.height);
              response->position_x.push_back(obj_x);
              response->position_y.push_back(obj_y);
              response->position_z.push_back(obj_z);

              auto mask_msg = cv_bridge::CvImage(observation_header, "mono8", full_raw_mask).toImageMsg();
              response->masks.push_back(*mask_msg);
          }
      }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        response->success = true;
        response->observation_header = observation_header;
        response->rgb_image = *cv_bridge::CvImage(observation_header, "bgr8", img).toImageMsg();
        {
          std::lock_guard<std::mutex> lock(yolo_mutex_);
          latest_yolo_response_ = *response;
          has_yolo_result_ = true;
        }
        publishYoloWorldObservations(*response);
        RCLCPP_DEBUG(
          this->get_logger(),
          "YOLO inference & mask processing time: %ld ms", duration);
    }

  void publishYoloWorldObservations(
    const my_actions::srv::GetYoloInference::Response & result)
  {
    if (!publish_yolo_world_observations_ ||
        !update_object_client_->service_is_ready()) {
      return;
    }
    for (const auto &tracked_class : yolo_world_classes_) {
      bool detected = false;
      for (size_t index = 0; index < result.classes.size(); ++index) {
        if (result.classes[index] != tracked_class ||
            index >= result.position_x.size() ||
            index >= result.position_y.size() ||
            index >= result.position_z.size()) {
          continue;
        }
        detected = true;
        auto request =
          std::make_shared<my_actions::srv::UpdateObjectObservation::Request>();
        request->name = tracked_class;
        request->detected = true;
        request->position.header = result.observation_header;
        request->position.header.frame_id = "base_link";
        request->position.point.x = result.position_x[index];
        request->position.point.y = result.position_y[index];
        request->position.point.z = result.position_z[index];
        update_object_client_->async_send_request(request);
      }
      if (!detected) {
        auto request =
          std::make_shared<my_actions::srv::UpdateObjectObservation::Request>();
        request->name = tracked_class;
        request->detected = false;
        request->position.header = result.observation_header;
        request->position.header.frame_id = "base_link";
        update_object_client_->async_send_request(request);
      }
    }
  }

  void handleGetYoloInference(const std::shared_ptr<my_actions::srv::GetYoloInference::Request>,
                              std::shared_ptr<my_actions::srv::GetYoloInference::Response> response)
  {
      std::lock_guard<std::mutex> lock(yolo_mutex_);
      if (!has_yolo_result_) {
          response->success = false;
          RCLCPP_WARN(this->get_logger(), "No completed observation YOLO result; trigger /active_vision/trigger_scan first.");
          return;
      }
      *response = latest_yolo_response_;
  }

  void handleTriggerScan(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
      scan_next_frame_ = true;
      response->success = true;
      response->message = "Scan triggered for next frame";
  }

  void handleReloadCalibration(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                               std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    (void)request;
    std::lock_guard<std::mutex> lock(scan_mutex_);
    load_calibration();
    response->success = has_calibration_;
    response->message = has_calibration_ ? "Calibration matrix reloaded successfully" : "Failed to reload calibration matrix";
    RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
  }

  void handleGetObstacleLocation(const std::shared_ptr<my_actions::srv::GetObstacleLocation::Request> request,
                                 std::shared_ptr<my_actions::srv::GetObstacleLocation::Response> response)
  {
    (void)request;
    std::lock_guard<std::mutex> lock(scan_mutex_);

    // last_obstacle_voxel_ is stored in CAMERA frame from camCountCallback.
    // We must transform it to base_link frame before returning to the master node.
    if (has_calibration_ && latest_pose_ && latest_pose_->position.size() >= 6) {
      std::vector<float> q_joints(6);
      for (int i = 0; i < 6; ++i) q_joints[i] = latest_pose_->position[i];

      Eigen::Matrix4f T_base_cam = getCameraTransform(q_joints);
      Eigen::Vector4f pt_cam(last_obstacle_voxel_.x, last_obstacle_voxel_.y, last_obstacle_voxel_.z, 1.0f);
      Eigen::Vector4f pt_base = T_base_cam * pt_cam;

      response->point[0] = pt_base.x();
      response->point[1] = pt_base.y();
      response->point[2] = pt_base.z();

      RCLCPP_INFO(this->get_logger(),
        "[OBSTACLE] Transformed cam-frame [%.3f, %.3f, %.3f] -> base_link [%.3f, %.3f, %.3f]",
        last_obstacle_voxel_.x, last_obstacle_voxel_.y, last_obstacle_voxel_.z,
        pt_base.x(), pt_base.y(), pt_base.z());
    } else {
      RCLCPP_WARN(this->get_logger(),
        "[OBSTACLE] Cannot transform to base_link (calibration=%s, pose=%s). Returning raw camera-frame coordinates!",
        has_calibration_ ? "OK" : "MISSING",
        (latest_pose_ && latest_pose_->position.size() >= 6) ? "OK" : "MISSING");
      response->point[0] = last_obstacle_voxel_.x;
      response->point[1] = last_obstacle_voxel_.y;
      response->point[2] = last_obstacle_voxel_.z;
    }
  }

  void handleGet3DCoordinate(const std::shared_ptr<my_actions::srv::Get3DCoordinate::Request> request,
                             std::shared_ptr<my_actions::srv::Get3DCoordinate::Response> response)
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    response->success = false;
    response->corrected_u = request->u;
    response->corrected_v = request->v;
    if (!observation_pcd_ || !observation_pose_ || !has_calibration_) {
      response->status = "NO_SYNCHRONIZED_OBSERVATION";
      return;
    }
    const rclcpp::Time requested_stamp(request->observation_header.stamp);
    const rclcpp::Time observation_stamp(observation_pcd_->header.stamp);
    if (requested_stamp.nanoseconds() != 0 && requested_stamp != observation_stamp) {
      response->status = "OBSERVATION_TIMESTAMP_MISMATCH";
      return;
    }
    if (observation_pcd_->height <= 1 || observation_pcd_->width == 0) {
      response->status = "UNORGANIZED_DEPTH_CLOUD";
      return;
    }

    pcl::PointCloud<pcl::PointXYZ> organized;
    pcl::fromROSMsg(*observation_pcd_, organized);
    if (!organized.isOrganized()) {
      response->status = "UNORGANIZED_DEPTH_CLOUD";
      response->success = false;
      return;
    }

    const int cloud_width = static_cast<int>(organized.width);
    const int cloud_height = static_cast<int>(organized.height);
    const int requested_width = std::max(1, request->width);
    const int requested_height = std::max(1, request->height);
    const int center_u = std::clamp(request->u + requested_width / 2, 0, cloud_width - 1);
    const int center_v = std::clamp(request->v + requested_height / 2, 0, cloud_height - 1);
    pcl::PointXYZ selected;
    int selected_u = center_u;
    int selected_v = center_v;
    double best_pixel_distance = std::numeric_limits<double>::infinity();
    const auto consider_pixel = [&](int u, int v) {
      if (!observation_self_mask_.empty() &&
          v < observation_self_mask_.rows && u < observation_self_mask_.cols &&
          observation_self_mask_.at<uint8_t>(v, u) != 0) {
        return;
      }
      const auto &point = organized.at(u, v);
      if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z) || point.z <= 0.05f || point.z > 1.5f) {
        return;
      }
      const double du = static_cast<double>(u - center_u);
      const double dv = static_cast<double>(v - center_v);
      const double distance = du * du + dv * dv;
      if (distance < best_pixel_distance) {
        best_pixel_distance = distance;
        selected = point;
        selected_u = u;
        selected_v = v;
      }
    };
    const int min_u = std::clamp(request->u, 0, cloud_width - 1);
    const int min_v = std::clamp(request->v, 0, cloud_height - 1);
    const int max_u = std::clamp(request->u + requested_width - 1, 0, cloud_width - 1);
    const int max_v = std::clamp(request->v + requested_height - 1, 0, cloud_height - 1);
    for (int v = min_v; v <= max_v; ++v) {
      for (int u = min_u; u <= max_u; ++u) consider_pixel(u, v);
    }
    // RoboBrain may point one or two pixels off a thin handle. Search only a
    // small image-space neighborhood; unlike the old OctoMap raycast this
    // retains the full-resolution object surface and cannot jump to a coarse
    // 5 cm obstacle voxel.
    if (!std::isfinite(best_pixel_distance)) {
      constexpr int correction_radius_pixels = 8;
      for (int radius = 1; radius <= correction_radius_pixels; ++radius) {
        for (int dv = -radius; dv <= radius; ++dv) {
          for (int du = -radius; du <= radius; ++du) {
            if (std::max(std::abs(du), std::abs(dv)) != radius) continue;
            const int u = center_u + du;
            const int v = center_v + dv;
            if (u >= 0 && u < cloud_width && v >= 0 && v < cloud_height) {
              consider_pixel(u, v);
            }
          }
        }
        if (std::isfinite(best_pixel_distance)) break;
      }
    }
    if (!std::isfinite(best_pixel_distance)) {
      response->status = "NO_VALID_DEPTH_NEAR_PIXEL";
      return;
    }

    std::vector<float> q_joints(6);
    if (observation_pose_->position.size() < q_joints.size()) {
      response->status = "OBSERVATION_JOINT_STATE_INVALID";
      return;
    }
    for (size_t i = 0; i < q_joints.size(); ++i) {
      q_joints[i] = observation_pose_->position[i];
    }
    const Eigen::Matrix4f base_from_camera = getCameraTransform(q_joints);
    const Eigen::Vector4f point_base = base_from_camera * Eigen::Vector4f(
      selected.x, selected.y, selected.z, 1.0f);
    response->result_header = observation_pcd_->header;
    response->result_header.frame_id = "base_link";
    response->corrected_u = selected_u;
    response->corrected_v = selected_v;
    response->x = point_base.x();
    response->y = point_base.y();
    response->z = point_base.z();
    response->status = selected_u == center_u && selected_v == center_v ?
      "OK" : "OK_PIXEL_CORRECTED";
    response->success = true;
  }

  void handleGetNBVPose(const std::shared_ptr<my_actions::srv::GetNBVPose::Request> request,
                        std::shared_ptr<my_actions::srv::GetNBVPose::Response> response)
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    if (!latest_pose_) {
      response->success = false;
      return;
    }

    float target_x = request->target_x;
    float target_y = request->target_y;
    float target_z = request->target_z;
    float radius = request->radius;
    if (radius <= 0.0f) {
      radius = 0.3f; // Default 30cm radius
    }

    std::vector<float> q_joints(6);
    for (int i = 0; i < 6 && i < static_cast<int>(latest_pose_->position.size()); ++i) {
      q_joints[i] = latest_pose_->position[i];
    }
    Eigen::Matrix4f T_base_cam = getCameraTransform(q_joints);
    Eigen::Vector3f current_cam_pos(T_base_cam(0,3), T_base_cam(1,3), T_base_cam(2,3));

    int num_samples = 150;
    float phi = 3.14159265f * (3.f - std::sqrt(5.f)); // golden angle
    
    struct ValidCandidate {
      Eigen::Vector3f pos;
      float dist_to_current;
    };
    std::vector<ValidCandidate> valid_candidates;

    for (int i = 0; i < num_samples; ++i) {
      float y_sphere = 1.0f - (i / float(num_samples - 1)) * 2.0f; 
      float radius_at_y = std::sqrt(1.0f - y_sphere * y_sphere);
      float theta = phi * i;

      float x_sphere = std::cos(theta) * radius_at_y;
      float z_sphere = std::sin(theta) * radius_at_y;
      
      Eigen::Vector3f candidate_pos(
        target_x + x_sphere * radius,
        target_y + y_sphere * radius,
        target_z + z_sphere * radius
      );
      
      // Filter out points that are below table or too far behind the object
      if (candidate_pos.z() < 0.1f) continue;
      
      // Check occlusion using OctoMap raycasting
      octomap::point3d candidate_octo(candidate_pos.x(), candidate_pos.y(), candidate_pos.z());
      octomap::point3d direction(target_x - candidate_pos.x(), target_y - candidate_pos.y(), target_z - candidate_pos.z());
      
      octomap::point3d hit_pt;
      bool hit = octree_->castRay(candidate_octo, direction, hit_pt, true, radius);
      
      bool occluded = false;
      if (hit) {
        // If it hit something, check if the hit point is very close to the target
        float hit_dist = std::sqrt(
          std::pow(hit_pt.x() - target_x, 2) +
          std::pow(hit_pt.y() - target_y, 2) +
          std::pow(hit_pt.z() - target_z, 2)
        );
        if (hit_dist > 0.05f) { // If hit is more than 5cm away from target, it's occluded
          occluded = true;
        }
      }
      
      if (!occluded) {
        ValidCandidate vc;
        vc.pos = candidate_pos;
        vc.dist_to_current = (candidate_pos - current_cam_pos).norm();
        valid_candidates.push_back(vc);
      }
    }
    
    // Sort by distance to current camera position to minimize travel
    std::sort(valid_candidates.begin(), valid_candidates.end(), 
      [](const ValidCandidate& a, const ValidCandidate& b) {
        return a.dist_to_current < b.dist_to_current;
      });
      
    for (const auto& vc : valid_candidates) {
      response->candidate_x.push_back(vc.pos.x());
      response->candidate_y.push_back(vc.pos.y());
      response->candidate_z.push_back(vc.pos.z());
    }
    
    response->success = !valid_candidates.empty();
  }

  void triggerEmergencyStop()
  {
    if (estop_client_->wait_for_service(std::chrono::milliseconds(100))) {
      auto req = std::make_shared<std_srvs::srv::Empty::Request>();
      estop_client_->async_send_request(req);
      RCLCPP_WARN(this->get_logger(), "EMERGENCY STOP TRIGGERED from ACTIVE VISION.");
    }
  }
  
  Eigen::Matrix4f getCameraTransform(const std::vector<float>& q_joints) {
    auto urdf_rot = [](float r, float p, float y) {
      return (Eigen::AngleAxisf(y, Eigen::Vector3f::UnitZ()) *
              Eigen::AngleAxisf(p, Eigen::Vector3f::UnitY()) *
              Eigen::AngleAxisf(r, Eigen::Vector3f::UnitX())).toRotationMatrix();
    };

    Eigen::Matrix4f T01 = Eigen::Matrix4f::Identity();
    T01(2, 3) = 0.123f;
    T01.block<3,3>(0,0) = urdf_rot(0, 0, 0) * Eigen::AngleAxisf(q_joints[0], Eigen::Vector3f::UnitZ()).toRotationMatrix();

    Eigen::Matrix4f T12 = Eigen::Matrix4f::Identity();
    T12.block<3,3>(0,0) = urdf_rot(1.5707963f, -0.1357866f, -3.1415926f) * Eigen::AngleAxisf(q_joints[1], Eigen::Vector3f::UnitZ()).toRotationMatrix();

    Eigen::Matrix4f T23 = Eigen::Matrix4f::Identity();
    T23(0, 3) = 0.28503f;
    T23.block<3,3>(0,0) = urdf_rot(0, 0, -1.7938494f) * Eigen::AngleAxisf(q_joints[2], Eigen::Vector3f::UnitZ()).toRotationMatrix();

    Eigen::Matrix4f T34 = Eigen::Matrix4f::Identity();
    T34(0, 3) = -0.02198f;
    T34(1, 3) = -0.25075f;
    T34.block<3,3>(0,0) = urdf_rot(1.5707963f, 0, 0) * Eigen::AngleAxisf(q_joints[3], Eigen::Vector3f::UnitZ()).toRotationMatrix();

    Eigen::Matrix4f T45 = Eigen::Matrix4f::Identity();
    T45.block<3,3>(0,0) = urdf_rot(-1.5707963f, 0, 0) * Eigen::AngleAxisf(q_joints[4], Eigen::Vector3f::UnitZ()).toRotationMatrix();

    Eigen::Matrix4f T56 = Eigen::Matrix4f::Identity();
    T56(0, 3) = 8.8259e-05f;
    T56(1, 3) = -0.091f;
    T56.block<3,3>(0,0) = urdf_rot(1.5707963f, 0, 0) * Eigen::AngleAxisf(q_joints[5], Eigen::Vector3f::UnitZ()).toRotationMatrix();

    Eigen::Matrix4f T06 = T01 * T12 * T23 * T34 * T45 * T56;
    // Translate in base_link after synchronized manual FK*X, once for all
    // observation, segmentation, obstacle and visualization consumers.
    return translatedPerceptionTransform(T06 * X_matrix_, perception_position_offset_base_m_);
  }

  void publishBaseObservationCloud(const sensor_msgs::msg::PointCloud2 &cloud,
                                   const Eigen::Matrix4f &transform,
                                   const cv::Mat &self_mask) {
    geometry_msgs::msg::TransformStamped camera_transform;
    camera_transform.header.stamp = cloud.header.stamp;
    camera_transform.header.frame_id = "base_link";
    camera_transform.child_frame_id = cloud.header.frame_id;
    camera_transform.transform.translation.x = transform(0, 3);
    camera_transform.transform.translation.y = transform(1, 3);
    camera_transform.transform.translation.z = transform(2, 3);
    const Eigen::Quaternionf rotation(transform.block<3, 3>(0, 0));
    camera_transform.transform.rotation.x = rotation.x();
    camera_transform.transform.rotation.y = rotation.y();
    camera_transform.transform.rotation.z = rotation.z();
    camera_transform.transform.rotation.w = rotation.w();
    observation_camera_transform_pub_->publish(camera_transform);
    // Use a visualization-only child name. Reusing the driver's optical-frame
    // name would give that frame two TF parents. The observation cloud and all
    // grasp markers are already in base_link; this frame only renders the
    // calibrated camera pose/axes in RViz.
    auto rviz_camera_transform = camera_transform;
    rviz_camera_transform.child_frame_id = "calibrated_camera_frame";
    calibrated_camera_tf_broadcaster_->sendTransform(rviz_camera_transform);

    sensor_msgs::msg::PointCloud2 output;
    output.header.stamp = cloud.header.stamp;
    output.header.frame_id = "base_link";
    sensor_msgs::PointCloud2Modifier modifier(output);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    output.width = cloud.width;
    output.height = cloud.height;
    output.row_step = output.point_step * output.width;
    output.data.resize(static_cast<size_t>(output.row_step) * output.height);
    output.is_dense = false;
    sensor_msgs::PointCloud2ConstIterator<float> in_x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> in_y(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> in_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> out_x(output, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(output, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(output, "z");
    size_t pixel_index = 0;
    for (; in_x != in_x.end(); ++in_x, ++in_y, ++in_z, ++out_x, ++out_y, ++out_z, ++pixel_index) {
      const int u = static_cast<int>(pixel_index % cloud.width);
      const int v = static_cast<int>(pixel_index / cloud.width);
      if (!self_mask.empty() && v < self_mask.rows && u < self_mask.cols &&
          self_mask.at<uint8_t>(v, u) != 0) {
        *out_x = *out_y = *out_z = std::numeric_limits<float>::quiet_NaN();
        continue;
      }
      // Orbbec may encode missing registered depth as a finite (0, 0, 0).
      // Transforming that value creates a dense false surface at the camera
      // origin in base_link, so validate optical depth before applying TF.
      if (!std::isfinite(*in_x) || !std::isfinite(*in_y) || !std::isfinite(*in_z) ||
          *in_z <= 0.05f || *in_z > 1.0f) {
        *out_x = *out_y = *out_z = std::numeric_limits<float>::quiet_NaN();
        continue;
      }
      const Eigen::Vector4f base = transform * Eigen::Vector4f(*in_x, *in_y, *in_z, 1.0f);
      *out_x = base.x(); *out_y = base.y(); *out_z = base.z();
    }
    observation_cloud_pub_->publish(output);
  }

  void syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg, 
                    const sensor_msgs::msg::JointState::ConstSharedPtr& pose_msg)
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    latest_pcd_ = cloud_msg;
    latest_pose_ = pose_msg;
    
    // Map publishing moved to a dedicated timer to ensure it's always visible
    if (!scanning_active_ || !scan_next_frame_) {
      return;
    }

    auto now = this->now();
    // Always unthrottled (process every frame at max FPS)
    last_proc_time_ = now;

    rclcpp::Time t_cloud = cloud_msg->header.stamp;

    // Successfully matched a tight frame! Consume the trigger.
    scan_next_frame_ = false;
    observation_pcd_ = cloud_msg;
    observation_pose_ = pose_msg;
    if (pose_msg->position.size() < 6) {
      RCLCPP_WARN(this->get_logger(),
                  "[OBSERVATION] Joint sample has fewer than 6 arm joints.");
      return;
    }
    std::vector<float> q_joints(6);
    for (int index = 0; index < 6; ++index) {
      q_joints[index] = pose_msg->position[index];
    }
    const Eigen::Matrix4f final_transform = getCameraTransform(q_joints);
    double gripper_width_m = 0.05;
    for (size_t index = 0;
         index < pose_msg->name.size() && index < pose_msg->position.size();
         ++index) {
      if (pose_msg->name[index] == "gripper") {
        gripper_width_m = std::clamp(
          std::abs(pose_msg->position[index]), 0.0, 0.1);
        break;
      }
    }
    cv::Mat gripper_self_mask = buildGripperSelfMask(
      static_cast<int>(cloud_msg->width), static_cast<int>(cloud_msg->height),
      gripper_width_m);
    observation_self_mask_ = gripper_self_mask;
    // The RGB image, cloud, and joint state below form one immutable observation.
    // YOLO receives this exact RGB copy asynchronously; it never reads a later frame.
    cv::Mat captured_rgb;
    std_msgs::msg::Header captured_rgb_header;
    double rgb_depth_offset_s = std::numeric_limits<double>::infinity();
    double rgb_receipt_age_s = std::numeric_limits<double>::infinity();
    {
      std::lock_guard<std::mutex> color_lock(color_mutex_);
      for (const auto &frame : recent_color_frames_) {
        const double offset = std::abs((rclcpp::Time(frame.header.stamp) - t_cloud).seconds());
        if (offset < rgb_depth_offset_s) {
          rgb_depth_offset_s = offset;
          captured_rgb = frame.image;
          captured_rgb_header = frame.header;
        }
      }
      if (rgb_depth_offset_s > rgb_depth_sync_tolerance_s_) {
        // Some Orbbec firmware/ROS combinations publish color and registered
        // cloud headers from different device-clock domains. Preserve strict
        // header matching when it works; otherwise accept only the newest RGB
        // frame received immediately before this cloud callback. The bounded
        // receipt age prevents a pre-motion/stale frame from being paired
        // with a later depth observation.
        constexpr double maximum_receipt_age_s = 0.50;
        const rclcpp::Time cloud_receipt_time = this->now();
        captured_rgb.release();
        for (auto frame = recent_color_frames_.rbegin();
             frame != recent_color_frames_.rend(); ++frame) {
          const double age =
            (cloud_receipt_time - frame->receipt_time).seconds();
          if (age < 0.0 || age > maximum_receipt_age_s) continue;
          captured_rgb = frame->image;
          captured_rgb_header = frame->header;
          rgb_receipt_age_s = age;
          break;
        }
      }
    }
    std_msgs::msg::Header self_mask_header = cloud_msg->header;
    if (!captured_rgb.empty()) {
      self_mask_header.frame_id = captured_rgb_header.frame_id;
    }
    publishGripperSelfFilterDebug(
      gripper_self_mask, captured_rgb, self_mask_header, gripper_width_m);
    if (captured_rgb.empty()) {
      // A scan is an RGB-D contract, not merely a depth snapshot. Re-arm the
      // trigger so the next synchronized frame can complete the observation.
      scan_next_frame_ = true;
      // This is expected while an on-demand scan waits for the next matching
      // RGB frame. Keep it available for synchronization debugging without
      // flooding the normal INFO/WARN console.
      RCLCPP_DEBUG(this->get_logger(),
                   "[OBSERVATION] Scan %ld has no RGB frame within %.3f s (nearest=%.3f s); map updated without YOLO.",
                   t_cloud.nanoseconds(), rgb_depth_sync_tolerance_s_, rgb_depth_offset_s);
    } else {
      RCLCPP_DEBUG(
        this->get_logger(),
        "[OBSERVATION] Captured RGB-D observation: RGB/depth timestamp "
        "offset=%.3f s receipt_age=%.3f s.",
        rgb_depth_offset_s, rgb_receipt_age_s);
      // The response's observation stamp identifies the depth frame whose
      // organized pixels the mask indexes.  Preserve the RGB optical frame ID
      // while replacing its nearby device timestamp with the exact cloud stamp.
      captured_rgb_header.stamp = cloud_msg->header.stamp;
      observation_image_pub_->publish(
        *cv_bridge::CvImage(captured_rgb_header, "bgr8", captured_rgb).toImageMsg());
      if (enable_yolo_inference_) {
        if (!yolo_inference_running_.exchange(true)) {
          std::thread([
            this, image = std::move(captured_rgb),
            header = captured_rgb_header, cloud = cloud_msg,
            transform = final_transform,
            self_mask = gripper_self_mask]() mutable {
            auto yolo_response = std::make_shared<my_actions::srv::GetYoloInference::Response>();
            runYoloInference(
              image, header, cloud, transform, self_mask, yolo_response);
            yolo_inference_running_ = false;
          }).detach();
        } else {
          RCLCPP_WARN(this->get_logger(), "[OBSERVATION] YOLO is still processing; dropped this debug snapshot (latest completed state retained).");
        }
      }
    }

    try {
      // 2. Camera-Frame Pre-Filtering: Parse raw depth points
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_raw(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::fromROSMsg(*cloud_msg, *pcl_raw);

      if (pcl_raw->empty()) return;

      // Preserve pixel organization while using the exact same manual
      // calibration transform as the obstacle world.
      publishBaseObservationCloud(*cloud_msg, final_transform, gripper_self_mask);

      const float v_size = static_cast<float>(obstacle_voxel_size_);
      
      struct VoxelData {
        int count = 0;
        Point3D best_pt = {0, 0, 0};
        float min_d2 = 1e9f;
      };
      
      std::unordered_map<std::tuple<int, int, int>, VoxelData, VoxelHash> voxel_grid;

      size_t masked_valid_depth_points = 0;
      for (size_t point_index = 0; point_index < pcl_raw->points.size(); ++point_index) {
        const auto &pt = pcl_raw->points[point_index];
        const int u = static_cast<int>(point_index % pcl_raw->width);
        const int v = static_cast<int>(point_index / pcl_raw->width);
        if (!gripper_self_mask.empty() &&
            v < gripper_self_mask.rows && u < gripper_self_mask.cols &&
            gripper_self_mask.at<uint8_t>(v, u) != 0) {
          if (std::isfinite(pt.z) && pt.z > 0.0f) ++masked_valid_depth_points;
          continue;
        }
        float z = pt.z;

        // Apply spatial constraints (e.g. max range 1.0m, min 12cm to ignore arm itself)
        if (std::isnan(z) || z < 0.12f || z > 1.00f) continue;

        std::tuple<int, int, int> idx = {
          static_cast<int>(std::floor(pt.x / v_size)),
          static_cast<int>(std::floor(pt.y / v_size)),
          static_cast<int>(std::floor(pt.z / v_size))
        };
        
        float center_x = (std::get<0>(idx) + 0.5f) * v_size;
        float center_y = (std::get<1>(idx) + 0.5f) * v_size;
        float center_z = (std::get<2>(idx) + 0.5f) * v_size;
        
        float d2 = (pt.x - center_x) * (pt.x - center_x) +
                   (pt.y - center_y) * (pt.y - center_y) +
                   (pt.z - center_z) * (pt.z - center_z);

        auto& vdata = voxel_grid[idx];
        vdata.count++;
        if (d2 < vdata.min_d2) {
          vdata.min_d2 = d2;
          vdata.best_pt = {pt.x, pt.y, pt.z};
        }
      }

      // Filter: Keep voxels with the configured sample count (100 by default),
      // then select the point closest to the 3D voxel center.
      std::vector<Point3D> filtered_cam_pts;
      filtered_cam_pts.reserve(voxel_grid.size());

      for (const auto & kv : voxel_grid) {
        if (kv.second.count >= obstacle_min_points_per_voxel_) {
          filtered_cam_pts.push_back(kv.second.best_pt);
        }
      }

      if (filtered_cam_pts.empty()) return;

      // 3. Matrix Transformation ONLY on surviving voxelized points (~1,000 points)
      // Exact Sensor Origin in base_link coordinates
      Eigen::Vector4f cam_origin_base = final_transform * Eigen::Vector4f(0, 0, 0, 1.0f);
      octomap::point3d sensor_origin(cam_origin_base.x(), cam_origin_base.y(), cam_origin_base.z());

      // Transform points to base_link and check workspace bounds
      octomap::Pointcloud oct_scan;
      for (const auto & pt : filtered_cam_pts) {
        Eigen::Vector4f p_cam(pt.x, pt.y, pt.z, 1.0f);
        Eigen::Vector4f p_base = final_transform * p_cam;

        if (p_base.x() < -0.1f || p_base.x() > 0.7f ||
            p_base.y() < -0.7f || p_base.y() > 0.7f ||
            p_base.z() < -0.05f || p_base.z() > 0.7f) {
          continue;
        }
        oct_scan.push_back(p_base.x(), p_base.y(), p_base.z());
      }

      // 4. Native In-Memory 3D Octree Raycasting
      octree_->insertPointCloud(oct_scan, sensor_origin);

      // 5. Publish Occupied Voxels for RViz2
      pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZ>());
      for (octomap::OcTree::leaf_iterator it = octree_->begin_leafs(), end = octree_->end_leafs(); it != end; ++it) {
        if (octree_->isNodeOccupied(*it)) {
          global_cloud->points.push_back(pcl::PointXYZ(it.getX(), it.getY(), it.getZ()));
        }
      }
      // Simultaneous one-pass pruning: neighbor counts use the original world,
      // so this never recursively erodes surfaces. Object observations are untouched.
      std::vector<VoxelCell> cells;
      cells.reserve(global_cloud->points.size());
      for (const auto &point : global_cloud->points) {
        cells.push_back({static_cast<int>(std::floor(point.x/v_size)),
          static_cast<int>(std::floor(point.y/v_size)),
          static_cast<int>(std::floor(point.z/v_size))});
      }
      const auto keep = supportedVoxels(cells);
      size_t destination = 0;
      for (size_t index = 0; index < keep.size(); ++index) {
        if (keep[index]) global_cloud->points[destination++] = global_cloud->points[index];
      }
      global_cloud->points.resize(destination);
      global_cloud->width = global_cloud->points.size();
      global_cloud->height = 1;
      global_cloud->is_dense = true;

      sensor_msgs::msg::PointCloud2 out_msg;
      pcl::toROSMsg(*global_cloud, out_msg);
      out_msg.header.stamp = cloud_msg->header.stamp;
      out_msg.header.frame_id = "base_link";

      global_map_pub_->publish(out_msg);
      RCLCPP_DEBUG(
        this->get_logger(),
        "[SELF FILTER] width=%.3f m mask=%d px removed_depth=%zu global_voxels=%zu",
        gripper_width_m, cv::countNonZero(gripper_self_mask),
        masked_valid_depth_points, global_cloud->points.size());

    } catch (const std::exception & ex) {
      RCLCPP_WARN(this->get_logger(), "Exception in active vision server: %s", ex.what());
    }
  }

  void camCountCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& raw_msg)
  {
    // Correct once at ingress, around the optical origin, before manual FK*X.
    // Both the obstacle world and every synchronized grasp/placement frame use it.
    const auto msg = perception_measurement_ratio_ == 1. ? raw_msg :
      std::make_shared<const sensor_msgs::msg::PointCloud2>(
        correct_camera_measurements(*raw_msg, perception_measurement_ratio_));
    if (corrected_depth_pub_->get_subscription_count() > 0) {
      corrected_depth_pub_->publish(corrected_registered_depth(*msg));
    }
    if (collision_monitor_active_) {
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        
        struct VoxelInfo {
            int count = 0;
            float min_z = 1e9f;
            float center_x = 0.0f;
            float center_y = 0.0f;
        };
        
        std::unordered_map<std::tuple<int, int, int>, VoxelInfo, VoxelHash> voxel_grid;
        float v_size = 0.040f; // 4.0cm voxel size
        
        for (; iter_z != iter_z.end(); ++iter_x, ++iter_y, ++iter_z) {
            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;
            
            if (std::isnan(z) || z <= 0.0f) continue;
            
            std::tuple<int, int, int> idx = {
              static_cast<int>(std::floor(x / v_size)),
              static_cast<int>(std::floor(y / v_size)),
              static_cast<int>(std::floor(z / v_size))
            };
            
            auto& v = voxel_grid[idx];
            v.count++;
            if (z < v.min_z) {
                v.min_z = z;
                v.center_x = (std::get<0>(idx) + 0.5f) * v_size;
                v.center_y = (std::get<1>(idx) + 0.5f) * v_size;
            }
        }
        
          float min_valid_z = 100.0f;
          int collision_voxel_count = 0;
          Point3D closest_obstacle = {0.0f, 0.0f, 100.0f};

          for (const auto& kv : voxel_grid) {
              if (kv.second.count >= 10) { // Require at least 10 points to eliminate noise
                  if (kv.second.min_z < min_valid_z) {
                      min_valid_z = kv.second.min_z;
                  }
                  if (kv.second.min_z < 0.15f) { // Collision Trick: object within 15cm
                      collision_voxel_count++;
                      if (kv.second.min_z < closest_obstacle.z) {
                          closest_obstacle.x = kv.second.center_x;
                          closest_obstacle.y = kv.second.center_y;
                          closest_obstacle.z = kv.second.min_z;
                      }
                  }
              }
          }

          if (collision_voxel_count >= 3) {
              {
                  std::lock_guard<std::mutex> lock(scan_mutex_);
                  last_obstacle_voxel_.x = closest_obstacle.x;
                  last_obstacle_voxel_.y = closest_obstacle.y;
                  last_obstacle_voxel_.z = closest_obstacle.z;
              }
            std_msgs::msg::Bool alert;
            alert.data = true;
            collision_alert_pub_->publish(alert);
            RCLCPP_WARN(this->get_logger(), "[COLLISION MONITOR] Obstacle detected at %.3fm! Publishing alert on /active_vision/collision_alert and firing brakes!", min_valid_z);
            triggerEmergencyStop();
        }
    }
    cam_count_++;

    sensor_msgs::msg::JointState::ConstSharedPtr latest_pose;
    {
      std::lock_guard<std::mutex> lock(pose_mutex_);
      latest_pose = latest_raw_pose_;
    }
    if (latest_pose) {
      syncCallback(msg, latest_pose);
    }
  }

  void poseCountCallback(const sensor_msgs::msg::JointState::ConstSharedPtr& msg) {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    latest_raw_pose_ = msg;
    pose_count_++;
  }

  bool scanning_active_ = true;
  bool scan_next_frame_ = false;
  bool collision_monitor_active_;
  std::mutex scan_mutex_;
  Point3D last_obstacle_voxel_ = {0.0f, 0.0f, 0.0f};

  std::mutex pose_mutex_;
  std::string hand_eye_calibration_path_;
  Eigen::Matrix4f X_matrix_;
  bool has_calibration_ = false;
  double obstacle_voxel_size_ = 0.05;
  int obstacle_min_points_per_voxel_ = 100;

  std::unique_ptr<octomap::OcTree> octree_;
  rclcpp::Time last_proc_time_;

  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pc_sub_;
  message_filters::Subscriber<sensor_msgs::msg::JointState> pose_sub_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::JointState> SyncPolicy;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr observation_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr observation_image_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr
    observation_camera_transform_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster>
    calibrated_camera_tf_broadcaster_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr gripper_self_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr gripper_self_filter_debug_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr collision_monitor_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr collision_alert_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr trigger_scan_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reload_calib_srv_;
  rclcpp::Service<my_actions::srv::GetObstacleLocation>::SharedPtr get_obstacle_srv_;
  rclcpp::Service<my_actions::srv::Get3DCoordinate>::SharedPtr get_3d_coord_srv_;
  rclcpp::Service<my_actions::srv::GetNBVPose>::SharedPtr get_nbv_srv_;
  rclcpp::Client<std_srvs::srv::Empty>::SharedPtr estop_client_;
  rclcpp::Client<my_actions::srv::UpdateObjectObservation>::SharedPtr
    update_object_client_;

  Eigen::Vector3f perception_position_offset_base_m_ = Eigen::Vector3f::Zero();
  double perception_measurement_ratio_ = 1.;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr corrected_depth_pub_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_pcd_;
  sensor_msgs::msg::JointState::ConstSharedPtr latest_pose_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr observation_pcd_;
  sensor_msgs::msg::JointState::ConstSharedPtr observation_pose_;
  cv::Mat observation_self_mask_;
  sensor_msgs::msg::JointState::ConstSharedPtr latest_raw_pose_;


  rclcpp::Service<my_actions::srv::GetYoloInference>::SharedPtr get_yolo_srv_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  bool enable_gripper_self_filter_ = true;
  int gripper_self_filter_margin_px_ = 10;
  int gripper_self_filter_vertical_margin_px_ = 8;
  int gripper_calibration_width_px_ = 640;
  int gripper_calibration_height_px_ = 480;
  std::vector<double> gripper_calibration_widths_m_;
  std::vector<int64_t> gripper_left_closed_xy_;
  std::vector<int64_t> gripper_left_mid_xy_;
  std::vector<int64_t> gripper_left_open_xy_;
  std::vector<int64_t> gripper_right_closed_xy_;
  std::vector<int64_t> gripper_right_mid_xy_;
  std::vector<int64_t> gripper_right_open_xy_;
  cv::Mat latest_color_image_;
  std_msgs::msg::Header latest_color_header_;
  struct ColorFrame {
    std_msgs::msg::Header header;
    cv::Mat image;
    rclcpp::Time receipt_time;
  };
  std::deque<ColorFrame> recent_color_frames_;
  double rgb_depth_sync_tolerance_s_ = 0.15;
  std::mutex color_mutex_;
  std::mutex yolo_mutex_;
  my_actions::srv::GetYoloInference::Response latest_yolo_response_;
  bool has_yolo_result_ = false;
  bool enable_yolo_inference_ = false;
  bool publish_yolo_world_observations_ = false;
  std::vector<std::string> yolo_world_classes_;
  std::atomic<bool> yolo_inference_running_{false};
  std::unique_ptr<Ort::Env> env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output0_name_;
  std::string output1_name_;
  std::atomic<int> cam_count_{0};
  std::atomic<int> pose_count_{0};
};

} // namespace vision_node

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vision_node::ActiveVisionServer)

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vision_node::ActiveVisionServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
