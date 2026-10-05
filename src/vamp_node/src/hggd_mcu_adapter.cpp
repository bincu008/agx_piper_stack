#include "vamp_node/hggd_mcu_adapter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace vamp_node {
namespace {

constexpr int kNetworkWidth = 320;
constexpr int kNetworkHeight = 160;
constexpr int kFeatureWidth = 40;
constexpr int kFeatureHeight = 20;
constexpr int kAnchorCount = 6;
constexpr int kOrientationCount = 49;
constexpr int kMaximumCenters = 64;
constexpr int kGroupPoints = 512;
constexpr float kPi = 3.14159265358979323846F;

constexpr std::array<float, 7> kGammaAnchors{
  -0.6801842467F, -0.4057291142F, -0.2010591122F, -0.0118806262F,
   0.1776760781F,  0.3838477194F,  0.6558036250F};
constexpr std::array<float, 7> kBetaAnchors{
  -0.8427982676F, -0.5439105884F, -0.2614813184F,  0.0099522572F,
   0.2803918363F,  0.5589576439F,  0.8524077841F};

float sigmoid(const float value) {
  if (value >= 0.0F) {
    const float exponential = std::exp(-value);
    return 1.0F / (1.0F + exponential);
  }
  const float exponential = std::exp(value);
  return exponential / (1.0F + exponential);
}

bool finiteVector(const Eigen::Vector3f &value) {
  return value.allFinite() && value.squaredNorm() > 1.0e-12F;
}

struct AnchorCenter {
  int network_u = 0;
  int network_v = 0;
  int image_u = 0;
  int image_v = 0;
  float heat_score = 0.0F;
  float theta = 0.0F;
  float width_pixels = 0.0F;
  float width_m = 0.0F;
  float depth_offset_mm = 0.0F;
  Eigen::Vector3f camera_center = Eigen::Vector3f::Zero();
};

struct SceneFeaturePoint {
  Eigen::Vector3f camera_point = Eigen::Vector3f::Zero();
  std::array<float, 32> features{};
};

int tensorIndex(const int channel, const int u, const int v,
                const int width, const int height) {
  return channel * width * height + u * height + v;
}

cv::Mat imageView(const sensor_msgs::msg::Image &image) {
  if (image.encoding != "rgb8" && image.encoding != "bgr8") {
    throw std::invalid_argument("HGGD requires rgb8 or bgr8 observation image");
  }
  if (image.step < image.width * 3U ||
      image.data.size() < static_cast<std::size_t>(image.step) * image.height) {
    throw std::invalid_argument("HGGD RGB image has invalid storage");
  }
  return cv::Mat(static_cast<int>(image.height), static_cast<int>(image.width),
    CV_8UC3, const_cast<unsigned char *>(image.data.data()), image.step);
}

cv::Mat maskView(const sensor_msgs::msg::Image &mask) {
  if (mask.encoding != "mono8" && mask.encoding != "8UC1") {
    throw std::invalid_argument("HGGD target mask must be mono8");
  }
  if (mask.step < mask.width ||
      mask.data.size() < static_cast<std::size_t>(mask.step) * mask.height) {
    throw std::invalid_argument("HGGD target mask has invalid storage");
  }
  return cv::Mat(static_cast<int>(mask.height), static_cast<int>(mask.width),
    CV_8UC1, const_cast<unsigned char *>(mask.data.data()), mask.step);
}

std::vector<std::size_t> deterministicFarthestSamples(
    const std::vector<std::size_t> &members,
    const std::vector<SceneFeaturePoint> &scene,
    const Eigen::Vector3f &center) {
  std::vector<std::size_t> selected;
  selected.reserve(kGroupPoints);
  if (members.empty()) return selected;
  if (members.size() <= static_cast<std::size_t>(kGroupPoints)) {
    selected = members;
    for (std::size_t cursor = 0; selected.size() < kGroupPoints; ++cursor) {
      selected.push_back(members[cursor % members.size()]);
    }
    return selected;
  }
  std::vector<std::size_t> pool;
  constexpr std::size_t maximum_pool = 4096;
  const std::size_t stride = std::max<std::size_t>(1, members.size() / maximum_pool);
  for (std::size_t index = 0; index < members.size() && pool.size() < maximum_pool;
       index += stride) pool.push_back(members[index]);
  std::vector<float> nearest_distance(pool.size(),
    std::numeric_limits<float>::infinity());
  std::size_t current = 0;
  float greatest_radius = -1.0F;
  for (std::size_t index = 0; index < pool.size(); ++index) {
    const float radius = (scene[pool[index]].camera_point - center).squaredNorm();
    if (radius > greatest_radius) { greatest_radius = radius; current = index; }
  }
  for (int sample = 0; sample < kGroupPoints; ++sample) {
    selected.push_back(pool[current]);
    const Eigen::Vector3f chosen = scene[pool[current]].camera_point;
    float farthest = -1.0F;
    std::size_t next = current;
    for (std::size_t index = 0; index < pool.size(); ++index) {
      nearest_distance[index] = std::min(nearest_distance[index],
        (scene[pool[index]].camera_point - chosen).squaredNorm());
      if (nearest_distance[index] > farthest) {
        farthest = nearest_distance[index];
        next = index;
      }
    }
    current = next;
  }
  return selected;
}

}  // namespace

HggdMcuAdapter::HggdMcuAdapter(const std::string &model_directory,
                               const int intra_op_threads) {
  environment_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "hggd_mcu");
  session_options_ = std::make_unique<Ort::SessionOptions>();
  session_options_->SetIntraOpNumThreads(std::max(1, intra_op_threads));
  session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  const auto make_session = [&](const char *name) {
    const auto path = std::filesystem::path(model_directory) / name;
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("HGGD model missing: " + path.string());
    }
    return std::make_unique<Ort::Session>(*environment_, path.c_str(), *session_options_);
  };
  resnet_ = make_session("resnet.onnx");
  anchornet_ = make_session("anchornet.onnx");
  pointnet_ = make_session("pointnet.onnx");
  localnet_ = make_session("localnet.onnx");
}

HggdMcuAdapter::~HggdMcuAdapter() = default;

std::vector<GraspProposal> HggdMcuAdapter::detect(
    const sensor_msgs::msg::PointCloud2 &organized_cloud,
    const sensor_msgs::msg::Image &rgb_image,
    const sensor_msgs::msg::Image &target_mask,
    const HggdCameraIntrinsics &intrinsics,
    const Eigen::Isometry3f &cloud_to_camera,
    const Eigen::Isometry3f &camera_to_base,
    const float contact_depth_m,
    const std::size_t maximum_proposals,
    cv::Mat &heatmap_overlay,
    HggdMcuDiagnostics &diagnostics) {
  diagnostics = HggdMcuDiagnostics{};
  if (organized_cloud.height <= 1 || rgb_image.width != organized_cloud.width ||
      rgb_image.height != organized_cloud.height || target_mask.width != rgb_image.width ||
      target_mask.height != rgb_image.height) {
    throw std::invalid_argument("HGGD requires synchronized organized RGB-D and target mask");
  }
  if (!(intrinsics.fx > 0.0F && intrinsics.fy > 0.0F) ||
      !cloud_to_camera.matrix().allFinite() || !camera_to_base.matrix().allFinite()) {
    throw std::invalid_argument("HGGD received invalid camera calibration");
  }
  const cv::Mat source_rgb = imageView(rgb_image);
  const cv::Mat source_mask = maskView(target_mask);
  const int image_width = static_cast<int>(rgb_image.width);
  const int image_height = static_cast<int>(rgb_image.height);
  diagnostics.target_pixels = static_cast<std::size_t>(cv::countNonZero(source_mask));
  if (diagnostics.target_pixels == 0) throw std::invalid_argument("HGGD target mask is empty");

  std::vector<Eigen::Vector3f> camera_points(
    static_cast<std::size_t>(image_width) * image_height,
    Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()));
  cv::Mat source_depth(image_height, image_width, CV_32FC1, cv::Scalar(0.0F));
  sensor_msgs::PointCloud2ConstIterator<float> point_x(organized_cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> point_y(organized_cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> point_z(organized_cloud, "z");
  for (int v = 0; v < image_height; ++v) {
    for (int u = 0; u < image_width; ++u, ++point_x, ++point_y, ++point_z) {
      if (!std::isfinite(*point_x) || !std::isfinite(*point_y) || !std::isfinite(*point_z)) continue;
      const Eigen::Vector3f camera = cloud_to_camera * Eigen::Vector3f(*point_x, *point_y, *point_z);
      if (!camera.allFinite() || camera.z() <= 0.02F || camera.z() > 3.0F) continue;
      camera_points[static_cast<std::size_t>(v) * image_width + u] = camera;
      source_depth.at<float>(v, u) = camera.z();
    }
  }

  cv::Mat rgb_network, depth_network, mask_network;
  cv::resize(source_rgb, rgb_network, cv::Size(kNetworkWidth, kNetworkHeight));
  cv::resize(source_depth, depth_network, cv::Size(kNetworkWidth, kNetworkHeight), 0, 0, cv::INTER_NEAREST);
  cv::resize(source_mask, mask_network, cv::Size(kNetworkWidth, kNetworkHeight), 0, 0, cv::INTER_NEAREST);
  // Keep the tensor organized (HGGD coordinates depend on it), but remove all
  // RGB-D content outside the segmented target. This is a logical crop without
  // changing camera intrinsics or pixel-to-point correspondence.
  const float depth_mean = static_cast<float>(
      cv::mean(depth_network, (mask_network != 0) & (depth_network > 0.0F))[0]);
  std::vector<float> input(4 * kNetworkWidth * kNetworkHeight, 0.0F);
  for (int v = 0; v < kNetworkHeight; ++v) for (int u = 0; u < kNetworkWidth; ++u) {
    if (mask_network.at<uint8_t>(v, u) == 0) continue;
    const cv::Vec3b pixel = rgb_network.at<cv::Vec3b>(v, u);
    const int index = u * kNetworkHeight + v;
    const float depth = depth_network.at<float>(v, u);
    input[index] = depth > 0.0F ?
      std::clamp(depth - depth_mean, -1.0F, 1.0F) : 0.0F;
    input[kNetworkWidth * kNetworkHeight + index] = (rgb_image.encoding == "bgr8" ? pixel[2] : pixel[0]) / 255.0F;
    input[2 * kNetworkWidth * kNetworkHeight + index] = pixel[1] / 255.0F;
    input[3 * kNetworkWidth * kNetworkHeight + index] = (rgb_image.encoding == "bgr8" ? pixel[0] : pixel[2]) / 255.0F;
  }

  std::lock_guard<std::mutex> lock(inference_mutex_);
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::array<int64_t, 4> input_shape{1, 4, kNetworkWidth, kNetworkHeight};
  auto input_tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), input_shape.data(), input_shape.size());
  const auto backbone_started = std::chrono::steady_clock::now();
  const char *resnet_input_names[] = {"rgbd_image"};
  const char *resnet_output_names[] = {"layer_1_features", "layer_2_features", "layer_3_features", "layer_4_features", "layer_5_features"};
  auto backbone_features = resnet_->Run(Ort::RunOptions{nullptr}, resnet_input_names, &input_tensor, 1, resnet_output_names, 5);
  const char *anchor_input_names[] = {"layer_1_features", "layer_2_features", "layer_3_features", "layer_4_features", "layer_5_features"};
  const char *anchor_output_names[] = {"loc_map", "cls_mask", "theta_offset", "depth_offset", "width_offset", "perpoint_features"};
  auto anchor_outputs = anchornet_->Run(Ort::RunOptions{nullptr}, anchor_input_names, backbone_features.data(), backbone_features.size(), anchor_output_names, 6);
  diagnostics.backbone_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - backbone_started).count();

  const float *location_data = anchor_outputs[0].GetTensorData<float>();
  const float *class_data = anchor_outputs[1].GetTensorData<float>();
  const float *theta_data = anchor_outputs[2].GetTensorData<float>();
  const float *depth_data = anchor_outputs[3].GetTensorData<float>();
  const float *width_data = anchor_outputs[4].GetTensorData<float>();
  const float *perpoint_data = anchor_outputs[5].GetTensorData<float>();
  cv::Mat heatmap(kNetworkHeight, kNetworkWidth, CV_32FC1);
  for (int v = 0; v < kNetworkHeight; ++v) for (int u = 0; u < kNetworkWidth; ++u)
    heatmap.at<float>(v, u) = sigmoid(location_data[u * kNetworkHeight + v]);
  cv::GaussianBlur(heatmap, heatmap, cv::Size(9, 9), 10.0);
  heatmap.setTo(0.0F, mask_network == 0);
  cv::Mat normalized_heatmap, colored_heatmap;
  cv::normalize(heatmap, normalized_heatmap, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);
  cv::applyColorMap(normalized_heatmap, colored_heatmap, cv::COLORMAP_JET);
  cv::resize(colored_heatmap, colored_heatmap, source_rgb.size());
  cv::Mat source_bgr;
  if (rgb_image.encoding == "rgb8") cv::cvtColor(source_rgb, source_bgr, cv::COLOR_RGB2BGR);
  else source_bgr = source_rgb;
  cv::Mat blended;
  cv::addWeighted(source_bgr, 0.45, colored_heatmap, 0.55, 0.0, blended);
  heatmap_overlay = source_bgr.clone();
  blended.copyTo(heatmap_overlay, source_mask);

  struct GridMaximum { int u; int v; float score; };
  std::vector<GridMaximum> maxima;
  for (int gu = 0; gu < kFeatureWidth; ++gu) for (int gv = 0; gv < kFeatureHeight; ++gv) {
    GridMaximum maximum{gu * 8, gv * 8, -1.0F};
    for (int u = gu * 8; u < (gu + 1) * 8; ++u) for (int v = gv * 8; v < (gv + 1) * 8; ++v) {
      const float score = heatmap.at<float>(v, u);
      if (score > maximum.score) maximum = {u, v, score};
    }
    maxima.push_back(maximum);
  }
  std::stable_sort(maxima.begin(), maxima.end(), [](const auto &a, const auto &b) { return a.score > b.score; });
  std::vector<AnchorCenter> centers;
  const float scale_u = static_cast<float>(image_width) / kNetworkWidth;
  const float scale_v = static_cast<float>(image_height) / kNetworkHeight;
  for (const auto &maximum : maxima) {
    if (maximum.score <= 0.01F || centers.size() >= kMaximumCenters) break;
    const int gu = maximum.u / 8, gv = maximum.v / 8;
    int best_anchor = 0;
    float best_class = -1.0F;
    for (int anchor = 0; anchor < kAnchorCount; ++anchor) {
      const float score = sigmoid(class_data[tensorIndex(anchor, gu, gv, kFeatureWidth, kFeatureHeight)]);
      if (score > best_class) { best_class = score; best_anchor = anchor; }
    }
    const int attribute_index = tensorIndex(best_anchor, gu, gv, kFeatureWidth, kFeatureHeight);
    const float theta = kPi / kAnchorCount * (best_anchor + std::clamp(theta_data[attribute_index], -0.5F, 0.5F) + 0.5F) - kPi * 0.5F;
    bool duplicate = false;
    for (const auto &existing : centers) {
      const float distance = std::hypot(float(maximum.u - existing.network_u), float(maximum.v - existing.network_v));
      float angle = std::abs(theta - existing.theta); angle = std::min(kPi - angle, angle);
      if (distance < 8.0F && angle < kPi / 6.0F) { duplicate = true; break; }
    }
    if (duplicate) continue;
    AnchorCenter center;
    center.network_u = maximum.u; center.network_v = maximum.v;
    center.image_u = std::clamp(int(std::lround((maximum.u + 0.5F) * scale_u - 0.5F)), 0, image_width - 1);
    center.image_v = std::clamp(int(std::lround((maximum.v + 0.5F) * scale_v - 0.5F)), 0, image_height - 1);
    center.heat_score = maximum.score; center.theta = theta;
    center.depth_offset_mm = 40.0F * std::clamp(depth_data[attribute_index], -0.5F, 0.5F);
    center.width_pixels = 50.0F * std::exp(std::clamp(width_data[attribute_index], -1.0F, 1.0F));
    Eigen::Vector3f measured = Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
    int selected_u = center.image_u, selected_v = center.image_v;
    // Reflective/concave objects commonly have a depth hole exactly where
    // AnchorNet places its RGB heatmap maximum (for example, inside a metal
    // bowl). Search farther for measured geometry, but remain strictly inside
    // the EdgeSAM target mask so the snap cannot land on the tray/table.
    constexpr int maximum_masked_depth_snap_pixels = 32;
    for (int radius = 0;
         radius <= maximum_masked_depth_snap_pixels && !measured.allFinite();
         ++radius) for (int dv = -radius;
         dv <= radius && !measured.allFinite(); ++dv)
      for (int du = -radius; du <= radius; ++du) {
      if (std::max(std::abs(du), std::abs(dv)) != radius) continue;
      const int u = center.image_u + du, v = center.image_v + dv;
      if (u < 0 || v < 0 || u >= image_width || v >= image_height || source_mask.at<unsigned char>(v, u) == 0) continue;
      const auto &candidate = camera_points[static_cast<std::size_t>(v) * image_width + u];
      if (candidate.allFinite()) { measured = candidate; selected_u = u; selected_v = v; break; }
    }
    if (!measured.allFinite()) continue;
    const float depth = measured.z() + center.depth_offset_mm * 0.001F;
    if (!std::isfinite(depth) || depth <= 0.02F) continue;
    center.image_u = selected_u; center.image_v = selected_v;
    center.camera_center = {(selected_u - intrinsics.cx) * depth / intrinsics.fx, (selected_v - intrinsics.cy) * depth / intrinsics.fy, depth};
    const float metres_per_pixel = depth * std::sqrt(std::pow(std::cos(theta) * scale_u / intrinsics.fx, 2.0F) + std::pow(std::sin(theta) * scale_v / intrinsics.fy, 2.0F));
    center.width_m = center.width_pixels * metres_per_pixel;
    if (center.camera_center.allFinite() && std::isfinite(center.width_m) && center.width_m > 0.0F) centers.push_back(center);
  }
  diagnostics.heatmap_centers = centers.size();
  if (centers.empty()) return {};

  // Thin affordances such as a cup rim can be only a few pixels wide. The
  // previous uniform stride over the whole image discarded most of those
  // target pixels before PointNet grouping. Preserve the grounded object at
  // native organized-cloud resolution and bound only non-target context.
  std::vector<std::size_t> target_pixels;
  std::vector<std::size_t> context_pixels;
  target_pixels.reserve(diagnostics.target_pixels);
  context_pixels.reserve(camera_points.size() / 4);
  for (std::size_t i = 0; i < camera_points.size(); ++i) {
    if (!camera_points[i].allFinite()) continue;
    const int u = static_cast<int>(i % static_cast<std::size_t>(image_width));
    const int v = static_cast<int>(i / static_cast<std::size_t>(image_width));
    if (source_mask.at<unsigned char>(v, u) != 0) {
      target_pixels.push_back(i);
    } else {
      context_pixels.push_back(i);
    }
  }
  constexpr std::size_t maximum_context_points = 25600;
  const std::size_t context_stride = std::max<std::size_t>(
    1, (context_pixels.size() + maximum_context_points - 1) /
      maximum_context_points);
  std::array<Eigen::Vector3f, kFeatureWidth * kFeatureHeight> feature_points;
  for (auto &point : feature_points) {
    point = Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
  }
  for (int fu = 0; fu < kFeatureWidth; ++fu) for (int fv = 0; fv < kFeatureHeight; ++fv) {
    const int image_u = std::clamp(int(std::lround((fu + 0.5F) * image_width / float(kFeatureWidth) - 0.5F)), 0, image_width - 1);
    const int image_v = std::clamp(int(std::lround((fv + 0.5F) * image_height / float(kFeatureHeight) - 0.5F)), 0, image_height - 1);
    feature_points[fu * kFeatureHeight + fv] =
      camera_points[static_cast<std::size_t>(image_v) * image_width + image_u];
  }
  std::vector<std::size_t> scene_pixels = target_pixels;
  scene_pixels.reserve(target_pixels.size() + std::min(
    context_pixels.size(), maximum_context_points));
  for (std::size_t cursor = 0; cursor < context_pixels.size();
       cursor += context_stride) {
    scene_pixels.push_back(context_pixels[cursor]);
  }
  std::vector<SceneFeaturePoint> scene;
  scene.reserve(scene_pixels.size());
  for (const std::size_t pixel_index : scene_pixels) {
    SceneFeaturePoint point; point.camera_point = camera_points[pixel_index];
    std::array<float, 8> nearest_distances;
    std::array<int, 8> nearest_indices;
    nearest_distances.fill(std::numeric_limits<float>::infinity());
    nearest_indices.fill(-1);
    for (int feature_index = 0; feature_index < static_cast<int>(feature_points.size());
         ++feature_index) {
      if (!feature_points[feature_index].allFinite()) continue;
      const float distance =
        (feature_points[feature_index] - point.camera_point).squaredNorm();
      if (distance >= nearest_distances.back()) continue;
      int insertion = 7;
      while (insertion > 0 && distance < nearest_distances[insertion - 1]) {
        nearest_distances[insertion] = nearest_distances[insertion - 1];
        nearest_indices[insertion] = nearest_indices[insertion - 1];
        --insertion;
      }
      nearest_distances[insertion] = distance;
      nearest_indices[insertion] = feature_index;
    }
    for (int channel = 0; channel < 32; ++channel) {
      float value = -std::numeric_limits<float>::infinity();
      for (const int feature_index : nearest_indices) {
        if (feature_index < 0) continue;
        const int fu = feature_index / kFeatureHeight;
        const int fv = feature_index % kFeatureHeight;
        value = std::max(value, perpoint_data[tensorIndex(
          channel, fu, fv, kFeatureWidth, kFeatureHeight)]);
      }
      point.features[channel] = std::isfinite(value) ? value : 0.0F;
    }
    scene.push_back(point);
  }
  std::vector<AnchorCenter> grouped_centers;
  std::vector<std::vector<std::size_t>> grouped_indices;
  for (const auto &center : centers) {
    std::vector<std::size_t> members;
    const float radius_sq = center.width_m * center.width_m;
    for (std::size_t i = 0; i < scene.size(); ++i) if ((scene[i].camera_point - center.camera_center).squaredNorm() < radius_sq) members.push_back(i);
    if (members.size() <= 32) continue;
    grouped_centers.push_back(center);
    grouped_indices.push_back(deterministicFarthestSamples(members, scene, center.camera_center));
  }
  diagnostics.valid_local_groups = grouped_centers.size();
  if (grouped_centers.empty()) return {};

  const std::size_t group_count = grouped_centers.size();
  std::vector<float> pointnet_input(group_count * 35 * kGroupPoints, 0.0F), grasp_info(group_count * 3, 0.0F);
  for (std::size_t group = 0; group < group_count; ++group) {
    const auto &center = grouped_centers[group];
    for (int sample = 0; sample < kGroupPoints; ++sample) {
      const auto &point = scene[grouped_indices[group][sample]];
      const Eigen::Vector3f relative = point.camera_point - center.camera_center;
      for (int axis = 0; axis < 3; ++axis) pointnet_input[(group * 35 + axis) * kGroupPoints + sample] = relative[axis];
      for (int channel = 0; channel < 32; ++channel) pointnet_input[(group * 35 + 3 + channel) * kGroupPoints + sample] = point.features[channel];
    }
    grasp_info[group * 3] = center.theta; grasp_info[group * 3 + 1] = center.width_pixels; grasp_info[group * 3 + 2] = center.depth_offset_mm;
  }
  const auto local_started = std::chrono::steady_clock::now();
  const std::array<int64_t, 3> pointnet_shape{static_cast<int64_t>(group_count), 35, kGroupPoints};
  auto pointnet_tensor = Ort::Value::CreateTensor<float>(memory, pointnet_input.data(), pointnet_input.size(), pointnet_shape.data(), pointnet_shape.size());
  const char *pointnet_input_names[] = {"pointcloud"}, *pointnet_output_names[] = {"pointnet_features"};
  auto pointnet_outputs = pointnet_->Run(Ort::RunOptions{nullptr}, pointnet_input_names, &pointnet_tensor, 1, pointnet_output_names, 1);
  const std::array<int64_t, 2> info_shape{static_cast<int64_t>(group_count), 3};
  auto info_tensor = Ort::Value::CreateTensor<float>(memory, grasp_info.data(), grasp_info.size(), info_shape.data(), info_shape.size());
  const char *local_input_names[] = {"pointnet_features", "grasp_info"}, *local_output_names[] = {"anchor_pred", "offset_pred"};
  std::array<Ort::Value, 2> local_inputs{std::move(pointnet_outputs[0]), std::move(info_tensor)};
  auto local_outputs = localnet_->Run(Ort::RunOptions{nullptr}, local_input_names, local_inputs.data(), 2, local_output_names, 2);
  diagnostics.local_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - local_started).count();
  const float *logits = local_outputs[0].GetTensorData<float>(), *offsets = local_outputs[1].GetTensorData<float>();
  std::vector<GraspProposal> proposals;
  for (std::size_t group = 0; group < group_count; ++group) {
    std::array<int, kOrientationCount> indices{}; std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
      return logits[group * kOrientationCount + a] >
        logits[group * kOrientationCount + b];
    });
    const auto &center = grouped_centers[group];
    const Eigen::Vector3f initial_approach = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f initial_closing(-std::cos(center.theta) * scale_u / intrinsics.fx, std::sin(center.theta) * scale_v / intrinsics.fy, 0.0F);
    if (!finiteVector(initial_closing)) continue;
    initial_closing.normalize();
    Eigen::Matrix3f image_rotation; image_rotation.col(0) = initial_approach; image_rotation.col(1) = initial_closing; image_rotation.col(2) = initial_approach.cross(initial_closing).normalized();
    int retained_orientations = 0;
    for (int rank = 0;
         rank < kOrientationCount && retained_orientations < 10; ++rank) {
      const int orientation = indices[rank];
      const float gamma = kGammaAnchors[orientation % 7] * kPi * 0.5F, beta = kBetaAnchors[orientation / 7] * kPi * 0.5F;
      if (std::abs(gamma) >= kPi / 3.0F) continue;
      Eigen::Vector3f offset;
      for (int axis = 0; axis < 3; ++axis) offset[axis] = 0.02F * std::clamp(offsets[(group * kOrientationCount + orientation) * 3 + axis], -1.0F, 1.0F);
      const Eigen::Vector3f camera_center = center.camera_center + offset;
      const Eigen::Matrix3f camera_rotation = image_rotation * Eigen::AngleAxisf(gamma, Eigen::Vector3f::UnitZ()).toRotationMatrix() * Eigen::AngleAxisf(beta, Eigen::Vector3f::UnitY()).toRotationMatrix();
      if (!camera_center.allFinite() || camera_center.z() <= 0.02F || !camera_rotation.allFinite() || camera_rotation.determinant() < 0.99F) continue;
      const Eigen::Vector3f base_center = camera_to_base * camera_center;
      const Eigen::Matrix3f base_rotation = camera_to_base.linear() * camera_rotation;
      GraspProposal proposal; proposal.sample = base_center; proposal.approach = base_rotation.col(0).normalized(); proposal.closing = base_rotation.col(1).normalized(); proposal.height_axis = base_rotation.col(2).normalized();
      proposal.hand_base = base_center - proposal.approach * contact_depth_m; proposal.opening_m = center.width_m; proposal.quality = sigmoid(logits[group * kOrientationCount + orientation]); proposal.source = GraspProposalSource::kHggd;
      // Keep the heatmap center and its LocalNet orientation hypotheses
      // identifiable downstream. This lets RViz show one fundamental pose per
      // spatial center without changing which hypotheses reach validation.
      proposal.family_index = group;
      proposal.variant_index = static_cast<std::size_t>(rank);
      if (proposal.sample.allFinite() && proposal.hand_base.allFinite() && finiteVector(proposal.approach) && finiteVector(proposal.closing) && finiteVector(proposal.height_axis) && std::isfinite(proposal.opening_m)) {
        proposals.push_back(proposal);
        ++retained_orientations;
      }
    }
  }
  diagnostics.network_grasps = proposals.size();
  std::stable_sort(proposals.begin(), proposals.end(), [](const auto &a, const auto &b) { return a.quality > b.quality; });
  if (proposals.size() > maximum_proposals) proposals.resize(maximum_proposals);
  return proposals;
}

}  // namespace vamp_node
