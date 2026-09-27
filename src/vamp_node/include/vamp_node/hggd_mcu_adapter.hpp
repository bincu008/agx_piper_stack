#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "vamp_node/grasp_proposal.hpp"

namespace Ort {
class Env;
class Session;
class SessionOptions;
}  // namespace Ort

namespace vamp_node {

struct HggdCameraIntrinsics {
  float fx = 0.0F;
  float fy = 0.0F;
  float cx = 0.0F;
  float cy = 0.0F;
};

struct HggdMcuDiagnostics {
  std::size_t target_pixels = 0;
  std::size_t heatmap_centers = 0;
  std::size_t valid_local_groups = 0;
  std::size_t network_grasps = 0;
  double backbone_ms = 0.0;
  double local_ms = 0.0;
};

// CPU ONNX adapter for the split HGGD-MCU inference graph. It intentionally
// returns the repository's backend-neutral GraspProposal type so IK,
// collision checking, ranking, and RViz stay in VampPlannerServer.
class HggdMcuAdapter {
public:
  HggdMcuAdapter(const std::string &model_directory, int intra_op_threads);
  ~HggdMcuAdapter();

  HggdMcuAdapter(const HggdMcuAdapter &) = delete;
  HggdMcuAdapter &operator=(const HggdMcuAdapter &) = delete;

  std::vector<GraspProposal> detect(
    const sensor_msgs::msg::PointCloud2 &organized_cloud,
    const sensor_msgs::msg::Image &rgb_image,
    const sensor_msgs::msg::Image &target_mask,
    const HggdCameraIntrinsics &intrinsics,
    const Eigen::Isometry3f &cloud_to_camera,
    const Eigen::Isometry3f &camera_to_base,
    float contact_depth_m,
    std::size_t maximum_proposals,
    cv::Mat &heatmap_overlay,
    HggdMcuDiagnostics &diagnostics);

private:
  std::unique_ptr<Ort::Env> environment_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::unique_ptr<Ort::Session> resnet_;
  std::unique_ptr<Ort::Session> anchornet_;
  std::unique_ptr<Ort::Session> pointnet_;
  std::unique_ptr<Ort::Session> localnet_;
  std::mutex inference_mutex_;
};

}  // namespace vamp_node
