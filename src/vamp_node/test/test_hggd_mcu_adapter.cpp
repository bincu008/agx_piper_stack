#include <gtest/gtest.h>

#include <cmath>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "vamp_node/hggd_mcu_adapter.hpp"

namespace {

TEST(HggdMcuAdapter, RunsFourStageCpuPipelineAndCapsOutput) {
  constexpr int width = 320;
  constexpr int height = 160;
  constexpr float fx = 250.0F;
  constexpr float fy = 250.0F;
  constexpr float cx = width * 0.5F;
  constexpr float cy = height * 0.5F;

  sensor_msgs::msg::Image image;
  image.width = width;
  image.height = height;
  image.encoding = "bgr8";
  image.step = width * 3;
  image.data.resize(static_cast<std::size_t>(image.step) * height);
  for (int v = 0; v < height; ++v) for (int u = 0; u < width; ++u) {
    const std::size_t index = static_cast<std::size_t>(v) * image.step + u * 3;
    image.data[index] = static_cast<unsigned char>((u + v) % 255);
    image.data[index + 1] = static_cast<unsigned char>((2 * v) % 255);
    image.data[index + 2] = static_cast<unsigned char>((2 * u) % 255);
  }

  sensor_msgs::msg::Image mask;
  mask.width = width;
  mask.height = height;
  mask.encoding = "mono8";
  mask.step = width;
  mask.data.assign(static_cast<std::size_t>(width) * height, 255);

  sensor_msgs::msg::PointCloud2 cloud;
  cloud.width = width;
  cloud.height = height;
  cloud.header.frame_id = "camera_color_optical_frame";
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(static_cast<std::size_t>(width) * height);
  sensor_msgs::PointCloud2Iterator<float> x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> z(cloud, "z");
  for (int v = 0; v < height; ++v) for (int u = 0; u < width; ++u, ++x, ++y, ++z) {
    const float depth = 0.55F + 0.03F * std::sin(u * 0.04F) * std::cos(v * 0.05F);
    *x = (u - cx) * depth / fx;
    *y = (v - cy) * depth / fy;
    *z = depth;
  }

  const std::string model_directory =
    ament_index_cpp::get_package_share_directory("vamp_node") + "/models/hggd_mcu";
  vamp_node::HggdMcuAdapter adapter(model_directory, 2);
  vamp_node::HggdCameraIntrinsics intrinsics{fx, fy, cx, cy};
  vamp_node::HggdMcuDiagnostics diagnostics;
  cv::Mat heatmap;
  const auto proposals = adapter.detect(
    cloud, image, mask, intrinsics, Eigen::Isometry3f::Identity(),
    Eigen::Isometry3f::Identity(), 0.06F, 100, heatmap, diagnostics);

  EXPECT_FALSE(heatmap.empty());
  EXPECT_EQ(heatmap.cols, width);
  EXPECT_EQ(heatmap.rows, height);
  EXPECT_GT(diagnostics.heatmap_centers, 0U);
  EXPECT_GT(diagnostics.valid_local_groups, 0U);
  EXPECT_GT(diagnostics.network_grasps, 0U);
  EXPECT_FALSE(proposals.empty());
  EXPECT_LE(proposals.size(), 100U);
  for (const auto &proposal : proposals) {
    EXPECT_LT(proposal.family_index, diagnostics.valid_local_groups);
    EXPECT_LT(proposal.variant_index, 10U);
  }
  for (std::size_t index = 1; index < proposals.size(); ++index) {
    EXPECT_GE(proposals[index - 1].quality, proposals[index].quality);
  }
}

}  // namespace
