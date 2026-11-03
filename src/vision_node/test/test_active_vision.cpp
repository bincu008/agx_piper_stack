#include <cmath>
#include <gtest/gtest.h>
#include "vision_node/active_vision_utils.hpp"

TEST(MetricScaleTest, PreservesWhenRatioIsOne) {
  sensor_msgs::msg::PointCloud2 input;
  auto output = vision_node::correct_camera_measurements(input, 1.0);
  EXPECT_TRUE(output.data.empty());
}

TEST(MetricScaleTest, RejectsInvalidRatios) {
  sensor_msgs::msg::PointCloud2 input;
  EXPECT_THROW(vision_node::correct_camera_measurements(input, 0.0), std::invalid_argument);
  EXPECT_THROW(vision_node::correct_camera_measurements(input, -1.0), std::invalid_argument);
  EXPECT_THROW(vision_node::correct_camera_measurements(input, NAN), std::invalid_argument);
}

TEST(NbvPreferenceTest, BaseFacingDirectionPointsTowardBaseAndSlightlyDown) {
  const Eigen::Vector3f target(.4F, -.1F, 0.F), current(.5F, -.1F, .05F);
  const auto direction = vision_node::baseFacingViewDirection(target);
  EXPECT_GT(direction.z(), .96F);
  EXPECT_LT(direction.head<2>().dot(target.head<2>()), 0.F);
  const Eigen::Vector3f good = target + .45F * direction;
  const Eigen::Vector3f side = target + Eigen::Vector3f(.45F, 0.F, 0.F);
  EXPECT_LT(vision_node::nextViewCost(good, target, current),
            vision_node::nextViewCost(side, target, current));
  const Eigen::Vector3f opposite = target + Eigen::Vector3f(-.45F * direction.x(), -.45F * direction.y(), .45F * direction.z());
  EXPECT_LT(vision_node::nextViewCost(good, target, current),
            vision_node::nextViewCost(opposite, target, current));
  const auto samples = vision_node::nextViewSamples(target, .45F);
  EXPECT_EQ(samples.size(), 156u);
  for (const auto &p : samples) EXPECT_NEAR((p - target).norm(), .45F, 1.e-5F);
  EXPECT_TRUE(vision_node::nextViewSamples(target, 0.F).empty());
  EXPECT_TRUE(vision_node::baseFacingViewDirection(Eigen::Vector3f::Zero()).allFinite());
}

TEST(PerceptionTranslationTest, AddsOffsetInBaseAxes) {
  Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
  tf(0, 3) = 0.1F; tf(1, 3) = 0.2F; tf(2, 3) = 0.3F;
  Eigen::Vector3f offset(0.01F, -0.02F, 0.03F);
  auto res = vision_node::translatedPerceptionTransform(tf, offset);
  EXPECT_NEAR(res(0, 3), 0.11F, 1e-5F);
  EXPECT_NEAR(res(1, 3), 0.18F, 1e-5F);
  EXPECT_NEAR(res(2, 3), 0.33F, 1e-5F);
}

TEST(TargetVisibilityTest, RayClearThroughEmptySpace) {
  octomap::OcTree tree(0.01);
  EXPECT_TRUE(vision_node::viewRayClear(tree, {0, 0, 0}, {1, 0, 0}, 0.05F));
}

TEST(TargetVisibilityTest, RayBlockedByInterveningObstacle) {
  octomap::OcTree tree(0.01);
  tree.updateNode(octomap::point3d(0.5, 0.0, 0.0), true);
  tree.updateInnerOccupancy();
  EXPECT_FALSE(vision_node::viewRayClear(tree, {0, 0, 0}, {1, 0, 0}, 0.05F));
}

TEST(VoxelNeighborsTest, FiltersIsolatedVoxels) {
  std::vector<vision_node::VoxelCell> cells = {{0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {10, 10, 10}};
  auto keep = vision_node::supportedVoxels(cells);
  EXPECT_TRUE(keep[1]);
  EXPECT_FALSE(keep[3]);
}

