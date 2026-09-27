#include "vamp_node/local_grasp_refiner.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>
#include <limits>
#include <vector>

namespace
{
using vamp_node::CameraIntrinsics;
using vamp_node::LocalGraspRefiner;
using vamp_node::LocalGraspRefinerParameters;

std::vector<Eigen::Vector3f> makePatch(
  int width, int height, float extent_x, float extent_y,
  float depth = 0.5F, float curvature = 0.0F)
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<Eigen::Vector3f> cloud(
    static_cast<std::size_t>(width * height),
    Eigen::Vector3f(nan, nan, nan));
  for (int v = 2; v < height - 2; ++v) {
    const float y = extent_y *
      (2.0F * static_cast<float>(v - 2) /
      static_cast<float>(height - 5) - 1.0F);
    for (int u = 2; u < width - 2; ++u) {
      const float x = extent_x *
        (2.0F * static_cast<float>(u - 2) /
        static_cast<float>(width - 5) - 1.0F);
      cloud[static_cast<std::size_t>(v * width + u)] =
        Eigen::Vector3f(x, y, depth + curvature * x * x);
    }
  }
  return cloud;
}

LocalGraspRefinerParameters testParameters()
{
  LocalGraspRefinerParameters parameters;
  parameters.snap_radius_px = 8;
  parameters.snap_max_distance_m = 0.06F;
  parameters.neighbor_distance_m = 0.02F;
  parameters.minimum_region_points = 20;
  parameters.minimum_contact_points = 2;
  parameters.cross_section_half_length_m = 0.05F;
  return parameters;
}
}  // namespace

TEST(LocalGraspRefiner, ProjectsWithManualCameraMatrix)
{
  int u = -1;
  int v = -1;
  const CameraIntrinsics intrinsics{100.0F, 100.0F, 20.0F, 15.0F};
  EXPECT_TRUE(LocalGraspRefiner::projectBasePoint(
      Eigen::Vector3f(0.05F, -0.025F, 0.5F),
      Eigen::Isometry3f::Identity(), intrinsics, 64, 48, u, v));
  EXPECT_EQ(u, 30);
  EXPECT_EQ(v, 10);
}

TEST(LocalGraspRefiner, SnapsSeedAndFindsCurvedSurfaceCrossSection)
{
  constexpr int width = 45;
  constexpr int height = 45;
  const auto cloud = makePatch(width, height, 0.02F, 0.04F, 0.5F, 8.0F);
  LocalGraspRefiner refiner(testParameters());
  const auto result = refiner.refine(
    cloud, width, height, Eigen::Vector3f(0.0F, 0.0F, 0.52F),
    width / 2, height / 2, Eigen::Vector3f::Zero());

  ASSERT_TRUE(result.valid) << result.failure_reason;
  EXPECT_NEAR(result.seed.z(), 0.5F, 0.002F);
  EXPECT_GT(result.width_m, 0.03F);
  EXPECT_LT(result.width_m, 0.05F);
  EXPECT_GT(result.usable_depth_m, 0.06F);
}

TEST(LocalGraspRefiner, FallsBackToClosestGroundedTargetSurfaceInThreeDimensions)
{
  constexpr int width = 80;
  constexpr int height = 60;
  auto cloud = makePatch(width, height, 0.040F, 0.030F);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < 40; ++u) {
      cloud[static_cast<std::size_t>(v * width + u)] =
        Eigen::Vector3f(nan, nan, nan);
    }
  }
  const Eigen::Vector3f seed = cloud[static_cast<std::size_t>(30 * width + 48)];
  LocalGraspRefinerParameters parameters = testParameters();
  parameters.snap_radius_px = 3;
  parameters.snap_max_distance_m = 0.05F;
  LocalGraspRefiner refiner(parameters);

  // Deliberately project the old seed well outside the local pixel window.
  // The supplied cloud represents only the caller-filtered target support.
  const auto result = refiner.refine(
    cloud, width, height, seed,
    20, 30, Eigen::Vector3f::Zero());

  EXPECT_TRUE(result.used_global_target_search);
  EXPECT_GE(result.snapped_u, 40);
  EXPECT_NE(result.failure_reason, "NO_REOBSERVED_TARGET_DEPTH");
  EXPECT_NE(result.failure_reason, "NEAREST_VISIBLE_TARGET_TOO_FAR");
}

TEST(LocalGraspRefiner, CentersGraspBetweenAsymmetricSurfaceLimits)
{
  constexpr int width = 45;
  constexpr int height = 45;
  const auto cloud = makePatch(width, height, 0.02F, 0.04F);
  const int off_center_u = 12;
  const int center_v = height / 2;
  const Eigen::Vector3f seed =
    cloud[static_cast<std::size_t>(center_v * width + off_center_u)];
  LocalGraspRefiner refiner(testParameters());
  const auto result = refiner.refine(
    cloud, width, height, seed, off_center_u, center_v,
    Eigen::Vector3f::Zero());

  ASSERT_TRUE(result.valid) << result.failure_reason;
  EXPECT_NEAR(result.seed.x(), 0.0F, 0.003F);
  EXPECT_NEAR(result.seed.y(), 0.0F, 0.003F);
  EXPECT_LT(result.snap_distance_m, 1.0e-5F);
}

TEST(LocalGraspRefiner, RegionGrowingStopsAtTableDepthBoundary)
{
  constexpr int width = 45;
  constexpr int height = 45;
  auto cloud = makePatch(width, height, 0.04F, 0.04F);
  for (int v = 2; v < height - 2; ++v) {
    for (int u = width / 2 + 5; u < width - 2; ++u) {
      auto & point = cloud[static_cast<std::size_t>(v * width + u)];
      point.z() += 0.04F;
    }
  }
  auto parameters = testParameters();
  parameters.neighbor_depth_change_m = 0.015F;
  LocalGraspRefiner refiner(parameters);
  const auto result = refiner.refine(
    cloud, width, height, Eigen::Vector3f(-0.02F, 0.0F, 0.5F),
    width / 2 - 10, height / 2, Eigen::Vector3f::Zero());

  EXPECT_GT(result.region_points, 500U);
  EXPECT_LT(result.region_points, 1300U);
}

TEST(LocalGraspRefiner, RejectsCrossSectionWiderThanGripper)
{
  constexpr int width = 45;
  constexpr int height = 45;
  const auto cloud = makePatch(width, height, 0.065F, 0.065F);
  LocalGraspRefiner refiner(testParameters());
  const auto result = refiner.refine(
    cloud, width, height, Eigen::Vector3f(0.0F, 0.0F, 0.5F),
    width / 2, height / 2, Eigen::Vector3f::Zero());

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.failure_reason, "NO_VALID_CROSS_SECTION");
}

TEST(LocalGraspRefiner, RejectsInsufficientInsertionDepth)
{
  constexpr int width = 45;
  constexpr int height = 45;
  auto cloud = makePatch(width, height, 0.0045F, 0.0045F);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (auto & point : cloud) {
    if (point.allFinite() && point.head<2>().norm() > 0.0045F) {
      point = Eigen::Vector3f(nan, nan, nan);
    }
  }
  auto parameters = testParameters();
  parameters.minimum_width_m = 0.004F;
  LocalGraspRefiner refiner(parameters);
  const auto result = refiner.refine(
    cloud, width, height, Eigen::Vector3f(0.0F, 0.0F, 0.5F),
    width / 2, height / 2, Eigen::Vector3f::Zero());

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.failure_reason, "NO_VALID_CROSS_SECTION");
}

TEST(LocalGraspRefiner, RejectsExtremelyThinNoisyEdge)
{
  constexpr int width = 45;
  constexpr int height = 45;
  const auto cloud = makePatch(width, height, 0.002F, 0.04F);
  LocalGraspRefiner refiner(testParameters());
  const auto result = refiner.refine(
    cloud, width, height, Eigen::Vector3f(0.0F, 0.0F, 0.5F),
    width / 2, height / 2, Eigen::Vector3f::Zero());

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.failure_reason, "CROSS_SECTION_TOO_THIN");
}

TEST(LocalGraspRefiner, ExtractsReusableFiveCentimeterContactRegion)
{
  const std::vector<Eigen::Vector3f> points{
    {0.0F, 0.0F, 0.5F},
    {0.029F, 0.04F, 0.5F},
    {0.0501F, 0.0F, 0.5F},
    {0.0F, 0.0F, 0.58F}};

  const auto region = LocalGraspRefiner::extractContactRegion(
    points, Eigen::Vector3f(0.0F, 0.0F, 0.5F), 0.05F);

  ASSERT_EQ(region.size(), 2U);
  EXPECT_TRUE(region[0].isApprox(points[0]));
  EXPECT_TRUE(region[1].isApprox(points[1]));
}

TEST(LocalGraspRefiner, MergesViewsWithoutDuplicatePointInflation)
{
  const std::vector<Eigen::Vector3f> first{
    {-0.01F, 0.0F, 0.5F}, {0.0F, 0.0F, 0.5F}};
  const std::vector<Eigen::Vector3f> second{
    {0.0002F, 0.0002F, 0.5002F}, {0.01F, 0.0F, 0.5F}};

  const auto merged = LocalGraspRefiner::mergeContactRegions(
    first, second, 0.001F);

  EXPECT_EQ(merged.size(), 3U);
}

TEST(LocalGraspRefiner, FindsMinimumOpeningAcrossMergedContactViews)
{
  constexpr int width = 45;
  constexpr int height = 45;
  const auto organized = makePatch(width, height, 0.015F, 0.035F);
  std::vector<Eigen::Vector3f> first_view;
  std::vector<Eigen::Vector3f> second_view;
  for (const auto & point : organized) {
    if (!point.allFinite()) continue;
    if (point.x() <= 0.002F) first_view.push_back(point);
    if (point.x() >= -0.002F) second_view.push_back(point);
  }
  const auto merged = LocalGraspRefiner::mergeContactRegions(
    first_view, second_view, 0.0005F);
  auto parameters = testParameters();
  parameters.angle_step_degrees = 5.0F;
  LocalGraspRefiner refiner(parameters);

  const auto result = refiner.refinePointRegion(
    merged, Eigen::Vector3f(0.0F, 0.0F, 0.5F),
    Eigen::Vector3f::Zero());

  ASSERT_TRUE(result.valid) << result.failure_reason;
  EXPECT_GT(result.width_m, 0.025F);
  EXPECT_LT(result.width_m, 0.035F);
  EXPECT_GT(result.usable_depth_m, 0.05F);
}
