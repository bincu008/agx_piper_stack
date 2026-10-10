#include <array>
#include <random>
#include <gtest/gtest.h>
#include "vamp_node/placement_point_index.hpp"
#include <vamp/collision/sphere_sphere.hh>

TEST(CollisionPointIndex, MatchesExactScanIncludingContactExemptions)
{
  std::mt19937 random(1234);
  std::uniform_real_distribution<float> coordinate(-0.6F, 0.6F);
  std::vector<std::array<float, 3>> points;
  for (int index = 0; index < 3000; ++index) {
    points.push_back({coordinate(random), coordinate(random), coordinate(random)});
  }
  const vamp_node::PlacementPointIndex index(points);
  for (int trial = 0; trial < 200; ++trial) {
    const Eigen::Vector3f center(coordinate(random), coordinate(random), coordinate(random));
    const float radius = 0.05F;
    const auto blocked = [&](size_t id, const Eigen::Vector3f &point) {
      // A classification exemption must refer to the original point index.
      return id % 3 != 0 && vamp::collision::sphere_sphere_sql2(
        center.x(), center.y(), center.z(), radius-0.0025F,
        point.x(), point.y(), point.z(), 0.0025F) < 0;
    };
    bool reference = true;
    for (size_t id = 0; id < points.size(); ++id) {
      if (blocked(id, Eigen::Vector3f(points[id][0], points[id][1], points[id][2]))) {
        reference = false;
        break;
      }
    }
    EXPECT_EQ(reference, index.sphereClearWithPredicate(center, radius, blocked));
  }
}

TEST(CollisionPointIndex, DoesNotSkipCellBoundariesOrNonExemptContacts)
{
  const std::vector<std::array<float, 3>> points{{0.02F, 0.0F, 0.0F}, {-0.02F, 0.0F, 0.0F}};
  const vamp_node::PlacementPointIndex index(points);
  EXPECT_FALSE(index.sphereClearWithPredicate(Eigen::Vector3f::Zero(), 0.020001F,
    [](size_t, const auto &) { return true; }));
  EXPECT_TRUE(index.sphereClearWithPredicate(Eigen::Vector3f::Zero(), 0.020001F,
    [](size_t, const auto &) { return false; }));
  const std::vector<Eigen::Vector3f> empty;
  EXPECT_TRUE(vamp_node::PlacementPointIndex(empty).sphereClear(Eigen::Vector3f::Zero(), 0.02F));
}

TEST(CollisionPointIndex, InPlaceSceneChangeCannotReuseStaleGeometry)
{
  vamp_node::ExactPointIndexCache cache;
  std::vector<std::array<float, 3>> points{{0.0F, 0.0F, 0.0F}};
  EXPECT_FALSE(cache.get(points).sphereClear(Eigen::Vector3f::Zero(), 0.01F));
  const auto *original = &cache.get(points);
  EXPECT_EQ(original, &cache.get(points));
  points[0][0] = 0.5F;
  EXPECT_TRUE(cache.get(points).sphereClear(Eigen::Vector3f::Zero(), 0.01F));
  points[0][0] = 0.0F;
  EXPECT_FALSE(cache.get(points).sphereClear(Eigen::Vector3f::Zero(), 0.01F));
}
