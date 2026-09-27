#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "vamp_node/cartesian_grasp_approach.hpp"

namespace {

Eigen::Isometry3f poseAt(float x, float y = 0.0F, float z = 0.0F)
{
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.translation() = Eigen::Vector3f(x, y, z);
  return pose;
}

std::vector<float> startJoints()
{
  return {0.0F, 0.8F, -1.4F, 0.0F, 0.0F, 0.0F};
}

TEST(CartesianGraspApproach, InterpolatesAtMostTwoMillimetresWithFixedOrientation)
{
  const auto start = poseAt(0.0F);
  auto goal = poseAt(0.005F, 0.003F, 0.0F);
  goal.linear() = Eigen::AngleAxisf(0.4F, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  std::vector<std::vector<float>> seeds;
  const auto plan = vamp_node::planCartesianApproach(
    start, goal, startJoints(), 0.010F, 0.2F,
    [&seeds](const Eigen::Isometry3f &, const std::vector<float> &seed,
             std::vector<float> &solution) {
      seeds.push_back(seed);
      solution = seed;
      solution[0] += 0.01F;
      return true;
    },
    [](const std::vector<float> &, std::size_t, bool) {return true;});

  ASSERT_TRUE(plan.success) << plan.failure_reason;
  ASSERT_EQ(plan.waypoint_poses.size(), 3U);
  ASSERT_EQ(plan.waypoint_joints.size(), 3U);
  ASSERT_EQ(seeds.size(), 3U);
  EXPECT_EQ(seeds[1], plan.waypoint_joints[0]);
  EXPECT_EQ(seeds[2], plan.waypoint_joints[1]);
  EXPECT_NEAR(plan.waypoint_poses.back().translation().x(), 0.005F, 1.0e-6F);
  EXPECT_NEAR(plan.waypoint_poses.back().translation().y(), 0.003F, 1.0e-6F);
  for (std::size_t index = 0; index < plan.waypoint_poses.size(); ++index) {
    EXPECT_TRUE(plan.waypoint_poses[index].linear().isApprox(goal.linear()));
    const auto &previous = index == 0 ? start : plan.waypoint_poses[index - 1];
    EXPECT_LE(
      (plan.waypoint_poses[index].translation() - previous.translation()).norm(),
      0.002F + 1.0e-6F);
  }
}

TEST(CartesianGraspApproach, ReportsSequentialIkFailureWithoutPartialPath)
{
  int calls = 0;
  const auto plan = vamp_node::planCartesianApproach(
    poseAt(0.0F), poseAt(0.004F), startJoints(), 0.002F, 0.2F,
    [&calls](const Eigen::Isometry3f &, const std::vector<float> &seed,
             std::vector<float> &solution) {
      ++calls;
      if (calls == 2) return false;
      solution = seed;
      solution[0] += 0.01F;
      return true;
    },
    [](const std::vector<float> &, std::size_t, bool) {return true;});

  EXPECT_FALSE(plan.success);
  EXPECT_EQ(plan.failure_reason, "IK_FAILED_AT_WAYPOINT_2");
  EXPECT_TRUE(plan.waypoint_joints.empty());
}

TEST(CartesianGraspApproach, RejectsJointDiscontinuity)
{
  const auto plan = vamp_node::planCartesianApproach(
    poseAt(0.0F), poseAt(0.002F), startJoints(), 0.002F, 0.2F,
    [](const Eigen::Isometry3f &, const std::vector<float> &seed,
       std::vector<float> &solution) {
      solution = seed;
      solution[0] += 0.3F;
      return true;
    },
    [](const std::vector<float> &, std::size_t, bool) {return true;});

  EXPECT_FALSE(plan.success);
  EXPECT_EQ(plan.failure_reason, "JOINT_DISCONTINUITY_AT_WAYPOINT_1");
  EXPECT_TRUE(plan.waypoint_poses.empty());
}

TEST(CartesianGraspApproach, RejectsCollisionAndOnlyMarksFinalWaypoint)
{
  std::vector<bool> final_flags;
  const auto plan = vamp_node::planCartesianApproach(
    poseAt(0.0F), poseAt(0.004F), startJoints(), 0.002F, 0.2F,
    [](const Eigen::Isometry3f &, const std::vector<float> &seed,
       std::vector<float> &solution) {
      solution = seed;
      return true;
    },
    [&final_flags](const std::vector<float> &, std::size_t index, bool final) {
      final_flags.push_back(final);
      return index != 1;
    });

  EXPECT_FALSE(plan.success);
  EXPECT_EQ(plan.failure_reason, "COLLISION_AT_WAYPOINT_2");
  ASSERT_EQ(final_flags.size(), 2U);
  EXPECT_FALSE(final_flags[0]);
  EXPECT_TRUE(final_flags[1]);
  EXPECT_TRUE(plan.waypoint_joints.empty());
}

}  // namespace
