#include <gtest/gtest.h>
#include "vamp_node/guided_joint_lift.hpp"

namespace {
Eigen::Isometry3f fk(const std::vector<float> &q) {
  auto pose=Eigen::Isometry3f::Identity();
  pose.translation()=Eigen::Vector3f(.1F*q[1],0.F,q[0]);
  pose.linear()=Eigen::AngleAxisf(q[2]+q[3],Eigen::Vector3f::UnitY()).toRotationMatrix();
  return pose;
}
const std::vector<float> start{.1F,.1F,-.1F,.1F,0.F,0.F};
}

TEST(GuidedJointLift, RisesWithLateralGuidanceAndDenseValidEdges) {
  size_t checks=0;
  const auto plan=vamp_node::planGuidedJointLift(start,.2F,.087F,.1F,fk,
    [](const auto &) {return Eigen::Vector3f::UnitX().eval();},
    [&](const auto &a,const auto &b) {
      ++checks;
      EXPECT_LE(vamp_node::maximumJointDelta(a,b),.005F);
      EXPECT_LE((fk(a).translation()-fk(b).translation()).norm(),.002F);
      EXPECT_TRUE(vamp_node::jointsWithinPiperLimits(b));
      EXPECT_TRUE(vamp_node::objectRemainsUpright(fk(b),fk(start),.087F));
      return true;
    });
  ASSERT_TRUE(plan.success) << plan.failure_reason;
  EXPECT_GT(checks,1U);
  EXPECT_GE(fk(plan.waypoint_joints.back()).translation().z(),.3F);
  EXPECT_GT(fk(plan.waypoint_joints.back()).translation().x(),fk(start).translation().x());
}

TEST(GuidedJointLift, FailedEdgeNeverReturnsExecutablePrefix) {
  const auto plan=vamp_node::planGuidedJointLift(start,.2F,.087F,.1F,fk,
    [](const auto &) {return Eigen::Vector3f::Zero().eval();},
    [](const auto &,const auto &b) {return fk(b).translation().z()<.15F;});
  EXPECT_FALSE(plan.success);
  EXPECT_TRUE(plan.waypoint_joints.empty());
  EXPECT_TRUE(plan.waypoint_poses.empty());
}

TEST(GuidedJointLift, InvalidHeightOrGuidanceIsRejected) {
  const auto safe=[](const auto &,const auto &) {return true;};
  auto plan=vamp_node::planGuidedJointLift(start,-.1F,.087F,.1F,fk,
    [](const auto &) {return Eigen::Vector3f::Zero().eval();},safe);
  EXPECT_FALSE(plan.success);
  plan=vamp_node::planGuidedJointLift(start,.1F,.087F,.1F,fk,
    [](const auto &) {return Eigen::Vector3f::Constant(NAN).eval();},safe);
  EXPECT_FALSE(plan.success);
  EXPECT_TRUE(plan.waypoint_joints.empty());
}
