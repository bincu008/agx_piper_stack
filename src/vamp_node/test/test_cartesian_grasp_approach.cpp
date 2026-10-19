#include <gtest/gtest.h>

#include <cmath>
#include <vector>
#include <limits>

#include "vamp_node/cartesian_grasp_approach.hpp"
#include "vamp_node/grasp_position_offset.hpp"

namespace {

TEST(GraspPositionOffset, BaseTranslationDoesNotRotateWithTheTool)
{
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.linear() = Eigen::AngleAxisf(1.57F, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  pose.translation() = Eigen::Vector3f(.3F, .2F, .1F);
  const auto offset = vamp_node::validatedGraspPositionOffset({-.005, .005, .005});
  const auto corrected = vamp_node::correctedGraspTarget(pose, offset);
  EXPECT_TRUE(corrected.translation().isApprox(Eigen::Vector3f(.295F, .205F, .105F)));
  EXPECT_TRUE(corrected.rotation().isApprox(pose.rotation()));
  EXPECT_TRUE(pose.translation().isApprox(Eigen::Vector3f(.3F, .2F, .1F)));
  EXPECT_TRUE(vamp_node::correctedGraspTarget(pose, offset, false).matrix().isApprox(pose.matrix()));
  EXPECT_THROW(vamp_node::validatedGraspPositionOffset({1., 2.}), std::invalid_argument);
  EXPECT_THROW(vamp_node::validatedGraspPositionOffset(
    {0., std::numeric_limits<double>::infinity(), 0.}), std::invalid_argument);
}

TEST(GraspSearchBudget, EmptyResultContinuesButSuccessfulFastPathStops)
{
  EXPECT_FALSE(vamp_node::graspSearchBudgetExhausted(100, 100, 600, false));
  EXPECT_FALSE(vamp_node::graspSearchBudgetExhausted(599, 100, 600, false));
  EXPECT_TRUE(vamp_node::graspSearchBudgetExhausted(600, 100, 600, false));
  EXPECT_TRUE(vamp_node::graspSearchBudgetExhausted(100, 100, 600, true));
  EXPECT_FALSE(vamp_node::graspSearchBudgetExhausted(99, 100, 600, true));
}

TEST(LiftCollisionPolicy, RestoresChecksForTransfer)
{
  EXPECT_FALSE(vamp_node::checkLiftRobotEnvironment(true, true));
  EXPECT_TRUE(vamp_node::checkLiftRobotEnvironment(false, true));
  EXPECT_TRUE(vamp_node::checkLiftRobotEnvironment(true, false));
  EXPECT_TRUE(vamp_node::checkLiftRobotEnvironment(false, false));
}

TEST(JointApproach, BoundsSamplesAndRejectsCollisionWithoutPartialPath)
{
  const std::vector<float> start{0.0F, 0.8F, -1.4F, 0.0F, 0.0F, 0.0F};
  auto goal = start; goal[0] = 0.2F;
  const auto fk = [](const auto &q) {
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.translation().x() = q[0]; return pose;
  };
  const auto plan = vamp_node::planJointApproach(start, goal, fk,
    [](const auto &, std::size_t, bool) { return true; });
  ASSERT_TRUE(plan.success);
  auto previous = start;
  for (const auto &q : plan.waypoint_joints) {
    EXPECT_LE(vamp_node::maximumJointDelta(previous, q), 0.00501F);
    EXPECT_LE((fk(q).translation()-fk(previous).translation()).norm(), 0.00201F);
    previous = q;
  }
  EXPECT_EQ(previous, goal);
  const auto blocked = vamp_node::planJointApproach(start, goal, fk,
    [](const auto &, std::size_t i, bool) { return i < 2; });
  EXPECT_FALSE(blocked.success);
  EXPECT_TRUE(blocked.waypoint_joints.empty());
}

TEST(StandoffAlignment, RotatesGraduallyAtFixedPositionUsingPreviousSolution)
{
  Eigen::Isometry3f start = Eigen::Isometry3f::Identity();
  start.translation() = Eigen::Vector3f(0.2F, 0.1F, 0.3F);
  const Eigen::Matrix3f goal = Eigen::AngleAxisf(0.15F, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  const std::vector<float> seed{0.0F, 0.8F, -1.4F, 0.0F, 0.0F, 0.0F};
  std::size_t calls = 0;
  const auto plan = vamp_node::planStandoffAlignment(start, goal, seed, 0.08F,
    [&](const Eigen::Isometry3f &pose, const std::vector<float> &previous,
        std::vector<float> &solution) {
      EXPECT_LT((pose.translation()-start.translation()).norm(), 1.0e-7F);
      EXPECT_NEAR(previous[0], static_cast<float>(calls)*0.001F, 1.0e-6F);
      solution = previous;
      solution[0] += 0.001F;
      ++calls;
      return true;
    }, [](const auto &, std::size_t, bool) { return true; });
  ASSERT_TRUE(plan.success);
  ASSERT_GT(calls, 1U);
  Eigen::Matrix3f previous = start.linear();
  for (const auto &pose : plan.waypoint_poses) {
    EXPECT_LE(Eigen::AngleAxisf(previous.transpose()*pose.linear()).angle(), 0.017454F);
    previous = pose.linear();
  }
  EXPECT_LT((previous-goal).norm(), 1.0e-6F);
}

TEST(StandoffAlignment, CollisionDropsWholePrefixAndExcessRotationIsRejected)
{
  const auto start = Eigen::Isometry3f::Identity();
  const std::vector<float> seed{0.0F, 0.8F, -1.4F, 0.0F, 0.0F, 0.0F};
  const auto ik = [](const auto &, const auto &previous, auto &solution) {
    solution = previous; return true;
  };
  const auto blocked = vamp_node::planStandoffAlignment(start,
    Eigen::AngleAxisf(0.15F, Eigen::Vector3f::UnitZ()).toRotationMatrix(), seed, 0.08F,
    ik, [](const auto &, std::size_t index, bool) { return index == 0; });
  EXPECT_FALSE(blocked.success);
  EXPECT_EQ(blocked.failure_reason, "STANDOFF_ALIGNMENT_COLLISION_AT_WAYPOINT_2");
  EXPECT_TRUE(blocked.waypoint_joints.empty());
  const auto excessive = vamp_node::planStandoffAlignment(start,
    Eigen::AngleAxisf(0.5F, Eigen::Vector3f::UnitZ()).toRotationMatrix(), seed, 0.08F,
    ik, [](const auto &, std::size_t, bool) { return true; });
  EXPECT_EQ(excessive.failure_reason, "STANDOFF_ALIGNMENT_EXCEEDS_20_DEGREES");
}

TEST(CartesianApproachCollisionPolicy, BypassKeepsArmAndPalmChecks)
{
  for (std::size_t sphere = 0; sphere < 96; ++sphere) {
    EXPECT_TRUE(vamp_node::finalApproachChecksEnvironmentSphere(sphere, true));
    EXPECT_EQ(vamp_node::finalApproachChecksEnvironmentSphere(sphere, false), sphere < 82);
  }
}

TEST(CartesianApproachLimits, TwoDegreeMarginAppliesToEveryJointAndResets)
{
  const float margin = 2.0F * static_cast<float>(M_PI) / 180.0F;
  vamp_node::configurePiperJointLimitTolerance(margin);
  for (std::size_t joint = 0; joint < 6; ++joint) {
    auto joints = std::vector<float>{0.0F, 0.8F, -1.4F, 0.0F, 0.0F, 0.0F};
    joints[joint] = vamp_node::kPiperNominalJointLower[joint]-margin+0.0001F;
    EXPECT_TRUE(vamp_node::jointsWithinPiperLimits(joints));
    joints[joint] -= 0.001F;
    EXPECT_FALSE(vamp_node::jointsWithinPiperLimits(joints));
    joints[joint] = vamp_node::kPiperNominalJointUpper[joint]+margin-0.0001F;
    EXPECT_TRUE(vamp_node::jointsWithinPiperLimits(joints));
    joints[joint] += 0.001F;
    EXPECT_FALSE(vamp_node::jointsWithinPiperLimits(joints));
  }
  vamp_node::configurePiperJointLimitTolerance(0.0F);
  EXPECT_EQ(vamp_node::kPiperJointLower, vamp_node::kPiperNominalJointLower);
}

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

TEST(InitialGraspCollision, OnlyInitialGraspOmitsPalmAndFingers)
{
  for (std::size_t sphere = 0; sphere < 96; ++sphere) {
    EXPECT_EQ(vamp_node::initialGraspChecksEnvironmentSphere(sphere, true), sphere <= 67);
    EXPECT_TRUE(vamp_node::initialGraspChecksEnvironmentSphere(sphere, false));
    EXPECT_TRUE(vamp_node::finalApproachChecksEnvironmentSphere(sphere, true));
  }
}

TEST(OptionalJointApproach, RetainsPrefixAndFixedEndpointWhenLaterIntermediateFails)
{
  const std::vector<float> start{0., .8F, -1.4F, 0., 0., 0.};
  auto goal = start; goal[0] = .04F;
  auto near = start; near[0] = .02F;
  auto bad = start; bad[1] = -10.F;
  const auto fk = [](const auto &q) {
    Eigen::Isometry3f p = Eigen::Isometry3f::Identity();
    p.translation().x() = q[0]; return p;
  };
  const auto clear = [](const auto &, std::size_t, bool) {return true;};
  const auto plan = vamp_node::planOptionalJointApproach(start, goal, {near, bad}, fk, clear);
  ASSERT_TRUE(plan.success);
  EXPECT_EQ(plan.failure_reason, "OK;NEAR_GRASP_USED");
  EXPECT_EQ(plan.waypoint_joints.back(), goal);
  EXPECT_NE(std::find(plan.waypoint_joints.begin(), plan.waypoint_joints.end(), near),
    plan.waypoint_joints.end());
  const auto direct = vamp_node::planOptionalJointApproach(start, goal, {bad}, fk, clear);
  ASSERT_TRUE(direct.success);
  EXPECT_EQ(direct.failure_reason, "OK;NEAR_GRASP_SKIPPED");
  EXPECT_EQ(direct.waypoint_joints.back(), goal);
  const auto blocked = vamp_node::planOptionalJointApproach(start, goal, {}, fk,
    [](const auto &, std::size_t, bool) {return false;});
  EXPECT_FALSE(blocked.success);
  EXPECT_TRUE(blocked.waypoint_joints.empty());
}

TEST(ApproachPalmSphere, SkipsOnly80WhenExplicitlyScoped)
{
  for (std::size_t sphere = 0; sphere < 96; ++sphere) {
    EXPECT_EQ(vamp_node::approachPalmSphereEnabled(sphere, true), sphere != 80);
    EXPECT_TRUE(vamp_node::approachPalmSphereEnabled(sphere, false));
  }
}

TEST(FeedbackPlanningEntry, PreservesFeedbackAndNominalTargets)
{
  std::vector<float> measured{0.F, -.010367F, -.656F, -.119F, 1.23585F, .344F};
  const auto original = measured;
  const auto entry = vamp_node::nominalFeedbackEntry(measured, .034906585F);
  ASSERT_TRUE(entry.has_value());
  EXPECT_EQ(measured, original);
  EXPECT_FLOAT_EQ((*entry)[1], vamp_node::kPiperJointLower[1]);
  EXPECT_FLOAT_EQ((*entry)[4], vamp_node::kPiperJointUpper[4]);
  EXPECT_FALSE(vamp_node::nominalFeedbackEntry(measured, 0.F));
  measured[1] = -.04F;
  EXPECT_FALSE(vamp_node::nominalFeedbackEntry(measured, .034906585F));
  measured[1] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(vamp_node::nominalFeedbackEntry(measured, .034906585F));
}

#include "vamp_node/upright_motion.hpp"

TEST(UprightMotion, ConstrainsObjectAxisNotToolRollAndPitch)
{
  Eigen::Isometry3f attachment=Eigen::Isometry3f::Identity();
  attachment.linear()=Eigen::AngleAxisf(.8F,Eigen::Vector3f::UnitY()).toRotationMatrix();
  auto tool=attachment;
  tool.linear()=Eigen::AngleAxisf(1.5F,Eigen::Vector3f::UnitZ()).toRotationMatrix()*attachment.linear();
  EXPECT_TRUE(vamp_node::objectRemainsUpright(tool,attachment,.05235988F));
  tool.linear()=Eigen::AngleAxisf(.1F,Eigen::Vector3f::UnitX()).toRotationMatrix()*tool.linear();
  EXPECT_FALSE(vamp_node::objectRemainsUpright(tool,attachment,.05235988F));
}

TEST(UprightMotion, RejectsTiltBetweenUprightEndpointsWithoutPartialOutput)
{
  const auto fk=[](const std::vector<float> &q) {
    auto pose=Eigen::Isometry3f::Identity();
    pose.linear()=Eigen::AngleAxisf(.15F*std::sin(q[0]*3.14159265F),Eigen::Vector3f::UnitX()).toRotationMatrix();
    return pose;
  };
  const std::vector<float> start(6,0.F);
  auto goal=start;goal[0]=1.F;
  EXPECT_TRUE(vamp_node::objectRemainsUpright(fk(goal),fk(start),.05235988F));
  const auto path=vamp_node::validateUprightPath(start,{goal},fk,fk(start),.05235988F,
    [](const auto &,size_t,bool) {return true;});
  EXPECT_FALSE(path.success);
  EXPECT_TRUE(path.waypoint_joints.empty());
}

TEST(UprightMotion, SamplesYawOnlyPathAndRetainsCollisionChecking)
{
  const auto fk=[](const std::vector<float> &q) {
    auto pose=Eigen::Isometry3f::Identity();
    pose.linear()=Eigen::AngleAxisf(q[0],Eigen::Vector3f::UnitZ()).toRotationMatrix();
    return pose;
  };
  const std::vector<float> start(6,0.F);
  auto goal=start;goal[0]=1.F;
  auto path=vamp_node::validateUprightPath(start,{goal},fk,fk(start),.05235988F,
    [](const auto &,size_t,bool) {return true;});
  ASSERT_TRUE(path.success);
  EXPECT_GE(path.waypoint_joints.size(),200U);
  EXPECT_EQ(path.waypoint_joints.back(),goal);
  path=vamp_node::validateUprightPath(start,{goal},fk,fk(start),.05235988F,
    [](const auto &q,size_t,bool) {return q[0]<.4F || q[0]>.6F;});
  EXPECT_FALSE(path.success);
  EXPECT_TRUE(path.waypoint_joints.empty());
}

TEST(LiftRequest, NoFixedHeightCapButRejectsNonfiniteAndNegative)
{
  EXPECT_TRUE(vamp_node::validLiftDistance(0.0F));
  EXPECT_TRUE(vamp_node::validLiftDistance(0.001F));
  EXPECT_TRUE(vamp_node::validLiftDistance(0.15F));
  EXPECT_TRUE(vamp_node::validLiftDistance(1.0F));
  EXPECT_FALSE(vamp_node::validLiftDistance(-0.01F));
  EXPECT_FALSE(vamp_node::validLiftDistance(std::numeric_limits<float>::infinity()));
  EXPECT_FALSE(vamp_node::validLiftDistance(std::numeric_limits<float>::quiet_NaN()));
}

TEST(CartesianApproach, HugeFiniteDistanceRejectsBeforeAllocationOrIk)
{
  auto start = Eigen::Isometry3f::Identity();
  auto goal = start;
  goal.translation().z() = 1.e30F;
  bool called = false;
  const auto plan = vamp_node::planCartesianApproach(start, goal,
    std::vector<float>{0.F, 0.F, 0.F, 0.F, 0.F, 0.F}, .002F, .05F,
    [&](const auto &, const auto &, auto &) {called = true; return false;},
    [](const auto &, size_t, bool) {return true;});
  EXPECT_FALSE(plan.success);
  EXPECT_FALSE(called);
  EXPECT_EQ(plan.failure_reason, "CARTESIAN_WAYPOINT_BUDGET_EXCEEDED");
  EXPECT_TRUE(plan.waypoint_joints.empty());
}
