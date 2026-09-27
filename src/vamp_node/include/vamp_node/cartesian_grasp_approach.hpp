#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "vamp_node/deterministic_ik.hpp"

namespace vamp_node
{

// This utility deliberately has no ROS or hardware dependencies. The VAMP
// service supplies the IK and collision callbacks, while tests can inject
// deterministic stand-ins for each one.
struct CartesianApproachPlan
{
  bool success = false;
  std::string failure_reason;
  std::vector<Eigen::Isometry3f> waypoint_poses;
  std::vector<std::vector<float>> waypoint_joints;
};

using CartesianIkSolver = std::function<bool(
  const Eigen::Isometry3f &, const std::vector<float> &, std::vector<float> &)>;
using CartesianCollisionChecker = std::function<bool(
  const std::vector<float> &, std::size_t, bool)>;

inline bool finiteIsometry(const Eigen::Isometry3f &pose)
{
  return pose.matrix().allFinite() &&
    pose.linear().col(0).squaredNorm() > 1.0e-8F &&
    pose.linear().col(1).squaredNorm() > 1.0e-8F &&
    pose.linear().col(2).squaredNorm() > 1.0e-8F;
}

inline bool jointsWithinPiperLimits(const std::vector<float> &joints)
{
  if (joints.size() != 6) return false;
  for (std::size_t index = 0; index < joints.size(); ++index) {
    if (!std::isfinite(joints[index]) ||
      joints[index] < kPiperJointLower[index] - 1.0e-5F ||
      joints[index] > kPiperJointUpper[index] + 1.0e-5F)
    {
      return false;
    }
  }
  return true;
}

inline float maximumJointDelta(
  const std::vector<float> &first, const std::vector<float> &second)
{
  float maximum = 0.0F;
  for (std::size_t index = 0; index < 6; ++index) {
    maximum = std::max(maximum, std::abs(first[index] - second[index]));
  }
  return maximum;
}

inline CartesianApproachPlan planCartesianApproach(
  const Eigen::Isometry3f &start_pose,
  const Eigen::Isometry3f &goal_pose,
  const std::vector<float> &start_joints,
  float maximum_translation_step_m,
  float maximum_joint_step_rad,
  const CartesianIkSolver &solve_ik,
  const CartesianCollisionChecker &check_collision)
{
  CartesianApproachPlan result;
  if (!finiteIsometry(start_pose) || !finiteIsometry(goal_pose)) {
    result.failure_reason = "INVALID_CARTESIAN_POSE";
    return result;
  }
  if (!jointsWithinPiperLimits(start_joints)) {
    result.failure_reason = "INVALID_START_JOINTS";
    return result;
  }
  if (!solve_ik || !check_collision) {
    result.failure_reason = "PLANNER_CALLBACK_UNAVAILABLE";
    return result;
  }
  if (!std::isfinite(maximum_translation_step_m) ||
    maximum_translation_step_m <= 0.0F ||
    !std::isfinite(maximum_joint_step_rad) || maximum_joint_step_rad <= 0.0F)
  {
    result.failure_reason = "INVALID_FINAL_APPROACH_LIMITS";
    return result;
  }

  // The service is never allowed to loosen the 2 mm Cartesian bound.
  const float translation_step = std::min(maximum_translation_step_m, 0.002F);
  const float distance =
    (goal_pose.translation() - start_pose.translation()).norm();
  const std::size_t step_count = std::max<std::size_t>(
    1, static_cast<std::size_t>(std::ceil(distance / translation_step)));

  result.waypoint_poses.reserve(step_count);
  result.waypoint_joints.reserve(step_count);
  std::vector<float> previous_joints = start_joints;
  for (std::size_t index = 0; index < step_count; ++index) {
    const float fraction = static_cast<float>(index + 1) /
      static_cast<float>(step_count);
    Eigen::Isometry3f waypoint = goal_pose;
    waypoint.translation() = start_pose.translation() +
      fraction * (goal_pose.translation() - start_pose.translation());
    // The refined orientation is held fixed for the complete final segment.
    waypoint.linear() = goal_pose.linear();

    std::vector<float> solved_joints;
    if (!solve_ik(waypoint, previous_joints, solved_joints)) {
      std::ostringstream reason;
      reason << "IK_FAILED_AT_WAYPOINT_" << (index + 1);
      result.failure_reason = reason.str();
      result.waypoint_poses.clear();
      result.waypoint_joints.clear();
      return result;
    }
    if (!jointsWithinPiperLimits(solved_joints)) {
      std::ostringstream reason;
      reason << "JOINT_LIMIT_VIOLATION_AT_WAYPOINT_" << (index + 1);
      result.failure_reason = reason.str();
      result.waypoint_poses.clear();
      result.waypoint_joints.clear();
      return result;
    }
    if (maximumJointDelta(previous_joints, solved_joints) >
      maximum_joint_step_rad)
    {
      std::ostringstream reason;
      reason << "JOINT_DISCONTINUITY_AT_WAYPOINT_" << (index + 1);
      result.failure_reason = reason.str();
      result.waypoint_poses.clear();
      result.waypoint_joints.clear();
      return result;
    }

    const bool final_waypoint = index + 1 == step_count;
    if (!check_collision(solved_joints, index, final_waypoint)) {
      std::ostringstream reason;
      reason << "COLLISION_AT_WAYPOINT_" << (index + 1);
      result.failure_reason = reason.str();
      result.waypoint_poses.clear();
      result.waypoint_joints.clear();
      return result;
    }

    result.waypoint_poses.push_back(waypoint);
    result.waypoint_joints.push_back(solved_joints);
    previous_joints = std::move(solved_joints);
  }

  result.success = true;
  result.failure_reason = "OK";
  return result;
}

}  // namespace vamp_node
