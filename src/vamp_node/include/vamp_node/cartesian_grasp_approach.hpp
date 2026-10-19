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

// Zero retains the service default; finite positive requests have no height cap.
inline bool validLiftDistance(float distance)
{
  return std::isfinite(distance) && distance >= 0.0F;
}

inline bool graspSearchBudgetExhausted(int tested, int fast_limit, int hard_limit,
  bool have_valid_grasp)
{
  return tested >= hard_limit || (have_valid_grasp && tested >= fast_limit);
}

inline bool checkLiftRobotEnvironment(bool lifting, bool bypass)
{
  return !lifting || !bypass;
}

inline bool approachPalmSphereEnabled(std::size_t sphere, bool skip_80)
{
  return !skip_80 || sphere != 80;
}

inline bool initialGraspChecksEnvironmentSphere(std::size_t sphere, bool arm_only)
{
  return !arm_only || sphere <= 67;
}

inline bool finalApproachChecksEnvironmentSphere(std::size_t sphere, bool check_fingers)
{
  // 0..73 arm/wrist, 74..81 palm/base, 82..95 finger bodies/pads.
  return check_fingers || sphere < 82;
}

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
  const double distance =
    (goal_pose.translation() - start_pose.translation()).cast<double>().norm();
  const double requested_steps = std::ceil(distance / translation_step);
  // Bound planner resources independently of the operator's requested height.
  // Check before conversion/reserve, including very large finite requests.
  if (!std::isfinite(requested_steps) || requested_steps > 4096.0) {
    result.failure_reason = "CARTESIAN_WAYPOINT_BUDGET_EXCEEDED";
    return result;
  }
  const std::size_t step_count = std::max<std::size_t>(
    1, static_cast<std::size_t>(requested_steps));

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

// Joint-space transition: orientation may change while translation progresses.
// Sample at <=0.005 rad and additionally bound adjacent TCP samples to 2 mm.
inline CartesianApproachPlan planJointApproach(
  const std::vector<float> &start, const std::vector<float> &goal,
  const std::function<Eigen::Isometry3f(const std::vector<float> &)> &fk,
  const CartesianCollisionChecker &check)
{
  CartesianApproachPlan result;
  if (!jointsWithinPiperLimits(start) || !jointsWithinPiperLimits(goal) || !fk || !check) {
    result.failure_reason = "INVALID_JOINT_APPROACH_INPUT";
    return result;
  }
  std::size_t count = std::max<std::size_t>(1,
    static_cast<std::size_t>(std::ceil(maximumJointDelta(start, goal)/0.005F)));
  for (; count <= 8192; count *= 2) {
    result.waypoint_poses.clear();
    result.waypoint_joints.clear();
    Eigen::Isometry3f previous = fk(start);
    if (!finiteIsometry(previous)) break;
    bool refine = false;
    for (std::size_t i = 0; i < count; ++i) {
      std::vector<float> joints(6);
      const float fraction = static_cast<float>(i+1)/static_cast<float>(count);
      for (std::size_t j = 0; j < 6; ++j) joints[j] = i+1 == count ? goal[j] : start[j]+fraction*(goal[j]-start[j]);
      const auto pose = fk(joints);
      if (!finiteIsometry(pose)) {
        result.failure_reason = "JOINT_APPROACH_INVALID_FK";
        result.waypoint_joints.clear(); result.waypoint_poses.clear(); return result;
      }
      if ((pose.translation()-previous.translation()).norm() > 0.002F) {
        refine = true; break;
      }
      result.waypoint_joints.push_back(joints);
      result.waypoint_poses.push_back(pose);
      previous = pose;
    }
    if (refine) continue;
    for (std::size_t i = 0; i < count; ++i) {
      if (!check(result.waypoint_joints[i], i, i+1 == count)) {
        result.failure_reason = "JOINT_APPROACH_COLLISION_AT_WAYPOINT_"+std::to_string(i+1);
        result.waypoint_joints.clear(); result.waypoint_poses.clear(); return result;
      }
    }
    result.success = true; result.failure_reason = "OK"; return result;
  }
  result.waypoint_joints.clear(); result.waypoint_poses.clear();
  result.failure_reason = "JOINT_APPROACH_SAMPLING_LIMIT";
  return result;
}

// Preserve the supplied endpoint; optional intermediates cannot invalidate it.
inline CartesianApproachPlan planOptionalJointApproach(
  const std::vector<float> &start, const std::vector<float> &goal,
  const std::vector<std::vector<float>> &optional,
  const std::function<Eigen::Isometry3f(const std::vector<float> &)> &fk,
  const CartesianCollisionChecker &check)
{
  CartesianApproachPlan prefix;
  std::vector<float> previous = start;
  for (const auto &near : optional) {
    auto segment = planJointApproach(previous, near, fk, check);
    if (!segment.success) break;
    auto tail = planJointApproach(near, goal, fk, check);
    if (!tail.success) break;
    // Keep only intermediates with a validated connection to the fixed goal.
    prefix.waypoint_joints.insert(prefix.waypoint_joints.end(),
      segment.waypoint_joints.begin(), segment.waypoint_joints.end());
    prefix.waypoint_poses.insert(prefix.waypoint_poses.end(),
      segment.waypoint_poses.begin(), segment.waypoint_poses.end());
    previous = near;
  }
  auto tail = planJointApproach(previous, goal, fk, check);
  if (!tail.success) return planJointApproach(start, goal, fk, check);
  const bool used = !prefix.waypoint_joints.empty();
  prefix.waypoint_joints.insert(prefix.waypoint_joints.end(),
    tail.waypoint_joints.begin(), tail.waypoint_joints.end());
  prefix.waypoint_poses.insert(prefix.waypoint_poses.end(),
    tail.waypoint_poses.begin(), tail.waypoint_poses.end());
  prefix.success = true;
  prefix.failure_reason = used ? "OK;NEAR_GRASP_USED" : "OK;NEAR_GRASP_SKIPPED";
  return prefix;
}

inline CartesianApproachPlan planStandoffAlignment(
  const Eigen::Isometry3f &start_pose, const Eigen::Matrix3f &goal_rotation,
  const std::vector<float> &start_joints, float maximum_joint_step_rad,
  const CartesianIkSolver &solve_ik, const CartesianCollisionChecker &check_collision)
{
  CartesianApproachPlan result;
  Eigen::Isometry3f goal = start_pose;
  goal.linear() = goal_rotation;
  if (!finiteIsometry(start_pose) || !finiteIsometry(goal) ||
    !jointsWithinPiperLimits(start_joints) || !solve_ik || !check_collision ||
    !std::isfinite(maximum_joint_step_rad) || maximum_joint_step_rad <= 0.0F)
  {
    result.failure_reason = "INVALID_STANDOFF_ALIGNMENT_INPUT";
    return result;
  }
  Eigen::Quaternionf first(start_pose.linear()), last(goal_rotation);
  first.normalize();
  last.normalize();
  const float angle = first.angularDistance(last);
  constexpr float degree = 0.01745329252F;
  if (angle > 20.0F * degree + 0.0001F) {
    result.failure_reason = "STANDOFF_ALIGNMENT_EXCEEDS_20_DEGREES";
    return result;
  }
  if (angle < 0.0001F) {
    result.success = true;
    result.failure_reason = "OK";
    return result;
  }
  const std::size_t count = static_cast<std::size_t>(std::ceil(angle / degree));
  auto previous = start_joints;
  for (std::size_t index = 0; index < count; ++index) {
    Eigen::Isometry3f waypoint = start_pose;
    waypoint.linear() = first.slerp(static_cast<float>(index+1)/count, last).toRotationMatrix();
    std::vector<float> solution;
    std::string reason;
    if (!solve_ik(waypoint, previous, solution)) reason = "IK_FAILED";
    else if (!jointsWithinPiperLimits(solution)) reason = "JOINT_LIMIT_VIOLATION";
    else if (maximumJointDelta(previous, solution) > maximum_joint_step_rad)
      reason = "JOINT_DISCONTINUITY";
    else if (!check_collision(solution, index, index+1 == count)) reason = "COLLISION";
    if (!reason.empty()) {
      result.failure_reason = "STANDOFF_ALIGNMENT_"+reason+"_AT_WAYPOINT_"+
        std::to_string(index+1);
      result.waypoint_poses.clear();
      result.waypoint_joints.clear();
      return result;
    }
    result.waypoint_poses.push_back(waypoint);
    result.waypoint_joints.push_back(solution);
    previous = std::move(solution);
  }
  result.success = true;
  result.failure_reason = "OK";
  return result;
}

}  // namespace vamp_node
