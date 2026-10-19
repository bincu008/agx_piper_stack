#pragma once

#include "vamp_node/cartesian_grasp_approach.hpp"

namespace vamp_node {

// Source upright objects have +Z along base_link gravity. The attachment maps
// that OBJECT axis into the tool frame; tool roll/pitch are not constrained.
inline bool objectRemainsUpright(const Eigen::Isometry3f &tool,
  const Eigen::Isometry3f &source_tool, float tolerance_rad)
{
  if (!finiteIsometry(tool) || !finiteIsometry(source_tool) ||
      !std::isfinite(tolerance_rad) || tolerance_rad < 0.F) return false;
  const Eigen::Vector3f up = tool.linear()*source_tool.linear().transpose()*Eigen::Vector3f::UnitZ();
  return up.dot(Eigen::Vector3f::UnitZ()) >= std::cos(tolerance_rad)-1.e-6F;
}

// Reuse checked joint interpolation to validate between IK/RRT waypoints.
// A partial segment never becomes an executable result.
inline CartesianApproachPlan validateUprightPath(const std::vector<float> &start,
  const std::vector<std::vector<float>> &path,
  const std::function<Eigen::Isometry3f(const std::vector<float> &)> &fk,
  const Eigen::Isometry3f &attachment, float tolerance,
  const CartesianCollisionChecker &check)
{
  CartesianApproachPlan result;
  auto previous = start;
  if (!objectRemainsUpright(fk(start), attachment, tolerance)) {
    result.failure_reason = "OBJECT_NOT_UPRIGHT_AT_START"; return result;
  }
  for (const auto &target : path) {
    const auto edge = planJointApproach(previous, target, fk,
      [&](const auto &q, size_t index, bool terminal) {
        return objectRemainsUpright(fk(q),attachment,tolerance) && check(q,index,terminal);
      });
    if (!edge.success) {
      result.failure_reason = "UPRIGHT_EDGE_REJECTED:"+edge.failure_reason;
      result.waypoint_joints.clear(); return result;
    }
    result.waypoint_joints.insert(result.waypoint_joints.end(),
      edge.waypoint_joints.begin(),edge.waypoint_joints.end());
    previous=target;
  }
  result.success=!path.empty();
  return result;
}

}  // namespace vamp_node
