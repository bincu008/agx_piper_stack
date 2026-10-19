#pragma once
#include <Eigen/Dense>

namespace vision_node
{
// Compose in base axes, not camera axes. No additional cloud traversal.
inline Eigen::Matrix4f translatedPerceptionTransform(
  const Eigen::Matrix4f & manual_fk_x, const Eigen::Vector3f & offset)
{
  Eigen::Matrix4f result = manual_fk_x;
  result.block<3, 1>(0, 3) += offset;
  return result;
}
}
