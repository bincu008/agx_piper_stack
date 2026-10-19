#ifndef VAMP_NODE__GRASP_POSITION_OFFSET_HPP_
#define VAMP_NODE__GRASP_POSITION_OFFSET_HPP_

#include <cmath>
#include <stdexcept>
#include <vector>
#include <Eigen/Geometry>

namespace vamp_node
{
inline Eigen::Vector3f validatedGraspPositionOffset(const std::vector<double> & values)
{
    if (values.size() != 3 || !std::isfinite(values[0]) ||
        !std::isfinite(values[1]) || !std::isfinite(values[2])) {
        throw std::invalid_argument("grasp_position_offset_base_m requires three finite metres");
    }
    return Eigen::Vector3f(values[0], values[1], values[2]);
}

inline Eigen::Isometry3f correctedGraspTarget(
    const Eigen::Isometry3f & target, const Eigen::Vector3f & offset, bool apply = true)
{
    Eigen::Isometry3f result = target;
    if (apply) result.translation() += offset;
    return result;
}
}  // namespace vamp_node
#endif  // VAMP_NODE__GRASP_POSITION_OFFSET_HPP_
