#pragma once

#include "vamp_node/placement_point_index.hpp"

namespace vamp_node {
// A semantic silhouette can contain separated near/far surfaces of one hollow
// object. Require substantial measured support and multiple real 3-D neighbors;
// median camera range alone is not an object-membership test.
inline bool substantialSurfaceConnected(const std::vector<Eigen::Vector3f> &points,
    const PlacementPointIndex &foreground, size_t minimum_points, float distance)
{
    if (points.size()<minimum_points || distance<=0.F || !std::isfinite(distance)) return false;
    size_t neighbors=0;
    for (const auto &point:points) {
        if (point.allFinite() && !foreground.sphereClear(point,distance) && ++neighbors>=3) return true;
    }
    return false;
}
}  // namespace vamp_node
