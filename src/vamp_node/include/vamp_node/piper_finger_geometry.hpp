#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <Eigen/Geometry>

namespace vamp_node {
// Generated Piper spheres freeze each slider at +/-35 mm (70 mm opening).
// Retain every detailed sphere/radius, translating only the two finger links.
inline Eigen::Vector3f piperFingerSphereCenter(size_t index,
    const Eigen::Vector3f &center, const Eigen::Isometry3f &pose, float opening) {
    if (index < 82 || index >= 96 || !std::isfinite(opening) || opening < 0.0F)
        return center;
    const float side = index < 89 ? -1.0F : 1.0F;
    return center + pose.linear().col(1) *
        (side * 0.5F * (std::clamp(opening, 0.0F, 0.10F) - 0.070F));
}
// Conservative bounds of BOTH supplied Piper collision STL meshes, transformed
// by their URDF slider origins (z=138 mm, roll=pi/2, mirrored yaw).
// Detailed spheres remain the broad phase; these bounds only reject hits that
// cannot touch either mesh. Never use this refinement for palm/arm spheres.
inline bool piperFingerMeshBoundsContain(size_t index, const Eigen::Vector3f &point,
    const Eigen::Isometry3f &pose, float opening, float padding = 0.0025F) {
    if (index < 82 || index >= 96 || !std::isfinite(opening) || opening < 0.0F)
        return true;
    const Eigen::Vector3f local = pose.inverse() * point;
    const float gap = 0.5F * std::clamp(opening, 0.0F, 0.10F);
    const float inner = index < 89 ? -gap : gap;
    const float outer = index < 89 ? -gap - 0.0265F : gap + 0.0265F;
    return std::abs(local.x()) <= 0.028F + padding &&
        local.y() >= std::min(inner, outer) - padding &&
        local.y() <= std::max(inner, outer) + padding &&
        local.z() >= 0.0615F - padding && local.z() <= 0.138F + padding;
}
}  // namespace vamp_node
