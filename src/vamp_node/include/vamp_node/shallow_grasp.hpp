#pragma once

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <limits>
#include <vector>

namespace vamp_node {
struct GraspSupportPlane {
  Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
  float offset = 0.F;
};

inline std::optional<GraspSupportPlane> fitGraspSupportPlane(
    const std::vector<Eigen::Vector3f> &scene, const Eigen::Vector3f &center) {
  std::vector<Eigen::Vector3f> local;
  std::vector<float> heights;
  for (const auto &p : scene) {
    if (p.allFinite() && (p-center).head<2>().squaredNorm() <= .16F*.16F) {
      local.push_back(p); heights.push_back(p.z());
    }
  }
  if (local.size()<32) return std::nullopt;
  const size_t percentile=heights.size()/5;
  std::nth_element(heights.begin(),heights.begin()+percentile,heights.end());
  const float height=heights[percentile];
  std::vector<Eigen::Vector3f> patch;
  for (const auto &p : local) if (std::abs(p.z()-height)<=.006F) patch.push_back(p);
  GraspSupportPlane plane;
  for (int iteration=0;iteration<2;++iteration) {
    if (patch.size()<32) return std::nullopt;
    Eigen::Vector3f mean=Eigen::Vector3f::Zero();
    for (const auto &p : patch) mean+=p;
    mean/=static_cast<float>(patch.size());
    Eigen::Matrix3f covariance=Eigen::Matrix3f::Zero();
    for (const auto &p : patch) {
      const Eigen::Vector3f delta=p-mean;
      covariance+=delta*delta.transpose();
    }
    covariance/=static_cast<float>(patch.size());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(covariance);
    if (eig.info()!=Eigen::Success || eig.eigenvalues()[1]<.0001F) return std::nullopt;
    plane.normal=eig.eigenvectors().col(0);
    if (plane.normal.z()<0) plane.normal=-plane.normal;
    if (plane.normal.z()<.9659F) return std::nullopt;
    plane.offset=plane.normal.dot(mean);
    patch.clear();
    for (const auto &p : local)
      if (std::abs(plane.normal.dot(p)-plane.offset)<=.003F) patch.push_back(p);
  }
  return plane;
}

inline float fingerSupportClearance(const Eigen::Isometry3f &pose,
    const GraspSupportPlane &plane, float opening, float length, float width,
    float height, float tcp_offset) {
  float clearance=std::numeric_limits<float>::infinity();
  for (float sign : {-1.F,1.F}) for (float x : {-height*.5F,height*.5F})
    for (float y : {sign*opening*.5F,sign*(opening*.5F+width)})
      for (float z : {0.F,length}) {
        const Eigen::Vector3f vertex=pose*Eigen::Vector3f(x,y,z-tcp_offset);
        clearance=std::min(clearance,plane.normal.dot(vertex)-plane.offset);
      }
  return clearance;
}

inline bool measuredPadContactsSupported(const Eigen::Isometry3f &pose,
    const std::array<Eigen::Vector3f,2> &contacts, float opening, float length,
    float height, float tcp_offset) {
  if (!pose.matrix().allFinite() || !std::isfinite(opening) || opening<=0) return false;
  for (const auto &contact : contacts) {
    const Eigen::Vector3f local=pose.linear().transpose()*(contact-pose.translation());
    const float depth=local.z()+tcp_offset;
    if (!local.allFinite() || depth<.002F || depth>length-.002F ||
        std::abs(local.x())>height*.5F+.003F ||
        std::abs(std::abs(local.y())-opening*.5F)>.004F) return false;
  }
  return true;
}

// Center the same measured pair on the pads, prefer 4 mm tip engagement,
// then try 7/10 mm only if the shallower target cannot pass ordinary checks.
inline std::vector<Eigen::Isometry3f> shallowContactTargets(
    const Eigen::Isometry3f &original, const std::array<Eigen::Vector3f,2> &contacts,
    float opening, float length, float width, float height, float tcp_offset,
    const GraspSupportPlane &plane) {
  std::vector<Eigen::Isometry3f> targets;
  const Eigen::Vector3f midpoint=(contacts[0]+contacts[1])*.5F;
  const float depth_span=std::abs((contacts[0]-contacts[1]).dot(original.linear().col(2)));
  for (float engagement : {.004F,.007F,.010F}) {
    auto target=original;
    const float mean_depth=length-engagement-depth_span*.5F;
    target.translation()=midpoint-original.linear().col(2)*(mean_depth-tcp_offset);
    const float clearance=fingerSupportClearance(target,plane,
      std::min(.10F,opening+.02F),length,width,height,tcp_offset);
    // Include 1 mm for the bounded refinement solve, not a geometry shrink.
    const float rise=std::max(0.F,.004F-clearance);
    if (rise>.010F) continue;
    target.translation()+=plane.normal*rise;
    if ((target.translation()-original.translation()).norm()>.020F ||
        !measuredPadContactsSupported(target,contacts,opening,length,height,tcp_offset)) continue;
    targets.push_back(target);
  }
  return targets;
}
}  // namespace vamp_node
