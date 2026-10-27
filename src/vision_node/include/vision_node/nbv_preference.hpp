#pragma once
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <vector>

namespace vision_node {
inline Eigen::Vector3f baseFacingViewDirection(const Eigen::Vector3f &target) {
  Eigen::Vector3f toward_base(-target.x(), -target.y(), 0.F);
  if (toward_base.norm() < 1.e-6F) toward_base = -Eigen::Vector3f::UnitX();
  toward_base.normalize();
  constexpr float tilt = 15.F * 3.14159265358979323846F / 180.F;
  return toward_base*std::sin(tilt)+Eigen::Vector3f::UnitZ()*std::cos(tilt);
}
inline float nextViewCost(const Eigen::Vector3f &position,
    const Eigen::Vector3f &target, const Eigen::Vector3f &current) {
  const auto offset = position-target;
  if (!offset.allFinite() || offset.norm() < 1.e-6F) return INFINITY;
  return 4.F*(1.F-offset.normalized().dot(baseFacingViewDirection(target)))+
    .1F*(position-current).norm()/offset.norm();
}
inline std::vector<Eigen::Vector3f> nextViewSamples(
    const Eigen::Vector3f &target, float radius) {
  std::vector<Eigen::Vector3f> samples;
  if (!target.allFinite() || !std::isfinite(radius) || radius <= 0.F) return samples;
  const auto preferred = baseFacingViewDirection(target);
  samples.push_back(target+radius*preferred);
  samples.push_back(target+radius*Eigen::Vector3f::UnitZ());
  for (float angle : {-30.F, 30.F, -60.F, 60.F}) {
    const auto rotation = Eigen::AngleAxisf(angle*3.14159265358979323846F/180.F,
                                          Eigen::Vector3f::UnitZ());
    samples.push_back(target+radius*(rotation*preferred));
  }
  constexpr float golden = 3.14159265358979323846F*(3.F-2.2360679775F);
  for (int i=0; i<150; ++i) {
    const float y=1.F-2.F*i/149.F, r=std::sqrt(std::max(0.F,1.F-y*y));
    samples.push_back(target+radius*Eigen::Vector3f(
      std::cos(golden*i)*r, y, std::sin(golden*i)*r));
  }
  return samples;
}
}  // namespace vision_node
