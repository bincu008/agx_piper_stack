#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <Eigen/Core>

namespace vision_node {
// A summary index of the same synchronized geometry used by ActiveVision.
// It never replaces the fine occupancy/collision world. Only 16 times persist.
class SpatialGrid {
public:
  static constexpr float side = .30F;
  std::array<int64_t, 16> last_scanned{};
  std::array<unsigned, 16> observed{};  // Latest-frame derived view, not history.

  static Eigen::Vector3f minimum(int index) {
    return {side * (index % 2), -.60F + side * ((index / 2) % 4),
            side * (index / 8)};
  }
  static std::string id(int index) {
    return "x" + std::to_string(index % 2) + "_y" +
      std::to_string((index / 2) % 4) + "_z" + std::to_string(index / 8);
  }
  static int index(const Eigen::Vector3f &p) {
    if (!p.allFinite() || p.x()<0.F || p.x()>=.60F || p.y()<-.60F ||
        p.y()>=.60F || p.z()<0.F || p.z()>=.60F) return -1;
    return static_cast<int>(std::floor(p.x()/side)) +
      2*static_cast<int>(std::floor((p.y()+.60F)/side)) +
      8*static_cast<int>(std::floor(p.z()/side));
  }
  static std::string cell(const Eigen::Vector3f &p) {
    const int value=index(p); return value<0 ? "" : id(value);
  }
  // Exact bounded ray-segment/AABB intersection, including empty traversed cells.
  // Positive-length intersections only: a grazing edge cannot refresh a cell.
  static bool traverses(const Eigen::Vector3f &origin, const Eigen::Vector3f &hit, int i) {
    if (!origin.allFinite() || !hit.allFinite() || (hit-origin).norm()<1.e-5F) return false;
    const auto low=minimum(i); const Eigen::Vector3f high=low+Eigen::Vector3f::Constant(side);
    const Eigen::Vector3f delta=hit-origin;
    float begin=0.F,end=1.F;
    for (int axis=0;axis<3;++axis) {
      if (std::abs(delta[axis])<1.e-8F) {
        if (origin[axis]<low[axis] || origin[axis]>=high[axis]) return false;
      } else {
        float a=(low[axis]-origin[axis])/delta[axis];
        float b=(high[axis]-origin[axis])/delta[axis];
        if (a>b) std::swap(a,b);
        begin=std::max(begin,a);end=std::min(end,b);
        if (end-begin<=1.e-6F) return false;
      }
    }
    return end-begin>1.e-6F;
  }
  void update(const Eigen::Vector3f &origin, const std::vector<Eigen::Vector3f> &hits,
              int64_t stamp) {
    observed.fill(0);
    if (!origin.allFinite() || stamp<=0) return;
    for (const auto &hit:hits) {
      if (!hit.allFinite()) continue;
      const int endpoint=index(hit);
      for (int i=0;i<16;++i)
        if (i==endpoint || traverses(origin,hit,i)) ++observed[i];
    }
    for (int i=0;i<16;++i)
      if (observed[i]>=3) last_scanned[i]=std::max(last_scanned[i],stamp);
  }
};
}  // namespace vision_node
