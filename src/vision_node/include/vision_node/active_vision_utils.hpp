#ifndef VISION_NODE__ACTIVE_VISION_UTILS_HPP_
#define VISION_NODE__ACTIVE_VISION_UTILS_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <octomap/OcTree.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace vision_node
{

// ---------------------------------------------------------
// Metric scale camera measurement correction
// ---------------------------------------------------------
inline sensor_msgs::msg::PointCloud2 correct_camera_measurements(
  const sensor_msgs::msg::PointCloud2 & input, double measured_to_real_ratio)
{
  if (!std::isfinite(measured_to_real_ratio) || measured_to_real_ratio <= 0.) {
    throw std::invalid_argument("perception_measurement_ratio must be finite and positive");
  }
  auto output = input;
  if (measured_to_real_ratio == 1. || output.data.empty()) return output;
  const float correction = static_cast<float>(1. / measured_to_real_ratio);
  sensor_msgs::PointCloud2Iterator<float> x(output, "x"), y(output, "y"), z(output, "z");
  for (; x != x.end(); ++x, ++y, ++z) {
    *x *= correction; *y *= correction; *z *= correction;
  }
  return output;
}

inline sensor_msgs::msg::Image corrected_registered_depth(
  const sensor_msgs::msg::PointCloud2 & cloud)
{
  sensor_msgs::msg::Image depth;
  depth.header = cloud.header;
  depth.height = cloud.height;
  depth.width = cloud.width;
  depth.encoding = "32FC1";
  depth.is_bigendian = cloud.is_bigendian;
  depth.step = cloud.width * sizeof(float);
  depth.data.resize(static_cast<size_t>(depth.step) * depth.height);
  sensor_msgs::PointCloud2ConstIterator<float> z(cloud, "z");
  size_t offset = 0;
  for (; z != z.end(); ++z, offset += sizeof(float)) {
    std::memcpy(depth.data.data() + offset, &*z, sizeof(float));
  }
  return depth;
}

// ---------------------------------------------------------
// Perception translation in base axes
// ---------------------------------------------------------
inline Eigen::Matrix4f translatedPerceptionTransform(
  const Eigen::Matrix4f & manual_fk_x, const Eigen::Vector3f & offset)
{
  Eigen::Matrix4f result = manual_fk_x;
  result.block<3, 1>(0, 3) += offset;
  return result;
}

// ---------------------------------------------------------
// NBV direction and candidate sampling
// ---------------------------------------------------------
inline Eigen::Vector3f baseFacingViewDirection(const Eigen::Vector3f & target)
{
  Eigen::Vector3f toward_base(-target.x(), -target.y(), 0.F);
  if (toward_base.norm() < 1.e-6F) toward_base = -Eigen::Vector3f::UnitX();
  toward_base.normalize();
  constexpr float tilt = 15.F * 3.14159265358979323846F / 180.F;
  return toward_base * std::sin(tilt) + Eigen::Vector3f::UnitZ() * std::cos(tilt);
}

inline float nextViewCost(
  const Eigen::Vector3f & position, const Eigen::Vector3f & target,
  const Eigen::Vector3f & current)
{
  const auto offset = position - target;
  if (!offset.allFinite() || offset.norm() < 1.e-6F) return INFINITY;
  return 4.F * (1.F - offset.normalized().dot(baseFacingViewDirection(target))) +
    .1F * (position - current).norm() / offset.norm();
}

inline std::vector<Eigen::Vector3f> nextViewSamples(
  const Eigen::Vector3f & target, float radius)
{
  std::vector<Eigen::Vector3f> samples;
  if (!target.allFinite() || !std::isfinite(radius) || radius <= 0.F) return samples;
  const auto preferred = baseFacingViewDirection(target);
  samples.push_back(target + radius * preferred);
  samples.push_back(target + radius * Eigen::Vector3f::UnitZ());
  for (float angle : {-30.F, 30.F, -60.F, 60.F}) {
    const auto rotation = Eigen::AngleAxisf(
      angle * 3.14159265358979323846F / 180.F, Eigen::Vector3f::UnitZ());
    samples.push_back(target + radius * (rotation * preferred));
  }
  constexpr float golden = 3.14159265358979323846F * (3.F - 2.2360679775F);
  for (int i = 0; i < 150; ++i) {
    const float y = 1.F - 2.F * i / 149.F, r = std::sqrt(std::max(0.F, 1.F - y * y));
    samples.push_back(target + radius * Eigen::Vector3f(
      std::cos(golden * i) * r, y, std::sin(golden * i) * r));
  }
  return samples;
}

// ---------------------------------------------------------
// Target visibility and ray casting
// ---------------------------------------------------------
inline bool viewRayClear(
  const octomap::OcTree & tree, const Eigen::Vector3f & origin,
  const Eigen::Vector3f & target, float margin)
{
  if (!origin.allFinite() || !target.allFinite() || !std::isfinite(margin) || margin < 0.F)
    return false;
  const Eigen::Vector3f delta = target - origin;
  const float length = delta.norm();
  if (length <= margin) return false;
  octomap::point3d hit;
  if (!tree.castRay(octomap::point3d(origin.x(), origin.y(), origin.z()),
                    octomap::point3d(delta.x(), delta.y(), delta.z()), hit, true, length - margin))
    return true;
  return (Eigen::Vector3f(hit.x(), hit.y(), hit.z()) - origin).norm() >= length - margin;
}

struct VisibilityResult
{
  unsigned samples = 0, in_view = 0, clear = 0;
  float fraction = 0.F;
  bool visible = false;
};

inline VisibilityResult targetVisibility(
  const std::vector<Eigen::Vector3f> & scene, const std::vector<Eigen::Vector3f> & target,
  const Eigen::Isometry3f & base_from_camera, float fx, float fy, float cx, float cy,
  unsigned width, unsigned height)
{
  VisibilityResult result;
  constexpr float resolution = .005F;
  octomap::OcTree target_tree(resolution), scene_tree(resolution);
  for (const auto & p : target)
    if (p.allFinite()) target_tree.updateNode(octomap::point3d(p.x(), p.y(), p.z()), true, true);
  for (const auto & p : scene) {
    if (!p.allFinite()) continue;
    const octomap::point3d point(p.x(), p.y(), p.z());
    if (!target_tree.search(point)) scene_tree.updateNode(point, true, true);
  }
  scene_tree.updateInnerOccupancy();
  const auto camera_from_base = base_from_camera.inverse();
  const Eigen::Vector3f origin = base_from_camera.translation();
  const size_t count = std::min<size_t>(64, target.size());
  for (size_t i = 0; i < count; ++i) {
    const auto & p = target[count == 1 ? 0 : i * (target.size() - 1) / (count - 1)];
    if (!p.allFinite()) continue;
    ++result.samples;
    const Eigen::Vector3f optical = camera_from_base * p;
    if (optical.z() < .05F || optical.z() > 1.F) continue;
    const float u = fx * optical.x() / optical.z() + cx;
    const float v = fy * optical.y() / optical.z() + cy;
    if (u < 0.F || v < 0.F || u >= width || v >= height) continue;
    ++result.in_view;
    if (viewRayClear(scene_tree, origin, p, .010F)) ++result.clear;
  }
  if (result.samples) result.fraction = static_cast<float>(result.clear) / result.samples;
  result.visible = result.samples > 0 && result.fraction >= .6F;
  return result;
}

// ---------------------------------------------------------
// Voxel neighborhood filtering
// ---------------------------------------------------------
using VoxelCell = std::array<int, 3>;

struct VoxelCellHash
{
  std::size_t operator()(const VoxelCell & cell) const
  {
    std::size_t hash = 0;
    for (const auto value : cell)
      hash ^= std::hash<int>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

inline std::vector<bool> supportedVoxels(const std::vector<VoxelCell> & cells)
{
  const std::unordered_set<VoxelCell, VoxelCellHash> occupied(cells.begin(), cells.end());
  std::vector<bool> keep;
  keep.reserve(cells.size());
  for (const auto & cell : cells) {
    int neighbors = 0;
    for (int x = -1; x <= 1 && neighbors < 2; ++x)
      for (int y = -1; y <= 1 && neighbors < 2; ++y)
        for (int z = -1; z <= 1 && neighbors < 2; ++z) {
          if (x == 0 && y == 0 && z == 0) continue;
          neighbors += occupied.count({cell[0] + x, cell[1] + y, cell[2] + z});
        }
    keep.push_back(neighbors >= 2);
  }
  return keep;
}

}  // namespace vision_node

#endif  // VISION_NODE__ACTIVE_VISION_UTILS_HPP_

