#ifndef VAMP_NODE__PLANAR_BODY_MASK_HPP_
#define VAMP_NODE__PLANAR_BODY_MASK_HPP_
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>
#include <vector>
namespace vamp_node {
// A dish's connected image mask can include its lower stand. Retain the thin
// surface and its raised rim using a measured plane, without an assumed diameter.
inline std::vector<bool> planar_body_mask(const std::vector<Eigen::Vector3f> & points)
{
  std::vector<bool> keep(points.size(), true);
  if (points.size() < 80) return keep;
  std::mt19937 generator(17);
  std::uniform_int_distribution<size_t> index(0, points.size()-1);
  Eigen::Vector3f best_normal = Eigen::Vector3f::UnitZ();
  float best_offset = 0.; size_t best_count = 0;
  for (int trial = 0; trial < 96; ++trial) {
    const auto & a = points[index(generator)];
    Eigen::Vector3f normal = (points[index(generator)]-a).cross(points[index(generator)]-a);
    if (normal.norm() < 1e-8f) continue;
    normal.normalize(); if (normal.z() < 0.) normal = -normal;
    if (normal.z() < .9f) continue;
    const float offset = -normal.dot(a);
    size_t count = 0;
    for (const auto & p : points) if (std::abs(normal.dot(p)+offset) <= .004f) ++count;
    if (count > best_count) { best_count = count; best_normal = normal; best_offset = offset; }
  }
  if (best_count < points.size()*.4) return keep;
  for (size_t i = 0; i < points.size(); ++i) {
    const float height = best_normal.dot(points[i])+best_offset;
    keep[i] = height >= -.004f && height <= .022f;
  }
  return keep;
}
}  // namespace vamp_node
#endif
