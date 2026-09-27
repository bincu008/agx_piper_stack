#include "vamp_node/local_grasp_refiner.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>
#include <unordered_set>

namespace vamp_node
{
namespace
{
bool validPoint(const Eigen::Vector3f & point)
{
  return point.allFinite() && point.squaredNorm() > 1.0e-10F;
}

float percentile(std::vector<float> values, float fraction)
{
  const std::size_t index = static_cast<std::size_t>(std::clamp(
      fraction, 0.0F, 1.0F) * static_cast<float>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return values[index];
}

Eigen::Vector3f estimateNormal(
  const std::vector<Eigen::Vector3f> & cloud, int width, int height,
  int u, int v)
{
  if (u <= 0 || v <= 0 || u + 1 >= width || v + 1 >= height) {
    return Eigen::Vector3f::Zero();
  }
  const auto & left = cloud[static_cast<std::size_t>(v * width + u - 1)];
  const auto & right = cloud[static_cast<std::size_t>(v * width + u + 1)];
  const auto & up = cloud[static_cast<std::size_t>((v - 1) * width + u)];
  const auto & down = cloud[static_cast<std::size_t>((v + 1) * width + u)];
  if (!validPoint(left) || !validPoint(right) || !validPoint(up) || !validPoint(down)) {
    return Eigen::Vector3f::Zero();
  }
  Eigen::Vector3f normal = (right - left).cross(down - up);
  if (!normal.allFinite() || normal.squaredNorm() < 1.0e-10F) {
    return Eigen::Vector3f::Zero();
  }
  return normal.normalized();
}

LocalGraspRefinement fitMinimumOpening(
  const LocalGraspRefinerParameters & parameters,
  const std::vector<Eigen::Vector3f> & region,
  const Eigen::Vector3f & corrected_seed,
  const Eigen::Vector3f & camera_origin,
  LocalGraspRefinement result)
{
  result.seed = corrected_seed;
  result.region_points = region.size();
  if (region.size() < static_cast<std::size_t>(parameters.minimum_region_points)) {
    result.failure_reason = "INSUFFICIENT_CONNECTED_SURFACE";
    return result;
  }
  Eigen::Vector3f approach = corrected_seed - camera_origin;
  if (!approach.allFinite() || approach.squaredNorm() < 1.0e-8F) {
    result.failure_reason = "DEGENERATE_CAMERA_APPROACH";
    return result;
  }
  approach.normalize();
  Eigen::Vector3f basis_x = approach.cross(Eigen::Vector3f::UnitZ());
  if (basis_x.squaredNorm() < 1.0e-6F) {
    basis_x = approach.cross(Eigen::Vector3f::UnitY());
  }
  basis_x.normalize();
  const Eigen::Vector3f basis_y = approach.cross(basis_x).normalized();

  float best_width = std::numeric_limits<float>::infinity();
  float minimum_observed_width = std::numeric_limits<float>::infinity();
  float best_depth = 0.0F;
  float best_center_offset = 0.0F;
  Eigen::Vector3f best_closing = basis_x;
  constexpr float radians_per_degree = 3.14159265358979323846F / 180.0F;
  for (float angle = 0.0F; angle < 180.0F;
    angle += parameters.angle_step_degrees)
  {
    const float radians = angle * radians_per_degree;
    const Eigen::Vector3f closing =
      std::cos(radians) * basis_x + std::sin(radians) * basis_y;
    const Eigen::Vector3f longitudinal = approach.cross(closing).normalized();
    std::vector<float> lateral;
    std::vector<float> along;
    lateral.reserve(region.size());
    along.reserve(region.size());
    for (const auto & point : region) {
      const Eigen::Vector3f delta = point - corrected_seed;
      const float length = delta.dot(longitudinal);
      if (std::abs(length) <= parameters.cross_section_half_length_m) {
        lateral.push_back(delta.dot(closing));
        along.push_back(length);
      }
    }
    if (lateral.size() < static_cast<std::size_t>(parameters.minimum_region_points)) {
      continue;
    }
    const float minimum = percentile(lateral, 0.03F);
    const float maximum = percentile(lateral, 0.97F);
    const float width_m = maximum - minimum;
    const float center_offset = 0.5F * (minimum + maximum);
    const float usable_depth = percentile(along, 0.95F) - percentile(along, 0.05F);
    const int left_contacts = static_cast<int>(std::count_if(
        lateral.begin(), lateral.end(), [minimum, &parameters](float value) {
          return value <= minimum + parameters.contact_band_m;
        }));
    const int right_contacts = static_cast<int>(std::count_if(
        lateral.begin(), lateral.end(), [maximum, &parameters](float value) {
          return value >= maximum - parameters.contact_band_m;
        }));
    if (usable_depth >= parameters.minimum_usable_depth_m &&
      left_contacts >= parameters.minimum_contact_points &&
      right_contacts >= parameters.minimum_contact_points)
    {
      minimum_observed_width = std::min(minimum_observed_width, width_m);
    }
    if (minimum >= -0.002F || maximum <= 0.002F ||
      usable_depth < parameters.minimum_usable_depth_m ||
      left_contacts < parameters.minimum_contact_points ||
      right_contacts < parameters.minimum_contact_points ||
      width_m < parameters.minimum_width_m ||
      width_m > parameters.maximum_width_m ||
      std::abs(center_offset) > parameters.maximum_center_shift_m)
    {
      continue;
    }
    if (width_m < best_width - 1.0e-5F ||
      (std::abs(width_m - best_width) <= 1.0e-5F && usable_depth > best_depth))
    {
      best_width = width_m;
      best_depth = usable_depth;
      best_center_offset = center_offset;
      best_closing = closing.normalized();
    }
  }
  if (minimum_observed_width < parameters.minimum_width_m) {
    result.failure_reason = "CROSS_SECTION_TOO_THIN";
    return result;
  }
  if (!std::isfinite(best_width)) {
    result.failure_reason = "NO_VALID_CROSS_SECTION";
    return result;
  }
  result.valid = true;
  result.seed = corrected_seed + best_center_offset * best_closing;
  result.closing_axis = best_closing;
  result.approach_axis = approach;
  result.width_m = best_width;
  result.usable_depth_m = best_depth;
  return result;
}

struct RegionVoxelHash
{
  std::size_t operator()(const std::tuple<int, int, int> & key) const noexcept
  {
    const auto x = static_cast<std::size_t>(std::get<0>(key));
    const auto y = static_cast<std::size_t>(std::get<1>(key));
    const auto z = static_cast<std::size_t>(std::get<2>(key));
    return x * 73856093U ^ y * 19349663U ^ z * 83492791U;
  }
};
}  // namespace

LocalGraspRefiner::LocalGraspRefiner(LocalGraspRefinerParameters parameters)
: parameters_(parameters)
{
}

bool LocalGraspRefiner::projectBasePoint(
  const Eigen::Vector3f & point_base,
  const Eigen::Isometry3f & base_from_camera,
  const CameraIntrinsics & intrinsics, int width, int height,
  int & u, int & v)
{
  const Eigen::Vector3f point_camera = base_from_camera.inverse() * point_base;
  if (!point_camera.allFinite() || point_camera.z() <= 1.0e-4F ||
    intrinsics.fx <= 0.0F || intrinsics.fy <= 0.0F)
  {
    return false;
  }
  u = static_cast<int>(std::lround(
      intrinsics.fx * point_camera.x() / point_camera.z() + intrinsics.cx));
  v = static_cast<int>(std::lround(
      intrinsics.fy * point_camera.y() / point_camera.z() + intrinsics.cy));
  return u >= 0 && v >= 0 && u < width && v < height;
}

std::vector<Eigen::Vector3f> LocalGraspRefiner::extractContactRegion(
  const std::vector<Eigen::Vector3f> & points,
  const Eigen::Vector3f & center, float radius_m)
{
  std::vector<Eigen::Vector3f> region;
  if (!center.allFinite() || radius_m <= 0.0F) return region;
  const float radius_sq = radius_m * radius_m;
  region.reserve(points.size());
  for (const auto & point : points) {
    if (validPoint(point) && (point - center).squaredNorm() <= radius_sq) {
      region.push_back(point);
    }
  }
  return region;
}

std::vector<Eigen::Vector3f> LocalGraspRefiner::mergeContactRegions(
  const std::vector<Eigen::Vector3f> & first,
  const std::vector<Eigen::Vector3f> & second, float duplicate_voxel_m)
{
  std::vector<Eigen::Vector3f> merged;
  if (duplicate_voxel_m <= 0.0F) return merged;
  merged.reserve(first.size() + second.size());
  std::unordered_set<std::tuple<int, int, int>, RegionVoxelHash> occupied;
  const auto append = [&](const std::vector<Eigen::Vector3f> & points) {
      for (const auto & point : points) {
        if (!validPoint(point)) continue;
        const auto key = std::make_tuple(
          static_cast<int>(std::lround(point.x() / duplicate_voxel_m)),
          static_cast<int>(std::lround(point.y() / duplicate_voxel_m)),
          static_cast<int>(std::lround(point.z() / duplicate_voxel_m)));
        if (occupied.insert(key).second) merged.push_back(point);
      }
    };
  append(first);
  append(second);
  return merged;
}

LocalGraspRefinement LocalGraspRefiner::refinePointRegion(
  const std::vector<Eigen::Vector3f> & region_base,
  const Eigen::Vector3f & seed_base,
  const Eigen::Vector3f & camera_origin_base) const
{
  LocalGraspRefinement result;
  float best_distance = std::numeric_limits<float>::infinity();
  Eigen::Vector3f corrected_seed = Eigen::Vector3f::Zero();
  for (const auto & point : region_base) {
    if (!validPoint(point)) continue;
    const float distance = (point - seed_base).norm();
    if (distance < best_distance) {
      best_distance = distance;
      corrected_seed = point;
    }
  }
  result.snap_distance_m = best_distance;
  if (!std::isfinite(best_distance)) {
    result.failure_reason = "EMPTY_MERGED_CONTACT_REGION";
    return result;
  }
  if (best_distance > parameters_.snap_max_distance_m) {
    result.failure_reason = "MERGED_CONTACT_REGION_TOO_FAR";
    return result;
  }
  return fitMinimumOpening(
    parameters_, region_base, corrected_seed, camera_origin_base, result);
}

LocalGraspRefinement LocalGraspRefiner::refine(
  const std::vector<Eigen::Vector3f> & cloud, int width, int height,
  const Eigen::Vector3f & seed_base, int projected_u, int projected_v,
  const Eigen::Vector3f & camera_origin) const
{
  LocalGraspRefinement result;
  // Preserve the manual-X reprojection for diagnostics even when no measured
  // surface can be snapped. Previously an early return exposed [-1, -1],
  // hiding where the old-view HGGD seed landed in the new image.
  result.seed_u = projected_u;
  result.seed_v = projected_v;
  if (width <= 2 || height <= 2 ||
    cloud.size() != static_cast<std::size_t>(width * height))
  {
    result.failure_reason = "INVALID_ORGANIZED_CLOUD";
    return result;
  }

  float best_distance = std::numeric_limits<float>::infinity();
  int seed_u = -1;
  int seed_v = -1;
  for (int dv = -parameters_.snap_radius_px; dv <= parameters_.snap_radius_px; ++dv) {
    for (int du = -parameters_.snap_radius_px; du <= parameters_.snap_radius_px; ++du) {
      const int u = projected_u + du;
      const int v = projected_v + dv;
      if (u < 0 || v < 0 || u >= width || v >= height ||
        du * du + dv * dv > parameters_.snap_radius_px * parameters_.snap_radius_px)
      {
        continue;
      }
      const auto & point = cloud[static_cast<std::size_t>(v * width + u)];
      if (!validPoint(point)) {
        continue;
      }
      const float distance = (point - seed_base).norm();
      if (distance < best_distance) {
        best_distance = distance;
        seed_u = u;
        seed_v = v;
      }
    }
  }
  if (seed_u < 0 || best_distance > parameters_.snap_max_distance_m) {
    // The caller has already removed everything outside the reobserved,
    // grounded target support. A pixel-only window is therefore unnecessarily
    // brittle when the wrist viewpoint changes or the hand-eye projection is a
    // few pixels off. Fall back to the closest target-supported 3D sample; the
    // same metric distance gate still prevents snapping to unrelated geometry.
    best_distance = std::numeric_limits<float>::infinity();
    seed_u = -1;
    seed_v = -1;
    for (int v = 0; v < height; ++v) {
      for (int u = 0; u < width; ++u) {
        const auto & point = cloud[static_cast<std::size_t>(v * width + u)];
        if (!validPoint(point)) {
          continue;
        }
        const float distance = (point - seed_base).norm();
        if (distance < best_distance) {
          best_distance = distance;
          seed_u = u;
          seed_v = v;
        }
      }
    }
    result.used_global_target_search = true;
  }
  result.snap_distance_m = best_distance;
  result.snapped_u = seed_u;
  result.snapped_v = seed_v;
  if (seed_u < 0) {
    result.failure_reason = "NO_REOBSERVED_TARGET_DEPTH";
    return result;
  }
  if (best_distance > parameters_.snap_max_distance_m) {
    result.failure_reason = "NEAREST_VISIBLE_TARGET_TOO_FAR";
    return result;
  }

  const Eigen::Vector3f corrected_seed =
    cloud[static_cast<std::size_t>(seed_v * width + seed_u)];
  const float roi_radius_sq = parameters_.roi_radius_m * parameters_.roi_radius_m;
  std::vector<unsigned char> accepted(cloud.size(), 0);
  std::queue<int> frontier;
  const int seed_index = seed_v * width + seed_u;
  accepted[static_cast<std::size_t>(seed_index)] = 1;
  frontier.push(seed_index);
  std::vector<Eigen::Vector3f> region;
  region.reserve(4096);
  const int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  while (!frontier.empty()) {
    const int index = frontier.front();
    frontier.pop();
    const int u = index % width;
    const int v = index / width;
    const auto & current = cloud[static_cast<std::size_t>(index)];
    region.push_back(current);
    const Eigen::Vector3f current_normal = estimateNormal(cloud, width, height, u, v);
    const float current_depth = (current - camera_origin).norm();
    for (const auto & offset : offsets) {
      const int next_u = u + offset[0];
      const int next_v = v + offset[1];
      if (next_u < 0 || next_v < 0 || next_u >= width || next_v >= height) {
        continue;
      }
      const int next_index = next_v * width + next_u;
      if (accepted[static_cast<std::size_t>(next_index)]) {
        continue;
      }
      const auto & next = cloud[static_cast<std::size_t>(next_index)];
      if (!validPoint(next) || (next - corrected_seed).squaredNorm() > roi_radius_sq ||
        (next - current).norm() > parameters_.neighbor_distance_m ||
        std::abs((next - camera_origin).norm() - current_depth) >
        parameters_.neighbor_depth_change_m)
      {
        continue;
      }
      const Eigen::Vector3f next_normal =
        estimateNormal(cloud, width, height, next_u, next_v);
      if (current_normal.squaredNorm() > 0.5F && next_normal.squaredNorm() > 0.5F &&
        std::abs(current_normal.dot(next_normal)) < parameters_.minimum_normal_cosine)
      {
        continue;
      }
      accepted[static_cast<std::size_t>(next_index)] = 1;
      frontier.push(next_index);
    }
  }

  result.seed_u = seed_u;
  result.seed_v = seed_v;
  result.snapped_u = seed_u;
  result.snapped_v = seed_v;
  result.snap_distance_m = best_distance;
  return fitMinimumOpening(
    parameters_, region, corrected_seed, camera_origin, result);
}

}  // namespace vamp_node
