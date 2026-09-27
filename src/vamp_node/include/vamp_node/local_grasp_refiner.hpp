#pragma once

#include <Eigen/Geometry>

#include <cstddef>
#include <string>
#include <vector>

namespace vamp_node
{

struct CameraIntrinsics
{
  float fx = 0.0F;
  float fy = 0.0F;
  float cx = 0.0F;
  float cy = 0.0F;
};

struct LocalGraspRefinerParameters
{
  float roi_radius_m = 0.10F;
  int snap_radius_px = 18;
  float snap_max_distance_m = 0.05F;
  float neighbor_distance_m = 0.006F;
  float neighbor_depth_change_m = 0.006F;
  float minimum_normal_cosine = 0.90F;
  int minimum_region_points = 40;
  float angle_step_degrees = 5.0F;
  float minimum_width_m = 0.008F;
  float maximum_width_m = 0.10F;
  float minimum_usable_depth_m = 0.01F;
  // Piper's jaw contact pad is 20 mm high. Measure a local 24 mm strip,
  // rather than allowing a 100 mm strip to absorb a handle loop/cup body.
  float cross_section_half_length_m = 0.012F;
  float maximum_center_shift_m = 0.03F;
  float contact_band_m = 0.006F;
  int minimum_contact_points = 3;
};

struct LocalGraspRefinement
{
  bool valid = false;
  std::string failure_reason;
  int seed_u = -1;
  int seed_v = -1;
  int snapped_u = -1;
  int snapped_v = -1;
  bool used_global_target_search = false;
  float snap_distance_m = 0.0F;
  Eigen::Vector3f seed = Eigen::Vector3f::Zero();
  Eigen::Vector3f closing_axis = Eigen::Vector3f::UnitX();
  Eigen::Vector3f approach_axis = Eigen::Vector3f::UnitZ();
  float width_m = 0.0F;
  float usable_depth_m = 0.0F;
  std::size_t region_points = 0;
};

class LocalGraspRefiner
{
public:
  explicit LocalGraspRefiner(LocalGraspRefinerParameters parameters = {});

  static bool projectBasePoint(
    const Eigen::Vector3f & point_base,
    const Eigen::Isometry3f & base_from_camera,
    const CameraIntrinsics & intrinsics, int width, int height,
    int & u, int & v);

  // Reusable base_link-space extraction for local manipulation geometry.
  static std::vector<Eigen::Vector3f> extractContactRegion(
    const std::vector<Eigen::Vector3f> & points,
    const Eigen::Vector3f & center, float radius_m);

  static std::vector<Eigen::Vector3f> mergeContactRegions(
    const std::vector<Eigen::Vector3f> & first,
    const std::vector<Eigen::Vector3f> & second,
    float duplicate_voxel_m = 0.001F);

  LocalGraspRefinement refine(
    const std::vector<Eigen::Vector3f> & organized_cloud_base,
    int width, int height, const Eigen::Vector3f & seed_base,
    int projected_u, int projected_v,
    const Eigen::Vector3f & camera_origin_base) const;

  LocalGraspRefinement refinePointRegion(
    const std::vector<Eigen::Vector3f> & region_base,
    const Eigen::Vector3f & seed_base,
    const Eigen::Vector3f & camera_origin_base) const;

private:
  LocalGraspRefinerParameters parameters_;
};

}  // namespace vamp_node
