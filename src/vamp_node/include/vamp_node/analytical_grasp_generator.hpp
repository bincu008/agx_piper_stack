#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <vector>

#include "vamp_node/grasp_proposal.hpp"

namespace vamp_node {

struct AnalyticalGraspParameters {
  int maximum_proposals = 300;
  float voxel_size_m = 0.005F;
  float roll_step_degrees = 15.0F;
  float normal_radius_m = 0.008F;
  int min_normal_neighbors = 10;
  float dedup_position_m = 0.010F;
  float dedup_angle_degrees = 10.0F;
  float min_contact_center_separation_m = 0.025F;
  int local_pca_searches = 50;
  int global_pca_searches = 50;
  int maximum_approach_variants_per_pair = 2;
  float minimum_opening_m = 0.005F;
  float maximum_opening_m = 0.100F;
  float contact_depth_m = 0.060F;
  float finger_length_m = 0.070F;
  float finger_width_m = 0.010F;
  float finger_height_m = 0.020F;
  float opening_clearance_m = 0.004F;
};

struct AnalyticalGraspDiagnostics {
  std::size_t isolated_points = 0;
  std::size_t downsampled_points = 0;
  std::size_t normal_points = 0;
  std::size_t contact_pairs = 0;
  std::size_t local_contact_pairs = 0;
  std::size_t global_contact_pairs = 0;
  std::size_t selected_contact_families = 0;
  std::size_t selected_local_families = 0;
  std::size_t selected_global_families = 0;
  std::size_t generated = 0;
  std::size_t deduplicated = 0;
};

// Deterministic single-view grasp proposal generator. It uses stable voxel
// ordering, PCA normals, normal-supported contact pairs, and PCA extent pairs
// as a conservative fallback for partial RGB-D observations.
class AnalyticalGraspGenerator {
 public:
  explicit AnalyticalGraspGenerator(AnalyticalGraspParameters parameters = {});

  std::vector<GraspProposal> generate(
      const std::vector<Eigen::Vector3f> &target_points,
      const Eigen::Vector3f &seed,
      const Eigen::Vector3f &camera_origin,
      AnalyticalGraspDiagnostics *diagnostics = nullptr) const;

 private:
  AnalyticalGraspParameters parameters_;
};

}  // namespace vamp_node
