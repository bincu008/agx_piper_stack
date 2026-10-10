#pragma once

#include <Eigen/Core>

namespace vamp_node {

enum class GraspProposalSource {
  kKnownCloud,
  kHggd,
};

enum class CloudProposalOrigin {
  kLocalPca,
  kGlobalPca,
};

enum class LocalGraspSkill {
  kUnknown,
  kPinch,
  kPowerGrasp,
};

// Backend-neutral proposal consumed by the existing IK, collision, ranking,
// and RViz stages. `quality` is a geometry score for analytical proposals and
// the unmodified classifier score for the legacy GPD backend.
struct GraspProposal {
  Eigen::Vector3f sample = Eigen::Vector3f::Zero();
  Eigen::Vector3f hand_base = Eigen::Vector3f::Zero();
  Eigen::Vector3f approach = Eigen::Vector3f::UnitZ();
  Eigen::Vector3f closing = Eigen::Vector3f::UnitX();
  Eigen::Vector3f height_axis = Eigen::Vector3f::UnitY();
  Eigen::Vector3f contact_left = Eigen::Vector3f::Zero();
  Eigen::Vector3f contact_right = Eigen::Vector3f::Zero();
  Eigen::Vector3f local_dominant_axis = Eigen::Vector3f::Zero();
  float opening_m = 0.0F;
  float quality = 0.0F;
  bool has_contacts = false;
  // True when contacts are from a bounded fitted cross-section rather than
  // two directly observed surface samples in a partial RGB-D view.
  bool model_supported_contacts = false;
  // True when the opening was measured between the two nearest boundaries of
  // one contiguous thin silhouette. This prevents later 1-D projection from
  // expanding a handle/rim grasp across an adjacent body or aperture.
  bool silhouette_supported_contacts = false;
  // True for a broad, locally planar top-surface hypothesis. These poses use
  // measured opposing wall contacts and proceed directly to full-pose IK;
  // they must not be collapsed back into the side-contact-axis solver.
  bool top_surface_grasp = false;
  // A broad grasp across opposing object walls.  This tag is used only for
  // topology-aware ranking; it does not bypass IK or collision validation.
  bool side_surface_grasp = false;
  // True when local geometry identified an upper-rim cross-section. Rim
  // contacts are legitimate endpoints of the visible material and must not
  // be rejected by the handle junction/endpoint guard merely because the
  // user requested the whole object instead of explicitly naming its rim.
  bool rim_grasp = false;
  bool half_antipodal = false;
  bool full_antipodal = false;
  bool has_local_dominant_axis = false;
  // Selected from measured local geometry, never from an object-class label.
  LocalGraspSkill grasp_skill = LocalGraspSkill::kUnknown;
  GraspProposalSource source = GraspProposalSource::kKnownCloud;
  CloudProposalOrigin cloud_origin = CloudProposalOrigin::kLocalPca;
  std::size_t family_index = 0;
  std::size_t variant_index = 0;
};

}  // namespace vamp_node
