#include "vamp_node/gpd_grasp_adapter.hpp"

#include <stdexcept>

#include <gpd/grasp_detector.h>
#include <gpd/util/cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vamp_node {

GpdGraspAdapter::GpdGraspAdapter(const std::string &config_file)
    : detector_(std::make_unique<gpd::GraspDetector>(config_file)) {}

GpdGraspAdapter::~GpdGraspAdapter() = default;

std::vector<GraspProposal>
GpdGraspAdapter::detect(const std::vector<Eigen::Vector3f> &scene_points,
                        const std::vector<Eigen::Vector3f> &target_samples,
                        const Eigen::Vector3f &camera_origin) {
  if (scene_points.empty()) {
    throw std::invalid_argument("GPD scene point cloud is empty");
  }
  if (target_samples.empty()) {
    throw std::invalid_argument("GPD target sample set is empty");
  }

  auto pcl_cloud = std::make_shared<gpd::util::PointCloudRGB>();
  pcl_cloud->reserve(scene_points.size());
  for (const auto &point : scene_points) {
    pcl::PointXYZRGBA pcl_point;
    pcl_point.x = point.x();
    pcl_point.y = point.y();
    pcl_point.z = point.z();
    pcl_point.r = pcl_point.g = pcl_point.b = 128;
    pcl_point.a = 255;
    pcl_cloud->push_back(pcl_point);
  }

  Eigen::MatrixXi camera_source = Eigen::MatrixXi::Ones(1, scene_points.size());
  Eigen::Matrix3Xd view_points(3, 1);
  view_points.col(0) = camera_origin.cast<double>();
  gpd::util::Cloud cloud(pcl_cloud, camera_source, view_points);

  Eigen::Matrix3Xd samples(3, target_samples.size());
  for (size_t index = 0; index < target_samples.size(); ++index) {
    samples.col(index) = target_samples[index].cast<double>();
  }
  cloud.setSamples(samples);

  std::lock_guard<std::mutex> lock(detector_mutex_);
  detector_->preprocessPointCloud(cloud);
  auto hands = detector_->detectGrasps(cloud);

  std::vector<GraspProposal> proposals;
  proposals.reserve(hands.size());
  for (const auto &hand : hands) {
    GraspProposal proposal;
    proposal.sample = hand->getSample().cast<float>();
    proposal.hand_base = hand->getPosition().cast<float>();
    proposal.approach = hand->getApproach().cast<float>().normalized();
    proposal.closing = hand->getBinormal().cast<float>().normalized();
    proposal.height_axis = hand->getAxis().cast<float>().normalized();
    proposal.opening_m = static_cast<float>(hand->getGraspWidth());
    proposal.quality = static_cast<float>(hand->getScore());
    proposal.half_antipodal = hand->isHalfAntipodal();
    proposal.full_antipodal = hand->isFullAntipodal();
    proposal.source = GraspProposalSource::kGpd;
    proposal.analytical_origin = AnalyticalProposalOrigin::kLegacyGpd;
    if (proposal.sample.allFinite() && proposal.hand_base.allFinite() &&
        proposal.approach.allFinite() &&
        proposal.closing.allFinite() && proposal.height_axis.allFinite()) {
      proposals.push_back(proposal);
    }
  }
  return proposals;
}

} // namespace vamp_node
