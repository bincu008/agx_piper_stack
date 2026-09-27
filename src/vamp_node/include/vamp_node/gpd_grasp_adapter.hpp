#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "vamp_node/grasp_proposal.hpp"

namespace gpd {
class GraspDetector;
}

namespace vamp_node {

// Thin boundary around upstream GPD. ROS conversion, IK, collision checking,
// and visualization deliberately remain in VampPlannerServer.
class GpdGraspAdapter {
public:
  explicit GpdGraspAdapter(const std::string &config_file);
  ~GpdGraspAdapter();

  GpdGraspAdapter(const GpdGraspAdapter &) = delete;
  GpdGraspAdapter &operator=(const GpdGraspAdapter &) = delete;

  std::vector<GraspProposal>
  detect(const std::vector<Eigen::Vector3f> &scene_points,
         const std::vector<Eigen::Vector3f> &target_samples,
         const Eigen::Vector3f &camera_origin);

private:
  std::unique_ptr<gpd::GraspDetector> detector_;
  std::mutex detector_mutex_;
};

} // namespace vamp_node
