#include "vision_node/perception_translation.hpp"
#include <stdexcept>

int main()
{
  Eigen::Matrix4f manual = Eigen::Matrix4f::Identity();
  manual.block<3, 3>(0, 0) = Eigen::AngleAxisf(1.2f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  manual.block<3, 1>(0, 3) = Eigen::Vector3f(.3f, .1f, .2f);
  const Eigen::Vector3f offset(-.005f, .005f, .005f);
  const auto corrected = vision_node::translatedPerceptionTransform(manual, offset);
  for (const auto & point : {Eigen::Vector4f(.1f, .2f, .4f, 1.f),
                            Eigen::Vector4f(.2f, .3f, .5f, 1.f)}) {
    const Eigen::Vector4f delta = corrected * point - manual * point;
    if (!delta.head<3>().isApprox(offset, 1e-5f)) {
      throw std::runtime_error("Translation must be identical in base axes for all points");
    }
  }
  if (!corrected.block<3, 3>(0, 0).isApprox(manual.block<3, 3>(0, 0))) {
    throw std::runtime_error("Rotation changed");
  }
  if (!vision_node::translatedPerceptionTransform(manual, Eigen::Vector3f::Zero()).isApprox(manual)) {
    throw std::runtime_error("Zero offset changed manual calibration");
  }
}
