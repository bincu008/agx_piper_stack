#pragma once

#include <cmath>

#include <Eigen/Core>

namespace vamp_node {

inline bool gpdPosesAreDuplicates(
    const Eigen::Vector3f &position_a, const Eigen::Vector3f &approach_a,
    const Eigen::Vector3f &closing_a, const Eigen::Vector3f &position_b,
    const Eigen::Vector3f &approach_b, const Eigen::Vector3f &closing_b,
    float position_threshold_m, float approach_threshold_degrees,
    float closing_threshold_degrees) {
  constexpr float kPi = 3.14159265358979323846F;
  const float approach_cosine = std::cos(approach_threshold_degrees * kPi / 180.0F);
  const float closing_cosine = std::cos(closing_threshold_degrees * kPi / 180.0F);
  return (position_a - position_b).norm() < position_threshold_m &&
    approach_a.dot(approach_b) > approach_cosine &&
    std::abs(closing_a.dot(closing_b)) > closing_cosine;
}

}  // namespace vamp_node
