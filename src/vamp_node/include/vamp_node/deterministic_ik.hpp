#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace vamp_node
{

using PiperJointArray = std::array<float, 6>;

// A placement search may abandon a stationary seed and try the next branch.
// Continuing improvement, however small, resets this guard.
class IkStagnationGuard
{
public:
  explicit IkStagnationGuard(int limit) : limit_(limit) {}
  bool stalled(float error)
  {
    if (!std::isfinite(error)) return true;
    if (error < best_) { best_ = error; stale_ = 0; }
    else { ++stale_; }
    return limit_ > 0 && stale_ >= limit_;
  }
private:
  int limit_;
  int stale_ = 0;
  float best_ = std::numeric_limits<float>::infinity();
};


inline constexpr PiperJointArray kPiperNominalJointLower{
  -2.6179938F, 0.0F, -2.9670597F, -1.7453292F, -1.2217304F, -2.0943951F};
inline constexpr PiperJointArray kPiperNominalJointUpper{
   2.6179938F, 3.1415926F, 0.0F, 1.7453292F, 1.2217304F, 2.0943951F};
inline PiperJointArray kPiperJointLower = kPiperNominalJointLower;
inline PiperJointArray kPiperJointUpper = kPiperNominalJointUpper;

// Keep measured feedback unchanged; return a separate nominal planning entry.
// Callers must collision-check the complete measured-to-entry segment before
// using it, and may never issue the out-of-bounds measured state as a target.
inline std::optional<PiperJointArray> nominalFeedbackEntry(
  const std::vector<float> &feedback, float tolerance)
{
  if (feedback.size() != 6 || !std::isfinite(tolerance) || tolerance < 0.F ||
      tolerance > 2.F * 3.14159265358979323846F / 180.F + 1.e-6F) return std::nullopt;
  PiperJointArray entry{};
  for (size_t j = 0; j < 6; ++j) {
    if (!std::isfinite(feedback[j]) ||
        feedback[j] < kPiperJointLower[j]-tolerance-1.e-5F ||
        feedback[j] > kPiperJointUpper[j]+tolerance+1.e-5F) return std::nullopt;
    entry[j] = std::clamp(feedback[j], kPiperJointLower[j], kPiperJointUpper[j]);
  }
  return entry;
}

// Set once at node startup, before callbacks or planning threads run.
inline void configurePiperJointLimitTolerance(float radians)
{
  for (std::size_t joint = 0; joint < 6; ++joint) {
    kPiperJointLower[joint] = kPiperNominalJointLower[joint] - radians;
    kPiperJointUpper[joint] = kPiperNominalJointUpper[joint] + radians;
  }
}

inline float haltonValue(const int index, const int base)
{
  float result = 0.0F;
  float factor = 1.0F;
  int value = index;
  while (value > 0) {
    factor /= static_cast<float>(base);
    result += factor * static_cast<float>(value % base);
    value /= base;
  }
  return result;
}

inline PiperJointArray clampPiperJoints(PiperJointArray joints)
{
  for (std::size_t index = 0; index < joints.size(); ++index) {
    joints[index] = std::clamp(joints[index], kPiperJointLower[index], kPiperJointUpper[index]);
  }
  return joints;
}

inline std::vector<PiperJointArray> deterministicIkSeedBank(
  const PiperJointArray &current,
  const std::optional<PiperJointArray> &previous,
  const PiperJointArray &home,
  const int halton_seed_count)
{
  std::vector<PiperJointArray> result;
  const auto add_unique = [&result](PiperJointArray seed) {
    seed = clampPiperJoints(seed);
    for (const auto &existing : result) {
      float squared_distance = 0.0F;
      for (std::size_t joint = 0; joint < seed.size(); ++joint) {
        const float delta = seed[joint] - existing[joint];
        squared_distance += delta * delta;
      }
      if (squared_distance < 0.01F) {
        return;
      }
    }
    result.push_back(seed);
  };

  add_unique(current);
  if (previous.has_value()) {
    add_unique(*previous);
  }
  add_unique(home);

  PiperJointArray midpoint{};
  for (std::size_t joint = 0; joint < midpoint.size(); ++joint) {
    midpoint[joint] = 0.5F * (kPiperJointLower[joint] + kPiperJointUpper[joint]);
  }
  add_unique(midpoint);

  constexpr std::array<float, 4> offsets{{0.35F, -0.35F, 0.70F, -0.70F}};
  for (const float offset : offsets) {
    PiperJointArray candidate = current;
    candidate[0] += offset;
    candidate[1] -= offset;
    candidate[2] += offset;
    candidate[4] -= offset;
    candidate[5] += offset;
    add_unique(candidate);
  }

  constexpr std::array<int, 6> bases{{2, 3, 5, 7, 11, 13}};
  for (int sample = 1; sample <= std::max(0, halton_seed_count); ++sample) {
    PiperJointArray candidate{};
    for (std::size_t joint = 0; joint < candidate.size(); ++joint) {
      candidate[joint] = kPiperJointLower[joint] + haltonValue(sample, bases[joint]) *
        (kPiperJointUpper[joint] - kPiperJointLower[joint]);
    }
    add_unique(candidate);
  }
  return result;
}

inline float piperJointTravel(const PiperJointArray &candidate, const PiperJointArray &current)
{
  float squared_distance = 0.0F;
  for (std::size_t joint = 0; joint < candidate.size(); ++joint) {
    const float delta = candidate[joint] - current[joint];
    squared_distance += delta * delta;
  }
  return std::sqrt(squared_distance);
}

inline float piperJointLimitMargin(const PiperJointArray &joints)
{
  float margin = 1.0F;
  for (std::size_t joint = 0; joint < joints.size(); ++joint) {
    const float range = kPiperJointUpper[joint] - kPiperJointLower[joint];
    margin = std::min(margin, 2.0F * std::min(
      (joints[joint] - kPiperJointLower[joint]) / range,
      (kPiperJointUpper[joint] - joints[joint]) / range));
  }
  return margin;
}

inline bool deterministicIkSolutionPreferred(
  const PiperJointArray &candidate, const std::size_t candidate_seed_index,
  const PiperJointArray &best, const std::size_t best_seed_index,
  const PiperJointArray &current)
{
  constexpr float epsilon = 1.0e-6F;
  const float candidate_travel = piperJointTravel(candidate, current);
  const float best_travel = piperJointTravel(best, current);
  if (candidate_travel < best_travel - epsilon) {
    return true;
  }
  if (std::abs(candidate_travel - best_travel) > epsilon) {
    return false;
  }
  const float candidate_margin = piperJointLimitMargin(candidate);
  const float best_margin = piperJointLimitMargin(best);
  if (candidate_margin > best_margin + epsilon) {
    return true;
  }
  if (std::abs(candidate_margin - best_margin) > epsilon) {
    return false;
  }
  return candidate_seed_index < best_seed_index;
}

}  // namespace vamp_node
