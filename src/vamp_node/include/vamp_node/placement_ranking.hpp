#pragma once

#include <cmath>

namespace vamp_node
{
// Feasibility is checked before ranking. Keep the packing objective primary;
// pose error and joint travel break ties between equally compact placements.
inline bool preferPlacement(float score, float cost, float best_score, float best_cost)
{
  if (!std::isfinite(score) || !std::isfinite(cost)) return false;
  constexpr float score_tolerance = 1.0e-9F;
  return score > best_score + score_tolerance ||
    (std::abs(score - best_score) <= score_tolerance && cost < best_cost);
}
}  // namespace vamp_node
