#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_map>

namespace vamp_node {
// Frozen initial finger contact region and overlap energy. Re-evaluating a
// waypoint resets only its accumulator, never the segment's original allowance.
// Independent of the attached-object sparse-point budget.
class WithdrawalContacts {
public:
    bool seed(std::size_t point, float penetration) {
        if (!std::isfinite(penetration) || penetration < -.002F || penetration > .030F)
            return false;
        initial_[point] = std::max(initial_[point], std::max(0.F, penetration));
        maximum_ = std::max(maximum_, penetration);
        return true;
    }
    void begin() { current_.clear(); }
    bool permits(std::size_t point, float penetration, float travel, bool terminal) {
        if (terminal || !std::isfinite(travel) || travel < 0.F || travel > .030F ||
            !std::isfinite(penetration) || penetration < 0.F ||
            initial_.find(point) == initial_.end() || penetration > maximum_+.002F)
            return false;
        current_[point] = std::max(current_[point], penetration);
        return true;
    }
    bool separating(float travel) const {
        if (!std::isfinite(travel) || travel < 0.F) return false;
        float initial_energy=0.F, current_energy=0.F;
        for (const auto &sample:initial_) initial_energy+=sample.second*sample.second;
        for (const auto &sample:current_) current_energy+=sample.second*sample.second;
        // Tangential motion along a released rim need not shed overlap at a
        // prescribed rate. Never grow the original budget; permits() bounds
        // its region/depth/distance and requires a completely clear endpoint.
        return current_energy<=initial_energy+std::min(1.e-6F,initial_energy*.01F);
    }
    float initialEnergy() const { return energy(initial_); }
    float currentEnergy() const { return energy(current_); }
    std::size_t size() const { return initial_.size(); }
private:
    static float energy(const std::unordered_map<std::size_t,float> &points) {
        float result=0.F;
        for (const auto &sample:points) result+=sample.second*sample.second;
        return result;
    }
    std::unordered_map<std::size_t, float> initial_, current_;
    float maximum_=0.F;
};
}  // namespace vamp_node
