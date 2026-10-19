#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace vamp_node {
// APF guidance and bounded departure contact accounting only. A successful
// endpoint must additionally pass the original full robot/payload checker.
class PayloadEscapeField {
public:
    using Cell = std::tuple<int, int, int>;
    struct Probe {
        Eigen::Vector3f center;
        float radius;
        std::function<bool(const Eigen::Vector3f &)> contains;
    };
    struct State {
        Eigen::Vector3f force = Eigen::Vector3f::Zero();
        float energy = 0.F;
        float penetration = 0.F;
        std::set<Cell> contacts;
    };
    explicit PayloadEscapeField(const std::vector<Eigen::Vector3f> &points) {
        for (const auto &p : points) {
            // Keep exact immutable points for contact bounds. Five-millimetre
            // cells count contact extent; they must not erase collision samples.
            if (!p.allFinite()) continue;
            cells_[cell(p, .02F)].push_back(p);
        }
    }
    State evaluate(const std::vector<Probe> &probes) const {
        State state;
        Eigen::Vector3f separating_force=Eigen::Vector3f::Zero();
        for (const auto &probe : probes) {
            const float reach = probe.radius + .015F;
            const auto low = cell(probe.center-Eigen::Vector3f::Constant(reach), .02F);
            const auto high = cell(probe.center+Eigen::Vector3f::Constant(reach), .02F);
            Eigen::Vector3f force = Eigen::Vector3f::Zero();
            size_t samples = 0;
            for (int x=std::get<0>(low); x<=std::get<0>(high); ++x)
                for (int y=std::get<1>(low); y<=std::get<1>(high); ++y)
                    for (int z=std::get<2>(low); z<=std::get<2>(high); ++z) {
                        const auto found=cells_.find({x,y,z});
                        if (found==cells_.end()) continue;
                        for (const auto &p : found->second) {
                            const Eigen::Vector3f delta=probe.center-p;
                            const float distance=delta.norm();
                            if (distance>=reach || (probe.contains && !probe.contains(p))) continue;
                            ++samples;
                            const Eigen::Vector3f direction=distance>.00001F ?
                                (delta/distance).eval() : Eigen::Vector3f::UnitZ().eval();
                            force+=direction*std::min(2.F,(reach-distance)/.015F);
                            const float depth=std::max(0.F,probe.radius-distance);
                            if (depth>0) {
                                separating_force+=depth*direction;
                                state.energy+=depth*depth;
                                state.penetration=std::max(state.penetration,depth);
                                state.contacts.insert(cell(p,.005F));
                            }
                        }
                    }
            if (samples) state.force+=force/static_cast<float>(samples);
        }
        // Prioritize the gradient of actual overlap over broad nearby surfaces.
        // Otherwise a dense, clear table can steer toward the few real contacts.
        if (separating_force.norm()>1.e-8F) state.force=separating_force;
        return state;
    }
    // Bounds apply to a complete escape attempt, never reset per waypoint.
    // A small discretization allowance is permitted, not cumulative worsening.
    static bool admissible(const State &initial, const State &best,
        const State &next, const std::set<Cell> &segment_contacts) {
        auto combined=segment_contacts;
        combined.insert(next.contacts.begin(),next.contacts.end());
        return initial.penetration<=.010F && next.penetration<=.010F &&
            next.penetration<=initial.penetration+.0005F &&
            next.energy<=best.energy*1.25F+1.e-7F &&
            next.energy<=initial.energy*1.25F+1.e-7F &&
            next.contacts.size()<=initial.contacts.size()+4 && combined.size()<=128;
    }
private:
    struct Hash {
        size_t operator()(const Cell &v) const {
            return static_cast<size_t>(std::get<0>(v))*73856093U ^
                static_cast<size_t>(std::get<1>(v))*19349663U ^
                static_cast<size_t>(std::get<2>(v))*83492791U;
        }
    };
    static Cell cell(const Eigen::Vector3f &p, float resolution) {
        return {static_cast<int>(std::floor(p.x()/resolution)),
                static_cast<int>(std::floor(p.y()/resolution)),
                static_cast<int>(std::floor(p.z()/resolution))};
    }
    std::unordered_map<Cell,std::vector<Eigen::Vector3f>,Hash> cells_;
};
}  // namespace vamp_node
