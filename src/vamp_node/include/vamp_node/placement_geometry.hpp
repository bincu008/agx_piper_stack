#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <Eigen/Geometry>

namespace vamp_node {
// Account for the narrow-phase point neighborhood when accepting source-table
// depth variation. This applies only to labeled bottom/support contact, never
// to other objects or to transfer/final placement.
inline float initialLiftSupportBand(float departure_allowance) {
    if (departure_allowance <= 0.0F) return 0.004F;
    // Fit uncertainty, point neighborhood and the existing bottom-contact
    // margin are separate distances; this does not enlarge penetration.
    return 0.004F + 0.003F + 0.003F +
        std::clamp(departure_allowance, 0.0F, 0.002F);
}
inline float initialLiftContactTravel(float departure_allowance) {
    if (departure_allowance <= 0.0F) return 0.006F;
    return initialLiftSupportBand(departure_allowance) + 0.003F;
}
// Match the scene's 3 mm point-collision neighborhood at source edges. The
// existing bottom-only contact and 1 mm penetration limits still apply.
inline bool initialLiftSupportSample(const Eigen::Vector3f &point,
    const Eigen::Vector3f &minimum, const Eigen::Vector3f &maximum, float bottom_z) {
    return std::abs(point.z()-bottom_z)<=0.003F &&
        point.x()>=minimum.x()-0.003F && point.x()<=maximum.x()+0.003F &&
        point.y()>=minimum.y()-0.003F && point.y()<=maximum.y()+0.003F;
}

inline std::vector<float> placementHoverDistances(float maximum, bool layout) {
    if (!std::isfinite(maximum) || maximum <= 0.0F) return {};
    if (!layout) return {maximum};
    const float minimum = std::min(maximum, 0.02F);
    std::vector<float> distances;
    for (float distance = maximum; distance > minimum + 1e-6F; distance -= 0.02F) {
        distances.push_back(distance);
    }
    distances.push_back(minimum);
    return distances;
}

inline bool padNormalsAligned(const Eigen::Vector3f &left,
    const Eigen::Vector3f &right, const Eigen::Vector3f &closing,
    float cosine) {
    if (!left.allFinite() || !right.allFinite() || !closing.allFinite() ||
        left.norm() < 1e-6F || right.norm() < 1e-6F || closing.norm() < 1e-6F) {
        return false;
    }
    const auto l = left.normalized();
    const auto r = right.normalized();
    const auto c = closing.normalized();
    return l.dot(c) >= cosine && r.dot(c) <= -cosine && l.dot(r) <= -cosine;
}

inline bool fingerBodiesClear(const std::vector<Eigen::Vector3f> &environment,
    const Eigen::Isometry3f &pose, float opening, float tcp_offset,
    float width, float height, float length) {
    const Eigen::Isometry3f inverse = pose.inverse();
    for (const auto &point : environment) {
        const Eigen::Vector3f local = inverse * point;
        const float depth = local.z() - (-tcp_offset + length*0.5F);
        if (std::abs(local.x()) > height*0.5F + 0.0025F ||
            std::abs(depth) > length*0.5F + 0.0025F) continue;
        for (const float side : {-1.0F, 1.0F}) {
            if (std::abs(local.y() - side*(opening+width)*0.5F) <=
                width*0.5F + 0.0025F) return false;
        }
    }
    return true;
}

// Point geometry only; ROS observations and robot FK are supplied by the node.
// Support contact is restricted to the held object's bottom and exposed support
// samples. Multiple support tops may permit bottom contact; other geometry blocks.
class AttachedObjectScene {
public:
    // Immutable geometry only: no collision decision is cached. Preserve source
    // ordering so support labels, first-hit diagnostics and noise budgets agree.
    class SourceBounds {
        friend class AttachedObjectScene;
        struct Box { Eigen::Vector3f low, high; };
        static constexpr size_t block_size = 128;
        std::vector<Eigen::Vector3f> snapshot_;
        std::vector<Box> blocks_;
        Box whole_;
        bool valid_ = true;
        bool matches(const std::vector<Eigen::Vector3f> &source) const {
            return valid_ && source.size() == snapshot_.size() &&
                (source.empty() || std::memcmp(source.data(), snapshot_.data(),
                    source.size()*sizeof(Eigen::Vector3f)) == 0);
        }
    public:
        explicit SourceBounds(const std::vector<Eigen::Vector3f> &source) : snapshot_(source) {
            if (source.empty()) { valid_ = false; return; }
            whole_ = {source.front(), source.front()};
            for (size_t begin = 0; begin < source.size(); begin += block_size) {
                Box box{source[begin], source[begin]};
                for (size_t i = begin; i < std::min(source.size(), begin+block_size); ++i) {
                    valid_ = valid_ && source[i].allFinite();
                    box.low = box.low.cwiseMin(source[i]);
                    box.high = box.high.cwiseMax(source[i]);
                }
                blocks_.push_back(box);
                whole_.low = whole_.low.cwiseMin(box.low);
                whole_.high = whole_.high.cwiseMax(box.high);
            }
        }
    };
    // A segment-wide budget, not a fresh allowance at every waypoint. Count
    // unique obstacle samples (never source/sample pairs) and bound their
    // occupied cells. Separate sparse clusters have no distance restriction.
    struct NoiseBudget {
        size_t maximum_points = 0;
        size_t maximum_voxels = 0;
        std::vector<Eigen::Vector3f> points;
        size_t observed_points = 0;
        size_t observed_voxels = 0;

        bool add(const Eigen::Vector3f &point) {
            if (std::any_of(points.begin(), points.end(), [&](const auto &existing) {
                return (point.array() == existing.array()).all();
            })) return true;
            points.push_back(point);
            observed_points = points.size();
            if (points.size() > maximum_points || maximum_points == 0) return false;
            std::vector<Eigen::Vector3i> voxels;
            for (const auto &sample : points) {
                const Eigen::Vector3i voxel =
                    (sample.array() / 0.005F).floor().cast<int>().matrix();
                if (std::none_of(voxels.begin(), voxels.end(), [&](const auto &existing) {
                    return (voxel.array() == existing.array()).all();
                })) voxels.push_back(voxel);
            }
            observed_voxels = voxels.size();
            return observed_voxels <= maximum_voxels;
        }
    };
    struct Sample { Eigen::Vector3f point; bool support; };
    struct CollisionInfo {
        Eigen::Vector3f held_point;
        Eigen::Vector3f obstacle_point;
        bool named_support = false;
        bool source_bottom = false;
    };
    struct InitialContact {
        float distance_squared = std::numeric_limits<float>::infinity();
        Eigen::Vector3f obstacle = Eigen::Vector3f::Zero();
    };
    std::vector<InitialContact> initialContacts(
        const std::vector<Eigen::Vector3f> &source) const {
        std::vector<InitialContact> result(source.size());
        for (size_t index = 0; index < source.size(); ++index) {
            const auto cell = cellOf(source[index]);
            for (int x = -1; x <= 1; ++x)
                for (int y = -1; y <= 1; ++y)
                    for (int z = -1; z <= 1; ++z) {
                        const auto entry = cells_.find(cell + Eigen::Vector3i(x,y,z));
                        if (entry == cells_.end()) continue;
                        for (const auto &sample : entry->second) {
                            const float distance = (source[index] - sample.point).squaredNorm();
                            if (distance <= 9e-6F && distance < result[index].distance_squared)
                                result[index] = {distance, sample.point};
                        }
                    }
        }
        return result;
    }
    void add(const Eigen::Vector3f &point, bool support) {
        if (point.allFinite()) {
            cells_[cellOf(point)].push_back({point, support});
            coarse_cells_.insert(coarseCellOf(point));
        }
    }
    bool clear(const std::vector<Eigen::Vector3f> &source,
        const Eigen::Isometry3f &delta, float bottom_z, bool final_contact,
        CollisionInfo *collision = nullptr,
        const std::vector<uint8_t> *bottom_flags = nullptr,
        float support_penetration_m = 0.001F,
        const std::vector<InitialContact> *departure_contacts = nullptr,
        float departure_slack_m = 0.0F,
        NoiseBudget *noise_budget = nullptr,
        const SourceBounds *bounds = nullptr) const {
        const float slack = std::clamp(departure_slack_m, 0.0F, 0.002F);
        NoiseBudget proposed;
        if (noise_budget) proposed = *noise_budget;
        const bool indexed = bounds && bounds->matches(source) && delta.matrix().allFinite();
        if (indexed && !boundsMayCollide(bounds->whole_, delta)) return true;
        for (size_t source_index = 0; source_index < source.size(); ++source_index) {
            if (indexed && source_index % SourceBounds::block_size == 0 &&
                !boundsMayCollide(bounds->blocks_[source_index/SourceBounds::block_size], delta)) {
                source_index = std::min(source.size(), source_index+SourceBounds::block_size)-1;
                continue;
            }
            const auto &source_point = source[source_index];
            const bool source_bottom = bottom_flags && bottom_flags->size() == source.size() ?
                (*bottom_flags)[source_index] != 0 : source_point.z() <= bottom_z + 0.003F;
            const Eigen::Vector3f point = delta * source_point;
            const Eigen::Vector3i cell = cellOf(point);
            // The unchanged 3 mm narrow test can only reach cells in this
            // conservative AABB. Retain the original neighborhood and order.
            const Eigen::Vector3i lower = cellOf(point - Eigen::Vector3f::Constant(0.003001F))
                .cwiseMax(cell - Eigen::Vector3i::Ones());
            const Eigen::Vector3i upper = cellOf(point + Eigen::Vector3f::Constant(0.003001F))
                .cwiseMin(cell + Eigen::Vector3i::Ones());
            for (int x = lower.x(); x <= upper.x(); ++x) {
                for (int y = lower.y(); y <= upper.y(); ++y) {
                    for (int z = lower.z(); z <= upper.z(); ++z) {
                        const auto entry = cells_.find(Eigen::Vector3i(x,y,z));
                        if (entry == cells_.end()) continue;
                        for (const auto &sample : entry->second) {
                            if ((point - sample.point).squaredNorm() > 9e-6F) continue;
                            if (final_contact && sample.support &&
                                source_bottom &&
                                point.z() >= sample.point.z() - std::clamp(support_penetration_m, 0.001F, 0.003F)) continue;
                            // Opt-in source departure only: retain the measured
                            // initial clearance within the configured small
                            // allowance. Never admit new contacts or movement
                            // beyond the bounded 6 mm. The caller must still
                            // require ordinary clearance at departure completion.
                            if (departure_contacts && departure_contacts->size() == source.size()) {
                                const auto &initial = (*departure_contacts)[source_index];
                                const Eigen::Vector3f movement = point - source_point;
                                const float initial_distance = std::sqrt(initial.distance_squared);
                                const float minimum_distance = std::max(0.0F, initial_distance - slack);
                                if ((slack > 0.0F || initial.distance_squared >= 0.00025F*0.00025F) &&
                                    initial.distance_squared <= 9e-6F &&
                                    movement.norm() <= 0.006F &&
                                    (source_point-sample.point).squaredNorm() <= 9e-6F &&
                                    (point-sample.point).squaredNorm() + 1e-10F >= minimum_distance*minimum_distance &&
                                    movement.dot(source_point-initial.obstacle) >=
                                        -slack*initial_distance - 1e-9F) continue;
                            }
                            if (noise_budget && proposed.add(sample.point)) continue;
                            if (noise_budget) {
                                noise_budget->observed_points = proposed.observed_points;
                                noise_budget->observed_voxels = proposed.observed_voxels;
                            }
                            if (collision) {
                                collision->held_point = point;
                                collision->obstacle_point = sample.point;
                                collision->named_support = sample.support;
                                collision->source_bottom = source_bottom;
                            }
                            return false;
                        }
                    }
                }
            }
        }
        if (noise_budget) *noise_budget = std::move(proposed);
        return true;
    }
private:
    struct Hash {
        std::size_t operator()(const Eigen::Vector3i &v) const {
            return static_cast<std::size_t>(static_cast<uint32_t>(v.x())) * 73856093ULL ^
                static_cast<std::size_t>(static_cast<uint32_t>(v.y())) * 19349663ULL ^
                static_cast<std::size_t>(static_cast<uint32_t>(v.z())) * 83492791ULL;
        }
    };
    struct Equal {
        bool operator()(const Eigen::Vector3i &a, const Eigen::Vector3i &b) const {
            return (a.array() == b.array()).all();
        }
    };
    static Eigen::Vector3i cellOf(const Eigen::Vector3f &point) {
        return (point.array() / 0.005F).floor().cast<int>().matrix();
    }
    static Eigen::Vector3i coarseCellOf(const Eigen::Vector3f &point) {
        return (point.array() / 0.04F).floor().cast<int>().matrix();
    }
    bool boundsMayCollide(const SourceBounds::Box &box, const Eigen::Isometry3f &delta) const {
        const Eigen::Vector3f center = delta*((box.low+box.high)*0.5F);
        // Outward padding covers the unchanged 3 mm narrow-phase radius plus
        // conservative float rounding; occupied boxes always use exact points.
        const Eigen::Vector3f half = delta.linear().cwiseAbs()*((box.high-box.low)*0.5F)+
            Eigen::Vector3f::Constant(0.0031F);
        const auto low = coarseCellOf(center-half), high = coarseCellOf(center+half);
        for (int x=low.x(); x<=high.x(); ++x)
            for (int y=low.y(); y<=high.y(); ++y)
                for (int z=low.z(); z<=high.z(); ++z)
                    if (coarse_cells_.find(Eigen::Vector3i(x,y,z)) != coarse_cells_.end()) return true;
        return false;
    }
    std::unordered_map<Eigen::Vector3i, std::vector<Sample>, Hash, Equal> cells_;
    std::unordered_set<Eigen::Vector3i, Hash, Equal> coarse_cells_;
};
}  // namespace vamp_node
