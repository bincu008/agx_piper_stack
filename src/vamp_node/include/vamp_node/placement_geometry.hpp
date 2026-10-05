#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <Eigen/Geometry>

namespace vamp_node {
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
// Support contact is restricted to the held object's bottom and named support
// samples. Only the selected support top may permit bottom contact; other geometry blocks.
class AttachedObjectScene {
public:
    struct Sample { Eigen::Vector3f point; bool support; };
    struct CollisionInfo {
        Eigen::Vector3f held_point;
        Eigen::Vector3f obstacle_point;
        bool named_support = false;
        bool source_bottom = false;
    };
    void add(const Eigen::Vector3f &point, bool support) {
        if (point.allFinite()) cells_[cellOf(point)].push_back({point, support});
    }
    bool clear(const std::vector<Eigen::Vector3f> &source,
        const Eigen::Isometry3f &delta, float bottom_z, bool final_contact,
        CollisionInfo *collision = nullptr) const {
        for (const auto &source_point : source) {
            const Eigen::Vector3f point = delta * source_point;
            const Eigen::Vector3i cell = cellOf(point);
            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    for (int z = -1; z <= 1; ++z) {
                        const auto entry = cells_.find(Eigen::Vector3i(cell + Eigen::Vector3i(x,y,z)));
                        if (entry == cells_.end()) continue;
                        for (const auto &sample : entry->second) {
                            if ((point - sample.point).squaredNorm() > 9e-6F) continue;
                            if (final_contact && sample.support &&
                                source_point.z() <= bottom_z + 0.003F &&
                                point.z() >= sample.point.z() - 0.001F) continue;
                            if (collision) {
                                collision->held_point = point;
                                collision->obstacle_point = sample.point;
                                collision->named_support = sample.support;
                                collision->source_bottom = source_point.z() <= bottom_z + 0.003F;
                            }
                            return false;
                        }
                    }
                }
            }
        }
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
    std::unordered_map<Eigen::Vector3i, std::vector<Sample>, Hash, Equal> cells_;
};
}  // namespace vamp_node
