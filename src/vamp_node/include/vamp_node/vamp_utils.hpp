#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vamp/collision/environment.hh>
#include <vamp/vector.hh>

#include "vamp_node/placement_point_index.hpp"

namespace vamp_node {

// --- Empty Hand Withdrawal Contacts ---
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
            initial_.find(point) == initial_.end() || penetration > maximum_ + .002F)
            return false;
        current_[point] = std::max(current_[point], penetration);
        return true;
    }
    bool separating(float travel) const {
        if (!std::isfinite(travel) || travel < 0.F) return false;
        float initial_energy = 0.F, current_energy = 0.F;
        for (const auto &sample : initial_) initial_energy += sample.second * sample.second;
        for (const auto &sample : current_) current_energy += sample.second * sample.second;
        return current_energy <= initial_energy + std::min(1.e-6F, initial_energy * .01F);
    }
    float initialEnergy() const { return energy(initial_); }
    float currentEnergy() const { return energy(current_); }
    std::size_t size() const { return initial_.size(); }
private:
    static float energy(const std::unordered_map<std::size_t, float> &points) {
        float result = 0.F;
        for (const auto &sample : points) result += sample.second * sample.second;
        return result;
    }
    std::unordered_map<std::size_t, float> initial_, current_;
    float maximum_ = 0.F;
};

// --- Foreground Components Connectivity ---
inline bool substantialSurfaceConnected(const std::vector<Eigen::Vector3f> &points,
    const PlacementPointIndex &foreground, size_t minimum_points, float distance)
{
    if (points.size() < minimum_points || distance <= 0.F || !std::isfinite(distance)) return false;
    size_t neighbors = 0;
    for (const auto &point : points) {
        if (point.allFinite() && !foreground.sphereClear(point, distance) && ++neighbors >= 3) return true;
    }
    return false;
}

// --- Grasp Diversity Duplicate Check ---
inline bool gpdPosesAreDuplicates(
    const Eigen::Vector3f &position_a, const Eigen::Vector3f &approach_a,
    const Eigen::Vector3f &closing_a, const Eigen::Vector3f &position_b,
    const Eigen::Vector3f &approach_b, const Eigen::Vector3f &closing_b,
    float position_threshold_m, float approach_threshold_degrees,
    float closing_threshold_degrees)
{
    constexpr float kPi = 3.14159265358979323846F;
    const float approach_cosine = std::cos(approach_threshold_degrees * kPi / 180.0F);
    const float closing_cosine = std::cos(closing_threshold_degrees * kPi / 180.0F);
    return (position_a - position_b).norm() < position_threshold_m &&
        approach_a.dot(approach_b) > approach_cosine &&
        std::abs(closing_a.dot(closing_b)) > closing_cosine;
}

// --- Grasp Position Offset Correction ---
inline Eigen::Vector3f validatedGraspPositionOffset(const std::vector<double> &values)
{
    if (values.size() != 3 || !std::isfinite(values[0]) ||
        !std::isfinite(values[1]) || !std::isfinite(values[2])) {
        throw std::invalid_argument("grasp_position_offset_base_m requires three finite metres");
    }
    return Eigen::Vector3f(values[0], values[1], values[2]);
}

inline Eigen::Isometry3f correctedGraspTarget(
    const Eigen::Isometry3f &target, const Eigen::Vector3f &offset, bool apply = true)
{
    Eigen::Isometry3f result = target;
    if (apply) result.translation() += offset;
    return result;
}

// --- Placement Candidate Ranking ---
inline bool preferPlacement(float score, float cost, float best_score, float best_cost)
{
    if (!std::isfinite(score) || !std::isfinite(cost)) return false;
    constexpr float score_tolerance = 1.0e-9F;
    return score > best_score + score_tolerance ||
        (std::abs(score - best_score) <= score_tolerance && cost < best_cost);
}

// --- Planar Body Mask for Dishes / Plates ---
inline std::vector<bool> planar_body_mask(const std::vector<Eigen::Vector3f> &points)
{
    std::vector<bool> keep(points.size(), true);
    if (points.size() < 80) return keep;
    std::mt19937 generator(17);
    std::uniform_int_distribution<size_t> index(0, points.size() - 1);
    Eigen::Vector3f best_normal = Eigen::Vector3f::UnitZ();
    float best_offset = 0.F;
    size_t best_count = 0;
    for (int trial = 0; trial < 96; ++trial) {
        const auto &a = points[index(generator)];
        Eigen::Vector3f normal = (points[index(generator)] - a).cross(points[index(generator)] - a);
        if (normal.norm() < 1e-8f) continue;
        normal.normalize();
        if (normal.z() < 0.F) normal = -normal;
        if (normal.z() < .9F) continue;
        const float offset = -normal.dot(a);
        size_t count = 0;
        for (const auto &p : points) {
            if (std::abs(normal.dot(p) + offset) <= .004F) ++count;
        }
        if (count > best_count) {
            best_count = count;
            best_normal = normal;
            best_offset = offset;
        }
    }
    if (best_count < points.size() * .4F) return keep;
    for (size_t i = 0; i < points.size(); ++i) {
        const float height = best_normal.dot(points[i]) + best_offset;
        keep[i] = height >= -.004F && height <= .022F;
    }
    return keep;
}

// --- Piper Finger Collision Geometry ---
inline Eigen::Vector3f piperFingerSphereCenter(size_t index,
    const Eigen::Vector3f &center, const Eigen::Isometry3f &pose, float opening)
{
    if (index < 82 || index >= 96 || !std::isfinite(opening) || opening < 0.0F)
        return center;
    const float side = index < 89 ? -1.0F : 1.0F;
    return center + pose.linear().col(1) *
        (side * 0.5F * (std::clamp(opening, 0.0F, 0.10F) - 0.070F));
}

inline bool piperFingerMeshBoundsContain(size_t index, const Eigen::Vector3f &point,
    const Eigen::Isometry3f &pose, float opening, float padding = 0.0025F)
{
    if (index < 82 || index >= 96 || !std::isfinite(opening) || opening < 0.0F)
        return true;
    const Eigen::Vector3f local = pose.inverse() * point;
    const float gap = 0.5F * std::clamp(opening, 0.0F, 0.10F);
    const float inner = index < 89 ? -gap : gap;
    const float outer = index < 89 ? -gap - 0.0265F : gap + 0.0265F;
    return std::abs(local.x()) <= 0.028F + padding &&
        local.y() >= std::min(inner, outer) - padding &&
        local.y() <= std::max(inner, outer) + padding &&
        local.z() >= 0.0615F - padding && local.z() <= 0.138F + padding;
}

// --- Organized XYZ PointCloud2 Reader ---
class OrganizedXYZReader {
public:
    explicit OrganizedXYZReader(const sensor_msgs::msg::PointCloud2 &cloud) : cloud_(cloud) {
        const char *names[] = {"x", "y", "z"};
        for (size_t axis = 0; axis < 3; ++axis) {
            bool found = false;
            for (const auto &field : cloud.fields) {
                if (field.name != names[axis]) continue;
                offsets_[axis] = field.offset;
                found = field.offset <= cloud.point_step && cloud.point_step - field.offset >= 4;
                break;
            }
            if (!found) return;
        }
        valid_ = cloud.point_step > 0;
        base_frame_ = cloud.header.frame_id == "base_link";
    }
    bool read(int u, int v, std::array<float, 3> &point) const {
        if (!valid_ || cloud_.height <= 1 || u < 0 || v < 0 ||
            u >= static_cast<int>(cloud_.width) || v >= static_cast<int>(cloud_.height)) return false;
        const size_t index = static_cast<size_t>(v) * cloud_.width + static_cast<size_t>(u);
        const size_t offset = index * cloud_.point_step;
        if (offset > cloud_.data.size() || cloud_.data.size() - offset < cloud_.point_step) return false;
        for (size_t axis = 0; axis < 3; ++axis) {
            std::memcpy(&point[axis], cloud_.data.data() + offset + offsets_[axis], sizeof(float));
            if (!std::isfinite(point[axis])) return false;
        }
        return base_frame_ || point[2] > 0.05F;
    }
private:
    const sensor_msgs::msg::PointCloud2 &cloud_;
    std::array<size_t, 3> offsets_{};
    bool valid_ = false;
    bool base_frame_ = false;
};

// --- Payload Checked Robot Validator ---
template <typename Base>
struct PayloadCheckedRobot : Base {
    inline static thread_local std::function<bool(const std::vector<float> &)> validator;

    template <std::size_t width>
    static bool fkcc(
        const vamp::collision::Environment<vamp::FloatVector<width>> &environment,
        const typename Base::template ConfigurationBlock<width> &configuration) noexcept {
        if (!Base::template fkcc<width>(environment, configuration) || !validator) return false;
        for (std::size_t lane = 0; lane < width; ++lane) {
            std::vector<float> joints(Base::dimension);
            for (std::size_t joint = 0; joint < Base::dimension; ++joint) {
                joints[joint] = configuration[joint].to_array()[lane];
            }
            if (!validator(joints)) return false;
        }
        return true;
    }

    class ScopedValidator {
    public:
        explicit ScopedValidator(std::function<bool(const std::vector<float> &)> callback)
            : previous_(std::move(validator)) { validator = std::move(callback); }
        ~ScopedValidator() { validator = std::move(previous_); }
        ScopedValidator(const ScopedValidator &) = delete;
        ScopedValidator &operator=(const ScopedValidator &) = delete;
    private:
        std::function<bool(const std::vector<float> &)> previous_;
    };
};

}  // namespace vamp_node

