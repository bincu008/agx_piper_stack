#pragma once
#include <algorithm>
#include <optional>
#include <vector>
#include <Eigen/Core>

namespace vamp_node {
// Recover only measured, locally supported rim samples. Candidate work is bounded.
inline std::optional<Eigen::Vector3f> supportedRimSeed(
    const std::vector<Eigen::Vector3f> &points,
    const Eigen::Vector3f &seed, float top_z) {
    std::vector<std::pair<float, size_t>> candidates;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto &p = points[i];
        const float distance = (p.head<2>() - seed.head<2>()).squaredNorm();
        if (p.allFinite() && std::abs(p.z() - top_z) <= 0.025F &&
            std::abs(p.z() - seed.z()) <= 0.050F && distance < 0.060F * 0.060F)
            candidates.emplace_back(distance, i);
    }
    const size_t count = std::min<size_t>(32, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + count, candidates.end());
    for (size_t i = 0; i < count; ++i) {
        const auto &candidate = points[candidates[i].second];
        size_t support = 0;
        for (const auto &p : points) {
            if ((p.head<2>() - candidate.head<2>()).squaredNorm() <= 0.008F * 0.008F &&
                std::abs(p.z() - candidate.z()) <= 0.006F && ++support >= 30)
                return candidate;
        }
    }
    return std::nullopt;
}
// Retain different measured contact regions instead of collapsing all seeds
// onto the nearest rim. The same local-support checks apply to every region.
inline std::vector<Eigen::Vector3f> supportedRimSeeds(
    const std::vector<Eigen::Vector3f> &points,
    const Eigen::Vector3f &seed, float top_z) {
    std::vector<Eigen::Vector3f> result;
    auto first = supportedRimSeed(points, seed, top_z);
    if (first) result.push_back(*first);
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    size_t finite_count = 0;
    for (const auto &point : points) {
        if (point.allFinite()) { centroid += point; ++finite_count; }
    }
    if (!finite_count) return result;
    centroid /= static_cast<float>(finite_count);
    for (int quadrant = 0; quadrant < 4 && result.size() < 4; ++quadrant) {
        std::vector<Eigen::Vector3f> region;
        for (const auto &point : points) {
            const int sector = (point.x() >= centroid.x() ? 1 : 0) +
                (point.y() >= centroid.y() ? 2 : 0);
            if (point.allFinite() && sector == quadrant &&
                (point.head<2>() - seed.head<2>()).squaredNorm() < 0.060F * 0.060F &&
                std::abs(point.z() - seed.z()) <= 0.050F &&
                std::abs(point.z() - top_z) <= 0.025F) region.push_back(point);
        }
        if (region.empty()) continue;
        const auto outer = std::max_element(region.begin(), region.end(),
            [&centroid](const auto &left, const auto &right) {
                return (left.head(2) - centroid.head(2)).squaredNorm() <
                    (right.head(2) - centroid.head(2)).squaredNorm();
            });
        const auto candidate = supportedRimSeed(region, *outer, top_z);
        if (!candidate) continue;
        bool distinct = true;
        for (const auto &existing : result)
            if ((*candidate - existing).norm() < 0.025F) distinct = false;
        if (distinct) result.push_back(*candidate);
    }
    return result;
}
}  // namespace vamp_node
