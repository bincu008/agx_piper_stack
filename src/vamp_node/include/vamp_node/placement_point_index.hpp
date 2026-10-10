#pragma once

#include <cmath>
#include <array>
#include <cstring>
#include <deque>
#include <memory>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <Eigen/Geometry>

namespace vamp_node {
// Index exact points; do not downsample or reduce any collision radius.
class PlacementPointIndex {
public:
    template<class Point>
    explicit PlacementPointIndex(const std::vector<Point> &points) {
        for (size_t index = 0; index < points.size(); ++index) {
            const Eigen::Vector3f point(points[index][0], points[index][1], points[index][2]);
            if (point.allFinite()) cells_[cell(point)].push_back({point, index});
        }
    }
    template<class Predicate>
    bool sphereClearWithPredicate(const Eigen::Vector3f &center, float radius,
        Predicate collides) const {
        return boxClearIndexed(center-Eigen::Vector3f::Constant(radius),
            center+Eigen::Vector3f::Constant(radius), collides);
    }
    bool sphereClear(const Eigen::Vector3f &center, float radius) const {
        return boxClear(center - Eigen::Vector3f::Constant(radius),
            center + Eigen::Vector3f::Constant(radius), [&](const auto &point) {
                return (point-center).squaredNorm() < radius*radius;
            });
    }
    bool fingersClear(const Eigen::Isometry3f &pose, float opening,
        float tcp_offset, float width, float height, float length) const {
        const Eigen::Vector3f local_center(0,0,-tcp_offset+length*0.5F);
        const Eigen::Vector3f half(height*0.5F+0.0025F,
            opening*0.5F+width+0.0025F, length*0.5F+0.0025F);
        const Eigen::Vector3f center=pose*local_center;
        const Eigen::Vector3f extent=pose.linear().cwiseAbs()*half;
        const Eigen::Isometry3f inverse=pose.inverse();
        return boxClear(center-extent,center+extent,[&](const auto &point) {
            const auto local=inverse*point;
            if (std::abs(local.x())>half.x() ||
                std::abs(local.z()-local_center.z())>half.z()) return false;
            for (float side : {-1.0F,1.0F}) {
                if (std::abs(local.y()-side*(opening+width)*0.5F)<=
                    width*0.5F+0.0025F) return true;
            }
            return false;
        });
    }
private:
    struct Hash {
        size_t operator()(const Eigen::Vector3i &v) const {
            return static_cast<std::size_t>(static_cast<uint32_t>(v.x())) * 73856093ULL ^
                static_cast<std::size_t>(static_cast<uint32_t>(v.y())) * 19349663ULL ^
                static_cast<std::size_t>(static_cast<uint32_t>(v.z())) * 83492791ULL;
        }
    };
    struct Equal {
        bool operator()(const Eigen::Vector3i &a,const Eigen::Vector3i &b) const {
            return (a.array()==b.array()).all();
        }
    };
    static Eigen::Vector3i cell(const Eigen::Vector3f &p) {
        return (p.array()/0.02F).floor().cast<int>().matrix();
    }
    template<class Predicate>
    bool boxClear(const Eigen::Vector3f &lower,const Eigen::Vector3f &upper,
        Predicate collides) const {
        return boxClearIndexed(lower, upper, [&](size_t, const auto &point) {
            return collides(point);
        });
    }
    template<class Predicate>
    bool boxClearIndexed(const Eigen::Vector3f &lower,const Eigen::Vector3f &upper,
        Predicate collides) const {
        const auto lo=cell(lower-Eigen::Vector3f::Constant(1e-6F));
        const auto hi=cell(upper+Eigen::Vector3f::Constant(1e-6F));
        for (int x=lo.x();x<=hi.x();++x)
            for (int y=lo.y();y<=hi.y();++y)
                for (int z=lo.z();z<=hi.z();++z) {
                    auto found=cells_.find(Eigen::Vector3i(x,y,z));
                    if (found==cells_.end()) continue;
                    for (const auto &entry:found->second)
                        if (collides(entry.index, entry.point)) return false;
                }
        return true;
    }
    struct Entry { Eigen::Vector3f point; size_t index; };
    std::unordered_map<Eigen::Vector3i,std::vector<Entry>,Hash,Equal> cells_;
};

// Cache geometry only. Complete byte comparisons catch in-place scene edits;
// no collision decision, classification, FK or IK result is cached here.
class ExactPointIndexCache {
public:
    using Point = std::array<float, 3>;
    const PlacementPointIndex &get(const std::vector<Point> &points) {
        for (auto &entry : entries_) {
            if (entry.snapshot.size() == points.size() && (points.empty() ||
                std::memcmp(entry.snapshot.data(), points.data(),
                    points.size()*sizeof(Point)) == 0)) return *entry.index;
        }
        if (entries_.size() == 2) entries_.pop_back();
        entries_.push_front({points, std::make_unique<PlacementPointIndex>(points)});
        return *entries_.front().index;
    }
private:
    struct Entry {
        std::vector<Point> snapshot;
        std::unique_ptr<PlacementPointIndex> index;
    };
    std::deque<Entry> entries_;
};
}  // namespace vamp_node
