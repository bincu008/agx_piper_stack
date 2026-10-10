#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include "vamp_node/placement_geometry.hpp"
#include "vamp_node/placement_point_index.hpp"

namespace vamp_node {
// Cache only exact immutable geometry, never IK or collision outcomes.
class LayoutSceneCache {
public:
    struct Geometry {
        std::vector<std::array<float, 3>> environment;
        std::vector<Eigen::Vector3f> points;
        std::vector<uint8_t> support_labels;
        AttachedObjectScene attached;
        std::unique_ptr<PlacementPointIndex> index;
    };
    std::shared_ptr<const Geometry> get(
        const sensor_msgs::msg::PointCloud2 &cloud, const std::string &mode) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (geometry_ && mode == mode_ && cloud == snapshot_) return geometry_;
        auto result = std::make_shared<Geometry>();
        const size_t count = static_cast<size_t>(cloud.width) * cloud.height;
        result->environment.reserve(count);
        result->points.reserve(count);
        result->support_labels.reserve(count);
        sensor_msgs::PointCloud2ConstIterator<uint8_t> support(cloud, "support");
        sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z");
        for (; x != x.end(); ++x, ++y, ++z, ++support) {
            if (!std::isfinite(*x) || !std::isfinite(*y) || !std::isfinite(*z)) continue;
            result->environment.push_back({*x, *y, *z});
            result->points.emplace_back(*x, *y, *z);
            result->support_labels.push_back(*support);
            const bool contact_support = mode == "stack" ? *support == 2 :
                mode == "auto" ? (*support == 1 || *support == 2) : *support == 1;
            result->attached.add(result->points.back(), contact_support);
        }
        result->index = std::make_unique<PlacementPointIndex>(result->points);
        // Retain only one scene and bound the retained input size. Large
        // requests still receive the same checks, without retaining a cache.
        geometry_.reset();
        snapshot_ = sensor_msgs::msg::PointCloud2();
        if (cloud.data.size() <= 8 * 1024 * 1024) {
            snapshot_ = cloud;
            mode_ = mode;
            geometry_ = result;
        }
        return result;
    }
private:
    std::mutex mutex_;
    sensor_msgs::msg::PointCloud2 snapshot_;
    std::string mode_;
    std::shared_ptr<const Geometry> geometry_;
};
}  // namespace vamp_node
