#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace vamp_node {
// Read the same native FLOAT32 values as the existing ROS iterator adapter,
// resolving field offsets once rather than for every organized-cloud pixel.
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
        // Preserve the existing linear organized iterator convention.
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
}  // namespace vamp_node
