#ifndef VISION_NODE__METRIC_SCALE_HPP_
#define VISION_NODE__METRIC_SCALE_HPP_

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace vision_node
{
inline sensor_msgs::msg::PointCloud2 correct_camera_measurements(
  const sensor_msgs::msg::PointCloud2 & input, double measured_to_real_ratio)
{
  if (!std::isfinite(measured_to_real_ratio) || measured_to_real_ratio <= 0.) {
    throw std::invalid_argument("perception_measurement_ratio must be finite and positive");
  }
  auto output = input;
  if (measured_to_real_ratio == 1. || output.data.empty()) return output;
  const float correction = static_cast<float>(1. / measured_to_real_ratio);
  sensor_msgs::PointCloud2Iterator<float> x(output, "x"), y(output, "y"), z(output, "z");
  for (; x != x.end(); ++x, ++y, ++z) {
    *x *= correction; *y *= correction; *z *= correction;
  }
  return output;
}

inline sensor_msgs::msg::Image corrected_registered_depth(
  const sensor_msgs::msg::PointCloud2 & cloud)
{
  sensor_msgs::msg::Image depth;
  depth.header = cloud.header;
  depth.height = cloud.height;
  depth.width = cloud.width;
  depth.encoding = "32FC1";  // Metres, reconstructed from the same corrected XYZ.
  depth.is_bigendian = cloud.is_bigendian;
  depth.step = cloud.width * sizeof(float);
  depth.data.resize(static_cast<size_t>(depth.step) * depth.height);
  sensor_msgs::PointCloud2ConstIterator<float> z(cloud, "z");
  size_t offset = 0;
  for (; z != z.end(); ++z, offset += sizeof(float)) {
    std::memcpy(depth.data.data() + offset, &*z, sizeof(float));
  }
  return depth;
}
}  // namespace vision_node
#endif  // VISION_NODE__METRIC_SCALE_HPP_
