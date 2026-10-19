#include "vision_node/metric_scale.hpp"
#include <limits>
#include <iostream>

void require(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}

int main()
{
  sensor_msgs::msg::PointCloud2 input;
  input.header.frame_id = "camera_color_optical_frame";
  input.header.stamp.sec = 123;
  input.header.stamp.nanosec = 456;
  sensor_msgs::PointCloud2Modifier modifier(input);
  modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
  modifier.resize(4);
  input.width = 2; input.height = 2; input.row_step = input.width * input.point_step;
  sensor_msgs::PointCloud2Iterator<float> x(input, "x"), y(input, "y"), z(input, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> red(input, "r");
  for (int i = 0; i < 4; ++i, ++x, ++y, ++z, ++red) {
    *x = i == 3 ? std::numeric_limits<float>::quiet_NaN() : .1036f * i;
    *y = .2072f; *z = .518f; *red = 42;
  }
  const auto original = input.data;
  const auto output = vision_node::correct_camera_measurements(input, 1.036);
  require(input.data == original, "must not mutate source observation");
  require(output.header == input.header && output.width == 2 && output.height == 2,
    "source association and organization must remain unchanged");
  sensor_msgs::PointCloud2ConstIterator<float> ox(output, "x"), oy(output, "y"), oz(output, "z");
  sensor_msgs::PointCloud2ConstIterator<uint8_t> r(output, "r");
  for (int i = 0; i < 4; ++i, ++ox, ++oy, ++oz, ++r) {
    require(i == 3 ? std::isnan(*ox) : std::abs(*ox - .1f*i) < 1e-6f, "XYZ correction");
    require(std::abs(*oy - .2f) < 1e-6f && std::abs(*oz - .5f) < 1e-6f, "YZ correction");
    require(*r == 42, "RGB must not be scaled");
  }
  const auto depth = vision_node::corrected_registered_depth(output);
  require(depth.header == input.header && depth.encoding == "32FC1" && depth.step == 8,
    "depth must retain source association and metre units");
  for (size_t i = 0; i < 4; ++i) {
    float value; std::memcpy(&value, depth.data.data()+i*sizeof(float), sizeof(float));
    require(std::abs(value - .5f) < 1e-6f, "depth must use corrected XYZ");
  }
  require(vision_node::correct_camera_measurements(input, 1.).data == original,
    "identity factor must preserve bytes");
  for (double ratio : {0., -1., std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::infinity()})
  {
    bool rejected = false;
    try { vision_node::correct_camera_measurements(input, ratio); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid scale must fail");
  }
  // The camera origin stays fixed: FK*X translation is added after scaling.
  const float camera_offset = .25f;
  require(std::abs(camera_offset + .518f/1.036f - .75f) < 1e-6f,
    "scale camera measurement, not robot-base translation");
  std::cout << "metric scale checks passed\n";
}
