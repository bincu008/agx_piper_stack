#include <array>
#include <algorithm>
#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <agx_arm_msgs/msg/gripper_status.hpp>
#include <vamp/robots/piper.hh>
#include "vamp_node/piper_finger_geometry.hpp"

class CollisionSpheresViewer : public rclcpp::Node
{
public:
  CollisionSpheresViewer() : Node("collision_spheres_viewer")
  {
    pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/vamp_debug/current_collision_spheres", rclcpp::QoS(1).transient_local());
    gripper_ = create_subscription<agx_arm_msgs::msg::GripperStatus>(
      "/feedback/gripper_status", rclcpp::SensorDataQoS(),
      [this](agx_arm_msgs::msg::GripperStatus::ConstSharedPtr msg) {
        if (std::isfinite(msg->width)) { opening_ = msg->width; grip_time_ = now(); }
      });
    joints_ = create_subscription<sensor_msgs::msg::JointState>(
      "/feedback/joint_states", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) { publish(*msg); });
  }
private:
  void publish(const sensor_msgs::msg::JointState &msg)
  {
    using Robot = vamp::robots::Piper;
    std::array<float, 6> q{};
    for (size_t j = 0; j < 6; ++j) {
      const auto name = "joint"+std::to_string(j+1);
      const auto found = std::find(msg.name.begin(), msg.name.end(), name);
      if (found == msg.name.end()) return;
      const auto index = std::distance(msg.name.begin(), found);
      if (static_cast<size_t>(index) >= msg.position.size() ||
          !std::isfinite(msg.position[index])) return;
      q[j] = msg.position[index];
    }
    Robot::ConfigurationBlock<1> input;
    for (size_t j = 0; j < 6; ++j) input[j][0] = q[j];
    Robot::Spheres<1> spheres;
    Robot::template sphere_fk<1>(input, spheres);
    const auto pose = Robot::eefk(q);
    const float opening = (now()-grip_time_).seconds() < .5 ? opening_ : -1.F;
    visualization_msgs::msg::MarkerArray output;
    for (size_t s = 0; s < Robot::n_spheres; ++s) {
      visualization_msgs::msg::Marker marker;
      marker.header = msg.header;
      marker.header.frame_id = "base_link";
      marker.ns = "collision_spheres";
      marker.id = static_cast<int>(s);
      marker.type = marker.SPHERE;
      marker.action = marker.ADD;
      marker.lifetime = rclcpp::Duration::from_seconds(.5);
      const auto center = vamp_node::piperFingerSphereCenter(s,
        Eigen::Vector3f(spheres.x[{s, 0}], spheres.y[{s, 0}], spheres.z[{s, 0}]),
        pose, opening);
      marker.pose.position.x = center.x();
      marker.pose.position.y = center.y();
      marker.pose.position.z = center.z();
      marker.pose.orientation.w = 1.;
      marker.scale.x = marker.scale.y = marker.scale.z = 2.*spheres.r[{s, 0}];
      marker.color.a = .35;
      marker.color.r = s >= 74 && s < 82 ? 1.F : .15F;
      marker.color.g = s >= 82 ? 1.F : .25F;
      marker.color.b = s < 74 ? 1.F : .15F;
      if (s == 80) { marker.color.r = 1.F; marker.color.g = .1F; marker.color.b = .7F; }
      output.markers.push_back(marker);
      marker.ns = "sphere_indices";
      marker.type = marker.TEXT_VIEW_FACING;
      marker.pose.position.z += spheres.r[{s, 0}];
      marker.scale.z = .009;
      marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.F;
      marker.text = std::to_string(s);
      output.markers.push_back(marker);
    }
    pub_->publish(output);
  }
  float opening_ = .07F;
  rclcpp::Time grip_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joints_;
  rclcpp::Subscription<agx_arm_msgs::msg::GripperStatus>::SharedPtr gripper_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CollisionSpheresViewer>());
  rclcpp::shutdown();
}
