"""Open the focused grasp-and-placement RViz review configuration."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("vamp_node"), "rviz", "grasp_debug.rviz")
    fixed_frame = LaunchConfiguration("fixed_frame")
    return LaunchDescription([
        DeclareLaunchArgument(
            "fixed_frame",
            default_value="base_link",
            description=(
                "RViz-only fixed frame. Use camera_color_optical_frame when "
                "running the standalone AnyPlace node without the grasp stack."
            ),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="grasp_debug_rviz",
            arguments=["-d", config, "-f", fixed_frame],
            output="screen",
        ),
    ])
