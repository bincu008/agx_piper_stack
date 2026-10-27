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
            "rviz_config", default_value=config,
            description="RViz display configuration (visualization only)."),
        DeclareLaunchArgument(
            "fixed_frame",
            default_value="base_link",
            description=(
                "Visualization-only fixed frame; use base_link for the arm stack."
            ),
        ),
        Node(
            package="vamp_node",
            executable="collision_spheres_viewer",
            output="screen",
        ),
        # Visualization-only identity child for RViz frame discovery.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="grasp_debug_reference",
            arguments=["--frame-id", "base_link",
                       "--child-frame-id", "collision_spheres_view_origin"],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="grasp_debug_rviz",
            arguments=["-d", LaunchConfiguration("rviz_config"), "-f", fixed_frame],
            output="screen",
        ),
    ])
