"""Review accepted grasp fingers, actual tool axes and angle-tuning status."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_directory('vamp_node')
    return LaunchDescription([
        DeclareLaunchArgument('fixed_frame', default_value='base_link'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(
            share, 'rviz', 'grasp_angles.rviz')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                share, 'launch', 'grasp_debug_rviz.launch.py')),
            launch_arguments={
                'fixed_frame': LaunchConfiguration('fixed_frame'),
                'rviz_config': LaunchConfiguration('rviz_config'),
            }.items()),
    ])
