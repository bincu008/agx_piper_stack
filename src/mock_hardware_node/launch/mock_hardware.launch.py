"""Launch recorded RGB-D and arm feedback publishers."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'snapshot_directory',
            default_value=PathJoinSubstitution([
                FindPackageShare('mock_hardware_node'),
                'snapshots', 'current',
            ])),
        DeclareLaunchArgument('topic_prefix', default_value=''),
        DeclareLaunchArgument('allow_hardware_conflict', default_value='false'),
        Node(
            package='mock_hardware_node',
            executable='mock_hardware',
            name='mock_hardware',
            output='screen',
            parameters=[{
                'snapshot_directory': LaunchConfiguration(
                    'snapshot_directory'),
                'topic_prefix': LaunchConfiguration('topic_prefix'),
                'allow_hardware_conflict': LaunchConfiguration(
                    'allow_hardware_conflict'),
            }],
        ),
    ])
