"""Start only recorded sensors and calculation nodes on an isolated ROS domain."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def configure(context):
    """No camera driver, motor controller or physical-execution switch is exposed."""
    from task_pipeline.simulator import simulator_domain
    domain = simulator_domain(LaunchConfiguration('simulator_domain_id').perform(context))
    mock = Path(get_package_share_directory('mock_hardware_node'))/'launch/mock_hardware.launch.py'
    stack = Path(get_package_share_directory('master_node'))/'launch/reasoning_stack.launch.py'
    viewer = Path(get_package_share_directory('vamp_node'))/'launch/grasp_debug_rviz.launch.py'
    options = {'inference_profile': LaunchConfiguration('inference_profile').perform(context),
               'allow_physical_grasp_execution': 'false', 'placement_backend': 'heightmap'}
    for name in ('api_server_url', 'model_name'):
        value = LaunchConfiguration(name).perform(context)
        if value:
            options[name] = value
    return [
        SetEnvironmentVariable('ROS_DOMAIN_ID', str(domain)),
        Node(package='task_pipeline', executable='simulator_pipeline',
             name='simulator_pipeline', output='screen'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(mock)), launch_arguments={
            'snapshot_directory': LaunchConfiguration('snapshot_directory'),
            'allow_hardware_conflict': 'false'}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(stack)),
                                 launch_arguments=options.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(viewer)),
                                 condition=IfCondition(LaunchConfiguration('rviz'))),
    ]


def generate_launch_description():
    """Use portable installed scene paths and the existing inference configuration."""
    snapshot = Path(get_package_share_directory('mock_hardware_node'))/'snapshots/current'
    return LaunchDescription([
        DeclareLaunchArgument('simulator_domain_id', default_value='89'),
        DeclareLaunchArgument('snapshot_directory', default_value=str(snapshot)),
        DeclareLaunchArgument('inference_profile', default_value='lmstudio'),
        DeclareLaunchArgument('api_server_url', default_value='http://127.0.0.1:1234/v1'),
        DeclareLaunchArgument('model_name', default_value='robobrain2.5-4b'),
        DeclareLaunchArgument('rviz', default_value='false'),
        OpaqueFunction(function=configure),
    ])
