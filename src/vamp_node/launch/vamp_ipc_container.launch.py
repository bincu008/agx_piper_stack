import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    container_name = 'vamp_ipc_container'
    
    # 1. Component Container with Zero-Copy IPC enabled
    container = ComposableNodeContainer(
        name=container_name,
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='vision_node',
                plugin='VisionServer',
                name='vision_server',
                namespace='',
                parameters=[{'use_intra_process_comms': True}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='vamp_node',
                plugin='VampServer',
                name='vamp_server',
                namespace='',
                parameters=[{'use_intra_process_comms': True}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )

    # 2. Include Orbbec Camera launch file, configured to attach into our IPC container
    orbbec_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('orbbec_camera'),
                'launch',
                'orbbec_camera.launch.py'
            ])
        ]),
        launch_arguments={
            'use_intra_process_comms': 'true',
            'attach_component_container_enable': 'true',
            'attach_component_container_name': container_name,
        }.items()
    )

    return LaunchDescription([
        container,
        orbbec_camera_launch,
    ])
