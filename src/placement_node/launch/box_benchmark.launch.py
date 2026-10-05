"""Run the calculation-only fixture beside the existing mock reasoning stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = {
        'box_count': ('10', int),
        'optimize_pipeline': ('true', bool),
        'incremental_grasp_fallback': ('false', bool),
        'grasp_early_stop_count': ('4', int),
        'placement_early_stop_count': ('32', int),
        'placement_validation_batch_size': ('8', int),
        'box_gap_m': ('0.04', float),
        'placement_policy': ('edge_then_stack', str),
        'auto_run': ('false', bool),
        'plan_pick_place_motion': ('true', bool),
        'output_directory': ('~/box_benchmark', str),
        'candidate_count': ('128', int),
        'yaw_step_deg': ('15.0', float),
        'floor_refinement_candidate_count': ('0', int),
        'floor_refinement_yaw_step_deg': ('5.0', float),
    }
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=default)
          for name, (default, _) in arguments.items()],
        Node(package='placement_node', executable='box_benchmark',
             name='box_benchmark', output='screen', parameters=[{
                 name: ParameterValue(LaunchConfiguration(name), value_type=kind)
                 for name, (_, kind) in arguments.items()}]),
    ])
