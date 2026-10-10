"""Launch the composed robot core without camera or arm hardware drivers."""

import os


from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
import yaml


def generate_launch_description():
    profile_path = os.path.join(get_package_share_directory('master_node'),
                                'config', 'inference', 'profiles.yaml')
    with open(profile_path, encoding='utf-8') as stream:
        profiles = yaml.safe_load(stream)
    inference_profile = LaunchConfiguration('inference_profile')
    numeric_environment = {
        name: LaunchConfiguration('numeric_threads')
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}

    def profile_default(key):
        values = {name: str(value[key]) for name, value in profiles.items()}
        return PythonExpression([repr(values), "['", inference_profile, "']"])

    self_filter_config = os.path.join(
        get_package_share_directory('vision_node'), 'config',
        'gripper_self_filter.yaml')
    default_calibration_path = os.path.join(
        get_package_share_directory('master_node'), 'config', 'calibration',
        'hand_eye_calibration.yaml')
    hand_eye_calibration_path = LaunchConfiguration(
        'hand_eye_calibration_path')
    pipeline_profiling = LaunchConfiguration('pipeline_profiling')
    api_server_ip = LaunchConfiguration('api_server_ip')
    api_server_port = LaunchConfiguration('api_server_port')
    api_server_url = LaunchConfiguration('api_server_url')
    model_name = LaunchConfiguration('model_name')
    request_timeout_sec = LaunchConfiguration('request_timeout_sec')
    vlm_max_tokens = LaunchConfiguration('vlm_max_tokens')
    vlm_temperature = LaunchConfiguration('vlm_temperature')
    vlm_image_max_dimension = LaunchConfiguration('vlm_image_max_dimension')
    grasp_plan_cache_ttl_sec = LaunchConfiguration('grasp_plan_cache_ttl_sec')
    grasp_tcp_offset_m = LaunchConfiguration('grasp_tcp_offset_m')
    grasp_observation_cache_size = LaunchConfiguration(
        'grasp_observation_cache_size')
    maximum_ranking_pool_candidates = LaunchConfiguration(
        'maximum_ranking_pool_candidates')
    natural_wide_opening_weight = LaunchConfiguration(
        'natural_wide_opening_weight')
    enable_grasp_path_preplanning = LaunchConfiguration(
        'enable_grasp_path_preplanning')
    grasp_path_length_weight = LaunchConfiguration(
        'grasp_path_length_weight')
    require_measured_antipodal_contacts = LaunchConfiguration(
        'require_measured_antipodal_contacts')
    antipodal_normal_cosine = LaunchConfiguration(
        'antipodal_normal_cosine')
    antipodal_normal_radius_m = LaunchConfiguration(
        'antipodal_normal_radius_m')
    antipodal_min_normal_neighbors = LaunchConfiguration(
        'antipodal_min_normal_neighbors')
    grasp_contact_depth_m = LaunchConfiguration('grasp_contact_depth_m')
    target_collision_clearance_m = LaunchConfiguration(
        'target_collision_clearance_m')
    aligned_zero_pose_yaw_limit_degrees = LaunchConfiguration(
        'aligned_zero_pose_yaw_limit_degrees')
    aligned_zero_pose_pitch_limit_degrees = LaunchConfiguration(
        'aligned_zero_pose_pitch_limit_degrees')
    aligned_zero_pose_roll_limit_degrees = LaunchConfiguration(
        'aligned_zero_pose_roll_limit_degrees')
    handle_approach_roll_limit_degrees = LaunchConfiguration(
        'handle_approach_roll_limit_degrees')
    handle_approach_roll_step_degrees = LaunchConfiguration(
        'handle_approach_roll_step_degrees')
    handle_visible_approach_min_cosine = LaunchConfiguration(
        'handle_visible_approach_min_cosine')
    handle_approach_dedup_degrees = LaunchConfiguration(
        'handle_approach_dedup_degrees')
    silhouette_min_tangent_anisotropy = LaunchConfiguration(
        'silhouette_min_tangent_anisotropy')
    pregrasp_offset_m = LaunchConfiguration('pregrasp_offset_m')
    pregrasp_orientation_tolerance_degrees = LaunchConfiguration(
        'pregrasp_orientation_tolerance_degrees')
    pregrasp_position_tolerance_m = LaunchConfiguration(
        'pregrasp_position_tolerance_m')
    pregrasp_opening_clearance_m = LaunchConfiguration(
        'pregrasp_opening_clearance_m')
    target_contact_surface_tolerance_m = LaunchConfiguration(
        'target_contact_surface_tolerance_m')
    pinch_max_opening_m = LaunchConfiguration('pinch_max_opening_m')
    pinch_min_free_gap_m = LaunchConfiguration('pinch_min_free_gap_m')
    grasp_candidate_backend = LaunchConfiguration('grasp_candidate_backend')
    hggd_model_directory = LaunchConfiguration('hggd_model_directory')
    hggd_cpu_threads = LaunchConfiguration('hggd_cpu_threads')
    hggd_maximum_proposals = LaunchConfiguration('hggd_maximum_proposals')
    maximum_candidates_to_test = LaunchConfiguration('maximum_candidates_to_test')
    maximum_valid_candidates = LaunchConfiguration('maximum_valid_candidates')
    grasp_visualization_limit = LaunchConfiguration('grasp_visualization_limit')
    hggd_diversity_position_m = LaunchConfiguration('hggd_diversity_position_m')
    hggd_diversity_approach_degrees = LaunchConfiguration(
        'hggd_diversity_approach_degrees')
    hggd_diversity_closing_degrees = LaunchConfiguration(
        'hggd_diversity_closing_degrees')
    deterministic_ik = LaunchConfiguration('deterministic_ik')
    deterministic_ik_seed_count = LaunchConfiguration('deterministic_ik_seed_count')
    final_approach_translation_step_m = LaunchConfiguration(
        'final_approach_translation_step_m')
    final_approach_max_joint_step_rad = LaunchConfiguration(
        'final_approach_max_joint_step_rad')
    final_approach_contact_phase_m = LaunchConfiguration(
        'final_approach_contact_phase_m')
    allow_physical_grasp_execution = LaunchConfiguration(
        'allow_physical_grasp_execution')
    allow_model_based_body_grasp_execution = LaunchConfiguration(
        'allow_model_based_body_grasp_execution')
    require_positive_grasp_score_for_grasp = LaunchConfiguration(
        'require_positive_grasp_score_for_grasp')
    require_measured_contacts_for_grasp = LaunchConfiguration(
        'require_measured_contacts_for_grasp')
    enable_view_repositioning = LaunchConfiguration(
        'enable_view_repositioning')
    enable_grasp_collision_check = LaunchConfiguration(
        'enable_grasp_collision_check')
    allow_pregrasp_collision_override = LaunchConfiguration(
        'allow_pregrasp_collision_override')
    grasp_allowed_contact_radius_m = LaunchConfiguration(
        'grasp_allowed_contact_radius_m')
    pregrasp_target_exclusion_radius_m = LaunchConfiguration(
        'pregrasp_target_exclusion_radius_m')
    gripper_open_force_n = LaunchConfiguration('gripper_open_force_n')
    gripper_close_force_n = LaunchConfiguration('gripper_close_force_n')
    gripper_close_compression_m = LaunchConfiguration(
        'gripper_close_compression_m')
    post_grasp_lift_m = LaunchConfiguration('post_grasp_lift_m')
    gripper_min_open_feedback_m = LaunchConfiguration(
        'gripper_min_open_feedback_m')
    world_perception_backend = LaunchConfiguration(
        'world_perception_backend')
    grasp_target_mask_backend = LaunchConfiguration(
        'grasp_target_mask_backend')
    use_yolo_world_perception = PythonExpression([
        "'", world_perception_backend, "'.lower() == 'yolo'",
    ])
    core_container = ComposableNodeContainer(
        name='robot_core_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='vision_node',
                plugin='vision_node::ActiveVisionServer',
                name='active_vision_server',
                parameters=[self_filter_config, {
                    'hand_eye_calibration_path': hand_eye_calibration_path,
                    'enable_yolo_inference': ParameterValue(
                        use_yolo_world_perception, value_type=bool),
                    'publish_yolo_world_observations': ParameterValue(
                        use_yolo_world_perception, value_type=bool),
                    'yolo_world_classes': ['cup'],
                }],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='vamp_node',
                plugin='vamp_node::VampPlannerServer',
                name='vamp_planner_server',
                parameters=[{
                    'placement_fast_ik': ParameterValue(
                        LaunchConfiguration('placement_fast_ik'), value_type=bool),
                    'placement_position_tolerance_m': ParameterValue(
                        LaunchConfiguration('placement_position_tolerance_m'),
                        value_type=float),
                    'placement_descent_orientation_tolerance_rad': ParameterValue(
                        LaunchConfiguration('placement_descent_orientation_tolerance_rad'),
                        value_type=float),
                    'placement_descent_step_m': ParameterValue(
                        LaunchConfiguration('placement_descent_step_m'), value_type=float),
                    'placement_descent_max_joint_step_rad': ParameterValue(
                        LaunchConfiguration('placement_descent_max_joint_step_rad'),
                        value_type=float),
                    'placement_contact_distance_m': ParameterValue(
                        LaunchConfiguration('placement_contact_distance_m'), value_type=float),
                    'grasp_tcp_offset_m': ParameterValue(
                        grasp_tcp_offset_m, value_type=float
                    ),
                    'grasp_observation_cache_size': ParameterValue(
                        grasp_observation_cache_size, value_type=int),
                    'maximum_ranking_pool_candidates': ParameterValue(
                        maximum_ranking_pool_candidates, value_type=int),
                    'natural_wide_opening_weight': ParameterValue(
                        natural_wide_opening_weight, value_type=float),
                    'pipeline_profiling': ParameterValue(pipeline_profiling, value_type=bool),
                    'placement_allow_separating_contact_departure': ParameterValue(
                        LaunchConfiguration('placement_allow_separating_contact_departure'),
                        value_type=bool),
                    'placement_bypass_attached_collision_during_lift': ParameterValue(
                        LaunchConfiguration('placement_bypass_attached_collision_during_lift'),
                        value_type=bool),
                    'placement_departure_contact_tolerance_m': ParameterValue(
                        LaunchConfiguration('placement_departure_contact_tolerance_m'),
                        value_type=float),
                    'placement_noise_max_points': ParameterValue(
                        LaunchConfiguration('placement_noise_max_points'), value_type=int),
                    'placement_noise_max_voxels': ParameterValue(
                        LaunchConfiguration('placement_noise_max_voxels'), value_type=int),
                    'enable_grasp_path_preplanning': ParameterValue(
                        enable_grasp_path_preplanning, value_type=bool),
                    'grasp_path_length_weight': ParameterValue(
                        grasp_path_length_weight, value_type=float),
                    'require_measured_antipodal_contacts': ParameterValue(
                        require_measured_antipodal_contacts, value_type=bool),
                    'antipodal_normal_cosine': ParameterValue(
                        antipodal_normal_cosine, value_type=float),
                    'antipodal_normal_radius_m': ParameterValue(
                        antipodal_normal_radius_m, value_type=float),
                    'antipodal_min_normal_neighbors': ParameterValue(
                        antipodal_min_normal_neighbors, value_type=int),
                    'grasp_contact_depth_m': ParameterValue(
                        grasp_contact_depth_m, value_type=float),
                    'target_collision_clearance_m': ParameterValue(
                        target_collision_clearance_m, value_type=float),
                    'aligned_zero_pose_yaw_limit_degrees': ParameterValue(
                        aligned_zero_pose_yaw_limit_degrees,
                        value_type=float),
                    'aligned_zero_pose_pitch_limit_degrees': ParameterValue(
                        aligned_zero_pose_pitch_limit_degrees,
                        value_type=float),
                    'aligned_zero_pose_roll_limit_degrees': ParameterValue(
                        aligned_zero_pose_roll_limit_degrees,
                        value_type=float),
                    'handle_approach_roll_limit_degrees': ParameterValue(
                        handle_approach_roll_limit_degrees, value_type=float),
                    'handle_approach_roll_step_degrees': ParameterValue(
                        handle_approach_roll_step_degrees, value_type=float),
                    'handle_visible_approach_min_cosine': ParameterValue(
                        handle_visible_approach_min_cosine, value_type=float),
                    'handle_approach_dedup_degrees': ParameterValue(
                        handle_approach_dedup_degrees, value_type=float),
                    'silhouette_min_tangent_anisotropy': ParameterValue(
                        silhouette_min_tangent_anisotropy, value_type=float),
                    'pregrasp_offset_m': ParameterValue(
                        pregrasp_offset_m, value_type=float),
                    'pregrasp_orientation_tolerance_degrees': ParameterValue(
                        pregrasp_orientation_tolerance_degrees,
                        value_type=float),
                    'pregrasp_position_tolerance_m': ParameterValue(
                        pregrasp_position_tolerance_m,
                        value_type=float),
                    'pregrasp_opening_clearance_m': ParameterValue(
                        pregrasp_opening_clearance_m, value_type=float),
                    'target_contact_surface_tolerance_m': ParameterValue(
                        target_contact_surface_tolerance_m, value_type=float),
                    'pinch_max_opening_m': ParameterValue(
                        pinch_max_opening_m, value_type=float),
                    'pinch_min_free_gap_m': ParameterValue(
                        pinch_min_free_gap_m, value_type=float),
                    'grasp_candidate_backend': grasp_candidate_backend,
                    'hggd_model_directory': hggd_model_directory,
                    'hggd_cpu_threads': ParameterValue(
                        hggd_cpu_threads, value_type=int),
                    'hggd_maximum_proposals': ParameterValue(
                        hggd_maximum_proposals, value_type=int),
                    'maximum_candidates_to_test': ParameterValue(
                        maximum_candidates_to_test, value_type=int),
                    'maximum_valid_candidates': ParameterValue(
                        maximum_valid_candidates, value_type=int),
                    'grasp_visualization_limit': ParameterValue(
                        grasp_visualization_limit, value_type=int),
                    'hggd_diversity_position_m': ParameterValue(
                        hggd_diversity_position_m, value_type=float),
                    'hggd_diversity_approach_degrees': ParameterValue(
                        hggd_diversity_approach_degrees, value_type=float),
                    'hggd_diversity_closing_degrees': ParameterValue(
                        hggd_diversity_closing_degrees, value_type=float),
                    'deterministic_ik': ParameterValue(
                        deterministic_ik, value_type=bool),
                    'deterministic_ik_seed_count': ParameterValue(
                        deterministic_ik_seed_count, value_type=int),
                    'final_approach_translation_step_m': ParameterValue(
                        final_approach_translation_step_m, value_type=float),
                    'final_approach_max_joint_step_rad': ParameterValue(
                        final_approach_max_joint_step_rad, value_type=float),
                    'final_approach_contact_phase_m': ParameterValue(
                        final_approach_contact_phase_m, value_type=float),
                    'enable_grasp_collision_check': ParameterValue(
                        enable_grasp_collision_check, value_type=bool),
                    'allow_pregrasp_collision_candidates': ParameterValue(
                        allow_pregrasp_collision_override, value_type=bool),
                }],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )

    # Keep network waits in a separate executor from perception and VAMP.
    vlm = Node(
        package='vlm_node', executable='vlm_reasoning_node',
        name='vlm_reasoning_node', output='screen',
        parameters=[{
            'placement_support_mode': LaunchConfiguration('layout_support_mode'),
            'api_server_ip': api_server_ip,
            'api_server_port': ParameterValue(
                api_server_port, value_type=int
            ),
            'api_server_url': api_server_url,
            'model_name': model_name,
            'request_timeout_sec': ParameterValue(
                request_timeout_sec, value_type=int
            ),
            'max_tokens': ParameterValue(
                vlm_max_tokens, value_type=int),
            'temperature': ParameterValue(
                vlm_temperature, value_type=float),
            'image_max_dimension': ParameterValue(
                vlm_image_max_dimension, value_type=int),
            'grasp_plan_cache_ttl_sec': ParameterValue(
                grasp_plan_cache_ttl_sec, value_type=int
            ),
            'enable_task_dispatch': False,
            'verbose': True,
            'world_perception_backend': world_perception_backend,
            'grasp_target_mask_backend': grasp_target_mask_backend,
            'enable_view_repositioning': ParameterValue(
                enable_view_repositioning, value_type=bool),
        }],
    )

    master = Node(
        package='master_node',
        executable='piper_master_control',
        additional_env=numeric_environment,
        name='piper_master_control',
        output='screen',
        parameters=[{
            'api_dispatch_in_master': ParameterValue(
                LaunchConfiguration('api_dispatch_in_master'), value_type=bool),
            'hand_eye_calibration_path': hand_eye_calibration_path,
            'allow_physical_grasp_execution': ParameterValue(
                allow_physical_grasp_execution, value_type=bool),
            'allow_model_based_body_grasp_execution': ParameterValue(
                allow_model_based_body_grasp_execution, value_type=bool),
            'require_positive_grasp_score_for_grasp': ParameterValue(
                require_positive_grasp_score_for_grasp, value_type=bool),
            'require_measured_contacts_for_grasp': ParameterValue(
                require_measured_contacts_for_grasp, value_type=bool),
            'allow_pregrasp_collision_override': ParameterValue(
                allow_pregrasp_collision_override, value_type=bool),
            'grasp_allowed_contact_radius_m': ParameterValue(
                grasp_allowed_contact_radius_m, value_type=float),
            'pregrasp_target_exclusion_radius_m': ParameterValue(
                pregrasp_target_exclusion_radius_m, value_type=float),
            'gripper_open_force_n': ParameterValue(
                gripper_open_force_n, value_type=float),
            'gripper_close_force_n': ParameterValue(
                gripper_close_force_n, value_type=float),
            'gripper_close_compression_m': ParameterValue(
                gripper_close_compression_m, value_type=float),
            'post_grasp_lift_m': ParameterValue(
                post_grasp_lift_m, value_type=float),
            'gripper_min_open_feedback_m': ParameterValue(
                gripper_min_open_feedback_m, value_type=float),
            'enable_camera_collision_monitor': False,
        }],
    )

    pipeline = Node(
        package='task_pipeline', executable='pipeline_node',
        name='pipeline_node', output='screen',
        condition=UnlessCondition(LaunchConfiguration('api_dispatch_in_master')),
    )
    placement = Node(
        package='placement_node',
        executable='placement_node',
        additional_env=numeric_environment,
        name='placement_node',
        output='screen',
        parameters=[{
            'hand_eye_calibration_path': hand_eye_calibration_path,
            'api_server_url': api_server_url,
            'model_name': model_name,
            'request_timeout_sec': ParameterValue(
                request_timeout_sec, value_type=float),
            'placement_backend': LaunchConfiguration('placement_backend'),
            'layout_support_mode': LaunchConfiguration('layout_support_mode'),
            'floor_slot_adaptive': ParameterValue(
                LaunchConfiguration('floor_slot_adaptive'), value_type=bool),
            'floor_slot_max_size_m': ParameterValue(
                LaunchConfiguration('floor_slot_max_size_m'), value_type=float),
            'floor_slot_min_size_m': ParameterValue(
                LaunchConfiguration('floor_slot_min_size_m'), value_type=float),
            'floor_slot_share_unused_space': ParameterValue(
                LaunchConfiguration('floor_slot_share_unused_space'), value_type=bool),
            'layout_candidate_count': ParameterValue(
                LaunchConfiguration('layout_candidate_count'), value_type=int),
            'packing_stack_minimum_coverage': ParameterValue(
                LaunchConfiguration('stack_minimum_coverage'), value_type=float),
            'candidate_count': ParameterValue(
                LaunchConfiguration('anyplace_candidate_count'),
                value_type=int),
            'refinement_steps': ParameterValue(
                LaunchConfiguration('anyplace_refinement_steps'),
                value_type=int),
            'checkpoint_path': LaunchConfiguration(
                'anyplace_checkpoint_path'),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'hand_eye_calibration_path',
            default_value=default_calibration_path,
            description=(
                'Camera-to-tool X-matrix YAML shared by perception, master, '
                'and placement nodes.'
            ),
        ),
        DeclareLaunchArgument('pipeline_profiling', default_value='false',
                              description='Log per-request IK, collision and engine timings.'),
        DeclareLaunchArgument('api_dispatch_in_master', default_value='true',
                              description='Host API dispatch in the master; '
                                          'false restores RPC proxy.'),
        DeclareLaunchArgument('numeric_threads', default_value='1',
                              description='BLAS/OpenMP threads per Python planning process.'),
        DeclareLaunchArgument('inference_profile', default_value='server',
                              choices=list(profiles),
                              description='Select server or lmstudio endpoint defaults.'),
        DeclareLaunchArgument(
            'api_server_ip', default_value=profile_default('api_server_ip'),
            description='Inference host; overrides the selected profile.'),
        DeclareLaunchArgument(
            'api_server_port', default_value=profile_default('api_server_port')),
        DeclareLaunchArgument(
            'api_server_url',
            default_value=[
                'http://', api_server_ip, ':', api_server_port, '/v1'],
            description='OpenAI-compatible RoboBrain API base URL.'),
        DeclareLaunchArgument('model_name', default_value=profile_default('model_name')),
        DeclareLaunchArgument(
            'request_timeout_sec', default_value=profile_default('request_timeout_sec')),
        DeclareLaunchArgument('vlm_max_tokens', default_value='256'),
        DeclareLaunchArgument('vlm_temperature', default_value='0.0'),
        DeclareLaunchArgument('vlm_image_max_dimension', default_value='1024'),
        DeclareLaunchArgument('stack_minimum_coverage', default_value='0.6'),
        DeclareLaunchArgument('placement_candidate_count', default_value='16',
                              description='Ranked single-object placement pose budget'),
        DeclareLaunchArgument('anyplace_candidate_count',
                              default_value=LaunchConfiguration('placement_candidate_count'),
                              description='Legacy alias for placement_candidate_count'),
        DeclareLaunchArgument('layout_support_mode', default_value='floor_slots'),
        DeclareLaunchArgument('floor_slot_share_unused_space', default_value='false'),
        DeclareLaunchArgument('floor_slot_adaptive', default_value='true'),
        DeclareLaunchArgument('floor_slot_max_size_m', default_value='0.125'),
        DeclareLaunchArgument('floor_slot_min_size_m', default_value='0.06'),
        DeclareLaunchArgument('layout_candidate_count', default_value='16'),
        DeclareLaunchArgument('placement_fast_ik', default_value='true'),
        DeclareLaunchArgument('placement_position_tolerance_m', default_value='0.001'),
        DeclareLaunchArgument('placement_descent_orientation_tolerance_rad', default_value='0.01'),
        DeclareLaunchArgument('placement_descent_step_m', default_value='0.002'),
        DeclareLaunchArgument('placement_descent_max_joint_step_rad', default_value='0.05'),
        DeclareLaunchArgument('placement_contact_distance_m', default_value='0.006'),
        DeclareLaunchArgument('anyplace_refinement_steps', default_value='50'),
        DeclareLaunchArgument(
            'placement_backend',
            default_value='heightmap',
            description=(
                "Placement candidate backend: 'heightmap' for deterministic "
                "packing or 'anyplace' for the retained diffusion model."
            ),
        ),
        DeclareLaunchArgument(
            'anyplace_checkpoint_path',
            default_value='',
            description=(
                'Optional AnyPlace checkpoint. Not used by the default '
                'height-map placement backend.'
            )),
        DeclareLaunchArgument(
            'grasp_plan_cache_ttl_sec',
            default_value='60',
            description=(
                'Maximum age of the RViz-reviewed #1 grasp that [GRASP] may '
                'execute; valid range is 1-60 seconds.'
            ),
        ),
        DeclareLaunchArgument(
            'world_perception_backend',
            default_value='robobrain',
            description=(
                "World inventory backend: 'yolo' for fast asynchronous "
                "instance perception, or 'robobrain' for VLM perception."
            ),
        ),
        DeclareLaunchArgument(
            'grasp_target_mask_backend',
            default_value='edgesam',
            choices=['edgesam'],
            description='Target mask: RoboBrain grounding followed by EdgeSAM.',
        ),
        DeclareLaunchArgument(
            'allow_physical_grasp_execution',
            default_value='false',
            description=(
                'Publish pre-grasp, grasp, gripper, and retreat commands. '
                'Leave false until RViz validation is complete.'
            ),
        ),
        DeclareLaunchArgument(
            'allow_model_based_body_grasp_execution',
            default_value='false',
            description=(
                'Permit only collision-checked CUP_BODY_MODEL candidates to '
                'execute. This does not relax measured-contact checks for '
                'other grasp types.'
            ),
        ),
        DeclareLaunchArgument(
            'require_positive_grasp_score_for_grasp',
            default_value='true',
            description=(
                'Require the legacy GPD classifier decision for physical '
                'execution. Set false only when measured-contact, IK, and '
                'collision validation are enabled.'
            ),
        ),
        DeclareLaunchArgument(
            'require_measured_contacts_for_grasp',
            default_value='true',
            description=(
                'Require a measured opposing contact pair and clear finger '
                'sweep before physical execution.'
            ),
        ),
        DeclareLaunchArgument(
            'enable_view_repositioning',
            default_value='false',
            description=(
                'Permit an off-screen GRAB request to move only to a '
                'calibrated NBV before grasp planning.'
            ),
        ),
        DeclareLaunchArgument(
            'enable_grasp_collision_check',
            default_value='true',
            description=(
                'Endpoint collision validation for grasp candidates. '
                'Physical GRAB remains rejected unless this is true.'
            ),
        ),
        DeclareLaunchArgument(
            'allow_pregrasp_collision_override',
            default_value='false',
            description=(
                'Retain a grasp-safe candidate whose pre-grasp endpoint '
                'intersects only the explicit local target-contact volume. '
                'All other VAMP collision checks remain blocking.'
            ),
        ),
        DeclareLaunchArgument(
            'grasp_allowed_contact_radius_m',
            default_value='0.035',
            description=(
                'Radius around the selected handle contact center excluded '
                'from VAMP only during grasp approach/retreat.'
            ),
        ),
        DeclareLaunchArgument(
            'pregrasp_target_exclusion_radius_m',
            default_value='0.0',
            description=(
                'Disabled by default. When positive, excludes only the '
                'selected target within this radius while planning to the '
                'pre-grasp pose. The robot geometry and all other obstacles '
                'remain collision checked.'
            ),
        ),
        DeclareLaunchArgument(
            'gripper_open_force_n',
            default_value='1.0',
            description='Piper gripper opening force in the driver range 0.5-3.0 N.',
        ),
        DeclareLaunchArgument(
            'gripper_close_force_n',
            default_value='1.0',
            description='Piper gripper closing force in the driver range 0.5-3.0 N.',
        ),
        DeclareLaunchArgument(
            'gripper_close_compression_m',
            default_value='0.003',
            description=(
                'Distance commanded below the measured target thickness to '
                'establish contact; replaces the unsafe fixed 20 mm closure.'
            ),
        ),
        DeclareLaunchArgument(
            'post_grasp_lift_m',
            default_value='0.10',
            description=(
                'Base-frame vertical lift after gripper contact is confirmed; '
                'VAMP pose IK and path validation are required.'
            ),
        ),
        DeclareLaunchArgument(
            'gripper_min_open_feedback_m',
            default_value='0.01',
            description=(
                'Minimum gripper feedback opening considered not closed; '
                'used when the absolute width encoder is uncalibrated.'
            ),
        ),
        DeclareLaunchArgument(
            'grasp_candidate_backend', default_value='hggd', choices=['hggd'],
            description='Deployed proposal engine: HGGD.'),
        DeclareLaunchArgument('hggd_model_directory', default_value=''),
        DeclareLaunchArgument('placement_allow_separating_contact_departure',
                              default_value='true'),
        DeclareLaunchArgument('placement_departure_contact_tolerance_m',
                              default_value='0.001'),
        DeclareLaunchArgument('placement_bypass_attached_collision_during_lift',
                              default_value='true', description='Skip held-object '
                              'environment collision during lift only; '
                              'all other stages retain checks.'),
        DeclareLaunchArgument('placement_noise_max_points', default_value='32'),
        DeclareLaunchArgument('placement_noise_max_voxels', default_value='10'),
        DeclareLaunchArgument('hggd_cpu_threads', default_value='4'),
        DeclareLaunchArgument('hggd_maximum_proposals', default_value='100'),
        DeclareLaunchArgument('maximum_candidates_to_test', default_value='600'),
        DeclareLaunchArgument('maximum_valid_candidates', default_value='100'),
        DeclareLaunchArgument('grasp_visualization_limit', default_value='100'),
        DeclareLaunchArgument('hggd_diversity_position_m', default_value='0.030'),
        DeclareLaunchArgument('hggd_diversity_approach_degrees', default_value='20.0'),
        DeclareLaunchArgument('hggd_diversity_closing_degrees', default_value='30.0'),
        DeclareLaunchArgument('deterministic_ik', default_value='true'),
        DeclareLaunchArgument('deterministic_ik_seed_count', default_value='32'),
        DeclareLaunchArgument(
            'final_approach_translation_step_m', default_value='0.002',
            description=(
                'Maximum Cartesian translation per final-approach waypoint.'
            ),
        ),
        DeclareLaunchArgument(
            'final_approach_max_joint_step_rad', default_value='0.08',
            description=(
                'Maximum change of any joint between adjacent waypoints.'
            ),
        ),
        DeclareLaunchArgument(
            'final_approach_contact_phase_m', default_value='0.006',
            description=(
                'Final distance where only intended inner-pad target contact '
                'may be excluded from collision rejection.'
            ),
        ),
        DeclareLaunchArgument(
            'grasp_tcp_offset_m',
            default_value='-0.0615',
            description=(
                "Offset from GPD's finger-region base to Piper gripper_base "
                'along the grasp approach axis.'
            ),
        ),
        DeclareLaunchArgument(
            'grasp_observation_cache_size',
            default_value='20',
            description=(
                'Number of synchronized clouds retained while RoboBrain '
                'prepares a grasp request.'
            ),
        ),
        DeclareLaunchArgument(
            'maximum_ranking_pool_candidates',
            default_value='300',
            description='Maximum valid candidates included in dynamic ranking.',
        ),
        DeclareLaunchArgument(
            'natural_wide_opening_weight',
            default_value='1.0',
            description='Weight of the valid-pool-relative jaw-width penalty.',
        ),
        DeclareLaunchArgument(
            'enable_grasp_path_preplanning',
            default_value='true',
            description=(
                'Plan collision-free paths to all retained HGGD pre-grasps '
                'before final ranking.'
            ),
        ),
        DeclareLaunchArgument(
            'grasp_path_length_weight',
            default_value='0.5',
            description='Penalty weight for normalized pre-grasp path length.',
        ),
        DeclareLaunchArgument(
            'require_measured_antipodal_contacts',
            default_value='true',
            description=(
                'Require opposing measured target-cloud contacts and clear '
                'finger closing volume.'
            ),
        ),
        DeclareLaunchArgument(
            'antipodal_normal_cosine',
            default_value='0.90',
            description='Minimum contact-normal alignment (0.90 is about 26 degrees).',
        ),
        DeclareLaunchArgument(
            'antipodal_normal_radius_m',
            default_value='0.008',
            description='Radius used to estimate each measured contact normal.',
        ),
        DeclareLaunchArgument(
            'antipodal_min_normal_neighbors',
            default_value='10',
            description='Minimum local points required for a measured contact normal.',
        ),
        DeclareLaunchArgument(
            'grasp_contact_depth_m',
            default_value='0.070',
            description='Contact plane depth from the 70 mm finger base.',
        ),
        DeclareLaunchArgument(
            'target_collision_clearance_m',
            default_value='0.050',
            description='Radius around measured contacts where inner jaw pads may touch '
                        'the segmented target.',
        ),
        DeclareLaunchArgument(
            'aligned_zero_pose_yaw_limit_degrees',
            default_value='10.0',
            description='Maximum zero-pose approach azimuth change.',
        ),
        DeclareLaunchArgument(
            'aligned_zero_pose_pitch_limit_degrees',
            default_value='90.0',
            description='Maximum zero-pose approach elevation change.',
        ),
        DeclareLaunchArgument(
            'aligned_zero_pose_roll_limit_degrees',
            default_value='90.0',
            description='Maximum zero-pose jaw rotation about its approach.',
        ),
        DeclareLaunchArgument(
            'handle_approach_roll_limit_degrees',
            default_value='100.0',
            description='Maximum jaw roll about each candidate approach axis.',
        ),
        DeclareLaunchArgument(
            'handle_approach_roll_step_degrees',
            default_value='10.0',
            description='Angular spacing of handle jaw-roll hypotheses.',
        ),
        DeclareLaunchArgument(
            'handle_visible_approach_min_cosine',
            default_value='0.25',
            description='Minimum dot product with the camera-visible free-space approach.',
        ),
        DeclareLaunchArgument(
            'handle_approach_dedup_degrees',
            default_value='5.0',
            description='Minimum angular separation between handle approach variants.',
        ),
        DeclareLaunchArgument(
            'silhouette_min_tangent_anisotropy',
            default_value='1.15',
            description='Minimum local mask PCA aspect ratio for a handle-thickness trace.',
        ),
        DeclareLaunchArgument(
            'pregrasp_offset_m',
            default_value='0.120',
            description=(
                'Preferred Cartesian standoff; planner falls back across '
                'reachable, target-visible offsets from 0.05 to 0.15 m.'
            ),
        ),
        DeclareLaunchArgument(
            'pregrasp_position_tolerance_m',
            default_value='0.008',
            description=(
                'Position tolerance used only for pre-grasp IK.'
            ),
        ),
        DeclareLaunchArgument(
            'pregrasp_orientation_tolerance_degrees',
            default_value='20.0',
            description=(
                'Maximum orientation relaxation used only for pre-grasp IK.'
            ),
        ),
        DeclareLaunchArgument(
            'pregrasp_opening_clearance_m',
            default_value='0.015',
            description=(
                'Extra jaw opening used for pre-grasp and finger-sweep '
                'validation.'
            ),
        ),
        DeclareLaunchArgument(
            'target_contact_surface_tolerance_m',
            default_value='0.006',
            description=(
                'Silhouette contact registration tolerance in metres. Only '
                'points immediately beside intended contacts are tolerated; '
                'outer finger-body and scene collision checks remain active.'
            ),
        ),
        DeclareLaunchArgument(
            'pinch_max_opening_m',
            default_value='0.050',
            description=(
                'Largest geometry-measured feature width classified as a pinch.'
            ),
        ),
        DeclareLaunchArgument(
            'pinch_min_free_gap_m',
            default_value='0.013',
            description=(
                'Required free gap beside a narrow feature for a 10 mm jaw '
                'plus clearance.'
            ),
        ),
        core_container,
        vlm,
        pipeline,
        master,
        placement,
    ])
