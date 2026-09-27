"""Launch the composed robot core without camera or arm hardware drivers."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    self_filter_config = os.path.join(
        get_package_share_directory('vision_node'), 'config',
        'gripper_self_filter.yaml')
    default_calibration_path = os.path.join(
        get_package_share_directory('master_node'), 'config', 'calibration',
        'hand_eye_calibration.yaml')
    hand_eye_calibration_path = LaunchConfiguration(
        'hand_eye_calibration_path')
    api_server_ip = LaunchConfiguration('api_server_ip')
    api_server_port = LaunchConfiguration('api_server_port')
    model_name = LaunchConfiguration('model_name')
    request_timeout_sec = LaunchConfiguration('request_timeout_sec')
    grasp_plan_cache_ttl_sec = LaunchConfiguration('grasp_plan_cache_ttl_sec')
    gpd_camera_frame = LaunchConfiguration('gpd_camera_frame')
    gpd_tcp_offset_m = LaunchConfiguration('gpd_tcp_offset_m')
    grasp_observation_cache_size = LaunchConfiguration(
        'grasp_observation_cache_size')
    maximum_ranking_pool_candidates = LaunchConfiguration(
        'maximum_ranking_pool_candidates')
    natural_wide_opening_weight = LaunchConfiguration(
        'natural_wide_opening_weight')
    require_gpd_full_antipodal = LaunchConfiguration(
        'require_gpd_full_antipodal')
    require_measured_antipodal_contacts = LaunchConfiguration(
        'require_measured_antipodal_contacts')
    antipodal_normal_cosine = LaunchConfiguration(
        'antipodal_normal_cosine')
    antipodal_normal_radius_m = LaunchConfiguration(
        'antipodal_normal_radius_m')
    antipodal_min_normal_neighbors = LaunchConfiguration(
        'antipodal_min_normal_neighbors')
    grasp_contact_depth_m = LaunchConfiguration('grasp_contact_depth_m')
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
    gpd_visualization_limit = LaunchConfiguration('gpd_visualization_limit')
    gpd_diversity_position_m = LaunchConfiguration('gpd_diversity_position_m')
    gpd_diversity_approach_degrees = LaunchConfiguration('gpd_diversity_approach_degrees')
    gpd_diversity_closing_degrees = LaunchConfiguration('gpd_diversity_closing_degrees')
    analytical_maximum_proposals = LaunchConfiguration('analytical_maximum_proposals')
    analytical_voxel_size_m = LaunchConfiguration('analytical_voxel_size_m')
    analytical_roll_step_degrees = LaunchConfiguration('analytical_roll_step_degrees')
    analytical_normal_radius_m = LaunchConfiguration('analytical_normal_radius_m')
    analytical_min_normal_neighbors = LaunchConfiguration('analytical_min_normal_neighbors')
    analytical_dedup_position_m = LaunchConfiguration('analytical_dedup_position_m')
    analytical_dedup_angle_degrees = LaunchConfiguration('analytical_dedup_angle_degrees')
    analytical_min_contact_center_separation_m = LaunchConfiguration(
        'analytical_min_contact_center_separation_m')
    analytical_local_pca_searches = LaunchConfiguration('analytical_local_pca_searches')
    analytical_global_pca_searches = LaunchConfiguration('analytical_global_pca_searches')
    analytical_maximum_approach_variants_per_pair = LaunchConfiguration(
        'analytical_maximum_approach_variants_per_pair')
    analytical_accepted_candidate_separation_m = LaunchConfiguration(
        'analytical_accepted_candidate_separation_m')
    deterministic_ik = LaunchConfiguration('deterministic_ik')
    deterministic_ik_seed_count = LaunchConfiguration('deterministic_ik_seed_count')
    final_approach_translation_step_m = LaunchConfiguration(
        'final_approach_translation_step_m')
    final_approach_max_joint_step_rad = LaunchConfiguration(
        'final_approach_max_joint_step_rad')
    final_approach_contact_phase_m = LaunchConfiguration(
        'final_approach_contact_phase_m')
    enable_gpd_score_filter = LaunchConfiguration(
        'enable_gpd_score_filter')
    allow_physical_grasp_execution = LaunchConfiguration(
        'allow_physical_grasp_execution')
    allow_model_based_body_grasp_execution = LaunchConfiguration(
        'allow_model_based_body_grasp_execution')
    require_positive_gpd_score_for_grasp = LaunchConfiguration(
        'require_positive_gpd_score_for_grasp')
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
                    'gpd_camera_frame': gpd_camera_frame,
                    'gpd_tcp_offset_m': ParameterValue(
                        gpd_tcp_offset_m, value_type=float
                    ),
                    'grasp_observation_cache_size': ParameterValue(
                        grasp_observation_cache_size, value_type=int),
                    'maximum_ranking_pool_candidates': ParameterValue(
                        maximum_ranking_pool_candidates, value_type=int),
                    'natural_wide_opening_weight': ParameterValue(
                        natural_wide_opening_weight, value_type=float),
                    'require_gpd_full_antipodal': ParameterValue(
                        require_gpd_full_antipodal, value_type=bool),
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
                    'gpd_visualization_limit': ParameterValue(
                        gpd_visualization_limit, value_type=int),
                    'gpd_diversity_position_m': ParameterValue(
                        gpd_diversity_position_m, value_type=float),
                    'gpd_diversity_approach_degrees': ParameterValue(
                        gpd_diversity_approach_degrees, value_type=float),
                    'gpd_diversity_closing_degrees': ParameterValue(
                        gpd_diversity_closing_degrees, value_type=float),
                    'analytical_maximum_proposals': ParameterValue(
                        analytical_maximum_proposals, value_type=int),
                    'analytical_voxel_size_m': ParameterValue(
                        analytical_voxel_size_m, value_type=float),
                    'analytical_roll_step_degrees': ParameterValue(
                        analytical_roll_step_degrees, value_type=float),
                    'analytical_normal_radius_m': ParameterValue(
                        analytical_normal_radius_m, value_type=float),
                    'analytical_min_normal_neighbors': ParameterValue(
                        analytical_min_normal_neighbors, value_type=int),
                    'analytical_dedup_position_m': ParameterValue(
                        analytical_dedup_position_m, value_type=float),
                    'analytical_dedup_angle_degrees': ParameterValue(
                        analytical_dedup_angle_degrees, value_type=float),
                    'analytical_min_contact_center_separation_m': ParameterValue(
                        analytical_min_contact_center_separation_m, value_type=float),
                    'analytical_local_pca_searches': ParameterValue(
                        analytical_local_pca_searches, value_type=int),
                    'analytical_global_pca_searches': ParameterValue(
                        analytical_global_pca_searches, value_type=int),
                    'analytical_maximum_approach_variants_per_pair': ParameterValue(
                        analytical_maximum_approach_variants_per_pair, value_type=int),
                    'analytical_accepted_candidate_separation_m': ParameterValue(
                        analytical_accepted_candidate_separation_m, value_type=float),
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
                    'enable_gpd_score_filter': ParameterValue(
                        enable_gpd_score_filter, value_type=bool),
                }],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='vlm_node',
                plugin='VLMReasoningNode',
                name='vlm_reasoning_node',
                parameters=[{
                    'api_server_ip': api_server_ip,
                    'api_server_port': ParameterValue(
                        api_server_port, value_type=int
                    ),
                    'model_name': model_name,
                    'request_timeout_sec': ParameterValue(
                        request_timeout_sec, value_type=int
                    ),
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
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )

    master = Node(
        package='master_node',
        executable='piper_master_control',
        name='piper_master_control',
        output='screen',
        parameters=[{
            'hand_eye_calibration_path': hand_eye_calibration_path,
            'allow_physical_grasp_execution': ParameterValue(
                allow_physical_grasp_execution, value_type=bool),
            'allow_model_based_body_grasp_execution': ParameterValue(
                allow_model_based_body_grasp_execution, value_type=bool),
            'require_positive_gpd_score_for_grasp': ParameterValue(
                require_positive_gpd_score_for_grasp, value_type=bool),
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

    anyplace = Node(
        package='anyplace_ros',
        executable='anyplace_node',
        name='anyplace_cpu_node',
        output='screen',
        parameters=[{
            'hand_eye_calibration_path': hand_eye_calibration_path,
            'api_server_url': ParameterValue(
                ['http://', api_server_ip, ':', api_server_port],
                value_type=str),
            'model_name': model_name,
            'placement_backend': LaunchConfiguration('placement_backend'),
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
        DeclareLaunchArgument(
            'api_server_ip',
            default_value='127.0.0.1',
            description='LM Studio host reachable from the ROS machine.',
        ),
        DeclareLaunchArgument('api_server_port', default_value='1234'),
        DeclareLaunchArgument('model_name', default_value='robobrain2.5-4b'),
        DeclareLaunchArgument('request_timeout_sec', default_value='120'),
        DeclareLaunchArgument('anyplace_candidate_count', default_value='8'),
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
            description=(
                "Target-mask backend: 'edgesam' (RoboBrain box followed by "
                "class-agnostic segmentation), 'robobrain' (box only), or "
                "'yolo_intersection'."
            ),
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
            'require_positive_gpd_score_for_grasp',
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
            'enable_gpd_score_filter',
            default_value='true',
            description=(
                'Filter candidates at the raw GPD classifier boundary. The '
                'current Piper geometry requires false for diagnostic tuning.'
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
            'gpd_camera_frame',
            default_value='',
            description=(
                'Deprecated and ignored; transformations come exclusively '
                'from synchronized manual FK * calibrated X.'
            ),
        ),
        DeclareLaunchArgument(
            'grasp_candidate_backend', default_value='hggd',
            description='Proposal backend: analytical, gpd, hybrid, or hggd.'),
        DeclareLaunchArgument('hggd_model_directory', default_value=''),
        DeclareLaunchArgument('hggd_cpu_threads', default_value='4'),
        DeclareLaunchArgument('hggd_maximum_proposals', default_value='100'),
        DeclareLaunchArgument('maximum_candidates_to_test', default_value='600'),
        DeclareLaunchArgument('maximum_valid_candidates', default_value='100'),
        DeclareLaunchArgument('gpd_visualization_limit', default_value='100'),
        DeclareLaunchArgument('gpd_diversity_position_m', default_value='0.015'),
        DeclareLaunchArgument('gpd_diversity_approach_degrees', default_value='15.0'),
        DeclareLaunchArgument('gpd_diversity_closing_degrees', default_value='15.0'),
        DeclareLaunchArgument('analytical_maximum_proposals', default_value='300'),
        DeclareLaunchArgument('analytical_voxel_size_m', default_value='0.005'),
        DeclareLaunchArgument('analytical_roll_step_degrees', default_value='15.0'),
        DeclareLaunchArgument('analytical_normal_radius_m', default_value='0.008'),
        DeclareLaunchArgument('analytical_min_normal_neighbors', default_value='10'),
        DeclareLaunchArgument('analytical_dedup_position_m', default_value='0.010'),
        DeclareLaunchArgument('analytical_dedup_angle_degrees', default_value='10.0'),
        DeclareLaunchArgument(
            'analytical_min_contact_center_separation_m', default_value='0.025'),
        DeclareLaunchArgument(
            'analytical_local_pca_searches', default_value='50'),
        DeclareLaunchArgument(
            'analytical_global_pca_searches', default_value='50'),
        DeclareLaunchArgument(
            'analytical_maximum_approach_variants_per_pair', default_value='2'),
        DeclareLaunchArgument(
            'analytical_accepted_candidate_separation_m', default_value='0.025'),
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
            'gpd_tcp_offset_m',
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
            'require_gpd_full_antipodal',
            default_value='false',
            description=(
                'Require the legacy GPD full-antipodal label in addition to '
                'measured contact validation. Keep false for partial single-view clouds.'),
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
            default_value='0.060',
            description='Contact plane depth from the 70 mm finger base.',
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
            default_value='0.200',
            description=(
                'Preferred Cartesian standoff; planner falls back across '
                'reachable, target-visible offsets from 0.10 to 0.20 m.'
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
        master,
        anyplace,
    ])
