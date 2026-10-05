#!/usr/bin/env python3

# Standard Library
import os
import time
import json
import threading
import uuid
import xml.etree.ElementTree as ET

# Third-Party
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from collections import deque

# ROS 2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.action import ActionClient
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, SetBool, Empty
from std_msgs.msg import Float32MultiArray
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter as RclpyParameter
from ament_index_python.packages import get_package_share_directory

from my_actions.srv import (
    AppendTask, CalculateFallbackPose, Chat, CheckAndReplan, ExecuteGrasp,
    ExecutePipeline, FlushPipeline, Get3DCoordinate, GetNBVPose, GetWorldState,
    GetObstacleLocation, LookAt, MoveRelative, PlanAPF, ResolveObject,
    PlanCartesianGraspApproach, RefineGrasp, ToggleContinuousNavigation,
    UpdateObjectObservation,
)
from task_pipeline.registry import endpoint
from rclpy.serialization import deserialize_message, serialize_message

from agx_arm_msgs.msg import AgxArmStatus, GripperStatus
from master_node.task_manager import Task, TaskManager, canonical_task_type
from master_node.world_state import WorldStateManager
from my_actions.srv import CaptureObservation, CacheGraspRefinement, GraspStage

from master_node.step_runtime import MasterStepRuntime
from master_node.execution_api import ExecutionAPI

class BaseTask(Task):
    def __init__(self, node_ref, task_type, target_name, coordinates):
        super().__init__(task_type, target_name)
        self.node = node_ref
        self.target_name = target_name
        self.coordinates = list(coordinates)
        
    def execute(self):
        raise NotImplementedError

class LookAtTask(BaseTask):
    def execute(self):
        import math
        object_state = self.node.world_state.get_object(self.target_name)
        coordinates = (
            object_state["position"]
            if object_state is not None else self.coordinates
        )
        target_x, target_y, target_z = coordinates
        
        tried_rays = []

        while rclpy.ok() and not self.node.pipeline_tracker.stop_requested:
            # 1. Fetch unoccluded NBV candidates
            req_nbv = GetNBVPose.Request()
            req_nbv.target_x = target_x
            req_nbv.target_y = target_y
            req_nbv.target_z = target_z
            req_nbv.radius = 0.4 # 40cm radius to ensure unoccluded path up to 40cm
            
            future_nbv = self.node.get_nbv_client.call_async(req_nbv)
            while rclpy.ok() and not future_nbv.done():
                time.sleep(0.05)
                
            res_nbv = future_nbv.result()
            if not res_nbv.success or len(res_nbv.candidate_x) == 0:
                self.node.get_logger().error("[PIPELINE] Vision Node failed to find any unoccluded NBV candidates.")
                return False
                
            self.node.get_logger().info(f"[PIPELINE] Found {len(res_nbv.candidate_x)} unoccluded NBV candidates. Filtering...")
            
            # 2. Iterate candidates
            need_rescan = False
            for i in range(len(res_nbv.candidate_x)):
                if self.node.pipeline_tracker.stop_requested:
                    return False
                    
                cand_x = res_nbv.candidate_x[i]
                cand_y = res_nbv.candidate_y[i]
                cand_z = res_nbv.candidate_z[i]
                
                # Compute normalized direction vector
                dx = cand_x - target_x
                dy = cand_y - target_y
                dz = cand_z - target_z
                mag = math.sqrt(dx**2 + dy**2 + dz**2)
                if mag < 0.001:
                    continue
                dir_vec = (dx/mag, dy/mag, dz/mag)
                
                # Check if this ray was already tried
                already_tried = False
                for tr in tried_rays:
                    dot = dir_vec[0]*tr[0] + dir_vec[1]*tr[1] + dir_vec[2]*tr[2]
                    if dot > 0.99: # highly similar ray (within ~8 degrees)
                        already_tried = True
                        break
                
                if already_tried:
                    continue
                
                self.node.get_logger().info(f"[PIPELINE] Testing Ray {i}...")
                
                # Test distances 30cm, 35cm, 40cm along this ray
                distances = [0.30, 0.35, 0.40]
                ik_success = False
                best_joints = None
                successful_dist = 0
                
                for d in distances:
                    test_x = target_x + dir_vec[0] * d
                    test_y = target_y + dir_vec[1] * d
                    test_z = target_z + dir_vec[2] * d
                    
                    req_ik = LookAt.Request()
                    req_ik.look_at_x = target_x
                    req_ik.look_at_y = target_y
                    req_ik.look_at_z = target_z
                    req_ik.current_pose_x = test_x
                    req_ik.current_pose_y = test_y
                    req_ik.current_pose_z = test_z
                    if not self.node.populate_look_at_ik_request(
                            req_ik, (test_x, test_y, test_z),
                            (target_x, target_y, target_z)):
                        continue
                    if self.node.current_joint_states and len(self.node.current_joint_states.position) >= 6:
                        req_ik.current_joints = list(self.node.current_joint_states.position[:6])
                    else:
                        req_ik.current_joints = [0.0] * 6
                    
                    future = self.node.look_at_ik_client.call_async(req_ik)
                    while rclpy.ok() and not future.done():
                        time.sleep(0.05)
                        
                    res_ik = future.result()
                    
                    # Retry with neutral seed if first attempt fails due to local minima
                    if not res_ik.success:
                        req_ik.current_joints = [0.0] * 6
                        future_retry = self.node.look_at_ik_client.call_async(req_ik)
                        while rclpy.ok() and not future_retry.done():
                            time.sleep(0.05)
                        res_ik = future_retry.result()

                    if res_ik.success and len(res_ik.joint_angles) == 6:
                        ik_success = True
                        best_joints = res_ik.joint_angles
                        successful_dist = d
                        self.node.get_logger().info(f"[PIPELINE] Ray {i} is reachable at distance {d}m! Moving to target...")
                        break
                
                if not ik_success:
                    self.node.get_logger().info(f"[PIPELINE] Ray {i} unreachable at all distances. Discarding ray.")
                    tried_rays.append(dir_vec)
                    continue
                    
                # Use move_to_joints by passing joints to VAMP for path planning
                if self.node.current_joint_states and len(self.node.current_joint_states.position) >= 6:
                    curr_pos = list(self.node.current_joint_states.position[:6])
                else:
                    curr_pos = [0.0] * 6
                    
                req_replan = CheckAndReplan.Request()
                req_replan.current_joints = curr_pos
                req_replan.remaining_path_flat = best_joints
                
                future_replan = self.node.check_replan_client.call_async(req_replan)
                while rclpy.ok() and not future_replan.done():
                    time.sleep(0.05)
                    
                res_replan = future_replan.result()
                if not res_replan.success or not res_replan.path_flat:
                    self.node.get_logger().warn(f"[PIPELINE] VAMP Planner failed to generate path to ray {i}. Discarding ray...")
                    tried_rays.append(dir_vec)
                    continue
                    
                planned_path = [list(res_replan.path_flat[k:k+6]) for k in range(0, len(res_replan.path_flat), 6)]
                success = self.node.execute_trajectory_with_monitoring(planned_path, check_interval=float('inf'))
                
                if success:
                    # 4. Verify with VLM
                    self.node.get_logger().info(f"[PIPELINE] Reached ray {i}. Verifying view with VLM...")
                    vlm_req = Chat.Request()
                    vlm_req.prompt = f"Can you clearly see the {self.target_name}?"
                    vlm_future = self.node.vlm_chat_client.call_async(vlm_req)
                    while rclpy.ok() and not vlm_future.done():
                        time.sleep(0.05)
                    
                    vlm_res = vlm_future.result()
                    reply = vlm_res.response.lower()
                    if "yes" in reply:
                        self.node.get_logger().info(f"[PIPELINE] {self.task_type} completed successfully")
                        return True
                    else:
                        self.node.get_logger().warn(f"[PIPELINE] VLM reported view is occluded: {vlm_res.response}. Triggering scan and finding new pose...")
                        tried_rays.append(dir_vec)
                        # Trigger scan routine to map dynamic obstacle
                        self.node.trigger_single_scan()
                        time.sleep(1.0) # Wait for map to update
                        need_rescan = True
                        break # Break the candidate loop to fetch a new NBV list based on new map!
                else:
                    self.node.get_logger().warn(f"[PIPELINE] Arm was blocked or interrupted moving to ray {i}. Discarding ray...")
                    tried_rays.append(dir_vec)
                    continue

            if not need_rescan:
                # If we got here and didn't trigger a rescan, it means all rays were exhausted or unreachable.
                self.node.get_logger().error(f"[PIPELINE] All NBV candidates exhausted or unreachable.")
                return False
                
        return False

class PiperMasterControl(Node):
    ARM_JOINT_NAMES = [f'joint{index}' for index in range(1, 7)]
    GRIPPER_MIN_FORCE_N = 0.5
    GRIPPER_MAX_FORCE_N = 3.0

    def __init__(self):
        super().__init__('piper_master_control')
        self.get_logger().info("Initializing Piper Master Control Node (Director)...")

        self.world_state = WorldStateManager()

        self.cbg_kinematics = MutuallyExclusiveCallbackGroup()
        self.cbg_services = MutuallyExclusiveCallbackGroup()
        self.cbg_actions = MutuallyExclusiveCallbackGroup()
        self.cbg_clients = MutuallyExclusiveCallbackGroup()
        self.cbg_subs = MutuallyExclusiveCallbackGroup()
        self.cbg_state = ReentrantCallbackGroup()
        
        # State Variables
        self.state = "IDLE"
        self.show_rgb = False
        self.show_yolo = False
        self.executing_path = False
        self.current_path = []
        self.path_index = 0
        self.collision_detected = False

        self.current_joint_states = None
        self.current_tcp_pose = None
        self.current_rgb = None
        self.current_camera_info = None
        self.current_gripper_status = None
        self.current_arm_status = None
        self.cached_grasp_plans = {}
        self.refined_grasp_plans = {}

        self.torque_history = [deque(maxlen=50) for _ in range(6)]
        self.physical_collision_detected = False
        self.torque_spike_threshold = 4.0  # Increased to bypass acceleration spikes
        # Camera collision braking is opt-in.  Path planning and hardware E-stop
        # interfaces remain available; this only suppresses camera alerts from
        # braking or interrupting master-controlled motion.
        self.declare_parameter('enable_camera_collision_monitor', False)
        self.enable_camera_collision_monitor = self.get_parameter(
            'enable_camera_collision_monitor').get_parameter_value().bool_value
        self.declare_parameter('allow_physical_grasp_execution', False)
        self.allow_physical_grasp_execution = self.get_parameter(
            'allow_physical_grasp_execution').get_parameter_value().bool_value
        self.declare_parameter('require_grasp_collision_validation', True)
        self.require_grasp_collision_validation = self.get_parameter(
            'require_grasp_collision_validation').get_parameter_value().bool_value
        self.declare_parameter('require_positive_gpd_score_for_grasp', True)
        self.require_positive_gpd_score_for_grasp = self.get_parameter(
            'require_positive_gpd_score_for_grasp').get_parameter_value().bool_value
        self.declare_parameter('require_measured_contacts_for_grasp', True)
        self.require_measured_contacts_for_grasp = self.get_parameter(
            'require_measured_contacts_for_grasp').get_parameter_value().bool_value
        self.declare_parameter('allow_model_based_body_grasp_execution', False)
        self.allow_model_based_body_grasp_execution = self.get_parameter(
            'allow_model_based_body_grasp_execution').get_parameter_value().bool_value
        self.declare_parameter('allow_pregrasp_collision_override', False)
        self.allow_pregrasp_collision_override = self.get_parameter(
            'allow_pregrasp_collision_override').get_parameter_value().bool_value
        self.declare_parameter('grasp_allowed_contact_radius_m', 0.035)
        self.grasp_allowed_contact_radius_m = self.get_parameter(
            'grasp_allowed_contact_radius_m').get_parameter_value().double_value
        self.declare_parameter('pregrasp_target_exclusion_radius_m', 0.0)
        self.pregrasp_target_exclusion_radius_m = float(np.clip(
            self.get_parameter('pregrasp_target_exclusion_radius_m').
            get_parameter_value().double_value, 0.0, 0.08))
        self.declare_parameter('gripper_open_force_n', 1.0)
        self.gripper_open_force_n = self.get_parameter(
            'gripper_open_force_n').get_parameter_value().double_value
        self.declare_parameter('gripper_close_force_n', 1.0)
        self.gripper_close_force_n = self.get_parameter(
            'gripper_close_force_n').get_parameter_value().double_value
        self.declare_parameter('gripper_close_compression_m', 0.003)
        self.gripper_close_compression_m = float(np.clip(
            self.get_parameter('gripper_close_compression_m').
            get_parameter_value().double_value, 0.0, 0.01))
        self.declare_parameter('post_grasp_lift_m', 0.10)
        self.post_grasp_lift_m = float(np.clip(
            self.get_parameter('post_grasp_lift_m').get_parameter_value().double_value,
            0.0, 0.15))
        self.declare_parameter('gripper_min_open_feedback_m', 0.01)
        self.gripper_min_open_feedback_m = float(np.clip(
            self.get_parameter('gripper_min_open_feedback_m').get_parameter_value().double_value,
            0.001, 0.05))
        self.declare_parameter('tracked_object_association_m', 0.05)
        self.object_association_m = self.get_parameter(
            'tracked_object_association_m').get_parameter_value().double_value
        self.declare_parameter('tracked_object_miss_threshold', 3)
        self.object_miss_threshold = self.get_parameter(
            'tracked_object_miss_threshold').get_parameter_value().integer_value
        self.declare_parameter('hand_eye_calibration_path', '')

        self.bridge = CvBridge()
        
        # Publisher
        self.joint_pub = self.create_publisher(
            JointState, '/control/joint_states', 10, callback_group=self.cbg_kinematics)
        self.param_client = self.create_client(
            SetParameters, '/agx_arm_ctrl_single_node/set_parameters', callback_group=self.cbg_clients)
        
        # Action Clients
        self.check_replan_client = self.create_client(CheckAndReplan, endpoint('plan_motion'), callback_group=self.cbg_clients)
        self.toggle_nav_client = self.create_client(ToggleContinuousNavigation, endpoint('toggle_continuous_navigation'), callback_group=self.cbg_clients)
        self.get_obstacle_client = self.create_client(GetObstacleLocation, endpoint('get_obstacle_location'), callback_group=self.cbg_clients)
        self.calc_fallback_client = self.create_client(CalculateFallbackPose, endpoint('calculate_fallback_pose'), callback_group=self.cbg_clients)
        self.plan_apf_client = self.create_client(PlanAPF, endpoint('plan_apf'), callback_group=self.cbg_clients)
        
        self.collision_monitor_pub = self.create_publisher(Bool, '/active_vision/enable_collision_monitor', 10)
        self.collision_alert_sub = self.create_subscription(Bool, '/active_vision/collision_alert', self.collision_alert_callback, 10, callback_group=self.cbg_subs)
        if not self.enable_camera_collision_monitor:
            self.collision_monitor_pub.publish(Bool(data=False))
        
        self.trigger_scan_client = self.create_client(Trigger, endpoint('request_scan'), callback_group=self.cbg_clients)
        self.reload_client = self.create_client(Trigger, '/active_vision/reload_calibration', callback_group=self.cbg_clients)
        self.estop_client = self.create_client(Empty, '/emergency_stop', callback_group=self.cbg_clients)
        self.look_at_ik_client = self.create_client(LookAt, endpoint('solve_pose'), callback_group=self.cbg_clients)
        self.get_nbv_client = self.create_client(GetNBVPose, endpoint('get_nbv_candidates'), callback_group=self.cbg_clients)
        self.get_3d_coordinate_client = self.create_client(
            Get3DCoordinate, endpoint('get_3d_coordinate'),
            callback_group=self.cbg_clients)
        self.refine_grasp_client = self.create_client(
            RefineGrasp, endpoint('refine_grasp'),
            callback_group=self.cbg_clients)
        self.plan_cartesian_grasp_client = self.create_client(
            PlanCartesianGraspApproach,
            endpoint('plan_final_approach'),
            callback_group=self.cbg_clients)

        # Services
        self.pipeline_tracker = TaskManager(
            world_state=self.world_state,
            logger=self.get_logger(),
        )
        
        # Pipeline Services
        self.srv_append = self.create_service(AppendTask, '/master/append_task', self.append_task_callback, callback_group=self.cbg_services)
        self.srv_exec = self.create_service(ExecutePipeline, '/master/execute_pipeline', self.execute_pipeline_callback, callback_group=self.cbg_services)
        self.srv_flush = self.create_service(FlushPipeline, '/master/flush_pipeline', self.flush_pipeline_callback, callback_group=self.cbg_services)
        self.srv_update_object = self.create_service(
            UpdateObjectObservation, '/master/update_object_observation',
            self.update_object_observation_callback,
            callback_group=self.cbg_services)
        self.srv_resolve_object = self.create_service(
            ResolveObject, '/master/resolve_object',
            self.resolve_object_callback, callback_group=self.cbg_services)
        self.srv_execute_grasp = self.create_service(
            ExecuteGrasp, '/master/execute_grasp',
            self.execute_grasp_callback, callback_group=self.cbg_services)
        self.srv_get_world_state = self.create_service(
            GetWorldState, '/master/get_world_state',
            self.get_world_state_callback, callback_group=self.cbg_state)
        
        # VLM client for visual verification
        self.vlm_chat_client = self.create_client(Chat, '/vlm_reasoning_node/chat_prompt', callback_group=self.cbg_clients)
        
        self.srv_calib = self.create_service(
            Trigger, '/piper_master/start_calibration', self.start_calibration_callback, callback_group=self.cbg_services)
        self.srv_scan = self.create_service(
            Trigger, '/piper_master/start_global_scan', self.start_global_scan_callback, callback_group=self.cbg_services)
        self.srv_nav = self.create_service(
            Trigger, '/piper_master/start_continuous_navigation', self.start_continuous_nav_callback, callback_group=self.cbg_services)
        self.srv_move_joints = self.create_service(
            ToggleContinuousNavigation, '/piper_master/move_to_joints', self.move_to_joints_callback, callback_group=self.cbg_services)
        self.srv_look_at = self.create_service(
            LookAt, '/master/look_at_target', self.look_at_target_callback, callback_group=self.cbg_services)
        self.srv_move_relative = self.create_service(
            MoveRelative, '/master/move_relative', self.move_relative_callback, callback_group=self.cbg_services)

        self.step_runtime = MasterStepRuntime(self)
        self.execution_api = ExecutionAPI(self)

        self.calibration_poses = [
            [-0.08, 0.0, 0.0, 0.0, 0.04, 0.0],
            [-0.08, 0.4, -0.5, 0.0, 0.45, 0.2],
            [0.7, 0.2, -0.5, -1, 0.9, 1],
            [0.3, 0.2, -1, -0.2, 1.2, 0],
            [-0.9, 0.2, -0.45, 1, 0.9, -1],
            [-0.3, 0.2, -1, 0.2, 1.2, 0]
        ]
        
        self.scan_poses = [
            [-1, 0, 0, 0, 0.4, 0],
            [0, 0, 0, 0, 0.4, 0],
            [1, 0, 0, 0, 0.4, 0],
            [1, 0.5, -1, 0, 1.2, 0],
			#[1.2, 1.5, -1.2, -1, 1.22, 0],
            [0, 0.5, -1, 0, 1.2, 0],
			#[-1.2, 1.5, -1.2, 1, 1.22, 0],
            [-1, 0.5, -1, 0, 1.2, 0]
        ]

        self.hand_eye_matrix = None
        
        # Statically set arm speed to 5% at boot
        threading.Thread(target=self._set_initial_speed, daemon=True).start()

        self.load_calibration()
        self.load_joint_limits()
        
        self.camera_sync_delay = 2
        self.joint_state_check_delay = 0.1
        self.scan_interval = 60
        self.last_idle_time = time.time()
        self.ignore_camera_collisions = not self.enable_camera_collision_monitor
        self.path_lock = threading.Lock()
        
        # Threads
        self.calibration_thread = None
        self.scan_thread = threading.Thread(target=self.scan_routine_loop, daemon=True)
        self.scan_thread.start()

        # Subscribe to Kinematics always
        self.joint_sub = self.create_subscription(JointState, '/feedback/joint_states', self.joint_callback, 10, callback_group=self.cbg_kinematics)
        self.pose_sub = self.create_subscription(PoseStamped, '/feedback/tcp_pose', self.pose_callback, 10, callback_group=self.cbg_kinematics)
        self.gripper_sub = self.create_subscription(
            GripperStatus, '/feedback/gripper_status',
            self.gripper_callback, 10, callback_group=self.cbg_kinematics)
        self.arm_status_sub = self.create_subscription(
            AgxArmStatus, '/feedback/arm_status',
            self.arm_status_callback, 10, callback_group=self.cbg_kinematics)
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10,
            callback_group=self.cbg_subs)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_callback, 10,
            callback_group=self.cbg_subs)
        self.last_joint_positions = None

    @property
    def state(self):
        return self.world_state.robot_state

    @state.setter
    def state(self, value):
        self.world_state.set_robot_state(value)

    def load_calibration(self):
        calib_file = self.get_parameter(
            'hand_eye_calibration_path').get_parameter_value().string_value
        if not calib_file:
            calib_file = os.path.join(
                get_package_share_directory('master_node'), 'config',
                'calibration', 'hand_eye_calibration.yaml')
        if os.path.exists(calib_file):
            try:
                with open(calib_file, 'r') as f:
                    data = yaml.safe_load(f)
                    self.hand_eye_matrix = np.array(data['hand_eye_transform'])
                self.get_logger().info("Loaded Hand-Eye Calibration.")
            except Exception as e:
                self.get_logger().error(f"Failed to load calibration: {e}")
        else:
            self.get_logger().warn("NO HAND-EYE CALIBRATION FOUND!")

    def load_joint_limits(self):
        self.joint_limits = []
        urdf_path = os.path.join(
            get_package_share_directory('agx_arm_description'),
            'agx_arm_urdf', 'piper', 'urdf', 'piper_description.urdf')
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for j in range(1, 7):
                joint_name = f'joint{j}'
                lower, upper = -3.1415926, 3.1415926
                for joint in root.findall('joint'):
                    if joint.get('name') == joint_name:
                        limit_tag = joint.find('limit')
                        if limit_tag is not None:
                            lower = float(limit_tag.get('lower', lower))
                            upper = float(limit_tag.get('upper', upper))
                        break
                self.joint_limits.append((lower, upper))
            self.get_logger().info(f"Loaded joint limits from URDF: {self.joint_limits}")
        except Exception as e:
            self.get_logger().error(f"Failed to load URDF joint limits from {urdf_path}: {e}. Using default limits.")
            self.joint_limits = [
                (-2.6179938, 2.6179938),
                (0.0, 3.1415926),
                (-2.9670597, 0.0),
                (-1.7453292, 1.7453292),
                (-1.2217304, 1.2217304),
                (-2.0943951, 2.0943951)
            ]

    def scan_routine_loop(self):
        while rclpy.ok():
            time.sleep(1.0)
            if self.state == "IDLE":
                self.trigger_single_scan()
            else:
                captured_state = self.state
                while self.state != "IDLE" and rclpy.ok():
                    time.sleep(0.01)
                
                if captured_state == "MOVING":
                    time.sleep(0.1) # 100ms settle
                
                self.get_logger().debug(
                    f"[ROUTINE TRIGGER] Firing Single-Shot Scan after "
                    f"completing {captured_state} state.")
                self.trigger_single_scan()

    def trigger_single_scan(self, wait_for_result=False, timeout_sec=3.0):
        if self.trigger_scan_client.wait_for_service(timeout_sec=1.0):
            req = Trigger.Request()
            future = self.trigger_scan_client.call_async(req)
            self.get_logger().debug("[SCAN] Triggered single scan on /active_vision/trigger_scan")
            if not wait_for_result:
                return True
            deadline = time.monotonic() + timeout_sec
            while (rclpy.ok() and not future.done() and
                   time.monotonic() < deadline):
                time.sleep(0.02)
            result = future.result() if future.done() else None
            return result is not None and result.success
        else:
            self.get_logger().warn(
                "[SCAN] /active_vision/trigger_scan service not available!",
                once=True)
        return False

    def perform_scan_routine(self):
        # 1. Settle for 100ms
        time.sleep(0.1)
        # 2. Trigger Single Scan
        self.trigger_single_scan()

    def collision_alert_callback(self, msg):
        if not self.enable_camera_collision_monitor or self.ignore_camera_collisions:
            self.get_logger().debug("[COLLISION ALERT] Ignoring collision alert (camera collisions ignored during APF/dodge)")
            return
        if msg.data and not self.collision_detected:
            self.get_logger().warn("[URGENT] COLLISION DETECTED BY CAMERA! BRAKING! (state=%s, ignore_camera=%s)" % (self.state, self.ignore_camera_collisions))
            self.collision_detected = True
            
            # Slam brakes
            if self.current_joint_states:
                self._publish_joint_state_command(
                    arm_positions=self.current_joint_states.position[:6])
                self.get_logger().info("[COLLISION ALERT] Brakes applied at joints: %s" % [round(x, 3) for x in self.current_joint_states.position[:6]])
                
            # Disable collision monitor to prevent spam
            cm_msg = Bool()
            cm_msg.data = False
            self.collision_monitor_pub.publish(cm_msg)
            self.get_logger().info("[COLLISION ALERT] Collision monitor disabled to prevent spam")
            
            # Run scan routine
            self.perform_scan_routine()
        elif msg.data and self.collision_detected:
            self.get_logger().debug("[COLLISION ALERT] Duplicate alert ignored (already handling collision)")

    def joint_callback(self, msg):
        self.current_joint_states = msg
        
        # Test routine: Physical collision awareness via effort
        # (DISABLED AS REQUESTED)
        # if not self.physical_collision_detected and len(msg.effort) >= 6 and self.state == "MOVING":
        #     for i in range(6):
        #         current_effort = msg.effort[i]
        #         if len(self.torque_history[i]) > 0:
        #             moving_average = sum(self.torque_history[i]) / len(self.torque_history[i])
        #             if abs(current_effort - moving_average) > self.torque_spike_threshold:
        #                 self.get_logger().error(f"[URGENT] PHYSICAL COLLISION DETECTED on joint {i+1} (Spike: {abs(current_effort - moving_average):.2f} Nm)! BRAKING AND ESTOP!")
        #                 self.physical_collision_detected = True
        #                 
        #                 # Call E-STOP service
        #                 if self.estop_client.wait_for_service(timeout_sec=0.1):
        #                     req = Empty.Request()
        #                     self.estop_client.call_async(req)
        #                 else:
        #                     self.get_logger().error("E-STOP service not available!")
        #                 
        #                 # Freeze arm locally just in case
        #                 brake_msg = JointState()
        #                 brake_msg.position = self.current_joint_states.position[:6]
        #                 self.joint_pub.publish(brake_msg)
        #                 return
        #         
        #         self.torque_history[i].append(current_effort)

    def pose_callback(self, msg):
        self.current_tcp_pose = msg

    def gripper_callback(self, msg):
        self.current_gripper_status = msg

    def arm_status_callback(self, msg):
        self.current_arm_status = msg

    def rgb_callback(self, msg):
        self.current_rgb = msg

    def info_callback(self, msg):
        self.current_camera_info = msg
        
    def _set_initial_speed(self):
        waiting_logged = False
        while rclpy.ok():
            try:
                if self.param_client.wait_for_service(timeout_sec=5.0):
                    self.set_arm_speed(20)
                    return
            except Exception:
                if not rclpy.ok():
                    return
                raise
            if not waiting_logged:
                self.get_logger().info(
                    "Arm controller is not running; boot speed will be set "
                    "when its parameter service becomes available."
                )
                waiting_logged = True
        
    def set_arm_speed(self, speed_percent: int):
        if not self.param_client.service_is_ready():
            if not self.param_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn(f"Parameter service /agx_arm_ctrl_single_node/set_parameters not ready. Cannot set speed to {speed_percent}%")
                return False
        req = SetParameters.Request()
        req.parameters = [RclpyParameter('speed_percent', RclpyParameter.Type.INTEGER, int(speed_percent)).to_parameter_msg()]
        future = self.param_client.call_async(req)
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done() or future.result() is None:
            self.get_logger().error(
                f'[SPEED COMMAND] Timed out setting arm speed to {speed_percent}%')
            return False
        results = future.result().results
        if not results or not results[0].successful:
            reason = results[0].reason if results else 'no parameter result'
            self.get_logger().error(
                f'[SPEED COMMAND] Failed to set arm speed to {speed_percent}%: '
                f'{reason}')
            return False
        self.get_logger().info(
            f'[SPEED COMMAND] Arm speed set to {speed_percent}%')
        return True

    def _publish_joint_state_command(
            self, arm_positions=None, gripper_width=None,
            gripper_force=None):
        """Publish a driver-compliant partial or combined joint command."""
        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()

        if arm_positions is not None:
            if len(arm_positions) != len(self.ARM_JOINT_NAMES):
                raise ValueError('arm command must contain exactly 6 joints')
            command.name.extend(self.ARM_JOINT_NAMES)
            command.position.extend(float(value) for value in arm_positions)

        if gripper_width is not None:
            force = 1.0 if gripper_force is None else float(gripper_force)
            if not self.GRIPPER_MIN_FORCE_N <= force <= self.GRIPPER_MAX_FORCE_N:
                self.get_logger().error(
                    f'[JOINT COMMAND] Gripper force {force:.2f} N is outside '
                    f'the driver range [{self.GRIPPER_MIN_FORCE_N:.1f}, '
                    f'{self.GRIPPER_MAX_FORCE_N:.1f}] N; command rejected.')
                return False
            command.name.append('gripper')
            command.position.append(float(np.clip(gripper_width, 0.0, 0.1)))
            command.effort = [0.0] * (len(command.name) - 1) + [force]

        if not command.name:
            return False
        self.joint_pub.publish(command)
        return True

    def move_arm_to(self, pose, disable_collision_monitor=False, tolerance=0.02):
        self.current_target_pose = pose
        
        # Reset collision monitors and torque history for new movement
        self.collision_detected = False
        self.physical_collision_detected = False
        for q in self.torque_history:
            q.clear()
            
        monitor_camera_collision = self.enable_camera_collision_monitor and not disable_collision_monitor
        if monitor_camera_collision:
            cm_msg = Bool()
            cm_msg.data = True
            self.collision_monitor_pub.publish(cm_msg)
            self.get_logger().debug("[MOVE_ARM] Collision monitor ENABLED for move to %s" % [round(x, 3) for x in pose])
        else:
            self.collision_monitor_pub.publish(Bool(data=False))
            self.get_logger().debug("[MOVE_ARM] Collision monitor DISABLED for move to %s" % [round(x, 3) for x in pose])
            
        try:
            # Quick limit check and clamping
            clamped_pose = []
            for i, val in enumerate(pose):
                if hasattr(self, 'joint_limits') and i < len(self.joint_limits):
                    lower, upper = self.joint_limits[i]
                    if val < lower:
                        self.get_logger().warn(f"Joint {i+1} angle {val:.4f} below lower limit {lower:.4f}. Clamping to {lower:.4f}.")
                        clamped_val = lower
                    elif val > upper:
                        self.get_logger().warn(f"Joint {i+1} angle {val:.4f} above upper limit {upper:.4f}. Clamping to {upper:.4f}.")
                        clamped_val = upper
                    else:
                        clamped_val = val
                    clamped_pose.append(clamped_val)
                else:
                    clamped_pose.append(val)

            # self.get_logger().info(f"[ARM MOTION DEBUG] Moving to joint target: {[round(x, 3) for x in clamped_pose]}")

            arrived = False
            self._publish_joint_state_command(arm_positions=clamped_pose)
            for _ in range(100):
                if monitor_camera_collision and (self.collision_detected or self.physical_collision_detected):
                    self.get_logger().warn("[ARM MOTION DEBUG] Collision detected during movement! Aborting.")
                    break
                
                time.sleep(self.joint_state_check_delay)
                if self.current_joint_states is not None:
                    if len(self.current_joint_states.position) >= len(clamped_pose):
                        errors = [abs(curr - target) for curr, target in zip(self.current_joint_states.position, clamped_pose)]
                        if all(e <= tolerance for e in errors):
                            arrived = True
                            break
            # if arrived:
            #     self.get_logger().info(f"[ARM MOTION DEBUG] Successfully reached target: {[round(x, 3) for x in clamped_pose]}")
            # else:
            #     self.get_logger().warn(f"[ARM MOTION DEBUG] Motion loop ended before reaching exact tolerance: {[round(x, 3) for x in clamped_pose]}")
            return arrived
        finally:
            if monitor_camera_collision:
                cm_msg = Bool()
                cm_msg.data = False
                self.collision_monitor_pub.publish(cm_msg)

    def append_task_callback(self, request, response):
        task_type = canonical_task_type(request.task_type)
        if task_type == "OBSERVE_OBJECT":
            if len(request.coordinates) != 3:
                response.success = False
                response.message = "OBSERVE_OBJECT requires a 3D world position."
                return response

            self.world_state.upsert_object(
                request.target_name,
                name=request.target_name,
                position=request.coordinates,
                role="target",
                visible=True,
            )
            task = LookAtTask(
                self,
                task_type,
                request.target_name,
                request.coordinates,
            )
            self.pipeline_tracker.append(task)
            response.success = True
            response.message = f"Appended {task_type} to task queue."
            self.get_logger().info(f"[MASTER NODE] {response.message}")
        else:
            response.success = False
            response.message = f"Unsupported task type: {request.task_type}"
            self.get_logger().warn(f"[MASTER NODE] {response.message}")
        return response
        
    def execute_pipeline_callback(self, request, response):
        if self.pipeline_tracker.is_executing:
            response.success = False
            response.message = "Pipeline is already executing."
            self.get_logger().warn(f"[MASTER NODE] {response.message}")
            return response
            
        success = self.pipeline_tracker.execute()
        response.success = success
        response.message = "Pipeline execution started."
        self.get_logger().info(f"[MASTER NODE] {response.message}")
        return response
        
    def flush_pipeline_callback(self, request, response):
        self.pipeline_tracker.flush()
        response.success = True
        response.message = "Pipeline flushed."
        self.get_logger().info(f"[MASTER NODE] {response.message}")
        return response

    @staticmethod
    def _message_stamp(message):
        if message is None or not hasattr(message, 'header'):
            return None
        return {
            'sec': int(message.header.stamp.sec),
            'nanosec': int(message.header.stamp.nanosec),
        }

    def get_world_state_callback(self, request, response):
        """Return the master-owned semantic state plus live robot feedback."""
        del request
        snapshot = self.world_state.snapshot()
        robot = snapshot['robot']
        robot['pipeline_executing'] = self.pipeline_tracker.is_executing
        robot['physical_grasp_execution_enabled'] = (
            self.allow_physical_grasp_execution)
        robot['camera_collision_monitor_enabled'] = (
            self.enable_camera_collision_monitor)

        joints = self.current_joint_states
        robot['joint_feedback'] = None if joints is None else {
            'timestamp': self._message_stamp(joints),
            'names': list(joints.name),
            'positions_rad_or_m': [float(value) for value in joints.position],
            'velocities': [float(value) for value in joints.velocity],
            'efforts': [float(value) for value in joints.effort],
        }

        tcp = self.current_tcp_pose
        robot['tcp_pose'] = None if tcp is None else {
            'timestamp': self._message_stamp(tcp),
            'frame_id': tcp.header.frame_id,
            'position_m': {
                'x': float(tcp.pose.position.x),
                'y': float(tcp.pose.position.y),
                'z': float(tcp.pose.position.z),
            },
            'orientation_xyzw': {
                'x': float(tcp.pose.orientation.x),
                'y': float(tcp.pose.orientation.y),
                'z': float(tcp.pose.orientation.z),
                'w': float(tcp.pose.orientation.w),
            },
        }

        gripper = self.current_gripper_status
        robot['gripper'] = None if gripper is None else {
            'timestamp': self._message_stamp(gripper),
            'width_m': float(gripper.width),
            'force_n': float(gripper.force),
            'driver_enabled': bool(gripper.driver_enable_status),
            'homed': bool(gripper.homing_status),
            'faults': {
                'voltage_too_low': bool(gripper.voltage_too_low),
                'motor_overheating': bool(gripper.motor_overheating),
                'driver_overcurrent': bool(gripper.driver_overcurrent),
                'driver_overheating': bool(gripper.driver_overheating),
                'sensor': bool(gripper.sensor_status),
                'driver': bool(gripper.driver_error_status),
            },
        }

        arm = self.current_arm_status
        robot['arm_status'] = None if arm is None else {
            'timestamp': self._message_stamp(arm),
            'control_mode': int(arm.ctrl_mode),
            'status': int(arm.arm_status),
            'mode_feedback': int(arm.mode_feedback),
            'teach_status': int(arm.teach_status),
            'motion_status': int(arm.motion_status),
            'trajectory_number': int(arm.trajectory_num),
            'error_status': int(arm.err_status),
            'joint_angle_limit': [bool(value) for value in arm.joint_angle_limit],
            'joint_communication_error': [
                bool(value) for value in arm.communication_status_joint],
        }

        snapshot['perception'] = {
            'rgb_feedback_available': self.current_rgb is not None,
            'rgb_timestamp': self._message_stamp(self.current_rgb),
            'camera_info_available': self.current_camera_info is not None,
            'camera_resolution': None if self.current_camera_info is None else {
                'width': int(self.current_camera_info.width),
                'height': int(self.current_camera_info.height),
            },
            'hand_eye_calibration_loaded': self.hand_eye_matrix is not None,
        }
        snapshot['cached_grasp_plans'] = [
            {
                'object_id': object_id,
                'name': plan['name'],
                'candidate_id': int(plan['candidate'].id),
                'ranking_score': float(plan['candidate'].ranking_score),
                'jaw_opening_m': float(plan['candidate'].jaw_opening_m),
            }
            for object_id, plan in self.cached_grasp_plans.items()
        ]
        snapshot['object_count'] = len(snapshot['objects'])
        if hasattr(self, 'step_runtime'):
            snapshot['api_pipeline'] = json.loads(self.step_runtime.store.snapshot()[0])
        response.state_json = json.dumps(snapshot, indent=2, sort_keys=True)
        return response

    def _object_expected_in_current_view(self, position):
        """Return true only when projection and current depth support visibility."""
        if self.current_tcp_pose is None or self.hand_eye_matrix is None:
            return False
        tcp_transform = np.eye(4)
        tcp_transform[:3, :3] = Rotation.from_quat([
            self.current_tcp_pose.pose.orientation.x,
            self.current_tcp_pose.pose.orientation.y,
            self.current_tcp_pose.pose.orientation.z,
            self.current_tcp_pose.pose.orientation.w,
        ]).as_matrix()
        tcp_transform[:3, 3] = [
            self.current_tcp_pose.pose.position.x,
            self.current_tcp_pose.pose.position.y,
            self.current_tcp_pose.pose.position.z,
        ]
        camera_from_base = np.linalg.inv(
            tcp_transform @ self.hand_eye_matrix)
        point_camera = camera_from_base @ np.array([
            position[0], position[1], position[2], 1.0])
        if point_camera[2] <= 0.05:
            return False

        if self.current_camera_info is not None:
            fx = self.current_camera_info.k[0]
            fy = self.current_camera_info.k[4]
            cx = self.current_camera_info.k[2]
            cy = self.current_camera_info.k[5]
            image_width = self.current_camera_info.width
            image_height = self.current_camera_info.height
        else:
            fx, fy = 459.581, 459.839
            cx, cy = 320.197, 240.590
            image_width, image_height = 640, 480
        u = int(round(fx * point_camera[0] / point_camera[2] + cx))
        v = int(round(fy * point_camera[1] / point_camera[2] + cy))
        if not (5 <= u < image_width - 5 and 5 <= v < image_height - 5):
            return False

        if not self.get_3d_coordinate_client.wait_for_service(timeout_sec=0.2):
            return False
        depth_request = Get3DCoordinate.Request()
        depth_request.u = u
        depth_request.v = v
        depth_request.width = 1
        depth_request.height = 1
        future = self.get_3d_coordinate_client.call_async(depth_request)
        deadline = time.monotonic() + 1.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done() or not future.result().success:
            return False
        hit = future.result()
        camera_origin = (tcp_transform @ self.hand_eye_matrix)[:3, 3]
        expected_distance = np.linalg.norm(
            np.asarray(position) - camera_origin)
        observed_distance = np.linalg.norm(
            np.array([hit.x, hit.y, hit.z]) - camera_origin)
        # A substantially nearer return means another surface occludes the
        # tracked object. Background behind the old position means the object
        # should have been visible and therefore counts as a genuine miss.
        return observed_distance >= expected_distance - 0.05

    def update_object_observation_callback(self, request, response):
        if request.detected:
            if request.position.header.frame_id not in ('', 'base_link'):
                response.success = False
                response.message = (
                    'Object observations must be expressed in base_link.')
                return response
            position = [
                request.position.point.x,
                request.position.point.y,
                request.position.point.z,
            ]
            object_id = self.world_state.associate_observation(
                request.name, position, self.object_association_m)
            response.success = True
            response.object_id = object_id
            response.message = f'UPDATED {object_id}'
            self.get_logger().debug(
                f'[WORLD STATE] {request.name} -> {object_id} at '
                f'{[round(value, 3) for value in position]}')
            return response

        expected = any(
            self._object_expected_in_current_view(record['position'])
            for _, record in self.world_state.objects_named(request.name)
        )
        if not expected:
            response.success = True
            response.message = 'MISS_IGNORED_OUTSIDE_FOV_OR_OCCLUDED'
            return response
        removed = self.world_state.record_expected_miss(
            request.name, self.object_miss_threshold)
        response.success = True
        response.message = (
            f'REMOVED {removed}' if removed else 'EXPECTED_VIEW_MISS_RECORDED')
        self.get_logger().debug(
            f'[WORLD STATE] {request.name}: {response.message}')
        return response

    def resolve_object_callback(self, request, response):
        """Resolve a unique semantic object name into its last world pose."""
        matches = self.world_state.objects_named(request.name)
        if not matches:
            response.success = False
            response.status = 'UNKNOWN'
            return response
        if len(matches) != 1:
            response.success = False
            response.status = 'AMBIGUOUS'
            return response
        object_id, record = matches[0]
        response.success = True
        response.status = 'RESOLVED'
        response.object_id = object_id
        response.position.header.stamp = self.get_clock().now().to_msg()
        response.position.header.frame_id = 'base_link'
        response.position.point.x = float(record['position'][0])
        response.position.point.y = float(record['position'][1])
        response.position.point.z = float(record['position'][2])
        response.visible = bool(record.get('visible', False))
        return response

    def populate_look_at_ik_request(
            self, request, camera_position, target_position):
        """Convert a desired optical-camera pose into the calibrated TCP pose."""
        if self.hand_eye_matrix is None:
            self.get_logger().error(
                '[NBV] Cannot convert camera pose without hand-eye calibration.')
            return False

        camera_position = np.asarray(camera_position, dtype=float)
        target_position = np.asarray(target_position, dtype=float)
        forward = target_position - camera_position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return False
        forward /= forward_norm

        # ROS optical frame: +x right, +y down, +z forward. Keep image-down
        # near base -Z while looking at the target.
        base_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, base_up)
        if np.linalg.norm(right) < 1e-4:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)

        base_camera = np.eye(4)
        base_camera[:3, :3] = np.column_stack((right, down, forward))
        base_camera[:3, 3] = camera_position
        base_tcp = base_camera @ np.linalg.inv(self.hand_eye_matrix)
        quaternion = Rotation.from_matrix(base_tcp[:3, :3]).as_quat()

        request.use_tcp_pose = True
        request.tcp_pose.position.x = float(base_tcp[0, 3])
        request.tcp_pose.position.y = float(base_tcp[1, 3])
        request.tcp_pose.position.z = float(base_tcp[2, 3])
        request.tcp_pose.orientation.x = float(quaternion[0])
        request.tcp_pose.orientation.y = float(quaternion[1])
        request.tcp_pose.orientation.z = float(quaternion[2])
        request.tcp_pose.orientation.w = float(quaternion[3])
        return True

    def _command_gripper(self, width, force):
        accepted = self._publish_joint_state_command(
            gripper_width=width, gripper_force=force)
        if accepted:
            self.get_logger().info(
                f'[GRIPPER] Commanded width={width:.3f}m force={force:.1f}N '
                'via /control/joint_states')
        return accepted

    def _wait_for_gripper_opening(
            self, target_width, tolerance_m=0.004, timeout_sec=4.0):
        del tolerance_m
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            status = self.current_gripper_status
            # Some Piper grippers report an uncalibrated absolute width.  The
            # execution gate therefore verifies only that the gripper did not
            # remain closed after the opening command, not that it reached the
            # requested geometric aperture.
            if (status is not None and
                    status.width > self.gripper_min_open_feedback_m):
                return True
            time.sleep(0.05)

        status = self.current_gripper_status
        if status is None:
            detail = 'no /feedback/gripper_status received'
        else:
            detail = (
                f'feedback width={status.width:.3f}m, '
                f'driver_enabled={status.driver_enable_status}, '
                f'homed={status.homing_status}')
        self.get_logger().error(
            f'[GRIPPER] Did not report open feedback above '
            f'{self.gripper_min_open_feedback_m:.3f}m after command '
            f'{target_width:.3f}m: {detail}')
        return False

    def _wait_for_gripper_contact(self, timeout_sec=2.0):
        """Require feedback evidence after closing; a command is not contact."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            status = self.current_gripper_status
            if (status is not None and status.width > 0.001 and
                    status.force >= 0.2):
                return True
            time.sleep(0.05)
        return False

    @staticmethod
    def _pose_error(reference, measured):
        if reference is None or measured is None:
            return None
        if (reference.header.frame_id != 'base_link' or
                measured.header.frame_id != 'base_link'):
            return None
        try:
            reference_position = np.array([
                reference.pose.position.x, reference.pose.position.y,
                reference.pose.position.z], dtype=float)
            measured_position = np.array([
                measured.pose.position.x, measured.pose.position.y,
                measured.pose.position.z], dtype=float)
            reference_rotation = Rotation.from_quat([
                reference.pose.orientation.x, reference.pose.orientation.y,
                reference.pose.orientation.z, reference.pose.orientation.w])
            measured_rotation = Rotation.from_quat([
                measured.pose.orientation.x, measured.pose.orientation.y,
                measured.pose.orientation.z, measured.pose.orientation.w])
            orientation_error = (
                reference_rotation.inv() * measured_rotation).magnitude()
        except (ValueError, TypeError):
            return None
        return float(np.linalg.norm(measured_position - reference_position)), float(
            orientation_error)

    def _verify_refined_pregrasp(self, refined_plan):
        if (self.current_joint_states is None or
                len(self.current_joint_states.position) < 6):
            return False, 'CURRENT_JOINT_STATE_UNAVAILABLE'
        pregrasp_error = max(abs(float(current) - float(expected))
                             for current, expected in zip(
                                 self.current_joint_states.position[:6],
                                 refined_plan['pregrasp_joints']))
        if pregrasp_error > 0.08:
            return False, (
                'NOT_AT_REFINED_PREGRASP: maximum joint error='
                f'{pregrasp_error:.3f}rad; run [PRE_GRASP] again')
        pose_error = self._pose_error(
            refined_plan.get('pregrasp_pose'), self.current_tcp_pose)
        if pose_error is None:
            return False, 'CURRENT_TCP_POSE_UNAVAILABLE_OR_INVALID'
        position_error, orientation_error = pose_error
        if position_error > 0.010 or orientation_error > 0.15:
            return False, (
                'NOT_AT_REFINED_PREGRASP: measured pose error='
                f'{position_error:.4f}m/{orientation_error:.3f}rad; '
                'run [PRE_GRASP] again')
        return True, 'OK'

    def _execute_validated_final_approach(self, path):
        """Execute only VAMP's prevalidated Cartesian waypoint sequence."""
        if self.state != 'IDLE':
            return False, 'MASTER_NODE_BUSY'
        if not path:
            return False, 'FINAL_APPROACH_EMPTY_VALIDATED_PATH'
        self.state = 'MOVING'
        try:
            for index, waypoint in enumerate(path, start=1):
                if self.physical_collision_detected or self.collision_detected:
                    return False, (
                        f'FINAL_APPROACH_COLLISION_BEFORE_WAYPOINT_{index}')
                if len(waypoint) != 6:
                    return False, (
                        f'FINAL_APPROACH_INVALID_WAYPOINT_{index}')
                # No APF dodge, global replan, RGB tracking, or alternate IK
                # is permitted after VAMP has returned the validated sequence.
                if not self.move_arm_to(
                        waypoint, disable_collision_monitor=False,
                        tolerance=0.01):
                    return False, (
                        f'FINAL_APPROACH_EXECUTION_FAILED_AT_WAYPOINT_{index}')
                if self.physical_collision_detected or self.collision_detected:
                    return False, (
                        f'FINAL_APPROACH_COLLISION_AT_WAYPOINT_{index}')
            return True, 'OK'
        finally:
            self.state = 'IDLE'

    def _execute_refined_final_approach(self, path, refined_width):
        """Run the gated open/slow-cartesian/close sequence, without lifting."""
        success = False
        motion_completed = False
        gripper_contact = False
        status = 'FINAL_APPROACH_NOT_STARTED'
        # Once the prevalidated plan is accepted, every exit path restores the
        # controller speed, including an opening or speed-command failure.
        restore_speed_required = True
        restore_speed_failed = False
        try:
            opening = min(0.10, max(0.0, float(refined_width) + 0.01))
            if not self._command_gripper(opening, self.gripper_open_force_n):
                status = 'FINAL_APPROACH_GRIPPER_OPEN_COMMAND_FAILED'
            elif not self._wait_for_gripper_opening(opening):
                status = 'FINAL_APPROACH_GRIPPER_OPEN_FEEDBACK_FAILED'
            else:
                if not self.set_arm_speed(5):
                    status = 'FINAL_APPROACH_SPEED_5_PERCENT_FAILED'
                else:
                    path_success, path_status = (
                        self._execute_validated_final_approach(path))
                    if not path_success:
                        status = path_status
                    else:
                        motion_completed = True
                        if not self._command_gripper(0.0, 1.0):
                            status = 'FINAL_APPROACH_GRIPPER_CLOSE_COMMAND_FAILED'
                        elif not self._wait_for_gripper_contact():
                            status = (
                                'GRIPPER_CONTACT_UNCONFIRMED: close command '
                                'was issued, but feedback did not confirm contact')
                        else:
                            gripper_contact = True
                            success = True
                            status = 'FINAL_APPROACH_COMPLETE_CONTACT_CONFIRMED'
        except Exception as error:
            status = f'FINAL_APPROACH_EXCEPTION: {error}'
        finally:
            if restore_speed_required:
                try:
                    restore_speed_failed = not self.set_arm_speed(20)
                except Exception as error:
                    restore_speed_failed = True
                    self.get_logger().error(
                        f'[SPEED COMMAND] Exception restoring 20% speed: {error}')
        if restore_speed_failed:
            success = False
            status = f'{status}; ARM_SPEED_RESTORE_20_PERCENT_FAILED'
        return success, motion_completed, gripper_contact, status

    def _run_grasp_joint_target(
            self, joints, allowed_contact_center=None,
            allowed_contact_radius_m=0.0):
        move_request = ToggleContinuousNavigation.Request()
        move_request.target_joints = list(joints)
        move_response = ToggleContinuousNavigation.Response()
        if allowed_contact_center is None or allowed_contact_radius_m <= 0.0:
            return self.move_to_joints_callback(
                move_request, move_response).success

        return self._move_grasp_target_with_contact_allowance(
            move_request.target_joints, allowed_contact_center,
            allowed_contact_radius_m)

    def _lift_confirmed_grasp(self, candidate, contact_center):
        """Lift a held object in base +Z using VAMP pose IK and path planning."""
        if self.post_grasp_lift_m <= 0.0:
            return True
        if not self.look_at_ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                '[GRASP] VAMP pose-IK service unavailable for post-grasp lift.')
            return False
        if (self.current_joint_states is not None and
                len(self.current_joint_states.position) >= 6):
            current_joints = list(self.current_joint_states.position[:6])
        else:
            current_joints = list(candidate.grasp_joints)
        if len(current_joints) != 6:
            self.get_logger().error(
                '[GRASP] No valid joint state available for post-grasp lift.')
            return False

        lift_request = LookAt.Request()
        lift_request.use_tcp_pose = True
        lift_request.tcp_pose = candidate.grasp_pose.pose
        lift_request.tcp_pose.position.z += self.post_grasp_lift_m
        lift_request.ignore_collision = True
        lift_request.current_joints = current_joints
        future = self.look_at_ik_client.call_async(lift_request)
        while rclpy.ok() and not future.done():
            time.sleep(0.05)
        result = future.result()
        if not result.success or len(result.joint_angles) != 6:
            self.get_logger().error(
                f'[GRASP] VAMP pose IK rejected the '
                f'{self.post_grasp_lift_m:.3f} m post-grasp lift.')
            return False
        self.get_logger().info(
            f'[GRASP] Lifting confirmed grasp by {self.post_grasp_lift_m:.3f} m '
            'with direct IK execution (bypassing VAMP replanning).')
        return self.execute_trajectory_with_monitoring(
            [result.joint_angles], check_interval=float('inf'), disable_collision_monitor=True)

    def _move_grasp_target_with_contact_allowance(
            self, target_joints, contact_center, contact_radius_m):
        """Plan and execute while excluding only the intended contact volume."""
        if self.state != 'IDLE':
            return False
        if (not self.current_joint_states or
                len(self.current_joint_states.position) < 6):
            self.get_logger().error(
                '[GRASP] Current joint feedback is unavailable.')
            return False
        if not self.check_replan_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                '[GRASP] VAMP path-planning service is unavailable.')
            return False
        self.state = 'BUSY'
        try:
            request = CheckAndReplan.Request()
            request.current_joints = list(
                self.current_joint_states.position[:6])
            request.remaining_path_flat = list(target_joints)
            request.allow_target_contact = True
            request.allowed_contact_center = contact_center
            request.allowed_contact_radius_m = float(contact_radius_m)
            future = self.check_replan_client.call_async(request)
            while rclpy.ok() and not future.done():
                time.sleep(0.05)
            result = future.result()
            if not result.success or not result.path_flat:
                self.get_logger().error(
                    '[GRASP] VAMP rejected motion even with the local target '
                    'contact allowance; no motion was executed.')
                return False
            path = [
                list(result.path_flat[index:index + 6])
                for index in range(0, len(result.path_flat), 6)
            ]
            self.get_logger().info(
                f'[GRASP] Executing VAMP trajectory with {len(path)} '
                f'waypoints; allowed-contact radius='
                f'{contact_radius_m:.3f}m.')
            return self.execute_trajectory_with_monitoring(
                path, check_interval=float('inf'))
        finally:
            self.state = 'IDLE'

    def execute_grasp_callback(self, request, response):
        cache_key = request.object_id or request.object_name.casefold()
        self.cached_grasp_plans[cache_key] = {
            'name': request.object_name,
            'header': request.observation_header,
            'candidate': request.candidate,
        }
        if not request.execute:
            # A newly reviewed FIND_GRASP result invalidates any close-up
            # refinement derived from an older observation.
            self.refined_grasp_plans.pop(cache_key, None)
            response.success = True
            response.status = (
                f'PLAN_CACHED: {request.object_name}; no motion requested')
            return response

        if not self.allow_physical_grasp_execution:
            response.success = False
            response.status = (
                'PHYSICAL_EXECUTION_DISABLED: plan was cached but no arm or '
                'gripper command was published')
            self.get_logger().warn(f'[GRASP] {response.status}')
            return response
        if request.refine_only:
            # Discard an older local refinement before moving and rebuilding
            # it from the new pre-grasp observation.
            self.refined_grasp_plans.pop(cache_key, None)
        if not request.candidate.ik_valid:
            response.success = False
            response.status = 'REJECTED: candidate has no valid IK solution'
            return response
        if (self.require_grasp_collision_validation and
                (not request.candidate.collision_checked or
                 not request.candidate.grasp_collision_valid)):
            response.success = False
            response.status = (
                'REJECTED: a collision-checked, collision-free grasp '
                'endpoint is required for physical execution')
            return response
        if (request.candidate.pregrasp_collision_overridden and
                not self.allow_pregrasp_collision_override):
            response.success = False
            response.status = (
                'REJECTED: VAMP found collision at the pre-grasp pose and '
                'allow_pregrasp_collision_override is false')
            return response
        body_model_candidate = request.candidate.validation_status.startswith(
            'CUP_BODY_MODEL_')
        if (body_model_candidate and
                not self.allow_model_based_body_grasp_execution):
            response.success = False
            response.status = (
                'REJECTED: model-derived cup-body grasp requires '
                'allow_model_based_body_grasp_execution=true')
            return response
        if (self.require_measured_contacts_for_grasp and
                not request.candidate.measured_contacts_valid and
                not (body_model_candidate and
                     self.allow_model_based_body_grasp_execution)):
            response.success = False
            response.status = (
                'REJECTED: measured opposing contacts and clear finger sweep '
                'are required')
            return response
        if (self.require_positive_gpd_score_for_grasp and
                request.candidate.score <= 0.0):
            response.success = False
            response.status = 'REJECTED: GPD classifier score is not positive'
            return response

        refined_plan = None
        if not request.refine_only:
            refined_plan = self.refined_grasp_plans.get(cache_key)
            if refined_plan is None:
                response.success = False
                response.status = (
                    'PRE_GRASP_REQUIRED: run [PRE_GRASP] successfully before '
                    '[GRASP]; no refined final grasp is cached')
                return response
            if refined_plan['candidate_id'] != int(request.candidate.id):
                response.success = False
                response.status = 'REFINED_GRASP_CANDIDATE_MISMATCH'
                return response
            at_pregrasp, pregrasp_status = self._verify_refined_pregrasp(
                refined_plan)
            if not at_pregrasp:
                response.success = False
                response.status = pregrasp_status
                return response

        context_id = 'grasp_' + uuid.uuid4().hex

        def call_step(name, query):
            _, _, result = self.step_runtime.store.run_one(
                context_id, uuid.uuid4().hex, name, serialize_message(query))
            if not result.success:
                raise RuntimeError(result.status)
            from task_pipeline.registry import FUNCTIONS
            return deserialize_message(bytes(result.response_cdr), FUNCTIONS[name].service_type.Response)

        stage_request = GraspStage.Request(object_name=cache_key)
        try:
            if request.refine_only:
                call_step('move_pregrasp', stage_request)
                response.motion_completed = True
                time.sleep(0.1)
                refined = None
                for attempt in range(3):
                    call_step('capture_observation', CaptureObservation.Request())
                    refine_query = RefineGrasp.Request(
                        previous_observation_header=request.observation_header,
                        candidate=request.candidate)
                    # Transient missing-frame replies are valid typed failures,
                    # not permission to move the arm or replace the grasp seed.
                    _, _, item = self.step_runtime.store.run_one(
                        context_id, uuid.uuid4().hex, 'refine_grasp', serialize_message(refine_query))
                    refined = deserialize_message(bytes(item.response_cdr), RefineGrasp.Response)
                    if refined.valid or not refined.failure_reason.startswith((
                            'SYNCHRONIZED_PREGRASP_RGBD_UNAVAILABLE',
                            'NO_NEW_PREGRASP_OBSERVATION')):
                        break
                if refined is None or not refined.valid:
                    raise RuntimeError('GRASP_REFINEMENT_REJECTED: ' + (
                        'no response' if refined is None else refined.failure_reason))
                call_step('cache_grasp_refinement', CacheGraspRefinement.Request(
                    object_name=cache_key, refinement_cdr=serialize_message(refined)))
                response.success = True
                response.status = json.dumps({
                    'status': 'REFINED_GRASP_READY', 'candidate_id': refined.candidate_id,
                    'failure_reason': refined.failure_reason,
                    'seed_pixel': [refined.seed_u, refined.seed_v],
                    'grasp_line_pixels': [[refined.left_u, refined.left_v],
                                          [refined.right_u, refined.right_v]],
                    'seed_3d_base': [refined.corrected_seed_base.x,
                                     refined.corrected_seed_base.y,
                                     refined.corrected_seed_base.z],
                    'closing_angle_rad': refined.closing_angle_rad,
                    'grasp_width_m': refined.grasp_width_m,
                    'usable_grasp_depth_m': refined.usable_grasp_depth_m,
                    'debug_topic': '/vamp_planner/refined_grasp_debug',
                    'context_id': context_id})
            else:
                for name in ('plan_grasp_approach', 'open_grasp_gripper',
                             'execute_grasp_approach', 'close_grasp_gripper',
                             'verify_grasp_contact'):
                    stage = call_step(name, stage_request)
                    response.motion_completed = response.motion_completed or stage.motion_completed
                    response.gripper_contact = response.gripper_contact or stage.gripper_contact
                response.success = True
                response.status = 'FINAL_APPROACH_COMPLETE_CONTACT_CONFIRMED'
        except Exception as error:
            response.success = False
            response.status = str(error)
        finally:
            if not request.refine_only:
                if not self.set_arm_speed(20):
                    response.success = False
                    response.status += '; ARM_SPEED_RESTORE_20_PERCENT_FAILED'
        return response

    def move_to_joints_callback(self, request, response):
        if self.state != "IDLE":
            response.success = False
            return response
        if len(request.target_joints) != 6:
            self.get_logger().error(f"move_to_joints requires 6 joint angles, got {len(request.target_joints)}")
            response.success = False
            return response
        
        target_goal = list(request.target_joints)
        self.state = "BUSY"
        self.get_logger().info(f"[VAMP GUIDED MOVE] Master received target joints request: {[round(x, 3) for x in target_goal]}")

        # Reset physical collision monitor for new movement
        self.physical_collision_detected = False
        for q in self.torque_history:
            q.clear()

        try:
            max_loops = 5
            for loop_count in range(max_loops):
                if not self.current_joint_states or len(self.current_joint_states.position) < 6:
                    self.get_logger().warn("Waiting for current joint states...")
                    time.sleep(0.5)
                    continue

                curr_pos = list(self.current_joint_states.position[:6])
                errors = [abs(c - g) for c, g in zip(curr_pos, target_goal)]
                if all(e <= 0.01 for e in errors):
                    self.get_logger().info("[VAMP GUIDED MOVE] Target goal successfully reached!")
                    response.success = True
                    return response

                self.get_logger().info(f"[VAMP GUIDED MOVE] Loop {loop_count+1}/{max_loops}: Requesting VAMP path planning to target...")

                if not self.check_replan_client.wait_for_service(timeout_sec=3.0):
                    self.get_logger().error("VAMP Planner service /vamp_planner/check_and_replan not available!")
                    response.success = False
                    return response

                req_replan = CheckAndReplan.Request()
                req_replan.current_joints = curr_pos
                req_replan.remaining_path_flat = target_goal

                future = self.check_replan_client.call_async(req_replan)
                while rclpy.ok() and not future.done():
                    time.sleep(0.05)

                res_replan = future.result()
                if not res_replan.success or not res_replan.path_flat:
                    self.get_logger().error("[MASTER NODE] No safe trajectory returned from VAMP planner! ABORTING: Staying still, no fallback raw movement will be performed.")
                    response.success = False
                    return response
                
                planned_path = [list(res_replan.path_flat[i:i+6]) for i in range(0, len(res_replan.path_flat), 6)]

                self.get_logger().info(f"[VAMP GUIDED MOVE] Executing VAMP trajectory with {len(planned_path)} waypoints...")
                success = self.execute_trajectory_with_monitoring(planned_path, check_interval=float('inf'))
                if not success:
                    self.get_logger().warn("[VAMP GUIDED MOVE] Trajectory interrupted or aborted.")

            # Final position check
            if self.current_joint_states and len(self.current_joint_states.position) >= 6:
                curr_pos = list(self.current_joint_states.position[:6])
                errors = [abs(c - g) for c, g in zip(curr_pos, target_goal)]
                response.success = all(e <= 0.02 for e in errors)
            else:
                response.success = False
            return response

        finally:
            self.state = "IDLE"

    def look_at_target_callback(self, request, response):
        if self.state != "IDLE":
            response.success = False
            return response
            
        self.get_logger().info(f"[MASTER NODE] Requesting NBV to look at target: ({request.look_at_x:.3f}, {request.look_at_y:.3f}, {request.look_at_z:.3f})")

        if not self.get_nbv_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Vision get_nbv_candidates service not available!")
            response.success = False
            return response
            
        if not self.look_at_ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("VAMP look_at_ik service not available!")
            response.success = False
            return response

        # 1. Fetch unoccluded NBV candidates
        req_nbv = GetNBVPose.Request()
        req_nbv.target_x = request.look_at_x
        req_nbv.target_y = request.look_at_y
        req_nbv.target_z = request.look_at_z
        req_nbv.radius = 0.3 # 30cm radius
        
        future_nbv = self.get_nbv_client.call_async(req_nbv)
        while rclpy.ok() and not future_nbv.done():
            time.sleep(0.05)
            
        res_nbv = future_nbv.result()
        if not res_nbv.success or len(res_nbv.candidate_x) == 0:
            self.get_logger().error("[MASTER NODE] Vision Node failed to find any unoccluded NBV candidates.")
            response.success = False
            return response
            
        self.get_logger().info(f"[MASTER NODE] Found {len(res_nbv.candidate_x)} unoccluded NBV candidates. Testing IK...")
        
        # 2. Iterate candidates for reachable IK
        best_joints = None
        for i in range(len(res_nbv.candidate_x)):
            req_ik = LookAt.Request()
            req_ik.look_at_x = request.look_at_x
            req_ik.look_at_y = request.look_at_y
            req_ik.look_at_z = request.look_at_z
            req_ik.current_pose_x = res_nbv.candidate_x[i]
            req_ik.current_pose_y = res_nbv.candidate_y[i]
            req_ik.current_pose_z = res_nbv.candidate_z[i]
            if not self.populate_look_at_ik_request(
                    req_ik,
                    (res_nbv.candidate_x[i], res_nbv.candidate_y[i],
                     res_nbv.candidate_z[i]),
                    (request.look_at_x, request.look_at_y,
                     request.look_at_z)):
                continue
            if self.current_joint_states and len(self.current_joint_states.position) >= 6:
                req_ik.current_joints = list(self.current_joint_states.position[:6])
            else:
                req_ik.current_joints = [0.0] * 6
            
            future = self.look_at_ik_client.call_async(req_ik)
            while rclpy.ok() and not future.done():
                time.sleep(0.05)
                
            res_ik = future.result()
            if res_ik.success and len(res_ik.joint_angles) == 6:
                best_joints = res_ik.joint_angles
                self.get_logger().info(f"[MASTER NODE] Candidate {i} is reachable! Handing off to move_to_joints...")
                break
                
        if best_joints is None:
            self.get_logger().error("[MASTER NODE] VAMP failed to find IK solution for any NBV candidates.")
            response.success = False
            return response

        # Use existing move_to_joints behavior by wrapping request
        move_req = ToggleContinuousNavigation.Request()
        move_req.target_joints = best_joints
        move_res = ToggleContinuousNavigation.Response()
        
        move_res = self.move_to_joints_callback(move_req, move_res)
        response.success = move_res.success
        response.joint_angles = best_joints
        return response

    def move_relative_callback(self, request, response):
        if self.state != "IDLE":
            response.success = False
            response.message = "Node is BUSY."
            return response

        if not self.current_tcp_pose:
            response.success = False
            response.message = "No TCP pose available."
            return response
            
        tcp = np.array([
            self.current_tcp_pose.pose.position.x,
            self.current_tcp_pose.pose.position.y,
            self.current_tcp_pose.pose.position.z
        ])
        target = np.array([request.target_x, request.target_y, request.target_z])
        
        # Calculate Forward vector
        forward = target - tcp
        norm = np.linalg.norm(forward)
        if norm < 0.001:
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward = forward / norm
            
        up_global = np.array([0.0, 0.0, 1.0])
        
        # Calculate Right vector
        right = np.cross(up_global, forward)
        if np.linalg.norm(right) < 0.001:
            right = np.array([0.0, 1.0, 0.0])
        else:
            right = right / np.linalg.norm(right)
            
        # Calculate Up local vector
        up_local = np.cross(forward, right)
        
        shift = np.array([0.0, 0.0, 0.0])
        d = request.direction.upper()
        s = request.step_size
        
        if d == "CLOSER":
            shift = forward * s
        elif d == "FURTHER":
            shift = -forward * s
        elif d == "LEFT":
            shift = -right * s
        elif d == "RIGHT":
            shift = right * s
        elif d == "UP":
            shift = up_local * s
        elif d == "DOWN":
            shift = -up_local * s
        else:
            response.success = False
            response.message = f"Invalid direction: {d}"
            return response
            
        new_tcp = tcp + shift
        
        # Call LookAtIK with new_tcp
        req_ik = LookAt.Request()
        req_ik.look_at_x = request.target_x
        req_ik.look_at_y = request.target_y
        req_ik.look_at_z = request.target_z
        req_ik.current_pose_x = new_tcp[0]
        req_ik.current_pose_y = new_tcp[1]
        req_ik.current_pose_z = new_tcp[2]
        
        if self.current_joint_states and len(self.current_joint_states.position) >= 6:
            req_ik.current_joints = list(self.current_joint_states.position[:6])
        else:
            req_ik.current_joints = [0.0] * 6
            
        self.get_logger().info(f"[MASTER NODE] Moving relative '{d}' by {s}m. IK target: {new_tcp}")
        
        future = self.look_at_ik_client.call_async(req_ik)
        while rclpy.ok() and not future.done():
            time.sleep(0.05)
            
        res_ik = future.result()
        if res_ik.success and len(res_ik.joint_angles) == 6:
            move_req = ToggleContinuousNavigation.Request()
            move_req.target_joints = res_ik.joint_angles
            move_res = ToggleContinuousNavigation.Response()
            
            move_res = self.move_to_joints_callback(move_req, move_res)
            response.success = move_res.success
            response.joint_angles = res_ik.joint_angles
            response.message = "Relative move executed."
        else:
            response.success = False
            response.message = "VAMP failed to find IK solution for relative move."
            self.get_logger().error(response.message)
            
        return response

    def start_global_scan_callback(self, request, response):
        if self.state != "IDLE":
            response.success = False
            response.message = "Node is BUSY."
            return response
            
        # Reset physical collision monitor for new movement
        self.physical_collision_detected = False
        for q in self.torque_history:
            q.clear()
            
        success, msg = self.perform_global_scan()
        response.success = success
        response.message = msg
        return response

    def execute_fallback_sequence(self, target_joints=None):
        self.get_logger().info("[FALLBACK] Executing fallback sequence... (target_joints=%s)" % (
            [round(x, 3) for x in target_joints] if target_joints is not None else 'None'))
        
        # 1. Get Obstacle Location
        if not self.get_obstacle_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("[FALLBACK] /active_vision/get_obstacle_location service unavailable!")
            return False
            
        req_obs = GetObstacleLocation.Request()
        future_obs = self.get_obstacle_client.call_async(req_obs)
        while rclpy.ok() and not future_obs.done():
            time.sleep(0.05)
            
        res_obs = future_obs.result()
            
        obstacle_point = res_obs.point
        self.get_logger().info(f"[FALLBACK] Obstacle location retrieved: [{obstacle_point[0]:.3f}, {obstacle_point[1]:.3f}, {obstacle_point[2]:.3f}]")
        
        # 2. Get current TCP
        if not self.current_tcp_pose:
            self.get_logger().error("[FALLBACK] Current TCP pose unknown!")
            return False
        
        tcp_point = [
            self.current_tcp_pose.pose.position.x,
            self.current_tcp_pose.pose.position.y,
            self.current_tcp_pose.pose.position.z
        ]
        self.get_logger().info(f"[FALLBACK] Current TCP: [{tcp_point[0]:.3f}, {tcp_point[1]:.3f}, {tcp_point[2]:.3f}]")
        
        # 3. Calculate Fallback Pose (~40cm retreat from obstacle)
        if not self.calc_fallback_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("[FALLBACK] /vamp_planner/calculate_fallback_pose service unavailable!")
            return False
            
        req_calc = CalculateFallbackPose.Request()
        req_calc.obstacle_point_base_link = obstacle_point
        req_calc.tcp_point_base_link = tcp_point
        req_calc.current_joints = list(self.current_joint_states.position[:6])
        
        self.get_logger().info(f"[FALLBACK] Requesting fallback pose from VAMP (current joints: {[round(x, 3) for x in self.current_joint_states.position[:6]]})")
        future_calc = self.calc_fallback_client.call_async(req_calc)
        while rclpy.ok() and not future_calc.done():
            time.sleep(0.05)
            
        res_calc = future_calc.result()
        if not res_calc.success:
            self.get_logger().error("[FALLBACK] VAMP failed to find a valid fallback pose!")
            return False
            
        fallback_joints = list(res_calc.fallback_joints)
        self.get_logger().info(f"[FALLBACK] Valid fallback pose found: {[round(x, 3) for x in fallback_joints]}")
        
        # 4. Move to fallback pose (collision monitor disabled — we're retreating)
        self.get_logger().info("[FALLBACK] Moving to fallback pose (collision monitor OFF)...")
        self.state = "MOVING"
        if not self.move_arm_to(fallback_joints, disable_collision_monitor=True):
            self.get_logger().error("[FALLBACK] Failed to reach fallback pose!")
            return False
            
        # 5. Perform scan at fallback pose to update global 3D map
        self.get_logger().info("[FALLBACK] Reached fallback pose. Triggering scan to update global 3D map...")
        self.state = "IDLE"
        self.trigger_single_scan()
        time.sleep(2.0) # Wait 2 seconds for scan to complete and map to merge
        
        if target_joints is not None:
            self.get_logger().info(f"[FALLBACK] Attempting APF Local Planner: fallback -> waypoint {[round(x, 3) for x in target_joints]}")
            if not self.plan_apf_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().error("[FALLBACK] /vamp_planner/plan_apf service unavailable!")
                return False
                
            req_apf = PlanAPF.Request()
            req_apf.current_joints = fallback_joints
            req_apf.target_joints = list(target_joints)
            
            future_apf = self.plan_apf_client.call_async(req_apf)
            while rclpy.ok() and not future_apf.done():
                time.sleep(0.05)
                
            res_apf = future_apf.result()
            if res_apf.success:
                path_flat = list(res_apf.path_flat)
                num_waypoints = len(path_flat) // 6
                self.get_logger().info(f"[FALLBACK] APF dodge path generated ({num_waypoints} waypoints)! Executing at reduced speed...")
                self.set_arm_speed(5)
                path_unflat = [path_flat[i:i+6] for i in range(0, len(path_flat), 6)]
                success = self.execute_trajectory_with_monitoring(path_unflat, check_interval=float('inf'), disable_collision_monitor=True)
                self.set_arm_speed(20)
                if success:
                    self.get_logger().info("[FALLBACK] APF dodge maneuver successful!")
                    return True
                else:
                    self.get_logger().error("[FALLBACK] APF dodge maneuver failed during execution.")
                    return False
            else:
                self.get_logger().warn("[FALLBACK] APF Local Planner failed (local minima). Full global replan required.")
                return False
        
        return True

    def perform_global_scan(self):
        if self.hand_eye_matrix is None:
            return False, "Cannot scan without hand-eye calibration matrix."

        self.state = "BUSY"
        try:
            self.get_logger().info("Starting Global Voxel Mapping Scan...")
            
            i = 0
            while i < len(self.scan_poses):
                pose = self.scan_poses[i]
                self.state = "MOVING"
                
                self.get_logger().info(f"Moving to scan pose {i+1}...")
                if not self.move_arm_to(pose):
                    if self.physical_collision_detected:
                        self.get_logger().error(f"PHYSICAL COLLISION aborted global scan at pose {i+1}.")
                        self.state = "IDLE"
                        return False, "Global scan aborted due to physical collision."
                    elif self.collision_detected:
                        self.get_logger().warn(f"Collision detected while moving to scan pose {i+1}. Initiating fallback sequence.")
                        if self.execute_fallback_sequence(target_joints=pose):
                            self.get_logger().info(f"APF local dodge reached pose {i+1} successfully.")
                        else:
                            self.get_logger().error(f"Fallback and APF failed. Aborting global scan.")
                            self.state = "IDLE"
                            return False, "Global scan aborted due to failed fallback/APF."
                    else:
                        self.get_logger().warn(f"Failed to reach scan pose {i+1} without collision. Skipping.")
                        i += 1
                        continue
                
                self.state = "IDLE"
                self.trigger_single_scan()
                
                self.state = "BUSY"
                i += 1

            self.get_logger().info("Returning home...")
            while True:
                self.state = "MOVING"
                if not self.move_arm_to([0, 0, 0, 0, 0, 0]):
                    if self.physical_collision_detected:
                        self.get_logger().error(f"PHYSICAL COLLISION aborted returning home.")
                        break
                    elif self.collision_detected:
                        self.get_logger().warn(f"Collision detected while returning home. Initiating fallback sequence.")
                        if self.execute_fallback_sequence():
                            continue
                        else:
                            self.get_logger().error(f"Fallback failed returning home.")
                            break
                    else:
                        break # Failed for other reason
                break
                
            self.state = "IDLE"
            return True, "Global scan complete!"
        finally:
            if self.state != "IDLE":
                self.state = "IDLE"

        self.get_logger().info("Exporting Global Point Cloud...")
        if self.export_client.wait_for_service(timeout_sec=5.0):
            req = Trigger.Request()
            future = self.export_client.call_async(req)
            while rclpy.ok() and not future.done():
                time.sleep(0.1)
            res = future.result()
            if res.success:
                self.get_logger().info(res.message)
            else:
                self.get_logger().error(res.message)
        else:
            self.get_logger().error("VAMP export service not available.")

        self.state = "IDLE"
        self.last_idle_time = time.time()
        return True, "Global scan complete!"

    def start_continuous_nav_callback(self, request, response):
        if self.state != "IDLE":
            response.success = False
            response.message = f"Node is BUSY (current state: {self.state})."
            return response
        
        target = list(request.target_joints) if len(request.target_joints) == 6 else [-0.8, 2.0, -1.0, 0.0, -0.5, 0.0]
        threading.Thread(target=self.perform_continuous_navigation, args=(target,), daemon=True).start()
        response.success = True
        response.message = f"Continuous navigation started in background thread with target {target}."
        return response

    def perform_continuous_navigation(self, target_goal):
        self.state = "NAVIGATING"
        self.get_logger().info(f"Starting Continuous Navigation Mode towards target {target_goal}...")

        # Reset physical collision monitor for new movement
        self.physical_collision_detected = False
        for q in self.torque_history:
            q.clear()

        if not self.toggle_nav_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("ToggleContinuousNavigation service not available!")
            self.state = "IDLE"
            return

        req_toggle = ToggleContinuousNavigation.Request()
        req_toggle.target_joints = target_goal
        future_toggle = self.toggle_nav_client.call_async(req_toggle)
        while rclpy.ok() and not future_toggle.done():
            time.sleep(0.05)
        
        res_toggle = future_toggle.result()
        if not res_toggle or not res_toggle.success:
            self.get_logger().error("Failed to enable continuous navigation in VAMP.")
            self.state = "IDLE"
            return

        if not self.check_replan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("CheckAndReplan service not available!")
            self.state = "IDLE"
            return

        req_replan = CheckAndReplan.Request()
        if self.current_joint_states and len(self.current_joint_states.position) >= 6:
            req_replan.current_joints = list(self.current_joint_states.position[:6])
        else:
            req_replan.current_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        req_replan.remaining_path_flat = target_goal

        with self.path_lock:
            self.latest_replanned_path = None

        self.get_logger().info("Requesting initial trajectory from VAMP...")
        future_replan = self.check_replan_client.call_async(req_replan)
        while rclpy.ok() and not future_replan.done():
            time.sleep(0.05)

        res_replan = future_replan.result()
        initial_path = None
        if res_replan and res_replan.success:
            path_flat = list(res_replan.path_flat)
            if len(path_flat) >= 6 and len(path_flat) % 6 == 0:
                initial_path = [
                    path_flat[index:index + 6]
                    for index in range(0, len(path_flat), 6)
                ]

        if not initial_path:
            self.get_logger().error("[MASTER NODE] VAMP failed to find a valid trajectory! ABORTING: No fallback raw movement will be executed.")
            req_toggle.target_joints = []
            self.toggle_nav_client.call_async(req_toggle)
            self.state = "IDLE"
            return

        success = self.execute_trajectory_with_monitoring(initial_path, check_interval=float('inf'))
        if success:
            self.get_logger().info("Continuous navigation reached goal successfully!")
        else:
            self.get_logger().warn("Continuous navigation aborted or failed.")

        req_toggle.target_joints = []
        self.toggle_nav_client.call_async(req_toggle)
        self.state = "IDLE"
        self.last_idle_time = time.time()

    def execute_trajectory_with_monitoring(self, path, check_interval=3.0, disable_collision_monitor=False):
        monitor_camera_collision = self.enable_camera_collision_monitor and not disable_collision_monitor
        self.get_logger().info(f"[TRAJECTORY] Starting trajectory execution: {len(path)} waypoints, collision_monitor={'ON' if monitor_camera_collision else 'OFF'}")
        if not monitor_camera_collision:
            self.ignore_camera_collisions = True
        trajectory_succeeded = False
        try:
            with self.path_lock:
                self.current_path = list(path)
                self.path_index = 0
                self.executing_path = True
                self.latest_replanned_path = None

            while self.executing_path and rclpy.ok():
                with self.path_lock:
                    if self.path_index >= len(self.current_path):
                        self.get_logger().info("[TRAJECTORY] Reached end of trajectory!")
                        self.executing_path = False
                        trajectory_succeeded = True
                        break
                    target_pose = self.current_path[self.path_index]
                    total_waypoints = len(self.current_path)
                
                self.state = "MOVING"
                # Use looser tolerance for intermediate waypoints during dodging for smoother movement
                is_last_waypoint = (self.path_index == total_waypoints - 1)
                current_tolerance = 0.08 if (disable_collision_monitor and not is_last_waypoint) else 0.02
                self.get_logger().info(f"[TRAJECTORY] Waypoint {self.path_index+1}/{total_waypoints}: {[round(x, 3) for x in target_pose]} (tol={current_tolerance})")
                arrived = self.move_arm_to(target_pose, disable_collision_monitor=disable_collision_monitor, tolerance=current_tolerance)
                self.state = "IDLE"
                
                # Sleep to let the scanner thread detect IDLE and scan (only for global path, avoid stutter in dodge)
                if monitor_camera_collision:
                    time.sleep(0.2)
        
                if self.physical_collision_detected:
                    self.get_logger().error("[TRAJECTORY] Execution ABORTED due to physical collision!")
                    self.executing_path = False
                    break
        
                if monitor_camera_collision and (self.collision_detected or not arrived):
                    reason = "Obstacle Collision" if self.collision_detected else "Position Mismatch / Timeout"
                    self.get_logger().warn(f"[TRAJECTORY] {reason} at waypoint {self.path_index+1}/{total_waypoints}!")
            
                    if self.collision_detected:
                        # APF targets the CURRENT waypoint we were trying to reach (single-waypoint dodge)
                        # NOT the final trajectory goal — APF only interrupts one waypoint
                        apf_target = target_pose
                        self.get_logger().info(f"[TRAJECTORY] Initiating fallback+APF to dodge to current waypoint: {[round(x, 3) for x in apf_target]}")
                        if self.execute_fallback_sequence(target_joints=apf_target):
                            self.get_logger().info(f"[TRAJECTORY] APF dodge successful! Resuming normal trajectory from waypoint {self.path_index+2}/{total_waypoints}...")
                            # APF only interrupts a single waypoint — advance and continue normally
                            with self.path_lock:
                                self.path_index += 1
                                self.collision_detected = False
                            continue
                        else:
                            self.get_logger().info("[TRAJECTORY] APF dodge failed, falling back to VAMP global replanner.")
            
                    if self.check_replan_client.wait_for_service(timeout_sec=2.0):
                        self.state = "BUSY"
                        req = CheckAndReplan.Request()
                        req.current_joints = list(self.current_joint_states.position[:6]) if self.current_joint_states else list(target_pose)
                
                        # Send only the final goal to force RRTC to plan a brand new path from the current position
                        with self.path_lock:
                            req.remaining_path_flat = list(self.current_path[-1])
                
                        self.get_logger().info("[TRAJECTORY] Calling /vamp_planner/check_and_replan for global replan...")
                        future = self.check_replan_client.call_async(req)
                        while rclpy.ok() and not future.done():
                            time.sleep(0.05)
                
                        res = future.result()
                        if res and res.success:
                            path_flat = list(res.path_flat)
                            if len(path_flat) >= 6:
                                new_path = [path_flat[i:i+6] for i in range(0, len(path_flat), 6)]
                                self.get_logger().info(f"[TRAJECTORY] Adopting new globally replanned trajectory ({len(new_path)} waypoints)!")
                                with self.path_lock:
                                    self.current_path = new_path
                                    self.path_index = 0
                                continue
                            else:
                                self.get_logger().info("[TRAJECTORY] Path clear or no replan returned. Advancing...")
                        else:
                            self.get_logger().error("[TRAJECTORY] VAMP global replanner failed to find a valid path! Aborting.")
                            self.executing_path = False
                            self.state = "IDLE"
                            return False
                    
                elif not monitor_camera_collision and not arrived:
                    mode_str = "unmonitored dodge" if disable_collision_monitor else "unmonitored trajectory"
                    self.get_logger().warn(f"[TRAJECTORY] Position Mismatch / Timeout during {mode_str} at waypoint {self.path_index+1}! Aborting.")
                    self.executing_path = False
                    break
                    
                # If no collision or replanning (or successfully reached), advance
                with self.path_lock:
                    self.path_index += 1
                    self.collision_detected = False
    
            self.state = "IDLE"
            return trajectory_succeeded
        finally:
            if not monitor_camera_collision:
                self.ignore_camera_collisions = not self.enable_camera_collision_monitor
                self.get_logger().debug("[TRAJECTORY] Restored camera collision-monitor policy after trajectory")
            self.executing_path = False

    def start_calibration_callback(self, request, response):
        if self.state != "IDLE":
            response.success = False
            response.message = "Node is BUSY."
            return response
            
        self.state = "CALIBRATING"
        self.get_logger().info("Starting Hand-Eye Calibration...")

        # RGB and CameraInfo are continuously subscribed for both world-state
        # health reporting and calibration capture.
        time.sleep(2.0)
        
        if self.current_camera_info is None:
            self.state = "IDLE"
            self.set_arm_speed(20)
            response.success = False
            response.message = "No CameraInfo received. Cannot calibrate."
            return response

        self.set_arm_speed(80)  # Move at 80% speed for hand-eye calibration moves

        K = np.array(self.current_camera_info.k).reshape((3, 3))
        D = np.array(self.current_camera_info.d)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        board = cv2.aruco.CharucoBoard((5, 7), 0.0325, 0.01625, dictionary)
        charuco_detector = cv2.aruco.CharucoDetector(board)

        R_target2cam_list = []
        t_target2cam_list = []
        R_gripper2base_list = []
        t_gripper2base_list = []


        for i, pose in enumerate(self.calibration_poses):
            self.get_logger().info(f"[Hand-Eye] Moving to Pose {i+1}/{len(self.calibration_poses)}...")
            if not self.move_arm_to(pose):
                self.get_logger().warn(f"Failed to reach pose {i+1}. Skipping.")
                continue

            # Wait 4 seconds total (approx 2-3s for arm to reach target, plus >1s for it to completely stabilize)
            time.sleep(4.0)

            if self.current_rgb is None or self.current_joint_states is None or len(self.current_joint_states.position) < 6:
                continue

            try:
                cv_img = self.bridge.imgmsg_to_cv2(self.current_rgb, "bgr8")
            except Exception as e:
                continue

            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(cv_img)

            if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
                continue

            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, K, D, None, None)

            if not valid:
                continue

            R_target2cam, _ = cv2.Rodrigues(rvec)
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(tvec)

            joints = self.current_joint_states.position
            
            def urdf_rot(r, p, y):
                return Rotation.from_euler('xyz', [r, p, y]).as_matrix()

            T01 = np.eye(4)
            T01[2, 3] = 0.123
            T01[:3, :3] = urdf_rot(0, 0, 0) @ Rotation.from_euler('z', joints[0]).as_matrix()

            T12 = np.eye(4)
            T12[:3, :3] = urdf_rot(1.5707963, -0.1357866, -3.1415926) @ Rotation.from_euler('z', joints[1]).as_matrix()

            T23 = np.eye(4)
            T23[0, 3] = 0.28503
            T23[:3, :3] = urdf_rot(0, 0, -1.7938494) @ Rotation.from_euler('z', joints[2]).as_matrix()

            T34 = np.eye(4)
            T34[0, 3] = -0.02198
            T34[1, 3] = -0.25075
            T34[:3, :3] = urdf_rot(1.5707963, 0, 0) @ Rotation.from_euler('z', joints[3]).as_matrix()

            T45 = np.eye(4)
            T45[:3, :3] = urdf_rot(-1.5707963, 0, 0) @ Rotation.from_euler('z', joints[4]).as_matrix()

            T56 = np.eye(4)
            T56[0, 3] = 8.8259e-05
            T56[1, 3] = -0.091
            T56[:3, :3] = urdf_rot(1.5707963, 0, 0) @ Rotation.from_euler('z', joints[5]).as_matrix()

            T_g2b = T01 @ T12 @ T23 @ T34 @ T45 @ T56

            R_gripper2base_list.append(T_g2b[:3, :3])
            t_gripper2base_list.append(T_g2b[:3, 3].reshape(3, 1))
            self.get_logger().info(f"Captured data for pose {i+1}.")

        if len(R_target2cam_list) < 3:
            self.state = "IDLE"
            self.set_arm_speed(20)
            response.success = False
            response.message = "Not enough valid poses detected."
            return response

        self.get_logger().info("Computing Hand-Eye Calibration (Tsai)...")
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list, t_target2cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        X_matrix = np.eye(4)
        X_matrix[:3, :3] = R_cam2gripper
        X_matrix[:3, 3] = t_cam2gripper.flatten()

        self.hand_eye_matrix = X_matrix
        save_dir = os.path.expanduser('~/.ros')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, 'hand_eye_calibration.yaml')

        try:
            with open(save_file, 'w') as f:
                yaml.dump({'hand_eye_transform': X_matrix.tolist()}, f, default_flow_style=False)
            
            # Request VAMP to reload the matrix
            if self.reload_client.wait_for_service(timeout_sec=2.0):
                req = Trigger.Request()
                future = self.reload_client.call_async(req)
                while rclpy.ok() and not future.done():
                    time.sleep(0.1)
                res = future.result()
                if res.success:
                    self.get_logger().info(res.message)
                else:
                    self.get_logger().error(res.message)
            else:
                self.get_logger().error("VAMP reload calibration service not available.")

            response.success = True
            response.message = "Calibration completed successfully!"
        except Exception:
            response.success = False
            response.message = "Failed to save calibration file."

        self.set_arm_speed(20)
        self.state = "IDLE"
        self.last_idle_time = time.time()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PiperMasterControl()
    executor = MultiThreadedExecutor(num_threads=16)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
