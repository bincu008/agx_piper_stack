"""One public function catalog used by production and benchmark clients."""

from dataclasses import dataclass

from my_actions import srv

from std_srvs.srv import Trigger


@dataclass(frozen=True)
class Function:
    """A stable operation and its existing computational backend."""

    name: str
    service_type: object
    backend: str
    input_mode: str = 'observed'

    @property
    def endpoint(self):
        """Return the canonical API address."""
        return f'/pipeline/{self.name}'


FUNCTIONS = {item.name: item for item in (
    Function('commit_floor_slots', srv.CommitFloorSlots, '/placement/commit_floor_slots'),
    Function('capture_observation', srv.CaptureObservation,
             '/pipeline/_capture_observation', 'rgbd'),
    Function('run_step', srv.RunPipelineStep, '/master/run_step'),
    Function('submit_steps', srv.SubmitPipelineSteps, '/master/submit_steps'),
    Function('get_step_context', srv.GetStepContext, '/master/get_step_context'),
    Function('clear_step_context', srv.ClearStepContext, '/master/clear_step_context'),
    Function('store_step_result', srv.StorePipelineResult, '/master/store_step_result'),
    Function('search_placements', srv.SearchPlacements, '/placement/search_placements',
             'prepared_geometry'),
    Function('validate_placements', srv.PlanPlacement, '/vamp_planner/plan_placement'),
    Function('move_to', srv.MoveTo, '/master/move_to'),
    Function('execute_trajectory', srv.ExecuteTrajectory, '/master/execute_trajectory'),
    Function('set_gripper', srv.GripperCommand, '/master/set_gripper'),
    Function('select_grasp', srv.SelectGrasp, '/master/select_grasp'),
    Function('cache_grasp_plan', srv.CacheGraspPlan, '/master/cache_grasp_plan'),
    Function('get_cached_grasp', srv.GetCachedGrasp, '/master/get_cached_grasp'),
    Function('get_arm_state', srv.GetArmState, '/master/get_arm_state'),
    Function('cache_grasp_refinement', srv.CacheGraspRefinement,
             '/master/cache_grasp_refinement'),
    Function('move_pregrasp', srv.GraspStage, '/master/move_pregrasp'),
    Function('plan_grasp_approach', srv.GraspStage, '/master/plan_grasp_approach'),
    Function('open_grasp_gripper', srv.GraspStage, '/master/open_grasp_gripper'),
    Function('execute_grasp_approach', srv.GraspStage, '/master/execute_grasp_approach'),
    Function('close_grasp_gripper', srv.GraspStage, '/master/close_grasp_gripper'),
    Function('verify_grasp_contact', srv.GraspStage, '/master/verify_grasp_contact'),
    Function('prepare_grasp_input', srv.PrepareGraspInput, '/master/prepare_grasp_input'),
    Function('prepare_cloud_scene', srv.PrepareCloudScene, '/master/prepare_cloud_scene',
             'known_cloud'),
    Function('request_scan', Trigger, '/active_vision/trigger_scan', 'trigger_ack'),
    Function('get_3d_coordinate', srv.Get3DCoordinate, '/active_vision/get_3d_coordinate'),
    Function('infer_yolo', srv.GetYoloInference, '/active_vision/get_yolo_inference', 'rgb'),
    Function('get_nbv_candidates', srv.GetNBVPose, '/active_vision/get_nbv_candidates'),
    Function('get_obstacle_location', srv.GetObstacleLocation,
             '/active_vision/get_obstacle_location'),
    Function('update_object', srv.UpdateObjectObservation, '/master/update_object_observation'),
    Function('resolve_object', srv.ResolveObject, '/master/resolve_object'),
    Function('append_task', srv.AppendTask, '/master/append_task'),
    Function('execute_pipeline', srv.ExecutePipeline, '/master/execute_pipeline'),
    Function('flush_pipeline', srv.FlushPipeline, '/master/flush_pipeline'),

    Function('verify_presence', srv.VerifyPresence,
             '/vlm_reasoning_node/verify_presence', 'presence'),
    Function('ground_object', srv.GroundObject,
             '/vlm_reasoning_node/ground_object', 'rgb'),
    Function('infer_vision', srv.VisionInference,
             '/vlm_reasoning_node/infer_vision', 'rgb'),
    Function('segment_object', srv.SegmentGroundedImage,
             '/vlm_reasoning_node/segment_grounded_image', 'rgb'),
    Function('calculate_grasps', srv.FindGraspCandidates,
             '/vamp_planner/find_grasp_candidates', 'rgbd'),
    Function('calculate_cloud_grasps', srv.FindCloudGrasps,
             '/vamp_planner/find_cloud_grasps', 'known_cloud'),
    Function('find_placements', srv.FindPlacementCandidates,
             '/placement/find_placement_candidates'),
    Function('plan_placement', srv.PlanPlacement,
             '/vamp_planner/plan_placement'),
    Function('plan_layout', srv.PlanObjectLayout,
             '/placement/plan_object_layout'),
    Function('refine_grasp', srv.RefineGrasp,
             '/vamp_planner/refine_grasp', 'rgbd'),
    Function('plan_final_approach', srv.PlanCartesianGraspApproach,
             '/vamp_planner/plan_cartesian_grasp_approach'),
    Function('solve_pose', srv.LookAt, '/vamp_planner/look_at_ik'),
    Function('plan_motion', srv.CheckAndReplan,
             '/vamp_planner/check_and_replan'),
    Function('calculate_fallback_pose', srv.CalculateFallbackPose,
             '/vamp_planner/calculate_fallback_pose'),
    Function('plan_apf', srv.PlanAPF, '/vamp_planner/plan_apf'),
    Function('toggle_continuous_navigation', srv.ToggleContinuousNavigation,
             '/vamp_planner/toggle_continuous_navigation'),
    Function('look_at_target', srv.LookAt, '/master/look_at_target'),
    Function('execute_grasp', srv.ExecuteGrasp, '/master/execute_grasp'),
    Function('get_world_state', srv.GetWorldState, '/master/get_world_state'),
    Function('pin_placement_observation', srv.PinLayoutObservation,
             '/placement/pin_layout_observation'),
    Function('pin_grasp_observation', srv.PinLayoutObservation,
             '/vamp_planner/pin_layout_observation'),
)}


def endpoint(name):
    """Resolve a function without duplicating ROS topic names in callers."""
    return FUNCTIONS[name].endpoint


def catalog():
    """Describe transport and timing scope for local agents."""
    items = [{
        'name': f.name, 'endpoint': f.endpoint,
        'type': (('std_srvs' if f.service_type is Trigger else 'my_actions') +
                 '/srv/' + f.service_type.__name__),
        'backend': f.backend, 'input_mode': f.input_mode,
        'request_fields': f.service_type.Request.get_fields_and_field_types(),
        'response_fields': f.service_type.Response.get_fields_and_field_types(),
        'timing_scope': 'inclusive API wall time',
        'category': ('compatibility_facade' if f.name in {
            'plan_layout', 'execute_grasp', 'look_at_target'} else 'microstep'),
        'completion': ('queue_started' if f.name == 'execute_pipeline' else
                       'flush_requested' if f.name == 'flush_pipeline' else
                       'capture_armed' if f.name == 'request_scan' else 'operation_response'),
    } for f in FUNCTIONS.values()]
    items.append({
        'name': 'run_steps', 'endpoint': '/pipeline/run_steps',
        'type': 'my_actions/action/RunPipelineSteps',
        'completion': 'all queued steps attempted, failure or cancellation boundary',
        'timing_scope': 'action end-to-end wall time',
    })
    return items
