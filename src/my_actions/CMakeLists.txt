cmake_minimum_required(VERSION 3.8)
project(my_actions)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GraspCandidate.msg"
  "msg/PlacementCandidate.msg"
  "msg/PipelineTiming.msg"
  "msg/StepBinding.msg"
  "msg/PipelineStep.msg"
  "msg/PipelineStepResult.msg"
  "srv/RunPipelineStep.srv"
  "srv/SubmitPipelineSteps.srv"
  "srv/GetStepContext.srv"
  "srv/ClearStepContext.srv"
  "srv/StorePipelineResult.srv"
  "action/RunPipelineSteps.action"
  "srv/MoveTo.srv"
  "srv/ExecuteTrajectory.srv"
  "srv/GripperCommand.srv"
  "srv/SearchPlacements.srv"
  "srv/SelectGrasp.srv"
  "srv/CacheGraspPlan.srv"
  "srv/GetCachedGrasp.srv"
  "srv/GetArmState.srv"
  "srv/GraspStage.srv"
  "srv/PrepareGraspInput.srv"
  "srv/CacheGraspRefinement.srv"
  "srv/VisionInference.srv"
  "srv/GroundObject.srv"
  "srv/VerifyPresence.srv"
  "srv/CaptureObservation.srv"
  "srv/GetPipelineFunctions.srv"
  "srv/GetPipelineMetrics.srv"
  "srv/ToggleContinuousNavigation.srv"
  "srv/CheckAndReplan.srv"
  "srv/GetObstacleLocation.srv"
  "srv/Get3DCoordinate.srv"
  "srv/Chat.srv"
  "srv/SegmentGroundedImage.srv"
  "srv/CalculateFallbackPose.srv"
  "srv/PlanAPF.srv"
  "srv/LookAt.srv"
  "srv/GetNBVPose.srv"
  "srv/MoveRelative.srv"
  "srv/AppendTask.srv"
  "srv/ExecutePipeline.srv"
  "srv/FlushPipeline.srv"
  "srv/GetYoloInference.srv"
  "srv/FindGraspCandidates.srv"
  "srv/FindCloudGrasps.srv"
  "srv/FindPlacementCandidates.srv"
  "srv/PlanPlacement.srv"
  "srv/PlanObjectLayout.srv"
  "srv/PinLayoutObservation.srv"
  "srv/UpdateObjectObservation.srv"
  "srv/ResolveObject.srv"
  "srv/ExecuteGrasp.srv"
  "srv/RefineGrasp.srv"
  "srv/PlanCartesianGraspApproach.srv"
  "srv/GetWorldState.srv"
  DEPENDENCIES sensor_msgs geometry_msgs std_msgs action_msgs
)

ament_package()
