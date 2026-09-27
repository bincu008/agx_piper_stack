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

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GraspCandidate.msg"
  "msg/PlacementCandidate.msg"
  "srv/ToggleContinuousNavigation.srv"
  "srv/CheckAndReplan.srv"
  "srv/GetObstacleLocation.srv"
  "srv/Get3DCoordinate.srv"
  "srv/Chat.srv"
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
  "srv/FindPlacementCandidates.srv"
  "srv/PlanPlacement.srv"
  "srv/UpdateObjectObservation.srv"
  "srv/ResolveObject.srv"
  "srv/ExecuteGrasp.srv"
  "srv/RefineGrasp.srv"
  "srv/PlanCartesianGraspApproach.srv"
  "srv/GetWorldState.srv"
  DEPENDENCIES sensor_msgs geometry_msgs std_msgs
)

ament_package()
