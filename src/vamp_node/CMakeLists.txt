cmake_minimum_required(VERSION 3.8)
project(vamp_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(my_actions REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(gpd_vendor REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vamp REQUIRED)

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS $ENV{ONNXRUNTIME_ROOT}
  PATH_SUFFIXES include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS $ENV{ONNXRUNTIME_ROOT}
  PATH_SUFFIXES lib lib64)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime was not found. Install it or set ONNXRUNTIME_ROOT to its prefix.")
endif()

# Dedicated VAMP-MVT Motion Planner Node
add_library(analytical_grasp_generator STATIC src/analytical_grasp_generator.cpp)
set_target_properties(analytical_grasp_generator PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(analytical_grasp_generator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(analytical_grasp_generator Eigen3::Eigen)
target_compile_features(analytical_grasp_generator PUBLIC cxx_std_17)

add_library(vamp_planner_server_component SHARED
  src/vamp_planner_server.cpp
  src/local_grasp_refiner.cpp
  src/gpd_grasp_adapter.cpp
  src/hggd_mcu_adapter.cpp)
target_compile_definitions(vamp_planner_server_component PRIVATE "COMPONENT_BUILD")
target_include_directories(vamp_planner_server_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR})
target_compile_options(vamp_planner_server_component PRIVATE ${PCL_DEFINITIONS})
target_compile_features(vamp_planner_server_component PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  vamp_planner_server_component
  "rclcpp"
  "rclcpp_components"
  "sensor_msgs"
  "std_msgs"
  "std_srvs"
  "my_actions"
  "geometry_msgs"
  "visualization_msgs"
  "ament_index_cpp"
  "gpd_vendor"
)
target_link_libraries(vamp_planner_server_component
  analytical_grasp_generator
  Eigen3::Eigen
  vamp::vamp
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIBRARY}
)
rclcpp_components_register_node(vamp_planner_server_component PLUGIN "vamp_node::VampPlannerServer" EXECUTABLE vamp_planner_server)
set_target_properties(vamp_planner_server_component PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "\$ORIGIN/vamp_node")

install(TARGETS
  vamp_planner_server_component vamp_planner_server
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch rviz models
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_payload_checked_robot test/test_payload_checked_robot.cpp)
  target_include_directories(test_payload_checked_robot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_payload_checked_robot vamp::vamp)
  ament_add_gtest(test_pipeline_timing test/test_pipeline_timing.cpp)
  target_include_directories(test_pipeline_timing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_add_gtest(test_placement_geometry test/test_placement_geometry.cpp)
  target_include_directories(test_placement_geometry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_placement_geometry Eigen3::Eigen)
  ament_add_gtest(test_placement_ranking test/test_placement_ranking.cpp)
  target_include_directories(test_placement_ranking PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_add_gtest(test_analytical_grasp_generator
    test/test_analytical_grasp_generator.cpp)
  target_link_libraries(test_analytical_grasp_generator
    analytical_grasp_generator Eigen3::Eigen)
  target_include_directories(test_analytical_grasp_generator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_add_gtest(test_hggd_mcu_adapter test/test_hggd_mcu_adapter.cpp)
  target_link_libraries(test_hggd_mcu_adapter vamp_planner_server_component)
  ament_target_dependencies(test_hggd_mcu_adapter
    ament_index_cpp sensor_msgs)
  target_include_directories(test_hggd_mcu_adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
  ament_add_gtest(test_local_grasp_refiner test/test_local_grasp_refiner.cpp)
  target_sources(test_local_grasp_refiner PRIVATE src/local_grasp_refiner.cpp)
  target_link_libraries(test_local_grasp_refiner Eigen3::Eigen)
  target_include_directories(test_local_grasp_refiner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_add_gtest(test_cartesian_grasp_approach
    test/test_cartesian_grasp_approach.cpp)
  target_link_libraries(test_cartesian_grasp_approach Eigen3::Eigen)
  target_include_directories(test_cartesian_grasp_approach PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

ament_package()
