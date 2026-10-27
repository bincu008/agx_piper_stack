cmake_minimum_required(VERSION 3.8)
project(vision_node)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(my_actions REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(octomap REQUIRED)

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

add_library(active_vision_server_component SHARED src/active_vision_server.cpp)
target_compile_definitions(active_vision_server_component PRIVATE "COMPONENT_BUILD")
target_include_directories(active_vision_server_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR})
target_compile_features(active_vision_server_component PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  active_vision_server_component
  "rclcpp"
  "rclcpp_components"
  "sensor_msgs"
  "std_srvs"
  "tf2"
  "tf2_ros"
  "tf2_sensor_msgs"
  "pcl_conversions"
  "pcl_ros"
  "my_actions"
  "cv_bridge"
  "ament_index_cpp"
)
target_link_libraries(active_vision_server_component yaml-cpp ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES} ${ONNXRUNTIME_LIBRARY})
rclcpp_components_register_node(active_vision_server_component PLUGIN "vision_node::ActiveVisionServer" EXECUTABLE active_vision_server)

install(TARGETS active_vision_server_component active_vision_server
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

# Add rpath so it finds the library at runtime (relative to executable)
target_include_directories(active_vision_server_component PRIVATE include)
if(BUILD_TESTING)
  enable_testing()
  add_executable(target_visibility_test test/test_target_visibility.cpp)
  target_include_directories(target_visibility_test PRIVATE include ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS})
  target_link_libraries(target_visibility_test ${OCTOMAP_LIBRARIES})
  target_compile_options(target_visibility_test PRIVATE -UNDEBUG)
  add_test(NAME target_visibility_test COMMAND target_visibility_test)
  add_executable(nbv_preference_test test/test_nbv_preference.cpp)
  target_include_directories(nbv_preference_test PRIVATE include ${PCL_INCLUDE_DIRS})
  add_test(NAME nbv_preference_test COMMAND nbv_preference_test)
  add_executable(voxel_neighbors_test test/test_voxel_neighbors.cpp)
  target_include_directories(voxel_neighbors_test PRIVATE include)
  add_test(NAME voxel_neighbors_test COMMAND voxel_neighbors_test)
  add_executable(perception_translation_test test/test_perception_translation.cpp)
  target_include_directories(perception_translation_test PRIVATE include ${PCL_INCLUDE_DIRS})
  add_test(NAME perception_translation_test COMMAND perception_translation_test)
  add_executable(metric_scale_test test/test_metric_scale.cpp)
  target_include_directories(metric_scale_test PRIVATE include)
  ament_target_dependencies(metric_scale_test sensor_msgs)
  add_test(NAME metric_scale_test COMMAND metric_scale_test)
endif()
ament_package()
