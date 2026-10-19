cmake_minimum_required(VERSION 3.8)
project(vlm_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(my_actions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

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

add_library(vlm_reasoning_component SHARED
  src/vlm_reasoning_node.cpp
  src/edge_sam_onnx.cpp)
target_include_directories(vlm_reasoning_component PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(vlm_reasoning_component 
  ament_index_cpp
  rclcpp 
  rclcpp_components
  rclcpp_action 
  sensor_msgs 
  std_msgs 
  std_srvs
  my_actions 
  cv_bridge 
  image_transport 
  OpenCV
)
target_link_libraries(vlm_reasoning_component ${ONNXRUNTIME_LIBRARY})

rclcpp_components_register_node(
  vlm_reasoning_component
  PLUGIN "VLMReasoningNode"
  EXECUTABLE vlm_reasoning_node
  EXECUTOR MultiThreadedExecutor)

install(TARGETS
  vlm_reasoning_component
  vlm_reasoning_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models)
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_layout_prompt test/test_layout_prompt.cpp)
  target_include_directories(test_layout_prompt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_features(test_layout_prompt PRIVATE cxx_std_17)
  ament_add_gtest(test_inference_capture test/test_inference_capture.cpp)
  target_include_directories(test_inference_capture PRIVATE include)
  ament_add_gtest(test_grounding test/test_grounding.cpp)
  target_include_directories(test_grounding PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_features(test_grounding PRIVATE cxx_std_17)
endif()

ament_package()
