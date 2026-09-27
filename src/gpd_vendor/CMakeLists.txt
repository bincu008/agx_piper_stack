cmake_minimum_required(VERSION 3.16)
project(gpd_vendor)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
include(ExternalProject)

set(GPD_REVISION "6327f20eabfcba41a05fdd2e2ba408153dc2e958")
set(GPD_ARCHIVE "" CACHE FILEPATH "Optional local GPD source archive (offline builds)")
set(GPD_STAGE "${CMAKE_CURRENT_BINARY_DIR}/gpd_stage")
set(GPD_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/gpd_src")
set(GPD_DOWNLOAD_ARGS
  URL "https://codeload.github.com/atenpas/gpd/tar.gz/${GPD_REVISION}"
  TLS_VERIFY TRUE
  DOWNLOAD_NO_PROGRESS TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
if(GPD_ARCHIVE)
  set(GPD_DOWNLOAD_ARGS
    URL "${GPD_ARCHIVE}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
endif()

ExternalProject_Add(gpd_external
  ${GPD_DOWNLOAD_ARGS}
  SOURCE_DIR "${GPD_SOURCE}"
  PATCH_COMMAND
    ${CMAKE_COMMAND} -DGPD_SOURCE_DIR=<SOURCE_DIR>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_gpd.cmake
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${GPD_STAGE}
    -DUSE_OPENVINO=OFF
    -DUSE_CAFFE=OFF
    -DUSE_OPENCV=OFF
    -DUSE_PCL_GPU=OFF
    -DBUILD_DATA_GENERATION=OFF
  BUILD_BYPRODUCTS "${GPD_STAGE}/lib/libgpd.so")

add_custom_target(gpd_vendor_build ALL DEPENDS gpd_external)

set(GPD_VENDOR_SHARE "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}")
configure_file(config/piper_eigen_params.cfg.in piper_eigen_params.cfg @ONLY)

install(DIRECTORY "${GPD_STAGE}/include/" DESTINATION include)
install(DIRECTORY "${GPD_STAGE}/lib/" DESTINATION lib)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/piper_eigen_params.cfg"
  config/piper_hand_geometry.cfg
  config/image_geometry_15channels.cfg
  DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY "${GPD_SOURCE}/models/lenet/15channels/params"
  DESTINATION share/${PROJECT_NAME}/models/lenet/15channels)
install(FILES "${GPD_SOURCE}/LICENSE.md" DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(gpd)
ament_export_dependencies(Eigen3 OpenCV PCL)
ament_package()
