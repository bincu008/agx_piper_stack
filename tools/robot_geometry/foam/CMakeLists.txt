cmake_minimum_required(VERSION 3.1...3.21)
project(
  foam
  VERSION 0.1
  LANGUAGES C CXX
)

option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output." ON)
if(FORCE_COLORED_OUTPUT)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
  endif()
endif()

set(CMAKE_MODULE_PATH
  "${CMAKE_MODULE_PATH}"
  "${CMAKE_ROOT_DIR}/cmake/Modules"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
)

include(CompilerSettings)
include(HelperFunctions)

message(STATUS "Building ${CMAKE_BUILD_TYPE}")

add_subdirectory(external)

set(FOAM_PROGRAMS_DIR ${CMAKE_SOURCE_DIR}/foam/external)

get_all_targets(all_targets)
foreach(target ${all_targets})
  add_custom_target(${target}_fake ALL)
  add_dependencies(${target}_fake ${target})
  add_custom_command(TARGET ${target}_fake POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${target}> ${FOAM_PROGRAMS_DIR})
endforeach()
