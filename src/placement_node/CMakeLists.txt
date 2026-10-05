cmake_minimum_required(VERSION 3.8)
project(placement_node)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
install(DIRECTORY placement_node/
  DESTINATION "${PYTHON_INSTALL_DIR}/placement_node"
  PATTERN "__pycache__" EXCLUDE
)

install(PROGRAMS
  scripts/placement_node
  scripts/box_benchmark
  scripts/benchmark_grounding
  scripts/download_anyplace_checkpoint
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

install(FILES README.md ANYPLACE_NOTICE.md LICENSE
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_box_benchmark test/test_box_benchmark.py)
  ament_add_pytest_test(test_candidate_selection test/test_candidate_selection.py)
  ament_add_pytest_test(test_model test/test_model.py)
  ament_add_pytest_test(test_inference test/test_inference.py)
  ament_add_pytest_test(test_robobrain_api test/test_robobrain_api.py)
  ament_add_pytest_test(test_layout_debug test/test_layout_debug.py)
endif()

ament_package()
