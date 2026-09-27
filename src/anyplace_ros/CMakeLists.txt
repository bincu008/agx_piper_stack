cmake_minimum_required(VERSION 3.8)
project(anyplace_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
install(DIRECTORY anyplace_ros/
  DESTINATION "${PYTHON_INSTALL_DIR}/anyplace_ros"
  PATTERN "__pycache__" EXCLUDE
)

install(PROGRAMS
  scripts/anyplace_node
  scripts/download_anyplace_checkpoint
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES README.md ANYPLACE_NOTICE.md LICENSE
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_model test/test_model.py)
  ament_add_pytest_test(test_inference test/test_inference.py)
endif()

ament_package()
