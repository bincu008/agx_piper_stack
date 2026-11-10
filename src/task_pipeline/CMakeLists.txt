cmake_minimum_required(VERSION 3.8)
project(task_pipeline)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/pipeline_node scripts/pipeline_task scripts/mock_bowl_cup DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/pipeline_mcp DESTINATION lib/${PROJECT_NAME})
install(FILES ../../docs/mcp/WORKFLOW.md ../../docs/mcp/NANOBOT.md
  ../../docs/mcp/PREPARATION_ONLY.md
  ../../docs/mcp/LIVE_PERCEPTION.md
  ../../docs/mcp/SPATIAL_MAP.md
  ../../docs/mcp/AGENT_PIPELINE.md
  ../../docs/mcp/TASK_INTERFACE.md
  ../../docs/mcp/TASK_TOOLS.json
  ../../docs/mcp/CONTROL_TOOLS.json
  ../../docs/mcp/CONTROL_TOOLS.md
  ../../docs/mcp/WORKSPACE_TOOLS.md
  ../../docs/mcp/WORKSPACE_TOOLS.json
  ../../docs/mcp/WORKSPACE_PROFILE_TOOLS.json
  ../../docs/mcp/TOOL_DEPENDENCIES.md
  ../../docs/mcp/requirements.txt ../../docs/mcp/MOCK_VALIDATION.md
  DESTINATION share/${PROJECT_NAME}/mcp)
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_pipeline test/test_pipeline.py)
  ament_add_pytest_test(test_mock_bowl_cup test/test_mock_bowl_cup.py)
  ament_add_pytest_test(test_context test/test_context.py)
  ament_add_pytest_test(test_recipe_obstacles test/test_recipe_obstacles.py)
  ament_add_pytest_test(test_mcp test/test_mcp.py)
  ament_add_pytest_test(test_mcp_dependencies test/test_mcp_dependencies.py)
  ament_add_pytest_test(test_mcp_execution test/test_mcp_execution.py)
  ament_add_pytest_test(test_mcp_tasks test/test_mcp_tasks.py)
  ament_add_pytest_test(test_mcp_controls test/test_mcp_controls.py)
  ament_add_pytest_test(test_mcp_workspace test/test_mcp_workspace.py)
  ament_add_pytest_test(test_mcp_reconcile test/test_mcp_reconcile.py)
  ament_add_pytest_test(test_mcp_preparation test/test_mcp_preparation.py)
endif()
ament_package()
