cmake_minimum_required(VERSION 3.16)

project(brain_cpp LANGUAGES CXX)

add_subdirectory(../milp/cpp_interface ${CMAKE_CURRENT_BINARY_DIR}/milp_cpp_interface)
add_subdirectory(../perch/cpp_interface ${CMAKE_CURRENT_BINARY_DIR}/perch_cpp_interface)
add_subdirectory(../mppi/cpp ${CMAKE_CURRENT_BINARY_DIR}/mppi_cpp_interface)
add_subdirectory(../drl_env/cpp_interface ${CMAKE_CURRENT_BINARY_DIR}/ppo_avoidance_cpp_interface)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(brain_cpp
    src/mission_context.cpp
    src/mission_event.cpp
    src/mission_state.cpp
    src/target_tracking_fsm.cpp
    src/scenario_initializer.cpp
    src/mission_fsm.cpp
    src/mission_brain.cpp
    src/external_adapters.cpp
    src/mppi_planner_client.cpp
    src/mppi_route_planner.cpp
    src/isaac_python_environment.cpp
    src/parallel_platform_controller.cpp
    ../search_planner/cpp_interface/SarSearchPlannerClient.cpp
)

target_include_directories(brain_cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../search_planner/cpp_interface
)

target_compile_options(brain_cpp
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(brain_cpp_demo app/main.cpp)
target_link_libraries(brain_cpp_demo PRIVATE brain_cpp)
target_link_libraries(brain_cpp PUBLIC milp_cpp_interface)
target_link_libraries(brain_cpp PUBLIC perch_cpp_interface)
target_link_libraries(brain_cpp PUBLIC ql_path_planner)
target_link_libraries(brain_cpp PUBLIC ppo_avoidance)

add_executable(brain_cpp_smoke_test tests/smoke_test.cpp)
target_link_libraries(brain_cpp_smoke_test PRIVATE brain_cpp)

add_executable(brain_cpp_perch_bridge_test tests/perch_bridge_test.cpp)
target_link_libraries(brain_cpp_perch_bridge_test PRIVATE brain_cpp)

add_executable(brain_cpp_mppi_route_test tests/mppi_route_test.cpp)
target_link_libraries(brain_cpp_mppi_route_test PRIVATE brain_cpp)

add_executable(brain_cpp_patrol_export tools/patrol_plan_export.cpp)
target_link_libraries(brain_cpp_patrol_export PRIVATE brain_cpp)

add_executable(brain_cpp_six_uav_mission_export tools/six_uav_mission_export.cpp)
target_link_libraries(brain_cpp_six_uav_mission_export PRIVATE brain_cpp)

enable_testing()
add_test(
    NAME brain_cpp_smoke_test
    COMMAND brain_cpp_smoke_test
            ${CMAKE_CURRENT_SOURCE_DIR}/../drl_env/cpp_interface/models/ppo_v9.bin
)
add_test(NAME brain_cpp_perch_bridge_test COMMAND brain_cpp_perch_bridge_test)
add_test(NAME brain_cpp_mppi_route_test COMMAND brain_cpp_mppi_route_test)
set_tests_properties(
    brain_cpp_perch_bridge_test brain_cpp_mppi_route_test
    PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    set_property(
        TEST ql_path_planner_smoke
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../mppi/cpp
        PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/mppi_compat"
    )
endif()
