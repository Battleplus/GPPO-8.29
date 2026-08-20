cmake_minimum_required(VERSION 3.18)
project(ql_path_planner_cpp LANGUAGES CXX)

# The planner source uses Python 3.10 syntax (PEP 604 union types).
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)

add_library(ql_path_planner SHARED src/ql_path_planner.cpp)
target_compile_features(ql_path_planner PUBLIC cxx_std_17)
target_include_directories(ql_path_planner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ql_path_planner PRIVATE Python3::Python)
target_compile_definitions(ql_path_planner PRIVATE QL_PATH_PLANNER_BUILD)

add_executable(ql_path_planner_example example/main.cpp)
target_link_libraries(ql_path_planner_example PRIVATE ql_path_planner)

enable_testing()
add_test(
    NAME ql_path_planner_smoke
    COMMAND ql_path_planner_example ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)
