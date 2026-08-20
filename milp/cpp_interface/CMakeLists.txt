cmake_minimum_required(VERSION 3.16)

project(milp_cpp_interface LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_library(milp_cpp_interface milp_c_api.cpp)
target_compile_features(milp_cpp_interface PUBLIC cxx_std_17)
target_include_directories(
    milp_cpp_interface
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(milp_cpp_interface PUBLIC Python3::Python)

get_target_property(MILP_CPP_INTERFACE_TYPE milp_cpp_interface TYPE)
if(MILP_CPP_INTERFACE_TYPE STREQUAL "STATIC_LIBRARY")
    target_compile_definitions(milp_cpp_interface PUBLIC MILP_STATIC)
else()
    target_compile_definitions(milp_cpp_interface PRIVATE MILP_CPP_INTERFACE_BUILD)
endif()

if(MSVC)
    target_compile_options(milp_cpp_interface PRIVATE /W4)
else()
    target_compile_options(milp_cpp_interface PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(example_single_aoi example_single_aoi.cpp)
target_link_libraries(example_single_aoi PRIVATE milp_cpp_interface)

add_executable(example_multi_aoi example_multi_aoi.cpp)
target_link_libraries(example_multi_aoi PRIVATE milp_cpp_interface)
