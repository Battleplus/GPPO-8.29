add_library(perch_cpp_interface
    PerchPositionSelectorClient.cpp
)

target_include_directories(perch_cpp_interface
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(perch_cpp_interface PUBLIC cxx_std_17)

target_compile_options(perch_cpp_interface
    PRIVATE -Wall -Wextra -Wpedantic
)
