# MILP C++ Interface

This directory exposes the Python MILP allocation module to a C++ main
program. It provides two separate JSON call modes:

- Single AOI: `milp_single_aoi_solve_json`
- Multi AOI: `milp_multi_aoi_run_json`

The C++ layer calls `milp/cpp_bridge.py`. That bridge imports only:

- `task_interface`
- `multi_aoi_interface`
- `execution_output_interface`

It does not import `frontend_app`, `visualization`, `streamlit`, or `plotly`.
Calling this interface executes only the core allocation path.

## Runtime Requirements

Install only the core Python dependencies for headless C++ use:

```bash
pip install numpy python-mip
```

The visualization dependencies in `milp/requirements.txt` are not needed for
this C++ interface unless you also run the Streamlit frontend.

## Build

From this directory:

```bash
cmake -S . -B build
cmake --build build --config Release
```

If CMake finds the wrong Python installation, pass the Python root explicitly:

```bash
cmake -S . -B build -DPython3_ROOT_DIR=/path/to/python
```

On Windows, use the Python environment that already has `numpy` and
`python-mip` installed.

## C++ Usage

Prefer passing an absolute path to the `milp` directory. The C layer accepts
UTF-8 paths and also falls back to the Windows local code page for narrow
strings, which helps when the project path contains Chinese characters:

```cpp
#include "MilpClient.hpp"

milp::SingleAoiMilpClient single("D:/path/to/milp");
std::string single_output = single.solveJson(single_input_json);
std::string single_output_from_file = single.solveFile("D:/path/to/input.json");

milp::MultiAoiMilpClient multi("D:/path/to/milp");
std::string multi_output = multi.runJson(multi_input_json);
std::string multi_output_from_file = multi.runFile("D:/path/to/multi_aoi_input.json");
```

`solveJson()` and `runJson()` expect JSON text content. `solveFile()` and
`runFile()` expect a JSON file path. The file content is read as UTF-8.

The single-AOI input follows `milp/templates/input_template.json`. The
single-AOI output follows `milp/templates/output_template.json`.

The multi-AOI input follows the dictionary accepted by
`MultiAOITaskAllocator.run()`. The output contains:

```json
{
  "status": "AOI_PLAN_READY",
  "aoi_route_state": {},
  "current_aoi_plan": {}
}
```

When `status` is `ALL_AOI_FINISHED`, `current_aoi_plan` is `null`.

## C ABI

If the main program cannot use C++ wrappers, include `milp_c_api.h` and call:

```cpp
char* error = nullptr;
MilpHandle handle = milp_create("D:/path/to/milp", "cbc", 3.0, 0, &error);

char* output = nullptr;
int status = milp_single_aoi_solve_json(handle, input_json, &output, &error);
int file_status = milp_single_aoi_solve_file(handle, input_path, &output, &error);
int multi_status = milp_multi_aoi_run_json(handle, multi_input_json, &output, &error);
int multi_file_status = milp_multi_aoi_run_file(handle, multi_input_path, &output, &error);

milp_free_string(output);
milp_free_string(error);
milp_destroy(handle);
```

Every string returned through `char**` must be released with
`milp_free_string`.

## Examples

After building, run from `build`:

```bash
./example_single_aoi ../..
./example_multi_aoi ../..
```

To call with JSON files instead of the built-in example payloads:

```bash
./example_single_aoi ../.. ../templates/input_template.json
./example_multi_aoi ../.. ../scenarios/multi_aoi_example.json
```

The argument must point to the `milp` directory. If omitted, the examples use
`../..`, which works when they are run from `milp/cpp_interface/build`. The
optional second argument is the input JSON file path.

cmakelist：


# 你的主程序
add_executable(main_program
    main.cpp
)

# 加入 MILP C++ 接口子目录
add_subdirectory(
    "D:/my_document/研究生/项目/54所直升机和无人机协同项目/all-code/54_20/milp/cpp_interface"
    "${CMAKE_BINARY_DIR}/milp_cpp_interface_build"
)

# 主程序链接 MILP 接口库
target_link_libraries(main_program
    PRIVATE
        milp_cpp_interface
)
