# C++ wrapper for the QL path planner

This shared library embeds the current Python MPPI implementation. C++ sends
a UTF-8 JSON request and receives the same JSON result produced by
`FormationPlanResult.to_json()`. Python and NumPy types do not cross the ABI.

## Build on Windows

Use Python 3.10+ containing NumPy and point CMake at that same Python. On
Windows, use a compiler compatible with that Python distribution (normally
MSVC for python.org/Conda builds):

```powershell
cmake -S ql/scripts/cpp -B ql/scripts/cpp/build -DPython3_EXECUTABLE=(Get-Command python).Source
cmake --build ql/scripts/cpp/build --config Release
```

Run the example from the repository root:

```powershell
ql/scripts/cpp/build/Release/ql_path_planner_example.exe (Get-Location).Path
```

The executable must be able to find the Python DLL selected by CMake. Keep it
on `PATH`, or use the same developer shell used to configure the build.

## C++ API

```cpp
#include "ql_path_planner.h"

ql::PathPlanner planner("D:/code/xiangmu");
std::string result = planner.plan(R"({
  "team_count": 4,
  "start": [-800, -600, 80],
  "goal": [800, 600, 80],
  "formation": "v_shape",
  "spacing": 40,
  "verbose": false
})");
```

The library also exports a plain C ABI: `ql_planner_initialize`,
`ql_plan_json`, `ql_planner_last_error`, `ql_planner_free`, and
`ql_planner_shutdown`. Strings returned by `ql_plan_json` must be released
with `ql_planner_free`.

Optional request keys include `depth_spacing`, `cruise_altitude`,
`member_assignments`, `map_size_units`, `meters_per_unit`,
`terrain_vertical_exaggeration`, `planner_config`, and `obstacles`. Fields in
`planner_config` map directly to `MPPIConfig`.
