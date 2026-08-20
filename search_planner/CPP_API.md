# SAR 搜索路径规划 C++ 调用接口

## 目标

C++ 总函数只需要调用一个搜索路径规划函数，输出一个路径点 list：

```cpp
std::vector<sar_search_planner::Waypoint> path =
    sar_search_planner::PlanSarSearchPath(tasks);
```

底层仍复用现有 Python 规划器，C++ 侧通过 `search_planner/cpp_plan_bridge.py` 子进程调用规划器并读取路径点。

## C++ 头文件

```cpp
#include "cpp_interface/SarSearchPlannerClient.hpp"
```

## 输入

```cpp
struct SearchTask {
    std::string platform_id;
    double center_x_km;
    double center_y_km;
    double width_km;
    double height_km;
    std::string pattern;
    double altitude_agl_m;
};
```

同一个 `platform_id` 出现多次时，规划器会自动把多个区域串成一条循环路径。

## 输出

```cpp
struct Waypoint {
    std::string platform_id;
    int point_index;
    double x;
    double y;
    double z;
    double terrain_z;
    double yaw_deg;
    double total_km;
};
```

返回值就是：

```cpp
std::vector<Waypoint>
```

也就是路径点 list。多无人机任务会把所有平台的路径点放在同一个 list 里，可通过 `platform_id` 区分。

## C++ 调用示例

```cpp
#include "cpp_interface/SarSearchPlannerClient.hpp"

#include <iostream>
#include <vector>

int main() {
    using namespace sar_search_planner;

    std::vector<SearchTask> tasks = {
        {"Blue_CH4_Recon", 37.5, 42.5, 25.0, 25.0, "racetrack", 5000.0},
        {"Blue_CH4_Recon_2", 62.5, 42.5, 25.0, 25.0, "racetrack", 5000.0},
        {"Blue_CH4_StrikeRecon", 37.5, 17.5, 25.0, 25.0, "figure_eight", 5000.0},
        {"Blue_CH4_StrikeRecon", 62.5, 17.5, 25.0, 25.0, "figure_eight", 5000.0},
    };

    std::vector<Waypoint> path = PlanSarSearchPath(tasks);

    for (const Waypoint& wp : path) {
        std::cout << wp.platform_id
                  << " #" << wp.point_index
                  << " x=" << wp.x
                  << " y=" << wp.y
                  << " z=" << wp.z
                  << " yaw=" << wp.yaw_deg
                  << "\n";
    }

    return 0;
}
```

## 编译示例

在项目根目录执行：

```bash
g++ -std=c++17 ^
  search_planner/cpp_interface/example_main.cpp ^
  search_planner/cpp_interface/SarSearchPlannerClient.cpp ^
  -o search_planner/cpp_interface/example_main.exe
```

Linux 写法：

```bash
g++ -std=c++17 \
  search_planner/cpp_interface/example_main.cpp \
  search_planner/cpp_interface/SarSearchPlannerClient.cpp \
  -o search_planner/cpp_interface/example_main
```

## 直接调用桥接脚本

如果不走 C++ 封装，也可以直接生成顶层路径点数组：

```bash
python search_planner/cpp_plan_bridge.py \
  --input request.json \
  --output waypoints.json \
  --format list_json
```

`waypoints.json` 格式：

```json
[
  {
    "platform_id": "UAV_1",
    "point_index": 0,
    "x": 30.62,
    "y": -86.96,
    "z": 119.15,
    "terrain_z": 69.15,
    "yaw_deg": 30.0,
    "total_km": 64.66
  }
]
```
