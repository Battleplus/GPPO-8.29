# SAR 搜索路径规划 — 调用接口说明

## 1. 入口函数

```python
from sar_search_planner import plan_search_mission
```

`plan_search_mission` 是唯一的对外入口，任务分配模块调用它即可。

---

## 2. 输入格式

传入一个 `list[dict]`，每个 dict 描述一个搜索任务：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `platform_id` | str | 是 | 平台名称，如 `"Blue_CH4_Recon"` |
| `center_km` | tuple[float,float] | 是 | 搜索区域中心坐标 (cx, cy)，单位 km |
| `width_km` | float | 是 | 搜索区域宽度，单位 km |
| `height_km` | float | 否 | 搜索区域高度，默认等于 width_km |
| `pattern` | str | 否 | 搜索模式，默认 `"racetrack"` |

`pattern` 可选值：

| 值 | 说明 |
|----|------|
| `"racetrack"` | 跑道形搜索 |
| `"sar_polygon"` | 多边形搜索 |
| `"sar_rounded"` | 圆角多边形搜索 |
| `"figure_eight"` | 8 字形搜索 |

### 示例 1：四机各搜一格

```python
assignments = [
    {"platform_id": "Blue_CH4_Recon",       "center_km": (47.5, -17.5), "width_km": 25, "pattern": "racetrack"},
    {"platform_id": "Blue_CH4_Recon_2",     "center_km": (72.5, -17.5), "width_km": 25, "pattern": "sar_polygon"},
    {"platform_id": "Blue_CH4_StrikeRecon",  "center_km": (47.5, -42.5), "width_km": 25, "pattern": "sar_rounded"},
    {"platform_id": "Blue_CH4_StrikeRecon_2","center_km": (72.5, -42.5), "width_km": 25, "pattern": "figure_eight"},
]

result = plan_search_mission(assignments)
```

### 示例 2：单机负责两区（任务重分配）

同一 `platform_id` 出现多次时，自动构建多区循环路径：

```python
assignments = [
    {"platform_id": "Blue_CH4_Recon", "center_km": (47.5, -17.5), "width_km": 25, "pattern": "racetrack"},
    {"platform_id": "Blue_CH4_Recon", "center_km": (72.5, -17.5), "width_km": 25, "pattern": "sar_polygon"},
    {"platform_id": "Blue_CH4_StrikeRecon", "center_km": (47.5, -42.5), "width_km": 25, "pattern": "sar_rounded"},
]

result = plan_search_mission(assignments)
# Blue_CH4_Recon 在两区之间循环，Blue_CH4_StrikeRecon 单区搜索
```

---

## 3. 返回值

`dict[str, SearchMissionPlan]`，key 为 `platform_id`。

### SearchMissionPlan 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `platform_id` | str | 平台名称 |
| `waypoints` | list[Waypoint] | 完整飞行路径 |
| `total_km` | float | 路径总长 (km) |
| `region_waypoints` | list[list[Waypoint]] | 每个区域的闭环路径，用于可视化 |
| `search_areas` | list[SearchArea] | 每个区域的搜索范围，用于画边界框 |

### Waypoint 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `x` | float | 场景坐标 X（东） |
| `y` | float | 场景坐标 Y（北） |
| `z` | float | 绝对高度（场景单位） |
| `terrain_z` | float | 该点地形高度 |
| `yaw_deg` | float | 航向角（0=北，顺时针） |

### 使用示例

```python
result = plan_search_mission(assignments)

for pid, plan in result.items():
    print(f"\n{pid}: {plan.total_km:.1f} km, {len(plan.waypoints)} 航点")

    # 飞行路径（可直接用于控制无人机循迹）
    for wp in plan.waypoints[:3]:
        print(f"  ({wp.x:.0f}, {wp.y:.0f}, z={wp.z:.0f}) yaw={wp.yaw_deg:.0f}°")

    # 每个区域的可视化数据
    for i, (rwp, sa) in enumerate(zip(plan.region_waypoints, plan.search_areas)):
        print(f"  区域{i}: {len(rwp)} 航点, "
              f"范围({sa.x_min:.0f},{sa.y_min:.0f})~({sa.x_max:.0f},{sa.y_max:.0f})")
```

---

## 4. 内部流程

```
任务分配 → plan_search_mission()
              │
              ├─ 按 platform_id 分组
              │
              ├─ 单区平台：plan(PlannerConfig) → 直接输出
              │
              └─ 多区平台：每区 plan() → build_multi_region_cycle()
                   │
                   ├─ 出口：离下一区中心最近的航点
                   ├─ 入口：离上一区出口最近的航点
                   ├─ 保证入口→出口 ≥ loop 的 70%
                   └─ A* 避障过渡
```

`build_multi_region_cycle` 的过渡策略：
- **出口**：当前区域路径上离下一区中心最近的航点
- **入口**：下一区路径上离上一区出口最近的航点
- **覆盖保证**：入口→出口至少覆盖整圈的 70%，不足则推到出口前 70% 处
- **过渡路径**：A* 避障，遇山绕行

---

## 5. 场景坐标

- 地图中心 = (0, 0)，范围 300 km × 300 km
- km → 场景单位：`scene_units = km * 10`（100 m/unit）
- 坐标系：X=东，Y=北，Z=高度

---

## 6. 路径形状参数

以下参数固定在 `PlannerConfig` 中，无需每次传入：

| 参数 | 默认值 | 用于 |
|------|--------|------|
| `racetrack_length_km` | 18 | 跑道形 |
| `racetrack_width_km` | 14 | 跑道形 |
| `sar_radius_km` | 10 | 多边形 / 圆角多边形 |
| `sar_sides` | 6 | 多边形 / 圆角多边形 |
| `sar_turn_radius_km` | 5 | 圆角多边形 |
| `eight_radius_km` | 6 | 8 字形 |
| `eight_line_km` | 18 | 8 字形 |
| `angle_deg` | 30 | 路径旋转角 |
| `altitude_agl_m` | 5000 | 飞行高度 |
| `clockwise` | True | 飞行方向 |

如需自定义，在 `sar_search_planner/config.py` 中修改。
