# 任务分配模块 —— 大脑接入文档

## 1. 模块位置

将 `代码-v2/` 目录放置到大脑项目的合适位置后，确保以下路径在 Python path 中：

```python
import sys
sys.path.insert(0, "<项目根>/代码-v2")
```

依赖仅需 `numpy` + `python-mip`：

```bash
pip install numpy python-mip
```

## 2. 模块结构

```
代码-v2/
├── task_interface.py           ← 单 AOI 入口（大脑最常用）
├── multi_aoi_interface.py      ← 多 AOI 入口（含排序）
├── config/settings.py          ← GlobalSettings 配置
├── core/snapshot.py            ← 输入数据结构定义
├── core/allocation.py          ← 输出数据结构定义
├── allocation/                 ← MILP 求解器内部（大脑无需感知）
├── aoi/                        ← AOI 排序内部（大脑无需感知）
├── utils/                      ← 工具函数
├── scenarios/                  ← JSON 场景模板（参考用）
└── tests/                      ← 测试
```

大脑只需关心 **两个入口文件** 和 **四个数据结构文件**。

## 3. 快速开始

### 3.1 单 AOI 模式（最简调用）

```python
from task_interface import TaskAllocator, make_snapshot, make_grid, make_target, make_platform

# 1) 构造栅格（每个 AOI 固定 5 个栅格: c0 巡逻区 + c1~c4 子区）
grids = [
    make_grid("c0", center_xy=(175, 125), width_km=50, height_km=50, weather_w=0.2),
    make_grid("c1", center_xy=(162, 112), weather_w=0.15),
    make_grid("c2", center_xy=(188, 112), weather_w=0.40),
    make_grid("c3", center_xy=(162, 138), weather_w=0.55),
    make_grid("c4", center_xy=(188, 138), weather_w=0.72),
]

# 2) 构造目标（仅 alive=True 且 confirmed=True 的目标参与 MILP 优化）
targets = [
    make_target("g1", "RADAR", pos_xy=(162, 112), confirmed=True,  value=0.97, threat=0.91),
    make_target("g2", "CP",    pos_xy=(188, 112), confirmed=True,  value=0.85, threat=0.60),
    make_target("g3", "AV",    pos_xy=(162, 138), confirmed=False, value=0.70, threat=0.50),  # 未确认，不参与
]

# 3) 构造平台
platforms = [
    make_platform("U1", "UAV",  pos_xy=(150, -50)),   # 传感器固定为 EO/SAR/ESM，无弹药
    make_platform("U2", "UAV",  pos_xy=(150, -50)),
    make_platform("U3", "UAV",  pos_xy=(150, -50)),
    make_platform("H1", "HELI", pos_xy=(150, -50)),   # 传感器固定为 MMW/EOIR，含弹药
]

# 4) 组装快照
snap = make_snapshot(
    cycle_id=0,
    grids=grids,
    targets=targets,
    platforms=platforms,
    staging_position=(150.0, -50.0),   # 集结区坐标 (km)
    commander_AOI=["A_3_4"],            # 当前 AOI 标识
)

# 5) 求解（verbose=0 静默，=1 终端输出）
allocator = TaskAllocator(solver="cbc", time_limit_s=3.0, verbose=0)
plan = allocator.solve(snap)

# 6) 读取结果
print(f"状态: {plan.status}")            # "OPTIMAL" / "FEASIBLE" / "TIME_LIMIT"
print(f"目标值: {plan.objective:.2f}")    # 有打击时约 800~1000，纯侦察时为负值
print(f"求解耗时: {plan.solve_time_ms:.0f} ms")

for ra in plan.recon_assignments:
    print(f"侦察: {ra.pid} → {ra.sensor} → {ra.cell}  role={ra.role}")

for sa in plan.strike_assignments:
    print(f"打击: {sa.pid} → {sa.target} {sa.munition}×{sa.qty}  role={sa.role}")
```

### 3.2 多 AOI 模式（含排序）

```python
from multi_aoi_interface import MultiAOITaskAllocator

allocator = MultiAOITaskAllocator(solver="cbc", time_limit_s=3.0, verbose=0)

# 首次调用
input_data = {
    "aois": [
        {"id": "A_5_6", "row": 5, "col": 6, "priority": 0.8,
         "target_prior": 0.70, "target_value": 0.97, "target_threat": 0.91},
        {"id": "A_3_4", "row": 3, "col": 4, "priority": 0.6,
         "target_prior": 0.55, "target_value": 0.78, "target_threat": 0.55},
        {"id": "A_1_5", "row": 1, "col": 5, "priority": 0.9,
         "target_prior": 0.40, "target_value": 0.68, "target_threat": 0.48},
    ],
    "platforms": {
        "UAV":  {"count": 5, "pos": [150, -50],
                 "sensors": ["EO", "SAR", "ESM"],
                 "munitions": {"HF": 0, "RKT": 0, "GUN": 0}},
        "HELI": {"count": 2, "pos": [150, -50],
                 "sensors": ["MMW", "EOIR"],
                 "munitions": {"HF": 16, "RKT": 76, "GUN": 1200}},
    },
    "targets": [
        {"tid": "g1", "type": "RADAR", "pos": [265, 238], "value": 0.97, "threat": 0.91,
         "confirmed": True, "alive": True},
        {"tid": "g2", "type": "CP",    "pos": [175, 125], "value": 0.85, "threat": 0.60,
         "confirmed": True, "alive": True},
    ],
    "staging_position": [150, -50],
    "aoi_route_state": None,        # 首次传 null
    "execution_feedback": None,     # 首次传 null
}

# 逐 AOI 循环执行
while True:
    result = allocator.run(input_data)

    if result["status"] == "ALL_AOI_FINISHED":
        break

    aoi_plan = result["current_aoi_plan"]
    print(f"\n当前 AOI: {aoi_plan['aoi']}  状态: {aoi_plan['solve_status']}")

    for task in aoi_plan["tasks"]:
        if task["task_type"] == "recon":
            print(f"  侦察: {task['platform']} → {task['sensor']} → {task['cell']}")
        else:
            print(f"  打击: {task['platform']} → {task['target']} "
                  f"{task['munition']}×{task['qty']} ({task['role']})")

    # ═══ 大脑在此执行任务清单，等待底层规控反馈 ═══

    # 准备下一次调用：带回 aoi_route_state + 执行反馈
    input_data["aoi_route_state"] = result["aoi_route_state"]
    input_data["execution_feedback"] = {
        "aoi_id": aoi_plan["aoi"],
        "aoi_status": "FINISHED",           # 大脑标记当前 AOI 完成
        "coverage_rate": 0.85,              # 实际侦察覆盖率
        "detected_targets": ["g1", "g2"],   # 本 AOI 中实际探测到的目标
        "destroyed_targets": ["g1"],        # 本 AOI 中实际摧毁的目标
        "elapsed_time": 12.0,
    }
    # 更新目标状态（大脑维护全局目标列表）
    input_data["targets"] = [...]
```

### 3.3 从 JSON 文件加载（批量场景测试）

```python
from task_interface import load_snapshot_from_json, TaskAllocator

snap = load_snapshot_from_json("scenarios/default.json")
plan = TaskAllocator(solver="cbc", verbose=0).solve(snap)
```

JSON 文件格式见 `scenarios/default.json`。

## 4. 核心数据结构速查

### 4.1 输入 → SituationSnapshot

| 字段 | 类型 | 说明 |
|------|------|------|
| `grids` | `List[GridInfo]` | 5 个栅格（c0 巡逻区 + c1~c4 子区），含天气/地形/先验概率 |
| `targets` | `List[TargetInfo]` | 目标列表，仅 `alive=True AND confirmed=True` 参与 MILP |
| `platforms` | `List[PlatformInfo]` | UAV（EO/SAR/ESM，无弹药）和 HELI（MMW/EOIR，含弹药） |
| `sensor_params` | `List[SensorParams]` | 传感器物理参数（P0, 作用距离 R, 天气敏感性） |
| `staging_position` | `np.ndarray (2,)` | 平台出发位置 (km)，默认 `[150, -50]` |
| `commander_AOI` | `List[str]` | 当前 AOI 标识，如 `["A_3_4"]` |

### 4.2 GridInfo（栅格）

| 字段 | 类型 | 说明 |
|------|------|------|
| `cell_id` | `str` | 标识: `"c0"`(巡逻区), `"c1"`~`"c4"`(子区) |
| `center` | `np.ndarray (2,)` | 栅格中心 `[x, y]` km |
| `weather_w` | `float [0,1]` | 天气系数，≥0.80 时 EO 禁用 |
| `terrain_level` | `int` | 0=平原, 1=丘陵, 2=山地 |
| `target_prior` | `float [0,1]` | 目标存在先验概率 |
| `covered` | `bool` | 是否已被有效侦察覆盖 |

### 4.3 TargetInfo（目标）

| 字段 | 类型 | 说明 |
|------|------|------|
| `tid` | `str` | 唯一标识 |
| `type` | `str` | `"RADAR"` / `"CP"` / `"AV"` |
| `pos_est` | `np.ndarray (2,)` | 位置估计 `[x, y]` km |
| `confirmed` | `bool` | 是否已确认（**False 时 MILP 不分配打击**） |
| `alive` | `bool` | 是否存活（**False 时 MILP 不分配打击**） |
| `value` | `float [0,1]` | 作战价值 |
| `threat` | `float [0,1]` | 威胁度 |

### 4.4 PlatformInfo（平台）

| 字段 | 类型 | 说明 |
|------|------|------|
| `pid` | `str` | 标识，UAV 用 `"U1"`~`"U5"`，HELI 用 `"H1"`~`"H2"` |
| `type` | `str` | `"UAV"` 或 `"HELI"` |
| `pos` | `np.ndarray (2,)` | 当前位置 `[x, y]` km |
| `sensors_available` | `List[str]` | UAV: `["EO","SAR","ESM"]`; HELI: `["MMW","EOIR"]` |
| `munitions` | `dict` | HELI: `{"HF":16,"RKT":76,"GUN":1200}`; UAV: 全 0 |
| `lost` | `bool` | 战损标志 |

### 4.5 输出 → AllocationPlan

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | `str` | `"OPTIMAL"` / `"FEASIBLE"` / `"TIME_LIMIT"` / `"INFEASIBLE"` |
| `objective` | `float` | 目标函数值，有打击时约 800~1000，纯侦察时为负 |
| `solve_time_ms` | `float` | 求解耗时 |
| `recon_assignments` | `List[ReconAssignment]` | 侦察任务列表 |
| `strike_assignments` | `List[StrikeAssignment]` | 打击任务列表 |

**ReconAssignment**: `pid`, `sensor`(EO/SAR/ESM), `cell`(c0~c4), `role`

**StrikeAssignment**: `pid`, `target`(tid), `munition`(HF/RKT/GUN), `qty`, `role`(lead/striker/wing/support)

## 5. 关键约束说明

大脑构造快照时需确保以下条件满足，否则求解器返回 `INFEASIBLE`：

| 约束 | 说明 |
|------|------|
| UAV 数量 ≥ 1 | 至少 1 架 UAV 执行 ESM 全覆盖 |
| 传感器可选 | EO/SAR/ESM 三种传感器的 P0/R 参数需在 sensor_params 中提供 |
| 目标在 AOI 内 | 单 AOI 模式下目标坐标应在 AOI 边界内，否则打击分配可能为空 |
| EO 天气门限 | weather_w ≥ 0.80 的格子上 EO 被自动禁用 |
| 弹药充足 | HELI 弹药需足够覆盖打击需求 |
| 平台未战损 | `lost=True` 的平台自动排除 |

## 6. 典型使用模式

### 模式 A：大脑逐轮调用（单 AOI 滚动优化）

```
大脑维护世界状态
     ↓
构造 SituationSnapshot（当前 AOI + 当前目标状态 + 当前弹药）
     ↓
TaskAllocator.solve() → AllocationPlan
     ↓
大脑解析任务清单 → 下发执行单元
     ↓
执行反馈 → 大脑更新世界状态 → 下一轮
```

### 模式 B：大脑多 AOI 排序 + 逐个执行

```
大脑指定多个候选 AOI
     ↓
MultiAOITaskAllocator.run() 首次 → AOI 排序 + 第一个 AOI 的分配方案
     ↓
大脑执行第一个 AOI → 反馈 FINISHED
     ↓
MultiAOITaskAllocator.run() 续调 → 自动推进到第二个 AOI
     ↓
... 循环直到 ALL_AOI_FINISHED
```

### 模式 C：批量场景测试

```python
from task_interface import load_all_scenarios, TaskAllocator

allocator = TaskAllocator(solver="cbc", verbose=0)
for snap in load_all_scenarios("scenarios"):
    plan = allocator.solve(snap)
    assert plan.status in ("OPTIMAL", "FEASIBLE")
```

## 7. solver 参数调优

```python
allocator = TaskAllocator(
    solver="cbc",         # "cbc"|"gurobi"|"ortools"|"highs"
    time_limit_s=3.0,     # 单次求解时限，默认 3s
    mip_gap=1e-3,         # MIP 间隙阈值，默认 0.1%
    verbose=0,            # 0=静默 1=摘要 2=详细（含求解器日志）
)
```

- 生产环境：`verbose=0`
- 调试环境：`verbose=1`
- 排查不可行解：`verbose=2`
