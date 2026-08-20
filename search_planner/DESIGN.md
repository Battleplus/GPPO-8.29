# SAR无人机搜索路径规划 — 详细设计方案

## 运行方式

### 环境要求

- Python 3.10+
- numpy（项目已安装）
- 必须在 `54_20-master/` 目录下运行（模块依赖 `scenes.air_combat_scene`）

### 快速开始



```python
from sar_search_planner import PlannerConfig, plan

# 方式一：任意矩形边界（最灵活，推荐）
config = PlannerConfig(
    area_bounds_km=(-80, -40, 80, 40),   # 地图西南-东北角矩形
    pattern="racetrack",                  # 跑道形搜索
    angle_deg=30,                         # 偏转角
)
result = plan(config)

# 方式二：中心 + 宽高
config = PlannerConfig(
    area_center_km=(60, -30),             # 区域中心
    area_width_km=20, area_height_km=15,  # 20km×15km
    pattern="sar_rounded",                # 圆角多边形搜索
)

# 方式三：12×12 网格快捷选择
config = PlannerConfig(
    grid_row=6, grid_col=7,               # 第7行第8列（~25km×25km）
    pattern="figure_eight",               # 8字形搜索
)
```

### 查看结果

```python
# 航点列表
for wp in result.waypoints[:5]:
    print(f"({wp.x:.1f}, {wp.y:.1f}, z={wp.z:.1f}m, yaw={wp.yaw_deg:.0f}°)")

# 统计信息
print(result.stats)
# {'collision_count_before': 0, 'collision_count_after': 0,
#  'path_length_km': 91.5, 'mountains_in_area': 3,
#  'raw_waypoint_count': 18, 'final_waypoint_count': 18}

# 区域内的山体
for m in result.mountains_in_area:
    print(f"{m.obstacle_id}: r={m.radius_units:.0f}u, h={m.height_units:.1f}u")
```

### 可选：在 Isaac Sim 中可视化

**方法一：直接运行可视化脚本**

```bash
cd d:/files/54所/54_20-master/54_20-master
python run_sar_visualize.py
```

脚本会自动启动 Isaac Sim、构建场景、运行规划器、叠加可视化元素。
图例：绿色球 = 安全航点，红色球 = 近山航点，黄色线框 = 搜索区域，红色圆柱 = 山体。

> 修改 `run_sar_visualize.py` 中 `main()` 函数里的 `PlannerConfig(...)` 即可切换区域/模式/高度。

**方法二：在代码中手动调用**

```python
# 需要先获取 stage（参考 test_air_combat_scene.py）
from sar_search_planner import export_waypoints_usd, export_search_area_boundary, export_mountain_overlay

export_search_area_boundary(stage, result.search_area)
export_mountain_overlay(stage, result.mountains_in_area)
export_waypoints_usd(stage, result.waypoints)
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `altitude_agl_m` | 6000 | 飞行高度 (m)，降低到 ~800m 可触发山体绕避 |
| `dubins_turn_radius_m` | 5500 | 最小转弯半径 (m) |
| `obstacle_safety_xy_m` | 500 | 山体水平安全余量 (m) |
| `astar_grid_resolution_m` | 500 | A* 栅格分辨率 (m) |
| `pattern` | `"racetrack"` | 路径模式：racetrack / sar_polygon / sar_rounded / figure_eight |

---

## 1. 概述

基于 `test_air_combat_scene.py` 的山地场景（300km×300km，100m/unit），在任意指定的 1/144 小区域（约 25km×25km，12×12 网格）内，为 SAR 无人机生成搜索路径。路径形状参考 `新版xj/Source/demo/SetModel.cpp` 中的四种模式，并引入 **A\* 全局山体避障** + **Dubins 曲线动力学平滑**，输出可直接交予飞控执行的三维航点序列。

## 2. 模块架构

```
sar_search_planner/
├── __init__.py                # 对外统一API导出
├── config.py                  # 全局配置 / 默认参数
├── area.py                    # 搜索区域定义（12×12 网格 / 自定义中心 + 大小）
├── terrain.py                 # 2.5D 高程栅格网构建 + 地形查询接口
├── obstacles.py               # 山体障碍物圆柱体模型 + 碰撞检测
├── astar_planner.py           # A* 全局路径规划（在 2.5D 栅格上绕开超高山体）
├── path_smoother.py           # Dubins 曲线连接 + 航点平滑
├── planner.py                 # 主协调器：组装各模块，输入配置 → 输出最终航点
├── visualize.py               # USD 可视化叠加层（可选，用于 Isaac Sim 内回放）
└── paths/
    ├── __init__.py            # 路径工厂函数
    ├── sar_polygon.py         # 多边形螺旋（参考 SetModel_SAR，C++ L547）
    ├── sar_rounded.py         # 圆角多边形螺旋（参考 SetModel_SAR_Rounded，C++ L705）
    ├── racetrack.py           # 跑道形光栅扫描（参考 SetModel_Photoelectric，C++ L13）
    └── figure_eight.py        # 8字形（参考 SetModel_AntiRadiation，C++ L317）
```

纯 Python 实现，仅依赖 `numpy` 和 `math`，所有模块计算独立于 Isaac Sim 运行时。

## 3. 各模块详细设计

---

### 3.1 配置 (`config.py`)

```python
@dataclass
class PlannerConfig:
    # ── 搜索区域（三选一，优先级：bounds > center > grid） ──
    grid_row: int | None = None              # 12×12 网格行号 [0,11]（快捷方式，每个格 25×25km）
    grid_col: int | None = None              # 12×12 网格列号 [0,11]
    area_center_km: tuple[float, float] | None = None  # 区域中心 (cx_km, cy_km)，地图中心为 (0,0)
    area_width_km: float = 25.0              # 区域宽度 X 方向 (km)
    area_height_km: float = 25.0             # 区域高度 Y 方向 (km)，默认与宽度相同
    area_bounds_km: tuple[float, float, float, float] | None = None
        # 任意矩形边界 (x_min_km, y_min_km, x_max_km, y_max_km)，地图中心为原点

    # ── 路径模式 ──
    pattern: str = "racetrack"         # "racetrack" | "sar_polygon" | "sar_rounded" | "figure_eight"
    angle_deg: float = 0.0             # 路径偏转角（0=正北，顺时针为正）
    clockwise: bool = True             # 遍历方向

    # ── 飞行参数 ──
    altitude_agl_m: float = 6000.0     # 相对地面高度 (m)
    cruise_speed_mps: float = 200.0    # 巡航速度 (m/s)，用于 Dubins 曲线

    # ── 避障参数 ──
    obstacle_safety_xy_m: float = 500.0   # 山体水平安全余量 (m)
    obstacle_safety_z_m: float = 120.0    # 山体垂直安全余量 (m)
    astar_grid_resolution_m: float = 500.0  # A* 栅格分辨率 (m)

    # ── Dubins 平滑参数 ──
    dubins_turn_radius_m: float = 5500.0  # 最小转弯半径 (m)，demo 推荐 ≥ 5000
    dubins_sample_step_m: float = 800.0   # Dubins 路径采样步长 (m)

    # ── 各模式专用参数 ──
    # racetrack
    racetrack_length_km: float = 20.0
    racetrack_width_km: float = 12.0
    racetrack_path_count: int = 14     # 跑道折返段数

    # sar_polygon / sar_rounded
    sar_radius_km: float = 10.0        # 多边形外接圆半径 (km)
    sar_sides: int = 8                 # 多边形边数 (3-20)
    sar_loops: int = 1                 # 圈数
    sar_turn_radius_km: float = 5.5    # 仅 sar_rounded 使用

    # figure_eight
    eight_radius_km: float = 12.0      # 两圆半径 (km)
    eight_line_km: float = 25.0        # 两圆心间距 (km)，必须 > 2*radius
    eight_loops: int = 1               # 圈数
```

**默认配置继承链**：`PlannerConfig()` 全使用默认值；用户可逐字段覆盖。内部使用 `meters_per_unit = 100.0` 完成 km/m ↔ scene_unit 的转换。

---

### 3.2 搜索区域 (`area.py`)

**目标**：提供三种灵活的接口，让用户可以**在大场景中的任意位置、任意大小**划定搜索矩形区域。

#### 坐标系说明

全图场景坐标系：原点为地图中心，X 向东，Y 向北。地图范围 `[-1500, 1500] × [-1500, 1500]`（场景单位，300km×300km @ 100m/unit）。

> 接口统一使用 **km 偏移量**（以地图中心为原点），内部自动转换到场景单位。

#### 三种区域指定方式

```python
@dataclass
class SearchArea:
    """
    搜索区域矩形。内部统一使用场景单位。
    提供便捷属性用于坐标转换和边界查询。
    """
    x_min: float            # 场景单位
    x_max: float
    y_min: float
    y_max: float

    @property
    def center_x(self) -> float: ...
    @property
    def center_y(self) -> float: ...
    @property
    def width_units(self) -> float: ...
    @property
    def height_units(self) -> float: ...
    @property
    def width_km(self) -> float: ...
    @property
    def height_km(self) -> float: ...


# ─── 方式一：任意矩形边界（最灵活，推荐） ───

def area_from_bounds_km(x_min_km: float, y_min_km: float,
                        x_max_km: float, y_max_km: float,
                        map_size_km: float = 300.0,
                        meters_per_unit: float = 100.0) -> SearchArea:
    """
    直接指定矩形边界（km 偏移量，以地图中心为原点）。
    自动裁剪到地图边界内。
    支持任意长宽比。

    示例：
      # 地图西南角 20km×30km 的区域
      area = area_from_bounds_km(-150, -150, -130, -120)

      # 地图东北角 15km×15km 的区域
      area = area_from_bounds_km(80, 60, 95, 75)
    """


# ─── 方式二：中心 + 宽高（任意大小的矩形） ───

def area_from_center_km(center_x_km: float, center_y_km: float,
                        width_km: float, height_km: float | None = None,
                        map_size_km: float = 300.0,
                        meters_per_unit: float = 100.0) -> SearchArea:
    """
    指定区域中心和宽高。
    height_km 若为 None 则等于 width_km（正方形）。
    自动裁剪到地图边界内。

    示例：
      # 以 (-80, 40) km 为中心，20km×15km 的矩形区域
      area = area_from_center_km(-80, 40, 20, 15)

      # 以 (60, -30) km 为中心，10km×10km 的正方形区域
      area = area_from_center_km(60, -30, 10)
    """


# ─── 方式三：12×12 网格快捷选择（1/144 场景） ───

def area_from_grid_cell(row: int, col: int,
                        map_size_km: float = 300.0,
                        meters_per_unit: float = 100.0) -> SearchArea:
    """
    将全图 12×12 均匀划分，选择指定格号。
    每个格约 25km×25km。
    row=0 最南，col=0 最西。
    row/col 范围 [0, 11]。

    示例：
      area = area_from_grid_cell(6, 7)  # 第7行第8列
    """


# ─── 辅助：场景单位边界查询 ───

def clamp_area_to_map(area: SearchArea,
                      map_size_units: float = 3000.0) -> SearchArea:
    """将 SearchArea 裁剪到地图范围内，若完全出界则抛出 ValueError。"""
```

#### 选择建议

| 场景 | 推荐接口 |
|------|---------|
| 已知具体坐标范围 | `area_from_bounds_km()` |
| 已知中心点和覆盖范围 | `area_from_center_km()` |
| 快速 1/144 网格测试 | `area_from_grid_cell()` |

三种方式内部统一转换为 `SearchArea`（场景单位），后续所有模块只依赖 `SearchArea`。

---

### 3.3 2.5D 高程栅格 (`terrain.py`)

**目标**：在搜索区域内建立均匀采样的高程图，为 A\* 规划器和 Z 高度赋值提供 O(1) 查询。

```python
class TerrainGrid:
    """
    搜索区域内的 2.5D 高程栅格。
    底层调用 scenes.air_combat_scene.terrain_height() 进行采样。
    """

    def __init__(self, search_area: SearchArea,
                 resolution_m: float = 500.0,
                 map_size_units: float = 3000.0,
                 mountain_height_units: float = 15.0,
                 meters_per_unit: float = 100.0):
        """
        resolution_m: 栅格分辨率（默认 500m）。
        栅格尺寸 ≈ (25km / 500m) × (25km / 500m) = 50×50 = 2500 个 cell。
        """

    def height_at(self, x: float, y: float) -> float:
        """双线性插值查询任意点的地形高度。返回场景单位。"""

    def is_passable(self, x: float, y: float,
                    mountain_obstacles: list[MountainObstacle],
                    safety_xy: float) -> bool:
        """
        判断点 (x,y) 是否可通行（不在任何山体圆柱内）。
        safety_xy 为水平安全余量（场景单位）。
        """

    def grid(self) -> np.ndarray:
        """返回 2D 高度数组 (rows×cols)，供 A* 使用。"""

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """场景坐标 → 栅格索引。"""

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """栅格索引 → 场景坐标（格子中心）。"""
```

**构建过程**：
1. 计算栅格行列数：`cols = ceil(width / resolution_units) + 1`，`rows` 同理
2. 遍历每个格点 `(col, row)`，映射回场景坐标 `(x, y)`
3. 调用 `terrain_height(x, y, map_size_units, mountain_height_units)` 填充高度值
4. 缓存为 `(rows, cols)` 的 `np.ndarray`

---

### 3.4 山体障碍物 (`obstacles.py`)

```python
@dataclass
class MountainObstacle:
    obstacle_id: str       # e.g. "MountainPeak_West"
    center_x: float        # 场景单位
    center_y: float
    radius_units: float    # 圆柱半径
    height_units: float    # 山顶最大高度
```

**构建函数**：

```python
def build_all_mountains(map_size_units: float = 3000.0,
                        mountain_height_units: float = 15.0) -> list[MountainObstacle]:
    """
    从 scenes.air_combat_scene.MOUNTAIN_OBSTACLE_SPECS 计算全部 5 座山的参数。
    计算逻辑与 _build_environment_obstacles() 中的 mountain 分支一致：
      - x = nx * half, y = ny * half
      - z = terrain_height(x, y, map_size, height_scale)
      - radius = max(3.0, map_size * radius_ratio)
      - height_units = max(z, height_scale * 0.35)
    """

def filter_mountains_in_area(mountains: list[MountainObstacle],
                             search_area: SearchArea) -> list[MountainObstacle]:
    """
    筛选与搜索区域有交集的山体。
    判定：山体的 AABB (center ± radius) 与 search_area 矩形相交。
    """

def check_collision(x: float, y: float, z: float,
                    mountain: MountainObstacle,
                    safety_xy: float = 0.0) -> bool:
    """
    碰撞检测：2D 距离 < (radius + safety_xy) 且 z < height_units。
    """
```

**设计要点**：仅处理 5 座山体（`MOUNTAIN_OBSTACLE_SPECS`），不处理森林树木（46 棵独立树木）和石场（3 个石场区），这些留给局部避障模块。

---

### 3.5 A\* 全局山体避障 (`astar_planner.py`)

**目标**：当地形路径的某一段穿过山体障碍物圆柱时，使用 A\* 在 2.5D 高程栅格上寻找绕行路径。这是对原方案中简单"径向推出"策略的升级。

**算法设计**：

```
输入：起点 (sx, sy)、终点 (gx, gy)、TerrainGrid、MountainObstacle 列表
输出：绕行后的中间航点列表 [(x1,y1), (x2,y2), ...]，或空列表（原直线可直接通行）

1. 在起点和终点之间按步长采样直线段上的点
2. 若所有采样点都可通行（is_passable），返回空列表 → 无需绕行
3. 否则，在 TerrainGrid 上运行 A*：
   a. 起点 = world_to_grid(sx, sy)，终点 = world_to_grid(gx, gy)
   b. 启发函数：欧几里得距离（场景单位）
   c. 移动代价：
      - 直线代价 = 场景距离
      - 若目标格子在任意山体圆柱内：代价 += 1e9（不可通行）
      - 若目标格子地形坡度 > 阈值（如 30°）：代价 += 额外惩罚（可选）
   d. 8 邻域搜索（允许对角线移动，代价 ×√2）
   e. 开放列表使用优先队列（heapq）
4. A* 若找到路径：对路径点做 Douglas-Peucker 简化，输出关键拐点
   若未找到路径：返回空列表（保留原直线，交由上层决策——爬升或标记）
```

**关键参数**：
- `resolution_m = 500.0`（栅格分辨率，约 2500 个 cell / 25km² 区域）
- `safety_xy` = 从 PlannerConfig 传入的水平安全余量（默认 500m）
- `max_slope_deg = 30.0`（可选坡度过滤，用于排除悬崖）

**复杂度**：50×50 栅格上 A* 最坏约 2500 个节点，实际搜索空间更小（只在山体周围展开），单次 A* 约 <5ms。

**与后续模块的衔接**：A\* 输出的关键拐点列表，会插入到原始路径中替换直线段，然后再经过 Dubins 平滑。

---

### 3.6 路径生成 (`paths/*.py`)

四个路径生成器，统一接口：

```python
def generate_xy_waypoints(center_x: float, center_y: float,
                          angle_deg: float, clockwise: bool,
                          meters_per_unit: float,
                          **pattern_params) -> list[tuple[float, float]]:
    """
    返回 (x, y) 列表（场景单位），不含高度。
    """
```

#### 3.6.1 SAR 多边形 (`paths/sar_polygon.py`)

参考 `SetModel_SAR`（C++ L547-L693）。

**算法**：
1. `R = radius_km * 1000 / meters_per_unit`（外接圆半径，场景单位）
2. 生成 N 个顶点：`theta_i = -2π·i/sides`，`vx = R·cos(θ)`，`vy = R·sin(θ)`
3. 旋转：`rad = (90 - angle_deg) * π/180`，`(rx, ry) = (vx·cos(rad) - vy·sin(rad), vx·sin(rad) + vy·cos(rad))`
4. 平移至 center
5. 找最近顶点作为入口
6. 按顺时针/逆时针遍历所有顶点，loops 圈
7. 闭合回到第一顶点

#### 3.6.2 SAR 圆角多边形 (`paths/sar_rounded.py`)

参考 `SetModel_SAR_Rounded`（C++ L705-L928）。

**算法**：在上方多边形算法基础上，每个顶点处用圆弧替换尖角：
1. `r = turn_radius_km * 1000 / meters_per_unit`
2. `r = min(r, R * cos(π/sides) * 0.99)`（限制转弯半径不大于多边形可容纳的最大值）
3. 对于每个顶点 V[i]：
   - 计算切点距离 `L_t = r * tan(π/sides)`
   - 计算圆心距离 `D_c = r / sin(π/sides)`
   - incoming 切点 = V[i] 沿入边方向移动 L_t
   - outgoing 切点 = V[i] 沿出边方向移动 L_t
   - 圆心 C = V[i] 沿角平分线移动 D_c
   - 在入切点和出切点之间生成 K=3 个圆弧采样点
4. 输出路径：切点 → 圆弧点 → 切点 → 下一条边...

#### 3.6.3 跑道形 (`paths/racetrack.py`)

参考 `SetModel_Photoelectric`（C++ L13-L304）。

**算法**：
1. 矩形参数：`hl = length_km/2`, `hw = width_km/2`（场景单位）
2. 四角点：`(-hl,-hw), (+hl,-hw), (+hl,+hw), (-hl,+hw)`
3. 矩形边（0→1, 1→2 等顺时针）：
   - 长边（0→1, 2→3）：直线，细分为 `path_count/2 - 1` 个点
   - 短边（1→2, 3→0）：圆弧过渡，`arcCurvature = 0.7`
     - 圆心在边中点外侧偏移量 = `(dist/2) * tan(π·arcCurvature/4)`
     - 弧半径 = `(dist/2) / cos(π·arcCurvature/4)`
     - 细分 `seg_per_arc = 3` 个点
4. 旋转 + 平移至 center
5. 取最近入口点，按方向遍历

#### 3.6.4 8字形 (`paths/figure_eight.py`)

参考 `SetModel_AntiRadiation`（C++ L317-L535）。

**算法**：
1. `L = (line_km/2) * 1000 / meters_per_unit`（半间距），`R = radius_km * 1000 / meters_per_unit`
2. `α = acos(R / L)`
3. 左圆（圆心 `(-L, 0)`，弧 `α → 2π-α`，6 段）→ 直线段（3 段）→ 右圆（圆心 `(+L, 0)`，弧 `π-α → -π+α`，6 段）→ 直线段（3 段回到起点）
4. 旋转 + 平移至 center
5. 判断飞机在左半区还是右半区（`y_local > 0` 判断），决定顺/逆时针遍历

---

### 3.7 路径生成工厂 (`paths/__init__.py`)

```python
_PATTERN_REGISTRY: dict[str, Callable] = {
    "sar_polygon": generate_sar_polygon,
    "sar_rounded": generate_sar_rounded,
    "racetrack":   generate_racetrack,
    "figure_eight": generate_figure_eight,
}

def generate_path(pattern: str, center_x: float, center_y: float,
                  angle_deg: float, clockwise: bool,
                  meters_per_unit: float, **kwargs) -> list[tuple[float, float]]:
    """根据模式名称调度对应的生成函数。"""
```

---

### 3.8 Dubins 曲线平滑 (`path_smoother.py`)

**目标**：模仿 demo 中 `CutInSetPathOptLatLon_Sar()` 的模式——先飞一段直线到目标搜索图形入口，然后用 Dubins 曲线平滑连接所有断点。

**Dubins 路径核心**（参考 `Detect/dubins.h` 和 `dubins.cpp`）：

```
DubinsPathType: LSL, LSR, RSL, RSR, RLR, LRL (6种)
配置: q0 = (x0, y0, θ0), q1 = (x1, y1, θ1), rho = 转弯半径
输出: 3段路径 (弧-直线-弧 或 弧-弧-弧)，每段由长度和类型定义
```

**Python 实现策略**：参考开源 `dubins` 库算法，实现 `dubins_shortest_path()` 的核心逻辑：

```python
@dataclass
class DubinsPath:
    qi: tuple[float, float, float]      # 起点 (x, y, θ)
    param: tuple[float, float, float]   # 三段长度
    rho: float                          # 转弯半径
    type: int                           # 0-5，对应 LSL/.../LRL

def dubins_shortest_path(q0: tuple[float, float, float],
                         q1: tuple[float, float, float],
                         rho: float) -> DubinsPath:
    """计算两点间最短 Dubins 路径。"""

def sample_dubins_path(path: DubinsPath, step_size: float) -> list[tuple[float, float, float]]:
    """按步长采样 Dubins 路径，返回 (x, y, θ) 列表。"""
```

**平滑流程**：

```python
def smooth_waypoints(xy_waypoints: list[tuple[float, float]],
                     turn_radius: float,
                     sample_step: float,
                     start_pose: tuple[float, float, float] | None = None
                     ) -> list[tuple[float, float]]:
    """
    1. 计算每个航点的朝向 θ = atan2(y_next - y, x_next - x)
    2. 对于相邻航点 (p_i, p_{i+1})：
       a. 计算 Dubins 路径：q0 = (x_i, y_i, θ_i), q1 = (x_{i+1}, y_{i+1}, θ_{i+1}), rho = turn_radius
       b. 以 sample_step 采样 Dubins 路径
       c. 将采样点加入输出列表
    3. 返回平滑后的连续航点序列
    """
```

**设计要点**：
- 转弯半径默认 5500m（匹配 demo 中的 `kTurnRadiusM = 5500.0`）
- 采样步长默认 800m（匹配 demo 中的 `kNominalStepM = 1200.0` 逻辑）
- 若两点间距 < 2*转弯半径（无法构成 Dubins 曲线），保留为直线连接

---

### 3.9 主协调器 (`planner.py`)

**核心数据流**：

```
PlannerConfig
    │
    ▼
[1] 加载场景参数（map_size_km, meters_per_unit, mountain_height_m）
    │  from scenes.air_combat_scene.DEFAULT_AIR_COMBAT_CONFIG
    ▼
[2] 定义搜索区域
    │  area_from_grid_cell() 或 area_from_center()
    ├─→ SearchArea (场景单位)
    ▼
[3] 构建 TerrainGrid
    │  在搜索区域内以 resolution_m 采样高程
    ├─→ np.ndarray (rows×cols)
    ▼
[4] 检测范围内山体
    │  build_all_mountains() → filter_mountains_in_area()
    ├─→ list[MountainObstacle]（仅与搜索区域相交的山）
    ▼
[5] 生成 2D 搜索路径 (XY only)
    │  paths.generate_path(pattern, center, angle, ...)
    ├─→ list[(x, y)]（场景单位）
    ▼
[6] 全局 A* 避障（对每个直线段）
    │  对相邻航点 (p_i → p_{i+1})：
    │    a. 采样直线上的点，用 is_passable() 检测碰撞
    │    b. 若有碰撞 → A* 绕行 → 插入中间拐点
    │    c. 若无碰撞 → 保留原直线
    ├─→ list[(x, y)]（插入绕行拐点后的路径）
    ▼
[7] Z 高度赋值
    │  对每个 (x, y) 航点：
    │    terrain_z = terrain_grid.height_at(x, y)
    │    z = terrain_z + altitude_agl_m / meters_per_unit
    │  若航点落在山体上方且 z < mountain.height_units + safety_z：
    │    z = max(z, mountain.height_units + safety_z_m / meters_per_unit)
    ├─→ list[(x, y, z, terrain_z)]
    ▼
[8] Dubins 曲线平滑
    │  path_smoother.smooth_waypoints(xy_list, turn_radius, sample_step)
    ├─→ list[(x, y)]（平滑后的 2D 航点序列，更密集）
    ▼
[9] Z 高度再赋值
    │  对平滑后的每个点重新采样 terrain_z 并计算 z
    ├─→ list[Waypoint]
    ▼
[10] 后处理 & 统计
    │  • 去重（连续重复点合并）
    │  • 统计碰撞数（before / after ·······2）
    │  • 输出总路径长度
    ├─→ PlannerResult
```

**PlanerResult 结构**：

```python
@dataclass
class Waypoint:
    x: float           # 场景单位
    y: float
    z: float           # 绝对高度（场景单位）
    terrain_z: float   # 地面高度（场景单位）
    yaw_deg: float     # 航向角（0=北，顺时针为正）

@dataclass
class PlannerResult:
    waypoints: list[Waypoint]
    raw_waypoints_before_avoidance: list[Waypoint]
    mountains_in_area: list[MountainObstacle]
    search_area: SearchArea
    config: PlannerConfig
    stats: dict         # collision_count_before, collision_count_after,
                        # path_length_km, astar_calls, dubins_segments
```

**主入口函数**：

```python
def plan(config: PlannerConfig | None = None,
         map_size_km: float = 300.0,
         meters_per_unit: float = 100.0,
         mountain_height_m: float = 1500.0) -> PlannerResult:
    """
    主入口。
    若 config 为 None，使用全部默认参数。
    """
```

---

### 3.10 对外接口 (`__init__.py`)

```python
from .config import PlannerConfig
from .planner import plan, PlannerResult, Waypoint
from .area import SearchArea, area_from_grid_cell, area_from_center_km, area_from_bounds_km
from .obstacles import MountainObstacle
from .visualize import export_waypoints_usd, export_search_area_boundary

__all__ = [
    "PlannerConfig",
    "plan",
    "PlannerResult",
    "Waypoint",
    "SearchArea",
    "area_from_grid_cell",
    "area_from_center_km",
    "area_from_bounds_km",
    "MountainObstacle",
    "export_waypoints_usd",
    "export_search_area_boundary",
]
```

---

### 3.11 可视化 (`visualize.py`，可选)

```python
def export_waypoints_usd(stage, waypoints: list[Waypoint],
                         base_path: str = "/World/AirCombat/SAR_Search") -> None:
    """在 USD stage 中创建航点球体（绿色=正常，红色=曾经碰撞）和连线。"""

def export_search_area_boundary(stage, search_area: SearchArea,
                                base_path: str = "/World/AirCombat/SAR_Search") -> None:
    """绘制搜索区域边框（黄色线框）。"""

def export_mountain_overlay(stage, mountains: list[MountainObstacle],
                            base_path: str = "/World/AirCombat/SAR_Search") -> None:
    """绘制山体圆柱体（红色半透明）。"""
```

---

## 4. 全局数据流总览

```
            ┌──────────────────┐
            │   PlannerConfig  │  (用户指定区域、模式、参数)
            └────────┬─────────┘
                     │
     ┌───────────────┼───────────────────┐
     │               ▼                   │
     │  ┌──────────────────────┐         │
     │  │  scenes.air_combat_  │         │
     │  │  scene:              │         │
     │  │  · terrain_height()  │─────────┤
     │  │  · MOUNTAIN_OBSTACLE │         │
     │  │    _SPECS            │         │
     │  │  · DEFAULT_CONFIG    │         │
     │  └──────────────────────┘         │
     │                                   │
     ▼                                   ▼
┌─────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────┐
│  Area   │  │ Terrain  │  │  Obstacles   │  │   Paths    │
│  定义   │  │  2.5D栅格│  │  山体圆柱列表 │  │  2D形状生成│
└────┬────┘  └────┬─────┘  └──────┬───────┘  └─────┬──────┘
     │            │               │                │
     └────────────┼───────────────┼────────────────┘
                  │               │
                  ▼               ▼
         ┌────────────┐  ┌──────────────┐
         │  A* 全局   │  │ 原始 XY 航点 │
         │  山体避障  │◄─┤  (含碰撞点)  │
         └─────┬──────┘  └──────────────┘
               │
               ▼
         ┌────────────┐
         │ Z 高度赋值  │  (terrain_height + altitude_agl)
         └─────┬──────┘
               │
               ▼
         ┌────────────┐
         │  Dubins    │  (连接断点，动力学平滑)
         │  曲线平滑  │
         └─────┬──────┘
               │
               ▼
         ┌────────────┐
         │ 后处理     │  (去重, 统计, 最终航点)
         └─────┬──────┘
               │
               ▼
         ┌────────────┐
         │PlannerResult│ → 可对接飞控 / USD 可视化
         └────────────┘
```

---

## 5. 使用示例

```python
from sar_search_planner import PlannerConfig, plan, export_waypoints_usd

# ── 示例1：任意矩形边界（最灵活） ──
config = PlannerConfig(
    area_bounds_km=(-120, -50, -90, -20),   # 地图西南区域 30km×30km
    pattern="racetrack",
    angle_deg=30.0,
)
result = plan(config)

# ── 示例2：中心 + 宽高指定 ──
config2 = PlannerConfig(
    area_center_km=(60, -30),               # 地图东南方向
    area_width_km=20.0,
    area_height_km=15.0,                    # 矩形区域 20km×15km
    pattern="sar_rounded",
    sar_radius_km=8.0,
    sar_sides=6,
    sar_turn_radius_km=4.0,
)
result2 = plan(config2)

# ── 示例3：12×12 网格快捷选择 ──
config3 = PlannerConfig(
    grid_row=6, grid_col=7,                 # 第7行第8列（~25km×25km）
    pattern="figure_eight",
    altitude_agl_m=5000,
)
result3 = plan(config3)

# ── 示例4：在 Isaac Sim 中可视化 ──
# export_waypoints_usd(stage, result.waypoints)
```

---

## 6. 不做的事情

- **不** 处理树木/石场障碍物（由另一位同学的局部避障模块处理）
- **不** 修改 `scenes/air_combat_scene.py` 或其他现有文件
- **不** 实现实时在线重规划（只做离线路径生成，可被在线模块调用）
- **不** 在路径生成时考虑风速/天气等动态因素

## 7. 验证与测试

1. **单元测试**：每个路径生成器独立测试，验证输出航点形状与 C++ 版本一致
2. **碰撞检测测试**：构造已知在山体内/外的航点，验证 `check_collision()` 返回值正确
3. **A\* 绕行测试**：在包含座山的区域规划一条穿过山体的直线，验证 A\* 输出有效绕行路径
4. **Dubins 平滑测试**：输入折线路径，验证输出路径曲率连续、无大角度突变
5. **端到端测试**：对 144 个网格格号各运行一次 plan()（或用采样），验证无异常
6. **可视化验证**（可选）：在 Isaac Sim 中渲染航点和山体，目视检查
