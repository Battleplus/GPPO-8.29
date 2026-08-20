# 任务分配模块 —— 外部接口协议

> 本文档定义任务分配模块与外部模块之间的**数据格式协议**。
> 所有字段均有明确类型、范围、是否必填。使用 JSON 作为交换格式，语言无关。

---

## 1. 整体数据流

```
态势理解模块(大脑)                任务分配模块(本模块)              底层规控(执行层)
       │                                │                              │
       │── 态势输入 JSON ──→            │                              │
       │   (模式A: 单AOI 或 模式B: 多AOI)                              │
       │                                │                              │
       │                        ① AOI 排序（多AOI时）                   │
       │                        ② 构造当前AOI快照                        │
       │                        ③ MILP 求解                             │
       │                                │                              │
       │                                │── 执行清单 JSON ──→           │
       │                                │   (ExecutionOrder)           │
       │                                │                              │── 逐条执行
       │                                │                              │
       │←── 执行反馈 JSON ──────────────│──────────────────────────────│
       │    (ExecutionFeedback)         │                              │
       │                                │                              │
       │── 带 aoi_route_state ──────→   │  (多AOI时循环此步骤)           │
       │    的态势输入 JSON             │                              │
```

---

## 2. 两种调用模式

| 模式 | 适用场景 | 输入特征 | 调用方式 |
|------|---------|---------|---------|
| **A: 单 AOI** | 大脑已选定一个 AOI，直接求解 | 含 `aoi_id` + `grids` | 一次调用 |
| **B: 多 AOI** | 大脑提供 2~4 个候选 AOI，由模块排序后逐个执行 | 含 `aois` 数组 | 循环调用（每轮带回 `aoi_route_state` + `execution_feedback`） |

---

## 3. 公共数据结构

以下数据结构在两种模式中通用。

### 3.1 平台 (platforms 数组元素)

```json
{
  "pid": "U1",
  "type": "UAV",
  "pos": [150.0, -50.0],
  "sensors": ["EO", "SAR", "ESM"],
  "munitions": {"HF": 0, "RKT": 0, "GUN": 0},
  "alt": 2.0,
  "lost": false
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `pid` | string | 是 | 平台唯一标识，UAV 用 `U1`~`Un`，HELI 用 `H1`~`Hn` |
| `type` | string | 是 | `UAV` 或 `HELI` |
| `pos` | [float, float] | 是 | 当前位置 (x, y) km |
| `sensors` | string[] | 是 | UAV 固定 `["EO","SAR","ESM"]`；HELI 固定 `["MMW","EOIR"]` |
| `munitions` | object | 是 | HELI 必填 `{"HF":16,"RKT":76,"GUN":1200}`；UAV 全填 0 |
| `alt` | float | 是 | 高度 km，UAV=2.0，HELI=3.0 |
| `lost` | bool | 是 | 战损标志，true 时该平台不参与分配 |

### 3.2 目标 (targets 数组元素)

```json
{
  "tid": "g1",
  "type": "RADAR",
  "pos": [162.0, 112.0],
  "confirmed": true,
  "alive": true,
  "value": 0.97,
  "threat": 0.91,
  "pos_cov": [[0.1, 0.0], [0.0, 0.1]],
  "velocity": [0.0, 0.0]
}
```

| 字段 | 类型 | 必填 | 范围/约束 | 说明 |
|------|------|------|-----------|------|
| `tid` | string | 是 | 唯一 | 目标标识 |
| `type` | string | 是 | `RADAR` / `CP` / `AV` | 目标类型 |
| `pos` | [float, float] | 是 | — | 位置估计 (x, y) km |
| `confirmed` | bool | 是 | — | **false 时不参与打击分配（先侦后打门控）** |
| `alive` | bool | 是 | — | **false 时不参与任何分配** |
| `value` | float | 是 | [0, 1] | 归一化作戰价值 |
| `threat` | float | 是 | [0, 1] | 威胁度 |
| `pos_cov` | float[2][2] | 否 | — | 位置协方差矩阵，默认 `[[0.1,0],[0,0.1]]` |
| `velocity` | [float, float] | 否 | — | 速度估计 (vx, vy) km/tick，默认 `[0,0]` |

### 3.3 传感器参数 (sensors 数组元素)

```json
{
  "name": "EO",
  "P0": 0.85,
  "R": 15.0,
  "weather_sensitive": true
}
```

| 字段 | 类型 | 必填 | 范围 | 说明 |
|------|------|------|------|------|
| `name` | string | 是 | EO/SAR/ESM/MMW/EOIR | 传感器名称 |
| `P0` | float | 是 | [0, 1] | 理想条件下基础探测概率 |
| `R` | float | 是 | >0 | 最大作用距离 km |
| `weather_sensitive` | bool | 是 | — | 是否受天气衰减影响（EO:true, SAR:false, ESM:false） |

---

## 4. 模式 A：单 AOI 输入

**场景**: 大脑已选定一个 AOI，直接求解任务分配。

### 4.1 顶层结构

```json
{
  "aoi_id": "A_3_4",
  "staging_position": [150.0, -50.0],
  "grids": [ ... ],
  "targets": [ ... ],
  "platforms": [ ... ],
  "sensors": [ ... ],
  "commander_aoi": ["A_3_4"],
  "los_matrix": null,
  "occlusion_matrix": null
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `aoi_id` | string | 是 | AOI 标识，格式 `"A_{row}_{col}"`，如 `"A_3_4"` |
| `staging_position` | [float, float] | 否 | 平台出发坐标 (x, y) km，默认 `[150, -50]` |
| `grids` | array | 是 | 5 个栅格对象（c0 巡逻区 + c1~c4 子区） |
| `targets` | array | 否 | 目标列表，见 §3.2。建议仅传入坐标落在本 AOI 内的目标 |
| `platforms` | array | 是 | 平台列表，见 §3.1 |
| `sensors` | array | 否 | 传感器参数，见 §3.3。默认 EO/SAR/ESM |
| `commander_aoi` | string[] | 否 | 指挥官关注 AOI 列表 |
| `los_matrix` | float[][] | 否 | 直升机-目标通视矩阵，默认全 1 |
| `occlusion_matrix` | float[][] | 否 | 遮挡衰减矩阵，默认全 1 |

### 4.2 栅格对象 (grids 数组元素)

```json
{
  "cell_id": "c1",
  "center": [162.5, 112.5],
  "size": [25.0, 25.0],
  "weather_w": 0.15,
  "terrain_level": 0,
  "target_prior": 0.25,
  "covered": false
}
```

| 字段 | 类型 | 必填 | 范围/约束 | 说明 |
|------|------|------|-----------|------|
| `cell_id` | string | 是 | `c0`~`c4` | c0=巡逻区(50×50km)，c1~c4=子区(25×25km) |
| `center` | [float, float] | 是 | — | 栅格中心坐标 (x, y) km |
| `size` | [float, float] | 是 | c0用[50,50]，其余用[25,25] | 宽×高 km |
| `weather_w` | float | 是 | [0, 1] | 天气系数，**≥0.80 时 EO 传感器被禁用** |
| `terrain_level` | int | 是 | {0,1,2} | 0=平原, 1=丘陵, 2=山地 |
| `target_prior` | float | 是 | [0, 1] | 该栅格存在目标的先验概率 |
| `covered` | bool | 是 | — | 是否已被有效侦察覆盖 |

### 4.3 模式 A 完整示例

见 [§8.1 单 AOI 示例](#81-模式-a-单-aoi-完整示例)。

---

## 5. 模式 B：多 AOI 输入

**场景**: 大脑指定 2~4 个候选 AOI，本模块排序后逐个求解。

### 5.1 核心概念

| 概念 | 说明 |
|------|------|
| **AOI 排序** | 模块内部枚举全排列，选综合得分最高的执行顺序。得分 = 折扣价值 - 距离惩罚 |
| **逐 AOI 执行** | 每次调用只处理排序后的第一个待执行 AOI，返回该 AOI 的任务清单 |
| **aoi_route_state** | 排序结果 + 执行进度，由本模块产出、大脑保存、下次调用带回 |
| **execution_feedback** | 大脑告诉本模块"上一个 AOI 已完成"，触发推进到下一个 AOI |

**多 AOI 模式下，大脑不需要自己构造 grids**。模块会根据 AOI 的 row/col 自动生成 5 个栅格。

### 5.2 顶层结构

```json
{
  "aois": [ ... ],
  "platforms": { ... },
  "targets": [ ... ],
  "staging_position": [150.0, -50.0],
  "sensor_params": null,
  "grid_weather": null,
  "aoi_route_state": null,
  "execution_feedback": null,
  "cycle_id": 0
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `aois` | array | 是 | 2~4 个 AOI 描述对象，见 §5.3 |
| `platforms` | object | 是 | 按类型配置，见 §5.4（与模式 A 不同，用批量配置格式） |
| `targets` | array | 是 | **全局**目标列表，模块按坐标自动筛选归属 AOI，见 §3.2 |
| `staging_position` | [float, float] | 否 | 出发点坐标 (x, y) km，默认 `[150, -50]` |
| `sensor_params` | array | 否 | 传感器参数，见 §3.3。null 则用默认值 |
| `grid_weather` | object | 否 | 按 cell_id 覆盖天气，如 `{"c3": 0.85}` |
| `aoi_route_state` | object/null | **首次传 null**，后续带回 | AOI 执行状态，见 §5.5 |
| `execution_feedback` | object/null | **首次传 null**，后续带回 | 执行反馈，见 §5.6 |
| `cycle_id` | int | 否 | 分配轮次，默认 0 |

### 5.3 AOI 描述对象 (aois 数组元素)

```json
{
  "id": "A_5_6",
  "row": 5,
  "col": 6,
  "priority": 0.8,
  "target_prior": 0.70,
  "target_value": 0.97,
  "target_threat": 0.91
}
```

| 字段 | 类型 | 必填 | 范围 | 说明 |
|------|------|------|------|------|
| `id` | string | 是 | 唯一 | AOI 标识，如 `"A_5_6"` |
| `row` | int | 是 | 1~6 | AOI 行号（y 方向自上而下） |
| `col` | int | 是 | 1~6 | AOI 列号（x 方向自左而右） |
| `priority` | float | 是 | [0, 1] | 指挥员优先级，越高越靠前 |
| `target_prior` | float | 是 | [0, 1] | AOI 内目标存在先验概率 |
| `target_value` | float | 是 | [0, 1] | AOI 内目标平均作战价值 |
| `target_threat` | float | 是 | [0, 1] | AOI 内目标平均威胁度 |

**AOI 四个指标的计算公式**（由态势理解模块预计算）：

```
target_prior  = clamp(confirmed_count × 0.3 + hint_count × 0.15, 0.1, 0.95)
target_value  = mean(已确认目标.value)  or 0.5
target_threat = mean(已确认目标.threat) or 0.5

V(AOI) = 0.40 × priority + 0.20 × target_value + 0.20 × target_threat + 0.20 × target_prior
```

### 5.4 平台批量配置 (platforms 对象)

模式 B 中 platforms 不是数组，而是一个按类型聚合的配置对象：

```json
{
  "UAV": {
    "count": 5,
    "pos": [150, -50],
    "sensors": ["EO", "SAR", "ESM"],
    "munitions": {"HF": 0, "RKT": 0, "GUN": 0}
  },
  "HELI": {
    "count": 2,
    "pos": [150, -50],
    "sensors": ["MMW", "EOIR"],
    "munitions": {"HF": 16, "RKT": 76, "GUN": 1200}
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `UAV` / `HELI` | object | 至少一种 | 平台类型配置 |
| `count` | int | 是 | 该类型平台数量，UAV≥1 |
| `pos` | [float, float] | 是 | 该类型平台初始位置 (km) |
| `sensors` | string[] | 是 | UAV 固定 `["EO","SAR","ESM"]`；HELI 固定 `["MMW","EOIR"]` |
| `munitions` | object | 是 | HELI: `{"HF":16,"RKT":76,"GUN":1200}`；UAV: 全 0 |

模块内部会自动展开为 `U1, U2, ... Un` 和 `H1, H2, ... Hn`。

### 5.5 AOI 执行状态 (aoi_route_state)

本模块产出，大脑保存并带回。格式固定，大脑不应修改其内容。

```
首次调用 → null
后续调用 → 带回上次 result["aoi_route_state"]
```

```json
{
  "aoi_sequence": ["A_1_5", "A_3_4", "A_5_6"],
  "current_aoi_index": 1,
  "current_aoi": "A_3_4",
  "next_aoi": "A_5_6",
  "route_status": "RUNNING"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `aoi_sequence` | string[] | AOI 执行顺序（排序结果） |
| `current_aoi_index` | int | 当前正在处理的 AOI 在序列中的下标 |
| `current_aoi` | string | 当前 AOI ID，全部完成时为 null |
| `next_aoi` | string | 下一个 AOI ID，最后一个时为 null |
| `route_status` | string | `"RUNNING"` 或 `"ALL_FINISHED"` |

### 5.6 执行反馈 (execution_feedback)

大脑在完成当前 AOI 的所有任务后，填入此对象传给本模块以推进到下一 AOI。

```json
{
  "aoi_id": "A_1_5",
  "aoi_status": "FINISHED",
  "coverage_rate": 0.85,
  "detected_targets": ["g4", "g5"],
  "destroyed_targets": ["g4"],
  "elapsed_time": 12.0
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `aoi_id` | string | 是 | 刚完成的 AOI 标识，**必须与 current_aoi 一致** |
| `aoi_status` | string | 是 | 执行结果: `"FINISHED"` / `"ABORTED"`。仅 `"FINISHED"` 触发推进 |
| `coverage_rate` | float | 否 | 侦察覆盖率 [0, 1] |
| `detected_targets` | string[] | 否 | 实际探测到的目标 tid 列表 |
| `destroyed_targets` | string[] | 否 | 实际摧毁的目标 tid 列表 |
| `elapsed_time` | float | 否 | 该 AOI 耗时（仿真时间单位） |

---

## 6. 执行输出协议 (ExecutionOrder)

**方向**: 任务分配模块 → 底层规控
**用途**: 单次求解产出的任务清单。两种输入模式产出的输出格式**完全相同**。

### 6.1 顶层结构

```json
{
  "cycle_id": 0,
  "timestamp": 0.0,
  "aoi_id": "A_3_4",
  "solve_status": "OPTIMAL",
  "objective": 961.52,
  "solve_time_ms": 111.0,
  "tasks": [ ... ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `cycle_id` | int | 分配轮次 |
| `timestamp` | float | 仿真时间戳 |
| `aoi_id` | string | 当前 AOI 标识 |
| `solve_status` | string | `OPTIMAL` / `FEASIBLE` / `TIME_LIMIT` / `INFEASIBLE` |
| `objective` | float | 目标函数值。有打击时约 800~1000，纯侦察时为负值 |
| `solve_time_ms` | float | 求解耗时 (毫秒) |
| `tasks` | array | 有序任务列表，**先侦察后打击** |

### 6.2 侦察任务 (task_type="recon")

```json
{
  "platform": "U5",
  "task_type": "recon",
  "sensor": "ESM",
  "cell": "c0",
  "role": "area_scan",
  "aoi": "A_3_4"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `platform` | string | 是 | 执行平台 PID，如 `"U5"` |
| `task_type` | string | 是 | 固定 `"recon"` |
| `sensor` | string | 是 | 使用传感器: `EO` / `SAR` / `ESM` |
| `cell` | string | 是 | 目标栅格: `c0`~`c4` |
| `role` | string | 是 | `area_scan`(区域扫描) / `esm_patrol`(ESM 巡逻) |
| `aoi` | string | 是 | 所属 AOI 标识 |

### 6.3 打击任务 (task_type="strike")

```json
{
  "platform": "H1",
  "task_type": "strike",
  "target": "g1",
  "munition": "HF",
  "qty": 2,
  "role": "striker",
  "aoi": "A_3_4"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `platform` | string | 是 | 执行平台 PID |
| `task_type` | string | 是 | 固定 `"strike"` |
| `target` | string | 是 | 目标标识 |
| `munition` | string | 是 | `HF`(地狱火) / `RKT`(火箭弹) / `GUN`(机炮) |
| `qty` | int | 是 | 发射数量。**qty=0 表示支援（不发射弹药）** |
| `role` | string | 是 | 打击角色，见下表 |
| `aoi` | string | 是 | 所属 AOI 标识 |

**打击角色 (role)**:

| role | qty | 含义 | 执行层操作 |
|------|-----|------|------------|
| `lead` | >0 | 长机主攻 | 发射 qty 发弹药 |
| `striker` | >0 | 独立打击 | 发射 qty 发弹药 |
| `wing` | >0 | 僚机攻击 | 发射 qty 发弹药 |
| `lead_support` | 0 | 长机支援 | 不发射，仅提供编队平台数 |
| `striker_support` | 0 | 打击支援 | 不发射，仅提供编队平台数 |

### 6.4 底层规控执行伪代码

```
遍历 tasks 数组（已按先侦察后打击排序）:
    if task_type == "recon":
        派发 task.platform 使用 task.sensor 前往 task.cell 侦察
    if task_type == "strike":
        if qty > 0:
            派发 task.platform 使用 task.munition 发射 qty 发，攻击 task.target
        else:
            派发 task.platform 执行支援编队机动（不发射弹药）
```

### 6.5 输出完整示例（4 种典型场景）

以下每种场景均给出执行层开发者会收到的完整 JSON 及逐条任务的解读。

---

#### 场景 A：纯侦察（无确认目标 / 无打击）

**触发条件**: 所有目标 `confirmed=false`，或当前 AOI 内无目标。

<details>
<summary>点击展开 JSON</summary>

```json
{
  "cycle_id": 0,
  "timestamp": 0.0,
  "aoi_id": "A_1_5",
  "solve_status": "OPTIMAL",
  "objective": -3.44,
  "solve_time_ms": 117,
  "tasks": [
    {
      "platform": "U1", "task_type": "recon",
      "sensor": "SAR", "cell": "c4",
      "role": "area_scan", "aoi": "A_1_5"
    },
    {
      "platform": "U2", "task_type": "recon",
      "sensor": "SAR", "cell": "c2",
      "role": "area_scan", "aoi": "A_1_5"
    },
    {
      "platform": "U3", "task_type": "recon",
      "sensor": "EO",  "cell": "c0",
      "role": "area_scan", "aoi": "A_1_5"
    },
    {
      "platform": "U3", "task_type": "recon",
      "sensor": "SAR", "cell": "c1",
      "role": "area_scan", "aoi": "A_1_5"
    },
    {
      "platform": "U4", "task_type": "recon",
      "sensor": "SAR", "cell": "c3",
      "role": "area_scan", "aoi": "A_1_5"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "SAR", "cell": "c0",
      "role": "area_scan", "aoi": "A_1_5"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c0",
      "role": "area_scan", "aoi": "A_1_5"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c1",
      "role": "area_scan", "aoi": "A_1_5"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c2",
      "role": "area_scan", "aoi": "A_1_5"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c3",
      "role": "area_scan", "aoi": "A_1_5"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c4",
      "role": "area_scan", "aoi": "A_1_5"
    }
  ]
}
```

</details>

**逐条解读**:

| 条号 | 说明 | 执行层操作 |
|------|------|-----------|
| 0 | U1 用 SAR 扫 c4 子区 | 派 U1 飞向 c4(187.5, 137.5)，SAR 宽域扫描 |
| 1 | U2 用 SAR 扫 c2 子区 | 派 U2 飞向 c2(187.5, 112.5)，SAR 宽域扫描 |
| 2 | U3 用 EO 扫 c0 巡逻区 | 派 U3 飞向 c0(225, 25)，EO 光电侦察 |
| 3 | **同 U3**，再用 SAR 扫 c1 | U3 切换 SAR 模式扫 c1(212.5, 12.5)。注意：一架 UAV 可挂 EO+SAR 双传感器 |
| 4 | U4 用 SAR 扫 c3 | 派 U4 飞向 c3(212.5, 37.5) |
| 5 | U5 用 SAR 扫 c0 | U5 SAR 扫 c0 |
| 6~10 | **同 U5**，用 ESM 扫全部 5 格 | U5 ESM 广域巡逻，覆盖 c0~c4。ESM 作用距离 100km，飞在巡逻区即可全覆 |

**关键信息**:
- `objective = -3.44` 为负值 → 纯侦察模式，没有打击收益
- U3 和 U5 各挂了两种传感器（EO+SAR 和 SAR+ESM），符合 1~2 种传感器的约束
- 5 架 UAV 全部出动，2 架 HELI 未分配（无打击任务）
- 执行层需管控传感器的开关时机：U3 先在 c0 开 EO，再在 c1 开 SAR

---

#### 场景 B：侦察 + 双机编队打击（含支援角色）

**触发条件**: 存在已确认的高价值目标，2 架 HELI 组成长机-僚机编队。

<details>
<summary>点击展开 JSON</summary>

```json
{
  "cycle_id": 2,
  "timestamp": 24.0,
  "aoi_id": "A_3_4",
  "solve_status": "OPTIMAL",
  "objective": 961.52,
  "solve_time_ms": 74,
  "tasks": [
    {
      "platform": "U1", "task_type": "recon",
      "sensor": "SAR", "cell": "c4",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U2", "task_type": "recon",
      "sensor": "SAR", "cell": "c3",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U3", "task_type": "recon",
      "sensor": "EO",  "cell": "c0",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U3", "task_type": "recon",
      "sensor": "SAR", "cell": "c1",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U4", "task_type": "recon",
      "sensor": "SAR", "cell": "c2",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "SAR", "cell": "c0",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c0",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c1",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c2",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c3",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c4",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "H1", "task_type": "strike",
      "target": "g1", "munition": "HF", "qty": 2,
      "role": "lead", "aoi": "A_3_4"
    },
    {
      "platform": "H2", "task_type": "strike",
      "target": "g1", "munition": "", "qty": 0,
      "role": "lead_support", "aoi": "A_3_4"
    }
  ]
}
```

</details>

**逐条解读（仅打击部分，侦察同场景 A）**:

| 条号 | 平台 | 弹药 | 数量 | 角色 | 执行层操作 |
|------|------|------|------|------|-----------|
| 11 | H1 | HF(地狱火) | 2 | lead(长机主攻) | **发射 2 枚地狱火导弹攻击 g1**。长机负责实际开火 |
| 12 | H2 | — | 0 | lead_support | **不发射**。僚机承担支援角色：占位、电子压制、损伤评估。提供编队平台数以满足火力约束 |

**关键信息**:
- `objective = 961.52` 为正值 → 有打击收益（λ_strike=1000 × 命中概率）
- H2 的 `qty=0` 且 `munition=""` → 这是**编队支援角色，不要发射弹药**
- 执行层需要区分 `qty>0`（实际开火）和 `qty=0`（占位支援）
- 打击 target 字段为 `"g1"`(RADAR)，执行层需要从态势数据中查找目标坐标来引导火控

---

#### 场景 C：双目标打击（含不同弹药类型）

**触发条件**: 2 个已确认目标，配置不同弹药（HF 打 RADAR，RKT 打 CP）。

<details>
<summary>点击展开 JSON</summary>

```json
{
  "cycle_id": 3,
  "timestamp": 36.0,
  "aoi_id": "A_5_6",
  "solve_status": "OPTIMAL",
  "objective": 847.14,
  "solve_time_ms": 72,
  "tasks": [
    {
      "platform": "U1", "task_type": "recon",
      "sensor": "SAR", "cell": "c4",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "U2", "task_type": "recon",
      "sensor": "SAR", "cell": "c3",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "U3", "task_type": "recon",
      "sensor": "SAR", "cell": "c2",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "U4", "task_type": "recon",
      "sensor": "EO",  "cell": "c0",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "U4", "task_type": "recon",
      "sensor": "SAR", "cell": "c0",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "SAR", "cell": "c1",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c0",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c1",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c2",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c3",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c4",
      "role": "area_scan", "aoi": "A_5_6"
    },
    {
      "platform": "H1", "task_type": "strike",
      "target": "g2", "munition": "RKT", "qty": 4,
      "role": "striker", "aoi": "A_5_6"
    },
    {
      "platform": "H2", "task_type": "strike",
      "target": "g1", "munition": "HF",  "qty": 2,
      "role": "lead", "aoi": "A_5_6"
    }
  ]
}
```

</details>

**逐条解读（打击部分）**:

| 条号 | 平台 | 目标 | 弹药 | 数量 | 角色 | 执行层操作 |
|------|------|------|------|------|------|-----------|
| 11 | H1 | g2(CP) | RKT(火箭弹) | 4 | striker | 发射 **4 枚火箭弹**攻击 g2。指挥所目标适合用火箭弹覆盖 |
| 12 | H2 | g1(RADAR) | HF(地狱火) | 2 | lead | 发射 **2 枚地狱火**攻击 g1。雷达目标需要精确制导 |

**关键信息**:
- 这次没有支援角色 → 两架 HELI 都在实际开火（`qty` 均 > 0）
- `munition` 不同：H1 用 RKT（面覆盖），H2 用 HF（精确打击）
- 执行层需要根据 `munition` 类型选择不同的火控参数和弹道
- 打击的是两个不同目标，不存在同一目标被重复分配的问题

---

#### 场景 D：求解失败 / 不可行

**触发条件**: 约束冲突（如 UAV 数量不足、弹药不够等）。

<details>
<summary>点击展开 JSON</summary>

```json
{
  "cycle_id": 1,
  "timestamp": 12.0,
  "aoi_id": "A_3_4",
  "solve_status": "INFEASIBLE",
  "objective": 0.0,
  "solve_time_ms": 45,
  "tasks": []
}
```

</details>

**执行层处理**: 检测到 `solve_status == "INFEASIBLE"` 时，`tasks` 为空数组，**不能展开任何行动**。应立即向上层（大脑）报告不可行原因，由大脑调整态势条件后重新求解。

**常见不可行原因**:

| 原因 | 大脑修复方式 |
|------|-------------|
| UAV 不足 | 增加 UAV 数量或减少栅格要求 |
| 弹药耗尽 | 等待补给或放弃低价值目标 |
| 目标全部未确认 | 先执行纯侦察，确认后再打击 |
| 全部栅格天气 w≥0.80 且无 SAR | 等待天气好转或调配 SAR 无人机 |

---

## 7. 调用流程伪代码

### 7.1 模式 A：单 AOI（一次调用）

```
大脑:
    input_json = 构造模式A输入JSON（§4）
    调用 本模块.solve(input_json)
    得到 ExecutionOrder JSON（§6）
    遍历 tasks → 下发执行单元
```

### 7.2 模式 B：多 AOI（循环调用）

```
大脑:
    input_data = {
        "aois": [AOI_A, AOI_B, AOI_C],
        "platforms": {...},
        "targets": [...],
        "aoi_route_state": null,      // ← 首次为 null
        "execution_feedback": null,   // ← 首次为 null
    }
    
    while True:
        result = 本模块.run(input_data)
        
        if result["status"] == "ALL_AOI_FINISHED":
            break   // 全部 AOI 已处理完毕
        
        // ── 获取当前 AOI 的任务清单 ──
        plan = result["current_aoi_plan"]
        for task in plan["tasks"]:
            下发执行单元
        
        // ── 等待底层规控完成该 AOI 的所有任务 ──
        等待执行完成...
        
        // ── 准备下一次调用 ──
        input_data["aoi_route_state"] = result["aoi_route_state"]  // 原样带回
        input_data["execution_feedback"] = {
            "aoi_id": plan["aoi"],            // 刚完成的 AOI ID
            "aoi_status": "FINISHED",         // 完成标志
            "coverage_rate": 0.85,
            "detected_targets": [...],
            "destroyed_targets": [...],
            "elapsed_time": 12.0
        }
        // 更新弹药、目标状态等（大脑维护全局状态）
        input_data["targets"] = [...]   // 标记已摧毁目标 alive=false
        input_data["platforms"]["HELI"]["munitions"] = {...}  // 扣除弹药
        
```

---

## 8. 完整示例

### 8.1 模式 A：单 AOI 完整示例

<details>
<summary>点击展开 —— 态势输入 JSON</summary>

```json
{
  "aoi_id": "A_3_4",
  "staging_position": [150.0, -50.0],
  "grids": [
    {
      "cell_id": "c0", "center": [175.0, 125.0], "size": [50.0, 50.0],
      "weather_w": 0.20, "terrain_level": 0, "target_prior": 0.10, "covered": false
    },
    {
      "cell_id": "c1", "center": [162.5, 112.5], "size": [25.0, 25.0],
      "weather_w": 0.15, "terrain_level": 0, "target_prior": 0.25, "covered": false
    },
    {
      "cell_id": "c2", "center": [187.5, 112.5], "size": [25.0, 25.0],
      "weather_w": 0.40, "terrain_level": 0, "target_prior": 0.25, "covered": false
    },
    {
      "cell_id": "c3", "center": [162.5, 137.5], "size": [25.0, 25.0],
      "weather_w": 0.55, "terrain_level": 0, "target_prior": 0.25, "covered": false
    },
    {
      "cell_id": "c4", "center": [187.5, 137.5], "size": [25.0, 25.0],
      "weather_w": 0.72, "terrain_level": 0, "target_prior": 0.25, "covered": false
    }
  ],
  "targets": [
    {
      "tid": "g1", "type": "RADAR", "pos": [162.0, 112.0],
      "confirmed": true, "alive": true, "value": 0.97, "threat": 0.91
    },
    {
      "tid": "g2", "type": "CP", "pos": [188.0, 112.0],
      "confirmed": true, "alive": true, "value": 0.85, "threat": 0.60
    }
  ],
  "platforms": [
    {
      "pid": "U1", "type": "UAV", "pos": [150.0, -50.0],
      "sensors": ["EO", "SAR", "ESM"],
      "munitions": {"HF": 0, "RKT": 0, "GUN": 0}, "alt": 2.0, "lost": false
    },
    {
      "pid": "U2", "type": "UAV", "pos": [150.0, -50.0],
      "sensors": ["EO", "SAR", "ESM"],
      "munitions": {"HF": 0, "RKT": 0, "GUN": 0}, "alt": 2.0, "lost": false
    },
    {
      "pid": "U3", "type": "UAV", "pos": [150.0, -50.0],
      "sensors": ["EO", "SAR", "ESM"],
      "munitions": {"HF": 0, "RKT": 0, "GUN": 0}, "alt": 2.0, "lost": false
    },
    {
      "pid": "U4", "type": "UAV", "pos": [150.0, -50.0],
      "sensors": ["EO", "SAR", "ESM"],
      "munitions": {"HF": 0, "RKT": 0, "GUN": 0}, "alt": 2.0, "lost": false
    },
    {
      "pid": "U5", "type": "UAV", "pos": [150.0, -50.0],
      "sensors": ["EO", "SAR", "ESM"],
      "munitions": {"HF": 0, "RKT": 0, "GUN": 0}, "alt": 2.0, "lost": false
    },
    {
      "pid": "H1", "type": "HELI", "pos": [150.0, -50.0],
      "sensors": ["MMW", "EOIR"],
      "munitions": {"HF": 16, "RKT": 76, "GUN": 1200}, "alt": 3.0, "lost": false
    },
    {
      "pid": "H2", "type": "HELI", "pos": [150.0, -50.0],
      "sensors": ["MMW", "EOIR"],
      "munitions": {"HF": 16, "RKT": 76, "GUN": 1200}, "alt": 3.0, "lost": false
    }
  ],
  "sensors": [
    {"name": "EO",  "P0": 0.85, "R": 15.0,  "weather_sensitive": true},
    {"name": "SAR", "P0": 0.90, "R": 50.0,  "weather_sensitive": false},
    {"name": "ESM", "P0": 0.80, "R": 100.0, "weather_sensitive": false}
  ],
  "commander_aoi": ["A_3_4"]
}
```

</details>

<details>
<summary>点击展开 —— 执行输出 JSON</summary>

```json
{
  "cycle_id": 0,
  "timestamp": 0.0,
  "aoi_id": "A_3_4",
  "solve_status": "OPTIMAL",
  "objective": 961.52,
  "solve_time_ms": 111.0,
  "tasks": [
    {
      "platform": "U1", "task_type": "recon",
      "sensor": "SAR", "cell": "c4",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U2", "task_type": "recon",
      "sensor": "EO", "cell": "c0",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U2", "task_type": "recon",
      "sensor": "SAR", "cell": "c3",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U3", "task_type": "recon",
      "sensor": "SAR", "cell": "c0",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U4", "task_type": "recon",
      "sensor": "SAR", "cell": "c2",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "SAR", "cell": "c1",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c0",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c1",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c2",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c3",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "U5", "task_type": "recon",
      "sensor": "ESM", "cell": "c4",
      "role": "area_scan", "aoi": "A_3_4"
    },
    {
      "platform": "H1", "task_type": "strike",
      "target": "g1", "munition": "HF", "qty": 2,
      "role": "striker", "aoi": "A_3_4"
    },
    {
      "platform": "H2", "task_type": "strike",
      "target": "g1", "munition": "", "qty": 0,
      "role": "lead_support", "aoi": "A_3_4"
    }
  ]
}
```

</details>

### 8.2 模式 B：多 AOI 完整示例

<details>
<summary>点击展开 —— 首次调用输入 JSON</summary>

```json
{
  "aois": [
    {
      "id": "A_5_6",
      "row": 5, "col": 6,
      "priority": 0.8,
      "target_prior": 0.70,
      "target_value": 0.97,
      "target_threat": 0.91
    },
    {
      "id": "A_3_4",
      "row": 3, "col": 4,
      "priority": 0.6,
      "target_prior": 0.55,
      "target_value": 0.775,
      "target_threat": 0.55
    },
    {
      "id": "A_1_5",
      "row": 1, "col": 5,
      "priority": 0.9,
      "target_prior": 0.40,
      "target_value": 0.675,
      "target_threat": 0.475
    }
  ],
  "platforms": {
    "UAV": {
      "count": 5,
      "pos": [150, -50],
      "sensors": ["EO", "SAR", "ESM"],
      "munitions": {"HF": 0, "RKT": 0, "GUN": 0}
    },
    "HELI": {
      "count": 2,
      "pos": [150, -50],
      "sensors": ["MMW", "EOIR"],
      "munitions": {"HF": 16, "RKT": 76, "GUN": 1200}
    }
  },
  "targets": [
    {
      "tid": "g1", "type": "RADAR", "pos": [265, 238],
      "confirmed": true, "alive": true, "value": 0.97, "threat": 0.91
    },
    {
      "tid": "g2", "type": "CP", "pos": [175, 125],
      "confirmed": true, "alive": true, "value": 0.85, "threat": 0.60
    },
    {
      "tid": "g3", "type": "AV", "pos": [185, 135],
      "confirmed": true, "alive": true, "value": 0.70, "threat": 0.50
    },
    {
      "tid": "g4", "type": "AV", "pos": [225, 25],
      "confirmed": true, "alive": true, "value": 0.70, "threat": 0.50
    },
    {
      "tid": "g5", "type": "AV", "pos": [235, 35],
      "confirmed": true, "alive": true, "value": 0.65, "threat": 0.45
    }
  ],
  "staging_position": [150, -50],
  "grid_weather": {"c0": 0.20, "c1": 0.25, "c2": 0.40, "c3": 0.55, "c4": 0.72},
  "aoi_route_state": null,
  "execution_feedback": null,
  "cycle_id": 0
}
```

</details>

<details>
<summary>点击展开 —— 首次调用输出 JSON（含 AOI 排序结果）</summary>

```json
{
  "status": "AOI_PLAN_READY",
  "aoi_route_state": {
    "aoi_sequence": ["A_1_5", "A_3_4", "A_5_6"],
    "current_aoi_index": 0,
    "current_aoi": "A_1_5",
    "next_aoi": "A_3_4",
    "route_status": "RUNNING"
  },
  "current_aoi_plan": {
    "aoi": "A_1_5",
    "tasks": [
      {
        "platform": "U1", "task_type": "recon",
        "sensor": "SAR", "cell": "c4",
        "role": "area_scan", "aoi": "A_1_5"
      },
      {
        "platform": "U2", "task_type": "recon",
        "sensor": "SAR", "cell": "c2",
        "role": "area_scan", "aoi": "A_1_5"
      },
      {
        "platform": "U3", "task_type": "recon",
        "sensor": "EO", "cell": "c0",
        "role": "area_scan", "aoi": "A_1_5"
      },
      {
        "platform": "U3", "task_type": "recon",
        "sensor": "SAR", "cell": "c1",
        "role": "area_scan", "aoi": "A_1_5"
      },
      {
        "platform": "U4", "task_type": "recon",
        "sensor": "SAR", "cell": "c3",
        "role": "area_scan", "aoi": "A_1_5"
      },
      {
        "platform": "U5", "task_type": "recon",
        "sensor": "SAR", "cell": "c0",
        "role": "area_scan", "aoi": "A_1_5"
      },
      {
        "platform": "U5", "task_type": "recon",
        "sensor": "ESM", "cell": "c0",
        "role": "area_scan", "aoi": "A_1_5"
      },
      {
        "platform": "U5", "task_type": "recon",
        "sensor": "ESM", "cell": "c1",
        "role": "area_scan", "aoi": "A_1_5"
      },
      {
        "platform": "U5", "task_type": "recon",
        "sensor": "ESM", "cell": "c2",
        "role": "area_scan", "aoi": "A_1_5"
      },
      {
        "platform": "U5", "task_type": "recon",
        "sensor": "ESM", "cell": "c3",
        "role": "area_scan", "aoi": "A_1_5"
      },
      {
        "platform": "U5", "task_type": "recon",
        "sensor": "ESM", "cell": "c4",
        "role": "area_scan", "aoi": "A_1_5"
      }
    ],
    "solve_status": "OPTIMAL",
    "objective": -3.44,
    "solve_time_ms": 117
  }
}
```

</details>

<details>
<summary>点击展开 —— 第二次调用输入 JSON（带回 aoi_route_state + execution_feedback）</summary>

```json
{
  "aois": [
    {"id": "A_5_6", "row": 5, "col": 6, "priority": 0.8,
     "target_prior": 0.70, "target_value": 0.97, "target_threat": 0.91},
    {"id": "A_3_4", "row": 3, "col": 4, "priority": 0.6,
     "target_prior": 0.55, "target_value": 0.775, "target_threat": 0.55},
    {"id": "A_1_5", "row": 1, "col": 5, "priority": 0.9,
     "target_prior": 0.40, "target_value": 0.675, "target_threat": 0.475}
  ],
  "platforms": {
    "UAV": {
      "count": 5, "pos": [150, -50],
      "sensors": ["EO", "SAR", "ESM"],
      "munitions": {"HF": 0, "RKT": 0, "GUN": 0}
    },
    "HELI": {
      "count": 2, "pos": [150, -50],
      "sensors": ["MMW", "EOIR"],
      "munitions": {"HF": 16, "RKT": 76, "GUN": 1200}
    }
  },
  "targets": [
    {"tid": "g1", "type": "RADAR", "pos": [265, 238],
     "confirmed": true, "alive": true, "value": 0.97, "threat": 0.91},
    {"tid": "g2", "type": "CP", "pos": [175, 125],
     "confirmed": true, "alive": true, "value": 0.85, "threat": 0.60},
    {"tid": "g3", "type": "AV", "pos": [185, 135],
     "confirmed": true, "alive": true, "value": 0.70, "threat": 0.50},
    {"tid": "g4", "type": "AV", "pos": [225, 25],
     "confirmed": true, "alive": false, "value": 0.70, "threat": 0.50},
    {"tid": "g5", "type": "AV", "pos": [235, 35],
     "confirmed": true, "alive": true, "value": 0.65, "threat": 0.45}
  ],
  "staging_position": [150, -50],
  "grid_weather": {"c0": 0.20, "c1": 0.25, "c2": 0.40, "c3": 0.55, "c4": 0.72},
  "aoi_route_state": {
    "aoi_sequence": ["A_1_5", "A_3_4", "A_5_6"],
    "current_aoi_index": 0,
    "current_aoi": "A_1_5",
    "next_aoi": "A_3_4",
    "route_status": "RUNNING"
  },
  "execution_feedback": {
    "aoi_id": "A_1_5",
    "aoi_status": "FINISHED",
    "coverage_rate": 0.85,
    "detected_targets": ["g4", "g5"],
    "destroyed_targets": ["g4"],
    "elapsed_time": 12.0
  },
  "cycle_id": 1
}
```

> 注意变化：`g4` 的 `alive` 已改为 `false`（被摧毁），`execution_feedback` 填入完成信息。模块检测到 `aoi_status="FINISHED"` 且 `aoi_id="A_1_5"` 与 `current_aoi` 一致，自动推进到 `A_3_4`。

</details>

---

## 9. 关键约束速查表

大脑构造输入时需要保证以下条件，否则求解器可能返回 `INFEASIBLE`。

| 条件 | 说明 |
|------|------|
| UAV ≥ 1 架 | 至少需要 1 架 UAV 执行 ESM 广域覆盖 |
| 传感器参数完整 | sensor_params 中至少包含 EO, SAR, ESM 三种 |
| EO 天气门限 | `weather_w ≥ 0.80` 的栅格上 EO 被自动禁用，不影响可行性 |
| 目标 confirmed=true | `confirmed=false` 的目标仅参与侦察价值计算，不参与打击 |
| 目标 alive=true | `alive=false` 的目标完全不参与优化 |
| 弹药足够 | 打击任务所需的弹药量不能超过平台携带量 |
| 平台 lost=false | `lost=true` 的平台自动排除 |
| 多 AOI 时 aois 数量 | 2~4 个，全排列枚举（最多 4! = 24 种） |

---

## 10. 跨语言对接

接口协议是纯 JSON，与编程语言无关：

```
Python:   json.dumps(obj)   /  json.loads(str)
C++:      nlohmann::json    /  rapidjson
Java:     Jackson           /  Gson
Go:       encoding/json
JavaScript: JSON.stringify  /  JSON.parse
```

**外部模块开发者只需根据本文档的字段表构造/解析 JSON**，无需依赖本项目的任何 Python 文件。
