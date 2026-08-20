# 任务重分配模块 — C++ 对接接口

## 目标

该模块接收预分配结果和事件信息（这两个模块用json的格式，填到 ppo_allocation/cpp_interface_example.json 中），执行任务重分配，并输出 JSON 结果。
C++ 侧可通过请求 JSON 调用它，得到重分配结果。 


## 运行环境

- Python: 3.9+
- 依赖包: sb3-contrib, gymnasium, numpy
- 模型文件: ppo_allocation/results/models/run_20260605_210049/maskable_ppo_uav_task_allocation.zip

---

## 推荐调用方式

C++ 侧生成一个请求 JSON，然后调用桥接脚本；脚本返回 JSON 响应，并把结果写到输出文件。 

### 请求 JSON 格式（ppo_allocation/cpp_interface_example.json）

```json
{
  "model_path": "ppo_allocation/results/models/run_20260605_210049/maskable_ppo_uav_task_allocation.zip",
  "preallocation_path": "ppo_allocation/scenarios/output_template.json",
  "event": {
    "event_type": "UAV_DAMAGE",
    "uav_id": 1
  },
  "output_path": "ppo_allocation/results/cpp_interface_output.json",
  "deterministic": true
}
```


### C++ 调用步骤

1. 生成 ppo_allocation/cpp_interface_example.json
2. 调用：

```powershell
conda activate uav_ppo
python ppo_allocation/cpp_bridge.py --request-file ppo_allocation/cpp_interface_example.json
```

3. 读取 stdout 中的 JSON 响应
4. 也可以读取 output_path 指向的结果文件

### C++ 伪代码

```cpp
std::string cmd = "python ppo_allocation/cpp_bridge.py --request-file ppo_allocation/cpp_interface_example.json";
std::string output = exec(cmd);
// 解析 output
```

### 请求字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `model_path` | 是 | PPO 模型 .zip 路径 |
| `preallocation_path` | 二选一 | 预分配模块输出 JSON 文件路径 |
| `preallocation_json` | 二选一 | 预分配内容对象或 JSON 字符串 |
| `event` | 二选一 | 事件对象或 JSON 字符串 |
| `event_path` | 二选一 | 事件 JSON 文件路径 |
| `targets_extra` | 否 | 目标补充信息，仅 `TARGET_DISCOVERED` / `TARGET_DESTROYED` 时需要 |
| `cell_to_region` | 否 | cell 到 region 的映射，默认 `c1→R0, c2→R1, c3→R2, c4→R3` |
| `output_path` | 否 | 输出文件路径，默认写到当前目录的 `reallocation_result.json` |
| `deterministic` | 否 | 是否采用确定性推理，默认 `true` |

### 响应字段说明

| 字段 | 说明 |
|------|------|
| `success` | 是否成功 |
| `message` | 成功/失败说明 |
| `output_path` | 输出 JSON 文件的绝对路径 |
| `result` | 重分配结果对象 |
| `error` | 失败时的错误信息 |

---

## 事件格式

支持以下 4 种事件：

```json
{"event_type": "UAV_DAMAGE", "uav_id": 1}
{"event_type": "REGION_VACANCY", "region_id": 3}
{"event_type": "TARGET_DISCOVERED", "uav_id": 0, "target_id": 2}
{"event_type": "TARGET_DESTROYED", "target_id": 0}
```

---

## 输出格式

```json
{
  "aoi_id": "A_3_4",
  "event": "U1损毁，其1个区域出现空缺",
  "region_assignments": {
    "R0": "U0",
    "R1": "U3",
    "R2": "U2",
    "R3": "U3"
  },
  "uav_tasks": {
    "U0": {
      "alive": true,
      "task": "SEARCH",
      "sensor": "EO",
      "regions": ["R0"],
      "target_id": null,
      "target_points": [[12.5, 37.5]]
    }
  },
  "action_detail": "Region   PPO决策         变化\n  R0    KEEP        (不变)\n...",
  "repair_log": [],
  "snapshot": {
    "regions": {},
    "uavs": {},
    "targets": {},
    "event": {}
  }
}
```

## 关键概念

- `event`：事件信息
- `preallocation_path`：预分配结果文件
- `output_path`：结果输出文件
- `result`：重分配结果
- `success`：是否成功
