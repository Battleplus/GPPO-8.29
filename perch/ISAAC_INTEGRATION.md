# Perch / Brain 与 Isaac 联合使用

## 自动初始化启动

生产入口使用：

```bash
python brain/isaac_main.py
```

或在代码中显式构造运行环境：

```python
from brain.integration import IsaacAirCombatEnvironment

runtime = IsaacAirCombatEnvironment(headless=False)
brain = MissionBrain(
    context,
    MILPTaskAllocator(),
    MPPIFormationPlanner(),
    PositionSelector(),
    environment=runtime,
)
state = brain.start()
```

`brain.start()` 的顺序为：

1. 创建 `SimulationApp`；
2. 创建 USD stage 和 `AirCombatSceneState`；
3. 启动 Isaac timeline；
4. 从场景生成 `context.agents`、目标、地形、障碍物和传感器状态；
5. 调用侦察 MILP 分配；
6. 调用 MPPI 生成山地级侦察航路；
7. 停在 `RECON_PLAN_READY` 等待执行系统下发航路。

普通 Python 单元测试仍可不传 `environment`，只运行规划状态机。

## 侦察分配输入

`sync_context_from_air_combat_scene()` 自动从环境提取：

- 蓝方平台 ID、UAV/HELI 类型、当前位置、速度和航向；
- 平台传感器和弹药；
- 红方固定/机动目标、当前位置、存活和确认状态；
- 地形函数、地图尺度和天气；
- 山体、树木、岩石及当前障碍物接触。

MILP 使用公里坐标；Perch/MPPI 使用相同对象保存的 Isaac scene 坐标。

## 障碍物职责

- MPPI 只使用已有山地级障碍物规划全局航路。
- 新探测到的树木、岩石或局部障碍只更新环境快照和 Perch 阵位安全约束。
- `brain.step_environment()` 不会因为新障碍物自动重新调用 MPPI。

## 飞机损失

侦察 UAV 损失时：

```python
result = brain.report_platform_loss("Blue_Quad_Recon_1")
```

流程为：

```text
UAV_DAMAGE
  -> ppo_allocation.ReallocationService
  -> 搜索子区域局部重分配
  -> 将 PPO 内部 U0~U3 转回 Isaac 平台 ID
  -> 为新区域分配重新生成山地航路
  -> RECON_PLAN_READY
```

PPO 模型通过构造参数或环境变量配置：

```text
QL_PPO_ALLOCATION_MODEL=<model.zip>
```

当前 PPO 模型只覆盖四架子区域搜索 UAV；直升机损失不经过该 PPO。

## 攻击阵位两阶段选择

`PositionSelector` 默认执行以下顺序：

```text
任务分配
  -> perch.SituationUnderstanding 将实时 world_state 转成态势文字
  -> perch.AttackRegionSelector
  -> 本地专家知识检索 + 大模型输出 GeoJSON 候选区域
  -> 经纬度/局部坐标转换为 polygon_scene
  -> 按大模型 score 从高到低逐区域运行 FREA
  -> 输出所选区域内的具体攻击阵位点
```

大模型配置全部属于 `perch`，不依赖外部推荐目录：

```text
PERCH_LLM_PROVIDER=openai            # 或 local
PERCH_OPENAI_API_KEY=<key>
PERCH_OPENAI_BASE_URL=https://api.deepseek.com
PERCH_OPENAI_MODEL=deepseek-chat
PERCH_LLM_TIMEOUT_S=30
```

本地 Ollama/vLLM 可配置 `PERCH_LOCAL_BASE_URL` 和
`PERCH_LOCAL_MODEL`。当模型不可用时会生成带有
`source=perch:local_fallback` 的规则区域，再在该区域内选点。

`world_state.attack_region_mode` 可设为 `llm`（默认）、`demo` 或
`disabled`。`world_state.attack_region_strict` 默认为 `true`；全部推荐区域
均无可行点时返回失败，不会静默选择区域外阵位。只有显式设为 `false`
才允许全局 FREA 降级。

每次区域选择后，可直接检查以下字段审计模型输入和 RAG 召回结果：

```python
world_state["attack_region_situations"][target_id]       # 传给大模型的态势文字
world_state["attack_region_knowledge_sources"][target_id]  # 实际召回的知识文档
world_state["attack_regions"][target_id]                 # 大模型区域及坐标转换结果
```

态势文字由任务分配、平台位置和速度、目标类型和运动、实时武器包线、
其他威胁、地形障碍、天气、照射组以及 FREA 硬约束自动生成。实时武器
包线同时供大模型和 FREA 使用，优先级高于专家文档中的典型射程。

## 可解释输出

每个 `Position.metadata` 包含：

- 武器包线与制导方式；
- AGL、地形高度、目标距离和 LOS；
- 最近障碍物与净空；
- 暴露、射程偏差、进场代价；
- 各项硬约束；
- 中文选择说明。

每条独立攻击航路保存：

```text
platform_id -> target_id -> position_id -> route_index
```
