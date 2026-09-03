# GPPO-8.29：动态扰动下的多无人机任务分配

本项目研究随机事件、资源失效和弱通信条件下的多无人机动态任务分配，对比两种共享同一训练与评估合同的策略：

- **PPO-MLP**：将规范化图状态展平后输入 MLP；
- **GPPO-Adaptive**：使用异构图消息传递和自适应门控进行关系感知决策。

项目已经完成两模型 minimum-validation 的训练与证据闭环，并增加了面向多任务变化、连续事件、突发失效、临时变更和信息不完整的探索性极端压力测试。

当前证据不支持“GPPO 在所有场景全面优于 PPO”。更准确的结论是：

> GPPO-Adaptive 在搜索/跟踪资源强争用、需要等待未来资源释放的场景中具有研究价值；PPO-MLP 在固定小图、持续事件风暴和严格实时约束下仍是更快、更稳定的基线。

## 当前状态

| 项目 | 状态 |
|---|---|
| 正式模型 | PPO-MLP、GPPO-Adaptive |
| 训练 seeds | 1101、2202、3303 |
| 训练预算 | 每个 run 50,000 accepted decision steps |
| 正式训练 | 6/6 runs，共 300,000 steps |
| Checkpoints | 12/12，每个 run 固定保存 25k、50k |
| 正式评估 | 固定使用六个 50k checkpoints，不做 checkpoint selection |
| Held-out bank | 100 cases：Single、Sequential、Overlap、Burst、Unseen 各 20 |
| Required tests | 130/130 PASS |
| P0 Gate 证据 | `training_allowed=true`、`violations=[]` |
| 极端压力测试 | 7 类场景、42 条事件带、420 个回合 |
| 极端回合完整性 | 420/420 正常结束，reward invariant 全部通过 |
| 世界模型整合 | 已形成工程实施方案；当前发布代码尚未实现，不能作为现有能力声明 |

> 说明：仓库包含训练、评估和压力测试的机器可读证据，但不提交模型 checkpoint 二进制文件。复跑 checkpoint 相关实验时，需要另外提供与 manifest 中 SHA-256 匹配的六个 50k checkpoint。

## 系统流程

```text
Truth Event
    ↓
弱通信 Observation（延迟、丢包、乱序、重复）
    ↓
Confirmation State Machine
    ↓
Belief / 异构任务图更新
    ↓
PPO-MLP 或 GPPO-Adaptive 推理
    ↓
graph/action version 检查
    ↓
Command → ACK → Lease → Fencing
    ↓
局部重新分配与事件恢复
```

当前事件类型包括：

- `UAV_DAMAGE`：UAV 损毁并释放其原搜索任务；
- `TARGET_DISCOVERED`：搜索 UAV 转入跟踪，原区域进入待分配；
- `TARGET_DESTROYED`：跟踪资源释放；
- `REGION_VACANCY`：搜索区域产生空缺。

系统支持 Single、Sequential、Overlap 和 Burst 四类事件到达模式，并通过 graph/action version、ACK、lease 和 fencing token 防止过期动作、重复持有者和旧 ACK 复活。

## 主要实验结果

### 正式 minimum-validation

- PPO 与 GPPO 的总体成功率、合法覆盖率和 episode return 接近，没有形成稳定的全面胜负；
- PPO-MLP 平均推理约 `2.51 ms`，归一化距离更低；
- GPPO-Adaptive 平均推理约 `17.07 ms`，计算代价明显更高；
- GPPO 在 Sequential、Overlap、Burst 中出现部分正向趋势，但三个训练 seed 不足以支持普遍优越性声明；
- Test-Unseen 没有证明前馈图策略能够自动解决信息延迟或部分可观测问题。

### 极端多事件压力测试

七类探索性场景包括：

1. `atomic_triple_shock`：多个扰动同时到达；
2. `resource_collapse`：多 UAV/资源连续失效；
3. `tracking_saturation_release`：跟踪资源饱和并等待未来释放；
4. `out_of_order_reports`：因果报告乱序；
5. `long_blind_burst`：长时间信息盲区；
6. `task_churn`：任务新增、取消和重新出现；
7. `event_storm_8`：连续八事件风暴。

GPPO 最明确的收益出现在 `tracking_saturation_release`：相对 PPO，平均恢复延迟减少约 `0.678`，累计空缺减少约 `12.817`，episode return 增加约 `0.610`。代价是归一化距离增加约 `0.024`，推理慢约 `12.37 ms`。

在 `event_storm_8` 和 `task_churn` 中，GPPO 没有稳定超过 PPO。乱序报告下两个前馈模型基本打平，说明信息不完整需要循环记忆、belief state 或显式状态估计。

详细材料：

- [中文研究结论](experiments/extreme_scenarios/CONCLUSION_ZH.md)
- [向学姐汇报文档](experiments/extreme_scenarios/SENIOR_BRIEFING_ZH.md)
- [世界模型与 GPPO 输入整合实施规划](docs/WORLD_MODEL_GPPO_INTEGRATION_PLAN_ZH.md)
- [极端场景结果解读](experiments/extreme_scenarios/results_20260827/INTERPRETATION.md)
- [完整场景报告](experiments/extreme_scenarios/results_20260827/REPORT.md)
- [机器可读运行摘要](experiments/extreme_scenarios/results_20260827/run_summary.json)
- [SHA-256 inventory](experiments/extreme_scenarios/results_20260827/sha256_inventory.json)

## 动态扰动能力边界

当前已经实现：

- 连续多事件和原子 burst；
- 弱通信延迟、乱序、重复和确认流程；
- UAV 损毁后的节点/边更新与局部重分配；
- 推理期间新事件到达时拒绝旧动作并重新决策；
- 命令 ACK、区域 lease、fencing token 和唯一执行者约束；
- 42 条固定极端事件带的确定性回放。

尚未完整实现：

- 任务执行到一半时的显式暂停、抢占、迁移和恢复；
- 连续任务进度、剩余工作量和进度保留；
- 能耗不足、强制返航和任务移交；
- 信息年龄、TTL 和报告置信度进入策略图特征；
- 8/16/32 UAV 等动态图规模与拓扑泛化。

因此，当前系统能处理“决策过程中发生新事件”和“事件后局部重分配”，但还不能宣称已经闭合完整的执行中抢占式动态任务分配。

## 项目结构

```text
GPPO-8.29/
├─ ppo_allocation/       PPO/GPPO 环境、图构造、训练、评估与 Phase J
├─ event_runtime/        事件观测、确认、队列、并发、ACK/lease/fencing
├─ brain/                任务编排、Mission FSM 与执行适配器
├─ milp/                 任务分配优化模块
├─ mppi/                 航迹/运动规划模块
├─ search_planner/       搜索规划与任务演示
├─ scenes/               Isaac Sim 场景与平台模型
├─ sensors/              EO/SAR/跟踪与天气影响模型
├─ weapons/              武器与打击行为模型
├─ experiments/          独立实验、结论和机器可读证据
├─ configs/              冻结协议和场景配置
├─ handoff/              Gate、provenance、决策记录与复现资料
└─ scripts/              Gate、压力测试和审计脚本
```

## 环境安装

minimum-validation 的复现环境：

- Python `3.11.5`；
- NumPy `1.26.4`；
- PyTorch `2.7.1`；
- Gymnasium `1.2.3`；
- Stable-Baselines3 / sb3-contrib `2.8.0`。

Windows PowerShell 示例：

```powershell
git clone https://github.com/Battleplus/GPPO-8.29.git
Set-Location GPPO-8.29

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r ppo_allocation\requirements-random-event-lock.txt
```

若需要 MILP、Perch 或完整任务编排模块，再安装根目录依赖：

```powershell
python -m pip install -r requirements.txt
```

Isaac Sim 场景需要单独安装与当前机器匹配的 NVIDIA Isaac Sim，不能仅通过上述 pip 依赖完成配置。

## 快速验证

运行随机事件核心测试：

```powershell
Set-Location ppo_allocation
python -m unittest discover -s tests_random_event -v
```

从仓库根目录只读查看已封存的 P0 Gate 状态：

```powershell
Set-Location ..
python -c "import json; d=json.load(open('handoff/P0_GATE.json', encoding='utf-8')); print(d.get('training_allowed'), d.get('violations'))"
```

`handoff/P0_GATE.json` 是原 minimum-validation source/evidence 链生成的归档证据。当前仓库首页 README 的发布提交不是新的正式训练 source，不应重新运行 Gate builder，也不应据此启动或重做正式训练。

## 复跑极端场景

需要一个包含冻结 manifest 和六个已验证 50k checkpoints 的本地 artifact 根目录：

```powershell
python scripts\run_extreme_scenarios.py `
  --checkpoint-root E:\path\to\ppo_allocation `
  --output-dir E:\new\empty\extreme_results
```

脚本会拒绝复用非空输出目录，并在回放前校验六个 checkpoint 的 SHA-256。

审计“推理完成后、动作提交前发生新事件”时的旧动作拒绝：

```powershell
python scripts\audit_stale_decision_race.py `
  --checkpoint-root E:\path\to\ppo_allocation `
  --result-root E:\path\to\extreme_results
```

## 证据与阅读入口

- [世界模型资料与推荐阅读顺序](docs/world-model/README.md)
- [2026-09-10 世界模型最小闭环验收目标](docs/world-model/acceptance/2026-09-10_世界模型最小闭环验收目标.md)
- [世界模型整合实施计划 DOCX](docs/world-model/source/世界模型整合实施计划_2026-08-30.docx)
- [Minimum-validation 主合同](MINIMUM_VALIDATION_START_HERE.md)
- [P0 Gate 机器可读证据](handoff/P0_GATE.json)
- [训练证据目录](ppo_allocation/results/random_event/minimum_validation_50k_2afa8ec/training_evidence/)
- [固定评估证据目录](ppo_allocation/results/random_event/minimum_validation_50k_2afa8ec/evaluation_evidence/)
- [极端压力测试说明](experiments/extreme_scenarios/README.md)
- [中文最终结论](experiments/extreme_scenarios/CONCLUSION_ZH.md)

历史 handoff 文档中可能保留旧的 300k、GPPO-NoGate 或 GPPO-8.20 方案，它们仅用于追溯设计过程。当前有效的 minimum-validation 合同以本 README、`MINIMUM_VALIDATION_START_HERE.md` 中的两模型 50k 方案，以及对应机器可读 evidence 为准。

## 研究与解释边界

- 极端场景是在查看正式结果后设计的 post-hoc development bank，不是新的正式 held-out test；
- 不应声称 GPPO 已全面超过 PPO；
- 成功率在当前 action mask 与恢复机制下接近天花板，应同时报告累计空缺、恢复延迟、距离、负载和端到端时延；
- 新算法在这批极端场景上开发后，必须使用新冻结、未见过的 `Extreme-V2` hidden bank 做一次性验证；
- 本仓库用于研究与仿真验证，不代表真实无人系统部署安全认证。

## 世界模型整合规划（设计阶段，尚未实现）

《世界模型整合实施计划》的目标，是在不破坏当前事件确认与动作安全链路的前提下，为 PPO/GPPO 增加“历史记忆、未来事件风险和不确定性”上下文。方案参考 EAWM（Event-Aware World Model）的事件辅助表征思想，但不会直接照搬面向图像环境的生成式架构。

> 核心原则：世界模型只提供预测和辅助表征，不直接修改 `belief`、事件确认状态、动作 `mask` 或 graph/action version，也不拥有动作执行权限。模型给“风险”，状态机给“事实”。

### 推荐的最小闭环

```text
最近 8–16 个决策/观测时刻
    ↓
可见 belief/observation + 历史动作/执行结果 + 可靠性/版本信息
    ↓
结构化快照编码器 + 单层 GRU/TCN
    ↓
潜在状态 z_t + 事件风险 + 不确定性 + 下一状态增量
    ↓
PPO/GPPO 策略上下文，或 Router/Planner 路由依据
    ↓
原有 mask / confirmation / stale guard / ACK / lease / fencing 安全执行链
```

首版建议采用结构化事件感知状态模型（EASM），而不是直接引入大型像素级 Dreamer：复用或冻结当前图编码器，将短历史输入单层 GRU/TCN，先完成离线训练、概率校准和影子运行，再以可关闭的上下文接口接入 PPO 与 GPPO。默认上下文为全零，关闭世界模型时必须与现有模型保持兼容。

### 输入白名单、离线标签与输出

| 类别 | 建议内容 | 约束/用途 |
|---|---|---|
| 历史窗口 | 最近 8–16 个决策或观测时刻，变长序列使用 mask | 同时覆盖短时事件链与通信延迟，并控制推理成本 |
| 状态与图 | 当前 `belief` 图、候选 UAV–Region 边、动作 mask、上一步动作和执行结果 | 只读取决策时刻已经可见的信息 |
| 时间与版本 | `decision_time`、`received_time`、`state_version`、`graph_version`、`action_version` | 用于时序对齐和过期动作审计 |
| 观测可靠性 | `confidence`、`source`、`age`、乱序/重复标记、最近确认状态 | 区分“环境风险”和“信息看不清” |
| 潜在状态 | `z_t`，建议 64–96 维 | 为策略提供带记忆的紧凑上下文 |
| 事件预测 | 未来 H 步多标签 `event_within_H`、可选 `time_to_event`、校准概率 | 支持并发事件和提前预警，不只预测单一事件类别 |
| 状态预测 | 下一步覆盖、负载、距离、可用 UAV 数等 `state_delta` | 让潜在状态保留任务分配动力学，而非只记事件标签 |
| 策略上下文 | `[z_t, event_risk, uncertainty, obs_age]` | 在 Actor/Critic 同一位置注入，PPO 与 GPPO 使用相同接口 |

训练阶段可以用未来真值生成事件和状态监督标签，但在线特征导出器必须显式拒绝以下 `truth-only` 信息：尚未到达的报文、未来图/未来 mask、最优规划器答案、仅仿真器内部可见的丢包决定，以及未通过确认流程的真实事件发生时间。数据必须按完整 event tape/scenario/seed 划分训练、验证和测试，不能把同一事件带的相邻 transition 分到不同集合；开发集与一次性 `Extreme-V2` 隐藏集也必须隔离。

### 事件契约与安全边界

现有 `TARGET_DISCOVERED`、`TARGET_DESTROYED`、`REGION_VACANCY`、`UAV_DAMAGE` 可作为首批标签。若要覆盖文档中的四类核心扰动，还需先定义并测试事件契约，再训练模型：

- 短时紧急新任务：增加截止期、优先级、持续时间和撤销语义，例如 `URGENT_TASK_ARRIVAL` / `TASK_EXPIRE`；
- 节点损毁/恢复：补充损伤等级、不可用时长和替代/恢复语义，例如 `UAV_DAMAGE` / `UAV_RECOVER`；
- 能量不足：增加电量阈值、RTB、换电/充电、任务移交，例如 `LOW_ENERGY` / `RTB` / `BATTERY_READY`；
- 通信中断/恢复：在弱通信 profile 之外定义断链和恢复语义，例如 `COMM_INTERRUPTION` / `COMM_RECOVERY`。

世界模型只能预测已经定义并能重放的变化，不能弥补事件契约本身缺失的业务语义。确认状态机仍是 `belief` 的唯一事实入口；graph/action version、过期动作拒绝、ACK、lease、fencing token 和唯一执行者约束保持不变。若出现高不确定性、迟滞、频繁切换或假触发，应先退回影子模式或 PPO 默认路由，而不是放宽安全检查。

### 分阶段实施门

| 阶段 | 工作内容 | 通过条件与回退 |
|---|---|---|
| G0 基线冻结 | 锁定发布提交、配置、种子、硬件；完成三/五节点 schema diff | 基础 smoke、测试和回放可重复；问题只在基线修复 |
| G1 基础可用 | 跑通事件注入、确认、图更新、动作提交和指标导出 | 非法正式动作 0、过期动作被拒、成功率和覆盖链路完整 |
| G2 事件契约 | 补齐紧急任务、能量、通信中断/恢复等字段和最小测试 | 每类事件可触发、确认、恢复和确定性回放；失败则 feature flag 关闭 |
| G3 数据闭环 | 增加 history recorder、全 tape 切分、输入白名单和无泄漏审计 | 随机样本可逐条对齐，线上导出器的 truth-only 输入计数为 0 |
| G4 模型离线 | 训练、校准并消融 EASM；分析额外时延 | 达到模型门槛且时延可接受；否则缩短窗口/latent 或仅保留诊断头 |
| G5 影子运行 | 实时预测但不影响决策，对齐 confirmed event | 连续回放无状态污染、日志完整、假触发受控；失败则禁止策略接入 |
| G6 策略接入 | PPO/GPPO 同输入比较，增加按风险/耦合路由和 Planner 兜底 | 业务指标达到冻结门槛且隐藏集不退化；否则保留 PPO 默认路由 |

原计划的倒排节奏是：9 月 1–2 日完成 G1 与事件契约，9 月 3–4 日完成 recorder 和防泄漏数据，9 月 5–7 日完成离线模型与校准，9 月 8 日影子运行，9 月 9–11 日公平对比，9 月 12 日冻结算法和 `Extreme-V2`，9 月 13 日一次性隐藏验收，9 月 14–15 日整理证据和汇报。日期是建议稿，实际执行应以硬件、分支和五节点 schema 是否就绪为前提。

### 公平实验与验收指标

建议采用逐项消融，并保证环境、事件带、可见观测、动作空间、mask、奖励、PPO 超参数、训练步数、种子、评估案例和硬件完全一致：

| 组别 | 历史 | 事件辅助 | GES 门控 | 策略 | 目的 |
|---|---:|---:|---:|---|---|
| A | 否 | 否 | 否 | PPO / GPPO | 复现当前基线 |
| B | 是 | 否 | 否 | PPO / GPPO | 分离“记忆”本身的收益 |
| C | 是 | 是 | 否 | PPO / GPPO | 验证事件辅助监督 |
| D | 是 | 是 | 是 | PPO / GPPO | 验证事件密度/可靠性门控 |
| E | 是 | 是 | 是 | Router | PPO 默认，高耦合风险时 GPPO，不可行时 Planner |

指标分四层报告，不能只给总体 accuracy、episode return 或平均时延：

- 模型层：各事件 PR-AUC、macro-F1、Brier/ECE、H 步 recall、提前量、`state_delta` 误差和假触发率；
- 业务层：累计未覆盖、恢复延迟、归一化距离、负载差、切换次数和最终不可行；
- 系统层：模型额外 P50/P95/P99、端到端 P95/P99、吞吐、内存和过期动作拒绝数；
- 可靠性层：按事件类型、并发度、乱序、丢包和低置信度分层，报告均值、区间和最坏案例。

文档中的暂定门槛须在相同硬件、批大小和预热方式下重新冻结：基础安全要求非法正式动作 0、未拦截过期动作 0、最终不可行率 0；模型 `macro-F1 ≥ 0.70`、`ECE ≤ 0.10`、PR-AUC 相对频率基线提升至少 20%；EASM 额外 P95 不超过 2 ms，PPO 端到端 P95 不超过 5 ms，GPPO 路由目标不超过 20 ms；相对 PPO，累计未覆盖至少下降 10%、恢复延迟至少下降 5%，距离/负载退化不超过 5%。`Extreme-V2` 只运行一次，要求安全不退化且关键业务指标至少不劣于开发集选择的基线。

### 预期代码边界与证据包

建议新增 `ppo_allocation/world_model/`，包含 schema、recorder、model、loss、calibration 和 router；新增独立训练/评估入口及 `configs/world_model_v0.json`；仅对环境、图、模型和训练器增加小型、可回退接口，并配套无泄漏、序列切分、并发事件、校准、旧 checkpoint 兼容和额外时延测试。

最终证据包应包含：锁定提交和环境清单、事件契约、模型卡、A–E 实验主表、失败案例、代码/配置/数据/日志/图表的 SHA-256 inventory、一次性隐藏集记录，以及能够追溯到单次决策的 graph/action version 与事件时间线。世界模型的结果只能作为新增研究证据，不能改写本 README 前文已经封存的 50k minimum-validation 结论。

## 下一步研究方向

1. 按 G0–G3 先完成基线冻结、事件契约、history recorder、全 tape 切分和防泄漏审计；
2. 以影子模式训练和校准 EASM，并验证输入、模型、业务和系统四层指标；
3. 在不改变安全链路的前提下，将世界模型上下文公平接入 PPO 与 GPPO，并评估 Router/Planner 兜底；
4. 增加任务运行时状态机、执行中抢占事务、能耗、RTB、任务移交和切换成本；
5. 实现 Beam-MPC / Rolling-Horizon Planner，并研究低时延蒸馏与 Recurrent QR-DQN / R2D2；
6. 在动态规模和全新 `Extreme-V2` hidden bank 上一次性比较 PPO、GPPO 与规划方法。
