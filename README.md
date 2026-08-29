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

## 下一步研究方向

1. 增加任务运行时状态机与执行中抢占事务；
2. 增加能耗、RTB、任务移交和切换成本；
3. 将信息年龄、置信度和通信可靠性编码进异构图；
4. 实现 Beam-MPC / Rolling-Horizon Planner；
5. 将规划器蒸馏为低时延 MLP，并研究 Recurrent QR-DQN / R2D2；
6. 在动态规模和全新 Extreme-V2 hidden bank 上比较 PPO、GPPO 与规划方法。

执行中抢占与动态重分配已在独立研究合同中启动，详见：

- [Execution-Preemption V1 中文协议](docs/EXECUTION_PREEMPTION_V1_PROTOCOL_ZH.md)
- [Execution-Preemption V1 阶段性研究结论](docs/EXECUTION_PREEMPTION_CONCLUSION_ZH.md)
- [Execution-Preemption V1 机器可读合同](configs/execution_preemption_v1.json)
- [Execution-Preemption V1 算法分配边界](docs/ALLOCATION_BOUNDARY_V1_ZH.md)
- [Dynamic-Preemption-Dev 10×20 开发事件带](experiments/dynamic_preemption/README.md)
