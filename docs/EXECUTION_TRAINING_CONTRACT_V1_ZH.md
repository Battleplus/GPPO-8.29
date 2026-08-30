# Execution-Preemption V1：Reward、指标与训练合同

> 合同 ID：`execution-preemption-training-v1`
> 状态：`FROZEN_FOR_BASELINE_IMPLEMENTATION`
> 当前训练许可：`false`

## 1. 目的与边界

本合同冻结执行中抢占问题的新 reward、指标、比较方法、训练预算、seed、checkpoint 和规模晋级规则。它属于 `execution_preemption_v1` 新实验命名空间，不修改 `ppo_allocation/random_event/reward.py`，不复用旧 minimum-validation checkpoint、训练结果或 held-out 证据。

旧问题把事件后的局部重新分配作为主要决策；新问题加入连续任务进度、执行中抢占、迁移损失、恢复、RTB、Event/Task 图节点和动态规模，因此旧 checkpoint 与新状态—动作—回报语义不兼容。数值上继续使用已冻结的 seeds `1101/2202/3303`、50k budget 和 `[25k, 50k]` grid，但 seed 身份由 `contract_id/namespace/integer` 唯一确定，不能把同一个数字误认为旧实验产物可复用。

机器可读合同见 [`configs/execution_training_contract_v1.json`](../configs/execution_training_contract_v1.json)。

## 2. Reward 合同

每个 accepted allocation transition 先产生九个归一化软信号，取值均为 `[0,1]`：

| 信号 | 权重 | 含义 |
|---|---:|---|
| `weighted_progress_gain` | +4.0 | 按任务优先级加权的有效进度增量 |
| `urgent_deadline_miss_rate` | -10.0 | 当前区间紧急任务 deadline miss 比例 |
| `weighted_vacancy_time` | -5.0 | 按优先级和物理时间累计的任务空缺 |
| `progress_loss` | -4.0 | 抢占、迁移或重启造成的已完成工作损失 |
| `starvation_exposure` | -3.0 | 任务等待超过冻结阈值后的暴露量 |
| `switch_time` | -1.0 | 抢占与迁移带来的切换时间 |
| `energy_consumed` | -1.0 | 决策区间内归一化能耗 |
| `normalized_distance` | -0.5 | 分配产生的归一化移动距离 |
| `load_gap` | -0.5 | UAV 间任务负载差异 |

冻结公式为：

```text
r_t = Σ weight_i × normalized_transition_signal_i
```

所有方法必须由同一个适配器计算这些信号，算法不得自报 reward。缺少任一必需信号、出现非有限值或越出 `[0,1]` 都是硬错误，不允许静默补零。

### 2.1 安全不是 reward 惩罚

以下三项不进入标量 reward：

```text
resource_conflicts
stale_command_resurrections
energy_safety_violations
```

任一项大于 0 时，该 transition 不具备学习资格，运行必须停止并保留原始证据。这样可以防止策略用更高任务收益“购买”安全违规。

## 3. 指标合同

模型比较不能只报告 episode return。每条 paired tape 必须保存以下原始指标：

- 紧急任务总数、deadline miss 数和 miss rate；
- P0 事件数、正确处理数和处理率；
- 抢占响应时延；
- 被打断任务数、恢复数、右删失数、恢复率和已观察恢复时延；
- 累计加权空缺、累计进度损失、任务饥饿暴露和饥饿率；
- 累计切换时间、能耗、距离和平均负载差；
- 资源冲突、旧命令复活和能源安全违规；
- 推理时延 mean、P95 和 P99。

未恢复任务必须保留为 `right_censored_recovery_count`，不能删除，也不能把恢复时延写成 0。缺少分母时，rate 必须为 `null` 而不是伪造 0。

最低结果门为：

```text
资源冲突 = 0
旧命令复活 = 0
能源安全违规 = 0
P0 事件处理率 = 100%
紧急任务 deadline miss 相对学姐旧方法降低 ≥ 10%
累计任务空缺相对学姐旧方法降低 ≥ 10%
普通任务恢复率 ≥ 95%
```

如果基线对应指标为 0，不能把相对改善定义为通过；必须报告基线为零导致该比较不可判定。

## 4. 七个比较方法

| 方法 ID | 类型 | V1 语义 |
|---|---|---|
| `senior_legacy_method_v1` | 旧方法基线 | 经同一安全 shell 适配后的旧式分配逻辑 |
| `greedy_priority_v1` | 确定性基线 | 按任务优先级、deadline 与安全候选排序 |
| `ppo_mlp_reactive_v1` | 学习基线 | 安全 mask 内的反应式平坦状态分配，不接收规则决策上下文 |
| `gppo_adaptive_reactive_v1` | 图学习基线 | 安全 mask 内的反应式异构图分配，不接收规则决策上下文 |
| `beam_mpc_v1` | 规划基线 | 在冻结预测时域内搜索安全候选序列 |
| `ppo_mlp_rule_arbiter_v1` | 混合方法 | 接收冻结 `EventDecision` 上下文后选择安全 UAV—Task 边 |
| `gppo_adaptive_rule_arbiter_v1` | 混合方法 | 在异构图中编码 `EventDecision` 上下文后选择安全边 |

所有七种方法都必须经过安全候选 mask、proposal validator、graph version、ACK、lease 和 fencing。所谓“reactive”只表示不向策略暴露规则仲裁上下文，不表示允许模型控制 P0、RTB 或绕过安全约束。

## 5. 训练与 checkpoint 合同

四个学习方法使用同一合同：

```text
Seeds: 1101, 2202, 3303
Budget: 50,000 accepted decision steps / run
Checkpoints: 25,000 and 50,000
Fixed evaluation checkpoint: 50,000
Checkpoint selection: false
Training scales: 4, 8, 16 UAV
32 UAV: fixed 16-UAV 50k checkpoint 的 zero-shot scalability 检查
```

计划基数为：

```text
4 learned methods × 3 seeds × 3 training scales = 36 runs
36 runs × 2 checkpoints = 72 checkpoints
固定后续候选：36 个 50k checkpoints
```

32 UAV 只检查动态图适配、任务效果和尾部推理时延，不在 32 UAV hidden 结果上重训、调参或选择 checkpoint。如果 16→32 零样本迁移失败，应如实报告为扩展性边界，不能临时改变合同。

## 6. Seed 与事件带隔离

- Training namespace：`execution_preemption_v1/train`；
- Development bank：`Dynamic-Preemption-Dev`，当前 10×20 paired tapes；
- Hidden bank：`Dynamic-Preemption-Hidden-V1`，当前状态必须为 `NOT_GENERATED`；
- Hidden 只能在模型、reward、阈值和 checkpoint 使用规则冻结后生成；
- Development 与 Hidden 的 seed、事件时间和参数范围不允许重叠；
- Hidden 只能执行一次，不参与 checkpoint selection。

## 7. 规模晋级门

```text
4 UAV  → 语义、reward、mask、日志与 checkpoint smoke
8 UAV  → 执行一致性、worker 隔离与确定性复验
16 UAV → 主要算法比较
32 UAV → 固定 16-UAV 50k checkpoint 的零样本扩展与 P95/P99 时延
```

前一级未全部通过时不得进入下一级。任何 run steps 异常、checkpoint 缺失、SHA 不匹配、provenance 漂移或硬安全违规都必须停止并保留产物，不自动续训或修补。

## 8. 当前 Gate 状态

当前合同只冻结训练前语义：

```text
training_allowed = false
source_bound_launch_gate_created = false
training_started = false
validation_started = false
freeze_started = false
test_started = false
hidden_evaluation_started = false
```

Gymnasium 环境、PyTorch tensor 转换、实际 transition reward 接线及 PPO/GPPO 确定性短回放已经通过，但这只满足 launch Gate 的一部分前置条件，不构成训练授权。

只有 observation/action adapter、环境 reward 接线、日志 schema、训练 smoke、required tests 和 source-bound Gate 全部通过后，才能在新的 clean training worktree 中生成允许训练的 evidence。修改本配置中的 `training_allowed` 本身不能授权训练。

新 Gate 与旧 minimum-validation `handoff/P0_GATE.json` 完全分离。唯一允许的 evidence 路径是 `experiments/dynamic_preemption/evidence_v1/EXECUTION_PREEMPTION_V1_GATE.json`；source 到 evidence HEAD 出现任何其他文件、任何受保护源码变更或 checkpoint 都必须 fail-closed。当前 Gate 至少要求旧基线测试 130 项和新专项测试 111 项；后续新增基线测试时该下限还会随 source 一起冻结更新。

训练器已经实现 36 个 worker 的唯一目录、训练专用 tape namespace、PPO 更新、accepted-step 精确计数、25k/50k checkpoint、optimizer/RNG/provenance 封存以及只读 SHA inventory 复验。当前只运行了 2-step tiny smoke，没有启动任何 50k formal run。

## 9. 当前验证

纯函数测试已经覆盖：

- reward 精确加权和与分量可追溯；
- 软信号范围和有限值检查；
- 安全违规硬失败；
- recovery 右删失和缺失分母语义；
- mean/P95/P99 时延统计；
- 结果门通过、失败和零基线 fail-closed；
- budget、seed、checkpoint、方法集和旧证据禁用的 drift 检查；
- 合同 smoke 的字节级确定性和全阶段未启动标记。
- Gym/PyTorch framework rollout 的确定性、共享 mask/action、奖励接线与零优化器步数。

机器可读 smoke 见 [`training_contract_smoke.json`](../experiments/dynamic_preemption/dev_v1/training_contract_smoke.json)。该文件分类为 `training_precondition_contract_smoke_not_model_evidence`，不能用于声明模型效果。

框架回放证据见 [`framework_rollout_smoke.json`](../experiments/dynamic_preemption/dev_v1/framework_rollout_smoke.json)，其分类为 `framework_rollout_smoke_not_training_or_model_evidence`。
