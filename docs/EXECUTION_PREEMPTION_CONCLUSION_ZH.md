# 执行中抢占与动态重分配：阶段性研究结论

> 项目：GPPO-8.29 / `research/execution-preemption-v1`
> 基线：`main@a4207527f713e6f15dcdbc538134aeaca28a03ac`
> 结论日期：2026-08-29
> 文档性质：阶段性结论，不是 PPO/GPPO 最终效果报告

## 1. 一句话结论

本阶段已经把多无人机系统从“事件发生后重新分配”推进到可验证的“任务执行中抢占与动态重分配”机制：任务可在任意进度下继续、排队、暂停、抢占、迁移、终止或返航，且能保证进度可追溯、旧命令不能复活、资源不重复占用。当前证据证明的是**规则机制、事务一致性和算法接入边界正确**，尚不能证明 GPPO、PPO 或规划算法谁的任务效果最好。

## 2. 研究问题与应用场景

项目面向任务持续变化、多个事件连续或同时到达、突发失效、临时更改以及信息不完整的多无人机任务分配。典型运行过程不再是“一次分配后立即完成”，而是：

```text
任务正在执行
  → 新事件到达并被确认
  → 判断继续、排队、暂停、抢占、迁移、终止或返航
  → 保存已有进度并撤销旧命令
  → 更新动态图与安全候选集
  → 规则、PPO、GPPO 或 Planner 选择新的 UAV—Task 配对
  → 通过版本、ACK、lease 与 fencing 校验后恢复执行
```

因此，本项目真正要解决的不是单一静态分配问题，而是同时处理以下四类矛盾：

1. 紧急任务与已有任务进度之间的取舍；
2. 即时响应与全局任务收益之间的取舍；
3. 信息不完整与动作时效性之间的取舍；
4. 学习策略的灵活性与安全约束的确定性之间的取舍。

## 3. 已完成工作

### 3.1 执行语义与状态机

已冻结 `PENDING、ASSIGNED、RUNNING、PAUSED、PREEMPTED、MIGRATING、RESUMING、COMPLETED、FAILED、CANCELLED` 十种任务状态，以及 P0—P4 事件等级和同级事件确定性排序规则。

任务进度采用连续语义：

```text
progress += effective_work_rate × delta_time
remaining_work = 1 - progress
```

同一 UAV 恢复保留全部进度；换 UAV 迁移默认保留 90% 进度，并记录切换时间、迁移损失和中断次数。

### 3.2 抢占控制与原子事务

已实现独立的 `PreemptionController` 和运行时事务。每批已确认事件在事务副本中处理，只有唯一所有权、双向引用、图版本和命令 fencing 等不变量全部通过后才提交。

已经形成以下安全保证：

- 一架 UAV 同时最多执行一个 active task；
- 一个排他任务同时最多拥有一个 active UAV；
- 过期 graph version 的动作和 ACK 被拒绝；
- 低 fencing token 的迟到 ACK 不能恢复旧命令；
- 事件重复到达保持幂等；
- 任务进度只能由唯一 live executor 累计一次；
- 低能量、失联或损毁 UAV 不会进入新的安全候选集。

### 3.3 算法接入边界

安全仲裁与资源选择已经分离：确定性控制器负责 P0/P2 安全、抢占许可、RTB、旧命令撤销和唯一所有权；分配算法只在已冻结的安全候选集合中选择 UAV—Task 配对。

已实现三个统一接口分配器：

| 方法 | 当前作用 | 边界 |
|---|---|---|
| First Available | 最简单的确定性基线 | 按稳定顺序选首个安全 UAV |
| Max Energy Margin | 轻量 Greedy 基线 | 选择返航余量最大的安全 UAV |
| Callback Allocator | PPO/GPPO/Planner 适配入口 | 输出仍须通过版本和候选集校验 |

这意味着后续算法可以公平共享同一事件、候选集和提交协议，模型不能绕过安全规则。

### 3.4 开发场景与确定性证据

已建立 `Dynamic-Preemption-Dev` 开发集，共 10 类场景、每类 20 条固定事件带，合计 200 cases：

1. 搜索执行 40% 时出现紧急任务；
2. 搜索执行 90% 时出现低收益临时任务；
3. 不可抢占打击任务期间出现普通任务；
4. 执行 UAV 中途损毁；
5. 执行 UAV 电量不足并返航；
6. 通信延迟导致旧任务取消报告晚到；
7. 同一任务连续发生优先级变化；
8. 两个 P1 事件同时到达；
9. 新事件在重新分配推理期间到达；
10. 新事件在旧任务恢复过程中到达。

规则回放结果为 200/200 cases PASS，共覆盖 280 个事件决策，资源与事务不变量失败数为 0。决策覆盖如下：

| 决策 | 次数 |
|---|---:|
| CONTINUE | 120 |
| PREEMPT | 40 |
| QUEUE | 40 |
| RTB | 40 |
| MIGRATE | 20 |
| ABORT | 20 |

两个确定性分配器在同一开发集上共完成 400 个 allocator-tape runs，每个分配器处理 280 个事件决策和 80 个版本化分配请求，invariant failures 均为 0。

### 3.5 异构图与规模适配

已经形成新的五类节点图结构：`UAV、Task、Region、Target、Event`，以及执行、可执行、位置、依赖、事件影响和抢占等关系。动作空间定义为动态 UAV—Task 边，并保留显式 `NOOP` 与安全 action mask。

当前 4/8/16/32 UAV 的图构建 smoke 均可生成稳定、版本绑定的图快照，任务数按每架 UAV 两个任务同步扩展。该结果只证明图 schema 和动态尺寸构建可运行，**不是模型训练或效果证据**。

### 3.6 本次复验状态

截至本文生成时，隔离 worktree 中的验证结果为：

| 检查项 | 结果 |
|---|---:|
| Execution-Preemption 专项测试 | 43/43 PASS |
| 原 minimum-validation required tests | 130/130 PASS |
| 规则开发事件带 | 200/200 PASS |
| 确定性 allocator-tape runs | 400/400 PASS |
| 4/8/16/32 UAV 图 schema smoke | 4/4 PASS |
| Reward/metrics/training contract smoke | PASS，training_allowed=false |
| PPO/GPPO unified adapter smoke | 4/8/16/32 PASS，training_allowed=false |
| Direct/deferred atomic transaction parity | 400/400 PASS，state SHA parity 400/400 |
| 正式训练 | 未启动 |
| Validation / Freeze / Test / held-out | 未启动 |

上述测试用于确认新增研究模块未破坏既有基线，并不把旧 minimum-validation 证据复用为新模型效果证据。

## 4. 当前证据说明了什么

### 4.1 已经可以成立的结论

1. **规则化抢占机制可以闭环。** 十类极端动态事件已覆盖继续、排队、抢占、迁移、终止和返航路径，并通过确定性回放。
2. **并发安全可以独立于模型保证。** 即使算法返回非法 UAV、过期版本或错误 proposal，事务也会回滚，不污染 live runtime。
3. **连续进度比“分配即完成”更符合实际。** 系统可以量化抢占时机、进度损失、恢复率和切换成本，为后续算法比较提供业务指标。
4. **安全决策与学习决策应分层。** P0 安全、RTB、命令撤销和 fencing 由规则控制；学习策略只优化安全候选集内的分配，这一结构更可解释，也更容易审计。
5. **新问题必须重新训练。** 新增 Task/Event 节点、连续进度、迁移损失和抢占语义后，旧 minimum-validation checkpoint 与新合同不兼容，不能直接复用来形成新结论。

### 4.2 目前不能成立的结论

1. 不能声称 GPPO 在执行中抢占场景优于 PPO；
2. 不能声称 Greedy 与 First Available 效果相同——当前事件带主要验证接口和安全性，尚未形成足够的距离、负载和 deadline 差异；
3. 不能把 4/8/16/32 图 smoke 当作规模泛化实验；
4. 不能把开发集结果当作 hidden/held-out 结果；
5. 不能复用旧 minimum-validation 的模型或结论代替新合同下的训练和评估。

## 5. 不同方法何时使用

| 场景 | 推荐方法 | 优点 | 主要局限 |
|---|---|---|---|
| P0 安全、低能量返航、UAV 损毁 | 确定性规则控制器 | 可审计、反应稳定、能给出硬保证 | 不负责复杂全局收益优化 |
| 规模较小、关系简单、实时性极强 | Greedy + Rule Arbiter | 延迟低、实现简单、容易复现 | 容易局部最优，难预见未来资源释放 |
| 固定规模、状态可展平、要求低推理时延 | PPO-MLP + Rule Arbiter | 推理快，是可靠学习基线 | 对拓扑变化、关系结构和规模迁移表达较弱 |
| 多任务依赖、资源争用、拓扑和规模动态变化 | GPPO-Adaptive + Rule Arbiter | 能显式编码 UAV—Task—Event 关系，理论上更适合关系推理 | 推理更慢、训练更复杂，优势必须重新实验证明 |
| 小中规模、高价值任务、可获得较准确模型 | Beam-MPC / Rolling Horizon + Rule Arbiter | 能显式考虑未来事件、恢复成本和 deadline | 规模增大后搜索成本和 P95/P99 延迟可能迅速上升 |
| 信息长期不全、乱序和延迟严重 | Belief State + Recurrent Policy/Planner | 能利用历史估计隐状态 | 状态估计误差和训练稳定性带来新风险 |

综合当前场景，最值得验证的不是“纯 GPPO 替代全部方法”，而是以下混合结构：

```text
确定性安全与抢占仲裁
        +
PPO / GPPO / Beam-MPC 的安全候选集内分配
        +
版本化事务与命令 fencing
```

## 6. 哪些算法最可能突破 PPO 基线

以下判断是待验证假设，不是现有实验结论：

1. **GPPO-Adaptive** 最可能在任务依赖明显、UAV—Task 兼容关系稀疏、资源争用强、图规模变化大的场景超过 PPO-MLP；如果状态只是小规模稠密向量，图模型的额外开销可能得不到回报。
2. **Beam-MPC** 最可能在事件可预测、deadline 和切换成本清晰、需要等待未来资源释放的场景超过纯反应式 PPO；但 16/32 UAV 时必须严格检查决策时延。
3. **GPPO + Rule Arbiter** 是本项目最有研究价值的主方案，因为它同时保留关系建模能力与硬安全边界。
4. **PPO + Rule Arbiter** 应作为低时延学习基线，判断图结构本身是否真正带来增益。
5. 对长期信息缺失场景，单纯前馈 PPO/GPPO 都可能不足，应单独研究 recurrent policy、belief state 或显式状态估计，不能把观测缺失误认为分配器能力不足。

## 7. 推荐的正式比较设计

在启动模型训练前，应先冻结新的 observation、action、reward、metrics、训练预算、seed manifest 和 checkpoint 规则。随后按以下顺序推进：

1. 4 UAV：验证新环境、reward、mask、日志和 checkpoint 能完整闭环；
2. 8 UAV：验证并发一致性和算法间公平比较；
3. 16 UAV：主比较 PPO、GPPO 与 Beam-MPC；
4. 32 UAV：只在前三级全部通过后测试扩展性与决策时延；
5. 开发集只用于调参和诊断；算法及阈值冻结后，再生成 seed、事件时间和参数范围完全隔离的 `Dynamic-Preemption-Hidden-V1`；
6. hidden 只运行一次，并同时报告平均值、尾部风险和失败样本，不进行 checkpoint selection。

主要指标应包括：紧急任务 deadline miss、抢占响应时延、累计任务空缺、原任务恢复率、平均进度损失、任务饥饿率、距离、负载、能源安全违规、重复执行者，以及 P95/P99 决策时延。

建议继续保持以下硬门槛：

```text
资源冲突 = 0
旧命令复活 = 0
能源安全违规 = 0
P0 事件处理率 = 100%
紧急任务 deadline miss 相对旧方法降低 ≥ 10%
累计任务空缺相对旧方法降低 ≥ 10%
普通任务恢复率 ≥ 95%
```

## 8. 局限与风险

- 当前 200 条事件带是开发集，覆盖的是机制边界，不代表真实任务分布；
- 当前两个确定性分配器在这些 tapes 上选择计数相同，说明场景还需加入更强的空间、能耗、deadline 和负载差异；
- 图 schema、reward、指标、训练合同和统一 PPO/GPPO adapter 已经冻结，但 Gym/PyTorch/PyG 环境接线尚未实现；
- 尚未执行新合同下的 PPO/GPPO/Beam-MPC 训练与同带效果比较；
- 尚未生成独立 hidden bank，因此没有最终泛化结论；
- 32 UAV 的候选边会快速增长，必须同时报告算法效果和尾部推理时延；
- 信息长期缺失属于部分可观测问题，仅扩大图模型不能自动解决。

## 9. 阶段性总评

当前工作已经完成了研究中最容易被忽略、但决定结论是否可信的底层部分：先定义执行语义，再建立状态机和原子抢占事务，随后冻结算法边界与开发事件带。由此，后续无论接入 PPO、GPPO、Greedy 还是 Planner，都可以在相同安全约束、相同事件输入和相同证据口径下比较。

最稳妥的阶段性表述是：

> 已经证明执行中抢占与动态重分配机制在十类极端事件下能够保持安全、一致和可追溯；已经冻结 PPO、GPPO 与规划器共享的图、reward、指标、训练合同和版本化动作接口。尚未证明任何学习算法在新合同下优于 PPO，下一阶段必须完成框架接线和 source-bound Gate、重新训练，并通过独立 hidden bank 才能形成最终算法结论。

## 10. 向学姐汇报的建议口径

可以将本阶段汇报概括为三点：

1. **机制进展：** 系统已经从事件后重分配扩展到任务执行中的暂停、抢占、迁移、恢复和返航，并用版本、ACK、lease 和 fencing 防止旧命令复活。
2. **证据进展：** 十类场景、200 条开发事件带全部通过规则回放；两个确定性分配器共 400 次同带回放，没有资源冲突或事务不变量失败；4/8/16/32 UAV 图构建 smoke 已跑通。
3. **结论边界：** 目前完成的是机制、接口和训练合同验证，不是模型胜负。下一阶段会完成框架接线，重新训练 PPO/GPPO，并加入 Beam-MPC；只有独立 hidden 结果出来后才讨论谁超过 PPO。

## 11. 证据入口

- 协议：[Execution-Preemption V1](EXECUTION_PREEMPTION_V1_PROTOCOL_ZH.md)
- 算法边界：[Allocation Boundary V1](ALLOCATION_BOUNDARY_V1_ZH.md)
- 机器可读合同：[`configs/execution_preemption_v1.json`](../configs/execution_preemption_v1.json)
- 开发集回放报告：[`RULE_REPLAY_REPORT_ZH.md`](../experiments/dynamic_preemption/dev_v1/RULE_REPLAY_REPORT_ZH.md)
- 分配器回放摘要：[`allocator_replay_summary.json`](../experiments/dynamic_preemption/dev_v1/allocator_replay_summary.json)
- 图 schema：[`configs/execution_graph_v1.json`](../configs/execution_graph_v1.json)
- 图规模 smoke：[`graph_schema_smoke.json`](../experiments/dynamic_preemption/dev_v1/graph_schema_smoke.json)
- Reward、指标与训练合同：[Execution Training Contract V1](EXECUTION_TRAINING_CONTRACT_V1_ZH.md)
- 训练合同 smoke：[`training_contract_smoke.json`](../experiments/dynamic_preemption/dev_v1/training_contract_smoke.json)
- 统一策略适配器：[Policy Adapter V1](POLICY_ADAPTER_V1_ZH.md)
- Adapter smoke：[`policy_adapter_smoke.json`](../experiments/dynamic_preemption/dev_v1/policy_adapter_smoke.json)
- 延迟原子事务 parity：[`deferred_transaction_parity.json`](../experiments/dynamic_preemption/dev_v1/deferred_transaction_parity.json)
