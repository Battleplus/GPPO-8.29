# 随机事件触发 GPPO：实验设计与可复现协议

## 1. 研究边界与问题定义

本实验保留 `ppo_allocation` 的原任务语义：4 架 UAV、4 个搜索 Region、3 个 Target；UAV 的任务状态仅为 `SEARCH`、`TRACK`、`IDLE`。发生 `UAV_DAMAGE`、`TARGET_DISCOVERED`、`TARGET_DESTROYED` 或 `REGION_VACANCY` 后，只对受影响的搜索区域进行动态重分配。

论文在本项目中只提供方法结构，不把原任务改写为论文的“父任务—子任务”调度：

```text
真实/仿真状态与新事件
  -> 事件侦测、去重和队列
  -> UAV/Region/Target 异构图 + 动作 Mask
  -> AHGNN 按关系聚合消息
  -> Adaptive Gate 筛选、融合图消息
  -> Actor 为候选 UAV—Region 边评分
  -> Critic 对全图池化后的状态估计 V(G)
  -> 提交前检查 graph_version
  -> PPO 用轨迹端到端更新编码器、Gate、Actor 和 Critic
```

核心问题是：多个随机事件连续或重叠出现时，完整 GPPO 是否比原始 MLP-PPO 与启发式方法更快恢复覆盖，并同时降低累计空缺、转移距离、负载失衡和事件触发通信量。

## 2. 事件是如何被真实系统发现的

“事件触发决策”不等于系统完全不做定时监测。真实系统可低频或连续采集健康与感知信息，但只有状态变化越过判据时才触发分配决策和相应通信。

| 事件 | 原始信号 | 默认确认规则 | `occurred_at` 与 `observed_at` |
|---|---|---|---|
| `UAV_DAMAGE` | 飞控健康字、传感器故障字、心跳 | 明确故障码，或连续心跳超时达到阈值 | 故障发生时刻与协调器确认时刻，可因弱通信不同 |
| `TARGET_DISCOVERED` | 机载感知/融合跟踪器 | 目标置信度连续超过阈值并完成去重 | 首次可靠观测与事件包到达时刻 |
| `TARGET_DESTROYED` | 武器效果评估/跟踪终止确认 | 明确摧毁确认；单纯短时丢踪不等同摧毁 | 摧毁时刻与确认包到达时刻 |
| `REGION_VACANCY` | 分配表、UAV 模式和心跳的一致性检查 | Region 有负责人但负责人无法继续 SEARCH，或租约超时 | 失去合法负责人的时刻与监控器确认时刻 |

仿真调度器直接生成以上已确认事件，同时显式保存侦测延迟 `observed_at - occurred_at`。定时心跳只是探查手段，重新分配由事件触发，不在每个心跳周期无条件推理。

## 3. 事件带与状态机

### 3.1 冻结事件记录

每条不可变事件至少保存：

- `event_id`、`event_type`、`source_event`；
- `occurred_at`、`observed_at`；
- `affected_uavs`、`affected_regions`、`affected_targets`；
- `severity`、`payload`；
- `event_seed`、`state_version`。

事件带由 `(initial_seed, event_seed, mode, protocol_version)` 唯一确定。序列化使用 UTF-8、固定字段顺序、无 NaN、稳定浮点格式；相同输入的规范 JSON 字节和 SHA-256 必须一致。

### 3.2 条件抽样，不做拒绝采样

基础权重为 `UAV_DAMAGE=0.30`、`TARGET_DISCOVERED=0.30`、`TARGET_DESTROYED=0.20`、`REGION_VACANCY=0.20`。每次先由当前状态算出有效类型集合，再仅在该集合上重新归一化。选定类型后从排序稳定的合法实体集合中一次采样，禁止“最多重试 30 次”的拒绝式实现。

合法性约束：

- 损毁对象必须存活且当前实际承担搜索区域；损毁后释放其全部搜索区域，并 Mask 该 UAV 的所有边。
- 发现事件要求搜索 UAV 的负责区域内存在未发现目标；UAV 转 `TRACK`，释放原搜索区域，同时保留 `source_event=TARGET_DISCOVERED`，不能只记录成 `REGION_VACANCY`。
- 摧毁事件要求目标正在被跟踪且尚未摧毁；跟踪 UAV 被释放并重新进入搜索候选集合。
- 空缺事件随机清除一个当前合法负责人，Region 进入 `need_reassign`。

### 3.3 四种到达模式和 horizon

- `single`：每个独立子回合只注入一个事件，用来测量单事件局部恢复。Smoke 中“一条 single 事件带至少 3 个事件”解释为一个 bundle 含 3 个相互 reset 的单事件子回合，不能把三次事件误当成一个连续 episode。
- `sequential`：上一事件恢复后，才调度下一事件。
- `overlap`：上一事件仍有 pending Region 时，新事件可到达。
- `burst`：同一观察时间窗内到达 2–3 个事件；按冻结顺序原子应用后只构图一次。

普通 episode 默认 5 个事件；由事件数或物理时间结束，而不是首次 `all_valid` 就结束。只有事件源耗尽、队列清空且 `pending_regions` 为空才正常终止。当前无合法 UAV 但未来事件可能释放 UAV 时记录 `temporarily_unrecoverable`，只有 horizon 已结束且仍无恢复路径时才记录 `finally_infeasible`。

## 4. 执行中的任务和重叠事件

默认采用局部、非抢占语义：

- 已完成任务保持锁定；未受扰动的分配与 SEARCH UAV 不变。
- 只把损毁 UAV、转入 TRACK 的 UAV 或显式空缺事件涉及的 Region 放入 `pending_regions`。
- 普通新事件不能无条件打断无关任务。
- 原代码没有连续任务进度模型，因此本实验不记录虚构的“完成百分比”。若未来加入进度，必须用独立开关，并明确是从头重启还是保留剩余工作量。

运行时维护 `event_queue`、`pending_regions`、`graph_version` 和 `decision_version`：

1. 新事件被确认后先去重，再按 `(observed_at, occurred_at, tape_order, event_id)` 进入队列。
2. 同一 burst 的事件按 `tape_order` 在一个事务内应用；全部成功后提交状态并递增一次 `graph_version`。
3. 顺序/重叠事件每应用一个原子批次便递增 `graph_version`，只更新受影响的节点、边与 Mask。
4. 推理请求携带 `decision_version = graph_version`。Actor 推理期间若新事件使版本变化，返回动作必须拒绝；`stale_action_rejection_count += 1`，基于新图重新推理。
5. 已经开始执行但不受新事件影响的任务继续；受影响动作只有在版本检查通过后才提交。

这覆盖了两种现实情况：执行任务时突然来事件，以及恢复旧事件时再次来事件。后一种不会把旧结果静默覆盖，而是入队、合并图变更并使旧版本决策失效。

## 5. 图状态、动作和 GPPO

### 5.1 异构图

节点特征保持原任务语义：

- UAV：位置、`alive`、SEARCH/TRACK/IDLE、负责区域数、传感器类型、`sensor_failed`、`target_id`。
- Region：中心位置、`assigned_uav`、`need_reassign`、优先级、工作量、空缺持续时间。
- Target：位置、`discovered`、`tracked`、`destroyed`、`tracker_id`、所属 Region。

边关系为：

- UAV—Region：可分配性、当前分配、归一化距离、负载和传感器可用性；
- UAV—Target：发现或跟踪关系；
- Target—Region：目标所属关系。

所有离散 ID 用 one-hot/embedding 表示，不把编号大小伪装成连续物理量；距离和计数使用训练集统计量归一化。

### 5.2 网络与动作

AHGNN 对每种节点/边关系分别编码和注意力聚合。Adaptive Gate 使用局部表示、关系消息与事件上下文产生 `[0,1]` 门值，控制消息进入融合层。需要记录门值分布和梯度范数；`GPPO-NoGate` 保留相同图编码规模，将门固定为 1，形成机制消融。

Actor 的动作是候选边 `(UAV_i, Region_j)`，另加 `NOOP`。非法边 logit 在 softmax 前置为负无穷；`NOOP` 只在没有待分配 Region、暂时无合法边或协议明确要求等待后续事件时合法。Critic 对三类节点的全图表示进行池化，输出 `V(G)`。PPO 端到端更新图编码器、Gate、Actor 和 Critic。

必须在应用 Mask 前对 logits 做 softmax 并记录 `pre_mask_invalid_probability`。Mask 后的 100% 合法动作只证明约束实现正确，不能作为策略学会分配的证据。

## 6. 奖励与代价

每次合法决策使用状态代价差：

```text
r_t = J(s_before) - J(s_after)

J(s) = alpha * weighted_uncovered_regions
     + beta  * normalized_transfer_distance
     + gamma * load_gap
     + delta * switch_count
     + eta   * recovery_delay
```

其中 `weighted_uncovered_regions` 进一步分解为当前优先级加权未覆盖量与归一化 vacancy duration，因此日志同时保存 `coverage_component` 和 `vacancy_component`；`recovery_component` 对应事件到恢复的延迟项。另保存 `distance_component`、`balance_component`、`switch_component`。默认权重只由训练/Validation 调整并冻结到 Test；禁止看 Test 后修改。

动作合法性由硬 Mask 保证，仅为接口异常保留有限、可审计的小额惩罚，不用巨额非法惩罚替代约束。

## 7. 比较方法与 Oracle

七个可执行决策策略使用同一接口：

1. Masked Random；
2. Nearest Legal；
3. Min Load；
4. Greedy Cost；
5. 原始 PPO-MLP；
6. GPPO-NoGate；
7. GPPO-Adaptive。

此外使用 `Exhaustive Oracle` 作为第八个参照器：在 4 UAV × 4 Region 的当前合法分配空间枚举方案，返回最小 `J`，并报告 `oracle_regret = J_algorithm - J_oracle`。Oracle 不参与训练，也不占七个策略方法之一。

所有学习方法使用相同环境交互/决策步预算、相同训练 seed 集、同一冻结 Validation bank 和相同 checkpoint 规则。启发式方法不训练，但在完全相同评估 bank 上运行。

## 8. 公平配对：严格边界和可实施双协议

### 8.1 逻辑上的边界

连续多事件轨迹中，不同策略在第一次事件后通常产生不同负责人。若后续事件是“损毁当前承担 Region 的 UAV”，即使后续冻结同一个 UAV ID，该 UAV 在各策略下实际负责的 Region 也可能不同。因此，无法在不强行改写某个策略状态的前提下，同时保证：

1. 连续保留各策略自己的决策后果；
2. 下一事件的损毁 UAV 和实际派生 `affected_regions` 完全相同。

强制把各策略状态改回统一分配，或清除一个并非由被损毁 UAV 负责的冻结 Region，都会破坏任务语义。报告不得把“相同外生事件”虚称为“所有派生影响完全相同”。

### 8.2 双协议

为同时回答局部因果比较和连续系统鲁棒性，采用两个互补协议：

**P1：Anchored exact-pair（严格配对）**

- 每个事件/事件 burst 从冻结的共同 pre-event state snapshot 分支。
- 事件类型、时刻、UAV、Region、Target、状态变化和实际 `affected_*` 对所有算法逐字节相同。
- 用于 Test-Single、单个 burst 的局部恢复、Oracle regret 和配对显著性主结论。
- 每个分支独立恢复，不声称它代表跨多个已分叉决策的长期连续历史。

**P2：Continuous exogenous-tape（连续系统）**

- 每个算法从相同初态重放相同 `source_event`、发生/观察时刻、目标 UAV/Target/Region、severity 和 payload。
- 保留算法先前决策造成的状态分叉，测试 Sequential/Overlap/Burst 的真实闭环行为。
- 同时记录事件带的 `intended_affected_regions` 和各策略状态导出的 `actual_affected_regions`。若二者不同，不修改状态“凑一致”，并将该事件标记为 `endogenous_effect_divergence`。
- 共同事件带仍可控制外生随机性并做同 tape 配对差值，但结论措辞限定为“在相同外生冲击下的系统级表现”，不解释为完全相同受影响状态下的纯策略效果。

正式报告分别列出 P1 与 P2；不混合两类样本计算一个置信区间。P1 满足“受影响区域等完全相同”的严格比较要求，P2 满足多个连续/重叠事件的闭环研究要求。

## 9. 数据隔离、预算和 checkpoint

事件 seed 命名空间在 `configs/seed_manifest.json` 冻结。Train、Validation、Test 互不重叠；事件带生成后保存规范 JSON 和 SHA-256。Test 只由最终冻结 checkpoint 读取，不进入 early stopping、超参数、奖励权重或阈值选择。

执行顺序：

1. 自动化状态、事件、Mask、版本和 legacy 测试；
2. Smoke：single/sequential/overlap/burst 每种至少 20 个 bundle/tape，连续 tape 至少 3 个事件；
3. Preliminary：3 个训练 seed、同等预算、冻结 Validation、200 条 Test（五类各 40）；
4. Formal：5 个训练 seed、冻结 Validation、至少 1000 条从未评估的 Test（五类各 200）。

五类 Test：

- Test-Single：严格 P1 单事件分支 bundle；
- Test-Sequential：旧事件恢复后再到达；
- Test-Overlap：未恢复时新事件到达；
- Test-Burst：同一窗口 2–3 事件原子应用；
- Test-Unseen：改变事件类型权重和侦测延迟组合，但不改变事件语义。

`Test-Unseen` 冻结为 `unseen_shift_v1`：四类事件仍是同一套状态转移和合法性约束，
但基础权重从 `0.30/0.30/0.20/0.20` 改为 `0.15/0.15/0.30/0.40`；
同时把 single、sequential、overlap、burst 的侦测延迟分别移到
`[1.5,3.0]`、`[1.5,3.0]`、`[3.0,6.0]`、`[2.0,4.0]` 秒。
这属于事件混合与弱通信延迟的联合分布外测试，不引入第五类事件，也不改变任务背景。
Unseen 的底层时序模式按四模式循环并逐 tape 写入 manifest；训练只使用默认分布。

冻结 bank 由 seed manifest 直接展开，禁止运行时随机产生 seed。示例（在
`ppo_allocation` 下执行）：

```powershell
python run_random_event_experiment.py protocol-bank --tier preliminary --split validation
python run_random_event_experiment.py protocol-bank --tier preliminary --split test
```

Validation manifest 标记 `checkpoint_selection=true`；Test manifest 标记
`intended_use=final_evaluation_only`，且两者记录 protocol/seed manifest 的 SHA-256。

Preliminary 的 3 seed/200 tapes 必须在文件名、表格和图题标记为 `preliminary`，不能等同正式结论。

## 10. 指标、通信核算与统计检验

### 10.1 指标

主要指标：`event_recovery_success_rate`、`recovery_delay`、`recovery_decision_steps`、`weighted_coverage_auc`、`cumulative_vacancy_time`、`distance_cost`、`load_gap`、`switch_count`、`oracle_regret`、`infeasible_case_rate`。

触发与通信指标：`trigger_count`、`inference_count_per_event`、`communication_bytes`、`merged_event_count`、`stale_action_rejection_count`、`event_to_decision_latency`。

机制指标：`valid_action_count`、`action_mask_ratio`、`pre_mask_invalid_probability`、`actor_entropy`、`critic_value_loss`、`explained_variance`、`approximate_kl`、`ppo_clip_fraction`、Gate 的 mean/std/p10/p50/p90、`gate_gradient_norm`。

通信量按序列化后的事件包、变化节点/边、推理请求、动作响应的实际字节求和；不把 Python 对象内存大小当网络字节。该值是协议仿真量，未接真实无线链路前不得称为实测带宽。

所有指标按事件类型、事件模式、可行/不可行、算法、训练 seed 分层；额外报告 P1/P2、`endogenous_effect_divergence` 与事件侦测延迟分层。

### 10.2 统计规则

- 同一训练 seed 内，算法在相同 tape/snapshot 上形成配对差值。
- 先对每个训练 seed 的全部冻结 Test tape 求均值，再以训练 seed 为算法稳定性的独立重复；报告 raw seed 值、均值、标准差和 95% CI。
- Preliminary 只有 3 个训练 seed，CI 很宽，必须标记探索性；可补充分层 bootstrap 作为敏感性分析，但实例级 bootstrap CI 不能冒充 seed 级稳定性。
- Formal 使用 5 个训练 seed；主比较为 GPPO-Adaptive 对 PPO-MLP、GPPO-NoGate 和 Greedy Cost 的双侧配对检验与配对效应量。多主指标/多基线使用 Holm 校正。
- recovery delay 对未恢复样本同时报告成功条件分布和 horizon 右删失分析；不能只删除失败样本。
- 任何 GPPO 未胜出的结果原样报告，并从图规模小、状态同质化、Mask 过强、奖励失真、训练不足等可检验原因分析，不调整 Test 分布补救。

## 11. 必须通过的验证门

进入长训练前至少验证：

1. 同 seed 事件带字节相同，不同 seed 不同；
2. 四类事件在条件满足时均可生成；无 tracked target 时不能生成 `TARGET_DESTROYED`；
3. 损毁 UAV 的所有边被 Mask；目标摧毁后 tracker 重新可用；
4. 一个 episode 处理至少 5 个事件，事件未耗尽时 `all_valid` 不终止；
5. overlap 使旧版本动作失效，burst 按冻结顺序原子应用；
6. 七策略与 Oracle 读取同一冻结 bank；Test 不参与选模；
7. 原 cpp 接口、已有测试和 all-disabled/legacy 行为不变；
8. PPO-MLP 仍可运行，GPPO-NoGate/Adaptive 均可训练、保存和加载。

若依赖、接口或测试失败，停止对应长训练，先记录证据和最小修复方案。

## 12. 结果追溯与交付结构

每次运行生成 `run_manifest.json`，至少关联：git commit、Python/依赖锁、protocol hash、seed manifest hash、事件带 hash、模型配置、训练 seed、checkpoint hash、Validation 选择记录、原始逐事件日志与汇总脚本版本。

最终报告至少提供四张图：各算法累计 Region 空缺时间、各事件类型恢复延迟、相对 Oracle regret、通信量—覆盖质量 Pareto；图中明确区分 preliminary/formal 和 P1/P2。

本设计不预设 GPPO 必然优于 MLP-PPO。尤其在仅 4×4 的小图上，图结构优势可能有限；AHGNN 和 Gate 的价值必须由相同预算、NoGate 消融、严格事件配对和跨训练 seed 结果共同证明。
