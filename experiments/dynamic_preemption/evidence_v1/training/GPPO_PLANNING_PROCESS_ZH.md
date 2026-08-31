# GPPO 动态任务规划：能力边界、决策过程与效果观察

## 1. 结论先行

当前 `ExecutionGPPOAdaptive` 已经能够在 Execution-Preemption V1 合同内完成**事件触发的 UAV—Task 动态重新分配**：当任务临时增加、事件同时到达、UAV 状态变化或旧任务被抢占后，它会读取当前异构执行图，在合法动作集合中选择一个 UAV—Task 绑定或合法 `NOOP`。

这里的“规划”不是完整航迹规划。GPPO 当前不负责：

- 航迹生成、路径避障和连续控制；
- 直接决定 P0 安全动作、返航或强制抢占；
- 绕过动作掩码、版本、lease、fencing、ACK 或唯一所有权校验；
- 输出一个完整的多步任务序列。

更准确的定位是：

> 确定性控制器负责安全与可行性，GPPO 负责在合法候选中学习更合适的动态任务分配。

## 2. 一次规划是怎样发生的

```mermaid
flowchart LR
    A[运行中的 UAV 与任务] --> B[突发事件或状态变化]
    B --> C[PreemptionController]
    C -->|CONTINUE / QUEUE / PREEMPT / ABORT / RTB / MIGRATE| D[ExecutionRuntime 原子事务]
    D --> E[构建五节点异构图]
    E --> F[生成 UAV—Task 候选和动作掩码]
    F --> G[GPPO 关系消息传递与候选边评分]
    G --> H{选择合法动作}
    H -->|UAV—Task| I[proposal]
    H -->|NOOP| I
    I --> J[版本、租约、fencing、所有权校验]
    J -->|PASS| K[原子提交新分配]
    J -->|FAIL| L[拒绝提交并保留原状态]
```

规划过程分为五步：

1. **事件到达**：例如紧急任务插入、UAV 损毁、低能量、通信延迟或连续优先级变化。
2. **规则裁决**：确定性控制器先决定是否继续、排队、抢占、终止、返航或迁移。P0 和返航等安全语义不交给学习策略。
3. **图状态编码**：运行时把 UAV、任务、区域、目标、事件及其关系编码成异构图。
4. **合法候选评分**：动作掩码先排除无能力、低安全余量、状态冲突或其他非法组合；GPPO 只比较剩余候选。
5. **校验并提交**：被选动作必须通过图版本、lease、fencing、ACK 和唯一所有权检查，之后才改变真实运行状态。

## 3. GPPO 实际读取了什么

模型使用五类节点：

```text
UAV、Task、Region、Target、Event
```

使用七类冻结关系：

```text
UAV --executes--> Task
UAV --can_execute--> Task
Task --located_in--> Region
Task --depends_on--> Task
Event --affects--> UAV
Event --affects--> Task
Task --preempts--> Task
```

每类节点先由独立编码器映射到隐藏表示，然后经过两层 `AdaptiveRelationLayer`。每层会聚合邻居消息，并通过门控结构在“保留当前状态”和“接受关系消息”之间自适应融合。

动作头不是从整个图直接输出一个抽象编号，而是为每条合法的候选 UAV—Task 边评分：

```text
候选分值 = edge_actor(UAV 表示, Task 表示, 规则上下文)
```

另有独立的 `NOOP` 动作头。最终 logits 会再经过同一动作掩码，非法动作的概率质量不能进入实际选择。

## 4. 示例：紧急任务临时插入

假设当前状态为：

- `U1` 正在执行普通搜索任务 `T-old`，完成度 40%，能量 72%；
- `U2` 当前可接管新任务，能量 61%，位置更接近目标区域；
- `U3` 能量 28%，低于执行新任务所需安全余量；
- 新的 P1 任务 `T-urgent` 临时到达。

系统不会让 GPPO 自行决定是否违反规则抢占。首先由控制器判断该事件是否需要 `PREEMPT` 或 `QUEUE`。若仲裁结果允许重新分配，运行时可能生成：

```text
U1 → T-urgent    合法
U2 → T-urgent    合法
U3 → T-urgent    非法：安全余量不足
NOOP             视合同状态决定是否合法
```

GPPO 的规划作用体现在合法候选之间：

- `U1 → T-urgent` 可能响应较快，但会增加旧任务中断和恢复成本；
- `U2 → T-urgent` 可能减少进度损失，并利用更好的位置或能力关系；
- `U3 → T-urgent` 即使网络原始偏好较高，也会被动作掩码排除；
- `NOOP` 只有在不违反当前请求语义时才允许存在。

如果 GPPO 选择 `U2 → T-urgent`，proposal 仍需通过提交前校验。校验成功后，新任务才真正绑定到 `U2`；旧任务的进度和状态由原子事务保存，等待后续恢复。

这个例子展示的是规划机制，不代表现有 checkpoint 已经在正式 held-out 数据上证明 `U2` 一定优于其他选择。

## 5. Reactive 与 Rule-Arbiter 两种 GPPO

| 方法 | 输入差异 | 研究问题 |
|---|---|---|
| `gppo_adaptive_reactive_v1` | 异构执行图；规则上下文置零 | 仅依靠图状态和事件关系，能否学会动态分配？ |
| `gppo_adaptive_rule_arbiter_v1` | 异构执行图 + 冻结的 `EventDecision` 上下文 | 明确知道仲裁器刚做了什么后，能否减少事件语义歧义？ |

Rule-Arbiter 版本不会修改规则，也不会获得新的安全权限。它只是把已经确定的决策类型、事件等级、置信度、信息时效和 displaced task 等信息作为只读输入。

## 6. 极端场景中应该观察什么

| 场景 | GPPO 可发挥的作用 | 必须由规则/运行时保证 | 后续效果指标 |
|---|---|---|---|
| 多个紧急事件同时到达 | 利用 Event—Task—UAV 关系区分接管组合 | 事件排序、P0 优先和原子批处理 | deadline miss、累计空缺、响应时延 |
| 执行过程中再次到达新事件 | 根据更新后的图重新选择合法绑定 | graph version、过期 proposal 拒绝 | stale proposal、重复分配、恢复率 |
| UAV 损毁或通信中断 | 为受影响任务选择替代 UAV | 失效隔离、旧命令不能复活 | 迁移成功率、进度损失、空缺时间 |
| 低能量与紧急任务冲突 | 在剩余安全候选中选择代价更小的 UAV | 能量硬门禁、RTB 决策 | 能源违规必须为 0、任务完成质量 |
| 普通任务长期被连续抢占 | 平衡紧急响应与恢复旧任务 | 保存进度和唯一所有权 | starvation、普通任务恢复率 |
| 信息延迟或置信度不足 | 结合事件节点和上下文降低错误重分配 | 时效阈值、置信度门禁 | 错误迁移、无效切换、推理时延 |
| UAV 数由 4 扩展到 8/16 | 利用关系结构减少固定槽位表达压力 | 候选生成和容量约束 | 规模一致性、P95/P99 推理时延 |

## 7. 和 PPO、Greedy、Beam-MPC 的区别

| 方法 | 主要优势 | 主要不足 | 更可能适合的情况 |
|---|---|---|---|
| PPO-MLP | 结构简单、推理链短 | 关系需要从大向量中间接学习 | 状态规模较小、结构关系较弱 |
| GPPO-Adaptive | 显式建模 UAV—Task—Event 关系；更自然地适配关系变化 | 计算更复杂，可能有更高训练方差和时延 | 多任务、多事件、依赖与抢占关系明显 |
| Greedy | 可解释、速度快、无需训练 | 容易短视，难联合考虑恢复与长期代价 | 简单单事件、规则足以决定的场景 |
| Beam-MPC | 显式三步前瞻，可输出搜索 trace | 候选随规模增长，计算成本可能上升 | 短期未来比较可信、需要显式前瞻 |

当前 GPPO 仍属于事件触发的单步策略，而不是显式多步搜索。`beam_mpc_v1` 才是当前合同中的 horizon=3 前瞻规划基线。

因此不能仅凭模型结构判断 GPPO 必然更好：

- 简单单事件场景中，Greedy 可能已经足够强；
- 小规模状态中，PPO-MLP 可能更直接；
- 未来事件可预测时，Beam-MPC 可能利用前瞻信息；
- GPPO 最有希望体现价值的是多关系、多事件、状态频繁变化和部分信息不全的场景。

## 8. 当前已经证明的内容

截至本文提交前，已经封存的正式训练证据为 `UAV=4、seed=1101`：

```text
4/4 learned runs PASS
200,000 accepted decision steps
8/8 checkpoints（每个 run 为 25k、50k）
checkpoint SHA-256、Gate、provenance 和 stderr 复验 PASS
Validation / Freeze / Test / held-out 均未启动
```

其中两个 GPPO 方法都完成了精确 50,000 accepted decision steps，并生成了 25k、50k checkpoint。这证明：

- 模型前向、反向和 PPO 优化链路可运行；
- 同一动作掩码能保证最终动作合法；
- checkpoint 可以按训练合同生成并通过完整性复验；
- source、Gate、训练合同和 seed provenance 能够闭环。

`UAV=4、seed=2202` 的一次 GPPO-Reactive worker 在 `39,656/50,000` steps 处因 Windows 正常关机中断，stderr 为 0。原始 partial 产物已保留，未自动续训、修补或启动后续评估。该事件说明 campaign 尚未完成，不构成 GPPO 算法异常证据。

## 9. 当前尚未证明的内容

以下结论目前都不能对外宣称：

- GPPO 的任务效果已经优于 PPO；
- GPPO 已经超过 Greedy、Beam-MPC 或旧方法；
- Rule-Arbiter 输入已经带来显著提升；
- 4 UAV 的结果能够泛化到 8、16 或 32 UAV；
- 训练 reward 等于真实规划效果；
- 25k 与 50k checkpoint 中可以挑选较好者。

原因是正式 held-out evaluation 尚未执行，而且合同明确禁止 checkpoint selection。后续只能固定使用每个 run 的 50k checkpoint，在 paired held-out tapes 上统一比较。

## 10. 后续如何验证规划效果

完整训练证据封存后，规划效果应按以下顺序验证：

1. 固定每个 run 的 50k checkpoint，不比较或选择 25k；
2. 在隔离的 Freeze/evaluation worktree 中加载固定 checkpoint；
3. 所有方法使用相同的 held-out paired tapes；
4. 同时比较 PPO、GPPO、Greedy、Beam-MPC 和旧方法；
5. 统计紧急 deadline miss、抢占响应时延、累计空缺、恢复率、进度损失、starvation、能耗、航程、负载差以及 P95/P99 推理时延；
6. 资源冲突、旧命令复活和能源安全违规必须严格为 0；
7. 保留所有失败 case，不允许从统计中删除。

只有完成上述固定评估，才能把“能正常生成合法规划”升级为“规划效果优于某个基线”。

## 11. 代码与证据索引

- [GPPO/PPO 模型实现](../../../../execution_preemption/policy_models.py)
- [Gym rollout 与动作提交链](../../../../execution_preemption/gym_env.py)
- [五节点异构执行图](../../../../execution_preemption/graph.py)
- [动作适配与动作掩码](../../../../execution_preemption/adapter.py)
- [确定性抢占控制器](../../../../execution_preemption/controller.py)
- [原子执行运行时](../../../../execution_preemption/runtime.py)
- [正式训练合同](../../../../configs/execution_training_contract_v1.json)
- [Execution-Preemption V1 协议](../../../../docs/EXECUTION_PREEMPTION_V1_PROTOCOL_ZH.md)
- [当前正式训练说明](TRAINING_EXPLAINER_ZH.md)
- [当前训练进度与中断边界](CURRENT_TRAINING_PROGRESS_ZH.md)
- [UAV4/seed1101 机器可读训练证据](TRAINING_STAGE_UAV04_SEED1101.json)

适合向导师或学姐汇报的一句话是：

> 当前 GPPO 已经完成“安全规则约束下的动态任务重新分配”工程闭环，能够读取 UAV—Task—Event 异构关系并输出合法分配；但目前证明的是可运行性、合法性和训练证据完整性，是否优于 PPO、Greedy 或 Beam-MPC，必须等待完整训练封存后使用固定 50k checkpoint 做统一 held-out 比较。
