# Execution-Preemption V1：当前正式训练说明与作用分析

## 1. 一句话说明

本阶段训练的是**动态多任务、连续执行和突发事件条件下的 UAV—Task 重新分配策略**。模型不直接决定 P0 安全动作、是否返航、是否允许抢占，也不能绕过资源唯一性约束；这些安全关键语义由冻结的确定性仲裁器和原子事务层负责。学习策略只在仲裁后生成的合法候选集合中选择“哪架 UAV 接哪个任务”。

因此，这套实验要回答的不是“强化学习能否取代安全规则”，而是：

> 在安全规则固定、事件输入相同、动作空间相同的前提下，图结构、事件仲裁上下文和 PPO 学习能否改善动态任务重新分配的效率与恢复能力？

## 2. 系统中模型实际处于什么位置

```text
连续执行中的任务与 UAV 状态
          ↓
突发事件到达并按优先级排序
          ↓
确定性 PreemptionController
决定 CONTINUE / QUEUE / PREEMPT / ABORT / RTB / MIGRATE
          ↓
ExecutionRuntime 原子事务
保存进度、撤销旧命令、更新 graph_version、建立合法候选集
          ↓
PPO / GPPO 学习策略
只选择一个合法的 UAV—Task 绑定，或合法 NOOP
          ↓
版本、lease、fencing、ACK 和唯一所有权再次校验
          ↓
提交新分配，计算软指标 reward；硬安全违规直接停止
```

这条边界很重要：Reactive 版本也不是“无规则”或“不安全”版本。所有七种比较方法都必须经过同一安全壳和动作掩码。Reactive 与 Rule-Arbiter 的区别，只是模型是否能看到已经冻结的 `EventDecision` 上下文。

## 3. 正在训练的四种学习方法

| 方法 | 状态表达 | 是否看到仲裁上下文 | 主要研究作用 |
|---|---|---:|---|
| `ppo_mlp_reactive_v1` | 37,976 维定长扁平向量 | 否 | 非图 PPO 基线；测量普通 MLP 在动态分配中的能力 |
| `gppo_adaptive_reactive_v1` | 五类节点异构图 | 否 | 与 PPO-Reactive 对比，隔离“图结构表达”的增益 |
| `ppo_mlp_rule_arbiter_v1` | 扁平向量 + 冻结仲裁上下文 | 是 | 与 PPO-Reactive 对比，隔离“规则上下文”的增益 |
| `gppo_adaptive_rule_arbiter_v1` | 五类节点异构图 + 冻结仲裁上下文 | 是 | 检验图关系建模与规则上下文能否形成互补 |

四种方法共享：

- 完全相同的合法动作空间，容量为 3,073；
- 完全相同的事件 tapes、训练 seed、reward、50k 预算和 checkpoint 网格；
- 完全相同的安全动作掩码；
- 完全相同的 PPO 优化参数；
- 完全相同的正式 Gate、source/protocol provenance 和停止策略。

### 3.1 PPO-MLP Reactive

它把五类节点、节点存在位、关系矩阵和零值规则上下文编码为固定长度向量，再通过两层 64 维 `Tanh` MLP 输出动作 logits 和状态价值。

作用：提供最直接的学习基线。如果它已经能稳定完成任务，说明问题可能不需要复杂图网络；如果规模增加后明显退化，而 GPPO 更稳定，则能支持“显式关系建模有价值”的判断。

局限：固定槽位和大向量会引入较多填充；UAV—Task—Event 的结构关系需要由 MLP 间接学习，规模迁移能力通常更难保证。

### 3.2 GPPO-Adaptive Reactive

它使用五类节点：

```text
UAV、Task、Region、Target、Event
```

以及七类冻结关系：

```text
UAV --executes--> Task
UAV --can_execute--> Task
Task --located_in--> Region
Task --depends_on--> Task
Event --affects--> UAV
Event --affects--> Task
Task --preempts--> Task
```

模型先分别编码不同节点类型，再进行两层关系消息传递；每种节点通过自适应门控融合原状态与邻居消息。动作头直接读取候选 UAV、Task 和上下文表示。

作用：显式表达“谁能执行什么、事件影响谁、任务之间是否依赖或抢占”的结构。它最值得观察的是 UAV 数从 4 增加到 8、16 时，是否比定长 MLP 更能保持分配一致性和样本效率。

局限：计算链路更复杂，关系消息可能引入训练方差；若任务主要由简单优先级决定，图网络不一定优于 MLP。因此不能只凭结构更复杂就预设它必然更好。

### 3.3 PPO-MLP + Rule Arbiter

网络骨干仍是 PPO-MLP，但输入中加入冻结 `EventDecision` 上下文，例如事件等级、决策类型、信息时效、置信度以及已确定的 displaced task 等编码。

作用：检验模型是否因为“知道仲裁器刚刚做了什么”而更容易选择合适的接管 UAV。它可以减少部分可观测性和事件语义歧义，同时保持非图网络结构不变。

局限：策略可能过度依赖规则输出；如果未来仲裁协议改变，模型需要按新合同重新训练。Rule-Arbiter 版本不是让模型修改规则，而是给模型增加只读上下文。

### 3.4 GPPO-Adaptive + Rule Arbiter

这是五节点异构图、关系消息传递和冻结规则上下文的组合。理论假设是：规则层明确“为什么需要重新分配”，图层表达“哪些 UAV 与任务最适合重新组合”。

作用：验证“安全规则负责可行性，图策略负责效率”的分层方案是否优于单独使用图结构或单独提供规则上下文。

局限：这是参数与结构最复杂的方法；若它只在训练 tapes 上更好而 hidden tapes 不好，则可能存在过拟合。最终必须依靠固定 50k checkpoint 的 paired held-out 结果，而不是训练 reward 判断。

## 4. 哪些方法不需要训练

完整比较共七种方法，另外三种是确定性基线：

| 方法 | 类型 | 作用 |
|---|---|---|
| `senior_legacy_method_v1` | 适配后的旧方法 | 代表原有工作流程的参考下限；不是声称与学姐原实现逐行等价 |
| `greedy_priority_v1` | 贪心规则 | 判断复杂学习方法是否真正超过“优先级 + 能量余量 + 信息新鲜度” |
| `beam_mpc_v1` | Beam-MPC，horizon=3、width=8 | 提供有限前瞻规划基线，判断学习策略相对短视搜索的优势与代价 |

这三种方法不会生成训练 checkpoint，但会在后续使用与学习方法相同的 paired tapes、动作掩码和安全壳进行比较。

## 5. 训练数据是什么

正式训练使用独立命名空间：

```text
execution_preemption_v1/train
```

它不会复用 `Dynamic-Preemption-Dev`，也不会接触尚未生成的 `Dynamic-Preemption-Hidden-V1`。

每个训练 episode 根据以下不可混淆身份确定性生成：

```text
训练合同 / namespace / policy seed / UAV 数 / episode index
```

训练 tapes 覆盖十类动态事件模式，包括紧急任务插入、低收益临时任务、不可抢占任务、UAV 损毁、低能量返航、通信延迟、连续优先级变化、同时到达事件、推理期间再次到达事件以及恢复期间再次到达事件。任务规模在每架 UAV 对应 2 或 3 个任务之间交替。

这种设计的作用是：同一方法可以看到大量确定性但不重复的动态组合；不同方法在相同 seed 和规模下使用同一生成规则，从而减少比较中的输入偏差。

## 6. Reward 在鼓励什么

所有软信号先归一化到 `[0, 1]`，reward 为固定加权和：

| 信号 | 权重 | 含义 |
|---|---:|---|
| weighted progress gain | +4.0 | 鼓励高价值任务产生真实进度 |
| urgent deadline miss rate | -10.0 | 强烈惩罚紧急任务超期 |
| weighted vacancy time | -5.0 | 减少任务长期无人执行 |
| progress loss | -4.0 | 减少迁移和抢占造成的进度损失 |
| starvation exposure | -3.0 | 避免普通任务长期饥饿 |
| switch time | -1.0 | 避免无意义频繁切换 |
| energy consumed | -1.0 | 控制能耗 |
| normalized distance | -0.5 | 减少额外航程 |
| load gap | -0.5 | 改善 UAV 间负载均衡 |

以下三项不是可以拿效益交换的 reward 罚项，而是硬失败：

```text
resource conflicts
stale command resurrections
energy safety violations
```

只要出现任意硬安全违规，该 transition 就不具备学习资格，run 必须停止。这样可以避免模型用“更高任务收益”抵消安全错误。

## 7. 正式训练合同

```text
学习方法：4
Seeds：1101、2202、3303
训练规模：4、8、16 UAV
每个 run：50,000 accepted decision steps
Checkpoints：25,000、50,000
正式 runs：4 × 3 × 3 = 36
正式 checkpoints：36 × 2 = 72
固定评估对象：每个 run 的 50k checkpoint
Checkpoint selection：禁止
32 UAV：只做 16-UAV/50k checkpoint 的 zero-shot 扩展性检查，不重新训练
```

优化器与 PPO 参数：Adam，学习率 `3e-4`，`gamma=0.99`，`GAE lambda=0.95`，clip `0.2`，每 64 个 accepted decision steps 更新，4 个 update epochs，最大梯度范数 `0.5`。

每个 checkpoint 同时保存模型、优化器、Python/NumPy/Torch RNG 状态以及 source、Gate、训练合同 provenance。正式 runner 要求全新空目录，不支持 resume，也禁止复用旧 campaign 或旧 checkpoint。

## 8. 这套训练能分析出什么

四种学习方法构成一个清晰的 `2 × 2` 消融矩阵：

| 对比 | 可以回答的问题 |
|---|---|
| PPO-Reactive vs GPPO-Reactive | 只增加图结构后，关系建模是否有增益？ |
| PPO-Reactive vs PPO+Rule | 只增加仲裁上下文后，事件语义是否更容易利用？ |
| GPPO-Reactive vs GPPO+Rule | 图策略是否还能从规则上下文中获得额外增益？ |
| PPO+Rule vs GPPO+Rule | 在规则信息相同的条件下，图表达是否仍然重要？ |
| 四学习方法 vs Greedy | 学习是否超过简单规则？ |
| 四学习方法 vs Beam-MPC | 学习是否超过短视规划，并在时延上更合适？ |
| 三个 seeds | 差异是否稳定，而不是偶然初始化结果？ |
| 4→8→16 UAV | 方法是否随规模增长保持一致性和效率？ |

预期但尚未验证的研究假设是：

1. 4 UAV 时，PPO-MLP 可能已具有竞争力，因为状态规模较小；
2. UAV 和任务增多后，GPPO 对显式关系的利用可能更有优势；
3. Rule-Arbiter 上下文可能降低事件后的分配歧义，尤其对迁移、抢占和恢复场景有帮助；
4. GPPO+Rule 可能在复杂多事件场景中最好，但也最需要防范过拟合和推理时延增加；
5. Greedy 可能在简单单事件场景中非常强，因此学习方法必须在连续、多事件和信息不全场景中证明额外价值。

这些都只是待检验假设，不是当前实验结论。

## 9. 最终如何判断“有作用”

不能只看训练 reward。训练全部封存后，必须固定使用 50k checkpoint，在完全隔离的 held-out paired tapes 上比较：

- 紧急任务 deadline miss rate；
- 抢占响应时延；
- 累计加权任务空缺；
- 普通任务恢复率；
- 累计进度损失；
- 任务饥饿率；
- 能耗、航程和负载差；
- 平均、P95、P99 推理时延；
- 资源冲突、旧命令复活和能源安全违规是否严格为 0。

最低目标为：P0 处理率 100%，三项硬安全违规为 0，紧急 deadline miss 和累计空缺相对旧方法至少降低 10%，普通任务恢复率至少达到 95%。失败 case 不允许从统计中删除。

## 10. 当前进度与证据边界

截至 2026-08-30，已经封存的完整阶段为：

```text
UAV=4，seed=1101
4/4 learned runs PASS
总 accepted decision steps = 200,000
8/8 checkpoints（25k、50k）
checkpoint SHA-256、run inventory、Gate、provenance、stderr 全部复验 PASS
```

机器可读证据见 [`TRAINING_STAGE_UAV04_SEED1101.json`](TRAINING_STAGE_UAV04_SEED1101.json)，对应 evidence commit 为 `0488166f40ccbeb83ea46a9d0c9551f975ddf4ee`。

`UAV=4、seed=2202` 批次中，PPO-Reactive 已完成 50k；GPPO-Reactive 因 Windows 正常关机中断在 39,656/50,000 steps，stderr 为 0，后续两个 Rule-Arbiter workers 未启动。当前没有训练 worker 运行，也没有自动续训。该批次在全新 namespace 中完成必要重跑、达到 4/4 runs、8/8 checkpoints 并通过只读复验之前，不计入已封存进度。详细状态见 [`CURRENT_TRAINING_PROGRESS_ZH.md`](CURRENT_TRAINING_PROGRESS_ZH.md)。

当前可以得出的结论只有：

- 新训练合同、正式 runner、Gate 和 checkpoint provenance 已实际闭环；
- seed=1101 的四种学习方法都能精确完成 50k 训练并生成可复验 checkpoint；
- 训练过程中没有启动 Validation、Freeze、Test 或 held-out evaluation。

当前**不能**得出的结论包括：

- GPPO 已经优于 PPO；
- Rule-Arbiter 版本已经提高任务效果；
- 任一模型已经超过 Greedy、Beam-MPC 或旧方法；
- 训练 reward 可以代表泛化效果；
- 25k 或 50k 中哪一个更好——本合同禁止 checkpoint selection，后续固定使用 50k。

## 11. 相关文件

- [正式训练合同](../../../../configs/execution_training_contract_v1.json)
- [执行中抢占协议](../../../../docs/EXECUTION_PREEMPTION_V1_PROTOCOL_ZH.md)
- [训练 runner](../../../../execution_preemption/training.py)
- [PPO/GPPO 模型](../../../../execution_preemption/policy_models.py)
- [五节点异构图](../../../../execution_preemption/graph.py)
- [冻结 reward](../../../../execution_preemption/reward.py)
- [训练 tapes](../../../../execution_preemption/training_tapes.py)
- [UAV4/seed1101 机器可读训练证据](TRAINING_STAGE_UAV04_SEED1101.json)
- [GPPO 动态任务规划：能力边界、决策过程与效果观察](GPPO_PLANNING_PROCESS_ZH.md)
- [当前训练进度与中断边界](CURRENT_TRAINING_PROGRESS_ZH.md)

最重要的汇报口径是：

> 我们没有让神经网络接管安全规则，而是在确定性抢占与原子事务已经保证安全的前提下，训练 PPO/GPPO 优化事件后的 UAV—Task 重新分配。四种学习方法通过“是否使用图结构 × 是否提供规则上下文”的消融矩阵，分别检验关系建模和事件语义信息的作用；正式效果结论必须等 36 个 runs 全部封存后，固定使用 50k checkpoint 在 held-out paired tapes 上一次性得出。
