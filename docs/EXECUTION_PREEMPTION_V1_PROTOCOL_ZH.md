# Execution-Preemption V1：执行中抢占与动态重分配协议

## 1. 目标与边界

本协议解决以下问题：任务执行到任意进度时再次发生新事件，系统必须确定继续、排队、暂停、抢占、迁移、终止或返航，并保证旧命令失效、任务进度可追溯、资源不重复占用。

V1 以 `GPPO-8.29 main@a4207527f713e6f15dcdbc538134aeaca28a03ac` 为只读基线，在 `research/execution-preemption-v1` 独立开发。旧 minimum-validation、正式评估和极端场景归档保持只读。

V1 不修改 PPO、GPPO、旧 reward 或旧训练合同。安全和抢占由确定性 `PreemptionController` 决定；学习策略未来只接收已确认、已仲裁的状态并输出 UAV—Task 重新分配。因为连续进度和抢占改变了环境与回报语义，后续算法比较必须重新冻结合同、重新训练，不能复用旧模型形成新结论。

机器可读合同见 [`configs/execution_preemption_v1.json`](../configs/execution_preemption_v1.json)。

## 2. 任务状态机

```text
PENDING → ASSIGNED → RUNNING
                       ├─ PAUSED
                       ├─ PREEMPTED
                       ├─ MIGRATING → RESUMING → RUNNING
                       ├─ COMPLETED
                       ├─ FAILED
                       └─ CANCELLED

PAUSED / PREEMPTED
  ├─ RESUMING → RUNNING
  ├─ MIGRATING → RESUMING → RUNNING
  └─ CANCELLED / FAILED
```

每个 `TaskRuntime` 必须保存 `task_id`、类型、优先级、deadline、状态、进度、剩余工作量、当前 UAV、是否可抢占、恢复策略、中断次数、开始时间和最后更新时间。每次状态或进度变化都追加不可丢失的 `TaskProgressRecord`。

进度语义冻结为：

```text
total_work = 1.0
progress ∈ [0, 1]
remaining_work = 1 - progress
progress += effective_work_rate × delta_time
```

同一 UAV 恢复默认保留 100% 进度；换 UAV 默认保留 90%，允许后续实验在 80%–100% 范围扫描。V1 固定切换时间成本为 0.25 个仿真时间单位。参数改变必须产生新合同版本。

## 3. UAV 运行时

`UAVRuntime` 至少保存：

- `energy_ratio`；
- `reserve_energy`；
- `estimated_rtb_energy`；
- `active_task_id`；
- `availability`；
- `communication_state`；
- `last_seen_at`；
- 可执行任务类型集合。

只有通信正常、资源空闲、任务类型兼容，且 `energy_ratio > reserve_energy + estimated_rtb_energy` 的 UAV 才能接收新任务。V1 的每个 tick 由调用方显式提供工作率和能耗，避免把未经标定的物理参数隐藏在运行时中。

## 4. 事件与仲裁

V1 新增八类业务事件：

```text
TASK_ARRIVAL
TASK_CANCELLED
TASK_PRIORITY_CHANGED
TASK_DEADLINE_CHANGED
UAV_LOW_ENERGY
UAV_COMM_LOST
UAV_COMM_RECOVERED
EXECUTION_FAILURE
```

事件等级：

| 等级 | 典型事件 | 默认动作 |
|---|---|---|
| P0 | UAV 损毁、碰撞风险、能源无法返航 | 安全动作立即中断任务 |
| P1 | 紧急短时任务、高价值目标 | 可抢占低优先级、可抢占任务 |
| P2 | 低能量、持续失联 | RTB、暂停或迁移 |
| P3 | 普通任务到达、优先级或 deadline 变化 | 空闲分配或排队 |
| P4 | 低价值状态更新 | 当前任务继续 |

同级事件严格按以下键排序：deadline 最早、任务优先级最高、信息年龄最小、接收时间最早、`event_id`。一批已确认事件只增加一次 `graph_version`，并按确定性顺序在同一事务副本上执行。

## 5. 抢占规则

1. 新任务到达且有安全、兼容的空闲 UAV：不打断现有任务，直接分配空闲 UAV。
2. P0/P1 新任务没有空闲 UAV：只能抢占优先级更低且 `preemptible=true` 的任务；优先选择任务优先级最低、进度最少者。
3. 普通或低价值新任务没有空闲 UAV：进入队列。
4. 不可抢占任务不会被普通新任务打断；P0 能源安全动作可以强制终止占用并返航。
5. UAV 低能量：保存进度、撤销占用、UAV 进入 `RETURNING`。
6. UAV 失联：若存在兼容安全替代 UAV 且任务允许迁移，则迁移；否则暂停并隔离失联 UAV。
7. 执行失败：优先迁移可恢复任务；不允许迁移或无替代资源时标记失败。
8. 通信恢复只恢复 UAV 的候选资格，不自动恢复旧任务，防止迟到报告使旧执行器复活。

## 6. 原子事务与并发不变量

标准流程：

```text
确认事件批次
  → 复制当前 ExecutionRuntime
  → graph_version + 1
  → 撤销受影响的未 ACK 命令
  → 确定性事件排序与仲裁
  → 保存进度并执行暂停/抢占/迁移/RTB
  → 校验唯一所有权和双向引用
  → 全部通过后一次性替换 live runtime
```

任何异常都会丢弃事务副本，live runtime 的图版本、任务进度和资源占用均保持不变。

必须始终满足：

- 一个排他任务最多有一个 active UAV；
- 一架 UAV 最多执行一个 active task；
- Task→UAV 与 UAV→Task 引用双向一致；
- 终态任务不保留 UAV；
- graph version 不匹配的命令或 ACK 必须拒绝；
- 低 fencing token 的迟到 ACK 必须拒绝；
- 同一 task 或 UAV 同时最多存在一个未 ACK 命令；
- 进度只能由 live task 的单一执行者累计一次。

## 7. V1 开发场景

`Dynamic-Preemption-Dev` 冻结为 10 类、每类 20 条 paired tapes，共 200 cases：

1. 搜索执行 40% 时出现紧急任务；
2. 搜索执行 90% 时出现低收益临时任务；
3. 不可抢占打击任务期间出现普通任务；
4. 执行 UAV 中途损毁；
5. 执行 UAV 电量不足，需要返航；
6. 通信延迟导致旧任务取消报告晚到；
7. 同一任务连续发生优先级变化；
8. 两个 P1 事件同时到达；
9. 新事件在重新分配推理期间再次到达；
10. 新事件在旧任务恢复过程中再次到达。

该开发集用于调试，不能充当 held-out。算法、参数和阈值冻结后，必须重新生成 seed、事件时间与参数范围均不重叠的 `Dynamic-Preemption-Hidden-V1`。

## 8. 指标与验收

算法比较至少包括旧方法、Greedy Priority、PPO-MLP、GPPO-Adaptive、Beam-MPC、PPO + Rule Arbiter、GPPO + Rule Arbiter。

主要指标为紧急任务 deadline miss、抢占响应时延、累计任务空缺、原任务恢复率、平均进度损失、能源安全违规、重复执行者、任务饥饿率、距离、负载以及 P95/P99 决策时延。

最低验收标准：

```text
资源冲突 = 0
旧命令复活 = 0
能源安全违规 = 0
P0 事件处理率 = 100%
紧急任务 deadline miss 相对旧方法降低 ≥ 10%
累计空缺相对旧方法降低 ≥ 10%
普通任务恢复率 ≥ 95%
```

规模按 4、8、16、32 UAV 逐级放大；前一级语义和一致性门未全部通过时，不进入下一级。任务数量按 UAV:Task 约 1:2 或 1:3 同步放大。

## 9. 实现阶段与当前状态

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| V1-A | 冻结状态机、事件等级、进度和一致性协议 | 已实现 |
| V1-B | `TaskRuntime`、`UAVRuntime`、`EventDecision` | 已实现 |
| V1-C | 确定性 `PreemptionController` 与原子事务 | 已实现，待完整场景扩展 |
| V1-D | 10×20 固定开发 tapes | 未开始 |
| V1-E | 接入 PPO/GPPO 重新分配 | 未开始 |
| V1-F | 重新训练与 4/8/16 UAV 比较 | 未开始 |
| V1-G | Hidden-V1 一次性验证与 evidence 分支 | 未开始 |

## 10. 停止条件

出现以下任一情况必须停止当前实验，不自动修改旧证据或启动训练：

- 必须修改旧 minimum-validation、旧 reward 或旧 checkpoint 才能继续；
- 无法在事务层保证唯一执行者、旧命令失效或进度单次累计；
- 需要让 PPO/GPPO 直接决定 P0 安全动作；
- 开发集与 hidden 集的 seed 或参数范围发生泄漏；
- 在协议、参数和测试未冻结前要求启动正式训练。

最关键的实现顺序保持不变：先证明规则化抢占机制正确，再让 PPO/GPPO 参与重新分配，最后才研究模型是否应该学习抢占。
