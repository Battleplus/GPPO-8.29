# Execution-Preemption V1：算法分配边界

## 结论

V1 已将“是否继续、抢占、终止或返航”与“从安全候选集中选择哪架 UAV”分离。安全仲裁继续由确定性规则控制，旧方法、Greedy、PPO、GPPO 或 Planner 只能通过同一个版本化接口选择 UAV—Task 分配，不能绕过能量、通信、任务兼容性、graph version 或唯一所有权约束。

这一步只完成算法接入边界，没有把旧 PPO/GPPO checkpoint 直接接入新环境，也没有进行模型效果比较。连续进度、抢占和迁移已经改变环境语义，正式比较前仍须冻结新图、新 reward、新训练预算和 seed manifest，并重新训练模型。

## 接口

```text
Confirmed Runtime Event
        ↓
PreemptionController
        ├─ 规则决定 CONTINUE / QUEUE / PREEMPT / ABORT / RTB
        ↓
build_allocation_request()
        ├─ 固定 graph_version
        ├─ 固定 task_id 与 decision_type
        └─ 只暴露安全候选 UAV 集合
        ↓
Allocator.propose(request)
        ↓
validate_proposal()
        ├─ request_id 精确匹配
        ├─ graph_version 精确匹配 live runtime
        ├─ task_id 精确匹配
        └─ selected_uav 必须属于冻结候选集
        ↓
ExecutionRuntime 原子提交
```

## 已有分配器

| 分配器 | 用途 | 说明 |
|---|---|---|
| `FirstAvailableAllocator` | 旧式确定性基线 | 按 UAV ID 选择第一个安全候选 |
| `MaxEnergyMarginAllocator` | 轻量 Greedy 基线 | 选择扣除 reserve 与 RTB 需求后能量裕量最大的 UAV |
| `CallbackAllocator` | PPO/GPPO/Planner 接口 | 接收冻结请求并返回 proposal，结果仍须经过统一校验 |

`CallbackAllocator` 不是“信任模型”的逃生口。测试已证明回调选择失联、低能量或不在候选集中的 UAV 时，整个事件事务回滚，graph version 和任务所有权均不改变。

## 安全边界

算法不能决定或覆盖：

- P0/P2 能源安全与 RTB；
- 任务是否可抢占；
- 旧命令撤销、ACK 和 fencing；
- graph version；
- 一个 UAV 同时最多执行一个任务；
- 一个排他任务同时最多拥有一个执行者；
- 迁移进度保留率和切换成本。

算法可以决定：

- 在控制器已经允许分配或迁移之后，从冻结的安全候选集合中选择一个 UAV；
- 返回算法标识、分数和非安全性元数据，供后续证据记录。

## 当前验证

- Execution-Preemption 专项测试：34/34 PASS；
- 两个确定性分配器共享同一 200 条 development tapes；
- 共完成 400 个 allocator-tape runs；
- 每个分配器处理 280 个事件决策、80 个版本化分配请求；
- invariant failures：0；
- training started：false；
- model effectiveness evaluated：false。

机器可读结果见 [`allocator_replay_summary.json`](../experiments/dynamic_preemption/dev_v1/allocator_replay_summary.json)。两个确定性分配器在当前 tapes 上选择计数相同，这只说明当前测试集用于接口和一致性验证，不能据此判断二者效果相同。性能差异必须在包含距离、能耗、deadline 和动态任务负载的重新训练合同中评估。

## 下一步

1. 冻结 `UAV / Task / Region / Target / Event` 五类节点及关系 schema；
2. 冻结连续进度、抢占成本、deadline、能源与任务恢复的新 reward/metric 合同；
3. 实现 PPO/GPPO observation adapter，但保持 action mask 与 proposal validator 不可绕过；
4. 在 4 UAV 规模做训练前 smoke，再决定 8/16 UAV 扩展；
5. 所有模型重新训练后，才在同一 development bank 上做开发比较；
6. 算法和阈值冻结后生成独立 Hidden-V1，一次性评估。
