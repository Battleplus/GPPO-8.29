# Execution-Preemption V1 确定性基线冻结说明

> 状态：`FROZEN_FOR_BASELINE_IMPLEMENTATION`
>
> 证据性质：接口、确定性与安全回放；不是模型效果证据。

## 1. 统一边界

三种基线都只能接收 `AllocationRequest` 中已经通过通信、能量、任务兼容性和可用性筛选的候选 UAV。事件的 `CONTINUE / QUEUE / PREEMPT / MIGRATE / ABORT / RTB` 决策仍由 `PreemptionController` 完成。基线输出还必须通过统一的 request id、task id、graph version 和候选掩码校验。

因此，基线不能决定安全返航，不能绕过抢占规则，也不能把候选集之外的 UAV 写入事务。

## 2. 三种方法

| 方法 ID | 冻结语义 | 优点 | 局限 |
|---|---|---|---|
| `senior_legacy_method_v1` | 在安全候选集中按 UAV ID 词典序选择第一个 | 最简单、可复现、便于和旧式 first-legal 行为对照 | 不考虑能量裕量或未来任务；这是行为适配，不等同于学姐历史源码的逐行复现 |
| `greedy_priority_v1` | 事件和任务优先级先由规则控制器处理；候选 UAV 再按能量裕量降序、最近状态时间降序、UAV ID 排序 | 推理快、逻辑透明、对即时安全余量敏感 | 反应式，可能占用后续稀缺能力 |
| `beam_mpc_v1` | 在安全候选集上做 horizon=3、beam width=8 的确定性有限视野搜索；只读取运行时已知的 pending task forecast | 可以为即将到来的高优先级或稀缺能力任务保留资源 | 预测不准时可能无收益；规模增加时延更高；当前 smoke 不评价效果 |

Beam-MPC 的内部近似能耗只用于候选排序：

```text
planning_energy_cost = 0.05 + 0.15 × remaining_work
```

它不会修改环境真实能耗、reward、训练预算、seed 或 checkpoint grid。

## 3. 可区分性测试

合成请求冻结了三个安全候选：

- `U0`：词典序最先、能量较低、只支持 SEARCH；
- `U1`：中等能量、只支持 SEARCH；
- `U2`：能量最高，同时是唯一能处理下一项 URGENT 的候选。

结果应严格为：

```text
senior_legacy_method_v1 -> U0
greedy_priority_v1      -> U2
beam_mpc_v1             -> U1（保留 U2 给 URGENT）
```

这只证明三种决策逻辑不是换名字的同一实现，不证明 Beam-MPC 的任务效果更好。

## 4. 开发带回放

三种方法均回放固定 `Dynamic-Preemption-Dev`：10 场景 × 20 tapes = 200 cases，总计 600 次 allocator-tape runs。验收要求：

- 每个方法 200/200 tapes 完成；
- 每个方法 280 个事件决策；
- resource ownership、graph version、能量安全和旧命令不复活等运行时不变量全部通过；
- 相同输入重复生成相同 machine report；
- `training_started=false`；
- `model_effectiveness_evaluated=false`。

机器证据：[`baseline_replay_smoke.json`](../experiments/dynamic_preemption/dev_v1/baseline_replay_smoke.json)。

## 5. 结论边界

本阶段可以声称三种基线已经实现、语义冻结、通过相同安全 shell 并在 200 条开发事件带上无不变量失败。不能声称任何方法超过 PPO，也不能把开发回放当成 held-out 结果。真实性能比较必须等待新合同下的正式训练、训练证据封存和独立 Hidden-V1 一次性评估。
