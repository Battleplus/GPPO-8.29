# Dynamic-Preemption-Dev 规则回放结果

## 结论

`PreemptionController + ExecutionRuntime` 已在 10 类、200 条固定开发 tapes 上完成确定性回放，200/200 cases PASS，共处理 280 个事件决策。所有回放都满足任务与 UAV 唯一所有权、双向引用一致、旧事件幂等和进度单次累计约束。

本结果只证明 V1 规则机制和事务语义可以闭环，不代表 PPO、GPPO 或 Planner 的效果，也不是 hidden/held-out 结论。

## 决策覆盖

| 决策 | 次数 |
|---|---:|
| CONTINUE | 120 |
| PREEMPT | 40 |
| QUEUE | 40 |
| RTB | 40 |
| MIGRATE | 20 |
| ABORT | 20 |
| 合计 | 280 |

## 场景覆盖

| 场景 | Cases | 预期规则结果 |
|---|---:|---|
| urgent_at_40 | 20 | 20 PREEMPT |
| low_value_at_90 | 20 | 20 QUEUE |
| nonpreemptible_strike | 20 | 20 QUEUE |
| execution_uav_destroyed | 20 | 20 MIGRATE |
| low_energy_rtb | 20 | 20 RTB |
| delayed_task_cancel | 20 | 20 ABORT |
| repeated_priority_change | 20 | 40 CONTINUE |
| simultaneous_p1 | 20 | 40 CONTINUE |
| event_during_inference | 20 | 20 CONTINUE + 20 RTB |
| event_during_resume | 20 | 20 CONTINUE + 20 PREEMPT |

## 完整性

- `manifest.status = PASS`；
- `tape_count = 200`；
- `replayed_tape_count = 200`；
- `decision_count = 280`；
- `invariant_failures = 0`；
- 200 个 tape 文件 SHA-256 全部重新计算匹配；
- `training_started = false`；
- `checkpoint_selection = false`。

## 下一步边界

下一阶段可以把已确认事件后的“UAV—Task 选择”接口接入 PPO/GPPO/Beam-MPC，但 P0 安全动作、RTB、旧命令撤销和 fencing 仍必须由规则与事务层控制。接入会改变训练环境与 reward 语义，因此必须使用新的训练合同和新 checkpoint，禁止复用旧 minimum-validation 模型形成新算法结论。
