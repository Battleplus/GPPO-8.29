# 极端多事件场景探索性压力测试

> 这是 post-hoc exploratory stress test，不属于 minimum-validation held-out 证据，不能用于 checkpoint selection 或正式优越性声明。

- 固定 50k checkpoints：6/6，运行前 SHA-256 全部匹配
- 场景数：7
- 每场景事件 tapes：6
- 模型配对：2 variants × 3 training seeds
- 额外参照：Nearest Legal、Min Load、Greedy Cost、Current-Pending Exact Planner

## 场景设计

| 场景 | 设计 | 主要压力 |
|---|---|---|
| atomic_triple_shock（原子三重冲击） | UAV 损毁、目标发现、区域空缺在同一观测批次到达，随后目标释放与再次空缺。 | atomic_batch, coupled_reallocation, task_release |
| resource_collapse（三级资源坍缩） | 连续损毁三架 UAV，仅剩一架承担全部搜索任务，再连续制造区域空缺。 | resource_scarcity, rapid_damage, near_capacity_limit |
| tracking_saturation_release（跟踪饱和后延迟释放） | 三架 UAV 依次转入 TRACK，最后一架搜索 UAV 损毁；系统等待目标销毁报告释放一架 UAV。 | temporary_infeasibility, task_contention, delayed_release |
| out_of_order_reports（因果报告乱序） | 目标销毁报告先于目标发现报告到达，并交错 UAV 损毁与区域空缺报告。 | partial_observation, out_of_order, causal_inversion |
| long_blind_burst（长盲区突发批次） | 五个真实事件在 0.08 秒内发生，但全部延迟到 30 秒后才作为一个批次被观察。 | long_information_gap, atomic_batch, stale_world_model |
| task_churn（任务反复变更） | 目标发现与销毁快速交替，UAV 在 SEARCH/TRACK/释放之间频繁切换，最后再出现区域空缺。 | task_churn, rapid_release, switching_pressure |
| event_storm_8（八事件持续风暴） | 把有效事件序列压缩为八个高度重叠的观测，超过原协议每回合五事件的密度。 | long_horizon, dense_overlap, distribution_shift |

## PPO-MLP 与 GPPO-Adaptive 的场景均值

| 场景 | 方法 | 成功率 | 覆盖率 | 恢复延迟 | 累计空缺 | 距离 | 负载差 | 回报 | 最终不可行率 | 推理 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| atomic_triple_shock | PPO-MLP | 1.0000 | 1.0000 | 1.4000 | 6.3410 | 0.0859 | 0.5417 | 19.1044 | 0.0000 | 2.7920 |
| atomic_triple_shock | GPPO-Adaptive | 1.0000 | 1.0000 | 1.3889 | 6.5080 | 0.0862 | 0.5718 | 19.1324 | 0.0000 | 14.9806 |
| resource_collapse | PPO-MLP | 1.0000 | 1.0000 | 2.4867 | 9.0127 | 0.2014 | 0.2000 | 16.3379 | 0.0000 | 1.8541 |
| resource_collapse | GPPO-Adaptive | 1.0000 | 1.0000 | 2.4089 | 10.2942 | 0.2165 | 0.2000 | 17.4800 | 0.0000 | 13.8370 |
| tracking_saturation_release | PPO-MLP | 1.0000 | 1.0000 | 6.9667 | 110.3067 | 0.0803 | 0.3006 | 33.1817 | 0.0000 | 2.1110 |
| tracking_saturation_release | GPPO-Adaptive | 1.0000 | 1.0000 | 6.2889 | 97.4899 | 0.1044 | 0.2822 | 33.7914 | 0.0000 | 14.4855 |
| out_of_order_reports | PPO-MLP | 1.0000 | 1.0000 | 0.8667 | 0.6678 | 0.1031 | 0.3750 | 18.3658 | 0.0000 | 1.7958 |
| out_of_order_reports | GPPO-Adaptive | 1.0000 | 1.0000 | 0.8778 | 0.6679 | 0.1064 | 0.3681 | 18.3296 | 0.0000 | 13.5584 |
| long_blind_burst | PPO-MLP | 1.0000 | 1.0000 | 1.7000 | 10.3470 | 0.0181 | 0.5938 | 13.2033 | 0.0000 | 1.8033 |
| long_blind_burst | GPPO-Adaptive | 1.0000 | 1.0000 | 1.5667 | 10.0130 | 0.0161 | 0.6076 | 13.2012 | 0.0000 | 11.7434 |
| task_churn | PPO-MLP | 1.0000 | 1.0000 | 1.4000 | 1.6683 | 0.0638 | 1.2222 | 13.1568 | 0.0000 | 1.4041 |
| task_churn | GPPO-Adaptive | 1.0000 | 1.0000 | 1.3889 | 1.6131 | 0.0649 | 1.2253 | 12.8549 | 0.0000 | 10.4831 |
| event_storm_8 | PPO-MLP | 1.0000 | 1.0000 | 2.0783 | 16.6985 | 0.0411 | 0.4198 | 22.6294 | 0.0000 | 1.8822 |
| event_storm_8 | GPPO-Adaptive | 1.0000 | 1.0000 | 2.0922 | 17.2536 | 0.0778 | 0.5503 | 22.4432 | 0.0000 | 13.4772 |

## 配对差值（GPPO − PPO）

负数表示 GPPO 数值更低；是否更好取决于指标方向。回报和成功率越高越好，其余代价类指标越低越好。

| 场景 | 成功率差 | 恢复延迟差 | 累计空缺差 | 距离差 | 负载差 | 回报差 | 推理延迟差 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| atomic_triple_shock | 0.0000 | -0.0111 | 0.1670 | 0.0003 | 0.0301 | 0.0281 | 12.1885 |
| resource_collapse | 0.0000 | -0.0778 | 1.2816 | 0.0151 | 0.0000 | 1.1420 | 11.9828 |
| tracking_saturation_release | 0.0000 | -0.6778 | -12.8167 | 0.0240 | -0.0184 | 0.6096 | 12.3744 |
| out_of_order_reports | 0.0000 | 0.0111 | 0.0001 | 0.0033 | -0.0069 | -0.0362 | 11.7626 |
| long_blind_burst | 0.0000 | -0.1333 | -0.3340 | -0.0021 | 0.0139 | -0.0020 | 9.9401 |
| task_churn | 0.0000 | -0.0111 | -0.0552 | 0.0011 | 0.0031 | -0.3019 | 9.0790 |
| event_storm_8 | 0.0000 | 0.0139 | 0.5551 | 0.0367 | 0.1305 | -0.1862 | 11.5950 |

## 解释限制

- 这些场景由看到既有结果之后设计，是探索性压力测试，不是新的 held-out test。
- 当前环境在事件 observed_at 时才把事件送入 belief/runtime；长盲区指标主要衡量观测后的恢复，不能代表真实世界盲区损失。
- 当前策略没有循环记忆或 belief-state，因此乱序/缺失信息测试主要暴露系统合同和前馈策略的限制。
- 只有 3 个独立训练 seeds；差值仅描述本次样本，不作统计显著性或普遍优越性声明。
