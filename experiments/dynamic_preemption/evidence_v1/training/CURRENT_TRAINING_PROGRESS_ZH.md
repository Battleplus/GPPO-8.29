# Execution-Preemption V1 当前训练进度

## 1. 状态摘要

本页记录截至 **2026-08-31** 的权威训练状态。它区分“已经完成的磁盘产物”和“已经通过只读复验并封存到 evidence commit 的阶段”，避免把 partial run 或未封存批次计入正式完成量。

```text
训练合同：Execution-Preemption V1
正式 learned runs：36
正式 checkpoints：72
每个 run：50,000 accepted decision steps
checkpoint grid：25,000 / 50,000

已封存完整阶段：UAV=4 / seed=1101
已封存 runs：4/36
已封存 checkpoints：8/72
已封存 accepted decision steps：200,000

当前活动 worker：0
Validation / Freeze / Test / held-out：全部未启动
```

## 2. 已封存阶段

### UAV=4 / seed=1101

| 方法 | Steps | Checkpoints | 只读复验 | Evidence 状态 |
|---|---:|---|---|---|
| `ppo_mlp_reactive_v1` | 50,000 | 25k、50k | PASS | SEALED |
| `gppo_adaptive_reactive_v1` | 50,000 | 25k、50k | PASS | SEALED |
| `ppo_mlp_rule_arbiter_v1` | 50,000 | 25k、50k | PASS | SEALED |
| `gppo_adaptive_rule_arbiter_v1` | 50,000 | 25k、50k | PASS | SEALED |

该阶段合计：

```text
4/4 runs
200,000 accepted decision steps
8/8 checkpoints
checkpoint SHA-256 / Gate / provenance / stderr：PASS
```

机器可读证据为 [`TRAINING_STAGE_UAV04_SEED1101.json`](TRAINING_STAGE_UAV04_SEED1101.json)，阶段 evidence commit 为 `0488166f40ccbeb83ea46a9d0c9551f975ddf4ee`。

## 3. UAV=4 / seed=2202 当前状态

| 方法 | 当前磁盘状态 | Steps | Checkpoints | 是否计入已封存进度 |
|---|---|---:|---|---|
| `ppo_mlp_reactive_v1` | COMPLETE | 50,000 | 25k、50k | 否；所在四方法批次未闭环 |
| `gppo_adaptive_reactive_v1` | 系统关机中断 | 39,656 | 25k | 否；partial run |
| `ppo_mlp_rule_arbiter_v1` | 未启动 | 0 | 无 | 否 |
| `gppo_adaptive_rule_arbiter_v1` | 未启动 | 0 | 无 | 否 |

GPPO-Reactive worker 的中断由 Windows 正常关机造成：系统事件确认关机，worker stderr 为 0，没有发现 Python 或算法异常。原始 partial progress、620 条单调更新记录和 25k checkpoint 均保留。

当前合同不支持 resume。按照冻结停止策略：

- 不允许在原目录续训；
- 不允许把 39,656 steps 当作完成；
- 不允许自动修补 checkpoint；
- 如需继续，必须获得明确授权，并在新的空 output namespace 中从零重跑该 run；
- 在本批次 4/4 runs、8/8 checkpoints 和只读复验全部通过前，不进入 seed=3303。

## 4. 操作系统与 launcher 状态

只读检查结果：

```text
ExecutionPreemptionV1-Retry1-6d2b343：Ready
ExecutionPreemptionV1-UAV04-Seed2202：Ready
当前训练 worker：不存在
```

`Ready` 表示计划任务存在但当前没有运行，不表示训练正在进行。seed2202 的 `launcher_state.json` 仍保留关机前写入的 `RUNNING` 文本，因此它是过时快照；当前 worker/PID、计划任务状态和关机 incident 共同表明训练已经停止。

没有重复启动任何 worker，也没有自动续训。

## 5. 尚未开始的正式范围

以下正式范围尚未开始或尚未形成完整阶段证据：

- UAV=4 / seed=2202 的剩余训练闭环；
- UAV=4 / seed=3303；
- UAV=8 / seeds 1101、2202、3303；
- UAV=16 / seeds 1101、2202、3303；
- 16-UAV/50k checkpoint 到 32 UAV 的 zero-shot 扩展性检查；
- Validation、Freeze、Test 和 held-out evaluation。

完整合同仍为：

```text
4 learned methods × 3 seeds × 3 scales = 36 runs
36 runs × 2 checkpoints = 72 checkpoints
```

## 6. 当前可以使用 GPPO 到什么程度

已经封存的 seed1101 GPPO 50k checkpoint 可以用于研究原型演示和系统联调：

- 读取 UAV—Task—Region—Target—Event 异构图；
- 在事件触发后生成合法的 UAV—Task 分配；
- 配合确定性控制器完成抢占后的重新分配；
- 通过动作掩码和运行时校验阻止非法提交。

当前已经证明的是“GPPO 能运行、能完成训练、能生成合法规划、checkpoint 可复验”。尚未证明的是“GPPO 的规划效果优于 PPO、Greedy 或 Beam-MPC”。

## 7. 下一步门禁

若用户明确授权继续训练，下一步不是评估，而是：

1. 保留现有 seed2202 原始目录和 incident；
2. 为中断的 GPPO-Reactive run 建立全新空 output namespace；
3. 从 0 重新训练到精确 50,000 steps，不 resume；
4. 完成 seed2202 的两个 Rule-Arbiter runs；
5. 核对 4/4 runs、8/8 checkpoints、SHA-256、Gate、provenance 和 stderr；
6. 生成机器可读阶段证据并先推送 GitHub；
7. 从远端重新读取确认后，才进入 seed3303。

在完整训练 campaign 封存前，不启动 Validation、Freeze、Test 或 held-out evaluation。

## 8. Provenance

```text
Attested source：62c001d83d9df4663aaf73af9401bc07ebb1e776
Runtime evidence HEAD：6d2b343489efc547b87ccfa9acce7963c31481ba
UAV4/seed1101 stage evidence commit：0488166f40ccbeb83ea46a9d0c9551f975ddf4ee
Training contract SHA-256：31dec25b47790cc68194a8e4fdd27f33a22a39c86869275e2e27ee4471b3e776
Training namespace：execution_preemption_v1/train
```

相关说明：

- [GPPO 动态任务规划文档](GPPO_PLANNING_PROCESS_ZH.md)
- [正式训练说明与作用分析](TRAINING_EXPLAINER_ZH.md)
- [UAV4/seed1101 机器可读阶段证据](TRAINING_STAGE_UAV04_SEED1101.json)
