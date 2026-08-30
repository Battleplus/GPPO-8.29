# Execution-Preemption V1 可恢复进度

> 此文件是跨任务恢复入口。每个完整阶段通过测试并存档到 GitHub 后更新；恢复工作时必须先读取本文件、GitHub 远端分支和当前 worktree，三者以 Git/文件事实为准。

## 当前状态

```text
PROJECT: Execution-Preemption V1
SOURCE_BRANCH: research/execution-preemption-v1
ISOLATED_WORKTREE: E:\Z博士\26日\execution_preemption_v1
LATEST_ARCHIVED_REMOTE_STAGE: deterministic_baselines
LATEST_ARCHIVED_REMOTE_COMMIT: resolve refs/heads/research/execution-preemption-v1
CURRENT_STAGE: deterministic_baselines
CURRENT_STAGE_STATUS: COMPLETE_AND_ARCHIVED_BY_THIS_COMMIT
NEXT_STAGE: final_source_and_evidence_launch_gate
FORMAL_TRAINING_STARTED: false
VALIDATION_STARTED: false
FREEZE_STARTED: false
TEST_STARTED: false
HIDDEN_V1_GENERATED: false
```

## 已完成阶段

| 阶段 | 内容 | 远端存档 |
|---|---|---|
| 1 | 协议、连续运行时、事件仲裁与原子抢占 | `0ff863c…` |
| 2 | 五节点图、policy adapter、Gym/Torch framework rollout | `1185e1ed334e5a15735666d94582d01bcfe71f12` |
| 3 | 独立 source-bound Launch Gate 机制 | `e0aadb9bdde0b633325e78f5529aadbf4add8464` |
| 4 | 36-run 正式训练执行器、checkpoint/RNG/provenance 封装和 micro smoke | `35398d9bf430b8f866a21d21c712b82e9187250a` |
| 5 | 学姐旧方法适配、Greedy Priority、Beam-MPC 和 600 次开发带安全回放 | 当前 `research/execution-preemption-v1` HEAD |

## 当前阶段验证

```text
senior_legacy_method_v1: IMPLEMENTED
greedy_priority_v1: IMPLEMENTED
beam_mpc_v1: IMPLEMENTED
synthetic_method_differentiation: PASS (U0 / U2 / U1)
dev_tape_count: 200
allocator_count: 3
allocator_tape_runs: 600/600 PASS
execution_preemption_tests: 120/120 PASS
legacy_required_tests: 130/130 PASS
training_started: false
model_effectiveness_evaluated: false
```

## 下一步唯一动作

1. 从 GitHub 重新读取分支 HEAD、tree 和本文件，确认主分支仍为 `a420752…`；
2. 在新 clean source worktree 将合同状态冻结为 `FROZEN_FOR_SOURCE_ATTESTATION`；
3. 重新运行旧 130 项、新专项测试和全部 smoke；
4. 生成 source-bound Launch Gate，完成 source HEAD 与 clean evidence HEAD 的 formal check；
5. Gate 闭环前禁止正式训练；Gate 闭环后才能创建独立 training worktree。

## 禁止事项

- 不复用旧 minimum-validation Gate、smoke、checkpoint 或 campaign；
- 不修改旧 PPO/GPPO、旧 reward 或旧 S4 campaign；
- 不在 source/evidence worktree 启动训练；
- 不在训练证据封存前生成 Hidden-V1；
- 不提前运行 Validation、Freeze、Test 或 held-out evaluation；
- 开发带回放不能表述为算法效果证据。
