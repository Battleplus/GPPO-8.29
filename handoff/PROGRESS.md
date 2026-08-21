# 第二轮 P0 修复进度

## 当前状态

- 分支：`plan/lightweight-sfc-refactor`
- 基线：`ad488bc692dd2c093a56a43eb661f1bfa288315d`
- 工作树：dirty（第二轮修复尚未提交）
- `training_allowed`：`false`
- 未启动 Preliminary 300k 训练

## 本轮已完成的真实代码修复

### Truth State / Belief State

- `TruthStateTracker` 现在维护 `alive_uavs`、`damaged_uavs`、`discovered_targets`、`destroyed_targets`、`tracked_targets`、`vacant_regions`。
- 环境 reset 时只初始化 True State 的 UAV alive 集合；GraphObservation 不读取 True State。
- 未确认 Observation 不改变 belief、action mask、decision version 或 graph version。

### Confirmation

- `TARGET_DISCOVERED` 使用最多 5 个 distinct source/evidence opportunities，3 个独立 source 才确认。
- `TARGET_DESTROYED` 区分 authoritative 单条确认与 ordinary strong evidence 的两源确认。
- 同一 source、duplicate observation 不增加独立证据。
- heartbeat miss：`SUSPECTED` → 连续 3 次 `PROBE_REQUIRED` → probe timeout 确认；healthy telemetry 进入 `FALSE_ALARM`；第二独立 failure source 可确认。

### Graph Version / Runtime

- 只有 confirmed event 且确实改变 belief/decision-relevant state 时才增加 graph version。
- burst 三事件仍保持单次 graph version 增量。
- 未确认事件不会释放 lease、修改 pending regions 或触发 policy decision。

### Concurrency

- `ConcurrencyManager.receive_ack()` 现在校验 command_id、uav_id、fencing_token、command status。
- revoked/expired/rejected/completed command 的 late ACK 不得 resurrect。
- lease 执行层阻止同一区域多个有效 holder，并要求更高 fencing token。
- RuntimeBridge 实际执行 command → ACK → lease 生命周期，且环境 step 在 command 被拒绝时不再 mutation。

### P0 Gate / Hash

- `scripts/build_p0_gate.py` 新增真实执行项：reward invariant、confirmation timeline、unconfirmed no-decision、5 项 concurrency invariant、snapshot identity、overlap order、unseen isolation、model save/load、20×4 smoke。
- Gate 记录 `git_commit_sha`、`source_tree_hash`、source hashes、protocol SHA-256、seed manifest SHA-256。
- 训练入口现在验证 current HEAD、所有 source hashes、source tree hash、protocol hash、seed manifest hash。
- 不再采用“第一次 drift 自动 re-baseline”。

## 测试证据

命令：

```text
cd E:\Z博士\8.20\54_20-master\ppo_allocation
python -m unittest discover -s tests_random_event -v
```

结果：`61 tests`；新增逻辑测试全部 PASS；legacy 套件 2 errors：

1. `sb3_contrib` 未安装；
2. C++ bridge 子进程在当前环境返回非零。

命令：

```text
cd E:\Z博士\8.20\54_20-master
python scripts/build_p0_gate.py
```

结果：所有新增机器检查 PASS；`test_suites=FAIL` 仅因 locked legacy compatibility 未通过，因此 `training_allowed=false`。

Smoke 命令：

```text
cd E:\Z博士\8.20\54_20-master\ppo_allocation
python run_random_event_experiment.py smoke --output-dir results/random_event/round2_smoke_20260821_v2 --bank-name round2_smoke_20260821_v2 --tapes-per-mode 20 --events-per-tape 3 --master-seed 20260821 --max-decisions 60
```

实际结果：`80 tapes = 20×Single + 20×Sequential + 20×Overlap + 20×Burst`，manifest/raw tapes/summary/evidence 已保存。

## Phase 状态

- Phase A：PASS（真实状态记录）
- Phase B：PASS（冻结 seeds/protocol）
- Phase C：PASS（RuntimeBridge 主链路）
- Phase D：PASS（confirmation timelines）
- Phase E：PASS（执行级 command/ACK/lease/fencing 逻辑测试通过）
- Phase F：PASS（Fair PPO-MLP 与三 variant save/load）
- Phase G：PASS（四模式、burst atomicity、smoke）
- Phase H：PASS（机器 gate 与训练入口 hash 防护）
- Phase I：LOCKED-ENV PENDING（legacy compatibility 未在 Python 3.11 + sb3_contrib 环境完成）

## 当前仍然阻塞项

- 必须在锁定 Python 3.11 + 项目依赖 + `sb3_contrib` 环境重新运行完整 61-test suite 和 `scripts/build_p0_gate.py`。
- C++ bridge legacy test 也必须在锁定环境实际复验，不能预先归类为环境问题。
- 在上述 gate 全绿前，禁止 Preliminary 300k、Validation/Test 正式评估和任何长训练。
- L2 SFC/Isaac Sim 尚未开始。
