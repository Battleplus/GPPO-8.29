# Round-3 P0 收口 — 最终状态

## 当前状态

- 分支：`plan/lightweight-sfc-refactor`
- 代码 Commit：`e351a35`（round-2 修复）
- Gate Commit：`d44ff59`（round-2 gate）
- 当前 HEAD 将在本轮提交后更新
- `training_allowed`：`true`（P0 Gate 全绿）
- 未启动 Preliminary 300k 训练

## Round-3 修复内容

### 1. Gate commit self-reference 修复
- 新增 `attested_source_commit_sha` 字段
- `_check_p0_gate()` 验证当前 HEAD 包含 attested commit（git merge-base --is-ancestor）
- 允许 evidence-only commits（gate JSON、smoke、handoff）不使 attestation 过期
- 修改任何 protected source/config/test 文件后立即拒绝

### 2. Gate 验证最终 Python 3.11 smoke
- `SMOKE_SUMMARY_PATH` 指向 `smoke_20260821_final`
- 验证 `environment_metadata.json`：Python 3.11.x、sb3_contrib、stable-baselines3 已安装
- 验证 per-mode counts（各 20）和总 replayed_tape_count=80

### 3. ConcurrencyManager exact graph_version equality
- `validate()`、`is_valid_at()`、`reject_stale_action()` 全部改为 `!=` 而非 `>`
- 未来版本命令也被拒绝（command v5 / current v4 → REJECT）

### 4. Stale rejection rate 修复
- 新增 `injected_stale_submissions` / `injected_stale_rejected` counters
- `submit_stale_action()` 是唯一注入 stale 的方法
- 正常合法 action 不计入 stale_attempted
- `snapshot_concurrency()` 报告注入 stale 的 rejection rate

### 5. Fencing monotonicity gate probe 修复
- 不再使用 `cmd.fencing_token < cmd.fencing_token + 1`（无效测试）
- 改为：holder A → revoke → holder B（更高 token）→ 旧 token 创建 lease 被拒绝

### 6. 新增 7 个测试（61 → 68）
- `test_future_version_rejected`
- `test_is_valid_at_rejects_future`
- `test_action_version_mismatch_rejected`
- `test_late_ack_after_expire_rejected`
- `test_real_revoke_and_new_holder`
- `test_matching_action_version_accepted`
- `test_mismatched_action_version_rejected_by_bridge`

## 测试结果（Python 3.11.5 锁定环境）

**68 tests, 0 failures, 0 errors — ALL PASS**

## Gate 关键字段

- `attested_source_commit_sha`: 记录保护的源代码 commit
- `git_commit_sha`: 当前 gate 生成时的 commit
- `training_allowed`: true
- `violations`: []
- smoke_20x4: PASS（80 tapes, Python 3.11 metadata verified）
- concurrency_fencing_monotonicity: PASS（real revoke + new holder test）

## 验证结果

1. ✅ `_check_p0_gate()` 从当前 HEAD 通过
2. ✅ 修改 protected source 1 byte 后 `_check_p0_gate()` 拒绝
3. ✅ 未来 graph_version 命令被拒绝
4. ✅ stale action_version 被拒绝
5. ✅ 旧 fencing token 创建 lease 被拒绝
6. ✅ 新 fencing token 严格大于旧 token
7. ✅ 最终 Python 3.11 smoke 被 Gate 实际引用并验证
