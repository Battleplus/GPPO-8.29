# 执行进度跟踪（Phase A-H 完成版）

## 当前 HEAD

- 分支: `plan/lightweight-sfc-refactor`
- HEAD: `684a113` + 未提交重构改动

---

## Phase A: 重新建立可信状态

### 状态: 已完成

- [x] 检查 git status / diff / log
- [x] 诊断 P0 Gate 虚假状态并重写（training_allowed=false）
- [x] 重写 CURRENT_STATE.json / PROGRESS.md / DECISIONS.json

---

## Phase B: 统一 Frozen Protocol

### 状态: 已完成（配置 + CLI 对齐）

- [x] seed_manifest.json: preliminary seeds -> [1101,2202,3303]
- [x] seed_manifest.json: Validation -> 100 tapes (25x4, 无 Unseen)
- [x] seed_manifest.json: Test -> 200 tapes (5 类各 40, 含 Unseen)
- [x] random_event_protocol.json: burst window -> 0.1s
- [x] experiment.py CLI: seeds default -> 1101,2202,3303; variants 含 PPO-MLP
- [x] run_random_event_experiment.py docstring 更新
- [x] 验证 seed namespace 无泄漏（gate 机器验证）

---

## Phase C: Event Runtime 接入环境主链路

### 状态: 已完成

- [x] 创建 ppo_allocation/random_event/runtime_bridge.py
  - TruthStateTracker（真状态与信念分离）
  - DeterministicDetector（loss/duplicate/out-of-order/partition，seed 确定）
  - RuntimeBridge（ingest_truth_event -> confirmation -> apply_confirmed_to_env）
  - 多证据事件（TARGET_DISCOVERED 3-of-5, TARGET_DESTROYED >=2）从不同 source 生成观测
- [x] environment.py 接入 bridge（_ingest_observed_events 通过 bridge）
- [x] 未经确认事件不改 belief（bridge 路径只对 confirmed 事件改 env）

---

## Phase D: Confirmation 语义

### 状态: 已完成（state_machine.py + bridge 多源证据）

- [x] UAV_DAMAGE trusted failure -> 直接确认
- [x] TARGET_DISCOVERED -> 3-of-5 多源确认
- [x] TARGET_DESTROYED -> >=2 独立强证据
- [x] SUSPECTED / FALSE_ALARM / duplicate / late 计数

---

## Phase E: Concurrency 执行约束

### 状态: PARTIAL（concurrency.py 有数据结构；执行级不变量测试待 sb3 环境）

- [x] AssignmentCommand / ACK / AssignmentLease / FencingToken 已定义
- [ ] stale_action_rejection_rate == 1.0 执行级测试（gate 待环境）
- [ ] exclusive holder <= 1 执行级测试（gate 待环境）

---

## Phase F: Fair PPO-MLP

### 状态: 已完成

- [x] FairPPOMLP input_dim 修复（numel 扁平计数 384，而非 sum(shape[-1])=60）
- [x] FairPPOMLP.load() 重建 encoder/actor/critic 后 load_state_dict
- [x] PPOTrainer 支持 PPO-MLP variant
- [x] 三模型 save->load->deterministic inference equality 验证通过（gate 机器验证）

---

## Phase G: 四种事件模式

### 状态: 已完成

- [x] Burst: 调度器簇共享 observed_at + 环境原子批提交 -> 3-event graph_version delta == 1
- [x] Single: canonical snapshot SHA 一致
- [x] Sequential / Overlap / Unseen 配置冻结

---

## Phase H: 自动 P0 Gate

### 状态: 已完成

- [x] scripts/build_p0_gate.py（机器生成，禁止手工改 training_allowed）
- [x] 检查项: test_suites / seed_namespace_isolation / frozen_protocol_contract / burst_atomicity / model_save_load_determinism / source_hash_integrity
- [x] 训练入口 run_train() -> _check_p0_gate() 拒绝 RED gate

### 当前 Gate: RED（诚实状态）
- training_allowed: false
- 原因: legacy_compatibility 套件需要 sb3_contrib（锁定环境 Python 3.11）
- 通过项（机器验证）: core 19/19, seed isolation, protocol contract, burst atomicity, model determinism, hash integrity

---

## Phase I: 运行全部 P0 测试

### 状态: 已完成（gate 驱动）

- [x] python scripts/build_p0_gate.py 运行
- [x] 19/21 通过（2 个 legacy 错误 = 环境依赖缺失，非代码问题）
- [x] RED gate 正确拒绝训练

---

## 决策记录

### D-001: 重置所有状态声称（EXECUTED）
### D-002: 不推倒重写 event_runtime，用 runtime_bridge 接入（EXECUTED）
### D-003: 统一 seeds [1101,2202,3303]、Validation=100、burst=0.1s（EXECUTED）
### D-004: PPOTrainer 支持三 variant（EXECUTED）
### D-005: 机器生成 P0 Gate + 训练入口检查（EXECUTED）
### D-006: Gate 保持 RED 直到 legacy 套件在锁定环境通过（HONEST-CURRENT-STATE）

---

## 遗留风险 / TODO

- [ ] sb3_contrib 缺失 -> legacy_compatibility 套件无法运行，gate 保持 RED
- [ ] Python 3.14 被协议禁止训练，需锁定 Python 3.11 环境
- [ ] CPP bridge 测试需特定环境设置
- [ ] L2 SFC/Isaac Sim 未开始（按规划 L0/L1 稳定后进行）
- [ ] Preliminary 300k 训练未启动（gate 全绿后才允许）
