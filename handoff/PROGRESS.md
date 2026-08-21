# Round-2 P0 修复 — 最终状态

## 当前状态

- 分支：`plan/lightweight-sfc-refactor`
- 最终 Commit：`e351a35`
- 工作树：clean
- `training_allowed`：`true`（P0 Gate 全绿）
- 未启动 Preliminary 300k 训练（gate 转绿后需用户确认启动）

## 锁定验证环境

- Python 3.11.5 (`D:\anaconda`)
- torch 2.5.0+cpu
- numpy 2.0.2
- sb3-contrib 2.9.0
- stable-baselines3 2.9.0
- gymnasium 1.3.0

## 全部 61 测试 PASS（Python 3.11 实际运行）

### Test Suites

| Suite | Tests | Status |
|-------|-------|--------|
| core_contracts | 18 | ✅ PASS |
| training_contracts | 1 | ✅ PASS |
| event_runtime_integration | 8 | ✅ PASS |
| confirmation_timelines | 15 | ✅ PASS |
| concurrency_invariants | 13 | ✅ PASS |
| p0_gate_contract | 4 | ✅ PASS |
| legacy_compatibility | 2 | ✅ PASS |
| **Total** | **61** | **✅ ALL PASS** |

### 关键测试覆盖

#### Confirmation Timelines
- discovery 3-of-5: ✅ (0→not, 1→not, 2→not, 3→confirmed, 4→confirmed, 5→confirmed)
- duplicate does not increase independent count: ✅
- destruction authoritative single confirm: ✅
- destruction 2 same source → not confirmed: ✅
- destruction 2 independent → confirmed: ✅
- heartbeat 1 miss → SUSPECTED: ✅
- heartbeat 3 misses → PROBE_REQUIRED: ✅
- probe timeout → CONFIRMED: ✅
- healthy telemetry → FALSE_ALARM: ✅
- second independent source → CONFIRMED: ✅
- truth/belief isolation (2 of 5 → env unchanged): ✅

#### Concurrency Invariants
- graph_version exact match: ✅
- old version rejected: ✅
- ACK valid: ✅
- ACK wrong uav rejected: ✅
- ACK wrong fencing token rejected: ✅
- late ACK cannot resurrect: ✅
- exclusive holder ≤ 1: ✅
- two holders rejected: ✅
- revoke removes holder: ✅
- fencing token monotonicity: ✅
- action version stored: ✅
- adapter uses concurrency manager: ✅

#### Integration
- same seed → identical tape: ✅
- different seed → different tape: ✅
- same seed → same snapshot: ✅
- bridge observation count increments: ✅
- region vacancy confirms and modifies env: ✅
- UAV damage confirms and kills UAV: ✅
- concurrency counters initialized: ✅
- ACK resurrection count == 0: ✅

#### Event Mode Contracts
- burst 3-event → graph_version delta == 1: ✅
- single region vacancy → 1 increment: ✅
- overlap received_at ordering: ✅
- unseen isolation: ✅
- single snapshot identity: ✅
- four-mode reward invariant: ✅

#### Model Save/Load
- PPO-MLP: ✅
- GPPO-NoGate: ✅
- GPPO-Adaptive: ✅
- legacy MLP checkpoint → legal edge: ✅

## Smoke（20×4 = 80 tapes）

- Manifest: `results/random_event/smoke_20260821_final/tapes/smoke_20260821_final/manifest.json`
- Summary: `results/random_event/smoke_20260821_final/smoke_summary.json`
- Single 20, Sequential 20, Overlap 20, Burst 20 = 80 tapes replayed

## P0 Gate

- `training_allowed`: **true**
- `generated_by`: `scripts/build_p0_gate.py`
- `git_commit_sha`: `e351a358ce23a06223a7757529a662d47d31ad0a`
- `source_tree_hash`: `b64295d3330db410f7addc43426384d77c2a8a2055b700e45bacf7917b5780cf`
- `protocol_sha256`: `3d137a28d4a56737a2a1c29b59cd1000fd586fdea72682d1593f6c21fe289739`
- `seed_manifest_sha256`: `a47843efc244c0b62904b4ff21b19103cfb99911a943a49ce995c8f3102fbb43`
- 16 source file hashes all current
- violations: []

## Phase 状态

- Phase A: ✅ PASS — 真实状态记录
- Phase B: ✅ PASS — 冻结 seeds [1101,2202,3303]、protocol、validation/test bank
- Phase C: ✅ PASS — RuntimeBridge 主链路，TruthEvent→Detector→Observation→Confirmation→Belief→env
- Phase D: ✅ PASS — 3-of-5 discovery、dual-path destruction、heartbeat/probe/FALSE_ALARM
- Phase E: ✅ PASS — 执行级 command/ACK/lease/fencing 全部逻辑测试通过
- Phase F: ✅ PASS — Fair PPO-MLP input_dim 384、三 variant save/load 确定性
- Phase G: ✅ PASS — burst atomicity、四模式 reward invariant、80 tape smoke
- Phase H: ✅ PASS — 机器 gate 全绿、训练入口 HEAD/hash 验证

## 当前阻塞项

- L2 SFC/Isaac Sim 尚未开始（按规划 L0/L1 稳定后进行）
- Preliminary 300k 训练可启动（gate 全绿），等待用户确认

## 训练入口防护

experiment._check_p0_gate() 在每次 train 启动时验证：
- generated_by == scripts/build_p0_gate.py
- training_allowed == true
- current HEAD == gate git_commit_sha
- 所有 16 个 source file hash 完全一致
- protocol hash 一致
- seed_manifest hash 一致
- source_tree_hash 一致
