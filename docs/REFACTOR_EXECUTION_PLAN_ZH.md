# GPPO 项目重构执行规划（轻量验证优先 + SFC/Isaac Sim 后置集成）

> 本文档是交给后续 AI / 代码代理执行的主规划。
> 目标不是继续堆功能，而是把当前仓库重构成一条可验证、可复现、可逐层升级的实验链路。

## 0. 总原则

本项目不要求所有算法开发、训练和协议验证都运行在 SFC / Isaac Sim 高保真环境中。

正式采用三层验证体系：

1. **L0：逻辑与协议验证**
   - 普通 Python / CPU 即可。
   - 验证事件确认、并发一致性、reward、mask、seed、checkpoint、gate。

2. **L1：轻量算法仿真**
   - 使用 `RandomEventAllocationEnv` 及其后续重构版本。
   - 用于 PPO-MLP / GPPO-NoGate / GPPO-Adaptive 的训练、Validation、Test、统计分析。
   - 可在本地、服务器或 Google Colab GPU 运行。

3. **L2：SFC / Isaac Sim 高保真系统验证**
   - 仅在 L0/L1 稳定后进行。
   - 验证运动学、传感器、通信、地形、天气、执行时延和真实场景耦合。
   - 不重新定义算法协议，只替换环境后端适配器。

禁止把“L1 轻量环境结果”写成“已经验证 SFC / Isaac Sim 高保真环境性能”。

---

# 1. 目标架构

最终应形成以下结构：

```text
                    ┌──────────────────────────────┐
                    │ Frozen Experiment Protocol   │
                    │ seeds / modes / reward / PPO │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ Event Runtime                │
                    │ Truth → Observation          │
                    │ → Confirmation → Queue       │
                    │ → Belief / Versioning        │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ GraphObservationContract     │
                    │ ActionContract               │
                    └──────────────┬───────────────┘
                                   │
             ┌─────────────────────┴────────────────────┐
             │                                          │
   ┌─────────▼──────────┐                     ┌─────────▼──────────┐
   │ Lightweight Backend│                     │ SFC / Isaac Backend│
   │ RandomEvent Env     │                     │ high-fidelity sim  │
   └─────────┬──────────┘                     └─────────┬──────────┘
             │                                          │
             └─────────────────────┬────────────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │ PPO / GPPO Policies     │
                      │ MLP / NoGate / Adaptive │
                      └─────────────────────────┘
```

核心原则：**模型只依赖统一 observation/action contract，不直接依赖 Isaac Sim。**

---

# 2. 当前仓库的主要问题

## 2.1 P0 Gate 当前不可作为真实门禁

现有 `handoff/P0_GATE.json` 同时存在：

- `training_allowed: true`
- version validation = `PARTIAL`
- 18/21 tests passed
- source hashes = `pending`
- next_steps 仍包含 “Run full P0 gate tests”

这意味着 gate 只是状态记录，不是机器可执行的安全门禁。

### 重构要求

新增统一 gate 生成器，例如：

```text
scripts/build_p0_gate.py
```

Gate 必须由测试和 hash 自动生成，禁止手工把 `training_allowed` 改成 true。

任何以下情况都必须：

```text
training_allowed = false
```

- required test 未执行
- 任一 required test FAIL
- 任一 gate item 为 PARTIAL
- source hash 缺失
- protocol hash 缺失
- seed manifest hash 缺失
- 当前源码与 gate 中记录 hash 不一致
- Train / Validation / Test seed namespace 泄漏
- Test 可参与选模

---

# 3. Phase A：重新建立可信基线

## A1. 冻结当前状态

执行代理首先记录：

```bash
git status --short
git diff --stat
git log -1 --oneline
```

不得：

- `git reset --hard`
- `git checkout -- .`
- 删除历史模型
- 删除结果目录
- 覆盖原始 ZIP

更新：

```text
handoff/CURRENT_STATE.json
handoff/PROGRESS.md
handoff/DECISIONS.json
```

这些文件必须反映当前真实 HEAD，而不是旧审计状态。

## A2. 暂时关闭训练

在 Phase A–H 全部通过前：

```json
"training_allowed": false
```

所有长训练入口必须拒绝运行。

---

# 4. Phase B：统一 Frozen Protocol 与 Seed Manifest

当前配置存在协议漂移，必须先统一。

## B1. Preliminary 固定训练 seeds

统一为：

```text
1101
2202
3303
```

这三个 seed 同时用于：

- PPO-MLP
- GPPO-NoGate
- GPPO-Adaptive

不得某个模型使用 1/2/3、另一个使用 1101/2202/3303。

## B2. Validation

固定：

```text
100 tapes total
Single       25
Sequential   25
Overlap      25
Burst        25
Unseen        0
```

Validation 可以：

- checkpoint selection
- early stopping
- MLP width selection（如果协议允许且必须在 Test 前冻结）

Validation 不可以包含 unseen。

## B3. Test

固定：

```text
200 tapes total
Test-Single       40
Test-Sequential   40
Test-Overlap      40
Test-Burst        40
Test-Unseen       40
```

Test 禁止用于：

- checkpoint selection
- reward tuning
- architecture selection
- event probability tuning
- MLP width selection
- training step selection

## B4. 修复所有 CLI 默认值

需要同步检查：

```text
configs/seed_manifest.json
configs/random_event_protocol.json
configs/random_event_train.json
configs/random_event_validation.json
configs/random_event_test.json
ppo_allocation/random_event/experiment.py
colab_bundle/*
```

禁止文档、CLI、manifest 三套 seed 各不相同。

---

# 5. Phase C：把 Event Runtime 接进真正的环境主链路

这是当前重构最重要的一步。

当前环境不应再使用：

```text
observed_at → _apply_random_event() → state mutation
```

正式链路必须是：

```text
TruthEvent
  ↓
True State
  ↓
EventDetector
  ↓
ObservationTape
  ↓ received_at order
ConfirmationStateMachine
  ↓
ConfirmedEvent
  ↓
EventQueue
  ↓
Belief State
  ↓
GraphObservationContract
  ↓
Policy
```

## C1. True State 与 Belief State 必须分离

TruthEvent：

- 只能改变 true state
- 不能直接改变 policy 可见 graph

Observation：

- 只能进入 confirmation
- 不能直接释放 lease
- 不能直接改变 mask

ConfirmedEvent：

- 才可以改变 belief
- 才可以产生 pending region
- 才可以更新 graph version
- 才可以触发重新决策

## C2. 建议新增统一 runtime bridge

建议新增：

```text
ppo_allocation/random_event/runtime_bridge.py
```

职责：

```text
Truth tape
→ detector
→ observation tape
→ confirmation
→ event_runtime adapter
→ lightweight env state projection
```

不要在 `environment.py` 里复制一套新的 detector/state machine。

---

# 6. Phase D：完成真实的 Confirmation 语义

## D1. UAV Damage

必须实现：

```text
可信 ACTIVE_FAILURE_REPORT
→ direct confirm
```

普通 heartbeat：

```text
1 miss → SUSPECTED
2 miss → SUSPECTED
3 miss → launch probe
probe timeout OR independent second-source evidence
→ CONFIRMED
```

确认前收到 healthy telemetry：

```text
SUSPECTED → FALSE_ALARM
```

关键不变量：

```text
SUSPECTED 状态不得释放有效 assignment lease
```

## D2. Target Discovered

实现真实：

```text
3-of-5 confirmation
```

或者：

```text
authoritative multi-source confirmation
```

不能只是同一个 source 连续报三次就等价于三源确认。

## D3. Target Destroyed

允许：

```text
authoritative confirmation
```

或者：

```text
>=2 independent strong evidence sources
```

短时 loss-of-track 不得等价于 destroyed。

## D4. Tracking handoff

必须满足：

```text
TRACK command ACK
→ 才允许撤销发现者 SEARCH lease
```

---

# 7. Phase E：并发一致性必须从“数据结构”升级为“执行约束”

统一生命周期：

```text
graph_version
→ decision/action_version
→ latest mask validation
→ AssignmentCommand
→ ACK
→ AssignmentLease
→ FencingToken
→ renew / timeout / revoke
```

必须通过以下 invariants：

```text
stale_action_rejection_rate == 1.0
exclusive_task_valid_holder_count <= 1
duplicate_assignment_count == 0
unaffected_task_interruption_time == 0
```

## E1. Version validation

禁止只判断：

```python
current_graph_version > command.graph_version
```

应验证：

```text
command.graph_version == current graph version
command.action_version == active decision version
selected action remains legal under latest mask
```

## E2. ACK validation

ACK 必须核对：

```text
command_id
uav_id
fencing_token
command state
expiry
```

迟到 ACK 不得恢复已 revoked / superseded command。

## E3. Lease exclusivity

`create_lease()` 不允许无条件给同一个 region 创建第二个 ACTIVE holder。

新的 holder 只有在以下情况之一成立时才能获得 lease：

```text
old lease revoked
old lease expired
higher fencing token valid takeover
```

---

# 8. Phase F：重构三种学习模型到同一训练协议

三个正式 learned methods：

```text
PPO-MLP
GPPO-NoGate
GPPO-Adaptive
```

必须共享：

```text
GraphObservationContract
ActionContract
mask
reward
PPOConfig
training budget
checkpoint interval
seed
Validation bank
Test bank
```

唯一允许区别：

```text
PPO-MLP       = canonical flattened graph → MLP
GPPO-NoGate   = graph → AHGNN
GPPO-Adaptive = graph → AHGNN + adaptive gate
```

## F1. 引入统一 Policy Protocol

建议：

```python
class ActorCriticPolicy(Protocol):
    def act(...): ...
    def evaluate_action(...): ...
    def save(...): ...
```

`PPOTrainer` 不得再只接受两个 GPPO variant。

## F2. 修 PPO-MLP 输入维度

MLP input dimension 必须使用实际 flattened tensor 元素数量：

```text
sum(num_nodes[type] * feature_dim[type])
+
sum(num_edges[relation] * edge_feature_dim[relation])
+ optional global/runtime features
```

不能仅把 feature dimension 相加。

## F3. 修 PPO-MLP save/load

load 必须完整重建：

```text
encoder
edge_actor
noop_actor
critic
```

然后再 `load_state_dict()`。

三种模型均要有：

```text
train → save → load → deterministic inference equality
```

测试。

---

# 9. Phase G：重新定义四种 nominal mode + unseen

## G1. Single

真正语义：

```text
one case = one event
每个 case 从完全相同 initial snapshot reset
```

同一 bundle 中不同事件 branch：

```text
initial canonical snapshot SHA-256 必须相同
```

## G2. Sequential

```text
Event N fully recovered
→ Event N+1 may occur
```

不隐式 reset。

## G3. Overlap

```text
new event may arrive before previous recovery
process by received_at
```

早发生但晚收到的事件不得阻塞后发生但先收到的事件。

## G4. Burst

固定窗口：

```text
100 ms
```

同窗口 2–3 个 P1/P2 event：

```text
apply temporary state
merge affected + pending
NO intermediate policy call
build graph once
graph_version += 1 exactly once
policy inference once
atomic commit once
```

失败：

```text
rollback entire batch
```

## G5. Unseen

仅 Test 使用。

事件语义保持不变，只改变：

```text
event mixture
communication delay
loss
duplicate
out-of-order
partition severity
```

---

# 10. Phase H：Reward、Metrics 与实验可追溯性

## H1. Episode reward 唯一定义

```python
episode_return = sum(row["reward"] for row in decision_rows)
```

必须断言：

```python
assert episode_return == total_reward_check
```

一个 decision 服务多个 event 时：

```text
reward 只记录一次
affected_event_ids = [...]
```

禁止 event-level 再累计完整 decision reward。

## H2. Waiting cost

等待期间不创建假的 PPO transition。

空缺时间成本在下一个真实 decision step 结算，同时单独累计：

```text
cumulative_vacancy_time
```

## H3. 必须记录

至少：

```text
event recovery success
recovery delay
vacancy time
coverage AUC
distance
load gap
switch count
local_cost_regret
infeasible rate
communication bytes
stale rejection count
mask ratio
pre-mask invalid probability
inference latency
```

Adaptive 额外：

```text
gate mean/std/p10/p50/p90
gate gradient norm
```

---

# 11. Phase I：真正的 P0 Gate

## I1. 必须覆盖的测试

至少：

```text
14 design timelines
same seed deterministic
different seed divergence
duplicate idempotency
false alarm recovery
single snapshot equality
sequential recovery ordering
overlap received ordering
burst atomicity
burst rollback
unseen isolation
reward invariant x4 modes
Test forbidden for selection
CPP / legacy compatibility
PPO-MLP save/load
GPPO-NoGate save/load
GPPO-Adaptive save/load
```

## I2. Smoke

```text
Single       20
Sequential   20
Overlap      20
Burst        20
```

## I3. P0 hard requirements

```text
stale_action_rejection_rate == 1.0
exclusive_task_valid_holder_count <= 1
duplicate_assignment_count == 0
unaffected_task_interruption_time == 0
burst_three_event_graph_version_delta == 1
reward_invariant == true
```

全部 PASS 才允许：

```json
"training_allowed": true
```

---

# 12. Phase J：轻量 Preliminary 训练

只有 P0 真通过后才执行。

## J1. Learned models

```text
PPO-MLP
GPPO-NoGate
GPPO-Adaptive
```

## J2. Seeds

```text
1101
2202
3303
```

## J3. Budget

默认：

```text
300,000 decision steps / model / seed
checkpoint every 25,000 steps
```

如果算力不足，只允许：

```text
在所有模型训练开始前统一下调预算
```

不得训练完某模型以后再改变其他模型预算。

## J4. Validation checkpoint selection

字典序：

```text
1. lowest final infeasible rate
2. lowest cumulative weighted vacancy
3. lowest recovery delay
4. lowest fixed J
5. earliest checkpoint
```

选中后：

```text
freeze checkpoint SHA-256
```

## J5. Test

冻结模型后，每个最终 checkpoint 只在 Test bank 上运行一次。

---

# 13. Phase K：统计分析

训练 seed 是独立统计单位。

报告：

```text
raw seed values
mean
standard deviation
seed-level 95% CI
paired same-tape difference
effect size
Holm correction
```

主比较：

```text
Adaptive vs PPO-MLP
Adaptive vs NoGate
Adaptive vs Greedy
```

未恢复样本：

```text
conditional recovered-only distribution
+
right-censored analysis
```

禁止删除失败样本。

---

# 14. Phase L：SFC / Isaac Sim 高保真集成

这一阶段不是重新训练架构，而是环境适配。

建议新增：

```text
integration/
  contracts.py
  lightweight_adapter.py
  isaacsim_adapter.py
```

或者保持现有目录结构，但职责必须等价。

## L1. 共享 Contract

```text
EnvironmentBackend
  reset()
  advance_time()
  get_truth_state()
  get_observations()
  apply_assignment_command()
  receive_ack()
  build_graph_observation()
```

模型只能看：

```text
GraphObservationContract
```

模型输出只能是：

```text
ActionContract
```

## L2. Lightweight Adapter

负责连接：

```text
RandomEventAllocationEnv
EventRuntime
GraphObservation
```

## L3. Isaac Sim Adapter

负责把高保真状态转换为同一个 contract：

```text
UAV position / velocity
SEARCH / TRACK / IDLE mode
sensor status
region assignment
confirmed target state
communication state
lease state
graph version
```

policy 不允许直接 import Isaac Sim API。

## L4. L2 重点验证

只验证轻量环境抽象是否仍成立：

```text
motion feasibility
travel delay
sensor false positive / negative
terrain occlusion
weather degradation
communication latency / partition
command ACK delay
assignment execution latency
```

## L5. 高保真结论边界

只有 L2 实际跑完后才能使用：

```text
“在 SFC / Isaac Sim 高保真环境下...”
```

否则报告只能说：

```text
“在冻结的轻量随机事件仿真协议下...”
```

---

# 15. 文档与状态重构

当前部分文档存在状态过度声明，需要统一。

在完整 Preliminary 真正完成前：

```text
docs/FINAL_PRELIMINARY_REPORT_ZH.md
```

不得继续使用“所有 P0 已通过”“完整实验已完成”这类表述。

建议改名或重写为：

```text
docs/PRELIMINARY_READINESS_REPORT_ZH.md
```

状态建议统一：

```text
NOT_READY
P0_IN_PROGRESS
P0_READY
PRELIMINARY_TRAINING
PRELIMINARY_VALIDATION
PRELIMINARY_FROZEN
PRELIMINARY_TESTED
SFC_INTEGRATION
SFC_VALIDATED
```

---

# 16. 推荐执行顺序

后续 AI 必须按以下顺序执行，不得跳阶段：

```text
A. current-state re-audit
B. protocol + seed unification
C. event_runtime integration
D. confirmation semantics
E. concurrency enforcement
F. common PPO/GPPO trainer + PPO-MLP repair
G. mode semantics
H. reward / metrics / provenance
I. full tests + true P0 gate
J. lightweight preliminary training
K. statistics
L. SFC / Isaac Sim integration
```

任何阶段失败：

```text
STOP → fix → rerun validation
```

不得通过修改报告或 gate JSON 绕过失败。

---

# 17. 后续 AI 的执行提示

接手本规划的 AI 应遵循：

1. 先读历史入口 `docs/archive/MIMO_START_HERE_LEGACY.md`。
2. 再读本文件。
3. 再读 `handoff/MIMO_MASTER_TASK_ZH.md`。
4. 不推倒重写已有 `event_runtime/` 和 `random_event/`。
5. 每次修改优先增加测试，再修改实现。
6. 不运行长训练直到真实 P0 Gate PASS。
7. 不自动把 Test 用于调参。
8. 不伪造 Colab / SFC / Isaac Sim 运行结果。
9. 不把 L1 结果包装成 L2 结果。
10. 每阶段完成后更新 `handoff/PROGRESS.md` 和机器可读状态。

---

# 18. 最终验收标准

项目只有同时达到以下条件，才可以认为重构成功：

```text
[ ] Event Runtime 是实际主链路，不是旁路模块
[ ] True State / Belief State 明确分离
[ ] Confirmation 规则通过设计时间线
[ ] Command / ACK / Lease / Fencing 强制执行一致性
[ ] PPO-MLP / NoGate / Adaptive 共用同一 trainer contract
[ ] Single / Sequential / Overlap / Burst / Unseen 语义正确
[ ] Reward invariant 在所有模式通过
[ ] Frozen seed protocol 无泄漏
[ ] P0 Gate 由真实测试 + hash 自动生成
[ ] 训练入口强制检查 P0 Gate
[ ] Preliminary 使用 1101/2202/3303
[ ] Validation=100 nominal tapes
[ ] Test=200 tapes including unseen
[ ] Test 未参与模型选择
[ ] L1 轻量结果与 L2 高保真结果严格区分
[ ] Isaac Sim 通过 adapter 接入，不污染 policy/model 层
```

完成这些以后，再讨论正式 5-seed formal experiment。
