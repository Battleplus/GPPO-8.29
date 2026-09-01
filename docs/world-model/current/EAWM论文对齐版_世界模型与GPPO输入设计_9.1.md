# EAWM 论文对齐版：世界模型与 GPPO 输入设计

- 日期：2026-09-01
- 指定论文：`C:\Users\ASUS\Desktop\2601.19336v1.pdf`
- 论文：Peng et al., *From Observations to Events: Event-Aware World Model for Reinforcement Learning*，ICLR 2026
- 当前代码：`E:\Z博士\9.1日\GPPO-8.29`
- 状态：方法和输入合同设计，不代表已经完成代码集成或正式实验

## 1. 论文方法核对后的核心修正

上一版在没有拿到指定论文时，按 PlaNet/RSSM 设计成“世界模型预测事件风险，再把风险直接输入 GPPO”。这不是指定 EAWM 论文最核心、最严格的方法。

EAWM 的论文主张是：

1. 从连续观测变化中自动生成稀疏事件，不依赖人工事件标签；
2. 增加 Event Predictor，迫使潜在表示关注有决策意义的时空变化；
3. 用 Generic Event Segmentor（GES）检测事件边界，在事件预测与原观测预测之间重新分配训练注意力；
4. 策略使用世界模型潜在状态学习行为；
5. Event Predictor 和 Observation Predictor 主要是表示学习的辅助模块，论文明确指出它们不必直接进入策略训练/推理。

因此，论文对齐版应当区分：

```text
EAWM 核心：自动事件目标 + GES + event-aware representation learning
可选扩展：把语义事件风险显式送给 GPPO
```

如果直接输入未来 `UAV_DAMAGE/TARGET_DISCOVERED/...` 风险而不实现自动事件生成、GES 和论文式消融，只能称为“事件风险增强 GPPO”，不能称为论文对齐 EAWM-GPPO。

## 2. 当前 GPPO 是否能够正常使用

能，但其“规划”边界必须说准确。

当前 GPPO 可以：

- 接收 UAV、Region、Target 三类节点的当前 belief 图；
- 在 16 条 UAV–Region 边和一个 NOOP 中选择合法动作；
- 通过 action mask、graph/action version、ACK、lease、fencing 保证执行安全；
- 对确认后的 UAV 损毁、目标变化和区域空缺进行局部重新分配；
- 完成前向推理、最小 PPO 训练、checkpoint 保存和加载。

本地针对性测试结果：

```text
GPPO-Adaptive forward + save/load：PASS
minimal PPO training + save/load：PASS
```

当前 GPPO 不能：

- 通过学习到的世界模型预测未来图；
- 在潜空间生成 imagined trajectories；
- 直接输出航迹或替代 MPPI/search planner；
- 完成执行中任务的完整暂停、抢占、迁移和进度恢复；
- 使用 GitHub 中未提供的正式 50k checkpoint 开箱部署。

准确结论：

> 当前 GPPO 是可以正常运行的反应式动态图任务分配器；完成 EAWM 接入和潜空间想象训练后，才能称为事件感知世界模型辅助的预测式任务分配。

## 3. 论文规定的世界模型数据单元

EAWM 使用轨迹片段：

```text
O_t = [o_(t-k+1), ..., o_t]   观测片段
A_t = [a_(t-k+1), ..., a_t]   动作片段
r_t                            奖励
c_t                            episode continuation
e_t                            从观测自动生成的事件
```

映射到本项目：

```python
EAWMStep(
    observation_graph,       # 策略当时可见的 belief 图
    received_messages,       # decision_time 前已收到的信息
    action,                  # UAV–Region edge 或 NOOP
    reward,
    cost_components,
    continuation,
    auto_events,             # 由相邻可见观测自动计算
    feature_valid_mask,
    padding_mask,
    audit_metadata           # 仅审计，不喂模型
)
```

其中 `auto_events` 必须由可见观测变化产生。仿真器 truth event 和未来事件带只可用于离线解释/审计，不能作为 EAWM 在线输入。

## 4. 当前观测 `o_t` 的设计

### 4.1 当前图张量

```text
UAV nodes:       [4, 12]
Region nodes:    [4, 12]
Target nodes:    [3, 16]
candidate edges: [16, 2]
action mask:     [17]
```

节点和边字段继续以当前 `HeteroGraphState` 为唯一基础，不另建一套与策略不一致的状态。

### 4.2 多模态拆分

为了对应论文中的 ordinal/nominal modalities，必须把字段按语义分组，不能把整个图简单视为一个连续向量。

#### A. Ordinal：有方向和大小的连续量

- UAV 归一化位置 x/y；
- UAV 当前负载；
- Region priority、workload、vacancy duration；
- Target 可见位置 x/y；
- UAV–Region distance、load、communication quality；
- Observation confidence、severity、age、delay；
- reward 及 uncovered/distance/load/switch/recovery cost 分量；
- delta_time、pending count、legal action ratio。

#### B. Nominal：类别发生变化才构成事件

- UAV alive、sensor available、task type、target ID；
- Region assigned UAV、pending、assignment legal；
- Target type、discovered、tracked、destroyed、region ID、tracker ID；
- action edge/NOOP、selected UAV/Region；
- ACK/lease/fencing/stale 状态；
- action mask 每个位置是否合法。

#### C. Structural：图结构变化

- 节点新增/失效；
- assigned/contains/tracks/communicates 关系变化；
- 候选边新增/删除；
- action-mask support 变化。

当前固定 4/4/3 场景中，很多结构变化表现为字段切换；未来动态规模时必须显式建模 add/remove，并提供 entity validity mask。

#### D. Message/Evidence：弱通信观测流

- 已收到的 source/signal 类型；
- positive/negative；
- confidence、severity；
- message age、delay、sequence；
- duplicate、conflict；
- confirmation status 和 distinct-source progress。

这部分可看成独立模态。它允许世界模型在事件确认前学习运动/变化趋势，但没有权限直接修改 belief 和 action mask。

### 4.3 图编码

不能把 EAWM 限死在当前 4/4/3 规模。建议：

```text
typed node encoders
+ relation-aware edge encoders
+ entity validity masks
→ graph/set encoder
→ observation embedding z_t
```

第一版可复用 AHGNN 思路，但世界模型 encoder 与策略 encoder 参数应先分开，避免世界模型损失直接破坏已经能运行的 GPPO 表示。

## 5. 论文式自动事件生成器

### 5.1 Ordinal 事件

论文对连续变量使用归一化变化：

```text
delta_i = (o_t[i] - o_(t-1)[i]) / Range_i

event_i = +1, delta_i > C_o
          -1, delta_i < -C_o
           0, otherwise
```

本项目中 `Range_i` 只能由 train split 的物理范围或冻结统计确定，不能读取 test bank 后调整。

静态字段（例如固定 Region 中心）必须用 `event_eligible_mask=0` 排除，否则大量永不变化维度会稀释事件密度。

### 5.2 Nominal 事件

类别变化即事件：

```text
event_i = 1 if category_t != category_(t-1) else 0
```

one-hot 字段应先还原为一个类别组，再判断类别是否变化，不能把一次任务类型切换错误计算成两个独立事件。

### 5.3 Structural 事件

对节点、关系和合法动作 support 建立：

```text
ADD / REMOVE / ENABLE / DISABLE / REASSIGN
```

首版可编码为 nominal 多标签；后续动态规模版本再加入 entity-aligned event tokens。

### 5.4 Message 事件

消息流的连续字段按 ordinal 规则，source/signal/status 按 nominal 规则。重复消息本身可记录成 nominal event，但不能增加确认状态机的独立证据数。

### 5.5 语义事件标签不是 EAWM 核心标签

`UAV_DAMAGE`、`TARGET_DISCOVERED`、`TARGET_DESTROYED`、`REGION_VACANCY` 可以作为额外 domain head，但必须与自动事件目标分离：

```text
auto-event head：论文核心，自监督
semantic-event head：领域扩展，使用已确认事件或离线标签
```

正式消融必须能分别关闭二者。

## 6. Generic Event Segmentor（GES）

每种模态计算事件密度：

```text
alpha_t^(m) = 有事件的有效维度数 / 该模态有效维度数
```

论文 EADream 的简单形式：

```text
g_t^(m) = I(alpha_t^(m) < alpha_thr^(m))
```

解释：

- 事件稀疏、仍位于一个事件片段内部时，继续使用 event prediction 约束表示；
- 大量维度同时变化、检测到事件边界时，抑制 event-prediction loss，重新重视原观测预测；
- GES 是确定性函数，首版不增加可训练参数。

论文给出的 modality-level `alpha_thr` 起点是：ordinal `1.0`、nominal `0.5`。这些值可以作为 smoke 起点，但在本项目必须用 train split 预注册和冻结，不能根据正式测试结果调整。

事件损失：

```text
L_event = Σ_m beta_e^(m) * g_t^(m) * L_event^(m)
```

- ordinal polarity `{-1,0,+1}`：cross entropy；
- nominal/稀疏多标签：focal loss；
- event target 使用 stop-gradient；
- class imbalance 权重只能从 train split 计算。

## 7. EAWM 图世界模型

论文通用结构映射为：

```text
Sequence model:
    h_t, y_t = F(h_(t-1), Z_(t-1), A_(t-1))

Representation model:
    z_t ~ q(z_t | o_t, h_t)

Dynamics predictor:
    z_hat_t ~ p(z_hat_t | y_t)

Reward predictor:
    r_hat_t ~ p(r_hat_t | y_t, z_t)

Continuation predictor:
    c_hat_t ~ p(c_hat_t | y_t, z_t)

Observation predictor:
    o_hat_t ~ p(o_hat_t | y_t, z_hat_t)

Event predictor:
    e_hat_t ~ p(e_hat_t | stopgrad(y_t), y_(t+1), z_hat_t, z_t)
```

适合当前项目的 v0 规格：

| 组件 | 建议起点 |
|---|---:|
| graph observation embedding | 96 |
| action embedding | 32 |
| recurrent hidden `h_t/y_t` | 64 |
| stochastic/observation embedding `z_t` | 32 |
| training chunk length | 32 decisions |
| imagined horizon | 初期关闭，后续 1–3 |

世界模型必须预测：

- 下一图的 ordinal 字段或其增量；
- 下一图的 nominal/structural 字段；
- reward 和 cost components；
- continuation；
- 自动生成事件；
- 可选的领域语义事件。

不要求重建 identity、固定 Region 中心等与动态无关的静态量。

总损失：

```text
L = L_base_world_model
  + beta_o * L_event_aware_observation
  + beta_e * L_GES_event
```

其中 `L_base_world_model` 包含 dynamics/representation、reward、continuation 等基础损失。

## 8. 怎样输入 GPPO 才算论文对齐

### 8.1 策略输入

论文将 agent state 定义为 sequence output 和 observation embedding 的组合：

```text
s_t = [y_t, z_t]
```

对 GPPO，推荐保留当前图策略，同时把 `s_t` 作为上下文：

```text
当前 belief 图 → GPPO AHGNN
EAWM latent s_t → zero-init context projector
两者融合 → masked actor + critic
```

首版使用 global residual：

```python
pooled += global_context_projector(s_t)
edge_actor_input += edge_context_projector(s_t)
```

projector 最后一层零初始化，确保 EAWM 关闭/零 context 时旧 GPPO 输出不变。

### 8.2 Event Predictor 不直接喂策略

论文对齐主实验中：

- GPPO 输入 `[current graph, s_t]`；
- 不直接输入 `e_hat_t`；
- Event Predictor 只通过辅助损失塑造 `s_t`；
- behavior learning 时可关闭 observation/event decoder，减少推理开销。

“显式输入 predicted event risk”应当作为单独扩展组，不能与 EAWM 核心方法混在一起。

### 8.3 公平对照

至少需要：

| 方法 | 基础世界模型 | 自动事件预测 | GES | 显式事件风险输入 |
|---|---|---|---|---|
| GPPO | 否 | 否 | 否 | 否 |
| WM-GPPO | 是 | 否 | 否 | 否 |
| EA-noGES-GPPO | 是 | 是 | 否 | 否 |
| EAWM-GPPO | 是 | 是 | 是 | 否 |
| EAWM-Risk-GPPO | 是 | 是 | 是 | 是，领域扩展 |

只有 `EAWM-GPPO - WM-GPPO` 才能说明论文事件感知机制的增量收益。

## 9. 行为学习和“规划”的两级实现

### 9.1 第一级：真实轨迹上的 latent-context GPPO

先冻结或低学习率更新世界模型，在真实环境 transition 上训练 GPPO。该阶段主要验证事件感知表示是否改善反应式任务分配，不声称已经完成 latent imagination planning。

### 9.2 第二级：论文式 imagined behavior learning

世界模型通过独立 held-out 校准后，才允许：

1. 从真实 posterior state `[y_t,z_t]` 分支；
2. 用 dynamics prior 生成 1–3 步 imagined trajectory；
3. 预测 reward、continuation 和下一图/动作可行性；
4. 在 imagined latent states 上训练 actor-critic；
5. 不确定度过高的 imagined transition 丢弃或降权；
6. 真实和 imagined 数据分池，固定混合比例；
7. 最终线上动作仍由真实当前 mask/version/lease/fencing 校验。

由于本项目动作集合随图和 mask 改变，imagined mask 只能用于训练候选估计，绝不能成为执行安全依据。

可选的 rolling-horizon beam search 是工程扩展，不是该论文 EAWM 的必要组成部分，应在 EAWM 表示收益被证明后单独实现和命名。

## 10. 输入白名单与禁止字段

### 10.1 线上允许

- 当前 belief graph；
- decision_time 前已到达的 Observation；
- 当前可见 message age/delay/confidence/source/signal；
- 当前状态机只读 confirmation progress；
- 历史已执行动作及执行反馈；
- 历史 reward/cost；
- 当前 action mask；
- delta time 和 continuation。

### 10.2 只用于离线标签/审计

- truth event；
- truth-only `occurred_at`；
- 尚未到达的 Observation；
- future tape；
- 未来 graph/mask/reward；
- 未来 exact-planner action；
- test bank 统计与阈值调参信息。

### 10.3 永远不作为业务特征

- graph/action version 的绝对数值；
- observation cursor 绝对数值；
- event seed、tape hash、文件路径和 checkpoint 名称。

版本、cursor 和 hash 只用于对齐、stale 幂等和证据审计。

## 11. stale-safe 在线状态

正确决策顺序：

```text
begin_decision
→ 冻结 graph/action version、decision time、observation cursor
→ 构造 o_t 和 auto-events
→ 临时计算 EAWM posterior state
→ GPPO 推理
→ versioned ActionSubmission
→ submit_action
→ 只有动作/transition 被接受后才提交世界模型历史状态
```

若新事件在推理期间到达：

- 拒绝旧动作；
- 丢弃临时 posterior 更新；
- 不写 PPO rollout；
- 不写世界模型 executed transition；
- 不推进 accepted decision count；
- 使用新版本和新 cursor 重新计算。

实现上可采用 transactional hidden state：`propose_state()` 返回临时状态，`commit_state()` 只在 accepted transition 后调用。必须另有从最近轨迹 chunk 重算的审计路径。

## 12. 实施路线

### EA0：冻结 schema 和自动事件定义

交付：ordinal/nominal/structural 分组、Range、`C_o`、eligible mask、GES `alpha_thr`、配置与版本号。

通过条件：相同轨迹生成 byte-identical auto-events；所有阈值只来自 train split/物理合同。

### EA1：数据记录和防泄漏

交付：decision-level dataset、episode/tape/instance 分组 split、manifest、SHA-256、字段审计。

通过条件：不存在 `received_at > decision_time`；truth-only 在线字段为零；同一 tape/instance 不跨 split。

### EA2：基础 Graph World Model

先训练不含 Event Predictor/GES 的 `WM-GPPO` 基础世界模型。

通过条件：held-out 下一图、reward、continuation 优于 last-value/frequency/普通 GRU 基线；多步误差可解释。

### EA3：Event Predictor + GES

加入自动事件目标、stop-gradient、focal/CE 和 event-aware observation loss。

通过条件：event AUPRC/F1 优于频率基线；训练稳定；`EAWM-GPPO` 的潜表示在事件相关 probe 上优于 `WM-GPPO`。

### EA4：Shadow mode

实时更新 EAWM latent 和诊断，但不输入 GPPO。

通过条件：belief/mask/version/action 完全不变；stale 不双消费；P95 时延满足冻结预算。

### EA5：EAWM latent 接入 GPPO

世界模型先冻结，只训练 zero-init adapter 和策略。运行论文式对照矩阵。

通过条件：非法动作 0、stale 漏拦截 0、最终不可行不恶化；相对 WM-GPPO 在独立 bank 上出现稳定业务收益。

### EA6：Imagined actor-critic

先 1 步，再 3 步；固定真实/想象比例和不确定度门。

通过条件：收益必须出现在真实 held-out 环境，不能只提高 predicted return。

### EA7：Extreme-V2 一次性验收

旧 extreme bank 只能开发。模型、阈值、损失权重和策略冻结后，在全新 hidden bank 一次性运行。

## 13. 必须新增的代码

```text
ppo_allocation/eawm/
├── schema.py
├── modality_registry.py
├── event_generator.py
├── ges.py
├── recorder.py
├── graph_encoder.py
├── sequence_model.py
├── dynamics.py
├── observation_predictor.py
├── event_predictor.py
├── reward_continuation.py
├── losses.py
├── context_adapter.py
├── transactional_state.py
└── diagnostics.py
```

修改边界：

- `random_event/graph.py`：字段分组/valid mask/可选 latent context；
- `random_event/models.py`：zero-init latent context；
- `random_event/trainer.py`：real/imagined buffer 和 stale-safe commit；
- `random_event/environment.py`：decision snapshot 的 time/cursor；
- `random_event/runtime_bridge.py`：只读 received-message snapshot；
- `event_runtime/state_machine.py`：只读 evidence progress。

## 14. 必须新增的测试

```text
auto-event determinism
ordinal polarity and frozen range
nominal category grouping
structural add/remove events
static-feature eligible mask
GES boundary behavior
event target stop-gradient
no future observation
truth-field denylist
tape/instance split isolation
zero-context GPPO parity
event-predictor-off ablation
GES-off ablation
shadow no-state-mutation
stale transactional rollback
old checkpoint compatibility
real-mask execution validation
world-model and end-to-end latency
```

四个最高优先级断言：

```text
auto-events 只能来自当时可见观测
EAWM 预测不能修改 belief 或 action mask
stale 推理不能提交 hidden state 或 transition
论文对齐 EAWM 组不能直接把 event prediction 喂给策略
```

## 15. 推荐立即执行的最小闭环

```text
1. 冻结 observation modality registry
2. 实现 ordinal/nominal auto-event generator
3. 实现 GES 和 deterministic unit tests
4. 实现 decision recorder 与无泄漏审计
5. 先跑 last-value / GRU / base graph-WM
6. 再加入 Event Predictor，比较 WM vs EA-noGES vs EAWM
7. Shadow mode
8. 冻结 EAWM latent 接入 GPPO
9. 新 hidden bank 验证
10. 通过后再做 imagined actor-critic
```

第一阶段不要同时修改奖励、动作空间、GPPO 主干和世界模型。最小科学问题应该是：

> 在完全相同的图、动作、奖励、数据和预算下，自动事件预测与 GES 是否能使世界模型潜在表示更关注关键动态图变化，并最终改善 GPPO 的真实任务分配结果？

只有这个问题得到正面、可复现的答案，才继续增加显式语义风险、预测触发、beam planner 或偏好奖励。

## 16. 最终推荐输入

论文对齐的第一版输入冻结为：

```text
当前 belief 异构图
+ decision_time 前已到达的 message/evidence
+ 上一步 UAV–Region/NOOP 动作
+ reward 与 cost components
+ continuation
+ field-valid / entity-valid / padding masks
+ 从相邻可见观测自动生成的 ordinal/nominal/structural events
```

策略侧输入冻结为：

```text
当前 belief 图 + EAWM latent state [y_t, z_t]
```

自动事件和 GES 首先服务于世界模型表示学习，而不是直接替代安全确认状态机。最终执行安全仍由当前 action mask、graph/action version、ACK、lease 和 fencing 提供。
