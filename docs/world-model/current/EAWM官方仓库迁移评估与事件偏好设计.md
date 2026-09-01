# EAWM 官方仓库迁移评估与“事件偏好”设计

- 日期：2026-09-01
- 当前 GPPO：`E:\Z博士\9.1日\GPPO-8.29`
- EAWM 官方仓库：`https://github.com/MarquisDarwin/EAWM`
- 本地源码快照：`E:\Z博士\9.1日\EAWM-official`
- 核对提交：`269f71af5b3510fbcdb2c7d3ebeb22b1a15f5241`
- 本文定位：在已有论文对齐设计上，补充“官方代码能复用什么、不能复用什么”，并把事件注意与偏好学习拆开。

## 1. 结论

这个官方仓库对本项目有明显帮助，但应当把它作为**实现参照和消融模板**，不应把 EADream/EASimulus 整套代码直接塞进 GPPO。

最值得借鉴的是：

1. 自动事件生成的数据流；
2. Event Predictor 作为世界模型辅助头的接法；
3. 按事件密度调节损失的 GES；
4. image/vector/token 等模态分别计算事件和损失；
5. `event_pred`、`ges` 等开关式消融组织方式。

不适合直接搬运的是：

1. Atari 图像卷积解码器和 ROM 数据；
2. DreamerV3、Simulus/RetNet 的完整训练框架；
3. 面向像素运动的 MOG2 事件生成器；
4. 直接依赖其环境、tokenizer、checkpoint 的训练脚本。

本项目的主体是异构图上的 UAV–Region 分配，应该保留当前 GPPO 的图策略、17 维动作语义和安全 action mask，只把 EAWM 的事件学习思想重新实现成 **Graph-EAWM**。

## 2. 官方源码实际证明了什么

### 2.1 自动事件来自相邻观测，不依赖人工事件名称

`EASimulus/src/collector.py` 对不同模态分别生成事件：

- image：背景变化/运动掩码；
- vector：差分超过阈值，支持下降/不变/上升三类；
- token、token_2d：类别或格点值发生变化。

因此，本项目不能只拿 `UAV_DAMAGE`、`TARGET_DISCOVERED` 等仿真真值当作 EAWM 事件。论文对齐的事件应首先由策略当时可见的连续 belief 图变化自动生成；语义事件只能作为额外领域头或审计标签。

### 2.2 Event Predictor 是世界模型辅助头

`EADream/models.py` 在潜状态上增加 event decoder，并把 event loss 与 observation/reward/continuation/KL loss 一起训练。`EASimulus/src/models/world_model.py` 也为每种观测模态建立 event head 和不同事件损失。

这说明事件预测器的核心价值是迫使潜状态保留“发生了什么变化”，而不是把一个人工风险分直接追加到策略输入。

### 2.3 GES 的作用是处理事件密度，而不是给动作排喜好

官方仓库包含两种工程形式：

- EADream：事件密度超过 `event_pred_ratio` 时，抑制该时刻 event loss；
- EASimulus：按模态计算事件占比，再用 `event_weight_function` 连续调节事件损失。

它们都在解决同一个问题：边界时刻很多维度同时变化时，不让 event prediction 压过原始观测预测。GES 不是“这个事件比那个事件更重要”的人工奖励表。

### 2.4 策略读取 latent，不直接读取 event logits

`EADream/dreamer.py` 的 actor 接收 `dynamics.get_feat(latent)`；Event Predictor 留在世界模型训练损失中。对本项目，论文对齐主实验也应保持：

```text
自动事件 → 辅助训练 Graph-EAWM → 更好的 latent → GPPO actor/critic
```

显式把未来事件概率送进策略可以做，但必须标成扩展组，不能与论文核心组混为一谈。

## 3. 对 GPPO 的推荐总体结构

```text
策略当时可见的 belief 图 G_t
       + 已执行动作 a_(t-1)
       + 时间/有效性 mask
                 │
                 ▼
     typed graph/set encoder
                 │
                 ▼
   Graph-RSSM: deterministic h_t + stochastic z_t
        ┌────────┼─────────────┬──────────────┐
        ▼        ▼             ▼              ▼
 next graph   reward/cost   continuation   auto-event heads
 prediction   prediction     prediction      + GES
        └────────┴───────┬─────┴──────────────┘
                         ▼
                 latent context [h_t,z_t]
                         │
                         ▼
         当前 GraphActorCritic / GPPO
       16 条 UAV–Region 边 + 1 个 NOOP
                         │
                         ▼
        真实 action mask、版本与执行门控
```

安全边界不变：在线执行时，世界模型永远不能覆盖真实 action mask、ACK/lease/fencing 或最新 graph version。

## 4. 世界模型输入合同

推荐把一条因果转移固定成：

```python
Transition(
    graph_t,                 # 决策时可见的 belief 图
    received_evidence_t,     # decision_time 前已到达的信息
    action_t,                # 16 条边或 NOOP
    reward_t,
    cost_vector_t,           # uncovered/distance/load/switch/recovery...
    graph_tp1,               # 下一次可见 belief 图
    continuation_t,
    event_tp1,               # 只由 graph_t 与 graph_tp1 自动计算
    entity_valid_mask,
    feature_valid_mask,
    padding_mask,
    delta_time,
    schema_version,
)
```

关键约束：

- event 由 `t → t+1` 的可见观测生成，不能读取未来事件带；
- 训练样本必须保留 episode、scenario、seed、时间戳和版本；
- 静态字段从 `event_eligible_mask` 中排除；
- train/validation/test 按场景或 tape 切分，不能按单步随机切分；
- 动作记录必须是实际执行/确认后的动作，不能只记录策略原始建议。

## 5. 自动事件的具体设计

建议不要先定义一个扁平的“事件 ID”，而是按模态保留可解释目标。

### 5.1 Ordinal event：连续量的方向变化

适用字段：位置、距离、负载、vacancy duration、priority、confidence、severity、delay、通信质量、成本分量。

```text
delta_i = (x_(t+1,i) - x_(t,i)) / frozen_range_i

e_i = DOWN, delta_i < -C_i
      SAME, |delta_i| <= C_i
      UP,   delta_i > C_i
```

`frozen_range_i` 和 `C_i` 只能由物理范围或训练集统计确定，并在正式测试前冻结。

### 5.2 Nominal event：类别变化

适用字段：alive、task type、assigned UAV、target state、tracker、pending/confirmed、动作合法性。

```text
e_i = CHANGED if category_(t+1,i) != category_(t,i) else SAME
```

one-hot 应先还原成一个类别组再比较，避免一次类别切换被重复计数。

### 5.3 Structural event：图结构变化

建议编码：

```text
NODE_ADD / NODE_REMOVE
EDGE_ADD / EDGE_REMOVE
REASSIGN
ACTION_ENABLE / ACTION_DISABLE
```

当前固定 4/4/3 规模可先用 validity 和 relation change 表示，之后再升级为动态实体 token。

### 5.4 Evidence event：弱通信变化

建议编码：

```text
NEW_EVIDENCE / DUPLICATE / CONFLICT
CONFIDENCE_UP / CONFIDENCE_DOWN
CONFIRM / EXPIRE
```

这些事件来自已到达消息和 belief 更新。未到达消息、仿真器真值、未来确认结果不能进入在线输入。

## 6. GES 在图任务中的推荐实现

必须按模态计算，不能用整个图的一个全局密度：

```text
alpha_t^(m) = active_event_count / valid_event_eligible_count
```

第一版建议实现两个可切换版本：

```text
hard GES:   w_t^(m) = 1[alpha_t^(m) < tau_m]
smooth GES: 使用官方 EASimulus 的密度权重思想
```

主实验先用 hard GES，原因是阈值和边界行为容易审计；smooth GES 作为消融。不要把 EADream 的像素阈值 `0.05` 直接套到 UAV 图字段，图中有效维度、稀疏度和像素完全不同。

事件损失建议：

```text
L_event = Σ_m beta_m · w_t^(m) · CE/BCE(pred_event_m, auto_event_m)

L_WM = L_graph + L_reward + L_cost + L_cont
       + L_dyn_KL + L_event
```

类别不平衡可使用 focal loss或 class-balanced CE，但参数必须由训练集冻结。

## 7. GPPO 应该怎样读取世界模型

### 7.1 论文对齐主组

Actor 的每条候选边输入：

```text
[uav_embedding,
 region_embedding,
 candidate_edge_features,
 global_graph_pool,
 h_t,
 z_t]
```

Critic 输入：

```text
[global_graph_pool, h_t, z_t]
```

Event logits 不直接加入 actor/critic。这样改造不会破坏当前 16 条边 + NOOP 的动作接口，也能保留动态 action mask。

### 7.2 可选的显式事件风险组

如需研究“预测某类事件后提前避险”，可以额外建立：

```text
[h_t, z_t, predicted_semantic_event_risk]
```

但它应命名为 `EAWM-Risk-GPPO`，并与 `EAWM-GPPO` 单独对照。语义风险头只能使用训练期标签，线上不得偷看 truth event。

## 8. “事件偏好”必须拆成两种含义

### 8.1 含义 A：EAWM 的事件关注

这不是 preference learning。它包括：

- 自动事件目标；
- Event Predictor；
- GES；
- 事件感知 latent。

如果你的论文所说“事件偏好”只是模型更关注关键变化，建议统一改称“事件感知”或“事件关注”，避免审稿人误解为 PbRL。

### 8.2 含义 B：专家/人类偏好学习

如果确实要让专家比较“两段规划哪个更好”，应增加独立的 Event-conditioned Preference Reward Model，而不是修改 GES。

偏好样本：

```python
PreferencePair(
    segment_a,              # 同场景、相近初始 belief 的轨迹片段
    segment_b,
    label,                  # A / B / TIE / ABSTAIN
    event_boundary_type,
    annotator_id_hash,
    rubric_version,
)
```

片段不要随机截取，优先围绕自动检测到的事件边界取窗口，例如 `[t-3, t+8]`。A/B 必须尽量匹配初始状态、事件类型和难度，否则模型会学到场景差异而不是规划质量。

奖励模型输入：

```text
latent [h_t,z_t]
+ action edge embedding
+ auto-event embedding
+ validity masks
```

Bradley–Terry 训练目标：

```text
R_psi(segment) = Σ_t gamma^t r_psi(h_t,z_t,a_t,e_t)
P(A > B) = sigmoid(R_psi(A) - R_psi(B))
```

偏好标注界面可以展示这些解释指标，但不要把“人工总分”同时当标签又当输入：

- 区域空缺时长与覆盖恢复时间；
- 非法/过期动作数；
- 任务完成率；
- 总航程/平均距离；
- 最大负载与负载方差；
- 不必要切换/抢占次数；
- 关键事件后的响应延迟。

硬约束永远不交给偏好模型学习：非法动作、已失效 UAV、过期版本、未确认事件必须先由 action mask 和执行门控过滤。

训练时使用：

```text
r_total = r_env + lambda_pref · clip(normalize(r_pref)) - lambda_cost · cost
```

每个 GPPO rollout/epoch 内冻结奖励模型版本，记录 `reward_model_version`，避免 PPO 正在优化时奖励函数持续漂移。

## 9. 推荐实验矩阵

为了分别回答“世界模型有没有用”“事件学习有没有用”“偏好有没有用”，至少保留：

| 实验组 | 世界模型 | 自动事件头 | GES | 偏好奖励 |
|---|---:|---:|---:|---:|
| GPPO | 否 | 否 | 否 | 否 |
| WM-GPPO | 是 | 否 | 否 | 否 |
| EA-noGES-GPPO | 是 | 是 | 否 | 否 |
| EAWM-GPPO | 是 | 是 | 是 | 否 |
| PbGPPO | 否 | 否 | 否 | 是 |
| EAWM-PbGPPO | 是 | 是 | 是 | 是 |
| EAWM-Risk-GPPO（可选） | 是 | 是 | 是 | 否，显式风险输入 |

优先顺序：先跑通前四组，再决定是否做偏好模型。否则一旦结果提升，将无法判断来自潜状态、事件监督还是偏好奖励。

## 10. 实施顺序与验收门槛

### M0：冻结因果数据合同

- 完成 `Transition` schema、feature registry、event eligibility mask；
- 对时间戳、版本和在线可见性做单元测试；
- 任何未来字段进入输入都应测试失败。

### M1：自动事件生成器

- 实现 ordinal/nominal/structural/evidence 四类；
- 同一输入重复生成必须逐位一致；
- 输出每模态事件密度直方图，排查全零或全一塌缩。

### M2：无事件世界模型基线

- 先训练 `L_graph + L_reward + L_cost + L_cont + L_KL`；
- 检查一步预测、短期 rollout 和 calibration；
- 该阶段不修改 GPPO。

### M3：Event Predictor + GES

- 加事件头和 hard/smooth GES 开关；
- event F1/AUPRC 必须超过按频率猜测基线；
- 检查世界模型基础预测没有被 event loss 拖垮。

### M4：Shadow mode

- 在线维护 latent，但不影响动作；
- 检查 reset、stale action、乱序消息、并发事件下的隐状态一致性；
- transaction 失败时 latent 与 belief 一起回滚或重算。

### M5：接入 GPPO

- 只增加 `[h_t,z_t]`，保持动作语义和 action mask 不变；
- 从 `WM-GPPO → EA-noGES-GPPO → EAWM-GPPO` 逐级验收；
- 原 GPPO 的合法动作率、并发不变量和 checkpoint 流程不得退化。

### M6：可选偏好层

- 建立事件边界对齐的 A/B 片段；
- 先做离线 reward-model 一致性，再接 PPO；
- 单独报告标注者一致率、held-out preference accuracy 和 reward hacking 检查。

## 11. 官方代码的使用边界

官方仓库使用 GPL-3.0。若本项目不准备采用兼容的开源许可，不要直接复制官方实现代码；建议依据论文公式和接口思想自行实现，并在文档中注明方法来源。正式分发前应让项目负责人核对许可证义务。

本地 checkout 因官方仓库内 Atari ROM 的超长路径在 Windows 上未完整展开，但 EADream/EASimulus 核心源码已可审阅。本项目不需要 ROM，因此不影响上述设计；它不应被当成本项目的运行依赖。

## 12. 当前最小决策

现在不建议立即做偏好模型。先完成：

```text
M0 因果数据合同
→ M1 自动事件生成
→ M2 Graph World Model
→ M3 Event Predictor + GES
→ M4 shadow mode
→ M5 latent 接入 GPPO
```

只有当 `EAWM-GPPO` 相对 `WM-GPPO` 的增益和稳定性被验证后，再启动 `M6`。这条路线既保持当前 GPPO 可正常使用，也能清楚说明 EAWM 和偏好学习各自贡献。
