# 世界模型与 GPPO 输入设计及实施规划

> 注意：本文件是未取得指定论文时，按 PlaNet/RSSM 假设形成的第一版草案。用户随后提供的指定论文是 `2601.19336v1`（EAWM），最新权威方案请使用同目录的 `EAWM论文对齐版_世界模型与GPPO输入设计_9.1.md`。本草案仅保留作设计演变记录。

- 日期：2026-09-01
- 当前代码基线：`E:\Z博士\9.1日\GPPO-8.29`
- 论文依据：Hafner et al., *Learning Latent Dynamics for Planning from Pixels*（PlaNet，ICML 2019）
- 设计状态：输入与实验合同草案，不代表已经完成世界模型集成

## 1. 先回答：现在的 GPPO 能不能正常使用和规划？

结论是：**能正常完成当前定义下的动态任务分配，但不能把它称为完整世界模型规划或航迹规划。**

当前 GPPO 已经能够：

- 从 UAV、Region、Target 三类节点构成的异构图读取当前 belief 状态；
- 在 16 条 UAV–Region 候选边和 1 个 NOOP 中选择合法动作；
- 用 `action_mask` 屏蔽非法边；
- 在 UAV 损毁、目标发现/摧毁、区域空缺后进行局部重新分配；
- 处理事件延迟、重复、乱序和确认；
- 通过 graph/action version、ACK、lease、fencing 拒绝过期动作和重复持有者；
- 进行 GPPO 前向推理、最小 PPO 训练、checkpoint 保存和加载。

2026-09-01 在当前副本上重新运行的两个针对性测试均通过：

```text
GPPO-Adaptive 前向 + save/load：PASS
最小 GPPO PPO 训练 + save/load：PASS
```

但是当前 GPPO 做的是**任务分配决策**：选择“哪个 UAV 接哪个待分配 Region”，多次决策形成一个反应式分配过程。它不直接输出航迹点，不替代 `mppi/` 或 `search_planner/`；也不通过学习到的动态模型想象未来 1–5 步再选择动作。

因此准确表述应是：

> 当前 GPPO 是一个可运行、带安全约束的反应式动态图任务分配策略；完整的预测式规划仍需加入世界模型和可选的潜空间搜索器。

另外，GitHub 仓库没有正式 6 个 50k checkpoint，所以源码能训练/推理，不等于下载后已有可直接部署的正式训练策略。

## 2. 当前 GPPO 的真实输入

当前环境规模固定为：

```text
UAV = 4
Region = 4
Target = 3
候选 UAV–Region 边 = 4 × 4 = 16
动作数 = 16 + NOOP = 17
```

### 2.1 节点输入

| 节点 | 张量 | 当前字段 |
|---|---:|---|
| UAV | `[4, 12]` | alive、sensor available、task one-hot(3)、归一化 x/y、区域负载、target one-hot(4) |
| Region | `[4, 12]` | 归一化中心 x/y、priority、workload、vacancy duration、assigned UAV one-hot(5)、pending、当前分配是否合法 |
| Target | `[3, 16]` | target type one-hot(2)、discovered/tracked/destroyed、可见位置 x/y、region one-hot(4)、tracker one-hot(5) |

### 2.2 边输入

| 关系 | 边特征 |
|---|---|
| UAV → Region / Region → UAV | capable、距离、当前持有、UAV 负载、通信质量，共 5 维 |
| Region → Region | adjacency，1 维 |
| Target ↔ Region | discovered、destroyed，共 2 维 |
| UAV ↔ Target | tracking、discovered，共 2 维 |
| UAV → UAV | communication quality，1 维 |

现有 PPO-MLP 把上述节点和边特征展平为固定长度输入；现有 GPPO 通过关系独立的消息传递层编码。两者当前都没有时序记忆输入。

## 3. 从 PlaNet/RSSM 映射到本项目

PlaNet 的核心不是“预测一张未来图片”，而是：

1. 从部分观测推断当前潜在 belief；
2. 用上一状态和动作预测下一潜在状态；
3. 同时保留确定性记忆和随机潜变量；
4. 在潜空间预测多步奖励/状态；
5. 每次只执行规划序列的第一个动作，下一时刻收到新观测后重新规划。

本项目的对应关系为：

| PlaNet | 本项目 |
|---|---|
| 像素观测 `o_t` | 当前 belief 异构图 + 已到达 Observation + 执行反馈 |
| 连续动作 `a_t` | 离散 UAV–Region 边或 NOOP |
| RSSM state | 确定性记忆 `h_t` + 随机扰动状态 `z_t` |
| observation model | 下一 belief 图/图摘要预测头 |
| reward model | 下一步成本、空缺、恢复、距离和负载预测头 |
| CEM 连续规划 | 离散 masked beam search / rolling horizon |

这里不能直接使用 PlaNet 的图像解码器和连续动作 CEM。应保留 RSSM 的动作条件转移和随机/确定性双通路，把视觉编码器替换为异构图集合编码器，把 CEM 替换为适合离散变动作集合的 beam search。

## 4. 推荐的世界模型输入合同

世界模型的一个训练样本应以“决策时刻”为单位：

```python
WorldModelStep(
    graph_observation_t,       # 当前 belief 图
    received_evidence_t,       # decision_time 前已收到的观测证据
    previous_action,           # 上一 UAV–Region 边或 NOOP
    execution_feedback,        # accepted/stale/ACK/lease/reward/cost delta
    delta_time,
    decision_audit,            # 版本和 cursor，仅用于审计
    future_targets             # 只用于离线监督
)
```

### 4.1 输入 A：当前 belief 图

直接复用当前 `HeteroGraphState`，但世界模型应使用共享节点编码器与集合池化，不能依赖固定 4/4/3 节点数：

```text
UAV features    → shared MLP → mean + max ┐
Region features → shared MLP → mean + max ├→ graph embedding e_t
Target features → shared MLP → mean + max ┘
relation edges  → relation encoders ──────┘
```

建议首版图 embedding 为 96 维。节点数量变化时，集合编码不改变输出长度。

### 4.2 输入 B：已到达但未必已确认的观测证据

这些字段在决策时真实可见，可以进入世界模型，但只能读，不能直接改变 belief 或 mask：

- event type；
- affected UAV/Region/Target ID；
- source type、signal type、source ID；
- positive/negative；
- confidence、severity；
- emitted_at、received_at 及由此计算的 delay/age；
- sequence、duplicate 标记；
- confirmation status；
- distinct positive/failure source 数；
- heartbeat miss 数；
- 距离确认阈值的归一化 progress。

不要只按事件类型做一个全局计数，否则会丢失“哪架 UAV、哪个 Region、哪个 Target 有风险”。推荐构造三类实体证据：

```text
uav_evidence:    [N_uav, 8]
region_evidence: [N_region, 6]
target_evidence: [N_target, 10]
global_evidence: [16]
```

首版字段建议：

- UAV：damage progress、suspicion/probe 状态、confidence、message age、delay、source diversity、duplicate ratio、communication reliability；
- Region：vacancy progress、confidence、message age、delay、source diversity、pending age；
- Target：discovery progress、destruction progress、两类 confidence、两类 message age、source diversity、conflict、tracker reliability、feature-valid；
- Global：最近窗口观测量、重复率、乱序率、丢失估计、冲突数、确认数、pending 数、合法动作数/比例及缺失标记。

### 4.3 输入 C：上一动作

不要只输入动作整数，因为 UAV/Region 数量变化后整数语义不稳定。输入：

- action type：edge / NOOP；
- 被选 UAV 的节点 embedding；
- 被选 Region 的节点 embedding；
- 被选边的 5 维原始特征；
- 动作提交时的 mask-valid 标志。

该动作编码建议压缩到 32 维。

### 4.4 输入 D：执行反馈

- accepted / rejected / stale-rejected；
- ACK 状态；
- lease 创建、续期、撤销；
- fencing 是否变化；
- 上一步 reward；
- uncovered、distance、load gap、switch、recovery delay 的分量变化；
- legal action count delta；
- pending region count delta；
- `delta_time`。

建议压缩到 32 维。

### 4.5 padding 与有效性

历史窗口首版取最近 16 个决策时刻。episode 开头不足 16 步时使用零 padding，但必须同时提供：

```text
padding_mask: [16]
feature_valid flags
```

全零只能表示数值为零，不能隐含“字段可用且确定无风险”。

### 4.6 严格禁止作为线上输入

- 尚未到达的 Observation；
- 仿真器 truth event；
- truth-only `occurred_at`；
- future event tape；
- 未来图、未来 action mask、未来 reward；
- 未来 exact planner 答案；
- 事件生成 seed、隐藏场景参数。

`graph_version`、`action_version`、observation cursor 和 tape hash 只能用于对齐、幂等和审计，不作为模型业务特征。

## 5. 推荐的 Graph-RSSM v0

### 5.1 状态更新

```text
图与证据编码 e_t
上一动作编码 a_(t-1)
上一执行反馈 f_(t-1)

h_t = GRU(h_(t-1), z_(t-1), a_(t-1), f_(t-1), delta_t)
prior:     p(z_t | h_t)
posterior: q(z_t | h_t, e_t)
belief:    b_t = [h_t, z_t]
```

起始尺寸：

| 组件 | 维度 |
|---|---:|
| graph/evidence token | 128 |
| action embedding | 32 |
| feedback embedding | 32 |
| deterministic hidden `h_t` | 64 |
| stochastic latent `z_t` | 32 |
| GPPO global context | 32 |

现有 `E:\Z博士\src\uav_assignment\world_model.py` 只能作为原型参考：它是“摘要序列 → GRU → 下一事件分类”，没有动作条件 prior/posterior、随机潜变量、多步目标、异构图输入和校准，不能直接作为 Graph-RSSM 完成品移植。

### 5.2 预测头

至少输出：

```python
WorldModelOutput(
    global_context,           # [32]
    uav_risk,                 # [N_uav, 4]
    region_risk,              # [N_region, 4]
    target_risk,              # [N_target, 6]
    event_risk_h1_h3_h5,      # [3, 4]，四类事件，多标签
    event_uncertainty,        # [3, 4]
    next_state_delta,         # 业务状态增量
    cost_prediction,          # reward/cost 分解
    reliability              # age/conflict/ensemble disagreement
)
```

事件目标必须是多标签，因为同一窗口可能同时出现多个事件。建议同时区分：

- 未来 truth event 风险：仅作离线监督标签；
- 未来 Observation 到达风险；
- 未来 confirmed belief-change 风险。

这三者不能混成一个标签，否则模型无法区分“事件尚未发生”“事件发生但未观测”“观测已到但尚未确认”。

### 5.3 损失

```text
L = λ_event * focal_BCE(horizon 1/3/5)
  + λ_graph * Huber(next graph/business delta)
  + λ_cost * Huber(cost components)
  + β * KL(q(z_t) || p(z_t))
  + λ_over * multi-step latent overshooting
  + λ_cal * calibration regularization
```

第一阶段可以先关闭 latent overshooting，只验证单步时间轴正确；单步通过后再打开 3/5 步 overshooting。

## 6. 世界模型怎样接入 GPPO

### 6.1 第一阶段：只提供上下文，不做模型内搜索

先保持当前 GPPO 动作逻辑不变：

```text
current graph → AHGNN → node embeddings
Graph-RSSM → global/entity risk context
两者残差融合 → masked actor + critic
```

建议注入：

```python
uav_hidden    += proj_uav_risk(uav_risk)
region_hidden += proj_region_risk(region_risk)
target_hidden += proj_target_risk(target_risk)
pooled        += proj_global(global_context)
```

投影层最后一层零初始化。世界模型关闭或 context 全零时，新模型应与旧 GPPO 的 logits、value 和 deterministic action 数值等价。

为了公平判断世界模型收益，PPO-MLP 必须获得同一份 global context；GPPO 的实体级风险注入应另做清楚标记的结构消融。

### 6.2 第二阶段：风险感知 GPPO

冻结世界模型，仅训练 context adapter 和 PPO/GPPO。目标是回答：历史 belief 与未来风险特征是否改善恢复，而不是立即证明“规划”。

世界模型仍然无权修改：

- belief；
- pending_regions；
- action_mask；
- graph/action version；
- lease/fencing。

### 6.3 第三阶段：真正的潜空间滚动规划

只有离线预测和风险接入均通过后，再增加离散 beam planner：

1. 从当前 posterior belief `b_t` 出发；
2. 对当前合法 UAV–Region/NOOP 动作展开候选；
3. 用 RSSM prior 想象未来 3–5 个决策事件；
4. 预测累计 vacancy、recovery、distance、load、switch 和失败风险；
5. 使用 `expected cost + uncertainty penalty` 排序；
6. 只执行最优序列的第一个动作；
7. 下一时刻收到真实观测后重新推断 posterior 并重规划；
8. 最终执行前始终用真实当前 action mask 和版本合同重新校验。

离散动作与动态 mask 更适合 beam search，而不是直接照搬 PlaNet 的连续动作 CEM。首版建议：

```text
horizon = 3
beam width = 8
每步保留 top-k GPPO proposals + NOOP
风险高时才启用 planner
```

建议路由器：

```text
低风险：原 GPPO
中高风险：WM-GPPO
强资源争用/未来释放：Beam-RSSM planner
当前不可行：Current-Pending Exact Planner / 安全回退
```

## 7. stale 与并发规则

正确顺序必须是：

```text
begin_decision()
→ 冻结 decision_time / graph_version / action_version / observation cursor
→ 构造世界模型窗口
→ 世界模型 + GPPO/Planner 推理
→ ActionSubmission.from_decision(...)
→ submit_action()
```

如果推理期间新事件到达导致版本变化：

- 拒绝旧动作；
- 不向 PPO rollout 写 transition；
- 不把该动作记录为已执行世界模型 transition；
- 不永久推进在线 RSSM hidden；
- 不增加 accepted decision step；
- 使用新 cursor 和新版本重新推理。

首版用最近 16 个 token 重算 RSSM 状态，暂不维护可变在线 hidden，可使 stale 重试天然幂等。

## 8. 实施规划与门槛

### WM0：冻结合同与零上下文兼容

交付：schema、白名单/denylist、配置、零 context adapter、旧 checkpoint 兼容测试。

通过条件：世界模型关闭时原测试通过；固定图上新旧 logits/value/action 等价。

### WM1：数据记录与无泄漏审计

交付：decision-level recorder、tape/instance 级 split manifest、SHA-256 inventory、字段审计。

通过条件：`received_at <= decision_time`；truth-only 字段扫描为零；同一 tape/instance 不跨 split。

### WM2：离线 Graph-RSSM

先做频率、last-value、MLP、普通 GRU、确定性图模型基线，再训练随机 Graph-RSSM。

暂定通过条件：

- held-out tape + instance 上优于朴素基线；
- event PR-AUC 相对频率基线提升至少 20%；
- macro-F1 至少 0.70；
- ECE 不高于 0.10；
- 分别报告 horizon 1/3/5；
- CPU 额外 P95 时延不高于 2 ms。

### WM3：影子运行

世界模型实时预测但不输入策略。

通过条件：预测不改变 belief/mask/version/action；重复回放一致；stale 重试不重复消费 Observation。

### WM4：冻结世界模型接入 PPO/GPPO

对照：PPO、GPPO、WM-PPO、WM-GPPO。

暂定通过条件：非法动作 0、未拦截 stale 0、最终不可行不恶化；累计空缺下降至少 10% 或恢复延迟下降至少 5%，距离/负载退化不超过 5%。

### WM5：潜空间 beam planner

对照：WM-GPPO、WM-GPPO + beam、Current-Pending Exact Planner。

通过条件：收益必须出现在真实 held-out 回放，而不是只在世界模型预测回报中；P95/P99 端到端时延满足硬件预算。

### WM6：Extreme-V2 一次性验收

冻结模型、阈值、路由和 planner 参数后，仅在未见过的新事件带运行一次。旧 `extreme_scenarios` 只能作为开发集。

## 9. 必须新增的代码边界

```text
ppo_allocation/world_model/
├── schema.py
├── feature_builder.py
├── history.py
├── recorder.py
├── graph_encoder.py
├── rssm.py
├── heads.py
├── losses.py
├── calibration.py
├── context_adapter.py
├── beam_planner.py
└── diagnostics.py
```

需要小范围修改：

- `random_event/graph.py`：可选 context/risk 字段；
- `random_event/models.py`：零初始化的上下文注入；
- `random_event/trainer.py`：复制 context，stale 时不写 buffer/history；
- `random_event/environment.py`：decision snapshot 增加时间和 observation cursor；
- `random_event/runtime_bridge.py`：只读 Observation/evidence snapshot；
- `event_runtime/state_machine.py`：只读 confirmation progress 接口。

## 10. 最小测试集合

最重要的断言：

```text
世界模型预测不能修改 belief
世界模型预测不能修改 action_mask
未来 Observation/truth 不能进入线上特征
stale retry 不能重复消费历史
zero context 不能改变旧 GPPO 输出
执行动作必须由真实 mask/version 再校验
```

建议测试：

- feature whitelist / truth denylist；
- no future observation；
- tape + instance split isolation；
- history cursor idempotency；
- zero-context parity；
- old checkpoint compatibility；
- shadow-mode no state mutation；
- stale retry no double consume；
- concurrent-event multilabel；
- probability calibration；
- 1/3/5-step rollout error；
- world-model 与 end-to-end latency。

## 11. 推荐立即执行的顺序

```text
确认论文口径与方法名称
→ 冻结 WorldModelStep schema
→ 实现只读 Observation/evidence snapshot
→ 实现 recorder 和无泄漏测试
→ 生成 tape/instance 分组数据集
→ 跑 frequency / last-value / GRU / deterministic graph 基线
→ Graph-RSSM 离线训练与校准
→ shadow mode
→ frozen WM + PPO/GPPO
→ 离散 beam planner
→ Extreme-V2
```

第一版不要同时改世界模型、奖励函数、动作空间和 GPPO 网络。先冻结世界模型，给现有 PPO/GPPO增加可回退 context，才能判断收益究竟来自时序预测还是其他改动。

## 12. 最终推荐版本

首个可实现版本冻结为：

```text
当前三类节点 belief 图
+ 已到达未确认 Observation 证据
+ 上一动作和执行反馈
+ 16 个决策时刻历史
+ Graph-RSSM（h=64, z=32）
+ horizon 1/3/5 四类事件多标签预测
+ 32 维 global context + 实体风险特征
+ 冻结世界模型
+ PPO/GPPO 零初始化残差接入
+ shadow / zero-context 自动回退
```

这个版本先解决当前 GPPO 最大的结构性短板：无记忆、看不到未确认证据的累积趋势、不能表达未来扰动风险。等该版本在新隐藏集上确认有效，再增加潜空间 beam planning，才可以把系统称为“世界模型辅助的预测式 GPPO 规划”。
