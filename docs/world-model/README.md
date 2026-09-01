# GPPO 世界模型与事件感知研究资料

本目录整理 GPPO、EAWM 世界模型和事件偏好相关的研究记录。文档讨论的是**设计与实施规划**；除非文档明确写明已实现并通过测试，否则不代表对应模块已经进入生产代码。

## 最重要的结论

世界模型不会取代 GPPO，两者分工如下：

```text
belief 图、历史动作与已到达证据
              ↓
世界模型：维护记忆、学习潜状态、预测未来变化
              ↓
GPPO：在 UAV-Region 候选边与 NOOP 中选择动作
              ↓
真实 action mask、版本校验、ACK/lease/fencing
              ↓
执行
```

- 世界模型负责理解历史、处理部分可观测性，并在后续阶段生成 imagined trajectories。
- GPPO 仍是动作决策器，不改变现有候选边、NOOP 和安全门控语义。
- EAWM 的 Event Predictor 首先是世界模型的辅助训练目标，不是直接给策略使用的人工事件分数。
- 专家或人类偏好学习属于额外的 PbRL 层，应与 EAWM 主实验分开。

## 推荐阅读顺序

1. [`acceptance/2026-09-10_世界模型最小闭环验收目标.md`](acceptance/2026-09-10_世界模型最小闭环验收目标.md)
   - 9 月 10 日验收范围、量化门槛、倒排日期和 GPPO checkpoint 前置条件。
2. [`source/世界模型整合实施计划_2026-08-30.docx`](source/世界模型整合实施计划_2026-08-30.docx)
   - 12 页原始实施计划，供核对完整表格、职责、风险和证据清单。
3. [`research/GPPO-8.29研究笔记.md`](research/GPPO-8.29研究笔记.md)
   - 了解当前仓库结构、GPPO 已有能力和限制。
4. [`current/EAWM论文对齐版_世界模型与GPPO输入设计_9.1.md`](current/EAWM论文对齐版_世界模型与GPPO输入设计_9.1.md)
   - 对照 EAWM 论文，查看世界模型数据合同、自动事件、GES 和 GPPO 接口。
5. [`current/EAWM官方仓库迁移评估与事件偏好设计.md`](current/EAWM官方仓库迁移评估与事件偏好设计.md)
   - 查看官方仓库哪些部分可借鉴、哪些不适合直接迁移，以及事件偏好/PbRL 的独立设计。
6. [`archive/世界模型与GPPO输入设计及实施规划_9.1.md`](archive/世界模型与GPPO输入设计及实施规划_9.1.md)
   - 早期方案，仅保留设计演进记录；以 `current/` 中的论文对齐版本为准。

## 目录结构

```text
docs/world-model/
├── README.md
├── acceptance/
│   └── 2026-09-10_世界模型最小闭环验收目标.md
├── source/
│   └── 世界模型整合实施计划_2026-08-30.docx
├── current/
│   ├── EAWM论文对齐版_世界模型与GPPO输入设计_9.1.md
│   └── EAWM官方仓库迁移评估与事件偏好设计.md
├── research/
│   └── GPPO-8.29研究笔记.md
└── archive/
    └── 世界模型与GPPO输入设计及实施规划_9.1.md
```

## 设计主线

### 第一阶段：事件感知表示，不改变 GPPO 决策边界

```text
因果数据合同
→ 自动事件生成器
→ Graph World Model 基线
→ Event Predictor + GES
→ shadow mode
→ latent 接入 GPPO
```

第一阶段中，GPPO 输入增加世界模型潜状态 `[h_t, z_t]`，但 Event Predictor 输出不直接送入 actor。在线执行仍以真实 action mask 和最新状态版本为准。

### 第二阶段：可选的预测规划

世界模型在潜空间模拟不同动作造成的未来图状态、奖励、成本与 continuation，GPPO/actor-critic 在 imagined trajectories 上学习。该阶段完成后，系统才具有严格意义上的世界模型辅助预测规划能力。

### 第三阶段：可选的事件条件偏好学习

如果需要专家比较两段规划，应单独训练偏好奖励模型，并设置独立实验组：

```text
GPPO
WM-GPPO
EA-noGES-GPPO
EAWM-GPPO
PbGPPO
EAWM-PbGPPO
```

这样才能区分性能提升来自世界模型、事件监督、GES，还是偏好奖励。

## 当前状态说明

- 当前 GPPO 可以作为反应式动态图任务分配器运行。
- 当前 GPPO 并不等同于多步世界模型预测规划器。
- 归档训练证据记录了三个 GPPO-Adaptive 50k seed，但公开仓库未提交 checkpoint 二进制文件；没有取得匹配权重前，不能表述为“仓库内已有可直接加载的训练模型”。
- README 中的 300,000 steps 是 PPO 与 GPPO 六个 run 的合计；GPPO 部分是 3 × 50,000 = 150,000 steps，不是单个 GPPO 训练了 300,000 steps。
- 本目录文档已经给出 Graph-EAWM 输入、损失、事件类型、实验矩阵和实施顺序。
- 世界模型代码接入、完整训练和正式对照实验仍需按文档中的里程碑执行。

## 参考资料

- EAWM 论文：[*From Observations to Events: Event-Aware World Model for Reinforcement Learning*](https://arxiv.org/abs/2601.19336)
- EAWM 官方实现：[MarquisDarwin/EAWM](https://github.com/MarquisDarwin/EAWM)

## 许可证提醒

EAWM 官方仓库采用 GPL-3.0。若本项目不准备使用兼容许可证，建议依据论文公式自行实现，不要直接复制官方源代码；正式分发前请核对许可证义务。

---

整理日期：2026-09-01
