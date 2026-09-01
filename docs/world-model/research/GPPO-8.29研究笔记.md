# GPPO-8.29 项目研究笔记

- 研究日期：2026-09-01
- 项目地址：https://github.com/Battleplus/GPPO-8.29
- 本地源码：`E:\Z博士\9.1日\GPPO-8.29`
- 获取方式：Git 连接两次被重置，随后从 GitHub codeload 下载 `main` 分支源码快照并解压。

## 一句话结论

这是一个面向随机事件、资源失效和弱通信条件的多无人机动态任务分配研究原型。它的工程化证据链和并发安全机制做得比较认真，但现有数据只支持“GPPO-Adaptive 在特定的强资源争用/等待未来资源释放场景中有研究价值”，不支持“GPPO 全面优于 PPO”。

## 研究问题与方法

项目比较两种共享环境、动作掩码、奖励、训练预算和评估用例的策略：

1. `PPO-MLP`：把异构图中的节点和边特征按固定顺序展平，由 MLP 编码；
2. `GPPO-Adaptive`：对 UAV、区域、目标三类节点做关系感知消息传递，再用自适应门控融合自身状态与邻域信息。

正式 minimum-validation 使用：

- 2 个模型；
- 3 个训练 seed：1101、2202、3303；
- 每个 run 50,000 accepted decision steps，共 300,000 steps；
- 固定使用 50k checkpoint，不做 checkpoint selection；
- 100 个 held-out case：Single、Sequential、Overlap、Burst、Unseen 各 20 个。

## 系统结构

核心闭环是：真实事件 → 弱通信观测 → 事件确认状态机 → belief/异构图更新 → PPO 或 GPPO 决策 → graph/action version 校验 → command/ACK/lease/fencing → 局部重分配。

关键目录：

- `ppo_allocation/random_event/`：图构造、PPO trainer、PPO/GPPO 模型、环境、奖励和评估；
- `event_runtime/`：事件观测、确认、乱序/重复处理、命令 ACK、lease 和 fencing；
- `experiments/extreme_scenarios/`：7 类极端压力场景及机器可读结果；
- `handoff/`：P0 Gate、哈希证明、决策记录和复现说明；
- `scripts/`：训练 worker、Gate、进度监控、极端测试和 stale-decision 审计。

代码中的 GPPO 不是简单地把图展平：每种关系有独立消息线性层，消息按目标节点做均值聚合；门控在“新提议表示”和“原节点表示”之间逐维融合。Actor 对候选 UAV–区域边打分，并额外输出 NOOP；Critic 使用三类节点的全局均值池化。PPO 基线读取同一批节点、边特征和动作掩码，但其展平输入维度固定，因此目前并不能天然支持不同 UAV/区域数量的直接泛化。

## 正式 50k 结果

仓库封存的 `analysis_summary.json` 给出的结论较克制：

- 两种方法的合法覆盖率相同；
- GPPO 的总体 episode return 仅高约 0.0144，效应很小；
- GPPO 的恢复延迟低约 0.0084、累计空缺低约 0.0503，同样很小且不稳定；
- GPPO 的归一化距离高约 0.0112，三 seed 区间稳健地偏向 PPO；
- GPPO 的平均推理延迟高约 14.55 ms，明显慢于 PPO；
- seed 1101 偏向 PPO，2202 偏向 GPPO，3303 接近打平，结论缺乏 seed 稳定性；
- Burst 的三个 seed 都偏向 GPPO，Unseen 的三个 seed 都偏向 PPO，但各场景三-seed 置信区间仍跨过 0。

所以正式结果只能表述为：在当前固定小图、50k 预算和 100-case bank 下，两者质量表现总体接近，GPPO 没有形成全面优势，且付出了显著时延成本。

## 极端压力测试

仓库又做了 7 类 post-hoc 探索场景、42 条事件带、420 个策略回合。GPPO 最清晰的收益出现在 `tracking_saturation_release`：

- 平均恢复延迟相对 PPO 降低约 0.678；
- 累计空缺降低约 12.817；
- episode return 增加约 0.610；
- 代价是归一化距离增加约 0.024；
- 推理慢约 12.37 ms。

这说明图关系建模可能更适合“当前无可用资源，需要识别资源竞争关系并等待未来释放”的状态。它在 `event_storm_8`、`task_churn` 和乱序报告中没有稳定胜出；两个模型都是前馈策略，也没有真正解决部分可观测和长时记忆问题。

这些极端场景是在看过正式结果后设计的开发集，不能当作新的独立 held-out test。若据此继续改算法，必须冻结一套未见过的 Extreme-V2 hidden bank 做一次性验证。

## 可信度判断

值得肯定：

- 训练 seed、预算、checkpoint、held-out bank 和主指标有冻结合同；
- PPO 与 GPPO 使用相同输入信息、动作空间、mask、奖励和用例；
- 有 graph/action version、ACK、lease、fencing 和 stale-action rejection；
- P0 Gate 归档记录为 130/130 PASS，并包含协议、seed manifest 和源码树 SHA-256；
- 结果目录保留逐回合记录、成对差异、汇总、只读复核和 SHA-256 inventory；
- README 没有夸大结论，明确承认 GPPO 未全面胜出。

需要保留意见：

- 只有 3 个独立训练 seed，统计把握有限；
- 正式 bank 只有 100 cases，很多成功率指标接近天花板；
- 极端场景是 post-hoc 开发集，存在适应性分析偏差；
- 仓库未提交正式六个 50k checkpoint，只能核对结果证据，无法开箱复跑 checkpoint 实验；
- 当前没有执行中任务的完整暂停、抢占、迁移、进度保留和恢复事务；
- 尚未把信息年龄、TTL、置信度、通信可靠性纳入策略图特征；
- 尚未验证 8/16/32 UAV 等动态图规模泛化；
- 没有仓库级 `LICENSE`，公开使用、修改和再分发的授权边界不清；
- 没有 `pyproject.toml`/`setup.py`，项目仍偏研究脚本集合，安装和模块化体验一般；
- Isaac Sim 不是 pip 依赖的一部分，高保真场景不能按最小依赖直接复现。

## 本地轻量验证

在本机源码快照上运行了：

```powershell
Set-Location E:\Z博士\9.1日\GPPO-8.29\ppo_allocation
python -m unittest discover -s tests_random_event -v
```

结果为 130 项中 126 项通过、1 项失败、3 项报错。剩余问题均与本地复现条件有关：

- 当前源码是 GitHub ZIP 快照，没有 `.git`，依赖 commit/tree 的完整性测试无法通过；
- 当前环境缺少 `sb3_contrib`，两个 legacy compatibility 路径不能运行；
- 本机为 Python 3.14.4、NumPy 2.5.0、PyTorch 2.13.0+cpu，并非项目冻结环境。

这次测试不能替代仓库声明的冻结环境复现，但能确认主要环境、事件、并发、版本校验、模型前向/保存加载和轻量训练测试在较新环境中大部分可运行。

项目要求的复现环境为 Python 3.11.5、NumPy 1.26.4、PyTorch 2.7.1、Gymnasium 1.2.3、Stable-Baselines3/sb3-contrib 2.8.0。

## 建议的下一步

优先级建议：

1. 先建立独立 Python 3.11.5 环境，安装锁定依赖，并通过真正的 Git clone 保留 `.git`；
2. 向项目作者取得与 manifest SHA-256 匹配的 6 个 50k checkpoint，完成只读复现；
3. 新建未见过的 Extreme-V2 bank，检验 `tracking_saturation_release` 优势能否复现；
4. 增加任务执行状态机、剩余工作量、抢占/迁移/恢复和切换成本；
5. 将 AoI/TTL/置信度/通信可靠性编码进图，并加入循环记忆或 belief-state estimator；
6. 对 8/16/32 UAV 与变化区域数做规模泛化；
7. 同时比较 Rolling-Horizon/Beam-MPC、精确规划器和蒸馏后的低时延 MLP；
8. 补充 LICENSE、统一包管理、CI 和一键复现脚本，再考虑对外发布或论文复现。

## 最终判断

如果目标是“做一个可审计的动态多无人机任务分配研究原型”，这个仓库已有不错基础，尤其是事件确认、版本化动作提交和证据封存。如果目标是“证明 GPPO 在真实弱通信、多规模无人机系统中优于 PPO”，当前证据明显不够。最值得继续追的是资源强争用场景下的结构化优势，同时用新的隐藏测试集、更多 seed、动态图规模和记忆机制排除偶然性。
