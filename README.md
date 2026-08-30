# GPPO-8.29：执行中动态扰动与多无人机任务重分配

当前研究分支的主线是 **Execution-Preemption V1**：无人机正在执行任务时，如果再次发生紧急任务、UAV 损毁、能量不足或通信异常，系统能够确认事件、判断优先级、安全中断，并重新分配任务。

> 分支：`research/execution-preemption-v1`<br>
> 基线：`main@a4207527f713e6f15dcdbc538134aeaca28a03ac`（只读，未修改）<br>
> 当前结论：机制、接口、三种确定性基线和开发回放已完成；正式训练已在独立证据工作区启动，但完整 campaign 与 Hidden-V1 尚未完成。

## 当前进度

| 模块 | 状态 | 能说明什么 |
|---|---|---|
| 执行进度、暂停、抢占、迁移、恢复、RTB | 已实现 | 已具备执行中再分配的运行时语义 |
| 四类核心扰动 | 已实现 | 新任务、UAV 失败、能量不足、通信异常可进入统一流程 |
| 五节点异构图 | 已实现 | UAV、Task、Region、Target、Event 信息可显式建模 |
| 事件证据确认 | 已有框架 | 支持来源、时效、重复/迟到过滤和确认状态机；真实 IC 数据尚未接入 |
| 安全仲裁与原子提交 | 已实现 | 推理期间再来事件时，旧方案作废并基于新版本重算 |
| 三种确定性比较基线 | 已实现 | `senior_legacy_method_v1`、`greedy_priority_v1`、`beam_mpc_v1` |
| 开发事件带 | 已完成 | 10 类 × 20 条，三基线共 600 次安全回放通过 |
| 专项测试 | 120/120 PASS | 证明工程合同与安全不变量通过，不代表学习效果 |
| 正式新训练 | 进行中 | UAV4/seed1101 阶段已封存，seed2202 正在独立工作区运行；完整 campaign 尚未完成 |
| Validation / Hidden-V1 | 未开始 | 目前不能宣称 GPPO 优于 PPO |

## 突发事件处理链

```text
IC/仿真原始数据
  → Observation（来源、序列号、时间戳、置信度）
  → 去重、乱序与过期过滤
  → ConfirmedEvent（任务级确认事件）
  → P0-P4 确定性安全仲裁
  → 保存进度、暂停/抢占/迁移/RTB
  → 生成安全候选 UAV-Task 集
  → PPO / GPPO / Greedy / Beam-MPC 提案
  → request/version/SHA/mask 校验
  → 原子提交；若期间再来事件则整批作废并重算
  → ACK、执行与恢复验证
```

PPO/GPPO 只在安全层筛选后的候选中决定“哪架 UAV 执行哪个任务”，不能绕过能量、通信、唯一所有权、不可抢占任务和版本约束。

## 主要阅读入口

- [工程总说明：事件检测、五节点图、抢占机制、PPO/GPPO 区别与汇报结论](docs/EXECUTION_PREEMPTION_ENGINEERING_README_ZH.md)
- [当前可恢复进度](docs/EXECUTION_PREEMPTION_PROGRESS_ZH.md)
- [Execution-Preemption V1 协议](docs/EXECUTION_PREEMPTION_V1_PROTOCOL_ZH.md)
- [算法与安全层边界](docs/ALLOCATION_BOUNDARY_V1_ZH.md)
- [三种确定性比较基线](docs/EXECUTION_BASELINES_V1_ZH.md)
- [训练、奖励与指标合同](docs/EXECUTION_TRAINING_CONTRACT_V1_ZH.md)
- [PPO/GPPO 统一适配器](docs/POLICY_ADAPTER_V1_ZH.md)
- [文档分类索引](docs/README.md)

## 当前代码结构

```text
execution_preemption/           当前主线：执行状态、抢占控制、图、策略适配与训练框架
tests_execution_preemption/     当前主线专项测试
experiments/dynamic_preemption/ 当前 10×20 开发事件带与回放证据
configs/execution_preemption_v1.json
docs/EXECUTION_*.md              当前合同、进度、结论和汇报说明

ppo_allocation/                 旧随机事件 PPO/GPPO 基线与封存结果
event_runtime/                  旧事件观测/确认/ACK/lease/fencing 运行时
experiments/extreme_scenarios/  旧极端场景聚合结果
handoff/                        历史 Gate、provenance 和复现记录
```

旧模块仍保留，是因为新方法需要与学姐旧方法和 PPO 基线做同合同对照；它们不是当前开发入口。历史材料说明见 [旧随机事件研究索引](docs/archive/LEGACY_RANDOM_EVENT_INDEX_ZH.md)。

## 快速验证

在仓库根目录运行当前专项测试：

```powershell
python -m unittest discover -s tests_execution_preemption -v
```

生成开发事件带并执行三种确定性基线安全回放：

```powershell
python scripts\generate_dynamic_preemption_tapes.py
python scripts\run_execution_baselines.py
```

这些命令用于验证机制，不等同于正式训练或模型效果评估。

## 仓库保留策略

- 保留：当前源代码、合同、测试、固定事件带、聚合表、结论图和最小可复现实证；
- 不保留在分支最新版本：可再生成的逐回合原始日志、逐步轨迹、过期 smoke/dry-run 输出和无关大文件；
- 旧实验的完整原始文件仍可从 Git 历史提交 `524d3a4` 恢复；
- 模型 checkpoint 二进制不进入仓库，正式实验必须按 manifest 与 SHA-256 单独封存。

## 研究边界

- 当前 600 次开发回放证明的是安全机制和接口能运行，不是 PPO/GPPO 效果；
- 旧实验中的 100% 事件成功率只代表最终恢复到合法分配状态，不代表检测率、deadline 达成率或最优性；
- 五节点图提高表达能力，也增加消息传递和推理开销，后续必须与两节点消融比较；
- 正式结论必须来自冻结合同下的多 seed 训练、Validation、一次性 Hidden-V1，以及 PPO、GPPO、学姐旧方法和 Beam-MPC 的同条件对照。
