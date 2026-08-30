# Random-Event V0 历史材料索引

本页记录 Execution-Preemption V1 之前的随机事件 PPO/GPPO 实验。它们保留作学姐旧方法、PPO 基线、工程来源和结果边界的追溯材料，不再作为当前开发入口。

## 保留的最小证据

- `ppo_allocation/`：旧 PPO/GPPO 环境、训练和评估代码；
- `ppo_allocation/results/random_event/minimum_validation_50k_2afa8ec/`：minimum-validation 封存证据；
- `ppo_allocation/results/random_event/tapes/preliminary_validation_protocol/` 与 `preliminary_test_protocol/`：配置仍引用的固定事件带；
- `ppo_allocation/results/random_event/preliminary_eval/` 的环境信息、总体评估和分组分析；
- `experiments/extreme_scenarios/results_20260827/` 的固定 tapes、聚合指标、配对差值、报告和审计结论；
- `handoff/`：旧 Gate、决策记录和复现命令。
- `docs/archive/MIMO_START_HERE_LEGACY.md`：已从根目录移走的旧协作入口。

## 从最新分支清理的内容

以下内容均为可再生成或与当前研究无关的材料：

- preliminary evaluation 的 2,400 个逐回合原始日志及其索引；
- extreme-scenarios 的 420 个逐回合原始轨迹及其索引；
- 旧 minimum-validation 的 smoke、dry-run 和 test-train 临时输出；
- `perch/jj.pdf`：与当前算法仓库无代码引用的直升机对比材料；
- 根目录旧临时说明 `read me .md`。

清理不会改变保留的聚合数值和当前 README 图表，但旧 SHA-256 inventory 因指向已移除原始轨迹而不再保留。

## 恢复方式

清理前完整状态位于 Git 提交：

```text
524d3a4 docs: add execution preemption result figures
```

需要审计单个历史文件时，应从该提交按文件恢复，不要将整批旧日志重新提交到当前分支。例如：

```powershell
git show 524d3a4:path/to/file
```

如果要复跑旧实验，应在独立临时目录生成新的运行输出，并使用对应 manifest 和 checkpoint SHA-256 验证来源。
