# 本地上下文索引

## 唯一工作区

`E:\Z博士\8.20\54_20-master`

所有代码、配置、测试、Notebook、日志和报告都必须留在该目录。不得修改其他目录。

## 必读资料

| 顺序 | 本地文件 | 用途 |
|---:|---|---|
| 1 | `docs/archive/MIMO_START_HERE_LEGACY.md` | 历史安全边界和入口 |
| 2 | `handoff/MIMO_MASTER_TASK_ZH.md` | 完整执行任务书 |
| 3 | `handoff/COPY_PROVENANCE.json` | 复制来源、Git 状态、文件和参考资料哈希 |
| 4 | `references/REAL_EVENT_DETECTION_AND_CONCURRENCY_DESIGN_ZH.md` | Truth/Observation/ConfirmedEvent、ACK、lease、fencing 设计依据 |
| 5 | `docs/RANDOM_EVENT_BASELINE_AUDIT_ZH.md` | 原始 PPO 环境与指标问题审计 |
| 6 | `docs/RANDOM_EVENT_GPPO_DESIGN_ZH.md` | GPPO 图状态、实验协议和统计设计 |
| 7 | `docs/RANDOM_EVENT_PRELIMINARY_REPORT_ZH.md` | 旧 512-step 结果及不可沿用的结论 |
| 8 | `docs/RANDOM_EVENT_RESULTS_TEMPLATE_ZH.md` | 后续报告结构与措辞边界 |
| 9 | `configs/random_event_protocol.json` | 当前协议草案，需按任务书审计修正 |
| 10 | `configs/seed_manifest.json` | Train/Validation/Test seed 命名空间 |
| 11 | `configs/random_event_train.json` | 训练配置草案 |
| 12 | `configs/random_event_validation.json` | Validation 配置草案，必须移除 unseen |
| 13 | `configs/random_event_test.json` | 冻结 Test 配置草案 |

## 需要续写而非重建的代码

- `event_runtime/`
- `ppo_allocation/random_event/`
- `ppo_allocation/tests_random_event/`
- `ppo_allocation/run_random_event_experiment.py`
- `ppo_allocation/utils/sb3_compat.py`

这些文件可能只完成了一部分。先运行测试和检查接口，再做局部修改，不得假定已完成，也不得整目录重写。

## 原始兼容性入口

- `ppo_allocation/tests/test_cpp_interface.py`
- `ppo_allocation/cpp_bridge.py`
- `ppo_allocation/reallocation_service.py`
- `ppo_allocation/results/models/run_20260605_210049/maskable_ppo_uav_task_allocation.zip`

旧模型仅称为 `Legacy MLP-PPO`，不能作为公平 PPO-MLP 主基线。

## 参考归档

- `references/54_20-master-original.zip`
- SHA-256：`7B55726CCA79E7380C63C60FF4522EF5C09851B4CCA326774549B12265726D97`
- 用途：只验证来源，禁止解压覆盖。

## 禁止事项

- 不访问或修改旧工作目录。
- 不从网络下载另一个同名仓库覆盖本项目。
- 不在 P0 gate 通过前运行长训练。
- 不使用 Test 选模或调参。
- 不把旧 preliminary 结果写成 GPPO 优于 PPO 的证据。
- 不自动 commit、push、reset 或删除用户结果。
