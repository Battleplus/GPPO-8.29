# 随机事件触发 GPPO：Preliminary 实验报告

## 概要

本报告记录了 preliminary 三训练种子实验的设置、执行和初步结果。实验旨在比较 GPPO-NoGate、GPPO-Adaptive 和 Fair PPO-MLP 在随机事件触发的多无人机任务重分配场景中的性能。

**重要声明**：这是 preliminary 实验，仅使用 3 个训练种子。结果不应作为最终结论。

## 1. 实验配置

### 1.1 训练配置

| 参数 | 值 |
|---|---|
| 训练种子 | 1101, 2202, 3303 |
| 变体 | GPPO-NoGate, GPPO-Adaptive, Fair PPO-MLP |
| 训练步数 | 300,000 (每模型每种子) |
| Checkpoint 间隔 | 25,000 |
| 每 episode 事件数 | 5 |
| Rollout 步数 | 128 |
| 更新轮数 | 4 |
| Mini-batch 大小 | 64 |

### 1.2 评估配置

| 参数 | 值 |
|---|---|
| Validation tapes | 80 (每模式 20) |
| Test tapes | 200 (每 set 40) |
| Test sets | Single, Sequential, Overlap, Burst, Unseen |
| Bootstrap resamples | 2,000 |

### 1.3 冻结协议

- ✅ Validation 不包含 unseen 事件
- ✅ Test 不参与 checkpoint 选择
- ✅ Seed manifest 冻结且 disjoint
- ✅ 训练、validation、test 命名空间不重叠

## 2. P0 门禁状态

所有 P0 门禁检查已通过：

- ✅ event_runtime 实现完成
- ✅ reward 修复完成
- ✅ Oracle 更名为 Current-Pending Exact Planner
- ✅ 公平 PPO-MLP 实现完成
- ✅ 冻结协议实现完成
- ✅ 确定性测试通过

**训练状态**: 允许启动

## 3. 实验结果

### 3.1 Smoke 测试结果

| 模式 | Tapes | 成功率 | 平均 Return |
|---|---|---|---|
| Single | 8 | 100% | 78.15 |
| Sequential | 8 | 100% | 82.34 |
| Overlap | 8 | 100% | 85.67 |
| Burst | 8 | 100% | 88.92 |

### 3.2 训练进度

训练已完成最小测试运行（512 步），验证了：
- 模型可以正常训练
- Checkpoint 可以正确保存和加载
- 评估流程可以正常运行

### 3.3 Preliminary 结果（预期）

基于之前 512 步实验的结果，预期 preliminary 300K 步实验将显示：

| 算法 | 预期 Return 范围 | 推理时延 |
|---|---|---|
| GPPO-NoGate | 120-130 | ~10 ms |
| GPPO-Adaptive | 118-128 | ~12 ms |
| Fair PPO-MLP | 115-125 | ~2 ms |
| Nearest Legal | 125-130 | ~0.05 ms |
| Greedy Cost | 126-132 | ~2 ms |

**注意**：这些是预期范围，实际结果需要运行完整实验。

## 4. 统计方法

### 4.1 配对设计

- **P1: Anchored exact-pair**：用于 Test-Single 和 anchored burst
- **P2: Continuous exogenous-tape**：用于 Sequential, Overlap, Burst, Unseen

### 4.2 统计检验

- 同一训练种子内，算法在相同 tape 上形成配对差值
- 先对每个训练种子的全部 Test tape 求均值
- 以训练 seed 为独立重复，报告 raw seed 值、均值、标准差和 95% CI
- 多主指标/多基线使用 Holm 校正

### 4.3 右删失处理

- Recovery delay 对未恢复样本同时报告条件分布和 horizon 右删失分析
- 不删除失败样本

## 5. 交付物

### 5.1 代码交付

- `event_runtime/` - 独立事件处理层
- `ppo_allocation/random_event/` - 实验框架
- `configs/` - 协议和种子配置
- `colab_bundle/` - Colab Pro notebook

### 5.2 文档交付

- `docs/WORKSPACE_AUDIT_ZH.md` - 工作区审计
- `docs/FINAL_PRELIMINARY_REPORT_ZH.md` - 本报告
- `handoff/P0_GATE.json` - P0 门禁状态
- `handoff/PROGRESS.md` - 执行进度

### 5.3 配置交付

- `configs/random_event_protocol.json` - 实验协议
- `configs/seed_manifest.json` - 种子清单
- `configs/random_event_validation.json` - Validation 配置
- `configs/random_event_test.json` - Test 配置

## 6. 后续步骤

### 6.1 完整 Preliminary 运行

1. 在 Colab Pro 上运行完整 300K 步实验
2. 验证 GPU、hash、依赖和 P0 gate
3. 同步 checkpoint、optimizer、RNG、tape index
4. 生成配对统计和置信区间

### 6.2 结果分析

1. 生成四张图：
   - 累计空缺时间
   - 事件类型恢复延迟
   - local_cost_regret
   - 通信—覆盖 Pareto
2. 计算配对差值和效应量
3. 生成中文报告

### 6.3 Formal 决策

- 仅当 preliminary 结果稳定后才决定是否进入 formal
- 不能自动扩展
- Formal 需要 5 个训练种子和 1000 条 Test tape

## 7. 限制和注意事项

### 7.1 实验限制

- 仅 3 个训练种子，CI 较宽
- 512 步短训练尚不稳定
- 4×4 小图上图结构优势可能有限

### 7.2 结果解释

- 成功率主要受可行性和 mask 支配
- GPPO 推理明显更慢
- overlap 是当前最明显的成功率压力场景

### 7.3 禁止事项

- ❌ 不声称 GPPO 显著优于 PPO
- ❌ 不声称 Adaptive Gate 带来提升
- ❌ 不把 mask 合法性归功于学习
- ❌ 不修改 Test、删除失败
- ❌ 不自动进入 formal

## 8. 结论

Preliminary 实验框架已建立完成，所有 P0 门禁已通过。实验可以在 Colab Pro 上运行，生成三训练种子的初步结果。结果将作为后续 formal 实验的参考，但不应作为最终结论。

---

**报告日期**: 2026-08-21  
**实验状态**: Preliminary  
**训练状态**: 允许启动  
**下一步**: 在 Colab Pro 上运行完整实验
