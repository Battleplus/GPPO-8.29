# 随机事件触发 GPPO 实验结果报告模板

> 本模板只定义如何报告证据，不预设 GPPO 必须优于基线。所有 `【待填】` 必须由冻结协议下的原始日志或统计程序替换；缺失结果应写“未测”，不能填 0。

## 1. 一句话结论

本次为【预实验 / 正式实验】。在【测试集合名称】上，GPPO-Adaptive 相对 PPO-MLP 的主要指标【指标名】变化为【绝对差，方向】，95% CI【下界，上界】，配对样本数 `n=【】`；【置信区间不跨 0，因此在本协议内有条件性证据 / 置信区间跨 0，不能声称存在稳定优势】。对于【失败的事件类型或模式】，GPPO【如实描述失败、退化或未恢复】，因此结论不扩展到【未覆盖条件】。

措辞规则：

- **预实验**：至少 3 个训练 seed、每个测试 bank 至少 200 条冻结 event tape。只能写“初步证据”“在当前条件下”，不能写“普遍优于”或“已充分收敛”。
- **正式实验**：5 个训练 seed、每个测试 bank 1000 条冻结 event tape，并完成预注册的统计比较和多重比较校正。即便达到此规模，也只对已测试的任务规模、扰动分布和协议成立。
- 少于预实验规模时统一标为 **smoke/实现验证**，只证明代码链路可运行，不能作算法优劣结论。

## 2. 实验身份与可追溯性

| 项目 | 值 |
|---|---|
| 结论等级 | 【smoke / preliminary / formal】 |
| Git commit | `【】` |
| Python / PyTorch / 设备 | 【】 |
| requirements lock SHA-256 | `【】` |
| protocol SHA-256 | `【】` |
| seed manifest SHA-256 | `【】` |
| 训练 seed | 【逐个列出，不只写数量】 |
| 验证集选模规则 | 【】 |
| checkpoint 及 SHA-256 | 【】 |
| 测试 event tape bank 及 SHA-256 | 【】 |
| 原始事件日志 | 【路径】 |
| 汇总 JSON | 【路径】 |
| 聚合/绘图代码版本 | 【】 |

必须明确 train、validation、test 的 event seed 完全隔离；测试集一经冻结不得用于调参、选 checkpoint 或修改奖励权重。

## 3. 协议与公平性

任务背景保持为 4 UAV、4 Region、3 Target，任务模式为 SEARCH / TRACK / IDLE。每个普通 episode 默认【5】个事件，事件间隔和 burst 结构由冻结 event tape 决定。

所有算法应读取同一条 tape 的相同外源字段：`event_id`、`event_type`、`occurred_at`、`observed_at`、`source_event`、受影响实体、`severity`、`payload`、`event_seed`、`state_version`。若不同算法此前动作导致同一外源事件产生不同内生影响，必须分别记录 `intended_affected_regions` 与 `actual_affected_regions`，报告 `endogenous_effect_divergence`，不能把它们描述成完全相同的状态轨迹。

比较算法：

1. GPPO-Adaptive；
2. GPPO-NoGate；
3. 原始 MLP-PPO；
4. Masked Random；
5. Nearest Legal；
6. Min Load；
7. Greedy Cost；
8. Exhaustive Oracle（仅在可穷举规模上作为参考上界/后悔值基准）。

## 4. 四事件类型覆盖与 trace

下表每个正式测试 bank 都必须填写。事件数为 0 表示该测试不具备相应覆盖，不能只在总平均里隐藏。

| 事件类型 | tape 数 | 事件数 | 成功数/率 | 恢复延迟（已观测 n/总 n） | 最终不可行数 | trace 样例 |
|---|---:|---:|---:|---:|---:|---|
| `UAV_DAMAGE` | 【】 | 【】 | 【】 | 【】 | 【】 | 【event_id + 日志行/JSON path】 |
| `TARGET_DISCOVERED` | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| `TARGET_DESTROYED` | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| `REGION_VACANCY` | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |

每个 trace 至少能沿以下链条追踪：

`occurred_at → observed_at → event_queue → graph_version → decision_version → stale/recompute → action edge/NOOP → pending_regions → recovery/final infeasible`。

连续事件另需给出 single、sequential、overlap、burst、unseen 五类 test bank 的 tape 数、每 tape 事件数分布和事件类型分布。

## 5. 主要结果

### 5.1 按算法汇总

每个单元建议写 `mean ± SD [95% CI], n`；恢复失败的延迟属于右删失，不得用 0 替代。成功率同时给出分子/分母。

| 算法 | 事件成功率 ↑ | 合法覆盖率 ↑ | 累积加权空缺时间 ↓ | 恢复延迟 ↓ | 距离 ↓ | 负载差 ↓ | 切换次数 ↓ | 最终不可行率 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPPO-Adaptive | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| GPPO-NoGate | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| PPO-MLP | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| Masked Random | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| Nearest Legal | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| Min Load | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| Greedy Cost | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| Exhaustive Oracle | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |

### 5.2 按事件模式与类型分层

分别报告 single、sequential、overlap、burst、unseen；再按四类事件拆分成功率、恢复延迟和累积空缺时间。若 Simpson 悖论导致总体结论与分层结论方向不同，以分层结果和事件分布解释为主。

| 模式 | 事件类型 | GPPO-Adaptive 成功率 | PPO-MLP 成功率 | 配对差及 95% CI | GPPO 恢复延迟 | PPO 恢复延迟 |
|---|---|---:|---:|---:|---:|---:|
| single | 【四类逐行】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| sequential | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| overlap | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| burst | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| unseen | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |

## 6. 配对统计检验

配对单位是同一训练 seed 下的同一条 event tape（或协议明确的 anchored snapshot），不是把不同 tape 的样本强行配对。训练 seed 是评估算法稳定性的独立重复单位；实例 bootstrap 不能替代多训练 seed。

预注册主要比较：

- GPPO-Adaptive vs PPO-MLP；
- GPPO-Adaptive vs GPPO-NoGate；
- GPPO-Adaptive vs Greedy Cost。

| 比较 | 指标与差值定义 | 配对 n | A 均值 | B 均值 | 平均配对差 | 95% bootstrap CI | Cohen dz / rank-biserial | 原始 p | Holm 校正 p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive − MLP | 【正/负代表谁更好】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| Adaptive − NoGate | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| Adaptive − Greedy | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |

报告时必须先写效应量和置信区间，再写显著性。对于成功率可用配对二元检验或对 tape 成功率作预注册配对分析；恢复延迟必须说明未恢复事件的右删失处理（如 capped horizon + 失败率并列报告，或生存分析）。

## 7. 机制指标：为什么有效或为什么失败

| 算法/阶段 | pre-mask 非法概率 ↓ | mask 比例 | Actor entropy | Critic value loss | explained variance | approx KL | clip fraction | Gate mean/std/p10/p50/p90 | Gate gradient norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPPO-Adaptive train | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| GPPO-Adaptive test | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 |
| GPPO-NoGate | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | 【】 | N/A | N/A |

还应报告推理时延（mean/p95）、事件到首动作时延、通信触发次数、通信字节数、通信抑制率、stale action 拒绝次数和修复次数。Mask 后动作 100% 合法只说明约束实现正确；`pre_mask_invalid_probability` 下降才可支持策略学习了约束。

## 8. 图表

使用以下命令生成四张固定图：

```powershell
python -m ppo_allocation.random_event.plotting `
  --summary results/random_event/summary.json `
  --raw results/random_event/raw.json `
  --output-dir results/random_event/figures `
  --metric episode_return
```

1. `01_event_type_mode_recovery.png`：事件类型 × 模式的成功率及恢复延迟；
2. `02_algorithm_performance_95ci.png`：算法性能均值和 95% CI；
3. `03_training_diagnostics.png`：loss、return、pre-mask invalid mass、Gate；
4. `04_event_timeline_graph_version.png`：事件到达、queue、pending、graph version 时间线。

图注必须写清样本单位、误差条定义、越大/越小越好，以及失败事件是否进入延迟统计。空白图表示输入 JSON 缺少相应记录，不得据此作结论。

## 9. 失败、反例与限制

必须单列，而不是只写正向均值：

- GPPO 低于哪一个基线、发生在哪个事件类型/模式/seed；
- 是否存在“总体提高但 burst 或 unseen 退化”；
- 未恢复事件、临时不可行和最终不可行分别有多少；
- Gate 是否饱和、塌缩或没有梯度；Critic 是否无解释力；
- 随机事件分布、4/4/3 小规模和模拟探测延迟限制了哪些外推；
- Exhaustive Oracle 是否因规模限制只覆盖部分样本；
- 同一外源 tape 下是否存在内生状态分歧，以及它如何影响配对解释。

推荐失败措辞：

> 在【模式/事件】上，GPPO-Adaptive 的【指标】比【基线】差【差值】，95% CI【】。因此当前结果不支持“GPPO 在随机扰动下全面更优”。可能机制与【证据指标】一致/尚无证据，需要通过【预注册的下一项实验】验证。

## 10. 最终可采用结论

> 在【任务规模】、【事件分布】、【训练 seed 数】和冻结的【test bank】条件下，【算法】相对【基线】在【主指标】上表现为【效应量及 95% CI】。优势【是否】在四类事件与 single/sequential/overlap/burst/unseen 中保持；【列出例外】。本结果属于【smoke/preliminary/formal】，因此只支持【恰当范围】的结论，不证明对真实传感器故障、通信丢包或未测试规模普遍有效。

