# GPPO 50k 最小验证实验：START HERE

> 本文件是 `plan/minimum-validation-50k` 分支的唯一主执行入口。
> 后续 AI / 代码代理应先完整阅读本文件，再执行任何修改。
>
> 当前目标不是完成大规模 benchmark，而是用尽可能短的等待时间，得到一组可信、可复现、导师能够直接理解的数据，回答：**GPPO-Adaptive 在当前随机事件驱动资源分配场景下是否比 PPO-MLP 更好，优势具体体现在哪里。**

---

## 0. 权威状态与分支边界

仓库：

```text
Battleplus/GPPO-8.20
```

当前工作分支：

```text
plan/minimum-validation-50k
```

该分支从旧 S4/E4 远端证据 HEAD 创建：

```text
e1ab61525c6f4158dd60605e344f06d84541e49e
```

旧 S4 frozen source：

```text
0b48d20c58028503818a953ced93216912db1c4a
```

重要：另一个本地工作副本曾产生 `c5de3bcf...` 的 3×3×100k effectiveness 实现，但该提交没有在远端 GitHub 上成为本分支的权威来源。后续 AI 不得依赖一个 GitHub 上不存在的本地提交；可以参考其思路，但必须以本分支真实文件为准重新实现并审计。

旧 `plan/effectiveness-100k` 方案不再是当前默认执行目标。100k 只作为未来扩展方案参考。

---

# 1. 当前研究目标

当前只做一个 **Minimum Validation Experiment**。

需要回答四个问题：

1. `GPPO-Adaptive` 是否比 `PPO-MLP` 更好？
2. 优势体现在哪些任务指标？
3. 三个训练 seed 上趋势是否基本一致？
4. 优势是否主要出现在 `Overlap / Burst / Unseen` 这类动态性更强的场景？

本阶段不要求：

- 完成 300k benchmark；
- 完成 100k effectiveness benchmark；
- 做所有消融实验；
- 做复杂 checkpoint selection；
- 做完整 Validation → Freeze → Test 流水线；
- 证明所有指标都全面优于 PPO。

本阶段成功标准是：**用最小实验成本得到一组公平、可复现、有物理意义的数据，能够支持或否定“Adaptive 比 PPO 更适合当前随机事件场景”的判断。**

---

# 2. Frozen Minimum Validation Contract

## 2.1 Methods

只训练两个方法：

```text
PPO-MLP
GPPO-Adaptive
```

暂不训练：

```text
GPPO-NoGate
```

原因：NoGate 是机制消融，用于回答“收益来自图结构还是 adaptive gate”。它不是当前“先证明比 PPO 好”所必需的。如果导师后续要求机制解释，再追加 NoGate。

## 2.2 Training seeds

固定：

```text
1101
2202
3303
```

两个模型必须使用完全相同的训练 seed。

不得：

- 挑 seed；
- 删除表现差的 seed；
- Test 后重跑某个 seed；
- 某个模型使用不同 seed。

## 2.3 Training budget

固定：

```text
50,000 accepted decision steps / model / seed
```

总训练量：

```text
2 models × 3 seeds × 50,000
= 300,000 accepted decision steps
```

这是当前正式最小验证预算。

## 2.4 Checkpoints

只保留：

```text
25,000
50,000
```

总 checkpoint：

```text
2 models × 3 seeds × 2 checkpoints
= 12 checkpoints
```

正式评估统一使用：

```text
50,000-step checkpoint
```

`25k` 仅用于：

- 检查训练是否正常；
- 看一个简单学习趋势；
- 诊断 50k 结果是否完全异常。

**不做 checkpoint selection。**

禁止使用 held-out evaluation 结果来选择 25k 还是 50k。

---

# 3. Held-out Evaluation Bank

不再建立复杂的 Validation 选模流程。

训练完成后，两个模型的三个 seed 使用完全相同的固定 held-out evaluation bank：

```text
100 cases total

Single       20
Sequential   20
Overlap      20
Burst        20
Unseen       20
```

必须满足：

- evaluation seeds 与 training seeds 隔离；
- 两个算法看到完全相同的 case/tape；
- evaluation bank 在正式运行前冻结；
- 不因为结果不好而改 case；
- 不用 evaluation 结果反向调 reward / architecture / budget。

这里的 `Unseen` 仍只用于 held-out evaluation，不允许泄漏到训练。

---

# 4. 核心指标

只把三个指标作为当前导师汇报主线。

## 4.1 Primary 1 — Final infeasible rate ↓

回答：

> 随机事件发生后，最终有多少 case 无法形成满足约束的可执行资源分配？

这是最直观的鲁棒性指标。

## 4.2 Primary 2 — Recovery latency ↓

回答：

> UAV damage、目标发现、通信异常或多事件扰动发生以后，系统需要多久恢复到合理分配？

这是动态恢复能力的核心指标。

## 4.3 Secondary — Cumulative weighted vacancy ↓

回答：

> 从扰动发生到恢复期间，资源/任务位置空缺了多少、持续了多久？

这个指标用于解释恢复过程的资源利用效率。

## 4.4 其他指标

`fixed J`、reward、mask、安全协议等仍可记录，用于 sanity check / appendix，但不要作为当前主结论的主要叙事。

性能提升不得通过违反以下约束获得：

```text
mask
version
lease
fencing
concurrency invariants
reward invariant
```

---

# 5. 最终要回答的结果结构

最终先看整体：

```text
PPO-MLP vs GPPO-Adaptive
```

再按场景拆分：

```text
Single
Sequential
Overlap
Burst
Unseen
```

重点看：

```text
Overlap
Burst
Unseen
```

理想但非预设的结果形态：

- Single：差异可能较小；
- Sequential：Adaptive 开始出现收益；
- Overlap / Burst：Adaptive 优势更明显；
- Unseen：Adaptive 性能退化更小。

禁止预设 Adaptive 必须赢；如果结果不支持，就如实报告。

---

# 6. 最低统计要求

本阶段不做过度复杂的统计包装。

必须报告：

```text
每个 training seed 的原始结果
mean
standard deviation
paired same-case difference
improvement vs PPO (%)
```

训练 seed 是独立训练单位。

不要把 100 个 evaluation case 当成 100 个独立训练 replicate。

如果需要做显著性/置信区间，可以追加，但不是当前完成最小验证的前置条件。

---

# 7. 最终导师汇报输出

只要求 1 张表 + 2 张图。

## Table 1 — Summary

```text
Method | Infeasible ↓ | Recovery Latency ↓ | Weighted Vacancy ↓
PPO-MLP
GPPO-Adaptive
Improvement vs PPO (%)
```

同时保留 3 个 seed 的原始数值。

## Figure 1 — Scene-wise comparison

横轴：

```text
Single / Sequential / Overlap / Burst / Unseen
```

比较：

```text
PPO-MLP
GPPO-Adaptive
```

优先展示：

```text
infeasible rate
recovery latency
```

如一张图无法清晰表达，可分别输出两张，但不要额外增加不必要图表。

## Figure 2 — Seed stability

显示：

```text
seed 1101
seed 2202
seed 3303
```

检查 Adaptive 相对 PPO 的方向是否一致。

---

# 8. 允许形成的结论

只有数据支持时，允许使用类似：

> 在相同 50k 训练预算和相同 held-out 随机事件评估集下，GPPO-Adaptive 相比 PPO-MLP 在最终不可行率和事件后恢复时间上表现更好；该趋势在三个训练随机种子上基本一致，并且在 Overlap、Burst 和 Unseen 场景中优势更明显。

如果只有部分指标改善，应精确写清楚哪些指标改善。

禁止写：

```text
全面碾压 PPO
所有场景都显著更优
已经证明适用于 Isaac Sim / SFC 高保真环境
```

除非有对应数据。

当前 L1 结果只能表述为：

```text
在冻结的轻量随机事件仿真环境下……
```

---

# 9. 工程实现要求

后续 AI 应尽量复用现有 S4 结构，不进行与最小验证无关的大重构。

需要真正实现：

```text
formal methods = [PPO-MLP, GPPO-Adaptive]
formal seeds = [1101, 2202, 3303]
formal budget = 50000
checkpoint grid = [25000, 50000]
expected checkpoints = 12
fixed evaluation checkpoint = 50000
```

必须更新所有真正影响 formal protocol 的地方，例如按实际代码依赖检查：

```text
configs/random_event_protocol.json
configs/seed_manifest.json
相关 train/evaluation config
ppo_allocation/random_event/phase_j.py
formal CLI
Gate/hash coverage
相关 tests
```

不要只改文档。

---

# 10. Exact Progress 必须保留

用户需要看到真实进度。

正式新版本应提供 read-only / observability-only 的 exact progress heartbeat。

建议输出：

```text
live_progress.json
```

至少包含：

```json
{
  "variant": "GPPO-Adaptive",
  "seed": 1101,
  "total_steps": 32768,
  "target_steps": 50000,
  "update_count": 0,
  "last_checkpoint": null,
  "elapsed_seconds": 0,
  "steps_per_second": 0,
  "estimated_remaining_seconds": 0,
  "updated_at": "..."
}
```

要求：

- `total_steps` 必须直接来自 trainer 的真实 accepted steps；
- heartbeat 写入不得改变 RNG；
- 写失败不得改变训练状态；
- atomic replace；
- 不作为训练/评估输入；
- 不改变 checkpoint 内容和算法语义。

同时提供只读 PowerShell monitor，显示：

```text
Current model
Current seed
Exact total_steps
Current run %
Campaign %
Checkpoint x/12
steps/s
ETA
PID
RAM
stderr status
```

不要再用 checkpoint 间线性估算来冒充 exact steps。

---

# 11. Gate 与测试

因为 formal protocol 会变化，旧 S4 Gate 不可直接复用。

实现修改后必须重新运行 required tests，并重新生成 Gate。

至少验证：

```text
formal methods exactly PPO-MLP + GPPO-Adaptive
formal seeds exactly 1101/2202/3303
formal budget exactly 50000
checkpoint grid exactly [25000, 50000]
expected checkpoint count == 12
fixed evaluation checkpoint == 50000
no checkpoint selection path in minimum-validation formal flow
no Train/Evaluation seed leakage
progress reports exact trainer.total_steps
progress writer does not alter RNG
progress writer failure does not alter training
save/load works for both models
reward invariant unchanged
concurrency invariants unchanged
source/protocol/seed hashes present
```

Gate 只有全部 required tests PASS 且 source/hash attestation 一致时：

```json
"training_allowed": true
```

不得复制旧 Gate JSON。

---

# 12. Smoke

正式训练前只做短 smoke。

目标：

- 两个模型都能启动；
- 三个 seed 路径正确；
- progress heartbeat 正常；
- checkpoint 正常；
- save/load 正常；
- output namespace 正确；
- stderr 无异常；
- Gate 可在 formal entry 再验证。

Smoke 输出必须使用独立 namespace，不能混进 formal 50k 结果。

---

# 13. 新 campaign 输出目录

不要复用旧目录：

```text
preliminary_formal_s4_restart1
```

建议新目录：

```text
ppo_allocation/results/random_event/minimum_validation_50k_run1
```

先确认 `_relative_path()` 行为，避免：

```text
ppo_allocation/ppo_allocation/results/...
```

---

# 14. 旧 S4 300k campaign 的处理

旧 S4 campaign 可能仍在本地运行或保留结果。

后续 AI 不得擅自：

- kill 旧进程；
- 删除旧 checkpoint；
- 覆盖旧目录；
- resume；
- 把旧 checkpoint 拼入新 50k 实验。

如果用户明确决定停止旧 300k campaign，应记录：

```text
ABORTED_BY_PROTOCOL_CHANGE
usable_as_formal_result = false
```

已有 checkpoint 只能作为：

```text
pilot/reference/debug evidence
```

不得作为新 Minimum Validation 的正式结果。

---

# 15. 执行顺序

后续 AI 必须按以下顺序：

```text
M0. current-state audit
↓
M1. freeze 50k minimum-validation protocol
↓
M2. implement protocol + exact progress
↓
M3. update/add tests
↓
M4. rebuild P0 Gate
↓
M5. short smoke
↓
STOP
↓
report READY / NOT READY
↓
等待用户明确授权
↓
M6. formal 2×3×50k training
↓
training completion audit
↓
M7. fixed held-out evaluation
↓
M8. summary statistics + table + figures
```

在 M5 完成前，不得启动正式训练。

没有用户明确授权，不得执行 M6。

---

# 16. M0–M5 当前交付要求

后续 AI 现在应只做到 READY，不要自动训练。

完成后返回：

```text
1. changed files
2. source commit SHA
3. protocol SHA256
4. seed manifest SHA256
5. source tree SHA256
6. test count / pass count
7. Gate summary
8. smoke summary
9. progress monitor text sample
10. exact formal command（只报告，不执行）
11. READY / NOT READY
```

同时说明：

```text
formal training started = false
held-out evaluation started = false
```

---

# 17. 明确禁止事项

未经用户单独明确授权，不得：

```text
开始正式 2×3×50k training
继续/恢复旧 campaign
运行 held-out evaluation
改 Test/Evaluation case 以追求更好结果
增加训练预算到 100k/300k
加入 GPPO-NoGate
并行 6/9 个 worker
改 reward 来追求更好结果
删除失败 seed
删除旧 campaign
merge master
push 到其他无关分支
```

若实现失败：

```text
STOP → 修复 → rerun tests/Gate/smoke
```

不得通过手工修改 Gate/报告绕过失败。

---

# 18. Future Extension（当前不做）

只有在 50k 最小验证结果不足以回答研究问题时，才考虑：

```text
A. 增加 GPPO-NoGate 做消融
B. 扩展到 100k
C. 增加更完整统计
D. 做 300k benchmark
E. 做 SFC / Isaac Sim 高保真验证
```

这些都不是当前 Minimum Validation 的前置条件。

---

# 19. 一句话执行原则

> **先用 2 个模型 × 3 个 seed × 50k，在完全相同的 100-case held-out 随机事件集上比较 infeasible rate、recovery latency 和 weighted vacancy；先回答“Adaptive 是否比 PPO 好、好在哪里”，其余复杂实验以后再补。**
