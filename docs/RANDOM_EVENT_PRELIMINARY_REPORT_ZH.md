# 随机事件触发 GPPO：三训练种子初步实验报告

## 结论先行

这次实验已经证明“代码链路可运行”，但**没有证明 GPPO 普遍优于原 MLP-PPO 或规则算法**。每个 GPPO 模型只训练了 512 个决策步（4 次 PPO 更新），Critic 的 explained variance 约为 0，Adaptive Gate 也几乎停留在 0.5。因此，以下结果只能标为 preliminary（初步结果），不能写成收敛后的算法结论。

在 200 条相同事件带上，三训练种子平均后的 episode return 为：GPPO-NoGate `122.546`、GPPO-Adaptive `121.453`、旧 MLP-PPO `121.040`、Masked Random `120.161`、Exhaustive Oracle `125.107`、Min Load `125.813`、Nearest Legal `126.363`、Greedy Cost `127.032`。也就是说：

- NoGate 相对 MLP-PPO 的三种子平均增益为 `+1.506`，但训练种子 t 95% CI 为 `[-7.977, 10.989]`；Adaptive 为 `+0.413 [-10.063, 10.889]`。两者都不能排除换一个训练种子后优势消失。
- GPPO 对距离和负载均衡的平均值优于 MLP-PPO、Masked Random，但 return 仍输给 Greedy Cost、Nearest Legal、Min Load 和 Oracle。
- 以 return 看，本轮整体赢家是 Greedy Cost，而不是 GPPO；两种 GPPO 中 NoGate 的三种子均值比 Adaptive 高 `1.093`，但样本太少，不能由此否定 Gate。
- GPPO 推理明显更慢：NoGate 平均约 `9.970 ms/decision`、Adaptive 约 `11.526 ms/decision`，旧 MLP-PPO 为 `1.111 ms/decision`，规则算法约 `0.048–1.696 ms/decision`。
- 成功率集中在 `0.939–0.943`，所有算法的 weighted uncovered 都是 `0.82`，说明当前成功率主要受事件本身可行性和严格动作掩码控制，而不是已经学出了明显不同的策略质量。

## 1. 实验范围与可追溯性

| 项目 | 本轮实际值 |
|---|---:|
| GPPO 变体 | GPPO-NoGate、GPPO-Adaptive |
| 独立训练种子 | 1、2、3 |
| 每个 checkpoint 训练量 | 512 steps，4 次 PPO update |
| 测试事件带 | 200 条 |
| 每条事件数 | 5 |
| 测试事件总数 | 1,000 |
| 模式 | single、sequential、overlap、burst，各 50 条 |
| Smoke bank | 80 条，各模式 20 条；每条 3 事件，共 240 事件 |

测试 bank 的四类事件计数如下。它们来自 manifest 的 1,000 个事件，不是按名义概率推算：

| 事件类型 | Test-200 数量 | 占比 | Smoke-80 数量 |
|---|---:|---:|---:|
| `TARGET_DISCOVERED` | 347 | 34.7% | 85 |
| `UAV_DAMAGE` | 299 | 29.9% | 87 |
| `REGION_VACANCY` | 254 | 25.4% | 55 |
| `TARGET_DESTROYED` | 100 | 10.0% | 13 |

旧 MLP-PPO 不是口头基线：它实际加载了 `results/models/run_20260605_210049/maskable_ppo_uav_task_allocation.zip`，checkpoint SHA-256 为 `5a9be7153d33532ce99c61f13c8151549cc6cc919ec75fad150d05bc78dec5da`，并完整跑过同一批 200 条事件带。

主要证据：

- `ppo_allocation/results/random_event/training_summary.json`
- `ppo_allocation/results/random_event/preliminary_eval/evaluation_summary.json`
- `ppo_allocation/results/random_event/preliminary_eval/grouped_analysis.json`
- `ppo_allocation/results/random_event/tapes/preliminary_test200/manifest.json`
- `ppo_allocation/results/random_event/tapes/smoke/manifest.json`
- `ppo_allocation/results/random_event/preliminary_eval/raw_trace_index.json`

## 2. 两种不确定性不能混为一谈

本报告同时给出两类区间，但它们回答的是不同问题：

1. **训练种子层面的 t 95% CI（n=3）**：先在 200 条配对事件带上求每个 checkpoint 相对基线的平均差，再把三个训练 checkpoint 当作三个独立样本；自由度为 2，使用 `t*=4.3026527`。它回答“重新训练一次，平均效果会不会变化”。由于只有三个种子，区间会很宽。
2. **三种子 ensemble 的逐事件带 bootstrap 95% CI**：先在每条 tape 上平均三个 GPPO checkpoint，再与同一 tape 的基线配对，最后对 200 条 tape 重采样 5,000 次。它回答“固定这三个 checkpoint 时，换一批同分布 tape 会不会变化”。它**不包含新训练种子的风险**，因此不能用较窄的 bootstrap CI 代替 n=3 的训练种子 CI。

所有差值均使用“oriented improvement”：正数表示 GPPO 更好。对 return、成功率、通信抑制率，正数就是 GPPO−基线；对恢复延迟、未覆盖、距离、负载差、切换、时延等越小越好的指标，符号已经反向。区间是描述性 95% 区间，未做多重比较校正。

## 3. 每个训练种子相对所有基线的配对 return 差

下表每个 seed 值都是同一 checkpoint 在相同 200 条 tape 上相对基线的配对平均 return 差。最后两列分别是训练种子 t 区间和固定三 checkpoint 的逐 tape bootstrap 区间。

| GPPO | 比较对象 | seed 1 | seed 2 | seed 3 | 三种子均值 [t 95% CI] | ensemble [tape bootstrap 95% CI] |
|---|---|---:|---:|---:|---:|---:|
| NoGate | MLP-PPO | +5.140 | +1.851 | -2.472 | +1.506 [-7.977, 10.989] | +1.506 [-2.041, 5.053] |
| NoGate | Masked Random | +6.019 | +2.730 | -1.593 | +2.385 [-7.098, 11.869] | +2.385 [-1.824, 6.762] |
| NoGate | Nearest Legal | -0.183 | -3.472 | -7.795 | -3.817 [-13.300, 5.667] | -3.817 [-7.263, -0.362] |
| NoGate | Min Load | +0.367 | -2.922 | -7.245 | -3.267 [-12.750, 6.217] | -3.267 [-5.795, -0.705] |
| NoGate | Greedy Cost | -0.853 | -4.141 | -8.464 | -4.486 [-13.970, 4.997] | -4.486 [-7.313, -1.770] |
| NoGate | Exhaustive Oracle | +1.072 | -2.217 | -6.540 | -2.561 [-12.045, 6.922] | -2.561 [-5.018, -0.125] |
| Adaptive | MLP-PPO | +5.193 | -2.783 | -1.171 | +0.413 [-10.063, 10.889] | +0.413 [-3.172, 3.976] |
| Adaptive | Masked Random | +6.072 | -1.904 | -0.292 | +1.292 [-9.184, 11.768] | +1.292 [-2.920, 5.454] |
| Adaptive | Nearest Legal | -0.130 | -8.106 | -6.494 | -4.910 [-15.386, 5.566] | -4.910 [-7.794, -1.975] |
| Adaptive | Min Load | +0.420 | -7.556 | -5.943 | -4.360 [-14.836, 6.116] | -4.360 [-6.684, -2.043] |
| Adaptive | Greedy Cost | -0.800 | -8.775 | -7.163 | -5.579 [-16.055, 4.897] | -5.579 [-8.245, -2.990] |
| Adaptive | Exhaustive Oracle | +1.125 | -6.850 | -5.238 | -3.654 [-14.130, 6.821] | -3.654 [-5.908, -1.375] |

seed 1 的两个 GPPO 都比较强，seed 2/3 明显较弱。这是“512 步短训练尚不稳定”的直接证据。固定 ensemble 的 tape-bootstrap 区间显示，它在这批分布上通常输给四个较强规则/Oracle；但训练种子 t 区间都跨 0，说明三次训练还不足以稳定确定差异大小。

## 4. 主要指标：实际均值

GPPO 行是先对每个训练种子的 200 条 tape 求均值，再对三个种子取平均；其他算法是 200 条 tape 的均值。

| 算法 | 成功率↑ | 恢复延迟↓ | 累计未覆盖↓ | 距离↓ | 负载差↓ | 切换↓ | return↑ | 推理 ms↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPPO-NoGate（3-seed mean） | 0.9400 | 1.6806 | 7.4783 | 0.1420 | 0.6379 | 6.308 | 122.546 | 9.970 |
| GPPO-Adaptive（3-seed mean） | 0.9423 | 1.6812 | 7.5667 | 0.1467 | 0.7330 | 6.307 | 121.453 | 11.526 |
| MLP-PPO | 0.9420 | 1.6655 | 7.5450 | 0.1562 | 0.8569 | 6.255 | 121.040 | 1.111 |
| Masked Random | 0.9430 | 1.6600 | 7.6600 | 0.1734 | 1.0331 | 6.275 | 120.161 | 0.048 |
| Nearest Legal | 0.9400 | 1.7161 | 7.7200 | 0.1363 | 0.6556 | 6.470 | 126.363 | 0.057 |
| Min Load | 0.9390 | 1.6978 | 7.4650 | 0.1346 | 0.4410 | 6.375 | 125.813 | 0.081 |
| Greedy Cost | 0.9390 | 1.7055 | 7.6050 | 0.1389 | 0.4438 | 6.405 | **127.032** | 1.696 |
| Exhaustive Oracle | 0.9400 | 1.6961 | 7.4900 | 0.1398 | 0.4612 | 6.355 | 125.107 | 1.561 |

这里的 Oracle 是当前逐决策枚举目标下的 “Exhaustive Oracle”，不是整个多事件时域的全局最优解。因此它的 episode return 低于 Greedy Cost 并不构成数学矛盾，也不能把 `125.107` 当作理论上界。

### 相对旧 MLP-PPO 的关键差异

| 变体 | 指标 | 三训练种子 oriented improvement [t 95% CI] | ensemble [tape bootstrap 95% CI] | 判断 |
|---|---|---:|---:|---|
| NoGate | 成功率 | -0.0020 [-0.0045, 0.0005] | -0.0020 [-0.0057, 0.0013] | 无稳定差异 |
| Adaptive | 成功率 | +0.0003 [-0.0025, 0.0032] | +0.0003 [-0.0030, 0.0037] | 无稳定差异 |
| NoGate | 恢复延迟 | -0.0150 [-0.0579, 0.0279] | -0.0150 [-0.0511, 0.0206] | GPPO 均值略慢，区间跨 0 |
| Adaptive | 恢复延迟 | -0.0156 [-0.0779, 0.0466] | -0.0156 [-0.0528, 0.0224] | GPPO 均值略慢，区间跨 0 |
| NoGate | 累计未覆盖 | +0.0667 [-0.1743, 0.3076] | +0.0667 [-0.2750, 0.4317] | 均值略好，不确定 |
| Adaptive | 累计未覆盖 | -0.0217 [-0.1780, 0.1346] | -0.0217 [-0.3734, 0.3517] | 均值略差，不确定 |
| NoGate | 归一化距离 | +0.0142 [-0.0041, 0.0325] | +0.0142 [0.0101, 0.0186] | 固定 ensemble 在 tape 上更好；seed 风险未排除 |
| Adaptive | 归一化距离 | +0.0095 [-0.0031, 0.0222] | +0.0095 [0.0052, 0.0138] | 同上 |
| NoGate | 负载差 | +0.2190 [-0.4387, 0.8767] | +0.2190 [0.1435, 0.2987] | 固定 ensemble 更好；seed 间波动很大 |
| Adaptive | 负载差 | +0.1239 [-0.5297, 0.7775] | +0.1239 [0.0455, 0.2039] | 同上 |
| NoGate | 切换次数 | -0.0533 [-0.2377, 0.1310] | -0.0533 [-0.2200, 0.1133] | 无稳定差异 |
| Adaptive | 切换次数 | -0.0517 [-0.3105, 0.2072] | -0.0517 [-0.2217, 0.1234] | 无稳定差异 |
| NoGate | return | +1.506 [-7.977, 10.989] | +1.506 [-2.041, 5.053] | 不能声称胜出 |
| Adaptive | return | +0.413 [-10.063, 10.889] | +0.413 [-3.172, 3.976] | 不能声称胜出 |
| NoGate | 推理时延 | -8.859 [-9.987, -7.731] ms | -8.859 [-8.949, -8.773] ms | 明确更慢 |
| Adaptive | 推理时延 | -10.415 [-11.563, -9.266] ms | -10.415 [-10.636, -10.201] ms | 明确更慢 |

## 5. 四种事件时序模式

每个单元合并了该变体的 3 个训练种子和对应模式的 50 条 tape，即 150 个 seed-tape 记录。不同模式的决策机会和物理时间结构不同，所以不能仅凭 raw return 横向断言某个模式“更容易”。

| 变体/模式 | 成功率↑ | 恢复延迟↓ | 累计未覆盖↓ | 距离↓ | 负载差↓ | 切换↓ | return↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| NoGate / single | 0.9480 | 1.6987 | 10.7667 | 0.1459 | 0.6803 | 5.993 | 119.041 |
| NoGate / sequential | 0.9480 | 1.5212 | 6.9533 | 0.1537 | 0.7069 | 6.100 | 121.872 |
| NoGate / overlap | **0.9173** | 1.7564 | 6.4400 | 0.1247 | 0.5756 | 6.653 | 124.651 |
| NoGate / burst | 0.9467 | 1.7459 | 5.7533 | 0.1437 | 0.5888 | 6.487 | 124.620 |
| Adaptive / single | 0.9480 | 1.6891 | 10.8067 | 0.1499 | 0.7891 | 5.960 | 117.176 |
| Adaptive / sequential | 0.9480 | 1.5222 | 7.0400 | 0.1585 | 0.8230 | 6.100 | 120.317 |
| Adaptive / overlap | **0.9227** | 1.7577 | 6.5867 | 0.1303 | 0.6563 | 6.647 | 123.955 |
| Adaptive / burst | 0.9507 | 1.7556 | 5.8333 | 0.1480 | 0.6638 | 6.520 | 124.364 |

最值得继续测试的是 overlap：两种 GPPO 的成功率都在该模式最低（NoGate `0.9173`、Adaptive `0.9227`），说明“前一扰动尚未完全恢复时又来新事件”仍是压力点。single 的累计未覆盖较高与该模式的时间间隔/决策时域有关，不能直接解读为单事件反而更难。

## 6. 临时不可行、最终不可行与成功率

| 算法 | temporary infeasible rate | final infeasible rate | weighted uncovered | legal coverage rate |
|---|---:|---:|---:|---:|
| GPPO-NoGate（3-seed mean） | 0.0493 | 0.0600 | 0.82 | 0.795 |
| GPPO-Adaptive（3-seed mean） | 0.0490 | 0.0577 | 0.82 | 0.795 |
| MLP-PPO | 0.0500 | 0.0580 | 0.82 | 0.795 |
| Masked Random | 0.0490 | 0.0570 | 0.82 | 0.795 |
| 规则/Oracle 范围 | 0.0490 | 0.0600–0.0610 | 0.82 | 0.795 |

四组几乎相同不是“所有算法都学得一样好”，而是说明这部分结果由共同的事件带、可行性约束和动作 mask 主导。最终不可行事件不能被任何合法分配动作修好；临时不可行则允许在后续状态/事件变化后恢复。因此，`event_success_rate ≈ 1 - final_infeasible_rate`，不能把约 94% 的成功率直接归因于策略网络。

## 7. Mask、Gate、Critic 和通信机制诊断

### 动作掩码依赖

GPPO-NoGate 的 pre-mask invalid probability 三种子均值为 `0.84279`，Adaptive 为 `0.84350`；环境实际 mask rate 也约为 `0.842`。这表示未加 mask 的网络会把约 84% 的概率质量放在非法边上，接近“没有主动学会压低非法动作”的状态。严格 mask 保证最终执行动作合法，所以成功率仍可很高。当前证据更支持“mask 在兜底”，而不是“Actor 已理解约束”。

### Adaptive Gate

Adaptive 的 gate mean 为 `0.49690 ± 0.00084`（三个 seed mean 的样本标准差），gate variance 仅 `5.44e-5`；NoGate 按定义为 1。Gate 目前几乎是一个固定的 0.5 缩放，尚未表现出随事件、节点类型或图状态显著开合的自适应性。短训练下 Adaptive 没有优于 NoGate，不能据此判断论文中的 Gate 思路无效，只能说本轮没有学出来。

### Critic 与 PPO 更新

最终 update 的三种子平均诊断如下：

| 变体 | explained variance | value loss | approx KL | clip fraction | entropy |
|---|---:|---:|---:|---:|---:|
| NoGate | 0.000325 | 2480.20 | 5.73e-8 | 0.0 | 0.6730 |
| Adaptive | 0.000273 | 2505.73 | 1.90e-7 | 0.0 | 0.6758 |

评估集上的 value absolute error 三种子均值为 NoGate `17.452`、Adaptive `17.831`，value squared error 分别为 `325.875`、`339.622`。explained variance 约为 0、clip fraction 为 0、KL 极小，说明 4 次更新远未形成可靠 Critic，也没有发生实质性的 PPO clipping。把这个 checkpoint 称为“已收敛 GPPO”是不准确的。

### 通信与端到端时延

所有算法共享事件触发通信逻辑，平均 communication bytes 约为 2.13 KB/episode，通信触发约 4.2 次/episode，抑制率约 0.29–0.31；差异主要来自各策略产生的决策步数，而不是 GPPO 学到了独立的通信协议。`event_to_action_latency_ms` 包含事件观测时间模型，均值约 156–169 ms，和纯模型 `inference_latency_ms` 不是同一指标；判断部署算力开销应看后者。

## 8. 哪些结论可以写，哪些不能写

可以写：

- 四类随机事件、四种时序模式、连续多事件回放、临时/最终不可行状态和旧 MLP checkpoint 已经进入同一评估链路。
- 在固定三个短训 checkpoint 的 200-tape ensemble 上，GPPO 相对旧 MLP 和 Random 在距离、负载差上更好；NoGate 的平均 return 略高于旧 MLP。
- 当前较强规则算法在 return 上整体胜过两个 GPPO；GPPO 推理慢约一个数量级。
- overlap 是当前最明显的成功率压力场景。

不能写：

- “GPPO 显著优于 PPO”——return 的训练种子 t CI 跨 0。
- “Adaptive Gate 带来提升”——本轮 NoGate 平均更好，且 Gate 几乎恒定在 0.5。
- “94% 成功率证明网络学会了重分配”——成功率高度受可行性和 mask 支配。
- “Oracle 是全局上界”——当前 Oracle 是局部枚举基线。
- “512 步已经收敛”——Critic、KL、clip fraction 和 seed 波动都与此相反。

## 9. 下一轮正式实验建议

1. 先把训练从 512 steps 扩展，并用独立 validation bank 选 checkpoint；至少记录学习曲线直到 explained variance、KL、clip fraction、pre-mask invalid probability 和 gate 分布出现可解释变化。
2. 正式结论使用不少于 5 个训练 seed；训练种子是主要不确定性，单纯把 test tape 从 200 增到 1,000 不能替代更多训练 seed。
3. 把 overlap/burst 作为压力测试单列，报告成功率、恢复延迟的尾部（P90/P95），而不仅是均值。
4. 对 Gate 增加按事件类型、节点类型、层和时间的分组图；若仍固定在 0.5，再检查门控梯度、初始化与正则项。
5. 继续保留 MLP-PPO、Masked Random、Nearest Legal、Min Load、Greedy Cost、局部 Exhaustive Oracle；不要只挑较弱基线。
6. 在冻结正式 test bank 前锁定奖励权重、事件概率、训练步数和选模规则，避免根据 test 结果调参。

本报告的所有更细粒度数值（每个变体×比较对象×指标×训练种子的配对差、t 区间、ensemble bootstrap 区间、四模式均值和机制诊断）均保存在 `grouped_analysis.json`，可由原始 `evaluation_summary.json` 逐项追溯。
