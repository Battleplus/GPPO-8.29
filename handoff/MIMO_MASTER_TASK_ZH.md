# MiMo-V2.5-Pro 总任务书：P0 修复到 Colab Pro Preliminary

## 1. 角色与目标

你是本项目的执行型代码代理。直接在现有代码上完成协议修复、独立事件处理层、公平 PPO-MLP、冻结实验协议、Colab Pro 三训练 seed preliminary、统计分析和中文报告。

不要只给方案；但任何长训练必须等待 P0 gate。不得伪造运行结果。

唯一工作目录：

`E:\Z博士\8.20\54_20-master`

开始前按 `MIMO_START_HERE.md` 和 `handoff/LOCAL_CONTEXT_INDEX.md` 的固定顺序读取本地资料。不要去其他路径搜索。

## 2. 不可违反的边界

1. 保留当前 dirty worktree，先记录 `git status --short`、`git diff --stat`、`git log -1 --oneline`。
2. 禁止 `git reset --hard`、`git checkout --`、批量覆盖和删除已有模型、结果、场景。
3. 不自动 commit、push 或创建远端分支。
4. 原始 `ppo_allocation` 核心环境和旧模型保留；新事件层通过独立 adapter 接入。
5. 旧 checkpoint 只称 `Legacy MLP-PPO`。
6. P0 未通过前禁止长训练、禁止 Test 调参、禁止声称 GPPO 优于 PPO。
7. 旧 512-step 结果只作历史工程证据，不与修复后结果直接合并。
8. 原始 ZIP 只核验来源，禁止解压覆盖。

## 3. 任务背景

保留 4 UAV、4 Region、3 Target，以及 `SEARCH/TRACK/IDLE` 任务语义。支持：

- `UAV_DAMAGE`
- `TARGET_DISCOVERED`
- `TARGET_DESTROYED`
- `REGION_VACANCY`

只重分配真正受影响的搜索任务，不改成父任务—子任务调度。

最终比较：

- Legacy MLP-PPO（仅迁移参考）
- Masked Random
- Nearest Legal
- Min Load
- Greedy Cost
- Current-Pending Exact Planner
- 公平 PPO-MLP
- GPPO-NoGate
- GPPO-Adaptive

## 4. 阶段 0：证据化审计

先审计，不训练。核对：

- `event_runtime/` 已实现和缺失内容；
- single、sequential、overlap、burst、unseen 的实际语义；
- reward 是否按事件重复累计；
- `event_return` 是否被错误求和为 `episode_return`；
- Planner 是否仅优化当前 pending；
- seed manifest 是否实际被训练器使用；
- Validation 是否错误包含 unseen；
- Test 是否曾参与选模；
- legacy CPP 与模型加载是否仍能运行。

输出：

- `docs/WORKSPACE_AUDIT_ZH.md`
- `handoff/CURRENT_STATE.json`
- `handoff/PROGRESS.md`
- `handoff/DECISIONS.json`

## 5. 阶段 1：独立事件处理层

在现有 `event_runtime/` 上续写，目标数据流：

```text
TruthEvent
  -> Observation
  -> ConfirmationStateMachine
  -> ConfirmedEvent
  -> EventQueue
  -> Belief State / Adapter
  -> DeterministicNearestLegalPolicy
  -> Version validation / Command / ACK / Lease / FencingToken
```

至少完成：

- `events.py`
- `observation.py`
- `detector.py`
- `state_machine.py`
- `scheduler.py`
- `queue.py`
- `concurrency.py`
- `lease.py`
- `adapter.py`
- `metrics.py`
- `replay.py`

对象至少包括 `TruthEvent`、`Observation`、`ConfirmedEvent`、两种 tape、Detector、状态机、队列、Command、ACK、Lease、FencingToken、Adapter 和 DeterministicNearestLegalPolicy。

必须分开：

- `occurred_at`
- `emitted_at`
- `received_at`
- `suspected_at`
- `confirmed_at`
- `resolved_at`

TruthEvent 只更新 True State；Observation 只进入确认器；只有 ConfirmedEvent 能更新 Belief、释放任务、改变 mask 和触发决策。策略不得读取未确认真值。

确认规则：

- 可信硬故障报告可直接确认；
- 单次心跳缺失只进入 `SUSPECTED`；
- 连续 3 次缺失后 probe，probe 超时或第二来源证据才确认；
- 确认前健康遥测转 `FALSE_ALARM`；
- `SUSPECTED` 不释放有效 lease；
- Target 发现采用 3-of-5 或权威多源确认；
- Target 摧毁采用权威确认或至少两份独立强证据；
- 短时丢踪不得编码成摧毁；
- TRACK ACK 后才撤销发现者 SEARCH lease。

Observation 层用固定 seed 确定性模拟 delay、loss、duplicate、out-of-order、false positive、false negative、partition 和 recovery。Truth tape 与 Observation tape 独立保存、独立重放、规范 JSON 字节稳定。

## 6. 阶段 2：并发一致性

依次实现：

```text
graph_version
-> action_version
-> latest mask validation
-> AssignmentCommand
-> ACK
-> AssignmentLease
-> FencingToken
-> revoke / timeout
```

不变量：

- 状态变化后旧动作 100% 拒绝；
- 排他任务有效 holder 始终不超过 1；
- 同一区域不能同时有效分配给两个 UAV；
- 迟到 ACK 不能恢复撤销任务；
- 分区恢复后低 token 命令失效；
- 无关普通事件不中断既有任务；
- 命令无 ACK 时新事件仍可入队；
- lease 超时后新 holder 可用更高 token 接管。

假决策器只能选择最近合法 UAV–Region edge，使用未来 GPPO 相同 mask，携带 graph/action version，提交前重验；它不用于证明性能。

## 7. 阶段 3：五种事件模式

### single

- 每条 case 一个事件；
- 每条 case 从完全相同初始快照 reset；
- snapshot 规范 JSON 与 SHA-256 相同。

### sequential

- 上一事件恢复后才出现下一事件；
- 保留完整状态演化，不隐式 reset。

### overlap

- 未恢复时允许新事件到达；
- 按 `received_at` 处理；
- 未到达的早发事件不能阻塞已收到后发事件；
- 多事件可共享一次决策，reward 只记一次。

### burst

- 同一 100 ms window 内 2–3 个 P1/P2 事件先应用临时状态；
- 合并 affected 与 pending；
- 中间不调用策略；
- 构图一次；
- `graph_version + 1`；
- 策略调用一次；
- 失败不留下半提交状态。

P0 安全事件不等待完整窗口；同一接收批次已到达事件可共同原子提交。

### unseen

- 仅最终 Test；
- 禁止进入 Train 和 Validation；
- 事件语义不变，类型概率、通信延迟、丢包、重复、乱序和分区更强。

默认训练概率 `0.30/0.30/0.20/0.20`，unseen 为 `0.15/0.15/0.30/0.40`。只在当前有效类型间一次重归一化，不循环拒绝采样。

## 8. 阶段 4：Reward 与指标修复

唯一 episode return 定义：

```python
episode_return = sum(row["reward"] for row in decision_rows)
assert episode_return == total_reward_check
```

single、sequential、overlap、burst 全部逐 episode 验证。一次动作服务多个事件时，decision row 只记录一次 reward 并保存 `affected_event_ids`；事件级记录不可再保存并累加完整 `event_return`。

等待期间不创建虚假 PPO step。空缺时间代价在下一真实决策步结算，累计空缺同时作为独立物理指标。

记录探测、确认、决策、恢复、空缺、距离、负载、切换、不一致、通信和机制指标。完整字段以本地设计报告为准。

## 9. 阶段 5：Planner 语义修正

把 `Exhaustive Oracle` 全面改为：

`Current-Pending Exact Planner / 当前待分配区域精确规划器`

它只枚举当前 pending Regions 的合法联合分配，不预知未来事件，不是全局上界。指标：

```text
local_cost_regret = J(algorithm_after) - J(exact_planner_after)
```

同步修改代码、配置、JSON、图、表和文档。

## 10. 阶段 6：公平 PPO-MLP

新增共享 `GraphObservationContract`。PPO-MLP 与两个 GPPO 使用相同 Truth/Observation/Confirmed 流、图信息、pending/queue/lease/communication/version 特征、edge+NOOP 动作、mask、reward、PPO 参数、预算、seed、Validation 和 checkpoint 间隔。

唯一区别：PPO-MLP 读取规范顺序展平图；GPPO-NoGate 用 AHGNN；GPPO-Adaptive 用 AHGNN+Gate。

报告参数量、近似 FLOPs、推理时延、显存和 pre-mask 非法概率。MLP 宽度在 Test 前冻结。

## 11. 阶段 7：冻结协议

训练 seeds：`1101, 2202, 3303`。

Validation：100 tapes，四个 nominal 模式各 25，不含 unseen。

Test：200 tapes，Single/Sequential/Overlap/Burst/Unseen 各 40。最终 checkpoint 只运行一次。

Test 禁止用于选 checkpoint、训练步数、reward、事件概率、结构、MLP 宽度和 early stopping。

Validation 字典序：最低 final infeasible rate；最低累计加权空缺；最低恢复延迟；最低固定 J；仍相同时选较早 checkpoint。

程序必须实际读取 seed manifest，并记录 Git/dirty/source tree、依赖、协议、seed、tape、checkpoint、Validation 记录和原始日志哈希。

## 12. 阶段 8：自动化测试与 P0 Gate

至少覆盖：14 条设计时间线、same/different seed、duplicate 幂等、single snapshot、sequential 顺序、overlap received order、burst 原子性、unseen 隔离、四模式 reward、Test 不选模、CPP/legacy、三种学习模型 save/load。

本地 smoke：五模式各 20 tapes。

必须满足：

```text
stale_action_rejection_rate == 1.0
exclusive_task_valid_holder_count <= 1
duplicate_assignment_count == 0
unaffected_task_interruption_time == 0
burst_three_event_graph_version_delta == 1
reward_invariant == true
```

生成 `handoff/P0_GATE.json`，含测试、命令、输出、源码/协议/seed hashes 与 `training_allowed`。训练入口必须重验；文件变化或 gate=false 时拒绝训练。

## 13. 阶段 9：Colab Pro Preliminary

P0 通过后生成并实际验证 `colab_bundle/`：Notebook、锁定依赖、源码 snapshot+hash、协议、banks、README。

模型为 PPO-MLP、GPPO-NoGate、GPPO-Adaptive；每个使用 seeds 1101/2202/3303。默认预算每模型每 seed 300,000 decision steps，每 25,000 steps checkpoint。若资源不足，只能在所有训练开始前统一下调并冻结。

Notebook 必须验证 GPU、hash、依赖和 P0 gate；同步 checkpoint、optimizer、RNG、tape index 到 Drive；断线恢复时验证配置 hash。

Validation 选择 checkpoint 后冻结 SHA-256，再运行一次 Test。不得自动进入 5-seed formal。

若当前 Agent 无法操作已登录 Colab，只能交付已本地验证的 Notebook 并明确标记未训练；禁止伪造结果或称 Colab 阶段完成。

## 14. 阶段 10：统计与报告

训练 seed 是独立单位。报告每 seed 原值、均值、标准差、seed-level 95% CI、同 seed 同 tape 配对差、效应量和 Holm 校正。tape bootstrap 只能作固定 checkpoint 敏感性分析。

主比较：Adaptive vs PPO-MLP、Adaptive vs NoGate、Adaptive vs Greedy。未恢复样本同时报告条件分布和右删失。

如实判断 GPPO、AHGNN、Gate 是否胜出；禁止修改 Test、删除失败、把 mask 合法性归功于学习。

至少四图：累计空缺、事件类型恢复延迟、local_cost_regret、通信—覆盖 Pareto。图中注明 preliminary、三 seed、Test bank、nominal/unseen 与 CI 含义。

## 15. 交付与停止条件

代码、配置、结果、报告和 handoff 文件全部留在唯一工作区。至少输出：

- `docs/WORKSPACE_AUDIT_ZH.md`
- `docs/EVENT_RUNTIME_DESIGN_ZH.md`
- `docs/EVENT_RUNTIME_TEST_REPORT_ZH.md`
- `docs/P0_PROTOCOL_FIX_REPORT_ZH.md`
- `docs/FAIR_PPO_BASELINE_DESIGN_ZH.md`
- `docs/COLAB_PRELIMINARY_RUN_REPORT_ZH.md`
- `docs/FINAL_PRELIMINARY_REPORT_ZH.md`
- `handoff/P0_GATE.json`
- `handoff/PROGRESS.md`
- `handoff/COMPLETION_AUDIT.md`
- `handoff/REPRODUCTION_COMMANDS_ZH.md`

每阶段更新 `handoff/PROGRESS.md`，记录命令和证据。遇到 blocker 时先完成不受阻部分，再报告证据、尝试和最小方案。

只有所有要求有运行证据才能宣布完成；Notebook 生成不等于 Colab 训练完成。不要自动进入 Formal。

## 16. 第一条回复与立即动作

第一条回复只需给出：

1. 识别到的唯一项目根目录；
2. 当前 Git HEAD 和 dirty 摘要；
3. 已存在、可续写的实现；
4. P0 缺口；
5. 分阶段执行计划；
6. 即将运行的第一组只读/测试命令。

随后直接开始审计和执行，不要停留在重新解释任务书，也不要询问本文件已经给出的默认决定。
