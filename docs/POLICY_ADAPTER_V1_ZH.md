# Execution-Preemption V1：PPO/GPPO/Planner 统一适配器

> Adapter ID：`execution-preemption-policy-adapter-v1`
> 状态：`FROZEN_FOR_FRAMEWORK_INTEGRATION`
> 当前训练许可：`false`

## 1. 结论

V1 已把同一个版本化执行图编码成两种策略输入：PPO-MLP 使用固定长度平坦向量，GPPO/Planner 使用保留节点类型和关系类型的动态图。两类输入不各自生成动作，而是共享同一张 action table、同一个安全 mask、同一 graph version 和同一 graph SHA-256。

这一步解决的是“如何公平、确定地把同一运行状态交给不同算法”，没有加载模型权重，也没有开始训练或比较模型效果。

机器可读 schema 见 [`configs/execution_policy_adapter_v1.json`](../configs/execution_policy_adapter_v1.json)。

## 2. 固定容量

为了让 PPO-MLP 的输入和输出维度在 4/8/16/32 UAV 间保持一致，V1 固定最大容量为：

| 类型 | 容量 |
|---|---:|
| UAV | 32 |
| Task | 96 |
| Region | 96 |
| Target | 32 |
| Event | 8 |

Task 容量对应最多每架 UAV 3 个任务。容量不足不是截断条件；任何节点数量超过合同容量都会硬失败，必须产生新合同版本后才能继续。

## 3. PPO-MLP 平坦输入

PPO 输入按以下固定顺序拼接：

```text
五类节点特征（节点 ID 字典序，零填充）
→ 五类节点 presence mask
→ 七类关系的稠密二值 incidence
→ 16 维规则上下文
```

冻结维度为：

```text
flat_observation_dimension = 37,976
layout_sha256 = f903860f4ede2ffd8a0ac79fdaf90486c06232169211d9d81e993b41ef9ec544
```

所有值必须为有限数且位于 `[0,1]`。节点不存在时对应特征填 0，同时 presence mask 为 0，避免把“真实全零节点”与 padding 混淆。

## 4. GPPO/Planner 异构输入

异构输入保留：

- 五类节点的 ID 和特征矩阵；
- 七类关系各自的 `(source_index, destination_index)`；
- 空关系类型仍显式保留为空数组；
- 与 PPO 完全相同的规则上下文和动作空间快照。

当前适配器只输出 Python 不可变数据，不导入 PyTorch、PyG 或模型代码。框架 tensor 转换属于下一层，不能改变节点顺序、关系顺序、action table 或 mask。

## 5. 统一动作空间

动作表固定为 3,073 位：

```text
index 0 = NOOP
index 1..3072 = 1 + uav_slot × 96 + task_slot
```

动作表覆盖最大 `32×96` 个 UAV—Task slot。当前图不存在的 slot、业务上不可执行的配对以及不属于当前 `AllocationRequest` 的配对都被 mask。

一次策略动作要提交为 proposal，必须连续通过：

```text
action index 在固定容量内
→ mask=true
→ graph_version 与 live runtime 完全一致
→ graph_sha256 与当前快照完全一致
→ task_id 与 AllocationRequest 一致
→ uav_id 属于请求冻结的安全候选集
→ allocator_id 非空
```

`NOOP` 是合法策略动作，但不能伪装成一个必需的 allocation proposal。控制器可把 NOOP 解释为继续等待或排队；proposal 接口会显式拒绝它。

## 6. Reactive 与 Rule-Arbiter 语义

适配器预留 16 维上下文：

```text
context_present
decision_type one-hot (7)
event_priority one-hot (5)
normalized information_age
confidence
has_displaced_task
```

- Reactive 变体收到全零上下文；
- Rule-Arbiter 变体收到与当前 graph version 绑定的 `EventDecision`；
- 两者都只能从安全 mask 中选择；
- 两者都不能决定 P0、RTB、任务是否允许抢占、command 撤销或 fencing。

这样可以比较“规则上下文是否帮助分配”，同时不通过关闭安全层制造不公平或危险的对照组。

## 7. 版本与竞态保护

只检查 graph version 仍可能漏掉“同一版本对象被意外改写”的实现错误，因此动作同时绑定：

```text
graph_version
graph_sha256
```

推理完成到提交动作之间如果再次发生事件，或者快照内容变化，旧动作会在解码阶段被拒绝。即便绕过解码，后续 `validate_proposal()` 仍会再次检查 request、task、UAV 候选集和 live graph version。

## 8. 当前 smoke 证据

4/8/16/32 UAV 均已完成确定性 adapter smoke：

| UAV | Tasks | 固定输入维度 | 动作容量 | 当前请求候选数 | Flat/GPPO 共用动作表 |
|---:|---:|---:|---:|---:|---|
| 4 | 8 | 37,976 | 3,073 | 4 | PASS |
| 8 | 16 | 37,976 | 3,073 | 8 | PASS |
| 16 | 32 | 37,976 | 3,073 | 16 | PASS |
| 32 | 64 | 37,976 | 3,073 | 32 | PASS |

专项测试覆盖：

- 四规模固定维度和容量；
- PPO/GPPO action space 完全相同；
- request 对全图候选集的二次收窄；
- Reactive/Rule context 区分；
- action 解码与 proposal validator 闭环；
- 过期 version、错误 graph SHA、masked action 和错误 NOOP fail-closed；
- 节点容量溢出硬失败；
- 重复构建字节级确定性。

机器可读结果见 [`policy_adapter_smoke.json`](../experiments/dynamic_preemption/dev_v1/policy_adapter_smoke.json)，其分类为 `adapter_pretraining_smoke_not_model_evidence`。

## 9. 尚未完成

- 适配到 Gymnasium reset/step 生命周期；
- 把平坦输入转换为 PPO tensor；
- 把异构输入转换为 GPPO/PyG tensor；
- 从实际 environment transition 生成冻结 reward signals；
- 训练日志、checkpoint 和 RNG 状态封存；
- source-bound 训练 Gate；
- 任何模型训练、Validation、Freeze、Test 或 Hidden 评估。

下一步必须先实现一个只跑极短 deterministic rollout 的 framework smoke，并证明 reward、mask、action 解码、事务提交和日志字段一致；在此之前 `training_allowed` 继续保持 `false`。

