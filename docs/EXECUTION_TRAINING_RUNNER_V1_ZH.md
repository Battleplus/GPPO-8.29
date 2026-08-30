# Execution-Preemption V1：训练运行器与证据封存

> 状态：`IMPLEMENTED_TINY_SMOKE_PASS`
> 正式训练：未启动

## 1. 正式训练基数

四个学习方法严格使用相同训练合同：

```text
ppo_mlp_reactive_v1
gppo_adaptive_reactive_v1
ppo_mlp_rule_arbiter_v1
gppo_adaptive_rule_arbiter_v1

× seeds 1101 / 2202 / 3303
× 4 / 8 / 16 UAV
= 36 runs
```

每个 run 精确 50,000 accepted allocation decisions，在 25,000 和 50,000 各保存一次 checkpoint，总计 72 个。32 UAV 不训练，只允许固定使用 16-UAV 的 50k checkpoint 做后续零样本扩展验证。

每个 worker 的路径唯一固定为：

```text
<output-root>/<method>/uav_<04|08|16>/seed_<1101|2202|3303>/
```

不存在 resume 或覆盖语义；目标目录非空时立即拒绝。

## 2. Training tapes

训练输入使用独立 namespace `execution_preemption_v1/train`，由 method 外部的 `(policy_seed, uav_count, episode_index)` 确定生成。它不会读取或复用 `Dynamic-Preemption-Dev` 的 200 条文件，也不会读取尚未生成的 Hidden-V1。

训练 tapes 保持十类业务语义，但使用独立 seed 身份、时间偏移、能量扰动，并按 4/8/16 UAV 与每架 2/3 tasks 扩展。相同 tuple 字节级确定；不同 tuple seed 唯一。

## 3. PPO 合同

```text
Optimizer: Adam
Learning rate: 3e-4
Gamma: 0.99
GAE lambda: 0.95
Clip range: 0.2
Value coefficient: 0.5
Entropy coefficient: 0.01
Max gradient norm: 0.5
Rollout: 64 accepted decisions
Update epochs: 4
Hidden dimension: 64
GPPO relation layers: 2
```

Reactive 与 Rule-Arbiter 变体共享同一 action table、mask、事务和 reward。Reactive 的 16 维规则上下文全零；Rule-Arbiter 只额外接收冻结的 `EventDecision` 上下文。P0/RTB、安全候选、命令撤销和 fencing 始终不交给模型。

## 4. Gate-before-write

正式 worker 入口是 `scripts/run_execution_preemption_training.py`。它不提供 budget、checkpoint、优化器或超参数覆盖参数。正式模式在创建输出目录、模型或 optimizer 之前，先调用 `_check_execution_launch_gate(require_fully_clean=true)`；Gate RED 时不留下训练产物。

## 5. 每个 run 的封存内容

```text
progress.json
updates.jsonl
run_manifest.json
sha256_inventory.json
checkpoints/step_025000.pt
checkpoints/step_050000.pt
```

checkpoint 包含模型、optimizer、Python/NumPy/Torch RNG、accepted step、episode index、source/evidence provenance 和合同 SHA。每个 checkpoint 同时记录：

- 文件 SHA-256：证明该具体文件未变化；
- 规范化 model-state SHA-256：证明权重状态；
- RNG-state SHA-256：证明随机状态。

只读 verifier 会复算所有文件 inventory、检查日志 steps 严格单调、核对最终 step 与 checkpoint grid，并拒绝任何 provenance 漂移或多余/缺失文件。

## 6. 当前 tiny smoke

四个学习方法分别进行两次相同 seed 的 2-step 微型训练，共 8 runs。结果证明：

- optimizer 确实执行更新；
- 1/2 step checkpoint 均生成并通过文件 SHA 复验；
- 相同 seed 的 model-state SHA 与 RNG-state SHA 一致；
- fresh output、旧 checkpoint 禁用、旧 campaign 禁用均生效；
- Validation、Freeze、Test、Hidden 和 checkpoint selection 均未发生。

机器证据：`experiments/dynamic_preemption/dev_v1/training_runner_smoke.json`。

这只是训练器功能证据，不是 50k 模型、算法效果或泛化证据。
