# Execution-Preemption V1：Source-bound 训练前 Gate

> Gate：`EXECUTION_PREEMPTION_V1_LAUNCH_GATE`  
> Source 状态：`FROZEN_FOR_SOURCE_ATTESTATION`  
> 当前正式训练：未启动

## 1. 目的

执行中抢占引入了新的状态、图、动作、reward 和事务语义，因此不能复用旧 minimum-validation 的 `handoff/P0_GATE.json`、smoke 或 checkpoint。本 Gate 是新实验唯一的训练前授权记录：只有机器检查全部 PASS 时，生成文件中的 `training_allowed` 才可能为 `true`。

生成器：`scripts/build_execution_launch_gate.py`  
正式入口检查：`execution_preemption.gate._check_execution_launch_gate()`  
唯一 evidence 文件：

```text
experiments/dynamic_preemption/evidence_v1/EXECUTION_PREEMPTION_V1_GATE.json
```

## 2. Source 与 Evidence 链

冻结流程必须是：

```text
research/execution-preemption-v1 最终远端 source SHA
        ↓ 新 ASCII clean source worktree
运行 130 项旧 required tests + 至少 100 项新专项测试
        ↓
逐条复算 200 tapes、合同、smoke 与 protected source SHA-256
        ↓
生成唯一 Gate JSON，并在 source HEAD 调 formal checker
        ↓ 只提交 Gate JSON
evidence/execution-preemption-v1 evidence HEAD
        ↓ 新 clean evidence worktree
重新运行测试并调 formal checker(require_fully_clean=true)
```

本地开发 commit 与 GitHub commit 即使 tree 相同，也不能互相替代 source SHA。Gate 只绑定推送后 `refs/heads/research/execution-preemption-v1` 的精确远端 SHA。

## 3. 强制检查

Gate 同时验证：

- source worktree 在生成前完全 clean；
- source SHA 等于 GitHub 研究分支 HEAD；
- source tree SHA 与新实验 protected-file SHA-256 inventory；
- 原 minimum-validation required tests 至少 130 项；
- Execution-Preemption 专项测试至少 100 项；
- 10 类 × 20 条开发 tapes 的文件 SHA、canonical SHA、seed 与场景基数；
- 400 allocator-tape runs 和零 invariant failure；
- 4/8/16/32 图与 adapter smoke；
- 400/400 direct/deferred decision 与 state SHA parity；
- PPO/GPPO Gym/PyTorch rollout、共享 mask、零安全违规、零 optimizer step；
- training/Validation/Freeze/Test/Hidden 均未启动；
- Hidden bank 和正式训练输出 namespace 均不存在。

PyG 是当前仓库原生 PyTorch GPPO 的非必需可选依赖。本机 PyG 导入失败会原样记录，但不能伪报 PASS，也不会单独阻止当前 backend。

## 4. Evidence 白名单

source 到 runtime HEAD 的提交差异必须为空，或严格只包含上述 Gate JSON。以下内容一律拒绝：

- 任意 `*.py`、配置、测试或文档修改；
- 任意 checkpoint、模型权重、训练日志或输出；
- 任意第二个 evidence JSON；
- 旧 `handoff/P0_GATE.json` 或旧 smoke 更新；
- source 不是 runtime HEAD 祖先；
- protected worktree 有未提交改动。

Source HEAD 允许仅有刚生成、尚未提交的 Gate JSON，以便先实际调用 formal checker。正式 training worktree 必须从 clean evidence HEAD 创建，并用 `require_fully_clean=true` 再检查一次。

## 5. 当前结论边界

Gate PASS 只说明“源码、合同、接口、测试和训练前 smoke 足以允许下一阶段创建新的训练运行器/训练产物”，不说明模型有效，也不代表任何 50k run 已完成。Gate evidence 生成前，静态合同中的 `training_allowed` 继续保持 `false`；不得通过手改配置绕过生成器。
