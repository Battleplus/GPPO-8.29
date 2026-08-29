# Dynamic-Preemption-Dev

该目录保存 `execution-preemption-v1` 的确定性开发事件带。它用于验证任务连续进度、规则化抢占、迁移、返航和并发一致性，不是正式 held-out test，也不用于 checkpoint selection。

## 当前开发集

- 10 类场景；
- 每类 20 条固定 paired tapes；
- 总计 200 cases、200 个互不重复的 development seeds；
- 规则控制器回放 200/200 PASS；
- 共验证 280 个事件决策；
- 资源冲突、旧命令复活和进度重复累计均为 0；
- 没有启动训练或模型效果评估。

入口文件：

- [场景目录](dev_v1/scenario_catalog.json)
- [机器可读 manifest 与 SHA-256 inventory](dev_v1/manifest.json)
- [规则回放结果说明](dev_v1/RULE_REPLAY_REPORT_ZH.md)
- [分配器接口回放摘要](dev_v1/allocator_replay_summary.json)
- [V1 协议](../../docs/EXECUTION_PREEMPTION_V1_PROTOCOL_ZH.md)

## 复现

生成器拒绝复用非空输出目录。复验时应指定新的空目录：

```powershell
python scripts\build_dynamic_preemption_dev.py `
  --output-dir E:\path\to\new_empty_dev_bank
```

当前 tapes 是算法共享的开发输入。未来接入 PPO、GPPO 或 Planner 时，必须保持事件内容、顺序和 seed 不变。算法及阈值冻结后，需另行生成与本目录完全隔离的 `Dynamic-Preemption-Hidden-V1`。
