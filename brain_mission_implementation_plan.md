# 大脑任务流梳理与改造计划

## Summary

你的流程方向基本正确，但当前代码只实现了一部分：`brain.start()` 已能初始化/同步场景，MILP 侦察分配、MPPI 航路、`search_planner` 巡航、DRL 局部避障、UAV 损毁 PPO 重分配、MILP 打击链路都已有雏形。

当前主要缺口是：发现目标时立即插队打击并同步侦察重分配、编队独立线程、多 AOI/多片侦察的 Brain 适配。

本计划按以下方向落地：

- Brain 统一负责初始化。
- 目标发现后并行触发 UAV 侦察重分配和 HELI 打击分配。
- 编队/执行层放到独立执行线程里跑。

## Key Changes

- 在 Brain 初始化阶段补一个 `ScenarioInitializer`：把场景/外部输入统一整理成 `MissionContext.world_state`，包括 `aoi/aois`、天气、地形、平台、目标、集结点、坐标系信息。
- MILP 输入保持当前约定：`aoi(row,col)`、`platforms(pid,type,pos,sensors,munitions,alt,lost)`、`targets(tid,type,pos,value,threat,confirmed,alive)`、`grid_weather/grid_terrain`、`staging_position`。
- “多片侦察”统一抽象为 `ReconArea`：单 AOI 走现有 `c0..c4`；多 AOI 走 `MultiAOITaskAllocator.run()`，并由 Brain 保存/回传 `aoi_route_state` 和 `execution_feedback`。
- 新增一个 mission 级 `FlightExecutionWorker` 独立线程，内部管理每架飞机的 `PlatformRunner`，阶段为 `FORMATION_TRANSIT -> PATROL -> TRACK/STRIKE_TRANSIT -> ATTACK/DONE/LOST`。
- 线程边界：执行线程独占仿真 `step` 和平台控制；Brain 线程只通过队列收事件、发新计划，避免多个线程同时改 Isaac scene。
- 目标发现时新增事件 `TARGET_DETECTED`：Brain 更新目标为 `confirmed=true`，把目标加入 `pending_strike_targets`，同时调用 PPO 的 `TARGET_DISCOVERED` 做剩余侦察区重分配，并调用 MILP 打击分配。
- 飞机损毁时继续走事件触发：UAV 损毁调用 PPO `UAV_DAMAGE` 后重建巡航/航路；HELI 损毁或打击失败则重新调用 MILP action 分配未完成目标。
- 打击阵位按你的设定新增 `target_point` 模式：阵位选择器输出目标点本身，MPPI 规划到目标点，到达后执行打击。
- 状态向量策略：大脑持续维护最新态势快照，但 PPO/MILP 只在事件发生时读取最新快照并推理；DRL 避障才是每帧/每航段实时输入 54 维观测向量。
- 保留 FSM 里的 `REPLAN` 作为“算法失败重试”；战术重规划改为事件处理器，不再混在失败重试状态里。

## Interfaces

- `MissionContext` 增加：`aois`、`aoi_route_state`、`pending_strike_targets`、`engaged_targets`、`active_action_plans`、`runtime_events`。
- `MissionEventType` 增加：`TARGET_DETECTED`、`PLATFORM_LOST`、`RECON_CELL_DONE`、`STRIKE_POSITION_REACHED`、`ATTACK_FINISHED`。
- `MissionBrain` 增加：`handle_target_detected(payload)`、`handle_platform_loss(payload)`、`handle_attack_finished(payload)`，均返回可下发给执行线程的新计划。
- `PPOAllocationAdapter` 增加：`handle_target_discovered()`、`handle_target_destroyed()`，复用现有 `ReallocationService.handle_event()`。
- `MILPTaskAllocator.allocate_action()` 支持按 `target_ids` 或 `pending_strike_targets` 限定打击范围，避免重复给已接战目标分配飞机。

## Test Plan

- 单元测试：初始化后能生成合法 MILP 输入；单 AOI 和多 AOI 都能得到侦察任务。
- 单元测试：`TARGET_DETECTED` 会标记目标确认、调用 PPO `TARGET_DISCOVERED`、调用 MILP action，并返回打击航路，同时侦察继续。
- 单元测试：`PLATFORM_LOST` 对 UAV 调 PPO 并重建巡航，对 HELI 重新分配未完成打击目标。
- 单元测试：PPO 只在事件时收到状态，DRL 在巡航航段内逐帧收到观测。
- 集成测试：假场景跑完整链路：初始化 -> MILP 侦察 -> 编队线程到达 -> `search_planner` 巡航 -> 发现目标 -> 并行打击 -> 目标摧毁 -> UAV 释放/重分配 -> 任务完成。
- 并发测试：执行线程可正常 `start/stop/join`，异常会变成 Brain 的 `EXECUTION_FAILED`，不会遗留运行线程。

## Assumptions

- “好几片侦察区域”同时兼容单 AOI 内 `c0..c4` 和多个 AOI；未传 `aois` 时默认按当前单 AOI 逻辑。
- 编队线程采用“一个 mission 一个执行线程”，不是每架飞机一个线程。
- 发现目标后，发现者 UAV 默认转跟踪，原搜索区域由 PPO 重分配。
- 已分配打击的目标进入 `engaged_targets`，不会被下一次 MILP 重复分配。
- PPO 模型或真实 MILP 不可用时，继续使用现有占位/降级逻辑保持流程可跑。
