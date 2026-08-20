# UAV PPO Task Allocation

本项目是一个无人机搜索任务局部重分配的 PPO 代码框架。

核心设定：

- PPO 不负责跟踪任务；
- 目标发现后，由规则直接让发现无人机进入 TRACK；
- PPO 只负责突发事件后的搜索区域局部重分配；
- 动作为 `MultiDiscrete([6, 6, 6, 6])`；
- 每个区域动作 `a_Rj ∈ {KEEP, U0, U1, U2, U3, NO_UAV}`；
- 使用 `sb3-contrib` 的 `MaskablePPO` 支持动作 mask；
- 训练阶段暂时使用随机事件生成器；
- 后续真实应用时，可将完整仿真产生的事件注入环境；
- `apply.py` 输出标准 JSON，并生成 before/after 可视化动画。

## 安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py
```

模型会保存到：

```text
results/models/<run_name>/maskable_ppo_uav_task_allocation.zip
```

## 测试

修改 `evaluate.py` 中的模型路径后运行：

```bash
python evaluate.py
```

## 应用模型并生成 JSON 与动画

```bash
python apply.py --model results/models/<run_name>/maskable_ppo_uav_task_allocation.zip
```

输出：

```text
results/eval/ppo_assignment_output.json
results/eval/ppo_before_after_snapshots.json
results/eval/ppo_before_after.gif
```

## 给路径规划模块的 JSON 输出示例

```json
{
  "region_assignments": {
    "R0": "U1",
    "R1": "U2",
    "R2": "U1",
    "R3": "U3"
  },
  "uav_tasks": {
    "U1": {
      "alive": true,
      "task": "SEARCH",
      "sensor": "SAR",
      "regions": ["R0", "R2"],
      "target_id": null,
      "target_points": [[12.5, 37.5], [12.5, 12.5]]
    }
  }
}
```
