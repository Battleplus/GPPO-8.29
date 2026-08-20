"""评估指标追踪模块。

MetricsTracker 在 episode 内逐步累积关键指标，
episode 结束后输出汇总统计，用于模型评估和训练监控。
"""

from collections import defaultdict
from config import NO_UAV, TaskType


class MetricsTracker:
    """单 episode 的指标追踪器。

    在每个 step 后调用 update() 累积统计量，
    episode 结束后调用 summary() 获取汇总指标。

    追踪的指标：
      - 区域分配率（已被合法分配的区域占比）
      - 平均奖励
      - 负载均衡差距（最大负载 - 最小负载）
      - 非法动作计数
      - NO_UAV 动作计数
      - PPO 局部重分配成功率（事件解决率）
    """

    def __init__(self):
        """初始化追踪器，重置所有统计量。"""
        self.reset()

    def reset(self):
        """重置所有累积统计量和步数计数器。"""
        self.data = defaultdict(float)
        self.steps = 0

    def update(self, env, reward, info):
        """根据当前环境状态和 step 返回信息更新统计量。

        Args:
            env:    UAVTaskAllocationEnv 环境对象
            reward: 当前步的即时奖励
            info:   env.step() 返回的 info 字典
        """
        self.steps += 1

        # 区域分配率：已被分配的合法区域占比
        assigned = sum(1 for r in env.regions.values() if r.assigned_uav != NO_UAV)
        self.data["region_assignment_rate_sum"] += assigned / len(env.regions)

        # 累积奖励
        self.data["reward_sum"] += reward

        # 负载均衡差距
        loads = [len(u.regions) for u in env.uavs.values() if u.alive and u.task != TaskType.TRACK]
        if len(loads) > 1:
            self.data["load_gap_sum"] += max(loads) - min(loads)

        # 非法动作/NO_UAV 计数
        self.data["invalid_action_count"] += info.get("invalid_count", 0)
        self.data["no_uav_count"] += info.get("no_uav_count", 0)

        # 事件成功解决次数
        self.data["event_success_count"] += float(info.get("event_success", False))

    def summary(self):
        """计算 episode 结束后的汇总统计。

        Returns:
            dict: 各项指标的均值/总和，键名与 README/评估输出对齐
                - avg_region_assignment_rate:     平均区域分配率
                - avg_reward:                     平均奖励
                - avg_load_gap:                   平均负载差距
                - invalid_action_count:           总非法动作数
                - no_uav_count:                   总 NO_UAV 动作数
                - ppo_local_reallocation_success_rate: 重分配成功率
        """
        denom = max(1, self.steps)  # 防止除以 0
        return {
            "avg_region_assignment_rate": self.data["region_assignment_rate_sum"] / denom,
            "avg_reward": self.data["reward_sum"] / denom,
            "avg_load_gap": self.data["load_gap_sum"] / denom,
            "invalid_action_count": self.data["invalid_action_count"],
            "no_uav_count": self.data["no_uav_count"],
            "ppo_local_reallocation_success_rate": self.data["event_success_count"] / denom,
        }
