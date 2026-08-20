"""突发事件实体模型。

事件是触发 PPO 局部重分配的信号。
每次决策步，环境生成一个事件，PPO 据此输出新的区域分配动作。
"""

from dataclasses import dataclass, field
from typing import List, Optional
from config import EventType, NO_UAV


@dataclass
class Event:
    """突发事件，驱动无人机搜索区域的重分配。

    Attributes:
        event_type:             事件类型枚举
        affected_regions:       受影响的区域编号列表（需要重新分配的区域）
        released_uav:           被释放的无人机编号（TARGET_DESTROYED 事件中）
        damaged_uav:            受损的无人机编号（UAV_DAMAGE 事件中）
        weather_disabled_uav:   因天气恶化而失效的无人机编号（WEATHER_INVALID 事件中，已弃用）
        description:            事件的可读描述文本
    """
    event_type: EventType
    affected_regions: List[int] = field(default_factory=list)
    released_uav: int = NO_UAV
    damaged_uav: int = NO_UAV
    weather_disabled_uav: int = NO_UAV
    description: str = ""

    def is_target_destroyed(self) -> bool:
        """判断当前事件是否为'目标被摧毁'类型。

        此事件特殊之处在于：它释放了一架无人机（原追踪者），
        PPO 可以将其重新分配到搜索任务中。
        """
        return self.event_type == EventType.TARGET_DESTROYED
