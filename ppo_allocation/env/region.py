"""搜索区域实体模型。

每个区域是地图上的一个矩形子区域，具有天气状态和分配信息。
"""

from dataclasses import dataclass
from config import Weather, NO_UAV


@dataclass
class Region:
    """搜索区域实体。

    Attributes:
        rid:           区域唯一编号 (0 ~ NUM_REGIONS-1)
        center_x:      区域中心点 x 坐标
        center_y:      区域中心点 y 坐标
        weather:       当前天气 (SUNNY / RAINY)，保留字段（当前全部使用 SAR，天气不影响传感器）
        assigned_uav:  当前分配的搜索无人机编号 (NO_UAV 表示未分配)
        need_reassign: 是否需要 PPO 重新分配（事件触发后置为 True）
        priority:      区域优先级（可用于加权分配，当前默认为 1.0）
    """
    rid: int
    center_x: float
    center_y: float
    weather: Weather
    assigned_uav: int = NO_UAV
    need_reassign: bool = False
    priority: float = 1.0
