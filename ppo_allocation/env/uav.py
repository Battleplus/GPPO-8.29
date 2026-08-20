"""无人机实体模型。

使用 dataclass 定义无人机的状态属性，
包含位置、传感器类型、存活状态、当前任务、负责区域等。
"""

from dataclasses import dataclass, field
from typing import Set
from config import SensorType, TaskType, NO_TARGET


@dataclass
class UAV:
    """无人机实体。

    Attributes:
        uid:            无人机唯一编号 (0 ~ NUM_UAVS-1)
        x, y:           当前二维坐标位置
        sensor:         搭载的传感器类型 (EO 或 SAR)
        alive:          是否存活（损毁后不再参与分配）
        sensor_failed:  传感器是否故障
        task:           当前任务类型 (SEARCH / TRACK / IDLE)
        regions:        当前负责的搜索区域集合（跟踪任务时为空）
        target_id:      正在跟踪的目标编号（仅 TRACK 任务有效；否则为 NO_TARGET）
    """
    uid: int
    x: float
    y: float
    sensor: SensorType
    alive: bool = True
    sensor_failed: bool = False
    task: TaskType = TaskType.SEARCH
    regions: Set[int] = field(default_factory=set)
    target_id: int = NO_TARGET

    @property
    def assigned_count(self) -> int:
        """返回该无人机当前负责的搜索区域数量。"""
        return len(self.regions)

    def clear_search_regions(self) -> Set[int]:
        """清空该无人机的所有搜索区域。

        如果当前任务是 SEARCH，将其置为 IDLE。
        用于无人机转入跟踪或被摧毁时，释放其原来的搜索区域。

        Returns:
            旧的区域集合副本，供调用者后续处理（如标记 need_reassign）。
        """
        old_regions = set(self.regions)
        self.regions.clear()
        if self.task == TaskType.SEARCH:
            self.task = TaskType.IDLE
        return old_regions
