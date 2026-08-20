"""目标实体模型。

目标分布在地图各区域中，无人机在搜索过程中可以发现并追踪它们。
"""

from dataclasses import dataclass
from config import TargetType, NO_UAV


@dataclass
class Target:
    """目标/威胁实体。

    Attributes:
        tid:          目标唯一编号 (0 ~ NUM_TARGETS-1)
        target_type:  目标类型 (CAR: 可移动车辆 / COMMAND: 固定指挥所)
        x, y:         目标当前坐标位置
        region:       目标所在的区域编号
        movable:      是否可移动（CAR 为 True，COMMAND 为 False）
        discovered:   是否已被无人机发现
        tracked:      是否正在被无人机跟踪
        destroyed:    是否已被摧毁
        tracker_id:   正在跟踪此目标的无人机编号 (NO_UAV 表示未被跟踪)
    """
    tid: int
    target_type: TargetType
    x: float
    y: float
    region: int
    movable: bool = False
    discovered: bool = False
    tracked: bool = False
    destroyed: bool = False
    tracker_id: int = NO_UAV
