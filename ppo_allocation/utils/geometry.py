"""几何计算工具模块。

提供常用的几何/距离计算函数。
"""

import math


def distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """计算两点之间的欧几里得距离。

    常用于计算无人机到区域中心的距离，作为分配代价的一部分。

    Args:
        x1, y1: 第一个点的坐标
        x2, y2: 第二个点的坐标

    Returns:
        float: 两点间的直线距离
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
