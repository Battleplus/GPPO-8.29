"""
aoi 子包：多 AOI 排序与任务调度模块。

对外导出：
  - AoiInfo            — 单个 AOI 的描述信息
  - AOIRouteState      — AOI 执行状态（排序结果 + 当前进度）
  - ExecutionFeedback  — 底层规控执行反馈
  - AOIRouter          — AOI 排序器
  - MultiAOIController — 多 AOI 任务控制器
"""

from .aoi_state import AoiInfo, AOIRouteState, ExecutionFeedback
from .aoi_router import AOIRouter
from .multi_aoi_controller import MultiAOIController

__all__ = [
    "AoiInfo",
    "AOIRouteState",
    "ExecutionFeedback",
    "AOIRouter",
    "MultiAOIController",
]
