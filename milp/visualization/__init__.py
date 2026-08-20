"""
可视化前端包 —— 与后端 MILP 解耦的只读适配层。

提供:
  - build_visualization_state: 纯函数，snapshot+plan → 可序列化 viz dict
  - render_figure:            Plotly 战场地图渲染
  - load_default_snapshot:    默认演示场景（复用测试样例）
"""

from visualization.state_builder import build_visualization_state
from visualization.battlefield_map import render_figure
from visualization.scenario import load_default_snapshot
from visualization.constants import (
    MAP_SIZE_KM, COLOR_UAV, COLOR_HELI, COLOR_TARGET,
    SYMBOL_UAV, SYMBOL_HELI, SYMBOL_TARGET,
)

__all__ = [
    "build_visualization_state",
    "render_figure",
    "load_default_snapshot",
    "MAP_SIZE_KM",
    "COLOR_UAV", "COLOR_HELI", "COLOR_TARGET",
    "SYMBOL_UAV", "SYMBOL_HELI", "SYMBOL_TARGET",
]
