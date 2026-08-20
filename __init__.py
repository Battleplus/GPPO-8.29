"""SAR UAV search path planner with mountain obstacle avoidance.

Provides:
  - Four search-pattern generators (racetrack, polygon, rounded polygon, figure-8)
  - A* global path planning over 2.5-D elevation grids
  - Dubins-curve smoothing for flyable waypoint sequences
  - Arbitrary search-area selection within the 300-km combat scene

Quick start
-----------
>>> from sar_search_planner import PlannerConfig, plan
>>> config = PlannerConfig(
...     area_bounds_km=(-50, -30, 50, 30),
...     pattern="racetrack",
... )
>>> result = plan(config)
>>> for wp in result.waypoints[:5]:
...     print(f"({wp.x:.1f}, {wp.y:.1f}, z={wp.z:.1f})")
"""

from .search_planner.config import PlannerConfig
from .search_planner.planner import plan, PlannerResult, Waypoint
from .search_planner.area import (
    SearchArea,
    area_from_bounds_km,
    area_from_center_km,
    area_from_grid_cell,
)
from .search_planner.obstacles import MountainObstacle
from .search_planner.visualize import export_waypoints_usd, export_search_area_boundary, export_mountain_overlay, export_sar_swath, export_uav_platform
from .search_planner.mission import (
    MissionPlan,
    SearchMissionPlan,
    compute_start_near_area,
    find_best_entry,
    plan_mission,
    plan_mission_for_platform,
    plan_search_mission,
    export_mission_usd,
    compute_best_exit,
    compute_tangent_entry,
    build_multi_region_cycle,
)

__all__ = [
    "PlannerConfig",
    "plan",
    "PlannerResult",
    "Waypoint",
    "SearchArea",
    "area_from_bounds_km",
    "area_from_center_km",
    "area_from_grid_cell",
    "MountainObstacle",
    "export_waypoints_usd",
    "export_search_area_boundary",
    "export_mountain_overlay",
    "export_sar_swath",
    "export_uav_platform",
    "MissionPlan",
    "SearchMissionPlan",
    "compute_start_near_area",
    "find_best_entry",
    "plan_mission",
    "plan_mission_for_platform",
    "plan_search_mission",
    "export_mission_usd",
    "compute_best_exit",
    "compute_tangent_entry",
    "build_multi_region_cycle",
]
