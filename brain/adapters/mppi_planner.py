"""MPPI adapter with independent one-aircraft/one-target strike routes."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.result import AlgorithmResult
from ..domain.route import FormationPlan, Route
from ..domain.task import one_to_one_strike_tasks

logger = logging.getLogger(__name__)
_PROJECT = Path(__file__).resolve().parents[2]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

_MPPI_AVAILABLE = False
try:
    from mppi.planner import FormationMPPIPlanner as _FormationMPPIPlanner
    from mppi.obstacles import build_obstacles as _build_obstacles
    _MPPI_AVAILABLE = True
except ImportError:
    logger.warning("MPPI unavailable; using deterministic straight paths")


def _km_to_scene(
    km_x: float,
    km_y: float,
    meters_per_unit: float = 100.0,
    map_size_km: float = 300.0,
) -> tuple[float, float]:
    scale = 1000.0 / meters_per_unit
    half = map_size_km * 0.5
    return (km_x - half) * scale, (km_y - half) * scale


def _aoi_center_km(world: dict) -> tuple[float, float]:
    aoi = world.get("aoi", {"row": 3, "col": 4})
    row = int(aoi.get("row", 3))
    col = int(aoi.get("col", 4))
    return ((col - 1) * 50.0 + 25.0, (row - 1) * 50.0 + 25.0)


def _cell_centers_km(world: dict) -> dict[str, tuple[float, float]]:
    """返回 AOI 内各 cell 的中心坐标 (KM).

    AOI 网格布局 (50×50 km)::

            ┌──────────┬──────────┐
            │    c3    │    c4    │  每个子区 25×25 km
            │ 左上子区  │ 右上子区  │
            ├──────────┼──────────┤
            │    c1    │    c2    │
            │ 左下子区  │ 右下子区  │
            └──────────┴──────────┘
            c0 = 整个 AOI 中心 (巡逻区)

    Returns:
        ``{"c0": (x, y), "c1": (x, y), ..., "c4": (x, y)}``
        坐标单位 km.
    """
    aoi = world.get("aoi", {"row": 3, "col": 4})
    row = int(aoi.get("row", 3))
    col = int(aoi.get("col", 4))
    x0 = (col - 1) * 50.0   # AOI 左边界
    y0 = (row - 1) * 50.0   # AOI 下边界
    cx = x0 + 25.0           # AOI 中心 x
    cy = y0 + 25.0           # AOI 中心 y
    half = 12.5              # 子区半宽
    return {
        "c0": (cx, cy),                     # 巡逻区中心 = AOI 中心
        "c1": (x0 + half, y0 + half),       # 左下
        "c2": (x0 + 3 * half, y0 + half),   # 右下
        "c3": (x0 + half, y0 + 3 * half),   # 左上
        "c4": (x0 + 3 * half, y0 + 3 * half),  # 右上
    }


class MPPIFormationPlanner:
    """Plan formations for recon and independent routes for strike."""

    def __init__(
        self,
        map_size_units: float = 3000.0,
        meters_per_unit: float = 100.0,
        terrain_vertical_exaggeration: float = 10.0,
    ) -> None:
        self._map_size_units = map_size_units
        self._meters_per_unit = meters_per_unit
        self._terrain_vertical_exaggeration = (
            terrain_vertical_exaggeration
        )
        self._planner = None

    def plan_recon_route(
        self,
        context: Any,
        recon_allocation: Any,
    ) -> AlgorithmResult:
        """为侦察任务规划编队航路, 按 cell 分组, 每组飞往对应 cell 中心.

        流程:
        1. 从 ``recon_allocation`` 中提取各平台的 cell 分配
        2. 按 cell 分组 (同一 cell 的平台编队飞行)
        3. 每组从集结区 → cell 中心规划航路
        4. 单 cell → 编队飞行; 多 cell → 独立航路
        """
        if not recon_allocation:
            return AlgorithmResult.fail("No recon allocation to plan")

        # -- 按 cell 分组平台 ----------------------------------------------
        cell_platforms: dict[str, list[str]] = {}
        for task in recon_allocation:
            cell = str(getattr(task, "cell", "c0"))
            platform = str(getattr(task, "platform", ""))
            if platform:
                cell_platforms.setdefault(cell, []).append(platform)

        if not cell_platforms:
            return AlgorithmResult.fail("No recon platforms in allocation")

        world = getattr(context, "world_state", {})
        meters_per_unit = float(
            world.get("meters_per_unit", self._meters_per_unit)
        )
        map_size_km = float(world.get("map_size_km", 300.0))
        staging = world.get("staging_position", [150.0, 150.0])
        sx, sy = _km_to_scene(
            float(staging[0]), float(staging[1]),
            meters_per_unit, map_size_km,
        )
        start = (sx, sy, 80.0)

        cell_centers = _cell_centers_km(world)

        # -- 单 cell: 编队飞行 (原行为) ------------------------------------
        if len(cell_platforms) == 1:
            cell = next(iter(cell_platforms))
            platforms = cell_platforms[cell]
            goal_km = cell_centers.get(cell, cell_centers["c0"])
            gx, gy = _km_to_scene(
                goal_km[0], goal_km[1], meters_per_unit, map_size_km,
            )
            goal = (gx, gy, 80.0)
            roles = ["leader"] + [
                f"wing_{i}" for i in range(1, len(platforms))
            ]
            if _MPPI_AVAILABLE:
                return self._plan_formation_real(
                    team_count=len(platforms),
                    start=start, goal=goal,
                    formation="v_shape", phase="recon",
                )
            return self._plan_formation_placeholder(
                len(platforms), start, goal, "v_shape", roles,
            )

        # -- 多 cell: 独立航路 (类似 strike 模式) -------------------------
        routes: list[Route] = []
        for cell, platforms in cell_platforms.items():
            goal_km = cell_centers.get(cell, cell_centers["c0"])
            gx, gy = _km_to_scene(
                goal_km[0], goal_km[1], meters_per_unit, map_size_km,
            )
            goal = (gx, gy, 80.0)

            for platform in platforms:
                if _MPPI_AVAILABLE:
                    waypoints = self._plan_single_real(
                        start, goal, f"recon:{platform}:{cell}",
                    )
                    planner_name = "MPPI"
                else:
                    waypoints = self._straight_path(start, goal)
                    planner_name = "straight_line_fallback"

                if not waypoints:
                    return AlgorithmResult.fail(
                        f"Route planning failed for {platform} → {cell}"
                    )

                # 查对应的 task 获取 role/sensor
                task = next(
                    (t for t in recon_allocation
                     if str(getattr(t, "platform", "")) == platform
                      and str(getattr(t, "cell", "")) == cell),
                    None,
                )
                route = Route(
                    platform=platform,
                    target_id=cell,           # cell 作为目的地标识
                    position_id=f"cell_{cell}",
                    waypoints=waypoints,
                    role=str(getattr(task, "role", "area_scan")) if task else "area_scan",
                    metadata={
                        "cell": cell,
                        "cell_center_km": list(goal_km),
                        "sensor": str(getattr(task, "sensor", "")) if task else "",
                        "aoi": str(getattr(task, "aoi", "")) if task else "",
                        "planner": planner_name,
                    },
                )
                routes.append(route)

        plan = FormationPlan(
            formation_type="individual_recon",
            team_count=len(routes),
            success=True,
            center_path=routes[0].waypoints if routes else [],
            team_paths=[r.waypoints for r in routes],
            formation_roles=[r.role for r in routes],
            routes=routes,
            assignment_map=[
                {
                    "platform_id": r.platform,
                    "cell": r.target_id,
                    "route_index": i,
                }
                for i, r in enumerate(routes)
            ],
            planner_stats={
                "mode": "per_cell_recon",
                "route_count": len(routes),
                "cells": list(cell_platforms.keys()),
                "explanation": (
                    "各 UAV 独立飞往分配的侦察 cell; "
                    "同一 cell 内多机编队飞行, 不同 cell 间独立航路"
                ),
            },
        )
        return AlgorithmResult.ok(plan)

    def plan_action_route(
        self,
        context: Any,
        action_allocation: Any,
        selected_positions: Any = None,
    ) -> AlgorithmResult:
        """Plan one route per matched aircraft, preserving target identity."""
        tasks = one_to_one_strike_tasks(action_allocation or [])
        if not tasks:
            return AlgorithmResult.fail(
                "No one-to-one strike assignments to route"
            )
        if not selected_positions:
            return AlgorithmResult.fail("No selected attack positions")

        best_position: dict[tuple[str, str], Any] = {}
        for position in selected_positions:
            metadata = getattr(position, "metadata", {}) or {}
            key = (
                str(metadata.get("platform_id", "")),
                str(metadata.get("target_id", "")),
            )
            if not all(key):
                continue
            current = best_position.get(key)
            current_rank = (
                int(getattr(current, "metadata", {}).get("rank", 999))
                if current is not None else 999
            )
            if int(metadata.get("rank", 999)) < current_rank:
                best_position[key] = position

        world = getattr(context, "world_state", {})
        routes: list[Route] = []
        assignment_map: list[dict[str, Any]] = []
        for task in tasks:
            key = (str(task.platform), str(task.target))
            position = best_position.get(key)
            if position is None:
                return AlgorithmResult.fail(
                    "Missing selected position for "
                    f"platform={task.platform}, target={task.target}"
                )

            start = self._platform_start(
                context, str(task.platform), world
            )
            goal = (
                float(position.x),
                float(position.y),
                float(position.z),
            )
            if _MPPI_AVAILABLE:
                waypoints = self._plan_single_real(
                    start, goal, f"action:{task.platform}:{task.target}"
                )
                planner_name = "MPPI"
            else:
                waypoints = self._straight_path(start, goal)
                planner_name = "straight_line_fallback"
            if not waypoints:
                return AlgorithmResult.fail(
                    f"Route planning failed for {task.platform}"
                )

            position_meta = getattr(position, "metadata", {}) or {}
            explanation = [
                f"{task.platform} 仅绑定目标 {task.target}",
                (
                    f"航路终点采用阵位 {position.pos_id}，"
                    f"主弹种 {task.munition}"
                ),
                (
                    f"起点=({start[0]:.1f},{start[1]:.1f},{start[2]:.1f})，"
                    f"终点=({goal[0]:.1f},{goal[1]:.1f},{goal[2]:.1f})"
                ),
                f"规划器={planner_name}",
            ]
            route = Route(
                platform=str(task.platform),
                target_id=str(task.target),
                position_id=str(position.pos_id),
                waypoints=waypoints,
                role=str(getattr(task, "role", "striker")),
                metadata={
                    "munition": str(task.munition),
                    "assigned_munitions": dict(
                        task.assigned_munitions
                        or {str(task.munition): int(task.qty)}
                    ),
                    "planner": planner_name,
                    "position_explanation": position_meta.get(
                        "explanation", []
                    ),
                    "explanation": explanation,
                },
            )
            routes.append(route)
            assignment_map.append({
                "platform_id": route.platform,
                "target_id": route.target_id,
                "position_id": route.position_id,
                "munition": str(task.munition),
                "route_index": len(routes) - 1,
                "explanation": explanation,
            })

        plan = FormationPlan(
            formation_type="individual_strike",
            team_count=len(routes),
            success=True,
            center_path=routes[0].waypoints if routes else [],
            team_paths=[route.waypoints for route in routes],
            formation_roles=[route.role for route in routes],
            routes=routes,
            assignment_map=assignment_map,
            planner_stats={
                "mode": "one_aircraft_one_target",
                "route_count": len(routes),
                "platform_ids": [route.platform for route in routes],
                "target_ids": [route.target_id for route in routes],
                "explanation": (
                    "每架飞机使用自己的当前位置、目标阵位和独立航路；"
                    "不再共享首个推荐阵位"
                ),
            },
        )
        return AlgorithmResult.ok(plan)

    def _get_planner(self):
        if self._planner is None:
            self._planner = _FormationMPPIPlanner(
                map_size_units=self._map_size_units,
                meters_per_unit=self._meters_per_unit,
                terrain_vertical_exaggeration=(
                    self._terrain_vertical_exaggeration
                ),
                obstacles=_build_obstacles(),
            )
        return self._planner

    def _plan_single_real(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        phase: str,
    ) -> list[list[float]]:
        try:
            result = self._get_planner().plan(
                team_count=1,
                start=np.array(start, dtype=float),
                goal=np.array(goal, dtype=float),
                formation="column",
                spacing=1.0,
                verbose=False,
            )
            if not result.success:
                return []
            raw_path = (
                result.team_paths[0]
                if result.team_paths else result.center_path
            )
            return [
                waypoint.tolist()
                if hasattr(waypoint, "tolist") else list(waypoint)
                for waypoint in raw_path
            ]
        except Exception:
            logger.exception("MPPI %s failed", phase)
            return []

    def _plan_formation_real(
        self,
        team_count: int,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        formation: str,
        phase: str,
    ) -> AlgorithmResult:
        try:
            result = self._get_planner().plan(
                team_count=team_count,
                start=np.array(start, dtype=float),
                goal=np.array(goal, dtype=float),
                formation=formation,
                spacing=40.0,
                verbose=False,
            )
            plan = FormationPlan(
                formation_type=result.formation_type,
                team_count=result.team_count,
                success=result.success,
                center_path=[wp.tolist() for wp in result.center_path],
                team_paths=[
                    [wp.tolist() for wp in path]
                    for path in result.team_paths
                ],
                formation_roles=list(result.formation_roles),
                planner_stats=dict(result.planner_stats),
            )
            return AlgorithmResult.ok(plan)
        except Exception as exc:
            logger.exception("MPPI %s planning failed", phase)
            return AlgorithmResult.fail(str(exc))

    @staticmethod
    def _straight_path(
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        count: int = 10,
    ) -> list[list[float]]:
        start_np = np.array(start, dtype=float)
        goal_np = np.array(goal, dtype=float)
        return [
            (
                start_np + (goal_np - start_np) * (i / (count - 1))
            ).tolist()
            for i in range(count)
        ]

    def _plan_formation_placeholder(
        self,
        team_count: int,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        formation: str,
        roles: list[str],
    ) -> AlgorithmResult:
        center = self._straight_path(start, goal)
        offsets = np.linspace(-15.0, 15.0, max(1, team_count))
        paths = [
            [[wp[0], wp[1] + float(offsets[index]), wp[2]]
             for wp in center]
            for index in range(team_count)
        ]
        return AlgorithmResult.ok(FormationPlan(
            formation_type=formation,
            team_count=team_count,
            success=True,
            center_path=center,
            team_paths=paths,
            formation_roles=roles,
        ))

    def _platform_start(
        self,
        context: Any,
        platform_id: str,
        world: dict,
    ) -> tuple[float, float, float]:
        states = world.get("platform_states", [])
        if isinstance(states, dict):
            states = [
                {"platform_id": key, **value}
                for key, value in states.items()
            ]
        for state in states:
            sid = state.get(
                "platform_id", state.get("entity_id", state.get("pid"))
            )
            if str(sid) != platform_id:
                continue
            raw = state.get(
                "position_scene", state.get("position")
            )
            if raw is not None and len(raw) >= 3:
                return float(raw[0]), float(raw[1]), float(raw[2])

        meters_per_unit = float(
            world.get("meters_per_unit", self._meters_per_unit)
        )
        map_size_km = float(world.get("map_size_km", 300.0))
        terrain_fn = world.get("terrain_fn")
        for agent in getattr(context, "agents", []):
            if str(getattr(agent, "pid", "")) != platform_id:
                continue
            raw = getattr(agent, "position", (150.0, 150.0))
            x, y = _km_to_scene(
                float(raw[0]),
                float(raw[1]),
                meters_per_unit,
                map_size_km,
            )
            ground = (
                float(terrain_fn(x, y)) if callable(terrain_fn) else 0.0
            )
            z = ground + (
                float(getattr(agent, "altitude_km", 0.3))
                * 1000.0 / meters_per_unit
            )
            return x, y, z

        staging = world.get("staging_position", [150.0, 150.0])
        x, y = _km_to_scene(
            float(staging[0]),
            float(staging[1]),
            meters_per_unit,
            map_size_km,
        )
        ground = (
            float(terrain_fn(x, y)) if callable(terrain_fn) else 0.0
        )
        return x, y, ground + 300.0 / meters_per_unit
