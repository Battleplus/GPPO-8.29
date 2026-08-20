"""Narrow JSON bridge used by the C++ wrapper.

Keep Python objects on this side of the ABI.  The C++ caller only exchanges
UTF-8 JSON strings, so NumPy and Python implementation details never leak
into application code.
"""

from __future__ import annotations

import json
from dataclasses import fields

import numpy as np

from .mppi import MPPIConfig
from .obstacles import CylindricalObstacle, build_obstacles
from .planner import FormationMPPIPlanner


def _config(values: dict) -> MPPIConfig:
    allowed = {item.name for item in fields(MPPIConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown planner_config keys: {sorted(unknown)}")
    return MPPIConfig(**values)


def _obstacles(values: list[dict] | None) -> list[CylindricalObstacle]:
    if values is None:
        return build_obstacles()
    allowed = {item.name for item in fields(CylindricalObstacle)}
    result = []
    for index, value in enumerate(values):
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"obstacles[{index}] has unknown keys: {sorted(unknown)}")
        result.append(CylindricalObstacle(**value))
    return result


def plan_json(request_json: str) -> str:
    """Run one formation plan and return its serialized result."""
    request = json.loads(request_json)
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object")

    start = np.asarray(request.get("start", [-800.0, -600.0, 8.0]), dtype=float)
    goal = np.asarray(request.get("goal", [800.0, 600.0, 8.0]), dtype=float)
    if start.shape != (3,) or goal.shape != (3,):
        raise ValueError("start and goal must each contain exactly three numbers")

    planner_config = _config(request.get("planner_config", {}))
    planner = FormationMPPIPlanner(
        map_size_units=float(request.get("map_size_units", 3000.0)),
        meters_per_unit=float(request.get("meters_per_unit", 100.0)),
        terrain_vertical_exaggeration=float(
            request.get("terrain_vertical_exaggeration", 10.0)
        ),
        obstacles=_obstacles(request.get("obstacles")),
        planner_config=planner_config,
    )
    result = planner.plan(
        team_count=int(request.get("team_count", 4)),
        start=start,
        goal=goal,
        formation=str(request.get("formation", "v_shape")),
        spacing=float(request.get("spacing", 40.0)),
        depth_spacing=request.get("depth_spacing"),
        cruise_altitude=request.get("cruise_altitude"),
        member_assignments=request.get("member_assignments"),
        verbose=bool(request.get("verbose", False)),
    )
    return result.to_json(indent=None)
