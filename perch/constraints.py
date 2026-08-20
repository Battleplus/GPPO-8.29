"""Hard constraints for attack-position optimisation."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .terrain_analyzer import TerrainFn, terrain_raycast


ConstraintFn = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        float,
        float,
        float,
        TerrainFn,
    ],
    float,
]

DEFAULT_MAX_DESIGNATION_ANGLE_DEG = 45.0


def c_los(
    pos: np.ndarray,
    target: np.ndarray,
    designator: np.ndarray | None,
    min_range: float,
    max_range: float,
    max_designation_angle_deg: float,
    terrain_fn: TerrainFn,
) -> float:
    """Require clear terrain LOS from shooter and optional designator."""
    _ = min_range, max_range, max_designation_angle_deg
    if not terrain_raycast(pos, target, terrain_fn):
        return 1.0
    if designator is not None and not terrain_raycast(
        designator, target, terrain_fn
    ):
        return 1.0
    return 0.0


def c_designation_zone(
    pos: np.ndarray,
    target: np.ndarray,
    designator: np.ndarray | None,
    min_range: float,
    max_range: float,
    max_designation_angle_deg: float,
    terrain_fn: TerrainFn,
) -> float:
    """Keep shooter/designator geometry within the guidance-angle limit."""
    _ = min_range, max_range, terrain_fn
    if designator is None:
        return 0.0

    bearing_pos = math.degrees(math.atan2(
        float(pos[1]) - float(target[1]),
        float(pos[0]) - float(target[0]),
    ))
    bearing_des = math.degrees(math.atan2(
        float(designator[1]) - float(target[1]),
        float(designator[0]) - float(target[0]),
    ))
    delta = abs((bearing_pos - bearing_des + 180.0) % 360.0 - 180.0)
    return max(0.0, delta - float(max_designation_angle_deg)) / max(
        1e-6, float(max_designation_angle_deg)
    )


def c_weapon_range(
    pos: np.ndarray,
    target: np.ndarray,
    designator: np.ndarray | None,
    min_range: float,
    max_range: float,
    max_designation_angle_deg: float,
    terrain_fn: TerrainFn,
) -> float:
    """Require three-dimensional slant range inside the weapon envelope."""
    _ = designator, max_designation_angle_deg, terrain_fn
    dist = float(np.linalg.norm(pos - target))
    if dist < min_range:
        return (min_range - dist) / max(1e-6, min_range)
    if dist > max_range:
        return (dist - max_range) / max(1e-6, max_range)
    return 0.0


def c_terrain_valid(
    pos: np.ndarray,
    target: np.ndarray,
    designator: np.ndarray | None,
    min_range: float,
    max_range: float,
    max_designation_angle_deg: float,
    terrain_fn: TerrainFn,
) -> float:
    """Reject positions below the terrain surface."""
    _ = target, designator, min_range, max_range
    _ = max_designation_angle_deg
    z_terrain = float(terrain_fn(float(pos[0]), float(pos[1])))
    if float(pos[2]) < z_terrain:
        return (z_terrain - float(pos[2])) / max(1.0, abs(z_terrain))
    return 0.0


CONSTRAINT_FUNCTIONS: list[ConstraintFn] = [
    c_los,
    c_designation_zone,
    c_weapon_range,
    c_terrain_valid,
]

CONSTRAINT_NAMES: list[str] = [
    "LOS",
    "DesignationZone",
    "WeaponRange",
    "TerrainValid",
]
