"""Obstacle definitions for air combat scene -- mountain-based avoidance.

The obstacle positions and sizes are pre-computed from scene constants:
  SCENE_MAP_SIZE = 3000 units  (300 km / 100 m per unit)
  SCENE_HALF     = 1500 units
  SCENE_MOUNTAIN_HEIGHT = 15 units  (1500 m / 100 m/unit, physical collision height)

Only MOUNTAINS are used as obstacles for path planning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Scene constants (matching air_combat_scene.py defaults)
# ---------------------------------------------------------------------------

_SCENE_MAP_SIZE = 3000.0          # real_size_km=300 / meters_per_unit=100
_SCENE_HALF = _SCENE_MAP_SIZE * 0.5  # 1500.0
_SCENE_MOUNTAIN_HEIGHT = 15.0     # 1500m / 100 m/unit (physical, not visual)

# ---------------------------------------------------------------------------
# Raw specs (name, nx, ny, radius_ratio) -- from air_combat_scene.py
# ---------------------------------------------------------------------------

_MOUNTAIN_RAW: tuple[tuple[str, float, float, float], ...] = (
    ("MountainPeak_West",        -0.58, -0.20, 0.050),
    ("MountainRidge_Northwest",  -0.24,  0.34, 0.056),
    ("MountainPeak_North",       -0.04,  0.42, 0.052),
    ("MountainPeak_East",         0.34,  0.06, 0.048),
    ("MountainRidge_Northeast",   0.58,  0.44, 0.050),
)

# ---------------------------------------------------------------------------
# Pre-computed absolute obstacle parameters (cannot be accidentally changed)
# ---------------------------------------------------------------------------

MOUNTAIN_OBSTACLES: tuple[tuple[str, float, float, float], ...] = tuple(
    (name, nx * _SCENE_HALF, ny * _SCENE_HALF, radius_ratio * _SCENE_HALF)
    for name, nx, ny, radius_ratio in _MOUNTAIN_RAW
)
# Results in:
#   MountainPeak_West:        (-870.0, -300.0, 75.0)
#   MountainRidge_Northwest:  (-360.0,  510.0, 84.0)
#   MountainPeak_North:       ( -60.0,  630.0, 78.0)
#   MountainPeak_East:        ( 510.0,   90.0, 72.0)
#   MountainRidge_Northeast:  ( 870.0,  660.0, 75.0)


@dataclass
class CylindricalObstacle:
    """Cylindrical obstacle (mountain / forest / rock zone)."""

    name: str
    category: str
    center_xy: np.ndarray
    radius: float
    height: float
    priority: int


def build_obstacles(
    include_mountains: bool = True,
) -> list[CylindricalObstacle]:
    """Build obstacle list using pre-computed scene constants.

    Obstacle positions and sizes are FIXED, matching air_combat_scene.py.
    No parameters can accidentally change them.

    Args:
        include_mountains: Always True for now (only mountains supported).

    Returns:
        List of CylindricalObstacle (5 mountains).
    """
    obstacles: list[CylindricalObstacle] = []
    if include_mountains:
        for name, cx, cy, radius in MOUNTAIN_OBSTACLES:
            obstacles.append(CylindricalObstacle(
                name=name,
                category="mountain",
                center_xy=np.array([cx, cy], dtype=float),
                radius=radius,
                height=_SCENE_MOUNTAIN_HEIGHT,
                priority=10,
            ))
    return obstacles


def is_position_blocked(
    position_xy: np.ndarray,
    altitude: float,
    obstacles: list[CylindricalObstacle],
    clearance: float = 2.0,
) -> bool:
    """Check if a position is blocked by any obstacle.

    A position is blocked when:
      - distance to obstacle center < obstacle radius + clearance
      - AND altitude < obstacle height + clearance

    Args:
        position_xy: [x, y] coordinates in scene units.
        altitude: Current altitude in scene units.
        obstacles: List of obstacles to check against.
        clearance: Safety margin in scene units.

    Returns:
        True if blocked by at least one obstacle.
    """
    for obs in obstacles:
        dist = float(np.linalg.norm(position_xy - obs.center_xy))
        if dist < obs.radius + clearance:
            if altitude < obs.height + clearance:
                return True
    return False


# ---------------------------------------------------------------------------
# Terrain height function (copied from air_combat_scene.py)
# ---------------------------------------------------------------------------

def terrain_elevation(x: float, y: float, map_size: float = 3000.0, height_scale: float = 150.0) -> float:
    """Compute procedural terrain elevation at (x, y).

    This is an exact copy of ``air_combat_scene.terrain_height``,
    using the VISUAL height scale (150 = 15 physical * 10 exaggeration).

    Args:
        x, y: Position in scene units.
        map_size: Map size (default 3000).
        height_scale: Max terrain height in units (default 150 = visual).

    Returns:
        Terrain surface elevation at (x, y), in scene units.
    """
    half = max(1.0, map_size * 0.5)
    nx = float(x) / half
    ny = float(y) / half

    edge = max(abs(nx), abs(ny))
    edge_falloff = max(0.0, min(1.0, (1.0 - edge) / 0.16))
    edge_falloff = edge_falloff * edge_falloff * (3.0 - 2.0 * edge_falloff)

    warp_x = nx + 0.11 * math.sin(ny * math.pi * 2.2) + 0.055 * math.sin((nx + ny) * math.pi * 4.5)
    warp_y = ny + 0.10 * math.sin(nx * math.pi * 2.0) - 0.065 * math.cos((nx - ny) * math.pi * 3.5)
    ridge_a = math.exp(-((warp_y - 0.20 * math.sin(warp_x * math.pi * 1.6) - 0.10) ** 2) / 0.018)
    ridge_b = 0.82 * math.exp(-((warp_y + 0.35 * math.sin(warp_x * math.pi * 0.9) + 0.18) ** 2) / 0.034)
    ridge_c = 0.52 * math.exp(-((warp_x - 0.18 * math.sin(warp_y * math.pi * 1.8) + 0.24) ** 2) / 0.030)
    ridge_d = 0.36 * math.exp(-((warp_x + warp_y * 0.42 - 0.04 * math.sin(warp_y * math.pi * 5.2)) ** 2) / 0.022)

    peaks = 0.0
    for px, py, gain, spread in (
        (-0.58, -0.20, 0.68, 0.018),
        (-0.24, 0.34, 0.58, 0.020),
        (-0.04, 0.42, 0.92, 0.022),
        (0.34, 0.06, 1.00, 0.018),
        (0.58, 0.44, 0.64, 0.024),
    ):
        peaks += gain * math.exp(-(((warp_x - px) ** 2 + (warp_y - py) ** 2) / spread))

    valley = 0.62 * math.exp(-((warp_y + 0.08 + 0.18 * math.sin(warp_x * math.pi * 1.2)) ** 2) / 0.009)
    river_cut = 0.42 * math.exp(-((warp_x + 0.18 * math.sin(warp_y * math.pi * 2.4)) ** 2) / 0.007)
    escarpment = 0.22 * max(0.0, math.sin((warp_x * 1.3 - warp_y * 0.9) * math.pi * 3.0))
    rolling = (
        0.08 * math.sin(warp_x * math.pi * 5.0)
        + 0.06 * math.cos(warp_y * math.pi * 4.0)
        + 0.05 * math.sin((warp_x + warp_y) * math.pi * 9.0)
        + 0.035 * math.cos((warp_x * 1.7 - warp_y) * math.pi * 11.0)
    )
    normalized = max(
        0.0,
        min(
            1.0,
            0.06
            + 0.42 * ridge_a
            + 0.34 * ridge_b
            + 0.25 * ridge_c
            + ridge_d
            + peaks
            + escarpment
            + rolling
            - valley
            - river_cut,
        ),
    )
    normalized = min(1.0, normalized * 1.16) ** 0.86
    return height_scale * normalized * edge_falloff


def lift_waypoints_above_terrain(
    waypoints: list[np.ndarray],
    map_size: float = 3000.0,
    height_scale: float = 150.0,
    clearance: float = 15.0,
) -> list[np.ndarray]:
    """Adjust waypoint altitudes to be above terrain surface.

    Args:
        waypoints: List of [x, y, z] waypoints.
        map_size: Map size in units.
        height_scale: Visual terrain height scale (default 150).
        clearance: Minimum altitude above terrain (units).

    Returns:
        New waypoints with z lifted above terrain.
    """
    lifted: list[np.ndarray] = []
    for wp in waypoints:
        ground = terrain_elevation(float(wp[0]), float(wp[1]), map_size, height_scale)
        new_z = max(float(wp[2]), ground + clearance)
        lifted.append(np.array([float(wp[0]), float(wp[1]), new_z], dtype=float))
    return lifted
