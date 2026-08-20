from __future__ import annotations

import math
from dataclasses import dataclass

from .area import SearchArea


@dataclass
class MountainObstacle:
    """Cylindrical mountain obstacle model."""

    obstacle_id: str       # e.g. "MountainPeak_West"
    center_x: float        # scene units
    center_y: float
    radius_units: float    # cylinder radius
    height_units: float    # top-of-mountain height (scene units)


def build_all_mountains(
    map_size_units: float = 3000.0,
    visual_height_units: float = 150.0,
    meters_per_unit: float = 100.0,
    terrain_height_fn=None,
) -> list[MountainObstacle]:
    """Build MountainObstacle list from MOUNTAIN_OBSTACLE_SPECS.

    Mirrors the computation in _build_environment_obstacles() from
    scenes/air_combat_scene.py, mountain branch only.

    If *terrain_height_fn* is None the real scenes.air_combat_scene.terrain_height
    is imported lazily.
    """
    if terrain_height_fn is None:
        from scenes.air_combat_scene import terrain_height as _th

        terrain_height_fn = _th

    from scenes.air_combat_scene import MOUNTAIN_OBSTACLE_SPECS

    half = map_size_units * 0.5
    height_scale = visual_height_units
    obstacles: list[MountainObstacle] = []

    for name, nx, ny, radius_ratio in MOUNTAIN_OBSTACLE_SPECS:
        x = float(nx * half)
        y = float(ny * half)
        z = terrain_height_fn(x, y, map_size_units, height_scale)
        radius = max(3.0, map_size_units * radius_ratio)
        h = max(z, height_scale * 0.35)
        obstacles.append(
            MountainObstacle(
                obstacle_id=name,
                center_x=x,
                center_y=y,
                radius_units=radius,
                height_units=h,
            )
        )
    return obstacles


def filter_mountains_in_area(
    mountains: list[MountainObstacle],
    area: SearchArea,
) -> list[MountainObstacle]:
    """Return mountains whose bounding circle intersects *area*."""
    result: list[MountainObstacle] = []
    for m in mountains:
        # AABB-circle overlap test (fast rejection)
        closest_x = max(area.x_min, min(area.x_max, m.center_x))
        closest_y = max(area.y_min, min(area.y_max, m.center_y))
        dx = m.center_x - closest_x
        dy = m.center_y - closest_y
        if dx * dx + dy * dy <= m.radius_units * m.radius_units:
            result.append(m)
    return result


def check_collision(
    x: float,
    y: float,
    z: float,
    mountain: MountainObstacle,
    safety_xy: float = 0.0,
) -> bool:
    """Return True if (x, y, z) collides with the mountain cylinder."""
    dx = x - mountain.center_x
    dy = y - mountain.center_y
    dist2 = dx * dx + dy * dy
    limit = mountain.radius_units + safety_xy
    if dist2 >= limit * limit:
        return False
    return z < mountain.height_units


def collision_detail(
    x: float,
    y: float,
    z: float,
    mountain: MountainObstacle,
    safety_xy: float = 0.0,
) -> tuple[bool, float, float, float]:
    """Return (collides, overlap_dist, push_dir_x, push_dir_y).

    overlap_dist  > 0 means how far the point is inside the safety margin.
    push_dir is the unit vector from mountain centre to the point.
    """
    dx = x - mountain.center_x
    dy = y - mountain.center_y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-9:
        dist = 1e-9
        dx = 1e-9

    limit = mountain.radius_units + safety_xy
    if dist >= limit or z >= mountain.height_units:
        return False, 0.0, dx / dist, dy / dist

    overlap = limit - dist
    return True, overlap, dx / dist, dy / dist
