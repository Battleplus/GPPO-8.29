from __future__ import annotations

import math
from dataclasses import dataclass, field

from .area import (
    SearchArea,
    area_from_bounds_km,
    area_from_center_km,
    area_from_grid_cell,
)
from .config import PlannerConfig
from .obstacles import MountainObstacle, build_all_mountains, check_collision, filter_mountains_in_area
from .terrain import TerrainGrid
from .paths import generate_path
from .astar_planner import astar_avoid
from .path_smoother import smooth_waypoints


@dataclass
class Waypoint:
    """A single 3-D waypoint in scene units."""

    x: float
    y: float
    z: float          # absolute altitude (scene units)
    terrain_z: float   # ground height at this location
    yaw_deg: float     # heading (0=north, clockwise positive)


@dataclass
class PlannerResult:
    waypoints: list[Waypoint]
    raw_waypoints_before_avoidance: list[Waypoint]
    mountains_in_area: list[MountainObstacle]
    search_area: SearchArea
    config: PlannerConfig
    stats: dict = field(default_factory=dict)


def plan(config: PlannerConfig | None = None) -> PlannerResult:
    """Main entry point.  Returns a PlannerResult with final waypoints."""
    if config is None:
        config = PlannerConfig()

    meters_per_unit = config.meters_per_unit
    map_size_units = config.map_size_km * 1000.0 / meters_per_unit
    # Use *visual* heights so waypoints match what you see in Isaac Sim
    # (terrain has 10× vertical exaggeration for display)
    visual_height_units = (
        config.mountain_height_m * config.terrain_vertical_exaggeration / meters_per_unit
    )

    # ── [1] Define search area ─────────────────────────────
    area = _resolve_area(config, map_size_units)

    # ── [2] Build mountains & filter to area ────────────────
    all_mountains = build_all_mountains(
        map_size_units, visual_height_units, meters_per_unit
    )
    area_mountains = filter_mountains_in_area(all_mountains, area)

    # ── [3] Build terrain grid ──────────────────────────────
    terrain = TerrainGrid(
        area,
        area_mountains,
        resolution_m=config.astar_grid_resolution_m,
        safety_xy_m=config.obstacle_safety_xy_m,
        map_size_units=map_size_units,
        visual_height_units=visual_height_units,
        meters_per_unit=meters_per_unit,
    )

    # ── [4] Generate 2-D path ──────────────────────────────
    pattern_kwargs = _pattern_kwargs(config, meters_per_unit)
    # Auto-scale path to fit inside the search area if too large
    _auto_scale_pattern(config, pattern_kwargs, area)
    xy_raw = generate_path(
        pattern=config.pattern,
        center_x=area.center_x,
        center_y=area.center_y,
        angle_deg=config.angle_deg,
        clockwise=config.clockwise,
        meters_per_unit=meters_per_unit,
        **pattern_kwargs,
    )

    # ── [5] A* obstacle avoidance (per-segment) ────────────
    collision_before = _count_collisions(xy_raw, area_mountains, config, meters_per_unit)

    xy_avoided: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(xy_raw):
        xy_avoided.append((x, y))
        if i == 0:
            continue
        px, py = xy_avoided[-2]
        detour = astar_avoid(px, py, x, y, terrain)
        if detour:
            # Insert detour waypoints between px and the current point
            xy_avoided.pop()  # remove the endpoint
            xy_avoided.pop()  # remove the start of segment
            xy_avoided.append((px, py))  # re-add start
            xy_avoided.extend(detour)
            xy_avoided.append((x, y))

    # ── [6] First-pass Z assignment ────────────────────────
    alt_units = config.altitude_agl_m / meters_per_unit
    safety_z_units = config.obstacle_safety_z_m / meters_per_unit

    waypoints_raw: list[Waypoint] = []
    for i, (x, y) in enumerate(xy_avoided):
        tz = terrain.height_at(x, y)
        z = tz + alt_units
        # Altitude climb over mountains
        for m in area_mountains:
            if check_collision(x, y, z, m, config.obstacle_safety_xy_m / meters_per_unit):
                z = max(z, m.height_units + safety_z_units)
        wp = Waypoint(x=float(x), y=float(y), z=float(z), terrain_z=float(tz), yaw_deg=0.0)
        waypoints_raw.append(wp)

    # ── [7] Compute headings ───────────────────────────────
    for i in range(len(waypoints_raw)):
        j = (i + 1) % len(waypoints_raw)
        dx = waypoints_raw[j].x - waypoints_raw[i].x
        dy = waypoints_raw[j].y - waypoints_raw[i].y
        waypoints_raw[i].yaw_deg = math.degrees(math.atan2(dx, dy))

    # ── [8] Dubins smoothing ───────────────────────────────
    turn_r = config.dubins_turn_radius_m / meters_per_unit
    step = config.dubins_sample_step_m / meters_per_unit

    xy_for_dubins = [(wp.x, wp.y) for wp in waypoints_raw]
    # Use yaw of first waypoint as start pose for Dubins (in radians, heading=atan2(dx,dy))
    start_th = math.radians(waypoints_raw[0].yaw_deg)
    force_dubins = (config.pattern == "sar_polygon")
    xy_smoothed = smooth_waypoints(
        xy_for_dubins, turn_r, step, (xy_for_dubins[0][0], xy_for_dubins[0][1], start_th),
        force_dubins=force_dubins,
    )

    # ── [9] Z re-assignment after smoothing ────────────────
    waypoints_final: list[Waypoint] = []
    for i, (x, y) in enumerate(xy_smoothed):
        tz = terrain.height_at(x, y)
        z = tz + alt_units
        for m in area_mountains:
            if check_collision(x, y, z, m, config.obstacle_safety_xy_m / meters_per_unit):
                z = max(z, m.height_units + safety_z_units)
        yaw = 0.0
        if i < len(xy_smoothed) - 1:
            dx = xy_smoothed[i + 1][0] - x
            dy = xy_smoothed[i + 1][1] - y
            yaw = math.degrees(math.atan2(dx, dy))
        wp = Waypoint(x=float(x), y=float(y), z=float(z), terrain_z=float(tz), yaw_deg=float(yaw))
        waypoints_final.append(wp)

    # ── [10] Resample to uniform spacing ────────────────────
    spacing = config.dubins_sample_step_m / meters_per_unit  # e.g. 10 units = 1 km
    waypoints_final = _resample_uniform(waypoints_final, spacing, terrain, alt_units,
                                         area_mountains, config, meters_per_unit)

    # ── [11] Deduplicate consecutive near-duplicates ───────
    waypoints_final = _deduplicate(waypoints_final, tol=0.05)

    # ── Stats ──────────────────────────────────────────────
    collision_after = _count_collisions(
        [(w.x, w.y) for w in waypoints_final], area_mountains, config, meters_per_unit
    )
    path_length = 0.0
    for i in range(1, len(waypoints_final)):
        dx = waypoints_final[i].x - waypoints_final[i - 1].x
        dy = waypoints_final[i].y - waypoints_final[i - 1].y
        path_length += math.hypot(dx, dy) * meters_per_unit / 1000.0

    return PlannerResult(
        waypoints=waypoints_final,
        raw_waypoints_before_avoidance=waypoints_raw,
        mountains_in_area=area_mountains,
        search_area=area,
        config=config,
        stats={
            "collision_count_before": collision_before,
            "collision_count_after": collision_after,
            "path_length_km": round(path_length, 2),
            "mountains_in_area": len(area_mountains),
            "raw_waypoint_count": len(waypoints_raw),
            "final_waypoint_count": len(waypoints_final),
        },
    )


# ── helpers ──────────────────────────────────────────────────


def _auto_scale_pattern(
    config: PlannerConfig, kwargs: dict, area: SearchArea
) -> None:
    """Shrink path parameters so the full shape fits inside *area*."""
    area_half = min(area.half_width, area.half_height)

    if config.pattern == "figure_eight":
        span = kwargs["line_units"] / 2.0 + kwargs["radius_units"]
    elif config.pattern in ("sar_polygon", "sar_rounded"):
        span = kwargs["radius_units"]
    elif config.pattern == "racetrack":
        span = max(kwargs["length_units"], kwargs["width_units"]) / 2.0
    else:
        return

    if span > area_half:
        scale = area_half / span * 0.9
        for key in ("radius_units", "line_units", "length_units",
                     "width_units", "turn_radius_units"):
            if key in kwargs:
                kwargs[key] *= scale


def _resolve_area(config: PlannerConfig, map_size_units: float) -> SearchArea:
    """Pick the highest-priority area specification."""
    if config.area_bounds_km is not None:
        return area_from_bounds_km(
            *config.area_bounds_km,
            config.map_size_km,
            config.meters_per_unit,
        )
    if config.area_center_km is not None:
        return area_from_center_km(
            *config.area_center_km,
            config.area_width_km,
            config.area_height_km,
            config.map_size_km,
            config.meters_per_unit,
        )
    if config.grid_row is not None and config.grid_col is not None:
        return area_from_grid_cell(
            config.grid_row,
            config.grid_col,
            config.map_size_km,
            config.meters_per_unit,
        )
    # Default: centre of map, default size
    return area_from_center_km(
        0.0, 0.0, config.area_width_km, config.area_height_km,
        config.map_size_km, config.meters_per_unit,
    )


def _pattern_kwargs(config: PlannerConfig, mpu: float) -> dict:
    """Convert km-level config values to scene units for path generators."""
    p = config.pattern
    if p == "racetrack":
        return {
            "length_units": config.racetrack_length_km * 1000.0 / mpu,
            "width_units": config.racetrack_width_km * 1000.0 / mpu,
            "path_count": config.racetrack_path_count,
        }
    if p == "sar_polygon":
        return {
            "radius_units": config.sar_radius_km * 1000.0 / mpu,
            "sides": config.sar_sides,
            "loops": config.sar_loops,
        }
    if p == "sar_rounded":
        return {
            "radius_units": config.sar_radius_km * 1000.0 / mpu,
            "sides": config.sar_sides,
            "turn_radius_units": config.sar_turn_radius_km * 1000.0 / mpu,
        }
    if p == "figure_eight":
        return {
            "radius_units": config.eight_radius_km * 1000.0 / mpu,
            "line_units": config.eight_line_km * 1000.0 / mpu,
            "loops": config.eight_loops,
        }
    return {}


def _count_collisions(
    xy: list[tuple[float, float]],
    mountains: list[MountainObstacle],
    config: PlannerConfig,
    mpu: float,
) -> int:
    safety = config.obstacle_safety_xy_m / mpu
    alt = config.altitude_agl_m / mpu  # rough — just for XY pass
    count = 0
    for x, y in xy:
        for m in mountains:
            if check_collision(x, y, alt, m, safety):
                count += 1
                break
    return count


def _deduplicate(
    wps: list[Waypoint], tol: float = 0.05
) -> list[Waypoint]:
    if len(wps) < 2:
        return wps
    out = [wps[0]]
    for wp in wps[1:]:
        dx = wp.x - out[-1].x
        dy = wp.y - out[-1].y
        if math.hypot(dx, dy) > tol:
            out.append(wp)
    return out


def _resample_uniform(
    wps: list[Waypoint],
    spacing: float,
    terrain,
    alt_units: float,
    mountains: list,
    config,
    mpu: float,
) -> list[Waypoint]:
    """Resample waypoints at uniform spacing along the path (closed loop)."""
    if len(wps) < 2:
        return wps

    # Cumulative distances
    n = len(wps)
    cum = [0.0]
    for i in range(n):
        j = (i + 1) % n
        dx = wps[j].x - wps[i].x
        dy = wps[j].y - wps[i].y
        cum.append(cum[-1] + math.hypot(dx, dy))
    total = cum[-1]
    if total < spacing:
        return wps

    from .obstacles import check_collision
    safety_z_units = config.obstacle_safety_z_m / mpu

    num = max(2, int(total / spacing))
    result: list[Waypoint] = []
    for k in range(num):
        dist = k * total / num
        # Find segment containing dist
        for i in range(n):
            if cum[i + 1] >= dist:
                seg = cum[i + 1] - cum[i]
                t = (dist - cum[i]) / seg if seg > 1e-9 else 0.0
                t = max(0.0, min(1.0, t))
                j = (i + 1) % n
                x = wps[i].x + t * (wps[j].x - wps[i].x)
                y = wps[i].y + t * (wps[j].y - wps[i].y)
                tz = terrain.height_at(x, y)
                z = tz + alt_units
                for m in mountains:
                    if check_collision(x, y, z, m, config.obstacle_safety_xy_m / mpu):
                        z = max(z, m.height_units + safety_z_units)
                yaw = math.degrees(math.atan2(wps[j].x - wps[i].x, wps[j].y - wps[i].y))
                result.append(Waypoint(x=x, y=y, z=z, terrain_z=tz, yaw_deg=yaw))
                break

    return result
