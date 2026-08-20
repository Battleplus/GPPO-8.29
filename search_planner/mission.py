"""Mission executor: transit drone from start to search area, find best entry
point, follow the planned search path, and provide path interpolation for
animation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .area import SearchArea
from .planner import Waypoint, PlannerResult


def compute_start_near_area(
    search_area: SearchArea,
    offset_km: float = 15.0,
    direction_deg: float = 225.0,
    altitude_agl_m: float = 1200.0,
    meters_per_unit: float = 100.0,
    map_size_km: float = 300.0,
    height_scale: float = 10.0,
) -> tuple[float, float, float]:
    """Compute a drone start position near the search area.

    Places the drone *offset_km* outside the search-area boundary in the
    given compass direction (0=north, clockwise positive).  Default is
    southwest (225°) so the drone approaches from the opposite side of the
    typical patrol routes.

    Returns (x, y, z) in scene units.
    """
    from scenes.air_combat_scene import terrain_height

    # Find the boundary point of the search area in the given direction
    rad = math.radians(direction_deg)
    dir_x = math.sin(rad)   # east component of direction
    dir_y = math.cos(rad)   # north component of direction

    # Start at area centre, then walk outward until we hit the boundary,
    # then add the offset
    cx, cy = search_area.center_x, search_area.center_y
    hw, hh = search_area.half_width, search_area.half_height

    # Distance from centre to boundary in the given direction
    if abs(dir_x) < 1e-9:
        t_boundary = hh
    elif abs(dir_y) < 1e-9:
        t_boundary = hw
    else:
        tx = hw / abs(dir_x)
        ty = hh / abs(dir_y)
        t_boundary = min(tx, ty)

    offset_units = offset_km * 1000.0 / meters_per_unit
    t_start = t_boundary + offset_units

    x = cx + dir_x * t_start
    y = cy + dir_y * t_start

    map_size_units = map_size_km * 1000.0 / meters_per_unit
    ground_z = terrain_height(x, y, map_size_units, height_scale)
    z = ground_z + altitude_agl_m / meters_per_unit
    return x, y, z


@dataclass
class MissionPlan:
    """Complete SAR mission flight plan.

    transit_waypoints:  path from drone start to the entry point
    entry_index:        index into the original search waypoints where the
                        drone joins the search pattern
    search_waypoints:   the search path, reordered to start from entry_index
    """

    platform_id: str
    transit_waypoints: list[Waypoint]
    entry_index: int
    search_waypoints: list[Waypoint]

    @property
    def full_mission(self) -> list[Waypoint]:
        return self.transit_waypoints + self.search_waypoints

    @property
    def transit_distance_km(self) -> float:
        d = 0.0
        for i in range(1, len(self.transit_waypoints)):
            dx = self.transit_waypoints[i].x - self.transit_waypoints[i - 1].x
            dy = self.transit_waypoints[i].y - self.transit_waypoints[i - 1].y
            d += math.hypot(dx, dy)
        return d * 0.1  # scene units -> km at 100 m/unit

    def _cumulative_distances(self) -> list[float]:
        """Return cumulative scene-unit distances along full_mission."""
        wps = self.full_mission
        cum = [0.0]
        for i in range(1, len(wps)):
            dx = wps[i].x - wps[i - 1].x
            dy = wps[i].y - wps[i - 1].y
            dz = wps[i].z - wps[i - 1].z
            cum.append(cum[-1] + math.sqrt(dx * dx + dy * dy + dz * dz))
        return cum

    @property
    def total_distance_units(self) -> float:
        cum = self._cumulative_distances()
        return cum[-1] if cum else 0.0

    def interpolate(self, dist_units: float) -> tuple[float, float, float, float]:
        """Return (x, y, z, yaw_deg) at a given distance along the mission path.

        Clamps to path start/end.  Uses linear interpolation between waypoints.
        """
        wps = self.full_mission
        if not wps:
            return (0.0, 0.0, 0.0, 0.0)
        if dist_units <= 0.0:
            return (wps[0].x, wps[0].y, wps[0].z, wps[0].yaw_deg)

        cum = self._cumulative_distances()
        total = cum[-1]
        if dist_units >= total:
            return (wps[-1].x, wps[-1].y, wps[-1].z, wps[-1].yaw_deg)

        # Find the segment containing dist_units
        for i in range(1, len(cum)):
            if cum[i] >= dist_units:
                seg_start = cum[i - 1]
                seg_len = cum[i] - seg_start
                t = (dist_units - seg_start) / seg_len if seg_len > 0 else 0.0
                t = max(0.0, min(1.0, t))
                wp_a, wp_b = wps[i - 1], wps[i]
                x = wp_a.x + t * (wp_b.x - wp_a.x)
                y = wp_a.y + t * (wp_b.y - wp_a.y)
                z = wp_a.z + t * (wp_b.z - wp_a.z)
                yaw = wp_a.yaw_deg + t * _angle_delta(wp_a.yaw_deg, wp_b.yaw_deg)
                return (x, y, z, yaw)
        return (wps[-1].x, wps[-1].y, wps[-1].z, wps[-1].yaw_deg)


def _angle_delta(a: float, b: float) -> float:
    """Signed shortest angular distance b - a in degrees."""
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def find_best_entry(
    waypoints: list[Waypoint],
    start_x: float,
    start_y: float,
) -> int:
    """Return the index of the waypoint closest to (start_x, start_y)."""
    best_idx = 0
    best_d2 = float("inf")
    for i, wp in enumerate(waypoints):
        d2 = (wp.x - start_x) ** 2 + (wp.y - start_y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx


def plan_mission(
    result: PlannerResult,
    start_x: float,
    start_y: float,
    start_z: float,
    platform_id: str = "Blue_Quad_Recon_1",
) -> MissionPlan:
    """Plan a complete mission from drone start to search path.

    Uses A* obstacle avoidance on the transit segment so the drone
    navigates around mountains between its start position and the
    search-area entry point.

    Args:
        result: PlannerResult from plan().
        start_x, start_y, start_z: Drone starting position in scene units.
        platform_id: Human-readable ID for the platform.

    Returns:
        MissionPlan with transit, entry index, and reordered search waypoints.
    """
    from .obstacles import build_all_mountains, filter_mountains_in_area
    from .terrain import TerrainGrid
    from .astar_planner import astar_avoid

    cfg = result.config
    mpu = cfg.meters_per_unit
    map_size_units = cfg.map_size_km * 1000.0 / mpu
    visual_height_units = (
        cfg.mountain_height_m * cfg.terrain_vertical_exaggeration / mpu
    )

    entry_idx = find_best_entry(result.waypoints, start_x, start_y)
    entry_wp = result.waypoints[entry_idx]

    # ── Build transit-corridor area and terrain ───────────────
    margin_units = 3.0 * 1000.0 / mpu  # 3 km margin around corridor
    x_min = min(start_x, entry_wp.x) - margin_units
    x_max = max(start_x, entry_wp.x) + margin_units
    y_min = min(start_y, entry_wp.y) - margin_units
    y_max = max(start_y, entry_wp.y) + margin_units

    corridor = SearchArea(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    all_mountains = build_all_mountains(map_size_units, visual_height_units, mpu)
    corridor_mountains = filter_mountains_in_area(all_mountains, corridor)

    terrain = TerrainGrid(
        corridor, corridor_mountains,
        resolution_m=cfg.astar_grid_resolution_m,
        safety_xy_m=cfg.obstacle_safety_xy_m,
        map_size_units=map_size_units,
        visual_height_units=visual_height_units,
        meters_per_unit=mpu,
    )

    # ── A* transit path ──────────────────────────────────────
    detour = astar_avoid(start_x, start_y, entry_wp.x, entry_wp.y, terrain)

    alt_units = cfg.altitude_agl_m / mpu
    safety_z_units = cfg.obstacle_safety_z_m / mpu

    if detour:
        # Build transit waypoints from A* result
        transit_xy = [(start_x, start_y)] + detour + [(entry_wp.x, entry_wp.y)]
        transit = []
        for i, (x, y) in enumerate(transit_xy):
            tz = terrain.height_at(x, y)
            z = tz + alt_units
            for m in corridor_mountains:
                from .obstacles import check_collision
                if check_collision(x, y, z, m, cfg.obstacle_safety_xy_m / mpu):
                    z = max(z, m.height_units + safety_z_units)
            wp = Waypoint(x=float(x), y=float(y), z=float(z), terrain_z=float(tz), yaw_deg=0.0)
            transit.append(wp)
    else:
        transit = [
            Waypoint(x=start_x, y=start_y, z=start_z, terrain_z=0.0, yaw_deg=0.0),
            Waypoint(
                x=entry_wp.x, y=entry_wp.y, z=entry_wp.z,
                terrain_z=entry_wp.terrain_z, yaw_deg=0.0,
            ),
        ]

    # Compute transit headings
    for i in range(len(transit) - 1):
        dx = transit[i + 1].x - transit[i].x
        dy = transit[i + 1].y - transit[i].y
        transit[i].yaw_deg = math.degrees(math.atan2(dx, dy))

    reordered = result.waypoints[entry_idx:] + result.waypoints[:entry_idx]

    return MissionPlan(
        platform_id=platform_id,
        transit_waypoints=transit,
        entry_index=entry_idx,
        search_waypoints=reordered,
    )


def plan_mission_for_platform(
    result: PlannerResult,
    platform_id: str = "Blue_Quad_Recon_1",
    meters_per_unit: float = 100.0,
) -> MissionPlan:
    """Convenience wrapper: plan a mission with start near the search area."""
    sx, sy, sz = compute_start_near_area(
        result.search_area,
        offset_km=15.0,
        altitude_agl_m=result.config.altitude_agl_m,
        meters_per_unit=meters_per_unit,
    )
    return plan_mission(result, sx, sy, sz, platform_id)


def export_mission_usd(
    stage,
    mission: MissionPlan,
    base_path: str = "/World/AirCombat/SAR_Search",
) -> None:
    """Export mission waypoints to USD for visualization.

    Transit path: yellow/amber spheres + polyline.
    Search path:  blue spheres + polyline.
    Entry waypoint: white sphere (larger).
    """
    from pxr import Gf, Sdf, UsdGeom

    from .visualize import export_waypoints_usd, _ensure_path

    export_waypoints_usd(
        stage, mission.transit_waypoints,
        base_path=f"{base_path}/Mission/Transit",
        path_color=(1.0, 0.85, 0.05),
    )

    export_waypoints_usd(
        stage, mission.search_waypoints,
        base_path=f"{base_path}/Mission/Search",
        path_color=(0.0, 0.55, 1.0),
    )

    if mission.search_waypoints:
        wp = mission.search_waypoints[0]
        _ensure_path(stage, f"{base_path}/Mission")
        sphere = UsdGeom.Sphere.Define(
            stage, Sdf.Path(f"{base_path}/Mission/EntryMarker")
        )
        sphere.CreateRadiusAttr(4.0)
        UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(float(wp.x), float(wp.y), float(wp.z))
        )
        UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr(
            [Gf.Vec3f(1.0, 1.0, 1.0)]
        )


# ──────────────────────────────────────────────────────────────
#  Multi-region cycle helpers
# ──────────────────────────────────────────────────────────────


def compute_best_exit(
    waypoints: list[Waypoint],
    target_cx: float,
    target_cy: float,
) -> int:
    """Return index of the waypoint closest to (target_cx, target_cy)."""
    best = 0
    best_d2 = float("inf")
    for i, wp in enumerate(waypoints):
        d2 = (wp.x - target_cx) ** 2 + (wp.y - target_cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = i
    return best


def compute_tangent_entry(
    waypoints: list[Waypoint],
    approach_dx: float,
    approach_dy: float,
) -> int:
    """Return the waypoint index whose outgoing tangent best aligns
    with the approach direction.

    approach_dx, approach_dy : direction the drone is coming FROM
    (exit of previous region toward this region's centre).
    """
    n = len(waypoints)
    ad = math.sqrt(approach_dx * approach_dx + approach_dy * approach_dy)
    if ad < 1e-9:
        return 0

    best = 0
    best_dot = -float("inf")
    for i in range(n):
        j = (i + 1) % n
        tx = waypoints[j].x - waypoints[i].x
        ty = waypoints[j].y - waypoints[i].y
        tn = math.sqrt(tx * tx + ty * ty)
        if tn < 1e-9:
            continue
        dot = (tx * approach_dx + ty * approach_dy) / (tn * ad)
        if dot > best_dot:
            best_dot = dot
            best = i
    return best


def _build_transit(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    config: PlannerResult.config.__class__,  # PlannerConfig
    mpu: float,
    map_size_units: float,
    visual_height_units: float,
) -> list[Waypoint]:
    """A*-based transit from (x1,y1,z1) to (x2,y2,z2)."""
    from .area import SearchArea
    from .obstacles import build_all_mountains, filter_mountains_in_area, check_collision
    from .terrain import TerrainGrid
    from .astar_planner import astar_avoid

    margin_units = 3.0 * 1000.0 / mpu
    corridor = SearchArea(
        x_min=min(x1, x2) - margin_units, x_max=max(x1, x2) + margin_units,
        y_min=min(y1, y2) - margin_units, y_max=max(y1, y2) + margin_units,
    )
    all_m = build_all_mountains(map_size_units, visual_height_units, mpu)
    m_in_corridor = filter_mountains_in_area(all_m, corridor)
    terrain = TerrainGrid(
        corridor, m_in_corridor,
        resolution_m=config.astar_grid_resolution_m,
        safety_xy_m=config.obstacle_safety_xy_m,
        map_size_units=map_size_units,
        visual_height_units=visual_height_units,
        meters_per_unit=mpu,
    )
    detour = astar_avoid(x1, y1, x2, y2, terrain)
    alt_units = config.altitude_agl_m / mpu
    safety_z = config.obstacle_safety_z_m / mpu

    if detour:
        xy_seq = [(x1, y1)] + detour + [(x2, y2)]
    else:
        xy_seq = [(x1, y1), (x2, y2)]

    result: list[Waypoint] = []
    for x, y in xy_seq:
        tz = terrain.height_at(x, y)
        z = tz + alt_units
        for m in m_in_corridor:
            if check_collision(x, y, z, m, config.obstacle_safety_xy_m / mpu):
                z = max(z, m.height_units + safety_z)
        result.append(Waypoint(x=x, y=y, z=z, terrain_z=tz, yaw_deg=0.0))
    return result


def _segment_waypoints(wps: list[Waypoint], start_idx: int, end_idx: int) -> list[Waypoint]:
    """Return waypoints going forward from start_idx to end_idx (circular).

    When start_idx == end_idx this returns one full lap (closed loop)."""
    n = len(wps)
    if start_idx == end_idx:
        # Full loop: start→end (includes close point)
        return wps[start_idx:] + wps[:start_idx + 1]
    if end_idx > start_idx:
        return wps[start_idx:end_idx + 1]
    else:
        return wps[start_idx:] + wps[:end_idx + 1]


def build_multi_region_cycle(
    results: dict[str, PlannerResult],
    cycle_order: list[str],
    region_centers: dict[str, tuple[float, float]],
) -> list[Waypoint]:
    """Build a single continuous loop covering multiple regions in cycle order.

    Exit:  waypoint closest to the next region's centre.
    Entry: waypoint closest to the previous region's exit.
    This minimises transit distance between regions.

    Args:
        results:        dict region_id → PlannerResult
        cycle_order:    list of region_ids in visit order, e.g. ["A","B"]
        region_centers: dict region_id → (cx_scene_units, cy_scene_units)

    Returns:
        One flat list of Waypoints forming a closed cycle path.
    """
    if len(cycle_order) < 2:
        rid = cycle_order[0]
        return list(results[rid].waypoints)

    cfg = results[cycle_order[0]].config
    mpu = cfg.meters_per_unit
    map_size_units = cfg.map_size_km * 1000.0 / mpu
    visual_height_units = cfg.mountain_height_m * cfg.terrain_vertical_exaggeration / mpu

    num = len(cycle_order)
    entries: list[int] = [0] * num
    exits: list[int] = [0] * num

    # Compute cumulative distances for each region's loop
    cum_dists: list[list[float]] = []
    for i in range(num):
        wps = results[cycle_order[i]].waypoints
        n_w = len(wps)
        cum = [0.0]
        for k in range(n_w):
            nk = (k + 1) % n_w
            cum.append(cum[-1] + math.hypot(
                wps[nk].x - wps[k].x, wps[nk].y - wps[k].y))
        cum_dists.append(cum)

    # Entries: closest waypoint to previous exit (shortest transit in)
    for i in range(num):
        cur = cycle_order[i]
        prev = cycle_order[(i - 1) % num]
        wps = results[cur].waypoints
        prev_wps = results[prev].waypoints
        # For the first iteration, exits aren't computed yet — use previous
        # region's center as a proxy for the previous exit position
        pcy = region_centers[prev]
        px, py = pcy[0], pcy[1]
        entries[i] = find_best_entry(wps, px, py)

    # Exits: must be ≥ 70% of the loop ahead of entry.
    # In the valid range [entry + 70%loop, entry + 100%loop), pick the
    # waypoint closest to the next region's centre (shortest transit out).
    for i in range(num):
        cur = cycle_order[i]
        nxt = cycle_order[(i + 1) % num]
        wps = results[cur].waypoints
        ncx, ncy = region_centers[nxt]
        cum = cum_dists[i]
        total_loop = cum[-1]
        s_dist = cum[entries[i]]

        min_fwd = total_loop * 0.7
        best = -1
        best_d = float("inf")
        for k in range(len(wps)):
            # Forward distance from entry to waypoint k
            if cum[k] >= s_dist:
                fwd = cum[k] - s_dist
            else:
                fwd = total_loop - (s_dist - cum[k])
            if fwd >= min_fwd:
                d = math.hypot(wps[k].x - ncx, wps[k].y - ncy)
                if d < best_d:
                    best_d = d
                    best = k
        exits[i] = best if best >= 0 else compute_best_exit(wps, ncx, ncy)

    # Recompute entries now that exits are known (use actual exit positions)
    for i in range(num):
        cur = cycle_order[i]
        prev = cycle_order[(i - 1) % num]
        wps = results[cur].waypoints
        prev_wps = results[prev].waypoints
        prev_exit_idx = exits[(i - 1) % num]
        px = prev_wps[prev_exit_idx].x
        py = prev_wps[prev_exit_idx].y
        entries[i] = find_best_entry(wps, px, py)

    # Verify: after entry recomputation, exits must still be >= 70% away.
    # Iterate entry<->exit once more in case exits changed.
    for _ in range(2):
        changed = False
        for i in range(num):
            cur = cycle_order[i]
            nxt = cycle_order[(i + 1) % num]
            wps = results[cur].waypoints
            cum = cum_dists[i]
            total = cum[-1]
            s_dist = cum[entries[i]]
            e_dist = cum[exits[i]]
            if s_dist <= e_dist:
                fwd = e_dist - s_dist
            else:
                fwd = total - (s_dist - e_dist)
            if fwd < total * 0.7:
                ncx, ncy = region_centers[nxt]
                min_fwd = total * 0.7
                best_e = -1
                best_d = float("inf")
                for k in range(len(wps)):
                    if cum[k] >= s_dist:
                        fk = cum[k] - s_dist
                    else:
                        fk = total - (s_dist - cum[k])
                    if fk >= min_fwd:
                        d = math.hypot(wps[k].x - ncx, wps[k].y - ncy)
                        if d < best_d:
                            best_d = d
                            best_e = k
                if best_e >= 0:
                    exits[i] = best_e
                    changed = True
        if not changed:
            break
        for i in range(num):
            cur = cycle_order[i]
            prev = cycle_order[(i - 1) % num]
            wps = results[cur].waypoints
            prev_wps = results[prev].waypoints
            px = prev_wps[exits[(i - 1) % num]].x
            py = prev_wps[exits[(i - 1) % num]].y
            entries[i] = find_best_entry(wps, px, py)

    # ── Build the cycle path ──
    combined: list[Waypoint] = []

    for i in range(num):
        cur = cycle_order[i]
        nxt = cycle_order[(i + 1) % num]
        wps = results[cur].waypoints
        nxt_wps = results[nxt].waypoints

        # Fly from tangent-aligned entry to tangent-aligned exit
        seg = _segment_waypoints(wps, entries[i], exits[i])
        combined.extend(seg)

        # Transit: exit -> next region's entry
        exit_wp = wps[exits[i]]
        entry_wp = nxt_wps[entries[(i + 1) % num]]
        transit = _build_transit(
            exit_wp.x, exit_wp.y, exit_wp.z,
            entry_wp.x, entry_wp.y, entry_wp.z,
            cfg, mpu, map_size_units, visual_height_units,
        )
        combined.extend(transit[1:])

    # Compute headings
    for i in range(len(combined) - 1):
        dx = combined[i + 1].x - combined[i].x
        dy = combined[i + 1].y - combined[i].y
        combined[i].yaw_deg = math.degrees(math.atan2(dx, dy))

    return combined


# ──────────────────────────────────────────────────────────────
#  Public API — called by task-allocation module
# ──────────────────────────────────────────────────────────────


@dataclass
class SearchMissionPlan:
    """Output of plan_search_mission() for one platform."""

    platform_id: str
    waypoints: list[Waypoint]
    """Combined flight path (one or more region laps + transit segments)."""

    total_km: float
    """Total path length in km."""

    region_waypoints: list[list[Waypoint]]
    """Per-region closed-loop waypoints (for overlay display)."""

    search_areas: list[SearchArea]
    """Search-area objects (for drawing boundaries)."""


def plan_search_mission(
    assignments: list[dict],
) -> dict[str, SearchMissionPlan]:
    """Plan search paths for one or more platforms given task assignments.

    This is the primary entry point called by the task-allocation module.
    Each assignment specifies which platform covers which area(s) and the
    search pattern to use.  Platforms assigned to multiple areas get a
    cycle path that alternates between them (with A*-avoiding transits).

    Args:
        assignments: List of assignment dicts, each with::

            {
                "platform_id": "Blue_CH4_Recon",
                "center_km": (cx, cy),
                "width_km": 25,
                "height_km": 25,        # optional, defaults to width_km
                "pattern": "racetrack",  # racetrack|sar_polygon|sar_rounded|figure_eight
            }

    Returns:
        Dict mapping platform_id → SearchMissionPlan.
    """
    from .config import PlannerConfig
    from .planner import plan as run_plan

    # ── Step 1: plan each individual assignment ───────────────
    # Group assignments by platform
    platform_assignments: dict[str, list[dict]] = {}
    for a in assignments:
        pid = a["platform_id"]
        platform_assignments.setdefault(pid, []).append(a)

    results: dict[str, SearchMissionPlan] = {}

    for pid, area_list in platform_assignments.items():
        if len(area_list) == 1:
            # Single area — no cycle needed
            a = area_list[0]
            h = a.get("height_km", a["width_km"])
            cfg = PlannerConfig(
                area_center_km=tuple(a["center_km"]),
                area_width_km=a["width_km"],
                area_height_km=h,
                pattern=a.get("pattern", "racetrack"),
            )
            result = run_plan(cfg)
            wps = list(result.waypoints)
            results[pid] = SearchMissionPlan(
                platform_id=pid,
                waypoints=wps,
                total_km=result.stats["path_length_km"],
                region_waypoints=[wps],
                search_areas=[result.search_area],
            )
        else:
            # Multiple areas — build cycle path
            region_results: dict[str, PlannerResult] = {}
            region_centers: dict[str, tuple[float, float]] = {}
            mpu = 100.0
            scale = 1000.0 / mpu

            for i, a in enumerate(area_list):
                rid = f"r{i}"
                h = a.get("height_km", a["width_km"])
                cx, cy = a["center_km"]
                cfg = PlannerConfig(
                    area_center_km=(cx, cy),
                    area_width_km=a["width_km"],
                    area_height_km=h,
                    pattern=a.get("pattern", "racetrack"),
                )
                region_results[rid] = run_plan(cfg)
                region_centers[rid] = (cx * scale, cy * scale)

            rid_list = list(region_results.keys())
            cycle_wps = build_multi_region_cycle(
                region_results, rid_list, region_centers,
            )

            # Compute total km
            total = 0.0
            for j in range(1, len(cycle_wps)):
                dx = cycle_wps[j].x - cycle_wps[j - 1].x
                dy = cycle_wps[j].y - cycle_wps[j - 1].y
                total += math.hypot(dx, dy)
            total_km = total * region_results[rid_list[0]].config.meters_per_unit / 1000.0

            results[pid] = SearchMissionPlan(
                platform_id=pid,
                waypoints=cycle_wps,
                total_km=round(total_km, 1),
                region_waypoints=[list(r.waypoints) for r in region_results.values()],
                search_areas=[r.search_area for r in region_results.values()],
            )

    return results
