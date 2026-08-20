"""Simple 3D SAR phased multi-drone partition visualization.

This viewer does not depend on Isaac Sim. It uses the existing SAR planner
and draws a lightweight 3D coordinate scene with:
  - partitioned search areas
  - planned search routes
  - animated drone markers
  - sensor field-of-view cones and ground SAR footprints
  - phase switching for multi-stage missions

Run from the project root:
    python sar_search_planner/run_multi_drone_simple_viz.py

Useful options:
    python sar_search_planner/run_multi_drone_simple_viz.py --no-show
    python sar_search_planner/run_multi_drone_simple_viz.py --save sar_search_planner/exports/phased_multidrone.gif
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np

from sar_search_planner.config import PlannerConfig
from sar_search_planner.mission import build_multi_region_cycle
from sar_search_planner.planner import PlannerResult, Waypoint, plan


DRONE_COLORS = [
    (0.04, 0.47, 0.86),
    (0.95, 0.48, 0.10),
    (0.09, 0.62, 0.30),
    (0.62, 0.23, 0.82),
    (0.86, 0.22, 0.32),
    (0.10, 0.58, 0.62),
]

DEFAULT_PHASES = [
    {
        "phase_label": "Phase 1 - four drones search one area each",
        "phase_laps": 3,
        "quadrants": [
            {
                "_platform_id": "UAV_NW",
                "area_center_km": (47.5, 42.5),
                "area_width_km": 25,
                "area_height_km": 25,
                "pattern": "racetrack",
                "altitude_agl_m": 5000,
            },
            {
                "_platform_id": "UAV_NE",
                "area_center_km": (72.5, 42.5),
                "area_width_km": 25,
                "area_height_km": 25,
                "pattern": "sar_polygon",
                "altitude_agl_m": 5000,
            },
            {
                "_platform_id": "UAV_SW",
                "area_center_km": (47.5, 17.5),
                "area_width_km": 25,
                "area_height_km": 25,
                "pattern": "sar_rounded",
                "altitude_agl_m": 5000,
            },
            {
                "_platform_id": "UAV_SE",
                "area_center_km": (72.5, 17.5),
                "area_width_km": 25,
                "area_height_km": 25,
                "pattern": "figure_eight",
                "altitude_agl_m": 5000,
            },
        ],
    },
    {
        "phase_label": "Phase 2 - reassigned search after one drone leaves",
        "phase_laps": -1,
        "quadrants": [
            {
                "_platform_id": "UAV_NW",
                "area_center_km": (47.5, 42.5),
                "area_width_km": 25,
                "area_height_km": 25,
                "pattern": "racetrack",
                "altitude_agl_m": 5000,
            },
            {
                "_platform_id": "UAV_NE",
                "area_center_km": (72.5, 42.5),
                "area_width_km": 25,
                "area_height_km": 25,
                "pattern": "racetrack",
                "altitude_agl_m": 5000,
            },
            {
                "_platform_id": "UAV_SW",
                "area_center_km": (47.5, 17.5),
                "area_width_km": 25,
                "area_height_km": 25,
                "pattern": "figure_eight",
                "altitude_agl_m": 5000,
            },
            {
                "_platform_id": "UAV_SW",
                "area_center_km": (72.5, 17.5),
                "area_width_km": 25,
                "area_height_km": 25,
                "pattern": "figure_eight",
                "altitude_agl_m": 5000,
            },
        ],
    },
]


@dataclass
class PhaseInput:
    label: str
    laps: int
    configs: list[dict]


@dataclass
class DroneTrack:
    platform_id: str
    pattern: str
    color: tuple[float, float, float]
    waypoints: list[Waypoint]
    area: object | None
    meters_per_unit: float
    path_km: float
    total_units: float
    interpolate: Callable[[float], tuple[float, float, float, float]]


@dataclass
class ScenePhase:
    label: str
    laps: int
    regions: list[DroneTrack]
    drones: list[DroneTrack]


def _load_phase_inputs() -> list[PhaseInput]:
    """Load phased configs; fall back to a default two-phase mission."""
    config_path = Path(__file__).with_name("_active_config.json")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = DEFAULT_PHASES

    if isinstance(data, list):
        if data and all(isinstance(item, dict) and "quadrants" in item for item in data):
            return [
                PhaseInput(
                    label=str(item.get("phase_label", f"Phase {index + 1}")),
                    laps=int(item.get("phase_laps", -1)),
                    configs=[dict(q) for q in item.get("quadrants", [])],
                )
                for index, item in enumerate(data)
            ]
        if data and all(isinstance(item, dict) and "_platform_id" in item for item in data):
            return [
                PhaseInput(
                    label=" ",
                    laps=-1,
                    configs=[dict(q) for q in data],
                )
            ]

    return [
        PhaseInput(
            label=str(item["phase_label"]),
            laps=int(item["phase_laps"]),
            configs=[dict(q) for q in item["quadrants"]],
        )
        for item in DEFAULT_PHASES
    ]


def _platform_colors(phase_inputs: list[PhaseInput]) -> dict[str, tuple[float, float, float]]:
    platform_ids: list[str] = []
    for phase in phase_inputs:
        for cfg in phase.configs:
            pid = str(cfg.get("_platform_id", f"UAV_{len(platform_ids) + 1}"))
            if pid not in platform_ids:
                platform_ids.append(pid)
    return {
        pid: DRONE_COLORS[index % len(DRONE_COLORS)]
        for index, pid in enumerate(platform_ids)
    }


def _build_path_interpolator(
    waypoints: list[Waypoint],
) -> tuple[float, Callable[[float], tuple[float, float, float, float]]]:
    if not waypoints:
        def _empty(_dist: float) -> tuple[float, float, float, float]:
            return 0.0, 0.0, 0.0, 0.0
        return 0.0, _empty

    cleaned = [waypoints[0]]
    for wp in waypoints[1:]:
        prev = cleaned[-1]
        if math.hypot(wp.x - prev.x, wp.y - prev.y) > 0.01:
            cleaned.append(wp)
    waypoints = cleaned

    if len(waypoints) == 1:
        wp = waypoints[0]
        def _single(_dist: float) -> tuple[float, float, float, float]:
            return wp.x, wp.y, wp.z, wp.yaw_deg
        return 0.0, _single

    seg_lengths = []
    for i, wp in enumerate(waypoints):
        nxt = waypoints[(i + 1) % len(waypoints)]
        dx = nxt.x - wp.x
        dy = nxt.y - wp.y
        dz = nxt.z - wp.z
        seg_lengths.append(math.sqrt(dx * dx + dy * dy + dz * dz))

    cumulative = [0.0]
    for length in seg_lengths:
        cumulative.append(cumulative[-1] + length)
    total = cumulative[-1]

    def _interpolate(dist: float) -> tuple[float, float, float, float]:
        if total <= 1e-9:
            wp = waypoints[0]
            return wp.x, wp.y, wp.z, wp.yaw_deg

        d = dist % total
        for i in range(len(waypoints)):
            if cumulative[i + 1] >= d:
                seg_len = cumulative[i + 1] - cumulative[i]
                t = (d - cumulative[i]) / seg_len if seg_len > 1e-9 else 0.0
                a = waypoints[i]
                b = waypoints[(i + 1) % len(waypoints)]
                x = a.x + t * (b.x - a.x)
                y = a.y + t * (b.y - a.y)
                z = a.z + t * (b.z - a.z)
                yaw = math.degrees(math.atan2(b.x - a.x, b.y - a.y))
                return x, y, z, yaw
        wp = waypoints[-1]
        return wp.x, wp.y, wp.z, wp.yaw_deg

    return total, _interpolate


def _path_length_km(waypoints: list[Waypoint], meters_per_unit: float) -> float:
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i].x - waypoints[i - 1].x
        dy = waypoints[i].y - waypoints[i - 1].y
        dz = waypoints[i].z - waypoints[i - 1].z
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total * meters_per_unit / 1000.0


def _make_track(
    platform_id: str,
    pattern: str,
    color: tuple[float, float, float],
    waypoints: list[Waypoint],
    area: object | None,
    meters_per_unit: float,
    path_km: float,
) -> DroneTrack:
    total_units, interpolate = _build_path_interpolator(waypoints)
    return DroneTrack(
        platform_id=platform_id,
        pattern=pattern,
        color=color,
        waypoints=waypoints,
        area=area,
        meters_per_unit=meters_per_unit,
        path_km=path_km,
        total_units=total_units,
        interpolate=interpolate,
    )


def _build_scene_phases(
    phase_inputs: list[PhaseInput],
    colors: dict[str, tuple[float, float, float]],
) -> list[ScenePhase]:
    phases: list[ScenePhase] = []

    for phase_index, phase_input in enumerate(phase_inputs):
        print(
            f"[SAR_SIMPLE_3D] Phase {phase_index + 1}: "
            f"{phase_input.label} ({phase_input.laps} laps)"
        )
        regions: list[DroneTrack] = []
        grouped: dict[str, list[tuple[str, PlannerResult, DroneTrack]]] = defaultdict(list)

        for region_index, raw_cfg in enumerate(phase_input.configs):
            cfg_dict = dict(raw_cfg)
            platform_id = str(cfg_dict.pop("_platform_id", f"UAV_{region_index + 1}"))
            cfg = PlannerConfig(**cfg_dict)
            result = plan(cfg)
            color = colors[platform_id]
            path_km = float(result.stats.get("path_length_km", 0.0))
            region_track = _make_track(
                platform_id=platform_id,
                pattern=cfg.pattern,
                color=color,
                waypoints=list(result.waypoints),
                area=result.search_area,
                meters_per_unit=cfg.meters_per_unit,
                path_km=path_km,
            )
            regions.append(region_track)
            grouped[platform_id].append((f"{platform_id}_r{region_index}", result, region_track))
            print(
                f"[SAR_SIMPLE_3D]   {platform_id}: {cfg.pattern}, "
                f"{len(result.waypoints)} waypoints, {path_km:.1f} km"
            )

        drones: list[DroneTrack] = []
        for platform_id, items in grouped.items():
            if len(items) == 1:
                drones.append(items[0][2])
                continue

            results_map = {region_id: result for region_id, result, _track in items}
            cycle_order = list(results_map.keys())
            centers_map = {
                region_id: (result.search_area.center_x, result.search_area.center_y)
                for region_id, result, _track in items
            }
            cycle_wps = build_multi_region_cycle(results_map, cycle_order, centers_map)
            mpu = items[0][1].config.meters_per_unit
            path_km = _path_length_km(cycle_wps, mpu)
            pattern = f"{len(items)}-region cycle"
            drones.append(
                _make_track(
                    platform_id=platform_id,
                    pattern=pattern,
                    color=colors[platform_id],
                    waypoints=cycle_wps,
                    area=None,
                    meters_per_unit=mpu,
                    path_km=path_km,
                )
            )
            print(
                f"[SAR_SIMPLE_3D]   {platform_id}: reassigned to "
                f"{len(items)} regions, cycle {path_km:.1f} km"
            )

        phases.append(
            ScenePhase(
                label=phase_input.label,
                laps=phase_input.laps,
                regions=regions,
                drones=drones,
            )
        )

    return phases


def _to_km(value_units: float, meters_per_unit: float) -> float:
    return value_units * meters_per_unit / 1000.0


def _waypoint_arrays(track: DroneTrack) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mpu = track.meters_per_unit
    xs = np.array([_to_km(wp.x, mpu) for wp in track.waypoints])
    ys = np.array([_to_km(wp.y, mpu) for wp in track.waypoints])
    zs = np.array([_to_km(wp.z, mpu) for wp in track.waypoints])
    if len(xs) > 1:
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        zs = np.append(zs, zs[0])
    return xs, ys, zs


def _area_corners_km(track: DroneTrack) -> list[tuple[float, float, float]]:
    if track.area is None:
        return []
    mpu = track.meters_per_unit
    area = track.area
    return [
        (_to_km(area.x_min, mpu), _to_km(area.y_min, mpu), 0.0),
        (_to_km(area.x_max, mpu), _to_km(area.y_min, mpu), 0.0),
        (_to_km(area.x_max, mpu), _to_km(area.y_max, mpu), 0.0),
        (_to_km(area.x_min, mpu), _to_km(area.y_max, mpu), 0.0),
    ]


def _sensor_polygons(
    x_km: float,
    y_km: float,
    z_km: float,
    yaw_deg: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    yaw = math.radians(yaw_deg)
    fwd_x = math.sin(yaw)
    fwd_y = math.cos(yaw)
    right_x = math.cos(yaw)
    right_y = -math.sin(yaw)

    eo_range_km = 8.0
    eo_half_width_km = 3.6
    center_x = x_km + fwd_x * eo_range_km
    center_y = y_km + fwd_y * eo_range_km
    eo = [
        (x_km, y_km, z_km),
        (center_x + right_x * eo_half_width_km, center_y + right_y * eo_half_width_km, 0.0),
        (center_x - right_x * eo_half_width_km, center_y - right_y * eo_half_width_km, 0.0),
    ]

    sar_forward_km = float(os.environ.get("SAR_SIMPLE_SWATH_FORWARD_KM", "5.5"))
    sar_half_width_km = float(os.environ.get("SAR_SIMPLE_SWATH_HALF_WIDTH_KM", "10"))
    a_x = x_km - fwd_x * sar_forward_km * 0.35
    a_y = y_km - fwd_y * sar_forward_km * 0.35
    b_x = x_km + fwd_x * sar_forward_km
    b_y = y_km + fwd_y * sar_forward_km
    sar = [
        (a_x + right_x * sar_half_width_km, a_y + right_y * sar_half_width_km, 0.02),
        (a_x - right_x * sar_half_width_km, a_y - right_y * sar_half_width_km, 0.02),
        (b_x - right_x * sar_half_width_km, b_y - right_y * sar_half_width_km, 0.02),
        (b_x + right_x * sar_half_width_km, b_y + right_y * sar_half_width_km, 0.02),
    ]
    return eo, sar


def _all_tracks(phases: list[ScenePhase]) -> list[DroneTrack]:
    tracks: list[DroneTrack] = []
    for phase in phases:
        tracks.extend(phase.regions)
        tracks.extend(phase.drones)
    return tracks


def _track_loop_km(track: DroneTrack) -> float:
    return track.total_units * track.meters_per_unit / 1000.0


def _phase_lap_count(phase: ScenePhase, indefinite_laps: int) -> int:
    return max(1, phase.laps if phase.laps > 0 else indefinite_laps)


def _phase_reference_loop_km(phase: ScenePhase) -> float:
    if not phase.drones:
        return 1.0
    return max(1e-6, max(_track_loop_km(track) for track in phase.drones))


def _phase_travel_km(phase: ScenePhase, indefinite_laps: int) -> float:
    return _phase_reference_loop_km(phase) * _phase_lap_count(phase, indefinite_laps)


def _set_axes_equalish(ax, tracks: list[DroneTrack]) -> None:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = [0.0]

    for track in tracks:
        wx, wy, wz = _waypoint_arrays(track)
        xs.extend(wx.tolist())
        ys.extend(wy.tolist())
        zs.extend(wz.tolist())
        for x, y, _z in _area_corners_km(track):
            xs.append(x)
            ys.append(y)

    margin = 5.0
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin
    z_max = max(max(zs) + 2.0, 8.0)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(0.0, z_max)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, max(8.0, z_max * 2.0)))


def _style_axes(ax) -> None:
    ax.set_xlabel("East / km")
    ax.set_ylabel("North / km")
    ax.set_zlabel("Altitude / km")
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=32, azim=-58)


def _draw_static_scene(ax, phase: ScenePhase) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for track in phase.regions:
        corners = _area_corners_km(track)
        if not corners:
            continue

        area_poly = Poly3DCollection(
            [corners],
            facecolors=[(*track.color, 0.08)],
            edgecolors=[(*track.color, 0.85)],
            linewidths=1.6,
        )
        ax.add_collection3d(area_poly)

        closed = corners + [corners[0]]
        ax.plot(
            [p[0] for p in closed],
            [p[1] for p in closed],
            [0.0 for _ in closed],
            color=track.color,
            linewidth=1.8,
        )

        xs, ys, zs = _waypoint_arrays(track)
        ax.plot(xs, ys, zs, color=track.color, linewidth=1.8, alpha=0.78)
        ax.scatter(xs[0], ys[0], zs[0], color=track.color, s=28, marker="o")

        cx = sum(p[0] for p in corners) / 4.0
        cy = sum(p[1] for p in corners) / 4.0
        ax.text(
            cx,
            cy,
            0.25,
            f"{track.platform_id}\n{track.pattern}",
            color=track.color,
            ha="center",
            va="center",
            fontsize=8,
        )

    for track in phase.drones:
        if track.area is not None:
            continue
        xs, ys, zs = _waypoint_arrays(track)
        ax.plot(xs, ys, zs, color=track.color, linewidth=2.4, alpha=0.90, linestyle="--")


def _phase_frame_ranges(
    phases: list[ScenePhase],
    total_frames: int,
    indefinite_laps: int,
) -> list[tuple[int, int]]:
    weights = [max(1e-6, _phase_travel_km(phase, indefinite_laps)) for phase in phases]
    total_weight = sum(weights)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            end = total_frames
        else:
            end = cursor + max(2, round(total_frames * weight / total_weight))
        ranges.append((cursor, end))
        cursor = end
    return ranges


def _phase_for_frame(
    frame: int,
    phases: list[ScenePhase],
    ranges: list[tuple[int, int]],
    total_frames: int,
    indefinite_laps: int,
) -> tuple[int, float, float, int, int]:
    frame = frame % total_frames
    for index, (start, end) in enumerate(ranges):
        if start <= frame < end:
            duration = max(1, end - start)
            local = (frame - start) / duration
            laps = _phase_lap_count(phases[index], indefinite_laps)
            reference_loop_km = _phase_reference_loop_km(phases[index])
            phase_distance_km = local * reference_loop_km * laps
            lap_float = phase_distance_km / reference_loop_km
            path_progress = lap_float % 1.0
            lap_number = min(max(1, int(lap_float) + 1), max(1, laps))
            return index, local, phase_distance_km, lap_number, max(1, laps)
    return len(phases) - 1, 0.0, 0.0, 1, 1


def _make_animation(phases: list[ScenePhase], args: argparse.Namespace):
    if args.no_show or args.save:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    frame_ranges = _phase_frame_ranges(phases, args.frames, args.indefinite_laps)
    all_tracks = _all_tracks(phases)

    current_phase_idx = -1
    drone_artists = []
    heading_artists = []
    trail_artists = []
    trails: list[list[tuple[float, float, float]]] = []
    dynamic_collections = []
    info = None

    def setup_phase(phase_idx: int) -> None:
        nonlocal current_phase_idx, drone_artists, heading_artists
        nonlocal trail_artists, trails, dynamic_collections, info

        current_phase_idx = phase_idx
        dynamic_collections = []
        phase = phases[phase_idx]

        ax.clear()
        _style_axes(ax)
        _draw_static_scene(ax, phase)
        _set_axes_equalish(ax, all_tracks)

        drone_artists = []
        heading_artists = []
        trail_artists = []
        trails = [[] for _ in phase.drones]

        for track in phase.drones:
            (marker,) = ax.plot([], [], [], marker="^", markersize=10, color=track.color)
            (heading,) = ax.plot([], [], [], color=track.color, linewidth=2.5)
            (trail,) = ax.plot([], [], [], color=track.color, linewidth=3.0, alpha=0.18)
            drone_artists.append(marker)
            heading_artists.append(heading)
            trail_artists.append(trail)

        info = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=9)

    def update(frame: int):
        nonlocal dynamic_collections

        phase_idx, local, phase_distance_km, lap_number, lap_count = _phase_for_frame(
            frame, phases, frame_ranges, args.frames, args.indefinite_laps,
        )
        if phase_idx != current_phase_idx:
            setup_phase(phase_idx)

        for collection in dynamic_collections:
            collection.remove()
        dynamic_collections = []

        phase = phases[phase_idx]
        lines = [
            f"phase: {phase_idx + 1}/{len(phases)}",
            f"reference lap: {lap_number}/{lap_count}",
            f"phase progress: {local * 100:5.1f}%",
        ]

        for i, track in enumerate(phase.drones):
            dist = phase_distance_km * 1000.0 / track.meters_per_unit
            x_u, y_u, z_u, yaw = track.interpolate(dist)
            x = _to_km(x_u, track.meters_per_unit)
            y = _to_km(y_u, track.meters_per_unit)
            z = _to_km(z_u, track.meters_per_unit)

            drone_artists[i].set_data([x], [y])
            drone_artists[i].set_3d_properties([z])

            yaw_rad = math.radians(yaw)
            nose_len = 2.2
            hx = x + math.sin(yaw_rad) * nose_len
            hy = y + math.cos(yaw_rad) * nose_len
            heading_artists[i].set_data([x, hx], [y, hy])
            heading_artists[i].set_3d_properties([z, z])

            trails[i].append((x, y, 0.04))
            if len(trails[i]) > 42:
                trails[i].pop(0)
            trail_artists[i].set_data([p[0] for p in trails[i]], [p[1] for p in trails[i]])
            trail_artists[i].set_3d_properties([p[2] for p in trails[i]])

            eo, sar = _sensor_polygons(x, y, z, yaw)
            eo_poly = Poly3DCollection(
                [eo],
                facecolors=[(1.0, 0.12, 0.08, 0.17)],
                edgecolors=[(1.0, 0.12, 0.08, 0.55)],
                linewidths=1.0,
            )
            sar_poly = Poly3DCollection(
                [sar],
                facecolors=[(1.0, 0.86, 0.08, 0.23)],
                edgecolors=[(1.0, 0.72, 0.05, 0.65)],
                linewidths=1.0,
            )
            ax.add_collection3d(eo_poly)
            ax.add_collection3d(sar_poly)
            dynamic_collections.extend([eo_poly, sar_poly])

            loop_progress = ((dist % max(1e-6, track.total_units)) / max(1e-6, track.total_units)) * 100.0
            lines.append(
                f"{track.platform_id}: {track.pattern}, "
                f"{_track_loop_km(track):.1f} km loop, {loop_progress:4.0f}%"
            )

        if info is not None:
            info.set_text("\n".join(lines))

        return [
            *drone_artists,
            *heading_artists,
            *trail_artists,
            *(dynamic_collections or []),
        ]

    setup_phase(0)
    anim = FuncAnimation(
        fig,
        update,
        frames=args.frames,
        interval=args.interval,
        blit=False,
        repeat=True,
    )
    update(0)
    fig.tight_layout()
    return fig, anim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=int(os.environ.get("SAR_SIMPLE_FRAMES", "720")))
    parser.add_argument("--interval", type=int, default=int(os.environ.get("SAR_SIMPLE_INTERVAL_MS", "35")))
    parser.add_argument("--indefinite-laps", type=int, default=int(os.environ.get("SAR_SIMPLE_INDEFINITE_LAPS", "3")))
    parser.add_argument("--save", type=str, default=os.environ.get("SAR_SIMPLE_SAVE", ""))
    parser.add_argument("--no-show", action="store_true", help="Build one frame and exit.")
    args = parser.parse_args()
    args.frames = max(2, args.frames)
    args.indefinite_laps = max(1, args.indefinite_laps)

    phase_inputs = _load_phase_inputs()
    colors = _platform_colors(phase_inputs)
    print(f"[SAR_SIMPLE_3D] Building phased scene: {len(phase_inputs)} phase(s)")
    phases = _build_scene_phases(phase_inputs, colors)
    if not phases or not any(phase.drones for phase in phases):
        raise RuntimeError("No drone tracks were generated.")

    fig, anim = _make_animation(phases, args)

    if args.save:
        output = Path(args.save)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"[SAR_SIMPLE_3D] Saving animation to {output}")
        anim.save(str(output))
    elif not args.no_show:
        import matplotlib.pyplot as plt

        print("[SAR_SIMPLE_3D] Showing visualization window. Close it to stop.")
        plt.show()
    else:
        # The smoke-test mode intentionally builds one frame without rendering.
        anim._draw_was_started = True

    # Keep references alive until after show/save.
    _ = fig, anim


if __name__ == "__main__":
    main()
