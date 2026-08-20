from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .air_combat_specs import (
    ARMORED_VEHICLE,
    BLUE_FORWARD_BASE,
    COMMAND_POST,
    RADAR_SITE,
    RED_FORWARD_BASE,
    CH4_RECON,
    CH4_STRIKE_RECON,
    QUAD_RECON_UAV,
    QUAD_STRIKE_UAV,
    WZ21_LEADER,
    WZ21_WINGMAN,
    GroundTargetSpec,
    PlatformSpec,
)
from .aircraft_motion import BaseMotionModel, MotionState, build_motion_model
from sensors.air_combat_obstacles import compute_obstacle_contacts as compute_air_combat_obstacle_contacts


AIR_COMBAT_SCENE_VERSION = "2026-06-04.obstacle-sensing"


FOREST_ZONE_SPECS: tuple[tuple[str, float, float, float, float], ...] = (
    ("Forest_Northwest", -0.56, 0.30, 0.050, 28.0),
    ("Forest_CentralValley", -0.10, -0.10, 0.042, 24.0),
    ("Forest_EastSlope", 0.34, 0.24, 0.046, 30.0),
    ("Forest_SouthRoute", 0.18, -0.42, 0.038, 22.0),
)

MOUNTAIN_OBSTACLE_SPECS: tuple[tuple[str, float, float, float], ...] = (
    ("MountainPeak_West", -0.58, -0.20, 0.050),
    ("MountainRidge_Northwest", -0.24, 0.34, 0.056),
    ("MountainPeak_North", -0.04, 0.42, 0.052),
    ("MountainPeak_East", 0.34, 0.06, 0.048),
    ("MountainRidge_Northeast", 0.58, 0.44, 0.050),
)

ROCK_FIELD_SPECS: tuple[tuple[str, float, float, float], ...] = (
    ("RockField_WestPass", -0.40, -0.04, 0.026),
    ("RockField_CentralRidge", 0.10, 0.20, 0.030),
    ("RockField_EastSlope", 0.46, -0.12, 0.028),
)


FIXED_TARGET_REGION_SLOTS_KM: dict[str, list[tuple[float, float]]] = {
    # Relative kilometre coordinates.  The bridge converts these to MILP km as
    # [x + 150, y + 150] on the default 300 km map.
    "c0": [(96.0, 96.0), (104.0, 100.0), (100.0, 104.0)],
    "c1": [(71.0, 121.0), (79.0, 125.0), (75.0, 129.0)],
    "c2": [(121.0, 121.0), (129.0, 125.0), (125.0, 129.0)],
    "c3": [(71.0, 71.0), (79.0, 75.0), (75.0, 79.0)],
    "c4": [(121.0, 71.0), (129.0, 75.0), (125.0, 79.0)],
}


DEFAULT_AIR_COMBAT_CONFIG: dict[str, Any] = {
    "map": {
        "real_size_km": 300.0,
        "meters_per_unit": 100.0,
        "mountain_height_m": 1500.0,
        "terrain_vertical_exaggeration": 10.0,
        "terrain_source": "procedural_grid",
        "terrain_usd": None,
        "grid": 128,
        "grid_spacing_km": 25.0,
    },
    "visual": {
        "aircraft_scale": 180.0,
        "missile_visual_scale": 180.0,
        "ground_target_scale": 10.0,
        "armor_visual_scale": 180.0,
        "base_visual_scale": 5.0,
        "marker_scale": 1.0,
        "show_sensor_rings": False,
        "show_sensor_cones": False,
        "show_route_lines": False,
        "show_terrain_grid": False,
        "show_terrain_wireframe": False,
        "show_contours": False,
        "show_collision_body": False,
    },
    "weather": {
        "clouds": True,
        "cloud_count": 18,
        "cloud_opacity": 0.10,
        "cloud_altitude_m": 4300.0,
        "local_zones": True,
        "rain": True,
        "rain_count": 120,
        "rain_opacity": 0.24,
    },
    "bases": {
        "enabled": True,
    },
    "simulation": {
        "time_scale": 45.0,
        "aircraft_model": "kinematic",
    },
    "aircraft": {
        "terrain_clearance_m": 120.0,
    },
    "targets": {
        "radar_sites": 3,
        "command_posts": 2,
        "armored_vehicles": 10,
        "forward_bases": False,
    },
    "obstacles": {
        "enabled": True,
        "mountains": True,
        "forests": True,
        "rock_fields": True,
        "forest_tree_count": 140,
        "forest_tree_visual_scale": 220.0,
        "max_contacts_per_platform": 12,
    },
}


@dataclass
class AirPlatform:
    entity_id: str
    spec: PlatformSpec
    root_prim: object
    motion_model: BaseMotionModel
    reference_fn: Callable[[float], np.ndarray]
    rotor_prims: list[object] = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        return np.array(self.motion_model.state.position, dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array(self.motion_model.state.velocity, dtype=float)


@dataclass
class GroundTarget:
    target_id: str
    spec: GroundTargetSpec
    root_prim: object
    position: np.ndarray
    route_xy: list[tuple[float, float]] = field(default_factory=list)
    route_phase: float = 0.0
    route_speed_units: float = 0.0
    destroyed: bool = False
    destroyed_time_s: float | None = None


@dataclass
class EnvironmentObstacle:
    obstacle_id: str
    name: str
    category: str
    position: np.ndarray
    radius_units: float
    height_units: float
    priority: int
    blocks_los: bool = True


@dataclass
class AirCombatSceneState:
    stage: object
    config: dict[str, Any]
    map_size_units: float
    meters_per_unit: float
    mountain_height_units: float
    terrain_visual_height_units: float
    platforms: list[AirPlatform]
    targets: list[GroundTarget]
    obstacles: list[EnvironmentObstacle] = field(default_factory=list)
    contacts: list[dict[str, Any]] = field(default_factory=list)
    obstacle_contacts: list[dict[str, Any]] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)
    tactical_time_s: float = 0.0

    @property
    def map_size_km(self) -> float:
        return self.map_size_units * self.meters_per_unit / 1000.0

    def terrain_height(self, x: float, y: float) -> float:
        return terrain_height(x, y, self.map_size_units, self.mountain_height_units)

    def surface_height(self, x: float, y: float) -> float:
        return terrain_height(x, y, self.map_size_units, self.terrain_visual_height_units)

    def camera_view(self) -> tuple[list[float], list[float]]:
        if self.platforms:
            target_pos = self.platforms[0].position
            eye = [
                float(target_pos[0] + self.map_size_units * 0.10),
                float(target_pos[1] - self.map_size_units * 0.12),
                float(target_pos[2] + max(120.0, self.terrain_visual_height_units * 1.8)),
            ]
            target = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]
            return eye, target
        half = self.map_size_units * 0.5
        eye = [half * 0.56, -half * 0.70, self.terrain_visual_height_units + half * 0.28]
        target = [0.0, 0.0, self.terrain_visual_height_units * 0.45]
        return eye, target

    def update(self, tactical_dt: float, tactical_time_s: float) -> None:
        self.tactical_time_s = float(tactical_time_s)
        tactical_dt = max(1e-3, float(tactical_dt))
        for target in self.targets:
            if not target.destroyed and not target.spec.is_fixed and target.route_xy:
                _update_ground_target(self, target, tactical_dt)

        for platform in self.platforms:
            clearance_units = _air_platform_clearance_units(self.config, platform.spec, self.meters_per_unit)
            target_pos = platform.reference_fn(self.tactical_time_s)
            target_vel = _reference_velocity(platform.reference_fn, self.tactical_time_s)
            state = platform.motion_model.step(tactical_dt, target_pos, target_vel)
            state.position = _clamp_air_platform_above_terrain(state.position, self.map_size_units, self.terrain_visual_height_units, clearance_units)
            _set_root_pose(platform.root_prim, state.position, state)
            _spin_rotors(platform.rotor_prims, self.tactical_time_s)

        self.contacts = compute_sensor_contacts(self)
        self.obstacle_contacts = compute_obstacle_contacts(self)
        self.assignments = assign_targets(self)

    def contacts_for(self, platform_id: str) -> list[dict[str, Any]]:
        return [contact for contact in self.contacts if contact["platform_id"] == platform_id]

    def obstacle_contacts_for(self, platform_id: str) -> list[dict[str, Any]]:
        return [contact for contact in self.obstacle_contacts if contact["platform_id"] == platform_id]

    def mark_target_destroyed(self, target_id: str, cause: str = "weapon_effect") -> bool:
        for target in self.targets:
            if target.target_id == target_id:
                if target.destroyed:
                    return False
                target.destroyed = True
                target.destroyed_time_s = self.tactical_time_s
                target.route_xy = []
                target.route_speed_units = 0.0
                _mark_target_destroyed_visual(self.stage, target, self, cause)
                self.contacts = compute_sensor_contacts(self)
                self.obstacle_contacts = compute_obstacle_contacts(self)
                self.assignments = assign_targets(self)
                return True
        return False


def create_scene(stage, config: dict[str, Any] | None = None) -> AirCombatSceneState:
    cfg = _deep_merge(DEFAULT_AIR_COMBAT_CONFIG, config or {})
    meters_per_unit = float(cfg["map"].get("meters_per_unit", 100.0))
    map_size_units = float(cfg["map"].get("real_size_km", 300.0)) * 1000.0 / meters_per_unit
    mountain_height_units = float(cfg["map"].get("mountain_height_m", 1500.0)) / meters_per_unit
    visual_exaggeration = max(1.0, float(cfg["map"].get("terrain_vertical_exaggeration", 10.0)))
    terrain_visual_height_units = mountain_height_units * visual_exaggeration

    _setup_stage(stage, meters_per_unit)
    _create_roots(stage)
    _create_mountain_terrain(stage, cfg, map_size_units, terrain_visual_height_units, int(cfg["map"].get("grid", 112)))
    if bool(cfg["visual"].get("show_terrain_grid", False)):
        _create_terrain_grid(stage, map_size_units, terrain_visual_height_units, float(cfg["map"].get("grid_spacing_km", 25.0)), meters_per_unit)
    if bool(cfg["visual"].get("show_terrain_wireframe", False)):
        _create_ridge_wireframe(stage, map_size_units, terrain_visual_height_units)
    if bool(cfg["visual"].get("show_contours", False)):
        _create_contour_lines(stage, map_size_units, terrain_visual_height_units)
    if bool(cfg.get("bases", {}).get("enabled", True)):
        _create_forward_bases(stage, cfg, map_size_units, terrain_visual_height_units, meters_per_unit)
    if bool(cfg.get("weather", {}).get("clouds", True)):
        _create_cloud_layer(stage, cfg, map_size_units, terrain_visual_height_units, meters_per_unit)
    if bool(cfg["visual"].get("show_route_lines", True)):
        _create_mission_corridors(stage, map_size_units, terrain_visual_height_units, meters_per_unit)

    targets = _spawn_targets(stage, cfg, map_size_units, terrain_visual_height_units, meters_per_unit)
    platforms = _spawn_platforms(stage, cfg, map_size_units, terrain_visual_height_units, meters_per_unit)
    obstacles = _build_environment_obstacles(cfg, map_size_units, terrain_visual_height_units, meters_per_unit)

    scene = AirCombatSceneState(
        stage=stage,
        config=cfg,
        map_size_units=map_size_units,
        meters_per_unit=meters_per_unit,
        mountain_height_units=mountain_height_units,
        terrain_visual_height_units=terrain_visual_height_units,
        platforms=platforms,
        targets=targets,
        obstacles=obstacles,
    )
    scene.contacts = compute_sensor_contacts(scene)
    scene.obstacle_contacts = compute_obstacle_contacts(scene)
    scene.assignments = assign_targets(scene)
    print(
        f"[QL] Air combat scene ready: {scene.map_size_km:.0f} km x {scene.map_size_km:.0f} km, "
        f"terrain peak reference {cfg['map'].get('mountain_height_m', 1500.0):.0f} m, "
        f"visual terrain x{visual_exaggeration:.1f}, "
        f"platforms={len(platforms)}, targets={len(targets)}, obstacles={len(obstacles)}, version={AIR_COMBAT_SCENE_VERSION}"
    )
    return scene


def compute_sensor_contacts(scene: AirCombatSceneState) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for platform in scene.platforms:
        observer_pos = platform.position
        for target in scene.targets:
            if target.destroyed:
                continue
            if target.spec.faction == platform.spec.faction:
                continue
            target_pos = np.array(target.position, dtype=float)
            delta = target_pos - observer_pos
            distance_units = float(np.linalg.norm(delta))
            distance_km = distance_units * scene.meters_per_unit / 1000.0
            for sensor in platform.spec.sensors:
                if sensor.can_detect_emitters and not target.spec.is_radiating:
                    continue
                if not _sensor_accepts_target(sensor.target_kinds, target.spec.category):
                    continue
                if distance_km > sensor.max_range_km:
                    continue
                if not _sensor_scan_contains(scene, platform, sensor, delta):
                    continue
                if sensor.channel != "elint" and _terrain_masks_line(scene, observer_pos, target_pos):
                    continue
                contacts.append(
                    {
                        "platform_id": platform.entity_id,
                        "platform": platform.spec.name,
                        "target_id": target.target_id,
                        "target": target.spec.name,
                        "target_category": target.spec.category,
                        "sensor": sensor.name,
                        "channel": sensor.channel,
                        "azimuth_fov_deg": _sensor_value(sensor, "azimuth_fov_deg", 360.0),
                        "elevation_fov_deg": _sensor_value(sensor, "elevation_fov_deg", 60.0),
                        "scan_rate_hz": _sensor_value(sensor, "scan_rate_hz", 1.0),
                        "distance_km": distance_km,
                        "priority": target.spec.priority,
                    }
                )
    contacts.sort(key=lambda item: (-int(item["priority"]), float(item["distance_km"])))
    return contacts


def compute_obstacle_contacts(scene: AirCombatSceneState) -> list[dict[str, Any]]:
    if not bool(scene.config.get("obstacles", {}).get("enabled", True)):
        return []
    return compute_air_combat_obstacle_contacts(
        scene.platforms,
        scene.obstacles,
        tactical_time_s=scene.tactical_time_s,
        meters_per_unit=scene.meters_per_unit,
        max_contacts_per_platform=int(scene.config.get("obstacles", {}).get("max_contacts_per_platform", 12)),
        terrain_mask_fn=lambda observer, target: _terrain_masks_line(scene, observer, target),
    )


def assign_targets(scene: AirCombatSceneState) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    contacts_by_target: dict[str, list[dict[str, Any]]] = {}
    for contact in scene.contacts:
        contacts_by_target.setdefault(contact["target_id"], []).append(contact)

    for target in sorted(scene.targets, key=lambda item: item.spec.priority, reverse=True):
        if target.destroyed:
            continue
        contacts = contacts_by_target.get(target.target_id, [])
        if not contacts:
            continue
        detector = contacts[0]
        armed_platforms = [
            platform
            for platform in scene.platforms
            if platform.spec.faction != target.spec.faction and platform.spec.weapons.max_standoff_range_km > 0.0
        ]
        striker, weapon_name, weapon_margin = _select_striker(scene, target, armed_platforms)
        task_type = "strike" if striker is not None and target.spec.priority >= 70 else "recon"
        assignments.append(
            {
                "target_id": target.target_id,
                "target": target.spec.name,
                "task": task_type,
                "detected_by": detector["platform_id"],
                "sensor": detector["channel"],
                "assigned_to": striker.entity_id if striker is not None else detector["platform_id"],
                "weapon": weapon_name,
                "weapon_margin_km": weapon_margin,
                "priority": target.spec.priority,
            }
        )
    return assignments


def _build_environment_obstacles(cfg: dict[str, Any], map_size: float, height_scale: float, meters_per_unit: float) -> list[EnvironmentObstacle]:
    obstacle_cfg = cfg.get("obstacles", {})
    if not bool(obstacle_cfg.get("enabled", True)):
        return []
    obstacles: list[EnvironmentObstacle] = []
    half = map_size * 0.5

    if bool(obstacle_cfg.get("mountains", True)):
        for name, nx, ny, radius_ratio in MOUNTAIN_OBSTACLE_SPECS:
            x = float(nx * half)
            y = float(ny * half)
            z = terrain_height(x, y, map_size, height_scale)
            radius = max(3.0, map_size * float(radius_ratio))
            obstacles.append(
                EnvironmentObstacle(
                    obstacle_id=name,
                    name=name.replace("_", " "),
                    category="mountain",
                    position=np.array([x, y, z + max(30.0 / meters_per_unit, height_scale * 0.05)], dtype=float),
                    radius_units=radius,
                    height_units=max(z, height_scale * 0.35),
                    priority=62,
                )
            )

    if bool(obstacle_cfg.get("forests", True)):
        obstacles.extend(_generate_tree_obstacles(cfg, map_size, height_scale, meters_per_unit))

    if bool(obstacle_cfg.get("rock_fields", True)):
        for name, nx, ny, radius_ratio in ROCK_FIELD_SPECS:
            x = float(nx * half)
            y = float(ny * half)
            z = terrain_height(x, y, map_size, height_scale)
            radius = max(1.5, map_size * float(radius_ratio))
            obstacles.append(
                EnvironmentObstacle(
                    obstacle_id=name,
                    name=name.replace("_", " "),
                    category="rock",
                    position=np.array([x, y, z + max(8.0 / meters_per_unit, height_scale * 0.015)], dtype=float),
                    radius_units=radius,
                    height_units=max(8.0 / meters_per_unit, height_scale * 0.035),
                    priority=42,
                )
            )
    return obstacles


def _generate_tree_obstacles(cfg: dict[str, Any], map_size: float, height_scale: float, meters_per_unit: float) -> list[EnvironmentObstacle]:
    obstacle_cfg = cfg.get("obstacles", {})
    tree_samples = _forest_tree_samples(cfg, map_size, height_scale)
    visual_scale = max(1.0, float(obstacle_cfg.get("forest_tree_visual_scale", 220.0)))
    size_scale = visual_scale / meters_per_unit
    obstacles: list[EnvironmentObstacle] = []
    for sample in tree_samples:
        tree_height_units = float(sample["height_m"]) * size_scale
        crown_radius_units = float(sample["crown_radius_m"]) * size_scale
        obstacles.append(
            EnvironmentObstacle(
                obstacle_id=str(sample["tree_id"]),
                name=str(sample["tree_id"]).replace("_", " "),
                category="tree",
                position=np.array([float(sample["x"]), float(sample["y"]), float(sample["ground_z"]) + tree_height_units * 0.72], dtype=float),
                radius_units=max(0.25, crown_radius_units),
                height_units=max(0.4, tree_height_units),
                priority=46,
                blocks_los=False,
            )
        )
    return obstacles


def terrain_height(x: float, y: float, map_size: float, height_scale: float) -> float:
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


def _spawn_platforms(stage, cfg: dict[str, Any], map_size: float, height_scale: float, meters_per_unit: float) -> list[AirPlatform]:
    platform_specs: list[tuple[str, PlatformSpec, list[tuple[float, float]], float]] = [
        (
            "Blue_WZ21_Leader",
            WZ21_LEADER,
            _scale_path([(-146, -118), (-134, -104), (-112, -84), (-82, -58), (-48, -32), (-14, -12), (28, 4), (62, 18)], meters_per_unit),
            720.0,
        ),
        (
            "Blue_WZ21_Wingman",
            WZ21_WINGMAN,
            _scale_path([(-148, -128), (-136, -112), (-114, -92), (-86, -66), (-52, -40), (-18, -20), (22, -4), (56, 10)], meters_per_unit),
            690.0,
        ),
        (
            "Blue_CH4_Recon",
            CH4_RECON,
            _scale_path([(-150, -58), (-118, -44), (-78, -24), (-32, 0), (18, 28), (56, 66), (18, 92), (-48, 74), (-104, 28)], meters_per_unit),
            4600.0,
        ),
        (
            "Blue_CH4_StrikeRecon",
            CH4_STRIKE_RECON,
            _scale_path([(-150, -86), (-112, -72), (-68, -52), (-22, -28), (24, -2), (54, 24), (36, 48), (-8, 32), (-58, -10)], meters_per_unit),
            3800.0,
        ),
        (
            "Blue_Quad_Recon_1",
            QUAD_RECON_UAV,
            _scale_path([(-142, 28), (-126, 34), (-106, 42), (-80, 48), (-48, 50), (-22, 42), (-2, 24), (8, 4), (-18, -8), (-64, 2)], meters_per_unit),
            1250.0,
        ),
        (
            "Blue_Quad_Recon_2",
            QUAD_RECON_UAV,
            _scale_path([(-142, -24), (-122, -18), (-96, -8), (-70, 4), (-42, 10), (-14, 8), (12, -8), (4, -30), (-34, -38), (-76, -34)], meters_per_unit),
            1150.0,
        ),
        (
            "Blue_Quad_Strike_1",
            QUAD_STRIKE_UAV,
            _scale_path([(-140, 8), (-122, 14), (-98, 22), (-70, 26), (-42, 22), (-16, 14), (6, 2), (14, -12), (-10, -20), (-48, -14)], meters_per_unit),
            950.0,
        ),
        (
            "Blue_Quad_Strike_2",
            QUAD_STRIKE_UAV,
            _scale_path([(-140, -44), (-120, -38), (-94, -30), (-64, -20), (-34, -12), (-8, -8), (12, -18), (4, -38), (-28, -52), (-70, -52)], meters_per_unit),
            900.0,
        ),
    ]

    platforms: list[AirPlatform] = []
    sim_cfg = cfg.get("simulation", {})
    for idx, (entity_id, spec, route_xy, altitude_agl_m) in enumerate(platform_specs):
        clearance_units = _air_platform_clearance_units(cfg, spec, meters_per_unit)
        ref = _make_air_reference(route_xy, altitude_agl_m, map_size, height_scale, meters_per_unit, spec, clearance_units)
        start = ref(0.0)
        root, rotors = _spawn_air_platform(
            stage,
            f"/World/AirCombat/Platforms/{entity_id}",
            spec,
            start,
            meters_per_unit,
            float(cfg["visual"].get("aircraft_scale", 16.0)),
            idx,
            bool(cfg["visual"].get("show_collision_body", False)),
        )
        model = build_motion_model(
            {
                "type": sim_cfg.get("aircraft_model", "kinematic"),
                "max_speed": spec.cruise_speed_mps / meters_per_unit,
                "max_accel": max(0.05, spec.climb_rate_mps * 2.5 / meters_per_unit),
            },
            start,
            domain="air",
        )
        platforms.append(AirPlatform(entity_id=entity_id, spec=spec, root_prim=root, motion_model=model, reference_fn=ref, rotor_prims=rotors))
        if bool(cfg["visual"].get("show_sensor_rings", False)):
            _create_sensor_ring(stage, f"/World/AirCombat/SensorRings/{entity_id}_MaxSensor", start, _max_sensor_range_units(spec, meters_per_unit), spec.color)
        if bool(cfg["visual"].get("show_sensor_cones", False)):
            _create_sensor_cones(stage, f"/World/AirCombat/Platforms/{entity_id}/SensorCones", np.zeros(3, dtype=float), spec, meters_per_unit)
    return platforms


def _spawn_targets(stage, cfg: dict[str, Any], map_size: float, height_scale: float, meters_per_unit: float) -> list[GroundTarget]:
    targets: list[GroundTarget] = []
    if bool(cfg["targets"].get("forward_bases", True)) and bool(cfg.get("bases", {}).get("enabled", True)):
        blue_xy = _scale_path([(-128, -128)], meters_per_unit)[0]
        red_xy = _scale_path([(128, 112)], meters_per_unit)[0]
        targets.append(
            GroundTarget(
                target_id="BlueBase_Target",
                spec=BLUE_FORWARD_BASE,
                root_prim=stage.GetPrimAtPath("/World/AirCombat/Bases/BlueBase"),
                position=_base_target_position(blue_xy, map_size, height_scale, meters_per_unit),
            )
        )
        targets.append(
            GroundTarget(
                target_id="RedBase_Target",
                spec=RED_FORWARD_BASE,
                root_prim=stage.GetPrimAtPath("/World/AirCombat/Bases/RedBase"),
                position=_base_target_position(red_xy, map_size, height_scale, meters_per_unit),
            )
        )

    fixed_regions = FIXED_TARGET_REGION_SLOTS_KM
    radar_positions = _scale_path([
        fixed_regions["c0"][0],
        fixed_regions["c1"][0],
        fixed_regions["c2"][0],
    ], meters_per_unit)
    command_positions = _scale_path([
        fixed_regions["c0"][1],
        fixed_regions["c1"][1],
    ], meters_per_unit)
    armor_positions = _scale_path([
        fixed_regions["c0"][2],
        fixed_regions["c1"][2],
        fixed_regions["c2"][1],
        fixed_regions["c2"][2],
        fixed_regions["c3"][0],
        fixed_regions["c3"][1],
        fixed_regions["c3"][2],
        fixed_regions["c4"][0],
        fixed_regions["c4"][1],
        fixed_regions["c4"][2],
    ], meters_per_unit)
    armor_route = _scale_path([
        fixed_regions["c0"][2],
        fixed_regions["c1"][2],
        fixed_regions["c2"][2],
        fixed_regions["c4"][2],
        fixed_regions["c3"][2],
    ], meters_per_unit)

    for idx, (x, y) in enumerate(radar_positions[: int(cfg["targets"].get("radar_sites", 3))]):
        pos = _ground_position(x, y, map_size, height_scale, 1.4, meters_per_unit)
        root = _spawn_radar_site(stage, f"/World/AirCombat/Targets/Radar_{idx}", RADAR_SITE, pos, meters_per_unit, float(cfg["visual"].get("ground_target_scale", 10.0)))
        targets.append(GroundTarget(target_id=f"Radar_{idx}", spec=RADAR_SITE, root_prim=root, position=pos))

    for idx, (x, y) in enumerate(command_positions[: int(cfg["targets"].get("command_posts", 2))]):
        pos = _ground_position(x, y, map_size, height_scale, 1.2, meters_per_unit)
        root = _spawn_command_post(stage, f"/World/AirCombat/Targets/CommandPost_{idx}", COMMAND_POST, pos, meters_per_unit, float(cfg["visual"].get("ground_target_scale", 10.0)))
        targets.append(GroundTarget(target_id=f"CommandPost_{idx}", spec=COMMAND_POST, root_prim=root, position=pos))

    count = int(cfg["targets"].get("armored_vehicles", 10))
    armor_visual_scale = float(cfg["visual"].get("armor_visual_scale", cfg["visual"].get("ground_target_scale", 10.0)))
    armor_dims_units = _scaled_dims(ARMORED_VEHICLE.dimensions_m, meters_per_unit, armor_visual_scale)
    armor_ground_offset_m = max(1.2, armor_dims_units[2] * 0.19 * meters_per_unit)
    for idx in range(count):
        phase = idx / max(1, count)
        x, y = armor_positions[idx % len(armor_positions)]
        pos = _ground_position_for_dimensions(x, y, map_size, height_scale, armor_ground_offset_m, meters_per_unit, armor_dims_units)
        root = _spawn_armored_vehicle(stage, f"/World/AirCombat/Targets/Armor_{idx}", ARMORED_VEHICLE, pos, meters_per_unit, armor_visual_scale, idx)
        targets.append(
            GroundTarget(
                target_id=f"Armor_{idx}",
                spec=ARMORED_VEHICLE,
                root_prim=root,
                position=pos,
                route_xy=armor_route,
                route_phase=phase,
                route_speed_units=ARMORED_VEHICLE.mobility_speed_mps / meters_per_unit,
            )
        )
    if bool(cfg["visual"].get("show_route_lines", False)):
        _create_route_curve(stage, "/World/AirCombat/Routes/ArmorColumnRoute", armor_route, map_size, height_scale, (0.48, 0.32, 0.18))
    return targets


def _spawn_air_platform(stage, root_path: str, spec: PlatformSpec, pos: np.ndarray, meters_per_unit: float, visual_scale: float, idx: int, show_collision_body: bool):
    root = _make_root(stage, root_path, pos)
    asset_path = _resolve_asset_path(spec)
    if asset_path and _add_usd_reference(stage, asset_path, f"{root_path}/Model"):
        _set_root_scale(root, visual_scale / meters_per_unit)
        _create_sphere(stage, f"{root_path}/Beacon", (0.0, 0.0, _m_to_units(spec.height_m * 1.2, meters_per_unit, visual_scale)), _m_to_units(1.4, meters_per_unit, visual_scale), (0.0, 0.7, 1.0))
        _create_platform_marker(stage, root_path, spec, meters_per_unit, visual_scale)
        _create_platform_collision_body(stage, root_path, spec, meters_per_unit, visual_scale, visible=show_collision_body)
        return root, []

    if "quadrotor" in spec.role:
        rotors = _spawn_quadrotor_uav_body(stage, root_path, spec, meters_per_unit, visual_scale, idx)
    elif "uav" in spec.role:
        rotors = _spawn_ch4_uav_body(stage, root_path, spec, meters_per_unit, visual_scale, idx)
    else:
        rotors = _spawn_helicopter_body(stage, root_path, spec, meters_per_unit, visual_scale)
    _create_platform_marker(stage, root_path, spec, meters_per_unit, visual_scale)
    _create_platform_collision_body(stage, root_path, spec, meters_per_unit, visual_scale, visible=show_collision_body)
    return root, rotors


def _create_platform_collision_body(stage, root_path: str, spec: PlatformSpec, meters_per_unit: float, visual_scale: float, visible: bool) -> object:
    length = _m_to_units(max(spec.length_m, spec.rotor_diameter_m * 0.65), meters_per_unit, visual_scale)
    width = _m_to_units(spec.rotor_diameter_m if "uav" in spec.role else spec.rotor_diameter_m * 0.42, meters_per_unit, visual_scale)
    height = _m_to_units(max(spec.height_m, spec.rotor_diameter_m * 0.12), meters_per_unit, visual_scale)
    color = (0.0, 0.45, 1.0) if spec.faction == "Blue" else (1.0, 0.12, 0.04)
    prim = _create_box(stage, f"{root_path}/CollisionBody", (0.0, 0.0, 0.0), (length, width, height), color)
    _apply_collision_api(prim, visible=visible)
    return prim


def _spawn_helicopter_body(stage, root_path: str, spec: PlatformSpec, meters_per_unit: float, visual_scale: float) -> list[object]:
    is_apache_style = spec.faction == "Red" or "Apache" in spec.name or "AH-64" in spec.name
    fuselage_len = _m_to_units(spec.length_m * (0.54 if is_apache_style else 0.62), meters_per_unit, visual_scale)
    fuselage_width = _m_to_units(spec.rotor_diameter_m * (0.135 if is_apache_style else 0.105), meters_per_unit, visual_scale)
    fuselage_height = _m_to_units(spec.height_m * (0.40 if is_apache_style else 0.33), meters_per_unit, visual_scale)
    tail_len = _m_to_units(spec.length_m * (0.34 if is_apache_style else 0.36), meters_per_unit, visual_scale)
    rotor_diameter = _m_to_units(spec.rotor_diameter_m, meters_per_unit, visual_scale)
    rotor_z = _m_to_units(spec.height_m * (0.56 if is_apache_style else 0.52), meters_per_unit, visual_scale)
    canopy_color = (0.03, 0.06, 0.07) if spec.faction == "Blue" else (0.05, 0.04, 0.03)

    _create_box(stage, f"{root_path}/Fuselage", (0.0, 0.0, 0.0), (fuselage_len, fuselage_width, fuselage_height), spec.color)
    _create_box(stage, f"{root_path}/SensorNose", (fuselage_len * 0.54, 0.0, -fuselage_height * 0.04), (_m_to_units(spec.length_m * 0.10, meters_per_unit, visual_scale), fuselage_width * 0.72, fuselage_height * 0.64), spec.accent_color)
    if is_apache_style:
        _create_box(stage, f"{root_path}/FrontCanopy", (fuselage_len * 0.28, 0.0, fuselage_height * 0.42), (fuselage_len * 0.18, fuselage_width * 0.62, fuselage_height * 0.25), canopy_color)
        _create_box(stage, f"{root_path}/RearCanopy", (fuselage_len * 0.04, 0.0, fuselage_height * 0.50), (fuselage_len * 0.20, fuselage_width * 0.66, fuselage_height * 0.28), canopy_color)
        _create_box(stage, f"{root_path}/CheekSensorLeft", (fuselage_len * 0.46, fuselage_width * 0.42, -fuselage_height * 0.10), (fuselage_len * 0.12, fuselage_width * 0.18, fuselage_height * 0.34), spec.accent_color)
        _create_box(stage, f"{root_path}/CheekSensorRight", (fuselage_len * 0.46, -fuselage_width * 0.42, -fuselage_height * 0.10), (fuselage_len * 0.12, fuselage_width * 0.18, fuselage_height * 0.34), spec.accent_color)
    else:
        _create_box(stage, f"{root_path}/LongCanopy", (fuselage_len * 0.20, 0.0, fuselage_height * 0.44), (fuselage_len * 0.34, fuselage_width * 0.58, fuselage_height * 0.24), canopy_color)
        _create_box(stage, f"{root_path}/TaperedNose", (fuselage_len * 0.62, 0.0, fuselage_height * 0.08), (fuselage_len * 0.13, fuselage_width * 0.48, fuselage_height * 0.32), spec.color)
    _create_cylinder(stage, f"{root_path}/Mast", rotor_diameter * 0.018, _m_to_units(spec.height_m * 0.22, meters_per_unit, visual_scale), (0.0, 0.0, rotor_z * 0.78), spec.accent_color)
    _create_sphere(stage, f"{root_path}/MastRadarDome", (0.0, 0.0, rotor_z * 1.04), rotor_diameter * (0.045 if is_apache_style else 0.055), spec.accent_color)

    rotor_a = _create_box(stage, f"{root_path}/MainRotorA", (0.0, 0.0, rotor_z), (rotor_diameter, rotor_diameter * 0.025, max(0.02, fuselage_height * 0.06)), spec.accent_color)
    rotor_b = _create_box(stage, f"{root_path}/MainRotorB", (0.0, 0.0, rotor_z + 0.03), (rotor_diameter, rotor_diameter * 0.025, max(0.02, fuselage_height * 0.06)), spec.accent_color, rotate_xyz=(0.0, 0.0, 90.0))
    _create_box(stage, f"{root_path}/TailBoom", (-(fuselage_len * 0.45 + tail_len * 0.50), 0.0, fuselage_height * 0.08), (tail_len, fuselage_width * 0.28, fuselage_height * 0.28), spec.color)
    _create_box(stage, f"{root_path}/TailFin", (-(fuselage_len * 0.45 + tail_len * 0.88), 0.0, fuselage_height * 0.44), (tail_len * 0.18, fuselage_width * 0.13, fuselage_height * 0.84), spec.accent_color)
    tail_rotor = _create_box(stage, f"{root_path}/TailRotor", (-(fuselage_len * 0.45 + tail_len), 0.0, fuselage_height * 0.22), (_m_to_units(spec.height_m * 0.48, meters_per_unit, visual_scale), _m_to_units(spec.height_m * 0.04, meters_per_unit, visual_scale), _m_to_units(spec.height_m * 0.025, meters_per_unit, visual_scale)), spec.accent_color, rotate_xyz=(0.0, 90.0, 0.0))
    _create_cylinder(stage, f"{root_path}/ChinCannon", _m_to_units(spec.height_m * 0.025, meters_per_unit, visual_scale), _m_to_units(spec.length_m * 0.10, meters_per_unit, visual_scale), (fuselage_len * 0.60, 0.0, -fuselage_height * 0.55), spec.accent_color, rotate_xyz=(0.0, 90.0, 0.0))
    pylon_offsets = (-0.10, 0.16) if is_apache_style else (-0.04, 0.22)
    for side, y_sign in (("Left", 1.0), ("Right", -1.0)):
        wing_y = y_sign * fuselage_width * (0.88 if is_apache_style else 0.82)
        _create_box(stage, f"{root_path}/{side}StubWing", (fuselage_len * 0.08, wing_y, -fuselage_height * 0.08), (_m_to_units(spec.length_m * 0.18, meters_per_unit, visual_scale), fuselage_width * 0.48, _m_to_units(spec.height_m * 0.045, meters_per_unit, visual_scale)), spec.accent_color)
        for station_idx, x_ratio in enumerate(pylon_offsets):
            _create_cylinder(
                stage,
                f"{root_path}/{side}Store_{station_idx}",
                _m_to_units(spec.height_m * 0.048, meters_per_unit, visual_scale),
                _m_to_units(spec.length_m * 0.16, meters_per_unit, visual_scale),
                (fuselage_len * x_ratio, y_sign * fuselage_width * 1.12, -fuselage_height * 0.25),
                (0.18, 0.18, 0.16),
                rotate_xyz=(0.0, 90.0, 0.0),
            )
    return [rotor_a, rotor_b, tail_rotor]


def _spawn_ch4_uav_body(stage, root_path: str, spec: PlatformSpec, meters_per_unit: float, visual_scale: float, idx: int) -> list[object]:
    length = _m_to_units(spec.length_m * 1.38, meters_per_unit, visual_scale)
    span = _m_to_units(spec.rotor_diameter_m * 1.16, meters_per_unit, visual_scale)
    height = _m_to_units(spec.height_m, meters_per_unit, visual_scale)
    body_color = spec.color
    accent = spec.accent_color
    _create_box(stage, f"{root_path}/Fuselage", (0.0, 0.0, 0.0), (length, height * 0.24, height * 0.24), body_color)
    _create_box(stage, f"{root_path}/Nose", (length * 0.53, 0.0, height * 0.02), (length * 0.12, height * 0.18, height * 0.18), body_color)
    _create_box(stage, f"{root_path}/WingMain", (-length * 0.04, 0.0, height * 0.02), (length * 0.13, span, height * 0.040), accent)
    _create_box(stage, f"{root_path}/WingCenterFairing", (-length * 0.04, 0.0, height * 0.08), (length * 0.26, height * 0.34, height * 0.10), body_color)
    _create_box(stage, f"{root_path}/TailBoom", (-length * 0.47, 0.0, height * 0.02), (length * 0.36, height * 0.10, height * 0.10), body_color)
    _create_box(stage, f"{root_path}/TailPlane", (-length * 0.72, 0.0, height * 0.06), (length * 0.14, span * 0.30, height * 0.040), accent)
    _create_box(stage, f"{root_path}/VerticalTail", (-length * 0.72, 0.0, height * 0.25), (length * 0.08, height * 0.055, height * 0.45), accent)
    _create_cylinder(stage, f"{root_path}/NoseEO", height * 0.095, height * 0.14, (length * 0.58, 0.0, -height * 0.20), (0.02, 0.05, 0.06), rotate_xyz=(90.0, 0.0, 0.0))
    _create_box(stage, f"{root_path}/SARPod", (length * 0.02, 0.0, -height * 0.22), (length * 0.15, height * 0.12, height * 0.10), (0.12, 0.14, 0.14))
    _create_cylinder(stage, f"{root_path}/PusherPropA", height * 0.03, height * 0.54, (-length * 0.68, 0.0, height * 0.03), accent, rotate_xyz=(90.0, 0.0, 0.0))
    _create_cylinder(stage, f"{root_path}/PusherPropB", height * 0.03, height * 0.54, (-length * 0.68, 0.0, height * 0.03), accent, rotate_xyz=(0.0, 90.0, 0.0))
    if "strike" in spec.role:
        for side, y in (("L1", 0.22), ("L2", 0.36), ("R1", -0.22), ("R2", -0.36)):
            _create_cylinder(stage, f"{root_path}/Store_{side}", height * 0.040, length * 0.22, (length * 0.02, y * span, -height * 0.16), (0.22, 0.22, 0.18), rotate_xyz=(0.0, 90.0, 0.0))
    beacon_color = (0.0, 0.55, 1.0) if idx % 2 == 0 else (0.0, 0.85, 0.55)
    _create_sphere(stage, f"{root_path}/Beacon", (length * 0.12, 0.0, height * 0.34), max(0.08, height * 0.10), beacon_color)
    return []


def _spawn_quadrotor_uav_body(stage, root_path: str, spec: PlatformSpec, meters_per_unit: float, visual_scale: float, idx: int) -> list[object]:
    length = _m_to_units(spec.length_m, meters_per_unit, visual_scale)
    span = _m_to_units(spec.rotor_diameter_m * 1.18, meters_per_unit, visual_scale)
    height = _m_to_units(spec.height_m, meters_per_unit, visual_scale)
    arm_len = span * 0.86
    body_color = spec.color
    accent = spec.accent_color

    _create_box(stage, f"{root_path}/CenterBody", (0.0, 0.0, 0.0), (length * 0.72, length * 0.42, height * 0.42), body_color)
    _create_box(stage, f"{root_path}/BatteryPack", (-length * 0.10, 0.0, -height * 0.25), (length * 0.48, length * 0.26, height * 0.22), (0.06, 0.07, 0.07))
    _create_box(stage, f"{root_path}/ArmForeAft", (0.0, 0.0, height * 0.08), (arm_len, height * 0.09, height * 0.08), accent, rotate_xyz=(0.0, 0.0, 35.0))
    _create_box(stage, f"{root_path}/ArmAftFore", (0.0, 0.0, height * 0.08), (arm_len, height * 0.09, height * 0.08), accent, rotate_xyz=(0.0, 0.0, -35.0))
    _create_sphere(stage, f"{root_path}/EOBall", (length * 0.33, 0.0, -height * 0.44), max(0.06, height * 0.22), (0.02, 0.04, 0.05))

    rotor_radius = max(0.16, _m_to_units(spec.rotor_diameter_m * 0.21, meters_per_unit, visual_scale))
    rotor_blade_len = rotor_radius * 2.1
    rotor_z = height * 0.22
    rotor_offsets = (
        ("FrontLeft", length * 0.30, span * 0.34),
        ("FrontRight", length * 0.30, -span * 0.34),
        ("RearLeft", -length * 0.34, span * 0.34),
        ("RearRight", -length * 0.34, -span * 0.34),
    )
    rotors = []
    for rotor_idx, (name, x, y) in enumerate(rotor_offsets):
        _create_cylinder(stage, f"{root_path}/{name}Motor", height * 0.12, height * 0.18, (x, y, rotor_z - height * 0.06), accent)
        rotor = _create_box(
            stage,
            f"{root_path}/{name}QuadRotor",
            (x, y, rotor_z),
            (rotor_blade_len, max(0.035, height * 0.065), max(0.018, height * 0.026)),
            (0.04, 0.05, 0.05),
            rotate_xyz=(0.0, 0.0, 90.0 if rotor_idx % 2 else 0.0),
        )
        _create_torus_marker(stage, f"{root_path}/{name}RotorDisk", (x, y, rotor_z), rotor_radius, max(0.018, rotor_radius * 0.035), (0.10, 0.12, 0.12))
        rotors.append(rotor)

    if "strike" in spec.role:
        for side, y_sign in (("Left", 1.0), ("Right", -1.0)):
            _create_cylinder(
                stage,
                f"{root_path}/{side}LightMissile",
                max(0.035, height * 0.055),
                length * 0.62,
                (length * 0.02, y_sign * length * 0.38, -height * 0.34),
                (0.23, 0.22, 0.17),
                rotate_xyz=(0.0, 90.0, 0.0),
            )

    beacon_color = (0.0, 0.62, 1.0) if "recon" in spec.role else (1.0, 0.66, 0.05)
    _create_sphere(stage, f"{root_path}/Beacon", (0.0, 0.0, height * 0.72), max(0.10, height * 0.12), beacon_color)
    return rotors


def _create_platform_marker(stage, root_path: str, spec: PlatformSpec, meters_per_unit: float, visual_scale: float) -> None:
    marker_color = (0.0, 0.55, 1.0) if spec.faction == "Blue" else (1.0, 0.08, 0.02)
    role_color = _role_marker_color(spec)
    marker_radius = max(2.5, _m_to_units(2.4, meters_per_unit, visual_scale))
    marker_z = _m_to_units(spec.height_m * 2.2, meters_per_unit, visual_scale)
    _create_visibility_marker(
        stage,
        root_path,
        beacon_color=marker_color,
        panel_color=role_color,
        radius=marker_radius,
        z=marker_z,
        panel_name="FactionPanel",
        locator_name="VerticalLocator",
        ring_scale=4.2,
    )
    _create_role_marker(stage, root_path, spec, marker_radius, marker_z, role_color)
    _create_sensor_mast_marker(stage, root_path, spec, marker_radius, marker_z)


def _create_visibility_marker(
    stage,
    root_path: str,
    beacon_color: tuple[float, float, float],
    panel_color: tuple[float, float, float],
    radius: float,
    z: float,
    panel_name: str,
    locator_name: str,
    ring_scale: float,
    locator_radius: float | None = None,
    locator_height: float | None = None,
    panel_z_mul: float = 1.7,
    panel_width_mul: float = 3.6,
    panel_depth_mul: float = 0.34,
    panel_height_mul: float = 0.45,
    ring_z_mul: float = 0.9,
) -> None:
    locator_radius = radius * 0.10 if locator_radius is None else locator_radius
    locator_height = z + radius * 4.0 if locator_height is None else locator_height
    _create_cylinder(stage, f"{root_path}/{locator_name}", locator_radius, locator_height, (0.0, 0.0, locator_height * 0.5), beacon_color)
    _create_sphere(stage, f"{root_path}/HighVisibilityBeacon", (0.0, 0.0, z), radius, beacon_color)
    _create_box(stage, f"{root_path}/{panel_name}", (0.0, 0.0, z + radius * panel_z_mul), (radius * panel_width_mul, radius * panel_depth_mul, radius * panel_height_mul), panel_color)
    _create_torus_marker(stage, f"{root_path}/LocatorRing", (0.0, 0.0, z + radius * ring_z_mul), radius * ring_scale, radius * 0.12, beacon_color)


def _role_marker_color(spec: PlatformSpec) -> tuple[float, float, float]:
    if "leader" in spec.role:
        return (0.00, 0.90, 1.00)
    if "wingman" in spec.role:
        return (0.10, 1.00, 0.32)
    if "strike" in spec.role:
        return (1.00, 0.70, 0.08)
    if "recon" in spec.role:
        return (0.55, 0.35, 1.00)
    return (0.0, 0.55, 1.0)


def _create_role_marker(stage, root_path: str, spec: PlatformSpec, radius: float, marker_z: float, color: tuple[float, float, float]) -> None:
    z = marker_z + radius * 3.0
    if "leader" in spec.role:
        _create_cylinder(stage, f"{root_path}/RoleMarkerLeader", radius * 0.70, radius * 0.28, (0.0, 0.0, z), color)
        _create_cylinder(stage, f"{root_path}/RoleMarkerLeaderCore", radius * 0.32, radius * 0.34, (0.0, 0.0, z + radius * 0.16), (1.0, 1.0, 1.0))
    elif "wingman" in spec.role:
        _create_box(stage, f"{root_path}/RoleMarkerWingmanA", (0.0, 0.0, z), (radius * 2.0, radius * 0.28, radius * 0.28), color, rotate_xyz=(0.0, 0.0, 45.0))
        _create_box(stage, f"{root_path}/RoleMarkerWingmanB", (0.0, 0.0, z), (radius * 2.0, radius * 0.28, radius * 0.28), color, rotate_xyz=(0.0, 0.0, -45.0))
    elif "strike" in spec.role:
        _create_box(stage, f"{root_path}/RoleMarkerStrike", (0.0, 0.0, z), (radius * 1.65, radius * 1.65, radius * 0.34), color, rotate_xyz=(0.0, 0.0, 45.0))
    elif "recon" in spec.role:
        _create_torus_marker(stage, f"{root_path}/RoleMarkerRecon", (0.0, 0.0, z), radius * 1.25, radius * 0.12, color)
        _create_sphere(stage, f"{root_path}/RoleMarkerReconDot", (0.0, 0.0, z), radius * 0.22, color)


def _create_sensor_mast_marker(stage, root_path: str, spec: PlatformSpec, marker_radius: float, marker_z: float) -> None:
    radar_color = (0.0, 0.85, 1.0) if spec.faction == "Blue" else (1.0, 0.18, 0.05)
    _create_cylinder(stage, f"{root_path}/SensorMastHighlight", marker_radius * 0.18, marker_radius * 3.2, (0.0, 0.0, marker_z * 0.55), radar_color)
    _create_sphere(stage, f"{root_path}/SensorHeadHighlight", (0.0, 0.0, marker_z + marker_radius * 0.35), marker_radius * 0.42, radar_color)


def _spawn_radar_site(stage, root_path: str, spec: GroundTargetSpec, pos: np.ndarray, meters_per_unit: float, visual_scale: float):
    root = _make_root(stage, root_path, pos)
    sx, sy, sz = _scaled_dims(spec.dimensions_m, meters_per_unit, visual_scale)
    _create_box(stage, f"{root_path}/Cabin", (0.0, 0.0, sz * 0.20), (sx * 0.78, sy * 0.72, sz * 0.40), spec.color)
    _create_cylinder(stage, f"{root_path}/Mast", sx * 0.035, sz * 0.86, (0.0, 0.0, sz * 0.84), (0.18, 0.18, 0.16))
    _create_cylinder(stage, f"{root_path}/RadarDish", sx * 0.22, sx * 0.035, (0.0, 0.0, sz * 1.30), (0.82, 0.18, 0.12), rotate_xyz=(0.0, 90.0, 0.0))
    _create_sphere(stage, f"{root_path}/EmitterMarker", (0.0, 0.0, sz * 1.34), sx * 0.12, (1.0, 0.05, 0.0))
    try:
        from pxr import Sdf, UsdLux

        light = UsdLux.SphereLight.Define(stage, Sdf.Path(f"{root_path}/RadiationLight"))
        light.CreateIntensityAttr(450.0)
        light.CreateRadiusAttr(float(sx * 0.16))
        _set_xform(light.GetPrim(), translate=(0.0, 0.0, sz * 1.34))
    except Exception:
        pass
    return root


def _spawn_command_post(stage, root_path: str, spec: GroundTargetSpec, pos: np.ndarray, meters_per_unit: float, visual_scale: float):
    root = _make_root(stage, root_path, pos)
    sx, sy, sz = _scaled_dims(spec.dimensions_m, meters_per_unit, visual_scale)
    _create_box(stage, f"{root_path}/Bunker", (0.0, 0.0, sz * 0.22), (sx, sy, sz * 0.45), spec.color)
    _create_box(stage, f"{root_path}/Roof", (0.0, 0.0, sz * 0.50), (sx * 1.08, sy * 1.05, sz * 0.10), (0.23, 0.24, 0.22))
    _create_box(stage, f"{root_path}/OpsContainerA", (-sx * 0.32, sy * 0.74, sz * 0.22), (sx * 0.36, sy * 0.15, sz * 0.34), (0.28, 0.30, 0.28))
    _create_box(stage, f"{root_path}/OpsContainerB", (sx * 0.22, sy * 0.74, sz * 0.22), (sx * 0.32, sy * 0.15, sz * 0.34), (0.32, 0.31, 0.27))
    _create_cylinder(stage, f"{root_path}/CommsMast", sx * 0.018, sz * 0.90, (sx * 0.40, -sy * 0.34, sz * 0.88), (0.12, 0.12, 0.12))
    _create_sphere(stage, f"{root_path}/PriorityMarker", (0.0, 0.0, sz * 0.72), sx * 0.06, (1.0, 0.45, 0.0))
    return root


def _spawn_armored_vehicle(stage, root_path: str, spec: GroundTargetSpec, pos: np.ndarray, meters_per_unit: float, visual_scale: float, idx: int):
    root = _make_root(stage, root_path, pos)
    sx, sy, sz = _scaled_dims(spec.dimensions_m, meters_per_unit, visual_scale)
    hull = _create_box(stage, f"{root_path}/Hull", (0.0, 0.0, sz * 0.18), (sx, sy, sz * 0.38), spec.color)
    turret = _create_box(stage, f"{root_path}/Turret", (sx * 0.08, 0.0, sz * 0.47), (sx * 0.42, sy * 0.55, sz * 0.28), (0.52, 0.34, 0.18))
    gun = _create_cylinder(stage, f"{root_path}/Gun", sy * 0.035, sx * 0.74, (sx * 0.42, 0.0, sz * 0.52), (0.10, 0.10, 0.09), rotate_xyz=(0.0, 90.0, 0.0))
    track_l = _create_box(stage, f"{root_path}/TrackL", (0.0, -sy * 0.52, -sz * 0.05), (sx * 1.05, sy * 0.16, sz * 0.28), (0.05, 0.05, 0.05))
    track_r = _create_box(stage, f"{root_path}/TrackR", (0.0, sy * 0.52, -sz * 0.05), (sx * 1.05, sy * 0.16, sz * 0.28), (0.05, 0.05, 0.05))
    _apply_collision_to_prims((hull, turret, gun, track_l, track_r), visible=True)
    marker_color = (1.0, 0.10, 0.05) if idx % 2 == 0 else (1.0, 0.30, 0.05)
    _create_sphere(stage, f"{root_path}/HeatMarker", (-sx * 0.32, 0.0, sz * 0.62), sx * 0.05, marker_color)
    collision = _create_box(stage, f"{root_path}/CollisionBody", (0.0, 0.0, sz * 0.23), (sx * 1.10, sy * 1.20, sz * 0.70), (1.0, 0.08, 0.04))
    _apply_collision_api(collision, visible=False)
    beacon_radius = max(sx * 0.12, 0.55)
    marker_z = sz * 4.30
    _create_visibility_marker(
        stage,
        root_path,
        beacon_color=marker_color,
        panel_color=(1.0, 0.14, 0.04),
        radius=beacon_radius,
        z=marker_z,
        panel_name="EnemyPanel",
        locator_name="TargetLocatorMast",
        locator_radius=sx * 0.018,
        locator_height=marker_z,
        ring_scale=3.2,
        panel_z_mul=1.65,
        panel_width_mul=3.4,
        panel_height_mul=0.48,
        ring_z_mul=0.85,
    )
    _create_box(stage, f"{root_path}/RoleMarkerArmor", (0.0, 0.0, marker_z + beacon_radius * 2.85), (beacon_radius * 1.80, beacon_radius * 1.80, beacon_radius * 0.34), (1.0, 0.64, 0.10), rotate_xyz=(0.0, 0.0, 45.0))
    return root


def _mark_target_destroyed_visual(stage, target: GroundTarget, scene: AirCombatSceneState, cause: str) -> None:
    root_path = str(target.root_prim.GetPath())
    pos = target.position
    _dim_destroyed_target(stage, target.root_prim)
    marker_z = _ground_target_surface_offset_m(scene, target.spec) / scene.meters_per_unit + 0.9
    fire = _create_sphere(stage, f"{root_path}/DestroyedFireball", (0.0, 0.0, marker_z), max(0.5, 90.0 / scene.meters_per_unit), (1.0, 0.20, 0.02))
    _set_opacity(fire, 0.72)
    _bind_preview_material(stage, fire, f"{root_path}/DestroyedFireMaterial", (1.0, 0.20, 0.02), 0.72)
    smoke = _create_cylinder(stage, f"{root_path}/DestroyedSmokeColumn", max(0.7, 110.0 / scene.meters_per_unit), max(2.0, 380.0 / scene.meters_per_unit), (0.0, 0.0, marker_z + max(1.0, 190.0 / scene.meters_per_unit)), (0.08, 0.08, 0.08))
    _set_opacity(smoke, 0.38)
    _bind_preview_material(stage, smoke, f"{root_path}/DestroyedSmokeMaterial", (0.08, 0.08, 0.08), 0.38)
    _create_box(stage, f"{root_path}/DestroyedX_A", (0.0, 0.0, marker_z + max(0.8, 130.0 / scene.meters_per_unit)), (max(1.6, 260.0 / scene.meters_per_unit), max(0.12, 18.0 / scene.meters_per_unit), max(0.12, 18.0 / scene.meters_per_unit)), (1.0, 0.08, 0.02), rotate_xyz=(0.0, 0.0, 45.0))
    _create_box(stage, f"{root_path}/DestroyedX_B", (0.0, 0.0, marker_z + max(0.82, 132.0 / scene.meters_per_unit)), (max(1.6, 260.0 / scene.meters_per_unit), max(0.12, 18.0 / scene.meters_per_unit), max(0.12, 18.0 / scene.meters_per_unit)), (1.0, 0.08, 0.02), rotate_xyz=(0.0, 0.0, -45.0))
    print(f"[QL] Target destroyed: {target.target_id} category={target.spec.category} cause={cause} pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")


def _dim_destroyed_target(stage, root_prim) -> None:
    root_path = str(root_prim.GetPath())
    for suffix in ("Hull", "Turret", "Gun", "TrackL", "TrackR"):
        prim = stage.GetPrimAtPath(f"{root_path}/{suffix}")
        if prim and prim.IsValid():
            _set_display_color(prim, (0.05, 0.05, 0.045))
    for suffix in ("HighVisibilityBeacon", "EnemyPanel", "LocatorRing", "RoleMarkerArmor", "HeatMarker"):
        prim = stage.GetPrimAtPath(f"{root_path}/{suffix}")
        if prim and prim.IsValid():
            _set_visibility(prim, False)


def _setup_stage(stage, meters_per_unit: float) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))

    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    light.CreateIntensityAttr(1800.0)
    _set_xform(light.GetPrim(), rotate_xyz=(-42.0, 0.0, 35.0))

    physics_scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)


def _create_roots(stage) -> None:
    from pxr import Sdf, UsdGeom

    for path in (
        "/World/AirCombat",
        "/World/AirCombat/Platforms",
        "/World/AirCombat/Targets",
        "/World/AirCombat/Routes",
        "/World/AirCombat/SensorRings",
        "/World/AirCombat/SensorCones",
        "/World/AirCombat/Bases",
        "/World/AirCombat/MissionDemo",
        "/World/AirCombat/ObstacleOverlays",
        "/World/AirCombat/Materials",
        "/World/AirCombat/Weather",
        "/World/AirCombat/Weather/CloudZones",
        "/World/AirCombat/Weather/RainZones",
    ):
        UsdGeom.Xform.Define(stage, Sdf.Path(path))


def _create_mountain_terrain(stage, cfg: dict[str, Any], map_size: float, height_scale: float, grid: int) -> None:
    from pxr import Gf, Sdf, UsdGeom

    terrain_usd = os.environ.get("QL_AIR_COMBAT_TERRAIN_USD") or cfg.get("map", {}).get("terrain_usd")
    if terrain_usd:
        usd_path = Path(str(terrain_usd)).expanduser()
        if usd_path.exists() and _add_usd_reference(stage, str(usd_path), "/World/AirCombat/TerrainUsd"):
            terrain_root = stage.GetPrimAtPath("/World/AirCombat/TerrainUsd")
            _set_xform(terrain_root, scale=(map_size, map_size, max(1.0, height_scale)))
            _apply_collision_api(terrain_root, visible=True)
            print(f"[QL] Loaded terrain USD: {usd_path}")
            return
        print(f"[QL][WARN] Terrain USD not found or failed to load: {terrain_usd}; using procedural grid terrain.")

    half = map_size * 0.5
    step = map_size / max(2, grid)
    points = []
    for iy in range(grid + 1):
        y = -half + iy * step
        for ix in range(grid + 1):
            x = -half + ix * step
            points.append(Gf.Vec3f(float(x), float(y), float(terrain_height(x, y, map_size, height_scale))))

    face_counts = []
    face_indices = []
    for iy in range(grid):
        for ix in range(grid):
            i0 = iy * (grid + 1) + ix
            face_counts.append(4)
            face_indices.extend([i0, i0 + 1, i0 + grid + 2, i0 + grid + 1])

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/AirCombat/Terrain"))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.30, 0.19)])
    _apply_collision_api(mesh.GetPrim(), visible=True)
    meters_per_unit = float(cfg.get("map", {}).get("meters_per_unit", 100.0))
    _create_terrain_biome_overlays(stage, cfg, map_size, height_scale, meters_per_unit)


def _create_terrain_grid(stage, map_size: float, height_scale: float, spacing_km: float, meters_per_unit: float) -> None:
    from pxr import Gf, Sdf, UsdGeom

    spacing_units = max(1.0, spacing_km * 1000.0 / meters_per_unit)
    half = map_size * 0.5
    line_values = []
    v = -half
    while v <= half + 1e-6:
        line_values.append(v)
        v += spacing_units

    points = []
    counts = []
    samples = 80
    lift = max(0.03, height_scale * 0.004)
    for x in line_values:
        counts.append(samples + 1)
        for i in range(samples + 1):
            y = -half + map_size * i / samples
            points.append(Gf.Vec3f(float(x), float(y), float(terrain_height(x, y, map_size, height_scale) + lift)))
    for y in line_values:
        counts.append(samples + 1)
        for i in range(samples + 1):
            x = -half + map_size * i / samples
            points.append(Gf.Vec3f(float(x), float(y), float(terrain_height(x, y, map_size, height_scale) + lift)))

    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path("/World/AirCombat/TerrainGrid"))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr(counts)
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([0.9] * len(points))
    curves.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.10, 0.04)])


def _create_terrain_biome_overlays(stage, cfg: dict[str, Any], map_size: float, height_scale: float, meters_per_unit: float) -> None:
    half = map_size * 0.5
    rng = np.random.default_rng(42)

    obstacle_cfg = cfg.get("obstacles", {})
    if bool(obstacle_cfg.get("enabled", True)) and bool(obstacle_cfg.get("forests", True)):
        _create_procedural_forest_zones(stage, cfg, map_size, height_scale, meters_per_unit)

    for idx in range(96):
        x = float(rng.uniform(-half * 0.82, half * 0.82))
        y = float(rng.uniform(-half * 0.82, half * 0.82))
        z = terrain_height(x, y, map_size, height_scale)
        if z < height_scale * 0.32:
            continue
        radius = float(rng.uniform(1.0, 4.2))
        color = (0.25, 0.24, 0.21) if z < height_scale * 0.70 else (0.57, 0.58, 0.54)
        _create_sphere(stage, f"/World/AirCombat/TerrainRock_{idx}", (x, y, z + radius * 0.25), radius, color)


def _create_procedural_forest_zones(stage, cfg: dict[str, Any], map_size: float, height_scale: float, meters_per_unit: float) -> None:
    obstacle_cfg = cfg.get("obstacles", {})
    tree_samples = _forest_tree_samples(cfg, map_size, height_scale)
    if not tree_samples:
        return
    visual_scale = max(1.0, float(obstacle_cfg.get("forest_tree_visual_scale", 220.0)))
    created = 0
    for sample in tree_samples:
        _spawn_air_combat_tree(
            stage,
            f"/World/AirCombat/ObstacleOverlays/{sample['tree_id']}",
            float(sample["x"]),
            float(sample["y"]),
            float(sample["ground_z"]),
            meters_per_unit,
            visual_scale,
            sample,
        )
        created += 1
    print(f"[QL] Air-combat forest ready: zones={len(FOREST_ZONE_SPECS)}, trees={created}")


def _forest_tree_samples(cfg: dict[str, Any], map_size: float, height_scale: float) -> list[dict[str, Any]]:
    obstacle_cfg = cfg.get("obstacles", {})
    total_count = max(0, int(obstacle_cfg.get("forest_tree_count", 140)))
    if total_count <= 0:
        return []
    rng = np.random.default_rng(73)
    per_zone = max(1, total_count // max(1, len(FOREST_ZONE_SPECS)))
    samples: list[dict[str, Any]] = []
    half = map_size * 0.5
    for zone_idx, (name, nx, ny, radius_ratio, _canopy_height_m) in enumerate(FOREST_ZONE_SPECS):
        center_x = float(nx * half)
        center_y = float(ny * half)
        zone_radius = max(2.0, map_size * float(radius_ratio))
        zone_count = per_zone + (1 if zone_idx < total_count % max(1, len(FOREST_ZONE_SPECS)) else 0)
        for local_idx in range(zone_count):
            angle = float(rng.uniform(0.0, math.tau))
            radius = zone_radius * math.sqrt(float(rng.uniform(0.0, 1.0)))
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            ground_z = terrain_height(x, y, map_size, height_scale)
            if ground_z > height_scale * 0.92:
                ground_z = terrain_height(center_x, center_y, map_size, height_scale)
            tree_kind = "pine" if float(rng.uniform(0.0, 1.0)) < 0.58 else "broadleaf"
            trunk_h_m = float(rng.uniform(2.8, 5.0))
            crown_radius_m = float(rng.uniform(1.15, 1.75) if tree_kind == "pine" else rng.uniform(1.0, 1.55))
            samples.append(
                {
                    "tree_id": f"{name}_Tree_{local_idx}",
                    "zone": name,
                    "x": x,
                    "y": y,
                    "ground_z": ground_z,
                    "kind": tree_kind,
                    "trunk_h_m": trunk_h_m,
                    "trunk_r_m": float(rng.uniform(0.14, 0.24)),
                    "crown_radius_m": crown_radius_m,
                    "height_m": trunk_h_m + crown_radius_m * (2.2 if tree_kind == "pine" else 1.35),
                }
            )
    return samples


def _spawn_air_combat_tree(stage, root_path: str, x: float, y: float, ground_z: float, meters_per_unit: float, visual_scale: float, sample: dict[str, Any]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    size_scale = visual_scale / meters_per_unit
    seed_text = str(sample.get("tree_id", root_path))
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(seed_text)) % (2**32)
    rng = np.random.default_rng(seed)
    trunk_h = float(sample.get("trunk_h_m", 3.6)) * size_scale
    trunk_r = float(sample.get("trunk_r_m", 0.18)) * size_scale
    tree_kind = str(sample.get("kind", "pine"))

    _create_cylinder(
        stage,
        f"{root_path}_Trunk",
        trunk_r,
        trunk_h,
        (x, y, ground_z + trunk_h * 0.5),
        (0.30, 0.17, 0.08),
    )

    if tree_kind == "pine":
        for level in range(3):
            cone_h = float(rng.uniform(2.0, 3.0)) * (1.0 - level * 0.12) * size_scale
            cone_r = float(sample.get("crown_radius_m", 1.35)) * (1.0 - level * 0.18) * size_scale
            cone_z = ground_z + trunk_h * 0.55 + level * cone_h * 0.48
            cone = UsdGeom.Cone.Define(stage, Sdf.Path(f"{root_path}_Crown_{level}"))
            cone.CreateRadiusAttr(float(cone_r))
            cone.CreateHeightAttr(float(cone_h))
            UsdGeom.Xformable(cone.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(cone_z)))
            _set_display_color(cone.GetPrim(), (0.03, float(rng.uniform(0.26, 0.38)), 0.10))
    else:
        offsets = ((0.0, 0.0), (-0.55, 0.25), (0.52, -0.20))
        for level, offset in enumerate(offsets):
            crown_r = float(sample.get("crown_radius_m", 1.25)) * float(rng.uniform(0.88, 1.10)) * size_scale
            crown_x = x + offset[0] * size_scale
            crown_y = y + offset[1] * size_scale
            crown_z = ground_z + trunk_h + crown_r * 0.45
            _create_sphere(
                stage,
                f"{root_path}_Crown_{level}",
                (crown_x, crown_y, crown_z),
                crown_r,
                (0.04, float(rng.uniform(0.30, 0.48)), 0.12),
            )


def _create_contour_lines(stage, map_size: float, height_scale: float) -> None:
    from pxr import Gf, Sdf, UsdGeom

    half = map_size * 0.5
    sample_count = 120
    levels = [height_scale * ratio for ratio in (0.16, 0.28, 0.40, 0.52, 0.64, 0.76, 0.88)]
    points = []
    counts = []
    width_values = []
    for level_idx, level in enumerate(levels):
        segments = _marching_squares_segments(map_size, height_scale, sample_count, level)
        for (x1, y1), (x2, y2) in segments:
            z = level + 0.10 + level_idx * 0.004
            points.append(Gf.Vec3f(float(x1), float(y1), float(z)))
            points.append(Gf.Vec3f(float(x2), float(y2), float(z)))
            counts.append(2)
            width_values.extend([1.8, 1.8])

    if not points:
        return
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path("/World/AirCombat/TerrainContours"))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr(counts)
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr(width_values)
    curves.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.46, 0.25)])


def _create_ridge_wireframe(stage, map_size: float, height_scale: float) -> None:
    from pxr import Gf, Sdf, UsdGeom

    half = map_size * 0.5
    samples = 112
    stride = max(1, samples // 18)
    points = []
    counts = []
    widths = []
    lift = max(0.25, height_scale * 0.012)

    for iy in range(0, samples + 1, stride):
        y = -half + map_size * iy / samples
        counts.append(samples + 1)
        for ix in range(samples + 1):
            x = -half + map_size * ix / samples
            z = terrain_height(x, y, map_size, height_scale) + lift
            points.append(Gf.Vec3f(float(x), float(y), float(z)))
            widths.append(2.2)

    for ix in range(0, samples + 1, stride):
        x = -half + map_size * ix / samples
        counts.append(samples + 1)
        for iy in range(samples + 1):
            y = -half + map_size * iy / samples
            z = terrain_height(x, y, map_size, height_scale) + lift
            points.append(Gf.Vec3f(float(x), float(y), float(z)))
            widths.append(2.2)

    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path("/World/AirCombat/TerrainWireframe"))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr(counts)
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr(widths)
    curves.CreateDisplayColorAttr([Gf.Vec3f(0.01, 0.16, 0.05)])


def _create_cloud_layer(stage, cfg: dict[str, Any], map_size: float, height_scale: float, meters_per_unit: float) -> None:
    rng = np.random.default_rng(7)
    weather_cfg = cfg.get("weather", {})
    count = int(weather_cfg.get("cloud_count", 22))
    opacity = max(0.02, min(0.35, float(weather_cfg.get("cloud_opacity", 0.12))))
    altitude = float(weather_cfg.get("cloud_altitude_m", 5200.0)) / meters_per_unit
    half = map_size * 0.5
    zones = _weather_zones(map_size, meters_per_unit) if bool(weather_cfg.get("local_zones", True)) else ()
    for idx in range(count):
        if zones:
            zone = zones[idx % len(zones)]
            x, y = _sample_weather_zone_point(rng, zone)
        else:
            x = float(rng.uniform(-half * 0.82, half * 0.82))
            y = float(rng.uniform(-half * 0.82, half * 0.82))
            zone = None
        z = max(height_scale * 1.06, altitude + float(rng.uniform(-5.0, 5.0)))
        sx = float(rng.uniform(34.0, 88.0))
        sy = float(rng.uniform(16.0, 46.0))
        sz = float(rng.uniform(2.0, 7.0))
        color = zone["cloud_color"] if zone else (0.86, 0.88, 0.88)
        local_opacity = min(0.35, opacity * (zone["cloud_opacity_mul"] if zone else 1.0))
        prim = _create_sphere(stage, f"/World/AirCombat/Weather/CloudZones/Cloud_{idx}", (x, y, z), 1.0, color)
        _set_xform(prim, scale=(sx, sy, sz))
        _set_opacity(prim, local_opacity)
        _bind_preview_material(stage, prim, f"/World/AirCombat/Weather/CloudZones/CloudMaterial_{idx}", color, local_opacity)
    if bool(weather_cfg.get("rain", True)):
        _create_rain_zones(stage, cfg, map_size, height_scale, meters_per_unit, zones)


def _weather_zones(map_size: float, meters_per_unit: float) -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "NorthCloudDeck",
            "center": _scale_path([(36, 78)], meters_per_unit)[0],
            "radius": map_size * 0.13,
            "cloud_color": (0.74, 0.78, 0.80),
            "cloud_opacity_mul": 1.15,
            "rain": False,
        },
        {
            "name": "EastRainCell",
            "center": _scale_path([(88, -38)], meters_per_unit)[0],
            "radius": map_size * 0.11,
            "cloud_color": (0.58, 0.62, 0.65),
            "cloud_opacity_mul": 1.55,
            "rain": True,
        },
        {
            "name": "ValleyBrokenCloud",
            "center": _scale_path([(-44, 18)], meters_per_unit)[0],
            "radius": map_size * 0.09,
            "cloud_color": (0.84, 0.86, 0.84),
            "cloud_opacity_mul": 0.80,
            "rain": False,
        },
    )


def _sample_weather_zone_point(rng: np.random.Generator, zone: dict[str, Any]) -> tuple[float, float]:
    cx, cy = zone["center"]
    radius = float(zone["radius"])
    angle = float(rng.uniform(0.0, math.tau))
    distance = radius * math.sqrt(float(rng.uniform(0.0, 1.0)))
    return float(cx + math.cos(angle) * distance), float(cy + math.sin(angle) * distance)


def _create_rain_zones(stage, cfg: dict[str, Any], map_size: float, height_scale: float, meters_per_unit: float, zones: tuple[dict[str, Any], ...]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    weather_cfg = cfg.get("weather", {})
    rain_count = int(weather_cfg.get("rain_count", 120))
    opacity = max(0.05, min(0.60, float(weather_cfg.get("rain_opacity", 0.24))))
    rng = np.random.default_rng(19)
    active_zones = [zone for zone in zones if zone.get("rain")] if zones else [
        {"name": "RainCell", "center": _scale_path([(70, -36)], meters_per_unit)[0], "radius": map_size * 0.11}
    ]
    if not active_zones or rain_count <= 0:
        return

    points = []
    counts = []
    widths = []
    for idx in range(rain_count):
        zone = active_zones[idx % len(active_zones)]
        x, y = _sample_weather_zone_point(rng, zone)
        ground_z = terrain_height(x, y, map_size, height_scale)
        top_z = max(height_scale * 0.72, ground_z + float(rng.uniform(14.0, 28.0)))
        drop_len = float(rng.uniform(5.0, 14.0))
        slant_x = float(rng.uniform(-1.4, -0.3))
        slant_y = float(rng.uniform(0.2, 1.2))
        points.append(Gf.Vec3f(float(x), float(y), float(top_z)))
        points.append(Gf.Vec3f(float(x + slant_x), float(y + slant_y), float(max(ground_z + 0.25, top_z - drop_len))))
        counts.append(2)
        widths.extend([0.65, 0.65])

    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path("/World/AirCombat/Weather/RainZones/RainStreaks"))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr(counts)
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr(widths)
    curves.CreateDisplayColorAttr([Gf.Vec3f(0.46, 0.62, 0.78)])
    prim = curves.GetPrim()
    _set_opacity(prim, opacity)
    _bind_preview_material(stage, prim, "/World/AirCombat/Weather/RainZones/RainMaterial", (0.46, 0.62, 0.78), opacity)


def _create_forward_bases(stage, cfg: dict[str, Any], map_size: float, height_scale: float, meters_per_unit: float) -> None:
    blue_xy = _scale_path([(-128, -128)], meters_per_unit)[0]
    red_xy = _scale_path([(128, 112)], meters_per_unit)[0]
    visual_scale = float(cfg.get("visual", {}).get("base_visual_scale", 5.0))
    _create_base(stage, "/World/AirCombat/Bases/BlueBase", "Blue", blue_xy, map_size, height_scale, meters_per_unit, visual_scale)
    _create_base(stage, "/World/AirCombat/Bases/RedBase", "Red", red_xy, map_size, height_scale, meters_per_unit, visual_scale)


def _create_base(stage, root_path: str, faction: str, xy: tuple[float, float], map_size: float, height_scale: float, meters_per_unit: float, visual_scale: float) -> None:
    x, y = xy
    size = 1800.0 * max(1.0, float(visual_scale)) / meters_per_unit
    z = _base_platform_height(x, y, size, map_size, height_scale) + 0.22
    color = (0.13, 0.30, 0.42) if faction == "Blue" else (0.42, 0.18, 0.12)
    pad_color = (0.20, 0.22, 0.21)
    earth_color = (0.24, 0.27, 0.20)
    accent = (0.05, 0.60, 1.0) if faction == "Blue" else (1.0, 0.18, 0.05)

    _create_box(stage, f"{root_path}/EarthworkPad", (x, y, z - 0.50), (size * 1.28, size * 0.92, 1.0), earth_color)
    _create_box(stage, f"{root_path}/BasePad", (x, y, z), (size, size * 0.68, 0.18), pad_color)
    _create_box(stage, f"{root_path}/PerimeterNorth", (x, y + size * 0.39, z + 0.32), (size * 1.10, size * 0.035, 0.52), (0.08, 0.09, 0.08))
    _create_box(stage, f"{root_path}/PerimeterSouth", (x, y - size * 0.39, z + 0.32), (size * 1.10, size * 0.035, 0.52), (0.08, 0.09, 0.08))
    _create_box(stage, f"{root_path}/PerimeterWest", (x - size * 0.56, y, z + 0.32), (size * 0.035, size * 0.76, 0.52), (0.08, 0.09, 0.08))
    _create_box(stage, f"{root_path}/PerimeterEast", (x + size * 0.56, y, z + 0.32), (size * 0.035, size * 0.76, 0.52), (0.08, 0.09, 0.08))
    _create_box(stage, f"{root_path}/Taxiway", (x, y - size * 0.06, z + 0.18), (size * 0.78, size * 0.08, 0.08), (0.12, 0.13, 0.12))
    _create_box(stage, f"{root_path}/CenterLine", (x, y - size * 0.06, z + 0.26), (size * 0.70, size * 0.012, 0.04), (0.92, 0.86, 0.26))
    for idx, dx in enumerate((-0.34, -0.12, 0.12, 0.34)):
        px = x + size * dx
        py = y - size * 0.24
        _create_cylinder(stage, f"{root_path}/Helipad_{idx}", size * 0.058, 0.10, (px, py, z + 0.25), (0.18, 0.20, 0.20))
        _create_box(stage, f"{root_path}/HelipadMark_{idx}", (px, py, z + 0.33), (size * 0.09, size * 0.010, 0.035), accent)
        _create_box(stage, f"{root_path}/HelipadMarkCross_{idx}", (px, py, z + 0.34), (size * 0.010, size * 0.09, 0.035), accent)
    for idx, (dx, dy, sx, sy) in enumerate(((-0.34, 0.22, 0.16, 0.09), (-0.12, 0.23, 0.15, 0.08), (0.17, 0.24, 0.18, 0.10), (0.36, 0.12, 0.13, 0.08))):
        _create_box(stage, f"{root_path}/Shelter_{idx}", (x + size * dx, y + size * dy, z + 1.25), (size * sx, size * sy, 2.4), color)
        _create_box(stage, f"{root_path}/ShelterRoof_{idx}", (x + size * dx, y + size * dy, z + 2.55), (size * sx * 1.08, size * sy * 1.10, 0.22), (0.12, 0.13, 0.11))
    for idx, dx in enumerate((-0.44, -0.36, 0.42, 0.48)):
        _create_box(stage, f"{root_path}/VehicleRevetment_{idx}", (x + size * dx, y - size * 0.02, z + 0.42), (size * 0.045, size * 0.09, 0.40), (0.16, 0.18, 0.13))
    _create_cylinder(stage, f"{root_path}/CommsMast", size * 0.007, 18.0, (x - size * 0.50, y + size * 0.28, z + 9.2), accent)
    _create_sphere(stage, f"{root_path}/Beacon", (x - size * 0.50, y + size * 0.28, z + 18.6), size * 0.018, accent)
    _create_sphere(stage, f"{root_path}/BaseLocator", (x, y, z + 7.0), size * 0.025, accent)


def _base_platform_height(x: float, y: float, size: float, map_size: float, height_scale: float) -> float:
    samples = [
        (x, y),
        (x - size * 0.58, y - size * 0.40),
        (x - size * 0.58, y + size * 0.40),
        (x + size * 0.58, y - size * 0.40),
        (x + size * 0.58, y + size * 0.40),
        (x - size * 0.30, y),
        (x + size * 0.30, y),
    ]
    return max(terrain_height(px, py, map_size, height_scale) for px, py in samples)


def _create_mission_corridors(stage, map_size: float, height_scale: float, meters_per_unit: float) -> None:
    routes = {
        "BlueWZ21Ingress": (_scale_path([(-146, -118), (-134, -104), (-112, -84), (-82, -58), (-48, -32), (-14, -12), (28, 4), (62, 18)], meters_per_unit), (0.10, 0.48, 0.18)),
        "ReconUAVSouth": (_scale_path([(-150, -58), (-118, -44), (-78, -24), (-32, 0), (18, 28), (56, 66), (18, 92), (-48, 74), (-104, 28)], meters_per_unit), (0.35, 0.24, 0.92)),
        "StrikeReconSouth": (_scale_path([(-150, -86), (-112, -72), (-68, -52), (-22, -28), (24, -2), (54, 24), (36, 48), (-8, 32), (-58, -10)], meters_per_unit), (0.92, 0.62, 0.08)),
        "ReconUAVNorth": (_scale_path([(-150, 26), (-118, 44), (-78, 62), (-28, 78), (24, 72), (72, 42), (90, -6), (46, -38), (-10, -24)], meters_per_unit), (0.45, 0.32, 1.00)),
        "StrikeReconNorth": (_scale_path([(-150, -16), (-116, -4), (-74, 16), (-28, 34), (24, 40), (66, 18), (78, -24), (34, -48), (-22, -42)], meters_per_unit), (1.00, 0.72, 0.12)),
    }
    for name, (points, color) in routes.items():
        _create_route_curve(stage, f"/World/AirCombat/Routes/{name}", points, map_size, height_scale, color)


def _create_route_curve(stage, path: str, points_xy: list[tuple[float, float]], map_size: float, height_scale: float, color: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    points = [Gf.Vec3f(float(x), float(y), float(terrain_height(x, y, map_size, height_scale) + 0.12)) for x, y in points_xy]
    if points:
        points.append(points[0])
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr([len(points)])
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([2.0] * len(points))
    curves.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _create_sensor_ring(stage, path: str, center: np.ndarray, radius_units: float, color: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    if radius_units <= 0.0:
        return
    segments = 128
    points = []
    z = float(center[2]) + 0.04
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        points.append(Gf.Vec3f(float(center[0] + math.cos(a) * radius_units), float(center[1] + math.sin(a) * radius_units), z))
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr([len(points)])
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([1.25] * len(points))
    curves.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def _create_sensor_cones(stage, root_path: str, center: np.ndarray, spec: PlatformSpec, meters_per_unit: float) -> None:
    for idx, sensor in enumerate(spec.sensors):
        if sensor.channel == "elint":
            continue
        radius_units = min(sensor.max_range_km * 1000.0 / meters_per_unit, 1000.0)
        color = _sensor_color(sensor.channel, spec.faction)
        yaw_center = _sensor_value(sensor, "boresight_yaw_offset_deg", 0.0)
        half_fov = min(180.0, _sensor_value(sensor, "azimuth_fov_deg", 360.0) * 0.5)
        z = float(center[2]) + 0.18 + idx * 0.08
        path = f"{root_path}_{sensor.channel}_{idx}"
        _create_sector_curve(stage, path, center[:2], z, radius_units, yaw_center, half_fov, color)


def _sensor_color(channel: str, faction: str) -> tuple[float, float, float]:
    if channel == "eo_ir":
        return (0.10, 0.78, 0.95) if faction == "Blue" else (1.0, 0.36, 0.12)
    if channel == "sar":
        return (0.95, 0.80, 0.18)
    if channel == "mmw_radar":
        return (0.18, 0.95, 0.42) if faction == "Blue" else (1.0, 0.12, 0.08)
    return (0.85, 0.85, 0.85)


def _create_sector_curve(
    stage,
    path: str,
    center_xy,
    z: float,
    radius: float,
    yaw_center_deg: float,
    half_fov_deg: float,
    color: tuple[float, float, float],
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    segments = 36
    points = [Gf.Vec3f(float(center_xy[0]), float(center_xy[1]), float(z))]
    for i in range(segments + 1):
        a = math.radians(float(yaw_center_deg) - half_fov_deg + 2.0 * half_fov_deg * i / segments)
        points.append(Gf.Vec3f(float(center_xy[0] + math.cos(a) * radius), float(center_xy[1] + math.sin(a) * radius), float(z)))
    points.append(Gf.Vec3f(float(center_xy[0]), float(center_xy[1]), float(z)))
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr([len(points)])
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([2.2] * len(points))
    curves.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _update_ground_target(scene: AirCombatSceneState, target: GroundTarget, dt: float) -> None:
    length = _polyline_length(target.route_xy, closed=True)
    if length <= 1e-6:
        return
    target.route_phase = (target.route_phase + (target.route_speed_units * dt) / length) % 1.0
    x, y = _polyline_point(target.route_xy, target.route_phase, closed=True)
    next_x, next_y = _polyline_point(target.route_xy, (target.route_phase + 0.01) % 1.0, closed=True)
    offset_m = _ground_target_surface_offset_m(scene, target.spec)
    visual_cfg = scene.config.get("visual", {})
    if target.spec.category == "armor":
        visual_scale = float(visual_cfg.get("armor_visual_scale", visual_cfg.get("ground_target_scale", 10.0)))
        dims_units = _scaled_dims(target.spec.dimensions_m, scene.meters_per_unit, visual_scale)
        pos = _ground_position_for_dimensions(x, y, scene.map_size_units, scene.terrain_visual_height_units, offset_m, scene.meters_per_unit, dims_units)
        max_vertical_step = max(0.12, target.route_speed_units * dt * 0.45)
        dz = float(pos[2] - target.position[2])
        if abs(dz) > max_vertical_step:
            pos[2] = float(target.position[2] + math.copysign(max_vertical_step, dz))
        min_ground_z = terrain_height(x, y, scene.map_size_units, scene.terrain_visual_height_units) + offset_m / scene.meters_per_unit
        pos[2] = max(float(pos[2]), min_ground_z)
    else:
        pos = _ground_position(x, y, scene.map_size_units, scene.terrain_visual_height_units, offset_m, scene.meters_per_unit)
    target.position = pos
    yaw = math.degrees(math.atan2(next_y - y, next_x - x))
    _set_root_pose(target.root_prim, pos, MotionState(position=pos, velocity=np.zeros(3, dtype=float), yaw_deg=yaw))


def _ground_target_surface_offset_m(scene: AirCombatSceneState, spec: GroundTargetSpec) -> float:
    visual_cfg = scene.config.get("visual", {})
    if spec.category == "armor":
        visual_scale = float(visual_cfg.get("armor_visual_scale", visual_cfg.get("ground_target_scale", 10.0)))
    else:
        visual_scale = float(visual_cfg.get("ground_target_scale", 10.0))
    _, _, height_units = _scaled_dims(spec.dimensions_m, scene.meters_per_unit, visual_scale)
    support_factor = 0.19 if spec.category == "armor" else 0.12
    return max(1.2, height_units * support_factor * scene.meters_per_unit)


def _select_striker(scene: AirCombatSceneState, target: GroundTarget, platforms: list[AirPlatform]) -> tuple[AirPlatform | None, str | None, float | None]:
    best: tuple[float, AirPlatform, str, float] | None = None
    for platform in platforms:
        distance_km = float(np.linalg.norm(platform.position - target.position)) * scene.meters_per_unit / 1000.0
        for missile in platform.spec.weapons.missiles:
            if distance_km > missile.max_range_km:
                continue
            if target.spec.category in {"base", "command_post", "radar"} and "air-to-air" in missile.role:
                continue
            if target.spec.category == "armor" and "armor" not in missile.role and missile.armor_penetration_mm_rha is None:
                continue
            margin = missile.max_range_km - distance_km
            score = target.spec.priority + margin * 0.5
            if best is None or score > best[0]:
                best = (score, platform, missile.name, margin)
        for gun in platform.spec.weapons.guns:
            if distance_km <= gun.effective_range_km:
                margin = gun.effective_range_km - distance_km
                score = target.spec.priority + margin * 0.2
                if best is None or score > best[0]:
                    best = (score, platform, gun.name, margin)
    if best is None:
        return None, None, None
    return best[1], best[2], best[3]


def _terrain_masks_line(scene: AirCombatSceneState, observer: np.ndarray, target: np.ndarray) -> bool:
    samples = 18
    clearance = max(0.08, 40.0 / scene.meters_per_unit)
    for idx in range(1, samples):
        alpha = idx / samples
        p = observer * (1.0 - alpha) + target * alpha
        los_z = float(p[2])
        ground_z = scene.surface_height(float(p[0]), float(p[1]))
        if ground_z + clearance > los_z:
            return True
    return False


def _sensor_accepts_target(target_kinds: tuple[str, ...], category: str) -> bool:
    if category in target_kinds:
        return True
    if category == "base" and "command_post" in target_kinds:
        return True
    if category == "armor" and "vehicle" in target_kinds:
        return True
    if category == "radar" and "command_post" in target_kinds:
        return True
    return False


def _sensor_scan_contains(scene: AirCombatSceneState, platform: AirPlatform, sensor, delta: np.ndarray) -> bool:
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        return True

    horizontal = math.hypot(float(delta[0]), float(delta[1]))
    target_azimuth = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    target_elevation = math.degrees(math.atan2(float(delta[2]), max(1e-6, horizontal)))

    azimuth_fov = _sensor_value(sensor, "azimuth_fov_deg", 360.0)
    elevation_fov = _sensor_value(sensor, "elevation_fov_deg", 60.0)
    scan_rate_default = 0.25 if getattr(sensor, "channel", "") == "elint" else 1.0
    scan_rate = _sensor_value(sensor, "scan_rate_hz", scan_rate_default)
    dwell_time = _sensor_value(sensor, "dwell_time_s", 0.25)
    yaw_center = float(platform.motion_model.state.yaw_deg) + _sensor_value(sensor, "boresight_yaw_offset_deg", 0.0)
    if azimuth_fov >= 359.0:
        az_ok = True
    else:
        sweep_center = yaw_center
        scan_rate = max(0.0, scan_rate)
        if scan_rate > 0.0:
            sweep_center += 0.5 * azimuth_fov * math.sin(2.0 * math.pi * scan_rate * scene.tactical_time_s)
        az_err = _angle_error_deg(target_azimuth, sweep_center)
        az_ok = abs(az_err) <= max(0.5, azimuth_fov * 0.5)

    pitch_center = _sensor_value(sensor, "boresight_pitch_deg", -6.0)
    el_ok = abs(target_elevation - pitch_center) <= max(0.5, elevation_fov * 0.5)

    if not (az_ok and el_ok):
        return False

    scan_period = 1.0 / max(0.05, scan_rate)
    scan_phase = scene.tactical_time_s % scan_period
    return scan_phase <= max(dwell_time, scan_period * 0.35)


def _angle_error_deg(angle: float, center: float) -> float:
    return ((float(angle) - float(center) + 180.0) % 360.0) - 180.0


def _sensor_value(sensor, name: str, default: float) -> float:
    try:
        return float(getattr(sensor, name))
    except Exception:
        return float(default)


def _make_air_reference(
    points_xy: list[tuple[float, float]],
    altitude_agl_m: float,
    map_size: float,
    height_scale: float,
    meters_per_unit: float,
    spec: PlatformSpec,
    clearance_units: float,
) -> Callable[[float], np.ndarray]:
    altitude_units = altitude_agl_m / meters_per_unit
    cruise_units = spec.cruise_speed_mps / meters_per_unit
    service_ceiling_units = spec.service_ceiling_m / meters_per_unit
    length = max(1.0, _polyline_length(points_xy, closed=True))

    def _reference(t: float) -> np.ndarray:
        phase = ((float(t) * cruise_units) / length) % 1.0
        x, y = _polyline_point(points_xy, phase, closed=True)
        lookahead_phase = (phase + min(0.08, (cruise_units * 45.0) / length)) % 1.0
        lookahead_x, lookahead_y = _polyline_point(points_xy, lookahead_phase, closed=True)
        local_peak = _max_terrain_height_between(x, y, lookahead_x, lookahead_y, map_size, height_scale)
        ground_z = terrain_height(x, y, map_size, height_scale)
        desired_z = max(ground_z + altitude_units, local_peak + clearance_units)
        z = min(desired_z, service_ceiling_units)
        return np.array([x, y, max(local_peak + clearance_units, z)], dtype=float)

    return _reference


def _air_terrain_clearance_units(cfg: dict[str, Any], meters_per_unit: float) -> float:
    clearance_m = float(cfg.get("aircraft", {}).get("terrain_clearance_m", 120.0))
    return max(0.8, clearance_m / float(meters_per_unit))


def _air_platform_clearance_units(cfg: dict[str, Any], spec: PlatformSpec, meters_per_unit: float) -> float:
    visual_scale = float(cfg.get("visual", {}).get("aircraft_scale", 180.0))
    collision_half_height = _m_to_units(max(spec.height_m, spec.rotor_diameter_m * 0.12), meters_per_unit, visual_scale) * 0.5
    return _air_terrain_clearance_units(cfg, meters_per_unit) + collision_half_height


def _clamp_air_platform_above_terrain(pos: np.ndarray, map_size: float, height_scale: float, clearance_units: float) -> np.ndarray:
    safe_pos = np.array(pos, dtype=float)
    ground_z = terrain_height(float(safe_pos[0]), float(safe_pos[1]), map_size, height_scale)
    safe_pos[2] = max(float(safe_pos[2]), ground_z + clearance_units)
    return safe_pos


def _max_terrain_height_between(x1: float, y1: float, x2: float, y2: float, map_size: float, height_scale: float) -> float:
    max_z = terrain_height(x1, y1, map_size, height_scale)
    for idx in range(1, 7):
        ratio = idx / 6.0
        x = x1 + (x2 - x1) * ratio
        y = y1 + (y2 - y1) * ratio
        max_z = max(max_z, terrain_height(x, y, map_size, height_scale))
    return max_z


def _max_terrain_height_under_footprint(x: float, y: float, sx: float, sy: float, map_size: float, height_scale: float) -> float:
    half_x = max(0.35, float(sx) * 0.18)
    half_y = max(0.35, float(sy) * 0.22)
    center_z = terrain_height(x, y, map_size, height_scale)
    samples = (
        (x, y),
        (x - half_x, y - half_y),
        (x - half_x, y + half_y),
        (x + half_x, y - half_y),
        (x + half_x, y + half_y),
        (x - half_x, y),
        (x + half_x, y),
        (x, y - half_y),
        (x, y + half_y),
    )
    footprint_z = max(terrain_height(px, py, map_size, height_scale) for px, py in samples)
    return min(footprint_z, center_z + max(0.015, min(0.025, height_scale * 0.00015)))


def _reference_velocity(fn: Callable[[float], np.ndarray], t: float, eps: float = 0.5) -> np.ndarray:
    return (fn(float(t) + eps) - fn(max(0.0, float(t) - eps))) / (2.0 * eps)


def _ground_position(x: float, y: float, map_size: float, height_scale: float, offset_m: float, meters_per_unit: float) -> np.ndarray:
    return np.array([float(x), float(y), terrain_height(x, y, map_size, height_scale) + offset_m / meters_per_unit], dtype=float)


def _ground_position_for_dimensions(
    x: float,
    y: float,
    map_size: float,
    height_scale: float,
    offset_m: float,
    meters_per_unit: float,
    dims_units: tuple[float, float, float],
) -> np.ndarray:
    sx, sy, _ = dims_units
    ground_z = _max_terrain_height_under_footprint(float(x), float(y), sx, sy, map_size, height_scale)
    return np.array([float(x), float(y), ground_z + offset_m / meters_per_unit], dtype=float)


def _base_target_position(xy: tuple[float, float], map_size: float, height_scale: float, meters_per_unit: float) -> np.ndarray:
    x, y = xy
    return _ground_position(x, y, map_size, height_scale, 8.0, meters_per_unit)


def _scale_path(points_km: list[tuple[float, float]], meters_per_unit: float) -> list[tuple[float, float]]:
    units_per_km = 1000.0 / float(meters_per_unit)
    return [(x * units_per_km, y * units_per_km) for x, y in points_km]


def _polyline_length(points: list[tuple[float, float]], closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    segments = list(zip(points, points[1:]))
    if closed:
        segments.append((points[-1], points[0]))
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in segments)


def _polyline_point(points: list[tuple[float, float]], phase: float, closed: bool = False) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        return points[0]
    total = _polyline_length(points, closed=closed)
    if total <= 1e-9:
        return points[0]
    target = (float(phase) % 1.0 if closed else max(0.0, min(1.0, float(phase)))) * total
    walked = 0.0
    segments = list(zip(points, points[1:]))
    if closed:
        segments.append((points[-1], points[0]))
    for (x1, y1), (x2, y2) in segments:
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len <= 1e-9:
            continue
        if walked + seg_len >= target:
            ratio = (target - walked) / seg_len
            return x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio
        walked += seg_len
    return points[-1]


def _marching_squares_segments(
    map_size: float,
    height_scale: float,
    sample_count: int,
    level: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    half = map_size * 0.5
    step = map_size / max(2, sample_count)
    heights = []
    for iy in range(sample_count + 1):
        row = []
        y = -half + iy * step
        for ix in range(sample_count + 1):
            x = -half + ix * step
            row.append(terrain_height(x, y, map_size, height_scale))
        heights.append(row)

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for iy in range(sample_count):
        y0 = -half + iy * step
        y1 = y0 + step
        for ix in range(sample_count):
            x0 = -half + ix * step
            x1 = x0 + step
            corners = [
                (x0, y0, heights[iy][ix]),
                (x1, y0, heights[iy][ix + 1]),
                (x1, y1, heights[iy + 1][ix + 1]),
                (x0, y1, heights[iy + 1][ix]),
            ]
            edge_points = []
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                ax, ay, ah = corners[a]
                bx, by, bh = corners[b]
                if (ah < level and bh < level) or (ah > level and bh > level) or abs(ah - bh) < 1e-6:
                    continue
                ratio = (level - ah) / (bh - ah)
                if 0.0 <= ratio <= 1.0:
                    edge_points.append((ax + (bx - ax) * ratio, ay + (by - ay) * ratio))
            if len(edge_points) == 2:
                segments.append((edge_points[0], edge_points[1]))
            elif len(edge_points) == 4:
                segments.append((edge_points[0], edge_points[1]))
                segments.append((edge_points[2], edge_points[3]))
    return segments


def _make_root(stage, root_path: str, pos: np.ndarray):
    from pxr import Gf, Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_path)).GetPrim()
    xform = UsdGeom.Xformable(root)
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    return root


def _set_root_pose(root_prim, pos: np.ndarray, state: MotionState) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(root_prim)
    translate = xform.GetTranslateOp()
    if not translate:
        translate = xform.AddTranslateOp()
    rotate = xform.GetRotateXYZOp()
    if not rotate:
        rotate = xform.AddRotateXYZOp()
    translate.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    rotate.Set(Gf.Vec3f(float(state.roll_deg), float(state.pitch_deg), float(state.yaw_deg)))


def _set_root_scale(root_prim, scale: float) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(root_prim)
    op = xform.GetScaleOp() or xform.AddScaleOp()
    op.Set(Gf.Vec3f(float(scale), float(scale), float(scale)))


def _set_xform(prim, translate=None, rotate_xyz=None, scale=None) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    if translate is not None:
        op = xform.GetTranslateOp() or xform.AddTranslateOp()
        op.Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    if rotate_xyz is not None:
        op = xform.GetRotateXYZOp() or xform.AddRotateXYZOp()
        op.Set(Gf.Vec3f(float(rotate_xyz[0]), float(rotate_xyz[1]), float(rotate_xyz[2])))
    if scale is not None:
        op = xform.GetScaleOp() or xform.AddScaleOp()
        op.Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))


def _set_display_color(prim, color: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    if prim.IsA(UsdGeom.Gprim):
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def _set_visibility(prim, visible: bool) -> None:
    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(prim)
    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)


def _apply_collision_api(prim, visible: bool = False) -> None:
    try:
        from pxr import UsdPhysics

        UsdPhysics.CollisionAPI.Apply(prim)
        _set_visibility(prim, visible)
    except Exception as exc:
        print(f"[QL][WARN] Failed to apply collision API to {prim.GetPath()}: {exc}")


def _apply_collision_to_prims(prims, visible: bool = False) -> None:
    for prim in prims:
        _apply_collision_api(prim, visible=visible)


def _set_opacity(prim, opacity: float) -> None:
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Gprim):
        UsdGeom.Gprim(prim).CreateDisplayOpacityAttr([float(opacity)])


def _bind_preview_material(
    stage,
    prim,
    material_path: str,
    diffuse_color: tuple[float, float, float],
    opacity: float,
) -> None:
    try:
        from pxr import Gf, Sdf, UsdShade

        material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
        shader = UsdShade.Shader.Define(stage, Sdf.Path(material_path + "/PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_color))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    except Exception as exc:
        print(f"[QL][WARN] Failed to bind transparent material {material_path}: {exc}")


def _create_box(stage, path: str, translate: tuple[float, float, float], scale: tuple[float, float, float], color: tuple[float, float, float], rotate_xyz: tuple[float, float, float] | None = None):
    from pxr import Sdf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translate=translate, rotate_xyz=rotate_xyz, scale=scale)
    _set_display_color(cube.GetPrim(), color)
    return cube.GetPrim()


def _create_sphere(stage, path: str, translate: tuple[float, float, float], radius: float, color: tuple[float, float, float]):
    from pxr import Sdf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.CreateRadiusAttr(float(radius))
    _set_xform(sphere.GetPrim(), translate=translate)
    _set_display_color(sphere.GetPrim(), color)
    return sphere.GetPrim()


def _create_cylinder(
    stage,
    path: str,
    radius: float,
    height: float,
    translate: tuple[float, float, float],
    color: tuple[float, float, float],
    rotate_xyz: tuple[float, float, float] | None = None,
):
    from pxr import Sdf, UsdGeom

    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    _set_xform(cylinder.GetPrim(), translate=translate, rotate_xyz=rotate_xyz)
    _set_display_color(cylinder.GetPrim(), color)
    return cylinder.GetPrim()


def _create_torus_marker(
    stage,
    path: str,
    translate: tuple[float, float, float],
    major_radius: float,
    minor_radius: float,
    color: tuple[float, float, float],
):
    from pxr import Gf, Sdf, UsdGeom

    segments = 96
    points = []
    counts = []
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        points.append(Gf.Vec3f(float(translate[0] + math.cos(a0) * major_radius), float(translate[1] + math.sin(a0) * major_radius), float(translate[2])))
        points.append(Gf.Vec3f(float(translate[0] + math.cos(a1) * major_radius), float(translate[1] + math.sin(a1) * major_radius), float(translate[2])))
        counts.append(2)
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr(counts)
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([float(minor_radius * 2.0)] * len(points))
    curves.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return curves.GetPrim()


def _spin_rotors(rotor_prims: list[object], t: float) -> None:
    from pxr import Gf, UsdGeom

    angle = (float(t) * 900.0) % 360.0
    for idx, prim in enumerate(rotor_prims):
        xform = UsdGeom.Xformable(prim)
        op = xform.GetRotateXYZOp() or xform.AddRotateXYZOp()
        prim_name = prim.GetName() if hasattr(prim, "GetName") else ""
        if "QuadRotor" in prim_name:
            direction = 1.0 if idx % 2 == 0 else -1.0
            op.Set(Gf.Vec3f(0.0, 0.0, float(angle * direction + idx * 45.0)))
        elif "TailRotor" in prim_name or idx == 2:
            op.Set(Gf.Vec3f(0.0, 90.0, float(angle)))
        else:
            op.Set(Gf.Vec3f(0.0, 0.0, float(angle + idx * 90.0)))


def _scaled_dims(dimensions_m: tuple[float, float, float], meters_per_unit: float, visual_scale: float) -> tuple[float, float, float]:
    return tuple(_m_to_units(value, meters_per_unit, visual_scale) for value in dimensions_m)


def _m_to_units(value_m: float, meters_per_unit: float, visual_scale: float = 1.0) -> float:
    return float(value_m) * float(visual_scale) / float(meters_per_unit)


def _max_sensor_range_units(spec: PlatformSpec, meters_per_unit: float) -> float:
    if not spec.sensors:
        return 0.0
    return max(sensor.max_range_km for sensor in spec.sensors) * 1000.0 / meters_per_unit


def _resolve_asset_path(spec: PlatformSpec) -> str | None:
    raw_path = os.environ.get(spec.env_var or "") if spec.env_var else None
    if not raw_path:
        raw_path = spec.default_usd
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    return str(path) if path.exists() else None


def _add_usd_reference(stage, usd_path: str, prim_path: str) -> bool:
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        return bool(stage.GetPrimAtPath(prim_path))
    except Exception as exc:
        print(f"[QL][WARN] Failed to load USD asset {usd_path}: {exc}")
        return False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = _deep_merge(value, {})
        else:
            merged[key] = value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
