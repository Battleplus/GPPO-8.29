from __future__ import annotations

import math
import random
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from .ground_units import create_ground_units as _create_ground_units
    from .aircraft_motion import BaseMotionModel, MotionState, build_motion_model
    from .weather_effects import create_weather_visuals as _create_weather_visuals
except ImportError:
    from ground_units import create_ground_units as _create_ground_units
    from aircraft_motion import BaseMotionModel, MotionState, build_motion_model
    from weather_effects import create_weather_visuals as _create_weather_visuals


QL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = QL_ROOT.parent

RIVER_CENTER_POINTS: list[tuple[float, float]] = [
    (-300.0, -132.0),
    (-246.0, -118.0),
    (-198.0, -130.0),
    (-148.0, -96.0),
    (-92.0, -82.0),
    (-42.0, -66.0),
    (8.0, -52.0),
    (58.0, -34.0),
    (112.0, -10.0),
    (166.0, 20.0),
    (226.0, 44.0),
    (278.0, 58.0),
    (320.0, 72.0),
]

FAST_UAV_ORBIT_RADIUS = 14.0
FAST_UAV_ORBIT_RATE = 0.45
FAST_UAV_ALTITUDE_OFFSET = 26.0

FAST_HELI_ORBIT_RADIUS = 16.0
FAST_HELI_ORBIT_RATE = 0.35
FAST_HELI_ALTITUDE_OFFSET = 30.0


DEFAULT_SCENE_CONFIG: dict[str, Any] = {
    "terrain": {"map_size": 680.0, "height_scale": 26.0, "grid": 120},
    "forest": {"enabled": True, "tree_count": 160, "seed": 42},
    "rocks": {"enabled": True, "count": 45},
    "water": {
        "enabled": True,
        "river_path": RIVER_CENTER_POINTS,
        "river_width": 26.0,
        "bank_width": 44.0,
        "shore_width": 74.0,
        "sea_width": 90.0,
        "sea_length": 520.0,
        "coastal_enabled": True,
    },
    "battlefield": {"enabled": True, "arterial_width": 14.0},
    "ground_units": {
        "enabled": True,
        "tanks_each_side": 10,
        "submarines_each_side": 0,
        "uuv_count": 0,
        "tank_visual_scale": 1.4,
        "tank_ground_offset": 3.4,
        "tank_motion": {"enabled": False, "type": "dynamic", "max_speed": 7.0, "max_accel": 2.4, "drag": 0.35},
        "submarine_motion": {"enabled": False, "type": "surface", "max_speed": 5.5, "max_accel": 1.2, "drag": 0.55, "current_xy": [0.3, 0.0]},
        "uuv_motion": {"enabled": False, "type": "subsurface", "max_speed": 3.8, "max_accel": 1.1, "buoyancy_gain": 1.2, "vertical_damping": 1.6},
    },
    "emitter": {"position_xy": [18.0, -8.0], "height_offset": 1.2, "radius": 1.4},
    "aircraft": {
        "uav": False,
        "helicopter": False,
        "uav_start": [230.0, -142.0, 24.0],
        "helicopter_start": [-230.0, 142.0, 28.0],
        "formations_enabled": True,
        "queue_count_each_side": 10,
        "queue_spacing": 14.0,
        "queue_columns": 5,
        "queue_row_spacing": 16.0,
        "left_queue_origin": [-292.0, -214.0, 24.0],
        "right_queue_origin": [292.0, 214.0, 24.0],
        "uav_asset_path": None,
        "uav_asset_scale": 6.0,
        "uav_visual_scale": 2.3,
        "helicopter_asset_path": "/home/isaac/ql/asset/NASA/Ingenuity/ingenuity.usd",
        "helicopter_asset_scale": 1.0,
        "uav_motion": {"type": "kinematic", "max_speed": 30.0, "max_accel": 12.0},
        "helicopter_motion": {"type": "dynamic", "max_speed": 24.0, "max_accel": 8.0, "position_gain": 1.3, "velocity_gain": 1.8},
    },
    "camera": {"eye": [520.0, -520.0, 700.0], "target": [0.0, 0.0, 20.0]},
    "weather": {
        "preset": "clear",
        "cloud_cover": 0.05,
        "fog_density": 0.02,
        "rain_rate": 0.0,
        "sea_clutter": 0.10,
        "visual_clouds": True,
        "visual_rain": False,
    },
    "tracking": {
        "enabled": False,
        "report_interval": 1.0,
        "output_enabled": False,
        "sensors": {
            "eo": {"range": 380.0, "base_pd": 0.90, "measurement_noise": 3.5, "false_alarm_rate": 0.015},
            "sar": {"range": 520.0, "base_pd": 0.84, "measurement_noise": 5.5, "false_alarm_rate": 0.035},
            "arm": {"range": 680.0, "base_pd": 0.92, "measurement_noise": 4.0, "false_alarm_rate": 0.010},
        },
        "filter": {"alpha": 0.68, "beta": 0.22, "association_gate": 42.0, "track_timeout": 6.0},
    },
}


@dataclass
class PlatformEntity:
    entity_id: str
    domain: str
    category: str
    faction: str
    visual: dict
    motion_model: BaseMotionModel | None
    reference_fn: Callable[[float], np.ndarray] | None
    signature: dict[str, float]
    rf_emitter: bool = False

    def position(self) -> np.ndarray:
        if self.motion_model is not None:
            return np.array(self.motion_model.state.position, dtype=float)
        if self.visual.get("root_prim") is not None:
            attr = self.visual["root_prim"].GetAttribute("xformOp:translate")
            value = attr.Get()
            return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)
        return np.zeros(3, dtype=float)

    def velocity(self) -> np.ndarray:
        if self.motion_model is not None:
            return np.array(self.motion_model.state.velocity, dtype=float)
        return np.zeros(3, dtype=float)


@dataclass
class SceneState:
    config: dict[str, Any]
    map_size: float
    height_scale: float
    emitter_pos: np.ndarray
    uav: dict | None
    helicopter: dict | None
    entities: list[PlatformEntity]
    uav_motion: BaseMotionModel | None
    helicopter_motion: BaseMotionModel | None
    runtime_dt: float = 1.0 / 60.0
    _last_t: float | None = None

    @property
    def uav_root_path(self) -> str | None:
        if self.uav is None:
            return None
        return self.uav.get("root_path")

    def uav_patrol_pos(self, t: float) -> np.ndarray:
        return _fast_uav_patrol_pos(t, self.height_scale)

    def helicopter_patrol_pos(self, t: float) -> np.ndarray:
        return _fast_helicopter_patrol_pos(t, self.height_scale)

    def update(self, t: float) -> None:
        dt = self.runtime_dt if self._last_t is None else max(1e-3, float(t) - float(self._last_t))
        for entity in self.entities:
            if entity.reference_fn is None:
                continue
            target = np.array(entity.reference_fn(t), dtype=float)
            target_velocity = self._reference_velocity(entity.reference_fn, t)
            motion_state = (
                entity.motion_model.step(dt, target, target_velocity)
                if entity.motion_model is not None
                else MotionState(target, np.zeros(3, dtype=float))
            )
            _set_visual_pose(entity.visual, motion_state.position, t, motion_state)
        self._last_t = float(t)

    def _reference_velocity(self, fn, t: float, eps: float = 0.08) -> np.ndarray:
        prev_pos = np.array(fn(max(0.0, t - eps)), dtype=float)
        next_pos = np.array(fn(t + eps), dtype=float)
        return (next_pos - prev_pos) / max(1e-3, 2.0 * eps)

    def sensor_targets(self) -> list[dict[str, Any]]:
        targets = [
            {
                "entity_id": entity.entity_id,
                "domain": entity.domain,
                "category": entity.category,
                "faction": entity.faction,
                "position": entity.position(),
                "velocity": entity.velocity(),
                "signature": dict(entity.signature),
                "rf_emitter": bool(entity.rf_emitter),
            }
            for entity in self.entities
        ]
        targets.append(
            {
                "entity_id": "Emitter_0",
                "domain": "ground",
                "category": "emitter",
                "faction": "Neutral",
                "position": np.array(self.emitter_pos, dtype=float),
                "velocity": np.zeros(3, dtype=float),
                "signature": {"eo": 0.65, "sar": 0.55, "arm": 1.0},
                "rf_emitter": True,
            }
        )
        return targets

    def observer_snapshot(self) -> dict[str, Any] | None:
        for entity in self.entities:
            if entity.category == "uav":
                return {
                    "entity_id": entity.entity_id,
                    "faction": entity.faction,
                    "position": entity.position(),
                    "velocity": entity.velocity(),
                }
        return None


def terrain_height(x: float, y: float, map_size: float, height_scale: float) -> float:
    ax = abs(x)
    plain_start = map_size * 0.22
    plain_full = map_size * 0.34
    side_plain = min(1.0, max(0.0, (ax - plain_start) / max(1.0, plain_full - plain_start)))
    central_weight = 1.0 - side_plain
    r = math.sqrt(x * x + y * y) / max(1.0, map_size * 0.5)
    peaks = (
        0.58 * math.exp(-((x + 78.0) ** 2 + (y - 56.0) ** 2) / 5200.0)
        + 0.54 * math.exp(-((x - 72.0) ** 2 + (y + 66.0) ** 2) / 5000.0)
        + 0.38 * math.exp(-((x - 18.0) ** 2 + (y - 112.0) ** 2) / 4300.0)
        + 0.32 * math.exp(-((x + 8.0) ** 2 + (y + 8.0) ** 2) / 7600.0)
    )
    ridges = 0.08 * math.sin(x * 0.052 + y * 0.023) + 0.05 * math.cos(y * 0.050)
    plain_roll = 0.018 * math.sin(y * 0.045) + 0.012 * math.cos(x * 0.035)
    edge_falloff = max(0.0, 1.0 - r * 0.28)
    central_height = height_scale * (peaks + ridges) * edge_falloff
    river_dist = _distance_to_polyline(x, y, RIVER_CENTER_POINTS)
    river_cut = math.exp(-(river_dist * river_dist) / 560.0)
    central_height = max(0.0, central_height - height_scale * 0.78 * river_cut)
    plain_height = max(0.0, height_scale * plain_roll)
    height = max(0.0, central_height * central_weight + plain_height * side_plain)
    coast_start = map_size * 0.34
    coast_full = map_size * 0.44
    coast_weight = min(1.0, max(0.0, (x - coast_start) / max(1.0, coast_full - coast_start)))
    return max(0.0, height * (1.0 - coast_weight))


def _fast_uav_patrol_pos(t: float, height_scale: float) -> np.ndarray:
    return np.array(
        [
            FAST_UAV_ORBIT_RADIUS * math.cos(t * FAST_UAV_ORBIT_RATE),
            FAST_UAV_ORBIT_RADIUS * math.sin(t * FAST_UAV_ORBIT_RATE),
            height_scale + FAST_UAV_ALTITUDE_OFFSET + 2.0 * math.sin(t * 0.8),
        ],
        dtype=float,
    )


def _fast_helicopter_patrol_pos(t: float, height_scale: float) -> np.ndarray:
    return np.array(
        [
            FAST_HELI_ORBIT_RADIUS * math.cos(t * FAST_HELI_ORBIT_RATE + math.pi),
            FAST_HELI_ORBIT_RADIUS * math.sin(t * FAST_HELI_ORBIT_RATE + math.pi),
            height_scale + FAST_HELI_ALTITUDE_OFFSET + 1.2 * math.sin(t * 0.9),
        ],
        dtype=float,
    )


def _distance_to_polyline(x: float, y: float, points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return math.inf
    return min(_distance_to_segment(x, y, x1, y1, x2, y2) for (x1, y1), (x2, y2) in zip(points, points[1:]))


def _distance_to_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return math.hypot(x - x1, y - y1)
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg_len_sq))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(x - nearest_x, y - nearest_y)


def create_scene(stage, cuboid_cls, sphere_cls, config: dict[str, Any]) -> SceneState:
    from pxr import Gf, Sdf, UsdLux

    terrain_cfg = config.get("terrain", {})
    forest_cfg = config.get("forest", {})
    rocks_cfg = config.get("rocks", {})
    water_cfg = config.get("water", {})
    battlefield_cfg = config.get("battlefield", {})
    ground_units_cfg = config.get("ground_units", {})
    emitter_cfg = config.get("emitter", {})
    aircraft_cfg = config.get("aircraft", {})
    runtime_cfg = config.get("runtime", {})
    weather_cfg = config.get("weather", {})

    map_size = float(terrain_cfg.get("map_size", 220.0))
    height_scale = float(terrain_cfg.get("height_scale", 46.0))
    seed = int(forest_cfg.get("seed", config.get("seed", 42)))
    rng = random.Random(seed)

    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    sun_intensity = float(config.get("lighting", {}).get("sun_intensity", 1300.0))
    cloud_cover = float(weather_cfg.get("cloud_cover", 0.0))
    fog_density = float(weather_cfg.get("fog_density", 0.0))
    rain_rate = float(weather_cfg.get("rain_rate", 0.0))
    weather_light_scale = max(0.28, 1.0 - 0.45 * cloud_cover - 0.28 * fog_density - 0.22 * rain_rate)
    light.CreateIntensityAttr(sun_intensity * weather_light_scale)
    light.CreateAngleAttr(float(config.get("lighting", {}).get("sun_angle", 0.55)))

    _create_mountain_terrain(stage, map_size, height_scale, int(terrain_cfg.get("grid", 96)))
    if bool(water_cfg.get("enabled", True)):
        _create_water_features(stage, cuboid_cls, water_cfg, map_size, height_scale)
    if bool(forest_cfg.get("enabled", True)):
        _create_forest(stage, cuboid_cls, sphere_cls, int(forest_cfg.get("tree_count", 130)), map_size, height_scale, seed)
    if bool(rocks_cfg.get("enabled", True)):
        _create_rocks(stage, sphere_cls, rng, int(rocks_cfg.get("count", 30)), map_size, height_scale)
    weather_counts = _create_weather_visuals(stage, cuboid_cls, sphere_cls, weather_cfg, map_size, height_scale, seed)
    print(
        "[QL] Weather visuals ready: "
        f"preset={weather_cfg.get('preset', 'clear')} "
        f"clouds={weather_counts.get('clouds', 0)} fog={weather_counts.get('fog', 0)} "
        f"rain={weather_counts.get('rain', 0)} lights={weather_counts.get('lights', 0)}"
    )
    if bool(battlefield_cfg.get("enabled", True)):
        _create_battlefield_features(stage, cuboid_cls, sphere_cls, battlefield_cfg, map_size, height_scale)
    emitter_xy = emitter_cfg.get("position_xy", [18.0, -8.0])
    emitter_pos = np.array(
        [
            float(emitter_xy[0]),
            float(emitter_xy[1]),
            terrain_height(float(emitter_xy[0]), float(emitter_xy[1]), map_size, height_scale)
            + float(emitter_cfg.get("height_offset", 1.2)),
        ]
    )
    _create_radiation_emitter(stage, sphere_cls, emitter_pos, emitter_cfg)

    uav = None
    helicopter = None
    entities: list[PlatformEntity] = []
    uav_start = None
    heli_start = None
    if bool(aircraft_cfg.get("uav", True)):
        uav_start = np.array(aircraft_cfg.get("uav_start", [35.0, -42.0, 20.0]), dtype=float)
        uav_start[2] += height_scale
        uav = _spawn_drone(stage, cuboid_cls, sphere_cls, 0, uav_start, aircraft_cfg)
        _remove_physics(stage, "/World/UAV_0")
    if bool(aircraft_cfg.get("helicopter", True)):
        heli_start = np.array(aircraft_cfg.get("helicopter_start", [-36.0, 38.0, 24.0]), dtype=float)
        heli_start[2] += height_scale
        helicopter = _spawn_helicopter(stage, cuboid_cls, sphere_cls, 1, heli_start, aircraft_cfg)
        _remove_physics(stage, "/World/Helicopter_1")
    if bool(aircraft_cfg.get("formations_enabled", True)):
        _create_aircraft_queues(stage, cuboid_cls, sphere_cls, aircraft_cfg, map_size, height_scale)

    uav_motion_model = None
    helicopter_motion_model = None

    if uav is not None and uav_start is not None:
        uav_motion_model = build_motion_model(aircraft_cfg.get("uav_motion", {}), uav_start, domain="air")
        entities.append(
            PlatformEntity(
                entity_id="UAV_0",
                domain="air",
                category="uav",
                faction="Blue",
                visual=uav,
                motion_model=uav_motion_model,
                reference_fn=lambda t, hs=height_scale: _fast_uav_patrol_pos(t, hs),
                signature={"eo": 0.72, "sar": 0.68, "arm": 0.12},
                rf_emitter=False,
            )
        )
    if helicopter is not None and heli_start is not None:
        helicopter_motion_model = build_motion_model(aircraft_cfg.get("helicopter_motion", {}), heli_start, domain="air")
        entities.append(
            PlatformEntity(
                entity_id="Helicopter_1",
                domain="air",
                category="helicopter",
                faction="Red",
                visual=helicopter,
                motion_model=helicopter_motion_model,
                reference_fn=lambda t, hs=height_scale: _fast_helicopter_patrol_pos(t, hs),
                signature={"eo": 0.82, "sar": 0.76, "arm": 0.18},
                rf_emitter=False,
            )
        )

    if bool(ground_units_cfg.get("enabled", True)):
        for item in _create_ground_units(
            stage,
            cuboid_cls,
            sphere_cls,
            ground_units_cfg,
            map_size,
            height_scale,
            water_cfg,
            terrain_height,
        ):
            domain = str(item["domain"])
            category = str(item["category"])
            if category == "tank":
                motion_cfg = ground_units_cfg.get("tank_motion", {})
                z_value = float(item["spawn"][2])
                tank_visual_scale = float(ground_units_cfg.get("tank_visual_scale", 1.0))
                tank_ground_offset = float(ground_units_cfg.get("tank_ground_offset", 2.20))
                z_fn = lambda px, py: _area_ground_z(
                    px,
                    py,
                    9.0 * tank_visual_scale,
                    5.8 * tank_visual_scale,
                    map_size,
                    height_scale,
                    tank_ground_offset,
                )
                reference_fn = _make_path_reference(
                    list(item["path_points"]),
                    float(item["path_phase"]),
                    float(motion_cfg.get("max_speed", 7.0)) * 0.55,
                    z_value,
                    loop=True,
                    z_fn=z_fn,
                )
            elif category == "submarine":
                motion_cfg = ground_units_cfg.get("submarine_motion", {})
                z_value = float(item["spawn"][2])
                reference_fn = _make_path_reference(
                    list(item["path_points"]),
                    float(item["path_phase"]),
                    float(motion_cfg.get("max_speed", 5.5)) * 0.45,
                    z_value,
                    loop=True,
                    sway=1.2,
                )
            else:
                motion_cfg = ground_units_cfg.get("uuv_motion", {})
                z_value = float(item["spawn"][2]) - 0.35
                reference_fn = _make_path_reference(
                    list(item["path_points"]),
                    float(item["path_phase"]),
                    float(motion_cfg.get("max_speed", 3.8)) * 0.42,
                    z_value,
                    loop=True,
                    sway=0.8,
                )

            motion_enabled = bool(motion_cfg.get("enabled", True))
            if not motion_enabled:
                reference_fn = None

            motion_model = None
            if motion_enabled:
                motion_model = build_motion_model(motion_cfg, np.array(item["spawn"], dtype=float), domain=domain)

            entities.append(
                PlatformEntity(
                    entity_id=str(item["entity_id"]),
                    domain=domain,
                    category=category,
                    faction=str(item["faction"]),
                    visual=item["visual"],
                    motion_model=motion_model,
                    reference_fn=reference_fn,
                    signature=dict(item["signature"]),
                    rf_emitter=False,
                )
            )

    print("[QL] Scene ready: engineered mountain battlefield, emitter, aircraft queues.")
    return SceneState(
        config=config,
        map_size=map_size,
        height_scale=height_scale,
        emitter_pos=emitter_pos,
        uav=uav,
        helicopter=helicopter,
        entities=entities,
        uav_motion=uav_motion_model,
        helicopter_motion=helicopter_motion_model,
        runtime_dt=float(runtime_cfg.get("dt", 1.0 / 60.0)),
    )


def camera_view(config: dict[str, Any]) -> tuple[list[float], list[float]]:
    camera_cfg = config.get("camera", {})
    return camera_cfg.get("eye", [118.0, -145.0, 105.0]), camera_cfg.get("target", [0.0, 0.0, 24.0])


def _polyline_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:]))


def _polyline_point(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        return points[0]
    total = _polyline_length(points)
    if total <= 1e-9:
        return points[0]
    target = max(0.0, min(1.0, t)) * total
    walked = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len <= 1e-9:
            continue
        if walked + seg_len >= target:
            ratio = (target - walked) / seg_len
            return x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio
        walked += seg_len
    return points[-1]


def _make_path_reference(
    path_points: list[tuple[float, float]],
    start_phase: float,
    speed: float,
    z_value: float,
    loop: bool = True,
    sway: float = 0.0,
    z_fn: Callable[[float, float], float] | None = None,
) -> Callable[[float], np.ndarray]:
    total_length = max(1.0, _polyline_length(path_points))
    start_phase = float(start_phase)
    speed = max(0.1, float(speed))

    def _reference(t: float) -> np.ndarray:
        phase = start_phase + (float(t) * speed / total_length)
        if loop:
            phase = phase % 1.0
        else:
            phase = max(0.0, min(1.0, phase))
        x, y = _polyline_point(path_points, phase)
        if sway > 0.0:
            y += sway * math.sin(float(t) * 0.22 + phase * math.pi * 4.0)
        z = float(z_value) if z_fn is None else float(z_fn(float(x), float(y)))
        return np.array([float(x), float(y), z], dtype=float)

    return _reference


def _remove_physics(stage, prim_path: str) -> None:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        return
    for prim in Usd.PrimRange(root):
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.ArticulationRootAPI, UsdPhysics.MassAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)


def remove_physics(prim_path: str, stage=None) -> None:
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    _remove_physics(stage, prim_path)


def _create_mountain_terrain(stage, map_size: float, height_scale: float, grid: int) -> None:
    from pxr import Gf, Sdf, UsdGeom

    half = map_size * 0.5
    step = map_size / grid
    points = []
    for iy in range(grid + 1):
        y = -half + iy * step
        for ix in range(grid + 1):
            x = -half + ix * step
            points.append(Gf.Vec3f(x, y, terrain_height(x, y, map_size, height_scale)))

    face_counts = []
    face_indices = []
    for iy in range(grid):
        for ix in range(grid):
            i0 = iy * (grid + 1) + ix
            face_counts.append(4)
            face_indices.extend([i0, i0 + 1, i0 + grid + 2, i0 + grid + 1])

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/MountainTerrain"))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.22, 0.30, 0.18)])


def _set_display_color(prim, color: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _vec(values: Any) -> np.ndarray:
    return np.array(values, dtype=float)


def _box(
    cuboid_cls,
    prim_path: str,
    name: str,
    position: Any,
    scale: Any,
    color: Any,
):
    return cuboid_cls(
        prim_path=prim_path,
        name=name,
        position=_vec(position),
        size=1.0,
        scale=_vec(scale),
        color=_vec(color),
    )


def _ball(
    sphere_cls,
    prim_path: str,
    name: str,
    position: Any,
    radius: float,
    color: Any,
):
    return sphere_cls(
        prim_path=prim_path,
        name=name,
        position=_vec(position),
        radius=float(radius),
        color=_vec(color),
    )


def _cylinder(
    stage,
    prim_path: str,
    radius: float,
    height: float,
    position: Any,
    color: tuple[float, float, float],
    rotate_xyz: tuple[float, float, float] | None = None,
):
    from pxr import Gf, Sdf, UsdGeom

    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(prim_path))
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    cylinder_xf = UsdGeom.Xformable(cylinder.GetPrim())
    cylinder_xf.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    if rotate_xyz is not None:
        cylinder_xf.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz))
    _set_display_color(cylinder.GetPrim(), color)
    return cylinder.GetPrim()


def _resolve_asset_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    if "://" in str(raw_path):
        return str(raw_path)
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path) if path.exists() else None
    candidates = [
        (QL_ROOT / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _standby_drone_usd() -> str | None:
    try:
        from isaacsim.storage.native import get_assets_root_path
    except Exception:
        try:
            from omni.isaac.core.utils.nucleus import get_assets_root_path
        except Exception:
            return None

    assets_root = get_assets_root_path()
    if not assets_root:
        return None
    STANDBY_DRONE_USD = assets_root.rstrip("/") + "/Isaac/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd"
    return STANDBY_DRONE_USD


def _resolve_builtin_uav_asset() -> str | None:
    standby_drone_usd = _standby_drone_usd()
    if standby_drone_usd:
        return standby_drone_usd

    local_patterns = [
        "/home/isaac/ql/asset/NTNU/ARL-Robot-1/arl_robot_1.usd",
        "/home/*/ql/asset/NTNU/ARL-Robot-1/arl_robot_1.usd",
        "/home/*/isaac*/ql/asset/NTNU/ARL-Robot-1/arl_robot_1.usd",
    ]
    local_candidates: list[str] = []
    for pattern in local_patterns:
        for raw in sorted(glob.glob(pattern)):
            candidate = Path(raw).expanduser()
            if candidate.exists():
                local_candidates.append(candidate.resolve().as_posix())

    if local_candidates:
        return sorted(set(local_candidates), key=len)[0]
    return None


def _add_usd_reference(stage, usd_path: str, prim_path: str) -> bool:
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        return bool(stage.GetPrimAtPath(prim_path))
    except Exception as exc:
        print(f"[QL][WARN] Failed to load USD asset {usd_path}: {exc}")
        return False


def _spawn_fallback_tree(stage, cuboid_cls, sphere_cls, idx: int, x: float, y: float, z: float, rng: random.Random) -> None:
    from pxr import Gf, Sdf, UsdGeom

    trunk_h = rng.uniform(2.8, 5.0)
    trunk_r = rng.uniform(0.14, 0.24)
    tree_kind = "pine" if rng.random() < 0.58 else "broadleaf"

    _cylinder(
        stage,
        f"/World/Forest/Tree_{idx}_Trunk",
        trunk_r,
        trunk_h,
        (x, y, z + trunk_h * 0.5),
        (0.30, 0.17, 0.08),
    )

    if tree_kind == "pine":
        for level in range(3):
            cone_h = rng.uniform(2.0, 3.0) * (1.0 - level * 0.12)
            cone_r = rng.uniform(1.15, 1.75) * (1.0 - level * 0.18)
            cone_z = z + trunk_h * 0.55 + level * cone_h * 0.48
            cone = UsdGeom.Cone.Define(stage, Sdf.Path(f"/World/Forest/Tree_{idx}_Crown_{level}"))
            cone.CreateRadiusAttr(float(cone_r))
            cone.CreateHeightAttr(float(cone_h))
            UsdGeom.Xformable(cone.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(cone_z)))
            _set_display_color(cone.GetPrim(), (0.03, rng.uniform(0.26, 0.38), 0.10))
    else:
        for level, offset in enumerate([(0.0, 0.0), (-0.55, 0.25), (0.52, -0.20)]):
            crown_r = rng.uniform(1.0, 1.55)
            _ball(
                sphere_cls,
                prim_path=f"/World/Forest/Tree_{idx}_Crown_{level}",
                name=f"tree_{idx}_crown_{level}",
                position=np.array([x + offset[0], y + offset[1], z + trunk_h + crown_r * 0.45]),
                radius=crown_r,
                color=[0.04, rng.uniform(0.30, 0.48), 0.12],
            )


def _create_forest(stage, cuboid_cls, sphere_cls, tree_count: int, map_size: float, height_scale: float, seed: int) -> None:
    rng = random.Random(seed)
    half = map_size * 0.44
    created = 0
    for i in range(tree_count):
        for _ in range(20):
            x = rng.uniform(-half, half)
            y = rng.uniform(-half, half)
            if abs(x) < 14.0 and abs(y) < 14.0:
                continue
            if _distance_to_polyline(x, y, RIVER_CENTER_POINTS) < 28.0:
                continue
            z = terrain_height(x, y, map_size, height_scale)
            if z < height_scale * 0.95:
                _spawn_fallback_tree(stage, cuboid_cls, sphere_cls, i, x, y, z, rng)
                created += 1
                break
    print(f"[QL] Forest ready: source=procedural, trees={created}")


def _create_rocks(stage, sphere_cls, rng: random.Random, count: int, map_size: float, height_scale: float) -> None:
    created = 0
    attempts = 0
    while created < count and attempts < count * 20:
        attempts += 1
        x = rng.uniform(-map_size * 0.44, map_size * 0.44)
        y = rng.uniform(-map_size * 0.44, map_size * 0.44)
        if _distance_to_polyline(x, y, RIVER_CENTER_POINTS) < 18.0:
            continue
        z = terrain_height(x, y, map_size, height_scale) + 0.35
        _ball(
            sphere_cls,
            prim_path=f"/World/Rocks/Rock_{created}",
            name=f"rock_{created}",
            position=np.array([x, y, z]),
            radius=rng.uniform(0.35, 1.1),
            color=[0.28, 0.29, 0.27],
        )
        created += 1


def _path_points(raw_points: list[tuple[float, float]] | list[list[float]] | None) -> list[tuple[float, float]]:
    if not raw_points:
        return []
    return [(float(px), float(py)) for px, py in raw_points]


def _polyline_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:]))


def _polyline_point(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        return points[0]
    total = _polyline_length(points)
    if total <= 1e-9:
        return points[0]
    target = max(0.0, min(1.0, t)) * total
    walked = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len <= 1e-9:
            continue
        if walked + seg_len >= target:
            ratio = (target - walked) / seg_len
            return x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio
        walked += seg_len
    return points[-1]


def _segment_orientation(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _densify_polyline(points: list[tuple[float, float]], max_step: float) -> list[tuple[float, float]]:
    if len(points) < 2:
        return list(points)
    dense = [points[0]]
    step = max(4.0, float(max_step))
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len <= 1e-9:
            continue
        pieces = max(1, int(math.ceil(seg_len / step)))
        for idx in range(1, pieces + 1):
            t = idx / pieces
            dense.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    return dense


def _smooth_values(values: list[float], radius: int = 2) -> list[float]:
    if len(values) < 3:
        return list(values)
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        smoothed.append(sum(values[start:end]) / max(1, end - start))
    return smoothed


def _road_point_height(
    points: list[tuple[float, float]],
    idx: int,
    width: float,
    map_size: float,
    height_scale: float,
    clearance: float,
) -> float:
    x, y = points[idx]
    if len(points) == 1:
        dx, dy = 1.0, 0.0
    elif idx == 0:
        dx = points[1][0] - x
        dy = points[1][1] - y
    elif idx == len(points) - 1:
        dx = x - points[idx - 1][0]
        dy = y - points[idx - 1][1]
    else:
        dx = points[idx + 1][0] - points[idx - 1][0]
        dy = points[idx + 1][1] - points[idx - 1][1]
    length = max(1e-6, math.hypot(dx, dy))
    tx, ty = dx / length, dy / length
    nx, ny = -ty, tx
    lateral = max(width * 0.58, 4.0)
    longitudinal = max(width * 0.35, 5.0)
    offsets = [
        (0.0, 0.0),
        (nx * lateral, ny * lateral),
        (-nx * lateral, -ny * lateral),
        (nx * lateral * 0.5, ny * lateral * 0.5),
        (-nx * lateral * 0.5, -ny * lateral * 0.5),
        (tx * longitudinal, ty * longitudinal),
        (-tx * longitudinal, -ty * longitudinal),
    ]
    highest = max(terrain_height(x + ox, y + oy, map_size, height_scale) for ox, oy in offsets)
    return highest + clearance


def _create_oriented_box(
    stage,
    prim_path: str,
    center: tuple[float, float],
    size: tuple[float, float, float],
    color: list[float] | tuple[float, float, float],
    rotate_z_deg: float = 0.0,
    z: float = 0.0,
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(z)))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float(rotate_z_deg)))
    xform.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))
    _set_display_color(cube.GetPrim(), tuple(color))


def _create_polyline_strips(
    stage,
    base_path: str,
    points: list[tuple[float, float]],
    width: float,
    thickness: float,
    z: float,
    color: list[float] | tuple[float, float, float],
    overlap: float = 0.0,
) -> None:
    for idx, ((x1, y1), (x2, y2)) in enumerate(zip(points, points[1:])):
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len <= 1e-9:
            continue
        extend = max(0.0, overlap)
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        angle = _segment_orientation(x1, y1, x2, y2)
        _create_oriented_box(
            stage,
            f"{base_path}_{idx}",
            center,
            (seg_len + extend, width, thickness),
            color,
            angle,
            z,
        )


def _create_water_features(
    stage,
    cuboid_cls,
    config: dict[str, Any],
    map_size: float,
    height_scale: float,
) -> None:
    river_path = _path_points(config.get("river_path", RIVER_CENTER_POINTS))
    water_color = config.get("color", [0.05, 0.20, 0.36])
    river_width = float(config.get("river_width", 22.0))
    bank_width = float(config.get("bank_width", river_width * 1.7))
    shore_width = float(config.get("shore_width", river_width * 2.8))
    z = float(config.get("z", 0.22))
    bank_color = config.get("bank_color", [0.31, 0.26, 0.15])
    shore_color = config.get("shore_color", [0.24, 0.34, 0.18])

    if len(river_path) >= 2:
        _create_polyline_strips(stage, "/World/Water/RiverShore", river_path, shore_width, 0.16, z + 0.02, shore_color, overlap=4.0)
        _create_polyline_strips(stage, "/World/Water/RiverBank", river_path, bank_width, 0.14, z + 0.06, bank_color, overlap=2.0)
        _create_polyline_strips(stage, "/World/Water/River", river_path, river_width, 0.12, z + 0.12, water_color, overlap=0.5)

        bridge_color = config.get("bridge_color", [0.30, 0.30, 0.27])
        bridge_thickness = float(config.get("bridge_thickness", 0.48))
        bridge_width = float(config.get("bridge_width", river_width + 10.0))
        for idx, t in enumerate([0.22, 0.50, 0.76]):
            cx, cy = _polyline_point(river_path, t)
            nx, ny = _polyline_point(river_path, min(1.0, t + 0.03))
            angle = _segment_orientation(cx, cy, nx, ny)
            _create_oriented_box(
                stage,
                f"/World/Water/Bridge_{idx}",
                (cx, cy),
                (42.0, bridge_width, bridge_thickness),
                bridge_color,
                angle + 90.0,
                z + 1.55,
            )

    if bool(config.get("coastal_enabled", False)):
        sea_width = float(config.get("sea_width", 72.0))
        sea_length = float(config.get("sea_length", map_size * 0.84))
        sea_y = float(config.get("sea_y", 0.0))
        sea_x = float(config.get("sea_x", map_size * 0.5 + sea_width * 0.15))
        _box(
            cuboid_cls,
            prim_path="/World/Water/Coast",
            name="coast",
            position=np.array([sea_x, sea_y, z]),
            scale=np.array([sea_width, sea_length, 0.08]),
            color=water_color,
        )


def _ground_z(x: float, y: float, map_size: float, height_scale: float, offset: float = 0.08) -> float:
    return terrain_height(x, y, map_size, height_scale) + offset


def _area_ground_z(x: float, y: float, sx: float, sy: float, map_size: float, height_scale: float, offset: float) -> float:
    half_x = abs(float(sx)) * 0.5
    half_y = abs(float(sy)) * 0.5
    samples = [
        (x, y),
        (x - half_x, y - half_y),
        (x - half_x, y + half_y),
        (x + half_x, y - half_y),
        (x + half_x, y + half_y),
        (x - half_x, y),
        (x + half_x, y),
        (x, y - half_y),
        (x, y + half_y),
    ]
    highest = max(terrain_height(px, py, map_size, height_scale) for px, py in samples)
    return highest + offset


def _place_cuboid_on_ground(
    cuboid_cls,
    prim_path: str,
    name: str,
    x: float,
    y: float,
    scale: list[float],
    color: list[float],
    map_size: float,
    height_scale: float,
    z_offset: float = 0.08,
) -> None:
    z = _area_ground_z(x, y, float(scale[0]), float(scale[1]), map_size, height_scale, z_offset)
    _box(
        cuboid_cls,
        prim_path=prim_path,
        name=name,
        position=np.array([x, y, z]),
        scale=scale,
        color=color,
    )


def _place_oriented_box_on_ground(
    stage,
    prim_path: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: float,
    thickness: float,
    color: list[float] | tuple[float, float, float],
    map_size: float,
    height_scale: float,
    z_offset: float,
    extra_length: float = 0.0,
) -> None:
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 1e-9:
        return
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    nx = -(y2 - y1) / length
    ny = (x2 - x1) / length
    samples = []
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        px = x1 + (x2 - x1) * t
        py = y1 + (y2 - y1) * t
        samples.extend(
            [
                (px, py),
                (px + nx * width * 0.5, py + ny * width * 0.5),
                (px - nx * width * 0.5, py - ny * width * 0.5),
            ]
        )
    z = max(terrain_height(px, py, map_size, height_scale) for px, py in samples) + z_offset
    _create_oriented_box(
        stage,
        prim_path,
        (cx, cy),
        (length + extra_length, width, thickness),
        color,
        _segment_orientation(x1, y1, x2, y2),
        z,
    )


def _create_road_path(
    stage,
    base_path: str,
    points: list[tuple[float, float]],
    width: float,
    road_color: list[float],
    map_size: float,
    height_scale: float,
    shoulder_color: list[float] | None = None,
    lane_color: list[float] | None = None,
) -> None:
    shoulder_color = shoulder_color or [0.09, 0.10, 0.10]
    lane_color = lane_color or [0.78, 0.72, 0.28]
    dense_points = _densify_polyline(points, max_step=max(24.0, width * 2.2))
    if len(dense_points) < 2:
        return
    heights = [
        _road_point_height(dense_points, idx, width + 5.2, map_size, height_scale, clearance=0.92)
        for idx in range(len(dense_points))
    ]
    heights = _smooth_values(_smooth_values(heights, radius=2), radius=1)

    for idx, ((x1, y1), (x2, y2)) in enumerate(zip(dense_points, dense_points[1:])):
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len <= 1e-9:
            continue
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        angle = _segment_orientation(x1, y1, x2, y2)
        z = max(heights[idx], heights[idx + 1])
        _create_oriented_box(
            stage,
            f"{base_path}_Shoulder_{idx}",
            center,
            (seg_len + 3.0, width + 5.2, 0.16),
            shoulder_color,
            angle,
            z,
        )
        _create_oriented_box(
            stage,
            f"{base_path}_Asphalt_{idx}",
            center,
            (seg_len + 3.4, width, 0.18),
            road_color,
            angle,
            z + 0.03,
        )
        _create_oriented_box(
            stage,
            f"{base_path}_CenterLine_{idx}",
            center,
            (seg_len + 2.0, 0.85, 0.20),
            lane_color,
            angle,
            z + 0.07,
        )


def _create_cross_map_roads(
    stage,
    config: dict[str, Any],
    map_size: float,
    height_scale: float,
    road_color: list[float],
) -> None:
    half = map_size * 0.5
    road_width = float(config.get("arterial_width", 12.0))
    highway_a = [
        (-half, -232.0),
        (-208.0, -156.0),
        (-104.0, -78.0),
        (0.0, 0.0),
        (104.0, 78.0),
        (208.0, 156.0),
        (half, 232.0),
    ]
    highway_b = [
        (-half, 206.0),
        (-204.0, 136.0),
        (-98.0, 68.0),
        (0.0, 0.0),
        (98.0, -68.0),
        (204.0, -136.0),
        (half, -206.0),
    ]
    base_links = [
        [(-296.0, -190.0), (-268.0, -162.0), (-240.0, -134.0), (-212.0, -106.0), (-184.0, -78.0)],
        [(296.0, 190.0), (268.0, 162.0), (240.0, 134.0), (212.0, 106.0), (184.0, 78.0)],
        [(-96.0, 68.0), (-150.0, 36.0), (-202.0, -18.0), (-240.0, -90.0)],
        [(96.0, -68.0), (150.0, -36.0), (202.0, 18.0), (240.0, 90.0)],
    ]

    _create_road_path(stage, "/World/Battlefield/CrossRoad_A", highway_a, road_width, road_color, map_size, height_scale)
    _create_road_path(stage, "/World/Battlefield/CrossRoad_B", highway_b, road_width, road_color, map_size, height_scale)
    for idx, path in enumerate(base_links):
        _create_road_path(stage, f"/World/Battlefield/AccessRoad_{idx}", path, 8.0, road_color, map_size, height_scale)

    _place_oriented_box_on_ground(
        stage,
        "/World/Battlefield/Crossroad_Plaza",
        -18.0,
        -18.0,
        18.0,
        18.0,
        34.0,
        0.22,
        road_color,
        map_size,
        height_scale,
        1.04,
        extra_length=0.0,
    )


def _create_battlefield_features(
    stage,
    cuboid_cls,
    sphere_cls,
    config: dict[str, Any],
    map_size: float,
    height_scale: float,
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    road_color = config.get("road_color", [0.18, 0.19, 0.18])
    pad_color = config.get("pad_color", [0.24, 0.25, 0.23])
    earth_color = config.get("earthwork_color", [0.34, 0.27, 0.16])
    bunker_color = config.get("bunker_color", [0.36, 0.39, 0.34])

    _create_cross_map_roads(stage, config, map_size, height_scale, road_color)
    _create_road_path(stage, "/World/Battlefield/LeftBaseRoad_NS", [(-240.0, -190.0), (-240.0, -78.0)], 7.2, road_color, map_size, height_scale)
    _create_road_path(stage, "/World/Battlefield/LeftBaseRoad_EW", [(-296.0, -134.0), (-156.0, -134.0)], 7.2, road_color, map_size, height_scale)
    _create_road_path(stage, "/World/Battlefield/RightBaseRoad_NS", [(240.0, 190.0), (240.0, 78.0)], 7.2, road_color, map_size, height_scale)
    _create_road_path(stage, "/World/Battlefield/RightBaseRoad_EW", [(296.0, 134.0), (156.0, 134.0)], 7.2, road_color, map_size, height_scale)

    for idx, (x, y) in enumerate([(-240.0, -150.0), (240.0, 150.0)]):
        _place_cuboid_on_ground(cuboid_cls, f"/World/Battlefield/BaseSlab_{idx}", f"base_slab_{idx}", x, y, [104.0, 82.0, 1.10], [0.20, 0.22, 0.21], map_size, height_scale, 1.15)
        _place_cuboid_on_ground(cuboid_cls, f"/World/Battlefield/Apron_{idx}", f"apron_{idx}", x, y, [74.0, 56.0, 0.42], pad_color, map_size, height_scale, 1.85)
        _place_cuboid_on_ground(cuboid_cls, f"/World/Battlefield/CommandPad_{idx}", f"command_pad_{idx}", x, y + (30.0 if y < 0 else -30.0), [30.0, 13.0, 0.42], [0.26, 0.28, 0.27], map_size, height_scale, 2.05)
        _place_cuboid_on_ground(cuboid_cls, f"/World/Battlefield/TaxiMark_{idx}", f"taxi_mark_{idx}", x, y, [66.0, 1.2, 0.34], [0.72, 0.70, 0.26], map_size, height_scale, 2.28)

    for idx, (x, y, sx, sy) in enumerate(
        [
            (-262.0, -176.0, 82.0, 2.2),
            (-262.0, -104.0, 82.0, 2.2),
            (262.0, 176.0, 82.0, 2.2),
            (262.0, 104.0, 82.0, 2.2),
            (-184.0, -86.0, 54.0, 2.2),
            (184.0, 86.0, 54.0, 2.2),
        ]
    ):
        _place_cuboid_on_ground(cuboid_cls, f"/World/Battlefield/Earthwork_{idx}", f"earthwork_{idx}", x, y, [sx, sy, 1.0], earth_color, map_size, height_scale, 0.5)

    for idx, (x, y) in enumerate([(-286.0, -180.0), (-226.0, -172.0), (226.0, 172.0), (286.0, 180.0), (-172.0, -74.0), (172.0, 74.0)]):
        _place_cuboid_on_ground(cuboid_cls, f"/World/Battlefield/Bunker_{idx}", f"bunker_{idx}", x, y, [6.0, 4.4, 2.4], bunker_color, map_size, height_scale, 1.1)
        _place_cuboid_on_ground(cuboid_cls, f"/World/Battlefield/BunkerRoof_{idx}", f"bunker_roof_{idx}", x, y, [7.0, 5.2, 0.38], [0.20, 0.22, 0.20], map_size, height_scale, 2.45)

    for idx, (x, y) in enumerate([(-292.0, -202.0), (292.0, 202.0)]):
        z = _ground_z(x, y, map_size, height_scale, 3.0)
        mast = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"/World/Battlefield/Radar_{idx}_Mast"))
        mast.CreateRadiusAttr(0.22)
        mast.CreateHeightAttr(6.0)
        UsdGeom.Xformable(mast.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        _set_display_color(mast.GetPrim(), (0.18, 0.20, 0.20))
        sphere_cls(
            prim_path=f"/World/Battlefield/Radar_{idx}_Dish",
            name=f"radar_{idx}_dish",
            position=np.array([x, y, z + 3.3]),
            radius=1.2,
            color=np.array([0.52, 0.58, 0.58]),
        )

    for idx, (x, y) in enumerate([(-292.0, -204.0), (-280.0, -204.0), (-268.0, -204.0), (268.0, 204.0), (280.0, 204.0), (292.0, 204.0)]):
        _place_cuboid_on_ground(cuboid_cls, f"/World/Battlefield/Container_{idx}", f"container_{idx}", x, y, [8.0, 2.4, 2.2], [0.28, 0.34, 0.38], map_size, height_scale, 3.0)

    building_specs = [
        (-292.0, -92.0, 14.0, 10.0, 5.5),
        (-264.0, -92.0, 12.0, 12.0, 4.5),
        (-208.0, -190.0, 18.0, 9.0, 4.0),
        (-174.0, -190.0, 16.0, 10.0, 4.8),
        (292.0, 92.0, 14.0, 10.0, 5.5),
        (264.0, 92.0, 12.0, 12.0, 4.5),
        (208.0, 190.0, 18.0, 9.0, 4.0),
        (174.0, 190.0, 16.0, 10.0, 4.8),
    ]
    for idx, (x, y, sx, sy, sz) in enumerate(building_specs):
        _place_cuboid_on_ground(
            cuboid_cls,
            f"/World/Battlefield/Building_{idx}",
            f"building_{idx}",
            x,
            y,
            [sx, sy, sz],
            [0.30, 0.32, 0.31],
            map_size,
            height_scale,
            sz * 0.5 + 2.0,
        )


def _create_aircraft_queues(
    stage,
    cuboid_cls,
    sphere_cls,
    config: dict[str, Any],
    map_size: float,
    height_scale: float,
) -> None:
    count = int(config.get("queue_count_each_side", 5))
    spacing = float(config.get("queue_spacing", 9.0))
    columns = max(1, int(config.get("queue_columns", 5)))
    row_spacing = float(config.get("queue_row_spacing", spacing * 1.15))
    left_origin = np.array(config.get("left_queue_origin", [-78.0, -36.0, 18.0]), dtype=float)
    right_origin = np.array(config.get("right_queue_origin", [78.0, -36.0, 18.0]), dtype=float)

    for side, origin, x_step_sign, marker_sign, base_idx, faction in [
        ("Left", left_origin, 1.0, -1.0, 10, "Blue"),
        ("Right", right_origin, -1.0, 1.0, 30, "Red"),
    ]:
        for i in range(count):
            row = i // columns
            col = i % columns
            x = float(origin[0] + x_step_sign * col * spacing)
            y = float(origin[1] + (-1.0 if side == "Right" else 1.0) * row * row_spacing)
            z = _ground_z(x, y, map_size, height_scale, 0.0) + float(origin[2])
            drone_config = dict(config)
            drone_config["faction"] = faction
            visual = _spawn_drone(stage, cuboid_cls, sphere_cls, base_idx + i, np.array([x, y, z]), drone_config)
            _remove_physics(stage, visual["root_path"])
            _place_cuboid_on_ground(
                cuboid_cls,
                f"/World/Battlefield/{side}_QueueMarker_{i}",
                f"{side.lower()}_queue_marker_{i}",
                x + marker_sign * 4.0,
                y,
                [1.2, 5.0, 0.18],
                [0.20, 0.38, 0.70] if side == "Left" else [0.70, 0.24, 0.20],
                map_size,
                height_scale,
                0.2,
            )


def _create_radiation_emitter(stage, sphere_cls, emitter_pos: np.ndarray, config: dict[str, Any]) -> None:
    from pxr import Gf, Sdf, UsdLux

    _ball(
        sphere_cls,
        prim_path="/World/RadiationEmitter",
        name="radiation_emitter",
        position=emitter_pos,
        radius=float(config.get("radius", 1.4)),
        color=config.get("color", [1.0, 0.04, 0.0]),
    )
    emitter_light = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/RadiationEmitter/SignalLight"))
    emitter_light.CreateIntensityAttr(float(config.get("light_intensity", 1800.0)))
    emitter_light.CreateRadiusAttr(float(config.get("light_radius", 2.2)))
    emitter_light.CreateColorAttr(Gf.Vec3f(1.0, 0.05, 0.0))


def _spawn_drone(stage, cuboid_cls, sphere_cls, idx: int, pos: np.ndarray, config: dict[str, Any] | None = None) -> dict:
    from pxr import Gf, Sdf, UsdGeom

    config = config or {}
    faction = str(config.get("faction", "Blue"))
    body_color = np.array([0.20, 0.34, 0.58]) if faction == "Blue" else np.array([0.58, 0.24, 0.20])
    beacon_color = np.array([0.05, 0.65, 1.0]) if faction == "Blue" else np.array([1.0, 0.16, 0.08])
    root_path = f"/World/UAV_{idx}"
    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
    root_xf = UsdGeom.Xformable(root.GetPrim())
    root_xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    asset_path = _resolve_asset_path(config.get("uav_asset_path"))
    if asset_path is None:
        asset_path = _resolve_builtin_uav_asset()
        if asset_path:
            print(f"[QL] UAV asset: using standby NTNU ARL asset {asset_path}")
    if asset_path and _add_usd_reference(stage, asset_path, f"{root_path}/Model"):
        asset_scale = float(config.get("uav_asset_scale", 2.5))
        root_xf.AddScaleOp().Set(Gf.Vec3f(asset_scale, asset_scale, asset_scale))
        print(f"[QL] UAV asset loaded: {asset_path}")
        beacon = sphere_cls(
            prim_path=f"{root_path}/Beacon",
            name=f"uav_{idx}_beacon",
            position=np.array([0.0, 0.0, 0.85]),
            radius=0.22,
            color=beacon_color,
        )
        return {
            "kind": "uav_asset",
            "root_path": root_path,
            "root_prim": root.GetPrim(),
            "parts": [beacon],
            "base_offsets": [np.array([0.0, 0.0, 0.85])],
            "rotor_blades": [],
        }

    print("[QL][WARN] UAV asset not found; using procedural fallback drone")
    fallback_scale = float(config.get("uav_visual_scale", 1.0))
    if abs(fallback_scale - 1.0) > 1e-6:
        root_xf.AddScaleOp().Set(Gf.Vec3f(fallback_scale, fallback_scale, fallback_scale))

    body = cuboid_cls(
        prim_path=f"{root_path}/Body",
        name=f"uav_{idx}_body",
        position=np.array([0.0, 0.0, 0.0]),
        size=1.0,
        scale=np.array([2.2, 1.05, 0.42]),
        color=body_color,
    )
    parts = [body]
    part_offsets = [np.array([0.0, 0.0, 0.0])]
    rotor_blades = []
    for j, (dx, dy) in enumerate([(-0.85, -0.85), (-0.85, 0.85), (0.85, -0.85), (0.85, 0.85)]):
        arm_scale = np.array([2.4, 0.18, 0.12]) if j < 2 else np.array([0.18, 2.4, 0.12])
        arm_offset = np.array([dx * 0.75, dy * 0.75, 0.0])
        parts.append(
            cuboid_cls(
                prim_path=f"{root_path}/Arm_{j}",
                name=f"uav_{idx}_arm_{j}",
                position=arm_offset,
                size=1.0,
                scale=arm_scale,
                color=np.array([0.18, 0.22, 0.28]),
            )
        )
        part_offsets.append(arm_offset)
        rotor_offset = np.array([dx * 1.35, dy * 1.35, 0.14])
        parts.append(
            sphere_cls(
                prim_path=f"{root_path}/Rotor_{j}",
                name=f"uav_{idx}_rotor_{j}",
                position=rotor_offset,
                radius=0.38,
                color=np.array([0.03, 0.03, 0.03]),
            )
        )
        part_offsets.append(rotor_offset)
        for blade_idx, blade_scale in enumerate([np.array([1.25, 0.075, 0.035]), np.array([0.075, 1.25, 0.035])]):
            blade = cuboid_cls(
                prim_path=f"{root_path}/Rotor_{j}_Blade_{blade_idx}",
                name=f"uav_{idx}_rotor_{j}_blade_{blade_idx}",
                position=rotor_offset + np.array([0.0, 0.0, 0.02]),
                size=1.0,
                scale=blade_scale,
                color=np.array([0.02, 0.02, 0.02]),
            )
            parts.append(blade)
            part_offsets.append(rotor_offset + np.array([0.0, 0.0, 0.02]))
            rotor_blades.append({"part": blade, "offset": rotor_offset + np.array([0.0, 0.0, 0.02]), "phase": blade_idx * 90.0 + j * 22.5})
            rotor_blades[-1]["prim"] = stage.GetPrimAtPath(f"{root_path}/Rotor_{j}_Blade_{blade_idx}")
    beacon = sphere_cls(
        prim_path=f"{root_path}/Beacon",
        name=f"uav_{idx}_beacon",
        position=np.array([0.55, 0.0, 0.38]),
        radius=0.18,
        color=beacon_color,
    )
    parts.append(beacon)
    part_offsets.append(np.array([0.55, 0.0, 0.38]))
    return {
        "kind": "uav",
        "root_path": root_path,
        "root_prim": root.GetPrim(),
        "parts": parts,
        "base_offsets": part_offsets,
        "rotor_blades": rotor_blades,
    }


def _spawn_helicopter(stage, cuboid_cls, sphere_cls, idx: int, pos: np.ndarray, config: dict[str, Any] | None = None) -> dict:
    from pxr import Gf, Sdf, UsdGeom

    config = config or {}
    asset_path = _resolve_asset_path(config.get("helicopter_asset_path") or "/home/isaac/ql/asset/NASA/Ingenuity/ingenuity.usd")
    asset_scale = float(config.get("helicopter_asset_scale", 1.0))
    root_path = f"/World/Helicopter_{idx}"
    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
    root_xf = UsdGeom.Xformable(root.GetPrim())
    root_xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    if asset_path and _add_usd_reference(stage, asset_path, f"{root_path}/Model"):
        root_xf.AddScaleOp().Set(Gf.Vec3f(asset_scale, asset_scale, asset_scale))
        beacon = sphere_cls(
            prim_path=f"{root_path}/Beacon",
            name=f"helicopter_{idx}_beacon",
            position=pos + np.array([0.55, 0.0, 0.52]),
            radius=0.12,
            color=np.array([0.95, 0.06, 0.02]),
        )
        print(f"[QL] Helicopter asset loaded: {asset_path}")
        return {
            "kind": "helicopter_asset",
            "root_path": root_path,
            "root_prim": root.GetPrim(),
            "parts": [beacon],
            "base_offsets": [np.array([0.55, 0.0, 0.52])],
            "rotor_blades": [],
        }

    specs = [
        ("Body", [0.0, 0.0, 0.0], [1.9, 0.65, 0.55], [0.86, 0.88, 0.84]),
        ("Nose", [1.05, 0.0, 0.03], [0.65, 0.52, 0.42], [0.74, 0.85, 0.95]),
        ("Tail", [-1.75, 0.0, 0.06], [2.0, 0.18, 0.18], [0.86, 0.88, 0.84]),
        ("Fin", [-2.72, 0.0, 0.42], [0.16, 0.52, 0.85], [0.95, 0.20, 0.14]),
        ("Mast", [0.0, 0.0, 0.58], [0.12, 0.12, 0.48], [0.18, 0.18, 0.18]),
        ("RotorA", [0.0, 0.0, 0.88], [4.2, 0.10, 0.05], [0.04, 0.04, 0.04]),
        ("RotorB", [0.0, 0.0, 0.88], [0.10, 4.2, 0.05], [0.04, 0.04, 0.04]),
        ("SkidL", [0.15, -0.55, -0.45], [1.8, 0.08, 0.08], [0.18, 0.18, 0.18]),
        ("SkidR", [0.15, 0.55, -0.45], [1.8, 0.08, 0.08], [0.18, 0.18, 0.18]),
    ]
    parts = []
    offsets = []
    rotor_blades = []
    for name, offset, scale, color in specs:
        offsets.append(np.array(offset))
        parts.append(
            cuboid_cls(
                prim_path=f"{root_path}/{name}",
                name=f"helicopter_{idx}_{name.lower()}",
                position=pos + np.array(offset),
                size=1.0,
                scale=np.array(scale),
                color=np.array(color),
            )
        )
        if name in {"RotorA", "RotorB"}:
            rotor_blades.append(
                {
                    "part": parts[-1],
                    "offset": np.array(offset),
                    "phase": 0.0 if name == "RotorA" else 90.0,
                    "prim": stage.GetPrimAtPath(f"{root_path}/{name}"),
                }
            )
    offsets.append(np.array([0.55, 0.0, 0.52]))
    parts.append(
        sphere_cls(
            prim_path=f"{root_path}/Beacon",
            name=f"helicopter_{idx}_beacon",
            position=pos + offsets[-1],
            radius=0.12,
            color=np.array([0.95, 0.06, 0.02]),
        )
    )
    return {"kind": "helicopter", "root_path": root_path, "root_prim": root.GetPrim(), "parts": parts, "base_offsets": offsets, "rotor_blades": rotor_blades}


def _rotate_offset(offset: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))

    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)

    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    return rz @ ry @ rx @ np.array(offset, dtype=float)


def _set_visual_pose(visual: dict, pos: np.ndarray, t: float = 0.0, motion_state: MotionState | None = None) -> None:
    yaw_deg = 0.0 if motion_state is None else float(motion_state.yaw_deg)
    pitch_deg = 0.0 if motion_state is None else float(motion_state.pitch_deg)
    roll_deg = 0.0 if motion_state is None else float(motion_state.roll_deg)
    visual["_last_motion_state"] = motion_state
    if visual.get("root_prim") is not None:
        visual["root_prim"].GetAttribute("xformOp:translate").Set((float(pos[0]), float(pos[1]), float(pos[2])))
        rotate_attr = visual["root_prim"].GetAttribute("xformOp:rotateXYZ")
        if not rotate_attr:
            from pxr import UsdGeom

            rotate_attr = UsdGeom.Xformable(visual["root_prim"]).AddRotateXYZOp()
        rotate_attr.Set((float(roll_deg), float(pitch_deg), float(yaw_deg)))
        _spin_rotor_blades(visual, pos, t)
        return
    for part, offset in zip(visual["parts"], visual["base_offsets"]):
        rotated_offset = _rotate_offset(np.array(offset, dtype=float), yaw_deg, pitch_deg, roll_deg)
        part.set_world_pose(position=(pos + rotated_offset).tolist())
    _spin_rotor_blades(visual, pos, t)


def _spin_rotor_blades(visual: dict, pos: np.ndarray, t: float) -> None:
    from pxr import Gf, UsdGeom

    spin_deg = (t * 2400.0) % 360.0
    motion_state = visual.get("_last_motion_state")
    yaw_deg = 0.0 if motion_state is None else float(motion_state.yaw_deg)
    pitch_deg = 0.0 if motion_state is None else float(motion_state.pitch_deg)
    roll_deg = 0.0 if motion_state is None else float(motion_state.roll_deg)
    for blade in visual.get("rotor_blades", []):
        part = blade["part"]
        offset = blade["offset"]
        angle = spin_deg + float(blade.get("phase", 0.0))
        if visual.get("root_prim") is None:
            rotated_offset = _rotate_offset(np.array(offset, dtype=float), yaw_deg, pitch_deg, roll_deg)
            try:
                part.set_world_pose(position=(pos + rotated_offset).tolist(), orientation=None)
            except TypeError:
                part.set_world_pose(position=(pos + rotated_offset).tolist())
        prim = blade.get("prim") or getattr(part, "prim", None)
        if prim is not None:
            attr = prim.GetAttribute("xformOp:rotateXYZ")
            if not attr:
                attr = UsdGeom.Xformable(prim).AddRotateXYZOp()
            attr.Set(Gf.Vec3f(0.0, 0.0, float(angle)))
