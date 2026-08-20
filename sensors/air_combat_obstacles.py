from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np


class ObstacleLike(Protocol):
    obstacle_id: str
    name: str
    category: str
    position: np.ndarray
    radius_units: float
    height_units: float
    priority: int
    blocks_los: bool


class PlatformLike(Protocol):
    entity_id: str
    spec: Any
    motion_model: Any

    @property
    def position(self) -> np.ndarray:
        ...


def compute_obstacle_contacts(
    platforms: list[PlatformLike],
    obstacles: list[ObstacleLike],
    *,
    tactical_time_s: float,
    meters_per_unit: float,
    max_contacts_per_platform: int,
    terrain_mask_fn,
) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    max_contacts_per_platform = max(1, int(max_contacts_per_platform))
    for platform in platforms:
        platform_contacts: list[dict[str, Any]] = []
        observer_pos = platform.position
        for obstacle in obstacles:
            target_pos = np.array(obstacle.position, dtype=float)
            delta = target_pos - observer_pos
            distance_units = float(np.linalg.norm(delta))
            distance_km = distance_units * meters_per_unit / 1000.0
            for sensor in platform.spec.sensors:
                if not sensor_accepts_obstacle(sensor, obstacle.category):
                    continue
                effective_range_km = min(float(sensor.max_range_km), obstacle_sensor_range_km(sensor, obstacle))
                if distance_km > effective_range_km:
                    continue
                if not sensor_scan_contains(sensor, delta, platform_yaw_deg=float(platform.motion_model.state.yaw_deg), tactical_time_s=tactical_time_s):
                    continue
                if obstacle.blocks_los and sensor.channel not in {"sar", "elint"} and terrain_mask_fn(observer_pos, target_pos):
                    continue
                platform_contacts.append(
                    {
                        "platform_id": platform.entity_id,
                        "platform": platform.spec.name,
                        "obstacle_id": obstacle.obstacle_id,
                        "obstacle": obstacle.name,
                        "obstacle_category": obstacle.category,
                        "sensor": sensor.name,
                        "channel": sensor.channel,
                        "azimuth_fov_deg": sensor_value(sensor, "azimuth_fov_deg", 360.0),
                        "elevation_fov_deg": sensor_value(sensor, "elevation_fov_deg", 60.0),
                        "scan_rate_hz": sensor_value(sensor, "scan_rate_hz", 1.0),
                        "distance_km": distance_km,
                        "radius_km": obstacle.radius_units * meters_per_unit / 1000.0,
                        "height_m": obstacle.height_units * meters_per_unit,
                        "position": [float(value) for value in target_pos.tolist()],
                        "position_m": [float(value) * meters_per_unit for value in target_pos.tolist()],
                        "priority": obstacle.priority,
                    }
                )
        platform_contacts.sort(key=lambda item: (-int(item["priority"]), float(item["distance_km"])))
        contacts.extend(platform_contacts[:max_contacts_per_platform])
    contacts.sort(key=lambda item: (str(item["platform_id"]), -int(item["priority"]), float(item["distance_km"])))
    return contacts


def sensor_accepts_obstacle(sensor, category: str) -> bool:
    channel = str(getattr(sensor, "channel", "")).lower()
    modes = {str(mode).lower() for mode in getattr(sensor, "modes", ())}
    target_kinds = tuple(str(kind).lower() for kind in getattr(sensor, "target_kinds", ()))
    category = str(category).lower()
    if category in target_kinds or "obstacle" in target_kinds or "terrain" in target_kinds:
        return True
    if channel == "elint":
        return False
    if channel in {"sar", "mmw_radar"}:
        return category in {"mountain", "tree", "rock"}
    if channel == "eo_ir":
        return category in {"mountain", "tree", "rock"} and (
            "visible" in modes
            or "ir" in modes
            or "terrain_following" in modes
            or "terrain_avoidance" in modes
        )
    return "terrain_following" in modes or "terrain_avoidance" in modes


def obstacle_sensor_range_km(sensor, obstacle: ObstacleLike) -> float:
    channel = str(getattr(sensor, "channel", "")).lower()
    category = obstacle.category
    if category == "mountain":
        multiplier = 1.00 if channel in {"sar", "mmw_radar"} else 0.75
    elif category == "tree":
        multiplier = 0.42 if channel in {"sar", "mmw_radar"} else 0.24
    else:
        multiplier = 0.55 if channel in {"sar", "mmw_radar"} else 0.35
    signature_bonus = min(1.8, max(0.65, float(obstacle.radius_units) / 2.5))
    return max(1.0, float(getattr(sensor, "max_range_km", 0.0)) * multiplier * signature_bonus)


def sensor_scan_contains(sensor, delta: np.ndarray, *, platform_yaw_deg: float, tactical_time_s: float) -> bool:
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        return True

    horizontal = math.hypot(float(delta[0]), float(delta[1]))
    target_azimuth = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    target_elevation = math.degrees(math.atan2(float(delta[2]), max(1e-6, horizontal)))

    azimuth_fov = sensor_value(sensor, "azimuth_fov_deg", 360.0)
    elevation_fov = sensor_value(sensor, "elevation_fov_deg", 60.0)
    scan_rate_default = 0.25 if getattr(sensor, "channel", "") == "elint" else 1.0
    scan_rate = sensor_value(sensor, "scan_rate_hz", scan_rate_default)
    dwell_time = sensor_value(sensor, "dwell_time_s", 0.25)
    yaw_center = float(platform_yaw_deg) + sensor_value(sensor, "boresight_yaw_offset_deg", 0.0)
    if azimuth_fov >= 359.0:
        az_ok = True
    else:
        sweep_center = yaw_center
        scan_rate = max(0.0, scan_rate)
        if scan_rate > 0.0:
            sweep_center += 0.5 * azimuth_fov * math.sin(2.0 * math.pi * scan_rate * tactical_time_s)
        az_ok = abs(angle_error_deg(target_azimuth, sweep_center)) <= max(0.5, azimuth_fov * 0.5)

    pitch_center = sensor_value(sensor, "boresight_pitch_deg", -6.0)
    el_ok = abs(target_elevation - pitch_center) <= max(0.5, elevation_fov * 0.5)
    if not (az_ok and el_ok):
        return False

    scan_period = 1.0 / max(0.05, scan_rate)
    scan_phase = float(tactical_time_s) % scan_period
    return scan_phase <= max(dwell_time, scan_period * 0.35)


def sensor_value(sensor, name: str, default: float) -> float:
    try:
        return float(getattr(sensor, name))
    except Exception:
        return float(default)


def angle_error_deg(angle: float, center: float) -> float:
    return ((float(angle) - float(center) + 180.0) % 360.0) - 180.0
