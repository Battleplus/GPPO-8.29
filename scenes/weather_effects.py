from __future__ import annotations

import random
from typing import Any

import numpy as np


def _spawn_cloud_cluster(
    sphere_cls,
    prim_prefix: str,
    base_idx: int,
    center: tuple[float, float, float],
    radius_scale: float,
    color: np.ndarray,
    rng: random.Random,
) -> int:
    cx, cy, cz = center
    created = 0
    for sub_idx in range(4):
        radius = rng.uniform(12.0, 28.0) * radius_scale * (0.75 if sub_idx > 0 else 1.0)
        dx = rng.uniform(-18.0, 18.0) * radius_scale
        dy = rng.uniform(-12.0, 12.0) * radius_scale
        dz = rng.uniform(-3.0, 3.0) * radius_scale
        sphere = sphere_cls(
            prim_path=f"{prim_prefix}_{base_idx}_{sub_idx}",
            name=f"weather_cloud_{base_idx}_{sub_idx}",
            position=np.array([cx + dx, cy + dy, cz + dz], dtype=float),
            radius=radius,
            color=color,
        )
        try:
            sphere.prim.GetAttribute("xformOp:scale").Set((1.75, 1.15, 0.34))
        except Exception:
            pass
        created += 1
    return created


def _set_opacity(prim, opacity: float) -> None:
    try:
        from pxr import UsdGeom

        UsdGeom.Gprim(prim).CreateDisplayOpacityAttr().Set([float(max(0.0, min(1.0, opacity)))])
    except Exception:
        pass


def _create_cloud_underlights(stage, cloud_cover: float, fog_density: float, rain_rate: float, map_size: float, height_scale: float) -> int:
    if cloud_cover <= 0.08 and fog_density <= 0.10:
        return 0

    from pxr import Gf, Sdf, UsdGeom, UsdLux

    light_count = 0
    base_intensity = 900.0 + 2400.0 * cloud_cover + 1200.0 * fog_density + 700.0 * rain_rate
    light_z = height_scale + 46.0
    width = map_size * 0.62
    height = map_size * 0.46
    positions = [
        (0.0, 0.0, light_z, 1.0),
        (-map_size * 0.16, map_size * 0.12, light_z + 8.0, 0.58),
        (map_size * 0.18, -map_size * 0.10, light_z + 6.0, 0.52),
    ]

    for idx, (x, y, z, scale) in enumerate(positions):
        light_path = Sdf.Path(f"/World/Weather/CloudUnderLight_{idx}")
        light = UsdLux.RectLight.Define(stage, light_path)
        light.CreateIntensityAttr(float(base_intensity * scale))
        light.CreateColorAttr(Gf.Vec3f(0.80, 0.86, 1.0))
        light.CreateWidthAttr(float(width * scale))
        light.CreateHeightAttr(float(height * scale))
        xform = UsdGeom.Xformable(light.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
        light_count += 1

    return light_count


def _create_cloud_cluster_underlight(stage, idx: int, center: tuple[float, float, float], cloud_cover: float) -> int:
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    x, y, z = center
    light = UsdLux.RectLight.Define(stage, Sdf.Path(f"/World/Weather/CloudClusterUnderLight_{idx}"))
    light.CreateIntensityAttr(float(520.0 + 360.0 * cloud_cover))
    light.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
    light.CreateWidthAttr(54.0)
    light.CreateHeightAttr(38.0)
    xform = UsdGeom.Xformable(light.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z - 14.0)))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
    return 1


def create_weather_visuals(stage, cuboid_cls, sphere_cls, weather_cfg: dict[str, Any], map_size: float, height_scale: float, seed: int) -> dict[str, int]:
    from pxr import Sdf, UsdGeom

    UsdGeom.Xform.Define(stage, Sdf.Path("/World/Weather"))

    cloud_cover = float(weather_cfg.get("cloud_cover", 0.0))
    fog_density = float(weather_cfg.get("fog_density", 0.0))
    rain_rate = float(weather_cfg.get("rain_rate", 0.0))
    rng = random.Random(seed + 9917)
    cloud_created = 0
    fog_created = 0
    rain_created = 0
    light_created = _create_cloud_underlights(stage, cloud_cover, fog_density, rain_rate, map_size, height_scale)

    if bool(weather_cfg.get("visual_clouds", True)) and cloud_cover > 0.05:
        # Darker cloud deck so it is visible against Isaac's bright background.
        deck_count = max(3, int(3 + cloud_cover * 7))
        cloud_color = np.array(
            [
                0.96 - 0.03 * cloud_cover,
                0.97 - 0.025 * cloud_cover,
                0.98 - 0.02 * cloud_cover,
            ],
            dtype=float,
        )
        # Central storm ceiling / overcast layer
        for idx in range(max(1, int(1 + cloud_cover * 2))):
            x = rng.uniform(-map_size * 0.12, map_size * 0.12)
            y = rng.uniform(-map_size * 0.12, map_size * 0.12)
            z = height_scale + rng.uniform(80.0, 110.0)
            cuboid_cls(
                prim_path=f"/World/Weather/CloudDeck_{idx}",
                name=f"weather_cloud_deck_{idx}",
                position=np.array([x, y, z], dtype=float),
                size=1.0,
                scale=np.array(
                    [
                        rng.uniform(map_size * 0.12, map_size * 0.20),
                        rng.uniform(map_size * 0.08, map_size * 0.14),
                        rng.uniform(2.0, 4.5),
                    ],
                    dtype=float,
                ),
                color=cloud_color,
            )
            _set_opacity(stage.GetPrimAtPath(f"/World/Weather/CloudDeck_{idx}"), 0.04 + 0.02 * (1.0 - cloud_cover))
            cloud_created += 1
        for idx in range(deck_count):
            x = rng.uniform(-map_size * 0.26, map_size * 0.26)
            y = rng.uniform(-map_size * 0.26, map_size * 0.26)
            z = height_scale + rng.uniform(62.0, 96.0)
            cluster_center = (x, y, z)
            cloud_created += _spawn_cloud_cluster(
                sphere_cls,
                "/World/Weather/Cloud",
                idx,
                cluster_center,
                radius_scale=0.40 + cloud_cover * 0.55,
                color=cloud_color,
                rng=rng,
            )
            light_created += _create_cloud_cluster_underlight(stage, idx, cluster_center, cloud_cover)
            for sub_idx in range(4):
                _set_opacity(stage.GetPrimAtPath(f"/World/Weather/Cloud_{idx}_{sub_idx}"), 0.015 + 0.02 * (1.0 - cloud_cover))

    if fog_density > 0.05:
        fog_color = np.array(
            [
                0.90 + 0.03 * (1.0 - fog_density),
                0.91 + 0.025 * (1.0 - fog_density),
                0.92 + 0.02 * (1.0 - fog_density),
            ],
            dtype=float,
        )
        fog_count = max(3, int(3 + fog_density * 8))
        for idx in range(fog_count):
            x = rng.uniform(-map_size * 0.16, map_size * 0.16)
            y = rng.uniform(-map_size * 0.16, map_size * 0.16)
            z = height_scale * 0.10 + rng.uniform(2.0, 8.0)
            cuboid_cls(
                prim_path=f"/World/Weather/FogBank_{idx}",
                name=f"weather_fog_bank_{idx}",
                position=np.array([x, y, z], dtype=float),
                size=1.0,
                scale=np.array(
                    [
                        rng.uniform(18.0, 48.0) * (0.7 + fog_density),
                        rng.uniform(12.0, 34.0) * (0.7 + fog_density),
                        rng.uniform(2.2, 5.5) * (0.8 + fog_density),
                    ],
                    dtype=float,
                ),
                color=fog_color,
            )
            _set_opacity(stage.GetPrimAtPath(f"/World/Weather/FogBank_{idx}"), 0.012 + 0.02 * (1.0 - fog_density))
            fog_created += 1
        # Battlefield-center fog belt to make low visibility obvious.
        cuboid_cls(
            prim_path="/World/Weather/FogBelt_Main",
            name="weather_fog_belt_main",
            position=np.array([0.0, 0.0, height_scale * 0.10 + 4.0], dtype=float),
            size=1.0,
            scale=np.array([map_size * 0.18, map_size * 0.10, 4.5 + fog_density * 3.0], dtype=float),
            color=fog_color,
        )
        _set_opacity(stage.GetPrimAtPath("/World/Weather/FogBelt_Main"), 0.010 + 0.015 * (1.0 - fog_density))
        fog_created += 1

    if bool(weather_cfg.get("visual_rain", False)) and rain_rate > 0.05:
        rain_color = np.array([0.58, 0.68, 0.80], dtype=float)
        streak_count = max(90, int(110 + rain_rate * 220))
        # Focus rain in the center where the main camera looks.
        for idx in range(streak_count):
            x = rng.uniform(-map_size * 0.18, map_size * 0.18)
            y = rng.uniform(-map_size * 0.18, map_size * 0.18)
            z = height_scale + rng.uniform(12.0, 64.0)
            length = rng.uniform(5.0, 11.0) * (0.8 + rain_rate)
            tilt = rng.uniform(-1.0, 1.0)
            cuboid_cls(
                prim_path=f"/World/Weather/Rain_{idx}",
                name=f"weather_rain_{idx}",
                position=np.array([x + tilt, y, z], dtype=float),
                size=1.0,
                scale=np.array([0.10, 0.10, length], dtype=float),
                color=rain_color,
            )
            rain_created += 1

    return {"clouds": cloud_created, "fog": fog_created, "rain": rain_created, "lights": light_created}
