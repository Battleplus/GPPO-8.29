from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


QL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = QL_ROOT.parent

TerrainHeightFn = Callable[[float, float, float, float], float]


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


def create_ground_units(
    stage,
    cuboid_cls,
    sphere_cls,
    config: dict[str, Any],
    map_size: float,
    height_scale: float,
    water_cfg: dict[str, Any],
    terrain_height_fn: TerrainHeightFn,
) -> list[dict[str, Any]]:
    tank_count = int(config.get("tanks_each_side", 8))
    sub_count = int(config.get("submarines_each_side", 2))
    uuv_count = int(config.get("uuv_count", 4))
    tank_visual_scale = float(config.get("tank_visual_scale", 1.0))
    tank_ground_offset = float(config.get("tank_ground_offset", 2.20))
    tank_footprint_x = 9.0 * tank_visual_scale
    tank_footprint_y = 5.8 * tank_visual_scale
    spawned: list[dict[str, Any]] = []

    left_road = _tank_display_positions("Blue", tank_count)
    right_road = _tank_display_positions("Red", tank_count)

    for i in range(tank_count):
        lx, ly = left_road[i]
        rx, ry = right_road[i]
        lz = _area_ground_z(lx, ly, tank_footprint_x, tank_footprint_y, map_size, height_scale, tank_ground_offset, terrain_height_fn)
        rz = _area_ground_z(rx, ry, tank_footprint_x, tank_footprint_y, map_size, height_scale, tank_ground_offset, terrain_height_fn)
        spawned.append(
            {
                "entity_id": f"Tank_Blue_{i}",
                "domain": "ground",
                "category": "tank",
                "faction": "Blue",
                "visual": spawn_tank(stage, cuboid_cls, sphere_cls, i, np.array([lx, ly, lz]), "Blue", config),
                "spawn": np.array([lx, ly, lz], dtype=float),
                "path_points": left_road,
                "path_phase": min(0.92, 0.08 + 0.08 * i),
                "signature": {"eo": 0.92, "sar": 0.86, "arm": 0.08},
            }
        )
        spawned.append(
            {
                "entity_id": f"Tank_Red_{i}",
                "domain": "ground",
                "category": "tank",
                "faction": "Red",
                "visual": spawn_tank(stage, cuboid_cls, sphere_cls, i, np.array([rx, ry, rz]), "Red", config),
                "spawn": np.array([rx, ry, rz], dtype=float),
                "path_points": right_road,
                "path_phase": min(0.92, 0.08 + 0.08 * i),
                "signature": {"eo": 0.92, "sar": 0.86, "arm": 0.08},
            }
        )

    river_z = float(water_cfg.get("z", 0.22))
    river_path = _path_points(water_cfg.get("river_path"))
    for i in range(sub_count):
        if river_path:
            x, y = _polyline_point(river_path, 0.18 + i * 0.10)
        else:
            x = -8.0 + i * 16.0
            y = -72.0 + i * 28.0
        spawned.append(
            {
                "entity_id": f"Submarine_Blue_{i}",
                "domain": "surface",
                "category": "submarine",
                "faction": "Blue",
                "visual": spawn_submarine(stage, cuboid_cls, i, np.array([x, y, river_z + 0.55]), "Blue"),
                "spawn": np.array([x, y, river_z + 0.55], dtype=float),
                "path_points": river_path or [(x, y), (x + 16.0, y + 22.0)],
                "path_phase": 0.18 + i * 0.10,
                "signature": {"eo": 0.58, "sar": 0.80, "arm": 0.02},
            }
        )
    for i in range(sub_count):
        if river_path:
            x, y = _polyline_point(river_path, 0.68 + i * 0.10)
        else:
            x = 274.0 + i * 12.0
            y = -120.0 + i * 28.0
        spawned.append(
            {
                "entity_id": f"Submarine_Red_{i}",
                "domain": "surface",
                "category": "submarine",
                "faction": "Red",
                "visual": spawn_submarine(stage, cuboid_cls, i, np.array([x, y, river_z + 0.55]), "Red"),
                "spawn": np.array([x, y, river_z + 0.55], dtype=float),
                "path_points": river_path or [(x, y), (x - 16.0, y - 22.0)],
                "path_phase": 0.68 + i * 0.10,
                "signature": {"eo": 0.58, "sar": 0.80, "arm": 0.02},
            }
        )

    for i in range(uuv_count):
        if river_path and uuv_count > 1:
            x, y = _polyline_point(river_path, 0.16 + i * (0.68 / (uuv_count - 1)))
        elif river_path:
            x, y = _polyline_point(river_path, 0.50)
        else:
            y = -136.0 + i * 58.0
            x = 12.0 * math.sin(y * 0.024) - 8.0 * math.sin(y * 0.011)
        spawned.append(
            {
                "entity_id": f"UUV_{i}",
                "domain": "subsurface",
                "category": "uuv",
                "faction": "Blue",
                "visual": spawn_uuv(stage, cuboid_cls, sphere_cls, i, np.array([x, y, river_z + 1.15]), config),
                "spawn": np.array([x, y, river_z + 1.15], dtype=float),
                "path_points": river_path or [(x, y), (x + 18.0, y + 12.0)],
                "path_phase": 0.16 + (0.68 / max(1, uuv_count - 1)) * i if river_path else 0.4,
                "signature": {"eo": 0.12, "sar": 0.28, "arm": 0.0},
            }
        )

    return spawned


def _tank_display_positions(faction: str, count: int) -> list[tuple[float, float]]:
    columns = 5
    x_step = 18.0
    y_step = 22.0
    stagger = 7.0
    positions: list[tuple[float, float]] = []
    for i in range(max(0, int(count))):
        if faction == "Blue":
            field_positions = [
                (-286.0, -148.0),
                (-252.0, -96.0),
                (-304.0, -34.0),
                (-226.0, 42.0),
                (-174.0, -126.0),
                (-154.0, -22.0),
                (-106.0, -164.0),
                (-86.0, 22.0),
                (-42.0, -92.0),
                (-58.0, 112.0),
            ]
        else:
            field_positions = [
                (286.0, 148.0),
                (252.0, 96.0),
                (304.0, 34.0),
                (226.0, -42.0),
                (174.0, 126.0),
                (154.0, 22.0),
                (106.0, 164.0),
                (86.0, -22.0),
                (42.0, 92.0),
                (58.0, -112.0),
            ]
        if i < len(field_positions):
            x, y = field_positions[i]
        else:
            row = i // columns
            col = i % columns
            x = field_positions[-1][0] + (col - 2) * x_step + row * stagger
            y = field_positions[-1][1] + row * y_step
        positions.append((x, y))
    return positions


def spawn_tank(stage, cuboid_cls, sphere_cls, idx: int, pos: np.ndarray, faction: str, config: dict[str, Any]) -> dict:
    from pxr import Gf, Sdf, UsdGeom

    root_path = f"/World/Tanks/{faction}_{idx}"
    root, root_xf = _make_root(stage, root_path, pos)
    yaw_to_center_deg = math.degrees(math.atan2(float(-pos[1]), float(-pos[0])))
    root_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float(yaw_to_center_deg)))
    visual_scale = float(config.get("tank_visual_scale", 1.0))

    def s(value: float) -> float:
        return float(value) * visual_scale

    def sv(values: tuple[float, float, float] | list[float]) -> np.ndarray:
        return np.array([s(float(value)) for value in values], dtype=float)

    asset_path = _resolve_asset_path(config.get("tank_asset_path"))
    if asset_path and _add_usd_reference(stage, asset_path, f"{root_path}/Model"):
        scale = float(config.get("tank_asset_scale", 1.0)) * visual_scale
        root_xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        marker = sphere_cls(
            prim_path=f"{root_path}/FactionMarker",
            name=f"{faction.lower()}_tank_{idx}_marker",
            position=np.array([0.0, 0.0, s(2.0)]),
            radius=s(0.35),
            color=np.array([0.05, 0.35, 1.0]) if faction == "Blue" else np.array([1.0, 0.14, 0.08]),
        )
        return {"kind": "tank_asset", "root_path": root_path, "root_prim": root.GetPrim(), "parts": [marker], "base_offsets": [np.array([0.0, 0.0, s(2.0)])]}

    hull_color = np.array([0.24, 0.36, 0.20]) if faction == "Blue" else np.array([0.42, 0.28, 0.18])
    turret_color = np.array([0.30, 0.48, 0.22]) if faction == "Blue" else np.array([0.58, 0.32, 0.18])
    underglow_color = (0.05, 0.42, 1.0) if faction == "Blue" else (1.0, 0.16, 0.08)
    underglow_parts = _create_tank_underglow(stage, cuboid_cls, root_path, faction.lower(), idx, sv, s, underglow_color)
    _create_wedge_mesh(stage, f"{root_path}/Hull", (s(8.4), s(4.6), s(1.65)), tuple(hull_color.tolist()), top_scale=0.80)
    _create_wedge_mesh(stage, f"{root_path}/FrontGlacis", (s(3.0), s(4.2), s(0.52)), tuple((hull_color * 1.12).clip(0, 1).tolist()), top_scale=0.62)
    front_xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"{root_path}/FrontGlacis"))
    front_xf.AddTranslateOp().Set(Gf.Vec3d(s(3.15), 0.0, s(0.82)))
    front_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, -10.0, 0.0))
    turret = cuboid_cls(
        prim_path=f"{root_path}/Turret",
        name=f"{faction.lower()}_tank_{idx}_turret",
        position=sv([0.35, 0.0, 1.62]),
        size=1.0,
        scale=sv([3.55, 2.55, 1.22]),
        color=turret_color,
    )
    cupola = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{root_path}/CommanderCupola"))
    cupola.CreateRadiusAttr(s(0.48))
    cupola.CreateHeightAttr(s(0.42))
    cupola_xf = UsdGeom.Xformable(cupola.GetPrim())
    cupola_xf.AddTranslateOp().Set(Gf.Vec3d(s(-0.45), s(0.55), s(2.35)))
    _set_display_color(cupola.GetPrim(), tuple((turret_color * 0.86).tolist()))
    barrel = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{root_path}/Barrel"))
    barrel.CreateRadiusAttr(s(0.22))
    barrel.CreateHeightAttr(s(7.6))
    barrel_xf = UsdGeom.Xformable(barrel.GetPrim())
    barrel_xf.AddTranslateOp().Set(Gf.Vec3d(s(4.95), 0.0, s(1.96)))
    barrel_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))
    _set_display_color(barrel.GetPrim(), (0.12, 0.12, 0.11))
    muzzle = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{root_path}/MuzzleBrake"))
    muzzle.CreateRadiusAttr(s(0.32))
    muzzle.CreateHeightAttr(s(0.46))
    muzzle_xf = UsdGeom.Xformable(muzzle.GetPrim())
    muzzle_xf.AddTranslateOp().Set(Gf.Vec3d(s(8.85), 0.0, s(1.96)))
    muzzle_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))
    _set_display_color(muzzle.GetPrim(), (0.10, 0.10, 0.09))
    track_left = cuboid_cls(
        prim_path=f"{root_path}/TrackLeft",
        name=f"{faction.lower()}_tank_{idx}_track_left",
        position=sv([0.0, -2.35, -0.86]),
        size=1.0,
        scale=sv([8.8, 0.76, 0.98]),
        color=np.array([0.10, 0.10, 0.10]),
    )
    track_right = cuboid_cls(
        prim_path=f"{root_path}/TrackRight",
        name=f"{faction.lower()}_tank_{idx}_track_right",
        position=sv([0.0, 2.35, -0.86]),
        size=1.0,
        scale=sv([8.8, 0.76, 0.98]),
        color=np.array([0.10, 0.10, 0.10]),
    )
    skirt_left = cuboid_cls(
        prim_path=f"{root_path}/SideSkirtLeft",
        name=f"{faction.lower()}_tank_{idx}_skirt_left",
        position=sv([0.0, -2.75, -0.28]),
        size=1.0,
        scale=sv([8.7, 0.20, 0.76]),
        color=hull_color * 0.74,
    )
    skirt_right = cuboid_cls(
        prim_path=f"{root_path}/SideSkirtRight",
        name=f"{faction.lower()}_tank_{idx}_skirt_right",
        position=sv([0.0, 2.75, -0.28]),
        size=1.0,
        scale=sv([8.7, 0.20, 0.76]),
        color=hull_color * 0.74,
    )
    for side, y in [("L", -2.48), ("R", 2.48)]:
        for wheel_idx, x in enumerate([-3.1, -1.85, -0.6, 0.65, 1.9, 3.15]):
            wheel = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{root_path}/Wheel_{side}_{wheel_idx}"))
            wheel.CreateRadiusAttr(s(0.48))
            wheel.CreateHeightAttr(s(0.22))
            wheel_xf = UsdGeom.Xformable(wheel.GetPrim())
            wheel_xf.AddTranslateOp().Set(Gf.Vec3d(s(float(x)), s(float(y)), s(-0.92)))
            wheel_xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))
            _set_display_color(wheel.GetPrim(), (0.05, 0.05, 0.05))
    antenna = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{root_path}/Antenna"))
    antenna.CreateRadiusAttr(s(0.045))
    antenna.CreateHeightAttr(s(2.6))
    antenna_xf = UsdGeom.Xformable(antenna.GetPrim())
    antenna_xf.AddTranslateOp().Set(Gf.Vec3d(s(-1.30), s(-0.72), s(3.30)))
    antenna_xf.AddRotateXYZOp().Set(Gf.Vec3f(8.0, 0.0, 0.0))
    _set_display_color(antenna.GetPrim(), (0.05, 0.05, 0.04))
    marker = sphere_cls(
        prim_path=f"{root_path}/FactionMarker",
        name=f"{faction.lower()}_tank_{idx}_marker",
        position=sv([-2.3, 0.0, 1.62]),
        radius=s(0.28),
        color=np.array([0.05, 0.35, 1.0]) if faction == "Blue" else np.array([1.0, 0.14, 0.08]),
    )
    return {
        "kind": "tank",
        "root_path": root_path,
        "root_prim": root.GetPrim(),
        "parts": [turret, track_left, track_right, skirt_left, skirt_right, *underglow_parts, marker],
        "base_offsets": [
            sv([0.2, 0.0, 1.0]),
            sv([0.0, -1.5, -0.45]),
            sv([0.0, 1.5, -0.45]),
            sv([0.0, -2.75, -0.28]),
            sv([0.0, 2.75, -0.28]),
            sv([0.0, -3.20, -0.72]),
            sv([0.0, 3.20, -0.72]),
            sv([4.35, 0.0, -0.72]),
            sv([-4.35, 0.0, -0.72]),
            sv([-2.3, 0.0, 1.62]),
        ],
    }


def _create_tank_underglow(stage, cuboid_cls, root_path: str, faction_name: str, idx: int, sv, s, color: tuple[float, float, float]) -> list[object]:
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    glow_np = np.array(color, dtype=float)
    strip_specs = [
        ("LeftStrip", [0.0, -3.20, -0.72], [7.8, 0.18, 0.16]),
        ("RightStrip", [0.0, 3.20, -0.72], [7.8, 0.18, 0.16]),
        ("FrontStrip", [4.35, 0.0, -0.72], [0.18, 4.4, 0.16]),
        ("RearStrip", [-4.35, 0.0, -0.72], [0.18, 4.4, 0.16]),
    ]
    parts = []
    for name, position, scale in strip_specs:
        strip = cuboid_cls(
            prim_path=f"{root_path}/Underglow{name}",
            name=f"{faction_name}_tank_{idx}_underglow_{name.lower()}",
            position=sv(position),
            size=1.0,
            scale=sv(scale),
            color=glow_np,
        )
        parts.append(strip)
        prim = stage.GetPrimAtPath(f"{root_path}/Underglow{name}")
        if prim:
            _bind_emissive_material(stage, prim, f"{root_path}/Underglow{name}_Material", color, intensity=5.0)

    light = UsdLux.SphereLight.Define(stage, Sdf.Path(f"{root_path}/UnderglowLight"))
    light.CreateColorAttr(Gf.Vec3f(*color))
    light.CreateIntensityAttr(1800.0)
    light.CreateRadiusAttr(s(6.2))
    UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, s(-0.55)))
    return parts


def _bind_emissive_material(stage, prim, material_path: str, color: tuple[float, float, float], intensity: float) -> None:
    try:
        from pxr import Gf, Sdf, UsdShade

        material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
        shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(color[0] * intensity), float(color[1] * intensity), float(color[2] * intensity))
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    except Exception:
        pass


def spawn_submarine(stage, cuboid_cls, idx: int, pos: np.ndarray, faction: str) -> dict:
    from pxr import Gf, Sdf, UsdGeom

    root_path = f"/World/Submarines/{faction}_{idx}"
    root, _root_xf = _make_root(stage, root_path, pos)

    hull_color = np.array([0.08, 0.12, 0.14]) if faction == "Blue" else np.array([0.12, 0.10, 0.08])
    body = cuboid_cls(
        prim_path=f"{root_path}/Hull",
        name=f"{faction.lower()}_sub_{idx}_hull",
        position=np.array([0.0, 0.0, 0.0]),
        size=1.0,
        scale=np.array([7.2, 1.4, 1.3]),
        color=hull_color,
    )
    tower = cuboid_cls(
        prim_path=f"{root_path}/Tower",
        name=f"{faction.lower()}_sub_{idx}_tower",
        position=np.array([1.0, 0.0, 0.95]),
        size=1.0,
        scale=np.array([1.2, 0.82, 1.2]),
        color=np.array([0.12, 0.16, 0.18]),
    )
    fin = cuboid_cls(
        prim_path=f"{root_path}/Fin",
        name=f"{faction.lower()}_sub_{idx}_fin",
        position=np.array([-2.8, 0.0, -0.2]),
        size=1.0,
        scale=np.array([1.0, 0.18, 1.1]),
        color=np.array([0.14, 0.18, 0.20]),
    )
    sail = cuboid_cls(
        prim_path=f"{root_path}/Sail",
        name=f"{faction.lower()}_sub_{idx}_sail",
        position=np.array([1.0, 0.0, 2.0]),
        size=1.0,
        scale=np.array([1.0, 0.6, 0.2]),
        color=np.array([0.18, 0.20, 0.22]),
    )
    periscope = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{root_path}/Periscope"))
    periscope.CreateRadiusAttr(0.08)
    periscope.CreateHeightAttr(1.8)
    periscope_xf = UsdGeom.Xformable(periscope.GetPrim())
    periscope_xf.AddTranslateOp().Set(Gf.Vec3d(1.8, 0.0, 2.7))
    _set_display_color(periscope.GetPrim(), (0.16, 0.16, 0.16))
    return {
        "kind": "submarine",
        "root_path": root_path,
        "root_prim": root.GetPrim(),
        "parts": [body, tower, fin, sail],
        "base_offsets": [
            np.zeros(3),
            np.array([1.0, 0.0, 0.95]),
            np.array([-2.8, 0.0, -0.2]),
            np.array([1.0, 0.0, 2.0]),
        ],
    }


def spawn_uuv(stage, cuboid_cls, sphere_cls, idx: int, pos: np.ndarray, config: dict[str, Any]) -> dict:
    from pxr import Gf, Sdf, UsdGeom

    root_path = f"/World/UUV/UUV_{idx}"
    root, root_xf = _make_root(stage, root_path, pos)
    asset_path = _resolve_asset_path(config.get("uuv_asset_path"))
    if asset_path and _add_usd_reference(stage, asset_path, f"{root_path}/Model"):
        scale = float(config.get("uuv_asset_scale", 1.0))
        root_xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        marker = sphere_cls(
            prim_path=f"{root_path}/Marker",
            name=f"uuv_{idx}_marker",
            position=np.array([0.0, 0.0, 1.2]),
            radius=0.25,
            color=np.array([1.0, 0.78, 0.12]),
        )
        return {"kind": "uuv_asset", "root_path": root_path, "root_prim": root.GetPrim(), "parts": [marker], "base_offsets": [np.array([0.0, 0.0, 1.2])]}

    hull = UsdGeom.Capsule.Define(stage, Sdf.Path(f"{root_path}/Hull"))
    hull.CreateRadiusAttr(0.72)
    hull.CreateHeightAttr(4.9)
    hull_xf = UsdGeom.Xformable(hull.GetPrim())
    hull_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    hull_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))
    _set_display_color(hull.GetPrim(), (0.95, 0.66, 0.08))

    nose = sphere_cls(
        prim_path=f"{root_path}/Nose",
        name=f"uuv_{idx}_nose",
        position=np.array([2.7, 0.0, 0.0]),
        radius=0.74,
        color=np.array([1.0, 0.78, 0.12]),
    )
    sensor = sphere_cls(
        prim_path=f"{root_path}/Sensor",
        name=f"uuv_{idx}_sensor",
        position=np.array([1.3, 0.0, 0.95]),
        radius=0.22,
        color=np.array([0.05, 0.40, 0.95]),
    )
    for ring_idx, (dy, dz, sy, sz) in enumerate(
        [
            (0.0, 0.52, 0.90, 0.055),
            (0.0, -0.52, 0.90, 0.055),
            (-0.52, 0.0, 0.055, 0.90),
            (0.52, 0.0, 0.055, 0.90),
        ]
    ):
        cuboid_cls(
            prim_path=f"{root_path}/TailGuard_{ring_idx}",
            name=f"uuv_{idx}_tail_guard_{ring_idx}",
            position=np.array([-2.95, dy, dz]),
            size=1.0,
            scale=np.array([0.08, sy, sz]),
            color=np.array([0.08, 0.08, 0.08]),
        )

    prop_a = cuboid_cls(
        prim_path=f"{root_path}/PropA",
        name=f"uuv_{idx}_prop_a",
        position=np.array([-3.05, 0.0, 0.0]),
        size=1.0,
        scale=np.array([0.06, 0.90, 0.10]),
        color=np.array([0.06, 0.06, 0.06]),
    )
    prop_b = cuboid_cls(
        prim_path=f"{root_path}/PropB",
        name=f"uuv_{idx}_prop_b",
        position=np.array([-3.05, 0.0, 0.0]),
        size=1.0,
        scale=np.array([0.06, 0.10, 0.90]),
        color=np.array([0.06, 0.06, 0.06]),
    )

    fin_top = cuboid_cls(
        prim_path=f"{root_path}/TopFin",
        name=f"uuv_{idx}_top_fin",
        position=np.array([-1.9, 0.0, 0.78]),
        size=1.0,
        scale=np.array([0.70, 0.12, 0.70]),
        color=np.array([0.16, 0.16, 0.16]),
    )
    fin_side = cuboid_cls(
        prim_path=f"{root_path}/SideFin",
        name=f"uuv_{idx}_side_fin",
        position=np.array([-1.9, 0.0, 0.0]),
        size=1.0,
        scale=np.array([0.72, 1.35, 0.10]),
        color=np.array([0.16, 0.16, 0.16]),
    )
    thruster_l = sphere_cls(
        prim_path=f"{root_path}/ThrusterL",
        name=f"uuv_{idx}_thruster_l",
        position=np.array([-0.8, -0.92, 0.0]),
        radius=0.22,
        color=np.array([0.10, 0.10, 0.10]),
    )
    thruster_r = sphere_cls(
        prim_path=f"{root_path}/ThrusterR",
        name=f"uuv_{idx}_thruster_r",
        position=np.array([-0.8, 0.92, 0.0]),
        radius=0.22,
        color=np.array([0.10, 0.10, 0.10]),
    )
    return {
        "kind": "uuv",
        "root_path": root_path,
        "root_prim": root.GetPrim(),
        "parts": [nose, sensor, prop_a, prop_b, fin_top, fin_side, thruster_l, thruster_r],
        "base_offsets": [
            np.array([2.7, 0.0, 0.0]),
            np.array([1.3, 0.0, 0.95]),
            np.array([-3.05, 0.0, 0.0]),
            np.array([-3.05, 0.0, 0.0]),
            np.array([-1.9, 0.0, 0.78]),
            np.array([-1.9, 0.0, 0.0]),
            np.array([-0.8, -0.92, 0.0]),
            np.array([-0.8, 0.92, 0.0]),
        ],
    }


def _make_root(stage, root_path: str, pos: np.ndarray):
    from pxr import Gf, Sdf, UsdGeom

    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
    root_xf = UsdGeom.Xformable(root.GetPrim())
    root_xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    return root, root_xf


def _area_ground_z(
    x: float,
    y: float,
    sx: float,
    sy: float,
    map_size: float,
    height_scale: float,
    offset: float,
    terrain_height_fn: TerrainHeightFn,
) -> float:
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
    highest = max(terrain_height_fn(px, py, map_size, height_scale) for px, py in samples)
    return highest + offset


def _set_display_color(prim, color: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _resolve_asset_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        candidates = [
            (QL_ROOT / path).resolve(),
            (PROJECT_ROOT / path).resolve(),
        ]
    else:
        candidates = [path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _add_usd_reference(stage, usd_path: str, prim_path: str) -> bool:
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        return bool(stage.GetPrimAtPath(prim_path))
    except Exception as exc:
        print(f"[QL][WARN] Failed to load USD asset {usd_path}: {exc}")
        return False


def _create_wedge_mesh(
    stage,
    prim_path: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    top_scale: float = 0.72,
) -> object:
    from pxr import Gf, Sdf, UsdGeom

    sx, sy, sz = size
    bx, by, bz = sx * 0.5, sy * 0.5, sz * 0.5
    tx, ty = bx * top_scale, by * top_scale
    points = [
        Gf.Vec3f(-bx, -by, -bz),
        Gf.Vec3f(bx, -by, -bz),
        Gf.Vec3f(bx, by, -bz),
        Gf.Vec3f(-bx, by, -bz),
        Gf.Vec3f(-tx, -ty, bz),
        Gf.Vec3f(tx, -ty, bz),
        Gf.Vec3f(tx, ty, bz),
        Gf.Vec3f(-tx, ty, bz),
    ]
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr(
        [
            0, 1, 2, 3,
            4, 7, 6, 5,
            0, 4, 5, 1,
            1, 5, 6, 2,
            2, 6, 7, 3,
            3, 7, 4, 0,
        ]
    )
    mesh.CreateSubdivisionSchemeAttr("none")
    _set_display_color(mesh.GetPrim(), color)
    return mesh.GetPrim()
