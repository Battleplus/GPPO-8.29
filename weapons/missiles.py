from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _norm(vec: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length <= 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return vec / length


def _xyz_rotation_from_direction(direction: np.ndarray) -> tuple[float, float, float]:
    d = _norm(direction)
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    pitch = math.degrees(math.atan2(float(d[2]), math.hypot(float(d[0]), float(d[1]))))
    return 0.0, -pitch, yaw


@dataclass
class Missile:
    root_prim: object
    flame_prim: object
    flash_prim: object
    start: np.ndarray
    target: np.ndarray
    launch_t: float
    speed: float
    active: bool = False
    impacted: bool = False

    @property
    def flight_time(self) -> float:
        return max(0.001, float(np.linalg.norm(self.target - self.start)) / max(1.0, self.speed))


class MissileSystem:
    def __init__(self, stage, cuboid_cls, sphere_cls) -> None:
        self.stage = stage
        self.cuboid_cls = cuboid_cls
        self.sphere_cls = sphere_cls
        self.missiles: list[Missile] = []

    def _set_color(self, prim, color: tuple[float, float, float]) -> None:
        from pxr import Gf, UsdGeom

        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])

    def create_missile(
        self,
        name: str,
        start: np.ndarray,
        target: np.ndarray,
        launch_t: float,
        speed: float = 42.0,
    ) -> Missile:
        from pxr import Gf, Sdf, UsdGeom

        root_path = f"/World/Missiles/{name}"
        root = UsdGeom.Xform.Define(self.stage, Sdf.Path(root_path))
        root_xf = UsdGeom.Xformable(root.GetPrim())
        root_xf.AddTranslateOp().Set(Gf.Vec3d(float(start[0]), float(start[1]), float(start[2])))
        root_xf.AddRotateXYZOp().Set(Gf.Vec3f(*_xyz_rotation_from_direction(target - start)))

        body = UsdGeom.Cylinder.Define(self.stage, Sdf.Path(f"{root_path}/Body"))
        body.CreateRadiusAttr(0.16)
        body.CreateHeightAttr(2.8)
        body_xf = UsdGeom.Xformable(body.GetPrim())
        body_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        body_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))
        self._set_color(body.GetPrim(), (0.62, 0.64, 0.58))

        nose = UsdGeom.Cone.Define(self.stage, Sdf.Path(f"{root_path}/NoseCone"))
        nose.CreateRadiusAttr(0.17)
        nose.CreateHeightAttr(0.62)
        nose_xf = UsdGeom.Xformable(nose.GetPrim())
        nose_xf.AddTranslateOp().Set(Gf.Vec3d(1.70, 0.0, 0.0))
        nose_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))
        self._set_color(nose.GetPrim(), (0.15, 0.16, 0.15))

        nozzle = UsdGeom.Cylinder.Define(self.stage, Sdf.Path(f"{root_path}/Nozzle"))
        nozzle.CreateRadiusAttr(0.13)
        nozzle.CreateHeightAttr(0.16)
        nozzle_xf = UsdGeom.Xformable(nozzle.GetPrim())
        nozzle_xf.AddTranslateOp().Set(Gf.Vec3d(-1.46, 0.0, 0.0))
        nozzle_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))
        self._set_color(nozzle.GetPrim(), (0.04, 0.04, 0.04))

        for idx, (offset, scale) in enumerate(
            [
                ([-1.05, 0.0, 0.24], [0.52, 0.045, 0.34]),
                ([-1.05, 0.0, -0.24], [0.52, 0.045, 0.34]),
                ([-1.05, 0.24, 0.0], [0.52, 0.34, 0.045]),
                ([-1.05, -0.24, 0.0], [0.52, 0.34, 0.045]),
            ]
        ):
            self.cuboid_cls(
                prim_path=f"{root_path}/TailFin_{idx}",
                name=f"{name}_tail_fin_{idx}",
                position=np.array(offset),
                size=1.0,
                scale=np.array(scale),
                color=np.array([0.10, 0.11, 0.10]),
            )

        for idx, (offset, scale) in enumerate(
            [
                ([0.45, 0.0, 0.21], [0.34, 0.035, 0.22]),
                ([0.45, 0.0, -0.21], [0.34, 0.035, 0.22]),
            ]
        ):
            self.cuboid_cls(
                prim_path=f"{root_path}/Canard_{idx}",
                name=f"{name}_canard_{idx}",
                position=np.array(offset),
                size=1.0,
                scale=np.array(scale),
                color=np.array([0.14, 0.15, 0.14]),
            )

        flame = UsdGeom.Cone.Define(self.stage, Sdf.Path(f"{root_path}/ExhaustFlame"))
        flame.CreateRadiusAttr(0.14)
        flame.CreateHeightAttr(0.55)
        flame_xf = UsdGeom.Xformable(flame.GetPrim())
        flame_xf.AddTranslateOp().Set(Gf.Vec3d(-1.82, 0.0, 0.0))
        flame_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, -90.0, 0.0))
        flame_xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        self._set_color(flame.GetPrim(), (1.0, 0.34, 0.04))

        self.sphere_cls(
            prim_path=f"{root_path}/ImpactFlash",
            name=f"{name}_impact_flash",
            position=np.array([0.0, 0.0, -1000.0]),
            radius=0.05,
            color=np.array([1.0, 0.62, 0.06]),
        )
        flash_prim = self.stage.GetPrimAtPath(f"{root_path}/ImpactFlash")
        if not flash_prim.GetAttribute("xformOp:scale"):
            UsdGeom.Xformable(flash_prim).AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        missile = Missile(
            root_prim=root.GetPrim(),
            flame_prim=flame.GetPrim(),
            flash_prim=flash_prim,
            start=np.array(start, dtype=float),
            target=np.array(target, dtype=float),
            launch_t=float(launch_t),
            speed=float(speed),
        )
        self.missiles.append(missile)
        return missile

    def update(self, t: float) -> None:
        from pxr import Gf

        for missile in self.missiles:
            elapsed = float(t) - missile.launch_t
            translate_attr = missile.root_prim.GetAttribute("xformOp:translate")
            rotate_attr = missile.root_prim.GetAttribute("xformOp:rotateXYZ")
            flame_scale_attr = missile.flame_prim.GetAttribute("xformOp:scale")
            flash_translate_attr = missile.flash_prim.GetAttribute("xformOp:translate")
            flash_scale_attr = missile.flash_prim.GetAttribute("xformOp:scale")

            if elapsed < 0.0:
                translate_attr.Set(Gf.Vec3d(float(missile.start[0]), float(missile.start[1]), -1000.0))
                if flame_scale_attr:
                    flame_scale_attr.Set(Gf.Vec3f(0.01, 0.01, 0.01))
                continue

            alpha = min(1.0, elapsed / missile.flight_time)
            pos = missile.start + (missile.target - missile.start) * alpha
            translate_attr.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            rotate_attr.Set(Gf.Vec3f(*_xyz_rotation_from_direction(missile.target - missile.start)))
            missile.active = alpha < 1.0
            if flame_scale_attr:
                flame_size = 1.0 if missile.active else 0.01
                flame_scale_attr.Set(Gf.Vec3f(float(flame_size), float(flame_size), float(flame_size)))

            if alpha >= 1.0 and not missile.impacted:
                flash_translate_attr.Set(Gf.Vec3d(float(missile.target[0]), float(missile.target[1]), float(missile.target[2])))
                if flash_scale_attr:
                    flash_scale_attr.Set(Gf.Vec3f(12.0, 12.0, 12.0))
                missile.impacted = True
            elif missile.impacted:
                fade = max(0.05, 12.0 * (1.0 - min(1.0, elapsed - missile.flight_time)))
                if flash_scale_attr:
                    flash_scale_attr.Set(Gf.Vec3f(float(fade), float(fade), float(fade)))
