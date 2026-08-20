from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "renderer": "HydraStorm",
        "width": 1280,
        "height": 720,
    }
)

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from scenes.physx_motion_models import PhysXHelicopterVehicle, PhysXKinematicTracker, _rigid_body_pose


FRIENDLY_HELI_PATH = "/World/Friendly_WZ21"
ENEMY_HELI_PATH = "/World/Enemy_AH64"
DEFAULT_FRIENDLY_HELI_USD = "/home/isaac/ql/asset/WZ21/wz21.usd"
DEFAULT_ENEMY_HELI_USD = "/home/isaac/ql/asset/AH64/apache.usd"
FRIENDLY_START_POS = np.array([-18.0, -8.0, 32.0], dtype=float)
ENEMY_START_POS = np.array([18.0, 8.0, 28.0], dtype=float)
RUN_SECONDS = 30.0
DT = 1.0 / 60.0


@dataclass(frozen=True)
class HelicopterSpec:
    name: str
    env_var: str
    default_usd: str
    length_m: float
    height_m: float
    rotor_diameter_m: float
    empty_mass_kg: float
    payload_kg: float
    max_level_speed_kmh: float
    cruise_speed_kmh: float
    service_ceiling_m: float
    ferry_range_km: float
    combat_radius_km: float
    climb_rate_mps: float
    positive_g_limit: float
    negative_g_limit: float
    color: tuple[float, float, float]
    accent_color: tuple[float, float, float]

    @property
    def cruise_speed_mps(self) -> float:
        return self.cruise_speed_kmh / 3.6

    @property
    def max_level_speed_mps(self) -> float:
        return self.max_level_speed_kmh / 3.6

    @property
    def gross_mass_kg(self) -> float:
        return self.empty_mass_kg + self.payload_kg

    @property
    def max_force_n(self) -> float:
        return self.gross_mass_kg * 9.81 * self.positive_g_limit

    @property
    def max_torque_nm(self) -> float:
        return self.gross_mass_kg * self.length_m * 0.35


FRIENDLY_SPEC = HelicopterSpec(
    name="WZ-21",
    env_var="QL_FRIENDLY_HELI_USD",
    default_usd=DEFAULT_FRIENDLY_HELI_USD,
    length_m=19.4,
    height_m=4.5,
    rotor_diameter_m=16.2,
    empty_mass_kg=6500.0,
    payload_kg=4000.0,
    max_level_speed_kmh=360.0,
    cruise_speed_kmh=290.0,
    service_ceiling_m=6000.0,
    ferry_range_km=1500.0,
    combat_radius_km=560.0,
    climb_rate_mps=7.1,
    positive_g_limit=3.5,
    negative_g_limit=-0.5,
    color=(0.17, 0.26, 0.20),
    accent_color=(0.08, 0.09, 0.07),
)

ENEMY_SPEC = HelicopterSpec(
    name="AH-64 Apache",
    env_var="QL_ENEMY_HELI_USD",
    default_usd=DEFAULT_ENEMY_HELI_USD,
    length_m=15.06,
    height_m=3.87,
    rotor_diameter_m=14.63,
    empty_mass_kg=5165.0,
    payload_kg=43600.0,
    max_level_speed_kmh=293.0,
    cruise_speed_kmh=265.0,
    service_ceiling_m=4572.0,
    ferry_range_km=482.0,
    combat_radius_km=480.0,
    climb_rate_mps=12.7,
    positive_g_limit=3.5,
    negative_g_limit=-0.5,
    color=(0.23, 0.25, 0.16),
    accent_color=(0.06, 0.06, 0.05),
)


def cleanup_physics(stage, root_prim_path):
    root = stage.GetPrimAtPath(root_prim_path)
    if not root:
        return
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
        for api in (
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.ArticulationRootAPI,
            UsdPhysics.CollisionAPI,
            UsdPhysics.MassAPI,
        ):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)


def set_xform(prim, translate=None, rotate_xyz=None, scale=None):
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


def set_display_color(prim, color):
    if prim.IsA(UsdGeom.Gprim):
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def resolve_heli_asset(spec):
    env_path = os.environ.get(spec.env_var)
    if not env_path and spec is FRIENDLY_SPEC:
        env_path = os.environ.get("QL_HELI_USD")
    if env_path:
        expanded = Path(env_path).expanduser()
        return str(expanded) if expanded.exists() else env_path
    default_path = Path(spec.default_usd)
    if default_path.exists():
        return str(default_path)
    return None


def add_collision_box(stage, root_prim_path, size):
    cube_path = root_prim_path + "/CollisionBox"
    cube_geom = UsdGeom.Cube.Define(stage, Sdf.Path(cube_path))
    cube_geom.CreateSizeAttr(1.0)
    set_xform(cube_geom.GetPrim(), scale=size)
    UsdPhysics.CollisionAPI.Apply(cube_geom.GetPrim())
    UsdGeom.Imageable(cube_geom.GetPrim()).MakeInvisible()


def create_fallback_helicopter(stage, root_prim_path, spec):
    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_prim_path)).GetPrim()

    fuselage_len = spec.length_m * 0.58
    fuselage_width = spec.rotor_diameter_m * 0.11
    fuselage_height = spec.height_m * 0.32
    tail_len = spec.length_m * 0.32
    rotor_z = spec.height_m * 0.48

    fuselage = UsdGeom.Cube.Define(stage, Sdf.Path(root_prim_path + "/Fuselage")).GetPrim()
    UsdGeom.Cube(fuselage).CreateSizeAttr(1.0)
    set_xform(fuselage, scale=(fuselage_len, fuselage_width, fuselage_height))
    set_display_color(fuselage, spec.color)

    nose = UsdGeom.Cube.Define(stage, Sdf.Path(root_prim_path + "/SensorNose")).GetPrim()
    UsdGeom.Cube(nose).CreateSizeAttr(1.0)
    set_xform(nose, translate=(fuselage_len * 0.54, 0.0, -fuselage_height * 0.04), scale=(spec.length_m * 0.10, fuselage_width * 0.72, fuselage_height * 0.64))
    set_display_color(nose, spec.accent_color)

    mast = UsdGeom.Cylinder.Define(stage, Sdf.Path(root_prim_path + "/Mast")).GetPrim()
    UsdGeom.Cylinder(mast).CreateRadiusAttr(spec.rotor_diameter_m * 0.018)
    UsdGeom.Cylinder(mast).CreateHeightAttr(spec.height_m * 0.22)
    set_xform(mast, translate=(0.0, 0.0, rotor_z * 0.78))
    set_display_color(mast, spec.accent_color)

    main_rotor = UsdGeom.Cube.Define(stage, Sdf.Path(root_prim_path + "/MainRotor")).GetPrim()
    UsdGeom.Cube(main_rotor).CreateSizeAttr(1.0)
    set_xform(main_rotor, translate=(0.0, 0.0, rotor_z), scale=(spec.rotor_diameter_m, spec.rotor_diameter_m * 0.025, spec.height_m * 0.015))
    set_display_color(main_rotor, spec.accent_color)

    main_rotor_cross = UsdGeom.Cube.Define(stage, Sdf.Path(root_prim_path + "/MainRotorCross")).GetPrim()
    UsdGeom.Cube(main_rotor_cross).CreateSizeAttr(1.0)
    set_xform(main_rotor_cross, translate=(0.0, 0.0, rotor_z + 0.03), rotate_xyz=(0.0, 0.0, 90.0), scale=(spec.rotor_diameter_m, spec.rotor_diameter_m * 0.025, spec.height_m * 0.015))
    set_display_color(main_rotor_cross, spec.accent_color)

    tail_boom = UsdGeom.Cube.Define(stage, Sdf.Path(root_prim_path + "/TailBoom")).GetPrim()
    UsdGeom.Cube(tail_boom).CreateSizeAttr(1.0)
    set_xform(tail_boom, translate=(-(fuselage_len * 0.45 + tail_len * 0.50), 0.0, fuselage_height * 0.08), scale=(tail_len, fuselage_width * 0.28, fuselage_height * 0.28))
    set_display_color(tail_boom, spec.color)

    tail_rotor = UsdGeom.Cube.Define(stage, Sdf.Path(root_prim_path + "/TailRotor")).GetPrim()
    UsdGeom.Cube(tail_rotor).CreateSizeAttr(1.0)
    set_xform(tail_rotor, translate=(-(fuselage_len * 0.45 + tail_len), 0.0, fuselage_height * 0.22), rotate_xyz=(0.0, 90.0, 0.0), scale=(spec.height_m * 0.48, spec.height_m * 0.04, spec.height_m * 0.025))
    set_display_color(tail_rotor, spec.accent_color)

    cannon = UsdGeom.Cylinder.Define(stage, Sdf.Path(root_prim_path + "/ChinCannon")).GetPrim()
    UsdGeom.Cylinder(cannon).CreateRadiusAttr(spec.height_m * 0.025)
    UsdGeom.Cylinder(cannon).CreateHeightAttr(spec.length_m * 0.10)
    set_xform(cannon, translate=(fuselage_len * 0.60, 0.0, -fuselage_height * 0.55), rotate_xyz=(0.0, 90.0, 0.0))
    set_display_color(cannon, spec.accent_color)

    for side, y_sign in (("Left", 1.0), ("Right", -1.0)):
        pylon = UsdGeom.Cube.Define(stage, Sdf.Path(f"{root_prim_path}/{side}Pylon")).GetPrim()
        UsdGeom.Cube(pylon).CreateSizeAttr(1.0)
        set_xform(pylon, translate=(fuselage_len * 0.08, y_sign * fuselage_width * 0.78, -fuselage_height * 0.10), scale=(spec.length_m * 0.20, spec.rotor_diameter_m * 0.025, spec.height_m * 0.045))
        set_display_color(pylon, spec.accent_color)

        store = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{root_prim_path}/{side}Store")).GetPrim()
        UsdGeom.Cylinder(store).CreateRadiusAttr(spec.height_m * 0.055)
        UsdGeom.Cylinder(store).CreateHeightAttr(spec.length_m * 0.18)
        set_xform(store, translate=(fuselage_len * 0.08, y_sign * fuselage_width * 0.95, -fuselage_height * 0.23), rotate_xyz=(0.0, 90.0, 0.0))
        set_display_color(store, (0.18, 0.18, 0.16))

    return root


def spawn_helicopter(stage, prim_path, spec, start_pos):
    asset = resolve_heli_asset(spec)
    root = UsdGeom.Xform.Define(stage, Sdf.Path(prim_path)).GetPrim()
    set_xform(root, translate=start_pos)

    loaded_asset = False
    if asset:
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage

            add_reference_to_stage(usd_path=asset, prim_path=prim_path)
            loaded_asset = True
            print(f"[INFO] Loaded {spec.name} asset: {asset}")
        except Exception as exc:
            print(f"[WARN] Failed to load {spec.name} asset, using fallback model: {exc}")

    if not loaded_asset:
        stage.RemovePrim(Sdf.Path(prim_path))
        root = create_fallback_helicopter(stage, prim_path, spec)
        set_xform(root, translate=start_pos)
        print(f"[INFO] Using procedural fallback {spec.name}.")

    cleanup_physics(stage, prim_path)
    add_collision_box(stage, prim_path, (spec.length_m * 0.70, spec.rotor_diameter_m * 0.16, spec.height_m * 0.42))
    return stage.GetPrimAtPath(prim_path)


def target_at_time(t, radius=22.0, omega=0.25, z_base=30.0, phase=0.0):
    z = z_base + 2.0 * math.sin(t * 0.45 + phase)
    return np.array([radius * math.cos(t * omega + phase), radius * math.sin(t * omega + phase), z], dtype=float)


def target_velocity_at_time(t, radius=22.0, omega=0.25, z_base=30.0, phase=0.0):
    eps = 1.0 / 60.0
    return (
        target_at_time(t + eps, radius=radius, omega=omega, z_base=z_base, phase=phase)
        - target_at_time(max(0.0, t - eps), radius=radius, omega=omega, z_base=z_base, phase=phase)
    ) / (eps * 2.0)


def create_controller(heli_root, spec, motion_mode):
    if motion_mode == "dynamic":
        controller = PhysXHelicopterVehicle(
            heli_root,
            mass=spec.gross_mass_kg,
            max_force=spec.max_force_n,
            max_torque=spec.max_torque_nm,
        )
        print(f"[INFO] {spec.name} motion mode: dynamic helicopter controller")
    else:
        controller = PhysXKinematicTracker(
            heli_root,
            mass=spec.gross_mass_kg,
            max_speed=spec.cruise_speed_mps,
        )
        print(f"[INFO] {spec.name} motion mode: kinematic tracker")
    return controller


def main():
    stage = omni.usd.get_context().get_stage()
    if not stage:
        stage = Usd.Stage.CreateInMemory()
        omni.usd.get_context().set_stage(stage)

    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    light.CreateIntensityAttr(1200.0)

    physics_scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    friendly_root = spawn_helicopter(stage, FRIENDLY_HELI_PATH, FRIENDLY_SPEC, FRIENDLY_START_POS)
    enemy_root = spawn_helicopter(stage, ENEMY_HELI_PATH, ENEMY_SPEC, ENEMY_START_POS)

    motion_mode = os.environ.get("QL_HELI_MOTION_MODE", "kinematic").strip().lower()
    friendly_controller = create_controller(friendly_root, FRIENDLY_SPEC, motion_mode)
    enemy_controller = create_controller(enemy_root, ENEMY_SPEC, motion_mode)

    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=[48.0, -54.0, 42.0], target=[0.0, 0.0, 30.0])

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    t = 0.0
    frame_count = 0
    start = time.time()
    print("[TEST] WZ-21 and AH-64 Apache follow opposed circular paths. Press Ctrl+C or close the window to stop.")
    print(
        f"[DATA] {FRIENDLY_SPEC.name}: gross mass {FRIENDLY_SPEC.gross_mass_kg:.0f} kg, "
        f"cruise {FRIENDLY_SPEC.cruise_speed_kmh:.0f} km/h, ceiling {FRIENDLY_SPEC.service_ceiling_m:.0f} m"
    )
    print(
        f"[DATA] {ENEMY_SPEC.name}: gross mass {ENEMY_SPEC.gross_mass_kg:.0f} kg, "
        f"cruise {ENEMY_SPEC.cruise_speed_kmh:.0f} km/h, ceiling {ENEMY_SPEC.service_ceiling_m:.0f} m"
    )

    try:
        while simulation_app.is_running():
            friendly_target_pos = target_at_time(t, radius=24.0, omega=0.22, z_base=32.0, phase=0.0)
            friendly_target_vel = target_velocity_at_time(t, radius=24.0, omega=0.22, z_base=32.0, phase=0.0)
            enemy_target_pos = target_at_time(t, radius=20.0, omega=-0.26, z_base=28.0, phase=math.pi)
            enemy_target_vel = target_velocity_at_time(t, radius=20.0, omega=-0.26, z_base=28.0, phase=math.pi)

            friendly_state = friendly_controller.step(DT, friendly_target_pos, friendly_target_vel)
            enemy_state = enemy_controller.step(DT, enemy_target_pos, enemy_target_vel)

            if frame_count % 60 == 0:
                friendly_pos, _ = _rigid_body_pose(friendly_root)
                enemy_pos, _ = _rigid_body_pose(enemy_root)
                print(
                    f"Time: {t:5.1f}s | "
                    f"WZ-21 Pos: {friendly_pos.round(2)} Target: {friendly_target_pos.round(2)} "
                    f"Speed: {np.linalg.norm(friendly_state.velocity):.2f} m/s | "
                    f"AH-64 Pos: {enemy_pos.round(2)} Target: {enemy_target_pos.round(2)} "
                    f"Speed: {np.linalg.norm(enemy_state.velocity):.2f} m/s"
                )

            simulation_app.update()
            t += DT
            frame_count += 1
            if time.time() - start > RUN_SECONDS:
                break
    finally:
        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
