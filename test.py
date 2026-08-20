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
from pathlib import Path

import numpy as np
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from scenes.physx_motion_models import PhysXAerialVehicle, PhysXKinematicTracker, _rigid_body_pose


UAV_PATH = "/World/SimpleUAV"
START_POS = np.array([12.0, 0.0, 12.0], dtype=float)
RUN_SECONDS = 30.0
DT = 1.0 / 60.0


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


def resolve_drone_asset():
    env_path = os.environ.get("QL_DRONE_USD")
    if env_path:
        expanded = Path(env_path).expanduser()
        return str(expanded) if expanded.exists() else env_path

    try:
        from isaacsim.storage.native import get_assets_root_path

        assets_root = get_assets_root_path()
        if assets_root:
            return assets_root + "/Isaac/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd"
    except Exception:
        pass

    local_asset = Path(__file__).resolve().parent / "asset" / "ARL-Robot-1" / "arl_robot_1.usd"
    if local_asset.exists():
        return str(local_asset)
    return None


def add_collision_box(stage, root_prim_path, size):
    cube_path = root_prim_path + "/CollisionBox"
    cube_geom = UsdGeom.Cube.Define(stage, Sdf.Path(cube_path))
    cube_geom.CreateSizeAttr(1.0)
    set_xform(cube_geom.GetPrim(), scale=size)
    UsdPhysics.CollisionAPI.Apply(cube_geom.GetPrim())
    UsdGeom.Imageable(cube_geom.GetPrim()).MakeInvisible()


def create_fallback_drone(stage, root_prim_path):
    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_prim_path)).GetPrim()

    body = UsdGeom.Cube.Define(stage, Sdf.Path(root_prim_path + "/Body")).GetPrim()
    set_xform(body, scale=(1.6, 0.9, 0.35))
    set_display_color(body, (0.12, 0.20, 0.28))

    for name, angle_deg in (("ArmA", 0.0), ("ArmB", 90.0)):
        arm = UsdGeom.Cube.Define(stage, Sdf.Path(root_prim_path + "/" + name)).GetPrim()
        set_xform(arm, rotate_xyz=(0.0, 0.0, angle_deg), scale=(3.6, 0.16, 0.12))
        set_display_color(arm, (0.05, 0.05, 0.05))

    for i, (x, y) in enumerate(((1.8, 0.0), (-1.8, 0.0), (0.0, 1.8), (0.0, -1.8))):
        rotor = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{root_prim_path}/Rotor_{i}")).GetPrim()
        UsdGeom.Cylinder(rotor).CreateRadiusAttr(0.55)
        UsdGeom.Cylinder(rotor).CreateHeightAttr(0.05)
        set_xform(rotor, translate=(x, y, 0.18), rotate_xyz=(90.0, 0.0, 0.0))
        set_display_color(rotor, (0.02, 0.02, 0.02))

    return root


def spawn_drone(stage, prim_path):
    asset = resolve_drone_asset()
    root = UsdGeom.Xform.Define(stage, Sdf.Path(prim_path)).GetPrim()
    set_xform(root, translate=START_POS)

    loaded_asset = False
    if asset:
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage

            add_reference_to_stage(usd_path=asset, prim_path=prim_path)
            loaded_asset = True
            print(f"[INFO] Loaded drone asset: {asset}")
        except Exception as exc:
            print(f"[WARN] Failed to load drone asset, using fallback model: {exc}")

    if not loaded_asset:
        stage.RemovePrim(Sdf.Path(prim_path))
        root = create_fallback_drone(stage, prim_path)
        set_xform(root, translate=START_POS)
        print("[INFO] Using procedural fallback drone.")

    cleanup_physics(stage, prim_path)
    add_collision_box(stage, prim_path, (3.8, 3.8, 0.8))
    return stage.GetPrimAtPath(prim_path)


def target_at_time(t):
    radius = 12.0
    omega = 0.45
    z = 12.0 + 2.0 * math.sin(t * 0.8)
    return np.array([radius * math.cos(t * omega), radius * math.sin(t * omega), z], dtype=float)


def target_velocity_at_time(t):
    eps = 1.0 / 60.0
    return (target_at_time(t + eps) - target_at_time(max(0.0, t - eps))) / (eps * 2.0)


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

    uav_root = spawn_drone(stage, UAV_PATH)

    motion_mode = os.environ.get("QL_UAV_MOTION_MODE", "kinematic").strip().lower()
    if motion_mode == "dynamic":
        controller = PhysXAerialVehicle(uav_root, mass=30.0, max_force=1800.0, max_torque=250.0)
        print("[INFO] Motion mode: dynamic PhysX force controller")
    else:
        controller = PhysXKinematicTracker(uav_root, mass=30.0, max_speed=18.0)
        print("[INFO] Motion mode: kinematic trajectory tracker")

    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=[30.0, -32.0, 22.0], target=[0.0, 0.0, 12.0])

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    t = 0.0
    frame_count = 0
    start = time.time()
    print("[TEST] UAV follows a circular path. Press Ctrl+C or close the window to stop.")

    try:
        while simulation_app.is_running():
            target_pos = target_at_time(t)
            target_vel = target_velocity_at_time(t)
            state = controller.step(DT, target_pos, target_vel)

            if frame_count % 60 == 0:
                pos, _ = _rigid_body_pose(uav_root)
                print(
                    f"Time: {t:5.1f}s | Pos: {pos.round(2)} | "
                    f"Target: {target_pos.round(2)} | Speed: {np.linalg.norm(state.velocity):.2f} m/s"
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
