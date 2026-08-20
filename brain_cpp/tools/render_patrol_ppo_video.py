"""Render a task-area search_planner + PPO trajectory from an Isaac USD stage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--platform", default="Blue_CH4_Recon")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-output-frames", type=int, default=180)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def load_trajectory(path: Path) -> list[dict[str, float | int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"trajectory contains fewer than two samples: {path}")
    return [
        {
            "frame": int(row["frame"]),
            "time_s": float(row["time_s"]),
            "waypoint_index": int(row["waypoint_index"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "vx": float(row["vx"]),
            "vy": float(row["vy"]),
            "vz": float(row["vz"]),
            "clearance": float(row["clearance"]),
        }
        for row in rows
    ]


def sample_indices(sample_count: int, output_count: int) -> list[int]:
    output_count = max(2, min(sample_count, output_count))
    if output_count == sample_count:
        return list(range(sample_count))
    return sorted(
        {
            round(index * (sample_count - 1) / (output_count - 1))
            for index in range(output_count)
        }
    )


def hide_unrelated_scene(stage, platform_name: str) -> None:
    from pxr import UsdGeom

    platforms = stage.GetPrimAtPath("/World/AirCombat/Platforms")
    if platforms:
        for child in platforms.GetChildren():
            if child.GetName() != platform_name:
                UsdGeom.Imageable(child).MakeInvisible()
    platform = stage.GetPrimAtPath(
        f"/World/AirCombat/Platforms/{platform_name}"
    )
    visual_aids = {
        "VerticalLocator",
        "HighVisibilityBeacon",
        "FactionPanel",
        "LocatorRing",
        "RoleMarkerRecon",
        "RoleMarkerReconDot",
        "SensorMastHighlight",
        "SensorHeadHighlight",
    }
    if platform:
        for child in platform.GetChildren():
            if child.GetName() in visual_aids or child.GetName().startswith(
                "SensorCones_"
            ):
                UsdGeom.Imageable(child).MakeInvisible()
    for path in (
        "/World/AirCombat/SensorRings",
        "/World/AirCombat/PpoExecutedTrajectory",
        f"/World/AirCombat/Platforms/{platform_name}/SensorCones",
    ):
        prim = stage.GetPrimAtPath(path)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()


def create_replay_trail(stage):
    from pxr import Gf, Sdf, UsdGeom

    curve = UsdGeom.BasisCurves.Define(
        stage, Sdf.Path("/World/AirCombat/PpoReplayTrail")
    )
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0)]
    )
    curve.CreateWidthsAttr([3.2, 3.2])
    curve.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.95, 0.3)])
    return curve


def create_aircraft_marker(stage):
    from pxr import Gf, Sdf, UsdGeom

    marker = UsdGeom.Sphere.Define(
        stage, Sdf.Path("/World/AirCombat/PpoReplayAircraftMarker")
    )
    marker.CreateRadiusAttr(4.0)
    marker.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.05, 0.85)])
    translate = UsdGeom.Xformable(marker.GetPrim()).AddTranslateOp()
    return translate


def set_platform_pose(prim, row: dict[str, float | int]) -> None:
    from pxr import Gf

    yaw = math.degrees(math.atan2(float(row["vy"]), float(row["vx"])))
    horizontal_speed = math.hypot(float(row["vx"]), float(row["vy"]))
    pitch = -math.degrees(
        math.atan2(float(row["vz"]), max(horizontal_speed, 1e-6))
    )
    prim.GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(float(row["x"]), float(row["y"]), float(row["z"]))
    )
    prim.GetAttribute("xformOp:rotateXYZ").Set(
        Gf.Vec3f(0.0, max(-18.0, min(18.0, pitch)), yaw)
    )


def main() -> int:
    args = parse_args()
    stage_path = Path(args.stage).resolve()
    trajectory_path = Path(args.trajectory).resolve()
    frames_dir = Path(args.frames_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = load_trajectory(trajectory_path)
    indices = sample_indices(len(trajectory), args.max_output_frames)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "renderer": "HydraStorm",
            "width": args.width,
            "height": args.height,
        }
    )
    try:
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf
        from sensors.io import save_rgb

        context = omni.usd.get_context()
        context.open_stage(str(stage_path))
        for _ in range(30):
            simulation_app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"Isaac failed to open USD stage: {stage_path}")
        print("VIDEO_RENDER stage_opened", flush=True)

        platform_path = f"/World/AirCombat/Platforms/{args.platform}"
        platform_prim = stage.GetPrimAtPath(platform_path)
        if not platform_prim:
            raise RuntimeError(f"platform prim not found: {platform_path}")
        hide_unrelated_scene(stage, args.platform)
        trail = create_replay_trail(stage)
        aircraft_marker = create_aircraft_marker(stage)
        print("VIDEO_RENDER scene_prepared", flush=True)

        xs = [float(row["x"]) for row in trajectory]
        ys = [float(row["y"]) for row in trajectory]
        zs = [float(row["z"]) for row in trajectory]
        center_x = (min(xs) + max(xs)) * 0.5
        center_y = (min(ys) + max(ys)) * 0.5
        center_z = (min(zs) + max(zs)) * 0.5
        span = max(max(xs) - min(xs), max(ys) - min(ys), 120.0)
        eye = [
            center_x + span * 0.95,
            center_y - span * 1.25,
            center_z + span * 1.15,
        ]
        target = [center_x, center_y, center_z - 4.0]
        camera = rep.create.camera(position=eye, look_at=target)
        print("VIDEO_RENDER camera_created", flush=True)
        render_product = rep.create.render_product(
            camera,
            resolution=(args.width, args.height),
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])
        print("VIDEO_RENDER render_product_attached", flush=True)
        for _ in range(3):
            rep.orchestrator.step()
            simulation_app.update()

        rendered_files: list[str] = []
        for output_index, source_index in enumerate(indices):
            row = trajectory[source_index]
            set_platform_pose(platform_prim, row)
            aircraft_marker.Set(
                Gf.Vec3d(
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]) + 7.0,
                )
            )
            replay_points = [
                Gf.Vec3f(
                    float(item["x"]),
                    float(item["y"]),
                    float(item["z"]) + 1.2,
                )
                for item in trajectory[: source_index + 1]
            ]
            if len(replay_points) == 1:
                replay_points.append(replay_points[0])
            trail.GetCurveVertexCountsAttr().Set([len(replay_points)])
            trail.GetPointsAttr().Set(replay_points)
            trail.GetWidthsAttr().Set([3.2] * len(replay_points))
            rep.orchestrator.step()
            rgb = rgb_annotator.get_data()
            if rgb is None or rgb.size == 0:
                raise RuntimeError(f"Isaac RenderProduct returned no RGB data at frame {output_index}")
            output_path = frames_dir / f"frame_{output_index:04d}.png"
            save_rgb(rgb, output_path)
            rendered_files.append(str(output_path))
            if output_index == 0 or (output_index + 1) % 20 == 0:
                print(
                    f"VIDEO_RENDER frame={output_index + 1}/{len(indices)} "
                    f"source_frame={row['frame']} waypoint={row['waypoint_index']} "
                    f"rgb_mean={float(rgb[:, :, :3].mean()):.2f}",
                    flush=True,
                )

        rgb_annotator.detach()
        render_product.destroy()

        manifest = {
            "stage": str(stage_path),
            "trajectory": str(trajectory_path),
            "platform": args.platform,
            "source_samples": len(trajectory),
            "rendered_frames": len(rendered_files),
            "width": args.width,
            "height": args.height,
            "camera_eye": eye,
            "camera_target": target,
            "first_frame": rendered_files[0],
            "last_frame": rendered_files[-1],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
