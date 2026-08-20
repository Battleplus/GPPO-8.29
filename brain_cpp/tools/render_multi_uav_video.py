"""Render synchronized multi-UAV transit and area-search trajectories in Isaac."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COLORS = (
    (0.05, 0.90, 1.00),
    (1.00, 0.12, 0.78),
    (0.12, 1.00, 0.30),
    (1.00, 0.86, 0.05),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def load_grouped(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted = dict(row)
            for field in ("x", "y", "z", "vx", "vy", "vz", "time_s"):
                if field in converted:
                    converted[field] = float(converted[field])
            if "frame" in converted:
                converted["frame"] = int(converted["frame"])
            grouped[str(row["platform"])].append(converted)
    if len(grouped) < 2:
        raise ValueError("multi-UAV trajectory must contain at least two platforms")
    return dict(grouped)


def hide_scene_aids(stage, selected: set[str]) -> None:
    from pxr import UsdGeom

    platforms = stage.GetPrimAtPath("/World/AirCombat/Platforms")
    if platforms:
        for child in platforms.GetChildren():
            if child.GetName() not in selected:
                UsdGeom.Imageable(child).MakeInvisible()
                continue
            for visual in child.GetChildren():
                name = visual.GetName()
                if name in {
                    "VerticalLocator", "HighVisibilityBeacon", "FactionPanel",
                    "LocatorRing", "RoleMarkerRecon", "RoleMarkerReconDot",
                    "SensorMastHighlight", "SensorHeadHighlight",
                } or name.startswith("SensorCones_"):
                    UsdGeom.Imageable(visual).MakeInvisible()
    for path in (
        "/World/AirCombat/SensorRings",
        "/World/AirCombat/PpoExecutedTrajectory",
        "/World/AirCombat/PpoReplayTrail",
        "/World/AirCombat/PpoReplayAircraftMarker",
        "/World/AirCombat/BrainCppPatrol",
        "/World/AirCombat/PpoLocalObstacle",
    ):
        prim = stage.GetPrimAtPath(path)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()


def create_curve(stage, path: str, color, width: float, points):
    from pxr import Gf, Sdf, UsdGeom

    curve = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curve.CreateTypeAttr("linear")
    values = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points]
    if len(values) == 1:
        values.append(values[0])
    curve.CreateCurveVertexCountsAttr([len(values)])
    curve.CreatePointsAttr(values)
    curve.CreateWidthsAttr([width] * len(values))
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return curve


def create_marker(stage, path: str, color):
    from pxr import Gf, Sdf, UsdGeom

    marker = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    marker.CreateRadiusAttr(7.0)
    marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return UsdGeom.Xformable(marker.GetPrim()).AddTranslateOp()


def set_platform_pose(prim, row: dict) -> None:
    from pxr import Gf

    yaw = math.degrees(math.atan2(float(row["vy"]), float(row["vx"])))
    horizontal = math.hypot(float(row["vx"]), float(row["vy"]))
    pitch = -math.degrees(math.atan2(float(row["vz"]), max(horizontal, 1e-6)))
    prim.GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(float(row["x"]), float(row["y"]), float(row["z"]))
    )
    prim.GetAttribute("xformOp:rotateXYZ").Set(
        Gf.Vec3f(0.0, max(-18.0, min(18.0, pitch)), yaw)
    )


def main() -> int:
    args = parse_args()
    trajectory_path = Path(args.trajectory).resolve()
    plan_path = Path(args.plan).resolve()
    frames_dir = Path(args.frames_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories = load_grouped(trajectory_path)
    plans = load_grouped(plan_path)
    platforms = list(trajectories)
    frame_count = min(len(rows) for rows in trajectories.values())

    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": bool(args.headless),
        "renderer": "HydraStorm",
        "width": args.width,
        "height": args.height,
    })
    try:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf
        from sensors.io import save_rgb

        context = omni.usd.get_context()
        context.open_stage(str(Path(args.stage).resolve()))
        for _ in range(30):
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac failed to open the source stage")
        hide_scene_aids(stage, set(platforms))

        prims = {}
        trails = {}
        markers = {}
        all_points = []
        for index, platform in enumerate(platforms):
            color = COLORS[index % len(COLORS)]
            prim = stage.GetPrimAtPath(f"/World/AirCombat/Platforms/{platform}")
            if not prim:
                raise RuntimeError(f"platform prim not found: {platform}")
            prims[platform] = prim
            planned = [
                (row["x"], row["y"], float(row["z"]) + 2.0)
                for row in plans[platform]
            ]
            all_points.extend(planned)
            create_curve(
                stage,
                f"/World/AirCombat/MultiUav/Planned_{index}",
                tuple(component * 0.55 for component in color),
                2.0,
                planned,
            )
            trails[platform] = create_curve(
                stage,
                f"/World/AirCombat/MultiUav/Trail_{index}",
                color,
                4.5,
                [planned[0], planned[0]],
            )
            markers[platform] = create_marker(
                stage,
                f"/World/AirCombat/MultiUav/Marker_{index}",
                color,
            )

        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        center_x = (min(xs) + max(xs)) * 0.5
        center_y = (min(ys) + max(ys)) * 0.5
        span = max(max(xs) - min(xs), max(ys) - min(ys), 500.0)
        camera = rep.create.camera(
            position=(center_x + span * 0.60, center_y - span * 0.85, 1900.0),
            look_at=(center_x, center_y, 110.0),
        )
        product = rep.create.render_product(
            camera, resolution=(args.width, args.height)
        )
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        for _ in range(3):
            rep.orchestrator.step()
            app.update()

        rendered = []
        for frame in range(frame_count):
            for platform in platforms:
                row = trajectories[platform][frame]
                set_platform_pose(prims[platform], row)
                markers[platform].Set(Gf.Vec3d(
                    float(row["x"]), float(row["y"]), float(row["z"]) + 12.0
                ))
                points = [
                    Gf.Vec3f(float(item["x"]), float(item["y"]), float(item["z"]) + 3.0)
                    for item in trajectories[platform][: frame + 1]
                ]
                if len(points) == 1:
                    points.append(points[0])
                trails[platform].GetCurveVertexCountsAttr().Set([len(points)])
                trails[platform].GetPointsAttr().Set(points)
                trails[platform].GetWidthsAttr().Set([4.5] * len(points))
            rep.orchestrator.step()
            rgb = annotator.get_data()
            if rgb is None or rgb.size == 0:
                raise RuntimeError(f"empty RGB frame {frame}")
            output = frames_dir / f"frame_{frame:04d}.png"
            save_rgb(rgb, output)
            rendered.append(str(output))
            if frame == 0 or (frame + 1) % 20 == 0:
                phase = trajectories[platforms[0]][frame]["phase"]
                print(
                    f"MULTI_RENDER frame={frame + 1}/{frame_count} "
                    f"phase={phase} rgb_mean={float(rgb[:, :, :3].mean()):.2f}",
                    flush=True,
                )

        annotator.detach()
        product.destroy()
        manifest = {
            "stage": str(Path(args.stage).resolve()),
            "trajectory": str(trajectory_path),
            "plan": str(plan_path),
            "platforms": platforms,
            "rendered_frames": len(rendered),
            "width": args.width,
            "height": args.height,
            "first_frame": rendered[0],
            "last_frame": rendered[-1],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
