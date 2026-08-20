"""Run search_planner waypoints with PPO local avoidance inside the task area."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain.integration import IsaacAirCombatEnvironment
from drl_env.core import DETECT_RANGE, MAX_ACCEL, MAX_SPEED, SENSE_RANGE
from drl_env.drl_motion_model import DrlAvoidanceModel
from drl_env.numpy_policy import NumpyPolicy
from drl_env.run_global_path_ppo_local import DRONE_RADIUS, make_local_obstacle, resolve_policy_path


def load_brain_cpp_path(path: str | Path) -> tuple[dict[str, str], list[np.ndarray]]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"brain_cpp global path contains fewer than two points: {source}")
    metadata = {
        "platform": rows[0]["platform"],
        "sensor": rows[0]["sensor"],
        "pattern": rows[0]["pattern"],
        "cell": rows[0]["cell"],
        "source": str(source),
    }
    waypoints = [
        np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
        for row in rows
    ]
    return metadata, waypoints


def write_trajectory(path: Path, samples: list[dict[str, float | int]]) -> None:
    fields = ("frame", "time_s", "waypoint_index", "x", "y", "z", "vx", "vy", "vz", "clearance")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)


def create_isaac_overlays(stage, planner_points, physical_obstacle) -> None:
    from pxr import Gf, Sdf, UsdGeom
    from search_planner.planner import Waypoint
    from search_planner.visualize import export_waypoints_usd

    visual_points = [
        Waypoint(x=float(p[0]), y=float(p[1]), z=float(p[2]), terrain_z=0.0, yaw_deg=0.0)
        for p in planner_points
    ]
    export_waypoints_usd(
        stage,
        visual_points,
        base_path="/World/AirCombat/BrainCppPatrol",
        path_color=(0.05, 0.55, 1.0),
        label="brain_cpp",
    )
    obstacle = UsdGeom.Cylinder.Define(
        stage, Sdf.Path("/World/AirCombat/PpoLocalObstacle")
    )
    obstacle.CreateRadiusAttr(float(physical_obstacle.radius_units))
    obstacle.CreateHeightAttr(float(physical_obstacle.height_units))
    center_z = float(physical_obstacle.position[2] + physical_obstacle.height_units * 0.5)
    UsdGeom.Xformable(obstacle.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(
            float(physical_obstacle.position[0]),
            float(physical_obstacle.position[1]),
            center_z,
        )
    )
    UsdGeom.Gprim(obstacle.GetPrim()).CreateDisplayColorAttr(
        [Gf.Vec3f(0.95, 0.12, 0.04)]
    )
    UsdGeom.Gprim(obstacle.GetPrim()).CreateDisplayOpacityAttr([0.45])


def create_trajectory_overlay(stage, samples: list[dict[str, float | int]]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    points = [
        Gf.Vec3f(float(row["x"]), float(row["y"]), float(row["z"]) + 1.0)
        for row in samples
    ]
    curve = UsdGeom.BasisCurves.Define(
        stage, Sdf.Path("/World/AirCombat/PpoExecutedTrajectory")
    )
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(points)
    curve.CreateWidthsAttr([2.5] * len(points))
    curve.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.9, 0.3)])


def run(args: argparse.Namespace) -> dict[str, Any]:
    metadata, waypoints = load_brain_cpp_path(args.global_plan)
    policy_path = resolve_policy_path(args.policy)
    policy = NumpyPolicy(policy_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_config = {
        "simulation": {"time_scale": 3.0, "aircraft_model": "kinematic"},
        "obstacles": {
            "enabled": True,
            "mountains": False,
            "forests": False,
            "rock_fields": False,
            "max_contacts_per_platform": 12,
        },
        "weather": {"clouds": False, "local_zones": False, "rain": False},
        "visual": {"show_sensor_rings": True, "show_sensor_cones": True},
    }
    environment = IsaacAirCombatEnvironment(
        scene_config=scene_config,
        headless=args.headless,
    )
    samples: list[dict[str, float | int]] = []
    summary: dict[str, Any] = {}
    try:
        scene = environment.initialize()
        platform = next(
            (item for item in scene.platforms if item.entity_id == metadata["platform"]),
            None,
        )
        if platform is None:
            raise RuntimeError(f"Isaac platform not found: {metadata['platform']}")

        planned_obstacle = make_local_obstacle(waypoints)
        from scenes.air_combat_scene import (
            EnvironmentObstacle,
            _set_root_pose,
            compute_obstacle_contacts,
        )

        # The PPO was trained with full-height pillars represented by their
        # center. Isaac sensing uses the physical base; the contact adapter
        # converts base + half-height back to the trained representation.
        physical_height = max(20.0, float(planned_obstacle["position"][2]) * 2.0)
        physical_obstacle = EnvironmentObstacle(
            obstacle_id="PpoLocalObstacle",
            name="PPO local avoidance pillar",
            category="mountain",
            position=np.array(
                [planned_obstacle["position"][0], planned_obstacle["position"][1], 0.0],
                dtype=float,
            ),
            radius_units=float(planned_obstacle["radius_units"]),
            height_units=physical_height,
            priority=100,
            blocks_los=True,
        )
        scene.obstacles.append(physical_obstacle)
        create_isaac_overlays(environment.stage, waypoints, physical_obstacle)

        model = DrlAvoidanceModel(
            initial_position=waypoints[0],
            max_speed=MAX_SPEED,
            max_accel=MAX_ACCEL,
            policy=policy,
            waypoints=waypoints,
            obstacles=[],
            sense_range=SENSE_RANGE,
            detect_range=DETECT_RANGE,
            wp_reach_dist=8.0,
        )
        platform.motion_model = model
        _set_root_pose(platform.root_prim, model.state.position, model.state)

        contact_memory: dict[str, dict[str, Any]] = {}
        scene.obstacle_contacts = compute_obstacle_contacts(scene)
        first_contact_frame: int | None = None
        sar_contact_count = 0
        min_clearance = math.inf
        dt = 1.0 / 30.0

        for frame in range(args.max_frames):
            contacts = [
                contact
                for contact in scene.obstacle_contacts_for(platform.entity_id)
                if contact["obstacle_id"] == physical_obstacle.obstacle_id
                and contact["channel"] == "sar"
            ]
            for contact in contacts:
                sar_contact_count += 1
                if first_contact_frame is None:
                    first_contact_frame = frame
                contact_position = np.asarray(contact["position"], dtype=float)
                contact_position[2] += (
                    float(contact["height_m"]) / scene.meters_per_unit * 0.5
                )
                contact_memory[contact["obstacle_id"]] = {
                    "position": contact_position,
                    "radius_units": float(contact["radius_km"]) * 1000.0 / scene.meters_per_unit,
                }
            model.obstacles = list(contact_memory.values())

            environment.step(dt)
            position = model.state.position
            velocity = model.state.velocity
            clearance = (
                float(np.linalg.norm(position[:2] - physical_obstacle.position[:2]))
                - physical_obstacle.radius_units
                - DRONE_RADIUS
            )
            min_clearance = min(min_clearance, clearance)
            samples.append(
                {
                    "frame": frame + 1,
                    "time_s": (frame + 1) * 0.1,
                    "waypoint_index": model.wp_idx,
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                    "vx": float(velocity[0]),
                    "vy": float(velocity[1]),
                    "vz": float(velocity[2]),
                    "clearance": float(clearance),
                }
            )
            if model.wp_idx == len(waypoints) - 1 and float(
                np.linalg.norm(position - waypoints[-1])
            ) < 8.0:
                break

        completed = bool(
            model.wp_idx == len(waypoints) - 1
            and float(np.linalg.norm(model.state.position - waypoints[-1])) < 8.0
        )
        success = bool(
            completed
            and first_contact_frame is not None
            and model._stats["collisions"] == 0
            and min_clearance > 0.0
        )
        create_trajectory_overlay(environment.stage, samples)
        usd_path = output_dir / "brain_cpp_path_ppo_isaac.usda"
        environment.stage.GetRootLayer().Export(str(usd_path))
        trajectory_path = output_dir / "isaac_trajectory.csv"
        write_trajectory(trajectory_path, samples)
        summary = {
            "success": success,
            "runtime": "IsaacAirCombatEnvironment",
            "headless": bool(args.headless),
            "platform": metadata["platform"],
            "sensor": metadata["sensor"],
            "global_pattern": metadata["pattern"],
            "cell": metadata["cell"],
            "global_waypoint_count": len(waypoints),
            "frames": len(samples),
            "completed": completed,
            "waypoint_index": model.wp_idx,
            "sar_contact_count": sar_contact_count,
            "first_sar_contact_frame": first_contact_frame,
            "ppo_obstacle_memory_count": len(contact_memory),
            "collision_samples": model._stats["collisions"],
            "min_clearance_units": round(float(min_clearance), 6),
            "min_clearance_m": round(float(min_clearance) * scene.meters_per_unit, 3),
            "policy_path": str(policy_path),
            "global_plan_source": metadata["source"],
            "trajectory_csv": str(trajectory_path),
            "usd_stage": str(usd_path),
        }
        summary_path = output_dir / "summary.json"
        summary["summary_json"] = str(summary_path)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return summary
    finally:
        environment.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-plan", required=True)
    parser.add_argument("--policy")
    parser.add_argument(
        "--output-dir",
        default="drl_env/outputs/brain_cpp_ppo_isaac",
    )
    parser.add_argument("--platform", default="Blue_CH4_Recon")
    parser.add_argument("--max-frames", type=int, default=1600)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    return 0 if summary.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
