#!/usr/bin/env python3
"""Run MPPI transit then PPO coverage for the allocated UAVs inside Isaac Sim."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain.integration import IsaacAirCombatEnvironment  # noqa: E402
from drl_env.core import MAX_ACCEL, MAX_SPEED  # noqa: E402
from drl_env.drl_motion_model import DrlAvoidanceModel  # noqa: E402
from drl_env.numpy_policy import NumpyPolicy  # noqa: E402


COLORS = (
    (0.05, 0.90, 1.00),
    (1.00, 0.20, 0.68),
    (0.18, 1.00, 0.40),
    (1.00, 0.78, 0.10),
    (0.64, 0.36, 1.00),
)


class CountingPolicy:
    def __init__(self, base: NumpyPolicy) -> None:
        self.base = base
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def forward(self, observation: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.base.forward(observation)


class ReplayMotionModel:
    def __init__(self, points: list[np.ndarray]) -> None:
        from scenes.aircraft_motion import MotionState

        self.points = points
        self.index = 0
        self.state = MotionState(position=points[0].copy(), velocity=np.zeros(3))

    def step(self, dt: float, target_position=None, target_velocity=None):
        previous = self.state.position.copy()
        self.index = min(self.index + 1, len(self.points) - 1)
        self.state.position = self.points[self.index].copy()
        self.state.velocity = (self.state.position - previous) / max(1e-6, float(dt))
        if np.linalg.norm(self.state.velocity[:2]) > 1e-6:
            self.state.yaw_deg = math.degrees(math.atan2(
                float(self.state.velocity[1]), float(self.state.velocity[0])
            ))
        return self.state


class HoldMotionModel:
    def __init__(self, position: np.ndarray) -> None:
        from scenes.aircraft_motion import MotionState

        self.state = MotionState(position=position.copy(), velocity=np.zeros(3))

    def step(self, dt: float, target_position=None, target_velocity=None):
        self.state.velocity[:] = 0.0
        return self.state


def load_plan(path: Path) -> dict[str, dict[str, list[np.ndarray]]]:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"transit": [], "search": []}
    )
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expected = "mppi_follow" if row["phase"] == "transit" else "ppo_local"
            if row["controller"] != expected:
                raise ValueError(f"invalid controller boundary in {path}: {row}")
            grouped[row["platform"]][row["phase"]].append(np.array([
                float(row["x"]), float(row["y"]), float(row["z"])
            ]))
    return dict(grouped)


def sample_polyline(points: list[np.ndarray], count: int) -> list[np.ndarray]:
    lengths = [0.0]
    for index in range(1, len(points)):
        lengths.append(lengths[-1] + float(np.linalg.norm(points[index] - points[index - 1])))
    output: list[np.ndarray] = []
    segment = 1
    for target in np.linspace(0.0, lengths[-1], count):
        while segment < len(lengths) - 1 and lengths[segment] < target:
            segment += 1
        ratio = (target - lengths[segment - 1]) / max(
            1e-9, lengths[segment] - lengths[segment - 1]
        )
        output.append(points[segment - 1] * (1.0 - ratio) + points[segment] * ratio)
    return output


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


def create_local_obstacle(stage, path: str, position: np.ndarray, color):
    from pxr import Gf, Sdf, UsdGeom

    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cylinder.CreateRadiusAttr(2.0)
    cylinder.CreateHeightAttr(80.0)
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.Xformable(cylinder.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(float(position[0]), float(position[1]), 40.0)
    )


def hide_unassigned(stage, visible: set[str]) -> None:
    from pxr import UsdGeom

    root = stage.GetPrimAtPath("/World/AirCombat/Platforms")
    for child in root.GetChildren():
        if child.GetName() not in visible:
            UsdGeom.Imageable(child).MakeInvisible()
    rings = stage.GetPrimAtPath("/World/AirCombat/SensorRings")
    if rings:
        UsdGeom.Imageable(rings).MakeInvisible()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def patrol_end_requested(path: Path, field: str) -> bool:
    """Read the externally-owned patrol termination flag."""
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get(field, False)
    if not isinstance(value, bool):
        raise ValueError(f"{field!r} in {path} must be a JSON boolean")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    plans = load_plan(Path(args.plan))
    planning = json.loads(Path(args.planning_summary).read_text(encoding="utf-8"))
    active = list(plans)
    reserve = list(planning.get("reserve_aircraft", []))
    output_dir = Path(args.output_dir).resolve()
    frames_dir = output_dir / "isaac_frames"
    patrol_control_path = Path(args.patrol_control or args.planning_summary).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    environment = IsaacAirCombatEnvironment(
        scene_config={
            "simulation": {"time_scale": 1.0, "aircraft_model": "kinematic"},
            "weather": {"clouds": False, "local_zones": False, "rain": False},
            "visual": {
                "show_sensor_rings": False,
                "show_sensor_cones": False,
                "show_route_lines": False,
            },
        },
        headless=args.headless,
        app_config={"width": args.width, "height": args.height},
    )
    samples: list[dict[str, Any]] = []
    annotator = None
    product = None
    try:
        scene = environment.initialize()
        import omni.replicator.core as rep
        from pxr import Gf
        from sensors.io import save_rgb

        platform_map = {item.entity_id: item for item in scene.platforms}
        missing = [name for name in active + reserve if name not in platform_map]
        if missing:
            raise RuntimeError(f"Isaac platforms missing: {missing}")
        hide_unassigned(environment.stage, set(active + reserve))

        transit_models: dict[str, ReplayMotionModel] = {}
        trails = {}
        trail_points: dict[str, list[np.ndarray]] = defaultdict(list)
        all_points = []
        for index, name in enumerate(active):
            transit = sample_polyline(plans[name]["transit"], args.transit_steps + 1)
            model = ReplayMotionModel(transit)
            transit_models[name] = model
            platform_map[name].motion_model = model
            all_points.extend(plans[name]["transit"] + plans[name]["search"])
            create_curve(
                environment.stage,
                f"/World/AirCombat/PhaseMission/Plan_{index}",
                tuple(component * 0.52 for component in COLORS[index]),
                2.0,
                plans[name]["transit"] + plans[name]["search"],
            )
            trails[name] = create_curve(
                environment.stage,
                f"/World/AirCombat/PhaseMission/Trail_{index}",
                COLORS[index], 4.5, [transit[0], transit[0]],
            )
        for name in reserve:
            platform_map[name].motion_model = HoldMotionModel(platform_map[name].position)

        xs = [float(point[0]) for point in all_points]
        ys = [float(point[1]) for point in all_points]
        center_x = (min(xs) + max(xs)) * 0.5
        center_y = (min(ys) + max(ys)) * 0.5
        span = max(max(xs) - min(xs), max(ys) - min(ys), 800.0)
        camera = rep.create.camera(
            position=(center_x + span * 0.52, center_y - span * 0.72, 1850.0),
            look_at=(center_x, center_y, 100.0),
        )
        product = rep.create.render_product(camera, resolution=(args.width, args.height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        for _ in range(3):
            rep.orchestrator.step()
            environment.app.update()

        rendered = 0

        def capture(phase: str, sim_step: int) -> None:
            nonlocal rendered
            for index, name in enumerate(active):
                point = platform_map[name].position.copy()
                trail_points[name].append(point)
                values = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]) + 3.0)
                          for p in trail_points[name]]
                if len(values) == 1:
                    values.append(values[0])
                trails[name].GetCurveVertexCountsAttr().Set([len(values)])
                trails[name].GetPointsAttr().Set(values)
                trails[name].GetWidthsAttr().Set([4.5] * len(values))
            rep.orchestrator.step()
            rgb = annotator.get_data()
            if rgb is None or rgb.size == 0:
                raise RuntimeError(f"Isaac returned empty frame at {phase}:{sim_step}")
            save_rgb(rgb, frames_dir / f"frame_{rendered:04d}.png")
            rendered += 1
            if rendered == 1 or rendered % 20 == 0:
                print(
                    f"ISAAC_CAPTURE frame={rendered} phase={phase} "
                    f"sim_step={sim_step} rgb_mean={float(rgb[:, :, :3].mean()):.2f}",
                    flush=True,
                )

        for step in range(args.transit_steps):
            environment.step(args.dt)
            for name in active:
                state = platform_map[name].motion_model.state
                samples.append({
                    "platform": name, "phase": "transit", "controller": "mppi_follow",
                    "step": step, "ppo_called": 0, "patrol_cycle": 0,
                    "x": float(state.position[0]), "y": float(state.position[1]),
                    "z": float(state.position[2]),
                })
            if step % args.transit_render_stride == 0:
                capture("transit", step)

        base_policy = NumpyPolicy(args.policy)
        ppo_models: dict[str, DrlAvoidanceModel] = {}
        policies: dict[str, CountingPolicy] = {}
        patrol_cycles = {name: 0 for name in active}
        for index, name in enumerate(active):
            search = plans[name]["search"]
            obstacle_position = (search[0] + search[1]) * 0.5
            obstacle = {"position": obstacle_position, "radius_units": args.obstacle_radius}
            create_local_obstacle(
                environment.stage,
                f"/World/AirCombat/PhaseMission/LocalObstacle_{index}",
                obstacle_position,
                (1.0, 0.15, 0.05),
            )
            policy = CountingPolicy(base_policy)
            policies[name] = policy
            model = DrlAvoidanceModel(
                initial_position=platform_map[name].position,
                max_speed=MAX_SPEED,
                max_accel=MAX_ACCEL,
                policy=policy,
                waypoints=search,
                obstacles=[obstacle],
                wp_reach_dist=12.0,
            )
            ppo_models[name] = model
            platform_map[name].motion_model = model

        search_steps = 0
        patrol_end_received = False
        termination_reason = "recording_limit"
        for step in range(args.max_search_steps):
            environment.step(args.dt)
            search_steps = step + 1
            for name in active:
                model = ppo_models[name]
                state = model.state
                samples.append({
                    "platform": name, "phase": "search", "controller": "ppo_local",
                    "step": step, "ppo_called": 1,
                    "patrol_cycle": patrol_cycles[name],
                    "x": float(state.position[0]), "y": float(state.position[1]),
                    "z": float(state.position[2]),
                })
                if model.waypoint_completed and float(
                    np.linalg.norm(model.state.position - plans[name]["search"][-1])
                ) < 12.0:
                    patrol_cycles[name] += 1
                    model.replace_waypoints(plans[name]["search"])
            if step % args.search_render_stride == 0:
                capture("search", step)

            if step % args.patrol_control_poll_steps == 0 and patrol_end_requested(
                patrol_control_path, args.patrol_end_field
            ):
                patrol_end_received = True
                termination_reason = "patrol_end_signal"
                for name in active:
                    platform_map[name].motion_model = HoldMotionModel(
                        ppo_models[name].state.position
                    )
                capture("patrol_end", step)
                break

        stage_path = output_dir / "isaac_phase_mission.usda"
        environment.stage.GetRootLayer().Export(str(stage_path))
        trajectory_path = output_dir / "isaac_executed_trajectory.csv"
        write_rows(trajectory_path, samples)
        platform_results = []
        for name in active:
            model = ppo_models[name]
            platform_results.append({
                "platform": name,
                "patrol_cycles": patrol_cycles[name],
                "patrolling_at_stop": not patrol_end_received,
                "transit_ppo_calls": 0,
                "search_ppo_calls": policies[name].calls,
                "collisions": int(model._stats["collisions"]),
                "distance_to_current_waypoint": round(float(np.linalg.norm(
                    model.state.position - model.current_waypoint
                )), 6),
            })
        success = bool(
            all(item["transit_ppo_calls"] == 0 for item in platform_results)
            and all(item["search_ppo_calls"] > 0 for item in platform_results)
            and all(item["collisions"] == 0 for item in platform_results)
            and (patrol_end_received or all(
                item["patrol_cycles"] > 0 for item in platform_results
            ))
        )
        summary = {
            "success": success,
            "runtime": "IsaacAirCombatEnvironment",
            "active_aircraft_count": len(active),
            "reserve_aircraft": reserve,
            "hidden_helicopters": ["Blue_WZ21_Leader", "Blue_WZ21_Wingman"],
            "transit_controller": "mppi_follow",
            "search_controller": "ppo_local",
            "transit_steps": args.transit_steps,
            "search_steps": search_steps,
            "patrol_end_received": patrol_end_received,
            "termination_reason": termination_reason,
            "patrol_control_file": str(patrol_control_path),
            "patrol_end_field": args.patrol_end_field,
            "rendered_frames": rendered,
            "policy": str(Path(args.policy).resolve()),
            "platforms": platform_results,
            "trajectory_csv": str(trajectory_path),
            "usd_stage": str(stage_path),
            "frames_dir": str(frames_dir),
        }
        summary_path = output_dir / "isaac_execution_summary.json"
        summary["summary_json"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return summary
    finally:
        if annotator is not None and product is not None:
            annotator.detach()
            product.destroy()
        environment.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--planning-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", default="drl_env/models/ppo_drone_best_v9_reactive_phase3_l3.npz")
    parser.add_argument("--transit-steps", type=int, default=180)
    parser.add_argument("--max-search-steps", type=int, default=2000)
    parser.add_argument(
        "--patrol-control",
        help="JSON state file polled during patrol; defaults to --planning-summary",
    )
    parser.add_argument("--patrol-end-field", default="patrol_end")
    parser.add_argument("--patrol-control-poll-steps", type=int, default=10)
    parser.add_argument("--transit-render-stride", type=int, default=3)
    parser.add_argument("--search-render-stride", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--obstacle-radius", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.patrol_control_poll_steps < 1:
        parser.error("--patrol-control-poll-steps must be at least 1")
    try:
        result = run(args)
        return 0 if result.get("success") else 2
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
