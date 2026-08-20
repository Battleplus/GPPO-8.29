#!/usr/bin/env python3
"""Execute MPPI transit without PPO, then PPO-controlled coverage search."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drl_env.core import MAX_ACCEL, MAX_SPEED  # noqa: E402
from drl_env.drl_motion_model import DrlAvoidanceModel  # noqa: E402
from drl_env.numpy_policy import NumpyPolicy  # noqa: E402


class CountingPolicy:
    def __init__(self, policy: NumpyPolicy) -> None:
        self._policy = policy
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def forward(self, observation: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self._policy.forward(observation)


def load_plan(path: Path) -> dict[str, dict[str, list[np.ndarray]]]:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"transit": [], "search": []}
    )
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phase = row["phase"]
            controller = row["controller"]
            expected = "mppi_follow" if phase == "transit" else "ppo_local"
            if controller != expected:
                raise ValueError(
                    f"{row['platform']} phase={phase} uses {controller}, expected {expected}"
                )
            grouped[row["platform"]][phase].append(
                np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            )
    result = dict(grouped)
    for platform, phases in result.items():
        if len(phases["transit"]) < 2 or len(phases["search"]) < 2:
            raise ValueError(f"{platform} has incomplete phase plans")
    return result


def sample_polyline(points: list[np.ndarray], count: int) -> list[np.ndarray]:
    lengths = [0.0]
    for index in range(1, len(points)):
        lengths.append(lengths[-1] + float(np.linalg.norm(points[index] - points[index - 1])))
    samples: list[np.ndarray] = []
    segment = 1
    for target in np.linspace(0.0, lengths[-1], count):
        while segment < len(lengths) - 1 and lengths[segment] < target:
            segment += 1
        before, after = lengths[segment - 1], lengths[segment]
        ratio = (target - before) / max(1e-9, after - before)
        samples.append(points[segment - 1] * (1.0 - ratio) + points[segment] * ratio)
    return samples


def row(
    platform: str, frame: int, time_s: float, phase: str, controller: str,
    position: np.ndarray, velocity: np.ndarray, waypoint: int, ppo_called: int,
) -> dict[str, Any]:
    return {
        "platform": platform,
        "frame": frame,
        "time_s": round(time_s, 6),
        "phase": phase,
        "controller": controller,
        "waypoint_index": waypoint,
        "ppo_called": ppo_called,
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "vx": float(velocity[0]),
        "vy": float(velocity[1]),
        "vz": float(velocity[2]),
    }


def execute_platform(
    platform: str,
    phases: dict[str, list[np.ndarray]],
    base_policy: NumpyPolicy,
    transit_frames: int,
    dt: float,
    max_search_steps: int,
    obstacle_radius: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    transit = sample_polyline(phases["transit"], transit_frames)
    previous = transit[0]
    for frame, position in enumerate(transit):
        velocity = np.zeros(3) if frame == 0 else (position - previous) / dt
        samples.append(row(
            platform, frame, frame * dt, "transit", "mppi_follow",
            position, velocity, frame, 0,
        ))
        previous = position

    policy = CountingPolicy(base_policy)
    obstacle_position = (
        phases["search"][0] + phases["search"][1]
    ) * 0.5
    obstacles = [{
        "position": obstacle_position,
        "radius_units": obstacle_radius,
    }]
    model = DrlAvoidanceModel(
        initial_position=transit[-1],
        max_speed=MAX_SPEED,
        max_accel=MAX_ACCEL,
        policy=policy,
        waypoints=phases["search"],
        obstacles=obstacles,
        wp_reach_dist=12.0,
    )
    switch_frame = len(samples)
    completed = False
    for step in range(max_search_steps):
        state = model.step(dt)
        frame = switch_frame + step
        samples.append(row(
            platform, frame, frame * dt, "search", "ppo_local",
            state.position, state.velocity, model.wp_idx, 1,
        ))
        if model.waypoint_completed and float(
            np.linalg.norm(state.position - phases["search"][-1])
        ) < 12.0:
            completed = True
            break

    transit_ppo_calls = sum(item["ppo_called"] for item in samples if item["phase"] == "transit")
    search_ppo_calls = sum(item["ppo_called"] for item in samples if item["phase"] == "search")
    result = {
        "platform": platform,
        "completed": completed,
        "switch_frame": switch_frame,
        "transit_frames": transit_frames,
        "search_frames": len(samples) - switch_frame,
        "transit_ppo_calls": transit_ppo_calls,
        "search_ppo_calls": search_ppo_calls,
        "policy_forward_calls": policy.calls,
        "search_waypoints": len(phases["search"]),
        "waypoints_reached": int(model._stats["wp_reached"]),
        "collisions": int(model._stats["collisions"]),
        "local_obstacle_count": len(obstacles),
        "local_obstacle_radius": obstacle_radius,
        "minimum_obstacle_clearance": round(
            float(model._stats["min_obs_dist"] - obstacle_radius - 2.0), 6
        ),
        "final_distance": round(float(np.linalg.norm(
            model.state.position - phases["search"][-1]
        )), 6),
    }
    return samples, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policy",
        default="drl_env/models/ppo_drone_best_v9_reactive_phase3_l3.npz",
    )
    parser.add_argument("--transit-frames", type=int, default=180)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-search-steps", type=int, default=20000)
    parser.add_argument("--local-obstacle-radius", type=float, default=2.0)
    args = parser.parse_args()

    plan = load_plan(Path(args.plan))
    policy = NumpyPolicy(args.policy)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples: list[dict[str, Any]] = []
    platform_results: list[dict[str, Any]] = []
    for platform, phases in plan.items():
        samples, result = execute_platform(
            platform, phases, policy, args.transit_frames, args.dt,
            args.max_search_steps, args.local_obstacle_radius,
        )
        all_samples.extend(samples)
        platform_results.append(result)
        print(
            f"PHASE_TEST platform={platform} completed={result['completed']} "
            f"transit_ppo={result['transit_ppo_calls']} "
            f"search_ppo={result['search_ppo_calls']} collisions={result['collisions']}",
            flush=True,
        )

    trajectory_path = output_dir / "executed_trajectory.csv"
    fields = list(all_samples[0])
    with trajectory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_samples)

    success = bool(
        platform_results
        and all(item["completed"] for item in platform_results)
        and all(item["transit_ppo_calls"] == 0 for item in platform_results)
        and all(item["search_ppo_calls"] > 0 for item in platform_results)
        and all(item["policy_forward_calls"] == item["search_ppo_calls"] for item in platform_results)
        and all(item["collisions"] == 0 for item in platform_results)
        and all(item["minimum_obstacle_clearance"] > 0.0 for item in platform_results)
    )
    summary = {
        "success": success,
        "active_aircraft_count": len(platform_results),
        "transit_controller": "mppi_follow",
        "search_controller": "ppo_local",
        "policy": str(Path(args.policy).resolve()),
        "trajectory_csv": str(trajectory_path),
        "platforms": platform_results,
    }
    summary_path = output_dir / "execution_summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
