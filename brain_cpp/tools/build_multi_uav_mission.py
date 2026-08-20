#!/usr/bin/env python3
"""Build synchronized transit and area-search trajectories for multiple UAVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Assignment:
    platform: str
    task_id: str
    sensor: str
    start: tuple[float, float, float]
    center: tuple[float, float]
    color: str


ASSIGNMENTS = (
    Assignment(
        "Blue_CH4_Recon", "SEARCH_NW", "SAR",
        (-1500.0, -580.0, 180.0), (-520.0, 470.0), "cyan",
    ),
    Assignment(
        "Blue_CH4_StrikeRecon", "SEARCH_NE", "EOIR",
        (-1500.0, -860.0, 180.0), (100.0, 330.0), "magenta",
    ),
    Assignment(
        "Blue_Quad_Recon_1", "SEARCH_SW", "EO",
        (-1420.0, 280.0, 180.0), (-470.0, -390.0), "green",
    ),
    Assignment(
        "Blue_Quad_Recon_2", "SEARCH_SE", "SAR",
        (-1420.0, -240.0, 180.0), (330.0, -430.0), "yellow",
    ),
)


def transit_points(
    assignment: Assignment, goal: np.ndarray, index: int
) -> list[np.ndarray]:
    start = np.asarray(assignment.start, dtype=float)
    delta = goal - start
    normal = np.array([-delta[1], delta[0], 0.0], dtype=float)
    normal /= max(1e-6, float(np.linalg.norm(normal[:2])))
    bend = (80.0 + 30.0 * index) * (-1.0 if index % 2 else 1.0)
    control_1 = start + delta * 0.32 + normal * bend
    control_2 = start + delta * 0.70 - normal * bend * 0.55
    points: list[np.ndarray] = []
    for step in range(15):
        t = step / 14.0
        point = (
            (1.0 - t) ** 3 * start
            + 3.0 * (1.0 - t) ** 2 * t * control_1
            + 3.0 * (1.0 - t) * t**2 * control_2
            + t**3 * goal
        )
        point[2] = 180.0 + 8.0 * math.sin(math.pi * t)
        points.append(point)
    return points


def search_points(assignment: Assignment) -> list[np.ndarray]:
    cx, cy = assignment.center
    half_width = 180.0
    half_height = 140.0
    rows = 7
    points: list[np.ndarray] = []
    for row in range(rows):
        y = cy - half_height + 2.0 * half_height * row / (rows - 1)
        endpoints = (
            (cx - half_width, y), (cx + half_width, y)
        ) if row % 2 == 0 else (
            (cx + half_width, y), (cx - half_width, y)
        )
        for x, y_value in endpoints:
            points.append(np.array([x, y_value, 180.0], dtype=float))
    return points


def sample_polyline(points: list[np.ndarray], count: int) -> list[np.ndarray]:
    if count <= 1:
        return [points[0].copy()]
    lengths = [0.0]
    for index in range(1, len(points)):
        lengths.append(lengths[-1] + float(np.linalg.norm(
            points[index] - points[index - 1]
        )))
    total = lengths[-1]
    samples: list[np.ndarray] = []
    segment = 1
    for target_distance in np.linspace(0.0, total, count):
        while segment < len(lengths) - 1 and lengths[segment] < target_distance:
            segment += 1
        before = lengths[segment - 1]
        after = lengths[segment]
        ratio = (target_distance - before) / max(1e-9, after - before)
        samples.append(
            points[segment - 1] * (1.0 - ratio) + points[segment] * ratio
        )
    return samples


def write_plan(path: Path, plans: dict[str, dict]) -> None:
    fields = (
        "platform", "task_id", "sensor", "phase", "point_index",
        "x", "y", "z", "region_center_x", "region_center_y",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for platform, plan in plans.items():
            point_index = 0
            for phase in ("transit", "search"):
                for point in plan[phase]:
                    writer.writerow({
                        "platform": platform,
                        "task_id": plan["assignment"].task_id,
                        "sensor": plan["assignment"].sensor,
                        "phase": phase,
                        "point_index": point_index,
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "z": float(point[2]),
                        "region_center_x": plan["assignment"].center[0],
                        "region_center_y": plan["assignment"].center[1],
                    })
                    point_index += 1


def write_trajectory(
    path: Path, plans: dict[str, dict], frame_count: int, fps: float
) -> None:
    fields = (
        "platform", "frame", "time_s", "phase", "task_id", "sensor",
        "x", "y", "z", "vx", "vy", "vz",
    )
    transit_count = max(2, round(frame_count * 0.45))
    search_count = frame_count - transit_count + 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for platform, plan in plans.items():
            transit = sample_polyline(plan["transit"], transit_count)
            search = sample_polyline(plan["search"], search_count)
            positions = transit + search[1:]
            previous = positions[0]
            for frame, position in enumerate(positions):
                velocity = (
                    np.zeros(3, dtype=float)
                    if frame == 0 else (position - previous) * fps
                )
                writer.writerow({
                    "platform": platform,
                    "frame": frame,
                    "time_s": frame / fps,
                    "phase": "transit" if frame < transit_count else "search",
                    "task_id": plan["assignment"].task_id,
                    "sensor": plan["assignment"].sensor,
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                    "vx": float(velocity[0]),
                    "vy": float(velocity[1]),
                    "vz": float(velocity[2]),
                })
                previous = position


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plans: dict[str, dict] = {}
    for index, assignment in enumerate(ASSIGNMENTS):
        search = search_points(assignment)
        transit = transit_points(assignment, search[0], index)
        plans[assignment.platform] = {
            "assignment": assignment,
            "transit": transit,
            "search": search,
        }

    plan_path = output_dir / "multi_uav_task_plan.csv"
    trajectory_path = output_dir / "multi_uav_trajectory.csv"
    summary_path = output_dir / "multi_uav_summary.json"
    write_plan(plan_path, plans)
    write_trajectory(trajectory_path, plans, args.frames, args.fps)
    summary = {
        "success": True,
        "platform_count": len(ASSIGNMENTS),
        "frame_count": args.frames,
        "fps": args.fps,
        "duration_s": args.frames / args.fps,
        "plan_csv": str(plan_path),
        "trajectory_csv": str(trajectory_path),
        "video": str(output_dir / "mission_multi_uav.mp4"),
        "render_mode": "Isaac camera projection tactical overlay",
        "assignments": [
            {
                "platform": item.platform,
                "task_id": item.task_id,
                "sensor": item.sensor,
                "region_center": list(item.center),
                "color": item.color,
            }
            for item in ASSIGNMENTS
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
