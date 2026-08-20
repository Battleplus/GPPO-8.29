#!/usr/bin/env python3
"""Join an MPPI transit route with a task-area patrol path."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from mppi.mppi import MPPIConfig, MPPIPlanner
from mppi.obstacles import build_obstacles


def read_patrol(path: Path) -> tuple[dict[str, str], list[np.ndarray]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"patrol path has fewer than two points: {path}")
    metadata = {key: rows[0][key] for key in ("platform", "sensor", "pattern", "cell")}
    points = [np.array([float(row["x"]), float(row["y"]), float(row["z"])]) for row in rows]
    return metadata, points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patrol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", nargs=3, type=float, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--transit-waypoints", type=int, default=12)
    parser.add_argument("--patrol-waypoints", type=int, default=10)
    parser.add_argument("--cruise-altitude", type=float, default=180.0)
    args = parser.parse_args()

    metadata, patrol = read_patrol(Path(args.patrol))
    start = np.asarray(args.start, dtype=float)
    start[2] = max(start[2], args.cruise_altitude)
    for point in patrol:
        point[2] = max(point[2], args.cruise_altitude)
    patrol_count = max(2, min(len(patrol), args.patrol_waypoints))
    patrol_indices = sorted({
        round(index * (len(patrol) - 1) / (patrol_count - 1))
        for index in range(patrol_count)
    })
    patrol = [patrol[index] for index in patrol_indices]
    goal = patrol[0].copy()
    config = MPPIConfig(
        map_size_units=3000.0,
        map_origin=(-1500.0, -1500.0),
        max_altitude=220.0,
        num_samples=args.samples,
        num_iterations=args.iterations,
        horizon=args.horizon,
    )
    transit = MPPIPlanner(obstacles=build_obstacles(), config=config).plan(
        start=start,
        goal=goal,
        verbose=True,
    )
    if transit is None or len(transit) < 2:
        raise RuntimeError("MPPI did not produce a transit route")

    transit_count = max(2, min(len(transit), args.transit_waypoints))
    transit_indices = sorted({
        round(index * (len(transit) - 1) / (transit_count - 1))
        for index in range(transit_count)
    })
    transit_keypoints = [np.asarray(transit[index], dtype=float) for index in transit_indices]
    combined = list(transit_keypoints)
    if np.linalg.norm(combined[-1] - patrol[0]) > 1e-6:
        combined.append(patrol[0])
    combined.extend(patrol[1:])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("platform", "sensor", "pattern", "cell", "point_index", "x", "y", "z"))
        for index, point in enumerate(combined):
            writer.writerow((
                metadata["platform"], metadata["sensor"],
                "mppi_transit+" + metadata["pattern"], metadata["cell"], index,
                float(point[0]), float(point[1]), float(point[2]),
            ))
    print(
        f"MPPI_PATROL_PATH_OK platform={metadata['platform']} cell={metadata['cell']} "
        f"transit_dense={len(transit)} transit_keypoints={len(transit_keypoints)} "
        f"patrol_keypoints={len(patrol)} combined={len(combined)} "
        f"cruise_altitude={args.cruise_altitude} output={output.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
