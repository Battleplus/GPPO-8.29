"""Command-line bridge for calling the SAR planner from C++.

The C++ side writes a small JSON request, calls this script, and reads a
machine-friendly CSV or JSON response containing waypoints.

Example:
    python search_planner/cpp_plan_bridge.py --input request.json --output waypoints.csv --format csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

_PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from search_planner.config import PlannerConfig
from search_planner.mission import build_multi_region_cycle
from search_planner.planner import PlannerResult, Waypoint, plan


_CONFIG_FIELDS = {f.name for f in fields(PlannerConfig)}


def _load_request(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("request root must be a JSON object")
    return data


def _normalise_task(task: dict[str, Any], index: int) -> tuple[str, PlannerConfig]:
    data = dict(task)
    platform_id = str(
        data.pop("platform_id", data.pop("_platform_id", f"UAV_{index + 1}"))
    )

    # Friendly aliases for C++ callers.
    if "center_km" in data and "area_center_km" not in data:
        data["area_center_km"] = data.pop("center_km")
    if "width_km" in data and "area_width_km" not in data:
        data["area_width_km"] = data.pop("width_km")
    if "height_km" in data and "area_height_km" not in data:
        data["area_height_km"] = data.pop("height_km")

    unknown = sorted(k for k in data if k not in _CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            f"task {index} has unsupported field(s): {', '.join(unknown)}"
        )

    return platform_id, PlannerConfig(**data)


def _path_length_km(waypoints: list[Waypoint], meters_per_unit: float) -> float:
    total_units = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i].x - waypoints[i - 1].x
        dy = waypoints[i].y - waypoints[i - 1].y
        dz = waypoints[i].z - waypoints[i - 1].z
        total_units += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total_units * meters_per_unit / 1000.0


def _waypoint_dict(wp: Waypoint, index: int) -> dict[str, float | int]:
    return {
        "index": index,
        "x": float(wp.x),
        "y": float(wp.y),
        "z": float(wp.z),
        "terrain_z": float(wp.terrain_z),
        "yaw_deg": float(wp.yaw_deg),
    }


def _plan_request(request: dict[str, Any]) -> dict[str, Any]:
    if "tasks" in request:
        raw_tasks = request["tasks"]
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("'tasks' must be a non-empty list")
    else:
        raw_tasks = [request.get("config", request)]

    grouped: dict[str, list[tuple[str, PlannerResult]]] = defaultdict(list)
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict):
            raise ValueError(f"task {index} must be a JSON object")
        platform_id, cfg = _normalise_task(raw_task, index)
        result = plan(cfg)
        grouped[platform_id].append((f"{platform_id}_r{index}", result))

    plans = []
    for platform_id, items in grouped.items():
        if len(items) == 1:
            result = items[0][1]
            waypoints = list(result.waypoints)
            mpu = result.config.meters_per_unit
            total_km = float(result.stats.get("path_length_km", 0.0))
        else:
            results_map = {region_id: result for region_id, result in items}
            cycle_order = list(results_map.keys())
            centers = {
                region_id: (result.search_area.center_x, result.search_area.center_y)
                for region_id, result in items
            }
            waypoints = build_multi_region_cycle(results_map, cycle_order, centers)
            mpu = items[0][1].config.meters_per_unit
            total_km = _path_length_km(waypoints, mpu)

        plans.append(
            {
                "platform_id": platform_id,
                "total_km": round(total_km, 3),
                "waypoints": [
                    _waypoint_dict(wp, point_index)
                    for point_index, wp in enumerate(waypoints)
                ],
            }
        )

    return {"ok": True, "plans": plans}


def _write_json(path: str, response: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)


def _flatten_waypoints(response: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for plan_data in response["plans"]:
        for wp in plan_data["waypoints"]:
            points.append(
                {
                    "platform_id": plan_data["platform_id"],
                    "point_index": wp["index"],
                    "x": wp["x"],
                    "y": wp["y"],
                    "z": wp["z"],
                    "terrain_z": wp["terrain_z"],
                    "yaw_deg": wp["yaw_deg"],
                    "total_km": plan_data["total_km"],
                }
            )
    return points


def _write_list_json(path: str, response: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_flatten_waypoints(response), f, ensure_ascii=False, indent=2)


def _write_csv(path: str, response: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["platform_id", "point_index", "x", "y", "z", "terrain_z", "yaw_deg", "total_km"]
        )
        for plan_data in response["plans"]:
            for wp in plan_data["waypoints"]:
                writer.writerow(
                    [
                        plan_data["platform_id"],
                        wp["index"],
                        f"{wp['x']:.9f}",
                        f"{wp['y']:.9f}",
                        f"{wp['z']:.9f}",
                        f"{wp['terrain_z']:.9f}",
                        f"{wp['yaw_deg']:.9f}",
                        f"{plan_data['total_km']:.9f}",
                    ]
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON request path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--format", choices=("csv", "json", "list_json"), default="csv")
    args = parser.parse_args()

    try:
        request = _load_request(args.input)
        response = _plan_request(request)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "list_json":
            _write_list_json(str(output_path), response)
        elif args.format == "json":
            _write_json(str(output_path), response)
        else:
            _write_csv(str(output_path), response)
    except Exception as exc:
        error = {"ok": False, "error": str(exc)}
        try:
            _write_json(args.output, error)
        finally:
            print(f"[SAR_CPP_BRIDGE] ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
