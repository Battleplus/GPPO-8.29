#!/usr/bin/env python3
"""File-based bridge exposing Perch position selection to C++ callers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain.domain.task import StrikeTask
from perch.region_selector import AttackRegionSelector
from perch.selector import FREAPositionSelector


OUTPUT_FIELDS = (
    "pos_id",
    "x",
    "y",
    "z",
    "kind",
    "coordinate_frame",
    "platform_id",
    "target_id",
    "munition",
    "rank",
    "source",
    "target_range_km",
    "agl_m",
    "g_violation",
    "optimiser",
    "situation",
    "knowledge_sources",
)


def _load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Perch request root must be a JSON object")
    if not isinstance(data.get("tasks"), list) or not data["tasks"]:
        raise ValueError("Perch request must contain at least one task")
    return data


def _build_context(request: dict[str, Any]) -> tuple[Any, list[StrikeTask]]:
    world = dict(request.get("world_state") or {})
    world.setdefault("meters_per_unit", 100.0)
    world.setdefault("map_size_km", 300.0)
    world.setdefault("map_size_units", 3000.0)
    options = dict(request.get("options") or {})
    world["attack_region_mode"] = str(
        options.get("attack_region_mode", "llm")
    )
    world["attack_region_strict"] = bool(
        options.get("attack_region_strict", True)
    )
    if str(options.get("terrain_mode", "scene")).lower() == "flat":
        world["terrain_fn"] = lambda x, y: 0.0

    agents = []
    for raw in request.get("agents") or []:
        position = raw.get("position", raw.get("pos", [0.0, 0.0]))
        agents.append(SimpleNamespace(
            pid=str(raw.get("pid", "")),
            type=str(raw.get("type", "unknown")),
            position=(float(position[0]), float(position[1])),
            sensors=list(raw.get("sensors") or []),
            munitions=dict(raw.get("munitions") or {}),
            altitude_km=float(raw.get("altitude_km", 0.3)),
            lost=bool(raw.get("lost", False)),
        ))

    tasks = []
    for raw in request["tasks"]:
        tasks.append(StrikeTask(
            platform=str(raw["platform"]),
            target=str(raw["target"]),
            munition=str(raw.get("munition", "HF")),
            qty=int(raw.get("qty", 1)),
            role=str(raw.get("role", "lead")),
            aoi=str(raw.get("aoi", "")),
            assigned_munitions={
                str(name): int(qty)
                for name, qty in (raw.get("assigned_munitions") or {}).items()
            },
        ))
    return SimpleNamespace(world_state=world, agents=agents), tasks


def select_positions(request: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    context, tasks = _build_context(request)
    options = dict(request.get("options") or {})
    mode = str(options.get("attack_region_mode", "llm"))
    top_k = max(1, int(options.get("top_k", 3)))
    region_result = AttackRegionSelector(
        mode=mode,
        top_k=top_k,
    ).select_regions(context, tasks)
    if not region_result.success:
        raise RuntimeError(region_result.reason or "Perch region selection failed")

    position_result = FREAPositionSelector(
        preference=str(options.get("preference", "balanced")),
        top_k=top_k,
        use_pymoo=bool(options.get("use_pymoo", False)),
    ).select(context, tasks)
    if not position_result.success:
        raise RuntimeError(position_result.reason or "FREA position selection failed")
    audit = {
        "situations": context.world_state.get("attack_region_situations", {}),
        "knowledge_sources": context.world_state.get(
            "attack_region_knowledge_sources", {}
        ),
        "attack_regions": context.world_state.get("attack_regions", {}),
        "meters_per_unit": float(context.world_state["meters_per_unit"]),
        "map_size_km": float(context.world_state["map_size_km"]),
    }
    return list(position_result.data), audit


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _write_positions(
    path: Path,
    positions: list[Any],
    audit: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(OUTPUT_FIELDS) + "\n")
        for position in positions:
            metadata = dict(getattr(position, "metadata", {}) or {})
            target_id = str(metadata.get("target_id", ""))
            x, y, z = float(position.x), float(position.y), float(position.z)
            coordinate_frame = str(metadata.get("coordinate_frame", ""))
            if coordinate_frame == "isaac_scene":
                scale = float(audit["meters_per_unit"]) / 1000.0
                half_km = float(audit["map_size_km"]) * 0.5
                x = half_km + x * scale
                y = half_km + y * scale
                z = z * scale
                coordinate_frame = "mission_km"
            row = {
                "pos_id": position.pos_id,
                "x": x,
                "y": y,
                "z": z,
                "kind": position.kind,
                "coordinate_frame": coordinate_frame or "mission_km",
                "platform_id": metadata.get("platform_id", ""),
                "target_id": target_id,
                "munition": metadata.get("munition", ""),
                "rank": metadata.get("rank", ""),
                "source": metadata.get(
                    "attack_region_source", metadata.get("source", "perch:frea")
                ),
                "target_range_km": metadata.get("target_range_km", ""),
                "agl_m": metadata.get("agl_m", ""),
                "g_violation": metadata.get("g_violation", ""),
                "optimiser": metadata.get("optimiser", ""),
                "situation": _text(audit["situations"].get(target_id, "")),
                "knowledge_sources": _text(
                    audit["knowledge_sources"].get(target_id, [])
                ),
            }
            handle.write("\t".join(
                _text(row[field]).replace("\t", " ").replace(
                    "\r", ""
                ).replace("\n", "\\n")
                for field in OUTPUT_FIELDS
            ) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output")
    args = parser.parse_args()
    try:
        request = _load_request(Path(args.input))
        positions, audit = select_positions(request)
        _write_positions(Path(args.output), positions, audit)
        if args.audit_output:
            Path(args.audit_output).write_text(
                json.dumps(audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps({
            "success": True,
            "position_count": len(positions),
            "output": str(Path(args.output).resolve()),
        }, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
