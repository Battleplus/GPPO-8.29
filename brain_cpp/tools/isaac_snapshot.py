#!/usr/bin/env python3
"""Initialize the repository Isaac environment and print a C++ snapshot.

This helper is intentionally tiny.  The actual Isaac setup is delegated to the
existing Python runtime:

    brain.integration.IsaacAirCombatEnvironment
    brain.integration.sync_context_from_air_combat_scene
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from brain.integration import (  # noqa: E402
    IsaacAirCombatEnvironment,
    apply_task_area_overrides,
    load_mission_world,
    sync_context_from_air_combat_scene,
)


@dataclass
class SnapshotContext:
    """Minimal context required by the Isaac scene synchronization bridge.

    Importing ``brain.core`` before ``SimulationApp`` also imports the full
    planning stack and native solver dependencies.  Isaac's SimulationManager
    must own USD initialization before those modules are loaded.
    """

    agents: list[Any] = field(default_factory=list)
    world_state: dict[str, Any] = field(default_factory=dict)


def _join_map(values: dict[str, Any]) -> str:
    return ";".join(f"{key}={value}" for key, value in values.items())


def _join_list(values: list[Any]) -> str:
    return ";".join(str(value) for value in values)


def _truth(value: Any) -> str:
    return "1" if bool(value) else "0"


def _aoi_row_col(aoi_id: str) -> tuple[int, int]:
    parts = str(aoi_id).replace("-", "_").split("_")
    if len(parts) >= 3 and parts[0].upper() == "A":
        return int(parts[1]), int(parts[2])
    return 3, 4


def emit_snapshot(context: Any, scene: Any) -> None:
    world = context.world_state
    print("snapshot|begin")
    print("name|IsaacAirCombatEnvironment")
    print("initialized|1")
    print(f"time|{float(world.get('tactical_time_s', 0.0))}")

    staging = world.get("staging_position", [150.0, -50.0])
    print(f"staging|{float(staging[0])}|{float(staging[1])}")

    aois = world.get("aois") or world.get("task_areas") or []
    if not aois:
        commander = world.get("commander_AOI") or []
        aois = [{"id": item, "row": _aoi_row_col(item)[0], "col": _aoi_row_col(item)[1]} for item in commander]
    if not aois:
        aoi = world.get("aoi", {"row": 3, "col": 4})
        aois = [{"id": f"A_{int(aoi.get('row', 3))}_{int(aoi.get('col', 4))}", **aoi}]
    for item in aois:
        aoi_id = str(item.get("id") or f"A_{int(item.get('row', 3))}_{int(item.get('col', 4))}")
        print(f"aoi|{aoi_id}|{int(item.get('row', 3))}|{int(item.get('col', 4))}")

    for agent in context.agents:
        sensors = _join_list(list(getattr(agent, "sensors", []) or []))
        munition_values = dict(getattr(agent, "munitions", {}) or {})
        if not munition_values:
            munition_values = {"HF": 0, "RKT": 0, "GUN": 0}
        munitions = _join_map(munition_values)
        print(
            "agent|"
            f"{agent.pid}|{agent.type}|"
            f"{float(agent.position[0])}|{float(agent.position[1])}|"
            f"{float(agent.altitude_km)}|{_truth(getattr(agent, 'lost', False))}|"
            f"{sensors}|{munitions}"
        )

    for target in world.get("targets", []):
        tid = str(target.get("tid") or target.get("target_id") or "")
        pos = target.get("pos", [0.0, 0.0])
        print(
            "target|"
            f"{tid}|{target.get('type', 'AV')}|"
            f"{float(pos[0])}|{float(pos[1])}|"
            f"{float(target.get('value', 0.5))}|"
            f"{float(target.get('threat', 0.5))}|"
            f"{_truth(target.get('confirmed', False))}|"
            f"{_truth(target.get('alive', True))}"
        )

    for contact in getattr(scene, "contacts", []):
        print(
            "contact|"
            f"{contact.get('platform_id', '')}|{contact.get('target_id', '')}|"
            f"{contact.get('sensor', '')}|{contact.get('channel', '')}|"
            f"{float(contact.get('distance_km', 0.0))}|"
            f"{int(contact.get('priority', 0))}"
        )

    print(f"weather|{_join_map(world.get('weather', {}))}")
    print(f"terrain|{_join_map(world.get('terrain', {}))}")
    print("snapshot|end")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", choices=["initialize", "reset", "step"], default="initialize")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dt", type=float, default=0.0)
    parser.add_argument("--mission-input", default=None)
    parser.add_argument("--aoi", default=None)
    parser.add_argument("--aois", default=None)
    args = parser.parse_args()

    mission_world = load_mission_world(args.mission_input)
    apply_task_area_overrides(mission_world, aoi=args.aoi, aois=args.aois)

    scene_config = dict(
        mission_world.get("scene_config")
        or mission_world.get("isaac_scene_config")
        or {}
    )
    if mission_world.get("aoi"):
        scene_config.setdefault("mission", {})["aoi"] = dict(mission_world["aoi"])

    runtime = IsaacAirCombatEnvironment(scene_config=scene_config, headless=args.headless)
    context = SnapshotContext(
        world_state={"environment_type": "isaac_air_combat", **mission_world}
    )
    try:
        if args.command == "reset":
            scene = runtime.reset()
        elif args.command == "step":
            scene = runtime.step(args.dt)
        else:
            scene = runtime.initialize()
        sync_context_from_air_combat_scene(context, scene)
        emit_snapshot(context, scene)
        sys.stdout.flush()
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
