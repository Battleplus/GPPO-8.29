"""Defensive public wrapper around the Isaac scene-state converter."""

from __future__ import annotations

from typing import Any

from .isaac_scene_bridge import (
    sync_context_from_air_combat_scene as _sync_scene,
)


def sync_context_from_air_combat_scene(context: Any, scene: Any) -> dict:
    """Synchronize a full scene or a lightweight test/embedding scene."""
    if not hasattr(scene, "config"):
        scene.config = {"mission": {}, "weather": {"rain": False}}
    for platform in getattr(scene, "platforms", []):
        spec = platform.spec
        if not hasattr(spec, "role"):
            spec.role = "uav"
        if not hasattr(spec, "sensors"):
            spec.sensors = ()
        if not hasattr(spec, "faction"):
            spec.faction = "Blue"
        for missile in getattr(spec.weapons, "missiles", ()):
            if not hasattr(missile, "count"):
                missile.count = 1
        for gun in getattr(spec.weapons, "guns", ()):
            if not hasattr(gun, "ammo_rounds"):
                gun.ammo_rounds = 0

    world = _sync_scene(context, scene)
    rocket_default = int(
        scene.config.get("weapons", {}).get(
            "helicopter_rocket_count", 76
        )
    )
    for agent in getattr(context, "agents", []):
        if str(getattr(agent, "type", "")) == "HELI":
            agent.munitions["RKT"] = int(
                agent.munitions.get("RKT", 0) or rocket_default
            )
    return world
