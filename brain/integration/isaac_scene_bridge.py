"""Convert an AirCombatSceneState into Brain planning inputs."""

from __future__ import annotations

from typing import Any

from brain.domain.agent import AgentSpec


_TARGET_TYPE = {
    "radar": "RADAR",
    "command_post": "CP",
    "armor": "AV",
    "vehicle": "AV",
}

_SENSOR_TYPE = {
    "eo_ir": "EO",
    "sar": "SAR",
    "elint": "ESM",
    "esm": "ESM",
}


def sync_context_from_air_combat_scene(context: Any, scene: Any) -> dict:
    """Update world state and agent inputs from the current Isaac scene.

    Both MILP kilometre coordinates and exact Isaac scene coordinates are
    retained.  The former feed allocation; the latter feed Perch and MPPI.
    """
    world = getattr(context, "world_state", None)
    if world is None:
        raise TypeError("context must expose a mutable world_state dict")

    meters_per_unit = float(scene.meters_per_unit)
    map_size_units = float(scene.map_size_units)
    map_size_km = map_size_units * meters_per_unit / 1000.0
    half_km = map_size_km * 0.5

    def scene_xy_to_km(position) -> list[float]:
        return [
            float(position[0]) * meters_per_unit / 1000.0 + half_km,
            float(position[1]) * meters_per_unit / 1000.0 + half_km,
        ]

    world.update({
        "coordinate_frame": "isaac_scene",
        "map_size_units": map_size_units,
        "map_size_km": map_size_km,
        "meters_per_unit": meters_per_unit,
        "terrain_visual_height_units": float(
            scene.terrain_visual_height_units
        ),
        "terrain_fn": scene.surface_height,
        "tactical_time_s": float(scene.tactical_time_s),
    })
    mission_cfg = scene.config.get("mission", {})
    world.setdefault("aoi", dict(
        mission_cfg.get("aoi", {"row": 3, "col": 4})
    ))
    weather_level = (
        0.35
        if bool(scene.config.get("weather", {}).get("rain", False))
        else 0.0
    )
    world.setdefault(
        "weather", {f"c{index}": weather_level for index in range(5)}
    )
    world.setdefault(
        "terrain", {f"c{index}": 0 for index in range(5)}
    )

    old_agents = {
        str(agent.pid): agent for agent in getattr(context, "agents", [])
    }
    platform_states: list[dict[str, Any]] = []
    platform_weapon_envelopes: dict[str, dict[str, dict[str, Any]]] = {}
    agents: list[AgentSpec] = []
    staging_points: list[list[float]] = []

    for platform in scene.platforms:
        if str(getattr(platform.spec, "faction", "Blue")) != "Blue":
            continue
        platform_id = str(platform.entity_id)
        position = [float(value) for value in platform.position.tolist()]
        velocity = [float(value) for value in platform.velocity.tolist()]
        position_km = scene_xy_to_km(position)
        staging_points.append(position_km)
        platform_states.append({
            "platform_id": platform_id,
            "position_scene": position,
            "velocity_scene": velocity,
            "yaw_deg": float(platform.motion_model.state.yaw_deg),
            "position_km": position_km,
        })

        role = str(platform.spec.role).lower()
        platform_type = (
            "HELI" if "helicopter" in role else "UAV"
        )
        sensors: list[str] = []
        for sensor in platform.spec.sensors:
            mapped = _SENSOR_TYPE.get(str(sensor.channel).lower())
            if mapped and mapped not in sensors:
                sensors.append(mapped)
        munitions: dict[str, int] = {}
        ground_missiles = [
            missile for missile in platform.spec.weapons.missiles
            if "air-to-air" not in str(missile.role).lower()
        ]
        if ground_missiles:
            munitions["HF"] = sum(
                int(missile.count) for missile in ground_missiles
            )
        if platform.spec.weapons.guns:
            munitions["GUN"] = sum(
                int(gun.ammo_rounds)
                for gun in platform.spec.weapons.guns
            )
        if platform_type == "HELI":
            munitions.setdefault("RKT", 0)

        previous = old_agents.get(platform_id)
        agents.append(AgentSpec(
            pid=platform_id,
            type=platform_type,
            position=(position_km[0], position_km[1]),
            sensors=sensors,
            munitions=munitions,
            altitude_km=float(position[2]) * meters_per_unit / 1000.0,
            lost=bool(
                getattr(previous, "lost", False)
                or getattr(platform, "destroyed", False)
            ),
        ))

        envelopes: dict[str, dict[str, Any]] = {}
        if ground_missiles:
            missile = ground_missiles[0]
            guidance = tuple(str(item) for item in missile.guidance)
            envelopes["HF"] = {
                "name": str(missile.name),
                "min_range_km": float(missile.max_range_km) * 0.20,
                "max_range_km": float(missile.max_range_km),
                "optimal_range_km": float(missile.max_range_km) * 0.65,
                "guidance": guidance,
                "requires_designator": bool(
                    guidance == ("semi_active_laser",)
                    and not missile.fire_and_forget
                ),
            }
        if platform.spec.weapons.guns:
            gun = platform.spec.weapons.guns[0]
            envelopes["GUN"] = {
                "name": str(gun.name),
                "min_range_km": min(
                    0.2, float(gun.effective_range_km) * 0.20
                ),
                "max_range_km": float(gun.effective_range_km),
                "optimal_range_km": float(gun.effective_range_km) * 0.60,
                "guidance": ("line_of_sight",),
                "requires_designator": False,
            }
        platform_weapon_envelopes[platform_id] = envelopes

    context.agents = agents
    world["platform_states"] = platform_states
    world["platform_weapon_envelopes"] = platform_weapon_envelopes
    if staging_points and "staging_position" not in world:
        world["staging_position"] = [
            sum(point[axis] for point in staging_points)
            / len(staging_points)
            for axis in range(2)
        ]

    detected_ids = {
        str(contact["target_id"]) for contact in scene.contacts
    }
    old_targets = {
        str(item.get("tid", item.get("target_id", ""))): item
        for item in world.get("targets", [])
    }
    targets: list[dict[str, Any]] = []
    for target in scene.targets:
        if str(getattr(target.spec, "faction", "Red")) == "Blue":
            continue
        target_id = str(target.target_id)
        previous = old_targets.get(target_id, {})
        position = [float(value) for value in target.position.tolist()]
        category = str(target.spec.category)
        targets.append({
            **previous,
            "tid": target_id,
            "target_id": target_id,
            "type": _TARGET_TYPE.get(category, "AV"),
            "category": category,
            "pos": scene_xy_to_km(position),
            "position_scene": position,
            "is_fixed": bool(target.spec.is_fixed),
            "is_radiating": bool(target.spec.is_radiating),
            "mobility_speed_mps": float(target.spec.mobility_speed_mps),
            "value": float(target.spec.priority) / 100.0,
            "threat": float(target.spec.priority) / 100.0,
            "confirmed": bool(
                previous.get("confirmed", False)
                or target_id in detected_ids
            ),
            "alive": not bool(target.destroyed),
        })
    world["targets"] = targets

    obstacles: list[dict[str, Any]] = []
    for obstacle in scene.obstacles:
        position = [
            float(value) for value in obstacle.position.tolist()
        ]
        base_z = float(scene.surface_height(
            float(position[0]), float(position[1])
        ))
        if str(obstacle.category) == "mountain":
            top_z = max(
                base_z,
                float(position[2]),
                float(obstacle.height_units),
            )
        else:
            top_z = max(
                float(position[2]),
                base_z + float(obstacle.height_units),
            )
        obstacles.append({
            "obstacle_id": str(obstacle.obstacle_id),
            "category": str(obstacle.category),
            "position_scene": position,
            "radius_units": float(obstacle.radius_units),
            "base_z": base_z,
            "top_z": top_z,
            "height_units": float(obstacle.height_units),
            "priority": int(obstacle.priority),
            "blocks_los": bool(obstacle.blocks_los),
        })
    world["obstacles"] = obstacles
    world["obstacle_contacts"] = [
        dict(contact) for contact in scene.obstacle_contacts
    ]
    return world
