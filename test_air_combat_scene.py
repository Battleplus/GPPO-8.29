from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "renderer": "HydraStorm",
        "width": 1280,
        "height": 720,
    }
)

import os
import math
import time

import omni.timeline
import omni.usd
from pxr import Usd

from scenes import air_combat_scene


RUN_SECONDS = float(os.environ.get("QL_AIR_COMBAT_RUN_SECONDS", "120.0"))
DT = 1.0 / 30.0
DEFAULT_CFG = air_combat_scene.DEFAULT_AIR_COMBAT_CONFIG
MISSILE_FLIGHT_TIME_S = float(os.environ.get("QL_AIR_COMBAT_MISSILE_FLIGHT_S", "120.0"))
OPENING_MISSILE_DEMO = os.environ.get("QL_AIR_COMBAT_OPENING_MISSILE_DEMO", "1").strip().lower() in {"1", "true", "yes"}
OPENING_MISSILE_FLIGHT_TIME_S = float(os.environ.get("QL_AIR_COMBAT_OPENING_MISSILE_FLIGHT_S", str(MISSILE_FLIGHT_TIME_S * 2.5)))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARN] Invalid {name}={raw!r}; using {default}")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[WARN] Invalid {name}={raw!r}; using {default}")
        return default


def _sensor_value(sensor, name: str, default: float) -> float:
    try:
        return float(getattr(sensor, name))
    except (AttributeError, TypeError, ValueError):
        return float(default)


def build_config() -> dict:
    default_map = DEFAULT_CFG["map"]
    default_visual = DEFAULT_CFG["visual"]
    default_weather = DEFAULT_CFG["weather"]
    default_sim = DEFAULT_CFG["simulation"]
    default_aircraft = DEFAULT_CFG["aircraft"]
    default_targets = DEFAULT_CFG["targets"]
    default_obstacles = DEFAULT_CFG["obstacles"]
    return {
        "map": {
            "real_size_km": _env_float("QL_AIR_COMBAT_MAP_KM", float(default_map["real_size_km"])),
            "meters_per_unit": _env_float("QL_AIR_COMBAT_METERS_PER_UNIT", float(default_map["meters_per_unit"])),
            "mountain_height_m": _env_float("QL_AIR_COMBAT_MOUNTAIN_M", float(default_map["mountain_height_m"])),
            "terrain_vertical_exaggeration": _env_float("QL_AIR_COMBAT_TERRAIN_EXAGGERATION", float(default_map["terrain_vertical_exaggeration"])),
            "terrain_usd": os.environ.get("QL_AIR_COMBAT_TERRAIN_USD"),
            "grid": _env_int("QL_AIR_COMBAT_GRID", int(default_map["grid"])),
        },
        "visual": {
            "aircraft_scale": _env_float("QL_AIR_COMBAT_AIRCRAFT_SCALE", float(default_visual["aircraft_scale"])),
            "missile_visual_scale": _env_float("QL_AIR_COMBAT_MISSILE_SCALE", float(default_visual["missile_visual_scale"])),
            "armor_visual_scale": _env_float("QL_AIR_COMBAT_ARMOR_SCALE", float(default_visual["armor_visual_scale"])),
            "base_visual_scale": _env_float("QL_AIR_COMBAT_BASE_VISUAL_SCALE", float(default_visual["base_visual_scale"])),
            "show_sensor_rings": os.environ.get("QL_AIR_COMBAT_SHOW_SENSOR_RINGS", "0").strip().lower() in {"1", "true", "yes"},
            "show_sensor_cones": os.environ.get("QL_AIR_COMBAT_SHOW_SENSOR_CONES", "0").strip().lower() in {"1", "true", "yes"},
            "show_terrain_grid": os.environ.get("QL_AIR_COMBAT_SHOW_TERRAIN_GRID", "0").strip().lower() in {"1", "true", "yes"},
            "show_terrain_wireframe": os.environ.get("QL_AIR_COMBAT_SHOW_TERRAIN_WIREFRAME", "0").strip().lower() in {"1", "true", "yes"},
            "show_contours": os.environ.get("QL_AIR_COMBAT_SHOW_CONTOURS", "0").strip().lower() in {"1", "true", "yes"},
            "show_route_lines": os.environ.get("QL_AIR_COMBAT_SHOW_ROUTE_LINES", "0").strip().lower() in {"1", "true", "yes"},
            "show_collision_body": os.environ.get("QL_AIR_COMBAT_SHOW_COLLISION_BODY", "0").strip().lower() in {"1", "true", "yes"},
        },
        "weather": {
            "clouds": os.environ.get("QL_AIR_COMBAT_CLOUDS", "1" if default_weather["clouds"] else "0").strip().lower() in {"1", "true", "yes"},
            "cloud_opacity": _env_float("QL_AIR_COMBAT_CLOUD_OPACITY", float(default_weather["cloud_opacity"])),
            "cloud_count": _env_int("QL_AIR_COMBAT_CLOUD_COUNT", int(default_weather["cloud_count"])),
            "cloud_altitude_m": _env_float("QL_AIR_COMBAT_CLOUD_ALTITUDE_M", float(default_weather["cloud_altitude_m"])),
            "local_zones": os.environ.get("QL_AIR_COMBAT_LOCAL_WEATHER", "1" if default_weather["local_zones"] else "0").strip().lower() in {"1", "true", "yes"},
            "rain": os.environ.get("QL_AIR_COMBAT_RAIN", "1" if default_weather["rain"] else "0").strip().lower() in {"1", "true", "yes"},
            "rain_count": _env_int("QL_AIR_COMBAT_RAIN_COUNT", int(default_weather["rain_count"])),
            "rain_opacity": _env_float("QL_AIR_COMBAT_RAIN_OPACITY", float(default_weather["rain_opacity"])),
        },
        "bases": {
            "enabled": os.environ.get("QL_AIR_COMBAT_BASES", "1").strip().lower() not in {"0", "false", "no"},
        },
        "simulation": {
            "time_scale": _env_float("QL_AIR_COMBAT_TIME_SCALE", float(default_sim["time_scale"])),
            "aircraft_model": os.environ.get("QL_AIR_COMBAT_AIRCRAFT_MODEL", str(default_sim["aircraft_model"])),
        },
        "aircraft": {
            "terrain_clearance_m": _env_float("QL_AIR_COMBAT_TERRAIN_CLEARANCE_M", float(default_aircraft["terrain_clearance_m"])),
        },
        "targets": {
            "radar_sites": _env_int("QL_AIR_COMBAT_RADARS", int(default_targets["radar_sites"])),
            "command_posts": _env_int("QL_AIR_COMBAT_COMMAND_POSTS", int(default_targets["command_posts"])),
            "armored_vehicles": _env_int("QL_AIR_COMBAT_ARMOR", int(default_targets["armored_vehicles"])),
            "forward_bases": os.environ.get("QL_AIR_COMBAT_BASE_TARGETS", "0").strip().lower() in {"1", "true", "yes"},
        },
        "obstacles": {
            "enabled": os.environ.get("QL_AIR_COMBAT_OBSTACLES", "1" if default_obstacles["enabled"] else "0").strip().lower() in {"1", "true", "yes"},
            "mountains": os.environ.get("QL_AIR_COMBAT_OBSTACLE_MOUNTAINS", "1" if default_obstacles["mountains"] else "0").strip().lower() in {"1", "true", "yes"},
            "forests": os.environ.get("QL_AIR_COMBAT_OBSTACLE_FORESTS", "1" if default_obstacles["forests"] else "0").strip().lower() in {"1", "true", "yes"},
            "rock_fields": os.environ.get("QL_AIR_COMBAT_OBSTACLE_ROCKS", "1" if default_obstacles["rock_fields"] else "0").strip().lower() in {"1", "true", "yes"},
            "forest_tree_count": _env_int("QL_AIR_COMBAT_FOREST_TREES", int(default_obstacles["forest_tree_count"])),
            "forest_tree_visual_scale": _env_float("QL_AIR_COMBAT_FOREST_TREE_SCALE", float(default_obstacles["forest_tree_visual_scale"])),
            "max_contacts_per_platform": _env_int("QL_AIR_COMBAT_OBSTACLE_CONTACTS", int(default_obstacles["max_contacts_per_platform"])),
        },
    }


def print_scene_summary(scene: air_combat_scene.AirCombatSceneState) -> None:
    print(
        f"[TEST] Scene scale: {scene.map_size_km:.0f} km x {scene.map_size_km:.0f} km, "
        f"{scene.meters_per_unit:.0f} m/unit, mountain reference {scene.mountain_height_units * scene.meters_per_unit:.0f} m, "
        f"visual peak {scene.terrain_visual_height_units:.1f} scene units"
    )
    print("[TEST] Platforms:")
    for platform in scene.platforms:
        sensor_text = ", ".join(
            f"{sensor.channel}:{sensor.max_range_km:.0f}km/"
            f"{_sensor_value(sensor, 'azimuth_fov_deg', 360.0):.0f}deg/"
            f"{_sensor_value(sensor, 'scan_rate_hz', 1.0):.1f}Hz"
            for sensor in platform.spec.sensors
        )
        weapon_text = ", ".join(missile.name for missile in platform.spec.weapons.missiles) or "unarmed"
        print(f"  - {platform.entity_id}: {platform.spec.faction}, {platform.spec.name}, {platform.spec.role}, sensors[{sensor_text}], weapons[{weapon_text}]")
    print("[TEST] Targets:")
    for target in scene.targets:
        mobility = "fixed" if target.spec.is_fixed else "mobile"
        emitter = "radiating" if target.spec.is_radiating else "non-radiating"
        state = "destroyed" if getattr(target, "destroyed", False) else "active"
        print(f"  - {target.target_id}: {target.spec.name}, {target.spec.category}, {mobility}, {emitter}, {state}, priority={target.spec.priority}")
    print("[TEST] Environment obstacles:")
    for obstacle in scene.obstacles[:24]:
        radius_km = obstacle.radius_units * scene.meters_per_unit / 1000.0
        height_m = obstacle.height_units * scene.meters_per_unit
        pos_m = obstacle.position * scene.meters_per_unit
        print(
            f"  - {obstacle.obstacle_id}: {obstacle.category}, "
            f"pos_m=({pos_m[0]:.0f},{pos_m[1]:.0f},{pos_m[2]:.0f}), "
            f"radius={radius_km:.2f}km, height={height_m:.0f}m, priority={obstacle.priority}"
        )
    if len(scene.obstacles) > 24:
        print(f"  ... {len(scene.obstacles) - 24} more obstacles")


def print_tactical_status(scene: air_combat_scene.AirCombatSceneState, frame_count: int) -> None:
    destroyed_count = sum(1 for target in scene.targets if getattr(target, "destroyed", False))
    print(
        f"[STATUS] tactical_time={scene.tactical_time_s:7.1f}s frame={frame_count} "
        f"contacts={len(scene.contacts)} obstacles={len(scene.obstacle_contacts)} assignments={len(scene.assignments)} destroyed={destroyed_count}"
    )
    for platform in scene.platforms:
        pos = platform.position
        print(
            f"  {platform.entity_id:<20} pos=({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:5.1f}) "
            f"speed={scene.meters_per_unit * (platform.velocity @ platform.velocity) ** 0.5:5.1f}m/s"
        )
    for contact in scene.contacts[:8]:
        print(
            f"  contact {contact['platform_id']} -> {contact['target_id']} "
            f"via {contact['channel']} range={contact['distance_km']:.1f}km priority={contact['priority']}"
        )
    for contact in scene.obstacle_contacts[:8]:
        pos_m = contact.get("position_m", [0.0, 0.0, 0.0])
        print(
            f"  obstacle {contact['platform_id']} -> {contact['obstacle_id']} "
            f"type={contact['obstacle_category']} via {contact['channel']} "
            f"range={contact['distance_km']:.1f}km pos_m=({pos_m[0]:.0f},{pos_m[1]:.0f},{pos_m[2]:.0f})"
        )
    for assignment in scene.assignments[:6]:
        weapon = assignment["weapon"] or "none"
        print(
            f"  task {assignment['task']} target={assignment['target_id']} detected_by={assignment['detected_by']} "
            f"assigned_to={assignment['assigned_to']} weapon={weapon}"
        )
    print_uav_sensor_checks(scene)
    print_obstacle_sensor_checks(scene)


def print_uav_sensor_checks(scene: air_combat_scene.AirCombatSceneState) -> None:
    uav_ids = [platform.entity_id for platform in scene.platforms if "uav" in platform.spec.role]
    if not uav_ids:
        return
    print("  UAV sensor checks:")
    for platform_id in uav_ids:
        contacts = scene.contacts_for(platform_id)
        by_category: dict[str, int] = {}
        for contact in contacts:
            by_category[contact["target_category"]] = by_category.get(contact["target_category"], 0) + 1
        summary = ", ".join(f"{category}:{count}" for category, count in sorted(by_category.items())) or "none"
        first_hits = ", ".join(f"{contact['target_id']}@{contact['distance_km']:.1f}km/{contact['channel']}" for contact in contacts[:3])
        print(f"    {platform_id:<24} contacts[{summary}] first[{first_hits or 'none'}]")


def print_obstacle_sensor_checks(scene: air_combat_scene.AirCombatSceneState) -> None:
    platform_ids = [platform.entity_id for platform in scene.platforms]
    if not platform_ids:
        return
    print("  Obstacle sensor checks:")
    for platform_id in platform_ids:
        contacts = scene.obstacle_contacts_for(platform_id)
        by_category: dict[str, int] = {}
        for contact in contacts:
            category = contact["obstacle_category"]
            by_category[category] = by_category.get(category, 0) + 1
        summary = ", ".join(f"{category}:{count}" for category, count in sorted(by_category.items())) or "none"
        first_hits = ", ".join(
            f"{contact['obstacle_id']}@{contact['distance_km']:.1f}km/{contact['channel']}/"
            f"({contact.get('position_m', [0.0, 0.0, 0.0])[0]:.0f},{contact.get('position_m', [0.0, 0.0, 0.0])[1]:.0f})m"
            for contact in contacts[:3]
        )
        print(f"    {platform_id:<24} obstacles[{summary}] first[{first_hits or 'none'}]")


class MissionDemoController:
    def __init__(self, stage) -> None:
        self.stage = stage
        self.phase = "recon"
        self.reported_contacts: set[tuple[str, str]] = set()
        self.launch_times: dict[str, float] = {}
        self.impacted_targets: set[str] = set()
        self.selected_assignments: list[dict] = []
        self.missile_visuals: dict[str, dict] = {}
        self.last_phase_print_s = -999.0
        self.last_wait_print_s = -999.0
        self.opening_demo_started = False
        self.force_armor_strike = os.environ.get("QL_AIR_COMBAT_FORCE_ARMOR_STRIKE", "0").strip().lower() in {"1", "true", "yes"}

    def update(self, scene: air_combat_scene.AirCombatSceneState) -> None:
        recon_contacts = self._recon_contacts(scene)
        if scene.tactical_time_s - self.last_phase_print_s >= 45.0:
            self.last_phase_print_s = scene.tactical_time_s
            print(
                f"[MISSION] phase={self.phase} tactical_time={scene.tactical_time_s:.1f}s "
                f"recon_contacts={len(recon_contacts)} strike_tasks={len(self.selected_assignments) or len(scene.assignments)}"
            )

        self._print_new_recon_reports(recon_contacts)
        if OPENING_MISSILE_DEMO and not self.opening_demo_started and scene.tactical_time_s >= 3.0:
            self._launch_opening_missile_demo(scene)
        if self.phase == "recon" and self.force_armor_strike and scene.tactical_time_s >= 30.0:
            forced = self._forced_armor_assignment(scene)
            if forced is not None:
                self.selected_assignments = [forced]
                self.phase = "dispatch"
                print(
                    f"[MISSION] force_demo target={forced['target_id']} assigned_to={forced['assigned_to']} "
                    f"weapon={forced['weapon']}"
                )

        enough_recon = len({contact["target_id"] for contact in recon_contacts}) >= 4
        fallback_picture = scene.tactical_time_s >= 150.0 and any(assignment.get("weapon") for assignment in scene.assignments)
        if self.phase == "recon" and (enough_recon or fallback_picture):
            self.selected_assignments = self._select_assignments(scene, recon_contacts)
            if not self.selected_assignments:
                if scene.tactical_time_s - self.last_wait_print_s >= 45.0:
                    self.last_wait_print_s = scene.tactical_time_s
                    print("[MISSION] Recon picture built. Waiting for strike aircraft to enter weapon release range.")
                return
            self.phase = "dispatch"
            print(f"[MISSION] Recon picture built. Dispatching {len(self.selected_assignments)} strike aircraft/tasks.")
            for assignment in self.selected_assignments[:8]:
                print(
                    f"[MISSION] task target={assignment['target_id']} priority={assignment['priority']} "
                    f"detected_by={assignment['detected_by']} assigned_to={assignment['assigned_to']} weapon={assignment['weapon']}"
                )

        if self.phase == "dispatch":
            self._launch_available_weapons(scene)
            if any(assignment["target_id"] in self.launch_times for assignment in self.selected_assignments):
                self.phase = "engagement"

        if self.phase == "engagement":
            self._update_missile_visuals(scene)
            self._resolve_impacts(scene)
            if self.selected_assignments and len(self.impacted_targets) >= min(6, len(self.selected_assignments)):
                self.phase = "assessment"
                print(f"[MISSION] Battle damage assessment complete. confirmed_hits={len(self.impacted_targets)}")
        if OPENING_MISSILE_DEMO and self.phase != "engagement":
            self._update_missile_visuals(scene)

    def _recon_contacts(self, scene: air_combat_scene.AirCombatSceneState) -> list[dict]:
        recon_ids = {
            platform.entity_id
            for platform in scene.platforms
            if "recon" in platform.spec.role and "strike" not in platform.spec.role
        }
        return [contact for contact in scene.contacts if contact["platform_id"] in recon_ids]

    def _print_new_recon_reports(self, recon_contacts: list[dict]) -> None:
        for contact in recon_contacts[:16]:
            key = (contact["platform_id"], contact["target_id"])
            if key in self.reported_contacts:
                continue
            self.reported_contacts.add(key)
            print(
                f"[MISSION] recon_report observer={contact['platform_id']} target={contact['target_id']} "
                f"type={contact['target_category']} sensor={contact['channel']} range={contact['distance_km']:.1f}km"
            )

    def _select_assignments(self, scene: air_combat_scene.AirCombatSceneState, recon_contacts: list[dict]) -> list[dict]:
        recon_target_ids = {contact["target_id"] for contact in recon_contacts}
        selected = [
            assignment
            for assignment in scene.assignments
            if assignment.get("task") == "strike"
            and assignment.get("weapon")
            and assignment["target_id"] in recon_target_ids
        ]
        if not selected:
            selected = [
                assignment
                for assignment in scene.assignments
                if assignment.get("task") == "strike" and assignment.get("weapon")
            ]
        selected.sort(key=lambda item: (0 if str(item["target_id"]).startswith("Armor_") else 1, -int(item["priority"]), str(item["target_id"])))
        return selected[:8]

    def _forced_armor_assignment(self, scene: air_combat_scene.AirCombatSceneState) -> dict | None:
        target = next((item for item in scene.targets if item.spec.category == "armor" and not item.destroyed), None)
        shooter = next((item for item in scene.platforms if item.spec.weapons.missiles), None)
        if target is None or shooter is None:
            return None
        missile = shooter.spec.weapons.missiles[0]
        return {
            "target_id": target.target_id,
            "target": target.spec.name,
            "task": "strike",
            "detected_by": "force_demo",
            "sensor": "scripted",
            "assigned_to": shooter.entity_id,
            "weapon": missile.name,
            "weapon_margin_km": missile.max_range_km,
            "priority": target.spec.priority,
        }

    def _launch_opening_missile_demo(self, scene: air_combat_scene.AirCombatSceneState) -> None:
        shooter = next((item for item in scene.platforms if item.spec.weapons.missiles), None)
        if shooter is None:
            return
        shooter_pos = shooter.position
        start = [float(shooter_pos[0]), float(shooter_pos[1]), float(shooter_pos[2] + 6.0)]
        yaw = math.radians(float(shooter.motion_model.state.yaw_deg))
        forward = [math.cos(yaw), math.sin(yaw), 0.0]
        target = [
            float(start[0] + forward[0] * 180.0),
            float(start[1] + forward[1] * 180.0),
            float(start[2] + 18.0),
        ]
        self.opening_demo_started = True
        self.launch_times["OpeningMissileVisual"] = scene.tactical_time_s
        visual = _create_missile_trail(
            self.stage,
            "/World/AirCombat/MissionDemo/OpeningMissileVisual",
            start,
            target,
            scene,
        )
        visual["flight_time_s"] = OPENING_MISSILE_FLIGHT_TIME_S
        self.missile_visuals["OpeningMissileVisual"] = visual
        print(f"[MISSION] opening_missile_visual shooter={shooter.entity_id} no_target=true")

    def _launch_available_weapons(self, scene: air_combat_scene.AirCombatSceneState) -> None:
        for assignment in self.selected_assignments:
            target_id = assignment["target_id"]
            if target_id in self.launch_times:
                continue
            target = _target_by_id(scene, target_id)
            shooter = _platform_by_id(scene, assignment["assigned_to"])
            if target is None:
                continue
            self.launch_times[target_id] = scene.tactical_time_s
            _create_demo_marker(
                self.stage,
                f"/World/AirCombat/MissionDemo/Launch_{target_id}",
                target.position,
                (1.0, 0.72, 0.08),
                scene,
                lift_m=260.0,
                radius_m=180.0,
            )
            if shooter is not None:
                self.missile_visuals[target_id] = _create_missile_trail(
                    self.stage,
                    f"/World/AirCombat/MissionDemo/MissileTrail_{target_id}",
                    shooter.position,
                    target.position,
                    scene,
                )
            print(
                f"[MISSION] launch shooter={assignment['assigned_to']} weapon={assignment['weapon']} "
                f"target={target_id} margin={assignment.get('weapon_margin_km') or 0.0:.1f}km"
            )

    def _resolve_impacts(self, scene: air_combat_scene.AirCombatSceneState) -> None:
        for assignment in self.selected_assignments:
            target_id = assignment["target_id"]
            launch_time = self.launch_times.get(target_id)
            if launch_time is None or target_id in self.impacted_targets:
                continue
            if scene.tactical_time_s - launch_time < MISSILE_FLIGHT_TIME_S:
                continue
            target = _target_by_id(scene, target_id)
            if target is None:
                continue
            self.impacted_targets.add(target_id)
            destroyed = scene.mark_target_destroyed(target_id, cause=str(assignment.get("weapon") or "weapon"))
            _create_demo_marker(
                self.stage,
                f"/World/AirCombat/MissionDemo/Impact_{target_id}",
                target.position,
                (1.0, 0.18, 0.04),
                scene,
                lift_m=120.0,
                radius_m=260.0,
            )
            result = "destroyed" if destroyed or target.destroyed else "already_destroyed"
            print(f"[MISSION] impact target={target_id} category={target.spec.category} result={result}")

    def _update_missile_visuals(self, scene: air_combat_scene.AirCombatSceneState) -> None:
        for target_id, visual in self.missile_visuals.items():
            launch_time = self.launch_times.get(target_id)
            if launch_time is None or target_id in self.impacted_targets:
                continue
            flight_time_s = max(1.0, float(visual.get("flight_time_s", MISSILE_FLIGHT_TIME_S)))
            phase = max(0.0, min(1.0, (scene.tactical_time_s - launch_time) / flight_time_s))
            _set_missile_visual_phase(visual, phase)


def _target_by_id(scene: air_combat_scene.AirCombatSceneState, target_id: str):
    for target in scene.targets:
        if target.target_id == target_id:
            return target
    return None


def _platform_by_id(scene: air_combat_scene.AirCombatSceneState, platform_id: str):
    for platform in scene.platforms:
        if platform.entity_id == platform_id:
            return platform
    return None


def _create_demo_marker(stage, path: str, pos, color: tuple[float, float, float], scene: air_combat_scene.AirCombatSceneState, lift_m: float, radius_m: float = 70.0) -> None:
    from pxr import Gf, Sdf, UsdGeom

    z = float(pos[2]) + lift_m / scene.meters_per_unit
    radius = max(0.35, radius_m / scene.meters_per_unit)
    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.CreateRadiusAttr(float(radius))
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), z))
    UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayOpacityAttr([0.88])


def _create_missile_trail(stage, path: str, start_pos, target_pos, scene: air_combat_scene.AirCombatSceneState) -> dict:
    from pxr import Gf, Sdf, UsdGeom

    start = [float(start_pos[0]), float(start_pos[1]), float(start_pos[2])]
    end = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2]) + 120.0 / scene.meters_per_unit]
    visual_scale = float(scene.config.get("visual", {}).get("missile_visual_scale", scene.config.get("visual", {}).get("aircraft_scale", 180.0)))
    missile_length = max(2.2, 4.2 * visual_scale / scene.meters_per_unit)
    missile_radius = max(0.09, 0.20 * visual_scale / scene.meters_per_unit)
    flame_radius = max(0.16, 0.46 * visual_scale / scene.meters_per_unit)
    trail_width = max(0.10, 0.26 * visual_scale / scene.meters_per_unit)
    smoke_width = max(0.18, 0.52 * visual_scale / scene.meters_per_unit)
    apex = [
        (start[0] + end[0]) * 0.5,
        (start[1] + end[1]) * 0.5,
        max(start[2], end[2]) + 900.0 / scene.meters_per_unit,
    ]
    points = [Gf.Vec3f(*start), Gf.Vec3f(*apex), Gf.Vec3f(*end)]
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr([len(points)])
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([trail_width, trail_width, trail_width])
    curves.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.72, 0.08)])
    smoke = UsdGeom.BasisCurves.Define(stage, Sdf.Path(f"{path}_Smoke"))
    smoke.CreateTypeAttr("linear")
    smoke.CreateCurveVertexCountsAttr([len(points)])
    smoke.CreatePointsAttr(points)
    smoke.CreateWidthsAttr([smoke_width, smoke_width * 0.78, smoke_width * 0.52])
    smoke.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.72, 0.66)])
    UsdGeom.Gprim(smoke.GetPrim()).CreateDisplayOpacityAttr([0.22])

    mid = apex
    missile = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{path}_MissileBody"))
    missile.CreateRadiusAttr(float(missile_radius))
    missile.CreateHeightAttr(float(missile_length))
    xform = UsdGeom.Xformable(missile.GetPrim())
    missile_translate = xform.AddTranslateOp()
    missile_rotate = xform.AddRotateXYZOp()
    missile_translate.Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))
    missile_rotate.Set(Gf.Vec3f(0.0, 90.0, 0.0))
    UsdGeom.Gprim(missile.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.86, 0.52)])

    flame = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{path}_RocketFlame"))
    flame.CreateRadiusAttr(float(flame_radius))
    flame_xform = UsdGeom.Xformable(flame.GetPrim())
    flame_translate = flame_xform.AddTranslateOp()
    flame_translate.Set(Gf.Vec3d(float(mid[0] - 1.8), float(mid[1]), float(mid[2])))
    UsdGeom.Gprim(flame.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.30, 0.02)])

    _create_impact_ring(stage, f"{path}_PredictedImpactRing", end, scene, (1.0, 0.18, 0.04))
    visual = {
        "missile": missile.GetPrim(),
        "missile_translate": missile_translate,
        "missile_rotate": missile_rotate,
        "flame": flame.GetPrim(),
        "flame_translate": flame_translate,
        "points": (start, apex, end),
        "flame_offset": missile_length * 0.58,
    }
    _set_missile_visual_phase(visual, 0.0)
    return visual


def _set_missile_visual_phase(visual: dict, phase: float) -> None:
    from pxr import Gf, UsdGeom

    start, apex, end = visual["points"]
    phase = max(0.0, min(1.0, float(phase)))
    if phase < 0.5:
        local = phase / 0.5
        pos = [start[i] * (1.0 - local) + apex[i] * local for i in range(3)]
        next_pos = [apex[i] for i in range(3)]
    else:
        local = (phase - 0.5) / 0.5
        pos = [apex[i] * (1.0 - local) + end[i] * local for i in range(3)]
        next_pos = [end[i] for i in range(3)]
    yaw = math.degrees(math.atan2(next_pos[1] - pos[1], next_pos[0] - pos[0]))
    missile_translate = visual.get("missile_translate")
    missile_rotate = visual.get("missile_rotate")
    if missile_translate is None or missile_rotate is None:
        missile = UsdGeom.Xformable(visual["missile"])
        missile_translate = missile.GetTranslateOp() or missile.AddTranslateOp()
        missile_rotate = missile.GetRotateXYZOp() or missile.AddRotateXYZOp()
    missile_translate.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    missile_rotate.Set(Gf.Vec3f(0.0, 90.0, float(yaw)))
    flame_offset = float(visual.get("flame_offset", 1.8))
    flame_pos = [pos[0] - math.cos(math.radians(yaw)) * flame_offset, pos[1] - math.sin(math.radians(yaw)) * flame_offset, pos[2]]
    flame_translate = visual.get("flame_translate")
    if flame_translate is None:
        flame = UsdGeom.Xformable(visual["flame"])
        flame_translate = flame.GetTranslateOp() or flame.AddTranslateOp()
    flame_translate.Set(Gf.Vec3d(float(flame_pos[0]), float(flame_pos[1]), float(flame_pos[2])))


def _create_impact_ring(stage, path: str, center, scene: air_combat_scene.AirCombatSceneState, color: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    radius = max(4.0, 1300.0 / scene.meters_per_unit)
    z = float(center[2]) + 0.05
    points = []
    for idx in range(73):
        angle = 2.0 * math.pi * idx / 72
        points.append(Gf.Vec3f(float(center[0] + radius * math.cos(angle)), float(center[1] + radius * math.sin(angle)), z))
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr([len(points)])
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([5.0] * len(points))
    curves.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def main() -> None:
    print(f"[TEST] air_combat_scene module: {air_combat_scene.__file__}")
    print(f"[TEST] air_combat_scene version: {getattr(air_combat_scene, 'AIR_COMBAT_SCENE_VERSION', 'unknown')}")
    stage = omni.usd.get_context().get_stage()
    if not stage:
        stage = Usd.Stage.CreateInMemory()
        omni.usd.get_context().set_stage(stage)

    scene = air_combat_scene.create_scene(stage, build_config())

    from isaacsim.core.utils.viewports import set_camera_view

    eye, target = scene.camera_view()
    set_camera_view(eye=eye, target=target)
    print_scene_summary(scene)
    mission_demo = MissionDemoController(stage)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    start = time.time()
    frame_count = 0
    tactical_time_s = 0.0
    time_scale = float(scene.config.get("simulation", {}).get("time_scale", 45.0))
    print("[TEST] Air-combat mission assignment and path-planning scene is running. Close the window or press Ctrl+C to stop.")

    try:
        while simulation_app.is_running():
            tactical_dt = DT * time_scale
            tactical_time_s += tactical_dt
            scene.update(tactical_dt=tactical_dt, tactical_time_s=tactical_time_s)
            mission_demo.update(scene)

            if frame_count % 60 == 0:
                print_tactical_status(scene, frame_count)

            simulation_app.update()
            frame_count += 1
            if time.time() - start > RUN_SECONDS:
                break
    finally:
        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
