"""SAR search mission visualization with animated drone(s) in Isaac Sim.

Reads config from sar_search_planner/_active_config.json (written by run.sh).
- Single-config:  animates one drone on its search path.
- Multi-config without _platform_id:  comparison overlay (example 5), no animation.
- Multi-config with _platform_id:  multi-drone mode — each platform flies its
  own search path simultaneously (example 6,  2x2 grid).

Usage: /home/isaac/isaacsim/python.sh sar_search_planner/run_mission_viz.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "renderer": "HydraStorm",
        "width": 1280,
        "height": 720,
    }
)

import json
import math
import os
import sys
import time

# Ensure project root (parent of sar_search_planner/) is on sys.path
# so that 'scenes', 'scripts' etc. are importable.
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import omni.timeline
import omni.usd

from scenes import air_combat_scene
from sar_search_planner import (
    PlannerConfig,
    plan,
    export_waypoints_usd,
    export_search_area_boundary,
)
# from sar_search_planner.sensor_viz import SensorFOV, SensorFOVParams, CoverageMap

_PATTERN_COLORS = {
    "racetrack":    (0.0, 0.55, 1.0),
    "sar_polygon":  (1.0, 0.65, 0.05),
    "sar_rounded":  (0.05, 1.0, 0.25),
    "figure_eight": (0.85, 0.15, 1.0),
}

RUN_SECONDS = float(os.environ.get("QL_SAR_VIZ_RUN_SECONDS", "300.0"))
ANIM_SPEED_SCALE = float(os.environ.get("SAR_ANIM_SPEED_SCALE", "1.0"))
SHOW_PATHS = os.environ.get("SAR_SHOW_PATHS", "1") not in ("0", "false", "no")


def _load_configs():
    config_path = "sar_search_planner/_active_config.json"
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [{"area_center_km": [60, -30], "area_width_km": 20,
                 "area_height_km": 15, "pattern": "sar_rounded",
                 "sar_radius_km": 8, "sar_sides": 6,
                 "sar_turn_radius_km": 4, "altitude_agl_m": 5000}]
    if isinstance(data, list):
        return data
    return [data]


def _build_path_interpolator(wps):
    """Build a linear interpolator for a waypoint list (closed loop).

    Deduplicates consecutive near-identical waypoints and returns a
    distance-based interpolator.  Waypoints are already Dubins-smoothed
    by the planner so linear interpolation produces a flyable path.

    Returns (total_distance, interpolate_fn).
    """
    # Deduplicate consecutive near-identical points
    cleaned = [wps[0]]
    for wp in wps[1:]:
        dx = wp.x - cleaned[-1].x
        dy = wp.y - cleaned[-1].y
        dz = wp.z - cleaned[-1].z
        if math.sqrt(dx*dx + dy*dy + dz*dz) > 0.01:
            cleaned.append(wp)
    wps = cleaned
    n = len(wps)
    if n < 2:
        def _empty(dist):
            return (wps[0].x, wps[0].y, wps[0].z)
        return 0.0, _empty

    segs = []
    for i in range(n):
        j = (i + 1) % n
        dx = wps[j].x - wps[i].x
        dy = wps[j].y - wps[i].y
        dz = wps[j].z - wps[i].z
        segs.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    total = sum(segs)

    cum = [0.0]
    for s in segs:
        cum.append(cum[-1] + s)

    def _interpolate(dist):
        d = dist % total
        for i in range(n):
            if cum[i + 1] >= d:
                seg_len = cum[i + 1] - cum[i]
                t = (d - cum[i]) / seg_len if seg_len > 0 else 0.0
                t = max(0.0, min(1.0, t))
                j = (i + 1) % n
                return (
                    wps[i].x + t * (wps[j].x - wps[i].x),
                    wps[i].y + t * (wps[j].y - wps[i].y),
                    wps[i].z + t * (wps[j].z - wps[i].z),
                )
        return (wps[-1].x, wps[-1].y, wps[-1].z)

    return total, _interpolate


def main() -> None:
    print("[SAR_MISSION] Building air combat scene…")
    stage = omni.usd.get_context().get_stage()
    if not stage:
        stage = Usd.Stage.CreateInMemory()
        omni.usd.get_context().set_stage(stage)

    cfg = air_combat_scene.DEFAULT_AIR_COMBAT_CONFIG
    scene = air_combat_scene.create_scene(stage, cfg)

    from isaacsim.core.utils.viewports import set_camera_view

    eye, target = scene.camera_view()
    set_camera_view(eye=eye, target=target)

    # ── Load configs and run planner(s) ───────────────────────
    cfg_list = _load_configs()
    is_list = len(cfg_list) > 1 or isinstance(
        json.load(open("sar_search_planner/_active_config.json")), list
    )

    # Per-config processing: each may have pre-computed _cycle_waypoints
    # or normal PlannerConfig fields — or be a phased config.
    is_phased = is_list and any("phase_laps" in c for c in cfg_list)
    all_cycle = all("_cycle_waypoints" in c for c in cfg_list) if not is_phased else False
    has_platforms = any("_platform_id" in c for c in cfg_list) if not is_phased else False
    is_comparison = is_list and not has_platforms and not is_phased

    from sar_search_planner.planner import Waypoint as WP

    drone_configs: list[tuple[str, list, object, tuple]] = []
    all_region_data_per_phase: list = []  # for phased mode: per-phase region data
    phase_configs: list[dict] = []  # for phased mode
    mpu = 100.0

    if is_phased:
        # Multi-phase mode: plan each phase independently
        from sar_search_planner.mission import build_multi_region_cycle
        from sar_search_planner.config import PlannerConfig as PC
        from collections import defaultdict

        for pi, phase in enumerate(cfg_list):
            laps = phase.get("phase_laps", -1)
            label = phase.get("phase_label", f"Phase{pi+1}")
            quadrants = phase["quadrants"]
            print(f"[SAR_MISSION] Phase {pi+1}: {label} ({laps} laps)")
            has_platforms = any("_platform_id" in q for q in quadrants)

            # Plan each quadrant
            phase_results = []
            for q in quadrants:
                pid = q["_platform_id"]
                cfg = PC(**{k: v for k, v in q.items() if k != "_platform_id"})
                result = plan(cfg)
                phase_results.append((pid, result.waypoints, result.search_area,
                                     _PATTERN_COLORS.get(q.get("pattern", ""), (1.0, 1.0, 1.0))))
                mpu = result.config.meters_per_unit
                print(f"[SAR_MISSION]     {pid}: {len(result.waypoints)} wp, "
                      f"{result.stats['path_length_km']:.1f} km")

            all_region_data_per_phase.append(list(phase_results))

            # Group by platform and build cycle if needed
            grouped = defaultdict(list)
            for pid, wps, sa, color in phase_results:
                grouped[pid].append((wps, sa, color))

            phase_drones = []
            for pid, items in grouped.items():
                if len(items) == 1:
                    wps, sa, color = items[0]
                    phase_drones.append((pid, wps, sa, color))
                else:
                    region_ids = [f"{pid}_r{j}" for j in range(len(items))]
                    fake_cfg = PC(altitude_agl_m=5000)
                    results_map = {}
                    centers_map = {}
                    for rid, (wps, _sa, _c) in zip(region_ids, items):
                        fake_result = type('obj', (object,), {
                            'waypoints': wps, 'config': fake_cfg
                        })()
                        results_map[rid] = fake_result
                        cx = sum(w.x for w in wps) / len(wps)
                        cy = sum(w.y for w in wps) / len(wps)
                        centers_map[rid] = (cx, cy)
                    cycle_wps = build_multi_region_cycle(
                        results_map, region_ids, centers_map,
                    )
                    phase_drones.append((pid, cycle_wps, [], (1.0, 0.65, 0.05)))
                    print(f"[SAR_MISSION]     {pid}: {len(items)} regions → "
                          f"{len(cycle_wps)} cycle wp")
            phase_configs.append({
                "laps": laps,
                "drones": phase_drones,
            })

        # Use phase 1 for initial display data
        drone_configs = [d for d in phase_configs[0]["drones"]]
        all_region_data = all_region_data_per_phase[0]
    else:
        # Normal mode: run the planner for each config
        for i, cfg_dict in enumerate(cfg_list):
            pid = cfg_dict.pop("_platform_id", None)
            label = cfg_dict.get("pattern", f"path_{i}")
            extra = f" [{pid}]" if pid else ""
            print(f"[SAR_MISSION]   [{i+1}/{len(cfg_list)}] {label}{extra}: "
                  f"centre={cfg_dict.get('area_center_km')}, "
                  f"alt={cfg_dict.get('altitude_agl_m')}m")
            config = PlannerConfig(**cfg_dict)
            result = plan(config)
            cfg_dict["_platform_id"] = pid  # restore
            print(f"[SAR_MISSION]     -> {len(result.waypoints)} waypoints, "
                  f"{result.stats['path_length_km']:.1f} km, "
                  f"collisions {result.stats['collision_count_before']}"
                  f"->{result.stats['collision_count_after']}")
            mpu = result.config.meters_per_unit
            color = _PATTERN_COLORS.get(cfg_dict.get("pattern", ""), (1.0, 1.0, 1.0))
            drone_configs.append((pid, result.waypoints, result.search_area, color))

    # ── Group by platform_id: detect multi-region assignments ──
    # Save ALL original region data before grouping, so we always draw
    # every region's boundary + closed-loop path (not just active drones).
    all_region_data: list[tuple[str, list, object, tuple]] = list(drone_configs)

    if has_platforms:
        from collections import defaultdict
        grouped: dict[str, list[tuple[list, object, tuple]]] = defaultdict(list)
        for pid, wps, sa, color in drone_configs:
            grouped[pid].append((wps, sa, color))

        drone_configs.clear()
        for pid, items in grouped.items():
            if len(items) == 1:
                wps, sa, color = items[0]
                drone_configs.append((pid, wps, sa, color))
            else:
                from sar_search_planner.mission import build_multi_region_cycle
                from sar_search_planner.config import PlannerConfig as PC

                region_ids = [f"{pid}_r{i}" for i in range(len(items))]
                fake_cfg = PC(altitude_agl_m=5000)
                results_map = {}
                centers_map = {}
                for rid, (wps, sa, color) in zip(region_ids, items):
                    fake_result = type('obj', (object,), {
                        'waypoints': wps, 'config': fake_cfg
                    })()
                    results_map[rid] = fake_result
                    cx = sum(w.x for w in wps) / len(wps)
                    cy = sum(w.y for w in wps) / len(wps)
                    centers_map[rid] = (cx, cy)

                cycle_wps = build_multi_region_cycle(
                    results_map, region_ids, centers_map,
                )
                drone_configs.append((pid, cycle_wps, [], (1.0, 0.65, 0.05)))
                print(f"[SAR_MISSION]   {pid}: {len(items)} regions -> "
                      f"{len(cycle_wps)} cycle waypoints")

    # ── USD overlays ──────────────────────────────────────────
    print("[SAR_MISSION] Creating USD overlays…")

    # One big search-area rectangle enclosing all regions
    if all_region_data:
        all_x_min = all_x_max = all_y_min = all_y_max = None
        for _pid, _wps, sa, _color in all_region_data:
            if sa is not None:
                if all_x_min is None:
                    all_x_min, all_x_max = sa.x_min, sa.x_max
                    all_y_min, all_y_max = sa.y_min, sa.y_max
                else:
                    all_x_min = min(all_x_min, sa.x_min)
                    all_x_max = max(all_x_max, sa.x_max)
                    all_y_min = min(all_y_min, sa.y_min)
                    all_y_max = max(all_y_max, sa.y_max)
        if all_x_min is not None:
            from sar_search_planner.area import SearchArea
            big_area = SearchArea(
                x_min=all_x_min, x_max=all_x_max,
                y_min=all_y_min, y_max=all_y_max,
            )
            export_search_area_boundary(
                stage, big_area,
                base_path="/World/AirCombat/SAR_Search",
            )

    # Each region's closed-loop path in its pattern colour
    if SHOW_PATHS:
        for i, (_pid, wps, _sa, color) in enumerate(all_region_data):
            export_waypoints_usd(
                stage, wps,
                base_path=f"/World/AirCombat/SAR_Search/Region_{i}",
                path_color=color,
            )

    # ── Export USD file ───────────────────────────────────────
    import pathlib
    export_dir = pathlib.Path("sar_search_planner/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    example = os.environ.get("ACTIVE_EXAMPLE", "0")
    if all_cycle:
        usd_name = f"sar_example{example}_reassign_{len(drone_configs)}drones.usd"
    elif has_platforms:
        usd_name = f"sar_example{example}_4drone_2x2grid_25x25km.usd"
    elif is_comparison:
        usd_name = f"sar_example{example}_4patterns.usd"
    else:
        pid, _wps, _sa, _c = drone_configs[0]
        usd_name = f"sar_example{example}_{pid}.usd"
    usd_path = export_dir / usd_name
    stage.GetRootLayer().Export(str(usd_path))
    print(f"[SAR_MISSION] USD exported to: {usd_path}")

    # ── Prepare platform(s) for animation ─────────────────────
    from scenes.air_combat_scene import (
        _air_platform_clearance_units,
        _clamp_air_platform_above_terrain,
        _set_root_pose,
        _spin_rotors,
    )

    # Build platform name -> AirPlatform map
    plat_map = {p.entity_id: p for p in scene.platforms}

    # ── Set up each drone ──────────────────────────────────────
    # If pid is None (examples 1-5 without platform assignment),
    # fall back to the env var.
    _default_pid = os.environ.get("SAR_MISSION_PLATFORM", "Blue_CH4_Recon")
    drones = []
    for pid, wps, _sa, _color, *_rest in drone_configs:
        if pid is None:
            pid = _default_pid
        if pid not in plat_map:
            print(f"[SAR_MISSION] ERROR: platform '{pid}' not found!")
            continue
        plat = plat_map[pid]
        total_dist, interpolate = _build_path_interpolator(wps)
        cruise = plat.spec.cruise_speed_mps / mpu
        clearance = _air_platform_clearance_units(scene.config, plat.spec, mpu)

        def _make_ref(total_d, interp, c):
            def _ref(t):
                d = (t * c) % total_d
                x, y, z = interp(d)
                return np.array([x, y, z], dtype=float)
            return _ref

        plat.reference_fn = _make_ref(total_dist, interpolate, cruise)
        drones.append((plat, clearance, cruise, total_dist))
        path_km = total_dist * mpu / 1000.0
        print(f"[SAR_MISSION] {pid}: {len(wps)} wp, {path_km:.1f} km")

    if not drones:
        print("[SAR_MISSION] No drones to animate, exiting.")
        return

    # ── Simulation loop ───────────────────────────────────────
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    DT = 0.016
    time_scale = float(scene.config.get("simulation", {}).get(
        "time_scale", 45.0)) * ANIM_SPEED_SCALE
    tactical_time_s = 0.0

    # ── Sensor FOV visualisation ──────────────────────────────
    # from scenes.air_combat_scene import terrain_height

    # sensor_fovs: list[tuple[SensorFOV, object, float]] = []
    # coverage: CoverageMap | None = None
    # for plat, clearance, _cruise, _td in drones:
    #     # Pick EO (azimuth < 360) and SAR sensors
    #     eo_sensor = None
    #     sar_sensor = None
    #     for s in plat.spec.sensors:
    #         if s.channel == "sar":
    #             sar_sensor = s
    #         elif s.azimuth_fov_deg < 360 and eo_sensor is None:
    #             eo_sensor = s
    #     if eo_sensor is None and plat.spec.sensors:
    #         eo_sensor = plat.spec.sensors[0]
    #     if sar_sensor is None:
    #         sar_sensor = eo_sensor  # fallback to EO params for SAR swath

    #     if eo_sensor:
    #         fov = SensorFOV(
    #             stage,
    #             f"/World/AirCombat/SAR_Search/SensorFOV/{plat.entity_id}",
    #             SensorFOVParams.from_spec(eo_sensor),
    #             SensorFOVParams.from_spec(sar_sensor),
    #             terrain_height,
    #             meters_per_unit=mpu,
    #             map_size_units=scene.map_size_units,
    #             height_scale=scene.terrain_visual_height_units,
    #         )
    #         sensor_fovs.append((fov, plat, 0.0))

    # # Coverage map over the overall search area
    # if all_region_data:
    #     all_x = [r[2].x_min for r in all_region_data if r[2] is not None]
    #     all_x += [r[2].x_max for r in all_region_data if r[2] is not None]
    #     all_y = [r[2].y_min for r in all_region_data if r[2] is not None]
    #     all_y += [r[2].y_max for r in all_region_data if r[2] is not None]
    #     if all_x:
    #         coverage = CoverageMap(
    #             min(all_x) - 20, max(all_x) + 20,
    #             min(all_y) - 20, max(all_y) + 20,
    #             resolution_km=0.5,
    #         )
    #         coverage.create_prims(stage, "/World/AirCombat/SAR_Search")

    # ── Phase switching state ──────────────────────────────────
    current_phase = 0
    phase_start_tactical = 0.0          # tactical time when lap counting began
    _arrived = False                     # True once all drones near their paths
    _arrived = False  # reset each phase
    rtb_drones: list = []               # drones that have been retired (fly to base)

    frame_count = 0
    sim_start = time.time()
    if len(drones) > 1:
        print(f"[SAR_MISSION] {len(drones)} drones now flying!")
    else:
        print(f"[SAR_MISSION] Drone now following planned search path!")
    if is_phased:
        print(f"[SAR_MISSION] Phase 1: {phase_configs[0].get('laps', '?')} laps")
    print("[SAR_MISSION] Close window or Ctrl+C to stop.")

    def _switch_to_phase(phase_idx):
        """Replace drone reference_fns; offline drones get RTB."""
        nonlocal drones, current_phase, phase_start_tactical, _arrived, rtb_drones
        if phase_idx >= len(phase_configs):
            return
        pc = phase_configs[phase_idx]
        next_pids = {d[0] for d in pc["drones"]}

        new_rtb = []
        for plat, clearance, cruise, _td in drones:
            if plat.entity_id not in next_pids:
                cur = plat.motion_model.state.position.copy()
                base = np.array([-1400.0, -400.0, 50.0], dtype=float)
                dist_to_base = float(np.linalg.norm(base[:2] - cur[:2]))
                travel_time = dist_to_base / cruise if cruise > 1e-6 else 60.0
                def _rtb_ref(t, s=tactical_time_s, c=cur, b=base, tt=travel_time):
                    frac = min(1.0, (t - s) / tt) if tt > 0 else 1.0
                    return c + (b - c) * frac
                plat.reference_fn = _rtb_ref
                new_rtb.append((plat, clearance, cruise, 0.0))
                print(f"[SAR_MISSION]   {plat.entity_id} → RTB")
        rtb_drones.extend(new_rtb)

        new_drones = []
        for (pid, wps, _sa, _color) in pc["drones"]:
            plat_obj = plat_map.get(pid)
            if plat_obj is None:
                continue
            td, interp = _build_path_interpolator(wps)
            c = plat_obj.spec.cruise_speed_mps / mpu
            clearance = _air_platform_clearance_units(scene.config, plat_obj.spec, mpu)
            # Find the cycle waypoint closest to drone's current position
            cur_pos = plat_obj.motion_model.state.position.copy()
            best_idx = 0
            best_d2 = float("inf")
            for i, w in enumerate(wps):
                d2 = (w.x - cur_pos[0])**2 + (w.y - cur_pos[1])**2
                if d2 < best_d2:
                    best_d2 = d2
                    best_idx = i
            # Build interpolator starting from closest point
            reordered = wps[best_idx:] + wps[:best_idx]
            td2, interp2 = _build_path_interpolator(reordered)

            t0 = tactical_time_s  # capture switch moment
            def _mr(t, td=td2, interp=interp2, c=c, t0=t0):
                d = ((t - t0) * c) % td
                x, y, z = interp(d)
                return np.array([x, y, z], dtype=float)
            plat_obj.reference_fn = _mr
            new_drones.append((plat_obj, clearance, c, td))
        drones.clear()
        drones.extend(new_drones)
        current_phase = phase_idx
        _arrived = False
        phase_start_tactical = tactical_time_s
        laps_text = "indefinite" if pc["laps"] < 0 else f"{pc['laps']} laps"
        print(f"[SAR_MISSION] → Phase {phase_idx+1}: {laps_text}, "
              f"{len(new_drones)} drones")

    try:
        while simulation_app.is_running():
            tactical_dt = DT * time_scale
            tactical_time_s += tactical_dt

            # Wait until transit done + all drones arrived, then count laps
            if is_phased and not _arrived:
                all_close = True
                for plat, _cl, _cr, _td in drones:
                    tgt = plat.reference_fn(tactical_time_s)
                    pos = plat.motion_model.state.position
                    if np.linalg.norm(pos[:2] - tgt[:2]) > 50.0:
                        all_close = False
                        break
                if all_close:
                    _arrived = True
                    phase_start_tactical = tactical_time_s
                    print(f"[SAR_MISSION] Phase {current_phase+1}: arrived, "
                          f"counting laps")

            # Phase switching: count laps from arrival
            if is_phased and drones and _arrived:
                pc = phase_configs[current_phase]
                laps = pc.get("laps", -1)
                if laps > 0:
                        lap_time = drones[0][3] / (drones[0][2] + 1e-9)
                        elapsed = tactical_time_s - phase_start_tactical
                        if elapsed >= lap_time * laps:
                            next_p = current_phase + 1
                            if next_p < len(phase_configs):
                                _switch_to_phase(next_p)

            for plat, clearance, _cruise, _td in drones + rtb_drones:
                target_pos = plat.reference_fn(tactical_time_s)
                target_vel = np.zeros(3, dtype=float)
                state = plat.motion_model.step(tactical_dt, target_pos, target_vel)
                state.position = _clamp_air_platform_above_terrain(
                    state.position, scene.map_size_units,
                    scene.terrain_visual_height_units, clearance,
                )
                _set_root_pose(plat.root_prim, state.position, state)
                _spin_rotors(plat.rotor_prims, tactical_time_s)

            # # Update sensor FOVs
            # for fov, plat, _last_t in sensor_fovs:
            #     px, py, pz = plat.motion_model.state.position
            #     yaw = plat.motion_model.state.yaw_deg
            #     fov.update(float(px), float(py), float(pz), float(yaw))

            # # Update coverage map (every 15 frames, using trail end)
            # frame_count += 1
            # if coverage and frame_count % 15 == 0:
            #     for fov, plat, _last_t in sensor_fovs:
            #         if len(fov._trail) >= 2:
            #             hw = fov._sar_range * 0.20
            #             for tp in fov._trail[-5:]:
            #                 r_rad = math.radians(tp[3])
            #                 rx = -math.sin(r_rad)
            #                 ry = math.cos(r_rad)
            #                 coverage.mark_footprint([
            #                     (tp[0] + rx * hw, tp[1] + ry * hw, tp[2]),
            #                     (tp[0] - rx * hw, tp[1] - ry * hw, tp[2]),
            #                     (tp[0] + rx * hw * 0.5, tp[1] + ry * hw * 0.5, tp[2]),
            #                 ])
            #     coverage.update_prims()

            simulation_app.update()
            if time.time() - sim_start > RUN_SECONDS:
                break
    finally:
        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
