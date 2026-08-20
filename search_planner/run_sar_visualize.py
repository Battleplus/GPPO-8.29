import os as __os_plhdr
import sys as __sys_plhdr
__p = __os_plhdr.path.dirname(__os_plhdr.path.dirname(__os_plhdr.path.abspath(__file__)))
if __p not in __sys_plhdr.path:
    __sys_plhdr.path.insert(0, __p)

from isaacsim import SimulationApp

"""SAR search path visualization in Isaac Sim.

Reads config from sar_search_planner/_active_config.json (written by run.sh).
If the file is missing, falls back to a built-in default.
"""


def _load_configs():
    """Load config(s) from the JSON file written by run.sh.

    Returns (is_multi, list_of_config_dicts).
    Single-config files become a one-element list.
    """
    import json

    config_path = "sar_search_planner/_active_config.json"
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return False, [{
            "area_center_km": [60, -30],
            "area_width_km": 20,
            "area_height_km": 15,
            "pattern": "sar_rounded",
            "sar_radius_km": 8,
            "sar_sides": 6,
            "sar_turn_radius_km": 4,
            "altitude_agl_m": 5000,
        }]

    if isinstance(data, list):
        return True, data
    return False, [data]


# Colors for multi-pattern overlay
_PATTERN_COLORS = {
    "racetrack":    (0.0, 0.55, 1.0),   # blue
    "sar_polygon":  (1.0, 0.65, 0.05),  # orange
    "sar_rounded":  (0.05, 1.0, 0.25),  # green
    "figure_eight": (0.85, 0.15, 1.0),  # purple
}

simulation_app = SimulationApp(
    {
        "headless": False,
        "renderer": "HydraStorm",
        "width": 1280,
        "height": 720,
    }
)

import os
import time

import omni.timeline
import omni.usd
from pxr import Usd

from scenes import air_combat_scene
from sar_search_planner import (
    PlannerConfig,
    plan,
    export_waypoints_usd,
    export_search_area_boundary,
    export_sar_swath,
    export_uav_platform,
)

RUN_SECONDS = float(os.environ.get("QL_SAR_VIZ_RUN_SECONDS", "300.0"))


def main() -> None:
    # ── 1. Build the scene (same as test_air_combat_scene.py) ──
    print("[SAR_VIZ] Building air combat scene…")
    stage = omni.usd.get_context().get_stage()
    if not stage:
        stage = Usd.Stage.CreateInMemory()
        omni.usd.get_context().set_stage(stage)

    cfg = air_combat_scene.DEFAULT_AIR_COMBAT_CONFIG
    scene = air_combat_scene.create_scene(stage, cfg)

    from isaacsim.core.utils.viewports import set_camera_view

    eye, target = scene.camera_view()
    set_camera_view(eye=eye, target=target)

    # ── 2. Run the SAR planner(s) ────────────────────────────
    print("[SAR_VIZ] Running SAR search planner…")
    is_multi, cfg_list = _load_configs()

    results = []
    for i, cfg_dict in enumerate(cfg_list):
        label = cfg_dict.get("pattern", f"path_{i}")
        print(f"[SAR_VIZ]   [{i+1}/{len(cfg_list)}] {label}: "
              f"centre={cfg_dict.get('area_center_km')}, "
              f"alt={cfg_dict.get('altitude_agl_m')}m")
        config = PlannerConfig(**cfg_dict)
        result = plan(config)
        results.append((cfg_dict, result))
        print(f"[SAR_VIZ]     → {len(result.waypoints)} waypoints, "
              f"{result.stats['path_length_km']:.1f} km, "
              f"collisions {result.stats['collision_count_before']}→{result.stats['collision_count_after']}")

    # ── 3. USD overlays ────────────────────────────────────
    print("[SAR_VIZ] Creating USD overlays…")
    first_result = results[0][1]

    if is_multi:
        print(f"[SAR_VIZ] Multi-pattern comparison: {len(results)} paths")
        export_search_area_boundary(stage, first_result.search_area)
        for i, (cfg_dict, result) in enumerate(results):
            pattern = cfg_dict.get("pattern", f"path_{i}")
            color = _PATTERN_COLORS.get(pattern, (1.0, 1.0, 1.0))
            bp = f"/World/AirCombat/SAR_Search/{pattern}"
            export_waypoints_usd(stage, result.waypoints, base_path=bp, path_color=color, label=pattern)
            if i == 0:
                export_sar_swath(stage, result.waypoints, base_path=bp, color=color)
                if result.waypoints:
                    wp0 = result.waypoints[0]
                    export_uav_platform(stage, wp0.x, wp0.y, wp0.z)
        legend = ", ".join(
            f"{p}={['blue','orange','green','purple'][i]}"
            for i, p in enumerate(["racetrack","sar_polygon","sar_rounded","figure_eight"])
        )
        print(f"[SAR_VIZ] Legend: {legend}")
    else:
        export_search_area_boundary(stage, first_result.search_area)
        export_waypoints_usd(stage, first_result.waypoints, label=first_result.config.pattern)
        export_sar_swath(stage, first_result.waypoints)
        if first_result.waypoints:
            wp0 = first_result.waypoints[0]
            export_uav_platform(stage, wp0.x, wp0.y, wp0.z)

    # ── 4. Export USD file ──────────────────────────────────
    import pathlib
    export_dir = pathlib.Path("sar_search_planner/exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    example = os.environ.get("ACTIVE_EXAMPLE", "0")
    if is_multi:
        area = first_result.search_area
        usd_name = (f"sar_example{example}_4patterns"
                    f"_{area.center_x*100/1000:.0f}_{area.center_y*100/1000:.0f}"
                    f"_{area.width_km():.0f}x{area.height_km():.0f}km.usd")
    else:
        cfg = cfg_list[0]
        area = first_result.search_area
        pat = cfg.get("pattern", "unknown")
        usd_name = (f"sar_example{example}_{pat}"
                    f"_{area.center_x*100/1000:.0f}_{area.center_y*100/1000:.0f}"
                    f"_{area.width_km():.0f}x{area.height_km():.0f}km.usd")
    usd_path = export_dir / usd_name
    stage.GetRootLayer().Export(str(usd_path))
    print(f"[SAR_VIZ] USD exported to: {usd_path}")

    # ── 5. Simulation loop ─────────────────────────────────
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    start = time.time()
    print("[SAR_VIZ] Visualization running. Close the window or press Ctrl+C to stop.")
    print("[SAR_VIZ] Legend: yellow frame = search area, "
          "colored spheres = waypoints, blue strips = SAR swath")

    try:
        while simulation_app.is_running():
            simulation_app.update()
            if time.time() - start > RUN_SECONDS:
                break
    finally:
        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
