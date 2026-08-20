from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path
from typing import Any

import numpy as np

from scenes import mountain_forest_aircraft
from sensors import manager as sensor_manager
from sensors.tracking import DEFAULT_TRACKING_CONFIG, SensorTrackingSystem
from weather import WEATHER_PRESETS, resolve_weather_config
from weapons.missiles import MissileSystem


QL_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = QL_ROOT / "sensor_output"
DEFAULT_CONFIG = QL_ROOT / "configs" / "mountain_forest_aircraft.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven Isaac Sim mountain forest aircraft scene.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=None)
    parser.add_argument("--tree-count", type=int, default=None)
    parser.add_argument("--map-size", type=float, default=None)
    parser.add_argument("--height-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--uav-motion-model", type=str, choices=["kinematic", "dynamic"], default=None)
    parser.add_argument("--helicopter-motion-model", type=str, choices=["kinematic", "dynamic"], default=None)
    parser.add_argument("--uav-max-speed", type=float, default=None)
    parser.add_argument("--uav-max-accel", type=float, default=None)
    parser.add_argument("--helicopter-max-speed", type=float, default=None)
    parser.add_argument("--helicopter-max-accel", type=float, default=None)
    parser.add_argument("--capture-seconds", type=float, default=None)
    parser.add_argument("--capture-interval", type=float, default=None)
    parser.add_argument("--tracking-report-interval", type=float, default=None)
    parser.add_argument("--enable-tracking", action="store_true")
    parser.add_argument("--disable-tracking", action="store_true")
    parser.add_argument("--enable-tracking-output", action="store_true")
    parser.add_argument("--weather-preset", type=str, choices=sorted(WEATHER_PRESETS.keys()), default=None)
    parser.add_argument("--cloud-cover", type=float, default=None)
    parser.add_argument("--fog-density", type=float, default=None)
    parser.add_argument("--rain-rate", type=float, default=None)
    parser.add_argument("--sea-clutter", type=float, default=None)
    parser.add_argument("--sensor-resolution", type=int, default=None)
    parser.add_argument("--sensor-coverage", type=float, default=None)
    parser.add_argument("--sensor-altitude", type=float, default=None)
    parser.add_argument("--sensors", type=str, default=None)
    parser.add_argument("--sar-speckle-shape", type=float, default=None)
    parser.add_argument("--sar-edge-gain", type=float, default=None)
    parser.add_argument("--arm-red-threshold", type=int, default=None)
    parser.add_argument("--arm-red-ratio", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--enable-sensor-capture", action="store_true")
    parser.add_argument("--disable-missiles", action="store_true")
    parser.add_argument("--uav-missile-time", type=float, default=None)
    parser.add_argument("--helicopter-missile-time", type=float, default=None)
    parser.add_argument("--overview-screenshot", action="store_true")
    parser.add_argument("--overview-output", type=str, default=None)
    parser.add_argument("--overview-width", type=int, default=None)
    parser.add_argument("--overview-height", type=int, default=None)
    args, kit_args = parser.parse_known_args()
    args.kit_args = kit_args
    return args


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", ""}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, data)]
    if not path.exists():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, raw_value = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if value is None:
        return
    node = config
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _deep_merge(mountain_forest_aircraft.DEFAULT_SCENE_CONFIG, _load_simple_yaml(Path(args.config)))
    _set_nested(config, ("terrain", "map_size"), args.map_size)
    _set_nested(config, ("terrain", "height_scale"), args.height_scale)
    _set_nested(config, ("forest", "tree_count"), args.tree_count)
    _set_nested(config, ("forest", "seed"), args.seed)
    _set_nested(config, ("aircraft", "uav_motion", "type"), args.uav_motion_model)
    _set_nested(config, ("aircraft", "helicopter_motion", "type"), args.helicopter_motion_model)
    _set_nested(config, ("aircraft", "uav_motion", "max_speed"), args.uav_max_speed)
    _set_nested(config, ("aircraft", "uav_motion", "max_accel"), args.uav_max_accel)
    _set_nested(config, ("aircraft", "helicopter_motion", "max_speed"), args.helicopter_max_speed)
    _set_nested(config, ("aircraft", "helicopter_motion", "max_accel"), args.helicopter_max_accel)
    _set_nested(config, ("runtime", "hold_seconds"), args.hold_seconds)
    _set_nested(config, ("sensors", "capture_seconds"), args.capture_seconds)
    _set_nested(config, ("sensors", "capture_interval"), args.capture_interval)
    _set_nested(config, ("tracking", "report_interval"), args.tracking_report_interval)
    _set_nested(config, ("sensors", "resolution"), args.sensor_resolution)
    _set_nested(config, ("sensors", "coverage"), args.sensor_coverage)
    _set_nested(config, ("sensors", "altitude"), args.sensor_altitude)
    _set_nested(config, ("sensors", "names"), args.sensors)
    _set_nested(config, ("sensors", "sar", "speckle_shape"), args.sar_speckle_shape)
    _set_nested(config, ("sensors", "sar", "edge_gain"), args.sar_edge_gain)
    _set_nested(config, ("sensors", "arm", "red_threshold"), args.arm_red_threshold)
    _set_nested(config, ("sensors", "arm", "red_ratio"), args.arm_red_ratio)
    _set_nested(config, ("weapons", "uav_missile_time"), args.uav_missile_time)
    _set_nested(config, ("weapons", "helicopter_missile_time"), args.helicopter_missile_time)
    _set_nested(config, ("overview_screenshot", "output_path"), args.overview_output)
    _set_nested(config, ("overview_screenshot", "width"), args.overview_width)
    _set_nested(config, ("overview_screenshot", "height"), args.overview_height)
    if args.output_dir is not None:
        _set_nested(config, ("sensors", "output_dir"), args.output_dir)
    if args.enable_sensor_capture:
        _set_nested(config, ("sensors", "capture_enabled"), True)
    if args.enable_tracking:
        _set_nested(config, ("tracking", "enabled"), True)
    if args.disable_tracking:
        _set_nested(config, ("tracking", "enabled"), False)
    if args.enable_tracking_output:
        _set_nested(config, ("tracking", "enabled"), True)
        _set_nested(config, ("tracking", "output_enabled"), True)
    if args.disable_missiles:
        _set_nested(config, ("weapons", "enabled"), False)
    if args.overview_screenshot:
        _set_nested(config, ("overview_screenshot", "enabled"), True)
        overview_cfg = config.setdefault("overview_screenshot", {})
        runtime_cfg = config.setdefault("runtime", {})
        runtime_cfg["width"] = int(overview_cfg.get("width") or 2560)
        runtime_cfg["height"] = int(overview_cfg.get("height") or 1440)
        if args.hold_seconds is None:
            runtime_cfg["hold_seconds"] = 4.0
    if args.weather_preset is not None:
        weather_override: dict[str, Any] = {"preset": args.weather_preset}
        if args.cloud_cover is not None:
            weather_override["cloud_cover"] = args.cloud_cover
        if args.fog_density is not None:
            weather_override["fog_density"] = args.fog_density
        if args.rain_rate is not None:
            weather_override["rain_rate"] = args.rain_rate
        if args.sea_clutter is not None:
            weather_override["sea_clutter"] = args.sea_clutter
        config["weather"] = resolve_weather_config(weather_override)
    else:
        _set_nested(config, ("weather", "cloud_cover"), args.cloud_cover)
        _set_nested(config, ("weather", "fog_density"), args.fog_density)
        _set_nested(config, ("weather", "rain_rate"), args.rain_rate)
        _set_nested(config, ("weather", "sea_clutter"), args.sea_clutter)
        config["weather"] = resolve_weather_config(config.get("weather", {}))
    return config


def _make_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir:
        output_dir = Path(raw_output_dir).expanduser()
    else:
        output_dir = OUTPUT_ROOT / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _kit_extra_args(runtime_cfg: dict[str, Any], headless: bool, cli_extra_args: list[str] | None = None) -> list[str]:
    raw_extra_args = runtime_cfg.get("extra_args", [])
    if isinstance(raw_extra_args, str):
        extra_args = [raw_extra_args]
    elif isinstance(raw_extra_args, (list, tuple)):
        extra_args = [str(arg) for arg in raw_extra_args]
    else:
        extra_args = []
    if headless:
        extra_args.append("--no-window")
    if cli_extra_args:
        extra_args.extend(str(arg) for arg in cli_extra_args)
    return extra_args


def _create_missiles(stage, visual_sphere_cls, scene, config: dict[str, Any]) -> MissileSystem | None:
    from isaacsim.core.api.objects import VisualCuboid

    weapons_cfg = config.get("weapons", {})
    if not bool(weapons_cfg.get("enabled", True)):
        return None
    if scene.uav is None and scene.helicopter is None:
        return None

    missile_system = MissileSystem(stage, VisualCuboid, visual_sphere_cls)
    target_pos = scene.emitter_pos + np.array(weapons_cfg.get("target_offset", [0.0, 0.0, 0.6]), dtype=float)
    if scene.uav is not None:
        launch_t = float(weapons_cfg.get("uav_missile_time", 8.0))
        start = scene.uav_patrol_pos(launch_t) + np.array(weapons_cfg.get("uav_launch_offset", [0.6, 0.0, -0.35]), dtype=float)
        missile_system.create_missile("UAV_0_Missile", start=start, target=target_pos, launch_t=launch_t, speed=float(weapons_cfg.get("uav_speed", 45.0)))
    if scene.helicopter is not None:
        launch_t = float(weapons_cfg.get("helicopter_missile_time", 12.0))
        start = scene.helicopter_patrol_pos(launch_t) + np.array(weapons_cfg.get("helicopter_launch_offset", [1.2, 0.0, -0.55]), dtype=float)
        missile_system.create_missile(
            "Helicopter_1_Missile",
            start=start,
            target=target_pos,
            launch_t=launch_t,
            speed=float(weapons_cfg.get("helicopter_speed", 38.0)),
        )
    return missile_system


def _overview_screenshot_path(raw_path: str | None) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser()
    else:
        path = QL_ROOT / "outputs" / "overview_45deg.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _capture_overview_screenshot(output_path: Path) -> bool:
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            print("[QL][WARN] Overview screenshot skipped: active viewport not found.")
            return False
        if hasattr(viewport, "capture_to_file"):
            viewport.capture_to_file(str(output_path))
            print(f"[QL] Overview screenshot saved: {output_path}")
            return True

        from omni.kit.viewport.utility import capture_viewport_to_file

        try:
            capture = capture_viewport_to_file(viewport, str(output_path))
        except TypeError:
            capture = capture_viewport_to_file(str(output_path), viewport)
        if hasattr(capture, "wait_for_result"):
            capture.wait_for_result()
        print(f"[QL] Overview screenshot saved: {output_path}")
        return True
    except Exception as exc:
        print(f"[QL][WARN] Overview screenshot failed: {exc}")
        return False


def main() -> None:
    args = parse_args()
    config = load_config(args)
    runtime_cfg = config.get("runtime", {})
    sensors_cfg = config.get("sensors", {})
    weather_cfg = dict(config.get("weather", {}))

    from isaacsim import SimulationApp

    headless = bool(args.headless or runtime_cfg.get("headless", False))
    simulation_app = SimulationApp(
        {
            "width": int(runtime_cfg.get("width", 1280)),
            "height": int(runtime_cfg.get("height", 720)),
            "headless": headless,
            "renderer": runtime_cfg.get("renderer", "RaytracedLighting"),
            "extra_args": _kit_extra_args(runtime_cfg, headless, getattr(args, "kit_args", [])),
        }
    )

    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import VisualCuboid, VisualSphere
    from isaacsim.core.utils.viewports import set_camera_view

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    scene = mountain_forest_aircraft.create_scene(stage, VisualCuboid, VisualSphere, config)
    missile_system = _create_missiles(stage, VisualSphere, scene, config)

    sensor_paths = {}
    if scene.uav_root_path is not None:
        sensor_paths = sensor_manager.create_sensor_cameras(
            stage,
            drone_path=scene.uav_root_path,
            altitude=float(sensors_cfg.get("altitude") or scene.height_scale + 26.0),
            coverage=float(sensors_cfg.get("coverage", 70.0)),
            sensor_names=sensors_cfg.get("names", "eo,sar,arm"),
        )

    eye, target = mountain_forest_aircraft.camera_view(config)
    set_camera_view(eye=eye, target=target)
    world.reset()

    output_dir = None
    annotators = None
    if bool(sensors_cfg.get("capture_enabled", False)) and sensor_paths:
        output_dir = _make_output_dir(sensors_cfg.get("output_dir"))
        annotators = sensor_manager.setup_annotators(sensor_paths, int(sensors_cfg.get("resolution", 640)))
    tracking_cfg = config.get("tracking", DEFAULT_TRACKING_CONFIG)
    tracker = None
    next_track_t = 0.0
    track_frame_idx = 0
    if bool(tracking_cfg.get("enabled", True)):
        tracking_cfg = dict(tracking_cfg)
        tracking_cfg["weather"] = dict(weather_cfg)
        tracker = SensorTrackingSystem(
            config=tracking_cfg,
            terrain_height_fn=mountain_forest_aircraft.terrain_height,
            map_size=float(scene.map_size),
            height_scale=float(scene.height_scale),
            water_cfg=config.get("water", {}),
            seed=int(config.get("forest", {}).get("seed", 42)),
        )
        if output_dir is None and bool(tracking_cfg.get("output_enabled", False)):
            output_dir = _make_output_dir(sensors_cfg.get("output_dir"))
    sensor_process_params = {
        "eo": {
            "weather": weather_cfg,
        },
        "sar": {
            "speckle_shape": float(sensors_cfg.get("sar", {}).get("speckle_shape", 3.0)),
            "edge_gain": float(sensors_cfg.get("sar", {}).get("edge_gain", 0.55)),
            "weather": weather_cfg,
        },
        "arm": {
            "red_threshold": int(sensors_cfg.get("arm", {}).get("red_threshold", 135)),
            "red_ratio": float(sensors_cfg.get("arm", {}).get("red_ratio", 1.35)),
            "weather": weather_cfg,
        },
    }

    if missile_system is not None:
        print("[QL] Missile visuals enabled from config.")
    if output_dir is not None:
        print(f"[QL] Sensor capture enabled. Output dir: {output_dir}")
    else:
        print("[QL] Sensor capture disabled by default. Use --enable-sensor-capture or config sensors.capture_enabled=true.")
    print(
        "[QL] Weather preset="
        f"{weather_cfg.get('preset', 'clear')} cloud={float(weather_cfg.get('cloud_cover', 0.0)):.2f} "
        f"fog={float(weather_cfg.get('fog_density', 0.0)):.2f} rain={float(weather_cfg.get('rain_rate', 0.0)):.2f} "
        f"sea_clutter={float(weather_cfg.get('sea_clutter', 0.0)):.2f}"
    )
    if tracker is not None:
        print(f"[QL] Tracking enabled. Report interval={tracker.report_interval():.2f}s")
    print("[QL] Close Isaac Sim or press Ctrl+C in the terminal to stop.")

    start = time.time()
    step = 0
    dt = float(runtime_cfg.get("dt", 1.0 / 60.0))
    next_capture_t = 0.0
    capture_idx = 0
    capture_seconds = float(sensors_cfg.get("capture_seconds", 12.0))
    capture_interval = max(0.1, float(sensors_cfg.get("capture_interval", 1.0)))
    hold_seconds = float(runtime_cfg.get("hold_seconds", -1.0))
    overview_cfg = config.get("overview_screenshot", {})
    overview_enabled = bool(overview_cfg.get("enabled", False))
    overview_frame = int(overview_cfg.get("frame", 24))
    overview_output = _overview_screenshot_path(overview_cfg.get("output_path")) if overview_enabled else None
    overview_captured = False

    while simulation_app.is_running():
        t = step * dt
        scene.update(t)
        if missile_system is not None:
            missile_system.update(t)
        world.step(render=True)
        if overview_enabled and not overview_captured and overview_output is not None and step >= overview_frame:
            overview_captured = _capture_overview_screenshot(overview_output)
        if annotators is not None and output_dir is not None and t >= next_capture_t and t <= capture_seconds:
            sensor_manager.capture(annotators, output_dir, capture_idx, sensor_process_params)
            capture_idx += 1
            next_capture_t += capture_interval
        if tracker is not None and t >= next_track_t:
            observer = scene.observer_snapshot()
            if observer is not None:
                report = tracker.step(t, observer, scene.sensor_targets())
                if output_dir is not None:
                    tracker.save_report(output_dir, track_frame_idx, report)
                print(
                    f"[QL][Track] t={t:.1f}s detections={len(report['detections'])} tracks={len(report['tracks'])}"
                )
                track_frame_idx += 1
            next_track_t += tracker.report_interval()
        step += 1
        if hold_seconds >= 0 and time.time() - start >= hold_seconds:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
