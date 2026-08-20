"""Formation mission runner -- integrates Hybrid A* planner with air combat scene.

Usage:
    /home/isaac/isaacsim/python.sh scripts/run_formation_mission.py \
        --count 4 --formation v_shape \
        --start -800 -600 80 --goal 800 600 80

This script:
1. Runs the formation planner to get per-member waypoints.
2. Builds the air combat scene.
3. Replaces selected platform reference functions with waypoint followers.
4. Adds visual path markers (colored lines + waypoint spheres).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "renderer": "HydraStorm", "width": 1280, "height": 720})

import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

# Ensure ql/ and ql/scripts are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QL_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_QL_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scenes import air_combat_scene

# Direct imports (avoid triggering scripts/__init__.py)
from formation import FORMATION_NAMES, get_formation_offsets, get_formation_roles, distribute_team_waypoints
from obstacles import build_obstacles, lift_waypoints_above_terrain

# Planners -- prefer MPPI (cutting-edge), fall back to Hybrid A*
try:
    from mppi import MPPIConfig, MPPIPlanner
    _HAS_MPPI = True
except ImportError:
    _HAS_MPPI = False

from hybrid_astar import HybridAStarPlanner, PlannerConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DT = 1.0 / 30.0
DEFAULT_TIME_SCALE = 45.0
TEAM_COLORS = [
    (0.20, 0.80, 1.00),   # cyan
    (1.00, 0.55, 0.00),   # orange
    (0.40, 1.00, 0.40),   # green
    (1.00, 0.35, 0.35),   # red
    (0.80, 0.40, 1.00),   # purple
    (1.00, 1.00, 0.20),   # yellow
]
PATH_COLOR = (0.30, 0.90, 0.50)   # green path line
CENTER_COLOR = (1.00, 1.00, 1.00)  # white center path


# ---------------------------------------------------------------------------
# Waypoint-following reference function
# ---------------------------------------------------------------------------

def make_waypoint_follower(
    waypoints: list[np.ndarray],
    speed_units_per_s: float = 25.0,
) -> callable:
    """Create a reference_fn(t) that moves along waypoints at constant speed.

    Args:
        waypoints: List of [x, y, z] waypoints.
        speed_units_per_s: Cruise speed in scene units per second.

    Returns:
        A function f(tactical_time_s) -> np.ndarray position.
    """
    if not waypoints:
        return lambda t: np.zeros(3, dtype=float)

    # Pre-compute cumulative segment lengths
    seg_lengths: list[float] = []
    cumulative: list[float] = [0.0]
    for i in range(1, len(waypoints)):
        d = float(np.linalg.norm(waypoints[i] - waypoints[i - 1]))
        seg_lengths.append(d)
        cumulative.append(cumulative[-1] + d)
    total_length = cumulative[-1]

    def follower(tactical_time_s: float) -> np.ndarray:
        t = float(tactical_time_s)
        if t <= 0.0:
            return waypoints[0].copy()
        dist_traveled = t * speed_units_per_s
        if dist_traveled >= total_length:
            return waypoints[-1].copy()
        # Find segment
        for i in range(1, len(cumulative)):
            if dist_traveled <= cumulative[i]:
                seg_start = cumulative[i - 1]
                seg_len = cumulative[i] - seg_start
                alpha = (dist_traveled - seg_start) / max(seg_len, 1e-6)
                alpha = max(0.0, min(1.0, alpha))
                return waypoints[i - 1] * (1.0 - alpha) + waypoints[i] * alpha
        return waypoints[-1].copy()

    return follower


def make_path_tracker(
    waypoints: list[np.ndarray],
    platform,
    lookahead_units: float = 18.0,
) -> callable:
    """Create a reference_fn that stays just ahead of the aircraft on the path."""
    if not waypoints:
        return lambda t: np.zeros(3, dtype=float)
    if len(waypoints) == 1:
        return lambda t: waypoints[0].copy()

    cumulative: list[float] = [0.0]
    for i in range(1, len(waypoints)):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(waypoints[i] - waypoints[i - 1])))
    total_length = cumulative[-1]

    def point_at(distance: float) -> np.ndarray:
        distance = max(0.0, min(float(distance), total_length))
        for i in range(1, len(cumulative)):
            if distance <= cumulative[i]:
                seg_start = cumulative[i - 1]
                seg_len = max(1e-6, cumulative[i] - seg_start)
                alpha = (distance - seg_start) / seg_len
                return waypoints[i - 1] * (1.0 - alpha) + waypoints[i] * alpha
        return waypoints[-1].copy()

    def closest_distance(pos: np.ndarray) -> float:
        best_d2 = float("inf")
        best_s = 0.0
        p = np.array(pos, dtype=float)
        for i in range(1, len(waypoints)):
            a = waypoints[i - 1]
            b = waypoints[i]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 <= 1e-9:
                continue
            u = max(0.0, min(1.0, float(np.dot(p - a, ab) / ab2)))
            proj = a + ab * u
            d2 = float(np.dot(p - proj, p - proj))
            if d2 < best_d2:
                best_d2 = d2
                best_s = cumulative[i - 1] + math.sqrt(ab2) * u
        return best_s

    def tracker(_tactical_time_s: float) -> np.ndarray:
        s = closest_distance(platform.position)
        return point_at(s + lookahead_units)

    return tracker


# ---------------------------------------------------------------------------
# Member assignment helper
# ---------------------------------------------------------------------------

def _apply_member_assignments(
    team_paths: list[list[np.ndarray]],
    roles: list[str],
    member_assignments: dict[str, int],
) -> list[list[np.ndarray]]:
    """Reorder team_paths so team_paths[dst_idx] = path for the assigned role."""
    n = len(team_paths)
    reordered = [list(p) for p in team_paths]

    role_to_src_idx: dict[str, int] = {}
    for idx, role in enumerate(roles):
        role_to_src_idx[role] = idx

    for role, dst_idx in member_assignments.items():
        if role not in role_to_src_idx:
            print(f"  [PLAN][WARN] Unknown role '{role}', skipping. "
                  f"Available: {list(role_to_src_idx.keys())}")
            continue
        if not (0 <= dst_idx < n):
            print(f"  [PLAN][WARN] Output index {dst_idx} out of range "
                  f"[0, {n}), skipping role '{role}'")
            continue
        src_idx = role_to_src_idx[role]
        reordered[dst_idx] = list(team_paths[src_idx])
        print(f"  [PLAN] Assigned: {role} -> team_paths[{dst_idx}] "
              f"(was team_paths[{src_idx}])")

    return reordered


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _create_path_curve(
    stage,
    path: str,
    waypoints: list[np.ndarray],
    color: tuple[float, float, float],
    width: float = 1.5,
) -> None:
    """Draw a polyline through waypoints."""
    if len(waypoints) < 2:
        return
    pts = [Gf.Vec3f(float(wp[0]), float(wp[1]), float(wp[2])) for wp in waypoints]
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr([len(pts)])
    curves.CreatePointsAttr(pts)
    curves.CreateWidthsAttr([width] * len(pts))
    curves.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _create_waypoint_markers(
    stage,
    base_path: str,
    waypoints: list[np.ndarray],
    color: tuple[float, float, float],
    radius: float = 2.0,
    step: int = 5,
) -> None:
    """Place small spheres at every Nth waypoint."""
    for i, wp in enumerate(waypoints):
        if i % step != 0 and i != len(waypoints) - 1:
            continue
        sphere_path = f"{base_path}_wp{i}"
        sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(sphere_path))
        sphere.CreateRadiusAttr(float(radius))
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(float(wp[0]), float(wp[1]), float(wp[2])))
        UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])


def add_formation_visuals(
    stage,
    center_path: list[np.ndarray],
    team_paths: list[list[np.ndarray]],
    formation_type: str,
) -> None:
    """Add USD visuals for formation paths and waypoints."""
    root = "/World/FormationPlan"

    # Center path
    _create_path_curve(stage, f"{root}/CenterPath", center_path, CENTER_COLOR, width=2.0)
    _create_waypoint_markers(stage, f"{root}/Center", center_path, CENTER_COLOR, radius=3.0, step=10)

    # Per-member paths
    for i, path in enumerate(team_paths):
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        _create_path_curve(stage, f"{root}/Member{i}_Path", path, color, width=1.5)
        _create_waypoint_markers(stage, f"{root}/Member{i}", path, color, radius=2.0, step=10)

    print(f"[VIS] Added formation visuals under {root}: {len(team_paths)} member paths")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Formation Mission Runner")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--formation", type=str, default="v_shape",
                        choices=list(FORMATION_NAMES.keys()))
    parser.add_argument("--spacing", type=float, default=40.0)
    parser.add_argument("--start", type=float, nargs=3, default=[-800, -600, 80])
    parser.add_argument("--goal", type=float, nargs=3, default=[800, 600, 80])
    parser.add_argument("--speed", type=float, default=None,
                        help="Cruise speed in scene units/s (default = each platform cruise speed)")
    parser.add_argument("--max-altitude", type=float, default=200.0)
    parser.add_argument("--time-scale", type=float, default=45.0)
    parser.add_argument("--run-seconds", type=float, default=-1,
                        help="Max tactical seconds (-1 = unlimited, run until window closed)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Disable path visualization")
    parser.add_argument("--planner", type=str, default="mppi",
                        choices=["mppi", "hybrid_astar"],
                        help="Path planner: 'mppi' (default) or 'hybrid_astar'")
    # MPPI-specific arguments
    parser.add_argument("--mppi-samples", type=int, default=512,
                        help="MPPI samples per iteration")
    parser.add_argument("--mppi-iters", type=int, default=5,
                        help="MPPI refinement iterations")
    parser.add_argument("--mppi-horizon", type=int, default=50,
                        help="MPPI planning horizon")
    parser.add_argument("--mppi-temperature", type=float, default=1.0,
                        help="MPPI softmin temperature")
    parser.add_argument("--member-map", type=str, default=None,
                        help="Role-to-platform assignments, e.g. "
                        "'leader:0,left_wing_1:2,right_wing_1:1'. "
                        "Use --show-roles to list available role names.")
    parser.add_argument("--show-roles", action="store_true",
                        help="Print available role names for the formation and exit.")
    args = parser.parse_args()

    # Show available roles and exit
    if args.show_roles:
        roles = get_formation_roles(args.formation, args.count)
        print(f"Formation: {FORMATION_NAMES.get(args.formation, args.formation)}")
        print(f"Team size: {args.count}")
        print(f"Available role names (index -> role):")
        for i, role in enumerate(roles):
            print(f"  team_paths[{i}] -> {role}")
        print()
        print("Example --member-map usage:")
        if len(roles) >= 2:
            print(f"  --member-map '{roles[0]}:0,{roles[1]}:1'")
        return

    # ------------------------------------------------------------------
    # 1. Plan the formation mission
    # ------------------------------------------------------------------
    print(f"[PLAN] Formation: {FORMATION_NAMES.get(args.formation, args.formation)}, "
        f"members: {args.count}")
    print(f"[PLAN] Start: {tuple(args.start)}, Goal: {tuple(args.goal)}")

    obstacles = build_obstacles()

    # Parse member map
    member_assignments = None
    if args.member_map:
        member_assignments = {}
        for part in args.member_map.split(","):
            part = part.strip()
            if ":" in part:
                role, idx = part.split(":", 1)
                member_assignments[role.strip()] = int(idx.strip())
            elif "=" in part:
                role, idx = part.split("=", 1)
                member_assignments[role.strip()] = int(idx.strip())
        if member_assignments:
            print(f"[PLAN] Member assignments: {member_assignments}")

    if args.planner == "mppi" and _HAS_MPPI:
        planner_config = MPPIConfig(
            max_altitude=args.max_altitude,
            num_samples=args.mppi_samples,
            num_iterations=args.mppi_iters,
            horizon=args.mppi_horizon,
            temperature=args.mppi_temperature,
        )
        planner = MPPIPlanner(obstacles=obstacles, config=planner_config)
        print(f"[PLAN] Using MPPI planner "
              f"(samples={args.mppi_samples}, iters={args.mppi_iters}, "
              f"horizon={args.mppi_horizon})")
    else:
        if args.planner == "mppi":
            print("[PLAN] MPPI not available, falling back to Hybrid A*")
        planner_config = PlannerConfig(
            max_altitude=args.max_altitude,
        )
        planner = HybridAStarPlanner(obstacles=obstacles, config=planner_config)
        print("[PLAN] Using Hybrid A* planner")

    start = np.array(args.start, dtype=float)
    goal = np.array(args.goal, dtype=float)
    center_path = planner.plan(start=start, goal=goal, verbose=True)

    if center_path is None:
        print("[FATAL] No path found!")
        simulation_app.close()
        return

    offsets = get_formation_offsets(args.formation, args.count, args.spacing)
    team_paths = distribute_team_waypoints(center_path, offsets)

    # Apply member assignments: reorder team_paths per --member-map
    if member_assignments:
        roles = get_formation_roles(args.formation, args.count)
        team_paths = _apply_member_assignments(
            team_paths, roles, member_assignments
        )

    # Lift all waypoints above terrain surface
    print("[PLAN] Lifting waypoints above terrain surface...")
    center_path = lift_waypoints_above_terrain(center_path, clearance=15.0)
    for i in range(len(team_paths)):
        team_paths[i] = lift_waypoints_above_terrain(team_paths[i], clearance=15.0)
    # Update start/goal altitudes to match lifted values
    args.start[2] = float(center_path[0][2])
    args.goal[2] = float(center_path[-1][2])

    print(f"[PLAN] Center path: {len(center_path)} waypoints")
    roles = get_formation_roles(args.formation, args.count)
    for i, path in enumerate(team_paths):
        role_label = roles[i] if i < len(roles) else f"member_{i + 1}"
        print(f"[PLAN]   [{role_label}] team_paths[{i}]: {len(path)} waypoints, "
            f"start={path[0]}, end={path[-1]}")

    # ------------------------------------------------------------------
    # 2. Build scene
    # ------------------------------------------------------------------
    stage = omni.usd.get_context().get_stage()
    if not stage:
        stage = Usd.Stage.CreateInMemory()
        omni.usd.get_context().set_stage(stage)

    scene_config = air_combat_scene.DEFAULT_AIR_COMBAT_CONFIG
    # Reduce aircraft count to just our formation members
    scene_config = dict(scene_config)
    scene_config["aircraft"] = dict(scene_config.get("aircraft", {}))
    scene_config["aircraft"]["terrain_clearance_m"] = 120.0

    scene = air_combat_scene.create_scene(stage, scene_config)
    simulation_app.update()

    # ------------------------------------------------------------------
    # 3. Replace platform references with waypoint followers
    # ------------------------------------------------------------------
    # Pick the first N platforms to be our formation members
    platforms = scene.platforms[:args.count]
    if len(platforms) < args.count:
        print(f"[WARN] Scene only has {len(platforms)} platforms, "
            f"but formation needs {args.count}. Using available platforms.")

    for i, platform in enumerate(platforms):
        if i >= len(team_paths):
            break
        path_speed = (
            float(args.speed)
            if args.speed is not None
            else float(platform.spec.cruise_speed_mps) / float(scene.meters_per_unit)
        )
        platform.motion_model.max_speed = max(
            float(platform.motion_model.max_speed),
            path_speed * 1.15,
        )
        platform.reference_fn = make_path_tracker(
            team_paths[i],
            platform,
            lookahead_units=max(8.0, path_speed * 12.0),
        )
        # Teleport to start position
        platform.motion_model.state.position = team_paths[i][0].copy()
        platform.motion_model.state.velocity = np.zeros(3, dtype=float)
        print(
            f"[SETUP] {platform.entity_id} assigned to formation member {i + 1} "
            f"speed={path_speed:.2f} units/s max={platform.motion_model.max_speed:.2f}"
        )

    # Disable other platforms (keep them static at their spawn)
    for platform in scene.platforms[args.count:]:
        platform.reference_fn = lambda t, p=platform: p.position.copy()

    # Force one update at t=0 to sync USD prim positions with motion states
    scene.update(tactical_dt=0.001, tactical_time_s=0.0)
    simulation_app.update()
    print("[SETUP] Platform positions synced to waypoint starts.")

    # ------------------------------------------------------------------
    # 4. Visualization
    # ------------------------------------------------------------------
    if not args.no_vis:
        add_formation_visuals(stage, center_path, team_paths, args.formation)
        simulation_app.update()

    # Camera
    from isaacsim.core.utils.viewports import set_camera_view
    path_center = np.mean(np.array(center_path, dtype=float), axis=0)
    eye = [
        float(path_center[0] + scene.map_size_units * 0.28),
        float(path_center[1] - scene.map_size_units * 0.34),
        float(max(path_center[2] + 260.0, scene.terrain_visual_height_units + 180.0)),
    ]
    target = [float(path_center[0]), float(path_center[1]), float(path_center[2])]
    set_camera_view(eye=eye, target=target)
    simulation_app.update()

    # ------------------------------------------------------------------
    # 5. Run simulation
    # ------------------------------------------------------------------
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    import time
    start_time = time.time()
    frame_count = 0
    tactical_time_s = 0.0
    time_scale = float(args.time_scale)

    speed_msg = (
        f"{args.speed:.2f} units/s ({args.speed * scene.meters_per_unit:.0f} m/s)"
        if args.speed is not None
        else "platform cruise speeds"
    )
    print(f"[RUN] Time scale: {time_scale}x, cruise speed: {speed_msg}")
    if args.run_seconds > 0:
        print(f"[RUN] Will stop after {args.run_seconds:.0f}s tactical time.")
    else:
        print("[RUN] Running until window is closed (Ctrl+C to stop).")

    try:
        while simulation_app.is_running():
            tactical_dt = DT * time_scale
            tactical_time_s += tactical_dt
            scene.update(tactical_dt=tactical_dt, tactical_time_s=tactical_time_s)
            simulation_app.update()

            if frame_count % 300 == 0:
                elapsed = time.time() - start_time
                n = min(len(platforms), len(team_paths))
                arrived = sum(
                    1 for i in range(n)
                    if np.linalg.norm(platforms[i].position - team_paths[i][-1]) < 15.0
                )
                print(f"[RUN] wall={elapsed:.0f}s tactical={tactical_time_s:.0f}s "
                    f"frame={frame_count} arrived={arrived}/{n}")
                for i, p2 in enumerate(platforms):
                    pos = p2.position
                    ref = p2.reference_fn(tactical_time_s)
                    dist = np.linalg.norm(pos - ref)
                    print(f"  {p2.entity_id}: pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.1f}) "
                          f"ref=({ref[0]:.0f},{ref[1]:.0f},{ref[2]:.1f}) dist={dist:.1f}")
            n = min(len(platforms), len(team_paths))
            all_arrived = n > 0 and all(
                np.linalg.norm(platforms[i].position - team_paths[i][-1]) < 15.0
                for i in range(n)
            )
            if all_arrived and frame_count > 10:
                print(f"[RUN] All aircraft arrived. tactical={tactical_time_s:.0f}s")
                if tactical_time_s > 30:
                    import time as _time
                    print("[RUN] Holding 3 wall-seconds for observation...")
                    _time.sleep(3)
                    break

    except KeyboardInterrupt:
        print("[RUN] Interrupted by user.")
    finally:
        timeline.stop()
        simulation_app.close()
        print("[RUN] Done.")


if __name__ == "__main__":
    main()
