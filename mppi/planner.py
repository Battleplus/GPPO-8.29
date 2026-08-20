"""MPPI Formation Planner -- main entry point.

Uses Model Predictive Path Integral (MPPI) control for trajectory
optimisation with nonlinear aircraft dynamics.  Supports multi-agent
formation planning with obstacle avoidance.

Usage:
    python -m ql.scripts.planner

Or in code:
    from ql.scripts.planner import plan_formation_mission
    result = plan_formation_mission(
        team_count=4,
        start=(-800, -600, 8.0),
        goal=(800, 600, 8.0),
        formation="v_shape",
    )
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

# Support both `python -m ql.scripts.planner` and `python scripts/planner.py`
try:
    from .formation import (
        FORMATION_NAMES,
        distribute_team_waypoints,
        get_formation_offsets,
        get_formation_roles,
    )
    from .mppi import MPPIConfig, MPPIPlanner
    from .obstacles import CylindricalObstacle, build_obstacles, is_position_blocked
except ImportError:
    from formation import (  # type: ignore[no-redef]
        FORMATION_NAMES,
        distribute_team_waypoints,
        get_formation_offsets,
        get_formation_roles,
    )
    from mppi import MPPIConfig, MPPIPlanner  # type: ignore[no-redef]
    from obstacles import CylindricalObstacle, build_obstacles, is_position_blocked  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Plan result
# ---------------------------------------------------------------------------


@dataclass
class FormationPlanResult:
    """Result of a formation planning mission."""

    team_count: int
    formation_type: str
    formation_offsets: list[np.ndarray]
    center_path: list[np.ndarray]
    team_paths: list[list[np.ndarray]]  # [output_idx][wp_idx] = [x,y,z]
    success: bool
    planner_stats: dict = field(default_factory=dict)
    formation_roles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "team_count": self.team_count,
            "formation_type": self.formation_type,
            "formation_offsets": [o.tolist() for o in self.formation_offsets],
            "center_path": [wp.tolist() for wp in self.center_path],
            "team_paths": [
                [wp.tolist() for wp in path] for path in self.team_paths
            ],
            "success": self.success,
            "stats": self.planner_stats,
            "formation_roles": self.formation_roles,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        fmt_name = FORMATION_NAMES.get(self.formation_type, self.formation_type)
        lines = [
            f"===== Formation Plan Result =====",
            f"  Formation:   {fmt_name}",
            f"  Team size:   {self.team_count}",
            f"  Success:     {'YES' if self.success else 'NO'}",
            f"  Center WPs:  {len(self.center_path)}",
        ]
        stats = self.planner_stats
        if stats:
            if "path_length_units" in stats:
                lines.append(
                    f"  Path length: {stats['path_length_units']:.0f} units"
                    f" ({stats.get('path_length_km', 0):.1f} km)"
                )
        lines.append("")
        lines.append("  Per-member waypoints:")
        roles = self.formation_roles if self.formation_roles else [
            f"member_{i + 1}" for i in range(len(self.team_paths))
        ]
        for i, path in enumerate(self.team_paths):
            role_label = roles[i] if i < len(roles) else f"member_{i + 1}"
            lines.append(f"    [{role_label}] -> team_paths[{i}]: {len(path)} waypoints")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formation MPPI Planner
# ---------------------------------------------------------------------------


class FormationMPPIPlanner:
    """MPPI planner with fixed formation support.

    Workflow:
      1. Compute formation offsets from team_count and formation_type.
      2. Plan a center path using MPPI (nonlinear dynamics + obstacle cost).
      3. Distribute center waypoints to each team member via offsets.

    The MPPI planner samples hundreds of candidate trajectories in parallel,
    rolls them through a kinematic aircraft model, and selects the optimal
    path via importance-sampling re-weighting -- producing dynamically
    feasible, obstacle-free paths.
    """

    def __init__(
        self,
        map_size_units: float = 3000.0,
        meters_per_unit: float = 100.0,
        terrain_vertical_exaggeration: float = 10.0,
        obstacles: list[CylindricalObstacle] | None = None,
        planner_config: MPPIConfig | None = None,
    ):
        self.map_size_units = map_size_units
        self.meters_per_unit = meters_per_unit
        self.terrain_vertical_exaggeration = terrain_vertical_exaggeration
        self.mountain_height_units = (
            1500.0 / meters_per_unit * terrain_vertical_exaggeration
        )

        if obstacles is None:
            obstacles = build_obstacles()
        self.obstacles = obstacles

        if planner_config is None:
            planner_config = MPPIConfig(
                map_size_units=map_size_units,
                map_origin=(-map_size_units * 0.5, -map_size_units * 0.5),
                max_altitude=max(200.0, self.mountain_height_units + 50.0),
            )
        self.planner_config = planner_config
        self._planner = MPPIPlanner(
            obstacles=self.obstacles, config=planner_config
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        team_count: int,
        start: np.ndarray,
        goal: np.ndarray,
        formation: str = "v_shape",
        spacing: float = 40.0,
        depth_spacing: float | None = None,
        cruise_altitude: float | None = None,
        member_assignments: dict[str, int] | None = None,
        verbose: bool = True,
    ) -> FormationPlanResult:
        """Plan a formation mission with MPPI.

        Args:
            team_count: Number of members in the formation.
            start: Formation center start [x, y, z] in scene units.
            goal: Formation center goal [x, y, z] in scene units.
            formation: Formation type (see FORMATION_NAMES).
            spacing: Lateral spacing between members (scene units).
            depth_spacing: Depth spacing (default = spacing).
            cruise_altitude: If set, override start[2] and goal[2].
            member_assignments: Optional dict mapping role name -> output
                index in team_paths.  For example
                ``{"leader": 2, "left_wing_1": 0, "right_wing_1": 1}``
                puts the leader's path at ``team_paths[2]`` so that
                ``scene.platforms[2]`` flies as leader.  Roles not listed
                keep their default positions.  Use
                :func:`get_formation_roles` to see available role names.
            verbose: Print planning progress.

        Returns:
            FormationPlanResult with ``team_paths`` reordered per
            ``member_assignments``.
        """
        # Formation offsets and roles
        offsets = get_formation_offsets(
            formation_type=formation,
            team_size=team_count,
            spacing=spacing,
            depth_spacing=depth_spacing,
        )
        roles = get_formation_roles(formation, team_count)

        # Unify cruise altitude
        if cruise_altitude is not None:
            start = np.array(
                [start[0], start[1], cruise_altitude], dtype=float
            )
            goal = np.array(
                [goal[0], goal[1], cruise_altitude], dtype=float
            )

        if verbose:
            fmt_name = FORMATION_NAMES.get(formation, formation)
            print(f"[Planner] Formation: {fmt_name}, members: {team_count}")
            print(
                f"[Planner] Start: ({start[0]:.0f}, {start[1]:.0f}, "
                f"{start[2]:.1f})"
            )
            print(
                f"[Planner] Goal:  ({goal[0]:.0f}, {goal[1]:.0f}, "
                f"{goal[2]:.1f})"
            )
            print(f"[Planner] Spacing: {spacing:.0f} units")
            print(f"[Planner] Obstacles (mountains): {len(self.obstacles)}")
            print(
                f"[Planner] MPPI: {self.planner_config.num_samples} samples x "
                f"{self.planner_config.num_iterations} iters, "
                f"horizon={self.planner_config.horizon}"
            )

        # MPPI for center path
        center_path = self._planner.plan(
            start=start, goal=goal, verbose=verbose
        )
        stats: dict = {
            "algorithm": "MPPI",
            "num_samples": self.planner_config.num_samples,
            "num_iterations": self.planner_config.num_iterations,
            "horizon": self.planner_config.horizon,
        }

        if center_path is None:
            return FormationPlanResult(
                team_count=team_count,
                formation_type=formation,
                formation_offsets=offsets,
                center_path=[],
                team_paths=[],
                success=False,
                planner_stats=stats,
                formation_roles=roles,
            )

        # Path length statistics
        path_length = 0.0
        for i in range(1, len(center_path)):
            path_length += float(
                np.linalg.norm(center_path[i] - center_path[i - 1])
            )
        stats["path_length_units"] = path_length
        stats["path_length_km"] = (
            path_length * self.meters_per_unit / 1000.0
        )

        # B-spline smoothing (MPPI trajectories are already smooth,
        # but a light pass removes residual sampling noise)
        center_path = self._smooth_path(center_path)

        # Distribute to team members
        team_paths = distribute_team_waypoints(center_path, offsets)

        # Apply member assignments: reorder team_paths so that
        # team_paths[platform_idx] gets the path for the assigned role.
        if member_assignments:
            team_paths = self._apply_member_assignments(
                team_paths, roles, member_assignments, verbose
            )

        blocked_waypoints = self._count_blocked_waypoints(team_paths)
        stats["blocked_team_waypoints"] = blocked_waypoints

        if blocked_waypoints > 0 and verbose:
            print(
                f"[Planner][WARN] Formation waypoints inside obstacles: "
                f"{blocked_waypoints}. Increase spacing clearance or altitude."
            )

        # Compute smoothness metric
        curvature = self._path_curvature(center_path)
        stats["max_curvature"] = curvature

        if verbose:
            print(
                f"[Planner] Center path: {len(center_path)} waypoints, "
                f"total {path_length:.0f} units "
                f"({stats['path_length_km']:.1f} km), "
                f"max curvature={curvature:.4f}"
            )

        return FormationPlanResult(
            team_count=team_count,
            formation_type=formation,
            formation_offsets=offsets,
            center_path=center_path,
            team_paths=team_paths,
            success=blocked_waypoints == 0,
            planner_stats=stats,
            formation_roles=roles,
        )

    def plan_multiple_teams(
        self, teams: list[dict], verbose: bool = True
    ) -> list[FormationPlanResult]:
        """Plan multiple formations with MPPI.

        Args:
            teams: List of team specs, each a dict with keys:
                   "start", "goal", "count", "formation",
                   and optionally "spacing", "depth_spacing",
                   "cruise_altitude", "member_assignments".
            verbose: Print progress.

        Returns:
            List of FormationPlanResult, one per team.
        """
        results: list[FormationPlanResult] = []
        for i, team in enumerate(teams):
            if verbose:
                print(f"\n--- Team {i + 1}/{len(teams)} ---")
            result = self.plan(
                team_count=team.get("count", 4),
                start=np.array(team["start"], dtype=float),
                goal=np.array(team["goal"], dtype=float),
                formation=team.get("formation", "v_shape"),
                spacing=team.get("spacing", 40.0),
                depth_spacing=team.get("depth_spacing"),
                cruise_altitude=team.get("cruise_altitude"),
                member_assignments=team.get("member_assignments"),
                verbose=verbose,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_path(
        path: list[np.ndarray], window: int = 3
    ) -> list[np.ndarray]:
        """Simple moving-average path smoothing."""
        if len(path) < 3 or window < 2:
            return path
        smoothed: list[np.ndarray] = [path[0]]
        half = window // 2
        for i in range(1, len(path) - 1):
            lo = max(0, i - half)
            hi = min(len(path), i + half + 1)
            avg = np.mean(path[lo:hi], axis=0)
            avg[2] = path[i][2]  # preserve original altitude
            smoothed.append(avg)
        smoothed.append(path[-1])
        return smoothed

    @staticmethod
    def _path_curvature(path: list[np.ndarray]) -> float:
        """Compute maximum path curvature (rad/unit) for smoothness eval."""
        if len(path) < 3:
            return 0.0
        max_k = 0.0
        for i in range(1, len(path) - 1):
            a = path[i] - path[i - 1]
            b = path[i + 1] - path[i]
            cross = float(np.linalg.norm(np.cross(a[:2], b[:2])))
            dot = float(np.dot(a[:2], b[:2]))
            # Menger curvature: 4A / (|a|*|b|*|a+b|) ~ 2*sin(theta) / chord
            denom = float(np.linalg.norm(a[:2]))
            denom *= float(np.linalg.norm(b[:2]))
            if denom > 1e-9:
                max_k = max(max_k, cross / denom)
        return max_k

    def _count_blocked_waypoints(
        self,
        team_paths: list[list[np.ndarray]],
    ) -> int:
        blocked = 0
        for path in team_paths:
            for wp in path:
                if is_position_blocked(
                    wp[:2],
                    float(wp[2]),
                    self.obstacles,
                    self.planner_config.obstacle_clearance,
                ):
                    blocked += 1
        return blocked

    @staticmethod
    def _apply_member_assignments(
        team_paths: list[list[np.ndarray]],
        roles: list[str],
        member_assignments: dict[str, int],
        verbose: bool = False,
    ) -> list[list[np.ndarray]]:
        """Reorder team_paths so team_paths[platform_idx] = role's path.

        Args:
            team_paths: Original paths, index-aligned with roles.
            roles: Role names, same length as team_paths.
            member_assignments: Dict mapping role_name -> output_index.

        Returns:
            Reordered team_paths.
        """
        n = len(team_paths)
        # Start with copies of original paths as default fallback
        reordered = [list(p) for p in team_paths]

        # Build a lookup: role_name -> original team_paths index
        role_to_src_idx: dict[str, int] = {}
        for idx, role in enumerate(roles):
            role_to_src_idx[role] = idx

        if verbose:
            print("[Planner] Member assignments:")

        for role, dst_idx in member_assignments.items():
            if role not in role_to_src_idx:
                if verbose:
                    print(f"  [WARN] Unknown role '{role}', skipping. "
                          f"Available: {list(role_to_src_idx.keys())}")
                continue
            if not (0 <= dst_idx < n):
                if verbose:
                    print(f"  [WARN] Output index {dst_idx} out of range "
                          f"[0, {n}), skipping role '{role}'")
                continue
            src_idx = role_to_src_idx[role]
            reordered[dst_idx] = list(team_paths[src_idx])
            if verbose:
                print(f"  {role} -> team_paths[{dst_idx}] "
                      f"(was team_paths[{src_idx}])")

        return reordered


# ==========================================================================
# Backward-compatible alias
# ==========================================================================

FormationHybridAStarPlanner = FormationMPPIPlanner


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def plan_formation_mission(
    team_count: int = 4,
    start: tuple[float, float, float] = (-800.0, -600.0, 8.0),
    goal: tuple[float, float, float] = (800.0, 600.0, 8.0),
    formation: str = "v_shape",
    spacing: float = 40.0,
    **kwargs,
) -> FormationPlanResult:
    """One-stop formation planning via MPPI.

    Args:
        team_count: Number of formation members.
        start: (x, y, z) start in scene units.
        goal: (x, y, z) goal in scene units.
        formation: Formation type.
        spacing: Member spacing.

    Returns:
        FormationPlanResult.
    """
    planner = FormationMPPIPlanner()
    return planner.plan(
        team_count=team_count,
        start=np.array(start, dtype=float),
        goal=np.array(goal, dtype=float),
        formation=formation,
        spacing=spacing,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MPPI Formation Planner"
    )
    parser.add_argument(
        "--teams", type=int, default=1, help="Number of formations"
    )
    parser.add_argument(
        "--count", type=int, default=4, help="Members per team"
    )
    parser.add_argument(
        "--formation", type=str, default="v_shape",
        choices=list(FORMATION_NAMES.keys()), help="Formation type"
    )
    parser.add_argument(
        "--spacing", type=float, default=40.0, help="Member spacing (units)"
    )
    parser.add_argument(
        "--start", type=float, nargs=3, default=[-800, -600, 8],
        help="Start position: x y z"
    )
    parser.add_argument(
        "--goal", type=float, nargs=3, default=[800, 600, 8],
        help="Goal position: x y z"
    )
    parser.add_argument(
        "--altitude", type=float, default=None, help="Cruise altitude"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path"
    )
    parser.add_argument(
        "--map-size", type=float, default=3000.0, help="Map size in units"
    )
    parser.add_argument(
        "--num-samples", type=int, default=512,
        help="MPPI samples per iteration"
    )
    parser.add_argument(
        "--num-iters", type=int, default=5,
        help="MPPI refinement iterations"
    )
    parser.add_argument(
        "--horizon", type=int, default=50, help="MPPI planning horizon"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="MPPI softmin temperature"
    )
    parser.add_argument(
        "--max-altitude", type=float, default=200.0,
        help="Maximum altitude in scene units"
    )
    parser.add_argument(
        "--speed", type=float, default=80.0,
        help="Cruise speed (units/s)"
    )
    parser.add_argument(
        "--turn-std", type=float, default=20.0,
        help="Turn rate noise std (degrees)"
    )

    args = parser.parse_args()

    config = MPPIConfig(
        map_size_units=args.map_size,
        num_samples=args.num_samples,
        num_iterations=args.num_iters,
        horizon=args.horizon,
        temperature=args.temperature,
        max_altitude=args.max_altitude,
        cruise_speed=args.speed,
        turn_rate_std=math.radians(args.turn_std),
    )

    planner = FormationMPPIPlanner(
        map_size_units=args.map_size,
        planner_config=config,
    )

    results: list[FormationPlanResult] = []

    if args.teams == 1:
        result = planner.plan(
            team_count=args.count,
            start=np.array(args.start, dtype=float),
            goal=np.array(args.goal, dtype=float),
            formation=args.formation,
            spacing=args.spacing,
            cruise_altitude=args.altitude,
            verbose=True,
        )
        results.append(result)
        print(result.summary())
    else:
        # Multi-team: distribute starts/goals around a circle
        teams = []
        for i in range(args.teams):
            angle = 2.0 * math.pi * i / args.teams
            r = 600.0
            teams.append({
                "start": [
                    r * math.cos(angle),
                    r * math.sin(angle),
                    args.altitude or 8.0,
                ],
                "goal": [
                    -r * math.cos(angle),
                    -r * math.sin(angle),
                    args.altitude or 8.0,
                ],
                "count": args.count,
                "formation": args.formation,
                "spacing": args.spacing,
            })
        results = planner.plan_multiple_teams(teams, verbose=True)
        for i, r in enumerate(results):
            print(f"\nTeam {i + 1}:")
            print(r.summary())

    # Save output
    if args.output:
        all_results = [r.to_dict() for r in results]
        out_path = args.output
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            if len(all_results) == 1:
                json.dump(all_results[0], f, indent=2, ensure_ascii=False)
            else:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {out_path}")
