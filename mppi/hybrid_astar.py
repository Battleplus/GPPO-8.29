"""Hybrid A* 3D path planner for air combat scene.

Features:
  - 3D state space (x, y, z, heading): continuous position, discrete heading
  - Motion primitives: straight / turn left / turn right / climb / descend
  - Cylindrical obstacle collision checking (mountains)
  - Grid-discretized visited set with continuous coordinate progression
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

try:
    from .obstacles import CylindricalObstacle, is_position_blocked
except ImportError:
    from obstacles import CylindricalObstacle, is_position_blocked  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HybridAStarNode:
    """Hybrid A* search node."""

    x: float
    y: float
    z: float
    heading_idx: int   # discrete heading index in [0, N_HEADINGS)
    g: float = 0.0     # actual cost
    h: float = 0.0     # heuristic cost
    parent: HybridAStarNode | None = None
    action: int = 0    # action that reached this node

    @property
    def f(self) -> float:
        return self.g + self.h

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def __lt__(self, other: HybridAStarNode) -> bool:
        return self.f < other.f


@dataclass
class PlannerConfig:
    """Hybrid A* planner configuration."""

    # Map
    map_size_units: float = 3000.0
    map_origin: tuple[float, float] = (-1500.0, -1500.0)

    # Discretization
    xy_resolution: float = 15.0    # XY grid resolution (units)
    z_resolution: float = 3.0      # Z grid resolution (units)
    num_headings: int = 16         # discrete heading bins

    # Motion primitives
    step_size: float = 30.0        # forward step distance (units)
    turn_angle_deg: float = 22.5   # turn angle per primitive (degrees)
    climb_rate: float = 2.0        # altitude change per step (units)

    # Cost weights
    turn_penalty: float = 8.0
    climb_penalty: float = 3.0
    descend_penalty: float = 1.0

    # Safety
    obstacle_clearance: float = 3.0
    min_altitude: float = 0.5
    max_altitude: float = 25.0

    # Search limits
    max_iterations: int = 50000
    early_stop_threshold: float = 15.0   # half of default step_size (30)


# ---------------------------------------------------------------------------
# Motion primitives
# ---------------------------------------------------------------------------

ACTION_STRAIGHT = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_CLIMB = 3
ACTION_DESCEND = 4

ACTION_NAMES = {
    ACTION_STRAIGHT: "straight",
    ACTION_LEFT: "turn_left",
    ACTION_RIGHT: "turn_right",
    ACTION_CLIMB: "climb",
    ACTION_DESCEND: "descend",
}


def _wrap_heading_idx(idx: int, num_headings: int) -> int:
    return idx % num_headings


def _heading_to_radians(heading_idx: int, num_headings: int) -> float:
    """Discrete heading index -> radians (0=east, pi/2=north)."""
    return 2.0 * math.pi * heading_idx / num_headings


def _heading_vector(heading_idx: int, num_headings: int) -> np.ndarray:
    """Discrete heading -> unit direction vector [dx, dy]."""
    rad = _heading_to_radians(heading_idx, num_headings)
    return np.array([math.cos(rad), math.sin(rad)], dtype=float)


def _closest_heading_index(direction_xy: np.ndarray, num_headings: int) -> int:
    """Continuous direction vector -> nearest discrete heading index."""
    rad = math.atan2(float(direction_xy[1]), float(direction_xy[0]))
    if rad < 0:
        rad += 2.0 * math.pi
    idx = int(round(rad / (2.0 * math.pi) * num_headings)) % num_headings
    return idx


# ---------------------------------------------------------------------------
# Hybrid A* Planner
# ---------------------------------------------------------------------------


class HybridAStarPlanner:
    """Hybrid A* 3D path planner with mountain obstacle avoidance."""

    def __init__(
        self,
        obstacles: list[CylindricalObstacle],
        config: PlannerConfig | None = None,
    ):
        self.obstacles = obstacles
        self.config = config or PlannerConfig()
        # turn step = 45 degrees when N=16 (2 indices)
        self._heading_step = max(1, self.config.num_headings // 8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        verbose: bool = False,
    ) -> list[np.ndarray] | None:
        """Plan a path from start to goal.

        Args:
            start: [x, y, z] start position.
            goal: [x, y, z] goal position.
            verbose: Print search progress.

        Returns:
            List of waypoints [start, ..., goal], or None if unreachable.
        """
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)

        # Validate endpoints
        blocked = _check_blocked(
            start, self.obstacles, self.config.obstacle_clearance
        )
        if blocked:
            if verbose:
                print(
                    f"[HybridA*] Start {start} blocked by: "
                    + ", ".join(
                        f"{b.name}(r={b.radius:.0f},h={b.height:.1f},"
                        f"dist={b._block_dist:.0f})"
                        for b in blocked
                    )
                )
            return None
        blocked = _check_blocked(
            goal, self.obstacles, self.config.obstacle_clearance
        )
        if blocked:
            if verbose:
                print(
                    f"[HybridA*] Goal {goal} blocked by: "
                    + ", ".join(
                        f"{b.name}(r={b.radius:.0f},h={b.height:.1f},"
                        f"dist={b._block_dist:.0f})"
                        for b in blocked
                    )
                )
            return None

        # Initial heading: toward goal
        delta = goal[:2] - start[:2]
        start_heading = _closest_heading_index(delta, self.config.num_headings)

        start_node = HybridAStarNode(
            x=float(start[0]),
            y=float(start[1]),
            z=float(start[2]),
            heading_idx=start_heading,
            g=0.0,
            h=self._heuristic(start, goal),
        )

        return self._search(start_node, goal, verbose=verbose)

    # ------------------------------------------------------------------
    # Search core
    # ------------------------------------------------------------------

    def _search(
        self,
        start_node: HybridAStarNode,
        goal: np.ndarray,
        verbose: bool = False,
    ) -> list[np.ndarray] | None:
        cfg = self.config
        open_heap: list[HybridAStarNode] = []
        heapq.heappush(open_heap, start_node)

        # visited: (grid_x, grid_y, grid_z, heading_idx) -> best_g
        visited: dict[tuple[int, int, int, int], float] = {}
        start_key = self._grid_key(start_node)
        visited[start_key] = 0.0

        iterations = 0
        best_node: HybridAStarNode | None = None
        best_dist = float("inf")

        while open_heap and iterations < cfg.max_iterations:
            iterations += 1
            current = heapq.heappop(open_heap)

            # Check goal proximity
            dist_to_goal = float(np.linalg.norm(current.position - goal))
            if dist_to_goal < best_dist:
                best_dist = dist_to_goal
                best_node = current

            if dist_to_goal < cfg.early_stop_threshold:
                if verbose:
                    print(
                        f"[HybridA*] Path found! iterations={iterations}, "
                        f"cost={current.g:.1f}"
                    )
                return self._reconstruct_path(current, goal)

            # Expand actions
            for _action, child in self._expand(current, goal):
                child_key = self._grid_key(child)

                # Collision check
                if is_position_blocked(
                    np.array([child.x, child.y]),
                    child.z,
                    self.obstacles,
                    cfg.obstacle_clearance,
                ):
                    continue

                # Bounds check
                if not self._in_bounds(child):
                    continue

                # Visited pruning
                old_g = visited.get(child_key, float("inf"))
                if child.g >= old_g:
                    continue
                visited[child_key] = child.g

                heapq.heappush(open_heap, child)

        # Fallback: return best approximate path
        if best_node is not None and best_dist < cfg.early_stop_threshold * 10.0:
            if verbose:
                print(
                    f"[HybridA*] Approximate path (dist={best_dist:.1f}), "
                    f"iterations={iterations}"
                )
            return self._reconstruct_path(best_node, goal)

        if verbose:
            print(
                f"[HybridA*] No path found! iterations={iterations}, "
                f"best_dist={best_dist:.1f}"
            )
        return None

    # ------------------------------------------------------------------
    # Action expansion
    # ------------------------------------------------------------------

    def _expand(
        self, node: HybridAStarNode, goal: np.ndarray
    ) -> list[tuple[int, HybridAStarNode]]:
        cfg = self.config
        N = cfg.num_headings
        children: list[tuple[int, HybridAStarNode]] = []

        heading_rad = _heading_to_radians(node.heading_idx, N)
        dir_vec = np.array(
            [math.cos(heading_rad), math.sin(heading_rad)], dtype=float
        )
        step = cfg.step_size
        turn_step = self._heading_step

        # 1) Straight
        children.append(
            self._make_child(node, goal, ACTION_STRAIGHT,
                             dir_vec * step, 0, 0, dz=0)
        )

        # 2) Turn left
        left_idx = _wrap_heading_idx(node.heading_idx - turn_step, N)
        left_rad = _heading_to_radians(left_idx, N)
        left_dir = np.array(
            [math.cos(left_rad), math.sin(left_rad)], dtype=float
        )
        children.append(
            self._make_child(node, goal, ACTION_LEFT,
                             left_dir * step,
                             left_idx - node.heading_idx,
                             cfg.turn_penalty, dz=0)
        )

        # 3) Turn right
        right_idx = _wrap_heading_idx(node.heading_idx + turn_step, N)
        right_rad = _heading_to_radians(right_idx, N)
        right_dir = np.array(
            [math.cos(right_rad), math.sin(right_rad)], dtype=float
        )
        children.append(
            self._make_child(node, goal, ACTION_RIGHT,
                             right_dir * step,
                             right_idx - node.heading_idx,
                             cfg.turn_penalty, dz=0)
        )

        # 4) Climb (move forward + increase altitude)
        children.append(
            self._make_child(node, goal, ACTION_CLIMB,
                             dir_vec * step, 0,
                             cfg.climb_penalty, dz=cfg.climb_rate)
        )

        # 5) Descend (move forward + decrease altitude)
        children.append(
            self._make_child(node, goal, ACTION_DESCEND,
                             dir_vec * step, 0,
                             cfg.descend_penalty, dz=-cfg.climb_rate)
        )

        return children

    def _make_child(
        self,
        parent: HybridAStarNode,
        goal: np.ndarray,
        action: int,
        delta_xy: np.ndarray,
        heading_delta: int,
        extra_cost: float,
        dz: float = 0.0,
    ) -> tuple[int, HybridAStarNode]:
        cfg = self.config
        new_x = parent.x + float(delta_xy[0])
        new_y = parent.y + float(delta_xy[1])
        new_z = parent.z + dz
        new_heading = _wrap_heading_idx(
            parent.heading_idx + heading_delta, cfg.num_headings
        )

        # cost = distance + altitude_change_weight + extra (turn/climb penalty)
        step_cost = (
            float(np.linalg.norm(delta_xy)) + abs(dz) * 0.5 + extra_cost
        )
        new_g = parent.g + step_cost
        new_h = self._heuristic(
            np.array([new_x, new_y, new_z], dtype=float), goal
        )

        child = HybridAStarNode(
            x=new_x, y=new_y, z=new_z,
            heading_idx=new_heading,
            g=new_g, h=new_h,
            parent=parent, action=action,
        )
        return action, child

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _heuristic(self, pos: np.ndarray, goal: np.ndarray) -> float:
        """Euclidean + altitude-difference heuristic."""
        d_xy = float(np.linalg.norm(pos[:2] - goal[:2]))
        d_z = abs(float(pos[2]) - float(goal[2]))
        return d_xy + d_z * 2.0

    def _grid_key(self, node: HybridAStarNode) -> tuple[int, int, int, int]:
        cfg = self.config
        gx = int(round(node.x / cfg.xy_resolution))
        gy = int(round(node.y / cfg.xy_resolution))
        gz = int(round(node.z / cfg.z_resolution))
        return (gx, gy, gz, node.heading_idx)

    def _in_bounds(self, node: HybridAStarNode) -> bool:
        cfg = self.config
        half = cfg.map_size_units * 0.5
        return (
            -half <= node.x <= half
            and -half <= node.y <= half
            and cfg.min_altitude <= node.z <= cfg.max_altitude
        )

    @staticmethod
    def _reconstruct_path(
        node: HybridAStarNode, goal: np.ndarray
    ) -> list[np.ndarray]:
        """Backtrack from node and append goal."""
        path: list[np.ndarray] = []
        current: HybridAStarNode | None = node
        while current is not None:
            path.append(np.array(
                [current.x, current.y, current.z], dtype=float
            ))
            current = current.parent
        path.reverse()
        # Ensure exact goal
        if np.linalg.norm(path[-1] - goal) > 0.01:
            path.append(np.array(goal, dtype=float))
        return path


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _check_blocked(
    position: np.ndarray,
    obstacles: list[CylindricalObstacle],
    clearance: float,
) -> list[CylindricalObstacle]:
    """Return list of obstacles blocking a position (empty = clear).

    Attaches a temporary ``_block_dist`` attribute for diagnostic printing.
    """
    pos_xy = position[:2]
    alt = float(position[2])
    blocked: list[CylindricalObstacle] = []
    for obs in obstacles:
        dist = float(np.linalg.norm(pos_xy - obs.center_xy))
        if dist < obs.radius + clearance and alt < obs.height + clearance:
            obs._block_dist = dist  # type: ignore[attr-defined]
            blocked.append(obs)
    return blocked
