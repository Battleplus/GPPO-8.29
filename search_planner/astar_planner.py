from __future__ import annotations

import heapq
import math

from .terrain import TerrainGrid


def astar_avoid(
    x_start: float,
    y_start: float,
    x_goal: float,
    y_goal: float,
    terrain: TerrainGrid,
) -> list[tuple[float, float]]:
    """Run A* on the TerrainGrid to find a collision-free path.

    Returns an empty list if the straight line is already passable,
    or a list of intermediate (x, y) waypoints that avoid mountain
    obstacles.  The start/goal themselves are *not* included in the
    returned list — callers should splice the result between them.

    If no path can be found (rare — e.g. the goal itself is blocked),
    returns an empty list so the caller falls back to straight-line
    with altitude climb.
    """
    # Quick reject: if the straight line is clear, do nothing
    if _line_is_passable(x_start, y_start, x_goal, y_goal, terrain):
        return []

    sr, sc = terrain.world_to_grid(x_start, y_start)
    gr, gc = terrain.world_to_grid(x_goal, y_goal)

    if not terrain.is_passable(x_goal, y_goal):
        return []  # goal blocked — fall back

    # A* search
    open_heap: list[tuple[float, int, int, int]] = []
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {(sr, sc): 0.0}

    h0 = _heuristic(sr, sc, gr, gc)
    heapq.heappush(open_heap, (h0, 0, sr, sc))
    counter = 1  # tie-breaker

    found = False
    while open_heap:
        _, _, cr, cc = heapq.heappop(open_heap)
        if (cr, cc) == (gr, gc):
            found = True
            break

        current_g = g_score[(cr, cc)]
        for nr, nc in _neighbors(cr, cc, terrain):
            if not terrain.is_passable(
                *terrain.grid_to_world(nr, nc)
            ):
                continue
            move_cost = (
                1.414 if abs(nr - cr) == 1 and abs(nc - cc) == 1 else 1.0
            )
            tentative_g = current_g + move_cost
            if tentative_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = (cr, cc)
                h = _heuristic(nr, nc, gr, gc)
                heapq.heappush(open_heap, (tentative_g + h, counter, nr, nc))
                counter += 1

    if not found:
        return []  # no path — fall back

    # Reconstruct path
    grid_path: list[tuple[int, int]] = [(gr, gc)]
    while grid_path[-1] in came_from:
        grid_path.append(came_from[grid_path[-1]])
    grid_path.reverse()

    # Convert to world coords, skip start/goal adjacent nodes
    world: list[tuple[float, float]] = []
    for r, c in grid_path[1:-1]:
        world.append(terrain.grid_to_world(r, c))

    # Douglas-Peucker simplification
    return _simplify(world, epsilon=1.0)


# ── helpers ──────────────────────────────────────────────────

def _heuristic(r1: int, c1: int, r2: int, c2: int) -> float:
    return math.hypot(r1 - r2, c1 - c2)


def _neighbors(
    row: int, col: int, terrain: TerrainGrid
) -> list[tuple[int, int]]:
    """8-neighbourhood, valid & in-bounds."""
    nb: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < terrain.rows and 0 <= nc < terrain.cols:
                nb.append((nr, nc))
    return nb


def _line_is_passable(
    x1: float, y1: float, x2: float, y2: float,
    terrain: TerrainGrid, samples: int = 16,
) -> bool:
    """Sample the straight-line segment; return True if all points are passable."""
    for i in range(samples + 1):
        t = i / samples
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        if not terrain.is_passable(x, y):
            return False
    return True


def _simplify(
    pts: list[tuple[float, float]], epsilon: float
) -> list[tuple[float, float]]:
    """Douglas-Peucker line simplification."""
    if len(pts) <= 2:
        return pts

    max_dist = 0.0
    max_idx = 0
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy

    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        if denom < 1e-12:
            d = math.hypot(px - x1, py - y1)
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            d = math.hypot(px - proj_x, py - proj_y)
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist <= epsilon:
        return [pts[0], pts[-1]]

    left = _simplify(pts[: max_idx + 1], epsilon)
    right = _simplify(pts[max_idx:], epsilon)
    return left[:-1] + right
