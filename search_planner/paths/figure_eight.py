from __future__ import annotations

import math


def generate_figure_eight(
    center_x: float,
    center_y: float,
    angle_deg: float,
    clockwise: bool,
    radius_units: float,
    line_units: float,
    loops: int = 1,
) -> list[tuple[float, float]]:
    """Figure-8 pattern (SetModel_AntiRadiation).

    Two circles connected by straight tangent segments.

    Parameters
    ----------
    radius_units : circle radius in scene units.
    line_units : centre-to-centre distance (must be > 2 * radius_units).
    loops : number of figure-8 circuits.
    """
    L = line_units / 2.0  # half-distance (scene units)
    R = radius_units
    if L <= R:
        raise ValueError("line_units/2 must be > radius_units")

    alpha = math.acos(R / L)

    seg_half_circle = 6
    seg_line = 3

    # ── Build one-cycle local points ──
    local: list[tuple[float, float]] = []

    # Left circle
    left_start = alpha
    left_end = 2.0 * math.pi - alpha
    for i in range(seg_half_circle + 1):
        t = i / seg_half_circle
        theta = left_start + t * (left_end - left_start)
        local.append((-L + R * math.cos(theta), R * math.sin(theta)))

    # First straight line
    p_bl = local[-1]
    p_tr = (L + R * math.cos(math.pi - alpha), R * math.sin(math.pi - alpha))
    for i in range(1, seg_line + 1):
        t = i / seg_line
        local.append(
            (p_bl[0] + t * (p_tr[0] - p_bl[0]), p_bl[1] + t * (p_tr[1] - p_bl[1]))
        )

    # Right circle
    right_start = math.pi - alpha
    right_end = -math.pi + alpha
    for i in range(1, seg_half_circle + 1):
        t = i / seg_half_circle
        theta = right_start + t * (right_end - right_start)
        local.append((L + R * math.cos(theta), R * math.sin(theta)))

    # Second straight line (back to left)
    p_br = local[-1]
    p_tl = (-L + R * math.cos(alpha), R * math.sin(alpha))
    for i in range(1, seg_line):
        t = i / seg_line
        local.append(
            (p_br[0] + t * (p_tl[0] - p_br[0]), p_br[1] + t * (p_tl[1] - p_br[1]))
        )

    # ── Rotate & translate ──
    rad = math.radians(90.0 - angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    one_cycle: list[tuple[float, float]] = []
    for xl, yl in local:
        rx = xl * cos_a - yl * sin_a + center_x
        ry = xl * sin_a + yl * cos_a + center_y
        one_cycle.append((rx, ry))

    if not clockwise:
        one_cycle.reverse()

    # ── Output loops ──
    waypoints: list[tuple[float, float]] = []
    for _ in range(loops):
        waypoints.extend(one_cycle)
    waypoints.append(one_cycle[0])  # close
    return waypoints
