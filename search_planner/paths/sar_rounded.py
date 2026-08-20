from __future__ import annotations

import math


def generate_sar_rounded(
    center_x: float,
    center_y: float,
    angle_deg: float,
    clockwise: bool,
    radius_units: float,
    sides: int,
    turn_radius_units: float,
) -> list[tuple[float, float]]:
    """Rounded-polygon SAR pattern (SetModel_SAR_Rounded).

    Replaces sharp polygon vertices with circular arc segments.

    Parameters
    ----------
    radius_units : polygon circumradius in scene units.
    sides : 3..20
    turn_radius_units : corner rounding radius (scene units).
    """
    if sides < 3 or sides > 20:
        raise ValueError("sides must be 3..20")

    R = radius_units
    r = turn_radius_units
    max_r = R * math.cos(math.pi / sides) * 0.99
    if r > max_r:
        r = max(max_r, 1.0)

    rad = math.radians(90.0 - angle_deg)

    # Raw polygon vertices in ENU-like local frame
    V = []
    for i in range(sides):
        theta = -2.0 * math.pi * i / sides
        xl = R * math.cos(theta)
        yl = R * math.sin(theta)
        rx = xl * math.cos(rad) - yl * math.sin(rad)
        ry = xl * math.sin(rad) + yl * math.cos(rad)
        V.append((rx, ry))

    # Generate rounded loop
    L_t = r * math.tan(math.pi / sides)
    D_c = r / math.sin(math.pi / sides)
    K = 3  # arc subdivisions

    loop: list[tuple[float, float]] = []
    for i in range(sides):
        prev = (i - 1) % sides
        nxt = (i + 1) % sides

        vx, vy = V[i]
        px, py = V[prev]
        nx, ny = V[nxt]

        # Incoming direction: FROM vertex TO previous (back along edge)
        dxi = px - vx
        dyi = py - vy
        di = math.hypot(dxi, dyi)
        uxi, uyi = dxi / di, dyi / di

        # Outgoing direction: FROM vertex TO next (forward along edge)
        dxo = nx - vx
        dyo = ny - vy
        do = math.hypot(dxo, dyo)
        uxo, uyo = dxo / do, dyo / do

        # Tangency points (both + from vertex, along respective directions)
        tx_in = vx + uxi * L_t
        ty_in = vy + uyi * L_t
        tx_out = vx + uxo * L_t
        ty_out = vy + uyo * L_t

        # Arc centre
        bx = uxi + uxo
        by = uyi + uyo
        db = math.hypot(bx, by)
        cx = vx + (bx / db) * D_c
        cy = vy + (by / db) * D_c

        loop.append((tx_in, ty_in))

        # Arc points
        vx_start = tx_in - cx
        vy_start = ty_in - cy
        for k in range(1, K):
            beta = -(2.0 * math.pi / sides) * (k / K)
            cos_b = math.cos(beta)
            sin_b = math.sin(beta)
            px_a = cx + vx_start * cos_b - vy_start * sin_b
            py_a = cy + vx_start * sin_b + vy_start * cos_b
            loop.append((px_a, py_a))

        loop.append((tx_out, ty_out))

    # Translate to centre
    waypoints = [(x + center_x, y + center_y) for x, y in loop]

    if not clockwise:
        waypoints.reverse()

    # Close
    waypoints.append(waypoints[0])
    return waypoints
