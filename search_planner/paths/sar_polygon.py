from __future__ import annotations

import math


def generate_sar_polygon(
    center_x: float,
    center_y: float,
    angle_deg: float,
    clockwise: bool,
    radius_units: float,
    sides: int,
    loops: int = 1,
) -> list[tuple[float, float]]:
    """Polygon spiral SAR pattern (SetModel_SAR).

    Parameters
    ----------
    radius_units : polygon circumradius in scene units.
    sides : 3..20
    loops : number of full circuits.
    """
    if sides < 3 or sides > 20:
        raise ValueError("sides must be 3..20")

    rad = math.radians(90.0 - angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # Generate vertices in clockwise order (matching C++)
    poly_x: list[float] = []
    poly_y: list[float] = []
    for i in range(sides):
        theta = -2.0 * math.pi * i / sides
        xl = radius_units * math.cos(theta)
        yl = radius_units * math.sin(theta)
        poly_x.append(center_x + xl * cos_a - yl * sin_a)
        poly_y.append(center_y + xl * sin_a + yl * cos_a)

    # Build one-cycle ordered list
    ordered_x = list(poly_x)
    ordered_y = list(poly_y)

    if not clockwise:
        ordered_x.reverse()
        ordered_y.reverse()

    # Produce waypoints: start → loops × circuit → close
    waypoints: list[tuple[float, float]] = []
    for _ in range(loops):
        for i in range(sides):
            waypoints.append((ordered_x[i], ordered_y[i]))
    waypoints.append((ordered_x[0], ordered_y[0]))  # close loop
    return waypoints
