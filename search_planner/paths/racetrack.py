from __future__ import annotations

import math


def generate_racetrack(
    center_x: float,
    center_y: float,
    angle_deg: float,
    clockwise: bool,
    length_units: float,
    width_units: float,
    path_count: int = 14,
) -> list[tuple[float, float]]:
    """Racetrack / rounded-rectangle pattern (SetModel_Photoelectric).

    Parameters
    ----------
    length_units : long-edge span in scene units.
    width_units : separation between parallel long edges.
    path_count : total waypoint count per lap (default 14).
    """
    hl = length_units / 2.0
    hw = width_units / 2.0
    arc_curvature = 0.7

    seg_per_long = max(1, path_count // 2 - 1)
    seg_per_arc = 3

    rad = math.radians(90.0 - angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # 4 corners in local frame (CCW order)
    cx = [-hl, hl, hl, -hl]
    cy = [-hw, -hw, hw, hw]

    # Rotate + translate corners
    rx = [0.0] * 4
    ry = [0.0] * 4
    for i in range(4):
        rx[i] = cx[i] * cos_a - cy[i] * sin_a + center_x
        ry[i] = cx[i] * sin_a + cy[i] * cos_a + center_y

    # Build waypoints along perimeter (CCW: 0→1→2→3)
    pts_x: list[float] = []
    pts_y: list[float] = []

    for i in range(4):
        a = i
        b = (i + 1) % 4
        if i % 2 == 0:  # long edge → straight
            for j in range(1, seg_per_long + 1):
                t = j / seg_per_long
                pts_x.append(rx[a] + t * (rx[b] - rx[a]))
                pts_y.append(ry[a] + t * (ry[b] - ry[a]))
        else:  # short edge → outward arc
            dx = rx[b] - rx[a]
            dy = ry[b] - ry[a]
            dist = math.hypot(dx, dy)
            mid_x = (rx[a] + rx[b]) / 2.0
            mid_y = (ry[a] + ry[b]) / 2.0

            # Perpendicular unit vector (rotated 90° CCW)
            perp_x = -dy / dist
            perp_y = dx / dist

            offset = (dist / 2.0) * math.tan(math.pi * arc_curvature / 4.0)
            arc_cx = mid_x + perp_x * offset
            arc_cy = mid_y + perp_y * offset

            start_angle = math.atan2(ry[a] - arc_cy, rx[a] - arc_cx)
            end_angle = math.atan2(ry[b] - arc_cy, rx[b] - arc_cx)

            # Always go the short way
            while end_angle < start_angle:
                end_angle += 2 * math.pi
            while end_angle - start_angle > math.pi:
                end_angle -= 2 * math.pi

            arc_r = (dist / 2.0) / math.cos(math.pi * arc_curvature / 4.0)
            for j in range(1, seg_per_arc + 1):
                t = j / seg_per_arc
                ang = start_angle + t * (end_angle - start_angle)
                pts_x.append(arc_cx + arc_r * math.cos(ang))
                pts_y.append(arc_cy + arc_r * math.sin(ang))

    # Reverse traversal order if counter-clockwise requested
    if not clockwise:
        pts_x.reverse()
        pts_y.reverse()

    # Close the loop
    pts_x.append(pts_x[0])
    pts_y.append(pts_y[0])

    return list(zip(pts_x, pts_y))
