"""Formation management -- fixed formation offset computation.

Supported formations:
  - line_abreast  : line abreast (spread perpendicular to heading)
  - v_shape       : V formation (leader at front, wings sweep back)
  - echelon_left  : left echelon
  - echelon_right : right echelon
  - diamond       : diamond formation
  - column        : column (trail)

All offsets are relative to formation center and auto-rotated by heading.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Formation type registry
# ---------------------------------------------------------------------------

FORMATION_NAMES: dict[str, str] = {
    "line_abreast": "Line Abreast",
    "v_shape": "V Shape",
    "echelon_left": "Echelon Left",
    "echelon_right": "Echelon Right",
    "diamond": "Diamond",
    "column": "Column",
}


def get_formation_roles(
    formation_type: str,
    team_size: int,
) -> list[str]:
    """Return role name for each offset index in the formation.

    These names can be used with ``member_assignments`` to map
    specific platform indices to specific roles.

    Args:
        formation_type: One of the supported formation types.
        team_size: Number of members in the formation.

    Returns:
        List of role name strings, index-aligned with ``get_formation_offsets``.

    Example:
        >>> get_formation_roles("v_shape", 5)
        ['leader', 'left_wing_1', 'left_wing_2', 'right_wing_1', 'right_wing_2']
    """
    ft = formation_type.lower().strip()

    if ft == "v_shape":
        return _v_shape_roles(team_size)
    elif ft == "line_abreast":
        return _line_abreast_roles(team_size)
    elif ft == "echelon_left":
        return _echelon_left_roles(team_size)
    elif ft == "echelon_right":
        return _echelon_right_roles(team_size)
    elif ft == "diamond":
        return _diamond_roles(team_size)
    elif ft == "column":
        return _column_roles(team_size)
    else:
        valid = list(FORMATION_NAMES.keys())
        raise ValueError(
            f"Unknown formation type: {formation_type}. Supported: {valid}"
        )


def get_formation_offsets(
    formation_type: str,
    team_size: int,
    spacing: float = 30.0,
    depth_spacing: float | None = None,
) -> list[np.ndarray]:
    """Compute member offsets relative to formation center.

    Offsets use body frame: forward = +x, right = +y.
    These are later rotated to align with the flight heading.

    Args:
        formation_type: One of the supported formation types.
        team_size: Number of members in the formation.
        spacing: Lateral spacing between members (scene units).
        depth_spacing: Depth spacing (default = spacing).

    Returns:
        List of [dx, dy] offsets for each member.
    """
    if depth_spacing is None:
        depth_spacing = spacing

    ft = formation_type.lower().strip()

    if ft == "line_abreast":
        return _line_abreast(team_size, spacing)
    elif ft == "v_shape":
        return _v_shape(team_size, spacing, depth_spacing)
    elif ft == "echelon_left":
        return _echelon_left(team_size, spacing, depth_spacing)
    elif ft == "echelon_right":
        return _echelon_right(team_size, spacing, depth_spacing)
    elif ft == "diamond":
        return _diamond(team_size, spacing, depth_spacing)
    elif ft == "column":
        return _column(team_size, spacing)
    else:
        valid = list(FORMATION_NAMES.keys())
        raise ValueError(
            f"Unknown formation type: {formation_type}. Supported: {valid}"
        )


def rotate_offsets(
    offsets: list[np.ndarray],
    heading_rad: float,
) -> list[np.ndarray]:
    """Rotate formation offsets to world frame given a heading angle.

    Body frame: forward = +x, right = +y
    World frame: heading direction = (cos(h), sin(h))

    Args:
        offsets: Offsets in body frame.
        heading_rad: Heading angle in radians (0=east, pi/2=north).

    Returns:
        Rotated offsets in world frame.
    """
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    rotated: list[np.ndarray] = []
    for offset in offsets:
        dx = float(offset[0]) * cos_h - float(offset[1]) * sin_h
        dy = float(offset[0]) * sin_h + float(offset[1]) * cos_h
        rotated.append(np.array([dx, dy], dtype=float))
    return rotated


def distribute_team_waypoints(
    center_waypoints: list[np.ndarray],
    offsets: list[np.ndarray],
    heading_fn=None,
) -> list[list[np.ndarray]]:
    """Generate per-member waypoints from center path and formation offsets.

    Args:
        center_waypoints: Center path waypoints, each [x, y, z].
        offsets: Formation offsets in body frame [M, 2].
        heading_fn: Optional callable(wp_index, center_waypoints) -> heading_rad.
                    If None, heading is computed from path segment direction.

    Returns:
        List of per-member paths: team_paths[member_idx][wp_idx] = [x, y, z]
    """
    team_size = len(offsets)
    team_paths: list[list[np.ndarray]] = [[] for _ in range(team_size)]

    for i_wp, center_wp in enumerate(center_waypoints):
        center_xy = center_wp[:2]
        center_z = float(center_wp[2])

        # Determine heading at this waypoint
        if heading_fn is not None:
            heading = heading_fn(i_wp, center_waypoints)
        elif i_wp < len(center_waypoints) - 1:
            delta = center_waypoints[i_wp + 1][:2] - center_xy
            dist = float(np.linalg.norm(delta))
            heading = (
                math.atan2(float(delta[1]), float(delta[0]))
                if dist > 1e-6 else 0.0
            )
        elif i_wp > 0:
            delta = center_xy - center_waypoints[i_wp - 1][:2]
            dist = float(np.linalg.norm(delta))
            heading = (
                math.atan2(float(delta[1]), float(delta[0]))
                if dist > 1e-6 else 0.0
            )
        else:
            heading = 0.0

        rotated = rotate_offsets(offsets, heading)
        for i_member, rot_offset in enumerate(rotated):
            wp = np.array(
                [center_xy[0] + rot_offset[0],
                 center_xy[1] + rot_offset[1],
                 center_z],
                dtype=float,
            )
            team_paths[i_member].append(wp)

    return team_paths


# ---------------------------------------------------------------------------
# Offset generators (body frame: forward=+x, right=+y)
# ---------------------------------------------------------------------------


def _line_abreast(n: int, spacing: float) -> list[np.ndarray]:
    """Line abreast: spread along lateral axis, leader at center."""
    offsets: list[np.ndarray] = []
    half = (n - 1) / 2.0
    for i in range(n):
        dy = (i - half) * spacing
        offsets.append(np.array([0.0, dy], dtype=float))
    return offsets


def _v_shape(n: int, spacing: float, depth: float) -> list[np.ndarray]:
    """V formation: leader at front, wingmen sweep back and outward."""
    offsets: list[np.ndarray] = []
    offsets.append(np.array([0.0, 0.0], dtype=float))  # leader
    left_count = (n - 1) // 2
    right_count = n - 1 - left_count
    for i in range(1, left_count + 1):
        offsets.append(np.array([-i * depth, -i * spacing], dtype=float))
    for i in range(1, right_count + 1):
        offsets.append(np.array([-i * depth, i * spacing], dtype=float))
    return offsets


def _echelon_left(n: int, spacing: float, depth: float) -> list[np.ndarray]:
    """Left echelon: each member steps back and to the right."""
    offsets: list[np.ndarray] = []
    for i in range(n):
        offsets.append(np.array([-i * depth, i * spacing], dtype=float))
    return offsets


def _echelon_right(n: int, spacing: float, depth: float) -> list[np.ndarray]:
    """Right echelon: each member steps back and to the left."""
    offsets: list[np.ndarray] = []
    for i in range(n):
        offsets.append(np.array([-i * depth, -i * spacing], dtype=float))
    return offsets


def _diamond(n: int, spacing: float, depth: float) -> list[np.ndarray]:
    """Diamond formation."""
    if n == 1:
        return [np.array([0.0, 0.0], dtype=float)]
    if n == 2:
        return [
            np.array([0.0, 0.0], dtype=float),
            np.array([-depth, 0.0], dtype=float),
        ]
    if n == 3:
        return [
            np.array([0.0, 0.0], dtype=float),
            np.array([-depth, -spacing], dtype=float),
            np.array([-depth, spacing], dtype=float),
        ]
    if n == 4:
        return [
            np.array([0.0, 0.0], dtype=float),          # leader
            np.array([-depth, -spacing], dtype=float),   # left wing
            np.array([-depth, spacing], dtype=float),    # right wing
            np.array([-depth * 2, 0.0], dtype=float),    # tail
        ]
    # n >= 5: diamond core + extra on right side
    offsets = [
        np.array([0.0, 0.0], dtype=float),
        np.array([-depth, -spacing], dtype=float),
        np.array([-depth, spacing], dtype=float),
        np.array([-depth * 2, 0.0], dtype=float),
    ]
    extra = n - 4
    for i in range(1, extra + 1):
        offsets.append(np.array([-depth * 2, i * spacing], dtype=float))
    return offsets


def _column(n: int, spacing: float) -> list[np.ndarray]:
    """Column: trail behind leader along heading direction."""
    offsets: list[np.ndarray] = []
    for i in range(n):
        offsets.append(np.array([-i * spacing, 0.0], dtype=float))
    return offsets


# ---------------------------------------------------------------------------
# Role-name helpers (index-aligned with offset generators above)
# ---------------------------------------------------------------------------


def _v_shape_roles(n: int) -> list[str]:
    roles = ["leader"]
    left_count = (n - 1) // 2
    right_count = n - 1 - left_count
    for i in range(1, left_count + 1):
        roles.append(f"left_wing_{i}")
    for i in range(1, right_count + 1):
        roles.append(f"right_wing_{i}")
    return roles


def _line_abreast_roles(n: int) -> list[str]:
    if n == 1:
        return ["center"]
    half = (n - 1) / 2.0
    roles: list[str] = []
    for i in range(n):
        offset = i - half
        if abs(offset) < 0.01:
            roles.append("center")
        elif offset < 0:
            roles.append(f"left_{abs(int(offset))}")
        else:
            roles.append(f"right_{int(offset)}")
    return roles


def _echelon_left_roles(n: int) -> list[str]:
    roles = ["leader"]
    for i in range(1, n):
        roles.append(f"trail_{i}")
    return roles


def _echelon_right_roles(n: int) -> list[str]:
    roles = ["leader"]
    for i in range(1, n):
        roles.append(f"trail_{i}")
    return roles


def _diamond_roles(n: int) -> list[str]:
    if n == 1:
        return ["leader"]
    if n == 2:
        return ["leader", "trail"]
    if n == 3:
        return ["leader", "left_wing", "right_wing"]
    roles = ["leader", "left_wing", "right_wing", "tail"]
    for i in range(1, n - 3):
        roles.append(f"extra_{i}")
    return roles


def _column_roles(n: int) -> list[str]:
    roles = ["leader"]
    for i in range(1, n):
        roles.append(f"trail_{i}")
    return roles
