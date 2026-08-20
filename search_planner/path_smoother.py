"""Dubins path computation and waypoint smoothing.

Implements the classic Dubins shortest-path algorithm (LSL, LSR, RSL,
RSR, RLR, LRL) for connecting two (x, y, θ) configurations with a
minimum turning radius.

References
----------
* dubins.h / dubins.cpp from 新版xj/Source/Detect/
* Andrew Walker's open-source Dubins implementation (MIT)
"""

from __future__ import annotations

import math
from typing import Callable

# ── Dubins path types ─────────────────────────────────────────

LSL, LSR, RSL, RSR, RLR, LRL = range(6)

_SEG_NAMES = ("L", "S", "L", "L", "S", "R", "R", "S", "L", "R", "S", "R",
              "R", "L", "R", "L", "R", "L")

# Each path type is (seg0_type, seg1_type, seg2_type) where L=1, S=3, R=2
# fmt: off
_DIRDATA: list[list[int]] = [
    [1, 3, 1],  # LSL
    [1, 3, 2],  # LSR
    [2, 3, 1],  # RSL
    [2, 3, 2],  # RSR
    [2, 1, 2],  # RLR
    [1, 2, 1],  # LRL
]
# fmt: on


# ────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────


def dubins_shortest_path(
    q0: tuple[float, float, float],
    q1: tuple[float, float, float],
    rho: float,
) -> tuple[int, tuple[float, float, float]]:
    """Return (path_type, (len0, len1, len2)) of shortest Dubins path.

    Parameters
    ----------
    q0, q1 : (x, y, theta)  start and goal configurations.
    rho : minimum turning radius (> 0).
    """
    best_type = -1
    best_len = float("inf")
    best_params: tuple[float, float, float] = (0.0, 0.0, 0.0)

    for ptype in (LSL, LSR, RSL, RSR, RLR, LRL):
        ok, p0, p1, p2 = _dubins_word(q0, q1, rho, ptype)
        if not ok:
            continue
        total = p0 + p1 + p2
        if total < best_len:
            best_len = total
            best_type = ptype
            best_params = (p0, p1, p2)

    return best_type, best_params


def sample_dubins_path(
    path_type: int,
    params: tuple[float, float, float],
    q0: tuple[float, float, float],
    rho: float,
    step_size: float,
) -> list[tuple[float, float, float]]:
    """Sample a Dubins path at regular arc-length steps.

    Returns list of (x, y, theta).
    """
    if path_type < 0:
        return []

    total = params[0] + params[1] + params[2]
    if total < 1e-9:
        return [q0]

    seg_types = _DIRDATA[path_type]
    samples: list[tuple[float, float, float]] = [q0]

    t = step_size
    while t < total:
        q = _dubins_segment_sample(
            q0, seg_types, params, rho, t
        )
        samples.append(q)
        t += step_size

    # Always include the endpoint
    q_end = _dubins_segment_sample(q0, seg_types, params, rho, total)
    if samples[-1] != q_end:
        samples.append(q_end)
    return samples


def smooth_waypoints(
    xy_waypoints: list[tuple[float, float]],
    turn_radius: float,
    sample_step: float,
    start_pose: tuple[float, float, float] | None = None,
    force_dubins: bool = False,
) -> list[tuple[float, float]]:
    """Apply Dubins curve smoothing to a waypoint sequence.

    Parameters
    ----------
    xy_waypoints : raw XY path.
    turn_radius : Dubins minimum turning radius (scene units).
    sample_step : arc-length sampling step (scene units).
    start_pose : optional (x, y, theta_rad) for the first segment.
    force_dubins : if False (default), only smooths paths that actually
                   have sharp corners (>60° heading changes).  Set to
                   True to always apply Dubins.

    Returns
    -------
    Smoothed (or unmodified) list of (x, y) waypoints.
    """
    if len(xy_waypoints) < 2:
        return xy_waypoints

    n = len(xy_waypoints)
    headings: list[float] = []
    for i in range(n):
        j = (i + 1) % n
        dx = xy_waypoints[j][0] - xy_waypoints[i][0]
        dy = xy_waypoints[j][1] - xy_waypoints[i][1]
        headings.append(math.atan2(dy, dx))

    if start_pose is not None:
        headings[0] = start_pose[2]

    if not force_dubins:
        # Check whether this path actually needs smoothing.
        # Racetrack / figure-8 / sar_rounded are already smooth;
        # only sar_polygon has sharp vertices worth smoothing.
        max_angle = 0.0
        for i in range(1, n - 1):
            da = abs(_mod2pi(headings[i] - headings[i - 1]))
            if da > max_angle:
                max_angle = da
        if max_angle < math.radians(60.0):
            return xy_waypoints  # already smooth

    # Apply Dubins per-segment
    smoothed: list[tuple[float, float]] = [xy_waypoints[0]]

    for i in range(n - 1):
        sx, sy = xy_waypoints[i]
        gx, gy = xy_waypoints[i + 1]
        dist = math.hypot(gx - sx, gy - sy)

        if dist < turn_radius * 2.0 or turn_radius < 1e-6:
            smoothed.append((gx, gy))
            continue

        q0 = (sx, sy, headings[i])
        q1 = (gx, gy, headings[i + 1])

        ptype, params = dubins_shortest_path(q0, q1, turn_radius)
        if ptype < 0:
            smoothed.append((gx, gy))
            continue

        sampled = sample_dubins_path(ptype, params, q0, turn_radius, sample_step)
        for px, py, _ in sampled[1:]:
            smoothed.append((px, py))

    return smoothed


# ────────────────────────────────────────────────────────────────
#  Core Dubins math
# ────────────────────────────────────────────────────────────────


def _mod2pi(theta: float) -> float:
    v = math.fmod(theta, 2.0 * math.pi)
    if v < -math.pi:
        return v + 2.0 * math.pi
    if v >= math.pi:
        return v - 2.0 * math.pi
    return v


def _dubins_word(
    q0: tuple[float, float, float],
    q1: tuple[float, float, float],
    rho: float,
    ptype: int,
) -> tuple[bool, float, float, float]:
    """Compute the three segment lengths for a given path type."""
    x0, y0, th0 = q0
    x1, y1, th1 = q1

    dx = x1 - x0
    dy = y1 - y0
    D = math.hypot(dx, dy) / rho
    th = _mod2pi(math.atan2(dy, dx))
    alpha = _mod2pi(th0 - th)
    beta = _mod2pi(th1 - th)

    if ptype == LSL:
        return _dubins_LSL(alpha, beta, D)
    elif ptype == LSR:
        return _dubins_LSR(alpha, beta, D)
    elif ptype == RSL:
        return _dubins_RSL(alpha, beta, D)
    elif ptype == RSR:
        return _dubins_RSR(alpha, beta, D)
    elif ptype == RLR:
        return _dubins_RLR(alpha, beta, D)
    elif ptype == LRL:
        return _dubins_LRL(alpha, beta, D)
    return False, 0.0, 0.0, 0.0


def _dubins_LSL(alpha: float, beta: float, d: float) -> tuple[bool, float, float, float]:
    tmp0 = d + math.sin(alpha) - math.sin(beta)
    tmp1 = math.cos(alpha) + math.cos(beta)
    p_sq = tmp0 * tmp0 + tmp1 * tmp1
    if p_sq < 0:
        return False, 0.0, 0.0, 0.0
    p = math.sqrt(p_sq)
    t = _mod2pi(math.atan2(-math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)) - alpha)
    q = _mod2pi(beta - math.atan2(-math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)))
    return True, t, p, q


def _dubins_LSR(alpha: float, beta: float, d: float) -> tuple[bool, float, float, float]:
    tmp0 = d + math.sin(alpha) + math.sin(beta)
    tmp1 = math.cos(alpha) - math.cos(beta)
    p_sq = tmp0 * tmp0 + tmp1 * tmp1
    if p_sq < 0:
        return False, 0.0, 0.0, 0.0
    p = math.sqrt(p_sq)
    t = _mod2pi(math.atan2(math.cos(beta) - math.cos(alpha), d + math.sin(alpha) + math.sin(beta)) - alpha)
    q = _mod2pi(beta - math.atan2(math.cos(beta) - math.cos(alpha), d + math.sin(alpha) + math.sin(beta)))
    return True, t, p, q


def _dubins_RSL(alpha: float, beta: float, d: float) -> tuple[bool, float, float, float]:
    tmp0 = d - math.sin(alpha) - math.sin(beta)
    tmp1 = math.cos(beta) - math.cos(alpha)
    p_sq = tmp0 * tmp0 + tmp1 * tmp1
    if p_sq < 0:
        return False, 0.0, 0.0, 0.0
    p = math.sqrt(p_sq)
    t = _mod2pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) - math.sin(beta)))
    q = _mod2pi(math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) - math.sin(beta)) - beta)
    return True, t, p, q


def _dubins_RSR(alpha: float, beta: float, d: float) -> tuple[bool, float, float, float]:
    tmp0 = d - math.sin(alpha) + math.sin(beta)
    tmp1 = math.cos(alpha) + math.cos(beta)
    p_sq = tmp0 * tmp0 + tmp1 * tmp1
    if p_sq < 0:
        return False, 0.0, 0.0, 0.0
    p = math.sqrt(p_sq)
    t = _mod2pi(alpha - math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) + math.sin(beta)))
    q = _mod2pi(_mod2pi(math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) + math.sin(beta))) - beta)
    return True, t, p, q


def _dubins_RLR(alpha: float, beta: float, d: float) -> tuple[bool, float, float, float]:
    tmp0 = d - math.sin(alpha) + math.sin(beta)
    tmp1 = math.cos(alpha) + math.cos(beta)
    tmp = (6.0 - tmp0 * tmp0 - tmp1 * tmp1) / 8.0
    if abs(tmp) > 1.0:
        return False, 0.0, 0.0, 0.0
    p = _mod2pi(2.0 * math.pi - math.acos(tmp))
    t = _mod2pi(alpha - math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + p * 0.5)
    q = _mod2pi(alpha - beta - t + p)
    return True, t, p, q


def _dubins_LRL(alpha: float, beta: float, d: float) -> tuple[bool, float, float, float]:
    tmp0 = d + math.sin(alpha) - math.sin(beta)
    tmp1 = math.cos(alpha) + math.cos(beta)
    tmp = (6.0 - tmp0 * tmp0 - tmp1 * tmp1) / 8.0
    if abs(tmp) > 1.0:
        return False, 0.0, 0.0, 0.0
    p = _mod2pi(2.0 * math.pi - math.acos(tmp))
    t = _mod2pi(-alpha + math.atan2(math.cos(alpha) + math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + p * 0.5)
    q = _mod2pi(beta - alpha - t + p)
    return True, t, p, q


def _dubins_segment(
    seg_param: float,
    seg_type: int,
    qi: tuple[float, float, float],
    qf: list[float],
) -> None:
    """Advance from qi by seg_param along seg_type, writing result to qf."""
    qf[0] = qi[0]
    qf[1] = qi[1]
    qf[2] = qi[2]

    if seg_type == 1:  # L
        qf[0] += math.sin(qi[2] + seg_param) - math.sin(qi[2])
        qf[1] += -math.cos(qi[2] + seg_param) + math.cos(qi[2])
        qf[2] = _mod2pi(qi[2] + seg_param)
    elif seg_type == 2:  # R
        qf[0] += -math.sin(qi[2] - seg_param) + math.sin(qi[2])
        qf[1] += math.cos(qi[2] - seg_param) - math.cos(qi[2])
        qf[2] = _mod2pi(qi[2] - seg_param)
    elif seg_type == 3:  # S
        qf[0] += seg_param * math.cos(qi[2])
        qf[1] += seg_param * math.sin(qi[2])
        qf[2] = _mod2pi(qi[2])


def _dubins_segment_sample(
    q0: tuple[float, float, float],
    seg_types: list[int],
    params: tuple[float, float, float],
    rho: float,
    t: float,
) -> tuple[float, float, float]:
    """Return configuration at arc-length *t* along the path."""
    qi = list(q0)
    qf = [0.0, 0.0, 0.0]
    remaining = t
    for i in range(3):
        seg_len = params[i]
        if remaining <= seg_len:
            _dubins_segment(remaining / rho * rho, seg_types[i], tuple(qi), qf)
            return (qf[0], qf[1], qf[2])
        _dubins_segment(seg_len, seg_types[i], tuple(qi), qf)
        qi = qf
        remaining -= seg_len
    return (qf[0], qf[1], qf[2])
