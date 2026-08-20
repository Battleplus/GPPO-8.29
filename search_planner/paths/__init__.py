from __future__ import annotations

from typing import Callable

from . import sar_polygon, sar_rounded, racetrack, figure_eight

_PATTERNS: dict[str, Callable[..., list[tuple[float, float]]]] = {
    "sar_polygon": sar_polygon.generate_sar_polygon,
    "sar_rounded": sar_rounded.generate_sar_rounded,
    "racetrack": racetrack.generate_racetrack,
    "figure_eight": figure_eight.generate_figure_eight,
}


def generate_path(
    pattern: str,
    center_x: float,
    center_y: float,
    angle_deg: float,
    clockwise: bool,
    meters_per_unit: float = 100.0,
    **kwargs,
) -> list[tuple[float, float]]:
    """Dispatch to the appropriate path generator.

    Parameters
    ----------
    pattern : one of 'sar_polygon', 'sar_rounded', 'racetrack', 'figure_eight'.
    center_x, center_y : search area centre in scene units.
    angle_deg : 0 = north (+Y), clockwise positive.
    **kwargs : forwarded to the specific generator (units should already be
               converted to scene units by the caller).

    Returns
    -------
    List of (x, y) waypoints in scene units.
    """
    gen = _PATTERNS.get(pattern)
    if gen is None:
        raise ValueError(
            f"Unknown pattern '{pattern}'. "
            f"Choose from: {', '.join(sorted(_PATTERNS))}"
        )
    return gen(
        center_x=center_x,
        center_y=center_y,
        angle_deg=angle_deg,
        clockwise=clockwise,
        **kwargs,
    )
