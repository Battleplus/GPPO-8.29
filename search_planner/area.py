from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchArea:
    """Axis-aligned rectangular search region in scene units.

    Coordinate origin is the map centre.  X = east, Y = north.
    Full map extent: [-1500, 1500] × [-1500, 1500] for the default
    300 km × 300 km map at 100 m/unit.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    # ── derived properties ──

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) * 0.5

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) * 0.5

    @property
    def width_units(self) -> float:
        return self.x_max - self.x_min

    @property
    def height_units(self) -> float:
        return self.y_max - self.y_min

    @property
    def half_width(self) -> float:
        return self.width_units * 0.5

    @property
    def half_height(self) -> float:
        return self.height_units * 0.5

    def width_km(self, meters_per_unit: float = 100.0) -> float:
        return self.width_units * meters_per_unit / 1000.0

    def height_km(self, meters_per_unit: float = 100.0) -> float:
        return self.height_units * meters_per_unit / 1000.0

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            max(self.x_min, min(self.x_max, x)),
            max(self.y_min, min(self.y_max, y)),
        )

    def __repr__(self) -> str:
        return (
            f"SearchArea(x=[{self.x_min:.1f}, {self.x_max:.1f}], "
            f"y=[{self.y_min:.1f}, {self.y_max:.1f}], "
            f"size={self.width_units:.0f}×{self.height_units:.0f} u)"
        )


# ═══════════════════════════════════════════════════════════════
#  Factory functions (public API)
# ═══════════════════════════════════════════════════════════════

def area_from_bounds_km(
    x_min_km: float,
    y_min_km: float,
    x_max_km: float,
    y_max_km: float,
    map_size_km: float = 300.0,
    meters_per_unit: float = 100.0,
) -> SearchArea:
    """Create a SearchArea from arbitrary bounding-box corners in km offsets.

    All coordinates are relative to the map centre (0,0).
    The area is automatically clamped to the map boundary.
    """
    scale = 1000.0 / meters_per_unit  # km → scene units
    half_units = map_size_km * 1000.0 / meters_per_unit * 0.5

    x_min = max(-half_units, min(half_units, x_min_km * scale))
    x_max = max(-half_units, min(half_units, x_max_km * scale))
    y_min = max(-half_units, min(half_units, y_min_km * scale))
    y_max = max(-half_units, min(half_units, y_max_km * scale))

    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"Invalid bounds after clamping: x=[{x_min:.1f}, {x_max:.1f}], "
            f"y=[{y_min:.1f}, {y_max:.1f}]"
        )

    return SearchArea(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def area_from_center_km(
    center_x_km: float,
    center_y_km: float,
    width_km: float,
    height_km: float | None = None,
    map_size_km: float = 300.0,
    meters_per_unit: float = 100.0,
) -> SearchArea:
    """Create a SearchArea from a centre point and width/height in km.

    height_km defaults to width_km (square area).
    Clamped to the map boundary.
    """
    if height_km is None:
        height_km = width_km

    hw = width_km * 0.5
    hh = height_km * 0.5
    return area_from_bounds_km(
        center_x_km - hw, center_y_km - hh,
        center_x_km + hw, center_y_km + hh,
        map_size_km, meters_per_unit,
    )


def area_from_grid_cell(
    row: int,
    col: int,
    map_size_km: float = 300.0,
    meters_per_unit: float = 100.0,
) -> SearchArea:
    """Create a SearchArea from a 12×12 grid cell.

    row 0 = southernmost, col 0 = westernmost.  Each cell ≈ 25 km × 25 km.
    """
    GRID = 12
    if not (0 <= row < GRID and 0 <= col < GRID):
        raise ValueError(f"row/col must be 0..{GRID - 1}, got ({row}, {col})")

    cell_units = map_size_km * 1000.0 / meters_per_unit / GRID
    half = map_size_km * 1000.0 / meters_per_unit * 0.5

    x_min = -half + col * cell_units
    x_max = x_min + cell_units
    y_min = -half + row * cell_units
    y_max = y_min + cell_units

    return SearchArea(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
