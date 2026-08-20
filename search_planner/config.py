from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlannerConfig:
    """SAR search planner configuration.

    Search area is specified via ONE of three mutually-exclusive groups
    (checked in priority order):
      - area_bounds_km    (highest)  – arbitrary axis-aligned rectangle
      - area_center_km              – center + width/height
      - grid_row / grid_col  (lowest) – 12×12 grid shortcut
    """

    # ──────────────────────────────────
    #  Search area (pick one group)
    # ──────────────────────────────────
    area_bounds_km: tuple[float, float, float, float] | None = None
    """(x_min_km, y_min_km, x_max_km, y_max_km) relative to map centre."""

    area_center_km: tuple[float, float] | None = None  # (cx_km, cy_km)
    area_width_km: float = 25.0
    area_height_km: float | None = None  # None → same as width

    grid_row: int | None = None  # 0..11, 0 = southernmost
    grid_col: int | None = None  # 0..11, 0 = westernmost

    # ──────────────────────────────────
    #  Path pattern
    # ──────────────────────────────────
    pattern: str = "racetrack"
    """One of: racetrack | sar_polygon | sar_rounded | figure_eight"""

    angle_deg: float = 30.0       # path rotation (0=north, clockwise positive)
    clockwise: bool = True

    # ──────────────────────────────────
    #  Flight parameters
    # ──────────────────────────────────
    altitude_agl_m: float = 5000.0
    cruise_speed_mps: float = 200.0

    # ──────────────────────────────────
    #  Obstacle avoidance
    # ──────────────────────────────────
    obstacle_safety_xy_m: float = 500.0
    obstacle_safety_z_m: float = 120.0
    astar_grid_resolution_m: float = 500.0

    # ──────────────────────────────────
    #  Dubins smoothing
    # ──────────────────────────────────
    dubins_turn_radius_m: float = 5500.0
    dubins_sample_step_m: float = 3000.0  # waypoint spacing ~1 km

    # ──────────────────────────────────
    #  Pattern-specific knobs (fixed defaults for 25×25 km grid)
    # ──────────────────────────────────
    # racetrack
    racetrack_length_km: float = 18.0
    racetrack_width_km: float = 14.0
    racetrack_path_count: int = 14

    # sar_polygon / sar_rounded
    sar_radius_km: float = 10.0
    sar_sides: int = 6          # 3..20
    sar_loops: int = 1
    sar_turn_radius_km: float = 5.0  # only sar_rounded

    # figure_eight
    eight_radius_km: float = 6.0
    eight_line_km: float = 18.0      # must be > 2 * radius
    eight_loops: int = 1

    # ──────────────────────────────────
    #  Map defaults (read-only by planner)
    # ──────────────────────────────────
    map_size_km: float = 300.0
    meters_per_unit: float = 100.0
    mountain_height_m: float = 1500.0
    terrain_vertical_exaggeration: float = 10.0  # Isaac Sim visual exaggeration

    def __post_init__(self) -> None:
        if self.area_height_km is None:
            self.area_height_km = self.area_width_km
        if not (3 <= self.sar_sides <= 20):
            raise ValueError(f"sar_sides must be 3..20, got {self.sar_sides}")
        if self.pattern not in {"racetrack", "sar_polygon", "sar_rounded", "figure_eight"}:
            raise ValueError(f"Unknown pattern: {self.pattern}")
        if self.eight_line_km <= 2 * self.eight_radius_km:
            raise ValueError("eight_line_km must be > 2 * eight_radius_km")
