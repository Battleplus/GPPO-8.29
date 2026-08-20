from __future__ import annotations

import math

import numpy as np

from .area import SearchArea
from .obstacles import MountainObstacle, check_collision


class TerrainGrid:
    """2.5-D elevation grid for a SearchArea.

    Samples *terrain_height_fn* at regular resolution and supports
    bilinear interpolation and collision queries.
    """

    def __init__(
        self,
        area: SearchArea,
        mountains: list[MountainObstacle],
        resolution_m: float = 500.0,
        safety_xy_m: float = 500.0,
        map_size_units: float = 3000.0,
        visual_height_units: float = 15.0,
        meters_per_unit: float = 100.0,
        terrain_height_fn=None,
    ):
        self.area = area
        self.mountains = mountains
        self.resolution_m = resolution_m
        self.safety_xy_units = safety_xy_m / meters_per_unit
        self.map_size_units = map_size_units
        self.visual_height_units = visual_height_units
        self.meters_per_unit = meters_per_unit

        res_units = resolution_m / meters_per_unit
        self.cols = max(2, int(math.ceil(area.width_units / res_units)) + 1)
        self.rows = max(2, int(math.ceil(area.height_units / res_units)) + 1)

        if terrain_height_fn is None:
            from scenes.air_combat_scene import terrain_height as _th
            terrain_height_fn = _th
        self._height_fn = terrain_height_fn

        self._grid: np.ndarray = self._build_grid()

    # ── grid build ────────────────────────────────────────────

    def _build_grid(self) -> np.ndarray:
        grid = np.empty((self.rows, self.cols), dtype=np.float64)
        for row in range(self.rows):
            y = self.area.y_min + row * (self.area.height_units / max(1, self.rows - 1))
            for col in range(self.cols):
                x = self.area.x_min + col * (self.area.width_units / max(1, self.cols - 1))
                grid[row, col] = self._height_fn(
                    x, y, self.map_size_units, self.visual_height_units
                )
        return grid

    # ── public queries ────────────────────────────────────────

    def height_at(self, x: float, y: float) -> float:
        """Bilinear interpolation of terrain height at (x, y) in scene units."""
        fx = (x - self.area.x_min) / self.area.width_units * (self.cols - 1)
        fy = (y - self.area.y_min) / self.area.height_units * (self.rows - 1)

        col0 = max(0, min(self.cols - 2, int(math.floor(fx))))
        row0 = max(0, min(self.rows - 2, int(math.floor(fy))))
        col1 = min(col0 + 1, self.cols - 1)
        row1 = min(row0 + 1, self.rows - 1)

        tx = fx - col0
        ty = fy - row0
        tx = max(0.0, min(1.0, tx))
        ty = max(0.0, min(1.0, ty))

        h00 = self._grid[row0, col0]
        h10 = self._grid[row0, col1]
        h01 = self._grid[row1, col0]
        h11 = self._grid[row1, col1]

        return float(
            h00 * (1 - tx) * (1 - ty)
            + h10 * tx * (1 - ty)
            + h01 * (1 - tx) * ty
            + h11 * tx * ty
        )

    def is_passable(self, x: float, y: float) -> bool:
        """Check whether (x, y) is outside every mountain's XY safety margin.

        Uses z=0 so that any point within the mountain's horizontal
        footprint is treated as blocked regardless of altitude.
        """
        for m in self.mountains:
            if check_collision(x, y, 0.0, m, self.safety_xy_units):
                return False
        return True

    def grid(self) -> np.ndarray:
        """Return the raw 2-D height array (rows × cols)."""
        return self._grid

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Map scene-unit coordinates to (row, col) grid indices."""
        col = int(round((x - self.area.x_min) / self.area.width_units * (self.cols - 1)))
        row = int(round((y - self.area.y_min) / self.area.height_units * (self.rows - 1)))
        return (
            max(0, min(self.rows - 1, row)),
            max(0, min(self.cols - 1, col)),
        )

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Map (row, col) grid indices back to scene-unit coordinates."""
        x = self.area.x_min + col * self.area.width_units / max(1, self.cols - 1)
        y = self.area.y_min + row * self.area.height_units / max(1, self.rows - 1)
        return float(x), float(y)
