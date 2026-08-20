"""Terrain, line-of-sight, and obstacle analysis for attack positions."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .models import ObstacleVolume


TerrainFn = Callable[[float, float], float]
"""Signature: ``terrain_fn(x: float, y: float) -> z: float``."""


def terrain_raycast(
    observer: np.ndarray,
    target: np.ndarray,
    terrain_fn: TerrainFn,
    n_samples: int = 40,
) -> bool:
    """Return ``True`` when terrain does not block the 3-D line of sight."""
    ox, oy, oz = float(observer[0]), float(observer[1]), float(observer[2])
    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
    for i in range(1, n_samples):
        s = i / n_samples
        x = ox + (tx - ox) * s
        y = oy + (ty - oy) * s
        z_ray = oz + (tz - oz) * s
        if terrain_fn(x, y) > z_ray:
            return False
    return True


def obstacle_blocks_segment(
    observer: np.ndarray,
    target: np.ndarray,
    obstacles: list[ObstacleVolume],
    n_samples: int = 48,
) -> str | None:
    """Return the first LOS-blocking obstacle id, or ``None``."""
    delta = np.asarray(target, dtype=float) - np.asarray(observer, dtype=float)
    observer = np.asarray(observer, dtype=float)
    for obstacle in obstacles:
        if not obstacle.blocks_los:
            continue
        for i in range(1, n_samples):
            p = observer + delta * (i / n_samples)
            if not (obstacle.base_z <= float(p[2]) <= obstacle.top_z):
                continue
            dxy = float(np.linalg.norm(p[:2] - obstacle.center[:2]))
            if dxy <= obstacle.radius:
                return obstacle.obstacle_id
    return None


def obstacle_clearance_violation(
    position: np.ndarray,
    obstacles: list[ObstacleVolume],
    horizontal_clearance: float = 0.5,
    vertical_clearance: float = 0.3,
) -> float:
    """Return positive penetration when a rotor-safety volume is violated."""
    worst = 0.0
    for obstacle in obstacles:
        dxy = float(np.linalg.norm(position[:2] - obstacle.center[:2]))
        safe_radius = obstacle.radius + horizontal_clearance
        if dxy >= safe_radius:
            continue
        if float(position[2]) >= obstacle.top_z + vertical_clearance:
            continue
        if float(position[2]) <= obstacle.base_z - vertical_clearance:
            continue
        horizontal_penetration = (safe_radius - dxy) / max(1.0, safe_radius)
        worst = max(worst, horizontal_penetration)
    return worst


def nearest_obstacle_clearance(
    position: np.ndarray,
    obstacles: list[ObstacleVolume],
) -> tuple[str | None, float]:
    """Return nearest obstacle id and conservative 3-D surface clearance."""
    best_id: str | None = None
    best = float("inf")
    z = float(position[2])
    for obstacle in obstacles:
        radial_gap = float(
            np.linalg.norm(position[:2] - obstacle.center[:2])
        ) - obstacle.radius
        if obstacle.base_z <= z <= obstacle.top_z:
            clearance = radial_gap
        else:
            vertical_gap = min(
                abs(z - obstacle.base_z),
                abs(z - obstacle.top_z),
            )
            if radial_gap <= 0.0:
                clearance = vertical_gap
            else:
                clearance = math.hypot(radial_gap, vertical_gap)
        if clearance < best:
            best = clearance
            best_id = obstacle.obstacle_id
    return best_id, best


def terrain_mask_quality(
    position: np.ndarray,
    threat_positions: list[np.ndarray],
    terrain_fn: TerrainFn,
    max_threat_range: float = 80.0,
) -> float:
    """Return the fraction of nearby threats masked by terrain."""
    nearby = [
        t for t in threat_positions
        if float(np.linalg.norm(np.array(t[:2]) - position[:2]))
        <= max_threat_range
    ]
    if nearby:
        blocked = sum(
            1 for t in nearby
            if not terrain_raycast(position, np.array(t, dtype=float), terrain_fn)
        )
        return blocked / len(nearby)
    return _local_roughness_mask(position, terrain_fn)


def _local_roughness_mask(
    position: np.ndarray,
    terrain_fn: TerrainFn,
    radius: float = 200.0,
    n_sectors: int = 8,
    n_samples_per_sector: int = 6,
) -> float:
    """Estimate available masking from terrain roughness around a position."""
    px, py = float(position[0]), float(position[1])
    sector_scores: list[float] = []
    for si in range(n_sectors):
        theta = 2.0 * math.pi * si / n_sectors
        elevations: list[float] = []
        for ri in range(1, n_samples_per_sector + 1):
            r = radius * ri / n_samples_per_sector
            elevations.append(float(terrain_fn(
                px + r * math.cos(theta),
                py + r * math.sin(theta),
            )))
        if len(elevations) >= 2:
            sector_scores.append(float(np.std(elevations)))
    if not sector_scores:
        return 0.0
    return min(1.0, max(0.0, float(np.mean(sector_scores)) / 30.0))


def find_masked_positions(
    target: np.ndarray,
    terrain_fn: TerrainFn,
    n_candidates: int = 200,
    r_min: float = 20.0,
    r_max: float = 80.0,
    z_clearance: float = 15.0,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Sample positions around a target at a requested terrain clearance."""
    if rng is None:
        rng = np.random.default_rng(42)
    tx, ty = float(target[0]), float(target[1])
    candidates: list[np.ndarray] = []
    for _ in range(n_candidates * 2):
        r = float(rng.uniform(r_min, r_max))
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        x = tx + r * math.cos(theta)
        y = ty + r * math.sin(theta)
        z = float(terrain_fn(x, y)) + z_clearance
        candidates.append(np.array([x, y, z], dtype=float))
        if len(candidates) >= n_candidates:
            break
    return candidates
