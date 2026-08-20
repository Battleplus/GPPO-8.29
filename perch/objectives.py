"""Explainable objective functions for attack-position optimisation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .terrain_analyzer import TerrainFn, terrain_mask_quality


ObjectiveFn = Callable[
    [
        np.ndarray,
        np.ndarray,
        list[np.ndarray],
        TerrainFn,
        float,
        float,
        float,
        np.ndarray | None,
    ],
    float,
]


def f_exposure(
    pos: np.ndarray,
    target: np.ndarray,
    threats: list[np.ndarray],
    terrain_fn: TerrainFn,
    min_range: float = 16.0,
    max_range: float = 80.0,
    optimal_range: float = 52.0,
    start_position: np.ndarray | None = None,
) -> float:
    """Minimise exposure to known threat emitters/weapon locations."""
    _ = target, min_range, optimal_range, start_position
    return 1.0 - terrain_mask_quality(
        pos, threats, terrain_fn, max_range
    )


def f_range_deviation(
    pos: np.ndarray,
    target: np.ndarray,
    threats: list[np.ndarray],
    terrain_fn: TerrainFn,
    min_range: float = 16.0,
    max_range: float = 80.0,
    optimal_range: float = 52.0,
    start_position: np.ndarray | None = None,
) -> float:
    """Minimise slant-range deviation from the selected weapon sweet spot."""
    _ = threats, terrain_fn, start_position
    dist = float(np.linalg.norm(pos - target))
    if dist < min_range:
        return 10.0 * (min_range - dist) / max(1e-6, min_range)
    if dist > max_range:
        return 10.0 * (dist - max_range) / max(1e-6, max_range)
    return abs(dist - optimal_range) / max(1e-6, optimal_range)


def f_approach(
    pos: np.ndarray,
    target: np.ndarray,
    threats: list[np.ndarray],
    terrain_fn: TerrainFn,
    min_range: float = 16.0,
    max_range: float = 80.0,
    optimal_range: float = 52.0,
    start_position: np.ndarray | None = None,
) -> float:
    """Minimise exposure along ingress from the assigned aircraft position."""
    _ = target, min_range, max_range, optimal_range
    if start_position is None:
        staging = np.array([0.0, 0.0, float(pos[2])], dtype=float)
    else:
        staging = np.array(start_position, dtype=float)

    if not threats:
        dist = float(np.linalg.norm(pos[:2] - staging[:2]))
        return min(1.0, dist / 1000.0)

    n_samples = 5
    exposed = 0
    for i in range(1, n_samples + 1):
        sample = staging + (pos - staging) * (i / n_samples)
        sample[2] = max(
            float(sample[2]),
            float(terrain_fn(float(sample[0]), float(sample[1]))) + 0.01,
        )
        for threat in threats[:5]:
            if _has_los(sample, threat, terrain_fn):
                exposed += 1
                break
    return exposed / n_samples


def _has_los(
    observer: np.ndarray,
    target_threat: np.ndarray,
    terrain_fn: TerrainFn,
    n_samples: int = 20,
) -> bool:
    ox, oy, oz = (float(value) for value in observer[:3])
    tx, ty = float(target_threat[0]), float(target_threat[1])
    tz = (
        float(target_threat[2])
        if len(target_threat) > 2 else float(observer[2])
    )
    for i in range(1, n_samples):
        s = i / n_samples
        x = ox + (tx - ox) * s
        y = oy + (ty - oy) * s
        z_ray = oz + (tz - oz) * s
        if terrain_fn(x, y) > z_ray:
            return False
    return True
