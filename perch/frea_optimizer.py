"""FREA/R-NSGA-II multi-objective attack-position optimiser."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constraints import (
    CONSTRAINT_FUNCTIONS,
    CONSTRAINT_NAMES,
    DEFAULT_MAX_DESIGNATION_ANGLE_DEG,
)
from .models import ObstacleVolume
from .objectives import f_approach, f_exposure, f_range_deviation
from .terrain_analyzer import (
    TerrainFn,
    obstacle_blocks_segment,
    obstacle_clearance_violation,
)


@dataclass
class PreferencePreset:
    name: str
    description: str
    ref_points: np.ndarray


PRESET_PREFERENCES: dict[str, PreferencePreset] = {
    "survival_first": PreferencePreset(
        "survival_first",
        "Prioritise terrain masking; tolerate sub-optimal range",
        np.array([[0.10, 0.50, 0.50]]),
    ),
    "balanced": PreferencePreset(
        "balanced",
        "Equal weight on masking, range, and approach quality",
        np.array([[0.30, 0.30, 0.30]]),
    ),
    "aggressive": PreferencePreset(
        "aggressive",
        "Prioritise optimal weapon range; accept some exposure",
        np.array([[0.60, 0.10, 0.30]]),
    ),
}


def _encode(
    x: float,
    y: float,
    z_agl: float,
    target: np.ndarray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Map world XY and AGL to normalised range/bearing/AGL variables."""
    dx = x - float(target[0])
    dy = y - float(target[1])
    return np.array([
        (math.hypot(dx, dy) - r_min) / max(1e-6, r_max - r_min),
        (math.degrees(math.atan2(dy, dx)) % 360.0) / 360.0,
        (z_agl - z_min) / max(1e-6, z_max - z_min),
    ])


def _decode(
    x_norm: np.ndarray,
    target: np.ndarray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    terrain_fn: TerrainFn | None = None,
) -> np.ndarray:
    """Decode to world XYZ; the third decision variable is always AGL."""
    r = r_min + float(x_norm[0]) * (r_max - r_min)
    theta = float(x_norm[1]) * 2.0 * math.pi
    x = float(target[0]) + r * math.cos(theta)
    y = float(target[1]) + r * math.sin(theta)
    agl = z_min + float(x_norm[2]) * (z_max - z_min)
    ground = float(terrain_fn(x, y)) if terrain_fn is not None else 0.0
    return np.array([x, y, ground + agl], dtype=float)


_Problem = None
_BASE_CONSTRAINT_NAMES = [*CONSTRAINT_NAMES, "ObstacleClearance"]


def _get_problem_class():
    global _Problem
    if _Problem is None:
        from pymoo.core.problem import Problem as PymooProblem
        _Problem = PymooProblem
    return _Problem


class FREAProblem:
    """Optimisation problem for one aircraft, one weapon, and one target."""

    constraint_names = _BASE_CONSTRAINT_NAMES

    def __init__(
        self,
        target: np.ndarray,
        terrain_fn: TerrainFn,
        threats: list[np.ndarray] | None = None,
        designator: np.ndarray | None = None,
        obstacles: list[ObstacleVolume] | None = None,
        start_position: np.ndarray | None = None,
        r_min: float = 16.0,
        r_max: float = 80.0,
        z_min: float = 0.3,
        z_max: float = 3.0,
        min_range: float = 16.0,
        max_range: float = 80.0,
        optimal_range: float = 52.0,
        max_designation_angle_deg: float = (
            DEFAULT_MAX_DESIGNATION_ANGLE_DEG
        ),
        obstacle_clearance: float = 0.5,
        obstacle_vertical_clearance: float = 0.3,
        requires_designator: bool = False,
        allowed_xy_polygons: list[np.ndarray] | None = None,
    ):
        self.target = np.array(target, dtype=float)
        self.terrain_fn = terrain_fn
        self.threats = threats or []
        self.designator = designator
        self.obstacles = obstacles or []
        self.start_position = (
            None if start_position is None
            else np.array(start_position, dtype=float)
        )
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.optimal_range = float(optimal_range)
        self.max_designation_angle_deg = float(max_designation_angle_deg)
        self.obstacle_clearance = float(obstacle_clearance)
        self.obstacle_vertical_clearance = float(
            obstacle_vertical_clearance
        )
        self.requires_designator = bool(requires_designator)
        self.allowed_xy_polygons = [
            np.asarray(polygon, dtype=float)
            for polygon in (allowed_xy_polygons or [])
            if len(polygon) >= 3
        ]
        self.constraint_names = list(_BASE_CONSTRAINT_NAMES)
        if self.allowed_xy_polygons:
            self.constraint_names.append("AttackRegion")
        self._objectives = [f_exposure, f_range_deviation, f_approach]

    def decode(self, x_norm: np.ndarray) -> np.ndarray:
        return _decode(
            x_norm,
            self.target,
            self.r_min,
            self.r_max,
            self.z_min,
            self.z_max,
            self.terrain_fn,
        )

    def evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate normalised candidates into objectives and constraints."""
        n = X.shape[0]
        F = np.zeros((n, 3), dtype=float)
        G = np.zeros((n, len(self.constraint_names)), dtype=float)

        for i in range(n):
            pos = self.decode(X[i])
            for j, objective in enumerate(self._objectives):
                F[i, j] = objective(
                    pos,
                    self.target,
                    self.threats,
                    self.terrain_fn,
                    self.min_range,
                    self.max_range,
                    self.optimal_range,
                    self.start_position,
                )

            for j, constraint in enumerate(CONSTRAINT_FUNCTIONS):
                G[i, j] = constraint(
                    pos,
                    self.target,
                    self.designator,
                    self.min_range,
                    self.max_range,
                    self.max_designation_angle_deg,
                    self.terrain_fn,
                )

            blocking_id = obstacle_blocks_segment(
                pos, self.target, self.obstacles
            )
            if blocking_id is not None:
                G[i, 0] = max(1.0, G[i, 0])
            if self.requires_designator and self.designator is None:
                G[i, 1] = max(1.0, G[i, 1])
            obstacle_index = len(CONSTRAINT_FUNCTIONS)
            G[i, obstacle_index] = obstacle_clearance_violation(
                pos,
                self.obstacles,
                self.obstacle_clearance,
                self.obstacle_vertical_clearance,
            )
            if self.allowed_xy_polygons:
                G[i, obstacle_index + 1] = _attack_region_violation(
                    pos,
                    self.allowed_xy_polygons,
                )

        out["F"] = F
        out["G"] = G

    def grid_search(
        self,
        n_samples: int = 400,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the feasible non-dominated set from deterministic sampling."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.0, 1.0, size=(n_samples, 3))
        region_X = self._sample_region_candidates(rng, n_samples)
        if len(region_X):
            X = np.vstack([X, region_X])
        out: dict[str, np.ndarray] = {}
        self.evaluate(X, out)
        G = out["G"]
        feasible = np.all(G <= 0.0, axis=1)
        if not np.any(feasible):
            return (
                np.zeros((0, 3)),
                np.zeros((0, 3)),
                np.zeros((0, len(self.constraint_names))),
            )

        X_f, F_f, G_f = X[feasible], out["F"][feasible], G[feasible]
        pareto_mask = _non_dominated_rank(F_f) == 0
        return X_f[pareto_mask], F_f[pareto_mask], G_f[pareto_mask]

    def run_pymoo(
        self,
        preference: str = "balanced",
        pop_size: int = 100,
        n_gen: int = 40,
        seed: int = 42,
    ) -> dict:
        """Run pymoo R-NSGA-II and return only feasible solutions."""
        from pymoo.algorithms.moo.rnsga2 import RNSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize

        outer = self
        ProblemCls = _get_problem_class()

        class _InnerProblem(ProblemCls):
            def __init__(self_):
                super().__init__(
                    n_var=3,
                    n_obj=3,
                    n_ieq_constr=len(outer.constraint_names),
                    xl=np.zeros(3),
                    xu=np.ones(3),
                )

            def _evaluate(self_, X, out, *args, **kwargs):
                outer.evaluate(X, out)

        preset = PRESET_PREFERENCES.get(
            preference, PRESET_PREFERENCES["balanced"]
        )
        algorithm = RNSGA2(
            ref_points=preset.ref_points,
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=0.1, eta=20),
            epsilon=0.01,
        )
        result = minimize(
            _InnerProblem(),
            algorithm,
            ("n_gen", n_gen),
            seed=seed,
            verbose=False,
        )
        X = result.X if result.X is not None else np.zeros((0, 3))
        F = result.F if result.F is not None else np.zeros((0, 3))
        G = (
            result.G if result.G is not None
            else np.zeros((len(X), len(self.constraint_names)))
        )
        if len(X):
            X = np.atleast_2d(X)
            F = np.atleast_2d(F)
            G = np.atleast_2d(G)
            feasible = np.all(G <= 0.0, axis=1)
            X, F, G = X[feasible], F[feasible], G[feasible]
        return {"X": X, "F": F, "G": G}

    def _sample_region_candidates(
        self,
        rng: np.random.Generator,
        n_samples: int,
    ) -> np.ndarray:
        if not self.allowed_xy_polygons:
            return np.zeros((0, 3), dtype=float)
        target_count = max(100, n_samples)
        candidates: list[np.ndarray] = []
        max_attempts = target_count * 30
        attempts = 0
        while len(candidates) < target_count and attempts < max_attempts:
            attempts += 1
            polygon = self.allowed_xy_polygons[
                int(rng.integers(0, len(self.allowed_xy_polygons)))
            ]
            xs = polygon[:, 0]
            ys = polygon[:, 1]
            x = float(rng.uniform(float(np.min(xs)), float(np.max(xs))))
            y = float(rng.uniform(float(np.min(ys)), float(np.max(ys))))
            if not _point_in_any_polygon_xy(x, y, self.allowed_xy_polygons):
                continue
            z_agl = float(rng.uniform(self.z_min, self.z_max))
            encoded = _encode(
                x,
                y,
                z_agl,
                self.target,
                self.r_min,
                self.r_max,
                self.z_min,
                self.z_max,
            )
            if np.all((encoded >= 0.0) & (encoded <= 1.0)):
                candidates.append(encoded)
        if not candidates:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(candidates)


def _non_dominated_rank(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    ranks = np.zeros(n, dtype=int)
    dominated_by = np.zeros(n, dtype=int)
    dominates = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(F[i], F[j]):
                dominates[i].add(j)
            elif _dominates(F[j], F[i]):
                dominated_by[i] += 1
    front = [i for i in range(n) if dominated_by[i] == 0]
    rank = 0
    while front:
        for i in front:
            ranks[i] = rank
        next_front: list[int] = []
        for i in front:
            for j in dominates[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front.append(j)
        front = next_front
        rank += 1
    return ranks


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def select_by_reference(
    X_pareto: np.ndarray,
    F_pareto: np.ndarray,
    ref_point: np.ndarray,
    top_k: int = 5,
) -> np.ndarray:
    """Rank Pareto solutions by distance to the commander's aspiration."""
    _ = X_pareto
    if len(F_pareto) == 0:
        return np.array([], dtype=int)
    ref = np.array(ref_point, dtype=float).reshape(1, -1)
    distance = np.max(np.abs(F_pareto - ref), axis=1)
    return np.argsort(distance)[: min(top_k, len(F_pareto))]


def _attack_region_violation(
    pos: np.ndarray,
    polygons: list[np.ndarray],
) -> float:
    return 0.0 if _point_in_any_polygon_xy(pos[0], pos[1], polygons) else 1.0


def _point_in_any_polygon_xy(
    x: float,
    y: float,
    polygons: list[np.ndarray],
) -> bool:
    return any(_point_in_polygon_xy(x, y, polygon) for polygon in polygons)


def _point_in_polygon_xy(x: float, y: float, polygon: np.ndarray) -> bool:
    points = np.asarray(polygon, dtype=float)
    if len(points) < 3:
        return False
    if np.allclose(points[0, :2], points[-1, :2]):
        points = points[:-1]
    inside = False
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = float(points[i, 0]), float(points[i, 1])
        xj, yj = float(points[j, 0]), float(points[j, 1])
        if _point_on_segment(x, y, xi, yi, xj, yj):
            return True
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_at_y = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= x_at_y:
                inside = not inside
        j = i
    return inside


def _point_on_segment(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    if dot < -1e-9:
        return False
    length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return dot <= length_sq + 1e-9
