"""Explainable one-aircraft/one-target attack-position selector."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np

from brain.domain.position import Position
from brain.domain.result import AlgorithmResult
from brain.domain.task import one_to_one_strike_tasks

from .frea_optimizer import (
    FREAProblem,
    PRESET_PREFERENCES,
    _point_in_polygon_xy,
    select_by_reference,
)
from .models import (
    ObstacleVolume,
    WeaponEnvelope,
    resolve_weapon_envelope_km,
)
from .terrain_analyzer import (
    TerrainFn,
    nearest_obstacle_clearance,
    obstacle_blocks_segment,
    terrain_raycast,
)

logger = logging.getLogger(__name__)

_DEFAULT_METERS_PER_UNIT = 100.0
_DEFAULT_MAP_SIZE_KM = 300.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "off", "no"}
    return bool(value)


class FREAPositionSelector:
    """Select ranked positions for explicit aircraft/target bindings."""

    def __init__(
        self,
        preference: str = "balanced",
        top_k: int = 3,
        use_pymoo: bool = True,
    ) -> None:
        if preference not in PRESET_PREFERENCES:
            raise ValueError(
                f"Unknown preference '{preference}'. "
                f"Choose from: {list(PRESET_PREFERENCES)}"
            )
        self.preference = preference
        self.top_k = max(1, int(top_k))
        self.use_pymoo = bool(use_pymoo)

    def select(
        self,
        context: Any,
        action_allocation: Any,
    ) -> AlgorithmResult:
        """Optimise one independent position set per aircraft/target task."""
        world = getattr(context, "world_state", {})
        targets = {
            str(t.get("tid", t.get("target_id", ""))): t
            for t in world.get("targets", [])
            if t.get("alive", True)
        }
        if not targets:
            return AlgorithmResult.fail(
                "No alive targets for position selection"
            )

        if action_allocation:
            bindings = one_to_one_strike_tasks(action_allocation)
        else:
            bindings = [
                SimpleNamespace(
                    platform=f"UNASSIGNED_{index}",
                    target=target_id,
                    munition="HF",
                    qty=1,
                    assigned_munitions={"HF": 1},
                    role="strike",
                )
                for index, target_id in enumerate(targets)
                if targets[target_id].get("confirmed", False)
            ]
        bindings = [
            task for task in bindings if str(task.target) in targets
        ]
        if not bindings:
            return AlgorithmResult.fail(
                "No one-to-one aircraft/target assignments available"
            )

        terrain_fn = self._make_terrain_fn(world)
        obstacles = self._build_obstacles(world, terrain_fn)
        all_positions: list[Position] = []

        for task in bindings:
            platform_id = str(task.platform)
            target_id = str(task.target)
            target = targets[target_id]
            target_pos = self._target_scene_pos(target, world, terrain_fn)
            start_position = self._platform_scene_pos(
                context, platform_id, world, terrain_fn
            )
            weapon = self._weapon_envelope(
                world, platform_id, str(task.munition)
            )
            designator = self._designator_scene_pos(
                target, world, terrain_fn
            )
            threats = self._build_threats(
                world, terrain_fn, exclude_target_id=target_id
            )
            meters_per_unit = self._meters_per_unit(world)
            min_agl = (
                float(world.get("attack_min_agl_m", 30.0))
                / meters_per_unit
            )
            max_agl = (
                float(world.get("attack_max_agl_m", 300.0))
                / meters_per_unit
            )
            attack_regions = self._target_attack_regions(world, target_id)

            problem_kwargs = {
                "target": target_pos,
                "terrain_fn": terrain_fn,
                "threats": threats,
                "designator": designator,
                "obstacles": obstacles,
                "start_position": start_position,
                "r_min": max(0.05, weapon.min_range * 0.80),
                "r_max": max(weapon.max_range, weapon.min_range + 0.05),
                "z_min": max(0.01, min_agl),
                "z_max": max(min_agl + 0.01, max_agl),
                "min_range": weapon.min_range,
                "max_range": weapon.max_range,
                "optimal_range": weapon.optimal_range,
                "max_designation_angle_deg": float(
                    world.get("max_designation_angle_deg", 45.0)
                ),
                "obstacle_clearance": (
                    float(world.get("attack_obstacle_clearance_m", 50.0))
                    / meters_per_unit
                ),
                "obstacle_vertical_clearance": (
                    float(world.get(
                        "attack_obstacle_vertical_clearance_m", 30.0
                    ))
                    / meters_per_unit
                ),
                "requires_designator": weapon.requires_designator,
            }
            positions: list[Position] = []
            for attack_region in attack_regions:
                allowed_xy_polygons = self._allowed_xy_polygons([
                    attack_region
                ])
                if not allowed_xy_polygons:
                    continue
                region_kwargs = {
                    **problem_kwargs,
                    "allowed_xy_polygons": allowed_xy_polygons,
                }
                positions = self._optimise(
                    problem=FREAProblem(**region_kwargs),
                    platform_id=platform_id,
                    target_id=target_id,
                    task=task,
                    weapon=weapon,
                    obstacles=obstacles,
                    meters_per_unit=meters_per_unit,
                    attack_regions=[attack_region],
                )
                if positions:
                    break

            if not attack_regions:
                positions = self._optimise(
                    problem=FREAProblem(**problem_kwargs),
                    platform_id=platform_id,
                    target_id=target_id,
                    task=task,
                    weapon=weapon,
                    obstacles=obstacles,
                    meters_per_unit=meters_per_unit,
                )
            elif not positions and not _as_bool(
                world.get("attack_region_strict", True)
            ):
                logger.warning(
                    "No feasible attack position in any ranked region for "
                    "platform=%s, target=%s; explicit global fallback enabled",
                    platform_id,
                    target_id,
                )
                positions = self._optimise(
                    problem=FREAProblem(**problem_kwargs),
                    platform_id=platform_id,
                    target_id=target_id,
                    task=task,
                    weapon=weapon,
                    obstacles=obstacles,
                    meters_per_unit=meters_per_unit,
                    region_constraint_relaxed=True,
                )
            if not positions:
                scope = " in the selected regions" if attack_regions else ""
                return AlgorithmResult.fail(
                    f"No feasible attack position{scope} for "
                    f"platform={platform_id}, target={target_id}, "
                    f"munition={task.munition}"
                )
            all_positions.extend(positions)

        logger.info(
            "FREA selector: %d positions for %d one-to-one assignments "
            "(preference=%s)",
            len(all_positions),
            len(bindings),
            self.preference,
        )
        return AlgorithmResult.ok(all_positions)

    def _optimise(
        self,
        problem: FREAProblem,
        platform_id: str,
        target_id: str,
        task: Any,
        weapon: WeaponEnvelope,
        obstacles: list[ObstacleVolume],
        meters_per_unit: float,
        attack_regions: list[dict[str, Any]] | None = None,
        region_constraint_relaxed: bool = False,
    ) -> list[Position]:
        preset = PRESET_PREFERENCES[self.preference]
        optimiser_name = "grid_search"
        if self.use_pymoo:
            try:
                result = problem.run_pymoo(preference=self.preference)
                X, F, G = result["X"], result["F"], result["G"]
                optimiser_name = "pymoo_RNSGA2"
            except Exception as exc:
                logger.warning(
                    "pymoo failed (%s), falling back to deterministic search",
                    exc,
                )
                X, F, G = problem.grid_search()
        else:
            X, F, G = problem.grid_search()

        if len(X) == 0 and getattr(problem, "allowed_xy_polygons", None):
            X, F, G = problem.grid_search(n_samples=800)
            optimiser_name = "grid_search_region"

        if len(X) == 0:
            return []
        feasible = np.all(G <= 0.0, axis=1)
        X, F, G = X[feasible], F[feasible], G[feasible]
        if len(X) == 0:
            return []

        indices = select_by_reference(
            X, F, preset.ref_points[0], top_k=self.top_k
        )
        positions: list[Position] = []
        for rank, idx in enumerate(indices):
            pos = problem.decode(X[idx])
            ground_z = float(problem.terrain_fn(
                float(pos[0]), float(pos[1])
            ))
            agl_m = (float(pos[2]) - ground_z) * meters_per_unit
            range_km = (
                float(np.linalg.norm(pos - problem.target))
                * meters_per_unit / 1000.0
            )
            nearest_id, nearest_clearance = nearest_obstacle_clearance(
                pos, obstacles
            )
            nearest_clearance_m = (
                None if not np.isfinite(nearest_clearance)
                else nearest_clearance * meters_per_unit
            )
            terrain_los = terrain_raycast(
                pos, problem.target, problem.terrain_fn
            )
            blocking_obstacle = obstacle_blocks_segment(
                pos, problem.target, obstacles
            )
            f_vec = [float(value) for value in F[idx]]
            g_vec = [float(value) for value in G[idx]]
            constraint_status = {
                name: {
                    "satisfied": value <= 0.0,
                    "violation": max(0.0, value),
                }
                for name, value in zip(
                    problem.constraint_names, g_vec, strict=False
                )
            }
            manifest = dict(
                getattr(task, "assigned_munitions", {})
                or {str(task.munition): int(task.qty)}
            )
            attack_region = self._region_for_position(
                pos,
                attack_regions or [],
            )
            explanation = [
                (
                    f"{platform_id} 使用 {task.munition} 攻击 "
                    f"{target_id}，弹药分配={manifest}"
                ),
                (
                    f"目标斜距 {range_km:.2f} km，位于武器包线 "
                    f"{weapon.min_range * meters_per_unit / 1000.0:.2f}"
                    f"–{weapon.max_range * meters_per_unit / 1000.0:.2f} km"
                ),
                f"阵位离地高度 {agl_m:.1f} m",
                (
                    "目标视线通畅"
                    if terrain_los and blocking_obstacle is None
                    else f"目标视线受阻：{blocking_obstacle or 'terrain'}"
                ),
                (
                    "附近无建模障碍物"
                    if nearest_id is None
                    else (
                        f"最近障碍物 {nearest_id}，表面净空 "
                        f"{nearest_clearance_m:.1f} m"
                    )
                ),
                (
                    "综合代价：暴露 "
                    f"{f_vec[0]:.3f}，射程偏差 {f_vec[1]:.3f}，"
                    f"进场暴露 {f_vec[2]:.3f}"
                ),
            ]
            if attack_region:
                explanation.append(
                    "攻击阵位区域："
                    f"score={float(attack_region.get('score', 0.0)):.2f}，"
                    f"{attack_region.get('reasoning', '')}"
                )
            elif region_constraint_relaxed:
                explanation.append(
                    "候选区域内无可行点，已放宽为全局 FREA 选点"
                )

            region_metadata: dict[str, Any] = {}
            if attack_region:
                region_metadata.update({
                    "attack_region_score": float(
                        attack_region.get("score", 0.0)
                    ),
                    "attack_region_reasoning": str(
                        attack_region.get("reasoning", "")
                    ),
                    "attack_region_source": str(
                        attack_region.get("source", "")
                    ),
                    "attack_region_rank": int(
                        attack_region.get("rank", 0)
                    ),
                })
            if region_constraint_relaxed or (
                attack_regions and attack_region is None
            ):
                region_metadata["region_constraint_relaxed"] = True

            positions.append(Position(
                pos_id=(
                    f"{platform_id}_{target_id}_POS_{rank:02d}"
                ),
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]),
                kind="attack",
                metadata={
                    "platform_id": platform_id,
                    "target_id": target_id,
                    "munition": str(task.munition),
                    "assigned_munitions": manifest,
                    "rank": int(rank),
                    "coordinate_frame": "isaac_scene",
                    "altitude_reference": "MSL_scene_z",
                    "agl_m": float(agl_m),
                    "terrain_z": ground_z,
                    "target_range_km": range_km,
                    "weapon_envelope": {
                        "name": weapon.name,
                        "min_range_km": (
                            weapon.min_range * meters_per_unit / 1000.0
                        ),
                        "max_range_km": (
                            weapon.max_range * meters_per_unit / 1000.0
                        ),
                        "optimal_range_km": (
                            weapon.optimal_range
                            * meters_per_unit / 1000.0
                        ),
                        "guidance": list(weapon.guidance),
                        "requires_designator": (
                            weapon.requires_designator
                        ),
                    },
                    "f_exposure": f_vec[0],
                    "f_range": f_vec[1],
                    "f_approach": f_vec[2],
                    "objective_scores": {
                        "exposure_cost": f_vec[0],
                        "range_deviation_cost": f_vec[1],
                        "approach_exposure_cost": f_vec[2],
                    },
                    "constraints": constraint_status,
                    "g_violation": float(sum(
                        max(0.0, value) for value in g_vec
                    )),
                    "target_los_clear": bool(
                        terrain_los and blocking_obstacle is None
                    ),
                    "blocking_obstacle": blocking_obstacle,
                    "nearest_obstacle_id": nearest_id,
                    "nearest_obstacle_clearance_m": (
                        nearest_clearance_m
                    ),
                    "selection_reason": (
                        "满足全部硬约束，并在指挥偏好参考点下的"
                        "帕累托候选中排名靠前"
                    ),
                    "explanation": explanation,
                    "preference": self.preference,
                    "optimiser": optimiser_name,
                    **region_metadata,
                },
            ))
        return positions

    @staticmethod
    def _target_attack_regions(
        world: dict[str, Any],
        target_id: str,
    ) -> list[dict[str, Any]]:
        mode = str(world.get("attack_region_mode", "llm")).lower()
        if mode in {"disabled", "off", "none", "false", "0"}:
            return []
        raw = world.get("attack_regions", {})
        if isinstance(raw, dict):
            regions = raw.get(target_id, [])
        else:
            regions = raw
        valid = [
            region for region in regions
            if isinstance(region, dict)
        ]
        return sorted(
            valid,
            key=lambda region: (
                -float(region.get("score", 0.0)),
                int(region.get("rank", 0)),
            ),
        )

    @staticmethod
    def _allowed_xy_polygons(
        attack_regions: list[dict[str, Any]],
    ) -> list[np.ndarray]:
        polygons: list[np.ndarray] = []
        for region in attack_regions:
            raw = region.get("polygon_scene", [])
            try:
                polygon = np.asarray(raw, dtype=float)
            except Exception:
                continue
            if polygon.ndim != 2 or polygon.shape[0] < 3:
                continue
            polygons.append(polygon[:, :2])
        return polygons

    @staticmethod
    def _region_for_position(
        pos: np.ndarray,
        attack_regions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        x, y = float(pos[0]), float(pos[1])
        for region in attack_regions:
            raw = region.get("polygon_scene", [])
            try:
                polygon = np.asarray(raw, dtype=float)
            except Exception:
                continue
            if polygon.ndim != 2 or polygon.shape[0] < 3:
                continue
            if _point_in_polygon_xy(x, y, polygon[:, :2]):
                return region
        return None

    @staticmethod
    def _meters_per_unit(world: dict) -> float:
        return float(world.get(
            "meters_per_unit", _DEFAULT_METERS_PER_UNIT
        ))

    @classmethod
    def _map_size_km(cls, world: dict) -> float:
        if "map_size_km" in world:
            return float(world["map_size_km"])
        if "map_size_units" in world:
            return (
                float(world["map_size_units"])
                * cls._meters_per_unit(world) / 1000.0
            )
        return _DEFAULT_MAP_SIZE_KM

    @classmethod
    def _km_to_scene(
        cls,
        km_x: float,
        km_y: float,
        world: dict,
    ) -> tuple[float, float]:
        scale = 1000.0 / cls._meters_per_unit(world)
        half_km = cls._map_size_km(world) * 0.5
        return (
            (float(km_x) - half_km) * scale,
            (float(km_y) - half_km) * scale,
        )

    @staticmethod
    def _make_terrain_fn(world: dict) -> TerrainFn:
        supplied = world.get("terrain_fn")
        if callable(supplied):
            logger.info("FREA terrain: using Isaac-supplied terrain_fn")
            return supplied
        try:
            from scenes.air_combat_scene import terrain_height

            map_size = float(world.get("map_size_units", 3000.0))
            meters_per_unit = float(world.get(
                "meters_per_unit", _DEFAULT_METERS_PER_UNIT
            ))
            visual_height = world.get("terrain_visual_height_units")
            if visual_height is None:
                mountain_height_m = float(
                    world.get("mountain_height_m", 1500.0)
                )
                exaggeration = float(
                    world.get("terrain_vertical_exaggeration", 10.0)
                )
                visual_height = (
                    mountain_height_m / meters_per_unit * exaggeration
                )
            logger.info(
                "FREA terrain: using scenes.air_combat_scene.terrain_height"
            )
            return lambda x, y: float(terrain_height(
                x, y, map_size, float(visual_height)
            ))
        except Exception as exc:
            logger.warning(
                "FREA terrain unavailable (%s); using flat fallback", exc
            )
            return lambda x, y: 0.0

    @classmethod
    def _target_scene_pos(
        cls,
        target: dict,
        world: dict,
        terrain_fn: TerrainFn,
    ) -> np.ndarray:
        for key in ("position_scene", "pos_scene"):
            if key in target:
                raw = target[key]
                x, y = float(raw[0]), float(raw[1])
                z = (
                    float(raw[2]) if len(raw) > 2
                    else float(terrain_fn(x, y))
                )
                return np.array([x, y, z], dtype=float)
        pos_km = target.get("pos", [0.0, 0.0])
        x, y = cls._km_to_scene(pos_km[0], pos_km[1], world)
        target_height = (
            float(target.get("target_height_m", 2.0))
            / cls._meters_per_unit(world)
        )
        return np.array(
            [x, y, float(terrain_fn(x, y)) + target_height],
            dtype=float,
        )

    @classmethod
    def _designator_scene_pos(
        cls,
        target: dict,
        world: dict,
        terrain_fn: TerrainFn,
    ) -> np.ndarray | None:
        if target.get("designator_pos_scene") is not None:
            raw = target["designator_pos_scene"]
            x, y = float(raw[0]), float(raw[1])
            z = (
                float(raw[2]) if len(raw) > 2
                else float(terrain_fn(x, y)) + 0.02
            )
            return np.array([x, y, z], dtype=float)
        raw = target.get("designator_pos")
        if raw is None:
            return None
        x, y = cls._km_to_scene(raw[0], raw[1], world)
        z = (
            float(terrain_fn(x, y)) + 0.02
            if len(raw) <= 2
            else float(raw[2]) * 1000.0 / cls._meters_per_unit(world)
        )
        return np.array([x, y, z], dtype=float)

    @classmethod
    def _platform_scene_pos(
        cls,
        context: Any,
        platform_id: str,
        world: dict,
        terrain_fn: TerrainFn,
    ) -> np.ndarray:
        states = world.get("platform_states", [])
        if isinstance(states, dict):
            states = [
                {"platform_id": key, **value}
                for key, value in states.items()
            ]
        for state in states:
            sid = state.get(
                "platform_id", state.get("entity_id", state.get("pid"))
            )
            if str(sid) != platform_id:
                continue
            raw = state.get(
                "position_scene", state.get("position")
            )
            if raw is not None and len(raw) >= 3:
                return np.array(raw[:3], dtype=float)

        for agent in getattr(context, "agents", []):
            if str(getattr(agent, "pid", "")) != platform_id:
                continue
            pos = getattr(agent, "position", (0.0, 0.0))
            x, y = cls._km_to_scene(pos[0], pos[1], world)
            agl = (
                float(getattr(agent, "altitude_km", 0.3))
                * 1000.0 / cls._meters_per_unit(world)
            )
            return np.array(
                [x, y, float(terrain_fn(x, y)) + agl],
                dtype=float,
            )

        staging = world.get(
            "staging_position",
            [cls._map_size_km(world) * 0.5] * 2,
        )
        x, y = cls._km_to_scene(staging[0], staging[1], world)
        agl = (
            float(world.get("action_cruise_agl_m", 300.0))
            / cls._meters_per_unit(world)
        )
        return np.array(
            [x, y, float(terrain_fn(x, y)) + agl],
            dtype=float,
        )

    @classmethod
    def _build_threats(
        cls,
        world: dict,
        terrain_fn: TerrainFn,
        exclude_target_id: str,
    ) -> list[np.ndarray]:
        threats: list[np.ndarray] = []
        for target in world.get("targets", []):
            target_id = str(
                target.get("tid", target.get("target_id", ""))
            )
            if (
                target_id == exclude_target_id
                or not target.get("alive", True)
                or float(target.get("threat", 0.5)) <= 0.0
            ):
                continue
            threats.append(cls._target_scene_pos(
                target, world, terrain_fn
            ))
        return threats

    @classmethod
    def _build_obstacles(
        cls,
        world: dict,
        terrain_fn: TerrainFn,
    ) -> list[ObstacleVolume]:
        result: list[ObstacleVolume] = []
        meters_per_unit = cls._meters_per_unit(world)
        for index, raw in enumerate(world.get("obstacles", [])):
            if isinstance(raw, dict):
                getter = raw.get
            else:
                getter = lambda name, default=None, obj=raw: getattr(
                    obj, name, default
                )
            position = getter("position_scene")
            if position is None:
                position = getter("position", [0.0, 0.0, 0.0])
            center = np.array(position[:3], dtype=float)
            category = str(getter("category", "obstacle"))
            radius = getter("radius_units")
            if radius is None:
                radius_km = getter("radius_km")
                radius = (
                    float(radius_km) * 1000.0 / meters_per_unit
                    if radius_km is not None
                    else float(getter("radius", 0.0))
                )
            base_z = getter("base_z")
            if base_z is None:
                base_z = float(terrain_fn(
                    float(center[0]), float(center[1])
                ))
            top_z = getter("top_z")
            height = float(getter(
                "height_units", getter("height", 0.0)
            ))
            if top_z is None:
                if category == "mountain":
                    top_z = max(float(center[2]), height, float(base_z))
                else:
                    top_z = max(
                        float(center[2]),
                        float(base_z) + height,
                    )
            result.append(ObstacleVolume(
                obstacle_id=str(getter(
                    "obstacle_id", getter("name", f"obstacle_{index}")
                )),
                category=category,
                center=center,
                radius=max(0.0, float(radius)),
                base_z=float(base_z),
                top_z=float(top_z),
                blocks_los=bool(getter("blocks_los", True)),
            ))
        return result

    @classmethod
    def _weapon_envelope(
        cls,
        world: dict,
        platform_id: str,
        munition: str,
    ) -> WeaponEnvelope:
        settings = resolve_weapon_envelope_km(
            world, platform_id, munition
        )
        meters_per_unit = cls._meters_per_unit(world)
        scale = 1000.0 / meters_per_unit
        min_range = float(settings.get("min_range_km", 0.2)) * scale
        max_range = float(settings.get("max_range_km", 5.0)) * scale
        optimal = float(settings.get(
            "optimal_range_km",
            (min_range + max_range) * 0.5 / scale,
        )) * scale
        optimal = min(max(optimal, min_range), max_range)
        return WeaponEnvelope(
            name=str(settings.get("name", munition)),
            min_range=min_range,
            max_range=max_range,
            optimal_range=optimal,
            guidance=tuple(settings.get("guidance", ())),
            requires_designator=bool(
                settings.get("requires_designator", False)
            ),
        )
