"""Adapter for explainable one-aircraft/one-target position selection."""

from __future__ import annotations

import logging
from typing import Any

from ..domain.position import Position
from ..domain.result import AlgorithmResult
from ..domain.task import one_to_one_strike_tasks
from perch.region_selector import AttackRegionSelector

logger = logging.getLogger(__name__)
_FREA_AVAILABLE: bool | None = None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "off", "no"}
    return bool(value)


def _check_frea() -> bool:
    global _FREA_AVAILABLE
    if _FREA_AVAILABLE is None:
        try:
            from perch import FREAPositionSelector  # noqa: F401
            _FREA_AVAILABLE = True
        except Exception:
            _FREA_AVAILABLE = False
            logger.exception("PositionSelector: perch is unavailable")
    return _FREA_AVAILABLE


class PositionSelector:
    """Thin adapter around :class:`perch.FREAPositionSelector`."""

    def __init__(
        self,
        preference: str = "balanced",
        top_k: int = 3,
        strike_position_mode: str = "optimized",
        attack_region_mode: str = "llm",
        attack_region_selector: Any | None = None,
    ) -> None:
        self._preference = preference
        self._top_k = top_k
        self._strike_position_mode = strike_position_mode
        self._attack_region_mode = attack_region_mode
        self._attack_region_selector = (
            attack_region_selector
            if attack_region_selector is not None
            else AttackRegionSelector(mode=attack_region_mode, top_k=top_k)
        )

    def select(self, context: Any, action_allocation: Any) -> AlgorithmResult:
        world = getattr(context, "world_state", {})
        mode = str(
            world.get("strike_position_mode", self._strike_position_mode)
        )
        if mode == "target_point":
            return self._select_target_point(context, action_allocation)

        region_failed = self._prepare_attack_regions(
            context,
            action_allocation,
        )
        if _check_frea():
            result = self._select_frea(context, action_allocation)
            if result.success:
                if region_failed:
                    self._mark_region_selection_failed(
                        result.data,
                        region_failed,
                    )
                return result
            strict_regions = world.get("attack_regions") and _as_bool(
                world.get("attack_region_strict", True)
            )
            if strict_regions:
                logger.warning(
                    "FREA returned failure in strict attack regions: %s",
                    result.reason,
                )
                return result
            logger.warning(
                "FREA returned failure (%s), using explainable fallback",
                result.reason,
            )
        result = self._select_placeholder(context, action_allocation)
        if result.success and region_failed:
            self._mark_region_selection_failed(result.data, region_failed)
        return result

    def _prepare_attack_regions(
        self,
        context: Any,
        action_allocation: Any,
    ) -> str:
        world = getattr(context, "world_state", {})
        mode = str(
            world.get("attack_region_mode", self._attack_region_mode)
        ).lower()
        if mode in {"disabled", "off", "none", "false", "0"}:
            world["attack_regions"] = {}
            world.pop("attack_region_errors", None)
            return ""
        if not one_to_one_strike_tasks(action_allocation or []):
            world["attack_regions"] = {}
            world.pop("attack_region_errors", None)
            return ""

        result = self._attack_region_selector.select_regions(
            context,
            action_allocation,
        )
        if result.success:
            return ""

        reason = result.reason or "Attack-region selection failed"
        logger.warning("%s; falling back to global Perch/FREA", reason)
        world["attack_regions"] = {}
        world["attack_region_errors"] = [reason]
        return reason

    @staticmethod
    def _mark_region_selection_failed(
        positions: Any,
        reason: str,
    ) -> None:
        for position in positions or []:
            metadata = getattr(position, "metadata", None)
            if isinstance(metadata, dict):
                metadata["region_selection_failed"] = True
                metadata["region_selection_error"] = reason

    def _select_frea(
        self,
        context: Any,
        action_allocation: Any,
    ) -> AlgorithmResult:
        from perch import FREAPositionSelector

        selector = FREAPositionSelector(
            preference=self._preference,
            top_k=self._top_k,
            use_pymoo=True,
        )
        return selector.select(context, action_allocation)

    @staticmethod
    def _km_to_scene(
        world: dict,
        x_km: float,
        y_km: float,
    ) -> tuple[float, float]:
        meters_per_unit = float(world.get("meters_per_unit", 100.0))
        map_size_km = float(world.get("map_size_km", 300.0))
        scale = 1000.0 / meters_per_unit
        half = map_size_km * 0.5
        return (x_km - half) * scale, (y_km - half) * scale

    def _select_target_point(
        self,
        context: Any,
        action_allocation: Any,
    ) -> AlgorithmResult:
        tasks = one_to_one_strike_tasks(action_allocation or [])
        if not tasks:
            return AlgorithmResult.fail(
                "No one-to-one strike assignments for target-point positions"
            )
        world = getattr(context, "world_state", {})
        target_by_id = {
            str(target.get("tid", target.get("target_id", ""))): target
            for target in world.get("targets", [])
        }
        positions: list[Position] = []
        for task in tasks:
            target = target_by_id.get(str(task.target), {})
            if target.get("position_scene") is not None:
                raw = target["position_scene"]
                x = float(raw[0])
                y = float(raw[1])
                z = float(raw[2]) + 3.0 if len(raw) > 2 else 3.0
            elif target.get("pos") is not None:
                x, y = self._km_to_scene(
                    world,
                    float(target["pos"][0]),
                    float(target["pos"][1]),
                )
                z = 3.0
            else:
                return AlgorithmResult.fail(
                    f"Target {task.target} has no position for target_point mode"
                )
            manifest = (
                task.assigned_munitions
                or {str(task.munition): int(task.qty)}
            )
            positions.append(Position(
                pos_id=f"{task.platform}_{task.target}_TARGET_POINT",
                x=x,
                y=y,
                z=z,
                kind="attack",
                metadata={
                    "platform_id": str(task.platform),
                    "target_id": str(task.target),
                    "munition": str(task.munition),
                    "assigned_munitions": dict(manifest),
                    "rank": 0,
                    "source": "target_point",
                    "g_violation": 0.0,
                    "selection_reason": "按 target_point 模式直接使用目标点",
                    "explanation": [
                        f"{task.platform} 直接规划到目标 {task.target}",
                        f"主弹种 {task.munition}，分配={manifest}",
                    ],
                },
            ))
        return AlgorithmResult.ok(positions)

    def _select_placeholder(
        self,
        context: Any,
        action_allocation: Any,
    ) -> AlgorithmResult:
        """Deterministic fallback that preserves one-to-one task identity."""
        tasks = one_to_one_strike_tasks(action_allocation or [])
        if not tasks:
            return AlgorithmResult.fail(
                "No one-to-one strike assignments for fallback positions"
            )
        world = getattr(context, "world_state", {})
        target_by_id = {
            str(target.get("tid", target.get("target_id", ""))): target
            for target in world.get("targets", [])
        }
        staging = world.get("staging_position", [150.0, 150.0])
        start_x, start_y = self._km_to_scene(
            world, float(staging[0]), float(staging[1])
        )
        positions: list[Position] = []
        for index, task in enumerate(tasks):
            target = target_by_id.get(str(task.target), {})
            if target.get("position_scene") is not None:
                raw = target["position_scene"]
                goal_x, goal_y = float(raw[0]), float(raw[1])
                goal_z = float(raw[2]) + 3.0 if len(raw) > 2 else 3.0
            elif target.get("pos") is not None:
                goal_x, goal_y = self._km_to_scene(
                    world,
                    float(target["pos"][0]),
                    float(target["pos"][1]),
                )
                goal_z = 3.0
            else:
                goal_x = start_x + 50.0 + index * 10.0
                goal_y = start_y + 40.0 + index * 10.0
                goal_z = 3.0
            manifest = (
                task.assigned_munitions
                or {str(task.munition): int(task.qty)}
            )
            positions.append(Position(
                pos_id=f"{task.platform}_{task.target}_FALLBACK",
                x=goal_x - 20.0,
                y=goal_y,
                z=goal_z,
                kind="attack",
                metadata={
                    "platform_id": str(task.platform),
                    "target_id": str(task.target),
                    "munition": str(task.munition),
                    "assigned_munitions": dict(manifest),
                    "rank": 0,
                    "source": "explainable_fallback",
                    "g_violation": 0.0,
                    "selection_reason": (
                        "优化器不可用；采用目标西侧确定性备用阵位"
                    ),
                    "explanation": [
                        f"{task.platform} 独立攻击 {task.target}",
                        f"主弹种 {task.munition}，分配={manifest}",
                        "该点仅用于保持任务链运行，需要执行前复核",
                    ],
                },
            ))
        return AlgorithmResult.ok(positions)
