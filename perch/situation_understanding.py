"""Deterministic situation-to-text conversion for attack-region reasoning."""

from __future__ import annotations

import math
from typing import Any

from brain.domain.task import StrikeTask

from .models import resolve_weapon_envelope_km


class SituationUnderstanding:
    """Turn live world state into an auditable LLM input document."""

    def describe(
        self,
        world: dict[str, Any],
        task: StrikeTask,
        target: dict[str, Any],
        target_km: tuple[float, float],
        target_geo: tuple[float, float],
        target_geo_known: bool,
        agents: Any = None,
    ) -> str:
        platform_id = str(task.platform)
        munition = str(task.munition).upper()
        envelope = resolve_weapon_envelope_km(
            world, platform_id, munition
        )
        platform = self._platform_state(world, platform_id, agents or [])
        platform_km = self._position_km(platform, world)
        target_type = target.get("type", target.get("category", "unknown"))
        target_threat = target.get("threat", "unknown")
        terrain = world.get("terrain_summary") or target.get(
            "terrain_summary",
            "未提供文字摘要；后续 FREA 使用实时高程函数检查地形和通视",
        )
        weather = self._compact_mapping(world.get("weather", {}))
        manifest = dict(
            getattr(task, "assigned_munitions", {})
            or {munition: int(task.qty)}
        )

        if target_geo_known:
            lon, lat = target_geo
            coordinate_text = (
                f"目标地理坐标=[{lon:.6f}, {lat:.6f}]；"
                "输出多边形必须使用真实[经度, 纬度]。"
            )
        else:
            coordinate_text = (
                "场景无地理基准；以目标为局部经纬度原点"
                "[0.000000, 0.000000]，仅输出目标附近的小范围偏移。"
            )

        platform_text = "未知"
        if platform_km is not None:
            distance = math.hypot(
                platform_km[0] - target_km[0],
                platform_km[1] - target_km[1],
            )
            platform_text = (
                f"位置=({platform_km[0]:.2f}, {platform_km[1]:.2f})km，"
                f"距目标={distance:.2f}km"
            )
        velocity = self._first_not_none(
            platform.get("velocity_scene"), platform.get("velocity")
        )
        if velocity is not None:
            platform_text += f"，速度向量={self._format_vector(velocity)}"

        target_motion = self._first_not_none(
            target.get("velocity_km_s"),
            target.get("velocity_scene"),
            target.get("velocity"),
        )
        motion_text = (
            "静止或未提供"
            if target_motion is None
            else self._format_vector(target_motion)
        )
        designator = self._first_not_none(
            target.get("designator_pos"),
            target.get("designator_pos_scene"),
        )
        designator_text = (
            "未提供"
            if designator is None
            else self._format_vector(designator)
        )

        return "\n".join([
            "[任务]",
            f"平台={platform_id}；角色={task.role}；目标={task.target}；"
            f"主用弹药={munition}；数量={task.qty}；弹药清单={manifest}。",
            "[目标态势]",
            f"类型={target_type}；场景位置=({target_km[0]:.2f}, "
            f"{target_km[1]:.2f})km；威胁等级={target_threat}；"
            f"运动向量={motion_text}。",
            "[我方平台态势]",
            f"{platform_text}。",
            "[实时武器约束]",
            f"武器={envelope['name']}；有效射程="
            f"{envelope['min_range_km']:.1f}-"
            f"{envelope['max_range_km']:.1f}km；最佳射程="
            f"{envelope['optimal_range_km']:.1f}km；"
            f"制导={list(envelope['guidance'])}；"
            f"需要照射组={envelope['requires_designator']}。",
            "[威胁态势]",
            self._threat_summary(world, str(task.target), target_km),
            "[地形、障碍与天气]",
            f"地形={terrain}；附近障碍={self._obstacle_summary(world, target_km)}；"
            f"天气={weather}。",
            "[引导与FREA硬约束]",
            f"照射组位置={designator_text}；照射夹角上限="
            f"{float(world.get('max_designation_angle_deg', 45.0)):.1f}度；"
            f"攻击高度={float(world.get('attack_min_agl_m', 30.0)):.0f}-"
            f"{float(world.get('attack_max_agl_m', 300.0)):.0f}m AGL；"
            "候选区域内最终点必须同时满足武器斜距、目标通视、"
            "障碍净空和地形高度约束。",
            "[坐标输出]",
            coordinate_text,
            "请综合以上实时态势推荐1-3个直升机攻击阵位区域。",
        ])

    @classmethod
    def _threat_summary(
        cls,
        world: dict[str, Any],
        target_id: str,
        target_km: tuple[float, float],
    ) -> str:
        threats: list[tuple[float, str]] = []
        for item in world.get("targets", []):
            item_id = str(item.get("tid", item.get("target_id", "")))
            if item_id == target_id or not item.get("alive", True):
                continue
            level = float(item.get("threat", 0.0))
            if level <= 0.0:
                continue
            position = cls._position_km(item, world)
            if position is None:
                distance_text = "距离未知"
                distance = float("inf")
            else:
                distance = math.hypot(
                    position[0] - target_km[0],
                    position[1] - target_km[1],
                )
                distance_text = f"距任务目标{distance:.2f}km"
            kind = item.get("type", item.get("category", "unknown"))
            threats.append((
                distance,
                f"{item_id}({kind}, 威胁={level:.2f}, {distance_text})",
            ))
        threats.sort(key=lambda pair: pair[0])
        return "；".join(text for _, text in threats[:5]) or "无其他已知有效威胁"

    @classmethod
    def _obstacle_summary(
        cls,
        world: dict[str, Any],
        target_km: tuple[float, float],
    ) -> str:
        obstacles: list[tuple[float, str]] = []
        for index, item in enumerate(world.get("obstacles", [])):
            if not isinstance(item, dict):
                continue
            position = cls._position_km(item, world)
            if position is None:
                continue
            distance = math.hypot(
                position[0] - target_km[0],
                position[1] - target_km[1],
            )
            name = item.get("obstacle_id", item.get("name", f"obstacle_{index}"))
            category = item.get("category", "unknown")
            obstacles.append((distance, f"{name}({category}, {distance:.2f}km)"))
        obstacles.sort(key=lambda pair: pair[0])
        return "；".join(text for _, text in obstacles[:5]) or "无已建模障碍"

    @staticmethod
    def _platform_state(
        world: dict[str, Any], platform_id: str, agents: Any
    ) -> dict[str, Any]:
        states = world.get("platform_states", [])
        if isinstance(states, dict):
            state = states.get(platform_id, {})
            return dict(state) if isinstance(state, dict) else {}
        for state in states:
            if not isinstance(state, dict):
                continue
            state_id = state.get(
                "platform_id", state.get("entity_id", state.get("pid"))
            )
            if str(state_id) == platform_id:
                return state
        for agent in agents:
            if str(getattr(agent, "pid", "")) != platform_id:
                continue
            return {
                "position_km": getattr(agent, "position", None),
                "platform_type": getattr(agent, "type", "unknown"),
                "altitude_km": getattr(agent, "altitude_km", None),
            }
        return {}

    @staticmethod
    def _position_km(
        item: dict[str, Any], world: dict[str, Any]
    ) -> tuple[float, float] | None:
        raw = SituationUnderstanding._first_not_none(
            item.get("position_km"), item.get("pos")
        )
        if raw is not None:
            return float(raw[0]), float(raw[1])
        raw = SituationUnderstanding._first_not_none(
            item.get("position_scene"), item.get("pos_scene")
        )
        if raw is None:
            return None
        meters_per_unit = float(world.get("meters_per_unit", 100.0))
        half = float(world.get("map_size_km", 300.0)) * 0.5
        scale = meters_per_unit / 1000.0
        return half + float(raw[0]) * scale, half + float(raw[1]) * scale

    @staticmethod
    def _compact_mapping(value: Any) -> str:
        if not isinstance(value, dict) or not value:
            return "未提供"
        return ", ".join(
            f"{key}={value[key]}" for key in sorted(value)[:8]
        )

    @staticmethod
    def _format_vector(value: Any) -> str:
        try:
            return "[" + ", ".join(f"{float(v):.3f}" for v in value[:3]) + "]"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _first_not_none(*values: Any) -> Any:
        return next((value for value in values if value is not None), None)
