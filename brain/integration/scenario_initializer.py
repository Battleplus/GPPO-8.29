"""Normalize external scenario input into Brain's mission context."""

from __future__ import annotations

from typing import Any

from brain.domain.agent import AgentSpec
from brain.integration.mission_input import parse_aoi, parse_aoi_list


class ScenarioInitializer:
    """Build the canonical ``MissionContext.world_state`` view.

    The initializer is intentionally conservative: it preserves caller-provided
    fields, fills aliases used by MILP/multi-AOI modules, and only creates
    ``AgentSpec`` objects when the context does not already have agents.
    """

    DEFAULT_AOI = {"row": 3, "col": 4}
    DEFAULT_STAGING = [150.0, -50.0]

    def normalize(self, context: Any) -> dict[str, Any]:
        world = getattr(context, "world_state", None)
        if world is None:
            raise TypeError("context must expose a mutable world_state dict")

        self._normalize_aois(context, world)
        self._normalize_environment(world)
        self._normalize_agents(context, world)
        self._normalize_targets(world)
        self._normalize_runtime_state(context, world)
        return world

    def _normalize_aois(self, context: Any, world: dict[str, Any]) -> None:
        raw_aois = self._first_present(
            world,
            (
                "aois",
                "task_areas",
                "mission_areas",
                "recon_areas",
                "commander_AOI",
                "commander_aoi",
            ),
        )
        if raw_aois:
            raw_items = (
                [raw_aois]
                if isinstance(raw_aois, dict)
                else parse_aoi_list(raw_aois)
            )
            aois = [self._normalize_aoi(item, index) for index, item in enumerate(raw_items)]
            world["aois"] = aois
            world["aoi"] = {"row": aois[0]["row"], "col": aois[0]["col"]}
        else:
            aoi = self._first_present(
                world,
                ("aoi", "task_area", "mission_area"),
            ) or self.DEFAULT_AOI
            aoi = self._normalize_aoi(aoi, 0)
            world["aoi"] = {"row": aoi["row"], "col": aoi["col"]}
            aois = [aoi]
            world["aois"] = aois
        world["commander_AOI"] = [item["id"] for item in aois]
        world["task_areas"] = list(aois)
        if hasattr(context, "aois"):
            context.aois = aois

    @staticmethod
    def _normalize_aoi(raw: Any, index: int) -> dict[str, Any]:
        parsed = parse_aoi(raw)
        raw_dict = dict(raw) if isinstance(raw, dict) else {}
        row = int(parsed.get("row", 3))
        col = int(parsed.get("col", 4))
        aoi_id = str(parsed.get("id") or raw_dict.get("aoi") or f"A_{row}_{col}")
        return {
            **raw_dict,
            "id": aoi_id,
            "row": row,
            "col": col,
            "priority": float(parsed.get("priority", raw_dict.get("priority", 1.0))),
            "target_prior": float(raw_dict.get("target_prior", 0.25)),
            "target_value": float(raw_dict.get("target_value", raw_dict.get("value", 0.5))),
            "target_threat": float(raw_dict.get("target_threat", raw_dict.get("threat", 0.5))),
            "index": int(raw_dict.get("index", index)),
        }

    def _normalize_environment(self, world: dict[str, Any]) -> None:
        world.setdefault("staging_position", list(self.DEFAULT_STAGING))
        world.setdefault("coordinate_frame", "kilometer_grid")

        if "grid_weather" in world and "weather" not in world:
            world["weather"] = dict(world["grid_weather"])
        if "weather" in world and "grid_weather" not in world:
            world["grid_weather"] = dict(world["weather"])
        world.setdefault("weather", {f"c{index}": 0.0 for index in range(5)})
        world["weather"] = self._numeric_cell_map(world["weather"], float)
        world.setdefault("grid_weather", dict(world["weather"]))
        world["grid_weather"] = self._numeric_cell_map(world["grid_weather"], float)

        if "grid_terrain" in world and "terrain" not in world:
            world["terrain"] = dict(world["grid_terrain"])
        if "terrain" in world and "grid_terrain" not in world:
            world["grid_terrain"] = dict(world["terrain"])
        world.setdefault("terrain", {f"c{index}": 0 for index in range(5)})
        world["terrain"] = self._numeric_cell_map(world["terrain"], int)
        world.setdefault("grid_terrain", dict(world["terrain"]))
        world["grid_terrain"] = self._numeric_cell_map(world["grid_terrain"], int)

    def _normalize_agents(self, context: Any, world: dict[str, Any]) -> None:
        agents = list(getattr(context, "agents", []) or [])
        if not agents:
            agents = self._agents_from_world_platforms(world)
            if agents:
                context.agents = agents

        world["platforms"] = [
            {
                "pid": str(agent.pid),
                "type": str(agent.type),
                "pos": [float(agent.position[0]), float(agent.position[1])],
                "sensors": list(agent.sensors),
                "munitions": dict(agent.munitions),
                "alt": float(agent.altitude_km),
                "lost": bool(agent.lost),
            }
            for agent in list(getattr(context, "agents", []) or [])
        ]

    def _agents_from_world_platforms(self, world: dict[str, Any]) -> list[AgentSpec]:
        raw = world.get("platforms", [])
        if isinstance(raw, list):
            return [
                self._agent_from_platform_dict(item)
                for item in raw
                if isinstance(item, dict) and item.get("pid")
            ]
        if isinstance(raw, dict):
            return self._agents_from_platform_config(raw, world)
        return []

    @staticmethod
    def _agent_from_platform_dict(raw: dict[str, Any]) -> AgentSpec:
        pos = raw.get("pos", raw.get("position", [150.0, -50.0]))
        platform_type = str(raw.get("type", "UAV"))
        default_sensors = (
            ["EO", "SAR", "ESM"] if platform_type == "UAV" else ["MMW", "EOIR"]
        )
        return AgentSpec(
            pid=str(raw["pid"]),
            type=platform_type,
            position=(float(pos[0]), float(pos[1])),
            sensors=list(raw.get(
                "sensors",
                raw.get("sensors_mounted", default_sensors),
            )),
            munitions=dict(raw.get(
                "munitions",
                {"HF": 0, "RKT": 0, "GUN": 0}
                if platform_type == "UAV"
                else {"HF": 16, "RKT": 76, "GUN": 1200},
            )),
            altitude_km=float(raw.get("alt", raw.get("altitude_km", 2.0))),
            lost=bool(raw.get("lost", False)),
        )

    def _agents_from_platform_config(
        self,
        raw: dict[str, Any],
        world: dict[str, Any],
    ) -> list[AgentSpec]:
        staging = world.get("staging_position", self.DEFAULT_STAGING)
        agents: list[AgentSpec] = []
        for platform_type, cfg in raw.items():
            if not isinstance(cfg, dict):
                continue
            count = int(cfg.get("count", 0))
            prefix = "U" if str(platform_type) == "UAV" else "H"
            for index in range(1, count + 1):
                item = {
                    "pid": f"{prefix}{index}",
                    "type": str(platform_type),
                    "pos": cfg.get("pos", staging),
                    "sensors": cfg.get("sensors"),
                    "munitions": cfg.get("munitions"),
                    "alt": cfg.get("alt"),
                    "lost": False,
                }
                item = {key: value for key, value in item.items() if value is not None}
                agents.append(self._agent_from_platform_dict(item))
        return agents

    @staticmethod
    def _first_present(
        world: dict[str, Any],
        keys: tuple[str, ...],
    ) -> Any:
        for key in keys:
            value = world.get(key)
            if not value:
                continue
            if key in {"commander_AOI", "commander_aoi"}:
                try:
                    return parse_aoi_list(value)
                except ValueError:
                    return value
            return value
        return None

    @staticmethod
    def _numeric_cell_map(raw: Any, caster) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        result: dict[str, Any] = {}
        for key, value in raw.items():
            key_str = str(key)
            if key_str.startswith("_"):
                continue
            try:
                result[key_str] = caster(value)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _normalize_targets(world: dict[str, Any]) -> None:
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(world.get("targets", []) or []):
            target = dict(raw)
            tid = str(target.get("tid") or target.get("target_id") or f"g{index + 1}")
            pos = target.get("pos", target.get("position", [0.0, 0.0]))
            target.update({
                "tid": tid,
                "target_id": tid,
                "type": str(target.get("type", "AV")),
                "pos": [float(pos[0]), float(pos[1])],
                "value": float(target.get("value", 0.5)),
                "threat": float(target.get("threat", 0.5)),
                "confirmed": bool(target.get("confirmed", False)),
                "alive": bool(target.get("alive", True)),
            })
            normalized.append(target)
        world["targets"] = normalized

    @staticmethod
    def _normalize_runtime_state(context: Any, world: dict[str, Any]) -> None:
        context.aoi_route_state = world.get(
            "aoi_route_state", getattr(context, "aoi_route_state", None)
        )
        context.execution_feedback = world.get(
            "execution_feedback", getattr(context, "execution_feedback", None)
        )
        context.pending_strike_targets = list(
            dict.fromkeys(
                str(item)
                for item in world.get(
                    "pending_strike_targets",
                    getattr(context, "pending_strike_targets", []),
                )
            )
        )
        context.engaged_targets = set(
            str(item)
            for item in world.get(
                "engaged_targets",
                getattr(context, "engaged_targets", set()),
            )
        )
        context.active_action_plans = dict(
            world.get(
                "active_action_plans",
                getattr(context, "active_action_plans", {}),
            )
        )
        context.runtime_events = list(
            world.get(
                "runtime_events",
                getattr(context, "runtime_events", []),
            )
        )
        world["aoi_route_state"] = context.aoi_route_state
        world["execution_feedback"] = context.execution_feedback
        world["pending_strike_targets"] = list(context.pending_strike_targets)
        world["engaged_targets"] = sorted(context.engaged_targets)
        world["active_action_plans"] = dict(context.active_action_plans)
        world["runtime_events"] = list(context.runtime_events)
