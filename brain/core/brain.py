"""MissionBrain — planning FSM plus optional simulation lifecycle."""

from __future__ import annotations

import logging
from typing import Any

from ..adapters.ppo_reallocator import PPOAllocationAdapter
from ..domain.result import AlgorithmResult
from ..integration import ScenarioInitializer, sync_context_from_air_combat_scene
from .context import MissionContext
from .events import MissionEvent, MissionEventType
from .mission_fsm import MissionFSM
from .states import MissionState

logger = logging.getLogger(__name__)


class MissionBrain:
    """Orchestrate planning and an optional externally supplied environment."""

    def __init__(
        self,
        context: MissionContext,
        milp_adapter: Any,
        mppi_adapter: Any,
        position_adapter: Any,
        environment: Any | None = None,
        ppo_reallocator: Any | None = None,
    ) -> None:
        self.context = context
        self.environment = (
            environment
            or context.world_state.get("environment")
        )
        self.ppo_reallocator = (
            ppo_reallocator or PPOAllocationAdapter()
        )
        self.initializer = ScenarioInitializer()
        self.fsm = MissionFSM(
            context, milp_adapter, mppi_adapter, position_adapter
        )
        self.initializer.normalize(self.context)

    def _initialize_environment(self) -> AlgorithmResult:
        """Initialize/reset environment and synchronize its first snapshot."""
        try:
            self.initializer.normalize(self.context)
            if self.environment is not None:
                initialized = bool(
                    getattr(self.environment, "initialized", False)
                )
                if initialized and hasattr(self.environment, "reset"):
                    scene = self.environment.reset()
                else:
                    scene = self.environment.initialize()
                self.context.world_state["isaac_scene"] = scene
            else:
                scene = self.context.world_state.get("isaac_scene")

            if scene is not None:
                sync_context_from_air_combat_scene(self.context, scene)
                self.initializer.normalize(self.context)
                self.context.record_event(
                    "ENVIRONMENT_INITIALIZED",
                    detail=(
                        "Isaac air-combat environment initialized and "
                        "planning inputs synchronized"
                    ),
                    extra={
                        "platform_count": len(
                            self.context.world_state.get(
                                "platform_states", []
                            )
                        ),
                        "target_count": len(
                            self.context.world_state.get("targets", [])
                        ),
                    },
                )
            else:
                self.initializer.normalize(self.context)
            return AlgorithmResult.ok(scene)
        except Exception as exc:
            logger.exception("Environment initialization failed")
            self.context.last_error = str(exc)
            self.context.record_event(
                "ENVIRONMENT_INIT_FAILED",
                detail=str(exc),
            )
            return AlgorithmResult.fail(str(exc))

    def start(self) -> MissionState:
        """Initialize the configured environment, then start planning."""
        if self.context.state == MissionState.INIT:
            initialized = self._initialize_environment()
            if not initialized.success:
                self.context.state = MissionState.MISSION_FAILED
                return self.context.state
        return self.fsm.dispatch(MissionEvent.start())

    def dispatch(self, event: MissionEvent) -> MissionState:
        if event.type_ == MissionEventType.TARGET_DETECTED:
            self.handle_target_detected(event.data or {})
            return self.context.state
        if event.type_ == MissionEventType.PLATFORM_LOST:
            self.handle_platform_loss(event.data or {})
            return self.context.state
        if event.type_ == MissionEventType.ATTACK_FINISHED:
            self.handle_attack_finished(event.data or {})
            return self.context.state

        new_state = self.fsm.dispatch(event)
        logger.info("Brain state: %s", new_state.value)
        return new_state

    def step_environment(self, dt: float = 1.0 / 30.0) -> AlgorithmResult:
        """Advance Isaac and refresh state without automatically rerunning MPPI."""
        if self.environment is None:
            return AlgorithmResult.fail(
                "No environment runtime configured"
            )
        try:
            scene = self.environment.step(dt)
            sync_context_from_air_combat_scene(self.context, scene)
            return AlgorithmResult.ok(scene)
        except Exception as exc:
            logger.exception("Environment step failed")
            return AlgorithmResult.fail(str(exc))

    def report_platform_loss(self, platform_id: str) -> AlgorithmResult:
        """Backward-compatible wrapper for execution code."""
        return self.handle_platform_loss({"platform_id": str(platform_id)})

    def handle_target_detected(self, payload: Any) -> AlgorithmResult:
        """Confirm a target, repair recon assignment, and prepare strike."""
        payload = self._payload_dict(payload)
        self._sync_scene_snapshot()
        target_id = self._payload_target_id(payload)
        if not target_id:
            return AlgorithmResult.fail("TARGET_DETECTED missing target_id")

        target = self._ensure_target(target_id, payload)
        target["confirmed"] = True
        target["alive"] = bool(target.get("alive", True))
        if (
            target["alive"]
            and target_id not in self.context.pending_strike_targets
            and target_id not in self.context.engaged_targets
        ):
            self.context.pending_strike_targets.append(target_id)

        platform_id = self._payload_platform_id(payload)
        ppo_result: AlgorithmResult | None = None
        recon_route_result: AlgorithmResult | None = None
        if platform_id:
            ppo_result = self.ppo_reallocator.handle_target_discovered(
                self.context, platform_id, target_id
            )
            if ppo_result.success:
                recon_route_result = self.fsm.mppi.plan_recon_route(
                    self.context, self.context.recon_allocation
                )
                if recon_route_result.success:
                    self.context.recon_formation_plan = recon_route_result.data
            else:
                self.context.record_event(
                    "PPO_REALLOCATION_FAILED",
                    detail=ppo_result.reason,
                    extra={
                        "platform_id": platform_id,
                        "target_id": target_id,
                        "event": "TARGET_DISCOVERED",
                    },
                )

        action_result = self._build_action_plan(
            [target_id],
            source="TARGET_DETECTED",
        )
        self._sync_runtime_to_world()
        if not action_result.success:
            return action_result

        self.context.record_event(
            "TARGET_DETECTED",
            detail=f"{target_id} confirmed; immediate strike plan prepared",
            extra={"platform_id": platform_id or ""},
        )
        return AlgorithmResult.ok({
            "target_id": target_id,
            "ppo_reallocation": (
                ppo_result.data
                if ppo_result is not None and ppo_result.success
                else None
            ),
            "recon_formation_plan": (
                recon_route_result.data
                if recon_route_result is not None and recon_route_result.success
                else self.context.recon_formation_plan
            ),
            "action_allocation": self.context.action_allocation,
            "selected_positions": self.context.selected_positions,
            "action_formation_plan": self.context.action_formation_plan,
        })

    def handle_platform_loss(self, payload: Any) -> AlgorithmResult:
        """Reallocate reconnaissance or strike work after a platform loss."""
        payload = self._payload_dict(payload)
        platform_id = self._payload_platform_id(payload)
        if not platform_id:
            return AlgorithmResult.fail("PLATFORM_LOST missing platform_id")

        self._sync_scene_snapshot()
        agent = self._agent_by_id(platform_id)
        if agent is not None:
            agent.lost = True

        platform_type = str(
            getattr(agent, "type", payload.get("platform_type", "UAV"))
        )
        if platform_type == "HELI":
            return self._handle_heli_loss(platform_id)

        result = self.ppo_reallocator.handle_platform_loss(
            self.context, str(platform_id)
        )
        if not result.success:
            self.context.record_event(
                "PPO_REALLOCATION_FAILED",
                detail=result.reason,
                extra={"platform_id": str(platform_id)},
            )
            self.context.recon_allocation = [
                task for task in (self.context.recon_allocation or [])
                if str(getattr(task, "platform", "")) != str(platform_id)
            ]

        route_result = self.fsm.mppi.plan_recon_route(
            self.context, self.context.recon_allocation
        )
        if not route_result.success:
            return AlgorithmResult.fail(
                "Recon reallocation succeeded but route planning "
                f"failed: {route_result.reason}"
            )
        self.context.recon_formation_plan = route_result.data
        self.context.state = MissionState.RECON_PLAN_READY
        self.context.record_event(
            "PPO_REALLOCATED",
            detail=(
                f"{platform_id} lost; reconnaissance routes were regenerated"
            ),
            extra={
                "platform_id": str(platform_id),
                "explanation": (result.data or {}).get("explanation", ""),
            },
        )
        self._sync_runtime_to_world()
        return AlgorithmResult.ok({
            **(result.data or {}),
            "recon_formation_plan": route_result.data,
        })

    def handle_attack_finished(self, payload: Any) -> AlgorithmResult:
        """Update target state after a strike and release follow-on work."""
        payload = self._payload_dict(payload)
        self._sync_scene_snapshot()
        target_id = self._payload_target_id(payload)
        if not target_id:
            return AlgorithmResult.fail("ATTACK_FINISHED missing target_id")

        destroyed = bool(
            payload.get("destroyed", payload.get("success", True))
        )
        target = self._ensure_target(target_id, payload)
        target["confirmed"] = True
        if destroyed:
            target["alive"] = False
            self._discard_target_from_queues(target_id)
            ppo_result = self.ppo_reallocator.handle_target_destroyed(
                self.context, target_id
            )
            if ppo_result.success:
                route_result = self.fsm.mppi.plan_recon_route(
                    self.context, self.context.recon_allocation
                )
                if route_result.success:
                    self.context.recon_formation_plan = route_result.data
            self.context.record_event(
                "ATTACK_FINISHED",
                detail=f"{target_id} destroyed",
                extra={"target_id": target_id},
            )
            self._sync_runtime_to_world()
            return AlgorithmResult.ok({
                "target_id": target_id,
                "destroyed": True,
                "ppo_reallocation": (
                    ppo_result.data if ppo_result.success else None
                ),
                "recon_formation_plan": self.context.recon_formation_plan,
            })

        target["alive"] = True
        self.context.engaged_targets.discard(target_id)
        if target_id not in self.context.pending_strike_targets:
            self.context.pending_strike_targets.append(target_id)
        self.context.record_event(
            "ATTACK_FAILED",
            detail=f"{target_id} remains alive; rebuilding strike plan",
            extra={"target_id": target_id},
        )
        return self._build_action_plan([target_id], source="ATTACK_FAILED")

    def _handle_heli_loss(self, platform_id: str) -> AlgorithmResult:
        affected = [
            str(getattr(task, "target", ""))
            for task in (self.context.action_allocation or [])
            if str(getattr(task, "platform", "")) == str(platform_id)
        ]
        affected = [target for target in affected if target]
        for target_id in affected:
            self.context.engaged_targets.discard(target_id)
            if target_id not in self.context.pending_strike_targets:
                self.context.pending_strike_targets.append(target_id)

        if not affected:
            self.context.record_event(
                "PLATFORM_LOST",
                detail=f"{platform_id} lost; no active strike target was bound",
                extra={"platform_id": platform_id},
            )
            self._sync_runtime_to_world()
            return AlgorithmResult.ok({"platform_id": platform_id})

        result = self._build_action_plan(affected, source="HELI_LOST")
        if result.success:
            self.context.record_event(
                "PLATFORM_LOST",
                detail=f"{platform_id} lost; strike targets reallocated",
                extra={"platform_id": platform_id, "targets": affected},
            )
        return result

    def _build_action_plan(
        self,
        target_ids: list[str] | None,
        *,
        source: str,
    ) -> AlgorithmResult:
        allocation = self._allocate_action(target_ids=target_ids)
        if not allocation.success:
            self.context.record_event(
                "ACTION_REALLOCATION_FAILED",
                detail=allocation.reason,
                extra={"source": source, "target_ids": target_ids or []},
            )
            return allocation

        self.context.action_allocation = allocation.data
        selected = self.fsm.position_selector.select(
            self.context, self.context.action_allocation
        )
        if not selected.success:
            return AlgorithmResult.fail(
                f"Position selection failed during {source}: {selected.reason}"
            )
        self.context.selected_positions = selected.data

        route = self.fsm.mppi.plan_action_route(
            self.context,
            self.context.action_allocation,
            selected_positions=self.context.selected_positions,
        )
        if not route.success:
            return AlgorithmResult.fail(
                f"Action route planning failed during {source}: {route.reason}"
            )
        self.context.action_formation_plan = route.data

        assigned_targets = {
            str(getattr(task, "target", ""))
            for task in (self.context.action_allocation or [])
            if getattr(task, "target", "")
        }
        for target_id in assigned_targets:
            self.context.engaged_targets.add(target_id)
            if target_id in self.context.pending_strike_targets:
                self.context.pending_strike_targets.remove(target_id)
            self.context.active_action_plans[target_id] = {
                "source": source,
                "allocation": [
                    task for task in (self.context.action_allocation or [])
                    if str(getattr(task, "target", "")) == target_id
                ],
                "formation_plan": self.context.action_formation_plan,
            }

        self.context.record_event(
            "ACTION_PLAN_UPDATED",
            detail=f"{source}: strike plan ready for {sorted(assigned_targets)}",
            extra={"target_ids": sorted(assigned_targets)},
        )
        self._sync_runtime_to_world()
        return AlgorithmResult.ok({
            "action_allocation": self.context.action_allocation,
            "selected_positions": self.context.selected_positions,
            "action_formation_plan": self.context.action_formation_plan,
        })

    def _allocate_action(self, target_ids: list[str] | None) -> AlgorithmResult:
        try:
            return self.fsm.milp.allocate_action(
                self.context, target_ids=target_ids
            )
        except TypeError:
            return self.fsm.milp.allocate_action(self.context)

    def _sync_scene_snapshot(self) -> None:
        scene = self.context.world_state.get("isaac_scene")
        if scene is not None:
            sync_context_from_air_combat_scene(self.context, scene)
        self.initializer.normalize(self.context)

    @staticmethod
    def _payload_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return dict(payload)
        if payload is None:
            return {}
        return {"platform_id": str(payload)}

    @staticmethod
    def _payload_target_id(payload: dict[str, Any]) -> str:
        return str(
            payload.get("target_id")
            or payload.get("truth_id")
            or payload.get("tid")
            or ""
        )

    @staticmethod
    def _payload_platform_id(payload: dict[str, Any]) -> str:
        return str(
            payload.get("platform_id")
            or payload.get("uav_id")
            or payload.get("heli_id")
            or payload.get("pid")
            or ""
        )

    def _agent_by_id(self, platform_id: str) -> Any | None:
        return next(
            (
                agent for agent in self.context.agents
                if str(getattr(agent, "pid", "")) == str(platform_id)
            ),
            None,
        )

    def _ensure_target(
        self,
        target_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        targets = self.context.world_state.setdefault("targets", [])
        for target in targets:
            tid = str(target.get("tid", target.get("target_id", "")))
            if tid == str(target_id):
                return target
        pos = payload.get("pos", payload.get("position", [0.0, 0.0]))
        target = {
            "tid": str(target_id),
            "target_id": str(target_id),
            "type": str(payload.get("target_type", payload.get("type", "AV"))),
            "pos": [float(pos[0]), float(pos[1])],
            "value": float(payload.get("value", 0.5)),
            "threat": float(payload.get("threat", 0.5)),
            "confirmed": False,
            "alive": True,
        }
        targets.append(target)
        return target

    def _discard_target_from_queues(self, target_id: str) -> None:
        self.context.engaged_targets.discard(target_id)
        self.context.pending_strike_targets = [
            item for item in self.context.pending_strike_targets
            if item != target_id
        ]
        self.context.active_action_plans.pop(target_id, None)

    def _sync_runtime_to_world(self) -> None:
        world = self.context.world_state
        world["pending_strike_targets"] = list(
            self.context.pending_strike_targets
        )
        world["engaged_targets"] = sorted(self.context.engaged_targets)
        world["active_action_plans"] = dict(self.context.active_action_plans)
        world["aoi_route_state"] = self.context.aoi_route_state
        world["execution_feedback"] = self.context.execution_feedback
        world["runtime_events"] = list(self.context.runtime_events)

    def run_to_completion(self) -> MissionState:
        state = self.fsm.step_auto()
        while not state.is_waiting and not state.is_terminal:
            state = self.fsm.step_auto()
        return state

    def close(self) -> None:
        if self.environment is not None:
            closer = getattr(self.environment, "close", None)
            if callable(closer):
                closer()

    @property
    def current_state(self) -> MissionState:
        return self.fsm.current_state

    @property
    def is_terminal(self) -> bool:
        return self.current_state.is_terminal

    def summary(self) -> str:
        ctx = self.context
        return (
            f"[{ctx.mission_id}] state={ctx.state.value} "
            f"retry={ctx.retry_count}/{ctx.max_retry} "
            f"history_entries={len(ctx.history)}"
        )
