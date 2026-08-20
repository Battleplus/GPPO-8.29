"""Mission FSM — state-transition engine.

Design principles
-----------------
* **Algorithm states** (those ending in ``_BY_MILP``, ``_BY_MPPI``, or
  ``POSITION_SELECTING``) execute their adapter call **immediately** on
  entry and auto-advance on success.
* **Waiting states** (``*_PLAN_READY``, ``*_EXECUTING``,
  ``WAIT_RECON_RESULT``) block until an external event is dispatched.
* **MILP failures** go straight to ``MISSION_FAILED`` (no retry).
* **MPPI / PositionSelector failures** enter ``REPLAN``, which retries
  up to ``max_retry`` times before giving up.
* Every state change is recorded in ``context.history``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .context import MissionContext
from .events import MissionEvent, MissionEventType
from .states import MissionState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Adapter = Any  # duck-typed: must expose allocate_recon / plan_recon_route / …
TransitionHandler = Callable[[MissionContext, MissionEvent], MissionState | None]


class MissionFSM:
    """Finite-state machine that orchestrates the mission pipeline.

    Parameters
    ----------
    context:
        The shared mission context (mutated in-place).
    milp:
        MILP adapter object exposing ``allocate_recon()`` and ``allocate_action()``.
    mppi:
        MPPI adapter object exposing ``plan_recon_route()`` and ``plan_action_route()``.
    position_selector:
        Position adapter object exposing ``select()``.
    """

    def __init__(
        self,
        context: MissionContext,
        milp: Adapter,
        mppi: Adapter,
        position_selector: Adapter,
    ) -> None:
        self.ctx = context
        self.milp = milp
        self.mppi = mppi
        self.position_selector = position_selector

        # -- Transition table --------------------------------------------------
        # Maps (current_state, event_type) → handler.
        # Handlers return the *next* state or None (illegal transition).
        self._event_handlers: dict[
            tuple[MissionState, MissionEventType], TransitionHandler
        ] = {
            # -- Setup ---------------------------------------------------------
            (MissionState.INIT, MissionEventType.START): self._on_start,
            # -- Recon pipeline event-driven -----------------------------------
            (
                MissionState.RECON_PLAN_READY,
                MissionEventType.RECON_PLAN_DISPATCHED,
            ): self._on_recon_plan_dispatched,
            (
                MissionState.RECON_EXECUTING,
                MissionEventType.RECON_FINISHED,
            ): self._on_recon_execution_finished,
            (
                MissionState.WAIT_RECON_RESULT,
                MissionEventType.RECON_RESULT_RECEIVED,
            ): self._on_recon_result_received,
            # -- Action pipeline event-driven ----------------------------------
            (
                MissionState.ACTION_PLAN_READY,
                MissionEventType.ACTION_PLAN_DISPATCHED,
            ): self._on_action_plan_dispatched,
            (
                MissionState.ACTION_EXECUTING,
                MissionEventType.ACTION_FINISHED,
            ): self._on_action_execution_finished,
            # -- Global --------------------------------------------------------
            (MissionState.INIT, MissionEventType.RESET): self._on_reset,
        }

        # -- Auto-transition table --------------------------------------------
        # States that execute work *immediately* on entry and transition without
        # waiting for an external event.
        self._auto_handlers: dict[MissionState, Callable[[], MissionState]] = {
            MissionState.RECON_ALLOCATING_BY_MILP: self._do_recon_allocate,
            MissionState.RECON_PLANNING_BY_MPPI: self._do_recon_plan,
            MissionState.UPDATE_WORLD_STATE: self._do_update_world,
            MissionState.ACTION_ALLOCATING_BY_MILP: self._do_action_allocate,
            MissionState.POSITION_SELECTING: self._do_position_select,
            MissionState.ACTION_PLANNING_BY_MPPI: self._do_action_plan,
            MissionState.REPLAN: self._do_replan,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(self, event: MissionEvent) -> MissionState:
        """Deliver an external event to the FSM.

        Returns the new state after the transition(s).  If this triggers
        a chain of auto-transitions the *final* state is returned.
        """
        handler = self._event_handlers.get((self.ctx.state, event.type_))
        if handler is None:
            self.ctx.record_event(
                "ILLEGAL_TRANSITION",
                detail=f"No handler for ({self.ctx.state.value}, {event.type_.value})",
            )
            logger.warning(
                "FSM illegal transition: %s + %s",
                self.ctx.state.value,
                event.type_.value,
            )
            return self.ctx.state

        next_state = handler(self.ctx, event)
        if next_state is None:
            return self.ctx.state

        self._transition_to(next_state, event)
        return self._run_auto_chain()

    def step_auto(self) -> MissionState:
        """Execute the current auto-state handler (if any).

        Call this when the FSM is sitting in an auto-transition state
        (e.g. after construction and ``start()`` has been called, or
        after an event-driven transition that leads into an auto state).
        """
        return self._run_auto_chain()

    @property
    def current_state(self) -> MissionState:
        return self.ctx.state

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_start(
        self, ctx: MissionContext, _event: MissionEvent
    ) -> MissionState:
        ctx.record_event("START", detail="Mission initiated")
        return MissionState.RECON_ALLOCATING_BY_MILP

    def _on_recon_plan_dispatched(
        self, ctx: MissionContext, _event: MissionEvent
    ) -> MissionState:
        ctx.record_event("RECON_PLAN_DISPATCHED", detail="Recon formation plan dispatched to platforms")
        return MissionState.RECON_EXECUTING

    def _on_recon_execution_finished(
        self, ctx: MissionContext, _event: MissionEvent
    ) -> MissionState:
        ctx.record_event("RECON_FINISHED", detail="Reconnaissance execution completed")
        return MissionState.WAIT_RECON_RESULT

    def _on_recon_result_received(
        self, ctx: MissionContext, event: MissionEvent
    ) -> MissionState:
        if event.data is not None:
            ctx.recon_result = event.data
        ctx.record_event(
            "RECON_RESULT_RECEIVED",
            detail=f"Recon result received: {bool(event.data)}",
        )
        return MissionState.UPDATE_WORLD_STATE

    def _on_action_plan_dispatched(
        self, ctx: MissionContext, _event: MissionEvent
    ) -> MissionState:
        ctx.record_event("ACTION_PLAN_DISPATCHED", detail="Action formation plan dispatched to platforms")
        return MissionState.ACTION_EXECUTING

    def _on_action_execution_finished(
        self, ctx: MissionContext, _event: MissionEvent
    ) -> MissionState:
        ctx.record_event("ACTION_FINISHED", detail="Action execution completed")
        return MissionState.MISSION_COMPLETE

    def _on_reset(
        self, ctx: MissionContext, _event: MissionEvent
    ) -> MissionState:
        ctx.record_event("RESET", detail="Resetting mission to INIT")
        ctx.state = MissionState.INIT
        ctx.retry_count = 0
        ctx.last_failed_state = None
        ctx.last_error = ""
        ctx.recon_allocation = None
        ctx.action_allocation = None
        ctx.recon_formation_plan = None
        ctx.action_formation_plan = None
        ctx.recon_result = None
        ctx.selected_positions = None
        return MissionState.INIT

    # ------------------------------------------------------------------
    # Auto-transition handlers (algorithm calls)
    # ------------------------------------------------------------------

    def _do_recon_allocate(self) -> MissionState:
        self.ctx.record_event("RECON_ALLOCATING", detail="Calling MILPTaskAllocator.allocate_recon()")
        result = self.milp.allocate_recon(self.ctx)
        if result.success:
            self.ctx.recon_allocation = result.data
            self.ctx.record_event(
                "RECON_ALLOCATED",
                detail="MILP recon allocation succeeded",
                extra={"allocation": str(result.data)[:200]},
            )
            return MissionState.RECON_PLANNING_BY_MPPI

        self.ctx.last_error = result.reason
        self.ctx.record_event(
            "ALGORITHM_FAILED",
            detail=f"MILP recon allocation failed: {result.reason}",
            extra={"algorithm": "MILP", "phase": "recon"},
        )
        return MissionState.MISSION_FAILED

    def _do_recon_plan(self) -> MissionState:
        self.ctx.record_event("RECON_PLANNING", detail="Calling MPPIFormationPlanner.plan_recon_route()")
        result = self.mppi.plan_recon_route(self.ctx, self.ctx.recon_allocation)
        if result.success:
            self.ctx.recon_formation_plan = result.data
            self.ctx.record_event(
                "RECON_PLANNED",
                detail="MPPI recon planning succeeded",
            )
            return MissionState.RECON_PLAN_READY

        self.ctx.last_error = result.reason
        self.ctx.last_failed_state = MissionState.RECON_PLANNING_BY_MPPI.value
        self.ctx.record_event(
            "ALGORITHM_FAILED",
            detail=f"MPPI recon planning failed: {result.reason}",
            extra={"algorithm": "MPPI", "phase": "recon"},
        )
        return MissionState.REPLAN

    def _do_update_world(self) -> MissionState:
        self.ctx.record_event("UPDATE_WORLD", detail="Updating world state from recon result")

        recon = self.ctx.recon_result
        if recon and isinstance(recon, dict):
            # Apply recon findings to world_state targets
            detections = recon.get("detections", [])
            confirmed = recon.get("confirmed", [])
            targets = self.ctx.world_state.get("targets", [])
            for target in targets:
                tid = target.get("tid", target.get("target_id", ""))
                for det in detections:
                    if det.get("truth_id") == tid or det.get("target_id") == tid:
                        target["confirmed"] = True
                        break
                for cid in confirmed:
                    if cid == tid:
                        target["confirmed"] = True
                        break
                if (
                    target.get("alive", True)
                    and target.get("confirmed", False)
                    and str(tid) not in self.ctx.engaged_targets
                    and str(tid) not in self.ctx.pending_strike_targets
                ):
                    self.ctx.pending_strike_targets.append(str(tid))
            self.ctx.world_state["pending_strike_targets"] = list(
                self.ctx.pending_strike_targets
            )

        if self.ctx.has_pending_action_tasks():
            self.ctx.record_event("WORLD_UPDATED", detail="World state updated — pending action tasks remain")
            return MissionState.ACTION_ALLOCATING_BY_MILP

        self.ctx.record_event("WORLD_UPDATED", detail="World state updated — no pending tasks")
        return MissionState.MISSION_COMPLETE

    def _do_action_allocate(self) -> MissionState:
        self.ctx.record_event("ACTION_ALLOCATING", detail="Calling MILPTaskAllocator.allocate_action()")
        result = self.milp.allocate_action(self.ctx)
        if result.success:
            self.ctx.action_allocation = result.data
            assigned_targets = {
                str(getattr(task, "target", ""))
                for task in (result.data or [])
                if getattr(task, "target", "")
            }
            for target_id in assigned_targets:
                self.ctx.engaged_targets.add(target_id)
            self.ctx.pending_strike_targets = [
                target_id for target_id in self.ctx.pending_strike_targets
                if target_id not in assigned_targets
            ]
            self.ctx.world_state["engaged_targets"] = sorted(
                self.ctx.engaged_targets
            )
            self.ctx.world_state["pending_strike_targets"] = list(
                self.ctx.pending_strike_targets
            )
            self.ctx.record_event(
                "ACTION_ALLOCATED",
                detail="MILP action allocation succeeded",
            )
            return MissionState.POSITION_SELECTING

        self.ctx.last_error = result.reason
        self.ctx.record_event(
            "ALGORITHM_FAILED",
            detail=f"MILP action allocation failed: {result.reason}",
            extra={"algorithm": "MILP", "phase": "action"},
        )
        return MissionState.MISSION_FAILED

    def _do_position_select(self) -> MissionState:
        self.ctx.record_event("POSITION_SELECTING", detail="Calling PositionSelector.select()")
        result = self.position_selector.select(self.ctx, self.ctx.action_allocation)
        if result.success:
            self.ctx.selected_positions = result.data
            self.ctx.record_event(
                "POSITION_SELECTED",
                detail=f"Position selection succeeded: {result.data}",
            )
            return MissionState.ACTION_PLANNING_BY_MPPI

        self.ctx.last_error = result.reason
        self.ctx.last_failed_state = MissionState.POSITION_SELECTING.value
        self.ctx.record_event(
            "ALGORITHM_FAILED",
            detail=f"Position selection failed: {result.reason}",
            extra={"algorithm": "POSITION", "phase": "action"},
        )
        return MissionState.REPLAN

    def _do_action_plan(self) -> MissionState:
        self.ctx.record_event("ACTION_PLANNING", detail="Calling MPPIFormationPlanner.plan_action_route()")
        result = self.mppi.plan_action_route(
            self.ctx,
            self.ctx.action_allocation,
            selected_positions=self.ctx.selected_positions,
        )
        if result.success:
            self.ctx.action_formation_plan = result.data
            self.ctx.record_event(
                "ACTION_PLANNED",
                detail="MPPI action planning succeeded",
            )
            return MissionState.ACTION_PLAN_READY

        self.ctx.last_error = result.reason
        self.ctx.last_failed_state = MissionState.ACTION_PLANNING_BY_MPPI.value
        self.ctx.record_event(
            "ALGORITHM_FAILED",
            detail=f"MPPI action planning failed: {result.reason}",
            extra={"algorithm": "MPPI", "phase": "action"},
        )
        return MissionState.REPLAN

    def _do_replan(self) -> MissionState:
        self.ctx.retry_count += 1
        self.ctx.record_event(
            "REPLAN",
            detail=f"Retry {self.ctx.retry_count}/{self.ctx.max_retry} — "
            f"returning to {self.ctx.last_failed_state}",
        )

        if self.ctx.retry_count > self.ctx.max_retry:
            self.ctx.record_event(
                "MISSION_FAILED",
                detail=f"Max retries ({self.ctx.max_retry}) exceeded. "
                f"Last error: {self.ctx.last_error}",
            )
            return MissionState.MISSION_FAILED

        # Map the failed state back to its MissionState enum member
        failed = self.ctx.last_failed_state
        if failed is None:
            return MissionState.MISSION_FAILED

        try:
            return MissionState(failed)
        except ValueError:
            logger.error("Unknown failed state: %s", failed)
            return MissionState.MISSION_FAILED

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition_to(
        self, next_state: MissionState, event: MissionEvent | None = None
    ) -> None:
        old = self.ctx.state
        self.ctx.state = next_state
        logger.info(
            "FSM: %s → %s  (%s)",
            old.value,
            next_state.value,
            event.type_.value if event else "auto",
        )

    def _run_auto_chain(self) -> MissionState:
        """Keep executing auto-handlers until a waiting or terminal state."""
        max_chain = 20  # safety valve
        for _ in range(max_chain):
            handler = self._auto_handlers.get(self.ctx.state)
            if handler is None:
                break
            next_state = handler()
            self._transition_to(next_state)
            if next_state.is_waiting or next_state.is_terminal:
                break

        if self.ctx.state.is_terminal:
            self.ctx.record_event(
                "TERMINAL",
                detail=f"Mission ended in state {self.ctx.state.value}",
            )

        return self.ctx.state
