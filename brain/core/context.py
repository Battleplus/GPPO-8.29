"""MissionContext — the single source of truth carried through every state."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from .states import MissionState
from ..domain.agent import AgentSpec
from ..domain.task import TaskSpec


@dataclass
class MissionContext:
    """All state that the mission FSM reads and writes.

    The context is passed by reference to every transition handler and
    adapter call — there is only **one** instance per mission.
    """

    # -- Identity ---------------------------------------------------------
    mission_id: str
    """Unique mission identifier (e.g. a UUID or timestamp slug)."""

    # -- Current state ----------------------------------------------------
    state: MissionState = MissionState.INIT
    """The FSM's current state.  Updated by the FSM on every transition."""

    # -- Force composition ------------------------------------------------
    agents: list[AgentSpec] = field(default_factory=list)
    """All platforms (UAV + HELI) participating in the mission."""

    # -- Task queue -------------------------------------------------------
    tasks: list[TaskSpec] = field(default_factory=list)
    """Current task list (recon + strike), typically populated by MILP."""

    # -- World / environment ----------------------------------------------
    world_state: dict[str, Any] = field(default_factory=dict)
    """Free-form dictionary describing targets, weather, terrain, AOIs, etc.

    Expected top-level keys (matching the MILP input convention):

    * ``"targets"`` — list of ``TargetInfo``-compatible dicts
    * ``"aoi"`` — ``{"row": …, "col": …}``
    * ``"weather"`` — grid-cell weather dict
    * ``"terrain"`` — grid-cell terrain levels
    * ``"staging_position"`` — ``[x_km, y_km]``
    """

    aois: list[dict[str, Any]] = field(default_factory=list)
    """Normalized reconnaissance areas.  Single-AOI missions contain one item."""

    aoi_route_state: dict[str, Any] | None = None
    """Opaque route state returned by the multi-AOI allocator."""

    execution_feedback: dict[str, Any] | None = None
    """Latest execution feedback passed back into multi-AOI allocation."""

    pending_strike_targets: list[str] = field(default_factory=list)
    """Confirmed, alive targets waiting for strike allocation."""

    engaged_targets: set[str] = field(default_factory=set)
    """Targets already assigned to an active strike plan."""

    active_action_plans: dict[str, Any] = field(default_factory=dict)
    """Active strike plans keyed by target id."""

    runtime_events: list[dict[str, Any]] = field(default_factory=list)
    """Execution-thread events consumed by the Brain event handlers."""

    # -- Allocation results ------------------------------------------------
    recon_allocation: Any | None = None
    """Raw MILP reconnaissance allocation (e.g. list of ``ReconTask``)."""

    action_allocation: Any | None = None
    """Raw MILP strike allocation (e.g. list of ``StrikeTask``)."""

    # -- Formation plans --------------------------------------------------
    recon_formation_plan: Any | None = None
    """MPPI reconnaissance formation plan (``FormationPlan``)."""

    action_formation_plan: Any | None = None
    """MPPI action formation plan (``FormationPlan``)."""

    # -- Execution results ------------------------------------------------
    recon_result: Any | None = None
    """Result payload received after reconnaissance execution."""

    selected_positions: Any | None = None
    """Positions chosen by ``PositionSelector`` (list of ``Position``)."""

    # -- Retry bookkeeping ------------------------------------------------
    retry_count: int = 0
    max_retry: int = 3
    last_failed_state: str | None = None
    last_error: str = ""

    # -- Audit trail ------------------------------------------------------
    history: list[dict[str, Any]] = field(default_factory=list)
    """Ordered log of every state transition and significant event."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def record_event(
        self,
        event_type: str,
        detail: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append a timestamped entry to the history log."""
        self.history.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "mission_id": self.mission_id,
                "state": self.state.value,
                "event": event_type,
                "detail": detail,
                **(extra or {}),
            }
        )

    def has_pending_action_tasks(self) -> bool:
        """Return True only when at least one target was confirmed during recon."""
        targets = self.world_state.get("targets", [])
        return any(
            t.get("alive", True) and t.get("confirmed", False)
            for t in targets
        )


def make_context(
    mission_id: str = "",
    agents: list[AgentSpec] | None = None,
    world_state: dict[str, Any] | None = None,
    max_retry: int = 3,
) -> MissionContext:
    """Factory that creates a :class:`MissionContext` with sensible defaults."""
    return MissionContext(
        mission_id=mission_id or datetime.datetime.now().strftime("M%Y%m%d_%H%M%S"),
        agents=agents or [],
        world_state=world_state or {},
        max_retry=max_retry,
    )
