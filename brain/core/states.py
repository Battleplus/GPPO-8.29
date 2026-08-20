"""Mission states — the 13 states of the task-flow pipeline."""

from __future__ import annotations

from enum import Enum, auto


class MissionState(str, Enum):
    """Every state the mission FSM can occupy.

    States are organised in three groups:

    * **Setup** — ``INIT``
    * **Reconnaissance pipeline** — ``RECON_ALLOCATING_BY_MILP`` …
      ``UPDATE_WORLD_STATE``
    * **Action pipeline** — ``ACTION_ALLOCATING_BY_MILP`` …
      ``ACTION_EXECUTING``
    * **Terminal / recovery** — ``REPLAN``, ``MISSION_COMPLETE``,
      ``MISSION_FAILED``
    """

    INIT = "INIT"

    # -- Reconnaissance pipeline ------------------------------------------
    RECON_ALLOCATING_BY_MILP = "RECON_ALLOCATING_BY_MILP"
    RECON_PLANNING_BY_MPPI = "RECON_PLANNING_BY_MPPI"
    RECON_PLAN_READY = "RECON_PLAN_READY"
    RECON_EXECUTING = "RECON_EXECUTING"
    WAIT_RECON_RESULT = "WAIT_RECON_RESULT"
    UPDATE_WORLD_STATE = "UPDATE_WORLD_STATE"

    # -- Action pipeline --------------------------------------------------
    ACTION_ALLOCATING_BY_MILP = "ACTION_ALLOCATING_BY_MILP"
    POSITION_SELECTING = "POSITION_SELECTING"
    ACTION_PLANNING_BY_MPPI = "ACTION_PLANNING_BY_MPPI"
    ACTION_PLAN_READY = "ACTION_PLAN_READY"
    ACTION_EXECUTING = "ACTION_EXECUTING"

    # -- Terminal / recovery ----------------------------------------------
    REPLAN = "REPLAN"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    MISSION_FAILED = "MISSION_FAILED"

    @property
    def is_terminal(self) -> bool:
        """True for states that end the mission (success or failure)."""
        return self in (MissionState.MISSION_COMPLETE, MissionState.MISSION_FAILED)

    @property
    def is_waiting(self) -> bool:
        """True for states that wait for an *external* event before advancing."""
        return self in (
            MissionState.RECON_PLAN_READY,
            MissionState.RECON_EXECUTING,
            MissionState.WAIT_RECON_RESULT,
            MissionState.ACTION_PLAN_READY,
            MissionState.ACTION_EXECUTING,
        )
