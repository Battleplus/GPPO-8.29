"""Mission events — the 9 event types that drive the FSM."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MissionEventType(str, Enum):
    """Every event the FSM can receive."""

    START = "START"
    RECON_PLAN_DISPATCHED = "RECON_PLAN_DISPATCHED"
    RECON_FINISHED = "RECON_FINISHED"
    RECON_RESULT_RECEIVED = "RECON_RESULT_RECEIVED"
    ACTION_PLAN_DISPATCHED = "ACTION_PLAN_DISPATCHED"
    ACTION_FINISHED = "ACTION_FINISHED"
    TARGET_DETECTED = "TARGET_DETECTED"
    PLATFORM_LOST = "PLATFORM_LOST"
    RECON_CELL_DONE = "RECON_CELL_DONE"
    STRIKE_POSITION_REACHED = "STRIKE_POSITION_REACHED"
    ATTACK_FINISHED = "ATTACK_FINISHED"
    ALGORITHM_FAILED = "ALGORITHM_FAILED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    RESET = "RESET"


@dataclass
class MissionEvent:
    """An event dispatched to the FSM.

    Attributes:
        type_: The event type.
        data:  Optional payload (e.g. recon result dict, error message).
        source: Optional label indicating which component raised the event
                (useful for debugging).
    """

    type_: MissionEventType
    data: Any | None = None
    source: str = ""

    @classmethod
    def start(cls) -> MissionEvent:
        return cls(type_=MissionEventType.START)

    @classmethod
    def recon_plan_dispatched(cls) -> MissionEvent:
        return cls(type_=MissionEventType.RECON_PLAN_DISPATCHED)

    @classmethod
    def recon_finished(cls) -> MissionEvent:
        return cls(type_=MissionEventType.RECON_FINISHED)

    @classmethod
    def recon_result_received(cls, data: Any = None) -> MissionEvent:
        return cls(type_=MissionEventType.RECON_RESULT_RECEIVED, data=data)

    @classmethod
    def action_plan_dispatched(cls) -> MissionEvent:
        return cls(type_=MissionEventType.ACTION_PLAN_DISPATCHED)

    @classmethod
    def action_finished(cls) -> MissionEvent:
        return cls(type_=MissionEventType.ACTION_FINISHED)

    @classmethod
    def target_detected(cls, data: Any = None, source: str = "") -> MissionEvent:
        return cls(
            type_=MissionEventType.TARGET_DETECTED,
            data=data,
            source=source,
        )

    @classmethod
    def platform_lost(cls, data: Any = None, source: str = "") -> MissionEvent:
        return cls(
            type_=MissionEventType.PLATFORM_LOST,
            data=data,
            source=source,
        )

    @classmethod
    def recon_cell_done(cls, data: Any = None, source: str = "") -> MissionEvent:
        return cls(
            type_=MissionEventType.RECON_CELL_DONE,
            data=data,
            source=source,
        )

    @classmethod
    def strike_position_reached(
        cls, data: Any = None, source: str = ""
    ) -> MissionEvent:
        return cls(
            type_=MissionEventType.STRIKE_POSITION_REACHED,
            data=data,
            source=source,
        )

    @classmethod
    def attack_finished(cls, data: Any = None, source: str = "") -> MissionEvent:
        return cls(
            type_=MissionEventType.ATTACK_FINISHED,
            data=data,
            source=source,
        )

    @classmethod
    def algorithm_failed(cls, reason: str, source: str = "") -> MissionEvent:
        return cls(
            type_=MissionEventType.ALGORITHM_FAILED,
            data=reason,
            source=source,
        )

    @classmethod
    def execution_failed(cls, reason: str, source: str = "") -> MissionEvent:
        return cls(
            type_=MissionEventType.EXECUTION_FAILED,
            data=reason,
            source=source,
        )

    @classmethod
    def reset(cls) -> MissionEvent:
        return cls(type_=MissionEventType.RESET)
