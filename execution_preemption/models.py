"""Domain models for the execution-preemption V1 development contract.

The module is deliberately independent from PPO/GPPO.  It freezes execution
semantics that both policies can consume later, without allowing a neural
policy to decide safety-critical interruption or return-to-base behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import math
from typing import Any, Mapping


class TaskState(str, Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    PREEMPTED = "PREEMPTED"
    MIGRATING = "MIGRATING"
    RESUMING = "RESUMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ResumePolicy(str, Enum):
    SAME_UAV = "SAME_UAV"
    ANY_COMPATIBLE = "ANY_COMPATIBLE"
    RESTART = "RESTART"
    NEVER = "NEVER"


class UAVAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    PAUSED = "PAUSED"
    RETURNING = "RETURNING"
    FAILED = "FAILED"
    COMMUNICATION_LOST = "COMMUNICATION_LOST"


class CommunicationState(str, Enum):
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    LOST = "LOST"


class EventPriority(IntEnum):
    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4


class RuntimeEventType(str, Enum):
    TASK_ARRIVAL = "TASK_ARRIVAL"
    TASK_CANCELLED = "TASK_CANCELLED"
    TASK_PRIORITY_CHANGED = "TASK_PRIORITY_CHANGED"
    TASK_DEADLINE_CHANGED = "TASK_DEADLINE_CHANGED"
    UAV_LOW_ENERGY = "UAV_LOW_ENERGY"
    UAV_COMM_LOST = "UAV_COMM_LOST"
    UAV_COMM_RECOVERED = "UAV_COMM_RECOVERED"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"


class DecisionType(str, Enum):
    CONTINUE = "CONTINUE"
    QUEUE = "QUEUE"
    PAUSE = "PAUSE"
    PREEMPT = "PREEMPT"
    MIGRATE = "MIGRATE"
    ABORT = "ABORT"
    RTB = "RTB"


class CommandStatus(str, Enum):
    PROPOSED = "PROPOSED"
    ACKED = "ACKED"
    REJECTED = "REJECTED"
    REVOKED = "REVOKED"
    COMPLETED = "COMPLETED"


TERMINAL_TASK_STATES = {
    TaskState.COMPLETED,
    TaskState.FAILED,
    TaskState.CANCELLED,
}


@dataclass(frozen=True)
class TaskProgressRecord:
    at: float
    progress: float
    remaining_work: float
    state: TaskState
    assigned_uav: str | None
    reason: str


@dataclass
class TaskRuntime:
    task_id: str
    task_type: str
    priority: int
    deadline: float | None
    state: TaskState = TaskState.PENDING
    progress: float = 0.0
    remaining_work: float = 1.0
    assigned_uav: str | None = None
    last_assigned_uav: str | None = None
    preemptible: bool = True
    resume_policy: ResumePolicy = ResumePolicy.ANY_COMPATIBLE
    interruption_count: int = 0
    started_at: float | None = None
    last_updated_at: float = 0.0
    exclusive: bool = True
    progress_history: list[TaskProgressRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.task_type:
            raise ValueError("task_type is required")
        if self.priority < 0:
            raise ValueError("task priority must be non-negative")
        if self.deadline is not None and not math.isfinite(self.deadline):
            raise ValueError("deadline must be finite")
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be in [0, 1]")
        if not math.isclose(self.remaining_work, 1.0 - self.progress, abs_tol=1e-9):
            raise ValueError("remaining_work must equal 1 - progress")
        if self.state in TERMINAL_TASK_STATES and self.assigned_uav is not None:
            raise ValueError("terminal task cannot retain an assigned UAV")
        self.record(self.last_updated_at, "created")

    def record(self, at: float, reason: str) -> None:
        if not math.isfinite(at) or at < self.last_updated_at:
            raise ValueError("task timestamps must be finite and monotonic")
        self.last_updated_at = float(at)
        self.progress_history.append(TaskProgressRecord(
            at=float(at),
            progress=float(self.progress),
            remaining_work=float(self.remaining_work),
            state=self.state,
            assigned_uav=self.assigned_uav,
            reason=str(reason),
        ))

    def set_progress(self, progress: float, *, at: float, reason: str) -> None:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be in [0, 1]")
        self.progress = float(progress)
        self.remaining_work = 1.0 - self.progress
        self.record(at, reason)


@dataclass
class UAVRuntime:
    uav_id: str
    energy_ratio: float = 1.0
    reserve_energy: float = 0.10
    estimated_rtb_energy: float = 0.10
    active_task_id: str | None = None
    availability: UAVAvailability = UAVAvailability.AVAILABLE
    communication_state: CommunicationState = CommunicationState.CONNECTED
    last_seen_at: float = 0.0
    supported_task_types: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.uav_id:
            raise ValueError("uav_id is required")
        for name, value in (
            ("energy_ratio", self.energy_ratio),
            ("reserve_energy", self.reserve_energy),
            ("estimated_rtb_energy", self.estimated_rtb_energy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.last_seen_at < 0 or not math.isfinite(self.last_seen_at):
            raise ValueError("last_seen_at must be finite and non-negative")

    @property
    def energy_safe_for_new_task(self) -> bool:
        return self.energy_ratio > self.reserve_energy + self.estimated_rtb_energy

    def supports(self, task_type: str) -> bool:
        return not self.supported_task_types or task_type in self.supported_task_types


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    event_type: RuntimeEventType
    priority: EventPriority
    occurred_at: float
    received_at: float
    task_id: str | None = None
    uav_id: str | None = None
    task_priority: int = 0
    deadline: float | None = None
    confidence: float = 1.0
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if self.occurred_at < 0 or self.received_at < self.occurred_at:
            raise ValueError("event timestamps must satisfy 0 <= occurred_at <= received_at")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.deadline is not None and not math.isfinite(self.deadline):
            raise ValueError("deadline must be finite")

    @property
    def information_age(self) -> float:
        return self.received_at - self.occurred_at

    def ordering_key(self) -> tuple[int, float, int, float, float, str]:
        return (
            int(self.priority),
            self.deadline if self.deadline is not None else float("inf"),
            -int(self.task_priority),
            self.information_age,
            self.received_at,
            self.event_id,
        )


@dataclass(frozen=True)
class EventDecision:
    event_id: str
    priority: EventPriority
    information_age: float
    confidence: float
    decision: DecisionType
    displaced_task_id: str | None
    selected_uav: str | None
    reason: str
    graph_version: int
    allocation_request_id: str | None = None
    allocator_id: str | None = None


@dataclass
class ExecutionCommand:
    command_id: str
    task_id: str
    uav_id: str
    graph_version: int
    fencing_token: int
    issued_at: float
    status: CommandStatus = CommandStatus.PROPOSED
    reason: str = ""


@dataclass(frozen=True)
class ProgressPolicy:
    total_work: float = 1.0
    same_uav_retention: float = 1.0
    migration_retention: float = 0.90
    switch_time_cost: float = 0.25

    def __post_init__(self) -> None:
        if self.total_work != 1.0:
            raise ValueError("V1 normalizes total_work to 1.0")
        if not 0.8 <= self.migration_retention <= 1.0:
            raise ValueError("migration_retention must be in [0.8, 1.0]")
        if not 0.0 <= self.same_uav_retention <= 1.0:
            raise ValueError("same_uav_retention must be in [0, 1]")
        if self.switch_time_cost < 0:
            raise ValueError("switch_time_cost must be non-negative")
