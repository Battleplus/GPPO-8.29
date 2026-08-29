"""Atomic execution runtime for continuous progress and event preemption."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

from .controller import PreemptionController
from .models import (
    CommandStatus,
    CommunicationState,
    DecisionType,
    EventDecision,
    ExecutionCommand,
    ProgressPolicy,
    ResumePolicy,
    RuntimeEvent,
    RuntimeEventType,
    TaskRuntime,
    TaskState,
    TERMINAL_TASK_STATES,
    UAVAvailability,
    UAVRuntime,
)


class RuntimeInvariantError(RuntimeError):
    """Raised when a transaction would violate exclusive ownership."""


class StaleExecutionCommand(RuntimeError):
    """Raised when a command or ACK no longer matches the live graph/fence."""


@dataclass(frozen=True)
class EventBatchResult:
    graph_version_before: int
    graph_version_after: int
    decisions: tuple[EventDecision, ...]
    revoked_commands: tuple[str, ...]


class ExecutionRuntime:
    """Own task/UAV state and commit each confirmed-event batch atomically."""

    def __init__(self, *, progress_policy: ProgressPolicy | None = None) -> None:
        self.progress_policy = progress_policy or ProgressPolicy()
        self.tasks: dict[str, TaskRuntime] = {}
        self.uavs: dict[str, UAVRuntime] = {}
        self.commands: dict[str, ExecutionCommand] = {}
        self.graph_version = 0
        self._next_fencing_token = 0
        self._latest_fence_by_task: dict[str, int] = {}
        self._processed_events: dict[str, RuntimeEvent] = {}
        self.decision_log: list[EventDecision] = []

    def add_task(self, task: TaskRuntime) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"duplicate task_id {task.task_id}")
        self.tasks[task.task_id] = copy.deepcopy(task)
        self.validate_invariants()

    def add_uav(self, uav: UAVRuntime) -> None:
        if uav.uav_id in self.uavs:
            raise ValueError(f"duplicate uav_id {uav.uav_id}")
        self.uavs[uav.uav_id] = copy.deepcopy(uav)
        self.validate_invariants()

    def _task(self, task_id: str) -> TaskRuntime:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown task {task_id}") from exc

    def _uav(self, uav_id: str) -> UAVRuntime:
        try:
            return self.uavs[uav_id]
        except KeyError as exc:
            raise ValueError(f"unknown UAV {uav_id}") from exc

    def assign_task(self, task_id: str, uav_id: str, *, at: float, start: bool = True) -> None:
        task = self._task(task_id)
        uav = self._uav(uav_id)
        if task.state in TERMINAL_TASK_STATES:
            raise RuntimeInvariantError("terminal task cannot be assigned")
        if task.assigned_uav not in {None, uav_id}:
            raise RuntimeInvariantError("task already has another UAV")
        if uav.active_task_id not in {None, task_id}:
            raise RuntimeInvariantError("UAV already owns another active task")
        if uav.availability not in {UAVAvailability.AVAILABLE, UAVAvailability.PAUSED}:
            raise RuntimeInvariantError(f"UAV is not assignable: {uav.availability.value}")
        if uav.communication_state is not CommunicationState.CONNECTED:
            raise RuntimeInvariantError("communication-lost UAV cannot receive a task")
        if not uav.energy_safe_for_new_task:
            raise RuntimeInvariantError("UAV has insufficient energy margin")
        if not uav.supports(task.task_type):
            raise RuntimeInvariantError("UAV is incompatible with task type")
        task.assigned_uav = uav_id
        task.last_assigned_uav = uav_id
        task.state = TaskState.RUNNING if start else TaskState.ASSIGNED
        if start and task.started_at is None:
            task.started_at = at
        task.record(at, "assigned and started" if start else "assigned")
        uav.active_task_id = task_id
        uav.availability = UAVAvailability.BUSY
        self.validate_invariants()

    def start_task(self, task_id: str, *, at: float) -> None:
        task = self._task(task_id)
        if task.state not in {TaskState.ASSIGNED, TaskState.RESUMING}:
            raise RuntimeInvariantError(f"cannot start task from {task.state.value}")
        if task.assigned_uav is None:
            raise RuntimeInvariantError("assigned task has no UAV")
        task.state = TaskState.RUNNING
        if task.started_at is None:
            task.started_at = at
        task.record(at, "execution started")
        self._uav(task.assigned_uav).availability = UAVAvailability.BUSY

    def advance(
        self,
        delta_time: float,
        *,
        now: float,
        work_rate_by_task: dict[str, float] | None = None,
        energy_cost_by_uav: dict[str, float] | None = None,
    ) -> tuple[str, ...]:
        if delta_time < 0:
            raise ValueError("delta_time must be non-negative")
        work_rates = work_rate_by_task or {}
        energy_costs = energy_cost_by_uav or {}
        completed: list[str] = []
        for task in self.tasks.values():
            if task.state is not TaskState.RUNNING:
                continue
            rate = float(work_rates.get(task.task_id, 0.0))
            if rate < 0:
                raise ValueError("effective work rate must be non-negative")
            new_progress = min(1.0, task.progress + rate * delta_time)
            if new_progress != task.progress:
                task.set_progress(new_progress, at=now, reason="execution tick")
            if task.progress >= 1.0:
                uav_id = task.assigned_uav
                task.assigned_uav = None
                task.state = TaskState.COMPLETED
                task.record(now, "task completed")
                if uav_id is not None:
                    uav = self._uav(uav_id)
                    uav.active_task_id = None
                    uav.availability = UAVAvailability.AVAILABLE
                completed.append(task.task_id)
        for uav_id, energy_cost in energy_costs.items():
            if energy_cost < 0:
                raise ValueError("energy cost must be non-negative")
            uav = self._uav(uav_id)
            uav.energy_ratio = max(0.0, uav.energy_ratio - float(energy_cost))
        self.validate_invariants()
        return tuple(sorted(completed))

    def pause_task(self, task_id: str, *, at: float, reason: str) -> None:
        task = self._task(task_id)
        if task.state not in {TaskState.RUNNING, TaskState.ASSIGNED, TaskState.RESUMING}:
            raise RuntimeInvariantError(f"cannot pause task from {task.state.value}")
        task.state = TaskState.PAUSED
        task.interruption_count += 1
        task.record(at, reason)
        if task.assigned_uav is not None:
            self._uav(task.assigned_uav).availability = UAVAvailability.PAUSED

    def resume_task(self, task_id: str, uav_id: str, *, at: float) -> None:
        task = self._task(task_id)
        if task.resume_policy is ResumePolicy.NEVER:
            raise RuntimeInvariantError("task resume policy forbids resumption")
        if task.state is TaskState.PAUSED:
            if task.assigned_uav != uav_id:
                raise RuntimeInvariantError("paused task must first migrate before changing UAV")
            uav = self._uav(uav_id)
            if uav.communication_state is not CommunicationState.CONNECTED:
                raise RuntimeInvariantError("cannot resume while communication is unavailable")
            if not uav.energy_safe_for_new_task:
                raise RuntimeInvariantError("cannot resume without safe energy margin")
            task.state = TaskState.RESUMING
            task.record(at, "same-UAV resume initiated")
            task.state = TaskState.RUNNING
            task.record(at, "same-UAV execution resumed")
            uav.availability = UAVAvailability.BUSY
            self.validate_invariants()
            return
        if task.state is not TaskState.PREEMPTED:
            raise RuntimeInvariantError(f"cannot resume task from {task.state.value}")
        if task.resume_policy is ResumePolicy.SAME_UAV and task.last_assigned_uav != uav_id:
            raise RuntimeInvariantError("task must resume on its previous UAV")
        if task.last_assigned_uav is not None and task.last_assigned_uav != uav_id:
            task.state = TaskState.MIGRATING
            retained = task.progress * self.progress_policy.migration_retention
            if task.resume_policy is ResumePolicy.RESTART:
                retained = 0.0
            task.set_progress(retained, at=at, reason="resume migration retention applied")
        task.state = TaskState.RESUMING
        task.record(at, "resume initiated")
        self.assign_task(task_id, uav_id, at=at, start=True)

    def preempt_task(self, task_id: str, *, at: float, reason: str, safety_override: bool = False) -> str | None:
        task = self._task(task_id)
        if not task.preemptible and not safety_override:
            raise RuntimeInvariantError("task is not preemptible")
        old_uav = task.assigned_uav
        task.assigned_uav = None
        task.state = TaskState.PREEMPTED
        task.interruption_count += 1
        task.record(at, reason)
        if old_uav is not None:
            uav = self._uav(old_uav)
            uav.active_task_id = None
            if uav.availability not in {
                UAVAvailability.RETURNING,
                UAVAvailability.FAILED,
                UAVAvailability.COMMUNICATION_LOST,
            }:
                uav.availability = UAVAvailability.AVAILABLE
        return old_uav

    def migrate_task(self, task_id: str, new_uav_id: str, *, at: float, reason: str) -> None:
        task = self._task(task_id)
        old_uav = task.assigned_uav
        if old_uav == new_uav_id:
            raise RuntimeInvariantError("migration requires a different UAV")
        if task.resume_policy in {ResumePolicy.NEVER, ResumePolicy.SAME_UAV}:
            raise RuntimeInvariantError("task resume policy forbids migration")
        self.preempt_task(task_id, at=at, reason=f"migration initiated: {reason}")
        task.state = TaskState.MIGRATING
        retained = task.progress * self.progress_policy.migration_retention
        if task.resume_policy is ResumePolicy.RESTART:
            retained = 0.0
        task.set_progress(retained, at=at, reason="migration retention applied")
        task.state = TaskState.RESUMING
        task.record(at, "migration target selected")
        self.assign_task(task_id, new_uav_id, at=at, start=True)
        if old_uav is not None and old_uav in self.uavs:
            old = self.uavs[old_uav]
            if old.communication_state is CommunicationState.LOST:
                old.availability = UAVAvailability.COMMUNICATION_LOST

    def _detach_terminal(self, task: TaskRuntime, *, state: TaskState, at: float, reason: str) -> None:
        old_uav = task.assigned_uav
        task.assigned_uav = None
        task.state = state
        task.record(at, reason)
        if old_uav is not None:
            uav = self._uav(old_uav)
            uav.active_task_id = None
            if uav.availability not in {UAVAvailability.RETURNING, UAVAvailability.FAILED}:
                uav.availability = UAVAvailability.AVAILABLE

    def _revoke_affected_commands(self, event: RuntimeEvent) -> list[str]:
        revoked: list[str] = []
        for command in self.commands.values():
            if command.status is not CommandStatus.PROPOSED:
                continue
            if command.task_id == event.task_id or command.uav_id == event.uav_id:
                command.status = CommandStatus.REVOKED
                command.reason = f"revoked by {event.event_id}"
                revoked.append(command.command_id)
        return revoked

    def _apply_decision(self, event: RuntimeEvent, decision: EventDecision, *, at: float) -> None:
        if event.event_type is RuntimeEventType.TASK_PRIORITY_CHANGED:
            task = self._task(event.task_id or "")
            new_priority = int(event.payload.get("new_priority", event.task_priority))
            if new_priority < 0:
                raise ValueError("new_priority must be non-negative")
            task.priority = new_priority
            task.record(at, f"priority changed to {new_priority}")
        elif event.event_type is RuntimeEventType.TASK_DEADLINE_CHANGED:
            task = self._task(event.task_id or "")
            raw = event.payload.get("new_deadline", event.deadline)
            task.deadline = None if raw is None else float(raw)
            task.record(at, f"deadline changed to {task.deadline}")

        if decision.decision is DecisionType.CONTINUE:
            if event.event_type is RuntimeEventType.TASK_ARRIVAL and decision.selected_uav is not None:
                self.assign_task(event.task_id or "", decision.selected_uav, at=at, start=True)
            elif event.event_type is RuntimeEventType.UAV_COMM_RECOVERED:
                uav = self._uav(event.uav_id or "")
                uav.communication_state = CommunicationState.CONNECTED
                uav.last_seen_at = at
                uav.availability = (
                    UAVAvailability.BUSY if uav.active_task_id is not None
                    else UAVAvailability.AVAILABLE
                )
            return

        if decision.decision is DecisionType.QUEUE:
            if event.task_id is not None:
                task = self._task(event.task_id)
                if task.state not in TERMINAL_TASK_STATES and task.assigned_uav is None:
                    task.state = TaskState.PENDING
                    task.record(at, decision.reason)
            return

        if decision.decision is DecisionType.PREEMPT:
            if decision.displaced_task_id is None or decision.selected_uav is None:
                raise RuntimeInvariantError("preemption decision is incomplete")
            released = self.preempt_task(
                decision.displaced_task_id,
                at=at,
                reason=f"preempted by {event.event_id}",
            )
            if released != decision.selected_uav:
                raise RuntimeInvariantError("selected UAV is not the displaced task owner")
            self.assign_task(event.task_id or "", decision.selected_uav, at=at, start=True)
            return

        if decision.decision is DecisionType.PAUSE:
            if event.event_type is RuntimeEventType.UAV_COMM_LOST:
                uav = self._uav(event.uav_id or "")
                if decision.displaced_task_id is not None:
                    self.pause_task(decision.displaced_task_id, at=at, reason=decision.reason)
                uav.communication_state = CommunicationState.LOST
                uav.availability = UAVAvailability.COMMUNICATION_LOST
            elif decision.displaced_task_id is not None:
                self.pause_task(decision.displaced_task_id, at=at, reason=decision.reason)
            return

        if decision.decision is DecisionType.MIGRATE:
            if decision.displaced_task_id is None or decision.selected_uav is None:
                raise RuntimeInvariantError("migration decision is incomplete")
            if event.event_type is RuntimeEventType.UAV_COMM_LOST:
                failed_uav = self._uav(event.uav_id or "")
                failed_uav.communication_state = CommunicationState.LOST
                failed_uav.availability = UAVAvailability.COMMUNICATION_LOST
            self.migrate_task(
                decision.displaced_task_id,
                decision.selected_uav,
                at=at,
                reason=decision.reason,
            )
            if event.event_type is RuntimeEventType.EXECUTION_FAILURE and event.uav_id is not None:
                failed_uav = self._uav(event.uav_id)
                failed_uav.active_task_id = None
                failed_uav.availability = UAVAvailability.FAILED
            return

        if decision.decision is DecisionType.ABORT:
            task = self._task(decision.displaced_task_id or event.task_id or "")
            terminal = (
                TaskState.CANCELLED
                if event.event_type is RuntimeEventType.TASK_CANCELLED
                else TaskState.FAILED
            )
            self._detach_terminal(task, state=terminal, at=at, reason=decision.reason)
            return

        if decision.decision is DecisionType.RTB:
            uav = self._uav(event.uav_id or decision.selected_uav or "")
            if decision.displaced_task_id is not None:
                self.preempt_task(
                    decision.displaced_task_id,
                    at=at,
                    reason=decision.reason,
                    safety_override=True,
                )
            uav.active_task_id = None
            uav.availability = UAVAvailability.RETURNING
            return

        raise ValueError(f"unsupported decision {decision.decision.value}")

    def process_event_batch(
        self,
        events: Iterable[RuntimeEvent],
        controller: PreemptionController,
        *,
        now: float,
    ) -> EventBatchResult:
        ordered_input = tuple(sorted(events, key=RuntimeEvent.ordering_key))
        unique: dict[str, RuntimeEvent] = {}
        for event in ordered_input:
            existing = unique.get(event.event_id) or self._processed_events.get(event.event_id)
            if existing is not None:
                if existing != event:
                    raise ValueError(f"event_id {event.event_id} was reused with different content")
                continue
            unique[event.event_id] = event
        ordered = tuple(unique.values())
        if not ordered:
            return EventBatchResult(self.graph_version, self.graph_version, (), ())
        if any(event.received_at > now for event in ordered):
            raise ValueError("cannot process an event before it is received")

        staged = copy.deepcopy(self)
        before = staged.graph_version
        staged.graph_version += 1
        decisions: list[EventDecision] = []
        revoked: list[str] = []
        for event in ordered:
            revoked.extend(staged._revoke_affected_commands(event))
            decision = controller.decide(
                event,
                staged.tasks,
                staged.uavs,
                graph_version=staged.graph_version,
            )
            staged._apply_decision(event, decision, at=now)
            staged.validate_invariants()
            staged._processed_events[event.event_id] = event
            staged.decision_log.append(decision)
            decisions.append(decision)

        self.__dict__.clear()
        self.__dict__.update(staged.__dict__)
        return EventBatchResult(
            graph_version_before=before,
            graph_version_after=self.graph_version,
            decisions=tuple(decisions),
            revoked_commands=tuple(sorted(set(revoked))),
        )

    def issue_assignment_command(
        self,
        command_id: str,
        task_id: str,
        uav_id: str,
        *,
        expected_graph_version: int,
        at: float,
    ) -> ExecutionCommand:
        if command_id in self.commands:
            raise ValueError(f"duplicate command_id {command_id}")
        if expected_graph_version != self.graph_version:
            raise StaleExecutionCommand("command graph_version does not match runtime")
        task = self._task(task_id)
        uav = self._uav(uav_id)
        if task.state in TERMINAL_TASK_STATES:
            raise RuntimeInvariantError("cannot command a terminal task")
        if uav.active_task_id not in {None, task_id}:
            raise RuntimeInvariantError("command would double-book UAV")
        for command in self.commands.values():
            if command.status is CommandStatus.PROPOSED and (
                command.task_id == task_id or command.uav_id == uav_id
            ):
                raise RuntimeInvariantError("task or UAV already has a live command")
        self._next_fencing_token += 1
        token = self._next_fencing_token
        self._latest_fence_by_task[task_id] = token
        command = ExecutionCommand(
            command_id=command_id,
            task_id=task_id,
            uav_id=uav_id,
            graph_version=self.graph_version,
            fencing_token=token,
            issued_at=at,
        )
        self.commands[command_id] = command
        return command

    def acknowledge_command(
        self,
        command_id: str,
        *,
        graph_version: int,
        fencing_token: int,
        at: float,
    ) -> ExecutionCommand:
        try:
            command = self.commands[command_id]
        except KeyError as exc:
            raise ValueError(f"unknown command {command_id}") from exc
        if command.status is not CommandStatus.PROPOSED:
            raise StaleExecutionCommand("inactive command cannot be acknowledged")
        if graph_version != command.graph_version or graph_version != self.graph_version:
            command.status = CommandStatus.REJECTED
            command.reason = "stale graph version"
            raise StaleExecutionCommand(command.reason)
        if (
            fencing_token != command.fencing_token
            or self._latest_fence_by_task.get(command.task_id) != fencing_token
        ):
            command.status = CommandStatus.REJECTED
            command.reason = "stale fencing token"
            raise StaleExecutionCommand(command.reason)
        self.assign_task(command.task_id, command.uav_id, at=at, start=True)
        command.status = CommandStatus.ACKED
        return command

    def validate_invariants(self) -> None:
        owners: dict[str, str] = {}
        for uav in self.uavs.values():
            if uav.active_task_id is None:
                if uav.availability in {UAVAvailability.BUSY, UAVAvailability.PAUSED}:
                    raise RuntimeInvariantError("busy/paused UAV has no active task")
                continue
            if uav.availability not in {
                UAVAvailability.BUSY,
                UAVAvailability.PAUSED,
                UAVAvailability.COMMUNICATION_LOST,
            }:
                raise RuntimeInvariantError("active UAV has incompatible availability")
            if uav.active_task_id in owners:
                raise RuntimeInvariantError("task has more than one active UAV")
            owners[uav.active_task_id] = uav.uav_id
            task = self.tasks.get(uav.active_task_id)
            if task is None or task.assigned_uav != uav.uav_id:
                raise RuntimeInvariantError("UAV/task ownership is not bidirectional")
        seen_uavs: set[str] = set()
        for task in self.tasks.values():
            if task.assigned_uav is None:
                if task.state in {TaskState.ASSIGNED, TaskState.RUNNING, TaskState.RESUMING}:
                    raise RuntimeInvariantError("active task has no assigned UAV")
                continue
            if task.state in TERMINAL_TASK_STATES:
                raise RuntimeInvariantError("terminal task retains an assigned UAV")
            if task.assigned_uav in seen_uavs:
                raise RuntimeInvariantError("UAV is assigned to more than one task")
            seen_uavs.add(task.assigned_uav)
            uav = self.uavs.get(task.assigned_uav)
            if uav is None or uav.active_task_id != task.task_id:
                raise RuntimeInvariantError("task/UAV ownership is not bidirectional")
