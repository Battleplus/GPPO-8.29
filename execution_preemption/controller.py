"""Deterministic safety and preemption arbiter for contract V1."""

from __future__ import annotations

from collections.abc import Mapping

from .models import (
    CommunicationState,
    DecisionType,
    EventDecision,
    EventPriority,
    RuntimeEvent,
    RuntimeEventType,
    TaskRuntime,
    TaskState,
    UAVAvailability,
    UAVRuntime,
)


class PreemptionController:
    """Choose safety actions without consulting PPO/GPPO.

    PPO/GPPO will later select a UAV--Task allocation after this controller has
    made the safety-critical continue/preempt/migrate/RTB decision.
    """

    @staticmethod
    def _available_uavs(
        task: TaskRuntime,
        uavs: Mapping[str, UAVRuntime],
        *,
        exclude: set[str] | None = None,
    ) -> list[UAVRuntime]:
        excluded = exclude or set()
        return sorted(
            (
                uav for uav in uavs.values()
                if uav.uav_id not in excluded
                and uav.availability is UAVAvailability.AVAILABLE
                and uav.communication_state is CommunicationState.CONNECTED
                and uav.energy_safe_for_new_task
                and uav.supports(task.task_type)
            ),
            key=lambda item: (-item.energy_ratio, item.uav_id),
        )

    @staticmethod
    def _active_task_for_uav(
        uav_id: str | None,
        tasks: Mapping[str, TaskRuntime],
        uavs: Mapping[str, UAVRuntime],
    ) -> TaskRuntime | None:
        if uav_id is None or uav_id not in uavs:
            return None
        task_id = uavs[uav_id].active_task_id
        return tasks.get(task_id) if task_id is not None else None

    def decide(
        self,
        event: RuntimeEvent,
        tasks: Mapping[str, TaskRuntime],
        uavs: Mapping[str, UAVRuntime],
        *,
        graph_version: int,
    ) -> EventDecision:
        task = tasks.get(event.task_id) if event.task_id is not None else None
        active = self._active_task_for_uav(event.uav_id, tasks, uavs)
        decision = DecisionType.CONTINUE
        displaced: str | None = None
        selected: str | None = None
        reason = "event does not require interruption"

        if event.event_type is RuntimeEventType.TASK_ARRIVAL:
            if task is None:
                raise ValueError(f"arrival references unknown task {event.task_id!r}")
            idle = self._available_uavs(task, uavs)
            if idle:
                selected = idle[0].uav_id
                reason = "dispatch new task on an idle compatible UAV"
            else:
                candidates = sorted(
                    (
                        current for current in tasks.values()
                        if current.state is TaskState.RUNNING
                        and current.preemptible
                        and current.assigned_uav is not None
                        and current.priority < task.priority
                    ),
                    key=lambda item: (item.priority, item.progress, item.task_id),
                )
                if event.priority <= EventPriority.P1 and candidates:
                    victim = candidates[0]
                    decision = DecisionType.PREEMPT
                    displaced = victim.task_id
                    selected = victim.assigned_uav
                    reason = "urgent task preempts the lowest-priority eligible task"
                else:
                    decision = DecisionType.QUEUE
                    reason = "no idle UAV and no policy-permitted preemption"

        elif event.event_type is RuntimeEventType.TASK_CANCELLED:
            if task is None:
                raise ValueError(f"cancellation references unknown task {event.task_id!r}")
            decision = DecisionType.ABORT
            displaced = task.task_id
            selected = task.assigned_uav
            reason = "cancelled task must stop and release its resource"

        elif event.event_type in {
            RuntimeEventType.TASK_PRIORITY_CHANGED,
            RuntimeEventType.TASK_DEADLINE_CHANGED,
        }:
            if task is None:
                raise ValueError(f"update references unknown task {event.task_id!r}")
            if task.state in {TaskState.PENDING, TaskState.PREEMPTED, TaskState.PAUSED}:
                decision = DecisionType.QUEUE
                reason = "metadata updated; task remains queued for arbitration"
            else:
                reason = "metadata updated without interrupting a running task"

        elif event.event_type is RuntimeEventType.UAV_LOW_ENERGY:
            if event.uav_id is None or event.uav_id not in uavs:
                raise ValueError(f"low-energy event references unknown UAV {event.uav_id!r}")
            decision = DecisionType.RTB
            displaced = active.task_id if active is not None else None
            selected = event.uav_id
            reason = "energy safety has priority over task continuation"

        elif event.event_type is RuntimeEventType.UAV_COMM_LOST:
            if event.uav_id is None or event.uav_id not in uavs:
                raise ValueError(f"communication event references unknown UAV {event.uav_id!r}")
            if active is None:
                decision = DecisionType.PAUSE
                selected = event.uav_id
                reason = "isolate communication-lost UAV from new assignments"
            else:
                replacements = self._available_uavs(active, uavs, exclude={event.uav_id})
                displaced = active.task_id
                if active.preemptible and replacements:
                    decision = DecisionType.MIGRATE
                    selected = replacements[0].uav_id
                    reason = "migrate preemptible task away from communication-lost UAV"
                else:
                    decision = DecisionType.PAUSE
                    selected = event.uav_id
                    reason = "pause task because no safe migration target exists"

        elif event.event_type is RuntimeEventType.UAV_COMM_RECOVERED:
            if event.uav_id is None or event.uav_id not in uavs:
                raise ValueError(f"recovery event references unknown UAV {event.uav_id!r}")
            selected = event.uav_id
            reason = "restore UAV eligibility; task resumption remains explicit"

        elif event.event_type is RuntimeEventType.EXECUTION_FAILURE:
            failed_task = task or active
            if failed_task is None:
                raise ValueError("execution failure must reference a task or active UAV")
            displaced = failed_task.task_id
            replacements = self._available_uavs(
                failed_task,
                uavs,
                exclude={failed_task.assigned_uav} if failed_task.assigned_uav else set(),
            )
            if failed_task.preemptible and replacements:
                decision = DecisionType.MIGRATE
                selected = replacements[0].uav_id
                reason = "execution failure triggers migration to a safe compatible UAV"
            else:
                decision = DecisionType.ABORT
                selected = failed_task.assigned_uav
                reason = "execution failure cannot be recovered under the frozen policy"

        return EventDecision(
            event_id=event.event_id,
            priority=event.priority,
            information_age=event.information_age,
            confidence=event.confidence,
            decision=decision,
            displaced_task_id=displaced,
            selected_uav=selected,
            reason=reason,
            graph_version=graph_version,
        )
