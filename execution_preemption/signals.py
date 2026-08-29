"""Derive frozen reward signals from two ExecutionRuntime snapshots."""

from __future__ import annotations

import math

from .models import TaskState, TERMINAL_TASK_STATES
from .reward import TransitionSignals
from .runtime import ExecutionRuntime


PRIORITY_SCALE = 100.0
DECISION_HORIZON = 200.0
STARVATION_HORIZON = 100.0
SWITCH_TIME_HORIZON = 5.0
URGENT_PRIORITY_MIN = 80


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("reward signal source must be finite")
    return min(1.0, max(0.0, float(value)))


def _priority_weight(priority: int) -> float:
    return _clip01(float(priority) / PRIORITY_SCALE)


def derive_transition_signals(
    before: ExecutionRuntime,
    after: ExecutionRuntime,
    *,
    now: float,
    delta_time: float = 1.0,
    normalized_distance: float = 0.0,
    stale_command_resurrections: int = 0,
) -> TransitionSignals:
    """Compute algorithm-independent normalized signals after an atomic commit."""

    if not isinstance(before, ExecutionRuntime) or not isinstance(after, ExecutionRuntime):
        raise TypeError("before and after must be ExecutionRuntime")
    if not math.isfinite(now) or now < 0.0:
        raise ValueError("now must be finite and non-negative")
    if not math.isfinite(delta_time) or delta_time < 0.0:
        raise ValueError("delta_time must be finite and non-negative")
    if isinstance(stale_command_resurrections, bool) or stale_command_resurrections < 0:
        raise ValueError("stale_command_resurrections must be non-negative")
    before.validate_invariants()
    after.validate_invariants()

    task_ids = sorted(set(before.tasks) | set(after.tasks))
    total_weight = max(
        1.0,
        math.fsum(
            _priority_weight(
                after.tasks.get(task_id, before.tasks[task_id]).priority
            )
            for task_id in task_ids
        ),
    )
    weighted_progress_gain = 0.0
    progress_loss = 0.0
    interruption_delta = 0
    vacancy = 0.0
    starvation = 0.0
    urgent_count = 0
    urgent_miss = 0
    for task_id in task_ids:
        old = before.tasks.get(task_id)
        new = after.tasks.get(task_id)
        if new is None:
            continue
        old_progress = 0.0 if old is None else old.progress
        weight = _priority_weight(new.priority)
        weighted_progress_gain += weight * max(0.0, new.progress - old_progress)
        progress_loss += max(0.0, old_progress - new.progress)
        old_interruptions = 0 if old is None else old.interruption_count
        interruption_delta += max(0, new.interruption_count - old_interruptions)
        if new.state not in TERMINAL_TASK_STATES and new.state is not TaskState.RUNNING:
            vacancy += weight * new.remaining_work
            age = max(0.0, now - new.last_updated_at)
            starvation += weight * _clip01(age / STARVATION_HORIZON)
        if new.priority >= URGENT_PRIORITY_MIN and new.deadline is not None:
            urgent_count += 1
            if now > new.deadline and new.state is not TaskState.COMPLETED:
                urgent_miss += 1

    uav_ids = sorted(set(before.uavs) | set(after.uavs))
    energy_consumed = math.fsum(
        max(
            0.0,
            (before.uavs[uav_id].energy_ratio if uav_id in before.uavs else 0.0)
            - (after.uavs[uav_id].energy_ratio if uav_id in after.uavs else 0.0),
        )
        for uav_id in uav_ids
    ) / max(1, len(uav_ids))
    loads = [int(uav.active_task_id is not None) for uav in after.uavs.values()]
    load_gap = float(max(loads) - min(loads)) if len(loads) > 1 else 0.0

    resource_conflicts = 0
    active_task_owners: dict[str, int] = {}
    for uav in after.uavs.values():
        if uav.active_task_id is not None:
            active_task_owners[uav.active_task_id] = active_task_owners.get(uav.active_task_id, 0) + 1
    resource_conflicts += sum(max(0, count - 1) for count in active_task_owners.values())
    energy_violations = sum(
        1
        for uav in after.uavs.values()
        if uav.active_task_id is not None and not uav.energy_safe_for_new_task
    )

    return TransitionSignals(
        weighted_progress_gain=_clip01(weighted_progress_gain / total_weight),
        urgent_deadline_miss_rate=(urgent_miss / urgent_count if urgent_count else 0.0),
        weighted_vacancy_time=_clip01(
            vacancy / total_weight * delta_time / DECISION_HORIZON
        ),
        progress_loss=_clip01(progress_loss / max(1, len(task_ids))),
        starvation_exposure=_clip01(starvation / total_weight),
        switch_time=_clip01(
            interruption_delta * after.progress_policy.switch_time_cost
            / SWITCH_TIME_HORIZON / max(1, len(task_ids))
        ),
        energy_consumed=_clip01(energy_consumed),
        normalized_distance=_clip01(normalized_distance),
        load_gap=_clip01(load_gap),
        resource_conflicts=resource_conflicts,
        stale_command_resurrections=int(stale_command_resurrections),
        energy_safety_violations=energy_violations,
    )


__all__ = [
    "DECISION_HORIZON",
    "PRIORITY_SCALE",
    "STARVATION_HORIZON",
    "SWITCH_TIME_HORIZON",
    "URGENT_PRIORITY_MIN",
    "derive_transition_signals",
]
