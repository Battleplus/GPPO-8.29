"""Deterministic Dynamic-Preemption-Dev tape generation and replay."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
from typing import Any, Iterable, Mapping

from .controller import PreemptionController
from .models import (
    CommunicationState,
    EventDecision,
    EventPriority,
    ResumePolicy,
    RuntimeEvent,
    RuntimeEventType,
    TaskRuntime,
    TaskState,
    UAVAvailability,
    UAVRuntime,
)
from .runtime import ExecutionRuntime


SCHEMA_VERSION = 1
CONTRACT_ID = "execution-preemption-v1"
DEV_BANK = "Dynamic-Preemption-Dev"
CASES_PER_SCENARIO = 20

SCENARIO_CATALOG: tuple[dict[str, str], ...] = (
    {"id": "urgent_at_40", "description": "搜索执行 40% 时出现 P1 紧急任务"},
    {"id": "low_value_at_90", "description": "搜索执行 90% 时出现低收益临时任务"},
    {"id": "nonpreemptible_strike", "description": "不可抢占打击期间出现普通任务"},
    {"id": "execution_uav_destroyed", "description": "执行 UAV 中途失效并迁移任务"},
    {"id": "low_energy_rtb", "description": "执行 UAV 低能量并安全返航"},
    {"id": "delayed_task_cancel", "description": "旧任务取消报告延迟到达"},
    {"id": "repeated_priority_change", "description": "同一任务连续优先级变化"},
    {"id": "simultaneous_p1", "description": "两个 P1 事件原子同时到达"},
    {"id": "event_during_inference", "description": "重新分配窗口内再次到达安全事件"},
    {"id": "event_during_resume", "description": "通信恢复窗口内再次到达紧急事件"},
)

_SCENARIO_IDS = tuple(item["id"] for item in SCENARIO_CATALOG)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def tape_sha256(tape: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(tape)).hexdigest()


def _uav(uav_id: str, *, energy: float = 0.9) -> dict[str, Any]:
    return {
        "uav_id": uav_id,
        "energy_ratio": round(energy, 6),
        "reserve_energy": 0.1,
        "estimated_rtb_energy": 0.1,
        "communication_state": "CONNECTED",
        "supported_task_types": ["SEARCH", "STRIKE", "URGENT"],
    }


def _task(
    task_id: str,
    *,
    task_type: str = "SEARCH",
    priority: int = 10,
    deadline: float = 100.0,
    state: str = "PENDING",
    progress: float = 0.0,
    assigned_uav: str | None = None,
    preemptible: bool = True,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "priority": priority,
        "deadline": round(deadline, 6),
        "state": state,
        "progress": round(progress, 6),
        "assigned_uav": assigned_uav,
        "preemptible": preemptible,
        "resume_policy": "ANY_COMPATIBLE",
    }


def _event(
    event_id: str,
    event_type: str,
    priority: str,
    *,
    occurred_at: float,
    received_at: float | None = None,
    task_id: str | None = None,
    uav_id: str | None = None,
    task_priority: int = 0,
    deadline: float | None = None,
    payload: Mapping[str, Any] | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "priority": priority,
        "occurred_at": round(occurred_at, 6),
        "received_at": round(received_at if received_at is not None else occurred_at, 6),
        "task_id": task_id,
        "uav_id": uav_id,
        "task_priority": task_priority,
        "deadline": None if deadline is None else round(deadline, 6),
        "confidence": 1.0,
        "payload": dict(payload or {}),
        "batch_id": batch_id or event_id,
    }


def _busy_fleet(progress: float, *, base_preemptible: bool = True) -> tuple[list[dict], list[dict]]:
    uavs = [_uav(f"U{i}", energy=0.92 - i * 0.02) for i in range(4)]
    tasks = [
        _task(
            "BASE",
            priority=10,
            state="RUNNING",
            progress=progress,
            assigned_uav="U0",
            preemptible=base_preemptible,
        )
    ]
    for index in range(1, 4):
        tasks.append(_task(
            f"FILLER-{index}",
            priority=40 + index,
            state="RUNNING",
            progress=0.2,
            assigned_uav=f"U{index}",
        ))
    return uavs, tasks


def build_development_tape(scenario_id: str, case_index: int) -> dict[str, Any]:
    if scenario_id not in _SCENARIO_IDS:
        raise ValueError(f"unknown scenario {scenario_id}")
    if not 0 <= case_index < CASES_PER_SCENARIO:
        raise ValueError("case_index outside frozen development range")
    scenario_index = _SCENARIO_IDS.index(scenario_id)
    case_seed = 829_100_000 + scenario_index * 1_000 + case_index
    rng = random.Random(case_seed)
    t0 = round(5.0 + rng.uniform(0.0, 0.4), 6)
    events: list[dict[str, Any]] = []

    if scenario_id == "urgent_at_40":
        uavs, tasks = _busy_fleet(0.4)
        tasks.append(_task("URGENT", task_type="URGENT", priority=90, deadline=t0 + 3.0))
        events.append(_event(
            f"E-{case_index:02d}-urgent", "TASK_ARRIVAL", "P1",
            occurred_at=t0, task_id="URGENT", task_priority=90, deadline=t0 + 3.0,
        ))
    elif scenario_id == "low_value_at_90":
        uavs, tasks = _busy_fleet(0.9)
        tasks.append(_task("TEMP", priority=5, deadline=t0 + 20.0))
        events.append(_event(
            f"E-{case_index:02d}-temp", "TASK_ARRIVAL", "P3",
            occurred_at=t0, task_id="TEMP", task_priority=5, deadline=t0 + 20.0,
        ))
    elif scenario_id == "nonpreemptible_strike":
        uavs, tasks = _busy_fleet(0.5, base_preemptible=False)
        tasks[0].update({"task_type": "STRIKE", "priority": 80})
        tasks.append(_task("ORDINARY", priority=10, deadline=t0 + 20.0))
        events.append(_event(
            f"E-{case_index:02d}-ordinary", "TASK_ARRIVAL", "P3",
            occurred_at=t0, task_id="ORDINARY", task_priority=10,
        ))
    elif scenario_id == "execution_uav_destroyed":
        uavs = [_uav(f"U{i}") for i in range(4)]
        tasks = [_task("BASE", state="RUNNING", progress=0.45, assigned_uav="U0")]
        events.append(_event(
            f"E-{case_index:02d}-failure", "EXECUTION_FAILURE", "P0",
            occurred_at=t0, task_id="BASE", uav_id="U0",
        ))
    elif scenario_id == "low_energy_rtb":
        uavs = [_uav(f"U{i}") for i in range(4)]
        uavs[0]["energy_ratio"] = round(0.18 + rng.uniform(0.0, 0.01), 6)
        tasks = [_task("BASE", state="RUNNING", progress=0.55, assigned_uav="U0")]
        events.append(_event(
            f"E-{case_index:02d}-energy", "UAV_LOW_ENERGY", "P2",
            occurred_at=t0, uav_id="U0",
        ))
    elif scenario_id == "delayed_task_cancel":
        uavs = [_uav(f"U{i}") for i in range(4)]
        tasks = [_task("BASE", state="RUNNING", progress=0.65, assigned_uav="U0")]
        events.append(_event(
            f"E-{case_index:02d}-cancel", "TASK_CANCELLED", "P3",
            occurred_at=t0, received_at=t0 + 2.0 + rng.uniform(0.0, 0.5),
            task_id="BASE",
        ))
    elif scenario_id == "repeated_priority_change":
        uavs = [_uav(f"U{i}") for i in range(4)]
        tasks = [_task("BASE", state="RUNNING", progress=0.3, assigned_uav="U0")]
        events.extend([
            _event(
                f"E-{case_index:02d}-priority-a", "TASK_PRIORITY_CHANGED", "P3",
                occurred_at=t0, task_id="BASE", task_priority=20,
                payload={"new_priority": 20},
            ),
            _event(
                f"E-{case_index:02d}-priority-b", "TASK_PRIORITY_CHANGED", "P1",
                occurred_at=t0 + 0.2, task_id="BASE", task_priority=80,
                payload={"new_priority": 80},
            ),
        ])
    elif scenario_id == "simultaneous_p1":
        uavs = [_uav(f"U{i}") for i in range(4)]
        tasks = [
            _task("URGENT-A", task_type="URGENT", priority=80, deadline=t0 + 4.0),
            _task("URGENT-B", task_type="URGENT", priority=90, deadline=t0 + 2.0),
        ]
        batch = f"B-{case_index:02d}"
        events.extend([
            _event(
                f"E-{case_index:02d}-p1-a", "TASK_ARRIVAL", "P1",
                occurred_at=t0, task_id="URGENT-A", task_priority=80,
                deadline=t0 + 4.0, batch_id=batch,
            ),
            _event(
                f"E-{case_index:02d}-p1-b", "TASK_ARRIVAL", "P1",
                occurred_at=t0, task_id="URGENT-B", task_priority=90,
                deadline=t0 + 2.0, batch_id=batch,
            ),
        ])
    elif scenario_id == "event_during_inference":
        uavs = [_uav(f"U{i}") for i in range(4)]
        tasks = [_task("URGENT", task_type="URGENT", priority=90, deadline=t0 + 3.0)]
        events.extend([
            _event(
                f"E-{case_index:02d}-arrival", "TASK_ARRIVAL", "P1",
                occurred_at=t0, task_id="URGENT", task_priority=90,
                deadline=t0 + 3.0,
            ),
            _event(
                f"E-{case_index:02d}-energy", "UAV_LOW_ENERGY", "P2",
                occurred_at=t0 + 0.05, uav_id="U0",
            ),
        ])
    else:
        uavs, tasks = _busy_fleet(0.25)
        uavs[0]["communication_state"] = "LOST"
        tasks[0]["state"] = "PAUSED"
        tasks.append(_task("URGENT", task_type="URGENT", priority=95, deadline=t0 + 2.0))
        events.extend([
            _event(
                f"E-{case_index:02d}-recovered", "UAV_COMM_RECOVERED", "P2",
                occurred_at=t0, uav_id="U0",
            ),
            _event(
                f"E-{case_index:02d}-urgent", "TASK_ARRIVAL", "P1",
                occurred_at=t0 + 0.05, task_id="URGENT", task_priority=95,
                deadline=t0 + 2.0,
            ),
        ])

    tape = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "bank": DEV_BANK,
        "classification": "development_only_not_held_out",
        "paired": True,
        "scenario_id": scenario_id,
        "case_index": case_index,
        "case_seed": case_seed,
        "tape_id": f"{scenario_id}-{case_index:02d}-{case_seed}",
        "initial_state": {"uavs": uavs, "tasks": tasks},
        "events": events,
        "required_invariants": [
            "exclusive_task_owner_count<=1",
            "active_task_count_per_uav<=1",
            "stale_command_resurrection=0",
            "progress_double_count=0",
        ],
    }
    validate_tape(tape)
    return tape


def validate_tape(tape: Mapping[str, Any]) -> None:
    if tape.get("schema_version") != SCHEMA_VERSION or tape.get("contract_id") != CONTRACT_ID:
        raise ValueError("unsupported tape contract")
    if tape.get("scenario_id") not in _SCENARIO_IDS:
        raise ValueError("unknown scenario_id")
    uavs = list(tape["initial_state"]["uavs"])
    tasks = list(tape["initial_state"]["tasks"])
    uav_ids = [str(item["uav_id"]) for item in uavs]
    task_ids = [str(item["task_id"]) for item in tasks]
    if len(uav_ids) != len(set(uav_ids)) or len(task_ids) != len(set(task_ids)):
        raise ValueError("initial identifiers must be unique")
    assigned = [item.get("assigned_uav") for item in tasks if item.get("assigned_uav") is not None]
    if len(assigned) != len(set(assigned)) or any(item not in uav_ids for item in assigned):
        raise ValueError("initial UAV ownership must be exclusive and valid")
    event_ids: set[str] = set()
    for item in tape["events"]:
        if item["event_id"] in event_ids:
            raise ValueError("event ids must be unique")
        event_ids.add(item["event_id"])
        RuntimeEvent(
            event_id=item["event_id"],
            event_type=RuntimeEventType(item["event_type"]),
            priority=EventPriority[item["priority"]],
            occurred_at=float(item["occurred_at"]),
            received_at=float(item["received_at"]),
            task_id=item.get("task_id"),
            uav_id=item.get("uav_id"),
            task_priority=int(item.get("task_priority", 0)),
            deadline=item.get("deadline"),
            confidence=float(item.get("confidence", 1.0)),
            payload=item.get("payload", {}),
        )


def runtime_from_tape(tape: Mapping[str, Any]) -> ExecutionRuntime:
    validate_tape(tape)
    runtime = ExecutionRuntime()
    uav_specs = list(tape["initial_state"]["uavs"])
    for item in uav_specs:
        runtime.add_uav(UAVRuntime(
            uav_id=str(item["uav_id"]),
            # Initial assignments may already be running when a low-energy
            # event is observed.  Seed ownership first, then restore the tape's
            # live energy below.
            energy_ratio=1.0,
            reserve_energy=float(item["reserve_energy"]),
            estimated_rtb_energy=float(item["estimated_rtb_energy"]),
            communication_state=CommunicationState.CONNECTED,
            supported_task_types=frozenset(item.get("supported_task_types", ())),
        ))
    task_specs = list(tape["initial_state"]["tasks"])
    for item in task_specs:
        progress = float(item.get("progress", 0.0))
        runtime.add_task(TaskRuntime(
            task_id=str(item["task_id"]),
            task_type=str(item["task_type"]),
            priority=int(item["priority"]),
            deadline=float(item["deadline"]) if item.get("deadline") is not None else None,
            progress=progress,
            remaining_work=1.0 - progress,
            preemptible=bool(item.get("preemptible", True)),
            resume_policy=ResumePolicy(item.get("resume_policy", "ANY_COMPATIBLE")),
        ))
    for item in task_specs:
        assigned = item.get("assigned_uav")
        if assigned is None:
            continue
        runtime.assign_task(str(item["task_id"]), str(assigned), at=0.0, start=True)
        if item.get("state") == "PAUSED":
            runtime.pause_task(str(item["task_id"]), at=0.0, reason="tape initial pause")
    for item in uav_specs:
        uav = runtime.uavs[str(item["uav_id"])]
        uav.energy_ratio = float(item["energy_ratio"])
        if item.get("communication_state") == "LOST":
            uav.communication_state = CommunicationState.LOST
            uav.availability = UAVAvailability.COMMUNICATION_LOST
    runtime.validate_invariants()
    return runtime


def replay_tape(
    tape: Mapping[str, Any],
    *,
    controller: PreemptionController | None = None,
) -> tuple[ExecutionRuntime, tuple[EventDecision, ...]]:
    runtime = runtime_from_tape(tape)
    controller = controller or PreemptionController()
    grouped: list[list[RuntimeEvent]] = []
    batch_indexes: dict[str, int] = {}
    for item in tape["events"]:
        batch_id = str(item["batch_id"])
        if batch_id not in batch_indexes:
            batch_indexes[batch_id] = len(grouped)
            grouped.append([])
        index = batch_indexes[batch_id]
        grouped[index].append(RuntimeEvent(
            event_id=item["event_id"],
            event_type=RuntimeEventType(item["event_type"]),
            priority=EventPriority[item["priority"]],
            occurred_at=float(item["occurred_at"]),
            received_at=float(item["received_at"]),
            task_id=item.get("task_id"),
            uav_id=item.get("uav_id"),
            task_priority=int(item.get("task_priority", 0)),
            deadline=item.get("deadline"),
            confidence=float(item.get("confidence", 1.0)),
            payload=item.get("payload", {}),
        ))
    decisions: list[EventDecision] = []
    for events in grouped:
        now = max(item.received_at for item in events)
        result = runtime.process_event_batch(events, controller, now=now)
        expected_by_id = {
            source.event_id: _expected_rule_decision(str(tape["scenario_id"]), source)
            for source in events
        }
        actual_by_id = {item.event_id: item.decision.value for item in result.decisions}
        if actual_by_id != expected_by_id:
            raise AssertionError(f"decision mismatch: expected {expected_by_id}, got {actual_by_id}")
        decisions.extend(result.decisions)
        runtime.validate_invariants()
    return runtime, tuple(decisions)


def _expected_rule_decision(scenario_id: str, event: RuntimeEvent) -> str:
    single = {
        "urgent_at_40": "PREEMPT",
        "low_value_at_90": "QUEUE",
        "nonpreemptible_strike": "QUEUE",
        "execution_uav_destroyed": "MIGRATE",
        "low_energy_rtb": "RTB",
        "delayed_task_cancel": "ABORT",
        "repeated_priority_change": "CONTINUE",
        "simultaneous_p1": "CONTINUE",
    }
    if scenario_id in single:
        return single[scenario_id]
    if scenario_id == "event_during_inference":
        return "CONTINUE" if event.event_type is RuntimeEventType.TASK_ARRIVAL else "RTB"
    if scenario_id == "event_during_resume":
        return "CONTINUE" if event.event_type is RuntimeEventType.UAV_COMM_RECOVERED else "PREEMPT"
    raise ValueError(f"no rule oracle for scenario {scenario_id}")


def build_development_bank() -> tuple[dict[str, Any], ...]:
    tapes = tuple(
        build_development_tape(scenario_id, case_index)
        for scenario_id in _SCENARIO_IDS
        for case_index in range(CASES_PER_SCENARIO)
    )
    counts = Counter(tape["scenario_id"] for tape in tapes)
    if len(tapes) != 200 or any(count != CASES_PER_SCENARIO for count in counts.values()):
        raise AssertionError("development bank cardinality mismatch")
    if len({tape["case_seed"] for tape in tapes}) != len(tapes):
        raise AssertionError("development bank seeds are not unique")
    return tapes
