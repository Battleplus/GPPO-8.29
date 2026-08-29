"""Frozen heterogeneous observation schema for execution-preemption V1."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from .allocation import AllocationRequest
from .models import (
    CommunicationState,
    EventPriority,
    RuntimeEvent,
    TaskRuntime,
    TaskState,
    TERMINAL_TASK_STATES,
    UAVAvailability,
    UAVRuntime,
)
from .runtime import ExecutionRuntime


SCHEMA_ID = "execution-preemption-heterograph-v1"
SCHEMA_VERSION = 1
NODE_TYPES = ("UAV", "Task", "Region", "Target", "Event")
FEATURE_DIMENSIONS = {"UAV": 15, "Task": 17, "Region": 5, "Target": 6, "Event": 12}
RELATIONS = (
    ("UAV", "executes", "Task"),
    ("UAV", "can_execute", "Task"),
    ("Task", "located_in", "Region"),
    ("Task", "depends_on", "Task"),
    ("Event", "affects", "UAV"),
    ("Event", "affects", "Task"),
    ("Task", "preempts", "Task"),
)

PRIORITY_SCALE = 100.0
DEADLINE_HORIZON = 100.0
INFORMATION_AGE_HORIZON = 10.0
COMMUNICATION_AGE_HORIZON = 10.0
INTERRUPTION_COUNT_SCALE = 10.0


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _one_hot(value: Any, choices: tuple[Any, ...]) -> tuple[float, ...]:
    return tuple(1.0 if value == choice else 0.0 for choice in choices)


@dataclass(frozen=True)
class RegionRuntime:
    region_id: str
    vacant: bool = False
    demand: float = 0.0
    priority: int = 0
    deadline: float | None = None
    uncertainty: float = 0.0


@dataclass(frozen=True)
class TargetRuntime:
    target_id: str
    status: str = "UNKNOWN"
    confidence: float = 0.0
    priority: int = 0


@dataclass(frozen=True)
class GraphNode:
    node_type: str
    node_id: str
    features: tuple[float, ...]


@dataclass(frozen=True, order=True)
class GraphEdge:
    src_type: str
    relation: str
    dst_type: str
    src_id: str
    dst_id: str

    @property
    def relation_key(self) -> tuple[str, str, str]:
        return self.src_type, self.relation, self.dst_type


@dataclass(frozen=True)
class ExecutionGraphSnapshot:
    schema_id: str
    schema_version: int
    graph_version: int
    generated_at: float
    nodes: Mapping[str, tuple[GraphNode, ...]]
    edges: tuple[GraphEdge, ...]
    action_candidates: tuple[tuple[str, str], ...]
    noop_action: tuple[str, str] = ("NOOP", "NOOP")

    def validate(self) -> None:
        if self.schema_id != SCHEMA_ID or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported graph schema")
        if set(self.nodes) != set(NODE_TYPES):
            raise ValueError("all five node types must be present")
        node_ids: dict[str, set[str]] = {}
        for node_type in NODE_TYPES:
            items = self.nodes[node_type]
            ids = [item.node_id for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {node_type} node id")
            node_ids[node_type] = set(ids)
            for item in items:
                if item.node_type != node_type:
                    raise ValueError("node stored under wrong node type")
                if len(item.features) != FEATURE_DIMENSIONS[node_type]:
                    raise ValueError(f"{node_type} feature dimension mismatch")
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in item.features):
                    raise ValueError("features must be finite and normalized to [0, 1]")
        if tuple(sorted(self.edges)) != self.edges or len(set(self.edges)) != len(self.edges):
            raise ValueError("edges must be sorted and unique")
        for edge in self.edges:
            if edge.relation_key not in RELATIONS:
                raise ValueError(f"unknown relation {edge.relation_key}")
            if edge.src_id not in node_ids[edge.src_type] or edge.dst_id not in node_ids[edge.dst_type]:
                raise ValueError("edge references unknown node")
        expected_candidates = {
            (edge.src_id, edge.dst_id)
            for edge in self.edges
            if edge.relation_key == ("UAV", "can_execute", "Task")
        }
        if tuple(sorted(expected_candidates)) != self.action_candidates:
            raise ValueError("action candidates must exactly match can_execute edges")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "graph_version": self.graph_version,
            "generated_at": self.generated_at,
            "nodes": {
                node_type: [
                    {"node_id": item.node_id, "features": list(item.features)}
                    for item in self.nodes[node_type]
                ]
                for node_type in NODE_TYPES
            },
            "edges": [
                {
                    "src_type": edge.src_type,
                    "relation": edge.relation,
                    "dst_type": edge.dst_type,
                    "src_id": edge.src_id,
                    "dst_id": edge.dst_id,
                }
                for edge in self.edges
            ],
            "action_candidates": [list(item) for item in self.action_candidates],
            "noop_action": list(self.noop_action),
        }

    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _uav_features(uav: UAVRuntime, tasks: Mapping[str, TaskRuntime], now: float) -> tuple[float, ...]:
    active_progress = 0.0
    if uav.active_task_id is not None and uav.active_task_id in tasks:
        active_progress = tasks[uav.active_task_id].progress
    availability = tuple(UAVAvailability)
    communication = tuple(CommunicationState)
    return (
        _clip01(uav.energy_ratio),
        _clip01(uav.reserve_energy),
        _clip01(uav.estimated_rtb_energy),
        _clip01(uav.energy_ratio - uav.reserve_energy - uav.estimated_rtb_energy),
        _clip01(max(0.0, now - uav.last_seen_at) / COMMUNICATION_AGE_HORIZON),
        _clip01(active_progress),
        *_one_hot(uav.availability, availability),
        *_one_hot(uav.communication_state, communication),
    )


def _task_features(task: TaskRuntime, now: float) -> tuple[float, ...]:
    deadline_remaining = (
        1.0 if task.deadline is None
        else _clip01(max(0.0, task.deadline - now) / DEADLINE_HORIZON)
    )
    return (
        _clip01(task.priority / PRIORITY_SCALE),
        deadline_remaining,
        _clip01(task.progress),
        _clip01(task.remaining_work),
        1.0 if task.preemptible else 0.0,
        _clip01(task.interruption_count / INTERRUPTION_COUNT_SCALE),
        1.0 if task.exclusive else 0.0,
        *_one_hot(task.state, tuple(TaskState)),
    )


def _region_features(region: RegionRuntime, now: float) -> tuple[float, ...]:
    deadline_remaining = (
        1.0 if region.deadline is None
        else _clip01(max(0.0, region.deadline - now) / DEADLINE_HORIZON)
    )
    return (
        1.0 if region.vacant else 0.0,
        _clip01(region.demand),
        _clip01(region.priority / PRIORITY_SCALE),
        deadline_remaining,
        _clip01(region.uncertainty),
    )


def _target_features(target: TargetRuntime) -> tuple[float, ...]:
    statuses = ("UNKNOWN", "DETECTED", "TRACKED", "DESTROYED")
    status = target.status.upper()
    if status not in statuses:
        raise ValueError(f"unsupported target status {target.status}")
    return (
        *_one_hot(status, statuses),
        _clip01(target.confidence),
        _clip01(target.priority / PRIORITY_SCALE),
    )


def _event_features(event: RuntimeEvent, now: float) -> tuple[float, ...]:
    event_types = tuple(type_item for type_item in type(event.event_type))
    severity = float(event.payload.get("severity", 1.0))
    return (
        _clip01(int(event.priority) / max(int(EventPriority.P4), 1)),
        _clip01(severity),
        _clip01(max(0.0, now - event.occurred_at) / INFORMATION_AGE_HORIZON),
        _clip01(event.confidence),
        *_one_hot(event.event_type, event_types),
    )


def _can_execute(task: TaskRuntime, uav: UAVRuntime) -> bool:
    if task.state in TERMINAL_TASK_STATES:
        return False
    if task.state not in {
        TaskState.PENDING,
        TaskState.PAUSED,
        TaskState.PREEMPTED,
        TaskState.MIGRATING,
        TaskState.RESUMING,
    }:
        return False
    same_paused_owner = (
        task.state is TaskState.PAUSED
        and task.assigned_uav == uav.uav_id
        and uav.availability is UAVAvailability.PAUSED
    )
    if uav.availability is not UAVAvailability.AVAILABLE and not same_paused_owner:
        return False
    return (
        uav.communication_state is CommunicationState.CONNECTED
        and uav.energy_safe_for_new_task
        and uav.supports(task.task_type)
    )


def build_execution_graph(
    runtime: ExecutionRuntime,
    *,
    now: float,
    events: Iterable[RuntimeEvent] = (),
    regions: Iterable[RegionRuntime] = (),
    targets: Iterable[TargetRuntime] = (),
    preemption_links: Iterable[tuple[str, str]] = (),
    allocation_request: AllocationRequest | None = None,
) -> ExecutionGraphSnapshot:
    event_items = tuple(sorted(events, key=lambda item: item.event_id))
    region_items = tuple(sorted(regions, key=lambda item: item.region_id))
    target_items = tuple(sorted(targets, key=lambda item: item.target_id))
    tasks = runtime.tasks
    uavs = runtime.uavs
    nodes: dict[str, tuple[GraphNode, ...]] = {
        "UAV": tuple(
            GraphNode("UAV", item.uav_id, _uav_features(item, tasks, now))
            for item in sorted(uavs.values(), key=lambda value: value.uav_id)
        ),
        "Task": tuple(
            GraphNode("Task", item.task_id, _task_features(item, now))
            for item in sorted(tasks.values(), key=lambda value: value.task_id)
        ),
        "Region": tuple(
            GraphNode("Region", item.region_id, _region_features(item, now))
            for item in region_items
        ),
        "Target": tuple(
            GraphNode("Target", item.target_id, _target_features(item))
            for item in target_items
        ),
        "Event": tuple(
            GraphNode("Event", item.event_id, _event_features(item, now))
            for item in event_items
        ),
    }
    region_ids = {item.region_id for item in region_items}
    edges: set[GraphEdge] = set()
    for task in tasks.values():
        if task.assigned_uav is not None:
            edges.add(GraphEdge("UAV", "executes", "Task", task.assigned_uav, task.task_id))
        for uav in uavs.values():
            if _can_execute(task, uav):
                edges.add(GraphEdge("UAV", "can_execute", "Task", uav.uav_id, task.task_id))
        region_id = task.metadata.get("region_id")
        if region_id is not None and str(region_id) in region_ids:
            edges.add(GraphEdge("Task", "located_in", "Region", task.task_id, str(region_id)))
        for dependency in task.metadata.get("depends_on", ()):
            if str(dependency) in tasks:
                edges.add(GraphEdge("Task", "depends_on", "Task", task.task_id, str(dependency)))
    for event in event_items:
        if event.uav_id is not None and event.uav_id in uavs:
            edges.add(GraphEdge("Event", "affects", "UAV", event.event_id, event.uav_id))
        if event.task_id is not None and event.task_id in tasks:
            edges.add(GraphEdge("Event", "affects", "Task", event.event_id, event.task_id))
    for source, destination in preemption_links:
        if source in tasks and destination in tasks:
            edges.add(GraphEdge("Task", "preempts", "Task", source, destination))
    if allocation_request is not None:
        if allocation_request.graph_version != runtime.graph_version:
            raise ValueError("allocation request graph_version does not match runtime")
        if allocation_request.task_id not in tasks:
            raise ValueError("allocation request references unknown task")
        for candidate in allocation_request.candidates:
            if candidate.uav_id not in uavs:
                raise ValueError("allocation request references unknown UAV")
            if candidate.task_id != allocation_request.task_id:
                raise ValueError("allocation request candidate task_id mismatch")
            edges.add(GraphEdge(
                "UAV",
                "can_execute",
                "Task",
                candidate.uav_id,
                allocation_request.task_id,
            ))
    sorted_edges = tuple(sorted(edges))
    candidates = tuple(sorted(
        (edge.src_id, edge.dst_id)
        for edge in sorted_edges
        if edge.relation_key == ("UAV", "can_execute", "Task")
    ))
    snapshot = ExecutionGraphSnapshot(
        schema_id=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        graph_version=runtime.graph_version,
        generated_at=float(now),
        nodes=nodes,
        edges=sorted_edges,
        action_candidates=candidates,
    )
    snapshot.validate()
    return snapshot
