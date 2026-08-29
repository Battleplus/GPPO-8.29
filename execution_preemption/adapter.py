"""Deterministic PPO/GPPO/Planner observation and action adapter.

The adapter exposes one shared, version-bound action table.  PPO receives a
fixed-size flattened encoding; GPPO/Planner receives typed nodes and indexed
relations.  Both decode through the exact same mask and proposal validator.
No framework-specific tensors are created here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

from .allocation import (
    AllocationProposal,
    AllocationRequest,
    AllocationValidationError,
    validate_proposal,
)
from .graph import (
    ExecutionGraphSnapshot,
    FEATURE_DIMENSIONS,
    NODE_TYPES,
    RELATIONS,
)
from .models import DecisionType, EventDecision, EventPriority


ADAPTER_ID = "execution-preemption-policy-adapter-v1"
MAX_UAVS = 32
MAX_TASKS_PER_UAV = 3
MAX_TASKS = MAX_UAVS * MAX_TASKS_PER_UAV
MAX_REGIONS = MAX_TASKS
MAX_TARGETS = MAX_UAVS
MAX_EVENTS = 8
RULE_CONTEXT_DIMENSION = 16

NODE_CAPACITIES: Mapping[str, int] = {
    "UAV": MAX_UAVS,
    "Task": MAX_TASKS,
    "Region": MAX_REGIONS,
    "Target": MAX_TARGETS,
    "Event": MAX_EVENTS,
}

ACTION_CAPACITY = 1 + MAX_UAVS * MAX_TASKS


class AdapterValidationError(RuntimeError):
    """Raised when an observation/action cannot be bound to the live graph."""


@dataclass(frozen=True, slots=True)
class ActionSpaceSnapshot:
    adapter_id: str
    graph_version: int
    graph_sha256: str
    bindings: tuple[tuple[str, str] | None, ...]
    mask: tuple[bool, ...]

    def validate(self) -> None:
        if self.adapter_id != ADAPTER_ID:
            raise AdapterValidationError("unsupported adapter_id")
        if self.graph_version < 0 or len(self.graph_sha256) != 64:
            raise AdapterValidationError("invalid graph binding")
        if len(self.bindings) != ACTION_CAPACITY or len(self.mask) != ACTION_CAPACITY:
            raise AdapterValidationError("action capacity drift")
        if self.bindings[0] != ("NOOP", "NOOP") or not self.mask[0]:
            raise AdapterValidationError("action zero must be enabled NOOP")
        for index, enabled in enumerate(self.mask[1:], start=1):
            if enabled and self.bindings[index] is None:
                raise AdapterValidationError("enabled action has no binding")

    @property
    def valid_action_count(self) -> int:
        return sum(self.mask)


@dataclass(frozen=True, slots=True)
class FlatPolicyObservation:
    adapter_id: str
    graph_version: int
    graph_sha256: str
    vector: tuple[float, ...]
    node_ids: Mapping[str, tuple[str | None, ...]]
    action_space: ActionSpaceSnapshot
    rule_context_present: bool

    def validate(self) -> None:
        if self.adapter_id != ADAPTER_ID:
            raise AdapterValidationError("unsupported flat adapter")
        if len(self.vector) != FLAT_OBSERVATION_DIMENSION:
            raise AdapterValidationError("flat observation dimension drift")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in self.vector):
            raise AdapterValidationError("flat observation must be normalized")
        self.action_space.validate()
        if (self.graph_version, self.graph_sha256) != (
            self.action_space.graph_version,
            self.action_space.graph_sha256,
        ):
            raise AdapterValidationError("flat observation/action graph binding mismatch")


@dataclass(frozen=True, slots=True)
class HeteroPolicyObservation:
    adapter_id: str
    graph_version: int
    graph_sha256: str
    node_ids: Mapping[str, tuple[str, ...]]
    node_features: Mapping[str, tuple[tuple[float, ...], ...]]
    edge_indices: Mapping[tuple[str, str, str], tuple[tuple[int, int], ...]]
    rule_context: tuple[float, ...]
    action_space: ActionSpaceSnapshot

    def validate(self) -> None:
        if self.adapter_id != ADAPTER_ID:
            raise AdapterValidationError("unsupported hetero adapter")
        if set(self.node_ids) != set(NODE_TYPES) or set(self.node_features) != set(NODE_TYPES):
            raise AdapterValidationError("all node types are required")
        if set(self.edge_indices) != set(RELATIONS):
            raise AdapterValidationError("all relation types are required")
        for node_type in NODE_TYPES:
            if len(self.node_ids[node_type]) != len(self.node_features[node_type]):
                raise AdapterValidationError("node id/feature cardinality mismatch")
            for row in self.node_features[node_type]:
                if len(row) != FEATURE_DIMENSIONS[node_type]:
                    raise AdapterValidationError("node feature dimension drift")
        if len(self.rule_context) != RULE_CONTEXT_DIMENSION:
            raise AdapterValidationError("rule context dimension drift")
        self.action_space.validate()
        if (self.graph_version, self.graph_sha256) != (
            self.action_space.graph_version,
            self.action_space.graph_sha256,
        ):
            raise AdapterValidationError("hetero observation/action graph binding mismatch")


@dataclass(frozen=True, slots=True)
class DecodedPolicyAction:
    action_index: int
    graph_version: int
    graph_sha256: str
    noop: bool
    uav_id: str | None
    task_id: str | None


def _relation_capacity(relation: tuple[str, str, str]) -> int:
    return NODE_CAPACITIES[relation[0]] * NODE_CAPACITIES[relation[2]]


NODE_FEATURE_VECTOR_DIMENSION = sum(
    NODE_CAPACITIES[node_type] * FEATURE_DIMENSIONS[node_type]
    for node_type in NODE_TYPES
)
NODE_PRESENCE_VECTOR_DIMENSION = sum(NODE_CAPACITIES.values())
EDGE_VECTOR_DIMENSION = sum(_relation_capacity(relation) for relation in RELATIONS)
FLAT_OBSERVATION_DIMENSION = (
    NODE_FEATURE_VECTOR_DIMENSION
    + NODE_PRESENCE_VECTOR_DIMENSION
    + EDGE_VECTOR_DIMENSION
    + RULE_CONTEXT_DIMENSION
)


def adapter_layout() -> dict[str, object]:
    return {
        "adapter_id": ADAPTER_ID,
        "node_capacities": dict(NODE_CAPACITIES),
        "node_feature_dimensions": dict(FEATURE_DIMENSIONS),
        "relations": [list(item) for item in RELATIONS],
        "rule_context_dimension": RULE_CONTEXT_DIMENSION,
        "flat_observation_dimension": FLAT_OBSERVATION_DIMENSION,
        "action_capacity": ACTION_CAPACITY,
        "action_zero": ["NOOP", "NOOP"],
    }


def adapter_layout_sha256() -> str:
    payload = json.dumps(
        adapter_layout(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _slots(snapshot: ExecutionGraphSnapshot) -> tuple[
    dict[str, tuple[str | None, ...]],
    dict[str, dict[str, int]],
]:
    slot_ids: dict[str, tuple[str | None, ...]] = {}
    indices: dict[str, dict[str, int]] = {}
    for node_type in NODE_TYPES:
        ids = tuple(sorted(node.node_id for node in snapshot.nodes[node_type]))
        capacity = NODE_CAPACITIES[node_type]
        if len(ids) > capacity:
            raise AdapterValidationError(
                f"{node_type} count {len(ids)} exceeds frozen capacity {capacity}"
            )
        slot_ids[node_type] = ids + (None,) * (capacity - len(ids))
        indices[node_type] = {node_id: index for index, node_id in enumerate(ids)}
    return slot_ids, indices


def _rule_context(decision: EventDecision | None) -> tuple[float, ...]:
    if decision is None:
        return (0.0,) * RULE_CONTEXT_DIMENSION
    values = (
        1.0,
        *(1.0 if decision.decision is item else 0.0 for item in DecisionType),
        *(1.0 if decision.priority is item else 0.0 for item in EventPriority),
        min(1.0, max(0.0, float(decision.information_age) / 10.0)),
        min(1.0, max(0.0, float(decision.confidence))),
        1.0 if decision.displaced_task_id is not None else 0.0,
    )
    if len(values) != RULE_CONTEXT_DIMENSION:
        raise AdapterValidationError("internal rule context dimension mismatch")
    return tuple(float(value) for value in values)


def _validate_request_against_graph(
    snapshot: ExecutionGraphSnapshot,
    request: AllocationRequest | None,
) -> set[tuple[str, str]]:
    graph_candidates = set(snapshot.action_candidates)
    if request is None:
        return graph_candidates
    if request.graph_version != snapshot.graph_version:
        raise AdapterValidationError("allocation request graph_version mismatch")
    requested = {(candidate.uav_id, request.task_id) for candidate in request.candidates}
    if not requested:
        raise AdapterValidationError("allocation request has no candidate")
    if not requested.issubset(graph_candidates):
        raise AdapterValidationError("allocation request candidate is absent from graph mask")
    return requested


def _action_space(
    snapshot: ExecutionGraphSnapshot,
    slot_ids: Mapping[str, tuple[str | None, ...]],
    active_candidates: set[tuple[str, str]],
) -> ActionSpaceSnapshot:
    bindings: list[tuple[str, str] | None] = [None] * ACTION_CAPACITY
    mask = [False] * ACTION_CAPACITY
    bindings[0] = ("NOOP", "NOOP")
    mask[0] = True
    for uav_index, uav_id in enumerate(slot_ids["UAV"]):
        if uav_id is None:
            continue
        for task_index, task_id in enumerate(slot_ids["Task"]):
            if task_id is None:
                continue
            index = 1 + uav_index * MAX_TASKS + task_index
            pair = (uav_id, task_id)
            bindings[index] = pair
            mask[index] = pair in active_candidates
    value = ActionSpaceSnapshot(
        adapter_id=ADAPTER_ID,
        graph_version=snapshot.graph_version,
        graph_sha256=snapshot.sha256(),
        bindings=tuple(bindings),
        mask=tuple(mask),
    )
    value.validate()
    return value


def build_flat_observation(
    snapshot: ExecutionGraphSnapshot,
    *,
    request: AllocationRequest | None = None,
    decision: EventDecision | None = None,
) -> FlatPolicyObservation:
    snapshot.validate()
    if decision is not None and decision.graph_version != snapshot.graph_version:
        raise AdapterValidationError("EventDecision graph_version mismatch")
    slot_ids, indices = _slots(snapshot)
    candidates = _validate_request_against_graph(snapshot, request)
    graph_sha = snapshot.sha256()
    vector: list[float] = []
    node_lookup = {
        node_type: {node.node_id: node for node in snapshot.nodes[node_type]}
        for node_type in NODE_TYPES
    }
    for node_type in NODE_TYPES:
        zero_row = (0.0,) * FEATURE_DIMENSIONS[node_type]
        for node_id in slot_ids[node_type]:
            vector.extend(zero_row if node_id is None else node_lookup[node_type][node_id].features)
    for node_type in NODE_TYPES:
        vector.extend(0.0 if node_id is None else 1.0 for node_id in slot_ids[node_type])
    edge_sets = {
        relation: {
            (indices[edge.src_type][edge.src_id], indices[edge.dst_type][edge.dst_id])
            for edge in snapshot.edges
            if edge.relation_key == relation
        }
        for relation in RELATIONS
    }
    for relation in RELATIONS:
        src_capacity = NODE_CAPACITIES[relation[0]]
        dst_capacity = NODE_CAPACITIES[relation[2]]
        present = edge_sets[relation]
        vector.extend(
            1.0 if (src_index, dst_index) in present else 0.0
            for src_index in range(src_capacity)
            for dst_index in range(dst_capacity)
        )
    context = _rule_context(decision)
    vector.extend(context)
    action_space = _action_space(snapshot, slot_ids, candidates)
    value = FlatPolicyObservation(
        adapter_id=ADAPTER_ID,
        graph_version=snapshot.graph_version,
        graph_sha256=graph_sha,
        vector=tuple(vector),
        node_ids=slot_ids,
        action_space=action_space,
        rule_context_present=decision is not None,
    )
    value.validate()
    return value


def build_hetero_observation(
    snapshot: ExecutionGraphSnapshot,
    *,
    request: AllocationRequest | None = None,
    decision: EventDecision | None = None,
) -> HeteroPolicyObservation:
    snapshot.validate()
    if decision is not None and decision.graph_version != snapshot.graph_version:
        raise AdapterValidationError("EventDecision graph_version mismatch")
    slot_ids, indices = _slots(snapshot)
    candidates = _validate_request_against_graph(snapshot, request)
    node_ids = {
        node_type: tuple(node.node_id for node in sorted(
            snapshot.nodes[node_type], key=lambda item: item.node_id
        ))
        for node_type in NODE_TYPES
    }
    node_features = {
        node_type: tuple(node.features for node in sorted(
            snapshot.nodes[node_type], key=lambda item: item.node_id
        ))
        for node_type in NODE_TYPES
    }
    edge_indices = {
        relation: tuple(sorted(
            (indices[edge.src_type][edge.src_id], indices[edge.dst_type][edge.dst_id])
            for edge in snapshot.edges
            if edge.relation_key == relation
        ))
        for relation in RELATIONS
    }
    action_space = _action_space(snapshot, slot_ids, candidates)
    value = HeteroPolicyObservation(
        adapter_id=ADAPTER_ID,
        graph_version=snapshot.graph_version,
        graph_sha256=snapshot.sha256(),
        node_ids=node_ids,
        node_features=node_features,
        edge_indices=edge_indices,
        rule_context=_rule_context(decision),
        action_space=action_space,
    )
    value.validate()
    return value


def decode_policy_action(
    action_space: ActionSpaceSnapshot,
    action_index: int,
    *,
    current_graph_version: int,
    current_graph_sha256: str,
) -> DecodedPolicyAction:
    action_space.validate()
    if current_graph_version != action_space.graph_version:
        raise AdapterValidationError("policy action is stale against live graph_version")
    if current_graph_sha256 != action_space.graph_sha256:
        raise AdapterValidationError("policy action is stale against live graph hash")
    if isinstance(action_index, bool) or int(action_index) != action_index:
        raise AdapterValidationError("action_index must be an integer")
    index = int(action_index)
    if not 0 <= index < ACTION_CAPACITY:
        raise AdapterValidationError("action_index is outside frozen capacity")
    if not action_space.mask[index]:
        raise AdapterValidationError("action_index is masked")
    binding = action_space.bindings[index]
    if binding is None:
        raise AdapterValidationError("action_index has no binding")
    if index == 0:
        return DecodedPolicyAction(
            action_index=0,
            graph_version=action_space.graph_version,
            graph_sha256=action_space.graph_sha256,
            noop=True,
            uav_id=None,
            task_id=None,
        )
    return DecodedPolicyAction(
        action_index=index,
        graph_version=action_space.graph_version,
        graph_sha256=action_space.graph_sha256,
        noop=False,
        uav_id=binding[0],
        task_id=binding[1],
    )


def proposal_from_policy_action(
    request: AllocationRequest,
    action_space: ActionSpaceSnapshot,
    action_index: int,
    *,
    allocator_id: str,
    current_graph_version: int,
    current_graph_sha256: str,
) -> AllocationProposal:
    decoded = decode_policy_action(
        action_space,
        action_index,
        current_graph_version=current_graph_version,
        current_graph_sha256=current_graph_sha256,
    )
    if decoded.noop:
        raise AllocationValidationError("NOOP cannot satisfy a required allocation proposal")
    proposal = AllocationProposal(
        request_id=request.request_id,
        graph_version=decoded.graph_version,
        task_id=str(decoded.task_id),
        uav_id=str(decoded.uav_id),
        allocator_id=str(allocator_id),
        metadata={
            "adapter_id": ADAPTER_ID,
            "graph_sha256": decoded.graph_sha256,
            "action_index": decoded.action_index,
        },
    )
    return validate_proposal(request, proposal, current_graph_version=current_graph_version)


__all__ = [
    "ACTION_CAPACITY",
    "ADAPTER_ID",
    "AdapterValidationError",
    "ActionSpaceSnapshot",
    "DecodedPolicyAction",
    "EDGE_VECTOR_DIMENSION",
    "FLAT_OBSERVATION_DIMENSION",
    "FlatPolicyObservation",
    "HeteroPolicyObservation",
    "MAX_EVENTS",
    "MAX_REGIONS",
    "MAX_TARGETS",
    "MAX_TASKS",
    "MAX_TASKS_PER_UAV",
    "MAX_UAVS",
    "NODE_CAPACITIES",
    "NODE_FEATURE_VECTOR_DIMENSION",
    "NODE_PRESENCE_VECTOR_DIMENSION",
    "RULE_CONTEXT_DIMENSION",
    "adapter_layout",
    "adapter_layout_sha256",
    "build_flat_observation",
    "build_hetero_observation",
    "decode_policy_action",
    "proposal_from_policy_action",
]

