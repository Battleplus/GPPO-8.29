"""Versioned, algorithm-neutral UAV--Task allocation boundary.

Safety-critical CONTINUE/PREEMPT/ABORT/RTB decisions remain in the rule
controller.  An allocator is only allowed to select one UAV from a frozen,
validated candidate set.  This is the future integration point for legacy,
greedy, PPO, GPPO and planning methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .models import (
    CommunicationState,
    DecisionType,
    TaskRuntime,
    UAVAvailability,
    UAVRuntime,
)


class AllocationValidationError(RuntimeError):
    """Raised when an allocator proposal violates the frozen request."""


@dataclass(frozen=True)
class AllocationCandidate:
    task_id: str
    uav_id: str
    energy_ratio: float
    energy_margin: float
    last_seen_at: float
    supported_task_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class AllocationRequest:
    request_id: str
    graph_version: int
    task_id: str
    decision_type: DecisionType
    reason: str
    generated_at: float
    candidates: tuple[AllocationCandidate, ...]
    displaced_task_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidate_uav_ids(self) -> tuple[str, ...]:
        return tuple(candidate.uav_id for candidate in self.candidates)


@dataclass(frozen=True)
class AllocationProposal:
    request_id: str
    graph_version: int
    task_id: str
    uav_id: str
    allocator_id: str
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Allocator(Protocol):
    allocator_id: str

    def propose(self, request: AllocationRequest) -> AllocationProposal:
        """Choose exactly one UAV from ``request.candidates``."""


def build_allocation_request(
    *,
    request_id: str,
    graph_version: int,
    task: TaskRuntime,
    uavs: Sequence[UAVRuntime],
    decision_type: DecisionType,
    reason: str,
    generated_at: float,
    displaced_task_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AllocationRequest:
    if graph_version < 0:
        raise ValueError("graph_version must be non-negative")
    if not request_id:
        raise ValueError("request_id is required")
    candidates: list[AllocationCandidate] = []
    for uav in uavs:
        if uav.availability is not UAVAvailability.AVAILABLE:
            continue
        if uav.communication_state is not CommunicationState.CONNECTED:
            continue
        if not uav.energy_safe_for_new_task or not uav.supports(task.task_type):
            continue
        candidates.append(AllocationCandidate(
            task_id=task.task_id,
            uav_id=uav.uav_id,
            energy_ratio=float(uav.energy_ratio),
            energy_margin=float(
                uav.energy_ratio - uav.reserve_energy - uav.estimated_rtb_energy
            ),
            last_seen_at=float(uav.last_seen_at),
            supported_task_types=tuple(sorted(uav.supported_task_types)),
        ))
    candidates.sort(key=lambda item: item.uav_id)
    if not candidates:
        raise AllocationValidationError("allocation request has no safe candidate")
    return AllocationRequest(
        request_id=request_id,
        graph_version=int(graph_version),
        task_id=task.task_id,
        decision_type=decision_type,
        reason=str(reason),
        generated_at=float(generated_at),
        candidates=tuple(candidates),
        displaced_task_id=displaced_task_id,
        metadata=dict(metadata or {}),
    )


def validate_proposal(
    request: AllocationRequest,
    proposal: AllocationProposal,
    *,
    current_graph_version: int,
) -> AllocationProposal:
    if proposal.request_id != request.request_id:
        raise AllocationValidationError("proposal request_id mismatch")
    if proposal.graph_version != request.graph_version:
        raise AllocationValidationError("proposal graph_version differs from request")
    if proposal.graph_version != current_graph_version:
        raise AllocationValidationError("proposal is stale against live graph_version")
    if proposal.task_id != request.task_id:
        raise AllocationValidationError("proposal task_id mismatch")
    if proposal.uav_id not in request.candidate_uav_ids:
        raise AllocationValidationError("proposal selected a UAV outside the safe candidate set")
    if not proposal.allocator_id:
        raise AllocationValidationError("allocator_id is required")
    return proposal


class FirstAvailableAllocator:
    """Legacy-like deterministic baseline using lexical UAV order."""

    allocator_id = "first_available_v1"

    def propose(self, request: AllocationRequest) -> AllocationProposal:
        candidate = request.candidates[0]
        return AllocationProposal(
            request_id=request.request_id,
            graph_version=request.graph_version,
            task_id=request.task_id,
            uav_id=candidate.uav_id,
            allocator_id=self.allocator_id,
        )


class MaxEnergyMarginAllocator:
    """Greedy baseline selecting the largest post-RTB energy margin."""

    allocator_id = "max_energy_margin_v1"

    def propose(self, request: AllocationRequest) -> AllocationProposal:
        candidate = min(
            request.candidates,
            key=lambda item: (-item.energy_margin, item.last_seen_at, item.uav_id),
        )
        return AllocationProposal(
            request_id=request.request_id,
            graph_version=request.graph_version,
            task_id=request.task_id,
            uav_id=candidate.uav_id,
            allocator_id=self.allocator_id,
            score=candidate.energy_margin,
        )


class CallbackAllocator:
    """Adapter for a PPO/GPPO/planner callback returning a proposal.

    The callback sees only the frozen request.  Its output still passes the
    exact graph-version and safe-candidate validation performed by the rule
    controller.
    """

    def __init__(
        self,
        allocator_id: str,
        callback: Callable[[AllocationRequest], AllocationProposal | str],
    ) -> None:
        if not allocator_id:
            raise ValueError("allocator_id is required")
        self.allocator_id = allocator_id
        self._callback = callback

    def propose(self, request: AllocationRequest) -> AllocationProposal:
        result = self._callback(request)
        if isinstance(result, AllocationProposal):
            return result
        return AllocationProposal(
            request_id=request.request_id,
            graph_version=request.graph_version,
            task_id=request.task_id,
            uav_id=str(result),
            allocator_id=self.allocator_id,
        )
