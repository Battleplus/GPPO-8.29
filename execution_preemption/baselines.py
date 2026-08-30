"""Frozen deterministic baselines for Execution-Preemption V1.

All three allocators operate only on an already filtered AllocationRequest.
They cannot change the rule arbiter decision or escape the safety mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .allocation import AllocationCandidate, AllocationProposal, AllocationRequest


def _proposal(
    request: AllocationRequest,
    candidate: AllocationCandidate,
    allocator_id: str,
    *,
    score: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AllocationProposal:
    return AllocationProposal(
        request_id=request.request_id,
        graph_version=request.graph_version,
        task_id=request.task_id,
        uav_id=candidate.uav_id,
        allocator_id=allocator_id,
        score=score,
        metadata=dict(metadata or {}),
    )


class SeniorLegacyMethodAllocator:
    """Adapter for the legacy first-legal/lexical dispatch behaviour.

    This is an adapted behavioural baseline, not a claim that the historical
    implementation is reproduced instruction-for-instruction.
    """

    allocator_id = "senior_legacy_method_v1"

    def propose(self, request: AllocationRequest) -> AllocationProposal:
        candidate = min(request.candidates, key=lambda item: item.uav_id)
        return _proposal(
            request,
            candidate,
            self.allocator_id,
            metadata={"semantics": "adapted_legacy_first_legal_lexical"},
        )


class GreedyPriorityAllocator:
    """Reactive deterministic baseline after rule-level priority arbitration."""

    allocator_id = "greedy_priority_v1"

    def propose(self, request: AllocationRequest) -> AllocationProposal:
        candidate = min(
            request.candidates,
            key=lambda item: (-item.energy_margin, -item.last_seen_at, item.uav_id),
        )
        return _proposal(
            request,
            candidate,
            self.allocator_id,
            score=candidate.energy_margin,
            metadata={"ranking": "energy_margin_desc/freshness_desc/uav_id"},
        )


@dataclass(frozen=True, slots=True)
class _BeamState:
    score: float
    remaining: tuple[tuple[str, float], ...]
    used: tuple[str, ...]
    trace: tuple[str, ...]


class BeamMPCAllocator:
    """Small deterministic receding-horizon planner over safe candidates.

    The public forecast contains pending tasks already visible to the runtime.
    Search never expands an unsafe current action and uses deterministic
    tie-breaking, so repeated requests produce byte-identical proposals.
    """

    allocator_id = "beam_mpc_v1"

    def __init__(self, *, horizon: int = 3, beam_width: int = 8) -> None:
        if horizon < 1 or beam_width < 1:
            raise ValueError("horizon and beam_width must be positive")
        self.horizon = int(horizon)
        self.beam_width = int(beam_width)

    @staticmethod
    def _supports(candidate: AllocationCandidate, task_type: str) -> bool:
        return not candidate.supported_task_types or task_type in candidate.supported_task_types

    @staticmethod
    def _task_value(
        task: Mapping[str, Any], rank: int, generated_at: float
    ) -> float:
        priority = max(0.0, float(task.get("priority", 0.0))) / 100.0
        deadline = task.get("deadline")
        urgency = (
            0.0 if deadline is None
            else 1.0 / (1.0 + max(0.0, float(deadline) - generated_at))
        )
        return (priority + urgency) * (0.95 ** rank)

    @staticmethod
    def _energy_cost(task: Mapping[str, Any]) -> float:
        remaining = min(1.0, max(0.0, float(task.get("remaining_work", 1.0))))
        return 0.05 + 0.15 * remaining

    def _score_current(
        self,
        request: AllocationRequest,
        current: AllocationCandidate,
        forecast: tuple[Mapping[str, Any], ...],
    ) -> tuple[float, tuple[str, ...]]:
        current_task = request.metadata.get("current_task", {})
        if not isinstance(current_task, Mapping):
            current_task = {}
        remaining = {
            item.uav_id: float(item.energy_margin)
            for item in request.candidates
        }
        remaining[current.uav_id] -= self._energy_cost(current_task)
        states = [_BeamState(
            score=remaining[current.uav_id],
            remaining=tuple(sorted(remaining.items())),
            used=(current.uav_id,),
            trace=(f"current:{current.uav_id}",),
        )]
        candidates = {item.uav_id: item for item in request.candidates}
        for rank, task in enumerate(forecast[: max(0, self.horizon - 1)]):
            task_type = str(task.get("task_type", ""))
            value = self._task_value(task, rank, request.generated_at)
            cost = self._energy_cost(task)
            expanded: list[_BeamState] = []
            for state in states:
                energy = dict(state.remaining)
                # A missed forecast task remains explicit in the search.
                expanded.append(_BeamState(
                    score=state.score - 2.0 * value,
                    remaining=state.remaining,
                    used=state.used,
                    trace=state.trace + (f"skip:{task.get('task_id', rank)}",),
                ))
                for uav_id in sorted(candidates):
                    candidate = candidates[uav_id]
                    if uav_id in state.used or not self._supports(candidate, task_type):
                        continue
                    if energy[uav_id] <= cost:
                        continue
                    updated = dict(energy)
                    updated[uav_id] -= cost
                    expanded.append(_BeamState(
                        score=state.score + 2.0 * value + updated[uav_id],
                        remaining=tuple(sorted(updated.items())),
                        used=tuple(sorted((*state.used, uav_id))),
                        trace=state.trace + (f"forecast:{task.get('task_id', rank)}:{uav_id}",),
                    ))
            states = sorted(
                expanded,
                key=lambda item: (-item.score, item.trace, item.remaining),
            )[: self.beam_width]
        best = min(states, key=lambda item: (-item.score, item.trace, item.remaining))
        if not math.isfinite(best.score):
            raise ValueError("beam planner produced a non-finite score")
        return best.score, best.trace

    def propose(self, request: AllocationRequest) -> AllocationProposal:
        raw_forecast = request.metadata.get("forecast_tasks", ())
        if not isinstance(raw_forecast, (list, tuple)):
            raise ValueError("forecast_tasks must be a sequence")
        forecast = tuple(sorted(
            (item for item in raw_forecast if isinstance(item, Mapping)),
            key=lambda item: (
                -float(item.get("priority", 0.0)),
                float(item["deadline"]) if item.get("deadline") is not None else float("inf"),
                str(item.get("task_id", "")),
            ),
        ))
        ranked: list[tuple[float, tuple[str, ...], AllocationCandidate]] = []
        for candidate in request.candidates:
            score, trace = self._score_current(request, candidate, forecast)
            ranked.append((score, trace, candidate))
        score, trace, candidate = min(
            ranked,
            key=lambda item: (-item[0], item[1], item[2].uav_id),
        )
        return _proposal(
            request,
            candidate,
            self.allocator_id,
            score=score,
            metadata={
                "horizon": self.horizon,
                "beam_width": self.beam_width,
                "forecast_count": min(len(forecast), max(0, self.horizon - 1)),
                "winning_trace": list(trace),
            },
        )
