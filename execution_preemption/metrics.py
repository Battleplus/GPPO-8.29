"""Algorithm-independent metrics for Execution-Preemption V1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable

from .reward import TransitionSignals


METRICS_SCHEMA_ID = "execution-preemption-metrics-v1"


def _finite_non_negative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _mean(values: Iterable[float]) -> float | None:
    items = tuple(float(item) for item in values)
    return float(math.fsum(items) / len(items)) if items else None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    items = sorted(float(item) for item in values)
    if not items:
        return None
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    position = (len(items) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    fraction = position - lower
    return float(items[lower] * (1.0 - fraction) + items[upper] * fraction)


@dataclass(frozen=True, slots=True)
class ExecutionEpisodeMetrics:
    schema_id: str
    algorithm_id: str
    tape_id: str
    uav_count: int
    accepted_decision_count: int
    urgent_task_count: int
    urgent_deadline_miss_count: int
    urgent_deadline_miss_rate: float | None
    p0_event_count: int
    p0_handled_count: int
    p0_handling_rate: float | None
    displaced_task_count: int
    resumed_task_count: int
    right_censored_recovery_count: int
    normal_task_recovery_rate: float | None
    mean_recovery_latency: float | None
    mean_preemption_response_latency: float | None
    cumulative_weighted_vacancy: float
    cumulative_progress_loss: float
    cumulative_starvation_exposure: float
    cumulative_switch_time: float
    cumulative_energy_consumed: float
    cumulative_normalized_distance: float
    mean_load_gap: float | None
    task_count: int
    starved_task_count: int
    task_starvation_rate: float | None
    resource_conflicts: int
    stale_command_resurrections: int
    energy_safety_violations: int
    inference_latency_mean_ms: float | None
    inference_latency_p95_ms: float | None
    inference_latency_p99_ms: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionMetricAccumulator:
    algorithm_id: str
    tape_id: str
    uav_count: int
    _accepted_decisions: int = field(default=0, init=False, repr=False)
    _urgent_tasks: int = field(default=0, init=False, repr=False)
    _urgent_misses: int = field(default=0, init=False, repr=False)
    _p0_events: int = field(default=0, init=False, repr=False)
    _p0_handled: int = field(default=0, init=False, repr=False)
    _displaced_tasks: int = field(default=0, init=False, repr=False)
    _resumed_tasks: int = field(default=0, init=False, repr=False)
    _tasks: int = field(default=0, init=False, repr=False)
    _starved_tasks: int = field(default=0, init=False, repr=False)
    _vacancy: float = field(default=0.0, init=False, repr=False)
    _progress_loss: float = field(default=0.0, init=False, repr=False)
    _starvation: float = field(default=0.0, init=False, repr=False)
    _switch_time: float = field(default=0.0, init=False, repr=False)
    _energy: float = field(default=0.0, init=False, repr=False)
    _distance: float = field(default=0.0, init=False, repr=False)
    _resource_conflicts: int = field(default=0, init=False, repr=False)
    _stale_resurrections: int = field(default=0, init=False, repr=False)
    _energy_violations: int = field(default=0, init=False, repr=False)
    _recovery_latencies: list[float] = field(default_factory=list, init=False, repr=False)
    _preemption_latencies: list[float] = field(default_factory=list, init=False, repr=False)
    _inference_latencies: list[float] = field(default_factory=list, init=False, repr=False)
    _load_gaps: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.algorithm_id or not self.tape_id:
            raise ValueError("algorithm_id and tape_id are required")
        if isinstance(self.uav_count, bool) or int(self.uav_count) != self.uav_count or self.uav_count <= 0:
            raise ValueError("uav_count must be a positive integer")
        self.uav_count = int(self.uav_count)

    def record_transition(
        self,
        signals: TransitionSignals,
        *,
        inference_latency_ms: float | None = None,
        preemption_response_latency: float | None = None,
    ) -> None:
        if not isinstance(signals, TransitionSignals):
            raise TypeError("signals must be TransitionSignals")
        self._accepted_decisions += 1
        self._vacancy += signals.weighted_vacancy_time
        self._progress_loss += signals.progress_loss
        self._starvation += signals.starvation_exposure
        self._switch_time += signals.switch_time
        self._energy += signals.energy_consumed
        self._distance += signals.normalized_distance
        self._load_gaps.append(signals.load_gap)
        self._resource_conflicts += signals.resource_conflicts
        self._stale_resurrections += signals.stale_command_resurrections
        self._energy_violations += signals.energy_safety_violations
        if inference_latency_ms is not None:
            self._inference_latencies.append(
                _finite_non_negative("inference_latency_ms", inference_latency_ms)
            )
        if preemption_response_latency is not None:
            self._preemption_latencies.append(
                _finite_non_negative("preemption_response_latency", preemption_response_latency)
            )

    def record_event(self, *, urgent: bool = False, deadline_missed: bool = False,
                     p0: bool = False, p0_handled: bool = False) -> None:
        if deadline_missed and not urgent:
            raise ValueError("deadline_missed requires urgent=True")
        if p0_handled and not p0:
            raise ValueError("p0_handled requires p0=True")
        self._urgent_tasks += int(bool(urgent))
        self._urgent_misses += int(bool(deadline_missed))
        self._p0_events += int(bool(p0))
        self._p0_handled += int(bool(p0_handled))

    def record_displacement(self, *, resumed: bool, recovery_latency: float | None = None) -> None:
        if resumed and recovery_latency is None:
            raise ValueError("resumed displacement requires recovery_latency")
        if not resumed and recovery_latency is not None:
            raise ValueError("censored displacement cannot have recovery_latency")
        self._displaced_tasks += 1
        if resumed:
            self._resumed_tasks += 1
            self._recovery_latencies.append(
                _finite_non_negative("recovery_latency", float(recovery_latency))
            )

    def record_task_outcome(self, *, starved: bool) -> None:
        self._tasks += 1
        self._starved_tasks += int(bool(starved))

    def finalize(self) -> ExecutionEpisodeMetrics:
        urgent_rate = self._urgent_misses / self._urgent_tasks if self._urgent_tasks else None
        p0_rate = self._p0_handled / self._p0_events if self._p0_events else None
        recovery_rate = self._resumed_tasks / self._displaced_tasks if self._displaced_tasks else None
        starvation_rate = self._starved_tasks / self._tasks if self._tasks else None
        return ExecutionEpisodeMetrics(
            schema_id=METRICS_SCHEMA_ID,
            algorithm_id=self.algorithm_id,
            tape_id=self.tape_id,
            uav_count=self.uav_count,
            accepted_decision_count=self._accepted_decisions,
            urgent_task_count=self._urgent_tasks,
            urgent_deadline_miss_count=self._urgent_misses,
            urgent_deadline_miss_rate=urgent_rate,
            p0_event_count=self._p0_events,
            p0_handled_count=self._p0_handled,
            p0_handling_rate=p0_rate,
            displaced_task_count=self._displaced_tasks,
            resumed_task_count=self._resumed_tasks,
            right_censored_recovery_count=self._displaced_tasks - self._resumed_tasks,
            normal_task_recovery_rate=recovery_rate,
            mean_recovery_latency=_mean(self._recovery_latencies),
            mean_preemption_response_latency=_mean(self._preemption_latencies),
            cumulative_weighted_vacancy=float(self._vacancy),
            cumulative_progress_loss=float(self._progress_loss),
            cumulative_starvation_exposure=float(self._starvation),
            cumulative_switch_time=float(self._switch_time),
            cumulative_energy_consumed=float(self._energy),
            cumulative_normalized_distance=float(self._distance),
            mean_load_gap=_mean(self._load_gaps),
            task_count=self._tasks,
            starved_task_count=self._starved_tasks,
            task_starvation_rate=starvation_rate,
            resource_conflicts=self._resource_conflicts,
            stale_command_resurrections=self._stale_resurrections,
            energy_safety_violations=self._energy_violations,
            inference_latency_mean_ms=_mean(self._inference_latencies),
            inference_latency_p95_ms=_percentile(self._inference_latencies, 95.0),
            inference_latency_p99_ms=_percentile(self._inference_latencies, 99.0),
        )


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    status: str
    violations: tuple[str, ...]
    urgent_deadline_miss_relative_improvement: float | None
    cumulative_vacancy_relative_improvement: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _relative_improvement(candidate: float, baseline: float) -> float | None:
    if baseline <= 0.0:
        return None
    return float((baseline - candidate) / baseline)


def evaluate_acceptance(
    candidate: ExecutionEpisodeMetrics,
    baseline: ExecutionEpisodeMetrics,
) -> AcceptanceResult:
    """Evaluate the frozen V1 minimum outcome gates without ranking methods."""

    if candidate.uav_count != baseline.uav_count:
        raise ValueError("candidate and baseline must use the same UAV scale")
    violations: list[str] = []
    for name in (
        "resource_conflicts",
        "stale_command_resurrections",
        "energy_safety_violations",
    ):
        if getattr(candidate, name) != 0:
            violations.append(f"{name}_must_equal_zero")
    if candidate.p0_handling_rate != 1.0:
        violations.append("p0_handling_rate_must_equal_1")
    if candidate.normal_task_recovery_rate is None or candidate.normal_task_recovery_rate < 0.95:
        violations.append("normal_task_recovery_rate_below_0.95")
    if candidate.urgent_deadline_miss_rate is None or baseline.urgent_deadline_miss_rate is None:
        urgent_improvement = None
        violations.append("urgent_deadline_miss_rate_missing")
    else:
        urgent_improvement = _relative_improvement(
            candidate.urgent_deadline_miss_rate,
            baseline.urgent_deadline_miss_rate,
        )
        if urgent_improvement is None:
            violations.append("urgent_deadline_miss_baseline_zero")
        elif urgent_improvement < 0.10:
            violations.append("urgent_deadline_miss_improvement_below_0.10")
    vacancy_improvement = _relative_improvement(
        candidate.cumulative_weighted_vacancy,
        baseline.cumulative_weighted_vacancy,
    )
    if vacancy_improvement is None:
        violations.append("cumulative_vacancy_baseline_zero")
    elif vacancy_improvement < 0.10:
        violations.append("cumulative_vacancy_improvement_below_0.10")
    return AcceptanceResult(
        status="PASS" if not violations else "FAIL",
        violations=tuple(violations),
        urgent_deadline_miss_relative_improvement=urgent_improvement,
        cumulative_vacancy_relative_improvement=vacancy_improvement,
    )


__all__ = [
    "AcceptanceResult",
    "ExecutionEpisodeMetrics",
    "ExecutionMetricAccumulator",
    "METRICS_SCHEMA_ID",
    "evaluate_acceptance",
]

