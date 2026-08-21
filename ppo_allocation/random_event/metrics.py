"""Metrics and paired statistics for the random-event GPPO experiment.

The module intentionally depends only on the Python standard library and
NumPy.  It accepts plain mappings as well as the dataclasses below, which
makes the saved logs usable without importing the simulator or PyTorch.

Conventions
-----------
* Latencies are milliseconds.
* ``normalized_distance``, ``load_gap`` and diagnostic values are decision
  means inside an event, then event means inside an episode.
* Counts and return are summed.
* ``cumulative_uncovered_time`` integrates weighted uncovered demand over
  physical ``delta_time`` and is therefore not a decision-step count.
* A failed event keeps ``recovery_delay=None``.  Aggregates report both the
  number of observed recovery delays and final infeasibility, so censoring is
  explicit rather than silently replacing failure with zero delay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1


def _finite(value: Any) -> float | None:
    """Return a finite float, or ``None`` for missing/non-finite values."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return float(np.mean(clean)) if clean else None


def _sum(values: Iterable[Any]) -> float:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return float(np.sum(clean)) if clean else 0.0


def _variance(values: Iterable[Any]) -> float | None:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return float(np.var(clean, ddof=0)) if clean else None


def _percentile(values: Iterable[Any], percentile: float) -> float | None:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return float(np.percentile(clean, percentile)) if clean else None


def _flatten_numeric(value: Any) -> list[float]:
    """Flatten scalar/list/mapping/NumPy diagnostic values into finite floats."""

    if isinstance(value, Mapping):
        result: list[float] = []
        for key in sorted(value, key=str):
            result.extend(_flatten_numeric(value[key]))
        return result
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_numeric(item))
        return result
    number = _finite(value)
    return [] if number is None else [number]


def _record_dict(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        return asdict(record)
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError("metric records must be dataclasses or mappings")


@dataclass(frozen=True, slots=True)
class EventMetrics:
    """Final metrics for one exogenous event and its recovery decisions."""

    tape_id: str
    episode_id: str
    event_id: str
    event_type: str
    event_index: int
    success: bool
    legal_coverage_rate: float | None
    weighted_uncovered: float | None
    recovery_delay: float | None
    cumulative_uncovered_time: float
    normalized_distance: float | None
    load_gap: float | None
    switch_count: int
    repair_count: int
    temporary_infeasible: bool
    final_infeasible: bool
    event_return: float
    avg_reward: float | None
    decision_count: int
    inference_latency_ms: float | None
    inference_latency_p95_ms: float | None
    event_to_action_latency_ms: float | None
    communication_trigger_count: int
    communication_bytes: int
    communication_opportunities: int
    communication_suppressed: int
    communication_suppression_rate: float | None
    pre_mask_invalid_probability: float | None
    mask_rate: float | None
    gate_mean: float | None
    gate_variance: float | None
    value_error: float | None
    value_squared_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Aggregate of all event recoveries in one episode/event tape."""

    tape_id: str
    episode_id: str
    algorithm: str
    event_count: int
    event_success_count: int
    event_success_rate: float | None
    legal_coverage_rate: float | None
    weighted_uncovered: float | None
    recovery_delay: float | None
    recovery_delay_observed_count: int
    cumulative_uncovered_time: float
    normalized_distance: float | None
    load_gap: float | None
    switch_count: int
    repair_count: int
    temporary_infeasible_count: int
    temporary_infeasible_rate: float | None
    final_infeasible_count: int
    final_infeasible_rate: float | None
    episode_return: float
    avg_reward: float | None
    decision_count: int
    inference_latency_ms: float | None
    event_to_action_latency_ms: float | None
    communication_trigger_count: int
    communication_bytes: int
    communication_opportunities: int
    communication_suppressed: int
    communication_suppression_rate: float | None
    pre_mask_invalid_probability: float | None
    mask_rate: float | None
    gate_mean: float | None
    gate_variance: float | None
    value_error: float | None
    value_squared_error: float | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Canonical Phase J name; the underlying accumulator integrates the
        # protocol weighted-uncovered quantity over physical time.
        value["cumulative_weighted_vacancy"] = value["cumulative_uncovered_time"]
        return value


@dataclass(slots=True)
class EventMetricAccumulator:
    """Incrementally collect decision metrics for one active event.

    ``record_decision`` is deliberately keyword-only.  Callers can omit any
    unavailable diagnostic without inventing a numeric zero.
    """

    tape_id: str
    episode_id: str
    event_id: str
    event_type: str
    event_index: int
    _rewards: list[float] = field(default_factory=list, init=False, repr=False)
    _coverage: list[float] = field(default_factory=list, init=False, repr=False)
    _uncovered: list[float] = field(default_factory=list, init=False, repr=False)
    _distance: list[float] = field(default_factory=list, init=False, repr=False)
    _load_gap: list[float] = field(default_factory=list, init=False, repr=False)
    _inference_ms: list[float] = field(default_factory=list, init=False, repr=False)
    _event_to_action_ms: list[float] = field(default_factory=list, init=False, repr=False)
    _invalid_probability: list[float] = field(default_factory=list, init=False, repr=False)
    _mask_rate: list[float] = field(default_factory=list, init=False, repr=False)
    _gate_values: list[float] = field(default_factory=list, init=False, repr=False)
    _value_errors: list[float] = field(default_factory=list, init=False, repr=False)
    _switch_count: int = field(default=0, init=False, repr=False)
    _repair_count: int = field(default=0, init=False, repr=False)
    _temporary_infeasible: bool = field(default=False, init=False, repr=False)
    _final_infeasible: bool = field(default=False, init=False, repr=False)
    _cumulative_uncovered_time: float = field(default=0.0, init=False, repr=False)
    _communication_triggers: int = field(default=0, init=False, repr=False)
    _communication_bytes: int = field(default=0, init=False, repr=False)
    _communication_opportunities: int = field(default=0, init=False, repr=False)
    _communication_suppressed: int = field(default=0, init=False, repr=False)
    _decision_count: int = field(default=0, init=False, repr=False)

    @staticmethod
    def _append(target: list[float], value: Any) -> None:
        number = _finite(value)
        if number is not None:
            target.append(number)

    def record_decision(
        self,
        *,
        reward: float | None = None,
        legal_coverage_rate: float | None = None,
        weighted_uncovered: float | None = None,
        delta_time: float = 1.0,
        normalized_distance: float | None = None,
        load_gap: float | None = None,
        switch_count: int = 0,
        repair_count: int = 0,
        temporary_infeasible: bool = False,
        final_infeasible: bool = False,
        inference_latency_ms: float | None = None,
        event_to_action_latency_ms: float | None = None,
        communication_triggered: bool = False,
        communication_bytes: int = 0,
        communication_opportunities: int = 0,
        communication_suppressed: int | None = None,
        pre_mask_invalid_probability: float | None = None,
        mask_rate: float | None = None,
        gate_values: Any = None,
        predicted_value: float | None = None,
        value_target: float | None = None,
        value_error: float | None = None,
    ) -> None:
        self._decision_count += 1
        self._append(self._rewards, reward)
        self._append(self._coverage, legal_coverage_rate)
        self._append(self._uncovered, weighted_uncovered)
        self._append(self._distance, normalized_distance)
        self._append(self._load_gap, load_gap)
        self._append(self._inference_ms, inference_latency_ms)
        self._append(self._event_to_action_ms, event_to_action_latency_ms)
        self._append(self._invalid_probability, pre_mask_invalid_probability)
        self._append(self._mask_rate, mask_rate)
        self._gate_values.extend(_flatten_numeric(gate_values))

        explicit_error = _finite(value_error)
        prediction = _finite(predicted_value)
        target = _finite(value_target)
        if explicit_error is not None:
            self._value_errors.append(abs(explicit_error))
        elif prediction is not None and target is not None:
            self._value_errors.append(abs(prediction - target))

        uncovered = _finite(weighted_uncovered)
        duration = _finite(delta_time)
        if duration is not None and duration < 0:
            raise ValueError("delta_time must be non-negative")
        if uncovered is not None and duration is not None:
            self._cumulative_uncovered_time += uncovered * duration

        self._switch_count += max(0, int(switch_count))
        self._repair_count += max(0, int(repair_count))
        self._temporary_infeasible |= bool(temporary_infeasible)
        self._final_infeasible |= bool(final_infeasible)
        self._communication_triggers += int(bool(communication_triggered))
        self._communication_bytes += max(0, int(communication_bytes))
        self._communication_opportunities += max(0, int(communication_opportunities))
        if communication_suppressed is None:
            self._communication_suppressed += max(
                0, int(communication_opportunities) - int(bool(communication_triggered))
            )
        else:
            self._communication_suppressed += max(0, int(communication_suppressed))

    def finalize(
        self,
        *,
        success: bool,
        recovery_delay: float | None = None,
        final_infeasible: bool | None = None,
        final_legal_coverage_rate: float | None = None,
        final_weighted_uncovered: float | None = None,
    ) -> EventMetrics:
        """Freeze the active event into a JSON-ready record."""

        if recovery_delay is not None and _finite(recovery_delay) is None:
            raise ValueError("recovery_delay must be finite or None")
        if recovery_delay is not None and float(recovery_delay) < 0:
            raise ValueError("recovery_delay must be non-negative")
        coverage = _finite(final_legal_coverage_rate)
        if coverage is None:
            coverage = self._coverage[-1] if self._coverage else None
        uncovered = _finite(final_weighted_uncovered)
        if uncovered is None:
            uncovered = self._uncovered[-1] if self._uncovered else None
        opportunities = self._communication_opportunities
        suppressed = min(self._communication_suppressed, opportunities)
        suppression_rate = suppressed / opportunities if opportunities else None
        final_flag = self._final_infeasible if final_infeasible is None else bool(final_infeasible)
        return EventMetrics(
            tape_id=str(self.tape_id),
            episode_id=str(self.episode_id),
            event_id=str(self.event_id),
            event_type=str(self.event_type),
            event_index=int(self.event_index),
            success=bool(success),
            legal_coverage_rate=coverage,
            weighted_uncovered=uncovered,
            recovery_delay=_finite(recovery_delay),
            cumulative_uncovered_time=float(self._cumulative_uncovered_time),
            normalized_distance=_mean(self._distance),
            load_gap=_mean(self._load_gap),
            switch_count=self._switch_count,
            repair_count=self._repair_count,
            temporary_infeasible=self._temporary_infeasible,
            final_infeasible=final_flag,
            event_return=_sum(self._rewards),
            avg_reward=_mean(self._rewards),
            decision_count=self._decision_count,
            inference_latency_ms=_mean(self._inference_ms),
            inference_latency_p95_ms=_percentile(self._inference_ms, 95),
            event_to_action_latency_ms=(self._event_to_action_ms[0] if self._event_to_action_ms else None),
            communication_trigger_count=self._communication_triggers,
            communication_bytes=self._communication_bytes,
            communication_opportunities=opportunities,
            communication_suppressed=suppressed,
            communication_suppression_rate=suppression_rate,
            pre_mask_invalid_probability=_mean(self._invalid_probability),
            mask_rate=_mean(self._mask_rate),
            gate_mean=_mean(self._gate_values),
            gate_variance=_variance(self._gate_values),
            value_error=_mean(self._value_errors),
            value_squared_error=_mean(error * error for error in self._value_errors),
        )


def aggregate_episode(
    event_records: Sequence[EventMetrics | Mapping[str, Any]],
    *,
    algorithm: str,
    tape_id: str | None = None,
    episode_id: str | None = None,
) -> EpisodeMetrics:
    """Aggregate per-event rows without losing failure/censoring counts."""

    rows = [_record_dict(record) for record in event_records]
    if tape_id is None:
        tape_id = str(rows[0].get("tape_id", "")) if rows else ""
    if episode_id is None:
        episode_id = str(rows[0].get("episode_id", "")) if rows else ""
    if rows:
        if any(str(row.get("tape_id", tape_id)) != str(tape_id) for row in rows):
            raise ValueError("all event records must belong to one tape")
        if any(str(row.get("episode_id", episode_id)) != str(episode_id) for row in rows):
            raise ValueError("all event records must belong to one episode")

    count = len(rows)
    decisions = int(_sum(row.get("decision_count", 0) for row in rows))
    event_return = _sum(row.get("event_return") for row in rows)
    opportunities = int(_sum(row.get("communication_opportunities", 0) for row in rows))
    suppressed = min(int(_sum(row.get("communication_suppressed", 0) for row in rows)), opportunities)
    delays = [row.get("recovery_delay") for row in rows if _finite(row.get("recovery_delay")) is not None]
    value_errors = [row.get("value_error") for row in rows]
    return EpisodeMetrics(
        tape_id=str(tape_id),
        episode_id=str(episode_id),
        algorithm=str(algorithm),
        event_count=count,
        event_success_count=sum(bool(row.get("success", False)) for row in rows),
        event_success_rate=(_mean(bool(row.get("success", False)) for row in rows) if rows else None),
        legal_coverage_rate=_mean(row.get("legal_coverage_rate") for row in rows),
        weighted_uncovered=_mean(row.get("weighted_uncovered") for row in rows),
        recovery_delay=_mean(delays),
        recovery_delay_observed_count=len(delays),
        cumulative_uncovered_time=_sum(row.get("cumulative_uncovered_time") for row in rows),
        normalized_distance=_mean(row.get("normalized_distance") for row in rows),
        load_gap=_mean(row.get("load_gap") for row in rows),
        switch_count=int(_sum(row.get("switch_count") for row in rows)),
        repair_count=int(_sum(row.get("repair_count") for row in rows)),
        temporary_infeasible_count=sum(bool(row.get("temporary_infeasible", False)) for row in rows),
        temporary_infeasible_rate=(_mean(bool(row.get("temporary_infeasible", False)) for row in rows) if rows else None),
        final_infeasible_count=sum(bool(row.get("final_infeasible", False)) for row in rows),
        final_infeasible_rate=(_mean(bool(row.get("final_infeasible", False)) for row in rows) if rows else None),
        episode_return=event_return,
        avg_reward=(event_return / decisions if decisions else None),
        decision_count=decisions,
        inference_latency_ms=_weighted_event_mean(rows, "inference_latency_ms", "decision_count"),
        event_to_action_latency_ms=_mean(row.get("event_to_action_latency_ms") for row in rows),
        communication_trigger_count=int(_sum(row.get("communication_trigger_count") for row in rows)),
        communication_bytes=int(_sum(row.get("communication_bytes") for row in rows)),
        communication_opportunities=opportunities,
        communication_suppressed=suppressed,
        communication_suppression_rate=(suppressed / opportunities if opportunities else None),
        pre_mask_invalid_probability=_weighted_event_mean(rows, "pre_mask_invalid_probability", "decision_count"),
        mask_rate=_weighted_event_mean(rows, "mask_rate", "decision_count"),
        gate_mean=_weighted_event_mean(rows, "gate_mean", "decision_count"),
        gate_variance=_pooled_variance(rows, "gate_mean", "gate_variance", "decision_count"),
        value_error=_weighted_event_mean(rows, "value_error", "decision_count"),
        value_squared_error=_weighted_event_mean(rows, "value_squared_error", "decision_count"),
    )


def _weighted_event_mean(rows: Sequence[Mapping[str, Any]], value_key: str, weight_key: str) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        value = _finite(row.get(value_key))
        weight = _finite(row.get(weight_key))
        if value is not None and weight is not None and weight > 0:
            pairs.append((value, weight))
    if not pairs:
        return None
    return float(sum(value * weight for value, weight in pairs) / sum(weight for _, weight in pairs))


def _pooled_variance(
    rows: Sequence[Mapping[str, Any]], mean_key: str, variance_key: str, weight_key: str
) -> float | None:
    groups: list[tuple[float, float, float]] = []
    for row in rows:
        mean = _finite(row.get(mean_key))
        variance = _finite(row.get(variance_key))
        weight = _finite(row.get(weight_key))
        if mean is not None and variance is not None and weight is not None and weight > 0:
            groups.append((mean, variance, weight))
    if not groups:
        return None
    total = sum(weight for _, _, weight in groups)
    grand = sum(mean * weight for mean, _, weight in groups) / total
    return float(sum(weight * (variance + (mean - grand) ** 2) for mean, variance, weight in groups) / total)


_ID_FIELDS = {"tape_id", "episode_id", "algorithm"}


def aggregate_tapes(
    episode_records: Sequence[EpisodeMetrics | Mapping[str, Any]],
    *,
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize metrics across tapes with explicit finite sample counts."""

    rows = [_record_dict(record) for record in episode_records]
    if metrics is None:
        names = [item.name for item in fields(EpisodeMetrics) if item.name not in _ID_FIELDS]
    else:
        names = list(metrics)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tape_count": len({str(row.get("tape_id", "")) for row in rows}),
        "episode_count": len(rows),
        "metrics": {},
    }
    for name in names:
        values = [value for row in rows if (value := _finite(row.get(name))) is not None]
        result["metrics"][name] = descriptive_statistics(values)
    return result


def descriptive_statistics(values: Sequence[float]) -> dict[str, Any]:
    """Stable finite-only summary; sample SD is used when n > 1."""

    clean = np.asarray([value for item in values if (value := _finite(item)) is not None], dtype=float)
    if clean.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": int(clean.size),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
        "median": float(np.median(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile CI for the paired mean difference ``a - b``.

    A single resampled index vector is applied to both algorithms, preserving
    tape pairing.  Non-finite pairs are removed jointly.
    """

    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    pairs = [
        (left, right)
        for x, y in zip(a, b)
        if (left := _finite(x)) is not None and (right := _finite(y)) is not None
    ]
    if not pairs:
        return {"n": 0, "confidence": confidence, "lower": None, "upper": None, "seed": seed, "n_resamples": n_resamples}
    differences = np.asarray([left - right for left, right in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    # Chunking avoids allocating n_resamples*n for large formal evaluations.
    bootstrap_means = np.empty(n_resamples, dtype=float)
    chunk_size = max(1, min(n_resamples, 1_000_000 // max(1, differences.size)))
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, differences.size, size=(stop - start, differences.size))
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": int(differences.size),
        "confidence": float(confidence),
        "lower": float(np.quantile(bootstrap_means, alpha)),
        "upper": float(np.quantile(bootstrap_means, 1.0 - alpha)),
        "seed": int(seed),
        "n_resamples": int(n_resamples),
    }


def paired_difference(
    a: Sequence[float],
    b: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired effect report for ``a - b`` including bootstrap CI and Cohen dz."""

    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    pairs = [
        (left, right)
        for x, y in zip(a, b)
        if (left := _finite(x)) is not None and (right := _finite(y)) is not None
    ]
    left = np.asarray([item[0] for item in pairs], dtype=float)
    right = np.asarray([item[1] for item in pairs], dtype=float)
    difference = left - right
    if difference.size == 0:
        dz = None
    elif difference.size == 1:
        dz = None
    else:
        sd = float(np.std(difference, ddof=1))
        dz = (float(np.mean(difference)) / sd) if sd > 0 else None
    non_ties = difference[difference != 0]
    rank_biserial = None if non_ties.size == 0 else float((np.sum(non_ties > 0) - np.sum(non_ties < 0)) / non_ties.size)
    return {
        "definition": "a_minus_b",
        "n": int(difference.size),
        "mean_a": float(np.mean(left)) if left.size else None,
        "mean_b": float(np.mean(right)) if right.size else None,
        "mean_difference": float(np.mean(difference)) if difference.size else None,
        "median_difference": float(np.median(difference)) if difference.size else None,
        "std_difference": float(np.std(difference, ddof=1)) if difference.size > 1 else (0.0 if difference.size else None),
        "cohen_dz": dz,
        "rank_biserial": rank_biserial,
        "wins_a": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
        "wins_b": int(np.sum(difference < 0)),
        "bootstrap_ci": paired_bootstrap_ci(
            left.tolist(), right.tolist(), confidence=confidence, n_resamples=n_resamples, seed=seed
        ),
    }


def paired_metric_report(
    records_a: Sequence[EpisodeMetrics | Mapping[str, Any]],
    records_b: Sequence[EpisodeMetrics | Mapping[str, Any]],
    metric: str,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Join two algorithms by ``tape_id`` and calculate a paired effect."""

    def index(records: Sequence[EpisodeMetrics | Mapping[str, Any]]) -> dict[str, float]:
        result: dict[str, float] = {}
        for record in records:
            row = _record_dict(record)
            tape_id = str(row.get("tape_id", ""))
            if not tape_id:
                raise ValueError("paired records require tape_id")
            if tape_id in result:
                raise ValueError(f"duplicate tape_id: {tape_id}")
            value = _finite(row.get(metric))
            if value is not None:
                result[tape_id] = value
        return result

    a_by_tape, b_by_tape = index(records_a), index(records_b)
    tape_ids = sorted(set(a_by_tape) & set(b_by_tape))
    report = paired_difference(
        [a_by_tape[tape] for tape in tape_ids],
        [b_by_tape[tape] for tape in tape_ids],
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    return {"metric": metric, "paired_tape_ids": tape_ids, **report}


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def stable_json_dumps(value: Any, *, indent: int | None = 2) -> str:
    """Serialize deterministically; NaN/Infinity are converted to JSON null."""

    separators = (",", ":") if indent is None else None
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=separators,
        indent=indent,
        allow_nan=False,
    )


def write_metrics_json(path: str | Path, value: Any, *, indent: int | None = 2) -> Path:
    """Write UTF-8 metric JSON atomically and return its path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(stable_json_dumps(value, indent=indent) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


__all__ = [
    "SCHEMA_VERSION",
    "EpisodeMetrics",
    "EventMetricAccumulator",
    "EventMetrics",
    "aggregate_episode",
    "aggregate_tapes",
    "descriptive_statistics",
    "paired_bootstrap_ci",
    "paired_difference",
    "paired_metric_report",
    "stable_json_dumps",
    "write_metrics_json",
]
