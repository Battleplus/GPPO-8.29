"""Plot the four required random-event GPPO diagnostic figures.

The experiment runner intentionally writes plain JSON.  This module accepts
both a compact summary and one or more raw JSON files and does not depend on
pandas.  Its parser is deliberately tolerant: records may be top-level lists,
JSONL-like envelopes (``events``, ``episodes``, ``history`` or ``timeline``),
or nested below an algorithm/mode name.

Example
-------
python -m ppo_allocation.random_event.plotting \
    --summary results/random_event/summary.json \
    --raw results/random_event/raw.json \
    --output-dir results/random_event/figures
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EVENT_TYPES = ("UAV_DAMAGE", "TARGET_DISCOVERED", "TARGET_DESTROYED", "REGION_VACANCY")
MODE_ORDER = ("single", "sequential", "overlap", "burst", "unseen")
ALGORITHM_KEYS = ("algorithm", "algo", "policy", "method", "variant", "model")
MODE_KEYS = ("mode", "event_mode", "test_mode", "scenario_mode", "bank")


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return float(value) if isinstance(value, bool) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first(record: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Accept newline-delimited JSON as a convenience for long raw traces.
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _iter_records(value: Any, context: Mapping[str, Any] | None = None) -> Iterable[dict[str, Any]]:
    """Yield mapping records while carrying scalar envelope context.

    This does not assume one fixed result schema.  A mapping is yielded once,
    then nested lists/mappings are traversed.  Parent labels such as algorithm
    and mode are inherited by child records when the child omits them.
    """

    inherited = dict(context or {})
    if isinstance(value, Mapping):
        scalars = {
            str(key): item
            for key, item in value.items()
            if item is None or isinstance(item, (str, int, float, bool))
        }
        merged = {**inherited, **scalars, **dict(value)}
        yield merged
        child_context = {**inherited, **scalars}
        for key, item in value.items():
            if isinstance(item, (Mapping, list, tuple)):
                local = dict(child_context)
                # Common layout: {"algorithms": {"GPPO": {...}}} or
                # {"modes": {"burst": [...]}}.
                if str(key) not in {
                    "events", "event_records", "event_metrics", "episodes", "episode_records",
                    "history", "training_history", "timeline", "trace", "steps", "records",
                    "results", "metrics", "summary", "raw",
                }:
                    if "algorithm" not in local and _looks_like_algorithm(str(key)):
                        local["algorithm"] = str(key)
                    elif "mode" not in local and str(key).lower() in MODE_ORDER:
                        local["mode"] = str(key).lower()
                yield from _iter_records(item, local)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_records(item, inherited)


def _looks_like_algorithm(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("ppo", "gppo", "random", "nearest", "load", "greedy", "oracle"))


def _record_algorithm(record: Mapping[str, Any]) -> str:
    value = _first(record, ALGORITHM_KEYS, "unknown")
    return str(value)


def _record_mode(record: Mapping[str, Any]) -> str:
    value = str(_first(record, MODE_KEYS, "all")).lower()
    for mode in MODE_ORDER:
        if mode in value:
            return mode
    return value


def _record_event_type(record: Mapping[str, Any]) -> str | None:
    value = _first(record, ("event_type", "type", "source_event"))
    if value is None and isinstance(record.get("event"), Mapping):
        value = _first(record["event"], ("event_type", "type", "source_event"))
    if value is None:
        return None
    text = str(value).upper()
    return next((kind for kind in EVENT_TYPES if kind in text), text)


def _event_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        event_type = _record_event_type(record)
        if event_type is None:
            continue
        has_metric = any(
            key in record
            for key in ("success", "event_success", "recovered", "recovery_delay", "recovery_time")
        )
        if has_metric:
            row = dict(record)
            row["_event_type"] = event_type
            row["_algorithm"] = _record_algorithm(record)
            row["_mode"] = _record_mode(record)
            result.append(row)
    return result


def _mean_ci(values: Iterable[Any]) -> tuple[float | None, float | None, float | None, int]:
    clean = [number for item in values if (number := _finite(item)) is not None]
    if not clean:
        return None, None, None, 0
    center = mean(clean)
    half = 0.0 if len(clean) < 2 else 1.96 * stdev(clean) / math.sqrt(len(clean))
    return center, center - half, center + half, len(clean)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _empty(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#666")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_event_outcomes(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Figure 1: event type x mode success rate and recovery delay."""

    rows = _event_records(records)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    if not rows:
        _empty(axes[0], "No event-level success records found")
        _empty(axes[1], "No event-level recovery-delay records found")
        fig.suptitle("Event outcomes by type and mode")
        _save(fig, output)
        return

    modes = [mode for mode in MODE_ORDER if any(row["_mode"] == mode for row in rows)]
    modes += sorted({row["_mode"] for row in rows} - set(modes))
    event_types = [kind for kind in EVENT_TYPES if any(row["_event_type"] == kind for row in rows)]
    event_types += sorted({row["_event_type"] for row in rows} - set(event_types))
    groups = [(kind, mode) for kind in event_types for mode in modes]
    algorithms = sorted({row["_algorithm"] for row in rows})
    x = np.arange(len(groups), dtype=float)
    width = min(0.82 / max(1, len(algorithms)), 0.22)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(algorithms))))

    # Algorithm is the bar/line identity.  Event type and mode form the x
    # groups; this prevents averaging competing policies into one bar.
    for index, algorithm in enumerate(algorithms):
        success_means, delay_means, delay_errors = [], [], []
        for kind, mode in groups:
            subset = [
                row for row in rows
                if row["_algorithm"] == algorithm and row["_mode"] == mode and row["_event_type"] == kind
            ]
            successes = [
                _first(row, ("success", "event_success", "recovered")) for row in subset
            ]
            success_means.append(_mean_ci(successes)[0])
            center, lower, upper, _ = _mean_ci(
                _first(row, ("recovery_delay", "recovery_time", "event_to_recovery_delay"))
                for row in subset
            )
            delay_means.append(center)
            delay_errors.append(None if center is None else max(0.0, (upper or center) - center))
        offset = (index - (len(algorithms) - 1) / 2) * width
        axes[0].bar(x + offset, [np.nan if v is None else v for v in success_means], width,
                    label=algorithm, color=colors[index], edgecolor="white")
        axes[1].bar(
            x + offset,
            [np.nan if v is None else v for v in delay_means],
            width,
            yerr=[0.0 if v is None else v for v in delay_errors],
            capsize=2,
            color=colors[index],
            edgecolor="white",
        )

    axes[0].set_ylabel("Event success rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(title="Algorithm", ncol=min(4, len(algorithms)), frameon=False, fontsize=8)
    axes[1].set_ylabel("Recovery delay (95% CI)")
    axes[1].set_xticks(
        x,
        [f"{kind.replace('_', ' ')}\n{mode}" for kind, mode in groups],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    axes[1].set_xlabel("Event type × test mode")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Random-event recovery by event type and test mode")
    fig.tight_layout()
    _save(fig, output)


def _summary_algorithm_stats(records: Sequence[Mapping[str, Any]], metric: str) -> dict[str, tuple[float, float, float, int]]:
    """Read pre-aggregated {metrics: {name: {mean,...}}} records when present."""

    found: dict[str, tuple[float, float, float, int]] = {}
    aliases = (metric, "episode_return", "event_success_rate", "legal_coverage_rate", "recovery_delay")
    for record in records:
        algorithm = _record_algorithm(record)
        if algorithm == "unknown":
            continue
        metrics = record.get("metrics")
        candidates = metrics if isinstance(metrics, Mapping) else record
        for name in aliases:
            stat = candidates.get(name) if isinstance(candidates, Mapping) else None
            if isinstance(stat, Mapping):
                center = _finite(_first(stat, ("mean", "estimate", "value")))
                if center is None:
                    continue
                lower = _finite(_first(stat, ("ci_lower", "lower", "lower_95", "95ci_lower"), center))
                upper = _finite(_first(stat, ("ci_upper", "upper", "upper_95", "95ci_upper"), center))
                if lower == center and upper == center:
                    std = _finite(stat.get("std"))
                    n = int(_finite(stat.get("n")) or 0)
                    half = 1.96 * std / math.sqrt(n) if std is not None and n > 1 else 0.0
                    lower, upper = center - half, center + half
                found.setdefault(algorithm, (center, lower or center, upper or center, int(_finite(stat.get("n")) or 0)))
                break
    return found


def plot_algorithm_performance(records: Sequence[Mapping[str, Any]], output: Path, metric: str) -> None:
    """Figure 2: one comparable algorithm metric with 95% confidence intervals."""

    by_algorithm: dict[str, list[float]] = defaultdict(list)
    for record in records:
        algorithm = _record_algorithm(record)
        value = _finite(_first(record, (metric, "episode_return", "event_success_rate")))
        # Avoid interpreting one event as an episode performance row when an
        # explicit event_id is present, unless no episode identifier exists.
        if algorithm != "unknown" and value is not None:
            by_algorithm[algorithm].append(value)
    stats = {name: _mean_ci(values) for name, values in by_algorithm.items() if values}
    if not stats:
        stats = _summary_algorithm_stats(records, metric)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if not stats:
        _empty(ax, f"No algorithm-level metric found ({metric})")
        ax.set_title("Algorithm performance with 95% confidence intervals")
        _save(fig, output)
        return
    names = sorted(stats, key=lambda name: ("gppo" not in name.lower(), name.lower()))
    centers = [stats[name][0] for name in names]
    errors = [
        [max(0.0, stats[name][0] - stats[name][1]) for name in names],
        [max(0.0, stats[name][2] - stats[name][0]) for name in names],
    ]
    positions = np.arange(len(names))
    colors = ["#3b82f6" if "adaptive" in name.lower() else "#94a3b8" for name in names]
    ax.bar(positions, centers, yerr=errors, capsize=4, color=colors, edgecolor="white")
    ax.set_xticks(positions, names, rotation=24, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Algorithm performance: {metric} (mean and 95% CI)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, output)


def _training_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = {"update", "total_steps", "policy_loss", "value_loss", "episode_return", "rollout_return"}
    result = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        if not keys.intersection(record):
            continue
        step = _finite(_first(record, ("total_steps", "update", "epoch", "step")))
        if step is None:
            continue
        signature = (_record_algorithm(record), step, _first(record, ("policy_loss", "loss")))
        if signature in seen:
            continue
        seen.add(signature)
        result.append(dict(record))
    return result


def plot_training_diagnostics(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Figure 3: loss, return, invalid mass and Adaptive Gate history."""

    rows = _training_records(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    definitions = (
        ("Policy loss", ("policy_loss", "loss", "total_loss")),
        ("Rollout return", ("episode_return_mean", "episode_return", "rollout_return", "mean_return", "return")),
        ("Pre-mask invalid probability", ("pre_mask_invalid_probability", "rollout_pre_mask_invalid_probability", "invalid_mass")),
        ("Gate mean (NoGate is fixed at 1)", ("gate_mean", "rollout_gate_mean")),
    )
    for ax, (title, aliases) in zip(axes.reshape(-1), definitions):
        drawn = False
        for algorithm in sorted({_record_algorithm(row) for row in rows}):
            subset = [row for row in rows if _record_algorithm(row) == algorithm]
            subset.sort(key=lambda row: _finite(_first(row, ("total_steps", "update", "epoch", "step"))) or 0)
            values_by_step: dict[float, list[float]] = defaultdict(list)
            for row in subset:
                value = _finite(_first(row, aliases))
                if value is None and title.startswith("Gate") and isinstance(row.get("gate_means"), Mapping):
                    gate_values = [_finite(item) for item in row["gate_means"].values()]
                    gate_values = [item for item in gate_values if item is not None]
                    value = mean(gate_values) if gate_values else None
                if value is not None:
                    step = _finite(_first(row, ("total_steps", "update", "epoch", "step")))
                    values_by_step[step if step is not None else float(len(values_by_step))].append(value)
            x = sorted(values_by_step)
            y = [mean(values_by_step[step]) for step in x]
            if y:
                ax.plot(x, y, linewidth=1.8, marker="o", markersize=3, label=algorithm)
                if any(len(values_by_step[step]) > 1 for step in x):
                    lower = [min(values_by_step[step]) for step in x]
                    upper = [max(values_by_step[step]) for step in x]
                    ax.fill_between(x, lower, upper, alpha=0.12)
                drawn = True
        if not drawn:
            _empty(ax, f"No {title.lower()} records")
        else:
            ax.set_xlabel("Training step / update")
            ax.set_ylabel(title)
            ax.grid(alpha=0.25)
            if len({_record_algorithm(row) for row in rows}) > 1:
                ax.legend(frameon=False, fontsize=8)
        ax.set_title(title)
    fig.suptitle("PPO training and mechanism diagnostics")
    fig.tight_layout()
    _save(fig, output)


def _timeline_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result, seen = [], set()
    for record in records:
        if not any(key in record for key in (
            "occurred_at", "observed_at", "current_time", "simulation_time_after",
            "graph_version", "graph_version_after", "pending_regions", "pending_regions_after",
            "event_queue", "active_events_before",
        )):
            continue
        if _record_event_type(record) is None and not any(key in record for key in (
            "graph_version", "graph_version_after", "pending_regions", "pending_regions_after",
            "event_queue", "active_events_before",
        )):
            continue
        time = _finite(_first(record, (
            "simulation_time_after", "current_time", "observed_at", "occurred_at", "time", "step"
        ), 0.0)) or 0.0
        signature = (
            time,
            record.get("event_id"),
            _first(record, ("graph_version_after", "graph_version")),
            str(_first(record, ("pending_regions_after", "pending_regions"))),
        )
        if signature not in seen:
            seen.add(signature)
            result.append(dict(record))
    return sorted(result, key=lambda row: _finite(_first(row, (
        "simulation_time_after", "current_time", "observed_at", "occurred_at", "time", "step"
    ), 0.0)) or 0.0)


def _count(value: Any) -> float:
    if isinstance(value, Mapping):
        return float(len(value))
    if isinstance(value, (list, tuple, set)):
        return float(len(value))
    number = _finite(value)
    return number or 0.0


def plot_event_timeline(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Figure 4: event occurrences plus queue, pending and graph-version state."""

    rows = _timeline_records(records)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    if not rows:
        _empty(top, "No event timeline records found")
        _empty(bottom, "Expected time, event_queue, pending_regions and graph_version")
        fig.suptitle("Event timeline and graph-version trace")
        _save(fig, output)
        return
    times = np.asarray([
        _finite(_first(row, (
            "simulation_time_after", "current_time", "observed_at", "occurred_at", "time", "step"
        ), 0.0)) or 0.0
        for row in rows
    ])
    event_rows = [(time, _record_event_type(row)) for time, row in zip(times, rows) if _record_event_type(row)]
    type_to_y = {kind: index for index, kind in enumerate(EVENT_TYPES)}
    for time, kind in event_rows:
        y = type_to_y.get(kind, len(type_to_y))
        top.scatter([time], [y], s=55, zorder=3)
        top.axvline(time, color="#cbd5e1", linewidth=0.7, alpha=0.6)
    top.set_yticks(range(len(EVENT_TYPES)), [kind.replace("_", " ") for kind in EVENT_TYPES])
    top.set_ylabel("Event")
    top.grid(axis="x", alpha=0.2)

    pending = [
        _count(_first(row, ("pending_regions_after", "pending_regions", "pending_count"), 0))
        for row in rows
    ]
    queue = [
        _count(_first(row, ("event_queue", "active_events_before", "queue_length", "queue_size"), 0))
        for row in rows
    ]
    versions = [_finite(_first(row, ("graph_version_after", "graph_version"))) for row in rows]
    bottom.step(times, pending, where="post", label="pending regions", linewidth=2)
    bottom.step(times, queue, where="post", label="event queue", linewidth=2)
    if any(value is not None for value in versions):
        version_axis = bottom.twinx()
        version_axis.step(times, [np.nan if value is None else value for value in versions], where="post",
                          color="#7c3aed", linestyle="--", label="graph version")
        version_axis.set_ylabel("Graph version", color="#7c3aed")
        lines, labels = bottom.get_legend_handles_labels()
        extra_lines, extra_labels = version_axis.get_legend_handles_labels()
        bottom.legend(lines + extra_lines, labels + extra_labels, frameon=False, ncol=3)
    else:
        bottom.legend(frameon=False)
    bottom.set_xlabel("Physical time / decision time")
    bottom.set_ylabel("Queue / pending count")
    bottom.grid(alpha=0.25)
    fig.suptitle("Event arrival, recovery queue and graph-version evolution")
    fig.tight_layout()
    _save(fig, output)


def generate_all(summary_paths: Sequence[Path], raw_paths: Sequence[Path], output_dir: Path, metric: str) -> list[Path]:
    values = [_load_json(path) for path in [*summary_paths, *raw_paths]]
    records = [record for value in values for record in _iter_records(value)]
    destinations = [
        output_dir / "01_event_type_mode_recovery.png",
        output_dir / "02_algorithm_performance_95ci.png",
        output_dir / "03_training_diagnostics.png",
        output_dir / "04_event_timeline_graph_version.png",
    ]
    plot_event_outcomes(records, destinations[0])
    plot_algorithm_performance(records, destinations[1], metric)
    plot_training_diagnostics(records, destinations[2])
    plot_event_timeline(records, destinations[3])
    return destinations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", default=[], type=Path,
                        help="Experiment summary JSON; repeat for multiple files.")
    parser.add_argument("--raw", action="append", default=[], type=Path,
                        help="Raw event/episode/training/timeline JSON or JSONL; repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for four PNG files.")
    parser.add_argument("--metric", default="episode_return",
                        help="Algorithm metric for figure 2 (default: episode_return).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [*args.summary, *args.raw]
    if not inputs:
        raise SystemExit("at least one --summary or --raw JSON input is required")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit("input file(s) not found: " + ", ".join(missing))
    outputs = generate_all(args.summary, args.raw, args.output_dir, args.metric)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "generate_all",
    "plot_algorithm_performance",
    "plot_event_outcomes",
    "plot_event_timeline",
    "plot_training_diagnostics",
]
