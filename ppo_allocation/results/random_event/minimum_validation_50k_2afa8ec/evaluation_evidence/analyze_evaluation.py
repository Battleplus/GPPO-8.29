"""Audit and analyze the frozen two-model minimum-validation Test results.

This script is evidence-only.  It reads the immutable Phase J Freeze/Test
artifacts and writes tabular, machine-readable, and HTML reports beside itself.
It never loads a 25k checkpoint and never invokes training or evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import math
import platform
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


EVIDENCE_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = EVIDENCE_DIR.parent
PRELIMINARY_DIR = CAMPAIGN_DIR / "preliminary"
WORKTREE = Path(__file__).resolve().parents[5]
PPO_DIR = WORKTREE / "ppo_allocation"
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from ppo_allocation.random_event.experiment import PAIRED_METRICS  # noqa: E402
from ppo_allocation.random_event.metrics import paired_difference  # noqa: E402


VARIANTS = ("PPO-MLP", "GPPO-Adaptive")
SEEDS = (1101, 2202, 3303)
SCENARIOS = (
    "Test-Single",
    "Test-Sequential",
    "Test-Overlap",
    "Test-Burst",
    "Test-Unseen",
)
EXPECTED_METRICS = (
    "event_success_rate",
    "legal_coverage_rate",
    "recovery_delay",
    "cumulative_uncovered_time",
    "normalized_distance",
    "load_gap",
    "switch_count",
    "episode_return",
    "communication_bytes",
    "inference_latency_ms",
)
METRIC_DIRECTION = {
    "event_success_rate": "higher",
    "legal_coverage_rate": "higher",
    "recovery_delay": "lower",
    "cumulative_uncovered_time": "lower",
    "normalized_distance": "lower",
    "load_gap": "lower",
    "switch_count": "lower",
    "episode_return": "higher",
    "communication_bytes": "lower",
    "inference_latency_ms": "lower",
}
METRIC_LABEL = {
    "event_success_rate": "Event success rate",
    "legal_coverage_rate": "Legal coverage rate",
    "recovery_delay": "Recovery delay",
    "cumulative_uncovered_time": "Cumulative uncovered time",
    "normalized_distance": "Normalized distance",
    "load_gap": "Load gap",
    "switch_count": "Switch count",
    "episode_return": "Episode return",
    "communication_bytes": "Communication bytes",
    "inference_latency_ms": "Inference latency (ms)",
}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260827
T95_DF2 = 4.302652729911275


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def t95_ci(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.mean(values)
    if len(values) == 1:
        return None, None
    critical = T95_DF2 if len(values) == 3 else 1.96
    half = critical * sample_sd(values) / math.sqrt(len(values))
    return mean - half, mean + half


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, n_resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    chunk = max(1, min(n_resamples, 1_000_000 // max(1, array.size)))
    for start in range(0, n_resamples, chunk):
        stop = min(n_resamples, start + chunk)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def exact_signflip_p(values: Sequence[float]) -> float | None:
    """Exact two-sided paired sign-flip p-value over independent seed effects."""
    if len(values) < 2:
        return None
    array = np.asarray(values, dtype=float)
    observed = abs(float(array.mean()))
    stats = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(array)):
        stats.append(abs(float(np.mean(array * np.asarray(signs)))))
    return sum(value >= observed - 1e-15 for value in stats) / len(stats)


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    valid = [(index, value) for index, value in enumerate(p_values) if value is not None]
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(sorted(valid, key=lambda item: item[1])):
        candidate = min(1.0, (len(valid) - rank) * float(value))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "—"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    if abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def audit_and_load(*, require_checkpoint_files: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors: list[str] = []
    metric_tuple = tuple(PAIRED_METRICS)
    if metric_tuple != EXPECTED_METRICS:
        errors.append(f"paired metric contract drift: {metric_tuple!r}")

    manifest_path = PRELIMINARY_DIR / "tapes/preliminary_test_protocol/manifest.json"
    freeze_path = PRELIMINARY_DIR / "frozen_manifests.json"
    ledger_path = PRELIMINARY_DIR / "test_ledger.json"
    lock_path = PRELIMINARY_DIR / "formal_test_bank_lock.json"
    manifest = read_json(manifest_path)
    freeze = read_json(freeze_path)
    ledger = read_json(ledger_path)
    lock = read_json(lock_path)
    manifest_sha = sha256(manifest_path)
    freeze_sha = sha256(freeze_path)

    entries = manifest.get("entries", [])
    entry_by_id = {str(item.get("tape_id")): item for item in entries}
    expected_set_counts = {scenario: 20 for scenario in SCENARIOS}
    set_counts = Counter(str(item.get("set_name")) for item in entries)
    if len(entries) != 100 or len(entry_by_id) != 100:
        errors.append("held-out manifest is not exactly 100 unique tapes")
    if dict(set_counts) != expected_set_counts:
        errors.append(f"held-out scenario counts mismatch: {dict(set_counts)}")
    if manifest.get("checkpoint_selection") is not False:
        errors.append("held-out manifest indicates checkpoint selection")
    if manifest.get("reward_tuning") is not False:
        errors.append("held-out manifest indicates reward tuning")

    freeze_rows = freeze.get("freezes", [])
    freeze_by_key = {
        (str(item.get("variant")), int(item.get("training_seed"))): item
        for item in freeze_rows
    }
    expected_keys = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if freeze.get("formal") is not True:
        errors.append("freeze is not formal")
    if freeze.get("checkpoint_selection") is not False:
        errors.append("freeze used checkpoint selection")
    if freeze.get("fixed_evaluation_checkpoint") != 50_000:
        errors.append("freeze is not fixed at 50k")
    if set(freeze_by_key) != expected_keys or len(freeze_rows) != 6:
        errors.append("freeze is not exactly the 2x3 model/seed matrix")
    training_evidence_path = CAMPAIGN_DIR / "training_evidence/training_evidence.json"
    training_evidence = read_json(training_evidence_path)
    candidate_by_key = {
        (str(item.get("variant")), int(item.get("seed"))): item
        for item in training_evidence.get("fixed_50000_evaluation_candidates", [])
    }
    if set(candidate_by_key) != expected_keys:
        errors.append("training seal does not contain exactly six fixed 50k candidates")
    for key, item in freeze_by_key.items():
        if int(item.get("selected_step", 0)) != 50_000:
            errors.append(f"non-50k freeze: {key}")
        checkpoint = PPO_DIR / str(item.get("checkpoint_path"))
        if not checkpoint.is_file():
            if require_checkpoint_files:
                errors.append(f"missing frozen checkpoint: {key}")
            elif candidate_by_key.get(key, {}).get("sha256") != item.get("checkpoint_sha256"):
                errors.append(f"archived checkpoint SHA provenance mismatch: {key}")
        elif sha256(checkpoint) != item.get("checkpoint_sha256"):
            errors.append(f"frozen checkpoint SHA mismatch: {key}")

    if ledger.get("completed") is not True or len(ledger.get("entries", {})) != 6:
        errors.append("test ledger is not complete with six entries")
    if lock.get("completed") is not True:
        errors.append("formal test lock is not complete")
    if ledger.get("test_manifest_sha256") != manifest_sha:
        errors.append("ledger held-out manifest SHA mismatch")
    if lock.get("test_manifest_sha256") != manifest_sha:
        errors.append("lock held-out manifest SHA mismatch")
    if lock.get("freeze_manifest_sha256") != freeze_sha:
        errors.append("lock freeze SHA mismatch")

    rows: list[dict[str, Any]] = []
    result_files: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        variant, seed = key
        freeze_item = freeze_by_key.get(key)
        matches = [
            item
            for item in ledger.get("entries", {}).values()
            if (str(item.get("variant")), int(item.get("training_seed", 0))) == key
        ]
        if len(matches) != 1 or freeze_item is None:
            errors.append(f"ledger entry count for {key}: {len(matches)}")
            continue
        ledger_item = matches[0]
        result_path = PPO_DIR / str(ledger_item.get("result_path"))
        if not result_path.is_file():
            errors.append(f"missing result file: {key}")
            continue
        result_sha = sha256(result_path)
        if result_sha != ledger_item.get("result_sha"):
            errors.append(f"result SHA mismatch: {key}")
        if ledger_item.get("checkpoint_sha") != freeze_item.get("checkpoint_sha256"):
            errors.append(f"result/checkpoint provenance mismatch: {key}")
        if ledger_item.get("freeze_manifest_sha") != freeze_sha:
            errors.append(f"result/freeze provenance mismatch: {key}")
        if ledger_item.get("test_manifest_sha") != manifest_sha:
            errors.append(f"result/held-out provenance mismatch: {key}")

        result = read_json(result_path)
        result_rows = result.get("results", [])
        result_ids = [str(item.get("tape_id")) for item in result_rows]
        if result.get("tape_count") != 100 or len(result_rows) != 100:
            errors.append(f"result count is not 100: {key}")
        if len(set(result_ids)) != 100 or set(result_ids) != set(entry_by_id):
            errors.append(f"held-out tape set mismatch/duplicate: {key}")
        if result.get("test_manifest_sha256") != manifest_sha:
            errors.append(f"result manifest SHA mismatch: {key}")

        for item in result_rows:
            tape_id = str(item.get("tape_id"))
            tape = entry_by_id.get(tape_id)
            episode = item.get("episode", {})
            if tape is None:
                continue
            if episode.get("tape_id") != tape_id:
                errors.append(f"episode tape ID mismatch: {key}/{tape_id}")
            if item.get("mode") != tape.get("mode"):
                errors.append(f"episode mode mismatch: {key}/{tape_id}")
            row: dict[str, Any] = {
                "variant": variant,
                "training_seed": seed,
                "checkpoint_step": 50_000,
                "checkpoint_sha256": freeze_item.get("checkpoint_sha256"),
                "tape_id": tape_id,
                "scenario": tape.get("set_name"),
                "mode": tape.get("mode"),
                "test_manifest_sha256": manifest_sha,
            }
            for metric in EXPECTED_METRICS:
                value = episode.get(metric)
                if not finite(value):
                    errors.append(f"missing/non-finite {metric}: {key}/{tape_id}")
                row[metric] = value
            rows.append(row)
        result_files.append(
            {
                "variant": variant,
                "training_seed": seed,
                "path": result_path.relative_to(WORKTREE).as_posix(),
                "bytes": result_path.stat().st_size,
                "sha256": result_sha,
                "result_count": len(result_rows),
            }
        )

    composites = {
        (row["variant"], row["training_seed"], row["tape_id"]) for row in rows
    }
    scenario_result_counts = Counter(str(row["scenario"]) for row in rows)
    if len(rows) != 600 or len(composites) != 600:
        errors.append(f"expected 600 unique model-seed-tape results, got {len(rows)}/{len(composites)}")
    if scenario_result_counts != {scenario: 120 for scenario in SCENARIOS}:
        errors.append(f"600-result scenario counts mismatch: {dict(scenario_result_counts)}")

    state_files = sorted((PRELIMINARY_DIR / "test_state").glob("*.json"))
    if len(state_files) != 6:
        errors.append(f"expected six state journals, got {len(state_files)}")
    for state_path in state_files:
        state = read_json(state_path)
        result_path = PPO_DIR / str(state.get("result_path"))
        if state.get("state") != "consumed":
            errors.append(f"non-consumed state journal: {state_path.name}")
        if not result_path.is_file() or state.get("result_sha") != sha256(result_path):
            errors.append(f"state journal result SHA mismatch: {state_path.name}")

    if errors:
        raise SystemExit("AUDIT FAIL\n" + "\n".join(f"- {error}" for error in errors))

    audit = {
        "status": "PASS",
        "evidence_head": "2afa8ec1cb481deb57645dbd30240d90d32d2233",
        "attested_source_commit_sha": freeze.get("attested_source_commit_sha"),
        "source_tree_hash": freeze.get("source_tree_hash"),
        "protocol_sha256": freeze.get("protocol_sha256"),
        "seed_manifest_sha256": freeze.get("seed_manifest_sha256"),
        "freeze_manifest_sha256": freeze_sha,
        "test_manifest_sha256": manifest_sha,
        "freeze_count": 6,
        "fixed_checkpoint_step": 50_000,
        "checkpoint_selection": False,
        "held_out_tapes": 100,
        "scenario_tape_counts": dict(set_counts),
        "model_case_results": 600,
        "scenario_result_counts": dict(scenario_result_counts),
        "result_files": result_files,
        "metrics": list(EXPECTED_METRICS),
        "test_ledger_complete": True,
        "formal_test_lock_complete": True,
        "state_journals_consumed": 6,
        "formal_test_completed_at": lock.get("completed_at"),
        "checkpoint_binary_verification_policy": (
            "verify local binary SHA when present; in archive mode, require matching SHA provenance "
            "across the training seal and formal Freeze without loading checkpoint performance"
        ),
    }
    return rows, audit


def subset(
    rows: Sequence[dict[str, Any]],
    *,
    variant: str | None = None,
    seed: int | None = None,
    scenario: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (variant is None or row["variant"] == variant)
        and (seed is None or row["training_seed"] == seed)
        and (scenario is None or row["scenario"] == scenario)
    ]


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    groups: list[tuple[str, str, int | None, str | None]] = []
    for variant in VARIANTS:
        groups.append(("model", variant, None, None))
        groups.extend(("model_seed", variant, seed, None) for seed in SEEDS)
        groups.extend(("model_scenario", variant, None, scenario) for scenario in SCENARIOS)
        groups.extend(
            ("model_seed_scenario", variant, seed, scenario)
            for seed in SEEDS
            for scenario in SCENARIOS
        )
    for scope, variant, seed, scenario in groups:
        group = subset(rows, variant=variant, seed=seed, scenario=scenario)
        for metric_index, metric in enumerate(EXPECTED_METRICS):
            values = [float(row[metric]) for row in group]
            if seed is None:
                unit_values = [
                    statistics.mean(float(row[metric]) for row in subset(group, seed=member))
                    for member in SEEDS
                ]
                ci_lower, ci_upper = t95_ci(unit_values)
                ci_method = "t95_across_3_training_seed_means"
                independent_unit = "training_seed"
            else:
                unit_values = values
                ci_lower, ci_upper = bootstrap_mean_ci(
                    values,
                    seed=BOOTSTRAP_SEED + metric_index + seed + len(group),
                )
                ci_method = "case_bootstrap_95_not_seed_stability"
                independent_unit = "held_out_tape"
            result.append(
                {
                    "scope": scope,
                    "variant": variant,
                    "training_seed": seed,
                    "scenario": scenario,
                    "metric": metric,
                    "direction": METRIC_DIRECTION[metric],
                    "n_results": len(values),
                    "mean": statistics.mean(values),
                    "standard_deviation": sample_sd(values),
                    "independent_unit": independent_unit,
                    "unit_count": len(unit_values),
                    "ci95_lower": ci_lower,
                    "ci95_upper": ci_upper,
                    "ci_method": ci_method,
                }
            )
    return result


def paired_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    groups: list[tuple[str, int | None, str | None]] = [("overall", None, None)]
    groups.extend(("scenario", None, scenario) for scenario in SCENARIOS)
    groups.extend(("seed", seed, None) for seed in SEEDS)
    groups.extend(
        ("seed_scenario", seed, scenario) for seed in SEEDS for scenario in SCENARIOS
    )
    for group_index, (scope, seed, scenario) in enumerate(groups):
        group = subset(rows, seed=seed, scenario=scenario)
        indexed: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in group:
            indexed[(int(row["training_seed"]), str(row["tape_id"]))][str(row["variant"])] = row
        paired_keys = sorted(key for key, value in indexed.items() if set(value) == set(VARIANTS))
        family: list[dict[str, Any]] = []
        for metric_index, metric in enumerate(EXPECTED_METRICS):
            gppo = [float(indexed[key]["GPPO-Adaptive"][metric]) for key in paired_keys]
            ppo = [float(indexed[key]["PPO-MLP"][metric]) for key in paired_keys]
            effect = paired_difference(
                gppo,
                ppo,
                n_resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED + group_index * 100 + metric_index,
            )
            seed_effects = []
            for member in sorted({key[0] for key in paired_keys}):
                positions = [index for index, key in enumerate(paired_keys) if key[0] == member]
                seed_effects.append(statistics.mean(gppo[index] - ppo[index] for index in positions))
            seed_ci_lower, seed_ci_upper = t95_ci(seed_effects)
            direction_sign = 1.0 if METRIC_DIRECTION[metric] == "higher" else -1.0
            raw_delta = float(effect["mean_difference"])
            oriented_delta = direction_sign * raw_delta
            if seed_ci_lower is None or seed_ci_upper is None:
                favored = "descriptive_only"
                oriented_seed_lower = oriented_seed_upper = None
            else:
                oriented_seed_lower = min(direction_sign * seed_ci_lower, direction_sign * seed_ci_upper)
                oriented_seed_upper = max(direction_sign * seed_ci_lower, direction_sign * seed_ci_upper)
                if oriented_seed_lower > 0:
                    favored = "GPPO-Adaptive"
                elif oriented_seed_upper < 0:
                    favored = "PPO-MLP"
                elif oriented_delta == 0 and all(value == 0 for value in seed_effects):
                    favored = "tie"
                else:
                    favored = "uncertain"
            oriented_seed_effects = [direction_sign * value for value in seed_effects]
            if all(value == 0 for value in oriented_seed_effects):
                consistency = "all_tie"
            elif all(value > 0 for value in oriented_seed_effects):
                consistency = "all_seeds_favor_GPPO"
            elif all(value < 0 for value in oriented_seed_effects):
                consistency = "all_seeds_favor_PPO"
            else:
                consistency = "mixed_or_tied"
            if METRIC_DIRECTION[metric] == "higher":
                wins_gppo, wins_ppo = effect["wins_a"], effect["wins_b"]
            else:
                wins_gppo, wins_ppo = effect["wins_b"], effect["wins_a"]
            record = {
                "scope": scope,
                "training_seed": seed,
                "scenario": scenario,
                "metric": metric,
                "direction": METRIC_DIRECTION[metric],
                "n_paired_results": effect["n"],
                "mean_gppo": effect["mean_a"],
                "mean_ppo": effect["mean_b"],
                "mean_difference_gppo_minus_ppo": raw_delta,
                "median_difference_gppo_minus_ppo": effect["median_difference"],
                "case_pair_bootstrap_ci95_lower": effect["bootstrap_ci"]["lower"],
                "case_pair_bootstrap_ci95_upper": effect["bootstrap_ci"]["upper"],
                "case_pair_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "case_pair_bootstrap_note": "instance uncertainty; not independent training-seed stability",
                "seed_effect_count": len(seed_effects),
                "seed_mean_differences_gppo_minus_ppo": seed_effects,
                "seed_stability_ci95_lower": seed_ci_lower,
                "seed_stability_ci95_upper": seed_ci_upper,
                "seed_stability_ci_method": "paired seed-mean t interval, df=2" if len(seed_effects) == 3 else None,
                "oriented_mean_effect_positive_favors_gppo": oriented_delta,
                "oriented_seed_ci95_lower": oriented_seed_lower,
                "oriented_seed_ci95_upper": oriented_seed_upper,
                "effect_direction_by_seed_ci": favored,
                "seed_direction_consistency": consistency,
                "case_wins_gppo": wins_gppo,
                "case_ties": effect["ties"],
                "case_wins_ppo": wins_ppo,
                "cohen_dz_case_pairs": effect["cohen_dz"],
                "rank_biserial_case_pairs": effect["rank_biserial"],
                "seed_signflip_p_raw": exact_signflip_p(seed_effects),
                "seed_signflip_p_holm_10_metrics": None,
            }
            family.append(record)
        adjusted = holm_adjust([record["seed_signflip_p_raw"] for record in family])
        for record, p_adjusted in zip(family, adjusted):
            record["seed_signflip_p_holm_10_metrics"] = p_adjusted
        result.extend(family)
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serializable)


def find_effect(
    effects: Sequence[dict[str, Any]],
    *,
    scope: str,
    metric: str,
    seed: int | None = None,
    scenario: str | None = None,
) -> dict[str, Any]:
    matches = [
        row
        for row in effects
        if row["scope"] == scope
        and row["metric"] == metric
        and row["training_seed"] == seed
        and row["scenario"] == scenario
    ]
    if len(matches) != 1:
        raise ValueError(f"effect lookup returned {len(matches)} rows")
    return matches[0]


def build_conclusions(effects: Sequence[dict[str, Any]]) -> dict[str, Any]:
    overall = {metric: find_effect(effects, scope="overall", metric=metric) for metric in EXPECTED_METRICS}
    scenario_return = {
        scenario: find_effect(effects, scope="scenario", scenario=scenario, metric="episode_return")
        for scenario in SCENARIOS
    }
    seed_return = {
        seed: find_effect(effects, scope="seed", seed=seed, metric="episode_return")
        for seed in SEEDS
    }
    robust_gppo = [metric for metric, row in overall.items() if row["effect_direction_by_seed_ci"] == "GPPO-Adaptive"]
    robust_ppo = [metric for metric, row in overall.items() if row["effect_direction_by_seed_ci"] == "PPO-MLP"]
    ties = [metric for metric, row in overall.items() if row["effect_direction_by_seed_ci"] == "tie"]
    uncertain = [metric for metric, row in overall.items() if row["effect_direction_by_seed_ci"] == "uncertain"]
    return {
        "overall": {
            "assessment": "No uniform GPPO-Adaptive superiority is supported at the fixed 50k checkpoint.",
            "robust_by_three_seed_ci_favors_gppo": robust_gppo,
            "robust_by_three_seed_ci_favors_ppo": robust_ppo,
            "exact_ties": ties,
            "uncertain": uncertain,
            "key_deltas_gppo_minus_ppo": {
                metric: overall[metric]["mean_difference_gppo_minus_ppo"]
                for metric in EXPECTED_METRICS
            },
        },
        "scenario_episode_return": {
            scenario: {
                "mean_difference_gppo_minus_ppo": row["mean_difference_gppo_minus_ppo"],
                "seed_stability_ci95": [row["seed_stability_ci95_lower"], row["seed_stability_ci95_upper"]],
                "direction": row["effect_direction_by_seed_ci"],
                "seed_consistency": row["seed_direction_consistency"],
            }
            for scenario, row in scenario_return.items()
        },
        "seed_episode_return": {
            str(seed): {
                "mean_difference_gppo_minus_ppo": row["mean_difference_gppo_minus_ppo"],
                "case_pair_bootstrap_ci95": [
                    row["case_pair_bootstrap_ci95_lower"],
                    row["case_pair_bootstrap_ci95_upper"],
                ],
            }
            for seed, row in seed_return.items()
        },
        "answers": {
            "overall_gppo_vs_ppo": (
                "Quality outcomes are mostly equal, small, or mixed across seeds. "
                "GPPO-Adaptive has materially higher inference latency and higher normalized distance; "
                "the frozen evidence does not establish overall superiority over PPO-MLP."
            ),
            "scenarios": (
                "Episode return is directionally higher for GPPO-Adaptive in Sequential, Overlap, and Burst, "
                "and lower in Single and Unseen, but every scenario-level three-seed interval crosses zero."
            ),
            "seed_consistency": (
                "Episode-return effects are mixed across seeds (1101 favors PPO-MLP, 2202 favors GPPO-Adaptive, "
                "3303 is near a tie), so the quality-effect direction is not seed-stable."
            ),
            "evidence_sufficiency": (
                "The 50k evidence is sufficient for an auditable preliminary comparison under this frozen 100-case bank, "
                "but only three independent training seeds and mixed quality effects are insufficient for a general claim "
                "that one model is better. No additional checkpoint was inspected or selected."
            ),
        },
    }


def render_report(
    audit: dict[str, Any],
    aggregates: Sequence[dict[str, Any]],
    effects: Sequence[dict[str, Any]],
    conclusions: dict[str, Any],
) -> str:
    overall_effects = [row for row in effects if row["scope"] == "overall"]
    scenario_return = [
        find_effect(effects, scope="scenario", scenario=scenario, metric="episode_return")
        for scenario in SCENARIOS
    ]
    seed_return = [find_effect(effects, scope="seed", seed=seed, metric="episode_return") for seed in SEEDS]

    def status_class(value: str) -> str:
        if value == "GPPO-Adaptive":
            return "gppo"
        if value == "PPO-MLP":
            return "ppo"
        if value == "tie":
            return "tie"
        return "uncertain"

    overall_rows = []
    for row in overall_effects:
        ci = f"[{fmt(row['seed_stability_ci95_lower'])}, {fmt(row['seed_stability_ci95_upper'])}]"
        effect = html.escape(str(row["effect_direction_by_seed_ci"]))
        overall_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(METRIC_LABEL[row['metric']])}</strong><small>{html.escape(row['metric'])}</small></td>"
            f"<td>{html.escape(row['direction'])}</td>"
            f"<td>{fmt(row['mean_gppo'])}</td><td>{fmt(row['mean_ppo'])}</td>"
            f"<td>{fmt(row['mean_difference_gppo_minus_ppo'])}</td><td>{ci}</td>"
            f"<td>{html.escape(row['seed_direction_consistency'])}</td>"
            f"<td><span class='pill {status_class(row['effect_direction_by_seed_ci'])}'>{effect}</span></td>"
            "</tr>"
        )

    scenario_rows = []
    for row in scenario_return:
        seed_values = row["seed_mean_differences_gppo_minus_ppo"]
        scenario_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(row['scenario'])}</strong></td>"
            f"<td>{fmt(seed_values[0])}</td><td>{fmt(seed_values[1])}</td><td>{fmt(seed_values[2])}</td>"
            f"<td>{fmt(row['mean_difference_gppo_minus_ppo'])}</td>"
            f"<td>[{fmt(row['seed_stability_ci95_lower'])}, {fmt(row['seed_stability_ci95_upper'])}]</td>"
            f"<td><span class='pill {status_class(row['effect_direction_by_seed_ci'])}'>{html.escape(row['effect_direction_by_seed_ci'])}</span></td>"
            "</tr>"
        )

    seed_rows = []
    for row in seed_return:
        seed_rows.append(
            "<tr>"
            f"<td><strong>{row['training_seed']}</strong></td>"
            f"<td>{fmt(row['mean_gppo'])}</td><td>{fmt(row['mean_ppo'])}</td>"
            f"<td>{fmt(row['mean_difference_gppo_minus_ppo'])}</td>"
            f"<td>[{fmt(row['case_pair_bootstrap_ci95_lower'])}, {fmt(row['case_pair_bootstrap_ci95_upper'])}]</td>"
            "</tr>"
        )

    answers = conclusions["answers"]
    cards = [
        ("总体", answers["overall_gppo_vs_ppo"]),
        ("场景", answers["scenarios"]),
        ("Seeds", answers["seed_consistency"]),
        ("证据充分性", answers["evidence_sufficiency"]),
    ]
    answer_html = "".join(
        f"<article class='answer'><h3>{html.escape(title)}</h3><p>{html.escape(text)}</p></article>"
        for title, text in cards
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Two-model minimum-validation 50k evaluation</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ef;--panel:#f8fafc;--gppo:#0f766e;--ppo:#3158a8;--warn:#9a6700;--tie:#475569}}
*{{box-sizing:border-box}} body{{margin:0;background:#eef2f7;color:var(--ink);font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.55}}
main{{max-width:1220px;margin:0 auto;padding:32px 24px 64px}} .hero{{background:linear-gradient(135deg,#102a43,#155e75);color:white;border-radius:18px;padding:30px;box-shadow:0 18px 45px #0f172a26}}
.hero h1{{margin:0 0 8px;font-size:30px}} .hero p{{margin:4px 0;color:#dbeafe}} .stamp{{font-family:Consolas,monospace;font-size:12px;word-break:break-all}}
.kpis{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:18px 0}} .kpi{{background:white;border:1px solid var(--line);border-radius:14px;padding:16px}} .kpi b{{display:block;font-size:25px;color:#0f4c5c}} .kpi span{{color:var(--muted);font-size:13px}}
section{{background:white;border:1px solid var(--line);border-radius:16px;padding:22px;margin-top:16px;box-shadow:0 7px 20px #0f172a0b}} h2{{margin:0 0 14px;font-size:20px}} h3{{margin:0 0 7px;font-size:15px}} p{{margin:5px 0}}
.answers{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}} .answer{{padding:16px;border-radius:12px;background:var(--panel);border-left:4px solid #0f766e}}
.table-wrap{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}} th{{text-align:left;background:#edf2f7;color:#334155;position:sticky;top:0}} th,td{{padding:10px 9px;border-bottom:1px solid var(--line);vertical-align:top;white-space:nowrap}} td:first-child{{white-space:normal;min-width:185px}} small{{display:block;color:var(--muted);font-family:Consolas,monospace}}
.pill{{display:inline-block;padding:3px 8px;border-radius:999px;font-weight:650;font-size:11px}} .pill.gppo{{background:#ccfbf1;color:#115e59}} .pill.ppo{{background:#dbeafe;color:#1e3a8a}} .pill.uncertain{{background:#fef3c7;color:#854d0e}} .pill.tie{{background:#e2e8f0;color:#334155}}
.note{{padding:12px 14px;border-radius:10px;background:#fff7ed;border:1px solid #fed7aa;color:#7c2d12}} code{{font-family:Consolas,monospace;font-size:12px}} ul{{margin:8px 0 0;padding-left:20px}} .footer{{color:var(--muted);font-size:12px;margin-top:18px}}
@media(max-width:850px){{.kpis{{grid-template-columns:repeat(2,1fr)}}.answers{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<header class="hero"><h1>两模型 minimum-validation 固定 50k 对比</h1>
<p>PPO-MLP vs GPPO-Adaptive · 3 training seeds · 同一冻结 held-out bank</p>
<p class="stamp">Evidence HEAD: {html.escape(audit['evidence_head'])}<br>Test manifest SHA-256: {html.escape(audit['test_manifest_sha256'])}</p></header>
<div class="kpis">
<div class="kpi"><b>6 / 6</b><span>固定 50k checkpoints</span></div>
<div class="kpi"><b>100</b><span>冻结 held-out cases</span></div>
<div class="kpi"><b>600 / 600</b><span>完整 model-case results</span></div>
<div class="kpi"><b>5 × 20</b><span>场景平衡</span></div>
<div class="kpi"><b>0</b><span>checkpoint selection</span></div>
</div>
<section><h2>结论摘要</h2><div class="answers">{answer_html}</div></section>
<section><h2>总体：全部冻结指标</h2>
<p class="note">差值定义为 GPPO-Adaptive − PPO-MLP。Seed CI 是 3 个独立训练 seed 均值差的 95% t 区间；case bootstrap 只用于 tape 配对不确定性，不能替代 seed 稳定性。</p>
<div class="table-wrap"><table><thead><tr><th>指标</th><th>优向</th><th>GPPO</th><th>PPO</th><th>差值</th><th>3-seed 95% CI</th><th>seed 方向</th><th>判读</th></tr></thead><tbody>{''.join(overall_rows)}</tbody></table></div></section>
<section><h2>场景：Episode return 配对差</h2>
<div class="table-wrap"><table><thead><tr><th>场景</th><th>Seed 1101</th><th>Seed 2202</th><th>Seed 3303</th><th>均值差</th><th>3-seed 95% CI</th><th>判读</th></tr></thead><tbody>{''.join(scenario_rows)}</tbody></table></div></section>
<section><h2>Seed 一致性：Episode return</h2>
<div class="table-wrap"><table><thead><tr><th>Seed</th><th>GPPO</th><th>PPO</th><th>差值</th><th>case-pair bootstrap 95% CI</th></tr></thead><tbody>{''.join(seed_rows)}</tbody></table></div></section>
<section><h2>审计边界</h2><ul>
<li>Freeze 精确包含 PPO-MLP、GPPO-Adaptive × seeds 1101/2202/3303 的六个 50k checkpoint。</li>
<li>25k checkpoint 只由训练证据清单验证存在与 SHA；本报告未加载或比较其性能。</li>
<li>未执行 Validation，未做 checkpoint selection，未重训、续训、调参或修改协议。</li>
<li>统计只使用代码中冻结的 10 个 <code>PAIRED_METRICS</code>；Holm 校正应用于每个 10 指标 family 的 seed-level exact sign-flip p 值。</li>
<li>原始 600 条结果、分组统计及全部配对效应分别见同目录 CSV/JSONL/JSON。</li>
</ul></section>
<p class="footer">Generated from frozen Test completed at {html.escape(str(audit['formal_test_completed_at']))} · Analysis script SHA-256 {sha256(Path(__file__))}</p>
</main></body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-mode",
        action="store_true",
        help="permit absent checkpoint binaries and cross-check their SHA provenance from the sealed training evidence",
    )
    args = parser.parse_args(argv)
    rows, audit = audit_and_load(require_checkpoint_files=not args.archive_mode)
    aggregates = aggregate_rows(rows)
    effects = paired_rows(rows)
    conclusions = build_conclusions(effects)

    write_csv(EVIDENCE_DIR / "evaluation_rows.csv", rows)
    with (EVIDENCE_DIR / "evaluation_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_csv(EVIDENCE_DIR / "aggregate_metrics.csv", aggregates)
    write_csv(EVIDENCE_DIR / "paired_effects.csv", effects)
    write_json(EVIDENCE_DIR / "paired_effects.json", effects)
    write_json(EVIDENCE_DIR / "analysis_summary.json", conclusions)
    write_json(EVIDENCE_DIR / "result_integrity_audit.json", audit)
    reproduction = {
        "status": "PASS",
        "generated_at": audit["formal_test_completed_at"],
        "script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paired_metrics_imported_from": "ppo_allocation.random_event.experiment.PAIRED_METRICS",
        "paired_effect_implementation": "ppo_allocation.random_event.metrics.paired_difference",
        "seed_uncertainty": "paired seed-mean t interval with df=2",
        "multiplicity": "Holm correction within each 10-metric comparison family",
        "prohibited_operations": {
            "validation": False,
            "checkpoint_selection": False,
            "checkpoint_25000_performance_read": False,
            "retraining": False,
            "resume_training": False,
            "protocol_change": False,
        },
    }
    write_json(EVIDENCE_DIR / "analysis_reproduction.json", reproduction)
    report = render_report(audit, aggregates, effects, conclusions)
    (EVIDENCE_DIR / "comparison_report.html").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(rows),
                "aggregate_records": len(aggregates),
                "paired_effect_records": len(effects),
                "report": str(EVIDENCE_DIR / "comparison_report.html"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
